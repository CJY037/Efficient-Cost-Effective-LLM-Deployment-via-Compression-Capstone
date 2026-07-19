#!/usr/bin/env python3
"""Recover a pruned Llama 3.2 instruct checkpoint with LM + chat SFT training."""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import time
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from datasets import load_dataset
from torch.utils.data import IterableDataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)
from transformers.trainer_utils import get_last_checkpoint

try:
    import gpu_pin
except Exception:  # pragma: no cover - optional local helper
    gpu_pin = None


IGNORE_INDEX = -100
WIKITEXT_DATASET = "wikitext"
WIKITEXT_CONFIG = "wikitext-2-raw-v1"
EVAL_SPLIT = "validation"
DEFAULT_FINANCE_EVAL_ARTICLE = (
    "Acme Corp reported quarterly revenue of $4.8 billion, up 18% year over year. "
    "The company beat analyst earnings estimates, raised its full-year guidance, "
    "and said demand from enterprise customers is accelerating."
)
DEFAULT_SUMMARY_EVAL_ARTICLE = (
    "A regional bank launched a mobile platform that lets small businesses open accounts, "
    "send invoices, receive card payments, and apply for short-term working-capital loans. "
    "The bank said approvals use cash-flow data rather than relying only on traditional "
    "credit scores. Consumer advocates welcomed the lower fees but asked the bank to publish "
    "clear explanations of automated lending decisions and create an appeals process."
)


def normalize_device(device: str) -> str:
    if gpu_pin is None:
        return device
    gpu_pin.pin_single_gpu()
    return gpu_pin.apply_pinned_device(device)


def load_chat_template_if_needed(tokenizer, checkpoint: str) -> None:
    if getattr(tokenizer, "chat_template", None):
        return
    template_path = Path(checkpoint) / "chat_template.jinja"
    if template_path.is_file():
        tokenizer.chat_template = template_path.read_text(encoding="utf-8")


def drop_tokenizer_runtime_compat_flags(tokenizer) -> None:
    # Keep saved tokenizer metadata aligned with the Llama 3.2 Wikitext2 baseline.
    init_kwargs = getattr(tokenizer, "init_kwargs", None)
    if isinstance(init_kwargs, dict):
        init_kwargs.pop("fix_mistral_regex", None)


def configure_deterministic_generation(model, args) -> None:
    generation_config = getattr(model, "generation_config", None)
    if generation_config is None:
        return
    generation_config.repetition_penalty = args.generation_repetition_penalty
    generation_config.no_repeat_ngram_size = args.generation_no_repeat_ngram_size
    generation_config.do_sample = False
    generation_config.temperature = None
    generation_config.top_p = None


def normalize_messages(raw_messages: Any, system_prompt: Optional[str]) -> List[Dict[str, str]]:
    messages: List[Dict[str, str]] = []
    if not isinstance(raw_messages, list):
        if system_prompt:
            return [{"role": "system", "content": system_prompt}]
        return messages
    for item in raw_messages:
        if not isinstance(item, dict):
            continue
        role = item.get("role") or item.get("from")
        content = item.get("content") or item.get("value")
        if role in {"human", "user"}:
            role = "user"
        elif role in {"gpt", "assistant"}:
            role = "assistant"
        elif role != "system":
            continue
        if not isinstance(content, str):
            content = str(content)
        if content.strip():
            messages.append({"role": role, "content": content.strip()})

    # Llama's chat template expects at most one system turn, at the beginning.
    # Preserve a dataset-specific system instruction (important for constraint and
    # summarization data) instead of prepending a second generic system message.
    source_system = next((msg for msg in messages if msg["role"] == "system"), None)
    turns = [msg for msg in messages if msg["role"] != "system"]
    if source_system is not None:
        return [source_system, *turns]
    if system_prompt:
        return [{"role": "system", "content": system_prompt}, *turns]
    return turns


def encode_assistant_supervision(
    tokenizer,
    messages: Sequence[Dict[str, str]],
    max_seq_len: int,
) -> Tuple[List[int], List[int]]:
    input_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
    )
    labels = [IGNORE_INDEX] * len(input_ids)
    for idx, message in enumerate(messages):
        if message["role"] != "assistant":
            continue
        before_ids = tokenizer.apply_chat_template(
            messages[:idx],
            tokenize=True,
            add_generation_prompt=True,
        )
        after_ids = tokenizer.apply_chat_template(
            messages[: idx + 1],
            tokenize=True,
            add_generation_prompt=False,
        )
        start = min(len(before_ids), len(labels))
        end = min(len(after_ids), len(labels))
        for pos in range(start, end):
            labels[pos] = input_ids[pos]
    return input_ids[:max_seq_len], labels[:max_seq_len]


def find_messages(sample: Dict[str, Any], messages_column: str) -> List[Dict[str, str]]:
    value = sample.get(messages_column)
    if value is not None:
        return value
    for key in ("messages", "conversations", "conversation"):
        value = sample.get(key)
        if value is not None:
            return value
    return []


class StreamingChatSFTDataset(IterableDataset):
    def __init__(
        self,
        source_dataset,
        tokenizer,
        *,
        max_seq_len: int,
        messages_column: str,
        system_prompt: Optional[str],
    ) -> None:
        super().__init__()
        self.source_dataset = source_dataset
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.messages_column = messages_column
        self.system_prompt = system_prompt

    def set_epoch(self, epoch: int) -> None:
        if hasattr(self.source_dataset, "set_epoch"):
            self.source_dataset.set_epoch(epoch)

    def _assistant_labels(self, messages: Sequence[Dict[str, str]]) -> Tuple[List[int], List[int]]:
        return encode_assistant_supervision(self.tokenizer, messages, self.max_seq_len)

    def __iter__(self):
        for sample in self.source_dataset:
            messages = normalize_messages(
                find_messages(sample, self.messages_column),
                self.system_prompt,
            )
            if len(messages) < 2 or not any(msg["role"] == "assistant" for msg in messages):
                continue
            try:
                input_ids, labels = self._assistant_labels(messages)
            except Exception:
                continue
            if not input_ids or not any(label != IGNORE_INDEX for label in labels):
                continue
            yield {
                "input_ids": torch.tensor(input_ids, dtype=torch.long),
                "attention_mask": torch.ones(len(input_ids), dtype=torch.long),
                "labels": torch.tensor(labels, dtype=torch.long),
            }


def normalize_finance_label(label: Any) -> Optional[str]:
    text = str(label).strip().lower()
    if not text:
        return None
    if text in {"0", "negative", "bearish", "bad"}:
        return "Bad"
    if text in {"1", "neutral"}:
        return "Neutral"
    if text in {"2", "positive", "bullish", "good"}:
        return "Good"
    if "negative" in text:
        return "Bad"
    if "positive" in text:
        return "Good"
    if "neutral" in text:
        return "Neutral"
    return None


class StreamingFinanceSentimentDataset(IterableDataset):
    def __init__(
        self,
        source_dataset,
        tokenizer,
        *,
        max_seq_len: int,
        system_prompt: Optional[str],
        include_original_instruction: bool,
        include_good_bad_prompt: bool,
        include_eval_style_prompt: bool,
        include_neutral_prompt: bool,
        balance_labels: Optional[Sequence[str]],
        balance_buffer_size: int,
    ) -> None:
        super().__init__()
        self.source_dataset = source_dataset
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.system_prompt = system_prompt
        self.include_original_instruction = include_original_instruction
        self.include_good_bad_prompt = include_good_bad_prompt
        self.include_eval_style_prompt = include_eval_style_prompt
        self.include_neutral_prompt = include_neutral_prompt
        self.balance_labels = list(balance_labels or [])
        self.balance_buffer_size = max(1, int(balance_buffer_size))

    def set_epoch(self, epoch: int) -> None:
        if hasattr(self.source_dataset, "set_epoch"):
            self.source_dataset.set_epoch(epoch)

    def _to_training_item(
        self,
        user_content: str,
        assistant_content: str,
    ) -> Optional[Dict[str, torch.Tensor]]:
        messages: List[Dict[str, str]] = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.extend(
            [
                {"role": "user", "content": user_content.strip()},
                {"role": "assistant", "content": assistant_content.strip()},
            ]
        )
        try:
            input_ids = self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=False,
            )
            labels = [IGNORE_INDEX] * len(input_ids)
            before_ids = self.tokenizer.apply_chat_template(
                messages[:-1],
                tokenize=True,
                add_generation_prompt=True,
            )
        except Exception:
            return None

        start = min(len(before_ids), len(labels))
        for pos in range(start, len(labels)):
            labels[pos] = input_ids[pos]

        input_ids = input_ids[: self.max_seq_len]
        labels = labels[: self.max_seq_len]
        if not input_ids or not any(label != IGNORE_INDEX for label in labels):
            return None
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.ones(len(input_ids), dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }

    def _examples_from_sample(self, sample: Dict[str, Any]) -> Iterable[Tuple[str, str]]:
        text = sample.get("input") or sample.get("text") or sample.get("sentence")
        if not isinstance(text, str):
            text = str(text or "")
        text = text.strip()
        if not text:
            return

        output = sample.get("output", sample.get("label"))
        sentiment = normalize_finance_label(output)
        raw_output = str(output).strip()
        instruction = sample.get("instruction")
        if not isinstance(instruction, str) or not instruction.strip():
            instruction = "What is the sentiment of this financial text?"

        if self.include_original_instruction and raw_output:
            yield f"{instruction.strip()}\n\nText:\n{text}", raw_output

        if sentiment and self.include_neutral_prompt:
            yield (
                "Read the financial news or market text below.\n\n"
                f"Text:\n{text}\n\n"
                "Classify its likely investor sentiment. Answer with exactly one word: Good, Bad, or Neutral.",
                sentiment,
            )

        if sentiment in {"Good", "Bad"} and self.include_good_bad_prompt:
            yield (
                "Read the financial news or market text below.\n\n"
                f"Text:\n{text}\n\n"
                "Question: Is this good or bad for investors?\n"
                "Answer with one word only: Good or Bad.",
                sentiment,
            )

        if sentiment in {"Good", "Bad"} and self.include_eval_style_prompt:
            yield (
                "Read the financial news article below.\n\n"
                f"Article:\n{text}\n\n"
                "Classify the likely investor sentiment. Answer with exactly one word: Good or Bad.",
                sentiment,
            )

    def __iter__(self):
        if self.balance_labels:
            queues: Dict[str, deque] = defaultdict(deque)
            label_set = set(self.balance_labels)
            for sample in self.source_dataset:
                for user_content, assistant_content in self._examples_from_sample(sample):
                    label = normalize_finance_label(assistant_content)
                    if label not in label_set:
                        continue
                    item = self._to_training_item(user_content, assistant_content)
                    if item is None:
                        continue
                    queue = queues[label]
                    if len(queue) >= self.balance_buffer_size:
                        queue.popleft()
                    queue.append(item)
                while all(queues[label] for label in self.balance_labels):
                    for label in self.balance_labels:
                        yield queues[label].popleft()
            return

        for sample in self.source_dataset:
            for user_content, assistant_content in self._examples_from_sample(sample):
                item = self._to_training_item(user_content, assistant_content)
                if item is not None:
                    yield item


class StreamingTextLMDataset(IterableDataset):
    def __init__(self, source_dataset, tokenizer, *, max_seq_len: int, text_column: str) -> None:
        super().__init__()
        self.source_dataset = source_dataset
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.text_column = text_column

    def set_epoch(self, epoch: int) -> None:
        if hasattr(self.source_dataset, "set_epoch"):
            self.source_dataset.set_epoch(epoch)

    def __iter__(self):
        token_buffer: List[int] = []
        for sample in self.source_dataset:
            text = sample.get(self.text_column, "")
            if not isinstance(text, str):
                text = str(text)
            if not text.strip():
                continue
            ids = self.tokenizer(text, add_special_tokens=False)["input_ids"]
            if not ids:
                continue
            token_buffer.extend(ids)
            if self.tokenizer.eos_token_id is not None:
                token_buffer.append(self.tokenizer.eos_token_id)
            while len(token_buffer) >= self.max_seq_len:
                block = token_buffer[: self.max_seq_len]
                token_buffer = token_buffer[self.max_seq_len :]
                tensor = torch.tensor(block, dtype=torch.long)
                yield {
                    "input_ids": tensor,
                    "attention_mask": torch.ones_like(tensor),
                    "labels": tensor.clone(),
                }


class StreamingSummarizationDataset(IterableDataset):
    PROMPTS = (
        (
            "Summarize the following article accurately and concisely. Preserve the main "
            "facts, names, numbers, and conclusions. Do not add unsupported information."
        ),
        (
            "Write a concise standalone summary of the article below. Cover the central "
            "points and important evidence, and omit repetition and minor details."
        ),
        (
            "Explain the key points of this article in a clear short summary suitable for "
            "a reader who has not seen the original."
        ),
        (
            "Summarize this article in 3-5 concise bullet points. Keep factual details and "
            "do not speculate beyond the source."
        ),
    )

    def __init__(
        self,
        source_dataset,
        tokenizer,
        *,
        max_seq_len: int,
        article_column: str,
        summary_column: str,
        system_prompt: Optional[str],
        output_reserve_tokens: int,
        seed: int,
    ) -> None:
        super().__init__()
        self.source_dataset = source_dataset
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.article_column = article_column
        self.summary_column = summary_column
        self.system_prompt = system_prompt
        self.output_reserve_tokens = output_reserve_tokens
        self.seed = seed

    def set_epoch(self, epoch: int) -> None:
        if hasattr(self.source_dataset, "set_epoch"):
            self.source_dataset.set_epoch(epoch)

    def _build_item(self, article: str, summary: str, index: int) -> Optional[Dict[str, torch.Tensor]]:
        summary_ids = self.tokenizer(summary, add_special_tokens=False)["input_ids"]
        if not summary_ids:
            return None
        summary_budget = min(
            max(32, self.output_reserve_tokens),
            max(32, self.max_seq_len // 2),
        )
        summary_ids = summary_ids[:summary_budget]
        summary = self.tokenizer.decode(summary_ids, skip_special_tokens=True).strip()

        # Reserve room for the gold summary and chat-template overhead. This avoids
        # silently truncating every assistant token on long articles.
        article_ids = self.tokenizer(article, add_special_tokens=False)["input_ids"]
        article_budget = max(64, self.max_seq_len - len(summary_ids) - 192)
        article_ids = article_ids[:article_budget]
        prompt = self.PROMPTS[(index + self.seed) % len(self.PROMPTS)]

        while article_ids:
            article_text = self.tokenizer.decode(article_ids, skip_special_tokens=True).strip()
            raw_messages: List[Dict[str, str]] = [
                {
                    "role": "user",
                    "content": f"{prompt}\n\nArticle:\n{article_text}",
                },
                {"role": "assistant", "content": summary},
            ]
            messages = normalize_messages(raw_messages, self.system_prompt)
            try:
                input_ids, labels = encode_assistant_supervision(
                    self.tokenizer,
                    messages,
                    self.max_seq_len,
                )
            except Exception:
                return None
            if input_ids and any(label != IGNORE_INDEX for label in labels):
                return {
                    "input_ids": torch.tensor(input_ids, dtype=torch.long),
                    "attention_mask": torch.ones(len(input_ids), dtype=torch.long),
                    "labels": torch.tensor(labels, dtype=torch.long),
                }
            article_ids = article_ids[: max(0, len(article_ids) - 128)]
        return None

    def __iter__(self):
        for index, sample in enumerate(self.source_dataset):
            article = sample.get(self.article_column, "")
            summary = sample.get(self.summary_column, "")
            if not isinstance(article, str):
                article = str(article)
            if not isinstance(summary, str):
                summary = str(summary)
            article = article.strip()
            summary = summary.strip()
            if not article or not summary:
                continue
            item = self._build_item(article, summary, index)
            if item is not None:
                yield item


class StreamingMultipleChoiceQADataset(IterableDataset):
    def __init__(
        self,
        source_dataset,
        tokenizer,
        *,
        max_seq_len: int,
        system_prompt: Optional[str],
    ) -> None:
        super().__init__()
        self.source_dataset = source_dataset
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.system_prompt = system_prompt

    def set_epoch(self, epoch: int) -> None:
        if hasattr(self.source_dataset, "set_epoch"):
            self.source_dataset.set_epoch(epoch)

    def _answer_letter(self, answer: Any, choice_count: int) -> Optional[str]:
        letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        if isinstance(answer, int):
            index = answer
        else:
            text = str(answer).strip()
            if text.isdigit():
                index = int(text)
            elif len(text) == 1 and text.upper() in letters[:choice_count]:
                return text.upper()
            else:
                return None
        if 0 <= index < min(choice_count, len(letters)):
            return letters[index]
        return None

    def _build_item(self, sample: Dict[str, Any]) -> Optional[Dict[str, torch.Tensor]]:
        question = str(sample.get("question", "")).strip()
        choices = sample.get("choices")
        if not question or not isinstance(choices, list) or len(choices) < 2:
            return None
        choices = [str(choice).strip() for choice in choices if str(choice).strip()]
        answer = self._answer_letter(sample.get("answer"), len(choices))
        if answer is None:
            return None

        letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        options = "\n".join(f"{letters[index]}. {choice}" for index, choice in enumerate(choices))
        prompt = (
            "Answer the following multiple-choice question. "
            f"Respond with only one letter: {', '.join(letters[:len(choices)])}.\n\n"
            f"Question:\n{question}\n\nChoices:\n{options}"
        )
        messages = normalize_messages(
            [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": answer},
            ],
            self.system_prompt,
        )
        try:
            input_ids, labels = encode_assistant_supervision(
                self.tokenizer,
                messages,
                self.max_seq_len,
            )
        except Exception:
            return None
        if not input_ids or not any(label != IGNORE_INDEX for label in labels):
            return None
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.ones(len(input_ids), dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }

    def __iter__(self):
        for sample in self.source_dataset:
            item = self._build_item(sample)
            if item is not None:
                yield item


class StreamingExactInstructionDataset(IterableDataset):
    EXAMPLES: Sequence[Tuple[str, str]] = (
        (
            "Explain fintech in exactly three bullet points.",
            "- Mobile banking lets people manage accounts from a phone.\n"
            "- Digital wallets let customers pay without cash or cards.\n"
            "- Online lending lets borrowers apply for credit through an app.",
        ),
        (
            "Describe mobile banking in exactly two sentences.",
            "Mobile banking lets customers check balances, transfer money, and pay bills from a phone. "
            "It is convenient, but users still need strong passwords and secure networks.",
        ),
        (
            "Answer with only one word: Is higher revenue generally good or bad for investors?",
            "Good",
        ),
        (
            "Write one sentence about digital payments and include the exact word SECURITY.",
            "Digital payments need SECURITY controls to protect customers and merchants.",
        ),
        (
            "List exactly four examples of fintech services. Number them 1 through 4.",
            "1. Mobile banking\n2. Digital wallets\n3. Online lending\n4. Payment processing",
        ),
        (
            "Explain blockchain without using the word cryptocurrency.",
            "Blockchain is a shared digital ledger that records transactions across many computers so records are difficult to alter.",
        ),
        (
            "Give a heading named Summary, followed by exactly two bullet points about online lending.",
            "Summary\n- Online lending lets borrowers apply for loans through digital platforms.\n"
            "- Automated checks can speed up approvals, but lenders should explain decisions clearly.",
        ),
        (
            "Respond with exactly: Good",
            "Good",
        ),
        (
            "Classify the investor sentiment as exactly one word: Good or Bad.\n\n"
            "Article: Revenue rose 18%, earnings beat expectations, and management raised guidance.",
            "Good",
        ),
        (
            "Classify the investor sentiment as exactly one word: Good or Bad.\n\n"
            "Article: Shares fell after the company missed earnings estimates and cut its outlook.",
            "Bad",
        ),
        (
            "Give exactly two bullet points about payment fraud prevention.",
            "- Monitor unusual transaction patterns.\n- Use multi-factor authentication for sensitive actions.",
        ),
        (
            "Answer with only the requested label.\n\nIs strong customer growth good or bad for investors?",
            "Good",
        ),
    )

    def __init__(self, tokenizer, *, max_seq_len: int, system_prompt: Optional[str], seed: int) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.system_prompt = system_prompt
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        examples = list(self.EXAMPLES)
        while True:
            rng.shuffle(examples)
            for prompt, response in examples:
                messages = normalize_messages(
                    [
                        {"role": "user", "content": prompt},
                        {"role": "assistant", "content": response},
                    ],
                    self.system_prompt,
                )
                try:
                    input_ids, labels = encode_assistant_supervision(self.tokenizer, messages, self.max_seq_len)
                except Exception:
                    continue
                if not input_ids or not any(label != IGNORE_INDEX for label in labels):
                    continue
                yield {
                    "input_ids": torch.tensor(input_ids, dtype=torch.long),
                    "attention_mask": torch.ones(len(input_ids), dtype=torch.long),
                    "labels": torch.tensor(labels, dtype=torch.long),
                }


class WeightedMixtureDataset(IterableDataset):
    def __init__(
        self,
        datasets: Sequence[Tuple[str, IterableDataset, float]],
        *,
        seed: int,
    ) -> None:
        super().__init__()
        self.datasets = [(name, dataset, float(weight)) for name, dataset, weight in datasets if weight > 0]
        if not self.datasets:
            raise ValueError("At least one positive-weight dataset is required.")
        self.seed = seed

    def set_epoch(self, epoch: int) -> None:
        for _, dataset, _ in self.datasets:
            if hasattr(dataset, "set_epoch"):
                dataset.set_epoch(epoch)

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        worker_id = 0 if worker_info is None else worker_info.id
        rng = random.Random(self.seed + worker_id)
        names = [item[0] for item in self.datasets]
        sources = [item[1] for item in self.datasets]
        weights = [item[2] for item in self.datasets]
        iterators = [iter(source) for source in sources]
        cycles = [0] * len(sources)

        while True:
            source_index = rng.choices(range(len(sources)), weights=weights, k=1)[0]
            try:
                yield next(iterators[source_index])
            except StopIteration:
                cycles[source_index] += 1
                source = sources[source_index]
                if hasattr(source, "set_epoch"):
                    source.set_epoch(cycles[source_index])
                iterators[source_index] = iter(source)
                try:
                    yield next(iterators[source_index])
                except StopIteration as exc:
                    raise RuntimeError(
                        f"Mixture source {names[source_index]!r} produced no usable examples."
                    ) from exc


@dataclass
class SFTDataCollator:
    pad_token_id: int

    def __call__(self, features: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        max_len = max(int(item["input_ids"].shape[0]) for item in features)
        input_ids = []
        attention_mask = []
        labels = []
        for item in features:
            seq_len = int(item["input_ids"].shape[0])
            pad_len = max_len - seq_len
            input_ids.append(torch.nn.functional.pad(item["input_ids"], (0, pad_len), value=self.pad_token_id))
            attention_mask.append(torch.nn.functional.pad(item["attention_mask"], (0, pad_len), value=0))
            labels.append(torch.nn.functional.pad(item["labels"], (0, pad_len), value=IGNORE_INDEX))
        return {
            "input_ids": torch.stack(input_ids),
            "attention_mask": torch.stack(attention_mask),
            "labels": torch.stack(labels),
        }


def load_streaming_dataset(
    dataset_name: str,
    dataset_config: Optional[str],
    split: str,
    *,
    seed: int,
    shuffle_buffer: int,
    max_samples: Optional[int],
    skip_samples: int = 0,
):
    kwargs = {"split": split, "streaming": True}
    if dataset_config:
        dataset = load_dataset(dataset_name, dataset_config, **kwargs)
    else:
        dataset = load_dataset(dataset_name, **kwargs)
    if skip_samples > 0:
        dataset = dataset.skip(skip_samples)
    if shuffle_buffer > 0:
        dataset = dataset.shuffle(seed=seed, buffer_size=shuffle_buffer)
    if max_samples is not None and max_samples > 0:
        dataset = dataset.take(max_samples)
    return dataset


def maybe_apply_lora(model, args):
    if args.training_mode != "lora":
        return model
    from peft import LoraConfig, get_peft_model

    config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )
    model = get_peft_model(model, config)
    model.print_trainable_parameters()
    return model


def get_model_input_device(model) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        for _, buffer in model.named_buffers():
            return buffer.device
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_target_ids(input_ids: torch.Tensor, trg_len: int) -> torch.Tensor:
    target_ids = input_ids.clone()
    target_ids[:, :-trg_len] = IGNORE_INDEX
    return target_ids


def count_loss_tokens(target_ids: torch.Tensor) -> int:
    # Keep this identical to llama3_8b_wikitext2.py so recovery-time and
    # standalone perplexity numbers are directly comparable.
    valid_labels = int((target_ids != IGNORE_INDEX).sum().item())
    batch_size = int(target_ids.size(0))
    return max(valid_labels - batch_size, 0)


def get_eval_text_field(dataset, requested_text_column: str) -> str:
    if requested_text_column and requested_text_column in dataset.column_names:
        return requested_text_column
    for key in ["text", "sentence", "content"]:
        if key in dataset.column_names:
            return key
    raise KeyError(f"Could not find a text field in columns: {dataset.column_names}")


def build_eval_corpus_text(dataset, text_field: str) -> str:
    # Match llama3_8b_wikitext2.py exactly: preserve empty rows from the
    # columnar dataset and join the full validation split once.
    texts = ["" if value is None else str(value) for value in dataset[text_field]]
    if len(texts) == 0:
        raise ValueError("Validation corpus is empty.")
    return "\n\n".join(texts)


def collect_eval_token_ids(
    dataset,
    tokenizer,
    *,
    text_column: str,
    max_tokens: Optional[int],
) -> Tuple[torch.Tensor, int, int, int, str]:
    text_field = get_eval_text_field(dataset, text_column)
    corpus_text = build_eval_corpus_text(dataset, text_field)
    total_bytes = len(corpus_text.encode("utf-8"))
    document_count = int(len(dataset))
    input_ids = tokenizer(
        corpus_text,
        return_tensors="pt",
        add_special_tokens=False,
    )["input_ids"]
    if max_tokens is not None and max_tokens > 0:
        input_ids = input_ids[:, :max_tokens]

    if input_ids.size(1) < 2:
        raise ValueError("Not enough Wikitext validation tokens were prepared for perplexity evaluation.")
    return input_ids, document_count, total_bytes, len(corpus_text), text_field


def make_eval_windows(input_ids: torch.Tensor, max_length: int, stride: int) -> List[Dict[str, Any]]:
    seq_len = int(input_ids.size(1))
    windows: List[Dict[str, Any]] = []
    prev_end = 0

    for begin_loc in range(0, seq_len, stride):
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end
        windows.append(
            {
                "begin_loc": int(begin_loc),
                "end_loc": int(end_loc),
                "trg_len": int(trg_len),
                "input_ids": input_ids[:, begin_loc:end_loc],
            }
        )
        prev_end = end_loc
        if end_loc == seq_len:
            break

    return windows


@dataclass
class PerplexityEvalData:
    input_ids: torch.Tensor
    document_count: int
    total_bytes: int
    corpus_characters: int
    text_field: str
    max_length: int
    windows: List[Dict[str, Any]]


def prepare_perplexity_eval_data(model, tokenizer, args) -> PerplexityEvalData:
    print(
        f"Preparing perplexity eval on {args.eval_dataset_name}/"
        f"{args.eval_dataset_config} split={args.eval_split}..."
    )
    dataset = load_dataset(args.eval_dataset_name, args.eval_dataset_config, split=args.eval_split)
    input_ids, document_count, total_bytes, corpus_characters, text_field = collect_eval_token_ids(
        dataset,
        tokenizer,
        text_column=args.eval_text_column,
        max_tokens=args.eval_max_tokens,
    )
    context_limit = getattr(model.config, "max_position_embeddings", args.eval_max_length)
    effective_max_length = min(args.eval_max_length, int(context_limit))
    windows = make_eval_windows(input_ids, effective_max_length, args.eval_stride)
    print(f"Prepared {len(windows)} rolling perplexity windows.")
    return PerplexityEvalData(
        input_ids=input_ids,
        document_count=document_count,
        total_bytes=total_bytes,
        corpus_characters=corpus_characters,
        text_field=text_field,
        max_length=effective_max_length,
        windows=windows,
    )


def evaluate_prepared_perplexity(model, args, eval_data: PerplexityEvalData) -> Dict[str, Any]:
    print(
        f"Running perplexity eval on {args.eval_dataset_name}/"
        f"{args.eval_dataset_config} split={args.eval_split}..."
    )

    was_training = model.training
    model.eval()
    device = get_model_input_device(model)
    total_nll_nats = 0.0
    eval_token_count = 0
    window_token_counts: List[int] = []

    try:
        with torch.inference_mode():
            for idx, window in enumerate(eval_data.windows, start=1):
                window_input_ids = window["input_ids"].to(device)
                target_ids = build_target_ids(window_input_ids, window["trg_len"])
                loss_tokens = count_loss_tokens(target_ids)
                if loss_tokens == 0:
                    window_token_counts.append(0)
                    continue
                outputs = model(input_ids=window_input_ids, labels=target_ids)
                neg_log_likelihood = float(outputs.loss.item()) * loss_tokens
                total_nll_nats += neg_log_likelihood
                eval_token_count += loss_tokens
                window_token_counts.append(loss_tokens)
                if idx == 1 or idx == len(eval_data.windows) or idx % 25 == 0:
                    print(f"  [Perplexity] window {idx}/{len(eval_data.windows)}")
    finally:
        if was_training:
            model.train()

    if eval_token_count == 0:
        raise ValueError("No Wikitext loss tokens were evaluated.")

    avg_nll = total_nll_nats / eval_token_count
    token_perplexity = math.exp(avg_nll)
    bits_per_byte = (total_nll_nats / math.log(2.0)) / max(eval_data.total_bytes, 1)
    byte_perplexity = 2.0 ** bits_per_byte
    result = {
        "dataset": args.eval_dataset_name,
        "dataset_config": args.eval_dataset_config,
        "split": args.eval_split,
        "text_column": eval_data.text_field,
        "documents": int(eval_data.document_count),
        "corpus_characters": int(eval_data.corpus_characters),
        "total_tokens": int(eval_data.input_ids.size(1)),
        "total_nll_nats": float(total_nll_nats),
        "evaluated_tokens": int(eval_token_count),
        "evaluated_bytes": int(eval_data.total_bytes),
        "max_length": int(eval_data.max_length),
        "stride": int(args.eval_stride),
        "window_count": int(len(eval_data.windows)),
        "avg_nll": float(avg_nll),
        "token_perplexity": float(token_perplexity),
        "bits_per_byte": float(bits_per_byte),
        "byte_perplexity": float(byte_perplexity),
        "window_token_counts": window_token_counts,
    }
    print(f"Wikitext-2 validation perplexity: {result['token_perplexity']:.4f}")
    return result


def evaluate_wikitext2_perplexity(
    model,
    tokenizer,
    args,
    eval_data: Optional[PerplexityEvalData] = None,
) -> Dict[str, Any]:
    if eval_data is None:
        eval_data = prepare_perplexity_eval_data(model, tokenizer, args)
    return evaluate_prepared_perplexity(model, args, eval_data)


class PerplexityTargetCallback(TrainerCallback):
    def __init__(
        self,
        *,
        stage_name: str,
        output_dir: Path,
        args,
        eval_data: PerplexityEvalData,
        target_perplexity: float,
        min_steps: int,
        eval_interval: int,
        plateau_patience: int = 0,
        plateau_min_delta: float = 0.0,
    ) -> None:
        self.stage_name = stage_name
        self.output_path = output_dir / f"{stage_name}_perplexity_history.json"
        self.recovery_args = args
        self.eval_data = eval_data
        self.target_perplexity = target_perplexity
        self.min_steps = max(0, min_steps)
        self.eval_interval = max(1, eval_interval)
        self.plateau_patience = max(0, plateau_patience)
        self.plateau_min_delta = max(0.0, plateau_min_delta)
        self.best_perplexity = float("inf")
        self.bad_checks = 0
        self.history: List[Dict[str, Any]] = []

    def on_step_end(self, args, state, control, **kwargs):
        step = int(state.global_step)
        if step < self.min_steps:
            return control
        if step % self.eval_interval != 0 and step != int(state.max_steps):
            return control
        model = kwargs.get("model")
        if model is None:
            return control

        result = evaluate_prepared_perplexity(model, self.recovery_args, self.eval_data)
        optimizer = kwargs.get("optimizer")
        current_lr = None
        if optimizer is not None and optimizer.param_groups:
            current_lr = float(optimizer.param_groups[0]["lr"])

        perplexity = float(result["token_perplexity"])
        improved = perplexity < self.best_perplexity - self.plateau_min_delta
        if improved:
            self.best_perplexity = perplexity
            self.bad_checks = 0
        else:
            self.bad_checks += 1

        record = {
            "stage": self.stage_name,
            "step": step,
            "target_perplexity": self.target_perplexity,
            "learning_rate": current_lr,
            "best_perplexity": self.best_perplexity,
            "plateau_bad_checks": self.bad_checks,
            **result,
        }
        self.history.append(record)
        with self.output_path.open("w", encoding="utf-8") as handle:
            json.dump(self.history, handle, indent=2)

        if perplexity <= self.target_perplexity:
            print(
                f"{self.stage_name}: target perplexity {self.target_perplexity:.4f} "
                f"reached at step {step}."
            )
            control.should_training_stop = True
        elif self.plateau_patience > 0 and self.bad_checks >= self.plateau_patience:
            print(
                f"{self.stage_name}: stopping after {self.bad_checks} checks without "
                f"an improvement of at least {self.plateau_min_delta:.4f} perplexity."
            )
            control.should_training_stop = True
        return control


def parse_finance_eval_labels(raw_labels: str) -> List[str]:
    labels = [item.strip() for item in raw_labels.split(",") if item.strip()]
    normalized: List[str] = []
    for label in labels:
        canonical = normalize_finance_label(label) or label.strip()
        if canonical not in normalized:
            normalized.append(canonical)
    if len(normalized) < 2:
        raise ValueError("--finance-eval-labels must contain at least two labels.")
    return normalized


def format_label_choices(labels: Sequence[str]) -> str:
    if len(labels) == 2:
        return f"{labels[0]} or {labels[1]}"
    return ", ".join(labels[:-1]) + f", or {labels[-1]}"


def build_finance_eval_messages(
    article: str,
    system_prompt: Optional[str],
    labels: Sequence[str],
) -> List[Dict[str, str]]:
    messages: List[Dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    label_choices = format_label_choices(labels)
    messages.append(
        {
            "role": "user",
            "content": (
                "Read the financial news article below.\n\n"
                f"Article:\n{article.strip()}\n\n"
                "Classify the likely investor sentiment. Answer with exactly one word: "
                f"{label_choices}."
            ),
        }
    )
    return messages


def encode_chat_prompt(tokenizer, messages: Sequence[Dict[str, str]]) -> torch.Tensor:
    encoded = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    )
    if isinstance(encoded, list):
        encoded = torch.tensor([encoded], dtype=torch.long)
    return encoded


def score_finance_labels(model, tokenizer, prompt_ids: torch.Tensor, labels: Sequence[str]) -> Dict[str, float]:
    device = get_model_input_device(model)
    prompt_ids = prompt_ids.to(device)
    scores: Dict[str, float] = {}

    with torch.inference_mode():
        for label in labels:
            label_ids = tokenizer(label, add_special_tokens=False, return_tensors="pt")["input_ids"].to(device)
            full_ids = torch.cat([prompt_ids, label_ids], dim=1)
            target_ids = full_ids.clone()
            target_ids[:, : prompt_ids.size(1)] = IGNORE_INDEX
            # This is completion scoring, not rolling perplexity. Count valid
            # shifted completion labels directly so single-token labels work.
            loss_tokens = int((target_ids[:, 1:] != IGNORE_INDEX).sum().item())
            if loss_tokens == 0:
                scores[label] = float("-inf")
                continue
            outputs = model(input_ids=full_ids, labels=target_ids)
            scores[label] = -float(outputs.loss.item())

    return scores


def evaluate_finance_sentiment(model, tokenizer, args) -> Dict[str, Any]:
    print("Running finance sentiment smoke test...")
    model.eval()
    article = args.finance_eval_article.strip()
    labels = parse_finance_eval_labels(args.finance_eval_labels)
    messages = build_finance_eval_messages(article, args.system_prompt, labels)
    prompt_ids = encode_chat_prompt(tokenizer, messages)
    device = get_model_input_device(model)
    prompt_ids = prompt_ids.to(device)
    attention_mask = torch.ones_like(prompt_ids, dtype=torch.long, device=device)

    with torch.inference_mode():
        generated_ids = model.generate(
            input_ids=prompt_ids,
            attention_mask=attention_mask,
            max_new_tokens=args.finance_eval_max_new_tokens,
            do_sample=False,
            repetition_penalty=args.generation_repetition_penalty,
            no_repeat_ngram_size=args.generation_no_repeat_ngram_size,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    generated_text = tokenizer.decode(
        generated_ids[0, prompt_ids.size(1) :],
        skip_special_tokens=True,
    ).strip()

    label_scores = score_finance_labels(model, tokenizer, prompt_ids, labels)
    scored_label = max(label_scores, key=label_scores.get) if label_scores else ""
    result = {
        "article": article,
        "label_set": labels,
        "expected_label": args.finance_eval_expected_label,
        "generated_answer": generated_text,
        "scored_label": scored_label,
        "label_scores_avg_logprob": label_scores,
    }
    print(f"Finance sentiment generated answer: {generated_text!r}")
    print(f"Finance sentiment scored label: {scored_label}")
    if args.finance_eval_expected_label:
        print(f"Finance sentiment expected label: {args.finance_eval_expected_label}")
    return result


def generate_chat_response(
    model,
    tokenizer,
    *,
    prompt: str,
    system_prompt: Optional[str],
    max_new_tokens: int,
) -> str:
    messages = normalize_messages([{"role": "user", "content": prompt}], system_prompt)
    prompt_ids = encode_chat_prompt(tokenizer, messages)
    device = get_model_input_device(model)
    prompt_ids = prompt_ids.to(device)
    attention_mask = torch.ones_like(prompt_ids, dtype=torch.long, device=device)
    with torch.inference_mode():
        generated_ids = model.generate(
            input_ids=prompt_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            repetition_penalty=getattr(model.generation_config, "repetition_penalty", 1.0),
            no_repeat_ngram_size=getattr(model.generation_config, "no_repeat_ngram_size", 0),
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(
        generated_ids[0, prompt_ids.size(1) :],
        skip_special_tokens=True,
    ).strip()


def evaluate_instruction_and_summary(model, tokenizer, args) -> Dict[str, Any]:
    print("Running instruction-following and summarization smoke tests...")
    model.eval()
    instruction_prompts = [
        "Explain what fintech is in simple terms and give two concrete examples.",
        (
            "Follow these instructions exactly: explain the difference between a traditional "
            "bank and a fintech company in exactly three bullet points."
        ),
    ]
    instruction_results = [
        {
            "prompt": prompt,
            "response": generate_chat_response(
                model,
                tokenizer,
                prompt=prompt,
                system_prompt=args.system_prompt,
                max_new_tokens=args.instruction_eval_max_new_tokens,
            ),
        }
        for prompt in instruction_prompts
    ]
    summary_prompt = (
        "Summarize the following article in 2-3 concise sentences. Preserve the main facts "
        "and do not add unsupported information.\n\n"
        f"Article:\n{args.summary_eval_article.strip()}"
    )
    summary_result = {
        "article": args.summary_eval_article.strip(),
        "response": generate_chat_response(
            model,
            tokenizer,
            prompt=summary_prompt,
            system_prompt=args.system_prompt,
            max_new_tokens=args.summary_eval_max_new_tokens,
        ),
    }
    for item in instruction_results:
        print(f"Instruction smoke response: {item['response']!r}")
    print(f"Summarization smoke response: {summary_result['response']!r}")
    return {
        "instruction_following": instruction_results,
        "summarization": summary_result,
    }


def safe_divide(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def classification_metrics(expected: Sequence[str], predicted: Sequence[str]) -> Dict[str, Any]:
    labels = sorted(set(expected) | set(predicted))
    per_label: Dict[str, Any] = {}
    f1_values: List[float] = []
    correct = 0
    for gold, pred in zip(expected, predicted):
        correct += int(gold == pred)
    for label in labels:
        tp = sum(gold == label and pred == label for gold, pred in zip(expected, predicted))
        fp = sum(gold != label and pred == label for gold, pred in zip(expected, predicted))
        fn = sum(gold == label and pred != label for gold, pred in zip(expected, predicted))
        precision = safe_divide(tp, tp + fp)
        recall = safe_divide(tp, tp + fn)
        f1 = safe_divide(2 * precision * recall, precision + recall)
        f1_values.append(f1)
        per_label[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": sum(gold == label for gold in expected),
        }
    return {
        "accuracy": safe_divide(correct, len(expected)),
        "macro_f1": safe_divide(sum(f1_values), len(f1_values)),
        "per_label": per_label,
    }


def evaluate_finance_holdout(model, tokenizer, args) -> Dict[str, Any]:
    print("Running held-out finance sentiment benchmark...")
    dataset = load_dataset(
        args.finance_dataset_name,
        args.finance_dataset_config,
        split=args.finance_split,
    ) if args.finance_dataset_config else load_dataset(
        args.finance_dataset_name,
        split=args.finance_split,
    )
    limit = min(args.finance_benchmark_samples, args.finance_eval_holdout_samples, len(dataset))
    expected: List[str] = []
    predicted: List[str] = []
    examples: List[Dict[str, Any]] = []
    labels = parse_finance_eval_labels(args.finance_eval_labels)
    label_set = set(labels)
    skipped_outside_label_set = 0

    for sample in dataset.select(range(limit)):
        gold = normalize_finance_label(sample.get("output", sample.get("label")))
        text = sample.get("input") or sample.get("text") or sample.get("sentence")
        if gold is None or not isinstance(text, str) or not text.strip():
            continue
        if gold not in label_set:
            skipped_outside_label_set += 1
            continue
        messages = build_finance_eval_messages(text, args.system_prompt, labels)
        prompt_ids = encode_chat_prompt(tokenizer, messages)
        scores = score_finance_labels(model, tokenizer, prompt_ids, labels)
        pred = max(scores, key=scores.get)
        expected.append(gold)
        predicted.append(pred)
        if len(examples) < args.benchmark_example_limit:
            examples.append(
                {
                    "text_excerpt": text[:500],
                    "expected": gold,
                    "predicted": pred,
                    "scores": scores,
                }
            )

    result = {
        "dataset": args.finance_dataset_name,
        "split": args.finance_split,
        "label_set": labels,
        "held_out_prefix_rows": args.finance_eval_holdout_samples,
        "evaluated_examples": len(expected),
        "skipped_examples_outside_label_set": skipped_outside_label_set,
        **classification_metrics(expected, predicted),
        "prediction_counts": dict(Counter(predicted)),
        "examples": examples,
    }
    print(
        f"Finance holdout: accuracy={result['accuracy']:.4f}, "
        f"macro_f1={result['macro_f1']:.4f}, n={len(expected)}"
    )
    return result


def metric_tokens(text: str) -> List[str]:
    return re.findall(r"\b[\w'-]+\b", text.lower())


def ngram_f1(prediction: str, reference: str, n: int) -> float:
    pred_tokens = metric_tokens(prediction)
    ref_tokens = metric_tokens(reference)
    pred = Counter(tuple(pred_tokens[i : i + n]) for i in range(max(0, len(pred_tokens) - n + 1)))
    ref = Counter(tuple(ref_tokens[i : i + n]) for i in range(max(0, len(ref_tokens) - n + 1)))
    overlap = sum((pred & ref).values())
    precision = safe_divide(overlap, sum(pred.values()))
    recall = safe_divide(overlap, sum(ref.values()))
    return safe_divide(2 * precision * recall, precision + recall)


def lcs_length(left: Sequence[str], right: Sequence[str]) -> int:
    previous = [0] * (len(right) + 1)
    for left_token in left:
        current = [0]
        for index, right_token in enumerate(right, start=1):
            if left_token == right_token:
                current.append(previous[index - 1] + 1)
            else:
                current.append(max(previous[index], current[-1]))
        previous = current
    return previous[-1]


def rouge_l_f1(prediction: str, reference: str) -> float:
    pred = metric_tokens(prediction)
    ref = metric_tokens(reference)
    overlap = lcs_length(pred, ref)
    precision = safe_divide(overlap, len(pred))
    recall = safe_divide(overlap, len(ref))
    return safe_divide(2 * precision * recall, precision + recall)


def repetition_rate(text: str, n: int = 3) -> float:
    tokens = metric_tokens(text)
    ngrams = [tuple(tokens[i : i + n]) for i in range(max(0, len(tokens) - n + 1))]
    return 1.0 - safe_divide(len(set(ngrams)), len(ngrams)) if ngrams else 0.0


def evaluate_summarization_holdout(model, tokenizer, args) -> Dict[str, Any]:
    print("Running held-out summarization benchmark...")
    dataset = load_dataset(
        args.summarization_dataset_name,
        args.summarization_dataset_config,
        split=args.summarization_eval_split,
    )
    sample_count = min(args.summarization_benchmark_samples, len(dataset))
    rows = dataset.shuffle(seed=args.benchmark_seed).select(range(sample_count))
    metrics: List[Dict[str, float]] = []
    examples: List[Dict[str, Any]] = []

    for sample in rows:
        article = str(sample.get(args.summarization_article_column, "")).strip()
        reference = str(sample.get(args.summarization_summary_column, "")).strip()
        article_ids = tokenizer(article, add_special_tokens=False)["input_ids"]
        article_budget = max(128, args.max_seq_len - args.summary_eval_max_new_tokens - 192)
        article = tokenizer.decode(article_ids[:article_budget], skip_special_tokens=True).strip()
        prompt = (
            "Summarize the following article accurately in 2-4 concise sentences. Preserve "
            "the main facts, names, numbers, and conclusions. Do not add unsupported information."
            f"\n\nArticle:\n{article}"
        )
        prediction = generate_chat_response(
            model,
            tokenizer,
            prompt=prompt,
            system_prompt=args.system_prompt,
            max_new_tokens=args.summary_eval_max_new_tokens,
        )
        item_metrics = {
            "rouge1_f1": ngram_f1(prediction, reference, 1),
            "rouge2_f1": ngram_f1(prediction, reference, 2),
            "rougeL_f1": rouge_l_f1(prediction, reference),
            "repetition_3gram_rate": repetition_rate(prediction),
        }
        metrics.append(item_metrics)
        if len(examples) < args.benchmark_example_limit:
            examples.append(
                {
                    "article_excerpt": article[:700],
                    "reference": reference,
                    "prediction": prediction,
                    **item_metrics,
                }
            )

    means = {
        key: safe_divide(sum(item[key] for item in metrics), len(metrics))
        for key in ("rouge1_f1", "rouge2_f1", "rougeL_f1", "repetition_3gram_rate")
    }
    result = {
        "dataset": args.summarization_dataset_name,
        "split": args.summarization_eval_split,
        "evaluated_examples": len(metrics),
        **means,
        "examples": examples,
    }
    print(
        f"Summarization holdout: ROUGE-1={result['rouge1_f1']:.4f}, "
        f"ROUGE-L={result['rougeL_f1']:.4f}, n={len(metrics)}"
    )
    return result


def evaluate_instruction_compliance(model, tokenizer, args) -> Dict[str, Any]:
    print("Running deterministic instruction-compliance benchmark...")
    cases = [
        (
            "Explain fintech in exactly three bullet points.",
            lambda text: len(re.findall(r"(?m)^\s*(?:[-*•]|\d+[.)])\s+", text)) == 3,
        ),
        (
            "Describe mobile banking in exactly two sentences.",
            lambda text: len(re.findall(r"(?<=[.!?])(?:\s+|$)", text.strip())) == 2,
        ),
        (
            "Answer with only one word: Is higher revenue generally good or bad for investors?",
            lambda text: text.strip().lower().rstrip(".!") in {"good", "bad"},
        ),
        (
            "Write one sentence about digital payments and include the exact word SECURITY.",
            lambda text: "SECURITY" in text and len(re.findall(r"(?<=[.!?])(?:\s+|$)", text.strip())) == 1,
        ),
        (
            "List exactly four examples of fintech services. Number them 1 through 4.",
            lambda text: len(re.findall(r"(?m)^\s*[1-4][.)]\s+", text)) == 4,
        ),
        (
            "Explain blockchain without using the word cryptocurrency.",
            lambda text: "cryptocurrency" not in text.lower(),
        ),
        (
            "Give a heading named Summary, followed by exactly two bullet points about online lending.",
            lambda text: bool(re.search(r"(?im)^\s*summary\s*:?\s*$", text))
            and len(re.findall(r"(?m)^\s*[-*•]\s+", text)) == 2,
        ),
        (
            "Respond with exactly: Good",
            lambda text: text.strip() == "Good",
        ),
    ]
    results: List[Dict[str, Any]] = []
    for prompt, validator in cases:
        response = generate_chat_response(
            model,
            tokenizer,
            prompt=prompt,
            system_prompt=args.system_prompt,
            max_new_tokens=args.instruction_eval_max_new_tokens,
        )
        passed = bool(validator(response))
        results.append({"prompt": prompt, "response": response, "passed": passed})
    pass_count = sum(item["passed"] for item in results)
    result = {
        "evaluated_prompts": len(results),
        "passed": pass_count,
        "pass_rate": safe_divide(pass_count, len(results)),
        "results": results,
    }
    print(f"Instruction compliance: {pass_count}/{len(results)} passed.")
    return result


def evaluate_knowledge_mmlu(model, tokenizer, args) -> Dict[str, Any]:
    print("Running MMLU knowledge benchmark...")
    dataset = load_dataset(
        args.knowledge_benchmark_dataset,
        args.knowledge_benchmark_config,
        split=args.knowledge_benchmark_split,
        streaming=True,
    )
    dataset = dataset.shuffle(seed=args.benchmark_seed, buffer_size=10_000)
    letters = ["A", "B", "C", "D"]
    correct = 0
    total = 0
    subject_totals: Counter = Counter()
    subject_correct: Counter = Counter()
    examples: List[Dict[str, Any]] = []

    for sample in dataset.take(args.knowledge_benchmark_samples):
        choices = sample.get("choices")
        answer = sample.get("answer")
        if not isinstance(choices, list) or len(choices) != 4:
            continue
        try:
            answer_index = int(answer)
        except (TypeError, ValueError):
            continue
        if not 0 <= answer_index < 4:
            continue
        choice_text = "\n".join(f"{letter}. {text}" for letter, text in zip(letters, choices))
        prompt = (
            "Answer the following multiple-choice question. Respond with only the letter "
            "A, B, C, or D.\n\n"
            f"Question: {sample.get('question', '')}\n{choice_text}\n\nAnswer:"
        )
        messages = normalize_messages([{"role": "user", "content": prompt}], args.system_prompt)
        prompt_ids = encode_chat_prompt(tokenizer, messages)
        scores = score_finance_labels(model, tokenizer, prompt_ids, letters)
        prediction = max(scores, key=scores.get)
        expected = letters[answer_index]
        subject = str(sample.get("subject", "unknown"))
        is_correct = prediction == expected
        correct += int(is_correct)
        total += 1
        subject_totals[subject] += 1
        subject_correct[subject] += int(is_correct)
        if len(examples) < args.benchmark_example_limit:
            examples.append(
                {
                    "subject": subject,
                    "question": sample.get("question", ""),
                    "choices": choices,
                    "expected": expected,
                    "predicted": prediction,
                    "scores": scores,
                }
            )

    result = {
        "dataset": args.knowledge_benchmark_dataset,
        "config": args.knowledge_benchmark_config,
        "split": args.knowledge_benchmark_split,
        "evaluated_examples": total,
        "accuracy": safe_divide(correct, total),
        "subject_accuracy": {
            subject: safe_divide(subject_correct[subject], count)
            for subject, count in sorted(subject_totals.items())
        },
        "examples": examples,
    }
    print(f"MMLU knowledge accuracy: {result['accuracy']:.4f}, n={total}")
    return result


def run_stage(
    *,
    stage_name: str,
    model,
    tokenizer,
    train_dataset,
    output_dir: Path,
    args,
    max_steps: int,
    learning_rate: float,
    callbacks: Optional[Sequence[TrainerCallback]] = None,
    lr_scheduler_type: Optional[str] = None,
    warmup_ratio: Optional[float] = None,
    weight_decay: Optional[float] = None,
    adam_beta2: Optional[float] = None,
) -> Dict[str, Any]:
    if max_steps <= 0:
        return {"stage": stage_name, "requested_steps": max_steps, "skipped": True}
    stage_output_dir = output_dir / stage_name
    save_strategy = "steps" if args.save_steps > 0 else "no"
    training_args = TrainingArguments(
        output_dir=str(stage_output_dir),
        max_steps=max_steps,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=learning_rate,
        warmup_ratio=args.warmup_ratio if warmup_ratio is None else warmup_ratio,
        lr_scheduler_type=lr_scheduler_type or args.lr_scheduler_type,
        weight_decay=args.weight_decay if weight_decay is None else weight_decay,
        adam_beta1=args.adam_beta1,
        adam_beta2=args.adam_beta2 if adam_beta2 is None else adam_beta2,
        adam_epsilon=args.adam_epsilon,
        max_grad_norm=args.max_grad_norm,
        bf16=args.dtype == "bfloat16",
        fp16=args.dtype == "float16",
        logging_steps=args.logging_steps,
        save_strategy=save_strategy,
        save_steps=args.save_steps if args.save_steps > 0 else 500,
        save_total_limit=args.save_total_limit if args.save_total_limit > 0 else None,
        gradient_checkpointing=args.gradient_checkpointing,
        optim=args.optim,
        report_to=[],
        remove_unused_columns=False,
        dataloader_num_workers=args.dataloader_num_workers,
        dataloader_pin_memory=torch.cuda.is_available(),
        seed=args.seed,
        data_seed=args.seed,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=SFTDataCollator(tokenizer.pad_token_id),
        callbacks=list(callbacks or []),
    )
    resume_checkpoint = None
    if args.resume_from_checkpoint != "none":
        if args.resume_from_checkpoint == "auto":
            if stage_output_dir.exists():
                candidate_checkpoint = get_last_checkpoint(str(stage_output_dir))
                if is_usable_trainer_checkpoint(candidate_checkpoint):
                    resume_checkpoint = candidate_checkpoint
                elif candidate_checkpoint:
                    print(f"Ignoring incomplete checkpoint for {stage_name}: {candidate_checkpoint}")
        else:
            resume_checkpoint = args.resume_from_checkpoint
        if resume_checkpoint:
            print(f"Resuming {stage_name} from {resume_checkpoint}")
    train_result = trainer.train(resume_from_checkpoint=resume_checkpoint)
    result = {
        "stage": stage_name,
        "requested_steps": int(max_steps),
        "completed_steps": int(trainer.state.global_step),
        "learning_rate": float(learning_rate),
        "lr_scheduler_type": lr_scheduler_type or args.lr_scheduler_type,
        "weight_decay": args.weight_decay if weight_decay is None else weight_decay,
        "save_strategy": save_strategy,
        "save_steps": int(args.save_steps),
        "save_total_limit": int(args.save_total_limit),
        "resumed_from_checkpoint": resume_checkpoint,
        "train_metrics": train_result.metrics,
    }
    for callback in callbacks or []:
        if isinstance(callback, PerplexityTargetCallback):
            result["perplexity_history"] = callback.history
    del trainer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def write_summary(output_dir: Path, summary: Dict[str, Any]) -> None:
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)


def is_usable_trainer_checkpoint(checkpoint: Optional[str]) -> bool:
    if not checkpoint:
        return False
    path = Path(checkpoint)
    if not (path / "trainer_state.json").is_file():
        return False
    weight_files = list(path.glob("*.safetensors")) + list(path.glob("pytorch_model*.bin"))
    return bool(weight_files)


def save_final(model, tokenizer, output_dir: Path, args, started_at: float) -> Tuple[Path, Any, Dict[str, Any]]:
    # The output directory itself is the final Hugging Face checkpoint. Stage
    # trainers do not save model weights.
    checkpoint_dir = output_dir
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    if args.training_mode == "lora" and args.merge_lora:
        model = model.merge_and_unload()
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = True
    configure_deterministic_generation(model, args)
    model.save_pretrained(checkpoint_dir, safe_serialization=True)
    drop_tokenizer_runtime_compat_flags(tokenizer)
    tokenizer.save_pretrained(checkpoint_dir)
    summary = {
        "base_checkpoint": args.pruned_checkpoint,
        "output_checkpoint": str(checkpoint_dir),
        "training_mode": args.training_mode,
        "merged_lora": bool(args.training_mode == "lora" and args.merge_lora),
        "lm_recovery_steps": args.lm_recovery_steps,
        "lm_refinement_steps": args.lm_refinement_steps,
        "unified_task_mix_steps": args.unified_task_mix_steps,
        "chat_sft_steps": args.chat_sft_steps,
        "knowledge_sft_steps": args.knowledge_sft_steps,
        "mmlu_sft_steps": args.mmlu_sft_steps,
        "exact_instruction_sft_steps": args.exact_instruction_sft_steps,
        "constraint_sft_steps": args.constraint_sft_steps,
        "summary_instruction_sft_steps": args.summary_instruction_sft_steps,
        "summarization_sft_steps": args.summarization_sft_steps,
        "finance_sft_steps": args.finance_sft_steps,
        "stabilization_steps": args.stabilization_steps,
        "target_perplexity": args.target_perplexity,
        "lm_dataset": args.lm_dataset_name,
        "c4_dataset": args.c4_dataset_name,
        "wikipedia_lm_dataset": args.wikipedia_lm_dataset_name,
        "wikipedia_lm_dataset_config": args.wikipedia_lm_dataset_config,
        "chat_dataset": args.chat_dataset_name,
        "chat_dataset_config": args.chat_dataset_config,
        "knowledge_dataset": args.knowledge_dataset_name,
        "knowledge_dataset_config": args.knowledge_dataset_config,
        "mmlu_sft_dataset": args.mmlu_sft_dataset_name,
        "mmlu_sft_dataset_config": args.mmlu_sft_dataset_config,
        "mmlu_sft_split": args.mmlu_sft_split,
        "constraint_dataset": args.constraint_dataset_name,
        "constraint_dataset_config": args.constraint_dataset_config,
        "summary_instruction_dataset": args.summary_instruction_dataset_name,
        "summary_instruction_dataset_config": args.summary_instruction_dataset_config,
        "summarization_dataset": args.summarization_dataset_name,
        "summarization_dataset_config": args.summarization_dataset_config,
        "finance_dataset": args.finance_dataset_name,
        "finance_dataset_config": args.finance_dataset_config,
        "finance_eval_style_prompt": args.finance_eval_style_prompt,
        "max_seq_len": args.max_seq_len,
        "dtype": args.dtype,
        "runtime_seconds": time.time() - started_at,
    }
    write_summary(output_dir, summary)
    return checkpoint_dir, model, summary


def add_bool_argument(parser: argparse.ArgumentParser, name: str, default: bool) -> None:
    dest = name.replace("-", "_")
    group = parser.add_mutually_exclusive_group()
    group.add_argument(f"--{name}", dest=dest, action="store_true")
    group.add_argument(f"--no-{name}", dest=dest, action="store_false")
    parser.set_defaults(**{dest: default})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="General instruction recovery for a pruned Llama checkpoint.")
    parser.add_argument(
        "--pruned-checkpoint",
        default="./runs/llama32_etp_llm_pruner/pruned_checkpoint",
    )
    parser.add_argument(
        "--output-dir",
        default="./pruned-llama32-3b-instruct",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument("--training-mode", choices=["full", "lora"], default="full")
    add_bool_argument(parser, "merge-lora", default=True)

    parser.add_argument(
        "--lm-recovery-steps",
        type=int,
        default=20_000,
        help="Maximum LM recovery optimizer steps; target perplexity may stop this stage early.",
    )
    parser.add_argument("--lm-min-steps", type=int, default=5_000)
    parser.add_argument("--target-perplexity", type=float, default=12.0)
    parser.add_argument("--perplexity-check-steps", type=int, default=1_000)
    parser.add_argument("--perplexity-plateau-patience", type=int, default=6)
    parser.add_argument("--perplexity-plateau-min-delta", type=float, default=0.05)
    parser.add_argument("--lm-learning-rate", type=float, default=1e-5)
    parser.add_argument("--lm-lr-scheduler-type", default="constant_with_warmup")
    parser.add_argument("--lm-warmup-ratio", type=float, default=0.02)
    parser.add_argument("--lm-dataset-name", default="HuggingFaceFW/fineweb-edu")
    parser.add_argument("--lm-dataset-config", default="sample-10BT")
    parser.add_argument("--lm-split", default="train")
    parser.add_argument("--lm-text-column", default="text")
    parser.add_argument("--max-lm-samples", type=int, default=None)
    parser.add_argument("--c4-dataset-name", default="allenai/c4")
    parser.add_argument("--c4-dataset-config", default="en")
    parser.add_argument("--c4-split", default="train")
    parser.add_argument("--c4-text-column", default="text")
    parser.add_argument("--max-c4-samples", type=int, default=None)
    parser.add_argument("--wikipedia-lm-dataset-name", default="wikitext")
    parser.add_argument("--wikipedia-lm-dataset-config", default="wikitext-103-raw-v1")
    parser.add_argument("--wikipedia-lm-split", default="train")
    parser.add_argument("--wikipedia-lm-text-column", default="text")
    parser.add_argument("--max-wikipedia-lm-samples", type=int, default=None)
    parser.add_argument("--lm-fineweb-weight", type=float, default=0.65)
    parser.add_argument("--lm-c4-weight", type=float, default=0.20)
    parser.add_argument("--lm-wikipedia-weight", type=float, default=0.15)

    parser.add_argument("--lm-refinement-steps", type=int, default=20_000)
    parser.add_argument("--lm-refinement-min-steps", type=int, default=3_000)
    parser.add_argument("--lm-refinement-learning-rate", type=float, default=4e-6)
    parser.add_argument("--lm-refinement-fineweb-weight", type=float, default=0.35)
    parser.add_argument("--lm-refinement-c4-weight", type=float, default=0.15)
    parser.add_argument("--lm-refinement-wikipedia-weight", type=float, default=0.50)

    parser.add_argument("--chat-sft-steps", type=int, default=0)
    parser.add_argument("--chat-learning-rate", type=float, default=8e-6)
    parser.add_argument("--chat-dataset-name", default="HuggingFaceTB/smoltalk")
    parser.add_argument("--chat-dataset-config", default="smol-magpie-ultra")
    parser.add_argument("--chat-split", default="train")
    parser.add_argument("--messages-column", default="messages")
    parser.add_argument("--max-chat-samples", type=int, default=None)
    parser.add_argument("--system-prompt", default="You are a helpful, harmless, and honest assistant.")

    parser.add_argument("--knowledge-sft-steps", type=int, default=0)
    parser.add_argument("--knowledge-learning-rate", type=float, default=7e-6)
    parser.add_argument("--knowledge-dataset-name", default="HuggingFaceTB/smoltalk")
    parser.add_argument("--knowledge-dataset-config", default="openhermes-100k")
    parser.add_argument("--knowledge-split", default="train")
    parser.add_argument("--max-knowledge-samples", type=int, default=None)

    parser.add_argument("--mmlu-sft-steps", type=int, default=0)
    parser.add_argument("--mmlu-sft-learning-rate", type=float, default=4e-6)
    parser.add_argument("--mmlu-sft-dataset-name", default="cais/mmlu")
    parser.add_argument("--mmlu-sft-dataset-config", default="all")
    parser.add_argument("--mmlu-sft-split", default="auxiliary_train")
    parser.add_argument("--max-mmlu-sft-samples", type=int, default=None)

    parser.add_argument("--exact-instruction-sft-steps", type=int, default=0)
    parser.add_argument("--exact-instruction-learning-rate", type=float, default=2e-6)

    parser.add_argument("--constraint-sft-steps", type=int, default=0)
    parser.add_argument("--constraint-learning-rate", type=float, default=7e-6)
    parser.add_argument("--constraint-dataset-name", default="HuggingFaceTB/smoltalk")
    parser.add_argument("--constraint-dataset-config", default="smol-constraints")
    parser.add_argument("--constraint-split", default="train")
    parser.add_argument("--max-constraint-samples", type=int, default=None)

    parser.add_argument("--summary-instruction-sft-steps", type=int, default=0)
    parser.add_argument("--summary-instruction-learning-rate", type=float, default=7e-6)
    parser.add_argument("--summary-instruction-dataset-name", default="HuggingFaceTB/smoltalk")
    parser.add_argument("--summary-instruction-dataset-config", default="smol-summarize")
    parser.add_argument("--summary-instruction-split", default="train")
    parser.add_argument("--max-summary-instruction-samples", type=int, default=None)

    parser.add_argument("--summarization-sft-steps", type=int, default=0)
    parser.add_argument("--summarization-learning-rate", type=float, default=7e-6)
    parser.add_argument("--summarization-dataset-name", default="abisee/cnn_dailymail")
    parser.add_argument("--summarization-dataset-config", default="3.0.0")
    parser.add_argument("--summarization-split", default="train")
    parser.add_argument("--summarization-article-column", default="article")
    parser.add_argument("--summarization-summary-column", default="highlights")
    parser.add_argument("--summary-output-reserve-tokens", type=int, default=384)
    parser.add_argument("--max-summarization-samples", type=int, default=None)

    parser.add_argument("--finance-sft-steps", type=int, default=0)
    parser.add_argument("--finance-learning-rate", type=float, default=7e-6)
    parser.add_argument("--finance-dataset-name", default="FinGPT/fingpt-sentiment-train")
    parser.add_argument("--finance-dataset-config", default=None)
    parser.add_argument("--finance-split", default="train")
    parser.add_argument("--max-finance-samples", type=int, default=None)
    parser.add_argument("--finance-eval-holdout-samples", type=int, default=500)
    add_bool_argument(parser, "finance-original-instruction", default=True)
    add_bool_argument(parser, "finance-good-bad-prompt", default=True)
    add_bool_argument(parser, "finance-eval-style-prompt", default=False)
    add_bool_argument(parser, "finance-neutral-prompt", default=True)
    add_bool_argument(parser, "finance-balance-labels", default=False)
    parser.add_argument("--finance-balance-buffer-size", type=int, default=256)

    parser.add_argument("--unified-task-mix-steps", type=int, default=20_000)
    parser.add_argument("--unified-task-mix-min-steps", type=int, default=12_000)
    parser.add_argument("--unified-task-mix-learning-rate", type=float, default=2e-6)
    parser.add_argument("--unified-task-lm-weight", type=float, default=0.50)
    parser.add_argument("--unified-task-chat-weight", type=float, default=0.15)
    parser.add_argument("--unified-task-knowledge-weight", type=float, default=0.10)
    parser.add_argument("--unified-task-mmlu-weight", type=float, default=0.00)
    parser.add_argument("--unified-task-exact-instruction-weight", type=float, default=0.00)
    parser.add_argument("--unified-task-constraint-weight", type=float, default=0.05)
    parser.add_argument("--unified-task-summary-instruction-weight", type=float, default=0.05)
    parser.add_argument("--unified-task-summarization-weight", type=float, default=0.08)
    parser.add_argument("--unified-task-finance-weight", type=float, default=0.07)

    parser.add_argument("--stabilization-steps", type=int, default=0)
    parser.add_argument("--stabilization-min-steps", type=int, default=2_500)
    parser.add_argument("--stabilization-learning-rate", type=float, default=3e-6)
    parser.add_argument("--stabilization-lm-weight", type=float, default=0.55)
    parser.add_argument("--stabilization-chat-weight", type=float, default=0.15)
    parser.add_argument("--stabilization-knowledge-weight", type=float, default=0.10)
    parser.add_argument("--stabilization-mmlu-weight", type=float, default=0.00)
    parser.add_argument("--stabilization-exact-instruction-weight", type=float, default=0.00)
    parser.add_argument("--stabilization-constraint-weight", type=float, default=0.05)
    parser.add_argument("--stabilization-summary-instruction-weight", type=float, default=0.05)
    parser.add_argument("--stabilization-summarization-weight", type=float, default=0.07)
    parser.add_argument("--stabilization-finance-weight", type=float, default=0.03)

    add_bool_argument(parser, "eval-at-end", default=True)
    parser.add_argument("--eval-dataset-name", default=WIKITEXT_DATASET)
    parser.add_argument("--eval-dataset-config", default=WIKITEXT_CONFIG)
    parser.add_argument("--eval-split", default=EVAL_SPLIT)
    parser.add_argument("--eval-text-column", default="text")
    parser.add_argument("--eval-max-length", type=int, default=2048)
    parser.add_argument("--eval-stride", type=int, default=512)
    parser.add_argument("--eval-max-tokens", type=int, default=None)
    parser.add_argument("--finance-eval-article", default=DEFAULT_FINANCE_EVAL_ARTICLE)
    parser.add_argument("--finance-eval-expected-label", default="Good")
    parser.add_argument("--finance-eval-labels", default="Good,Bad")
    parser.add_argument("--finance-eval-max-new-tokens", type=int, default=1)
    parser.add_argument("--instruction-eval-max-new-tokens", type=int, default=192)
    parser.add_argument("--summary-eval-article", default=DEFAULT_SUMMARY_EVAL_ARTICLE)
    parser.add_argument("--summary-eval-max-new-tokens", type=int, default=160)
    parser.add_argument("--generation-repetition-penalty", type=float, default=1.08)
    parser.add_argument("--generation-no-repeat-ngram-size", type=int, default=4)
    parser.add_argument("--finance-benchmark-samples", type=int, default=200)
    parser.add_argument("--summarization-eval-split", default="validation")
    parser.add_argument("--summarization-benchmark-samples", type=int, default=25)
    parser.add_argument("--benchmark-example-limit", type=int, default=10)
    parser.add_argument("--benchmark-seed", type=int, default=1234)
    add_bool_argument(parser, "eval-mmlu", default=True)
    parser.add_argument("--knowledge-benchmark-dataset", default="cais/mmlu")
    parser.add_argument("--knowledge-benchmark-config", default="all")
    parser.add_argument("--knowledge-benchmark-split", default="test")
    parser.add_argument("--knowledge-benchmark-samples", type=int, default=100)

    parser.add_argument("--max-seq-len", type=int, default=2048)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=16)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--lr-scheduler-type", default="cosine")
    parser.add_argument("--optim", default="adamw_torch")
    parser.add_argument("--weight-decay", type=float, default=0.10)
    parser.add_argument("--adam-beta1", type=float, default=0.9)
    parser.add_argument("--adam-beta2", type=float, default=0.95)
    parser.add_argument("--adam-epsilon", type=float, default=1e-8)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    add_bool_argument(parser, "gradient-checkpointing", default=True)
    parser.add_argument("--shuffle-buffer", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--logging-steps", type=int, default=5)
    parser.add_argument("--dataloader-num-workers", type=int, default=0)
    parser.add_argument(
        "--save-steps",
        type=int,
        default=0,
        help="Save resumable Trainer checkpoints every N optimizer steps. 0 disables stage checkpoints.",
    )
    parser.add_argument(
        "--save-total-limit",
        type=int,
        default=1,
        help="Keep only this many checkpoints per stage when --save-steps is enabled.",
    )
    parser.add_argument(
        "--resume-from-checkpoint",
        default="auto",
        help="Use 'auto' to resume each stage from its latest checkpoint, 'none' to disable, or a checkpoint path.",
    )

    parser.add_argument("--lora-r", type=int, default=32)
    parser.add_argument("--lora-alpha", type=int, default=64)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    return parser.parse_args()


def main() -> None:
    started_at = time.time()
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(normalize_device(args.device))
    dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[args.dtype]

    tokenizer = AutoTokenizer.from_pretrained(
        args.pruned_checkpoint,
        use_fast=True,
        fix_mistral_regex=False,
    )
    drop_tokenizer_runtime_compat_flags(tokenizer)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    load_chat_template_if_needed(tokenizer, args.pruned_checkpoint)
    if tokenizer.chat_template is None:
        raise RuntimeError("This checkpoint tokenizer has no chat_template; use an instruct checkpoint.")

    model = AutoModelForCausalLM.from_pretrained(
        args.pruned_checkpoint,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        attn_implementation="sdpa",
    )
    model.config.use_cache = False
    if getattr(model.config, "pad_token_id", None) is None:
        model.config.pad_token_id = tokenizer.pad_token_id
    configure_deterministic_generation(model, args)
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model = maybe_apply_lora(model, args)
    model.to(device)

    if args.target_perplexity <= 0:
        raise ValueError("--target-perplexity must be positive.")
    if args.max_seq_len < 256:
        raise ValueError("--max-seq-len must be at least 256.")
    if args.save_steps < 0:
        raise ValueError("--save-steps must be >= 0.")
    if args.save_total_limit < 0:
        raise ValueError("--save-total-limit must be >= 0.")
    if args.finance_balance_buffer_size <= 0:
        raise ValueError("--finance-balance-buffer-size must be positive.")
    if args.generation_repetition_penalty < 1.0:
        raise ValueError("--generation-repetition-penalty must be >= 1.0.")
    if args.generation_no_repeat_ngram_size < 0:
        raise ValueError("--generation-no-repeat-ngram-size must be >= 0.")
    if args.resume_from_checkpoint not in {"auto", "none"}:
        resume_path = Path(args.resume_from_checkpoint)
        if not resume_path.exists():
            raise ValueError(
                "--resume-from-checkpoint must be 'auto', 'none', or an existing checkpoint path."
            )
    finance_eval_labels = parse_finance_eval_labels(args.finance_eval_labels)
    if args.finance_eval_expected_label:
        expected_label = normalize_finance_label(args.finance_eval_expected_label) or args.finance_eval_expected_label
        if expected_label not in set(finance_eval_labels):
            raise ValueError(
                "--finance-eval-expected-label must be included in --finance-eval-labels."
            )
    if not (
        args.finance_original_instruction
        or args.finance_good_bad_prompt
        or args.finance_eval_style_prompt
        or args.finance_neutral_prompt
    ):
        raise ValueError("At least one finance training prompt style must be enabled.")
    mixture_weights = [
        args.lm_fineweb_weight,
        args.lm_c4_weight,
        args.lm_wikipedia_weight,
        args.lm_refinement_fineweb_weight,
        args.lm_refinement_c4_weight,
        args.lm_refinement_wikipedia_weight,
        args.unified_task_lm_weight,
        args.unified_task_chat_weight,
        args.unified_task_knowledge_weight,
        args.unified_task_mmlu_weight,
        args.unified_task_exact_instruction_weight,
        args.unified_task_constraint_weight,
        args.unified_task_summary_instruction_weight,
        args.unified_task_summarization_weight,
        args.unified_task_finance_weight,
        args.stabilization_lm_weight,
        args.stabilization_chat_weight,
        args.stabilization_knowledge_weight,
        args.stabilization_mmlu_weight,
        args.stabilization_exact_instruction_weight,
        args.stabilization_constraint_weight,
        args.stabilization_summary_instruction_weight,
        args.stabilization_summarization_weight,
        args.stabilization_finance_weight,
    ]
    if any(weight < 0 for weight in mixture_weights):
        raise ValueError("Dataset mixture weights cannot be negative.")

    eval_data: Optional[PerplexityEvalData] = None
    if (
        args.lm_recovery_steps > 0
        or args.lm_refinement_steps > 0
        or args.unified_task_mix_steps > 0
        or args.stabilization_steps > 0
        or args.eval_at_end
    ):
        eval_data = prepare_perplexity_eval_data(model, tokenizer, args)

    stage_results: List[Dict[str, Any]] = []
    boundary_evals: Dict[str, Any] = {}
    if eval_data is not None:
        boundary_evals["before_training"] = evaluate_prepared_perplexity(model, args, eval_data)

    def build_chat_dataset(
        dataset_name: str,
        dataset_config: Optional[str],
        split: str,
        max_samples: Optional[int],
        *,
        seed_offset: int,
    ) -> StreamingChatSFTDataset:
        source = load_streaming_dataset(
            dataset_name,
            dataset_config,
            split,
            seed=args.seed + seed_offset,
            shuffle_buffer=args.shuffle_buffer,
            max_samples=max_samples,
        )
        return StreamingChatSFTDataset(
            source,
            tokenizer,
            max_seq_len=args.max_seq_len,
            messages_column=args.messages_column,
            system_prompt=args.system_prompt,
        )

    def build_text_lm_dataset(
        dataset_name: str,
        dataset_config: Optional[str],
        split: str,
        text_column: str,
        max_samples: Optional[int],
        *,
        seed_offset: int,
    ) -> StreamingTextLMDataset:
        source = load_streaming_dataset(
            dataset_name,
            dataset_config,
            split,
            seed=args.seed + seed_offset,
            shuffle_buffer=args.shuffle_buffer,
            max_samples=max_samples,
        )
        return StreamingTextLMDataset(
            source,
            tokenizer,
            max_seq_len=args.max_seq_len,
            text_column=text_column,
        )

    def build_lm_mixture(
        seed_offset: int,
        *,
        fineweb_weight: float,
        c4_weight: float,
        wikipedia_weight: float,
    ) -> WeightedMixtureDataset:
        entries: List[Tuple[str, IterableDataset, float]] = []
        if fineweb_weight > 0:
            entries.append(
                (
                    "fineweb_edu",
                    build_text_lm_dataset(
                        args.lm_dataset_name,
                        args.lm_dataset_config,
                        args.lm_split,
                        args.lm_text_column,
                        args.max_lm_samples,
                        seed_offset=seed_offset,
                    ),
                    fineweb_weight,
                )
            )
        if c4_weight > 0:
            entries.append(
                (
                    "c4",
                    build_text_lm_dataset(
                        args.c4_dataset_name,
                        args.c4_dataset_config,
                        args.c4_split,
                        args.c4_text_column,
                        args.max_c4_samples,
                        seed_offset=seed_offset + 1,
                    ),
                    c4_weight,
                )
            )
        if wikipedia_weight > 0:
            entries.append(
                (
                    "wikipedia_lm",
                    build_text_lm_dataset(
                        args.wikipedia_lm_dataset_name,
                        args.wikipedia_lm_dataset_config,
                        args.wikipedia_lm_split,
                        args.wikipedia_lm_text_column,
                        args.max_wikipedia_lm_samples,
                        seed_offset=seed_offset + 2,
                    ),
                    wikipedia_weight,
                )
            )
        return WeightedMixtureDataset(entries, seed=args.seed + 10_000 + seed_offset)

    def build_lm_dataset(seed_offset: int) -> WeightedMixtureDataset:
        return build_lm_mixture(
            seed_offset,
            fineweb_weight=args.lm_fineweb_weight,
            c4_weight=args.lm_c4_weight,
            wikipedia_weight=args.lm_wikipedia_weight,
        )

    def build_summarization_dataset(seed_offset: int) -> StreamingSummarizationDataset:
        source = load_streaming_dataset(
            args.summarization_dataset_name,
            args.summarization_dataset_config,
            args.summarization_split,
            seed=args.seed + seed_offset,
            shuffle_buffer=args.shuffle_buffer,
            max_samples=args.max_summarization_samples,
        )
        return StreamingSummarizationDataset(
            source,
            tokenizer,
            max_seq_len=args.max_seq_len,
            article_column=args.summarization_article_column,
            summary_column=args.summarization_summary_column,
            system_prompt=args.system_prompt,
            output_reserve_tokens=args.summary_output_reserve_tokens,
            seed=args.seed + seed_offset,
        )

    def build_finance_dataset(seed_offset: int) -> StreamingFinanceSentimentDataset:
        source = load_streaming_dataset(
            args.finance_dataset_name,
            args.finance_dataset_config,
            args.finance_split,
            seed=args.seed + seed_offset,
            shuffle_buffer=args.shuffle_buffer,
            max_samples=args.max_finance_samples,
            skip_samples=args.finance_eval_holdout_samples,
        )
        return StreamingFinanceSentimentDataset(
            source,
            tokenizer,
            max_seq_len=args.max_seq_len,
            system_prompt=args.system_prompt,
            include_original_instruction=args.finance_original_instruction,
            include_good_bad_prompt=args.finance_good_bad_prompt,
            include_eval_style_prompt=args.finance_eval_style_prompt,
            include_neutral_prompt=args.finance_neutral_prompt,
            balance_labels=finance_eval_labels if args.finance_balance_labels else None,
            balance_buffer_size=args.finance_balance_buffer_size,
        )

    def build_mmlu_sft_dataset(seed_offset: int) -> StreamingMultipleChoiceQADataset:
        source = load_streaming_dataset(
            args.mmlu_sft_dataset_name,
            args.mmlu_sft_dataset_config,
            args.mmlu_sft_split,
            seed=args.seed + seed_offset,
            shuffle_buffer=args.shuffle_buffer,
            max_samples=args.max_mmlu_sft_samples,
        )
        return StreamingMultipleChoiceQADataset(
            source,
            tokenizer,
            max_seq_len=args.max_seq_len,
            system_prompt=args.system_prompt,
        )

    def build_exact_instruction_dataset(seed_offset: int) -> StreamingExactInstructionDataset:
        return StreamingExactInstructionDataset(
            tokenizer,
            max_seq_len=args.max_seq_len,
            system_prompt=args.system_prompt,
            seed=args.seed + seed_offset,
        )

    if args.lm_recovery_steps > 0:
        callbacks: List[TrainerCallback] = []
        if eval_data is not None:
            callbacks.append(
                PerplexityTargetCallback(
                    stage_name="lm_recovery",
                    output_dir=output_dir,
                    args=args,
                    eval_data=eval_data,
                    target_perplexity=args.target_perplexity,
                    min_steps=args.lm_min_steps,
                    eval_interval=args.perplexity_check_steps,
                    plateau_patience=args.perplexity_plateau_patience,
                    plateau_min_delta=args.perplexity_plateau_min_delta,
                )
            )
        stage_results.append(
            run_stage(
                stage_name="lm_recovery",
                model=model,
                tokenizer=tokenizer,
                train_dataset=build_lm_dataset(0),
                output_dir=output_dir,
                args=args,
                max_steps=args.lm_recovery_steps,
                learning_rate=args.lm_learning_rate,
                callbacks=callbacks,
                lr_scheduler_type=args.lm_lr_scheduler_type,
                warmup_ratio=args.lm_warmup_ratio,
                weight_decay=args.weight_decay,
                adam_beta2=args.adam_beta2,
            )
        )

    if args.lm_refinement_steps > 0:
        refinement_callbacks: List[TrainerCallback] = []
        if eval_data is not None:
            refinement_callbacks.append(
                PerplexityTargetCallback(
                    stage_name="lm_refinement",
                    output_dir=output_dir,
                    args=args,
                    eval_data=eval_data,
                    target_perplexity=args.target_perplexity,
                    min_steps=args.lm_refinement_min_steps,
                    eval_interval=args.perplexity_check_steps,
                    plateau_patience=args.perplexity_plateau_patience,
                    plateau_min_delta=args.perplexity_plateau_min_delta,
                )
            )
        stage_results.append(
            run_stage(
                stage_name="lm_refinement",
                model=model,
                tokenizer=tokenizer,
                train_dataset=build_lm_mixture(
                    5,
                    fineweb_weight=args.lm_refinement_fineweb_weight,
                    c4_weight=args.lm_refinement_c4_weight,
                    wikipedia_weight=args.lm_refinement_wikipedia_weight,
                ),
                output_dir=output_dir,
                args=args,
                max_steps=args.lm_refinement_steps,
                learning_rate=args.lm_refinement_learning_rate,
                callbacks=refinement_callbacks,
                lr_scheduler_type="constant_with_warmup",
                warmup_ratio=args.lm_warmup_ratio,
                weight_decay=args.weight_decay,
                adam_beta2=args.adam_beta2,
            )
        )

    if args.chat_sft_steps > 0:
        stage_results.append(
            run_stage(
                stage_name="chat_sft",
                model=model,
                tokenizer=tokenizer,
                train_dataset=build_chat_dataset(
                    args.chat_dataset_name,
                    args.chat_dataset_config,
                    args.chat_split,
                    args.max_chat_samples,
                    seed_offset=10,
                ),
                output_dir=output_dir,
                args=args,
                max_steps=args.chat_sft_steps,
                learning_rate=args.chat_learning_rate,
            )
        )

    if args.knowledge_sft_steps > 0:
        stage_results.append(
            run_stage(
                stage_name="knowledge_sft",
                model=model,
                tokenizer=tokenizer,
                train_dataset=build_chat_dataset(
                    args.knowledge_dataset_name,
                    args.knowledge_dataset_config,
                    args.knowledge_split,
                    args.max_knowledge_samples,
                    seed_offset=20,
                ),
                output_dir=output_dir,
                args=args,
                max_steps=args.knowledge_sft_steps,
                learning_rate=args.knowledge_learning_rate,
            )
        )

    if args.mmlu_sft_steps > 0:
        stage_results.append(
            run_stage(
                stage_name="mmlu_sft",
                model=model,
                tokenizer=tokenizer,
                train_dataset=build_mmlu_sft_dataset(25),
                output_dir=output_dir,
                args=args,
                max_steps=args.mmlu_sft_steps,
                learning_rate=args.mmlu_sft_learning_rate,
            )
        )

    if args.exact_instruction_sft_steps > 0:
        stage_results.append(
            run_stage(
                stage_name="exact_instruction_sft",
                model=model,
                tokenizer=tokenizer,
                train_dataset=build_exact_instruction_dataset(28),
                output_dir=output_dir,
                args=args,
                max_steps=args.exact_instruction_sft_steps,
                learning_rate=args.exact_instruction_learning_rate,
            )
        )

    if args.constraint_sft_steps > 0:
        stage_results.append(
            run_stage(
                stage_name="constraint_sft",
                model=model,
                tokenizer=tokenizer,
                train_dataset=build_chat_dataset(
                    args.constraint_dataset_name,
                    args.constraint_dataset_config,
                    args.constraint_split,
                    args.max_constraint_samples,
                    seed_offset=30,
                ),
                output_dir=output_dir,
                args=args,
                max_steps=args.constraint_sft_steps,
                learning_rate=args.constraint_learning_rate,
            )
        )

    if args.summary_instruction_sft_steps > 0:
        stage_results.append(
            run_stage(
                stage_name="summary_instruction_sft",
                model=model,
                tokenizer=tokenizer,
                train_dataset=build_chat_dataset(
                    args.summary_instruction_dataset_name,
                    args.summary_instruction_dataset_config,
                    args.summary_instruction_split,
                    args.max_summary_instruction_samples,
                    seed_offset=40,
                ),
                output_dir=output_dir,
                args=args,
                max_steps=args.summary_instruction_sft_steps,
                learning_rate=args.summary_instruction_learning_rate,
            )
        )

    if args.summarization_sft_steps > 0:
        stage_results.append(
            run_stage(
                stage_name="summarization_sft",
                model=model,
                tokenizer=tokenizer,
                train_dataset=build_summarization_dataset(50),
                output_dir=output_dir,
                args=args,
                max_steps=args.summarization_sft_steps,
                learning_rate=args.summarization_learning_rate,
            )
        )

    if args.finance_sft_steps > 0:
        stage_results.append(
            run_stage(
                stage_name="finance_sft",
                model=model,
                tokenizer=tokenizer,
                train_dataset=build_finance_dataset(60),
                output_dir=output_dir,
                args=args,
                max_steps=args.finance_sft_steps,
                learning_rate=args.finance_learning_rate,
            )
        )

    if args.unified_task_mix_steps > 0:
        unified_entries: List[Tuple[str, IterableDataset, float]] = []
        if args.unified_task_lm_weight > 0:
            unified_entries.append(
                (
                    "language_modeling",
                    build_lm_mixture(
                        200,
                        fineweb_weight=args.lm_refinement_fineweb_weight,
                        c4_weight=args.lm_refinement_c4_weight,
                        wikipedia_weight=args.lm_refinement_wikipedia_weight,
                    ),
                    args.unified_task_lm_weight,
                )
            )
        if args.unified_task_chat_weight > 0:
            unified_entries.append(
                (
                    "general_chat",
                    build_chat_dataset(
                        args.chat_dataset_name,
                        args.chat_dataset_config,
                        args.chat_split,
                        args.max_chat_samples,
                        seed_offset=210,
                    ),
                    args.unified_task_chat_weight,
                )
            )
        if args.unified_task_knowledge_weight > 0:
            unified_entries.append(
                (
                    "knowledge",
                    build_chat_dataset(
                        args.knowledge_dataset_name,
                        args.knowledge_dataset_config,
                        args.knowledge_split,
                        args.max_knowledge_samples,
                        seed_offset=220,
                    ),
                    args.unified_task_knowledge_weight,
                )
            )
        if args.unified_task_mmlu_weight > 0:
            unified_entries.append(("mmlu_multiple_choice", build_mmlu_sft_dataset(225), args.unified_task_mmlu_weight))
        if args.unified_task_exact_instruction_weight > 0:
            unified_entries.append(
                ("exact_instruction", build_exact_instruction_dataset(228), args.unified_task_exact_instruction_weight)
            )
        if args.unified_task_constraint_weight > 0:
            unified_entries.append(
                (
                    "constraints",
                    build_chat_dataset(
                        args.constraint_dataset_name,
                        args.constraint_dataset_config,
                        args.constraint_split,
                        args.max_constraint_samples,
                        seed_offset=230,
                    ),
                    args.unified_task_constraint_weight,
                )
            )
        if args.unified_task_summary_instruction_weight > 0:
            unified_entries.append(
                (
                    "summary_instruction",
                    build_chat_dataset(
                        args.summary_instruction_dataset_name,
                        args.summary_instruction_dataset_config,
                        args.summary_instruction_split,
                        args.max_summary_instruction_samples,
                        seed_offset=240,
                    ),
                    args.unified_task_summary_instruction_weight,
                )
            )
        if args.unified_task_summarization_weight > 0:
            unified_entries.append(
                ("article_summarization", build_summarization_dataset(250), args.unified_task_summarization_weight)
            )
        if args.unified_task_finance_weight > 0:
            unified_entries.append(("finance", build_finance_dataset(260), args.unified_task_finance_weight))
        unified_task_dataset = WeightedMixtureDataset(unified_entries, seed=args.seed + 20_000)
        unified_callbacks: List[TrainerCallback] = []
        if eval_data is not None:
            unified_callbacks.append(
                PerplexityTargetCallback(
                    stage_name="unified_task_mix",
                    output_dir=output_dir,
                    args=args,
                    eval_data=eval_data,
                    target_perplexity=args.target_perplexity,
                    min_steps=args.unified_task_mix_min_steps,
                    eval_interval=args.perplexity_check_steps,
                )
            )
        stage_results.append(
            run_stage(
                stage_name="unified_task_mix",
                model=model,
                tokenizer=tokenizer,
                train_dataset=unified_task_dataset,
                output_dir=output_dir,
                args=args,
                max_steps=args.unified_task_mix_steps,
                learning_rate=args.unified_task_mix_learning_rate,
                callbacks=unified_callbacks,
                lr_scheduler_type="constant_with_warmup",
                warmup_ratio=args.lm_warmup_ratio,
                weight_decay=args.weight_decay,
                adam_beta2=args.adam_beta2,
            )
        )

    if eval_data is not None:
        boundary_evals["before_stabilization"] = evaluate_prepared_perplexity(model, args, eval_data)

    if args.stabilization_steps > 0:
        stabilization_entries: List[Tuple[str, IterableDataset, float]] = []
        if args.stabilization_lm_weight > 0:
            stabilization_entries.append(("lm", build_lm_dataset(100), args.stabilization_lm_weight))
        if args.stabilization_chat_weight > 0:
            stabilization_entries.append(
                (
                    "chat",
                    build_chat_dataset(
                        args.chat_dataset_name,
                        args.chat_dataset_config,
                        args.chat_split,
                        args.max_chat_samples,
                        seed_offset=110,
                    ),
                    args.stabilization_chat_weight,
                )
            )
        if args.stabilization_knowledge_weight > 0:
            stabilization_entries.append(
                (
                    "knowledge",
                    build_chat_dataset(
                        args.knowledge_dataset_name,
                        args.knowledge_dataset_config,
                        args.knowledge_split,
                        args.max_knowledge_samples,
                        seed_offset=120,
                    ),
                    args.stabilization_knowledge_weight,
                )
            )
        if args.stabilization_mmlu_weight > 0:
            stabilization_entries.append(("mmlu_multiple_choice", build_mmlu_sft_dataset(125), args.stabilization_mmlu_weight))
        if args.stabilization_exact_instruction_weight > 0:
            stabilization_entries.append(
                ("exact_instruction", build_exact_instruction_dataset(128), args.stabilization_exact_instruction_weight)
            )
        if args.stabilization_constraint_weight > 0:
            stabilization_entries.append(
                (
                    "constraints",
                    build_chat_dataset(
                        args.constraint_dataset_name,
                        args.constraint_dataset_config,
                        args.constraint_split,
                        args.max_constraint_samples,
                        seed_offset=130,
                    ),
                    args.stabilization_constraint_weight,
                )
            )
        if args.stabilization_summary_instruction_weight > 0:
            stabilization_entries.append(
                (
                    "summary_instruction",
                    build_chat_dataset(
                        args.summary_instruction_dataset_name,
                        args.summary_instruction_dataset_config,
                        args.summary_instruction_split,
                        args.max_summary_instruction_samples,
                        seed_offset=140,
                    ),
                    args.stabilization_summary_instruction_weight,
                )
            )
        if args.stabilization_summarization_weight > 0:
            stabilization_entries.append(("summarization", build_summarization_dataset(150), args.stabilization_summarization_weight))
        if args.stabilization_finance_weight > 0:
            stabilization_entries.append(("finance", build_finance_dataset(160), args.stabilization_finance_weight))
        mixture = WeightedMixtureDataset(stabilization_entries, seed=args.seed + 1000)
        stabilization_callbacks: List[TrainerCallback] = []
        if eval_data is not None:
            stabilization_callbacks.append(
                PerplexityTargetCallback(
                    stage_name="stabilization",
                    output_dir=output_dir,
                    args=args,
                    eval_data=eval_data,
                    target_perplexity=args.target_perplexity,
                    min_steps=args.stabilization_min_steps,
                    eval_interval=args.perplexity_check_steps,
                )
            )
        stage_results.append(
            run_stage(
                stage_name="stabilization",
                model=model,
                tokenizer=tokenizer,
                train_dataset=mixture,
                output_dir=output_dir,
                args=args,
                max_steps=args.stabilization_steps,
                learning_rate=args.stabilization_learning_rate,
                callbacks=stabilization_callbacks,
            )
        )

    checkpoint_dir, model, summary = save_final(model, tokenizer, output_dir, args, started_at)
    summary["stages"] = stage_results
    summary["perplexity_boundaries"] = boundary_evals
    write_summary(output_dir, summary)
    print(f"Saved recovered checkpoint to {checkpoint_dir}")

    if args.eval_at_end:
        eval_results: Dict[str, Any] = {}
        try:
            eval_results["wikitext2_validation"] = evaluate_wikitext2_perplexity(
                model,
                tokenizer,
                args,
                eval_data=eval_data,
            )
        except Exception as exc:
            eval_results["wikitext2_validation_error"] = repr(exc)
            print(f"Wikitext-2 validation perplexity eval failed: {exc!r}")

        try:
            eval_results["finance_sentiment"] = evaluate_finance_sentiment(model, tokenizer, args)
        except Exception as exc:
            eval_results["finance_sentiment_error"] = repr(exc)
            print(f"Finance sentiment smoke test failed: {exc!r}")

        try:
            eval_results["instruction_and_summarization"] = evaluate_instruction_and_summary(
                model,
                tokenizer,
                args,
            )
        except Exception as exc:
            eval_results["instruction_and_summarization_error"] = repr(exc)
            print(f"Instruction/summarization smoke tests failed: {exc!r}")

        try:
            eval_results["finance_holdout_benchmark"] = evaluate_finance_holdout(
                model,
                tokenizer,
                args,
            )
        except Exception as exc:
            eval_results["finance_holdout_benchmark_error"] = repr(exc)
            print(f"Finance holdout benchmark failed: {exc!r}")

        try:
            eval_results["summarization_holdout_benchmark"] = evaluate_summarization_holdout(
                model,
                tokenizer,
                args,
            )
        except Exception as exc:
            eval_results["summarization_holdout_benchmark_error"] = repr(exc)
            print(f"Summarization holdout benchmark failed: {exc!r}")

        try:
            eval_results["instruction_compliance_benchmark"] = evaluate_instruction_compliance(
                model,
                tokenizer,
                args,
            )
        except Exception as exc:
            eval_results["instruction_compliance_benchmark_error"] = repr(exc)
            print(f"Instruction compliance benchmark failed: {exc!r}")

        if args.eval_mmlu:
            try:
                eval_results["knowledge_mmlu_benchmark"] = evaluate_knowledge_mmlu(
                    model,
                    tokenizer,
                    args,
                )
            except Exception as exc:
                eval_results["knowledge_mmlu_benchmark_error"] = repr(exc)
                print(f"MMLU knowledge benchmark failed: {exc!r}")

        summary["eval"] = eval_results
        summary["runtime_seconds"] = time.time() - started_at
        write_summary(output_dir, summary)


if __name__ == "__main__":
    main()
