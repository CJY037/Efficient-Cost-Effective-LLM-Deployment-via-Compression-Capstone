#!/usr/bin/env python3
"""
Benchmark task-specific LLM inference speed for dense and structurally pruned
Llama 3.2 3B Instruct checkpoints.

The benchmark models standalone batch-1 LLM inference. It uses exact,
standardized tokenized prompt lengths and task-appropriate fixed output
budgets with an on-device KV cache. Both models receive identical token IDs.

Reported LLM inference metrics:
  - Prompt tokens and output tokens
  - Prefill latency and prefill tok/s
  - Decode latency/token and decode tok/s
  - End-to-end latency and end-to-end tok/s
  - Peak VRAM and loaded-model footprint
  - Prefill, decode and end-to-end speedup

Additional diagnostics include TTFT, total-token throughput, request
throughput, reserved VRAM and raw repeated measurements.

Timing is warm, synchronized wall-clock timing. Model loading, prompt
tokenization, HTTP/network overhead and browser rendering are reported or
excluded separately; they are not mixed into model generation latency.

Default task suite:
  - Uses synthetic exact-length prompts by default, matching the c2 demo run
  - Finance sentiment: 4,096 prompt tokens and 1 output token
  - Summarization: 8,192 prompt tokens and 256 output tokens
  - General generation: 8,192 prompt tokens and 20 output tokens
  - Use --prompt-source demo_files only when you intentionally want the
    benchmark folder article files for finance_sentiment and summarization

Default model comparison:
  dense  : meta-llama/Llama-3.2-3B-Instruct
  pruned : /home/user/capstone/pruned-llama32-3b-instruct

Example:
  CUDA_VISIBLE_DEVICES=1 python -u llama32_inference_speedup_benchmark.py

Quick smoke test:
  CUDA_VISIBLE_DEVICES=1 python -u llama32_inference_speedup_benchmark.py \
      --tasks summarization --summarization-prompt-lengths 128 \
      --prompts-per-length 1 --summarization-output-tokens 16 \
      --warmup 1 --repeats 3 --rounds 1 --bootstrap-samples 200

Long-article/summarization-oriented run:
  CUDA_VISIBLE_DEVICES=1 python -u llama32_inference_speedup_benchmark.py \
      --tasks summarization \
      --summarization-prompt-lengths 512,2048,4096,8192 \
      --summarization-output-tokens 256 \
      --warmup 3 --repeats 10 --rounds 2

Task + general report:
  CUDA_VISIBLE_DEVICES=1 python -u llama32_inference_speedup_benchmark.py \
      --tasks finance_sentiment,summarization,general_generation
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import os
import platform
import random
import statistics
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

# The benchmark launches nvidia-smi for optional telemetry after tokenization.
# Disable tokenizer worker pools to avoid fork-after-thread warnings and to keep
# CPU-side benchmark behavior predictable.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
import transformers
from huggingface_hub import constants as hf_constants
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers.generation.streamers import BaseStreamer


DEFAULT_DENSE_MODEL = "meta-llama/Llama-3.2-3B-Instruct"
DEFAULT_PRUNED_MODEL = "/home/user/capstone/pruned-llama32-3b-instruct"
DEFAULT_OUTPUT_JSON = "output/llama32_task_recovery_speedup.json"
BENCHMARK_TITLE = "Website benchmark Llama 3.2 speedup benchmark"
REPO_ROOT = Path(__file__).resolve().parent
BENCHMARKS_DIR = REPO_ROOT / "benchmarks"

FINANCE_SYSTEM_PROMPT = "You are a helpful, harmless, and honest assistant."
DEFAULT_SYSTEM_PROMPT = (
    f"{FINANCE_SYSTEM_PROMPT} Answer directly and concisely. "
    "Do not invent names, dates, companies, sources, forecasts, or statistics. "
    "If the prompt does not provide a fact, avoid pretending it is known."
)
FINANCE_CALIBRATION_ARTICLE = "No company-specific financial article is provided."
SUMMARIZATION_BENCHMARK_INSTRUCTION = (
    "Write a factual summary of this article. "
    "Cover the main points and important supporting details. "
    "Use only facts stated in the article. "
    "Do not add unsupported dates, numbers, names, or claims. "
    "Avoid repeating the same point."
)

ARTICLE_TEXT = """
Financial technology, commonly called fintech, applies software and modern
data systems to services such as payments, lending, insurance, investing and
financial planning. A digital payment platform may connect consumers,
merchants and banks while performing identity checks, fraud detection and
transaction monitoring. The customer sees a simple application, but the
service depends on secure infrastructure, regulatory controls and reliable
record keeping.

Organizations adopting new technology usually begin with a narrowly defined
problem. They establish a baseline, test the proposed system under realistic
conditions and monitor both quality and operational performance. A faster
system is not necessarily better if it produces unreliable results, and a more
accurate system may still be unsuitable if its latency or operating cost is
too high. Good evaluation therefore measures several dimensions and records
the workload, hardware and software configuration used for the test.

Long documents create an additional challenge. Important facts may occur near
the beginning, middle or end, and a useful summary must preserve the central
argument without copying every detail. The summary should distinguish facts
from opinions, retain important numbers and avoid introducing claims that are
not supported by the source. Clear structure helps readers understand the
result quickly while still allowing them to inspect the original material.
""".strip()

KNOWLEDGE_TEXT = """
Fintech is the use of digital technology to provide or improve financial
services. It covers consumer payments, bank transfers, lending, insurance,
investment platforms, fraud detection and financial planning. A useful
explanation should distinguish the customer-facing application from the
regulated infrastructure behind it. Payment products, for example, normally
depend on identity verification, transaction monitoring, secure storage,
banking partners and procedures for resolving errors.

The benefits can include lower transaction costs, faster service, broader
access and products tailored using data. The risks include cyberattacks,
privacy loss, unfair automated decisions, operational outages and financial
crime. Regulation differs across markets, but providers generally need clear
governance, audit trails, security controls and transparent communication with
customers. New products should be evaluated on reliability and consumer
outcomes rather than novelty alone.
""".strip()

FINANCE_TEXT = """
A regional payments company reported that quarterly revenue increased by 18
percent while operating expenses increased by 11 percent. Management said the
improvement came from higher merchant volume and lower fraud losses. However,
the company also warned that customer acquisition costs were rising and that
a new compliance program would increase spending during the next two
quarters. Cash reserves remained stable, debt was unchanged and no dividend
was announced.

An analyst reviewing this update should separate historical results from
forward-looking statements. Revenue growth and lower fraud losses are
positive evidence, while rising acquisition and compliance costs may pressure
future margins. The appropriate conclusion depends on the time horizon and
should preserve uncertainty instead of treating every favorable sentence as
proof of a uniformly positive outlook.
""".strip()

FINANCE_POSITIVE_TEXT = """
A payment processor reported quarterly revenue growth of 24 percent and
operating profit growth of 31 percent. Transaction volume increased across
every major region, fraud losses declined and management raised its full-year
guidance. The company also repaid part of its debt while maintaining its cash
reserves. Executives said merchant retention remained strong.
""".strip()

FINANCE_NEGATIVE_TEXT = """
A digital lender reported that loan defaults increased sharply during the
quarter. Revenue declined by 9 percent, the company recorded an operating
loss and management reduced its full-year forecast. Regulators also opened a
review of the firm's customer affordability checks. The company said it would
slow new lending and preserve cash until credit conditions improve.
""".strip()

SUMMARIZATION_PROMPT_VARIANTS = (
    {
        "name": "article_summarization",
        "instruction": "",
        "body": ARTICLE_TEXT,
    },
    {
        "name": "knowledge_article",
        "instruction": "",
        "body": KNOWLEDGE_TEXT,
    },
    {
        "name": "financial_article",
        "instruction": "",
        "body": FINANCE_TEXT,
    },
)

SUMMARIZATION_TASK_NAME = "summarization"

GENERAL_GENERATION_PROMPT_VARIANTS = (
    {
        "name": "article_summarization",
        "instruction": (
            "Summarize the following article. Preserve the main facts and important "
            "details, and do not add unsupported claims.\n\nArticle:\n"
        ),
        "body": ARTICLE_TEXT,
    },
    {
        "name": "knowledge_instruction",
        "instruction": (
            "Explain the following material clearly to a general reader. Organize the "
            "answer, follow the instruction precisely, and include the important "
            "benefits and risks.\n\nMaterial:\n"
        ),
        "body": KNOWLEDGE_TEXT,
    },
    {
        "name": "finance_analysis",
        "instruction": (
            "Analyze the following financial update. Identify positive and negative "
            "signals, preserve uncertainty, and provide a concise conclusion.\n\n"
            "Financial update:\n"
        ),
        "body": FINANCE_TEXT,
    },
)

FINANCE_GOOD_LABEL_CANDIDATES = (" Good", "Good")
FINANCE_BAD_LABEL_CANDIDATES = (" Bad", "Bad")

FINANCE_SENTIMENT_PROMPT_VARIANTS = (
    {
        "name": "positive_financial_news",
        "instruction": "",
        "body": FINANCE_POSITIVE_TEXT,
    },
    {
        "name": "negative_financial_news",
        "instruction": "",
        "body": FINANCE_NEGATIVE_TEXT,
    },
    {
        "name": "mixed_financial_news",
        "instruction": "",
        "body": FINANCE_TEXT,
    },
)

BENCHMARK_LATENCY_METRICS = (
    "prefill_latency_ms",
    "decode_ms_per_token",
    "total_latency_ms",
)
BENCHMARK_THROUGHPUT_METRICS = (
    "prefill_tok_s",
    "decode_tok_s",
    "e2e_tok_s",
)
BENCHMARK_MEMORY_METRICS = ("peak_vram_gb",)
BENCHMARK_METRICS = (
    "prefill_latency_ms",
    "prefill_tok_s",
    "decode_ms_per_token",
    "decode_tok_s",
    "total_latency_ms",
    "e2e_tok_s",
    "peak_vram_gb",
)

DIAGNOSTIC_LATENCY_METRICS = (
    "ttft_ms",
    "tpot_ms",
    "decode_ms",
    "e2e_ms",
    "generate_return_latency_ms",
    "post_final_token_overhead_ms",
)
DIAGNOSTIC_THROUGHPUT_METRICS = (
    "prefill_tokens_per_s",
    "decode_tokens_per_s",
    "output_tokens_per_s",
    "total_tokens_per_s",
    "requests_per_s",
)
DIAGNOSTIC_MEMORY_METRICS = (
    "peak_allocated_mb",
    "peak_extra_allocated_mb",
    "peak_reserved_mb",
)
ALL_METRICS = tuple(
    dict.fromkeys(
        BENCHMARK_METRICS
        + DIAGNOSTIC_LATENCY_METRICS
        + DIAGNOSTIC_THROUGHPUT_METRICS
        + DIAGNOSTIC_MEMORY_METRICS
    )
)

BENCHMARK_METRIC_LABELS = {
    "prefill_latency_ms": "Prefill latency (ms)",
    "prefill_tok_s": "Prefill tok/s",
    "decode_ms_per_token": "Decode latency/token (ms)",
    "decode_tok_s": "Decode tok/s",
    "total_latency_ms": "End-to-end latency (ms)",
    "e2e_tok_s": "End-to-end tok/s",
    "peak_vram_gb": "Peak VRAM (GB)",
}


def parse_int_list(value: str, *, name: str) -> List[int]:
    try:
        values = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"{name} must be a comma-separated integer list.") from exc
    if not values or any(item < 1 for item in values):
        raise argparse.ArgumentTypeError(f"{name} values must all be >= 1.")
    if len(set(values)) != len(values):
        raise argparse.ArgumentTypeError(f"{name} must not contain duplicates.")
    return values


def percentile(values: Sequence[float], q: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def summarize_values(values: Sequence[float]) -> Dict[str, float]:
    clean = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    if not clean:
        return {
            "mean": float("nan"),
            "p50": float("nan"),
            "p95": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
            "std": float("nan"),
            "cv_percent": float("nan"),
        }
    mean = statistics.fmean(clean)
    std = statistics.pstdev(clean) if len(clean) > 1 else 0.0
    return {
        "mean": mean,
        "p50": percentile(clean, 0.50),
        "p95": percentile(clean, 0.95),
        "min": min(clean),
        "max": max(clean),
        "std": std,
        "cv_percent": 100.0 * std / mean if mean else float("nan"),
    }


def geometric_mean(values: Iterable[float]) -> float:
    clean = [float(value) for value in values if value is not None and float(value) > 0]
    if not clean:
        return float("nan")
    return math.exp(statistics.fmean(math.log(value) for value in clean))


def safe_ratio(numerator: float, denominator: float) -> float:
    numerator = float(numerator)
    denominator = float(denominator)
    if not math.isfinite(numerator) or not math.isfinite(denominator) or denominator <= 0:
        return float("nan")
    return numerator / denominator


def reduction_percent(original: float, new: float) -> float:
    ratio = safe_ratio(new, original)
    return (1.0 - ratio) * 100.0 if math.isfinite(ratio) else float("nan")


def bootstrap_ratio_confidence_interval(
    dense_values: Sequence[float],
    pruned_values: Sequence[float],
    *,
    pruned_over_dense: bool,
    samples: int,
    seed: int,
) -> Dict[str, Any]:
    dense_clean = [float(value) for value in dense_values if math.isfinite(float(value))]
    pruned_clean = [float(value) for value in pruned_values if math.isfinite(float(value))]
    if not dense_clean or not pruned_clean or samples < 1:
        return {
            "low": float("nan"),
            "high": float("nan"),
            "confidence_level": 0.95,
            "bootstrap_samples": samples,
            "excludes_1": False,
            "pruned_faster_with_95pct_confidence": False,
        }

    rng = random.Random(seed)
    ratios: List[float] = []
    for _ in range(samples):
        dense_median = statistics.median(
            rng.choices(dense_clean, k=len(dense_clean))
        )
        pruned_median = statistics.median(
            rng.choices(pruned_clean, k=len(pruned_clean))
        )
        ratio = (
            safe_ratio(pruned_median, dense_median)
            if pruned_over_dense
            else safe_ratio(dense_median, pruned_median)
        )
        if math.isfinite(ratio):
            ratios.append(ratio)

    low = percentile(ratios, 0.025)
    high = percentile(ratios, 0.975)
    return {
        "low": low,
        "high": high,
        "confidence_level": 0.95,
        "bootstrap_samples": samples,
        "excludes_1": bool(math.isfinite(low) and math.isfinite(high) and (low > 1.0 or high < 1.0)),
        "pruned_faster_with_95pct_confidence": bool(math.isfinite(low) and low > 1.0),
    }


def nvidia_smi_snapshot() -> Dict[str, Any]:
    fields = (
        "index,uuid,name,driver_version,pstate,temperature.gpu,"
        "clocks.sm,clocks.mem,power.draw,power.limit,"
        "utilization.gpu,memory.used,memory.total"
    )
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                f"--query-gpu={fields}",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (FileNotFoundError, subprocess.SubprocessError) as exc:
        return {"available": False, "error": str(exc)}

    names = fields.split(",")
    gpus = []
    for line in completed.stdout.splitlines():
        values = [item.strip() for item in line.split(",")]
        if len(values) == len(names):
            gpus.append(dict(zip(names, values)))
    processes: List[Dict[str, str]] = []
    process_fields = "gpu_uuid,pid,process_name,used_gpu_memory"
    try:
        process_result = subprocess.run(
            [
                "nvidia-smi",
                f"--query-compute-apps={process_fields}",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
        process_names = process_fields.split(",")
        for line in process_result.stdout.splitlines():
            values = [item.strip() for item in line.split(",", maxsplit=3)]
            if len(values) == len(process_names):
                processes.append(dict(zip(process_names, values)))
    except (FileNotFoundError, subprocess.SubprocessError):
        pass
    return {"available": True, "gpus": gpus, "compute_processes": processes}


def resolve_model_source(model_name_or_path: str, cache_dir: Optional[str]) -> str:
    """Prefer an existing local path or complete Hugging Face cache snapshot."""
    direct = Path(model_name_or_path).expanduser()
    if direct.exists():
        return str(direct.resolve())

    cache_root = Path(cache_dir).expanduser() if cache_dir else Path(hf_constants.HF_HUB_CACHE)
    repo_dir = cache_root / f"models--{model_name_or_path.replace('/', '--')}"
    ref_path = repo_dir / "refs" / "main"
    if not ref_path.is_file():
        return model_name_or_path

    revision = ref_path.read_text(encoding="utf-8").strip()
    snapshot = repo_dir / "snapshots" / revision
    if not (snapshot / "config.json").is_file():
        return model_name_or_path

    has_weights = any(
        (snapshot / filename).is_file()
        for filename in (
            "model.safetensors",
            "model.safetensors.index.json",
            "pytorch_model.bin",
            "pytorch_model.bin.index.json",
        )
    )
    return str(snapshot.resolve()) if has_weights else model_name_or_path


def longest_common_prefix(first: Sequence[int], second: Sequence[int]) -> int:
    length = 0
    for left, right in zip(first, second):
        if left != right:
            break
        length += 1
    return length


def longest_common_suffix(first: Sequence[int], second: Sequence[int], prefix_len: int) -> int:
    max_length = min(len(first), len(second)) - prefix_len
    length = 0
    while length < max_length and first[-1 - length] == second[-1 - length]:
        length += 1
    return length


def prompt_has_summarization_instruction(prompt: str) -> bool:
    lower = prompt.lower()
    return any(
        term in lower
        for term in ("summarize", "summarise", "summary", "summarization", "summarisation")
    )


def build_summarization_prompt(article_text: str) -> str:
    return (
        f"{SUMMARIZATION_BENCHMARK_INSTRUCTION}\n\n"
        f"Article:\n{article_text.strip()}"
    )


def build_finance_sentiment_prompt(finance_text: str) -> str:
    return (
        "Read the financial news article below.\n\n"
        f"Article:\n{finance_text.strip()}\n\n"
        "Classify the likely investor sentiment. Answer with exactly one word: Good or Bad."
    )


def prepare_prompt_for_task(prompt: str, task_name: str) -> str:
    stripped = prompt.strip()
    if task_name == "finance_sentiment":
        return build_finance_sentiment_prompt(stripped)
    if task_name == SUMMARIZATION_TASK_NAME and not prompt_has_summarization_instruction(stripped):
        return build_summarization_prompt(stripped)
    return stripped


def system_prompt_for_task(task_name: str) -> str:
    return FINANCE_SYSTEM_PROMPT if task_name == "finance_sentiment" else DEFAULT_SYSTEM_PROMPT


def apply_website_chat_template_ids(
    tokenizer,
    user_content: str,
    *,
    system_prompt: str,
) -> List[int]:
    # Match llama32_3bi_web.py: render chat-template text, then tokenize it.
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content.strip()},
    ]
    if getattr(tokenizer, "chat_template", None):
        try:
            rendered = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            return [
                int(token_id)
                for token_id in tokenizer(rendered, add_special_tokens=True)["input_ids"]
            ]
        except Exception:
            pass
    rendered = f"System: {system_prompt}\n\nQuestion: {user_content.strip()}\nAnswer:"
    return [int(token_id) for token_id in tokenizer(rendered, add_special_tokens=True)["input_ids"]]


def build_website_task_prompt_ids(tokenizer, raw_prompt: str, task_name: str) -> torch.Tensor:
    prepared_prompt = prepare_prompt_for_task(raw_prompt, task_name)
    encoded = apply_website_chat_template_ids(
        tokenizer,
        prepared_prompt,
        system_prompt=system_prompt_for_task(task_name),
    )
    return torch.tensor(encoded, dtype=torch.long)


def build_exact_length_prompt(
    tokenizer,
    target_tokens: int,
    *,
    task_name: str,
    instruction: str,
    body_text: str,
    closing_instruction: str = "",
) -> torch.Tensor:
    """
    Construct a valid chat-template prefix and assistant-generation suffix with
    an article body truncated at the token level to exactly target_tokens.
    """
    base_ids = build_website_task_prompt_ids(
        tokenizer,
        instruction + closing_instruction,
        task_name,
    ).tolist()

    repetitions = max(8, target_tokens // 100)
    while True:
        content = (
            instruction
            + ((body_text + "\n\n") * repetitions)
            + closing_instruction
        )
        full_ids = build_website_task_prompt_ids(tokenizer, content, task_name).tolist()
        prefix_len = longest_common_prefix(base_ids, full_ids)
        suffix_len = longest_common_suffix(base_ids, full_ids, prefix_len)
        template_tokens = prefix_len + suffix_len
        available_body = len(full_ids) - template_tokens
        needed_body = target_tokens - template_tokens
        if needed_body < 0:
            raise ValueError(
                f"Prompt length {target_tokens} is smaller than the chat-template "
                f"overhead ({template_tokens} tokens)."
            )
        if available_body >= needed_body:
            break
        repetitions *= 2

    prefix = full_ids[:prefix_len]
    suffix = full_ids[len(full_ids) - suffix_len :] if suffix_len else []
    body = full_ids[prefix_len : len(full_ids) - suffix_len if suffix_len else len(full_ids)]
    fitted = prefix + body[:needed_body] + suffix
    if len(fitted) != target_tokens:
        raise RuntimeError(f"Built {len(fitted)} prompt tokens; expected {target_tokens}.")
    return torch.tensor(fitted, dtype=torch.long)


def load_tokenizer(source: str, cache_dir: Optional[str], local_files_only: bool):
    is_local = Path(source).exists()
    tokenizer = AutoTokenizer.from_pretrained(
        source,
        cache_dir=cache_dir,
        local_files_only=local_files_only or is_local,
        use_fast=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    if not getattr(tokenizer, "chat_template", None):
        raise ValueError("The selected tokenizer does not define a chat template.")
    return tokenizer


def demo_prompt_variants_for_task(task_name: str) -> List[Dict[str, Any]]:
    if task_name == "finance_sentiment":
        paths = (
            sorted(BENCHMARKS_DIR.glob("demo_long_finance_good_*.txt"))
            + sorted(BENCHMARKS_DIR.glob("demo_long_finance_bad_*.txt"))
        )
    elif task_name == SUMMARIZATION_TASK_NAME:
        paths = sorted(BENCHMARKS_DIR.glob("demo_summarization_article_*.txt"))
    else:
        return []

    variants: List[Dict[str, Any]] = []
    for path in paths:
        if not path.is_file():
            continue
        variants.append(
            {
                "name": path.stem,
                "raw_prompt": path.read_text(encoding="utf-8").strip(),
                "source_file": str(path.relative_to(REPO_ROOT)),
            }
        )
    return variants


def build_prompt_collections(
    tokenizer,
    task_workloads: Dict[str, Dict[str, Any]],
    selected_prompt_variants: Dict[str, List[Dict[str, Any]]],
    *,
    prompt_source: str,
) -> Tuple[Dict[str, Dict[int, List[torch.Tensor]]], Dict[str, List[Dict[str, Any]]]]:
    prompts: Dict[str, Dict[int, List[torch.Tensor]]] = {}
    prompt_records: Dict[str, List[Dict[str, Any]]] = {}

    for task_name, workload in task_workloads.items():
        prompts[task_name] = {}
        prompt_records[task_name] = []
        use_demo_files = (
            prompt_source == "demo_files"
            and task_name in {"finance_sentiment", SUMMARIZATION_TASK_NAME}
        )

        if use_demo_files:
            demo_variants = selected_prompt_variants[task_name]
            if not demo_variants:
                raise FileNotFoundError(
                    f"No demo prompt files found for {task_name} in {BENCHMARKS_DIR}."
                )
            for variant in demo_variants:
                prompt_tensor = build_website_task_prompt_ids(
                    tokenizer,
                    variant["raw_prompt"],
                    task_name,
                )
                prompt_length = int(prompt_tensor.numel())
                prompts[task_name].setdefault(prompt_length, []).append(prompt_tensor)
                prompt_records[task_name].append(
                    {
                        "name": variant["name"],
                        "source_file": variant.get("source_file"),
                        "prompt_tokens": prompt_length,
                        "token_checksum": int(prompt_tensor.sum().item()),
                    }
                )
            workload["prompt_lengths"] = sorted(prompts[task_name])
            continue

        for prompt_length in workload["prompt_lengths"]:
            prompts[task_name][prompt_length] = []
            for variant in selected_prompt_variants[task_name]:
                prompt_tensor = build_exact_length_prompt(
                    tokenizer,
                    prompt_length,
                    task_name=task_name,
                    instruction=variant["instruction"],
                    body_text=variant["body"],
                    closing_instruction=variant.get("closing_instruction", ""),
                )
                prompts[task_name][prompt_length].append(prompt_tensor)
                prompt_records[task_name].append(
                    {
                        "name": variant["name"],
                        "prompt_tokens": prompt_length,
                        "token_checksum": int(prompt_tensor.sum().item()),
                    }
                )

    return prompts, prompt_records


def finance_label_token_metadata(tokenizer) -> Dict[str, Any]:
    """
    Resolve the same one-token Good/Bad candidates used by llama32_3bi_web.py.

    The finance benchmark scores calibrated next-token logits for these
    candidates, selects the higher label, then times one cached continuation
    step. That mirrors the website's finance sentiment benchmark and makes
    decode_tok_s numeric even though the task returns one visible label token.
    """

    def one_token_candidates(candidates: Sequence[str]) -> List[Dict[str, Any]]:
        resolved: List[Dict[str, Any]] = []
        for candidate in candidates:
            token_ids = tokenizer(candidate, add_special_tokens=False)["input_ids"]
            if len(token_ids) == 1:
                resolved.append(
                    {
                        "text": candidate,
                        "stripped": candidate.strip(),
                        "token_id": int(token_ids[0]),
                    }
                )
        return resolved

    good = one_token_candidates(FINANCE_GOOD_LABEL_CANDIDATES)
    bad = one_token_candidates(FINANCE_BAD_LABEL_CANDIDATES)
    if not good or not bad:
        raise ValueError(
            "Could not resolve one-token Good/Bad label candidates for the selected tokenizer."
        )
    return {
        "good": good,
        "bad": bad,
        "implementation": "matches llama32_3bi_web.py calibrated finance Good/Bad scorer",
    }


def dtype_from_name(name: str) -> torch.dtype:
    mapping = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    return mapping[name]


def synchronize(device: torch.device) -> None:
    torch.cuda.synchronize(device)


def cleanup_cuda(device: Optional[torch.device] = None) -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        if device is not None:
            synchronize(device)


def force_module_weight(module: Optional[torch.nn.Module], weight: torch.nn.Parameter) -> None:
    if module is None:
        return
    if hasattr(module, "_parameters"):
        module._parameters["weight"] = weight
    try:
        if getattr(module, "weight", None) is not weight:
            module.weight = weight
    except Exception:
        pass


def ensure_tied_lm_head(model: torch.nn.Module) -> None:
    """Match the website loader when checkpoints omit a tied lm_head.weight."""
    try:
        input_embeddings = model.get_input_embeddings()
        output_embeddings = model.get_output_embeddings()
    except Exception:
        return
    if input_embeddings is None:
        return
    input_weight = getattr(input_embeddings, "weight", None)
    if input_weight is None or getattr(input_weight, "is_meta", False):
        return
    config_ties_weights = bool(getattr(getattr(model, "config", None), "tie_word_embeddings", True))
    output_weight = getattr(output_embeddings, "weight", None) if output_embeddings is not None else None
    output_is_missing = output_weight is None or getattr(output_weight, "is_meta", False)
    if not config_ties_weights and not output_is_missing:
        return
    force_module_weight(output_embeddings, input_weight)
    force_module_weight(getattr(model, "lm_head", None), input_weight)
    try:
        model.tie_weights()
    except Exception:
        pass
    input_embeddings = model.get_input_embeddings()
    input_weight = getattr(input_embeddings, "weight", None) if input_embeddings is not None else None
    if input_weight is None or getattr(input_weight, "is_meta", False):
        return
    force_module_weight(model.get_output_embeddings(), input_weight)
    force_module_weight(getattr(model, "lm_head", None), input_weight)


def model_parameter_metadata(model: torch.nn.Module) -> Dict[str, Any]:
    parameters = list(model.parameters())
    parameter_count = sum(parameter.numel() for parameter in parameters)
    trainable_count = sum(parameter.numel() for parameter in parameters if parameter.requires_grad)
    parameter_bytes = sum(parameter.numel() * parameter.element_size() for parameter in parameters)
    config = model.config
    return {
        "parameter_count": int(parameter_count),
        "trainable_parameter_count": int(trainable_count),
        "parameter_bytes": int(parameter_bytes),
        "parameter_gib": parameter_bytes / (1024**3),
        "architecture": {
            "hidden_size": getattr(config, "hidden_size", None),
            "intermediate_size": getattr(config, "intermediate_size", None),
            "num_hidden_layers": getattr(config, "num_hidden_layers", None),
            "num_attention_heads": getattr(config, "num_attention_heads", None),
            "num_key_value_heads": getattr(config, "num_key_value_heads", None),
            "vocab_size": getattr(config, "vocab_size", None),
            "max_position_embeddings": getattr(config, "max_position_embeddings", None),
        },
        "attention_implementation": getattr(config, "_attn_implementation", None),
    }


def load_model(
    source: str,
    *,
    dtype: torch.dtype,
    device: torch.device,
    cache_dir: Optional[str],
    local_files_only: bool,
    attn_implementation: str,
) -> Tuple[torch.nn.Module, float]:
    is_local = Path(source).exists()
    device_index = device.index
    if device.type == "cuda" and device_index is None:
        device_index = torch.cuda.current_device()
    device_map_value: Any = device_index if device.type == "cuda" else str(device)
    load_start = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        source,
        cache_dir=cache_dir,
        local_files_only=local_files_only or is_local,
        torch_dtype=dtype,
        device_map={"": device_map_value},
        low_cpu_mem_usage=True,
        attn_implementation=attn_implementation,
    )
    ensure_tied_lm_head(model)
    model.eval()
    model.config.use_cache = True
    synchronize(device)
    return model, time.perf_counter() - load_start


class SynchronizedTimingStreamer(BaseStreamer):
    """
    Record when each generated token step becomes available.

    Hugging Face first sends the input prompt to the streamer, followed by one
    generated-token tensor per decode step. Synchronizing before each timestamp
    produces user-visible streaming times instead of asynchronous CPU enqueue
    times. No detokenization is done, so content-dependent text formatting does
    not contaminate model speed comparisons.
    """

    def __init__(self, device: torch.device):
        self.device = device
        self.next_value_is_prompt = True
        self.token_step_times: List[float] = []
        self.generated_tokens = 0
        self.last_token_checksum = 0

    def put(self, value: torch.Tensor) -> None:
        if self.next_value_is_prompt:
            self.next_value_is_prompt = False
            return
        synchronize(self.device)
        now = time.perf_counter()
        flattened = value.reshape(-1)
        self.generated_tokens += int(flattened.numel())
        self.last_token_checksum = int(flattened[0].item())
        self.token_step_times.append(now)

    def end(self) -> None:
        return None


@torch.inference_mode()
def run_fixed_length_request(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    *,
    new_tokens: int,
    repetition_penalty: float,
    no_repeat_ngram_size: int,
    device: torch.device,
    baseline_allocated_bytes: int,
) -> Dict[str, float]:
    """
    Generate exactly new_tokens with greedy decoding.

    Uses the standard Hugging Face generate path. A synchronized
    streamer marks first-token and subsequent-token availability.
    """
    batch_size, prompt_tokens = [int(value) for value in input_ids.shape]
    torch.cuda.reset_peak_memory_stats(device)
    synchronize(device)

    streamer = SynchronizedTimingStreamer(device)
    e2e_start = time.perf_counter()
    sequences = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=new_tokens,
        min_new_tokens=new_tokens,
        do_sample=False,
        temperature=None,
        top_p=None,
        top_k=None,
        num_beams=1,
        use_cache=True,
        streamer=streamer,
        return_dict_in_generate=False,
        output_scores=False,
        pad_token_id=model.generation_config.pad_token_id,
        repetition_penalty=repetition_penalty,
        no_repeat_ngram_size=no_repeat_ngram_size,
    )
    synchronize(device)
    generate_return_seconds = time.perf_counter() - e2e_start

    expected_generated_tokens = batch_size * new_tokens
    actual_generated_tokens = int(sequences.shape[1] - prompt_tokens) * batch_size
    if actual_generated_tokens != expected_generated_tokens:
        raise RuntimeError(
            f"Generation produced {actual_generated_tokens} output tokens; "
            f"expected exactly {expected_generated_tokens}."
        )
    if streamer.generated_tokens != expected_generated_tokens:
        raise RuntimeError(
            f"Streamer observed {streamer.generated_tokens} output tokens; "
            f"expected {expected_generated_tokens}."
        )
    if len(streamer.token_step_times) != new_tokens:
        raise RuntimeError(
            f"Streamer observed {len(streamer.token_step_times)} generation steps; "
            f"expected {new_tokens}."
        )

    first_token_at = streamer.token_step_times[0]
    final_token_at = streamer.token_step_times[-1]
    ttft_seconds = first_token_at - e2e_start
    decode_seconds = (
        final_token_at - first_token_at
        if new_tokens > 1
        else 0.0
    )
    # Streaming request latency ends when the final output token is available.
    e2e_seconds = final_token_at - e2e_start

    decode_tokens = batch_size * max(new_tokens - 1, 0)
    output_tokens = batch_size * new_tokens
    input_tokens = batch_size * prompt_tokens
    peak_allocated = torch.cuda.max_memory_allocated(device)
    peak_reserved = torch.cuda.max_memory_reserved(device)

    checksum = streamer.last_token_checksum

    prefill_latency_ms = ttft_seconds * 1000.0
    prefill_tok_s = input_tokens / ttft_seconds
    decode_ms = decode_seconds * 1000.0
    decode_ms_per_token = (
        decode_ms / (new_tokens - 1)
        if new_tokens > 1
        else float("nan")
    )
    decode_tok_s = decode_tokens / decode_seconds if decode_tokens else float("nan")
    total_latency_ms = e2e_seconds * 1000.0
    e2e_tok_s = output_tokens / e2e_seconds
    peak_vram_gb = peak_allocated / (1024**3)

    return {
        # Exact names used by llama32_3bi_web.py.
        "prefill_latency_ms": prefill_latency_ms,
        "prefill_tok_s": prefill_tok_s,
        "decode_ms_per_token": decode_ms_per_token,
        "decode_tok_s": decode_tok_s,
        "total_latency_ms": total_latency_ms,
        "e2e_tok_s": e2e_tok_s,
        "peak_vram_gb": peak_vram_gb,
        # Additional standard serving diagnostics and compatibility aliases.
        "ttft_ms": prefill_latency_ms,
        "prefill_tokens_per_s": prefill_tok_s,
        "decode_ms": decode_ms,
        "tpot_ms": decode_ms_per_token,
        "decode_tokens_per_s": decode_tok_s,
        "e2e_ms": total_latency_ms,
        "output_tokens_per_s": e2e_tok_s,
        "total_tokens_per_s": (input_tokens + output_tokens) / e2e_seconds,
        "requests_per_s": batch_size / e2e_seconds,
        "generate_return_latency_ms": generate_return_seconds * 1000.0,
        "post_final_token_overhead_ms": max(
            0.0,
            (generate_return_seconds - e2e_seconds) * 1000.0,
        ),
        "peak_allocated_mb": peak_allocated / (1024**2),
        "peak_extra_allocated_mb": max(0, peak_allocated - baseline_allocated_bytes) / (1024**2),
        "peak_reserved_mb": peak_reserved / (1024**2),
        "output_checksum": checksum,
    }


@torch.inference_mode()
def run_live_generate_request(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    *,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    repetition_penalty: float,
    no_repeat_ngram_size: int,
    device: torch.device,
    baseline_allocated_bytes: int,
) -> Dict[str, float]:
    """
    Free generation timing.

    This follows llama32_3bi_web.py's live response path more closely than the
    fixed benchmark: no min_new_tokens, EOS is allowed, and sampling is enabled
    when temperature > 0. The output work can differ between models, so this is
    best interpreted as user-visible website behavior rather than an equal-work
    kernel benchmark.
    """
    batch_size, prompt_tokens = [int(value) for value in input_ids.shape]
    torch.cuda.reset_peak_memory_stats(device)
    synchronize(device)

    streamer = SynchronizedTimingStreamer(device)
    do_sample = temperature > 0.0
    generation_kwargs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "temperature": temperature if do_sample else None,
        "top_p": top_p if do_sample else None,
        "top_k": None,
        "num_beams": 1,
        "use_cache": True,
        "streamer": streamer,
        "return_dict_in_generate": False,
        "output_scores": False,
        "pad_token_id": model.generation_config.pad_token_id,
        "repetition_penalty": repetition_penalty,
        "no_repeat_ngram_size": no_repeat_ngram_size if no_repeat_ngram_size > 0 else None,
    }
    generation_kwargs = {
        key: value for key, value in generation_kwargs.items() if value is not None
    }

    e2e_start = time.perf_counter()
    sequences = model.generate(**generation_kwargs)
    synchronize(device)
    generate_return_seconds = time.perf_counter() - e2e_start

    actual_generated_tokens = int(sequences.shape[1] - prompt_tokens) * batch_size
    if streamer.generated_tokens != actual_generated_tokens:
        raise RuntimeError(
            f"Streamer observed {streamer.generated_tokens} output tokens; "
            f"generate returned {actual_generated_tokens}."
        )
    if actual_generated_tokens < 1 or not streamer.token_step_times:
        raise RuntimeError("Live generation produced no output tokens.")

    first_token_at = streamer.token_step_times[0]
    final_token_at = streamer.token_step_times[-1]
    ttft_seconds = first_token_at - e2e_start
    decode_seconds = (
        final_token_at - first_token_at
        if actual_generated_tokens > 1
        else 0.0
    )
    e2e_seconds = final_token_at - e2e_start

    decode_tokens = batch_size * max(actual_generated_tokens - 1, 0)
    input_tokens = batch_size * prompt_tokens
    peak_allocated = torch.cuda.max_memory_allocated(device)
    peak_reserved = torch.cuda.max_memory_reserved(device)

    prefill_latency_ms = ttft_seconds * 1000.0
    decode_ms = decode_seconds * 1000.0
    decode_ms_per_token = (
        decode_ms / max(actual_generated_tokens - 1, 1)
        if actual_generated_tokens > 1
        else float("nan")
    )
    decode_tok_s = (
        decode_tokens / decode_seconds
        if decode_tokens and decode_seconds > 0
        else float("nan")
    )
    total_latency_ms = e2e_seconds * 1000.0
    e2e_tok_s = actual_generated_tokens / e2e_seconds

    return {
        "prefill_latency_ms": prefill_latency_ms,
        "prefill_tok_s": input_tokens / ttft_seconds,
        "decode_ms_per_token": decode_ms_per_token,
        "decode_tok_s": decode_tok_s,
        "total_latency_ms": total_latency_ms,
        "e2e_tok_s": e2e_tok_s,
        "peak_vram_gb": peak_allocated / (1024**3),
        "ttft_ms": prefill_latency_ms,
        "prefill_tokens_per_s": input_tokens / ttft_seconds,
        "decode_ms": decode_ms,
        "tpot_ms": decode_ms_per_token,
        "decode_tokens_per_s": decode_tok_s,
        "e2e_ms": total_latency_ms,
        "output_tokens_per_s": e2e_tok_s,
        "total_tokens_per_s": (input_tokens + actual_generated_tokens) / e2e_seconds,
        "requests_per_s": batch_size / e2e_seconds,
        "generate_return_latency_ms": generate_return_seconds * 1000.0,
        "post_final_token_overhead_ms": max(
            0.0,
            (generate_return_seconds - e2e_seconds) * 1000.0,
        ),
        "peak_allocated_mb": peak_allocated / (1024**2),
        "peak_extra_allocated_mb": max(0, peak_allocated - baseline_allocated_bytes) / (1024**2),
        "peak_reserved_mb": peak_reserved / (1024**2),
        "output_checksum": streamer.last_token_checksum,
        "actual_output_tokens": actual_generated_tokens,
        "max_new_tokens": max_new_tokens,
    }


@torch.inference_mode()
def run_finance_sentiment_request(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    *,
    label_tokens: Dict[str, Any],
    calibration_input_ids: torch.Tensor,
    calibration_attention_mask: torch.Tensor,
    device: torch.device,
    baseline_allocated_bytes: int,
) -> Dict[str, float]:
    """
    Website-compatible finance sentiment timing.

    This mirrors llama32_3bi_web.py:
      1. Run one prefill forward pass over the prompt with use_cache=True.
      2. Score one-token Good/Bad candidates from calibrated final logits.
      3. Time one cached continuation step using the selected label token.

    The visible task output is still one token, but decode_ms_per_token and
    decode_tok_s are defined from the cached continuation step, matching the
    website's performance summary tiles.
    """
    batch_size, prompt_tokens = [int(value) for value in input_ids.shape]
    torch.cuda.reset_peak_memory_stats(device)
    synchronize(device)

    prefill_start = time.perf_counter()
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=True,
    )
    synchronize(device)
    prefill_seconds = time.perf_counter() - prefill_start

    synchronize(device)
    scoring_start = time.perf_counter()
    calibration_outputs = model(
        input_ids=calibration_input_ids,
        attention_mask=calibration_attention_mask,
        use_cache=False,
    )
    raw_log_probs = torch.log_softmax(outputs.logits[:, -1, :].float(), dim=-1)
    prior_log_probs = torch.log_softmax(calibration_outputs.logits[:, -1, :].float(), dim=-1)

    good_ids = torch.tensor(
        [int(item["token_id"]) for item in label_tokens["good"]],
        device=device,
        dtype=torch.long,
    )
    bad_ids = torch.tensor(
        [int(item["token_id"]) for item in label_tokens["bad"]],
        device=device,
        dtype=torch.long,
    )
    good_raw = raw_log_probs.index_select(1, good_ids)
    bad_raw = raw_log_probs.index_select(1, bad_ids)
    good_prior = prior_log_probs.index_select(1, good_ids)
    bad_prior = prior_log_probs.index_select(1, bad_ids)
    good_scores, good_indices = (good_raw - good_prior).max(dim=1)
    bad_scores, bad_indices = (bad_raw - bad_prior).max(dim=1)
    use_good = good_scores >= bad_scores
    selected_good_ids = good_ids.index_select(0, good_indices)
    selected_bad_ids = bad_ids.index_select(0, bad_indices)
    selected_token_ids = torch.where(use_good, selected_good_ids, selected_bad_ids)
    synchronize(device)
    scoring_seconds = time.perf_counter() - scoring_start

    decode_seconds = 0.0
    if getattr(outputs, "past_key_values", None) is not None:
        decode_input_ids = selected_token_ids.reshape(batch_size, 1).to(
            device=device,
            dtype=input_ids.dtype,
        )
        decode_attention_mask = torch.ones(
            (batch_size, prompt_tokens + 1),
            device=device,
            dtype=attention_mask.dtype,
        )
        synchronize(device)
        decode_start = time.perf_counter()
        model(
            input_ids=decode_input_ids,
            attention_mask=decode_attention_mask,
            past_key_values=outputs.past_key_values,
            use_cache=True,
        )
        synchronize(device)
        decode_seconds = time.perf_counter() - decode_start

    output_tokens = batch_size
    input_tokens = batch_size * prompt_tokens
    total_seconds = prefill_seconds + scoring_seconds + decode_seconds
    peak_allocated = torch.cuda.max_memory_allocated(device)
    peak_reserved = torch.cuda.max_memory_reserved(device)

    prefill_latency_ms = prefill_seconds * 1000.0
    decode_ms = decode_seconds * 1000.0
    total_latency_ms = total_seconds * 1000.0
    decode_tok_s = (
        output_tokens / decode_seconds
        if decode_seconds > 0
        else float("nan")
    )
    decode_ms_per_token = (
        decode_ms / output_tokens
        if output_tokens > 0 and decode_seconds > 0
        else float("nan")
    )
    e2e_tok_s = output_tokens / total_seconds
    selected_token_checksum = int(selected_token_ids.sum().item())
    good_count = int(use_good.sum().item())

    return {
        # Exact names used by llama32_3bi_web.py.
        "prefill_latency_ms": prefill_latency_ms,
        "prefill_tok_s": input_tokens / prefill_seconds,
        "decode_ms_per_token": decode_ms_per_token,
        "decode_tok_s": decode_tok_s,
        "total_latency_ms": total_latency_ms,
        "e2e_tok_s": e2e_tok_s,
        "peak_vram_gb": peak_allocated / (1024**3),
        # Additional standard serving diagnostics and compatibility aliases.
        "ttft_ms": prefill_latency_ms,
        "prefill_tokens_per_s": input_tokens / prefill_seconds,
        "decode_ms": decode_ms,
        "tpot_ms": decode_ms_per_token,
        "decode_tokens_per_s": decode_tok_s,
        "e2e_ms": total_latency_ms,
        "output_tokens_per_s": e2e_tok_s,
        "total_tokens_per_s": (input_tokens + output_tokens) / total_seconds,
        "requests_per_s": batch_size / total_seconds,
        "generate_return_latency_ms": total_latency_ms,
        "post_final_token_overhead_ms": 0.0,
        "peak_allocated_mb": peak_allocated / (1024**2),
        "peak_extra_allocated_mb": max(0, peak_allocated - baseline_allocated_bytes) / (1024**2),
        "peak_reserved_mb": peak_reserved / (1024**2),
        "output_checksum": selected_token_checksum,
        "finance_good_labels": good_count,
        "finance_bad_labels": batch_size - good_count,
        "label_scoring_latency_ms": scoring_seconds * 1000.0,
        "sentiment_scoring": "calibrated_next_token",
    }


def aggregate_samples(samples: List[Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    return {
        metric: summarize_values([sample[metric] for sample in samples])
        for metric in ALL_METRICS
    }


def benchmark_scenario(
    model: torch.nn.Module,
    prompt_ids_cpu: Sequence[torch.Tensor],
    *,
    task_name: str,
    batch_size: int,
    new_tokens: int,
    repetition_penalty: float,
    no_repeat_ngram_size: int,
    generation_mode: str,
    live_temperature: float,
    live_top_p: float,
    warmup: int,
    repeats: int,
    device: torch.device,
    baseline_allocated_bytes: int,
    finance_label_tokens: Optional[Dict[str, Any]],
    finance_calibration_prompt_ids: Optional[torch.Tensor],
    include_raw_samples: bool,
) -> Dict[str, Any]:
    if not prompt_ids_cpu:
        raise ValueError("At least one prompt variant is required.")
    prompt_tokens = int(prompt_ids_cpu[0].numel())
    if any(int(prompt.numel()) != prompt_tokens for prompt in prompt_ids_cpu):
        raise ValueError("Every prompt variant in one scenario must have the same token length.")

    prepared_batches: List[Tuple[torch.Tensor, torch.Tensor]] = []
    for offset in range(len(prompt_ids_cpu)):
        rows = [
            prompt_ids_cpu[(offset + batch_index) % len(prompt_ids_cpu)]
            for batch_index in range(batch_size)
        ]
        input_ids = torch.stack(rows, dim=0).to(device)
        attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=device)
        prepared_batches.append((input_ids, attention_mask))

    calibration_batch: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    if task_name == "finance_sentiment":
        if finance_calibration_prompt_ids is None:
            raise ValueError("finance_calibration_prompt_ids are required for finance_sentiment.")
        calibration_input_ids = finance_calibration_prompt_ids.reshape(1, -1).to(device)
        calibration_attention_mask = torch.ones_like(
            calibration_input_ids,
            dtype=torch.long,
            device=device,
        )
        calibration_batch = (calibration_input_ids, calibration_attention_mask)

    try:
        for request_index in range(warmup):
            input_ids, attention_mask = prepared_batches[request_index % len(prepared_batches)]
            if task_name == "finance_sentiment":
                if finance_label_tokens is None:
                    raise ValueError("finance_label_tokens are required for finance_sentiment.")
                if calibration_batch is None:
                    raise ValueError("finance calibration batch is required for finance_sentiment.")
                run_finance_sentiment_request(
                    model,
                    input_ids,
                    attention_mask,
                    label_tokens=finance_label_tokens,
                    calibration_input_ids=calibration_batch[0],
                    calibration_attention_mask=calibration_batch[1],
                    device=device,
                    baseline_allocated_bytes=baseline_allocated_bytes,
                )
            else:
                if generation_mode == "free":
                    run_live_generate_request(
                        model,
                        input_ids,
                        attention_mask,
                        max_new_tokens=new_tokens,
                        temperature=live_temperature,
                        top_p=live_top_p,
                        repetition_penalty=repetition_penalty,
                        no_repeat_ngram_size=no_repeat_ngram_size,
                        device=device,
                        baseline_allocated_bytes=baseline_allocated_bytes,
                    )
                else:
                    run_fixed_length_request(
                        model,
                        input_ids,
                        attention_mask,
                        new_tokens=new_tokens,
                        repetition_penalty=repetition_penalty,
                        no_repeat_ngram_size=no_repeat_ngram_size,
                        device=device,
                        baseline_allocated_bytes=baseline_allocated_bytes,
                    )

        samples = []
        for request_index in range(repeats):
            input_ids, attention_mask = prepared_batches[
                (warmup + request_index) % len(prepared_batches)
            ]
            if task_name == "finance_sentiment":
                if finance_label_tokens is None:
                    raise ValueError("finance_label_tokens are required for finance_sentiment.")
                if calibration_batch is None:
                    raise ValueError("finance calibration batch is required for finance_sentiment.")
                sample = run_finance_sentiment_request(
                    model,
                    input_ids,
                    attention_mask,
                    label_tokens=finance_label_tokens,
                    calibration_input_ids=calibration_batch[0],
                    calibration_attention_mask=calibration_batch[1],
                    device=device,
                    baseline_allocated_bytes=baseline_allocated_bytes,
                )
            else:
                if generation_mode == "free":
                    sample = run_live_generate_request(
                        model,
                        input_ids,
                        attention_mask,
                        max_new_tokens=new_tokens,
                        temperature=live_temperature,
                        top_p=live_top_p,
                        repetition_penalty=repetition_penalty,
                        no_repeat_ngram_size=no_repeat_ngram_size,
                        device=device,
                        baseline_allocated_bytes=baseline_allocated_bytes,
                    )
                else:
                    sample = run_fixed_length_request(
                        model,
                        input_ids,
                        attention_mask,
                        new_tokens=new_tokens,
                        repetition_penalty=repetition_penalty,
                        no_repeat_ngram_size=no_repeat_ngram_size,
                        device=device,
                        baseline_allocated_bytes=baseline_allocated_bytes,
                    )
            sample["prompt_variant_index"] = int(
                (warmup + request_index) % len(prepared_batches)
            )
            sample["metric_implementation"] = (
                "website_calibrated_finance_good_bad_scorer"
                if task_name == "finance_sentiment"
                else "free_generate"
                if generation_mode == "free"
                else "fixed_length_generate"
            )
            samples.append(sample)
        result: Dict[str, Any] = {
            "status": "ok",
            "task": task_name,
            "batch_size": batch_size,
            "prompt_tokens_per_request": prompt_tokens,
            "new_tokens_per_request": new_tokens,
            "prompt_variant_count": len(prompt_ids_cpu),
            "fixed_workload": {
                "prompt_tokens": prompt_tokens,
                "output_tokens": new_tokens,
            },
            "metric_implementation": (
                "llama32_3bi_web.py finance scorer: calibrated prefill logits + one cached label-token decode"
                if task_name == "finance_sentiment"
                else "free live-response streaming generate"
                if generation_mode == "free"
                else "standard fixed-length greedy generation"
            ),
            "warmup_requests": warmup,
            "measured_repeats": repeats,
            "stats": aggregate_samples(samples),
        }
        if include_raw_samples:
            result["samples"] = samples
        return result
    except torch.cuda.OutOfMemoryError as exc:
        return {
            "status": "oom",
            "task": task_name,
            "batch_size": batch_size,
            "prompt_tokens_per_request": prompt_tokens,
            "new_tokens_per_request": new_tokens,
            "error": str(exc),
        }
    finally:
        del prepared_batches
        cleanup_cuda(device)


def benchmark_model(
    label: str,
    model_name: str,
    source: str,
    prompt_ids: Dict[str, Dict[int, List[torch.Tensor]]],
    task_workloads: Dict[str, Dict[str, Any]],
    finance_label_tokens: Optional[Dict[str, Any]],
    finance_calibration_prompt_ids: Optional[torch.Tensor],
    args,
    dtype: torch.dtype,
    device: torch.device,
    *,
    round_index: int,
) -> Dict[str, Any]:
    telemetry_before = nvidia_smi_snapshot()
    print(f"\n=== Round {round_index}: loading {label}: {model_name} ===")
    model, load_seconds = load_model(
        source,
        dtype=dtype,
        device=device,
        cache_dir=args.cache_dir,
        local_files_only=args.local_files_only,
        attn_implementation=args.attn_implementation,
    )
    if model.generation_config.pad_token_id is None:
        eos_token_id = model.generation_config.eos_token_id
        if isinstance(eos_token_id, (list, tuple)):
            eos_token_id = eos_token_id[0] if eos_token_id else None
        if eos_token_id is None:
            raise ValueError(f"{label} model has neither pad_token_id nor eos_token_id.")
        model.generation_config.pad_token_id = int(eos_token_id)
        model.config.pad_token_id = int(eos_token_id)
    metadata = model_parameter_metadata(model)
    baseline_allocated_bytes = torch.cuda.memory_allocated(device)
    metadata.update(
        {
            "label": label,
            "requested_model": model_name,
            "resolved_source": source,
            "load_seconds": load_seconds,
            "gpu_memory_after_load_mb": baseline_allocated_bytes / (1024**2),
            "memory_footprint_gb": baseline_allocated_bytes / (1024**3),
            "dtype": str(dtype).replace("torch.", ""),
            "round_index": round_index,
            "telemetry_before_load": telemetry_before,
        }
    )

    print(
        f"Loaded in {load_seconds:.2f}s | params={metadata['parameter_count'] / 1e9:.3f}B "
        f"| GPU allocated={metadata['gpu_memory_after_load_mb']:.1f} MiB"
    )

    scenarios: List[Dict[str, Any]] = []
    for task_name, workload in task_workloads.items():
        print(f"\n--- Task: {task_name} ---")
        for batch_size in args.batch_sizes:
            for prompt_length in workload["prompt_lengths"]:
                new_tokens = int(workload["output_tokens"])
                print(
                    f"[{label}] task={task_name} batch={batch_size} "
                    f"prompt={prompt_length} output={new_tokens} ...",
                    flush=True,
                )
                scenario = benchmark_scenario(
                    model,
                    prompt_ids[task_name][prompt_length],
                    task_name=task_name,
                    batch_size=batch_size,
                    new_tokens=new_tokens,
                    repetition_penalty=args.repetition_penalty,
                    no_repeat_ngram_size=args.no_repeat_ngram_size,
                    generation_mode=args.generation_mode,
                    live_temperature=args.live_temperature,
                    live_top_p=args.live_top_p,
                    warmup=args.warmup,
                    repeats=args.repeats,
                    device=device,
                    baseline_allocated_bytes=baseline_allocated_bytes,
                    finance_label_tokens=finance_label_tokens,
                    finance_calibration_prompt_ids=finance_calibration_prompt_ids,
                    # Round aggregation needs raw samples internally. They are
                    # removed from the final report when --no-raw-samples is set.
                    include_raw_samples=True,
                )
                for sample in scenario.get("samples", []):
                    sample["round_index"] = round_index
                scenarios.append(scenario)
                if scenario["status"] == "ok":
                    stats = scenario["stats"]
                    decode_ms_p50 = stats["decode_ms_per_token"]["p50"]
                    decode_tok_s_p50 = stats["decode_tok_s"]["p50"]
                    decode_text = (
                        f"{decode_ms_p50:.2f} ms/token "
                        f"({decode_tok_s_p50:.1f} tok/s)"
                        if math.isfinite(decode_ms_p50) and math.isfinite(decode_tok_s_p50)
                        else "N/A"
                    )
                    print(
                        f"  prefill p50={stats['prefill_latency_ms']['p50']:.2f} ms "
                        f"({stats['prefill_tok_s']['p50']:.1f} tok/s) | "
                        f"decode p50={decode_text} | "
                        f"E2E p50={stats['total_latency_ms']['p50']:.2f} ms "
                        f"({stats['e2e_tok_s']['p50']:.1f} tok/s) | "
                        f"peak={stats['peak_vram_gb']['p50']:.2f} GB"
                    )
                else:
                    print("  OOM")

    metadata["telemetry_after_benchmark"] = nvidia_smi_snapshot()
    del model
    cleanup_cuda(device)
    return {"metadata": metadata, "scenarios": scenarios}


def scenario_key(row: Dict[str, Any]) -> Tuple[str, int, int, int]:
    return (
        str(row["task"]),
        int(row["batch_size"]),
        int(row["prompt_tokens_per_request"]),
        int(row["new_tokens_per_request"]),
    )


def merge_model_rounds(
    label: str,
    round_results: Sequence[Dict[str, Any]],
    *,
    include_raw_samples: bool,
) -> Dict[str, Any]:
    if not round_results:
        raise ValueError(f"No completed benchmark rounds for {label}.")

    first_metadata = dict(round_results[0]["metadata"])
    load_seconds = [float(result["metadata"]["load_seconds"]) for result in round_results]
    footprints = [
        float(result["metadata"]["memory_footprint_gb"])
        for result in round_results
    ]
    allocations = [
        float(result["metadata"]["gpu_memory_after_load_mb"])
        for result in round_results
    ]
    first_metadata.update(
        {
            "round_count": len(round_results),
            "round_indices": [
                int(result["metadata"]["round_index"])
                for result in round_results
            ],
            "load_seconds_by_round": load_seconds,
            "load_seconds_stats": summarize_values(load_seconds),
            "memory_footprint_gb_by_round": footprints,
            "memory_footprint_gb": statistics.median(footprints),
            "gpu_memory_after_load_mb_by_round": allocations,
            "gpu_memory_after_load_mb": statistics.median(allocations),
            "round_telemetry": [
                {
                    "round_index": result["metadata"]["round_index"],
                    "before_load": result["metadata"].get("telemetry_before_load"),
                    "after_benchmark": result["metadata"].get("telemetry_after_benchmark"),
                }
                for result in round_results
            ],
        }
    )

    rows_by_key: Dict[Tuple[str, int, int, int], List[Dict[str, Any]]] = {}
    for result in round_results:
        for row in result["scenarios"]:
            rows_by_key.setdefault(scenario_key(row), []).append(row)

    merged_scenarios: List[Dict[str, Any]] = []
    for key in sorted(rows_by_key):
        rows = rows_by_key[key]
        successful = [row for row in rows if row.get("status") == "ok"]
        if len(successful) != len(round_results):
            merged_scenarios.append(
                {
                    "status": "partial" if successful else "failed",
                    "task": key[0],
                    "batch_size": key[1],
                    "prompt_tokens_per_request": key[2],
                    "new_tokens_per_request": key[3],
                    "successful_rounds": len(successful),
                    "required_rounds": len(round_results),
                    "round_statuses": [row.get("status") for row in rows],
                }
            )
            continue

        samples = [
            sample
            for row in successful
            for sample in row.get("samples", [])
        ]
        if not samples:
            raise RuntimeError(
                "Internal raw samples are required to aggregate multiple rounds."
            )
        merged: Dict[str, Any] = {
            "status": "ok",
            "task": key[0],
            "batch_size": key[1],
            "prompt_tokens_per_request": key[2],
            "new_tokens_per_request": key[3],
            "prompt_variant_count": successful[0].get("prompt_variant_count", 1),
            "fixed_workload": {
                "prompt_tokens": key[2],
                "output_tokens": key[3],
            },
            "warmup_requests_total": sum(
                int(row.get("warmup_requests", 0))
                for row in successful
            ),
            "measured_repeats_total": len(samples),
            "round_count": len(successful),
            "stats": aggregate_samples(samples),
        }
        if include_raw_samples:
            merged["samples"] = samples
        merged_scenarios.append(merged)

    return {"metadata": first_metadata, "scenarios": merged_scenarios}


def compare_models(
    dense: Dict[str, Any],
    pruned: Dict[str, Any],
    *,
    bootstrap_samples: int,
    seed: int,
) -> List[Dict[str, Any]]:
    dense_by_key = {
        scenario_key(row): row
        for row in dense["scenarios"]
        if row.get("status") == "ok"
    }
    pruned_by_key = {
        scenario_key(row): row
        for row in pruned["scenarios"]
        if row.get("status") == "ok"
    }

    comparisons: List[Dict[str, Any]] = []
    for key in sorted(set(dense_by_key) & set(pruned_by_key)):
        dense_row = dense_by_key[key]
        pruned_row = pruned_by_key[key]
        dense_stats = dense_row["stats"]
        pruned_stats = pruned_row["stats"]
        dense_p50 = {metric: dense_stats[metric]["p50"] for metric in ALL_METRICS}
        pruned_p50 = {metric: pruned_stats[metric]["p50"] for metric in ALL_METRICS}
        dense_p95 = {metric: dense_stats[metric]["p95"] for metric in ALL_METRICS}
        pruned_p95 = {metric: pruned_stats[metric]["p95"] for metric in ALL_METRICS}

        # These formulas exactly match the three Performance Summary tiles in
        # llama32_3bi_web.py.
        performance_summary = {
            "prefill_speedup": safe_ratio(
                pruned_p50["prefill_tok_s"],
                dense_p50["prefill_tok_s"],
            ),
            "decode_speedup": safe_ratio(
                pruned_p50["decode_tok_s"],
                dense_p50["decode_tok_s"],
            ),
            "end_to_end_speedup": safe_ratio(
                pruned_p50["e2e_tok_s"],
                dense_p50["e2e_tok_s"],
            ),
        }
        confidence_intervals = {}
        for offset, (summary_name, metric_name) in enumerate(
            (
                ("prefill_speedup", "prefill_tok_s"),
                ("decode_speedup", "decode_tok_s"),
                ("end_to_end_speedup", "e2e_tok_s"),
            )
        ):
            confidence_intervals[summary_name] = bootstrap_ratio_confidence_interval(
                [sample[metric_name] for sample in dense_row.get("samples", [])],
                [sample[metric_name] for sample in pruned_row.get("samples", [])],
                pruned_over_dense=True,
                samples=bootstrap_samples,
                seed=(
                    seed
                    + offset
                    + sum(ord(char) for char in key[0]) * 1_000_000
                    + key[1] * 10_000
                    + key[2] * 10
                    + key[3]
                ),
            )

        speedups: Dict[str, float] = dict(performance_summary)
        for metric in BENCHMARK_LATENCY_METRICS + DIAGNOSTIC_LATENCY_METRICS:
            speedups[f"{metric}_speedup"] = safe_ratio(dense_p50[metric], pruned_p50[metric])
        for metric in BENCHMARK_THROUGHPUT_METRICS + DIAGNOSTIC_THROUGHPUT_METRICS:
            speedups[f"{metric}_speedup"] = safe_ratio(pruned_p50[metric], dense_p50[metric])

        speedups.update(
            {
                "prefill_latency_reduction_percent": reduction_percent(
                    dense_p50["prefill_latency_ms"],
                    pruned_p50["prefill_latency_ms"],
                ),
                "decode_latency_reduction_percent": reduction_percent(
                    dense_p50["decode_ms_per_token"],
                    pruned_p50["decode_ms_per_token"],
                ),
                "end_to_end_latency_reduction_percent": reduction_percent(
                    dense_p50["total_latency_ms"],
                    pruned_p50["total_latency_ms"],
                ),
                "peak_vram_ratio_dense_over_pruned": safe_ratio(
                    dense_p50["peak_vram_gb"],
                    pruned_p50["peak_vram_gb"],
                ),
                "peak_vram_reduction_percent": reduction_percent(
                    dense_p50["peak_vram_gb"],
                    pruned_p50["peak_vram_gb"],
                ),
            }
        )

        comparisons.append(
            {
                "task": key[0],
                "batch_size": key[1],
                "prompt_tokens_per_request": key[2],
                "new_tokens_per_request": key[3],
                "benchmark_metrics_p50": {
                    "dense": {
                        "prompt_tokens": key[2],
                        "output_tokens": key[3],
                        **{metric: dense_p50[metric] for metric in BENCHMARK_METRICS},
                    },
                    "pruned": {
                        "prompt_tokens": key[2],
                        "output_tokens": key[3],
                        **{metric: pruned_p50[metric] for metric in BENCHMARK_METRICS},
                    },
                },
                "benchmark_metrics_p95": {
                    "dense": {
                        "prompt_tokens": key[2],
                        "output_tokens": key[3],
                        **{metric: dense_p95[metric] for metric in BENCHMARK_METRICS},
                    },
                    "pruned": {
                        "prompt_tokens": key[2],
                        "output_tokens": key[3],
                        **{metric: pruned_p95[metric] for metric in BENCHMARK_METRICS},
                    },
                },
                "performance_summary": performance_summary,
                "performance_summary_95pct_ci": confidence_intervals,
                "dense_p50": dense_p50,
                "pruned_p50": pruned_p50,
                "dense_p95": dense_p95,
                "pruned_p95": pruned_p95,
                "speedups": speedups,
            }
        )
    return comparisons


def overall_speedups(comparisons: List[Dict[str, Any]]) -> Dict[str, float]:
    primary_names = ("prefill_speedup", "decode_speedup", "end_to_end_speedup")
    result = {
        f"geomean_{name}": geometric_mean(
            comparison["performance_summary"][name]
            for comparison in comparisons
        )
        for name in primary_names
    }
    result["mean_peak_vram_reduction_percent"] = statistics.fmean(
        comparison["speedups"]["peak_vram_reduction_percent"]
        for comparison in comparisons
    ) if comparisons else float("nan")
    return result


def speedups_by_task(comparisons: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for comparison in comparisons:
        grouped.setdefault(comparison["task"], []).append(comparison)
    return {
        task: overall_speedups(rows)
        for task, rows in sorted(grouped.items())
    }


def speedups_by_report_group(comparisons: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    groups = {
        "primary_website_tasks": {
            "description": "Finance sentiment plus summarization workloads used by the app.",
            "tasks": ("finance_sentiment", SUMMARIZATION_TASK_NAME),
        },
        "secondary_general_generation": {
            "description": "Broad prompt-following/general-generation workload.",
            "tasks": ("general_generation",),
        },
        "all_requested_workloads": {
            "description": "Geometric mean across every successful requested workload.",
            "tasks": tuple(sorted({comparison["task"] for comparison in comparisons})),
        },
    }
    result: Dict[str, Dict[str, Any]] = {}
    for group_name, group in groups.items():
        task_set = set(group["tasks"])
        rows = [comparison for comparison in comparisons if comparison["task"] in task_set]
        if not rows:
            continue
        result[group_name] = {
            "description": group["description"],
            "tasks": sorted({row["task"] for row in rows}),
            **overall_speedups(rows),
        }
    return result


def summarize_round_stability(
    round_level_comparisons: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    values_by_scenario: Dict[Tuple[str, int, int, int], Dict[str, List[float]]] = {}
    for round_result in round_level_comparisons:
        for comparison in round_result["comparisons"]:
            key = (
                comparison["task"],
                comparison["batch_size"],
                comparison["prompt_tokens_per_request"],
                comparison["new_tokens_per_request"],
            )
            metric_values = values_by_scenario.setdefault(
                key,
                {
                    "prefill_speedup": [],
                    "decode_speedup": [],
                    "end_to_end_speedup": [],
                },
            )
            for metric in metric_values:
                metric_values[metric].append(
                    comparison["performance_summary"][metric]
                )
    return [
        {
            "task": key[0],
            "batch_size": key[1],
            "prompt_tokens_per_request": key[2],
            "new_tokens_per_request": key[3],
            "round_speedup_stats": {
                metric: summarize_values(values)
                for metric, values in metric_values.items()
            },
        }
        for key, metric_values in sorted(values_by_scenario.items())
    ]


def print_comparison_table(comparisons: List[Dict[str, Any]]) -> None:
    if not comparisons:
        print("\nNo matching successful scenarios were available for comparison.")
        return
    print("\n=== Original vs Pruned Inference Speedup (P50) ===")
    print(
        f"{'Task':>18} {'B':>3} {'Prompt':>7} {'Output':>7} | "
        f"{'Prefill speedup [95% CI]':>27} "
        f"{'Decode speedup [95% CI]':>27} "
        f"{'E2E speedup [95% CI]':>27} {'VRAM saved':>11}"
    )
    print("-" * 138)
    for row in comparisons:
        summary = row["performance_summary"]
        intervals = row["performance_summary_95pct_ci"]

        def speedup_with_ci(name: str) -> str:
            interval = intervals[name]
            if not math.isfinite(summary[name]):
                return "N/A"
            return (
                f"{summary[name]:.3f}x "
                f"[{interval['low']:.3f}, {interval['high']:.3f}]"
            )

        print(
            f"{row['task']:>18} {row['batch_size']:3d} "
            f"{row['prompt_tokens_per_request']:7d} "
            f"{row['new_tokens_per_request']:7d} | "
            f"{speedup_with_ci('prefill_speedup'):>27} "
            f"{speedup_with_ci('decode_speedup'):>27} "
            f"{speedup_with_ci('end_to_end_speedup'):>27} "
            f"{row['speedups']['peak_vram_reduction_percent']:10.2f}%"
        )

    print("\n=== Detailed Inference Metrics: Original -> Pruned (P50) ===")
    for row in comparisons:
        dense = row["benchmark_metrics_p50"]["dense"]
        pruned = row["benchmark_metrics_p50"]["pruned"]
        print(
            f"\nTask {row['task']}, batch {row['batch_size']}, "
            f"prompt {row['prompt_tokens_per_request']} tokens, "
            f"output {row['new_tokens_per_request']} tokens"
        )
        print(f"{'Metric':<28} {'Original':>14} {'Pruned':>14} {'Relative change':>18}")
        print("-" * 78)
        for metric in BENCHMARK_METRICS:
            original = dense[metric]
            new = pruned[metric]
            if not math.isfinite(original) or not math.isfinite(new):
                print(
                    f"{BENCHMARK_METRIC_LABELS[metric]:<28} "
                    f"{'N/A':>14} {'N/A':>14} {'not defined':>18}"
                )
                continue
            if metric in BENCHMARK_THROUGHPUT_METRICS:
                relative = safe_ratio(new, original)
                change = f"{relative:.3f}x throughput"
            else:
                saved = reduction_percent(original, new)
                change = f"{saved:+.2f}% reduction"
            print(
                f"{BENCHMARK_METRIC_LABELS[metric]:<28} "
                f"{original:14.3f} {new:14.3f} {change:>18}"
            )
    print("\nP95 values and raw repeated measurements are saved in the JSON output.")


def write_json(path: str, payload: Dict[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, allow_nan=True)


def write_csv(path: str, comparisons: List[Dict[str, Any]]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, Any]] = []
    for comparison in comparisons:
        row: Dict[str, Any] = {
            "task": comparison["task"],
            "batch_size": comparison["batch_size"],
            "prompt_tokens": comparison["prompt_tokens_per_request"],
            "output_tokens": comparison["new_tokens_per_request"],
            "prefill_speedup": comparison["performance_summary"]["prefill_speedup"],
            "decode_speedup": comparison["performance_summary"]["decode_speedup"],
            "end_to_end_speedup": comparison["performance_summary"]["end_to_end_speedup"],
        }
        for speedup_name, interval in comparison["performance_summary_95pct_ci"].items():
            row[f"{speedup_name}_ci95_low"] = interval["low"]
            row[f"{speedup_name}_ci95_high"] = interval["high"]
            row[f"{speedup_name}_pruned_faster_95pct"] = interval[
                "pruned_faster_with_95pct_confidence"
            ]
        for percentile_name in ("p50", "p95"):
            for model_label in ("dense", "pruned"):
                values = comparison[f"{model_label}_{percentile_name}"]
                for metric, value in values.items():
                    row[f"{model_label}_{percentile_name}_{metric}"] = value
        row.update(comparison["speedups"])
        rows.append(row)

    if not rows:
        return
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def remove_raw_samples(models: Dict[str, Any]) -> None:
    for model_result in models.values():
        for scenario in model_result.get("scenarios", []):
            scenario.pop("samples", None)


def environment_metadata(device: Optional[torch.device]) -> Dict[str, Any]:
    metadata: Dict[str, Any] = {
        "python": platform.python_version(),
        "pytorch": torch.__version__,
        "transformers": transformers.__version__,
        "platform": platform.platform(),
        "cuda_runtime": torch.version.cuda,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "cudnn_version": torch.backends.cudnn.version(),
        "matmul_allow_tf32": bool(
            getattr(torch.backends.cuda.matmul, "allow_tf32", False)
        ),
        "cudnn_allow_tf32": bool(
            getattr(torch.backends.cudnn, "allow_tf32", False)
        ),
        "nvidia_smi": nvidia_smi_snapshot(),
    }
    if device is not None and torch.cuda.is_available():
        properties = torch.cuda.get_device_properties(device)
        metadata["gpu"] = {
            "name": properties.name,
            "compute_capability": f"{properties.major}.{properties.minor}",
            "total_memory_mb": properties.total_memory / (1024**2),
            "device": str(device),
        }
    return metadata


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark task-specific LLM inference speedup for dense vs pruned Llama 3.2 3B."
    )
    parser.add_argument("--dense-model", default=DEFAULT_DENSE_MODEL)
    parser.add_argument("--pruned-model", default=DEFAULT_PRUNED_MODEL)
    parser.add_argument(
        "--tokenizer-model",
        default=None,
        help="Tokenizer/chat template source. Defaults to --dense-model.",
    )
    parser.add_argument(
        "--tasks",
        default="general_generation,finance_sentiment,summarization",
        help=(
            "Comma-separated task suite: finance_sentiment,summarization,"
            "general_generation. Default is general prompt, finance sentiment "
            "and summarization."
        ),
    )
    parser.add_argument(
        "--prompt-source",
        choices=["demo_files", "synthetic"],
        default="synthetic",
        help=(
            "demo_files uses the same benchmark article files pasted into the website "
            "for finance_sentiment and summarization. synthetic uses exact token-length "
            "prompts for every task."
        ),
    )
    parser.add_argument(
        "--finance-prompt-lengths",
        default="4096",
        help="Exact input lengths for one-token financial sentiment inference.",
    )
    parser.add_argument(
        "--summarization-prompt-lengths",
        default="8192",
        help="Exact input lengths for summarization inference.",
    )
    parser.add_argument(
        "--general-prompt-lengths",
        default="8192",
        help="Exact input lengths for broad/general generation inference.",
    )
    parser.add_argument(
        "--batch-sizes",
        default="1",
        help=(
            "Static in-process batch sizes. Batch 1 models an interactive request. "
            "This is not an HTTP concurrency setting."
        ),
    )
    parser.add_argument(
        "--prompts-per-length",
        type=int,
        default=3,
        help="Distinct realistic prompt variants cycled at every exact input length.",
    )
    parser.add_argument("--finance-output-tokens", type=int, default=1)
    parser.add_argument("--summarization-output-tokens", type=int, default=256)
    parser.add_argument("--general-output-tokens", type=int, default=20)
    parser.add_argument(
        "--repetition-penalty",
        type=float,
        default=1.10,
        help="Generation setting used consistently by both models.",
    )
    parser.add_argument(
        "--no-repeat-ngram-size",
        type=int,
        default=4,
        help="Generation setting used consistently by both models.",
    )
    parser.add_argument(
        "--generation-mode",
        choices=["fixed", "free"],
        default="fixed",
        help=(
            "For non-finance tasks, fixed matches the website benchmark output "
            "budgets; free allows EOS and sampling for qualitative live-response timing."
        ),
    )
    parser.add_argument(
        "--live-temperature",
        type=float,
        default=0.6,
        help="Temperature used when --generation-mode free.",
    )
    parser.add_argument(
        "--live-top-p",
        type=float,
        default=0.9,
        help="Top-p used when --generation-mode free.",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=3,
        help="Unmeasured requests per scenario before timing.",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=10,
        help="Measured requests per scenario in each round.",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=2,
        help=(
            "Benchmark rounds. Even rounds reverse model order to reduce thermal/order bias; "
            "2 rounds x 10 repeats gives 20 measurements per scenario."
        ),
    )
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=2000,
        help="Bootstrap resamples used for 95%% confidence intervals on headline speedups.",
    )
    parser.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument(
        "--attn-implementation",
        choices=["sdpa", "eager", "flash_attention_2"],
        default="sdpa",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument(
        "--model-order",
        default="dense,pruned",
        help="First-round load order. Later rounds alternate automatically.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-json", default=DEFAULT_OUTPUT_JSON)
    parser.add_argument(
        "--output-csv",
        default=None,
        help="CSV output path. Only used with --write-csv.",
    )
    parser.add_argument(
        "--write-csv",
        dest="no_output_csv",
        action="store_false",
        help="Also write a CSV file. Default is JSON only.",
    )
    parser.add_argument(
        "--no-output-csv",
        dest="no_output_csv",
        action="store_true",
        help="Write only the JSON output file.",
    )
    parser.add_argument("--no-raw-samples", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate model configs and exact prompt construction without loading models on GPU.",
    )
    parser.set_defaults(no_output_csv=True)
    args = parser.parse_args()

    args.tasks = [item.strip() for item in args.tasks.split(",") if item.strip()]
    allowed_tasks = {"finance_sentiment", "summarization", "general_generation"}
    if not args.tasks or len(set(args.tasks)) != len(args.tasks) or not set(args.tasks) <= allowed_tasks:
        parser.error(
            "--tasks must contain finance_sentiment, summarization and/or "
            "general_generation without duplicates."
        )
    args.finance_prompt_lengths = parse_int_list(
        args.finance_prompt_lengths,
        name="finance-prompt-lengths",
    )
    args.summarization_prompt_lengths = parse_int_list(
        args.summarization_prompt_lengths,
        name="summarization-prompt-lengths",
    )
    args.general_prompt_lengths = parse_int_list(
        args.general_prompt_lengths,
        name="general-prompt-lengths",
    )
    args.batch_sizes = parse_int_list(args.batch_sizes, name="batch-sizes")
    args.model_order = [item.strip() for item in args.model_order.split(",") if item.strip()]
    if sorted(args.model_order) != ["dense", "pruned"]:
        parser.error("--model-order must contain dense and pruned exactly once.")
    if args.finance_output_tokens != 1:
        parser.error("--finance-output-tokens must be exactly 1 for label scoring.")
    if args.summarization_output_tokens < 2:
        parser.error("--summarization-output-tokens must be >= 2.")
    if args.general_output_tokens < 2:
        parser.error("--general-output-tokens must be >= 2.")
    if args.repetition_penalty <= 0:
        parser.error("--repetition-penalty must be > 0.")
    if args.no_repeat_ngram_size < 0:
        parser.error("--no-repeat-ngram-size must be >= 0.")
    if args.live_temperature < 0:
        parser.error("--live-temperature must be >= 0.")
    if not 0 < args.live_top_p <= 1:
        parser.error("--live-top-p must be in (0, 1].")
    max_prompt_variants = min(
        len(FINANCE_SENTIMENT_PROMPT_VARIANTS),
        len(SUMMARIZATION_PROMPT_VARIANTS),
        len(GENERAL_GENERATION_PROMPT_VARIANTS),
    )
    if args.prompts_per_length < 1 or args.prompts_per_length > max_prompt_variants:
        parser.error(
            f"--prompts-per-length must be between 1 and {max_prompt_variants}."
        )
    if args.warmup < 0:
        parser.error("--warmup must be >= 0.")
    if args.repeats < 2:
        parser.error("--repeats must be >= 2 so variability and percentiles are meaningful.")
    if args.rounds < 1:
        parser.error("--rounds must be >= 1.")
    if args.bootstrap_samples < 100:
        parser.error("--bootstrap-samples must be >= 100.")
    if args.no_output_csv:
        args.output_csv = None
    elif args.output_csv is None:
        args.output_csv = str(Path(args.output_json).with_suffix(".csv"))
    return args


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    dense_source = resolve_model_source(args.dense_model, args.cache_dir)
    pruned_source = resolve_model_source(args.pruned_model, args.cache_dir)
    tokenizer_name = args.tokenizer_model or args.dense_model
    tokenizer_source = resolve_model_source(tokenizer_name, args.cache_dir)

    tokenizer = load_tokenizer(tokenizer_source, args.cache_dir, args.local_files_only)
    finance_label_tokens = (
        finance_label_token_metadata(tokenizer)
        if "finance_sentiment" in args.tasks
        else None
    )
    finance_calibration_prompt_ids = (
        build_website_task_prompt_ids(
            tokenizer,
            FINANCE_CALIBRATION_ARTICLE,
            "finance_sentiment",
        )
        if "finance_sentiment" in args.tasks
        else None
    )
    task_workloads: Dict[str, Dict[str, Any]] = {}
    task_prompt_variants = {
        "finance_sentiment": FINANCE_SENTIMENT_PROMPT_VARIANTS,
        SUMMARIZATION_TASK_NAME: SUMMARIZATION_PROMPT_VARIANTS,
        "general_generation": GENERAL_GENERATION_PROMPT_VARIANTS,
    }
    if "finance_sentiment" in args.tasks:
        task_workloads["finance_sentiment"] = {
            "prompt_lengths": args.finance_prompt_lengths,
            "output_tokens": args.finance_output_tokens,
        }
    if "summarization" in args.tasks:
        task_workloads[SUMMARIZATION_TASK_NAME] = {
            "prompt_lengths": args.summarization_prompt_lengths,
            "output_tokens": args.summarization_output_tokens,
        }
    if "general_generation" in args.tasks:
        task_workloads["general_generation"] = {
            "prompt_lengths": args.general_prompt_lengths,
            "output_tokens": args.general_output_tokens,
        }

    selected_prompt_variants = {}
    for task_name in task_workloads:
        if (
            args.prompt_source == "demo_files"
            and task_name in {"finance_sentiment", SUMMARIZATION_TASK_NAME}
        ):
            selected_prompt_variants[task_name] = demo_prompt_variants_for_task(task_name)
        else:
            selected_prompt_variants[task_name] = task_prompt_variants[task_name][: args.prompts_per_length]
    prompts, prompt_records = build_prompt_collections(
        tokenizer,
        task_workloads,
        selected_prompt_variants,
        prompt_source=args.prompt_source,
    )

    dense_config = AutoConfig.from_pretrained(
        dense_source,
        cache_dir=args.cache_dir,
        local_files_only=args.local_files_only or Path(dense_source).exists(),
    )
    pruned_config = AutoConfig.from_pretrained(
        pruned_source,
        cache_dir=args.cache_dir,
        local_files_only=args.local_files_only or Path(pruned_source).exists(),
    )
    if dense_config.vocab_size != pruned_config.vocab_size:
        raise ValueError(
            f"Vocab mismatch: dense={dense_config.vocab_size}, pruned={pruned_config.vocab_size}."
        )
    max_requested_length = max(
        max(workload["prompt_lengths"]) + int(workload["output_tokens"])
        for workload in task_workloads.values()
    )
    for label, config in (("dense", dense_config), ("pruned", pruned_config)):
        if max_requested_length > int(config.max_position_embeddings):
            raise ValueError(
                f"{label} model supports {config.max_position_embeddings} positions, "
                f"but the benchmark requests {max_requested_length}."
            )

    print(BENCHMARK_TITLE)
    print(f"Dense model : {args.dense_model}")
    print(f"Pruned model: {args.pruned_model}")
    print(f"Tokenizer   : {tokenizer_name}")
    print("Task workloads:")
    for task_name, workload in task_workloads.items():
        print(
            f"  {task_name}: prompts={workload['prompt_lengths']}, "
            f"output={workload['output_tokens']} token(s), variants="
            + ", ".join(
                record["name"]
                for record in prompt_records[task_name]
            )
        )
    print(f"Batch sizes: {args.batch_sizes}")
    print(
        f"Rounds: {args.rounds} | warmup/round: {args.warmup} | "
        f"measured repeats/round: {args.repeats} | "
        f"total measured/scenario: {args.rounds * args.repeats}"
    )
    print(f"Dtype: {args.dtype} | attention: {args.attn_implementation}")

    payload: Dict[str, Any] = {
        "status": "planned" if args.dry_run else "running",
        "methodology": {
            "workload": (
                "decoder-only chat generation using website demo prompt files"
                if args.prompt_source == "demo_files"
                else "decoder-only chat generation with exact tokenized input lengths"
            ),
            "website_alignment": (
                "matches llama32_3bi_web.py benchmark task mechanics: system+user chat template, "
                "calibrated finance Good/Bad logit scoring plus one cached label-token decode step, "
                + (
                    "and summarization/general generation fixed-length greedy generation "
                    "with max_new_tokens=min_new_tokens"
                    if args.generation_mode == "fixed"
                    else "and summarization/general generation free streaming "
                    "generation with sampling and EOS allowed"
                )
            ),
            "generation": (
                "fixed-length greedy decoding for summarization/general_generation; "
                "min_new_tokens equals max_new_tokens so both models do equal output work"
                if args.generation_mode == "fixed"
                else "free decoding for summarization/general_generation; "
                "sampling is enabled and EOS can stop generation early"
            ),
            "cache": "on-device dynamic KV cache via past_key_values",
            "timing": "warm synchronized wall-clock model timing",
            "primary_statistic": "P50 across measured requests",
            "tail_statistic": "P95 across measured requests",
            "ttft_definition": "prefill forward plus selection of the first output token",
            "prefill_definition": "prompt processing through availability of the first output token",
            "decode_definition": (
                "for normal generation, time and throughput for output tokens after the first; "
                "for finance_sentiment, the llama32_3bi_web.py Good/Bad scorer's one cached "
                "label-token continuation step"
            ),
            "finance_sentiment_metric_note": (
                "finance_sentiment uses the website implementation: score Good/Bad from calibrated "
                "prefill logits, then time one cached continuation step so decode_tok_s is numeric "
                "in the same way as the UI performance summary"
            ),
            "tpot_definition": (
                "normal generation: decode time divided by output tokens after the first; "
                "finance_sentiment: cached label-token decode latency"
            ),
            "e2e_definition": "model generation from prepared GPU input IDs through the final token",
            "peak_vram_definition": "maximum allocated CUDA memory during inference, including loaded model footprint",
            "model_footprint_definition": "CUDA memory allocated immediately after model load and synchronization",
            "excluded_from_e2e": [
                "model loading",
                "tokenization/chat-template rendering",
                "text detokenization and UI stream rendering",
                "HTTP/network latency",
                "browser rendering",
            ],
            "percentiles": "p50 and p95 over repeated requests",
            "fairness": "same GPU, dtype, attention backend, prompt token IDs, batch and output length",
            "execution_isolation": "models are benchmarked sequentially so they do not contend for one GPU",
            "order_bias_control": "model load order alternates between rounds",
            "prompt_diversity": (
                "demo_files uses every selected benchmark article at its actual website prompt length; "
                "synthetic mode fits multiple realistic prompts to every exact requested input length"
            ),
            "confidence_intervals": (
                "nonparametric independent bootstrap of the ratio of P50 metrics"
            ),
            "scope": (
                "in-process single-GPU Hugging Face backend execution; excludes HTTP queueing, "
                "network transfer, detokenization/UI rendering and multi-user scheduler effects"
            ),
            "speedup_summary_formulas": {
                "prefill_speedup": "pruned prefill_tok_s / original prefill_tok_s",
                "decode_speedup": "pruned decode_tok_s / original decode_tok_s",
                "end_to_end_speedup": "pruned e2e_tok_s / original e2e_tok_s",
            },
        },
        "benchmark_metric_schema": {
            "fixed": {
                "prompt_tokens": "Input context length used for the benchmark.",
                "output_tokens": "Fixed generated output-token count.",
            },
            "measured": {
                "prefill_latency_ms": "Prompt-processing latency before decode begins.",
                "prefill_tok_s": "Prompt-processing throughput.",
                "decode_ms_per_token": "Average per-token decode latency after prefill.",
                "decode_tok_s": "Autoregressive decode throughput after prefill.",
                "total_latency_ms": "Total request latency including prefill and decode.",
                "e2e_tok_s": "Overall output throughput across the full request.",
                "peak_vram_gb": "Highest GPU memory allocated during the run.",
            },
            "performance_summary": [
                "prefill_speedup",
                "decode_speedup",
                "end_to_end_speedup",
            ],
        },
        "configuration": {
            "dense_model": args.dense_model,
            "dense_source": dense_source,
            "pruned_model": args.pruned_model,
            "pruned_source": pruned_source,
            "tokenizer_model": tokenizer_name,
            "tokenizer_source": tokenizer_source,
            "prompt_source": args.prompt_source,
            "tasks": {
                task_name: {
                    "prompt_lengths": workload["prompt_lengths"],
                    "output_tokens": workload["output_tokens"],
                    **(
                        {
                            "label_tokens": finance_label_tokens,
                            "calibration_prompt_tokens": int(finance_calibration_prompt_ids.numel())
                            if finance_calibration_prompt_ids is not None
                            else None,
                        }
                        if task_name == "finance_sentiment"
                        and finance_label_tokens is not None
                        else {}
                    ),
                    "prompt_variants": prompt_records[task_name],
                }
                for task_name, workload in task_workloads.items()
            },
            "prompts_per_length": args.prompts_per_length,
            "batch_sizes": args.batch_sizes,
            "generation_settings": {
                "generation_mode": args.generation_mode,
                "do_sample": args.generation_mode == "free" and args.live_temperature > 0.0,
                "temperature": args.live_temperature if args.generation_mode == "free" else 0.0,
                "top_p": args.live_top_p if args.generation_mode == "free" else None,
                "repetition_penalty": args.repetition_penalty,
                "no_repeat_ngram_size": args.no_repeat_ngram_size,
                "min_new_tokens_equals_max_new_tokens": args.generation_mode == "fixed",
            },
            "warmup": args.warmup,
            "repeats": args.repeats,
            "rounds": args.rounds,
            "total_measured_requests_per_scenario": args.repeats * args.rounds,
            "bootstrap_samples": args.bootstrap_samples,
            "dtype": args.dtype,
            "attn_implementation": args.attn_implementation,
            "model_order": args.model_order,
            "seed": args.seed,
        },
        "config_architectures": {
            "dense": {
                "hidden_size": dense_config.hidden_size,
                "intermediate_size": dense_config.intermediate_size,
                "num_hidden_layers": dense_config.num_hidden_layers,
                "num_attention_heads": dense_config.num_attention_heads,
                "num_key_value_heads": dense_config.num_key_value_heads,
            },
            "pruned": {
                "hidden_size": pruned_config.hidden_size,
                "intermediate_size": pruned_config.intermediate_size,
                "num_hidden_layers": pruned_config.num_hidden_layers,
                "num_attention_heads": pruned_config.num_attention_heads,
                "num_key_value_heads": pruned_config.num_key_value_heads,
            },
        },
        "environment": environment_metadata(None),
        "models": {},
        "comparisons": [],
        "overall_speedups": {},
        "speedups_by_task": {},
        "speedups_by_report_group": {},
        "model_footprint_comparison": {},
        "execution_rounds": [],
        "round_level_comparisons": [],
        "round_stability": [],
    }

    if args.dry_run:
        payload["status"] = "dry_run_complete"
        write_json(args.output_json, payload)
        print(f"Dry run complete. Wrote {args.output_json}")
        return

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this LLM inference speed benchmark.")
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    if args.dtype == "bfloat16" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("The selected GPU does not support bfloat16; use --dtype float16.")

    torch.set_float32_matmul_precision("high")
    payload["environment"] = environment_metadata(device)
    dtype = dtype_from_name(args.dtype)
    model_specs = {
        "dense": (args.dense_model, dense_source),
        "pruned": (args.pruned_model, pruned_source),
    }
    round_results: Dict[str, List[Dict[str, Any]]] = {
        "dense": [],
        "pruned": [],
    }
    first_round_order = list(args.model_order)
    for round_index in range(1, args.rounds + 1):
        order = (
            first_round_order
            if round_index % 2 == 1
            else list(reversed(first_round_order))
        )
        print(f"\n######## Benchmark round {round_index}/{args.rounds}: {' -> '.join(order)} ########")
        round_record: Dict[str, Any] = {
            "round_index": round_index,
            "model_order": order,
            "models": {},
        }
        for label in order:
            model_name, source = model_specs[label]
            result = benchmark_model(
                label,
                model_name,
                source,
                prompts,
                task_workloads,
                finance_label_tokens,
                finance_calibration_prompt_ids,
                args,
                dtype,
                device,
                round_index=round_index,
            )
            round_results[label].append(result)
            round_record["models"][label] = {
                "load_seconds": result["metadata"]["load_seconds"],
                "memory_footprint_gb": result["metadata"]["memory_footprint_gb"],
                "scenario_statuses": [
                    {
                        "task": row["task"],
                        "batch_size": row["batch_size"],
                        "prompt_tokens": row["prompt_tokens_per_request"],
                        "output_tokens": row["new_tokens_per_request"],
                        "status": row["status"],
                    }
                    for row in result["scenarios"]
                ],
            }
        payload["execution_rounds"].append(round_record)
        write_json(args.output_json, payload)

    payload["models"] = {
        label: merge_model_rounds(
            label,
            results,
            include_raw_samples=True,
        )
        for label, results in round_results.items()
    }
    payload["round_level_comparisons"] = [
        {
            "round_index": round_index + 1,
            "model_order": payload["execution_rounds"][round_index]["model_order"],
            "comparisons": compare_models(
                round_results["dense"][round_index],
                round_results["pruned"][round_index],
                bootstrap_samples=args.bootstrap_samples,
                seed=args.seed + (round_index + 1) * 1_000_000,
            ),
        }
        for round_index in range(args.rounds)
    ]
    payload["comparisons"] = compare_models(
        payload["models"]["dense"],
        payload["models"]["pruned"],
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
    )
    payload["round_stability"] = summarize_round_stability(
        payload["round_level_comparisons"]
    )
    dense_footprint = payload["models"]["dense"]["metadata"]["memory_footprint_gb"]
    pruned_footprint = payload["models"]["pruned"]["metadata"]["memory_footprint_gb"]
    payload["model_footprint_comparison"] = {
        "dense_memory_footprint_gb": dense_footprint,
        "pruned_memory_footprint_gb": pruned_footprint,
        "dense_over_pruned_ratio": safe_ratio(dense_footprint, pruned_footprint),
        "reduction_percent": reduction_percent(dense_footprint, pruned_footprint),
    }
    payload["overall_speedups"] = overall_speedups(payload["comparisons"])
    payload["speedups_by_task"] = speedups_by_task(payload["comparisons"])
    payload["speedups_by_report_group"] = speedups_by_report_group(payload["comparisons"])
    payload["status"] = "complete"
    print_comparison_table(payload["comparisons"])
    if args.no_raw_samples:
        remove_raw_samples(payload["models"])
    write_json(args.output_json, payload)
    if args.output_csv:
        write_csv(args.output_csv, payload["comparisons"])

    print(f"\nSaved JSON: {args.output_json}")
    if args.output_csv:
        print(f"Saved CSV : {args.output_csv}")
    if payload["speedups_by_task"]:
        print("Task-level inference speedups (geometric mean across input lengths):")
        for task_name, summary in payload["speedups_by_task"].items():
            decode_value = summary["geomean_decode_speedup"]
            decode_text = f"{decode_value:.3f}x" if math.isfinite(decode_value) else "N/A"
            print(f"  {task_name}:")
            print(f"    Prefill    : {summary['geomean_prefill_speedup']:.3f}x")
            print(f"    Decode     : {decode_text}")
            print(f"    End-to-end : {summary['geomean_end_to_end_speedup']:.3f}x")
            print(
                f"    Peak VRAM reduction (mean): "
                f"{summary['mean_peak_vram_reduction_percent']:.2f}%"
            )
        if payload["speedups_by_report_group"]:
            print("Report-group speedups:")
            for group_name, summary in payload["speedups_by_report_group"].items():
                decode_value = summary["geomean_decode_speedup"]
                decode_text = f"{decode_value:.3f}x" if math.isfinite(decode_value) else "N/A"
                print(f"  {group_name} ({', '.join(summary['tasks'])}):")
                print(f"    Prefill    : {summary['geomean_prefill_speedup']:.3f}x")
                print(f"    Decode     : {decode_text}")
                print(f"    End-to-end : {summary['geomean_end_to_end_speedup']:.3f}x")
        print(
            "  Loaded-model footprint reduction: "
            f"{payload['model_footprint_comparison']['reduction_percent']:.2f}%"
        )


if __name__ == "__main__":
    main()
