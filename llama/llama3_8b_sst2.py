import os
import gc
import json
import math
import time
import random
import warnings
import argparse
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
    matthews_corrcoef,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    import pynvml
except Exception:
    pynvml = None

warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# =========================================================
# Config
# =========================================================
SEED = 42
MODEL_NAME = "meta-llama/Meta-Llama-3-8B"
DATASET_NAME = "stanfordnlp/sst2"
EVAL_SPLIT = "validation"
TUNE_SOURCE_SPLIT = "train"

MAX_LENGTH = 192
BATCH_SIZE = 8
WARMUP_FRAC = 0.20
BENCHMARK_REPEATS = 5
USE_FIXED_SHAPE_PADDING = True
ALLOW_TF32 = True
ENABLE_CUDNN_BENCHMARK = True

DEFAULT_THRESHOLD_CANDIDATES = [0.50, 0.60, 0.65, 0.70, 0.75]

PROMPT_TEMPLATES = [
    "Classify the sentiment of this review.\n"
    "Review: {sentence}\n"
    "Answer with one word only: negative or positive.\n"
    "Sentiment:",
    "Decide whether the following review is negative or positive.\n"
    "Text: {sentence}\n"
    "Return only one label: negative or positive.\n"
    "Label:",
    "Review sentiment classification.\n"
    "Sentence: {sentence}\n"
    "Sentiment:"
]

NEGATIVE_LABELS = [" negative", " bad", " unfavorable"]
POSITIVE_LABELS = [" positive", " good", " favorable"]


# =========================================================
# Reproducibility + device
# =========================================================
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(SEED)

if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = ENABLE_CUDNN_BENCHMARK
    torch.backends.cuda.matmul.allow_tf32 = ALLOW_TF32
    torch.backends.cudnn.allow_tf32 = ALLOW_TF32


# =========================================================
# Helpers
# =========================================================
def get_torch_dtype():
    if torch.cuda.is_available():
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    return torch.float32


def get_model_input_device(model: torch.nn.Module) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        for _, buf in model.named_buffers():
            return buf.device
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def sync_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def cleanup() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def reset_peak_memory() -> None:
    if not torch.cuda.is_available():
        return
    for i in range(torch.cuda.device_count()):
        try:
            torch.cuda.reset_peak_memory_stats(i)
        except Exception:
            pass


def get_max_gpu_memory_gb() -> float:
    if not torch.cuda.is_available():
        return float("nan")
    peaks = []
    for i in range(torch.cuda.device_count()):
        try:
            peaks.append(torch.cuda.max_memory_allocated(i) / (1024 ** 3))
        except Exception:
            pass
    return max(peaks) if peaks else float("nan")


def compute_global_zero_fraction(model: torch.nn.Module) -> float:
    total = 0
    zeros = 0
    for _, p in model.named_parameters():
        if not p.is_floating_point():
            continue
        if p.layout != torch.strided:
            continue
        x = p.detach()
        total += x.numel()
        zeros += (x == 0).sum().item()
    return (zeros / total) if total > 0 else float("nan")


def batched(seq, batch_size: int):
    for i in range(0, len(seq), batch_size):
        yield i, seq[i:i + batch_size]


def format_elapsed_time(seconds: float) -> str:
    seconds = float(seconds)
    if seconds < 60:
        return f"{seconds:.3f} seconds"
    if seconds < 3600:
        return f"{seconds / 60.0:.3f} minutes"
    return f"{seconds / 3600.0:.3f} hours"


def normalize_threshold_candidates(candidates: List[float]) -> List[float]:
    if not candidates:
        raise ValueError("Threshold candidate list cannot be empty.")

    cleaned = []
    for t in candidates:
        t = float(t)
        if not (0.0 < t < 1.0):
            raise ValueError(f"Threshold must be strictly between 0 and 1, got {t}.")
        cleaned.append(round(t, 6))

    cleaned = sorted(set(cleaned))
    return cleaned


def compute_ece(
    y_true: np.ndarray,
    pos_probs: np.ndarray,
    threshold: float = 0.5,
    n_bins: int = 15,
) -> float:
    y_true = np.asarray(y_true).astype(int)
    pos_probs = np.asarray(pos_probs).astype(float)

    preds = (pos_probs >= threshold).astype(int)
    confidences = np.where(preds == 1, pos_probs, 1.0 - pos_probs)
    correctness = (preds == y_true).astype(float)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        if i == n_bins - 1:
            mask = (confidences >= lo) & (confidences <= hi)
        else:
            mask = (confidences >= lo) & (confidences < hi)

        if not np.any(mask):
            continue

        acc = correctness[mask].mean()
        conf = confidences[mask].mean()
        ece += mask.mean() * abs(acc - conf)

    return float(ece)


def select_threshold_from_candidates(
    y_true: np.ndarray,
    pos_probs: np.ndarray,
    candidates: List[float],
    metric: str = "f1",
) -> Tuple[float, float, List[Dict[str, float]]]:
    y_true = np.asarray(y_true).astype(int)
    pos_probs = np.asarray(pos_probs).astype(float)
    candidates = normalize_threshold_candidates(candidates)

    best_t: Optional[float] = None
    best_score = -1.0
    all_results: List[Dict[str, float]] = []

    for t in candidates:
        preds = (pos_probs >= t).astype(int)

        acc = accuracy_score(y_true, preds)
        bal_acc = balanced_accuracy_score(y_true, preds)
        f1 = f1_score(y_true, preds, zero_division=0)

        if metric == "accuracy":
            score = acc
        elif metric == "balanced_accuracy":
            score = bal_acc
        else:
            score = f1

        row = {
            "threshold": float(t),
            "accuracy": float(acc),
            "balanced_accuracy": float(bal_acc),
            "f1": float(f1),
            "score": float(score),
        }
        all_results.append(row)

        if (score > best_score + 1e-12) or (
            abs(score - best_score) <= 1e-12 and (best_t is None or t > best_t)
        ):
            best_score = float(score)
            best_t = float(t)

    if best_t is None:
        raise RuntimeError("Failed to select threshold from candidates.")

    return best_t, best_score, all_results


def validate_args(args) -> None:
    if not (0.0 < args.fixed_threshold < 1.0):
        raise ValueError("--fixed-threshold must be strictly between 0 and 1.")

    if not (0.0 < args.tune_size <= 1.0):
        raise ValueError("--tune-size must be in the interval (0, 1].")

    if not (0.0 <= args.warmup_frac < 1.0):
        raise ValueError("--warmup-frac must be in the interval [0, 1).")

    if args.batch_size < 1:
        raise ValueError("--batch-size must be >= 1.")

    if args.max_length < 2:
        raise ValueError("--max-length must be >= 2.")

    if args.benchmark_repeats < 1:
        raise ValueError("--benchmark-repeats must be >= 1.")

    args.threshold_candidates = normalize_threshold_candidates(args.threshold_candidates)


# =========================================================
# NVML helpers
# =========================================================
def init_nvml():
    if pynvml is None:
        return None
    try:
        pynvml.nvmlInit()
        return pynvml.nvmlDeviceGetHandleByIndex(0)
    except Exception:
        return None


def shutdown_nvml() -> None:
    if pynvml is None:
        return
    try:
        pynvml.nvmlShutdown()
    except Exception:
        pass


def get_total_energy_j(nvml_handle):
    if nvml_handle is None:
        return None
    try:
        return pynvml.nvmlDeviceGetTotalEnergyConsumption(nvml_handle) / 1000.0
    except Exception:
        return None


def get_power_watts(nvml_handle):
    if nvml_handle is None:
        return None
    try:
        return pynvml.nvmlDeviceGetPowerUsage(nvml_handle) / 1000.0
    except Exception:
        return None


# =========================================================
# Model loading
# =========================================================
def load_model_and_tokenizer(model_name: str):
    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    kwargs = {
        "torch_dtype": get_torch_dtype(),
        "low_cpu_mem_usage": True,
    }
    if torch.cuda.is_available():
        kwargs["device_map"] = "auto"

    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    model.eval()
    return model, tokenizer


# =========================================================
# Prompt scoring core
# =========================================================
def sequence_logprobs_batch(
    model,
    tokenizer,
    prompts: List[str],
    label_text: str,
    max_length: int,
    use_fixed_shape_padding: bool = True,
):
    device = get_model_input_device(model)
    label_ids = tokenizer(label_text, add_special_tokens=False).input_ids
    if len(label_ids) == 0:
        raise ValueError(f"Label text tokenized to empty sequence: {label_text!r}")

    pad_id = tokenizer.pad_token_id
    full_input_ids = []
    prompt_lens = []

    for prompt in prompts:
        prompt_ids = tokenizer(prompt, add_special_tokens=False).input_ids
        allowed_prompt_len = max_length - len(label_ids)
        if allowed_prompt_len < 1:
            raise ValueError(f"max_length={max_length} is too small for label {label_text!r}")

        prompt_ids = prompt_ids[:allowed_prompt_len]
        full_ids = prompt_ids + label_ids
        full_input_ids.append(full_ids)
        prompt_lens.append(len(prompt_ids))

    target_seq_len = max_length if use_fixed_shape_padding else max(len(x) for x in full_input_ids)

    input_ids = []
    attention_mask = []

    for ids in full_input_ids:
        if len(ids) > target_seq_len:
            ids = ids[:target_seq_len]
        pad_len = target_seq_len - len(ids)
        input_ids.append(ids + [pad_id] * pad_len)
        attention_mask.append([1] * len(ids) + [0] * pad_len)

    input_ids = torch.tensor(input_ids, dtype=torch.long, device=device)
    attention_mask = torch.tensor(attention_mask, dtype=torch.long, device=device)

    with torch.no_grad():
        out = model(input_ids=input_ids, attention_mask=attention_mask)
        log_probs = torch.log_softmax(out.logits[:, :-1, :], dim=-1)

    shift_labels = input_ids[:, 1:]
    gathered = log_probs.gather(2, shift_labels.unsqueeze(-1)).squeeze(-1)

    seq_scores = []
    label_len = len(label_ids)

    for i, prompt_len in enumerate(prompt_lens):
        start = prompt_len - 1
        end = start + label_len
        seq_scores.append(gathered[i, start:end].sum().item())

    seq_scores = np.array(seq_scores, dtype=np.float64)
    label_len_arr = np.full(len(prompts), label_len, dtype=np.int32)
    prompt_lens = np.array(prompt_lens, dtype=np.int32)
    return seq_scores, prompt_lens, label_len_arr


def score_batch_prompt_classifier(
    model,
    tokenizer,
    batch_texts: List[str],
    max_length: int,
):
    neg_scores_all = []
    pos_scores_all = []
    first_prompt_lens = None

    for template in PROMPT_TEMPLATES:
        prompts = [template.format(sentence=t) for t in batch_texts]

        for neg_label in NEGATIVE_LABELS:
            seq_scores, prompt_lens, label_lens = sequence_logprobs_batch(
                model=model,
                tokenizer=tokenizer,
                prompts=prompts,
                label_text=neg_label,
                max_length=max_length,
                use_fixed_shape_padding=USE_FIXED_SHAPE_PADDING,
            )
            norm_scores = seq_scores / label_lens
            neg_scores_all.append(norm_scores)
            if first_prompt_lens is None:
                first_prompt_lens = prompt_lens

        for pos_label in POSITIVE_LABELS:
            seq_scores, _, label_lens = sequence_logprobs_batch(
                model=model,
                tokenizer=tokenizer,
                prompts=prompts,
                label_text=pos_label,
                max_length=max_length,
                use_fixed_shape_padding=USE_FIXED_SHAPE_PADDING,
            )
            norm_scores = seq_scores / label_lens
            pos_scores_all.append(norm_scores)

    neg_avg = np.mean(np.stack(neg_scores_all, axis=0), axis=0)
    pos_avg = np.mean(np.stack(pos_scores_all, axis=0), axis=0)

    m = np.maximum(neg_avg, pos_avg)
    exp_neg = np.exp(neg_avg - m)
    exp_pos = np.exp(pos_avg - m)
    pos_probs = exp_pos / (exp_neg + exp_pos)

    margins = np.abs(2.0 * pos_probs - 1.0)

    return {
        "pos_probs": pos_probs,
        "margins": margins,
        "prompt_lens": first_prompt_lens,
    }


def evaluate_prompt_classifier(
    model,
    tokenizer,
    texts: List[str],
    batch_size: int,
    max_length: int,
    threshold: float = 0.5,
):
    n = len(texts)
    pos_probs = np.zeros(n, dtype=np.float64)
    margins = np.zeros(n, dtype=np.float64)
    prompt_lens_all = np.zeros(n, dtype=np.int32)

    for start, batch_texts in batched(texts, batch_size):
        end = start + len(batch_texts)
        batch_out = score_batch_prompt_classifier(
            model=model,
            tokenizer=tokenizer,
            batch_texts=batch_texts,
            max_length=max_length,
        )
        pos_probs[start:end] = batch_out["pos_probs"]
        margins[start:end] = batch_out["margins"]
        prompt_lens_all[start:end] = batch_out["prompt_lens"]

    preds = (pos_probs >= threshold).astype(int)

    return {
        "preds": preds,
        "pos_probs": pos_probs,
        "margins": margins,
        "prompt_lens": prompt_lens_all,
    }


# =========================================================
# Benchmarking pass
# =========================================================
def benchmark_pass(
    model,
    tokenizer,
    texts: List[str],
    batch_size: int,
    max_length: int,
    warmup_frac: float,
    repeats: int,
    nvml_handle,
):
    model.eval()

    all_batches = [batch_texts for _, batch_texts in batched(texts, batch_size)]
    total_batches = len(all_batches)

    warmup_batches = 0 if total_batches <= 1 else min(
        total_batches - 1,
        math.ceil(total_batches * warmup_frac),
    )

    measured_batches = all_batches[warmup_batches:]
    measured_samples_per_pass = sum(len(b) for b in measured_batches)

    if len(measured_batches) == 0 or measured_samples_per_pass == 0:
        return {
            "Model-only Latency (ms/sample, post-warmup)": np.nan,
            "Model-only Time (s, post-warmup)": np.nan,
            "Throughput (samples/s, post-warmup)": np.nan,
            "Measured Energy (J, post-warmup)": np.nan,
            "Energy per Sample (J/sample, post-warmup)": np.nan,
            "Warmup Batches": warmup_batches,
            "Benchmark Repeats": repeats,
            "P50 Batch Latency ms": np.nan,
            "P95 Batch Latency ms": np.nan,
        }

    with torch.inference_mode():
        for batch_texts in all_batches[:warmup_batches]:
            _ = score_batch_prompt_classifier(
                model=model,
                tokenizer=tokenizer,
                batch_texts=batch_texts,
                max_length=max_length,
            )
    sync_cuda()

    using_total_energy = get_total_energy_j(nvml_handle) is not None
    energy_start_j = None
    energy_end_j = None
    fallback_energy_j = 0.0

    total_model_time_ms = 0.0
    total_measured_samples = measured_samples_per_pass * repeats
    batch_latencies_ms = []

    start_event = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
    end_event = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None

    with torch.inference_mode():
        if using_total_energy:
            sync_cuda()
            energy_start_j = get_total_energy_j(nvml_handle)

        for _ in range(repeats):
            for batch_texts in measured_batches:
                if not using_total_energy:
                    sync_cuda()
                    p0 = get_power_watts(nvml_handle)
                    wall_t0 = time.perf_counter()

                if start_event is not None:
                    start_event.record()
                    _ = score_batch_prompt_classifier(
                        model=model,
                        tokenizer=tokenizer,
                        batch_texts=batch_texts,
                        max_length=max_length,
                    )
                    end_event.record()
                    sync_cuda()
                    elapsed_ms = start_event.elapsed_time(end_event)
                else:
                    wall0 = time.perf_counter()
                    _ = score_batch_prompt_classifier(
                        model=model,
                        tokenizer=tokenizer,
                        batch_texts=batch_texts,
                        max_length=max_length,
                    )
                    wall1 = time.perf_counter()
                    elapsed_ms = (wall1 - wall0) * 1000.0

                total_model_time_ms += elapsed_ms
                batch_latencies_ms.append(elapsed_ms)

                if not using_total_energy:
                    wall_t1 = time.perf_counter()
                    p1 = get_power_watts(nvml_handle)
                    if p0 is not None and p1 is not None:
                        fallback_energy_j += ((p0 + p1) / 2.0) * (wall_t1 - wall_t0)

        if using_total_energy:
            sync_cuda()
            energy_end_j = get_total_energy_j(nvml_handle)

    measured_model_time_s = total_model_time_ms / 1000.0
    avg_latency_ms = total_model_time_ms / total_measured_samples
    throughput_sps = total_measured_samples / measured_model_time_s

    if using_total_energy and energy_start_j is not None and energy_end_j is not None:
        measured_energy_j = max(0.0, energy_end_j - energy_start_j)
    else:
        measured_energy_j = fallback_energy_j

    energy_per_sample_j = measured_energy_j / total_measured_samples if total_measured_samples > 0 else np.nan

    return {
        "Model-only Latency (ms/sample, post-warmup)": avg_latency_ms,
        "Model-only Time (s, post-warmup)": measured_model_time_s,
        "Throughput (samples/s, post-warmup)": throughput_sps,
        "Measured Energy (J, post-warmup)": measured_energy_j,
        "Energy per Sample (J/sample, post-warmup)": energy_per_sample_j,
        "Warmup Batches": warmup_batches,
        "Benchmark Repeats": repeats,
        "P50 Batch Latency ms": float(np.percentile(batch_latencies_ms, 50)),
        "P95 Batch Latency ms": float(np.percentile(batch_latencies_ms, 95)),
    }


# =========================================================
# Result building
# =========================================================
def build_result(
    model_name: str,
    dataset_name: str,
    split: str,
    batch_size: int,
    max_length: int,
    labels: np.ndarray,
    preds: np.ndarray,
    pos_probs: np.ndarray,
    margins: np.ndarray,
    prompt_lens: np.ndarray,
    benchmark_stats: Dict[str, Any],
    dense_zero_fraction: float,
    notes: str,
    threshold: float,
    threshold_source: str,
    threshold_metric: str,
    threshold_candidates: List[float],
    threshold_search_results: List[Dict[str, float]],
    total_script_runtime_s: float,
):
    cm = confusion_matrix(labels, preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    cls_report = classification_report(
        labels,
        preds,
        labels=[0, 1],
        target_names=["negative", "positive"],
        output_dict=True,
        zero_division=0,
    )

    pos_precision, pos_recall, pos_f1, _ = precision_recall_fscore_support(
        labels,
        preds,
        labels=[1],
        average=None,
        zero_division=0,
    )
    pos_precision = float(pos_precision[0]) * 100.0
    pos_recall = float(pos_recall[0]) * 100.0
    pos_f1 = float(pos_f1[0]) * 100.0

    accuracy = accuracy_score(labels, preds) * 100.0
    bal_acc = balanced_accuracy_score(labels, preds) * 100.0
    mcc = matthews_corrcoef(labels, preds)

    try:
        roc_auc = roc_auc_score(labels, pos_probs) * 100.0
    except Exception:
        roc_auc = float("nan")

    probs_2col = np.column_stack([1.0 - pos_probs, pos_probs])
    probs_2col = np.clip(probs_2col, 1e-12, 1.0 - 1e-12)
    probs_2col = probs_2col / probs_2col.sum(axis=1, keepdims=True)

    try:
        ll = log_loss(labels, probs_2col, labels=[0, 1])
    except Exception:
        ll = float("nan")

    try:
        brier = brier_score_loss(labels, pos_probs)
    except Exception:
        brier = float("nan")

    ece = compute_ece(labels, pos_probs, threshold=threshold, n_bins=15)

    summary = {
        "Model Variant": model_name,
        "Accuracy (%)": accuracy,
        "Model-only Latency (ms/sample, post-warmup)": benchmark_stats["Model-only Latency (ms/sample, post-warmup)"],
        "Model-only Time (s, post-warmup)": benchmark_stats["Model-only Time (s, post-warmup)"],
        "Throughput (samples/s, post-warmup)": benchmark_stats["Throughput (samples/s, post-warmup)"],
        "Measured Energy (J, post-warmup)": benchmark_stats["Measured Energy (J, post-warmup)"],
        "Energy per Sample (J/sample, post-warmup)": benchmark_stats["Energy per Sample (J/sample, post-warmup)"],
        "Dense Zero Fraction": dense_zero_fraction,
        "Warmup Batches": benchmark_stats["Warmup Batches"],
        "Benchmark Repeats": benchmark_stats["Benchmark Repeats"],
        "Notes": notes,
        "Precision (%)": pos_precision,
        "Recall (%)": pos_recall,
        "F1 (%)": pos_f1,
        "Balanced Accuracy (%)": bal_acc,
        "MCC": mcc,
        "ROC-AUC (%)": roc_auc,
        "Log Loss": ll,
        "Brier Score": brier,
        "ECE (15 bins)": ece,
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
        "P50 Batch Latency ms": benchmark_stats["P50 Batch Latency ms"],
        "P95 Batch Latency ms": benchmark_stats["P95 Batch Latency ms"],
        "Max GPU Memory GB": get_max_gpu_memory_gb(),
        "Avg Prompt Tokens": float(np.mean(prompt_lens)),
        "P95 Prompt Tokens": float(np.percentile(prompt_lens, 95)),
        "Avg Confidence Margin": float(np.mean(margins)),
        "Threshold": threshold,
        "Threshold Source": threshold_source,
        "Threshold Metric": threshold_metric,
        "Threshold Candidates": threshold_candidates,
        "Dataset": dataset_name,
        "Split": split,
        "Batch Size": batch_size,
        "Max Length": max_length,
        "Prompt Template": " || ".join(PROMPT_TEMPLATES),
        "Negative Label Text": ", ".join(NEGATIVE_LABELS),
        "Positive Label Text": ", ".join(POSITIVE_LABELS),
        "Total Script Runtime (s)": total_script_runtime_s,
        "Total Script Runtime (min)": total_script_runtime_s / 60.0,
        "Total Script Runtime (formatted)": format_elapsed_time(total_script_runtime_s),
    }

    return {
        "summary": summary,
        "classification_report": cls_report,
        "confusion_matrix": cm.tolist(),
        "threshold_search_results": threshold_search_results,
    }


# =========================================================
# Printing
# =========================================================
def pretty_print_result(result: Dict[str, Any]) -> None:
    s = result["summary"]
    cr = result["classification_report"]
    cm = result["confusion_matrix"]
    threshold_rows = result.get("threshold_search_results", [])

    pd.set_option("display.width", 240)
    pd.set_option("display.max_colwidth", 160)

    run_info = pd.Series({
        "Model": s["Model Variant"],
        "Dataset": f'{s["Dataset"]} / {s["Split"]}',
        "Batch size": s["Batch Size"],
        "Max length": s["Max Length"],
        "Warmup batches": s["Warmup Batches"],
        "Benchmark repeats": s["Benchmark Repeats"],
        "Threshold": s["Threshold"],
        "Threshold source": s["Threshold Source"],
        "Threshold metric": s["Threshold Metric"],
        "Total script runtime": s["Total Script Runtime (formatted)"],
    })

    headline = pd.Series({
        "Accuracy (%)": s["Accuracy (%)"],
        "Precision (%)": s["Precision (%)"],
        "Recall (%)": s["Recall (%)"],
        "F1 (%)": s["F1 (%)"],
        "Balanced Accuracy (%)": s["Balanced Accuracy (%)"],
        "MCC": s["MCC"],
        "ROC-AUC (%)": s["ROC-AUC (%)"],
        "Log Loss": s["Log Loss"],
        "Brier Score": s["Brier Score"],
        "ECE (15 bins)": s["ECE (15 bins)"],
        "Avg Confidence Margin": s["Avg Confidence Margin"],
    }).round(4)

    perf = pd.Series({
        "Latency ms/sample": s["Model-only Latency (ms/sample, post-warmup)"],
        "Total model time s": s["Model-only Time (s, post-warmup)"],
        "Throughput samples/s": s["Throughput (samples/s, post-warmup)"],
        "Measured Energy J": s["Measured Energy (J, post-warmup)"],
        "Energy per Sample J": s["Energy per Sample (J/sample, post-warmup)"],
        "P50 batch latency ms": s["P50 Batch Latency ms"],
        "P95 batch latency ms": s["P95 Batch Latency ms"],
        "Max GPU Memory GB": s["Max GPU Memory GB"],
        "Dense Zero Fraction": s["Dense Zero Fraction"],
        "Avg Prompt Tokens": s["Avg Prompt Tokens"],
        "P95 Prompt Tokens": s["P95 Prompt Tokens"],
        "Total Script Runtime (s)": s["Total Script Runtime (s)"],
        "Total Script Runtime (min)": s["Total Script Runtime (min)"],
    }).round(4)

    cm_df = pd.DataFrame(
        cm,
        index=["Actual negative", "Actual positive"],
        columns=["Pred negative", "Pred positive"],
    )

    class_df = pd.DataFrame(cr).T
    keep = ["negative", "positive", "macro avg", "weighted avg"]
    class_df = class_df.loc[[k for k in keep if k in class_df.index]].round(4)

    print("\n=== Run info ===")
    print(run_info.to_string())

    if threshold_rows:
        threshold_df = pd.DataFrame(threshold_rows).sort_values(
            by=["score", "threshold"],
            ascending=[False, False],
        ).reset_index(drop=True)

        for col in ["threshold", "accuracy", "balanced_accuracy", "f1", "score"]:
            if col in threshold_df.columns:
                threshold_df[col] = threshold_df[col].map(lambda x: round(float(x), 4))

        print("\n=== Threshold search results ===")
        print(threshold_df.to_string(index=False))

    print("\n=== Headline metrics ===")
    print(headline.to_string())

    print("\n=== Confusion matrix ===")
    print(cm_df.to_string())

    print("\n=== Per-class report ===")
    print(class_df.to_string())

    print("\n=== Performance ===")
    print(perf.to_string())

    print("\n=== Diagnosis ===")
    print(f"TN={s['TN']} FP={s['FP']} FN={s['FN']} TP={s['TP']}")
    if s["Recall (%)"] < 40:
        print("Warning: the model is missing many positive examples.")
    if s["Precision (%)"] - s["Recall (%)"] > 20:
        print("Warning: predictions are too conservative for the positive class.")


# =========================================================
# Main
# =========================================================
def main():
    main_start = time.perf_counter()

    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default=MODEL_NAME)
    parser.add_argument("--dataset", type=str, default=DATASET_NAME)
    parser.add_argument("--eval-split", type=str, default=EVAL_SPLIT)
    parser.add_argument("--tune-source-split", type=str, default=TUNE_SOURCE_SPLIT)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--max-length", type=int, default=MAX_LENGTH)
    parser.add_argument("--warmup-frac", type=float, default=WARMUP_FRAC)
    parser.add_argument("--benchmark-repeats", type=int, default=BENCHMARK_REPEATS)
    parser.add_argument("--seed", type=int, default=SEED)

    parser.add_argument("--fixed-threshold", type=float, default=0.5)
    parser.add_argument("--tune-threshold-on-train", action="store_true")
    parser.add_argument("--tune-size", type=float, default=0.1)
    parser.add_argument(
        "--threshold-metric",
        type=str,
        default="f1",
        choices=["accuracy", "f1", "balanced_accuracy"],
    )
    parser.add_argument(
        "--threshold-candidates",
        type=float,
        nargs="+",
        default=DEFAULT_THRESHOLD_CANDIDATES,
        help="Candidate thresholds to evaluate on the calibration subset.",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default="output/llama3_8b_sst2_threshold_tuning_result.json",
    )

    args = parser.parse_args()
    validate_args(args)
    set_seed(args.seed)

    print(f"Loading dataset: {args.dataset} / {args.eval_split}")
    eval_ds = load_dataset(args.dataset, split=args.eval_split)
    eval_texts = list(eval_ds["sentence"])
    eval_labels = np.array(eval_ds["label"], dtype=np.int32)

    if args.tune_threshold_on_train:
        print(f"Loading threshold-tuning split: {args.dataset} / {args.tune_source_split}")
        tune_ds = load_dataset(args.dataset, split=args.tune_source_split)
        tune_texts_full = np.array(tune_ds["sentence"], dtype=object)
        tune_labels_full = np.array(tune_ds["label"], dtype=np.int32)

        tune_texts, _, tune_labels, _ = train_test_split(
            tune_texts_full,
            tune_labels_full,
            test_size=max(0.0, min(1.0, 1.0 - args.tune_size)),
            random_state=args.seed,
            stratify=tune_labels_full,
        )
        tune_texts = list(tune_texts)
        print(f"Calibration subset size: {len(tune_texts)}")
    else:
        tune_texts = None
        tune_labels = None

    model, tokenizer = load_model_and_tokenizer(args.model_name)
    nvml_handle = init_nvml()
    
    print("\n=== Device placement diagnostics ===")
    print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES", "<not set>"))

    if torch.cuda.is_available():
        print("Visible CUDA device count:", torch.cuda.device_count())
        print("Current CUDA device index:", torch.cuda.current_device())
        print("GPU name:", torch.cuda.get_device_name(0))
    else:
        print("CUDA not available.")

    device_map = getattr(model, "hf_device_map", None)
    print("hf_device_map:", device_map)

    if device_map is not None:
        unique_devices = sorted({str(v) for v in device_map.values()})
        print("Unique mapped devices:", unique_devices)

        has_cpu = any(str(v) == "cpu" for v in device_map.values())
        has_disk = any(str(v) == "disk" for v in device_map.values())
        has_cuda = any(str(v).startswith("cuda") or str(v).isdigit() for v in device_map.values())

        if has_cuda and not has_cpu and not has_disk:
            print("Placement check: model appears fully GPU-resident.")
        elif has_cuda and (has_cpu or has_disk):
            print("Placement check: model is partially offloaded (GPU + CPU/disk).")
        elif has_cpu or has_disk:
            print("Placement check: model is not fully on GPU.")
        else:
            print("Placement check: unable to classify placement cleanly.")


    try:
        threshold_search_results: List[Dict[str, float]] = []

        if args.tune_threshold_on_train:
            print("Tuning threshold on train-derived calibration subset...")
            print(f"Candidate thresholds: {args.threshold_candidates}")

            tune_outputs = evaluate_prompt_classifier(
                model=model,
                tokenizer=tokenizer,
                texts=tune_texts,
                batch_size=args.batch_size,
                max_length=args.max_length,
                threshold=0.5,
            )

            threshold, tune_score, threshold_search_results = select_threshold_from_candidates(
                tune_labels,
                tune_outputs["pos_probs"],
                candidates=args.threshold_candidates,
                metric=args.threshold_metric,
            )

            print("\nThreshold tuning results:")
            for row in threshold_search_results:
                print(
                    f"  threshold={row['threshold']:.2f} "
                    f"-> accuracy={row['accuracy']:.4f}, "
                    f"balanced_accuracy={row['balanced_accuracy']:.4f}, "
                    f"f1={row['f1']:.4f}"
                )

            print(f"\nSelected threshold: {threshold:.2f}")
            threshold_source = f"tuned_on_{args.tune_source_split}_calibration_subset"
            notes = (
                f"Prompt ensemble evaluation. Threshold selected on a calibration subset drawn from "
                f"{args.tune_source_split} using candidate thresholds {args.threshold_candidates} "
                f"and metric {args.threshold_metric}={tune_score:.4f}; final evaluation on untouched "
                f"{args.eval_split}."
            )
        else:
            threshold = float(args.fixed_threshold)
            threshold_source = "fixed"
            notes = (
                f"Prompt ensemble evaluation with fixed threshold={threshold:.2f}. "
                f"No threshold tuning was performed on the evaluation split."
            )
            print(f"Using fixed threshold: {threshold:.2f}")

        print("Running full evaluation on evaluation split...")
        full_outputs = evaluate_prompt_classifier(
            model=model,
            tokenizer=tokenizer,
            texts=eval_texts,
            batch_size=args.batch_size,
            max_length=args.max_length,
            threshold=threshold,
        )
        preds = full_outputs["preds"]

        sync_cuda()
        reset_peak_memory()
        print(f"Benchmark repeats: {args.benchmark_repeats}")

        bench = benchmark_pass(
            model=model,
            tokenizer=tokenizer,
            texts=eval_texts,
            batch_size=args.batch_size,
            max_length=args.max_length,
            warmup_frac=args.warmup_frac,
            repeats=args.benchmark_repeats,
            nvml_handle=nvml_handle,
        )

        total_script_runtime_s = time.perf_counter() - main_start
        dense_zero_fraction = compute_global_zero_fraction(model)

        result = build_result(
            model_name=args.model_name,
            dataset_name=args.dataset,
            split=args.eval_split,
            batch_size=args.batch_size,
            max_length=args.max_length,
            labels=eval_labels,
            preds=preds,
            pos_probs=full_outputs["pos_probs"],
            margins=full_outputs["margins"],
            prompt_lens=full_outputs["prompt_lens"],
            benchmark_stats=bench,
            dense_zero_fraction=dense_zero_fraction,
            notes=notes,
            threshold=threshold,
            threshold_source=threshold_source,
            threshold_metric=args.threshold_metric,
            threshold_candidates=args.threshold_candidates,
            threshold_search_results=threshold_search_results,
            total_script_runtime_s=total_script_runtime_s,
        )

        pretty_print_result(result)

        output_dir = os.path.dirname(args.output_json)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, allow_nan=True)

        print(f"\nSaved JSON to {args.output_json}")
        print(f"Total script runtime: {format_elapsed_time(total_script_runtime_s)}")

    finally:
        shutdown_nvml()
        cleanup()


if __name__ == "__main__":
    main()