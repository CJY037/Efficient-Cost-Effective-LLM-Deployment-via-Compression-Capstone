#!/usr/bin/env python3
"""
Measure inference memory for selected BERT SST-2 variants on real SST-2 batches.

Variants:
1) BERT Base (Baseline)            -> dense FP32
2) BERT Base FP16                  -> dense FP16
3) ETP-inspired 2:4 FP16           -> exact-mask sparse runtime evaluation
4) ETP-inspired 4:12               -> dense local checkpoint with zeros

This script focuses on actual inference memory on GPU:
- current allocated/reserved memory after model load
- peak allocated/reserved memory during representative inference

It mirrors the model-loading paths used in bert_sst2.py for these variants.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import random
import time
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer


SEED = 42
MODEL_NAME = "textattack/bert-base-uncased-SST-2"
DATASET_NAME = "stanfordnlp/sst2"
ETP_MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bert_sst2_etp_4of12")
ETP_2OF4_MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bert_sst2_etp_2of4")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def configure_torch() -> None:
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True


def cleanup() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def get_model_device(model: torch.nn.Module) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        for _, buf in model.named_buffers():
            return buf.device
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def bytes_to_gb(num_bytes: int) -> float:
    return float(num_bytes) / (1024 ** 3)


def current_memory_stats() -> Dict[str, float]:
    if not torch.cuda.is_available():
        return {
            "allocated_gb": float("nan"),
            "reserved_gb": float("nan"),
        }
    return {
        "allocated_gb": bytes_to_gb(torch.cuda.memory_allocated()),
        "reserved_gb": bytes_to_gb(torch.cuda.memory_reserved()),
    }


def peak_memory_stats() -> Dict[str, float]:
    if not torch.cuda.is_available():
        return {
            "peak_allocated_gb": float("nan"),
            "peak_reserved_gb": float("nan"),
        }
    return {
        "peak_allocated_gb": bytes_to_gb(torch.cuda.max_memory_allocated()),
        "peak_reserved_gb": bytes_to_gb(torch.cuda.max_memory_reserved()),
    }


def reset_peak_stats() -> None:
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def sync_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def build_eval_batches(
    tokenizer,
    texts: List[str],
    labels: torch.Tensor,
    *,
    batch_size: int,
    max_length: int,
) -> List[Dict[str, torch.Tensor]]:
    enc = tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )

    batches: List[Dict[str, torch.Tensor]] = []
    n = labels.size(0)
    keys = list(enc.keys())
    for i in range(0, n, batch_size):
        if i + batch_size > n:
            break
        batch = {k: enc[k][i:i + batch_size].contiguous() for k in keys}
        batch["labels"] = labels[i:i + batch_size].contiguous()
        batches.append(batch)
    return batches


def prepare_batch_inputs(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {
        key: value.to(device, non_blocking=True)
        for key, value in batch.items()
        if key != "labels"
    }


def compute_dense_zero_fraction(model: torch.nn.Module) -> float:
    total = 0
    zeros = 0
    for _, p in model.named_parameters():
        if not p.is_floating_point():
            continue
        if p.layout != torch.strided:
            continue
        x = p.detach()
        try:
            zero_count = int((x == 0).sum().item())
        except Exception:
            continue
        total += int(x.numel())
        zeros += zero_count
    return float(zeros / total) if total > 0 else float("nan")


def load_local_etp_2of4_sparse_bert(model_dir: str, masks_path: str | None = None) -> torch.nn.Module:
    try:
        from torch.sparse import to_sparse_semi_structured
    except Exception as exc:
        raise RuntimeError("torch.sparse.to_sparse_semi_structured is not available in this runtime.") from exc

    if masks_path is None:
        masks_path = os.path.join(model_dir, "prune_masks.pt")

    model = AutoModelForSequenceClassification.from_pretrained(
        model_dir,
        local_files_only=True,
        torch_dtype=torch.float16,
    )
    model = model.to(torch.device("cuda")).eval().half()

    mask_bundle = torch.load(masks_path, map_location="cpu")
    masks = mask_bundle.get("masks", {})
    converted = 0
    expected = 0
    skipped = []

    with torch.no_grad():
        for module_name, mask in masks.items():
            try:
                module = model.get_submodule(module_name)
            except AttributeError:
                skipped.append(f"{module_name}: missing submodule")
                continue
            if not isinstance(module, nn.Linear):
                continue
            expected += 1

            weight = module.weight.data
            masked = (weight * mask.to(device=weight.device, dtype=weight.dtype)).contiguous()
            try:
                sparse_weight = to_sparse_semi_structured(masked)
                module.weight = nn.Parameter(sparse_weight, requires_grad=False)
                converted += 1
            except Exception as exc:
                skipped.append(f"{module_name}: {exc}")

    if converted == 0:
        raise RuntimeError("No local 2:4 checkpoint layers were converted to SparseSemiStructuredTensor.")
    if converted != expected:
        raise RuntimeError(
            f"Converted {converted}/{expected} local 2:4 checkpoint layers to "
            f"SparseSemiStructuredTensor. First failures: {skipped[:3]}"
        )
    return model


def build_variant_specs() -> List[Dict[str, Any]]:
    return [
        {
            "name": "BERT Base (Baseline)",
            "tokenizer_ref": MODEL_NAME,
            "build_model": lambda: AutoModelForSequenceClassification.from_pretrained(MODEL_NAME).to("cuda").eval(),
            "notes": "Dense FP32 baseline.",
        },
        {
            "name": "BERT Base FP16",
            "tokenizer_ref": MODEL_NAME,
            "build_model": lambda: AutoModelForSequenceClassification.from_pretrained(
                MODEL_NAME,
                torch_dtype=torch.float16,
            ).to("cuda").eval(),
            "notes": "Dense FP16 baseline.",
        },
        {
            "name": "ETP-inspired 2:4 FP16",
            "tokenizer_ref": ETP_2OF4_MODEL_DIR,
            "build_model": lambda: load_local_etp_2of4_sparse_bert(ETP_2OF4_MODEL_DIR),
            "notes": "Exact-mask SparseSemiStructuredTensor runtime path.",
        },
        {
            "name": "ETP-inspired 4:12",
            "tokenizer_ref": ETP_MODEL_DIR,
            "build_model": lambda: AutoModelForSequenceClassification.from_pretrained(
                ETP_MODEL_DIR,
                local_files_only=True,
            ).to("cuda").eval(),
            "notes": "Dense local checkpoint with zeros; no sparse runtime backend.",
        },
    ]


@torch.inference_mode()
def run_inference_repeats(
    model: torch.nn.Module,
    cpu_batches: List[Dict[str, torch.Tensor]],
    *,
    warmup_repeats: int,
    measure_repeats: int,
) -> None:
    device = get_model_device(model)

    for _ in range(warmup_repeats):
        for batch in cpu_batches:
            inputs = prepare_batch_inputs(batch, device)
            outputs = model(**inputs)
            del outputs, inputs
    sync_cuda()

    reset_peak_stats()
    for _ in range(measure_repeats):
        for batch in cpu_batches:
            inputs = prepare_batch_inputs(batch, device)
            outputs = model(**inputs)
            del outputs, inputs
    sync_cuda()


def measure_variant_memory(
    variant: Dict[str, Any],
    *,
    val_texts: List[str],
    val_labels: torch.Tensor,
    batch_size: int,
    max_length: int,
    num_batches: int,
    warmup_repeats: int,
    measure_repeats: int,
) -> Dict[str, Any]:
    cleanup()
    model = None
    tokenizer = AutoTokenizer.from_pretrained(
        variant["tokenizer_ref"],
        local_files_only=variant["tokenizer_ref"].startswith(ETP_MODEL_DIR) or variant["tokenizer_ref"].startswith(ETP_2OF4_MODEL_DIR),
    )
    cpu_batches = build_eval_batches(
        tokenizer,
        val_texts,
        val_labels,
        batch_size=batch_size,
        max_length=max_length,
    )[:num_batches]
    if not cpu_batches:
        raise ValueError("No full evaluation batches available for the requested settings.")

    try:
        load_start = time.perf_counter()
        model = variant["build_model"]()
        sync_cuda()
        load_time_s = time.perf_counter() - load_start
        zero_fraction = compute_dense_zero_fraction(model)

        after_load = current_memory_stats()

        device = get_model_device(model)
        sample_inputs = prepare_batch_inputs(cpu_batches[0], device)
        sync_cuda()
        after_input = current_memory_stats()
        del sample_inputs
        sync_cuda()

        run_inference_repeats(
            model,
            cpu_batches,
            warmup_repeats=warmup_repeats,
            measure_repeats=measure_repeats,
        )
        after_inference = current_memory_stats()
        peak = peak_memory_stats()

        result = {
            "Model Variant": variant["name"],
            "Notes": variant["notes"],
            "Batch Size": int(batch_size),
            "Max Length": int(max_length),
            "Measured CPU Batches": int(len(cpu_batches)),
            "Warmup Repeats": int(warmup_repeats),
            "Measure Repeats": int(measure_repeats),
            "Model Load Time (s)": float(load_time_s),
            "Dense Zero Fraction": float(zero_fraction),
            "Allocated After Load (GB)": float(after_load["allocated_gb"]),
            "Reserved After Load (GB)": float(after_load["reserved_gb"]),
            "Allocated After Input Prep (GB)": float(after_input["allocated_gb"]),
            "Reserved After Input Prep (GB)": float(after_input["reserved_gb"]),
            "Peak Allocated During Inference (GB)": float(peak["peak_allocated_gb"]),
            "Peak Reserved During Inference (GB)": float(peak["peak_reserved_gb"]),
            "Allocated After Inference (GB)": float(after_inference["allocated_gb"]),
            "Reserved After Inference (GB)": float(after_inference["reserved_gb"]),
        }
    finally:
        try:
            del model
        except Exception:
            pass
        cleanup()

    return result


def parse_args():
    parser = argparse.ArgumentParser(description="Measure inference memory for selected BERT SST-2 variants.")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--num-batches", type=int, default=4)
    parser.add_argument("--warmup-repeats", type=int, default=3)
    parser.add_argument("--measure-repeats", type=int, default=5)
    parser.add_argument("--output-json", type=str, default="output/bert_sst2_memory_benchmark.json")
    return parser.parse_args()


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this memory benchmark.")
    if args.batch_size < 1 or args.max_length < 1 or args.num_batches < 1:
        raise ValueError("batch-size, max-length, and num-batches must be >= 1.")

    set_seed(SEED)
    configure_torch()

    dataset = load_dataset(DATASET_NAME)
    val_texts = list(dataset["validation"]["sentence"])
    val_labels = torch.tensor(dataset["validation"]["label"], dtype=torch.long)

    results: List[Dict[str, Any]] = []
    for variant in build_variant_specs():
        print(f"\n=== {variant['name']} ===")
        try:
            row = measure_variant_memory(
                variant,
                val_texts=val_texts,
                val_labels=val_labels,
                batch_size=args.batch_size,
                max_length=args.max_length,
                num_batches=args.num_batches,
                warmup_repeats=args.warmup_repeats,
                measure_repeats=args.measure_repeats,
            )
            results.append(row)
            print(json.dumps(row, indent=2))
        except Exception as exc:
            failure = {
                "Model Variant": variant["name"],
                "Notes": variant["notes"],
                "Failure": repr(exc),
            }
            results.append(failure)
            print(f"[WARN] {variant['name']} failed: {exc!r}")

    os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2, allow_nan=True)
    print(f"\nSaved JSON to {args.output_json}")


if __name__ == "__main__":
    main()
