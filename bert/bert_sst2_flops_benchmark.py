#!/usr/bin/env python3
"""
Measure runtime-aware forward FLOPs for selected BERT SST-2 variants.

Variants:
1) BERT Base (Baseline)   -> dense FP32
2) BERT Base FP16         -> dense FP16
3) ETP-inspired 2:4 FP16  -> sparse runtime on masked MLP layers
4) ETP-inspired 4:12      -> dense runtime with zeros in masked MLP layers

What this script reports:
- Dense-equivalent forward FLOPs for the actual SST-2 batch shape
- Effective nonzero forward FLOPs (if zero weights were perfectly skipped)
- Runtime-aware forward FLOPs estimate for the actual code path

For BERT in this repo:
- Baseline FP32/FP16 run dense, so runtime-aware FLOPs = dense-equivalent FLOPs
- ETP-inspired 4:12 also runs dense in bert_sst2.py, so runtime-aware FLOPs stay dense
- ETP-inspired 2:4 runs sparse on masked MLP linears, so runtime-aware FLOPs are reduced there

The FLOP count is matmul-dominant:
- all nn.Linear modules
- self-attention score/value matmuls

This is the standard practical way to compare transformer inference FLOPs.
"""

from __future__ import annotations

import argparse
import json
import os
import random
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


def build_eval_batch(
    tokenizer,
    texts: List[str],
    labels: torch.Tensor,
    *,
    batch_size: int,
    max_length: int,
) -> Dict[str, torch.Tensor]:
    enc = tokenizer(
        texts[:batch_size],
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    return {
        "input_ids": enc["input_ids"].contiguous(),
        "attention_mask": enc["attention_mask"].contiguous(),
        "labels": labels[:batch_size].contiguous(),
    }


def load_mask_ratios(masks_path: str) -> Dict[str, float]:
    bundle = torch.load(masks_path, map_location="cpu")
    masks = bundle.get("masks", {}) if isinstance(bundle, dict) else {}
    if not isinstance(masks, dict) or not masks:
        raise ValueError(f"No masks found in {masks_path!r}.")
    ratios: Dict[str, float] = {}
    for name, mask in masks.items():
        mask_f = mask.to(dtype=torch.float32)
        ratios[name] = float(mask_f.mean().item())
    return ratios


def collect_trace(
    model: torch.nn.Module,
    batch_inputs: Dict[str, torch.Tensor],
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    linear_trace: Dict[str, Dict[str, Any]] = {}
    attn_trace: Dict[str, Dict[str, Any]] = {}
    handles = []

    def make_linear_hook(name: str):
        def hook(module: nn.Linear, inputs, output):
            x = inputs[0]
            positions = int(np.prod(x.shape[:-1])) if x.dim() >= 2 else 1
            dense_flops = 2.0 * positions * module.in_features * module.out_features
            linear_trace[name] = {
                "input_shape": tuple(int(v) for v in x.shape),
                "output_shape": tuple(int(v) for v in output.shape),
                "in_features": int(module.in_features),
                "out_features": int(module.out_features),
                "positions": int(positions),
                "dense_flops": float(dense_flops),
            }
        return hook

    def make_attn_prehook(name: str):
        def prehook(module, inputs):
            hidden_states = inputs[0]
            batch_size, seq_length, hidden_size = [int(v) for v in hidden_states.shape]
            heads = int(module.num_attention_heads)
            head_dim = int(module.attention_head_size)
            dense_flops = 4.0 * batch_size * seq_length * seq_length * hidden_size
            attn_trace[name] = {
                "hidden_shape": tuple(int(v) for v in hidden_states.shape),
                "num_heads": heads,
                "head_dim": head_dim,
                "dense_flops": float(dense_flops),
            }
        return prehook

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            handles.append(module.register_forward_hook(make_linear_hook(name)))
        elif module.__class__.__name__ == "BertSelfAttention":
            handles.append(module.register_forward_pre_hook(make_attn_prehook(name)))

    try:
        with torch.inference_mode():
            _ = model(
                input_ids=batch_inputs["input_ids"],
                attention_mask=batch_inputs["attention_mask"],
            )
    finally:
        for handle in handles:
            handle.remove()

    return linear_trace, attn_trace


def build_variant_specs() -> List[Dict[str, Any]]:
    return [
        {
            "name": "BERT Base (Baseline)",
            "dtype": "fp32",
            "runtime_path": "dense",
            "mask_ratios": {},
            "notes": "Dense FP32 baseline.",
        },
        {
            "name": "BERT Base FP16",
            "dtype": "fp16",
            "runtime_path": "dense",
            "mask_ratios": {},
            "notes": "Dense FP16 baseline. FLOPs are the same as FP32; only dtype changes.",
        },
        {
            "name": "ETP-inspired 2:4 FP16",
            "dtype": "fp16",
            "runtime_path": "sparse_2to4_runtime",
            "mask_ratios": load_mask_ratios(os.path.join(ETP_2OF4_MODEL_DIR, "prune_masks.pt")),
            "notes": "Exact-mask sparse runtime on masked MLP linears.",
        },
        {
            "name": "ETP-inspired 4:12",
            "dtype": "fp32",
            "runtime_path": "dense_with_zero_weights",
            "mask_ratios": load_mask_ratios(os.path.join(ETP_MODEL_DIR, "prune_masks.pt")),
            "notes": "Dense runtime with zeros in masked MLP linears; no sparse backend in bert_sst2.py.",
        },
    ]


def summarize_variant_flops(
    variant: Dict[str, Any],
    linear_trace: Dict[str, Dict[str, Any]],
    attn_trace: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    dense_linear_flops = 0.0
    effective_linear_flops = 0.0
    runtime_linear_flops = 0.0
    masked_dense_linear_flops = 0.0
    masked_effective_linear_flops = 0.0

    mask_ratios = variant["mask_ratios"]
    masked_module_count = 0

    for module_name, info in linear_trace.items():
        dense_flops = float(info["dense_flops"])
        dense_linear_flops += dense_flops

        ratio = float(mask_ratios.get(module_name, 1.0))
        if module_name in mask_ratios:
            masked_module_count += 1
            masked_dense_linear_flops += dense_flops
            masked_effective_linear_flops += dense_flops * ratio

        effective_linear_flops += dense_flops * ratio

        if variant["runtime_path"] == "sparse_2to4_runtime":
            runtime_linear_flops += dense_flops * ratio
        else:
            runtime_linear_flops += dense_flops

    attention_matmul_flops = float(sum(info["dense_flops"] for info in attn_trace.values()))

    dense_total_flops = dense_linear_flops + attention_matmul_flops
    effective_total_flops = effective_linear_flops + attention_matmul_flops
    runtime_total_flops = runtime_linear_flops + attention_matmul_flops

    return {
        "Model Variant": variant["name"],
        "Dtype": variant["dtype"],
        "Runtime Path": variant["runtime_path"],
        "Notes": variant["notes"],
        "Masked Linear Module Count": int(masked_module_count),
        "Dense Linear FLOPs": float(dense_linear_flops),
        "Effective Nonzero Linear FLOPs": float(effective_linear_flops),
        "Runtime-Aware Linear FLOPs": float(runtime_linear_flops),
        "Masked Dense Linear FLOPs": float(masked_dense_linear_flops),
        "Masked Effective Linear FLOPs": float(masked_effective_linear_flops),
        "Attention Matmul FLOPs": float(attention_matmul_flops),
        "Dense-Equivalent Total FLOPs": float(dense_total_flops),
        "Effective Nonzero Total FLOPs": float(effective_total_flops),
        "Runtime-Aware Total FLOPs": float(runtime_total_flops),
        "Effective vs Dense Ratio": float(effective_total_flops / dense_total_flops),
        "Runtime-Aware vs Dense Ratio": float(runtime_total_flops / dense_total_flops),
    }


def add_common_metadata(
    row: Dict[str, Any],
    *,
    batch_size: int,
    seq_length: int,
    linear_trace: Dict[str, Dict[str, Any]],
    attn_trace: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    row = dict(row)
    row["Batch Size"] = int(batch_size)
    row["Sequence Length"] = int(seq_length)
    row["Linear Module Count"] = int(len(linear_trace))
    row["Attention Module Count"] = int(len(attn_trace))
    row["Dense-Equivalent FLOPs per Sample"] = float(row["Dense-Equivalent Total FLOPs"] / batch_size)
    row["Effective Nonzero FLOPs per Sample"] = float(row["Effective Nonzero Total FLOPs"] / batch_size)
    row["Runtime-Aware FLOPs per Sample"] = float(row["Runtime-Aware Total FLOPs"] / batch_size)
    return row


def parse_args():
    parser = argparse.ArgumentParser(description="Measure runtime-aware forward FLOPs for selected BERT SST-2 variants.")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--output-json", type=str, default="output/bert_sst2_flops_benchmark.json")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.batch_size < 1 or args.max_length < 1:
        raise ValueError("batch-size and max-length must be >= 1.")

    set_seed(SEED)

    dataset = load_dataset(DATASET_NAME)
    val_texts = list(dataset["validation"]["sentence"])
    val_labels = torch.tensor(dataset["validation"]["label"], dtype=torch.long)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    batch = build_eval_batch(
        tokenizer,
        val_texts,
        val_labels,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )

    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME).eval()
    linear_trace, attn_trace = collect_trace(model, batch)
    del model

    batch_size = int(batch["input_ids"].shape[0])
    seq_length = int(batch["input_ids"].shape[1])

    results = []
    for variant in build_variant_specs():
        row = summarize_variant_flops(variant, linear_trace, attn_trace)
        row = add_common_metadata(
            row,
            batch_size=batch_size,
            seq_length=seq_length,
            linear_trace=linear_trace,
            attn_trace=attn_trace,
        )
        results.append(row)
        print(json.dumps(row, indent=2))

    payload = {
        "summary": {
            "Batch Size": batch_size,
            "Sequence Length": seq_length,
            "Linear Module Count": len(linear_trace),
            "Attention Module Count": len(attn_trace),
            "Definition": (
                "FLOPs are matmul-dominant forward FLOPs: all nn.Linear layers plus "
                "self-attention score/value matmuls. Runtime-aware FLOPs follow the "
                "actual bert_sst2.py execution path for each variant."
            ),
        },
        "results": results,
    }

    os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, allow_nan=True)
    print(f"\nSaved JSON to {args.output_json}")


if __name__ == "__main__":
    main()
