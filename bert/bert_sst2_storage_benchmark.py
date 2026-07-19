#!/usr/bin/env python3
"""
Estimate theoretical parameter storage for selected BERT SST-2 variants.

Variants:
1) BERT Base (Baseline)   -> dense FP32
2) BERT Base FP16         -> dense FP16
3) ETP-inspired 2:4 FP16  -> theoretical compressed sparse storage
4) ETP-inspired 4:12      -> theoretical compressed sparse storage

This script is intentionally different from the runtime memory benchmark:
- it does NOT measure CUDA memory
- it estimates model parameter storage from parameter counts, nonzeros, and metadata

For sparse variants it reports:
- dense-equivalent storage at the same value precision
- ideal entropy lower bound for metadata (log2(n choose k) bits/group)
- practical fixed-code metadata estimate (ceil(log2(n choose k)) bits/group)

This is the standard analytical way to discuss compressed model storage.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from typing import Any, Dict, List, Optional

import torch
from transformers import AutoModelForSequenceClassification


MODEL_NAME = "textattack/bert-base-uncased-SST-2"
ETP_MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bert_sst2_etp_4of12")
ETP_2OF4_MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bert_sst2_etp_2of4")


def bits_to_bytes(bits: float) -> float:
    return float(bits) / 8.0


def bytes_to_mb(num_bytes: float) -> float:
    return float(num_bytes) / (1024 ** 2)


def bytes_to_gb(num_bytes: float) -> float:
    return float(num_bytes) / (1024 ** 3)


def load_total_param_count(model_ref: str, *, local_files_only: bool, torch_dtype=None) -> int:
    model = AutoModelForSequenceClassification.from_pretrained(
        model_ref,
        local_files_only=local_files_only,
        torch_dtype=torch_dtype,
    )
    try:
        return int(sum(int(p.numel()) for p in model.parameters()))
    finally:
        del model


def load_masks(masks_path: str) -> Dict[str, torch.Tensor]:
    bundle = torch.load(masks_path, map_location="cpu")
    masks = bundle.get("masks", {}) if isinstance(bundle, dict) else {}
    if not isinstance(masks, dict) or not masks:
        raise ValueError(f"No masks found in {masks_path!r}.")
    return masks


def infer_group_keep(mask_tensor: torch.Tensor) -> Optional[Dict[str, int]]:
    if mask_tensor.ndim != 2:
        return None
    rows, cols = [int(v) for v in mask_tensor.shape]
    for group_size, keep_k in ((4, 2), (12, 4)):
        usable_cols = (cols // group_size) * group_size
        if usable_cols == 0:
            continue
        grouped = mask_tensor[:, :usable_cols].to(dtype=torch.int32).view(rows, usable_cols // group_size, group_size)
        group_counts = grouped.sum(dim=2)
        if bool(torch.all(group_counts == keep_k).item()):
            return {"group_size": group_size, "keep_k": keep_k}
    return None


def analyze_mask_bundle(masks: Dict[str, torch.Tensor]) -> Dict[str, Any]:
    masked_param_count = 0
    masked_nonzero_count = 0
    full_group_count = 0
    tail_param_count = 0
    inferred_group_size = None
    inferred_keep_k = None

    for name, mask in masks.items():
        if mask.ndim != 2:
            masked_param_count += int(mask.numel())
            masked_nonzero_count += int(mask.sum().item())
            continue

        rows, cols = [int(v) for v in mask.shape]
        masked_param_count += rows * cols
        masked_nonzero_count += int(mask.sum().item())

        inferred = infer_group_keep(mask)
        if inferred is None:
            raise ValueError(f"Could not infer structured pattern for mask {name!r} with shape {tuple(mask.shape)}.")

        if inferred_group_size is None:
            inferred_group_size = inferred["group_size"]
            inferred_keep_k = inferred["keep_k"]
        elif (
            inferred_group_size != inferred["group_size"]
            or inferred_keep_k != inferred["keep_k"]
        ):
            raise ValueError(
                f"Inconsistent structured pattern: expected {inferred_keep_k}:{inferred_group_size}, "
                f"got {inferred['keep_k']}:{inferred['group_size']} in mask {name!r}."
            )

        usable_cols = (cols // inferred_group_size) * inferred_group_size
        full_group_count += rows * (usable_cols // inferred_group_size)
        tail_param_count += rows * (cols - usable_cols)

    if inferred_group_size is None or inferred_keep_k is None:
        raise ValueError("No structured 2D masks were found.")

    return {
        "masked_param_count": int(masked_param_count),
        "masked_nonzero_count": int(masked_nonzero_count),
        "full_group_count": int(full_group_count),
        "tail_param_count": int(tail_param_count),
        "group_size": int(inferred_group_size),
        "keep_k": int(inferred_keep_k),
    }


def estimate_sparse_storage(
    *,
    total_param_count: int,
    value_bits: int,
    mask_info: Dict[str, Any],
) -> Dict[str, Any]:
    masked_param_count = int(mask_info["masked_param_count"])
    masked_nonzero_count = int(mask_info["masked_nonzero_count"])
    full_group_count = int(mask_info["full_group_count"])
    group_size = int(mask_info["group_size"])
    keep_k = int(mask_info["keep_k"])

    unmasked_param_count = total_param_count - masked_param_count
    if unmasked_param_count < 0:
        raise ValueError("Masked parameter count exceeded total parameter count.")

    dense_bits = total_param_count * value_bits
    dense_bytes = bits_to_bytes(dense_bits)

    values_bits = (unmasked_param_count + masked_nonzero_count) * value_bits

    pattern_count = math.comb(group_size, keep_k)
    metadata_bits_entropy = full_group_count * math.log2(pattern_count)
    metadata_bits_fixed = full_group_count * math.ceil(math.log2(pattern_count))

    ideal_total_bits = values_bits + metadata_bits_entropy
    fixed_total_bits = values_bits + metadata_bits_fixed

    return {
        "Total Parameters": int(total_param_count),
        "Structured Masked Parameters": int(masked_param_count),
        "Structured Masked Nonzeros": int(masked_nonzero_count),
        "Unmasked Dense Parameters": int(unmasked_param_count),
        "Value Precision (bits)": int(value_bits),
        "Structured Pattern": f"{keep_k}:{group_size}",
        "Groups": int(full_group_count),
        "Tail Parameters Outside Full Groups": int(mask_info["tail_param_count"]),
        "Pattern Count per Group": int(pattern_count),
        "Entropy Metadata Bits per Group": float(math.log2(pattern_count)),
        "Fixed Metadata Bits per Group": int(math.ceil(math.log2(pattern_count))),
        "Dense-Equivalent Storage Bits": float(dense_bits),
        "Dense-Equivalent Storage Bytes": float(dense_bytes),
        "Dense-Equivalent Storage MB": float(bytes_to_mb(dense_bytes)),
        "Dense-Equivalent Storage GB": float(bytes_to_gb(dense_bytes)),
        "Sparse Value Storage Bits": float(values_bits),
        "Ideal Metadata Bits": float(metadata_bits_entropy),
        "Fixed Metadata Bits": float(metadata_bits_fixed),
        "Ideal Total Sparse Bits": float(ideal_total_bits),
        "Ideal Total Sparse Bytes": float(bits_to_bytes(ideal_total_bits)),
        "Ideal Total Sparse MB": float(bytes_to_mb(bits_to_bytes(ideal_total_bits))),
        "Ideal Total Sparse GB": float(bytes_to_gb(bits_to_bytes(ideal_total_bits))),
        "Fixed-Code Total Sparse Bits": float(fixed_total_bits),
        "Fixed-Code Total Sparse Bytes": float(bits_to_bytes(fixed_total_bits)),
        "Fixed-Code Total Sparse MB": float(bytes_to_mb(bits_to_bytes(fixed_total_bits))),
        "Fixed-Code Total Sparse GB": float(bytes_to_gb(bits_to_bytes(fixed_total_bits))),
        "Ideal Compression Ratio vs Dense": float(bits_to_bytes(ideal_total_bits) / dense_bytes),
        "Fixed-Code Compression Ratio vs Dense": float(bits_to_bytes(fixed_total_bits) / dense_bytes),
    }


def estimate_dense_storage(*, total_param_count: int, value_bits: int) -> Dict[str, Any]:
    total_bits = total_param_count * value_bits
    total_bytes = bits_to_bytes(total_bits)
    return {
        "Total Parameters": int(total_param_count),
        "Value Precision (bits)": int(value_bits),
        "Dense-Equivalent Storage Bits": float(total_bits),
        "Dense-Equivalent Storage Bytes": float(total_bytes),
        "Dense-Equivalent Storage MB": float(bytes_to_mb(total_bytes)),
        "Dense-Equivalent Storage GB": float(bytes_to_gb(total_bytes)),
    }


def build_variant_results() -> List[Dict[str, Any]]:
    base_total_params = load_total_param_count(MODEL_NAME, local_files_only=False)
    etp_2of4_total_params = load_total_param_count(ETP_2OF4_MODEL_DIR, local_files_only=True)
    etp_4of12_total_params = load_total_param_count(ETP_MODEL_DIR, local_files_only=True)

    masks_2of4 = load_masks(os.path.join(ETP_2OF4_MODEL_DIR, "prune_masks.pt"))
    masks_4of12 = load_masks(os.path.join(ETP_MODEL_DIR, "prune_masks.pt"))
    mask_info_2of4 = analyze_mask_bundle(masks_2of4)
    mask_info_4of12 = analyze_mask_bundle(masks_4of12)

    results = []

    baseline = estimate_dense_storage(total_param_count=base_total_params, value_bits=32)
    baseline.update(
        {
            "Model Variant": "BERT Base (Baseline)",
            "Runtime Path": "dense_fp32",
            "Notes": "Dense FP32 baseline. Storage estimate is straightforward parameter storage.",
        }
    )
    results.append(baseline)

    fp16 = estimate_dense_storage(total_param_count=base_total_params, value_bits=16)
    fp16.update(
        {
            "Model Variant": "BERT Base FP16",
            "Runtime Path": "dense_fp16",
            "Notes": "Dense FP16 baseline. Storage estimate assumes all parameters stored in FP16.",
        }
    )
    results.append(fp16)

    sparse_2of4 = estimate_sparse_storage(
        total_param_count=etp_2of4_total_params,
        value_bits=16,
        mask_info=mask_info_2of4,
    )
    sparse_2of4.update(
        {
            "Model Variant": "ETP-inspired 2:4 FP16",
            "Runtime Path": "sparse_2to4_runtime",
            "Notes": (
                "Theoretical compressed parameter storage using exact 2:4 masks. "
                "bert_sst2.py also uses a sparse runtime path for this variant."
            ),
        }
    )
    results.append(sparse_2of4)

    sparse_4of12 = estimate_sparse_storage(
        total_param_count=etp_4of12_total_params,
        value_bits=32,
        mask_info=mask_info_4of12,
    )
    sparse_4of12.update(
        {
            "Model Variant": "ETP-inspired 4:12",
            "Runtime Path": "dense_with_zero_weights",
            "Notes": (
                "Theoretical compressed parameter storage using exact 4:12 masks. "
                "bert_sst2.py does NOT run this variant with a sparse runtime backend."
            ),
        }
    )
    results.append(sparse_4of12)

    return results


def parse_args():
    parser = argparse.ArgumentParser(description="Estimate theoretical storage for selected BERT SST-2 variants.")
    parser.add_argument("--output-json", type=str, default="output/bert_sst2_storage_benchmark.json")
    return parser.parse_args()


def main():
    args = parse_args()
    results = build_variant_results()
    payload = {
        "summary": {
            "Definition": (
                "This script estimates model parameter storage, not runtime memory. "
                "Sparse variants include both an entropy lower bound and a fixed-code "
                "metadata estimate based on log2(n choose k) bits per structured group."
            ),
        },
        "results": results,
    }

    os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, allow_nan=True)
    print(json.dumps(payload, indent=2))
    print(f"\nSaved JSON to {args.output_json}")


if __name__ == "__main__":
    main()
