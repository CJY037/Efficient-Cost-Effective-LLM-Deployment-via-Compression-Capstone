#!/usr/bin/env python3
"""
Llama-3-8B three-way forward benchmark on real WikiText-2 validation windows.

This is the real-text counterpart to `llama3_8b_sparse_throughput_copy.py`.
It keeps the same dense-FP32 / dense-FP16 / sparse-FP16 2:4 comparison, but
replaces synthetic random token batches with tokenized WikiText-2 validation
windows grouped into fixed-shape batches.

Example:
  CUDA_VISIBLE_DEVICES=1 python -u llama3_8b_sparse_throughput_wikitext2.py \
      --batches 1,4,8,16 --seq 2048 --stride 512 \
      --warmup-batches 2 --measured-batches 8 --repeats 5
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import llama3_8b_wikitext2 as base
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from sparse_2to4 import supports_sparse_semi_structured
from torch.sparse import SparseSemiStructuredTensor


def resolve_model_source(model_name: str, hf_cache_dir: str) -> str:
    model_path = Path(model_name)
    if model_path.exists():
        return str(model_path)

    repo_cache_dir = Path(hf_cache_dir) / f"models--{model_name.replace('/', '--')}"
    ref_path = repo_cache_dir / "refs" / "main"
    if not ref_path.is_file():
        return model_name

    snapshot_name = ref_path.read_text().strip()
    if not snapshot_name:
        return model_name

    snapshot_dir = repo_cache_dir / "snapshots" / snapshot_name
    has_config = (snapshot_dir / "config.json").is_file()
    has_tokenizer = any((snapshot_dir / name).is_file() for name in ("tokenizer.json", "tokenizer.model"))
    has_weights = any(
        (snapshot_dir / name).is_file()
        for name in ("model.safetensors", "model.safetensors.index.json", "pytorch_model.bin")
    )
    if has_config and has_tokenizer and has_weights:
        return str(snapshot_dir)
    return model_name


def load_model(model_source: str, dtype: torch.dtype, load_kwargs: Dict[str, Any]) -> AutoModelForCausalLM:
    model = AutoModelForCausalLM.from_pretrained(
        model_source,
        torch_dtype=dtype,
        device_map={"": 0},
        low_cpu_mem_usage=True,
        **load_kwargs,
    )
    model.config.use_cache = False
    model.eval()
    return model


def load_tokenizer(tokenizer_name: str, hf_cache_dir: str):
    tokenizer_source = resolve_model_source(tokenizer_name, hf_cache_dir)
    kwargs: Dict[str, Any] = {"use_fast": True}
    if tokenizer_source == tokenizer_name:
        kwargs["cache_dir"] = hf_cache_dir
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, **kwargs)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer, tokenizer_source


def build_real_batches(
    tokenizer_name: str,
    hf_cache_dir: str,
    seq: int,
    stride: int,
    batches: List[int],
    warmup_batches: int,
    measured_batches: int,
) -> Tuple[Dict[int, List[Dict[str, torch.Tensor]]], Dict[str, Any]]:
    tokenizer, tokenizer_source = load_tokenizer(tokenizer_name, hf_cache_dir)
    ds = load_dataset(
        base.DATASET_NAME,
        base.DATASET_CONFIG,
        split=base.EVAL_SPLIT,
        cache_dir=hf_cache_dir,
    )
    text_field = base.get_text_field(ds)
    corpus_text = base.build_corpus_text(ds, text_field)
    encoded = tokenizer(corpus_text, return_tensors="pt")
    windows = base.make_windows(encoded["input_ids"], seq, stride)
    full_windows = [w["input_ids"].squeeze(0).contiguous() for w in windows if int(w["input_ids"].size(1)) == seq]

    if not full_windows:
        raise ValueError(f"No full-length WikiText-2 windows were created for seq={seq} stride={stride}.")

    total_batches_needed = warmup_batches + measured_batches
    batches_by_size: Dict[int, List[Dict[str, torch.Tensor]]] = {}
    for batch_size in batches:
        available_batches = len(full_windows) // batch_size
        if available_batches < total_batches_needed:
            raise ValueError(
                f"Need at least {total_batches_needed} real batches for batch={batch_size}, "
                f"but only {available_batches} are available from {len(full_windows)} windows. "
                "Reduce --measured-batches, reduce --warmup-batches, reduce --batches, "
                "or lower --seq / increase overlap with smaller --stride."
            )

        real_batches: List[Dict[str, torch.Tensor]] = []
        for batch_idx in range(total_batches_needed):
            start = batch_idx * batch_size
            chunk = full_windows[start:start + batch_size]
            input_ids = torch.stack(chunk, dim=0)
            attention_mask = torch.ones_like(input_ids, dtype=torch.long)
            real_batches.append(
                {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                }
            )
        batches_by_size[batch_size] = real_batches

    dataset_info = {
        "dataset_name": base.DATASET_NAME,
        "dataset_config": base.DATASET_CONFIG,
        "split": base.EVAL_SPLIT,
        "text_field": text_field,
        "tokenizer_name": tokenizer_name,
        "tokenizer_source": tokenizer_source,
        "corpus_bytes": len(corpus_text.encode("utf-8")),
        "token_count": int(encoded["input_ids"].numel()),
        "windows_total": len(windows),
        "full_windows_total": len(full_windows),
        "seq": seq,
        "stride": stride,
        "warmup_batches": warmup_batches,
        "measured_batches": measured_batches,
    }
    return batches_by_size, dataset_info


@torch.inference_mode()
def bench_real_batches(
    model: torch.nn.Module,
    real_batches: List[Dict[str, torch.Tensor]],
    device: torch.device,
    *,
    warmup_batches: int,
    repeats: int,
) -> Dict[str, float]:
    prepared = [
        {k: v.to(device=device, non_blocking=True) for k, v in batch.items()}
        for batch in real_batches
    ]
    warmup_slice = prepared[:warmup_batches]
    measured_slice = prepared[warmup_batches:]

    for batch in warmup_slice:
        model(**batch)
    base.sync_cuda()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    total_measured_batches = len(measured_slice) * repeats
    total_tokens = sum(int(batch["input_ids"].numel()) for batch in measured_slice) * repeats

    start.record()
    for _ in range(repeats):
        for batch in measured_slice:
            model(**batch)
    end.record()
    base.sync_cuda()

    elapsed_ms = start.elapsed_time(end)
    elapsed_s = elapsed_ms / 1000.0
    return {
        "ms_per_iter": elapsed_ms / total_measured_batches,
        "ms_per_token": elapsed_ms / total_tokens,
        "tok_per_s": total_tokens / elapsed_s,
        "measured_batches": total_measured_batches,
        "tokens_measured": total_tokens,
    }


def sweep_batches(
    model: torch.nn.Module,
    batches: List[int],
    real_batches_by_size: Dict[int, List[Dict[str, torch.Tensor]]],
    device: torch.device,
    *,
    warmup_batches: int,
    repeats: int,
    label: str,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for batch_size in batches:
        seq = int(real_batches_by_size[batch_size][0]["input_ids"].size(1))
        try:
            stats = bench_real_batches(
                model,
                real_batches_by_size[batch_size],
                device,
                warmup_batches=warmup_batches,
                repeats=repeats,
            )
            rows.append(
                {
                    "batch": batch_size,
                    "seq": seq,
                    "M": batch_size * seq,
                    "ms_per_iter": stats["ms_per_iter"],
                    "ms_per_token": stats["ms_per_token"],
                    "tok_per_s": stats["tok_per_s"],
                    "measured_batches": stats["measured_batches"],
                    "tokens_measured": stats["tokens_measured"],
                }
            )
        except torch.cuda.OutOfMemoryError:
            print(f"  [{label}] batch={batch_size} M={batch_size * seq}  OOM")
            base.cleanup()
    return rows


def _attach(prefix: str, row: Dict[str, Any], src: Dict[str, Any]) -> None:
    row[f"{prefix}_tok_per_s"] = src["tok_per_s"]
    row[f"{prefix}_ms_per_token"] = src["ms_per_token"]
    row[f"{prefix}_ms_per_iter"] = src["ms_per_iter"]


def merge_three(
    dense_fp32: List[Dict[str, Any]],
    dense_fp16: List[Dict[str, Any]],
    sparse_fp16: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    by32 = {r["batch"]: r for r in dense_fp32}
    by16 = {r["batch"]: r for r in dense_fp16}
    bysp = {r["batch"]: r for r in sparse_fp16}
    all_batches = sorted(set(by32) | set(by16) | set(bysp))
    merged: List[Dict[str, Any]] = []
    for batch_size in all_batches:
        ref = by32.get(batch_size) or by16.get(batch_size) or bysp[batch_size]
        row: Dict[str, Any] = {"batch": batch_size, "seq": ref["seq"], "M": ref["M"]}
        if batch_size in by32:
            _attach("dense_fp32", row, by32[batch_size])
        if batch_size in by16:
            _attach("dense_fp16", row, by16[batch_size])
        if batch_size in bysp:
            _attach("sparse_fp16", row, bysp[batch_size])
        d16 = by16.get(batch_size)
        sp = bysp.get(batch_size)
        d32 = by32.get(batch_size)
        if d16 and sp and d16["tok_per_s"] > 0:
            row["speedup_sparse_vs_dense_fp16"] = sp["tok_per_s"] / d16["tok_per_s"]
        if d32 and sp and d32["tok_per_s"] > 0:
            row["speedup_sparse_vs_dense_fp32"] = sp["tok_per_s"] / d32["tok_per_s"]
        merged.append(row)
    return merged


def _cell_tok(row: Dict[str, Any], prefix: str) -> str:
    key = f"{prefix}_tok_per_s"
    return f"{row[key]:11.1f}" if key in row else f"{'OOM':>11}"


def _cell_ms(row: Dict[str, Any], prefix: str) -> str:
    key = f"{prefix}_ms_per_token"
    return f"{row[key]:8.4f}" if key in row else f"{'—':>8}"


def print_combined_table(rows: List[Dict[str, Any]]) -> None:
    hdr = (
        f"\n{'batch':>5} {'M':>7} | "
        f"{'FP32 dense':^22} | {'FP16 dense':^22} | {'FP16 2:4 sparse':^22} | "
        f"{'sp/d16':>7} {'sp/d32':>7}"
    )
    sub = (
        f"{'':>5} {'':>7} | "
        f"{'tok/s':>11} {'ms/tok':>8} | "
        f"{'tok/s':>11} {'ms/tok':>8} | "
        f"{'tok/s':>11} {'ms/tok':>8} | "
        f"{'':>7} {'':>7}"
    )
    print(hdr)
    print(sub)
    print("-" * len(hdr))
    for row in rows:
        spd16 = row.get("speedup_sparse_vs_dense_fp16")
        spd32 = row.get("speedup_sparse_vs_dense_fp32")
        s16 = f"{spd16:6.3f}x" if spd16 is not None else f"{'—':>7}"
        s32 = f"{spd32:6.3f}x" if spd32 is not None else f"{'—':>7}"
        print(
            f"{row['batch']:5d} {row['M']:7d} | "
            f"{_cell_tok(row, 'dense_fp32')} {_cell_ms(row, 'dense_fp32')} | "
            f"{_cell_tok(row, 'dense_fp16')} {_cell_ms(row, 'dense_fp16')} | "
            f"{_cell_tok(row, 'sparse_fp16')} {_cell_ms(row, 'sparse_fp16')} | "
            f"{s16} {s32}"
        )


def run_dense_sweep(
    model_source: str,
    dtype: torch.dtype,
    load_kwargs: Dict[str, Any],
    batches: List[int],
    real_batches_by_size: Dict[int, List[Dict[str, torch.Tensor]]],
    *,
    warmup_batches: int,
    repeats: int,
) -> List[Dict[str, Any]]:
    label = f"dense-{dtype}".replace("torch.", "")
    print(f"\n=== Dense {dtype} ===")
    model = load_model(model_source, dtype, load_kwargs)
    device = base.get_model_input_device(model)
    rows = sweep_batches(
        model,
        batches,
        real_batches_by_size,
        device,
        warmup_batches=warmup_batches,
        repeats=repeats,
        label=label,
    )
    del model
    base.cleanup()
    return rows


def parse_args():
    parser = argparse.ArgumentParser(
        description="Dense FP32 + FP16 vs exact-mask FP16 2:4 sparse throughput on WikiText-2 validation windows"
    )
    parser.add_argument("--dense-model-name", default=base.MODEL_NAME)
    parser.add_argument("--sparse-model-name", default="./llama3_8b_c4_2of4_cuda")
    parser.add_argument("--tokenizer-name", default=None)
    parser.add_argument("--hf-cache-dir", default="./.hf_cache")
    parser.add_argument("--prune-masks-path", default=None)
    parser.add_argument("--sparse-dtype", default="float16", choices=["float16", "bfloat16"])
    parser.add_argument("--backend", default="cusparselt", choices=["cusparselt", "cutlass"])
    parser.add_argument("--skip-dense-fp32", action="store_true")
    parser.add_argument("--skip-dense-fp16", action="store_true")
    parser.add_argument("--skip-sparse", action="store_true")
    parser.add_argument("--seq", type=int, default=2048)
    parser.add_argument("--stride", type=int, default=512)
    parser.add_argument("--batches", default="1,4,8,16")
    parser.add_argument("--warmup-batches", type=int, default=2)
    parser.add_argument("--measured-batches", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-json",
        default="output/llama3_8b_c4_2of4_cuda_sparse_throughput_wikitext2_3way.json",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required.")
    if not supports_sparse_semi_structured() and not args.skip_sparse:
        raise RuntimeError("2:4 sparse semi-structured API not available.")

    SparseSemiStructuredTensor._FORCE_CUTLASS = args.backend == "cutlass"
    sparse_dtype = getattr(torch, args.sparse_dtype)
    batches = [int(b.strip()) for b in args.batches.split(",") if b.strip()]
    tokenizer_name = args.tokenizer_name or args.dense_model_name

    base.set_seed(args.seed)
    base.configure_torch()
    script_start = time.perf_counter()

    dense_model_source = resolve_model_source(args.dense_model_name, args.hf_cache_dir)
    sparse_model_source = resolve_model_source(args.sparse_model_name, args.hf_cache_dir)
    dense_load_kwargs: Dict[str, Any] = {}
    sparse_load_kwargs: Dict[str, Any] = {}
    if dense_model_source == args.dense_model_name:
        dense_load_kwargs["cache_dir"] = args.hf_cache_dir
    if sparse_model_source == args.sparse_model_name:
        sparse_load_kwargs["cache_dir"] = args.hf_cache_dir

    print(f"Device : {torch.cuda.get_device_name(0)}  cap={torch.cuda.get_device_capability()}")
    print(f"Dense model  : {args.dense_model_name}")
    if dense_model_source != args.dense_model_name:
        print(f"Dense source : {dense_model_source}")
    print(f"Sparse model : {args.sparse_model_name}")
    if sparse_model_source != args.sparse_model_name:
        print(f"Sparse source: {sparse_model_source}")
    print(f"Tokenizer    : {tokenizer_name}")
    print(f"Compare: dense FP32 | dense FP16 | sparse FP16 2:4  (backend={args.backend})")
    print(f"Seq={args.seq}  stride={args.stride}  batches={batches}")
    print(f"Warmup batches={args.warmup_batches}  measured batches={args.measured_batches}  repeats={args.repeats}")

    real_batches_by_size, dataset_info = build_real_batches(
        tokenizer_name,
        args.hf_cache_dir,
        args.seq,
        args.stride,
        batches,
        args.warmup_batches,
        args.measured_batches,
    )
    print(
        "WikiText-2 validation windows: "
        f"{dataset_info['full_windows_total']} full windows "
        f"(token_count={dataset_info['token_count']}, text_field={dataset_info['text_field']})"
    )

    dense_fp32_rows: List[Dict[str, Any]] = []
    dense_fp16_rows: List[Dict[str, Any]] = []
    if not args.skip_dense_fp32:
        dense_fp32_rows = run_dense_sweep(
            dense_model_source,
            torch.float32,
            dense_load_kwargs,
            batches,
            real_batches_by_size,
            warmup_batches=args.warmup_batches,
            repeats=args.repeats,
        )
    if not args.skip_dense_fp16:
        dense_fp16_rows = run_dense_sweep(
            dense_model_source,
            torch.float16,
            dense_load_kwargs,
            batches,
            real_batches_by_size,
            warmup_batches=args.warmup_batches,
            repeats=args.repeats,
        )

    sparse_rows: List[Dict[str, Any]] = []
    conversion_info: Dict[str, Any] = {}
    if not args.skip_sparse:
        print(f"\n=== Sparse 2:4 ({args.sparse_dtype}, exact prune masks) ===")
        sparse_model = load_model(sparse_model_source, sparse_dtype, sparse_load_kwargs)
        device = base.get_model_input_device(sparse_model)
        masks_path = base.resolve_prune_masks_path(args.sparse_model_name, args.prune_masks_path)
        masks = base.load_prune_masks(masks_path)
        conversion_info = base.apply_exact_2to4_runtime_masks(sparse_model, masks, sparse_dtype)
        conversion_info["mode"] = "prune_masks"
        conversion_info["masks_path"] = masks_path
        print(f"Applied prune_masks: {conversion_info['converted_count']} layers")

        sparse_rows = sweep_batches(
            sparse_model,
            batches,
            real_batches_by_size,
            device,
            warmup_batches=args.warmup_batches,
            repeats=args.repeats,
            label=f"sparse-{args.sparse_dtype}",
        )
        del sparse_model
        base.cleanup()

    merged = merge_three(dense_fp32_rows, dense_fp16_rows, sparse_rows)
    print_combined_table(merged)

    out = {
        "device": torch.cuda.get_device_name(0),
        "capability": list(torch.cuda.get_device_capability()),
        "dense_model_name": args.dense_model_name,
        "dense_model_source": dense_model_source,
        "sparse_model_name": args.sparse_model_name,
        "sparse_model_source": sparse_model_source,
        "tokenizer_name": tokenizer_name,
        "dataset": dataset_info,
        "sparse_dtype": args.sparse_dtype,
        "backend": args.backend,
        "seq": args.seq,
        "stride": args.stride,
        "batches_requested": batches,
        "warmup_batches": args.warmup_batches,
        "measured_batches": args.measured_batches,
        "repeats": args.repeats,
        "conversion": conversion_info,
        "dense_fp32_rows": dense_fp32_rows,
        "dense_fp16_rows": dense_fp16_rows,
        "sparse_fp16_rows": sparse_rows,
        "combined_rows": merged,
        "total_runtime_s": time.perf_counter() - script_start,
        "max_gpu_memory_gb": base.get_max_gpu_memory_gb(),
    }

    os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
    with open(args.output_json, "w") as handle:
        json.dump(out, handle, indent=2)

    print(f"\nWrote {args.output_json}")
    print(f"Total runtime: {base.format_elapsed_time(out['total_runtime_s'])}")
    print("Fair sparsity speedup column: sp/d16 (sparse FP16 vs dense FP16)")


if __name__ == "__main__":
    main()
