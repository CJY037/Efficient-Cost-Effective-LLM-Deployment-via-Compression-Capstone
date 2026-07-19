#!/usr/bin/env python3
"""
Two-stage Llama-3.2 pruning: ETP regularization + LLM-Pruner structural removal.

Stage 1 applies Exponential Torque regularization on dependency-graph groups from
LLM-Pruner (GQA-aware attention + MLP neurons), using coupled parameter norms.

Stage 2 physically prunes the regularized model with LLM-Pruner's MetaPruner
(Taylor/magnitude/random importance on the same dependency groups).

Reference: "Towards Universal & Efficient Model Compression via Exponential
Torque Pruning" (Modi et al., arXiv:2506.22015).
"""

from __future__ import annotations

import argparse
import gc
import json
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, get_scheduler
from transformers.models.llama.modeling_llama import LlamaRMSNorm

ROOT_DIR = Path(__file__).resolve().parent
VENDOR_DIR = ROOT_DIR / "third_party" / "LLM-Pruner"
if VENDOR_DIR.exists():
    sys.path.insert(0, str(VENDOR_DIR))

try:
    import LLMPruner.torch_pruning as tp
    from LLMPruner.evaluator.ppl import PPLMetric
    from LLMPruner.pruner import hf_llama_pruner as llama_pruner
    from LLMPruner.utils.logger import LoggerWithDepth
except ImportError as exc:
    raise ImportError(
        "LLM-Pruner is not available. Run `bash scripts/setup_llm_pruner.sh` first."
    ) from exc

from llama32_llm_prune import (
    build_forward_prompts,
    count_parameters,
    dependency_forward_fn,
    get_calibration_batch,
    open_calibration_stream,
    resolve_device,
    resolve_dtype,
    run_taylor_backward,
    save_hf_checkpoint,
    set_random_seed,
    sync_llama_config,
)


def resolve_pivot_index(pivot_index: Optional[int], num_units: int) -> int:
    if pivot_index is None:
        return num_units // 2
    if not 0 <= pivot_index < num_units:
        raise ValueError(
            f"`pivot_index` must be in [0, {num_units - 1}], got {pivot_index}."
        )
    return pivot_index


def build_llama_pruner_kwargs(
    model: nn.Module,
    attn_layers: List[int],
    mlp_layers: List[int],
    importance: object,
    pruning_ratio: float,
    global_pruning: bool,
    iterative_steps: int,
) -> Dict[str, object]:
    return {
        "importance": importance,
        "global_pruning": global_pruning,
        "iterative_steps": iterative_steps,
        "ch_sparsity": pruning_ratio,
        "ignored_layers": [],
        "channel_groups": {},
        "consecutive_groups": {
            layer.self_attn.k_proj: layer.self_attn.head_dim
            for layer in model.model.layers
        },
        "customized_pruners": {
            LlamaRMSNorm: llama_pruner.hf_rmsnorm_pruner,
        },
        "root_module_types": None,
        "root_instances": [
            model.model.layers[i].self_attn.k_proj for i in attn_layers
        ] + [model.model.layers[i].mlp.gate_proj for i in mlp_layers],
    }


class Stage1CalibrationStream:
    """Stream C4/BookCorpus calibration batches without prefetching everything."""

    def __init__(
        self,
        dataset_name: str,
        tokenizer,
        num_examples: int,
        seq_len: int,
        device: torch.device,
    ) -> None:
        self.dataset_name = dataset_name
        self.tokenizer = tokenizer
        self.num_examples = num_examples
        self.seq_len = seq_len
        self.device = device
        self.source = open_calibration_stream(dataset_name)

    def next_batch(self) -> torch.Tensor:
        return get_calibration_batch(
            dataset_name=self.dataset_name,
            tokenizer=self.tokenizer,
            num_examples=self.num_examples,
            seq_len=self.seq_len,
            device=self.device,
            source=self.source,
        )


class ETPDependencyGroupRegularizer:
    """
    Exponential Torque regularizer over LLM-Pruner dependency groups.

    For each root pruning group, coupled parameter L2 norms are aggregated across
    dependent modules (same alignment as MagnitudeImportance), then multiplied by
    an exponential distance weight from a pivot:

        penalty_i = ||w_i||_2 * lambda_base ** (distance(i, pivot) / scale)

    Pivot distance can be measured in either:
    - fixed mode: absolute channel index distance
    - ranked mode: distance in a norm-sorted rank order (rank 0 = largest norm)
    """

    _OUT_CHANNEL_FNS = {
        tp.prune_linear_out_channels,
        llama_pruner.hf_linear_pruner.prune_out_channels,
    }
    _IN_CHANNEL_FNS = {
        tp.prune_linear_in_channels,
        llama_pruner.hf_linear_pruner.prune_in_channels,
    }
    _PIVOT_MODES = {"fixed", "ranked"}

    def __init__(
        self,
        pruner: tp.pruner.MetaPruner,
        lambda_base: float,
        pivot_index: Optional[int] = None,
        pivot_mode: str = "fixed",
        distance_scale: Optional[float] = None,
    ) -> None:
        if lambda_base <= 1.0:
            raise ValueError(f"`lambda_base` must be > 1.0, got {lambda_base}.")
        pivot_mode = pivot_mode.lower()
        if pivot_mode not in self._PIVOT_MODES:
            raise ValueError(
                f"Unsupported pivot_mode `{pivot_mode}`. "
                f"Expected one of {sorted(self._PIVOT_MODES)}."
            )
        self.pruner = pruner
        self.lambda_base = float(lambda_base)
        self.pivot_index = pivot_index
        self.pivot_mode = pivot_mode
        self.distance_scale = distance_scale
        self.groups = list(
            pruner.DG.get_all_groups(
                ignored_layers=pruner.ignored_layers,
                root_module_types=pruner.root_module_types,
                root_instances=pruner.root_instances,
            )
        )
        if not self.groups:
            raise RuntimeError("No dependency groups were found for ETP regularization.")

    def _align_group_norms(self, group_norms: List[torch.Tensor]) -> torch.Tensor:
        if not group_norms:
            raise RuntimeError("Encountered a dependency group with no supported norms.")
        min_size = min(norm.shape[0] for norm in group_norms)
        aligned: List[torch.Tensor] = []
        for norm in group_norms:
            if norm.shape[0] > min_size and norm.shape[0] % min_size == 0:
                norm = norm.view(norm.shape[0] // min_size, min_size).sum(0)
            elif norm.shape[0] != min_size:
                continue
            aligned.append(norm)
        if not aligned:
            raise RuntimeError("Failed to align dependency-group norms.")
        return torch.stack(aligned, dim=0).sum(0)

    def _collapse_units(
        self,
        group_imp: torch.Tensor,
        ch_groups: int,
        consecutive_groups: int,
    ) -> torch.Tensor:
        if ch_groups > 1:
            group_imp = group_imp[: len(group_imp) // ch_groups]
        if consecutive_groups > 1:
            group_imp = group_imp.view(-1, consecutive_groups).sum(1)
        return group_imp

    def _unit_positions(
        self,
        num_units: int,
        device: torch.device,
        dtype: torch.dtype,
        unit_norms: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.pivot_mode == "fixed":
            return torch.arange(num_units, device=device, dtype=dtype)

        if unit_norms is None:
            raise ValueError("Ranked pivot mode requires `unit_norms`.")
        if unit_norms.shape[0] != num_units:
            raise ValueError(
                f"Expected `unit_norms` with {num_units} units, got {unit_norms.shape[0]}."
            )
        order = torch.argsort(unit_norms, descending=True)
        ranks = torch.empty_like(unit_norms, dtype=dtype)
        ranks[order] = torch.arange(num_units, device=device, dtype=dtype)
        return ranks

    def _distance_weights(
        self,
        num_units: int,
        device: torch.device,
        dtype: torch.dtype,
        unit_norms: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        pivot = resolve_pivot_index(self.pivot_index, num_units)
        positions = self._unit_positions(
            num_units=num_units,
            device=device,
            dtype=dtype,
            unit_norms=unit_norms,
        )
        distances = (positions - float(pivot)).abs()
        if self.distance_scale is None:
            scale = max(float(num_units - 1), 1.0)
        else:
            scale = float(self.distance_scale)
        normalized = distances / scale
        return torch.pow(
            torch.tensor(self.lambda_base, device=device, dtype=dtype),
            normalized,
        )

    def _group_unit_norms(
        self,
        group,
        ch_groups: int,
        consecutive_groups: int,
    ) -> torch.Tensor:
        group_norms: List[torch.Tensor] = []
        for dep, idxs in group:
            idxs = sorted(idxs)
            layer = dep.target.module
            prune_fn = dep.handler

            if prune_fn in self._OUT_CHANNEL_FNS:
                local_norm = layer.weight[idxs].flatten(1).norm(dim=1)
            elif prune_fn in self._IN_CHANNEL_FNS:
                local_norm = layer.weight.norm(dim=0)[idxs]
            elif prune_fn == llama_pruner.hf_rmsnorm_pruner.prune_out_channels:
                local_norm = layer.weight[idxs].abs()
            elif prune_fn == tp.prune_embedding_out_channels:
                local_norm = layer.weight[:, idxs].norm(dim=0)
            else:
                continue
            group_norms.append(local_norm)

        group_imp = self._align_group_norms(group_norms)
        return self._collapse_units(group_imp, ch_groups, consecutive_groups)

    def penalty(self) -> Tuple[torch.Tensor, Dict[str, float]]:
        total = None
        per_group: Dict[str, float] = {}
        for group_index, group in enumerate(self.groups):
            ch_groups = self.pruner.get_channel_groups(group)
            consecutive_groups = self.pruner.get_consecutive_groups(group)
            unit_norms = self._group_unit_norms(group, ch_groups, consecutive_groups)
            distance_weights = self._distance_weights(
                num_units=unit_norms.shape[0],
                device=unit_norms.device,
                dtype=unit_norms.dtype,
                unit_norms=unit_norms,
            )
            group_loss = (unit_norms * distance_weights).mean()
            root_module = group[0][0].target.module
            group_name = f"{group_index}:{root_module.__class__.__name__}"
            per_group[group_name] = float(group_loss.detach().item())
            total = group_loss if total is None else total + group_loss
        if total is None:
            raise RuntimeError("ETP regularizer produced no penalty terms.")
        return total / len(self.groups), per_group


def forward_lm_loss(model: nn.Module, input_ids: torch.Tensor) -> torch.Tensor:
    outputs = model(input_ids=input_ids, labels=input_ids, use_cache=False)
    return outputs.loss


def run_stage1_etp_regularization(
    model: nn.Module,
    regularizer: ETPDependencyGroupRegularizer,
    calibration_stream: Stage1CalibrationStream,
    steps_per_epoch: int,
    optimizer: torch.optim.Optimizer,
    scheduler,
    beta: float,
    grad_clip_norm: float,
    epochs: int,
    logger: LoggerWithDepth,
) -> Dict[str, float]:
    model.train()
    last_stats = {
        "task_loss": float("nan"),
        "aux_loss": float("nan"),
        "total_loss": float("nan"),
        "steps": 0,
        "sequences": 0,
    }

    for epoch in range(epochs):
        epoch_task = 0.0
        epoch_aux = 0.0
        epoch_total = 0.0
        step_count = 0

        progress = tqdm(
            range(steps_per_epoch),
            desc=f"ETP stage1 epoch {epoch + 1}/{epochs}",
            leave=False,
        )
        for _ in progress:
            input_ids = calibration_stream.next_batch()
            optimizer.zero_grad(set_to_none=True)

            task_loss = forward_lm_loss(model, input_ids)
            aux_loss, _ = regularizer.penalty()
            total_loss = task_loss + beta * aux_loss
            if not torch.isfinite(total_loss):
                logger.log(
                    f"Skipping non-finite ETP step "
                    f"(task={float(task_loss.detach().item())}, aux={float(aux_loss.detach().item())})."
                )
                continue

            total_loss.backward()
            if grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            optimizer.step()
            scheduler.step()

            step_count += 1
            epoch_task += float(task_loss.detach().item())
            epoch_aux += float(aux_loss.detach().item())
            epoch_total += float(total_loss.detach().item())
            progress.set_postfix(
                task=f"{task_loss.item():.4f}",
                aux=f"{aux_loss.item():.4f}",
            )

        if step_count > 0:
            last_stats = {
                "task_loss": epoch_task / step_count,
                "aux_loss": epoch_aux / step_count,
                "total_loss": epoch_total / step_count,
                "steps": last_stats["steps"] + step_count,
                "sequences": last_stats["sequences"] + step_count * calibration_stream.num_examples,
            }
            logger.log(
                "ETP stage1 epoch "
                f"{epoch + 1}/{epochs}: task={last_stats['task_loss']:.4f}, "
                f"aux={last_stats['aux_loss']:.4f}, total={last_stats['total_loss']:.4f}, "
                f"steps={step_count}"
            )
    return last_stats


def run_llm_pruner_stage(
    model: nn.Module,
    pruner: tp.pruner.MetaPruner,
    pruner_type: str,
    tokenizer,
    args: argparse.Namespace,
    device: torch.device,
    logger: LoggerWithDepth,
) -> int:
    model.zero_grad()
    logger.log("Start LLM-Pruner structural pruning")

    for step in range(args.iterative_steps):
        if pruner_type == "taylor":
            example_prompts = get_calibration_batch(
                dataset_name=args.calibration_dataset,
                tokenizer=tokenizer,
                num_examples=args.num_examples,
                seq_len=args.calibration_seq_len,
                device=device,
            )
            logger.log(
                f"Taylor backward for iterative step {step + 1}/{args.iterative_steps}"
            )
            run_taylor_backward(
                model=model,
                example_prompts=example_prompts,
                taylor=args.taylor,
                num_examples=args.num_examples,
                logger=logger,
            )
        pruner.step()
        for layer in model.model.layers:
            layer.self_attn.num_heads = (
                layer.self_attn.q_proj.weight.shape[0] // layer.self_attn.head_dim
            )
            layer.self_attn.num_key_value_heads = (
                layer.self_attn.k_proj.weight.shape[0] // layer.self_attn.head_dim
            )
        after_step_params = count_parameters(model)
        logger.log(
            f"After prune iter {step + 1}/{args.iterative_steps}, "
            f"#parameters={after_step_params}"
        )
    return after_step_params


def main(args: argparse.Namespace) -> None:
    set_random_seed(args.seed)

    device = resolve_device(args.device)
    eval_device = resolve_device(args.eval_device)
    dtype = resolve_dtype(args.torch_dtype, device)

    log_root = Path(args.log_root)
    output_dir = Path(args.output_dir) if args.output_dir else log_root / "pruned_checkpoint"
    stage1_dir = Path(args.stage1_output_dir) if args.stage1_output_dir else log_root / "etp_regularized_checkpoint"
    log_root.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    stage1_dir.mkdir(parents=True, exist_ok=True)

    logger = LoggerWithDepth(
        env_name=args.save_ckpt_log_name,
        config=args.__dict__,
        root_dir=str(log_root),
        setup_sublogger=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model,
        use_fast=True,
        trust_remote_code=args.trust_remote_code,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs = {
        "trust_remote_code": args.trust_remote_code,
        "low_cpu_mem_usage": True,
        "attn_implementation": "eager",
    }
    if device.type == "cuda":
        model_kwargs["dtype"] = dtype
    model = AutoModelForCausalLM.from_pretrained(args.base_model, **model_kwargs)
    if device.type != "cuda":
        model = model.to(dtype=torch.float32)
    model.to(device)
    if args.torch_dtype == "float16" or (args.torch_dtype == "auto" and dtype == torch.float16):
        model.half()

    if args.test_before_train:
        logger.log("Evaluating dense model before pruning...")
        model.to(eval_device)
        model.eval()
        before_ppl = PPLMetric(
            model,
            tokenizer,
            args.eval_datasets,
            args.max_seq_len,
            device=str(eval_device),
        )
        logger.log(f"PPL before pruning: {before_ppl}")
        model.to(device)

    pruner_type = args.pruner_type.lower()
    if pruner_type not in {"random", "l1", "l2", "taylor"}:
        raise ValueError(f"Unsupported pruner_type `{args.pruner_type}`.")

    for param in model.parameters():
        param.requires_grad_(True)
    before_params = count_parameters(model)

    if pruner_type == "random":
        importance = tp.importance.RandomImportance()
    elif pruner_type == "l1":
        importance = llama_pruner.MagnitudeImportance(p=1)
    elif pruner_type == "l2":
        importance = llama_pruner.MagnitudeImportance(p=2)
    else:
        importance = llama_pruner.TaylorImportance(
            group_reduction=args.grouping_strategy,
            taylor=args.taylor,
        )

    if not args.block_wise:
        raise ValueError("Llama support requires `--block_wise`.")

    attn_layers = list(range(args.block_attention_layer_start, args.block_attention_layer_end))
    mlp_layers = list(range(args.block_mlp_layer_start, args.block_mlp_layer_end))
    logger.log(f"Pruning attention layers: {attn_layers}")
    logger.log(f"Pruning MLP layers: {mlp_layers}")

    forward_prompts = build_forward_prompts(tokenizer, device)
    pruner_kwargs = build_llama_pruner_kwargs(
        model=model,
        attn_layers=attn_layers,
        mlp_layers=mlp_layers,
        importance=importance,
        pruning_ratio=args.pruning_ratio,
        global_pruning=args.global_pruning,
        iterative_steps=args.iterative_steps,
    )

    pruner = tp.pruner.MetaPruner(
        model,
        forward_prompts,
        forward_fn=dependency_forward_fn,
        **pruner_kwargs,
    )

    stage1_stats: Optional[Dict[str, float]] = None
    if not args.skip_etp_regularization and args.stage1_epochs > 0:
        total_stage1_sequences = (
            args.num_examples * args.stage1_steps_per_epoch * args.stage1_epochs
        )
        logger.log(
            f"Stage 1: ETP regularization on dependency groups "
            f"(beta={args.beta}, lambda_base={args.lambda_base}, "
            f"pivot_mode={args.pivot_mode}, "
            f"sequences={total_stage1_sequences})"
        )
        regularizer = ETPDependencyGroupRegularizer(
            pruner=pruner,
            lambda_base=args.lambda_base,
            pivot_index=args.pivot_index,
            pivot_mode=args.pivot_mode,
            distance_scale=args.distance_scale,
        )
        logger.log(f"Stage 1: regularizing {len(regularizer.groups)} dependency groups")
        calibration_stream = Stage1CalibrationStream(
            dataset_name=args.calibration_dataset,
            tokenizer=tokenizer,
            num_examples=args.num_examples,
            seq_len=args.calibration_seq_len,
            device=device,
        )
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.stage1_learning_rate,
            weight_decay=args.weight_decay,
        )
        total_steps = args.stage1_epochs * args.stage1_steps_per_epoch
        scheduler = get_scheduler(
            name=args.lr_scheduler,
            optimizer=optimizer,
            num_warmup_steps=args.warmup_steps,
            num_training_steps=total_steps,
        )
        stage1_stats = run_stage1_etp_regularization(
            model=model,
            regularizer=regularizer,
            calibration_stream=calibration_stream,
            steps_per_epoch=args.stage1_steps_per_epoch,
            optimizer=optimizer,
            scheduler=scheduler,
            beta=args.beta,
            grad_clip_norm=args.grad_clip_norm,
            epochs=args.stage1_epochs,
            logger=logger,
        )
        del regularizer, calibration_stream
        model.zero_grad(set_to_none=True)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.log(f"Saving ETP-regularized checkpoint to {stage1_dir}")
        save_hf_checkpoint(model=model, tokenizer=tokenizer, output_dir=stage1_dir)
    elif args.skip_etp_regularization:
        logger.log("Skipping Stage 1 ETP regularization (`--skip_etp_regularization`).")
    else:
        logger.log("Skipping Stage 1 ETP regularization (`--stage1_epochs` <= 0).")

    logger.log(f"Stage 2: LLM-Pruner structural pruning (pruner_type={pruner_type})")
    after_params = run_llm_pruner_stage(
        model=model,
        pruner=pruner,
        pruner_type=pruner_type,
        tokenizer=tokenizer,
        args=args,
        device=device,
        logger=logger,
    )

    sync_llama_config(model)
    keep_ratio = after_params / max(before_params, 1)
    logger.log(
        f"#Param before: {before_params}, after: {after_params}, "
        f"keep_ratio={keep_ratio:.4f}"
    )

    model.zero_grad()
    for name, param in model.named_parameters():
        if "weight" in name:
            param.grad = None
    del pruner
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if args.save_model:
        logger.log(f"Saving LLM-Pruner pickle checkpoint to {logger.best_checkpoint_path}")
        torch.save({"model": model, "tokenizer": tokenizer}, logger.best_checkpoint_path)

    logger.log(f"Saving HuggingFace checkpoint to {output_dir}")
    save_hf_checkpoint(model=model, tokenizer=tokenizer, output_dir=output_dir)

    summary = {
        "method": "etp_regularization_plus_llm_pruner",
        "base_model": args.base_model,
        "pruner_type": pruner_type,
        "pruning_ratio": args.pruning_ratio,
        "etp": {
            "lambda_base": args.lambda_base,
            "beta": args.beta,
            "pivot_index": args.pivot_index,
            "pivot_mode": args.pivot_mode,
            "distance_scale": args.distance_scale,
            "stage1_epochs": args.stage1_epochs,
            "stage1_steps_per_epoch": args.stage1_steps_per_epoch,
            "stage1_sequences_planned": (
                args.num_examples * args.stage1_steps_per_epoch * args.stage1_epochs
            ),
            "num_examples": args.num_examples,
            "calibration_seq_len": args.calibration_seq_len,
            "calibration_dataset": args.calibration_dataset,
            "stage1_learning_rate": args.stage1_learning_rate,
            "skipped": args.skip_etp_regularization,
            "final_stats": stage1_stats,
        },
        "parameters": {
            "before": before_params,
            "after": after_params,
            "keep_ratio": keep_ratio,
        },
        "layer_ranges": {
            "attention": [args.block_attention_layer_start, args.block_attention_layer_end],
            "mlp": [args.block_mlp_layer_start, args.block_mlp_layer_end],
        },
        "config_after_prune": sync_llama_config(model),
        "artifacts": {
            "hf_checkpoint_dir": str(output_dir),
            "etp_regularized_checkpoint_dir": str(stage1_dir),
            "llm_pruner_pickle": logger.best_checkpoint_path if args.save_model else None,
            "log_dir": logger.log_dir,
        },
    }

    if args.test_after_train:
        logger.log("Evaluating pruned model...")
        if eval_device.type == "cuda" and dtype in (torch.float16, torch.bfloat16):
            if dtype == torch.float16:
                model.half()
        model.to(eval_device)
        model.eval()
        after_ppl = PPLMetric(
            model,
            tokenizer,
            args.eval_datasets,
            args.max_seq_len,
            device=str(eval_device),
        )
        logger.log(f"PPL after pruning: {after_ppl}")
        summary["ppl_after_prune"] = after_ppl

    summary_path = output_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    logger.log(f"Wrote summary to {summary_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Llama-3.2 pruning with ETP regularization on LLM-Pruner dependency groups, "
            "followed by LLM-Pruner structural removal."
        )
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default="meta-llama/Llama-3.2-3B-Instruct",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="runs/llama32_etp_llmpruner/pruned_checkpoint",
    )
    parser.add_argument(
        "--stage1_output_dir",
        type=str,
        default="",
        help="Optional directory for the ETP-regularized dense checkpoint.",
    )
    parser.add_argument(
        "--save_ckpt_log_name",
        type=str,
        default="llama32_3b_etp_llmpruner",
    )
    parser.add_argument(
        "--log_root",
        type=str,
        default="runs/llama32_etp_llmpruner/prune_log",
    )
    parser.add_argument("--pruning_ratio", type=float, default=0.50)
    parser.add_argument(
        "--pruner_type",
        type=str,
        default="taylor",
        choices=["random", "l1", "l2", "taylor"],
    )
    parser.add_argument("--block_wise", action="store_true")
    parser.add_argument("--block_attention_layer_start", type=int, default=0)
    parser.add_argument("--block_attention_layer_end", type=int, default=28)
    parser.add_argument("--block_mlp_layer_start", type=int, default=0)
    parser.add_argument("--block_mlp_layer_end", type=int, default=28)
    parser.add_argument("--iterative_steps", type=int, default=1)
    parser.add_argument("--grouping_strategy", type=str, default="sum")
    parser.add_argument("--global_pruning", action="store_true")
    parser.add_argument(
        "--taylor",
        type=str,
        default="param_first",
        choices=["vectorize", "param_second", "param_first", "param_mix"],
    )
    parser.add_argument(
        "--num_examples",
        type=int,
        default=32,
        help="Calibration sequences per optimizer step (effective batch size).",
    )
    parser.add_argument(
        "--calibration_dataset",
        type=str,
        default="c4",
        choices=["c4", "bookcorpus"],
    )
    parser.add_argument("--calibration_seq_len", type=int, default=64)
    parser.add_argument("--max_seq_len", type=int, default=128)
    parser.add_argument(
        "--eval_datasets",
        nargs="+",
        default=["wikitext2"],
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--eval_device", type=str, default="cuda")
    parser.add_argument(
        "--torch_dtype",
        type=str,
        choices=["auto", "bfloat16", "float16", "float32"],
        default="auto",
    )
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--test_before_train", action="store_true")
    parser.add_argument("--test_after_train", action="store_true")
    parser.add_argument("--save_model", action="store_true")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--lambda_base", type=float, default=10.0)
    parser.add_argument("--beta", type=float, default=0.05)
    parser.add_argument("--pivot_index", type=int, default=None)
    parser.add_argument(
        "--pivot_mode",
        type=str,
        default="fixed",
        choices=["fixed", "ranked"],
        help=(
            "How to measure pivot distance. `fixed` uses channel index distance; "
            "`ranked` sorts units by current coupled L2 norm (rank 0 = largest) "
            "and applies distance in that rank order."
        ),
    )
    parser.add_argument(
        "--distance_scale",
        type=float,
        default=None,
        help="Distance normalizer; defaults to (num_units - 1) per group.",
    )
    parser.add_argument(
        "--stage1_epochs",
        type=int,
        default=1,
        help="Number of passes over stage1_steps_per_epoch.",
    )
    parser.add_argument(
        "--stage1_steps_per_epoch",
        type=int,
        default=1000,
        help="Fresh calibration batches per epoch (streamed, not prefetched).",
    )
    parser.add_argument("--stage1_learning_rate", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--grad_clip_norm", type=float, default=1.0)
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--lr_scheduler", type=str, default="cosine")
    parser.add_argument(
        "--skip_etp_regularization",
        action="store_true",
        help="Skip Stage 1 and run LLM-Pruner only.",
    )

    args = parser.parse_args()

    if not args.block_wise:
        args.block_wise = True
    if not args.save_model:
        args.save_model = True
    return args


if __name__ == "__main__":
    main(parse_args())
