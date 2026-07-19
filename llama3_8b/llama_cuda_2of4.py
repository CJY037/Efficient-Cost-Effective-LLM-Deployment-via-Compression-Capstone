# Llama 3 8B structured 2:4 pruning.
# Regularization penalizes the pruned tail weights, with the largest pruned
# weights (closest to the keep/prune boundary) receiving the strongest penalty.
#
# Run multi-GPU:
#   torchrun --nproc_per_node=8 llama_cuda_2of4.py \
#       --max-train-samples 262144 \
#       --max-eval-samples 2048 \
#       --num-epochs 2 \
#       --post-prune-epochs 2
#
# Run single GPU:
#   CUDA_VISIBLE_DEVICES=0 python llama_cuda_2of4.py --max-train-samples 524288 --max-eval-samples 2048

# Single-GPU memory-safe default:
#   python llama_cuda_2of4.py --optimizer adafactor --train-scope prune-target --max-train-samples 4096 --max-eval-samples 512

import argparse
import math
import os
import random
from functools import partial
from typing import Dict, Iterable, Iterator, List, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
from datasets import load_dataset
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, IterableDataset
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup
from transformers.optimization import Adafactor

try:
    from torch.distributed.fsdp import (
        FullStateDictConfig,
        FullyShardedDataParallel as FSDP,
        MixedPrecision,
        ShardingStrategy,
        StateDictType,
    )
    from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
except ImportError:
    FullStateDictConfig = None
    FSDP = None
    MixedPrecision = None
    ShardingStrategy = None
    StateDictType = None
    transformer_auto_wrap_policy = None

try:
    from transformers.models.llama.modeling_llama import LlamaDecoderLayer
except ImportError:
    LlamaDecoderLayer = None


def add_bool_argument(parser: argparse.ArgumentParser, name: str, default: bool) -> None:
    dest = name.replace("-", "_")
    option = f"--{name}"

    if hasattr(argparse, "BooleanOptionalAction"):
        parser.add_argument(option, action=argparse.BooleanOptionalAction, default=default)
        return

    group = parser.add_mutually_exclusive_group()
    group.add_argument(option, dest=dest, action="store_true")
    group.add_argument(f"--no-{name}", dest=dest, action="store_false")
    parser.set_defaults(**{dest: default})


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model-name", default="meta-llama/Meta-Llama-3-8B")
    p.add_argument("--train-dataset", default="allenai/c4")
    p.add_argument("--train-config", default="en")
    p.add_argument("--train-split", default="train")
    p.add_argument("--eval-dataset", default="wikitext")
    p.add_argument("--eval-config", default="wikitext-2-raw-v1")
    p.add_argument("--eval-split", default="validation")
    add_bool_argument(p, "streaming", default=True)
    p.add_argument("--shuffle-buffer", type=int, default=10_000)
    p.add_argument("--max-train-samples", type=int, default=2048)
    p.add_argument(
        "--max-eval-samples",
        type=int,
        default=None,
        help="Maximum number of raw evaluation examples/rows to include in rolling WikiText-2 perplexity. Default: full split.",
    )
    p.add_argument("--seq-length", type=int, default=512, help="Training sequence length.")
    p.add_argument(
        "--eval-max-length",
        type=int,
        default=2048,
        help="Rolling perplexity context length for evaluation.",
    )
    p.add_argument(
        "--eval-stride",
        type=int,
        default=512,
        help="Rolling perplexity stride for evaluation.",
    )
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--eval-batch-size", type=int, default=1)
    p.add_argument("--grad-accum-steps", type=int, default=16)
    p.add_argument("--num-epochs", type=int, default=1)
    p.add_argument("--post-prune-epochs", type=int, default=1)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--post-prune-lr", type=float, default=None)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--optimizer", default="adafactor", choices=["adafactor", "adamw"])
    p.add_argument("--train-scope", default="prune-target", choices=["all", "prune-target"])
    p.add_argument("--warmup-ratio", type=float, default=0.03)
    p.add_argument("--group-size", type=int, default=4)
    p.add_argument("--keep-k", type=int, default=2)
    p.add_argument("--rank-gamma", type=float, default=1.0)
    p.add_argument("--reg-strength", type=float, default=1e-6)
    p.add_argument("--reg-every-steps", type=int, default=20)
    p.add_argument("--reg-max-layers", type=int, default=8)
    p.add_argument("--prune-target", default="mlp", choices=["mlp", "attention", "all"])
    p.add_argument("--dtype", default="bfloat16", choices=["float32", "float16", "bfloat16"])
    p.add_argument("--distributed-strategy", default="ddp", choices=["ddp", "fsdp"])
    add_bool_argument(p, "gradient-checkpointing", default=True)
    p.add_argument("--skip-initial-eval", action="store_true")
    p.add_argument("--save-dir", default="./llama3_8b_c4_2of4_cuda")
    p.add_argument("--hf-cache-dir", default="./.hf_cache")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main_print(*args, **kwargs):
    if not dist.is_initialized() or dist.get_rank() == 0:
        print(*args, **kwargs)


def resolve_dtype(name: str):
    return {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[name]


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def is_main():
    return (not dist.is_initialized()) or dist.get_rank() == 0


def get_rank_world():
    if dist.is_initialized():
        return dist.get_rank(), dist.get_world_size()
    return 0, 1


def text_from_example(example):
    text = example.get("text", "")
    return text if isinstance(text, str) else ""


class TokenizedTextStream(IterableDataset):
    def __init__(self, source, tokenizer, seq_length, max_samples, rank, world_size):
        self.source = source
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        self.max_samples = max_samples
        self.rank = rank
        self.world_size = world_size

    def __iter__(self):
        emitted = 0
        accepted = 0
        for example in self.source:
            text = text_from_example(example).strip()
            if not text:
                continue
            if accepted % self.world_size != self.rank:
                accepted += 1
                continue
            accepted += 1

            enc = self.tokenizer(
                text,
                truncation=True,
                max_length=self.seq_length,
                padding="max_length",
                return_tensors="pt",
            )
            input_ids = enc["input_ids"][0]
            attention_mask = enc["attention_mask"][0]
            labels = input_ids.clone()
            labels[attention_mask == 0] = -100

            yield {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
            }

            emitted += 1
            if self.max_samples is not None and emitted >= self.max_samples:
                break


def make_dataset(args, tokenizer, train, rank, world_size):
    if train:
        ds = load_dataset(
            args.train_dataset,
            args.train_config,
            split=args.train_split,
            streaming=args.streaming,
        )
        if args.streaming:
            ds = ds.shuffle(buffer_size=args.shuffle_buffer, seed=args.seed)
        max_samples = math.ceil(args.max_train_samples / world_size) if args.max_train_samples is not None else None
    else:
        ds = load_dataset(
            args.eval_dataset,
            args.eval_config,
            split=args.eval_split,
            streaming=args.streaming,
        )
        max_samples = math.ceil(args.max_eval_samples / world_size) if args.max_eval_samples is not None else None
    return TokenizedTextStream(ds, tokenizer, args.seq_length, max_samples, rank, world_size)


def batch_count(max_samples, batch_size, world_size=1):
    if max_samples is None:
        return None
    return math.ceil(math.ceil(max_samples / world_size) / batch_size)


def get_text_field(ds):
    for key in ("text", "sentence", "content"):
        if key in ds.column_names:
            return key
    raise KeyError(f"Could not find a text field in columns: {ds.column_names}")


def build_corpus_text(ds, text_field: str):
    texts = ["" if x is None else str(x) for x in ds[text_field]]
    if not texts:
        raise ValueError("Evaluation corpus is empty.")
    return "\n\n".join(texts)


def make_windows(input_ids: torch.Tensor, max_length: int, stride: int):
    seq_len = input_ids.size(1)
    prev_end = 0
    windows = []

    for begin_loc in range(0, seq_len, stride):
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end
        window_ids = input_ids[:, begin_loc:end_loc]
        windows.append(
            {
                "begin_loc": int(begin_loc),
                "end_loc": int(end_loc),
                "trg_len": int(trg_len),
                "input_ids": window_ids,
            }
        )
        prev_end = end_loc
        if end_loc == seq_len:
            break

    return windows


def build_target_ids(input_ids: torch.Tensor, trg_len: int):
    target_ids = input_ids.clone()
    target_ids[:, :-trg_len] = -100
    return target_ids


def count_loss_tokens(target_ids: torch.Tensor) -> int:
    valid_labels = int((target_ids != -100).sum().item())
    batch_size = int(target_ids.size(0))
    return max(valid_labels - batch_size, 0)


def build_eval_windows(args, tokenizer, model):
    ds = load_dataset(
        args.eval_dataset,
        args.eval_config,
        split=args.eval_split,
        streaming=False,
    )
    if args.max_eval_samples is not None:
        ds = ds.select(range(min(args.max_eval_samples, len(ds))))

    text_field = get_text_field(ds)
    corpus_text = build_corpus_text(ds, text_field)
    encodings = tokenizer(corpus_text, return_tensors="pt", add_special_tokens=False)

    context_limit = getattr(unwrap_model(model).config, "max_position_embeddings", args.eval_max_length)
    effective_max_length = min(args.eval_max_length, int(context_limit))
    windows = make_windows(encodings.input_ids, effective_max_length, args.eval_stride)
    return windows, {
        "text_field": text_field,
        "corpus_examples": len(ds),
        "corpus_characters": len(corpus_text),
        "effective_max_length": effective_max_length,
    }


def unwrap_model(model):
    while True:
        if hasattr(model, "module"):
            model = model.module
            continue
        if FSDP is not None and isinstance(model, FSDP):
            model = model._fsdp_wrapped_module
            continue
        return model


def qualify_module_name(prefix: str, name: str) -> str:
    if not prefix:
        return name
    if not name:
        return prefix
    return f"{prefix}.{name}"


def is_fsdp_model(model: nn.Module) -> bool:
    return FSDP is not None and isinstance(model, FSDP)


def is_prunable_linear(name: str, module: nn.Module, target: str) -> bool:
    if not isinstance(module, nn.Linear):
        return False
    if any(blocked in name for blocked in ["lm_head", "embed_tokens"]):
        return False
    if target == "mlp":
        return ".mlp." in name
    if target == "attention":
        return ".self_attn." in name
    return ".layers." in name


def get_prunable_linears(model: nn.Module, target: str):
    model = unwrap_model(model)
    return [(name, module) for name, module in model.named_modules() if is_prunable_linear(name, module, target)]


def configure_trainable_parameters(model: nn.Module, train_scope: str, prune_target: str) -> Tuple[int, int]:
    base_model = unwrap_model(model)

    if train_scope == "all":
        for param in base_model.parameters():
            param.requires_grad_(True)
    elif train_scope == "prune-target":
        for param in base_model.parameters():
            param.requires_grad_(False)
        for _, module in get_prunable_linears(base_model, prune_target):
            for param in module.parameters(recurse=False):
                param.requires_grad_(True)
    else:
        raise ValueError(f"Unsupported train scope: {train_scope}")

    total = 0
    trainable = 0
    for param in base_model.parameters():
        count = param.numel()
        total += count
        if param.requires_grad:
            trainable += count
    return trainable, total


def get_fsdp_pruning_units(model: nn.Module):
    if not is_fsdp_model(model):
        return []

    units = []
    root = unwrap_model(model)
    for name, module in root.named_modules():
        if is_fsdp_model(module) and LlamaDecoderLayer is not None and isinstance(unwrap_model(module), LlamaDecoderLayer):
            units.append((name, module))
    return units if units else [("", model)]


def split_into_full_groups(weight_2d: torch.Tensor, group_size: int):
    rows, cols = weight_2d.shape
    usable_cols = (cols // group_size) * group_size
    grouped = weight_2d[:, :usable_cols].contiguous().view(rows, usable_cols // group_size, group_size)
    return grouped, usable_cols


def rank_regularization_for_weight(weight_2d: torch.Tensor, group_size: int, keep_k: int, gamma: float):
    grouped, usable_cols = split_into_full_groups(weight_2d, group_size)
    if usable_cols == 0:
        return weight_2d.new_zeros(())

    abs_grouped = grouped.abs()

    # Sort descending by magnitude.
    # The kept set is ranks 0..keep_k-1.
    # The pruned tail is ranks keep_k..group_size-1.
    sorted_idx = torch.sort(abs_grouped, dim=-1, descending=True).indices
    prune_idx = sorted_idx[..., keep_k:]
    pruned_vals = torch.gather(abs_grouped, dim=-1, index=prune_idx)

    # Strongest penalty on the largest pruned weights, i.e. closest to the cutoff.
    # This corresponds to a reversed weighting over the tail.
    penalty_weights = torch.pow(
        weight_2d.new_tensor(1.01),
        gamma * torch.arange(group_size - keep_k, 0, -1, device=weight_2d.device, dtype=weight_2d.dtype),
    ).view(1, 1, -1)

    return (pruned_vals * penalty_weights).mean()


def model_rank_regularization(model: nn.Module, target: str, group_size: int, keep_k: int, gamma: float, max_layers: int):
    if is_fsdp_model(model):
        terms = []
        units = get_fsdp_pruning_units(model)

        for unit_name, unit in units:
            recurse = unit is model
            with FSDP.summon_full_params(unit, recurse=recurse, writeback=False):
                for local_name, module in unit.named_modules():
                    full_name = qualify_module_name(unit_name, local_name)
                    if not is_prunable_linear(full_name, module, target):
                        continue
                    terms.append(rank_regularization_for_weight(module.weight, group_size, keep_k, gamma))
                    if max_layers > 0 and len(terms) >= max_layers:
                        break
            if max_layers > 0 and len(terms) >= max_layers:
                break

        if not terms:
            return torch.tensor(0.0, device=next(model.parameters()).device)
        return torch.stack(terms).mean()

    layers = get_prunable_linears(model, target)
    if max_layers > 0:
        layers = layers[:max_layers]
    terms = [rank_regularization_for_weight(mod.weight, group_size, keep_k, gamma) for _, mod in layers]
    if not terms:
        return torch.tensor(0.0, device=next(unwrap_model(model).parameters()).device)
    return torch.stack(terms).mean()


@torch.no_grad()
def build_hard_masks(model, target: str, group_size: int, keep_k: int):
    masks = {}
    total_params = 0
    total_zeroed = 0

    if is_fsdp_model(model):
        for unit_name, unit in get_fsdp_pruning_units(model):
            recurse = unit is model
            with FSDP.summon_full_params(unit, recurse=recurse, writeback=False):
                for local_name, mod in unit.named_modules():
                    full_name = qualify_module_name(unit_name, local_name)
                    if not is_prunable_linear(full_name, mod, target):
                        continue

                    w = mod.weight.data
                    grouped, usable_cols = split_into_full_groups(w, group_size)
                    mask = torch.ones(w.shape, dtype=torch.bool, device=w.device)

                    if usable_cols > 0:
                        mags = grouped.abs()
                        topk_idx = torch.topk(mags, k=keep_k, dim=-1).indices
                        grouped_mask = torch.zeros(grouped.shape, dtype=torch.bool, device=w.device)
                        grouped_mask.scatter_(-1, topk_idx, True)
                        mask[:, :usable_cols] = grouped_mask.view(w.shape[0], usable_cols)

                    masks[full_name] = mask.cpu()
                    total_params += mask.numel()
                    total_zeroed += int((~mask).sum().item())

        stats = {
            "total_prunable_weights": total_params,
            "total_zeroed_weights": total_zeroed,
            "global_sparsity": total_zeroed / max(total_params, 1),
        }
        return masks, stats

    for name, mod in get_prunable_linears(model, target):
        w = mod.weight.data
        grouped, usable_cols = split_into_full_groups(w, group_size)
        mask = torch.ones(w.shape, dtype=torch.bool, device=w.device)

        if usable_cols > 0:
            mags = grouped.abs()
            topk_idx = torch.topk(mags, k=keep_k, dim=-1).indices
            grouped_mask = torch.zeros(grouped.shape, dtype=torch.bool, device=w.device)
            grouped_mask.scatter_(-1, topk_idx, True)
            mask[:, :usable_cols] = grouped_mask.view(w.shape[0], usable_cols)

        masks[name] = mask.cpu()
        total_params += mask.numel()
        total_zeroed += int((~mask).sum().item())

    stats = {
        "total_prunable_weights": total_params,
        "total_zeroed_weights": total_zeroed,
        "global_sparsity": total_zeroed / max(total_params, 1),
    }
    return masks, stats


@torch.no_grad()
def enforce_masks(model, masks):
    if is_fsdp_model(model):
        for unit_name, unit in get_fsdp_pruning_units(model):
            recurse = unit is model
            with FSDP.summon_full_params(unit, recurse=recurse, writeback=True):
                for local_name, module in unit.named_modules():
                    full_name = qualify_module_name(unit_name, local_name)
                    if full_name not in masks:
                        continue
                    mask = masks[full_name].to(device=module.weight.device, non_blocking=True)
                    module.weight.data.masked_fill_(mask.logical_not_(), 0)
        return

    modules = dict(unwrap_model(model).named_modules())
    for name, mask in masks.items():
        mask = mask.to(device=modules[name].weight.device, non_blocking=True)
        modules[name].weight.data.masked_fill_(mask.logical_not_(), 0)


@torch.no_grad()
def evaluate_ppl(model, windows, device, rank: int, world_size: int, desc):
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    local_window_count = len(range(rank, len(windows), world_size))
    progress = tqdm(range(rank, len(windows), world_size), total=local_window_count, desc=desc, disable=not is_main())

    for window_idx in progress:
        window = windows[window_idx]
        input_ids = window["input_ids"].to(device, non_blocking=True)
        target_ids = build_target_ids(input_ids, window["trg_len"])
        loss_tokens = count_loss_tokens(target_ids)
        if loss_tokens == 0:
            continue

        with torch.autocast("cuda", enabled=(device.type == "cuda")):
            outputs = model(input_ids=input_ids, labels=target_ids)

        total_loss += outputs.loss.item() * loss_tokens
        total_tokens += loss_tokens

        if is_main():
            mean_loss = total_loss / max(total_tokens, 1)
            progress.set_postfix(loss=f"{mean_loss:.4f}", ppl=f"{math.exp(min(mean_loss, 20)):.2f}")

    totals = torch.tensor([total_loss, total_tokens], dtype=torch.float64, device=device)
    if dist.is_initialized():
        dist.all_reduce(totals, op=dist.ReduceOp.SUM)

    mean_loss = totals[0].item() / max(totals[1].item(), 1.0)
    return mean_loss, math.exp(min(mean_loss, 20))


def build_optimizer(model: nn.Module, optimizer_name: str, lr: float, weight_decay: float):
    params = [param for param in model.parameters() if param.requires_grad]
    if not params:
        raise RuntimeError("No trainable parameters. Check --train-scope and --prune-target.")

    if optimizer_name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    if optimizer_name == "adafactor":
        return Adafactor(
            params,
            lr=lr,
            weight_decay=weight_decay,
            scale_parameter=False,
            relative_step=False,
            warmup_init=False,
        )
    raise ValueError(f"Unsupported optimizer: {optimizer_name}")


def train_phase(model, loader, device, *, phase_name, num_epochs, lr, weight_decay, warmup_ratio, args, reg_strength, masks=None):
    if num_epochs <= 0:
        return

    optimizer = build_optimizer(model, args.optimizer, lr, weight_decay)
    _, world_size = get_rank_world()
    steps_per_epoch = batch_count(args.max_train_samples, args.batch_size, world_size) or len(loader)
    optimizer_steps = max(1, math.ceil(steps_per_epoch / max(1, args.grad_accum_steps)))
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(warmup_ratio * optimizer_steps * num_epochs),
        num_training_steps=optimizer_steps * num_epochs,
    )

    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        running_loss = 0.0
        running_count = 0

        progress = tqdm(loader, total=steps_per_epoch, desc=f"{phase_name} epoch {epoch+1}/{num_epochs}", disable=not is_main())

        for step, batch in enumerate(progress):
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}

            with torch.autocast("cuda", enabled=(device.type == "cuda")):
                outputs = model(**batch)
                task_loss = outputs.loss
                if reg_strength and step % args.reg_every_steps == 0:
                    reg_loss = model_rank_regularization(
                        model, args.prune_target, args.group_size, args.keep_k, args.rank_gamma, args.reg_max_layers
                    )
                else:
                    reg_loss = task_loss.new_zeros(())
                loss = task_loss + reg_strength * reg_loss

            (loss / max(1, args.grad_accum_steps)).backward()

            if (step + 1) % max(1, args.grad_accum_steps) == 0:
                optimizer.step()
                if masks:
                    enforce_masks(model, masks)
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            running_loss += task_loss.item()
            running_count += 1
            avg_task = running_loss / max(running_count, 1)
            progress.set_postfix(task=f"{task_loss.item():.4f}", avg=f"{avg_task:.4f}", reg=f"{reg_loss.item():.4f}")

            if args.max_train_samples is not None and step + 1 >= steps_per_epoch:
                break

        if is_main():
            print(f"{phase_name} epoch {epoch+1} mean task loss: {running_loss / max(running_count, 1):.4f}")


def main():
    args = parse_args()

    distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    if args.distributed_strategy == "fsdp":
        if not distributed or int(os.environ.get("WORLD_SIZE", "1")) < 2:
            raise RuntimeError("--distributed-strategy fsdp requires torchrun with at least 2 processes.")
        if FSDP is None or transformer_auto_wrap_policy is None or LlamaDecoderLayer is None:
            raise RuntimeError("FSDP support is not available in this environment.")

    if distributed:
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
        rank, world_size = get_rank_world()
        device = torch.device(f"cuda:{local_rank}")
    else:
        rank, world_size = 0, 1
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    set_seed(args.seed + rank)
    dtype = resolve_dtype(args.dtype)

    if not distributed and args.optimizer == "adamw":
        main_print("Warning: AdamW full-model fine-tuning of Llama 3 8B usually OOMs on a single 40 GB GPU. Use --optimizer adafactor.")

    main_print(
        f"CUDA Llama run: world={world_size}, device={device}, strategy={args.distributed_strategy}, optimizer={args.optimizer}"
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, cache_dir=args.hf_cache_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=dtype,
        cache_dir=args.hf_cache_dir,
        low_cpu_mem_usage=True,
    )
    model.config.pad_token_id = tokenizer.pad_token_id

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
        if args.train_scope != "all" and hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()

    trainable_params, total_params = configure_trainable_parameters(model, args.train_scope, args.prune_target)
    main_print(
        f"Train scope: {args.train_scope}; trainable parameters: {trainable_params:,} / {total_params:,} "
        f"({100.0 * trainable_params / max(total_params, 1):.2f}%)"
    )

    if distributed and args.distributed_strategy == "fsdp":
        mixed_precision = MixedPrecision(param_dtype=dtype, reduce_dtype=dtype, buffer_dtype=dtype)
        auto_wrap_policy = partial(transformer_auto_wrap_policy, transformer_layer_cls={LlamaDecoderLayer})
        model = FSDP(
            model,
            auto_wrap_policy=auto_wrap_policy,
            device_id=device,
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            mixed_precision=mixed_precision,
            use_orig_params=True,
            limit_all_gathers=True,
        )
    else:
        model.to(device)
        if distributed:
            model = DDP(model, device_ids=[device.index], output_device=device.index, find_unused_parameters=False)

    prunable = get_prunable_linears(model, args.prune_target)
    main_print(f"Prune target: {args.prune_target}")
    main_print(f"Local prunable linear modules per rank: {len(prunable)}")

    train_data = make_dataset(args, tokenizer, True, rank, world_size)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, num_workers=0)
    eval_windows, eval_meta = build_eval_windows(args, tokenizer, model)
    train_total = batch_count(args.max_train_samples, args.batch_size, world_size)

    main_print(
        "Rolling WikiText-2 eval: "
        f"examples={eval_meta['corpus_examples']}, chars={eval_meta['corpus_characters']}, "
        f"max_length={eval_meta['effective_max_length']}, stride={args.eval_stride}, "
        f"windows={len(eval_windows)}"
    )

    if not args.skip_initial_eval:
        loss, ppl = evaluate_ppl(model, eval_windows, device, rank, world_size, "WikiText2 eval before training")
        main_print(f"WikiText2 before training: loss={loss:.4f}, ppl={ppl:.2f}")

    train_phase(
        model,
        train_loader,
        device,
        phase_name="regularized C4 fine-tune",
        num_epochs=args.num_epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        args=args,
        reg_strength=args.reg_strength,
    )

    loss, ppl = evaluate_ppl(model, eval_windows, device, rank, world_size, "WikiText2 eval before prune")
    main_print(f"WikiText2 before prune: loss={loss:.4f}, ppl={ppl:.2f}")

    masks, stats = build_hard_masks(model, args.prune_target, args.group_size, args.keep_k)
    enforce_masks(model, masks)
    main_print(f"Applied hard 2:4 masks: {stats}")

    loss, ppl = evaluate_ppl(model, eval_windows, device, rank, world_size, "WikiText2 eval after prune")
    main_print(f"WikiText2 after prune: loss={loss:.4f}, ppl={ppl:.2f}")

    post_lr = args.post_prune_lr if args.post_prune_lr is not None else args.lr
    train_phase(
        model,
        train_loader,
        device,
        phase_name="post-prune C4 fine-tune",
        num_epochs=args.post_prune_epochs,
        lr=post_lr,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        args=args,
        reg_strength=0.0,
        masks=masks,
    )

    loss, ppl = evaluate_ppl(model, eval_windows, device, rank, world_size, "Final WikiText2 eval")
    main_print(f"Final WikiText2: loss={loss:.4f}, ppl={ppl:.2f}")

    if is_main():
        os.makedirs(args.save_dir, exist_ok=True)

    if distributed and args.distributed_strategy == "fsdp":
        save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, save_policy):
            state_dict = model.state_dict()
        if is_main():
            torch.save(state_dict, os.path.join(args.save_dir, "model_state_dict.pt"))
            unwrap_model(model).config.save_pretrained(args.save_dir)
            tokenizer.save_pretrained(args.save_dir)
            torch.save({"masks": masks}, os.path.join(args.save_dir, "prune_masks.pt"))
            print(f"Saved FSDP checkpoint + masks to {args.save_dir}")
    elif is_main():
        unwrap_model(model).save_pretrained(args.save_dir)
        tokenizer.save_pretrained(args.save_dir)
        torch.save({"masks": masks}, os.path.join(args.save_dir, "prune_masks.pt"))
        print(f"Saved model + masks to {args.save_dir}")

    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
