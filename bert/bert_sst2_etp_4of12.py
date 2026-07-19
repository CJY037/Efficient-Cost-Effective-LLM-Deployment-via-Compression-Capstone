# BERT exact-tail pruning on SST-2.
# Regularization penalizes the pruned tail weights, with the largest pruned
# weights (closest to the keep/prune boundary) receiving the strongest penalty.
#
# Single GPU:
#   python bert_sst2_etp.py --num-epochs 3 --post-prune-epochs 1
#
# Multi-GPU:
#   torchrun --nproc_per_node=4 bert_sst2_etp.py --batch-size 16 --grad-accum-steps 2

import argparse
import math
import os
import random
from contextlib import nullcontext

import torch
import torch.distributed as dist
import torch.nn as nn
from datasets import load_dataset
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm.auto import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model-name", default="textattack/bert-base-uncased-SST-2")
    p.add_argument("--dataset-name", default="stanfordnlp/sst2")
    p.add_argument("--dataset-config", default=None)
    p.add_argument("--train-split", default="train")
    p.add_argument("--eval-split", default="validation")
    p.add_argument("--max-train-samples", type=int, default=None)
    p.add_argument("--max-eval-samples", type=int, default=None)
    p.add_argument("--max-length", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--eval-batch-size", type=int, default=64)
    p.add_argument("--grad-accum-steps", type=int, default=1)
    p.add_argument("--num-epochs", type=int, default=1)
    p.add_argument("--post-prune-epochs", type=int, default=1)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--post-prune-lr", type=float, default=None)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup-ratio", type=float, default=0.1)
    p.add_argument("--group-size", type=int, default=12)
    p.add_argument("--keep-k", type=int, default=4)
    p.add_argument("--rank-gamma", type=float, default=1.0)
    p.add_argument("--reg-strength", type=float, default=1e-6)
    p.add_argument("--reg-every-steps", type=int, default=20)
    p.add_argument("--reg-max-layers", type=int, default=8)
    p.add_argument("--prune-target", default="mlp", choices=["mlp", "attention", "all"])
    p.add_argument("--dtype", default="float32", choices=["float32", "float16", "bfloat16"])
    p.add_argument("--gradient-checkpointing", dest="gradient_checkpointing", action="store_true")
    p.add_argument("--no-gradient-checkpointing", dest="gradient_checkpointing", action="store_false")
    p.set_defaults(gradient_checkpointing=False)
    p.add_argument("--skip-initial-eval", action="store_true")
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--save-dir", default="./bert_sst2_etp_4of12")
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


def autocast_context(device: torch.device, amp_dtype):
    if device.type != "cuda" or amp_dtype is None:
        return nullcontext()
    return torch.autocast(device_type="cuda", dtype=amp_dtype)


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def is_main():
    return (not dist.is_initialized()) or dist.get_rank() == 0


def get_rank_world():
    if dist.is_initialized():
        return dist.get_rank(), dist.get_world_size()
    return 0, 1


def unwrap_model(model):
    return model.module if hasattr(model, "module") else model


def load_hf_split(dataset_name: str, dataset_config: str, split: str, cache_dir: str):
    if dataset_config:
        return load_dataset(dataset_name, dataset_config, split=split, cache_dir=cache_dir)
    return load_dataset(dataset_name, split=split, cache_dir=cache_dir)


def make_dataset(args, tokenizer, split_name: str, max_samples: int, train: bool, rank: int, world_size: int):
    dataset = load_hf_split(args.dataset_name, args.dataset_config, split_name, args.hf_cache_dir)

    if max_samples is not None:
        max_samples = min(max_samples, len(dataset))
        if train:
            dataset = dataset.shuffle(seed=args.seed)
        dataset = dataset.select(range(max_samples))

    if (not train) and world_size > 1:
        dataset = dataset.shard(num_shards=world_size, index=rank, contiguous=True)

    def tokenize_batch(batch):
        return tokenizer(
            batch["sentence"],
            truncation=True,
            max_length=args.max_length,
            padding="max_length",
        )

    keep_columns = {"label"}
    remove_columns = [name for name in dataset.column_names if name not in keep_columns]
    dataset = dataset.map(
        tokenize_batch,
        batched=True,
        remove_columns=remove_columns,
        desc=f"Tokenizing {split_name}",
    )
    dataset = dataset.rename_column("label", "labels")

    tensor_columns = [name for name in tokenizer.model_input_names if name in dataset.column_names]
    tensor_columns.append("labels")
    dataset.set_format(type="torch", columns=tensor_columns)
    return dataset


def make_loader(args, dataset, batch_size: int, train: bool, rank: int, world_size: int):
    sampler = None
    shuffle = train

    if train and world_size > 1:
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            seed=args.seed,
            drop_last=False,
        )
        shuffle = False

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return loader, sampler


def is_attention_linear(name: str) -> bool:
    return any(
        token in name
        for token in (
            ".attention.self.query",
            ".attention.self.key",
            ".attention.self.value",
            ".attention.output.dense",
        )
    )


def is_mlp_linear(name: str) -> bool:
    return ".intermediate.dense" in name or (
        ".output.dense" in name and ".attention.output.dense" not in name
    )


def is_prunable_linear(name: str, module: nn.Module, target: str) -> bool:
    if not isinstance(module, nn.Linear):
        return False
    if ".encoder.layer." not in name:
        return False
    if any(blocked in name for blocked in ("embeddings", "pooler", "classifier")):
        return False
    if target == "mlp":
        return is_mlp_linear(name)
    if target == "attention":
        return is_attention_linear(name)
    return is_mlp_linear(name) or is_attention_linear(name)


def get_prunable_linears(model: nn.Module, target: str):
    model = unwrap_model(model)
    return [(name, module) for name, module in model.named_modules() if is_prunable_linear(name, module, target)]


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
    penalty_weights = torch.pow(
        weight_2d.new_tensor(1.01),
        gamma * torch.arange(group_size - keep_k, 0, -1, device=weight_2d.device, dtype=weight_2d.dtype),
    ).view(1, 1, -1)

    return (pruned_vals * penalty_weights).mean()


def model_rank_regularization(model: nn.Module, target: str, group_size: int, keep_k: int, gamma: float, max_layers: int):
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
    modules = dict(unwrap_model(model).named_modules())
    for name, mask in masks.items():
        module = modules[name]
        device_mask = mask.to(device=module.weight.device, non_blocking=True)
        module.weight.data.masked_fill_(device_mask.logical_not(), 0)


@torch.no_grad()
def evaluate_classifier(model, loader, device, amp_dtype, desc):
    model.eval()
    total_loss = 0.0
    total_correct = 0.0
    total_examples = 0.0
    progress = tqdm(loader, total=len(loader), desc=desc, disable=not is_main())

    for batch in progress:
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        labels = batch["labels"]

        with autocast_context(device, amp_dtype):
            outputs = model(**batch)

        preds = outputs.logits.argmax(dim=-1)
        batch_size = labels.size(0)
        total_loss += outputs.loss.item() * batch_size
        total_correct += (preds == labels).sum().item()
        total_examples += batch_size

        if is_main():
            mean_loss = total_loss / max(total_examples, 1.0)
            mean_acc = total_correct / max(total_examples, 1.0)
            progress.set_postfix(loss=f"{mean_loss:.4f}", acc=f"{100.0 * mean_acc:.2f}")

    totals = torch.tensor([total_loss, total_correct, total_examples], dtype=torch.float64, device=device)
    if dist.is_initialized():
        dist.all_reduce(totals, op=dist.ReduceOp.SUM)

    mean_loss = totals[0].item() / max(totals[2].item(), 1.0)
    accuracy = totals[1].item() / max(totals[2].item(), 1.0)
    return mean_loss, accuracy


def train_phase(
    model,
    loader,
    device,
    *,
    phase_name,
    num_epochs,
    lr,
    weight_decay,
    warmup_ratio,
    args,
    reg_strength,
    amp_dtype,
    masks=None,
):
    if num_epochs <= 0:
        return

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    steps_per_epoch = len(loader)
    optimizer_steps_per_epoch = max(1, math.ceil(steps_per_epoch / max(1, args.grad_accum_steps)))
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(warmup_ratio * optimizer_steps_per_epoch * num_epochs),
        num_training_steps=optimizer_steps_per_epoch * num_epochs,
    )

    for epoch in range(num_epochs):
        model.train()
        if isinstance(loader.sampler, DistributedSampler):
            loader.sampler.set_epoch(epoch)

        optimizer.zero_grad(set_to_none=True)
        running_loss = 0.0
        running_correct = 0.0
        running_examples = 0
        steps_since_update = 0

        progress = tqdm(
            loader,
            total=steps_per_epoch,
            desc=f"{phase_name} epoch {epoch + 1}/{num_epochs}",
            disable=not is_main(),
        )

        for step, batch in enumerate(progress):
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            labels = batch["labels"]

            with autocast_context(device, amp_dtype):
                outputs = model(**batch)
                task_loss = outputs.loss
                if reg_strength and step % args.reg_every_steps == 0:
                    reg_loss = model_rank_regularization(
                        model,
                        args.prune_target,
                        args.group_size,
                        args.keep_k,
                        args.rank_gamma,
                        args.reg_max_layers,
                    )
                else:
                    reg_loss = task_loss.new_zeros(())
                loss = task_loss + reg_strength * reg_loss

            (loss / max(1, args.grad_accum_steps)).backward()
            steps_since_update += 1

            if steps_since_update >= max(1, args.grad_accum_steps):
                optimizer.step()
                if masks:
                    enforce_masks(model, masks)
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                steps_since_update = 0

            preds = outputs.logits.argmax(dim=-1)
            running_loss += task_loss.item() * labels.size(0)
            running_correct += (preds == labels).sum().item()
            running_examples += labels.size(0)

            avg_loss = running_loss / max(running_examples, 1)
            avg_acc = running_correct / max(running_examples, 1)
            progress.set_postfix(
                task=f"{task_loss.item():.4f}",
                avg=f"{avg_loss:.4f}",
                acc=f"{100.0 * avg_acc:.2f}",
                reg=f"{reg_loss.item():.4f}",
            )

        if steps_since_update > 0:
            optimizer.step()
            if masks:
                enforce_masks(model, masks)
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

        totals = torch.tensor([running_loss, running_correct, running_examples], dtype=torch.float64, device=device)
        if dist.is_initialized():
            dist.all_reduce(totals, op=dist.ReduceOp.SUM)

        epoch_loss = totals[0].item() / max(totals[2].item(), 1.0)
        epoch_acc = totals[1].item() / max(totals[2].item(), 1.0)
        main_print(f"{phase_name} epoch {epoch + 1}: loss={epoch_loss:.4f}, acc={100.0 * epoch_acc:.2f}%")


def main():
    args = parse_args()
    if not 0 < args.keep_k < args.group_size:
        raise ValueError("--keep-k must be in the range 1..group-size-1.")

    distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    if distributed:
        if not torch.cuda.is_available():
            raise RuntimeError("Distributed execution requires CUDA for NCCL.")
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
        rank, world_size = get_rank_world()
        device = torch.device(f"cuda:{local_rank}")
    else:
        rank, world_size = 0, 1
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    set_seed(args.seed + rank)
    weight_dtype = resolve_dtype(args.dtype)
    if device.type != "cuda" and weight_dtype != torch.float32:
        raise ValueError("Non-float32 dtypes require CUDA in this script.")
    amp_dtype = weight_dtype if device.type == "cuda" and weight_dtype != torch.float32 else None

    main_print(f"BERT SST-2 run: world={world_size}, device={device}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, cache_dir=args.hf_cache_dir)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        cache_dir=args.hf_cache_dir,
    )

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    model.to(device)
    if weight_dtype != torch.float32:
        model.to(dtype=weight_dtype)

    if distributed:
        model = DDP(model, device_ids=[device.index], output_device=device.index, find_unused_parameters=False)

    prunable = get_prunable_linears(model, args.prune_target)
    main_print(f"Prune target: {args.prune_target}")
    main_print(f"Local prunable linear modules per rank: {len(prunable)}")

    train_data = make_dataset(args, tokenizer, args.train_split, args.max_train_samples, True, rank, world_size)
    eval_data = make_dataset(args, tokenizer, args.eval_split, args.max_eval_samples, False, rank, world_size)
    train_loader, _ = make_loader(args, train_data, args.batch_size, True, rank, world_size)
    eval_loader, _ = make_loader(args, eval_data, args.eval_batch_size, False, rank, world_size)

    main_print(f"Train examples: {len(train_data)}")
    main_print(f"Eval examples on this rank: {len(eval_data)}")

    if not args.skip_initial_eval:
        loss, acc = evaluate_classifier(model, eval_loader, device, amp_dtype, "SST-2 eval before training")
        main_print(f"SST-2 before training: loss={loss:.4f}, acc={100.0 * acc:.2f}%")

    train_phase(
        model,
        train_loader,
        device,
        phase_name="regularized SST-2 fine-tune",
        num_epochs=args.num_epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        args=args,
        reg_strength=args.reg_strength,
        amp_dtype=amp_dtype,
    )

    loss, acc = evaluate_classifier(model, eval_loader, device, amp_dtype, "SST-2 eval before prune")
    main_print(f"SST-2 before prune: loss={loss:.4f}, acc={100.0 * acc:.2f}%")

    masks, stats = build_hard_masks(model, args.prune_target, args.group_size, args.keep_k)
    enforce_masks(model, masks)
    main_print(f"Applied hard {args.keep_k}:{args.group_size} masks: {stats}")

    loss, acc = evaluate_classifier(model, eval_loader, device, amp_dtype, "SST-2 eval after prune")
    main_print(f"SST-2 after prune: loss={loss:.4f}, acc={100.0 * acc:.2f}%")

    post_lr = args.post_prune_lr if args.post_prune_lr is not None else args.lr
    train_phase(
        model,
        train_loader,
        device,
        phase_name="post-prune SST-2 fine-tune",
        num_epochs=args.post_prune_epochs,
        lr=post_lr,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        args=args,
        reg_strength=0.0,
        amp_dtype=amp_dtype,
        masks=masks,
    )

    loss, acc = evaluate_classifier(model, eval_loader, device, amp_dtype, "Final SST-2 eval")
    main_print(f"Final SST-2: loss={loss:.4f}, acc={100.0 * acc:.2f}%")

    if is_main():
        os.makedirs(args.save_dir, exist_ok=True)
        unwrap_model(model).save_pretrained(args.save_dir)
        tokenizer.save_pretrained(args.save_dir)
        torch.save(
            {
                "masks": masks,
                "mask_stats": stats,
                "args": vars(args),
            },
            os.path.join(args.save_dir, "prune_masks.pt"),
        )
        print(f"Saved model + masks to {args.save_dir}")

    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
