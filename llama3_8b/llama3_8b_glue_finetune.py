"""
Supervised GLUE fine-tuning for Llama checkpoints.

This is intentionally separate from the zero-shot GLUE eval scripts. Use it when
you want a fair supervised comparison such as:
  - dense Llama fine-tuned on GLUE
  - pruned Llama fine-tuned on GLUE

Example:
  CUDA_VISIBLE_DEVICES=0 python llama3_8b_glue_finetune.py \
      --model-name meta-llama/Meta-Llama-3-8B \
      --tasks cola sst2 mrpc \
      --num-epochs 3 \
      --train-scope head-only \
      --output-json output/llama3_8b_glue_finetune_dense.json

  CUDA_VISIBLE_DEVICES=0 python llama3_8b_glue_finetune.py \
      --model-name ./llama3_8b_c4_4of12_cuda \
      --tasks cola sst2 mrpc \
      --num-epochs 3 \
      --train-scope head-only \
      --output-json output/llama3_8b_glue_finetune_pruned.json
"""

import argparse
import gc
import json
import math
import os
import random
import time
import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from datasets import load_dataset
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    mean_absolute_error,
    mean_squared_error,
)
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import (
    Adafactor,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    get_linear_schedule_with_warmup,
)

try:
    import evaluate
except Exception:
    evaluate = None

try:
    from scipy.stats import pearsonr, spearmanr
except Exception:
    pearsonr = None
    spearmanr = None

try:
    import pynvml
except Exception:
    pynvml = None


warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"


MODEL_NAME = "meta-llama/Meta-Llama-3-8B"
DATASET_NAME = "nyu-mll/glue"
SEED = 42
MAX_LENGTH = 256
BATCH_SIZE = 2
EVAL_BATCH_SIZE = 4
GRAD_ACCUM_STEPS = 8
NUM_EPOCHS = 3
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.06
WARMUP_FRAC = 0.20
BENCHMARK_REPEATS = 5
ALLOW_TF32 = True
ENABLE_CUDNN_BENCHMARK = True
MAX_GRAD_NORM = 1.0
STANDARD_TASK_EPOCHS = {
    "cola": 5,
    "mrpc": 5,
    "rte": 5,
    "stsb": 5,
    "wnli": 5,
    "sst2": 3,
    "qnli": 3,
    "qqp": 2,
    "mnli": 2,
}

LEADERBOARD_TASKS = [
    "cola",
    "sst2",
    "mrpc",
    "stsb",
    "qqp",
    "mnli",
    "qnli",
    "rte",
    "wnli",
]
LEADERBOARD_COLUMNS = ["Score", "CoLA", "SST-2", "MRPC", "STS-B", "QQP", "MNLI", "QNLI", "RTE", "WNLI"]

TASK_CONFIGS: Dict[str, Dict[str, Any]] = {
    "cola": {
        "dataset_config": "cola",
        "train_split": "train",
        "eval_splits": {"validation": "validation"},
        "task_type": "classification",
        "sentence_keys": ("sentence",),
        "metric_note": "Matthew's Correlation Coefficient (MCC)",
    },
    "sst2": {
        "dataset_config": "sst2",
        "train_split": "train",
        "eval_splits": {"validation": "validation"},
        "task_type": "classification",
        "sentence_keys": ("sentence",),
        "metric_note": "Accuracy",
    },
    "mrpc": {
        "dataset_config": "mrpc",
        "train_split": "train",
        "eval_splits": {"validation": "validation"},
        "task_type": "classification",
        "sentence_keys": ("sentence1", "sentence2"),
        "metric_note": "F1 / Accuracy",
    },
    "stsb": {
        "dataset_config": "stsb",
        "train_split": "train",
        "eval_splits": {"validation": "validation"},
        "task_type": "regression",
        "sentence_keys": ("sentence1", "sentence2"),
        "metric_note": "Pearson / Spearman",
    },
    "qqp": {
        "dataset_config": "qqp",
        "train_split": "train",
        "eval_splits": {"validation": "validation"},
        "task_type": "classification",
        "sentence_keys": ("question1", "question2"),
        "metric_note": "F1 / Accuracy",
    },
    "mnli": {
        "dataset_config": "mnli",
        "train_split": "train",
        "eval_splits": {
            "validation_matched": "validation_matched",
            "validation_mismatched": "validation_mismatched",
        },
        "task_type": "classification",
        "sentence_keys": ("premise", "hypothesis"),
        "metric_note": "Matched Accuracy / Mismatched Accuracy",
    },
    "qnli": {
        "dataset_config": "qnli",
        "train_split": "train",
        "eval_splits": {"validation": "validation"},
        "task_type": "classification",
        "sentence_keys": ("question", "sentence"),
        "metric_note": "Accuracy",
    },
    "rte": {
        "dataset_config": "rte",
        "train_split": "train",
        "eval_splits": {"validation": "validation"},
        "task_type": "classification",
        "sentence_keys": ("sentence1", "sentence2"),
        "metric_note": "Accuracy",
    },
    "wnli": {
        "dataset_config": "wnli",
        "train_split": "train",
        "eval_splits": {"validation": "validation"},
        "task_type": "classification",
        "sentence_keys": ("sentence1", "sentence2"),
        "metric_note": "Accuracy",
    },
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Supervised GLUE fine-tuning for Llama checkpoints."
    )
    parser.add_argument("--model-name", type=str, default=MODEL_NAME)
    parser.add_argument("--tasks", type=str, nargs="+", default=LEADERBOARD_TASKS)
    parser.add_argument("--max-length", type=int, default=MAX_LENGTH)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--eval-batch-size", type=int, default=EVAL_BATCH_SIZE)
    parser.add_argument("--grad-accum-steps", type=int, default=GRAD_ACCUM_STEPS)
    parser.add_argument("--num-epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE)
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY)
    parser.add_argument("--warmup-ratio", type=float, default=WARMUP_RATIO)
    parser.add_argument("--warmup-frac", type=float, default=WARMUP_FRAC)
    parser.add_argument("--benchmark-repeats", type=int, default=BENCHMARK_REPEATS)
    parser.add_argument("--optimizer", type=str, default="adamw", choices=["adamw", "adafactor"])
    parser.add_argument("--max-grad-norm", type=float, default=MAX_GRAD_NORM)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-eval-samples", type=int, default=None)
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--train-scope", type=str, default="head-only", choices=["head-only", "last-n-layers", "all"])
    parser.add_argument("--num-unfrozen-layers", type=int, default=2)
    parser.add_argument("--task-epochs", type=str, nargs="*", default=None)
    parser.add_argument("--use-standard-task-epochs", action="store_true", default=False)
    parser.set_defaults(preserve_pruning_mask=True)
    parser.add_argument("--preserve-pruning-mask", dest="preserve_pruning_mask", action="store_true")
    parser.add_argument("--no-preserve-pruning-mask", dest="preserve_pruning_mask", action="store_false")
    parser.add_argument("--mask-min-zero-fraction", type=float, default=0.05)
    parser.add_argument("--max-mask-memory-gb", type=float, default=1.5)
    parser.add_argument("--gradient-checkpointing", action="store_true", default=False)
    parser.add_argument(
        "--apply-2to4-runtime",
        action="store_true",
        default=False,
        help=(
            "After fine-tuning, convert masked Llama MLP weights to "
            "SparseSemiStructuredTensor using prune_masks.pt for 2:4 runtime evaluation."
        ),
    )
    parser.add_argument(
        "--prune-masks-path",
        type=str,
        default=None,
        help="Path to prune_masks.pt. Defaults to <local model dir>/prune_masks.pt.",
    )
    parser.add_argument("--save-root", type=str, default=None)
    parser.add_argument("--output-json", type=str, default="output/llama3_8b_glue_finetune.json")
    parser.add_argument(
        "--resume-output-json",
        action="store_true",
        default=False,
        help=(
            "Resume from an existing partial --output-json file. Completed tasks are reused, "
            "remaining tasks are run and merged back into the same JSON."
        ),
    )
    parser.add_argument("--hf-cache-dir", type=str, default="./.hf_cache")
    return parser.parse_args()


def validate_args(args):
    if args.max_length < 8:
        raise ValueError("--max-length must be >= 8.")
    if args.batch_size < 1 or args.eval_batch_size < 1:
        raise ValueError("--batch-size and --eval-batch-size must be >= 1.")
    if args.grad_accum_steps < 1:
        raise ValueError("--grad-accum-steps must be >= 1.")
    if args.num_epochs < 1:
        raise ValueError("--num-epochs must be >= 1.")
    if args.learning_rate <= 0.0:
        raise ValueError("--learning-rate must be > 0.")
    if args.max_grad_norm < 0.0:
        raise ValueError("--max-grad-norm must be >= 0.")
    if not (0.0 <= args.warmup_ratio < 1.0):
        raise ValueError("--warmup-ratio must be in [0, 1).")
    if not (0.0 <= args.warmup_frac < 1.0):
        raise ValueError("--warmup-frac must be in [0, 1).")
    if args.benchmark_repeats < 1:
        raise ValueError("--benchmark-repeats must be >= 1.")
    if args.num_unfrozen_layers < 0:
        raise ValueError("--num-unfrozen-layers must be >= 0.")
    if not (0.0 <= args.mask_min_zero_fraction <= 1.0):
        raise ValueError("--mask-min-zero-fraction must be in [0, 1].")
    if args.max_mask_memory_gb < 0.0:
        raise ValueError("--max-mask-memory-gb must be >= 0.")
    if args.apply_2to4_runtime and args.dtype not in {"float16", "bfloat16"}:
        raise ValueError("--apply-2to4-runtime requires --dtype float16 or bfloat16.")
    if args.apply_2to4_runtime and not args.preserve_pruning_mask:
        raise ValueError("--apply-2to4-runtime requires --preserve-pruning-mask.")

    expanded = []
    for task in args.tasks:
        if task == "all":
            expanded.extend(LEADERBOARD_TASKS)
        else:
            expanded.append(task)
    args.tasks = list(dict.fromkeys(expanded))

    for task in args.tasks:
        if task not in TASK_CONFIGS:
            raise ValueError(f"Unsupported task: {task!r}. Valid tasks: {sorted(TASK_CONFIGS)}")

    args.task_epoch_overrides = parse_task_epoch_overrides(
        args.task_epochs,
        use_standard_schedule=args.use_standard_task_epochs,
    )
    for task_name, epochs in args.task_epoch_overrides.items():
        if task_name not in TASK_CONFIGS:
            raise ValueError(
                f"Unsupported task in --task-epochs: {task_name!r}. Valid tasks: {sorted(TASK_CONFIGS)}"
            )
        if epochs < 1:
            raise ValueError(f"Epoch override for task {task_name!r} must be >= 1, got {epochs}.")


def parse_task_epoch_overrides(items: Optional[List[str]], use_standard_schedule: bool) -> Dict[str, int]:
    overrides: Dict[str, int] = {}
    if use_standard_schedule:
        overrides.update(STANDARD_TASK_EPOCHS)

    if not items:
        return overrides

    for raw_item in items:
        if "=" not in raw_item:
            raise ValueError(
                "--task-epochs entries must look like task=epochs, for example: cola=5 qqp=2"
            )
        task_name, raw_epochs = raw_item.split("=", 1)
        task_name = task_name.strip().lower()
        raw_epochs = raw_epochs.strip()
        if not task_name:
            raise ValueError(f"Invalid --task-epochs entry {raw_item!r}: missing task name.")
        try:
            epochs = int(raw_epochs)
        except ValueError as exc:
            raise ValueError(
                f"Invalid epoch value in --task-epochs entry {raw_item!r}: {raw_epochs!r}"
            ) from exc
        overrides[task_name] = epochs
    return overrides


def get_task_num_epochs(args, task_name: str) -> int:
    return int(args.task_epoch_overrides.get(task_name, args.num_epochs))


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_dtype(name: str) -> torch.dtype:
    return {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[name]


def get_amp_dtype(device: torch.device, dtype: torch.dtype) -> Optional[torch.dtype]:
    if device.type != "cuda":
        return None
    if dtype == torch.float32:
        return None
    return dtype


def configure_torch() -> None:
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = ENABLE_CUDNN_BENCHMARK
        torch.backends.cuda.matmul.allow_tf32 = ALLOW_TF32
        torch.backends.cudnn.allow_tf32 = ALLOW_TF32


def sync_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def cleanup():
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
    for _, param in model.named_parameters():
        if not param.is_floating_point():
            continue
        tensor = param.detach()
        if param.layout != torch.strided:
            continue
        if "SparseSemiStructuredTensor" in type(tensor).__name__:
            continue
        try:
            zero_count = (tensor == 0).sum().item()
        except NotImplementedError:
            continue
        total += tensor.numel()
        zeros += zero_count
    return (zeros / total) if total > 0 else float("nan")


class SparseSemiStructuredLinear(torch.nn.Module):
    """Bias-safe wrapper for SparseSemiStructuredTensor inference."""

    def __init__(self, sparse_weight: torch.Tensor, bias: torch.Tensor):
        super().__init__()
        self.in_features = int(sparse_weight.shape[1])
        self.out_features = int(sparse_weight.shape[0])
        self.weight = torch.nn.Parameter(sparse_weight, requires_grad=False)
        self.register_buffer("bias", bias, persistent=False)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.linear(input_tensor, self.weight, self.bias)


def resolve_prune_masks_path(model_name: str, prune_masks_path: Optional[str]) -> str:
    if prune_masks_path:
        if not os.path.isfile(prune_masks_path):
            raise FileNotFoundError(f"Could not find prune masks file: {prune_masks_path!r}")
        return prune_masks_path

    candidate = os.path.join(model_name, "prune_masks.pt")
    if os.path.isfile(candidate):
        return candidate

    raise FileNotFoundError(
        "Could not resolve prune_masks.pt automatically. "
        "Pass --prune-masks-path explicitly or use a local --model-name directory."
    )


def load_prune_masks(masks_path: str) -> Dict[str, torch.Tensor]:
    mask_bundle = torch.load(masks_path, map_location="cpu")
    masks = mask_bundle.get("masks", {}) if isinstance(mask_bundle, dict) else {}
    if not isinstance(masks, dict) or not masks:
        raise ValueError(f"No masks found in {masks_path!r}.")
    return masks


def replace_submodule(root: torch.nn.Module, module_name: str, new_module: torch.nn.Module) -> None:
    if "." in module_name:
        parent_name, child_name = module_name.rsplit(".", 1)
        parent = root.get_submodule(parent_name)
    else:
        parent = root
        child_name = module_name
    setattr(parent, child_name, new_module)


def apply_exact_2to4_runtime_masks(
    model: torch.nn.Module,
    masks: Dict[str, torch.Tensor],
    dtype: torch.dtype,
) -> Dict[str, Any]:
    try:
        from torch.sparse import to_sparse_semi_structured
    except Exception as exc:
        raise RuntimeError(
            "torch.sparse.to_sparse_semi_structured is not available in this runtime."
        ) from exc

    model = model.to(dtype=dtype)
    converted = 0
    expected = 0
    skipped: List[str] = []

    with torch.no_grad():
        for module_name, mask in masks.items():
            try:
                module = model.get_submodule(module_name)
            except AttributeError:
                skipped.append(f"{module_name}: missing submodule")
                continue
            if not isinstance(module, torch.nn.Linear):
                continue

            expected += 1
            weight = module.weight.data
            if weight.device.type != "cuda":
                skipped.append(f"{module_name}: not on CUDA")
                continue
            if weight.shape != mask.shape:
                skipped.append(
                    f"{module_name}: mask shape {tuple(mask.shape)} != weight shape {tuple(weight.shape)}"
                )
                continue

            masked = (weight * mask.to(device=weight.device, dtype=weight.dtype)).contiguous()
            try:
                sparse_weight = to_sparse_semi_structured(masked)
                if module.bias is None:
                    bias = torch.zeros(
                        sparse_weight.shape[0],
                        device=weight.device,
                        dtype=weight.dtype,
                    )
                else:
                    bias = module.bias.detach().to(device=weight.device, dtype=weight.dtype).contiguous()
                sparse_module = SparseSemiStructuredLinear(sparse_weight, bias)
                replace_submodule(model, module_name, sparse_module)
                converted += 1
            except Exception as exc:
                skipped.append(f"{module_name}: {exc}")

    if converted == 0:
        raise RuntimeError(f"No layers converted from saved 2:4 masks. Skipped: {skipped[:5]}...")
    if converted != expected:
        raise RuntimeError(
            f"Converted {converted}/{expected} masked layers to SparseSemiStructuredTensor. "
            f"First failures: {skipped[:3]}"
        )

    return {
        "converted_count": converted,
        "skipped_count": len(skipped),
    }


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


def format_elapsed_time(seconds: float) -> str:
    seconds = float(seconds)
    if seconds < 60:
        return f"{seconds:.3f} seconds"
    if seconds < 3600:
        return f"{seconds / 60.0:.3f} minutes"
    return f"{seconds / 3600.0:.3f} hours"


def is_finite_number(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def format_leaderboard_value(value: Any) -> str:
    if not is_finite_number(value):
        return "N/A"
    return f"{float(value) * 100.0:.1f}"


def format_pair(first: Any, second: Any) -> str:
    first_s = format_leaderboard_value(first)
    second_s = format_leaderboard_value(second)
    if first_s == "N/A" and second_s == "N/A":
        return "N/A"
    return f"{first_s}/{second_s}"


def get_task_primary_score(result: Dict[str, Any], task_name: str) -> float:
    if task_name == "mnli":
        metrics = result.get("summary", {}).get("Official GLUE Metrics", {})
        return float(metrics.get("mnli_macro_accuracy", float("nan")))
    return float(result.get("summary", {}).get("Primary GLUE Score", float("nan")))


def build_leaderboard_row(results_by_task: Dict[str, Any]) -> Dict[str, str]:
    top_level_scores = [
        get_task_primary_score(results_by_task[task], task)
        for task in LEADERBOARD_TASKS
        if task in results_by_task and is_finite_number(get_task_primary_score(results_by_task[task], task))
    ]
    glue_score = float(np.mean(top_level_scores)) if top_level_scores else float("nan")

    cola = results_by_task.get("cola", {}).get("official_metrics", {})
    sst2 = results_by_task.get("sst2", {}).get("official_metrics", {})
    mrpc = results_by_task.get("mrpc", {}).get("official_metrics", {})
    stsb = results_by_task.get("stsb", {}).get("official_metrics", {})
    qqp = results_by_task.get("qqp", {}).get("official_metrics", {})
    qnli = results_by_task.get("qnli", {}).get("official_metrics", {})
    rte = results_by_task.get("rte", {}).get("official_metrics", {})
    wnli = results_by_task.get("wnli", {}).get("official_metrics", {})
    mnli = results_by_task.get("mnli", {}).get("summary", {}).get("Official GLUE Metrics", {})

    return {
        "Score": format_leaderboard_value(glue_score),
        "CoLA": format_leaderboard_value(cola.get("matthews_correlation")),
        "SST-2": format_leaderboard_value(sst2.get("accuracy")),
        "MRPC": format_pair(mrpc.get("f1"), mrpc.get("accuracy")),
        "STS-B": format_pair(stsb.get("pearson"), stsb.get("spearmanr")),
        "QQP": format_pair(qqp.get("f1"), qqp.get("accuracy")),
        "MNLI": format_pair(mnli.get("matched_accuracy"), mnli.get("mismatched_accuracy")),
        "QNLI": format_leaderboard_value(qnli.get("accuracy")),
        "RTE": format_leaderboard_value(rte.get("accuracy")),
        "WNLI": format_leaderboard_value(wnli.get("accuracy")),
    }


def get_label_names(ds) -> Optional[List[str]]:
    features = ds.features
    if "label" not in features:
        return None
    label_feature = features["label"]
    if hasattr(label_feature, "names") and label_feature.names is not None:
        return [str(x) for x in label_feature.names]
    return None


def build_rankdata(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=np.float64)
    sorted_x = x[order]
    i = 0
    while i < len(sorted_x):
        j = i + 1
        while j < len(sorted_x) and sorted_x[j] == sorted_x[i]:
            j += 1
        rank_value = 0.5 * (i + j - 1) + 1.0
        ranks[order[i:j]] = rank_value
        i = j
    return ranks


def compute_regression_metrics(preds: np.ndarray, refs: np.ndarray) -> Dict[str, float]:
    preds = preds.astype(np.float64)
    refs = refs.astype(np.float64)

    if pearsonr is not None:
        pearson = float(pearsonr(preds, refs)[0])
    else:
        pearson = float(np.corrcoef(preds, refs)[0, 1])

    if spearmanr is not None:
        spearman = float(spearmanr(preds, refs)[0])
    else:
        spearman = float(np.corrcoef(build_rankdata(preds), build_rankdata(refs))[0, 1])

    return {
        "pearson": pearson,
        "spearmanr": spearman,
    }


GLUE_METRIC_CACHE: Dict[str, Any] = {}


def compute_manual_official_metrics(task_name: str, preds: np.ndarray, refs: np.ndarray) -> Dict[str, float]:
    if task_name == "cola":
        return {"matthews_correlation": float(matthews_corrcoef(refs.astype(np.int32), preds.astype(np.int32)))}
    if task_name in {"sst2", "qnli", "rte", "wnli", "mnli"}:
        return {"accuracy": float(accuracy_score(refs.astype(np.int32), preds.astype(np.int32)))}
    if task_name in {"mrpc", "qqp"}:
        refs_i = refs.astype(np.int32)
        preds_i = preds.astype(np.int32)
        return {
            "accuracy": float(accuracy_score(refs_i, preds_i)),
            "f1": float(f1_score(refs_i, preds_i, zero_division=0)),
        }
    if task_name == "stsb":
        return compute_regression_metrics(preds, refs)
    raise ValueError(f"Unsupported task for metrics: {task_name}")


def compute_official_metrics(task_name: str, preds: np.ndarray, refs: np.ndarray) -> Dict[str, float]:
    if evaluate is None:
        return compute_manual_official_metrics(task_name, preds, refs)

    try:
        metric = GLUE_METRIC_CACHE.get(task_name)
        if metric is None:
            metric = evaluate.load("glue", task_name)
            GLUE_METRIC_CACHE[task_name] = metric
        result = metric.compute(predictions=preds, references=refs)
        return {str(key): float(value) for key, value in result.items()}
    except Exception:
        return compute_manual_official_metrics(task_name, preds, refs)


def compute_primary_glue_score(metrics: Dict[str, float]) -> float:
    if not metrics:
        return float("nan")
    values = [float(v) for v in metrics.values() if is_finite_number(v)]
    return float(np.mean(values)) if values else float("nan")


def load_task_dataset(task_name: str, split: str, cache_dir: str):
    cfg = TASK_CONFIGS[task_name]
    return load_dataset(DATASET_NAME, cfg["dataset_config"], split=split, cache_dir=cache_dir)


def tokenize_dataset(ds, tokenizer, task_name: str, max_length: int):
    cfg = TASK_CONFIGS[task_name]
    sentence_keys = cfg["sentence_keys"]

    def preprocess(batch):
        if len(sentence_keys) == 1:
            encoded = tokenizer(
                batch[sentence_keys[0]],
                truncation=True,
                max_length=max_length,
            )
        else:
            encoded = tokenizer(
                batch[sentence_keys[0]],
                batch[sentence_keys[1]],
                truncation=True,
                max_length=max_length,
            )
        encoded["labels"] = batch["label"]
        return encoded

    keep_columns = {"label"}
    remove_columns = [name for name in ds.column_names if name not in keep_columns]
    ds = ds.map(
        preprocess,
        batched=True,
        remove_columns=remove_columns,
        desc=f"Tokenizing {task_name}:{getattr(ds, 'split', 'dataset')}",
    )

    model_columns = ["input_ids", "attention_mask", "labels"]
    if "token_type_ids" in ds.column_names:
        model_columns.append("token_type_ids")
    ds.set_format(type="torch", columns=model_columns)
    return ds


def is_head_parameter(name: str) -> bool:
    return name.startswith("score.") or name.startswith("classifier.") or name.endswith(".score.weight") or name.endswith(".score.bias")


def get_transformer_layers(model) -> List[torch.nn.Module]:
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return list(model.model.layers)
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return list(model.transformer.h)
    return []


def configure_trainable_parameters(model, train_scope: str, num_unfrozen_layers: int) -> Tuple[int, int]:
    for param in model.parameters():
        param.requires_grad_(False)

    if train_scope == "all":
        for param in model.parameters():
            param.requires_grad_(True)
    elif train_scope == "head-only":
        for name, param in model.named_parameters():
            if is_head_parameter(name):
                param.requires_grad_(True)
    elif train_scope == "last-n-layers":
        for name, param in model.named_parameters():
            if is_head_parameter(name):
                param.requires_grad_(True)
        layers = get_transformer_layers(model)
        if num_unfrozen_layers > 0:
            for layer in layers[-num_unfrozen_layers:]:
                for param in layer.parameters():
                    param.requires_grad_(True)
        for name, param in model.named_parameters():
            if name.endswith("norm.weight") or name.endswith("norm.bias"):
                param.requires_grad_(True)
    else:
        raise ValueError(f"Unsupported train scope: {train_scope}")

    total = 0
    trainable = 0
    for param in model.parameters():
        count = param.numel()
        total += count
        if param.requires_grad:
            trainable += count
    if trainable == 0:
        raise RuntimeError("No trainable parameters were selected.")
    return trainable, total


def prepare_model(model_name: str, task_name: str, label_names: Optional[List[str]], dtype: torch.dtype, device: torch.device, gradient_checkpointing: bool, cache_dir: str):
    task_cfg = TASK_CONFIGS[task_name]
    num_labels = 1 if task_cfg["task_type"] == "regression" else len(label_names or [])
    if num_labels < 1:
        raise ValueError(f"Could not determine num_labels for task {task_name}.")

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        torch_dtype=dtype if device.type == "cuda" else torch.float32,
        # In older Transformers/PyTorch stacks, low_cpu_mem_usage can leave a
        # newly initialized classifier head on the meta device for some tasks
        # (for example STS-B with num_labels=1). Use the safer eager load path.
        low_cpu_mem_usage=False,
        ignore_mismatched_sizes=True,
        cache_dir=cache_dir,
    )
    if getattr(model.config, "pad_token_id", None) is None:
        model.config.pad_token_id = model.config.eos_token_id
    if task_cfg["task_type"] == "regression":
        model.config.problem_type = "regression"
    else:
        model.config.problem_type = "single_label_classification"
        if label_names:
            model.config.id2label = {i: name for i, name in enumerate(label_names)}
            model.config.label2id = {name: i for i, name in enumerate(label_names)}
    if gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
    model.to(device)
    return model


def maybe_apply_2to4_runtime(model, model_name: str, dtype: torch.dtype, device: torch.device, args):
    if not args.apply_2to4_runtime:
        return None
    if device.type != "cuda":
        raise RuntimeError("--apply-2to4-runtime requires CUDA.")

    masks_path = resolve_prune_masks_path(model_name, args.prune_masks_path)
    masks = load_prune_masks(masks_path)
    conversion = apply_exact_2to4_runtime_masks(model, masks, dtype)
    model.eval()
    return {
        "masks_path": masks_path,
        "converted_count": int(conversion["converted_count"]),
        "skipped_count": int(conversion["skipped_count"]),
    }


def get_trainable_parameter_list(model) -> List[torch.nn.Parameter]:
    return [param for param in model.parameters() if param.requires_grad]


def build_optimizer(model, optimizer_name: str, learning_rate: float, weight_decay: float):
    decay_params = []
    no_decay_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim == 1 or name.endswith(".bias") or "norm" in name.lower():
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    param_groups = []
    if decay_params:
        param_groups.append({"params": decay_params, "weight_decay": weight_decay})
    if no_decay_params:
        param_groups.append({"params": no_decay_params, "weight_decay": 0.0})

    if optimizer_name == "adamw":
        return torch.optim.AdamW(param_groups, lr=learning_rate)
    if optimizer_name == "adafactor":
        return Adafactor(
            param_groups,
            lr=learning_rate,
            scale_parameter=False,
            relative_step=False,
            warmup_init=False,
        )
    raise ValueError(f"Unsupported optimizer: {optimizer_name}")


def should_preserve_weight_mask(module_name: str, module, min_zero_fraction: float) -> bool:
    if is_head_parameter(module_name):
        return False
    if not isinstance(module, torch.nn.Linear):
        return False
    if not hasattr(module, "weight") or module.weight is None:
        return False
    weight = module.weight
    if not weight.requires_grad or weight.ndim != 2:
        return False
    if not weight.is_floating_point() or weight.layout != torch.strided:
        return False
    zero_count = int((weight.detach() == 0).sum().item())
    if zero_count == 0:
        return False
    zero_fraction = zero_count / max(weight.numel(), 1)
    return zero_fraction >= min_zero_fraction


def prepare_pruning_mask_state(model, preserve_pruning_mask: bool, min_zero_fraction: float, max_mask_memory_gb: float):
    if not preserve_pruning_mask:
        return None

    candidates = []
    total_mask_elems = 0
    for module_name, module in model.named_modules():
        if not should_preserve_weight_mask(module_name, module, min_zero_fraction):
            continue
        total_mask_elems += module.weight.numel()
        candidates.append((module_name, module))

    if not candidates:
        return None

    estimated_mask_gb = total_mask_elems / float(1024 ** 3)
    if max_mask_memory_gb > 0.0 and estimated_mask_gb > max_mask_memory_gb:
        raise RuntimeError(
            "Preserving the pruning mask for the selected trainable layers would require "
            f"about {estimated_mask_gb:.2f} GiB of mask memory, which exceeds "
            f"--max-mask-memory-gb={max_mask_memory_gb:.2f}. "
            "Use head-only or fewer unfrozen layers, or raise the limit explicitly."
        )

    entries = []
    total_zeroed = 0
    for module_name, module in candidates:
        mask = (module.weight.detach() != 0).clone().to(dtype=torch.bool, device=module.weight.device)
        total_zeroed += int((~mask).sum().item())
        entries.append(
            {
                "name": module_name,
                "module": module,
                "mask": mask,
                "numel": int(mask.numel()),
            }
        )

    return {
        "entries": entries,
        "masked_tensor_count": len(entries),
        "masked_weight_count": int(total_mask_elems),
        "zeroed_weight_count": int(total_zeroed),
        "estimated_mask_memory_gb": estimated_mask_gb,
    }


@torch.no_grad()
def enforce_pruning_masks(mask_state) -> None:
    if not mask_state:
        return
    for entry in mask_state["entries"]:
        module = entry["module"]
        module.weight.data.masked_fill_(~entry["mask"], 0)


@torch.no_grad()
def mask_pruned_gradients(mask_state) -> None:
    if not mask_state:
        return
    for entry in mask_state["entries"]:
        grad = entry["module"].weight.grad
        if grad is None:
            continue
        grad.mul_(entry["mask"])


def batch_sequence_lengths(batch: Dict[str, torch.Tensor]) -> torch.Tensor:
    if "attention_mask" in batch:
        return batch["attention_mask"].sum(dim=1).to(dtype=torch.int32)
    batch_size = int(batch["input_ids"].shape[0])
    seq_len = int(batch["input_ids"].shape[1])
    return torch.full((batch_size,), seq_len, dtype=torch.int32)


def prepare_model_inputs(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {
        key: value.to(device, non_blocking=True)
        for key, value in batch.items()
        if key != "labels"
    }


def forward_logits(model, batch: Dict[str, torch.Tensor], device: torch.device, amp_dtype: Optional[torch.dtype]) -> torch.Tensor:
    inputs = prepare_model_inputs(batch, device)
    with torch.autocast("cuda", dtype=amp_dtype, enabled=amp_dtype is not None):
        outputs = model(**inputs)
    return outputs.logits


def compute_probability_margins(probabilities: np.ndarray) -> np.ndarray:
    if probabilities.ndim != 2 or probabilities.shape[1] < 2:
        return np.full((probabilities.shape[0],), np.nan, dtype=np.float64)
    top2 = np.partition(probabilities, kth=probabilities.shape[1] - 2, axis=1)[:, -2:]
    return (top2[:, 1] - top2[:, 0]).astype(np.float64)


def train_model(
    model,
    train_loader,
    device: torch.device,
    amp_dtype: Optional[torch.dtype],
    args,
    num_epochs: int,
    mask_state=None,
) -> List[Dict[str, float]]:
    optimizer = build_optimizer(model, args.optimizer, args.learning_rate, args.weight_decay)
    trainable_params = get_trainable_parameter_list(model)
    optimizer_steps_per_epoch = max(1, math.ceil(len(train_loader) / max(1, args.grad_accum_steps)))
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(args.warmup_ratio * optimizer_steps_per_epoch * num_epochs),
        num_training_steps=optimizer_steps_per_epoch * num_epochs,
    )

    history: List[Dict[str, float]] = []
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        running_loss = 0.0
        batch_count = 0

        progress = tqdm(train_loader, total=len(train_loader), desc=f"Train epoch {epoch + 1}/{num_epochs}")
        for step, batch in enumerate(progress, start=1):
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            with torch.autocast("cuda", dtype=amp_dtype, enabled=amp_dtype is not None):
                outputs = model(**batch)
                loss = outputs.loss

            (loss / max(1, args.grad_accum_steps)).backward()

            if step % max(1, args.grad_accum_steps) == 0 or step == len(train_loader):
                mask_pruned_gradients(mask_state)
                if args.max_grad_norm > 0.0:
                    clip_grad_norm_(trainable_params, args.max_grad_norm)
                optimizer.step()
                enforce_pruning_masks(mask_state)
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            running_loss += float(loss.item())
            batch_count += 1
            progress.set_postfix(loss=f"{running_loss / max(batch_count, 1):.4f}")

        mean_loss = running_loss / max(batch_count, 1)
        history.append({"epoch": float(epoch + 1), "train_loss": float(mean_loss)})
    return history


def evaluate_split(model, loader, device: torch.device, amp_dtype: Optional[torch.dtype], task_name: str, label_names: Optional[List[str]]):
    task_cfg = TASK_CONFIGS[task_name]
    model.eval()

    logits_all = []
    labels_all = []
    sequence_lengths_all = []
    eval_start = time.perf_counter()
    with torch.inference_mode():
        for batch in tqdm(loader, total=len(loader), desc="Eval", leave=False):
            labels = batch["labels"]
            sequence_lengths_all.append(batch_sequence_lengths(batch).cpu())
            outputs = forward_logits(model, batch, device, amp_dtype)
            # NumPy does not support bf16 tensors directly, so normalize eval logits to fp32 on CPU.
            logits_all.append(outputs.detach().to(dtype=torch.float32).cpu())
            labels_all.append(labels.detach().cpu())

    logits = torch.cat(logits_all, dim=0).numpy()
    refs = torch.cat(labels_all, dim=0).numpy()
    sequence_lengths = torch.cat(sequence_lengths_all, dim=0).numpy()
    eval_runtime_s = time.perf_counter() - eval_start

    if task_cfg["task_type"] == "regression":
        preds = logits.reshape(-1).astype(np.float64)
        refs = refs.reshape(-1).astype(np.float64)
        official_metrics = compute_official_metrics(task_name, preds, refs)
        primary = compute_primary_glue_score(official_metrics)
        extra_metrics = {
            "mse": float(mean_squared_error(refs, preds)),
            "rmse": float(math.sqrt(mean_squared_error(refs, preds))),
            "mae": float(mean_absolute_error(refs, preds)),
        }
        return {
            "official_metrics": official_metrics,
            "extra_metrics": extra_metrics,
            "primary_score": primary,
            "predictions": preds,
            "references": refs,
            "sequence_lengths": sequence_lengths,
            "eval_runtime_s": eval_runtime_s,
        }

    preds = np.argmax(logits, axis=1).astype(np.int32)
    refs = refs.astype(np.int32)
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp_logits = np.exp(shifted)
    probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
    margins = compute_probability_margins(probs)
    official_metrics = compute_official_metrics(task_name, preds, refs)
    primary = compute_primary_glue_score(official_metrics)
    extra_metrics = {
        "accuracy": float(accuracy_score(refs, preds)),
        "macro_f1": float(f1_score(refs, preds, average="macro", zero_division=0)),
    }
    if task_name == "cola":
        extra_metrics["matthews_correlation"] = float(matthews_corrcoef(refs, preds))

    labels = list(range(len(label_names or [])))
    report = classification_report(
        refs,
        preds,
        labels=labels,
        target_names=label_names,
        output_dict=True,
        zero_division=0,
    )
    matrix = confusion_matrix(refs, preds, labels=labels).tolist()

    return {
        "official_metrics": official_metrics,
        "extra_metrics": extra_metrics,
        "primary_score": primary,
        "classification_report": report,
        "confusion_matrix": matrix,
        "predictions": preds,
        "references": refs,
        "probabilities": probs,
        "margins": margins,
        "sequence_lengths": sequence_lengths,
        "eval_runtime_s": eval_runtime_s,
    }


def benchmark_eval_loader(
    model,
    loader,
    device: torch.device,
    amp_dtype: Optional[torch.dtype],
    warmup_frac: float,
    repeats: int,
    nvml_handle,
) -> Dict[str, Any]:
    batches = list(loader)
    total_batches = len(batches)

    warmup_batches = 0 if total_batches <= 1 else min(
        total_batches - 1,
        math.ceil(total_batches * warmup_frac),
    )
    measured_batches = batches[warmup_batches:]
    measured_samples_per_pass = sum(int(batch["labels"].shape[0]) for batch in measured_batches)

    if len(measured_batches) == 0 or measured_samples_per_pass == 0:
        return {
            "Model-only Latency (ms/sample, post-warmup)": np.nan,
            "Model-only Time (s, post-warmup)": np.nan,
            "Throughput (samples/s, post-warmup)": np.nan,
            "Measured Energy (J, post-warmup)": np.nan,
            "Energy per Sample (J/sample, post-warmup)": np.nan,
            "Warmup Batches": warmup_batches,
            "Benchmark Repeats": repeats,
            "Measured Samples per Pass": measured_samples_per_pass,
            "P50 Batch Latency ms": np.nan,
            "P95 Batch Latency ms": np.nan,
        }

    model.eval()
    with torch.inference_mode():
        for batch in batches[:warmup_batches]:
            _ = forward_logits(model, batch, device, amp_dtype)
    sync_cuda()

    using_total_energy = get_total_energy_j(nvml_handle) is not None
    energy_start_j = None
    energy_end_j = None
    fallback_energy_j = 0.0
    fallback_energy_samples = 0

    total_model_time_ms = 0.0
    total_measured_samples = measured_samples_per_pass * repeats
    batch_latencies_ms: List[float] = []

    start_event = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
    end_event = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None

    with torch.inference_mode():
        if using_total_energy:
            sync_cuda()
            energy_start_j = get_total_energy_j(nvml_handle)

        for _ in range(repeats):
            for batch in measured_batches:
                if not using_total_energy:
                    sync_cuda()
                    power_start = get_power_watts(nvml_handle)
                    wall_t0 = time.perf_counter()

                if start_event is not None and end_event is not None:
                    start_event.record()
                    _ = forward_logits(model, batch, device, amp_dtype)
                    end_event.record()
                    sync_cuda()
                    elapsed_ms = start_event.elapsed_time(end_event)
                else:
                    wall0 = time.perf_counter()
                    _ = forward_logits(model, batch, device, amp_dtype)
                    wall1 = time.perf_counter()
                    elapsed_ms = (wall1 - wall0) * 1000.0

                total_model_time_ms += elapsed_ms
                batch_latencies_ms.append(elapsed_ms)

                if not using_total_energy:
                    wall_t1 = time.perf_counter()
                    power_end = get_power_watts(nvml_handle)
                    if power_start is not None and power_end is not None:
                        fallback_energy_j += ((power_start + power_end) / 2.0) * (wall_t1 - wall_t0)
                        fallback_energy_samples += 1

        if using_total_energy:
            sync_cuda()
            energy_end_j = get_total_energy_j(nvml_handle)

    measured_model_time_s = total_model_time_ms / 1000.0
    avg_latency_ms = total_model_time_ms / total_measured_samples
    throughput_sps = total_measured_samples / measured_model_time_s

    if using_total_energy and energy_start_j is not None and energy_end_j is not None:
        measured_energy_j = max(0.0, energy_end_j - energy_start_j)
    elif fallback_energy_samples > 0:
        measured_energy_j = fallback_energy_j
    else:
        measured_energy_j = np.nan

    energy_per_sample_j = (
        measured_energy_j / total_measured_samples
        if total_measured_samples > 0 and is_finite_number(measured_energy_j)
        else np.nan
    )

    return {
        "Model-only Latency (ms/sample, post-warmup)": avg_latency_ms,
        "Model-only Time (s, post-warmup)": measured_model_time_s,
        "Throughput (samples/s, post-warmup)": throughput_sps,
        "Measured Energy (J, post-warmup)": measured_energy_j,
        "Energy per Sample (J/sample, post-warmup)": energy_per_sample_j,
        "Warmup Batches": warmup_batches,
        "Benchmark Repeats": repeats,
        "Measured Samples per Pass": measured_samples_per_pass,
        "P50 Batch Latency ms": float(np.percentile(batch_latencies_ms, 50)),
        "P95 Batch Latency ms": float(np.percentile(batch_latencies_ms, 95)),
    }


def save_checkpoint(model, tokenizer, save_dir: str, task_result: Dict[str, Any]):
    os.makedirs(save_dir, exist_ok=True)
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    save_task_result(save_dir, task_result)


def save_task_result(save_dir: str, task_result: Dict[str, Any]):
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, "result.json"), "w", encoding="utf-8") as handle:
        json.dump(task_result, handle, indent=2, allow_nan=True)


def pretty_print_task_result(task_name: str, result: Dict[str, Any]):
    print("\n" + "=" * 68)
    print(f"  TASK: {task_name.upper()}")
    print("=" * 68)
    summary = result["summary"]

    if task_name == "mnli":
        metrics = summary["Official GLUE Metrics"]
        print(f"Matched accuracy     : {metrics['matched_accuracy'] * 100.0:.2f}")
        print(f"Mismatched accuracy  : {metrics['mismatched_accuracy'] * 100.0:.2f}")
        print(f"MNLI macro accuracy  : {metrics['mnli_macro_accuracy'] * 100.0:.2f}")
    else:
        print(f"Primary GLUE Score   : {summary['Primary GLUE Score'] * 100.0:.2f}")
        print(f"Official metrics     : {result['official_metrics']}")

    print(
        f"Trainable parameters : {summary['Trainable Parameters']:,} / "
        f"{summary['Total Parameters']:,} ({summary['Trainable Fraction (%)']:.2f}%)"
    )
    print(f"Train scope          : {summary['Train Scope']}")
    if task_name != "mnli":
        lat = summary.get("Model-only Latency (ms/sample, post-warmup)", float("nan"))
        thpt = summary.get("Throughput (samples/s, post-warmup)", float("nan"))
        if is_finite_number(lat):
            print(f"Model latency        : {float(lat):.3f} ms/sample")
        if is_finite_number(thpt):
            print(f"Throughput           : {float(thpt):.3f} samples/s")
    else:
        matched = result.get("subtasks", {}).get("mnli_matched", {}).get("benchmark_stats", {})
        mismatched = result.get("subtasks", {}).get("mnli_mismatched", {}).get("benchmark_stats", {})
        if is_finite_number(matched.get("Model-only Latency (ms/sample, post-warmup)")):
            print(
                "Matched latency      : "
                f"{float(matched['Model-only Latency (ms/sample, post-warmup)']):.3f} ms/sample"
            )
        if is_finite_number(mismatched.get("Model-only Latency (ms/sample, post-warmup)")):
            print(
                "Mismatched latency   : "
                f"{float(mismatched['Model-only Latency (ms/sample, post-warmup)']):.3f} ms/sample"
            )
    print(f"Task runtime         : {summary['Task Runtime (formatted)']}")


def run_single_task(
    task_name: str,
    tokenizer,
    device: torch.device,
    amp_dtype: Optional[torch.dtype],
    args,
    nvml_handle,
) -> Dict[str, Any]:
    task_cfg = TASK_CONFIGS[task_name]
    task_num_epochs = get_task_num_epochs(args, task_name)
    task_start = time.perf_counter()

    train_ds = load_task_dataset(task_name, task_cfg["train_split"], args.hf_cache_dir)
    if args.max_train_samples is not None:
        train_ds = train_ds.shuffle(seed=args.seed).select(range(min(args.max_train_samples, len(train_ds))))

    eval_sets = {}
    for split_name, split in task_cfg["eval_splits"].items():
        ds = load_task_dataset(task_name, split, args.hf_cache_dir)
        if args.max_eval_samples is not None:
            ds = ds.select(range(min(args.max_eval_samples, len(ds))))
        eval_sets[split_name] = ds

    label_names = None
    for ds in [train_ds] + list(eval_sets.values()):
        label_names = get_label_names(ds)
        if label_names is not None:
            break

    train_ds = tokenize_dataset(train_ds, tokenizer, task_name, args.max_length)
    tokenized_eval_sets = {
        name: tokenize_dataset(ds, tokenizer, task_name, args.max_length)
        for name, ds in eval_sets.items()
    }

    model = prepare_model(
        args.model_name,
        task_name,
        label_names,
        resolve_dtype(args.dtype),
        device,
        args.gradient_checkpointing,
        args.hf_cache_dir,
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    trainable_params, total_params = configure_trainable_parameters(model, args.train_scope, args.num_unfrozen_layers)

    collator = DataCollatorWithPadding(
        tokenizer,
        pad_to_multiple_of=8 if device.type == "cuda" else None,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collator,
        pin_memory=device.type == "cuda",
    )
    eval_loaders = {
        name: DataLoader(
            ds,
            batch_size=args.eval_batch_size,
            shuffle=False,
            collate_fn=collator,
            pin_memory=device.type == "cuda",
        )
        for name, ds in tokenized_eval_sets.items()
    }

    print(f"\nRunning supervised fine-tune for {task_name} ...")
    print(f"Train examples       : {len(train_ds)}")
    print(f"Eval splits          : {list(eval_loaders.keys())}")
    print(f"Epochs               : {task_num_epochs}")
    print(
        f"Trainable parameters : {trainable_params:,} / {total_params:,} "
        f"({100.0 * trainable_params / max(total_params, 1):.2f}%)"
    )

    mask_state = prepare_pruning_mask_state(
        model,
        preserve_pruning_mask=args.preserve_pruning_mask,
        min_zero_fraction=args.mask_min_zero_fraction,
        max_mask_memory_gb=args.max_mask_memory_gb,
    )
    if mask_state:
        print(
            f"Preserved pruning mask: {mask_state['masked_tensor_count']} tensors, "
            f"{mask_state['zeroed_weight_count']:,} zeroed weights "
            f"({mask_state['estimated_mask_memory_gb']:.2f} GiB mask state)"
        )
        enforce_pruning_masks(mask_state)

    initial_zero_fraction = compute_global_zero_fraction(model)
    train_start = time.perf_counter()
    train_history = train_model(
        model,
        train_loader,
        device,
        amp_dtype,
        args,
        num_epochs=task_num_epochs,
        mask_state=mask_state,
    )
    train_runtime_s = time.perf_counter() - train_start
    final_zero_fraction = compute_global_zero_fraction(model)
    zero_fraction_delta = final_zero_fraction - initial_zero_fraction

    if abs(zero_fraction_delta) > 1e-6 and args.train_scope != "head-only":
        print(
            f"  [Sparsity note] dense zero fraction changed from "
            f"{initial_zero_fraction:.4f} to {final_zero_fraction:.4f} during fine-tuning."
        )

    save_dir = os.path.join(args.save_root, task_name) if args.save_root else None
    if save_dir is not None:
        print(f"Saving fine-tuned dense checkpoint to {save_dir}")
        os.makedirs(save_dir, exist_ok=True)
        model.save_pretrained(save_dir)
        tokenizer.save_pretrained(save_dir)

    runtime_sparse_info = maybe_apply_2to4_runtime(
        model,
        args.model_name,
        resolve_dtype(args.dtype),
        device,
        args,
    )
    if runtime_sparse_info:
        print(
            f"Applied exact 2:4 runtime conversion: "
            f"{runtime_sparse_info['converted_count']} layers "
            f"(masks={runtime_sparse_info['masks_path']})"
        )

    if task_name == "mnli":
        matched = evaluate_split(model, eval_loaders["validation_matched"], device, amp_dtype, task_name, label_names)
        mismatched = evaluate_split(model, eval_loaders["validation_mismatched"], device, amp_dtype, task_name, label_names)
        reset_peak_memory()
        matched_bench = benchmark_eval_loader(
            model,
            eval_loaders["validation_matched"],
            device,
            amp_dtype,
            args.warmup_frac,
            args.benchmark_repeats,
            nvml_handle,
        )
        matched_max_mem_gb = get_max_gpu_memory_gb()
        reset_peak_memory()
        mismatched_bench = benchmark_eval_loader(
            model,
            eval_loaders["validation_mismatched"],
            device,
            amp_dtype,
            args.warmup_frac,
            args.benchmark_repeats,
            nvml_handle,
        )
        mismatched_max_mem_gb = get_max_gpu_memory_gb()
        matched_acc = float(matched["official_metrics"]["accuracy"])
        mismatched_acc = float(mismatched["official_metrics"]["accuracy"])
        primary = float((matched_acc + mismatched_acc) / 2.0)

        result = {
            "summary": {
                "Model Variant": args.model_name,
                "Dataset": DATASET_NAME,
                "Task": task_name,
                "Task Type": task_cfg["task_type"],
                "Official GLUE Metric": task_cfg["metric_note"],
                "Official GLUE Metrics": {
                    "matched_accuracy": matched_acc,
                    "mismatched_accuracy": mismatched_acc,
                    "mnli_macro_accuracy": primary,
                },
                "Primary GLUE Score": primary,
                "Train Scope": args.train_scope,
                "Num Unfrozen Layers": int(args.num_unfrozen_layers),
                "Trainable Parameters": int(trainable_params),
                "Total Parameters": int(total_params),
                "Trainable Fraction (%)": 100.0 * trainable_params / max(total_params, 1),
                "Batch Size": int(args.batch_size),
                "Eval Batch Size": int(args.eval_batch_size),
                "Max Length": int(args.max_length),
                "Train Examples": int(len(train_ds)),
                "Epochs": int(task_num_epochs),
                "Learning Rate": float(args.learning_rate),
                "Warmup Fraction": float(args.warmup_frac),
                "Benchmark Repeats": int(args.benchmark_repeats),
                "Optimizer": args.optimizer,
                "Max Grad Norm": float(args.max_grad_norm),
                "Training Runtime (s)": train_runtime_s,
                "Training Runtime (formatted)": format_elapsed_time(train_runtime_s),
                "Initial Dense Zero Fraction": initial_zero_fraction,
                "Final Dense Zero Fraction": final_zero_fraction,
                "Dense Zero Fraction Delta": zero_fraction_delta,
                "Preserved Pruning Mask": bool(mask_state is not None),
                "Task Runtime (s)": 0.0,
                "Task Runtime (formatted)": "",
            },
            "train_history": train_history,
            "subtasks": {
                "mnli_matched": {
                    "split": "validation_matched",
                    "official_metrics": matched["official_metrics"],
                    "extra_metrics": matched["extra_metrics"],
                    "classification_report": matched["classification_report"],
                    "confusion_matrix": matched["confusion_matrix"],
                    "benchmark_stats": matched_bench,
                    "Evaluation Runtime (s)": matched["eval_runtime_s"],
                    "Evaluation Runtime (formatted)": format_elapsed_time(matched["eval_runtime_s"]),
                    "Avg Eval Tokens": float(np.mean(matched["sequence_lengths"])),
                    "P95 Eval Tokens": float(np.percentile(matched["sequence_lengths"], 95)),
                    "Avg Confidence Margin": float(np.mean(matched["margins"])),
                    "Max GPU Memory GB": matched_max_mem_gb,
                },
                "mnli_mismatched": {
                    "split": "validation_mismatched",
                    "official_metrics": mismatched["official_metrics"],
                    "extra_metrics": mismatched["extra_metrics"],
                    "classification_report": mismatched["classification_report"],
                    "confusion_matrix": mismatched["confusion_matrix"],
                    "benchmark_stats": mismatched_bench,
                    "Evaluation Runtime (s)": mismatched["eval_runtime_s"],
                    "Evaluation Runtime (formatted)": format_elapsed_time(mismatched["eval_runtime_s"]),
                    "Avg Eval Tokens": float(np.mean(mismatched["sequence_lengths"])),
                    "P95 Eval Tokens": float(np.percentile(mismatched["sequence_lengths"], 95)),
                    "Avg Confidence Margin": float(np.mean(mismatched["margins"])),
                    "Max GPU Memory GB": mismatched_max_mem_gb,
                },
            },
        }
        if mask_state:
            result["summary"]["Masked Tensor Count"] = int(mask_state["masked_tensor_count"])
            result["summary"]["Masked Weight Count"] = int(mask_state["masked_weight_count"])
            result["summary"]["Zeroed Weight Count"] = int(mask_state["zeroed_weight_count"])
            result["summary"]["Mask Memory (GiB)"] = float(mask_state["estimated_mask_memory_gb"])
    else:
        split_name, loader = next(iter(eval_loaders.items()))
        eval_result = evaluate_split(model, loader, device, amp_dtype, task_name, label_names)
        reset_peak_memory()
        bench = benchmark_eval_loader(
            model,
            loader,
            device,
            amp_dtype,
            args.warmup_frac,
            args.benchmark_repeats,
            nvml_handle,
        )
        max_gpu_memory_gb = get_max_gpu_memory_gb()
        result = {
            "summary": {
                "Model Variant": args.model_name,
                "Dataset": DATASET_NAME,
                "Task": task_name,
                "Split": split_name,
                "Batch Size": int(args.batch_size),
                "Eval Batch Size": int(args.eval_batch_size),
                "Max Length": int(args.max_length),
                "Task Type": task_cfg["task_type"],
                "Official GLUE Metric": task_cfg["metric_note"],
                "Primary GLUE Score": float(eval_result["primary_score"]),
                "Model-only Latency (ms/sample, post-warmup)": bench["Model-only Latency (ms/sample, post-warmup)"],
                "Model-only Time (s, post-warmup)": bench["Model-only Time (s, post-warmup)"],
                "Throughput (samples/s, post-warmup)": bench["Throughput (samples/s, post-warmup)"],
                "Measured Energy (J, post-warmup)": bench["Measured Energy (J, post-warmup)"],
                "Energy per Sample (J/sample, post-warmup)": bench["Energy per Sample (J/sample, post-warmup)"],
                "Warmup Fraction": float(args.warmup_frac),
                "Warmup Batches": bench["Warmup Batches"],
                "Benchmark Repeats": bench["Benchmark Repeats"],
                "Measured Samples per Pass": bench["Measured Samples per Pass"],
                "P50 Batch Latency ms": bench["P50 Batch Latency ms"],
                "P95 Batch Latency ms": bench["P95 Batch Latency ms"],
                "Max GPU Memory GB": max_gpu_memory_gb,
                "Train Scope": args.train_scope,
                "Num Unfrozen Layers": int(args.num_unfrozen_layers),
                "Trainable Parameters": int(trainable_params),
                "Total Parameters": int(total_params),
                "Trainable Fraction (%)": 100.0 * trainable_params / max(total_params, 1),
                "Train Examples": int(len(train_ds)),
                "Epochs": int(task_num_epochs),
                "Learning Rate": float(args.learning_rate),
                "Optimizer": args.optimizer,
                "Max Grad Norm": float(args.max_grad_norm),
                "Training Runtime (s)": train_runtime_s,
                "Training Runtime (formatted)": format_elapsed_time(train_runtime_s),
                "Evaluation Runtime (s)": eval_result["eval_runtime_s"],
                "Evaluation Runtime (formatted)": format_elapsed_time(eval_result["eval_runtime_s"]),
                "Initial Dense Zero Fraction": initial_zero_fraction,
                "Final Dense Zero Fraction": final_zero_fraction,
                "Dense Zero Fraction Delta": zero_fraction_delta,
                "Preserved Pruning Mask": bool(mask_state is not None),
                "Avg Eval Tokens": float(np.mean(eval_result["sequence_lengths"])),
                "P95 Eval Tokens": float(np.percentile(eval_result["sequence_lengths"], 95)),
                "Task Runtime (s)": 0.0,
                "Task Runtime (formatted)": "",
            },
            "official_metrics": eval_result["official_metrics"],
            "extra_metrics": eval_result["extra_metrics"],
            "train_history": train_history,
            "benchmark_stats": bench,
        }
        if task_cfg["task_type"] == "classification":
            result["label_names"] = label_names
            result["classification_report"] = eval_result["classification_report"]
            result["confusion_matrix"] = eval_result["confusion_matrix"]
            result["summary"]["Avg Confidence Margin"] = float(np.mean(eval_result["margins"]))
        if mask_state:
            result["summary"]["Masked Tensor Count"] = int(mask_state["masked_tensor_count"])
            result["summary"]["Masked Weight Count"] = int(mask_state["masked_weight_count"])
            result["summary"]["Zeroed Weight Count"] = int(mask_state["zeroed_weight_count"])
            result["summary"]["Mask Memory (GiB)"] = float(mask_state["estimated_mask_memory_gb"])

    if runtime_sparse_info:
        result["summary"]["Runtime 2:4 Sparse"] = True
        result["summary"]["Runtime 2:4 Converted Layers"] = int(runtime_sparse_info["converted_count"])
        result["summary"]["Runtime 2:4 Skipped Layers"] = int(runtime_sparse_info["skipped_count"])
        result["summary"]["Runtime 2:4 Masks Path"] = runtime_sparse_info["masks_path"]
    else:
        result["summary"]["Runtime 2:4 Sparse"] = False

    task_runtime_s = time.perf_counter() - task_start
    result["summary"]["Task Runtime (s)"] = task_runtime_s
    result["summary"]["Task Runtime (formatted)"] = format_elapsed_time(task_runtime_s)

    if save_dir is not None:
        save_task_result(save_dir, result)

    del model
    cleanup()
    return result


def write_json(path: str, payload: Dict[str, Any]):
    output_dir = os.path.dirname(path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, allow_nan=True)


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def extract_saved_runtime_s(payload: Dict[str, Any]) -> float:
    summary = payload.get("summary", {}) or {}
    for key in ("Total Runtime (s)", "Runtime So Far (s)"):
        value = summary.get(key)
        if value is None:
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return 0.0


def load_resume_state(output_json_path: str, args) -> Tuple[Dict[str, Any], float]:
    if not os.path.exists(output_json_path):
        raise FileNotFoundError(
            f"--resume-output-json was set, but {output_json_path!r} does not exist."
        )

    payload = load_json(output_json_path)
    summary = payload.get("summary", {}) or {}
    saved_model = summary.get("Model Variant")
    if saved_model not in (None, args.model_name):
        raise ValueError(
            f"Resume JSON model mismatch: file has {saved_model!r}, but current --model-name is {args.model_name!r}."
        )

    compatibility_checks = [
        ("Train Scope", args.train_scope),
        ("Num Unfrozen Layers", args.num_unfrozen_layers),
        ("Global Epochs", args.num_epochs),
        ("Task Epoch Overrides", args.task_epoch_overrides),
        ("Use Standard Task Epochs", args.use_standard_task_epochs),
        ("Learning Rate", args.learning_rate),
        ("Optimizer", args.optimizer),
        ("Max Grad Norm", args.max_grad_norm),
        ("Preserve Pruning Mask", args.preserve_pruning_mask),
        ("Apply 2:4 Runtime", args.apply_2to4_runtime),
        ("Prune Masks Path", args.prune_masks_path),
        ("Batch Size", args.batch_size),
        ("Eval Batch Size", args.eval_batch_size),
        ("Gradient Accumulation Steps", args.grad_accum_steps),
        ("Max Length", args.max_length),
        ("Warmup Fraction", args.warmup_frac),
        ("Benchmark Repeats", args.benchmark_repeats),
    ]
    for key, current_value in compatibility_checks:
        saved_value = summary.get(key)
        if saved_value is None:
            continue
        if saved_value != current_value:
            raise ValueError(
                f"Resume JSON setting mismatch for {key!r}: file has {saved_value!r}, current run has {current_value!r}."
            )

    results_by_task = payload.get("results_by_task", {}) or {}
    return results_by_task, extract_saved_runtime_s(payload)


def main():
    args = parse_args()
    validate_args(args)
    set_seed(args.seed)
    configure_torch()

    requested_tasks = list(args.tasks)
    results_by_task: Dict[str, Any] = {}
    previous_runtime_s = 0.0
    completed_tasks = set()

    if args.resume_output_json:
        results_by_task, previous_runtime_s = load_resume_state(args.output_json, args)
        completed_tasks = set(results_by_task.keys())
        pending_tasks = [task for task in requested_tasks if task not in completed_tasks]
        print("\n=== Resume state ===")
        print(f"Resume file          : {args.output_json}")
        print(f"Completed tasks      : {sorted(completed_tasks)}")
        print(f"Pending tasks        : {pending_tasks}")
        print(f"Saved runtime        : {format_elapsed_time(previous_runtime_s)}")
        args.tasks = pending_tasks

        if not args.tasks:
            final_payload = {
                "leaderboard": build_leaderboard_row(results_by_task),
                "summary": {
                    "Model Variant": args.model_name,
                    "Tasks Run": list(results_by_task.keys()),
                    "Task Count": len(results_by_task),
                    "Train Scope": args.train_scope,
                    "Num Unfrozen Layers": args.num_unfrozen_layers,
                    "Global Epochs": args.num_epochs,
                    "Task Epoch Overrides": args.task_epoch_overrides,
                    "Use Standard Task Epochs": args.use_standard_task_epochs,
                    "Learning Rate": args.learning_rate,
                    "Optimizer": args.optimizer,
                    "Max Grad Norm": args.max_grad_norm,
                    "Preserve Pruning Mask": args.preserve_pruning_mask,
                    "Apply 2:4 Runtime": args.apply_2to4_runtime,
                    "Prune Masks Path": args.prune_masks_path,
                    "Batch Size": args.batch_size,
                    "Eval Batch Size": args.eval_batch_size,
                    "Gradient Accumulation Steps": args.grad_accum_steps,
                    "Max Length": args.max_length,
                    "Warmup Fraction": args.warmup_frac,
                    "Benchmark Repeats": args.benchmark_repeats,
                    "Total Runtime (s)": previous_runtime_s,
                    "Total Runtime (formatted)": format_elapsed_time(previous_runtime_s),
                    "Status": "complete",
                    "Resumed From Existing JSON": True,
                },
                "results_by_task": results_by_task,
            }
            write_json(args.output_json, final_payload)
            print("All requested tasks are already present in the resume JSON.")
            print(f"Saved final JSON to {args.output_json}")
            return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_dtype = resolve_dtype(args.dtype)
    amp_dtype = get_amp_dtype(device, model_dtype)

    print(f"Device               : {device}")
    print(f"Model dtype          : {model_dtype}")
    print(f"AMP dtype            : {amp_dtype}")
    print(f"Model                : {args.model_name}")
    print(f"Tasks                : {args.tasks}")
    print(f"Train scope          : {args.train_scope}")
    print(f"Optimizer            : {args.optimizer}")
    print(f"Max grad norm        : {args.max_grad_norm}")
    print(f"Preserve prune mask  : {args.preserve_pruning_mask}")
    print(f"Apply 2:4 runtime    : {args.apply_2to4_runtime}")
    if args.apply_2to4_runtime and args.prune_masks_path:
        print(f"Prune masks path     : {args.prune_masks_path}")
    if args.use_standard_task_epochs:
        print("Task epoch schedule  : standard")
    if args.task_epoch_overrides:
        print(f"Task epoch overrides : {args.task_epoch_overrides}")
    else:
        print(f"Global epochs        : {args.num_epochs}")
    print(f"Benchmark repeats    : {args.benchmark_repeats}")
    print(f"Warmup fraction      : {args.warmup_frac}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True, cache_dir=args.hf_cache_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    script_start = time.perf_counter()
    nvml_handle = init_nvml()

    try:
        for task_name in args.tasks:
            task_result = run_single_task(task_name, tokenizer, device, amp_dtype, args, nvml_handle)
            results_by_task[task_name] = task_result
            pretty_print_task_result(task_name, task_result)

            interim_payload = {
                "leaderboard": build_leaderboard_row(results_by_task),
                "summary": {
                    "Model Variant": args.model_name,
                    "Tasks Run So Far": list(results_by_task.keys()),
                    "Tasks Planned": requested_tasks,
                    "Train Scope": args.train_scope,
                    "Num Unfrozen Layers": args.num_unfrozen_layers,
                    "Global Epochs": args.num_epochs,
                    "Task Epoch Overrides": args.task_epoch_overrides,
                    "Use Standard Task Epochs": args.use_standard_task_epochs,
                    "Learning Rate": args.learning_rate,
                    "Optimizer": args.optimizer,
                    "Max Grad Norm": args.max_grad_norm,
                    "Preserve Pruning Mask": args.preserve_pruning_mask,
                    "Apply 2:4 Runtime": args.apply_2to4_runtime,
                    "Prune Masks Path": args.prune_masks_path,
                    "Batch Size": args.batch_size,
                    "Eval Batch Size": args.eval_batch_size,
                    "Gradient Accumulation Steps": args.grad_accum_steps,
                    "Max Length": args.max_length,
                    "Warmup Fraction": args.warmup_frac,
                    "Benchmark Repeats": args.benchmark_repeats,
                    "Status": "in_progress",
                    "Runtime So Far (s)": previous_runtime_s + (time.perf_counter() - script_start),
                    "Runtime So Far (formatted)": format_elapsed_time(previous_runtime_s + (time.perf_counter() - script_start)),
                    "Resumed From Existing JSON": args.resume_output_json,
                },
                "results_by_task": results_by_task,
            }
            write_json(args.output_json, interim_payload)

        total_runtime_s = previous_runtime_s + (time.perf_counter() - script_start)
        final_payload = {
            "leaderboard": build_leaderboard_row(results_by_task),
            "summary": {
                "Model Variant": args.model_name,
                "Tasks Run": list(results_by_task.keys()),
                "Task Count": len(results_by_task),
                "Train Scope": args.train_scope,
                "Num Unfrozen Layers": args.num_unfrozen_layers,
                "Global Epochs": args.num_epochs,
                "Task Epoch Overrides": args.task_epoch_overrides,
                "Use Standard Task Epochs": args.use_standard_task_epochs,
                "Learning Rate": args.learning_rate,
            "Optimizer": args.optimizer,
            "Max Grad Norm": args.max_grad_norm,
            "Preserve Pruning Mask": args.preserve_pruning_mask,
            "Apply 2:4 Runtime": args.apply_2to4_runtime,
            "Prune Masks Path": args.prune_masks_path,
            "Batch Size": args.batch_size,
            "Eval Batch Size": args.eval_batch_size,
            "Gradient Accumulation Steps": args.grad_accum_steps,
                "Max Length": args.max_length,
                "Warmup Fraction": args.warmup_frac,
                "Benchmark Repeats": args.benchmark_repeats,
                "Total Runtime (s)": total_runtime_s,
                "Total Runtime (formatted)": format_elapsed_time(total_runtime_s),
                "Status": "complete",
                "Resumed From Existing JSON": args.resume_output_json,
            },
            "results_by_task": results_by_task,
        }
        write_json(args.output_json, final_payload)

        print("\n" + "=" * 68)
        print("  FINAL SUPERVISED GLUE LEADERBOARD ROW")
        print("=" * 68)
        for column in LEADERBOARD_COLUMNS:
            print(f"{column:>8}: {final_payload['leaderboard'].get(column, 'N/A')}")
        print("=" * 68)
        print(f"\nSaved JSON to {args.output_json}")
        print(f"Total runtime: {format_elapsed_time(total_runtime_s)}")
    finally:
        shutdown_nvml()
        cleanup()


if __name__ == "__main__":
    main()
