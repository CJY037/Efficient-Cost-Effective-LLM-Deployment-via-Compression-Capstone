import os
import gc
import json
import math
import time
import random
import warnings
import argparse
import importlib.metadata
from typing import Dict, Any, List, Optional

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

try:
    import pynvml
except Exception:
    pynvml = None

warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# CUDA_VISIBLE_DEVICES=1 python llama3_8b_wikitext2.py
# CUDA_VISIBLE_DEVICES=1 python llama3_8b_wikitext2.py --model-dtype float16
# CUDA_VISIBLE_DEVICES=1 python llama3_8b_wikitext2.py --quantization int8
# CUDA_VISIBLE_DEVICES=0 python llama3_8b_wikitext2.py --model-name ./llama3_8b_c4_4of12_cuda --quantization original --output-json output/llama3_8b_wikitext2_validation_c4_4of12_cuda.json
# CUDA_VISIBLE_DEVICES=0 python llama3_8b_wikitext2.py --model-name ./llama3_8b_c4_4of12_cuda --quantization int8 --output-json output/llama3_8b_wikitext2_validation_c4_4of12_cuda_int8.json
# CUDA_VISIBLE_DEVICES=0 python llama3_8b_wikitext2.py --model-name ./llama3_8b_c4_2of4_cuda --quantization original --model-dtype float16 --apply-2to4-runtime --runtime-2to4-dtype float16 --output-json output/llama3_8b_wikitext2_validation_c4_2of4_runtime_fp16.json
# CUDA_VISIBLE_DEVICES=0 python llama3_8b_wikitext2.py --model-name meta-llama/Meta-Llama-3-8B --quantization original --model-dtype float16 --output-json output/llama3_8b_wikitext2_validation_original_fp16.json

# =========================================================
# Config
# =========================================================
SEED = 42
MODEL_NAME = "meta-llama/Meta-Llama-3-8B"
DATASET_NAME = "wikitext"
DATASET_CONFIG = "wikitext-2-raw-v1"
EVAL_SPLIT = "validation"

BATCH_SIZE = 1
MAX_LENGTH = 2048
STRIDE = 512
WARMUP_FRAC = 0.20
BENCHMARK_REPEATS = 5
ALLOW_TF32 = True
ENABLE_CUDNN_BENCHMARK = True
DEFAULT_QUANTIZATION = "original"
DEFAULT_MODEL_DTYPE = "auto"
QUANTIZATION_ALIASES = {
    "original": "original",
    "dense": "original",
    "int8": "int8",
}


# =========================================================
# Reproducibility + device
# =========================================================
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def configure_torch() -> None:
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = ENABLE_CUDNN_BENCHMARK
        torch.backends.cuda.matmul.allow_tf32 = ALLOW_TF32
        torch.backends.cudnn.allow_tf32 = ALLOW_TF32


def get_torch_dtype():
    if torch.cuda.is_available():
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    return torch.float32


def resolve_model_dtype(name: str):
    if name == "auto":
        return get_torch_dtype()
    return {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[name]


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
    """Ensure checkpoints with tied embeddings use the loaded input embedding."""
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


def get_model_input_device(model: torch.nn.Module) -> torch.device:
    hf_device_map = getattr(model, "hf_device_map", None)
    if isinstance(hf_device_map, dict):
        for device in hf_device_map.values():
            device_str = str(device)
            if device_str not in {"cpu", "disk", "meta"}:
                if device_str.isdigit():
                    return torch.device(f"cuda:{device_str}")
                return torch.device(device_str)
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
        x = p.detach()
        if p.layout != torch.strided:
            continue
        if "SparseSemiStructuredTensor" in type(x).__name__:
            continue
        try:
            zero_count = (x == 0).sum().item()
        except NotImplementedError:
            continue
        total += x.numel()
        zeros += zero_count
    return (zeros / total) if total > 0 else float("nan")


class SparseSemiStructuredLinear(torch.nn.Module):
    """Bias-safe wrapper for SparseSemiStructuredTensor inference."""

    def __init__(self, sparse_weight: torch.Tensor, bias: torch.Tensor):
        super().__init__()
        self.in_features = int(sparse_weight.shape[1])
        self.out_features = int(sparse_weight.shape[0])
        self.compute_dtype = sparse_weight.dtype
        self.weight = torch.nn.Parameter(sparse_weight, requires_grad=False)
        self.register_buffer("bias", bias, persistent=False)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        if input_tensor.device != self.weight.device or input_tensor.dtype != self.compute_dtype:
            input_tensor = input_tensor.to(device=self.weight.device, dtype=self.compute_dtype)
        bias = self.bias
        if bias is not None and (bias.device != self.weight.device or bias.dtype != self.compute_dtype):
            bias = bias.to(device=self.weight.device, dtype=self.compute_dtype)
        return torch.nn.functional.linear(input_tensor, self.weight, bias)


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

            target_weight = weight if weight.dtype == dtype else weight.to(dtype=dtype)
            masked = (target_weight * mask.to(device=weight.device, dtype=target_weight.dtype)).contiguous()
            try:
                sparse_weight = to_sparse_semi_structured(masked)
                if module.bias is None:
                    bias = torch.zeros(
                        sparse_weight.shape[0],
                        device=weight.device,
                        dtype=target_weight.dtype,
                    )
                else:
                    bias = module.bias.detach().to(device=weight.device, dtype=target_weight.dtype).contiguous()
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


def format_elapsed_time(seconds: float) -> str:
    seconds = float(seconds)
    if seconds < 60:
        return f"{seconds:.3f} seconds"
    if seconds < 3600:
        return f"{seconds / 60.0:.3f} minutes"
    return f"{seconds / 3600.0:.3f} hours"


def get_package_version(package_name: str) -> str:
    try:
        return importlib.metadata.version(package_name)
    except importlib.metadata.PackageNotFoundError:
        return "not_installed"


def normalize_quantization_mode(quantization: str) -> str:
    try:
        return QUANTIZATION_ALIASES[quantization]
    except KeyError as exc:
        valid = ", ".join(sorted(QUANTIZATION_ALIASES))
        raise ValueError(f"Unsupported quantization mode: {quantization!r}. Valid modes: {valid}") from exc


def normalize_device_map(device_map):
    if device_map is None:
        return None
    if not isinstance(device_map, dict):
        return str(device_map)
    return {str(k): str(v) for k, v in device_map.items()}


def get_model_memory_footprint_bytes(model: torch.nn.Module) -> int:
    if hasattr(model, "get_memory_footprint"):
        try:
            return int(model.get_memory_footprint())
        except Exception:
            pass

    total = 0
    for p in model.parameters():
        total += p.numel() * p.element_size()
    for b in model.buffers():
        total += b.numel() * b.element_size()
    return int(total)


def validate_int8_runtime() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("bitsandbytes INT8 inference requires a CUDA-capable GPU.")

    try:
        import bitsandbytes  # noqa: F401
    except Exception as exc:
        raise RuntimeError(
            "bitsandbytes is required for --quantization int8. Install it with "
            "`pip install bitsandbytes` in the active environment."
        ) from exc

    major, minor = torch.cuda.get_device_capability(0)
    capability = major + (minor / 10.0)
    if capability < 7.5:
        raise RuntimeError(
            "bitsandbytes LLM.int8() is intended for NVIDIA GPUs with compute "
            f"capability >= 7.5; detected {major}.{minor}."
        )


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
def load_model_and_tokenizer(model_name: str, quantization: str, model_dtype_name: str):
    quantization = normalize_quantization_mode(quantization)
    model_dtype = resolve_model_dtype(model_dtype_name)
    print(f"Loading model: {model_name}")
    print(f"Quantization: {quantization}")
    print(f"Model dtype: {model_dtype}")
    load_start = time.perf_counter()

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    kwargs: Dict[str, Any] = {
        "low_cpu_mem_usage": True,
    }

    if quantization == "int8":
        validate_int8_runtime()
        kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        kwargs["device_map"] = "auto"
        kwargs["torch_dtype"] = "auto"
    elif quantization == "original":
        kwargs["torch_dtype"] = model_dtype
        if torch.cuda.is_available():
            kwargs["device_map"] = "auto"
    else:
        raise ValueError(f"Unsupported quantization mode: {quantization!r}")

    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    ensure_tied_lm_head(model)
    if getattr(model.config, "pad_token_id", None) is None:
        model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = False
    model.eval()

    load_time_s = time.perf_counter() - load_start
    memory_footprint_bytes = get_model_memory_footprint_bytes(model)
    load_info = {
        "quantization": quantization,
        "model_dtype": str(model_dtype),
        "model_dtype_arg": model_dtype_name,
        "quantization_backend": "bitsandbytes" if quantization == "int8" else "none",
        "bitsandbytes_version": get_package_version("bitsandbytes"),
        "transformers_version": get_package_version("transformers"),
        "torch_version": torch.__version__,
        "model_memory_footprint_bytes": memory_footprint_bytes,
        "model_memory_footprint_gb": memory_footprint_bytes / (1024 ** 3),
        "model_load_time_s": load_time_s,
        "hf_device_map": normalize_device_map(getattr(model, "hf_device_map", None)),
        "input_device": str(get_model_input_device(model)),
    }

    print(f"Loaded {quantization} model in {format_elapsed_time(load_time_s)}.")
    print(f"Memory footprint: {load_info['model_memory_footprint_gb']:.3f} GB")
    print(f"Input device: {load_info['input_device']}")
    return model, tokenizer, load_info


# =========================================================
# Dataset / corpus helpers
# =========================================================
def get_text_field(ds) -> str:
    for key in ["text", "sentence", "content"]:
        if key in ds.column_names:
            return key
    raise KeyError(f"Could not find a text field in columns: {ds.column_names}")


def build_corpus_text(ds, text_field: str) -> str:
    texts = ["" if x is None else str(x) for x in ds[text_field]]
    if len(texts) == 0:
        raise ValueError("Validation corpus is empty.")
    return "\n\n".join(texts)


def make_windows(input_ids: torch.Tensor, max_length: int, stride: int) -> List[Dict[str, Any]]:
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


def build_target_ids(input_ids: torch.Tensor, trg_len: int) -> torch.Tensor:
    target_ids = input_ids.clone()
    target_ids[:, :-trg_len] = -100
    return target_ids


def count_loss_tokens(target_ids: torch.Tensor) -> int:
    valid_labels = int((target_ids != -100).sum().item())
    batch_size = int(target_ids.size(0))
    return max(valid_labels - batch_size, 0)


# =========================================================
# Evaluation
# =========================================================
def evaluate_perplexity(model, windows: List[Dict[str, Any]], total_bytes: int):
    device = get_model_input_device(model)
    total_nll_nats = 0.0
    eval_token_count = 0
    window_token_counts = []

    with torch.inference_mode():
        for idx, w in enumerate(windows, start=1):
            input_ids = w["input_ids"].to(device)
            target_ids = build_target_ids(input_ids, w["trg_len"])
            loss_tokens = count_loss_tokens(target_ids)

            if loss_tokens == 0:
                window_token_counts.append(0)
                continue

            outputs = model(input_ids=input_ids, labels=target_ids)
            neg_log_likelihood = float(outputs.loss.item()) * loss_tokens

            total_nll_nats += neg_log_likelihood
            eval_token_count += loss_tokens
            window_token_counts.append(loss_tokens)

            if idx == 1 or idx == len(windows) or idx % 25 == 0:
                print(f"  [Perplexity] window {idx}/{len(windows)}")

    if eval_token_count == 0:
        raise ValueError("No loss tokens were evaluated; check max_length/stride/input preparation.")

    avg_nll = total_nll_nats / eval_token_count
    token_perplexity = math.exp(avg_nll)
    bits_per_byte = (total_nll_nats / math.log(2.0)) / max(total_bytes, 1)
    byte_perplexity = 2.0 ** bits_per_byte

    return {
        "total_nll_nats": float(total_nll_nats),
        "avg_nll": float(avg_nll),
        "token_perplexity": float(token_perplexity),
        "bits_per_byte": float(bits_per_byte),
        "byte_perplexity": float(byte_perplexity),
        "evaluated_tokens": int(eval_token_count),
        "evaluated_bytes": int(total_bytes),
        "window_token_counts": window_token_counts,
    }


# =========================================================
# Benchmarking
# =========================================================
def benchmark_windows(
    model,
    windows: List[Dict[str, Any]],
    warmup_frac: float,
    repeats: int,
    nvml_handle,
):
    total_windows = len(windows)
    warmup_windows = 0 if total_windows <= 1 else min(
        total_windows - 1,
        math.ceil(total_windows * warmup_frac),
    )

    device = get_model_input_device(model)

    prepared_windows = []
    for w in windows:
        input_ids = w["input_ids"].to(device)
        target_ids = build_target_ids(input_ids, w["trg_len"])
        loss_tokens = count_loss_tokens(target_ids)
        prepared_windows.append(
            {
                "input_ids": input_ids,
                "target_ids": target_ids,
                "loss_tokens": loss_tokens,
            }
        )

    measured_windows = prepared_windows[warmup_windows:]
    measured_tokens_per_pass = sum(int(w["loss_tokens"]) for w in measured_windows)

    if len(measured_windows) == 0 or measured_tokens_per_pass == 0:
        return {
            "Model-only Latency (ms/token, post-warmup)": np.nan,
            "Model-only Time (s, post-warmup)": np.nan,
            "Throughput (tokens/s, post-warmup)": np.nan,
            "Measured Energy (J, post-warmup)": np.nan,
            "Energy per Token (J/token, post-warmup)": np.nan,
            "Warmup Windows": warmup_windows,
            "Benchmark Repeats": repeats,
            "P50 Window Latency ms": np.nan,
            "P95 Window Latency ms": np.nan,
        }

    with torch.inference_mode():
        for w in prepared_windows[:warmup_windows]:
            _ = model(input_ids=w["input_ids"], labels=w["target_ids"])

    sync_cuda()

    using_total_energy = get_total_energy_j(nvml_handle) is not None
    energy_start_j = None
    energy_end_j = None
    fallback_energy_j = 0.0

    total_model_time_ms = 0.0
    total_measured_tokens = measured_tokens_per_pass * repeats
    window_latencies_ms = []

    start_event = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
    end_event = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None

    with torch.inference_mode():
        if using_total_energy:
            sync_cuda()
            energy_start_j = get_total_energy_j(nvml_handle)

        for _ in range(repeats):
            for w in measured_windows:
                if not using_total_energy:
                    sync_cuda()
                    p0 = get_power_watts(nvml_handle)
                    wall_t0 = time.perf_counter()

                if start_event is not None:
                    start_event.record()
                    _ = model(input_ids=w["input_ids"], labels=w["target_ids"])
                    end_event.record()
                    sync_cuda()
                    elapsed_ms = start_event.elapsed_time(end_event)
                else:
                    wall0 = time.perf_counter()
                    _ = model(input_ids=w["input_ids"], labels=w["target_ids"])
                    wall1 = time.perf_counter()
                    elapsed_ms = (wall1 - wall0) * 1000.0

                total_model_time_ms += elapsed_ms
                window_latencies_ms.append(elapsed_ms)

                if not using_total_energy:
                    wall_t1 = time.perf_counter()
                    p1 = get_power_watts(nvml_handle)
                    if p0 is not None and p1 is not None:
                        fallback_energy_j += ((p0 + p1) / 2.0) * (wall_t1 - wall_t0)

        if using_total_energy:
            sync_cuda()
            energy_end_j = get_total_energy_j(nvml_handle)

    measured_model_time_s = total_model_time_ms / 1000.0
    throughput_tps = total_measured_tokens / measured_model_time_s
    latency_ms_per_token = total_model_time_ms / total_measured_tokens

    if using_total_energy and energy_start_j is not None and energy_end_j is not None:
        measured_energy_j = max(0.0, energy_end_j - energy_start_j)
    else:
        measured_energy_j = fallback_energy_j

    energy_per_token_j = measured_energy_j / total_measured_tokens if total_measured_tokens > 0 else np.nan

    return {
        "Model-only Latency (ms/token, post-warmup)": float(latency_ms_per_token),
        "Model-only Time (s, post-warmup)": float(measured_model_time_s),
        "Throughput (tokens/s, post-warmup)": float(throughput_tps),
        "Measured Energy (J, post-warmup)": float(measured_energy_j),
        "Energy per Token (J/token, post-warmup)": float(energy_per_token_j),
        "Warmup Windows": int(warmup_windows),
        "Benchmark Repeats": int(repeats),
        "P50 Window Latency ms": float(np.percentile(window_latencies_ms, 50)),
        "P95 Window Latency ms": float(np.percentile(window_latencies_ms, 95)),
    }


# =========================================================
# Result builder
# =========================================================
def build_result(
    model_name: str,
    dataset_name: str,
    dataset_config: str,
    split: str,
    text_field: str,
    corpus_documents: int,
    corpus_characters: int,
    max_length: int,
    stride: int,
    window_count: int,
    eval_stats: Dict[str, Any],
    benchmark_stats: Dict[str, Any],
    dense_zero_fraction: float,
    total_runtime_s: float,
    load_info: Dict[str, Any],
    runtime_sparse_info: Optional[Dict[str, Any]],
):
    model_variant = model_name
    if load_info["quantization"] == "int8":
        model_variant = f"{model_name} (bitsandbytes INT8)"
    elif load_info["quantization"] == "original":
        model_variant = f"{model_name} (original, unquantized)"

    summary = {
        "Model Variant": model_variant,
        "Base Model": model_name,
        "Quantization": load_info["quantization"],
        "Quantization Backend": load_info["quantization_backend"],
        "BitsAndBytes Version": load_info["bitsandbytes_version"],
        "Transformers Version": load_info["transformers_version"],
        "Torch Version": load_info["torch_version"],
        "Model Load Time (s)": load_info["model_load_time_s"],
        "Model Memory Footprint (bytes)": load_info["model_memory_footprint_bytes"],
        "Model Memory Footprint (GB)": load_info["model_memory_footprint_gb"],
        "Dataset": dataset_name,
        "Dataset Config": dataset_config,
        "Split": split,
        "Text Field": text_field,
        "Corpus Documents": corpus_documents,
        "Corpus Characters": corpus_characters,
        "Max Length": max_length,
        "Stride": stride,
        "Window Count": window_count,
        "Average NLL (nats/token)": eval_stats["avg_nll"],
        "Token Perplexity": eval_stats["token_perplexity"],
        "Bits per Byte": eval_stats["bits_per_byte"],
        "Byte Perplexity": eval_stats["byte_perplexity"],
        "Evaluated Tokens": eval_stats["evaluated_tokens"],
        "Evaluated Bytes": eval_stats["evaluated_bytes"],
        "Model-only Latency (ms/token, post-warmup)": benchmark_stats["Model-only Latency (ms/token, post-warmup)"],
        "Model-only Time (s, post-warmup)": benchmark_stats["Model-only Time (s, post-warmup)"],
        "Throughput (tokens/s, post-warmup)": benchmark_stats["Throughput (tokens/s, post-warmup)"],
        "Measured Energy (J, post-warmup)": benchmark_stats["Measured Energy (J, post-warmup)"],
        "Energy per Token (J/token, post-warmup)": benchmark_stats["Energy per Token (J/token, post-warmup)"],
        "Warmup Windows": benchmark_stats["Warmup Windows"],
        "Benchmark Repeats": benchmark_stats["Benchmark Repeats"],
        "P50 Window Latency ms": benchmark_stats["P50 Window Latency ms"],
        "P95 Window Latency ms": benchmark_stats["P95 Window Latency ms"],
        "Max GPU Memory GB": get_max_gpu_memory_gb(),
        "Dense Zero Fraction": dense_zero_fraction,
        "Input Device": load_info["input_device"],
        "HF Device Map": load_info["hf_device_map"],
        "Model Dtype": load_info["model_dtype"],
        "Model Dtype Arg": load_info["model_dtype_arg"],
        "Total Script Runtime (s)": total_runtime_s,
        "Total Script Runtime (min)": total_runtime_s / 60.0,
        "Total Script Runtime (formatted)": format_elapsed_time(total_runtime_s),
    }
    if runtime_sparse_info:
        summary["Runtime 2:4 Sparse"] = True
        summary["Runtime 2:4 Converted Layers"] = int(runtime_sparse_info["converted_count"])
        summary["Runtime 2:4 Skipped Layers"] = int(runtime_sparse_info["skipped_count"])
        summary["Runtime 2:4 Masks Path"] = runtime_sparse_info["masks_path"]
        summary["Runtime 2:4 Dtype"] = runtime_sparse_info["dtype"]
    else:
        summary["Runtime 2:4 Sparse"] = False

    return {
        "summary": summary,
        "window_token_counts": eval_stats["window_token_counts"],
    }


# =========================================================
# CLI
# =========================================================
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default=MODEL_NAME)
    parser.add_argument(
        "--quantization",
        type=str,
        choices=sorted(QUANTIZATION_ALIASES),
        default=DEFAULT_QUANTIZATION,
        help=(
            "Model loading mode. Default is original/unquantized for a baseline "
            "perplexity run; use int8 for bitsandbytes LLM.int8 inference. "
            "dense is accepted as an alias for original."
        ),
    )
    parser.add_argument(
        "--model-dtype",
        type=str,
        choices=["auto", "float32", "float16", "bfloat16"],
        default=DEFAULT_MODEL_DTYPE,
        help=(
            "Dense/original model dtype. On A100, auto usually resolves to bfloat16. "
            "Use float16 for an explicit fp16 evaluation."
        ),
    )
    parser.add_argument("--dataset", type=str, default=DATASET_NAME)
    parser.add_argument("--dataset-config", type=str, default=DATASET_CONFIG)
    parser.add_argument("--eval-split", type=str, default=EVAL_SPLIT)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--max-length", type=int, default=MAX_LENGTH)
    parser.add_argument("--stride", type=int, default=STRIDE)
    parser.add_argument("--warmup-frac", type=float, default=WARMUP_FRAC)
    parser.add_argument("--benchmark-repeats", type=int, default=BENCHMARK_REPEATS)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--max-documents", type=int, default=None)
    parser.add_argument(
        "--apply-2to4-runtime",
        action="store_true",
        default=False,
        help=(
            "Load prune_masks.pt and convert masked Llama linears to "
            "SparseSemiStructuredTensor for exact 2:4 runtime evaluation."
        ),
    )
    parser.add_argument(
        "--prune-masks-path",
        type=str,
        default=None,
        help="Path to prune_masks.pt. Defaults to <local model dir>/prune_masks.pt.",
    )
    parser.add_argument(
        "--runtime-2to4-dtype",
        type=str,
        choices=["float16", "bfloat16"],
        default="float16",
        help="Dtype to use before exact 2:4 runtime conversion. float16 is the safer choice.",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
    )
    return parser.parse_args()


def finalize_args(args):
    args.quantization = normalize_quantization_mode(args.quantization)
    if args.output_json is None:
        args.output_json = f"output/llama3_8b_wikitext2_{args.eval_split}_{args.quantization}.json"
    return args


def validate_args(args):
    if args.batch_size != 1:
        raise ValueError("For rolling perplexity evaluation, --batch-size must remain 1.")
    if args.max_length < 32:
        raise ValueError("--max-length must be >= 32.")
    if args.stride < 1:
        raise ValueError("--stride must be >= 1.")
    if args.benchmark_repeats < 1:
        raise ValueError("--benchmark-repeats must be >= 1.")
    if not (0.0 <= args.warmup_frac < 1.0):
        raise ValueError("--warmup-frac must be in [0, 1).")
    if args.apply_2to4_runtime and args.quantization != "original":
        raise ValueError("--apply-2to4-runtime requires --quantization original.")
    if args.apply_2to4_runtime and not torch.cuda.is_available():
        raise RuntimeError("--apply-2to4-runtime requires CUDA.")
    if args.quantization == "int8" and args.model_dtype != "auto":
        raise ValueError("--model-dtype only applies to --quantization original.")


# =========================================================
# Main
# =========================================================
def main():
    script_start = time.perf_counter()
    args = parse_args()
    args = finalize_args(args)
    validate_args(args)

    set_seed(args.seed)
    configure_torch()

    model, tokenizer, load_info = load_model_and_tokenizer(
        args.model_name,
        quantization=args.quantization,
        model_dtype_name=args.model_dtype,
    )
    nvml_handle = init_nvml()
    runtime_sparse_info = None

    if args.apply_2to4_runtime:
        runtime_dtype = getattr(torch, args.runtime_2to4_dtype)
        masks_path = resolve_prune_masks_path(args.model_name, args.prune_masks_path)
        masks = load_prune_masks(masks_path)
        conversion = apply_exact_2to4_runtime_masks(model, masks, runtime_dtype)
        model.eval()
        runtime_sparse_info = {
            "masks_path": masks_path,
            "converted_count": int(conversion["converted_count"]),
            "skipped_count": int(conversion["skipped_count"]),
            "dtype": args.runtime_2to4_dtype,
        }
        print(
            "Applied exact 2:4 runtime conversion: "
            f"{runtime_sparse_info['converted_count']} layers "
            f"(masks={runtime_sparse_info['masks_path']}, dtype={runtime_sparse_info['dtype']})"
        )

    print("\n=== Device placement diagnostics ===")
    print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES", ""))
    if torch.cuda.is_available():
        print("Visible CUDA device count:", torch.cuda.device_count())
        print("Current CUDA device index:", torch.cuda.current_device())
        print("GPU name:", torch.cuda.get_device_name(0))
    else:
        print("CUDA not available.")
    print("hf_device_map:", load_info["hf_device_map"])

    print("\n=== Evaluation configuration ===")
    print(f"  Dataset          : {args.dataset}/{args.dataset_config}")
    print(f"  Split            : {args.eval_split}")
    print(f"  Quantization     : {args.quantization}")
    print(f"  Model dtype      : {args.model_dtype}")
    print(f"  Runtime 2:4      : {args.apply_2to4_runtime}")
    if runtime_sparse_info:
        print(f"  Runtime 2:4 dtype: {runtime_sparse_info['dtype']}")
    print(f"  Max length       : {args.max_length}")
    print(f"  Stride           : {args.stride}")
    print(f"  Benchmark repeats: {args.benchmark_repeats}")

    try:
        ds = load_dataset(args.dataset, args.dataset_config, split=args.eval_split)

        if args.max_documents is not None:
            ds = ds.select(range(min(args.max_documents, len(ds))))

        text_field = get_text_field(ds)
        corpus_text = build_corpus_text(ds, text_field)
        total_bytes = len(corpus_text.encode("utf-8"))

        encodings = tokenizer(corpus_text, return_tensors="pt", add_special_tokens=False)

        context_limit = getattr(model.config, "max_position_embeddings", args.max_length)
        effective_max_length = min(args.max_length, int(context_limit))

        windows = make_windows(
            input_ids=encodings.input_ids,
            max_length=effective_max_length,
            stride=args.stride,
        )
        print(f"Prepared {len(windows)} rolling perplexity windows.")

        eval_stats = evaluate_perplexity(
            model=model,
            windows=windows,
            total_bytes=total_bytes,
        )

        sync_cuda()
        reset_peak_memory()
        bench = benchmark_windows(
            model=model,
            windows=windows,
            warmup_frac=args.warmup_frac,
            repeats=args.benchmark_repeats,
            nvml_handle=nvml_handle,
        )

        total_runtime_s = time.perf_counter() - script_start
        dense_zero_fraction = compute_global_zero_fraction(model)

        result = build_result(
            model_name=args.model_name,
            dataset_name=args.dataset,
            dataset_config=args.dataset_config,
            split=args.eval_split,
            text_field=text_field,
            corpus_documents=len(ds),
            corpus_characters=len(corpus_text),
            max_length=effective_max_length,
            stride=args.stride,
            window_count=len(windows),
            eval_stats=eval_stats,
            benchmark_stats=bench,
            dense_zero_fraction=dense_zero_fraction,
            total_runtime_s=total_runtime_s,
            load_info=load_info,
            runtime_sparse_info=runtime_sparse_info,
        )

        output_dir = os.path.dirname(args.output_json)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, allow_nan=True)

        print(f"\nSaved JSON to {args.output_json}")
        print(f"Model variant: {result['summary']['Model Variant']}")
        print(f"Token perplexity: {result['summary']['Token Perplexity']:.6f}")
        print(f"Bits per byte: {result['summary']['Bits per Byte']:.6f}")
        print(f"Throughput: {result['summary']['Throughput (tokens/s, post-warmup)']:.3f} tokens/s")
        print(f"Total script runtime: {format_elapsed_time(total_runtime_s)}")

    finally:
        shutdown_nvml()
        cleanup()


if __name__ == "__main__":
    main()
