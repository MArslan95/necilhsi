from __future__ import annotations

"""SCOPE-HSI experiment driver.

Locked architectural contract
-----------------------------
The complete runtime follows one representation and one decision path:

    HSI -> Mamba backbone -> Q_raw -> global-RMS canonical Q -> classifier W
                                             |
                                             +-> {I_c,S_c,N_c}

Q is the only learned sample representation.  {I_c,S_c,N_c} are historical
class-discriminative statistics estimated from Q; W is the one persistent
all-seen classifier acting on the same Q.

At an incremental phase, the trainer is responsible for observing one common
coordinate movement R from paired current-class Q before/after adaptation and
for applying that same movement to the complete historical decision-relevant
state:

    I <- R I,
    S[:,:,d] <- R S[:,:,d] R^T,
    W <- R W.

This file does not implement preservation/transport mathematics.  It validates
experiment arguments, constructs the HSI protocol, dispatches the trainer, and
writes reporting-only artifacts.  Old gradient-projection preservation is not
part of this runtime.
"""

import argparse
import copy
import csv
import json
import math
import os
import random
from numbers import Integral, Real
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np
import torch

from data.hsi_dataloader_pytorch import (
    ExtractLabeledPixelIndex,
    FitHSIPreprocessorFromProtocol,
    LoadRawHSIData,
    SaveHSIPreprocessor,
)
from data.incremental_dataset import (
    IncrementalHSIDatasetManager,
    build_incremental_protocol,
)
from models.necil_model import NECILModel
from trainers.trainer import Trainer
from utils.eval import (
    class_structure_diagnostics,
    collect_reporting_features,
    decision_state_diagnostics,
    transport_drift_diagnostics,
)
from utils.qualitative_report import generate_phase_qualitative_maps
from utils.visualize import (
    save_geometry_diagnostic_figures,
    save_incremental_tsne_comparison,
)


# -----------------------------------------------------------------------------
# Generic IO / reproducibility
# -----------------------------------------------------------------------------


def str2bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    token = str(value).strip().lower()
    if token in {"1", "true", "yes", "y", "on"}:
        return True
    if token in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"invalid boolean value {value!r}")


def _parse_class_order(value: str) -> Optional[list[int]]:
    token = str(value or "").strip()
    if not token:
        return None

    parts = token.split(",")
    if any(not part.strip() for part in parts):
        raise ValueError(
            "class_order must be a comma-separated list without empty entries"
        )
    try:
        values = [int(part.strip()) for part in parts]
    except ValueError as exc:
        raise ValueError(
            "class_order must be comma-separated integer IDs"
        ) from exc

    if (
        not values
        or len(values) != len(set(values))
        or any(value < 0 for value in values)
    ):
        raise ValueError(
            "class_order must contain unique non-negative IDs"
        )
    return values


def _exact_int(
    value: object,
    *,
    name: str,
    minimum: Optional[int] = None,
    maximum: Optional[int] = None,
) -> int:
    if torch.is_tensor(value):
        if value.numel() != 1:
            raise ValueError(f"{name} must be an integer scalar")
        value = value.item()

    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be an integer")

    if isinstance(value, Integral):
        result = int(value)
    elif isinstance(value, Real):
        number = float(value)
        if not math.isfinite(number) or not number.is_integer():
            raise ValueError(f"{name} must be a finite integer")
        result = int(number)
    else:
        raise ValueError(f"{name} must be an integer")

    if minimum is not None and result < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    if maximum is not None and result > maximum:
        raise ValueError(f"{name} must be <= {maximum}")
    return result


def _finite_float(
    value: object,
    *,
    name: str,
    minimum: Optional[float] = None,
    strict_minimum: bool = False,
) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    if minimum is not None:
        if strict_minimum and not result > minimum:
            raise ValueError(f"{name} must be > {minimum}")
        if not strict_minimum and result < minimum:
            raise ValueError(f"{name} must be >= {minimum}")
    return result


def _json_safe(value: Any) -> Any:
    if torch.is_tensor(value):
        tensor = value.detach().cpu()
        return tensor.item() if tensor.numel() == 1 else tensor.tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def save_json(path: str, value: Mapping[str, Any]) -> str:
    destination = os.path.abspath(path)
    os.makedirs(os.path.dirname(destination) or ".", exist_ok=True)
    temporary = destination + ".tmp"
    try:
        with open(temporary, "w", encoding="utf-8") as stream:
            json.dump(_json_safe(value), stream, indent=2, sort_keys=True)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    except Exception:
        if os.path.exists(temporary):
            os.remove(temporary)
        raise
    return destination


def save_csv(
    path: str,
    rows: Sequence[Mapping[str, Any]],
) -> Optional[str]:
    values = list(rows)
    if not values:
        return None

    destination = os.path.abspath(path)
    os.makedirs(os.path.dirname(destination) or ".", exist_ok=True)

    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in values:
        for name in row:
            if name not in seen:
                fieldnames.append(name)
                seen.add(name)

    temporary = destination + ".tmp"
    try:
        with open(
            temporary,
            "w",
            newline="",
            encoding="utf-8-sig",
        ) as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames)
            writer.writeheader()
            for row in values:
                writer.writerow(
                    {name: _json_safe(row.get(name)) for name in fieldnames}
                )
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    except Exception:
        if os.path.exists(temporary):
            os.remove(temporary)
        raise
    return destination


def set_seed(seed: int, deterministic: bool) -> None:
    value = _exact_int(
        seed,
        name="seed",
        minimum=0,
        maximum=(2**32) - 1,
    )
    os.environ["PYTHONHASHSEED"] = str(value)

    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(value)

    torch.backends.cudnn.benchmark = not deterministic
    torch.backends.cudnn.deterministic = deterministic

    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
        torch.backends.cuda.matmul.allow_tf32 = not deterministic
    if hasattr(torch.backends, "cudnn") and hasattr(torch.backends.cudnn, "allow_tf32"):
        torch.backends.cudnn.allow_tf32 = not deterministic

    torch.use_deterministic_algorithms(bool(deterministic), warn_only=False)


def resolve_device(value: str) -> str:
    token = str(value).strip().lower().replace("gpu", "cuda")
    try:
        requested = torch.device(token)
    except (TypeError, RuntimeError) as exc:
        raise ValueError(f"invalid device {value!r}") from exc

    if requested.type == "cpu":
        return "cpu"
    if requested.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError(f"requested device {value!r} is unavailable")

    index = torch.cuda.current_device() if requested.index is None else requested.index
    if index is None or not 0 <= int(index) < torch.cuda.device_count():
        raise RuntimeError(f"CUDA device index {index} is unavailable")

    torch.cuda.set_device(int(index))
    return f"cuda:{int(index)}"


def _load_band_positions(
    path: str,
    expected_bands: int,
) -> Optional[list[float]]:
    token = str(path or "").strip()
    if not token:
        return None
    if not os.path.isfile(token):
        raise FileNotFoundError(token)

    values = np.load(token) if token.lower().endswith(".npy") else np.loadtxt(token)
    positions = np.asarray(values, dtype=np.float32).reshape(-1)
    if positions.size != int(expected_bands):
        raise ValueError(
            f"band position file has {positions.size} values; expected {expected_bands}"
        )
    if not np.isfinite(positions).all() or np.any(np.diff(positions) <= 0):
        raise ValueError("band positions must be finite and strictly increasing")
    return positions.tolist()


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "SCOPE-HSI: one canonical HSI representation Q, one persistent "
            "angular classifier, Q-derived class structure, and one shared "
            "coordinate transport during incremental learning."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    experiment = parser.add_argument_group("Experiment")
    experiment.add_argument("--dataset", default="IP")
    experiment.add_argument("--data_dir", default="./datasets")
    experiment.add_argument("--save_dir", default="./results_scope")
    experiment.add_argument("--seed", type=int, default=42)
    experiment.add_argument(
        "--protocol_seed",
        type=int,
        default=None,
        help=(
            "Optional fixed seed for class order and train/val/test membership. "
            "If omitted, the protocol uses the run seed exactly as before. "
            "For pure optimization-seed robustness with --seeds, set one fixed "
            "--protocol_seed (for example 42)."
        ),
    )
    experiment.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=None,
        help=(
            "Optional independent base-run seeds. When provided, main.py runs one "
            "fresh phase-0 experiment per seed under save_dir/seed_<seed>/ and "
            "writes aggregate mean/std reports. This is currently a base-phase "
            "robustness mode; use --seed for a normal single run."
        ),
    )
    experiment.add_argument("--device", default="cuda:0")
    experiment.add_argument("--num_workers", type=int, default=0)
    experiment.add_argument("--deterministic", type=str2bool, default=True)
    experiment.add_argument(
        "--run_mode",
        choices=("base", "incremental", "all"),
        default="base",
        help=(
            "base: train/finalize phase 0 only; incremental: restore a finalized "
            "checkpoint and continue; all: train phase 0 then all increments."
        ),
    )
    experiment.add_argument(
        "--resume_checkpoint",
        default="",
        help="Required for --run_mode incremental.",
    )

    protocol = parser.add_argument_group("Class-incremental protocol")
    protocol.add_argument("--base_classes", type=int, default=6)
    protocol.add_argument("--increment", type=int, default=6)
    protocol.add_argument(
        "--class_order",
        default="",
        help="Complete zero-based class permutation.",
    )
    protocol.add_argument("--shuffle_order", type=str2bool, default=False)
    protocol.add_argument("--train_ratio", type=float, default=0.20)
    protocol.add_argument(
        "--val_ratio",
        type=float,
        default=0.10,
        help=(
            "Reporting-only validation fraction; it never selects/restores a "
            "checkpoint or stops training."
        ),
    )
    protocol.add_argument(
        "--split_strategy",
        choices=("random_pixel", "spatial_block"),
        default="random_pixel",
    )
    protocol.add_argument("--spatial_block_size", type=int, default=33)
    protocol.add_argument(
        "--require_patch_disjoint",
        type=str2bool,
        default=False,
    )
    protocol.add_argument(
        "--require_phase_context_isolation",
        type=str2bool,
        default=None,
    )

    preprocessing = parser.add_argument_group("HSI preprocessing")
    preprocessing.add_argument("--patch_size", type=int, default=11)
    preprocessing.add_argument("--no_pca", action="store_true")
    preprocessing.add_argument("--pca_components", type=int, default=30)
    preprocessing.add_argument("--pca_whiten", type=str2bool, default=False)
    preprocessing.add_argument("--band_positions_file", default="")

    backbone = parser.add_argument_group("Ordered spectral-spatial HSI backbone")
    backbone.add_argument("--d_model", type=int, default=128)
    backbone.add_argument(
        "--spectral_token_count",
        type=int,
        default=16,
        help=(
            "K ordered contiguous spectral-region features retained in canonical "
            "Q; class statistics are I_c[K,D] and S_c[K,K,D]."
        ),
    )
    backbone.add_argument("--d_state", type=int, default=16)
    backbone.add_argument("--d_conv", type=int, default=4)
    backbone.add_argument("--expand", type=int, default=2)
    backbone.add_argument("--num_spectral_layers", type=int, default=2)
    backbone.add_argument("--num_spatial_layers", type=int, default=2)
    backbone.add_argument("--stem_norm_groups", type=int, default=8)
    backbone.add_argument("--mamba_dropout", type=float, default=0.0)

    base = parser.add_argument_group("Phase-0 optimization")
    base.add_argument("--epochs_base", type=int, default=100)
    base.add_argument("--batch_size", type=int, default=64)
    base.add_argument("--eval_batch_size", type=int, default=256)
    base.add_argument("--lr", type=float, default=1e-4)
    base.add_argument("--weight_decay", type=float, default=1e-4)
    base.add_argument("--gradient_clip", type=float, default=5.0)
    base.add_argument(
        "--spatial_d4_augmentation",
        type=str2bool,
        default=False,
        help=(
            "Optional phase-0 spatial D4 patch augmentation. It changes only H/W, "
            "never spectral/channel order, and is not part of the core SCOPE method."
        ),
    )

    incremental = parser.add_argument_group("Incremental optimization")
    incremental.add_argument("--epochs_inc", type=int, default=15)
    incremental.add_argument("--lr_inc", type=float, default=1e-4)

    reporting = parser.add_argument_group("Reporting")
    reporting.add_argument("--save_visualizations", type=str2bool, default=True)
    reporting.add_argument("--visualization_cmap", default="nipy_spectral")
    reporting.add_argument("--visualization_dpi", type=int, default=300)

    return parser


def validate_args(args: argparse.Namespace) -> None:
    positive_integer_fields = (
        "base_classes",
        "increment",
        "patch_size",
        "d_model",
        "spectral_token_count",
        "d_state",
        "d_conv",
        "expand",
        "num_spectral_layers",
        "num_spatial_layers",
        "stem_norm_groups",
        "epochs_base",
        "epochs_inc",
        "batch_size",
        "eval_batch_size",
        "spatial_block_size",
        "visualization_dpi",
    )
    for name in positive_integer_fields:
        _exact_int(getattr(args, name), name=name, minimum=1)

    _exact_int(args.num_workers, name="num_workers", minimum=0)
    _exact_int(args.seed, name="seed", minimum=0, maximum=(2**32) - 1)
    if args.protocol_seed is not None:
        _exact_int(
            args.protocol_seed,
            name="protocol_seed",
            minimum=0,
            maximum=(2**32) - 1,
        )

    if args.seeds is not None:
        seed_values = [
            _exact_int(
                value,
                name="seeds value",
                minimum=0,
                maximum=(2**32) - 1,
            )
            for value in args.seeds
        ]
        if not seed_values:
            raise ValueError("--seeds requires at least one seed")
        if len(seed_values) != len(set(seed_values)):
            raise ValueError("--seeds must contain unique values")
        if str(args.run_mode) != "base":
            raise ValueError(
                "--seeds is currently for independent base-phase validation only; "
                "use --seed for incremental/all runs"
            )

    if not bool(args.no_pca):
        _exact_int(args.pca_components, name="pca_components", minimum=1)

    if int(args.base_classes) < 2:
        raise ValueError("base_classes must be at least 2")
    if int(args.spectral_token_count) < 2:
        raise ValueError("spectral_token_count must be at least 2")

    if int(args.patch_size) % 2 == 0:
        raise ValueError("patch_size must be odd")

    groups = int(args.stem_norm_groups)
    width = int(args.d_model)
    if groups > width or width % groups != 0:
        raise ValueError("stem_norm_groups must divide d_model")

    train_ratio = _finite_float(
        args.train_ratio,
        name="train_ratio",
        minimum=0.0,
        strict_minimum=True,
    )
    val_ratio = _finite_float(args.val_ratio, name="val_ratio", minimum=0.0)
    if train_ratio + val_ratio >= 1.0:
        raise ValueError("train_ratio + val_ratio must be < 1")

    dropout = _finite_float(args.mamba_dropout, name="mamba_dropout", minimum=0.0)
    if dropout >= 1.0:
        raise ValueError("mamba_dropout must be < 1")

    _finite_float(args.lr, name="lr", minimum=0.0, strict_minimum=True)
    _finite_float(args.lr_inc, name="lr_inc", minimum=0.0, strict_minimum=True)
    _finite_float(args.weight_decay, name="weight_decay", minimum=0.0)
    _finite_float(args.gradient_clip, name="gradient_clip", minimum=0.0)

    if str(args.run_mode) == "incremental" and not str(args.resume_checkpoint).strip():
        raise ValueError("--resume_checkpoint is required for --run_mode incremental")

    class_order = _parse_class_order(args.class_order)
    if class_order is not None and bool(args.shuffle_order):
        raise ValueError("class_order and shuffle_order are mutually exclusive")

    if args.require_phase_context_isolation is not None and not isinstance(
        args.require_phase_context_isolation,
        bool,
    ):
        raise ValueError("require_phase_context_isolation must be bool or None")
    if not isinstance(args.spatial_d4_augmentation, bool):
        raise ValueError("spatial_d4_augmentation must be bool")


# -----------------------------------------------------------------------------
# Dataset / protocol construction
# -----------------------------------------------------------------------------


def _build_data(
    args: argparse.Namespace,
) -> tuple[IncrementalHSIDatasetManager, Dict[str, Any], Mapping[str, Any]]:
    raw_cube, gt, class_count, names, _, label_policy = LoadRawHSIData(
        args.dataset,
        base_dir=args.data_dir,
    )
    labels, coords = ExtractLabeledPixelIndex(gt, label_policy=label_policy)

    if int(args.base_classes) > int(class_count):
        raise ValueError(
            f"base_classes={args.base_classes} exceeds dataset classes={class_count}"
        )

    class_order = _parse_class_order(args.class_order)
    if class_order is not None and set(class_order) != set(range(int(class_count))):
        raise ValueError(
            "class_order must be a permutation of "
            f"0..{int(class_count) - 1}"
        )

    protocol_seed = (
        int(args.seed)
        if args.protocol_seed is None
        else int(args.protocol_seed)
    )

    protocol = build_incremental_protocol(
        labels,
        coords,
        gt_shape=gt.shape,
        class_count=int(class_count),
        base_classes=int(args.base_classes),
        increment=int(args.increment),
        train_ratio=float(args.train_ratio),
        val_ratio=float(args.val_ratio),
        seed=protocol_seed,
        shuffle_order=bool(args.shuffle_order),
        class_order=class_order,
        target_names=names,
        split_strategy=str(args.split_strategy),
        spatial_block_size=int(args.spatial_block_size),
    )

    required_protocol_fields = {
        "phase_to_classes",
        "phase_train_indices",
        "base_train_indices",
        "validation_enabled",
        "split_by_class",
    }
    missing_protocol_fields = required_protocol_fields - set(protocol)
    if missing_protocol_fields:
        raise RuntimeError(
            "incremental protocol is incompatible with the current SCOPE-HSI "
            f"runtime; missing fields={sorted(missing_protocol_fields)}"
        )
    if not isinstance(protocol["validation_enabled"], bool):
        raise RuntimeError("protocol validation_enabled must be bool")

    processed_cube, ordered_cube, preprocessing_state = FitHSIPreprocessorFromProtocol(
        raw_cube,
        coords=coords,
        protocol=protocol,
        apply_reduction=not bool(args.no_pca),
        reduction_method="PCA",
        n_components=int(args.pca_components),
        whiten=bool(args.pca_whiten),
    )

    args.num_bands = int(processed_cube.shape[2])
    args.spectral_bands = int(ordered_cube.shape[2])
    args.raw_num_bands = int(ordered_cube.shape[2])

    if int(args.spectral_token_count) > int(args.spectral_bands):
        raise ValueError(
            "spectral_token_count cannot exceed the number of ordered raw bands "
            f"({args.spectral_bands})"
        )

    args.band_positions = _load_band_positions(
        args.band_positions_file,
        args.spectral_bands,
    )
    args.band_wavelengths = None

    # The spectral path uses the same phase-0 TRAIN-only raw-band normalization
    # fitted by the preprocessor; there is no independent second fitting scope.
    args.spectral_normalization_mean = np.asarray(
        preprocessing_state["normalization_mean"],
        dtype=np.float32,
    ).tolist()
    args.spectral_normalization_std = np.asarray(
        preprocessing_state["normalization_std"],
        dtype=np.float32,
    ).tolist()

    manager = IncrementalHSIDatasetManager(
        processed_cube=processed_cube,
        ordered_spectral_cube=ordered_cube,
        labels=labels,
        coords=coords,
        protocol=protocol,
        target_names=names,
        gt_shape=gt.shape,
        patch_size=int(args.patch_size),
        num_workers=int(args.num_workers),
        device=str(args.device),
        seed=int(args.seed),
        require_patch_disjoint=bool(args.require_patch_disjoint),
        require_phase_context_isolation=args.require_phase_context_isolation,
        context_policy="full_scene_reflect",
    )

    if not hasattr(manager, "validation_enabled") or not isinstance(
        manager.validation_enabled,
        bool,
    ):
        raise RuntimeError(
            "IncrementalHSIDatasetManager must expose boolean validation_enabled"
        )
    if manager.validation_enabled != protocol["validation_enabled"]:
        raise RuntimeError("manager validation capability disagrees with protocol")
    if not hasattr(manager, "base_train_indices"):
        raise RuntimeError(
            "IncrementalHSIDatasetManager is missing canonical base_train_indices"
        )

    expected_phase_sizes = [int(args.base_classes)]
    remaining = int(class_count) - int(args.base_classes)
    while remaining > 0:
        current = min(int(args.increment), remaining)
        expected_phase_sizes.append(current)
        remaining -= current

    actual_phase_sizes = [
        len(manager.phase_to_classes[phase])
        for phase in sorted(manager.phase_to_classes)
    ]
    if actual_phase_sizes != expected_phase_sizes:
        raise RuntimeError(
            "constructed class-incremental schedule is inconsistent: "
            f"expected={expected_phase_sizes}, actual={actual_phase_sizes}"
        )

    split_counts = {
        int(class_id): {
            split: int(np.asarray(protocol["split_by_class"][class_id][split]).size)
            for split in ("train", "val", "test")
        }
        for class_id in range(int(class_count))
    }

    data_summary: Dict[str, Any] = {
        "dataset": str(args.dataset),
        "optimization_seed": int(args.seed),
        "protocol_seed": int(protocol_seed),
        "raw_shape": list(raw_cube.shape),
        "gt_shape": list(gt.shape),
        "labeled_sample_count": int(labels.size),
        "class_count": int(class_count),
        "phase_sizes": list(actual_phase_sizes),
        "phase_schedule": {
            int(phase): [int(value) for value in classes]
            for phase, classes in protocol["phase_to_classes"].items()
        },
        "class_order_original_ids": list(manager.class_order_original_ids),
        "class_names_global_order": list(manager.target_names),
        "processed_bands": int(processed_cube.shape[2]),
        "raw_spectral_bands": int(ordered_cube.shape[2]),
        "spectral_region_count": int(args.spectral_token_count),
        "sample_representation": "Q[K,D] = Q_raw / RMS(Q_raw)",
        "persistent_class_statistics": "{I_c[K,D],S_c[K,K,D],N_c}",
        "persistent_classifier": "one angular W_c[K,D] row per seen class",
        "incremental_consistency_rule": "one R: I->RI, S->RSR^T, W->RW",
        "preprocessing_fit_scope": str(preprocessing_state["fit_scope"]),
        "preprocessing_fit_pixel_count": int(preprocessing_state["fit_pixel_count"]),
        "preprocessing_frozen_across_phases": True,
        "validation_enabled": bool(manager.validation_enabled),
        "split_strategy": manager.split_strategy,
        "spatial_partition_mode": manager.spatial_partition_mode,
        "require_patch_disjoint": manager.require_patch_disjoint,
        "require_phase_context_isolation": manager.require_phase_context_isolation,
        "context_policy": manager.context_policy,
        "spatial_leakage_audit": manager.spatial_leakage_report(),
        "split_counts_by_global_class": split_counts,
        "current_phase_training_classes_only": True,
        "cumulative_seen_class_evaluation": True,
        "stores_old_training_samples": False,
        "stores_old_features": False,
        "replay": False,
        "teacher": False,
        "knowledge_distillation": False,
    }
    return manager, data_summary, preprocessing_state


# -----------------------------------------------------------------------------
# Shared reporting helpers
# -----------------------------------------------------------------------------


def _mapping_value(mapping: Any, class_id: int) -> Any:
    if not isinstance(mapping, Mapping):
        return None
    if class_id in mapping:
        return mapping[class_id]
    return mapping.get(str(class_id))


def _metric_summary(metrics: Optional[Mapping[str, Any]]) -> Optional[Dict[str, Any]]:
    if metrics is None:
        return None
    return {
        "oa": float(metrics["overall_accuracy"]),
        "ba": float(metrics["balanced_accuracy"]),
        "min_class_accuracy": float(metrics["minimum_class_accuracy"]),
        "kappa": float(metrics["kappa"]),
        "cosine_margin": float(metrics["mean_cosine_margin"]),
        "feature_rms": float(metrics["mean_feature_rms"]),
    }


def _history_rows(history: Sequence[Mapping[str, Any]]) -> list[Dict[str, Any]]:
    rows: list[Dict[str, Any]] = []
    for record in history:
        rows.append(
            {
                "epoch": int(record["epoch"]),
                "learning_rate": float(record["learning_rate"]),
                "classification_loss": float(record["classification_loss"]),
                "train_oa": float(record["train_oa"]),
                "train_ba": float(record["train_ba"]),
                "validation_oa": record.get("validation_oa"),
                "validation_ba": record.get("validation_ba"),
            }
        )
    return rows


def _classwise_rows(
    *,
    result: Mapping[str, Any],
    class_ids: Sequence[int],
    class_names: Sequence[str],
) -> list[Dict[str, Any]]:
    train = result["train"]
    validation = result.get("validation")
    test = result["test"]

    rows: list[Dict[str, Any]] = []
    for class_id in [int(v) for v in class_ids]:
        rows.append(
            {
                "class_id": class_id,
                "display_id": class_id + 1,
                "class_name": str(class_names[class_id]),
                "train_samples": _mapping_value(train.get("class_counts"), class_id),
                "train_accuracy": _mapping_value(train.get("per_class_accuracy"), class_id),
                "validation_samples": (
                    None
                    if not isinstance(validation, Mapping)
                    else _mapping_value(validation.get("class_counts"), class_id)
                ),
                "validation_accuracy": (
                    None
                    if not isinstance(validation, Mapping)
                    else _mapping_value(validation.get("per_class_accuracy"), class_id)
                ),
                "test_samples": _mapping_value(test.get("class_counts"), class_id),
                "test_accuracy": _mapping_value(test.get("per_class_accuracy"), class_id),
                "test_cosine_margin": _mapping_value(
                    test.get("per_class_mean_cosine_margin"),
                    class_id,
                ),
            }
        )
    return rows


def _confusion_rows(
    metrics: Mapping[str, Any],
    class_ids: Sequence[int],
) -> list[Dict[str, Any]]:
    matrix = np.asarray(metrics["confusion_matrix"], dtype=np.int64)
    ids = [int(v) for v in class_ids]
    if matrix.shape != (len(ids), len(ids)):
        raise RuntimeError("confusion matrix shape disagrees with class_ids")

    rows: list[Dict[str, Any]] = []
    for row_index, class_id in enumerate(ids):
        row: Dict[str, Any] = {
            "true_class_id": class_id,
            "true_display_id": class_id + 1,
        }
        for column_index, predicted_id in enumerate(ids):
            row[f"pred_{predicted_id}"] = int(matrix[row_index, column_index])
        rows.append(row)
    return rows


def _print_metrics(name: str, metrics: Optional[Mapping[str, Any]]) -> None:
    if metrics is None:
        print(f"{name:<10} | disabled")
        return
    print(
        f"{name:<10} | "
        f"OA={100.0 * float(metrics['overall_accuracy']):.2f}% | "
        f"BA={100.0 * float(metrics['balanced_accuracy']):.2f}% | "
        f"MinClass={100.0 * float(metrics['minimum_class_accuracy']):.2f}% | "
        f"Kappa={float(metrics['kappa']):.4f} | "
        f"CosGap={float(metrics['mean_cosine_margin']):.4f} | "
        f"QRMS={float(metrics['mean_feature_rms']):.4f}"
    )


# -----------------------------------------------------------------------------
# Phase-0 reporting
# -----------------------------------------------------------------------------


def _report_base_phase(
    *,
    args: argparse.Namespace,
    dataset: IncrementalHSIDatasetManager,
    model: NECILModel,
    result: Mapping[str, Any],
) -> Dict[str, Any]:
    class_ids = [int(value) for value in result["class_ids"]]
    model.validate_finalized_state()

    phase_dir = os.path.join(args.save_dir, "phase_0")
    os.makedirs(phase_dir, exist_ok=True)

    class_structure = class_structure_diagnostics(
        model,
        class_ids,
        target_names=dataset.target_names,
    )
    decision_state = decision_state_diagnostics(model, class_ids)

    history_path = save_csv(
        os.path.join(phase_dir, "training_history.csv"),
        _history_rows(result["history"]),
    )
    classwise_path = save_csv(
        os.path.join(phase_dir, "classwise_metrics.csv"),
        _classwise_rows(
            result=result,
            class_ids=class_ids,
            class_names=dataset.target_names,
        ),
    )
    confusion_path = save_csv(
        os.path.join(phase_dir, "confusion_matrix.csv"),
        _confusion_rows(result["test"], class_ids),
    )

    class_structure_path = save_json(
        os.path.join(phase_dir, "class_structure_report.json"),
        {
            "phase": 0,
            "class_structure": class_structure,
            "decision_state": decision_state,
        },
    )

    if bool(args.save_visualizations):
        figure_dir = os.path.join(phase_dir, "figures")
        save_geometry_diagnostic_figures(
            output_dir=os.path.join(figure_dir, "class_structure"),
            train=result["train"],
            validation=result["validation"],
            test=result["test"],
            structural_geometry=class_structure,
            class_ids=class_ids,
            target_names=dataset.target_names,
            dpi=args.visualization_dpi,
        )
        generate_phase_qualitative_maps(
            model=model,
            dataset=dataset,
            phase=0,
            output_dir=os.path.join(figure_dir, "qualitative"),
            device=args.device,
            batch_size=args.eval_batch_size,
            cmap_name=args.visualization_cmap,
            dpi=args.visualization_dpi,
        )

    phase_report = {
        "phase": 0,
        "seen_class_ids": class_ids,
        "epochs": int(result["final_epoch"]),
        "train": _metric_summary(result["train"]),
        "validation": _metric_summary(result["validation"]),
        "test": _metric_summary(result["test"]),
        "mean_identity_decision_margin": float(
            decision_state["mean_identity_decision_margin"]
        ),
        "protocol": {
            "exemplar_memory_size": 0,
            "replay": False,
            "stored_old_samples": 0,
            "stored_old_features": 0,
            "teacher": False,
            "knowledge_distillation": False,
        },
    }
    phase_report_path = save_json(
        os.path.join(phase_dir, "phase_report.json"),
        phase_report,
    )

    print("\nPhase 0 finalized.")
    _print_metrics("Train", result["train"])
    _print_metrics("Validation", result["validation"])
    _print_metrics("Test", result["test"])
    print(
        "I/W decision margin | "
        f"mean={float(decision_state['mean_identity_decision_margin']):.6f}"
    )
    print(f"Checkpoint       : {result['checkpoint']}")
    print(f"Phase report     : {phase_report_path}")
    print(f"Structure report : {class_structure_path}")
    print(f"Training history : {history_path}")
    print(f"Classwise metrics: {classwise_path}")
    print(f"Confusion matrix : {confusion_path}")
    return phase_report


# -----------------------------------------------------------------------------
# Incremental reporting -- consumes, never estimates, the trainer's phase R
# -----------------------------------------------------------------------------


def _load_reporting_reference_model(
    *,
    args: argparse.Namespace,
    checkpoint_path: str,
    expected_class_ids: Sequence[int],
) -> NECILModel:
    source = os.path.abspath(str(checkpoint_path))
    if not os.path.isfile(source):
        raise FileNotFoundError(source)

    payload = torch.load(source, map_location=args.device)
    if not isinstance(payload, Mapping):
        raise RuntimeError("reporting checkpoint payload must be a mapping")
    state = payload.get("model_state")
    if not isinstance(state, Mapping):
        raise RuntimeError("reporting checkpoint lacks model_state")

    reference = NECILModel(args)
    reference.to(torch.device(args.device))
    try:
        reference.load_state_dict(state, strict=True)
    except RuntimeError as exc:
        raise RuntimeError(
            "previous checkpoint is incompatible with the current model"
        ) from exc

    reference.eval()
    reference.validate_finalized_state()
    expected = [int(v) for v in expected_class_ids]
    if list(reference.classifier_class_ids) != expected:
        raise RuntimeError(
            "previous checkpoint classifier classes do not match old classes"
        )
    if list(reference.committed_class_ids) != expected:
        raise RuntimeError(
            "previous checkpoint class statistics do not match old classes"
        )
    return reference


def _incremental_representation_report(
    *,
    args: argparse.Namespace,
    dataset: IncrementalHSIDatasetManager,
    current_model: NECILModel,
    phase: int,
    old_class_ids: Sequence[int],
    new_class_ids: Sequence[int],
    previous_checkpoint_path: str,
    phase_transform: torch.Tensor,
    figure_output_dir: Optional[str],
) -> Dict[str, Any]:
    phase_id = int(phase)
    if phase_id <= 0:
        raise ValueError("incremental representation reporting requires phase > 0")

    old_ids = [int(v) for v in old_class_ids]
    new_ids = [int(v) for v in new_class_ids]
    seen_ids = old_ids + new_ids
    if not old_ids or not new_ids or len(seen_ids) != len(set(seen_ids)):
        raise ValueError("old/new class IDs are invalid")

    reporting_loader = dataset.get_reporting_dataloader(
        phase_id,
        batch_size=int(args.eval_batch_size),
    )
    previous_model = _load_reporting_reference_model(
        args=args,
        checkpoint_path=previous_checkpoint_path,
        expected_class_ids=old_ids,
    )

    try:
        previous_old = collect_reporting_features(
            previous_model,
            reporting_loader,
            class_ids=old_ids,
            device=args.device,
            require_coordinates=True,
        )
        current_old = collect_reporting_features(
            current_model,
            reporting_loader,
            class_ids=old_ids,
            device=args.device,
            require_coordinates=True,
        )

        transport_report = transport_drift_diagnostics(
            previous_old,
            current_old,
            old_class_ids=old_ids,
            transform=phase_transform,
        )

        tsne_path: Optional[str] = None
        if figure_output_dir is not None:
            current_seen = collect_reporting_features(
                current_model,
                reporting_loader,
                class_ids=seen_ids,
                device=args.device,
                require_coordinates=True,
            )
            os.makedirs(figure_output_dir, exist_ok=True)
            tsne_path = save_incremental_tsne_comparison(
                os.path.join(
                    figure_output_dir,
                    f"phase_{phase_id - 1:02d}_vs_phase_{phase_id:02d}_tsne.png",
                ),
                base_features=previous_old["ordered_features"],
                base_labels=previous_old["labels"],
                current_features=current_seen["ordered_features"],
                current_labels=current_seen["labels"],
                old_class_ids=old_ids,
                new_class_ids=new_ids,
                target_names=dataset.target_names,
                total_classes=len(dataset.target_names),
                cmap_name=str(args.visualization_cmap),
                seed=int(args.seed),
                dpi=int(args.visualization_dpi),
            )

        return {"transport_drift": transport_report, "tsne": tsne_path}
    finally:
        del previous_model
        if torch.cuda.is_available() and torch.device(args.device).type == "cuda":
            torch.cuda.empty_cache()


def _incremental_history_rows(
    history: Sequence[Mapping[str, Any]],
) -> list[Dict[str, Any]]:
    rows: list[Dict[str, Any]] = []
    for item in history:
        rows.append(
            {
                "epoch": int(item["epoch"]),
                "learning_rate": float(item["learning_rate"]),
                "classification_loss": float(item["classification_loss"]),
                "new_train_oa": float(item["new_train_oa"]),
                "new_train_ba": float(item["new_train_ba"]),
                "validation_new_ba": item.get("validation_new_ba"),
                "transport_raw_feature_change_rms": item.get(
                    "transport_raw_feature_change_rms"
                ),
                "transport_aligned_feature_residual_rms": item.get(
                    "transport_aligned_feature_residual_rms"
                ),
                "transport_orthogonality_residual": item.get(
                    "transport_orthogonality_residual"
                ),
            }
        )
    return rows


def _report_incremental_phase(
    *,
    args: argparse.Namespace,
    dataset: IncrementalHSIDatasetManager,
    model: NECILModel,
    result: Mapping[str, Any],
    previous_checkpoint_path: str,
) -> Dict[str, Any]:
    phase = int(result["phase"])
    old_ids = [int(v) for v in result["old_class_ids"]]
    new_ids = [int(v) for v in result["new_class_ids"]]
    seen_ids = [int(v) for v in result["seen_class_ids"]]
    if seen_ids != old_ids + new_ids:
        raise RuntimeError("incremental seen classes must equal old followed by new")

    if "phase_transform" not in result:
        raise RuntimeError(
            "incremental trainer is not aligned with locked SCOPE: "
            "missing transient phase_transform"
        )
    phase_transform = torch.as_tensor(
        result["phase_transform"],
        device=model.device,
        dtype=model.dtype,
    )
    expected_transform_shape = (
        int(model.spectral_token_count),
        int(model.spectral_token_count),
    )
    if tuple(phase_transform.shape) != expected_transform_shape:
        raise RuntimeError(
            f"phase_transform must have shape {expected_transform_shape}"
        )

    model.validate_finalized_state()
    phase_dir = os.path.join(args.save_dir, f"phase_{phase}")
    os.makedirs(phase_dir, exist_ok=True)
    figure_dir = os.path.join(phase_dir, "figures")

    history_path = save_csv(
        os.path.join(phase_dir, "training_history.csv"),
        _incremental_history_rows(result["history"]),
    )
    classwise_path = save_csv(
        os.path.join(phase_dir, "classwise_metrics.csv"),
        [
            {
                "class_id": class_id,
                "display_id": class_id + 1,
                "class_name": str(dataset.target_names[class_id]),
                "phase_role": "old" if class_id in set(old_ids) else "new",
                "test_samples": _mapping_value(
                    result["cumulative_test"].get("class_counts"), class_id
                ),
                "test_accuracy": _mapping_value(
                    result["cumulative_test"].get("per_class_accuracy"), class_id
                ),
                "test_cosine_margin": _mapping_value(
                    result["cumulative_test"].get("per_class_mean_cosine_margin"),
                    class_id,
                ),
            }
            for class_id in seen_ids
        ],
    )
    confusion_path = save_csv(
        os.path.join(phase_dir, "confusion_matrix.csv"),
        _confusion_rows(result["cumulative_test"], seen_ids),
    )

    representation_report = _incremental_representation_report(
        args=args,
        dataset=dataset,
        current_model=model,
        phase=phase,
        old_class_ids=old_ids,
        new_class_ids=new_ids,
        previous_checkpoint_path=previous_checkpoint_path,
        phase_transform=phase_transform,
        figure_output_dir=(
            os.path.join(figure_dir, "representation")
            if bool(args.save_visualizations)
            else None
        ),
    )

    class_structure = class_structure_diagnostics(
        model,
        seen_ids,
        target_names=dataset.target_names,
    )
    decision_state = decision_state_diagnostics(model, seen_ids)

    diagnostic_path = save_json(
        os.path.join(phase_dir, "class_structure_report.json"),
        {
            "phase": phase,
            "old_class_ids": old_ids,
            "new_class_ids": new_ids,
            "class_structure": class_structure,
            "decision_state": decision_state,
            "transport_drift": representation_report["transport_drift"],
        },
    )

    old_group = result["old_new"]["old"]
    new_group = result["old_new"]["new"]
    transport_drift = representation_report["transport_drift"]

    phase_report = {
        "phase": phase,
        "old_class_ids": old_ids,
        "new_class_ids": new_ids,
        "seen_class_ids": seen_ids,
        "epochs": int(result["final_epoch"]),
        "cumulative": _metric_summary(result["cumulative_test"]),
        "old": {
            "ba": float(old_group["balanced_accuracy"]),
            "min_class_accuracy": float(old_group["minimum_class_accuracy"]),
            "cosine_margin": float(old_group["mean_cosine_margin"]),
        },
        "new": {
            "ba": float(new_group["balanced_accuracy"]),
            "min_class_accuracy": float(new_group["minimum_class_accuracy"]),
            "cosine_margin": float(new_group["mean_cosine_margin"]),
        },
        "harmonic_old_new_accuracy": float(
            result["old_new"]["harmonic_old_new_accuracy"]
        ),
        "forgetting": {
            "old_balanced_accuracy_before": float(
                result["forgetting"]["old_balanced_accuracy_before"]
            ),
            "old_balanced_accuracy_after": float(
                result["forgetting"]["old_balanced_accuracy_after"]
            ),
            "old_balanced_accuracy_drop": float(
                result["forgetting"]["old_balanced_accuracy_drop"]
            ),
        },
        "transport": {
            "raw_feature_change_rms": float(
                transport_drift["raw_feature_change_rms"]
            ),
            "aligned_feature_residual_rms": float(
                transport_drift["aligned_feature_residual_rms"]
            ),
            "mean_raw_identity_drift_rms": float(
                transport_drift["mean_raw_identity_drift_rms"]
            ),
            "mean_aligned_identity_residual_rms": float(
                transport_drift["mean_aligned_identity_residual_rms"]
            ),
            "mean_raw_structure_drift_rms": float(
                transport_drift["mean_raw_structure_drift_rms"]
            ),
            "mean_aligned_structure_residual_rms": float(
                transport_drift["mean_aligned_structure_residual_rms"]
            ),
        },
        "mean_identity_decision_margin": float(
            decision_state["mean_identity_decision_margin"]
        ),
        "protocol": {
            "exemplar_memory_size": 0,
            "replay": False,
            "stored_old_samples": 0,
            "stored_old_features": 0,
            "teacher": False,
            "knowledge_distillation": False,
        },
    }
    phase_report_path = save_json(
        os.path.join(phase_dir, "phase_report.json"),
        phase_report,
    )

    if bool(args.save_visualizations):
        generate_phase_qualitative_maps(
            model=model,
            dataset=dataset,
            phase=phase,
            output_dir=os.path.join(figure_dir, "qualitative"),
            device=args.device,
            batch_size=args.eval_batch_size,
            cmap_name=args.visualization_cmap,
            dpi=args.visualization_dpi,
        )

    print(f"\nPhase {phase} finalized.")
    _print_metrics("Cumulative", result["cumulative_test"])
    print(
        f"Old BA={100.0 * float(old_group['balanced_accuracy']):.2f}% | "
        f"New BA={100.0 * float(new_group['balanced_accuracy']):.2f}% | "
        f"H={100.0 * float(result['old_new']['harmonic_old_new_accuracy']):.2f}%"
    )
    print(
        "Transport | "
        f"raw Q={float(transport_drift['raw_feature_change_rms']):.6f} | "
        f"aligned Q={float(transport_drift['aligned_feature_residual_rms']):.6f} | "
        f"raw I={float(transport_drift['mean_raw_identity_drift_rms']):.6f} | "
        f"aligned I={float(transport_drift['mean_aligned_identity_residual_rms']):.6f} | "
        f"raw S={float(transport_drift['mean_raw_structure_drift_rms']):.6f} | "
        f"aligned S={float(transport_drift['mean_aligned_structure_residual_rms']):.6f}"
    )
    print(f"Checkpoint       : {result['checkpoint']}")
    print(f"Phase report     : {phase_report_path}")
    print(f"Structure report : {diagnostic_path}")
    print(f"Training history : {history_path}")
    print(f"Classwise metrics: {classwise_path}")
    print(f"Confusion matrix : {confusion_path}")
    if representation_report["tsne"] is not None:
        print(f"Joint t-SNE      : {representation_report['tsne']}")
    return phase_report


# -----------------------------------------------------------------------------
# Experiment
# -----------------------------------------------------------------------------


def run_experiment(args: argparse.Namespace) -> Dict[str, Any]:
    os.makedirs(args.save_dir, exist_ok=True)

    dataset, data_summary, preprocessing_state = _build_data(args)
    save_json(os.path.join(args.save_dir, "data_protocol.json"), data_summary)
    SaveHSIPreprocessor(
        os.path.join(args.save_dir, "phase_0_preprocessor.npz"),
        preprocessing_state,
    )
    save_json(os.path.join(args.save_dir, "experiment_config.json"), vars(args))

    model = NECILModel(args)
    trainer = Trainer(model, dataset, args)
    phase_reports: Dict[int, Dict[str, Any]] = {}

    mode = str(args.run_mode)
    if mode in {"base", "all"}:
        base_result = trainer.run_base_only(
            epochs=args.epochs_base,
            batch_size=args.batch_size,
            lr=args.lr,
        )
        phase_reports[0] = _report_base_phase(
            args=args,
            dataset=dataset,
            model=model,
            result=base_result,
        )
        previous_checkpoint_path = str(base_result["checkpoint"])
        next_phase: Optional[int] = 1
    else:
        restored = trainer.load_checkpoint(args.resume_checkpoint)
        previous_checkpoint_path = str(restored["checkpoint"])
        next_phase = restored["next_phase"]
        if next_phase is None:
            raise RuntimeError("checkpoint already contains the final protocol phase")

    if mode in {"incremental", "all"}:
        if next_phase is None:
            raise RuntimeError("no incremental phase remains")
        for phase in range(int(next_phase), int(dataset.nb_tasks)):
            result = trainer.run_incremental_phase(
                phase=phase,
                epochs=args.epochs_inc,
                batch_size=args.batch_size,
                lr=args.lr_inc,
            )
            phase_reports[phase] = _report_incremental_phase(
                args=args,
                dataset=dataset,
                model=model,
                result=result,
                previous_checkpoint_path=previous_checkpoint_path,
            )
            previous_checkpoint_path = str(result["checkpoint"])

    finalized_phases = [int(v) for v in dataset.finalized_phases]
    if not finalized_phases:
        raise RuntimeError("experiment completed without a finalized phase")
    if finalized_phases != list(range(len(finalized_phases))):
        raise RuntimeError("finalized phases are not contiguous from zero")

    latest = finalized_phases[-1]
    model.validate_finalized_state()
    latest_loader = dataset.get_reporting_dataloader(
        latest,
        batch_size=args.eval_batch_size,
    )
    latest_metrics = trainer.evaluate_loader(
        latest_loader,
        class_ids=list(model.classifier_class_ids),
        target_class_ids=list(model.classifier_class_ids),
    )

    run_summary = {
        "dataset": str(args.dataset),
        "seed": int(args.seed),
        "run_mode": mode,
        "finalized_phases": finalized_phases,
        "seen_class_ids": list(model.classifier_class_ids),
        "latest_test": _metric_summary(latest_metrics),
        "protocol": {
            "exemplar_memory_size": 0,
            "replay": False,
            "stored_old_samples": 0,
            "stored_old_features": 0,
            "teacher": False,
            "knowledge_distillation": False,
        },
    }
    summary_path = save_json(
        os.path.join(args.save_dir, "run_summary.json"),
        run_summary,
    )
    run_summary["run_summary"] = summary_path

    print("\nRun complete.")
    print(f"Finalized phases : {finalized_phases}")
    print(f"Seen classes     : {list(model.classifier_class_ids)}")
    print(f"Run summary      : {summary_path}")
    return run_summary



# -----------------------------------------------------------------------------
# Independent multi-seed base validation
# -----------------------------------------------------------------------------


def _sample_statistics(values: Sequence[float]) -> Dict[str, float]:
    array = np.asarray([float(value) for value in values], dtype=np.float64)
    if array.ndim != 1 or array.size == 0 or not np.isfinite(array).all():
        raise ValueError("aggregate values must be a non-empty finite vector")
    return {
        "mean": float(array.mean()),
        "std": float(array.std(ddof=1)) if array.size > 1 else 0.0,
        "min": float(array.min()),
        "max": float(array.max()),
    }


def _load_json_mapping(path: str) -> Dict[str, Any]:
    source = os.path.abspath(path)
    if not os.path.isfile(source):
        raise FileNotFoundError(source)
    with open(source, "r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, Mapping):
        raise RuntimeError(f"{source} must contain a JSON object")
    return dict(value)


def _load_csv_rows(path: str) -> list[Dict[str, str]]:
    source = os.path.abspath(path)
    if not os.path.isfile(source):
        raise FileNotFoundError(source)
    with open(source, "r", newline="", encoding="utf-8-sig") as stream:
        return [dict(row) for row in csv.DictReader(stream)]


def _optional_csv_float(value: object, *, name: str) -> Optional[float]:
    if value is None:
        return None
    token = str(value).strip()
    if token == "" or token.lower() == "none":
        return None
    return _finite_float(token, name=name)


def _csv_exact_int(
    value: object,
    *,
    name: str,
    minimum: Optional[int] = None,
    maximum: Optional[int] = None,
) -> int:
    """Parse an exact integer field returned as text by csv.DictReader.

    Keep _exact_int() strict for runtime values. Text conversion is allowed only
    at this serialization boundary, where csv.DictReader is documented to
    return strings.
    """
    if value is None:
        raise ValueError(f"{name} is missing")
    token = str(value).strip()
    if not token:
        raise ValueError(f"{name} is empty")
    try:
        parsed = int(token, 10)
    except ValueError as exc:
        raise ValueError(f"{name} must contain an integer") from exc
    return _exact_int(
        parsed,
        name=name,
        minimum=minimum,
        maximum=maximum,
    )


def _aggregate_multiseed_classwise(
    *,
    per_seed_rows: Sequence[tuple[int, Sequence[Mapping[str, Any]]]],
) -> list[Dict[str, Any]]:
    if not per_seed_rows:
        raise ValueError("per_seed_rows cannot be empty")

    by_class: Dict[int, Dict[str, Any]] = {}
    expected_ids: Optional[list[int]] = None

    for seed, rows in per_seed_rows:
        class_ids: list[int] = []
        for row in rows:
            class_id = _csv_exact_int(
                row["class_id"],
                name=f"class_id for seed {seed}",
                minimum=0,
            )
            class_ids.append(class_id)
            bucket = by_class.setdefault(
                class_id,
                {
                    "display_id": class_id + 1,
                    "class_name": str(row["class_name"]),
                    "train_accuracy": [],
                    "validation_accuracy": [],
                    "test_accuracy": [],
                    "test_cosine_margin": [],
                },
            )
            if bucket["class_name"] != str(row["class_name"]):
                raise RuntimeError(
                    f"class name changed across seeds for class {class_id}"
                )

            for key in (
                "train_accuracy",
                "validation_accuracy",
                "test_accuracy",
                "test_cosine_margin",
            ):
                parsed = _optional_csv_float(
                    row.get(key),
                    name=f"{key} for seed {seed}, class {class_id}",
                )
                if parsed is not None:
                    bucket[key].append(parsed)

        if len(class_ids) != len(set(class_ids)):
            raise RuntimeError(f"classwise report for seed {seed} has duplicate classes")
        if expected_ids is None:
            expected_ids = class_ids
        elif class_ids != expected_ids:
            raise RuntimeError(
                "classwise class order changed across independent seed runs"
            )

    outputs: list[Dict[str, Any]] = []
    for class_id in expected_ids or []:
        bucket = by_class[class_id]
        row: Dict[str, Any] = {
            "class_id": class_id,
            "display_id": bucket["display_id"],
            "class_name": bucket["class_name"],
        }
        for key in (
            "train_accuracy",
            "validation_accuracy",
            "test_accuracy",
            "test_cosine_margin",
        ):
            values = bucket[key]
            if values:
                stats = _sample_statistics(values)
                row[f"{key}_mean"] = stats["mean"]
                row[f"{key}_std"] = stats["std"]
                row[f"{key}_min"] = stats["min"]
                row[f"{key}_max"] = stats["max"]
            else:
                row[f"{key}_mean"] = None
                row[f"{key}_std"] = None
                row[f"{key}_min"] = None
                row[f"{key}_max"] = None
        outputs.append(row)
    return outputs


def run_multiseed_base(args: argparse.Namespace) -> Dict[str, Any]:
    if args.seeds is None:
        raise ValueError("run_multiseed_base requires --seeds")
    if str(args.run_mode) != "base":
        raise ValueError("multi-seed validation currently requires --run_mode base")

    seeds = [
        _exact_int(
            value,
            name="seeds value",
            minimum=0,
            maximum=(2**32) - 1,
        )
        for value in args.seeds
    ]
    if len(seeds) != len(set(seeds)):
        raise ValueError("--seeds must contain unique values")

    root = os.path.abspath(args.save_dir)
    os.makedirs(root, exist_ok=True)

    metric_names = (
        "oa",
        "ba",
        "min_class_accuracy",
        "kappa",
        "cosine_margin",
        "feature_rms",
    )
    per_seed_phase_reports: list[tuple[int, Dict[str, Any]]] = []
    per_seed_classwise: list[tuple[int, Sequence[Mapping[str, Any]]]] = []
    run_rows: list[Dict[str, Any]] = []

    print("\nIndependent base-phase multi-seed validation")
    print(f"Seeds    : {seeds}")
    print(f"Root dir : {root}")

    for index, seed in enumerate(seeds, start=1):
        seed_args = copy.deepcopy(args)
        seed_args.seed = int(seed)
        seed_args.seeds = None
        seed_args.run_mode = "base"
        seed_args.resume_checkpoint = ""
        seed_args.save_dir = os.path.join(root, f"seed_{seed}")

        # Every seed is a complete independent experiment: new protocol/split,
        # preprocessing fit, model, classifier, class statistics, trainer, and optimizer.
        set_seed(seed_args.seed, bool(seed_args.deterministic))

        print(
            f"\n{'=' * 78}\n"
            f"Base seed {seed} ({index}/{len(seeds)})\n"
            f"Output: {seed_args.save_dir}\n"
            f"{'=' * 78}"
        )
        run_experiment(seed_args)

        phase_report = _load_json_mapping(
            os.path.join(seed_args.save_dir, "phase_0", "phase_report.json")
        )
        per_seed_phase_reports.append((seed, phase_report))

        classwise_rows = _load_csv_rows(
            os.path.join(seed_args.save_dir, "phase_0", "classwise_metrics.csv")
        )
        per_seed_classwise.append((seed, classwise_rows))

        row: Dict[str, Any] = {
            "seed": seed,
            "output_dir": seed_args.save_dir,
            "mean_identity_decision_margin": float(
                phase_report["mean_identity_decision_margin"]
            ),
        }
        for split in ("train", "validation", "test"):
            metrics = phase_report.get(split)
            if metrics is None:
                for metric in metric_names:
                    row[f"{split}_{metric}"] = None
                continue
            if not isinstance(metrics, Mapping):
                raise RuntimeError(
                    f"phase report {split} metrics must be a mapping"
                )
            for metric in metric_names:
                row[f"{split}_{metric}"] = float(metrics[metric])
        run_rows.append(row)

    aggregate: Dict[str, Any] = {}
    summary_rows: list[Dict[str, Any]] = []

    for split in ("train", "validation", "test"):
        split_statistics: Dict[str, Any] = {}
        for metric in metric_names:
            values = [
                float(report[split][metric])
                for _, report in per_seed_phase_reports
                if isinstance(report.get(split), Mapping)
            ]
            if not values:
                continue
            stats = _sample_statistics(values)
            split_statistics[metric] = stats
            summary_rows.append(
                {
                    "scope": split,
                    "metric": metric,
                    "mean": stats["mean"],
                    "std": stats["std"],
                    "min": stats["min"],
                    "max": stats["max"],
                    "seed_count": len(values),
                }
            )
        aggregate[split] = split_statistics

    margin_stats = _sample_statistics(
        [
            float(report["mean_identity_decision_margin"])
            for _, report in per_seed_phase_reports
        ]
    )
    aggregate["mean_identity_decision_margin"] = margin_stats
    summary_rows.append(
        {
            "scope": "decision_state",
            "metric": "mean_identity_decision_margin",
            "mean": margin_stats["mean"],
            "std": margin_stats["std"],
            "min": margin_stats["min"],
            "max": margin_stats["max"],
            "seed_count": len(seeds),
        }
    )

    classwise_aggregate = _aggregate_multiseed_classwise(
        per_seed_rows=per_seed_classwise
    )

    runs_path = save_csv(
        os.path.join(root, "multiseed_runs.csv"),
        run_rows,
    )
    summary_csv_path = save_csv(
        os.path.join(root, "multiseed_summary.csv"),
        summary_rows,
    )
    classwise_path = save_csv(
        os.path.join(root, "multiseed_classwise.csv"),
        classwise_aggregate,
    )

    fixed_protocol = args.protocol_seed is not None
    summary = {
        "mode": "independent_base_multiseed",
        "dataset": str(args.dataset),
        "seeds": seeds,
        "seed_count": len(seeds),
        "protocol_seed": (
            None if args.protocol_seed is None else int(args.protocol_seed)
        ),
        "robustness_scope": (
            "optimization_seed_on_fixed_protocol"
            if fixed_protocol
            else "full_pipeline_seed_including_protocol_split"
        ),
        "standard_deviation": "sample standard deviation (ddof=1); 0 for one seed",
        "independence_contract": {
            "fresh_protocol_per_seed": not fixed_protocol,
            "fresh_train_val_test_split_per_seed": not fixed_protocol,
            "fresh_preprocessing_fit_per_seed": True,
            "same_preprocessing_membership_across_seeds": fixed_protocol,
            "fresh_model_per_seed": True,
            "fresh_classifier_per_seed": True,
            "fresh_class_statistics_per_seed": True,
            "fresh_trainer_optimizer_per_seed": True,
            "state_reused_between_seeds": False,
        },
        "aggregate": aggregate,
        "per_seed_output_dirs": {
            str(seed): os.path.join(root, f"seed_{seed}")
            for seed in seeds
        },
        "artifacts": {
            "per_seed_runs_csv": runs_path,
            "aggregate_summary_csv": summary_csv_path,
            "aggregate_classwise_csv": classwise_path,
        },
    }
    summary_json_path = save_json(
        os.path.join(root, "multiseed_summary.json"),
        summary,
    )
    summary["artifacts"]["aggregate_summary_json"] = summary_json_path

    print("\nMulti-seed base validation complete.")
    test_ba = aggregate.get("test", {}).get("ba")
    test_oa = aggregate.get("test", {}).get("oa")
    test_min = aggregate.get("test", {}).get("min_class_accuracy")
    if isinstance(test_oa, Mapping):
        print(
            "Test OA       : "
            f"{100.0 * float(test_oa['mean']):.2f} ± "
            f"{100.0 * float(test_oa['std']):.2f}%"
        )
    if isinstance(test_ba, Mapping):
        print(
            "Test BA       : "
            f"{100.0 * float(test_ba['mean']):.2f} ± "
            f"{100.0 * float(test_ba['std']):.2f}%"
        )
    if isinstance(test_min, Mapping):
        print(
            "Test MinClass : "
            f"{100.0 * float(test_min['mean']):.2f} ± "
            f"{100.0 * float(test_min['std']):.2f}%"
        )
    print(f"Summary JSON  : {summary_json_path}")
    print(f"Summary CSV   : {summary_csv_path}")
    print(f"Classwise CSV : {classwise_path}")
    return summary

def main(argv: Optional[Sequence[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    validate_args(args)

    seeds_for_initialization = (
        [int(value) for value in args.seeds]
        if args.seeds is not None
        else [int(args.seed)]
    )

    # Set deterministic CUDA environment/RNG state before resolving the CUDA device.
    # Each independent seed run is reseeded again inside run_multiseed_base().
    set_seed(seeds_for_initialization[0], bool(args.deterministic))
    args.device = resolve_device(args.device)
    args.save_dir = os.path.abspath(args.save_dir)

    if args.seeds is not None:
        run_multiseed_base(args)
    else:
        run_experiment(args)


if __name__ == "__main__":
    main()









# from __future__ import annotations

# """SCOPE-HSI experiment driver.

# Locked architectural contract
# -----------------------------
# The complete runtime follows one representation and one decision path:

#     HSI -> Mamba backbone -> Q_raw -> global-RMS canonical Q -> classifier W
#                                              |
#                                              +-> {I_c,S_c,N_c}

# Q is the only learned sample representation.  {I_c,S_c,N_c} are historical
# class-discriminative statistics estimated from Q; W is the one persistent
# all-seen classifier acting on the same Q.

# At an incremental phase, the trainer is responsible for observing one common
# coordinate movement R from paired current-class Q before/after adaptation and
# for applying that same movement to the complete historical decision-relevant
# state:

#     I <- R I,
#     S[:,:,d] <- R S[:,:,d] R^T,
#     W <- R W.

# This file does not implement preservation/transport mathematics.  It validates
# experiment arguments, constructs the HSI protocol, dispatches the trainer, and
# writes reporting-only artifacts.  Old gradient-projection preservation is not
# part of this runtime.
# """

# import argparse
# import csv
# import json
# import math
# import os
# import random
# from numbers import Integral, Real
# from typing import Any, Dict, Mapping, Optional, Sequence

# import numpy as np
# import torch

# from data.hsi_dataloader_pytorch import (
#     ExtractLabeledPixelIndex,
#     FitHSIPreprocessorFromProtocol,
#     LoadRawHSIData,
#     SaveHSIPreprocessor,
# )
# from data.incremental_dataset import (
#     IncrementalHSIDatasetManager,
#     build_incremental_protocol,
# )
# from models.necil_model import NECILModel
# from trainers.trainer import Trainer
# from utils.eval import (
#     class_structure_diagnostics,
#     collect_reporting_features,
#     decision_state_diagnostics,
#     transport_drift_diagnostics,
# )
# from utils.qualitative_report import generate_phase_qualitative_maps
# from utils.visualize import (
#     save_geometry_diagnostic_figures,
#     save_incremental_tsne_comparison,
# )


# # -----------------------------------------------------------------------------
# # Generic IO / reproducibility
# # -----------------------------------------------------------------------------


# def str2bool(value: Any) -> bool:
#     if isinstance(value, bool):
#         return value
#     token = str(value).strip().lower()
#     if token in {"1", "true", "yes", "y", "on"}:
#         return True
#     if token in {"0", "false", "no", "n", "off"}:
#         return False
#     raise argparse.ArgumentTypeError(f"invalid boolean value {value!r}")


# def _parse_class_order(value: str) -> Optional[list[int]]:
#     token = str(value or "").strip()
#     if not token:
#         return None

#     parts = token.split(",")
#     if any(not part.strip() for part in parts):
#         raise ValueError(
#             "class_order must be a comma-separated list without empty entries"
#         )
#     try:
#         values = [int(part.strip()) for part in parts]
#     except ValueError as exc:
#         raise ValueError(
#             "class_order must be comma-separated integer IDs"
#         ) from exc

#     if (
#         not values
#         or len(values) != len(set(values))
#         or any(value < 0 for value in values)
#     ):
#         raise ValueError(
#             "class_order must contain unique non-negative IDs"
#         )
#     return values


# def _exact_int(
#     value: object,
#     *,
#     name: str,
#     minimum: Optional[int] = None,
#     maximum: Optional[int] = None,
# ) -> int:
#     if torch.is_tensor(value):
#         if value.numel() != 1:
#             raise ValueError(f"{name} must be an integer scalar")
#         value = value.item()

#     if isinstance(value, (bool, np.bool_)):
#         raise ValueError(f"{name} must be an integer")

#     if isinstance(value, Integral):
#         result = int(value)
#     elif isinstance(value, Real):
#         number = float(value)
#         if not math.isfinite(number) or not number.is_integer():
#             raise ValueError(f"{name} must be a finite integer")
#         result = int(number)
#     else:
#         raise ValueError(f"{name} must be an integer")

#     if minimum is not None and result < minimum:
#         raise ValueError(f"{name} must be >= {minimum}")
#     if maximum is not None and result > maximum:
#         raise ValueError(f"{name} must be <= {maximum}")
#     return result


# def _finite_float(
#     value: object,
#     *,
#     name: str,
#     minimum: Optional[float] = None,
#     strict_minimum: bool = False,
# ) -> float:
#     result = float(value)
#     if not math.isfinite(result):
#         raise ValueError(f"{name} must be finite")
#     if minimum is not None:
#         if strict_minimum and not result > minimum:
#             raise ValueError(f"{name} must be > {minimum}")
#         if not strict_minimum and result < minimum:
#             raise ValueError(f"{name} must be >= {minimum}")
#     return result


# def _json_safe(value: Any) -> Any:
#     if torch.is_tensor(value):
#         tensor = value.detach().cpu()
#         return tensor.item() if tensor.numel() == 1 else tensor.tolist()
#     if isinstance(value, np.ndarray):
#         return value.tolist()
#     if isinstance(value, (np.integer, np.floating, np.bool_)):
#         return value.item()
#     if isinstance(value, Mapping):
#         return {str(key): _json_safe(item) for key, item in value.items()}
#     if isinstance(value, (list, tuple, set)):
#         return [_json_safe(item) for item in value]
#     if isinstance(value, float) and not math.isfinite(value):
#         return str(value)
#     if isinstance(value, (str, int, float, bool)) or value is None:
#         return value
#     return str(value)


# def save_json(path: str, value: Mapping[str, Any]) -> str:
#     destination = os.path.abspath(path)
#     os.makedirs(os.path.dirname(destination) or ".", exist_ok=True)
#     temporary = destination + ".tmp"
#     try:
#         with open(temporary, "w", encoding="utf-8") as stream:
#             json.dump(_json_safe(value), stream, indent=2, sort_keys=True)
#             stream.flush()
#             os.fsync(stream.fileno())
#         os.replace(temporary, destination)
#     except Exception:
#         if os.path.exists(temporary):
#             os.remove(temporary)
#         raise
#     return destination


# def save_csv(
#     path: str,
#     rows: Sequence[Mapping[str, Any]],
# ) -> Optional[str]:
#     values = list(rows)
#     if not values:
#         return None

#     destination = os.path.abspath(path)
#     os.makedirs(os.path.dirname(destination) or ".", exist_ok=True)

#     fieldnames: list[str] = []
#     seen: set[str] = set()
#     for row in values:
#         for name in row:
#             if name not in seen:
#                 fieldnames.append(name)
#                 seen.add(name)

#     temporary = destination + ".tmp"
#     try:
#         with open(
#             temporary,
#             "w",
#             newline="",
#             encoding="utf-8-sig",
#         ) as stream:
#             writer = csv.DictWriter(stream, fieldnames=fieldnames)
#             writer.writeheader()
#             for row in values:
#                 writer.writerow(
#                     {name: _json_safe(row.get(name)) for name in fieldnames}
#                 )
#             stream.flush()
#             os.fsync(stream.fileno())
#         os.replace(temporary, destination)
#     except Exception:
#         if os.path.exists(temporary):
#             os.remove(temporary)
#         raise
#     return destination


# def set_seed(seed: int, deterministic: bool) -> None:
#     value = _exact_int(
#         seed,
#         name="seed",
#         minimum=0,
#         maximum=(2**32) - 1,
#     )
#     os.environ["PYTHONHASHSEED"] = str(value)

#     if deterministic:
#         os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

#     random.seed(value)
#     np.random.seed(value)
#     torch.manual_seed(value)
#     if torch.cuda.is_available():
#         torch.cuda.manual_seed_all(value)

#     torch.backends.cudnn.benchmark = not deterministic
#     torch.backends.cudnn.deterministic = deterministic

#     if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
#         torch.backends.cuda.matmul.allow_tf32 = not deterministic
#     if hasattr(torch.backends, "cudnn") and hasattr(torch.backends.cudnn, "allow_tf32"):
#         torch.backends.cudnn.allow_tf32 = not deterministic

#     torch.use_deterministic_algorithms(bool(deterministic), warn_only=False)


# def resolve_device(value: str) -> str:
#     token = str(value).strip().lower().replace("gpu", "cuda")
#     try:
#         requested = torch.device(token)
#     except (TypeError, RuntimeError) as exc:
#         raise ValueError(f"invalid device {value!r}") from exc

#     if requested.type == "cpu":
#         return "cpu"
#     if requested.type != "cuda" or not torch.cuda.is_available():
#         raise RuntimeError(f"requested device {value!r} is unavailable")

#     index = torch.cuda.current_device() if requested.index is None else requested.index
#     if index is None or not 0 <= int(index) < torch.cuda.device_count():
#         raise RuntimeError(f"CUDA device index {index} is unavailable")

#     torch.cuda.set_device(int(index))
#     return f"cuda:{int(index)}"


# def _load_band_positions(
#     path: str,
#     expected_bands: int,
# ) -> Optional[list[float]]:
#     token = str(path or "").strip()
#     if not token:
#         return None
#     if not os.path.isfile(token):
#         raise FileNotFoundError(token)

#     values = np.load(token) if token.lower().endswith(".npy") else np.loadtxt(token)
#     positions = np.asarray(values, dtype=np.float32).reshape(-1)
#     if positions.size != int(expected_bands):
#         raise ValueError(
#             f"band position file has {positions.size} values; expected {expected_bands}"
#         )
#     if not np.isfinite(positions).all() or np.any(np.diff(positions) <= 0):
#         raise ValueError("band positions must be finite and strictly increasing")
#     return positions.tolist()


# # -----------------------------------------------------------------------------
# # CLI
# # -----------------------------------------------------------------------------


# def build_parser() -> argparse.ArgumentParser:
#     parser = argparse.ArgumentParser(
#         description=(
#             "SCOPE-HSI: one canonical HSI representation Q, one persistent "
#             "angular classifier, Q-derived class structure, and one shared "
#             "coordinate transport during incremental learning."
#         ),
#         formatter_class=argparse.ArgumentDefaultsHelpFormatter,
#     )

#     experiment = parser.add_argument_group("Experiment")
#     experiment.add_argument("--dataset", default="IP")
#     experiment.add_argument("--data_dir", default="./datasets")
#     experiment.add_argument("--save_dir", default="./results_scope_hsi")
#     experiment.add_argument("--seed", type=int, default=42)
#     experiment.add_argument("--device", default="cuda:0")
#     experiment.add_argument("--num_workers", type=int, default=0)
#     experiment.add_argument("--deterministic", type=str2bool, default=True)
#     experiment.add_argument(
#         "--run_mode",
#         choices=("base", "incremental", "all"),
#         default="base",
#         help=(
#             "base: train/finalize phase 0 only; incremental: restore a finalized "
#             "checkpoint and continue; all: train phase 0 then all increments."
#         ),
#     )
#     experiment.add_argument(
#         "--resume_checkpoint",
#         default="",
#         help="Required for --run_mode incremental.",
#     )

#     protocol = parser.add_argument_group("Class-incremental protocol")
#     protocol.add_argument("--base_classes", type=int, default=6)
#     protocol.add_argument("--increment", type=int, default=6)
#     protocol.add_argument(
#         "--class_order",
#         default="",
#         help="Complete zero-based class permutation.",
#     )
#     protocol.add_argument("--shuffle_order", type=str2bool, default=False)
#     protocol.add_argument("--train_ratio", type=float, default=0.20)
#     protocol.add_argument(
#         "--val_ratio",
#         type=float,
#         default=0.10,
#         help=(
#             "Reporting-only validation fraction; it never selects/restores a "
#             "checkpoint or stops training."
#         ),
#     )
#     protocol.add_argument(
#         "--split_strategy",
#         choices=("random_pixel", "spatial_block"),
#         default="random_pixel",
#     )
#     protocol.add_argument("--spatial_block_size", type=int, default=33)
#     protocol.add_argument(
#         "--require_patch_disjoint",
#         type=str2bool,
#         default=False,
#     )
#     protocol.add_argument(
#         "--require_phase_context_isolation",
#         type=str2bool,
#         default=None,
#     )

#     preprocessing = parser.add_argument_group("HSI preprocessing")
#     preprocessing.add_argument("--patch_size", type=int, default=11)
#     preprocessing.add_argument("--no_pca", action="store_true")
#     preprocessing.add_argument("--pca_components", type=int, default=30)
#     preprocessing.add_argument("--pca_whiten", type=str2bool, default=False)
#     preprocessing.add_argument("--band_positions_file", default="")

#     backbone = parser.add_argument_group("Ordered spectral-spatial HSI backbone")
#     backbone.add_argument("--d_model", type=int, default=128)
#     backbone.add_argument(
#         "--spectral_token_count",
#         type=int,
#         default=16,
#         help=(
#             "K ordered contiguous spectral-region features retained in canonical "
#             "Q; class statistics are I_c[K,D] and S_c[K,K,D]."
#         ),
#     )
#     backbone.add_argument("--d_state", type=int, default=16)
#     backbone.add_argument("--d_conv", type=int, default=4)
#     backbone.add_argument("--expand", type=int, default=2)
#     backbone.add_argument("--num_spectral_layers", type=int, default=2)
#     backbone.add_argument("--num_spatial_layers", type=int, default=2)
#     backbone.add_argument("--stem_norm_groups", type=int, default=8)
#     backbone.add_argument("--mamba_dropout", type=float, default=0.0)

#     base = parser.add_argument_group("Phase-0 optimization")
#     base.add_argument("--epochs_base", type=int, default=100)
#     base.add_argument("--batch_size", type=int, default=64)
#     base.add_argument("--eval_batch_size", type=int, default=256)
#     base.add_argument("--lr", type=float, default=1e-4)
#     base.add_argument("--weight_decay", type=float, default=1e-4)
#     base.add_argument("--gradient_clip", type=float, default=5.0)
#     base.add_argument(
#         "--spatial_d4_augmentation",
#         type=str2bool,
#         default=False,
#         help=(
#             "Optional phase-0 spatial D4 patch augmentation. It changes only H/W, "
#             "never spectral/channel order, and is not part of the core SCOPE method."
#         ),
#     )

#     incremental = parser.add_argument_group("Incremental optimization")
#     incremental.add_argument("--epochs_inc", type=int, default=15)
#     incremental.add_argument("--lr_inc", type=float, default=1e-4)

#     reporting = parser.add_argument_group("Reporting")
#     reporting.add_argument("--save_visualizations", type=str2bool, default=True)
#     reporting.add_argument("--visualization_cmap", default="nipy_spectral")
#     reporting.add_argument("--visualization_dpi", type=int, default=300)

#     return parser


# def validate_args(args: argparse.Namespace) -> None:
#     positive_integer_fields = (
#         "base_classes",
#         "increment",
#         "patch_size",
#         "d_model",
#         "spectral_token_count",
#         "d_state",
#         "d_conv",
#         "expand",
#         "num_spectral_layers",
#         "num_spatial_layers",
#         "stem_norm_groups",
#         "epochs_base",
#         "epochs_inc",
#         "batch_size",
#         "eval_batch_size",
#         "spatial_block_size",
#         "visualization_dpi",
#     )
#     for name in positive_integer_fields:
#         _exact_int(getattr(args, name), name=name, minimum=1)

#     _exact_int(args.num_workers, name="num_workers", minimum=0)
#     _exact_int(args.seed, name="seed", minimum=0, maximum=(2**32) - 1)

#     if not bool(args.no_pca):
#         _exact_int(args.pca_components, name="pca_components", minimum=1)

#     if int(args.base_classes) < 2:
#         raise ValueError("base_classes must be at least 2")
#     if int(args.spectral_token_count) < 2:
#         raise ValueError("spectral_token_count must be at least 2")

#     if int(args.patch_size) % 2 == 0:
#         raise ValueError("patch_size must be odd")

#     groups = int(args.stem_norm_groups)
#     width = int(args.d_model)
#     if groups > width or width % groups != 0:
#         raise ValueError("stem_norm_groups must divide d_model")

#     train_ratio = _finite_float(
#         args.train_ratio,
#         name="train_ratio",
#         minimum=0.0,
#         strict_minimum=True,
#     )
#     val_ratio = _finite_float(args.val_ratio, name="val_ratio", minimum=0.0)
#     if train_ratio + val_ratio >= 1.0:
#         raise ValueError("train_ratio + val_ratio must be < 1")

#     dropout = _finite_float(args.mamba_dropout, name="mamba_dropout", minimum=0.0)
#     if dropout >= 1.0:
#         raise ValueError("mamba_dropout must be < 1")

#     _finite_float(args.lr, name="lr", minimum=0.0, strict_minimum=True)
#     _finite_float(args.lr_inc, name="lr_inc", minimum=0.0, strict_minimum=True)
#     _finite_float(args.weight_decay, name="weight_decay", minimum=0.0)
#     _finite_float(args.gradient_clip, name="gradient_clip", minimum=0.0)

#     if str(args.run_mode) == "incremental" and not str(args.resume_checkpoint).strip():
#         raise ValueError("--resume_checkpoint is required for --run_mode incremental")

#     class_order = _parse_class_order(args.class_order)
#     if class_order is not None and bool(args.shuffle_order):
#         raise ValueError("class_order and shuffle_order are mutually exclusive")

#     if args.require_phase_context_isolation is not None and not isinstance(
#         args.require_phase_context_isolation,
#         bool,
#     ):
#         raise ValueError("require_phase_context_isolation must be bool or None")
#     if not isinstance(args.spatial_d4_augmentation, bool):
#         raise ValueError("spatial_d4_augmentation must be bool")


# # -----------------------------------------------------------------------------
# # Dataset / protocol construction
# # -----------------------------------------------------------------------------


# def _build_data(
#     args: argparse.Namespace,
# ) -> tuple[IncrementalHSIDatasetManager, Dict[str, Any], Mapping[str, Any]]:
#     raw_cube, gt, class_count, names, _, label_policy = LoadRawHSIData(
#         args.dataset,
#         base_dir=args.data_dir,
#     )
#     labels, coords = ExtractLabeledPixelIndex(gt, label_policy=label_policy)

#     if int(args.base_classes) > int(class_count):
#         raise ValueError(
#             f"base_classes={args.base_classes} exceeds dataset classes={class_count}"
#         )

#     class_order = _parse_class_order(args.class_order)
#     if class_order is not None and set(class_order) != set(range(int(class_count))):
#         raise ValueError(
#             "class_order must be a permutation of "
#             f"0..{int(class_count) - 1}"
#         )

#     protocol = build_incremental_protocol(
#         labels,
#         coords,
#         gt_shape=gt.shape,
#         class_count=int(class_count),
#         base_classes=int(args.base_classes),
#         increment=int(args.increment),
#         train_ratio=float(args.train_ratio),
#         val_ratio=float(args.val_ratio),
#         seed=int(args.seed),
#         shuffle_order=bool(args.shuffle_order),
#         class_order=class_order,
#         target_names=names,
#         split_strategy=str(args.split_strategy),
#         spatial_block_size=int(args.spatial_block_size),
#     )

#     required_protocol_fields = {
#         "phase_to_classes",
#         "phase_train_indices",
#         "base_train_indices",
#         "validation_enabled",
#         "split_by_class",
#     }
#     missing_protocol_fields = required_protocol_fields - set(protocol)
#     if missing_protocol_fields:
#         raise RuntimeError(
#             "incremental protocol is incompatible with the current SCOPE-HSI "
#             f"runtime; missing fields={sorted(missing_protocol_fields)}"
#         )
#     if not isinstance(protocol["validation_enabled"], bool):
#         raise RuntimeError("protocol validation_enabled must be bool")

#     processed_cube, ordered_cube, preprocessing_state = FitHSIPreprocessorFromProtocol(
#         raw_cube,
#         coords=coords,
#         protocol=protocol,
#         apply_reduction=not bool(args.no_pca),
#         reduction_method="PCA",
#         n_components=int(args.pca_components),
#         whiten=bool(args.pca_whiten),
#     )

#     args.num_bands = int(processed_cube.shape[2])
#     args.spectral_bands = int(ordered_cube.shape[2])
#     args.raw_num_bands = int(ordered_cube.shape[2])

#     if int(args.spectral_token_count) > int(args.spectral_bands):
#         raise ValueError(
#             "spectral_token_count cannot exceed the number of ordered raw bands "
#             f"({args.spectral_bands})"
#         )

#     args.band_positions = _load_band_positions(
#         args.band_positions_file,
#         args.spectral_bands,
#     )
#     args.band_wavelengths = None

#     # The spectral path uses the same phase-0 TRAIN-only raw-band normalization
#     # fitted by the preprocessor; there is no independent second fitting scope.
#     args.spectral_normalization_mean = np.asarray(
#         preprocessing_state["normalization_mean"],
#         dtype=np.float32,
#     ).tolist()
#     args.spectral_normalization_std = np.asarray(
#         preprocessing_state["normalization_std"],
#         dtype=np.float32,
#     ).tolist()

#     manager = IncrementalHSIDatasetManager(
#         processed_cube=processed_cube,
#         ordered_spectral_cube=ordered_cube,
#         labels=labels,
#         coords=coords,
#         protocol=protocol,
#         target_names=names,
#         gt_shape=gt.shape,
#         patch_size=int(args.patch_size),
#         num_workers=int(args.num_workers),
#         device=str(args.device),
#         seed=int(args.seed),
#         require_patch_disjoint=bool(args.require_patch_disjoint),
#         require_phase_context_isolation=args.require_phase_context_isolation,
#         context_policy="full_scene_reflect",
#     )

#     if not hasattr(manager, "validation_enabled") or not isinstance(
#         manager.validation_enabled,
#         bool,
#     ):
#         raise RuntimeError(
#             "IncrementalHSIDatasetManager must expose boolean validation_enabled"
#         )
#     if manager.validation_enabled != protocol["validation_enabled"]:
#         raise RuntimeError("manager validation capability disagrees with protocol")
#     if not hasattr(manager, "base_train_indices"):
#         raise RuntimeError(
#             "IncrementalHSIDatasetManager is missing canonical base_train_indices"
#         )

#     expected_phase_sizes = [int(args.base_classes)]
#     remaining = int(class_count) - int(args.base_classes)
#     while remaining > 0:
#         current = min(int(args.increment), remaining)
#         expected_phase_sizes.append(current)
#         remaining -= current

#     actual_phase_sizes = [
#         len(manager.phase_to_classes[phase])
#         for phase in sorted(manager.phase_to_classes)
#     ]
#     if actual_phase_sizes != expected_phase_sizes:
#         raise RuntimeError(
#             "constructed class-incremental schedule is inconsistent: "
#             f"expected={expected_phase_sizes}, actual={actual_phase_sizes}"
#         )

#     split_counts = {
#         int(class_id): {
#             split: int(np.asarray(protocol["split_by_class"][class_id][split]).size)
#             for split in ("train", "val", "test")
#         }
#         for class_id in range(int(class_count))
#     }

#     data_summary: Dict[str, Any] = {
#         "dataset": str(args.dataset),
#         "raw_shape": list(raw_cube.shape),
#         "gt_shape": list(gt.shape),
#         "labeled_sample_count": int(labels.size),
#         "class_count": int(class_count),
#         "phase_sizes": list(actual_phase_sizes),
#         "phase_schedule": {
#             int(phase): [int(value) for value in classes]
#             for phase, classes in protocol["phase_to_classes"].items()
#         },
#         "class_order_original_ids": list(manager.class_order_original_ids),
#         "class_names_global_order": list(manager.target_names),
#         "processed_bands": int(processed_cube.shape[2]),
#         "raw_spectral_bands": int(ordered_cube.shape[2]),
#         "spectral_region_count": int(args.spectral_token_count),
#         "sample_representation": "Q[K,D] = Q_raw / RMS(Q_raw)",
#         "persistent_class_statistics": "{I_c[K,D],S_c[K,K,D],N_c}",
#         "persistent_classifier": "one angular W_c[K,D] row per seen class",
#         "incremental_consistency_rule": "one R: I->RI, S->RSR^T, W->RW",
#         "preprocessing_fit_scope": str(preprocessing_state["fit_scope"]),
#         "preprocessing_fit_pixel_count": int(preprocessing_state["fit_pixel_count"]),
#         "preprocessing_frozen_across_phases": True,
#         "validation_enabled": bool(manager.validation_enabled),
#         "split_strategy": manager.split_strategy,
#         "spatial_partition_mode": manager.spatial_partition_mode,
#         "require_patch_disjoint": manager.require_patch_disjoint,
#         "require_phase_context_isolation": manager.require_phase_context_isolation,
#         "context_policy": manager.context_policy,
#         "spatial_leakage_audit": manager.spatial_leakage_report(),
#         "split_counts_by_global_class": split_counts,
#         "current_phase_training_classes_only": True,
#         "cumulative_seen_class_evaluation": True,
#         "stores_old_training_samples": False,
#         "stores_old_features": False,
#         "replay": False,
#         "teacher": False,
#         "knowledge_distillation": False,
#     }
#     return manager, data_summary, preprocessing_state


# # -----------------------------------------------------------------------------
# # Shared reporting helpers
# # -----------------------------------------------------------------------------


# def _mapping_value(mapping: Any, class_id: int) -> Any:
#     if not isinstance(mapping, Mapping):
#         return None
#     if class_id in mapping:
#         return mapping[class_id]
#     return mapping.get(str(class_id))


# def _metric_summary(metrics: Optional[Mapping[str, Any]]) -> Optional[Dict[str, Any]]:
#     if metrics is None:
#         return None
#     return {
#         "oa": float(metrics["overall_accuracy"]),
#         "ba": float(metrics["balanced_accuracy"]),
#         "min_class_accuracy": float(metrics["minimum_class_accuracy"]),
#         "kappa": float(metrics["kappa"]),
#         "cosine_margin": float(metrics["mean_cosine_margin"]),
#         "feature_rms": float(metrics["mean_feature_rms"]),
#     }


# def _history_rows(history: Sequence[Mapping[str, Any]]) -> list[Dict[str, Any]]:
#     rows: list[Dict[str, Any]] = []
#     for record in history:
#         rows.append(
#             {
#                 "epoch": int(record["epoch"]),
#                 "learning_rate": float(record["learning_rate"]),
#                 "classification_loss": float(record["classification_loss"]),
#                 "train_oa": float(record["train_oa"]),
#                 "train_ba": float(record["train_ba"]),
#                 "validation_oa": record.get("validation_oa"),
#                 "validation_ba": record.get("validation_ba"),
#             }
#         )
#     return rows


# def _classwise_rows(
#     *,
#     result: Mapping[str, Any],
#     class_ids: Sequence[int],
#     class_names: Sequence[str],
# ) -> list[Dict[str, Any]]:
#     train = result["train"]
#     validation = result.get("validation")
#     test = result["test"]

#     rows: list[Dict[str, Any]] = []
#     for class_id in [int(v) for v in class_ids]:
#         rows.append(
#             {
#                 "class_id": class_id,
#                 "display_id": class_id + 1,
#                 "class_name": str(class_names[class_id]),
#                 "train_samples": _mapping_value(train.get("class_counts"), class_id),
#                 "train_accuracy": _mapping_value(train.get("per_class_accuracy"), class_id),
#                 "validation_samples": (
#                     None
#                     if not isinstance(validation, Mapping)
#                     else _mapping_value(validation.get("class_counts"), class_id)
#                 ),
#                 "validation_accuracy": (
#                     None
#                     if not isinstance(validation, Mapping)
#                     else _mapping_value(validation.get("per_class_accuracy"), class_id)
#                 ),
#                 "test_samples": _mapping_value(test.get("class_counts"), class_id),
#                 "test_accuracy": _mapping_value(test.get("per_class_accuracy"), class_id),
#                 "test_cosine_margin": _mapping_value(
#                     test.get("per_class_mean_cosine_margin"),
#                     class_id,
#                 ),
#             }
#         )
#     return rows


# def _confusion_rows(
#     metrics: Mapping[str, Any],
#     class_ids: Sequence[int],
# ) -> list[Dict[str, Any]]:
#     matrix = np.asarray(metrics["confusion_matrix"], dtype=np.int64)
#     ids = [int(v) for v in class_ids]
#     if matrix.shape != (len(ids), len(ids)):
#         raise RuntimeError("confusion matrix shape disagrees with class_ids")

#     rows: list[Dict[str, Any]] = []
#     for row_index, class_id in enumerate(ids):
#         row: Dict[str, Any] = {
#             "true_class_id": class_id,
#             "true_display_id": class_id + 1,
#         }
#         for column_index, predicted_id in enumerate(ids):
#             row[f"pred_{predicted_id}"] = int(matrix[row_index, column_index])
#         rows.append(row)
#     return rows


# def _print_metrics(name: str, metrics: Optional[Mapping[str, Any]]) -> None:
#     if metrics is None:
#         print(f"{name:<10} | disabled")
#         return
#     print(
#         f"{name:<10} | "
#         f"OA={100.0 * float(metrics['overall_accuracy']):.2f}% | "
#         f"BA={100.0 * float(metrics['balanced_accuracy']):.2f}% | "
#         f"MinClass={100.0 * float(metrics['minimum_class_accuracy']):.2f}% | "
#         f"Kappa={float(metrics['kappa']):.4f} | "
#         f"CosGap={float(metrics['mean_cosine_margin']):.4f} | "
#         f"QRMS={float(metrics['mean_feature_rms']):.4f}"
#     )


# # -----------------------------------------------------------------------------
# # Phase-0 reporting
# # -----------------------------------------------------------------------------


# def _report_base_phase(
#     *,
#     args: argparse.Namespace,
#     dataset: IncrementalHSIDatasetManager,
#     model: NECILModel,
#     result: Mapping[str, Any],
# ) -> Dict[str, Any]:
#     class_ids = [int(value) for value in result["class_ids"]]
#     model.validate_finalized_state()

#     phase_dir = os.path.join(args.save_dir, "phase_0")
#     os.makedirs(phase_dir, exist_ok=True)

#     class_structure = class_structure_diagnostics(
#         model,
#         class_ids,
#         target_names=dataset.target_names,
#     )
#     decision_state = decision_state_diagnostics(model, class_ids)

#     history_path = save_csv(
#         os.path.join(phase_dir, "training_history.csv"),
#         _history_rows(result["history"]),
#     )
#     classwise_path = save_csv(
#         os.path.join(phase_dir, "classwise_metrics.csv"),
#         _classwise_rows(
#             result=result,
#             class_ids=class_ids,
#             class_names=dataset.target_names,
#         ),
#     )
#     confusion_path = save_csv(
#         os.path.join(phase_dir, "confusion_matrix.csv"),
#         _confusion_rows(result["test"], class_ids),
#     )

#     class_structure_path = save_json(
#         os.path.join(phase_dir, "class_structure_report.json"),
#         {
#             "phase": 0,
#             "class_structure": class_structure,
#             "decision_state": decision_state,
#         },
#     )

#     if bool(args.save_visualizations):
#         figure_dir = os.path.join(phase_dir, "figures")
#         save_geometry_diagnostic_figures(
#             output_dir=os.path.join(figure_dir, "class_structure"),
#             train=result["train"],
#             validation=result["validation"],
#             test=result["test"],
#             structural_geometry=class_structure,
#             class_ids=class_ids,
#             target_names=dataset.target_names,
#             dpi=args.visualization_dpi,
#         )
#         generate_phase_qualitative_maps(
#             model=model,
#             dataset=dataset,
#             phase=0,
#             output_dir=os.path.join(figure_dir, "qualitative"),
#             device=args.device,
#             batch_size=args.eval_batch_size,
#             cmap_name=args.visualization_cmap,
#             dpi=args.visualization_dpi,
#         )

#     phase_report = {
#         "phase": 0,
#         "seen_class_ids": class_ids,
#         "epochs": int(result["final_epoch"]),
#         "train": _metric_summary(result["train"]),
#         "validation": _metric_summary(result["validation"]),
#         "test": _metric_summary(result["test"]),
#         "mean_identity_decision_margin": float(
#             decision_state["mean_identity_decision_margin"]
#         ),
#         "protocol": {
#             "exemplar_memory_size": 0,
#             "replay": False,
#             "stored_old_samples": 0,
#             "stored_old_features": 0,
#             "teacher": False,
#             "knowledge_distillation": False,
#         },
#     }
#     phase_report_path = save_json(
#         os.path.join(phase_dir, "phase_report.json"),
#         phase_report,
#     )

#     print("\nPhase 0 finalized.")
#     _print_metrics("Train", result["train"])
#     _print_metrics("Validation", result["validation"])
#     _print_metrics("Test", result["test"])
#     print(
#         "I/W decision margin | "
#         f"mean={float(decision_state['mean_identity_decision_margin']):.6f}"
#     )
#     print(f"Checkpoint       : {result['checkpoint']}")
#     print(f"Phase report     : {phase_report_path}")
#     print(f"Structure report : {class_structure_path}")
#     print(f"Training history : {history_path}")
#     print(f"Classwise metrics: {classwise_path}")
#     print(f"Confusion matrix : {confusion_path}")
#     return phase_report


# # -----------------------------------------------------------------------------
# # Incremental reporting -- consumes, never estimates, the trainer's phase R
# # -----------------------------------------------------------------------------


# def _load_reporting_reference_model(
#     *,
#     args: argparse.Namespace,
#     checkpoint_path: str,
#     expected_class_ids: Sequence[int],
# ) -> NECILModel:
#     source = os.path.abspath(str(checkpoint_path))
#     if not os.path.isfile(source):
#         raise FileNotFoundError(source)

#     payload = torch.load(source, map_location=args.device)
#     if not isinstance(payload, Mapping):
#         raise RuntimeError("reporting checkpoint payload must be a mapping")
#     state = payload.get("model_state")
#     if not isinstance(state, Mapping):
#         raise RuntimeError("reporting checkpoint lacks model_state")

#     reference = NECILModel(args)
#     reference.to(torch.device(args.device))
#     try:
#         reference.load_state_dict(state, strict=True)
#     except RuntimeError as exc:
#         raise RuntimeError(
#             "previous checkpoint is incompatible with the current model"
#         ) from exc

#     reference.eval()
#     reference.validate_finalized_state()
#     expected = [int(v) for v in expected_class_ids]
#     if list(reference.classifier_class_ids) != expected:
#         raise RuntimeError(
#             "previous checkpoint classifier classes do not match old classes"
#         )
#     if list(reference.committed_class_ids) != expected:
#         raise RuntimeError(
#             "previous checkpoint class statistics do not match old classes"
#         )
#     return reference


# def _incremental_representation_report(
#     *,
#     args: argparse.Namespace,
#     dataset: IncrementalHSIDatasetManager,
#     current_model: NECILModel,
#     phase: int,
#     old_class_ids: Sequence[int],
#     new_class_ids: Sequence[int],
#     previous_checkpoint_path: str,
#     phase_transform: torch.Tensor,
#     figure_output_dir: Optional[str],
# ) -> Dict[str, Any]:
#     phase_id = int(phase)
#     if phase_id <= 0:
#         raise ValueError("incremental representation reporting requires phase > 0")

#     old_ids = [int(v) for v in old_class_ids]
#     new_ids = [int(v) for v in new_class_ids]
#     seen_ids = old_ids + new_ids
#     if not old_ids or not new_ids or len(seen_ids) != len(set(seen_ids)):
#         raise ValueError("old/new class IDs are invalid")

#     reporting_loader = dataset.get_reporting_dataloader(
#         phase_id,
#         batch_size=int(args.eval_batch_size),
#     )
#     previous_model = _load_reporting_reference_model(
#         args=args,
#         checkpoint_path=previous_checkpoint_path,
#         expected_class_ids=old_ids,
#     )

#     try:
#         previous_old = collect_reporting_features(
#             previous_model,
#             reporting_loader,
#             class_ids=old_ids,
#             device=args.device,
#             require_coordinates=True,
#         )
#         current_old = collect_reporting_features(
#             current_model,
#             reporting_loader,
#             class_ids=old_ids,
#             device=args.device,
#             require_coordinates=True,
#         )

#         transport_report = transport_drift_diagnostics(
#             previous_old,
#             current_old,
#             old_class_ids=old_ids,
#             transform=phase_transform,
#         )

#         tsne_path: Optional[str] = None
#         if figure_output_dir is not None:
#             current_seen = collect_reporting_features(
#                 current_model,
#                 reporting_loader,
#                 class_ids=seen_ids,
#                 device=args.device,
#                 require_coordinates=True,
#             )
#             os.makedirs(figure_output_dir, exist_ok=True)
#             tsne_path = save_incremental_tsne_comparison(
#                 os.path.join(
#                     figure_output_dir,
#                     f"phase_{phase_id - 1:02d}_vs_phase_{phase_id:02d}_tsne.png",
#                 ),
#                 base_features=previous_old["ordered_features"],
#                 base_labels=previous_old["labels"],
#                 current_features=current_seen["ordered_features"],
#                 current_labels=current_seen["labels"],
#                 old_class_ids=old_ids,
#                 new_class_ids=new_ids,
#                 target_names=dataset.target_names,
#                 total_classes=len(dataset.target_names),
#                 cmap_name=str(args.visualization_cmap),
#                 seed=int(args.seed),
#                 dpi=int(args.visualization_dpi),
#             )

#         return {"transport_drift": transport_report, "tsne": tsne_path}
#     finally:
#         del previous_model
#         if torch.cuda.is_available() and torch.device(args.device).type == "cuda":
#             torch.cuda.empty_cache()


# def _incremental_history_rows(
#     history: Sequence[Mapping[str, Any]],
# ) -> list[Dict[str, Any]]:
#     rows: list[Dict[str, Any]] = []
#     for item in history:
#         rows.append(
#             {
#                 "epoch": int(item["epoch"]),
#                 "learning_rate": float(item["learning_rate"]),
#                 "classification_loss": float(item["classification_loss"]),
#                 "new_train_oa": float(item["new_train_oa"]),
#                 "new_train_ba": float(item["new_train_ba"]),
#                 "validation_new_ba": item.get("validation_new_ba"),
#                 "transport_raw_feature_change_rms": item.get(
#                     "transport_raw_feature_change_rms"
#                 ),
#                 "transport_aligned_feature_residual_rms": item.get(
#                     "transport_aligned_feature_residual_rms"
#                 ),
#                 "transport_orthogonality_residual": item.get(
#                     "transport_orthogonality_residual"
#                 ),
#             }
#         )
#     return rows


# def _report_incremental_phase(
#     *,
#     args: argparse.Namespace,
#     dataset: IncrementalHSIDatasetManager,
#     model: NECILModel,
#     result: Mapping[str, Any],
#     previous_checkpoint_path: str,
# ) -> Dict[str, Any]:
#     phase = int(result["phase"])
#     old_ids = [int(v) for v in result["old_class_ids"]]
#     new_ids = [int(v) for v in result["new_class_ids"]]
#     seen_ids = [int(v) for v in result["seen_class_ids"]]
#     if seen_ids != old_ids + new_ids:
#         raise RuntimeError("incremental seen classes must equal old followed by new")

#     if "phase_transform" not in result:
#         raise RuntimeError(
#             "incremental trainer is not aligned with locked SCOPE: "
#             "missing transient phase_transform"
#         )
#     phase_transform = torch.as_tensor(
#         result["phase_transform"],
#         device=model.device,
#         dtype=model.dtype,
#     )
#     expected_transform_shape = (
#         int(model.spectral_token_count),
#         int(model.spectral_token_count),
#     )
#     if tuple(phase_transform.shape) != expected_transform_shape:
#         raise RuntimeError(
#             f"phase_transform must have shape {expected_transform_shape}"
#         )

#     model.validate_finalized_state()
#     phase_dir = os.path.join(args.save_dir, f"phase_{phase}")
#     os.makedirs(phase_dir, exist_ok=True)
#     figure_dir = os.path.join(phase_dir, "figures")

#     history_path = save_csv(
#         os.path.join(phase_dir, "training_history.csv"),
#         _incremental_history_rows(result["history"]),
#     )
#     classwise_path = save_csv(
#         os.path.join(phase_dir, "classwise_metrics.csv"),
#         [
#             {
#                 "class_id": class_id,
#                 "display_id": class_id + 1,
#                 "class_name": str(dataset.target_names[class_id]),
#                 "phase_role": "old" if class_id in set(old_ids) else "new",
#                 "test_samples": _mapping_value(
#                     result["cumulative_test"].get("class_counts"), class_id
#                 ),
#                 "test_accuracy": _mapping_value(
#                     result["cumulative_test"].get("per_class_accuracy"), class_id
#                 ),
#                 "test_cosine_margin": _mapping_value(
#                     result["cumulative_test"].get("per_class_mean_cosine_margin"),
#                     class_id,
#                 ),
#             }
#             for class_id in seen_ids
#         ],
#     )
#     confusion_path = save_csv(
#         os.path.join(phase_dir, "confusion_matrix.csv"),
#         _confusion_rows(result["cumulative_test"], seen_ids),
#     )

#     representation_report = _incremental_representation_report(
#         args=args,
#         dataset=dataset,
#         current_model=model,
#         phase=phase,
#         old_class_ids=old_ids,
#         new_class_ids=new_ids,
#         previous_checkpoint_path=previous_checkpoint_path,
#         phase_transform=phase_transform,
#         figure_output_dir=(
#             os.path.join(figure_dir, "representation")
#             if bool(args.save_visualizations)
#             else None
#         ),
#     )

#     class_structure = class_structure_diagnostics(
#         model,
#         seen_ids,
#         target_names=dataset.target_names,
#     )
#     decision_state = decision_state_diagnostics(model, seen_ids)

#     diagnostic_path = save_json(
#         os.path.join(phase_dir, "class_structure_report.json"),
#         {
#             "phase": phase,
#             "old_class_ids": old_ids,
#             "new_class_ids": new_ids,
#             "class_structure": class_structure,
#             "decision_state": decision_state,
#             "transport_drift": representation_report["transport_drift"],
#         },
#     )

#     old_group = result["old_new"]["old"]
#     new_group = result["old_new"]["new"]
#     transport_drift = representation_report["transport_drift"]

#     phase_report = {
#         "phase": phase,
#         "old_class_ids": old_ids,
#         "new_class_ids": new_ids,
#         "seen_class_ids": seen_ids,
#         "epochs": int(result["final_epoch"]),
#         "cumulative": _metric_summary(result["cumulative_test"]),
#         "old": {
#             "ba": float(old_group["balanced_accuracy"]),
#             "min_class_accuracy": float(old_group["minimum_class_accuracy"]),
#             "cosine_margin": float(old_group["mean_cosine_margin"]),
#         },
#         "new": {
#             "ba": float(new_group["balanced_accuracy"]),
#             "min_class_accuracy": float(new_group["minimum_class_accuracy"]),
#             "cosine_margin": float(new_group["mean_cosine_margin"]),
#         },
#         "harmonic_old_new_accuracy": float(
#             result["old_new"]["harmonic_old_new_accuracy"]
#         ),
#         "forgetting": {
#             "old_balanced_accuracy_before": float(
#                 result["forgetting"]["old_balanced_accuracy_before"]
#             ),
#             "old_balanced_accuracy_after": float(
#                 result["forgetting"]["old_balanced_accuracy_after"]
#             ),
#             "old_balanced_accuracy_drop": float(
#                 result["forgetting"]["old_balanced_accuracy_drop"]
#             ),
#         },
#         "transport": {
#             "raw_feature_change_rms": float(
#                 transport_drift["raw_feature_change_rms"]
#             ),
#             "aligned_feature_residual_rms": float(
#                 transport_drift["aligned_feature_residual_rms"]
#             ),
#             "mean_raw_identity_drift_rms": float(
#                 transport_drift["mean_raw_identity_drift_rms"]
#             ),
#             "mean_aligned_identity_residual_rms": float(
#                 transport_drift["mean_aligned_identity_residual_rms"]
#             ),
#             "mean_raw_structure_drift_rms": float(
#                 transport_drift["mean_raw_structure_drift_rms"]
#             ),
#             "mean_aligned_structure_residual_rms": float(
#                 transport_drift["mean_aligned_structure_residual_rms"]
#             ),
#         },
#         "mean_identity_decision_margin": float(
#             decision_state["mean_identity_decision_margin"]
#         ),
#         "protocol": {
#             "exemplar_memory_size": 0,
#             "replay": False,
#             "stored_old_samples": 0,
#             "stored_old_features": 0,
#             "teacher": False,
#             "knowledge_distillation": False,
#         },
#     }
#     phase_report_path = save_json(
#         os.path.join(phase_dir, "phase_report.json"),
#         phase_report,
#     )

#     if bool(args.save_visualizations):
#         generate_phase_qualitative_maps(
#             model=model,
#             dataset=dataset,
#             phase=phase,
#             output_dir=os.path.join(figure_dir, "qualitative"),
#             device=args.device,
#             batch_size=args.eval_batch_size,
#             cmap_name=args.visualization_cmap,
#             dpi=args.visualization_dpi,
#         )

#     print(f"\nPhase {phase} finalized.")
#     _print_metrics("Cumulative", result["cumulative_test"])
#     print(
#         f"Old BA={100.0 * float(old_group['balanced_accuracy']):.2f}% | "
#         f"New BA={100.0 * float(new_group['balanced_accuracy']):.2f}% | "
#         f"H={100.0 * float(result['old_new']['harmonic_old_new_accuracy']):.2f}%"
#     )
#     print(
#         "Transport | "
#         f"raw Q={float(transport_drift['raw_feature_change_rms']):.6f} | "
#         f"aligned Q={float(transport_drift['aligned_feature_residual_rms']):.6f} | "
#         f"raw I={float(transport_drift['mean_raw_identity_drift_rms']):.6f} | "
#         f"aligned I={float(transport_drift['mean_aligned_identity_residual_rms']):.6f} | "
#         f"raw S={float(transport_drift['mean_raw_structure_drift_rms']):.6f} | "
#         f"aligned S={float(transport_drift['mean_aligned_structure_residual_rms']):.6f}"
#     )
#     print(f"Checkpoint       : {result['checkpoint']}")
#     print(f"Phase report     : {phase_report_path}")
#     print(f"Structure report : {diagnostic_path}")
#     print(f"Training history : {history_path}")
#     print(f"Classwise metrics: {classwise_path}")
#     print(f"Confusion matrix : {confusion_path}")
#     if representation_report["tsne"] is not None:
#         print(f"Joint t-SNE      : {representation_report['tsne']}")
#     return phase_report


# # -----------------------------------------------------------------------------
# # Experiment
# # -----------------------------------------------------------------------------


# def run_experiment(args: argparse.Namespace) -> Dict[str, Any]:
#     os.makedirs(args.save_dir, exist_ok=True)

#     dataset, data_summary, preprocessing_state = _build_data(args)
#     save_json(os.path.join(args.save_dir, "data_protocol.json"), data_summary)
#     SaveHSIPreprocessor(
#         os.path.join(args.save_dir, "phase_0_preprocessor.npz"),
#         preprocessing_state,
#     )
#     save_json(os.path.join(args.save_dir, "experiment_config.json"), vars(args))

#     model = NECILModel(args)
#     trainer = Trainer(model, dataset, args)
#     phase_reports: Dict[int, Dict[str, Any]] = {}

#     mode = str(args.run_mode)
#     if mode in {"base", "all"}:
#         base_result = trainer.run_base_only(
#             epochs=args.epochs_base,
#             batch_size=args.batch_size,
#             lr=args.lr,
#         )
#         phase_reports[0] = _report_base_phase(
#             args=args,
#             dataset=dataset,
#             model=model,
#             result=base_result,
#         )
#         previous_checkpoint_path = str(base_result["checkpoint"])
#         next_phase: Optional[int] = 1
#     else:
#         restored = trainer.load_checkpoint(args.resume_checkpoint)
#         previous_checkpoint_path = str(restored["checkpoint"])
#         next_phase = restored["next_phase"]
#         if next_phase is None:
#             raise RuntimeError("checkpoint already contains the final protocol phase")

#     if mode in {"incremental", "all"}:
#         if next_phase is None:
#             raise RuntimeError("no incremental phase remains")
#         for phase in range(int(next_phase), int(dataset.nb_tasks)):
#             result = trainer.run_incremental_phase(
#                 phase=phase,
#                 epochs=args.epochs_inc,
#                 batch_size=args.batch_size,
#                 lr=args.lr_inc,
#             )
#             phase_reports[phase] = _report_incremental_phase(
#                 args=args,
#                 dataset=dataset,
#                 model=model,
#                 result=result,
#                 previous_checkpoint_path=previous_checkpoint_path,
#             )
#             previous_checkpoint_path = str(result["checkpoint"])

#     finalized_phases = [int(v) for v in dataset.finalized_phases]
#     if not finalized_phases:
#         raise RuntimeError("experiment completed without a finalized phase")
#     if finalized_phases != list(range(len(finalized_phases))):
#         raise RuntimeError("finalized phases are not contiguous from zero")

#     latest = finalized_phases[-1]
#     model.validate_finalized_state()
#     latest_loader = dataset.get_reporting_dataloader(
#         latest,
#         batch_size=args.eval_batch_size,
#     )
#     latest_metrics = trainer.evaluate_loader(
#         latest_loader,
#         class_ids=list(model.classifier_class_ids),
#         target_class_ids=list(model.classifier_class_ids),
#     )

#     run_summary = {
#         "dataset": str(args.dataset),
#         "seed": int(args.seed),
#         "run_mode": mode,
#         "finalized_phases": finalized_phases,
#         "seen_class_ids": list(model.classifier_class_ids),
#         "latest_test": _metric_summary(latest_metrics),
#         "protocol": {
#             "exemplar_memory_size": 0,
#             "replay": False,
#             "stored_old_samples": 0,
#             "stored_old_features": 0,
#             "teacher": False,
#             "knowledge_distillation": False,
#         },
#     }
#     summary_path = save_json(
#         os.path.join(args.save_dir, "run_summary.json"),
#         run_summary,
#     )
#     run_summary["run_summary"] = summary_path

#     print("\nRun complete.")
#     print(f"Finalized phases : {finalized_phases}")
#     print(f"Seen classes     : {list(model.classifier_class_ids)}")
#     print(f"Run summary      : {summary_path}")
#     return run_summary


# def main(argv: Optional[Sequence[str]] = None) -> None:
#     args = build_parser().parse_args(argv)
#     validate_args(args)
#     set_seed(args.seed, args.deterministic)
#     args.device = resolve_device(args.device)
#     args.save_dir = os.path.abspath(args.save_dir)
#     run_experiment(args)


# if __name__ == "__main__":
#     main()
