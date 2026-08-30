from __future__ import annotations

"""SCOPE-HSI experiment driver.

Runtime contract
----------------

    HSI -> Mamba backbone -> Q_raw -> global-RMS canonical Q -> classifier W
                                             |
                                             +-> {I_c,S_c,N_c}

Phase 0 learns Q and the persistent angular classifier. Finalized class
statistics are estimated from the same Q. Incremental phases freeze Q, append
current-class classifier rows initialized from current identities, and jointly
optimize all seen classifier rows using real current-class risk plus retained
historical I_c/S_c structural classification risk.

This driver validates arguments, constructs the non-exemplar protocol,
dispatches training, and writes reporting-only artifacts.
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
    decision_state_diagnostics,
    reference_anchored_spectral_region_diagnostics,
)
from utils.qualitative_report import generate_phase_qualitative_maps
from utils.visualize import (
    save_historical_spectral_decision_figure,
    save_incremental_training_figure,
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
            "angular classifier, and historical structural classifier supervision "
            "for exemplar-free incremental learning."
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
            "Optional independent experiment seeds. With --run_mode base or all, "
            "main.py runs one fresh experiment per seed under save_dir/seed_<seed>/ "
            "and writes aggregate mean/std reports. Use one fixed --protocol_seed "
            "to measure optimization robustness on identical data splits."
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
    reporting.add_argument("--save_visualizations", type=str2bool, default=False)
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
        if str(args.run_mode) == "incremental":
            raise ValueError(
                "--seeds supports --run_mode base or all; incremental resume runs "
                "require an explicit checkpoint per seed"
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
        "incremental_consistency_rule": (
            "Q frozen; all seen W jointly optimized from equal-class current "
            "real risk plus reference-anchored old-old retention and old-new "
            "spectral-region exclusion from historical I/S"
        ),
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


def _metric_summary(
    metrics: Optional[Mapping[str, Any]],
) -> Optional[Dict[str, Any]]:
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
    validation = result.get("validation")
    rows: list[Dict[str, Any]] = []
    for class_id in [int(v) for v in class_ids]:
        rows.append(
            {
                "class_id": class_id,
                "display_id": class_id + 1,
                "class_name": str(class_names[class_id]),
                "train_accuracy": _mapping_value(
                    result["train"].get("per_class_accuracy"), class_id
                ),
                "validation_accuracy": (
                    None
                    if not isinstance(validation, Mapping)
                    else _mapping_value(
                        validation.get("per_class_accuracy"), class_id
                    )
                ),
                "test_accuracy": _mapping_value(
                    result["test"].get("per_class_accuracy"), class_id
                ),
                "test_cosine_margin": _mapping_value(
                    result["test"].get("per_class_mean_cosine_margin"), class_id
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


def _print_metrics(
    name: str,
    metrics: Optional[Mapping[str, Any]],
) -> None:
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

    class_structure = class_structure_diagnostics(
        model,
        class_ids,
        target_names=dataset.target_names,
    )
    decision_state = decision_state_diagnostics(model, class_ids)
    structure_report_path = save_json(
        os.path.join(phase_dir, "class_structure_report.json"),
        {
            "phase": 0,
            "class_structure": class_structure,
            "decision_state": decision_state,
        },
    )

    visualizations: Dict[str, Any] = {}
    if bool(args.save_visualizations):
        visualizations["qualitative"] = generate_phase_qualitative_maps(
            model=model,
            dataset=dataset,
            phase=0,
            output_dir=os.path.join(phase_dir, "figures", "qualitative"),
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
        "visualizations": visualizations,
    }
    phase_report_path = save_json(
        os.path.join(phase_dir, "phase_report.json"),
        phase_report,
    )

    print("\nPhase 0 finalized.")
    _print_metrics("Train", result["train"])
    _print_metrics("Validation", result["validation"])
    _print_metrics("Test", result["test"])
    print(f"Checkpoint       : {result['checkpoint']}")
    print(f"Phase report     : {phase_report_path}")
    print(f"Structure report : {structure_report_path}")
    print(f"Training history : {history_path}")
    print(f"Classwise metrics: {classwise_path}")
    print(f"Confusion matrix : {confusion_path}")
    return phase_report


# -----------------------------------------------------------------------------
# Incremental reporting
# -----------------------------------------------------------------------------


def _incremental_history_rows(
    history: Sequence[Mapping[str, Any]],
) -> list[Dict[str, Any]]:
    return [
        {
            "epoch": int(item["epoch"]),
            "learning_rate": float(item["learning_rate"]),
            "current_real_risk": float(item["current_real_risk"]),
            "historical_spectral_decision_risk": float(
                item["historical_spectral_decision_risk"]
            ),
            "historical_identity_ce": float(item["historical_identity_ce"]),
            "historical_spectral_score_mean": float(
                item["historical_spectral_score_mean"]
            ),
            "historical_spectral_score_min": float(
                item["historical_spectral_score_min"]
            ),
            "historical_dispersion_mean": float(
                item["historical_dispersion_mean"]
            ),
            "historical_zero_dispersion_fraction": float(
                item["historical_zero_dispersion_fraction"]
            ),
            "total_loss": float(item["total_loss"]),
            "new_train_ba": float(item["new_train_ba"]),
            "validation_new_ba": item.get("validation_new_ba"),
            "historical_identity_accuracy": float(
                item["historical_identity_accuracy"]
            ),
            "historical_identity_margin": float(
                item["historical_identity_margin"]
            ),
            "old_old_retention_risk": float(item["old_old_retention_risk"]),
            "old_new_exclusion_risk": float(item["old_new_exclusion_risk"]),
            "old_old_reference_min_score": float(
                item["old_old_reference_min_score"]
            ),
            "old_old_current_min_score": float(
                item["old_old_current_min_score"]
            ),
            "old_old_max_degradation": float(
                item["old_old_max_degradation"]
            ),
            "old_old_violation_fraction": float(
                item["old_old_violation_fraction"]
            ),
            "old_new_min_score": float(item["old_new_min_score"]),
            "old_new_violation_fraction": float(
                item["old_new_violation_fraction"]
            ),
        }
        for item in history
    ]


def _report_incremental_phase(
    *,
    args: argparse.Namespace,
    dataset: IncrementalHSIDatasetManager,
    model: NECILModel,
    result: Mapping[str, Any],
) -> Dict[str, Any]:
    phase = int(result["phase"])
    old_ids = [int(v) for v in result["old_class_ids"]]
    new_ids = [int(v) for v in result["new_class_ids"]]
    seen_ids = [int(v) for v in result["seen_class_ids"]]
    model.validate_finalized_state()

    phase_dir = os.path.join(args.save_dir, f"phase_{phase}")
    os.makedirs(phase_dir, exist_ok=True)

    method = dict(result["method_state"])
    spectral_decision = reference_anchored_spectral_region_diagnostics(
        model,
        historical_class_ids=old_ids,
        new_class_ids=new_ids,
        seen_class_ids=seen_ids,
        reference_old_old_score=method["reference_old_old_score_matrix"],
        target_names=dataset.target_names,
    )
    spectral_by_class = {
        int(row["class_id"]): row for row in spectral_decision["per_class"]
    }

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
                "phase_role": "historical" if class_id in set(old_ids) else "current",
                "test_accuracy": _mapping_value(
                    result["cumulative_test"].get("per_class_accuracy"), class_id
                ),
                "test_cosine_margin": _mapping_value(
                    result["cumulative_test"].get(
                        "per_class_mean_cosine_margin"
                    ),
                    class_id,
                ),
                "worst_rival_class_id": (
                    spectral_by_class[class_id]["worst_rival_class_id"]
                    if class_id in spectral_by_class
                    else None
                ),
                "minimum_spectral_decision_score": (
                    spectral_by_class[class_id]["minimum_spectral_decision_score"]
                    if class_id in spectral_by_class
                    else None
                ),
                "worst_old_old_rival_class_id": (
                    spectral_by_class[class_id]["worst_old_old_rival_class_id"]
                    if class_id in spectral_by_class
                    else None
                ),
                "worst_old_old_degradation": (
                    spectral_by_class[class_id]["worst_old_old_degradation"]
                    if class_id in spectral_by_class
                    else None
                ),
                "worst_old_new_rival_class_id": (
                    spectral_by_class[class_id]["worst_old_new_rival_class_id"]
                    if class_id in spectral_by_class
                    else None
                ),
                "worst_old_new_score": (
                    spectral_by_class[class_id]["worst_old_new_score"]
                    if class_id in spectral_by_class
                    else None
                ),
            }
            for class_id in seen_ids
        ],
    )
    confusion_path = save_csv(
        os.path.join(phase_dir, "confusion_matrix.csv"),
        _confusion_rows(result["cumulative_test"], seen_ids),
    )
    spectral_report_path = save_json(
        os.path.join(phase_dir, "historical_spectral_decision_report.json"),
        {
            "phase": phase,
            "diagnostics": spectral_decision,
            "phase_origin_confusion": result.get("phase_origin_confusion"),
        },
    )

    visualizations: Dict[str, Any] = {}
    if bool(args.save_visualizations):
        figure_dir = os.path.join(phase_dir, "figures")
        visualizations["training"] = save_incremental_training_figure(
            output_dir=os.path.join(figure_dir, "training"),
            history=result["history"],
            dpi=args.visualization_dpi,
        )
        visualizations["historical_spectral_decision"] = (
            save_historical_spectral_decision_figure(
                output_dir=os.path.join(figure_dir, "spectral_decision"),
                diagnostics=spectral_decision,
                dpi=args.visualization_dpi,
            )
        )
        visualizations["qualitative"] = generate_phase_qualitative_maps(
            model=model,
            dataset=dataset,
            phase=phase,
            output_dir=os.path.join(figure_dir, "qualitative"),
            device=args.device,
            batch_size=args.eval_batch_size,
            cmap_name=args.visualization_cmap,
            dpi=args.visualization_dpi,
        )

    old_group = result["old_new"]["old"]
    new_group = result["old_new"]["new"]
    phase_report = {
        "phase": phase,
        "strategy": str(method["strategy"]),
        "old_class_ids": old_ids,
        "new_class_ids": new_ids,
        "seen_class_ids": seen_ids,
        "cumulative": _metric_summary(result["cumulative_test"]),
        "old_ba": float(old_group["balanced_accuracy"]),
        "new_ba": float(new_group["balanced_accuracy"]),
        "harmonic_old_new_accuracy": float(
            result["old_new"]["harmonic_old_new_accuracy"]
        ),
        "forgetting": float(result["forgetting"]["old_balanced_accuracy_drop"]),
        "historical_spectral_decision_risk": float(
            method["final_historical_spectral_decision_risk"]
        ),
        "historical_identity_ce": float(method["final_historical_identity_ce"]),
        "historical_spectral_score_mean": float(
            method["final_historical_spectral_score_mean"]
        ),
        "historical_spectral_score_min": float(
            method["final_historical_spectral_score_min"]
        ),
        "historical_dispersion_mean": float(
            method["final_historical_dispersion_mean"]
        ),
        "historical_identity_accuracy": float(
            method["final_historical_identity_accuracy"]
        ),
        "historical_identity_margin": float(
            method["final_historical_identity_margin"]
        ),
        "old_old_retention_risk": float(method["final_old_old_retention_risk"]),
        "old_new_exclusion_risk": float(method["final_old_new_exclusion_risk"]),
        "old_old_reference_min_score": float(
            method["final_old_old_reference_min_score"]
        ),
        "old_old_current_min_score": float(
            method["final_old_old_current_min_score"]
        ),
        "old_old_max_degradation": float(
            method["final_old_old_max_degradation"]
        ),
        "old_old_violation_fraction": float(
            method["final_old_old_violation_fraction"]
        ),
        "old_new_min_score": float(method["final_old_new_min_score"]),
        "old_new_violation_fraction": float(
            method["final_old_new_violation_fraction"]
        ),
        "phase_origin_confusion": result.get("phase_origin_confusion"),
        "visualizations": visualizations,
        "invariants": {
            "backbone_unchanged": bool(method["backbone_unchanged"]),
            "historical_geometry_unchanged": bool(
                method["historical_geometry_unchanged"]
            ),
            "classifier_rows_jointly_optimized": bool(
                method["classifier_rows_jointly_optimized"]
            ),
            "classifier_rows_unit_norm": bool(
                method["classifier_rows_unit_norm"]
            ),
        },
    }
    phase_report_path = save_json(
        os.path.join(phase_dir, "phase_report.json"), phase_report
    )

    print(f"\nPhase {phase} finalized.")
    _print_metrics("Cumulative", result["cumulative_test"])
    print(
        f"OldBA={100.0 * phase_report['old_ba']:.2f}% | "
        f"NewBA={100.0 * phase_report['new_ba']:.2f}% | "
        f"H={100.0 * phase_report['harmonic_old_new_accuracy']:.2f}% | "
        f"F={100.0 * phase_report['forgetting']:.2f}%"
    )
    print(
        f"HistRisk={phase_report['historical_spectral_decision_risk']:.4f} | "
        f"I-CE={phase_report['historical_identity_ce']:.4f} | "
        f"Rmean={phase_report['historical_spectral_score_mean']:.4f} | "
        f"Rmin={phase_report['historical_spectral_score_min']:.4f} | "
        f"HistIAcc={100.0 * phase_report['historical_identity_accuracy']:.2f}% | "
        f"HistMargin={phase_report['historical_identity_margin']:.4f}"
    )
    print(
        f"Region | OO={phase_report['old_old_retention_risk']:.4f} | "
        f"ON={phase_report['old_new_exclusion_risk']:.4f} | "
        f"OOrefMin={phase_report['old_old_reference_min_score']:.4f} | "
        f"OOcurMin={phase_report['old_old_current_min_score']:.4f} | "
        f"OOdeg={phase_report['old_old_max_degradation']:.4f} | "
        f"OOviol={100.0 * phase_report['old_old_violation_fraction']:.2f}% | "
        f"ONmin={phase_report['old_new_min_score']:.4f} | "
        f"ONviol={100.0 * phase_report['old_new_violation_fraction']:.2f}%"
    )
    print(
        "Invariant | "
        f"Q={phase_report['invariants']['backbone_unchanged']} | "
        f"oldI/S={phase_report['invariants']['historical_geometry_unchanged']} | "
        f"Wjoint={phase_report['invariants']['classifier_rows_jointly_optimized']} | "
        f"Wunit={phase_report['invariants']['classifier_rows_unit_norm']}"
    )
    print(f"Checkpoint       : {result['checkpoint']}")
    print(f"Phase report     : {phase_report_path}")
    print(f"Spectral report  : {spectral_report_path}")
    print(f"Training history : {history_path}")
    print(f"Classwise metrics: {classwise_path}")
    print(f"Confusion matrix : {confusion_path}")
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
        next_phase: Optional[int] = 1
    else:
        restored = trainer.load_checkpoint(args.resume_checkpoint)
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
            )

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
        "incremental_strategy": (
            "stable_Q_reference_anchored_spectral_decision_regions"
            if latest > 0
            else None
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


def run_multiseed_experiments(args: argparse.Namespace) -> Dict[str, Any]:
    if args.seeds is None:
        raise ValueError("run_multiseed_experiments requires --seeds")
    if str(args.run_mode) not in {"base", "all"}:
        raise ValueError("multi-seed experiments support --run_mode base or all")

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

    print(f"\nIndependent multi-seed {args.run_mode} validation")
    print(f"Seeds    : {seeds}")
    print(f"Root dir : {root}")

    for index, seed in enumerate(seeds, start=1):
        seed_args = copy.deepcopy(args)
        seed_args.seed = int(seed)
        seed_args.seeds = None
        seed_args.run_mode = str(args.run_mode)
        seed_args.resume_checkpoint = ""
        seed_args.save_dir = os.path.join(root, f"seed_{seed}")

        # Every seed is a complete independent experiment: new protocol/split,
        # preprocessing fit, model, classifier, class statistics, trainer, and optimizer.
        set_seed(seed_args.seed, bool(seed_args.deterministic))

        print(
            f"\n{'=' * 78}\n"
            f"Seed {seed} ({index}/{len(seeds)}) | mode={args.run_mode}\n"
            f"Output: {seed_args.save_dir}\n"
            f"{'=' * 78}"
        )
        seed_run_summary = run_experiment(seed_args)

        phase_report = _load_json_mapping(
            os.path.join(seed_args.save_dir, "phase_0", "phase_report.json")
        )
        per_seed_phase_reports.append((seed, phase_report))

        classwise_rows = _load_csv_rows(
            os.path.join(seed_args.save_dir, "phase_0", "classwise_metrics.csv")
        )
        per_seed_classwise.append((seed, classwise_rows))

        latest_test = seed_run_summary.get("latest_test")
        if not isinstance(latest_test, Mapping):
            raise RuntimeError("multi-seed run summary lacks latest_test metrics")
        finalized = [int(v) for v in seed_run_summary.get("finalized_phases", [])]
        if not finalized:
            raise RuntimeError("multi-seed run finalized no phase")
        final_phase = finalized[-1]
        row: Dict[str, Any] = {
            "seed": seed,
            "output_dir": seed_args.save_dir,
            "final_phase": final_phase,
            "mean_identity_decision_margin": float(
                phase_report["mean_identity_decision_margin"]
            ),
        }
        for metric in metric_names:
            row[f"final_test_{metric}"] = float(latest_test[metric])
        if final_phase > 0:
            final_phase_report = _load_json_mapping(
                os.path.join(
                    seed_args.save_dir, f"phase_{final_phase}", "phase_report.json"
                )
            )
            for key in (
                "old_ba",
                "new_ba",
                "harmonic_old_new_accuracy",
                "forgetting",
                "old_old_max_degradation",
                "old_old_violation_fraction",
                "old_new_min_score",
                "old_new_violation_fraction",
            ):
                row[f"final_{key}"] = float(final_phase_report[key])

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

    final_statistics: Dict[str, Any] = {}
    for metric in metric_names:
        values = [float(row[f"final_test_{metric}"]) for row in run_rows]
        stats = _sample_statistics(values)
        final_statistics[metric] = stats
        summary_rows.append(
            {
                "scope": "final_test",
                "metric": metric,
                "mean": stats["mean"],
                "std": stats["std"],
                "min": stats["min"],
                "max": stats["max"],
                "seed_count": len(values),
            }
        )
    aggregate["final_test"] = final_statistics

    if str(args.run_mode) == "all":
        incremental_keys = (
            "old_ba",
            "new_ba",
            "harmonic_old_new_accuracy",
            "forgetting",
            "old_old_max_degradation",
            "old_old_violation_fraction",
            "old_new_min_score",
            "old_new_violation_fraction",
        )
        aggregate_incremental: Dict[str, Any] = {}
        for key in incremental_keys:
            values = [float(row[f"final_{key}"]) for row in run_rows]
            stats = _sample_statistics(values)
            aggregate_incremental[key] = stats
            summary_rows.append(
                {
                    "scope": "final_incremental",
                    "metric": key,
                    "mean": stats["mean"],
                    "std": stats["std"],
                    "min": stats["min"],
                    "max": stats["max"],
                    "seed_count": len(values),
                }
            )
        aggregate["final_incremental"] = aggregate_incremental

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
        "mode": f"independent_{args.run_mode}_multiseed",
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

    print(f"\nMulti-seed {args.run_mode} validation complete.")
    report_scope = aggregate.get("final_test", {})
    test_ba = report_scope.get("ba")
    test_oa = report_scope.get("oa")
    test_min = report_scope.get("min_class_accuracy")
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
    # Each independent seed run is reseeded again inside run_multiseed_experiments().
    set_seed(seeds_for_initialization[0], bool(args.deterministic))
    args.device = resolve_device(args.device)
    args.save_dir = os.path.abspath(args.save_dir)

    if args.seeds is not None:
        run_multiseed_experiments(args)
    else:
        run_experiment(args)


if __name__ == "__main__":
    main()
