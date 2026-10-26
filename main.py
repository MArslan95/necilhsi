from __future__ import annotations
import argparse
import copy
import csv
import json
import math
import os
import random
import sys
from numbers import Integral, Real
from typing import Any, Dict, Mapping, Optional, Sequence
import numpy as np
import torch

# Preserve --help without the external dataset modules while retaining the
# original supervised HSI preprocessing and protocol for actual experiments.
try:
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
except ModuleNotFoundError as exc:
    if not (exc.name == "data" or (exc.name or "").startswith("data.")):
        raise

    def _missing_data_dependency(*_args, **_kwargs):
        raise ImportError(
            "Real HSI training requires the original data/"
            "hsi_dataloader_pytorch.py and data/incremental_dataset.py"
        )

    ExtractLabeledPixelIndex = FitHSIPreprocessorFromProtocol = _missing_data_dependency
    LoadRawHSIData = SaveHSIPreprocessor = _missing_data_dependency
    IncrementalHSIDatasetManager = build_incremental_protocol = _missing_data_dependency
from models.necil_model import NECILModel
from trainers.trainer import Trainer
from utils.eval import (
    class_structure_diagnostics,
    structural_decision_diagnostics,
    cross_phase_cds_decision_diagnostics,
    summarize_evaluation_metrics,
)

try:
    from utils.qualitative_report import generate_phase_qualitative_maps
except ImportError:

    def generate_phase_qualitative_maps(**_kwargs):
        # This optional module was not in the supplied file set.
        return {
            "status": "qualitative reporting unavailable: include the original utils/qualitative_report.py and utils/visualize.py"
        }


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
        raise ValueError("class_order must be comma-separated without empty entries")

    values = [int(part.strip()) for part in parts]

    if len(values) != len(set(values)) or any(value < 0 for value in values):

        raise ValueError("class_order must contain unique non-negative IDs")

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


def save_csv(path: str, rows: Sequence[Mapping[str, Any]]) -> Optional[str]:

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

        with open(temporary, "w", newline="", encoding="utf-8-sig") as stream:

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

    value = _exact_int(seed, name="seed", minimum=0, maximum=(2**32) - 1)

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

    requested = torch.device(token)

    if requested.type == "cpu":

        return "cpu"

    if requested.type == "mps":

        backend = getattr(torch.backends, "mps", None)

        if requested.index is not None or backend is None or not backend.is_available():

            raise RuntimeError(f"MPS device {value} is unavailable")

        return "mps"

    if requested.type != "cuda" or not torch.cuda.is_available():

        raise RuntimeError(f"requested device {value!r} is unavailable")

    index = torch.cuda.current_device() if requested.index is None else requested.index

    if index is None or not 0 <= int(index) < torch.cuda.device_count():

        raise RuntimeError(f"CUDA device index {index} is unavailable")

    torch.cuda.set_device(int(index))

    return f"cuda:{int(index)}"


def _load_band_positions(path: str, expected_bands: int) -> Optional[list[float]]:

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


def build_parser() -> argparse.ArgumentParser:

    parser = argparse.ArgumentParser(
        description=(
            "Exemplar-free NECIL-HSI using one learned low-rank class covariance, "
            "native-band spectral variation, historical directional margin protection, "
            "trainable HSIMamba and one deployed all-seen cosine head."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    experiment = parser.add_argument_group("Experiment")
    experiment.add_argument("--dataset", default="IP")
    experiment.add_argument("--data_dir", default="./datasets")
    experiment.add_argument("--save_dir", default="./results_cds_hsi")
    experiment.add_argument("--seed", type=int, default=42)
    experiment.add_argument("--protocol_seed", type=int, default=None)
    experiment.add_argument("--seeds", type=int, nargs="+", default=None)
    experiment.add_argument(
        "--device",
        default="cuda:0",
        help="cpu, cuda[:index], or mps (Apple Silicon)",
    )
    experiment.add_argument("--num_workers", type=int, default=0)
    experiment.add_argument("--deterministic", type=str2bool, default=True)

    experiment.add_argument(
        "--run_mode",
        choices=("base", "incremental", "all"),
        default="base",
    )
    experiment.add_argument("--max_phase", type=int, default=None)
    experiment.add_argument("--resume_checkpoint", default="")
    protocol = parser.add_argument_group("Class-incremental protocol")
    protocol.add_argument("--base_classes", type=int, default=6)
    protocol.add_argument("--increment", type=int, default=6)
    protocol.add_argument("--class_order", default="")
    protocol.add_argument("--shuffle_order", type=str2bool, default=False)
    protocol.add_argument("--train_ratio", type=float, default=0.20)
    protocol.add_argument("--val_ratio", type=float, default=0.10)

    protocol.add_argument(
        "--split_strategy",
        choices=("random_pixel", "spatial_block"),
        default="random_pixel",
    )
    protocol.add_argument("--spatial_block_size", type=int, default=33)
    protocol.add_argument("--require_patch_disjoint", type=str2bool, default=False)
    protocol.add_argument(
        "--require_phase_context_isolation", type=str2bool, default=None
    )
    preprocessing = parser.add_argument_group("HSI preprocessing")
    preprocessing.add_argument("--patch_size", type=int, default=11)
    preprocessing.add_argument("--no_pca", action="store_true")
    preprocessing.add_argument("--pca_components", type=int, default=30)
    preprocessing.add_argument("--pca_whiten", type=str2bool, default=False)
    preprocessing.add_argument("--band_positions_file", default="")
    ##### Ordered spectral-spatial backbone ######
    backbone = parser.add_argument_group("Ordered spectral-spatial backbone")
    backbone.add_argument("--d_model", type=int, default=128)
    backbone.add_argument("--spectral_token_count", type=int, default=16)
    backbone.add_argument("--d_state", type=int, default=16)
    backbone.add_argument("--d_conv", type=int, default=4)
    backbone.add_argument("--expand", type=int, default=2)
    backbone.add_argument("--num_spectral_layers", type=int, default=2)
    backbone.add_argument("--num_spatial_layers", type=int, default=2)
    backbone.add_argument("--stem_norm_groups", type=int, default=8)
    backbone.add_argument("--mamba_dropout", type=float, default=0.0)

    optimization = parser.add_argument_group("Optimization")
    optimization.add_argument("--epochs_base", type=int, default=80)
    optimization.add_argument("--epochs_inc", type=int, default=35)
    optimization.add_argument("--batch_size", type=int, default=64)
    optimization.add_argument("--eval_batch_size", type=int, default=256)
    optimization.add_argument("--lr", type=float, default=1e-4)
    optimization.add_argument("--lr_inc", type=float, default=1e-4)
    optimization.add_argument("--weight_decay", type=float, default=1e-4)
    optimization.add_argument("--gradient_clip", type=float, default=5.0)
    #### NECIL HSI Method ####
    method = parser.add_argument_group("NECIL-HSI method")
    method.add_argument(
        "--historical_risk_mode",
        choices=("directional",),
        default="directional",
        help="Unified CDS old-class margin/low-rank variance supervision",
    )
    method.add_argument("--contrastive_weight", type=float, default=1.0)
    audit = parser.add_argument_group(
        "Read-only diagnosis (does not select checkpoints)"
    )
    audit.add_argument("--audit_enabled", type=str2bool, default=False)
    audit.add_argument("--audit_probe_steps", type=int, default=60)
    audit.add_argument("--audit_max_per_class", type=int, default=48)
    method.add_argument(
        "--historical_negative_mode",
        choices=("classifier", "mean", "distribution"),
        default="distribution",
        help="Historical contrastive negatives: deployed W, normalized moments' means, or transient compact-Gaussian Q draws",
    )
    method.add_argument(
        "--historical_negative_samples_per_class",
        type=int,
        default=32,
        help="Positive draw count per old class in distribution mode; mean uses 1 and classifier 0",
    )
    method.add_argument(
        "--kd_logit_weight",
        type=float,
        default=0.0,
        help="legacy ablation only; 0 disables logit KD in the CDS method",
    )
    method.add_argument("--kd_temperature", type=float, default=2.0)
    method.add_argument(
        "--cds_weight",
        type=float,
        default=1.0,
        help="weight for historical Whole-CDS intra/inter preservation during representation training",
    )

    method.add_argument(
        "--head_admission_epochs",
        type=int,
        default=5,
        help="Optional incoming-row initialization on real current HSI; 0 disables",
    )
    method.add_argument(
        "--head_admission_lr",
        type=float,
        default=1e-3,
        help="Optimizer LR for new-class classifier-only admission",
    )
    method.add_argument(
        "--historical_transport",
        type=str2bool,
        default=True,
        help="Attempt held-out validated orthogonal historical Q mean/covariance transport; rejected maps are identity",
    )

    method.add_argument(
        "--representation_train_new_head",
        type=str2bool,
        default=True,
        help="Required true: jointly train seen cosine rows with the backbone under historical CDS protection",
    )
    method.add_argument(
        "--representation_ce_scope",
        choices=("all_seen",),
        default="all_seen",
    )

    method.add_argument(
        "--representation_select_best_val",
        type=str2bool,
        default=False,
        help="Deprecated: always false. Keep final joint-training epoch; never select or restore earlier Q/W.",
    )

    method.add_argument(
        "--support_head_epochs",
        type=int,
        default=0,
        help="Existing same-head training epochs after final alignment; 0 commits the last frame used in representation training",
    )
    method.add_argument(
        "--support_head_lr",
        type=float,
        default=1e-3,
        help="Learning rate for optional fixed-Q, same-head training",
    )

    method.add_argument(
        "--final_head_steps",
        type=int,
        default=0,
        help="Maximum Pareto-accepted final W-only consistency steps using the finalized CDS",
    )

    method.add_argument(
        "--feature_kd_weight",
        type=float,
        default=10.0,
        help="ordered Feature-KD weight on paired real current HSI; retained in every incremental phase",
    )

    reporting = parser.add_argument_group("Reporting")

    reporting.add_argument("--save_visualizations", type=str2bool, default=False)

    reporting.add_argument("--visualization_cmap", default="nipy_spectral")

    reporting.add_argument("--visualization_dpi", type=int, default=300)

    return parser


def validate_args(args: argparse.Namespace) -> None:
    """Prevent silent fallback to the obsolete conditional/joint CDS method."""
    if args.representation_select_best_val:
        raise ValueError(
            "Phase-1 final-epoch training is mandatory: checkpoint selection is disabled; "
            "use --representation_select_best_val false"
        )
    if args.historical_risk_mode != "directional":
        raise ValueError("unified CDS requires --historical_risk_mode directional")
    if not args.representation_train_new_head:
        raise ValueError("unified CDS requires --representation_train_new_head true")
    if args.representation_ce_scope != "all_seen":
        raise ValueError("unified CDS requires --representation_ce_scope all_seen")
    if args.kd_logit_weight != 0:
        raise ValueError(
            "unified CDS uses Feature KD, not logit KD: --kd_logit_weight 0"
        )

    if args.historical_negative_mode not in ("classifier", "mean", "distribution"):
        raise ValueError(
            "historical_negative_mode must be classifier, mean or distribution"
        )
    _exact_int(
        args.historical_negative_samples_per_class,
        name="historical_negative_samples_per_class",
        minimum=1,
    )

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
    _exact_int(args.support_head_epochs, name="support_head_epochs", minimum=0)
    _finite_float(
        args.support_head_lr, name="support_head_lr", minimum=0.0, strict_minimum=True
    )
    _exact_int(args.final_head_steps, name="final_head_steps", minimum=0)
    if int(args.final_head_steps) != 0:
        raise ValueError(
            "final_head_steps was superseded by support_head_epochs; use --final_head_steps 0"
        )
    _exact_int(args.head_admission_epochs, name="head_admission_epochs", minimum=0)
    _exact_int(args.audit_probe_steps, name="audit_probe_steps", minimum=1)
    _exact_int(args.audit_max_per_class, name="audit_max_per_class", minimum=1)
    _finite_float(
        args.head_admission_lr,
        name="head_admission_lr",
        minimum=0.0,
        strict_minimum=True,
    )

    if args.max_phase is not None:

        _exact_int(args.max_phase, name="max_phase", minimum=0)

    _exact_int(args.seed, name="seed", minimum=0, maximum=(2**32) - 1)

    if args.protocol_seed is not None:

        _exact_int(
            args.protocol_seed, name="protocol_seed", minimum=0, maximum=(2**32) - 1
        )

    if args.seeds is not None:

        seeds = [
            _exact_int(value, name="seeds value", minimum=0, maximum=(2**32) - 1)
            for value in args.seeds
        ]

        if len(seeds) != len(set(seeds)):

            raise ValueError("--seeds must contain unique values")

        if str(args.run_mode) == "incremental":

            raise ValueError("--seeds is not supported with incremental resume")

    if not bool(args.no_pca):

        _exact_int(args.pca_components, name="pca_components", minimum=1)

    if int(args.base_classes) < 2 or int(args.spectral_token_count) < 2:

        raise ValueError("base_classes and spectral_token_count must be at least 2")

    if int(args.patch_size) % 2 == 0:

        raise ValueError("patch_size must be odd")

    if (
        int(args.stem_norm_groups) > int(args.d_model)
        or int(args.d_model) % int(args.stem_norm_groups) != 0
    ):

        raise ValueError("stem_norm_groups must divide d_model")

    train_ratio = _finite_float(
        args.train_ratio, name="train_ratio", minimum=0.0, strict_minimum=True
    )

    val_ratio = _finite_float(args.val_ratio, name="val_ratio", minimum=0.0)

    if train_ratio + val_ratio >= 1.0:

        raise ValueError("train_ratio + val_ratio must be < 1")

    if _finite_float(args.mamba_dropout, name="mamba_dropout", minimum=0.0) >= 1.0:

        raise ValueError("mamba_dropout must be < 1")

    for name in ("lr", "lr_inc"):

        _finite_float(getattr(args, name), name=name, minimum=0.0, strict_minimum=True)

    for name in (
        "weight_decay",
        "gradient_clip",
        "contrastive_weight",
        "kd_logit_weight",
        "cds_weight",
        "feature_kd_weight",
    ):

        _finite_float(getattr(args, name), name=name, minimum=0.0)

    _finite_float(
        args.kd_temperature, name="kd_temperature", minimum=0.0, strict_minimum=True
    )

    if str(args.run_mode) == "incremental" and not str(args.resume_checkpoint).strip():

        raise ValueError("--resume_checkpoint is required for --run_mode incremental")

    class_order = _parse_class_order(args.class_order)

    if class_order is not None and bool(args.shuffle_order):

        raise ValueError("class_order and shuffle_order are mutually exclusive")


def _build_data(
    args: argparse.Namespace,
) -> tuple[IncrementalHSIDatasetManager, Dict[str, Any], Mapping[str, Any]]:

    raw_cube, gt, class_count, names, _, label_policy = LoadRawHSIData(
        args.dataset, base_dir=args.data_dir
    )

    args.class_count = int(class_count)

    labels, coords = ExtractLabeledPixelIndex(gt, label_policy=label_policy)

    if int(args.base_classes) > int(class_count):

        raise ValueError(
            f"base_classes={args.base_classes} exceeds dataset classes={class_count}"
        )

    class_order = _parse_class_order(args.class_order)

    if class_order is not None and set(class_order) != set(range(int(class_count))):
        raise ValueError(
            f"class_order must be a permutation of 0..{int(class_count)-1}"
        )

    protocol_seed = (
        int(args.seed) if args.protocol_seed is None else int(args.protocol_seed)
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
        raise ValueError("spectral_token_count cannot exceed ordered raw bands")
    args.band_positions = _load_band_positions(
        args.band_positions_file, args.spectral_bands
    )
    args.band_wavelengths = None

    # The model fits both its raw-spectrum standardizer and ordered spectral-evidence

    # normalization from the real Phase-0 TRAIN spectra.

    args.spectral_normalization_mean = None

    args.spectral_normalization_std = None

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

    manager.preprocessing_state = preprocessing_state

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
        "phase_schedule": {
            int(phase): [int(value) for value in classes]
            for phase, classes in protocol["phase_to_classes"].items()
        },
        "processed_bands": int(processed_cube.shape[2]),
        "raw_spectral_bands": int(ordered_cube.shape[2]),
        "spectral_token_count": int(args.spectral_token_count),
        "preprocessing_fit_scope": str(preprocessing_state["fit_scope"]),
        "preprocessing_fit_pixel_count": int(preprocessing_state["fit_pixel_count"]),
        "validation_enabled": bool(manager.validation_enabled),
        "split_strategy": manager.split_strategy,
        "require_patch_disjoint": manager.require_patch_disjoint,
        "require_phase_context_isolation": manager.require_phase_context_isolation,
        "context_policy": manager.context_policy,
        "split_counts_by_global_class": split_counts,
        "stores_old_training_samples": False,
        "stores_old_feature_rows": False,
    }

    return manager, data_summary, preprocessing_state


def _mapping_value(mapping: Any, class_id: int) -> Any:
    if not isinstance(mapping, Mapping):
        return None
    return mapping[class_id] if class_id in mapping else mapping.get(str(class_id))


def _metric_summary(metrics: Optional[Mapping[str, Any]]) -> Optional[Dict[str, Any]]:
    return summarize_evaluation_metrics(metrics)


def _history_rows(history: Sequence[Mapping[str, Any]]) -> list[Dict[str, Any]]:
    return [dict(row) for row in history]


def _confusion_rows(
    metrics: Mapping[str, Any], class_ids: Sequence[int]
) -> list[Dict[str, Any]]:
    matrix = np.asarray(metrics["confusion_matrix"], dtype=np.int64)
    ids = [int(value) for value in class_ids]
    if matrix.shape != (len(ids), len(ids)):
        raise RuntimeError("confusion matrix shape disagrees with class_ids")

    rows = []

    for r, class_id in enumerate(ids):

        row: Dict[str, Any] = {
            "true_class_id": class_id,
            "true_display_id": class_id + 1,
        }
        for c, pred_id in enumerate(ids):
            row[f"pred_{pred_id}"] = int(matrix[r, c])
        rows.append(row)

    return rows


def _classwise_rows(
    *,
    metrics: Mapping[str, Any],
    class_ids: Sequence[int],
    class_names: Sequence[str],
    old_ids: Sequence[int] = (),
) -> list[Dict[str, Any]]:

    old = set(int(value) for value in old_ids)

    rows = []

    for class_id in [int(value) for value in class_ids]:

        rows.append(
            {
                "class_id": class_id,
                "display_id": class_id + 1,
                "class_name": str(class_names[class_id]),
                "phase_role": "historical" if class_id in old else "current",
                "test_accuracy": _mapping_value(
                    metrics.get("per_class_accuracy"), class_id
                ),
                "test_logit_margin": _mapping_value(
                    metrics.get("per_class_mean_logit_margin"), class_id
                ),
                "test_energy_margin": _mapping_value(
                    metrics.get("per_class_mean_energy_margin"), class_id
                ),
                "test_raw_energy_margin": _mapping_value(
                    metrics.get("per_class_mean_raw_energy_margin"), class_id
                ),
            }
        )

    return rows


def _print_metrics(name: str, metrics: Optional[Mapping[str, Any]]) -> None:

    if metrics is None:

        print(f"{name:<10} | disabled")

        return

    def number(value, *, percent=False):
        if value is None:
            return "N/A"
        return f"{100*float(value):.2f}%" if percent else f"{float(value):.4f}"

    print(
        f"{name:<10} | OA={number(metrics['overall_accuracy'], percent=True)} | "
        f"BA={number(metrics['balanced_accuracy'], percent=True)} | "
        f"MinClass={number(metrics['minimum_class_accuracy'], percent=True)} | "
        f"LogitMargin={number(metrics['mean_logit_margin'])} | "
        f"NormEnergy={number(metrics['mean_energy_margin'])} | "
        f"NormViol={number(metrics['energy_violation_fraction'], percent=True)} | "
        f"RawEnergy={number(metrics.get('mean_raw_energy_margin', metrics['mean_energy_margin']))}"
    )


def _report_base_phase(
    *,
    args: argparse.Namespace,
    dataset: IncrementalHSIDatasetManager,
    model: NECILModel,
    result: Mapping[str, Any],
) -> Dict[str, Any]:

    class_ids = [int(value) for value in result["class_ids"]]

    phase_dir = os.path.join(args.save_dir, "phase_0")

    os.makedirs(phase_dir, exist_ok=True)

    history_path = save_csv(
        os.path.join(phase_dir, "training_history.csv"),
        _history_rows(result["history"]),
    )

    classwise_path = save_csv(
        os.path.join(phase_dir, "classwise_metrics.csv"),
        _classwise_rows(
            metrics=result["test"],
            class_ids=class_ids,
            class_names=dataset.target_names,
        ),
    )

    confusion_path = save_csv(
        os.path.join(phase_dir, "confusion_matrix.csv"),
        _confusion_rows(result["test"], class_ids),
    )

    structure = class_structure_diagnostics(
        model, class_ids, target_names=dataset.target_names
    )

    decision = structural_decision_diagnostics(model, class_ids)

    structure_path = save_json(
        os.path.join(phase_dir, "class_structure_report.json"),
        {"phase": 0, "class_structure": structure, "decision_diagnostics": decision},
    )

    visualizations: Dict[str, Any] = {}

    if bool(args.save_visualizations):

        visualizations = generate_phase_qualitative_maps(
            model=model,
            dataset=dataset,
            phase=0,
            output_dir=os.path.join(phase_dir, "figures", "qualitative"),
            device=args.device,
            batch_size=args.eval_batch_size,
            cmap_name=args.visualization_cmap,
            dpi=args.visualization_dpi,
        )

    report = {
        "phase": 0,
        "seen_class_ids": class_ids,
        "epochs": int(result["final_epoch"]),
        "train": _metric_summary(result["train"]),
        "validation": _metric_summary(result["validation"]),
        "test": _metric_summary(result["test"]),
        "class_structure": structure,
        "decision": decision,
        "visualizations": visualizations,
        "feature_overlap_audit": result.get(
            "feature_overlap_audit", {"status": "disabled"}
        ),
    }

    if bool(args.audit_enabled):
        save_json(
            os.path.join(phase_dir, "audit_report.json"),
            {
                "phase": 0,
                "read_only": True,
                "split": "train",
                "feature_overlap": result.get("feature_overlap_audit", {}),
            },
        )
    report_path = save_json(os.path.join(phase_dir, "phase_report.json"), report)

    print("\nPhase 0 finalized.")
    _print_metrics("Train", result["train"])
    _print_metrics("Validation", result["validation"])
    _print_metrics("Test", result["test"])
    print(f"Checkpoint       : {result['checkpoint']}")
    print(f"Phase report     : {report_path}")
    print(f"Structure report : {structure_path}")
    print(f"Training history : {history_path}")
    print(f"Classwise metrics: {classwise_path}")
    print(f"Confusion matrix : {confusion_path}")

    return report


def _report_incremental_phase(
    *,
    args: argparse.Namespace,
    dataset: IncrementalHSIDatasetManager,
    model: NECILModel,
    result: Mapping[str, Any],
) -> Dict[str, Any]:
    phase = int(result["phase"])
    old_ids = [int(value) for value in result["old_class_ids"]]
    new_ids = [int(value) for value in result["new_class_ids"]]
    seen_ids = [int(value) for value in result["seen_class_ids"]]
    phase_dir = os.path.join(args.save_dir, f"phase_{phase}")
    os.makedirs(phase_dir, exist_ok=True)

    method = dict(result["method_state"])

    structure = class_structure_diagnostics(
        model, seen_ids, target_names=dataset.target_names
    )

    decision = structural_decision_diagnostics(model, seen_ids)

    cross_phase = cross_phase_cds_decision_diagnostics(
        model, old_class_ids=old_ids, new_class_ids=new_ids
    )

    history_path = save_csv(
        os.path.join(phase_dir, "training_history.csv"),
        _history_rows(result["history"]),
    )

    classwise_path = save_csv(
        os.path.join(phase_dir, "classwise_metrics.csv"),
        _classwise_rows(
            metrics=result["cumulative_test"],
            class_ids=seen_ids,
            class_names=dataset.target_names,
            old_ids=old_ids,
        ),
    )

    confusion_path = save_csv(
        os.path.join(phase_dir, "confusion_matrix.csv"),
        _confusion_rows(result["cumulative_test"], seen_ids),
    )

    structure_path = save_json(
        os.path.join(phase_dir, "class_structure_report.json"),
        {
            "phase": phase,
            "class_structure": structure,
            "decision_diagnostics": decision,
            "cross_phase_decision": cross_phase,
            "method_state": method,
            "phase_origin_confusion": result.get("phase_origin_confusion"),
        },
    )

    visualizations: Dict[str, Any] = {}

    if bool(args.save_visualizations):

        visualizations = generate_phase_qualitative_maps(
            model=model,
            dataset=dataset,
            phase=phase,
            output_dir=os.path.join(phase_dir, "figures", "qualitative"),
            device=args.device,
            batch_size=args.eval_batch_size,
            cmap_name=args.visualization_cmap,
            dpi=args.visualization_dpi,
        )

    old_group = result["old_new"]["old"]

    new_group = result["old_new"]["new"]

    report = {
        "phase": phase,
        "old_class_ids": old_ids,
        "new_class_ids": new_ids,
        "seen_class_ids": seen_ids,
        "epochs": int(result["final_epoch"]),
        "cumulative_test": _metric_summary(result["cumulative_test"]),
        "old_ba": float(old_group["balanced_accuracy"]),
        "new_ba": float(new_group["balanced_accuracy"]),
        "harmonic_old_new_accuracy": float(
            result["old_new"]["harmonic_old_new_accuracy"]
        ),
        "forgetting": float(result["forgetting"]),
        "method_state": method,
        "class_structure": structure,
        "decision": decision,
        "cross_phase_decision": cross_phase,
        "phase_origin_confusion": result.get("phase_origin_confusion"),
        "visualizations": visualizations,
        "integrated_audit": method.get("integrated_audit", {"status": "disabled"}),
    }

    if bool(args.audit_enabled):
        save_json(
            os.path.join(phase_dir, "audit_report.json"),
            {
                "phase": phase,
                "read_only": True,
                "integrated_audit": method.get("integrated_audit", {}),
            },
        )
    report_path = save_json(os.path.join(phase_dir, "phase_report.json"), report)

    print(f"\nPhase {phase} finalized.")

    _print_metrics("Cumulative", result["cumulative_test"])

    print(
        f"OldBA={100*report['old_ba']:.2f}% | NewBA={100*report['new_ba']:.2f}% | "
        f"H={100*report['harmonic_old_new_accuracy']:.2f}% | "
        f"F={100*report['forgetting']:.2f}%"
    )

    print(
        "Joint incremental training | "
        f"Epochs={int(method['representation_epochs'])} "
        f"JointObj={method['final_joint_classification_loss']:.4f} "
        f"CEreal={method['final_real_current_classification_loss']:.4f} "
        f"SepMargin={method['final_cds_separation_margin']:.4f} "
        f"SepViol={100*method['final_cds_separation_violation_fraction']:.2f}% "
        f"Pres={method['final_cds_preservation_loss']:.4f} "
        f"[Intra={method['final_cds_intra_loss']:.4f},Inter={method['final_cds_inter_loss']:.4f}] "
        f"FKD={method['final_feature_kd_loss']:.4f}"
    )

    print(
        "Historical contrastive negatives | "
        f"Mode={method.get('historical_negative_mode', 'classifier')} "
        f"SamplesPerClass={method.get('historical_negative_samples_per_class', 0)} "
        f"Aggregation={method.get('historical_negative_aggregation', 'single_direction')} "
        "| classifier supervision remains directional"
    )

    print(
        "Whole-CDS orthogonal-frame diagnostics | "
        f"RotationApplied={method['phase_transport_accepted_on_new_class_holdout']} "
        f"Frame={method.get('historical_frame_source', 'legacy')} "
        f"MapErr={method['phase_mapping_error']:.4f} "
        f"Drift={method['representation_drift_rms']:.4f} "
        f"ScaleDev={method['phase_scale_deviation_rms']:.4f} "
        f"Translation={method['phase_translation_rms']:.4f}"
    )
    head_history = method.get("historical_support_reconstruction") or []
    selected_head_epoch = int(method.get("selected_support_head_epoch", 0))
    if head_history:
        selected_rows = [
            row
            for row in head_history
            if int(row.get("epoch", -1)) == selected_head_epoch
        ]
        if len(selected_rows) != 1:
            raise RuntimeError(
                "selected head epoch missing or duplicated in reconstruction history"
            )
        selected = selected_rows[0]
        old_risk = float(selected["historical_support_loss"])
        new_ce = float(selected["selection_new_ce"])
        combined = float(selected["selection_objective"])
        if (
            bool(method.get("support_reconstruction_active"))
            and selected_head_epoch > 0
        ):
            summary = (
                f"same-head final epoch {selected_head_epoch} retained (audit only)"
            )
        elif bool(method.get("support_reconstruction_active")):
            summary = "same-head training configured; no optimizer epoch completed"
        else:
            summary = "no post-hoc training; final-epoch Q+W retained"
        print(
            "Joint Whole-CDS cosine classifier | "
            f"{summary} | "
            f"audit_source={selected.get('selection_source', 'unknown')} "
            f"OldRisk={old_risk:.4f} "
            f"NewCE={new_ce:.4f} "
            f"JointAudit={combined:.4f}"
        )
    else:
        print(
            "Direct CDS classifier state | "
            f"JointLoss={method['final_direct_cds_classifier_loss']:.4f} "
            f"HistoricalCDS={method['final_direct_cds_historical_loss']:.4f} "
            f"CurrentReal={method['final_direct_cds_current_loss']:.4f}"
        )
    print(
        "Read-only CDS classifier diagnostic | "
        f"OldNewViol={100*method['final_whole_cds_old_new_violation_fraction']:.2f}%"
    )

    print(
        "Covariance-aware learned/native spectral pair risk | "
        f"Learned={method['classifier_learned_pair_risk_mean']:.4f} "
        f"Physical={method['classifier_physical_pair_risk_mean']:.4f} "
        f"Effective={method['classifier_effective_pair_risk_mean']:.4f}"
    )

    print(
        "Classifier movement during joint training | "
        f"OldMax={method['historical_classifier_training_max_change']:.3e} "
        f"OldDir={method['historical_classifier_training_direction_change']:.3e} "
        f"NewMax={method['incoming_classifier_training_max_change']:.3e} "
        f"NewDir={method['incoming_classifier_training_direction_change']:.3e}"
    )

    # Phase-start old-versus-all is intentionally unavailable: W_new has not
    # yet been admitted in a meaningful measurement. Preserve N/A instead of
    # throwing TypeError when the trainer correctly stores None.
    start_old_all = method.get("phase_start_old_all_seen_ba")
    start_old_all_text = (
        "N/A" if start_old_all is None else f"{100 * float(start_old_all):.2f}%"
    )
    print(
        "Competition (final old/new: reporting; post-rep new: train) | "
        f"StartOldOnly/All={100*method['phase_start_old_ba']:.2f}%/"
        f"{start_old_all_text} | "
        f"PostRepOldOnly/All={100*method['post_rep_old_only_ba']:.2f}%/"
        f"{100*method['post_rep_old_all_seen_ba']:.2f}% | "
        f"FinalOldOnly/All={100*method['final_old_only_ba']:.2f}%/"
        f"{100*method['final_old_all_seen_ba']:.2f}% | "
        f"FinalNewOnly/All={100*method['final_new_only_ba']:.2f}%/"
        f"{100*method['final_new_all_seen_ba']:.2f}%"
    )

    print(f"Checkpoint       : {result['checkpoint']}")
    print(f"Phase report     : {report_path}")
    print(f"Structure report : {structure_path}")
    print(f"Training history : {history_path}")
    print(f"Classwise metrics: {classwise_path}")
    print(f"Confusion matrix : {confusion_path}")

    return report


def run_experiment(args: argparse.Namespace) -> Dict[str, Any]:

    if str(args.run_mode) in {"all", "incremental"}:
        Trainer.validate_runtime_interfaces()
    os.makedirs(args.save_dir, exist_ok=True)

    dataset, data_summary, preprocessing_state = _build_data(args)
    save_json(os.path.join(args.save_dir, "data_protocol.json"), data_summary)
    SaveHSIPreprocessor(
        os.path.join(args.save_dir, "phase_0_preprocessor.npz"), preprocessing_state
    )
    save_json(os.path.join(args.save_dir, "experiment_config.json"), vars(args))
    model = NECILModel(args)
    trainer = Trainer(model, dataset, args)
    phase_reports: Dict[int, Dict[str, Any]] = {}
    mode = str(args.run_mode)
    if mode in {"base", "all"}:
        base_result = trainer.run_base_only(
            epochs=args.epochs_base, batch_size=args.batch_size, lr=args.lr
        )
        phase_reports[0] = _report_base_phase(
            args=args, dataset=dataset, model=model, result=base_result
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
        highest_phase = int(dataset.nb_tasks) - 1
        if args.max_phase is not None:
            highest_phase = min(highest_phase, int(args.max_phase))
        for phase in range(int(next_phase), highest_phase + 1):

            result = trainer.run_incremental_phase(
                phase=phase,
                epochs=args.epochs_inc,
                batch_size=args.batch_size,
                lr=args.lr_inc,
            )
            phase_reports[phase] = _report_incremental_phase(
                args=args, dataset=dataset, model=model, result=result
            )

    finalized_phases = [int(value) for value in dataset.finalized_phases]

    if not finalized_phases or finalized_phases != list(range(len(finalized_phases))):

        raise RuntimeError("experiment did not finalize a contiguous phase prefix")

    latest = finalized_phases[-1]

    latest_metrics = trainer.evaluate_loader(
        dataset.get_reporting_dataloader(latest, batch_size=args.eval_batch_size),
        class_ids=list(model.committed_class_ids),
        target_class_ids=list(model.committed_class_ids),
    )

    run_summary = {
        "dataset": str(args.dataset),
        "seed": int(args.seed),
        "run_mode": mode,
        "finalized_phases": finalized_phases,
        "seen_class_ids": list(model.committed_class_ids),
        "latest_test": _metric_summary(latest_metrics),
        "protocol": {
            "exemplar_memory_size": 0,
            "stored_old_samples": 0,
            "stored_old_feature_rows": 0,
            "stored_synthetic_support": 0,
            "persistent_decision_sufficient_statistics": False,
            "stored_compact_class_moments": True,
            "stored_native_spectral_variances": True,
            "stored_decision_feature_rows": 0,
            "transient_synthetic_support_during_incremental_phase": False,
            "temporary_previous_backbone_reference": latest > 0,
            "temporary_previous_classifier_reference": latest > 0,
            "previous_model_reference_persisted": False,
        },
    }

    summary_path = save_json(
        os.path.join(args.save_dir, "run_summary.json"), run_summary
    )

    run_summary["run_summary"] = summary_path
    print("\nRun complete.")
    print(f"Finalized phases : {finalized_phases}")
    print(f"Seen classes     : {list(model.committed_class_ids)}")
    print(f"Run summary      : {summary_path}")
    return run_summary


def _sample_statistics(values: Sequence[float]) -> Dict[str, float]:

    array = np.asarray([float(value) for value in values], dtype=np.float64)
    if array.ndim != 1 or array.size == 0 or not np.isfinite(array).all():
        raise ValueError("aggregate values must be non-empty and finite")

    return {
        "mean": float(array.mean()),
        "std": float(array.std(ddof=1)) if array.size > 1 else 0.0,
        "min": float(array.min()),
        "max": float(array.max()),
    }


def run_multiseed_experiments(args: argparse.Namespace) -> Dict[str, Any]:
    if args.seeds is None or str(args.run_mode) not in {"base", "all"}:

        raise ValueError("multi-seed mode requires --seeds and run_mode base/all")
    seeds = [int(value) for value in args.seeds]
    root = os.path.abspath(args.save_dir)
    os.makedirs(root, exist_ok=True)
    run_rows: list[Dict[str, Any]] = []
    for seed in seeds:

        seed_args = copy.deepcopy(args)
        seed_args.seed = seed
        seed_args.seeds = None
        seed_args.resume_checkpoint = ""
        seed_args.save_dir = os.path.join(root, f"seed_{seed}")
        set_seed(seed, bool(seed_args.deterministic))
        summary = run_experiment(seed_args)
        latest = summary["latest_test"]
        row = {"seed": seed, "output_dir": seed_args.save_dir}
        for key, value in latest.items():

            row[f"final_{key}"] = value

        if str(args.run_mode) == "all" and max(summary["finalized_phases"]) > 0:

            final_phase = max(summary["finalized_phases"])

            report_path = os.path.join(
                seed_args.save_dir, f"phase_{final_phase}", "phase_report.json"
            )
            with open(report_path, "r", encoding="utf-8") as stream:
                report = json.load(stream)
            for key in ("old_ba", "new_ba", "harmonic_old_new_accuracy", "forgetting"):
                row[key] = float(report[key])

        run_rows.append(row)

    metrics = sorted(key for key in run_rows[0] if key.startswith("final_"))

    aggregate = {
        key: _sample_statistics([float(row[key]) for row in run_rows])
        for key in metrics
    }

    if str(args.run_mode) == "all" and "old_ba" in run_rows[0]:

        for key in ("old_ba", "new_ba", "harmonic_old_new_accuracy", "forgetting"):

            aggregate[key] = _sample_statistics([float(row[key]) for row in run_rows])

    runs_path = save_csv(os.path.join(root, "multiseed_runs.csv"), run_rows)

    summary = {
        "mode": f"independent_{args.run_mode}_multiseed",
        "dataset": str(args.dataset),
        "seeds": seeds,
        "protocol_seed": args.protocol_seed,
        "aggregate": aggregate,
        "runs_csv": runs_path,
    }

    summary_path = save_json(os.path.join(root, "multiseed_summary.json"), summary)
    summary["summary_json"] = summary_path
    return summary


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    validate_args(args)
    seed0 = int(args.seeds[0]) if args.seeds is not None else int(args.seed)
    set_seed(seed0, bool(args.deterministic))
    args.device = resolve_device(args.device)
    args.save_dir = os.path.abspath(args.save_dir)
    if args.seeds is not None:
        run_multiseed_experiments(args)
    else:
        run_experiment(args)


if __name__ == "__main__":
    main()


# from __future__ import annotations
# import argparse
# import copy
# import csv
# import json
# import math
# import os
# import random
# import sys
# from numbers import Integral, Real
# from typing import Any, Dict, Mapping, Optional, Sequence
# import numpy as np
# import torch

# # Preserve --help without the external dataset modules while retaining the
# # original supervised HSI preprocessing and protocol for actual experiments.
# try:
#     from data.hsi_dataloader_pytorch import (
#         ExtractLabeledPixelIndex,
#         FitHSIPreprocessorFromProtocol,
#         LoadRawHSIData,
#         SaveHSIPreprocessor,
#     )
#     from data.incremental_dataset import (
#         IncrementalHSIDatasetManager,
#         build_incremental_protocol,
#     )
# except ModuleNotFoundError as exc:
#     if not (exc.name == "data" or (exc.name or "").startswith("data.")):
#         raise

#     def _missing_data_dependency(*_args, **_kwargs):
#         raise ImportError(
#             "Real HSI training requires the original data/"
#             "hsi_dataloader_pytorch.py and data/incremental_dataset.py"
#         )

#     ExtractLabeledPixelIndex = FitHSIPreprocessorFromProtocol = _missing_data_dependency
#     LoadRawHSIData = SaveHSIPreprocessor = _missing_data_dependency
#     IncrementalHSIDatasetManager = build_incremental_protocol = _missing_data_dependency
# from models.necil_model import NECILModel
# from trainers.trainer import Trainer
# from utils.eval import (
#     class_structure_diagnostics,
#     structural_decision_diagnostics,
#     cross_phase_cds_decision_diagnostics,
#     summarize_evaluation_metrics,
# )

# try:
#     from utils.qualitative_report import generate_phase_qualitative_maps
# except ImportError:

#     def generate_phase_qualitative_maps(**_kwargs):
#         # This optional module was not in the supplied file set.
#         return {
#             "status": "qualitative reporting unavailable: include the original utils/qualitative_report.py and utils/visualize.py"
#         }


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
#         raise ValueError("class_order must be comma-separated without empty entries")

#     values = [int(part.strip()) for part in parts]

#     if len(values) != len(set(values)) or any(value < 0 for value in values):

#         raise ValueError("class_order must contain unique non-negative IDs")

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


# def save_csv(path: str, rows: Sequence[Mapping[str, Any]]) -> Optional[str]:

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

#         with open(temporary, "w", newline="", encoding="utf-8-sig") as stream:

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

#     value = _exact_int(seed, name="seed", minimum=0, maximum=(2**32) - 1)

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

#     requested = torch.device(token)

#     if requested.type == "cpu":

#         return "cpu"

#     if requested.type == "mps":

#         backend = getattr(torch.backends, "mps", None)

#         if requested.index is not None or backend is None or not backend.is_available():

#             raise RuntimeError(f"MPS device {value} is unavailable")

#         return "mps"

#     if requested.type != "cuda" or not torch.cuda.is_available():

#         raise RuntimeError(f"requested device {value!r} is unavailable")

#     index = torch.cuda.current_device() if requested.index is None else requested.index

#     if index is None or not 0 <= int(index) < torch.cuda.device_count():

#         raise RuntimeError(f"CUDA device index {index} is unavailable")

#     torch.cuda.set_device(int(index))

#     return f"cuda:{int(index)}"


# def _load_band_positions(path: str, expected_bands: int) -> Optional[list[float]]:

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


# def build_parser() -> argparse.ArgumentParser:

#     parser = argparse.ArgumentParser(
#         description=(
#             "Exemplar-free NECIL-HSI using one learned low-rank class covariance, "
#             "native-band spectral variation, historical directional margin protection, "
#             "trainable HSIMamba and one deployed all-seen cosine head."
#         ),
#         formatter_class=argparse.ArgumentDefaultsHelpFormatter,
#     )

#     experiment = parser.add_argument_group("Experiment")
#     experiment.add_argument("--dataset", default="IP")
#     experiment.add_argument("--data_dir", default="./datasets")
#     experiment.add_argument("--save_dir", default="./results_cds_hsi")
#     experiment.add_argument("--seed", type=int, default=42)
#     experiment.add_argument("--protocol_seed", type=int, default=None)
#     experiment.add_argument("--seeds", type=int, nargs="+", default=None)
#     experiment.add_argument(
#         "--device",
#         default="cuda:0",
#         help="cpu, cuda[:index], or mps (Apple Silicon)",
#     )
#     experiment.add_argument("--num_workers", type=int, default=0)
#     experiment.add_argument("--deterministic", type=str2bool, default=True)

#     experiment.add_argument(
#         "--run_mode",
#         choices=("base", "incremental", "all"),
#         default="base",
#     )
#     experiment.add_argument("--max_phase", type=int, default=None)
#     experiment.add_argument("--resume_checkpoint", default="")
#     protocol = parser.add_argument_group("Class-incremental protocol")
#     protocol.add_argument("--base_classes", type=int, default=6)
#     protocol.add_argument("--increment", type=int, default=6)
#     protocol.add_argument("--class_order", default="")
#     protocol.add_argument("--shuffle_order", type=str2bool, default=False)
#     protocol.add_argument("--train_ratio", type=float, default=0.20)
#     protocol.add_argument("--val_ratio", type=float, default=0.10)

#     protocol.add_argument(
#         "--split_strategy",
#         choices=("random_pixel", "spatial_block"),
#         default="random_pixel",
#     )
#     protocol.add_argument("--spatial_block_size", type=int, default=33)
#     protocol.add_argument("--require_patch_disjoint", type=str2bool, default=False)
#     protocol.add_argument(
#         "--require_phase_context_isolation", type=str2bool, default=None
#     )
#     preprocessing = parser.add_argument_group("HSI preprocessing")
#     preprocessing.add_argument("--patch_size", type=int, default=11)
#     preprocessing.add_argument("--no_pca", action="store_true")
#     preprocessing.add_argument("--pca_components", type=int, default=30)
#     preprocessing.add_argument("--pca_whiten", type=str2bool, default=False)
#     preprocessing.add_argument("--band_positions_file", default="")
#     ##### Ordered spectral-spatial backbone ######
#     backbone = parser.add_argument_group("Ordered spectral-spatial backbone")
#     backbone.add_argument("--d_model", type=int, default=128)
#     backbone.add_argument("--spectral_token_count", type=int, default=16)
#     backbone.add_argument("--d_state", type=int, default=16)
#     backbone.add_argument("--d_conv", type=int, default=4)
#     backbone.add_argument("--expand", type=int, default=2)
#     backbone.add_argument("--num_spectral_layers", type=int, default=2)
#     backbone.add_argument("--num_spatial_layers", type=int, default=2)
#     backbone.add_argument("--stem_norm_groups", type=int, default=8)
#     backbone.add_argument("--mamba_dropout", type=float, default=0.0)

#     optimization = parser.add_argument_group("Optimization")
#     optimization.add_argument("--epochs_base", type=int, default=80)
#     optimization.add_argument("--epochs_inc", type=int, default=35)
#     optimization.add_argument("--batch_size", type=int, default=64)
#     optimization.add_argument("--eval_batch_size", type=int, default=256)
#     optimization.add_argument("--lr", type=float, default=1e-4)
#     optimization.add_argument("--lr_inc", type=float, default=1e-4)
#     optimization.add_argument("--weight_decay", type=float, default=1e-4)
#     optimization.add_argument("--gradient_clip", type=float, default=5.0)
#     #### NECIL HSI Method ####
#     method = parser.add_argument_group("NECIL-HSI method")
#     method.add_argument(
#         "--historical_risk_mode",
#         choices=("directional",),
#         default="directional",
#         help="Unified CDS old-class margin/low-rank variance supervision",
#     )
#     method.add_argument("--contrastive_weight", type=float, default=1.0)
#     method.add_argument(
#         "--historical_negative_mode",
#         choices=("classifier", "mean", "distribution"),
#         default="distribution",
#         help="Historical contrastive negatives: deployed W, normalized moments' means, or transient compact-Gaussian Q draws",
#     )
#     method.add_argument(
#         "--historical_negative_samples_per_class",
#         type=int,
#         default=32,
#         help="Positive draw count per old class in distribution mode; mean uses 1 and classifier 0",
#     )
#     method.add_argument(
#         "--kd_logit_weight",
#         type=float,
#         default=0.0,
#         help="legacy ablation only; 0 disables logit KD in the CDS method",
#     )
#     method.add_argument("--kd_temperature", type=float, default=2.0)
#     method.add_argument(
#         "--cds_weight",
#         type=float,
#         default=1.0,
#         help="weight for historical Whole-CDS intra/inter preservation during representation training",
#     )

#     method.add_argument(
#         "--head_admission_epochs",
#         type=int,
#         default=5,
#         help="Optional incoming-row initialization on real current HSI; 0 disables",
#     )
#     method.add_argument(
#         "--head_admission_lr",
#         type=float,
#         default=1e-3,
#         help="Optimizer LR for new-class classifier-only admission",
#     )
#     method.add_argument(
#         "--historical_transport",
#         type=str2bool,
#         default=True,
#         help="Attempt held-out validated orthogonal historical Q mean/covariance transport; rejected maps are identity",
#     )

#     method.add_argument(
#         "--representation_train_new_head",
#         type=str2bool,
#         default=True,
#         help="Required true: jointly train seen cosine rows with the backbone under historical CDS protection",
#     )
#     method.add_argument(
#         "--representation_ce_scope",
#         choices=("all_seen",),
#         default="all_seen",
#     )

#     method.add_argument(
#         "--representation_select_best_val",
#         type=str2bool,
#         default=False,
#         help="Deprecated: always false. Keep final joint-training epoch; never select or restore earlier Q/W.",
#     )

#     method.add_argument(
#         "--support_head_epochs",
#         type=int,
#         default=0,
#         help="Existing same-head training epochs after final alignment; 0 commits the last frame used in representation training",
#     )
#     method.add_argument(
#         "--support_head_lr",
#         type=float,
#         default=1e-3,
#         help="Learning rate for optional fixed-Q, same-head training",
#     )

#     method.add_argument(
#         "--final_head_steps",
#         type=int,
#         default=0,
#         help="Maximum Pareto-accepted final W-only consistency steps using the finalized CDS",
#     )

#     method.add_argument(
#         "--feature_kd_weight",
#         type=float,
#         default=10.0,
#         help="ordered Feature-KD weight on paired real current HSI; retained in every incremental phase",
#     )

#     reporting = parser.add_argument_group("Reporting")

#     reporting.add_argument("--save_visualizations", type=str2bool, default=False)

#     reporting.add_argument("--visualization_cmap", default="nipy_spectral")

#     reporting.add_argument("--visualization_dpi", type=int, default=300)

#     return parser


# def validate_args(args: argparse.Namespace) -> None:
#     """Prevent silent fallback to the obsolete conditional/joint CDS method."""
#     if args.representation_select_best_val:
#         raise ValueError(
#             "Phase-1 final-epoch training is mandatory: checkpoint selection is disabled; "
#             "use --representation_select_best_val false"
#         )
#     if args.historical_risk_mode != "directional":
#         raise ValueError("unified CDS requires --historical_risk_mode directional")
#     if not args.representation_train_new_head:
#         raise ValueError("unified CDS requires --representation_train_new_head true")
#     if args.representation_ce_scope != "all_seen":
#         raise ValueError("unified CDS requires --representation_ce_scope all_seen")
#     if args.kd_logit_weight != 0:
#         raise ValueError(
#             "unified CDS uses Feature KD, not logit KD: --kd_logit_weight 0"
#         )

#     if args.historical_negative_mode not in ("classifier", "mean", "distribution"):
#         raise ValueError(
#             "historical_negative_mode must be classifier, mean or distribution"
#         )
#     _exact_int(
#         args.historical_negative_samples_per_class,
#         name="historical_negative_samples_per_class",
#         minimum=1,
#     )

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
#     _exact_int(args.support_head_epochs, name="support_head_epochs", minimum=0)
#     _finite_float(
#         args.support_head_lr, name="support_head_lr", minimum=0.0, strict_minimum=True
#     )
#     _exact_int(args.final_head_steps, name="final_head_steps", minimum=0)
#     if int(args.final_head_steps) != 0:
#         raise ValueError(
#             "final_head_steps was superseded by support_head_epochs; use --final_head_steps 0"
#         )
#     _exact_int(args.head_admission_epochs, name="head_admission_epochs", minimum=0)
#     _finite_float(
#         args.head_admission_lr,
#         name="head_admission_lr",
#         minimum=0.0,
#         strict_minimum=True,
#     )

#     if args.max_phase is not None:

#         _exact_int(args.max_phase, name="max_phase", minimum=0)

#     _exact_int(args.seed, name="seed", minimum=0, maximum=(2**32) - 1)

#     if args.protocol_seed is not None:

#         _exact_int(
#             args.protocol_seed, name="protocol_seed", minimum=0, maximum=(2**32) - 1
#         )

#     if args.seeds is not None:

#         seeds = [
#             _exact_int(value, name="seeds value", minimum=0, maximum=(2**32) - 1)
#             for value in args.seeds
#         ]

#         if len(seeds) != len(set(seeds)):

#             raise ValueError("--seeds must contain unique values")

#         if str(args.run_mode) == "incremental":

#             raise ValueError("--seeds is not supported with incremental resume")

#     if not bool(args.no_pca):

#         _exact_int(args.pca_components, name="pca_components", minimum=1)

#     if int(args.base_classes) < 2 or int(args.spectral_token_count) < 2:

#         raise ValueError("base_classes and spectral_token_count must be at least 2")

#     if int(args.patch_size) % 2 == 0:

#         raise ValueError("patch_size must be odd")

#     if (
#         int(args.stem_norm_groups) > int(args.d_model)
#         or int(args.d_model) % int(args.stem_norm_groups) != 0
#     ):

#         raise ValueError("stem_norm_groups must divide d_model")

#     train_ratio = _finite_float(
#         args.train_ratio, name="train_ratio", minimum=0.0, strict_minimum=True
#     )

#     val_ratio = _finite_float(args.val_ratio, name="val_ratio", minimum=0.0)

#     if train_ratio + val_ratio >= 1.0:

#         raise ValueError("train_ratio + val_ratio must be < 1")

#     if _finite_float(args.mamba_dropout, name="mamba_dropout", minimum=0.0) >= 1.0:

#         raise ValueError("mamba_dropout must be < 1")

#     for name in ("lr", "lr_inc"):

#         _finite_float(getattr(args, name), name=name, minimum=0.0, strict_minimum=True)

#     for name in (
#         "weight_decay",
#         "gradient_clip",
#         "contrastive_weight",
#         "kd_logit_weight",
#         "cds_weight",
#         "feature_kd_weight",
#     ):

#         _finite_float(getattr(args, name), name=name, minimum=0.0)

#     _finite_float(
#         args.kd_temperature, name="kd_temperature", minimum=0.0, strict_minimum=True
#     )

#     if str(args.run_mode) == "incremental" and not str(args.resume_checkpoint).strip():

#         raise ValueError("--resume_checkpoint is required for --run_mode incremental")

#     class_order = _parse_class_order(args.class_order)

#     if class_order is not None and bool(args.shuffle_order):

#         raise ValueError("class_order and shuffle_order are mutually exclusive")


# def _build_data(
#     args: argparse.Namespace,
# ) -> tuple[IncrementalHSIDatasetManager, Dict[str, Any], Mapping[str, Any]]:

#     raw_cube, gt, class_count, names, _, label_policy = LoadRawHSIData(
#         args.dataset, base_dir=args.data_dir
#     )

#     args.class_count = int(class_count)

#     labels, coords = ExtractLabeledPixelIndex(gt, label_policy=label_policy)

#     if int(args.base_classes) > int(class_count):

#         raise ValueError(
#             f"base_classes={args.base_classes} exceeds dataset classes={class_count}"
#         )

#     class_order = _parse_class_order(args.class_order)

#     if class_order is not None and set(class_order) != set(range(int(class_count))):
#         raise ValueError(
#             f"class_order must be a permutation of 0..{int(class_count)-1}"
#         )

#     protocol_seed = (
#         int(args.seed) if args.protocol_seed is None else int(args.protocol_seed)
#     )

#     protocol = build_incremental_protocol(
#         labels,
#         coords,
#         gt_shape=gt.shape,
#         class_count=int(class_count),
#         base_classes=int(args.base_classes),
#         increment=int(args.increment),
#         train_ratio=float(args.train_ratio),
#         val_ratio=float(args.val_ratio),
#         seed=protocol_seed,
#         shuffle_order=bool(args.shuffle_order),
#         class_order=class_order,
#         target_names=names,
#         split_strategy=str(args.split_strategy),
#         spatial_block_size=int(args.spatial_block_size),
#     )

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
#         raise ValueError("spectral_token_count cannot exceed ordered raw bands")
#     args.band_positions = _load_band_positions(
#         args.band_positions_file, args.spectral_bands
#     )
#     args.band_wavelengths = None

#     # The model fits both its raw-spectrum standardizer and ordered spectral-evidence

#     # normalization from the real Phase-0 TRAIN spectra.

#     args.spectral_normalization_mean = None

#     args.spectral_normalization_std = None

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

#     manager.preprocessing_state = preprocessing_state

#     split_counts = {
#         int(class_id): {
#             split: int(np.asarray(protocol["split_by_class"][class_id][split]).size)
#             for split in ("train", "val", "test")
#         }
#         for class_id in range(int(class_count))
#     }

#     data_summary: Dict[str, Any] = {
#         "dataset": str(args.dataset),
#         "optimization_seed": int(args.seed),
#         "protocol_seed": int(protocol_seed),
#         "raw_shape": list(raw_cube.shape),
#         "gt_shape": list(gt.shape),
#         "labeled_sample_count": int(labels.size),
#         "class_count": int(class_count),
#         "phase_schedule": {
#             int(phase): [int(value) for value in classes]
#             for phase, classes in protocol["phase_to_classes"].items()
#         },
#         "processed_bands": int(processed_cube.shape[2]),
#         "raw_spectral_bands": int(ordered_cube.shape[2]),
#         "spectral_token_count": int(args.spectral_token_count),
#         "preprocessing_fit_scope": str(preprocessing_state["fit_scope"]),
#         "preprocessing_fit_pixel_count": int(preprocessing_state["fit_pixel_count"]),
#         "validation_enabled": bool(manager.validation_enabled),
#         "split_strategy": manager.split_strategy,
#         "require_patch_disjoint": manager.require_patch_disjoint,
#         "require_phase_context_isolation": manager.require_phase_context_isolation,
#         "context_policy": manager.context_policy,
#         "split_counts_by_global_class": split_counts,
#         "stores_old_training_samples": False,
#         "stores_old_feature_rows": False,
#     }

#     return manager, data_summary, preprocessing_state


# def _mapping_value(mapping: Any, class_id: int) -> Any:
#     if not isinstance(mapping, Mapping):
#         return None
#     return mapping[class_id] if class_id in mapping else mapping.get(str(class_id))


# def _metric_summary(metrics: Optional[Mapping[str, Any]]) -> Optional[Dict[str, Any]]:
#     return summarize_evaluation_metrics(metrics)


# def _history_rows(history: Sequence[Mapping[str, Any]]) -> list[Dict[str, Any]]:
#     return [dict(row) for row in history]


# def _confusion_rows(
#     metrics: Mapping[str, Any], class_ids: Sequence[int]
# ) -> list[Dict[str, Any]]:
#     matrix = np.asarray(metrics["confusion_matrix"], dtype=np.int64)
#     ids = [int(value) for value in class_ids]
#     if matrix.shape != (len(ids), len(ids)):
#         raise RuntimeError("confusion matrix shape disagrees with class_ids")

#     rows = []

#     for r, class_id in enumerate(ids):

#         row: Dict[str, Any] = {
#             "true_class_id": class_id,
#             "true_display_id": class_id + 1,
#         }
#         for c, pred_id in enumerate(ids):
#             row[f"pred_{pred_id}"] = int(matrix[r, c])
#         rows.append(row)

#     return rows


# def _classwise_rows(
#     *,
#     metrics: Mapping[str, Any],
#     class_ids: Sequence[int],
#     class_names: Sequence[str],
#     old_ids: Sequence[int] = (),
# ) -> list[Dict[str, Any]]:

#     old = set(int(value) for value in old_ids)

#     rows = []

#     for class_id in [int(value) for value in class_ids]:

#         rows.append(
#             {
#                 "class_id": class_id,
#                 "display_id": class_id + 1,
#                 "class_name": str(class_names[class_id]),
#                 "phase_role": "historical" if class_id in old else "current",
#                 "test_accuracy": _mapping_value(
#                     metrics.get("per_class_accuracy"), class_id
#                 ),
#                 "test_logit_margin": _mapping_value(
#                     metrics.get("per_class_mean_logit_margin"), class_id
#                 ),
#                 "test_energy_margin": _mapping_value(
#                     metrics.get("per_class_mean_energy_margin"), class_id
#                 ),
#                 "test_raw_energy_margin": _mapping_value(
#                     metrics.get("per_class_mean_raw_energy_margin"), class_id
#                 ),
#             }
#         )

#     return rows


# def _print_metrics(name: str, metrics: Optional[Mapping[str, Any]]) -> None:

#     if metrics is None:

#         print(f"{name:<10} | disabled")

#         return

#     def number(value, *, percent=False):
#         if value is None:
#             return "N/A"
#         return f"{100*float(value):.2f}%" if percent else f"{float(value):.4f}"

#     print(
#         f"{name:<10} | OA={number(metrics['overall_accuracy'], percent=True)} | "
#         f"BA={number(metrics['balanced_accuracy'], percent=True)} | "
#         f"MinClass={number(metrics['minimum_class_accuracy'], percent=True)} | "
#         f"LogitMargin={number(metrics['mean_logit_margin'])} | "
#         f"NormEnergy={number(metrics['mean_energy_margin'])} | "
#         f"NormViol={number(metrics['energy_violation_fraction'], percent=True)} | "
#         f"RawEnergy={number(metrics.get('mean_raw_energy_margin', metrics['mean_energy_margin']))}"
#     )


# def _report_base_phase(
#     *,
#     args: argparse.Namespace,
#     dataset: IncrementalHSIDatasetManager,
#     model: NECILModel,
#     result: Mapping[str, Any],
# ) -> Dict[str, Any]:

#     class_ids = [int(value) for value in result["class_ids"]]

#     phase_dir = os.path.join(args.save_dir, "phase_0")

#     os.makedirs(phase_dir, exist_ok=True)

#     history_path = save_csv(
#         os.path.join(phase_dir, "training_history.csv"),
#         _history_rows(result["history"]),
#     )

#     classwise_path = save_csv(
#         os.path.join(phase_dir, "classwise_metrics.csv"),
#         _classwise_rows(
#             metrics=result["test"],
#             class_ids=class_ids,
#             class_names=dataset.target_names,
#         ),
#     )

#     confusion_path = save_csv(
#         os.path.join(phase_dir, "confusion_matrix.csv"),
#         _confusion_rows(result["test"], class_ids),
#     )

#     structure = class_structure_diagnostics(
#         model, class_ids, target_names=dataset.target_names
#     )

#     decision = structural_decision_diagnostics(model, class_ids)

#     structure_path = save_json(
#         os.path.join(phase_dir, "class_structure_report.json"),
#         {"phase": 0, "class_structure": structure, "decision_diagnostics": decision},
#     )

#     visualizations: Dict[str, Any] = {}

#     if bool(args.save_visualizations):

#         visualizations = generate_phase_qualitative_maps(
#             model=model,
#             dataset=dataset,
#             phase=0,
#             output_dir=os.path.join(phase_dir, "figures", "qualitative"),
#             device=args.device,
#             batch_size=args.eval_batch_size,
#             cmap_name=args.visualization_cmap,
#             dpi=args.visualization_dpi,
#         )

#     report = {
#         "phase": 0,
#         "seen_class_ids": class_ids,
#         "epochs": int(result["final_epoch"]),
#         "train": _metric_summary(result["train"]),
#         "validation": _metric_summary(result["validation"]),
#         "test": _metric_summary(result["test"]),
#         "class_structure": structure,
#         "decision": decision,
#         "visualizations": visualizations,
#     }

#     report_path = save_json(os.path.join(phase_dir, "phase_report.json"), report)

#     print("\nPhase 0 finalized.")
#     _print_metrics("Train", result["train"])
#     _print_metrics("Validation", result["validation"])
#     _print_metrics("Test", result["test"])
#     print(f"Checkpoint       : {result['checkpoint']}")
#     print(f"Phase report     : {report_path}")
#     print(f"Structure report : {structure_path}")
#     print(f"Training history : {history_path}")
#     print(f"Classwise metrics: {classwise_path}")
#     print(f"Confusion matrix : {confusion_path}")

#     return report


# def _report_incremental_phase(
#     *,
#     args: argparse.Namespace,
#     dataset: IncrementalHSIDatasetManager,
#     model: NECILModel,
#     result: Mapping[str, Any],
# ) -> Dict[str, Any]:
#     phase = int(result["phase"])
#     old_ids = [int(value) for value in result["old_class_ids"]]
#     new_ids = [int(value) for value in result["new_class_ids"]]
#     seen_ids = [int(value) for value in result["seen_class_ids"]]
#     phase_dir = os.path.join(args.save_dir, f"phase_{phase}")
#     os.makedirs(phase_dir, exist_ok=True)

#     method = dict(result["method_state"])

#     structure = class_structure_diagnostics(
#         model, seen_ids, target_names=dataset.target_names
#     )

#     decision = structural_decision_diagnostics(model, seen_ids)

#     cross_phase = cross_phase_cds_decision_diagnostics(
#         model, old_class_ids=old_ids, new_class_ids=new_ids
#     )

#     history_path = save_csv(
#         os.path.join(phase_dir, "training_history.csv"),
#         _history_rows(result["history"]),
#     )

#     classwise_path = save_csv(
#         os.path.join(phase_dir, "classwise_metrics.csv"),
#         _classwise_rows(
#             metrics=result["cumulative_test"],
#             class_ids=seen_ids,
#             class_names=dataset.target_names,
#             old_ids=old_ids,
#         ),
#     )

#     confusion_path = save_csv(
#         os.path.join(phase_dir, "confusion_matrix.csv"),
#         _confusion_rows(result["cumulative_test"], seen_ids),
#     )

#     structure_path = save_json(
#         os.path.join(phase_dir, "class_structure_report.json"),
#         {
#             "phase": phase,
#             "class_structure": structure,
#             "decision_diagnostics": decision,
#             "cross_phase_decision": cross_phase,
#             "method_state": method,
#             "phase_origin_confusion": result.get("phase_origin_confusion"),
#         },
#     )

#     visualizations: Dict[str, Any] = {}

#     if bool(args.save_visualizations):

#         visualizations = generate_phase_qualitative_maps(
#             model=model,
#             dataset=dataset,
#             phase=phase,
#             output_dir=os.path.join(phase_dir, "figures", "qualitative"),
#             device=args.device,
#             batch_size=args.eval_batch_size,
#             cmap_name=args.visualization_cmap,
#             dpi=args.visualization_dpi,
#         )

#     old_group = result["old_new"]["old"]

#     new_group = result["old_new"]["new"]

#     report = {
#         "phase": phase,
#         "old_class_ids": old_ids,
#         "new_class_ids": new_ids,
#         "seen_class_ids": seen_ids,
#         "epochs": int(result["final_epoch"]),
#         "cumulative_test": _metric_summary(result["cumulative_test"]),
#         "old_ba": float(old_group["balanced_accuracy"]),
#         "new_ba": float(new_group["balanced_accuracy"]),
#         "harmonic_old_new_accuracy": float(
#             result["old_new"]["harmonic_old_new_accuracy"]
#         ),
#         "forgetting": float(result["forgetting"]),
#         "method_state": method,
#         "class_structure": structure,
#         "decision": decision,
#         "cross_phase_decision": cross_phase,
#         "phase_origin_confusion": result.get("phase_origin_confusion"),
#         "visualizations": visualizations,
#     }

#     report_path = save_json(os.path.join(phase_dir, "phase_report.json"), report)

#     print(f"\nPhase {phase} finalized.")

#     _print_metrics("Cumulative", result["cumulative_test"])

#     print(
#         f"OldBA={100*report['old_ba']:.2f}% | NewBA={100*report['new_ba']:.2f}% | "
#         f"H={100*report['harmonic_old_new_accuracy']:.2f}% | "
#         f"F={100*report['forgetting']:.2f}%"
#     )

#     print(
#         "Joint incremental training | "
#         f"Epochs={int(method['representation_epochs'])} "
#         f"JointObj={method['final_joint_classification_loss']:.4f} "
#         f"CEreal={method['final_real_current_classification_loss']:.4f} "
#         f"SepMargin={method['final_cds_separation_margin']:.4f} "
#         f"SepViol={100*method['final_cds_separation_violation_fraction']:.2f}% "
#         f"Pres={method['final_cds_preservation_loss']:.4f} "
#         f"[Intra={method['final_cds_intra_loss']:.4f},Inter={method['final_cds_inter_loss']:.4f}] "
#         f"FKD={method['final_feature_kd_loss']:.4f}"
#     )

#     print(
#         "Historical contrastive negatives | "
#         f"Mode={method.get('historical_negative_mode', 'classifier')} "
#         f"SamplesPerClass={method.get('historical_negative_samples_per_class', 0)} "
#         f"Aggregation={method.get('historical_negative_aggregation', 'single_direction')} "
#         "| classifier supervision remains directional"
#     )

#     print(
#         "Whole-CDS orthogonal-frame diagnostics | "
#         f"RotationApplied={method['phase_transport_accepted_on_new_class_holdout']} "
#         f"Frame={method.get('historical_frame_source', 'legacy')} "
#         f"MapErr={method['phase_mapping_error']:.4f} "
#         f"Drift={method['representation_drift_rms']:.4f} "
#         f"ScaleDev={method['phase_scale_deviation_rms']:.4f} "
#         f"Translation={method['phase_translation_rms']:.4f}"
#     )
#     head_history = method.get("historical_support_reconstruction") or []
#     selected_head_epoch = int(method.get("selected_support_head_epoch", 0))
#     if head_history:
#         selected_rows = [
#             row
#             for row in head_history
#             if int(row.get("epoch", -1)) == selected_head_epoch
#         ]
#         if len(selected_rows) != 1:
#             raise RuntimeError(
#                 "selected head epoch missing or duplicated in reconstruction history"
#             )
#         selected = selected_rows[0]
#         old_risk = float(selected["historical_support_loss"])
#         new_ce = float(selected["selection_new_ce"])
#         combined = float(selected["selection_objective"])
#         if (
#             bool(method.get("support_reconstruction_active"))
#             and selected_head_epoch > 0
#         ):
#             summary = (
#                 f"same-head final epoch {selected_head_epoch} retained (audit only)"
#             )
#         elif bool(method.get("support_reconstruction_active")):
#             summary = "same-head training configured; no optimizer epoch completed"
#         else:
#             summary = "no post-hoc training; final-epoch Q+W retained"
#         print(
#             "Joint Whole-CDS cosine classifier | "
#             f"{summary} | "
#             f"audit_source={selected.get('selection_source', 'unknown')} "
#             f"OldRisk={old_risk:.4f} "
#             f"NewCE={new_ce:.4f} "
#             f"JointAudit={combined:.4f}"
#         )
#     else:
#         print(
#             "Direct CDS classifier state | "
#             f"JointLoss={method['final_direct_cds_classifier_loss']:.4f} "
#             f"HistoricalCDS={method['final_direct_cds_historical_loss']:.4f} "
#             f"CurrentReal={method['final_direct_cds_current_loss']:.4f}"
#         )
#     print(
#         "Read-only CDS classifier diagnostic | "
#         f"OldNewViol={100*method['final_whole_cds_old_new_violation_fraction']:.2f}%"
#     )

#     print(
#         "Covariance-aware learned/native spectral pair risk | "
#         f"Learned={method['classifier_learned_pair_risk_mean']:.4f} "
#         f"Physical={method['classifier_physical_pair_risk_mean']:.4f} "
#         f"Effective={method['classifier_effective_pair_risk_mean']:.4f}"
#     )

#     print(
#         "Classifier movement during joint training | "
#         f"OldMax={method['historical_classifier_training_max_change']:.3e} "
#         f"OldDir={method['historical_classifier_training_direction_change']:.3e} "
#         f"NewMax={method['incoming_classifier_training_max_change']:.3e} "
#         f"NewDir={method['incoming_classifier_training_direction_change']:.3e}"
#     )

#     # Phase-start old-versus-all is intentionally unavailable: W_new has not
#     # yet been admitted in a meaningful measurement. Preserve N/A instead of
#     # throwing TypeError when the trainer correctly stores None.
#     start_old_all = method.get("phase_start_old_all_seen_ba")
#     start_old_all_text = (
#         "N/A" if start_old_all is None else f"{100 * float(start_old_all):.2f}%"
#     )
#     print(
#         "Competition (final old/new: reporting; post-rep new: train) | "
#         f"StartOldOnly/All={100*method['phase_start_old_ba']:.2f}%/"
#         f"{start_old_all_text} | "
#         f"PostRepOldOnly/All={100*method['post_rep_old_only_ba']:.2f}%/"
#         f"{100*method['post_rep_old_all_seen_ba']:.2f}% | "
#         f"FinalOldOnly/All={100*method['final_old_only_ba']:.2f}%/"
#         f"{100*method['final_old_all_seen_ba']:.2f}% | "
#         f"FinalNewOnly/All={100*method['final_new_only_ba']:.2f}%/"
#         f"{100*method['final_new_all_seen_ba']:.2f}%"
#     )

#     print(f"Checkpoint       : {result['checkpoint']}")
#     print(f"Phase report     : {report_path}")
#     print(f"Structure report : {structure_path}")
#     print(f"Training history : {history_path}")
#     print(f"Classwise metrics: {classwise_path}")
#     print(f"Confusion matrix : {confusion_path}")

#     return report


# def run_experiment(args: argparse.Namespace) -> Dict[str, Any]:

#     if str(args.run_mode) in {"all", "incremental"}:
#         Trainer.validate_runtime_interfaces()
#     os.makedirs(args.save_dir, exist_ok=True)

#     dataset, data_summary, preprocessing_state = _build_data(args)
#     save_json(os.path.join(args.save_dir, "data_protocol.json"), data_summary)
#     SaveHSIPreprocessor(
#         os.path.join(args.save_dir, "phase_0_preprocessor.npz"), preprocessing_state
#     )
#     save_json(os.path.join(args.save_dir, "experiment_config.json"), vars(args))
#     model = NECILModel(args)
#     trainer = Trainer(model, dataset, args)
#     phase_reports: Dict[int, Dict[str, Any]] = {}
#     mode = str(args.run_mode)
#     if mode in {"base", "all"}:
#         base_result = trainer.run_base_only(
#             epochs=args.epochs_base, batch_size=args.batch_size, lr=args.lr
#         )
#         phase_reports[0] = _report_base_phase(
#             args=args, dataset=dataset, model=model, result=base_result
#         )
#         next_phase: Optional[int] = 1

#     else:

#         restored = trainer.load_checkpoint(args.resume_checkpoint)
#         next_phase = restored["next_phase"]
#         if next_phase is None:
#             raise RuntimeError("checkpoint already contains the final protocol phase")
#     if mode in {"incremental", "all"}:
#         if next_phase is None:
#             raise RuntimeError("no incremental phase remains")
#         highest_phase = int(dataset.nb_tasks) - 1
#         if args.max_phase is not None:
#             highest_phase = min(highest_phase, int(args.max_phase))
#         for phase in range(int(next_phase), highest_phase + 1):

#             result = trainer.run_incremental_phase(
#                 phase=phase,
#                 epochs=args.epochs_inc,
#                 batch_size=args.batch_size,
#                 lr=args.lr_inc,
#             )
#             phase_reports[phase] = _report_incremental_phase(
#                 args=args, dataset=dataset, model=model, result=result
#             )

#     finalized_phases = [int(value) for value in dataset.finalized_phases]

#     if not finalized_phases or finalized_phases != list(range(len(finalized_phases))):

#         raise RuntimeError("experiment did not finalize a contiguous phase prefix")

#     latest = finalized_phases[-1]

#     latest_metrics = trainer.evaluate_loader(
#         dataset.get_reporting_dataloader(latest, batch_size=args.eval_batch_size),
#         class_ids=list(model.committed_class_ids),
#         target_class_ids=list(model.committed_class_ids),
#     )

#     run_summary = {
#         "dataset": str(args.dataset),
#         "seed": int(args.seed),
#         "run_mode": mode,
#         "finalized_phases": finalized_phases,
#         "seen_class_ids": list(model.committed_class_ids),
#         "latest_test": _metric_summary(latest_metrics),
#         "protocol": {
#             "exemplar_memory_size": 0,
#             "stored_old_samples": 0,
#             "stored_old_feature_rows": 0,
#             "stored_synthetic_support": 0,
#             "persistent_decision_sufficient_statistics": False,
#             "stored_compact_class_moments": True,
#             "stored_native_spectral_variances": True,
#             "stored_decision_feature_rows": 0,
#             "transient_synthetic_support_during_incremental_phase": False,
#             "temporary_previous_backbone_reference": latest > 0,
#             "temporary_previous_classifier_reference": latest > 0,
#             "previous_model_reference_persisted": False,
#         },
#     }

#     summary_path = save_json(
#         os.path.join(args.save_dir, "run_summary.json"), run_summary
#     )

#     run_summary["run_summary"] = summary_path
#     print("\nRun complete.")
#     print(f"Finalized phases : {finalized_phases}")
#     print(f"Seen classes     : {list(model.committed_class_ids)}")
#     print(f"Run summary      : {summary_path}")
#     return run_summary


# def _sample_statistics(values: Sequence[float]) -> Dict[str, float]:

#     array = np.asarray([float(value) for value in values], dtype=np.float64)
#     if array.ndim != 1 or array.size == 0 or not np.isfinite(array).all():
#         raise ValueError("aggregate values must be non-empty and finite")

#     return {
#         "mean": float(array.mean()),
#         "std": float(array.std(ddof=1)) if array.size > 1 else 0.0,
#         "min": float(array.min()),
#         "max": float(array.max()),
#     }


# def run_multiseed_experiments(args: argparse.Namespace) -> Dict[str, Any]:
#     if args.seeds is None or str(args.run_mode) not in {"base", "all"}:

#         raise ValueError("multi-seed mode requires --seeds and run_mode base/all")
#     seeds = [int(value) for value in args.seeds]
#     root = os.path.abspath(args.save_dir)
#     os.makedirs(root, exist_ok=True)
#     run_rows: list[Dict[str, Any]] = []
#     for seed in seeds:

#         seed_args = copy.deepcopy(args)
#         seed_args.seed = seed
#         seed_args.seeds = None
#         seed_args.resume_checkpoint = ""
#         seed_args.save_dir = os.path.join(root, f"seed_{seed}")
#         set_seed(seed, bool(seed_args.deterministic))
#         summary = run_experiment(seed_args)
#         latest = summary["latest_test"]
#         row = {"seed": seed, "output_dir": seed_args.save_dir}
#         for key, value in latest.items():

#             row[f"final_{key}"] = value

#         if str(args.run_mode) == "all" and max(summary["finalized_phases"]) > 0:

#             final_phase = max(summary["finalized_phases"])

#             report_path = os.path.join(
#                 seed_args.save_dir, f"phase_{final_phase}", "phase_report.json"
#             )
#             with open(report_path, "r", encoding="utf-8") as stream:
#                 report = json.load(stream)
#             for key in ("old_ba", "new_ba", "harmonic_old_new_accuracy", "forgetting"):
#                 row[key] = float(report[key])

#         run_rows.append(row)

#     metrics = sorted(key for key in run_rows[0] if key.startswith("final_"))

#     aggregate = {
#         key: _sample_statistics([float(row[key]) for row in run_rows])
#         for key in metrics
#     }

#     if str(args.run_mode) == "all" and "old_ba" in run_rows[0]:

#         for key in ("old_ba", "new_ba", "harmonic_old_new_accuracy", "forgetting"):

#             aggregate[key] = _sample_statistics([float(row[key]) for row in run_rows])

#     runs_path = save_csv(os.path.join(root, "multiseed_runs.csv"), run_rows)

#     summary = {
#         "mode": f"independent_{args.run_mode}_multiseed",
#         "dataset": str(args.dataset),
#         "seeds": seeds,
#         "protocol_seed": args.protocol_seed,
#         "aggregate": aggregate,
#         "runs_csv": runs_path,
#     }

#     summary_path = save_json(os.path.join(root, "multiseed_summary.json"), summary)
#     summary["summary_json"] = summary_path
#     return summary


# def main(argv: Optional[Sequence[str]] = None) -> None:
#     args = build_parser().parse_args(argv)
#     validate_args(args)
#     seed0 = int(args.seeds[0]) if args.seeds is not None else int(args.seed)
#     set_seed(seed0, bool(args.deterministic))
#     args.device = resolve_device(args.device)
#     args.save_dir = os.path.abspath(args.save_dir)
#     if args.seeds is not None:
#         run_multiseed_experiments(args)
#     else:
#         run_experiment(args)


# if __name__ == "__main__":
#     main()

# from __future__ import annotations
# import argparse
# import copy
# import csv
# import json
# import math
# import os
# import random
# import sys
# from numbers import Integral, Real
# from typing import Any, Dict, Mapping, Optional, Sequence
# import numpy as np
# import torch

# # Preserve --help without the external dataset modules while retaining the
# # original supervised HSI preprocessing and protocol for actual experiments.
# try:
#     from data.hsi_dataloader_pytorch import (
#         ExtractLabeledPixelIndex,
#         FitHSIPreprocessorFromProtocol,
#         LoadRawHSIData,
#         SaveHSIPreprocessor,
#     )
#     from data.incremental_dataset import (
#         IncrementalHSIDatasetManager,
#         build_incremental_protocol,
#     )
# except ModuleNotFoundError as exc:
#     if not (exc.name == "data" or (exc.name or "").startswith("data.")):
#         raise

#     def _missing_data_dependency(*_args, **_kwargs):
#         raise ImportError(
#             "Real HSI training requires the original data/"
#             "hsi_dataloader_pytorch.py and data/incremental_dataset.py"
#         )

#     ExtractLabeledPixelIndex = FitHSIPreprocessorFromProtocol = _missing_data_dependency
#     LoadRawHSIData = SaveHSIPreprocessor = _missing_data_dependency
#     IncrementalHSIDatasetManager = build_incremental_protocol = _missing_data_dependency
# from models.necil_model import NECILModel
# from trainers.trainer import Trainer
# from utils.eval import (
#     class_structure_diagnostics,
#     structural_decision_diagnostics,
#     cross_phase_cds_decision_diagnostics,
#     summarize_evaluation_metrics,
# )

# try:
#     from utils.qualitative_report import generate_phase_qualitative_maps
# except ImportError:

#     def generate_phase_qualitative_maps(**_kwargs):
#         # This optional module was not in the supplied file set.
#         return {
#             "status": "qualitative reporting unavailable: include the original utils/qualitative_report.py and utils/visualize.py"
#         }


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
#         raise ValueError("class_order must be comma-separated without empty entries")

#     values = [int(part.strip()) for part in parts]

#     if len(values) != len(set(values)) or any(value < 0 for value in values):

#         raise ValueError("class_order must contain unique non-negative IDs")

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


# def save_csv(path: str, rows: Sequence[Mapping[str, Any]]) -> Optional[str]:

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

#         with open(temporary, "w", newline="", encoding="utf-8-sig") as stream:

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

#     value = _exact_int(seed, name="seed", minimum=0, maximum=(2**32) - 1)

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

#     requested = torch.device(token)

#     if requested.type == "cpu":

#         return "cpu"

#     if requested.type == "mps":

#         backend = getattr(torch.backends, "mps", None)

#         if requested.index is not None or backend is None or not backend.is_available():

#             raise RuntimeError(f"MPS device {value} is unavailable")

#         return "mps"

#     if requested.type != "cuda" or not torch.cuda.is_available():

#         raise RuntimeError(f"requested device {value!r} is unavailable")

#     index = torch.cuda.current_device() if requested.index is None else requested.index

#     if index is None or not 0 <= int(index) < torch.cuda.device_count():

#         raise RuntimeError(f"CUDA device index {index} is unavailable")

#     torch.cuda.set_device(int(index))

#     return f"cuda:{int(index)}"


# def _load_band_positions(path: str, expected_bands: int) -> Optional[list[float]]:

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


# def build_parser() -> argparse.ArgumentParser:

#     parser = argparse.ArgumentParser(
#         description=(
#             "Exemplar-free NECIL-HSI using one learned low-rank class covariance, "
#             "native-band spectral variation, historical directional margin protection, "
#             "trainable HSIMamba and one deployed all-seen cosine head."
#         ),
#         formatter_class=argparse.ArgumentDefaultsHelpFormatter,
#     )

#     experiment = parser.add_argument_group("Experiment")
#     experiment.add_argument("--dataset", default="IP")
#     experiment.add_argument("--data_dir", default="./datasets")
#     experiment.add_argument("--save_dir", default="./results_cds_hsi")
#     experiment.add_argument("--seed", type=int, default=42)
#     experiment.add_argument("--protocol_seed", type=int, default=None)
#     experiment.add_argument("--seeds", type=int, nargs="+", default=None)
#     experiment.add_argument(
#         "--device",
#         default="cuda:0",
#         help="cpu, cuda[:index], or mps (Apple Silicon)",
#     )
#     experiment.add_argument("--num_workers", type=int, default=0)
#     experiment.add_argument("--deterministic", type=str2bool, default=True)

#     experiment.add_argument(
#         "--run_mode",
#         choices=("base", "incremental", "all"),
#         default="base",
#     )
#     experiment.add_argument("--max_phase", type=int, default=None)
#     experiment.add_argument("--resume_checkpoint", default="")
#     protocol = parser.add_argument_group("Class-incremental protocol")
#     protocol.add_argument("--base_classes", type=int, default=6)
#     protocol.add_argument("--increment", type=int, default=6)
#     protocol.add_argument("--class_order", default="")
#     protocol.add_argument("--shuffle_order", type=str2bool, default=False)
#     protocol.add_argument("--train_ratio", type=float, default=0.20)
#     protocol.add_argument("--val_ratio", type=float, default=0.10)

#     protocol.add_argument(
#         "--split_strategy",
#         choices=("random_pixel", "spatial_block"),
#         default="random_pixel",
#     )
#     protocol.add_argument("--spatial_block_size", type=int, default=33)
#     protocol.add_argument("--require_patch_disjoint", type=str2bool, default=False)
#     protocol.add_argument(
#         "--require_phase_context_isolation", type=str2bool, default=None
#     )
#     preprocessing = parser.add_argument_group("HSI preprocessing")
#     preprocessing.add_argument("--patch_size", type=int, default=11)
#     preprocessing.add_argument("--no_pca", action="store_true")
#     preprocessing.add_argument("--pca_components", type=int, default=30)
#     preprocessing.add_argument("--pca_whiten", type=str2bool, default=False)
#     preprocessing.add_argument("--band_positions_file", default="")
#     ##### Ordered spectral-spatial backbone ######
#     backbone = parser.add_argument_group("Ordered spectral-spatial backbone")
#     backbone.add_argument("--d_model", type=int, default=128)
#     backbone.add_argument("--spectral_token_count", type=int, default=16)
#     backbone.add_argument("--d_state", type=int, default=16)
#     backbone.add_argument("--d_conv", type=int, default=4)
#     backbone.add_argument("--expand", type=int, default=2)
#     backbone.add_argument("--num_spectral_layers", type=int, default=2)
#     backbone.add_argument("--num_spatial_layers", type=int, default=2)
#     backbone.add_argument("--stem_norm_groups", type=int, default=8)
#     backbone.add_argument("--mamba_dropout", type=float, default=0.0)

#     optimization = parser.add_argument_group("Optimization")
#     optimization.add_argument("--epochs_base", type=int, default=80)
#     optimization.add_argument("--epochs_inc", type=int, default=35)
#     optimization.add_argument("--batch_size", type=int, default=64)
#     optimization.add_argument("--eval_batch_size", type=int, default=256)
#     optimization.add_argument("--lr", type=float, default=1e-4)
#     optimization.add_argument("--lr_inc", type=float, default=1e-4)
#     optimization.add_argument("--weight_decay", type=float, default=1e-4)
#     optimization.add_argument("--gradient_clip", type=float, default=5.0)
#     #### NECIL HSI Method ####
#     method = parser.add_argument_group("NECIL-HSI method")
#     method.add_argument(
#         "--historical_risk_mode",
#         choices=("directional",),
#         default="directional",
#         help="Unified CDS old-class margin/low-rank variance supervision",
#     )
#     method.add_argument("--contrastive_weight", type=float, default=1.0)
#     method.add_argument(
#         "--kd_logit_weight",
#         type=float,
#         default=0.0,
#         help="legacy ablation only; 0 disables logit KD in the CDS method",
#     )
#     method.add_argument("--kd_temperature", type=float, default=2.0)
#     method.add_argument(
#         "--cds_weight",
#         type=float,
#         default=1.0,
#         help="weight for historical Whole-CDS intra/inter preservation during representation training",
#     )

#     method.add_argument(
#         "--head_admission_epochs",
#         type=int,
#         default=5,
#         help="Optional incoming-row initialization on real current HSI; 0 disables",
#     )
#     method.add_argument(
#         "--head_admission_lr",
#         type=float,
#         default=1e-3,
#         help="Optimizer LR for new-class classifier-only admission",
#     )
#     method.add_argument(
#         "--historical_transport",
#         type=str2bool,
#         default=True,
#         help="Attempt held-out validated orthogonal historical Q mean/covariance transport; rejected maps are identity",
#     )

#     method.add_argument(
#         "--representation_train_new_head",
#         type=str2bool,
#         default=True,
#         help="Required true: jointly train seen cosine rows with the backbone under historical CDS protection",
#     )
#     method.add_argument(
#         "--representation_ce_scope",
#         choices=("all_seen",),
#         default="all_seen",
#     )

#     method.add_argument(
#         "--representation_select_best_val",
#         type=str2bool,
#         default=False,
#         help="Deprecated: always false. Keep final joint-training epoch; never select or restore earlier Q/W.",
#     )

#     method.add_argument(
#         "--support_head_epochs",
#         type=int,
#         default=0,
#         help="Existing same-head training epochs after final alignment; 0 commits the last frame used in representation training",
#     )
#     method.add_argument(
#         "--support_head_lr",
#         type=float,
#         default=1e-3,
#         help="Learning rate for optional fixed-Q, same-head training",
#     )

#     method.add_argument(
#         "--final_head_steps",
#         type=int,
#         default=0,
#         help="Maximum Pareto-accepted final W-only consistency steps using the finalized CDS",
#     )

#     method.add_argument(
#         "--feature_kd_weight",
#         type=float,
#         default=10.0,
#         help="ordered Feature-KD weight on paired real current HSI; retained in every incremental phase",
#     )

#     reporting = parser.add_argument_group("Reporting")

#     reporting.add_argument("--save_visualizations", type=str2bool, default=False)

#     reporting.add_argument("--visualization_cmap", default="nipy_spectral")

#     reporting.add_argument("--visualization_dpi", type=int, default=300)

#     return parser


# def validate_args(args: argparse.Namespace) -> None:
#     """Prevent silent fallback to the obsolete conditional/joint CDS method."""
#     if args.representation_select_best_val:
#         raise ValueError(
#             "Phase-1 final-epoch training is mandatory: checkpoint selection is disabled; "
#             "use --representation_select_best_val false"
#         )
#     if args.historical_risk_mode != "directional":
#         raise ValueError("unified CDS requires --historical_risk_mode directional")
#     if not args.representation_train_new_head:
#         raise ValueError("unified CDS requires --representation_train_new_head true")
#     if args.representation_ce_scope != "all_seen":
#         raise ValueError("unified CDS requires --representation_ce_scope all_seen")
#     if args.kd_logit_weight != 0:
#         raise ValueError(
#             "unified CDS uses Feature KD, not logit KD: --kd_logit_weight 0"
#         )

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
#     _exact_int(args.support_head_epochs, name="support_head_epochs", minimum=0)
#     _finite_float(
#         args.support_head_lr, name="support_head_lr", minimum=0.0, strict_minimum=True
#     )
#     _exact_int(args.final_head_steps, name="final_head_steps", minimum=0)
#     if int(args.final_head_steps) != 0:
#         raise ValueError(
#             "final_head_steps was superseded by support_head_epochs; use --final_head_steps 0"
#         )
#     _exact_int(args.head_admission_epochs, name="head_admission_epochs", minimum=0)
#     _finite_float(
#         args.head_admission_lr,
#         name="head_admission_lr",
#         minimum=0.0,
#         strict_minimum=True,
#     )

#     if args.max_phase is not None:

#         _exact_int(args.max_phase, name="max_phase", minimum=0)

#     _exact_int(args.seed, name="seed", minimum=0, maximum=(2**32) - 1)

#     if args.protocol_seed is not None:

#         _exact_int(
#             args.protocol_seed, name="protocol_seed", minimum=0, maximum=(2**32) - 1
#         )

#     if args.seeds is not None:

#         seeds = [
#             _exact_int(value, name="seeds value", minimum=0, maximum=(2**32) - 1)
#             for value in args.seeds
#         ]

#         if len(seeds) != len(set(seeds)):

#             raise ValueError("--seeds must contain unique values")

#         if str(args.run_mode) == "incremental":

#             raise ValueError("--seeds is not supported with incremental resume")

#     if not bool(args.no_pca):

#         _exact_int(args.pca_components, name="pca_components", minimum=1)

#     if int(args.base_classes) < 2 or int(args.spectral_token_count) < 2:

#         raise ValueError("base_classes and spectral_token_count must be at least 2")

#     if int(args.patch_size) % 2 == 0:

#         raise ValueError("patch_size must be odd")

#     if (
#         int(args.stem_norm_groups) > int(args.d_model)
#         or int(args.d_model) % int(args.stem_norm_groups) != 0
#     ):

#         raise ValueError("stem_norm_groups must divide d_model")

#     train_ratio = _finite_float(
#         args.train_ratio, name="train_ratio", minimum=0.0, strict_minimum=True
#     )

#     val_ratio = _finite_float(args.val_ratio, name="val_ratio", minimum=0.0)

#     if train_ratio + val_ratio >= 1.0:

#         raise ValueError("train_ratio + val_ratio must be < 1")

#     if _finite_float(args.mamba_dropout, name="mamba_dropout", minimum=0.0) >= 1.0:

#         raise ValueError("mamba_dropout must be < 1")

#     for name in ("lr", "lr_inc"):

#         _finite_float(getattr(args, name), name=name, minimum=0.0, strict_minimum=True)

#     for name in (
#         "weight_decay",
#         "gradient_clip",
#         "contrastive_weight",
#         "kd_logit_weight",
#         "cds_weight",
#         "feature_kd_weight",
#     ):

#         _finite_float(getattr(args, name), name=name, minimum=0.0)

#     _finite_float(
#         args.kd_temperature, name="kd_temperature", minimum=0.0, strict_minimum=True
#     )

#     if str(args.run_mode) == "incremental" and not str(args.resume_checkpoint).strip():

#         raise ValueError("--resume_checkpoint is required for --run_mode incremental")

#     class_order = _parse_class_order(args.class_order)

#     if class_order is not None and bool(args.shuffle_order):

#         raise ValueError("class_order and shuffle_order are mutually exclusive")


# def _build_data(
#     args: argparse.Namespace,
# ) -> tuple[IncrementalHSIDatasetManager, Dict[str, Any], Mapping[str, Any]]:

#     raw_cube, gt, class_count, names, _, label_policy = LoadRawHSIData(
#         args.dataset, base_dir=args.data_dir
#     )

#     args.class_count = int(class_count)

#     labels, coords = ExtractLabeledPixelIndex(gt, label_policy=label_policy)

#     if int(args.base_classes) > int(class_count):

#         raise ValueError(
#             f"base_classes={args.base_classes} exceeds dataset classes={class_count}"
#         )

#     class_order = _parse_class_order(args.class_order)

#     if class_order is not None and set(class_order) != set(range(int(class_count))):
#         raise ValueError(
#             f"class_order must be a permutation of 0..{int(class_count)-1}"
#         )

#     protocol_seed = (
#         int(args.seed) if args.protocol_seed is None else int(args.protocol_seed)
#     )

#     protocol = build_incremental_protocol(
#         labels,
#         coords,
#         gt_shape=gt.shape,
#         class_count=int(class_count),
#         base_classes=int(args.base_classes),
#         increment=int(args.increment),
#         train_ratio=float(args.train_ratio),
#         val_ratio=float(args.val_ratio),
#         seed=protocol_seed,
#         shuffle_order=bool(args.shuffle_order),
#         class_order=class_order,
#         target_names=names,
#         split_strategy=str(args.split_strategy),
#         spatial_block_size=int(args.spatial_block_size),
#     )

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
#         raise ValueError("spectral_token_count cannot exceed ordered raw bands")
#     args.band_positions = _load_band_positions(
#         args.band_positions_file, args.spectral_bands
#     )
#     args.band_wavelengths = None

#     # The model fits both its raw-spectrum standardizer and ordered spectral-evidence

#     # normalization from the real Phase-0 TRAIN spectra.

#     args.spectral_normalization_mean = None

#     args.spectral_normalization_std = None

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

#     manager.preprocessing_state = preprocessing_state

#     split_counts = {
#         int(class_id): {
#             split: int(np.asarray(protocol["split_by_class"][class_id][split]).size)
#             for split in ("train", "val", "test")
#         }
#         for class_id in range(int(class_count))
#     }

#     data_summary: Dict[str, Any] = {
#         "dataset": str(args.dataset),
#         "optimization_seed": int(args.seed),
#         "protocol_seed": int(protocol_seed),
#         "raw_shape": list(raw_cube.shape),
#         "gt_shape": list(gt.shape),
#         "labeled_sample_count": int(labels.size),
#         "class_count": int(class_count),
#         "phase_schedule": {
#             int(phase): [int(value) for value in classes]
#             for phase, classes in protocol["phase_to_classes"].items()
#         },
#         "processed_bands": int(processed_cube.shape[2]),
#         "raw_spectral_bands": int(ordered_cube.shape[2]),
#         "spectral_token_count": int(args.spectral_token_count),
#         "preprocessing_fit_scope": str(preprocessing_state["fit_scope"]),
#         "preprocessing_fit_pixel_count": int(preprocessing_state["fit_pixel_count"]),
#         "validation_enabled": bool(manager.validation_enabled),
#         "split_strategy": manager.split_strategy,
#         "require_patch_disjoint": manager.require_patch_disjoint,
#         "require_phase_context_isolation": manager.require_phase_context_isolation,
#         "context_policy": manager.context_policy,
#         "split_counts_by_global_class": split_counts,
#         "stores_old_training_samples": False,
#         "stores_old_feature_rows": False,
#     }

#     return manager, data_summary, preprocessing_state


# def _mapping_value(mapping: Any, class_id: int) -> Any:
#     if not isinstance(mapping, Mapping):
#         return None
#     return mapping[class_id] if class_id in mapping else mapping.get(str(class_id))


# def _metric_summary(metrics: Optional[Mapping[str, Any]]) -> Optional[Dict[str, Any]]:
#     return summarize_evaluation_metrics(metrics)


# def _history_rows(history: Sequence[Mapping[str, Any]]) -> list[Dict[str, Any]]:
#     return [dict(row) for row in history]


# def _confusion_rows(
#     metrics: Mapping[str, Any], class_ids: Sequence[int]
# ) -> list[Dict[str, Any]]:
#     matrix = np.asarray(metrics["confusion_matrix"], dtype=np.int64)
#     ids = [int(value) for value in class_ids]
#     if matrix.shape != (len(ids), len(ids)):
#         raise RuntimeError("confusion matrix shape disagrees with class_ids")

#     rows = []

#     for r, class_id in enumerate(ids):

#         row: Dict[str, Any] = {
#             "true_class_id": class_id,
#             "true_display_id": class_id + 1,
#         }
#         for c, pred_id in enumerate(ids):
#             row[f"pred_{pred_id}"] = int(matrix[r, c])
#         rows.append(row)

#     return rows


# def _classwise_rows(
#     *,
#     metrics: Mapping[str, Any],
#     class_ids: Sequence[int],
#     class_names: Sequence[str],
#     old_ids: Sequence[int] = (),
# ) -> list[Dict[str, Any]]:

#     old = set(int(value) for value in old_ids)

#     rows = []

#     for class_id in [int(value) for value in class_ids]:

#         rows.append(
#             {
#                 "class_id": class_id,
#                 "display_id": class_id + 1,
#                 "class_name": str(class_names[class_id]),
#                 "phase_role": "historical" if class_id in old else "current",
#                 "test_accuracy": _mapping_value(
#                     metrics.get("per_class_accuracy"), class_id
#                 ),
#                 "test_logit_margin": _mapping_value(
#                     metrics.get("per_class_mean_logit_margin"), class_id
#                 ),
#                 "test_energy_margin": _mapping_value(
#                     metrics.get("per_class_mean_energy_margin"), class_id
#                 ),
#                 "test_raw_energy_margin": _mapping_value(
#                     metrics.get("per_class_mean_raw_energy_margin"), class_id
#                 ),
#             }
#         )

#     return rows


# def _print_metrics(name: str, metrics: Optional[Mapping[str, Any]]) -> None:

#     if metrics is None:

#         print(f"{name:<10} | disabled")

#         return

#     def number(value, *, percent=False):
#         if value is None:
#             return "N/A"
#         return f"{100*float(value):.2f}%" if percent else f"{float(value):.4f}"

#     print(
#         f"{name:<10} | OA={number(metrics['overall_accuracy'], percent=True)} | "
#         f"BA={number(metrics['balanced_accuracy'], percent=True)} | "
#         f"MinClass={number(metrics['minimum_class_accuracy'], percent=True)} | "
#         f"LogitMargin={number(metrics['mean_logit_margin'])} | "
#         f"NormEnergy={number(metrics['mean_energy_margin'])} | "
#         f"NormViol={number(metrics['energy_violation_fraction'], percent=True)} | "
#         f"RawEnergy={number(metrics.get('mean_raw_energy_margin', metrics['mean_energy_margin']))}"
#     )


# def _report_base_phase(
#     *,
#     args: argparse.Namespace,
#     dataset: IncrementalHSIDatasetManager,
#     model: NECILModel,
#     result: Mapping[str, Any],
# ) -> Dict[str, Any]:

#     class_ids = [int(value) for value in result["class_ids"]]

#     phase_dir = os.path.join(args.save_dir, "phase_0")

#     os.makedirs(phase_dir, exist_ok=True)

#     history_path = save_csv(
#         os.path.join(phase_dir, "training_history.csv"),
#         _history_rows(result["history"]),
#     )

#     classwise_path = save_csv(
#         os.path.join(phase_dir, "classwise_metrics.csv"),
#         _classwise_rows(
#             metrics=result["test"],
#             class_ids=class_ids,
#             class_names=dataset.target_names,
#         ),
#     )

#     confusion_path = save_csv(
#         os.path.join(phase_dir, "confusion_matrix.csv"),
#         _confusion_rows(result["test"], class_ids),
#     )

#     structure = class_structure_diagnostics(
#         model, class_ids, target_names=dataset.target_names
#     )

#     decision = structural_decision_diagnostics(model, class_ids)

#     structure_path = save_json(
#         os.path.join(phase_dir, "class_structure_report.json"),
#         {"phase": 0, "class_structure": structure, "decision_diagnostics": decision},
#     )

#     visualizations: Dict[str, Any] = {}

#     if bool(args.save_visualizations):

#         visualizations = generate_phase_qualitative_maps(
#             model=model,
#             dataset=dataset,
#             phase=0,
#             output_dir=os.path.join(phase_dir, "figures", "qualitative"),
#             device=args.device,
#             batch_size=args.eval_batch_size,
#             cmap_name=args.visualization_cmap,
#             dpi=args.visualization_dpi,
#         )

#     report = {
#         "phase": 0,
#         "seen_class_ids": class_ids,
#         "epochs": int(result["final_epoch"]),
#         "train": _metric_summary(result["train"]),
#         "validation": _metric_summary(result["validation"]),
#         "test": _metric_summary(result["test"]),
#         "class_structure": structure,
#         "decision": decision,
#         "visualizations": visualizations,
#     }

#     report_path = save_json(os.path.join(phase_dir, "phase_report.json"), report)

#     print("\nPhase 0 finalized.")
#     _print_metrics("Train", result["train"])
#     _print_metrics("Validation", result["validation"])
#     _print_metrics("Test", result["test"])
#     print(f"Checkpoint       : {result['checkpoint']}")
#     print(f"Phase report     : {report_path}")
#     print(f"Structure report : {structure_path}")
#     print(f"Training history : {history_path}")
#     print(f"Classwise metrics: {classwise_path}")
#     print(f"Confusion matrix : {confusion_path}")

#     return report


# def _report_incremental_phase(
#     *,
#     args: argparse.Namespace,
#     dataset: IncrementalHSIDatasetManager,
#     model: NECILModel,
#     result: Mapping[str, Any],
# ) -> Dict[str, Any]:
#     phase = int(result["phase"])
#     old_ids = [int(value) for value in result["old_class_ids"]]
#     new_ids = [int(value) for value in result["new_class_ids"]]
#     seen_ids = [int(value) for value in result["seen_class_ids"]]
#     phase_dir = os.path.join(args.save_dir, f"phase_{phase}")
#     os.makedirs(phase_dir, exist_ok=True)

#     method = dict(result["method_state"])

#     structure = class_structure_diagnostics(
#         model, seen_ids, target_names=dataset.target_names
#     )

#     decision = structural_decision_diagnostics(model, seen_ids)

#     cross_phase = cross_phase_cds_decision_diagnostics(
#         model, old_class_ids=old_ids, new_class_ids=new_ids
#     )

#     history_path = save_csv(
#         os.path.join(phase_dir, "training_history.csv"),
#         _history_rows(result["history"]),
#     )

#     classwise_path = save_csv(
#         os.path.join(phase_dir, "classwise_metrics.csv"),
#         _classwise_rows(
#             metrics=result["cumulative_test"],
#             class_ids=seen_ids,
#             class_names=dataset.target_names,
#             old_ids=old_ids,
#         ),
#     )

#     confusion_path = save_csv(
#         os.path.join(phase_dir, "confusion_matrix.csv"),
#         _confusion_rows(result["cumulative_test"], seen_ids),
#     )

#     structure_path = save_json(
#         os.path.join(phase_dir, "class_structure_report.json"),
#         {
#             "phase": phase,
#             "class_structure": structure,
#             "decision_diagnostics": decision,
#             "cross_phase_decision": cross_phase,
#             "method_state": method,
#             "phase_origin_confusion": result.get("phase_origin_confusion"),
#         },
#     )

#     visualizations: Dict[str, Any] = {}

#     if bool(args.save_visualizations):

#         visualizations = generate_phase_qualitative_maps(
#             model=model,
#             dataset=dataset,
#             phase=phase,
#             output_dir=os.path.join(phase_dir, "figures", "qualitative"),
#             device=args.device,
#             batch_size=args.eval_batch_size,
#             cmap_name=args.visualization_cmap,
#             dpi=args.visualization_dpi,
#         )

#     old_group = result["old_new"]["old"]

#     new_group = result["old_new"]["new"]

#     report = {
#         "phase": phase,
#         "old_class_ids": old_ids,
#         "new_class_ids": new_ids,
#         "seen_class_ids": seen_ids,
#         "epochs": int(result["final_epoch"]),
#         "cumulative_test": _metric_summary(result["cumulative_test"]),
#         "old_ba": float(old_group["balanced_accuracy"]),
#         "new_ba": float(new_group["balanced_accuracy"]),
#         "harmonic_old_new_accuracy": float(
#             result["old_new"]["harmonic_old_new_accuracy"]
#         ),
#         "forgetting": float(result["forgetting"]),
#         "method_state": method,
#         "class_structure": structure,
#         "decision": decision,
#         "cross_phase_decision": cross_phase,
#         "phase_origin_confusion": result.get("phase_origin_confusion"),
#         "visualizations": visualizations,
#     }

#     report_path = save_json(os.path.join(phase_dir, "phase_report.json"), report)

#     print(f"\nPhase {phase} finalized.")

#     _print_metrics("Cumulative", result["cumulative_test"])

#     print(
#         f"OldBA={100*report['old_ba']:.2f}% | NewBA={100*report['new_ba']:.2f}% | "
#         f"H={100*report['harmonic_old_new_accuracy']:.2f}% | "
#         f"F={100*report['forgetting']:.2f}%"
#     )

#     print(
#         "Joint incremental training | "
#         f"Epochs={int(method['representation_epochs'])} "
#         f"JointObj={method['final_joint_classification_loss']:.4f} "
#         f"CEreal={method['final_real_current_classification_loss']:.4f} "
#         f"SepMargin={method['final_cds_separation_margin']:.4f} "
#         f"SepViol={100*method['final_cds_separation_violation_fraction']:.2f}% "
#         f"Pres={method['final_cds_preservation_loss']:.4f} "
#         f"[Intra={method['final_cds_intra_loss']:.4f},Inter={method['final_cds_inter_loss']:.4f}] "
#         f"FKD={method['final_feature_kd_loss']:.4f}"
#     )

#     print(
#         "Whole-CDS orthogonal-frame diagnostics | "
#         f"RotationApplied={method['phase_transport_accepted_on_new_class_holdout']} "
#         f"Frame={method.get('historical_frame_source', 'legacy')} "
#         f"MapErr={method['phase_mapping_error']:.4f} "
#         f"Drift={method['representation_drift_rms']:.4f} "
#         f"ScaleDev={method['phase_scale_deviation_rms']:.4f} "
#         f"Translation={method['phase_translation_rms']:.4f}"
#     )
#     head_history = method.get("historical_support_reconstruction") or []
#     selected_head_epoch = int(method.get("selected_support_head_epoch", 0))
#     if head_history:
#         selected_rows = [
#             row
#             for row in head_history
#             if int(row.get("epoch", -1)) == selected_head_epoch
#         ]
#         if len(selected_rows) != 1:
#             raise RuntimeError(
#                 "selected head epoch missing or duplicated in reconstruction history"
#             )
#         selected = selected_rows[0]
#         old_risk = float(selected["historical_support_loss"])
#         new_ce = float(selected["selection_new_ce"])
#         combined = float(selected["selection_objective"])
#         if (
#             bool(method.get("support_reconstruction_active"))
#             and selected_head_epoch > 0
#         ):
#             summary = (
#                 f"same-head final epoch {selected_head_epoch} retained (audit only)"
#             )
#         elif bool(method.get("support_reconstruction_active")):
#             summary = "same-head training configured; no optimizer epoch completed"
#         else:
#             summary = "no post-hoc training; final-epoch Q+W retained"
#         print(
#             "Joint Whole-CDS cosine classifier | "
#             f"{summary} | "
#             f"audit_source={selected.get('selection_source', 'unknown')} "
#             f"OldRisk={old_risk:.4f} "
#             f"NewCE={new_ce:.4f} "
#             f"JointAudit={combined:.4f}"
#         )
#     else:
#         print(
#             "Direct CDS classifier state | "
#             f"JointLoss={method['final_direct_cds_classifier_loss']:.4f} "
#             f"HistoricalCDS={method['final_direct_cds_historical_loss']:.4f} "
#             f"CurrentReal={method['final_direct_cds_current_loss']:.4f}"
#         )
#     print(
#         "Read-only CDS classifier diagnostic | "
#         f"OldNewViol={100*method['final_whole_cds_old_new_violation_fraction']:.2f}%"
#     )

#     print(
#         "Covariance-aware learned/native spectral pair risk | "
#         f"Learned={method['classifier_learned_pair_risk_mean']:.4f} "
#         f"Physical={method['classifier_physical_pair_risk_mean']:.4f} "
#         f"Effective={method['classifier_effective_pair_risk_mean']:.4f}"
#     )

#     print(
#         "Classifier movement during joint training | "
#         f"OldMax={method['historical_classifier_training_max_change']:.3e} "
#         f"OldDir={method['historical_classifier_training_direction_change']:.3e} "
#         f"NewMax={method['incoming_classifier_training_max_change']:.3e} "
#         f"NewDir={method['incoming_classifier_training_direction_change']:.3e}"
#     )

#     # Phase-start old-versus-all is intentionally unavailable: W_new has not
#     # yet been admitted in a meaningful measurement. Preserve N/A instead of
#     # throwing TypeError when the trainer correctly stores None.
#     start_old_all = method.get("phase_start_old_all_seen_ba")
#     start_old_all_text = (
#         "N/A" if start_old_all is None else f"{100 * float(start_old_all):.2f}%"
#     )
#     print(
#         "Competition (final old/new: reporting; post-rep new: train) | "
#         f"StartOldOnly/All={100*method['phase_start_old_ba']:.2f}%/"
#         f"{start_old_all_text} | "
#         f"PostRepOldOnly/All={100*method['post_rep_old_only_ba']:.2f}%/"
#         f"{100*method['post_rep_old_all_seen_ba']:.2f}% | "
#         f"FinalOldOnly/All={100*method['final_old_only_ba']:.2f}%/"
#         f"{100*method['final_old_all_seen_ba']:.2f}% | "
#         f"FinalNewOnly/All={100*method['final_new_only_ba']:.2f}%/"
#         f"{100*method['final_new_all_seen_ba']:.2f}%"
#     )

#     print(f"Checkpoint       : {result['checkpoint']}")
#     print(f"Phase report     : {report_path}")
#     print(f"Structure report : {structure_path}")
#     print(f"Training history : {history_path}")
#     print(f"Classwise metrics: {classwise_path}")
#     print(f"Confusion matrix : {confusion_path}")

#     return report


# def run_experiment(args: argparse.Namespace) -> Dict[str, Any]:

#     os.makedirs(args.save_dir, exist_ok=True)

#     dataset, data_summary, preprocessing_state = _build_data(args)
#     save_json(os.path.join(args.save_dir, "data_protocol.json"), data_summary)
#     SaveHSIPreprocessor(
#         os.path.join(args.save_dir, "phase_0_preprocessor.npz"), preprocessing_state
#     )
#     save_json(os.path.join(args.save_dir, "experiment_config.json"), vars(args))
#     model = NECILModel(args)
#     trainer = Trainer(model, dataset, args)
#     phase_reports: Dict[int, Dict[str, Any]] = {}
#     mode = str(args.run_mode)
#     if mode in {"base", "all"}:
#         base_result = trainer.run_base_only(
#             epochs=args.epochs_base, batch_size=args.batch_size, lr=args.lr
#         )
#         phase_reports[0] = _report_base_phase(
#             args=args, dataset=dataset, model=model, result=base_result
#         )
#         next_phase: Optional[int] = 1

#     else:

#         restored = trainer.load_checkpoint(args.resume_checkpoint)
#         next_phase = restored["next_phase"]
#         if next_phase is None:
#             raise RuntimeError("checkpoint already contains the final protocol phase")
#     if mode in {"incremental", "all"}:
#         if next_phase is None:
#             raise RuntimeError("no incremental phase remains")
#         highest_phase = int(dataset.nb_tasks) - 1
#         if args.max_phase is not None:
#             highest_phase = min(highest_phase, int(args.max_phase))
#         for phase in range(int(next_phase), highest_phase + 1):

#             result = trainer.run_incremental_phase(
#                 phase=phase,
#                 epochs=args.epochs_inc,
#                 batch_size=args.batch_size,
#                 lr=args.lr_inc,
#             )
#             phase_reports[phase] = _report_incremental_phase(
#                 args=args, dataset=dataset, model=model, result=result
#             )

#     finalized_phases = [int(value) for value in dataset.finalized_phases]

#     if not finalized_phases or finalized_phases != list(range(len(finalized_phases))):

#         raise RuntimeError("experiment did not finalize a contiguous phase prefix")

#     latest = finalized_phases[-1]

#     latest_metrics = trainer.evaluate_loader(
#         dataset.get_reporting_dataloader(latest, batch_size=args.eval_batch_size),
#         class_ids=list(model.committed_class_ids),
#         target_class_ids=list(model.committed_class_ids),
#     )

#     run_summary = {
#         "dataset": str(args.dataset),
#         "seed": int(args.seed),
#         "run_mode": mode,
#         "finalized_phases": finalized_phases,
#         "seen_class_ids": list(model.committed_class_ids),
#         "latest_test": _metric_summary(latest_metrics),
#         "protocol": {
#             "exemplar_memory_size": 0,
#             "stored_old_samples": 0,
#             "stored_old_feature_rows": 0,
#             "stored_synthetic_support": 0,
#             "persistent_decision_sufficient_statistics": False,
#             "stored_compact_class_moments": True,
#             "stored_native_spectral_variances": True,
#             "stored_decision_feature_rows": 0,
#             "transient_synthetic_support_during_incremental_phase": False,
#             "temporary_previous_backbone_reference": latest > 0,
#             "temporary_previous_classifier_reference": latest > 0,
#             "previous_model_reference_persisted": False,
#         },
#     }

#     summary_path = save_json(
#         os.path.join(args.save_dir, "run_summary.json"), run_summary
#     )

#     run_summary["run_summary"] = summary_path
#     print("\nRun complete.")
#     print(f"Finalized phases : {finalized_phases}")
#     print(f"Seen classes     : {list(model.committed_class_ids)}")
#     print(f"Run summary      : {summary_path}")
#     return run_summary


# def _sample_statistics(values: Sequence[float]) -> Dict[str, float]:

#     array = np.asarray([float(value) for value in values], dtype=np.float64)
#     if array.ndim != 1 or array.size == 0 or not np.isfinite(array).all():
#         raise ValueError("aggregate values must be non-empty and finite")

#     return {
#         "mean": float(array.mean()),
#         "std": float(array.std(ddof=1)) if array.size > 1 else 0.0,
#         "min": float(array.min()),
#         "max": float(array.max()),
#     }


# def run_multiseed_experiments(args: argparse.Namespace) -> Dict[str, Any]:
#     if args.seeds is None or str(args.run_mode) not in {"base", "all"}:

#         raise ValueError("multi-seed mode requires --seeds and run_mode base/all")
#     seeds = [int(value) for value in args.seeds]
#     root = os.path.abspath(args.save_dir)
#     os.makedirs(root, exist_ok=True)
#     run_rows: list[Dict[str, Any]] = []
#     for seed in seeds:

#         seed_args = copy.deepcopy(args)
#         seed_args.seed = seed
#         seed_args.seeds = None
#         seed_args.resume_checkpoint = ""
#         seed_args.save_dir = os.path.join(root, f"seed_{seed}")
#         set_seed(seed, bool(seed_args.deterministic))
#         summary = run_experiment(seed_args)
#         latest = summary["latest_test"]
#         row = {"seed": seed, "output_dir": seed_args.save_dir}
#         for key, value in latest.items():

#             row[f"final_{key}"] = value

#         if str(args.run_mode) == "all" and max(summary["finalized_phases"]) > 0:

#             final_phase = max(summary["finalized_phases"])

#             report_path = os.path.join(
#                 seed_args.save_dir, f"phase_{final_phase}", "phase_report.json"
#             )
#             with open(report_path, "r", encoding="utf-8") as stream:
#                 report = json.load(stream)
#             for key in ("old_ba", "new_ba", "harmonic_old_new_accuracy", "forgetting"):
#                 row[key] = float(report[key])

#         run_rows.append(row)

#     metrics = sorted(key for key in run_rows[0] if key.startswith("final_"))

#     aggregate = {
#         key: _sample_statistics([float(row[key]) for row in run_rows])
#         for key in metrics
#     }

#     if str(args.run_mode) == "all" and "old_ba" in run_rows[0]:

#         for key in ("old_ba", "new_ba", "harmonic_old_new_accuracy", "forgetting"):

#             aggregate[key] = _sample_statistics([float(row[key]) for row in run_rows])

#     runs_path = save_csv(os.path.join(root, "multiseed_runs.csv"), run_rows)

#     summary = {
#         "mode": f"independent_{args.run_mode}_multiseed",
#         "dataset": str(args.dataset),
#         "seeds": seeds,
#         "protocol_seed": args.protocol_seed,
#         "aggregate": aggregate,
#         "runs_csv": runs_path,
#     }

#     summary_path = save_json(os.path.join(root, "multiseed_summary.json"), summary)
#     summary["summary_json"] = summary_path
#     return summary


# def main(argv: Optional[Sequence[str]] = None) -> None:
#     args = build_parser().parse_args(argv)
#     validate_args(args)
#     seed0 = int(args.seeds[0]) if args.seeds is not None else int(args.seed)
#     set_seed(seed0, bool(args.deterministic))
#     args.device = resolve_device(args.device)
#     args.save_dir = os.path.abspath(args.save_dir)
#     if args.seeds is not None:
#         run_multiseed_experiments(args)
#     else:
#         run_experiment(args)


# if __name__ == "__main__":
#     main()
