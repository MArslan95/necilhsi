from __future__ import annotations

"""HSI loading and preprocessing for class-incremental experiments.

The loader maps foreground labels once, keeps the processed spectral-spatial
cube consistent across phases, and exposes ordinary PyTorch dataset helpers.
The incremental protocol decides which samples and classes are available.
Training-only spatial symmetry augmentation is deliberately not performed here;
the base trainer applies it after batch loading so preprocessing and all
evaluation/reference loaders remain deterministic and unchanged.
"""

import os
from numbers import Integral, Real
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import scipy.io as sio
import torch
from torch.utils.data import DataLoader, Dataset


DATASET_INFO: Dict[str, Dict[str, Any]] = {
    "IP": {
        "data_file": "Indian_pines_corrected.mat", "data_key": "indian_pines_corrected",
        "gt_file": "Indian_pines_gt.mat", "gt_key": "indian_pines_gt", "num_classes": 16,
        "target_names": [
            "Alfalfa", "Corn-notill", "Corn-mintill", "Corn", "Grass-pasture",
            "Grass-trees", "Grass-pasture-mowed", "Hay-windrowed", "Oats",
            "Soybean-notill", "Soybean-mintill", "Soybean-clean", "Wheat", "Woods",
            "Buildings-Grass-Trees-Drives", "Stone-Steel-Towers",
        ],
    },
    "SA": {
        "data_file": "Salinas_corrected.mat", "data_key": "salinas_corrected",
        "gt_file": "Salinas_gt.mat", "gt_key": "salinas_gt", "num_classes": 16,
        "target_names": [
            "Weeds_1", "Weeds_2", "Fallow", "Fallow_rough_plow", "Fallow_smooth",
            "Stubble", "Celery", "Grapes_untrained", "Soil_vinyard_develop",
            "Corn_Weeds", "Lettuce_4wk", "Lettuce_5wk", "Lettuce_6wk",
            "Lettuce_7wk", "Vinyard_untrained", "Vinyard_trellis",
        ],
    },
    "PU": {
        "data_file": "PaviaU.mat", "data_key": "paviaU", "gt_file": "PaviaU_gt.mat",
        "gt_key": "paviaU_gt", "num_classes": 9,
        "target_names": ["Asphalt", "Meadows", "Gravel", "Trees", "Painted", "Soil", "Bitumen", "Bricks", "Shadows"],
    },
    "PC": {
        "data_file": "Pavia.mat", "data_key": "pavia", "gt_file": "Pavia_gt.mat",
        "gt_key": "pavia_gt", "num_classes": 9,
        "target_names": ["Water", "Trees", "Asphalt", "Bricks", "Bitumen", "Tiles", "Shadows", "Meadows", "Soil"],
    },
    "UH13": {
        "data_file": "HU13.mat", "data_key": "HSI", "gt_file": "HU13_gt.mat",
        "gt_key": "gt", "num_classes": 15,
        "target_names": [
            "Healthy grass", "Stressed grass", "Synthetic grass", "Trees", "Soil", "Water",
            "Residential", "Commercial", "Road", "Highway", "Railway", "Parking Lot 1",
            "Parking Lot 2", "Tennis Court", "Running Track",
        ],
    },
    "BS": {
        "data_file": "Botswana.mat", "data_key": "Botswana", "gt_file": "Botswana_gt.mat",
        "gt_key": "Botswana_gt", "num_classes": 14,
        "target_names": [
            "Water", "Hippo Grass", "Floodplain Grasses 1", "Floodplain Grasses 2", "Reeds 1",
            "Riparian", "Firescar 2", "Island Interior", "Woodlands", "Acacia Shrublands",
            "Acacia Grasslands", "Short Mopane", "Mixed Mopane", "Exposed Soils",
        ],
    },
    "LK": {
        "data_file": "WHU_Hi_LongKou.mat", "data_key": "WHU_Hi_LongKou",
        "gt_file": "WHU_Hi_LongKou_gt.mat", "gt_key": "WHU_Hi_LongKou_gt", "num_classes": 9,
        "target_names": ["Corn", "Cotton", "Sesame", "Broad-leaf soybean", "Narrow-leaf soybean", "Rice", "Water", "Roads and houses", "Mixed weed"],
    },
    "HH": {
        "data_file": "WHU_Hi_HongHu.mat", "data_key": "WHU_Hi_HongHu",
        "gt_file": "WHU_Hi_HongHu_gt.mat", "gt_key": "WHU_Hi_HongHu_gt", "num_classes": 22,
        "target_names": [
            "Red roof", "Road", "Bare soil", "Cotton", "Cotton firewood", "Rape", "Chinese cabbage",
            "Pakchoi", "Cabbage", "Tuber mustard", "Brassica parachinensis", "Brassica chinensis",
            "Small Brassica chinensis", "Lactuca sativa", "Celtuce", "Film covered lettuce",
            "Romaine lettuce", "Carrot", "White radish", "Garlic sprout", "Broad bean", "Tree",
        ],
    },
    "HC": {
        "data_file": "WHU_Hi_HanChuan.mat", "data_key": "WHU_Hi_HanChuan",
        "gt_file": "WHU_Hi_HanChuan_gt.mat", "gt_key": "WHU_Hi_HanChuan_gt", "num_classes": 16,
        "target_names": ["Strawberry", "Cowpea", "Soybean", "Sorghum", "Water spinach", "Watermelon", "Greens", "Trees", "Grass", "Red roof", "Gray roof", "Plastic", "Bare soil", "Road", "Bright object", "Water"],
    },
    "QUH": {
        "data_file": "QUH-Qingyun.mat", "data_key": "Chengqu", "gt_file": "QUH-Qingyun_GT.mat",
        "gt_key": "ChengquGT", "num_classes": 6, "background_label": None,
        "target_names": ["Trees", "Concrete building", "Car", "Ironhide building", "Plastic playground", "Asphalt road"],
    },
    "PI": {
        "data_file": "QUH-Pingan.mat", "data_key": "Haigang", "gt_file": "QUH-Pingan_GT.mat",
        "gt_key": "HaigangGT", "num_classes": 10, "background_label": None,
        "target_names": ["Ship", "Seawater", "Trees", "Concrete structure building", "Floating pier", "Brick houses", "Steel houses", "Wharf construction land", "Car", "Road"],
    },
    "TH": {
        "data_file": "QUH-Tangdaowan.mat", "data_key": "Tangdaowan", "gt_file": "QUH-Tangdaowan_GT.mat",
        "gt_key": "TangdaowanGT", "num_classes": 18, "background_label": None,
        "target_names": ["Rubber track", "Flagging", "Sandy", "Asphalt", "Boardwalk", "Rocky shallows", "Grassland", "Bulrush", "Gravel road", "Ligustrum vicaryi", "Coniferous pine", "Spiraea", "Bare soil", "Buxus sinica", "Photinia serrulata", "Populus", "Ulmus pumila L", "Seawater"],
    },
}

IGNORE_LABEL_VALUES = (255,)


def _exact_integer(value: object, *, name: str, minimum: Optional[int] = None) -> int:
    """Convert an integer-like scalar without silent truncation."""
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
        if not np.isfinite(number) or not number.is_integer():
            raise ValueError(f"{name} must be a finite integer")
        result = int(number)
    else:
        raise ValueError(f"{name} must be an integer")
    if minimum is not None and result < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return result


def _exact_bool(value: object, *, name: str) -> bool:
    """Parse an explicit Boolean without bool("false") coercion."""
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, Integral) and not isinstance(value, bool):
        integer = int(value)
        if integer in (0, 1):
            return bool(integer)
    if isinstance(value, str):
        token = value.strip().lower()
        if token in {"true", "1", "yes", "y"}:
            return True
        if token in {"false", "0", "no", "n"}:
            return False
    raise ValueError(f"{name} must be a Boolean")


def _integer_array(value: object, *, name: str) -> np.ndarray:
    """Return exact signed-int64 data without silent coercion or overflow."""
    raw = np.asarray(value)
    if raw.dtype == np.bool_ or raw.dtype.kind not in "iuf":
        raise ValueError(f"{name} must contain integer values")

    info = np.iinfo(np.int64)

    if np.issubdtype(raw.dtype, np.floating):
        if (
            not np.isfinite(raw).all()
            or not np.equal(raw, np.round(raw)).all()
        ):
            raise ValueError(f"{name} must contain finite integer values")

        # float64 cannot exactly represent int64.max; use the exact
        # representable half-open interval [-2**63, 2**63).
        lower = float(-(2**63))
        upper_exclusive = float(2**63)
        if raw.size and (
            np.any(raw < lower)
            or np.any(raw >= upper_exclusive)
        ):
            raise ValueError(
                f"{name} contains values outside signed int64 range"
            )

    elif raw.size and np.issubdtype(raw.dtype, np.unsignedinteger):
        if np.any(raw > info.max):
            raise ValueError(
                f"{name} contains values outside signed int64 range"
            )

    elif raw.size:
        if np.any(raw < info.min) or np.any(raw > info.max):
            raise ValueError(
                f"{name} contains values outside signed int64 range"
            )

    return raw.astype(np.int64, copy=False)


def _load_mat_key(path: str, key: str) -> np.ndarray:
    payload = sio.loadmat(path)
    if key not in payload:
        available = sorted(name for name in payload if not name.startswith("__"))
        raise KeyError(f"{key!r} not found in {path}; available keys={available}")
    return payload[key]


def resolve_label_policy(method: str, gt: np.ndarray) -> Dict[str, Any]:
    if method not in DATASET_INFO:
        raise ValueError(f"unknown dataset {method!r}; available={sorted(DATASET_INFO)}")
    info = DATASET_INFO[method]
    labels_map = _integer_array(gt, name="ground truth")
    values = [
        int(v)
        for v in np.unique(labels_map)
        if int(v) >= 0 and int(v) not in IGNORE_LABEL_VALUES
    ]
    background = info.get("background_label", 0)
    classes = values if background is None else [v for v in values if v != background]
    classes = sorted(classes)
    expected = _exact_integer(info["num_classes"], name="num_classes", minimum=1)
    if len(classes) != expected:
        raise RuntimeError(
            f"{method}: expected {expected} foreground classes, found {classes}"
        )
    names = [str(v) for v in info["target_names"]]
    if len(names) != expected:
        raise RuntimeError(f"{method}: target_names does not match num_classes")
    raw_to_class = {raw_label: class_id for class_id, raw_label in enumerate(classes)}
    return {
        "method": method,
        "background_label": background,
        "raw_class_values": classes,
        "label_to_train": raw_to_class,
        "train_to_label": {value: key for key, value in raw_to_class.items()},
        "num_classes": expected,
        "target_names": names,
    }


def ExtractLabeledPixelIndex(
    gt: np.ndarray,
    *,
    label_policy: Mapping[str, Any],
) -> Tuple[np.ndarray, np.ndarray]:
    labels_map = _integer_array(gt, name="GT")
    if labels_map.ndim != 2:
        raise ValueError("GT must have shape [H,W]")
    raw_classes = _integer_array(
        label_policy["raw_class_values"], name="raw_class_values"
    ).reshape(-1)
    rows, cols = np.where(np.isin(labels_map, raw_classes))
    coords = np.stack((rows, cols), axis=1).astype(np.int64, copy=False)
    if np.unique(coords, axis=0).shape[0] != coords.shape[0]:
        raise RuntimeError("label extraction produced duplicate coordinates")
    mapping = {
        _exact_integer(k, name="raw label"): _exact_integer(v, name="class ID", minimum=0)
        for k, v in label_policy["label_to_train"].items()
    }
    labels = np.asarray(
        [mapping[int(v)] for v in labels_map[rows, cols]], dtype=np.int64
    )
    return labels, coords


def BuildPreprocessFitMask(
    gt_shape: Sequence[int],
    coords: np.ndarray,
    sample_indices: Sequence[int],
    **_: Any,
) -> np.ndarray:
    shape = tuple(_exact_integer(v, name="gt_shape value", minimum=1) for v in gt_shape)
    coordinates = _integer_array(coords, name="coords")
    indices = _integer_array(sample_indices, name="sample_indices").reshape(-1)
    if len(shape) != 2 or coordinates.ndim != 2 or coordinates.shape[1] != 2:
        raise ValueError("gt_shape and coords are invalid")
    if coordinates.shape[0] == 0:
        raise ValueError("coords cannot be empty")
    if np.unique(coordinates, axis=0).shape[0] != coordinates.shape[0]:
        raise ValueError("coords must identify unique labelled pixels")
    if (
        np.any(coordinates[:, 0] < 0)
        or np.any(coordinates[:, 0] >= shape[0])
        or np.any(coordinates[:, 1] < 0)
        or np.any(coordinates[:, 1] >= shape[1])
    ):
        raise ValueError("coords lie outside gt_shape")
    if indices.size == 0 or indices.min() < 0 or indices.max() >= len(coordinates):
        raise ValueError("sample_indices must select valid labelled pixels")
    if np.unique(indices).size != indices.size:
        raise ValueError("sample_indices cannot contain duplicates")
    mask = np.zeros(shape, dtype=bool)
    selected = coordinates[indices]
    mask[selected[:, 0], selected[:, 1]] = True
    if int(mask.sum()) != int(indices.size):
        raise RuntimeError("preprocess fit mask lost selected training pixels")
    return mask


def _fit_preprocessor(
    cube: np.ndarray,
    fit_mask: np.ndarray,
    *,
    n_components: Optional[int],
    whiten: bool,
) -> Dict[str, Any]:
    raw = np.asarray(cube, dtype=np.float32)
    mask = np.asarray(fit_mask, dtype=bool)

    if (
        raw.ndim != 3
        or raw.shape[2] == 0
        or not np.isfinite(raw).all()
        or mask.shape != raw.shape[:2]
        or not mask.any()
    ):
        raise ValueError(
            "cube must be finite [H,W,B] and fit_mask must select pixels"
        )

    pixels = raw[mask]
    mean = pixels.mean(axis=0).astype(np.float32)
    observed_std = pixels.std(axis=0, ddof=0)
    std = np.where(
        observed_std > 0.0,
        observed_std,
        1.0,
    ).astype(np.float32)

    whiten_value = _exact_bool(
        whiten,
        name="whiten",
    )

    state: Dict[str, Any] = {
        "raw_bands": int(raw.shape[-1]),
        "normalization_mean": mean,
        "normalization_std": std,
        "fit_pixel_count": int(pixels.shape[0]),
        "pca_components": np.empty(
            (0, raw.shape[-1]),
            dtype=np.float32,
        ),
        "pca_mean": np.empty(0, dtype=np.float32),
        "pca_variance": np.empty(0, dtype=np.float32),
        "whiten": whiten_value,
        "fit_scope": "provided_mask",
    }

    if n_components is not None:
        component_count = _exact_integer(
            n_components,
            name="n_components",
            minimum=1,
        )

        maximum_components = min(
            int(raw.shape[-1]),
            int(pixels.shape[0]),
        )
        if component_count > maximum_components:
            raise ValueError(
                "n_components cannot exceed min(raw_bands, "
                f"base_train_fit_pixels)={maximum_components}; "
                f"got {component_count}"
            )
        if pixels.shape[0] < 2:
            raise ValueError(
                "PCA requires at least two base-TRAIN fit pixels"
            )

        from sklearn.decomposition import PCA

        normalized_pixels = (
            pixels - mean
        ) / std

        pca = PCA(
            n_components=component_count,
            whiten=False,
            svd_solver="auto",
        ).fit(normalized_pixels)

        state["pca_components"] = (
            pca.components_.astype(np.float32)
        )
        state["pca_mean"] = (
            pca.mean_.astype(np.float32)
        )
        state["pca_variance"] = (
            pca.explained_variance_.astype(np.float32)
        )

    return state


def ValidateHSIPreprocessorState(
    state: Mapping[str, Any],
    *,
    require_base_train: bool = False,
) -> bool:
    required = {
        "raw_bands",
        "fit_pixel_count",
        "normalization_mean",
        "normalization_std",
        "pca_components",
        "pca_mean",
        "pca_variance",
        "whiten",
        "fit_scope",
    }
    missing = required - set(state)
    if missing:
        raise ValueError(
            f"preprocessor state is missing keys {sorted(missing)}"
        )

    bands = _exact_integer(
        state["raw_bands"],
        name="raw_bands",
        minimum=1,
    )
    fit_pixel_count = _exact_integer(
        state["fit_pixel_count"],
        name="fit_pixel_count",
        minimum=1,
    )
    whiten = _exact_bool(
        state["whiten"],
        name="whiten",
    )

    mean = np.asarray(
        state["normalization_mean"],
        dtype=np.float32,
    ).reshape(-1)
    std = np.asarray(
        state["normalization_std"],
        dtype=np.float32,
    ).reshape(-1)
    components = np.asarray(
        state["pca_components"],
        dtype=np.float32,
    )
    pca_mean = np.asarray(
        state["pca_mean"],
        dtype=np.float32,
    ).reshape(-1)
    variance = np.asarray(
        state["pca_variance"],
        dtype=np.float32,
    ).reshape(-1)

    if mean.shape != (bands,) or std.shape != (bands,):
        raise ValueError(
            "normalization state has incompatible dimensions"
        )
    if (
        not np.isfinite(mean).all()
        or not np.isfinite(std).all()
        or np.any(std <= 0.0)
    ):
        raise ValueError(
            "normalization state contains invalid values"
        )

    scope = str(state["fit_scope"]).strip()
    if scope not in {
        "provided_mask",
        "base_train",
    }:
        raise ValueError(
            "preprocessor fit_scope is invalid"
        )
    if require_base_train and scope != "base_train":
        raise ValueError(
            "persistent HSI preprocessing must be fitted on base_train only"
        )

    if (
        components.ndim != 2
        or components.shape[1] != bands
    ):
        raise ValueError(
            "PCA components have an incompatible shape"
        )
    if not np.isfinite(components).all():
        raise ValueError(
            "PCA components contain NaN/Inf"
        )

    component_count = int(
        components.shape[0]
    )

    if component_count == 0:
        if pca_mean.size != 0 or variance.size != 0:
            raise ValueError(
                "PCA-disabled state must have empty pca_mean and pca_variance"
            )
        if whiten:
            raise ValueError(
                "whitening cannot be enabled without PCA components"
            )
        return True

    if component_count > min(
        bands,
        fit_pixel_count,
    ):
        raise ValueError(
            "PCA component count exceeds the dimensions of its fit data"
        )

    if pca_mean.shape != (bands,):
        raise ValueError(
            "PCA mean has incompatible dimensions"
        )
    if variance.shape != (component_count,):
        raise ValueError(
            "PCA variance has incompatible dimensions"
        )
    if (
        not np.isfinite(pca_mean).all()
        or not np.isfinite(variance).all()
        or np.any(variance < 0.0)
    ):
        raise ValueError(
            "PCA state contains invalid values"
        )
    if whiten and np.any(variance <= 0.0):
        raise ValueError(
            "whitening requires strictly positive PCA variance"
        )

    return True


def ApplyHSIPreprocessor(
    HSI_raw: np.ndarray,
    state: Mapping[str, Any],
) -> Tuple[np.ndarray, np.ndarray]:
    ValidateHSIPreprocessorState(state)
    raw = np.asarray(HSI_raw, dtype=np.float32)
    if raw.ndim != 3 or raw.shape[-1] != int(state["raw_bands"]):
        raise ValueError("raw cube does not match preprocessing state")
    if not np.isfinite(raw).all():
        raise ValueError("raw cube contains NaN or Inf")

    mean = np.asarray(state["normalization_mean"], dtype=np.float32)
    std = np.asarray(state["normalization_std"], dtype=np.float32)
    processed = (raw - mean) / std

    components = np.asarray(state["pca_components"], dtype=np.float32)
    if components.size:
        flat = processed.reshape(-1, processed.shape[-1])
        transformed = (
            flat - np.asarray(state["pca_mean"], dtype=np.float32)
        ) @ components.T
        if bool(state["whiten"]):
            transformed /= np.sqrt(
                np.asarray(state["pca_variance"], dtype=np.float32)
            )
        processed = transformed.reshape(raw.shape[0], raw.shape[1], -1)

    processed = np.asarray(processed, dtype=np.float32)
    if not np.isfinite(processed).all():
        raise RuntimeError("preprocessing produced NaN/Inf")

    # The second cube intentionally remains the original ordered full-band HSI.
    # The spectral backbone applies the same base-train band statistics itself.
    return np.ascontiguousarray(processed), np.ascontiguousarray(raw)


def FitHSIPreprocessor(
    HSI_raw: np.ndarray,
    *,
    fit_mask: Optional[np.ndarray] = None,
    apply_reduction: bool = True,
    reduction_method: str = "PCA",
    n_components: int = 30,
    whiten: bool = False,
    **_: Any,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    use_reduction = _exact_bool(
        apply_reduction,
        name="apply_reduction",
    )
    whiten_value = _exact_bool(
        whiten,
        name="whiten",
    )
    if (
        use_reduction
        and str(reduction_method).strip().lower() != "pca"
    ):
        raise ValueError(
            "only PCA reduction is supported"
        )
    if whiten_value and not use_reduction:
        raise ValueError(
            "whiten=True requires PCA reduction"
        )

    raw = np.asarray(HSI_raw, dtype=np.float32)
    if raw.ndim != 3 or raw.shape[-1] <= 0 or not np.isfinite(raw).all():
        raise ValueError("HSI_raw must be finite [H,W,B]")
    if fit_mask is None:
        raise ValueError(
            "fit_mask is required; preprocessing statistics must be fitted "
            "on base-train centres only"
        )
    mask = np.asarray(fit_mask, dtype=bool)
    state = _fit_preprocessor(
        raw,
        mask,
        n_components=(
            _exact_integer(n_components, name="n_components", minimum=1)
            if use_reduction
            else None
        ),
        whiten=whiten_value,
    )
    processed, ordered = ApplyHSIPreprocessor(raw, state)
    return processed, ordered, state


def FitHSIPreprocessorFromProtocol(
    HSI_raw: np.ndarray,
    *,
    coords: np.ndarray,
    protocol: Mapping[str, Any],
    **kwargs: Any,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Fit persistent preprocessing from exactly phase-0 TRAIN centres.

    This validation happens before fitting because preprocessing precedes
    IncrementalHSIDatasetManager construction.  A malformed protocol therefore
    cannot leak future/validation/test rows into normalization or PCA and only
    fail later when the manager is created.
    """
    required = {
        "base_train_indices",
        "phase_train_indices",
        "phase_to_classes",
        "global_labels",
    }
    missing = required - set(protocol)
    if missing:
        raise ValueError(
            "protocol is missing preprocessing-critical fields "
            f"{sorted(missing)}"
        )

    coordinates = _integer_array(
        coords,
        name="coords",
    )
    if (
        coordinates.ndim != 2
        or coordinates.shape[1] != 2
    ):
        raise ValueError(
            "coords must have shape [N,2]"
        )

    global_labels = _integer_array(
        protocol["global_labels"],
        name="protocol global_labels",
    ).reshape(-1)
    if global_labels.shape != (
        coordinates.shape[0],
    ):
        raise ValueError(
            "protocol global_labels must align with coords"
        )
    if bool((global_labels < 0).any()):
        raise ValueError(
            "protocol global_labels must be non-negative"
        )

    fit_indices = _integer_array(
        protocol["base_train_indices"],
        name="base_train_indices",
    ).reshape(-1)
    if fit_indices.size == 0:
        raise ValueError(
            "base_train_indices cannot be empty"
        )
    if (
        fit_indices.min() < 0
        or fit_indices.max() >= coordinates.shape[0]
    ):
        raise ValueError(
            "base_train_indices are outside labelled samples"
        )
    if np.unique(fit_indices).size != fit_indices.size:
        raise ValueError(
            "base_train_indices cannot contain duplicates"
        )

    phase_train = protocol["phase_train_indices"]
    if (
        not isinstance(phase_train, Mapping)
        or 0 not in phase_train
    ):
        raise ValueError(
            "protocol phase_train_indices must contain phase 0"
        )

    phase0_indices = _integer_array(
        phase_train[0],
        name="phase_train_indices[0]",
    ).reshape(-1)
    if (
        phase0_indices.size == 0
        or np.unique(phase0_indices).size
        != phase0_indices.size
    ):
        raise ValueError(
            "phase_train_indices[0] must be non-empty and unique"
        )
    if not np.array_equal(
        np.sort(phase0_indices),
        np.sort(fit_indices),
    ):
        raise ValueError(
            "protocol base_train_indices disagree with "
            "phase_train_indices[0]"
        )

    phase_to_classes = protocol["phase_to_classes"]
    if (
        not isinstance(phase_to_classes, Mapping)
        or 0 not in phase_to_classes
    ):
        raise ValueError(
            "protocol phase_to_classes must contain phase 0"
        )

    base_classes = [
        _exact_integer(
            value,
            name="phase-0 class ID",
            minimum=0,
        )
        for value in phase_to_classes[0]
    ]
    if (
        not base_classes
        or len(base_classes) != len(set(base_classes))
    ):
        raise ValueError(
            "phase-0 class IDs must be non-empty and unique"
        )

    observed_base_labels = set(
        int(value)
        for value in np.unique(
            global_labels[fit_indices]
        )
    )
    if observed_base_labels != set(base_classes):
        raise ValueError(
            "base_train_indices must contain exactly the phase-0 classes; "
            f"observed={sorted(observed_base_labels)}, "
            f"expected={sorted(base_classes)}"
        )

    fit_mask = BuildPreprocessFitMask(
        np.asarray(HSI_raw).shape[:2],
        coordinates,
        fit_indices,
    )
    processed, ordered, state = FitHSIPreprocessor(
        HSI_raw,
        fit_mask=fit_mask,
        **kwargs,
    )

    state["fit_scope"] = "base_train"
    ValidateHSIPreprocessorState(
        state,
        require_base_train=True,
    )

    if int(
        state["fit_pixel_count"]
    ) != int(
        fit_indices.size
    ):
        raise RuntimeError(
            "preprocessor fit count disagrees with base TRAIN split"
        )

    return processed, ordered, state


def SaveHSIPreprocessor(
    path: str,
    state: Mapping[str, Any],
) -> str:
    ValidateHSIPreprocessorState(
        state,
        require_base_train=True,
    )

    destination = os.path.abspath(path)
    os.makedirs(
        os.path.dirname(destination) or ".",
        exist_ok=True,
    )
    temporary = destination + ".tmp"

    try:
        with open(temporary, "wb") as stream:
            np.savez_compressed(
                stream,
                raw_bands=np.asarray(
                    int(state["raw_bands"])
                ),
                fit_pixel_count=np.asarray(
                    int(state["fit_pixel_count"])
                ),
                normalization_mean=np.asarray(
                    state["normalization_mean"],
                    dtype=np.float32,
                ),
                normalization_std=np.asarray(
                    state["normalization_std"],
                    dtype=np.float32,
                ),
                pca_components=np.asarray(
                    state["pca_components"],
                    dtype=np.float32,
                ),
                pca_mean=np.asarray(
                    state["pca_mean"],
                    dtype=np.float32,
                ),
                pca_variance=np.asarray(
                    state["pca_variance"],
                    dtype=np.float32,
                ),
                whiten=np.asarray(
                    _exact_bool(
                        state["whiten"],
                        name="whiten",
                    )
                ),
                fit_scope=np.asarray(
                    str(state["fit_scope"])
                ),
            )
            stream.flush()
            os.fsync(stream.fileno())

        os.replace(
            temporary,
            destination,
        )
    except Exception:
        if os.path.exists(temporary):
            os.remove(temporary)
        raise

    return destination


def LoadHSIPreprocessor(path: str) -> Dict[str, Any]:
    with np.load(os.path.abspath(path), allow_pickle=False) as archive:
        state = {
            "raw_bands": int(archive["raw_bands"].item()),
            "fit_pixel_count": int(archive["fit_pixel_count"].item()),
            "normalization_mean": archive["normalization_mean"].astype(np.float32),
            "normalization_std": archive["normalization_std"].astype(np.float32),
            "pca_components": archive["pca_components"].astype(np.float32),
            "pca_mean": archive["pca_mean"].astype(np.float32),
            "pca_variance": archive["pca_variance"].astype(np.float32),
            "whiten": bool(archive["whiten"].item()),
            "fit_scope": str(archive["fit_scope"].item()),
        }
    ValidateHSIPreprocessorState(state, require_base_train=True)
    return state


def LoadRawHSIData(
    method: str,
    base_dir: str = "./datasets",
) -> Tuple[np.ndarray, np.ndarray, int, List[str], bool, Dict[str, Any]]:
    if method not in DATASET_INFO:
        raise ValueError(f"unknown dataset {method!r}; available={sorted(DATASET_INFO)}")
    info = DATASET_INFO[method]
    data_path = os.path.join(base_dir, info["data_file"])
    gt_path = os.path.join(base_dir, info["gt_file"])
    if not os.path.isfile(data_path):
        raise FileNotFoundError(data_path)
    if not os.path.isfile(gt_path):
        raise FileNotFoundError(gt_path)
    raw = np.asarray(_load_mat_key(data_path, info["data_key"]), dtype=np.float32)
    gt = _integer_array(_load_mat_key(gt_path, info["gt_key"]), name="ground truth")
    if raw.ndim != 3 or gt.ndim != 2 or raw.shape[:2] != gt.shape:
        raise ValueError(f"incompatible HSI/GT shapes: {raw.shape}, {gt.shape}")
    if raw.shape[-1] <= 0 or not np.isfinite(raw).all():
        raise RuntimeError("HSI contains invalid values")
    policy = resolve_label_policy(method, gt)
    return (
        raw,
        gt,
        policy["num_classes"],
        policy["target_names"],
        policy["background_label"] is not None,
        policy,
    )


class HSICubeDataset(Dataset):
    """Generic pre-extracted patch dataset.

    This helper is not the class-incremental protocol dataset. NECIL/EFCIL code
    should use IncrementalHSIDatasetManager, which controls phase availability.
    """

    def __init__(self, patches: np.ndarray, labels: np.ndarray) -> None:
        x = np.ascontiguousarray(patches, dtype=np.float32)
        y = _integer_array(labels, name="labels").reshape(-1)
        if x.ndim != 4 or len(x) == 0:
            raise ValueError("patches must be non-empty rank-four data")
        if len(x) != len(y):
            raise ValueError("patch and label counts differ")
        if not np.isfinite(x).all() or np.any(y < 0):
            raise ValueError("patches/labels contain invalid values")
        self.patches = torch.from_numpy(x)
        self.labels = torch.from_numpy(y)

    def __len__(self) -> int:
        return int(self.labels.numel())

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.patches[index], self.labels[index]


def ImageCubes(
    HSI: np.ndarray,
    GT: np.ndarray,
    WS: int = 11,
    *,
    label_policy: Mapping[str, Any],
    pytorch_format: bool = True,
    **_: Any,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generic patch extraction only; it does not define a leakage-safe split."""
    cube = np.asarray(HSI, dtype=np.float32)
    ground_truth = _integer_array(GT, name="GT")
    if (
        cube.ndim != 3
        or cube.shape[2] == 0
        or ground_truth.ndim != 2
        or cube.shape[:2] != ground_truth.shape
    ):
        raise ValueError("HSI must be [H,W,B] and align with rank-two GT")
    if not np.isfinite(cube).all():
        raise ValueError("HSI contains NaN or Inf")
    labels, coords = ExtractLabeledPixelIndex(GT, label_policy=label_policy)
    size = _exact_integer(WS, name="WS", minimum=1)
    if size % 2 == 0:
        raise ValueError("WS must be odd")
    radius = size // 2
    padded = np.pad(
        cube,
        ((radius, radius), (radius, radius), (0, 0)),
        mode="reflect",
    )
    patches = np.empty(
        (len(labels), size, size, cube.shape[-1]),
        dtype=np.float32,
    )
    for index, (row, col) in enumerate(coords):
        patches[index] = padded[row : row + size, col : col + size]
    if pytorch_format:
        patches = np.moveaxis(patches, -1, 1).copy()
    return patches, labels, coords


def create_dataloader(
    patches: np.ndarray,
    labels: np.ndarray,
    *,
    batch_size: int = 64,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = False,
    **_: Any,
) -> DataLoader:
    size = _exact_integer(batch_size, name="batch_size", minimum=1)
    workers = _exact_integer(num_workers, name="num_workers", minimum=0)
    return DataLoader(
        HSICubeDataset(patches, labels),
        batch_size=size,
        shuffle=bool(shuffle),
        num_workers=workers,
        pin_memory=bool(pin_memory),
    )


__all__ = [
    "DATASET_INFO",
    "LoadRawHSIData",
    "resolve_label_policy",
    "ExtractLabeledPixelIndex",
    "BuildPreprocessFitMask",
    "FitHSIPreprocessor",
    "FitHSIPreprocessorFromProtocol",
    "ApplyHSIPreprocessor",
    "ValidateHSIPreprocessorState",
    "SaveHSIPreprocessor",
    "LoadHSIPreprocessor",
    "ImageCubes",
    "HSICubeDataset",
    "create_dataloader",
]











# from __future__ import annotations

# """HSI loading and preprocessing for class-incremental experiments.

# The loader maps foreground labels once, keeps the processed spectral-spatial
# cube consistent across phases, and exposes ordinary PyTorch dataset helpers.
# The incremental protocol decides which samples and classes are available.
# Training-only spatial symmetry augmentation is deliberately not performed here;
# the base trainer applies it after batch loading so preprocessing and all
# evaluation/reference loaders remain deterministic and unchanged.
# """

# import os
# from numbers import Integral, Real
# from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

# import numpy as np
# import scipy.io as sio
# import torch
# from torch.utils.data import DataLoader, Dataset


# DATASET_INFO: Dict[str, Dict[str, Any]] = {
#     "IP": {
#         "data_file": "Indian_pines_corrected.mat", "data_key": "indian_pines_corrected",
#         "gt_file": "Indian_pines_gt.mat", "gt_key": "indian_pines_gt", "num_classes": 16,
#         "target_names": [
#             "Alfalfa", "Corn-notill", "Corn-mintill", "Corn", "Grass-pasture",
#             "Grass-trees", "Grass-pasture-mowed", "Hay-windrowed", "Oats",
#             "Soybean-notill", "Soybean-mintill", "Soybean-clean", "Wheat", "Woods",
#             "Buildings-Grass-Trees-Drives", "Stone-Steel-Towers",
#         ],
#     },
#     "SA": {
#         "data_file": "Salinas_corrected.mat", "data_key": "salinas_corrected",
#         "gt_file": "Salinas_gt.mat", "gt_key": "salinas_gt", "num_classes": 16,
#         "target_names": [
#             "Weeds_1", "Weeds_2", "Fallow", "Fallow_rough_plow", "Fallow_smooth",
#             "Stubble", "Celery", "Grapes_untrained", "Soil_vinyard_develop",
#             "Corn_Weeds", "Lettuce_4wk", "Lettuce_5wk", "Lettuce_6wk",
#             "Lettuce_7wk", "Vinyard_untrained", "Vinyard_trellis",
#         ],
#     },
#     "PU": {
#         "data_file": "PaviaU.mat", "data_key": "paviaU", "gt_file": "PaviaU_gt.mat",
#         "gt_key": "paviaU_gt", "num_classes": 9,
#         "target_names": ["Asphalt", "Meadows", "Gravel", "Trees", "Painted", "Soil", "Bitumen", "Bricks", "Shadows"],
#     },
#     "PC": {
#         "data_file": "Pavia.mat", "data_key": "pavia", "gt_file": "Pavia_gt.mat",
#         "gt_key": "pavia_gt", "num_classes": 9,
#         "target_names": ["Water", "Trees", "Asphalt", "Bricks", "Bitumen", "Tiles", "Shadows", "Meadows", "Soil"],
#     },
#     "UH13": {
#         "data_file": "HU13.mat", "data_key": "HSI", "gt_file": "HU13_gt.mat",
#         "gt_key": "gt", "num_classes": 15,
#         "target_names": [
#             "Healthy grass", "Stressed grass", "Synthetic grass", "Trees", "Soil", "Water",
#             "Residential", "Commercial", "Road", "Highway", "Railway", "Parking Lot 1",
#             "Parking Lot 2", "Tennis Court", "Running Track",
#         ],
#     },
#     "BS": {
#         "data_file": "Botswana.mat", "data_key": "Botswana", "gt_file": "Botswana_gt.mat",
#         "gt_key": "Botswana_gt", "num_classes": 14,
#         "target_names": [
#             "Water", "Hippo Grass", "Floodplain Grasses 1", "Floodplain Grasses 2", "Reeds 1",
#             "Riparian", "Firescar 2", "Island Interior", "Woodlands", "Acacia Shrublands",
#             "Acacia Grasslands", "Short Mopane", "Mixed Mopane", "Exposed Soils",
#         ],
#     },
#     "LK": {
#         "data_file": "WHU_Hi_LongKou.mat", "data_key": "WHU_Hi_LongKou",
#         "gt_file": "WHU_Hi_LongKou_gt.mat", "gt_key": "WHU_Hi_LongKou_gt", "num_classes": 9,
#         "target_names": ["Corn", "Cotton", "Sesame", "Broad-leaf soybean", "Narrow-leaf soybean", "Rice", "Water", "Roads and houses", "Mixed weed"],
#     },
#     "HH": {
#         "data_file": "WHU_Hi_HongHu.mat", "data_key": "WHU_Hi_HongHu",
#         "gt_file": "WHU_Hi_HongHu_gt.mat", "gt_key": "WHU_Hi_HongHu_gt", "num_classes": 22,
#         "target_names": [
#             "Red roof", "Road", "Bare soil", "Cotton", "Cotton firewood", "Rape", "Chinese cabbage",
#             "Pakchoi", "Cabbage", "Tuber mustard", "Brassica parachinensis", "Brassica chinensis",
#             "Small Brassica chinensis", "Lactuca sativa", "Celtuce", "Film covered lettuce",
#             "Romaine lettuce", "Carrot", "White radish", "Garlic sprout", "Broad bean", "Tree",
#         ],
#     },
#     "HC": {
#         "data_file": "WHU_Hi_HanChuan.mat", "data_key": "WHU_Hi_HanChuan",
#         "gt_file": "WHU_Hi_HanChuan_gt.mat", "gt_key": "WHU_Hi_HanChuan_gt", "num_classes": 16,
#         "target_names": ["Strawberry", "Cowpea", "Soybean", "Sorghum", "Water spinach", "Watermelon", "Greens", "Trees", "Grass", "Red roof", "Gray roof", "Plastic", "Bare soil", "Road", "Bright object", "Water"],
#     },
#     "QUH": {
#         "data_file": "QUH-Qingyun.mat", "data_key": "Chengqu", "gt_file": "QUH-Qingyun_GT.mat",
#         "gt_key": "ChengquGT", "num_classes": 6, "background_label": None,
#         "target_names": ["Trees", "Concrete building", "Car", "Ironhide building", "Plastic playground", "Asphalt road"],
#     },
#     "PI": {
#         "data_file": "QUH-Pingan.mat", "data_key": "Haigang", "gt_file": "QUH-Pingan_GT.mat",
#         "gt_key": "HaigangGT", "num_classes": 10, "background_label": None,
#         "target_names": ["Ship", "Seawater", "Trees", "Concrete structure building", "Floating pier", "Brick houses", "Steel houses", "Wharf construction land", "Car", "Road"],
#     },
#     "TH": {
#         "data_file": "QUH-Tangdaowan.mat", "data_key": "Tangdaowan", "gt_file": "QUH-Tangdaowan_GT.mat",
#         "gt_key": "TangdaowanGT", "num_classes": 18, "background_label": None,
#         "target_names": ["Rubber track", "Flagging", "Sandy", "Asphalt", "Boardwalk", "Rocky shallows", "Grassland", "Bulrush", "Gravel road", "Ligustrum vicaryi", "Coniferous pine", "Spiraea", "Bare soil", "Buxus sinica", "Photinia serrulata", "Populus", "Ulmus pumila L", "Seawater"],
#     },
# }

# IGNORE_LABEL_VALUES = (255,)


# def _exact_integer(value: object, *, name: str, minimum: Optional[int] = None) -> int:
#     """Convert an integer-like scalar without silent truncation."""
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
#         if not np.isfinite(number) or not number.is_integer():
#             raise ValueError(f"{name} must be a finite integer")
#         result = int(number)
#     else:
#         raise ValueError(f"{name} must be an integer")
#     if minimum is not None and result < minimum:
#         raise ValueError(f"{name} must be >= {minimum}")
#     return result


# def _exact_bool(value: object, *, name: str) -> bool:
#     """Parse an explicit Boolean without bool("false") coercion."""
#     if isinstance(value, (bool, np.bool_)):
#         return bool(value)
#     if isinstance(value, Integral) and not isinstance(value, bool):
#         integer = int(value)
#         if integer in (0, 1):
#             return bool(integer)
#     if isinstance(value, str):
#         token = value.strip().lower()
#         if token in {"true", "1", "yes", "y"}:
#             return True
#         if token in {"false", "0", "no", "n"}:
#             return False
#     raise ValueError(f"{name} must be a Boolean")


# def _integer_array(value: object, *, name: str) -> np.ndarray:
#     """Return exact signed-int64 data without silent coercion or overflow."""
#     raw = np.asarray(value)
#     if raw.dtype == np.bool_ or raw.dtype.kind not in "iuf":
#         raise ValueError(f"{name} must contain integer values")

#     info = np.iinfo(np.int64)

#     if np.issubdtype(raw.dtype, np.floating):
#         if (
#             not np.isfinite(raw).all()
#             or not np.equal(raw, np.round(raw)).all()
#         ):
#             raise ValueError(f"{name} must contain finite integer values")

#         # float64 cannot exactly represent int64.max; use the exact
#         # representable half-open interval [-2**63, 2**63).
#         lower = float(-(2**63))
#         upper_exclusive = float(2**63)
#         if raw.size and (
#             np.any(raw < lower)
#             or np.any(raw >= upper_exclusive)
#         ):
#             raise ValueError(
#                 f"{name} contains values outside signed int64 range"
#             )

#     elif raw.size and np.issubdtype(raw.dtype, np.unsignedinteger):
#         if np.any(raw > info.max):
#             raise ValueError(
#                 f"{name} contains values outside signed int64 range"
#             )

#     elif raw.size:
#         if np.any(raw < info.min) or np.any(raw > info.max):
#             raise ValueError(
#                 f"{name} contains values outside signed int64 range"
#             )

#     return raw.astype(np.int64, copy=False)


# def _load_mat_key(path: str, key: str) -> np.ndarray:
#     payload = sio.loadmat(path)
#     if key not in payload:
#         available = sorted(name for name in payload if not name.startswith("__"))
#         raise KeyError(f"{key!r} not found in {path}; available keys={available}")
#     return payload[key]


# def resolve_label_policy(method: str, gt: np.ndarray) -> Dict[str, Any]:
#     if method not in DATASET_INFO:
#         raise ValueError(f"unknown dataset {method!r}; available={sorted(DATASET_INFO)}")
#     info = DATASET_INFO[method]
#     labels_map = _integer_array(gt, name="ground truth")
#     values = [
#         int(v)
#         for v in np.unique(labels_map)
#         if int(v) >= 0 and int(v) not in IGNORE_LABEL_VALUES
#     ]
#     background = info.get("background_label", 0)
#     classes = values if background is None else [v for v in values if v != background]
#     classes = sorted(classes)
#     expected = _exact_integer(info["num_classes"], name="num_classes", minimum=1)
#     if len(classes) != expected:
#         raise RuntimeError(
#             f"{method}: expected {expected} foreground classes, found {classes}"
#         )
#     names = [str(v) for v in info["target_names"]]
#     if len(names) != expected:
#         raise RuntimeError(f"{method}: target_names does not match num_classes")
#     raw_to_class = {raw_label: class_id for class_id, raw_label in enumerate(classes)}
#     return {
#         "method": method,
#         "background_label": background,
#         "raw_class_values": classes,
#         "label_to_train": raw_to_class,
#         "train_to_label": {value: key for key, value in raw_to_class.items()},
#         "num_classes": expected,
#         "target_names": names,
#     }


# def ExtractLabeledPixelIndex(
#     gt: np.ndarray,
#     *,
#     label_policy: Mapping[str, Any],
# ) -> Tuple[np.ndarray, np.ndarray]:
#     labels_map = _integer_array(gt, name="GT")
#     if labels_map.ndim != 2:
#         raise ValueError("GT must have shape [H,W]")
#     raw_classes = _integer_array(
#         label_policy["raw_class_values"], name="raw_class_values"
#     ).reshape(-1)
#     rows, cols = np.where(np.isin(labels_map, raw_classes))
#     coords = np.stack((rows, cols), axis=1).astype(np.int64, copy=False)
#     if np.unique(coords, axis=0).shape[0] != coords.shape[0]:
#         raise RuntimeError("label extraction produced duplicate coordinates")
#     mapping = {
#         _exact_integer(k, name="raw label"): _exact_integer(v, name="class ID", minimum=0)
#         for k, v in label_policy["label_to_train"].items()
#     }
#     labels = np.asarray(
#         [mapping[int(v)] for v in labels_map[rows, cols]], dtype=np.int64
#     )
#     return labels, coords


# def BuildPreprocessFitMask(
#     gt_shape: Sequence[int],
#     coords: np.ndarray,
#     sample_indices: Sequence[int],
#     **_: Any,
# ) -> np.ndarray:
#     shape = tuple(_exact_integer(v, name="gt_shape value", minimum=1) for v in gt_shape)
#     coordinates = _integer_array(coords, name="coords")
#     indices = _integer_array(sample_indices, name="sample_indices").reshape(-1)
#     if len(shape) != 2 or coordinates.ndim != 2 or coordinates.shape[1] != 2:
#         raise ValueError("gt_shape and coords are invalid")
#     if coordinates.shape[0] == 0:
#         raise ValueError("coords cannot be empty")
#     if np.unique(coordinates, axis=0).shape[0] != coordinates.shape[0]:
#         raise ValueError("coords must identify unique labelled pixels")
#     if (
#         np.any(coordinates[:, 0] < 0)
#         or np.any(coordinates[:, 0] >= shape[0])
#         or np.any(coordinates[:, 1] < 0)
#         or np.any(coordinates[:, 1] >= shape[1])
#     ):
#         raise ValueError("coords lie outside gt_shape")
#     if indices.size == 0 or indices.min() < 0 or indices.max() >= len(coordinates):
#         raise ValueError("sample_indices must select valid labelled pixels")
#     if np.unique(indices).size != indices.size:
#         raise ValueError("sample_indices cannot contain duplicates")
#     mask = np.zeros(shape, dtype=bool)
#     selected = coordinates[indices]
#     mask[selected[:, 0], selected[:, 1]] = True
#     if int(mask.sum()) != int(indices.size):
#         raise RuntimeError("preprocess fit mask lost selected training pixels")
#     return mask


# def _fit_preprocessor(
#     cube: np.ndarray,
#     fit_mask: np.ndarray,
#     *,
#     n_components: Optional[int],
#     whiten: bool,
# ) -> Dict[str, Any]:
#     raw = np.asarray(cube, dtype=np.float32)
#     mask = np.asarray(fit_mask, dtype=bool)

#     if (
#         raw.ndim != 3
#         or raw.shape[2] == 0
#         or not np.isfinite(raw).all()
#         or mask.shape != raw.shape[:2]
#         or not mask.any()
#     ):
#         raise ValueError(
#             "cube must be finite [H,W,B] and fit_mask must select pixels"
#         )

#     pixels = raw[mask]
#     mean = pixels.mean(axis=0).astype(np.float32)
#     observed_std = pixels.std(axis=0, ddof=0)
#     std = np.where(
#         observed_std > 0.0,
#         observed_std,
#         1.0,
#     ).astype(np.float32)

#     whiten_value = _exact_bool(
#         whiten,
#         name="whiten",
#     )

#     state: Dict[str, Any] = {
#         "raw_bands": int(raw.shape[-1]),
#         "normalization_mean": mean,
#         "normalization_std": std,
#         "fit_pixel_count": int(pixels.shape[0]),
#         "pca_components": np.empty(
#             (0, raw.shape[-1]),
#             dtype=np.float32,
#         ),
#         "pca_mean": np.empty(0, dtype=np.float32),
#         "pca_variance": np.empty(0, dtype=np.float32),
#         "whiten": whiten_value,
#         "fit_scope": "provided_mask",
#     }

#     if n_components is not None:
#         component_count = _exact_integer(
#             n_components,
#             name="n_components",
#             minimum=1,
#         )

#         maximum_components = min(
#             int(raw.shape[-1]),
#             int(pixels.shape[0]),
#         )
#         if component_count > maximum_components:
#             raise ValueError(
#                 "n_components cannot exceed min(raw_bands, "
#                 f"base_train_fit_pixels)={maximum_components}; "
#                 f"got {component_count}"
#             )
#         if pixels.shape[0] < 2:
#             raise ValueError(
#                 "PCA requires at least two base-TRAIN fit pixels"
#             )

#         from sklearn.decomposition import PCA

#         normalized_pixels = (
#             pixels - mean
#         ) / std

#         pca = PCA(
#             n_components=component_count,
#             whiten=False,
#             svd_solver="auto",
#         ).fit(normalized_pixels)

#         state["pca_components"] = (
#             pca.components_.astype(np.float32)
#         )
#         state["pca_mean"] = (
#             pca.mean_.astype(np.float32)
#         )
#         state["pca_variance"] = (
#             pca.explained_variance_.astype(np.float32)
#         )

#     return state


# def ValidateHSIPreprocessorState(
#     state: Mapping[str, Any],
#     *,
#     require_base_train: bool = False,
# ) -> bool:
#     required = {
#         "raw_bands",
#         "fit_pixel_count",
#         "normalization_mean",
#         "normalization_std",
#         "pca_components",
#         "pca_mean",
#         "pca_variance",
#         "whiten",
#         "fit_scope",
#     }
#     missing = required - set(state)
#     if missing:
#         raise ValueError(
#             f"preprocessor state is missing keys {sorted(missing)}"
#         )

#     bands = _exact_integer(
#         state["raw_bands"],
#         name="raw_bands",
#         minimum=1,
#     )
#     fit_pixel_count = _exact_integer(
#         state["fit_pixel_count"],
#         name="fit_pixel_count",
#         minimum=1,
#     )
#     whiten = _exact_bool(
#         state["whiten"],
#         name="whiten",
#     )

#     mean = np.asarray(
#         state["normalization_mean"],
#         dtype=np.float32,
#     ).reshape(-1)
#     std = np.asarray(
#         state["normalization_std"],
#         dtype=np.float32,
#     ).reshape(-1)
#     components = np.asarray(
#         state["pca_components"],
#         dtype=np.float32,
#     )
#     pca_mean = np.asarray(
#         state["pca_mean"],
#         dtype=np.float32,
#     ).reshape(-1)
#     variance = np.asarray(
#         state["pca_variance"],
#         dtype=np.float32,
#     ).reshape(-1)

#     if mean.shape != (bands,) or std.shape != (bands,):
#         raise ValueError(
#             "normalization state has incompatible dimensions"
#         )
#     if (
#         not np.isfinite(mean).all()
#         or not np.isfinite(std).all()
#         or np.any(std <= 0.0)
#     ):
#         raise ValueError(
#             "normalization state contains invalid values"
#         )

#     scope = str(state["fit_scope"]).strip()
#     if scope not in {
#         "provided_mask",
#         "base_train",
#     }:
#         raise ValueError(
#             "preprocessor fit_scope is invalid"
#         )
#     if require_base_train and scope != "base_train":
#         raise ValueError(
#             "persistent HSI preprocessing must be fitted on base_train only"
#         )

#     if (
#         components.ndim != 2
#         or components.shape[1] != bands
#     ):
#         raise ValueError(
#             "PCA components have an incompatible shape"
#         )
#     if not np.isfinite(components).all():
#         raise ValueError(
#             "PCA components contain NaN/Inf"
#         )

#     component_count = int(
#         components.shape[0]
#     )

#     if component_count == 0:
#         if pca_mean.size != 0 or variance.size != 0:
#             raise ValueError(
#                 "PCA-disabled state must have empty pca_mean and pca_variance"
#             )
#         if whiten:
#             raise ValueError(
#                 "whitening cannot be enabled without PCA components"
#             )
#         return True

#     if component_count > min(
#         bands,
#         fit_pixel_count,
#     ):
#         raise ValueError(
#             "PCA component count exceeds the dimensions of its fit data"
#         )

#     if pca_mean.shape != (bands,):
#         raise ValueError(
#             "PCA mean has incompatible dimensions"
#         )
#     if variance.shape != (component_count,):
#         raise ValueError(
#             "PCA variance has incompatible dimensions"
#         )
#     if (
#         not np.isfinite(pca_mean).all()
#         or not np.isfinite(variance).all()
#         or np.any(variance < 0.0)
#     ):
#         raise ValueError(
#             "PCA state contains invalid values"
#         )
#     if whiten and np.any(variance <= 0.0):
#         raise ValueError(
#             "whitening requires strictly positive PCA variance"
#         )

#     return True


# def ApplyHSIPreprocessor(
#     HSI_raw: np.ndarray,
#     state: Mapping[str, Any],
# ) -> Tuple[np.ndarray, np.ndarray]:
#     ValidateHSIPreprocessorState(state)
#     raw = np.asarray(HSI_raw, dtype=np.float32)
#     if raw.ndim != 3 or raw.shape[-1] != int(state["raw_bands"]):
#         raise ValueError("raw cube does not match preprocessing state")
#     if not np.isfinite(raw).all():
#         raise ValueError("raw cube contains NaN or Inf")

#     mean = np.asarray(state["normalization_mean"], dtype=np.float32)
#     std = np.asarray(state["normalization_std"], dtype=np.float32)
#     processed = (raw - mean) / std

#     components = np.asarray(state["pca_components"], dtype=np.float32)
#     if components.size:
#         flat = processed.reshape(-1, processed.shape[-1])
#         transformed = (
#             flat - np.asarray(state["pca_mean"], dtype=np.float32)
#         ) @ components.T
#         if bool(state["whiten"]):
#             transformed /= np.sqrt(
#                 np.asarray(state["pca_variance"], dtype=np.float32)
#             )
#         processed = transformed.reshape(raw.shape[0], raw.shape[1], -1)

#     processed = np.asarray(processed, dtype=np.float32)
#     if not np.isfinite(processed).all():
#         raise RuntimeError("preprocessing produced NaN/Inf")

#     # The second cube intentionally remains the original ordered full-band HSI.
#     # The spectral backbone applies the same base-train band statistics itself.
#     return np.ascontiguousarray(processed), np.ascontiguousarray(raw)


# def FitHSIPreprocessor(
#     HSI_raw: np.ndarray,
#     *,
#     fit_mask: Optional[np.ndarray] = None,
#     apply_reduction: bool = True,
#     reduction_method: str = "PCA",
#     n_components: int = 30,
#     whiten: bool = False,
#     **_: Any,
# ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
#     use_reduction = _exact_bool(
#         apply_reduction,
#         name="apply_reduction",
#     )
#     whiten_value = _exact_bool(
#         whiten,
#         name="whiten",
#     )
#     if (
#         use_reduction
#         and str(reduction_method).strip().lower() != "pca"
#     ):
#         raise ValueError(
#             "only PCA reduction is supported"
#         )
#     if whiten_value and not use_reduction:
#         raise ValueError(
#             "whiten=True requires PCA reduction"
#         )

#     raw = np.asarray(HSI_raw, dtype=np.float32)
#     if raw.ndim != 3 or raw.shape[-1] <= 0 or not np.isfinite(raw).all():
#         raise ValueError("HSI_raw must be finite [H,W,B]")
#     if fit_mask is None:
#         raise ValueError(
#             "fit_mask is required; preprocessing statistics must be fitted "
#             "on base-train centres only"
#         )
#     mask = np.asarray(fit_mask, dtype=bool)
#     state = _fit_preprocessor(
#         raw,
#         mask,
#         n_components=(
#             _exact_integer(n_components, name="n_components", minimum=1)
#             if use_reduction
#             else None
#         ),
#         whiten=whiten_value,
#     )
#     processed, ordered = ApplyHSIPreprocessor(raw, state)
#     return processed, ordered, state


# def FitHSIPreprocessorFromProtocol(
#     HSI_raw: np.ndarray,
#     *,
#     coords: np.ndarray,
#     protocol: Mapping[str, Any],
#     **kwargs: Any,
# ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
#     """Fit persistent preprocessing from exactly phase-0 TRAIN centres.

#     This validation happens before fitting because preprocessing precedes
#     IncrementalHSIDatasetManager construction.  A malformed protocol therefore
#     cannot leak future/validation/test rows into normalization or PCA and only
#     fail later when the manager is created.
#     """
#     required = {
#         "base_train_indices",
#         "phase_train_indices",
#         "phase_to_classes",
#         "global_labels",
#     }
#     missing = required - set(protocol)
#     if missing:
#         raise ValueError(
#             "protocol is missing preprocessing-critical fields "
#             f"{sorted(missing)}"
#         )

#     coordinates = _integer_array(
#         coords,
#         name="coords",
#     )
#     if (
#         coordinates.ndim != 2
#         or coordinates.shape[1] != 2
#     ):
#         raise ValueError(
#             "coords must have shape [N,2]"
#         )

#     global_labels = _integer_array(
#         protocol["global_labels"],
#         name="protocol global_labels",
#     ).reshape(-1)
#     if global_labels.shape != (
#         coordinates.shape[0],
#     ):
#         raise ValueError(
#             "protocol global_labels must align with coords"
#         )
#     if bool((global_labels < 0).any()):
#         raise ValueError(
#             "protocol global_labels must be non-negative"
#         )

#     fit_indices = _integer_array(
#         protocol["base_train_indices"],
#         name="base_train_indices",
#     ).reshape(-1)
#     if fit_indices.size == 0:
#         raise ValueError(
#             "base_train_indices cannot be empty"
#         )
#     if (
#         fit_indices.min() < 0
#         or fit_indices.max() >= coordinates.shape[0]
#     ):
#         raise ValueError(
#             "base_train_indices are outside labelled samples"
#         )
#     if np.unique(fit_indices).size != fit_indices.size:
#         raise ValueError(
#             "base_train_indices cannot contain duplicates"
#         )

#     phase_train = protocol["phase_train_indices"]
#     if (
#         not isinstance(phase_train, Mapping)
#         or 0 not in phase_train
#     ):
#         raise ValueError(
#             "protocol phase_train_indices must contain phase 0"
#         )

#     phase0_indices = _integer_array(
#         phase_train[0],
#         name="phase_train_indices[0]",
#     ).reshape(-1)
#     if (
#         phase0_indices.size == 0
#         or np.unique(phase0_indices).size
#         != phase0_indices.size
#     ):
#         raise ValueError(
#             "phase_train_indices[0] must be non-empty and unique"
#         )
#     if not np.array_equal(
#         np.sort(phase0_indices),
#         np.sort(fit_indices),
#     ):
#         raise ValueError(
#             "protocol base_train_indices disagree with "
#             "phase_train_indices[0]"
#         )

#     phase_to_classes = protocol["phase_to_classes"]
#     if (
#         not isinstance(phase_to_classes, Mapping)
#         or 0 not in phase_to_classes
#     ):
#         raise ValueError(
#             "protocol phase_to_classes must contain phase 0"
#         )

#     base_classes = [
#         _exact_integer(
#             value,
#             name="phase-0 class ID",
#             minimum=0,
#         )
#         for value in phase_to_classes[0]
#     ]
#     if (
#         not base_classes
#         or len(base_classes) != len(set(base_classes))
#     ):
#         raise ValueError(
#             "phase-0 class IDs must be non-empty and unique"
#         )

#     observed_base_labels = set(
#         int(value)
#         for value in np.unique(
#             global_labels[fit_indices]
#         )
#     )
#     if observed_base_labels != set(base_classes):
#         raise ValueError(
#             "base_train_indices must contain exactly the phase-0 classes; "
#             f"observed={sorted(observed_base_labels)}, "
#             f"expected={sorted(base_classes)}"
#         )

#     fit_mask = BuildPreprocessFitMask(
#         np.asarray(HSI_raw).shape[:2],
#         coordinates,
#         fit_indices,
#     )
#     processed, ordered, state = FitHSIPreprocessor(
#         HSI_raw,
#         fit_mask=fit_mask,
#         **kwargs,
#     )

#     state["fit_scope"] = "base_train"
#     ValidateHSIPreprocessorState(
#         state,
#         require_base_train=True,
#     )

#     if int(
#         state["fit_pixel_count"]
#     ) != int(
#         fit_indices.size
#     ):
#         raise RuntimeError(
#             "preprocessor fit count disagrees with base TRAIN split"
#         )

#     return processed, ordered, state


# def SaveHSIPreprocessor(
#     path: str,
#     state: Mapping[str, Any],
# ) -> str:
#     ValidateHSIPreprocessorState(
#         state,
#         require_base_train=True,
#     )

#     destination = os.path.abspath(path)
#     os.makedirs(
#         os.path.dirname(destination) or ".",
#         exist_ok=True,
#     )
#     temporary = destination + ".tmp"

#     try:
#         with open(temporary, "wb") as stream:
#             np.savez_compressed(
#                 stream,
#                 raw_bands=np.asarray(
#                     int(state["raw_bands"])
#                 ),
#                 fit_pixel_count=np.asarray(
#                     int(state["fit_pixel_count"])
#                 ),
#                 normalization_mean=np.asarray(
#                     state["normalization_mean"],
#                     dtype=np.float32,
#                 ),
#                 normalization_std=np.asarray(
#                     state["normalization_std"],
#                     dtype=np.float32,
#                 ),
#                 pca_components=np.asarray(
#                     state["pca_components"],
#                     dtype=np.float32,
#                 ),
#                 pca_mean=np.asarray(
#                     state["pca_mean"],
#                     dtype=np.float32,
#                 ),
#                 pca_variance=np.asarray(
#                     state["pca_variance"],
#                     dtype=np.float32,
#                 ),
#                 whiten=np.asarray(
#                     _exact_bool(
#                         state["whiten"],
#                         name="whiten",
#                     )
#                 ),
#                 fit_scope=np.asarray(
#                     str(state["fit_scope"])
#                 ),
#             )
#             stream.flush()
#             os.fsync(stream.fileno())

#         os.replace(
#             temporary,
#             destination,
#         )
#     except Exception:
#         if os.path.exists(temporary):
#             os.remove(temporary)
#         raise

#     return destination


# def LoadHSIPreprocessor(path: str) -> Dict[str, Any]:
#     with np.load(os.path.abspath(path), allow_pickle=False) as archive:
#         state = {
#             "raw_bands": int(archive["raw_bands"].item()),
#             "fit_pixel_count": int(archive["fit_pixel_count"].item()),
#             "normalization_mean": archive["normalization_mean"].astype(np.float32),
#             "normalization_std": archive["normalization_std"].astype(np.float32),
#             "pca_components": archive["pca_components"].astype(np.float32),
#             "pca_mean": archive["pca_mean"].astype(np.float32),
#             "pca_variance": archive["pca_variance"].astype(np.float32),
#             "whiten": bool(archive["whiten"].item()),
#             "fit_scope": str(archive["fit_scope"].item()),
#         }
#     ValidateHSIPreprocessorState(state, require_base_train=True)
#     return state


# def LoadRawHSIData(
#     method: str,
#     base_dir: str = "./datasets",
# ) -> Tuple[np.ndarray, np.ndarray, int, List[str], bool, Dict[str, Any]]:
#     if method not in DATASET_INFO:
#         raise ValueError(f"unknown dataset {method!r}; available={sorted(DATASET_INFO)}")
#     info = DATASET_INFO[method]
#     data_path = os.path.join(base_dir, info["data_file"])
#     gt_path = os.path.join(base_dir, info["gt_file"])
#     if not os.path.isfile(data_path):
#         raise FileNotFoundError(data_path)
#     if not os.path.isfile(gt_path):
#         raise FileNotFoundError(gt_path)
#     raw = np.asarray(_load_mat_key(data_path, info["data_key"]), dtype=np.float32)
#     gt = _integer_array(_load_mat_key(gt_path, info["gt_key"]), name="ground truth")
#     if raw.ndim != 3 or gt.ndim != 2 or raw.shape[:2] != gt.shape:
#         raise ValueError(f"incompatible HSI/GT shapes: {raw.shape}, {gt.shape}")
#     if raw.shape[-1] <= 0 or not np.isfinite(raw).all():
#         raise RuntimeError("HSI contains invalid values")
#     policy = resolve_label_policy(method, gt)
#     return (
#         raw,
#         gt,
#         policy["num_classes"],
#         policy["target_names"],
#         policy["background_label"] is not None,
#         policy,
#     )


# class HSICubeDataset(Dataset):
#     """Generic pre-extracted patch dataset.

#     This helper is not the class-incremental protocol dataset. NECIL/EFCIL code
#     should use IncrementalHSIDatasetManager, which controls phase availability.
#     """

#     def __init__(self, patches: np.ndarray, labels: np.ndarray) -> None:
#         x = np.ascontiguousarray(patches, dtype=np.float32)
#         y = _integer_array(labels, name="labels").reshape(-1)
#         if x.ndim != 4 or len(x) == 0:
#             raise ValueError("patches must be non-empty rank-four data")
#         if len(x) != len(y):
#             raise ValueError("patch and label counts differ")
#         if not np.isfinite(x).all() or np.any(y < 0):
#             raise ValueError("patches/labels contain invalid values")
#         self.patches = torch.from_numpy(x)
#         self.labels = torch.from_numpy(y)

#     def __len__(self) -> int:
#         return int(self.labels.numel())

#     def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
#         return self.patches[index], self.labels[index]


# def ImageCubes(
#     HSI: np.ndarray,
#     GT: np.ndarray,
#     WS: int = 11,
#     *,
#     label_policy: Mapping[str, Any],
#     pytorch_format: bool = True,
#     **_: Any,
# ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
#     """Generic patch extraction only; it does not define a leakage-safe split."""
#     cube = np.asarray(HSI, dtype=np.float32)
#     ground_truth = _integer_array(GT, name="GT")
#     if (
#         cube.ndim != 3
#         or cube.shape[2] == 0
#         or ground_truth.ndim != 2
#         or cube.shape[:2] != ground_truth.shape
#     ):
#         raise ValueError("HSI must be [H,W,B] and align with rank-two GT")
#     if not np.isfinite(cube).all():
#         raise ValueError("HSI contains NaN or Inf")
#     labels, coords = ExtractLabeledPixelIndex(GT, label_policy=label_policy)
#     size = _exact_integer(WS, name="WS", minimum=1)
#     if size % 2 == 0:
#         raise ValueError("WS must be odd")
#     radius = size // 2
#     padded = np.pad(
#         cube,
#         ((radius, radius), (radius, radius), (0, 0)),
#         mode="reflect",
#     )
#     patches = np.empty(
#         (len(labels), size, size, cube.shape[-1]),
#         dtype=np.float32,
#     )
#     for index, (row, col) in enumerate(coords):
#         patches[index] = padded[row : row + size, col : col + size]
#     if pytorch_format:
#         patches = np.moveaxis(patches, -1, 1).copy()
#     return patches, labels, coords


# def create_dataloader(
#     patches: np.ndarray,
#     labels: np.ndarray,
#     *,
#     batch_size: int = 64,
#     shuffle: bool = True,
#     num_workers: int = 0,
#     pin_memory: bool = False,
#     **_: Any,
# ) -> DataLoader:
#     size = _exact_integer(batch_size, name="batch_size", minimum=1)
#     workers = _exact_integer(num_workers, name="num_workers", minimum=0)
#     return DataLoader(
#         HSICubeDataset(patches, labels),
#         batch_size=size,
#         shuffle=bool(shuffle),
#         num_workers=workers,
#         pin_memory=bool(pin_memory),
#     )


# __all__ = [
#     "DATASET_INFO",
#     "LoadRawHSIData",
#     "resolve_label_policy",
#     "ExtractLabeledPixelIndex",
#     "BuildPreprocessFitMask",
#     "FitHSIPreprocessor",
#     "FitHSIPreprocessorFromProtocol",
#     "ApplyHSIPreprocessor",
#     "ValidateHSIPreprocessorState",
#     "SaveHSIPreprocessor",
#     "LoadHSIPreprocessor",
#     "ImageCubes",
#     "HSICubeDataset",
#     "create_dataloader",
# ]
