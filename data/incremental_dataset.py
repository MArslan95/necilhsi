from __future__ import annotations

"""Single leakage-auditable non-exemplar class-incremental HSI protocol.

Protocol contract
-----------------
* Phase t training exposes only the training centres of classes introduced at t.
* Validation during an active phase exposes only current-phase validation data.
* Cumulative validation/test over all seen classes is evaluation-only and is
  available only after the phase is finalized.
* The spatial branch receives the processed/PCA patch.
* The spectral branch receives the pre-PCA ordered centre spectrum from the
  dataset cube; corrected datasets may already exclude sensor bands.
* Those two inputs are fused by the backbone; the dataset does not create a
  second learned representation.
* No optimization/model-selection API returns historical training samples after its phase.
* A finalized-phase map loader may read all labelled seen-class pixels strictly for
  post-training qualitative reporting; those rows are never an optimization source.
* Optional strict spatial audits detect patch-footprint overlap across data
  splits and across unavailable old/future labelled samples.

Important: a standard random-pixel HSI split is centre-disjoint but generally
NOT patch-disjoint when patch_size > 1. Therefore it must not be described as
spatial-leakage-free unless the strict audits actually pass.
"""

import random
from numbers import Integral, Real
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


_SPLITS = ("train", "val", "test")


def _exact_integer(value: object, *, name: str, minimum: Optional[int] = None) -> int:
    """Return an exact signed-int64-compatible scalar without silent truncation."""
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

    info = np.iinfo(np.int64)
    if result < int(info.min) or result > int(info.max):
        raise ValueError(f"{name} is outside signed int64 range")
    if minimum is not None and result < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return result


def _exact_bool(value: object, *, name: str) -> bool:
    """Parse an explicit Boolean without Python's bool('false') pitfall."""
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
    """Return exact int64 data without Boolean, fractional, or overflow coercion."""
    raw = np.asarray(value)
    if raw.dtype == np.bool_ or raw.dtype.kind not in "iuf":
        raise ValueError(f"{name} must contain integer values")

    info = np.iinfo(np.int64)
    if np.issubdtype(raw.dtype, np.floating):
        if not np.isfinite(raw).all() or not np.equal(raw, np.round(raw)).all():
            raise ValueError(f"{name} must contain finite integer values")
        # float64 cannot represent int64.max exactly; use the exact
        # representable half-open interval [-2**63, 2**63).
        lower = float(-(2**63))
        upper_exclusive = float(2**63)
        if raw.size and (
            np.any(raw < lower) or np.any(raw >= upper_exclusive)
        ):
            raise ValueError(f"{name} contains values outside signed int64 range")
    elif raw.size and np.issubdtype(raw.dtype, np.unsignedinteger):
        if np.any(raw > info.max):
            raise ValueError(f"{name} contains values outside signed int64 range")
    elif raw.size:
        if np.any(raw < info.min) or np.any(raw > info.max):
            raise ValueError(f"{name} contains values outside signed int64 range")

    return raw.astype(np.int64, copy=False)


def _ids(values: Iterable[int], *, name: str, allow_empty: bool = False) -> List[int]:
    result = [
        _exact_integer(value, name=f"{name} value", minimum=0)
        for value in values
    ]
    if not result and not allow_empty:
        raise ValueError(f"{name} cannot be empty")
    if len(result) != len(set(result)):
        raise ValueError(f"{name} must contain unique IDs")
    return result


def _normalize_split_name(split: str) -> str:
    token = str(split).strip().lower()
    return {
        "training": "train",
        "validation": "val",
        "testing": "test",
    }.get(token, token)


_SPECTRAL_AXIS_ALIASES = {
    "wavelength": "wavelength_nm",
    "wavelengths": "wavelength_nm",
    "wavelength_nm": "wavelength_nm",
    "wavelengths_nm": "wavelength_nm",
    "nm": "wavelength_nm",
    "band": "ordered_band",
    "bands": "ordered_band",
    "band_index": "ordered_band",
    "ordered_band": "ordered_band",
}


def _validated_spectral_axis(
    spectral_axis: Optional[Sequence[float]],
    *,
    raw_bands: int,
    spectral_axis_type: str,
) -> Tuple[np.ndarray, str]:
    """Validate the one spectral coordinate shared by the model and CDS."""
    bands = _exact_integer(raw_bands, name="raw_bands", minimum=1)
    token = _SPECTRAL_AXIS_ALIASES.get(
        str(spectral_axis_type).strip().lower()
    )
    if token is None:
        raise ValueError(
            "spectral_axis_type must be 'wavelength_nm' or 'ordered_band'"
        )
    if spectral_axis is None:
        if token != "ordered_band":
            raise ValueError(
                "physical wavelength coordinates must be supplied explicitly; "
                "they are never synthesized from band indices"
            )
        axis = np.arange(bands, dtype=np.float32)
    else:
        axis = np.asarray(spectral_axis, dtype=np.float64).reshape(-1)
    if axis.shape != (bands,):
        raise ValueError(
            "spectral_axis must contain one coordinate per ordered raw band"
        )
    if not np.isfinite(axis).all():
        raise ValueError("spectral_axis contains NaN/Inf")
    if bands > 1 and not np.all(np.diff(axis) > 0.0):
        raise ValueError("spectral_axis must be strictly increasing")
    if token == "wavelength_nm" and np.any(axis <= 0.0):
        raise ValueError("physical wavelengths in nanometres must be positive")
    return axis.astype(np.float32), token


def _seed_worker(_: int) -> None:
    seed = int(torch.initial_seed() % (2**32))
    random.seed(seed)
    np.random.seed(seed)



def _seed32(value: object, *, name: str) -> int:
    """Validate the exact integer domain accepted by NumPy RandomState."""
    result = _exact_integer(
        value,
        name=name,
        minimum=0,
    )
    if result > (2**32) - 1:
        raise ValueError(
            f"{name} must be <= 2**32-1"
        )
    return result


def _derived_seed32(base_seed: int, component: int) -> int:
    """Derive a deterministic per-class seed without leaving uint32 range."""
    base = _seed32(
        base_seed,
        name="base seed",
    )
    part = _exact_integer(
        component,
        name="seed component",
        minimum=0,
    )
    return int(
        (base + 1009 * part)
        % (2**32)
    )

def _phase_schedule(
    class_count: int,
    base_classes: int,
    increment: int,
) -> Tuple[List[int], Dict[int, List[int]]]:
    total = _exact_integer(class_count, name="class_count", minimum=1)
    base = _exact_integer(base_classes, name="base_classes", minimum=1)
    step = _exact_integer(increment, name="increment", minimum=1)
    if base > total:
        raise ValueError("base_classes cannot exceed class_count")
    sizes = [base]
    remaining = total - base
    while remaining:
        current = min(step, remaining)
        sizes.append(current)
        remaining -= current
    phases: Dict[int, List[int]] = {}
    start = 0
    for phase, size in enumerate(sizes):
        phases[phase] = list(range(start, start + size))
        start += size
    return sizes, phases


def _split_counts(
    total: int,
    *,
    train_ratio: float,
    val_ratio: float,
) -> Tuple[int, int, int]:
    """Allocate one fixed per-class split; validation may be disabled.

    Train and test are mandatory.  Validation is optional because some
    published HSI benchmarks provide train/test masks only.
    """
    train_fraction = float(train_ratio)
    val_fraction = float(val_ratio)

    if (
        not np.isfinite(train_fraction)
        or not np.isfinite(val_fraction)
        or train_fraction <= 0.0
        or val_fraction < 0.0
        or train_fraction + val_fraction >= 1.0
    ):
        raise ValueError(
            "require finite train_ratio>0, val_ratio>=0, and "
            "train_ratio+val_ratio<1"
        )

    required_split_count = 3 if val_fraction > 0.0 else 2
    count = _exact_integer(
        total,
        name="class sample count",
        minimum=required_split_count,
    )

    quotas = np.asarray(
        (
            train_fraction,
            val_fraction,
            1.0 - train_fraction - val_fraction,
        ),
        dtype=np.float64,
    ) * count
    counts = np.floor(quotas).astype(np.int64)
    remainder = count - int(counts.sum())
    if remainder:
        order = np.argsort(-(quotas - counts), kind="stable")
        counts[order[:remainder]] += 1

    required = (0, 2) if val_fraction == 0.0 else (0, 1, 2)
    for destination in required:
        if counts[destination] == 0:
            candidates = [
                index
                for index in required
                if index != destination and counts[index] > 1
            ]
            if not candidates:
                raise ValueError(
                    "class is too small to realize the requested non-empty splits"
                )
            source = max(candidates, key=lambda index: int(counts[index]))
            counts[source] -= 1
            counts[destination] += 1

    if val_fraction == 0.0:
        # A zero validation ratio is an explicit train/test-only protocol.
        if counts[1] != 0:
            counts[2] += counts[1]
            counts[1] = 0

    n_train, n_val, n_test = (int(v) for v in counts)
    if n_train + n_val + n_test != count:
        raise RuntimeError("split allocation does not conserve class samples")
    if n_train <= 0 or n_test <= 0:
        raise RuntimeError("train and test splits must be non-empty")
    if val_fraction > 0.0 and n_val <= 0:
        raise RuntimeError("positive val_ratio produced an empty validation split")
    return n_train, n_val, n_test

def _random_class_split(
    indices: np.ndarray,
    *,
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> Dict[str, np.ndarray]:
    values = _integer_array(indices, name="class indices").reshape(-1)
    if np.unique(values).size != values.size:
        raise ValueError("class indices contain duplicates")
    n_train, n_val, _ = _split_counts(
        len(values),
        train_ratio=train_ratio,
        val_ratio=val_ratio,
    )
    rng = np.random.RandomState(_exact_integer(seed, name="split seed", minimum=0))
    shuffled = rng.permutation(values)
    train = shuffled[:n_train]
    val = shuffled[n_train : n_train + n_val]
    test = shuffled[n_train + n_val :]
    return {
        "train": train,
        "val": val,
        "test": test,
        "all": np.concatenate((train, val, test)),
    }


def _spatial_class_split(
    indices: np.ndarray,
    coords: np.ndarray,
    *,
    train_ratio: float,
    val_ratio: float,
    block_size: int,
    seed: int,
) -> Dict[str, np.ndarray]:
    """Classwise spatial-block split.

    This is a benchmark split option, not a proof of patch disjointness.
    Patch-overlap audits are computed separately.
    """
    values = _integer_array(indices, name="class indices").reshape(-1)
    xy = _integer_array(coords, name="coords")[values]
    size = _exact_integer(block_size, name="spatial_block_size", minimum=1)
    max_block_col = int(np.max(xy[:, 1] // size)) + 1
    block_ids = (xy[:, 0] // size) * max_block_col + xy[:, 1] // size
    blocks = [values[block_ids == block] for block in np.unique(block_ids)]

    validation_enabled = float(val_ratio) > 0.0
    minimum_blocks = 3 if validation_enabled else 2
    if len(blocks) < minimum_blocks:
        raise RuntimeError(
            "a class occupies too few spatial blocks for the requested split"
        )

    rng = np.random.RandomState(
        _exact_integer(seed, name="split seed", minimum=0)
    )
    rng.shuffle(blocks)

    target_train, target_val, _ = _split_counts(
        len(values),
        train_ratio=train_ratio,
        val_ratio=val_ratio,
    )

    train_blocks: List[np.ndarray] = []
    val_blocks: List[np.ndarray] = []
    test_blocks: List[np.ndarray] = []
    train_count = 0
    val_count = 0

    for block_index, block in enumerate(blocks):
        remaining = len(blocks) - block_index
        reserve_for_test = 1
        reserve_for_val = 1 if validation_enabled and not val_blocks else 0

        if (
            train_count < target_train
            and remaining > reserve_for_test + reserve_for_val
        ):
            train_blocks.append(block)
            train_count += len(block)
        elif (
            validation_enabled
            and val_count < target_val
            and remaining > reserve_for_test
        ):
            val_blocks.append(block)
            val_count += len(block)
        else:
            test_blocks.append(block)

    if not train_blocks or not test_blocks:
        raise RuntimeError("spatial block allocation produced an empty train/test split")
    if validation_enabled and not val_blocks:
        raise RuntimeError("spatial block allocation produced an empty validation split")

    train = np.concatenate(train_blocks)
    val = (
        np.concatenate(val_blocks)
        if val_blocks
        else np.empty(0, dtype=np.int64)
    )
    test = np.concatenate(test_blocks)

    return {
        "train": train,
        "val": val,
        "test": test,
        "all": np.concatenate((train, val, test)),
    }

def _validate_split_by_class(
    split_by_class: Mapping[int, Mapping[str, np.ndarray]],
    labels: np.ndarray,
    class_count: int,
) -> None:
    y = _integer_array(
        labels,
        name="global labels",
    ).reshape(-1)
    total_classes = _exact_integer(
        class_count,
        name="class_count",
        minimum=1,
    )

    if y.size == 0:
        raise ValueError(
            "global labels cannot be empty"
        )
    observed_classes = sorted(
        int(value)
        for value in np.unique(y)
    )
    if observed_classes != list(
        range(total_classes)
    ):
        raise ValueError(
            "global labels must contain exactly contiguous class IDs "
            f"0..{total_classes - 1}; observed={observed_classes}"
        )

    if set(int(v) for v in split_by_class) != set(range(total_classes)):
        raise ValueError("split_by_class does not cover every class")

    used: set[int] = set()
    for class_id in range(total_classes):
        row = split_by_class[class_id]
        missing = set((*_SPLITS, "all")) - set(row)
        if missing:
            raise ValueError(
                f"class {class_id} split is missing {sorted(missing)}"
            )

        arrays = {
            name: _integer_array(
                row[name],
                name=f"class {class_id} {name} indices",
            ).reshape(-1)
            for name in (*_SPLITS, "all")
        }

        if arrays["train"].size == 0 or arrays["test"].size == 0:
            raise ValueError(
                f"class {class_id} must have non-empty train and test splits"
            )
        if arrays["all"].size == 0:
            raise ValueError(f"class {class_id} all split cannot be empty")

        for name, array in arrays.items():
            if np.unique(array).size != array.size:
                raise ValueError(
                    f"class {class_id} {name} split contains duplicates"
                )
            if array.size and (array.min() < 0 or array.max() >= y.size):
                raise ValueError(
                    f"class {class_id} {name} split contains invalid indices"
                )

        train_set, val_set, test_set = (
            set(arrays[name].tolist()) for name in _SPLITS
        )
        if train_set & val_set or train_set & test_set or val_set & test_set:
            raise ValueError(f"class {class_id} train/val/test indices overlap")

        union = train_set | val_set | test_set
        if set(arrays["all"].tolist()) != union:
            raise ValueError(f"class {class_id} all indices are inconsistent")

        for array in arrays.values():
            if array.size and not np.all(y[array] == class_id):
                raise ValueError(f"class {class_id} split contains another class")

        expected = set(np.flatnonzero(y == class_id).tolist())
        if union != expected:
            raise ValueError(
                f"class {class_id} split does not account for every sample"
            )
        if used & expected:
            raise ValueError("a sample appears in more than one class")
        used |= expected

def _collect_indices(
    split_by_class: Mapping[int, Mapping[str, np.ndarray]],
    class_ids: Sequence[int],
    split: str,
) -> np.ndarray:
    token = _normalize_split_name(split)
    if token not in {*_SPLITS, "all"}:
        raise ValueError(f"unknown split {split!r}")
    ids = _ids(class_ids, name="class_ids", allow_empty=True)
    if not ids:
        return np.empty(0, dtype=np.int64)
    return np.concatenate(
        [
            _integer_array(
                split_by_class[class_id][token],
                name=f"class {class_id} {token} indices",
            ).reshape(-1)
            for class_id in ids
        ]
    )


def build_incremental_protocol(
    input_labels: np.ndarray,
    input_coords: np.ndarray,
    *,
    gt_shape: Sequence[int],
    class_count: int,
    base_classes: int,
    increment: int,
    train_ratio: float,
    val_ratio: float,
    seed: int = 0,
    shuffle_order: bool = False,
    class_order: Optional[Sequence[int]] = None,
    target_names: Optional[Sequence[str]] = None,
    split_strategy: str = "random_pixel",
    spatial_block_size: int = 33,
    predefined_splits: Optional[Mapping[int, Mapping[str, Sequence[int]]]] = None,
    **_: Any,
) -> Dict[str, Any]:
    original_labels = _integer_array(input_labels, name="input_labels").reshape(-1)
    coords = _integer_array(input_coords, name="input_coords")
    shape = tuple(
        _exact_integer(v, name="gt_shape value", minimum=1) for v in gt_shape
    )
    total_classes = _exact_integer(class_count, name="class_count", minimum=1)
    protocol_seed = _seed32(
        seed,
        name="seed",
    )

    if len(shape) != 2 or coords.shape != (len(original_labels), 2):
        raise ValueError("labels, coordinates and gt_shape are inconsistent")
    if original_labels.size == 0:
        raise ValueError("input_labels cannot be empty")
    if np.unique(coords, axis=0).shape[0] != coords.shape[0]:
        raise ValueError("coords must identify unique labelled pixels")
    if (
        np.any(coords[:, 0] < 0)
        or np.any(coords[:, 0] >= shape[0])
        or np.any(coords[:, 1] < 0)
        or np.any(coords[:, 1] >= shape[1])
    ):
        raise ValueError("coords lie outside gt_shape")

    original_ids = sorted(int(v) for v in np.unique(original_labels))
    if original_ids != list(range(total_classes)):
        raise ValueError("input labels must be contiguous zero-based class IDs")

    shuffle_classes = _exact_bool(shuffle_order, name="shuffle_order")
    if class_order is not None and shuffle_classes:
        raise ValueError("choose either class_order or shuffle_order, not both")
    if class_order is not None:
        order = _ids(class_order, name="class_order")
    elif shuffle_classes:
        order = np.random.RandomState(protocol_seed).permutation(
            original_ids
        ).tolist()
    else:
        order = original_ids
    if set(order) != set(original_ids) or len(order) != len(original_ids):
        raise ValueError("class_order must be a complete class permutation")

    original_to_global = {
        original: global_id for global_id, original in enumerate(order)
    }
    global_to_original = {
        global_id: original for original, global_id in original_to_global.items()
    }
    global_labels = np.asarray(
        [original_to_global[int(label)] for label in original_labels],
        dtype=np.int64,
    )
    phase_sizes, phase_to_classes = _phase_schedule(
        total_classes, base_classes, increment
    )

    strategy = str(split_strategy).strip().lower()
    if predefined_splits is not None:
        predefined_original = {
            _exact_integer(class_id, name="predefined class ID", minimum=0): row
            for class_id, row in predefined_splits.items()
        }
        if set(predefined_original) != set(original_ids):
            raise ValueError(
                "predefined_splits must be keyed by every original class ID"
            )
        split_by_class = {}
        for original_id, row in predefined_original.items():
            if "train" not in row or "test" not in row:
                raise ValueError(
                    f"predefined class {original_id} must provide train and test indices"
                )
            mapped = {
                "train": _integer_array(
                    row["train"],
                    name=f"predefined class {original_id} train indices",
                ).reshape(-1),
                "val": _integer_array(
                    row.get("val", np.empty(0, dtype=np.int64)),
                    name=f"predefined class {original_id} val indices",
                ).reshape(-1),
                "test": _integer_array(
                    row["test"],
                    name=f"predefined class {original_id} test indices",
                ).reshape(-1),
            }
            mapped["all"] = np.concatenate(
                (mapped["train"], mapped["val"], mapped["test"])
            )
            split_by_class[original_to_global[original_id]] = mapped
        strategy = "predefined"
    elif strategy == "random_pixel":
        split_by_class = {
            class_id: _random_class_split(
                np.flatnonzero(global_labels == class_id),
                train_ratio=train_ratio,
                val_ratio=val_ratio,
                seed=_derived_seed32(
                    protocol_seed,
                    global_to_original[class_id],
                ),
            )
            for class_id in range(total_classes)
        }
    elif strategy == "spatial_block":
        split_by_class = {
            class_id: _spatial_class_split(
                np.flatnonzero(global_labels == class_id),
                coords,
                train_ratio=train_ratio,
                val_ratio=val_ratio,
                block_size=spatial_block_size,
                seed=_derived_seed32(
                    protocol_seed,
                    global_to_original[class_id],
                ),
            )
            for class_id in range(total_classes)
        }
    else:
        raise ValueError(
            "split_strategy must be random_pixel, spatial_block, or predefined"
        )

    _validate_split_by_class(split_by_class, global_labels, total_classes)

    validation_presence = [
        bool(split_by_class[class_id]["val"].size)
        for class_id in range(total_classes)
    ]
    if any(validation_presence) and not all(validation_presence):
        raise ValueError(
            "validation must be consistently enabled for every class or "
            "disabled for every class"
        )
    validation_enabled = all(validation_presence)

    names = [] if target_names is None else [str(v) for v in target_names]
    if names and len(names) != total_classes:
        raise ValueError("target_names must contain one name per class")
    global_names = [names[original] for original in order] if names else []

    phase_train = {
        phase: _collect_indices(split_by_class, classes, "train")
        for phase, classes in phase_to_classes.items()
    }
    phase_val = {
        phase: _collect_indices(split_by_class, classes, "val")
        for phase, classes in phase_to_classes.items()
    }
    phase_test = {
        phase: _collect_indices(split_by_class, classes, "test")
        for phase, classes in phase_to_classes.items()
    }
    cumulative_test = {
        phase: _collect_indices(
            split_by_class,
            list(range(sum(phase_sizes[: phase + 1]))),
            "test",
        )
        for phase in phase_to_classes
    }

    return {
        "training_mode": "non_exemplar_class_incremental",
        "seed": protocol_seed,
        "class_count": total_classes,
        "base_classes": _exact_integer(
            base_classes, name="base_classes", minimum=1
        ),
        "increment": _exact_integer(increment, name="increment", minimum=1),
        "phase_sizes": phase_sizes,
        "phase_to_classes": phase_to_classes,
        "class_order_original_ids": order,
        "original_to_global": original_to_global,
        "global_to_original": global_to_original,
        "original_labels": original_labels,
        "global_labels": global_labels,
        "global_target_names": global_names,
        "split_by_class": split_by_class,
        "phase_train_indices": phase_train,
        "phase_val_indices": phase_val,
        "phase_test_indices": phase_test,
        "cumulative_test_indices": cumulative_test,
        # Canonical preprocessing membership. This is exactly phase-0 TRAIN,
        # not a second independently constructed split.
        "base_train_indices": phase_train[0].copy(),
        "split_strategy": strategy,
        "spatial_partition_mode": (
            "classwise_block_assignment"
            if strategy == "spatial_block"
            else "pixel_indices"
        ),
        "single_head_global_labels": True,
        "current_phase_training_classes_only": True,
        "cumulative_seen_class_evaluation": True,
        "stores_old_training_samples": False,
        "validation_enabled": bool(validation_enabled),
        "preprocessing_fit_phase": 0,
        "preprocessing_frozen_across_phases": True,
    }


class IncrementalHSIPatchDataset(Dataset):
    def __init__(
        self,
        manager: "IncrementalHSIDatasetManager",
        indices: Sequence[int],
    ) -> None:
        self.manager = manager
        self.indices = _integer_array(indices, name="dataset indices").reshape(-1)
        if self.indices.size == 0:
            raise ValueError("dataset indices cannot be empty")
        if self.indices.min() < 0 or self.indices.max() >= len(manager.labels):
            raise ValueError("dataset indices are outside labelled samples")
        if np.unique(self.indices).size != self.indices.size:
            raise ValueError("dataset indices cannot contain duplicates")

    def __len__(self) -> int:
        return int(self.indices.size)

    def __getitem__(self, item: int) -> Dict[str, torch.Tensor]:
        item_id = _exact_integer(item, name="dataset item", minimum=0)
        if item_id >= len(self):
            raise IndexError(item_id)
        sample_index = int(self.indices[item_id])
        row, col = self.manager.coords[sample_index]
        patch = self.manager.extract_patch(int(row), int(col))
        raw_center_spectrum = self.manager.ordered_spectral_cube[
            int(row), int(col)
        ]
        return {
            "image": torch.from_numpy(
                np.moveaxis(patch, -1, 0).copy()
            ).float(),
            "raw_center_spectrum": torch.from_numpy(
                raw_center_spectrum.copy()
            ).float(),
            "label": torch.tensor(
                int(self.manager.labels[sample_index]),
                dtype=torch.long,
            ),
            "coord": torch.tensor(
                (int(row), int(col)),
                dtype=torch.long,
            ),
            "sample_index": torch.tensor(sample_index, dtype=torch.long),
        }


class IncrementalHSIDatasetManager:
    """Current-class training and seen-class evaluation with spatial audits."""

    def __init__(
        self,
        *,
        processed_cube: np.ndarray,
        labels: np.ndarray,
        coords: np.ndarray,
        protocol: Mapping[str, Any],
        target_names: Sequence[str],
        gt_shape: Sequence[int],
        patch_size: int,
        ordered_spectral_cube: Optional[np.ndarray] = None,
        spectral_axis: Optional[Sequence[float]] = None,
        spectral_axis_type: str = "ordered_band",
        num_workers: int = 0,
        device: str = "cpu",
        seed: Optional[int] = None,
        require_patch_disjoint: bool = False,
        require_phase_context_isolation: Optional[bool] = None,
        context_policy: str = "full_scene_reflect",
        **_: Any,
    ) -> None:
        self.processed_cube = np.ascontiguousarray(
            processed_cube, dtype=np.float32
        )
        self.gt_shape = tuple(
            _exact_integer(v, name="gt_shape value", minimum=1)
            for v in gt_shape
        )
        self.coords = _integer_array(coords, name="coords")
        supplied_labels = _integer_array(labels, name="labels").reshape(-1)
        original_labels = _integer_array(
            protocol["original_labels"], name="protocol original_labels"
        ).reshape(-1)
        global_labels = _integer_array(
            protocol["global_labels"], name="protocol global labels"
        ).reshape(-1)
        if (
            np.array_equal(
                supplied_labels,
                original_labels,
            )
            or np.array_equal(
                supplied_labels,
                global_labels,
            )
        ):
            self.labels = global_labels
        else:
            raise ValueError(
                "labels do not match the protocol"
            )

        protocol_class_count = _exact_integer(
            protocol["class_count"],
            name="protocol class_count",
            minimum=1,
        )
        observed_global_classes = sorted(
            int(value)
            for value in np.unique(self.labels)
        )
        if observed_global_classes != list(
            range(protocol_class_count)
        ):
            raise ValueError(
                "protocol global labels must contain exactly contiguous "
                f"class IDs 0..{protocol_class_count - 1}; "
                f"observed={observed_global_classes}"
            )

        if (
            self.processed_cube.ndim != 3
            or self.processed_cube.shape[:2] != self.gt_shape
            or self.processed_cube.shape[2] <= 0
            or not np.isfinite(self.processed_cube).all()
        ):
            raise ValueError("processed_cube must be finite and match gt_shape")
        if self.coords.shape != (len(self.labels), 2):
            raise ValueError("coords must align with labels")
        if np.unique(self.coords, axis=0).shape[0] != self.coords.shape[0]:
            raise ValueError("coords contain duplicate labelled-pixel locations")
        if (
            np.any(self.coords[:, 0] < 0)
            or np.any(self.coords[:, 0] >= self.gt_shape[0])
            or np.any(self.coords[:, 1] < 0)
            or np.any(self.coords[:, 1] >= self.gt_shape[1])
        ):
            raise ValueError("coords contain positions outside gt_shape")

        if ordered_spectral_cube is None:
            raise ValueError(
                "ordered_spectral_cube is required for the raw spectral branch"
            )
        ordered = np.asarray(ordered_spectral_cube, dtype=np.float32)
        if (
            ordered.ndim != 3
            or ordered.shape[:2] != self.gt_shape
            or ordered.shape[2] <= 0
            or not np.isfinite(ordered).all()
        ):
            raise ValueError("ordered_spectral_cube is invalid")
        self.ordered_spectral_cube = np.ascontiguousarray(ordered)
        self.spectral_axis, self.spectral_axis_type = _validated_spectral_axis(
            spectral_axis,
            raw_bands=int(self.ordered_spectral_cube.shape[-1]),
            spectral_axis_type=spectral_axis_type,
        )
        self.spectral_axis_is_physical = (
            self.spectral_axis_type == "wavelength_nm"
        )

        self.patch_size = _exact_integer(
            patch_size, name="patch_size", minimum=1
        )
        if self.patch_size % 2 == 0:
            raise ValueError("patch_size must be odd")
        radius = self.patch_size // 2
        self.padded_cube = np.pad(
            self.processed_cube,
            ((radius, radius), (radius, radius), (0, 0)),
            mode="reflect",
        )

        self.phase_to_classes = {
            _exact_integer(phase, name="phase ID", minimum=0): _ids(
                classes, name=f"phase_to_classes[{phase}]"
            )
            for phase, classes in protocol["phase_to_classes"].items()
        }
        if sorted(self.phase_to_classes) != list(range(len(self.phase_to_classes))):
            raise ValueError("phase IDs must be contiguous from zero")
        class_count = protocol_class_count
        flattened = [
            class_id
            for phase in range(len(self.phase_to_classes))
            for class_id in self.phase_to_classes[phase]
        ]
        if flattened != list(range(class_count)):
            raise ValueError(
                "phase schedule must cover global class IDs once in order"
            )

        # The protocol must expose one canonical base-training membership for
        # preprocessing, and it must be exactly the phase-0 training split.
        if "base_train_indices" not in protocol:
            raise ValueError("protocol is missing base_train_indices")
        if "phase_train_indices" not in protocol:
            raise ValueError("protocol is missing phase_train_indices")
        phase_train_indices = protocol["phase_train_indices"]
        if not isinstance(phase_train_indices, Mapping) or 0 not in phase_train_indices:
            raise ValueError("protocol phase_train_indices must contain phase 0")
        base_train_indices = _integer_array(
            protocol["base_train_indices"],
            name="protocol base_train_indices",
        ).reshape(-1)
        phase0_train_indices = _integer_array(
            phase_train_indices[0],
            name="protocol phase_train_indices[0]",
        ).reshape(-1)
        if (
            base_train_indices.size == 0
            or phase0_train_indices.size == 0
            or not np.array_equal(
                np.sort(base_train_indices),
                np.sort(phase0_train_indices),
            )
        ):
            raise ValueError(
                "protocol base_train_indices must equal phase_train_indices[0]"
            )
        self.base_train_indices = base_train_indices.copy()

        self._split_by_class = {
            _exact_integer(class_id, name="split class ID", minimum=0): {
                name: _integer_array(
                    values, name=f"class {class_id} {name} indices"
                )
                for name, values in row.items()
            }
            for class_id, row in protocol["split_by_class"].items()
        }
        _validate_split_by_class(
            self._split_by_class, self.labels, class_count
        )

        # Validate every protocol membership field against the single canonical
        # split_by_class state.  This prevents preprocessing/training/reporting
        # from silently consuming different sample memberships.
        for field, split_name in (
            ("phase_train_indices", "train"),
            ("phase_val_indices", "val"),
            ("phase_test_indices", "test"),
        ):
            raw = protocol.get(field)
            if not isinstance(raw, Mapping):
                raise ValueError(f"protocol {field} must be a phase mapping")
            if set(int(key) for key in raw) != set(self.phase_to_classes):
                raise ValueError(
                    f"protocol {field} must contain every phase exactly once"
                )

            for phase_id, classes in self.phase_to_classes.items():
                declared = _integer_array(
                    raw[phase_id],
                    name=f"protocol {field}[{phase_id}]",
                ).reshape(-1)
                expected = _collect_indices(
                    self._split_by_class,
                    classes,
                    split_name,
                )
                if not np.array_equal(
                    np.sort(declared),
                    np.sort(expected),
                ):
                    raise ValueError(
                        f"protocol {field}[{phase_id}] disagrees with split_by_class"
                    )

        cumulative_test = protocol.get("cumulative_test_indices")
        if not isinstance(cumulative_test, Mapping):
            raise ValueError(
                "protocol cumulative_test_indices must be a phase mapping"
            )
        if set(int(key) for key in cumulative_test) != set(self.phase_to_classes):
            raise ValueError(
                "protocol cumulative_test_indices must contain every phase"
            )
        for phase_id in self.phase_to_classes:
            declared = _integer_array(
                cumulative_test[phase_id],
                name=f"protocol cumulative_test_indices[{phase_id}]",
            ).reshape(-1)
            expected = _collect_indices(
                self._split_by_class,
                self.get_seen_classes(phase_id),
                "test",
            )
            if not np.array_equal(
                np.sort(declared),
                np.sort(expected),
            ):
                raise ValueError(
                    "protocol cumulative_test_indices "
                    f"[{phase_id}] disagrees with split_by_class"
                )

        preprocessing_fit_phase = _exact_integer(
            protocol.get("preprocessing_fit_phase", 0),
            name="protocol preprocessing_fit_phase",
            minimum=0,
        )
        if preprocessing_fit_phase != 0:
            raise ValueError("persistent HSI preprocessing must be fitted in phase 0")

        preprocessing_frozen = _exact_bool(
            protocol.get("preprocessing_frozen_across_phases", True),
            name="protocol preprocessing_frozen_across_phases",
        )
        if not preprocessing_frozen:
            raise ValueError(
                "persistent HSI preprocessing must remain frozen across phases"
            )

        names = [str(v) for v in target_names]
        global_names = [
            str(v) for v in protocol.get("global_target_names", [])
        ]
        if global_names and len(global_names) != class_count:
            raise ValueError("protocol global_target_names has wrong length")
        if not global_names and len(names) != class_count:
            raise ValueError("target_names must contain one name per class")
        self.class_order_original_ids = _ids(
            protocol["class_order_original_ids"],
            name="class_order_original_ids",
        )
        if set(self.class_order_original_ids) != set(range(class_count)):
            raise ValueError(
                "class_order_original_ids must be a complete original-class permutation"
            )
        self.target_names = global_names or [
            names[index] for index in self.class_order_original_ids
        ]

        self.global_to_original = {
            _exact_integer(k, name="global class ID", minimum=0):
            _exact_integer(v, name="original class ID", minimum=0)
            for k, v in protocol["global_to_original"].items()
        }
        expected_global_to_original = {
            global_id: original_id
            for global_id, original_id in enumerate(self.class_order_original_ids)
        }
        if self.global_to_original != expected_global_to_original:
            raise ValueError(
                "global_to_original disagrees with class_order_original_ids"
            )

        self.split_strategy = str(protocol["split_strategy"]).strip().lower()
        self.spatial_partition_mode = str(
            protocol.get("spatial_partition_mode", "pixel_indices")
        )

        validation_presence = [
            bool(self._split_by_class[class_id]["val"].size)
            for class_id in range(class_count)
        ]
        if any(validation_presence) and not all(validation_presence):
            raise ValueError(
                "validation must be present for every class or absent for every class"
            )
        actual_validation_enabled = all(validation_presence)
        declared_validation_enabled = _exact_bool(
            protocol.get("validation_enabled", actual_validation_enabled),
            name="protocol validation_enabled",
        )
        if declared_validation_enabled != actual_validation_enabled:
            raise ValueError(
                "protocol validation_enabled disagrees with validation split membership"
            )
        self.validation_enabled = actual_validation_enabled

        self.require_patch_disjoint = _exact_bool(
            require_patch_disjoint,
            name="require_patch_disjoint",
        )
        self.require_phase_context_isolation = (
            self.require_patch_disjoint
            if require_phase_context_isolation is None
            else _exact_bool(
                require_phase_context_isolation,
                name="require_phase_context_isolation",
            )
        )
        self.context_policy = str(context_policy).strip()
        if self.context_policy != "full_scene_reflect":
            raise ValueError(
                "only context_policy='full_scene_reflect' is implemented"
            )

        # Protocol randomness and optimization/DataLoader randomness are
        # intentionally separate. ``protocol_seed`` fixes class order and
        # train/val/test membership; ``self.seed`` controls only DataLoader
        # shuffling/worker randomness for this optimization run.  Multi-seed
        # robustness therefore supports, for example, protocol_seed=42 with
        # optimization seeds 42/123/70 without changing the benchmark split.
        self.protocol_seed = _seed32(
            protocol["seed"],
            name="protocol seed",
        )
        self.seed = (
            self.protocol_seed
            if seed is None
            else _seed32(seed, name="dataset-manager optimization seed")
        )

        self.num_workers = _exact_integer(
            num_workers, name="num_workers", minimum=0
        )
        self.pin_memory = str(device).startswith("cuda")
        self.current_phase: Optional[int] = None
        self.finalized_phases: List[int] = []

        self._split_patch_disjoint = self._audit_split_patch_disjoint()
        self._phase_context_isolated = self._audit_phase_context_isolation()

        if self.require_patch_disjoint and not self._split_patch_disjoint:
            raise RuntimeError(
                "train/val/test patch footprints overlap. A random-pixel centre "
                "split is not spatial-leakage-free for patch models. Use a "
                "guarded/predefined spatial split or explicitly disable the "
                "strict audit only for benchmark-comparability experiments."
            )
        if (
            self.require_phase_context_isolation
            and not self._phase_context_isolated
        ):
            raise RuntimeError(
                "at least one phase-training patch contains an annotated labeled "
                "center unavailable in that phase (old, future, validation, or test). "
                "This violates strict spatial phase isolation."
            )

    @property
    def nb_tasks(self) -> int:
        return len(self.phase_to_classes)

    @property
    def base_classes(self) -> List[int]:
        return list(self.phase_to_classes[0])

    def _validate_phase(self, phase: int) -> int:
        value = _exact_integer(phase, name="phase", minimum=0)
        if value not in self.phase_to_classes:
            raise ValueError(f"unknown phase {phase}")
        return value

    def start_phase(self, phase: int) -> None:
        value = self._validate_phase(phase)
        if self.current_phase is not None:
            raise RuntimeError(f"phase {self.current_phase} is already active")
        if value in self.finalized_phases:
            raise RuntimeError(f"phase {value} is already finalized")
        if self.finalized_phases != list(range(value)):
            raise RuntimeError(
                "phases must start in order after prior finalization"
            )
        self.current_phase = value

    def finalize_phase(self, phase: int) -> None:
        value = self._validate_phase(phase)
        if self.current_phase != value:
            raise RuntimeError(f"phase {value} is not the active phase")
        self.finalized_phases.append(value)
        self.current_phase = None

    def restore_finalized_phases(
        self,
        phases: Sequence[int],
    ) -> None:
        """Restore checkpoint protocol state without exposing historical data."""
        if self.current_phase is not None:
            raise RuntimeError(
                "cannot restore finalized phases while a phase is active"
            )
        if self.finalized_phases:
            raise RuntimeError(
                "restore_finalized_phases requires a fresh unfinalized manager"
            )
        values = [
            self._validate_phase(value)
            for value in phases
        ]
        if values != list(range(len(values))):
            raise ValueError(
                "restored finalized phases must be contiguous from zero"
            )
        self.finalized_phases = values

    def get_task_size(self, phase: int) -> int:
        return len(self.phase_to_classes[self._validate_phase(phase)])

    def get_new_classes(self, phase: int) -> List[int]:
        return list(self.phase_to_classes[self._validate_phase(phase)])

    def get_old_classes(self, phase: int) -> List[int]:
        value = self._validate_phase(phase)
        return [
            class_id
            for previous in range(value)
            for class_id in self.phase_to_classes[previous]
        ]

    def get_seen_classes(self, phase: int) -> List[int]:
        return self.get_old_classes(phase) + self.get_new_classes(phase)

    def extract_patch(self, row: int, col: int) -> np.ndarray:
        row_id = _exact_integer(row, name="patch row", minimum=0)
        col_id = _exact_integer(col, name="patch col", minimum=0)
        if row_id >= self.gt_shape[0] or col_id >= self.gt_shape[1]:
            raise ValueError("patch centre lies outside gt_shape")
        return self.padded_cube[
            row_id : row_id + self.patch_size,
            col_id : col_id + self.patch_size,
        ]

    def _indices_mask(self, indices: np.ndarray) -> np.ndarray:
        mask = np.zeros(self.gt_shape, dtype=bool)
        values = _integer_array(indices, name="audit indices").reshape(-1)
        if values.size:
            if values.min() < 0 or values.max() >= len(self.labels):
                raise ValueError("audit indices are outside labelled samples")
            if np.unique(values).size != values.size:
                raise ValueError("audit indices contain duplicates")
            xy = self.coords[values]
            mask[xy[:, 0], xy[:, 1]] = True
        return mask

    def _patch_overlap_count(
        self,
        left_indices: np.ndarray,
        right_indices: np.ndarray,
    ) -> int:
        """Count left centers whose extracted patch overlaps a right patch."""
        left = _integer_array(left_indices, name="left indices").reshape(-1)
        right = _integer_array(right_indices, name="right indices").reshape(-1)
        if left.size == 0 or right.size == 0:
            return 0

        # Two odd P x P patches overlap when center Chebyshev distance <= P-1.
        from scipy.ndimage import maximum_filter

        right_mask = self._indices_mask(right)
        radius = self.patch_size - 1
        dilated = (
            right_mask
            if radius == 0
            else maximum_filter(
                right_mask.astype(np.uint8),
                size=2 * radius + 1,
                mode="constant",
                cval=0,
            ).astype(bool)
        )
        xy = self.coords[left]
        return int(dilated[xy[:, 0], xy[:, 1]].sum())

    def _forbidden_center_in_patch_count(
        self,
        train_indices: np.ndarray,
        forbidden_indices: np.ndarray,
    ) -> int:
        """Count train patches containing an unavailable labeled center.

        This differs from patch-vs-patch overlap. During phase-t training the
        unavailable old/future/validation/test sample is not itself extracted as
        a patch. The strict observable leakage event is that its labeled center
        spectrum lies inside the current train patch.
        """
        train = _integer_array(train_indices, name="train indices").reshape(-1)
        forbidden = _integer_array(
            forbidden_indices, name="forbidden indices"
        ).reshape(-1)
        if train.size == 0 or forbidden.size == 0:
            return 0

        from scipy.ndimage import maximum_filter

        forbidden_mask = self._indices_mask(forbidden)
        radius = self.patch_size // 2
        dilated = (
            forbidden_mask
            if radius == 0
            else maximum_filter(
                forbidden_mask.astype(np.uint8),
                size=2 * radius + 1,
                mode="constant",
                cval=0,
            ).astype(bool)
        )
        xy = self.coords[train]
        return int(dilated[xy[:, 0], xy[:, 1]].sum())

    def _audit_split_patch_disjoint(self) -> bool:
        all_classes = list(range(len(self._split_by_class)))
        split_indices = {
            split: _collect_indices(
                self._split_by_class, all_classes, split
            )
            for split in _SPLITS
        }
        for left_index, left_name in enumerate(_SPLITS):
            for right_name in _SPLITS[left_index + 1 :]:
                if self._patch_overlap_count(
                    split_indices[left_name],
                    split_indices[right_name],
                ):
                    return False
        return True

    def _audit_phase_context_isolation(self) -> bool:
        all_indices = np.arange(len(self.labels), dtype=np.int64)
        for phase in range(self.nb_tasks):
            available = _collect_indices(
                self._split_by_class,
                self.phase_to_classes[phase],
                "train",
            )
            available_set = np.zeros(len(self.labels), dtype=bool)
            available_set[available] = True
            forbidden = all_indices[~available_set]
            if self._forbidden_center_in_patch_count(available, forbidden):
                return False
        return True

    def spatial_leakage_report(self) -> Dict[str, Any]:
        return {
            "patch_size": self.patch_size,
            "split_patch_disjoint": bool(self._split_patch_disjoint),
            "phase_training_context_isolated": bool(
                self._phase_context_isolated
            ),
            "strict_split_required": self.require_patch_disjoint,
            "strict_phase_context_required": self.require_phase_context_isolation,
            "context_policy": self.context_policy,
            "guarantee_scope": (
                "train/val/test extracted patch footprints plus unavailable annotated "
                "labeled centers inside phase-training patches; unlabeled/background "
                "scene pixels cannot be assigned a task identity by the benchmark"
            ),
        }

    def _make_dataset(
        self,
        class_ids: Sequence[int],
        split: str,
    ) -> IncrementalHSIPatchDataset:
        return IncrementalHSIPatchDataset(
            self,
            _collect_indices(self._split_by_class, class_ids, split),
        )

    def _loader(
        self,
        dataset: Dataset,
        *,
        batch_size: int,
        shuffle: bool,
        phase: int,
        split: str,
    ) -> DataLoader:
        size = _exact_integer(batch_size, name="batch_size", minimum=1)
        phase_id = self._validate_phase(phase)
        offset = {"train": 11, "val": 17, "test": 23, "all": 29}[split]
        generator = torch.Generator().manual_seed(
            self.seed + 1009 * phase_id + offset
        )
        return DataLoader(
            dataset,
            batch_size=size,
            shuffle=_exact_bool(shuffle, name="shuffle"),
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=False,
            persistent_workers=self.num_workers > 0,
            worker_init_fn=_seed_worker if self.num_workers > 0 else None,
            generator=generator,
        )

    def get_phase_dataloader(
        self,
        phase: int,
        split: str = "train",
        batch_size: int = 64,
        shuffle: bool = True,
    ) -> DataLoader:
        phase_id = self._validate_phase(phase)
        if self.current_phase != phase_id:
            raise RuntimeError(
                "phase train/validation loaders require an active phase"
            )
        token = _normalize_split_name(split)
        if token not in {"train", "val"}:
            raise ValueError("active phase loaders are train/val only")
        if token == "val" and not self.validation_enabled:
            raise RuntimeError("validation is disabled by this experiment protocol")
        dataset = self._make_dataset(
            self.get_new_classes(phase_id), token
        )
        return self._loader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle if token == "train" else False,
            phase=phase_id,
            split=token,
        )

    def get_phase_structure_dataloader(
        self,
        phase: int,
        *,
        batch_size: int = 256,
    ) -> DataLoader:
        """Deterministic full current-TRAIN loader for structure estimation.

        This deliberately reuses the active phase TRAIN membership, exposes no
        historical rows, disables shuffling, and therefore provides the single
        legal source for epoch refresh/final commit of current-class
        I/V/Mspec/Vspec.
        """
        return self.get_phase_dataloader(
            phase,
            split="train",
            batch_size=batch_size,
            shuffle=False,
        )

    def get_cumulative_dataloader(
        self,
        phase: int,
        split: str = "test",
        batch_size: int = 256,
        shuffle: bool = False,
    ) -> DataLoader:
        phase_id = self._validate_phase(phase)
        if phase_id not in self.finalized_phases:
            raise RuntimeError(
                "cumulative evaluation requires a finalized phase"
            )
        token = _normalize_split_name(split)
        if token not in {"val", "test"}:
            raise ValueError("cumulative loaders are evaluation-only")
        if token == "val" and not self.validation_enabled:
            raise RuntimeError("validation is disabled by this experiment protocol")
        dataset = self._make_dataset(
            self.get_seen_classes(phase_id), token
        )
        return self._loader(
            dataset,
            batch_size=batch_size,
            shuffle=bool(shuffle),
            phase=phase_id,
            split=token,
        )

    def get_reporting_dataloader(
        self,
        phase: int,
        *,
        batch_size: int = 256,
    ) -> DataLoader:
        """Test-only reporting loader. Never mixes train/val samples into metrics."""
        return self.get_cumulative_dataloader(
            phase,
            split="test",
            batch_size=batch_size,
            shuffle=False,
        )

    def get_map_dataloader(
        self,
        phase: int,
        *,
        batch_size: int = 256,
    ) -> DataLoader:
        """All labeled seen-class pixels for qualitative maps only; never metrics/training."""
        phase_id = self._validate_phase(phase)
        if phase_id not in self.finalized_phases:
            raise RuntimeError("map generation requires a finalized phase")
        dataset = self._make_dataset(
            self.get_seen_classes(phase_id), "all"
        )
        return self._loader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            phase=phase_id,
            split="all",
        )

    def base_phase_contract_report(self) -> Dict[str, Any]:
        """Return read-only checks for the finalized phase-0 data contract."""
        base_ids = self.get_new_classes(0)
        train_indices = _collect_indices(
            self._split_by_class,
            base_ids,
            "train",
        )
        val_indices = _collect_indices(
            self._split_by_class,
            base_ids,
            "val",
        )
        test_indices = _collect_indices(
            self._split_by_class,
            base_ids,
            "test",
        )
        if not np.array_equal(
            np.sort(train_indices),
            np.sort(self.base_train_indices),
        ):
            raise RuntimeError(
                "base TRAIN membership disagrees with preprocessing membership"
            )
        if (
            np.intersect1d(train_indices, val_indices).size
            or np.intersect1d(train_indices, test_indices).size
            or np.intersect1d(val_indices, test_indices).size
        ):
            raise RuntimeError("base train/val/test centre indices overlap")
        observed = sorted(
            int(v)
            for v in np.unique(self.labels[train_indices])
        )
        if observed != sorted(base_ids):
            raise RuntimeError(
                "base TRAIN split does not contain exactly the base classes"
            )
        return {
            "base_class_ids": list(base_ids),
            "train_count": int(train_indices.size),
            "validation_count": int(val_indices.size),
            "test_count": int(test_indices.size),
            "preprocess_fit_count": int(self.base_train_indices.size),
            "train_equals_preprocess_fit": True,
            "center_splits_disjoint": True,
            "validation_enabled": bool(self.validation_enabled),
            "spatial_leakage": self.spatial_leakage_report(),
        }

    def protocol_report(self) -> Dict[str, Any]:
        return {
            "training_mode": "non_exemplar_class_incremental",
            "protocol_seed": int(self.protocol_seed),
            "optimization_dataloader_seed": int(self.seed),
            "tasks": self.nb_tasks,
            "phase_to_classes": {
                phase: list(classes)
                for phase, classes in self.phase_to_classes.items()
            },
            "class_order_original_ids": list(self.class_order_original_ids),
            "split_strategy": self.split_strategy,
            "spatial_partition_mode": self.spatial_partition_mode,
            "current_phase": self.current_phase,
            "finalized_phases": list(self.finalized_phases),
            "single_head_global_labels": True,
            "current_phase_training_classes_only": True,
            "cumulative_seen_class_evaluation": True,
            "old_training_data_public_api": False,
            "stores_old_training_samples": False,
            "validation_enabled": bool(self.validation_enabled),
            "preprocessing_fit_phase": 0,
            "base_train_sample_count": int(self.base_train_indices.size),
            "preprocessing_frozen_across_phases": True,
            "spectral_axis_type": self.spectral_axis_type,
            "spectral_axis_is_physical": bool(self.spectral_axis_is_physical),
            "spectral_axis_length": int(self.spectral_axis.size),
            "spatial_leakage": self.spatial_leakage_report(),
        }

    def assert_exemplar_free_contract(self) -> bool:
        if self.finalized_phases != list(range(len(self.finalized_phases))):
            raise RuntimeError("finalized phase state is not contiguous")
        if self.current_phase is not None:
            if self.current_phase in self.finalized_phases:
                raise RuntimeError(
                    "an active phase cannot also be finalized"
                )
            if self.current_phase != len(self.finalized_phases):
                raise RuntimeError(
                    "active phase must immediately follow finalized phases"
                )
        if any(
            token in self.__dict__
            for token in ("exemplars", "replay_buffer", "memory_samples")
        ):
            raise RuntimeError("dataset manager contains forbidden replay state")
        if self.require_patch_disjoint and not self._split_patch_disjoint:
            raise RuntimeError("strict split patch-disjoint contract is violated")
        if (
            self.require_phase_context_isolation
            and not self._phase_context_isolated
        ):
            raise RuntimeError(
                "strict phase-context isolation contract is violated"
            )
        _validated_spectral_axis(
            self.spectral_axis,
            raw_bands=int(self.ordered_spectral_cube.shape[-1]),
            spectral_axis_type=self.spectral_axis_type,
        )
        return True


__all__ = [
    "build_incremental_protocol",
    "IncrementalHSIPatchDataset",
    "IncrementalHSIDatasetManager",
]







# from __future__ import annotations

# """Single leakage-auditable non-exemplar class-incremental HSI protocol.

# Protocol contract
# -----------------
# * Phase t training exposes only the training centres of classes introduced at t.
# * Validation during an active phase exposes only current-phase validation data.
# * Cumulative validation/test over all seen classes is evaluation-only and is
#   available only after the phase is finalized.
# * The spatial branch receives the processed/PCA patch.
# * The spectral branch receives the pre-PCA ordered centre spectrum from the
#   dataset cube; corrected datasets may already exclude sensor bands.
# * Those two inputs are fused by the backbone; the dataset does not create a
#   second learned representation.
# * No optimization/model-selection API returns historical training samples after its phase.
# * A finalized-phase map loader may read all labelled seen-class pixels strictly for
#   post-training qualitative reporting; those rows are never an optimization source.
# * Optional strict spatial audits detect patch-footprint overlap across data
#   splits and across unavailable old/future labelled samples.

# Important: a standard random-pixel HSI split is centre-disjoint but generally
# NOT patch-disjoint when patch_size > 1. Therefore it must not be described as
# spatial-leakage-free unless the strict audits actually pass.
# """

# import random
# from numbers import Integral, Real
# from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

# import numpy as np
# import torch
# from torch.utils.data import DataLoader, Dataset


# _SPLITS = ("train", "val", "test")


# def _exact_integer(value: object, *, name: str, minimum: Optional[int] = None) -> int:
#     """Return an exact signed-int64-compatible scalar without silent truncation."""
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

#     info = np.iinfo(np.int64)
#     if result < int(info.min) or result > int(info.max):
#         raise ValueError(f"{name} is outside signed int64 range")
#     if minimum is not None and result < minimum:
#         raise ValueError(f"{name} must be >= {minimum}")
#     return result


# def _exact_bool(value: object, *, name: str) -> bool:
#     """Parse an explicit Boolean without Python's bool('false') pitfall."""
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
#     """Return exact int64 data without Boolean, fractional, or overflow coercion."""
#     raw = np.asarray(value)
#     if raw.dtype == np.bool_ or raw.dtype.kind not in "iuf":
#         raise ValueError(f"{name} must contain integer values")

#     info = np.iinfo(np.int64)
#     if np.issubdtype(raw.dtype, np.floating):
#         if not np.isfinite(raw).all() or not np.equal(raw, np.round(raw)).all():
#             raise ValueError(f"{name} must contain finite integer values")
#         # float64 cannot represent int64.max exactly; use the exact
#         # representable half-open interval [-2**63, 2**63).
#         lower = float(-(2**63))
#         upper_exclusive = float(2**63)
#         if raw.size and (
#             np.any(raw < lower) or np.any(raw >= upper_exclusive)
#         ):
#             raise ValueError(f"{name} contains values outside signed int64 range")
#     elif raw.size and np.issubdtype(raw.dtype, np.unsignedinteger):
#         if np.any(raw > info.max):
#             raise ValueError(f"{name} contains values outside signed int64 range")
#     elif raw.size:
#         if np.any(raw < info.min) or np.any(raw > info.max):
#             raise ValueError(f"{name} contains values outside signed int64 range")

#     return raw.astype(np.int64, copy=False)


# def _ids(values: Iterable[int], *, name: str, allow_empty: bool = False) -> List[int]:
#     result = [
#         _exact_integer(value, name=f"{name} value", minimum=0)
#         for value in values
#     ]
#     if not result and not allow_empty:
#         raise ValueError(f"{name} cannot be empty")
#     if len(result) != len(set(result)):
#         raise ValueError(f"{name} must contain unique IDs")
#     return result


# def _normalize_split_name(split: str) -> str:
#     token = str(split).strip().lower()
#     return {
#         "training": "train",
#         "validation": "val",
#         "testing": "test",
#     }.get(token, token)


# _SPECTRAL_AXIS_ALIASES = {
#     "wavelength": "wavelength_nm",
#     "wavelengths": "wavelength_nm",
#     "wavelength_nm": "wavelength_nm",
#     "wavelengths_nm": "wavelength_nm",
#     "nm": "wavelength_nm",
#     "band": "ordered_band",
#     "bands": "ordered_band",
#     "band_index": "ordered_band",
#     "ordered_band": "ordered_band",
# }


# def _validated_spectral_axis(
#     spectral_axis: Optional[Sequence[float]],
#     *,
#     raw_bands: int,
#     spectral_axis_type: str,
# ) -> Tuple[np.ndarray, str]:
#     """Validate the one spectral coordinate shared by the model and CDS."""
#     bands = _exact_integer(raw_bands, name="raw_bands", minimum=1)
#     token = _SPECTRAL_AXIS_ALIASES.get(
#         str(spectral_axis_type).strip().lower()
#     )
#     if token is None:
#         raise ValueError(
#             "spectral_axis_type must be 'wavelength_nm' or 'ordered_band'"
#         )
#     if spectral_axis is None:
#         if token != "ordered_band":
#             raise ValueError(
#                 "physical wavelength coordinates must be supplied explicitly; "
#                 "they are never synthesized from band indices"
#             )
#         axis = np.arange(bands, dtype=np.float32)
#     else:
#         axis = np.asarray(spectral_axis, dtype=np.float64).reshape(-1)
#     if axis.shape != (bands,):
#         raise ValueError(
#             "spectral_axis must contain one coordinate per ordered raw band"
#         )
#     if not np.isfinite(axis).all():
#         raise ValueError("spectral_axis contains NaN/Inf")
#     if bands > 1 and not np.all(np.diff(axis) > 0.0):
#         raise ValueError("spectral_axis must be strictly increasing")
#     if token == "wavelength_nm" and np.any(axis <= 0.0):
#         raise ValueError("physical wavelengths in nanometres must be positive")
#     return axis.astype(np.float32), token


# def _seed_worker(_: int) -> None:
#     seed = int(torch.initial_seed() % (2**32))
#     random.seed(seed)
#     np.random.seed(seed)



# def _seed32(value: object, *, name: str) -> int:
#     """Validate the exact integer domain accepted by NumPy RandomState."""
#     result = _exact_integer(
#         value,
#         name=name,
#         minimum=0,
#     )
#     if result > (2**32) - 1:
#         raise ValueError(
#             f"{name} must be <= 2**32-1"
#         )
#     return result


# def _derived_seed32(base_seed: int, component: int) -> int:
#     """Derive a deterministic per-class seed without leaving uint32 range."""
#     base = _seed32(
#         base_seed,
#         name="base seed",
#     )
#     part = _exact_integer(
#         component,
#         name="seed component",
#         minimum=0,
#     )
#     return int(
#         (base + 1009 * part)
#         % (2**32)
#     )

# def _phase_schedule(
#     class_count: int,
#     base_classes: int,
#     increment: int,
# ) -> Tuple[List[int], Dict[int, List[int]]]:
#     total = _exact_integer(class_count, name="class_count", minimum=1)
#     base = _exact_integer(base_classes, name="base_classes", minimum=1)
#     step = _exact_integer(increment, name="increment", minimum=1)
#     if base > total:
#         raise ValueError("base_classes cannot exceed class_count")
#     sizes = [base]
#     remaining = total - base
#     while remaining:
#         current = min(step, remaining)
#         sizes.append(current)
#         remaining -= current
#     phases: Dict[int, List[int]] = {}
#     start = 0
#     for phase, size in enumerate(sizes):
#         phases[phase] = list(range(start, start + size))
#         start += size
#     return sizes, phases


# def _split_counts(
#     total: int,
#     *,
#     train_ratio: float,
#     val_ratio: float,
# ) -> Tuple[int, int, int]:
#     """Allocate one fixed per-class split; validation may be disabled.

#     Train and test are mandatory.  Validation is optional because some
#     published HSI benchmarks provide train/test masks only.
#     """
#     train_fraction = float(train_ratio)
#     val_fraction = float(val_ratio)

#     if (
#         not np.isfinite(train_fraction)
#         or not np.isfinite(val_fraction)
#         or train_fraction <= 0.0
#         or val_fraction < 0.0
#         or train_fraction + val_fraction >= 1.0
#     ):
#         raise ValueError(
#             "require finite train_ratio>0, val_ratio>=0, and "
#             "train_ratio+val_ratio<1"
#         )

#     required_split_count = 3 if val_fraction > 0.0 else 2
#     count = _exact_integer(
#         total,
#         name="class sample count",
#         minimum=required_split_count,
#     )

#     quotas = np.asarray(
#         (
#             train_fraction,
#             val_fraction,
#             1.0 - train_fraction - val_fraction,
#         ),
#         dtype=np.float64,
#     ) * count
#     counts = np.floor(quotas).astype(np.int64)
#     remainder = count - int(counts.sum())
#     if remainder:
#         order = np.argsort(-(quotas - counts), kind="stable")
#         counts[order[:remainder]] += 1

#     required = (0, 2) if val_fraction == 0.0 else (0, 1, 2)
#     for destination in required:
#         if counts[destination] == 0:
#             candidates = [
#                 index
#                 for index in required
#                 if index != destination and counts[index] > 1
#             ]
#             if not candidates:
#                 raise ValueError(
#                     "class is too small to realize the requested non-empty splits"
#                 )
#             source = max(candidates, key=lambda index: int(counts[index]))
#             counts[source] -= 1
#             counts[destination] += 1

#     if val_fraction == 0.0:
#         # A zero validation ratio is an explicit train/test-only protocol.
#         if counts[1] != 0:
#             counts[2] += counts[1]
#             counts[1] = 0

#     n_train, n_val, n_test = (int(v) for v in counts)
#     if n_train + n_val + n_test != count:
#         raise RuntimeError("split allocation does not conserve class samples")
#     if n_train <= 0 or n_test <= 0:
#         raise RuntimeError("train and test splits must be non-empty")
#     if val_fraction > 0.0 and n_val <= 0:
#         raise RuntimeError("positive val_ratio produced an empty validation split")
#     return n_train, n_val, n_test

# def _random_class_split(
#     indices: np.ndarray,
#     *,
#     train_ratio: float,
#     val_ratio: float,
#     seed: int,
# ) -> Dict[str, np.ndarray]:
#     values = _integer_array(indices, name="class indices").reshape(-1)
#     if np.unique(values).size != values.size:
#         raise ValueError("class indices contain duplicates")
#     n_train, n_val, _ = _split_counts(
#         len(values),
#         train_ratio=train_ratio,
#         val_ratio=val_ratio,
#     )
#     rng = np.random.RandomState(_exact_integer(seed, name="split seed", minimum=0))
#     shuffled = rng.permutation(values)
#     train = shuffled[:n_train]
#     val = shuffled[n_train : n_train + n_val]
#     test = shuffled[n_train + n_val :]
#     return {
#         "train": train,
#         "val": val,
#         "test": test,
#         "all": np.concatenate((train, val, test)),
#     }


# def _spatial_class_split(
#     indices: np.ndarray,
#     coords: np.ndarray,
#     *,
#     train_ratio: float,
#     val_ratio: float,
#     block_size: int,
#     seed: int,
# ) -> Dict[str, np.ndarray]:
#     """Classwise spatial-block split.

#     This is a benchmark split option, not a proof of patch disjointness.
#     Patch-overlap audits are computed separately.
#     """
#     values = _integer_array(indices, name="class indices").reshape(-1)
#     xy = _integer_array(coords, name="coords")[values]
#     size = _exact_integer(block_size, name="spatial_block_size", minimum=1)
#     max_block_col = int(np.max(xy[:, 1] // size)) + 1
#     block_ids = (xy[:, 0] // size) * max_block_col + xy[:, 1] // size
#     blocks = [values[block_ids == block] for block in np.unique(block_ids)]

#     validation_enabled = float(val_ratio) > 0.0
#     minimum_blocks = 3 if validation_enabled else 2
#     if len(blocks) < minimum_blocks:
#         raise RuntimeError(
#             "a class occupies too few spatial blocks for the requested split"
#         )

#     rng = np.random.RandomState(
#         _exact_integer(seed, name="split seed", minimum=0)
#     )
#     rng.shuffle(blocks)

#     target_train, target_val, _ = _split_counts(
#         len(values),
#         train_ratio=train_ratio,
#         val_ratio=val_ratio,
#     )

#     train_blocks: List[np.ndarray] = []
#     val_blocks: List[np.ndarray] = []
#     test_blocks: List[np.ndarray] = []
#     train_count = 0
#     val_count = 0

#     for block_index, block in enumerate(blocks):
#         remaining = len(blocks) - block_index
#         reserve_for_test = 1
#         reserve_for_val = 1 if validation_enabled and not val_blocks else 0

#         if (
#             train_count < target_train
#             and remaining > reserve_for_test + reserve_for_val
#         ):
#             train_blocks.append(block)
#             train_count += len(block)
#         elif (
#             validation_enabled
#             and val_count < target_val
#             and remaining > reserve_for_test
#         ):
#             val_blocks.append(block)
#             val_count += len(block)
#         else:
#             test_blocks.append(block)

#     if not train_blocks or not test_blocks:
#         raise RuntimeError("spatial block allocation produced an empty train/test split")
#     if validation_enabled and not val_blocks:
#         raise RuntimeError("spatial block allocation produced an empty validation split")

#     train = np.concatenate(train_blocks)
#     val = (
#         np.concatenate(val_blocks)
#         if val_blocks
#         else np.empty(0, dtype=np.int64)
#     )
#     test = np.concatenate(test_blocks)

#     return {
#         "train": train,
#         "val": val,
#         "test": test,
#         "all": np.concatenate((train, val, test)),
#     }

# def _validate_split_by_class(
#     split_by_class: Mapping[int, Mapping[str, np.ndarray]],
#     labels: np.ndarray,
#     class_count: int,
# ) -> None:
#     y = _integer_array(
#         labels,
#         name="global labels",
#     ).reshape(-1)
#     total_classes = _exact_integer(
#         class_count,
#         name="class_count",
#         minimum=1,
#     )

#     if y.size == 0:
#         raise ValueError(
#             "global labels cannot be empty"
#         )
#     observed_classes = sorted(
#         int(value)
#         for value in np.unique(y)
#     )
#     if observed_classes != list(
#         range(total_classes)
#     ):
#         raise ValueError(
#             "global labels must contain exactly contiguous class IDs "
#             f"0..{total_classes - 1}; observed={observed_classes}"
#         )

#     if set(int(v) for v in split_by_class) != set(range(total_classes)):
#         raise ValueError("split_by_class does not cover every class")

#     used: set[int] = set()
#     for class_id in range(total_classes):
#         row = split_by_class[class_id]
#         missing = set((*_SPLITS, "all")) - set(row)
#         if missing:
#             raise ValueError(
#                 f"class {class_id} split is missing {sorted(missing)}"
#             )

#         arrays = {
#             name: _integer_array(
#                 row[name],
#                 name=f"class {class_id} {name} indices",
#             ).reshape(-1)
#             for name in (*_SPLITS, "all")
#         }

#         if arrays["train"].size == 0 or arrays["test"].size == 0:
#             raise ValueError(
#                 f"class {class_id} must have non-empty train and test splits"
#             )
#         if arrays["all"].size == 0:
#             raise ValueError(f"class {class_id} all split cannot be empty")

#         for name, array in arrays.items():
#             if np.unique(array).size != array.size:
#                 raise ValueError(
#                     f"class {class_id} {name} split contains duplicates"
#                 )
#             if array.size and (array.min() < 0 or array.max() >= y.size):
#                 raise ValueError(
#                     f"class {class_id} {name} split contains invalid indices"
#                 )

#         train_set, val_set, test_set = (
#             set(arrays[name].tolist()) for name in _SPLITS
#         )
#         if train_set & val_set or train_set & test_set or val_set & test_set:
#             raise ValueError(f"class {class_id} train/val/test indices overlap")

#         union = train_set | val_set | test_set
#         if set(arrays["all"].tolist()) != union:
#             raise ValueError(f"class {class_id} all indices are inconsistent")

#         for array in arrays.values():
#             if array.size and not np.all(y[array] == class_id):
#                 raise ValueError(f"class {class_id} split contains another class")

#         expected = set(np.flatnonzero(y == class_id).tolist())
#         if union != expected:
#             raise ValueError(
#                 f"class {class_id} split does not account for every sample"
#             )
#         if used & expected:
#             raise ValueError("a sample appears in more than one class")
#         used |= expected

# def _collect_indices(
#     split_by_class: Mapping[int, Mapping[str, np.ndarray]],
#     class_ids: Sequence[int],
#     split: str,
# ) -> np.ndarray:
#     token = _normalize_split_name(split)
#     if token not in {*_SPLITS, "all"}:
#         raise ValueError(f"unknown split {split!r}")
#     ids = _ids(class_ids, name="class_ids", allow_empty=True)
#     if not ids:
#         return np.empty(0, dtype=np.int64)
#     return np.concatenate(
#         [
#             _integer_array(
#                 split_by_class[class_id][token],
#                 name=f"class {class_id} {token} indices",
#             ).reshape(-1)
#             for class_id in ids
#         ]
#     )


# def build_incremental_protocol(
#     input_labels: np.ndarray,
#     input_coords: np.ndarray,
#     *,
#     gt_shape: Sequence[int],
#     class_count: int,
#     base_classes: int,
#     increment: int,
#     train_ratio: float,
#     val_ratio: float,
#     seed: int = 0,
#     shuffle_order: bool = False,
#     class_order: Optional[Sequence[int]] = None,
#     target_names: Optional[Sequence[str]] = None,
#     split_strategy: str = "random_pixel",
#     spatial_block_size: int = 33,
#     predefined_splits: Optional[Mapping[int, Mapping[str, Sequence[int]]]] = None,
#     **_: Any,
# ) -> Dict[str, Any]:
#     original_labels = _integer_array(input_labels, name="input_labels").reshape(-1)
#     coords = _integer_array(input_coords, name="input_coords")
#     shape = tuple(
#         _exact_integer(v, name="gt_shape value", minimum=1) for v in gt_shape
#     )
#     total_classes = _exact_integer(class_count, name="class_count", minimum=1)
#     protocol_seed = _seed32(
#         seed,
#         name="seed",
#     )

#     if len(shape) != 2 or coords.shape != (len(original_labels), 2):
#         raise ValueError("labels, coordinates and gt_shape are inconsistent")
#     if original_labels.size == 0:
#         raise ValueError("input_labels cannot be empty")
#     if np.unique(coords, axis=0).shape[0] != coords.shape[0]:
#         raise ValueError("coords must identify unique labelled pixels")
#     if (
#         np.any(coords[:, 0] < 0)
#         or np.any(coords[:, 0] >= shape[0])
#         or np.any(coords[:, 1] < 0)
#         or np.any(coords[:, 1] >= shape[1])
#     ):
#         raise ValueError("coords lie outside gt_shape")

#     original_ids = sorted(int(v) for v in np.unique(original_labels))
#     if original_ids != list(range(total_classes)):
#         raise ValueError("input labels must be contiguous zero-based class IDs")

#     shuffle_classes = _exact_bool(shuffle_order, name="shuffle_order")
#     if class_order is not None and shuffle_classes:
#         raise ValueError("choose either class_order or shuffle_order, not both")
#     if class_order is not None:
#         order = _ids(class_order, name="class_order")
#     elif shuffle_classes:
#         order = np.random.RandomState(protocol_seed).permutation(
#             original_ids
#         ).tolist()
#     else:
#         order = original_ids
#     if set(order) != set(original_ids) or len(order) != len(original_ids):
#         raise ValueError("class_order must be a complete class permutation")

#     original_to_global = {
#         original: global_id for global_id, original in enumerate(order)
#     }
#     global_to_original = {
#         global_id: original for original, global_id in original_to_global.items()
#     }
#     global_labels = np.asarray(
#         [original_to_global[int(label)] for label in original_labels],
#         dtype=np.int64,
#     )
#     phase_sizes, phase_to_classes = _phase_schedule(
#         total_classes, base_classes, increment
#     )

#     strategy = str(split_strategy).strip().lower()
#     if predefined_splits is not None:
#         predefined_original = {
#             _exact_integer(class_id, name="predefined class ID", minimum=0): row
#             for class_id, row in predefined_splits.items()
#         }
#         if set(predefined_original) != set(original_ids):
#             raise ValueError(
#                 "predefined_splits must be keyed by every original class ID"
#             )
#         split_by_class = {}
#         for original_id, row in predefined_original.items():
#             if "train" not in row or "test" not in row:
#                 raise ValueError(
#                     f"predefined class {original_id} must provide train and test indices"
#                 )
#             mapped = {
#                 "train": _integer_array(
#                     row["train"],
#                     name=f"predefined class {original_id} train indices",
#                 ).reshape(-1),
#                 "val": _integer_array(
#                     row.get("val", np.empty(0, dtype=np.int64)),
#                     name=f"predefined class {original_id} val indices",
#                 ).reshape(-1),
#                 "test": _integer_array(
#                     row["test"],
#                     name=f"predefined class {original_id} test indices",
#                 ).reshape(-1),
#             }
#             mapped["all"] = np.concatenate(
#                 (mapped["train"], mapped["val"], mapped["test"])
#             )
#             split_by_class[original_to_global[original_id]] = mapped
#         strategy = "predefined"
#     elif strategy == "random_pixel":
#         split_by_class = {
#             class_id: _random_class_split(
#                 np.flatnonzero(global_labels == class_id),
#                 train_ratio=train_ratio,
#                 val_ratio=val_ratio,
#                 seed=_derived_seed32(
#                     protocol_seed,
#                     global_to_original[class_id],
#                 ),
#             )
#             for class_id in range(total_classes)
#         }
#     elif strategy == "spatial_block":
#         split_by_class = {
#             class_id: _spatial_class_split(
#                 np.flatnonzero(global_labels == class_id),
#                 coords,
#                 train_ratio=train_ratio,
#                 val_ratio=val_ratio,
#                 block_size=spatial_block_size,
#                 seed=_derived_seed32(
#                     protocol_seed,
#                     global_to_original[class_id],
#                 ),
#             )
#             for class_id in range(total_classes)
#         }
#     else:
#         raise ValueError(
#             "split_strategy must be random_pixel, spatial_block, or predefined"
#         )

#     _validate_split_by_class(split_by_class, global_labels, total_classes)

#     validation_presence = [
#         bool(split_by_class[class_id]["val"].size)
#         for class_id in range(total_classes)
#     ]
#     if any(validation_presence) and not all(validation_presence):
#         raise ValueError(
#             "validation must be consistently enabled for every class or "
#             "disabled for every class"
#         )
#     validation_enabled = all(validation_presence)

#     names = [] if target_names is None else [str(v) for v in target_names]
#     if names and len(names) != total_classes:
#         raise ValueError("target_names must contain one name per class")
#     global_names = [names[original] for original in order] if names else []

#     phase_train = {
#         phase: _collect_indices(split_by_class, classes, "train")
#         for phase, classes in phase_to_classes.items()
#     }
#     phase_val = {
#         phase: _collect_indices(split_by_class, classes, "val")
#         for phase, classes in phase_to_classes.items()
#     }
#     phase_test = {
#         phase: _collect_indices(split_by_class, classes, "test")
#         for phase, classes in phase_to_classes.items()
#     }
#     cumulative_test = {
#         phase: _collect_indices(
#             split_by_class,
#             list(range(sum(phase_sizes[: phase + 1]))),
#             "test",
#         )
#         for phase in phase_to_classes
#     }

#     return {
#         "training_mode": "non_exemplar_class_incremental",
#         "seed": protocol_seed,
#         "class_count": total_classes,
#         "base_classes": _exact_integer(
#             base_classes, name="base_classes", minimum=1
#         ),
#         "increment": _exact_integer(increment, name="increment", minimum=1),
#         "phase_sizes": phase_sizes,
#         "phase_to_classes": phase_to_classes,
#         "class_order_original_ids": order,
#         "original_to_global": original_to_global,
#         "global_to_original": global_to_original,
#         "original_labels": original_labels,
#         "global_labels": global_labels,
#         "global_target_names": global_names,
#         "split_by_class": split_by_class,
#         "phase_train_indices": phase_train,
#         "phase_val_indices": phase_val,
#         "phase_test_indices": phase_test,
#         "cumulative_test_indices": cumulative_test,
#         # Canonical preprocessing membership. This is exactly phase-0 TRAIN,
#         # not a second independently constructed split.
#         "base_train_indices": phase_train[0].copy(),
#         "split_strategy": strategy,
#         "spatial_partition_mode": (
#             "classwise_block_assignment"
#             if strategy == "spatial_block"
#             else "pixel_indices"
#         ),
#         "single_head_global_labels": True,
#         "current_phase_training_classes_only": True,
#         "cumulative_seen_class_evaluation": True,
#         "stores_old_training_samples": False,
#         "validation_enabled": bool(validation_enabled),
#         "preprocessing_fit_phase": 0,
#         "preprocessing_frozen_across_phases": True,
#     }


# class IncrementalHSIPatchDataset(Dataset):
#     def __init__(
#         self,
#         manager: "IncrementalHSIDatasetManager",
#         indices: Sequence[int],
#     ) -> None:
#         self.manager = manager
#         self.indices = _integer_array(indices, name="dataset indices").reshape(-1)
#         if self.indices.size == 0:
#             raise ValueError("dataset indices cannot be empty")
#         if self.indices.min() < 0 or self.indices.max() >= len(manager.labels):
#             raise ValueError("dataset indices are outside labelled samples")
#         if np.unique(self.indices).size != self.indices.size:
#             raise ValueError("dataset indices cannot contain duplicates")

#     def __len__(self) -> int:
#         return int(self.indices.size)

#     def __getitem__(self, item: int) -> Dict[str, torch.Tensor]:
#         item_id = _exact_integer(item, name="dataset item", minimum=0)
#         if item_id >= len(self):
#             raise IndexError(item_id)
#         sample_index = int(self.indices[item_id])
#         row, col = self.manager.coords[sample_index]
#         patch = self.manager.extract_patch(int(row), int(col))
#         raw_center_spectrum = self.manager.ordered_spectral_cube[
#             int(row), int(col)
#         ]
#         return {
#             "image": torch.from_numpy(
#                 np.moveaxis(patch, -1, 0).copy()
#             ).float(),
#             "raw_center_spectrum": torch.from_numpy(
#                 raw_center_spectrum.copy()
#             ).float(),
#             "label": torch.tensor(
#                 int(self.manager.labels[sample_index]),
#                 dtype=torch.long,
#             ),
#             "coord": torch.tensor(
#                 (int(row), int(col)),
#                 dtype=torch.long,
#             ),
#             "sample_index": torch.tensor(sample_index, dtype=torch.long),
#         }


# class IncrementalHSIDatasetManager:
#     """Current-class training and seen-class evaluation with spatial audits."""

#     def __init__(
#         self,
#         *,
#         processed_cube: np.ndarray,
#         labels: np.ndarray,
#         coords: np.ndarray,
#         protocol: Mapping[str, Any],
#         target_names: Sequence[str],
#         gt_shape: Sequence[int],
#         patch_size: int,
#         ordered_spectral_cube: Optional[np.ndarray] = None,
#         spectral_axis: Optional[Sequence[float]] = None,
#         spectral_axis_type: str = "ordered_band",
#         num_workers: int = 0,
#         device: str = "cpu",
#         seed: Optional[int] = None,
#         require_patch_disjoint: bool = False,
#         require_phase_context_isolation: Optional[bool] = None,
#         context_policy: str = "full_scene_reflect",
#         **_: Any,
#     ) -> None:
#         self.processed_cube = np.ascontiguousarray(
#             processed_cube, dtype=np.float32
#         )
#         self.gt_shape = tuple(
#             _exact_integer(v, name="gt_shape value", minimum=1)
#             for v in gt_shape
#         )
#         self.coords = _integer_array(coords, name="coords")
#         supplied_labels = _integer_array(labels, name="labels").reshape(-1)
#         original_labels = _integer_array(
#             protocol["original_labels"], name="protocol original_labels"
#         ).reshape(-1)
#         global_labels = _integer_array(
#             protocol["global_labels"], name="protocol global labels"
#         ).reshape(-1)
#         if (
#             np.array_equal(
#                 supplied_labels,
#                 original_labels,
#             )
#             or np.array_equal(
#                 supplied_labels,
#                 global_labels,
#             )
#         ):
#             self.labels = global_labels
#         else:
#             raise ValueError(
#                 "labels do not match the protocol"
#             )

#         protocol_class_count = _exact_integer(
#             protocol["class_count"],
#             name="protocol class_count",
#             minimum=1,
#         )
#         observed_global_classes = sorted(
#             int(value)
#             for value in np.unique(self.labels)
#         )
#         if observed_global_classes != list(
#             range(protocol_class_count)
#         ):
#             raise ValueError(
#                 "protocol global labels must contain exactly contiguous "
#                 f"class IDs 0..{protocol_class_count - 1}; "
#                 f"observed={observed_global_classes}"
#             )

#         if (
#             self.processed_cube.ndim != 3
#             or self.processed_cube.shape[:2] != self.gt_shape
#             or self.processed_cube.shape[2] <= 0
#             or not np.isfinite(self.processed_cube).all()
#         ):
#             raise ValueError("processed_cube must be finite and match gt_shape")
#         if self.coords.shape != (len(self.labels), 2):
#             raise ValueError("coords must align with labels")
#         if np.unique(self.coords, axis=0).shape[0] != self.coords.shape[0]:
#             raise ValueError("coords contain duplicate labelled-pixel locations")
#         if (
#             np.any(self.coords[:, 0] < 0)
#             or np.any(self.coords[:, 0] >= self.gt_shape[0])
#             or np.any(self.coords[:, 1] < 0)
#             or np.any(self.coords[:, 1] >= self.gt_shape[1])
#         ):
#             raise ValueError("coords contain positions outside gt_shape")

#         if ordered_spectral_cube is None:
#             raise ValueError(
#                 "ordered_spectral_cube is required for the raw spectral branch"
#             )
#         ordered = np.asarray(ordered_spectral_cube, dtype=np.float32)
#         if (
#             ordered.ndim != 3
#             or ordered.shape[:2] != self.gt_shape
#             or ordered.shape[2] <= 0
#             or not np.isfinite(ordered).all()
#         ):
#             raise ValueError("ordered_spectral_cube is invalid")
#         self.ordered_spectral_cube = np.ascontiguousarray(ordered)
#         self.spectral_axis, self.spectral_axis_type = _validated_spectral_axis(
#             spectral_axis,
#             raw_bands=int(self.ordered_spectral_cube.shape[-1]),
#             spectral_axis_type=spectral_axis_type,
#         )
#         self.spectral_axis_is_physical = (
#             self.spectral_axis_type == "wavelength_nm"
#         )

#         self.patch_size = _exact_integer(
#             patch_size, name="patch_size", minimum=1
#         )
#         if self.patch_size % 2 == 0:
#             raise ValueError("patch_size must be odd")
#         radius = self.patch_size // 2
#         self.padded_cube = np.pad(
#             self.processed_cube,
#             ((radius, radius), (radius, radius), (0, 0)),
#             mode="reflect",
#         )

#         self.phase_to_classes = {
#             _exact_integer(phase, name="phase ID", minimum=0): _ids(
#                 classes, name=f"phase_to_classes[{phase}]"
#             )
#             for phase, classes in protocol["phase_to_classes"].items()
#         }
#         if sorted(self.phase_to_classes) != list(range(len(self.phase_to_classes))):
#             raise ValueError("phase IDs must be contiguous from zero")
#         class_count = protocol_class_count
#         flattened = [
#             class_id
#             for phase in range(len(self.phase_to_classes))
#             for class_id in self.phase_to_classes[phase]
#         ]
#         if flattened != list(range(class_count)):
#             raise ValueError(
#                 "phase schedule must cover global class IDs once in order"
#             )

#         # The protocol must expose one canonical base-training membership for
#         # preprocessing, and it must be exactly the phase-0 training split.
#         if "base_train_indices" not in protocol:
#             raise ValueError("protocol is missing base_train_indices")
#         if "phase_train_indices" not in protocol:
#             raise ValueError("protocol is missing phase_train_indices")
#         phase_train_indices = protocol["phase_train_indices"]
#         if not isinstance(phase_train_indices, Mapping) or 0 not in phase_train_indices:
#             raise ValueError("protocol phase_train_indices must contain phase 0")
#         base_train_indices = _integer_array(
#             protocol["base_train_indices"],
#             name="protocol base_train_indices",
#         ).reshape(-1)
#         phase0_train_indices = _integer_array(
#             phase_train_indices[0],
#             name="protocol phase_train_indices[0]",
#         ).reshape(-1)
#         if (
#             base_train_indices.size == 0
#             or phase0_train_indices.size == 0
#             or not np.array_equal(
#                 np.sort(base_train_indices),
#                 np.sort(phase0_train_indices),
#             )
#         ):
#             raise ValueError(
#                 "protocol base_train_indices must equal phase_train_indices[0]"
#             )
#         self.base_train_indices = base_train_indices.copy()

#         self._split_by_class = {
#             _exact_integer(class_id, name="split class ID", minimum=0): {
#                 name: _integer_array(
#                     values, name=f"class {class_id} {name} indices"
#                 )
#                 for name, values in row.items()
#             }
#             for class_id, row in protocol["split_by_class"].items()
#         }
#         _validate_split_by_class(
#             self._split_by_class, self.labels, class_count
#         )

#         # Validate every protocol membership field against the single canonical
#         # split_by_class state.  This prevents preprocessing/training/reporting
#         # from silently consuming different sample memberships.
#         for field, split_name in (
#             ("phase_train_indices", "train"),
#             ("phase_val_indices", "val"),
#             ("phase_test_indices", "test"),
#         ):
#             raw = protocol.get(field)
#             if not isinstance(raw, Mapping):
#                 raise ValueError(f"protocol {field} must be a phase mapping")
#             if set(int(key) for key in raw) != set(self.phase_to_classes):
#                 raise ValueError(
#                     f"protocol {field} must contain every phase exactly once"
#                 )

#             for phase_id, classes in self.phase_to_classes.items():
#                 declared = _integer_array(
#                     raw[phase_id],
#                     name=f"protocol {field}[{phase_id}]",
#                 ).reshape(-1)
#                 expected = _collect_indices(
#                     self._split_by_class,
#                     classes,
#                     split_name,
#                 )
#                 if not np.array_equal(
#                     np.sort(declared),
#                     np.sort(expected),
#                 ):
#                     raise ValueError(
#                         f"protocol {field}[{phase_id}] disagrees with split_by_class"
#                     )

#         cumulative_test = protocol.get("cumulative_test_indices")
#         if not isinstance(cumulative_test, Mapping):
#             raise ValueError(
#                 "protocol cumulative_test_indices must be a phase mapping"
#             )
#         if set(int(key) for key in cumulative_test) != set(self.phase_to_classes):
#             raise ValueError(
#                 "protocol cumulative_test_indices must contain every phase"
#             )
#         for phase_id in self.phase_to_classes:
#             declared = _integer_array(
#                 cumulative_test[phase_id],
#                 name=f"protocol cumulative_test_indices[{phase_id}]",
#             ).reshape(-1)
#             expected = _collect_indices(
#                 self._split_by_class,
#                 self.get_seen_classes(phase_id),
#                 "test",
#             )
#             if not np.array_equal(
#                 np.sort(declared),
#                 np.sort(expected),
#             ):
#                 raise ValueError(
#                     "protocol cumulative_test_indices "
#                     f"[{phase_id}] disagrees with split_by_class"
#                 )

#         preprocessing_fit_phase = _exact_integer(
#             protocol.get("preprocessing_fit_phase", 0),
#             name="protocol preprocessing_fit_phase",
#             minimum=0,
#         )
#         if preprocessing_fit_phase != 0:
#             raise ValueError("persistent HSI preprocessing must be fitted in phase 0")

#         preprocessing_frozen = _exact_bool(
#             protocol.get("preprocessing_frozen_across_phases", True),
#             name="protocol preprocessing_frozen_across_phases",
#         )
#         if not preprocessing_frozen:
#             raise ValueError(
#                 "persistent HSI preprocessing must remain frozen across phases"
#             )

#         names = [str(v) for v in target_names]
#         global_names = [
#             str(v) for v in protocol.get("global_target_names", [])
#         ]
#         if global_names and len(global_names) != class_count:
#             raise ValueError("protocol global_target_names has wrong length")
#         if not global_names and len(names) != class_count:
#             raise ValueError("target_names must contain one name per class")
#         self.class_order_original_ids = _ids(
#             protocol["class_order_original_ids"],
#             name="class_order_original_ids",
#         )
#         if set(self.class_order_original_ids) != set(range(class_count)):
#             raise ValueError(
#                 "class_order_original_ids must be a complete original-class permutation"
#             )
#         self.target_names = global_names or [
#             names[index] for index in self.class_order_original_ids
#         ]

#         self.global_to_original = {
#             _exact_integer(k, name="global class ID", minimum=0):
#             _exact_integer(v, name="original class ID", minimum=0)
#             for k, v in protocol["global_to_original"].items()
#         }
#         expected_global_to_original = {
#             global_id: original_id
#             for global_id, original_id in enumerate(self.class_order_original_ids)
#         }
#         if self.global_to_original != expected_global_to_original:
#             raise ValueError(
#                 "global_to_original disagrees with class_order_original_ids"
#             )

#         self.split_strategy = str(protocol["split_strategy"]).strip().lower()
#         self.spatial_partition_mode = str(
#             protocol.get("spatial_partition_mode", "pixel_indices")
#         )

#         validation_presence = [
#             bool(self._split_by_class[class_id]["val"].size)
#             for class_id in range(class_count)
#         ]
#         if any(validation_presence) and not all(validation_presence):
#             raise ValueError(
#                 "validation must be present for every class or absent for every class"
#             )
#         actual_validation_enabled = all(validation_presence)
#         declared_validation_enabled = _exact_bool(
#             protocol.get("validation_enabled", actual_validation_enabled),
#             name="protocol validation_enabled",
#         )
#         if declared_validation_enabled != actual_validation_enabled:
#             raise ValueError(
#                 "protocol validation_enabled disagrees with validation split membership"
#             )
#         self.validation_enabled = actual_validation_enabled

#         self.require_patch_disjoint = _exact_bool(
#             require_patch_disjoint,
#             name="require_patch_disjoint",
#         )
#         self.require_phase_context_isolation = (
#             self.require_patch_disjoint
#             if require_phase_context_isolation is None
#             else _exact_bool(
#                 require_phase_context_isolation,
#                 name="require_phase_context_isolation",
#             )
#         )
#         self.context_policy = str(context_policy).strip()
#         if self.context_policy != "full_scene_reflect":
#             raise ValueError(
#                 "only context_policy='full_scene_reflect' is implemented"
#             )

#         # Protocol randomness and optimization/DataLoader randomness are
#         # intentionally separate. ``protocol_seed`` fixes class order and
#         # train/val/test membership; ``self.seed`` controls only DataLoader
#         # shuffling/worker randomness for this optimization run.  Multi-seed
#         # robustness therefore supports, for example, protocol_seed=42 with
#         # optimization seeds 42/123/70 without changing the benchmark split.
#         self.protocol_seed = _seed32(
#             protocol["seed"],
#             name="protocol seed",
#         )
#         self.seed = (
#             self.protocol_seed
#             if seed is None
#             else _seed32(seed, name="dataset-manager optimization seed")
#         )

#         self.num_workers = _exact_integer(
#             num_workers, name="num_workers", minimum=0
#         )
#         self.pin_memory = str(device).startswith("cuda")
#         self.current_phase: Optional[int] = None
#         self.finalized_phases: List[int] = []

#         self._split_patch_disjoint = self._audit_split_patch_disjoint()
#         self._phase_context_isolated = self._audit_phase_context_isolation()

#         if self.require_patch_disjoint and not self._split_patch_disjoint:
#             raise RuntimeError(
#                 "train/val/test patch footprints overlap. A random-pixel centre "
#                 "split is not spatial-leakage-free for patch models. Use a "
#                 "guarded/predefined spatial split or explicitly disable the "
#                 "strict audit only for benchmark-comparability experiments."
#             )
#         if (
#             self.require_phase_context_isolation
#             and not self._phase_context_isolated
#         ):
#             raise RuntimeError(
#                 "at least one phase-training patch contains an annotated labeled "
#                 "center unavailable in that phase (old, future, validation, or test). "
#                 "This violates strict spatial phase isolation."
#             )

#     @property
#     def nb_tasks(self) -> int:
#         return len(self.phase_to_classes)

#     @property
#     def base_classes(self) -> List[int]:
#         return list(self.phase_to_classes[0])

#     def _validate_phase(self, phase: int) -> int:
#         value = _exact_integer(phase, name="phase", minimum=0)
#         if value not in self.phase_to_classes:
#             raise ValueError(f"unknown phase {phase}")
#         return value

#     def start_phase(self, phase: int) -> None:
#         value = self._validate_phase(phase)
#         if self.current_phase is not None:
#             raise RuntimeError(f"phase {self.current_phase} is already active")
#         if value in self.finalized_phases:
#             raise RuntimeError(f"phase {value} is already finalized")
#         if self.finalized_phases != list(range(value)):
#             raise RuntimeError(
#                 "phases must start in order after prior finalization"
#             )
#         self.current_phase = value

#     def finalize_phase(self, phase: int) -> None:
#         value = self._validate_phase(phase)
#         if self.current_phase != value:
#             raise RuntimeError(f"phase {value} is not the active phase")
#         self.finalized_phases.append(value)
#         self.current_phase = None

#     def restore_finalized_phases(
#         self,
#         phases: Sequence[int],
#     ) -> None:
#         """Restore checkpoint protocol state without exposing historical data."""
#         if self.current_phase is not None:
#             raise RuntimeError(
#                 "cannot restore finalized phases while a phase is active"
#             )
#         if self.finalized_phases:
#             raise RuntimeError(
#                 "restore_finalized_phases requires a fresh unfinalized manager"
#             )
#         values = [
#             self._validate_phase(value)
#             for value in phases
#         ]
#         if values != list(range(len(values))):
#             raise ValueError(
#                 "restored finalized phases must be contiguous from zero"
#             )
#         self.finalized_phases = values

#     def get_task_size(self, phase: int) -> int:
#         return len(self.phase_to_classes[self._validate_phase(phase)])

#     def get_new_classes(self, phase: int) -> List[int]:
#         return list(self.phase_to_classes[self._validate_phase(phase)])

#     def get_old_classes(self, phase: int) -> List[int]:
#         value = self._validate_phase(phase)
#         return [
#             class_id
#             for previous in range(value)
#             for class_id in self.phase_to_classes[previous]
#         ]

#     def get_seen_classes(self, phase: int) -> List[int]:
#         return self.get_old_classes(phase) + self.get_new_classes(phase)

#     def extract_patch(self, row: int, col: int) -> np.ndarray:
#         row_id = _exact_integer(row, name="patch row", minimum=0)
#         col_id = _exact_integer(col, name="patch col", minimum=0)
#         if row_id >= self.gt_shape[0] or col_id >= self.gt_shape[1]:
#             raise ValueError("patch centre lies outside gt_shape")
#         return self.padded_cube[
#             row_id : row_id + self.patch_size,
#             col_id : col_id + self.patch_size,
#         ]

#     def _indices_mask(self, indices: np.ndarray) -> np.ndarray:
#         mask = np.zeros(self.gt_shape, dtype=bool)
#         values = _integer_array(indices, name="audit indices").reshape(-1)
#         if values.size:
#             if values.min() < 0 or values.max() >= len(self.labels):
#                 raise ValueError("audit indices are outside labelled samples")
#             if np.unique(values).size != values.size:
#                 raise ValueError("audit indices contain duplicates")
#             xy = self.coords[values]
#             mask[xy[:, 0], xy[:, 1]] = True
#         return mask

#     def _patch_overlap_count(
#         self,
#         left_indices: np.ndarray,
#         right_indices: np.ndarray,
#     ) -> int:
#         """Count left centers whose extracted patch overlaps a right patch."""
#         left = _integer_array(left_indices, name="left indices").reshape(-1)
#         right = _integer_array(right_indices, name="right indices").reshape(-1)
#         if left.size == 0 or right.size == 0:
#             return 0

#         # Two odd P x P patches overlap when center Chebyshev distance <= P-1.
#         from scipy.ndimage import maximum_filter

#         right_mask = self._indices_mask(right)
#         radius = self.patch_size - 1
#         dilated = (
#             right_mask
#             if radius == 0
#             else maximum_filter(
#                 right_mask.astype(np.uint8),
#                 size=2 * radius + 1,
#                 mode="constant",
#                 cval=0,
#             ).astype(bool)
#         )
#         xy = self.coords[left]
#         return int(dilated[xy[:, 0], xy[:, 1]].sum())

#     def _forbidden_center_in_patch_count(
#         self,
#         train_indices: np.ndarray,
#         forbidden_indices: np.ndarray,
#     ) -> int:
#         """Count train patches containing an unavailable labeled center.

#         This differs from patch-vs-patch overlap. During phase-t training the
#         unavailable old/future/validation/test sample is not itself extracted as
#         a patch. The strict observable leakage event is that its labeled center
#         spectrum lies inside the current train patch.
#         """
#         train = _integer_array(train_indices, name="train indices").reshape(-1)
#         forbidden = _integer_array(
#             forbidden_indices, name="forbidden indices"
#         ).reshape(-1)
#         if train.size == 0 or forbidden.size == 0:
#             return 0

#         from scipy.ndimage import maximum_filter

#         forbidden_mask = self._indices_mask(forbidden)
#         radius = self.patch_size // 2
#         dilated = (
#             forbidden_mask
#             if radius == 0
#             else maximum_filter(
#                 forbidden_mask.astype(np.uint8),
#                 size=2 * radius + 1,
#                 mode="constant",
#                 cval=0,
#             ).astype(bool)
#         )
#         xy = self.coords[train]
#         return int(dilated[xy[:, 0], xy[:, 1]].sum())

#     def _audit_split_patch_disjoint(self) -> bool:
#         all_classes = list(range(len(self._split_by_class)))
#         split_indices = {
#             split: _collect_indices(
#                 self._split_by_class, all_classes, split
#             )
#             for split in _SPLITS
#         }
#         for left_index, left_name in enumerate(_SPLITS):
#             for right_name in _SPLITS[left_index + 1 :]:
#                 if self._patch_overlap_count(
#                     split_indices[left_name],
#                     split_indices[right_name],
#                 ):
#                     return False
#         return True

#     def _audit_phase_context_isolation(self) -> bool:
#         all_indices = np.arange(len(self.labels), dtype=np.int64)
#         for phase in range(self.nb_tasks):
#             available = _collect_indices(
#                 self._split_by_class,
#                 self.phase_to_classes[phase],
#                 "train",
#             )
#             available_set = np.zeros(len(self.labels), dtype=bool)
#             available_set[available] = True
#             forbidden = all_indices[~available_set]
#             if self._forbidden_center_in_patch_count(available, forbidden):
#                 return False
#         return True

#     def spatial_leakage_report(self) -> Dict[str, Any]:
#         return {
#             "patch_size": self.patch_size,
#             "split_patch_disjoint": bool(self._split_patch_disjoint),
#             "phase_training_context_isolated": bool(
#                 self._phase_context_isolated
#             ),
#             "strict_split_required": self.require_patch_disjoint,
#             "strict_phase_context_required": self.require_phase_context_isolation,
#             "context_policy": self.context_policy,
#             "guarantee_scope": (
#                 "train/val/test extracted patch footprints plus unavailable annotated "
#                 "labeled centers inside phase-training patches; unlabeled/background "
#                 "scene pixels cannot be assigned a task identity by the benchmark"
#             ),
#         }

#     def _make_dataset(
#         self,
#         class_ids: Sequence[int],
#         split: str,
#     ) -> IncrementalHSIPatchDataset:
#         return IncrementalHSIPatchDataset(
#             self,
#             _collect_indices(self._split_by_class, class_ids, split),
#         )

#     def _loader(
#         self,
#         dataset: Dataset,
#         *,
#         batch_size: int,
#         shuffle: bool,
#         phase: int,
#         split: str,
#     ) -> DataLoader:
#         size = _exact_integer(batch_size, name="batch_size", minimum=1)
#         phase_id = self._validate_phase(phase)
#         offset = {"train": 11, "val": 17, "test": 23, "all": 29}[split]
#         generator = torch.Generator().manual_seed(
#             self.seed + 1009 * phase_id + offset
#         )
#         return DataLoader(
#             dataset,
#             batch_size=size,
#             shuffle=_exact_bool(shuffle, name="shuffle"),
#             num_workers=self.num_workers,
#             pin_memory=self.pin_memory,
#             drop_last=False,
#             persistent_workers=self.num_workers > 0,
#             worker_init_fn=_seed_worker if self.num_workers > 0 else None,
#             generator=generator,
#         )

#     def get_phase_dataloader(
#         self,
#         phase: int,
#         split: str = "train",
#         batch_size: int = 64,
#         shuffle: bool = True,
#     ) -> DataLoader:
#         phase_id = self._validate_phase(phase)
#         if self.current_phase != phase_id:
#             raise RuntimeError(
#                 "phase train/validation loaders require an active phase"
#             )
#         token = _normalize_split_name(split)
#         if token not in {"train", "val"}:
#             raise ValueError("active phase loaders are train/val only")
#         if token == "val" and not self.validation_enabled:
#             raise RuntimeError("validation is disabled by this experiment protocol")
#         dataset = self._make_dataset(
#             self.get_new_classes(phase_id), token
#         )
#         return self._loader(
#             dataset,
#             batch_size=batch_size,
#             shuffle=shuffle if token == "train" else False,
#             phase=phase_id,
#             split=token,
#         )

#     def get_phase_structure_dataloader(
#         self,
#         phase: int,
#         *,
#         batch_size: int = 256,
#     ) -> DataLoader:
#         """Deterministic full current-TRAIN loader for structure estimation.

#         This deliberately reuses the active phase TRAIN membership, exposes no
#         historical rows, disables shuffling, and therefore provides the single
#         legal source for epoch refresh/final commit of current-class
#         I/V/Mspec/Vspec.
#         """
#         return self.get_phase_dataloader(
#             phase,
#             split="train",
#             batch_size=batch_size,
#             shuffle=False,
#         )

#     def get_cumulative_dataloader(
#         self,
#         phase: int,
#         split: str = "test",
#         batch_size: int = 256,
#         shuffle: bool = False,
#     ) -> DataLoader:
#         phase_id = self._validate_phase(phase)
#         if phase_id not in self.finalized_phases:
#             raise RuntimeError(
#                 "cumulative evaluation requires a finalized phase"
#             )
#         token = _normalize_split_name(split)
#         if token not in {"val", "test"}:
#             raise ValueError("cumulative loaders are evaluation-only")
#         if token == "val" and not self.validation_enabled:
#             raise RuntimeError("validation is disabled by this experiment protocol")
#         dataset = self._make_dataset(
#             self.get_seen_classes(phase_id), token
#         )
#         return self._loader(
#             dataset,
#             batch_size=batch_size,
#             shuffle=bool(shuffle),
#             phase=phase_id,
#             split=token,
#         )

#     def get_reporting_dataloader(
#         self,
#         phase: int,
#         *,
#         batch_size: int = 256,
#     ) -> DataLoader:
#         """Test-only reporting loader. Never mixes train/val samples into metrics."""
#         return self.get_cumulative_dataloader(
#             phase,
#             split="test",
#             batch_size=batch_size,
#             shuffle=False,
#         )

#     def get_map_dataloader(
#         self,
#         phase: int,
#         *,
#         batch_size: int = 256,
#     ) -> DataLoader:
#         """All labeled seen-class pixels for qualitative maps only; never metrics/training."""
#         phase_id = self._validate_phase(phase)
#         if phase_id not in self.finalized_phases:
#             raise RuntimeError("map generation requires a finalized phase")
#         dataset = self._make_dataset(
#             self.get_seen_classes(phase_id), "all"
#         )
#         return self._loader(
#             dataset,
#             batch_size=batch_size,
#             shuffle=False,
#             phase=phase_id,
#             split="all",
#         )

#     def base_phase_contract_report(self) -> Dict[str, Any]:
#         """Return read-only checks for the finalized phase-0 data contract."""
#         base_ids = self.get_new_classes(0)
#         train_indices = _collect_indices(
#             self._split_by_class,
#             base_ids,
#             "train",
#         )
#         val_indices = _collect_indices(
#             self._split_by_class,
#             base_ids,
#             "val",
#         )
#         test_indices = _collect_indices(
#             self._split_by_class,
#             base_ids,
#             "test",
#         )
#         if not np.array_equal(
#             np.sort(train_indices),
#             np.sort(self.base_train_indices),
#         ):
#             raise RuntimeError(
#                 "base TRAIN membership disagrees with preprocessing membership"
#             )
#         if (
#             np.intersect1d(train_indices, val_indices).size
#             or np.intersect1d(train_indices, test_indices).size
#             or np.intersect1d(val_indices, test_indices).size
#         ):
#             raise RuntimeError("base train/val/test centre indices overlap")
#         observed = sorted(
#             int(v)
#             for v in np.unique(self.labels[train_indices])
#         )
#         if observed != sorted(base_ids):
#             raise RuntimeError(
#                 "base TRAIN split does not contain exactly the base classes"
#             )
#         return {
#             "base_class_ids": list(base_ids),
#             "train_count": int(train_indices.size),
#             "validation_count": int(val_indices.size),
#             "test_count": int(test_indices.size),
#             "preprocess_fit_count": int(self.base_train_indices.size),
#             "train_equals_preprocess_fit": True,
#             "center_splits_disjoint": True,
#             "validation_enabled": bool(self.validation_enabled),
#             "spatial_leakage": self.spatial_leakage_report(),
#         }

#     def protocol_report(self) -> Dict[str, Any]:
#         return {
#             "training_mode": "non_exemplar_class_incremental",
#             "protocol_seed": int(self.protocol_seed),
#             "optimization_dataloader_seed": int(self.seed),
#             "tasks": self.nb_tasks,
#             "phase_to_classes": {
#                 phase: list(classes)
#                 for phase, classes in self.phase_to_classes.items()
#             },
#             "class_order_original_ids": list(self.class_order_original_ids),
#             "split_strategy": self.split_strategy,
#             "spatial_partition_mode": self.spatial_partition_mode,
#             "current_phase": self.current_phase,
#             "finalized_phases": list(self.finalized_phases),
#             "single_head_global_labels": True,
#             "current_phase_training_classes_only": True,
#             "cumulative_seen_class_evaluation": True,
#             "old_training_data_public_api": False,
#             "stores_old_training_samples": False,
#             "validation_enabled": bool(self.validation_enabled),
#             "preprocessing_fit_phase": 0,
#             "base_train_sample_count": int(self.base_train_indices.size),
#             "preprocessing_frozen_across_phases": True,
#             "spectral_axis_type": self.spectral_axis_type,
#             "spectral_axis_is_physical": bool(self.spectral_axis_is_physical),
#             "spectral_axis_length": int(self.spectral_axis.size),
#             "spatial_leakage": self.spatial_leakage_report(),
#         }

#     def assert_exemplar_free_contract(self) -> bool:
#         if self.finalized_phases != list(range(len(self.finalized_phases))):
#             raise RuntimeError("finalized phase state is not contiguous")
#         if self.current_phase is not None:
#             if self.current_phase in self.finalized_phases:
#                 raise RuntimeError(
#                     "an active phase cannot also be finalized"
#                 )
#             if self.current_phase != len(self.finalized_phases):
#                 raise RuntimeError(
#                     "active phase must immediately follow finalized phases"
#                 )
#         if any(
#             token in self.__dict__
#             for token in ("exemplars", "replay_buffer", "memory_samples")
#         ):
#             raise RuntimeError("dataset manager contains forbidden replay state")
#         if self.require_patch_disjoint and not self._split_patch_disjoint:
#             raise RuntimeError("strict split patch-disjoint contract is violated")
#         if (
#             self.require_phase_context_isolation
#             and not self._phase_context_isolated
#         ):
#             raise RuntimeError(
#                 "strict phase-context isolation contract is violated"
#             )
#         _validated_spectral_axis(
#             self.spectral_axis,
#             raw_bands=int(self.ordered_spectral_cube.shape[-1]),
#             spectral_axis_type=self.spectral_axis_type,
#         )
#         return True


# __all__ = [
#     "build_incremental_protocol",
#     "IncrementalHSIPatchDataset",
#     "IncrementalHSIDatasetManager",
# ]








# from __future__ import annotations

# """Single leakage-auditable non-exemplar class-incremental HSI protocol.

# Protocol contract
# -----------------
# * Phase t training exposes only the training centres of classes introduced at t.
# * Validation during an active phase exposes only current-phase validation data.
# * Cumulative validation/test over all seen classes is evaluation-only and is
#   available only after the phase is finalized.
# * The spatial branch receives the processed/PCA patch.
# * The spectral branch receives the pre-PCA ordered centre spectrum from the
#   dataset cube; corrected datasets may already exclude sensor bands.
# * Those two inputs are fused by the backbone; the dataset does not create a
#   second learned representation.
# * No optimization/model-selection API returns historical training samples after its phase.
# * A finalized-phase map loader may read all labelled seen-class pixels strictly for
#   post-training qualitative reporting; those rows are never an optimization source.
# * Optional strict spatial audits detect patch-footprint overlap across data
#   splits and across unavailable old/future labelled samples.

# Important: a standard random-pixel HSI split is centre-disjoint but generally
# NOT patch-disjoint when patch_size > 1. Therefore it must not be described as
# spatial-leakage-free unless the strict audits actually pass.
# """

# import random
# from numbers import Integral, Real
# from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

# import numpy as np
# import torch
# from torch.utils.data import DataLoader, Dataset


# _SPLITS = ("train", "val", "test")


# def _exact_integer(value: object, *, name: str, minimum: Optional[int] = None) -> int:
#     """Return an exact signed-int64-compatible scalar without silent truncation."""
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

#     info = np.iinfo(np.int64)
#     if result < int(info.min) or result > int(info.max):
#         raise ValueError(f"{name} is outside signed int64 range")
#     if minimum is not None and result < minimum:
#         raise ValueError(f"{name} must be >= {minimum}")
#     return result


# def _exact_bool(value: object, *, name: str) -> bool:
#     """Parse an explicit Boolean without Python's bool('false') pitfall."""
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
#     """Return exact int64 data without Boolean, fractional, or overflow coercion."""
#     raw = np.asarray(value)
#     if raw.dtype == np.bool_ or raw.dtype.kind not in "iuf":
#         raise ValueError(f"{name} must contain integer values")

#     info = np.iinfo(np.int64)
#     if np.issubdtype(raw.dtype, np.floating):
#         if not np.isfinite(raw).all() or not np.equal(raw, np.round(raw)).all():
#             raise ValueError(f"{name} must contain finite integer values")
#         # float64 cannot represent int64.max exactly; use the exact
#         # representable half-open interval [-2**63, 2**63).
#         lower = float(-(2**63))
#         upper_exclusive = float(2**63)
#         if raw.size and (
#             np.any(raw < lower) or np.any(raw >= upper_exclusive)
#         ):
#             raise ValueError(f"{name} contains values outside signed int64 range")
#     elif raw.size and np.issubdtype(raw.dtype, np.unsignedinteger):
#         if np.any(raw > info.max):
#             raise ValueError(f"{name} contains values outside signed int64 range")
#     elif raw.size:
#         if np.any(raw < info.min) or np.any(raw > info.max):
#             raise ValueError(f"{name} contains values outside signed int64 range")

#     return raw.astype(np.int64, copy=False)


# def _ids(values: Iterable[int], *, name: str, allow_empty: bool = False) -> List[int]:
#     result = [
#         _exact_integer(value, name=f"{name} value", minimum=0)
#         for value in values
#     ]
#     if not result and not allow_empty:
#         raise ValueError(f"{name} cannot be empty")
#     if len(result) != len(set(result)):
#         raise ValueError(f"{name} must contain unique IDs")
#     return result


# def _normalize_split_name(split: str) -> str:
#     token = str(split).strip().lower()
#     return {
#         "training": "train",
#         "validation": "val",
#         "testing": "test",
#     }.get(token, token)


# _SPECTRAL_AXIS_ALIASES = {
#     "wavelength": "wavelength_nm",
#     "wavelengths": "wavelength_nm",
#     "wavelength_nm": "wavelength_nm",
#     "wavelengths_nm": "wavelength_nm",
#     "nm": "wavelength_nm",
#     "band": "ordered_band",
#     "bands": "ordered_band",
#     "band_index": "ordered_band",
#     "ordered_band": "ordered_band",
# }


# def _validated_spectral_axis(
#     spectral_axis: Optional[Sequence[float]],
#     *,
#     raw_bands: int,
#     spectral_axis_type: str,
# ) -> Tuple[np.ndarray, str]:
#     """Validate the one spectral coordinate shared by the model and CDS."""
#     bands = _exact_integer(raw_bands, name="raw_bands", minimum=1)
#     token = _SPECTRAL_AXIS_ALIASES.get(
#         str(spectral_axis_type).strip().lower()
#     )
#     if token is None:
#         raise ValueError(
#             "spectral_axis_type must be 'wavelength_nm' or 'ordered_band'"
#         )
#     if spectral_axis is None:
#         if token != "ordered_band":
#             raise ValueError(
#                 "physical wavelength coordinates must be supplied explicitly; "
#                 "they are never synthesized from band indices"
#             )
#         axis = np.arange(bands, dtype=np.float32)
#     else:
#         axis = np.asarray(spectral_axis, dtype=np.float64).reshape(-1)
#     if axis.shape != (bands,):
#         raise ValueError(
#             "spectral_axis must contain one coordinate per ordered raw band"
#         )
#     if not np.isfinite(axis).all():
#         raise ValueError("spectral_axis contains NaN/Inf")
#     if bands > 1 and not np.all(np.diff(axis) > 0.0):
#         raise ValueError("spectral_axis must be strictly increasing")
#     if token == "wavelength_nm" and np.any(axis <= 0.0):
#         raise ValueError("physical wavelengths in nanometres must be positive")
#     return axis.astype(np.float32), token


# def _seed_worker(_: int) -> None:
#     seed = int(torch.initial_seed() % (2**32))
#     random.seed(seed)
#     np.random.seed(seed)



# def _seed32(value: object, *, name: str) -> int:
#     """Validate the exact integer domain accepted by NumPy RandomState."""
#     result = _exact_integer(
#         value,
#         name=name,
#         minimum=0,
#     )
#     if result > (2**32) - 1:
#         raise ValueError(
#             f"{name} must be <= 2**32-1"
#         )
#     return result


# def _derived_seed32(base_seed: int, component: int) -> int:
#     """Derive a deterministic per-class seed without leaving uint32 range."""
#     base = _seed32(
#         base_seed,
#         name="base seed",
#     )
#     part = _exact_integer(
#         component,
#         name="seed component",
#         minimum=0,
#     )
#     return int(
#         (base + 1009 * part)
#         % (2**32)
#     )

# def _phase_schedule(
#     class_count: int,
#     base_classes: int,
#     increment: int,
# ) -> Tuple[List[int], Dict[int, List[int]]]:
#     total = _exact_integer(class_count, name="class_count", minimum=1)
#     base = _exact_integer(base_classes, name="base_classes", minimum=1)
#     step = _exact_integer(increment, name="increment", minimum=1)
#     if base > total:
#         raise ValueError("base_classes cannot exceed class_count")
#     sizes = [base]
#     remaining = total - base
#     while remaining:
#         current = min(step, remaining)
#         sizes.append(current)
#         remaining -= current
#     phases: Dict[int, List[int]] = {}
#     start = 0
#     for phase, size in enumerate(sizes):
#         phases[phase] = list(range(start, start + size))
#         start += size
#     return sizes, phases


# def _split_counts(
#     total: int,
#     *,
#     train_ratio: float,
#     val_ratio: float,
# ) -> Tuple[int, int, int]:
#     """Allocate one fixed per-class split; validation may be disabled.

#     Train and test are mandatory.  Validation is optional because some
#     published HSI benchmarks provide train/test masks only.
#     """
#     train_fraction = float(train_ratio)
#     val_fraction = float(val_ratio)

#     if (
#         not np.isfinite(train_fraction)
#         or not np.isfinite(val_fraction)
#         or train_fraction <= 0.0
#         or val_fraction < 0.0
#         or train_fraction + val_fraction >= 1.0
#     ):
#         raise ValueError(
#             "require finite train_ratio>0, val_ratio>=0, and "
#             "train_ratio+val_ratio<1"
#         )

#     required_split_count = 3 if val_fraction > 0.0 else 2
#     count = _exact_integer(
#         total,
#         name="class sample count",
#         minimum=required_split_count,
#     )

#     quotas = np.asarray(
#         (
#             train_fraction,
#             val_fraction,
#             1.0 - train_fraction - val_fraction,
#         ),
#         dtype=np.float64,
#     ) * count
#     counts = np.floor(quotas).astype(np.int64)
#     remainder = count - int(counts.sum())
#     if remainder:
#         order = np.argsort(-(quotas - counts), kind="stable")
#         counts[order[:remainder]] += 1

#     required = (0, 2) if val_fraction == 0.0 else (0, 1, 2)
#     for destination in required:
#         if counts[destination] == 0:
#             candidates = [
#                 index
#                 for index in required
#                 if index != destination and counts[index] > 1
#             ]
#             if not candidates:
#                 raise ValueError(
#                     "class is too small to realize the requested non-empty splits"
#                 )
#             source = max(candidates, key=lambda index: int(counts[index]))
#             counts[source] -= 1
#             counts[destination] += 1

#     if val_fraction == 0.0:
#         # A zero validation ratio is an explicit train/test-only protocol.
#         if counts[1] != 0:
#             counts[2] += counts[1]
#             counts[1] = 0

#     n_train, n_val, n_test = (int(v) for v in counts)
#     if n_train + n_val + n_test != count:
#         raise RuntimeError("split allocation does not conserve class samples")
#     if n_train <= 0 or n_test <= 0:
#         raise RuntimeError("train and test splits must be non-empty")
#     if val_fraction > 0.0 and n_val <= 0:
#         raise RuntimeError("positive val_ratio produced an empty validation split")
#     return n_train, n_val, n_test

# def _random_class_split(
#     indices: np.ndarray,
#     *,
#     train_ratio: float,
#     val_ratio: float,
#     seed: int,
# ) -> Dict[str, np.ndarray]:
#     values = _integer_array(indices, name="class indices").reshape(-1)
#     if np.unique(values).size != values.size:
#         raise ValueError("class indices contain duplicates")
#     n_train, n_val, _ = _split_counts(
#         len(values),
#         train_ratio=train_ratio,
#         val_ratio=val_ratio,
#     )
#     rng = np.random.RandomState(_exact_integer(seed, name="split seed", minimum=0))
#     shuffled = rng.permutation(values)
#     train = shuffled[:n_train]
#     val = shuffled[n_train : n_train + n_val]
#     test = shuffled[n_train + n_val :]
#     return {
#         "train": train,
#         "val": val,
#         "test": test,
#         "all": np.concatenate((train, val, test)),
#     }


# def _spatial_class_split(
#     indices: np.ndarray,
#     coords: np.ndarray,
#     *,
#     train_ratio: float,
#     val_ratio: float,
#     block_size: int,
#     seed: int,
# ) -> Dict[str, np.ndarray]:
#     """Classwise spatial-block split.

#     This is a benchmark split option, not a proof of patch disjointness.
#     Patch-overlap audits are computed separately.
#     """
#     values = _integer_array(indices, name="class indices").reshape(-1)
#     xy = _integer_array(coords, name="coords")[values]
#     size = _exact_integer(block_size, name="spatial_block_size", minimum=1)
#     max_block_col = int(np.max(xy[:, 1] // size)) + 1
#     block_ids = (xy[:, 0] // size) * max_block_col + xy[:, 1] // size
#     blocks = [values[block_ids == block] for block in np.unique(block_ids)]

#     validation_enabled = float(val_ratio) > 0.0
#     minimum_blocks = 3 if validation_enabled else 2
#     if len(blocks) < minimum_blocks:
#         raise RuntimeError(
#             "a class occupies too few spatial blocks for the requested split"
#         )

#     rng = np.random.RandomState(
#         _exact_integer(seed, name="split seed", minimum=0)
#     )
#     rng.shuffle(blocks)

#     target_train, target_val, _ = _split_counts(
#         len(values),
#         train_ratio=train_ratio,
#         val_ratio=val_ratio,
#     )

#     train_blocks: List[np.ndarray] = []
#     val_blocks: List[np.ndarray] = []
#     test_blocks: List[np.ndarray] = []
#     train_count = 0
#     val_count = 0

#     for block_index, block in enumerate(blocks):
#         remaining = len(blocks) - block_index
#         reserve_for_test = 1
#         reserve_for_val = 1 if validation_enabled and not val_blocks else 0

#         if (
#             train_count < target_train
#             and remaining > reserve_for_test + reserve_for_val
#         ):
#             train_blocks.append(block)
#             train_count += len(block)
#         elif (
#             validation_enabled
#             and val_count < target_val
#             and remaining > reserve_for_test
#         ):
#             val_blocks.append(block)
#             val_count += len(block)
#         else:
#             test_blocks.append(block)

#     if not train_blocks or not test_blocks:
#         raise RuntimeError("spatial block allocation produced an empty train/test split")
#     if validation_enabled and not val_blocks:
#         raise RuntimeError("spatial block allocation produced an empty validation split")

#     train = np.concatenate(train_blocks)
#     val = (
#         np.concatenate(val_blocks)
#         if val_blocks
#         else np.empty(0, dtype=np.int64)
#     )
#     test = np.concatenate(test_blocks)

#     return {
#         "train": train,
#         "val": val,
#         "test": test,
#         "all": np.concatenate((train, val, test)),
#     }

# def _validate_split_by_class(
#     split_by_class: Mapping[int, Mapping[str, np.ndarray]],
#     labels: np.ndarray,
#     class_count: int,
# ) -> None:
#     y = _integer_array(
#         labels,
#         name="global labels",
#     ).reshape(-1)
#     total_classes = _exact_integer(
#         class_count,
#         name="class_count",
#         minimum=1,
#     )

#     if y.size == 0:
#         raise ValueError(
#             "global labels cannot be empty"
#         )
#     observed_classes = sorted(
#         int(value)
#         for value in np.unique(y)
#     )
#     if observed_classes != list(
#         range(total_classes)
#     ):
#         raise ValueError(
#             "global labels must contain exactly contiguous class IDs "
#             f"0..{total_classes - 1}; observed={observed_classes}"
#         )

#     if set(int(v) for v in split_by_class) != set(range(total_classes)):
#         raise ValueError("split_by_class does not cover every class")

#     used: set[int] = set()
#     for class_id in range(total_classes):
#         row = split_by_class[class_id]
#         missing = set((*_SPLITS, "all")) - set(row)
#         if missing:
#             raise ValueError(
#                 f"class {class_id} split is missing {sorted(missing)}"
#             )

#         arrays = {
#             name: _integer_array(
#                 row[name],
#                 name=f"class {class_id} {name} indices",
#             ).reshape(-1)
#             for name in (*_SPLITS, "all")
#         }

#         if arrays["train"].size == 0 or arrays["test"].size == 0:
#             raise ValueError(
#                 f"class {class_id} must have non-empty train and test splits"
#             )
#         if arrays["all"].size == 0:
#             raise ValueError(f"class {class_id} all split cannot be empty")

#         for name, array in arrays.items():
#             if np.unique(array).size != array.size:
#                 raise ValueError(
#                     f"class {class_id} {name} split contains duplicates"
#                 )
#             if array.size and (array.min() < 0 or array.max() >= y.size):
#                 raise ValueError(
#                     f"class {class_id} {name} split contains invalid indices"
#                 )

#         train_set, val_set, test_set = (
#             set(arrays[name].tolist()) for name in _SPLITS
#         )
#         if train_set & val_set or train_set & test_set or val_set & test_set:
#             raise ValueError(f"class {class_id} train/val/test indices overlap")

#         union = train_set | val_set | test_set
#         if set(arrays["all"].tolist()) != union:
#             raise ValueError(f"class {class_id} all indices are inconsistent")

#         for array in arrays.values():
#             if array.size and not np.all(y[array] == class_id):
#                 raise ValueError(f"class {class_id} split contains another class")

#         expected = set(np.flatnonzero(y == class_id).tolist())
#         if union != expected:
#             raise ValueError(
#                 f"class {class_id} split does not account for every sample"
#             )
#         if used & expected:
#             raise ValueError("a sample appears in more than one class")
#         used |= expected

# def _collect_indices(
#     split_by_class: Mapping[int, Mapping[str, np.ndarray]],
#     class_ids: Sequence[int],
#     split: str,
# ) -> np.ndarray:
#     token = _normalize_split_name(split)
#     if token not in {*_SPLITS, "all"}:
#         raise ValueError(f"unknown split {split!r}")
#     ids = _ids(class_ids, name="class_ids", allow_empty=True)
#     if not ids:
#         return np.empty(0, dtype=np.int64)
#     return np.concatenate(
#         [
#             _integer_array(
#                 split_by_class[class_id][token],
#                 name=f"class {class_id} {token} indices",
#             ).reshape(-1)
#             for class_id in ids
#         ]
#     )


# def build_incremental_protocol(
#     input_labels: np.ndarray,
#     input_coords: np.ndarray,
#     *,
#     gt_shape: Sequence[int],
#     class_count: int,
#     base_classes: int,
#     increment: int,
#     train_ratio: float,
#     val_ratio: float,
#     seed: int = 0,
#     shuffle_order: bool = False,
#     class_order: Optional[Sequence[int]] = None,
#     target_names: Optional[Sequence[str]] = None,
#     split_strategy: str = "random_pixel",
#     spatial_block_size: int = 33,
#     predefined_splits: Optional[Mapping[int, Mapping[str, Sequence[int]]]] = None,
#     **_: Any,
# ) -> Dict[str, Any]:
#     original_labels = _integer_array(input_labels, name="input_labels").reshape(-1)
#     coords = _integer_array(input_coords, name="input_coords")
#     shape = tuple(
#         _exact_integer(v, name="gt_shape value", minimum=1) for v in gt_shape
#     )
#     total_classes = _exact_integer(class_count, name="class_count", minimum=1)
#     protocol_seed = _seed32(
#         seed,
#         name="seed",
#     )

#     if len(shape) != 2 or coords.shape != (len(original_labels), 2):
#         raise ValueError("labels, coordinates and gt_shape are inconsistent")
#     if original_labels.size == 0:
#         raise ValueError("input_labels cannot be empty")
#     if np.unique(coords, axis=0).shape[0] != coords.shape[0]:
#         raise ValueError("coords must identify unique labelled pixels")
#     if (
#         np.any(coords[:, 0] < 0)
#         or np.any(coords[:, 0] >= shape[0])
#         or np.any(coords[:, 1] < 0)
#         or np.any(coords[:, 1] >= shape[1])
#     ):
#         raise ValueError("coords lie outside gt_shape")

#     original_ids = sorted(int(v) for v in np.unique(original_labels))
#     if original_ids != list(range(total_classes)):
#         raise ValueError("input labels must be contiguous zero-based class IDs")

#     shuffle_classes = _exact_bool(shuffle_order, name="shuffle_order")
#     if class_order is not None and shuffle_classes:
#         raise ValueError("choose either class_order or shuffle_order, not both")
#     if class_order is not None:
#         order = _ids(class_order, name="class_order")
#     elif shuffle_classes:
#         order = np.random.RandomState(protocol_seed).permutation(
#             original_ids
#         ).tolist()
#     else:
#         order = original_ids
#     if set(order) != set(original_ids) or len(order) != len(original_ids):
#         raise ValueError("class_order must be a complete class permutation")

#     original_to_global = {
#         original: global_id for global_id, original in enumerate(order)
#     }
#     global_to_original = {
#         global_id: original for original, global_id in original_to_global.items()
#     }
#     global_labels = np.asarray(
#         [original_to_global[int(label)] for label in original_labels],
#         dtype=np.int64,
#     )
#     phase_sizes, phase_to_classes = _phase_schedule(
#         total_classes, base_classes, increment
#     )

#     strategy = str(split_strategy).strip().lower()
#     if predefined_splits is not None:
#         predefined_original = {
#             _exact_integer(class_id, name="predefined class ID", minimum=0): row
#             for class_id, row in predefined_splits.items()
#         }
#         if set(predefined_original) != set(original_ids):
#             raise ValueError(
#                 "predefined_splits must be keyed by every original class ID"
#             )
#         split_by_class = {}
#         for original_id, row in predefined_original.items():
#             if "train" not in row or "test" not in row:
#                 raise ValueError(
#                     f"predefined class {original_id} must provide train and test indices"
#                 )
#             mapped = {
#                 "train": _integer_array(
#                     row["train"],
#                     name=f"predefined class {original_id} train indices",
#                 ).reshape(-1),
#                 "val": _integer_array(
#                     row.get("val", np.empty(0, dtype=np.int64)),
#                     name=f"predefined class {original_id} val indices",
#                 ).reshape(-1),
#                 "test": _integer_array(
#                     row["test"],
#                     name=f"predefined class {original_id} test indices",
#                 ).reshape(-1),
#             }
#             mapped["all"] = np.concatenate(
#                 (mapped["train"], mapped["val"], mapped["test"])
#             )
#             split_by_class[original_to_global[original_id]] = mapped
#         strategy = "predefined"
#     elif strategy == "random_pixel":
#         split_by_class = {
#             class_id: _random_class_split(
#                 np.flatnonzero(global_labels == class_id),
#                 train_ratio=train_ratio,
#                 val_ratio=val_ratio,
#                 seed=_derived_seed32(
#                     protocol_seed,
#                     global_to_original[class_id],
#                 ),
#             )
#             for class_id in range(total_classes)
#         }
#     elif strategy == "spatial_block":
#         split_by_class = {
#             class_id: _spatial_class_split(
#                 np.flatnonzero(global_labels == class_id),
#                 coords,
#                 train_ratio=train_ratio,
#                 val_ratio=val_ratio,
#                 block_size=spatial_block_size,
#                 seed=_derived_seed32(
#                     protocol_seed,
#                     global_to_original[class_id],
#                 ),
#             )
#             for class_id in range(total_classes)
#         }
#     else:
#         raise ValueError(
#             "split_strategy must be random_pixel, spatial_block, or predefined"
#         )

#     _validate_split_by_class(split_by_class, global_labels, total_classes)

#     validation_presence = [
#         bool(split_by_class[class_id]["val"].size)
#         for class_id in range(total_classes)
#     ]
#     if any(validation_presence) and not all(validation_presence):
#         raise ValueError(
#             "validation must be consistently enabled for every class or "
#             "disabled for every class"
#         )
#     validation_enabled = all(validation_presence)

#     names = [] if target_names is None else [str(v) for v in target_names]
#     if names and len(names) != total_classes:
#         raise ValueError("target_names must contain one name per class")
#     global_names = [names[original] for original in order] if names else []

#     phase_train = {
#         phase: _collect_indices(split_by_class, classes, "train")
#         for phase, classes in phase_to_classes.items()
#     }
#     phase_val = {
#         phase: _collect_indices(split_by_class, classes, "val")
#         for phase, classes in phase_to_classes.items()
#     }
#     phase_test = {
#         phase: _collect_indices(split_by_class, classes, "test")
#         for phase, classes in phase_to_classes.items()
#     }
#     cumulative_test = {
#         phase: _collect_indices(
#             split_by_class,
#             list(range(sum(phase_sizes[: phase + 1]))),
#             "test",
#         )
#         for phase in phase_to_classes
#     }

#     return {
#         "training_mode": "non_exemplar_class_incremental",
#         "seed": protocol_seed,
#         "class_count": total_classes,
#         "base_classes": _exact_integer(
#             base_classes, name="base_classes", minimum=1
#         ),
#         "increment": _exact_integer(increment, name="increment", minimum=1),
#         "phase_sizes": phase_sizes,
#         "phase_to_classes": phase_to_classes,
#         "class_order_original_ids": order,
#         "original_to_global": original_to_global,
#         "global_to_original": global_to_original,
#         "original_labels": original_labels,
#         "global_labels": global_labels,
#         "global_target_names": global_names,
#         "split_by_class": split_by_class,
#         "phase_train_indices": phase_train,
#         "phase_val_indices": phase_val,
#         "phase_test_indices": phase_test,
#         "cumulative_test_indices": cumulative_test,
#         # Canonical preprocessing membership. This is exactly phase-0 TRAIN,
#         # not a second independently constructed split.
#         "base_train_indices": phase_train[0].copy(),
#         "split_strategy": strategy,
#         "spatial_partition_mode": (
#             "classwise_block_assignment"
#             if strategy == "spatial_block"
#             else "pixel_indices"
#         ),
#         "single_head_global_labels": True,
#         "current_phase_training_classes_only": True,
#         "cumulative_seen_class_evaluation": True,
#         "stores_old_training_samples": False,
#         "validation_enabled": bool(validation_enabled),
#         "preprocessing_fit_phase": 0,
#         "preprocessing_frozen_across_phases": True,
#     }


# class IncrementalHSIPatchDataset(Dataset):
#     def __init__(
#         self,
#         manager: "IncrementalHSIDatasetManager",
#         indices: Sequence[int],
#     ) -> None:
#         self.manager = manager
#         self.indices = _integer_array(indices, name="dataset indices").reshape(-1)
#         if self.indices.size == 0:
#             raise ValueError("dataset indices cannot be empty")
#         if self.indices.min() < 0 or self.indices.max() >= len(manager.labels):
#             raise ValueError("dataset indices are outside labelled samples")
#         if np.unique(self.indices).size != self.indices.size:
#             raise ValueError("dataset indices cannot contain duplicates")

#     def __len__(self) -> int:
#         return int(self.indices.size)

#     def __getitem__(self, item: int) -> Dict[str, torch.Tensor]:
#         item_id = _exact_integer(item, name="dataset item", minimum=0)
#         if item_id >= len(self):
#             raise IndexError(item_id)
#         sample_index = int(self.indices[item_id])
#         row, col = self.manager.coords[sample_index]
#         patch = self.manager.extract_patch(int(row), int(col))
#         raw_center_spectrum = self.manager.ordered_spectral_cube[
#             int(row), int(col)
#         ]
#         return {
#             "image": torch.from_numpy(
#                 np.moveaxis(patch, -1, 0).copy()
#             ).float(),
#             "raw_center_spectrum": torch.from_numpy(
#                 raw_center_spectrum.copy()
#             ).float(),
#             "label": torch.tensor(
#                 int(self.manager.labels[sample_index]),
#                 dtype=torch.long,
#             ),
#             "coord": torch.tensor(
#                 (int(row), int(col)),
#                 dtype=torch.long,
#             ),
#             "sample_index": torch.tensor(sample_index, dtype=torch.long),
#         }


# class IncrementalHSIDatasetManager:
#     """Current-class training and seen-class evaluation with spatial audits."""

#     def __init__(
#         self,
#         *,
#         processed_cube: np.ndarray,
#         labels: np.ndarray,
#         coords: np.ndarray,
#         protocol: Mapping[str, Any],
#         target_names: Sequence[str],
#         gt_shape: Sequence[int],
#         patch_size: int,
#         ordered_spectral_cube: Optional[np.ndarray] = None,
#         spectral_axis: Optional[Sequence[float]] = None,
#         spectral_axis_type: str = "ordered_band",
#         num_workers: int = 0,
#         device: str = "cpu",
#         seed: Optional[int] = None,
#         require_patch_disjoint: bool = False,
#         require_phase_context_isolation: Optional[bool] = None,
#         context_policy: str = "full_scene_reflect",
#         **_: Any,
#     ) -> None:
#         self.processed_cube = np.ascontiguousarray(
#             processed_cube, dtype=np.float32
#         )
#         self.gt_shape = tuple(
#             _exact_integer(v, name="gt_shape value", minimum=1)
#             for v in gt_shape
#         )
#         self.coords = _integer_array(coords, name="coords")
#         supplied_labels = _integer_array(labels, name="labels").reshape(-1)
#         original_labels = _integer_array(
#             protocol["original_labels"], name="protocol original_labels"
#         ).reshape(-1)
#         global_labels = _integer_array(
#             protocol["global_labels"], name="protocol global labels"
#         ).reshape(-1)
#         if (
#             np.array_equal(
#                 supplied_labels,
#                 original_labels,
#             )
#             or np.array_equal(
#                 supplied_labels,
#                 global_labels,
#             )
#         ):
#             self.labels = global_labels
#         else:
#             raise ValueError(
#                 "labels do not match the protocol"
#             )

#         protocol_class_count = _exact_integer(
#             protocol["class_count"],
#             name="protocol class_count",
#             minimum=1,
#         )
#         observed_global_classes = sorted(
#             int(value)
#             for value in np.unique(self.labels)
#         )
#         if observed_global_classes != list(
#             range(protocol_class_count)
#         ):
#             raise ValueError(
#                 "protocol global labels must contain exactly contiguous "
#                 f"class IDs 0..{protocol_class_count - 1}; "
#                 f"observed={observed_global_classes}"
#             )

#         if (
#             self.processed_cube.ndim != 3
#             or self.processed_cube.shape[:2] != self.gt_shape
#             or self.processed_cube.shape[2] <= 0
#             or not np.isfinite(self.processed_cube).all()
#         ):
#             raise ValueError("processed_cube must be finite and match gt_shape")
#         if self.coords.shape != (len(self.labels), 2):
#             raise ValueError("coords must align with labels")
#         if np.unique(self.coords, axis=0).shape[0] != self.coords.shape[0]:
#             raise ValueError("coords contain duplicate labelled-pixel locations")
#         if (
#             np.any(self.coords[:, 0] < 0)
#             or np.any(self.coords[:, 0] >= self.gt_shape[0])
#             or np.any(self.coords[:, 1] < 0)
#             or np.any(self.coords[:, 1] >= self.gt_shape[1])
#         ):
#             raise ValueError("coords contain positions outside gt_shape")

#         if ordered_spectral_cube is None:
#             raise ValueError(
#                 "ordered_spectral_cube is required for the raw spectral branch"
#             )
#         ordered = np.asarray(ordered_spectral_cube, dtype=np.float32)
#         if (
#             ordered.ndim != 3
#             or ordered.shape[:2] != self.gt_shape
#             or ordered.shape[2] <= 0
#             or not np.isfinite(ordered).all()
#         ):
#             raise ValueError("ordered_spectral_cube is invalid")
#         self.ordered_spectral_cube = np.ascontiguousarray(ordered)
#         self.spectral_axis, self.spectral_axis_type = _validated_spectral_axis(
#             spectral_axis,
#             raw_bands=int(self.ordered_spectral_cube.shape[-1]),
#             spectral_axis_type=spectral_axis_type,
#         )
#         self.spectral_axis_is_physical = (
#             self.spectral_axis_type == "wavelength_nm"
#         )

#         self.patch_size = _exact_integer(
#             patch_size, name="patch_size", minimum=1
#         )
#         if self.patch_size % 2 == 0:
#             raise ValueError("patch_size must be odd")
#         radius = self.patch_size // 2
#         self.padded_cube = np.pad(
#             self.processed_cube,
#             ((radius, radius), (radius, radius), (0, 0)),
#             mode="reflect",
#         )

#         self.phase_to_classes = {
#             _exact_integer(phase, name="phase ID", minimum=0): _ids(
#                 classes, name=f"phase_to_classes[{phase}]"
#             )
#             for phase, classes in protocol["phase_to_classes"].items()
#         }
#         if sorted(self.phase_to_classes) != list(range(len(self.phase_to_classes))):
#             raise ValueError("phase IDs must be contiguous from zero")
#         class_count = protocol_class_count
#         flattened = [
#             class_id
#             for phase in range(len(self.phase_to_classes))
#             for class_id in self.phase_to_classes[phase]
#         ]
#         if flattened != list(range(class_count)):
#             raise ValueError(
#                 "phase schedule must cover global class IDs once in order"
#             )

#         # The protocol must expose one canonical base-training membership for
#         # preprocessing, and it must be exactly the phase-0 training split.
#         if "base_train_indices" not in protocol:
#             raise ValueError("protocol is missing base_train_indices")
#         if "phase_train_indices" not in protocol:
#             raise ValueError("protocol is missing phase_train_indices")
#         phase_train_indices = protocol["phase_train_indices"]
#         if not isinstance(phase_train_indices, Mapping) or 0 not in phase_train_indices:
#             raise ValueError("protocol phase_train_indices must contain phase 0")
#         base_train_indices = _integer_array(
#             protocol["base_train_indices"],
#             name="protocol base_train_indices",
#         ).reshape(-1)
#         phase0_train_indices = _integer_array(
#             phase_train_indices[0],
#             name="protocol phase_train_indices[0]",
#         ).reshape(-1)
#         if (
#             base_train_indices.size == 0
#             or phase0_train_indices.size == 0
#             or not np.array_equal(
#                 np.sort(base_train_indices),
#                 np.sort(phase0_train_indices),
#             )
#         ):
#             raise ValueError(
#                 "protocol base_train_indices must equal phase_train_indices[0]"
#             )
#         self.base_train_indices = base_train_indices.copy()

#         self._split_by_class = {
#             _exact_integer(class_id, name="split class ID", minimum=0): {
#                 name: _integer_array(
#                     values, name=f"class {class_id} {name} indices"
#                 )
#                 for name, values in row.items()
#             }
#             for class_id, row in protocol["split_by_class"].items()
#         }
#         _validate_split_by_class(
#             self._split_by_class, self.labels, class_count
#         )

#         # Validate every protocol membership field against the single canonical
#         # split_by_class state.  This prevents preprocessing/training/reporting
#         # from silently consuming different sample memberships.
#         for field, split_name in (
#             ("phase_train_indices", "train"),
#             ("phase_val_indices", "val"),
#             ("phase_test_indices", "test"),
#         ):
#             raw = protocol.get(field)
#             if not isinstance(raw, Mapping):
#                 raise ValueError(f"protocol {field} must be a phase mapping")
#             if set(int(key) for key in raw) != set(self.phase_to_classes):
#                 raise ValueError(
#                     f"protocol {field} must contain every phase exactly once"
#                 )

#             for phase_id, classes in self.phase_to_classes.items():
#                 declared = _integer_array(
#                     raw[phase_id],
#                     name=f"protocol {field}[{phase_id}]",
#                 ).reshape(-1)
#                 expected = _collect_indices(
#                     self._split_by_class,
#                     classes,
#                     split_name,
#                 )
#                 if not np.array_equal(
#                     np.sort(declared),
#                     np.sort(expected),
#                 ):
#                     raise ValueError(
#                         f"protocol {field}[{phase_id}] disagrees with split_by_class"
#                     )

#         cumulative_test = protocol.get("cumulative_test_indices")
#         if not isinstance(cumulative_test, Mapping):
#             raise ValueError(
#                 "protocol cumulative_test_indices must be a phase mapping"
#             )
#         if set(int(key) for key in cumulative_test) != set(self.phase_to_classes):
#             raise ValueError(
#                 "protocol cumulative_test_indices must contain every phase"
#             )
#         for phase_id in self.phase_to_classes:
#             declared = _integer_array(
#                 cumulative_test[phase_id],
#                 name=f"protocol cumulative_test_indices[{phase_id}]",
#             ).reshape(-1)
#             expected = _collect_indices(
#                 self._split_by_class,
#                 self.get_seen_classes(phase_id),
#                 "test",
#             )
#             if not np.array_equal(
#                 np.sort(declared),
#                 np.sort(expected),
#             ):
#                 raise ValueError(
#                     "protocol cumulative_test_indices "
#                     f"[{phase_id}] disagrees with split_by_class"
#                 )

#         preprocessing_fit_phase = _exact_integer(
#             protocol.get("preprocessing_fit_phase", 0),
#             name="protocol preprocessing_fit_phase",
#             minimum=0,
#         )
#         if preprocessing_fit_phase != 0:
#             raise ValueError("persistent HSI preprocessing must be fitted in phase 0")

#         preprocessing_frozen = _exact_bool(
#             protocol.get("preprocessing_frozen_across_phases", True),
#             name="protocol preprocessing_frozen_across_phases",
#         )
#         if not preprocessing_frozen:
#             raise ValueError(
#                 "persistent HSI preprocessing must remain frozen across phases"
#             )

#         names = [str(v) for v in target_names]
#         global_names = [
#             str(v) for v in protocol.get("global_target_names", [])
#         ]
#         if global_names and len(global_names) != class_count:
#             raise ValueError("protocol global_target_names has wrong length")
#         if not global_names and len(names) != class_count:
#             raise ValueError("target_names must contain one name per class")
#         self.class_order_original_ids = _ids(
#             protocol["class_order_original_ids"],
#             name="class_order_original_ids",
#         )
#         if set(self.class_order_original_ids) != set(range(class_count)):
#             raise ValueError(
#                 "class_order_original_ids must be a complete original-class permutation"
#             )
#         self.target_names = global_names or [
#             names[index] for index in self.class_order_original_ids
#         ]

#         self.global_to_original = {
#             _exact_integer(k, name="global class ID", minimum=0):
#             _exact_integer(v, name="original class ID", minimum=0)
#             for k, v in protocol["global_to_original"].items()
#         }
#         expected_global_to_original = {
#             global_id: original_id
#             for global_id, original_id in enumerate(self.class_order_original_ids)
#         }
#         if self.global_to_original != expected_global_to_original:
#             raise ValueError(
#                 "global_to_original disagrees with class_order_original_ids"
#             )

#         self.split_strategy = str(protocol["split_strategy"]).strip().lower()
#         self.spatial_partition_mode = str(
#             protocol.get("spatial_partition_mode", "pixel_indices")
#         )

#         validation_presence = [
#             bool(self._split_by_class[class_id]["val"].size)
#             for class_id in range(class_count)
#         ]
#         if any(validation_presence) and not all(validation_presence):
#             raise ValueError(
#                 "validation must be present for every class or absent for every class"
#             )
#         actual_validation_enabled = all(validation_presence)
#         declared_validation_enabled = _exact_bool(
#             protocol.get("validation_enabled", actual_validation_enabled),
#             name="protocol validation_enabled",
#         )
#         if declared_validation_enabled != actual_validation_enabled:
#             raise ValueError(
#                 "protocol validation_enabled disagrees with validation split membership"
#             )
#         self.validation_enabled = actual_validation_enabled

#         self.require_patch_disjoint = _exact_bool(
#             require_patch_disjoint,
#             name="require_patch_disjoint",
#         )
#         self.require_phase_context_isolation = (
#             self.require_patch_disjoint
#             if require_phase_context_isolation is None
#             else _exact_bool(
#                 require_phase_context_isolation,
#                 name="require_phase_context_isolation",
#             )
#         )
#         self.context_policy = str(context_policy).strip()
#         if self.context_policy != "full_scene_reflect":
#             raise ValueError(
#                 "only context_policy='full_scene_reflect' is implemented"
#             )

#         # Protocol randomness and optimization/DataLoader randomness are
#         # intentionally separate. ``protocol_seed`` fixes class order and
#         # train/val/test membership; ``self.seed`` controls only DataLoader
#         # shuffling/worker randomness for this optimization run.  Multi-seed
#         # robustness therefore supports, for example, protocol_seed=42 with
#         # optimization seeds 42/123/70 without changing the benchmark split.
#         self.protocol_seed = _seed32(
#             protocol["seed"],
#             name="protocol seed",
#         )
#         self.seed = (
#             self.protocol_seed
#             if seed is None
#             else _seed32(seed, name="dataset-manager optimization seed")
#         )

#         self.num_workers = _exact_integer(
#             num_workers, name="num_workers", minimum=0
#         )
#         self.pin_memory = str(device).startswith("cuda")
#         self.current_phase: Optional[int] = None
#         self.finalized_phases: List[int] = []

#         self._split_patch_disjoint = self._audit_split_patch_disjoint()
#         self._phase_context_isolated = self._audit_phase_context_isolation()

#         if self.require_patch_disjoint and not self._split_patch_disjoint:
#             raise RuntimeError(
#                 "train/val/test patch footprints overlap. A random-pixel centre "
#                 "split is not spatial-leakage-free for patch models. Use a "
#                 "guarded/predefined spatial split or explicitly disable the "
#                 "strict audit only for benchmark-comparability experiments."
#             )
#         if (
#             self.require_phase_context_isolation
#             and not self._phase_context_isolated
#         ):
#             raise RuntimeError(
#                 "at least one phase-training patch contains an annotated labeled "
#                 "center unavailable in that phase (old, future, validation, or test). "
#                 "This violates strict spatial phase isolation."
#             )

#     @property
#     def nb_tasks(self) -> int:
#         return len(self.phase_to_classes)

#     @property
#     def base_classes(self) -> List[int]:
#         return list(self.phase_to_classes[0])

#     def _validate_phase(self, phase: int) -> int:
#         value = _exact_integer(phase, name="phase", minimum=0)
#         if value not in self.phase_to_classes:
#             raise ValueError(f"unknown phase {phase}")
#         return value

#     def start_phase(self, phase: int) -> None:
#         value = self._validate_phase(phase)
#         if self.current_phase is not None:
#             raise RuntimeError(f"phase {self.current_phase} is already active")
#         if value in self.finalized_phases:
#             raise RuntimeError(f"phase {value} is already finalized")
#         if self.finalized_phases != list(range(value)):
#             raise RuntimeError(
#                 "phases must start in order after prior finalization"
#             )
#         self.current_phase = value

#     def finalize_phase(self, phase: int) -> None:
#         value = self._validate_phase(phase)
#         if self.current_phase != value:
#             raise RuntimeError(f"phase {value} is not the active phase")
#         self.finalized_phases.append(value)
#         self.current_phase = None

#     def restore_finalized_phases(
#         self,
#         phases: Sequence[int],
#     ) -> None:
#         """Restore checkpoint protocol state without exposing historical data."""
#         if self.current_phase is not None:
#             raise RuntimeError(
#                 "cannot restore finalized phases while a phase is active"
#             )
#         if self.finalized_phases:
#             raise RuntimeError(
#                 "restore_finalized_phases requires a fresh unfinalized manager"
#             )
#         values = [
#             self._validate_phase(value)
#             for value in phases
#         ]
#         if values != list(range(len(values))):
#             raise ValueError(
#                 "restored finalized phases must be contiguous from zero"
#             )
#         self.finalized_phases = values

#     def get_task_size(self, phase: int) -> int:
#         return len(self.phase_to_classes[self._validate_phase(phase)])

#     def get_new_classes(self, phase: int) -> List[int]:
#         return list(self.phase_to_classes[self._validate_phase(phase)])

#     def get_old_classes(self, phase: int) -> List[int]:
#         value = self._validate_phase(phase)
#         return [
#             class_id
#             for previous in range(value)
#             for class_id in self.phase_to_classes[previous]
#         ]

#     def get_seen_classes(self, phase: int) -> List[int]:
#         return self.get_old_classes(phase) + self.get_new_classes(phase)

#     def extract_patch(self, row: int, col: int) -> np.ndarray:
#         row_id = _exact_integer(row, name="patch row", minimum=0)
#         col_id = _exact_integer(col, name="patch col", minimum=0)
#         if row_id >= self.gt_shape[0] or col_id >= self.gt_shape[1]:
#             raise ValueError("patch centre lies outside gt_shape")
#         return self.padded_cube[
#             row_id : row_id + self.patch_size,
#             col_id : col_id + self.patch_size,
#         ]

#     def _indices_mask(self, indices: np.ndarray) -> np.ndarray:
#         mask = np.zeros(self.gt_shape, dtype=bool)
#         values = _integer_array(indices, name="audit indices").reshape(-1)
#         if values.size:
#             if values.min() < 0 or values.max() >= len(self.labels):
#                 raise ValueError("audit indices are outside labelled samples")
#             if np.unique(values).size != values.size:
#                 raise ValueError("audit indices contain duplicates")
#             xy = self.coords[values]
#             mask[xy[:, 0], xy[:, 1]] = True
#         return mask

#     def _patch_overlap_count(
#         self,
#         left_indices: np.ndarray,
#         right_indices: np.ndarray,
#     ) -> int:
#         """Count left centers whose extracted patch overlaps a right patch."""
#         left = _integer_array(left_indices, name="left indices").reshape(-1)
#         right = _integer_array(right_indices, name="right indices").reshape(-1)
#         if left.size == 0 or right.size == 0:
#             return 0

#         # Two odd P x P patches overlap when center Chebyshev distance <= P-1.
#         from scipy.ndimage import maximum_filter

#         right_mask = self._indices_mask(right)
#         radius = self.patch_size - 1
#         dilated = (
#             right_mask
#             if radius == 0
#             else maximum_filter(
#                 right_mask.astype(np.uint8),
#                 size=2 * radius + 1,
#                 mode="constant",
#                 cval=0,
#             ).astype(bool)
#         )
#         xy = self.coords[left]
#         return int(dilated[xy[:, 0], xy[:, 1]].sum())

#     def _forbidden_center_in_patch_count(
#         self,
#         train_indices: np.ndarray,
#         forbidden_indices: np.ndarray,
#     ) -> int:
#         """Count train patches containing an unavailable labeled center.

#         This differs from patch-vs-patch overlap. During phase-t training the
#         unavailable old/future/validation/test sample is not itself extracted as
#         a patch. The strict observable leakage event is that its labeled center
#         spectrum lies inside the current train patch.
#         """
#         train = _integer_array(train_indices, name="train indices").reshape(-1)
#         forbidden = _integer_array(
#             forbidden_indices, name="forbidden indices"
#         ).reshape(-1)
#         if train.size == 0 or forbidden.size == 0:
#             return 0

#         from scipy.ndimage import maximum_filter

#         forbidden_mask = self._indices_mask(forbidden)
#         radius = self.patch_size // 2
#         dilated = (
#             forbidden_mask
#             if radius == 0
#             else maximum_filter(
#                 forbidden_mask.astype(np.uint8),
#                 size=2 * radius + 1,
#                 mode="constant",
#                 cval=0,
#             ).astype(bool)
#         )
#         xy = self.coords[train]
#         return int(dilated[xy[:, 0], xy[:, 1]].sum())

#     def _audit_split_patch_disjoint(self) -> bool:
#         all_classes = list(range(len(self._split_by_class)))
#         split_indices = {
#             split: _collect_indices(
#                 self._split_by_class, all_classes, split
#             )
#             for split in _SPLITS
#         }
#         for left_index, left_name in enumerate(_SPLITS):
#             for right_name in _SPLITS[left_index + 1 :]:
#                 if self._patch_overlap_count(
#                     split_indices[left_name],
#                     split_indices[right_name],
#                 ):
#                     return False
#         return True

#     def _audit_phase_context_isolation(self) -> bool:
#         all_indices = np.arange(len(self.labels), dtype=np.int64)
#         for phase in range(self.nb_tasks):
#             available = _collect_indices(
#                 self._split_by_class,
#                 self.phase_to_classes[phase],
#                 "train",
#             )
#             available_set = np.zeros(len(self.labels), dtype=bool)
#             available_set[available] = True
#             forbidden = all_indices[~available_set]
#             if self._forbidden_center_in_patch_count(available, forbidden):
#                 return False
#         return True

#     def spatial_leakage_report(self) -> Dict[str, Any]:
#         return {
#             "patch_size": self.patch_size,
#             "split_patch_disjoint": bool(self._split_patch_disjoint),
#             "phase_training_context_isolated": bool(
#                 self._phase_context_isolated
#             ),
#             "strict_split_required": self.require_patch_disjoint,
#             "strict_phase_context_required": self.require_phase_context_isolation,
#             "context_policy": self.context_policy,
#             "guarantee_scope": (
#                 "train/val/test extracted patch footprints plus unavailable annotated "
#                 "labeled centers inside phase-training patches; unlabeled/background "
#                 "scene pixels cannot be assigned a task identity by the benchmark"
#             ),
#         }

#     def _make_dataset(
#         self,
#         class_ids: Sequence[int],
#         split: str,
#     ) -> IncrementalHSIPatchDataset:
#         return IncrementalHSIPatchDataset(
#             self,
#             _collect_indices(self._split_by_class, class_ids, split),
#         )

#     def _loader(
#         self,
#         dataset: Dataset,
#         *,
#         batch_size: int,
#         shuffle: bool,
#         phase: int,
#         split: str,
#     ) -> DataLoader:
#         size = _exact_integer(batch_size, name="batch_size", minimum=1)
#         phase_id = self._validate_phase(phase)
#         offset = {"train": 11, "val": 17, "test": 23, "all": 29}[split]
#         generator = torch.Generator().manual_seed(
#             self.seed + 1009 * phase_id + offset
#         )
#         return DataLoader(
#             dataset,
#             batch_size=size,
#             shuffle=_exact_bool(shuffle, name="shuffle"),
#             num_workers=self.num_workers,
#             pin_memory=self.pin_memory,
#             drop_last=False,
#             persistent_workers=self.num_workers > 0,
#             worker_init_fn=_seed_worker if self.num_workers > 0 else None,
#             generator=generator,
#         )

#     def get_phase_dataloader(
#         self,
#         phase: int,
#         split: str = "train",
#         batch_size: int = 64,
#         shuffle: bool = True,
#     ) -> DataLoader:
#         phase_id = self._validate_phase(phase)
#         if self.current_phase != phase_id:
#             raise RuntimeError(
#                 "phase train/validation loaders require an active phase"
#             )
#         token = _normalize_split_name(split)
#         if token not in {"train", "val"}:
#             raise ValueError("active phase loaders are train/val only")
#         if token == "val" and not self.validation_enabled:
#             raise RuntimeError("validation is disabled by this experiment protocol")
#         dataset = self._make_dataset(
#             self.get_new_classes(phase_id), token
#         )
#         return self._loader(
#             dataset,
#             batch_size=batch_size,
#             shuffle=shuffle if token == "train" else False,
#             phase=phase_id,
#             split=token,
#         )

#     def get_phase_structure_dataloader(
#         self,
#         phase: int,
#         *,
#         batch_size: int = 256,
#     ) -> DataLoader:
#         """Deterministic full current-TRAIN loader for structure estimation.

#         This deliberately reuses the active phase TRAIN membership, exposes no
#         historical rows, disables shuffling, and therefore provides the single
#         legal source for epoch refresh/final commit of current-class
#         I/V/Mspec/Vspec.
#         """
#         return self.get_phase_dataloader(
#             phase,
#             split="train",
#             batch_size=batch_size,
#             shuffle=False,
#         )

#     def get_cumulative_dataloader(
#         self,
#         phase: int,
#         split: str = "test",
#         batch_size: int = 256,
#         shuffle: bool = False,
#     ) -> DataLoader:
#         phase_id = self._validate_phase(phase)
#         if phase_id not in self.finalized_phases:
#             raise RuntimeError(
#                 "cumulative evaluation requires a finalized phase"
#             )
#         token = _normalize_split_name(split)
#         if token not in {"val", "test"}:
#             raise ValueError("cumulative loaders are evaluation-only")
#         if token == "val" and not self.validation_enabled:
#             raise RuntimeError("validation is disabled by this experiment protocol")
#         dataset = self._make_dataset(
#             self.get_seen_classes(phase_id), token
#         )
#         return self._loader(
#             dataset,
#             batch_size=batch_size,
#             shuffle=bool(shuffle),
#             phase=phase_id,
#             split=token,
#         )

#     def get_reporting_dataloader(
#         self,
#         phase: int,
#         *,
#         batch_size: int = 256,
#     ) -> DataLoader:
#         """Test-only reporting loader. Never mixes train/val samples into metrics."""
#         return self.get_cumulative_dataloader(
#             phase,
#             split="test",
#             batch_size=batch_size,
#             shuffle=False,
#         )

#     def get_map_dataloader(
#         self,
#         phase: int,
#         *,
#         batch_size: int = 256,
#     ) -> DataLoader:
#         """All labeled seen-class pixels for qualitative maps only; never metrics/training."""
#         phase_id = self._validate_phase(phase)
#         if phase_id not in self.finalized_phases:
#             raise RuntimeError("map generation requires a finalized phase")
#         dataset = self._make_dataset(
#             self.get_seen_classes(phase_id), "all"
#         )
#         return self._loader(
#             dataset,
#             batch_size=batch_size,
#             shuffle=False,
#             phase=phase_id,
#             split="all",
#         )

#     def base_phase_contract_report(self) -> Dict[str, Any]:
#         """Return read-only checks for the finalized phase-0 data contract."""
#         base_ids = self.get_new_classes(0)
#         train_indices = _collect_indices(
#             self._split_by_class,
#             base_ids,
#             "train",
#         )
#         val_indices = _collect_indices(
#             self._split_by_class,
#             base_ids,
#             "val",
#         )
#         test_indices = _collect_indices(
#             self._split_by_class,
#             base_ids,
#             "test",
#         )
#         if not np.array_equal(
#             np.sort(train_indices),
#             np.sort(self.base_train_indices),
#         ):
#             raise RuntimeError(
#                 "base TRAIN membership disagrees with preprocessing membership"
#             )
#         if (
#             np.intersect1d(train_indices, val_indices).size
#             or np.intersect1d(train_indices, test_indices).size
#             or np.intersect1d(val_indices, test_indices).size
#         ):
#             raise RuntimeError("base train/val/test centre indices overlap")
#         observed = sorted(
#             int(v)
#             for v in np.unique(self.labels[train_indices])
#         )
#         if observed != sorted(base_ids):
#             raise RuntimeError(
#                 "base TRAIN split does not contain exactly the base classes"
#             )
#         return {
#             "base_class_ids": list(base_ids),
#             "train_count": int(train_indices.size),
#             "validation_count": int(val_indices.size),
#             "test_count": int(test_indices.size),
#             "preprocess_fit_count": int(self.base_train_indices.size),
#             "train_equals_preprocess_fit": True,
#             "center_splits_disjoint": True,
#             "validation_enabled": bool(self.validation_enabled),
#             "spatial_leakage": self.spatial_leakage_report(),
#         }

#     def protocol_report(self) -> Dict[str, Any]:
#         return {
#             "training_mode": "non_exemplar_class_incremental",
#             "protocol_seed": int(self.protocol_seed),
#             "optimization_dataloader_seed": int(self.seed),
#             "tasks": self.nb_tasks,
#             "phase_to_classes": {
#                 phase: list(classes)
#                 for phase, classes in self.phase_to_classes.items()
#             },
#             "class_order_original_ids": list(self.class_order_original_ids),
#             "split_strategy": self.split_strategy,
#             "spatial_partition_mode": self.spatial_partition_mode,
#             "current_phase": self.current_phase,
#             "finalized_phases": list(self.finalized_phases),
#             "single_head_global_labels": True,
#             "current_phase_training_classes_only": True,
#             "cumulative_seen_class_evaluation": True,
#             "old_training_data_public_api": False,
#             "stores_old_training_samples": False,
#             "validation_enabled": bool(self.validation_enabled),
#             "preprocessing_fit_phase": 0,
#             "base_train_sample_count": int(self.base_train_indices.size),
#             "preprocessing_frozen_across_phases": True,
#             "spectral_axis_type": self.spectral_axis_type,
#             "spectral_axis_is_physical": bool(self.spectral_axis_is_physical),
#             "spectral_axis_length": int(self.spectral_axis.size),
#             "spatial_leakage": self.spatial_leakage_report(),
#         }

#     def assert_exemplar_free_contract(self) -> bool:
#         if self.finalized_phases != list(range(len(self.finalized_phases))):
#             raise RuntimeError("finalized phase state is not contiguous")
#         if self.current_phase is not None:
#             if self.current_phase in self.finalized_phases:
#                 raise RuntimeError(
#                     "an active phase cannot also be finalized"
#                 )
#             if self.current_phase != len(self.finalized_phases):
#                 raise RuntimeError(
#                     "active phase must immediately follow finalized phases"
#                 )
#         if any(
#             token in self.__dict__
#             for token in ("exemplars", "replay_buffer", "memory_samples")
#         ):
#             raise RuntimeError("dataset manager contains forbidden replay state")
#         if self.require_patch_disjoint and not self._split_patch_disjoint:
#             raise RuntimeError("strict split patch-disjoint contract is violated")
#         if (
#             self.require_phase_context_isolation
#             and not self._phase_context_isolated
#         ):
#             raise RuntimeError(
#                 "strict phase-context isolation contract is violated"
#             )
#         _validated_spectral_axis(
#             self.spectral_axis,
#             raw_bands=int(self.ordered_spectral_cube.shape[-1]),
#             spectral_axis_type=self.spectral_axis_type,
#         )
#         return True


# __all__ = [
#     "build_incremental_protocol",
#     "IncrementalHSIPatchDataset",
#     "IncrementalHSIDatasetManager",
# ]
