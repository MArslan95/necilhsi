from __future__ import annotations

"""Reporting-only visualization for the complete role-separated NECIL-HSI architecture.

The visualizations follow the same semantic separation as the model:

1. Classifier visualizations
   - qualitative prediction maps;
   - per-class accuracy;
   - cosine decision margin.

2. Class-structure visualizations
   - retained class identity I_c;
   - retained class structure S_c;
   - inter-class identity/structure differences.

3. Incremental-preservation visualizations
   - old/new cumulative accuracy;
   - stability-plasticity summary;
   - structural preservation spectrum.

4. Representation-evolution visualization
   - one JOINT t-SNE embedding for base-vs-incremental features.

Important
---------
These functions are reporting-only. They never participate in optimization,
checkpoint state, exemplar storage, replay, or class decisions.

Internal labels are zero-based global class IDs. Rendered map value 0 is
reserved for background/unseen pixels; semantic class c is rendered as c+1.

The geometry/class-structure statistics {I_c,S_c,N_c} never produce prediction
logits. The angular classifier remains the only decision mechanism.

For incremental t-SNE, the two phases are embedded jointly. Independent t-SNE
fits must not be compared as feature drift because their coordinates are not
aligned.
"""

import math
import os
from numbers import Integral, Real
from typing import Any, Dict, Mapping, Optional, Sequence

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Ellipse, Patch


# -----------------------------------------------------------------------------
# Generic validation / naming
# -----------------------------------------------------------------------------


def _as_int(value: object, name: str) -> int:
    """Convert an exact integer scalar without silent truncation."""
    if isinstance(value, (np.bool_, bool)):
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

    int64 = np.iinfo(np.int64)
    if result < int64.min or result > int64.max:
        raise ValueError(f"{name} lies outside the supported int64 range")
    return result


def _finite_float(value: object, *, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _integer_array(values: object, *, name: str) -> np.ndarray:
    """Return exact int64 values without silent truncation or uint wrap."""
    raw = np.asarray(values)
    if raw.dtype == np.bool_ or np.iscomplexobj(raw):
        raise ValueError(f"{name} must contain integer values")
    if raw.dtype.kind not in "iuf":
        raise ValueError(f"{name} must contain numeric integer values")

    if np.issubdtype(raw.dtype, np.floating):
        if (
            not np.isfinite(raw).all()
            or not np.equal(raw, np.round(raw)).all()
        ):
            raise ValueError(f"{name} must contain finite integer values")

    if raw.size:
        int64 = np.iinfo(np.int64)
        if np.issubdtype(raw.dtype, np.unsignedinteger):
            if np.any(raw > int64.max):
                raise ValueError(f"{name} contains values outside int64 range")
        else:
            if np.any(raw < int64.min) or np.any(raw > int64.max):
                raise ValueError(f"{name} contains values outside int64 range")

    return raw.astype(np.int64, copy=False)


def _class_ids(
    values: Sequence[int],
    *,
    total_classes: Optional[int] = None,
) -> list[int]:
    ids = [_as_int(value, "class_id") for value in values]
    if not ids:
        raise ValueError("class_ids must not be empty")
    if len(ids) != len(set(ids)) or any(value < 0 for value in ids):
        raise ValueError("class_ids must contain unique non-negative IDs")

    if total_classes is not None:
        total = _as_int(total_classes, "total_classes")
        if total <= 0 or max(ids) >= total:
            raise ValueError("total_classes is inconsistent with class_ids")
    return ids


def _class_name(
    class_id: int,
    target_names: Optional[Sequence[str]],
) -> str:
    names = list(target_names or [])
    return (
        str(names[class_id])
        if class_id < len(names)
        else f"Class {class_id + 1}"
    )


def _output_path(path: str) -> str:
    destination = os.path.abspath(path)
    os.makedirs(os.path.dirname(destination) or ".", exist_ok=True)
    return destination


# -----------------------------------------------------------------------------
# Consistent map palette
# -----------------------------------------------------------------------------


def _palette(
    total_classes: int,
    cmap_name: str,
) -> tuple[ListedColormap, BoundaryNorm]:
    total = _as_int(total_classes, "total_classes")
    if total <= 0:
        raise ValueError("total_classes must be positive")

    try:
        source = plt.get_cmap(str(cmap_name))
    except ValueError as exc:
        raise ValueError(f"unknown matplotlib colormap {cmap_name!r}") from exc

    positions = (np.arange(total, dtype=np.float64) + 0.5) / float(total)
    class_colours = source(positions)
    background = np.asarray([[0.0, 0.0, 0.0, 1.0]], dtype=np.float64)
    cmap = ListedColormap(np.vstack((background, class_colours)))
    boundaries = np.arange(-0.5, total + 1.5, 1.0, dtype=np.float64)
    norm = BoundaryNorm(boundaries, ncolors=cmap.N, clip=True)
    return cmap, norm


# -----------------------------------------------------------------------------
# Qualitative maps
# -----------------------------------------------------------------------------


def _validated_labels(
    values: np.ndarray,
    *,
    name: str,
    class_ids: Sequence[int],
    total_classes: int,
) -> np.ndarray:
    result = _integer_array(values, name=name).reshape(-1)
    total = _as_int(total_classes, "total_classes")

    if total <= 0:
        raise ValueError("total_classes must be positive")
    if result.size == 0:
        raise ValueError(f"{name} must not be empty")
    if np.any(result < 0) or np.any(result >= total):
        raise ValueError(f"{name} contains IDs outside the dataset label range")

    allowed = set(int(value) for value in class_ids)
    outside = sorted(set(result.tolist()) - allowed)
    if outside:
        raise ValueError(f"{name} contains phase-invisible classes: {outside}")
    return result


def _validated_coordinates(
    coords: np.ndarray,
    *,
    count: int,
    gt_shape: Sequence[int],
) -> tuple[np.ndarray, tuple[int, int]]:
    result = _integer_array(coords, name="coords")
    expected_count = _as_int(count, "count")

    if expected_count < 0:
        raise ValueError("count must be non-negative")
    if result.shape != (expected_count, 2):
        raise ValueError(f"coords must have shape {(expected_count, 2)}")
    if np.unique(result, axis=0).shape[0] != result.shape[0]:
        raise ValueError("coords contain duplicate image locations")

    if len(gt_shape) != 2:
        raise ValueError("gt_shape must contain [height,width]")
    shape = (
        _as_int(gt_shape[0], "gt_shape[0]"),
        _as_int(gt_shape[1], "gt_shape[1]"),
    )
    if min(shape) <= 0:
        raise ValueError("gt_shape must be positive")

    rows = result[:, 0]
    cols = result[:, 1]
    if (
        np.any(rows < 0)
        or np.any(rows >= shape[0])
        or np.any(cols < 0)
        or np.any(cols >= shape[1])
    ):
        raise ValueError("coords contain positions outside gt_shape")
    return result, shape


def _render_map(
    path: str,
    values: np.ndarray,
    *,
    total_classes: int,
    cmap_name: str,
    dpi: int,
) -> str:
    destination = _output_path(path)

    array = np.asarray(values, dtype=np.int32)
    if array.ndim != 2:
        raise ValueError("qualitative map must be two-dimensional")
    if np.any(array < 0) or np.any(array > int(total_classes)):
        raise ValueError("qualitative map contains invalid rendered IDs")

    dpi_value = _as_int(dpi, "dpi")
    if dpi_value <= 0:
        raise ValueError("dpi must be positive")

    cmap, norm = _palette(total_classes, cmap_name)
    height, width = array.shape
    long_side = 7.2
    aspect = float(width) / float(height)
    figure_size = (
        (long_side, long_side / aspect)
        if aspect >= 1.0
        else (long_side * aspect, long_side)
    )

    figure = plt.figure(figsize=figure_size, dpi=dpi_value, frameon=False)
    axis = figure.add_axes([0.0, 0.0, 1.0, 1.0])
    axis.imshow(array, cmap=cmap, norm=norm, interpolation="nearest")
    axis.set_axis_off()

    figure.savefig(
        destination,
        dpi=dpi_value,
        bbox_inches=None,
        transparent=False,
        pad_inches=0.0,
    )
    plt.close(figure)
    return destination


def _render_legend(
    path: str,
    *,
    class_ids: Sequence[int],
    target_names: Optional[Sequence[str]],
    total_classes: int,
    cmap_name: str,
    dpi: int,
) -> str:
    destination = _output_path(path)
    cmap, norm = _palette(total_classes, cmap_name)

    handles = [
        Patch(
            facecolor=cmap(norm(0)),
            edgecolor="none",
            label="Background / unseen",
        )
    ]
    handles.extend(
        Patch(
            facecolor=cmap(norm(class_id + 1)),
            edgecolor="none",
            label=f"{class_id + 1}. {_class_name(class_id, target_names)}",
        )
        for class_id in class_ids
    )

    columns = 1 if len(handles) <= 8 else 2
    rows = int(math.ceil(len(handles) / columns))
    figure = plt.figure(
        figsize=(4.6 * columns, max(1.2, 0.42 * rows)),
        dpi=int(dpi),
    )
    axis = figure.add_axes([0.0, 0.0, 1.0, 1.0])
    axis.set_axis_off()
    axis.legend(
        handles=handles,
        loc="center",
        ncol=columns,
        frameon=False,
        fontsize=10,
    )

    figure.savefig(
        destination,
        dpi=int(dpi),
        bbox_inches="tight",
        transparent=False,
        pad_inches=0.04,
    )
    plt.close(figure)
    return destination


def save_full_phase_qualitative_maps(
    *,
    output_dir: str,
    phase: int,
    gt_shape: Sequence[int],
    coords: np.ndarray,
    targets: np.ndarray,
    predictions: np.ndarray,
    class_ids: Sequence[int],
    total_classes: int,
    target_names: Optional[Sequence[str]] = None,
    cmap_name: str = "nipy_spectral",
    dpi: int = 300,
) -> Dict[str, str]:
    """Save separate full-scene ground-truth, prediction, and legend images."""
    phase_id = _as_int(phase, "phase")
    if phase_id < 0:
        raise ValueError("phase must be non-negative")

    total = _as_int(total_classes, "total_classes")
    if total <= 0:
        raise ValueError("total_classes must be positive")

    dpi_value = _as_int(dpi, "dpi")
    if dpi_value <= 0:
        raise ValueError("dpi must be positive")

    ids = _class_ids(class_ids, total_classes=total)

    if target_names is not None and len(target_names) != total:
        raise ValueError("target_names must contain one name per dataset class")

    truth = _validated_labels(
        targets,
        name="targets",
        class_ids=ids,
        total_classes=total,
    )
    prediction = _validated_labels(
        predictions,
        name="predictions",
        class_ids=ids,
        total_classes=total,
    )
    if prediction.size != truth.size:
        raise ValueError("targets and predictions must align")

    coordinates, shape = _validated_coordinates(
        coords,
        count=truth.size,
        gt_shape=gt_shape,
    )

    gt_map = np.zeros(shape, dtype=np.int32)
    pred_map = np.zeros(shape, dtype=np.int32)
    rows = coordinates[:, 0]
    cols = coordinates[:, 1]
    gt_map[rows, cols] = truth.astype(np.int32) + 1
    pred_map[rows, cols] = prediction.astype(np.int32) + 1

    root = os.path.abspath(output_dir)
    os.makedirs(root, exist_ok=True)
    prefix = f"phase_{phase_id:02d}"

    return {
        "ground_truth": _render_map(
            os.path.join(root, f"{prefix}_ground_truth.png"),
            gt_map,
            total_classes=total,
            cmap_name=str(cmap_name),
            dpi=dpi_value,
        ),
        "prediction": _render_map(
            os.path.join(root, f"{prefix}_prediction.png"),
            pred_map,
            total_classes=total,
            cmap_name=str(cmap_name),
            dpi=dpi_value,
        ),
        "legend": _render_legend(
            os.path.join(root, f"{prefix}_legend.png"),
            class_ids=ids,
            target_names=target_names,
            total_classes=total,
            cmap_name=str(cmap_name),
            dpi=dpi_value,
        ),
    }


# -----------------------------------------------------------------------------
# Metric helpers
# -----------------------------------------------------------------------------


def _mapping_metric(
    metrics: Mapping[str, Any],
    key: str,
    class_id: int,
) -> float:
    values = metrics.get(key)
    if not isinstance(values, Mapping):
        raise ValueError(f"metrics lack {key}")

    value = values.get(class_id, values.get(str(class_id)))
    if value is None:
        raise ValueError(f"{key} lacks class {class_id}")

    return _finite_float(value, name=f"{key}[{class_id}]")



def _structure_class_value(
    structural_geometry: Mapping[str, Any],
    class_id: int,
    key: str,
) -> float:
    rows = structural_geometry.get("classes")
    if not isinstance(rows, Sequence):
        raise ValueError("structural_geometry lacks classes")

    for row in rows:
        if isinstance(row, Mapping) and int(row.get("class_id", -1)) == class_id:
            return _finite_float(row[key], name=f"{key}[{class_id}]")

    raise ValueError(f"structural geometry lacks class {class_id}")


# -----------------------------------------------------------------------------
# Base-phase classifier / class-structure figures
# -----------------------------------------------------------------------------


def _save_grouped_metric(
    path: str,
    *,
    title: str,
    ylabel: str,
    class_ids: Sequence[int],
    target_names: Optional[Sequence[str]],
    train: Mapping[str, Any],
    validation: Optional[Mapping[str, Any]],
    test: Mapping[str, Any],
    key: str,
    percentage: bool,
    dpi: int,
) -> str:
    ids = [int(v) for v in class_ids]
    specs: list[tuple[str, Mapping[str, Any]]] = [("Train", train)]
    if validation is not None:
        specs.append(("Validation", validation))
    specs.append(("Test", test))

    x = np.arange(len(ids), dtype=np.float64)
    width = 0.78 / float(len(specs))
    figure, axis = plt.subplots(
        figsize=(max(7.4, 0.95 * len(ids)), 4.8),
        dpi=int(dpi),
    )

    for index, (label, metrics) in enumerate(specs):
        values = np.asarray(
            [_mapping_metric(metrics, key, class_id) for class_id in ids],
            dtype=np.float64,
        )
        if percentage:
            values = 100.0 * values

        offset = (index - (len(specs) - 1) / 2.0) * width
        axis.bar(x + offset, values, width=width, label=label)

    axis.set_xticks(x)
    axis.set_xticklabels(
        [
            f"{class_id + 1}\n{_class_name(class_id, target_names)}"
            for class_id in ids
        ],
        rotation=32,
        ha="right",
    )
    axis.set_ylabel(ylabel)
    axis.set_title(title)
    axis.legend(frameon=False)
    axis.grid(axis="y", alpha=0.2)
    figure.tight_layout()

    destination = _output_path(path)
    figure.savefig(destination, dpi=int(dpi), bbox_inches="tight")
    plt.close(figure)
    return destination


def _save_class_structure_summary(
    path: str,
    *,
    structural_geometry: Mapping[str, Any],
    class_ids: Sequence[int],
    target_names: Optional[Sequence[str]],
    dpi: int,
) -> str:
    """Show only the two persistent class-level structural magnitudes: I_c and S_c."""
    ids = [int(v) for v in class_ids]
    identity = np.asarray(
        [
            _structure_class_value(structural_geometry, class_id, "identity_rms")
            for class_id in ids
        ],
        dtype=np.float64,
    )
    structure = np.asarray(
        [
            _structure_class_value(
                structural_geometry,
                class_id,
                "structure_unique_rms",
            )
            for class_id in ids
        ],
        dtype=np.float64,
    )

    x = np.arange(len(ids), dtype=np.float64)
    width = 0.36
    figure, axis = plt.subplots(
        figsize=(max(7.4, 0.95 * len(ids)), 4.8),
        dpi=int(dpi),
    )
    axis.bar(x - width / 2.0, identity, width=width, label=r"Identity $I_c$")
    axis.bar(x + width / 2.0, structure, width=width, label=r"Structure $S_c$")

    axis.set_xticks(x)
    axis.set_xticklabels(
        [
            f"{class_id + 1}\n{_class_name(class_id, target_names)}"
            for class_id in ids
        ],
        rotation=32,
        ha="right",
    )
    axis.set_ylabel("RMS magnitude")
    axis.set_title("Retained class-discriminative statistics")
    axis.legend(frameon=False)
    axis.grid(axis="y", alpha=0.2)
    figure.tight_layout()

    destination = _output_path(path)
    figure.savefig(destination, dpi=int(dpi), bbox_inches="tight")
    plt.close(figure)
    return destination


def _pair_matrix(
    structural_geometry: Mapping[str, Any],
    class_ids: Sequence[int],
    key: str,
) -> np.ndarray:
    ids = [int(v) for v in class_ids]
    index = {class_id: i for i, class_id in enumerate(ids)}
    matrix = np.zeros((len(ids), len(ids)), dtype=np.float64)

    pairs = structural_geometry.get("pairs")
    if not isinstance(pairs, Sequence):
        raise ValueError("structural_geometry lacks pairs")

    for row in pairs:
        if not isinstance(row, Mapping):
            continue

        left = int(row["left_class_id"])
        right = int(row["right_class_id"])
        if left not in index or right not in index:
            continue

        value = _finite_float(row[key], name=f"{key}[{left},{right}]")
        i, j = index[left], index[right]
        matrix[i, j] = value
        matrix[j, i] = value

    return matrix


def _save_pair_matrix(
    path: str,
    *,
    structural_geometry: Mapping[str, Any],
    class_ids: Sequence[int],
    target_names: Optional[Sequence[str]],
    key: str,
    title: str,
    cbar_label: str,
    dpi: int,
) -> str:
    ids = [int(v) for v in class_ids]
    matrix = _pair_matrix(structural_geometry, ids, key)

    figure, axis = plt.subplots(
        figsize=(max(5.8, 0.48 * len(ids) + 3.0), max(5.2, 0.46 * len(ids) + 2.6)),
        dpi=int(dpi),
    )
    image = axis.imshow(matrix, interpolation="nearest")

    ticks = np.arange(len(ids))
    labels = [f"{class_id + 1}" for class_id in ids]
    axis.set_xticks(ticks)
    axis.set_yticks(ticks)
    axis.set_xticklabels(labels)
    axis.set_yticklabels(labels)
    axis.set_xlabel("Class")
    axis.set_ylabel("Class")
    axis.set_title(title)

    colorbar = figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    colorbar.set_label(cbar_label)

    # For small class sets, values improve interpretability. For large class
    # sets the annotations become unreadable, so they are omitted.
    if len(ids) <= 8:
        for i in range(len(ids)):
            for j in range(len(ids)):
                if i == j:
                    continue
                value = matrix[i, j]
                axis.text(
                    j,
                    i,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    fontsize=7,
                )

    figure.tight_layout()
    destination = _output_path(path)
    figure.savefig(destination, dpi=int(dpi), bbox_inches="tight")
    plt.close(figure)
    return destination


def save_geometry_diagnostic_figures(
    *,
    output_dir: str,
    train: Mapping[str, Any],
    validation: Optional[Mapping[str, Any]],
    test: Mapping[str, Any],
    structural_geometry: Mapping[str, Any],
    class_ids: Sequence[int],
    target_names: Optional[Sequence[str]] = None,
    dpi: int = 300,
) -> Dict[str, str]:
    """Save the minimal phase-0 figures needed by the current architecture.

    Kept:
    - class accuracy;
    - angular decision margin;
    - I_c/S_c magnitude summary;
    - inter-class identity difference;
    - inter-class structure difference.

    Removed from the default figure set:
    - duplicate A_c/diag(S_c) plots;
    - duplicate C_c/mean_D(S_c) plots;
    - separate feature-variation and cross-region magnitude plots.

    Those quantities remain available in geometry_report.json when needed.
    """
    root = os.path.abspath(output_dir)
    os.makedirs(root, exist_ok=True)

    dpi_value = _as_int(dpi, "dpi")
    if dpi_value <= 0:
        raise ValueError("dpi must be positive")

    return {
        "class_accuracy": _save_grouped_metric(
            os.path.join(root, "classifier_class_accuracy.png"),
            title="Per-class classification accuracy",
            ylabel="Accuracy (%)",
            class_ids=class_ids,
            target_names=target_names,
            train=train,
            validation=validation,
            test=test,
            key="per_class_accuracy",
            percentage=True,
            dpi=dpi_value,
        ),
        "class_cosine_margin": _save_grouped_metric(
            os.path.join(root, "classifier_class_cosine_margin.png"),
            title="Target-to-nearest-rival angular margin",
            ylabel="Cosine margin",
            class_ids=class_ids,
            target_names=target_names,
            train=train,
            validation=validation,
            test=test,
            key="per_class_mean_cosine_margin",
            percentage=False,
            dpi=dpi_value,
        ),
        "class_structure": _save_class_structure_summary(
            os.path.join(root, "class_structure_summary.png"),
            structural_geometry=structural_geometry,
            class_ids=class_ids,
            target_names=target_names,
            dpi=dpi_value,
        ),
        "inter_class_identity_difference": _save_pair_matrix(
            os.path.join(root, "inter_class_identity_difference.png"),
            structural_geometry=structural_geometry,
            class_ids=class_ids,
            target_names=target_names,
            key="identity_difference_rms",
            title=r"Inter-class identity difference",
            cbar_label=r"RMS$(I_c-I_d)$",
            dpi=dpi_value,
        ),
        "inter_class_structure_difference": _save_pair_matrix(
            os.path.join(root, "inter_class_structure_difference.png"),
            structural_geometry=structural_geometry,
            class_ids=class_ids,
            target_names=target_names,
            key="structure_difference_rms",
            title=r"Inter-class structural difference",
            cbar_label=r"RMS$(S_c-S_d)$",
            dpi=dpi_value,
        ),
    }


# -----------------------------------------------------------------------------
# Incremental classifier / preservation figures
# -----------------------------------------------------------------------------


def _save_incremental_class_accuracy(
    path: str,
    *,
    cumulative_test: Mapping[str, Any],
    old_class_ids: Sequence[int],
    new_class_ids: Sequence[int],
    target_names: Optional[Sequence[str]],
    dpi: int,
) -> str:
    old_ids = [int(v) for v in old_class_ids]
    new_ids = [int(v) for v in new_class_ids]
    seen_ids = old_ids + new_ids

    values = np.asarray(
        [
            100.0
            * _mapping_metric(
                cumulative_test,
                "per_class_accuracy",
                class_id,
            )
            for class_id in seen_ids
        ],
        dtype=np.float64,
    )

    x = np.arange(len(seen_ids), dtype=np.float64)
    figure, axis = plt.subplots(
        figsize=(max(8.0, 0.75 * len(seen_ids)), 4.8),
        dpi=int(dpi),
    )

    old_count = len(old_ids)
    axis.bar(x[:old_count], values[:old_count], label="Old classes")
    axis.bar(x[old_count:], values[old_count:], label="New classes")

    axis.set_xticks(x)
    axis.set_xticklabels(
        [
            f"{class_id + 1}\n{_class_name(class_id, target_names)}"
            for class_id in seen_ids
        ],
        rotation=35,
        ha="right",
    )
    axis.set_ylim(0.0, 100.0)
    axis.set_ylabel("Test accuracy (%)")
    axis.set_title("Cumulative per-class accuracy after incremental learning")
    axis.axvline(old_count - 0.5, linewidth=1.0, linestyle="--", alpha=0.6)
    axis.legend(frameon=False)
    axis.grid(axis="y", alpha=0.2)
    figure.tight_layout()

    destination = _output_path(path)
    figure.savefig(destination, dpi=int(dpi), bbox_inches="tight")
    plt.close(figure)
    return destination


def _save_stability_plasticity(
    path: str,
    *,
    old_before: Mapping[str, Any],
    old_new: Mapping[str, Any],
    forgetting: Mapping[str, Any],
    dpi: int,
) -> str:
    old_before_ba = 100.0 * _finite_float(
        old_before["balanced_accuracy"],
        name="old_before.balanced_accuracy",
    )
    old_after_ba = 100.0 * _finite_float(
        old_new["old"]["balanced_accuracy"],
        name="old_new.old.balanced_accuracy",
    )
    new_ba = 100.0 * _finite_float(
        old_new["new"]["balanced_accuracy"],
        name="old_new.new.balanced_accuracy",
    )
    harmonic = 100.0 * _finite_float(
        old_new["harmonic_old_new_accuracy"],
        name="old_new.harmonic_old_new_accuracy",
    )
    forgetting_pp = 100.0 * _finite_float(
        forgetting["old_balanced_accuracy_drop"],
        name="forgetting.old_balanced_accuracy_drop",
    )

    labels = ["Old before", "Old after", "New", "H"]
    values = [old_before_ba, old_after_ba, new_ba, harmonic]

    figure, axis = plt.subplots(figsize=(6.8, 4.8), dpi=int(dpi))
    bars = axis.bar(np.arange(len(values)), values)

    axis.set_xticks(np.arange(len(values)))
    axis.set_xticklabels(labels)
    axis.set_ylim(0.0, 100.0)
    axis.set_ylabel("Balanced accuracy (%)")
    axis.set_title("Incremental stability-plasticity")
    axis.grid(axis="y", alpha=0.2)

    for bar, value in zip(bars, values):
        axis.text(
            bar.get_x() + bar.get_width() / 2.0,
            value + 1.0,
            f"{value:.1f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    axis.text(
        0.99,
        0.03,
        f"Old-class forgetting: {forgetting_pp:.2f} pp",
        transform=axis.transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
    )

    figure.tight_layout()
    destination = _output_path(path)
    figure.savefig(destination, dpi=int(dpi), bbox_inches="tight")
    plt.close(figure)
    return destination


def _save_preservation_spectrum(
    path: str,
    *,
    preservation: Mapping[str, Any],
    dpi: int,
) -> str:
    """Plot the generalized relevance of the compact identity modes.

    The final protected representation rank can be larger because it also
    includes class-specific (S_c - mean S) responses of those identity modes.
    """
    values = np.asarray(
        preservation.get("generalized_eigenvalues", ()),
        dtype=np.float64,
    ).reshape(-1)
    if values.size == 0:
        raise ValueError("preservation lacks generalized_eigenvalues")
    if not np.isfinite(values).all():
        raise ValueError(
            "preservation generalized_eigenvalues contain NaN/Inf"
        )

    identity_rank = _as_int(
        preservation.get(
            "identity_discriminative_rank",
            values.size,
        ),
        "identity_discriminative_rank",
    )
    representation_rank = _as_int(
        preservation["representation_rank"],
        "representation_rank",
    )
    feature_dim = _as_int(
        preservation["feature_dim"],
        "feature_dim",
    )
    if (
        identity_rank <= 0
        or representation_rank < identity_rank
        or feature_dim < representation_rank
    ):
        raise ValueError(
            "preservation ranks/dimensions are inconsistent"
        )
    if values.size != identity_rank:
        raise ValueError(
            "generalized_eigenvalues must match identity_discriminative_rank"
        )

    x = np.arange(1, values.size + 1)
    figure, axis = plt.subplots(
        figsize=(6.8, 4.6),
        dpi=int(dpi),
    )
    axis.plot(x, values, marker="o")
    axis.set_xticks(x)
    axis.set_xlabel("Identity-discriminative mode")
    axis.set_ylabel(r"Generalized relevance $\rho$")
    axis.set_title("Historical discriminative-mode spectrum")
    axis.grid(alpha=0.2)

    axis.text(
        0.99,
        0.97,
        (
            f"Identity modes: {identity_rank}\n"
            f"Protected structural rank: "
            f"{representation_rank}/{feature_dim}"
        ),
        transform=axis.transAxes,
        ha="right",
        va="top",
        fontsize=9,
    )

    figure.tight_layout()
    destination = _output_path(path)
    figure.savefig(
        destination,
        dpi=int(dpi),
        bbox_inches="tight",
    )
    plt.close(figure)
    return destination


def _save_structure_drift(
    path: str,
    *,
    structure_drift: Mapping[str, Any],
    target_names: Optional[Sequence[str]],
    dpi: int,
) -> str:
    """Plot realized old-class I/S drift on aligned held-out samples."""
    rows = structure_drift.get("classes")
    if not isinstance(rows, Sequence) or not rows:
        raise ValueError("structure_drift lacks per-class rows")

    class_ids: list[int] = []
    identity: list[float] = []
    structure: list[float] = []
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("structure_drift class row must be a mapping")
        class_id = _as_int(row["class_id"], "class_id")
        class_ids.append(class_id)
        identity.append(
            _finite_float(
                row["identity_drift_rms"],
                name=f"identity_drift_rms[{class_id}]",
            )
        )
        structure.append(
            _finite_float(
                row["structure_drift_rms"],
                name=f"structure_drift_rms[{class_id}]",
            )
        )

    x = np.arange(len(class_ids), dtype=np.float64)
    width = 0.36
    figure, axis = plt.subplots(
        figsize=(max(7.4, 0.9 * len(class_ids)), 4.8),
        dpi=int(dpi),
    )
    axis.bar(
        x - width / 2.0,
        identity,
        width=width,
        label=r"Identity drift $\Delta I$",
    )
    axis.bar(
        x + width / 2.0,
        structure,
        width=width,
        label=r"Structure drift $\Delta S$",
    )
    axis.set_xticks(x)
    axis.set_xticklabels(
        [
            f"{class_id + 1}\n{_class_name(class_id, target_names)}"
            for class_id in class_ids
        ],
        rotation=32,
        ha="right",
    )
    axis.set_ylabel("RMS drift")
    axis.set_title("Realized old-class representation drift")
    axis.legend(frameon=False)
    axis.grid(axis="y", alpha=0.2)
    figure.tight_layout()

    destination = _output_path(path)
    figure.savefig(
        destination,
        dpi=int(dpi),
        bbox_inches="tight",
    )
    plt.close(figure)
    return destination

def save_incremental_diagnostic_figures(
    *,
    output_dir: str,
    cumulative_test: Mapping[str, Any],
    old_before: Mapping[str, Any],
    old_new: Mapping[str, Any],
    forgetting: Mapping[str, Any],
    preservation: Mapping[str, Any],
    old_class_ids: Sequence[int],
    new_class_ids: Sequence[int],
    target_names: Optional[Sequence[str]] = None,
    structure_drift: Optional[Mapping[str, Any]] = None,
    dpi: int = 300,
) -> Dict[str, str]:
    """Save the figures that directly answer the incremental research questions.

    1. Which old/new classes remain discriminative?
    2. What is the stability-plasticity balance?
    3. How large is the protected class-discriminative subspace?

    This function intentionally does not plot training-only support/replay,
    pairwise-boundary, adapter, KD, or preservation-loss quantities because
    they are not part of the current architecture.
    """
    root = os.path.abspath(output_dir)
    os.makedirs(root, exist_ok=True)

    dpi_value = _as_int(dpi, "dpi")
    if dpi_value <= 0:
        raise ValueError("dpi must be positive")

    outputs = {
        "class_accuracy": _save_incremental_class_accuracy(
            os.path.join(root, "incremental_class_accuracy.png"),
            cumulative_test=cumulative_test,
            old_class_ids=old_class_ids,
            new_class_ids=new_class_ids,
            target_names=target_names,
            dpi=dpi_value,
        ),
        "stability_plasticity": _save_stability_plasticity(
            os.path.join(root, "stability_plasticity.png"),
            old_before=old_before,
            old_new=old_new,
            forgetting=forgetting,
            dpi=dpi_value,
        ),
        "preservation_spectrum": _save_preservation_spectrum(
            os.path.join(root, "structure_preservation_spectrum.png"),
            preservation=preservation,
            dpi=dpi_value,
        ),
    }
    if structure_drift is not None:
        outputs["structure_drift"] = _save_structure_drift(
            os.path.join(root, "old_class_structure_drift.png"),
            structure_drift=structure_drift,
            target_names=target_names,
            dpi=dpi_value,
        )
    return outputs


# -----------------------------------------------------------------------------
# Joint base-vs-incremental t-SNE
# -----------------------------------------------------------------------------


def _feature_matrix(
    features: np.ndarray,
    *,
    name: str,
) -> np.ndarray:
    value = np.asarray(features, dtype=np.float64)
    if value.ndim < 2 or value.shape[0] == 0:
        raise ValueError(f"{name} must contain at least one sample")
    value = value.reshape(value.shape[0], -1)
    if not np.isfinite(value).all():
        raise ValueError(f"{name} contains NaN/Inf")
    return value


def _feature_labels(
    labels: np.ndarray,
    *,
    count: int,
    name: str,
) -> np.ndarray:
    value = np.asarray(labels)
    if value.ndim != 1 or value.shape[0] != count:
        raise ValueError(f"{name} must be one-dimensional and aligned with features")

    if not np.issubdtype(value.dtype, np.integer):
        if not np.isfinite(value).all() or not np.equal(value, np.round(value)).all():
            raise ValueError(f"{name} must contain finite integer class IDs")

    return value.astype(np.int64, copy=False)


def _ellipse_parameters(
    points: np.ndarray,
    *,
    n_std: float = 1.5,
) -> Optional[tuple[np.ndarray, float, float, float]]:
    """Return a reporting-only 2-D covariance ellipse for embedded points.

    This ellipse is a visual summary in t-SNE space. It is NOT the stored S_c
    statistic and must not be reported as a quantitative S_c estimate.
    """
    if points.shape[0] < 3:
        return None

    center = points.mean(axis=0)
    covariance = np.cov(points.T)
    if covariance.shape != (2, 2) or not np.isfinite(covariance).all():
        return None

    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.maximum(eigenvalues[order], 0.0)
    eigenvectors = eigenvectors[:, order]

    width, height = 2.0 * float(n_std) * np.sqrt(eigenvalues)
    angle = math.degrees(
        math.atan2(eigenvectors[1, 0], eigenvectors[0, 0])
    )
    return center, float(width), float(height), float(angle)


def _draw_embedding_panel(
    axis: Any,
    embedding: np.ndarray,
    labels: np.ndarray,
    *,
    class_ids: Sequence[int],
    new_class_ids: Sequence[int],
    class_colours: Mapping[int, Any],
    target_names: Optional[Sequence[str]],
    title: str,
) -> None:
    new_set = set(int(v) for v in new_class_ids)

    for class_id in [int(v) for v in class_ids]:
        mask = labels == class_id
        if not np.any(mask):
            continue

        points = embedding[mask]
        marker = "^" if class_id in new_set else "o"
        colour = class_colours[class_id]

        axis.scatter(
            points[:, 0],
            points[:, 1],
            s=16,
            marker=marker,
            alpha=0.72,
            linewidths=0.5 if class_id in new_set else 0.0,
            edgecolors="black" if class_id in new_set else "none",
            c=[colour],
            label=_class_name(class_id, target_names),
        )

        ellipse = _ellipse_parameters(points)
        if ellipse is not None:
            center, width, height, angle = ellipse
            axis.add_patch(
                Ellipse(
                    xy=center,
                    width=width,
                    height=height,
                    angle=angle,
                    facecolor="none",
                    edgecolor=colour,
                    linewidth=1.0,
                    alpha=0.9,
                )
            )
            axis.scatter(
                [center[0]],
                [center[1]],
                marker="x",
                s=40,
                linewidths=1.1,
                c=[colour],
                zorder=5,
            )

    axis.set_title(title)
    axis.set_xticks([])
    axis.set_yticks([])


def save_incremental_tsne_comparison(
    path: str,
    *,
    base_features: np.ndarray,
    base_labels: np.ndarray,
    current_features: np.ndarray,
    current_labels: np.ndarray,
    old_class_ids: Sequence[int],
    new_class_ids: Sequence[int],
    target_names: Optional[Sequence[str]] = None,
    total_classes: Optional[int] = None,
    cmap_name: str = "nipy_spectral",
    seed: int = 42,
    perplexity: float = 30.0,
    dpi: int = 300,
) -> str:
    """Save a shared-coordinate base-vs-incremental representation figure.

    Required protocol
    -----------------
    - `base_features`: features from the finalized previous/base checkpoint.
    - `current_features`: features from the finalized current checkpoint.
    - the same held-out old-class samples should be present in both when
      interpreting old-class movement;
    - current_features may additionally contain the current phase's new classes.

    The function fits ONE t-SNE model to the concatenated feature matrix and
    then splits the resulting coordinates into two panels. This makes the
    phase comparison visually meaningful.

    The input feature arrays are reporting-time values only and must not be
    saved into checkpoints or used by optimization.
    """
    try:
        from sklearn.manifold import TSNE
    except ImportError as exc:
        raise RuntimeError(
            "save_incremental_tsne_comparison requires scikit-learn"
        ) from exc

    x_base = _feature_matrix(base_features, name="base_features")
    x_current = _feature_matrix(current_features, name="current_features")

    if x_base.shape[1] != x_current.shape[1]:
        raise ValueError("base/current feature dimensions must match")

    y_base = _feature_labels(
        base_labels,
        count=x_base.shape[0],
        name="base_labels",
    )
    y_current = _feature_labels(
        current_labels,
        count=x_current.shape[0],
        name="current_labels",
    )

    old_ids = _class_ids(old_class_ids)
    new_ids = _class_ids(new_class_ids) if new_class_ids else []

    if set(old_ids).intersection(new_ids):
        raise ValueError("old_class_ids and new_class_ids must be disjoint")

    if not set(np.unique(y_base)).issubset(set(old_ids)):
        raise ValueError("base_labels must contain old classes only")

    allowed_current = set(old_ids).union(new_ids)
    if not set(np.unique(y_current)).issubset(allowed_current):
        raise ValueError("current_labels contain classes outside old+new IDs")

    x_joint = np.concatenate([x_base, x_current], axis=0)
    sample_count = int(x_joint.shape[0])
    if sample_count < 4:
        raise ValueError("t-SNE requires at least four total samples")

    requested_perplexity = float(perplexity)
    if not math.isfinite(requested_perplexity) or requested_perplexity <= 0.0:
        raise ValueError("perplexity must be finite and positive")

    actual_perplexity = min(
        requested_perplexity,
        max(2.0, (sample_count - 1) / 3.0),
    )
    if actual_perplexity >= sample_count:
        actual_perplexity = float(sample_count - 1)

    embedding = TSNE(
        n_components=2,
        init="pca",
        learning_rate="auto",
        perplexity=actual_perplexity,
        random_state=int(seed),
    ).fit_transform(x_joint)

    base_embedding = embedding[: x_base.shape[0]]
    current_embedding = embedding[x_base.shape[0] :]

    all_ids = sorted(allowed_current)
    resolved_total = (
        int(total_classes)
        if total_classes is not None
        else (
            len(target_names)
            if target_names is not None
            else max(all_ids) + 1
        )
    )
    if resolved_total <= max(all_ids):
        raise ValueError(
            "total_classes is inconsistent with t-SNE class IDs"
        )
    class_cmap, class_norm = _palette(
        resolved_total,
        str(cmap_name),
    )
    class_colours = {
        class_id: class_cmap(class_norm(class_id + 1))
        for class_id in all_ids
    }

    figure, axes = plt.subplots(
        1,
        2,
        figsize=(11.0, 4.8),
        dpi=int(dpi),
        constrained_layout=True,
    )

    _draw_embedding_panel(
        axes[0],
        base_embedding,
        y_base,
        class_ids=old_ids,
        new_class_ids=(),
        class_colours=class_colours,
        target_names=target_names,
        title="(a) Previous finalized phase",
    )
    _draw_embedding_panel(
        axes[1],
        current_embedding,
        y_current,
        class_ids=all_ids,
        new_class_ids=new_ids,
        class_colours=class_colours,
        target_names=target_names,
        title="(b) Current finalized phase",
    )

    handles, labels = axes[1].get_legend_handles_labels()
    unique: Dict[str, Any] = {}
    for handle, label in zip(handles, labels):
        unique.setdefault(label, handle)

    figure.legend(
        unique.values(),
        unique.keys(),
        loc="lower center",
        ncol=min(6, max(1, len(unique))),
        frameon=False,
        bbox_to_anchor=(0.5, -0.02),
        fontsize=8,
    )

    destination = _output_path(path)
    figure.savefig(
        destination,
        dpi=int(dpi),
        bbox_inches="tight",
        pad_inches=0.04,
    )
    plt.close(figure)
    return destination


__all__ = [
    "save_full_phase_qualitative_maps",
    "save_geometry_diagnostic_figures",
    "save_incremental_diagnostic_figures",
    "save_incremental_tsne_comparison",
]





# from __future__ import annotations

# """Reporting-only visualization for clean role-separated NECIL-HSI.

# Internal labels are zero-based global class IDs. Rendered value 0 is reserved
# for background/unseen pixels; semantic class c is rendered as c+1.

# Prediction figures visualize the persistent cosine classifier. Geometry figures
# visualize only stored historical statistics {I_c,S_c,N_c} and derived reports.
# No geometry statistic is treated as a classifier score.
# """

# import math
# import os
# from numbers import Integral, Real
# from typing import Any, Dict, Mapping, Optional, Sequence

# import matplotlib
# matplotlib.use("Agg")

# import matplotlib.pyplot as plt
# import numpy as np
# from matplotlib.colors import BoundaryNorm, ListedColormap
# from matplotlib.patches import Patch



# def _as_int(value: object, name: str) -> int:
#     """Convert an exact integer scalar without truncation or int64 wraparound."""
#     if isinstance(value, (np.bool_, bool)):
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

#     int64 = np.iinfo(np.int64)
#     if result < int64.min or result > int64.max:
#         raise ValueError(f"{name} lies outside the supported int64 range")
#     return result


# def _integer_array(values: object, *, name: str) -> np.ndarray:
#     """Return exact int64 values without silent float truncation or uint wrap."""
#     raw = np.asarray(values)
#     if raw.dtype == np.bool_ or np.iscomplexobj(raw):
#         raise ValueError(f"{name} must contain integer values")
#     if raw.dtype.kind not in "iuf":
#         raise ValueError(f"{name} must contain numeric integer values")

#     if np.issubdtype(raw.dtype, np.floating):
#         if (
#             not np.isfinite(raw).all()
#             or not np.equal(raw, np.round(raw)).all()
#         ):
#             raise ValueError(f"{name} must contain finite integer values")

#     if raw.size:
#         int64 = np.iinfo(np.int64)
#         if np.issubdtype(raw.dtype, np.unsignedinteger):
#             if np.any(raw > int64.max):
#                 raise ValueError(f"{name} contains values outside int64 range")
#         else:
#             if np.any(raw < int64.min) or np.any(raw > int64.max):
#                 raise ValueError(f"{name} contains values outside int64 range")

#     return raw.astype(np.int64, copy=False)


# def _class_ids(
#     values: Sequence[int],
#     *,
#     total_classes: Optional[int] = None,
# ) -> list[int]:
#     ids = [_as_int(value, "class_id") for value in values]
#     if not ids:
#         raise ValueError("class_ids must not be empty")
#     if len(ids) != len(set(ids)) or any(value < 0 for value in ids):
#         raise ValueError(
#             "class_ids must contain unique non-negative IDs"
#         )
#     if total_classes is not None:
#         total = _as_int(total_classes, "total_classes")
#         if total <= 0 or max(ids) >= total:
#             raise ValueError(
#                 "total_classes is inconsistent with class_ids"
#             )
#     return ids


# def _class_name(
#     class_id: int,
#     target_names: Optional[Sequence[str]],
# ) -> str:
#     names = list(target_names or [])
#     return (
#         str(names[class_id])
#         if class_id < len(names)
#         else f"Class {class_id + 1}"
#     )


# def _palette(
#     total_classes: int,
#     cmap_name: str,
# ) -> tuple[ListedColormap, BoundaryNorm]:
#     total = _as_int(total_classes, "total_classes")
#     if total <= 0:
#         raise ValueError("total_classes must be positive")

#     try:
#         source = plt.get_cmap(str(cmap_name))
#     except ValueError as exc:
#         raise ValueError(
#             f"unknown matplotlib colormap {cmap_name!r}"
#         ) from exc

#     positions = (
#         np.arange(total, dtype=np.float64) + 0.5
#     ) / float(total)
#     class_colours = source(positions)
#     background = np.asarray(
#         [[0.0, 0.0, 0.0, 1.0]],
#         dtype=np.float64,
#     )
#     cmap = ListedColormap(
#         np.vstack((background, class_colours))
#     )
#     boundaries = np.arange(
#         -0.5,
#         total + 1.5,
#         1.0,
#         dtype=np.float64,
#     )
#     return cmap, BoundaryNorm(
#         boundaries,
#         ncolors=cmap.N,
#         clip=True,
#     )



# def _validated_labels(
#     values: np.ndarray,
#     *,
#     name: str,
#     class_ids: Sequence[int],
#     total_classes: int,
# ) -> np.ndarray:
#     result = _integer_array(values, name=name).reshape(-1)
#     total = _as_int(total_classes, "total_classes")
#     if total <= 0:
#         raise ValueError("total_classes must be positive")
#     if result.size == 0:
#         raise ValueError(f"{name} must not be empty")
#     if np.any(result < 0) or np.any(result >= total):
#         raise ValueError(
#             f"{name} contains IDs outside the dataset label range"
#         )

#     allowed = set(int(value) for value in class_ids)
#     outside = sorted(set(result.tolist()) - allowed)
#     if outside:
#         raise ValueError(
#             f"{name} contains phase-invisible classes: {outside}"
#         )
#     return result


# def _validated_coordinates(
#     coords: np.ndarray,
#     *,
#     count: int,
#     gt_shape: Sequence[int],
# ) -> tuple[np.ndarray, tuple[int, int]]:
#     result = _integer_array(coords, name="coords")
#     expected_count = _as_int(count, "count")
#     if expected_count < 0:
#         raise ValueError("count must be non-negative")
#     if result.shape != (expected_count, 2):
#         raise ValueError(
#             f"coords must have shape {(expected_count, 2)}"
#         )
#     if np.unique(result, axis=0).shape[0] != result.shape[0]:
#         raise ValueError("coords contain duplicate image locations")

#     if len(gt_shape) != 2:
#         raise ValueError("gt_shape must contain [height,width]")
#     shape = (
#         _as_int(gt_shape[0], "gt_shape[0]"),
#         _as_int(gt_shape[1], "gt_shape[1]"),
#     )
#     if min(shape) <= 0:
#         raise ValueError("gt_shape must be positive")

#     rows = result[:, 0]
#     cols = result[:, 1]
#     if (
#         np.any(rows < 0)
#         or np.any(rows >= shape[0])
#         or np.any(cols < 0)
#         or np.any(cols >= shape[1])
#     ):
#         raise ValueError("coords contain positions outside gt_shape")
#     return result, shape

# def _render_map(
#     path: str,
#     values: np.ndarray,
#     *,
#     total_classes: int,
#     cmap_name: str,
#     dpi: int,
# ) -> str:
#     destination = os.path.abspath(path)
#     os.makedirs(
#         os.path.dirname(destination) or ".",
#         exist_ok=True,
#     )

#     array = np.asarray(values, dtype=np.int32)
#     if array.ndim != 2:
#         raise ValueError(
#             "qualitative map must be two-dimensional"
#         )
#     if (
#         np.any(array < 0)
#         or np.any(array > int(total_classes))
#     ):
#         raise ValueError(
#             "qualitative map contains invalid rendered IDs"
#         )
#     if int(dpi) <= 0:
#         raise ValueError("dpi must be positive")

#     cmap, norm = _palette(
#         total_classes,
#         cmap_name,
#     )
#     height, width = array.shape
#     long_side = 7.2
#     aspect = float(width) / float(height)
#     figure_size = (
#         (long_side, long_side / aspect)
#         if aspect >= 1.0
#         else (long_side * aspect, long_side)
#     )

#     figure = plt.figure(
#         figsize=figure_size,
#         dpi=int(dpi),
#         frameon=False,
#     )
#     axis = figure.add_axes(
#         [0.0, 0.0, 1.0, 1.0]
#     )
#     axis.imshow(
#         array,
#         cmap=cmap,
#         norm=norm,
#         interpolation="nearest",
#     )
#     axis.set_axis_off()

#     figure.savefig(
#         destination,
#         dpi=int(dpi),
#         bbox_inches=None,
#         transparent=False,
#         pad_inches=0.0,
#     )
#     plt.close(figure)
#     return destination


# def _render_legend(
#     path: str,
#     *,
#     class_ids: Sequence[int],
#     target_names: Optional[Sequence[str]],
#     total_classes: int,
#     cmap_name: str,
#     dpi: int,
# ) -> str:
#     destination = os.path.abspath(path)
#     os.makedirs(
#         os.path.dirname(destination) or ".",
#         exist_ok=True,
#     )

#     cmap, norm = _palette(
#         total_classes,
#         cmap_name,
#     )
#     handles = [
#         Patch(
#             facecolor=cmap(norm(0)),
#             edgecolor="none",
#             label="Background / unseen",
#         )
#     ]
#     handles.extend(
#         Patch(
#             facecolor=cmap(norm(class_id + 1)),
#             edgecolor="none",
#             label=(
#                 f"{class_id + 1}. "
#                 f"{_class_name(class_id, target_names)}"
#             ),
#         )
#         for class_id in class_ids
#     )

#     columns = 1 if len(handles) <= 8 else 2
#     rows = int(math.ceil(len(handles) / columns))
#     figure = plt.figure(
#         figsize=(
#             4.6 * columns,
#             max(1.2, 0.42 * rows),
#         ),
#         dpi=int(dpi),
#     )
#     axis = figure.add_axes(
#         [0.0, 0.0, 1.0, 1.0]
#     )
#     axis.set_axis_off()
#     axis.legend(
#         handles=handles,
#         loc="center",
#         ncol=columns,
#         frameon=False,
#         fontsize=10,
#     )

#     figure.savefig(
#         destination,
#         dpi=int(dpi),
#         bbox_inches="tight",
#         transparent=False,
#         pad_inches=0.04,
#     )
#     plt.close(figure)
#     return destination


# def save_full_phase_qualitative_maps(
#     *,
#     output_dir: str,
#     phase: int,
#     gt_shape: Sequence[int],
#     coords: np.ndarray,
#     targets: np.ndarray,
#     predictions: np.ndarray,
#     class_ids: Sequence[int],
#     total_classes: int,
#     target_names: Optional[Sequence[str]] = None,
#     cmap_name: str = "nipy_spectral",
#     dpi: int = 300,
# ) -> Dict[str, str]:
#     """Save separate full labeled-scene ground-truth and prediction maps."""
#     phase_id = _as_int(phase, "phase")
#     if phase_id < 0:
#         raise ValueError("phase must be non-negative")
#     total = _as_int(total_classes, "total_classes")
#     if total <= 0:
#         raise ValueError("total_classes must be positive")
#     dpi_value = _as_int(dpi, "dpi")
#     if dpi_value <= 0:
#         raise ValueError("dpi must be positive")

#     ids = _class_ids(
#         class_ids,
#         total_classes=total,
#     )
#     if (
#         target_names is not None
#         and len(target_names) != total
#     ):
#         raise ValueError(
#             "target_names must contain one name per dataset class"
#         )

#     truth = _validated_labels(
#         targets,
#         name="targets",
#         class_ids=ids,
#         total_classes=total,
#     )
#     prediction = _validated_labels(
#         predictions,
#         name="predictions",
#         class_ids=ids,
#         total_classes=total,
#     )
#     if prediction.size != truth.size:
#         raise ValueError(
#             "targets and predictions must align"
#         )

#     coordinates, shape = _validated_coordinates(
#         coords,
#         count=truth.size,
#         gt_shape=gt_shape,
#     )

#     gt_map = np.zeros(
#         shape,
#         dtype=np.int32,
#     )
#     pred_map = np.zeros(
#         shape,
#         dtype=np.int32,
#     )
#     rows = coordinates[:, 0]
#     cols = coordinates[:, 1]
#     gt_map[rows, cols] = (
#         truth.astype(np.int32) + 1
#     )
#     pred_map[rows, cols] = (
#         prediction.astype(np.int32) + 1
#     )

#     root = os.path.abspath(output_dir)
#     os.makedirs(root, exist_ok=True)
#     prefix = f"phase_{phase_id:02d}"

#     return {
#         "ground_truth": _render_map(
#             os.path.join(
#                 root,
#                 f"{prefix}_ground_truth.png",
#             ),
#             gt_map,
#             total_classes=total,
#             cmap_name=str(cmap_name),
#             dpi=dpi_value,
#         ),
#         "prediction": _render_map(
#             os.path.join(
#                 root,
#                 f"{prefix}_prediction.png",
#             ),
#             pred_map,
#             total_classes=total,
#             cmap_name=str(cmap_name),
#             dpi=dpi_value,
#         ),
#         "legend": _render_legend(
#             os.path.join(
#                 root,
#                 f"{prefix}_legend.png",
#             ),
#             class_ids=ids,
#             target_names=target_names,
#             total_classes=total,
#             cmap_name=str(cmap_name),
#             dpi=dpi_value,
#         ),
#     }



# def _mapping_metric(metrics: Mapping[str, Any], key: str, class_id: int) -> float:
#     values = metrics.get(key)
#     if not isinstance(values, Mapping):
#         raise ValueError(f"metrics lack {key}")
#     value = values.get(class_id, values.get(str(class_id)))
#     if value is None:
#         raise ValueError(f"{key} lacks class {class_id}")
#     result = float(value)
#     if not math.isfinite(result):
#         raise ValueError(f"{key} for class {class_id} must be finite")
#     return result


# def _save_grouped_metric(
#     path: str,
#     *,
#     title: str,
#     ylabel: str,
#     class_ids: Sequence[int],
#     target_names: Optional[Sequence[str]],
#     train: Mapping[str, Any],
#     validation: Optional[Mapping[str, Any]],
#     test: Mapping[str, Any],
#     key: str,
#     percentage: bool,
#     dpi: int,
# ) -> str:
#     ids = [int(v) for v in class_ids]
#     specs: list[tuple[str, Mapping[str, Any]]] = [("Train", train)]
#     if validation is not None:
#         specs.append(("Validation", validation))
#     specs.append(("Test", test))

#     x = np.arange(len(ids), dtype=np.float64)
#     width = 0.8 / float(len(specs))
#     fig, ax = plt.subplots(figsize=(max(7.0, 0.9 * len(ids)), 4.8), dpi=int(dpi))
#     for index, (label, metrics) in enumerate(specs):
#         values = np.asarray(
#             [_mapping_metric(metrics, key, class_id) for class_id in ids],
#             dtype=np.float64,
#         )
#         if percentage:
#             values = 100.0 * values
#         offset = (index - (len(specs) - 1) / 2.0) * width
#         ax.bar(x + offset, values, width=width, label=label)
#     ax.set_xticks(x)
#     ax.set_xticklabels(
#         [f"{class_id + 1}\n{_class_name(class_id, target_names)}" for class_id in ids],
#         rotation=35,
#         ha="right",
#     )
#     ax.set_ylabel(ylabel)
#     ax.set_title(title)
#     ax.legend()
#     fig.tight_layout()
#     destination = os.path.abspath(path)
#     os.makedirs(os.path.dirname(destination) or ".", exist_ok=True)
#     fig.savefig(destination, dpi=int(dpi), bbox_inches="tight")
#     plt.close(fig)
#     return destination


# def _structure_class_value(
#     structural_geometry: Mapping[str, Any],
#     class_id: int,
#     key: str,
# ) -> float:
#     rows = structural_geometry.get("classes")
#     if not isinstance(rows, Sequence):
#         raise ValueError("structural_geometry lacks classes")
#     for row in rows:
#         if isinstance(row, Mapping) and int(row.get("class_id", -1)) == class_id:
#             value = float(row[key])
#             if not math.isfinite(value):
#                 raise ValueError(f"non-finite {key} for class {class_id}")
#             return value
#     raise ValueError(f"structural geometry lacks class {class_id}")


# def _save_structure_bar(
#     path: str,
#     *,
#     structural_geometry: Mapping[str, Any],
#     class_ids: Sequence[int],
#     target_names: Optional[Sequence[str]],
#     key: str,
#     title: str,
#     ylabel: str,
#     dpi: int,
# ) -> str:
#     ids = [int(v) for v in class_ids]
#     values = [_structure_class_value(structural_geometry, class_id, key) for class_id in ids]
#     x = np.arange(len(ids), dtype=np.float64)
#     fig, ax = plt.subplots(figsize=(max(7.0, 0.9 * len(ids)), 4.8), dpi=int(dpi))
#     ax.bar(x, values)
#     ax.set_xticks(x)
#     ax.set_xticklabels(
#         [f"{class_id + 1}\n{_class_name(class_id, target_names)}" for class_id in ids],
#         rotation=35,
#         ha="right",
#     )
#     ax.set_ylabel(ylabel)
#     ax.set_title(title)
#     fig.tight_layout()
#     destination = os.path.abspath(path)
#     os.makedirs(os.path.dirname(destination) or ".", exist_ok=True)
#     fig.savefig(destination, dpi=int(dpi), bbox_inches="tight")
#     plt.close(fig)
#     return destination


# def _save_pair_matrix(
#     path: str,
#     *,
#     structural_geometry: Mapping[str, Any],
#     class_ids: Sequence[int],
#     target_names: Optional[Sequence[str]],
#     key: str,
#     title: str,
#     dpi: int,
# ) -> str:
#     ids = [int(v) for v in class_ids]
#     index = {class_id: i for i, class_id in enumerate(ids)}
#     matrix = np.zeros((len(ids), len(ids)), dtype=np.float64)
#     pairs = structural_geometry.get("pairs")
#     if not isinstance(pairs, Sequence):
#         raise ValueError("structural_geometry lacks pairs")
#     for row in pairs:
#         if not isinstance(row, Mapping):
#             continue
#         left = int(row["left_class_id"])
#         right = int(row["right_class_id"])
#         if left not in index or right not in index:
#             continue
#         value = float(row[key])
#         if not math.isfinite(value):
#             raise ValueError(f"pair metric {key} is non-finite")
#         i, j = index[left], index[right]
#         matrix[i, j] = value
#         matrix[j, i] = value

#     fig, ax = plt.subplots(figsize=(6.4, 5.6), dpi=int(dpi))
#     image = ax.imshow(matrix, interpolation="nearest")
#     labels = [f"{class_id + 1}" for class_id in ids]
#     ax.set_xticks(np.arange(len(ids)))
#     ax.set_yticks(np.arange(len(ids)))
#     ax.set_xticklabels(labels)
#     ax.set_yticklabels(labels)
#     ax.set_xlabel("Class")
#     ax.set_ylabel("Class")
#     ax.set_title(title)
#     fig.colorbar(image, ax=ax)
#     fig.tight_layout()
#     destination = os.path.abspath(path)
#     os.makedirs(os.path.dirname(destination) or ".", exist_ok=True)
#     fig.savefig(destination, dpi=int(dpi), bbox_inches="tight")
#     plt.close(fig)
#     return destination


# def save_geometry_diagnostic_figures(
#     *,
#     output_dir: str,
#     train: Mapping[str, Any],
#     validation: Optional[Mapping[str, Any]],
#     test: Mapping[str, Any],
#     structural_geometry: Mapping[str, Any],
#     class_ids: Sequence[int],
#     target_names: Optional[Sequence[str]] = None,
#     dpi: int = 300,
# ) -> Dict[str, str]:
#     """Save classifier-performance and statistical-memory figures separately."""
#     root = os.path.abspath(output_dir)
#     os.makedirs(root, exist_ok=True)
#     dpi_value = _as_int(dpi, "dpi")
#     if dpi_value <= 0:
#         raise ValueError("dpi must be positive")

#     outputs = {
#         "class_accuracy": _save_grouped_metric(
#             os.path.join(root, "classifier_class_accuracy.png"),
#             title="Classifier class accuracy",
#             ylabel="Accuracy (%)",
#             class_ids=class_ids,
#             target_names=target_names,
#             train=train,
#             validation=validation,
#             test=test,
#             key="per_class_accuracy",
#             percentage=True,
#             dpi=dpi_value,
#         ),
#         "class_cosine_margin": _save_grouped_metric(
#             os.path.join(root, "classifier_class_cosine_margin.png"),
#             title="Classifier target-to-nearest-rival cosine margin",
#             ylabel="Cosine margin",
#             class_ids=class_ids,
#             target_names=target_names,
#             train=train,
#             validation=validation,
#             test=test,
#             key="per_class_mean_cosine_margin",
#             percentage=False,
#             dpi=dpi_value,
#         ),
#     }

#     for key, title, ylabel in (
#         ("identity_rms", "Stored class identity magnitude", "RMS(I_c)"),
#         ("structure_unique_rms", "Stored feature-resolved structure magnitude", "RMS(unique S_c)"),
#         ("feature_variation_mean", "Derived within-region feature variation", "mean diag(S_c)"),
#         ("cross_region_offdiagonal_rms", "Derived cross-region covariation magnitude", "off-diagonal RMS(mean_D S_c)"),
#     ):
#         outputs[key] = _save_structure_bar(
#             os.path.join(root, f"geometry_{key}.png"),
#             structural_geometry=structural_geometry,
#             class_ids=class_ids,
#             target_names=target_names,
#             key=key,
#             title=title,
#             ylabel=ylabel,
#             dpi=dpi_value,
#         )

#     outputs["inter_class_identity_difference"] = _save_pair_matrix(
#         os.path.join(root, "geometry_inter_class_identity_difference.png"),
#         structural_geometry=structural_geometry,
#         class_ids=class_ids,
#         target_names=target_names,
#         key="identity_difference_rms",
#         title="Inter-class identity difference RMS",
#         dpi=dpi_value,
#     )
#     outputs["inter_class_structure_difference"] = _save_pair_matrix(
#         os.path.join(root, "geometry_inter_class_structure_difference.png"),
#         structural_geometry=structural_geometry,
#         class_ids=class_ids,
#         target_names=target_names,
#         key="structure_difference_rms",
#         title="Inter-class full-structure difference RMS",
#         dpi=dpi_value,
#     )
#     return outputs


# __all__ = [
#     "save_full_phase_qualitative_maps",
#     "save_geometry_diagnostic_figures",
# ]