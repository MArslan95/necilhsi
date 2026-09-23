from __future__ import annotations
import math
import os
from numbers import Integral, Real
from typing import Dict, Mapping, Optional, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch


def _exact_int(value: object, name: str, *, minimum: int | None = None) -> int:
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
    return result


def _int_array(values: object, *, name: str) -> np.ndarray:
    raw = np.asarray(values)
    if raw.dtype == np.bool_ or raw.dtype.kind not in "iuf" or np.iscomplexobj(raw):
        raise ValueError(f"{name} must contain integer values")
    if np.issubdtype(raw.dtype, np.floating):
        if not np.isfinite(raw).all() or not np.equal(raw, np.round(raw)).all():
            raise ValueError(f"{name} must contain finite integer values")
    return raw.astype(np.int64, copy=False)


def _class_ids(values: Sequence[int], total_classes: int) -> list[int]:
    total = _exact_int(total_classes, "total_classes", minimum=1)
    ids = [_exact_int(v, "class_id", minimum=0) for v in values]
    if not ids or len(ids) != len(set(ids)) or max(ids) >= total:
        raise ValueError("class_ids must be unique visible dataset class IDs")
    return ids


def _palette(total_classes: int, cmap_name: str) -> tuple[ListedColormap, BoundaryNorm]:
    total = _exact_int(total_classes, "total_classes", minimum=1)
    try:
        source = plt.get_cmap(str(cmap_name))
    except ValueError as exc:
        raise ValueError(f"unknown matplotlib colormap {cmap_name!r}") from exc
    positions = (np.arange(total, dtype=np.float64) + 0.5) / float(total)
    background = np.asarray([[0.0, 0.0, 0.0, 1.0]], dtype=np.float64)
    cmap = ListedColormap(np.vstack((background, source(positions))))
    boundaries = np.arange(-0.5, total + 1.5, 1.0, dtype=np.float64)
    return cmap, BoundaryNorm(boundaries, ncolors=cmap.N, clip=True)


def _render_map(path: str, values: np.ndarray, *, total_classes: int, cmap_name: str, dpi: int) -> str:
    array = np.asarray(values, dtype=np.int32)
    if array.ndim != 2 or np.any(array < 0) or np.any(array > int(total_classes)):
        raise ValueError("qualitative map contains invalid rendered IDs")
    destination = os.path.abspath(path)
    os.makedirs(os.path.dirname(destination) or ".", exist_ok=True)
    dpi_value = _exact_int(dpi, "dpi", minimum=1)
    cmap, norm = _palette(total_classes, cmap_name)
    h, w = array.shape
    aspect = float(w) / float(h)
    long_side = 7.2
    size = (long_side, long_side / aspect) if aspect >= 1.0 else (long_side * aspect, long_side)
    figure = plt.figure(figsize=size, dpi=dpi_value, frameon=False)
    axis = figure.add_axes([0.0, 0.0, 1.0, 1.0])
    axis.imshow(array, cmap=cmap, norm=norm, interpolation="nearest")
    axis.set_axis_off()
    figure.savefig(destination, dpi=dpi_value, bbox_inches=None, transparent=False, pad_inches=0.0)
    plt.close(figure)
    return destination


def _render_legend(path: str, *, class_ids: Sequence[int], target_names: Sequence[str], total_classes: int, cmap_name: str, dpi: int) -> str:
    destination = os.path.abspath(path)
    os.makedirs(os.path.dirname(destination) or ".", exist_ok=True)
    cmap, norm = _palette(total_classes, cmap_name)
    handles = [Patch(facecolor=cmap(norm(0)), edgecolor="none", label="Background / unseen")]
    names = list(target_names)
    for class_id in class_ids:
        name = str(names[class_id]) if class_id < len(names) else f"Class {class_id + 1}"
        handles.append(Patch(facecolor=cmap(norm(class_id + 1)), edgecolor="none", label=f"{class_id + 1}. {name}"))
    columns = 1 if len(handles) <= 8 else 2
    rows = int(math.ceil(len(handles) / columns))
    figure = plt.figure(figsize=(4.6 * columns, max(1.2, 0.42 * rows)), dpi=int(dpi))
    axis = figure.add_axes([0.0, 0.0, 1.0, 1.0])
    axis.set_axis_off()
    axis.legend(handles=handles, loc="center", ncol=columns, frameon=False, fontsize=10)
    figure.savefig(destination, dpi=int(dpi), bbox_inches="tight", transparent=False, pad_inches=0.04)
    plt.close(figure)
    return destination


def _render_margin_map(path: str, values: np.ndarray, *, dpi: int) -> str:
    destination = os.path.abspath(path)
    os.makedirs(os.path.dirname(destination) or ".", exist_ok=True)
    array = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(array)
    if array.ndim != 2 or not finite.any():
        raise ValueError("margin map must contain a finite labeled pixel")
    bound = float(np.max(np.abs(array[finite])))
    if not math.isfinite(bound) or bound <= 0.0:
        bound = 1.0
    cmap = plt.get_cmap("coolwarm").copy()
    cmap.set_bad("black")
    figure, axis = plt.subplots(figsize=(7.2, 6.0), dpi=int(dpi))
    image = axis.imshow(np.ma.masked_invalid(array), cmap=cmap, vmin=-bound, vmax=bound, interpolation="nearest")
    axis.set_title("Classifier decision margin")
    axis.set_axis_off()
    colorbar = figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    colorbar.set_label("Target-to-nearest-rival logit margin")
    figure.tight_layout()
    figure.savefig(destination, dpi=int(dpi), bbox_inches="tight")
    plt.close(figure)
    return destination


def _render_error_map(path: str, values: np.ndarray, *, labels: Mapping[int, str], dpi: int) -> str:
    destination = os.path.abspath(path)
    os.makedirs(os.path.dirname(destination) or ".", exist_ok=True)
    array = np.asarray(values, dtype=np.int32)
    if array.ndim != 2:
        raise ValueError("error map must be two-dimensional")
    max_code = max(labels) if labels else 0
    cmap = plt.get_cmap("tab10", max_code + 1)
    figure, axis = plt.subplots(figsize=(7.2, 6.0), dpi=int(dpi))
    axis.imshow(array, cmap=cmap, vmin=-0.5, vmax=max_code + 0.5, interpolation="nearest")
    axis.set_title("Classification error type")
    axis.set_axis_off()
    handles = [Patch(facecolor=cmap(code), edgecolor="none", label=label) for code, label in sorted(labels.items())]
    axis.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False)
    figure.tight_layout()
    figure.savefig(destination, dpi=int(dpi), bbox_inches="tight")
    plt.close(figure)
    return destination


def save_full_phase_qualitative_maps(
    *, output_dir: str, phase: int, gt_shape: Sequence[int], coords: np.ndarray,
    targets: np.ndarray, predictions: np.ndarray, class_ids: Sequence[int],
    total_classes: int, target_names: Optional[Sequence[str]] = None,
    decision_margins: Optional[np.ndarray] = None,
    phase_to_class_ids: Optional[Mapping[int, Sequence[int]]] = None,
    cmap_name: str = "nipy_spectral", dpi: int = 300,
) -> Dict[str, str]:
    phase_id = _exact_int(phase, "phase", minimum=0)
    total = _exact_int(total_classes, "total_classes", minimum=1)
    dpi_value = _exact_int(dpi, "dpi", minimum=1)
    ids = _class_ids(class_ids, total)
    truth = _int_array(targets, name="targets").reshape(-1)
    prediction = _int_array(predictions, name="predictions").reshape(-1)
    coordinates = _int_array(coords, name="coords")
    shape = tuple(_exact_int(v, "gt_shape", minimum=1) for v in gt_shape)
    if len(shape) != 2 or coordinates.shape != (truth.size, 2) or prediction.shape != truth.shape:
        raise ValueError("qualitative inputs are misaligned")
    if np.unique(coordinates, axis=0).shape[0] != coordinates.shape[0]:
        raise ValueError("coords contain duplicate image locations")
    allowed = set(ids)
    if set(truth.tolist()) - allowed or set(prediction.tolist()) - allowed:
        raise ValueError("targets/predictions contain phase-invisible classes")
    if np.any(coordinates[:, 0] < 0) or np.any(coordinates[:, 0] >= shape[0]) or np.any(coordinates[:, 1] < 0) or np.any(coordinates[:, 1] >= shape[1]):
        raise ValueError("coords lie outside gt_shape")

    rows, cols = coordinates[:, 0], coordinates[:, 1]
    gt_map = np.zeros(shape, dtype=np.int32)
    pred_map = np.zeros(shape, dtype=np.int32)
    gt_map[rows, cols] = truth.astype(np.int32) + 1
    pred_map[rows, cols] = prediction.astype(np.int32) + 1

    correct = truth == prediction
    error_codes = np.zeros(truth.size, dtype=np.int32)
    error_codes[correct] = 1
    if phase_to_class_ids is None:
        error_codes[~correct] = 2
        error_labels = {0: "Background / unseen", 1: "Correct", 2: "Incorrect"}
    else:
        phase_by_class: dict[int, int] = {}
        groups: dict[int, list[int]] = {}
        for raw_phase, raw_classes in sorted(phase_to_class_ids.items()):
            phase_origin = _exact_int(raw_phase, "phase origin", minimum=0)
            group = _class_ids(raw_classes, total)
            if set(group) & set(phase_by_class):
                raise ValueError("phase_to_class_ids contains duplicate classes")
            groups[phase_origin] = group
            phase_by_class.update({class_id: phase_origin for class_id in group})
        if set(phase_by_class) != allowed:
            raise ValueError("phase_to_class_ids must partition visible classes")
        error_labels = {0: "Background / unseen", 1: "Correct"}
        transition_code: dict[tuple[int, int], int] = {}
        code = 2
        for true_phase in sorted(groups):
            for pred_phase in sorted(groups):
                transition_code[(true_phase, pred_phase)] = code
                error_labels[code] = f"Wrong within P{true_phase}" if true_phase == pred_phase else f"P{true_phase} -> P{pred_phase}"
                code += 1
        for index, (target, pred) in enumerate(zip(truth.tolist(), prediction.tolist())):
            if not correct[index]:
                error_codes[index] = transition_code[(phase_by_class[target], phase_by_class[pred])]
    error_map = np.zeros(shape, dtype=np.int32)
    error_map[rows, cols] = error_codes

    root = os.path.abspath(output_dir)
    os.makedirs(root, exist_ok=True)
    prefix = f"phase_{phase_id:02d}"
    names = list(target_names or [])
    outputs = {
        "ground_truth": _render_map(os.path.join(root, f"{prefix}_ground_truth.png"), gt_map, total_classes=total, cmap_name=cmap_name, dpi=dpi_value),
        "prediction": _render_map(os.path.join(root, f"{prefix}_prediction.png"), pred_map, total_classes=total, cmap_name=cmap_name, dpi=dpi_value),
        "error_type": _render_error_map(os.path.join(root, f"{prefix}_error_types.png"), error_map, labels=error_labels, dpi=dpi_value),
        "legend": _render_legend(os.path.join(root, f"{prefix}_legend.png"), class_ids=ids, target_names=names, total_classes=total, cmap_name=cmap_name, dpi=dpi_value),
    }
    if decision_margins is not None:
        margins = np.asarray(decision_margins, dtype=np.float64).reshape(-1)
        if margins.shape != truth.shape or not np.isfinite(margins).all():
            raise ValueError("decision_margins must be finite and aligned")
        margin_map = np.full(shape, np.nan, dtype=np.float64)
        margin_map[rows, cols] = margins
        outputs["decision_margin"] = _render_margin_map(os.path.join(root, f"{prefix}_decision_margin.png"), margin_map, dpi=dpi_value)
    return outputs


__all__ = ["save_full_phase_qualitative_maps"]










# from __future__ import annotations

# """Reporting-only qualitative visualization for finalized NECIL-HSI phases.

# Only separate per-phase figures are saved. Visualization never participates in
# optimization, model selection, CDS construction, or checkpoint state.
# """

# import math
# import os
# from numbers import Integral, Real
# from typing import Dict, Mapping, Optional, Sequence

# import matplotlib
# matplotlib.use("Agg")
# import matplotlib.pyplot as plt
# import numpy as np
# from matplotlib.colors import BoundaryNorm, ListedColormap
# from matplotlib.patches import Patch


# def _exact_int(value: object, name: str, *, minimum: int | None = None) -> int:
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
#     return result


# def _int_array(values: object, *, name: str) -> np.ndarray:
#     raw = np.asarray(values)
#     if raw.dtype == np.bool_ or raw.dtype.kind not in "iuf" or np.iscomplexobj(raw):
#         raise ValueError(f"{name} must contain integer values")
#     if np.issubdtype(raw.dtype, np.floating):
#         if not np.isfinite(raw).all() or not np.equal(raw, np.round(raw)).all():
#             raise ValueError(f"{name} must contain finite integer values")
#     return raw.astype(np.int64, copy=False)


# def _class_ids(values: Sequence[int], total_classes: int) -> list[int]:
#     total = _exact_int(total_classes, "total_classes", minimum=1)
#     ids = [_exact_int(v, "class_id", minimum=0) for v in values]
#     if not ids or len(ids) != len(set(ids)) or max(ids) >= total:
#         raise ValueError("class_ids must be unique visible dataset class IDs")
#     return ids


# def _palette(total_classes: int, cmap_name: str) -> tuple[ListedColormap, BoundaryNorm]:
#     total = _exact_int(total_classes, "total_classes", minimum=1)
#     try:
#         source = plt.get_cmap(str(cmap_name))
#     except ValueError as exc:
#         raise ValueError(f"unknown matplotlib colormap {cmap_name!r}") from exc
#     positions = (np.arange(total, dtype=np.float64) + 0.5) / float(total)
#     background = np.asarray([[0.0, 0.0, 0.0, 1.0]], dtype=np.float64)
#     cmap = ListedColormap(np.vstack((background, source(positions))))
#     boundaries = np.arange(-0.5, total + 1.5, 1.0, dtype=np.float64)
#     return cmap, BoundaryNorm(boundaries, ncolors=cmap.N, clip=True)


# def _render_map(path: str, values: np.ndarray, *, total_classes: int, cmap_name: str, dpi: int) -> str:
#     array = np.asarray(values, dtype=np.int32)
#     if array.ndim != 2 or np.any(array < 0) or np.any(array > int(total_classes)):
#         raise ValueError("qualitative map contains invalid rendered IDs")
#     destination = os.path.abspath(path)
#     os.makedirs(os.path.dirname(destination) or ".", exist_ok=True)
#     dpi_value = _exact_int(dpi, "dpi", minimum=1)
#     cmap, norm = _palette(total_classes, cmap_name)
#     h, w = array.shape
#     aspect = float(w) / float(h)
#     long_side = 7.2
#     size = (long_side, long_side / aspect) if aspect >= 1.0 else (long_side * aspect, long_side)
#     figure = plt.figure(figsize=size, dpi=dpi_value, frameon=False)
#     axis = figure.add_axes([0.0, 0.0, 1.0, 1.0])
#     axis.imshow(array, cmap=cmap, norm=norm, interpolation="nearest")
#     axis.set_axis_off()
#     figure.savefig(destination, dpi=dpi_value, bbox_inches=None, transparent=False, pad_inches=0.0)
#     plt.close(figure)
#     return destination


# def _render_legend(path: str, *, class_ids: Sequence[int], target_names: Sequence[str], total_classes: int, cmap_name: str, dpi: int) -> str:
#     destination = os.path.abspath(path)
#     os.makedirs(os.path.dirname(destination) or ".", exist_ok=True)
#     cmap, norm = _palette(total_classes, cmap_name)
#     handles = [Patch(facecolor=cmap(norm(0)), edgecolor="none", label="Background / unseen")]
#     names = list(target_names)
#     for class_id in class_ids:
#         name = str(names[class_id]) if class_id < len(names) else f"Class {class_id + 1}"
#         handles.append(Patch(facecolor=cmap(norm(class_id + 1)), edgecolor="none", label=f"{class_id + 1}. {name}"))
#     columns = 1 if len(handles) <= 8 else 2
#     rows = int(math.ceil(len(handles) / columns))
#     figure = plt.figure(figsize=(4.6 * columns, max(1.2, 0.42 * rows)), dpi=int(dpi))
#     axis = figure.add_axes([0.0, 0.0, 1.0, 1.0])
#     axis.set_axis_off()
#     axis.legend(handles=handles, loc="center", ncol=columns, frameon=False, fontsize=10)
#     figure.savefig(destination, dpi=int(dpi), bbox_inches="tight", transparent=False, pad_inches=0.04)
#     plt.close(figure)
#     return destination


# def _render_margin_map(path: str, values: np.ndarray, *, dpi: int) -> str:
#     destination = os.path.abspath(path)
#     os.makedirs(os.path.dirname(destination) or ".", exist_ok=True)
#     array = np.asarray(values, dtype=np.float64)
#     finite = np.isfinite(array)
#     if array.ndim != 2 or not finite.any():
#         raise ValueError("margin map must contain a finite labeled pixel")
#     bound = float(np.max(np.abs(array[finite])))
#     if not math.isfinite(bound) or bound <= 0.0:
#         bound = 1.0
#     cmap = plt.get_cmap("coolwarm").copy()
#     cmap.set_bad("black")
#     figure, axis = plt.subplots(figsize=(7.2, 6.0), dpi=int(dpi))
#     image = axis.imshow(np.ma.masked_invalid(array), cmap=cmap, vmin=-bound, vmax=bound, interpolation="nearest")
#     axis.set_title("Classifier decision margin")
#     axis.set_axis_off()
#     colorbar = figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
#     colorbar.set_label("Target-to-nearest-rival logit margin")
#     figure.tight_layout()
#     figure.savefig(destination, dpi=int(dpi), bbox_inches="tight")
#     plt.close(figure)
#     return destination


# def _render_error_map(path: str, values: np.ndarray, *, labels: Mapping[int, str], dpi: int) -> str:
#     destination = os.path.abspath(path)
#     os.makedirs(os.path.dirname(destination) or ".", exist_ok=True)
#     array = np.asarray(values, dtype=np.int32)
#     if array.ndim != 2:
#         raise ValueError("error map must be two-dimensional")
#     max_code = max(labels) if labels else 0
#     cmap = plt.get_cmap("tab10", max_code + 1)
#     figure, axis = plt.subplots(figsize=(7.2, 6.0), dpi=int(dpi))
#     axis.imshow(array, cmap=cmap, vmin=-0.5, vmax=max_code + 0.5, interpolation="nearest")
#     axis.set_title("Classification error type")
#     axis.set_axis_off()
#     handles = [Patch(facecolor=cmap(code), edgecolor="none", label=label) for code, label in sorted(labels.items())]
#     axis.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False)
#     figure.tight_layout()
#     figure.savefig(destination, dpi=int(dpi), bbox_inches="tight")
#     plt.close(figure)
#     return destination


# def save_full_phase_qualitative_maps(
#     *, output_dir: str, phase: int, gt_shape: Sequence[int], coords: np.ndarray,
#     targets: np.ndarray, predictions: np.ndarray, class_ids: Sequence[int],
#     total_classes: int, target_names: Optional[Sequence[str]] = None,
#     decision_margins: Optional[np.ndarray] = None,
#     phase_to_class_ids: Optional[Mapping[int, Sequence[int]]] = None,
#     cmap_name: str = "nipy_spectral", dpi: int = 300,
# ) -> Dict[str, str]:
#     phase_id = _exact_int(phase, "phase", minimum=0)
#     total = _exact_int(total_classes, "total_classes", minimum=1)
#     dpi_value = _exact_int(dpi, "dpi", minimum=1)
#     ids = _class_ids(class_ids, total)
#     truth = _int_array(targets, name="targets").reshape(-1)
#     prediction = _int_array(predictions, name="predictions").reshape(-1)
#     coordinates = _int_array(coords, name="coords")
#     shape = tuple(_exact_int(v, "gt_shape", minimum=1) for v in gt_shape)
#     if len(shape) != 2 or coordinates.shape != (truth.size, 2) or prediction.shape != truth.shape:
#         raise ValueError("qualitative inputs are misaligned")
#     if np.unique(coordinates, axis=0).shape[0] != coordinates.shape[0]:
#         raise ValueError("coords contain duplicate image locations")
#     allowed = set(ids)
#     if set(truth.tolist()) - allowed or set(prediction.tolist()) - allowed:
#         raise ValueError("targets/predictions contain phase-invisible classes")
#     if np.any(coordinates[:, 0] < 0) or np.any(coordinates[:, 0] >= shape[0]) or np.any(coordinates[:, 1] < 0) or np.any(coordinates[:, 1] >= shape[1]):
#         raise ValueError("coords lie outside gt_shape")

#     rows, cols = coordinates[:, 0], coordinates[:, 1]
#     gt_map = np.zeros(shape, dtype=np.int32)
#     pred_map = np.zeros(shape, dtype=np.int32)
#     gt_map[rows, cols] = truth.astype(np.int32) + 1
#     pred_map[rows, cols] = prediction.astype(np.int32) + 1

#     correct = truth == prediction
#     error_codes = np.zeros(truth.size, dtype=np.int32)
#     error_codes[correct] = 1
#     if phase_to_class_ids is None:
#         error_codes[~correct] = 2
#         error_labels = {0: "Background / unseen", 1: "Correct", 2: "Incorrect"}
#     else:
#         phase_by_class: dict[int, int] = {}
#         groups: dict[int, list[int]] = {}
#         for raw_phase, raw_classes in sorted(phase_to_class_ids.items()):
#             phase_origin = _exact_int(raw_phase, "phase origin", minimum=0)
#             group = _class_ids(raw_classes, total)
#             if set(group) & set(phase_by_class):
#                 raise ValueError("phase_to_class_ids contains duplicate classes")
#             groups[phase_origin] = group
#             phase_by_class.update({class_id: phase_origin for class_id in group})
#         if set(phase_by_class) != allowed:
#             raise ValueError("phase_to_class_ids must partition visible classes")
#         error_labels = {0: "Background / unseen", 1: "Correct"}
#         transition_code: dict[tuple[int, int], int] = {}
#         code = 2
#         for true_phase in sorted(groups):
#             for pred_phase in sorted(groups):
#                 transition_code[(true_phase, pred_phase)] = code
#                 error_labels[code] = f"Wrong within P{true_phase}" if true_phase == pred_phase else f"P{true_phase} -> P{pred_phase}"
#                 code += 1
#         for index, (target, pred) in enumerate(zip(truth.tolist(), prediction.tolist())):
#             if not correct[index]:
#                 error_codes[index] = transition_code[(phase_by_class[target], phase_by_class[pred])]
#     error_map = np.zeros(shape, dtype=np.int32)
#     error_map[rows, cols] = error_codes

#     root = os.path.abspath(output_dir)
#     os.makedirs(root, exist_ok=True)
#     prefix = f"phase_{phase_id:02d}"
#     names = list(target_names or [])
#     outputs = {
#         "ground_truth": _render_map(os.path.join(root, f"{prefix}_ground_truth.png"), gt_map, total_classes=total, cmap_name=cmap_name, dpi=dpi_value),
#         "prediction": _render_map(os.path.join(root, f"{prefix}_prediction.png"), pred_map, total_classes=total, cmap_name=cmap_name, dpi=dpi_value),
#         "error_type": _render_error_map(os.path.join(root, f"{prefix}_error_types.png"), error_map, labels=error_labels, dpi=dpi_value),
#         "legend": _render_legend(os.path.join(root, f"{prefix}_legend.png"), class_ids=ids, target_names=names, total_classes=total, cmap_name=cmap_name, dpi=dpi_value),
#     }
#     if decision_margins is not None:
#         margins = np.asarray(decision_margins, dtype=np.float64).reshape(-1)
#         if margins.shape != truth.shape or not np.isfinite(margins).all():
#             raise ValueError("decision_margins must be finite and aligned")
#         margin_map = np.full(shape, np.nan, dtype=np.float64)
#         margin_map[rows, cols] = margins
#         outputs["decision_margin"] = _render_margin_map(os.path.join(root, f"{prefix}_decision_margin.png"), margin_map, dpi=dpi_value)
#     return outputs


# __all__ = ["save_full_phase_qualitative_maps"]
