from __future__ import annotations

"""Read-only classifier evaluation and class-structure diagnostics.

Classifier metrics and class-structure statistics are reported separately.
Class statistics never produce logits or labels.
"""

from contextlib import contextmanager
import math
from numbers import Integral, Real
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np
import torch

from models.classifier import PersistentCosineClassifier
from models.geometry_bank import HSIClassDiscriminativeGeometry

Tensor = torch.Tensor


def _as_int(value: object, *, name: str) -> int:
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
    if result < 0:
        raise ValueError(f"{name} must be non-negative")
    return result


def _class_ids(values: Sequence[int], *, name: str = "class_ids") -> list[int]:
    ids = [_as_int(v, name=f"{name} value") for v in values]
    if not ids:
        raise ValueError(f"{name} cannot be empty")
    if len(ids) != len(set(ids)):
        raise ValueError(f"{name} must contain unique IDs")
    return ids


@contextmanager
def _temporary_eval_state(model: Any):
    states = {module: bool(module.training) for module in model.modules()}
    try:
        model.eval()
        yield
    finally:
        for module, state in states.items():
            module.training = state


def _confusion(
    targets: np.ndarray,
    predictions: np.ndarray,
    ids: Sequence[int],
) -> np.ndarray:
    index = {class_id: row for row, class_id in enumerate(ids)}
    matrix = np.zeros((len(ids), len(ids)), dtype=np.int64)
    for target, prediction in zip(targets.tolist(), predictions.tolist()):
        if target not in index or prediction not in index:
            raise RuntimeError("confusion input contains an unknown class ID")
        matrix[index[target], index[prediction]] += 1
    return matrix


def _kappa(matrix: np.ndarray) -> float:
    total = float(matrix.sum())
    if total <= 0.0:
        raise ValueError("confusion matrix is empty")
    observed = float(np.trace(matrix)) / total
    expected = float(
        (matrix.sum(axis=1) * matrix.sum(axis=0)).sum()
    ) / (total * total)
    denominator = 1.0 - expected
    return 0.0 if denominator == 0.0 else (observed - expected) / denominator


def _unpack_eval_batch(
    batch: Mapping[str, Any],
    *,
    model: Any,
    device: torch.device,
) -> tuple[Tensor, Tensor, Tensor]:
    if not isinstance(batch, Mapping):
        raise TypeError("HSI batches must be mappings")
    required = {"image", "raw_center_spectrum", "label"}
    missing = required - set(batch)
    if missing:
        raise KeyError(f"evaluation batch missing {sorted(missing)}")

    patch = torch.as_tensor(batch["image"], device=device, dtype=model.dtype)
    spectrum = torch.as_tensor(
        batch["raw_center_spectrum"],
        device=device,
        dtype=model.dtype,
    )
    labels = torch.as_tensor(batch["label"], device=device).flatten()

    backbone = model.backbone
    expected_patch = (
        patch.size(0) if patch.ndim > 0 else 0,
        int(backbone.patch_bands),
        int(backbone.patch_size),
        int(backbone.patch_size),
    )
    expected_spectrum = (
        patch.size(0) if patch.ndim > 0 else 0,
        int(backbone.spectral_bands),
    )
    if patch.ndim != 4 or patch.size(0) == 0 or tuple(patch.shape) != expected_patch:
        raise RuntimeError("evaluation patch shape disagrees with backbone")
    if spectrum.ndim != 2 or tuple(spectrum.shape) != expected_spectrum:
        raise RuntimeError("evaluation spectrum shape disagrees with backbone")
    if not bool(torch.isfinite(patch).all()) or not bool(torch.isfinite(spectrum).all()):
        raise RuntimeError("evaluation input contains NaN or Inf")
    if labels.numel() != patch.size(0) or labels.dtype == torch.bool or labels.is_complex():
        raise RuntimeError("evaluation labels are invalid")
    if torch.is_floating_point(labels):
        if not bool(torch.isfinite(labels).all()) or not bool(labels.eq(labels.round()).all()):
            raise RuntimeError("evaluation labels must be finite integers")
    labels = labels.to(torch.long)
    if bool((labels < 0).any()):
        raise RuntimeError("evaluation labels must be non-negative")
    return patch, spectrum, labels


@torch.no_grad()
def evaluate_loader(
    model: Any,
    loader: Any,
    *,
    class_ids: Sequence[int],
    target_class_ids: Optional[Sequence[int]] = None,
    geometry_bank: Optional[HSIClassDiscriminativeGeometry] = None,
    device: str | torch.device,
) -> Dict[str, Any]:
    """Evaluate the persistent classifier.

    ``geometry_bank`` is accepted only for API compatibility with older trainer
    wrappers.  If supplied it is validated but never used for classification.
    """
    ids = _class_ids(class_ids)
    target_ids = ids if target_class_ids is None else _class_ids(
        target_class_ids,
        name="target_class_ids",
    )
    outside = sorted(set(target_ids) - set(ids))
    if outside:
        raise ValueError(f"target_class_ids are outside classifier classes: {outside}")

    evaluation_device = torch.device(device)
    if torch.device(model.device) != evaluation_device:
        raise RuntimeError("model and evaluation device disagree")
    model.validate_model_state()
    if not isinstance(model.classifier, PersistentCosineClassifier):
        raise RuntimeError("model classifier must be PersistentCosineClassifier")

    if geometry_bank is not None:
        if not isinstance(geometry_bank, HSIClassDiscriminativeGeometry):
            raise TypeError("geometry_bank must be HSIClassDiscriminativeGeometry")
        geometry_bank.validate_state()

    classifier_ids = set(model.classifier_class_ids)
    missing_classifier = sorted(set(ids) - classifier_ids)
    if missing_classifier:
        raise RuntimeError(
            f"classifier lacks requested class IDs {missing_classifier}"
        )

    predictions: list[Tensor] = []
    labels_rows: list[Tensor] = []
    target_cosine_rows: list[Tensor] = []
    rival_cosine_rows: list[Tensor] = []
    margin_rows: list[Tensor] = []
    qrms_rows: list[Tensor] = []
    class_tensor: Tensor | None = None

    with _temporary_eval_state(model):
        for batch in loader:
            patch, spectrum, labels = _unpack_eval_batch(
                batch,
                model=model,
                device=evaluation_device,
            )
            output = model.extract_features(
                patch,
                center_spectrum=spectrum,
                return_aux=False,
            )
            q = output.ordered_features
            classification = model.classify_features(q, class_ids=ids)

            if class_tensor is None:
                class_tensor = classification.class_ids.detach().clone()
            elif not bool(torch.equal(class_tensor, classification.class_ids)):
                raise RuntimeError("classifier class-column order changed across batches")

            allowed_labels = torch.tensor(ids, device=labels.device, dtype=torch.long)
            if not bool(torch.isin(labels, allowed_labels).all()):
                raise RuntimeError("evaluation loader contains a class outside class_ids")

            matches = labels.unsqueeze(1).eq(classification.class_ids.unsqueeze(0))
            counts = matches.sum(dim=1)
            if not bool(counts.eq(1).all()):
                raise RuntimeError("could not map evaluation labels to classifier columns")
            targets_local = matches.to(torch.long).argmax(dim=1)
            rows = torch.arange(labels.numel(), device=labels.device)
            cosine = classification.cosine_similarity
            target_cosine = cosine[rows, targets_local]
            rival = cosine.clone()
            rival[rows, targets_local] = -torch.inf
            nearest_rival = rival.max(dim=1).values

            flat = q.reshape(q.size(0), -1)
            qrms = torch.linalg.vector_norm(flat, ord=2, dim=1) / math.sqrt(
                float(flat.size(1))
            )

            predictions.append(classification.prediction.detach())
            labels_rows.append(labels.detach())
            target_cosine_rows.append(target_cosine.detach())
            rival_cosine_rows.append(nearest_rival.detach())
            margin_rows.append((target_cosine - nearest_rival).detach())
            qrms_rows.append(qrms.detach())

    if not predictions or class_tensor is None:
        raise RuntimeError("cannot evaluate an empty loader")

    prediction_all = torch.cat(predictions, dim=0)
    labels_all = torch.cat(labels_rows, dim=0).flatten()
    target_cosine_all = torch.cat(target_cosine_rows, dim=0)
    rival_cosine_all = torch.cat(rival_cosine_rows, dim=0)
    margin_all = torch.cat(margin_rows, dim=0)
    qrms_all = torch.cat(qrms_rows, dim=0)

    target_mask = torch.zeros_like(labels_all, dtype=torch.bool)
    for class_id in target_ids:
        target_mask |= labels_all.eq(class_id)
    if not bool(target_mask.any()):
        raise RuntimeError("evaluation loader has no requested target classes")

    labels = labels_all[target_mask]
    prediction = prediction_all[target_mask]
    target_cosine = target_cosine_all[target_mask]
    rival_cosine = rival_cosine_all[target_mask]
    margin = margin_all[target_mask]
    qrms = qrms_all[target_mask]

    per_class_accuracy: Dict[int, float] = {}
    per_class_margin: Dict[int, float] = {}
    class_counts: Dict[int, int] = {}
    for class_id in target_ids:
        mask = labels.eq(class_id)
        count = int(mask.sum().item())
        if count <= 0:
            raise RuntimeError(f"evaluation target class {class_id} is absent")
        class_counts[class_id] = count
        per_class_accuracy[class_id] = float(
            prediction[mask].eq(labels[mask]).to(torch.float32).mean().item()
        )
        per_class_margin[class_id] = float(margin[mask].mean().item())

    # Keep all seen columns in the confusion matrix.  This matters for future
    # old/new summaries because a target old class may be predicted as a new one.
    confusion = _confusion(
        labels.detach().cpu().numpy(),
        prediction.detach().cpu().numpy(),
        ids,
    )
    oa = float(prediction.eq(labels).to(torch.float32).mean().item())
    ba = float(sum(per_class_accuracy.values()) / len(per_class_accuracy))

    return {
        "overall_accuracy": oa,
        "accuracy": oa,
        "balanced_accuracy": ba,
        "minimum_class_accuracy": float(min(per_class_accuracy.values())),
        "kappa": float(_kappa(confusion)),
        "confusion_matrix": confusion.tolist(),
        "class_counts": class_counts,
        "per_class_accuracy": per_class_accuracy,
        "mean_cosine_margin": float(margin.mean().item()),
        "per_class_mean_cosine_margin": per_class_margin,
        "mean_feature_rms": float(qrms.mean().item()),
        "mean_structural_feature_rms": float(qrms.mean().item()),
    }



@torch.no_grad()
def collect_reporting_features(
    model: Any,
    loader: Any,
    *,
    class_ids: Sequence[int],
    device: str | torch.device,
) -> Dict[str, Any]:
    """Collect canonical Q for post-training representation analysis only.

    The returned per-sample features are ordinary function outputs.  This
    helper does not write them to model state/checkpoints and must not be called
    from an optimization path.
    """
    ids = _class_ids(class_ids)
    evaluation_device = torch.device(device)
    if torch.device(model.device) != evaluation_device:
        raise RuntimeError("model and reporting device disagree")

    allowed = torch.tensor(
        ids,
        device=evaluation_device,
        dtype=torch.long,
    )
    feature_rows: list[Tensor] = []
    label_rows: list[Tensor] = []
    coordinate_rows: list[Tensor] = []
    saw_coordinates: Optional[bool] = None

    with _temporary_eval_state(model):
        for batch in loader:
            patch, spectrum, labels = _unpack_eval_batch(
                batch,
                model=model,
                device=evaluation_device,
            )
            mask = torch.isin(labels, allowed)
            if not bool(mask.any()):
                continue

            output = model.extract_features(
                patch,
                center_spectrum=spectrum,
                return_aux=False,
            )
            q = output.ordered_features
            if q.ndim != 3 or q.size(0) != labels.numel():
                raise RuntimeError(
                    "reporting canonical features have invalid shape"
                )
            if not bool(torch.isfinite(q).all()):
                raise RuntimeError(
                    "reporting canonical features contain NaN/Inf"
                )

            feature_rows.append(q[mask].detach().cpu())
            label_rows.append(labels[mask].detach().cpu())

            has_coordinates = "coord" in batch
            if saw_coordinates is None:
                saw_coordinates = has_coordinates
            elif saw_coordinates != has_coordinates:
                raise RuntimeError(
                    "coord must be present in every reporting batch or none"
                )
            if has_coordinates:
                coords = torch.as_tensor(batch["coord"]).detach().cpu()
                if coords.shape != (labels.numel(), 2):
                    raise RuntimeError(
                        "reporting coord must have shape [N,2]"
                    )
                coordinate_rows.append(coords[mask.detach().cpu()])

    if not feature_rows:
        raise RuntimeError(
            "reporting loader contains none of the requested classes"
        )

    features = torch.cat(feature_rows, dim=0)
    labels = torch.cat(label_rows, dim=0).flatten()
    observed = sorted(int(v) for v in labels.unique().tolist())
    if observed != sorted(ids):
        raise RuntimeError(
            f"reporting features cover classes {observed}, expected {sorted(ids)}"
        )

    result: Dict[str, Any] = {
        "ordered_features": features.numpy(),
        "labels": labels.numpy().astype(np.int64, copy=False),
    }
    if saw_coordinates:
        coords = torch.cat(coordinate_rows, dim=0)
        if coords.shape != (labels.numel(), 2):
            raise RuntimeError(
                "reporting coordinates are row-misaligned"
            )
        result["coords"] = coords.numpy().astype(np.int64, copy=False)
    return result


def _empirical_class_structure(
    features: np.ndarray,
    labels: np.ndarray,
    class_ids: Sequence[int],
) -> Dict[int, tuple[np.ndarray, np.ndarray]]:
    q = np.asarray(features, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int64).reshape(-1)
    if q.ndim != 3 or q.shape[0] != y.size:
        raise ValueError(
            "empirical structure requires features [N,K,D] aligned with labels"
        )
    if not np.isfinite(q).all():
        raise ValueError("empirical structure features contain NaN/Inf")

    output: Dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for class_id in _class_ids(class_ids):
        rows = q[y == class_id]
        if rows.shape[0] == 0:
            raise ValueError(
                f"empirical structure lacks class {class_id}"
            )
        identity = rows.mean(axis=0)
        residual = rows - identity[None, ...]
        structure = np.einsum(
            "nkd,nld->kld",
            residual,
            residual,
            optimize=True,
        ) / float(rows.shape[0])
        structure = 0.5 * (
            structure + np.swapaxes(structure, 0, 1)
        )
        output[class_id] = (identity, structure)
    return output


def structure_drift_diagnostics(
    previous: Mapping[str, Any],
    current: Mapping[str, Any],
    *,
    old_class_ids: Sequence[int],
) -> Dict[str, Any]:
    """Measure realized old-class I/S drift on identical held-out samples.

    This is reporting-only.  It does not compare the persistent training
    statistics directly because those are phase-acquisition references.  The
    diagnostic recomputes empirical I/S from the same held-out old samples
    under two finalized checkpoints.
    """
    ids = _class_ids(old_class_ids)

    previous_features = np.asarray(previous["ordered_features"])
    current_features = np.asarray(current["ordered_features"])
    previous_labels = np.asarray(previous["labels"], dtype=np.int64)
    current_labels = np.asarray(current["labels"], dtype=np.int64)

    if previous_features.shape != current_features.shape:
        raise ValueError(
            "previous/current old reporting features must have identical shape"
        )
    if not np.array_equal(previous_labels, current_labels):
        raise ValueError(
            "previous/current old reporting labels are not aligned"
        )

    previous_coords = previous.get("coords")
    current_coords = current.get("coords")
    if (previous_coords is None) != (current_coords is None):
        raise ValueError(
            "previous/current reporting coordinate availability differs"
        )
    if previous_coords is not None and not np.array_equal(
        np.asarray(previous_coords),
        np.asarray(current_coords),
    ):
        raise ValueError(
            "previous/current old reporting samples are not coordinate-aligned"
        )

    before = _empirical_class_structure(
        previous_features,
        previous_labels,
        ids,
    )
    after = _empirical_class_structure(
        current_features,
        current_labels,
        ids,
    )

    classes: list[Dict[str, Any]] = []
    identity_values: list[float] = []
    structure_values: list[float] = []

    for class_id in ids:
        identity_before, structure_before = before[class_id]
        identity_after, structure_after = after[class_id]
        identity_drift = float(
            np.sqrt(
                np.mean(
                    np.square(identity_after - identity_before)
                )
            )
        )
        structure_drift = float(
            np.sqrt(
                np.mean(
                    np.square(structure_after - structure_before)
                )
            )
        )
        identity_values.append(identity_drift)
        structure_values.append(structure_drift)
        classes.append(
            {
                "class_id": class_id,
                "identity_drift_rms": identity_drift,
                "structure_drift_rms": structure_drift,
            }
        )

    return {
        "old_class_ids": ids,
        "mean_identity_drift_rms": float(np.mean(identity_values)),
        "mean_structure_drift_rms": float(np.mean(structure_values)),
        "classes": classes,
    }


def _rms(value: Tensor) -> float:
    tensor = torch.as_tensor(value)
    return float(torch.sqrt(tensor.square().mean()).item())


@torch.no_grad()
def geometry_diagnostics(
    model: Any,
    class_ids: Sequence[int],
    *,
    target_names: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """Summarize retained {I_c,S_c,N_c} class statistics."""
    ids = _class_ids(class_ids)
    bank = getattr(model, "geometry_bank", None)
    if not isinstance(bank, HSIClassDiscriminativeGeometry):
        raise RuntimeError("model.geometry_bank must be HSIClassDiscriminativeGeometry")
    bank.validate_state()

    missing = sorted(set(ids) - set(bank.seen_class_ids))
    if missing:
        raise RuntimeError(f"geometry bank lacks requested classes {missing}")

    names = list(target_names or [])
    classes: list[Dict[str, Any]] = []
    for class_id in ids:
        view = bank.class_geometry(class_id)
        structure = view.structure
        k = int(structure.size(0))
        upper = torch.triu_indices(k, k, device=structure.device)
        unique_structure = structure[upper[0], upper[1], :]

        cross = view.cross_region_covariation
        offdiag_mask = ~torch.eye(k, dtype=torch.bool, device=cross.device)
        offdiag = cross[offdiag_mask]

        classes.append(
            {
                "class_id": class_id,
                "class_name": (
                    str(names[class_id])
                    if class_id < len(names)
                    else f"Class {class_id + 1}"
                ),
                "sample_count": int(view.sample_count),
                "identity_rms": _rms(view.identity),
                "structure_unique_rms": _rms(unique_structure),
                "feature_variation_mean": float(view.feature_variation.mean().item()),
                "cross_region_offdiagonal_rms": (
                    _rms(offdiag) if offdiag.numel() else 0.0
                ),
            }
        )

    pairs: list[Dict[str, Any]] = []
    for left_pos, left_id in enumerate(ids):
        for right_id in ids[left_pos + 1 :]:
            relation = bank.inter_class_geometry(left_id, right_id)
            pairs.append(
                {
                    "left_class_id": left_id,
                    "right_class_id": right_id,
                    "left_class_name": (
                        str(names[left_id])
                        if left_id < len(names)
                        else f"Class {left_id + 1}"
                    ),
                    "right_class_name": (
                        str(names[right_id])
                        if right_id < len(names)
                        else f"Class {right_id + 1}"
                    ),
                    "identity_difference_rms": _rms(
                        relation.identity_difference
                    ),
                    "structure_difference_rms": _rms(
                        relation.structure_difference
                    ),
                }
            )

    return {
        "class_ids": ids,
        "spectral_region_count": int(bank.spectral_token_count),
        "feature_dimension": int(bank.spectral_token_dim),
        "classes": classes,
        "pairs": pairs,
    }


def _mapping_value(
    mapping: Mapping[str | int, Any],
    class_id: int,
    *,
    field: str,
) -> float:
    value = mapping[class_id] if class_id in mapping else mapping.get(str(class_id))
    if value is None:
        raise ValueError(f"{field} lacks class {class_id}")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field} for class {class_id} must be finite")
    return result


def summarize_class_group(
    metrics: Mapping[str, Any],
    class_ids: Sequence[int],
) -> Dict[str, Any]:
    ids = _class_ids(class_ids)
    accuracy_source = metrics.get("per_class_accuracy")
    margin_source = metrics.get("per_class_mean_cosine_margin")
    if not isinstance(accuracy_source, Mapping) or not isinstance(margin_source, Mapping):
        raise ValueError("metrics lack classifier classwise fields")
    accuracies = [
        _mapping_value(accuracy_source, class_id, field="per_class_accuracy")
        for class_id in ids
    ]
    margins = [
        _mapping_value(margin_source, class_id, field="per_class_mean_cosine_margin")
        for class_id in ids
    ]
    return {
        "class_ids": ids,
        "balanced_accuracy": sum(accuracies) / len(accuracies),
        "minimum_class_accuracy": min(accuracies),
        "mean_cosine_margin": sum(margins) / len(margins),
    }


def old_new_summary(
    metrics: Mapping[str, Any],
    *,
    old_class_ids: Sequence[int],
    new_class_ids: Sequence[int],
) -> Dict[str, Any]:
    old = summarize_class_group(metrics, old_class_ids)
    new = summarize_class_group(metrics, new_class_ids)
    old_accuracy = float(old["balanced_accuracy"])
    new_accuracy = float(new["balanced_accuracy"])
    denominator = old_accuracy + new_accuracy
    harmonic = 0.0 if denominator == 0.0 else 2.0 * old_accuracy * new_accuracy / denominator
    return {
        "old": old,
        "new": new,
        "harmonic_old_new_accuracy": harmonic,
    }


__all__ = [
    "evaluate_loader",
    "collect_reporting_features",
    "structure_drift_diagnostics",
    "geometry_diagnostics",
    "summarize_class_group",
    "old_new_summary",
]













# from __future__ import annotations

# """Read-only classifier evaluation and class-structure diagnostics.

# Classifier metrics and class-structure statistics are reported separately.
# Class statistics never produce logits or labels.
# """

# from contextlib import contextmanager
# import math
# from numbers import Integral, Real
# from typing import Any, Dict, Mapping, Optional, Sequence

# import numpy as np
# import torch

# from models.classifier import PersistentCosineClassifier
# from models.geometry_bank import HSIClassDiscriminativeGeometry

# Tensor = torch.Tensor


# def _as_int(value: object, *, name: str) -> int:
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
#     if result < 0:
#         raise ValueError(f"{name} must be non-negative")
#     return result


# def _class_ids(values: Sequence[int], *, name: str = "class_ids") -> list[int]:
#     ids = [_as_int(v, name=f"{name} value") for v in values]
#     if not ids:
#         raise ValueError(f"{name} cannot be empty")
#     if len(ids) != len(set(ids)):
#         raise ValueError(f"{name} must contain unique IDs")
#     return ids


# @contextmanager
# def _temporary_eval_state(model: Any):
#     states = {module: bool(module.training) for module in model.modules()}
#     try:
#         model.eval()
#         yield
#     finally:
#         for module, state in states.items():
#             module.training = state


# def _confusion(
#     targets: np.ndarray,
#     predictions: np.ndarray,
#     ids: Sequence[int],
# ) -> np.ndarray:
#     index = {class_id: row for row, class_id in enumerate(ids)}
#     matrix = np.zeros((len(ids), len(ids)), dtype=np.int64)
#     for target, prediction in zip(targets.tolist(), predictions.tolist()):
#         if target not in index or prediction not in index:
#             raise RuntimeError("confusion input contains an unknown class ID")
#         matrix[index[target], index[prediction]] += 1
#     return matrix


# def _kappa(matrix: np.ndarray) -> float:
#     total = float(matrix.sum())
#     if total <= 0.0:
#         raise ValueError("confusion matrix is empty")
#     observed = float(np.trace(matrix)) / total
#     expected = float(
#         (matrix.sum(axis=1) * matrix.sum(axis=0)).sum()
#     ) / (total * total)
#     denominator = 1.0 - expected
#     return 0.0 if denominator == 0.0 else (observed - expected) / denominator


# def _unpack_eval_batch(
#     batch: Mapping[str, Any],
#     *,
#     model: Any,
#     device: torch.device,
# ) -> tuple[Tensor, Tensor, Tensor]:
#     if not isinstance(batch, Mapping):
#         raise TypeError("HSI batches must be mappings")
#     required = {"image", "raw_center_spectrum", "label"}
#     missing = required - set(batch)
#     if missing:
#         raise KeyError(f"evaluation batch missing {sorted(missing)}")

#     patch = torch.as_tensor(batch["image"], device=device, dtype=model.dtype)
#     spectrum = torch.as_tensor(
#         batch["raw_center_spectrum"],
#         device=device,
#         dtype=model.dtype,
#     )
#     labels = torch.as_tensor(batch["label"], device=device).flatten()

#     backbone = model.backbone
#     expected_patch = (
#         patch.size(0) if patch.ndim > 0 else 0,
#         int(backbone.patch_bands),
#         int(backbone.patch_size),
#         int(backbone.patch_size),
#     )
#     expected_spectrum = (
#         patch.size(0) if patch.ndim > 0 else 0,
#         int(backbone.spectral_bands),
#     )
#     if patch.ndim != 4 or patch.size(0) == 0 or tuple(patch.shape) != expected_patch:
#         raise RuntimeError("evaluation patch shape disagrees with backbone")
#     if spectrum.ndim != 2 or tuple(spectrum.shape) != expected_spectrum:
#         raise RuntimeError("evaluation spectrum shape disagrees with backbone")
#     if not bool(torch.isfinite(patch).all()) or not bool(torch.isfinite(spectrum).all()):
#         raise RuntimeError("evaluation input contains NaN or Inf")
#     if labels.numel() != patch.size(0) or labels.dtype == torch.bool or labels.is_complex():
#         raise RuntimeError("evaluation labels are invalid")
#     if torch.is_floating_point(labels):
#         if not bool(torch.isfinite(labels).all()) or not bool(labels.eq(labels.round()).all()):
#             raise RuntimeError("evaluation labels must be finite integers")
#     labels = labels.to(torch.long)
#     if bool((labels < 0).any()):
#         raise RuntimeError("evaluation labels must be non-negative")
#     return patch, spectrum, labels


# @torch.no_grad()
# def evaluate_loader(
#     model: Any,
#     loader: Any,
#     *,
#     class_ids: Sequence[int],
#     target_class_ids: Optional[Sequence[int]] = None,
#     geometry_bank: Optional[HSIClassDiscriminativeGeometry] = None,
#     device: str | torch.device,
# ) -> Dict[str, Any]:
#     """Evaluate the persistent classifier.

#     ``geometry_bank`` is accepted only for API compatibility with older trainer
#     wrappers.  If supplied it is validated but never used for classification.
#     """
#     ids = _class_ids(class_ids)
#     target_ids = ids if target_class_ids is None else _class_ids(
#         target_class_ids,
#         name="target_class_ids",
#     )
#     outside = sorted(set(target_ids) - set(ids))
#     if outside:
#         raise ValueError(f"target_class_ids are outside classifier classes: {outside}")

#     evaluation_device = torch.device(device)
#     if torch.device(model.device) != evaluation_device:
#         raise RuntimeError("model and evaluation device disagree")
#     model.validate_model_state()
#     if not isinstance(model.classifier, PersistentCosineClassifier):
#         raise RuntimeError("model classifier must be PersistentCosineClassifier")

#     if geometry_bank is not None:
#         if not isinstance(geometry_bank, HSIClassDiscriminativeGeometry):
#             raise TypeError("geometry_bank must be HSIClassDiscriminativeGeometry")
#         geometry_bank.validate_state()

#     classifier_ids = set(model.classifier_class_ids)
#     missing_classifier = sorted(set(ids) - classifier_ids)
#     if missing_classifier:
#         raise RuntimeError(
#             f"classifier lacks requested class IDs {missing_classifier}"
#         )

#     predictions: list[Tensor] = []
#     labels_rows: list[Tensor] = []
#     target_cosine_rows: list[Tensor] = []
#     rival_cosine_rows: list[Tensor] = []
#     margin_rows: list[Tensor] = []
#     qrms_rows: list[Tensor] = []
#     class_tensor: Tensor | None = None

#     with _temporary_eval_state(model):
#         for batch in loader:
#             patch, spectrum, labels = _unpack_eval_batch(
#                 batch,
#                 model=model,
#                 device=evaluation_device,
#             )
#             output = model.extract_features(
#                 patch,
#                 center_spectrum=spectrum,
#                 return_aux=False,
#             )
#             q = output.ordered_features
#             classification = model.classify_features(q, class_ids=ids)

#             if class_tensor is None:
#                 class_tensor = classification.class_ids.detach().clone()
#             elif not bool(torch.equal(class_tensor, classification.class_ids)):
#                 raise RuntimeError("classifier class-column order changed across batches")

#             allowed_labels = torch.tensor(ids, device=labels.device, dtype=torch.long)
#             if not bool(torch.isin(labels, allowed_labels).all()):
#                 raise RuntimeError("evaluation loader contains a class outside class_ids")

#             matches = labels.unsqueeze(1).eq(classification.class_ids.unsqueeze(0))
#             counts = matches.sum(dim=1)
#             if not bool(counts.eq(1).all()):
#                 raise RuntimeError("could not map evaluation labels to classifier columns")
#             targets_local = matches.to(torch.long).argmax(dim=1)
#             rows = torch.arange(labels.numel(), device=labels.device)
#             cosine = classification.cosine_similarity
#             target_cosine = cosine[rows, targets_local]
#             rival = cosine.clone()
#             rival[rows, targets_local] = -torch.inf
#             nearest_rival = rival.max(dim=1).values

#             flat = q.reshape(q.size(0), -1)
#             qrms = torch.linalg.vector_norm(flat, ord=2, dim=1) / math.sqrt(
#                 float(flat.size(1))
#             )

#             predictions.append(classification.prediction.detach())
#             labels_rows.append(labels.detach())
#             target_cosine_rows.append(target_cosine.detach())
#             rival_cosine_rows.append(nearest_rival.detach())
#             margin_rows.append((target_cosine - nearest_rival).detach())
#             qrms_rows.append(qrms.detach())

#     if not predictions or class_tensor is None:
#         raise RuntimeError("cannot evaluate an empty loader")

#     prediction_all = torch.cat(predictions, dim=0)
#     labels_all = torch.cat(labels_rows, dim=0).flatten()
#     target_cosine_all = torch.cat(target_cosine_rows, dim=0)
#     rival_cosine_all = torch.cat(rival_cosine_rows, dim=0)
#     margin_all = torch.cat(margin_rows, dim=0)
#     qrms_all = torch.cat(qrms_rows, dim=0)

#     target_mask = torch.zeros_like(labels_all, dtype=torch.bool)
#     for class_id in target_ids:
#         target_mask |= labels_all.eq(class_id)
#     if not bool(target_mask.any()):
#         raise RuntimeError("evaluation loader has no requested target classes")

#     labels = labels_all[target_mask]
#     prediction = prediction_all[target_mask]
#     target_cosine = target_cosine_all[target_mask]
#     rival_cosine = rival_cosine_all[target_mask]
#     margin = margin_all[target_mask]
#     qrms = qrms_all[target_mask]

#     per_class_accuracy: Dict[int, float] = {}
#     per_class_margin: Dict[int, float] = {}
#     class_counts: Dict[int, int] = {}
#     for class_id in target_ids:
#         mask = labels.eq(class_id)
#         count = int(mask.sum().item())
#         if count <= 0:
#             raise RuntimeError(f"evaluation target class {class_id} is absent")
#         class_counts[class_id] = count
#         per_class_accuracy[class_id] = float(
#             prediction[mask].eq(labels[mask]).to(torch.float32).mean().item()
#         )
#         per_class_margin[class_id] = float(margin[mask].mean().item())

#     # Keep all seen columns in the confusion matrix.  This matters for future
#     # old/new summaries because a target old class may be predicted as a new one.
#     confusion = _confusion(
#         labels.detach().cpu().numpy(),
#         prediction.detach().cpu().numpy(),
#         ids,
#     )
#     oa = float(prediction.eq(labels).to(torch.float32).mean().item())
#     ba = float(sum(per_class_accuracy.values()) / len(per_class_accuracy))

#     return {
#         "overall_accuracy": oa,
#         "accuracy": oa,
#         "balanced_accuracy": ba,
#         "minimum_class_accuracy": float(min(per_class_accuracy.values())),
#         "kappa": float(_kappa(confusion)),
#         "confusion_matrix": confusion.tolist(),
#         "class_counts": class_counts,
#         "per_class_accuracy": per_class_accuracy,
#         "mean_cosine_margin": float(margin.mean().item()),
#         "per_class_mean_cosine_margin": per_class_margin,
#         "mean_feature_rms": float(qrms.mean().item()),
#         "mean_structural_feature_rms": float(qrms.mean().item()),
#     }



# def _rms(value: Tensor) -> float:
#     tensor = torch.as_tensor(value)
#     return float(torch.sqrt(tensor.square().mean()).item())


# @torch.no_grad()
# def geometry_diagnostics(
#     model: Any,
#     class_ids: Sequence[int],
#     *,
#     target_names: Optional[Sequence[str]] = None,
# ) -> Dict[str, Any]:
#     """Summarize retained {I_c,S_c,N_c} class statistics."""
#     ids = _class_ids(class_ids)
#     bank = getattr(model, "geometry_bank", None)
#     if not isinstance(bank, HSIClassDiscriminativeGeometry):
#         raise RuntimeError("model.geometry_bank must be HSIClassDiscriminativeGeometry")
#     bank.validate_state()

#     missing = sorted(set(ids) - set(bank.seen_class_ids))
#     if missing:
#         raise RuntimeError(f"geometry bank lacks requested classes {missing}")

#     names = list(target_names or [])
#     classes: list[Dict[str, Any]] = []
#     for class_id in ids:
#         view = bank.class_geometry(class_id)
#         structure = view.structure
#         k = int(structure.size(0))
#         upper = torch.triu_indices(k, k, device=structure.device)
#         unique_structure = structure[upper[0], upper[1], :]

#         cross = view.cross_region_covariation
#         offdiag_mask = ~torch.eye(k, dtype=torch.bool, device=cross.device)
#         offdiag = cross[offdiag_mask]

#         classes.append(
#             {
#                 "class_id": class_id,
#                 "class_name": (
#                     str(names[class_id])
#                     if class_id < len(names)
#                     else f"Class {class_id + 1}"
#                 ),
#                 "sample_count": int(view.sample_count),
#                 "identity_rms": _rms(view.identity),
#                 "structure_unique_rms": _rms(unique_structure),
#                 "feature_variation_mean": float(view.feature_variation.mean().item()),
#                 "cross_region_offdiagonal_rms": (
#                     _rms(offdiag) if offdiag.numel() else 0.0
#                 ),
#             }
#         )

#     pairs: list[Dict[str, Any]] = []
#     for left_pos, left_id in enumerate(ids):
#         for right_id in ids[left_pos + 1 :]:
#             relation = bank.inter_class_geometry(left_id, right_id)
#             pairs.append(
#                 {
#                     "left_class_id": left_id,
#                     "right_class_id": right_id,
#                     "left_class_name": (
#                         str(names[left_id])
#                         if left_id < len(names)
#                         else f"Class {left_id + 1}"
#                     ),
#                     "right_class_name": (
#                         str(names[right_id])
#                         if right_id < len(names)
#                         else f"Class {right_id + 1}"
#                     ),
#                     "identity_difference_rms": _rms(
#                         relation.identity_difference
#                     ),
#                     "structure_difference_rms": _rms(
#                         relation.structure_difference
#                     ),
#                 }
#             )

#     return {
#         "class_ids": ids,
#         "spectral_region_count": int(bank.spectral_token_count),
#         "feature_dimension": int(bank.spectral_token_dim),
#         "classes": classes,
#         "pairs": pairs,
#     }


# def _mapping_value(
#     mapping: Mapping[str | int, Any],
#     class_id: int,
#     *,
#     field: str,
# ) -> float:
#     value = mapping[class_id] if class_id in mapping else mapping.get(str(class_id))
#     if value is None:
#         raise ValueError(f"{field} lacks class {class_id}")
#     result = float(value)
#     if not math.isfinite(result):
#         raise ValueError(f"{field} for class {class_id} must be finite")
#     return result


# def summarize_class_group(
#     metrics: Mapping[str, Any],
#     class_ids: Sequence[int],
# ) -> Dict[str, Any]:
#     ids = _class_ids(class_ids)
#     accuracy_source = metrics.get("per_class_accuracy")
#     margin_source = metrics.get("per_class_mean_cosine_margin")
#     if not isinstance(accuracy_source, Mapping) or not isinstance(margin_source, Mapping):
#         raise ValueError("metrics lack classifier classwise fields")
#     accuracies = [
#         _mapping_value(accuracy_source, class_id, field="per_class_accuracy")
#         for class_id in ids
#     ]
#     margins = [
#         _mapping_value(margin_source, class_id, field="per_class_mean_cosine_margin")
#         for class_id in ids
#     ]
#     return {
#         "class_ids": ids,
#         "balanced_accuracy": sum(accuracies) / len(accuracies),
#         "minimum_class_accuracy": min(accuracies),
#         "mean_cosine_margin": sum(margins) / len(margins),
#     }


# def old_new_summary(
#     metrics: Mapping[str, Any],
#     *,
#     old_class_ids: Sequence[int],
#     new_class_ids: Sequence[int],
# ) -> Dict[str, Any]:
#     old = summarize_class_group(metrics, old_class_ids)
#     new = summarize_class_group(metrics, new_class_ids)
#     old_accuracy = float(old["balanced_accuracy"])
#     new_accuracy = float(new["balanced_accuracy"])
#     denominator = old_accuracy + new_accuracy
#     harmonic = 0.0 if denominator == 0.0 else 2.0 * old_accuracy * new_accuracy / denominator
#     return {
#         "old": old,
#         "new": new,
#         "harmonic_old_new_accuracy": harmonic,
#     }


# __all__ = [
#     "evaluate_loader",
#     "geometry_diagnostics",
#     "summarize_class_group",
#     "old_new_summary",
# ]


