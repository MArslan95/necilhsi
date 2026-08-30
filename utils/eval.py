from __future__ import annotations

"""Read-only evaluation for the final SCOPE-HSI architecture.

Classification always uses the one persistent classifier W on canonical Q.
Retained {I_c,S_c,N_c} are historical class state used by incremental
classifier training but never form a second prediction head. Evaluation remains
read-only and focuses on cumulative classification, phase-origin confusion,
classwise margins, and compatibility between historical identities and the
persistent classifier directions.

Nothing in this file mutates training state or participates in optimization.
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


# -----------------------------------------------------------------------------
# Validation
# -----------------------------------------------------------------------------


def _as_int(value: object, *, name: str) -> int:
    if torch.is_tensor(value):
        if value.numel() != 1:
            raise ValueError(f"{name} must be an integer scalar")
        if value.dtype == torch.bool or value.is_complex() or torch.is_floating_point(value):
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


def _class_ids(
    values: Sequence[int],
    *,
    name: str = "class_ids",
) -> list[int]:
    ids = [_as_int(v, name=f"{name} value") for v in values]
    if not ids:
        raise ValueError(f"{name} cannot be empty")
    if len(ids) != len(set(ids)):
        raise ValueError(f"{name} must contain unique IDs")
    return ids


@contextmanager
def _temporary_eval_state(model: Any):
    states = [(module, bool(module.training)) for module in model.modules()]
    try:
        model.eval()
        yield
    finally:
        for module, state in states:
            module.training = state


def _strict_labels(
    raw: Any,
    *,
    device: torch.device,
    rows: int,
    name: str,
) -> Tensor:
    labels_raw = torch.as_tensor(raw)
    if labels_raw.dtype == torch.bool or labels_raw.is_complex():
        raise RuntimeError(f"{name} must use an integer dtype")
    if torch.is_floating_point(labels_raw):
        raise RuntimeError(f"{name} must use an integer dtype")

    labels = labels_raw.to(device=device, dtype=torch.long).flatten()
    if labels.shape != (rows,):
        raise RuntimeError(f"{name} must contain one label per sample")
    if bool((labels < 0).any()):
        raise RuntimeError(f"{name} must be non-negative")
    return labels


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

    patch = torch.as_tensor(
        batch["image"],
        device=device,
        dtype=model.dtype,
    )
    spectrum = torch.as_tensor(
        batch["raw_center_spectrum"],
        device=device,
        dtype=model.dtype,
    )

    backbone = model.backbone
    expected_patch = (
        int(backbone.patch_bands),
        int(backbone.patch_size),
        int(backbone.patch_size),
    )
    if (
        patch.ndim != 4
        or patch.size(0) == 0
        or tuple(patch.shape[1:]) != expected_patch
    ):
        raise RuntimeError(
            "evaluation patch shape disagrees with backbone"
        )

    expected_spectrum = (
        patch.size(0),
        int(backbone.spectral_bands),
    )
    if (
        spectrum.ndim != 2
        or tuple(spectrum.shape) != expected_spectrum
    ):
        raise RuntimeError(
            "evaluation spectrum shape disagrees with backbone"
        )

    if (
        not bool(torch.isfinite(patch).all())
        or not bool(torch.isfinite(spectrum).all())
    ):
        raise RuntimeError("evaluation input contains NaN or Inf")

    labels = _strict_labels(
        batch["label"],
        device=device,
        rows=patch.size(0),
        name="evaluation labels",
    )
    return patch, spectrum, labels


def _validate_canonical_q(q: Tensor) -> Tensor:
    value = torch.as_tensor(q)
    if value.ndim != 3 or value.size(0) == 0:
        raise RuntimeError("canonical Q must be non-empty [N,K,D]")
    if not torch.is_floating_point(value) or value.is_complex():
        raise RuntimeError("canonical Q must be real floating point")
    if not bool(torch.isfinite(value).all()):
        raise RuntimeError("canonical Q contains NaN or Inf")

    flat = value.reshape(value.size(0), -1)
    qrms = torch.linalg.vector_norm(flat, ord=2, dim=1) / math.sqrt(
        float(flat.size(1))
    )
    tolerance = 2e-5 if value.dtype == torch.float32 else 1e-10
    if not bool(
        torch.allclose(
            qrms,
            torch.ones_like(qrms),
            rtol=tolerance,
            atol=tolerance,
        )
    ):
        raise RuntimeError(
            "evaluation features are not the model's canonical Q"
        )
    return qrms


# -----------------------------------------------------------------------------
# Classification metrics
# -----------------------------------------------------------------------------


def _confusion(
    targets: np.ndarray,
    predictions: np.ndarray,
    ids: Sequence[int],
) -> np.ndarray:
    if targets.shape != predictions.shape:
        raise ValueError("confusion targets/predictions are not aligned")

    index = {class_id: row for row, class_id in enumerate(ids)}
    matrix = np.zeros((len(ids), len(ids)), dtype=np.int64)
    for target, prediction in zip(targets.tolist(), predictions.tolist()):
        if target not in index or prediction not in index:
            raise RuntimeError(
                "confusion input contains an unknown class ID"
            )
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
    return (
        0.0
        if abs(denominator) <= np.finfo(np.float64).eps
        else (observed - expected) / denominator
    )


@torch.no_grad()
def evaluate_loader(
    model: Any,
    loader: Any,
    *,
    class_ids: Sequence[int],
    target_class_ids: Optional[Sequence[int]] = None,
    device: str | torch.device,
) -> Dict[str, Any]:
    """Evaluate only the persistent classifier on canonical Q.

    ``class_ids`` defines every competing classifier column. ``target_class_ids``
    optionally restricts which ground-truth classes are summarized while rivals
    still include all classes in ``class_ids``.  This is the correct old/new
    evaluation behavior for cumulative NECIL.
    """
    ids = _class_ids(class_ids)
    target_ids = (
        ids
        if target_class_ids is None
        else _class_ids(target_class_ids, name="target_class_ids")
    )
    outside = sorted(set(target_ids) - set(ids))
    if outside:
        raise ValueError(
            f"target_class_ids are outside classifier classes: {outside}"
        )

    evaluation_device = torch.device(device)
    if torch.device(model.device) != evaluation_device:
        raise RuntimeError("model and evaluation device disagree")

    model.validate_model_state()
    if not isinstance(model.classifier, PersistentCosineClassifier):
        raise RuntimeError(
            "model classifier must be PersistentCosineClassifier"
        )

    missing_classifier = sorted(
        set(ids) - set(model.classifier_class_ids)
    )
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
    class_tensor: Optional[Tensor] = None

    allowed_ids = torch.tensor(
        ids,
        device=evaluation_device,
        dtype=torch.long,
    )

    with _temporary_eval_state(model):
        for batch in loader:
            patch, spectrum, labels = _unpack_eval_batch(
                batch,
                model=model,
                device=evaluation_device,
            )

            if not bool(torch.isin(labels, allowed_ids).all()):
                raise RuntimeError(
                    "evaluation loader contains a class outside class_ids"
                )

            output = model.extract_features(
                patch,
                center_spectrum=spectrum,
                return_aux=False,
            )
            q = output.ordered_features
            qrms = _validate_canonical_q(q)

            classification = model.classify_features(
                q,
                class_ids=ids,
            )

            if class_tensor is None:
                class_tensor = classification.class_ids.detach().clone()
            elif not bool(
                torch.equal(class_tensor, classification.class_ids)
            ):
                raise RuntimeError(
                    "classifier class-column order changed across batches"
                )

            # Check the deployed classifier contract, not merely its shapes.
            logits = classification.logits
            cosine = classification.cosine_similarity
            if logits.shape != cosine.shape:
                raise RuntimeError(
                    "classifier logits/cosine shapes disagree"
                )
            if not bool(torch.isfinite(logits).all()) or not bool(
                torch.isfinite(cosine).all()
            ):
                raise RuntimeError(
                    "classifier produced NaN or Inf"
                )
            local_from_logits = logits.argmax(dim=1)
            local_from_cosine = cosine.argmax(dim=1)
            if not bool(torch.equal(local_from_logits, local_from_cosine)):
                raise RuntimeError(
                    "classifier logits and cosine imply different predictions"
                )
            expected_prediction = classification.class_ids.index_select(
                0,
                local_from_logits,
            )
            if not bool(
                torch.equal(expected_prediction, classification.prediction)
            ):
                raise RuntimeError(
                    "classifier prediction violates deployed argmax rule"
                )

            matches = labels.unsqueeze(1).eq(
                classification.class_ids.unsqueeze(0)
            )
            counts = matches.sum(dim=1)
            if not bool(counts.eq(1).all()):
                raise RuntimeError(
                    "could not map evaluation labels to classifier columns"
                )

            targets_local = matches.to(torch.long).argmax(dim=1)
            rows = torch.arange(
                labels.numel(),
                device=evaluation_device,
            )
            target_cosine = cosine[rows, targets_local]
            rival = cosine.clone()
            rival[rows, targets_local] = -torch.inf
            nearest_rival = rival.max(dim=1).values

            predictions.append(classification.prediction.detach())
            labels_rows.append(labels.detach())
            target_cosine_rows.append(target_cosine.detach())
            rival_cosine_rows.append(nearest_rival.detach())
            margin_rows.append(
                (target_cosine - nearest_rival).detach()
            )
            qrms_rows.append(qrms.detach())

    if not predictions or class_tensor is None:
        raise RuntimeError("cannot evaluate an empty loader")

    prediction_all = torch.cat(predictions, dim=0)
    labels_all = torch.cat(labels_rows, dim=0).flatten()
    target_cosine_all = torch.cat(target_cosine_rows, dim=0)
    rival_cosine_all = torch.cat(rival_cosine_rows, dim=0)
    margin_all = torch.cat(margin_rows, dim=0)
    qrms_all = torch.cat(qrms_rows, dim=0)

    target_mask = torch.isin(
        labels_all,
        torch.tensor(
            target_ids,
            device=labels_all.device,
            dtype=torch.long,
        ),
    )
    if not bool(target_mask.any()):
        raise RuntimeError(
            "evaluation loader has no requested target classes"
        )

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
            raise RuntimeError(
                f"evaluation target class {class_id} is absent"
            )
        class_counts[class_id] = count
        per_class_accuracy[class_id] = float(
            prediction[mask]
            .eq(labels[mask])
            .to(torch.float32)
            .mean()
            .item()
        )
        per_class_margin[class_id] = float(
            margin[mask].mean().item()
        )

    # Keep all seen prediction columns.  For an old-only target evaluation,
    # old -> new errors therefore remain visible rather than being discarded.
    confusion = _confusion(
        labels.detach().cpu().numpy(),
        prediction.detach().cpu().numpy(),
        ids,
    )

    oa = float(
        prediction.eq(labels).to(torch.float32).mean().item()
    )
    ba = float(
        sum(per_class_accuracy.values()) / len(per_class_accuracy)
    )

    return {
        "class_ids": list(ids),
        "target_class_ids": list(target_ids),
        "overall_accuracy": oa,
        "accuracy": oa,
        "balanced_accuracy": ba,
        "minimum_class_accuracy": float(
            min(per_class_accuracy.values())
        ),
        "kappa": float(_kappa(confusion)),
        "confusion_matrix": confusion.tolist(),
        "class_counts": class_counts,
        "per_class_accuracy": per_class_accuracy,
        "mean_target_cosine": float(target_cosine.mean().item()),
        "mean_nearest_rival_cosine": float(
            rival_cosine.mean().item()
        ),
        "mean_cosine_margin": float(margin.mean().item()),
        "per_class_mean_cosine_margin": per_class_margin,
        "mean_feature_rms": float(qrms.mean().item()),
    }


# -----------------------------------------------------------------------------
# Reporting feature collection
# -----------------------------------------------------------------------------


@torch.no_grad()
def collect_reporting_features(
    model: Any,
    loader: Any,
    *,
    class_ids: Sequence[int],
    device: str | torch.device,
    require_coordinates: bool = True,
) -> Dict[str, Any]:
    """Collect canonical Q for post-training analysis only.

    HSI pixel coordinates are required by default because before/after drift
    claims are meaningful only for identical pixels in identical row order.
    """
    ids = _class_ids(class_ids)
    evaluation_device = torch.device(device)
    if torch.device(model.device) != evaluation_device:
        raise RuntimeError(
            "model and reporting device disagree"
        )

    allowed = torch.tensor(
        ids,
        device=evaluation_device,
        dtype=torch.long,
    )

    feature_rows: list[Tensor] = []
    label_rows: list[Tensor] = []
    coordinate_rows: list[Tensor] = []
    coordinate_presence: Optional[bool] = None

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
            _validate_canonical_q(q)

            feature_rows.append(
                q[mask].detach().cpu()
            )
            label_rows.append(
                labels[mask].detach().cpu()
            )

            has_coordinates = "coord" in batch
            if coordinate_presence is None:
                coordinate_presence = has_coordinates
            elif coordinate_presence != has_coordinates:
                raise RuntimeError(
                    "coord must be present in every reporting batch or none"
                )

            if has_coordinates:
                coords_raw = torch.as_tensor(batch["coord"])
                if (
                    coords_raw.dtype == torch.bool
                    or coords_raw.is_complex()
                    or torch.is_floating_point(coords_raw)
                ):
                    raise RuntimeError(
                        "reporting coord must use an integer dtype"
                    )
                coords = coords_raw.to(torch.long).detach().cpu()
                if coords.shape != (labels.numel(), 2):
                    raise RuntimeError(
                        "reporting coord must have shape [N,2]"
                    )
                coordinate_rows.append(
                    coords[mask.detach().cpu()]
                )

    if not feature_rows:
        raise RuntimeError(
            "reporting loader contains none of the requested classes"
        )

    features = torch.cat(feature_rows, dim=0)
    labels = torch.cat(label_rows, dim=0).flatten()

    observed = sorted(int(v) for v in labels.unique().tolist())
    if observed != sorted(ids):
        raise RuntimeError(
            f"reporting features cover classes {observed}, "
            f"expected {sorted(ids)}"
        )

    result: Dict[str, Any] = {
        "ordered_features": features.numpy(),
        "labels": labels.numpy().astype(np.int64, copy=False),
    }

    if coordinate_presence:
        coords = torch.cat(coordinate_rows, dim=0)
        if coords.shape != (labels.numel(), 2):
            raise RuntimeError(
                "reporting coordinates are row-misaligned"
            )
        if torch.unique(coords, dim=0).size(0) != coords.size(0):
            raise RuntimeError(
                "reporting collection contains duplicate HSI coordinates"
            )
        result["coords"] = coords.numpy().astype(
            np.int64,
            copy=False,
        )
    elif require_coordinates:
        raise RuntimeError(
            "reporting drift diagnostics require HSI pixel coordinates"
        )

    return result


# -----------------------------------------------------------------------------
# Retained class-structure / classifier compatibility
# -----------------------------------------------------------------------------


def _rms(value: Tensor) -> float:
    tensor = torch.as_tensor(value)
    if tensor.numel() == 0:
        raise ValueError("cannot compute RMS of an empty tensor")
    if not bool(torch.isfinite(tensor).all()):
        raise ValueError("RMS input contains NaN or Inf")
    return float(
        torch.sqrt(tensor.to(torch.float64).square().mean()).item()
    )


@torch.no_grad()
def class_structure_diagnostics(
    model: Any,
    class_ids: Sequence[int],
    *,
    target_names: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """Summarize retained {I_c,S_c,N_c}; never classify from them."""
    ids = _class_ids(class_ids)
    bank = getattr(model, "geometry_bank", None)
    if not isinstance(bank, HSIClassDiscriminativeGeometry):
        raise RuntimeError(
            "model.geometry_bank must be HSIClassDiscriminativeGeometry"
        )

    bank.validate_state()
    missing = sorted(set(ids) - set(bank.seen_class_ids))
    if missing:
        raise RuntimeError(
            f"class-structure bank lacks requested classes {missing}"
        )

    names = list(target_names or [])
    classes: list[Dict[str, Any]] = []

    for class_id in ids:
        view = bank.class_geometry(class_id)
        structure = view.structure
        k = int(structure.size(0))
        upper = torch.triu_indices(
            k,
            k,
            device=structure.device,
        )
        unique_structure = structure[
            upper[0],
            upper[1],
            :,
        ]

        cross = view.cross_region_covariation
        offdiag_mask = ~torch.eye(
            k,
            dtype=torch.bool,
            device=cross.device,
        )
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
                "feature_variation_mean": float(
                    view.feature_variation.mean().item()
                ),
                "cross_region_offdiagonal_rms": (
                    _rms(offdiag)
                    if offdiag.numel()
                    else 0.0
                ),
            }
        )

    pairs: list[Dict[str, Any]] = []
    for left_pos, left_id in enumerate(ids):
        for right_id in ids[left_pos + 1:]:
            relation = bank.inter_class_geometry(
                left_id,
                right_id,
            )
            pairs.append(
                {
                    "left_class_id": left_id,
                    "right_class_id": right_id,
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


# Keep old reporting function name as an alias only.
geometry_diagnostics = class_structure_diagnostics


@torch.no_grad()
def decision_state_diagnostics(
    model: Any,
    class_ids: Sequence[int],
) -> Dict[str, Any]:
    """Measure I_c <-> W compatibility in the same canonical Q coordinate.

    For each retained class identity I_c, compute its response to every
    persistent normalized classifier row W_j.  This does not replace sample
    evaluation; it checks that the retained structural state and actual decision
    state remain mutually consistent after phase transitions.
    """
    ids = _class_ids(class_ids)
    if not isinstance(
        getattr(model, "classifier", None),
        PersistentCosineClassifier,
    ):
        raise RuntimeError(
            "model classifier must be PersistentCosineClassifier"
        )
    if not isinstance(
        getattr(model, "geometry_bank", None),
        HSIClassDiscriminativeGeometry,
    ):
        raise RuntimeError(
            "model class-structure bank is invalid"
        )

    model.validate_model_state()

    batch = model.geometry_bank.class_geometries(ids)
    rows = model.classifier.row_indices(ids)
    weights = model.classifier.weight.detach().index_select(0, rows)

    flat_identity = batch.identity.reshape(len(ids), -1)
    weight_norm = torch.linalg.vector_norm(
        weights,
        dim=1,
        keepdim=True,
    )
    floor = torch.as_tensor(
        torch.finfo(weights.dtype).tiny,
        device=weights.device,
        dtype=weights.dtype,
    )
    directions = weights / weight_norm.clamp_min(floor)
    response = flat_identity @ directions.transpose(0, 1)

    diagonal = response.diagonal()
    competitor = response.clone()
    index = torch.arange(len(ids), device=response.device)
    competitor[index, index] = -torch.inf
    nearest = competitor.max(dim=1).values
    margin = diagonal - nearest

    return {
        "class_ids": ids,
        "identity_classifier_response_matrix": (
            response.detach().cpu().tolist()
        ),
        "self_response": {
            class_id: float(diagonal[pos].item())
            for pos, class_id in enumerate(ids)
        },
        "nearest_rival_response": {
            class_id: float(nearest[pos].item())
            for pos, class_id in enumerate(ids)
        },
        "identity_decision_margin": {
            class_id: float(margin[pos].item())
            for pos, class_id in enumerate(ids)
        },
        "mean_identity_decision_margin": float(
            margin.mean().item()
        ),
    }


# -----------------------------------------------------------------------------
# Old/new summaries
# -----------------------------------------------------------------------------


def _mapping_value(
    mapping: Mapping[str | int, Any],
    class_id: int,
    *,
    field: str,
) -> float:
    value = (
        mapping[class_id]
        if class_id in mapping
        else mapping.get(str(class_id))
    )
    if value is None:
        raise ValueError(
            f"{field} lacks class {class_id}"
        )
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(
            f"{field} for class {class_id} must be finite"
        )
    return result


def summarize_class_group(
    metrics: Mapping[str, Any],
    class_ids: Sequence[int],
) -> Dict[str, Any]:
    ids = _class_ids(class_ids)

    accuracy_source = metrics.get("per_class_accuracy")
    margin_source = metrics.get(
        "per_class_mean_cosine_margin"
    )
    if (
        not isinstance(accuracy_source, Mapping)
        or not isinstance(margin_source, Mapping)
    ):
        raise ValueError(
            "metrics lack classifier classwise fields"
        )

    accuracies = [
        _mapping_value(
            accuracy_source,
            class_id,
            field="per_class_accuracy",
        )
        for class_id in ids
    ]
    margins = [
        _mapping_value(
            margin_source,
            class_id,
            field="per_class_mean_cosine_margin",
        )
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
    old = summarize_class_group(
        metrics,
        old_class_ids,
    )
    new = summarize_class_group(
        metrics,
        new_class_ids,
    )

    old_accuracy = float(old["balanced_accuracy"])
    new_accuracy = float(new["balanced_accuracy"])
    denominator = old_accuracy + new_accuracy
    harmonic = (
        0.0
        if denominator == 0.0
        else (
            2.0
            * old_accuracy
            * new_accuracy
            / denominator
        )
    )

    return {
        "old": old,
        "new": new,
        "harmonic_old_new_accuracy": harmonic,
    }




def old_new_confusion_diagnostics(
    metrics: Mapping[str, Any],
    *,
    old_class_ids: Sequence[int],
    new_class_ids: Sequence[int],
) -> Dict[str, Any]:
    """Summarize directional old/new classifier bias from cumulative confusion.

    Rates are conditional on the true group, so class-count imbalance between
    old and new groups cannot hide directional bias.
    """
    old_ids = _class_ids(old_class_ids, name="old_class_ids")
    new_ids = _class_ids(new_class_ids, name="new_class_ids")
    if set(old_ids).intersection(new_ids):
        raise ValueError("old_class_ids and new_class_ids must be disjoint")

    ids = _class_ids(metrics.get("class_ids", []), name="metrics class_ids")
    if ids != old_ids + new_ids:
        raise ValueError("metrics class order must equal old classes followed by new classes")
    matrix = np.asarray(metrics.get("confusion_matrix"), dtype=np.int64)
    if matrix.shape != (len(ids), len(ids)) or np.any(matrix < 0):
        raise ValueError("metrics confusion_matrix is invalid")

    old_n = len(old_ids)
    old_true = int(matrix[:old_n, :].sum())
    new_true = int(matrix[old_n:, :].sum())
    if old_true <= 0 or new_true <= 0:
        raise ValueError("old/new confusion diagnostics require both groups")

    old_to_new = int(matrix[:old_n, old_n:].sum())
    new_to_old = int(matrix[old_n:, :old_n].sum())
    old_within_error = int(matrix[:old_n, :old_n].sum() - np.trace(matrix[:old_n, :old_n]))
    new_within_error = int(matrix[old_n:, old_n:].sum() - np.trace(matrix[old_n:, old_n:]))

    return {
        "old_true_count": old_true,
        "new_true_count": new_true,
        "old_to_new_count": old_to_new,
        "new_to_old_count": new_to_old,
        "old_to_new_rate": old_to_new / float(old_true),
        "new_to_old_rate": new_to_old / float(new_true),
        "old_within_old_error_count": old_within_error,
        "new_within_new_error_count": new_within_error,
        "old_within_old_error_rate": old_within_error / float(old_true),
        "new_within_new_error_rate": new_within_error / float(new_true),
        "group_confusion_matrix": [
            [old_true - old_to_new, old_to_new],
            [new_to_old, new_true - new_to_old],
        ],
    }



def phase_origin_confusion_diagnostics(
    metrics: Mapping[str, Any],
    *,
    phase_groups: Mapping[int, Sequence[int]],
) -> Dict[str, Any]:
    """Aggregate cumulative class confusion by the phase that introduced a class.

    This avoids the ambiguity of a binary old/new report after phase 2, where
    P0->P1 errors are otherwise counted as old->old.  The function is reporting
    only and never participates in optimization.
    """
    ids = _class_ids(metrics.get("class_ids", []), name="metrics class_ids")
    matrix = np.asarray(metrics.get("confusion_matrix"), dtype=np.int64)
    if matrix.shape != (len(ids), len(ids)) or np.any(matrix < 0):
        raise ValueError("metrics confusion_matrix is invalid")

    if not isinstance(phase_groups, Mapping) or not phase_groups:
        raise ValueError("phase_groups must be a non-empty mapping")

    parsed_groups: Dict[int, list[int]] = {}
    seen: set[int] = set()
    for raw_phase, raw_ids in phase_groups.items():
        phase_id = _as_int(raw_phase, name="phase_groups phase")
        group_ids = _class_ids(
            raw_ids,
            name=f"phase_{phase_id}_class_ids",
        )
        overlap = seen.intersection(group_ids)
        if overlap:
            raise ValueError(
                f"phase_groups contain duplicate classes {sorted(overlap)}"
            )
        seen.update(group_ids)
        parsed_groups[phase_id] = group_ids

    if seen != set(ids):
        missing = sorted(set(ids) - seen)
        extra = sorted(seen - set(ids))
        raise ValueError(
            "phase_groups must partition metrics class_ids exactly; "
            f"missing={missing}, extra={extra}"
        )

    phases = sorted(parsed_groups)
    index = {class_id: position for position, class_id in enumerate(ids)}
    group_matrix = np.zeros(
        (len(phases), len(phases)),
        dtype=np.int64,
    )

    for true_position, true_phase in enumerate(phases):
        true_rows = [index[c] for c in parsed_groups[true_phase]]
        for pred_position, pred_phase in enumerate(phases):
            pred_columns = [index[c] for c in parsed_groups[pred_phase]]
            group_matrix[true_position, pred_position] = int(
                matrix[np.ix_(true_rows, pred_columns)].sum()
            )

    row_totals = group_matrix.sum(axis=1)
    if np.any(row_totals <= 0):
        raise ValueError(
            "phase-origin confusion requires samples for every true phase"
        )
    rates = (
        group_matrix.astype(np.float64)
        / row_totals[:, None].astype(np.float64)
    )

    return {
        "phase_ids": phases,
        "phase_to_class_ids": {
            int(phase): list(parsed_groups[phase])
            for phase in phases
        },
        "count_matrix": group_matrix.tolist(),
        "rate_matrix": rates.tolist(),
        "true_phase_counts": {
            int(phase): int(row_totals[position])
            for position, phase in enumerate(phases)
        },
    }


@torch.no_grad()
def reference_anchored_spectral_region_diagnostics(
    model: Any,
    *,
    historical_class_ids: Sequence[int],
    new_class_ids: Sequence[int],
    seen_class_ids: Sequence[int],
    reference_old_old_score: Any,
    target_names: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """Report reference retention and old-new spectral-region exclusion.

    The report uses the exact boundary quantities that supervise the classifier:

        score_cj = <I_c,U_c-U_j> / sigma_bar_cj

    when retained spread is non-zero, with the model's deterministic zero-spread
    branch otherwise.  Old-old current scores are compared with the phase-start
    reference ``max(r_ref,0)``.  Old-new scores are reported directly because
    those boundaries did not exist before the current phase.
    """
    old_ids = _class_ids(historical_class_ids, name="historical_class_ids")
    new_ids = _class_ids(new_class_ids, name="new_class_ids")
    seen_ids = _class_ids(seen_class_ids, name="seen_class_ids")
    if set(old_ids).intersection(new_ids):
        raise ValueError("historical_class_ids and new_class_ids must be disjoint")
    if set(seen_ids) != set(old_ids).union(new_ids):
        raise ValueError("seen classes must equal historical plus new classes")

    safety = model.boundary_safety_scores(
        historical_class_ids=old_ids,
        rival_class_ids=seen_ids,
    )
    if tuple(safety.historical_class_ids) != tuple(old_ids):
        raise RuntimeError("historical boundary class order changed")
    if tuple(safety.rival_class_ids) != tuple(seen_ids):
        raise RuntimeError("rival boundary class order changed")

    gap = safety.mean_gap.detach()
    dispersion = safety.dispersion_upper_bound.detach()
    score = safety.score.detach()
    expected = (len(old_ids), len(seen_ids))
    for name, value in (
        ("mean_gap", gap),
        ("dispersion_upper_bound", dispersion),
        ("spectral_decision_score", score),
    ):
        if tuple(value.shape) != expected:
            raise RuntimeError(f"{name} has shape {tuple(value.shape)}, expected {expected}")
        if not bool(torch.isfinite(value).all()):
            raise RuntimeError(f"{name} contains NaN or Inf")

    reference = torch.as_tensor(
        reference_old_old_score,
        device=score.device,
        dtype=score.dtype,
    )
    if reference.shape != (len(old_ids), len(old_ids)):
        raise ValueError("reference_old_old_score has invalid shape")
    if not bool(torch.isfinite(reference).all()):
        raise ValueError("reference_old_old_score contains NaN or Inf")

    seen_position = {class_id: pos for pos, class_id in enumerate(seen_ids)}
    old_columns = torch.tensor(
        [seen_position[class_id] for class_id in old_ids],
        device=score.device,
        dtype=torch.long,
    )
    new_columns = torch.tensor(
        [seen_position[class_id] for class_id in new_ids],
        device=score.device,
        dtype=torch.long,
    )
    current_old_old = score.index_select(1, old_columns)
    current_old_new = score.index_select(1, new_columns)
    target_old_old = reference.clamp_min(0.0)
    self_mask = torch.eye(
        len(old_ids), device=score.device, dtype=torch.bool
    )
    degradation = (target_old_old - current_old_old).clamp_min(0.0)
    degradation = degradation.masked_fill(self_mask, 0.0)

    valid_old_old = ~self_mask
    old_old_reference_values = reference[valid_old_old]
    old_old_current_values = current_old_old[valid_old_old]
    old_old_degradation_values = degradation[valid_old_old]

    names = list(target_names or [])
    per_class: list[Dict[str, Any]] = []
    all_rival_scores: list[Tensor] = []
    for old_pos, class_id in enumerate(old_ids):
        old_rival_positions = [p for p in range(len(old_ids)) if p != old_pos]
        oo_deg = degradation[old_pos, old_rival_positions]
        oo_current = current_old_old[old_pos, old_rival_positions]
        oo_reference = reference[old_pos, old_rival_positions]
        if oo_deg.numel() == 0:
            raise RuntimeError(f"historical class {class_id} has no old rival")
        worst_oo_local = int(oo_deg.argmax().item())
        worst_oo_pos = old_rival_positions[worst_oo_local]
        worst_oo_id = old_ids[worst_oo_pos]

        on_scores = current_old_new[old_pos]
        if on_scores.numel() == 0:
            raise RuntimeError(f"historical class {class_id} has no new rival")
        worst_on_local = int(on_scores.argmin().item())
        worst_on_id = new_ids[worst_on_local]

        # Preserve legacy all-seen worst-rival fields for classwise CSVs.
        self_seen_pos = seen_position[class_id]
        rival_mask = torch.ones(len(seen_ids), dtype=torch.bool, device=score.device)
        rival_mask[self_seen_pos] = False
        rival_scores = score[old_pos][rival_mask]
        rival_gaps = gap[old_pos][rival_mask]
        rival_dispersion = dispersion[old_pos][rival_mask]
        rival_ids = [seen_ids[i] for i in range(len(seen_ids)) if i != self_seen_pos]
        worst_all_local = int(rival_scores.argmin().item())
        worst_all_id = int(rival_ids[worst_all_local])
        all_rival_scores.append(rival_scores)

        per_class.append(
            {
                "class_id": int(class_id),
                "class_name": (
                    str(names[class_id])
                    if 0 <= class_id < len(names)
                    else f"Class {class_id + 1}"
                ),
                "worst_rival_class_id": worst_all_id,
                "worst_rival_class_name": (
                    str(names[worst_all_id])
                    if 0 <= worst_all_id < len(names)
                    else f"Class {worst_all_id + 1}"
                ),
                "worst_mean_gap": float(rival_gaps[worst_all_local].item()),
                "worst_dispersion_upper_bound": float(
                    rival_dispersion[worst_all_local].item()
                ),
                "minimum_spectral_decision_score": float(
                    rival_scores[worst_all_local].item()
                ),
                "mean_spectral_decision_score": float(rival_scores.mean().item()),
                "worst_old_old_rival_class_id": int(worst_oo_id),
                "worst_old_old_reference_score": float(
                    oo_reference[worst_oo_local].item()
                ),
                "worst_old_old_current_score": float(
                    oo_current[worst_oo_local].item()
                ),
                "worst_old_old_degradation": float(
                    oo_deg[worst_oo_local].item()
                ),
                "worst_old_new_rival_class_id": int(worst_on_id),
                "worst_old_new_score": float(on_scores[worst_on_local].item()),
                "old_new_boundary_violated": bool(
                    on_scores[worst_on_local].item() <= 0.0
                ),
            }
        )

    flattened = torch.cat(all_rival_scores, dim=0)
    old_new_violation = current_old_new <= 0
    return {
        "historical_class_ids": list(old_ids),
        "new_class_ids": list(new_ids),
        "seen_class_ids": list(seen_ids),
        "mean_spectral_decision_score": float(flattened.mean().item()),
        "minimum_spectral_decision_score": float(flattened.min().item()),
        "old_old_reference_min_score": float(old_old_reference_values.min().item()),
        "old_old_current_min_score": float(old_old_current_values.min().item()),
        "old_old_max_degradation": float(old_old_degradation_values.max().item()),
        "old_old_violation_count": int((old_old_degradation_values > 0).sum().item()),
        "old_old_pair_count": int(old_old_degradation_values.numel()),
        "old_new_min_score": float(current_old_new.min().item()),
        "old_new_violation_count": int(old_new_violation.sum().item()),
        "old_new_pair_count": int(old_new_violation.numel()),
        "per_class": per_class,
    }


__all__ = [
    "evaluate_loader",
    "collect_reporting_features",
    "class_structure_diagnostics",
    "geometry_diagnostics",
    "decision_state_diagnostics",
    "summarize_class_group",
    "old_new_summary",
    "old_new_confusion_diagnostics",
    "phase_origin_confusion_diagnostics",
    "reference_anchored_spectral_region_diagnostics",
]














# from __future__ import annotations

# """Read-only evaluation for the final SCOPE-HSI architecture.

# Classification always uses the one persistent classifier W on canonical Q.
# Retained {I_c,S_c,N_c} are historical class state used by incremental
# classifier training but never form a second prediction head. Evaluation remains
# read-only and focuses on cumulative classification, phase-origin confusion,
# classwise margins, and compatibility between historical identities and the
# persistent classifier directions.

# Nothing in this file mutates training state or participates in optimization.
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


# # -----------------------------------------------------------------------------
# # Validation
# # -----------------------------------------------------------------------------


# def _as_int(value: object, *, name: str) -> int:
#     if torch.is_tensor(value):
#         if value.numel() != 1:
#             raise ValueError(f"{name} must be an integer scalar")
#         if value.dtype == torch.bool or value.is_complex() or torch.is_floating_point(value):
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


# def _class_ids(
#     values: Sequence[int],
#     *,
#     name: str = "class_ids",
# ) -> list[int]:
#     ids = [_as_int(v, name=f"{name} value") for v in values]
#     if not ids:
#         raise ValueError(f"{name} cannot be empty")
#     if len(ids) != len(set(ids)):
#         raise ValueError(f"{name} must contain unique IDs")
#     return ids


# @contextmanager
# def _temporary_eval_state(model: Any):
#     states = [(module, bool(module.training)) for module in model.modules()]
#     try:
#         model.eval()
#         yield
#     finally:
#         for module, state in states:
#             module.training = state


# def _strict_labels(
#     raw: Any,
#     *,
#     device: torch.device,
#     rows: int,
#     name: str,
# ) -> Tensor:
#     labels_raw = torch.as_tensor(raw)
#     if labels_raw.dtype == torch.bool or labels_raw.is_complex():
#         raise RuntimeError(f"{name} must use an integer dtype")
#     if torch.is_floating_point(labels_raw):
#         raise RuntimeError(f"{name} must use an integer dtype")

#     labels = labels_raw.to(device=device, dtype=torch.long).flatten()
#     if labels.shape != (rows,):
#         raise RuntimeError(f"{name} must contain one label per sample")
#     if bool((labels < 0).any()):
#         raise RuntimeError(f"{name} must be non-negative")
#     return labels


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

#     patch = torch.as_tensor(
#         batch["image"],
#         device=device,
#         dtype=model.dtype,
#     )
#     spectrum = torch.as_tensor(
#         batch["raw_center_spectrum"],
#         device=device,
#         dtype=model.dtype,
#     )

#     backbone = model.backbone
#     expected_patch = (
#         int(backbone.patch_bands),
#         int(backbone.patch_size),
#         int(backbone.patch_size),
#     )
#     if (
#         patch.ndim != 4
#         or patch.size(0) == 0
#         or tuple(patch.shape[1:]) != expected_patch
#     ):
#         raise RuntimeError(
#             "evaluation patch shape disagrees with backbone"
#         )

#     expected_spectrum = (
#         patch.size(0),
#         int(backbone.spectral_bands),
#     )
#     if (
#         spectrum.ndim != 2
#         or tuple(spectrum.shape) != expected_spectrum
#     ):
#         raise RuntimeError(
#             "evaluation spectrum shape disagrees with backbone"
#         )

#     if (
#         not bool(torch.isfinite(patch).all())
#         or not bool(torch.isfinite(spectrum).all())
#     ):
#         raise RuntimeError("evaluation input contains NaN or Inf")

#     labels = _strict_labels(
#         batch["label"],
#         device=device,
#         rows=patch.size(0),
#         name="evaluation labels",
#     )
#     return patch, spectrum, labels


# def _validate_canonical_q(q: Tensor) -> Tensor:
#     value = torch.as_tensor(q)
#     if value.ndim != 3 or value.size(0) == 0:
#         raise RuntimeError("canonical Q must be non-empty [N,K,D]")
#     if not torch.is_floating_point(value) or value.is_complex():
#         raise RuntimeError("canonical Q must be real floating point")
#     if not bool(torch.isfinite(value).all()):
#         raise RuntimeError("canonical Q contains NaN or Inf")

#     flat = value.reshape(value.size(0), -1)
#     qrms = torch.linalg.vector_norm(flat, ord=2, dim=1) / math.sqrt(
#         float(flat.size(1))
#     )
#     tolerance = 2e-5 if value.dtype == torch.float32 else 1e-10
#     if not bool(
#         torch.allclose(
#             qrms,
#             torch.ones_like(qrms),
#             rtol=tolerance,
#             atol=tolerance,
#         )
#     ):
#         raise RuntimeError(
#             "evaluation features are not the model's canonical Q"
#         )
#     return qrms


# # -----------------------------------------------------------------------------
# # Classification metrics
# # -----------------------------------------------------------------------------


# def _confusion(
#     targets: np.ndarray,
#     predictions: np.ndarray,
#     ids: Sequence[int],
# ) -> np.ndarray:
#     if targets.shape != predictions.shape:
#         raise ValueError("confusion targets/predictions are not aligned")

#     index = {class_id: row for row, class_id in enumerate(ids)}
#     matrix = np.zeros((len(ids), len(ids)), dtype=np.int64)
#     for target, prediction in zip(targets.tolist(), predictions.tolist()):
#         if target not in index or prediction not in index:
#             raise RuntimeError(
#                 "confusion input contains an unknown class ID"
#             )
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
#     return (
#         0.0
#         if abs(denominator) <= np.finfo(np.float64).eps
#         else (observed - expected) / denominator
#     )


# @torch.no_grad()
# def evaluate_loader(
#     model: Any,
#     loader: Any,
#     *,
#     class_ids: Sequence[int],
#     target_class_ids: Optional[Sequence[int]] = None,
#     device: str | torch.device,
# ) -> Dict[str, Any]:
#     """Evaluate only the persistent classifier on canonical Q.

#     ``class_ids`` defines every competing classifier column. ``target_class_ids``
#     optionally restricts which ground-truth classes are summarized while rivals
#     still include all classes in ``class_ids``.  This is the correct old/new
#     evaluation behavior for cumulative NECIL.
#     """
#     ids = _class_ids(class_ids)
#     target_ids = (
#         ids
#         if target_class_ids is None
#         else _class_ids(target_class_ids, name="target_class_ids")
#     )
#     outside = sorted(set(target_ids) - set(ids))
#     if outside:
#         raise ValueError(
#             f"target_class_ids are outside classifier classes: {outside}"
#         )

#     evaluation_device = torch.device(device)
#     if torch.device(model.device) != evaluation_device:
#         raise RuntimeError("model and evaluation device disagree")

#     model.validate_model_state()
#     if not isinstance(model.classifier, PersistentCosineClassifier):
#         raise RuntimeError(
#             "model classifier must be PersistentCosineClassifier"
#         )

#     missing_classifier = sorted(
#         set(ids) - set(model.classifier_class_ids)
#     )
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
#     class_tensor: Optional[Tensor] = None

#     allowed_ids = torch.tensor(
#         ids,
#         device=evaluation_device,
#         dtype=torch.long,
#     )

#     with _temporary_eval_state(model):
#         for batch in loader:
#             patch, spectrum, labels = _unpack_eval_batch(
#                 batch,
#                 model=model,
#                 device=evaluation_device,
#             )

#             if not bool(torch.isin(labels, allowed_ids).all()):
#                 raise RuntimeError(
#                     "evaluation loader contains a class outside class_ids"
#                 )

#             output = model.extract_features(
#                 patch,
#                 center_spectrum=spectrum,
#                 return_aux=False,
#             )
#             q = output.ordered_features
#             qrms = _validate_canonical_q(q)

#             classification = model.classify_features(
#                 q,
#                 class_ids=ids,
#             )

#             if class_tensor is None:
#                 class_tensor = classification.class_ids.detach().clone()
#             elif not bool(
#                 torch.equal(class_tensor, classification.class_ids)
#             ):
#                 raise RuntimeError(
#                     "classifier class-column order changed across batches"
#                 )

#             # Check the deployed classifier contract, not merely its shapes.
#             logits = classification.logits
#             cosine = classification.cosine_similarity
#             if logits.shape != cosine.shape:
#                 raise RuntimeError(
#                     "classifier logits/cosine shapes disagree"
#                 )
#             if not bool(torch.isfinite(logits).all()) or not bool(
#                 torch.isfinite(cosine).all()
#             ):
#                 raise RuntimeError(
#                     "classifier produced NaN or Inf"
#                 )
#             local_from_logits = logits.argmax(dim=1)
#             local_from_cosine = cosine.argmax(dim=1)
#             if not bool(torch.equal(local_from_logits, local_from_cosine)):
#                 raise RuntimeError(
#                     "classifier logits and cosine imply different predictions"
#                 )
#             expected_prediction = classification.class_ids.index_select(
#                 0,
#                 local_from_logits,
#             )
#             if not bool(
#                 torch.equal(expected_prediction, classification.prediction)
#             ):
#                 raise RuntimeError(
#                     "classifier prediction violates deployed argmax rule"
#                 )

#             matches = labels.unsqueeze(1).eq(
#                 classification.class_ids.unsqueeze(0)
#             )
#             counts = matches.sum(dim=1)
#             if not bool(counts.eq(1).all()):
#                 raise RuntimeError(
#                     "could not map evaluation labels to classifier columns"
#                 )

#             targets_local = matches.to(torch.long).argmax(dim=1)
#             rows = torch.arange(
#                 labels.numel(),
#                 device=evaluation_device,
#             )
#             target_cosine = cosine[rows, targets_local]
#             rival = cosine.clone()
#             rival[rows, targets_local] = -torch.inf
#             nearest_rival = rival.max(dim=1).values

#             predictions.append(classification.prediction.detach())
#             labels_rows.append(labels.detach())
#             target_cosine_rows.append(target_cosine.detach())
#             rival_cosine_rows.append(nearest_rival.detach())
#             margin_rows.append(
#                 (target_cosine - nearest_rival).detach()
#             )
#             qrms_rows.append(qrms.detach())

#     if not predictions or class_tensor is None:
#         raise RuntimeError("cannot evaluate an empty loader")

#     prediction_all = torch.cat(predictions, dim=0)
#     labels_all = torch.cat(labels_rows, dim=0).flatten()
#     target_cosine_all = torch.cat(target_cosine_rows, dim=0)
#     rival_cosine_all = torch.cat(rival_cosine_rows, dim=0)
#     margin_all = torch.cat(margin_rows, dim=0)
#     qrms_all = torch.cat(qrms_rows, dim=0)

#     target_mask = torch.isin(
#         labels_all,
#         torch.tensor(
#             target_ids,
#             device=labels_all.device,
#             dtype=torch.long,
#         ),
#     )
#     if not bool(target_mask.any()):
#         raise RuntimeError(
#             "evaluation loader has no requested target classes"
#         )

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
#             raise RuntimeError(
#                 f"evaluation target class {class_id} is absent"
#             )
#         class_counts[class_id] = count
#         per_class_accuracy[class_id] = float(
#             prediction[mask]
#             .eq(labels[mask])
#             .to(torch.float32)
#             .mean()
#             .item()
#         )
#         per_class_margin[class_id] = float(
#             margin[mask].mean().item()
#         )

#     # Keep all seen prediction columns.  For an old-only target evaluation,
#     # old -> new errors therefore remain visible rather than being discarded.
#     confusion = _confusion(
#         labels.detach().cpu().numpy(),
#         prediction.detach().cpu().numpy(),
#         ids,
#     )

#     oa = float(
#         prediction.eq(labels).to(torch.float32).mean().item()
#     )
#     ba = float(
#         sum(per_class_accuracy.values()) / len(per_class_accuracy)
#     )

#     return {
#         "class_ids": list(ids),
#         "target_class_ids": list(target_ids),
#         "overall_accuracy": oa,
#         "accuracy": oa,
#         "balanced_accuracy": ba,
#         "minimum_class_accuracy": float(
#             min(per_class_accuracy.values())
#         ),
#         "kappa": float(_kappa(confusion)),
#         "confusion_matrix": confusion.tolist(),
#         "class_counts": class_counts,
#         "per_class_accuracy": per_class_accuracy,
#         "mean_target_cosine": float(target_cosine.mean().item()),
#         "mean_nearest_rival_cosine": float(
#             rival_cosine.mean().item()
#         ),
#         "mean_cosine_margin": float(margin.mean().item()),
#         "per_class_mean_cosine_margin": per_class_margin,
#         "mean_feature_rms": float(qrms.mean().item()),
#     }


# # -----------------------------------------------------------------------------
# # Reporting feature collection
# # -----------------------------------------------------------------------------


# @torch.no_grad()
# def collect_reporting_features(
#     model: Any,
#     loader: Any,
#     *,
#     class_ids: Sequence[int],
#     device: str | torch.device,
#     require_coordinates: bool = True,
# ) -> Dict[str, Any]:
#     """Collect canonical Q for post-training analysis only.

#     HSI pixel coordinates are required by default because before/after drift
#     claims are meaningful only for identical pixels in identical row order.
#     """
#     ids = _class_ids(class_ids)
#     evaluation_device = torch.device(device)
#     if torch.device(model.device) != evaluation_device:
#         raise RuntimeError(
#             "model and reporting device disagree"
#         )

#     allowed = torch.tensor(
#         ids,
#         device=evaluation_device,
#         dtype=torch.long,
#     )

#     feature_rows: list[Tensor] = []
#     label_rows: list[Tensor] = []
#     coordinate_rows: list[Tensor] = []
#     coordinate_presence: Optional[bool] = None

#     with _temporary_eval_state(model):
#         for batch in loader:
#             patch, spectrum, labels = _unpack_eval_batch(
#                 batch,
#                 model=model,
#                 device=evaluation_device,
#             )
#             mask = torch.isin(labels, allowed)
#             if not bool(mask.any()):
#                 continue

#             output = model.extract_features(
#                 patch,
#                 center_spectrum=spectrum,
#                 return_aux=False,
#             )
#             q = output.ordered_features
#             _validate_canonical_q(q)

#             feature_rows.append(
#                 q[mask].detach().cpu()
#             )
#             label_rows.append(
#                 labels[mask].detach().cpu()
#             )

#             has_coordinates = "coord" in batch
#             if coordinate_presence is None:
#                 coordinate_presence = has_coordinates
#             elif coordinate_presence != has_coordinates:
#                 raise RuntimeError(
#                     "coord must be present in every reporting batch or none"
#                 )

#             if has_coordinates:
#                 coords_raw = torch.as_tensor(batch["coord"])
#                 if (
#                     coords_raw.dtype == torch.bool
#                     or coords_raw.is_complex()
#                     or torch.is_floating_point(coords_raw)
#                 ):
#                     raise RuntimeError(
#                         "reporting coord must use an integer dtype"
#                     )
#                 coords = coords_raw.to(torch.long).detach().cpu()
#                 if coords.shape != (labels.numel(), 2):
#                     raise RuntimeError(
#                         "reporting coord must have shape [N,2]"
#                     )
#                 coordinate_rows.append(
#                     coords[mask.detach().cpu()]
#                 )

#     if not feature_rows:
#         raise RuntimeError(
#             "reporting loader contains none of the requested classes"
#         )

#     features = torch.cat(feature_rows, dim=0)
#     labels = torch.cat(label_rows, dim=0).flatten()

#     observed = sorted(int(v) for v in labels.unique().tolist())
#     if observed != sorted(ids):
#         raise RuntimeError(
#             f"reporting features cover classes {observed}, "
#             f"expected {sorted(ids)}"
#         )

#     result: Dict[str, Any] = {
#         "ordered_features": features.numpy(),
#         "labels": labels.numpy().astype(np.int64, copy=False),
#     }

#     if coordinate_presence:
#         coords = torch.cat(coordinate_rows, dim=0)
#         if coords.shape != (labels.numel(), 2):
#             raise RuntimeError(
#                 "reporting coordinates are row-misaligned"
#             )
#         if torch.unique(coords, dim=0).size(0) != coords.size(0):
#             raise RuntimeError(
#                 "reporting collection contains duplicate HSI coordinates"
#             )
#         result["coords"] = coords.numpy().astype(
#             np.int64,
#             copy=False,
#         )
#     elif require_coordinates:
#         raise RuntimeError(
#             "reporting drift diagnostics require HSI pixel coordinates"
#         )

#     return result


# # -----------------------------------------------------------------------------
# # Retained class-structure / classifier compatibility
# # -----------------------------------------------------------------------------


# def _rms(value: Tensor) -> float:
#     tensor = torch.as_tensor(value)
#     if tensor.numel() == 0:
#         raise ValueError("cannot compute RMS of an empty tensor")
#     if not bool(torch.isfinite(tensor).all()):
#         raise ValueError("RMS input contains NaN or Inf")
#     return float(
#         torch.sqrt(tensor.to(torch.float64).square().mean()).item()
#     )


# @torch.no_grad()
# def class_structure_diagnostics(
#     model: Any,
#     class_ids: Sequence[int],
#     *,
#     target_names: Optional[Sequence[str]] = None,
# ) -> Dict[str, Any]:
#     """Summarize retained {I_c,S_c,N_c}; never classify from them."""
#     ids = _class_ids(class_ids)
#     bank = getattr(model, "geometry_bank", None)
#     if not isinstance(bank, HSIClassDiscriminativeGeometry):
#         raise RuntimeError(
#             "model.geometry_bank must be HSIClassDiscriminativeGeometry"
#         )

#     bank.validate_state()
#     missing = sorted(set(ids) - set(bank.seen_class_ids))
#     if missing:
#         raise RuntimeError(
#             f"class-structure bank lacks requested classes {missing}"
#         )

#     names = list(target_names or [])
#     classes: list[Dict[str, Any]] = []

#     for class_id in ids:
#         view = bank.class_geometry(class_id)
#         structure = view.structure
#         k = int(structure.size(0))
#         upper = torch.triu_indices(
#             k,
#             k,
#             device=structure.device,
#         )
#         unique_structure = structure[
#             upper[0],
#             upper[1],
#             :,
#         ]

#         cross = view.cross_region_covariation
#         offdiag_mask = ~torch.eye(
#             k,
#             dtype=torch.bool,
#             device=cross.device,
#         )
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
#                 "feature_variation_mean": float(
#                     view.feature_variation.mean().item()
#                 ),
#                 "cross_region_offdiagonal_rms": (
#                     _rms(offdiag)
#                     if offdiag.numel()
#                     else 0.0
#                 ),
#             }
#         )

#     pairs: list[Dict[str, Any]] = []
#     for left_pos, left_id in enumerate(ids):
#         for right_id in ids[left_pos + 1:]:
#             relation = bank.inter_class_geometry(
#                 left_id,
#                 right_id,
#             )
#             pairs.append(
#                 {
#                     "left_class_id": left_id,
#                     "right_class_id": right_id,
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


# # Keep old reporting function name as an alias only.
# geometry_diagnostics = class_structure_diagnostics


# @torch.no_grad()
# def decision_state_diagnostics(
#     model: Any,
#     class_ids: Sequence[int],
# ) -> Dict[str, Any]:
#     """Measure I_c <-> W compatibility in the same canonical Q coordinate.

#     For each retained class identity I_c, compute its response to every
#     persistent normalized classifier row W_j.  This does not replace sample
#     evaluation; it checks that the retained structural state and actual decision
#     state remain mutually consistent after phase transitions.
#     """
#     ids = _class_ids(class_ids)
#     if not isinstance(
#         getattr(model, "classifier", None),
#         PersistentCosineClassifier,
#     ):
#         raise RuntimeError(
#             "model classifier must be PersistentCosineClassifier"
#         )
#     if not isinstance(
#         getattr(model, "geometry_bank", None),
#         HSIClassDiscriminativeGeometry,
#     ):
#         raise RuntimeError(
#             "model class-structure bank is invalid"
#         )

#     model.validate_model_state()

#     batch = model.geometry_bank.class_geometries(ids)
#     rows = model.classifier.row_indices(ids)
#     weights = model.classifier.weight.detach().index_select(0, rows)

#     flat_identity = batch.identity.reshape(len(ids), -1)
#     weight_norm = torch.linalg.vector_norm(
#         weights,
#         dim=1,
#         keepdim=True,
#     )
#     floor = torch.as_tensor(
#         torch.finfo(weights.dtype).tiny,
#         device=weights.device,
#         dtype=weights.dtype,
#     )
#     directions = weights / weight_norm.clamp_min(floor)
#     response = flat_identity @ directions.transpose(0, 1)

#     diagonal = response.diagonal()
#     competitor = response.clone()
#     index = torch.arange(len(ids), device=response.device)
#     competitor[index, index] = -torch.inf
#     nearest = competitor.max(dim=1).values
#     margin = diagonal - nearest

#     return {
#         "class_ids": ids,
#         "identity_classifier_response_matrix": (
#             response.detach().cpu().tolist()
#         ),
#         "self_response": {
#             class_id: float(diagonal[pos].item())
#             for pos, class_id in enumerate(ids)
#         },
#         "nearest_rival_response": {
#             class_id: float(nearest[pos].item())
#             for pos, class_id in enumerate(ids)
#         },
#         "identity_decision_margin": {
#             class_id: float(margin[pos].item())
#             for pos, class_id in enumerate(ids)
#         },
#         "mean_identity_decision_margin": float(
#             margin.mean().item()
#         ),
#     }


# # -----------------------------------------------------------------------------
# # Old/new summaries
# # -----------------------------------------------------------------------------


# def _mapping_value(
#     mapping: Mapping[str | int, Any],
#     class_id: int,
#     *,
#     field: str,
# ) -> float:
#     value = (
#         mapping[class_id]
#         if class_id in mapping
#         else mapping.get(str(class_id))
#     )
#     if value is None:
#         raise ValueError(
#             f"{field} lacks class {class_id}"
#         )
#     result = float(value)
#     if not math.isfinite(result):
#         raise ValueError(
#             f"{field} for class {class_id} must be finite"
#         )
#     return result


# def summarize_class_group(
#     metrics: Mapping[str, Any],
#     class_ids: Sequence[int],
# ) -> Dict[str, Any]:
#     ids = _class_ids(class_ids)

#     accuracy_source = metrics.get("per_class_accuracy")
#     margin_source = metrics.get(
#         "per_class_mean_cosine_margin"
#     )
#     if (
#         not isinstance(accuracy_source, Mapping)
#         or not isinstance(margin_source, Mapping)
#     ):
#         raise ValueError(
#             "metrics lack classifier classwise fields"
#         )

#     accuracies = [
#         _mapping_value(
#             accuracy_source,
#             class_id,
#             field="per_class_accuracy",
#         )
#         for class_id in ids
#     ]
#     margins = [
#         _mapping_value(
#             margin_source,
#             class_id,
#             field="per_class_mean_cosine_margin",
#         )
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
#     old = summarize_class_group(
#         metrics,
#         old_class_ids,
#     )
#     new = summarize_class_group(
#         metrics,
#         new_class_ids,
#     )

#     old_accuracy = float(old["balanced_accuracy"])
#     new_accuracy = float(new["balanced_accuracy"])
#     denominator = old_accuracy + new_accuracy
#     harmonic = (
#         0.0
#         if denominator == 0.0
#         else (
#             2.0
#             * old_accuracy
#             * new_accuracy
#             / denominator
#         )
#     )

#     return {
#         "old": old,
#         "new": new,
#         "harmonic_old_new_accuracy": harmonic,
#     }




# def old_new_confusion_diagnostics(
#     metrics: Mapping[str, Any],
#     *,
#     old_class_ids: Sequence[int],
#     new_class_ids: Sequence[int],
# ) -> Dict[str, Any]:
#     """Summarize directional old/new classifier bias from cumulative confusion.

#     Rates are conditional on the true group, so class-count imbalance between
#     old and new groups cannot hide directional bias.
#     """
#     old_ids = _class_ids(old_class_ids, name="old_class_ids")
#     new_ids = _class_ids(new_class_ids, name="new_class_ids")
#     if set(old_ids).intersection(new_ids):
#         raise ValueError("old_class_ids and new_class_ids must be disjoint")

#     ids = _class_ids(metrics.get("class_ids", []), name="metrics class_ids")
#     if ids != old_ids + new_ids:
#         raise ValueError("metrics class order must equal old classes followed by new classes")
#     matrix = np.asarray(metrics.get("confusion_matrix"), dtype=np.int64)
#     if matrix.shape != (len(ids), len(ids)) or np.any(matrix < 0):
#         raise ValueError("metrics confusion_matrix is invalid")

#     old_n = len(old_ids)
#     old_true = int(matrix[:old_n, :].sum())
#     new_true = int(matrix[old_n:, :].sum())
#     if old_true <= 0 or new_true <= 0:
#         raise ValueError("old/new confusion diagnostics require both groups")

#     old_to_new = int(matrix[:old_n, old_n:].sum())
#     new_to_old = int(matrix[old_n:, :old_n].sum())
#     old_within_error = int(matrix[:old_n, :old_n].sum() - np.trace(matrix[:old_n, :old_n]))
#     new_within_error = int(matrix[old_n:, old_n:].sum() - np.trace(matrix[old_n:, old_n:]))

#     return {
#         "old_true_count": old_true,
#         "new_true_count": new_true,
#         "old_to_new_count": old_to_new,
#         "new_to_old_count": new_to_old,
#         "old_to_new_rate": old_to_new / float(old_true),
#         "new_to_old_rate": new_to_old / float(new_true),
#         "old_within_old_error_count": old_within_error,
#         "new_within_new_error_count": new_within_error,
#         "old_within_old_error_rate": old_within_error / float(old_true),
#         "new_within_new_error_rate": new_within_error / float(new_true),
#         "group_confusion_matrix": [
#             [old_true - old_to_new, old_to_new],
#             [new_to_old, new_true - new_to_old],
#         ],
#     }



# def phase_origin_confusion_diagnostics(
#     metrics: Mapping[str, Any],
#     *,
#     phase_groups: Mapping[int, Sequence[int]],
# ) -> Dict[str, Any]:
#     """Aggregate cumulative class confusion by the phase that introduced a class.

#     This avoids the ambiguity of a binary old/new report after phase 2, where
#     P0->P1 errors are otherwise counted as old->old.  The function is reporting
#     only and never participates in optimization.
#     """
#     ids = _class_ids(metrics.get("class_ids", []), name="metrics class_ids")
#     matrix = np.asarray(metrics.get("confusion_matrix"), dtype=np.int64)
#     if matrix.shape != (len(ids), len(ids)) or np.any(matrix < 0):
#         raise ValueError("metrics confusion_matrix is invalid")

#     if not isinstance(phase_groups, Mapping) or not phase_groups:
#         raise ValueError("phase_groups must be a non-empty mapping")

#     parsed_groups: Dict[int, list[int]] = {}
#     seen: set[int] = set()
#     for raw_phase, raw_ids in phase_groups.items():
#         phase_id = _as_int(raw_phase, name="phase_groups phase")
#         group_ids = _class_ids(
#             raw_ids,
#             name=f"phase_{phase_id}_class_ids",
#         )
#         overlap = seen.intersection(group_ids)
#         if overlap:
#             raise ValueError(
#                 f"phase_groups contain duplicate classes {sorted(overlap)}"
#             )
#         seen.update(group_ids)
#         parsed_groups[phase_id] = group_ids

#     if seen != set(ids):
#         missing = sorted(set(ids) - seen)
#         extra = sorted(seen - set(ids))
#         raise ValueError(
#             "phase_groups must partition metrics class_ids exactly; "
#             f"missing={missing}, extra={extra}"
#         )

#     phases = sorted(parsed_groups)
#     index = {class_id: position for position, class_id in enumerate(ids)}
#     group_matrix = np.zeros(
#         (len(phases), len(phases)),
#         dtype=np.int64,
#     )

#     for true_position, true_phase in enumerate(phases):
#         true_rows = [index[c] for c in parsed_groups[true_phase]]
#         for pred_position, pred_phase in enumerate(phases):
#             pred_columns = [index[c] for c in parsed_groups[pred_phase]]
#             group_matrix[true_position, pred_position] = int(
#                 matrix[np.ix_(true_rows, pred_columns)].sum()
#             )

#     row_totals = group_matrix.sum(axis=1)
#     if np.any(row_totals <= 0):
#         raise ValueError(
#             "phase-origin confusion requires samples for every true phase"
#         )
#     rates = (
#         group_matrix.astype(np.float64)
#         / row_totals[:, None].astype(np.float64)
#     )

#     return {
#         "phase_ids": phases,
#         "phase_to_class_ids": {
#             int(phase): list(parsed_groups[phase])
#             for phase in phases
#         },
#         "count_matrix": group_matrix.tolist(),
#         "rate_matrix": rates.tolist(),
#         "true_phase_counts": {
#             int(phase): int(row_totals[position])
#             for position, phase in enumerate(phases)
#         },
#     }


# @torch.no_grad()
# def historical_spectral_decision_diagnostics(
#     model: Any,
#     *,
#     historical_class_ids: Sequence[int],
#     seen_class_ids: Sequence[int],
#     target_names: Optional[Sequence[str]] = None,
# ) -> Dict[str, Any]:
#     """Report the retained HSI class-region safety used by incremental training.

#     For each historical class ``c`` and every seen rival ``j != c`` this reads
#     the exact quantities used by the active preservation objective:

#         mean_gap[c,j]   = <I_c, U_c-U_j>_F,
#         dispersion[c,j] = upper std bound from full S_c[K,K,D],
#         score[c,j]      = mean_gap / dispersion when dispersion is non-zero.

#     The function is reporting-only.  It identifies the worst rival for each
#     historical class so structural scores can be compared directly with the
#     classwise confusion matrix.
#     """
#     old_ids = _class_ids(historical_class_ids, name="historical_class_ids")
#     seen_ids = _class_ids(seen_class_ids, name="seen_class_ids")
#     if not set(old_ids).issubset(seen_ids):
#         raise ValueError("historical_class_ids must be a subset of seen_class_ids")

#     safety = model.boundary_safety_scores(
#         historical_class_ids=old_ids,
#         rival_class_ids=seen_ids,
#     )
#     if tuple(safety.historical_class_ids) != tuple(old_ids):
#         raise RuntimeError("historical boundary class order changed")
#     if tuple(safety.rival_class_ids) != tuple(seen_ids):
#         raise RuntimeError("rival boundary class order changed")

#     gap = safety.mean_gap.detach()
#     dispersion = safety.dispersion_upper_bound.detach()
#     score = safety.score.detach()
#     expected = (len(old_ids), len(seen_ids))
#     for name, value in (
#         ("mean_gap", gap),
#         ("dispersion_upper_bound", dispersion),
#         ("spectral_decision_score", score),
#     ):
#         if tuple(value.shape) != expected:
#             raise RuntimeError(f"{name} has shape {tuple(value.shape)}, expected {expected}")
#         if not bool(torch.isfinite(value).all()):
#             raise RuntimeError(f"{name} contains NaN or Inf")

#     seen_position = {class_id: pos for pos, class_id in enumerate(seen_ids)}
#     names = list(target_names or [])
#     per_class: list[Dict[str, Any]] = []
#     all_scores: list[Tensor] = []

#     for old_pos, class_id in enumerate(old_ids):
#         self_pos = seen_position[class_id]
#         rival_mask = torch.ones(len(seen_ids), dtype=torch.bool, device=score.device)
#         rival_mask[self_pos] = False
#         rival_scores = score[old_pos][rival_mask]
#         rival_gaps = gap[old_pos][rival_mask]
#         rival_dispersion = dispersion[old_pos][rival_mask]
#         rival_ids = [seen_ids[i] for i in range(len(seen_ids)) if i != self_pos]
#         if rival_scores.numel() == 0:
#             raise RuntimeError(f"historical class {class_id} has no rival")

#         worst_local = int(rival_scores.argmin().item())
#         worst_rival_id = int(rival_ids[worst_local])
#         all_scores.append(rival_scores)
#         per_class.append(
#             {
#                 "class_id": int(class_id),
#                 "class_name": (
#                     str(names[class_id])
#                     if 0 <= class_id < len(names)
#                     else f"Class {class_id + 1}"
#                 ),
#                 "worst_rival_class_id": worst_rival_id,
#                 "worst_rival_class_name": (
#                     str(names[worst_rival_id])
#                     if 0 <= worst_rival_id < len(names)
#                     else f"Class {worst_rival_id + 1}"
#                 ),
#                 "worst_mean_gap": float(rival_gaps[worst_local].item()),
#                 "worst_dispersion_upper_bound": float(
#                     rival_dispersion[worst_local].item()
#                 ),
#                 "minimum_spectral_decision_score": float(
#                     rival_scores[worst_local].item()
#                 ),
#                 "mean_spectral_decision_score": float(rival_scores.mean().item()),
#             }
#         )

#     flattened = torch.cat(all_scores, dim=0)
#     return {
#         "historical_class_ids": list(old_ids),
#         "seen_class_ids": list(seen_ids),
#         "mean_spectral_decision_score": float(flattened.mean().item()),
#         "minimum_spectral_decision_score": float(flattened.min().item()),
#         "per_class": per_class,
#     }

# __all__ = [
#     "evaluate_loader",
#     "collect_reporting_features",
#     "class_structure_diagnostics",
#     "geometry_diagnostics",
#     "decision_state_diagnostics",
#     "summarize_class_group",
#     "old_new_summary",
#     "old_new_confusion_diagnostics",
#     "phase_origin_confusion_diagnostics",
#     "historical_spectral_decision_diagnostics",
# ]
