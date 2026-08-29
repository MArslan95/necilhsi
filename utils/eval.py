from __future__ import annotations

"""Read-only evaluation for the locked SCOPE-HSI architecture.

Roles
-----
1. Classification metrics are produced only by the one persistent classifier W
   acting on the one canonical representation Q.
2. Retained {I_c,S_c,N_c} are reported as class-structure diagnostics; they do
   not produce logits or labels.
3. Coordinate-transport diagnostics test the SCOPE stability assumption by
   comparing the same HSI pixels before/after an update, including per-spectral-
   region residuals and transport-aligned historical I/S drift.
4. Decision-state diagnostics measure compatibility between retained class
   identities I_c and persistent classifier directions W_c in the same Q-space.

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
# Reporting feature collection and transport diagnostics
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


def _aligned_reporting_arrays(
    previous: Mapping[str, Any],
    current: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    previous_features = np.asarray(
        previous["ordered_features"],
        dtype=np.float64,
    )
    current_features = np.asarray(
        current["ordered_features"],
        dtype=np.float64,
    )
    previous_labels = np.asarray(
        previous["labels"],
        dtype=np.int64,
    ).reshape(-1)
    current_labels = np.asarray(
        current["labels"],
        dtype=np.int64,
    ).reshape(-1)

    if (
        previous_features.ndim != 3
        or current_features.shape != previous_features.shape
        or previous_features.shape[0] != previous_labels.size
    ):
        raise ValueError(
            "previous/current reporting Q must be aligned [N,K,D]"
        )
    if not np.isfinite(previous_features).all() or not np.isfinite(
        current_features
    ).all():
        raise ValueError(
            "previous/current reporting Q contains NaN/Inf"
        )
    if not np.array_equal(previous_labels, current_labels):
        raise ValueError(
            "previous/current reporting labels are not aligned"
        )

    if "coords" not in previous or "coords" not in current:
        raise ValueError(
            "before/after drift diagnostics require HSI pixel coordinates"
        )
    previous_coords = np.asarray(
        previous["coords"],
        dtype=np.int64,
    )
    current_coords = np.asarray(
        current["coords"],
        dtype=np.int64,
    )
    if (
        previous_coords.shape != (previous_labels.size, 2)
        or current_coords.shape != previous_coords.shape
        or not np.array_equal(previous_coords, current_coords)
    ):
        raise ValueError(
            "previous/current reporting HSI pixels are not coordinate-aligned"
        )

    return (
        previous_features,
        current_features,
        previous_labels,
        previous_coords,
    )


def _empirical_class_structure(
    features: np.ndarray,
    labels: np.ndarray,
    class_ids: Sequence[int],
) -> Dict[int, tuple[np.ndarray, np.ndarray]]:
    q = np.asarray(features, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int64).reshape(-1)

    if q.ndim != 3 or q.shape[0] != y.size:
        raise ValueError(
            "empirical structure requires Q [N,K,D] aligned with labels"
        )
    if not np.isfinite(q).all():
        raise ValueError(
            "empirical structure Q contains NaN/Inf"
        )

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
    """Raw realized old-class I/S drift on identical held-out HSI pixels."""
    ids = _class_ids(old_class_ids)
    previous_features, current_features, labels, _ = (
        _aligned_reporting_arrays(previous, current)
    )

    before = _empirical_class_structure(
        previous_features,
        labels,
        ids,
    )
    after = _empirical_class_structure(
        current_features,
        labels,
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
                    np.square(
                        identity_after - identity_before
                    )
                )
            )
        )
        structure_drift = float(
            np.sqrt(
                np.mean(
                    np.square(
                        structure_after - structure_before
                    )
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
        "mean_identity_drift_rms": float(
            np.mean(identity_values)
        ),
        "mean_structure_drift_rms": float(
            np.mean(structure_values)
        ),
        "classes": classes,
    }


def transport_drift_diagnostics(
    previous: Mapping[str, Any],
    current: Mapping[str, Any],
    *,
    old_class_ids: Sequence[int],
    transform: Tensor | np.ndarray,
) -> Dict[str, Any]:
    """Test whether one SCOPE transform explains realized historical drift.

    Reports both raw and transport-aligned changes for:
      * held-out old-sample Q,
      * each ordered spectral region,
      * old-class identity I,
      * old-class structure S.

    This diagnostic never changes model state.
    """
    ids = _class_ids(old_class_ids)
    q_before, q_after, labels, _ = _aligned_reporting_arrays(
        previous,
        current,
    )

    k = int(q_before.shape[1])
    transform_tensor = torch.as_tensor(transform)
    source_dtype = transform_tensor.dtype
    r = np.asarray(
        transform_tensor.detach().cpu(),
        dtype=np.float64,
    )
    if r.shape != (k, k) or not np.isfinite(r).all():
        raise ValueError(
            f"transform must be finite [{k},{k}]"
        )

    eye = np.eye(k, dtype=np.float64)
    orthogonality = float(
        np.max(np.abs(r.T @ r - eye))
    )
    tolerance = (
        5e-5
        if source_dtype == torch.float32
        else 1e-10
    )
    if orthogonality > tolerance:
        raise ValueError(
            "transport_drift_diagnostics requires an orthogonal transform"
        )

    q_aligned = np.einsum(
        "ka,nad->nkd",
        r,
        q_before,
        optimize=True,
    )

    raw_difference = q_after - q_before
    aligned_difference = q_after - q_aligned

    raw_feature_rms = float(
        np.sqrt(np.mean(np.square(raw_difference)))
    )
    aligned_feature_rms = float(
        np.sqrt(np.mean(np.square(aligned_difference)))
    )

    per_region = []
    for region in range(k):
        per_region.append(
            {
                "spectral_region": region,
                "raw_feature_change_rms": float(
                    np.sqrt(
                        np.mean(
                            np.square(
                                raw_difference[:, region, :]
                            )
                        )
                    )
                ),
                "aligned_feature_residual_rms": float(
                    np.sqrt(
                        np.mean(
                            np.square(
                                aligned_difference[:, region, :]
                            )
                        )
                    )
                ),
            }
        )

    before = _empirical_class_structure(
        q_before,
        labels,
        ids,
    )
    after = _empirical_class_structure(
        q_after,
        labels,
        ids,
    )

    classes: list[Dict[str, Any]] = []
    raw_i_values: list[float] = []
    aligned_i_values: list[float] = []
    raw_s_values: list[float] = []
    aligned_s_values: list[float] = []

    for class_id in ids:
        i_before, s_before = before[class_id]
        i_after, s_after = after[class_id]

        i_aligned = np.einsum(
            "ka,ad->kd",
            r,
            i_before,
            optimize=True,
        )
        s_aligned = np.einsum(
            "ka,abd,lb->kld",
            r,
            s_before,
            r,
            optimize=True,
        )

        raw_i = float(
            np.sqrt(np.mean(np.square(i_after - i_before)))
        )
        aligned_i = float(
            np.sqrt(np.mean(np.square(i_after - i_aligned)))
        )
        raw_s = float(
            np.sqrt(np.mean(np.square(s_after - s_before)))
        )
        aligned_s = float(
            np.sqrt(np.mean(np.square(s_after - s_aligned)))
        )

        raw_i_values.append(raw_i)
        aligned_i_values.append(aligned_i)
        raw_s_values.append(raw_s)
        aligned_s_values.append(aligned_s)

        classes.append(
            {
                "class_id": class_id,
                "raw_identity_drift_rms": raw_i,
                "aligned_identity_residual_rms": aligned_i,
                "raw_structure_drift_rms": raw_s,
                "aligned_structure_residual_rms": aligned_s,
            }
        )

    return {
        "old_class_ids": ids,
        "orthogonality_residual": orthogonality,
        "raw_feature_change_rms": raw_feature_rms,
        "aligned_feature_residual_rms": aligned_feature_rms,
        "per_spectral_region": per_region,
        "mean_raw_identity_drift_rms": float(
            np.mean(raw_i_values)
        ),
        "mean_aligned_identity_residual_rms": float(
            np.mean(aligned_i_values)
        ),
        "mean_raw_structure_drift_rms": float(
            np.mean(raw_s_values)
        ),
        "mean_aligned_structure_residual_rms": float(
            np.mean(aligned_s_values)
        ),
        "classes": classes,
    }


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


__all__ = [
    "evaluate_loader",
    "collect_reporting_features",
    "structure_drift_diagnostics",
    "transport_drift_diagnostics",
    "class_structure_diagnostics",
    "geometry_diagnostics",
    "decision_state_diagnostics",
    "summarize_class_group",
    "old_new_summary",
]

