from __future__ import annotations
from contextlib import contextmanager
import math
from numbers import Integral, Real
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np
import torch

from models.classifier import CosineClassificationHead
from models.geometry_bank import (
    ClassDiscriminativeStructureState,
    HSIClassDiscriminativeStructureBank,
    whole_cds_pair_risk,
    ordered_variation_operator,
)

Tensor = torch.Tensor


def _as_int(value: object, *, name: str) -> int:
    if isinstance(value, bool):
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
    result = [_as_int(value, name=f"{name} value") for value in values]
    if not result or len(result) != len(set(result)):
        raise ValueError(f"{name} must contain unique IDs")
    return result


@contextmanager
def _temporary_eval_state(model: Any):
    states = [(module, bool(module.training)) for module in model.modules()]
    try:
        model.eval()
        yield
    finally:
        for module, state in states:
            module.training = state


def _unpack(
    batch: Mapping[str, Any],
    *,
    model: Any,
    device: torch.device,
) -> tuple[Tensor, Tensor, Tensor]:
    required = {"image", "raw_center_spectrum", "label"}
    missing = required - set(batch)
    if missing:
        raise KeyError(f"evaluation batch missing {sorted(missing)}")
    patch = torch.as_tensor(batch["image"], device=device, dtype=model.dtype)
    spectrum = torch.as_tensor(
        batch["raw_center_spectrum"], device=device, dtype=model.dtype
    )
    raw_labels = torch.as_tensor(batch["label"])
    if (
        raw_labels.dtype == torch.bool
        or raw_labels.is_complex()
        or torch.is_floating_point(raw_labels)
    ):
        raise RuntimeError("evaluation labels must use an integer dtype")
    labels = raw_labels.to(device=device, dtype=torch.long).flatten()
    if labels.shape != (patch.size(0),):
        raise RuntimeError("evaluation labels are misaligned")
    return patch, spectrum, labels


def _confusion(
    targets: np.ndarray, predictions: np.ndarray, ids: Sequence[int]
) -> np.ndarray:
    index = {class_id: row for row, class_id in enumerate(ids)}
    matrix = np.zeros((len(ids), len(ids)), dtype=np.int64)
    for target, prediction in zip(targets.tolist(), predictions.tolist()):
        if int(target) not in index or int(prediction) not in index:
            raise RuntimeError("confusion input contains a class outside class_ids")
        matrix[index[int(target)], index[int(prediction)]] += 1
    return matrix


def _kappa(matrix: np.ndarray) -> float:
    total = float(matrix.sum())
    if total <= 0.0:
        return 0.0
    observed = float(np.trace(matrix)) / total
    expected = float((matrix.sum(axis=1) * matrix.sum(axis=0)).sum()) / (total * total)
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
    decision_cds: ClassDiscriminativeStructureState | None = None,
) -> Dict[str, Any]:
    ids = _class_ids(class_ids)
    target_ids = (
        ids
        if target_class_ids is None
        else _class_ids(target_class_ids, name="target_class_ids")
    )
    if set(target_ids) - set(ids):
        raise ValueError("target_class_ids must be a subset of class_ids")

    evaluation_device = torch.device(device)
    model.validate_model_state()
    if not isinstance(model.classifier, CosineClassificationHead):
        raise RuntimeError("classifier must be CosineClassificationHead")

    predictions: list[Tensor] = []
    labels_rows: list[Tensor] = []
    margin_rows: list[Tensor] = []
    energy_margin_rows: list[Tensor] = []
    raw_energy_margin_rows: list[Tensor] = []
    probability_rows: list[Tensor] = []
    learned_rows: list[Tensor] = []
    physical_rows: list[Tensor] = []
    whole_rows: list[Tensor] = []
    cosine_rows: list[Tensor] = []
    ce_rows: list[Tensor] = []
    has_rivals = len(ids) > 1
    allowed = torch.tensor(ids, device=evaluation_device, dtype=torch.long)

    with _temporary_eval_state(model):
        for batch in loader:
            patch, spectrum, labels = _unpack(
                batch, model=model, device=evaluation_device
            )
            if target_class_ids is not None:
                # A cumulative reporting loader may contain other targets.
                # Filter before mapping labels into a restricted decision set.
                target_tensor = labels.new_tensor(target_ids)
                selected_rows = torch.isin(labels, target_tensor)
                if not bool(selected_rows.any()):
                    continue
                patch, spectrum, labels = (
                    patch[selected_rows], spectrum[selected_rows], labels[selected_rows]
                )
            if not bool(torch.isin(labels, allowed).all()):
                raise RuntimeError(
                    "evaluation loader contains a class outside class_ids"
                )
            q = model.extract_features(
                patch,
                center_spectrum=spectrum,
                return_aux=False,
            ).ordered_features
            p = model.physical_spectral_evidence(spectrum)
            output = model.classify_features(
                q,
                physical_spectral_evidence=p,
                class_ids=ids,
                cds=decision_cds,
                trainable_class_ids=(),
            )

            matches = labels.unsqueeze(1).eq(output.class_ids.unsqueeze(0))
            if not bool(matches.sum(dim=1).eq(1).all()):
                raise RuntimeError("could not map labels to classifier columns")
            target_local = matches.to(torch.long).argmax(dim=1)
            rows = torch.arange(labels.numel(), device=labels.device)
            if has_rivals:
                target_logit = output.logits[rows, target_local]
                rival_logits = output.logits.clone()
                rival_logits[rows, target_local] = -torch.inf
                max_rival = rival_logits.max(dim=1).values
                logit_margin = target_logit - max_rival
                rival_energy = torch.logsumexp(rival_logits, dim=1)
                raw_energy_margin = target_logit - rival_energy
                rival_count = int(output.logits.size(1)) - 1
                normalized_rival_energy = rival_energy - target_logit.new_tensor(
                    math.log(float(rival_count))
                )
                energy_margin = target_logit - normalized_rival_energy
                margin_rows.append(logit_margin.detach())
                energy_margin_rows.append(energy_margin.detach())
                raw_energy_margin_rows.append(raw_energy_margin.detach())

            predictions.append(output.prediction.detach())
            labels_rows.append(labels.detach())
            ce_rows.append(torch.nn.functional.cross_entropy(
                output.logits, target_local, reduction="none"
            ).detach())
            probability_rows.append(output.probabilities[rows, target_local].detach())
            learned_rows.append(output.learned_support[rows, target_local].detach())
            physical_rows.append(output.physical_support[rows, target_local].detach())
            whole_rows.append(output.whole_support[rows, target_local].detach())
            cosine_rows.append(output.cosine_scores[rows, target_local].detach())

    if not predictions:
        raise RuntimeError("cannot evaluate an empty loader")

    prediction_all = torch.cat(predictions)
    labels_all = torch.cat(labels_rows)
    margin_all = torch.cat(margin_rows) if has_rivals else None
    energy_margin_all = torch.cat(energy_margin_rows) if has_rivals else None
    raw_energy_margin_all = torch.cat(raw_energy_margin_rows) if has_rivals else None
    ce_all = torch.cat(ce_rows)
    probability_all = torch.cat(probability_rows)
    learned_all = torch.cat(learned_rows)
    physical_all = torch.cat(physical_rows)
    whole_all = torch.cat(whole_rows)
    cosine_all = torch.cat(cosine_rows)

    target_tensor = torch.tensor(target_ids, device=labels_all.device, dtype=torch.long)
    mask = torch.isin(labels_all, target_tensor)
    labels = labels_all[mask]
    prediction = prediction_all[mask]
    margin = margin_all[mask] if has_rivals else None
    energy_margin = energy_margin_all[mask] if has_rivals else None
    raw_energy_margin = raw_energy_margin_all[mask] if has_rivals else None
    ce = ce_all[mask]
    target_probability = probability_all[mask]
    target_learned = learned_all[mask]
    target_physical = physical_all[mask]
    target_whole = whole_all[mask]
    target_cosine = cosine_all[mask]

    per_class_accuracy: Dict[int, float] = {}
    per_class_margin: Dict[int, float | None] = {}
    per_class_energy_margin: Dict[int, float | None] = {}
    per_class_raw_energy_margin: Dict[int, float | None] = {}
    per_class_ce: Dict[int, float] = {}
    per_class_support: Dict[int, float] = {}
    class_counts: Dict[int, int] = {}
    for class_id in target_ids:
        class_mask = labels.eq(class_id)
        count = int(class_mask.sum().item())
        if count <= 0:
            raise RuntimeError(f"target class {class_id} is absent")
        class_counts[class_id] = count
        per_class_accuracy[class_id] = float(
            prediction[class_mask].eq(labels[class_mask]).float().mean().item()
        )
        per_class_margin[class_id] = (
            float(margin[class_mask].mean().item()) if has_rivals else None
        )
        per_class_energy_margin[class_id] = (
            float(energy_margin[class_mask].mean().item()) if has_rivals else None
        )
        per_class_raw_energy_margin[class_id] = (
            float(raw_energy_margin[class_mask].mean().item()) if has_rivals else None
        )
        per_class_ce[class_id] = float(ce[class_mask].mean().item())
        per_class_support[class_id] = float(target_whole[class_mask].mean().item())

    confusion = _confusion(labels.cpu().numpy(), prediction.cpu().numpy(), ids)
    oa = float(prediction.eq(labels).float().mean().item())
    ba = float(sum(per_class_accuracy.values()) / len(per_class_accuracy))
    return {
        "class_ids": ids,
        "target_class_ids": target_ids,
        "overall_accuracy": oa,
        "accuracy": oa,
        "balanced_accuracy": ba,
        "minimum_class_accuracy": float(min(per_class_accuracy.values())),
        "kappa": float(_kappa(confusion)),
        "confusion_matrix": confusion.tolist(),
        "class_counts": class_counts,
        "per_class_accuracy": per_class_accuracy,
        "cross_entropy": float(ce.mean().item()),
        "per_class_cross_entropy": per_class_ce,
        "per_class_mean_logit_margin": per_class_margin,
        "per_class_mean_energy_margin": per_class_energy_margin,
        "per_class_mean_raw_energy_margin": per_class_raw_energy_margin,
        "per_class_mean_whole_cds_support": per_class_support,
        # Rival-based diagnostics are undefined for a one-class decision set.
        "mean_logit_margin": float(margin.mean().item()) if has_rivals else None,
        "mean_energy_margin": float(energy_margin.mean().item()) if has_rivals else None,
        "energy_violation_fraction": (
            float(energy_margin.le(0).float().mean().item()) if has_rivals else None
        ),
        "mean_raw_energy_margin": (
            float(raw_energy_margin.mean().item()) if has_rivals else None
        ),
        "raw_energy_violation_fraction": float(
            raw_energy_margin.le(0).float().mean().item()
        ) if has_rivals else None,
        "mean_target_probability": float(target_probability.mean().item()),
        "mean_target_learned_support": float(target_learned.mean().item()),
        "mean_target_physical_support": float(target_physical.mean().item()),
        "mean_target_whole_cds_support": float(target_whole.mean().item()),
        # Kept as a diagnostic compatibility key; it is not the decision score.
        "mean_target_cosine": float(target_cosine.mean().item()),
    }


def _optional_float(value: Any) -> float | None:
    return None if value is None else float(value)


def summarize_evaluation_metrics(
    metrics: Optional[Mapping[str, Any]],
) -> Optional[Dict[str, Any]]:
    if metrics is None:
        return None
    required = (
        "overall_accuracy",
        "balanced_accuracy",
        "minimum_class_accuracy",
        "kappa",
        "mean_logit_margin",
        "mean_energy_margin",
        "energy_violation_fraction",
        "mean_target_probability",
    )
    missing = [name for name in required if name not in metrics]
    if missing:
        raise RuntimeError(f"evaluation/reporting contract mismatch: missing {missing}")
    return {
        "oa": float(metrics["overall_accuracy"]),
        "ba": float(metrics["balanced_accuracy"]),
        "min_class_accuracy": float(metrics["minimum_class_accuracy"]),
        "kappa": float(metrics["kappa"]),
        "logit_margin": _optional_float(metrics["mean_logit_margin"]),
        "energy_margin": _optional_float(metrics["mean_energy_margin"]),
        "energy_violation_fraction": _optional_float(metrics["energy_violation_fraction"]),
        "raw_energy_margin": _optional_float(
            metrics.get("mean_raw_energy_margin", metrics["mean_energy_margin"])
        ),
        "raw_energy_violation_fraction": _optional_float(
            metrics.get(
                "raw_energy_violation_fraction", metrics["energy_violation_fraction"]
            )
        ),
        "target_probability": float(metrics["mean_target_probability"]),
        "learned_support": float(metrics.get("mean_target_learned_support", 0.0)),
        "physical_support": float(metrics.get("mean_target_physical_support", 0.0)),
        "whole_cds_support": float(metrics.get("mean_target_whole_cds_support", 0.0)),
    }


def _ordered_variation_statistics(variation: Tensor) -> tuple[Tensor, Tensor]:
    """Signed V_Q/V_P interval widths; not covariance or IQR scales."""
    value = torch.as_tensor(variation)
    if value.ndim != 4 or value.size(1) != 2:
        raise ValueError("signed variation must be [C,2,L,F]")
    width = (value[:, 1] - value[:, 0]).clamp_min(0)
    return (width > 0).to(value.dtype).mean(dim=1), width.mean(dim=1)


def _unified_moment_population_trace(moments: Mapping[str, Tensor]) -> tuple[Tensor, Tensor]:
    """Population covariance trace from one low-rank sample covariance/class.

    The bank stores an unbiased N-1 sample covariance approximation. For
    diagnostics of an empirical class distribution, multiply by (N-1)/N.
    This reports the compact approximation, not an exact dense covariance.
    """
    mean = moments["mean"]
    basis = moments["basis"]
    eigenvalues = moments["eigenvalues"]
    residual = moments["residual_variance"]
    counts = moments["counts"]
    if (mean.ndim != 2 or basis.ndim != 3 or
        basis.shape[:2] != (mean.shape[0], mean.shape[1]) or
        eigenvalues.shape != (mean.shape[0], basis.shape[2]) or
        residual.shape != (mean.shape[0],) or
        counts.shape != (mean.shape[0],)):
        raise ValueError("unified CDS moments have inconsistent class/rank shapes")
    if not bool((counts >= 1).all()) or not all(bool(torch.isfinite(x).all()) for x in
         (mean, basis, eigenvalues, residual)):
        raise ValueError("invalid unified CDS moments")
    if bool((eigenvalues < 0).any()) or bool((residual < 0).any()):
        raise ValueError("negative CDS covariance component")
    active_rank = (basis.square().sum(dim=1) > 1e-10).sum(dim=1)
    trace = eigenvalues.sum(dim=1) + (mean.shape[1] - active_rank).to(residual) * residual
    population_trace = trace * (counts - 1).clamp_min(0).to(trace) / counts.to(trace)
    return population_trace, active_rank


@torch.no_grad()
def class_structure_diagnostics(
    model: Any,
    class_ids: Sequence[int],
    *,
    target_names: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    ids = _class_ids(class_ids)
    bank = model.structure_bank
    if not isinstance(bank, HSIClassDiscriminativeStructureBank):
        raise RuntimeError("structure_bank must be HSIClassDiscriminativeStructureBank")
    structure = bank.class_structures(ids)
    cds = bank.build_cds(structure)
    relations = cds.relations
    class_count = len(ids)
    offdiag = ~torch.eye(
        class_count, device=structure.learned_identity.device, dtype=torch.bool
    )

    learned_identity = relations.learned_identity_proximity_by_region.mean(dim=2)
    learned_relation = relations.learned_relation_proximity
    physical_identity = relations.physical_identity_proximity_by_band.mean(dim=2)
    physical_relation = relations.physical_relation_proximity
    physical_combined = relations.physical_combined_proximity
    moments = bank.class_moments(ids)
    intra_population_trace, active_covariance_rank = _unified_moment_population_trace(moments)
    pair_risk, learned_covariance_risk, native_spectral_risk = whole_cds_pair_risk(cds)
    learned_active, learned_spread = _ordered_variation_statistics(
        structure.learned_variation
    )
    physical_active, physical_spread = _ordered_variation_statistics(
        structure.physical_variation
    )

    def offdiag_mean(matrix: Tensor) -> float:
        return float(matrix[offdiag].mean().item()) if bool(offdiag.any()) else 1.0

    if bool(offdiag.any()):
        masked = physical_combined.masked_fill(~offdiag, -torch.inf)
        flat = int(masked.argmax().item())
        left, right = divmod(flat, class_count)
        most_similar_pair = [ids[left], ids[right]]
        maximum_physical_proximity = float(masked[left, right].item())
    else:
        most_similar_pair = None
        maximum_physical_proximity = 1.0

    names = list(target_names or [])
    classes = []
    for position, class_id in enumerate(ids):
        # One distribution/class. No old spectral condition mass or groups exist.
        counts = moments["counts"][position]
        intra_trace = intra_population_trace[position]
        native_band_variance = moments["physical_variance"][position]
        native_adjacent_variance = moments["physical_delta_variance"][position]
        classes.append(
            {
                "class_id": class_id,
                "sample_count": int(counts.item()),
                "learned_covariance_rank": int(active_covariance_rank[position].item()),
                "learned_intra_covariance_trace": float(intra_trace.item()),
                "learned_intra_covariance_trace_kind": "population trace estimated from low-rank sample covariance",
                "native_band_variance_mean": float(native_band_variance.mean().item()),
                "native_adjacent_difference_variance_mean": float(native_adjacent_variance.mean().item()),
                "class_name": (
                    str(names[class_id])
                    if class_id < len(names)
                    else f"Class {class_id + 1}"
                ),
                "learned_identity_norm": float(
                    torch.linalg.vector_norm(
                        structure.learned_identity[position]
                    ).item()
                ),
                "learned_absolute_deviation_norm": float(
                    structure.learned_absolute_deviation[position].norm().item()
                ),
                "learned_variation_norm": float(
                    torch.linalg.vector_norm(
                        structure.learned_variation[position]
                    ).item()
                ),
                "physical_identity_norm": float(
                    torch.linalg.vector_norm(
                        structure.physical_identity[position]
                    ).item()
                ),
                "physical_variation_norm": float(
                    torch.linalg.vector_norm(
                        structure.physical_variation[position]
                    ).item()
                ),
                "learned_active_relation_fraction": float(
                    learned_active[position].mean().item()
                ),
                "learned_positive_relation_scale": float(
                    learned_spread[position].mean().item()
                ),
                "physical_active_relation_fraction": float(
                    physical_active[position].mean().item()
                ),
                "physical_positive_relation_scale": float(
                    physical_spread[position].mean().item()
                ),
            }
        )

    return {
        "class_ids": ids,
        "mean_learned_identity_proximity": offdiag_mean(learned_identity),
        "mean_learned_relation_proximity": offdiag_mean(learned_relation),
        "mean_learned_covariance_pair_risk": offdiag_mean(learned_covariance_risk),
        "mean_native_spectral_pair_risk": offdiag_mean(native_spectral_risk),
        "mean_joint_covariance_spectral_pair_risk": offdiag_mean(pair_risk),
        "legacy_relation_metrics_are_diagnostic_only": True,
        "mean_physical_identity_proximity": offdiag_mean(physical_identity),
        "mean_physical_relation_proximity": offdiag_mean(physical_relation),
        "maximum_physical_proximity": maximum_physical_proximity,
        "most_physically_similar_class_pair": most_similar_pair,
        "classes": classes,
    }


@torch.no_grad()
def structural_decision_diagnostics(
    model: Any, class_ids: Sequence[int]
) -> Dict[str, Any]:
    ids = _class_ids(class_ids)
    structure = model.structure_bank.class_structures(ids)
    learned_variation = structure.learned_variation
    active, spread = _ordered_variation_statistics(learned_variation)
    moment_trace, active_rank = _unified_moment_population_trace(
        model.structure_bank.class_moments(ids)
    )
    return {
        "active_cds_geometry": "one mean and one low-rank covariance per class; native spectral mean, variance, adjacent differences",
        "mean_learned_covariance_trace": float(moment_trace.mean().item()),
        "mean_learned_covariance_rank": float(active_rank.to(moment_trace).mean().item()),
        "legacy_interval_diagnostics_are_not_active_supervision": True,
        "class_ids": ids,
        "classifier_parameter_count": int(model.classifier.parameter_count),
        "classifier_rule": "single cosine classifier over Q; inference is tau*cosine(Q,W); Whole-CDS is training-time structural supervision and read-only evaluation evidence",
        "classifier_logit_scale": float(model.classifier.logit_scale.detach().item()),
        "classifier_beta": 0.0,
        "mean_learned_relation_variation_norm": float(
            torch.linalg.vector_norm(
                learned_variation.reshape(learned_variation.size(0), -1), dim=1
            )
            .mean()
            .item()
        ),
        "mean_active_ordered_relation_fraction": float(active.mean().item()),
        "mean_positive_ordered_relation_scale": float(spread.mean().item()),
    }


@torch.no_grad()
def cross_phase_cds_decision_diagnostics(
    model: Any,
    *,
    old_class_ids: Sequence[int],
    new_class_ids: Sequence[int],
) -> Dict[str, Any]:
    old_ids = _class_ids(old_class_ids, name="old_class_ids")
    new_ids = _class_ids(new_class_ids, name="new_class_ids")
    seen_ids = old_ids + new_ids
    structure = model.structure_bank.class_structures(seen_ids)
    cds = model.structure_bank.build_cds(structure)
    relations = cds.relations
    class_count = len(seen_ids)
    old_count = len(old_ids)
    old_mask = torch.arange(class_count, device=model.device) < old_count
    diagonal = torch.eye(class_count, device=model.device, dtype=torch.bool)
    cross_mask = (~diagonal) & (old_mask.unsqueeze(1) ^ old_mask.unsqueeze(0))
    joint, learned, physical = whole_cds_pair_risk(cds)

    output = model.classifier(
        structure.learned_identity.to(device=model.device, dtype=model.dtype),
        physical_spectral_evidence=structure.physical_identity.to(
            device=model.device, dtype=model.dtype
        ),
        class_ids=structure.class_ids.to(device=model.device),
        cds=cds,
        trainable_class_ids=(),
    )
    target = torch.arange(class_count, device=model.device)
    target_logit = output.logits[target, target]
    rival = output.logits.clone()
    rival[target, target] = -torch.inf
    margin = target_logit - rival.max(dim=1).values

    def masked_mean(matrix: Tensor, mask: Tensor) -> float:
        return float(matrix[mask].mean().item()) if bool(mask.any()) else 0.0

    return {
        "old_class_ids": old_ids,
        "new_class_ids": new_ids,
        "cross_phase_learned_proximity_mean": masked_mean(learned, cross_mask),
        "cross_phase_physical_proximity_mean": masked_mean(physical, cross_mask),
        "cross_phase_joint_proximity_mean": masked_mean(joint, cross_mask),
        "class_identity_cosine_margin_mean": float(margin.mean().item()),
        "class_identity_cosine_margin_min": float(margin.min().item()),
    }


def _mapping_value(mapping: Mapping[Any, Any] | None, class_id: int) -> float | None:
    if mapping is None:
        raise RuntimeError("required per-class mapping is absent")
    if class_id in mapping:
        return _optional_float(mapping[class_id])
    key = str(class_id)
    if key in mapping:
        return _optional_float(mapping[key])
    raise KeyError(class_id)


def summarize_class_group(
    metrics: Mapping[str, Any], class_ids: Sequence[int]
) -> Dict[str, Any]:
    ids = _class_ids(class_ids)
    accuracies = [
        _mapping_value(metrics.get("per_class_accuracy"), class_id) for class_id in ids
    ]
    margins = [
        _mapping_value(metrics.get("per_class_mean_logit_margin"), class_id)
        for class_id in ids
    ]
    energy_margins = [
        _mapping_value(metrics.get("per_class_mean_energy_margin"), class_id)
        for class_id in ids
    ]
    return {
        "class_ids": ids,
        "balanced_accuracy": float(sum(accuracies) / len(accuracies)),
        "minimum_class_accuracy": float(min(accuracies)),
        "mean_logit_margin": (
            None if any(value is None for value in margins)
            else float(sum(margins) / len(margins))
        ),
        "mean_energy_margin": (
            None if any(value is None for value in energy_margins)
            else float(sum(energy_margins) / len(energy_margins))
        ),
    }


def old_new_summary(
    metrics: Mapping[str, Any],
    *,
    old_class_ids: Sequence[int],
    new_class_ids: Sequence[int],
) -> Dict[str, Any]:
    old = summarize_class_group(metrics, old_class_ids)
    new = summarize_class_group(metrics, new_class_ids)
    a = float(old["balanced_accuracy"])
    b = float(new["balanced_accuracy"])
    harmonic = 0.0 if a + b <= 0.0 else 2.0 * a * b / (a + b)
    return {"old": old, "new": new, "harmonic_old_new_accuracy": harmonic}


def phase_origin_confusion_diagnostics(
    metrics: Mapping[str, Any],
    *,
    phase_to_classes: Mapping[int, Sequence[int]],
) -> Dict[str, Any]:
    matrix = np.asarray(metrics["confusion_matrix"], dtype=np.int64)
    ids = [int(value) for value in metrics["class_ids"]]
    index = {class_id: position for position, class_id in enumerate(ids)}
    phases = sorted(int(phase) for phase in phase_to_classes)
    result: Dict[str, Any] = {}
    for true_phase in phases:
        true_ids = [
            int(value) for value in phase_to_classes[true_phase] if int(value) in index
        ]
        if not true_ids:
            continue
        rows = [index[value] for value in true_ids]
        row_total = int(matrix[rows, :].sum())
        row: Dict[str, float] = {}
        for predicted_phase in phases:
            predicted_ids = [
                int(value)
                for value in phase_to_classes[predicted_phase]
                if int(value) in index
            ]
            columns = [index[value] for value in predicted_ids]
            count = int(matrix[np.ix_(rows, columns)].sum()) if columns else 0
            row[str(predicted_phase)] = (
                0.0 if row_total == 0 else count / float(row_total)
            )
        result[str(true_phase)] = row
    return result


@torch.no_grad()
def whole_cds_classifier_diagnostics(
    model: Any,
    class_ids: Sequence[int],
    *,
    historical_class_count: int,
) -> Dict[str, float]:
    ids = _class_ids(class_ids)
    old_count = int(historical_class_count)
    if old_count <= 0 or old_count >= len(ids):
        raise ValueError("historical_class_count must split old and current classes")
    structure = model.structure_bank.class_structures(ids)
    cds = model.structure_bank.build_cds(structure)
    output = model.classifier(
        structure.learned_identity.to(device=model.device, dtype=model.dtype),
        physical_spectral_evidence=structure.physical_identity.to(
            device=model.device, dtype=model.dtype
        ),
        class_ids=structure.class_ids.to(device=model.device),
        cds=cds,
        trainable_class_ids=(),
    )
    target = torch.arange(len(ids), device=model.device)
    target_logit = output.logits[target, target]
    rival = output.logits.clone()
    rival[target, target] = -torch.inf
    margin = target_logit - rival.max(dim=1).values
    return {
        "whole_cds_margin_mean": float(margin.mean().item()),
        "whole_cds_margin_min": float(margin.min().item()),
        "learned_self_support_mean": float(
            output.learned_support[target, target].mean().item()
        ),
        "physical_self_support_mean": float(
            output.physical_support[target, target].mean().item()
        ),
        "whole_self_support_mean": float(
            output.whole_support[target, target].mean().item()
        ),
    }


__all__ = [
    "evaluate_loader",
    "summarize_evaluation_metrics",
    "class_structure_diagnostics",
    "structural_decision_diagnostics",
    "cross_phase_cds_decision_diagnostics",
    "summarize_class_group",
    "old_new_summary",
    "phase_origin_confusion_diagnostics",
    "whole_cds_classifier_diagnostics",
]




# from __future__ import annotations
# from contextlib import contextmanager
# import math
# from numbers import Integral, Real
# from typing import Any, Dict, Mapping, Optional, Sequence

# import numpy as np
# import torch

# from models.classifier import CosineClassificationHead
# from models.geometry_bank import (
#     ClassDiscriminativeStructureState,
#     HSIClassDiscriminativeStructureBank,
#     combine_learned_physical_relevance,
#     ordered_variation_operator,
# )

# Tensor = torch.Tensor


# def _as_int(value: object, *, name: str) -> int:
#     if isinstance(value, bool):
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
#     result = [_as_int(value, name=f"{name} value") for value in values]
#     if not result or len(result) != len(set(result)):
#         raise ValueError(f"{name} must contain unique IDs")
#     return result


# @contextmanager
# def _temporary_eval_state(model: Any):
#     states = [(module, bool(module.training)) for module in model.modules()]
#     try:
#         model.eval()
#         yield
#     finally:
#         for module, state in states:
#             module.training = state


# def _unpack(
#     batch: Mapping[str, Any],
#     *,
#     model: Any,
#     device: torch.device,
# ) -> tuple[Tensor, Tensor, Tensor]:
#     required = {"image", "raw_center_spectrum", "label"}
#     missing = required - set(batch)
#     if missing:
#         raise KeyError(f"evaluation batch missing {sorted(missing)}")
#     patch = torch.as_tensor(batch["image"], device=device, dtype=model.dtype)
#     spectrum = torch.as_tensor(
#         batch["raw_center_spectrum"], device=device, dtype=model.dtype
#     )
#     raw_labels = torch.as_tensor(batch["label"])
#     if (
#         raw_labels.dtype == torch.bool
#         or raw_labels.is_complex()
#         or torch.is_floating_point(raw_labels)
#     ):
#         raise RuntimeError("evaluation labels must use an integer dtype")
#     labels = raw_labels.to(device=device, dtype=torch.long).flatten()
#     if labels.shape != (patch.size(0),):
#         raise RuntimeError("evaluation labels are misaligned")
#     return patch, spectrum, labels


# def _confusion(
#     targets: np.ndarray, predictions: np.ndarray, ids: Sequence[int]
# ) -> np.ndarray:
#     index = {class_id: row for row, class_id in enumerate(ids)}
#     matrix = np.zeros((len(ids), len(ids)), dtype=np.int64)
#     for target, prediction in zip(targets.tolist(), predictions.tolist()):
#         if int(target) not in index or int(prediction) not in index:
#             raise RuntimeError("confusion input contains a class outside class_ids")
#         matrix[index[int(target)], index[int(prediction)]] += 1
#     return matrix


# def _kappa(matrix: np.ndarray) -> float:
#     total = float(matrix.sum())
#     if total <= 0.0:
#         return 0.0
#     observed = float(np.trace(matrix)) / total
#     expected = float((matrix.sum(axis=1) * matrix.sum(axis=0)).sum()) / (total * total)
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
#     decision_cds: ClassDiscriminativeStructureState | None = None,
# ) -> Dict[str, Any]:
#     ids = _class_ids(class_ids)
#     target_ids = (
#         ids
#         if target_class_ids is None
#         else _class_ids(target_class_ids, name="target_class_ids")
#     )
#     if set(target_ids) - set(ids):
#         raise ValueError("target_class_ids must be a subset of class_ids")

#     evaluation_device = torch.device(device)
#     model.validate_model_state()
#     if not isinstance(model.classifier, CosineClassificationHead):
#         raise RuntimeError("classifier must be CosineClassificationHead")

#     predictions: list[Tensor] = []
#     labels_rows: list[Tensor] = []
#     margin_rows: list[Tensor] = []
#     energy_margin_rows: list[Tensor] = []
#     raw_energy_margin_rows: list[Tensor] = []
#     probability_rows: list[Tensor] = []
#     learned_rows: list[Tensor] = []
#     physical_rows: list[Tensor] = []
#     whole_rows: list[Tensor] = []
#     cosine_rows: list[Tensor] = []
#     allowed = torch.tensor(ids, device=evaluation_device, dtype=torch.long)

#     with _temporary_eval_state(model):
#         for batch in loader:
#             patch, spectrum, labels = _unpack(
#                 batch, model=model, device=evaluation_device
#             )
#             if not bool(torch.isin(labels, allowed).all()):
#                 raise RuntimeError(
#                     "evaluation loader contains a class outside class_ids"
#                 )
#             q = model.extract_features(
#                 patch,
#                 center_spectrum=spectrum,
#                 return_aux=False,
#             ).ordered_features
#             p = model.physical_spectral_evidence(spectrum)
#             output = model.classify_features(
#                 q,
#                 physical_spectral_evidence=p,
#                 class_ids=ids,
#                 cds=decision_cds,
#                 trainable_class_ids=(),
#             )

#             matches = labels.unsqueeze(1).eq(output.class_ids.unsqueeze(0))
#             if not bool(matches.sum(dim=1).eq(1).all()):
#                 raise RuntimeError("could not map labels to classifier columns")
#             target_local = matches.to(torch.long).argmax(dim=1)
#             rows = torch.arange(labels.numel(), device=labels.device)
#             target_logit = output.logits[rows, target_local]
#             rival_logits = output.logits.clone()
#             rival_logits[rows, target_local] = -torch.inf
#             max_rival = rival_logits.max(dim=1).values
#             logit_margin = target_logit - max_rival
#             rival_energy = torch.logsumexp(rival_logits, dim=1)
#             raw_energy_margin = target_logit - rival_energy
#             rival_count = int(output.logits.size(1)) - 1
#             if rival_count <= 0:
#                 raise RuntimeError("energy diagnostic requires at least two classes")
#             normalized_rival_energy = rival_energy - target_logit.new_tensor(
#                 math.log(float(rival_count))
#             )
#             energy_margin = target_logit - normalized_rival_energy

#             predictions.append(output.prediction.detach())
#             labels_rows.append(labels.detach())
#             margin_rows.append(logit_margin.detach())
#             energy_margin_rows.append(energy_margin.detach())
#             raw_energy_margin_rows.append(raw_energy_margin.detach())
#             probability_rows.append(output.probabilities[rows, target_local].detach())
#             learned_rows.append(output.learned_support[rows, target_local].detach())
#             physical_rows.append(output.physical_support[rows, target_local].detach())
#             whole_rows.append(output.whole_support[rows, target_local].detach())
#             cosine_rows.append(output.cosine_scores[rows, target_local].detach())

#     if not predictions:
#         raise RuntimeError("cannot evaluate an empty loader")

#     prediction_all = torch.cat(predictions)
#     labels_all = torch.cat(labels_rows)
#     margin_all = torch.cat(margin_rows)
#     energy_margin_all = torch.cat(energy_margin_rows)
#     raw_energy_margin_all = torch.cat(raw_energy_margin_rows)
#     probability_all = torch.cat(probability_rows)
#     learned_all = torch.cat(learned_rows)
#     physical_all = torch.cat(physical_rows)
#     whole_all = torch.cat(whole_rows)
#     cosine_all = torch.cat(cosine_rows)

#     target_tensor = torch.tensor(target_ids, device=labels_all.device, dtype=torch.long)
#     mask = torch.isin(labels_all, target_tensor)
#     labels = labels_all[mask]
#     prediction = prediction_all[mask]
#     margin = margin_all[mask]
#     energy_margin = energy_margin_all[mask]
#     raw_energy_margin = raw_energy_margin_all[mask]
#     target_probability = probability_all[mask]
#     target_learned = learned_all[mask]
#     target_physical = physical_all[mask]
#     target_whole = whole_all[mask]
#     target_cosine = cosine_all[mask]

#     per_class_accuracy: Dict[int, float] = {}
#     per_class_margin: Dict[int, float] = {}
#     per_class_energy_margin: Dict[int, float] = {}
#     per_class_raw_energy_margin: Dict[int, float] = {}
#     per_class_support: Dict[int, float] = {}
#     class_counts: Dict[int, int] = {}
#     for class_id in target_ids:
#         class_mask = labels.eq(class_id)
#         count = int(class_mask.sum().item())
#         if count <= 0:
#             raise RuntimeError(f"target class {class_id} is absent")
#         class_counts[class_id] = count
#         per_class_accuracy[class_id] = float(
#             prediction[class_mask].eq(labels[class_mask]).float().mean().item()
#         )
#         per_class_margin[class_id] = float(margin[class_mask].mean().item())
#         per_class_energy_margin[class_id] = float(
#             energy_margin[class_mask].mean().item()
#         )
#         per_class_raw_energy_margin[class_id] = float(
#             raw_energy_margin[class_mask].mean().item()
#         )
#         per_class_support[class_id] = float(target_whole[class_mask].mean().item())

#     confusion = _confusion(labels.cpu().numpy(), prediction.cpu().numpy(), ids)
#     oa = float(prediction.eq(labels).float().mean().item())
#     ba = float(sum(per_class_accuracy.values()) / len(per_class_accuracy))
#     return {
#         "class_ids": ids,
#         "target_class_ids": target_ids,
#         "overall_accuracy": oa,
#         "accuracy": oa,
#         "balanced_accuracy": ba,
#         "minimum_class_accuracy": float(min(per_class_accuracy.values())),
#         "kappa": float(_kappa(confusion)),
#         "confusion_matrix": confusion.tolist(),
#         "class_counts": class_counts,
#         "per_class_accuracy": per_class_accuracy,
#         "per_class_mean_logit_margin": per_class_margin,
#         "per_class_mean_energy_margin": per_class_energy_margin,
#         "per_class_mean_raw_energy_margin": per_class_raw_energy_margin,
#         "per_class_mean_whole_cds_support": per_class_support,
#         "mean_logit_margin": float(margin.mean().item()),
#         "mean_energy_margin": float(energy_margin.mean().item()),
#         "energy_violation_fraction": float(energy_margin.le(0).float().mean().item()),
#         "mean_raw_energy_margin": float(raw_energy_margin.mean().item()),
#         "raw_energy_violation_fraction": float(
#             raw_energy_margin.le(0).float().mean().item()
#         ),
#         "mean_target_probability": float(target_probability.mean().item()),
#         "mean_target_learned_support": float(target_learned.mean().item()),
#         "mean_target_physical_support": float(target_physical.mean().item()),
#         "mean_target_whole_cds_support": float(target_whole.mean().item()),
#         # Kept as a diagnostic compatibility key; it is not the decision score.
#         "mean_target_cosine": float(target_cosine.mean().item()),
#     }


# def summarize_evaluation_metrics(
#     metrics: Optional[Mapping[str, Any]],
# ) -> Optional[Dict[str, Any]]:
#     if metrics is None:
#         return None
#     required = (
#         "overall_accuracy",
#         "balanced_accuracy",
#         "minimum_class_accuracy",
#         "kappa",
#         "mean_logit_margin",
#         "mean_energy_margin",
#         "energy_violation_fraction",
#         "mean_target_probability",
#     )
#     missing = [name for name in required if name not in metrics]
#     if missing:
#         raise RuntimeError(f"evaluation/reporting contract mismatch: missing {missing}")
#     return {
#         "oa": float(metrics["overall_accuracy"]),
#         "ba": float(metrics["balanced_accuracy"]),
#         "min_class_accuracy": float(metrics["minimum_class_accuracy"]),
#         "kappa": float(metrics["kappa"]),
#         "logit_margin": float(metrics["mean_logit_margin"]),
#         "energy_margin": float(metrics["mean_energy_margin"]),
#         "energy_violation_fraction": float(metrics["energy_violation_fraction"]),
#         "raw_energy_margin": float(
#             metrics.get("mean_raw_energy_margin", metrics["mean_energy_margin"])
#         ),
#         "raw_energy_violation_fraction": float(
#             metrics.get(
#                 "raw_energy_violation_fraction", metrics["energy_violation_fraction"]
#             )
#         ),
#         "target_probability": float(metrics["mean_target_probability"]),
#         "learned_support": float(metrics.get("mean_target_learned_support", 0.0)),
#         "physical_support": float(metrics.get("mean_target_physical_support", 0.0)),
#         "whole_cds_support": float(metrics.get("mean_target_whole_cds_support", 0.0)),
#     }


# def _ordered_variation_statistics(variation: Tensor) -> tuple[Tensor, Tensor]:
#     """Signed V_Q/V_P interval widths; not covariance or IQR scales."""
#     value = torch.as_tensor(variation)
#     if value.ndim != 4 or value.size(1) != 2:
#         raise ValueError("signed variation must be [C,2,L,F]")
#     width = (value[:, 1] - value[:, 0]).clamp_min(0)
#     return (width > 0).to(value.dtype).mean(dim=1), width.mean(dim=1)


# @torch.no_grad()
# def class_structure_diagnostics(
#     model: Any,
#     class_ids: Sequence[int],
#     *,
#     target_names: Optional[Sequence[str]] = None,
# ) -> Dict[str, Any]:
#     ids = _class_ids(class_ids)
#     bank = model.structure_bank
#     if not isinstance(bank, HSIClassDiscriminativeStructureBank):
#         raise RuntimeError("structure_bank must be HSIClassDiscriminativeStructureBank")
#     structure = bank.class_structures(ids)
#     cds = bank.build_cds(structure)
#     relations = cds.relations
#     class_count = len(ids)
#     offdiag = ~torch.eye(
#         class_count, device=structure.learned_identity.device, dtype=torch.bool
#     )

#     learned_identity = relations.learned_identity_proximity_by_region.mean(dim=2)
#     learned_relation = relations.learned_relation_proximity
#     physical_identity = relations.physical_identity_proximity_by_band.mean(dim=2)
#     physical_relation = relations.physical_relation_proximity
#     physical_combined = relations.physical_combined_proximity
#     learned_active, learned_spread = _ordered_variation_statistics(
#         structure.learned_variation
#     )
#     physical_active, physical_spread = _ordered_variation_statistics(
#         structure.physical_variation
#     )

#     def offdiag_mean(matrix: Tensor) -> float:
#         return float(matrix[offdiag].mean().item()) if bool(offdiag.any()) else 1.0

#     if bool(offdiag.any()):
#         masked = physical_combined.masked_fill(~offdiag, -torch.inf)
#         flat = int(masked.argmax().item())
#         left, right = divmod(flat, class_count)
#         most_similar_pair = [ids[left], ids[right]]
#         maximum_physical_proximity = float(masked[left, right].item())
#     else:
#         most_similar_pair = None
#         maximum_physical_proximity = 1.0

#     names = list(target_names or [])
#     classes = []
#     for position, class_id in enumerate(ids):
#         classes.append(
#             {
#                 "class_id": class_id,
#                 "class_name": (
#                     str(names[class_id])
#                     if class_id < len(names)
#                     else f"Class {class_id + 1}"
#                 ),
#                 "learned_identity_norm": float(
#                     torch.linalg.vector_norm(
#                         structure.learned_identity[position]
#                     ).item()
#                 ),
#                 "learned_absolute_deviation_norm": float(
#                     structure.learned_absolute_deviation[position].norm().item()
#                 ),
#                 "learned_variation_norm": float(
#                     torch.linalg.vector_norm(
#                         structure.learned_variation[position]
#                     ).item()
#                 ),
#                 "physical_identity_norm": float(
#                     torch.linalg.vector_norm(
#                         structure.physical_identity[position]
#                     ).item()
#                 ),
#                 "physical_variation_norm": float(
#                     torch.linalg.vector_norm(
#                         structure.physical_variation[position]
#                     ).item()
#                 ),
#                 "learned_active_relation_fraction": float(
#                     learned_active[position].mean().item()
#                 ),
#                 "learned_positive_relation_scale": float(
#                     learned_spread[position].mean().item()
#                 ),
#                 "physical_active_relation_fraction": float(
#                     physical_active[position].mean().item()
#                 ),
#                 "physical_positive_relation_scale": float(
#                     physical_spread[position].mean().item()
#                 ),
#             }
#         )

#     return {
#         "class_ids": ids,
#         "mean_learned_identity_proximity": offdiag_mean(learned_identity),
#         "mean_learned_relation_proximity": offdiag_mean(learned_relation),
#         "mean_physical_identity_proximity": offdiag_mean(physical_identity),
#         "mean_physical_relation_proximity": offdiag_mean(physical_relation),
#         "maximum_physical_proximity": maximum_physical_proximity,
#         "most_physically_similar_class_pair": most_similar_pair,
#         "classes": classes,
#     }


# @torch.no_grad()
# def structural_decision_diagnostics(
#     model: Any, class_ids: Sequence[int]
# ) -> Dict[str, Any]:
#     ids = _class_ids(class_ids)
#     structure = model.structure_bank.class_structures(ids)
#     learned_variation = structure.learned_variation
#     active, spread = _ordered_variation_statistics(learned_variation)
#     return {
#         "class_ids": ids,
#         "classifier_parameter_count": int(model.classifier.parameter_count),
#         "classifier_rule": "single cosine classifier over Q; inference is tau*cosine(Q,W); Whole-CDS is training-time structural supervision and read-only evaluation evidence",
#         "classifier_logit_scale": float(model.classifier.logit_scale.detach().item()),
#         "classifier_beta": 0.0,
#         "mean_learned_relation_variation_norm": float(
#             torch.linalg.vector_norm(
#                 learned_variation.reshape(learned_variation.size(0), -1), dim=1
#             )
#             .mean()
#             .item()
#         ),
#         "mean_active_ordered_relation_fraction": float(active.mean().item()),
#         "mean_positive_ordered_relation_scale": float(spread.mean().item()),
#     }


# @torch.no_grad()
# def cross_phase_cds_decision_diagnostics(
#     model: Any,
#     *,
#     old_class_ids: Sequence[int],
#     new_class_ids: Sequence[int],
# ) -> Dict[str, Any]:
#     old_ids = _class_ids(old_class_ids, name="old_class_ids")
#     new_ids = _class_ids(new_class_ids, name="new_class_ids")
#     seen_ids = old_ids + new_ids
#     structure = model.structure_bank.class_structures(seen_ids)
#     cds = model.structure_bank.build_cds(structure)
#     relations = cds.relations
#     class_count = len(seen_ids)
#     old_count = len(old_ids)
#     old_mask = torch.arange(class_count, device=model.device) < old_count
#     diagonal = torch.eye(class_count, device=model.device, dtype=torch.bool)
#     cross_mask = (~diagonal) & (old_mask.unsqueeze(1) ^ old_mask.unsqueeze(0))
#     learned = relations.learned_combined_proximity
#     physical = relations.physical_combined_proximity
#     joint = combine_learned_physical_relevance(learned, physical)

#     output = model.classifier(
#         structure.learned_identity.to(device=model.device, dtype=model.dtype),
#         physical_spectral_evidence=structure.physical_identity.to(
#             device=model.device, dtype=model.dtype
#         ),
#         class_ids=structure.class_ids.to(device=model.device),
#         cds=cds,
#         trainable_class_ids=(),
#     )
#     target = torch.arange(class_count, device=model.device)
#     target_logit = output.logits[target, target]
#     rival = output.logits.clone()
#     rival[target, target] = -torch.inf
#     margin = target_logit - rival.max(dim=1).values

#     def masked_mean(matrix: Tensor, mask: Tensor) -> float:
#         return float(matrix[mask].mean().item()) if bool(mask.any()) else 0.0

#     return {
#         "old_class_ids": old_ids,
#         "new_class_ids": new_ids,
#         "cross_phase_learned_proximity_mean": masked_mean(learned, cross_mask),
#         "cross_phase_physical_proximity_mean": masked_mean(physical, cross_mask),
#         "cross_phase_joint_proximity_mean": masked_mean(joint, cross_mask),
#         "class_identity_cosine_margin_mean": float(margin.mean().item()),
#         "class_identity_cosine_margin_min": float(margin.min().item()),
#     }


# def _mapping_value(mapping: Mapping[Any, Any] | None, class_id: int) -> float:
#     if mapping is None:
#         raise RuntimeError("required per-class mapping is absent")
#     if class_id in mapping:
#         return float(mapping[class_id])
#     key = str(class_id)
#     if key in mapping:
#         return float(mapping[key])
#     raise KeyError(class_id)


# def summarize_class_group(
#     metrics: Mapping[str, Any], class_ids: Sequence[int]
# ) -> Dict[str, Any]:
#     ids = _class_ids(class_ids)
#     accuracies = [
#         _mapping_value(metrics.get("per_class_accuracy"), class_id) for class_id in ids
#     ]
#     margins = [
#         _mapping_value(metrics.get("per_class_mean_logit_margin"), class_id)
#         for class_id in ids
#     ]
#     energy_margins = [
#         _mapping_value(metrics.get("per_class_mean_energy_margin"), class_id)
#         for class_id in ids
#     ]
#     return {
#         "class_ids": ids,
#         "balanced_accuracy": float(sum(accuracies) / len(accuracies)),
#         "minimum_class_accuracy": float(min(accuracies)),
#         "mean_logit_margin": float(sum(margins) / len(margins)),
#         "mean_energy_margin": float(sum(energy_margins) / len(energy_margins)),
#     }


# def old_new_summary(
#     metrics: Mapping[str, Any],
#     *,
#     old_class_ids: Sequence[int],
#     new_class_ids: Sequence[int],
# ) -> Dict[str, Any]:
#     old = summarize_class_group(metrics, old_class_ids)
#     new = summarize_class_group(metrics, new_class_ids)
#     a = float(old["balanced_accuracy"])
#     b = float(new["balanced_accuracy"])
#     harmonic = 0.0 if a + b <= 0.0 else 2.0 * a * b / (a + b)
#     return {"old": old, "new": new, "harmonic_old_new_accuracy": harmonic}


# def phase_origin_confusion_diagnostics(
#     metrics: Mapping[str, Any],
#     *,
#     phase_to_classes: Mapping[int, Sequence[int]],
# ) -> Dict[str, Any]:
#     matrix = np.asarray(metrics["confusion_matrix"], dtype=np.int64)
#     ids = [int(value) for value in metrics["class_ids"]]
#     index = {class_id: position for position, class_id in enumerate(ids)}
#     phases = sorted(int(phase) for phase in phase_to_classes)
#     result: Dict[str, Any] = {}
#     for true_phase in phases:
#         true_ids = [
#             int(value) for value in phase_to_classes[true_phase] if int(value) in index
#         ]
#         if not true_ids:
#             continue
#         rows = [index[value] for value in true_ids]
#         row_total = int(matrix[rows, :].sum())
#         row: Dict[str, float] = {}
#         for predicted_phase in phases:
#             predicted_ids = [
#                 int(value)
#                 for value in phase_to_classes[predicted_phase]
#                 if int(value) in index
#             ]
#             columns = [index[value] for value in predicted_ids]
#             count = int(matrix[np.ix_(rows, columns)].sum()) if columns else 0
#             row[str(predicted_phase)] = (
#                 0.0 if row_total == 0 else count / float(row_total)
#             )
#         result[str(true_phase)] = row
#     return result


# @torch.no_grad()
# def whole_cds_classifier_diagnostics(
#     model: Any,
#     class_ids: Sequence[int],
#     *,
#     historical_class_count: int,
# ) -> Dict[str, float]:
#     ids = _class_ids(class_ids)
#     old_count = int(historical_class_count)
#     if old_count <= 0 or old_count >= len(ids):
#         raise ValueError("historical_class_count must split old and current classes")
#     structure = model.structure_bank.class_structures(ids)
#     cds = model.structure_bank.build_cds(structure)
#     output = model.classifier(
#         structure.learned_identity.to(device=model.device, dtype=model.dtype),
#         physical_spectral_evidence=structure.physical_identity.to(
#             device=model.device, dtype=model.dtype
#         ),
#         class_ids=structure.class_ids.to(device=model.device),
#         cds=cds,
#         trainable_class_ids=(),
#     )
#     target = torch.arange(len(ids), device=model.device)
#     target_logit = output.logits[target, target]
#     rival = output.logits.clone()
#     rival[target, target] = -torch.inf
#     margin = target_logit - rival.max(dim=1).values
#     return {
#         "whole_cds_margin_mean": float(margin.mean().item()),
#         "whole_cds_margin_min": float(margin.min().item()),
#         "learned_self_support_mean": float(
#             output.learned_support[target, target].mean().item()
#         ),
#         "physical_self_support_mean": float(
#             output.physical_support[target, target].mean().item()
#         ),
#         "whole_self_support_mean": float(
#             output.whole_support[target, target].mean().item()
#         ),
#     }


# __all__ = [
#     "evaluate_loader",
#     "summarize_evaluation_metrics",
#     "class_structure_diagnostics",
#     "structural_decision_diagnostics",
#     "cross_phase_cds_decision_diagnostics",
#     "summarize_class_group",
#     "old_new_summary",
#     "phase_origin_confusion_diagnostics",
#     "whole_cds_classifier_diagnostics",
# ]


# from __future__ import annotations
# from contextlib import contextmanager
# import math
# from numbers import Integral, Real
# from typing import Any, Dict, Mapping, Optional, Sequence

# import numpy as np
# import torch

# from models.classifier import CosineClassificationHead
# from models.geometry_bank import (
#     ClassDiscriminativeStructureState,
#     HSIClassDiscriminativeStructureBank,
#     combine_learned_physical_relevance,
#     ordered_variation_operator,
# )

# Tensor = torch.Tensor


# def _as_int(value: object, *, name: str) -> int:
#     if isinstance(value, bool):
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
#     result = [_as_int(value, name=f"{name} value") for value in values]
#     if not result or len(result) != len(set(result)):
#         raise ValueError(f"{name} must contain unique IDs")
#     return result


# @contextmanager
# def _temporary_eval_state(model: Any):
#     states = [(module, bool(module.training)) for module in model.modules()]
#     try:
#         model.eval()
#         yield
#     finally:
#         for module, state in states:
#             module.training = state


# def _unpack(
#     batch: Mapping[str, Any],
#     *,
#     model: Any,
#     device: torch.device,
# ) -> tuple[Tensor, Tensor, Tensor]:
#     required = {"image", "raw_center_spectrum", "label"}
#     missing = required - set(batch)
#     if missing:
#         raise KeyError(f"evaluation batch missing {sorted(missing)}")
#     patch = torch.as_tensor(batch["image"], device=device, dtype=model.dtype)
#     spectrum = torch.as_tensor(
#         batch["raw_center_spectrum"], device=device, dtype=model.dtype
#     )
#     raw_labels = torch.as_tensor(batch["label"])
#     if (
#         raw_labels.dtype == torch.bool
#         or raw_labels.is_complex()
#         or torch.is_floating_point(raw_labels)
#     ):
#         raise RuntimeError("evaluation labels must use an integer dtype")
#     labels = raw_labels.to(device=device, dtype=torch.long).flatten()
#     if labels.shape != (patch.size(0),):
#         raise RuntimeError("evaluation labels are misaligned")
#     return patch, spectrum, labels


# def _confusion(
#     targets: np.ndarray, predictions: np.ndarray, ids: Sequence[int]
# ) -> np.ndarray:
#     index = {class_id: row for row, class_id in enumerate(ids)}
#     matrix = np.zeros((len(ids), len(ids)), dtype=np.int64)
#     for target, prediction in zip(targets.tolist(), predictions.tolist()):
#         if int(target) not in index or int(prediction) not in index:
#             raise RuntimeError("confusion input contains a class outside class_ids")
#         matrix[index[int(target)], index[int(prediction)]] += 1
#     return matrix


# def _kappa(matrix: np.ndarray) -> float:
#     total = float(matrix.sum())
#     if total <= 0.0:
#         return 0.0
#     observed = float(np.trace(matrix)) / total
#     expected = float((matrix.sum(axis=1) * matrix.sum(axis=0)).sum()) / (total * total)
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
#     decision_cds: ClassDiscriminativeStructureState | None = None,
# ) -> Dict[str, Any]:
#     ids = _class_ids(class_ids)
#     target_ids = (
#         ids
#         if target_class_ids is None
#         else _class_ids(target_class_ids, name="target_class_ids")
#     )
#     if set(target_ids) - set(ids):
#         raise ValueError("target_class_ids must be a subset of class_ids")

#     evaluation_device = torch.device(device)
#     model.validate_model_state()
#     if not isinstance(model.classifier, CosineClassificationHead):
#         raise RuntimeError("classifier must be CosineClassificationHead")

#     predictions: list[Tensor] = []
#     labels_rows: list[Tensor] = []
#     margin_rows: list[Tensor] = []
#     energy_margin_rows: list[Tensor] = []
#     raw_energy_margin_rows: list[Tensor] = []
#     probability_rows: list[Tensor] = []
#     learned_rows: list[Tensor] = []
#     physical_rows: list[Tensor] = []
#     whole_rows: list[Tensor] = []
#     cosine_rows: list[Tensor] = []
#     allowed = torch.tensor(ids, device=evaluation_device, dtype=torch.long)

#     with _temporary_eval_state(model):
#         for batch in loader:
#             patch, spectrum, labels = _unpack(
#                 batch, model=model, device=evaluation_device
#             )
#             if not bool(torch.isin(labels, allowed).all()):
#                 raise RuntimeError(
#                     "evaluation loader contains a class outside class_ids"
#                 )
#             q = model.extract_features(
#                 patch,
#                 center_spectrum=spectrum,
#                 return_aux=False,
#             ).ordered_features
#             p = model.physical_spectral_evidence(spectrum)
#             output = model.classify_features(
#                 q,
#                 physical_spectral_evidence=p,
#                 class_ids=ids,
#                 cds=decision_cds,
#                 trainable_class_ids=(),
#             )

#             matches = labels.unsqueeze(1).eq(output.class_ids.unsqueeze(0))
#             if not bool(matches.sum(dim=1).eq(1).all()):
#                 raise RuntimeError("could not map labels to classifier columns")
#             target_local = matches.to(torch.long).argmax(dim=1)
#             rows = torch.arange(labels.numel(), device=labels.device)
#             target_logit = output.logits[rows, target_local]
#             rival_logits = output.logits.clone()
#             rival_logits[rows, target_local] = -torch.inf
#             max_rival = rival_logits.max(dim=1).values
#             logit_margin = target_logit - max_rival
#             rival_energy = torch.logsumexp(rival_logits, dim=1)
#             raw_energy_margin = target_logit - rival_energy
#             rival_count = int(output.logits.size(1)) - 1
#             if rival_count <= 0:
#                 raise RuntimeError("energy diagnostic requires at least two classes")
#             normalized_rival_energy = rival_energy - target_logit.new_tensor(
#                 math.log(float(rival_count))
#             )
#             energy_margin = target_logit - normalized_rival_energy

#             predictions.append(output.prediction.detach())
#             labels_rows.append(labels.detach())
#             margin_rows.append(logit_margin.detach())
#             energy_margin_rows.append(energy_margin.detach())
#             raw_energy_margin_rows.append(raw_energy_margin.detach())
#             probability_rows.append(output.probabilities[rows, target_local].detach())
#             learned_rows.append(output.learned_support[rows, target_local].detach())
#             physical_rows.append(output.physical_support[rows, target_local].detach())
#             whole_rows.append(output.whole_support[rows, target_local].detach())
#             cosine_rows.append(output.cosine_scores[rows, target_local].detach())

#     if not predictions:
#         raise RuntimeError("cannot evaluate an empty loader")

#     prediction_all = torch.cat(predictions)
#     labels_all = torch.cat(labels_rows)
#     margin_all = torch.cat(margin_rows)
#     energy_margin_all = torch.cat(energy_margin_rows)
#     raw_energy_margin_all = torch.cat(raw_energy_margin_rows)
#     probability_all = torch.cat(probability_rows)
#     learned_all = torch.cat(learned_rows)
#     physical_all = torch.cat(physical_rows)
#     whole_all = torch.cat(whole_rows)
#     cosine_all = torch.cat(cosine_rows)

#     target_tensor = torch.tensor(target_ids, device=labels_all.device, dtype=torch.long)
#     mask = torch.isin(labels_all, target_tensor)
#     labels = labels_all[mask]
#     prediction = prediction_all[mask]
#     margin = margin_all[mask]
#     energy_margin = energy_margin_all[mask]
#     raw_energy_margin = raw_energy_margin_all[mask]
#     target_probability = probability_all[mask]
#     target_learned = learned_all[mask]
#     target_physical = physical_all[mask]
#     target_whole = whole_all[mask]
#     target_cosine = cosine_all[mask]

#     per_class_accuracy: Dict[int, float] = {}
#     per_class_margin: Dict[int, float] = {}
#     per_class_energy_margin: Dict[int, float] = {}
#     per_class_raw_energy_margin: Dict[int, float] = {}
#     per_class_support: Dict[int, float] = {}
#     class_counts: Dict[int, int] = {}
#     for class_id in target_ids:
#         class_mask = labels.eq(class_id)
#         count = int(class_mask.sum().item())
#         if count <= 0:
#             raise RuntimeError(f"target class {class_id} is absent")
#         class_counts[class_id] = count
#         per_class_accuracy[class_id] = float(
#             prediction[class_mask].eq(labels[class_mask]).float().mean().item()
#         )
#         per_class_margin[class_id] = float(margin[class_mask].mean().item())
#         per_class_energy_margin[class_id] = float(
#             energy_margin[class_mask].mean().item()
#         )
#         per_class_raw_energy_margin[class_id] = float(
#             raw_energy_margin[class_mask].mean().item()
#         )
#         per_class_support[class_id] = float(target_whole[class_mask].mean().item())

#     confusion = _confusion(labels.cpu().numpy(), prediction.cpu().numpy(), ids)
#     oa = float(prediction.eq(labels).float().mean().item())
#     ba = float(sum(per_class_accuracy.values()) / len(per_class_accuracy))
#     return {
#         "class_ids": ids,
#         "target_class_ids": target_ids,
#         "overall_accuracy": oa,
#         "accuracy": oa,
#         "balanced_accuracy": ba,
#         "minimum_class_accuracy": float(min(per_class_accuracy.values())),
#         "kappa": float(_kappa(confusion)),
#         "confusion_matrix": confusion.tolist(),
#         "class_counts": class_counts,
#         "per_class_accuracy": per_class_accuracy,
#         "per_class_mean_logit_margin": per_class_margin,
#         "per_class_mean_energy_margin": per_class_energy_margin,
#         "per_class_mean_raw_energy_margin": per_class_raw_energy_margin,
#         "per_class_mean_whole_cds_support": per_class_support,
#         "mean_logit_margin": float(margin.mean().item()),
#         "mean_energy_margin": float(energy_margin.mean().item()),
#         "energy_violation_fraction": float(energy_margin.le(0).float().mean().item()),
#         "mean_raw_energy_margin": float(raw_energy_margin.mean().item()),
#         "raw_energy_violation_fraction": float(
#             raw_energy_margin.le(0).float().mean().item()
#         ),
#         "mean_target_probability": float(target_probability.mean().item()),
#         "mean_target_learned_support": float(target_learned.mean().item()),
#         "mean_target_physical_support": float(target_physical.mean().item()),
#         "mean_target_whole_cds_support": float(target_whole.mean().item()),
#         # Kept as a diagnostic compatibility key; it is not the decision score.
#         "mean_target_cosine": float(target_cosine.mean().item()),
#     }


# def summarize_evaluation_metrics(
#     metrics: Optional[Mapping[str, Any]],
# ) -> Optional[Dict[str, Any]]:
#     if metrics is None:
#         return None
#     required = (
#         "overall_accuracy",
#         "balanced_accuracy",
#         "minimum_class_accuracy",
#         "kappa",
#         "mean_logit_margin",
#         "mean_energy_margin",
#         "energy_violation_fraction",
#         "mean_target_probability",
#     )
#     missing = [name for name in required if name not in metrics]
#     if missing:
#         raise RuntimeError(f"evaluation/reporting contract mismatch: missing {missing}")
#     return {
#         "oa": float(metrics["overall_accuracy"]),
#         "ba": float(metrics["balanced_accuracy"]),
#         "min_class_accuracy": float(metrics["minimum_class_accuracy"]),
#         "kappa": float(metrics["kappa"]),
#         "logit_margin": float(metrics["mean_logit_margin"]),
#         "energy_margin": float(metrics["mean_energy_margin"]),
#         "energy_violation_fraction": float(metrics["energy_violation_fraction"]),
#         "raw_energy_margin": float(
#             metrics.get("mean_raw_energy_margin", metrics["mean_energy_margin"])
#         ),
#         "raw_energy_violation_fraction": float(
#             metrics.get(
#                 "raw_energy_violation_fraction", metrics["energy_violation_fraction"]
#             )
#         ),
#         "target_probability": float(metrics["mean_target_probability"]),
#         "learned_support": float(metrics.get("mean_target_learned_support", 0.0)),
#         "physical_support": float(metrics.get("mean_target_physical_support", 0.0)),
#         "whole_cds_support": float(metrics.get("mean_target_whole_cds_support", 0.0)),
#     }


# def _ordered_variation_statistics(variation: Tensor) -> tuple[Tensor, Tensor]:
#     """Signed V_Q/V_P interval widths; not covariance or IQR scales."""
#     value = torch.as_tensor(variation)
#     if value.ndim != 4 or value.size(1) != 2:
#         raise ValueError("signed variation must be [C,2,L,F]")
#     width = (value[:, 1] - value[:, 0]).clamp_min(0)
#     return (width > 0).to(value.dtype).mean(dim=1), width.mean(dim=1)


# @torch.no_grad()
# def class_structure_diagnostics(
#     model: Any,
#     class_ids: Sequence[int],
#     *,
#     target_names: Optional[Sequence[str]] = None,
# ) -> Dict[str, Any]:
#     ids = _class_ids(class_ids)
#     bank = model.structure_bank
#     if not isinstance(bank, HSIClassDiscriminativeStructureBank):
#         raise RuntimeError("structure_bank must be HSIClassDiscriminativeStructureBank")
#     structure = bank.class_structures(ids)
#     cds = bank.build_cds(structure)
#     relations = cds.relations
#     class_count = len(ids)
#     offdiag = ~torch.eye(
#         class_count, device=structure.learned_identity.device, dtype=torch.bool
#     )

#     learned_identity = relations.learned_identity_proximity_by_region.mean(dim=2)
#     learned_relation = relations.learned_relation_proximity
#     physical_identity = relations.physical_identity_proximity_by_band.mean(dim=2)
#     physical_relation = relations.physical_relation_proximity
#     physical_combined = relations.physical_combined_proximity
#     learned_active, learned_spread = _ordered_variation_statistics(
#         structure.learned_variation
#     )
#     physical_active, physical_spread = _ordered_variation_statistics(
#         structure.physical_variation
#     )

#     def offdiag_mean(matrix: Tensor) -> float:
#         return float(matrix[offdiag].mean().item()) if bool(offdiag.any()) else 1.0

#     if bool(offdiag.any()):
#         masked = physical_combined.masked_fill(~offdiag, -torch.inf)
#         flat = int(masked.argmax().item())
#         left, right = divmod(flat, class_count)
#         most_similar_pair = [ids[left], ids[right]]
#         maximum_physical_proximity = float(masked[left, right].item())
#     else:
#         most_similar_pair = None
#         maximum_physical_proximity = 1.0

#     names = list(target_names or [])
#     classes = []
#     for position, class_id in enumerate(ids):
#         classes.append(
#             {
#                 "class_id": class_id,
#                 "class_name": (
#                     str(names[class_id])
#                     if class_id < len(names)
#                     else f"Class {class_id + 1}"
#                 ),
#                 "learned_identity_norm": float(
#                     torch.linalg.vector_norm(
#                         structure.learned_identity[position]
#                     ).item()
#                 ),
#                 "learned_coupled_active_rank": int(
#                     structure.learned_coupled_active_rank[position].item()
#                 ),
#                 "learned_coupled_residual_scale": float(
#                     structure.learned_residual_scale[position].item()
#                 ),
#                 "learned_variation_norm": float(
#                     torch.linalg.vector_norm(
#                         structure.learned_variation[position]
#                     ).item()
#                 ),
#                 "physical_identity_norm": float(
#                     torch.linalg.vector_norm(
#                         structure.physical_identity[position]
#                     ).item()
#                 ),
#                 "physical_variation_norm": float(
#                     torch.linalg.vector_norm(
#                         structure.physical_variation[position]
#                     ).item()
#                 ),
#                 "learned_active_relation_fraction": float(
#                     learned_active[position].mean().item()
#                 ),
#                 "learned_positive_relation_scale": float(
#                     learned_spread[position].mean().item()
#                 ),
#                 "physical_active_relation_fraction": float(
#                     physical_active[position].mean().item()
#                 ),
#                 "physical_positive_relation_scale": float(
#                     physical_spread[position].mean().item()
#                 ),
#             }
#         )

#     return {
#         "class_ids": ids,
#         "mean_learned_identity_proximity": offdiag_mean(learned_identity),
#         "mean_learned_relation_proximity": offdiag_mean(learned_relation),
#         "mean_physical_identity_proximity": offdiag_mean(physical_identity),
#         "mean_physical_relation_proximity": offdiag_mean(physical_relation),
#         "maximum_physical_proximity": maximum_physical_proximity,
#         "most_physically_similar_class_pair": most_similar_pair,
#         "classes": classes,
#     }


# @torch.no_grad()
# def structural_decision_diagnostics(
#     model: Any, class_ids: Sequence[int]
# ) -> Dict[str, Any]:
#     ids = _class_ids(class_ids)
#     structure = model.structure_bank.class_structures(ids)
#     learned_variation = structure.learned_variation
#     active, spread = _ordered_variation_statistics(learned_variation)
#     return {
#         "class_ids": ids,
#         "classifier_parameter_count": int(model.classifier.parameter_count),
#         "classifier_rule": "single cosine classifier over Q; inference is tau*cosine(Q,W); Whole-CDS is training-time structural supervision and read-only evaluation evidence",
#         "classifier_logit_scale": float(model.classifier.logit_scale.detach().item()),
#         "classifier_beta": 0.0,
#         "mean_learned_relation_variation_norm": float(
#             torch.linalg.vector_norm(
#                 learned_variation.reshape(learned_variation.size(0), -1), dim=1
#             )
#             .mean()
#             .item()
#         ),
#         "mean_active_ordered_relation_fraction": float(active.mean().item()),
#         "mean_positive_ordered_relation_scale": float(spread.mean().item()),
#     }


# @torch.no_grad()
# def cross_phase_cds_decision_diagnostics(
#     model: Any,
#     *,
#     old_class_ids: Sequence[int],
#     new_class_ids: Sequence[int],
# ) -> Dict[str, Any]:
#     old_ids = _class_ids(old_class_ids, name="old_class_ids")
#     new_ids = _class_ids(new_class_ids, name="new_class_ids")
#     seen_ids = old_ids + new_ids
#     structure = model.structure_bank.class_structures(seen_ids)
#     cds = model.structure_bank.build_cds(structure)
#     relations = cds.relations
#     class_count = len(seen_ids)
#     old_count = len(old_ids)
#     old_mask = torch.arange(class_count, device=model.device) < old_count
#     diagonal = torch.eye(class_count, device=model.device, dtype=torch.bool)
#     cross_mask = (~diagonal) & (old_mask.unsqueeze(1) ^ old_mask.unsqueeze(0))
#     learned = relations.learned_combined_proximity
#     physical = relations.physical_combined_proximity
#     joint = combine_learned_physical_relevance(learned, physical)

#     output = model.classifier(
#         structure.learned_identity.to(device=model.device, dtype=model.dtype),
#         physical_spectral_evidence=structure.physical_identity.to(
#             device=model.device, dtype=model.dtype
#         ),
#         class_ids=structure.class_ids.to(device=model.device),
#         cds=cds,
#         trainable_class_ids=(),
#     )
#     target = torch.arange(class_count, device=model.device)
#     target_logit = output.logits[target, target]
#     rival = output.logits.clone()
#     rival[target, target] = -torch.inf
#     margin = target_logit - rival.max(dim=1).values

#     def masked_mean(matrix: Tensor, mask: Tensor) -> float:
#         return float(matrix[mask].mean().item()) if bool(mask.any()) else 0.0

#     return {
#         "old_class_ids": old_ids,
#         "new_class_ids": new_ids,
#         "cross_phase_learned_proximity_mean": masked_mean(learned, cross_mask),
#         "cross_phase_physical_proximity_mean": masked_mean(physical, cross_mask),
#         "cross_phase_joint_proximity_mean": masked_mean(joint, cross_mask),
#         "class_identity_cosine_margin_mean": float(margin.mean().item()),
#         "class_identity_cosine_margin_min": float(margin.min().item()),
#     }


# def _mapping_value(mapping: Mapping[Any, Any] | None, class_id: int) -> float:
#     if mapping is None:
#         raise RuntimeError("required per-class mapping is absent")
#     if class_id in mapping:
#         return float(mapping[class_id])
#     key = str(class_id)
#     if key in mapping:
#         return float(mapping[key])
#     raise KeyError(class_id)


# def summarize_class_group(
#     metrics: Mapping[str, Any], class_ids: Sequence[int]
# ) -> Dict[str, Any]:
#     ids = _class_ids(class_ids)
#     accuracies = [
#         _mapping_value(metrics.get("per_class_accuracy"), class_id) for class_id in ids
#     ]
#     margins = [
#         _mapping_value(metrics.get("per_class_mean_logit_margin"), class_id)
#         for class_id in ids
#     ]
#     energy_margins = [
#         _mapping_value(metrics.get("per_class_mean_energy_margin"), class_id)
#         for class_id in ids
#     ]
#     return {
#         "class_ids": ids,
#         "balanced_accuracy": float(sum(accuracies) / len(accuracies)),
#         "minimum_class_accuracy": float(min(accuracies)),
#         "mean_logit_margin": float(sum(margins) / len(margins)),
#         "mean_energy_margin": float(sum(energy_margins) / len(energy_margins)),
#     }


# def old_new_summary(
#     metrics: Mapping[str, Any],
#     *,
#     old_class_ids: Sequence[int],
#     new_class_ids: Sequence[int],
# ) -> Dict[str, Any]:
#     old = summarize_class_group(metrics, old_class_ids)
#     new = summarize_class_group(metrics, new_class_ids)
#     a = float(old["balanced_accuracy"])
#     b = float(new["balanced_accuracy"])
#     harmonic = 0.0 if a + b <= 0.0 else 2.0 * a * b / (a + b)
#     return {"old": old, "new": new, "harmonic_old_new_accuracy": harmonic}


# def phase_origin_confusion_diagnostics(
#     metrics: Mapping[str, Any],
#     *,
#     phase_to_classes: Mapping[int, Sequence[int]],
# ) -> Dict[str, Any]:
#     matrix = np.asarray(metrics["confusion_matrix"], dtype=np.int64)
#     ids = [int(value) for value in metrics["class_ids"]]
#     index = {class_id: position for position, class_id in enumerate(ids)}
#     phases = sorted(int(phase) for phase in phase_to_classes)
#     result: Dict[str, Any] = {}
#     for true_phase in phases:
#         true_ids = [
#             int(value) for value in phase_to_classes[true_phase] if int(value) in index
#         ]
#         if not true_ids:
#             continue
#         rows = [index[value] for value in true_ids]
#         row_total = int(matrix[rows, :].sum())
#         row: Dict[str, float] = {}
#         for predicted_phase in phases:
#             predicted_ids = [
#                 int(value)
#                 for value in phase_to_classes[predicted_phase]
#                 if int(value) in index
#             ]
#             columns = [index[value] for value in predicted_ids]
#             count = int(matrix[np.ix_(rows, columns)].sum()) if columns else 0
#             row[str(predicted_phase)] = (
#                 0.0 if row_total == 0 else count / float(row_total)
#             )
#         result[str(true_phase)] = row
#     return result


# @torch.no_grad()
# def whole_cds_classifier_diagnostics(
#     model: Any,
#     class_ids: Sequence[int],
#     *,
#     historical_class_count: int,
# ) -> Dict[str, float]:
#     ids = _class_ids(class_ids)
#     old_count = int(historical_class_count)
#     if old_count <= 0 or old_count >= len(ids):
#         raise ValueError("historical_class_count must split old and current classes")
#     structure = model.structure_bank.class_structures(ids)
#     cds = model.structure_bank.build_cds(structure)
#     output = model.classifier(
#         structure.learned_identity.to(device=model.device, dtype=model.dtype),
#         physical_spectral_evidence=structure.physical_identity.to(
#             device=model.device, dtype=model.dtype
#         ),
#         class_ids=structure.class_ids.to(device=model.device),
#         cds=cds,
#         trainable_class_ids=(),
#     )
#     target = torch.arange(len(ids), device=model.device)
#     target_logit = output.logits[target, target]
#     rival = output.logits.clone()
#     rival[target, target] = -torch.inf
#     margin = target_logit - rival.max(dim=1).values
#     return {
#         "whole_cds_margin_mean": float(margin.mean().item()),
#         "whole_cds_margin_min": float(margin.min().item()),
#         "learned_self_support_mean": float(
#             output.learned_support[target, target].mean().item()
#         ),
#         "physical_self_support_mean": float(
#             output.physical_support[target, target].mean().item()
#         ),
#         "whole_self_support_mean": float(
#             output.whole_support[target, target].mean().item()
#         ),
#     }


# __all__ = [
#     "evaluate_loader",
#     "summarize_evaluation_metrics",
#     "class_structure_diagnostics",
#     "structural_decision_diagnostics",
#     "cross_phase_cds_decision_diagnostics",
#     "summarize_class_group",
#     "old_new_summary",
#     "phase_origin_confusion_diagnostics",
#     "whole_cds_classifier_diagnostics",
# ]


# from __future__ import annotations
# from contextlib import contextmanager
# import math
# from numbers import Integral, Real
# from typing import Any, Dict, Mapping, Optional, Sequence

# import numpy as np
# import torch

# from models.classifier import CosineClassificationHead
# from models.geometry_bank import (
#     ClassDiscriminativeStructureState,
#     HSIClassDiscriminativeStructureBank,
#     combine_learned_physical_relevance,
#     ordered_variation_operator,
# )

# Tensor = torch.Tensor


# def _as_int(value: object, *, name: str) -> int:
#     if isinstance(value, bool):
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
#     result = [_as_int(value, name=f"{name} value") for value in values]
#     if not result or len(result) != len(set(result)):
#         raise ValueError(f"{name} must contain unique IDs")
#     return result


# @contextmanager
# def _temporary_eval_state(model: Any):
#     states = [(module, bool(module.training)) for module in model.modules()]
#     try:
#         model.eval()
#         yield
#     finally:
#         for module, state in states:
#             module.training = state


# def _unpack(
#     batch: Mapping[str, Any],
#     *,
#     model: Any,
#     device: torch.device,
# ) -> tuple[Tensor, Tensor, Tensor]:
#     required = {"image", "raw_center_spectrum", "label"}
#     missing = required - set(batch)
#     if missing:
#         raise KeyError(f"evaluation batch missing {sorted(missing)}")
#     patch = torch.as_tensor(batch["image"], device=device, dtype=model.dtype)
#     spectrum = torch.as_tensor(
#         batch["raw_center_spectrum"], device=device, dtype=model.dtype
#     )
#     raw_labels = torch.as_tensor(batch["label"])
#     if (
#         raw_labels.dtype == torch.bool
#         or raw_labels.is_complex()
#         or torch.is_floating_point(raw_labels)
#     ):
#         raise RuntimeError("evaluation labels must use an integer dtype")
#     labels = raw_labels.to(device=device, dtype=torch.long).flatten()
#     if labels.shape != (patch.size(0),):
#         raise RuntimeError("evaluation labels are misaligned")
#     return patch, spectrum, labels


# def _confusion(
#     targets: np.ndarray, predictions: np.ndarray, ids: Sequence[int]
# ) -> np.ndarray:
#     index = {class_id: row for row, class_id in enumerate(ids)}
#     matrix = np.zeros((len(ids), len(ids)), dtype=np.int64)
#     for target, prediction in zip(targets.tolist(), predictions.tolist()):
#         if int(target) not in index or int(prediction) not in index:
#             raise RuntimeError("confusion input contains a class outside class_ids")
#         matrix[index[int(target)], index[int(prediction)]] += 1
#     return matrix


# def _kappa(matrix: np.ndarray) -> float:
#     total = float(matrix.sum())
#     if total <= 0.0:
#         return 0.0
#     observed = float(np.trace(matrix)) / total
#     expected = float((matrix.sum(axis=1) * matrix.sum(axis=0)).sum()) / (total * total)
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
#     decision_cds: ClassDiscriminativeStructureState | None = None,
# ) -> Dict[str, Any]:
#     ids = _class_ids(class_ids)
#     target_ids = (
#         ids
#         if target_class_ids is None
#         else _class_ids(target_class_ids, name="target_class_ids")
#     )
#     if set(target_ids) - set(ids):
#         raise ValueError("target_class_ids must be a subset of class_ids")

#     evaluation_device = torch.device(device)
#     model.validate_model_state()
#     if not isinstance(model.classifier, CosineClassificationHead):
#         raise RuntimeError("classifier must be CosineClassificationHead")

#     predictions: list[Tensor] = []
#     labels_rows: list[Tensor] = []
#     margin_rows: list[Tensor] = []
#     energy_margin_rows: list[Tensor] = []
#     raw_energy_margin_rows: list[Tensor] = []
#     probability_rows: list[Tensor] = []
#     learned_rows: list[Tensor] = []
#     physical_rows: list[Tensor] = []
#     whole_rows: list[Tensor] = []
#     cosine_rows: list[Tensor] = []
#     allowed = torch.tensor(ids, device=evaluation_device, dtype=torch.long)

#     with _temporary_eval_state(model):
#         for batch in loader:
#             patch, spectrum, labels = _unpack(
#                 batch, model=model, device=evaluation_device
#             )
#             if not bool(torch.isin(labels, allowed).all()):
#                 raise RuntimeError(
#                     "evaluation loader contains a class outside class_ids"
#                 )
#             q = model.extract_features(
#                 patch,
#                 center_spectrum=spectrum,
#                 return_aux=False,
#             ).ordered_features
#             p = model.physical_spectral_evidence(spectrum)
#             output = model.classify_features(
#                 q,
#                 physical_spectral_evidence=p,
#                 class_ids=ids,
#                 cds=decision_cds,
#                 trainable_class_ids=(),
#             )

#             matches = labels.unsqueeze(1).eq(output.class_ids.unsqueeze(0))
#             if not bool(matches.sum(dim=1).eq(1).all()):
#                 raise RuntimeError("could not map labels to classifier columns")
#             target_local = matches.to(torch.long).argmax(dim=1)
#             rows = torch.arange(labels.numel(), device=labels.device)
#             target_logit = output.logits[rows, target_local]
#             rival_logits = output.logits.clone()
#             rival_logits[rows, target_local] = -torch.inf
#             max_rival = rival_logits.max(dim=1).values
#             logit_margin = target_logit - max_rival
#             rival_energy = torch.logsumexp(rival_logits, dim=1)
#             raw_energy_margin = target_logit - rival_energy
#             rival_count = int(output.logits.size(1)) - 1
#             if rival_count <= 0:
#                 raise RuntimeError("energy diagnostic requires at least two classes")
#             normalized_rival_energy = rival_energy - target_logit.new_tensor(
#                 math.log(float(rival_count))
#             )
#             energy_margin = target_logit - normalized_rival_energy

#             predictions.append(output.prediction.detach())
#             labels_rows.append(labels.detach())
#             margin_rows.append(logit_margin.detach())
#             energy_margin_rows.append(energy_margin.detach())
#             raw_energy_margin_rows.append(raw_energy_margin.detach())
#             probability_rows.append(output.probabilities[rows, target_local].detach())
#             learned_rows.append(output.learned_support[rows, target_local].detach())
#             physical_rows.append(output.physical_support[rows, target_local].detach())
#             whole_rows.append(output.whole_support[rows, target_local].detach())
#             cosine_rows.append(output.cosine_scores[rows, target_local].detach())

#     if not predictions:
#         raise RuntimeError("cannot evaluate an empty loader")

#     prediction_all = torch.cat(predictions)
#     labels_all = torch.cat(labels_rows)
#     margin_all = torch.cat(margin_rows)
#     energy_margin_all = torch.cat(energy_margin_rows)
#     raw_energy_margin_all = torch.cat(raw_energy_margin_rows)
#     probability_all = torch.cat(probability_rows)
#     learned_all = torch.cat(learned_rows)
#     physical_all = torch.cat(physical_rows)
#     whole_all = torch.cat(whole_rows)
#     cosine_all = torch.cat(cosine_rows)

#     target_tensor = torch.tensor(target_ids, device=labels_all.device, dtype=torch.long)
#     mask = torch.isin(labels_all, target_tensor)
#     labels = labels_all[mask]
#     prediction = prediction_all[mask]
#     margin = margin_all[mask]
#     energy_margin = energy_margin_all[mask]
#     raw_energy_margin = raw_energy_margin_all[mask]
#     target_probability = probability_all[mask]
#     target_learned = learned_all[mask]
#     target_physical = physical_all[mask]
#     target_whole = whole_all[mask]
#     target_cosine = cosine_all[mask]

#     per_class_accuracy: Dict[int, float] = {}
#     per_class_margin: Dict[int, float] = {}
#     per_class_energy_margin: Dict[int, float] = {}
#     per_class_raw_energy_margin: Dict[int, float] = {}
#     per_class_support: Dict[int, float] = {}
#     class_counts: Dict[int, int] = {}
#     for class_id in target_ids:
#         class_mask = labels.eq(class_id)
#         count = int(class_mask.sum().item())
#         if count <= 0:
#             raise RuntimeError(f"target class {class_id} is absent")
#         class_counts[class_id] = count
#         per_class_accuracy[class_id] = float(
#             prediction[class_mask].eq(labels[class_mask]).float().mean().item()
#         )
#         per_class_margin[class_id] = float(margin[class_mask].mean().item())
#         per_class_energy_margin[class_id] = float(
#             energy_margin[class_mask].mean().item()
#         )
#         per_class_raw_energy_margin[class_id] = float(
#             raw_energy_margin[class_mask].mean().item()
#         )
#         per_class_support[class_id] = float(target_whole[class_mask].mean().item())

#     confusion = _confusion(labels.cpu().numpy(), prediction.cpu().numpy(), ids)
#     oa = float(prediction.eq(labels).float().mean().item())
#     ba = float(sum(per_class_accuracy.values()) / len(per_class_accuracy))
#     return {
#         "class_ids": ids,
#         "target_class_ids": target_ids,
#         "overall_accuracy": oa,
#         "accuracy": oa,
#         "balanced_accuracy": ba,
#         "minimum_class_accuracy": float(min(per_class_accuracy.values())),
#         "kappa": float(_kappa(confusion)),
#         "confusion_matrix": confusion.tolist(),
#         "class_counts": class_counts,
#         "per_class_accuracy": per_class_accuracy,
#         "per_class_mean_logit_margin": per_class_margin,
#         "per_class_mean_energy_margin": per_class_energy_margin,
#         "per_class_mean_raw_energy_margin": per_class_raw_energy_margin,
#         "per_class_mean_whole_cds_support": per_class_support,
#         "mean_logit_margin": float(margin.mean().item()),
#         "mean_energy_margin": float(energy_margin.mean().item()),
#         "energy_violation_fraction": float(energy_margin.le(0).float().mean().item()),
#         "mean_raw_energy_margin": float(raw_energy_margin.mean().item()),
#         "raw_energy_violation_fraction": float(
#             raw_energy_margin.le(0).float().mean().item()
#         ),
#         "mean_target_probability": float(target_probability.mean().item()),
#         "mean_target_learned_support": float(target_learned.mean().item()),
#         "mean_target_physical_support": float(target_physical.mean().item()),
#         "mean_target_whole_cds_support": float(target_whole.mean().item()),
#         # Kept as a diagnostic compatibility key; it is not the decision score.
#         "mean_target_cosine": float(target_cosine.mean().item()),
#     }


# def summarize_evaluation_metrics(
#     metrics: Optional[Mapping[str, Any]],
# ) -> Optional[Dict[str, Any]]:
#     if metrics is None:
#         return None
#     required = (
#         "overall_accuracy",
#         "balanced_accuracy",
#         "minimum_class_accuracy",
#         "kappa",
#         "mean_logit_margin",
#         "mean_energy_margin",
#         "energy_violation_fraction",
#         "mean_target_probability",
#     )
#     missing = [name for name in required if name not in metrics]
#     if missing:
#         raise RuntimeError(f"evaluation/reporting contract mismatch: missing {missing}")
#     return {
#         "oa": float(metrics["overall_accuracy"]),
#         "ba": float(metrics["balanced_accuracy"]),
#         "min_class_accuracy": float(metrics["minimum_class_accuracy"]),
#         "kappa": float(metrics["kappa"]),
#         "logit_margin": float(metrics["mean_logit_margin"]),
#         "energy_margin": float(metrics["mean_energy_margin"]),
#         "energy_violation_fraction": float(metrics["energy_violation_fraction"]),
#         "raw_energy_margin": float(
#             metrics.get("mean_raw_energy_margin", metrics["mean_energy_margin"])
#         ),
#         "raw_energy_violation_fraction": float(
#             metrics.get(
#                 "raw_energy_violation_fraction", metrics["energy_violation_fraction"]
#             )
#         ),
#         "target_probability": float(metrics["mean_target_probability"]),
#         "learned_support": float(metrics.get("mean_target_learned_support", 0.0)),
#         "physical_support": float(metrics.get("mean_target_physical_support", 0.0)),
#         "whole_cds_support": float(metrics.get("mean_target_whole_cds_support", 0.0)),
#     }


# def _ordered_variation_statistics(variation: Tensor) -> tuple[Tensor, Tensor]:
#     """Signed V_Q/V_P interval widths; not covariance or IQR scales."""
#     value = torch.as_tensor(variation)
#     if value.ndim != 4 or value.size(1) != 2:
#         raise ValueError("signed variation must be [C,2,L,F]")
#     width = (value[:, 1] - value[:, 0]).clamp_min(0)
#     return (width > 0).to(value.dtype).mean(dim=1), width.mean(dim=1)


# @torch.no_grad()
# def class_structure_diagnostics(
#     model: Any,
#     class_ids: Sequence[int],
#     *,
#     target_names: Optional[Sequence[str]] = None,
# ) -> Dict[str, Any]:
#     ids = _class_ids(class_ids)
#     bank = model.structure_bank
#     if not isinstance(bank, HSIClassDiscriminativeStructureBank):
#         raise RuntimeError("structure_bank must be HSIClassDiscriminativeStructureBank")
#     structure = bank.class_structures(ids)
#     cds = bank.build_cds(structure)
#     relations = cds.relations
#     class_count = len(ids)
#     offdiag = ~torch.eye(
#         class_count, device=structure.learned_identity.device, dtype=torch.bool
#     )

#     learned_identity = relations.learned_identity_proximity_by_region.mean(dim=2)
#     learned_relation = relations.learned_relation_proximity
#     physical_identity = relations.physical_identity_proximity_by_band.mean(dim=2)
#     physical_relation = relations.physical_relation_proximity
#     physical_combined = relations.physical_combined_proximity
#     learned_active, learned_spread = _ordered_variation_statistics(
#         structure.learned_variation
#     )
#     physical_active, physical_spread = _ordered_variation_statistics(
#         structure.physical_variation
#     )

#     def offdiag_mean(matrix: Tensor) -> float:
#         return float(matrix[offdiag].mean().item()) if bool(offdiag.any()) else 1.0

#     if bool(offdiag.any()):
#         masked = physical_combined.masked_fill(~offdiag, -torch.inf)
#         flat = int(masked.argmax().item())
#         left, right = divmod(flat, class_count)
#         most_similar_pair = [ids[left], ids[right]]
#         maximum_physical_proximity = float(masked[left, right].item())
#     else:
#         most_similar_pair = None
#         maximum_physical_proximity = 1.0

#     names = list(target_names or [])
#     classes = []
#     for position, class_id in enumerate(ids):
#         classes.append(
#             {
#                 "class_id": class_id,
#                 "class_name": (
#                     str(names[class_id])
#                     if class_id < len(names)
#                     else f"Class {class_id + 1}"
#                 ),
#                 "learned_identity_norm": float(
#                     torch.linalg.vector_norm(
#                         structure.learned_identity[position]
#                     ).item()
#                 ),
#                 "learned_coupled_active_rank": int(
#                     structure.learned_coupled_active_rank[position].item()
#                 ),
#                 "learned_coupled_residual_scale": float(
#                     structure.learned_residual_scale[position].item()
#                 ),
#                 "learned_variation_norm": float(
#                     torch.linalg.vector_norm(
#                         structure.learned_variation[position]
#                     ).item()
#                 ),
#                 "physical_identity_norm": float(
#                     torch.linalg.vector_norm(
#                         structure.physical_identity[position]
#                     ).item()
#                 ),
#                 "physical_variation_norm": float(
#                     torch.linalg.vector_norm(
#                         structure.physical_variation[position]
#                     ).item()
#                 ),
#                 "learned_active_relation_fraction": float(
#                     learned_active[position].mean().item()
#                 ),
#                 "learned_positive_relation_scale": float(
#                     learned_spread[position].mean().item()
#                 ),
#                 "physical_active_relation_fraction": float(
#                     physical_active[position].mean().item()
#                 ),
#                 "physical_positive_relation_scale": float(
#                     physical_spread[position].mean().item()
#                 ),
#             }
#         )

#     return {
#         "class_ids": ids,
#         "mean_learned_identity_proximity": offdiag_mean(learned_identity),
#         "mean_learned_relation_proximity": offdiag_mean(learned_relation),
#         "mean_physical_identity_proximity": offdiag_mean(physical_identity),
#         "mean_physical_relation_proximity": offdiag_mean(physical_relation),
#         "maximum_physical_proximity": maximum_physical_proximity,
#         "most_physically_similar_class_pair": most_similar_pair,
#         "classes": classes,
#     }


# @torch.no_grad()
# def structural_decision_diagnostics(
#     model: Any, class_ids: Sequence[int]
# ) -> Dict[str, Any]:
#     ids = _class_ids(class_ids)
#     structure = model.structure_bank.class_structures(ids)
#     learned_variation = structure.learned_variation
#     active, spread = _ordered_variation_statistics(learned_variation)
#     return {
#         "class_ids": ids,
#         "classifier_parameter_count": int(model.classifier.parameter_count),
#         "classifier_rule": "single cosine classifier over Q; inference is tau*cosine(Q,W); Whole-CDS is training-time structural supervision and read-only evaluation evidence",
#         "classifier_logit_scale": float(model.classifier.logit_scale.detach().item()),
#         "classifier_beta": 0.0,
#         "mean_learned_relation_variation_norm": float(
#             torch.linalg.vector_norm(
#                 learned_variation.reshape(learned_variation.size(0), -1), dim=1
#             )
#             .mean()
#             .item()
#         ),
#         "mean_active_ordered_relation_fraction": float(active.mean().item()),
#         "mean_positive_ordered_relation_scale": float(spread.mean().item()),
#     }


# @torch.no_grad()
# def cross_phase_cds_decision_diagnostics(
#     model: Any,
#     *,
#     old_class_ids: Sequence[int],
#     new_class_ids: Sequence[int],
# ) -> Dict[str, Any]:
#     old_ids = _class_ids(old_class_ids, name="old_class_ids")
#     new_ids = _class_ids(new_class_ids, name="new_class_ids")
#     seen_ids = old_ids + new_ids
#     structure = model.structure_bank.class_structures(seen_ids)
#     cds = model.structure_bank.build_cds(structure)
#     relations = cds.relations
#     class_count = len(seen_ids)
#     old_count = len(old_ids)
#     old_mask = torch.arange(class_count, device=model.device) < old_count
#     diagonal = torch.eye(class_count, device=model.device, dtype=torch.bool)
#     cross_mask = (~diagonal) & (old_mask.unsqueeze(1) ^ old_mask.unsqueeze(0))
#     learned = relations.learned_combined_proximity
#     physical = relations.physical_combined_proximity
#     joint = combine_learned_physical_relevance(learned, physical)

#     output = model.classifier(
#         structure.learned_identity.to(device=model.device, dtype=model.dtype),
#         physical_spectral_evidence=structure.physical_identity.to(
#             device=model.device, dtype=model.dtype
#         ),
#         class_ids=structure.class_ids.to(device=model.device),
#         cds=cds,
#         trainable_class_ids=(),
#     )
#     target = torch.arange(class_count, device=model.device)
#     target_logit = output.logits[target, target]
#     rival = output.logits.clone()
#     rival[target, target] = -torch.inf
#     margin = target_logit - rival.max(dim=1).values

#     def masked_mean(matrix: Tensor, mask: Tensor) -> float:
#         return float(matrix[mask].mean().item()) if bool(mask.any()) else 0.0

#     return {
#         "old_class_ids": old_ids,
#         "new_class_ids": new_ids,
#         "cross_phase_learned_proximity_mean": masked_mean(learned, cross_mask),
#         "cross_phase_physical_proximity_mean": masked_mean(physical, cross_mask),
#         "cross_phase_joint_proximity_mean": masked_mean(joint, cross_mask),
#         "class_identity_cosine_margin_mean": float(margin.mean().item()),
#         "class_identity_cosine_margin_min": float(margin.min().item()),
#     }


# def _mapping_value(mapping: Mapping[Any, Any] | None, class_id: int) -> float:
#     if mapping is None:
#         raise RuntimeError("required per-class mapping is absent")
#     if class_id in mapping:
#         return float(mapping[class_id])
#     key = str(class_id)
#     if key in mapping:
#         return float(mapping[key])
#     raise KeyError(class_id)


# def summarize_class_group(
#     metrics: Mapping[str, Any], class_ids: Sequence[int]
# ) -> Dict[str, Any]:
#     ids = _class_ids(class_ids)
#     accuracies = [
#         _mapping_value(metrics.get("per_class_accuracy"), class_id) for class_id in ids
#     ]
#     margins = [
#         _mapping_value(metrics.get("per_class_mean_logit_margin"), class_id)
#         for class_id in ids
#     ]
#     energy_margins = [
#         _mapping_value(metrics.get("per_class_mean_energy_margin"), class_id)
#         for class_id in ids
#     ]
#     return {
#         "class_ids": ids,
#         "balanced_accuracy": float(sum(accuracies) / len(accuracies)),
#         "minimum_class_accuracy": float(min(accuracies)),
#         "mean_logit_margin": float(sum(margins) / len(margins)),
#         "mean_energy_margin": float(sum(energy_margins) / len(energy_margins)),
#     }


# def old_new_summary(
#     metrics: Mapping[str, Any],
#     *,
#     old_class_ids: Sequence[int],
#     new_class_ids: Sequence[int],
# ) -> Dict[str, Any]:
#     old = summarize_class_group(metrics, old_class_ids)
#     new = summarize_class_group(metrics, new_class_ids)
#     a = float(old["balanced_accuracy"])
#     b = float(new["balanced_accuracy"])
#     harmonic = 0.0 if a + b <= 0.0 else 2.0 * a * b / (a + b)
#     return {"old": old, "new": new, "harmonic_old_new_accuracy": harmonic}


# def phase_origin_confusion_diagnostics(
#     metrics: Mapping[str, Any],
#     *,
#     phase_to_classes: Mapping[int, Sequence[int]],
# ) -> Dict[str, Any]:
#     matrix = np.asarray(metrics["confusion_matrix"], dtype=np.int64)
#     ids = [int(value) for value in metrics["class_ids"]]
#     index = {class_id: position for position, class_id in enumerate(ids)}
#     phases = sorted(int(phase) for phase in phase_to_classes)
#     result: Dict[str, Any] = {}
#     for true_phase in phases:
#         true_ids = [
#             int(value) for value in phase_to_classes[true_phase] if int(value) in index
#         ]
#         if not true_ids:
#             continue
#         rows = [index[value] for value in true_ids]
#         row_total = int(matrix[rows, :].sum())
#         row: Dict[str, float] = {}
#         for predicted_phase in phases:
#             predicted_ids = [
#                 int(value)
#                 for value in phase_to_classes[predicted_phase]
#                 if int(value) in index
#             ]
#             columns = [index[value] for value in predicted_ids]
#             count = int(matrix[np.ix_(rows, columns)].sum()) if columns else 0
#             row[str(predicted_phase)] = (
#                 0.0 if row_total == 0 else count / float(row_total)
#             )
#         result[str(true_phase)] = row
#     return result


# @torch.no_grad()
# def whole_cds_classifier_diagnostics(
#     model: Any,
#     class_ids: Sequence[int],
#     *,
#     historical_class_count: int,
# ) -> Dict[str, float]:
#     ids = _class_ids(class_ids)
#     old_count = int(historical_class_count)
#     if old_count <= 0 or old_count >= len(ids):
#         raise ValueError("historical_class_count must split old and current classes")
#     structure = model.structure_bank.class_structures(ids)
#     cds = model.structure_bank.build_cds(structure)
#     output = model.classifier(
#         structure.learned_identity.to(device=model.device, dtype=model.dtype),
#         physical_spectral_evidence=structure.physical_identity.to(
#             device=model.device, dtype=model.dtype
#         ),
#         class_ids=structure.class_ids.to(device=model.device),
#         cds=cds,
#         trainable_class_ids=(),
#     )
#     target = torch.arange(len(ids), device=model.device)
#     target_logit = output.logits[target, target]
#     rival = output.logits.clone()
#     rival[target, target] = -torch.inf
#     margin = target_logit - rival.max(dim=1).values
#     return {
#         "whole_cds_margin_mean": float(margin.mean().item()),
#         "whole_cds_margin_min": float(margin.min().item()),
#         "learned_self_support_mean": float(
#             output.learned_support[target, target].mean().item()
#         ),
#         "physical_self_support_mean": float(
#             output.physical_support[target, target].mean().item()
#         ),
#         "whole_self_support_mean": float(
#             output.whole_support[target, target].mean().item()
#         ),
#     }


# __all__ = [
#     "evaluate_loader",
#     "summarize_evaluation_metrics",
#     "class_structure_diagnostics",
#     "structural_decision_diagnostics",
#     "cross_phase_cds_decision_diagnostics",
#     "summarize_class_group",
#     "old_new_summary",
#     "phase_origin_confusion_diagnostics",
#     "whole_cds_classifier_diagnostics",
# ]
