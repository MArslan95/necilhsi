from __future__ import annotations

"""Read-only evaluation for the Whole-CDS NECIL-HSI decision rule."""

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
    combine_learned_physical_relevance,
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
    if raw_labels.dtype == torch.bool or raw_labels.is_complex() or torch.is_floating_point(raw_labels):
        raise RuntimeError("evaluation labels must use an integer dtype")
    labels = raw_labels.to(device=device, dtype=torch.long).flatten()
    if labels.shape != (patch.size(0),):
        raise RuntimeError("evaluation labels are misaligned")
    return patch, spectrum, labels


def _confusion(targets: np.ndarray, predictions: np.ndarray, ids: Sequence[int]) -> np.ndarray:
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
    return 0.0 if abs(denominator) <= np.finfo(np.float64).eps else (observed - expected) / denominator


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
    target_ids = ids if target_class_ids is None else _class_ids(target_class_ids, name="target_class_ids")
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
    allowed = torch.tensor(ids, device=evaluation_device, dtype=torch.long)

    with _temporary_eval_state(model):
        for batch in loader:
            patch, spectrum, labels = _unpack(batch, model=model, device=evaluation_device)
            if not bool(torch.isin(labels, allowed).all()):
                raise RuntimeError("evaluation loader contains a class outside class_ids")
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
            target_logit = output.logits[rows, target_local]
            rival_logits = output.logits.clone()
            rival_logits[rows, target_local] = -torch.inf
            max_rival = rival_logits.max(dim=1).values
            logit_margin = target_logit - max_rival
            rival_energy = torch.logsumexp(rival_logits, dim=1)
            raw_energy_margin = target_logit - rival_energy
            rival_count = int(output.logits.size(1)) - 1
            if rival_count <= 0:
                raise RuntimeError("energy diagnostic requires at least two classes")
            normalized_rival_energy = rival_energy - target_logit.new_tensor(
                math.log(float(rival_count))
            )
            energy_margin = target_logit - normalized_rival_energy

            predictions.append(output.prediction.detach())
            labels_rows.append(labels.detach())
            margin_rows.append(logit_margin.detach())
            energy_margin_rows.append(energy_margin.detach())
            raw_energy_margin_rows.append(raw_energy_margin.detach())
            probability_rows.append(output.probabilities[rows, target_local].detach())
            learned_rows.append(output.learned_support[rows, target_local].detach())
            physical_rows.append(output.physical_support[rows, target_local].detach())
            whole_rows.append(output.whole_support[rows, target_local].detach())
            cosine_rows.append(output.cosine_scores[rows, target_local].detach())

    if not predictions:
        raise RuntimeError("cannot evaluate an empty loader")

    prediction_all = torch.cat(predictions)
    labels_all = torch.cat(labels_rows)
    margin_all = torch.cat(margin_rows)
    energy_margin_all = torch.cat(energy_margin_rows)
    raw_energy_margin_all = torch.cat(raw_energy_margin_rows)
    probability_all = torch.cat(probability_rows)
    learned_all = torch.cat(learned_rows)
    physical_all = torch.cat(physical_rows)
    whole_all = torch.cat(whole_rows)
    cosine_all = torch.cat(cosine_rows)

    target_tensor = torch.tensor(target_ids, device=labels_all.device, dtype=torch.long)
    mask = torch.isin(labels_all, target_tensor)
    labels = labels_all[mask]
    prediction = prediction_all[mask]
    margin = margin_all[mask]
    energy_margin = energy_margin_all[mask]
    raw_energy_margin = raw_energy_margin_all[mask]
    target_probability = probability_all[mask]
    target_learned = learned_all[mask]
    target_physical = physical_all[mask]
    target_whole = whole_all[mask]
    target_cosine = cosine_all[mask]

    per_class_accuracy: Dict[int, float] = {}
    per_class_margin: Dict[int, float] = {}
    per_class_energy_margin: Dict[int, float] = {}
    per_class_raw_energy_margin: Dict[int, float] = {}
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
        per_class_margin[class_id] = float(margin[class_mask].mean().item())
        per_class_energy_margin[class_id] = float(energy_margin[class_mask].mean().item())
        per_class_raw_energy_margin[class_id] = float(raw_energy_margin[class_mask].mean().item())
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
        "per_class_mean_logit_margin": per_class_margin,
        "per_class_mean_energy_margin": per_class_energy_margin,
        "per_class_mean_raw_energy_margin": per_class_raw_energy_margin,
        "per_class_mean_whole_cds_support": per_class_support,
        "mean_logit_margin": float(margin.mean().item()),
        "mean_energy_margin": float(energy_margin.mean().item()),
        "energy_violation_fraction": float(energy_margin.le(0).float().mean().item()),
        "mean_raw_energy_margin": float(raw_energy_margin.mean().item()),
        "raw_energy_violation_fraction": float(raw_energy_margin.le(0).float().mean().item()),
        "mean_target_probability": float(target_probability.mean().item()),
        "mean_target_learned_support": float(target_learned.mean().item()),
        "mean_target_physical_support": float(target_physical.mean().item()),
        "mean_target_whole_cds_support": float(target_whole.mean().item()),
        # Kept as a diagnostic compatibility key; it is not the decision score.
        "mean_target_cosine": float(target_cosine.mean().item()),
    }


def summarize_evaluation_metrics(metrics: Optional[Mapping[str, Any]]) -> Optional[Dict[str, Any]]:
    if metrics is None:
        return None
    required = (
        "overall_accuracy", "balanced_accuracy", "minimum_class_accuracy", "kappa",
        "mean_logit_margin", "mean_energy_margin", "energy_violation_fraction",
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
        "logit_margin": float(metrics["mean_logit_margin"]),
        "energy_margin": float(metrics["mean_energy_margin"]),
        "energy_violation_fraction": float(metrics["energy_violation_fraction"]),
        "raw_energy_margin": float(metrics.get("mean_raw_energy_margin", metrics["mean_energy_margin"])),
        "raw_energy_violation_fraction": float(metrics.get("raw_energy_violation_fraction", metrics["energy_violation_fraction"])),
        "target_probability": float(metrics["mean_target_probability"]),
        "learned_support": float(metrics.get("mean_target_learned_support", 0.0)),
        "physical_support": float(metrics.get("mean_target_physical_support", 0.0)),
        "whole_cds_support": float(metrics.get("mean_target_whole_cds_support", 0.0)),
    }


def _ordered_variation_statistics(variation: Tensor) -> tuple[Tensor, Tensor]:
    value = ordered_variation_operator(variation)
    if value.ndim != 4 or value.size(1) != value.size(2):
        raise ValueError("variation must be [C,L,L,F]")
    class_count, length, _, feature_count = value.shape
    upper = torch.triu(torch.ones(length, length, device=value.device, dtype=torch.bool), diagonal=1)
    pairs = value[:, upper, :].reshape(class_count, -1, feature_count)
    active = pairs.gt(0).to(value.dtype).mean(dim=1)
    positive_sum = pairs.sum(dim=1)
    positive_count = pairs.gt(0).sum(dim=1).clamp_min(1).to(value.dtype)
    mean_positive = positive_sum / positive_count
    return active, mean_positive


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
    offdiag = ~torch.eye(class_count, device=structure.learned_identity.device, dtype=torch.bool)

    learned_identity = relations.learned_identity_proximity_by_region.mean(dim=2)
    learned_relation = relations.learned_relation_proximity
    physical_identity = relations.physical_identity_proximity_by_band.mean(dim=2)
    physical_relation = relations.physical_relation_proximity
    physical_combined = relations.physical_combined_proximity
    learned_active, learned_spread = _ordered_variation_statistics(structure.learned_variation)
    physical_active, physical_spread = _ordered_variation_statistics(structure.physical_variation)

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
        classes.append({
            "class_id": class_id,
            "class_name": str(names[class_id]) if class_id < len(names) else f"Class {class_id + 1}",
            "learned_identity_norm": float(torch.linalg.vector_norm(structure.learned_identity[position]).item()),
            "learned_variation_norm": float(torch.linalg.vector_norm(structure.learned_variation[position]).item()),
            "physical_identity_norm": float(torch.linalg.vector_norm(structure.physical_identity[position]).item()),
            "physical_variation_norm": float(torch.linalg.vector_norm(structure.physical_variation[position]).item()),
            "learned_active_relation_fraction": float(learned_active[position].mean().item()),
            "learned_positive_relation_scale": float(learned_spread[position].mean().item()),
            "physical_active_relation_fraction": float(physical_active[position].mean().item()),
            "physical_positive_relation_scale": float(physical_spread[position].mean().item()),
        })

    return {
        "class_ids": ids,
        "mean_learned_identity_proximity": offdiag_mean(learned_identity),
        "mean_learned_relation_proximity": offdiag_mean(learned_relation),
        "mean_physical_identity_proximity": offdiag_mean(physical_identity),
        "mean_physical_relation_proximity": offdiag_mean(physical_relation),
        "maximum_physical_proximity": maximum_physical_proximity,
        "most_physically_similar_class_pair": most_similar_pair,
        "classes": classes,
    }


@torch.no_grad()
def structural_decision_diagnostics(model: Any, class_ids: Sequence[int]) -> Dict[str, Any]:
    ids = _class_ids(class_ids)
    structure = model.structure_bank.class_structures(ids)
    learned_variation = ordered_variation_operator(structure.learned_variation)
    active, spread = _ordered_variation_statistics(learned_variation)
    return {
        "class_ids": ids,
        "classifier_parameter_count": int(model.classifier.parameter_count),
        "classifier_rule": "single all-seen bias-free cosine classifier over Q; Whole-CDS reconstructs/protects it during training and is read-only diagnostic evidence at evaluation",
        "classifier_logit_scale": float(model.classifier.logit_scale.detach().item()),
        "mean_learned_relation_variation_norm": float(
            torch.linalg.vector_norm(learned_variation.reshape(learned_variation.size(0), -1), dim=1).mean().item()
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
    learned = relations.learned_combined_proximity
    physical = relations.physical_combined_proximity
    joint = combine_learned_physical_relevance(learned, physical)

    output = model.classifier(
        structure.learned_identity.to(device=model.device, dtype=model.dtype),
        physical_spectral_evidence=structure.physical_identity.to(device=model.device, dtype=model.dtype),
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


def _mapping_value(mapping: Mapping[Any, Any] | None, class_id: int) -> float:
    if mapping is None:
        raise RuntimeError("required per-class mapping is absent")
    if class_id in mapping:
        return float(mapping[class_id])
    key = str(class_id)
    if key in mapping:
        return float(mapping[key])
    raise KeyError(class_id)


def summarize_class_group(metrics: Mapping[str, Any], class_ids: Sequence[int]) -> Dict[str, Any]:
    ids = _class_ids(class_ids)
    accuracies = [_mapping_value(metrics.get("per_class_accuracy"), class_id) for class_id in ids]
    margins = [_mapping_value(metrics.get("per_class_mean_logit_margin"), class_id) for class_id in ids]
    energy_margins = [_mapping_value(metrics.get("per_class_mean_energy_margin"), class_id) for class_id in ids]
    return {
        "class_ids": ids,
        "balanced_accuracy": float(sum(accuracies) / len(accuracies)),
        "minimum_class_accuracy": float(min(accuracies)),
        "mean_logit_margin": float(sum(margins) / len(margins)),
        "mean_energy_margin": float(sum(energy_margins) / len(energy_margins)),
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
        true_ids = [int(value) for value in phase_to_classes[true_phase] if int(value) in index]
        if not true_ids:
            continue
        rows = [index[value] for value in true_ids]
        row_total = int(matrix[rows, :].sum())
        row: Dict[str, float] = {}
        for predicted_phase in phases:
            predicted_ids = [
                int(value) for value in phase_to_classes[predicted_phase] if int(value) in index
            ]
            columns = [index[value] for value in predicted_ids]
            count = int(matrix[np.ix_(rows, columns)].sum()) if columns else 0
            row[str(predicted_phase)] = 0.0 if row_total == 0 else count / float(row_total)
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
        physical_spectral_evidence=structure.physical_identity.to(device=model.device, dtype=model.dtype),
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
        "learned_self_support_mean": float(output.learned_support[target, target].mean().item()),
        "physical_self_support_mean": float(output.physical_support[target, target].mean().item()),
        "whole_self_support_mean": float(output.whole_support[target, target].mean().item()),
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

# """Read-only evaluation for the finalized HSI CDS architecture."""

# from contextlib import contextmanager
# import math
# from numbers import Integral, Real
# from typing import Any, Dict, Mapping, Optional, Sequence

# import numpy as np
# import torch

# from models.classifier import (
#     SpectralCDSClassifier,
#     ordered_structural_reliability,
# )
# from models.cds_decision_region import HSICDSDecisionRegionRegulator
# from models.geometry_bank import (
#     HSIClassDiscriminativeStructureBank,
#     ordered_variation_operator,
#     whole_cds_pair_risk,
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


# def _unpack(batch: Mapping[str, Any], *, model: Any, device: torch.device) -> tuple[Tensor, Tensor, Tensor]:
#     required = {"image", "raw_center_spectrum", "label"}
#     missing = required - set(batch)
#     if missing:
#         raise KeyError(f"evaluation batch missing {sorted(missing)}")
#     patch = torch.as_tensor(batch["image"], device=device, dtype=model.dtype)
#     spectrum = torch.as_tensor(batch["raw_center_spectrum"], device=device, dtype=model.dtype)
#     raw_labels = torch.as_tensor(batch["label"])
#     if raw_labels.dtype == torch.bool or raw_labels.is_complex() or torch.is_floating_point(raw_labels):
#         raise RuntimeError("evaluation labels must use an integer dtype")
#     labels = raw_labels.to(device=device, dtype=torch.long).flatten()
#     if labels.shape != (patch.size(0),):
#         raise RuntimeError("evaluation labels are misaligned")
#     return patch, spectrum, labels


# def _confusion(targets: np.ndarray, predictions: np.ndarray, ids: Sequence[int]) -> np.ndarray:
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
#     return 0.0 if abs(denominator) <= np.finfo(np.float64).eps else (observed - expected) / denominator


# @torch.no_grad()
# def evaluate_loader(
#     model: Any,
#     loader: Any,
#     *,
#     class_ids: Sequence[int],
#     target_class_ids: Optional[Sequence[int]] = None,
#     structures=None,
#     device: str | torch.device,
# ) -> Dict[str, Any]:
#     ids = _class_ids(class_ids)
#     target_ids = ids if target_class_ids is None else _class_ids(target_class_ids, name="target_class_ids")
#     if set(target_ids) - set(ids):
#         raise ValueError("target_class_ids must be a subset of class_ids")
#     if structures is not None:
#         structure_ids = [int(value) for value in structures.class_ids.detach().cpu().tolist()]
#         if structure_ids != ids:
#             raise ValueError("supplied structures must match class_ids exactly")

#     evaluation_device = torch.device(device)
#     model.validate_model_state()
#     if not isinstance(model.classifier, SpectralCDSClassifier):
#         raise RuntimeError("classifier must be SpectralCDSClassifier")

#     evaluation_cds = (
#         model.build_class_discriminative_structure(structures)
#         if structures is not None
#         else model.class_discriminative_structure(ids)
#     )

#     predictions: list[Tensor] = []
#     labels_rows: list[Tensor] = []
#     margin_rows: list[Tensor] = []
#     support_rows: list[Tensor] = []
#     identity_support_rows: list[Tensor] = []
#     structural_support_rows: list[Tensor] = []
#     structural_margin_rows: list[Tensor] = []
#     allowed = torch.tensor(ids, device=evaluation_device, dtype=torch.long)

#     with _temporary_eval_state(model):
#         for batch in loader:
#             patch, spectrum, labels = _unpack(batch, model=model, device=evaluation_device)
#             if not bool(torch.isin(labels, allowed).all()):
#                 raise RuntimeError("evaluation loader contains a class outside class_ids")
#             q = model.extract_features(patch, center_spectrum=spectrum, return_aux=False).ordered_features
#             output = model.classify_features(q, center_spectrum=spectrum, cds=evaluation_cds)

#             matches = labels.unsqueeze(1).eq(output.class_ids.unsqueeze(0))
#             if not bool(matches.sum(dim=1).eq(1).all()):
#                 raise RuntimeError("could not map labels to classifier columns")
#             target_local = matches.to(torch.long).argmax(dim=1)
#             rows = torch.arange(labels.numel(), device=labels.device)
#             target_logit = output.logits[rows, target_local]
#             rival = output.logits.clone()
#             rival[rows, target_local] = -torch.inf
#             margin = target_logit - rival.max(dim=1).values

#             predictions.append(output.prediction.detach())
#             labels_rows.append(labels.detach())
#             margin_rows.append(margin.detach())
#             support_rows.append(output.decision_support[rows, target_local].detach())
#             identity_support_rows.append(output.identity_support[rows, target_local].detach())
#             target_structural = output.structural_support[rows, target_local]
#             rival_structural = output.structural_support.clone()
#             rival_structural[rows, target_local] = -torch.inf
#             structural_margin = target_structural - rival_structural.max(dim=1).values
#             structural_support_rows.append(target_structural.detach())
#             structural_margin_rows.append(structural_margin.detach())

#     if not predictions:
#         raise RuntimeError("cannot evaluate an empty loader")

#     prediction_all = torch.cat(predictions)
#     labels_all = torch.cat(labels_rows)
#     margin_all = torch.cat(margin_rows)
#     support_all = torch.cat(support_rows)
#     identity_support_all = torch.cat(identity_support_rows)
#     structural_support_all = torch.cat(structural_support_rows)
#     structural_margin_all = torch.cat(structural_margin_rows)
#     target_tensor = torch.tensor(target_ids, device=labels_all.device, dtype=torch.long)
#     mask = torch.isin(labels_all, target_tensor)
#     labels = labels_all[mask]
#     prediction = prediction_all[mask]
#     margin = margin_all[mask]
#     structural_margin = structural_margin_all[mask]

#     per_class_accuracy: Dict[int, float] = {}
#     per_class_margin: Dict[int, float] = {}
#     per_class_structural_margin: Dict[int, float] = {}
#     class_counts: Dict[int, int] = {}
#     for class_id in target_ids:
#         class_mask = labels.eq(class_id)
#         count = int(class_mask.sum().item())
#         if count <= 0:
#             raise RuntimeError(f"target class {class_id} is absent")
#         class_counts[class_id] = count
#         per_class_accuracy[class_id] = float(prediction[class_mask].eq(labels[class_mask]).float().mean().item())
#         per_class_margin[class_id] = float(margin[class_mask].mean().item())
#         per_class_structural_margin[class_id] = float(structural_margin[class_mask].mean().item())

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
#         "per_class_mean_structural_support_margin": per_class_structural_margin,
#         "mean_logit_margin": float(margin.mean().item()),
#         "mean_structural_support_margin": float(structural_margin.mean().item()),
#         "minimum_structural_support_margin": float(structural_margin.min().item()),
#         "structural_overlap_fraction": float(structural_margin.le(0.0).to(structural_margin.dtype).mean().item()),
#         "mean_target_decision_support": float(support_all[mask].mean().item()),
#         "mean_target_identity_support": float(identity_support_all[mask].mean().item()),
#         "mean_target_structural_support": float(structural_support_all[mask].mean().item()),
#     }


# def summarize_evaluation_metrics(metrics: Optional[Mapping[str, Any]]) -> Optional[Dict[str, Any]]:
#     """Canonical reporting schema owned by the evaluator."""
#     if metrics is None:
#         return None
#     required = (
#         "overall_accuracy",
#         "balanced_accuracy",
#         "minimum_class_accuracy",
#         "kappa",
#         "mean_logit_margin",
#         "mean_target_decision_support",
#         "mean_target_identity_support",
#         "mean_target_structural_support",
#         "mean_structural_support_margin",
#         "minimum_structural_support_margin",
#         "structural_overlap_fraction",
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
#         "decision_support": float(metrics["mean_target_decision_support"]),
#         "identity_support": float(metrics["mean_target_identity_support"]),
#         "structural_support": float(metrics["mean_target_structural_support"]),
#         "structural_support_margin": float(metrics["mean_structural_support_margin"]),
#         "minimum_structural_support_margin": float(metrics["minimum_structural_support_margin"]),
#         "structural_overlap_fraction": float(metrics["structural_overlap_fraction"]),
#     }




# def _diagonal_variation_statistics(variation: Tensor) -> tuple[Tensor, Tensor]:
#     """Return per-class active fraction and mean diagonal absolute IQR extent."""
#     value = ordered_variation_operator(variation)
#     if value.ndim != 4 or value.size(1) != value.size(2):
#         raise ValueError("variation must be [C,L,L,F]")
#     length = value.size(1)
#     index = torch.arange(length, device=value.device)
#     diagonal = value[:, index, index, :]
#     active = diagonal.gt(0).to(value.dtype).mean(dim=(1, 2))
#     mean_extent = diagonal.mean(dim=(1, 2))
#     return active, mean_extent

# def _ordered_variation_statistics(variation: Tensor) -> tuple[Tensor, Tensor]:
#     """Direct statistics for V_Q/V_P without spectral decomposition.

#     Returns per-class/channel/component active ordered-pair fraction and mean
#     positive relation-spread magnitude. V is used exactly as stored: a direct
#     non-negative pairwise relation spread, never as a covariance matrix.
#     """
#     value = ordered_variation_operator(variation)
#     if value.ndim != 4 or value.size(1) != value.size(2):
#         raise ValueError("variation must be [C,L,L,F]")
#     c, length, _, features = value.shape
#     upper = torch.triu(
#         torch.ones(length, length, device=value.device, dtype=torch.bool),
#         diagonal=1,
#     )
#     pairs = value[:, upper, :].reshape(c, -1, features)
#     active = pairs.gt(0).to(value.dtype).mean(dim=1)
#     positive_sum = pairs.sum(dim=1)
#     positive_count = pairs.gt(0).sum(dim=1).clamp_min(1).to(value.dtype)
#     mean_positive = positive_sum / positive_count
#     return active, mean_positive


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
#     batch = bank.class_structures(ids)
#     cds = bank.build_cds(batch)
#     relations = cds.relations
#     c = len(ids)
#     offdiag = ~torch.eye(c, device=batch.identity.device, dtype=torch.bool)
#     learned_identity = relations.learned_identity_cosine_by_region.mean(dim=2)
#     spectral_identity = relations.spectral_identity_cosine_by_component.mean(dim=2)
#     learned_variation = relations.learned_variation_overlap
#     spectral_variation = relations.spectral_variation_overlap
#     confusability = relations.spectral_confusability
#     learned_active, learned_spread = _ordered_variation_statistics(batch.variation)
#     physical_active, physical_spread = _ordered_variation_statistics(batch.spectral_variation)
#     learned_absolute_active, learned_absolute_extent = _diagonal_variation_statistics(batch.variation)
#     physical_absolute_active, physical_absolute_extent = _diagonal_variation_statistics(batch.spectral_variation)

#     def offdiag_mean(matrix: Tensor) -> float:
#         return float(matrix[offdiag].mean().item()) if bool(offdiag.any()) else 1.0

#     if bool(offdiag.any()):
#         masked = confusability.masked_fill(~offdiag, -torch.inf)
#         flat = int(masked.argmax().item())
#         left, right = divmod(flat, c)
#         pair = [ids[left], ids[right]]
#         maximum_confusability = float(masked[left, right].item())
#     else:
#         pair = None
#         maximum_confusability = 1.0

#     names = list(target_names or [])
#     classes = []
#     for position, class_id in enumerate(ids):
#         classes.append({
#             "class_id": class_id,
#             "class_name": str(names[class_id]) if class_id < len(names) else f"Class {class_id + 1}",
#             "sample_count": int(batch.sample_counts[position].item()),
#             "learned_identity_norm": float(torch.linalg.vector_norm(batch.identity[position]).item()),
#             "learned_complete_variation_norm": float(torch.linalg.vector_norm(batch.variation[position]).item()),
#             "learned_active_absolute_fraction": float(learned_absolute_active[position].item()),
#             "learned_mean_absolute_iqr_extent": float(learned_absolute_extent[position].item()),
#             "physical_spectral_identity_norm": float(torch.linalg.vector_norm(batch.spectral_identity[position]).item()),
#             "physical_complete_variation_norm": float(torch.linalg.vector_norm(batch.spectral_variation[position]).item()),
#             "physical_active_absolute_fraction": float(physical_absolute_active[position].item()),
#             "physical_mean_absolute_iqr_extent": float(physical_absolute_extent[position].item()),
#             "learned_active_relation_fraction": float(learned_active[position].mean().item()),
#             "learned_mean_positive_relation_spread": float(learned_spread[position].mean().item()),
#             "physical_active_relation_fraction": float(physical_active[position].mean().item()),
#             "physical_mean_positive_relation_spread": float(physical_spread[position].mean().item()),
#         })
#     return {
#         "class_ids": ids,
#         "persistent_state": "{I_Q,V_Q,I_P,V_P,N}",
#         "learned_identity_role": "robust_ordered_spectral_identity",
#         "learned_V_Q_role": "diagonal_absolute_IQR_plus_offdiagonal_ordered_relation_IQR_independent_of_I_Q",
#         "N_role": "number_of_real_TRAIN_samples_that_supported_the_stored_CDS",
#         "physical_spectral_role": "full_band_level_slope_curvature_identity_with_diagonal_absolute_and_offdiagonal_relation_IQR; used in whole-CDS pair risk, not concatenated into Q",
#         "native_spectral_band_count": int(batch.spectral_identity.size(1)),
#         "mean_learned_identity_relation": offdiag_mean(learned_identity),
#         "mean_learned_variation_relation": offdiag_mean(learned_variation),
#         "mean_physical_identity_relation": offdiag_mean(spectral_identity),
#         "mean_physical_variation_relation": offdiag_mean(spectral_variation),
#         "mean_physical_region_confusability": offdiag_mean(confusability),
#         "maximum_physical_region_confusability": maximum_confusability,
#         "most_physically_confusable_class_pair": pair,
#         "classes": classes,
#     }


# @torch.no_grad()
# def structural_decision_diagnostics(model: Any, class_ids: Sequence[int]) -> Dict[str, Any]:
#     ids = _class_ids(class_ids)
#     batch = model.structure_bank.class_structures(ids)
#     variation = ordered_variation_operator(batch.variation)
#     active, spread = _ordered_variation_statistics(variation)
#     reliability = ordered_structural_reliability(variation)
#     eps = torch.finfo(variation.dtype).eps
#     effective_regions = 1.0 / reliability.square().sum(dim=1).clamp_min(eps)
#     total_norm = torch.linalg.vector_norm(variation.reshape(variation.size(0), -1), dim=1)
#     return {
#         "class_ids": ids,
#         "classifier_parameter_count": int(model.classifier.parameter_count),
#         "classifier_rule": "single_bias_free_scaled_cosine_head; direct_Q_supervised_contrast_shapes_features_and_whole_CDS_old_to_current_clearance_protects_boundaries",
#         "classifier_logit_scale": float(model.classifier.logit_scale_value.detach().item()),
#         "mean_learned_relation_variation_norm": float(total_norm.mean().item()),
#         "mean_active_ordered_relation_fraction": float(active.mean().item()),
#         "mean_positive_ordered_relation_spread": float(spread.mean().item()),
#         "mean_effective_ordered_regions": float(effective_regions.mean().item()),
#         "minimum_effective_ordered_regions": float(effective_regions.min().item()),
#     }


# @torch.no_grad()
# def cross_phase_cds_decision_diagnostics(
#     model: Any,
#     *,
#     old_class_ids: Sequence[int],
#     new_class_ids: Sequence[int],
# ) -> Dict[str, Any]:
#     """Report whole-CDS classifier clearance and pair-risk diagnostics.

#     No class identity tensor is ever fed to the classifier as a pseudo sample.
#     The deployed classifier is assessed against complete learned CDS extents
#     (I_Q,V_Q), while I_P,V_P enter the same boundary objective through the
#     whole-CDS pair risk.
#     """
#     old_ids = _class_ids(old_class_ids, name="old_class_ids")
#     new_ids = _class_ids(new_class_ids, name="new_class_ids")
#     seen_ids = old_ids + new_ids
#     batch = model.structure_bank.class_structures(seen_ids)
#     cds = model.structure_bank.build_cds(batch)
#     ids = batch.class_ids.to(device=model.device, dtype=torch.long)

#     regulator = HSICDSDecisionRegionRegulator()
#     boundary = regulator(
#         classifier=model.classifier,
#         seen_cds=cds,
#         historical_class_count=len(old_ids),
#     )

#     risk, learned, physical = whole_cds_pair_risk(cds)
#     c = len(seen_ids)
#     old_count = len(old_ids)
#     old_mask = torch.arange(c, device=model.device) < old_count
#     diagonal = torch.eye(c, device=model.device, dtype=torch.bool)
#     cross = (~diagonal) & (old_mask.unsqueeze(1) ^ old_mask.unsqueeze(0))
#     old_old = (~diagonal) & old_mask.unsqueeze(1) & old_mask.unsqueeze(0)
#     new_mask = ~old_mask
#     new_new = (~diagonal) & new_mask.unsqueeze(1) & new_mask.unsqueeze(0)
#     active, spread = _ordered_variation_statistics(batch.variation)

#     return {
#         "old_class_ids": old_ids,
#         "new_class_ids": new_ids,
#         "whole_cds_boundary_clearance_mean": float(boundary.structural_clearance_mean.item()),
#         "whole_cds_boundary_clearance_min": float(boundary.structural_clearance_min.item()),
#         "whole_cds_absolute_clearance_mean": float(boundary.absolute_clearance_mean.item()),
#         "whole_cds_relational_clearance_mean": float(boundary.relational_clearance_mean.item()),
#         "whole_cds_unsafe_pair_fraction": float(boundary.unsafe_pair_fraction.item()),
#         "cross_phase_pair_risk_mean": float(risk[cross].mean().item()),
#         "cross_phase_learned_region_risk_mean": float(learned[cross].mean().item()),
#         "cross_phase_physical_region_risk_mean": float(physical[cross].mean().item()),
#         "old_old_pair_risk_mean": float(risk[old_old].mean().item()) if bool(old_old.any()) else 0.0,
#         "new_new_pair_risk_mean": float(risk[new_new].mean().item()) if bool(new_new.any()) else 0.0,
#         "mean_active_ordered_relation_fraction": float(active.mean().item()),
#         "mean_positive_ordered_relation_spread": float(spread.mean().item()),
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


# def summarize_class_group(metrics: Mapping[str, Any], class_ids: Sequence[int]) -> Dict[str, Any]:
#     ids = _class_ids(class_ids)
#     accuracies = [_mapping_value(metrics.get("per_class_accuracy"), class_id) for class_id in ids]
#     margins = [_mapping_value(metrics.get("per_class_mean_logit_margin"), class_id) for class_id in ids]
#     return {
#         "class_ids": ids,
#         "balanced_accuracy": float(sum(accuracies) / len(accuracies)),
#         "minimum_class_accuracy": float(min(accuracies)),
#         "mean_logit_margin": float(sum(margins) / len(margins)),
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
#         true_ids = [int(value) for value in phase_to_classes[true_phase] if int(value) in index]
#         if not true_ids:
#             continue
#         rows = [index[value] for value in true_ids]
#         row_total = int(matrix[rows, :].sum())
#         row: Dict[str, float] = {}
#         for pred_phase in phases:
#             pred_ids = [int(value) for value in phase_to_classes[pred_phase] if int(value) in index]
#             cols = [index[value] for value in pred_ids]
#             count = int(matrix[np.ix_(rows, cols)].sum()) if cols else 0
#             row[str(pred_phase)] = 0.0 if row_total == 0 else count / float(row_total)
#         result[str(true_phase)] = row
#     return result


# __all__ = [
#     "evaluate_loader",
#     "summarize_evaluation_metrics",
#     "class_structure_diagnostics",
#     "structural_decision_diagnostics",
#     "cross_phase_cds_decision_diagnostics",
#     "summarize_class_group",
#     "old_new_summary",
#     "phase_origin_confusion_diagnostics",
# ]





# from __future__ import annotations

# """Read-only evaluation for the finalized HSI CDS architecture."""

# from contextlib import contextmanager
# import math
# from numbers import Integral, Real
# from typing import Any, Dict, Mapping, Optional, Sequence

# import numpy as np
# import torch

# from models.classifier import (
#     SpectralCDSClassifier,
#     cds_identity_direction,
#     ordered_structural_reliability,
# )
# from models.geometry_bank import (
#     HSIClassDiscriminativeStructureBank,
#     ordered_variation_operator,
#     whole_cds_pair_risk,
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


# def _unpack(batch: Mapping[str, Any], *, model: Any, device: torch.device) -> tuple[Tensor, Tensor, Tensor]:
#     required = {"image", "raw_center_spectrum", "label"}
#     missing = required - set(batch)
#     if missing:
#         raise KeyError(f"evaluation batch missing {sorted(missing)}")
#     patch = torch.as_tensor(batch["image"], device=device, dtype=model.dtype)
#     spectrum = torch.as_tensor(batch["raw_center_spectrum"], device=device, dtype=model.dtype)
#     raw_labels = torch.as_tensor(batch["label"])
#     if raw_labels.dtype == torch.bool or raw_labels.is_complex() or torch.is_floating_point(raw_labels):
#         raise RuntimeError("evaluation labels must use an integer dtype")
#     labels = raw_labels.to(device=device, dtype=torch.long).flatten()
#     if labels.shape != (patch.size(0),):
#         raise RuntimeError("evaluation labels are misaligned")
#     return patch, spectrum, labels


# def _confusion(targets: np.ndarray, predictions: np.ndarray, ids: Sequence[int]) -> np.ndarray:
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
#     return 0.0 if abs(denominator) <= np.finfo(np.float64).eps else (observed - expected) / denominator


# @torch.no_grad()
# def evaluate_loader(
#     model: Any,
#     loader: Any,
#     *,
#     class_ids: Sequence[int],
#     target_class_ids: Optional[Sequence[int]] = None,
#     structures=None,
#     device: str | torch.device,
# ) -> Dict[str, Any]:
#     ids = _class_ids(class_ids)
#     target_ids = ids if target_class_ids is None else _class_ids(target_class_ids, name="target_class_ids")
#     if set(target_ids) - set(ids):
#         raise ValueError("target_class_ids must be a subset of class_ids")
#     if structures is not None:
#         structure_ids = [int(value) for value in structures.class_ids.detach().cpu().tolist()]
#         if structure_ids != ids:
#             raise ValueError("supplied structures must match class_ids exactly")

#     evaluation_device = torch.device(device)
#     model.validate_model_state()
#     if not isinstance(model.classifier, SpectralCDSClassifier):
#         raise RuntimeError("classifier must be SpectralCDSClassifier")

#     evaluation_cds = (
#         model.build_class_discriminative_structure(structures)
#         if structures is not None
#         else model.class_discriminative_structure(ids)
#     )

#     predictions: list[Tensor] = []
#     labels_rows: list[Tensor] = []
#     margin_rows: list[Tensor] = []
#     support_rows: list[Tensor] = []
#     identity_support_rows: list[Tensor] = []
#     structural_support_rows: list[Tensor] = []
#     allowed = torch.tensor(ids, device=evaluation_device, dtype=torch.long)

#     with _temporary_eval_state(model):
#         for batch in loader:
#             patch, spectrum, labels = _unpack(batch, model=model, device=evaluation_device)
#             if not bool(torch.isin(labels, allowed).all()):
#                 raise RuntimeError("evaluation loader contains a class outside class_ids")
#             q = model.extract_features(patch, center_spectrum=spectrum, return_aux=False).ordered_features
#             output = model.classify_features(q, center_spectrum=spectrum, cds=evaluation_cds)

#             matches = labels.unsqueeze(1).eq(output.class_ids.unsqueeze(0))
#             if not bool(matches.sum(dim=1).eq(1).all()):
#                 raise RuntimeError("could not map labels to classifier columns")
#             target_local = matches.to(torch.long).argmax(dim=1)
#             rows = torch.arange(labels.numel(), device=labels.device)
#             target_logit = output.logits[rows, target_local]
#             rival = output.logits.clone()
#             rival[rows, target_local] = -torch.inf
#             margin = target_logit - rival.max(dim=1).values

#             predictions.append(output.prediction.detach())
#             labels_rows.append(labels.detach())
#             margin_rows.append(margin.detach())
#             support_rows.append(output.decision_support[rows, target_local].detach())
#             identity_support_rows.append(output.identity_support[rows, target_local].detach())
#             structural_support_rows.append(output.structural_support[rows, target_local].detach())

#     if not predictions:
#         raise RuntimeError("cannot evaluate an empty loader")

#     prediction_all = torch.cat(predictions)
#     labels_all = torch.cat(labels_rows)
#     margin_all = torch.cat(margin_rows)
#     support_all = torch.cat(support_rows)
#     identity_support_all = torch.cat(identity_support_rows)
#     structural_support_all = torch.cat(structural_support_rows)
#     target_tensor = torch.tensor(target_ids, device=labels_all.device, dtype=torch.long)
#     mask = torch.isin(labels_all, target_tensor)
#     labels = labels_all[mask]
#     prediction = prediction_all[mask]
#     margin = margin_all[mask]

#     per_class_accuracy: Dict[int, float] = {}
#     per_class_margin: Dict[int, float] = {}
#     class_counts: Dict[int, int] = {}
#     for class_id in target_ids:
#         class_mask = labels.eq(class_id)
#         count = int(class_mask.sum().item())
#         if count <= 0:
#             raise RuntimeError(f"target class {class_id} is absent")
#         class_counts[class_id] = count
#         per_class_accuracy[class_id] = float(prediction[class_mask].eq(labels[class_mask]).float().mean().item())
#         per_class_margin[class_id] = float(margin[class_mask].mean().item())

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
#         "mean_logit_margin": float(margin.mean().item()),
#         "mean_target_decision_support": float(support_all[mask].mean().item()),
#         "mean_target_identity_support": float(identity_support_all[mask].mean().item()),
#         "mean_target_structural_support": float(structural_support_all[mask].mean().item()),
#     }


# def summarize_evaluation_metrics(metrics: Optional[Mapping[str, Any]]) -> Optional[Dict[str, Any]]:
#     """Canonical reporting schema owned by the evaluator."""
#     if metrics is None:
#         return None
#     required = (
#         "overall_accuracy",
#         "balanced_accuracy",
#         "minimum_class_accuracy",
#         "kappa",
#         "mean_logit_margin",
#         "mean_target_decision_support",
#         "mean_target_identity_support",
#         "mean_target_structural_support",
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
#         "decision_support": float(metrics["mean_target_decision_support"]),
#         "identity_support": float(metrics["mean_target_identity_support"]),
#         "structural_support": float(metrics["mean_target_structural_support"]),
#     }


# def _ordered_variation_statistics(variation: Tensor) -> tuple[Tensor, Tensor]:
#     """Direct statistics for V_Q/V_P without spectral decomposition.

#     Returns per-class/channel/component active ordered-pair fraction and mean
#     positive relation-spread magnitude. V is used exactly as stored: a direct
#     non-negative pairwise relation spread, never as a covariance matrix.
#     """
#     value = ordered_variation_operator(variation)
#     if value.ndim != 4 or value.size(1) != value.size(2):
#         raise ValueError("variation must be [C,L,L,F]")
#     c, length, _, features = value.shape
#     upper = torch.triu(
#         torch.ones(length, length, device=value.device, dtype=torch.bool),
#         diagonal=1,
#     )
#     pairs = value[:, upper, :].reshape(c, -1, features)
#     active = pairs.gt(0).to(value.dtype).mean(dim=1)
#     positive_sum = pairs.sum(dim=1)
#     positive_count = pairs.gt(0).sum(dim=1).clamp_min(1).to(value.dtype)
#     mean_positive = positive_sum / positive_count
#     return active, mean_positive


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
#     batch = bank.class_structures(ids)
#     cds = bank.build_cds(batch)
#     relations = cds.relations
#     c = len(ids)
#     offdiag = ~torch.eye(c, device=batch.identity.device, dtype=torch.bool)
#     learned_identity = relations.learned_identity_cosine_by_region.mean(dim=2)
#     spectral_identity = relations.spectral_identity_cosine_by_component.mean(dim=2)
#     learned_variation = relations.learned_variation_overlap
#     spectral_variation = relations.spectral_variation_overlap
#     confusability = relations.spectral_confusability
#     learned_active, learned_spread = _ordered_variation_statistics(batch.variation)
#     physical_active, physical_spread = _ordered_variation_statistics(batch.spectral_variation)

#     def offdiag_mean(matrix: Tensor) -> float:
#         return float(matrix[offdiag].mean().item()) if bool(offdiag.any()) else 1.0

#     if bool(offdiag.any()):
#         masked = confusability.masked_fill(~offdiag, -torch.inf)
#         flat = int(masked.argmax().item())
#         left, right = divmod(flat, c)
#         pair = [ids[left], ids[right]]
#         maximum_confusability = float(masked[left, right].item())
#     else:
#         pair = None
#         maximum_confusability = 1.0

#     names = list(target_names or [])
#     classes = []
#     for position, class_id in enumerate(ids):
#         classes.append({
#             "class_id": class_id,
#             "class_name": str(names[class_id]) if class_id < len(names) else f"Class {class_id + 1}",
#             "sample_count": int(batch.sample_counts[position].item()),
#             "learned_identity_norm": float(torch.linalg.vector_norm(batch.identity[position]).item()),
#             "learned_relation_variation_norm": float(torch.linalg.vector_norm(batch.variation[position]).item()),
#             "physical_spectral_identity_norm": float(torch.linalg.vector_norm(batch.spectral_identity[position]).item()),
#             "physical_relation_variation_norm": float(torch.linalg.vector_norm(batch.spectral_variation[position]).item()),
#             "learned_active_relation_fraction": float(learned_active[position].mean().item()),
#             "learned_mean_positive_relation_spread": float(learned_spread[position].mean().item()),
#             "physical_active_relation_fraction": float(physical_active[position].mean().item()),
#             "physical_mean_positive_relation_spread": float(physical_spread[position].mean().item()),
#         })
#     return {
#         "class_ids": ids,
#         "persistent_state": "{I_Q,V_Q,I_P,V_P,N}",
#         "learned_identity_role": "robust_ordered_spectral_identity",
#         "learned_V_Q_role": "direct_intra_class_ordered_relation_IQR_spread_independent_of_I_Q",
#         "N_role": "number_of_real_TRAIN_samples_that_supported_the_stored_CDS",
#         "physical_spectral_role": "full_band_level_slope_curvature_identity_and_direct_relation_spread",
#         "native_spectral_band_count": int(batch.spectral_identity.size(1)),
#         "mean_learned_identity_relation": offdiag_mean(learned_identity),
#         "mean_learned_variation_relation": offdiag_mean(learned_variation),
#         "mean_physical_identity_relation": offdiag_mean(spectral_identity),
#         "mean_physical_variation_relation": offdiag_mean(spectral_variation),
#         "mean_physical_region_confusability": offdiag_mean(confusability),
#         "maximum_physical_region_confusability": maximum_confusability,
#         "most_physically_confusable_class_pair": pair,
#         "classes": classes,
#     }


# @torch.no_grad()
# def structural_decision_diagnostics(model: Any, class_ids: Sequence[int]) -> Dict[str, Any]:
#     ids = _class_ids(class_ids)
#     batch = model.structure_bank.class_structures(ids)
#     variation = ordered_variation_operator(batch.variation)
#     active, spread = _ordered_variation_statistics(variation)
#     reliability = ordered_structural_reliability(variation)
#     eps = torch.finfo(variation.dtype).eps
#     effective_regions = 1.0 / reliability.square().sum(dim=1).clamp_min(eps)
#     total_norm = torch.linalg.vector_norm(variation.reshape(variation.size(0), -1), dim=1)
#     return {
#         "class_ids": ids,
#         "classifier_parameter_count": int(model.classifier.parameter_count),
#         "classifier_rule": "single_bias_free_scaled_cosine_head; historical_CDS_enters_training_as_temporary_classifier_support",
#         "classifier_logit_scale": float(model.classifier.logit_scale_value.detach().item()),
#         "mean_learned_relation_variation_norm": float(total_norm.mean().item()),
#         "mean_active_ordered_relation_fraction": float(active.mean().item()),
#         "mean_positive_ordered_relation_spread": float(spread.mean().item()),
#         "mean_effective_ordered_regions": float(effective_regions.mean().item()),
#         "minimum_effective_ordered_regions": float(effective_regions.min().item()),
#     }


# @torch.no_grad()
# def cross_phase_cds_decision_diagnostics(
#     model: Any,
#     *,
#     old_class_ids: Sequence[int],
#     new_class_ids: Sequence[int],
# ) -> Dict[str, Any]:
#     """Report actual classifier identity competition plus whole-CDS pair risk."""
#     old_ids = _class_ids(old_class_ids, name="old_class_ids")
#     new_ids = _class_ids(new_class_ids, name="new_class_ids")
#     seen_ids = old_ids + new_ids
#     batch = model.structure_bank.class_structures(seen_ids)
#     cds = model.structure_bank.build_cds(batch)
#     ids = batch.class_ids.to(device=model.device)
#     identity = batch.identity.to(device=model.device, dtype=model.dtype)
#     output = model.classifier(identity, class_ids=ids, trainable_class_ids=())
#     target = torch.arange(len(seen_ids), device=model.device)
#     target_logit = output.logits[target, target]
#     rival = output.logits.clone()
#     rival[target, target] = -torch.inf
#     margin = target_logit - rival.max(dim=1).values

#     risk, learned, physical = whole_cds_pair_risk(cds)
#     c = len(seen_ids)
#     old_count = len(old_ids)
#     old_mask = torch.arange(c, device=model.device) < old_count
#     diagonal = torch.eye(c, device=model.device, dtype=torch.bool)
#     cross = (~diagonal) & (old_mask.unsqueeze(1) ^ old_mask.unsqueeze(0))
#     old_old = (~diagonal) & old_mask.unsqueeze(1) & old_mask.unsqueeze(0)
#     new_mask = ~old_mask
#     new_new = (~diagonal) & new_mask.unsqueeze(1) & new_mask.unsqueeze(0)
#     active, spread = _ordered_variation_statistics(batch.variation)
#     prediction_local = output.logits.argmax(dim=1)
#     identity_correct = prediction_local.eq(target)
#     return {
#         "old_class_ids": old_ids,
#         "new_class_ids": new_ids,
#         "identity_support_accuracy_diagnostic": float(identity_correct.float().mean().item()),
#         "old_identity_support_accuracy_diagnostic": float(identity_correct[:old_count].float().mean().item()),
#         "new_identity_support_accuracy_diagnostic": float(identity_correct[old_count:].float().mean().item()),
#         "identity_classifier_margin_mean": float(margin.mean().item()),
#         "identity_classifier_margin_min": float(margin.min().item()),
#         "cross_phase_pair_risk_mean": float(risk[cross].mean().item()),
#         "cross_phase_learned_region_risk_mean": float(learned[cross].mean().item()),
#         "cross_phase_physical_region_risk_mean": float(physical[cross].mean().item()),
#         "old_old_pair_risk_mean": float(risk[old_old].mean().item()) if bool(old_old.any()) else 0.0,
#         "new_new_pair_risk_mean": float(risk[new_new].mean().item()) if bool(new_new.any()) else 0.0,
#         "mean_active_ordered_relation_fraction": float(active.mean().item()),
#         "mean_positive_ordered_relation_spread": float(spread.mean().item()),
#         "diagnostic_rule": "real_old_new_BA_and_actual_classifier_logits_are_primary; no_eigen_or_covariance_diagnostics",
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


# def summarize_class_group(metrics: Mapping[str, Any], class_ids: Sequence[int]) -> Dict[str, Any]:
#     ids = _class_ids(class_ids)
#     accuracies = [_mapping_value(metrics.get("per_class_accuracy"), class_id) for class_id in ids]
#     margins = [_mapping_value(metrics.get("per_class_mean_logit_margin"), class_id) for class_id in ids]
#     return {
#         "class_ids": ids,
#         "balanced_accuracy": float(sum(accuracies) / len(accuracies)),
#         "minimum_class_accuracy": float(min(accuracies)),
#         "mean_logit_margin": float(sum(margins) / len(margins)),
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
#         true_ids = [int(value) for value in phase_to_classes[true_phase] if int(value) in index]
#         if not true_ids:
#             continue
#         rows = [index[value] for value in true_ids]
#         row_total = int(matrix[rows, :].sum())
#         row: Dict[str, float] = {}
#         for pred_phase in phases:
#             pred_ids = [int(value) for value in phase_to_classes[pred_phase] if int(value) in index]
#             cols = [index[value] for value in pred_ids]
#             count = int(matrix[np.ix_(rows, cols)].sum()) if cols else 0
#             row[str(pred_phase)] = 0.0 if row_total == 0 else count / float(row_total)
#         result[str(true_phase)] = row
#     return result


# __all__ = [
#     "evaluate_loader",
#     "summarize_evaluation_metrics",
#     "class_structure_diagnostics",
#     "structural_decision_diagnostics",
#     "cross_phase_cds_decision_diagnostics",
#     "summarize_class_group",
#     "old_new_summary",
#     "phase_origin_confusion_diagnostics",
# ]
