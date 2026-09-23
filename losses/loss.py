from __future__ import annotations
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from models.geometry_bank import (
    ClassDiscriminativeStructureState,
    ClassStructureBatch,
    normalize_ordered_decision_features,
    ordered_cds_support_components,
    whole_cds_pair_risk,
)

Tensor = torch.Tensor


@dataclass(frozen=True)
class ClassificationObjective:
    loss: Tensor
    accuracy: Tensor
    targets_local: Tensor


@dataclass(frozen=True)
class SupervisedContrastiveObjective:
    loss: Tensor
    valid_anchor_fraction: Tensor
    positive_similarity_mean: Tensor
    negative_similarity_mean: Tensor
    similarity_gap: Tensor


@dataclass(frozen=True)
class LogitDistillationObjective:
    loss: Tensor
    teacher_entropy_mean: Tensor
    student_entropy_mean: Tensor


@dataclass(frozen=True)
class FeatureDistillationObjective:
    loss: Tensor
    cosine_similarity_mean: Tensor
    rms_drift: Tensor


@dataclass(frozen=True)
class CDSPreservationObjective:
    loss: Tensor
    intra_loss: Tensor
    inter_loss: Tensor
    physical_absolute_support_mean: Tensor
    physical_relation_support_mean: Tensor
    physical_relevance_mean: Tensor
    teacher_absolute_support_mean: Tensor
    teacher_relation_support_mean: Tensor
    student_absolute_support_mean: Tensor
    student_relation_support_mean: Tensor


@dataclass(frozen=True)
class CDSSeparationObjective:
    loss: Tensor
    learned_positive_support_mean: Tensor
    learned_historical_rival_support_mean: Tensor
    learned_support_margin_mean: Tensor
    learned_support_margin_min: Tensor
    physical_confusability_mean: Tensor
    effective_pair_risk_mean: Tensor
    violation_fraction: Tensor

@dataclass(frozen=True)
class WholeCDSPICObjective:
    loss: Tensor
    valid_anchor_fraction: Tensor
    positive_similarity_mean: Tensor
    current_negative_similarity_mean: Tensor
    historical_negative_similarity_mean: Tensor
    learned_support_margin_mean: Tensor
    learned_support_margin_min: Tensor
    physical_confusability_mean: Tensor
    effective_pair_risk_mean: Tensor
    violation_fraction: Tensor


def _targets_local(labels_global: Tensor, class_ids: Tensor) -> Tensor:
    labels = torch.as_tensor(labels_global)
    ids = torch.as_tensor(class_ids)
    if labels.ndim != 1 or labels.numel() == 0 or labels.dtype != torch.long:
        raise ValueError("labels_global must be non-empty torch.long [N]")
    if ids.ndim != 1 or ids.numel() < 2 or ids.dtype != torch.long:
        raise ValueError("class_ids must be torch.long [C], C>=2")
    if labels.device != ids.device:
        raise ValueError("labels_global and class_ids must share device")
    matches = labels.unsqueeze(1).eq(ids.unsqueeze(0))
    if not bool(matches.sum(dim=1).eq(1).all()):
        raise ValueError("a target label is absent from class_ids")
    return matches.to(torch.long).argmax(dim=1)


def _weighted_mean(per_sample: Tensor, sample_weights: Tensor | None) -> Tensor:
    values = torch.as_tensor(per_sample)
    if values.ndim != 1 or values.numel() == 0:
        raise ValueError("per_sample must be non-empty [N]")
    if sample_weights is None:
        return values.mean()
    weights = torch.as_tensor(sample_weights)
    if weights.shape != values.shape or weights.device != values.device or weights.dtype != values.dtype:
        raise ValueError("sample_weights must match per-sample loss")
    if not bool(torch.isfinite(weights).all()) or bool((weights < 0).any()) or not bool(weights.sum() > 0):
        raise ValueError("sample_weights must be finite, non-negative and have positive mass")
    return (values * weights).sum() / weights.sum()


def _weighted_matrix_mean(values: Tensor, weights: Tensor) -> Tensor:
    value = torch.as_tensor(values)
    weight = torch.as_tensor(weights)
    if value.shape != weight.shape or value.device != weight.device or value.dtype != weight.dtype:
        raise ValueError("values and weights must have identical shape/device/dtype")
    if not bool(torch.isfinite(value).all()) or not bool(torch.isfinite(weight).all()):
        raise ValueError("weighted matrix contains NaN or Inf")
    if bool((weight < 0).any()):
        raise ValueError("weights must be non-negative")
    mass = weight.sum()
    if not bool(mass > 0):
        return value.sum() * 0.0
    return (value * weight).sum() / mass


def classification_objective(
    *,
    logits: Tensor,
    class_ids: Tensor,
    labels_global: Tensor,
    sample_weights: Tensor | None = None,
) -> ClassificationObjective:
    scores = torch.as_tensor(logits)
    ids = torch.as_tensor(class_ids)
    if scores.ndim != 2 or scores.size(0) == 0 or scores.size(1) < 2:
        raise ValueError("logits must be non-empty [N,C], C>=2")
    if ids.shape != (scores.size(1),) or ids.dtype != torch.long or ids.device != scores.device:
        raise ValueError("class_ids must match logits")
    if not bool(torch.isfinite(scores).all()):
        raise ValueError("logits contain NaN or Inf")
    targets = _targets_local(labels_global, ids)
    per_sample = F.cross_entropy(scores, targets, reduction="none")
    loss = _weighted_mean(per_sample, sample_weights)
    accuracy = scores.argmax(dim=1).eq(targets).to(scores.dtype).mean()
    if not bool(torch.isfinite(loss)):
        raise RuntimeError("classification loss became NaN or Inf")
    return ClassificationObjective(loss=loss, accuracy=accuracy, targets_local=targets)


def supervised_contrastive_q_loss(
    *,
    ordered_features: Tensor,
    labels_global: Tensor,
    logit_scale: Tensor,
    sample_weights: Tensor | None = None,
) -> SupervisedContrastiveObjective:
    q = torch.as_tensor(ordered_features)
    labels = torch.as_tensor(labels_global)
    if q.ndim != 3 or q.size(0) == 0 or q.size(1) < 2 or q.size(2) <= 0:
        raise ValueError("ordered_features must be non-empty [N,K,D], K>=2")
    if not torch.is_floating_point(q) or q.is_complex() or not bool(torch.isfinite(q).all()):
        raise ValueError("ordered_features must be finite real floating point")
    if labels.shape != (q.size(0),) or labels.dtype != torch.long or labels.device != q.device:
        raise ValueError("labels_global must be torch.long [N] on the Q device")

    scale = torch.as_tensor(logit_scale, device=q.device, dtype=q.dtype).detach()
    if scale.ndim != 0 or not bool(torch.isfinite(scale)) or not bool(scale > 0):
        raise ValueError("logit_scale must be a finite positive scalar")

    eps = torch.finfo(q.dtype).eps
    unit = F.normalize(q.reshape(q.size(0), -1), p=2, dim=1, eps=eps)
    similarity = (unit @ unit.t()).clamp(-1.0, 1.0) * scale

    n = int(q.size(0))
    eye = torch.eye(n, device=q.device, dtype=torch.bool)
    same = labels[:, None].eq(labels[None, :])
    positive_mask = same & ~eye
    negative_mask = (~same) & ~eye
    candidate_mask = ~eye

    masked_similarity = similarity.masked_fill(~candidate_mask, -torch.inf)
    log_denom = torch.logsumexp(masked_similarity, dim=1)
    positive_count = positive_mask.sum(dim=1)
    valid_anchor = positive_count > 0

    if bool(valid_anchor.any()):
        log_probability = similarity - log_denom.unsqueeze(1)
        per_anchor = -(
            log_probability * positive_mask.to(dtype=q.dtype)
        ).sum(dim=1) / positive_count.clamp_min(1).to(dtype=q.dtype)
        valid_loss = per_anchor[valid_anchor]
        if sample_weights is None:
            loss = valid_loss.mean()
        else:
            weights = torch.as_tensor(sample_weights)
            if weights.shape != (n,) or weights.device != q.device or weights.dtype != q.dtype:
                raise ValueError("sample_weights must be [N] and match Q device/dtype")
            loss = _weighted_mean(valid_loss, weights[valid_anchor])
    else:
        loss = q.sum() * 0.0

    positive_values = similarity[positive_mask]
    negative_values = similarity[negative_mask]
    zero = q.new_zeros(())
    positive_mean = positive_values.mean() if positive_values.numel() else zero
    negative_mean = negative_values.mean() if negative_values.numel() else zero
    gap = positive_mean - negative_mean
    valid_fraction = valid_anchor.to(dtype=q.dtype).mean()

    values = (loss, valid_fraction, positive_mean, negative_mean, gap)
    if not all(bool(torch.isfinite(value)) for value in values):
        raise RuntimeError("supervised Q contrastive objective became NaN or Inf")
    return SupervisedContrastiveObjective(
        loss=loss,
        valid_anchor_fraction=valid_fraction,
        positive_similarity_mean=positive_mean,
        negative_similarity_mean=negative_mean,
        similarity_gap=gap,
    )


def logit_distillation_loss(
    *,
    student_old_logits: Tensor,
    teacher_old_logits: Tensor,
    temperature: float,
    sample_weights: Tensor | None = None,
) -> LogitDistillationObjective:
    student = torch.as_tensor(student_old_logits)
    teacher = torch.as_tensor(teacher_old_logits).detach()
    if student.ndim != 2 or student.size(0) == 0 or student.size(1) < 2:
        raise ValueError("student_old_logits must be non-empty [N,O], O>=2")
    if teacher.shape != student.shape or teacher.device != student.device or teacher.dtype != student.dtype:
        raise ValueError("teacher_old_logits must match student_old_logits")
    if not bool(torch.isfinite(student).all()) or not bool(torch.isfinite(teacher).all()):
        raise ValueError("KD logits contain NaN or Inf")

    t = float(temperature)
    if not torch.isfinite(torch.tensor(t)) or t <= 0.0:
        raise ValueError("temperature must be finite and positive")

    student_log_probability = F.log_softmax(student / t, dim=1)
    teacher_log_probability = F.log_softmax(teacher / t, dim=1)
    teacher_probability = teacher_log_probability.exp()
    per_sample_kl = (
        teacher_probability * (teacher_log_probability - student_log_probability)
    ).sum(dim=1) * (t * t)
    loss = _weighted_mean(per_sample_kl, sample_weights)

    student_probability = student_log_probability.exp()
    teacher_entropy = -(teacher_probability * teacher_log_probability).sum(dim=1).mean()
    student_entropy = -(student_probability * student_log_probability).sum(dim=1).mean()
    if not all(bool(torch.isfinite(value)) for value in (loss, teacher_entropy, student_entropy)):
        raise RuntimeError("logit distillation became NaN or Inf")
    return LogitDistillationObjective(
        loss=loss,
        teacher_entropy_mean=teacher_entropy,
        student_entropy_mean=student_entropy,
    )



def ordered_feature_distillation_loss(
    *,
    reference_features: Tensor,
    current_features: Tensor,
    sample_weights: Tensor | None = None,
) -> FeatureDistillationObjective:
    """FCS-style feature KD in the same ordered Q coordinate used by CDS.

    The frozen previous representation is a training reference only.  Both Q^-
    and Q^+ are mapped to the canonical cosine/CDS coordinate before the loss so
    this term constrains semantic direction rather than arbitrary feature norm.
    Token order is preserved; no second representation is created.
    """
    reference = normalize_ordered_decision_features(
        torch.as_tensor(reference_features)
    ).detach()
    current = normalize_ordered_decision_features(torch.as_tensor(current_features))
    if current.shape != reference.shape:
        raise ValueError("current_features must match reference_features")
    if current.device != reference.device or current.dtype != reference.dtype:
        raise ValueError("reference/current Q must share device/dtype")
    delta = current - reference

    # IMPORTANT: do not use sqrt(mean(delta**2)) as the optimization loss.
    # At the beginning of an incremental phase the current backbone is an exact
    # copy of the frozen teacher, so delta == 0.  The forward value sqrt(0) is
    # finite, but its derivative is undefined and PyTorch returns NaN gradients.
    #
    # Because both tensors are already globally L2-normalized per sample,
    #   0.5 * ||q_t - q_{t-1}||_2^2 == 1 - cos(q_t, q_{t-1}).
    # This squared chord/cosine distance has the same semantic objective, is
    # exactly zero at phase start, and has an exactly zero (finite) gradient
    # when student and teacher coincide.  Token order is still preserved because
    # the distance is computed over the ordered KxD tensor without pooling into
    # a second representation.
    per_sample = 0.5 * delta.square().sum(dim=(1, 2))
    loss = _weighted_mean(per_sample, sample_weights)
    flat_ref = reference.reshape(reference.size(0), -1)
    flat_cur = current.reshape(current.size(0), -1)
    cosine = F.cosine_similarity(flat_cur, flat_ref, dim=1).mean()

    # RMS is diagnostic only.  Detaching avoids retaining a second sqrt branch
    # in the optimization graph and makes the phase-start zero-drift case safe.
    with torch.no_grad():
        rms = delta.detach().square().mean().sqrt()
    if not all(bool(torch.isfinite(v)) for v in (loss, cosine, rms)):
        raise RuntimeError("ordered feature distillation became NaN or Inf")
    return FeatureDistillationObjective(
        loss=loss,
        cosine_similarity_mean=cosine,
        rms_drift=rms,
    )

def historical_cds_preservation_loss(
    *,
    reference_features: Tensor,
    current_features: Tensor,
    physical_spectral_evidence: Tensor,
    reference_historical_structure: ClassStructureBatch,
    current_historical_structure: ClassStructureBatch,
) -> CDSPreservationObjective:
    """Preserve relations to the entire historical CDS on real current HSI.

    Teacher and student supports are both evaluated against the immutable
    finalized Phase-(t-1) historical CDS during incremental SGD.  Ordered feature
    KD keeps Q_t compatible with that reference; semantic migration happens once
    after convergence.  The full old-class support vector is preserved,
    and pairwise differences among all historical classes provide the inter-CDS
    term.  Native physical CDS weights the preservation and remains unchanged.
    """
    ref_struct = reference_historical_structure
    cur_struct = current_historical_structure
    if not isinstance(ref_struct, ClassStructureBatch) or ref_struct.class_count <= 0:
        raise ValueError("reference_historical_structure must be non-empty")
    if not isinstance(cur_struct, ClassStructureBatch) or cur_struct.class_count != ref_struct.class_count:
        raise ValueError("current_historical_structure must match historical class count")
    if not torch.equal(ref_struct.class_ids, cur_struct.class_ids):
        raise ValueError("reference/current historical class IDs disagree")
    if not torch.equal(ref_struct.physical_identity, cur_struct.physical_identity):
        raise ValueError("historical I_P must remain phase-stable")
    if not torch.equal(ref_struct.physical_variation, cur_struct.physical_variation):
        raise ValueError("historical V_P must remain phase-stable")

    reference = normalize_ordered_decision_features(torch.as_tensor(reference_features)).detach()
    current = normalize_ordered_decision_features(torch.as_tensor(current_features))
    physical = torch.as_tensor(physical_spectral_evidence).detach()
    if current.shape != reference.shape or current.device != reference.device or current.dtype != reference.dtype:
        raise ValueError("current_features must match reference_features")
    if physical.ndim != 3 or physical.size(0) != reference.size(0) or physical.size(2) != 1:
        raise ValueError("physical_spectral_evidence must be [N,B,1]")
    if physical.device != reference.device or physical.dtype != reference.dtype:
        raise ValueError("Q and physical spectral evidence must share device/dtype")

    with torch.no_grad():
        physical_absolute, physical_relation, physical_relevance = ordered_cds_support_components(
            physical,
            ref_struct.physical_identity.detach(),
            ref_struct.physical_variation.detach(),
        )
        teacher_absolute, teacher_relation, teacher_complete = ordered_cds_support_components(
            reference,
            ref_struct.learned_identity.detach(),
            ref_struct.learned_variation.detach(),
        )

    student_absolute, student_relation, student_complete = ordered_cds_support_components(
        current,
        cur_struct.learned_identity.detach(),
        cur_struct.learned_variation.detach(),
    )

    intra_error = 0.5 * (
        (student_absolute - teacher_absolute).abs()
        + (student_relation - teacher_relation).abs()
    )
    intra_loss = _weighted_matrix_mean(intra_error, physical_relevance)

    class_count = ref_struct.class_count
    if class_count >= 2:
        upper = torch.triu(
            torch.ones(class_count, class_count, device=current.device, dtype=torch.bool),
            diagonal=1,
        )
        teacher_pair = teacher_complete.unsqueeze(2) - teacher_complete.unsqueeze(1)
        student_pair = student_complete.unsqueeze(2) - student_complete.unsqueeze(1)
        pair_error = (student_pair - teacher_pair).abs()[:, upper]
        pair_weight = torch.sqrt(
            physical_relevance.unsqueeze(2) * physical_relevance.unsqueeze(1)
        )[:, upper]
        inter_loss = _weighted_matrix_mean(pair_error, pair_weight)
    else:
        inter_loss = current.sum() * 0.0

    loss = intra_loss + inter_loss
    values = (
        loss, intra_loss, inter_loss,
        physical_absolute.mean(), physical_relation.mean(), physical_relevance.mean(),
        teacher_absolute.mean(), teacher_relation.mean(),
        student_absolute.mean(), student_relation.mean(),
    )
    if not all(bool(torch.isfinite(value)) for value in values):
        raise RuntimeError("historical CDS preservation objective became NaN or Inf")

    return CDSPreservationObjective(
        loss=loss,
        intra_loss=intra_loss,
        inter_loss=inter_loss,
        physical_absolute_support_mean=physical_absolute.mean(),
        physical_relation_support_mean=physical_relation.mean(),
        physical_relevance_mean=physical_relevance.mean(),
        teacher_absolute_support_mean=teacher_absolute.mean(),
        teacher_relation_support_mean=teacher_relation.mean(),
        student_absolute_support_mean=student_absolute.mean(),
        student_relation_support_mean=student_relation.mean(),
    )


def whole_cds_pic_loss(
    *,
    current_features: Tensor,
    labels_global: Tensor,
    physical_spectral_evidence: Tensor,
    seen_cds: ClassDiscriminativeStructureState,
    historical_class_count: int,
    logit_scale: Tensor,
    sample_weights: Tensor | None = None,
) -> WholeCDSPICObjective:
    """Prototype-involved contrastive learning using the complete HSI Whole-CDS.

    Incremental batches contain only current classes.  Ordinary supervised
    contrastive learning therefore supplies only current/current attraction and
    repulsion and can keep increasing plasticity while historical structure
    deteriorates.  This objective follows the FCS separation principle but uses
    the HSI Whole-CDS rather than a stored prototype:

    * same current class: positive;
    * different current class: negative;
    * every historical ``I_Q``: historical negative anchor;
    * ``V_Q`` and ``I_P/V_P`` enter through Whole-CDS pair risk and
      sample-specific physical confusability, strengthening dangerous old/new
      negatives without creating a second classifier.

    The persistent memory remains exactly ``{I_Q,V_Q,I_P,V_P}``; no old feature
    rows or synthetic samples are stored.
    """
    if not isinstance(seen_cds, ClassDiscriminativeStructureState):
        raise TypeError("seen_cds must be ClassDiscriminativeStructureState")
    q = normalize_ordered_decision_features(torch.as_tensor(current_features))
    labels = torch.as_tensor(labels_global, device=q.device, dtype=torch.long).flatten()
    physical = torch.as_tensor(
        physical_spectral_evidence, device=q.device, dtype=q.dtype
    ).detach()
    if labels.shape != (q.size(0),):
        raise ValueError("labels_global must be [N]")
    if physical.ndim != 3 or physical.size(0) != q.size(0) or physical.size(2) != 1:
        raise ValueError("physical_spectral_evidence must be [N,B,1]")
    if not bool(torch.isfinite(physical).all()):
        raise ValueError("physical_spectral_evidence contains NaN or Inf")

    classes = seen_cds.classes
    ids = classes.class_ids.to(device=q.device, dtype=torch.long).flatten()
    count = int(ids.numel())
    old_count = int(historical_class_count)
    if count < 2 or old_count <= 0 or old_count >= count:
        raise ValueError("historical_class_count must split old and current classes")
    if classes.learned_identity.shape[1:] != q.shape[1:]:
        raise ValueError("current Q and learned CDS use different coordinates")
    if classes.physical_identity.shape[1:] != physical.shape[1:]:
        raise ValueError("physical evidence and physical CDS use different coordinates")

    matches = labels.unsqueeze(1).eq(ids.unsqueeze(0))
    if not bool(matches.sum(dim=1).eq(1).all()):
        raise ValueError("a current label is absent from seen CDS")
    target_index = matches.to(torch.long).argmax(dim=1)
    if bool((target_index < old_count).any()):
        raise ValueError("Whole-CDS PIC expects current-class labels only")

    scale = torch.as_tensor(logit_scale, device=q.device, dtype=q.dtype).detach()
    if scale.ndim != 0 or not bool(torch.isfinite(scale)) or not bool(scale > 0):
        raise ValueError("logit_scale must be a finite positive scalar")
    eps = torch.finfo(q.dtype).eps

    flat = F.normalize(q.reshape(q.size(0), -1), p=2, dim=1, eps=eps)
    current_similarity = (flat @ flat.t()).clamp(-1.0, 1.0) * scale
    n = int(q.size(0))
    eye = torch.eye(n, device=q.device, dtype=torch.bool)
    same = labels[:, None].eq(labels[None, :])
    positive_mask = same & ~eye
    current_negative_mask = (~same) & ~eye
    candidate_mask = ~eye

    historical_identity = classes.learned_identity[:old_count].to(
        device=q.device, dtype=q.dtype
    ).detach()
    historical_anchor = F.normalize(
        historical_identity.reshape(old_count, -1), p=2, dim=1, eps=eps
    )
    historical_similarity = (flat @ historical_anchor.t()).clamp(-1.0, 1.0) * scale

    # Whole-CDS pair risk contains learned identity+relation and physical
    # identity+relation structure.  Sample-specific physical support further
    # emphasizes spectral confusions actually present in the current sample.
    with torch.no_grad():
        _, _, physical_support = ordered_cds_support_components(
            physical,
            classes.physical_identity.to(device=q.device, dtype=q.dtype).detach(),
            classes.physical_variation.to(device=q.device, dtype=q.dtype).detach(),
        )
        effective_risk, _learned_risk, _physical_risk = whole_cds_pair_risk(seen_cds)
        effective_risk = effective_risk.to(device=q.device, dtype=q.dtype)
        rows = torch.arange(n, device=q.device)
        positive_physical = physical_support[rows, target_index]
        rival_physical = physical_support[:, :old_count]
        spectral_confusability = torch.sqrt(
            positive_physical.unsqueeze(1).clamp_min(0.0)
            * rival_physical.clamp_min(0.0)
        ).clamp(0.0, 1.0)
        pair_risk = effective_risk.index_select(0, target_index)[:, :old_count]
        historical_weight = 1.0 + pair_risk * spectral_confusability

    current_logits = current_similarity.masked_fill(~candidate_mask, -torch.inf)
    historical_logits = historical_similarity + historical_weight.clamp_min(eps).log()
    log_denom = torch.logsumexp(torch.cat((current_logits, historical_logits), dim=1), dim=1)
    positive_count = positive_mask.sum(dim=1)
    valid_anchor = positive_count > 0
    if bool(valid_anchor.any()):
        log_probability = current_similarity - log_denom.unsqueeze(1)
        per_anchor = -(
            log_probability * positive_mask.to(dtype=q.dtype)
        ).sum(dim=1) / positive_count.clamp_min(1).to(dtype=q.dtype)
        if sample_weights is None:
            loss = per_anchor[valid_anchor].mean()
        else:
            weights = torch.as_tensor(sample_weights)
            if weights.shape != (n,) or weights.device != q.device or weights.dtype != q.dtype:
                raise ValueError("sample_weights must be [N] and match Q device/dtype")
            loss = _weighted_mean(per_anchor[valid_anchor], weights[valid_anchor])
    else:
        loss = q.sum() * 0.0

    # Structural diagnostics retain the same interpretable old/new CDS margin
    # used by the previous separation loss.
    _, _, learned_support = ordered_cds_support_components(
        q,
        classes.learned_identity.to(device=q.device, dtype=q.dtype).detach(),
        classes.learned_variation.to(device=q.device, dtype=q.dtype).detach(),
    )
    rows = torch.arange(n, device=q.device)
    positive_learned = learned_support[rows, target_index]
    rival_learned = learned_support[:, :old_count]
    support_margin = positive_learned.unsqueeze(1) - rival_learned

    zero = q.new_zeros(())
    positive_values = current_similarity[positive_mask]
    current_negative_values = current_similarity[current_negative_mask]
    positive_mean = positive_values.mean() if positive_values.numel() else zero
    current_negative_mean = (
        current_negative_values.mean() if current_negative_values.numel() else zero
    )
    historical_negative_mean = historical_similarity.mean()
    valid_fraction = valid_anchor.to(dtype=q.dtype).mean()
    values = (
        loss,
        valid_fraction,
        positive_mean,
        current_negative_mean,
        historical_negative_mean,
        support_margin.mean(),
        support_margin.min(),
        spectral_confusability.mean(),
        pair_risk.mean(),
        support_margin.le(0.0).to(q.dtype).mean(),
    )
    if not all(bool(torch.isfinite(v)) for v in values):
        raise RuntimeError("Whole-CDS PIC objective became NaN or Inf")
    return WholeCDSPICObjective(
        loss=loss,
        valid_anchor_fraction=valid_fraction,
        positive_similarity_mean=positive_mean,
        current_negative_similarity_mean=current_negative_mean,
        historical_negative_similarity_mean=historical_negative_mean,
        learned_support_margin_mean=support_margin.mean(),
        learned_support_margin_min=support_margin.min(),
        physical_confusability_mean=spectral_confusability.mean(),
        effective_pair_risk_mean=pair_risk.mean(),
        violation_fraction=support_margin.le(0.0).to(q.dtype).mean(),
    )


def cross_phase_cds_separation_loss(
    *,
    current_features: Tensor,
    labels_global: Tensor,
    physical_spectral_evidence: Tensor,
    seen_cds: ClassDiscriminativeStructureState,
    historical_class_count: int,
    logit_scale: Tensor,
    sample_weights: Tensor | None = None,
) -> CDSSeparationObjective:
    """Keep real current Q outside migrated historical CDS regions.

    This is the feature-space counterpart of classifier-side Whole-CDS
    reconstruction.  The positive is the current class CDS.  Every historical
    class is a rival.  Separation is measured only in learned Q support, while
    native spectral evidence and whole-CDS pair risk *prioritize* spectrally
    credible old/new conflicts.  P therefore cannot invent a semantic boundary
    on its own and no classifier weight participates in this loss.
    """
    if not isinstance(seen_cds, ClassDiscriminativeStructureState):
        raise TypeError("seen_cds must be ClassDiscriminativeStructureState")
    q = normalize_ordered_decision_features(torch.as_tensor(current_features))
    p = torch.as_tensor(
        physical_spectral_evidence, device=q.device, dtype=q.dtype
    ).detach()
    labels = torch.as_tensor(labels_global, device=q.device, dtype=torch.long).flatten()
    if labels.shape != (q.size(0),):
        raise ValueError("labels_global must be [N]")
    if p.ndim != 3 or p.size(0) != q.size(0) or p.size(2) != 1:
        raise ValueError("physical_spectral_evidence must be [N,B,1]")
    if not bool(torch.isfinite(p).all()):
        raise ValueError("physical_spectral_evidence contains NaN or Inf")

    classes = seen_cds.classes
    ids = classes.class_ids.to(device=q.device, dtype=torch.long).flatten()
    count = int(ids.numel())
    old_count = int(historical_class_count)
    if count < 2 or old_count <= 0 or old_count >= count:
        raise ValueError("historical_class_count must split old and current classes")
    if classes.learned_identity.shape[1:] != q.shape[1:]:
        raise ValueError("current Q and learned CDS use different coordinates")
    if classes.physical_identity.shape[1:] != p.shape[1:]:
        raise ValueError("physical evidence and physical CDS use different coordinates")

    matches = labels.unsqueeze(1).eq(ids.unsqueeze(0))
    if not bool(matches.sum(dim=1).eq(1).all()):
        raise ValueError("a current label is absent from seen CDS")
    target_index = matches.to(torch.long).argmax(dim=1)
    if bool((target_index < old_count).any()):
        raise ValueError("cross-phase separation expects current-class labels only")

    _, _, learned_support = ordered_cds_support_components(
        q,
        classes.learned_identity.to(device=q.device, dtype=q.dtype).detach(),
        classes.learned_variation.to(device=q.device, dtype=q.dtype).detach(),
    )
    with torch.no_grad():
        _, _, physical_support = ordered_cds_support_components(
            p,
            classes.physical_identity.to(device=q.device, dtype=q.dtype).detach(),
            classes.physical_variation.to(device=q.device, dtype=q.dtype).detach(),
        )
        effective_risk, _learned_risk, _physical_risk = whole_cds_pair_risk(seen_cds)
        effective_risk = effective_risk.to(device=q.device, dtype=q.dtype)

    rows = torch.arange(q.size(0), device=q.device)
    positive_learned = learned_support[rows, target_index]
    positive_physical = physical_support[rows, target_index]
    rival_learned = learned_support[:, :old_count]
    rival_physical = physical_support[:, :old_count]

    margin = positive_learned.unsqueeze(1) - rival_learned
    pair_risk = effective_risk.index_select(0, target_index)[:, :old_count]
    # Sample-specific spectral ambiguity strengthens only a relation already
    # judged relevant in learned+physical class structure.
    spectral_confusability = torch.sqrt(
        positive_physical.unsqueeze(1).clamp_min(0.0)
        * rival_physical.clamp_min(0.0)
    ).clamp(0.0, 1.0)
    pair_weight = 1.0 + pair_risk * spectral_confusability

    scale = torch.as_tensor(logit_scale, device=q.device, dtype=q.dtype).detach()
    if scale.ndim != 0 or not bool(torch.isfinite(scale)) or not bool(scale > 0):
        raise ValueError("logit_scale must be a finite positive scalar")
    pair_loss = F.softplus(-scale * margin)
    per_sample = (pair_weight * pair_loss).sum(dim=1) / pair_weight.sum(dim=1).clamp_min(
        torch.finfo(q.dtype).eps
    )
    loss = _weighted_mean(per_sample, sample_weights)

    values = (
        loss,
        positive_learned.mean(),
        rival_learned.mean(),
        margin.mean(),
        margin.min(),
        spectral_confusability.mean(),
        pair_risk.mean(),
        margin.le(0.0).to(q.dtype).mean(),
    )
    if not all(bool(torch.isfinite(value)) for value in values):
        raise RuntimeError("cross-phase CDS separation became NaN or Inf")
    return CDSSeparationObjective(
        loss=loss,
        learned_positive_support_mean=positive_learned.mean(),
        learned_historical_rival_support_mean=rival_learned.mean(),
        learned_support_margin_mean=margin.mean(),
        learned_support_margin_min=margin.min(),
        physical_confusability_mean=spectral_confusability.mean(),
        effective_pair_risk_mean=pair_risk.mean(),
        violation_fraction=margin.le(0.0).to(q.dtype).mean(),
    )


__all__ = [
    "ClassificationObjective",
    "SupervisedContrastiveObjective",
    "LogitDistillationObjective",
    "FeatureDistillationObjective",
    "CDSPreservationObjective",
    "CDSSeparationObjective",
    "WholeCDSPICObjective",
    "classification_objective",
    "supervised_contrastive_q_loss",
    "logit_distillation_loss",
    "ordered_feature_distillation_loss",
    "historical_cds_preservation_loss",
    "whole_cds_pic_loss",
    "cross_phase_cds_separation_loss",
]








# from __future__ import annotations
# from dataclasses import dataclass

# import torch
# import torch.nn.functional as F

# from models.geometry_bank import (
#     ClassStructureBatch,
#     normalize_ordered_decision_features,
#     sanitize_ordered_variation,
# )

# Tensor = torch.Tensor


# @dataclass(frozen=True)
# class ClassificationObjective:
#     loss: Tensor
#     accuracy: Tensor
#     targets_local: Tensor


# @dataclass(frozen=True)
# class SupervisedContrastiveObjective:
#     loss: Tensor
#     valid_anchor_fraction: Tensor
#     positive_similarity_mean: Tensor
#     negative_similarity_mean: Tensor
#     similarity_gap: Tensor


# @dataclass(frozen=True)
# class ReferenceQConsistencyObjective:
#     loss: Tensor
#     cosine_similarity_mean: Tensor
#     rms_drift: Tensor


# @dataclass(frozen=True)
# class LogitDistillationObjective:
#     loss: Tensor
#     teacher_entropy_mean: Tensor
#     student_entropy_mean: Tensor


# @dataclass(frozen=True)
# class CDSPreservationObjective:
#     loss: Tensor
#     intra_loss: Tensor
#     inter_loss: Tensor
#     physical_absolute_support_mean: Tensor
#     physical_relation_support_mean: Tensor
#     physical_relevance_mean: Tensor
#     teacher_absolute_support_mean: Tensor
#     teacher_relation_support_mean: Tensor
#     student_absolute_support_mean: Tensor
#     student_relation_support_mean: Tensor


# def _targets_local(labels_global: Tensor, class_ids: Tensor) -> Tensor:
#     labels = torch.as_tensor(labels_global)
#     ids = torch.as_tensor(class_ids)
#     if labels.ndim != 1 or labels.numel() == 0 or labels.dtype != torch.long:
#         raise ValueError("labels_global must be non-empty torch.long [N]")
#     if ids.ndim != 1 or ids.numel() < 2 or ids.dtype != torch.long:
#         raise ValueError("class_ids must be torch.long [C], C>=2")
#     if labels.device != ids.device:
#         raise ValueError("labels_global and class_ids must share device")
#     matches = labels.unsqueeze(1).eq(ids.unsqueeze(0))
#     if not bool(matches.sum(dim=1).eq(1).all()):
#         raise ValueError("a target label is absent from class_ids")
#     return matches.to(torch.long).argmax(dim=1)


# def _weighted_mean(per_sample: Tensor, sample_weights: Tensor | None) -> Tensor:
#     values = torch.as_tensor(per_sample)
#     if values.ndim != 1 or values.numel() == 0:
#         raise ValueError("per_sample must be non-empty [N]")
#     if sample_weights is None:
#         return values.mean()
#     weights = torch.as_tensor(sample_weights)
#     if weights.shape != values.shape or weights.device != values.device or weights.dtype != values.dtype:
#         raise ValueError("sample_weights must match per-sample loss")
#     if not bool(torch.isfinite(weights).all()) or bool((weights < 0).any()) or not bool(weights.sum() > 0):
#         raise ValueError("sample_weights must be finite, non-negative and have positive mass")
#     return (values * weights).sum() / weights.sum()


# def _weighted_matrix_mean(values: Tensor, weights: Tensor) -> Tensor:
#     value = torch.as_tensor(values)
#     weight = torch.as_tensor(weights)
#     if value.shape != weight.shape or value.device != weight.device or value.dtype != weight.dtype:
#         raise ValueError("values and weights must have identical shape/device/dtype")
#     if not bool(torch.isfinite(value).all()) or not bool(torch.isfinite(weight).all()):
#         raise ValueError("weighted matrix contains NaN or Inf")
#     if bool((weight < 0).any()):
#         raise ValueError("weights must be non-negative")
#     mass = weight.sum()
#     if not bool(mass > 0):
#         return value.sum() * 0.0
#     return (value * weight).sum() / mass


# def classification_objective(
#     *,
#     logits: Tensor,
#     class_ids: Tensor,
#     labels_global: Tensor,
#     sample_weights: Tensor | None = None,
# ) -> ClassificationObjective:
#     scores = torch.as_tensor(logits)
#     ids = torch.as_tensor(class_ids)
#     if scores.ndim != 2 or scores.size(0) == 0 or scores.size(1) < 2:
#         raise ValueError("logits must be non-empty [N,C], C>=2")
#     if ids.shape != (scores.size(1),) or ids.dtype != torch.long or ids.device != scores.device:
#         raise ValueError("class_ids must match logits")
#     if not bool(torch.isfinite(scores).all()):
#         raise ValueError("logits contain NaN or Inf")
#     targets = _targets_local(labels_global, ids)
#     per_sample = F.cross_entropy(scores, targets, reduction="none")
#     loss = _weighted_mean(per_sample, sample_weights)
#     accuracy = scores.argmax(dim=1).eq(targets).to(scores.dtype).mean()
#     if not bool(torch.isfinite(loss)):
#         raise RuntimeError("classification loss became NaN or Inf")
#     return ClassificationObjective(loss=loss, accuracy=accuracy, targets_local=targets)


# def supervised_contrastive_q_loss(
#     *,
#     ordered_features: Tensor,
#     labels_global: Tensor,
#     logit_scale: Tensor,
#     sample_weights: Tensor | None = None,
# ) -> SupervisedContrastiveObjective:
#     q = torch.as_tensor(ordered_features)
#     labels = torch.as_tensor(labels_global)
#     if q.ndim != 3 or q.size(0) == 0 or q.size(1) < 2 or q.size(2) <= 0:
#         raise ValueError("ordered_features must be non-empty [N,K,D], K>=2")
#     if not torch.is_floating_point(q) or q.is_complex() or not bool(torch.isfinite(q).all()):
#         raise ValueError("ordered_features must be finite real floating point")
#     if labels.shape != (q.size(0),) or labels.dtype != torch.long or labels.device != q.device:
#         raise ValueError("labels_global must be torch.long [N] on the Q device")

#     scale = torch.as_tensor(logit_scale, device=q.device, dtype=q.dtype).detach()
#     if scale.ndim != 0 or not bool(torch.isfinite(scale)) or not bool(scale > 0):
#         raise ValueError("logit_scale must be a finite positive scalar")

#     eps = torch.finfo(q.dtype).eps
#     unit = F.normalize(q.reshape(q.size(0), -1), p=2, dim=1, eps=eps)
#     similarity = (unit @ unit.t()).clamp(-1.0, 1.0) * scale

#     n = int(q.size(0))
#     eye = torch.eye(n, device=q.device, dtype=torch.bool)
#     same = labels[:, None].eq(labels[None, :])
#     positive_mask = same & ~eye
#     negative_mask = (~same) & ~eye
#     candidate_mask = ~eye

#     masked_similarity = similarity.masked_fill(~candidate_mask, -torch.inf)
#     log_denom = torch.logsumexp(masked_similarity, dim=1)
#     positive_count = positive_mask.sum(dim=1)
#     valid_anchor = positive_count > 0

#     if bool(valid_anchor.any()):
#         log_probability = similarity - log_denom.unsqueeze(1)
#         per_anchor = -(
#             log_probability * positive_mask.to(dtype=q.dtype)
#         ).sum(dim=1) / positive_count.clamp_min(1).to(dtype=q.dtype)
#         valid_loss = per_anchor[valid_anchor]
#         if sample_weights is None:
#             loss = valid_loss.mean()
#         else:
#             weights = torch.as_tensor(sample_weights)
#             if weights.shape != (n,) or weights.device != q.device or weights.dtype != q.dtype:
#                 raise ValueError("sample_weights must be [N] and match Q device/dtype")
#             loss = _weighted_mean(valid_loss, weights[valid_anchor])
#     else:
#         loss = q.sum() * 0.0

#     positive_values = similarity[positive_mask]
#     negative_values = similarity[negative_mask]
#     zero = q.new_zeros(())
#     positive_mean = positive_values.mean() if positive_values.numel() else zero
#     negative_mean = negative_values.mean() if negative_values.numel() else zero
#     gap = positive_mean - negative_mean
#     valid_fraction = valid_anchor.to(dtype=q.dtype).mean()

#     values = (loss, valid_fraction, positive_mean, negative_mean, gap)
#     if not all(bool(torch.isfinite(value)) for value in values):
#         raise RuntimeError("supervised Q contrastive objective became NaN or Inf")
#     return SupervisedContrastiveObjective(
#         loss=loss,
#         valid_anchor_fraction=valid_fraction,
#         positive_similarity_mean=positive_mean,
#         negative_similarity_mean=negative_mean,
#         similarity_gap=gap,
#     )


# def reference_q_consistency_loss(
#     *,
#     reference_features: Tensor,
#     current_features: Tensor,
# ) -> ReferenceQConsistencyObjective:
#     reference = normalize_ordered_decision_features(torch.as_tensor(reference_features)).detach()
#     current = normalize_ordered_decision_features(torch.as_tensor(current_features))
#     if current.shape != reference.shape or current.device != reference.device or current.dtype != reference.dtype:
#         raise ValueError("current_features must match reference_features")
#     if not bool(torch.isfinite(reference).all()) or not bool(torch.isfinite(current).all()):
#         raise ValueError("reference/current Q contains NaN or Inf")

#     reference_flat = reference.reshape(reference.size(0), -1)
#     current_flat = current.reshape(current.size(0), -1)
#     cosine = (reference_flat * current_flat).sum(dim=1).clamp(-1.0, 1.0)
#     loss = (1.0 - cosine).mean()
#     rms = (current_flat - reference_flat).square().mean(dim=1).sqrt().mean()
#     if not bool(torch.isfinite(loss)) or not bool(torch.isfinite(rms)):
#         raise RuntimeError("reference-Q consistency became NaN or Inf")
#     return ReferenceQConsistencyObjective(
#         loss=loss,
#         cosine_similarity_mean=cosine.mean(),
#         rms_drift=rms,
#     )


# def logit_distillation_loss(
#     *,
#     student_old_logits: Tensor,
#     teacher_old_logits: Tensor,
#     temperature: float,
#     sample_weights: Tensor | None = None,
# ) -> LogitDistillationObjective:
#     student = torch.as_tensor(student_old_logits)
#     teacher = torch.as_tensor(teacher_old_logits).detach()
#     if student.ndim != 2 or student.size(0) == 0 or student.size(1) < 2:
#         raise ValueError("student_old_logits must be non-empty [N,O], O>=2")
#     if teacher.shape != student.shape or teacher.device != student.device or teacher.dtype != student.dtype:
#         raise ValueError("teacher_old_logits must match student_old_logits")
#     if not bool(torch.isfinite(student).all()) or not bool(torch.isfinite(teacher).all()):
#         raise ValueError("KD logits contain NaN or Inf")

#     t = float(temperature)
#     if not torch.isfinite(torch.tensor(t)) or t <= 0.0:
#         raise ValueError("temperature must be finite and positive")

#     student_log_probability = F.log_softmax(student / t, dim=1)
#     teacher_log_probability = F.log_softmax(teacher / t, dim=1)
#     teacher_probability = teacher_log_probability.exp()
#     per_sample_kl = (
#         teacher_probability * (teacher_log_probability - student_log_probability)
#     ).sum(dim=1) * (t * t)
#     loss = _weighted_mean(per_sample_kl, sample_weights)

#     student_probability = student_log_probability.exp()
#     teacher_entropy = -(teacher_probability * teacher_log_probability).sum(dim=1).mean()
#     student_entropy = -(student_probability * student_log_probability).sum(dim=1).mean()
#     if not all(bool(torch.isfinite(value)) for value in (loss, teacher_entropy, student_entropy)):
#         raise RuntimeError("logit distillation became NaN or Inf")
#     return LogitDistillationObjective(
#         loss=loss,
#         teacher_entropy_mean=teacher_entropy,
#         student_entropy_mean=student_entropy,
#     )


# def _ordered_cds_support_components(
#     sample: Tensor,
#     identity: Tensor,
#     variation: Tensor,
# ) -> tuple[Tensor, Tensor, Tensor]:
#     """Absolute, ordered-relation and complete CDS support [N,C]."""
#     x = torch.as_tensor(sample)
#     center = torch.as_tensor(identity)
#     robust_scale = sanitize_ordered_variation(torch.as_tensor(variation))
#     if x.ndim != 3 or x.size(0) == 0 or x.size(1) < 2 or x.size(2) <= 0:
#         raise ValueError("sample must be non-empty [N,L,F], L>=2")
#     if center.ndim != 3 or tuple(center.shape[1:]) != tuple(x.shape[1:]):
#         raise ValueError("identity must be [C,L,F] matching sample")
#     class_count, length, feature_count = center.shape
#     if robust_scale.shape != (class_count, length, length, feature_count):
#         raise ValueError("variation must be [C,L,L,F] matching identity")
#     if any(value.device != x.device or value.dtype != x.dtype for value in (center, robust_scale)):
#         raise ValueError("sample/identity/variation must share device/dtype")
#     if not all(bool(torch.isfinite(value).all()) for value in (x, center, robust_scale)):
#         raise ValueError("sample CDS tensors contain NaN or Inf")

#     eps = torch.finfo(x.dtype).eps
#     diagonal = torch.arange(length, device=x.device)
#     absolute_scale = robust_scale[:, diagonal, diagonal, :]
#     absolute_deviation = (x[:, None] - center[None]).abs()
#     absolute_element = absolute_scale[None] / (
#         absolute_scale[None] + absolute_deviation + eps
#     )
#     absolute_both_zero = (absolute_scale[None] <= eps) & (absolute_deviation <= eps)
#     absolute_element = torch.where(absolute_both_zero, torch.ones_like(absolute_element), absolute_element)
#     absolute_support = absolute_element.mean(dim=(2, 3)).clamp(0.0, 1.0)

#     upper = torch.triu(torch.ones(length, length, device=x.device, dtype=torch.bool), diagonal=1)
#     if int(upper.sum().item()) <= 0:
#         raise RuntimeError("ordered CDS support has no off-diagonal relations")

#     sample_relation = x[:, :, None, :] - x[:, None, :, :]
#     relation_outputs: list[Tensor] = []
#     for class_index in range(class_count):
#         identity_relation = center[class_index, :, None, :] - center[class_index, None, :, :]
#         deviation = (sample_relation - identity_relation.unsqueeze(0)).abs()
#         scale = robust_scale[class_index].unsqueeze(0)
#         element = scale / (scale + deviation + eps)
#         both_zero = (scale <= eps) & (deviation <= eps)
#         element = torch.where(both_zero, torch.ones_like(element), element)
#         selected = element[:, upper, :]
#         relation_outputs.append(selected.mean(dim=(1, 2)).clamp(0.0, 1.0))

#     relation_support = torch.stack(relation_outputs, dim=1)
#     complete_support = torch.sqrt(
#         absolute_support.clamp_min(0.0) * relation_support.clamp_min(0.0)
#     ).clamp(0.0, 1.0)
#     return absolute_support, relation_support, complete_support


# def historical_cds_preservation_loss(
#     *,
#     reference_features: Tensor,
#     current_features: Tensor,
#     physical_spectral_evidence: Tensor,
#     reference_historical_structure: ClassStructureBatch,
#     current_historical_structure: ClassStructureBatch,
# ) -> CDSPreservationObjective:
#     """Preserve relations to the entire historical CDS on real current HSI.

#     Teacher support is evaluated in the finalized Phase-(t-1) Q coordinate.
#     Student support is evaluated in the current Q coordinate using the task-wise
#     transported historical CDS.  The full old-class support vector is preserved,
#     and pairwise differences among all historical classes provide the inter-CDS
#     term.  Native physical CDS weights the preservation and remains unchanged.
#     """
#     ref_struct = reference_historical_structure
#     cur_struct = current_historical_structure
#     if not isinstance(ref_struct, ClassStructureBatch) or ref_struct.class_count <= 0:
#         raise ValueError("reference_historical_structure must be non-empty")
#     if not isinstance(cur_struct, ClassStructureBatch) or cur_struct.class_count != ref_struct.class_count:
#         raise ValueError("current_historical_structure must match historical class count")
#     if not torch.equal(ref_struct.class_ids, cur_struct.class_ids):
#         raise ValueError("reference/current historical class IDs disagree")
#     if not torch.equal(ref_struct.physical_identity, cur_struct.physical_identity):
#         raise ValueError("historical I_P must remain phase-stable")
#     if not torch.equal(ref_struct.physical_variation, cur_struct.physical_variation):
#         raise ValueError("historical V_P must remain phase-stable")

#     reference = normalize_ordered_decision_features(torch.as_tensor(reference_features)).detach()
#     current = normalize_ordered_decision_features(torch.as_tensor(current_features))
#     physical = torch.as_tensor(physical_spectral_evidence).detach()
#     if current.shape != reference.shape or current.device != reference.device or current.dtype != reference.dtype:
#         raise ValueError("current_features must match reference_features")
#     if physical.ndim != 3 or physical.size(0) != reference.size(0) or physical.size(2) != 1:
#         raise ValueError("physical_spectral_evidence must be [N,B,1]")
#     if physical.device != reference.device or physical.dtype != reference.dtype:
#         raise ValueError("Q and physical spectral evidence must share device/dtype")

#     with torch.no_grad():
#         physical_absolute, physical_relation, physical_relevance = _ordered_cds_support_components(
#             physical,
#             ref_struct.physical_identity.detach(),
#             ref_struct.physical_variation.detach(),
#         )
#         teacher_absolute, teacher_relation, teacher_complete = _ordered_cds_support_components(
#             reference,
#             ref_struct.learned_identity.detach(),
#             ref_struct.learned_variation.detach(),
#         )

#     student_absolute, student_relation, student_complete = _ordered_cds_support_components(
#         current,
#         cur_struct.learned_identity.detach(),
#         cur_struct.learned_variation.detach(),
#     )

#     intra_error = 0.5 * (
#         (student_absolute - teacher_absolute).abs()
#         + (student_relation - teacher_relation).abs()
#     )
#     intra_loss = _weighted_matrix_mean(intra_error, physical_relevance)

#     class_count = ref_struct.class_count
#     if class_count >= 2:
#         upper = torch.triu(
#             torch.ones(class_count, class_count, device=current.device, dtype=torch.bool),
#             diagonal=1,
#         )
#         teacher_pair = teacher_complete.unsqueeze(2) - teacher_complete.unsqueeze(1)
#         student_pair = student_complete.unsqueeze(2) - student_complete.unsqueeze(1)
#         pair_error = (student_pair - teacher_pair).abs()[:, upper]
#         pair_weight = torch.sqrt(
#             physical_relevance.unsqueeze(2) * physical_relevance.unsqueeze(1)
#         )[:, upper]
#         inter_loss = _weighted_matrix_mean(pair_error, pair_weight)
#     else:
#         inter_loss = current.sum() * 0.0

#     loss = intra_loss + inter_loss
#     values = (
#         loss, intra_loss, inter_loss,
#         physical_absolute.mean(), physical_relation.mean(), physical_relevance.mean(),
#         teacher_absolute.mean(), teacher_relation.mean(),
#         student_absolute.mean(), student_relation.mean(),
#     )
#     if not all(bool(torch.isfinite(value)) for value in values):
#         raise RuntimeError("historical CDS preservation objective became NaN or Inf")

#     return CDSPreservationObjective(
#         loss=loss,
#         intra_loss=intra_loss,
#         inter_loss=inter_loss,
#         physical_absolute_support_mean=physical_absolute.mean(),
#         physical_relation_support_mean=physical_relation.mean(),
#         physical_relevance_mean=physical_relevance.mean(),
#         teacher_absolute_support_mean=teacher_absolute.mean(),
#         teacher_relation_support_mean=teacher_relation.mean(),
#         student_absolute_support_mean=student_absolute.mean(),
#         student_relation_support_mean=student_relation.mean(),
#     )


# __all__ = [
#     "ClassificationObjective",
#     "SupervisedContrastiveObjective",
#     "ReferenceQConsistencyObjective",
#     "LogitDistillationObjective",
#     "CDSPreservationObjective",
#     "classification_objective",
#     "supervised_contrastive_q_loss",
#     "reference_q_consistency_loss",
#     "logit_distillation_loss",
#     "historical_cds_preservation_loss",
# ]





# from __future__ import annotations

# """Training objectives for the finalized non-exemplar HSI-CDS architecture.

# Incremental representation adaptation uses only real current HSI.  Historical
# information enters only through the frozen previous model and persistent CDS
# {I_Q,V_Q,I_P,V_P,N}.  No transient HSI reconstruction, pseudo historical feature,
# Gaussian/covariance replay, or hard-IQR region is defined here.
# """

# from dataclasses import dataclass

# import torch
# import torch.nn.functional as F

# from models.geometry_bank import (
#     ClassStructureBatch,
#     normalize_ordered_decision_features,
#     sanitize_ordered_variation,
# )

# Tensor = torch.Tensor


# @dataclass(frozen=True)
# class ClassificationObjective:
#     loss: Tensor
#     accuracy: Tensor
#     targets_local: Tensor


# @dataclass(frozen=True)
# class SupervisedContrastiveObjective:
#     loss: Tensor
#     valid_anchor_fraction: Tensor
#     positive_similarity_mean: Tensor
#     negative_similarity_mean: Tensor
#     similarity_gap: Tensor


# @dataclass(frozen=True)
# class ReferenceQConsistencyObjective:
#     loss: Tensor
#     cosine_similarity_mean: Tensor
#     rms_drift: Tensor


# @dataclass(frozen=True)
# class LogitDistillationObjective:
#     loss: Tensor
#     teacher_entropy_mean: Tensor
#     student_entropy_mean: Tensor


# @dataclass(frozen=True)
# class CDSPreservationObjective:
#     loss: Tensor
#     intra_loss: Tensor
#     inter_loss: Tensor
#     physical_absolute_support_mean: Tensor
#     physical_relation_support_mean: Tensor
#     physical_relevance_mean: Tensor
#     teacher_absolute_support_mean: Tensor
#     teacher_relation_support_mean: Tensor
#     student_absolute_support_mean: Tensor
#     student_relation_support_mean: Tensor


# def _targets_local(labels_global: Tensor, class_ids: Tensor) -> Tensor:
#     labels = torch.as_tensor(labels_global)
#     ids = torch.as_tensor(class_ids)
#     if labels.ndim != 1 or labels.numel() == 0 or labels.dtype != torch.long:
#         raise ValueError("labels_global must be non-empty torch.long [N]")
#     if ids.ndim != 1 or ids.numel() < 2 or ids.dtype != torch.long:
#         raise ValueError("class_ids must be torch.long [C], C>=2")
#     if labels.device != ids.device:
#         raise ValueError("labels_global and class_ids must share device")
#     matches = labels.unsqueeze(1).eq(ids.unsqueeze(0))
#     if not bool(matches.sum(dim=1).eq(1).all()):
#         raise ValueError("a target label is absent from class_ids")
#     return matches.to(torch.long).argmax(dim=1)


# def _weighted_mean(per_sample: Tensor, sample_weights: Tensor | None) -> Tensor:
#     values = torch.as_tensor(per_sample)
#     if values.ndim != 1 or values.numel() == 0:
#         raise ValueError("per_sample must be non-empty [N]")
#     if sample_weights is None:
#         return values.mean()
#     weights = torch.as_tensor(sample_weights)
#     if weights.shape != values.shape or weights.device != values.device or weights.dtype != values.dtype:
#         raise ValueError("sample_weights must match per-sample loss")
#     if not bool(torch.isfinite(weights).all()) or bool((weights < 0).any()) or not bool(weights.sum() > 0):
#         raise ValueError("sample_weights must be finite, non-negative and have positive mass")
#     return (values * weights).sum() / weights.sum()


# def _weighted_matrix_mean(values: Tensor, weights: Tensor) -> Tensor:
#     value = torch.as_tensor(values)
#     weight = torch.as_tensor(weights)
#     if value.shape != weight.shape or value.device != weight.device or value.dtype != weight.dtype:
#         raise ValueError("values and weights must have identical shape/device/dtype")
#     if not bool(torch.isfinite(value).all()) or not bool(torch.isfinite(weight).all()):
#         raise ValueError("weighted matrix contains NaN or Inf")
#     if bool((weight < 0).any()):
#         raise ValueError("weights must be non-negative")
#     mass = weight.sum()
#     if not bool(mass > 0):
#         return value.sum() * 0.0
#     return (value * weight).sum() / mass


# def classification_objective(
#     *,
#     logits: Tensor,
#     class_ids: Tensor,
#     labels_global: Tensor,
#     sample_weights: Tensor | None = None,
# ) -> ClassificationObjective:
#     scores = torch.as_tensor(logits)
#     ids = torch.as_tensor(class_ids)
#     if scores.ndim != 2 or scores.size(0) == 0 or scores.size(1) < 2:
#         raise ValueError("logits must be non-empty [N,C], C>=2")
#     if ids.shape != (scores.size(1),) or ids.dtype != torch.long or ids.device != scores.device:
#         raise ValueError("class_ids must match logits")
#     if not bool(torch.isfinite(scores).all()):
#         raise ValueError("logits contain NaN or Inf")
#     targets = _targets_local(labels_global, ids)
#     per_sample = F.cross_entropy(scores, targets, reduction="none")
#     loss = _weighted_mean(per_sample, sample_weights)
#     accuracy = scores.argmax(dim=1).eq(targets).to(scores.dtype).mean()
#     if not bool(torch.isfinite(loss)):
#         raise RuntimeError("classification loss became NaN or Inf")
#     return ClassificationObjective(loss=loss, accuracy=accuracy, targets_local=targets)


# def supervised_contrastive_q_loss(
#     *,
#     ordered_features: Tensor,
#     labels_global: Tensor,
#     logit_scale: Tensor,
#     sample_weights: Tensor | None = None,
# ) -> SupervisedContrastiveObjective:
#     q = torch.as_tensor(ordered_features)
#     labels = torch.as_tensor(labels_global)
#     if q.ndim != 3 or q.size(0) == 0 or q.size(1) < 2 or q.size(2) <= 0:
#         raise ValueError("ordered_features must be non-empty [N,K,D], K>=2")
#     if not torch.is_floating_point(q) or q.is_complex() or not bool(torch.isfinite(q).all()):
#         raise ValueError("ordered_features must be finite real floating point")
#     if labels.shape != (q.size(0),) or labels.dtype != torch.long or labels.device != q.device:
#         raise ValueError("labels_global must be torch.long [N] on the Q device")

#     scale = torch.as_tensor(logit_scale, device=q.device, dtype=q.dtype).detach()
#     if scale.ndim != 0 or not bool(torch.isfinite(scale)) or not bool(scale > 0):
#         raise ValueError("logit_scale must be a finite positive scalar")

#     eps = torch.finfo(q.dtype).eps
#     unit = F.normalize(q.reshape(q.size(0), -1), p=2, dim=1, eps=eps)
#     similarity = (unit @ unit.t()).clamp(-1.0, 1.0) * scale

#     n = int(q.size(0))
#     eye = torch.eye(n, device=q.device, dtype=torch.bool)
#     same = labels[:, None].eq(labels[None, :])
#     positive_mask = same & ~eye
#     negative_mask = (~same) & ~eye
#     candidate_mask = ~eye

#     masked_similarity = similarity.masked_fill(~candidate_mask, -torch.inf)
#     log_denom = torch.logsumexp(masked_similarity, dim=1)
#     positive_count = positive_mask.sum(dim=1)
#     valid_anchor = positive_count > 0

#     if bool(valid_anchor.any()):
#         log_probability = similarity - log_denom.unsqueeze(1)
#         per_anchor = -(
#             log_probability * positive_mask.to(dtype=q.dtype)
#         ).sum(dim=1) / positive_count.clamp_min(1).to(dtype=q.dtype)
#         valid_loss = per_anchor[valid_anchor]
#         if sample_weights is None:
#             loss = valid_loss.mean()
#         else:
#             weights = torch.as_tensor(sample_weights)
#             if weights.shape != (n,) or weights.device != q.device or weights.dtype != q.dtype:
#                 raise ValueError("sample_weights must be [N] and match Q device/dtype")
#             loss = _weighted_mean(valid_loss, weights[valid_anchor])
#     else:
#         loss = q.sum() * 0.0

#     positive_values = similarity[positive_mask]
#     negative_values = similarity[negative_mask]
#     zero = q.new_zeros(())
#     positive_mean = positive_values.mean() if positive_values.numel() else zero
#     negative_mean = negative_values.mean() if negative_values.numel() else zero
#     gap = positive_mean - negative_mean
#     valid_fraction = valid_anchor.to(dtype=q.dtype).mean()

#     values = (loss, valid_fraction, positive_mean, negative_mean, gap)
#     if not all(bool(torch.isfinite(value)) for value in values):
#         raise RuntimeError("supervised Q contrastive objective became NaN or Inf")
#     return SupervisedContrastiveObjective(
#         loss=loss,
#         valid_anchor_fraction=valid_fraction,
#         positive_similarity_mean=positive_mean,
#         negative_similarity_mean=negative_mean,
#         similarity_gap=gap,
#     )


# def reference_q_consistency_loss(
#     *,
#     reference_features: Tensor,
#     current_features: Tensor,
# ) -> ReferenceQConsistencyObjective:
#     reference = normalize_ordered_decision_features(torch.as_tensor(reference_features)).detach()
#     current = normalize_ordered_decision_features(torch.as_tensor(current_features))
#     if current.shape != reference.shape or current.device != reference.device or current.dtype != reference.dtype:
#         raise ValueError("current_features must match reference_features")
#     if not bool(torch.isfinite(reference).all()) or not bool(torch.isfinite(current).all()):
#         raise ValueError("reference/current Q contains NaN or Inf")

#     reference_flat = reference.reshape(reference.size(0), -1)
#     current_flat = current.reshape(current.size(0), -1)
#     cosine = (reference_flat * current_flat).sum(dim=1).clamp(-1.0, 1.0)
#     loss = (1.0 - cosine).mean()
#     rms = (current_flat - reference_flat).square().mean(dim=1).sqrt().mean()
#     if not bool(torch.isfinite(loss)) or not bool(torch.isfinite(rms)):
#         raise RuntimeError("reference-Q consistency became NaN or Inf")
#     return ReferenceQConsistencyObjective(
#         loss=loss,
#         cosine_similarity_mean=cosine.mean(),
#         rms_drift=rms,
#     )


# def logit_distillation_loss(
#     *,
#     student_old_logits: Tensor,
#     teacher_old_logits: Tensor,
#     temperature: float,
#     sample_weights: Tensor | None = None,
# ) -> LogitDistillationObjective:
#     student = torch.as_tensor(student_old_logits)
#     teacher = torch.as_tensor(teacher_old_logits).detach()
#     if student.ndim != 2 or student.size(0) == 0 or student.size(1) < 2:
#         raise ValueError("student_old_logits must be non-empty [N,O], O>=2")
#     if teacher.shape != student.shape or teacher.device != student.device or teacher.dtype != student.dtype:
#         raise ValueError("teacher_old_logits must match student_old_logits")
#     if not bool(torch.isfinite(student).all()) or not bool(torch.isfinite(teacher).all()):
#         raise ValueError("KD logits contain NaN or Inf")

#     t = float(temperature)
#     if not torch.isfinite(torch.tensor(t)) or t <= 0.0:
#         raise ValueError("temperature must be finite and positive")

#     student_log_probability = F.log_softmax(student / t, dim=1)
#     teacher_log_probability = F.log_softmax(teacher / t, dim=1)
#     teacher_probability = teacher_log_probability.exp()
#     per_sample_kl = (
#         teacher_probability * (teacher_log_probability - student_log_probability)
#     ).sum(dim=1) * (t * t)
#     loss = _weighted_mean(per_sample_kl, sample_weights)

#     student_probability = student_log_probability.exp()
#     teacher_entropy = -(teacher_probability * teacher_log_probability).sum(dim=1).mean()
#     student_entropy = -(student_probability * student_log_probability).sum(dim=1).mean()
#     if not all(bool(torch.isfinite(value)) for value in (loss, teacher_entropy, student_entropy)):
#         raise RuntimeError("logit distillation became NaN or Inf")
#     return LogitDistillationObjective(
#         loss=loss,
#         teacher_entropy_mean=teacher_entropy,
#         student_entropy_mean=student_entropy,
#     )


# def _ordered_cds_support_components(
#     sample: Tensor,
#     identity: Tensor,
#     variation: Tensor,
# ) -> tuple[Tensor, Tensor, Tensor]:
#     """Absolute, ordered-relation and complete CDS support [N,C]."""
#     x = torch.as_tensor(sample)
#     center = torch.as_tensor(identity)
#     robust_scale = sanitize_ordered_variation(torch.as_tensor(variation))
#     if x.ndim != 3 or x.size(0) == 0 or x.size(1) < 2 or x.size(2) <= 0:
#         raise ValueError("sample must be non-empty [N,L,F], L>=2")
#     if center.ndim != 3 or tuple(center.shape[1:]) != tuple(x.shape[1:]):
#         raise ValueError("identity must be [C,L,F] matching sample")
#     class_count, length, feature_count = center.shape
#     if robust_scale.shape != (class_count, length, length, feature_count):
#         raise ValueError("variation must be [C,L,L,F] matching identity")
#     if any(value.device != x.device or value.dtype != x.dtype for value in (center, robust_scale)):
#         raise ValueError("sample/identity/variation must share device/dtype")
#     if not all(bool(torch.isfinite(value).all()) for value in (x, center, robust_scale)):
#         raise ValueError("sample CDS tensors contain NaN or Inf")

#     eps = torch.finfo(x.dtype).eps
#     diagonal = torch.arange(length, device=x.device)
#     absolute_scale = robust_scale[:, diagonal, diagonal, :]
#     absolute_deviation = (x[:, None] - center[None]).abs()
#     absolute_element = absolute_scale[None] / (
#         absolute_scale[None] + absolute_deviation + eps
#     )
#     absolute_both_zero = (absolute_scale[None] <= eps) & (absolute_deviation <= eps)
#     absolute_element = torch.where(absolute_both_zero, torch.ones_like(absolute_element), absolute_element)
#     absolute_support = absolute_element.mean(dim=(2, 3)).clamp(0.0, 1.0)

#     upper = torch.triu(torch.ones(length, length, device=x.device, dtype=torch.bool), diagonal=1)
#     if int(upper.sum().item()) <= 0:
#         raise RuntimeError("ordered CDS support has no off-diagonal relations")

#     sample_relation = x[:, :, None, :] - x[:, None, :, :]
#     relation_outputs: list[Tensor] = []
#     for class_index in range(class_count):
#         identity_relation = center[class_index, :, None, :] - center[class_index, None, :, :]
#         deviation = (sample_relation - identity_relation.unsqueeze(0)).abs()
#         scale = robust_scale[class_index].unsqueeze(0)
#         element = scale / (scale + deviation + eps)
#         both_zero = (scale <= eps) & (deviation <= eps)
#         element = torch.where(both_zero, torch.ones_like(element), element)
#         selected = element[:, upper, :]
#         relation_outputs.append(selected.mean(dim=(1, 2)).clamp(0.0, 1.0))

#     relation_support = torch.stack(relation_outputs, dim=1)
#     complete_support = torch.sqrt(
#         absolute_support.clamp_min(0.0) * relation_support.clamp_min(0.0)
#     ).clamp(0.0, 1.0)
#     return absolute_support, relation_support, complete_support


# def historical_cds_preservation_loss(
#     *,
#     reference_features: Tensor,
#     current_features: Tensor,
#     physical_spectral_evidence: Tensor,
#     reference_historical_structure: ClassStructureBatch,
#     current_historical_structure: ClassStructureBatch,
# ) -> CDSPreservationObjective:
#     """Preserve relations to the entire historical CDS on real current HSI.

#     Teacher support is evaluated in the finalized Phase-(t-1) Q coordinate.
#     Student support is evaluated in the current Q coordinate using the task-wise
#     transported historical CDS.  The full old-class support vector is preserved,
#     and pairwise differences among all historical classes provide the inter-CDS
#     term.  Native physical CDS weights the preservation and remains unchanged.
#     """
#     ref_struct = reference_historical_structure
#     cur_struct = current_historical_structure
#     if not isinstance(ref_struct, ClassStructureBatch) or ref_struct.class_count <= 0:
#         raise ValueError("reference_historical_structure must be non-empty")
#     if not isinstance(cur_struct, ClassStructureBatch) or cur_struct.class_count != ref_struct.class_count:
#         raise ValueError("current_historical_structure must match historical class count")
#     if not torch.equal(ref_struct.class_ids, cur_struct.class_ids):
#         raise ValueError("reference/current historical class IDs disagree")
#     if not torch.equal(ref_struct.physical_identity, cur_struct.physical_identity):
#         raise ValueError("historical I_P must remain phase-stable")
#     if not torch.equal(ref_struct.physical_variation, cur_struct.physical_variation):
#         raise ValueError("historical V_P must remain phase-stable")

#     reference = normalize_ordered_decision_features(torch.as_tensor(reference_features)).detach()
#     current = normalize_ordered_decision_features(torch.as_tensor(current_features))
#     physical = torch.as_tensor(physical_spectral_evidence).detach()
#     if current.shape != reference.shape or current.device != reference.device or current.dtype != reference.dtype:
#         raise ValueError("current_features must match reference_features")
#     if physical.ndim != 3 or physical.size(0) != reference.size(0) or physical.size(2) != 1:
#         raise ValueError("physical_spectral_evidence must be [N,B,1]")
#     if physical.device != reference.device or physical.dtype != reference.dtype:
#         raise ValueError("Q and physical spectral evidence must share device/dtype")

#     with torch.no_grad():
#         physical_absolute, physical_relation, physical_relevance = _ordered_cds_support_components(
#             physical,
#             ref_struct.physical_identity.detach(),
#             ref_struct.physical_variation.detach(),
#         )
#         teacher_absolute, teacher_relation, teacher_complete = _ordered_cds_support_components(
#             reference,
#             ref_struct.learned_identity.detach(),
#             ref_struct.learned_variation.detach(),
#         )

#     student_absolute, student_relation, student_complete = _ordered_cds_support_components(
#         current,
#         cur_struct.learned_identity.detach(),
#         cur_struct.learned_variation.detach(),
#     )

#     intra_error = 0.5 * (
#         (student_absolute - teacher_absolute).abs()
#         + (student_relation - teacher_relation).abs()
#     )
#     intra_loss = _weighted_matrix_mean(intra_error, physical_relevance)

#     class_count = ref_struct.class_count
#     if class_count >= 2:
#         upper = torch.triu(
#             torch.ones(class_count, class_count, device=current.device, dtype=torch.bool),
#             diagonal=1,
#         )
#         teacher_pair = teacher_complete.unsqueeze(2) - teacher_complete.unsqueeze(1)
#         student_pair = student_complete.unsqueeze(2) - student_complete.unsqueeze(1)
#         pair_error = (student_pair - teacher_pair).abs()[:, upper]
#         pair_weight = torch.sqrt(
#             physical_relevance.unsqueeze(2) * physical_relevance.unsqueeze(1)
#         )[:, upper]
#         inter_loss = _weighted_matrix_mean(pair_error, pair_weight)
#     else:
#         inter_loss = current.sum() * 0.0

#     loss = intra_loss + inter_loss
#     values = (
#         loss, intra_loss, inter_loss,
#         physical_absolute.mean(), physical_relation.mean(), physical_relevance.mean(),
#         teacher_absolute.mean(), teacher_relation.mean(),
#         student_absolute.mean(), student_relation.mean(),
#     )
#     if not all(bool(torch.isfinite(value)) for value in values):
#         raise RuntimeError("historical CDS preservation objective became NaN or Inf")

#     return CDSPreservationObjective(
#         loss=loss,
#         intra_loss=intra_loss,
#         inter_loss=inter_loss,
#         physical_absolute_support_mean=physical_absolute.mean(),
#         physical_relation_support_mean=physical_relation.mean(),
#         physical_relevance_mean=physical_relevance.mean(),
#         teacher_absolute_support_mean=teacher_absolute.mean(),
#         teacher_relation_support_mean=teacher_relation.mean(),
#         student_absolute_support_mean=student_absolute.mean(),
#         student_relation_support_mean=student_relation.mean(),
#     )


# __all__ = [
#     "ClassificationObjective",
#     "SupervisedContrastiveObjective",
#     "ReferenceQConsistencyObjective",
#     "LogitDistillationObjective",
#     "CDSPreservationObjective",
#     "classification_objective",
#     "supervised_contrastive_q_loss",
#     "reference_q_consistency_loss",
#     "logit_distillation_loss",
#     "historical_cds_preservation_loss",
# ]








# from __future__ import annotations

# """Losses for the full-CDS historical-supervision NECIL-HSI architecture.

# The deployed decision rule remains one trainable bias-free cosine classifier.
# Persistent class state remains exactly {I_Q,V_Q,I_P,V_P,N}; inter-class CDS is
# re-derived from the stored state every refresh.

# Training roles:
#     direct-Q contrast -> same real-sample contrastive separation in base and incremental phases;

# Incremental roles:
#     current CE       -> current-class plasticity;
#     HDC              -> preserve the previous historical decision relation;
#     CDS separation   -> keep real current HSI away from risky historical regions;
#     decision recovery -> after representation learning, real current Q is combined
#                         with transient historical CDS boundary witnesses in one
#                         class-balanced all-seen CE while old rows stay frozen.

# Persistent memory remains exactly {I_Q,V_Q,I_P,V_P,N}. No covariance/Gaussian
# model, input inversion, second classifier, classifier-row pseudo-anchor, or
# persistent pseudo-feature buffer is introduced. Transient CDS witnesses are
# regenerated from the current boundary and discarded after each classifier update.
# """

# from dataclasses import dataclass

# import torch
# import torch.nn.functional as F

# from models.classifier import SpectralCDSClassifier
# from models.geometry_bank import ClassDiscriminativeStructureState, whole_cds_pair_risk

# Tensor = torch.Tensor


# @dataclass(frozen=True)
# class ClassificationObjective:
#     loss: Tensor
#     accuracy: Tensor
#     targets_local: Tensor


# @dataclass(frozen=True)
# class SupervisedContrastiveObjective:
#     loss: Tensor
#     valid_anchor_fraction: Tensor
#     positive_similarity_mean: Tensor
#     negative_similarity_mean: Tensor
#     similarity_gap: Tensor


# @dataclass(frozen=True)
# class HistoricalDecisionConsistencyObjective:
#     loss: Tensor
#     relation_rmse: Tensor
#     teacher_relation_norm_mean: Tensor
#     student_relation_norm_mean: Tensor
#     pair_weight_mean: Tensor
#     learned_region_risk_mean: Tensor
#     spectral_region_risk_mean: Tensor


# @dataclass(frozen=True)
# class HSICDSSeparationObjective:
#     loss: Tensor
#     margin_mean: Tensor
#     margin_min: Tensor
#     pair_weight_mean: Tensor
#     learned_region_risk_mean: Tensor
#     spectral_region_risk_mean: Tensor


# @dataclass(frozen=True)
# class BalancedAllSeenClassifierObjective:
#     """Class-balanced all-seen classifier adaptation diagnostics."""

#     loss: Tensor
#     accuracy: Tensor
#     historical_loss: Tensor
#     current_loss: Tensor
#     historical_accuracy: Tensor
#     current_accuracy: Tensor
#     per_class_loss: Tensor
#     class_ids: Tensor


# def _targets_local(labels_global: Tensor, class_ids: Tensor) -> Tensor:
#     labels = torch.as_tensor(labels_global)
#     ids = torch.as_tensor(class_ids)
#     if labels.ndim != 1 or labels.numel() == 0 or labels.dtype != torch.long:
#         raise ValueError("labels_global must be non-empty torch.long [N]")
#     if ids.ndim != 1 or ids.numel() < 2 or ids.dtype != torch.long:
#         raise ValueError("class_ids must be torch.long [C], C>=2")
#     if labels.device != ids.device:
#         raise ValueError("labels_global and class_ids must share device")
#     matches = labels.unsqueeze(1).eq(ids.unsqueeze(0))
#     if not bool(matches.sum(dim=1).eq(1).all()):
#         raise ValueError("a target label is absent from class_ids")
#     return matches.to(torch.long).argmax(dim=1)


# def _weighted_mean(per_sample: Tensor, sample_weights: Tensor | None) -> Tensor:
#     if sample_weights is None:
#         return per_sample.mean()
#     weights = torch.as_tensor(sample_weights)
#     if weights.shape != per_sample.shape or weights.device != per_sample.device or weights.dtype != per_sample.dtype:
#         raise ValueError("sample_weights must match per-sample loss")
#     if not bool(torch.isfinite(weights).all()) or bool((weights < 0).any()) or not bool(weights.sum() > 0):
#         raise ValueError("sample_weights must be finite, non-negative and have positive mass")
#     return (per_sample * weights).sum() / weights.sum()


# def _effective_hsi_pair_risk(learned: Tensor, spectral: Tensor) -> Tensor:
#     """Let physical HSI difficulty strengthen, never suppress/invent, Q-space risk."""
#     learned_value = torch.as_tensor(learned)
#     spectral_value = torch.as_tensor(
#         spectral, device=learned_value.device, dtype=learned_value.dtype
#     )
#     if learned_value.shape != spectral_value.shape:
#         raise ValueError("learned/spectral risk must have identical shape")
#     if not all(bool(torch.isfinite(v).all()) for v in (learned_value, spectral_value)):
#         raise ValueError("learned/spectral risk contains NaN or Inf")
#     learned_value = learned_value.clamp(0.0, 1.0)
#     spectral_value = spectral_value.clamp(0.0, 1.0)
#     return (
#         learned_value
#         + (1.0 - learned_value) * learned_value * spectral_value
#     ).clamp(0.0, 1.0)


# def classification_objective(
#     *,
#     logits: Tensor,
#     class_ids: Tensor,
#     labels_global: Tensor,
#     sample_weights: Tensor | None = None,
# ) -> ClassificationObjective:
#     scores = torch.as_tensor(logits)
#     ids = torch.as_tensor(class_ids)
#     if scores.ndim != 2 or scores.size(0) == 0 or scores.size(1) < 2:
#         raise ValueError("logits must be non-empty [N,C], C>=2")
#     if ids.shape != (scores.size(1),) or ids.dtype != torch.long or ids.device != scores.device:
#         raise ValueError("class_ids must match logits")
#     if not bool(torch.isfinite(scores).all()):
#         raise ValueError("logits contain NaN or Inf")
#     targets = _targets_local(labels_global, ids)
#     per_sample = F.cross_entropy(scores, targets, reduction="none")
#     loss = _weighted_mean(per_sample, sample_weights)
#     accuracy = scores.argmax(dim=1).eq(targets).to(scores.dtype).mean()
#     if not bool(torch.isfinite(loss)):
#         raise RuntimeError("classification loss became NaN or Inf")
#     return ClassificationObjective(loss=loss, accuracy=accuracy, targets_local=targets)


# def supervised_contrastive_q_loss(
#     *,
#     ordered_features: Tensor,
#     labels_global: Tensor,
#     logit_scale: Tensor,
#     sample_weights: Tensor | None = None,
# ) -> SupervisedContrastiveObjective:
#     """Supervised contrast directly on the deployed ordered Q representation.

#     The same real-sample contrast is used in Phase 0 and every incremental phase.
#     It introduces no projection head, prototype, replay state, CDS anchor, or
#     classifier parameter. Each sample is contrasted only against other real
#     samples in the current minibatch. Flattened Q is L2-normalized so this
#     objective uses the same angular geometry as the deployed cosine classifier.
#     The classifier's shared positive scale is detached: contrastive learning
#     updates Q only and cannot calibrate or bias classifier rows.

#     Anchors without a same-class peer in the minibatch are ignored.
#     """
#     q = torch.as_tensor(ordered_features)
#     labels = torch.as_tensor(labels_global)
#     if q.ndim != 3 or q.size(0) == 0 or q.size(1) < 2 or q.size(2) <= 0:
#         raise ValueError("ordered_features must be non-empty [N,K,D], K>=2")
#     if not torch.is_floating_point(q) or q.is_complex() or not bool(torch.isfinite(q).all()):
#         raise ValueError("ordered_features must be finite real floating point")
#     if labels.shape != (q.size(0),) or labels.dtype != torch.long or labels.device != q.device:
#         raise ValueError("labels_global must be torch.long [N] on the Q device")

#     scale = torch.as_tensor(logit_scale, device=q.device, dtype=q.dtype).detach()
#     if scale.ndim != 0 or not bool(torch.isfinite(scale)) or not bool(scale > 0):
#         raise ValueError("logit_scale must be a finite positive scalar")

#     eps = torch.finfo(q.dtype).eps
#     unit = F.normalize(q.reshape(q.size(0), -1), p=2, dim=1, eps=eps)
#     similarity = (unit @ unit.t()).clamp(-1.0, 1.0) * scale

#     n = int(q.size(0))
#     eye = torch.eye(n, device=q.device, dtype=torch.bool)
#     same = labels[:, None].eq(labels[None, :])
#     positive_mask = same & ~eye
#     negative_mask = (~same) & ~eye
#     candidate_mask = ~eye

#     masked_similarity = similarity.masked_fill(~candidate_mask, -torch.inf)
#     log_denom = torch.logsumexp(masked_similarity, dim=1)
#     positive_count = positive_mask.sum(dim=1)
#     valid_anchor = positive_count > 0

#     if bool(valid_anchor.any()):
#         log_probability = similarity - log_denom.unsqueeze(1)
#         per_anchor = -(
#             log_probability * positive_mask.to(dtype=q.dtype)
#         ).sum(dim=1) / positive_count.clamp_min(1).to(dtype=q.dtype)
#         valid_loss = per_anchor[valid_anchor]
#         if sample_weights is None:
#             loss = valid_loss.mean()
#         else:
#             weights = torch.as_tensor(sample_weights)
#             if weights.shape != (n,) or weights.device != q.device or weights.dtype != q.dtype:
#                 raise ValueError("sample_weights must be [N] and match Q device/dtype")
#             loss = _weighted_mean(valid_loss, weights[valid_anchor])
#     else:
#         loss = q.sum() * 0.0

#     positive_values = similarity[positive_mask]
#     negative_values = similarity[negative_mask]
#     zero = q.new_zeros(())
#     positive_mean = positive_values.mean() if positive_values.numel() else zero
#     negative_mean = negative_values.mean() if negative_values.numel() else zero
#     gap = positive_mean - negative_mean
#     valid_fraction = valid_anchor.to(dtype=q.dtype).mean()

#     values = (loss, valid_fraction, positive_mean, negative_mean, gap)
#     if not all(bool(torch.isfinite(value)) for value in values):
#         raise RuntimeError("supervised Q contrastive objective became NaN or Inf")

#     return SupervisedContrastiveObjective(
#         loss=loss,
#         valid_anchor_fraction=valid_fraction,
#         positive_similarity_mean=positive_mean,
#         negative_similarity_mean=negative_mean,
#         similarity_gap=gap,
#     )


# def historical_decision_consistency_loss(
#     *,
#     reference_features: Tensor,
#     current_features: Tensor,
#     historical_classifier_rows: Tensor,
#     labels_global: Tensor,
#     seen_cds: ClassDiscriminativeStructureState,
#     historical_class_count: int,
# ) -> HistoricalDecisionConsistencyObjective:
#     """Preserve phase-start old-class relative decision geometry on current HSI."""
#     ref = torch.as_tensor(reference_features)
#     cur = torch.as_tensor(current_features)
#     rows = torch.as_tensor(historical_classifier_rows)
#     labels = torch.as_tensor(labels_global)
#     old_count = int(historical_class_count)
#     if ref.ndim != 3 or ref.size(0) == 0 or cur.shape != ref.shape:
#         raise ValueError("reference/current features must share non-empty [N,K,D]")
#     if old_count < 2 or rows.shape != (old_count, ref.size(1) * ref.size(2)):
#         raise ValueError("historical_classifier_rows must be [O,K*D], O>=2")
#     if rows.device != ref.device or rows.dtype != ref.dtype or cur.device != ref.device or cur.dtype != ref.dtype:
#         raise ValueError("HDC tensors must share device/dtype")
#     if labels.shape != (ref.size(0),) or labels.dtype != torch.long or labels.device != ref.device:
#         raise ValueError("labels_global must be torch.long [N]")
#     if not isinstance(seen_cds, ClassDiscriminativeStructureState):
#         raise TypeError("seen_cds must be ClassDiscriminativeStructureState")

#     eps = torch.finfo(ref.dtype).eps
#     old_unit = F.normalize(rows.detach(), p=2, dim=1, eps=eps)
#     ref_unit = F.normalize(ref.reshape(ref.size(0), -1), p=2, dim=1, eps=eps)
#     cur_unit = F.normalize(cur.reshape(cur.size(0), -1), p=2, dim=1, eps=eps)
#     teacher = F.linear(ref_unit, old_unit).detach()
#     student = F.linear(cur_unit, old_unit)
#     teacher_centered = teacher - teacher.mean(dim=1, keepdim=True)
#     student_centered = student - student.mean(dim=1, keepdim=True)

#     seen_ids = seen_cds.classes.class_ids.to(device=labels.device, dtype=torch.long)
#     matches = labels.unsqueeze(1).eq(seen_ids.unsqueeze(0))
#     if not bool(matches.sum(dim=1).eq(1).all()):
#         raise RuntimeError("current label absent from all-seen CDS")
#     current_rows = matches.to(torch.long).argmax(dim=1)
#     _risk, learned, spectral = whole_cds_pair_risk(seen_cds)
#     selected_learned = learned.index_select(0, current_rows)[:, :old_count].to(dtype=ref.dtype)
#     selected_spectral = spectral.index_select(0, current_rows)[:, :old_count].to(dtype=ref.dtype)
#     weight = _effective_hsi_pair_risk(selected_learned, selected_spectral)
#     mass = weight.sum().clamp_min(eps)

#     teacher_scale = torch.sqrt((weight * teacher_centered.square()).sum() / mass).detach().clamp_min(eps)
#     teacher_relation = (teacher_centered / teacher_scale).detach()
#     student_relation = student_centered / teacher_scale
#     difference = student_relation - teacher_relation
#     loss = (weight * difference.square()).sum() / mass
#     rmse = difference.square().mean().sqrt()
#     values = (
#         loss,
#         rmse,
#         teacher_relation.norm(dim=1).mean(),
#         student_relation.norm(dim=1).mean(),
#         weight.mean(),
#         selected_learned.mean(),
#         selected_spectral.mean(),
#     )
#     if not all(bool(torch.isfinite(v)) for v in values):
#         raise RuntimeError("historical decision consistency became NaN or Inf")
#     return HistoricalDecisionConsistencyObjective(
#         loss=loss,
#         relation_rmse=rmse,
#         teacher_relation_norm_mean=teacher_relation.norm(dim=1).mean(),
#         student_relation_norm_mean=student_relation.norm(dim=1).mean(),
#         pair_weight_mean=weight.mean(),
#         learned_region_risk_mean=selected_learned.mean(),
#         spectral_region_risk_mean=selected_spectral.mean(),
#     )


# def hsi_cds_separation_loss(
#     *,
#     logits: Tensor,
#     class_ids: Tensor,
#     labels_global: Tensor,
#     seen_cds: ClassDiscriminativeStructureState,
#     historical_class_count: int,
#     sample_weights: Tensor | None = None,
# ) -> HSICDSSeparationObjective:
#     """Violation-only new->old separation using the complete all-seen HSI CDS.

#     The true current-class logit only needs to clear each historical rival.  Once
#     ``z_true - z_old >= 0`` the corresponding pair is already on the correct side
#     of the deployed cosine boundary and this objective contributes exactly zero.
#     Whole-CDS risk from {I_Q,V_Q,I_P,V_P} prioritizes *violating* pairs but never
#     turns a correctly classified pair into an always-on repulsion force.

#     The trainer calls this loss with every classifier row detached, therefore only
#     Q/backbone parameters can receive gradients from this objective.
#     """
#     scores = torch.as_tensor(logits)
#     ids = torch.as_tensor(class_ids)
#     labels = torch.as_tensor(labels_global)
#     old_count = int(historical_class_count)
#     if scores.ndim != 2 or scores.size(0) == 0:
#         raise ValueError("logits must be non-empty [N,C]")
#     if ids.shape != (scores.size(1),) or ids.dtype != torch.long or ids.device != scores.device:
#         raise ValueError("class_ids must match logits")
#     if old_count <= 0 or old_count >= int(ids.numel()):
#         raise ValueError("historical_class_count must split all-seen classes")
#     if labels.shape != (scores.size(0),) or labels.dtype != torch.long or labels.device != scores.device:
#         raise ValueError("labels_global must be torch.long [N]")

#     targets = _targets_local(labels, ids)
#     if bool((targets < old_count).any()):
#         raise ValueError("HSI-CDS separation expects current-phase labels only")

#     true_logits = scores.gather(1, targets.unsqueeze(1)).squeeze(1)
#     old_logits = scores[:, :old_count]
#     margin = true_logits.unsqueeze(1) - old_logits

#     _risk, learned, spectral = whole_cds_pair_risk(seen_cds)
#     learned_selected = learned.index_select(0, targets)[:, :old_count].to(dtype=scores.dtype)
#     spectral_selected = spectral.index_select(0, targets)[:, :old_count].to(dtype=scores.dtype)
#     selected = _effective_hsi_pair_risk(learned_selected, spectral_selected)

#     # Every actual violation matters. Whole-HSI CDS increases the priority of
#     # spectrally/structurally confusable pairs without suppressing low-risk errors.
#     priority = 1.0 + selected
#     violation = F.relu(-margin)
#     active = margin.lt(0.0)
#     active_priority = priority * active.to(dtype=scores.dtype)
#     eps = torch.finfo(scores.dtype).eps

#     numerator = (active_priority * violation).sum(dim=1)
#     active_count = active.to(dtype=scores.dtype).sum(dim=1)
#     per_sample = torch.where(
#         active_count > 0,
#         numerator / active_count.clamp_min(1.0),
#         numerator * 0.0,
#     )
#     loss = _weighted_mean(per_sample, sample_weights)

#     values = (
#         loss,
#         margin.mean(),
#         margin.min(),
#         selected.mean(),
#         learned_selected.mean(),
#         spectral_selected.mean(),
#     )
#     if not all(bool(torch.isfinite(v)) for v in values):
#         raise RuntimeError("HSI-CDS separation became NaN or Inf")

#     return HSICDSSeparationObjective(
#         loss=loss,
#         margin_mean=margin.mean(),
#         margin_min=margin.min(),
#         pair_weight_mean=selected.mean(),
#         learned_region_risk_mean=learned_selected.mean(),
#         spectral_region_risk_mean=spectral_selected.mean(),
#     )




# def cds_support_all_seen_classifier_loss(
#     *,
#     classifier: SpectralCDSClassifier,
#     historical_features: Tensor,
#     historical_labels: Tensor,
#     current_features: Tensor,
#     current_labels: Tensor,
#     seen_class_ids: Tensor,
#     trainable_class_ids: Tensor,
#     historical_sample_weights: Tensor | None = None,
# ) -> BalancedAllSeenClassifierObjective:
#     """Class-balanced all-seen CE from CDS old support and real current Q."""
#     if not isinstance(classifier, SpectralCDSClassifier):
#         raise TypeError("classifier must be SpectralCDSClassifier")

#     old_q = torch.as_tensor(historical_features)
#     new_q = torch.as_tensor(current_features)
#     old_y = torch.as_tensor(historical_labels)
#     new_y = torch.as_tensor(current_labels)
#     ids = torch.as_tensor(seen_class_ids)
#     trainable = torch.as_tensor(trainable_class_ids)

#     expected_shape = (classifier.spectral_token_count, classifier.spectral_token_dim)
#     if old_q.ndim != 3 or old_q.size(0) == 0 or tuple(old_q.shape[1:]) != expected_shape:
#         raise ValueError("historical_features must be non-empty [H,K,D] matching classifier")
#     if new_q.ndim != 3 or new_q.size(0) == 0 or tuple(new_q.shape[1:]) != expected_shape:
#         raise ValueError("current_features must be non-empty [N,K,D] matching classifier")
#     if old_q.device != new_q.device or old_q.dtype != new_q.dtype:
#         raise ValueError("historical/current features must share device/dtype")
#     if classifier.weight.device != old_q.device or classifier.weight.dtype != old_q.dtype:
#         raise ValueError("classifier and CA features must share device/dtype")
#     if old_y.shape != (old_q.size(0),) or old_y.dtype != torch.long or old_y.device != old_q.device:
#         raise ValueError("historical_labels must be torch.long [H] on feature device")
#     if new_y.shape != (new_q.size(0),) or new_y.dtype != torch.long or new_y.device != new_q.device:
#         raise ValueError("current_labels must be torch.long [N] on feature device")
#     if ids.ndim != 1 or ids.numel() < 2 or ids.dtype != torch.long or ids.device != old_q.device:
#         raise ValueError("seen_class_ids must be torch.long [C] on feature device")
#     if trainable.ndim != 1 or trainable.numel() == 0 or trainable.dtype != torch.long or trainable.device != old_q.device:
#         raise ValueError("trainable_class_ids must be non-empty torch.long on feature device")
#     if not all(bool(torch.isfinite(value).all()) for value in (old_q, new_q)):
#         raise ValueError("classifier-adaptation features contain NaN or Inf")

#     old_unique = torch.unique(old_y, sorted=True)
#     new_unique = torch.unique(new_y, sorted=True)
#     if bool(old_unique[:, None].eq(new_unique[None, :]).any()):
#         raise ValueError("historical and current labels must be disjoint")
#     if not bool(old_unique[:, None].eq(ids[None, :]).any(dim=1).all()):
#         raise ValueError("historical CDS support contains a class outside seen_class_ids")
#     if not bool(new_unique[:, None].eq(ids[None, :]).any(dim=1).all()):
#         raise ValueError("current minibatch contains a class outside seen_class_ids")
#     if not bool(old_unique[:, None].eq(trainable[None, :]).any(dim=1).all()):
#         raise ValueError("historical classifier rows must be trainable from historical CDS support")
#     if not bool(new_unique[:, None].eq(trainable[None, :]).any(dim=1).all()):
#         raise ValueError("current classifier rows must be trainable from real current Q")
#     present = torch.unique(torch.cat((old_y, new_y), dim=0), sorted=True)

#     old_weights = None
#     if historical_sample_weights is not None:
#         old_weights = torch.as_tensor(
#             historical_sample_weights, device=old_q.device, dtype=old_q.dtype
#         )
#         if old_weights.shape != (old_q.size(0),):
#             raise ValueError("historical_sample_weights must be [H]")
#         if not bool(torch.isfinite(old_weights).all()) or bool((old_weights <= 0).any()):
#             raise ValueError("historical_sample_weights must be finite and positive")

#     q_all = torch.cat((old_q, new_q), dim=0)
#     y_all = torch.cat((old_y, new_y), dim=0)
#     output = classifier(
#         q_all,
#         class_ids=ids,
#         trainable_class_ids=trainable,
#     )
#     targets = _targets_local(y_all, output.class_ids)
#     per_sample = F.cross_entropy(output.logits, targets, reduction="none")

#     per_class_losses: list[Tensor] = []
#     historical_rows = torch.arange(y_all.numel(), device=y_all.device) < old_y.numel()
#     for class_id in present:
#         mask = y_all.eq(class_id)
#         if old_weights is not None and bool((mask & historical_rows).any()):
#             old_mask = old_y.eq(class_id)
#             class_weights = old_weights[old_mask]
#             class_loss = per_sample[: old_y.numel()][old_mask]
#             per_class_losses.append(
#                 (class_weights * class_loss).sum() / class_weights.sum().clamp_min(torch.finfo(class_loss.dtype).eps)
#             )
#         else:
#             per_class_losses.append(per_sample[mask].mean())
#     per_class = torch.stack(per_class_losses, dim=0)
#     loss = per_class.mean()

#     prediction = output.prediction
#     accuracy = prediction.eq(y_all).to(dtype=output.logits.dtype).mean()
#     historical_mask = torch.arange(y_all.numel(), device=y_all.device) < old_y.numel()
#     current_mask = ~historical_mask
#     historical_accuracy = prediction[historical_mask].eq(old_y).to(output.logits.dtype).mean()
#     current_accuracy = prediction[current_mask].eq(new_y).to(output.logits.dtype).mean()

#     old_class_mask = present[:, None].eq(old_unique[None, :]).any(dim=1)
#     new_class_mask = present[:, None].eq(new_unique[None, :]).any(dim=1)
#     historical_loss = per_class[old_class_mask].mean()
#     current_loss = per_class[new_class_mask].mean()

#     values = (
#         loss, accuracy, historical_loss, current_loss,
#         historical_accuracy, current_accuracy, per_class.mean(),
#     )
#     if not all(bool(torch.isfinite(value)) for value in values):
#         raise RuntimeError("CDS-support all-seen classifier adaptation became NaN or Inf")

#     return BalancedAllSeenClassifierObjective(
#         loss=loss,
#         accuracy=accuracy,
#         historical_loss=historical_loss,
#         current_loss=current_loss,
#         historical_accuracy=historical_accuracy,
#         current_accuracy=current_accuracy,
#         per_class_loss=per_class,
#         class_ids=present,
#     )


# def balanced_all_seen_classifier_loss(
#     *,
#     classifier: SpectralCDSClassifier,
#     historical_features: Tensor,
#     historical_labels: Tensor,
#     current_features: Tensor,
#     current_labels: Tensor,
#     seen_class_ids: Tensor,
# ) -> BalancedAllSeenClassifierObjective:
#     """Backward-compatible wrapper for the previous all-row-trainable CA API."""
#     return cds_support_all_seen_classifier_loss(
#         classifier=classifier,
#         historical_features=historical_features,
#         historical_labels=historical_labels,
#         current_features=current_features,
#         current_labels=current_labels,
#         seen_class_ids=seen_class_ids,
#         trainable_class_ids=seen_class_ids,
#     )


# __all__ = [
#     "ClassificationObjective",
#     "SupervisedContrastiveObjective",
#     "HistoricalDecisionConsistencyObjective",
#     "HSICDSSeparationObjective",
#     "BalancedAllSeenClassifierObjective",
#     "classification_objective",
#     "supervised_contrastive_q_loss",
#     "historical_decision_consistency_loss",
#     "hsi_cds_separation_loss",
#     "cds_support_all_seen_classifier_loss",
#     "balanced_all_seen_classifier_loss",
# ]

