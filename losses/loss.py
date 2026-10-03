from __future__ import annotations
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from models.geometry_bank import (
    ClassDiscriminativeStructureState,
    ClassStructureBatch,
    combine_learned_physical_relevance,
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
class WholeCDSContrastiveObjective:
    loss: Tensor
    valid_anchor_fraction: Tensor
    positive_similarity_mean: Tensor
    current_negative_similarity_mean: Tensor
    historical_negative_similarity_mean: Tensor
    whole_support_margin_mean: Tensor
    whole_support_margin_min: Tensor
    physical_confusability_mean: Tensor
    effective_pair_risk_mean: Tensor
    violation_fraction: Tensor

    @property
    def negative_similarity_mean(self) -> Tensor:
        return self.current_negative_similarity_mean

    @property
    def similarity_gap(self) -> Tensor:
        return self.positive_similarity_mean - self.current_negative_similarity_mean

    @property
    def learned_support_margin_mean(self) -> Tensor:
        return self.whole_support_margin_mean

    @property
    def learned_support_margin_min(self) -> Tensor:
        return self.whole_support_margin_min


# Compatibility alias only; active trainers use WholeCDSContrastiveObjective.
WholeCDSPICObjective = WholeCDSContrastiveObjective


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
    if (
        weights.shape != values.shape
        or weights.device != values.device
        or weights.dtype != values.dtype
    ):
        raise ValueError("sample_weights must match per-sample loss")
    if (
        not bool(torch.isfinite(weights).all())
        or bool((weights < 0).any())
        or not bool(weights.sum() > 0)
    ):
        raise ValueError(
            "sample_weights must be finite, non-negative and have positive mass"
        )
    return (values * weights).sum() / weights.sum()


def _weighted_matrix_mean(values: Tensor, weights: Tensor) -> Tensor:
    value = torch.as_tensor(values)
    weight = torch.as_tensor(weights)
    if (
        value.shape != weight.shape
        or value.device != weight.device
        or value.dtype != weight.dtype
    ):
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
    if (
        ids.shape != (scores.size(1),)
        or ids.dtype != torch.long
        or ids.device != scores.device
    ):
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
    if (
        not torch.is_floating_point(q)
        or q.is_complex()
        or not bool(torch.isfinite(q).all())
    ):
        raise ValueError("ordered_features must be finite real floating point")
    if (
        labels.shape != (q.size(0),)
        or labels.dtype != torch.long
        or labels.device != q.device
    ):
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
        per_anchor = -(log_probability * positive_mask.to(dtype=q.dtype)).sum(
            dim=1
        ) / positive_count.clamp_min(1).to(dtype=q.dtype)
        valid_loss = per_anchor[valid_anchor]
        if sample_weights is None:
            loss = valid_loss.mean()
        else:
            weights = torch.as_tensor(sample_weights)
            if (
                weights.shape != (n,)
                or weights.device != q.device
                or weights.dtype != q.dtype
            ):
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
    if (
        teacher.shape != student.shape
        or teacher.device != student.device
        or teacher.dtype != student.dtype
    ):
        raise ValueError("teacher_old_logits must match student_old_logits")
    if not bool(torch.isfinite(student).all()) or not bool(
        torch.isfinite(teacher).all()
    ):
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
    if not all(
        bool(torch.isfinite(value))
        for value in (loss, teacher_entropy, student_entropy)
    ):
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


def _whole_cds_support_components(
    *,
    learned_features: Tensor,
    physical_features: Tensor,
    structure: ClassStructureBatch,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    q = torch.as_tensor(learned_features)
    p = torch.as_tensor(physical_features, device=q.device, dtype=q.dtype)
    if q.ndim != 3 or p.ndim != 3 or q.size(0) != p.size(0):
        raise ValueError("learned/physical features must be aligned rank-three batches")

    i_q = structure.learned_identity.to(device=q.device, dtype=q.dtype).detach()
    v_q = structure.learned_variation.to(device=q.device, dtype=q.dtype).detach()
    i_p = structure.physical_identity.to(device=q.device, dtype=q.dtype).detach()
    v_p = structure.physical_variation.to(device=q.device, dtype=q.dtype).detach()

    learned_abs, learned_rel, learned_full = ordered_cds_support_components(q, i_q, v_q)
    physical_abs, physical_rel, physical_full = ordered_cds_support_components(
        p, i_p, v_p
    )

    # Q remains the semantic carrier. Physical spectral evidence can strengthen a
    # learned relation but cannot create class support when learned support is zero.
    # This makes I_P/V_P causal in training without creating a second classifier.
    whole_abs = combine_learned_physical_relevance(learned_abs, physical_abs)
    whole_rel = combine_learned_physical_relevance(learned_rel, physical_rel)
    whole_full = combine_learned_physical_relevance(learned_full, physical_full)
    values = (
        whole_abs,
        whole_rel,
        whole_full,
        physical_abs,
        physical_rel,
        physical_full,
    )
    if not all(bool(torch.isfinite(value).all()) for value in values):
        raise RuntimeError("Whole-CDS support contains NaN or Inf")
    return values


def _whole_cds_cross_entropy(
    *,
    whole_absolute: Tensor,
    whole_relation: Tensor,
    whole_complete: Tensor,
    targets_local: Tensor,
    sample_weights: Tensor | None,
) -> Tensor:
    """Class-balanced CE on absolute, relation and complete Whole-CDS support."""
    eps = torch.finfo(whole_complete.dtype).eps
    losses = []
    for support in (whole_absolute, whole_relation, whole_complete):
        per_sample = F.cross_entropy(
            support.clamp_min(eps).log(), targets_local, reduction="none"
        )
        losses.append(_weighted_mean(per_sample, sample_weights))
    return sum(losses) / float(len(losses))


def historical_cds_preservation_loss(
    *,
    reference_features: Tensor,
    current_features: Tensor,
    physical_spectral_evidence: Tensor,
    reference_historical_structure: ClassStructureBatch,
    current_historical_structure: ClassStructureBatch,
) -> CDSPreservationObjective:
    """Preserve historical Whole-CDS intra/inter structure on real current HSI.

    Both teacher and student are evaluated against the same finalized historical
    coordinate during incremental SGD.  The optimized supports are *direct*
    Whole-CDS scores: learned ordered structure (I_Q,V_Q) is geometrically fused
    with native spectral structure (I_P,V_P) before intra- and inter-class errors
    are formed.  Physical CDS therefore participates in the actual preserved
    class score rather than merely weighting a learned-space loss.
    """
    ref_struct = reference_historical_structure
    cur_struct = current_historical_structure
    if not isinstance(ref_struct, ClassStructureBatch) or ref_struct.class_count <= 0:
        raise ValueError("reference_historical_structure must be non-empty")
    if (
        not isinstance(cur_struct, ClassStructureBatch)
        or cur_struct.class_count != ref_struct.class_count
    ):
        raise ValueError(
            "current_historical_structure must match historical class count"
        )
    if not torch.equal(ref_struct.class_ids, cur_struct.class_ids):
        raise ValueError("reference/current historical class IDs disagree")
    if not torch.equal(ref_struct.physical_identity, cur_struct.physical_identity):
        raise ValueError("historical I_P must remain phase-stable")
    if not torch.equal(ref_struct.physical_variation, cur_struct.physical_variation):
        raise ValueError("historical V_P must remain phase-stable")

    reference = normalize_ordered_decision_features(
        torch.as_tensor(reference_features)
    ).detach()
    current = normalize_ordered_decision_features(torch.as_tensor(current_features))
    physical = torch.as_tensor(
        physical_spectral_evidence, device=current.device, dtype=current.dtype
    ).detach()
    if (
        current.shape != reference.shape
        or current.device != reference.device
        or current.dtype != reference.dtype
    ):
        raise ValueError("current_features must match reference_features")
    if (
        physical.ndim != 3
        or physical.size(0) != reference.size(0)
        or physical.size(2) != 1
    ):
        raise ValueError("physical_spectral_evidence must be [N,B,1]")

    with torch.no_grad():
        (
            teacher_abs,
            teacher_rel,
            teacher_full,
            physical_abs,
            physical_rel,
            physical_full,
        ) = _whole_cds_support_components(
            learned_features=reference,
            physical_features=physical,
            structure=ref_struct,
        )
    student_abs, student_rel, student_full, _, _, _ = _whole_cds_support_components(
        learned_features=current,
        physical_features=physical,
        structure=cur_struct,
    )

    # Intra-CDS: preserve the class-wise absolute and ordered-relation support.
    intra_error = 0.5 * (
        (student_abs - teacher_abs).abs() + (student_rel - teacher_rel).abs()
    )
    intra_loss = intra_error.mean()

    # Inter-CDS: preserve all historical pairwise support differences.
    class_count = ref_struct.class_count
    if class_count >= 2:
        upper = torch.triu(
            torch.ones(
                class_count, class_count, device=current.device, dtype=torch.bool
            ),
            diagonal=1,
        )
        teacher_pair = teacher_full.unsqueeze(2) - teacher_full.unsqueeze(1)
        student_pair = student_full.unsqueeze(2) - student_full.unsqueeze(1)
        inter_loss = (student_pair - teacher_pair).abs()[:, upper].mean()
    else:
        inter_loss = current.sum() * 0.0

    loss = intra_loss + inter_loss
    values = (
        loss,
        intra_loss,
        inter_loss,
        physical_abs.mean(),
        physical_rel.mean(),
        physical_full.mean(),
        teacher_abs.mean(),
        teacher_rel.mean(),
        student_abs.mean(),
        student_rel.mean(),
    )
    if not all(bool(torch.isfinite(value)) for value in values):
        raise RuntimeError(
            "historical Whole-CDS preservation objective became NaN or Inf"
        )

    return CDSPreservationObjective(
        loss=loss,
        intra_loss=intra_loss,
        inter_loss=inter_loss,
        physical_absolute_support_mean=physical_abs.mean(),
        physical_relation_support_mean=physical_rel.mean(),
        physical_relevance_mean=physical_full.mean(),
        teacher_absolute_support_mean=teacher_abs.mean(),
        teacher_relation_support_mean=teacher_rel.mean(),
        student_absolute_support_mean=student_abs.mean(),
        student_relation_support_mean=student_rel.mean(),
    )


def whole_cds_contrastive_loss(
    *,
    current_features: Tensor,
    labels_global: Tensor,
    physical_spectral_evidence: Tensor,
    seen_cds: ClassDiscriminativeStructureState,
    historical_class_count: int,
    logit_scale: Tensor,
    sample_weights: Tensor | None = None,
) -> WholeCDSContrastiveObjective:
    """Unified continual contrastive separation in the single ordered Q space.

    Base phase: the historical set is empty, so this is ordinary supervised
    contrastive learning over real current samples.

    Incremental phase: the *same* contrastive denominator is extended with
    historical Whole-CDS negatives. Their logits use complete sample-to-CDS
    support from I_Q/V_Q and I_P/V_P, while learned+physical inter-class risk
    increases the mass of dangerous current/old rivals. There is no separate PIC
    loss and no prototype-only historical anchor.
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
    if count < 2 or old_count < 0 or old_count >= count:
        raise ValueError("historical_class_count must be in [0, C-1]")
    structure = ClassStructureBatch(
        class_ids=classes.class_ids.to(device=q.device),
        sample_counts=classes.sample_counts.to(device=q.device),
        learned_identity=classes.learned_identity.to(device=q.device, dtype=q.dtype),
        learned_variation=classes.learned_variation.to(device=q.device, dtype=q.dtype),
        physical_identity=classes.physical_identity.to(device=q.device, dtype=q.dtype),
        physical_variation=classes.physical_variation.to(
            device=q.device, dtype=q.dtype
        ),
    )
    if structure.learned_identity.shape[1:] != q.shape[1:]:
        raise ValueError("current Q and learned CDS use different coordinates")
    if structure.physical_identity.shape[1:] != physical.shape[1:]:
        raise ValueError("physical evidence and physical CDS use different coordinates")

    matches = labels.unsqueeze(1).eq(ids.unsqueeze(0))
    if not bool(matches.sum(dim=1).eq(1).all()):
        raise ValueError("a current label is absent from seen CDS")
    target_index = matches.to(torch.long).argmax(dim=1)
    if old_count > 0 and bool((target_index < old_count).any()):
        raise ValueError("incremental batches must contain only incoming classes")

    scale = torch.as_tensor(logit_scale, device=q.device, dtype=q.dtype).detach()
    if scale.ndim != 0 or not bool(torch.isfinite(scale)) or not bool(scale > 0):
        raise ValueError("logit_scale must be a finite positive scalar")
    eps = torch.finfo(q.dtype).eps

    # Current/current supervised contrastive logits.
    flat = F.normalize(q.reshape(q.size(0), -1), p=2, dim=1, eps=eps)
    current_similarity = (flat @ flat.t()).clamp(-1.0, 1.0) * scale
    n = int(q.size(0))
    eye = torch.eye(n, device=q.device, dtype=torch.bool)
    same = labels[:, None].eq(labels[None, :])
    positive_mask = same & ~eye
    negative_mask = (~same) & ~eye
    current_logits = current_similarity.masked_fill(eye, -torch.inf)

    # Whole-CDS support is required for historical negatives and diagnostics. The
    # physical branch is fused with learned support but remains training evidence;
    # deployed inference is still cosine(Q,W).
    whole_abs, whole_rel, whole_full, _physical_abs, _physical_rel, physical_full = (
        _whole_cds_support_components(
            learned_features=q,
            physical_features=physical,
            structure=structure,
        )
    )
    del whole_abs, whole_rel
    rows = torch.arange(n, device=q.device)
    positive_whole = whole_full[rows, target_index]

    zero = q.new_zeros(())
    if old_count > 0:
        old_whole = whole_full[:, :old_count]
        # Map [0,1] CDS support to the same bounded similarity range as cosine.
        historical_similarity = (2.0 * old_whole - 1.0).clamp(-1.0, 1.0)
        with torch.no_grad():
            effective_risk, _, _ = whole_cds_pair_risk(seen_cds)
            old_pair_risk = effective_risk.to(
                device=q.device, dtype=q.dtype
            ).index_select(0, target_index)[:, :old_count]
            positive_physical = physical_full[rows, target_index]
            old_physical = physical_full[:, :old_count]
            old_physical_confusability = torch.sqrt(
                positive_physical.unsqueeze(1).clamp_min(0.0)
                * old_physical.clamp_min(0.0)
            ).clamp(0.0, 1.0)
            historical_weight = 1.0 + old_pair_risk * old_physical_confusability
        historical_logits = (
            scale * historical_similarity + historical_weight.clamp_min(eps).log()
        )
        denominator_logits = torch.cat((current_logits, historical_logits), dim=1)
        support_margin_matrix = positive_whole.unsqueeze(1) - old_whole
        support_margin = support_margin_matrix.min(dim=1).values
        physical_confusability = old_physical_confusability.mean()
        pair_risk = old_pair_risk.mean()
        historical_negative_mean = historical_similarity.mean()
    else:
        denominator_logits = current_logits
        rival_mask = ~torch.eye(count, device=q.device, dtype=torch.bool).index_select(
            0, target_index
        )
        rival_whole = whole_full.masked_fill(~rival_mask, -torch.inf)
        support_margin = positive_whole - rival_whole.max(dim=1).values
        physical_confusability = physical_full.mean()
        pair_risk = zero
        historical_negative_mean = zero

    log_denom = torch.logsumexp(denominator_logits, dim=1)
    positive_count = positive_mask.sum(dim=1)
    valid_anchor = positive_count > 0
    if bool(valid_anchor.any()):
        log_probability = current_similarity - log_denom.unsqueeze(1)
        per_anchor = -(log_probability * positive_mask.to(dtype=q.dtype)).sum(
            dim=1
        ) / positive_count.clamp_min(1).to(dtype=q.dtype)
        if sample_weights is None:
            loss = per_anchor[valid_anchor].mean()
        else:
            weights = torch.as_tensor(sample_weights)
            if (
                weights.shape != (n,)
                or weights.device != q.device
                or weights.dtype != q.dtype
            ):
                raise ValueError("sample_weights must be [N] and match Q device/dtype")
            loss = _weighted_mean(per_anchor[valid_anchor], weights[valid_anchor])
    else:
        loss = q.sum() * 0.0

    positive_values = current_similarity[positive_mask]
    negative_values = current_similarity[negative_mask]
    positive_mean = positive_values.mean() if positive_values.numel() else zero
    current_negative_mean = negative_values.mean() if negative_values.numel() else zero
    valid_fraction = valid_anchor.to(dtype=q.dtype).mean()
    values = (
        loss,
        valid_fraction,
        positive_mean,
        current_negative_mean,
        historical_negative_mean,
        support_margin.mean(),
        support_margin.min(),
        physical_confusability,
        pair_risk,
    )
    if not all(bool(torch.isfinite(v)) for v in values):
        raise RuntimeError("unified Whole-CDS contrastive objective became NaN or Inf")

    return WholeCDSContrastiveObjective(
        loss=loss,
        valid_anchor_fraction=valid_fraction,
        positive_similarity_mean=positive_mean,
        current_negative_similarity_mean=current_negative_mean,
        historical_negative_similarity_mean=historical_negative_mean,
        whole_support_margin_mean=support_margin.mean(),
        whole_support_margin_min=support_margin.min(),
        physical_confusability_mean=physical_confusability,
        effective_pair_risk_mean=pair_risk,
        violation_fraction=support_margin.le(0.0).to(q.dtype).mean(),
    )


# Backward-compatible name; active trainers use the continual name below.
def whole_cds_pic_loss(**kwargs) -> WholeCDSContrastiveObjective:
    return whole_cds_contrastive_loss(**kwargs)


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
        positive_physical.unsqueeze(1).clamp_min(0.0) * rival_physical.clamp_min(0.0)
    ).clamp(0.0, 1.0)
    pair_weight = 1.0 + pair_risk * spectral_confusability

    scale = torch.as_tensor(logit_scale, device=q.device, dtype=q.dtype).detach()
    if scale.ndim != 0 or not bool(torch.isfinite(scale)) or not bool(scale > 0):
        raise ValueError("logit_scale must be a finite positive scalar")
    pair_loss = F.softplus(-scale * margin)
    per_sample = (pair_weight * pair_loss).sum(dim=1) / pair_weight.sum(
        dim=1
    ).clamp_min(torch.finfo(q.dtype).eps)
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
    "WholeCDSContrastiveObjective",
    "WholeCDSPICObjective",
    "classification_objective",
    "supervised_contrastive_q_loss",
    "logit_distillation_loss",
    "ordered_feature_distillation_loss",
    "historical_cds_preservation_loss",
    "whole_cds_contrastive_loss",
    "whole_cds_pic_loss",
    "cross_phase_cds_separation_loss",
]


# from __future__ import annotations
# from dataclasses import dataclass

# import torch
# import torch.nn.functional as F

# from models.geometry_bank import (
#     ClassDiscriminativeStructureState,
#     ClassStructureBatch,
#     combine_learned_physical_relevance,
#     normalize_ordered_decision_features,
#     ordered_cds_support_components,
#     whole_cds_pair_risk,
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
# class LogitDistillationObjective:
#     loss: Tensor
#     teacher_entropy_mean: Tensor
#     student_entropy_mean: Tensor


# @dataclass(frozen=True)
# class FeatureDistillationObjective:
#     loss: Tensor
#     cosine_similarity_mean: Tensor
#     rms_drift: Tensor


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


# @dataclass(frozen=True)
# class CDSSeparationObjective:
#     loss: Tensor
#     learned_positive_support_mean: Tensor
#     learned_historical_rival_support_mean: Tensor
#     learned_support_margin_mean: Tensor
#     learned_support_margin_min: Tensor
#     physical_confusability_mean: Tensor
#     effective_pair_risk_mean: Tensor
#     violation_fraction: Tensor

# @dataclass(frozen=True)
# class WholeCDSContrastiveObjective:
#     loss: Tensor
#     valid_anchor_fraction: Tensor
#     positive_similarity_mean: Tensor
#     current_negative_similarity_mean: Tensor
#     historical_negative_similarity_mean: Tensor
#     whole_support_margin_mean: Tensor
#     whole_support_margin_min: Tensor
#     physical_confusability_mean: Tensor
#     effective_pair_risk_mean: Tensor
#     violation_fraction: Tensor

#     @property
#     def negative_similarity_mean(self) -> Tensor:
#         return self.current_negative_similarity_mean

#     @property
#     def similarity_gap(self) -> Tensor:
#         return self.positive_similarity_mean - self.current_negative_similarity_mean

#     @property
#     def learned_support_margin_mean(self) -> Tensor:
#         return self.whole_support_margin_mean

#     @property
#     def learned_support_margin_min(self) -> Tensor:
#         return self.whole_support_margin_min


# # Compatibility alias only; active trainers use WholeCDSContrastiveObjective.
# WholeCDSPICObjective = WholeCDSContrastiveObjective


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


# def ordered_feature_distillation_loss(
#     *,
#     reference_features: Tensor,
#     current_features: Tensor,
#     sample_weights: Tensor | None = None,
# ) -> FeatureDistillationObjective:
#     """FCS-style feature KD in the same ordered Q coordinate used by CDS.

#     The frozen previous representation is a training reference only.  Both Q^-
#     and Q^+ are mapped to the canonical cosine/CDS coordinate before the loss so
#     this term constrains semantic direction rather than arbitrary feature norm.
#     Token order is preserved; no second representation is created.
#     """
#     reference = normalize_ordered_decision_features(
#         torch.as_tensor(reference_features)
#     ).detach()
#     current = normalize_ordered_decision_features(torch.as_tensor(current_features))
#     if current.shape != reference.shape:
#         raise ValueError("current_features must match reference_features")
#     if current.device != reference.device or current.dtype != reference.dtype:
#         raise ValueError("reference/current Q must share device/dtype")
#     delta = current - reference

#     # IMPORTANT: do not use sqrt(mean(delta**2)) as the optimization loss.
#     # At the beginning of an incremental phase the current backbone is an exact
#     # copy of the frozen teacher, so delta == 0.  The forward value sqrt(0) is
#     # finite, but its derivative is undefined and PyTorch returns NaN gradients.
#     #
#     # Because both tensors are already globally L2-normalized per sample,
#     #   0.5 * ||q_t - q_{t-1}||_2^2 == 1 - cos(q_t, q_{t-1}).
#     # This squared chord/cosine distance has the same semantic objective, is
#     # exactly zero at phase start, and has an exactly zero (finite) gradient
#     # when student and teacher coincide.  Token order is still preserved because
#     # the distance is computed over the ordered KxD tensor without pooling into
#     # a second representation.
#     per_sample = 0.5 * delta.square().sum(dim=(1, 2))
#     loss = _weighted_mean(per_sample, sample_weights)
#     flat_ref = reference.reshape(reference.size(0), -1)
#     flat_cur = current.reshape(current.size(0), -1)
#     cosine = F.cosine_similarity(flat_cur, flat_ref, dim=1).mean()

#     # RMS is diagnostic only.  Detaching avoids retaining a second sqrt branch
#     # in the optimization graph and makes the phase-start zero-drift case safe.
#     with torch.no_grad():
#         rms = delta.detach().square().mean().sqrt()
#     if not all(bool(torch.isfinite(v)) for v in (loss, cosine, rms)):
#         raise RuntimeError("ordered feature distillation became NaN or Inf")
#     return FeatureDistillationObjective(
#         loss=loss,
#         cosine_similarity_mean=cosine,
#         rms_drift=rms,
#     )

# def _whole_cds_support_components(
#     *,
#     learned_features: Tensor,
#     physical_features: Tensor,
#     structure: ClassStructureBatch,
# ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
#     q = torch.as_tensor(learned_features)
#     p = torch.as_tensor(physical_features, device=q.device, dtype=q.dtype)
#     if q.ndim != 3 or p.ndim != 3 or q.size(0) != p.size(0):
#         raise ValueError("learned/physical features must be aligned rank-three batches")

#     i_q = structure.learned_identity.to(device=q.device, dtype=q.dtype).detach()
#     v_q = structure.learned_variation.to(device=q.device, dtype=q.dtype).detach()
#     i_p = structure.physical_identity.to(device=q.device, dtype=q.dtype).detach()
#     v_p = structure.physical_variation.to(device=q.device, dtype=q.dtype).detach()

#     learned_abs, learned_rel, learned_full = ordered_cds_support_components(q, i_q, v_q)
#     physical_abs, physical_rel, physical_full = ordered_cds_support_components(p, i_p, v_p)

#     # Q remains the semantic carrier. Physical spectral evidence can strengthen a
#     # learned relation but cannot create class support when learned support is zero.
#     # This makes I_P/V_P causal in training without creating a second classifier.
#     whole_abs = combine_learned_physical_relevance(learned_abs, physical_abs)
#     whole_rel = combine_learned_physical_relevance(learned_rel, physical_rel)
#     whole_full = combine_learned_physical_relevance(learned_full, physical_full)
#     values = (whole_abs, whole_rel, whole_full, physical_abs, physical_rel, physical_full)
#     if not all(bool(torch.isfinite(value).all()) for value in values):
#         raise RuntimeError("Whole-CDS support contains NaN or Inf")
#     return values


# def _whole_cds_cross_entropy(
#     *,
#     whole_absolute: Tensor,
#     whole_relation: Tensor,
#     whole_complete: Tensor,
#     targets_local: Tensor,
#     sample_weights: Tensor | None,
# ) -> Tensor:
#     """Class-balanced CE on absolute, relation and complete Whole-CDS support."""
#     eps = torch.finfo(whole_complete.dtype).eps
#     losses = []
#     for support in (whole_absolute, whole_relation, whole_complete):
#         per_sample = F.cross_entropy(support.clamp_min(eps).log(), targets_local, reduction="none")
#         losses.append(_weighted_mean(per_sample, sample_weights))
#     return sum(losses) / float(len(losses))


# def historical_cds_preservation_loss(
#     *,
#     reference_features: Tensor,
#     current_features: Tensor,
#     physical_spectral_evidence: Tensor,
#     reference_historical_structure: ClassStructureBatch,
#     current_historical_structure: ClassStructureBatch,
# ) -> CDSPreservationObjective:
#     """Preserve historical Whole-CDS intra/inter structure on real current HSI.

#     Both teacher and student are evaluated against the same finalized historical
#     coordinate during incremental SGD.  The optimized supports are *direct*
#     Whole-CDS scores: learned ordered structure (I_Q,V_Q) is geometrically fused
#     with native spectral structure (I_P,V_P) before intra- and inter-class errors
#     are formed.  Physical CDS therefore participates in the actual preserved
#     class score rather than merely weighting a learned-space loss.
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
#     physical = torch.as_tensor(physical_spectral_evidence, device=current.device, dtype=current.dtype).detach()
#     if current.shape != reference.shape or current.device != reference.device or current.dtype != reference.dtype:
#         raise ValueError("current_features must match reference_features")
#     if physical.ndim != 3 or physical.size(0) != reference.size(0) or physical.size(2) != 1:
#         raise ValueError("physical_spectral_evidence must be [N,B,1]")

#     with torch.no_grad():
#         teacher_abs, teacher_rel, teacher_full, physical_abs, physical_rel, physical_full = (
#             _whole_cds_support_components(
#                 learned_features=reference,
#                 physical_features=physical,
#                 structure=ref_struct,
#             )
#         )
#     student_abs, student_rel, student_full, _, _, _ = _whole_cds_support_components(
#         learned_features=current,
#         physical_features=physical,
#         structure=cur_struct,
#     )

#     # Intra-CDS: preserve the class-wise absolute and ordered-relation support.
#     intra_error = 0.5 * ((student_abs - teacher_abs).abs() + (student_rel - teacher_rel).abs())
#     intra_loss = intra_error.mean()

#     # Inter-CDS: preserve all historical pairwise support differences.
#     class_count = ref_struct.class_count
#     if class_count >= 2:
#         upper = torch.triu(
#             torch.ones(class_count, class_count, device=current.device, dtype=torch.bool), diagonal=1
#         )
#         teacher_pair = teacher_full.unsqueeze(2) - teacher_full.unsqueeze(1)
#         student_pair = student_full.unsqueeze(2) - student_full.unsqueeze(1)
#         inter_loss = (student_pair - teacher_pair).abs()[:, upper].mean()
#     else:
#         inter_loss = current.sum() * 0.0

#     loss = intra_loss + inter_loss
#     values = (
#         loss, intra_loss, inter_loss,
#         physical_abs.mean(), physical_rel.mean(), physical_full.mean(),
#         teacher_abs.mean(), teacher_rel.mean(), student_abs.mean(), student_rel.mean(),
#     )
#     if not all(bool(torch.isfinite(value)) for value in values):
#         raise RuntimeError("historical Whole-CDS preservation objective became NaN or Inf")

#     return CDSPreservationObjective(
#         loss=loss,
#         intra_loss=intra_loss,
#         inter_loss=inter_loss,
#         physical_absolute_support_mean=physical_abs.mean(),
#         physical_relation_support_mean=physical_rel.mean(),
#         physical_relevance_mean=physical_full.mean(),
#         teacher_absolute_support_mean=teacher_abs.mean(),
#         teacher_relation_support_mean=teacher_rel.mean(),
#         student_absolute_support_mean=student_abs.mean(),
#         student_relation_support_mean=student_rel.mean(),
#     )


# def whole_cds_contrastive_loss(
#     *,
#     current_features: Tensor,
#     labels_global: Tensor,
#     physical_spectral_evidence: Tensor,
#     seen_cds: ClassDiscriminativeStructureState,
#     historical_class_count: int,
#     logit_scale: Tensor,
#     sample_weights: Tensor | None = None,
# ) -> WholeCDSContrastiveObjective:
#     """Unified continual contrastive separation in the single ordered Q space.

#     Base phase: the historical set is empty, so this is ordinary supervised
#     contrastive learning over real current samples.

#     Incremental phase: the *same* contrastive denominator is extended with
#     historical Whole-CDS negatives. Their logits use complete sample-to-CDS
#     support from I_Q/V_Q and I_P/V_P, while learned+physical inter-class risk
#     increases the mass of dangerous current/old rivals. There is no separate PIC
#     loss and no prototype-only historical anchor.
#     """
#     if not isinstance(seen_cds, ClassDiscriminativeStructureState):
#         raise TypeError("seen_cds must be ClassDiscriminativeStructureState")
#     q = normalize_ordered_decision_features(torch.as_tensor(current_features))
#     labels = torch.as_tensor(labels_global, device=q.device, dtype=torch.long).flatten()
#     physical = torch.as_tensor(
#         physical_spectral_evidence, device=q.device, dtype=q.dtype
#     ).detach()
#     if labels.shape != (q.size(0),):
#         raise ValueError("labels_global must be [N]")
#     if physical.ndim != 3 or physical.size(0) != q.size(0) or physical.size(2) != 1:
#         raise ValueError("physical_spectral_evidence must be [N,B,1]")
#     if not bool(torch.isfinite(physical).all()):
#         raise ValueError("physical_spectral_evidence contains NaN or Inf")

#     classes = seen_cds.classes
#     ids = classes.class_ids.to(device=q.device, dtype=torch.long).flatten()
#     count = int(ids.numel())
#     old_count = int(historical_class_count)
#     if count < 2 or old_count < 0 or old_count >= count:
#         raise ValueError("historical_class_count must be in [0, C-1]")
#     structure = ClassStructureBatch(
#         class_ids=classes.class_ids.to(device=q.device),
#         sample_counts=classes.sample_counts.to(device=q.device),
#         learned_identity=classes.learned_identity.to(device=q.device, dtype=q.dtype),
#         learned_variation=classes.learned_variation.to(device=q.device, dtype=q.dtype),
#         physical_identity=classes.physical_identity.to(device=q.device, dtype=q.dtype),
#         physical_variation=classes.physical_variation.to(device=q.device, dtype=q.dtype),
#     )
#     if structure.learned_identity.shape[1:] != q.shape[1:]:
#         raise ValueError("current Q and learned CDS use different coordinates")
#     if structure.physical_identity.shape[1:] != physical.shape[1:]:
#         raise ValueError("physical evidence and physical CDS use different coordinates")

#     matches = labels.unsqueeze(1).eq(ids.unsqueeze(0))
#     if not bool(matches.sum(dim=1).eq(1).all()):
#         raise ValueError("a current label is absent from seen CDS")
#     target_index = matches.to(torch.long).argmax(dim=1)
#     if old_count > 0 and bool((target_index < old_count).any()):
#         raise ValueError("incremental batches must contain only incoming classes")

#     scale = torch.as_tensor(logit_scale, device=q.device, dtype=q.dtype).detach()
#     if scale.ndim != 0 or not bool(torch.isfinite(scale)) or not bool(scale > 0):
#         raise ValueError("logit_scale must be a finite positive scalar")
#     eps = torch.finfo(q.dtype).eps

#     # Current/current supervised contrastive logits.
#     flat = F.normalize(q.reshape(q.size(0), -1), p=2, dim=1, eps=eps)
#     current_similarity = (flat @ flat.t()).clamp(-1.0, 1.0) * scale
#     n = int(q.size(0))
#     eye = torch.eye(n, device=q.device, dtype=torch.bool)
#     same = labels[:, None].eq(labels[None, :])
#     positive_mask = same & ~eye
#     negative_mask = (~same) & ~eye
#     current_logits = current_similarity.masked_fill(eye, -torch.inf)

#     # Whole-CDS support is required for historical negatives and diagnostics. The
#     # physical branch is fused with learned support but remains training evidence;
#     # deployed inference is still cosine(Q,W).
#     whole_abs, whole_rel, whole_full, _physical_abs, _physical_rel, physical_full = (
#         _whole_cds_support_components(
#             learned_features=q,
#             physical_features=physical,
#             structure=structure,
#         )
#     )
#     del whole_abs, whole_rel
#     rows = torch.arange(n, device=q.device)
#     positive_whole = whole_full[rows, target_index]

#     zero = q.new_zeros(())
#     if old_count > 0:
#         old_whole = whole_full[:, :old_count]
#         # Map [0,1] CDS support to the same bounded similarity range as cosine.
#         historical_similarity = (2.0 * old_whole - 1.0).clamp(-1.0, 1.0)
#         with torch.no_grad():
#             effective_risk, _, _ = whole_cds_pair_risk(seen_cds)
#             old_pair_risk = effective_risk.to(
#                 device=q.device, dtype=q.dtype
#             ).index_select(0, target_index)[:, :old_count]
#             positive_physical = physical_full[rows, target_index]
#             old_physical = physical_full[:, :old_count]
#             old_physical_confusability = torch.sqrt(
#                 positive_physical.unsqueeze(1).clamp_min(0.0)
#                 * old_physical.clamp_min(0.0)
#             ).clamp(0.0, 1.0)
#             historical_weight = 1.0 + old_pair_risk * old_physical_confusability
#         historical_logits = (
#             scale * historical_similarity
#             + historical_weight.clamp_min(eps).log()
#         )
#         denominator_logits = torch.cat((current_logits, historical_logits), dim=1)
#         support_margin_matrix = positive_whole.unsqueeze(1) - old_whole
#         support_margin = support_margin_matrix.min(dim=1).values
#         physical_confusability = old_physical_confusability.mean()
#         pair_risk = old_pair_risk.mean()
#         historical_negative_mean = historical_similarity.mean()
#     else:
#         denominator_logits = current_logits
#         rival_mask = ~torch.eye(count, device=q.device, dtype=torch.bool).index_select(
#             0, target_index
#         )
#         rival_whole = whole_full.masked_fill(~rival_mask, -torch.inf)
#         support_margin = positive_whole - rival_whole.max(dim=1).values
#         physical_confusability = physical_full.mean()
#         pair_risk = zero
#         historical_negative_mean = zero

#     log_denom = torch.logsumexp(denominator_logits, dim=1)
#     positive_count = positive_mask.sum(dim=1)
#     valid_anchor = positive_count > 0
#     if bool(valid_anchor.any()):
#         log_probability = current_similarity - log_denom.unsqueeze(1)
#         per_anchor = -(
#             log_probability * positive_mask.to(dtype=q.dtype)
#         ).sum(dim=1) / positive_count.clamp_min(1).to(dtype=q.dtype)
#         if sample_weights is None:
#             loss = per_anchor[valid_anchor].mean()
#         else:
#             weights = torch.as_tensor(sample_weights)
#             if weights.shape != (n,) or weights.device != q.device or weights.dtype != q.dtype:
#                 raise ValueError("sample_weights must be [N] and match Q device/dtype")
#             loss = _weighted_mean(per_anchor[valid_anchor], weights[valid_anchor])
#     else:
#         loss = q.sum() * 0.0

#     positive_values = current_similarity[positive_mask]
#     negative_values = current_similarity[negative_mask]
#     positive_mean = positive_values.mean() if positive_values.numel() else zero
#     current_negative_mean = negative_values.mean() if negative_values.numel() else zero
#     valid_fraction = valid_anchor.to(dtype=q.dtype).mean()
#     values = (
#         loss,
#         valid_fraction,
#         positive_mean,
#         current_negative_mean,
#         historical_negative_mean,
#         support_margin.mean(),
#         support_margin.min(),
#         physical_confusability,
#         pair_risk,
#     )
#     if not all(bool(torch.isfinite(v)) for v in values):
#         raise RuntimeError("unified Whole-CDS contrastive objective became NaN or Inf")

#     return WholeCDSContrastiveObjective(
#         loss=loss,
#         valid_anchor_fraction=valid_fraction,
#         positive_similarity_mean=positive_mean,
#         current_negative_similarity_mean=current_negative_mean,
#         historical_negative_similarity_mean=historical_negative_mean,
#         whole_support_margin_mean=support_margin.mean(),
#         whole_support_margin_min=support_margin.min(),
#         physical_confusability_mean=physical_confusability,
#         effective_pair_risk_mean=pair_risk,
#         violation_fraction=support_margin.le(0.0).to(q.dtype).mean(),
#     )


# # Backward-compatible name; active trainers use the continual name below.
# def whole_cds_pic_loss(**kwargs) -> WholeCDSContrastiveObjective:
#     return whole_cds_contrastive_loss(**kwargs)


# def cross_phase_cds_separation_loss(
#     *,
#     current_features: Tensor,
#     labels_global: Tensor,
#     physical_spectral_evidence: Tensor,
#     seen_cds: ClassDiscriminativeStructureState,
#     historical_class_count: int,
#     logit_scale: Tensor,
#     sample_weights: Tensor | None = None,
# ) -> CDSSeparationObjective:
#     """Keep real current Q outside migrated historical CDS regions.

#     This is the feature-space counterpart of classifier-side Whole-CDS
#     reconstruction.  The positive is the current class CDS.  Every historical
#     class is a rival.  Separation is measured only in learned Q support, while
#     native spectral evidence and whole-CDS pair risk *prioritize* spectrally
#     credible old/new conflicts.  P therefore cannot invent a semantic boundary
#     on its own and no classifier weight participates in this loss.
#     """
#     if not isinstance(seen_cds, ClassDiscriminativeStructureState):
#         raise TypeError("seen_cds must be ClassDiscriminativeStructureState")
#     q = normalize_ordered_decision_features(torch.as_tensor(current_features))
#     p = torch.as_tensor(
#         physical_spectral_evidence, device=q.device, dtype=q.dtype
#     ).detach()
#     labels = torch.as_tensor(labels_global, device=q.device, dtype=torch.long).flatten()
#     if labels.shape != (q.size(0),):
#         raise ValueError("labels_global must be [N]")
#     if p.ndim != 3 or p.size(0) != q.size(0) or p.size(2) != 1:
#         raise ValueError("physical_spectral_evidence must be [N,B,1]")
#     if not bool(torch.isfinite(p).all()):
#         raise ValueError("physical_spectral_evidence contains NaN or Inf")

#     classes = seen_cds.classes
#     ids = classes.class_ids.to(device=q.device, dtype=torch.long).flatten()
#     count = int(ids.numel())
#     old_count = int(historical_class_count)
#     if count < 2 or old_count <= 0 or old_count >= count:
#         raise ValueError("historical_class_count must split old and current classes")
#     if classes.learned_identity.shape[1:] != q.shape[1:]:
#         raise ValueError("current Q and learned CDS use different coordinates")
#     if classes.physical_identity.shape[1:] != p.shape[1:]:
#         raise ValueError("physical evidence and physical CDS use different coordinates")

#     matches = labels.unsqueeze(1).eq(ids.unsqueeze(0))
#     if not bool(matches.sum(dim=1).eq(1).all()):
#         raise ValueError("a current label is absent from seen CDS")
#     target_index = matches.to(torch.long).argmax(dim=1)
#     if bool((target_index < old_count).any()):
#         raise ValueError("cross-phase separation expects current-class labels only")

#     _, _, learned_support = ordered_cds_support_components(
#         q,
#         classes.learned_identity.to(device=q.device, dtype=q.dtype).detach(),
#         classes.learned_variation.to(device=q.device, dtype=q.dtype).detach(),
#     )
#     with torch.no_grad():
#         _, _, physical_support = ordered_cds_support_components(
#             p,
#             classes.physical_identity.to(device=q.device, dtype=q.dtype).detach(),
#             classes.physical_variation.to(device=q.device, dtype=q.dtype).detach(),
#         )
#         effective_risk, _learned_risk, _physical_risk = whole_cds_pair_risk(seen_cds)
#         effective_risk = effective_risk.to(device=q.device, dtype=q.dtype)

#     rows = torch.arange(q.size(0), device=q.device)
#     positive_learned = learned_support[rows, target_index]
#     positive_physical = physical_support[rows, target_index]
#     rival_learned = learned_support[:, :old_count]
#     rival_physical = physical_support[:, :old_count]

#     margin = positive_learned.unsqueeze(1) - rival_learned
#     pair_risk = effective_risk.index_select(0, target_index)[:, :old_count]
#     # Sample-specific spectral ambiguity strengthens only a relation already
#     # judged relevant in learned+physical class structure.
#     spectral_confusability = torch.sqrt(
#         positive_physical.unsqueeze(1).clamp_min(0.0)
#         * rival_physical.clamp_min(0.0)
#     ).clamp(0.0, 1.0)
#     pair_weight = 1.0 + pair_risk * spectral_confusability

#     scale = torch.as_tensor(logit_scale, device=q.device, dtype=q.dtype).detach()
#     if scale.ndim != 0 or not bool(torch.isfinite(scale)) or not bool(scale > 0):
#         raise ValueError("logit_scale must be a finite positive scalar")
#     pair_loss = F.softplus(-scale * margin)
#     per_sample = (pair_weight * pair_loss).sum(dim=1) / pair_weight.sum(dim=1).clamp_min(
#         torch.finfo(q.dtype).eps
#     )
#     loss = _weighted_mean(per_sample, sample_weights)

#     values = (
#         loss,
#         positive_learned.mean(),
#         rival_learned.mean(),
#         margin.mean(),
#         margin.min(),
#         spectral_confusability.mean(),
#         pair_risk.mean(),
#         margin.le(0.0).to(q.dtype).mean(),
#     )
#     if not all(bool(torch.isfinite(value)) for value in values):
#         raise RuntimeError("cross-phase CDS separation became NaN or Inf")
#     return CDSSeparationObjective(
#         loss=loss,
#         learned_positive_support_mean=positive_learned.mean(),
#         learned_historical_rival_support_mean=rival_learned.mean(),
#         learned_support_margin_mean=margin.mean(),
#         learned_support_margin_min=margin.min(),
#         physical_confusability_mean=spectral_confusability.mean(),
#         effective_pair_risk_mean=pair_risk.mean(),
#         violation_fraction=margin.le(0.0).to(q.dtype).mean(),
#     )


# __all__ = [
#     "ClassificationObjective",
#     "SupervisedContrastiveObjective",
#     "LogitDistillationObjective",
#     "FeatureDistillationObjective",
#     "CDSPreservationObjective",
#     "CDSSeparationObjective",
#     "WholeCDSContrastiveObjective",
#     "WholeCDSPICObjective",
#     "classification_objective",
#     "supervised_contrastive_q_loss",
#     "logit_distillation_loss",
#     "ordered_feature_distillation_loss",
#     "historical_cds_preservation_loss",
#     "whole_cds_contrastive_loss",
#     "whole_cds_pic_loss",
#     "cross_phase_cds_separation_loss",
# ]


# from __future__ import annotations
# from dataclasses import dataclass

# import torch
# import torch.nn.functional as F

# from models.geometry_bank import (
#     ClassDiscriminativeStructureState,
#     ClassStructureBatch,
#     normalize_ordered_decision_features,
#     ordered_cds_support_components,
#     whole_cds_pair_risk,
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
# class LogitDistillationObjective:
#     loss: Tensor
#     teacher_entropy_mean: Tensor
#     student_entropy_mean: Tensor


# @dataclass(frozen=True)
# class FeatureDistillationObjective:
#     loss: Tensor
#     cosine_similarity_mean: Tensor
#     rms_drift: Tensor


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


# @dataclass(frozen=True)
# class CDSSeparationObjective:
#     loss: Tensor
#     learned_positive_support_mean: Tensor
#     learned_historical_rival_support_mean: Tensor
#     learned_support_margin_mean: Tensor
#     learned_support_margin_min: Tensor
#     physical_confusability_mean: Tensor
#     effective_pair_risk_mean: Tensor
#     violation_fraction: Tensor

# @dataclass(frozen=True)
# class WholeCDSPICObjective:
#     loss: Tensor
#     valid_anchor_fraction: Tensor
#     positive_similarity_mean: Tensor
#     current_negative_similarity_mean: Tensor
#     historical_negative_similarity_mean: Tensor
#     learned_support_margin_mean: Tensor
#     learned_support_margin_min: Tensor
#     physical_confusability_mean: Tensor
#     effective_pair_risk_mean: Tensor
#     violation_fraction: Tensor

#     @property
#     def negative_similarity_mean(self) -> Tensor:
#         return self.current_negative_similarity_mean

#     @property
#     def similarity_gap(self) -> Tensor:
#         return self.positive_similarity_mean - self.current_negative_similarity_mean


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


# def ordered_feature_distillation_loss(
#     *,
#     reference_features: Tensor,
#     current_features: Tensor,
#     sample_weights: Tensor | None = None,
# ) -> FeatureDistillationObjective:
#     """FCS-style feature KD in the same ordered Q coordinate used by CDS.

#     The frozen previous representation is a training reference only.  Both Q^-
#     and Q^+ are mapped to the canonical cosine/CDS coordinate before the loss so
#     this term constrains semantic direction rather than arbitrary feature norm.
#     Token order is preserved; no second representation is created.
#     """
#     reference = normalize_ordered_decision_features(
#         torch.as_tensor(reference_features)
#     ).detach()
#     current = normalize_ordered_decision_features(torch.as_tensor(current_features))
#     if current.shape != reference.shape:
#         raise ValueError("current_features must match reference_features")
#     if current.device != reference.device or current.dtype != reference.dtype:
#         raise ValueError("reference/current Q must share device/dtype")
#     delta = current - reference

#     # IMPORTANT: do not use sqrt(mean(delta**2)) as the optimization loss.
#     # At the beginning of an incremental phase the current backbone is an exact
#     # copy of the frozen teacher, so delta == 0.  The forward value sqrt(0) is
#     # finite, but its derivative is undefined and PyTorch returns NaN gradients.
#     #
#     # Because both tensors are already globally L2-normalized per sample,
#     #   0.5 * ||q_t - q_{t-1}||_2^2 == 1 - cos(q_t, q_{t-1}).
#     # This squared chord/cosine distance has the same semantic objective, is
#     # exactly zero at phase start, and has an exactly zero (finite) gradient
#     # when student and teacher coincide.  Token order is still preserved because
#     # the distance is computed over the ordered KxD tensor without pooling into
#     # a second representation.
#     per_sample = 0.5 * delta.square().sum(dim=(1, 2))
#     loss = _weighted_mean(per_sample, sample_weights)
#     flat_ref = reference.reshape(reference.size(0), -1)
#     flat_cur = current.reshape(current.size(0), -1)
#     cosine = F.cosine_similarity(flat_cur, flat_ref, dim=1).mean()

#     # RMS is diagnostic only.  Detaching avoids retaining a second sqrt branch
#     # in the optimization graph and makes the phase-start zero-drift case safe.
#     with torch.no_grad():
#         rms = delta.detach().square().mean().sqrt()
#     if not all(bool(torch.isfinite(v)) for v in (loss, cosine, rms)):
#         raise RuntimeError("ordered feature distillation became NaN or Inf")
#     return FeatureDistillationObjective(
#         loss=loss,
#         cosine_similarity_mean=cosine,
#         rms_drift=rms,
#     )

# def _whole_cds_support_components(
#     *,
#     learned_features: Tensor,
#     physical_features: Tensor,
#     structure: ClassStructureBatch,
# ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
#     q = torch.as_tensor(learned_features)
#     p = torch.as_tensor(physical_features, device=q.device, dtype=q.dtype)
#     if q.ndim != 3 or p.ndim != 3 or q.size(0) != p.size(0):
#         raise ValueError("learned/physical features must be aligned rank-three batches")
#     i_q = structure.learned_identity.to(device=q.device, dtype=q.dtype).detach()
#     v_q = structure.learned_variation.to(device=q.device, dtype=q.dtype).detach()
#     i_p = structure.physical_identity.to(device=q.device, dtype=q.dtype).detach()
#     v_p = structure.physical_variation.to(device=q.device, dtype=q.dtype).detach()
#     learned_abs, learned_rel, learned_full = ordered_cds_support_components(q, i_q, v_q)
#     physical_abs, physical_rel, physical_full = ordered_cds_support_components(p, i_p, v_p)
#     # Q and W live in the learned decision coordinate.  Native spectral CDS is
#     # evidence about which learned-space boundaries deserve stronger protection;
#     # it is not a second class score that disappears at inference.
#     return learned_abs, learned_rel, learned_full, physical_abs, physical_rel, physical_full


# def _whole_cds_cross_entropy(
#     *,
#     whole_absolute: Tensor,
#     whole_relation: Tensor,
#     whole_complete: Tensor,
#     targets_local: Tensor,
#     sample_weights: Tensor | None,
# ) -> Tensor:
#     """Class-balanced CE on absolute, relation and complete Whole-CDS support."""
#     eps = torch.finfo(whole_complete.dtype).eps
#     losses = []
#     for support in (whole_absolute, whole_relation, whole_complete):
#         per_sample = F.cross_entropy(support.clamp_min(eps).log(), targets_local, reduction="none")
#         losses.append(_weighted_mean(per_sample, sample_weights))
#     return sum(losses) / float(len(losses))


# def historical_cds_preservation_loss(
#     *,
#     reference_features: Tensor,
#     current_features: Tensor,
#     physical_spectral_evidence: Tensor,
#     reference_historical_structure: ClassStructureBatch,
#     current_historical_structure: ClassStructureBatch,
# ) -> CDSPreservationObjective:
#     """Preserve historical Whole-CDS intra/inter structure on real current HSI.

#     Both teacher and student are evaluated against the same finalized historical
#     coordinate during incremental SGD.  The optimized supports are *direct*
#     Whole-CDS scores: learned ordered structure (I_Q,V_Q) is geometrically fused
#     with native spectral structure (I_P,V_P) before intra- and inter-class errors
#     are formed.  Physical CDS therefore participates in the actual preserved
#     class score rather than merely weighting a learned-space loss.
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
#     physical = torch.as_tensor(physical_spectral_evidence, device=current.device, dtype=current.dtype).detach()
#     if current.shape != reference.shape or current.device != reference.device or current.dtype != reference.dtype:
#         raise ValueError("current_features must match reference_features")
#     if physical.ndim != 3 or physical.size(0) != reference.size(0) or physical.size(2) != 1:
#         raise ValueError("physical_spectral_evidence must be [N,B,1]")

#     with torch.no_grad():
#         teacher_abs, teacher_rel, teacher_full, physical_abs, physical_rel, physical_full = (
#             _whole_cds_support_components(
#                 learned_features=reference,
#                 physical_features=physical,
#                 structure=ref_struct,
#             )
#         )
#     student_abs, student_rel, student_full, _, _, _ = _whole_cds_support_components(
#         learned_features=current,
#         physical_features=physical,
#         structure=cur_struct,
#     )

#     # Intra-CDS: preserve the class-wise absolute and ordered-relation support.
#     intra_error = 0.5 * ((student_abs - teacher_abs).abs() + (student_rel - teacher_rel).abs())
#     intra_loss = intra_error.mean()

#     # Inter-CDS: preserve all historical pairwise support differences.
#     class_count = ref_struct.class_count
#     if class_count >= 2:
#         upper = torch.triu(
#             torch.ones(class_count, class_count, device=current.device, dtype=torch.bool), diagonal=1
#         )
#         teacher_pair = teacher_full.unsqueeze(2) - teacher_full.unsqueeze(1)
#         student_pair = student_full.unsqueeze(2) - student_full.unsqueeze(1)
#         inter_loss = (student_pair - teacher_pair).abs()[:, upper].mean()
#     else:
#         inter_loss = current.sum() * 0.0

#     loss = intra_loss + inter_loss
#     values = (
#         loss, intra_loss, inter_loss,
#         physical_abs.mean(), physical_rel.mean(), physical_full.mean(),
#         teacher_abs.mean(), teacher_rel.mean(), student_abs.mean(), student_rel.mean(),
#     )
#     if not all(bool(torch.isfinite(value)) for value in values):
#         raise RuntimeError("historical Whole-CDS preservation objective became NaN or Inf")

#     return CDSPreservationObjective(
#         loss=loss,
#         intra_loss=intra_loss,
#         inter_loss=inter_loss,
#         physical_absolute_support_mean=physical_abs.mean(),
#         physical_relation_support_mean=physical_rel.mean(),
#         physical_relevance_mean=physical_full.mean(),
#         teacher_absolute_support_mean=teacher_abs.mean(),
#         teacher_relation_support_mean=teacher_rel.mean(),
#         student_absolute_support_mean=student_abs.mean(),
#         student_relation_support_mean=student_rel.mean(),
#     )


# def whole_cds_contrastive_loss(
#     *,
#     current_features: Tensor,
#     labels_global: Tensor,
#     physical_spectral_evidence: Tensor,
#     seen_cds: ClassDiscriminativeStructureState,
#     historical_class_count: int,
#     logit_scale: Tensor,
#     sample_weights: Tensor | None = None,
# ) -> WholeCDSPICObjective:
#     """One continual contrastive objective for base and incremental phases.

#     The objective has two simultaneous parts:
#       1) ordinary sample/sample supervised contrastive learning in the single Q;
#       2) direct all-seen Whole-CDS discrimination using absolute, ordered-relation,
#          and complete supports fused from (I_Q,V_Q) and (I_P,V_P).

#     In the base phase ``historical_class_count=0``.  In incremental phases the
#     same class-structure softmax naturally includes historical CDS as rivals.
#     There is no I_Q-only historical anchor and spectral information is not merely
#     a weighting term.
#     """
#     if not isinstance(seen_cds, ClassDiscriminativeStructureState):
#         raise TypeError("seen_cds must be ClassDiscriminativeStructureState")
#     q = normalize_ordered_decision_features(torch.as_tensor(current_features))
#     labels = torch.as_tensor(labels_global, device=q.device, dtype=torch.long).flatten()
#     physical = torch.as_tensor(physical_spectral_evidence, device=q.device, dtype=q.dtype).detach()
#     if labels.shape != (q.size(0),):
#         raise ValueError("labels_global must be [N]")
#     if physical.ndim != 3 or physical.size(0) != q.size(0) or physical.size(2) != 1:
#         raise ValueError("physical_spectral_evidence must be [N,B,1]")

#     classes = seen_cds.classes
#     ids = classes.class_ids.to(device=q.device, dtype=torch.long).flatten()
#     count = int(ids.numel())
#     old_count = int(historical_class_count)
#     if count < 2 or old_count < 0 or old_count >= count:
#         raise ValueError("historical_class_count must be in [0, C-1]")
#     structure = ClassStructureBatch(
#         class_ids=classes.class_ids.to(device=q.device),
#         sample_counts=classes.sample_counts.to(device=q.device),
#         learned_identity=classes.learned_identity.to(device=q.device, dtype=q.dtype),
#         learned_variation=classes.learned_variation.to(device=q.device, dtype=q.dtype),
#         physical_identity=classes.physical_identity.to(device=q.device, dtype=q.dtype),
#         physical_variation=classes.physical_variation.to(device=q.device, dtype=q.dtype),
#     )
#     if structure.learned_identity.shape[1:] != q.shape[1:]:
#         raise ValueError("current Q and learned CDS use different coordinates")
#     if structure.physical_identity.shape[1:] != physical.shape[1:]:
#         raise ValueError("physical evidence and physical CDS use different coordinates")

#     matches = labels.unsqueeze(1).eq(ids.unsqueeze(0))
#     if not bool(matches.sum(dim=1).eq(1).all()):
#         raise ValueError("a current label is absent from seen CDS")
#     target_index = matches.to(torch.long).argmax(dim=1)
#     if old_count > 0 and bool((target_index < old_count).any()):
#         raise ValueError("incremental batches must contain only incoming classes")

#     scale = torch.as_tensor(logit_scale, device=q.device, dtype=q.dtype).detach()
#     if scale.ndim != 0 or not bool(torch.isfinite(scale)) or not bool(scale > 0):
#         raise ValueError("logit_scale must be a finite positive scalar")
#     eps = torch.finfo(q.dtype).eps

#     # Sample/sample supervised contrastive term.
#     flat = F.normalize(q.reshape(q.size(0), -1), p=2, dim=1, eps=eps)
#     similarity = (flat @ flat.t()).clamp(-1.0, 1.0) * scale
#     n = int(q.size(0))
#     eye = torch.eye(n, device=q.device, dtype=torch.bool)
#     same = labels[:, None].eq(labels[None, :])
#     positive_mask = same & ~eye
#     negative_mask = (~same) & ~eye
#     logits = similarity.masked_fill(eye, -torch.inf)
#     log_denom = torch.logsumexp(logits, dim=1)
#     positive_count = positive_mask.sum(dim=1)
#     valid_anchor = positive_count > 0
#     if bool(valid_anchor.any()):
#         log_probability = similarity - log_denom.unsqueeze(1)
#         per_anchor = -(log_probability * positive_mask.to(q.dtype)).sum(dim=1) / positive_count.clamp_min(1).to(q.dtype)
#         if sample_weights is None:
#             supcon_loss = per_anchor[valid_anchor].mean()
#         else:
#             weights = torch.as_tensor(sample_weights)
#             if weights.shape != (n,) or weights.device != q.device or weights.dtype != q.dtype:
#                 raise ValueError("sample_weights must be [N] and match Q device/dtype")
#             supcon_loss = _weighted_mean(per_anchor[valid_anchor], weights[valid_anchor])
#     else:
#         supcon_loss = q.sum() * 0.0

#     # Direct class-structure term.  All four CDS fields are inside the optimized
#     # class score, and all seen classes compete in the same softmax.
#     learned_abs, learned_rel, learned_full, physical_abs, physical_rel, physical_full = (
#         _whole_cds_support_components(
#             learned_features=q, physical_features=physical, structure=structure
#         )
#     )
#     structural_loss = _whole_cds_cross_entropy(
#         whole_absolute=learned_abs,
#         whole_relation=learned_rel,
#         whole_complete=learned_full,
#         targets_local=target_index,
#         sample_weights=sample_weights,
#     )

#     rows = torch.arange(n, device=q.device)
#     positive_whole = learned_full[rows, target_index]
#     rival_mask = ~torch.eye(count, device=q.device, dtype=torch.bool).index_select(0, target_index)
#     rival_whole = learned_full.masked_fill(~rival_mask, -torch.inf)
#     hardest_rival = rival_whole.max(dim=1).values
#     support_margin = positive_whole - hardest_rival

#     # Whole-CDS inter-class information must affect optimization, not only
#     # diagnostics.  In incremental phases every historical CDS is an explicit
#     # rival.  The learned+physical class-pair risk and sample-specific physical
#     # confusability strengthen exactly the old/new pairs that are structurally
#     # dangerous.  No pseudo feature/image is created.
#     zero = q.new_zeros(())
#     if old_count > 0:
#         with torch.no_grad():
#             effective_risk, _, _ = whole_cds_pair_risk(seen_cds)
#             old_pair_risk = effective_risk.to(
#                 device=q.device, dtype=q.dtype
#             ).index_select(0, target_index)[:, :old_count]
#             positive_physical = physical_full[rows, target_index]
#             old_physical = physical_full[:, :old_count]
#             old_physical_confusability = torch.sqrt(
#                 positive_physical.unsqueeze(1).clamp_min(0.0)
#                 * old_physical.clamp_min(0.0)
#             ).clamp(0.0, 1.0)
#             old_pair_weight = 1.0 + old_pair_risk * old_physical_confusability

#         old_margin = positive_whole.unsqueeze(1) - learned_full[:, :old_count]
#         old_pair_loss = F.softplus(-scale * old_margin)
#         per_sample_old_separation = (
#             old_pair_weight * old_pair_loss
#         ).sum(dim=1) / old_pair_weight.sum(dim=1).clamp_min(eps)
#         historical_separation_loss = _weighted_mean(
#             per_sample_old_separation, sample_weights
#         )
#         physical_confusability = old_physical_confusability.mean()
#         pair_risk = old_pair_risk.mean()
#     else:
#         historical_separation_loss = zero
#         physical_confusability = physical_full.mean()
#         pair_risk = zero

#     # One separation objective: current/current SupCon + all-seen Whole-CDS CE
#     # + risk-aware old/new CDS margin.  The final term is zero in the base phase.
#     loss = supcon_loss + structural_loss + historical_separation_loss

#     positive_values = similarity[positive_mask]
#     negative_values = similarity[negative_mask]
#     positive_mean = positive_values.mean() if positive_values.numel() else zero
#     current_negative_mean = negative_values.mean() if negative_values.numel() else zero
#     historical_negative_mean = (
#         learned_full[:, :old_count].mean() if old_count > 0 else zero
#     )
#     valid_fraction = valid_anchor.to(dtype=q.dtype).mean()

#     values = (loss, supcon_loss, structural_loss, support_margin.mean(), support_margin.min())
#     if not all(bool(torch.isfinite(v)) for v in values):
#         raise RuntimeError("Whole-CDS contrastive objective became NaN or Inf")
#     return WholeCDSPICObjective(
#         loss=loss,
#         valid_anchor_fraction=valid_fraction,
#         positive_similarity_mean=positive_mean,
#         current_negative_similarity_mean=current_negative_mean,
#         historical_negative_similarity_mean=historical_negative_mean,
#         learned_support_margin_mean=support_margin.mean(),
#         learned_support_margin_min=support_margin.min(),
#         physical_confusability_mean=physical_confusability,
#         effective_pair_risk_mean=pair_risk,
#         violation_fraction=support_margin.le(0.0).to(q.dtype).mean(),
#     )


# # Backward-compatible name; active trainers use the continual name below.
# def whole_cds_pic_loss(**kwargs) -> WholeCDSPICObjective:
#     return whole_cds_contrastive_loss(**kwargs)


# def cross_phase_cds_separation_loss(
#     *,
#     current_features: Tensor,
#     labels_global: Tensor,
#     physical_spectral_evidence: Tensor,
#     seen_cds: ClassDiscriminativeStructureState,
#     historical_class_count: int,
#     logit_scale: Tensor,
#     sample_weights: Tensor | None = None,
# ) -> CDSSeparationObjective:
#     """Keep real current Q outside migrated historical CDS regions.

#     This is the feature-space counterpart of classifier-side Whole-CDS
#     reconstruction.  The positive is the current class CDS.  Every historical
#     class is a rival.  Separation is measured only in learned Q support, while
#     native spectral evidence and whole-CDS pair risk *prioritize* spectrally
#     credible old/new conflicts.  P therefore cannot invent a semantic boundary
#     on its own and no classifier weight participates in this loss.
#     """
#     if not isinstance(seen_cds, ClassDiscriminativeStructureState):
#         raise TypeError("seen_cds must be ClassDiscriminativeStructureState")
#     q = normalize_ordered_decision_features(torch.as_tensor(current_features))
#     p = torch.as_tensor(
#         physical_spectral_evidence, device=q.device, dtype=q.dtype
#     ).detach()
#     labels = torch.as_tensor(labels_global, device=q.device, dtype=torch.long).flatten()
#     if labels.shape != (q.size(0),):
#         raise ValueError("labels_global must be [N]")
#     if p.ndim != 3 or p.size(0) != q.size(0) or p.size(2) != 1:
#         raise ValueError("physical_spectral_evidence must be [N,B,1]")
#     if not bool(torch.isfinite(p).all()):
#         raise ValueError("physical_spectral_evidence contains NaN or Inf")

#     classes = seen_cds.classes
#     ids = classes.class_ids.to(device=q.device, dtype=torch.long).flatten()
#     count = int(ids.numel())
#     old_count = int(historical_class_count)
#     if count < 2 or old_count <= 0 or old_count >= count:
#         raise ValueError("historical_class_count must split old and current classes")
#     if classes.learned_identity.shape[1:] != q.shape[1:]:
#         raise ValueError("current Q and learned CDS use different coordinates")
#     if classes.physical_identity.shape[1:] != p.shape[1:]:
#         raise ValueError("physical evidence and physical CDS use different coordinates")

#     matches = labels.unsqueeze(1).eq(ids.unsqueeze(0))
#     if not bool(matches.sum(dim=1).eq(1).all()):
#         raise ValueError("a current label is absent from seen CDS")
#     target_index = matches.to(torch.long).argmax(dim=1)
#     if bool((target_index < old_count).any()):
#         raise ValueError("cross-phase separation expects current-class labels only")

#     _, _, learned_support = ordered_cds_support_components(
#         q,
#         classes.learned_identity.to(device=q.device, dtype=q.dtype).detach(),
#         classes.learned_variation.to(device=q.device, dtype=q.dtype).detach(),
#     )
#     with torch.no_grad():
#         _, _, physical_support = ordered_cds_support_components(
#             p,
#             classes.physical_identity.to(device=q.device, dtype=q.dtype).detach(),
#             classes.physical_variation.to(device=q.device, dtype=q.dtype).detach(),
#         )
#         effective_risk, _learned_risk, _physical_risk = whole_cds_pair_risk(seen_cds)
#         effective_risk = effective_risk.to(device=q.device, dtype=q.dtype)

#     rows = torch.arange(q.size(0), device=q.device)
#     positive_learned = learned_support[rows, target_index]
#     positive_physical = physical_support[rows, target_index]
#     rival_learned = learned_support[:, :old_count]
#     rival_physical = physical_support[:, :old_count]

#     margin = positive_learned.unsqueeze(1) - rival_learned
#     pair_risk = effective_risk.index_select(0, target_index)[:, :old_count]
#     # Sample-specific spectral ambiguity strengthens only a relation already
#     # judged relevant in learned+physical class structure.
#     spectral_confusability = torch.sqrt(
#         positive_physical.unsqueeze(1).clamp_min(0.0)
#         * rival_physical.clamp_min(0.0)
#     ).clamp(0.0, 1.0)
#     pair_weight = 1.0 + pair_risk * spectral_confusability

#     scale = torch.as_tensor(logit_scale, device=q.device, dtype=q.dtype).detach()
#     if scale.ndim != 0 or not bool(torch.isfinite(scale)) or not bool(scale > 0):
#         raise ValueError("logit_scale must be a finite positive scalar")
#     pair_loss = F.softplus(-scale * margin)
#     per_sample = (pair_weight * pair_loss).sum(dim=1) / pair_weight.sum(dim=1).clamp_min(
#         torch.finfo(q.dtype).eps
#     )
#     loss = _weighted_mean(per_sample, sample_weights)

#     values = (
#         loss,
#         positive_learned.mean(),
#         rival_learned.mean(),
#         margin.mean(),
#         margin.min(),
#         spectral_confusability.mean(),
#         pair_risk.mean(),
#         margin.le(0.0).to(q.dtype).mean(),
#     )
#     if not all(bool(torch.isfinite(value)) for value in values):
#         raise RuntimeError("cross-phase CDS separation became NaN or Inf")
#     return CDSSeparationObjective(
#         loss=loss,
#         learned_positive_support_mean=positive_learned.mean(),
#         learned_historical_rival_support_mean=rival_learned.mean(),
#         learned_support_margin_mean=margin.mean(),
#         learned_support_margin_min=margin.min(),
#         physical_confusability_mean=spectral_confusability.mean(),
#         effective_pair_risk_mean=pair_risk.mean(),
#         violation_fraction=margin.le(0.0).to(q.dtype).mean(),
#     )


# __all__ = [
#     "ClassificationObjective",
#     "SupervisedContrastiveObjective",
#     "LogitDistillationObjective",
#     "FeatureDistillationObjective",
#     "CDSPreservationObjective",
#     "CDSSeparationObjective",
#     "WholeCDSPICObjective",
#     "classification_objective",
#     "supervised_contrastive_q_loss",
#     "logit_distillation_loss",
#     "ordered_feature_distillation_loss",
#     "historical_cds_preservation_loss",
#     "whole_cds_contrastive_loss",
#     "whole_cds_pic_loss",
#     "cross_phase_cds_separation_loss",
# ]


# from __future__ import annotations
# from dataclasses import dataclass

# import torch
# import torch.nn.functional as F

# from models.geometry_bank import (
#     ClassDiscriminativeStructureState,
#     ClassStructureBatch,
#     normalize_ordered_decision_features,
#     ordered_cds_support_components,
#     whole_cds_pair_risk,
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
# class LogitDistillationObjective:
#     loss: Tensor
#     teacher_entropy_mean: Tensor
#     student_entropy_mean: Tensor


# @dataclass(frozen=True)
# class FeatureDistillationObjective:
#     loss: Tensor
#     cosine_similarity_mean: Tensor
#     rms_drift: Tensor


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


# @dataclass(frozen=True)
# class CDSSeparationObjective:
#     loss: Tensor
#     learned_positive_support_mean: Tensor
#     learned_historical_rival_support_mean: Tensor
#     learned_support_margin_mean: Tensor
#     learned_support_margin_min: Tensor
#     physical_confusability_mean: Tensor
#     effective_pair_risk_mean: Tensor
#     violation_fraction: Tensor

# @dataclass(frozen=True)
# class WholeCDSPICObjective:
#     loss: Tensor
#     valid_anchor_fraction: Tensor
#     positive_similarity_mean: Tensor
#     current_negative_similarity_mean: Tensor
#     historical_negative_similarity_mean: Tensor
#     learned_support_margin_mean: Tensor
#     learned_support_margin_min: Tensor
#     physical_confusability_mean: Tensor
#     effective_pair_risk_mean: Tensor
#     violation_fraction: Tensor

#     @property
#     def negative_similarity_mean(self) -> Tensor:
#         return self.current_negative_similarity_mean

#     @property
#     def similarity_gap(self) -> Tensor:
#         return self.positive_similarity_mean - self.current_negative_similarity_mean


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


# def ordered_feature_distillation_loss(
#     *,
#     reference_features: Tensor,
#     current_features: Tensor,
#     sample_weights: Tensor | None = None,
# ) -> FeatureDistillationObjective:
#     """FCS-style feature KD in the same ordered Q coordinate used by CDS.

#     The frozen previous representation is a training reference only.  Both Q^-
#     and Q^+ are mapped to the canonical cosine/CDS coordinate before the loss so
#     this term constrains semantic direction rather than arbitrary feature norm.
#     Token order is preserved; no second representation is created.
#     """
#     reference = normalize_ordered_decision_features(
#         torch.as_tensor(reference_features)
#     ).detach()
#     current = normalize_ordered_decision_features(torch.as_tensor(current_features))
#     if current.shape != reference.shape:
#         raise ValueError("current_features must match reference_features")
#     if current.device != reference.device or current.dtype != reference.dtype:
#         raise ValueError("reference/current Q must share device/dtype")
#     delta = current - reference

#     # IMPORTANT: do not use sqrt(mean(delta**2)) as the optimization loss.
#     # At the beginning of an incremental phase the current backbone is an exact
#     # copy of the frozen teacher, so delta == 0.  The forward value sqrt(0) is
#     # finite, but its derivative is undefined and PyTorch returns NaN gradients.
#     #
#     # Because both tensors are already globally L2-normalized per sample,
#     #   0.5 * ||q_t - q_{t-1}||_2^2 == 1 - cos(q_t, q_{t-1}).
#     # This squared chord/cosine distance has the same semantic objective, is
#     # exactly zero at phase start, and has an exactly zero (finite) gradient
#     # when student and teacher coincide.  Token order is still preserved because
#     # the distance is computed over the ordered KxD tensor without pooling into
#     # a second representation.
#     per_sample = 0.5 * delta.square().sum(dim=(1, 2))
#     loss = _weighted_mean(per_sample, sample_weights)
#     flat_ref = reference.reshape(reference.size(0), -1)
#     flat_cur = current.reshape(current.size(0), -1)
#     cosine = F.cosine_similarity(flat_cur, flat_ref, dim=1).mean()

#     # RMS is diagnostic only.  Detaching avoids retaining a second sqrt branch
#     # in the optimization graph and makes the phase-start zero-drift case safe.
#     with torch.no_grad():
#         rms = delta.detach().square().mean().sqrt()
#     if not all(bool(torch.isfinite(v)) for v in (loss, cosine, rms)):
#         raise RuntimeError("ordered feature distillation became NaN or Inf")
#     return FeatureDistillationObjective(
#         loss=loss,
#         cosine_similarity_mean=cosine,
#         rms_drift=rms,
#     )

# def _whole_cds_support_components(
#     *,
#     learned_features: Tensor,
#     physical_features: Tensor,
#     structure: ClassStructureBatch,
# ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
#     """Return direct learned/physical/Whole-CDS class supports.

#     The Whole-CDS scores are the geometric fusion of the canonical learned
#     ordered support (I_Q,V_Q) and native spectral support (I_P,V_P).  These
#     fused scores are used directly by the optimization objectives; spectral
#     information is therefore part of the class score itself rather than only
#     a diagnostic or pair weight.
#     """
#     q = torch.as_tensor(learned_features)
#     p = torch.as_tensor(physical_features, device=q.device, dtype=q.dtype)
#     if q.ndim != 3 or p.ndim != 3 or q.size(0) != p.size(0):
#         raise ValueError("learned/physical features must be aligned rank-three batches")
#     i_q = structure.learned_identity.to(device=q.device, dtype=q.dtype).detach()
#     v_q = structure.learned_variation.to(device=q.device, dtype=q.dtype).detach()
#     i_p = structure.physical_identity.to(device=q.device, dtype=q.dtype).detach()
#     v_p = structure.physical_variation.to(device=q.device, dtype=q.dtype).detach()
#     learned_abs, learned_rel, learned_full = ordered_cds_support_components(q, i_q, v_q)
#     physical_abs, physical_rel, physical_full = ordered_cds_support_components(p, i_p, v_p)
#     eps = torch.finfo(q.dtype).eps

#     def fuse(a: Tensor, b: Tensor) -> Tensor:
#         return torch.sqrt(a.clamp_min(eps) * b.clamp_min(eps)).clamp(0.0, 1.0)

#     whole_abs = fuse(learned_abs, physical_abs)
#     whole_rel = fuse(learned_rel, physical_rel)
#     whole_full = fuse(learned_full, physical_full)
#     return whole_abs, whole_rel, whole_full, physical_abs, physical_rel, physical_full


# def _whole_cds_cross_entropy(
#     *,
#     whole_absolute: Tensor,
#     whole_relation: Tensor,
#     whole_complete: Tensor,
#     targets_local: Tensor,
#     sample_weights: Tensor | None,
# ) -> Tensor:
#     """Class-balanced CE on absolute, relation and complete Whole-CDS support."""
#     eps = torch.finfo(whole_complete.dtype).eps
#     losses = []
#     for support in (whole_absolute, whole_relation, whole_complete):
#         per_sample = F.cross_entropy(support.clamp_min(eps).log(), targets_local, reduction="none")
#         losses.append(_weighted_mean(per_sample, sample_weights))
#     return sum(losses) / float(len(losses))


# def historical_cds_preservation_loss(
#     *,
#     reference_features: Tensor,
#     current_features: Tensor,
#     physical_spectral_evidence: Tensor,
#     reference_historical_structure: ClassStructureBatch,
#     current_historical_structure: ClassStructureBatch,
# ) -> CDSPreservationObjective:
#     """Preserve historical Whole-CDS intra/inter structure on real current HSI.

#     Both teacher and student are evaluated against the same finalized historical
#     coordinate during incremental SGD.  The optimized supports are *direct*
#     Whole-CDS scores: learned ordered structure (I_Q,V_Q) is geometrically fused
#     with native spectral structure (I_P,V_P) before intra- and inter-class errors
#     are formed.  Physical CDS therefore participates in the actual preserved
#     class score rather than merely weighting a learned-space loss.
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
#     physical = torch.as_tensor(physical_spectral_evidence, device=current.device, dtype=current.dtype).detach()
#     if current.shape != reference.shape or current.device != reference.device or current.dtype != reference.dtype:
#         raise ValueError("current_features must match reference_features")
#     if physical.ndim != 3 or physical.size(0) != reference.size(0) or physical.size(2) != 1:
#         raise ValueError("physical_spectral_evidence must be [N,B,1]")

#     with torch.no_grad():
#         teacher_abs, teacher_rel, teacher_full, physical_abs, physical_rel, physical_full = (
#             _whole_cds_support_components(
#                 learned_features=reference,
#                 physical_features=physical,
#                 structure=ref_struct,
#             )
#         )
#     student_abs, student_rel, student_full, _, _, _ = _whole_cds_support_components(
#         learned_features=current,
#         physical_features=physical,
#         structure=cur_struct,
#     )

#     # Intra-CDS: preserve the class-wise absolute and ordered-relation support.
#     intra_error = 0.5 * ((student_abs - teacher_abs).abs() + (student_rel - teacher_rel).abs())
#     intra_loss = intra_error.mean()

#     # Inter-CDS: preserve all historical pairwise support differences.
#     class_count = ref_struct.class_count
#     if class_count >= 2:
#         upper = torch.triu(
#             torch.ones(class_count, class_count, device=current.device, dtype=torch.bool), diagonal=1
#         )
#         teacher_pair = teacher_full.unsqueeze(2) - teacher_full.unsqueeze(1)
#         student_pair = student_full.unsqueeze(2) - student_full.unsqueeze(1)
#         inter_loss = (student_pair - teacher_pair).abs()[:, upper].mean()
#     else:
#         inter_loss = current.sum() * 0.0

#     loss = intra_loss + inter_loss
#     values = (
#         loss, intra_loss, inter_loss,
#         physical_abs.mean(), physical_rel.mean(), physical_full.mean(),
#         teacher_abs.mean(), teacher_rel.mean(), student_abs.mean(), student_rel.mean(),
#     )
#     if not all(bool(torch.isfinite(value)) for value in values):
#         raise RuntimeError("historical Whole-CDS preservation objective became NaN or Inf")

#     return CDSPreservationObjective(
#         loss=loss,
#         intra_loss=intra_loss,
#         inter_loss=inter_loss,
#         physical_absolute_support_mean=physical_abs.mean(),
#         physical_relation_support_mean=physical_rel.mean(),
#         physical_relevance_mean=physical_full.mean(),
#         teacher_absolute_support_mean=teacher_abs.mean(),
#         teacher_relation_support_mean=teacher_rel.mean(),
#         student_absolute_support_mean=student_abs.mean(),
#         student_relation_support_mean=student_rel.mean(),
#     )


# def whole_cds_contrastive_loss(
#     *,
#     current_features: Tensor,
#     labels_global: Tensor,
#     physical_spectral_evidence: Tensor,
#     seen_cds: ClassDiscriminativeStructureState,
#     historical_class_count: int,
#     logit_scale: Tensor,
#     sample_weights: Tensor | None = None,
# ) -> WholeCDSPICObjective:
#     """One continual contrastive objective for base and incremental phases.

#     The objective has two simultaneous parts:
#       1) ordinary sample/sample supervised contrastive learning in the single Q;
#       2) direct all-seen Whole-CDS discrimination using absolute, ordered-relation,
#          and complete supports fused from (I_Q,V_Q) and (I_P,V_P).

#     In the base phase ``historical_class_count=0``.  In incremental phases the
#     same class-structure softmax naturally includes historical CDS as rivals.
#     There is no I_Q-only historical anchor and spectral information is not merely
#     a weighting term.
#     """
#     if not isinstance(seen_cds, ClassDiscriminativeStructureState):
#         raise TypeError("seen_cds must be ClassDiscriminativeStructureState")
#     q = normalize_ordered_decision_features(torch.as_tensor(current_features))
#     labels = torch.as_tensor(labels_global, device=q.device, dtype=torch.long).flatten()
#     physical = torch.as_tensor(physical_spectral_evidence, device=q.device, dtype=q.dtype).detach()
#     if labels.shape != (q.size(0),):
#         raise ValueError("labels_global must be [N]")
#     if physical.ndim != 3 or physical.size(0) != q.size(0) or physical.size(2) != 1:
#         raise ValueError("physical_spectral_evidence must be [N,B,1]")

#     classes = seen_cds.classes
#     ids = classes.class_ids.to(device=q.device, dtype=torch.long).flatten()
#     count = int(ids.numel())
#     old_count = int(historical_class_count)
#     if count < 2 or old_count < 0 or old_count >= count:
#         raise ValueError("historical_class_count must be in [0, C-1]")
#     structure = ClassStructureBatch(
#         class_ids=classes.class_ids.to(device=q.device),
#         sample_counts=classes.sample_counts.to(device=q.device),
#         learned_identity=classes.learned_identity.to(device=q.device, dtype=q.dtype),
#         learned_variation=classes.learned_variation.to(device=q.device, dtype=q.dtype),
#         physical_identity=classes.physical_identity.to(device=q.device, dtype=q.dtype),
#         physical_variation=classes.physical_variation.to(device=q.device, dtype=q.dtype),
#     )
#     if structure.learned_identity.shape[1:] != q.shape[1:]:
#         raise ValueError("current Q and learned CDS use different coordinates")
#     if structure.physical_identity.shape[1:] != physical.shape[1:]:
#         raise ValueError("physical evidence and physical CDS use different coordinates")

#     matches = labels.unsqueeze(1).eq(ids.unsqueeze(0))
#     if not bool(matches.sum(dim=1).eq(1).all()):
#         raise ValueError("a current label is absent from seen CDS")
#     target_index = matches.to(torch.long).argmax(dim=1)
#     if old_count > 0 and bool((target_index < old_count).any()):
#         raise ValueError("incremental batches must contain only incoming classes")

#     scale = torch.as_tensor(logit_scale, device=q.device, dtype=q.dtype).detach()
#     if scale.ndim != 0 or not bool(torch.isfinite(scale)) or not bool(scale > 0):
#         raise ValueError("logit_scale must be a finite positive scalar")
#     eps = torch.finfo(q.dtype).eps

#     # Sample/sample supervised contrastive term.
#     flat = F.normalize(q.reshape(q.size(0), -1), p=2, dim=1, eps=eps)
#     similarity = (flat @ flat.t()).clamp(-1.0, 1.0) * scale
#     n = int(q.size(0))
#     eye = torch.eye(n, device=q.device, dtype=torch.bool)
#     same = labels[:, None].eq(labels[None, :])
#     positive_mask = same & ~eye
#     negative_mask = (~same) & ~eye
#     logits = similarity.masked_fill(eye, -torch.inf)
#     log_denom = torch.logsumexp(logits, dim=1)
#     positive_count = positive_mask.sum(dim=1)
#     valid_anchor = positive_count > 0
#     if bool(valid_anchor.any()):
#         log_probability = similarity - log_denom.unsqueeze(1)
#         per_anchor = -(log_probability * positive_mask.to(q.dtype)).sum(dim=1) / positive_count.clamp_min(1).to(q.dtype)
#         if sample_weights is None:
#             supcon_loss = per_anchor[valid_anchor].mean()
#         else:
#             weights = torch.as_tensor(sample_weights)
#             if weights.shape != (n,) or weights.device != q.device or weights.dtype != q.dtype:
#                 raise ValueError("sample_weights must be [N] and match Q device/dtype")
#             supcon_loss = _weighted_mean(per_anchor[valid_anchor], weights[valid_anchor])
#     else:
#         supcon_loss = q.sum() * 0.0

#     # Direct class-structure term.  All four CDS fields are inside the optimized
#     # class score, and all seen classes compete in the same softmax.
#     whole_abs, whole_rel, whole_full, physical_abs, physical_rel, physical_full = (
#         _whole_cds_support_components(
#             learned_features=q, physical_features=physical, structure=structure
#         )
#     )
#     structural_loss = _whole_cds_cross_entropy(
#         whole_absolute=whole_abs,
#         whole_relation=whole_rel,
#         whole_complete=whole_full,
#         targets_local=target_index,
#         sample_weights=sample_weights,
#     )

#     rows = torch.arange(n, device=q.device)
#     positive_whole = whole_full[rows, target_index]
#     rival_mask = ~torch.eye(count, device=q.device, dtype=torch.bool).index_select(0, target_index)
#     rival_whole = whole_full.masked_fill(~rival_mask, -torch.inf)
#     hardest_rival = rival_whole.max(dim=1).values
#     support_margin = positive_whole - hardest_rival

#     # Whole-CDS inter-class information must affect optimization, not only
#     # diagnostics.  In incremental phases every historical CDS is an explicit
#     # rival.  The learned+physical class-pair risk and sample-specific physical
#     # confusability strengthen exactly the old/new pairs that are structurally
#     # dangerous.  No pseudo feature/image is created.
#     zero = q.new_zeros(())
#     if old_count > 0:
#         with torch.no_grad():
#             effective_risk, _, _ = whole_cds_pair_risk(seen_cds)
#             old_pair_risk = effective_risk.to(
#                 device=q.device, dtype=q.dtype
#             ).index_select(0, target_index)[:, :old_count]
#             positive_physical = physical_full[rows, target_index]
#             old_physical = physical_full[:, :old_count]
#             old_physical_confusability = torch.sqrt(
#                 positive_physical.unsqueeze(1).clamp_min(0.0)
#                 * old_physical.clamp_min(0.0)
#             ).clamp(0.0, 1.0)
#             old_pair_weight = 1.0 + old_pair_risk * old_physical_confusability

#         old_margin = positive_whole.unsqueeze(1) - whole_full[:, :old_count]
#         old_pair_loss = F.softplus(-scale * old_margin)
#         per_sample_old_separation = (
#             old_pair_weight * old_pair_loss
#         ).sum(dim=1) / old_pair_weight.sum(dim=1).clamp_min(eps)
#         historical_separation_loss = _weighted_mean(
#             per_sample_old_separation, sample_weights
#         )
#         physical_confusability = old_physical_confusability.mean()
#         pair_risk = old_pair_risk.mean()
#     else:
#         historical_separation_loss = zero
#         physical_confusability = physical_full.mean()
#         pair_risk = zero

#     # One separation objective: current/current SupCon + all-seen Whole-CDS CE
#     # + risk-aware old/new CDS margin.  The final term is zero in the base phase.
#     loss = supcon_loss + structural_loss + historical_separation_loss

#     positive_values = similarity[positive_mask]
#     negative_values = similarity[negative_mask]
#     positive_mean = positive_values.mean() if positive_values.numel() else zero
#     current_negative_mean = negative_values.mean() if negative_values.numel() else zero
#     historical_negative_mean = (
#         whole_full[:, :old_count].mean() if old_count > 0 else zero
#     )
#     valid_fraction = valid_anchor.to(dtype=q.dtype).mean()

#     values = (loss, supcon_loss, structural_loss, support_margin.mean(), support_margin.min())
#     if not all(bool(torch.isfinite(v)) for v in values):
#         raise RuntimeError("Whole-CDS contrastive objective became NaN or Inf")
#     return WholeCDSPICObjective(
#         loss=loss,
#         valid_anchor_fraction=valid_fraction,
#         positive_similarity_mean=positive_mean,
#         current_negative_similarity_mean=current_negative_mean,
#         historical_negative_similarity_mean=historical_negative_mean,
#         learned_support_margin_mean=support_margin.mean(),
#         learned_support_margin_min=support_margin.min(),
#         physical_confusability_mean=physical_confusability,
#         effective_pair_risk_mean=pair_risk,
#         violation_fraction=support_margin.le(0.0).to(q.dtype).mean(),
#     )


# # Backward-compatible name; active trainers use the continual name below.
# def whole_cds_pic_loss(**kwargs) -> WholeCDSPICObjective:
#     return whole_cds_contrastive_loss(**kwargs)


# def cross_phase_cds_separation_loss(
#     *,
#     current_features: Tensor,
#     labels_global: Tensor,
#     physical_spectral_evidence: Tensor,
#     seen_cds: ClassDiscriminativeStructureState,
#     historical_class_count: int,
#     logit_scale: Tensor,
#     sample_weights: Tensor | None = None,
# ) -> CDSSeparationObjective:
#     """Keep real current Q outside migrated historical CDS regions.

#     This is the feature-space counterpart of classifier-side Whole-CDS
#     reconstruction.  The positive is the current class CDS.  Every historical
#     class is a rival.  Separation is measured only in learned Q support, while
#     native spectral evidence and whole-CDS pair risk *prioritize* spectrally
#     credible old/new conflicts.  P therefore cannot invent a semantic boundary
#     on its own and no classifier weight participates in this loss.
#     """
#     if not isinstance(seen_cds, ClassDiscriminativeStructureState):
#         raise TypeError("seen_cds must be ClassDiscriminativeStructureState")
#     q = normalize_ordered_decision_features(torch.as_tensor(current_features))
#     p = torch.as_tensor(
#         physical_spectral_evidence, device=q.device, dtype=q.dtype
#     ).detach()
#     labels = torch.as_tensor(labels_global, device=q.device, dtype=torch.long).flatten()
#     if labels.shape != (q.size(0),):
#         raise ValueError("labels_global must be [N]")
#     if p.ndim != 3 or p.size(0) != q.size(0) or p.size(2) != 1:
#         raise ValueError("physical_spectral_evidence must be [N,B,1]")
#     if not bool(torch.isfinite(p).all()):
#         raise ValueError("physical_spectral_evidence contains NaN or Inf")

#     classes = seen_cds.classes
#     ids = classes.class_ids.to(device=q.device, dtype=torch.long).flatten()
#     count = int(ids.numel())
#     old_count = int(historical_class_count)
#     if count < 2 or old_count <= 0 or old_count >= count:
#         raise ValueError("historical_class_count must split old and current classes")
#     if classes.learned_identity.shape[1:] != q.shape[1:]:
#         raise ValueError("current Q and learned CDS use different coordinates")
#     if classes.physical_identity.shape[1:] != p.shape[1:]:
#         raise ValueError("physical evidence and physical CDS use different coordinates")

#     matches = labels.unsqueeze(1).eq(ids.unsqueeze(0))
#     if not bool(matches.sum(dim=1).eq(1).all()):
#         raise ValueError("a current label is absent from seen CDS")
#     target_index = matches.to(torch.long).argmax(dim=1)
#     if bool((target_index < old_count).any()):
#         raise ValueError("cross-phase separation expects current-class labels only")

#     _, _, learned_support = ordered_cds_support_components(
#         q,
#         classes.learned_identity.to(device=q.device, dtype=q.dtype).detach(),
#         classes.learned_variation.to(device=q.device, dtype=q.dtype).detach(),
#     )
#     with torch.no_grad():
#         _, _, physical_support = ordered_cds_support_components(
#             p,
#             classes.physical_identity.to(device=q.device, dtype=q.dtype).detach(),
#             classes.physical_variation.to(device=q.device, dtype=q.dtype).detach(),
#         )
#         effective_risk, _learned_risk, _physical_risk = whole_cds_pair_risk(seen_cds)
#         effective_risk = effective_risk.to(device=q.device, dtype=q.dtype)

#     rows = torch.arange(q.size(0), device=q.device)
#     positive_learned = learned_support[rows, target_index]
#     positive_physical = physical_support[rows, target_index]
#     rival_learned = learned_support[:, :old_count]
#     rival_physical = physical_support[:, :old_count]

#     margin = positive_learned.unsqueeze(1) - rival_learned
#     pair_risk = effective_risk.index_select(0, target_index)[:, :old_count]
#     # Sample-specific spectral ambiguity strengthens only a relation already
#     # judged relevant in learned+physical class structure.
#     spectral_confusability = torch.sqrt(
#         positive_physical.unsqueeze(1).clamp_min(0.0)
#         * rival_physical.clamp_min(0.0)
#     ).clamp(0.0, 1.0)
#     pair_weight = 1.0 + pair_risk * spectral_confusability

#     scale = torch.as_tensor(logit_scale, device=q.device, dtype=q.dtype).detach()
#     if scale.ndim != 0 or not bool(torch.isfinite(scale)) or not bool(scale > 0):
#         raise ValueError("logit_scale must be a finite positive scalar")
#     pair_loss = F.softplus(-scale * margin)
#     per_sample = (pair_weight * pair_loss).sum(dim=1) / pair_weight.sum(dim=1).clamp_min(
#         torch.finfo(q.dtype).eps
#     )
#     loss = _weighted_mean(per_sample, sample_weights)

#     values = (
#         loss,
#         positive_learned.mean(),
#         rival_learned.mean(),
#         margin.mean(),
#         margin.min(),
#         spectral_confusability.mean(),
#         pair_risk.mean(),
#         margin.le(0.0).to(q.dtype).mean(),
#     )
#     if not all(bool(torch.isfinite(value)) for value in values):
#         raise RuntimeError("cross-phase CDS separation became NaN or Inf")
#     return CDSSeparationObjective(
#         loss=loss,
#         learned_positive_support_mean=positive_learned.mean(),
#         learned_historical_rival_support_mean=rival_learned.mean(),
#         learned_support_margin_mean=margin.mean(),
#         learned_support_margin_min=margin.min(),
#         physical_confusability_mean=spectral_confusability.mean(),
#         effective_pair_risk_mean=pair_risk.mean(),
#         violation_fraction=margin.le(0.0).to(q.dtype).mean(),
#     )


# __all__ = [
#     "ClassificationObjective",
#     "SupervisedContrastiveObjective",
#     "LogitDistillationObjective",
#     "FeatureDistillationObjective",
#     "CDSPreservationObjective",
#     "CDSSeparationObjective",
#     "WholeCDSPICObjective",
#     "classification_objective",
#     "supervised_contrastive_q_loss",
#     "logit_distillation_loss",
#     "ordered_feature_distillation_loss",
#     "historical_cds_preservation_loss",
#     "whole_cds_contrastive_loss",
#     "whole_cds_pic_loss",
#     "cross_phase_cds_separation_loss",
# ]


# from __future__ import annotations
# from dataclasses import dataclass

# import torch
# import torch.nn.functional as F

# from models.geometry_bank import (
#     ClassDiscriminativeStructureState,
#     ClassStructureBatch,
#     normalize_ordered_decision_features,
#     ordered_cds_support_components,
#     whole_cds_pair_risk,
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
# class LogitDistillationObjective:
#     loss: Tensor
#     teacher_entropy_mean: Tensor
#     student_entropy_mean: Tensor


# @dataclass(frozen=True)
# class FeatureDistillationObjective:
#     loss: Tensor
#     cosine_similarity_mean: Tensor
#     rms_drift: Tensor


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


# @dataclass(frozen=True)
# class CDSSeparationObjective:
#     loss: Tensor
#     learned_positive_support_mean: Tensor
#     learned_historical_rival_support_mean: Tensor
#     learned_support_margin_mean: Tensor
#     learned_support_margin_min: Tensor
#     physical_confusability_mean: Tensor
#     effective_pair_risk_mean: Tensor
#     violation_fraction: Tensor

# @dataclass(frozen=True)
# class WholeCDSPICObjective:
#     loss: Tensor
#     valid_anchor_fraction: Tensor
#     positive_similarity_mean: Tensor
#     current_negative_similarity_mean: Tensor
#     historical_negative_similarity_mean: Tensor
#     learned_support_margin_mean: Tensor
#     learned_support_margin_min: Tensor
#     physical_confusability_mean: Tensor
#     effective_pair_risk_mean: Tensor
#     violation_fraction: Tensor


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


# def ordered_feature_distillation_loss(
#     *,
#     reference_features: Tensor,
#     current_features: Tensor,
#     sample_weights: Tensor | None = None,
# ) -> FeatureDistillationObjective:
#     """FCS-style feature KD in the same ordered Q coordinate used by CDS.

#     The frozen previous representation is a training reference only.  Both Q^-
#     and Q^+ are mapped to the canonical cosine/CDS coordinate before the loss so
#     this term constrains semantic direction rather than arbitrary feature norm.
#     Token order is preserved; no second representation is created.
#     """
#     reference = normalize_ordered_decision_features(
#         torch.as_tensor(reference_features)
#     ).detach()
#     current = normalize_ordered_decision_features(torch.as_tensor(current_features))
#     if current.shape != reference.shape:
#         raise ValueError("current_features must match reference_features")
#     if current.device != reference.device or current.dtype != reference.dtype:
#         raise ValueError("reference/current Q must share device/dtype")
#     delta = current - reference

#     # IMPORTANT: do not use sqrt(mean(delta**2)) as the optimization loss.
#     # At the beginning of an incremental phase the current backbone is an exact
#     # copy of the frozen teacher, so delta == 0.  The forward value sqrt(0) is
#     # finite, but its derivative is undefined and PyTorch returns NaN gradients.
#     #
#     # Because both tensors are already globally L2-normalized per sample,
#     #   0.5 * ||q_t - q_{t-1}||_2^2 == 1 - cos(q_t, q_{t-1}).
#     # This squared chord/cosine distance has the same semantic objective, is
#     # exactly zero at phase start, and has an exactly zero (finite) gradient
#     # when student and teacher coincide.  Token order is still preserved because
#     # the distance is computed over the ordered KxD tensor without pooling into
#     # a second representation.
#     per_sample = 0.5 * delta.square().sum(dim=(1, 2))
#     loss = _weighted_mean(per_sample, sample_weights)
#     flat_ref = reference.reshape(reference.size(0), -1)
#     flat_cur = current.reshape(current.size(0), -1)
#     cosine = F.cosine_similarity(flat_cur, flat_ref, dim=1).mean()

#     # RMS is diagnostic only.  Detaching avoids retaining a second sqrt branch
#     # in the optimization graph and makes the phase-start zero-drift case safe.
#     with torch.no_grad():
#         rms = delta.detach().square().mean().sqrt()
#     if not all(bool(torch.isfinite(v)) for v in (loss, cosine, rms)):
#         raise RuntimeError("ordered feature distillation became NaN or Inf")
#     return FeatureDistillationObjective(
#         loss=loss,
#         cosine_similarity_mean=cosine,
#         rms_drift=rms,
#     )

# def historical_cds_preservation_loss(
#     *,
#     reference_features: Tensor,
#     current_features: Tensor,
#     physical_spectral_evidence: Tensor,
#     reference_historical_structure: ClassStructureBatch,
#     current_historical_structure: ClassStructureBatch,
# ) -> CDSPreservationObjective:
#     """Preserve relations to the entire historical CDS on real current HSI.

#     Teacher and student supports are both evaluated against the immutable
#     finalized Phase-(t-1) historical CDS during incremental SGD.  Ordered feature
#     KD keeps Q_t compatible with that reference; semantic migration happens once
#     after convergence.  The full old-class support vector is preserved,
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
#         physical_absolute, physical_relation, physical_relevance = ordered_cds_support_components(
#             physical,
#             ref_struct.physical_identity.detach(),
#             ref_struct.physical_variation.detach(),
#         )
#         teacher_absolute, teacher_relation, teacher_complete = ordered_cds_support_components(
#             reference,
#             ref_struct.learned_identity.detach(),
#             ref_struct.learned_variation.detach(),
#         )

#     student_absolute, student_relation, student_complete = ordered_cds_support_components(
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


# def whole_cds_pic_loss(
#     *,
#     current_features: Tensor,
#     labels_global: Tensor,
#     physical_spectral_evidence: Tensor,
#     seen_cds: ClassDiscriminativeStructureState,
#     historical_class_count: int,
#     logit_scale: Tensor,
#     sample_weights: Tensor | None = None,
# ) -> WholeCDSPICObjective:
#     """Prototype-involved contrastive learning using the complete HSI Whole-CDS.

#     Incremental batches contain only current classes.  Ordinary supervised
#     contrastive learning therefore supplies only current/current attraction and
#     repulsion and can keep increasing plasticity while historical structure
#     deteriorates.  This objective follows the FCS separation principle but uses
#     the HSI Whole-CDS rather than a stored prototype:

#     * same current class: positive;
#     * different current class: negative;
#     * every historical ``I_Q``: historical negative anchor;
#     * ``V_Q`` and ``I_P/V_P`` enter through Whole-CDS pair risk and
#       sample-specific physical confusability, strengthening dangerous old/new
#       negatives without creating a second classifier.

#     The persistent memory remains exactly ``{I_Q,V_Q,I_P,V_P}``; no old feature
#     rows or synthetic samples are stored.
#     """
#     if not isinstance(seen_cds, ClassDiscriminativeStructureState):
#         raise TypeError("seen_cds must be ClassDiscriminativeStructureState")
#     q = normalize_ordered_decision_features(torch.as_tensor(current_features))
#     labels = torch.as_tensor(labels_global, device=q.device, dtype=torch.long).flatten()
#     physical = torch.as_tensor(
#         physical_spectral_evidence, device=q.device, dtype=q.dtype
#     ).detach()
#     if labels.shape != (q.size(0),):
#         raise ValueError("labels_global must be [N]")
#     if physical.ndim != 3 or physical.size(0) != q.size(0) or physical.size(2) != 1:
#         raise ValueError("physical_spectral_evidence must be [N,B,1]")
#     if not bool(torch.isfinite(physical).all()):
#         raise ValueError("physical_spectral_evidence contains NaN or Inf")

#     classes = seen_cds.classes
#     ids = classes.class_ids.to(device=q.device, dtype=torch.long).flatten()
#     count = int(ids.numel())
#     old_count = int(historical_class_count)
#     if count < 2 or old_count <= 0 or old_count >= count:
#         raise ValueError("historical_class_count must split old and current classes")
#     if classes.learned_identity.shape[1:] != q.shape[1:]:
#         raise ValueError("current Q and learned CDS use different coordinates")
#     if classes.physical_identity.shape[1:] != physical.shape[1:]:
#         raise ValueError("physical evidence and physical CDS use different coordinates")

#     matches = labels.unsqueeze(1).eq(ids.unsqueeze(0))
#     if not bool(matches.sum(dim=1).eq(1).all()):
#         raise ValueError("a current label is absent from seen CDS")
#     target_index = matches.to(torch.long).argmax(dim=1)
#     if bool((target_index < old_count).any()):
#         raise ValueError("Whole-CDS PIC expects current-class labels only")

#     scale = torch.as_tensor(logit_scale, device=q.device, dtype=q.dtype).detach()
#     if scale.ndim != 0 or not bool(torch.isfinite(scale)) or not bool(scale > 0):
#         raise ValueError("logit_scale must be a finite positive scalar")
#     eps = torch.finfo(q.dtype).eps

#     flat = F.normalize(q.reshape(q.size(0), -1), p=2, dim=1, eps=eps)
#     current_similarity = (flat @ flat.t()).clamp(-1.0, 1.0) * scale
#     n = int(q.size(0))
#     eye = torch.eye(n, device=q.device, dtype=torch.bool)
#     same = labels[:, None].eq(labels[None, :])
#     positive_mask = same & ~eye
#     current_negative_mask = (~same) & ~eye
#     candidate_mask = ~eye

#     historical_identity = classes.learned_identity[:old_count].to(
#         device=q.device, dtype=q.dtype
#     ).detach()
#     historical_anchor = F.normalize(
#         historical_identity.reshape(old_count, -1), p=2, dim=1, eps=eps
#     )
#     historical_similarity = (flat @ historical_anchor.t()).clamp(-1.0, 1.0) * scale

#     # Whole-CDS pair risk contains learned identity+relation and physical
#     # identity+relation structure.  Sample-specific physical support further
#     # emphasizes spectral confusions actually present in the current sample.
#     with torch.no_grad():
#         _, _, physical_support = ordered_cds_support_components(
#             physical,
#             classes.physical_identity.to(device=q.device, dtype=q.dtype).detach(),
#             classes.physical_variation.to(device=q.device, dtype=q.dtype).detach(),
#         )
#         effective_risk, _learned_risk, _physical_risk = whole_cds_pair_risk(seen_cds)
#         effective_risk = effective_risk.to(device=q.device, dtype=q.dtype)
#         rows = torch.arange(n, device=q.device)
#         positive_physical = physical_support[rows, target_index]
#         rival_physical = physical_support[:, :old_count]
#         spectral_confusability = torch.sqrt(
#             positive_physical.unsqueeze(1).clamp_min(0.0)
#             * rival_physical.clamp_min(0.0)
#         ).clamp(0.0, 1.0)
#         pair_risk = effective_risk.index_select(0, target_index)[:, :old_count]
#         historical_weight = 1.0 + pair_risk * spectral_confusability

#     current_logits = current_similarity.masked_fill(~candidate_mask, -torch.inf)
#     historical_logits = historical_similarity + historical_weight.clamp_min(eps).log()
#     log_denom = torch.logsumexp(torch.cat((current_logits, historical_logits), dim=1), dim=1)
#     positive_count = positive_mask.sum(dim=1)
#     valid_anchor = positive_count > 0
#     if bool(valid_anchor.any()):
#         log_probability = current_similarity - log_denom.unsqueeze(1)
#         per_anchor = -(
#             log_probability * positive_mask.to(dtype=q.dtype)
#         ).sum(dim=1) / positive_count.clamp_min(1).to(dtype=q.dtype)
#         if sample_weights is None:
#             loss = per_anchor[valid_anchor].mean()
#         else:
#             weights = torch.as_tensor(sample_weights)
#             if weights.shape != (n,) or weights.device != q.device or weights.dtype != q.dtype:
#                 raise ValueError("sample_weights must be [N] and match Q device/dtype")
#             loss = _weighted_mean(per_anchor[valid_anchor], weights[valid_anchor])
#     else:
#         loss = q.sum() * 0.0

#     # Structural diagnostics retain the same interpretable old/new CDS margin
#     # used by the previous separation loss.
#     _, _, learned_support = ordered_cds_support_components(
#         q,
#         classes.learned_identity.to(device=q.device, dtype=q.dtype).detach(),
#         classes.learned_variation.to(device=q.device, dtype=q.dtype).detach(),
#     )
#     rows = torch.arange(n, device=q.device)
#     positive_learned = learned_support[rows, target_index]
#     rival_learned = learned_support[:, :old_count]
#     support_margin = positive_learned.unsqueeze(1) - rival_learned

#     zero = q.new_zeros(())
#     positive_values = current_similarity[positive_mask]
#     current_negative_values = current_similarity[current_negative_mask]
#     positive_mean = positive_values.mean() if positive_values.numel() else zero
#     current_negative_mean = (
#         current_negative_values.mean() if current_negative_values.numel() else zero
#     )
#     historical_negative_mean = historical_similarity.mean()
#     valid_fraction = valid_anchor.to(dtype=q.dtype).mean()
#     values = (
#         loss,
#         valid_fraction,
#         positive_mean,
#         current_negative_mean,
#         historical_negative_mean,
#         support_margin.mean(),
#         support_margin.min(),
#         spectral_confusability.mean(),
#         pair_risk.mean(),
#         support_margin.le(0.0).to(q.dtype).mean(),
#     )
#     if not all(bool(torch.isfinite(v)) for v in values):
#         raise RuntimeError("Whole-CDS PIC objective became NaN or Inf")
#     return WholeCDSPICObjective(
#         loss=loss,
#         valid_anchor_fraction=valid_fraction,
#         positive_similarity_mean=positive_mean,
#         current_negative_similarity_mean=current_negative_mean,
#         historical_negative_similarity_mean=historical_negative_mean,
#         learned_support_margin_mean=support_margin.mean(),
#         learned_support_margin_min=support_margin.min(),
#         physical_confusability_mean=spectral_confusability.mean(),
#         effective_pair_risk_mean=pair_risk.mean(),
#         violation_fraction=support_margin.le(0.0).to(q.dtype).mean(),
#     )


# def cross_phase_cds_separation_loss(
#     *,
#     current_features: Tensor,
#     labels_global: Tensor,
#     physical_spectral_evidence: Tensor,
#     seen_cds: ClassDiscriminativeStructureState,
#     historical_class_count: int,
#     logit_scale: Tensor,
#     sample_weights: Tensor | None = None,
# ) -> CDSSeparationObjective:
#     """Keep real current Q outside migrated historical CDS regions.

#     This is the feature-space counterpart of classifier-side Whole-CDS
#     reconstruction.  The positive is the current class CDS.  Every historical
#     class is a rival.  Separation is measured only in learned Q support, while
#     native spectral evidence and whole-CDS pair risk *prioritize* spectrally
#     credible old/new conflicts.  P therefore cannot invent a semantic boundary
#     on its own and no classifier weight participates in this loss.
#     """
#     if not isinstance(seen_cds, ClassDiscriminativeStructureState):
#         raise TypeError("seen_cds must be ClassDiscriminativeStructureState")
#     q = normalize_ordered_decision_features(torch.as_tensor(current_features))
#     p = torch.as_tensor(
#         physical_spectral_evidence, device=q.device, dtype=q.dtype
#     ).detach()
#     labels = torch.as_tensor(labels_global, device=q.device, dtype=torch.long).flatten()
#     if labels.shape != (q.size(0),):
#         raise ValueError("labels_global must be [N]")
#     if p.ndim != 3 or p.size(0) != q.size(0) or p.size(2) != 1:
#         raise ValueError("physical_spectral_evidence must be [N,B,1]")
#     if not bool(torch.isfinite(p).all()):
#         raise ValueError("physical_spectral_evidence contains NaN or Inf")

#     classes = seen_cds.classes
#     ids = classes.class_ids.to(device=q.device, dtype=torch.long).flatten()
#     count = int(ids.numel())
#     old_count = int(historical_class_count)
#     if count < 2 or old_count <= 0 or old_count >= count:
#         raise ValueError("historical_class_count must split old and current classes")
#     if classes.learned_identity.shape[1:] != q.shape[1:]:
#         raise ValueError("current Q and learned CDS use different coordinates")
#     if classes.physical_identity.shape[1:] != p.shape[1:]:
#         raise ValueError("physical evidence and physical CDS use different coordinates")

#     matches = labels.unsqueeze(1).eq(ids.unsqueeze(0))
#     if not bool(matches.sum(dim=1).eq(1).all()):
#         raise ValueError("a current label is absent from seen CDS")
#     target_index = matches.to(torch.long).argmax(dim=1)
#     if bool((target_index < old_count).any()):
#         raise ValueError("cross-phase separation expects current-class labels only")

#     _, _, learned_support = ordered_cds_support_components(
#         q,
#         classes.learned_identity.to(device=q.device, dtype=q.dtype).detach(),
#         classes.learned_variation.to(device=q.device, dtype=q.dtype).detach(),
#     )
#     with torch.no_grad():
#         _, _, physical_support = ordered_cds_support_components(
#             p,
#             classes.physical_identity.to(device=q.device, dtype=q.dtype).detach(),
#             classes.physical_variation.to(device=q.device, dtype=q.dtype).detach(),
#         )
#         effective_risk, _learned_risk, _physical_risk = whole_cds_pair_risk(seen_cds)
#         effective_risk = effective_risk.to(device=q.device, dtype=q.dtype)

#     rows = torch.arange(q.size(0), device=q.device)
#     positive_learned = learned_support[rows, target_index]
#     positive_physical = physical_support[rows, target_index]
#     rival_learned = learned_support[:, :old_count]
#     rival_physical = physical_support[:, :old_count]

#     margin = positive_learned.unsqueeze(1) - rival_learned
#     pair_risk = effective_risk.index_select(0, target_index)[:, :old_count]
#     # Sample-specific spectral ambiguity strengthens only a relation already
#     # judged relevant in learned+physical class structure.
#     spectral_confusability = torch.sqrt(
#         positive_physical.unsqueeze(1).clamp_min(0.0)
#         * rival_physical.clamp_min(0.0)
#     ).clamp(0.0, 1.0)
#     pair_weight = 1.0 + pair_risk * spectral_confusability

#     scale = torch.as_tensor(logit_scale, device=q.device, dtype=q.dtype).detach()
#     if scale.ndim != 0 or not bool(torch.isfinite(scale)) or not bool(scale > 0):
#         raise ValueError("logit_scale must be a finite positive scalar")
#     pair_loss = F.softplus(-scale * margin)
#     per_sample = (pair_weight * pair_loss).sum(dim=1) / pair_weight.sum(dim=1).clamp_min(
#         torch.finfo(q.dtype).eps
#     )
#     loss = _weighted_mean(per_sample, sample_weights)

#     values = (
#         loss,
#         positive_learned.mean(),
#         rival_learned.mean(),
#         margin.mean(),
#         margin.min(),
#         spectral_confusability.mean(),
#         pair_risk.mean(),
#         margin.le(0.0).to(q.dtype).mean(),
#     )
#     if not all(bool(torch.isfinite(value)) for value in values):
#         raise RuntimeError("cross-phase CDS separation became NaN or Inf")
#     return CDSSeparationObjective(
#         loss=loss,
#         learned_positive_support_mean=positive_learned.mean(),
#         learned_historical_rival_support_mean=rival_learned.mean(),
#         learned_support_margin_mean=margin.mean(),
#         learned_support_margin_min=margin.min(),
#         physical_confusability_mean=spectral_confusability.mean(),
#         effective_pair_risk_mean=pair_risk.mean(),
#         violation_fraction=margin.le(0.0).to(q.dtype).mean(),
#     )


# __all__ = [
#     "ClassificationObjective",
#     "SupervisedContrastiveObjective",
#     "LogitDistillationObjective",
#     "FeatureDistillationObjective",
#     "CDSPreservationObjective",
#     "CDSSeparationObjective",
#     "WholeCDSPICObjective",
#     "classification_objective",
#     "supervised_contrastive_q_loss",
#     "logit_distillation_loss",
#     "ordered_feature_distillation_loss",
#     "historical_cds_preservation_loss",
#     "whole_cds_pic_loss",
#     "cross_phase_cds_separation_loss",
# ]


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
