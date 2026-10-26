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
    spectral_conditioned_learned_support,
    native_spectral_variation_support,
    whole_cds_pair_risk,
)

Tensor = torch.Tensor


@dataclass(frozen=True)
class ClassificationObjective:
    loss: Tensor
    accuracy: Tensor
    targets_local: Tensor


@dataclass(frozen=True)
class BalancedAllSeenClassificationObjective:
    loss: Tensor
    historical_loss: Tensor
    current_loss: Tensor
    per_class_loss: Tensor
    historical_accuracy: Tensor
    current_accuracy: Tensor


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
    # One transient score per current sample, obtained ONLY from the frozen
    # teacher and committed historical CDS. Never persisted between phases.
    teacher_historical_relevance: Tensor


@dataclass(frozen=True)
class CDSSeparationObjective:
    loss: Tensor
    positive_decision_score_mean: Tensor
    historical_rival_decision_score_mean: Tensor
    decision_margin_mean: Tensor
    decision_margin_min: Tensor
    physical_confusability_mean: Tensor
    effective_pair_risk_mean: Tensor
    violation_fraction: Tensor

    # Backward-compatible aliases for existing history/report consumers.
    @property
    def learned_positive_support_mean(self) -> Tensor:
        return self.positive_decision_score_mean

    @property
    def learned_historical_rival_support_mean(self) -> Tensor:
        return self.historical_rival_decision_score_mean

    @property
    def learned_support_margin_mean(self) -> Tensor:
        return self.decision_margin_mean

    @property
    def learned_support_margin_min(self) -> Tensor:
        return self.decision_margin_min


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
    unbiased_sample_weighting: bool = False,
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
    if unbiased_sample_weighting and sample_weights is not None:
        # The incoming class-frequency weights are N/(C*N_c). Uniform
        # minibatch sampling therefore needs mean(w_i CE_i), NOT divide by
        # stochastic sum(w_i), to estimate equal-class CE without bias.
        weights = torch.as_tensor(
            sample_weights, device=scores.device, dtype=scores.dtype
        )
        if (
            weights.shape != per_sample.shape
            or not bool(torch.isfinite(weights).all())
            or bool((weights < 0).any())
        ):
            raise ValueError("invalid fixed global class-frequency importance weights")
        loss = (per_sample * weights).mean()
    else:
        loss = _weighted_mean(per_sample, sample_weights)
    accuracy = scores.argmax(dim=1).eq(targets).to(scores.dtype).mean()
    if not bool(torch.isfinite(loss)):
        raise RuntimeError("classification loss became NaN or Inf")
    return ClassificationObjective(loss=loss, accuracy=accuracy, targets_local=targets)


def class_balanced_all_seen_classification_objective(
    *,
    historical_logits: Tensor,
    historical_labels_global: Tensor,
    current_logits: Tensor,
    current_labels_global: Tensor,
    class_ids: Tensor,
    historical_class_ids: Tensor,
    current_class_ids: Tensor,
    pair_risk: Tensor | None = None,
    current_sample_weights: Tensor | None = None,
) -> BalancedAllSeenClassificationObjective:
    """Class-balanced positive supervision for the single all-seen head.

    Historical logits come from detached Whole-CDS distribution supports; current
    logits come from *real current HSI features detached from the backbone*.  Both
    use the same deployed cosine head. Historical CE is averaged per historical
    class exactly. Current CE uses the phase class-frequency weights when supplied,
    giving an unbiased class-balanced estimate even when one stochastic minibatch
    does not contain every incoming class. The two group losses are then combined
    in proportion to the number of classes in each group, which approximates one
    equal vote per seen class without requiring synthetic current support.
    """
    ids = torch.as_tensor(class_ids)
    old_ids = torch.as_tensor(historical_class_ids, device=ids.device, dtype=torch.long)
    new_ids = torch.as_tensor(current_class_ids, device=ids.device, dtype=torch.long)
    old_labels = torch.as_tensor(
        historical_labels_global, device=ids.device, dtype=torch.long
    ).flatten()
    new_labels = torch.as_tensor(
        current_labels_global, device=ids.device, dtype=torch.long
    ).flatten()
    old_scores = torch.as_tensor(historical_logits)
    new_scores = torch.as_tensor(current_logits)
    if ids.ndim != 1 or ids.numel() < 2 or ids.dtype != torch.long:
        raise ValueError("class_ids must be torch.long [C]")
    if old_scores.ndim != 2 or new_scores.ndim != 2:
        raise ValueError("historical/current logits must be rank-two")
    if old_scores.size(1) != ids.numel() or new_scores.size(1) != ids.numel():
        raise ValueError(
            "historical/current logits must use identical all-seen classes"
        )
    if (
        old_scores.size(0) != old_labels.numel()
        or new_scores.size(0) != new_labels.numel()
    ):
        raise ValueError("classifier logits and labels are misaligned")
    if not bool(torch.isin(old_labels, old_ids).all()):
        raise ValueError("historical support labels are outside historical classes")
    if not bool(torch.isin(new_labels, new_ids).all()):
        raise ValueError("current labels are outside current classes")

    old_targets = _targets_local(old_labels, ids)
    new_targets = _targets_local(new_labels, ids)

    if pair_risk is not None:
        risk = torch.as_tensor(pair_risk, device=ids.device, dtype=old_scores.dtype)
        if risk.shape != (ids.numel(), ids.numel()):
            raise ValueError("pair_risk must be [C,C] aligned with class_ids")
        if not bool(torch.isfinite(risk).all()) or bool((risk < 0).any()):
            raise ValueError("pair_risk must be finite and non-negative")
        risk = risk.clamp(0.0, 1.0)
        # High learned+spectral rivalry strengthens that rival in the softmax
        # denominator.  The target-class offset is forced to zero.
        old_offset = torch.log1p(risk.index_select(0, old_targets))
        new_offset = torch.log1p(risk.index_select(0, new_targets))
        old_offset = old_offset.scatter(1, old_targets[:, None], 0.0)
        new_offset = new_offset.scatter(1, new_targets[:, None], 0.0)
        old_scores = old_scores + old_offset
        new_scores = new_scores + new_offset

    old_sample = F.cross_entropy(old_scores, old_targets, reduction="none")
    new_sample = F.cross_entropy(new_scores, new_targets, reduction="none")

    # Historical supports are explicitly class-balanced: the distribution
    # realizer contributes the same number of supports for every old class.
    old_class_loss = torch.stack(
        [old_sample[old_labels.eq(cid)].mean() for cid in old_ids]
    )
    if bool(torch.isnan(old_class_loss).any()):
        raise RuntimeError("historical Whole-CDS support misses a stored class")
    historical_loss = old_class_loss.mean()

    # Current positives must remain real HSI. A stochastic minibatch may omit an
    # incoming class, so do not manufacture a current pseudo-support merely to make
    # a per-class stack rectangular. The phase-frequency weights already computed
    # by the trainer give each incoming class equal expected mass across the epoch.
    if current_sample_weights is None:
        current_loss = new_sample.mean()
    else:
        weights = torch.as_tensor(
            current_sample_weights, device=new_sample.device, dtype=new_sample.dtype
        ).flatten()
        if (
            weights.shape != new_sample.shape
            or not bool(torch.isfinite(weights).all())
            or bool((weights < 0).any())
            or not bool(weights.sum() > 0)
        ):
            raise ValueError(
                "current_sample_weights must be finite non-negative [N] with positive mass"
            )
        current_loss = (new_sample * weights).mean()

    old_count = int(old_ids.numel())
    new_count = int(new_ids.numel())
    loss = (
        float(old_count) * historical_loss + float(new_count) * current_loss
    ) / float(old_count + new_count)

    # Reporting keeps exact historical per-class losses plus one current-group
    # scalar; the optimization itself remains class-count balanced as above.
    per_class = torch.cat((old_class_loss, current_loss.reshape(1)), dim=0)
    historical_accuracy = (
        old_scores.argmax(dim=1).eq(old_targets).to(old_scores.dtype).mean()
    )
    current_accuracy = (
        new_scores.argmax(dim=1).eq(new_targets).to(new_scores.dtype).mean()
    )
    if not all(
        bool(torch.isfinite(v))
        for v in (
            loss,
            historical_loss,
            current_loss,
            historical_accuracy,
            current_accuracy,
        )
    ):
        raise RuntimeError("balanced all-seen classification became NaN or Inf")
    return BalancedAllSeenClassificationObjective(
        loss=loss,
        historical_loss=historical_loss,
        current_loss=current_loss,
        per_class_loss=per_class,
        historical_accuracy=historical_accuracy,
        current_accuracy=current_accuracy,
    )


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
    historical_relevance: Tensor | None = None,
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
    # Every current sample is constrained by the frozen previous model. Old-like
    # samples receive up to twice the normalized KD importance, but a sample with
    # zero old-class support STILL retains the full baseline KD weight. This is
    # not an old-feature replay or a new model head.
    if historical_relevance is not None:
        relevance = torch.as_tensor(
            historical_relevance, device=current.device, dtype=current.dtype
        ).detach()
        if relevance.shape != (current.size(0),) or not bool(
            torch.isfinite(relevance).all()
        ):
            raise ValueError("historical_relevance must be finite [N]")
        if bool((relevance < 0).any()) or bool((relevance > 1).any()):
            raise ValueError("historical_relevance must be in [0,1]")
        importance = 1.0 + relevance
        if sample_weights is None:
            effective_weights = importance
        else:
            effective_weights = (
                torch.as_tensor(
                    sample_weights, device=current.device, dtype=current.dtype
                )
                * importance
            )
        loss = _weighted_mean(per_sample, effective_weights)
    else:
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


def _finite_support_sqrt(value: Tensor) -> Tensor:
    """Preserve zero support with a finite zero-gradient convention at zero.

    Conditional compatibility can underflow to zero. Unprotected sqrt(0) then
    creates 0*infinity NaNs even when the preservation error itself is zero.
    Normal positive values retain their square root; only subnormals are floored.
    """
    return torch.where(
        value > 0,
        value.clamp_min(torch.finfo(value.dtype).tiny).sqrt(),
        torch.zeros_like(value),
    )


def _whole_cds_support_components(
    *,
    learned_features: Tensor,
    physical_features: Tensor,
    structure: ClassStructureBatch,
    moments: dict[str, Tensor] | None = None,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    q = torch.as_tensor(learned_features)
    p = torch.as_tensor(physical_features, device=q.device, dtype=q.dtype).detach()
    if q.ndim != 3 or p.ndim != 3 or q.size(0) != p.size(0):
        raise ValueError("learned/physical features must be aligned rank-three batches")

    i_q = structure.learned_identity.to(device=q.device, dtype=q.dtype).detach()
    v_q = structure.learned_variation.to(device=q.device, dtype=q.dtype).detach()
    i_p = structure.physical_identity.to(device=q.device, dtype=q.dtype).detach()
    v_p = structure.physical_variation.to(device=q.device, dtype=q.dtype).detach()

    # The class has ONE learned distribution. A center compatibility provides
    # identity, and the low-rank covariance provides within-class variation;
    # inactive V_Q/A_Q legacy fields are not optimized or double-counted.
    flat = normalize_ordered_decision_features(q).flatten(1)
    mean = i_q.flatten(1)
    distance = torch.cdist(flat.unsqueeze(0), mean.unsqueeze(0)).squeeze(0)
    learned_abs = (1.0 / (1.0 + distance)).clamp(0, 1)
    unified = {
        "mean": mean,
        "basis": structure.learned_covariance_basis.to(
            device=q.device, dtype=q.dtype
        ).detach(),
        "eigenvalues": structure.learned_covariance_eigenvalues.to(
            device=q.device, dtype=q.dtype
        ).detach(),
        "residual_variance": structure.learned_covariance_residual.to(
            device=q.device, dtype=q.dtype
        ).detach(),
    }
    learned_rel = spectral_conditioned_learned_support(q, unified)
    learned_full = _finite_support_sqrt(learned_abs * learned_rel)
    physical_abs, physical_rel, physical_full = native_spectral_variation_support(
        p,
        structure.physical_identity,
        structure.physical_variance,
        structure.physical_delta_variance,
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
    unbiased_sample_weighting: bool = False,
) -> Tensor:
    """Class-balanced CE on absolute, relation and complete Whole-CDS support."""
    eps = torch.finfo(whole_complete.dtype).eps
    losses = []
    for support in (whole_absolute, whole_relation, whole_complete):
        per_sample = F.cross_entropy(
            support.clamp_min(eps).log(), targets_local, reduction="none"
        )
        if unbiased_sample_weighting and sample_weights is not None:
            weights = torch.as_tensor(
                sample_weights, device=per_sample.device, dtype=per_sample.dtype
            )
            if (
                weights.shape != per_sample.shape
                or not bool(torch.isfinite(weights).all())
                or bool((weights < 0).any())
            ):
                raise ValueError("support CE weights must be finite nonnegative [N]")
            losses.append((per_sample * weights).mean())
        else:
            losses.append(_weighted_mean(per_sample, sample_weights))
    return sum(losses) / float(len(losses))


def whole_cds_support_classification_loss(
    *,
    learned_features: Tensor,
    physical_features: Tensor,
    labels_global: Tensor,
    structure: ClassStructureBatch,
    sample_weights: Tensor | None = None,
    unbiased_sample_weighting: bool = False,
) -> Tensor:
    """Classify real samples by fused learned and native-spectral CDS support."""
    q = torch.as_tensor(learned_features)
    ids = structure.class_ids.to(device=q.device, dtype=torch.long)
    labels = torch.as_tensor(labels_global, device=q.device, dtype=torch.long)
    if labels.shape != (q.size(0),):
        raise ValueError("labels_global must match the feature batch")
    targets = _targets_local(labels, ids)
    absolute, relation, complete, _, _, _ = _whole_cds_support_components(
        learned_features=q,
        physical_features=physical_features,
        structure=structure,
    )
    return _whole_cds_cross_entropy(
        whole_absolute=absolute,
        whole_relation=relation,
        whole_complete=complete,
        targets_local=targets,
        sample_weights=sample_weights,
        unbiased_sample_weighting=unbiased_sample_weighting,
    )


def historical_cds_preservation_loss(
    *,
    reference_features: Tensor,
    current_features: Tensor,
    physical_spectral_evidence: Tensor,
    reference_historical_structure: ClassStructureBatch,
    current_historical_structure: ClassStructureBatch,
    reference_moments: dict[str, Tensor] | None = None,
    current_moments: dict[str, Tensor] | None = None,
    historical_pair_risk: Tensor | None = None,
) -> CDSPreservationObjective:
    """Preserve historical Whole-CDS intra/inter structure on real current HSI.

    Each feature tensor must match its supplied structure coordinate. The active
    trainer maps the frozen teacher and its statistics into one fixed epoch frame.
    This observed-current alignment does not certify unavailable old-class drift.
    The optimized supports are *direct*
    Whole-CDS scores: learned mean and low-rank covariance is geometrically fused
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
            moments=reference_moments,
        )
    student_abs, student_rel, student_full, _, _, _ = _whole_cds_support_components(
        learned_features=current,
        physical_features=physical,
        structure=cur_struct,
        moments=current_moments,
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
        if historical_pair_risk is None:
            # Standalone calls still use the exact same CDS pair-risk function.
            from models.geometry_bank import (
                ClassDiscriminativeStructureState,
                derive_class_relations,
            )

            temporary_cds = ClassDiscriminativeStructureState(
                classes=ref_struct,
                relations=derive_class_relations(ref_struct),
                spectral_band_positions=ref_struct.physical_identity.new_zeros(
                    ref_struct.physical_identity.size(1)
                ),
            )
            pair_risk = whole_cds_pair_risk(temporary_cds)[0]
        else:
            pair_risk = torch.as_tensor(
                historical_pair_risk, device=current.device, dtype=current.dtype
            )
            if pair_risk.shape != (class_count, class_count) or not bool(
                torch.isfinite(pair_risk).all()
            ):
                raise ValueError("historical class-pair risk must be finite [C,C]")
        weight = (1.0 + pair_risk[upper]).detach()
        inter_error = (student_pair - teacher_pair).abs()[:, upper]
        inter_loss = (inter_error * weight[None]).sum() / (
            weight.sum() * inter_error.size(0)
        ).clamp_min(1e-12)
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
        teacher_historical_relevance=teacher_full.detach()
        .max(dim=1)
        .values.clamp(0.0, 1.0),
    )


def _historical_distribution_negative_logits(
    flat: Tensor,
    features: Tensor,
    labels: Tensor,
    old_ids: Tensor,
    scale: Tensor,
    ordered_shape: tuple[int, int],
) -> tuple[Tensor, Tensor]:
    """One log-mean-exp negative per old class; gradient belongs only to flat Q."""
    if features is None or labels is None:
        raise ValueError(
            "distribution/mean contrast requires historical features and labels"
        )
    draws = torch.as_tensor(features, device=flat.device, dtype=flat.dtype).detach()
    labels = torch.as_tensor(labels, device=flat.device).detach()
    if (
        draws.ndim != 3
        or draws.size(0) < 1
        or tuple(draws.shape[1:]) != ordered_shape
        or not bool(torch.isfinite(draws).all())
    ):
        raise ValueError(
            "historical negative features must be finite non-empty [S,K,D] in the same ordered coordinates as Q"
        )
    if labels.dtype != torch.long or labels.shape != (draws.size(0),):
        raise ValueError("historical negative labels must be [S] torch.long")
    if not torch.equal(labels.unique(sorted=True), old_ids.sort().values):
        raise ValueError(
            "historical negatives must cover exactly the historical classes"
        )
    draws = draws.flatten(1)
    norms = draws.norm(dim=1, keepdim=True)
    if not bool((norms > torch.finfo(draws.dtype).eps).all()):
        raise ValueError("historical negative feature has no stable cosine direction")
    draws = draws / norms
    similarities = (flat @ draws.T).clamp(-1.0, 1.0)
    logits, means = [], []
    for cid in old_ids:
        mask = labels.eq(cid)
        # Coverage was checked above; each class receives one denominator term,
        # irrespective of how many draws represent it.
        selected = similarities[:, mask]
        logits.append(
            torch.logsumexp(scale * selected, dim=1)
            - selected.new_tensor(selected.size(1)).log()
        )
        means.append(selected.mean(dim=1))
    return torch.stack(logits, dim=1), torch.stack(means, dim=1)


def whole_cds_contrastive_loss(
    *,
    current_features: Tensor,
    labels_global: Tensor,
    physical_spectral_evidence: Tensor,
    seen_cds: ClassDiscriminativeStructureState,
    historical_class_count: int,
    logit_scale: Tensor,
    sample_weights: Tensor | None = None,
    historical_support=None,
    class_pair_risk: Tensor | None = None,
    historical_classifier=None,
    historical_negative_mode: str = "classifier",
    historical_negative_features: Tensor | None = None,
    historical_negative_labels: Tensor | None = None,
) -> WholeCDSContrastiveObjective:
    """Unified continual contrastive separation in the single ordered Q space.

    Base phase: the historical set is empty, so this is ordinary supervised
    contrastive learning over real current samples.

    Incremental phase: classifier mode retains the detached deployed old W
    directions. Distribution/mean modes consume explicit detached historical Q
    and contribute one log-mean-exp negative per old class. Distribution mode
    therefore uses intra-class spread in the optimized denominator. Existing
    spectral confusability and class rivalry still weight each class negative.
    Base callers retain the real-only supervised contrastive objective.
    """
    if historical_negative_mode not in ("classifier", "mean", "distribution"):
        raise ValueError(
            "historical_negative_mode must be classifier, mean or distribution"
        )
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
    # Use the authoritative class structure from geometry_bank.py.
    # All tensor transfers are performed in _whole_cds_support_components().
    structure = classes
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
        if historical_support is None:
            raise RuntimeError(
                "historical Whole-CDS contrastive requires the CDS moment reader"
            )
        old_ids = tuple(map(int, ids[:old_count].tolist()))
        # Supports above already use the caller's explicit historical frame.
        # Never replace them with the canonical bank's unmapped coordinates.
        old_whole = whole_full[:, :old_count]
        if historical_negative_mode == "classifier":
            if (
                historical_negative_features is not None
                or historical_negative_labels is not None
            ):
                raise ValueError(
                    "classifier contrast must not silently ignore historical draws"
                )
            historical_logits_raw = historical_support.historical_negative_logits(
                q,
                old_ids,
                seen_cds,
                scale,
                classifier=historical_classifier,
            )
            historical_similarity = historical_logits_raw / scale
        else:
            historical_logits_raw, historical_similarity = (
                _historical_distribution_negative_logits(
                    flat,
                    historical_negative_features,
                    historical_negative_labels,
                    ids[:old_count],
                    scale,
                    tuple(q.shape[1:]),
                )
            )
        with torch.no_grad():
            if class_pair_risk is None:
                effective_risk = whole_cds_pair_risk(seen_cds)[0]
            else:
                effective_risk = torch.as_tensor(class_pair_risk)
                if (
                    effective_risk.shape != (count, count)
                    or not bool(torch.isfinite(effective_risk).all())
                    or bool((effective_risk < 0).any())
                    or bool((effective_risk > 1).any())
                ):
                    raise ValueError(
                        "class_pair_risk must be bounded [C,C] CDS rivalry"
                    )
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
        # Every historical class remains a negative. Detached structural weights
        # lie in [1,2]; support weights never remove a competing old class.
        historical_logits = (
            historical_logits_raw + historical_weight.clamp_min(eps).log()
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

    if old_count > 0 and bool((~valid_anchor).any()):
        # No same-class pair? Use the REAL class's detached current CDS
        # identity as an anchor, not an invented historical pseudo-example.
        # This keeps historical negatives effective for rare HSI classes.
        missing = ~valid_anchor
        centers = F.normalize(
            structure.learned_identity.to(q).flatten(1).detach(),
            dim=1,
            eps=eps,
        )
        own_center = centers.index_select(0, target_index[missing])
        positive_logit = (flat[missing] * own_center).sum(1) * scale
        negative_denom = log_denom[missing]
        fallback = torch.logaddexp(negative_denom, positive_logit) - positive_logit
        if sample_weights is None:
            # Average over ALL anchor rows: current pairs and CDS fallback.
            if bool(valid_anchor.any()):
                valid_per_anchor = per_anchor[valid_anchor]
                loss = (valid_per_anchor.sum() + fallback.sum()) / n
            else:
                loss = fallback.mean()
        else:
            weight = torch.as_tensor(sample_weights, device=q.device, dtype=q.dtype)
            if weight.shape != (n,) or not bool(torch.isfinite(weight).all()):
                raise ValueError("contrastive sample weights must be finite [N]")
            per_row = q.new_zeros(n)
            if bool(valid_anchor.any()):
                per_row[valid_anchor] = per_anchor[valid_anchor]
            per_row[missing] = fallback
            loss = _weighted_mean(per_row, weight)

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



@torch.no_grad()
def deterministic_class_audit_indices(labels_global: Tensor, *, max_per_class: int = 48) -> Tensor:
    """Capped class-stratified audit rows, using only observed label order."""
    y = torch.as_tensor(labels_global, dtype=torch.long).flatten()
    if not y.numel() or isinstance(max_per_class, bool) or not isinstance(max_per_class, int) or max_per_class <= 0:
        raise ValueError("invalid audit class sampling request")
    groups = []
    for cls in y.unique(sorted=True):
        locations = y.eq(cls).nonzero(as_tuple=True)[0]
        if locations.numel() > max_per_class:
            positions = torch.linspace(
                0, locations.numel() - 1, max_per_class,
                device=locations.device,
            ).round().long()
            locations = locations.index_select(0, positions)
        groups.append(locations)
    return torch.cat(groups, dim=0)

@torch.no_grad()
def feature_overlap_audit(
    current_features: Tensor,
    labels_global: Tensor,
    *,
    historical_means: Tensor | None = None,
) -> dict[str, float | int | str | None]:
    """Read-only cosine overlap: REAL current pairs and optional MODELED old means.

    This is not an old-class accuracy estimate. No gradient, sampling, or model
    state is modified. Missing real positives/negatives are returned as None.
    """
    q = torch.as_tensor(current_features)
    labels = torch.as_tensor(labels_global, device=q.device, dtype=torch.long).flatten()
    if q.ndim != 3 or q.shape[0] != labels.numel() or q.shape[0] == 0:
        raise ValueError("current Q must be non-empty [N,K,D], labels [N]")
    if not bool(torch.isfinite(q).all()):
        raise ValueError("Q contains nonfinite values")
    z = F.normalize(q.detach().reshape(q.size(0), -1), dim=1)
    sim = (z @ z.T).clamp(-1.0, 1.0)
    same = labels[:, None].eq(labels[None, :])
    self_mask = torch.eye(len(labels), dtype=torch.bool, device=q.device)
    positives = same & ~self_mask
    negatives = ~same
    valid_pos = positives.any(dim=1)
    valid_neg = negatives.any(dim=1)
    eligible = valid_pos & valid_neg
    positive_count = positives.sum(dim=1).clamp_min(1)
    positive_mean = (sim * positives).sum(dim=1) / positive_count
    strongest_neg = sim.masked_fill(~negatives, -torch.inf).max(dim=1).values

    def mean_or_none(values: Tensor):
        return float(values.mean().item()) if values.numel() else None

    report: dict[str, float | int | str | None] = {
        "sample_count": int(z.size(0)),
        "real_positive_anchor_fraction": float(valid_pos.float().mean().item()),
        "real_negative_anchor_fraction": float(valid_neg.float().mean().item()),
        "real_pair_eligible_fraction": float(eligible.float().mean().item()),
        "real_positive_cosine": mean_or_none(positive_mean[eligible]),
        "hard_current_negative_cosine": mean_or_none(strongest_neg[valid_neg]),
        "real_positive_vs_hard_negative_gap": mean_or_none((positive_mean - strongest_neg)[eligible]),
        "real_hard_negative_violation_fraction": mean_or_none((strongest_neg >= positive_mean).float()[eligible]),
        "historical_mean_max_cosine": None,
        "historical_support_kind": "none",
        "evaluation_scope": "current_real_minibatch_and_optional_modeled_old_means",
    }
    if historical_means is not None:
        old = torch.as_tensor(historical_means, device=q.device, dtype=q.dtype).detach()
        if old.ndim != 3 or old.shape[1:] != q.shape[1:] or old.size(0) == 0:
            raise ValueError("historical_means must be non-empty [C_old,K,D] in Q coordinates")
        if not bool(torch.isfinite(old).all()):
            raise ValueError("historical_means contains nonfinite values")
        old_unit = F.normalize(old.reshape(old.size(0), -1), dim=1)
        report["historical_mean_max_cosine"] = float(
            (z @ old_unit.T).clamp(-1.0, 1.0).max(dim=1).values.mean().item()
        )
        report["historical_support_kind"] = "modeled_historical_class_means_not_real_old_samples"
    return report


def fixed_q_classifier_probe(
    train_features: Tensor,
    train_labels_global: Tensor,
    validation_features: Tensor,
    validation_labels_global: Tensor,
    *,
    class_ids,
    deployed_weights: Tensor,
    logit_scale: Tensor,
    steps: int = 60,
) -> dict[str, float | int | str | bool]:
    """Fit throwaway current-only W to FIXED Q; report only current validation.

    Never writes the deployed W or CDS bank, never accesses old examples, and
    never selects/restores a checkpoint. Full-batch class-balanced CE; no RNG.
    """
    ids = torch.as_tensor(class_ids, dtype=torch.long, device=train_features.device).flatten()
    if ids.numel() < 2 or ids.unique().numel() != ids.numel():
        raise ValueError("probe needs at least two unique current class IDs")
    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1:
        raise ValueError("probe steps must be a positive integer")
    train = torch.as_tensor(train_features).detach()
    val = torch.as_tensor(validation_features, device=train.device, dtype=train.dtype).detach()
    yl = torch.as_tensor(train_labels_global, device=train.device, dtype=torch.long).flatten()
    yv = torch.as_tensor(validation_labels_global, device=train.device, dtype=torch.long).flatten()
    if train.ndim != 3 or val.ndim != 3 or train.shape[1:] != val.shape[1:] or train.size(0) != yl.numel() or val.size(0) != yv.numel() or not train.size(0) or not val.size(0):
        raise ValueError("probe features and labels must align [N,K,D]")
    if not bool(torch.isfinite(train).all()) or not bool(torch.isfinite(val).all()):
        raise ValueError("probe received nonfinite Q")
    if not bool(torch.isin(yl, ids).all()) or not bool(torch.isin(yv, ids).all()):
        raise ValueError("probe must use real incoming-class labels only")
    mapping_train = yl[:, None].eq(ids).long().argmax(1)
    mapping_val = yv[:, None].eq(ids).long().argmax(1)
    counts = torch.bincount(mapping_train, minlength=ids.numel())
    if bool((counts == 0).any()):
        raise ValueError("each current class needs training examples")
    val_counts = torch.bincount(mapping_val, minlength=ids.numel())
    if bool((val_counts == 0).any()):
        return {"status": "skipped_validation_missing_current_classes", "used_for_selection": False,
                "training_count": int(len(yl)), "validation_count": int(len(yv))}
    f_train = F.normalize(train.flatten(1), p=2, dim=1)
    f_val = F.normalize(val.flatten(1), p=2, dim=1)
    fixed = torch.as_tensor(deployed_weights, device=train.device, dtype=train.dtype).detach()
    if fixed.shape != (ids.numel(), f_train.size(1)):
        raise ValueError("deployed_weights must be [C_new, K*D]")
    scale = torch.as_tensor(logit_scale, device=train.device, dtype=train.dtype).detach()
    if scale.ndim or not bool(torch.isfinite(scale)) or not bool(scale > 0):
        raise ValueError("invalid classifier scale")
    # Initialize from observed current TRAIN means without consuming RNG.
    prototypes = torch.stack([f_train[mapping_train == k].mean(0) for k in range(len(ids))])
    initial = F.normalize(prototypes, dim=1)
    w = torch.nn.Parameter(initial.clone())
    optimizer = torch.optim.Adam([w], lr=0.1, weight_decay=0.0)
    weights = len(yl) / (len(ids) * counts.index_select(0, mapping_train).float())
    with torch.enable_grad():
        for _ in range(steps):
            optimizer.zero_grad(set_to_none=True)
            logits = scale * (f_train @ F.normalize(w, dim=1).T)
            ce = F.cross_entropy(logits, mapping_train, reduction="none")
            loss = (ce * weights).mean()
            if not bool(torch.isfinite(loss)):
                raise RuntimeError("nonfinite fixed-Q probe loss")
            loss.backward()
            optimizer.step()
    with torch.no_grad():
        def ba(model_weights: Tensor):
            predictions = (f_val @ F.normalize(model_weights, dim=1).T).argmax(1)
            return float(torch.stack([
                predictions[mapping_val == i].eq(mapping_val[mapping_val == i]).float().mean()
                for i in range(ids.numel())
            ]).mean().item())
        deployed_ba = ba(fixed)
        trained_ba = ba(w.detach())
    return {
        "status": "evaluated_current_validation_only",
        "used_for_selection": False,
        "changes_deployed_model": False,
        "training_count": int(len(yl)),
        "validation_count": int(len(yv)),
        "steps": int(steps),
        "deployed_new_only_validation_ba": deployed_ba,
        "probe_new_only_validation_ba": trained_ba,
        "probe_minus_deployed_ba": trained_ba - deployed_ba,
        "scope": "current_train_fit_current_validation_only_no_old_examples",
    }

# Backward-compatible name; active trainers use the continual name below.
def whole_cds_pic_loss(**kwargs) -> WholeCDSContrastiveObjective:
    return whole_cds_contrastive_loss(**kwargs)


def cross_phase_cds_separation_loss(
    *,
    current_cosine_scores: Tensor,
    class_ids: Tensor,
    labels_global: Tensor,
    physical_spectral_evidence: Tensor,
    seen_cds: ClassDiscriminativeStructureState,
    historical_class_count: int,
    sample_weights: Tensor | None = None,
) -> CDSSeparationObjective:
    """Risk-weighted cross-phase separation in the deployed cosine decision space.

    Historical information must constrain the same decision geometry used at
    inference.  The previous implementation measured a CDS-support margin, which
    could be positive even when the deployed cosine classifier still assigned
    current samples to historical classes.

    This objective uses the actual pairwise cosine decision margin

        cos(Q, W_y) - cos(Q, W_old)

    for every real current sample against every historical rival. Whole-CDS remains
    the historical knowledge source: learned+physical pair risk and sample-specific
    spectral ambiguity prioritize dangerous old/new conflicts. No extra margin,
    temperature, phase bias, synthetic feature, or prototype classifier is added.

    The caller detaches all classifier rows on this representation path. Gradients
    therefore flow to current Q only. Classifier-row learning is supplied by the
    separate balanced historical-positive + real-current-positive branch.
    """
    if not isinstance(seen_cds, ClassDiscriminativeStructureState):
        raise TypeError("seen_cds must be ClassDiscriminativeStructureState")

    scores = torch.as_tensor(current_cosine_scores)
    ids = torch.as_tensor(class_ids, device=scores.device, dtype=torch.long).flatten()
    labels = torch.as_tensor(
        labels_global, device=scores.device, dtype=torch.long
    ).flatten()
    p = torch.as_tensor(
        physical_spectral_evidence, device=scores.device, dtype=scores.dtype
    ).detach()

    if (
        scores.ndim != 2
        or scores.size(0) == 0
        or scores.size(1) != ids.numel()
        or not bool(torch.isfinite(scores).all())
    ):
        raise ValueError(
            "current_cosine_scores must be finite [N,C] aligned with class_ids"
        )
    if labels.shape != (scores.size(0),):
        raise ValueError("labels_global must be [N]")
    if ids.numel() < 2 or int(torch.unique(ids).numel()) != int(ids.numel()):
        raise ValueError("class_ids must contain at least two unique classes")
    if p.ndim != 3 or p.size(0) != scores.size(0) or p.size(2) != 1:
        raise ValueError("physical_spectral_evidence must be [N,B,1]")
    if not bool(torch.isfinite(p).all()):
        raise ValueError("physical_spectral_evidence contains NaN or Inf")

    classes = seen_cds.classes
    stored_ids = classes.class_ids.to(device=scores.device, dtype=torch.long).flatten()
    if not torch.equal(ids, stored_ids):
        raise ValueError(
            "classifier class_ids and seen CDS class_ids must match exactly"
        )

    count = int(ids.numel())
    old_count = int(historical_class_count)
    if old_count <= 0 or old_count >= count:
        raise ValueError("historical_class_count must split old and current classes")
    if classes.physical_identity.shape[1:] != p.shape[1:]:
        raise ValueError("physical evidence and physical CDS use different coordinates")

    matches = labels.unsqueeze(1).eq(ids.unsqueeze(0))
    if not bool(matches.sum(dim=1).eq(1).all()):
        raise ValueError("a current label is absent from seen CDS")
    target_index = matches.to(torch.long).argmax(dim=1)
    if bool((target_index < old_count).any()):
        raise ValueError("cross-phase separation expects current-class labels only")

    rows = torch.arange(scores.size(0), device=scores.device)
    positive_score = scores[rows, target_index]
    historical_rival_score = scores[:, :old_count]
    decision_margin = positive_score.unsqueeze(1) - historical_rival_score

    with torch.no_grad():
        _, _, physical_support = ordered_cds_support_components(
            p,
            classes.physical_identity.to(
                device=scores.device, dtype=scores.dtype
            ).detach(),
            classes.physical_variation.to(
                device=scores.device, dtype=scores.dtype
            ).detach(),
        )
        effective_risk, _learned_risk, _physical_risk = whole_cds_pair_risk(seen_cds)
        effective_risk = effective_risk.to(device=scores.device, dtype=scores.dtype)

        positive_physical = physical_support[rows, target_index]
        rival_physical = physical_support[:, :old_count]
        spectral_confusability = torch.sqrt(
            positive_physical.unsqueeze(1).clamp_min(0.0)
            * rival_physical.clamp_min(0.0)
        ).clamp(0.0, 1.0)
        pair_risk = effective_risk.index_select(0, target_index)[:, :old_count]
        # Keep every historical rival active. Whole-CDS/spectral evidence only
        # prioritizes conflicts; it does not define a second decision boundary.
        pair_weight = 1.0 + pair_risk * spectral_confusability

    # Match the deployed argmax failure event.  Classification fails when *any*
    # historical rival beats the current target, therefore averaging all pairwise
    # violations can hide one catastrophic rival.  Optimize the worst risk-weighted
    # violation per sample instead.
    pair_violation = pair_weight * F.relu(-decision_margin)
    per_sample, hardest_index = pair_violation.max(dim=1)
    loss = _weighted_mean(per_sample, sample_weights)

    hardest_rival_score = historical_rival_score.gather(
        1, hardest_index.unsqueeze(1)
    ).squeeze(1)
    hardest_margin = decision_margin.gather(1, hardest_index.unsqueeze(1)).squeeze(1)
    sample_violation = decision_margin.le(0.0).any(dim=1).to(scores.dtype)

    values = (
        loss,
        positive_score.mean(),
        hardest_rival_score.mean(),
        hardest_margin.mean(),
        hardest_margin.min(),
        spectral_confusability.mean(),
        pair_risk.mean(),
        sample_violation.mean(),
    )
    if not all(bool(torch.isfinite(value)) for value in values):
        raise RuntimeError("cross-phase decision separation became NaN or Inf")

    return CDSSeparationObjective(
        loss=loss,
        positive_decision_score_mean=positive_score.mean(),
        historical_rival_decision_score_mean=hardest_rival_score.mean(),
        decision_margin_mean=hardest_margin.mean(),
        decision_margin_min=hardest_margin.min(),
        physical_confusability_mean=spectral_confusability.mean(),
        effective_pair_risk_mean=pair_risk.mean(),
        violation_fraction=sample_violation.mean(),
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
    "whole_cds_support_classification_loss",
    "whole_cds_pic_loss",
    "cross_phase_cds_separation_loss",
    "BalancedAllSeenClassificationObjective",
    "class_balanced_all_seen_classification_objective",
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
#     spectral_conditioned_learned_support,
#     native_spectral_variation_support,
#     whole_cds_pair_risk,
# )

# Tensor = torch.Tensor


# @dataclass(frozen=True)
# class ClassificationObjective:
#     loss: Tensor
#     accuracy: Tensor
#     targets_local: Tensor


# @dataclass(frozen=True)
# class BalancedAllSeenClassificationObjective:
#     loss: Tensor
#     historical_loss: Tensor
#     current_loss: Tensor
#     per_class_loss: Tensor
#     historical_accuracy: Tensor
#     current_accuracy: Tensor


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
#     # One transient score per current sample, obtained ONLY from the frozen
#     # teacher and committed historical CDS. Never persisted between phases.
#     teacher_historical_relevance: Tensor


# @dataclass(frozen=True)
# class CDSSeparationObjective:
#     loss: Tensor
#     positive_decision_score_mean: Tensor
#     historical_rival_decision_score_mean: Tensor
#     decision_margin_mean: Tensor
#     decision_margin_min: Tensor
#     physical_confusability_mean: Tensor
#     effective_pair_risk_mean: Tensor
#     violation_fraction: Tensor

#     # Backward-compatible aliases for existing history/report consumers.
#     @property
#     def learned_positive_support_mean(self) -> Tensor:
#         return self.positive_decision_score_mean

#     @property
#     def learned_historical_rival_support_mean(self) -> Tensor:
#         return self.historical_rival_decision_score_mean

#     @property
#     def learned_support_margin_mean(self) -> Tensor:
#         return self.decision_margin_mean

#     @property
#     def learned_support_margin_min(self) -> Tensor:
#         return self.decision_margin_min


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
#     if (
#         weights.shape != values.shape
#         or weights.device != values.device
#         or weights.dtype != values.dtype
#     ):
#         raise ValueError("sample_weights must match per-sample loss")
#     if (
#         not bool(torch.isfinite(weights).all())
#         or bool((weights < 0).any())
#         or not bool(weights.sum() > 0)
#     ):
#         raise ValueError(
#             "sample_weights must be finite, non-negative and have positive mass"
#         )
#     return (values * weights).sum() / weights.sum()


# def _weighted_matrix_mean(values: Tensor, weights: Tensor) -> Tensor:
#     value = torch.as_tensor(values)
#     weight = torch.as_tensor(weights)
#     if (
#         value.shape != weight.shape
#         or value.device != weight.device
#         or value.dtype != weight.dtype
#     ):
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
#     unbiased_sample_weighting: bool = False,
# ) -> ClassificationObjective:
#     scores = torch.as_tensor(logits)
#     ids = torch.as_tensor(class_ids)
#     if scores.ndim != 2 or scores.size(0) == 0 or scores.size(1) < 2:
#         raise ValueError("logits must be non-empty [N,C], C>=2")
#     if (
#         ids.shape != (scores.size(1),)
#         or ids.dtype != torch.long
#         or ids.device != scores.device
#     ):
#         raise ValueError("class_ids must match logits")
#     if not bool(torch.isfinite(scores).all()):
#         raise ValueError("logits contain NaN or Inf")
#     targets = _targets_local(labels_global, ids)
#     per_sample = F.cross_entropy(scores, targets, reduction="none")
#     if unbiased_sample_weighting and sample_weights is not None:
#         # The incoming class-frequency weights are N/(C*N_c). Uniform
#         # minibatch sampling therefore needs mean(w_i CE_i), NOT divide by
#         # stochastic sum(w_i), to estimate equal-class CE without bias.
#         weights = torch.as_tensor(
#             sample_weights, device=scores.device, dtype=scores.dtype
#         )
#         if (
#             weights.shape != per_sample.shape
#             or not bool(torch.isfinite(weights).all())
#             or bool((weights < 0).any())
#         ):
#             raise ValueError("invalid fixed global class-frequency importance weights")
#         loss = (per_sample * weights).mean()
#     else:
#         loss = _weighted_mean(per_sample, sample_weights)
#     accuracy = scores.argmax(dim=1).eq(targets).to(scores.dtype).mean()
#     if not bool(torch.isfinite(loss)):
#         raise RuntimeError("classification loss became NaN or Inf")
#     return ClassificationObjective(loss=loss, accuracy=accuracy, targets_local=targets)


# def class_balanced_all_seen_classification_objective(
#     *,
#     historical_logits: Tensor,
#     historical_labels_global: Tensor,
#     current_logits: Tensor,
#     current_labels_global: Tensor,
#     class_ids: Tensor,
#     historical_class_ids: Tensor,
#     current_class_ids: Tensor,
#     pair_risk: Tensor | None = None,
#     current_sample_weights: Tensor | None = None,
# ) -> BalancedAllSeenClassificationObjective:
#     """Class-balanced positive supervision for the single all-seen head.

#     Historical logits come from detached Whole-CDS distribution supports; current
#     logits come from *real current HSI features detached from the backbone*.  Both
#     use the same deployed cosine head. Historical CE is averaged per historical
#     class exactly. Current CE uses the phase class-frequency weights when supplied,
#     giving an unbiased class-balanced estimate even when one stochastic minibatch
#     does not contain every incoming class. The two group losses are then combined
#     in proportion to the number of classes in each group, which approximates one
#     equal vote per seen class without requiring synthetic current support.
#     """
#     ids = torch.as_tensor(class_ids)
#     old_ids = torch.as_tensor(historical_class_ids, device=ids.device, dtype=torch.long)
#     new_ids = torch.as_tensor(current_class_ids, device=ids.device, dtype=torch.long)
#     old_labels = torch.as_tensor(
#         historical_labels_global, device=ids.device, dtype=torch.long
#     ).flatten()
#     new_labels = torch.as_tensor(
#         current_labels_global, device=ids.device, dtype=torch.long
#     ).flatten()
#     old_scores = torch.as_tensor(historical_logits)
#     new_scores = torch.as_tensor(current_logits)
#     if ids.ndim != 1 or ids.numel() < 2 or ids.dtype != torch.long:
#         raise ValueError("class_ids must be torch.long [C]")
#     if old_scores.ndim != 2 or new_scores.ndim != 2:
#         raise ValueError("historical/current logits must be rank-two")
#     if old_scores.size(1) != ids.numel() or new_scores.size(1) != ids.numel():
#         raise ValueError(
#             "historical/current logits must use identical all-seen classes"
#         )
#     if (
#         old_scores.size(0) != old_labels.numel()
#         or new_scores.size(0) != new_labels.numel()
#     ):
#         raise ValueError("classifier logits and labels are misaligned")
#     if not bool(torch.isin(old_labels, old_ids).all()):
#         raise ValueError("historical support labels are outside historical classes")
#     if not bool(torch.isin(new_labels, new_ids).all()):
#         raise ValueError("current labels are outside current classes")

#     old_targets = _targets_local(old_labels, ids)
#     new_targets = _targets_local(new_labels, ids)

#     if pair_risk is not None:
#         risk = torch.as_tensor(pair_risk, device=ids.device, dtype=old_scores.dtype)
#         if risk.shape != (ids.numel(), ids.numel()):
#             raise ValueError("pair_risk must be [C,C] aligned with class_ids")
#         if not bool(torch.isfinite(risk).all()) or bool((risk < 0).any()):
#             raise ValueError("pair_risk must be finite and non-negative")
#         risk = risk.clamp(0.0, 1.0)
#         # High learned+spectral rivalry strengthens that rival in the softmax
#         # denominator.  The target-class offset is forced to zero.
#         old_offset = torch.log1p(risk.index_select(0, old_targets))
#         new_offset = torch.log1p(risk.index_select(0, new_targets))
#         old_offset = old_offset.scatter(1, old_targets[:, None], 0.0)
#         new_offset = new_offset.scatter(1, new_targets[:, None], 0.0)
#         old_scores = old_scores + old_offset
#         new_scores = new_scores + new_offset

#     old_sample = F.cross_entropy(old_scores, old_targets, reduction="none")
#     new_sample = F.cross_entropy(new_scores, new_targets, reduction="none")

#     # Historical supports are explicitly class-balanced: the distribution
#     # realizer contributes the same number of supports for every old class.
#     old_class_loss = torch.stack(
#         [old_sample[old_labels.eq(cid)].mean() for cid in old_ids]
#     )
#     if bool(torch.isnan(old_class_loss).any()):
#         raise RuntimeError("historical Whole-CDS support misses a stored class")
#     historical_loss = old_class_loss.mean()

#     # Current positives must remain real HSI. A stochastic minibatch may omit an
#     # incoming class, so do not manufacture a current pseudo-support merely to make
#     # a per-class stack rectangular. The phase-frequency weights already computed
#     # by the trainer give each incoming class equal expected mass across the epoch.
#     if current_sample_weights is None:
#         current_loss = new_sample.mean()
#     else:
#         weights = torch.as_tensor(
#             current_sample_weights, device=new_sample.device, dtype=new_sample.dtype
#         ).flatten()
#         if (
#             weights.shape != new_sample.shape
#             or not bool(torch.isfinite(weights).all())
#             or bool((weights < 0).any())
#             or not bool(weights.sum() > 0)
#         ):
#             raise ValueError(
#                 "current_sample_weights must be finite non-negative [N] with positive mass"
#             )
#         current_loss = (new_sample * weights).mean()

#     old_count = int(old_ids.numel())
#     new_count = int(new_ids.numel())
#     loss = (
#         float(old_count) * historical_loss + float(new_count) * current_loss
#     ) / float(old_count + new_count)

#     # Reporting keeps exact historical per-class losses plus one current-group
#     # scalar; the optimization itself remains class-count balanced as above.
#     per_class = torch.cat((old_class_loss, current_loss.reshape(1)), dim=0)
#     historical_accuracy = (
#         old_scores.argmax(dim=1).eq(old_targets).to(old_scores.dtype).mean()
#     )
#     current_accuracy = (
#         new_scores.argmax(dim=1).eq(new_targets).to(new_scores.dtype).mean()
#     )
#     if not all(
#         bool(torch.isfinite(v))
#         for v in (
#             loss,
#             historical_loss,
#             current_loss,
#             historical_accuracy,
#             current_accuracy,
#         )
#     ):
#         raise RuntimeError("balanced all-seen classification became NaN or Inf")
#     return BalancedAllSeenClassificationObjective(
#         loss=loss,
#         historical_loss=historical_loss,
#         current_loss=current_loss,
#         per_class_loss=per_class,
#         historical_accuracy=historical_accuracy,
#         current_accuracy=current_accuracy,
#     )


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
#     if (
#         not torch.is_floating_point(q)
#         or q.is_complex()
#         or not bool(torch.isfinite(q).all())
#     ):
#         raise ValueError("ordered_features must be finite real floating point")
#     if (
#         labels.shape != (q.size(0),)
#         or labels.dtype != torch.long
#         or labels.device != q.device
#     ):
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
#         per_anchor = -(log_probability * positive_mask.to(dtype=q.dtype)).sum(
#             dim=1
#         ) / positive_count.clamp_min(1).to(dtype=q.dtype)
#         valid_loss = per_anchor[valid_anchor]
#         if sample_weights is None:
#             loss = valid_loss.mean()
#         else:
#             weights = torch.as_tensor(sample_weights)
#             if (
#                 weights.shape != (n,)
#                 or weights.device != q.device
#                 or weights.dtype != q.dtype
#             ):
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
#     if (
#         teacher.shape != student.shape
#         or teacher.device != student.device
#         or teacher.dtype != student.dtype
#     ):
#         raise ValueError("teacher_old_logits must match student_old_logits")
#     if not bool(torch.isfinite(student).all()) or not bool(
#         torch.isfinite(teacher).all()
#     ):
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
#     if not all(
#         bool(torch.isfinite(value))
#         for value in (loss, teacher_entropy, student_entropy)
#     ):
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
#     historical_relevance: Tensor | None = None,
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
#     # Every current sample is constrained by the frozen previous model. Old-like
#     # samples receive up to twice the normalized KD importance, but a sample with
#     # zero old-class support STILL retains the full baseline KD weight. This is
#     # not an old-feature replay or a new model head.
#     if historical_relevance is not None:
#         relevance = torch.as_tensor(
#             historical_relevance, device=current.device, dtype=current.dtype
#         ).detach()
#         if relevance.shape != (current.size(0),) or not bool(
#             torch.isfinite(relevance).all()
#         ):
#             raise ValueError("historical_relevance must be finite [N]")
#         if bool((relevance < 0).any()) or bool((relevance > 1).any()):
#             raise ValueError("historical_relevance must be in [0,1]")
#         importance = 1.0 + relevance
#         if sample_weights is None:
#             effective_weights = importance
#         else:
#             effective_weights = (
#                 torch.as_tensor(
#                     sample_weights, device=current.device, dtype=current.dtype
#                 )
#                 * importance
#             )
#         loss = _weighted_mean(per_sample, effective_weights)
#     else:
#         loss = _weighted_mean(per_sample, sample_weights)
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


# def _finite_support_sqrt(value: Tensor) -> Tensor:
#     """Preserve zero support with a finite zero-gradient convention at zero.

#     Conditional compatibility can underflow to zero. Unprotected sqrt(0) then
#     creates 0*infinity NaNs even when the preservation error itself is zero.
#     Normal positive values retain their square root; only subnormals are floored.
#     """
#     return torch.where(
#         value > 0,
#         value.clamp_min(torch.finfo(value.dtype).tiny).sqrt(),
#         torch.zeros_like(value),
#     )


# def _whole_cds_support_components(
#     *,
#     learned_features: Tensor,
#     physical_features: Tensor,
#     structure: ClassStructureBatch,
#     moments: dict[str, Tensor] | None = None,
# ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
#     q = torch.as_tensor(learned_features)
#     p = torch.as_tensor(physical_features, device=q.device, dtype=q.dtype).detach()
#     if q.ndim != 3 or p.ndim != 3 or q.size(0) != p.size(0):
#         raise ValueError("learned/physical features must be aligned rank-three batches")

#     i_q = structure.learned_identity.to(device=q.device, dtype=q.dtype).detach()
#     v_q = structure.learned_variation.to(device=q.device, dtype=q.dtype).detach()
#     i_p = structure.physical_identity.to(device=q.device, dtype=q.dtype).detach()
#     v_p = structure.physical_variation.to(device=q.device, dtype=q.dtype).detach()

#     # The class has ONE learned distribution. A center compatibility provides
#     # identity, and the low-rank covariance provides within-class variation;
#     # inactive V_Q/A_Q legacy fields are not optimized or double-counted.
#     flat = normalize_ordered_decision_features(q).flatten(1)
#     mean = i_q.flatten(1)
#     distance = torch.cdist(flat.unsqueeze(0), mean.unsqueeze(0)).squeeze(0)
#     learned_abs = (1.0 / (1.0 + distance)).clamp(0, 1)
#     unified = {
#         "mean": mean,
#         "basis": structure.learned_covariance_basis.to(
#             device=q.device, dtype=q.dtype
#         ).detach(),
#         "eigenvalues": structure.learned_covariance_eigenvalues.to(
#             device=q.device, dtype=q.dtype
#         ).detach(),
#         "residual_variance": structure.learned_covariance_residual.to(
#             device=q.device, dtype=q.dtype
#         ).detach(),
#     }
#     learned_rel = spectral_conditioned_learned_support(q, unified)
#     learned_full = _finite_support_sqrt(learned_abs * learned_rel)
#     physical_abs, physical_rel, physical_full = native_spectral_variation_support(
#         p,
#         structure.physical_identity,
#         structure.physical_variance,
#         structure.physical_delta_variance,
#     )

#     # Q remains the semantic carrier. Physical spectral evidence can strengthen a
#     # learned relation but cannot create class support when learned support is zero.
#     # This makes I_P/V_P causal in training without creating a second classifier.
#     whole_abs = combine_learned_physical_relevance(learned_abs, physical_abs)
#     whole_rel = combine_learned_physical_relevance(learned_rel, physical_rel)
#     whole_full = combine_learned_physical_relevance(learned_full, physical_full)
#     values = (
#         whole_abs,
#         whole_rel,
#         whole_full,
#         physical_abs,
#         physical_rel,
#         physical_full,
#     )
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
#     unbiased_sample_weighting: bool = False,
# ) -> Tensor:
#     """Class-balanced CE on absolute, relation and complete Whole-CDS support."""
#     eps = torch.finfo(whole_complete.dtype).eps
#     losses = []
#     for support in (whole_absolute, whole_relation, whole_complete):
#         per_sample = F.cross_entropy(
#             support.clamp_min(eps).log(), targets_local, reduction="none"
#         )
#         if unbiased_sample_weighting and sample_weights is not None:
#             weights = torch.as_tensor(
#                 sample_weights, device=per_sample.device, dtype=per_sample.dtype
#             )
#             if (
#                 weights.shape != per_sample.shape
#                 or not bool(torch.isfinite(weights).all())
#                 or bool((weights < 0).any())
#             ):
#                 raise ValueError("support CE weights must be finite nonnegative [N]")
#             losses.append((per_sample * weights).mean())
#         else:
#             losses.append(_weighted_mean(per_sample, sample_weights))
#     return sum(losses) / float(len(losses))


# def whole_cds_support_classification_loss(
#     *,
#     learned_features: Tensor,
#     physical_features: Tensor,
#     labels_global: Tensor,
#     structure: ClassStructureBatch,
#     sample_weights: Tensor | None = None,
#     unbiased_sample_weighting: bool = False,
# ) -> Tensor:
#     """Classify real samples by fused learned and native-spectral CDS support."""
#     q = torch.as_tensor(learned_features)
#     ids = structure.class_ids.to(device=q.device, dtype=torch.long)
#     labels = torch.as_tensor(labels_global, device=q.device, dtype=torch.long)
#     if labels.shape != (q.size(0),):
#         raise ValueError("labels_global must match the feature batch")
#     targets = _targets_local(labels, ids)
#     absolute, relation, complete, _, _, _ = _whole_cds_support_components(
#         learned_features=q,
#         physical_features=physical_features,
#         structure=structure,
#     )
#     return _whole_cds_cross_entropy(
#         whole_absolute=absolute,
#         whole_relation=relation,
#         whole_complete=complete,
#         targets_local=targets,
#         sample_weights=sample_weights,
#         unbiased_sample_weighting=unbiased_sample_weighting,
#     )


# def historical_cds_preservation_loss(
#     *,
#     reference_features: Tensor,
#     current_features: Tensor,
#     physical_spectral_evidence: Tensor,
#     reference_historical_structure: ClassStructureBatch,
#     current_historical_structure: ClassStructureBatch,
#     reference_moments: dict[str, Tensor] | None = None,
#     current_moments: dict[str, Tensor] | None = None,
#     historical_pair_risk: Tensor | None = None,
# ) -> CDSPreservationObjective:
#     """Preserve historical Whole-CDS intra/inter structure on real current HSI.

#     Each feature tensor must match its supplied structure coordinate. The active
#     trainer maps the frozen teacher and its statistics into one fixed epoch frame.
#     This observed-current alignment does not certify unavailable old-class drift.
#     The optimized supports are *direct*
#     Whole-CDS scores: learned mean and low-rank covariance is geometrically fused
#     with native spectral structure (I_P,V_P) before intra- and inter-class errors
#     are formed.  Physical CDS therefore participates in the actual preserved
#     class score rather than merely weighting a learned-space loss.
#     """
#     ref_struct = reference_historical_structure
#     cur_struct = current_historical_structure
#     if not isinstance(ref_struct, ClassStructureBatch) or ref_struct.class_count <= 0:
#         raise ValueError("reference_historical_structure must be non-empty")
#     if (
#         not isinstance(cur_struct, ClassStructureBatch)
#         or cur_struct.class_count != ref_struct.class_count
#     ):
#         raise ValueError(
#             "current_historical_structure must match historical class count"
#         )
#     if not torch.equal(ref_struct.class_ids, cur_struct.class_ids):
#         raise ValueError("reference/current historical class IDs disagree")
#     if not torch.equal(ref_struct.physical_identity, cur_struct.physical_identity):
#         raise ValueError("historical I_P must remain phase-stable")
#     if not torch.equal(ref_struct.physical_variation, cur_struct.physical_variation):
#         raise ValueError("historical V_P must remain phase-stable")

#     reference = normalize_ordered_decision_features(
#         torch.as_tensor(reference_features)
#     ).detach()
#     current = normalize_ordered_decision_features(torch.as_tensor(current_features))
#     physical = torch.as_tensor(
#         physical_spectral_evidence, device=current.device, dtype=current.dtype
#     ).detach()
#     if (
#         current.shape != reference.shape
#         or current.device != reference.device
#         or current.dtype != reference.dtype
#     ):
#         raise ValueError("current_features must match reference_features")
#     if (
#         physical.ndim != 3
#         or physical.size(0) != reference.size(0)
#         or physical.size(2) != 1
#     ):
#         raise ValueError("physical_spectral_evidence must be [N,B,1]")

#     with torch.no_grad():
#         (
#             teacher_abs,
#             teacher_rel,
#             teacher_full,
#             physical_abs,
#             physical_rel,
#             physical_full,
#         ) = _whole_cds_support_components(
#             learned_features=reference,
#             physical_features=physical,
#             structure=ref_struct,
#             moments=reference_moments,
#         )
#     student_abs, student_rel, student_full, _, _, _ = _whole_cds_support_components(
#         learned_features=current,
#         physical_features=physical,
#         structure=cur_struct,
#         moments=current_moments,
#     )

#     # Intra-CDS: preserve the class-wise absolute and ordered-relation support.
#     intra_error = 0.5 * (
#         (student_abs - teacher_abs).abs() + (student_rel - teacher_rel).abs()
#     )
#     intra_loss = intra_error.mean()

#     # Inter-CDS: preserve all historical pairwise support differences.
#     class_count = ref_struct.class_count
#     if class_count >= 2:
#         upper = torch.triu(
#             torch.ones(
#                 class_count, class_count, device=current.device, dtype=torch.bool
#             ),
#             diagonal=1,
#         )
#         teacher_pair = teacher_full.unsqueeze(2) - teacher_full.unsqueeze(1)
#         student_pair = student_full.unsqueeze(2) - student_full.unsqueeze(1)
#         if historical_pair_risk is None:
#             # Standalone calls still use the exact same CDS pair-risk function.
#             from models.geometry_bank import (
#                 ClassDiscriminativeStructureState,
#                 derive_class_relations,
#             )

#             temporary_cds = ClassDiscriminativeStructureState(
#                 classes=ref_struct,
#                 relations=derive_class_relations(ref_struct),
#                 spectral_band_positions=ref_struct.physical_identity.new_zeros(
#                     ref_struct.physical_identity.size(1)
#                 ),
#             )
#             pair_risk = whole_cds_pair_risk(temporary_cds)[0]
#         else:
#             pair_risk = torch.as_tensor(
#                 historical_pair_risk, device=current.device, dtype=current.dtype
#             )
#             if pair_risk.shape != (class_count, class_count) or not bool(
#                 torch.isfinite(pair_risk).all()
#             ):
#                 raise ValueError("historical class-pair risk must be finite [C,C]")
#         weight = (1.0 + pair_risk[upper]).detach()
#         inter_error = (student_pair - teacher_pair).abs()[:, upper]
#         inter_loss = (inter_error * weight[None]).sum() / (
#             weight.sum() * inter_error.size(0)
#         ).clamp_min(1e-12)
#     else:
#         inter_loss = current.sum() * 0.0

#     loss = intra_loss + inter_loss
#     values = (
#         loss,
#         intra_loss,
#         inter_loss,
#         physical_abs.mean(),
#         physical_rel.mean(),
#         physical_full.mean(),
#         teacher_abs.mean(),
#         teacher_rel.mean(),
#         student_abs.mean(),
#         student_rel.mean(),
#     )
#     if not all(bool(torch.isfinite(value)) for value in values):
#         raise RuntimeError(
#             "historical Whole-CDS preservation objective became NaN or Inf"
#         )

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
#         teacher_historical_relevance=teacher_full.detach()
#         .max(dim=1)
#         .values.clamp(0.0, 1.0),
#     )


# def _historical_distribution_negative_logits(
#     flat: Tensor,
#     features: Tensor,
#     labels: Tensor,
#     old_ids: Tensor,
#     scale: Tensor,
#     ordered_shape: tuple[int, int],
# ) -> tuple[Tensor, Tensor]:
#     """One log-mean-exp negative per old class; gradient belongs only to flat Q."""
#     if features is None or labels is None:
#         raise ValueError(
#             "distribution/mean contrast requires historical features and labels"
#         )
#     draws = torch.as_tensor(features, device=flat.device, dtype=flat.dtype).detach()
#     labels = torch.as_tensor(labels, device=flat.device).detach()
#     if (
#         draws.ndim != 3
#         or draws.size(0) < 1
#         or tuple(draws.shape[1:]) != ordered_shape
#         or not bool(torch.isfinite(draws).all())
#     ):
#         raise ValueError(
#             "historical negative features must be finite non-empty [S,K,D] in the same ordered coordinates as Q"
#         )
#     if labels.dtype != torch.long or labels.shape != (draws.size(0),):
#         raise ValueError("historical negative labels must be [S] torch.long")
#     if not torch.equal(labels.unique(sorted=True), old_ids.sort().values):
#         raise ValueError(
#             "historical negatives must cover exactly the historical classes"
#         )
#     draws = draws.flatten(1)
#     norms = draws.norm(dim=1, keepdim=True)
#     if not bool((norms > torch.finfo(draws.dtype).eps).all()):
#         raise ValueError("historical negative feature has no stable cosine direction")
#     draws = draws / norms
#     similarities = (flat @ draws.T).clamp(-1.0, 1.0)
#     logits, means = [], []
#     for cid in old_ids:
#         mask = labels.eq(cid)
#         # Coverage was checked above; each class receives one denominator term,
#         # irrespective of how many draws represent it.
#         selected = similarities[:, mask]
#         logits.append(
#             torch.logsumexp(scale * selected, dim=1)
#             - selected.new_tensor(selected.size(1)).log()
#         )
#         means.append(selected.mean(dim=1))
#     return torch.stack(logits, dim=1), torch.stack(means, dim=1)


# def whole_cds_contrastive_loss(
#     *,
#     current_features: Tensor,
#     labels_global: Tensor,
#     physical_spectral_evidence: Tensor,
#     seen_cds: ClassDiscriminativeStructureState,
#     historical_class_count: int,
#     logit_scale: Tensor,
#     sample_weights: Tensor | None = None,
#     historical_support=None,
#     class_pair_risk: Tensor | None = None,
#     historical_classifier=None,
#     historical_negative_mode: str = "classifier",
#     historical_negative_features: Tensor | None = None,
#     historical_negative_labels: Tensor | None = None,
# ) -> WholeCDSContrastiveObjective:
#     """Unified continual contrastive separation in the single ordered Q space.

#     Base phase: the historical set is empty, so this is ordinary supervised
#     contrastive learning over real current samples.

#     Incremental phase: classifier mode retains the detached deployed old W
#     directions. Distribution/mean modes consume explicit detached historical Q
#     and contribute one log-mean-exp negative per old class. Distribution mode
#     therefore uses intra-class spread in the optimized denominator. Existing
#     spectral confusability and class rivalry still weight each class negative.
#     Base callers retain the real-only supervised contrastive objective.
#     """
#     if historical_negative_mode not in ("classifier", "mean", "distribution"):
#         raise ValueError(
#             "historical_negative_mode must be classifier, mean or distribution"
#         )
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
#     # Use the authoritative class structure from geometry_bank.py.
#     # All tensor transfers are performed in _whole_cds_support_components().
#     structure = classes
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
#         if historical_support is None:
#             raise RuntimeError(
#                 "historical Whole-CDS contrastive requires the CDS moment reader"
#             )
#         old_ids = tuple(map(int, ids[:old_count].tolist()))
#         # Supports above already use the caller's explicit historical frame.
#         # Never replace them with the canonical bank's unmapped coordinates.
#         old_whole = whole_full[:, :old_count]
#         if historical_negative_mode == "classifier":
#             if (
#                 historical_negative_features is not None
#                 or historical_negative_labels is not None
#             ):
#                 raise ValueError(
#                     "classifier contrast must not silently ignore historical draws"
#                 )
#             historical_logits_raw = historical_support.historical_negative_logits(
#                 q,
#                 old_ids,
#                 seen_cds,
#                 scale,
#                 classifier=historical_classifier,
#             )
#             historical_similarity = historical_logits_raw / scale
#         else:
#             historical_logits_raw, historical_similarity = (
#                 _historical_distribution_negative_logits(
#                     flat,
#                     historical_negative_features,
#                     historical_negative_labels,
#                     ids[:old_count],
#                     scale,
#                     tuple(q.shape[1:]),
#                 )
#             )
#         with torch.no_grad():
#             if class_pair_risk is None:
#                 effective_risk = whole_cds_pair_risk(seen_cds)[0]
#             else:
#                 effective_risk = torch.as_tensor(class_pair_risk)
#                 if (
#                     effective_risk.shape != (count, count)
#                     or not bool(torch.isfinite(effective_risk).all())
#                     or bool((effective_risk < 0).any())
#                     or bool((effective_risk > 1).any())
#                 ):
#                     raise ValueError(
#                         "class_pair_risk must be bounded [C,C] CDS rivalry"
#                     )
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
#         # Every historical class remains a negative. Detached structural weights
#         # lie in [1,2]; support weights never remove a competing old class.
#         historical_logits = (
#             historical_logits_raw + historical_weight.clamp_min(eps).log()
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
#         per_anchor = -(log_probability * positive_mask.to(dtype=q.dtype)).sum(
#             dim=1
#         ) / positive_count.clamp_min(1).to(dtype=q.dtype)
#         if sample_weights is None:
#             loss = per_anchor[valid_anchor].mean()
#         else:
#             weights = torch.as_tensor(sample_weights)
#             if (
#                 weights.shape != (n,)
#                 or weights.device != q.device
#                 or weights.dtype != q.dtype
#             ):
#                 raise ValueError("sample_weights must be [N] and match Q device/dtype")
#             loss = _weighted_mean(per_anchor[valid_anchor], weights[valid_anchor])
#     else:
#         loss = q.sum() * 0.0

#     if old_count > 0 and bool((~valid_anchor).any()):
#         # No same-class pair? Use the REAL class's detached current CDS
#         # identity as an anchor, not an invented historical pseudo-example.
#         # This keeps historical negatives effective for rare HSI classes.
#         missing = ~valid_anchor
#         centers = F.normalize(
#             structure.learned_identity.to(q).flatten(1).detach(),
#             dim=1,
#             eps=eps,
#         )
#         own_center = centers.index_select(0, target_index[missing])
#         positive_logit = (flat[missing] * own_center).sum(1) * scale
#         negative_denom = log_denom[missing]
#         fallback = torch.logaddexp(negative_denom, positive_logit) - positive_logit
#         if sample_weights is None:
#             # Average over ALL anchor rows: current pairs and CDS fallback.
#             if bool(valid_anchor.any()):
#                 valid_per_anchor = per_anchor[valid_anchor]
#                 loss = (valid_per_anchor.sum() + fallback.sum()) / n
#             else:
#                 loss = fallback.mean()
#         else:
#             weight = torch.as_tensor(sample_weights, device=q.device, dtype=q.dtype)
#             if weight.shape != (n,) or not bool(torch.isfinite(weight).all()):
#                 raise ValueError("contrastive sample weights must be finite [N]")
#             per_row = q.new_zeros(n)
#             if bool(valid_anchor.any()):
#                 per_row[valid_anchor] = per_anchor[valid_anchor]
#             per_row[missing] = fallback
#             loss = _weighted_mean(per_row, weight)

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
#     current_cosine_scores: Tensor,
#     class_ids: Tensor,
#     labels_global: Tensor,
#     physical_spectral_evidence: Tensor,
#     seen_cds: ClassDiscriminativeStructureState,
#     historical_class_count: int,
#     sample_weights: Tensor | None = None,
# ) -> CDSSeparationObjective:
#     """Risk-weighted cross-phase separation in the deployed cosine decision space.

#     Historical information must constrain the same decision geometry used at
#     inference.  The previous implementation measured a CDS-support margin, which
#     could be positive even when the deployed cosine classifier still assigned
#     current samples to historical classes.

#     This objective uses the actual pairwise cosine decision margin

#         cos(Q, W_y) - cos(Q, W_old)

#     for every real current sample against every historical rival. Whole-CDS remains
#     the historical knowledge source: learned+physical pair risk and sample-specific
#     spectral ambiguity prioritize dangerous old/new conflicts. No extra margin,
#     temperature, phase bias, synthetic feature, or prototype classifier is added.

#     The caller detaches all classifier rows on this representation path. Gradients
#     therefore flow to current Q only. Classifier-row learning is supplied by the
#     separate balanced historical-positive + real-current-positive branch.
#     """
#     if not isinstance(seen_cds, ClassDiscriminativeStructureState):
#         raise TypeError("seen_cds must be ClassDiscriminativeStructureState")

#     scores = torch.as_tensor(current_cosine_scores)
#     ids = torch.as_tensor(class_ids, device=scores.device, dtype=torch.long).flatten()
#     labels = torch.as_tensor(
#         labels_global, device=scores.device, dtype=torch.long
#     ).flatten()
#     p = torch.as_tensor(
#         physical_spectral_evidence, device=scores.device, dtype=scores.dtype
#     ).detach()

#     if (
#         scores.ndim != 2
#         or scores.size(0) == 0
#         or scores.size(1) != ids.numel()
#         or not bool(torch.isfinite(scores).all())
#     ):
#         raise ValueError(
#             "current_cosine_scores must be finite [N,C] aligned with class_ids"
#         )
#     if labels.shape != (scores.size(0),):
#         raise ValueError("labels_global must be [N]")
#     if ids.numel() < 2 or int(torch.unique(ids).numel()) != int(ids.numel()):
#         raise ValueError("class_ids must contain at least two unique classes")
#     if p.ndim != 3 or p.size(0) != scores.size(0) or p.size(2) != 1:
#         raise ValueError("physical_spectral_evidence must be [N,B,1]")
#     if not bool(torch.isfinite(p).all()):
#         raise ValueError("physical_spectral_evidence contains NaN or Inf")

#     classes = seen_cds.classes
#     stored_ids = classes.class_ids.to(device=scores.device, dtype=torch.long).flatten()
#     if not torch.equal(ids, stored_ids):
#         raise ValueError(
#             "classifier class_ids and seen CDS class_ids must match exactly"
#         )

#     count = int(ids.numel())
#     old_count = int(historical_class_count)
#     if old_count <= 0 or old_count >= count:
#         raise ValueError("historical_class_count must split old and current classes")
#     if classes.physical_identity.shape[1:] != p.shape[1:]:
#         raise ValueError("physical evidence and physical CDS use different coordinates")

#     matches = labels.unsqueeze(1).eq(ids.unsqueeze(0))
#     if not bool(matches.sum(dim=1).eq(1).all()):
#         raise ValueError("a current label is absent from seen CDS")
#     target_index = matches.to(torch.long).argmax(dim=1)
#     if bool((target_index < old_count).any()):
#         raise ValueError("cross-phase separation expects current-class labels only")

#     rows = torch.arange(scores.size(0), device=scores.device)
#     positive_score = scores[rows, target_index]
#     historical_rival_score = scores[:, :old_count]
#     decision_margin = positive_score.unsqueeze(1) - historical_rival_score

#     with torch.no_grad():
#         _, _, physical_support = ordered_cds_support_components(
#             p,
#             classes.physical_identity.to(
#                 device=scores.device, dtype=scores.dtype
#             ).detach(),
#             classes.physical_variation.to(
#                 device=scores.device, dtype=scores.dtype
#             ).detach(),
#         )
#         effective_risk, _learned_risk, _physical_risk = whole_cds_pair_risk(seen_cds)
#         effective_risk = effective_risk.to(device=scores.device, dtype=scores.dtype)

#         positive_physical = physical_support[rows, target_index]
#         rival_physical = physical_support[:, :old_count]
#         spectral_confusability = torch.sqrt(
#             positive_physical.unsqueeze(1).clamp_min(0.0)
#             * rival_physical.clamp_min(0.0)
#         ).clamp(0.0, 1.0)
#         pair_risk = effective_risk.index_select(0, target_index)[:, :old_count]
#         # Keep every historical rival active. Whole-CDS/spectral evidence only
#         # prioritizes conflicts; it does not define a second decision boundary.
#         pair_weight = 1.0 + pair_risk * spectral_confusability

#     # Match the deployed argmax failure event.  Classification fails when *any*
#     # historical rival beats the current target, therefore averaging all pairwise
#     # violations can hide one catastrophic rival.  Optimize the worst risk-weighted
#     # violation per sample instead.
#     pair_violation = pair_weight * F.relu(-decision_margin)
#     per_sample, hardest_index = pair_violation.max(dim=1)
#     loss = _weighted_mean(per_sample, sample_weights)

#     hardest_rival_score = historical_rival_score.gather(
#         1, hardest_index.unsqueeze(1)
#     ).squeeze(1)
#     hardest_margin = decision_margin.gather(1, hardest_index.unsqueeze(1)).squeeze(1)
#     sample_violation = decision_margin.le(0.0).any(dim=1).to(scores.dtype)

#     values = (
#         loss,
#         positive_score.mean(),
#         hardest_rival_score.mean(),
#         hardest_margin.mean(),
#         hardest_margin.min(),
#         spectral_confusability.mean(),
#         pair_risk.mean(),
#         sample_violation.mean(),
#     )
#     if not all(bool(torch.isfinite(value)) for value in values):
#         raise RuntimeError("cross-phase decision separation became NaN or Inf")

#     return CDSSeparationObjective(
#         loss=loss,
#         positive_decision_score_mean=positive_score.mean(),
#         historical_rival_decision_score_mean=hardest_rival_score.mean(),
#         decision_margin_mean=hardest_margin.mean(),
#         decision_margin_min=hardest_margin.min(),
#         physical_confusability_mean=spectral_confusability.mean(),
#         effective_pair_risk_mean=pair_risk.mean(),
#         violation_fraction=sample_violation.mean(),
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
#     "whole_cds_support_classification_loss",
#     "whole_cds_pic_loss",
#     "cross_phase_cds_separation_loss",
#     "BalancedAllSeenClassificationObjective",
#     "class_balanced_all_seen_classification_objective",
# ]


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
#     spectral_conditioned_learned_support,
#     native_spectral_variation_support,
#     whole_cds_pair_risk,
# )

# Tensor = torch.Tensor


# @dataclass(frozen=True)
# class ClassificationObjective:
#     loss: Tensor
#     accuracy: Tensor
#     targets_local: Tensor


# @dataclass(frozen=True)
# class BalancedAllSeenClassificationObjective:
#     loss: Tensor
#     historical_loss: Tensor
#     current_loss: Tensor
#     per_class_loss: Tensor
#     historical_accuracy: Tensor
#     current_accuracy: Tensor


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
#     # One transient score per current sample, obtained ONLY from the frozen
#     # teacher and committed historical CDS. Never persisted between phases.
#     teacher_historical_relevance: Tensor


# @dataclass(frozen=True)
# class CDSSeparationObjective:
#     loss: Tensor
#     positive_decision_score_mean: Tensor
#     historical_rival_decision_score_mean: Tensor
#     decision_margin_mean: Tensor
#     decision_margin_min: Tensor
#     physical_confusability_mean: Tensor
#     effective_pair_risk_mean: Tensor
#     violation_fraction: Tensor

#     # Backward-compatible aliases for existing history/report consumers.
#     @property
#     def learned_positive_support_mean(self) -> Tensor:
#         return self.positive_decision_score_mean

#     @property
#     def learned_historical_rival_support_mean(self) -> Tensor:
#         return self.historical_rival_decision_score_mean

#     @property
#     def learned_support_margin_mean(self) -> Tensor:
#         return self.decision_margin_mean

#     @property
#     def learned_support_margin_min(self) -> Tensor:
#         return self.decision_margin_min


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
#     if (
#         weights.shape != values.shape
#         or weights.device != values.device
#         or weights.dtype != values.dtype
#     ):
#         raise ValueError("sample_weights must match per-sample loss")
#     if (
#         not bool(torch.isfinite(weights).all())
#         or bool((weights < 0).any())
#         or not bool(weights.sum() > 0)
#     ):
#         raise ValueError(
#             "sample_weights must be finite, non-negative and have positive mass"
#         )
#     return (values * weights).sum() / weights.sum()


# def _weighted_matrix_mean(values: Tensor, weights: Tensor) -> Tensor:
#     value = torch.as_tensor(values)
#     weight = torch.as_tensor(weights)
#     if (
#         value.shape != weight.shape
#         or value.device != weight.device
#         or value.dtype != weight.dtype
#     ):
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
#     unbiased_sample_weighting: bool = False,
# ) -> ClassificationObjective:
#     scores = torch.as_tensor(logits)
#     ids = torch.as_tensor(class_ids)
#     if scores.ndim != 2 or scores.size(0) == 0 or scores.size(1) < 2:
#         raise ValueError("logits must be non-empty [N,C], C>=2")
#     if (
#         ids.shape != (scores.size(1),)
#         or ids.dtype != torch.long
#         or ids.device != scores.device
#     ):
#         raise ValueError("class_ids must match logits")
#     if not bool(torch.isfinite(scores).all()):
#         raise ValueError("logits contain NaN or Inf")
#     targets = _targets_local(labels_global, ids)
#     per_sample = F.cross_entropy(scores, targets, reduction="none")
#     if unbiased_sample_weighting and sample_weights is not None:
#         # The incoming class-frequency weights are N/(C*N_c). Uniform
#         # minibatch sampling therefore needs mean(w_i CE_i), NOT divide by
#         # stochastic sum(w_i), to estimate equal-class CE without bias.
#         weights = torch.as_tensor(
#             sample_weights, device=scores.device, dtype=scores.dtype
#         )
#         if (
#             weights.shape != per_sample.shape
#             or not bool(torch.isfinite(weights).all())
#             or bool((weights < 0).any())
#         ):
#             raise ValueError("invalid fixed global class-frequency importance weights")
#         loss = (per_sample * weights).mean()
#     else:
#         loss = _weighted_mean(per_sample, sample_weights)
#     accuracy = scores.argmax(dim=1).eq(targets).to(scores.dtype).mean()
#     if not bool(torch.isfinite(loss)):
#         raise RuntimeError("classification loss became NaN or Inf")
#     return ClassificationObjective(loss=loss, accuracy=accuracy, targets_local=targets)


# def class_balanced_all_seen_classification_objective(
#     *,
#     historical_logits: Tensor,
#     historical_labels_global: Tensor,
#     current_logits: Tensor,
#     current_labels_global: Tensor,
#     class_ids: Tensor,
#     historical_class_ids: Tensor,
#     current_class_ids: Tensor,
#     pair_risk: Tensor | None = None,
#     current_sample_weights: Tensor | None = None,
# ) -> BalancedAllSeenClassificationObjective:
#     """Class-balanced positive supervision for the single all-seen head.

#     Historical logits come from detached Whole-CDS distribution supports; current
#     logits come from *real current HSI features detached from the backbone*.  Both
#     use the same deployed cosine head. Historical CE is averaged per historical
#     class exactly. Current CE uses the phase class-frequency weights when supplied,
#     giving an unbiased class-balanced estimate even when one stochastic minibatch
#     does not contain every incoming class. The two group losses are then combined
#     in proportion to the number of classes in each group, which approximates one
#     equal vote per seen class without requiring synthetic current support.
#     """
#     ids = torch.as_tensor(class_ids)
#     old_ids = torch.as_tensor(historical_class_ids, device=ids.device, dtype=torch.long)
#     new_ids = torch.as_tensor(current_class_ids, device=ids.device, dtype=torch.long)
#     old_labels = torch.as_tensor(
#         historical_labels_global, device=ids.device, dtype=torch.long
#     ).flatten()
#     new_labels = torch.as_tensor(
#         current_labels_global, device=ids.device, dtype=torch.long
#     ).flatten()
#     old_scores = torch.as_tensor(historical_logits)
#     new_scores = torch.as_tensor(current_logits)
#     if ids.ndim != 1 or ids.numel() < 2 or ids.dtype != torch.long:
#         raise ValueError("class_ids must be torch.long [C]")
#     if old_scores.ndim != 2 or new_scores.ndim != 2:
#         raise ValueError("historical/current logits must be rank-two")
#     if old_scores.size(1) != ids.numel() or new_scores.size(1) != ids.numel():
#         raise ValueError(
#             "historical/current logits must use identical all-seen classes"
#         )
#     if (
#         old_scores.size(0) != old_labels.numel()
#         or new_scores.size(0) != new_labels.numel()
#     ):
#         raise ValueError("classifier logits and labels are misaligned")
#     if not bool(torch.isin(old_labels, old_ids).all()):
#         raise ValueError("historical support labels are outside historical classes")
#     if not bool(torch.isin(new_labels, new_ids).all()):
#         raise ValueError("current labels are outside current classes")

#     old_targets = _targets_local(old_labels, ids)
#     new_targets = _targets_local(new_labels, ids)

#     if pair_risk is not None:
#         risk = torch.as_tensor(pair_risk, device=ids.device, dtype=old_scores.dtype)
#         if risk.shape != (ids.numel(), ids.numel()):
#             raise ValueError("pair_risk must be [C,C] aligned with class_ids")
#         if not bool(torch.isfinite(risk).all()) or bool((risk < 0).any()):
#             raise ValueError("pair_risk must be finite and non-negative")
#         risk = risk.clamp(0.0, 1.0)
#         # High learned+spectral rivalry strengthens that rival in the softmax
#         # denominator.  The target-class offset is forced to zero.
#         old_offset = torch.log1p(risk.index_select(0, old_targets))
#         new_offset = torch.log1p(risk.index_select(0, new_targets))
#         old_offset = old_offset.scatter(1, old_targets[:, None], 0.0)
#         new_offset = new_offset.scatter(1, new_targets[:, None], 0.0)
#         old_scores = old_scores + old_offset
#         new_scores = new_scores + new_offset

#     old_sample = F.cross_entropy(old_scores, old_targets, reduction="none")
#     new_sample = F.cross_entropy(new_scores, new_targets, reduction="none")

#     # Historical supports are explicitly class-balanced: the distribution
#     # realizer contributes the same number of supports for every old class.
#     old_class_loss = torch.stack(
#         [old_sample[old_labels.eq(cid)].mean() for cid in old_ids]
#     )
#     if bool(torch.isnan(old_class_loss).any()):
#         raise RuntimeError("historical Whole-CDS support misses a stored class")
#     historical_loss = old_class_loss.mean()

#     # Current positives must remain real HSI. A stochastic minibatch may omit an
#     # incoming class, so do not manufacture a current pseudo-support merely to make
#     # a per-class stack rectangular. The phase-frequency weights already computed
#     # by the trainer give each incoming class equal expected mass across the epoch.
#     if current_sample_weights is None:
#         current_loss = new_sample.mean()
#     else:
#         weights = torch.as_tensor(
#             current_sample_weights, device=new_sample.device, dtype=new_sample.dtype
#         ).flatten()
#         if (
#             weights.shape != new_sample.shape
#             or not bool(torch.isfinite(weights).all())
#             or bool((weights < 0).any())
#             or not bool(weights.sum() > 0)
#         ):
#             raise ValueError(
#                 "current_sample_weights must be finite non-negative [N] with positive mass"
#             )
#         current_loss = (new_sample * weights).mean()

#     old_count = int(old_ids.numel())
#     new_count = int(new_ids.numel())
#     loss = (
#         float(old_count) * historical_loss + float(new_count) * current_loss
#     ) / float(old_count + new_count)

#     # Reporting keeps exact historical per-class losses plus one current-group
#     # scalar; the optimization itself remains class-count balanced as above.
#     per_class = torch.cat((old_class_loss, current_loss.reshape(1)), dim=0)
#     historical_accuracy = (
#         old_scores.argmax(dim=1).eq(old_targets).to(old_scores.dtype).mean()
#     )
#     current_accuracy = (
#         new_scores.argmax(dim=1).eq(new_targets).to(new_scores.dtype).mean()
#     )
#     if not all(
#         bool(torch.isfinite(v))
#         for v in (
#             loss,
#             historical_loss,
#             current_loss,
#             historical_accuracy,
#             current_accuracy,
#         )
#     ):
#         raise RuntimeError("balanced all-seen classification became NaN or Inf")
#     return BalancedAllSeenClassificationObjective(
#         loss=loss,
#         historical_loss=historical_loss,
#         current_loss=current_loss,
#         per_class_loss=per_class,
#         historical_accuracy=historical_accuracy,
#         current_accuracy=current_accuracy,
#     )


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
#     if (
#         not torch.is_floating_point(q)
#         or q.is_complex()
#         or not bool(torch.isfinite(q).all())
#     ):
#         raise ValueError("ordered_features must be finite real floating point")
#     if (
#         labels.shape != (q.size(0),)
#         or labels.dtype != torch.long
#         or labels.device != q.device
#     ):
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
#         per_anchor = -(log_probability * positive_mask.to(dtype=q.dtype)).sum(
#             dim=1
#         ) / positive_count.clamp_min(1).to(dtype=q.dtype)
#         valid_loss = per_anchor[valid_anchor]
#         if sample_weights is None:
#             loss = valid_loss.mean()
#         else:
#             weights = torch.as_tensor(sample_weights)
#             if (
#                 weights.shape != (n,)
#                 or weights.device != q.device
#                 or weights.dtype != q.dtype
#             ):
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
#     if (
#         teacher.shape != student.shape
#         or teacher.device != student.device
#         or teacher.dtype != student.dtype
#     ):
#         raise ValueError("teacher_old_logits must match student_old_logits")
#     if not bool(torch.isfinite(student).all()) or not bool(
#         torch.isfinite(teacher).all()
#     ):
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
#     if not all(
#         bool(torch.isfinite(value))
#         for value in (loss, teacher_entropy, student_entropy)
#     ):
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
#     historical_relevance: Tensor | None = None,
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
#     # Every current sample is constrained by the frozen previous model. Old-like
#     # samples receive up to twice the normalized KD importance, but a sample with
#     # zero old-class support STILL retains the full baseline KD weight. This is
#     # not an old-feature replay or a new model head.
#     if historical_relevance is not None:
#         relevance = torch.as_tensor(
#             historical_relevance, device=current.device, dtype=current.dtype
#         ).detach()
#         if relevance.shape != (current.size(0),) or not bool(
#             torch.isfinite(relevance).all()
#         ):
#             raise ValueError("historical_relevance must be finite [N]")
#         if bool((relevance < 0).any()) or bool((relevance > 1).any()):
#             raise ValueError("historical_relevance must be in [0,1]")
#         importance = 1.0 + relevance
#         if sample_weights is None:
#             effective_weights = importance
#         else:
#             effective_weights = (
#                 torch.as_tensor(
#                     sample_weights, device=current.device, dtype=current.dtype
#                 )
#                 * importance
#             )
#         loss = _weighted_mean(per_sample, effective_weights)
#     else:
#         loss = _weighted_mean(per_sample, sample_weights)
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


# def _finite_support_sqrt(value: Tensor) -> Tensor:
#     """Preserve zero support with a finite zero-gradient convention at zero.

#     Conditional compatibility can underflow to zero. Unprotected sqrt(0) then
#     creates 0*infinity NaNs even when the preservation error itself is zero.
#     Normal positive values retain their square root; only subnormals are floored.
#     """
#     return torch.where(
#         value > 0,
#         value.clamp_min(torch.finfo(value.dtype).tiny).sqrt(),
#         torch.zeros_like(value),
#     )


# def _whole_cds_support_components(
#     *,
#     learned_features: Tensor,
#     physical_features: Tensor,
#     structure: ClassStructureBatch,
#     moments: dict[str, Tensor] | None = None,
# ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
#     q = torch.as_tensor(learned_features)
#     p = torch.as_tensor(physical_features, device=q.device, dtype=q.dtype).detach()
#     if q.ndim != 3 or p.ndim != 3 or q.size(0) != p.size(0):
#         raise ValueError("learned/physical features must be aligned rank-three batches")

#     i_q = structure.learned_identity.to(device=q.device, dtype=q.dtype).detach()
#     v_q = structure.learned_variation.to(device=q.device, dtype=q.dtype).detach()
#     i_p = structure.physical_identity.to(device=q.device, dtype=q.dtype).detach()
#     v_p = structure.physical_variation.to(device=q.device, dtype=q.dtype).detach()

#     # The class has ONE learned distribution. A center compatibility provides
#     # identity, and the low-rank covariance provides within-class variation;
#     # inactive V_Q/A_Q legacy fields are not optimized or double-counted.
#     flat = normalize_ordered_decision_features(q).flatten(1)
#     mean = i_q.flatten(1)
#     distance = torch.cdist(flat.unsqueeze(0), mean.unsqueeze(0)).squeeze(0)
#     learned_abs = (1.0 / (1.0 + distance)).clamp(0, 1)
#     unified = {
#         "mean": mean,
#         "basis": structure.learned_covariance_basis.to(
#             device=q.device, dtype=q.dtype
#         ).detach(),
#         "eigenvalues": structure.learned_covariance_eigenvalues.to(
#             device=q.device, dtype=q.dtype
#         ).detach(),
#         "residual_variance": structure.learned_covariance_residual.to(
#             device=q.device, dtype=q.dtype
#         ).detach(),
#     }
#     learned_rel = spectral_conditioned_learned_support(q, unified)
#     learned_full = _finite_support_sqrt(learned_abs * learned_rel)
#     physical_abs, physical_rel, physical_full = native_spectral_variation_support(
#         p,
#         structure.physical_identity,
#         structure.physical_variance,
#         structure.physical_delta_variance,
#     )

#     # Q remains the semantic carrier. Physical spectral evidence can strengthen a
#     # learned relation but cannot create class support when learned support is zero.
#     # This makes I_P/V_P causal in training without creating a second classifier.
#     whole_abs = combine_learned_physical_relevance(learned_abs, physical_abs)
#     whole_rel = combine_learned_physical_relevance(learned_rel, physical_rel)
#     whole_full = combine_learned_physical_relevance(learned_full, physical_full)
#     values = (
#         whole_abs,
#         whole_rel,
#         whole_full,
#         physical_abs,
#         physical_rel,
#         physical_full,
#     )
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
#         per_sample = F.cross_entropy(
#             support.clamp_min(eps).log(), targets_local, reduction="none"
#         )
#         losses.append(_weighted_mean(per_sample, sample_weights))
#     return sum(losses) / float(len(losses))


# def whole_cds_support_classification_loss(
#     *,
#     learned_features: Tensor,
#     physical_features: Tensor,
#     labels_global: Tensor,
#     structure: ClassStructureBatch,
#     sample_weights: Tensor | None = None,
# ) -> Tensor:
#     """Classify real samples by fused learned and native-spectral CDS support."""
#     q = torch.as_tensor(learned_features)
#     ids = structure.class_ids.to(device=q.device, dtype=torch.long)
#     labels = torch.as_tensor(labels_global, device=q.device, dtype=torch.long)
#     if labels.shape != (q.size(0),):
#         raise ValueError("labels_global must match the feature batch")
#     targets = _targets_local(labels, ids)
#     absolute, relation, complete, _, _, _ = _whole_cds_support_components(
#         learned_features=q,
#         physical_features=physical_features,
#         structure=structure,
#     )
#     return _whole_cds_cross_entropy(
#         whole_absolute=absolute,
#         whole_relation=relation,
#         whole_complete=complete,
#         targets_local=targets,
#         sample_weights=sample_weights,
#     )


# def historical_cds_preservation_loss(
#     *,
#     reference_features: Tensor,
#     current_features: Tensor,
#     physical_spectral_evidence: Tensor,
#     reference_historical_structure: ClassStructureBatch,
#     current_historical_structure: ClassStructureBatch,
#     reference_moments: dict[str, Tensor] | None = None,
#     current_moments: dict[str, Tensor] | None = None,
#     historical_pair_risk: Tensor | None = None,
# ) -> CDSPreservationObjective:
#     """Preserve historical Whole-CDS intra/inter structure on real current HSI.

#     The teacher is evaluated in the PREVIOUS finalized historical coordinate;
#     the student uses the aligned current-coordinate historical structure.
#     The alignment is an observed-current approximation, not certified old drift.  The optimized supports are *direct*
#     Whole-CDS scores: learned mean and low-rank covariance is geometrically fused
#     with native spectral structure (I_P,V_P) before intra- and inter-class errors
#     are formed.  Physical CDS therefore participates in the actual preserved
#     class score rather than merely weighting a learned-space loss.
#     """
#     ref_struct = reference_historical_structure
#     cur_struct = current_historical_structure
#     if not isinstance(ref_struct, ClassStructureBatch) or ref_struct.class_count <= 0:
#         raise ValueError("reference_historical_structure must be non-empty")
#     if (
#         not isinstance(cur_struct, ClassStructureBatch)
#         or cur_struct.class_count != ref_struct.class_count
#     ):
#         raise ValueError(
#             "current_historical_structure must match historical class count"
#         )
#     if not torch.equal(ref_struct.class_ids, cur_struct.class_ids):
#         raise ValueError("reference/current historical class IDs disagree")
#     if not torch.equal(ref_struct.physical_identity, cur_struct.physical_identity):
#         raise ValueError("historical I_P must remain phase-stable")
#     if not torch.equal(ref_struct.physical_variation, cur_struct.physical_variation):
#         raise ValueError("historical V_P must remain phase-stable")

#     reference = normalize_ordered_decision_features(
#         torch.as_tensor(reference_features)
#     ).detach()
#     current = normalize_ordered_decision_features(torch.as_tensor(current_features))
#     physical = torch.as_tensor(
#         physical_spectral_evidence, device=current.device, dtype=current.dtype
#     ).detach()
#     if (
#         current.shape != reference.shape
#         or current.device != reference.device
#         or current.dtype != reference.dtype
#     ):
#         raise ValueError("current_features must match reference_features")
#     if (
#         physical.ndim != 3
#         or physical.size(0) != reference.size(0)
#         or physical.size(2) != 1
#     ):
#         raise ValueError("physical_spectral_evidence must be [N,B,1]")

#     with torch.no_grad():
#         (
#             teacher_abs,
#             teacher_rel,
#             teacher_full,
#             physical_abs,
#             physical_rel,
#             physical_full,
#         ) = _whole_cds_support_components(
#             learned_features=reference,
#             physical_features=physical,
#             structure=ref_struct,
#             moments=reference_moments,
#         )
#     student_abs, student_rel, student_full, _, _, _ = _whole_cds_support_components(
#         learned_features=current,
#         physical_features=physical,
#         structure=cur_struct,
#         moments=current_moments,
#     )

#     # Intra-CDS: preserve the class-wise absolute and ordered-relation support.
#     intra_error = 0.5 * (
#         (student_abs - teacher_abs).abs() + (student_rel - teacher_rel).abs()
#     )
#     intra_loss = intra_error.mean()

#     # Inter-CDS: preserve all historical pairwise support differences.
#     class_count = ref_struct.class_count
#     if class_count >= 2:
#         upper = torch.triu(
#             torch.ones(
#                 class_count, class_count, device=current.device, dtype=torch.bool
#             ),
#             diagonal=1,
#         )
#         teacher_pair = teacher_full.unsqueeze(2) - teacher_full.unsqueeze(1)
#         student_pair = student_full.unsqueeze(2) - student_full.unsqueeze(1)
#         if historical_pair_risk is None:
#             # Standalone calls still use the exact same CDS pair-risk function.
#             from models.geometry_bank import (
#                 ClassDiscriminativeStructureState,
#                 derive_class_relations,
#             )

#             temporary_cds = ClassDiscriminativeStructureState(
#                 classes=ref_struct,
#                 relations=derive_class_relations(ref_struct),
#                 spectral_band_positions=ref_struct.physical_identity.new_zeros(
#                     ref_struct.physical_identity.size(1)
#                 ),
#             )
#             pair_risk = whole_cds_pair_risk(temporary_cds)[0]
#         else:
#             pair_risk = torch.as_tensor(
#                 historical_pair_risk, device=current.device, dtype=current.dtype
#             )
#             if pair_risk.shape != (class_count, class_count) or not bool(
#                 torch.isfinite(pair_risk).all()
#             ):
#                 raise ValueError("historical class-pair risk must be finite [C,C]")
#         weight = (1.0 + pair_risk[upper]).detach()
#         inter_error = (student_pair - teacher_pair).abs()[:, upper]
#         inter_loss = (inter_error * weight[None]).sum() / (
#             weight.sum() * inter_error.size(0)
#         ).clamp_min(1e-12)
#     else:
#         inter_loss = current.sum() * 0.0

#     loss = intra_loss + inter_loss
#     values = (
#         loss,
#         intra_loss,
#         inter_loss,
#         physical_abs.mean(),
#         physical_rel.mean(),
#         physical_full.mean(),
#         teacher_abs.mean(),
#         teacher_rel.mean(),
#         student_abs.mean(),
#         student_rel.mean(),
#     )
#     if not all(bool(torch.isfinite(value)) for value in values):
#         raise RuntimeError(
#             "historical Whole-CDS preservation objective became NaN or Inf"
#         )

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
#         teacher_historical_relevance=teacher_full.detach()
#         .max(dim=1)
#         .values.clamp(0.0, 1.0),
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
#     historical_support=None,
#     class_pair_risk: Tensor | None = None,
# ) -> WholeCDSContrastiveObjective:
#     """Unified continual contrastive separation in the single ordered Q space.

#     Base phase: the historical set is empty, so this is ordinary supervised
#     contrastive learning over real current samples.

#     Incremental phase: extend the denominator with conditional-mean historical
#     logits. Physical confusability and class-pair rivalry weight those negatives.
#     Complete covariance/ordered-support scores below are diagnostics, rather
#     than additional optimized logits. Covariance also acts in CDS preservation
#     and in the separately selected historical classifier objective.
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
#     # Use the authoritative class structure from geometry_bank.py.
#     # All tensor transfers are performed in _whole_cds_support_components().
#     structure = classes
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
#         if historical_support is None:
#             raise RuntimeError(
#                 "historical Whole-CDS contrastive requires the CDS moment reader"
#             )
#         old_ids = tuple(map(int, ids[:old_count].tolist()))
#         historical_classes = historical_support.cds_bank.class_structures(old_ids)
#         old_moments = historical_support.cds_bank.class_moments(
#             old_ids, device=q.device, dtype=q.dtype
#         )
#         _, _, old_whole_complete, _, _, _ = _whole_cds_support_components(
#             learned_features=q,
#             physical_features=physical,
#             structure=historical_classes,
#             moments=old_moments,
#         )
#         old_whole = old_whole_complete
#         whole_full = torch.cat((old_whole, whole_full[:, old_count:]), dim=1)
#         positive_whole = whole_full[rows, target_index]
#         # The same committed non-Gaussian support used by the historical head.
#         # Compatibility is a feature loss carrier; deployment remains cosine.
#         if historical_support is None:
#             raise RuntimeError(
#                 "Phase-1 contrastive requires historical CDS distribution support"
#             )
#         historical_logits_raw = historical_support.historical_negative_logits(
#             q, tuple(int(v) for v in ids[:old_count]), seen_cds, scale
#         )
#         historical_similarity = historical_logits_raw / scale
#         with torch.no_grad():
#             if class_pair_risk is None:
#                 effective_risk = whole_cds_pair_risk(seen_cds)[0]
#             else:
#                 effective_risk = torch.as_tensor(class_pair_risk)
#                 if (
#                     effective_risk.shape != (count, count)
#                     or not bool(torch.isfinite(effective_risk).all())
#                     or bool((effective_risk < 0).any())
#                     or bool((effective_risk > 1).any())
#                 ):
#                     raise ValueError(
#                         "class_pair_risk must be bounded [C,C] CDS rivalry"
#                     )
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
#         # The historical negative has a cosine-mean carrier and a separate
#         # dimensionless *Whole-CDS likelihood factor*. The latter depends on
#         # ordered V_Q, conditional covariance and native I_P/V_P, is never
#         # mistaken for a cosine, and has a real gradient to current Q.
#         historical_logits = (
#             historical_logits_raw
#             + old_whole.clamp_min(eps).log()
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
#         per_anchor = -(log_probability * positive_mask.to(dtype=q.dtype)).sum(
#             dim=1
#         ) / positive_count.clamp_min(1).to(dtype=q.dtype)
#         if sample_weights is None:
#             loss = per_anchor[valid_anchor].mean()
#         else:
#             weights = torch.as_tensor(sample_weights)
#             if (
#                 weights.shape != (n,)
#                 or weights.device != q.device
#                 or weights.dtype != q.dtype
#             ):
#                 raise ValueError("sample_weights must be [N] and match Q device/dtype")
#             loss = _weighted_mean(per_anchor[valid_anchor], weights[valid_anchor])
#     else:
#         loss = q.sum() * 0.0

#     if old_count > 0 and bool((~valid_anchor).any()):
#         # No same-class pair? Use the REAL class's detached current CDS
#         # identity as an anchor, not an invented historical pseudo-example.
#         # This keeps historical negatives effective for rare HSI classes.
#         missing = ~valid_anchor
#         centers = F.normalize(
#             structure.learned_identity.to(q).flatten(1).detach(),
#             dim=1,
#             eps=eps,
#         )
#         own_center = centers.index_select(0, target_index[missing])
#         positive_logit = (flat[missing] * own_center).sum(1) * scale
#         negative_denom = log_denom[missing]
#         fallback = torch.logaddexp(negative_denom, positive_logit) - positive_logit
#         if sample_weights is None:
#             # Average over ALL anchor rows: current pairs and CDS fallback.
#             if bool(valid_anchor.any()):
#                 valid_per_anchor = per_anchor[valid_anchor]
#                 loss = (valid_per_anchor.sum() + fallback.sum()) / n
#             else:
#                 loss = fallback.mean()
#         else:
#             weight = torch.as_tensor(sample_weights, device=q.device, dtype=q.dtype)
#             if weight.shape != (n,) or not bool(torch.isfinite(weight).all()):
#                 raise ValueError("contrastive sample weights must be finite [N]")
#             per_row = q.new_zeros(n)
#             if bool(valid_anchor.any()):
#                 per_row[valid_anchor] = per_anchor[valid_anchor]
#             per_row[missing] = fallback
#             loss = _weighted_mean(per_row, weight)

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
#     current_cosine_scores: Tensor,
#     class_ids: Tensor,
#     labels_global: Tensor,
#     physical_spectral_evidence: Tensor,
#     seen_cds: ClassDiscriminativeStructureState,
#     historical_class_count: int,
#     sample_weights: Tensor | None = None,
# ) -> CDSSeparationObjective:
#     """Risk-weighted cross-phase separation in the deployed cosine decision space.

#     Historical information must constrain the same decision geometry used at
#     inference.  The previous implementation measured a CDS-support margin, which
#     could be positive even when the deployed cosine classifier still assigned
#     current samples to historical classes.

#     This objective uses the actual pairwise cosine decision margin

#         cos(Q, W_y) - cos(Q, W_old)

#     for every real current sample against every historical rival. Whole-CDS remains
#     the historical knowledge source: learned+physical pair risk and sample-specific
#     spectral ambiguity prioritize dangerous old/new conflicts. No extra margin,
#     temperature, phase bias, synthetic feature, or prototype classifier is added.

#     The caller detaches all classifier rows on this representation path. Gradients
#     therefore flow to current Q only. Classifier-row learning is supplied by the
#     separate balanced historical-positive + real-current-positive branch.
#     """
#     if not isinstance(seen_cds, ClassDiscriminativeStructureState):
#         raise TypeError("seen_cds must be ClassDiscriminativeStructureState")

#     scores = torch.as_tensor(current_cosine_scores)
#     ids = torch.as_tensor(class_ids, device=scores.device, dtype=torch.long).flatten()
#     labels = torch.as_tensor(
#         labels_global, device=scores.device, dtype=torch.long
#     ).flatten()
#     p = torch.as_tensor(
#         physical_spectral_evidence, device=scores.device, dtype=scores.dtype
#     ).detach()

#     if (
#         scores.ndim != 2
#         or scores.size(0) == 0
#         or scores.size(1) != ids.numel()
#         or not bool(torch.isfinite(scores).all())
#     ):
#         raise ValueError(
#             "current_cosine_scores must be finite [N,C] aligned with class_ids"
#         )
#     if labels.shape != (scores.size(0),):
#         raise ValueError("labels_global must be [N]")
#     if ids.numel() < 2 or int(torch.unique(ids).numel()) != int(ids.numel()):
#         raise ValueError("class_ids must contain at least two unique classes")
#     if p.ndim != 3 or p.size(0) != scores.size(0) or p.size(2) != 1:
#         raise ValueError("physical_spectral_evidence must be [N,B,1]")
#     if not bool(torch.isfinite(p).all()):
#         raise ValueError("physical_spectral_evidence contains NaN or Inf")

#     classes = seen_cds.classes
#     stored_ids = classes.class_ids.to(device=scores.device, dtype=torch.long).flatten()
#     if not torch.equal(ids, stored_ids):
#         raise ValueError(
#             "classifier class_ids and seen CDS class_ids must match exactly"
#         )

#     count = int(ids.numel())
#     old_count = int(historical_class_count)
#     if old_count <= 0 or old_count >= count:
#         raise ValueError("historical_class_count must split old and current classes")
#     if classes.physical_identity.shape[1:] != p.shape[1:]:
#         raise ValueError("physical evidence and physical CDS use different coordinates")

#     matches = labels.unsqueeze(1).eq(ids.unsqueeze(0))
#     if not bool(matches.sum(dim=1).eq(1).all()):
#         raise ValueError("a current label is absent from seen CDS")
#     target_index = matches.to(torch.long).argmax(dim=1)
#     if bool((target_index < old_count).any()):
#         raise ValueError("cross-phase separation expects current-class labels only")

#     rows = torch.arange(scores.size(0), device=scores.device)
#     positive_score = scores[rows, target_index]
#     historical_rival_score = scores[:, :old_count]
#     decision_margin = positive_score.unsqueeze(1) - historical_rival_score

#     with torch.no_grad():
#         _, _, physical_support = ordered_cds_support_components(
#             p,
#             classes.physical_identity.to(
#                 device=scores.device, dtype=scores.dtype
#             ).detach(),
#             classes.physical_variation.to(
#                 device=scores.device, dtype=scores.dtype
#             ).detach(),
#         )
#         effective_risk, _learned_risk, _physical_risk = whole_cds_pair_risk(seen_cds)
#         effective_risk = effective_risk.to(device=scores.device, dtype=scores.dtype)

#         positive_physical = physical_support[rows, target_index]
#         rival_physical = physical_support[:, :old_count]
#         spectral_confusability = torch.sqrt(
#             positive_physical.unsqueeze(1).clamp_min(0.0)
#             * rival_physical.clamp_min(0.0)
#         ).clamp(0.0, 1.0)
#         pair_risk = effective_risk.index_select(0, target_index)[:, :old_count]
#         # Keep every historical rival active. Whole-CDS/spectral evidence only
#         # prioritizes conflicts; it does not define a second decision boundary.
#         pair_weight = 1.0 + pair_risk * spectral_confusability

#     # Match the deployed argmax failure event.  Classification fails when *any*
#     # historical rival beats the current target, therefore averaging all pairwise
#     # violations can hide one catastrophic rival.  Optimize the worst risk-weighted
#     # violation per sample instead.
#     pair_violation = pair_weight * F.relu(-decision_margin)
#     per_sample, hardest_index = pair_violation.max(dim=1)
#     loss = _weighted_mean(per_sample, sample_weights)

#     hardest_rival_score = historical_rival_score.gather(
#         1, hardest_index.unsqueeze(1)
#     ).squeeze(1)
#     hardest_margin = decision_margin.gather(1, hardest_index.unsqueeze(1)).squeeze(1)
#     sample_violation = decision_margin.le(0.0).any(dim=1).to(scores.dtype)

#     values = (
#         loss,
#         positive_score.mean(),
#         hardest_rival_score.mean(),
#         hardest_margin.mean(),
#         hardest_margin.min(),
#         spectral_confusability.mean(),
#         pair_risk.mean(),
#         sample_violation.mean(),
#     )
#     if not all(bool(torch.isfinite(value)) for value in values):
#         raise RuntimeError("cross-phase decision separation became NaN or Inf")

#     return CDSSeparationObjective(
#         loss=loss,
#         positive_decision_score_mean=positive_score.mean(),
#         historical_rival_decision_score_mean=hardest_rival_score.mean(),
#         decision_margin_mean=hardest_margin.mean(),
#         decision_margin_min=hardest_margin.min(),
#         physical_confusability_mean=spectral_confusability.mean(),
#         effective_pair_risk_mean=pair_risk.mean(),
#         violation_fraction=sample_violation.mean(),
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
#     "whole_cds_support_classification_loss",
#     "whole_cds_pic_loss",
#     "cross_phase_cds_separation_loss",
#     "BalancedAllSeenClassificationObjective",
#     "class_balanced_all_seen_classification_objective",
# ]
