from __future__ import annotations

"""Classification objectives for the stable-representation SCOPE-HSI architecture.

Role
----
Phase 0 optimizes the single HSI representation Q and the persistent angular
classifier with real data only.  Incremental phases keep Q fixed and optimize
that same classifier with two evidence sources: real current-class samples and
retained historical class state {I_c,S_c}. Historical supervision is
reference-anchored: old-old decision regions are retained relative to the
previous finalized classifier, while old-new boundaries exclude new rows from
retained HSI class regions. The same deployed classifier receives both signals.

The model supplies canonical

    Q(x) in R^{K x D},  RMS(Q)=1,

and the classifier produces

    l_c(x) = vec(Q)^T normalize(w_c) = sqrt(KD) cos(theta_c).

The deployed logits are used directly by cross entropy. Historical preservation
acts on the same normalized classifier through retained decision-region evidence;
there is no margin, temperature, extra logit scale, replay, teacher, or second
prediction head.

For an imbalanced HSI training set, ``class_balance_weights`` may contain the
fixed phase-level empirical weights

    a_c = N / (C N_c),

in classifier-column order.  Then the minibatch loss

    mean_i a_{y_i} CE_i

is an unbiased minibatch estimator of the empirical macro-average class CE
under ordinary uniform sample shuffling.  No tunable balancing coefficient is
introduced.
"""

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from models.classifier import ClassifierOutput

Tensor = torch.Tensor


@dataclass(frozen=True)
class ClassificationObjective:
    loss: Tensor
    cross_entropy: Tensor
    accuracy: Tensor
    balanced_accuracy: Tensor
    targets_local: Tensor
    mean_cosine_margin: Tensor


def _validate_class_ids(class_ids: Tensor, *, columns: int, device: torch.device) -> Tensor:
    ids = torch.as_tensor(class_ids)
    if ids.ndim != 1 or ids.shape != (columns,):
        raise ValueError(f"class_ids must have shape [{columns}]")
    if ids.dtype != torch.long:
        raise TypeError("class_ids must use torch.long")
    if ids.device != device:
        raise ValueError("class_ids and logits must share a device")
    if bool((ids < 0).any()) or ids.unique().numel() != ids.numel():
        raise ValueError("class_ids must contain unique non-negative IDs")
    return ids


def _targets_local(labels_global: Tensor, class_ids: Tensor) -> Tensor:
    labels = torch.as_tensor(labels_global)
    if labels.ndim != 1 or labels.numel() == 0:
        raise ValueError("labels_global must be non-empty rank-one")
    if labels.dtype != torch.long:
        raise TypeError("labels_global must use torch.long")
    if labels.device != class_ids.device:
        raise ValueError("labels_global and class_ids must share a device")

    matches = labels.unsqueeze(1).eq(class_ids.unsqueeze(0))
    counts = matches.sum(dim=1)
    if not bool(counts.eq(1).all()):
        missing = labels[counts.eq(0)].detach().cpu().unique().tolist()
        raise ValueError(f"labels contain class IDs absent from classifier: {missing}")
    return matches.to(torch.long).argmax(dim=1)


def _balanced_accuracy(pred_local: Tensor, targets_local: Tensor) -> Tensor:
    represented = targets_local.unique(sorted=True)
    if represented.numel() == 0:
        raise RuntimeError("no represented target classes")
    return torch.stack(
        [
            pred_local[targets_local.eq(class_index)]
            .eq(targets_local[targets_local.eq(class_index)])
            .to(torch.float32)
            .mean()
            for class_index in represented
        ]
    ).mean()


def cosine_classification_objective(
    *,
    output: ClassifierOutput,
    labels_global: Tensor,
    class_balance_weights: Tensor | None = None,
) -> ClassificationObjective:
    """Compute deployed-logit CE and read-only angular diagnostics.

    ``class_balance_weights`` must be one positive weight per classifier column.
    The base trainer derives these once from the complete phase-0 TRAIN split.
    Passing ``None`` gives ordinary sample-mean cross entropy.
    """
    if not isinstance(output, ClassifierOutput):
        raise TypeError("output must be ClassifierOutput")

    logits = torch.as_tensor(output.logits)
    cosine = torch.as_tensor(output.cosine_similarity)
    prediction = torch.as_tensor(output.prediction)

    if logits.ndim != 2 or logits.size(0) == 0 or logits.size(1) < 2:
        raise ValueError("logits must be non-empty [N,C] with C>=2")
    if not torch.is_floating_point(logits) or logits.is_complex():
        raise TypeError("logits must be real floating point")
    if not bool(torch.isfinite(logits).all()):
        raise ValueError("logits contain NaN or Inf")

    if cosine.shape != logits.shape:
        raise RuntimeError("cosine_similarity must match logits shape")
    if cosine.device != logits.device or cosine.dtype != logits.dtype:
        raise RuntimeError("cosine_similarity must share logits device/dtype")
    if not torch.is_floating_point(cosine) or cosine.is_complex():
        raise TypeError("cosine_similarity must be real floating point")
    if not bool(torch.isfinite(cosine).all()):
        raise ValueError("cosine_similarity contains NaN or Inf")

    class_ids = _validate_class_ids(
        output.class_ids,
        columns=logits.size(1),
        device=logits.device,
    )

    if prediction.shape != (logits.size(0),) or prediction.dtype != torch.long:
        raise RuntimeError("prediction must contain one torch.long class ID per row")
    if prediction.device != logits.device:
        raise RuntimeError("prediction and logits must share a device")

    labels = torch.as_tensor(labels_global)
    if labels.shape != (logits.size(0),):
        raise ValueError("labels_global must contain one rank-one label per sample")
    if labels.dtype != torch.long:
        raise TypeError("labels_global must use torch.long")
    if labels.device != logits.device:
        raise ValueError("labels_global and logits must share a device")

    targets_local = _targets_local(labels, class_ids)

    # The classifier contract is: logits are a positive class-independent scale
    # of cosine similarity.  Their class ordering must therefore be identical.
    pred_local = logits.argmax(dim=1)
    cosine_pred_local = cosine.argmax(dim=1)
    if not bool(torch.equal(pred_local, cosine_pred_local)):
        raise RuntimeError("logit and cosine class orderings disagree")
    expected_prediction = class_ids.index_select(0, pred_local)
    if not bool(torch.equal(prediction, expected_prediction)):
        raise RuntimeError("ClassifierOutput.prediction disagrees with deployed logits")

    per_sample_ce = F.cross_entropy(logits, targets_local, reduction="none")

    if class_balance_weights is None:
        per_sample_objective = per_sample_ce
    else:
        weights = torch.as_tensor(
            class_balance_weights,
            device=logits.device,
            dtype=logits.dtype,
        )
        if weights.shape != (logits.size(1),):
            raise ValueError(
                "class_balance_weights must contain one value per classifier column"
            )
        if not bool(torch.isfinite(weights).all()) or bool((weights <= 0).any()):
            raise ValueError("class_balance_weights must be finite and positive")
        per_sample_objective = per_sample_ce * weights.index_select(0, targets_local)

    cross_entropy = per_sample_objective.mean()
    if not bool(torch.isfinite(cross_entropy)):
        raise RuntimeError("classification cross entropy became NaN or Inf")

    accuracy = prediction.eq(labels).to(torch.float32).mean()
    balanced_accuracy = _balanced_accuracy(pred_local, targets_local)

    rows = torch.arange(cosine.size(0), device=cosine.device)
    target_score = cosine[rows, targets_local]
    rival_score = cosine.clone()
    rival_score[rows, targets_local] = -torch.inf
    mean_margin = (target_score - rival_score.max(dim=1).values).mean()
    if not bool(torch.isfinite(mean_margin)):
        raise RuntimeError("cosine margin became NaN or Inf")

    return ClassificationObjective(
        loss=cross_entropy,
        cross_entropy=cross_entropy,
        accuracy=accuracy,
        balanced_accuracy=balanced_accuracy,
        targets_local=targets_local,
        mean_cosine_margin=mean_margin,
    )



@dataclass(frozen=True)
class HistoricalSpectralDecisionObjective:
    """Reference-anchored historical decision-region supervision."""

    loss: Tensor
    identity_cross_entropy: Tensor
    identity_accuracy: Tensor
    mean_identity_margin: Tensor
    mean_spectral_decision_score: Tensor
    minimum_spectral_decision_score: Tensor
    mean_dispersion_upper_bound: Tensor
    zero_dispersion_fraction: Tensor
    per_class_risk: Tensor
    per_class_old_old_risk: Tensor
    per_class_old_new_risk: Tensor
    old_old_retention_risk: Tensor
    old_new_exclusion_risk: Tensor
    old_old_reference_min_score: Tensor
    old_old_current_min_score: Tensor
    old_old_max_degradation: Tensor
    old_old_violation_fraction: Tensor
    old_new_min_score: Tensor
    old_new_violation_fraction: Tensor


@dataclass(frozen=True)
class UnifiedIncrementalClassificationObjective:
    """Equal-class combination of current real and historical spectral decision risk."""

    loss: Tensor
    current_real_risk: Tensor
    historical_spectral_decision_risk: Tensor
    current_class_fraction: float
    historical_class_fraction: float


def _validate_historical_boundary_matrices(
    *,
    mean_gap: Tensor,
    dispersion_upper_bound: Tensor,
    historical_class_ids: Tensor,
    rival_class_ids: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    gap = torch.as_tensor(mean_gap)
    spread = torch.as_tensor(dispersion_upper_bound)
    old_ids = torch.as_tensor(historical_class_ids)
    rival_ids = torch.as_tensor(rival_class_ids)

    if gap.ndim != 2 or gap.size(0) == 0 or gap.size(1) < 2:
        raise ValueError("mean_gap must have shape [C_old,C_seen] with C_seen>=2")
    if spread.shape != gap.shape:
        raise ValueError("dispersion_upper_bound must match mean_gap")
    if gap.device != spread.device or gap.dtype != spread.dtype:
        raise ValueError("mean_gap and dispersion_upper_bound must share device/dtype")
    if not torch.is_floating_point(gap) or gap.is_complex():
        raise TypeError("historical boundary tensors must be real floating point")
    if gap.dtype not in (torch.float32, torch.float64):
        raise TypeError("historical spectral decision risk supports float32/float64")
    if not bool(torch.isfinite(gap).all()):
        raise ValueError("mean_gap contains NaN or Inf")
    if not bool(torch.isfinite(spread).all()):
        raise ValueError("dispersion_upper_bound contains NaN or Inf")
    if bool((spread < 0).any()):
        raise ValueError("dispersion_upper_bound must be non-negative")

    if old_ids.shape != (gap.size(0),) or old_ids.dtype != torch.long:
        raise ValueError("historical_class_ids must be torch.long [C_old]")
    if rival_ids.shape != (gap.size(1),) or rival_ids.dtype != torch.long:
        raise ValueError("rival_class_ids must be torch.long [C_seen]")
    if old_ids.device != gap.device or rival_ids.device != gap.device:
        raise ValueError("class IDs and historical boundary tensors must share device")
    if bool((old_ids < 0).any()) or old_ids.unique().numel() != old_ids.numel():
        raise ValueError("historical_class_ids must be unique and non-negative")
    if bool((rival_ids < 0).any()) or rival_ids.unique().numel() != rival_ids.numel():
        raise ValueError("rival_class_ids must be unique and non-negative")

    self_mask = old_ids.unsqueeze(1).eq(rival_ids.unsqueeze(0))
    if not bool(self_mask.sum(dim=1).eq(1).all()):
        raise ValueError(
            "each historical class must occur exactly once in rival_class_ids"
        )
    offdiag = ~self_mask
    if not bool(offdiag.any(dim=1).all()):
        raise ValueError("each historical class requires at least one rival")

    # The self pair is exactly u_c-u_c=0, hence mean gap and dispersion must be
    # numerical zero. This catches class-order or boundary-construction bugs.
    eps = torch.finfo(gap.dtype).eps
    gap_scale = torch.maximum(
        gap.detach().abs().amax(),
        torch.ones((), device=gap.device, dtype=gap.dtype),
    )
    spread_scale = torch.maximum(
        spread.detach().abs().amax(),
        torch.ones((), device=spread.device, dtype=spread.dtype),
    )
    gap_tol = 64.0 * eps * gap_scale
    spread_tol = 64.0 * eps * spread_scale
    if bool((gap[self_mask].abs() > gap_tol).any()):
        raise RuntimeError("historical self mean-gap is not numerical zero")
    if bool((spread[self_mask].abs() > spread_tol).any()):
        raise RuntimeError("historical self dispersion is not numerical zero")

    return gap, spread, old_ids, rival_ids, offdiag


def _spectral_decision_score(
    *,
    mean_gap: Tensor,
    dispersion_upper_bound: Tensor,
) -> tuple[Tensor, Tensor]:
    """Return the boundary score and numerical-zero-dispersion mask.

    For non-zero retained spread, the score is the standardized historical
    decision gap ``mu/sigma_bar``.  If the retained spread is numerical zero,
    the boundary direction is deterministic under the stored second-order
    state, so the raw mean gap is used as the exact decision quantity.  The
    branch threshold is derived only from machine precision and tensor scale.
    """
    gap = torch.as_tensor(mean_gap)
    spread = torch.as_tensor(dispersion_upper_bound)
    if gap.shape != spread.shape:
        raise ValueError("mean_gap and dispersion_upper_bound must match")

    eps = torch.finfo(spread.dtype).eps
    spread_scale = torch.maximum(
        spread.detach().amax(),
        torch.ones((), device=spread.device, dtype=spread.dtype),
    )
    zero_tol = 64.0 * eps * spread_scale
    standardized = spread.detach() > zero_tol
    denominator = torch.where(standardized, spread, torch.ones_like(spread))
    score = torch.where(standardized, gap / denominator, gap)
    if not bool(torch.isfinite(score).all()):
        raise RuntimeError("historical spectral decision score contains NaN/Inf")
    return score, ~standardized


def reference_anchored_historical_decision_objective(
    *,
    mean_gap: Tensor,
    dispersion_upper_bound: Tensor,
    historical_class_ids: Tensor,
    seen_class_ids: Tensor,
    new_class_ids: Tensor,
    reference_old_old_score: Tensor,
) -> HistoricalSpectralDecisionObjective:
    """Preserve old-old regions and exclude new rows from old HSI regions.

    The deployed classifier predicts class ``c`` over rival ``j`` exactly when

        <Q, U_c-U_j>_F > 0.

    Retained ``I_c`` gives the historical mean decision gap and retained
    ``S_c[K,K,D]`` gives a conservative standard-deviation upper bound along
    that *same* classifier boundary direction.  This function therefore acts
    directly on the decision geometry rather than on a surrogate feature head.

    Old-old pairs already had a finalized boundary at phase start.  Their
    reference score is preserved with

        target_cj = max(r_cj^ref, 0),
        L_cj^oo   = relu(target_cj - r_cj^cur).

    Thus a previously safe relation may not deteriorate, while a previously
    unsafe relation is not frozen as an error and is pushed at least to the
    correct side of the boundary.

    Old-new pairs have no historical reference.  Their historical side is
    protected with

        L_cn^on = softplus(-r_cn).

    Real current-class CE supplies the opposite side of those boundaries.
    Per historical class, the worst old-old violation and worst old-new
    invasion are retained separately and then averaged so both mandatory roles
    always receive gradient. Within each role, one dangerous rival is sufficient
    to forget a class.  No margin, temperature, replay, teacher, or old/new weighting
    coefficient is introduced.
    """
    gap, spread, old_ids, seen_ids, offdiag = _validate_historical_boundary_matrices(
        mean_gap=mean_gap,
        dispersion_upper_bound=dispersion_upper_bound,
        historical_class_ids=historical_class_ids,
        rival_class_ids=seen_class_ids,
    )
    new_ids = torch.as_tensor(new_class_ids)
    if new_ids.ndim != 1 or new_ids.numel() == 0 or new_ids.dtype != torch.long:
        raise ValueError("new_class_ids must be non-empty torch.long [C_new]")
    if new_ids.device != gap.device:
        raise ValueError("new_class_ids and boundary tensors must share device")
    if bool((new_ids < 0).any()) or new_ids.unique().numel() != new_ids.numel():
        raise ValueError("new_class_ids must be unique and non-negative")
    if bool(torch.isin(new_ids, old_ids).any()):
        raise ValueError("historical and new class IDs must be disjoint")
    if set(seen_ids.detach().cpu().tolist()) != (
        set(old_ids.detach().cpu().tolist()) | set(new_ids.detach().cpu().tolist())
    ):
        raise ValueError("seen_class_ids must equal historical plus new classes")

    reference = torch.as_tensor(reference_old_old_score)
    expected_reference = (old_ids.numel(), old_ids.numel())
    if reference.shape != expected_reference:
        raise ValueError(
            "reference_old_old_score must have shape "
            f"{expected_reference}"
        )
    if reference.device != gap.device or reference.dtype != gap.dtype:
        raise ValueError("reference old-old score must share device/dtype")
    if not bool(torch.isfinite(reference).all()):
        raise ValueError("reference old-old score contains NaN or Inf")

    score, zero_dispersion = _spectral_decision_score(
        mean_gap=gap,
        dispersion_upper_bound=spread,
    )

    # Resolve old/new classifier columns explicitly by global class ID.
    old_positions = []
    for class_id in old_ids:
        matches = torch.nonzero(seen_ids.eq(class_id), as_tuple=False).flatten()
        if matches.numel() != 1:
            raise RuntimeError("historical class column mapping is invalid")
        old_positions.append(int(matches.item()))
    new_positions = []
    for class_id in new_ids:
        matches = torch.nonzero(seen_ids.eq(class_id), as_tuple=False).flatten()
        if matches.numel() != 1:
            raise RuntimeError("new class column mapping is invalid")
        new_positions.append(int(matches.item()))

    old_index = torch.tensor(old_positions, device=gap.device, dtype=torch.long)
    new_index = torch.tensor(new_positions, device=gap.device, dtype=torch.long)
    current_old_old = score.index_select(1, old_index)
    current_old_new = score.index_select(1, new_index)

    # Rows and old columns use the same historical-class order.
    old_old_self = old_ids.unsqueeze(1).eq(old_ids.unsqueeze(0))
    if not bool(old_old_self.diagonal().all()):
        raise RuntimeError("old-old self-mask construction failed")
    old_old_offdiag = ~old_old_self

    # Preserve a successful historical relation; repair a failed one to the
    # natural decision threshold zero rather than preserving the failure.
    old_old_target = reference.clamp_min(0.0)
    old_old_pair_violation = F.relu(old_old_target - current_old_old)
    old_old_pair_violation = old_old_pair_violation.masked_fill(
        old_old_self, 0.0
    )
    per_class_old_old = old_old_pair_violation.masked_fill(
        old_old_self, -torch.inf
    ).amax(dim=1)
    if not bool(torch.isfinite(per_class_old_old).all()):
        raise RuntimeError("old-old retention risk contains NaN/Inf")

    # New classifier rows must remain outside every retained historical region.
    old_new_pair_risk = F.softplus(-current_old_new)
    per_class_old_new = old_new_pair_risk.amax(dim=1)
    if not bool(torch.isfinite(per_class_old_new).all()):
        raise RuntimeError("old-new exclusion risk contains NaN/Inf")

    # Old-old retention and old-new exclusion are independent obligations.
    # Taking a max between the two would starve the smaller role of gradient;
    # therefore each historical class receives the arithmetic mean of its two
    # mandatory worst-rival risks.  This is a fixed structural decomposition,
    # not a tunable weighting coefficient.
    per_class_risk = 0.5 * (per_class_old_old + per_class_old_new)
    historical_risk = per_class_risk.mean()
    if not bool(torch.isfinite(historical_risk)):
        raise RuntimeError("reference-anchored historical risk became NaN/Inf")

    # Reporting-only identity diagnostics over every seen rival.
    target_logit = torch.zeros(gap.size(0), 1, device=gap.device, dtype=gap.dtype)
    raw_rival_logits = (-gap).masked_fill(~offdiag, -torch.inf)
    identity_ce_per_class = torch.logsumexp(
        torch.cat((target_logit, raw_rival_logits), dim=1), dim=1
    )
    identity_ce = identity_ce_per_class.mean()
    masked_gap = gap.masked_fill(~offdiag, torch.inf)
    per_class_identity_margin = masked_gap.amin(dim=1)
    identity_accuracy = (per_class_identity_margin > 0).to(torch.float32).mean()
    mean_identity_margin = per_class_identity_margin.mean()

    valid_scores = score[offdiag]
    valid_spread = spread[offdiag]
    valid_zero = zero_dispersion[offdiag]
    mean_score = valid_scores.mean()
    minimum_score = valid_scores.amin()
    mean_spread = valid_spread.mean()
    zero_fraction = valid_zero.to(gap.dtype).mean()

    reference_valid = reference[old_old_offdiag]
    current_old_old_valid = current_old_old[old_old_offdiag]
    old_old_degradation = (
        old_old_target[old_old_offdiag] - current_old_old_valid
    ).clamp_min(0.0)
    old_old_reference_min = reference_valid.amin()
    old_old_current_min = current_old_old_valid.amin()
    old_old_max_degradation = old_old_degradation.amax()
    old_old_violation_fraction = (old_old_degradation > 0).to(gap.dtype).mean()
    old_new_min = current_old_new.amin()
    old_new_violation_fraction = (current_old_new <= 0).to(gap.dtype).mean()
    old_old_retention = per_class_old_old.mean()
    old_new_exclusion = per_class_old_new.mean()

    for name, value in (
        ("identity_cross_entropy", identity_ce),
        ("mean_identity_margin", mean_identity_margin),
        ("mean_spectral_decision_score", mean_score),
        ("minimum_spectral_decision_score", minimum_score),
        ("mean_dispersion_upper_bound", mean_spread),
        ("zero_dispersion_fraction", zero_fraction),
        ("old_old_retention_risk", old_old_retention),
        ("old_new_exclusion_risk", old_new_exclusion),
        ("old_old_reference_min_score", old_old_reference_min),
        ("old_old_current_min_score", old_old_current_min),
        ("old_old_max_degradation", old_old_max_degradation),
        ("old_old_violation_fraction", old_old_violation_fraction),
        ("old_new_min_score", old_new_min),
        ("old_new_violation_fraction", old_new_violation_fraction),
    ):
        if not bool(torch.isfinite(value)):
            raise RuntimeError(f"{name} became NaN/Inf")

    return HistoricalSpectralDecisionObjective(
        loss=historical_risk,
        identity_cross_entropy=identity_ce,
        identity_accuracy=identity_accuracy,
        mean_identity_margin=mean_identity_margin,
        mean_spectral_decision_score=mean_score,
        minimum_spectral_decision_score=minimum_score,
        mean_dispersion_upper_bound=mean_spread,
        zero_dispersion_fraction=zero_fraction,
        per_class_risk=per_class_risk,
        per_class_old_old_risk=per_class_old_old,
        per_class_old_new_risk=per_class_old_new,
        old_old_retention_risk=old_old_retention,
        old_new_exclusion_risk=old_new_exclusion,
        old_old_reference_min_score=old_old_reference_min,
        old_old_current_min_score=old_old_current_min,
        old_old_max_degradation=old_old_max_degradation,
        old_old_violation_fraction=old_old_violation_fraction,
        old_new_min_score=old_new_min,
        old_new_violation_fraction=old_new_violation_fraction,
    )

def unified_incremental_classification_objective(
    *,
    current_real_risk: Tensor,
    historical_spectral_decision_risk: Tensor,
    current_class_count: int,
    historical_class_count: int,
) -> UnifiedIncrementalClassificationObjective:
    """Combine current and historical risks with exact equal-class weighting.

    ``current_real_risk`` is the macro-average current-class real-sample CE.
    ``historical_spectral_decision_risk`` is the macro-average historical decision-
    region risk. Weighting by the number of classes gives one equal class-level
    contribution per seen class and introduces no old/new tuning coefficient.
    """
    current = torch.as_tensor(current_real_risk)
    historical = torch.as_tensor(historical_spectral_decision_risk)

    if current.ndim != 0 or historical.ndim != 0:
        raise ValueError("incremental component risks must be scalars")
    if current.device != historical.device or current.dtype != historical.dtype:
        raise ValueError("incremental component risks must share device/dtype")
    if not torch.is_floating_point(current) or current.is_complex():
        raise TypeError("incremental risks must be real floating point")
    if not bool(torch.isfinite(current)) or not bool(torch.isfinite(historical)):
        raise ValueError("incremental component risks must be finite")

    if (
        isinstance(current_class_count, bool)
        or not isinstance(current_class_count, int)
        or current_class_count <= 0
    ):
        raise ValueError("current_class_count must be a positive integer")
    if (
        isinstance(historical_class_count, bool)
        or not isinstance(historical_class_count, int)
        or historical_class_count <= 0
    ):
        raise ValueError("historical_class_count must be a positive integer")

    total_count = current_class_count + historical_class_count
    current_fraction = current_class_count / float(total_count)
    historical_fraction = historical_class_count / float(total_count)
    total = current_fraction * current + historical_fraction * historical
    if not bool(torch.isfinite(total)):
        raise RuntimeError("unified incremental classification risk became NaN/Inf")

    return UnifiedIncrementalClassificationObjective(
        loss=total,
        current_real_risk=current,
        historical_spectral_decision_risk=historical,
        current_class_fraction=current_fraction,
        historical_class_fraction=historical_fraction,
    )


__all__ = [
    "ClassificationObjective",
    "HistoricalSpectralDecisionObjective",
    "UnifiedIncrementalClassificationObjective",
    "cosine_classification_objective",
    "reference_anchored_historical_decision_objective",
    "unified_incremental_classification_objective",
]















# from __future__ import annotations

# """Classification objectives for the stable-representation SCOPE-HSI architecture.

# Role
# ----
# Phase 0 optimizes the single HSI representation Q and the persistent angular
# classifier with real data only.  Incremental phases keep Q fixed and optimize
# that same classifier with two evidence sources: real current-class samples and
# retained historical class state {I_c,S_c}.  Historical structure therefore
# directly supervises the deployed classifier instead of forming a second head or
# a one-sided boundary penalty.

# The model supplies canonical

#     Q(x) in R^{K x D},  RMS(Q)=1,

# and the classifier produces

#     l_c(x) = vec(Q)^T normalize(w_c) = sqrt(KD) cos(theta_c).

# The deployed logits are used directly by cross entropy.  There is no margin,
# temperature, extra logit scale, geometry energy, preservation term, replay, or
# teacher term.

# For an imbalanced HSI training set, ``class_balance_weights`` may contain the
# fixed phase-level empirical weights

#     a_c = N / (C N_c),

# in classifier-column order.  Then the minibatch loss

#     mean_i a_{y_i} CE_i

# is an unbiased minibatch estimator of the empirical macro-average class CE
# under ordinary uniform sample shuffling.  No tunable balancing coefficient is
# introduced.
# """

# from dataclasses import dataclass

# import torch
# import torch.nn.functional as F

# from models.classifier import ClassifierOutput

# Tensor = torch.Tensor


# @dataclass(frozen=True)
# class ClassificationObjective:
#     loss: Tensor
#     cross_entropy: Tensor
#     accuracy: Tensor
#     balanced_accuracy: Tensor
#     targets_local: Tensor
#     mean_cosine_margin: Tensor


# def _validate_class_ids(class_ids: Tensor, *, columns: int, device: torch.device) -> Tensor:
#     ids = torch.as_tensor(class_ids)
#     if ids.ndim != 1 or ids.shape != (columns,):
#         raise ValueError(f"class_ids must have shape [{columns}]")
#     if ids.dtype != torch.long:
#         raise TypeError("class_ids must use torch.long")
#     if ids.device != device:
#         raise ValueError("class_ids and logits must share a device")
#     if bool((ids < 0).any()) or ids.unique().numel() != ids.numel():
#         raise ValueError("class_ids must contain unique non-negative IDs")
#     return ids


# def _targets_local(labels_global: Tensor, class_ids: Tensor) -> Tensor:
#     labels = torch.as_tensor(labels_global)
#     if labels.ndim != 1 or labels.numel() == 0:
#         raise ValueError("labels_global must be non-empty rank-one")
#     if labels.dtype != torch.long:
#         raise TypeError("labels_global must use torch.long")
#     if labels.device != class_ids.device:
#         raise ValueError("labels_global and class_ids must share a device")

#     matches = labels.unsqueeze(1).eq(class_ids.unsqueeze(0))
#     counts = matches.sum(dim=1)
#     if not bool(counts.eq(1).all()):
#         missing = labels[counts.eq(0)].detach().cpu().unique().tolist()
#         raise ValueError(f"labels contain class IDs absent from classifier: {missing}")
#     return matches.to(torch.long).argmax(dim=1)


# def _balanced_accuracy(pred_local: Tensor, targets_local: Tensor) -> Tensor:
#     represented = targets_local.unique(sorted=True)
#     if represented.numel() == 0:
#         raise RuntimeError("no represented target classes")
#     return torch.stack(
#         [
#             pred_local[targets_local.eq(class_index)]
#             .eq(targets_local[targets_local.eq(class_index)])
#             .to(torch.float32)
#             .mean()
#             for class_index in represented
#         ]
#     ).mean()


# def cosine_classification_objective(
#     *,
#     output: ClassifierOutput,
#     labels_global: Tensor,
#     class_balance_weights: Tensor | None = None,
# ) -> ClassificationObjective:
#     """Compute deployed-logit CE and read-only angular diagnostics.

#     ``class_balance_weights`` must be one positive weight per classifier column.
#     The base trainer derives these once from the complete phase-0 TRAIN split.
#     Passing ``None`` gives ordinary sample-mean cross entropy.
#     """
#     if not isinstance(output, ClassifierOutput):
#         raise TypeError("output must be ClassifierOutput")

#     logits = torch.as_tensor(output.logits)
#     cosine = torch.as_tensor(output.cosine_similarity)
#     prediction = torch.as_tensor(output.prediction)

#     if logits.ndim != 2 or logits.size(0) == 0 or logits.size(1) < 2:
#         raise ValueError("logits must be non-empty [N,C] with C>=2")
#     if not torch.is_floating_point(logits) or logits.is_complex():
#         raise TypeError("logits must be real floating point")
#     if not bool(torch.isfinite(logits).all()):
#         raise ValueError("logits contain NaN or Inf")

#     if cosine.shape != logits.shape:
#         raise RuntimeError("cosine_similarity must match logits shape")
#     if cosine.device != logits.device or cosine.dtype != logits.dtype:
#         raise RuntimeError("cosine_similarity must share logits device/dtype")
#     if not torch.is_floating_point(cosine) or cosine.is_complex():
#         raise TypeError("cosine_similarity must be real floating point")
#     if not bool(torch.isfinite(cosine).all()):
#         raise ValueError("cosine_similarity contains NaN or Inf")

#     class_ids = _validate_class_ids(
#         output.class_ids,
#         columns=logits.size(1),
#         device=logits.device,
#     )

#     if prediction.shape != (logits.size(0),) or prediction.dtype != torch.long:
#         raise RuntimeError("prediction must contain one torch.long class ID per row")
#     if prediction.device != logits.device:
#         raise RuntimeError("prediction and logits must share a device")

#     labels = torch.as_tensor(labels_global)
#     if labels.shape != (logits.size(0),):
#         raise ValueError("labels_global must contain one rank-one label per sample")
#     if labels.dtype != torch.long:
#         raise TypeError("labels_global must use torch.long")
#     if labels.device != logits.device:
#         raise ValueError("labels_global and logits must share a device")

#     targets_local = _targets_local(labels, class_ids)

#     # The classifier contract is: logits are a positive class-independent scale
#     # of cosine similarity.  Their class ordering must therefore be identical.
#     pred_local = logits.argmax(dim=1)
#     cosine_pred_local = cosine.argmax(dim=1)
#     if not bool(torch.equal(pred_local, cosine_pred_local)):
#         raise RuntimeError("logit and cosine class orderings disagree")
#     expected_prediction = class_ids.index_select(0, pred_local)
#     if not bool(torch.equal(prediction, expected_prediction)):
#         raise RuntimeError("ClassifierOutput.prediction disagrees with deployed logits")

#     per_sample_ce = F.cross_entropy(logits, targets_local, reduction="none")

#     if class_balance_weights is None:
#         per_sample_objective = per_sample_ce
#     else:
#         weights = torch.as_tensor(
#             class_balance_weights,
#             device=logits.device,
#             dtype=logits.dtype,
#         )
#         if weights.shape != (logits.size(1),):
#             raise ValueError(
#                 "class_balance_weights must contain one value per classifier column"
#             )
#         if not bool(torch.isfinite(weights).all()) or bool((weights <= 0).any()):
#             raise ValueError("class_balance_weights must be finite and positive")
#         per_sample_objective = per_sample_ce * weights.index_select(0, targets_local)

#     cross_entropy = per_sample_objective.mean()
#     if not bool(torch.isfinite(cross_entropy)):
#         raise RuntimeError("classification cross entropy became NaN or Inf")

#     accuracy = prediction.eq(labels).to(torch.float32).mean()
#     balanced_accuracy = _balanced_accuracy(pred_local, targets_local)

#     rows = torch.arange(cosine.size(0), device=cosine.device)
#     target_score = cosine[rows, targets_local]
#     rival_score = cosine.clone()
#     rival_score[rows, targets_local] = -torch.inf
#     mean_margin = (target_score - rival_score.max(dim=1).values).mean()
#     if not bool(torch.isfinite(mean_margin)):
#         raise RuntimeError("cosine margin became NaN or Inf")

#     return ClassificationObjective(
#         loss=cross_entropy,
#         cross_entropy=cross_entropy,
#         accuracy=accuracy,
#         balanced_accuracy=balanced_accuracy,
#         targets_local=targets_local,
#         mean_cosine_margin=mean_margin,
#     )



# @dataclass(frozen=True)
# class HistoricalSpectralDecisionObjective:
#     """Historical all-seen spectral decision-region risk from retained {I_c,S_c}."""

#     loss: Tensor
#     identity_cross_entropy: Tensor
#     identity_accuracy: Tensor
#     mean_identity_margin: Tensor
#     mean_spectral_decision_score: Tensor
#     minimum_spectral_decision_score: Tensor
#     mean_dispersion_upper_bound: Tensor
#     zero_dispersion_fraction: Tensor
#     per_class_risk: Tensor


# @dataclass(frozen=True)
# class UnifiedIncrementalClassificationObjective:
#     """Equal-class combination of current real and historical spectral decision risk."""

#     loss: Tensor
#     current_real_risk: Tensor
#     historical_spectral_decision_risk: Tensor
#     current_class_fraction: float
#     historical_class_fraction: float


# def _validate_historical_boundary_matrices(
#     *,
#     mean_gap: Tensor,
#     dispersion_upper_bound: Tensor,
#     historical_class_ids: Tensor,
#     rival_class_ids: Tensor,
# ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
#     gap = torch.as_tensor(mean_gap)
#     spread = torch.as_tensor(dispersion_upper_bound)
#     old_ids = torch.as_tensor(historical_class_ids)
#     rival_ids = torch.as_tensor(rival_class_ids)

#     if gap.ndim != 2 or gap.size(0) == 0 or gap.size(1) < 2:
#         raise ValueError("mean_gap must have shape [C_old,C_seen] with C_seen>=2")
#     if spread.shape != gap.shape:
#         raise ValueError("dispersion_upper_bound must match mean_gap")
#     if gap.device != spread.device or gap.dtype != spread.dtype:
#         raise ValueError("mean_gap and dispersion_upper_bound must share device/dtype")
#     if not torch.is_floating_point(gap) or gap.is_complex():
#         raise TypeError("historical boundary tensors must be real floating point")
#     if gap.dtype not in (torch.float32, torch.float64):
#         raise TypeError("historical spectral decision risk supports float32/float64")
#     if not bool(torch.isfinite(gap).all()):
#         raise ValueError("mean_gap contains NaN or Inf")
#     if not bool(torch.isfinite(spread).all()):
#         raise ValueError("dispersion_upper_bound contains NaN or Inf")
#     if bool((spread < 0).any()):
#         raise ValueError("dispersion_upper_bound must be non-negative")

#     if old_ids.shape != (gap.size(0),) or old_ids.dtype != torch.long:
#         raise ValueError("historical_class_ids must be torch.long [C_old]")
#     if rival_ids.shape != (gap.size(1),) or rival_ids.dtype != torch.long:
#         raise ValueError("rival_class_ids must be torch.long [C_seen]")
#     if old_ids.device != gap.device or rival_ids.device != gap.device:
#         raise ValueError("class IDs and historical boundary tensors must share device")
#     if bool((old_ids < 0).any()) or old_ids.unique().numel() != old_ids.numel():
#         raise ValueError("historical_class_ids must be unique and non-negative")
#     if bool((rival_ids < 0).any()) or rival_ids.unique().numel() != rival_ids.numel():
#         raise ValueError("rival_class_ids must be unique and non-negative")

#     self_mask = old_ids.unsqueeze(1).eq(rival_ids.unsqueeze(0))
#     if not bool(self_mask.sum(dim=1).eq(1).all()):
#         raise ValueError(
#             "each historical class must occur exactly once in rival_class_ids"
#         )
#     offdiag = ~self_mask
#     if not bool(offdiag.any(dim=1).all()):
#         raise ValueError("each historical class requires at least one rival")

#     # The self pair is exactly u_c-u_c=0, hence mean gap and dispersion must be
#     # numerical zero. This catches class-order or boundary-construction bugs.
#     eps = torch.finfo(gap.dtype).eps
#     gap_scale = torch.maximum(
#         gap.detach().abs().amax(),
#         torch.ones((), device=gap.device, dtype=gap.dtype),
#     )
#     spread_scale = torch.maximum(
#         spread.detach().abs().amax(),
#         torch.ones((), device=spread.device, dtype=spread.dtype),
#     )
#     gap_tol = 64.0 * eps * gap_scale
#     spread_tol = 64.0 * eps * spread_scale
#     if bool((gap[self_mask].abs() > gap_tol).any()):
#         raise RuntimeError("historical self mean-gap is not numerical zero")
#     if bool((spread[self_mask].abs() > spread_tol).any()):
#         raise RuntimeError("historical self dispersion is not numerical zero")

#     return gap, spread, old_ids, rival_ids, offdiag


# def historical_spectral_decision_objective(
#     *,
#     mean_gap: Tensor,
#     dispersion_upper_bound: Tensor,
#     historical_class_ids: Tensor,
#     rival_class_ids: Tensor,
# ) -> HistoricalSpectralDecisionObjective:
#     """All-seen historical decision-region risk using retained I_c and S_c.

#     For historical class c and every seen rival j != c,

#         mu_cj = <I_c, u_c-u_j>_F

#     is the historical mean decision gap. ``dispersion_upper_bound[c,j]`` is the
#     conservative standard-deviation upper bound derived from the full retained
#     S_c[K,K,D] along that exact classifier-boundary direction.

#     When dispersion is numerically non-zero, use the dimensionless spectral decision
#     score

#         r_cj = mu_cj / sigma_bar_cj.

#     When dispersion is numerical zero, there is no distributional spread to
#     standardize, so use the exact deterministic mean gap r_cj = mu_cj.  This
#     branch is numerical, not a research threshold.

#     The per-class loss is

#         R_c = log(1 + sum_{j != c} exp(-r_cj)).

#     This is ordinary multiclass CE written in pairwise-gap form, but with each
#     historical rival gap standardized by the retained class spread when that
#     spread exists.  Therefore the same objective handles class identity,
#     intra-class structure, and all-seen inter-class competition without a
#     margin, pair weight, Gaussian assumption, replay, or extra loss coefficient.
#     """
#     gap, spread, old_ids, rival_ids, offdiag = _validate_historical_boundary_matrices(
#         mean_gap=mean_gap,
#         dispersion_upper_bound=dispersion_upper_bound,
#         historical_class_ids=historical_class_ids,
#         rival_class_ids=rival_class_ids,
#     )

#     eps = torch.finfo(spread.dtype).eps
#     spread_scale = torch.maximum(
#         spread.detach().amax(),
#         torch.ones((), device=spread.device, dtype=spread.dtype),
#     )
#     zero_tol = 64.0 * eps * spread_scale
#     standardized = spread.detach() > zero_tol

#     # Avoid evaluating gap/0 in the inactive branch of torch.where.
#     denominator = torch.where(
#         standardized,
#         spread,
#         torch.ones_like(spread),
#     )
#     standardized_gap = gap / denominator
#     score = torch.where(standardized, standardized_gap, gap)
#     if not bool(torch.isfinite(score).all()):
#         raise RuntimeError("historical spectral decision score contains NaN/Inf")

#     # Self pairs are not rivals.  A leading zero logit is the true class, while
#     # each rival receives logit -r_cj. Cross entropy is therefore exactly
#     # log(1 + sum_j exp(-r_cj)).
#     rival_logits = (-score).masked_fill(~offdiag, -torch.inf)
#     target_logit = torch.zeros(
#         gap.size(0),
#         1,
#         device=gap.device,
#         dtype=gap.dtype,
#     )
#     per_class_risk = torch.logsumexp(
#         torch.cat((target_logit, rival_logits), dim=1),
#         dim=1,
#     )
#     historical_risk = per_class_risk.mean()
#     if not bool(torch.isfinite(historical_risk)):
#         raise RuntimeError("historical spectral decision risk became NaN/Inf")

#     # Raw identity CE is reporting-only. It shows whether the class center is
#     # correct independently of the region-aware standardized risk.
#     raw_rival_logits = (-gap).masked_fill(~offdiag, -torch.inf)
#     identity_ce_per_class = torch.logsumexp(
#         torch.cat((target_logit, raw_rival_logits), dim=1),
#         dim=1,
#     )
#     identity_ce = identity_ce_per_class.mean()

#     # The nearest-rival raw gap is exactly the identity decision margin.
#     masked_gap = gap.masked_fill(~offdiag, torch.inf)
#     per_class_identity_margin = masked_gap.amin(dim=1)
#     identity_accuracy = (
#         per_class_identity_margin > 0
#     ).to(torch.float32).mean()
#     mean_identity_margin = per_class_identity_margin.mean()

#     valid_scores = score[offdiag]
#     valid_spread = spread[offdiag]
#     valid_standardized = standardized[offdiag]
#     mean_score = valid_scores.mean()
#     minimum_score = valid_scores.amin()
#     mean_spread = valid_spread.mean()
#     zero_fraction = (~valid_standardized).to(gap.dtype).mean()

#     for name, value in (
#         ("identity_cross_entropy", identity_ce),
#         ("mean_identity_margin", mean_identity_margin),
#         ("mean_spectral_decision_score", mean_score),
#         ("minimum_spectral_decision_score", minimum_score),
#         ("mean_dispersion_upper_bound", mean_spread),
#         ("zero_dispersion_fraction", zero_fraction),
#     ):
#         if not bool(torch.isfinite(value)):
#             raise RuntimeError(f"{name} became NaN/Inf")

#     return HistoricalSpectralDecisionObjective(
#         loss=historical_risk,
#         identity_cross_entropy=identity_ce,
#         identity_accuracy=identity_accuracy,
#         mean_identity_margin=mean_identity_margin,
#         mean_spectral_decision_score=mean_score,
#         minimum_spectral_decision_score=minimum_score,
#         mean_dispersion_upper_bound=mean_spread,
#         zero_dispersion_fraction=zero_fraction,
#         per_class_risk=per_class_risk,
#     )


# def unified_incremental_classification_objective(
#     *,
#     current_real_risk: Tensor,
#     historical_spectral_decision_risk: Tensor,
#     current_class_count: int,
#     historical_class_count: int,
# ) -> UnifiedIncrementalClassificationObjective:
#     """Combine current and historical risks with exact equal-class weighting.

#     ``current_real_risk`` is the macro-average current-class real-sample CE.
#     ``historical_spectral_decision_risk`` is the macro-average historical decision-
#     region risk. Weighting by the number of classes gives one equal class-level
#     contribution per seen class and introduces no old/new tuning coefficient.
#     """
#     current = torch.as_tensor(current_real_risk)
#     historical = torch.as_tensor(historical_spectral_decision_risk)

#     if current.ndim != 0 or historical.ndim != 0:
#         raise ValueError("incremental component risks must be scalars")
#     if current.device != historical.device or current.dtype != historical.dtype:
#         raise ValueError("incremental component risks must share device/dtype")
#     if not torch.is_floating_point(current) or current.is_complex():
#         raise TypeError("incremental risks must be real floating point")
#     if not bool(torch.isfinite(current)) or not bool(torch.isfinite(historical)):
#         raise ValueError("incremental component risks must be finite")

#     if (
#         isinstance(current_class_count, bool)
#         or not isinstance(current_class_count, int)
#         or current_class_count <= 0
#     ):
#         raise ValueError("current_class_count must be a positive integer")
#     if (
#         isinstance(historical_class_count, bool)
#         or not isinstance(historical_class_count, int)
#         or historical_class_count <= 0
#     ):
#         raise ValueError("historical_class_count must be a positive integer")

#     total_count = current_class_count + historical_class_count
#     current_fraction = current_class_count / float(total_count)
#     historical_fraction = historical_class_count / float(total_count)
#     total = current_fraction * current + historical_fraction * historical
#     if not bool(torch.isfinite(total)):
#         raise RuntimeError("unified incremental classification risk became NaN/Inf")

#     return UnifiedIncrementalClassificationObjective(
#         loss=total,
#         current_real_risk=current,
#         historical_spectral_decision_risk=historical,
#         current_class_fraction=current_fraction,
#         historical_class_fraction=historical_fraction,
#     )


# __all__ = [
#     "ClassificationObjective",
#     "HistoricalSpectralDecisionObjective",
#     "UnifiedIncrementalClassificationObjective",
#     "cosine_classification_objective",
#     "historical_spectral_decision_objective",
#     "unified_incremental_classification_objective",
# ]
