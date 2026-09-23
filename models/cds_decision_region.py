from __future__ import annotations
from dataclasses import dataclass
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.classifier import CosineClassificationHead
from models.geometry_bank import (
    ClassDiscriminativeStructureState,
    combine_learned_physical_relevance,
)

Tensor = torch.Tensor


@dataclass(frozen=True)
class WholeCDSClassifierProtectionObjective:
    loss: Tensor
    global_reconstruction_loss: Tensor
    old_class_reconstruction_loss: Tensor
    new_class_reconstruction_loss: Tensor
    old_old_retention_loss: Tensor

    historical_identity_margin_mean: Tensor
    historical_identity_margin_min: Tensor
    historical_deformation_exposure_mean: Tensor
    historical_structural_clearance_mean: Tensor
    historical_structural_clearance_min: Tensor

    current_identity_margin_mean: Tensor
    current_identity_margin_min: Tensor
    current_deformation_exposure_mean: Tensor
    current_structural_clearance_mean: Tensor
    current_structural_clearance_min: Tensor

    historical_internal_clearance_mean: Tensor
    historical_internal_clearance_min: Tensor
    historical_internal_retention_mean: Tensor
    historical_internal_retention_min: Tensor
    historical_internal_violation_fraction: Tensor

    joint_structural_clearance_mean: Tensor
    joint_structural_clearance_min: Tensor
    unsafe_pair_fraction: Tensor
    pair_weight_mean: Tensor
    learned_pair_risk_mean: Tensor
    physical_pair_risk_mean: Tensor
    effective_pair_risk_mean: Tensor
    old_new_structural_violation_mean: Tensor
    old_new_violation_fraction: Tensor

    reference_clearance_mean: Tensor
    reference_clearance_min: Tensor
    reference_retention_mean: Tensor
    reference_retention_min: Tensor
    reference_violation_fraction: Tensor

    @property
    def historical_robust_classification_loss(self) -> Tensor:
        return self.old_class_reconstruction_loss

    @property
    def current_robust_structural_loss(self) -> Tensor:
        return self.new_class_reconstruction_loss


@dataclass(frozen=True)
class IncomingCDSInitializationDiagnostics:
    joint_clearance_min: Tensor
    joint_clearance_mean: Tensor
    learned_relevance_mean: Tensor
    physical_relevance_mean: Tensor
    effective_relevance_mean: Tensor


@dataclass(frozen=True)
class WholeCDSAdmissionDiagnostics:
    initial_joint_clearance_min: Tensor
    initial_joint_clearance_mean: Tensor
    final_joint_clearance_min: Tensor
    final_joint_clearance_mean: Tensor
    initial_reconstruction_loss: Tensor
    final_reconstruction_loss: Tensor
    initial_old_source_loss: Tensor
    final_old_source_loss: Tensor
    initial_new_source_loss: Tensor
    final_new_source_loss: Tensor
    initial_current_ce_loss: Tensor
    final_current_ce_loss: Tensor
    initial_admission_loss: Tensor
    final_admission_loss: Tensor
    initial_cross_phase_loss: Tensor
    final_cross_phase_loss: Tensor
    unsafe_pair_fraction_before: Tensor
    unsafe_pair_fraction_after: Tensor
    optimization_iterations: int
    converged: bool


def _validate_seen_cds(
    *,
    classifier: CosineClassificationHead,
    seen_cds: ClassDiscriminativeStructureState,
    historical_class_count: int,
) -> tuple[Tensor, Tensor, Tensor, int, int, int, int]:
    if not isinstance(classifier, CosineClassificationHead):
        raise TypeError("classifier must be CosineClassificationHead")
    if not isinstance(seen_cds, ClassDiscriminativeStructureState):
        raise TypeError("seen_cds must be ClassDiscriminativeStructureState")

    classes = seen_cds.classes
    ids = classes.class_ids.to(device=classifier.weight.device, dtype=torch.long).flatten()
    count = int(ids.numel())
    old_count = int(historical_class_count)
    if count < 2 or old_count <= 0 or old_count >= count:
        raise ValueError("historical_class_count must split old and new seen classes")
    if int(torch.unique(ids).numel()) != count:
        raise ValueError("seen class IDs must be unique")

    identity = classes.learned_identity.to(
        device=classifier.weight.device, dtype=classifier.weight.dtype
    ).detach()
    variation = classes.learned_variation.to(
        device=classifier.weight.device, dtype=classifier.weight.dtype
    ).detach()
    if identity.ndim != 3 or identity.size(0) != count or identity.size(1) < 2:
        raise ValueError("I_Q must be [C,K,D], K>=2")
    c, k, d = identity.shape
    if variation.shape != (c, k, k, d):
        raise ValueError("V_Q must be [C,K,K,D]")
    if not bool(torch.isfinite(identity).all()) or not bool(torch.isfinite(variation).all()):
        raise ValueError("I_Q/V_Q contains NaN or Inf")
    if bool((variation < 0).any()):
        raise ValueError("V_Q must be non-negative")
    if int(classifier.feature_dim) != k * d:
        raise ValueError("classifier feature dimension does not match Q")
    return ids, identity, variation, old_count, count, k, d


def _historical_cds(
    *,
    reference_classifier: CosineClassificationHead,
    reference_historical_cds: ClassDiscriminativeStructureState,
    expected_ids: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    if not isinstance(reference_classifier, CosineClassificationHead):
        raise TypeError("reference_classifier must be CosineClassificationHead")
    if not isinstance(reference_historical_cds, ClassDiscriminativeStructureState):
        raise TypeError("reference_historical_cds must be ClassDiscriminativeStructureState")
    ids = reference_historical_cds.classes.class_ids.to(
        device=reference_classifier.weight.device, dtype=torch.long
    ).flatten()
    expected = expected_ids.to(device=ids.device, dtype=torch.long)
    if not torch.equal(ids, expected):
        raise ValueError("reference historical CDS IDs disagree with historical prefix")
    identity = reference_historical_cds.classes.learned_identity.to(
        device=reference_classifier.weight.device, dtype=reference_classifier.weight.dtype
    ).detach()
    variation = reference_historical_cds.classes.learned_variation.to(
        device=reference_classifier.weight.device, dtype=reference_classifier.weight.dtype
    ).detach()
    if identity.ndim != 3 or variation.shape != (
        identity.size(0), identity.size(1), identity.size(1), identity.size(2)
    ):
        raise ValueError("reference I_Q/V_Q shape mismatch")
    return ids, identity, variation


def _directed_identity_margin_and_relation_exposure(
    *,
    source_identity: Tensor,
    source_deformation: Tensor,
    boundary: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    """Directed robust cosine clearance defined by complete learned CDS.

    I_Q supplies class identity. V_Q contributes both diagonal coordinate IQR
    sensitivity and off-diagonal ordered-token relation IQR sensitivity. V_Q is
    never interpreted as covariance, a Gaussian, or a hard class box.
    """
    identity = torch.as_tensor(source_identity)
    deformation = torch.as_tensor(source_deformation)
    normal = torch.as_tensor(boundary)
    if identity.ndim != 3 or identity.size(0) <= 0 or identity.size(1) < 2:
        raise ValueError("source_identity must be [S,K,D]")
    s, k, d = identity.shape
    if deformation.shape != (s, k, k, d):
        raise ValueError("source_deformation must be [S,K,K,D]")
    if normal.ndim != 3 or normal.size(0) != s or normal.size(2) != k * d:
        raise ValueError("boundary must be [S,R,K*D]")
    if not all(bool(torch.isfinite(v).all()) for v in (identity, deformation, normal)):
        raise ValueError("boundary-clearance tensors contain NaN or Inf")

    eps = torch.finfo(identity.dtype).eps
    flat_identity = identity.reshape(s, -1)
    identity_norm = flat_identity.norm(dim=1, keepdim=True).clamp_min(eps)
    identity_unit = flat_identity / identity_norm
    margin = (normal * identity_unit[:, None, :]).sum(dim=2)

    tangent = (
        normal - margin[:, :, None] * identity_unit[:, None, :]
    ) / identity_norm[:, None, :]
    tangent_kd = tangent.reshape(s, normal.size(1), k, d)

    index = torch.arange(k, device=identity.device)
    absolute_scale = deformation[:, index, index, :]
    absolute_exposure = (
        tangent_kd.abs() * absolute_scale[:, None, :, :]
    ).sum(dim=(2, 3))

    relation_sensitivity = (
        tangent_kd[:, :, :, None, :] - tangent_kd[:, :, None, :, :]
    ).abs()
    upper = torch.triu(
        torch.ones(k, k, device=identity.device, dtype=torch.bool), diagonal=1
    )
    relational_exposure = (
        relation_sensitivity
        * deformation[:, None, :, :, :]
        * upper.view(1, 1, k, k, 1)
    ).sum(dim=(2, 3, 4)) / float(k)

    exposure = torch.maximum(absolute_exposure, relational_exposure)
    clearance = margin - exposure
    if not all(bool(torch.isfinite(v).all()) for v in (margin, exposure, clearance)):
        raise RuntimeError("whole-CDS boundary clearance became NaN or Inf")
    return margin, exposure, clearance


@torch.no_grad()
def _previous_historical_clearance(
    *,
    reference_classifier: CosineClassificationHead,
    reference_historical_cds: ClassDiscriminativeStructureState,
    historical_ids: Tensor,
) -> tuple[Tensor, Tensor]:
    ids, identity, variation = _historical_cds(
        reference_classifier=reference_classifier,
        reference_historical_cds=reference_historical_cds,
        expected_ids=historical_ids,
    )
    if ids.numel() < 2:
        raise ValueError("reference historical CDS needs at least two classes")
    eps = torch.finfo(identity.dtype).eps
    rows = F.normalize(reference_classifier.class_weight(ids), p=2, dim=1, eps=eps)
    boundary = rows[:, None, :] - rows[None, :, :]
    _, _, clearance = _directed_identity_margin_and_relation_exposure(
        source_identity=identity,
        source_deformation=variation,
        boundary=boundary,
    )
    diagonal = torch.eye(ids.numel(), device=clearance.device, dtype=torch.bool)
    floor = clearance.masked_fill(diagonal, torch.inf).min(dim=1).values
    if not bool(torch.isfinite(clearance).all()) or not bool(torch.isfinite(floor).all()):
        raise RuntimeError("previous historical clearance became invalid")
    return clearance.detach(), floor.detach()


def _effective_pair_risk(learned: Tensor, physical: Tensor) -> Tensor:
    """Use the single canonical Whole-CDS learned/physical fusion rule."""
    learned_value = torch.as_tensor(learned)
    physical_value = torch.as_tensor(
        physical, device=learned_value.device, dtype=learned_value.dtype
    )
    if learned_value.ndim != 2 or learned_value.shape != physical_value.shape:
        raise ValueError("learned/physical risk must share a rank-two shape")
    return combine_learned_physical_relevance(learned_value, physical_value)


def _classwise_robust_loss(
    *,
    clearance: Tensor,
    pair_weight: Tensor,
    valid_mask: Tensor,
    scale: Tensor,
) -> Tensor:
    if clearance.shape != pair_weight.shape or clearance.shape != valid_mask.shape:
        raise ValueError("clearance, pair_weight and valid_mask must share shape")
    if valid_mask.dtype != torch.bool:
        raise TypeError("valid_mask must be boolean")
    eps = torch.finfo(clearance.dtype).eps
    terms = pair_weight.clamp_min(eps).log() - scale * clearance
    terms = terms.masked_fill(~valid_mask, -torch.inf)
    if not bool(valid_mask.any(dim=1).all()):
        raise RuntimeError("every source class must have at least one rival")
    zero = torch.zeros((clearance.size(0), 1), device=clearance.device, dtype=clearance.dtype)
    return torch.logsumexp(torch.cat((zero, terms), dim=1), dim=1)


@torch.no_grad()
def initialize_incoming_classifier_rows_from_cds_identity(
    *,
    classifier: CosineClassificationHead,
    seen_cds: ClassDiscriminativeStructureState,
    historical_class_count: int,
) -> IncomingCDSInitializationDiagnostics:
   
    ids, identity, variation, old_count, count, _k, _d = _validate_seen_cds(
        classifier=classifier,
        seen_cds=seen_cds,
        historical_class_count=historical_class_count,
    )
    eps = torch.finfo(identity.dtype).eps
    unit_identity = F.normalize(identity.reshape(count, -1), p=2, dim=1, eps=eps)
    new_rows = unit_identity[old_count:]
    new_ids = ids[old_count:]
    classifier.weight.index_copy_(
        0, new_ids, new_rows.to(device=classifier.weight.device, dtype=classifier.weight.dtype)
    )

    old_rows = F.normalize(
        classifier.class_weight(ids[:old_count]), p=2, dim=1, eps=eps
    ).detach()
    old_boundary = old_rows[:, None, :] - new_rows[None, :, :]
    _, _, old_to_new = _directed_identity_margin_and_relation_exposure(
        source_identity=identity[:old_count],
        source_deformation=variation[:old_count],
        boundary=old_boundary,
    )
    new_boundary = new_rows[:, None, :] - old_rows[None, :, :]
    _, _, new_to_old = _directed_identity_margin_and_relation_exposure(
        source_identity=identity[old_count:],
        source_deformation=variation[old_count:],
        boundary=new_boundary,
    )
    joint = torch.minimum(old_to_new, new_to_old.transpose(0, 1).contiguous())

    learned_all = seen_cds.relations.learned_combined_proximity.to(
        device=identity.device, dtype=identity.dtype
    ).detach()
    physical_all = seen_cds.relations.physical_combined_proximity.to(
        device=identity.device, dtype=identity.dtype
    ).detach()
    effective_all = _effective_pair_risk(learned_all, physical_all)
    cross_learned = learned_all[:old_count, old_count:]
    cross_physical = physical_all[:old_count, old_count:]
    cross_effective = effective_all[:old_count, old_count:]

    if not all(bool(torch.isfinite(v).all()) for v in (new_rows, joint, cross_effective)):
        raise RuntimeError("incoming CDS identity initialization became NaN or Inf")

    return IncomingCDSInitializationDiagnostics(
        joint_clearance_min=joint.min(),
        joint_clearance_mean=joint.mean(),
        learned_relevance_mean=cross_learned.mean(),
        physical_relevance_mean=cross_physical.mean(),
        effective_relevance_mean=cross_effective.mean(),
    )


def _incoming_whole_cds_clearance(
    *,
    classifier: CosineClassificationHead,
    seen_cds: ClassDiscriminativeStructureState,
    historical_class_count: int,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Return joint old/new clearance and Whole-CDS cross-phase risk.

    The same I_Q/V_Q robust boundary geometry and the same learned/physical
    relevance fusion used by classifier reconstruction are used here.  This is
    therefore not an auxiliary classifier or a second calibration rule: it is
    the phase-start form of the same Whole-CDS decision-reconstruction contract.
    """
    ids, identity, variation, old_count, count, _k, _d = _validate_seen_cds(
        classifier=classifier,
        seen_cds=seen_cds,
        historical_class_count=historical_class_count,
    )
    eps = torch.finfo(identity.dtype).eps
    rows = F.normalize(classifier.class_weight(ids), p=2, dim=1, eps=eps)
    old_rows = rows[:old_count]
    new_rows = rows[old_count:]

    old_boundary = old_rows[:, None, :] - new_rows[None, :, :]
    _, _, old_to_new = _directed_identity_margin_and_relation_exposure(
        source_identity=identity[:old_count],
        source_deformation=variation[:old_count],
        boundary=old_boundary,
    )
    new_boundary = new_rows[:, None, :] - old_rows[None, :, :]
    _, _, new_to_old = _directed_identity_margin_and_relation_exposure(
        source_identity=identity[old_count:],
        source_deformation=variation[old_count:],
        boundary=new_boundary,
    )
    joint = torch.minimum(old_to_new, new_to_old.transpose(0, 1).contiguous())

    learned = seen_cds.relations.learned_combined_proximity.to(
        device=identity.device, dtype=identity.dtype
    ).detach()
    physical = seen_cds.relations.physical_combined_proximity.to(
        device=identity.device, dtype=identity.dtype
    ).detach()
    effective = _effective_pair_risk(learned, physical)
    cross_effective = effective[:old_count, old_count:]
    pair_weight = 1.0 + cross_effective
    if not all(bool(torch.isfinite(v).all()) for v in (joint, cross_effective, pair_weight)):
        raise RuntimeError("incoming Whole-CDS clearance contains NaN or Inf")
    return joint, cross_effective, pair_weight, ids


def _cross_phase_admission_loss(
    *,
    classifier: CosineClassificationHead,
    seen_cds: ClassDiscriminativeStructureState,
    historical_class_count: int,
) -> tuple[Tensor, Tensor]:
    joint, _effective, pair_weight, _ids = _incoming_whole_cds_clearance(
        classifier=classifier,
        seen_cds=seen_cds,
        historical_class_count=historical_class_count,
    )
    scale = classifier.logit_scale.to(device=joint.device, dtype=joint.dtype).detach()
    eps = torch.finfo(joint.dtype).eps
    # Smooth worst-pair Whole-CDS barrier.  ``joint`` is the minimum of the
    # old->new and new->old robust clearances, so a pair is admitted only when
    # both directions are safe.  log pair-risk weights prioritize spectrally and
    # structurally confusable pairs without creating a separate scoring rule.
    terms = pair_weight.clamp_min(eps).log() - scale * joint
    zero = torch.zeros(1, device=joint.device, dtype=joint.dtype)
    loss = torch.logsumexp(torch.cat((zero, terms.reshape(-1))), dim=0) / scale.clamp_min(eps)
    unsafe_fraction = joint.lt(0.0).to(joint.dtype).mean()
    if not bool(torch.isfinite(loss)) or not bool(torch.isfinite(unsafe_fraction)):
        raise RuntimeError("incoming Whole-CDS admission loss became NaN or Inf")
    return loss, unsafe_fraction


def _phase_start_current_ce(
    *,
    classifier: CosineClassificationHead,
    current_features: Tensor,
    current_labels: Tensor,
    seen_ids: Tensor,
    new_ids: Tensor,
    sample_weights: Tensor | None,
) -> Tensor:
    """Real-current positive support for classifier admission.

    Current TRAIN features are encoded by the frozen Phase-(t-1) backbone, so
    they live in the same coordinate as the historical CDS used at admission.
    All seen rows compete in the softmax, but only incoming rows receive gradient.
    No historical image or feature replay is introduced.
    """
    q = torch.as_tensor(current_features)
    labels = torch.as_tensor(current_labels)
    if q.ndim != 3 or q.size(0) <= 0 or q.size(1) < 2 or q.size(2) <= 0:
        raise ValueError("current_features must be non-empty [N,K,D]")
    if labels.shape != (q.size(0),) or labels.dtype != torch.long or labels.device != q.device:
        raise ValueError("current_labels must be torch.long [N] on the current-feature device")
    if q.device != classifier.weight.device or q.dtype != classifier.weight.dtype:
        raise ValueError("current features and classifier must share device/dtype")
    if not bool(torch.isfinite(q).all()):
        raise ValueError("current_features contain NaN or Inf")
    if not bool(torch.isin(labels, new_ids).all()):
        raise ValueError("phase-start current_labels must contain only incoming classes")

    output = classifier(
        q,
        class_ids=seen_ids,
        trainable_class_ids=new_ids,
    )
    matches = labels.unsqueeze(1).eq(seen_ids.unsqueeze(0))
    if not bool(matches.sum(dim=1).eq(1).all()):
        raise RuntimeError("current admission label is absent from seen class IDs")
    targets = matches.to(torch.long).argmax(dim=1)
    per_sample = F.cross_entropy(output.logits, targets, reduction="none")
    if sample_weights is None:
        loss = per_sample.mean()
    else:
        weights = torch.as_tensor(
            sample_weights, device=per_sample.device, dtype=per_sample.dtype
        )
        if weights.shape != per_sample.shape:
            raise ValueError("current_sample_weights must match current labels")
        if not bool(torch.isfinite(weights).all()) or bool((weights < 0).any()) or not bool(weights.sum() > 0):
            raise ValueError("current_sample_weights must be finite, non-negative and have positive mass")
        loss = (per_sample * weights).sum() / weights.sum()
    if not bool(torch.isfinite(loss)):
        raise RuntimeError("phase-start real-current CE became NaN or Inf")
    return loss


def reconstruct_incoming_classifier_rows_from_whole_cds(
    *,
    classifier: CosineClassificationHead,
    seen_cds: ClassDiscriminativeStructureState,
    historical_class_count: int,
    current_features: Tensor,
    current_labels: Tensor,
    current_sample_weights: Tensor | None = None,
    max_iter: int = 80,
    tolerance_grad: float = 1e-7,
    tolerance_change: float = 1e-9,
) -> WholeCDSAdmissionDiagnostics:
    """Reconstruct incoming rows before all-seen incremental SGD.

    The old implementation imposed a hard requirement that *every* bidirectional
    robust old/new CDS clearance be >= 0.  That condition is not generally
    feasible when two robust class structures overlap, and it can stop a valid
    incremental phase even though the correct research principle is to minimize
    decision interference rather than declare overlapping CDS mathematically
    separable.

    Admission now follows the same classifier-reconstruction principle used in
    the joint incremental stage:

      1. I_Q provides only the deterministic incoming-row seed.
      2. Complete old+new {I_Q,V_Q,I_P,V_P} supplies class-balanced structural
         supervision through ``whole_cds_class_balanced_classifier_loss``.
      3. Real current TRAIN Q (encoded by the frozen previous backbone) supplies
         current positive CE support against all seen rows.
      4. Only incoming classifier rows may change. Historical rows, future rows,
         the backbone and the shared logit scale remain fixed.

    Robust clearance and unsafe-pair fraction remain diagnostics.  They are not
    a hard feasibility gate, because negative robust clearance can be intrinsic
    to overlapping IQR-scale class structures rather than an optimizer failure.
    """
    if isinstance(max_iter, bool) or int(max_iter) <= 0:
        raise ValueError("max_iter must be a positive integer")
    if not math.isfinite(float(tolerance_grad)) or float(tolerance_grad) <= 0.0:
        raise ValueError("tolerance_grad must be finite and positive")
    if not math.isfinite(float(tolerance_change)) or float(tolerance_change) <= 0.0:
        raise ValueError("tolerance_change must be finite and positive")

    ids, _identity, _variation, old_count, count, _k, _d = _validate_seen_cds(
        classifier=classifier,
        seen_cds=seen_cds,
        historical_class_count=historical_class_count,
    )
    old_ids = ids[:old_count]
    new_ids = ids[old_count:]
    seen_set = set(int(v) for v in ids.detach().cpu().tolist())
    future_ids = torch.tensor(
        [i for i in range(classifier.total_class_count) if i not in seen_set],
        device=classifier.weight.device,
        dtype=torch.long,
    )

    # Deterministic I_Q direction is a seed only.
    init = initialize_incoming_classifier_rows_from_cds_identity(
        classifier=classifier,
        seen_cds=seen_cds,
        historical_class_count=historical_class_count,
    )
    seed_new_rows = classifier.class_weight(new_ids).detach().clone()
    old_snapshot = classifier.class_weight(old_ids).detach().clone()
    future_snapshot = (
        classifier.class_weight(future_ids).detach().clone()
        if future_ids.numel()
        else None
    )

    q = torch.as_tensor(
        current_features,
        device=classifier.weight.device,
        dtype=classifier.weight.dtype,
    )
    y = torch.as_tensor(current_labels, device=classifier.weight.device, dtype=torch.long).flatten()
    weights = None
    if current_sample_weights is not None:
        weights = torch.as_tensor(
            current_sample_weights,
            device=classifier.weight.device,
            dtype=classifier.weight.dtype,
        ).flatten()

    scale_requires_grad = bool(classifier.logit_scale_parameter.requires_grad)
    classifier.logit_scale_parameter.requires_grad_(False)
    classifier.logit_scale_parameter.grad = None
    classifier.weight.requires_grad_(True)

    def measure() -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        structural = whole_cds_class_balanced_classifier_loss(
            classifier=classifier,
            cds=seen_cds,
        )
        old_loss = structural.per_class_loss[:old_count].mean()
        new_loss = structural.per_class_loss[old_count:].mean()
        current_ce = _phase_start_current_ce(
            classifier=classifier,
            current_features=q,
            current_labels=y,
            seen_ids=ids,
            new_ids=new_ids,
            sample_weights=weights,
        )
        # Both terms are class/sample means in the same all-seen classifier.
        # Equal addition avoids a new arbitrary weighting hyperparameter.
        admission_loss = structural.loss + current_ce
        cross_loss, unsafe_fraction = _cross_phase_admission_loss(
            classifier=classifier,
            seen_cds=seen_cds,
            historical_class_count=old_count,
        )
        joint, _effective, _pair_weight, _ = _incoming_whole_cds_clearance(
            classifier=classifier,
            seen_cds=seen_cds,
            historical_class_count=old_count,
        )
        return (
            structural.loss,
            old_loss,
            new_loss,
            current_ce,
            admission_loss,
            cross_loss,
            unsafe_fraction,
            joint,
        )

    with torch.no_grad():
        (
            initial_reconstruction,
            initial_old_loss,
            initial_new_loss,
            initial_current_ce,
            initial_admission,
            initial_cross,
            initial_unsafe,
            initial_joint,
        ) = [v.detach() for v in measure()]

    optimizer = torch.optim.LBFGS(
        [classifier.weight],
        lr=1.0,
        max_iter=int(max_iter),
        tolerance_grad=float(tolerance_grad),
        tolerance_change=float(tolerance_change),
        history_size=min(20, int(max_iter)),
        line_search_fn="strong_wolfe",
    )

    def closure() -> Tensor:
        optimizer.zero_grad(set_to_none=True)
        structural = whole_cds_class_balanced_classifier_loss(
            classifier=classifier,
            cds=seen_cds,
        )
        current_ce = _phase_start_current_ce(
            classifier=classifier,
            current_features=q,
            current_labels=y,
            seen_ids=ids,
            new_ids=new_ids,
            sample_weights=weights,
        )
        loss = structural.loss + current_ce
        if not bool(torch.isfinite(loss)):
            raise RuntimeError("Whole-CDS admission objective became NaN or Inf")
        loss.backward()
        gradient = classifier.weight.grad
        if gradient is None or not bool(torch.isfinite(gradient).all()):
            raise RuntimeError("Whole-CDS admission gradient is missing or invalid")
        # Only incoming rows are owned by the admission problem.
        keep = torch.zeros(
            classifier.total_class_count,
            device=gradient.device,
            dtype=torch.bool,
        )
        keep.index_fill_(0, new_ids, True)
        gradient.mul_(keep.unsqueeze(1).to(dtype=gradient.dtype))
        return loss

    try:
        optimizer.step(closure)
        state = optimizer.state.get(classifier.weight, {})
        iterations = int(state.get("n_iter", int(max_iter)))

        with torch.no_grad():
            normalized_new = F.normalize(
                classifier.class_weight(new_ids),
                p=2,
                dim=1,
                eps=torch.finfo(classifier.weight.dtype).eps,
            )
            classifier.weight.index_copy_(0, new_ids, normalized_new)

            if not torch.equal(classifier.class_weight(old_ids), old_snapshot):
                raise RuntimeError("Whole-CDS admission changed a historical classifier row")
            if future_ids.numel() and not torch.equal(
                classifier.class_weight(future_ids), future_snapshot
            ):
                raise RuntimeError("Whole-CDS admission changed a future classifier row")

            final_values = [v.detach() for v in measure()]
            (
                final_reconstruction,
                final_old_loss,
                final_new_loss,
                final_current_ce,
                final_admission,
                final_cross,
                final_unsafe,
                final_joint,
            ) = final_values

            # Numerical optimization is allowed to improve the phase-start state,
            # never to make it worse.  If LBFGS ends above the deterministic seed
            # objective, restore the seed rather than admitting a degraded row.
            tolerance = 128.0 * torch.finfo(final_admission.dtype).eps * max(
                1.0, abs(float(initial_admission.item()))
            )
            if float(final_admission.item()) > float(initial_admission.item()) + tolerance:
                classifier.weight.index_copy_(0, new_ids, seed_new_rows)
                (
                    final_reconstruction,
                    final_old_loss,
                    final_new_loss,
                    final_current_ce,
                    final_admission,
                    final_cross,
                    final_unsafe,
                    final_joint,
                ) = [v.detach() for v in measure()]

            final_min = final_joint.min().detach()
            final_mean = final_joint.mean().detach()
    finally:
        classifier.logit_scale_parameter.requires_grad_(scale_requires_grad)
        classifier.logit_scale_parameter.grad = None

    converged = bool(
        final_admission
        <= initial_admission
        + initial_admission.new_tensor(
            128.0
            * torch.finfo(initial_admission.dtype).eps
            * max(1.0, abs(float(initial_admission.item())))
        )
    )
    if not converged:
        raise RuntimeError("Whole-CDS admission failed to preserve/improve its valid objective")

    return WholeCDSAdmissionDiagnostics(
        initial_joint_clearance_min=initial_joint.min().detach(),
        initial_joint_clearance_mean=initial_joint.mean().detach(),
        final_joint_clearance_min=final_min,
        final_joint_clearance_mean=final_mean,
        initial_reconstruction_loss=initial_reconstruction,
        final_reconstruction_loss=final_reconstruction,
        initial_old_source_loss=initial_old_loss,
        final_old_source_loss=final_old_loss,
        initial_new_source_loss=initial_new_loss,
        final_new_source_loss=final_new_loss,
        initial_current_ce_loss=initial_current_ce,
        final_current_ce_loss=final_current_ce,
        initial_admission_loss=initial_admission,
        final_admission_loss=final_admission,
        initial_cross_phase_loss=initial_cross,
        final_cross_phase_loss=final_cross,
        unsafe_pair_fraction_before=initial_unsafe,
        unsafe_pair_fraction_after=final_unsafe,
        optimization_iterations=iterations,
        converged=converged,
    )


@dataclass(frozen=True)
class WholeCDSClassBalancedObjective:
    loss: Tensor
    per_class_loss: Tensor
    clearance_mean: Tensor
    clearance_min: Tensor
    learned_pair_risk_mean: Tensor
    physical_pair_risk_mean: Tensor
    effective_pair_risk_mean: Tensor


def whole_cds_class_balanced_classifier_loss(
    *,
    classifier: CosineClassificationHead,
    cds: ClassDiscriminativeStructureState,
) -> WholeCDSClassBalancedObjective:
    """Class-balanced Whole-CDS classifier supervision for a single class set.

    This is the common classifier-side objective used in Phase 0 and incremental
    phases.  Each source class contributes equal mass.  I_Q/V_Q define robust
    directed cosine clearance and I_P/V_P contribute through the same learned +
    physical inter-class relevance already defined by the CDS bank.
    """
    if not isinstance(classifier, CosineClassificationHead):
        raise TypeError("classifier must be CosineClassificationHead")
    if not isinstance(cds, ClassDiscriminativeStructureState):
        raise TypeError("cds must be ClassDiscriminativeStructureState")
    classes = cds.classes
    ids = classes.class_ids.to(device=classifier.weight.device, dtype=torch.long).flatten()
    count = int(ids.numel())
    if count < 2 or int(torch.unique(ids).numel()) != count:
        raise ValueError("CDS classifier objective requires at least two unique classes")
    identity = classes.learned_identity.to(
        device=classifier.weight.device, dtype=classifier.weight.dtype
    ).detach()
    variation = classes.learned_variation.to(
        device=classifier.weight.device, dtype=classifier.weight.dtype
    ).detach()
    if identity.ndim != 3 or variation.shape != (count, identity.size(1), identity.size(1), identity.size(2)):
        raise ValueError("I_Q/V_Q shape mismatch")
    eps = torch.finfo(identity.dtype).eps
    rows = F.normalize(classifier.class_weight(ids), p=2, dim=1, eps=eps)
    boundary = rows[:, None, :] - rows[None, :, :]
    _, _, clearance = _directed_identity_margin_and_relation_exposure(
        source_identity=identity, source_deformation=variation, boundary=boundary
    )
    learned = cds.relations.learned_combined_proximity.to(
        device=identity.device, dtype=identity.dtype
    ).detach()
    physical = cds.relations.physical_combined_proximity.to(
        device=identity.device, dtype=identity.dtype
    ).detach()
    effective = _effective_pair_risk(learned, physical)
    pair_weight = 1.0 + effective
    valid = ~torch.eye(count, device=identity.device, dtype=torch.bool)
    scale = classifier.logit_scale.to(device=identity.device, dtype=identity.dtype).detach()
    per_class = _classwise_robust_loss(
        clearance=clearance, pair_weight=pair_weight, valid_mask=valid, scale=scale
    )
    loss = per_class.mean()
    offdiag = valid
    scalars=(loss, clearance[offdiag].mean(), clearance[offdiag].min(),
             learned[offdiag].mean(), physical[offdiag].mean(), effective[offdiag].mean())
    if not all(bool(torch.isfinite(v)) for v in scalars):
        raise RuntimeError("whole-CDS class-balanced objective became NaN or Inf")
    return WholeCDSClassBalancedObjective(
        loss=loss, per_class_loss=per_class,
        clearance_mean=clearance[offdiag].mean(), clearance_min=clearance[offdiag].min(),
        learned_pair_risk_mean=learned[offdiag].mean(),
        physical_pair_risk_mean=physical[offdiag].mean(),
        effective_pair_risk_mean=effective[offdiag].mean(),
    )


class HSICDSDecisionRegionRegulator(nn.Module):
    """Class-balanced all-seen classifier reconstruction from the whole CDS."""

    def __init__(self) -> None:
        super().__init__()

    @property
    def parameter_count(self) -> int:
        return 0

    def forward(
        self,
        *,
        classifier: CosineClassificationHead,
        seen_cds: ClassDiscriminativeStructureState,
        historical_class_count: int,
        reference_classifier: CosineClassificationHead | None = None,
        reference_historical_cds: ClassDiscriminativeStructureState | None = None,
    ) -> WholeCDSClassifierProtectionObjective:
        ids, identity, variation, old_count, count, _k, _d = _validate_seen_cds(
            classifier=classifier,
            seen_cds=seen_cds,
            historical_class_count=historical_class_count,
        )
        new_count = count - old_count
        eps = torch.finfo(identity.dtype).eps
        rows = classifier.class_weight(ids)
        unit_rows = F.normalize(rows, p=2, dim=1, eps=eps)
        old_rows = unit_rows[:old_count]
        new_rows = unit_rows[old_count:]

        relations = seen_cds.relations
        learned_all = relations.learned_combined_proximity.to(
            device=identity.device, dtype=identity.dtype
        ).detach()
        physical_all = relations.physical_combined_proximity.to(
            device=identity.device, dtype=identity.dtype
        ).detach()
        if learned_all.shape != (count, count) or physical_all.shape != (count, count):
            raise RuntimeError("whole-CDS relation matrices must be [C,C]")
        effective_all = _effective_pair_risk(learned_all, physical_all)
        pair_weight_all = 1.0 + effective_all

        scale = classifier.logit_scale.to(device=identity.device, dtype=identity.dtype).detach()
        if scale.ndim != 0 or not bool(torch.isfinite(scale)) or not bool(scale > 0):
            raise RuntimeError("cosine logit scale is invalid")

        # Historical source classes provide genuine historical positive evidence.
        # Their positive rows and all seen rivals remain live.
        old_boundary_all = old_rows[:, None, :] - unit_rows[None, :, :]
        old_margin_all, old_exposure_all, old_clearance_all = (
            _directed_identity_margin_and_relation_exposure(
                source_identity=identity[:old_count],
                source_deformation=variation[:old_count],
                boundary=old_boundary_all,
            )
        )
        old_mask = torch.ones(
            old_count, count, device=identity.device, dtype=torch.bool
        )
        old_index = torch.arange(old_count, device=identity.device)
        old_mask[old_index, old_index] = False
        old_per_class = _classwise_robust_loss(
            clearance=old_clearance_all,
            pair_weight=pair_weight_all[:old_count],
            valid_mask=old_mask,
            scale=scale,
        )
        old_reconstruction_loss = old_per_class.mean()

        # Current-source CDS is real current-class structure. It may update incoming
        # rows, but old rival rows are detached so current-only evidence cannot
        # rewrite historical rows. New-vs-new boundaries remain fully trainable.
        new_old_boundary = new_rows[:, None, :] - old_rows.detach()[None, :, :]
        new_old_margin, new_old_exposure, new_old_clearance = (
            _directed_identity_margin_and_relation_exposure(
                source_identity=identity[old_count:],
                source_deformation=variation[old_count:],
                boundary=new_old_boundary,
            )
        )
        new_new_boundary = new_rows[:, None, :] - new_rows[None, :, :]
        new_new_margin, new_new_exposure, new_new_clearance = (
            _directed_identity_margin_and_relation_exposure(
                source_identity=identity[old_count:],
                source_deformation=variation[old_count:],
                boundary=new_new_boundary,
            )
        )
        new_margin_all = torch.cat((new_old_margin, new_new_margin), dim=1)
        new_exposure_all = torch.cat((new_old_exposure, new_new_exposure), dim=1)
        new_clearance_all = torch.cat((new_old_clearance, new_new_clearance), dim=1)
        new_mask = torch.ones(
            new_count, count, device=identity.device, dtype=torch.bool
        )
        new_index = torch.arange(new_count, device=identity.device)
        new_mask[new_index, old_count + new_index] = False
        new_per_class = _classwise_robust_loss(
            clearance=new_clearance_all,
            pair_weight=pair_weight_all[old_count:],
            valid_mask=new_mask,
            scale=scale,
        )
        new_reconstruction_loss = new_per_class.mean()

        global_reconstruction_loss = (
            old_per_class.sum() + new_per_class.sum()
        ) / float(count)

        # Previous old-old robust clearance is the only historical classifier target
        # that existed before the current phase and is therefore retained exactly.
        old_internal_clearance = old_clearance_all[:, :old_count]
        old_internal_mask = ~torch.eye(
            old_count, device=identity.device, dtype=torch.bool
        )
        if reference_classifier is not None or reference_historical_cds is not None:
            if reference_classifier is None or reference_historical_cds is None:
                raise ValueError("reference classifier and historical CDS must be supplied together")
            reference_internal, reference_floor = _previous_historical_clearance(
                reference_classifier=reference_classifier,
                reference_historical_cds=reference_historical_cds,
                historical_ids=ids[:old_count],
            )
            reference_internal = reference_internal.to(
                device=identity.device, dtype=identity.dtype
            )
            reference_floor = reference_floor.to(
                device=identity.device, dtype=identity.dtype
            )
            internal_retention = old_internal_clearance - reference_internal
        else:
            reference_internal = identity.new_zeros((old_count, old_count))
            reference_floor = identity.new_zeros((old_count,))
            internal_retention = old_internal_clearance

        internal_violation = F.relu(-internal_retention)[old_internal_mask]
        if internal_violation.numel():
            old_old_retention_loss = (
                torch.logsumexp(scale * internal_violation, dim=0)
                - math.log(float(internal_violation.numel()))
            ).clamp_min(0.0)
        else:
            old_old_retention_loss = old_internal_clearance.sum() * 0.0

        loss = global_reconstruction_loss + old_old_retention_loss

        # Cross-phase diagnostics use both directions of the same whole-CDS
        # clearance. They are diagnostics, not a second objective.
        old_to_new = old_clearance_all[:, old_count:]
        new_to_old = new_old_clearance.transpose(0, 1).contiguous()
        joint_clearance = torch.minimum(old_to_new, new_to_old)
        cross_learned = learned_all[:old_count, old_count:]
        cross_physical = physical_all[:old_count, old_count:]
        cross_effective = effective_all[:old_count, old_count:]
        cross_weight = 1.0 + cross_effective
        old_new_violation = F.relu(-joint_clearance)
        old_new_violation_mean = (
            (cross_weight * old_new_violation).sum()
            / cross_weight.sum().clamp_min(eps)
        )
        old_new_unsafe = joint_clearance.lt(0.0).to(identity.dtype).mean()

        historical_margin = old_margin_all[:, old_count:]
        historical_exposure = old_exposure_all[:, old_count:]
        current_margin = new_old_margin.transpose(0, 1).contiguous()
        current_exposure = new_old_exposure.transpose(0, 1).contiguous()
        current_clearance = new_old_clearance.transpose(0, 1).contiguous()

        reference_retention = old_to_new - reference_floor[:, None]
        reference_violation = reference_retention.lt(0.0).to(identity.dtype).mean()
        internal_unsafe = internal_retention[old_internal_mask].lt(0.0).to(identity.dtype).mean()

        scalars = (
            loss,
            global_reconstruction_loss,
            old_reconstruction_loss,
            new_reconstruction_loss,
            old_old_retention_loss,
            historical_margin.mean(), historical_margin.min(),
            historical_exposure.mean(), old_to_new.mean(), old_to_new.min(),
            current_margin.mean(), current_margin.min(), current_exposure.mean(),
            current_clearance.mean(), current_clearance.min(),
            old_internal_clearance[old_internal_mask].mean(),
            old_internal_clearance[old_internal_mask].min(),
            internal_retention[old_internal_mask].mean(),
            internal_retention[old_internal_mask].min(), internal_unsafe,
            joint_clearance.mean(), joint_clearance.min(), cross_weight.mean(),
            cross_learned.mean(), cross_physical.mean(), cross_effective.mean(),
            old_new_violation_mean, old_new_unsafe,
            reference_floor.mean(), reference_floor.min(),
            reference_retention.mean(), reference_retention.min(), reference_violation,
        )
        if not all(bool(torch.isfinite(value)) for value in scalars):
            raise RuntimeError("whole-CDS classifier reconstruction became NaN or Inf")

        return WholeCDSClassifierProtectionObjective(
            loss=loss,
            global_reconstruction_loss=global_reconstruction_loss,
            old_class_reconstruction_loss=old_reconstruction_loss,
            new_class_reconstruction_loss=new_reconstruction_loss,
            old_old_retention_loss=old_old_retention_loss,
            historical_identity_margin_mean=historical_margin.mean(),
            historical_identity_margin_min=historical_margin.min(),
            historical_deformation_exposure_mean=historical_exposure.mean(),
            historical_structural_clearance_mean=old_to_new.mean(),
            historical_structural_clearance_min=old_to_new.min(),
            current_identity_margin_mean=current_margin.mean(),
            current_identity_margin_min=current_margin.min(),
            current_deformation_exposure_mean=current_exposure.mean(),
            current_structural_clearance_mean=current_clearance.mean(),
            current_structural_clearance_min=current_clearance.min(),
            historical_internal_clearance_mean=old_internal_clearance[old_internal_mask].mean(),
            historical_internal_clearance_min=old_internal_clearance[old_internal_mask].min(),
            historical_internal_retention_mean=internal_retention[old_internal_mask].mean(),
            historical_internal_retention_min=internal_retention[old_internal_mask].min(),
            historical_internal_violation_fraction=internal_unsafe,
            joint_structural_clearance_mean=joint_clearance.mean(),
            joint_structural_clearance_min=joint_clearance.min(),
            unsafe_pair_fraction=old_new_unsafe,
            pair_weight_mean=cross_weight.mean(),
            learned_pair_risk_mean=cross_learned.mean(),
            physical_pair_risk_mean=cross_physical.mean(),
            effective_pair_risk_mean=cross_effective.mean(),
            old_new_structural_violation_mean=old_new_violation_mean,
            old_new_violation_fraction=old_new_unsafe,
            reference_clearance_mean=reference_floor.mean(),
            reference_clearance_min=reference_floor.min(),
            reference_retention_mean=reference_retention.mean(),
            reference_retention_min=reference_retention.min(),
            reference_violation_fraction=reference_violation,
        )


__all__ = [
    "WholeCDSClassifierProtectionObjective",
    "IncomingCDSInitializationDiagnostics",
    "WholeCDSAdmissionDiagnostics",
    "WholeCDSClassBalancedObjective",
    "initialize_incoming_classifier_rows_from_cds_identity",
    "reconstruct_incoming_classifier_rows_from_whole_cds",
    "whole_cds_class_balanced_classifier_loss",
    "HSICDSDecisionRegionRegulator",
]







# from __future__ import annotations
# from dataclasses import dataclass
# import math

# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# from models.classifier import CosineClassificationHead
# from models.geometry_bank import ClassDiscriminativeStructureState

# Tensor = torch.Tensor


# @dataclass(frozen=True)
# class WholeCDSClassifierProtectionObjective:
#     loss: Tensor
#     global_reconstruction_loss: Tensor
#     old_class_reconstruction_loss: Tensor
#     new_class_reconstruction_loss: Tensor
#     old_old_retention_loss: Tensor

#     historical_identity_margin_mean: Tensor
#     historical_identity_margin_min: Tensor
#     historical_deformation_exposure_mean: Tensor
#     historical_structural_clearance_mean: Tensor
#     historical_structural_clearance_min: Tensor

#     current_identity_margin_mean: Tensor
#     current_identity_margin_min: Tensor
#     current_deformation_exposure_mean: Tensor
#     current_structural_clearance_mean: Tensor
#     current_structural_clearance_min: Tensor

#     historical_internal_clearance_mean: Tensor
#     historical_internal_clearance_min: Tensor
#     historical_internal_retention_mean: Tensor
#     historical_internal_retention_min: Tensor
#     historical_internal_violation_fraction: Tensor

#     joint_structural_clearance_mean: Tensor
#     joint_structural_clearance_min: Tensor
#     unsafe_pair_fraction: Tensor
#     pair_weight_mean: Tensor
#     learned_pair_risk_mean: Tensor
#     physical_pair_risk_mean: Tensor
#     effective_pair_risk_mean: Tensor
#     old_new_structural_violation_mean: Tensor
#     old_new_violation_fraction: Tensor

#     reference_clearance_mean: Tensor
#     reference_clearance_min: Tensor
#     reference_retention_mean: Tensor
#     reference_retention_min: Tensor
#     reference_violation_fraction: Tensor

#     @property
#     def historical_robust_classification_loss(self) -> Tensor:
#         return self.old_class_reconstruction_loss

#     @property
#     def current_robust_structural_loss(self) -> Tensor:
#         return self.new_class_reconstruction_loss


# @dataclass(frozen=True)
# class IncomingCDSInitializationDiagnostics:
#     joint_clearance_min: Tensor
#     joint_clearance_mean: Tensor
#     learned_relevance_mean: Tensor
#     physical_relevance_mean: Tensor
#     effective_relevance_mean: Tensor


# def _validate_seen_cds(
#     *,
#     classifier: CosineClassificationHead,
#     seen_cds: ClassDiscriminativeStructureState,
#     historical_class_count: int,
# ) -> tuple[Tensor, Tensor, Tensor, int, int, int, int]:
#     if not isinstance(classifier, CosineClassificationHead):
#         raise TypeError("classifier must be CosineClassificationHead")
#     if not isinstance(seen_cds, ClassDiscriminativeStructureState):
#         raise TypeError("seen_cds must be ClassDiscriminativeStructureState")

#     classes = seen_cds.classes
#     ids = classes.class_ids.to(device=classifier.weight.device, dtype=torch.long).flatten()
#     count = int(ids.numel())
#     old_count = int(historical_class_count)
#     if count < 2 or old_count <= 0 or old_count >= count:
#         raise ValueError("historical_class_count must split old and new seen classes")
#     if int(torch.unique(ids).numel()) != count:
#         raise ValueError("seen class IDs must be unique")

#     identity = classes.learned_identity.to(
#         device=classifier.weight.device, dtype=classifier.weight.dtype
#     ).detach()
#     variation = classes.learned_variation.to(
#         device=classifier.weight.device, dtype=classifier.weight.dtype
#     ).detach()
#     if identity.ndim != 3 or identity.size(0) != count or identity.size(1) < 2:
#         raise ValueError("I_Q must be [C,K,D], K>=2")
#     c, k, d = identity.shape
#     if variation.shape != (c, k, k, d):
#         raise ValueError("V_Q must be [C,K,K,D]")
#     if not bool(torch.isfinite(identity).all()) or not bool(torch.isfinite(variation).all()):
#         raise ValueError("I_Q/V_Q contains NaN or Inf")
#     if bool((variation < 0).any()):
#         raise ValueError("V_Q must be non-negative")
#     if int(classifier.feature_dim) != k * d:
#         raise ValueError("classifier feature dimension does not match Q")
#     return ids, identity, variation, old_count, count, k, d


# def _historical_cds(
#     *,
#     reference_classifier: CosineClassificationHead,
#     reference_historical_cds: ClassDiscriminativeStructureState,
#     expected_ids: Tensor,
# ) -> tuple[Tensor, Tensor, Tensor]:
#     if not isinstance(reference_classifier, CosineClassificationHead):
#         raise TypeError("reference_classifier must be CosineClassificationHead")
#     if not isinstance(reference_historical_cds, ClassDiscriminativeStructureState):
#         raise TypeError("reference_historical_cds must be ClassDiscriminativeStructureState")
#     ids = reference_historical_cds.classes.class_ids.to(
#         device=reference_classifier.weight.device, dtype=torch.long
#     ).flatten()
#     expected = expected_ids.to(device=ids.device, dtype=torch.long)
#     if not torch.equal(ids, expected):
#         raise ValueError("reference historical CDS IDs disagree with historical prefix")
#     identity = reference_historical_cds.classes.learned_identity.to(
#         device=reference_classifier.weight.device, dtype=reference_classifier.weight.dtype
#     ).detach()
#     variation = reference_historical_cds.classes.learned_variation.to(
#         device=reference_classifier.weight.device, dtype=reference_classifier.weight.dtype
#     ).detach()
#     if identity.ndim != 3 or variation.shape != (
#         identity.size(0), identity.size(1), identity.size(1), identity.size(2)
#     ):
#         raise ValueError("reference I_Q/V_Q shape mismatch")
#     return ids, identity, variation


# def _directed_identity_margin_and_relation_exposure(
#     *,
#     source_identity: Tensor,
#     source_deformation: Tensor,
#     boundary: Tensor,
# ) -> tuple[Tensor, Tensor, Tensor]:
#     """Directed robust cosine clearance defined by complete learned CDS.

#     I_Q supplies class identity. V_Q contributes both diagonal coordinate IQR
#     sensitivity and off-diagonal ordered-token relation IQR sensitivity. V_Q is
#     never interpreted as covariance, a Gaussian, or a hard class box.
#     """
#     identity = torch.as_tensor(source_identity)
#     deformation = torch.as_tensor(source_deformation)
#     normal = torch.as_tensor(boundary)
#     if identity.ndim != 3 or identity.size(0) <= 0 or identity.size(1) < 2:
#         raise ValueError("source_identity must be [S,K,D]")
#     s, k, d = identity.shape
#     if deformation.shape != (s, k, k, d):
#         raise ValueError("source_deformation must be [S,K,K,D]")
#     if normal.ndim != 3 or normal.size(0) != s or normal.size(2) != k * d:
#         raise ValueError("boundary must be [S,R,K*D]")
#     if not all(bool(torch.isfinite(v).all()) for v in (identity, deformation, normal)):
#         raise ValueError("boundary-clearance tensors contain NaN or Inf")

#     eps = torch.finfo(identity.dtype).eps
#     flat_identity = identity.reshape(s, -1)
#     identity_norm = flat_identity.norm(dim=1, keepdim=True).clamp_min(eps)
#     identity_unit = flat_identity / identity_norm
#     margin = (normal * identity_unit[:, None, :]).sum(dim=2)

#     tangent = (
#         normal - margin[:, :, None] * identity_unit[:, None, :]
#     ) / identity_norm[:, None, :]
#     tangent_kd = tangent.reshape(s, normal.size(1), k, d)

#     index = torch.arange(k, device=identity.device)
#     absolute_scale = deformation[:, index, index, :]
#     absolute_exposure = (
#         tangent_kd.abs() * absolute_scale[:, None, :, :]
#     ).sum(dim=(2, 3))

#     relation_sensitivity = (
#         tangent_kd[:, :, :, None, :] - tangent_kd[:, :, None, :, :]
#     ).abs()
#     upper = torch.triu(
#         torch.ones(k, k, device=identity.device, dtype=torch.bool), diagonal=1
#     )
#     relational_exposure = (
#         relation_sensitivity
#         * deformation[:, None, :, :, :]
#         * upper.view(1, 1, k, k, 1)
#     ).sum(dim=(2, 3, 4)) / float(k)

#     exposure = torch.maximum(absolute_exposure, relational_exposure)
#     clearance = margin - exposure
#     if not all(bool(torch.isfinite(v).all()) for v in (margin, exposure, clearance)):
#         raise RuntimeError("whole-CDS boundary clearance became NaN or Inf")
#     return margin, exposure, clearance


# @torch.no_grad()
# def _previous_historical_clearance(
#     *,
#     reference_classifier: CosineClassificationHead,
#     reference_historical_cds: ClassDiscriminativeStructureState,
#     historical_ids: Tensor,
# ) -> tuple[Tensor, Tensor]:
#     ids, identity, variation = _historical_cds(
#         reference_classifier=reference_classifier,
#         reference_historical_cds=reference_historical_cds,
#         expected_ids=historical_ids,
#     )
#     if ids.numel() < 2:
#         raise ValueError("reference historical CDS needs at least two classes")
#     eps = torch.finfo(identity.dtype).eps
#     rows = F.normalize(reference_classifier.class_weight(ids), p=2, dim=1, eps=eps)
#     boundary = rows[:, None, :] - rows[None, :, :]
#     _, _, clearance = _directed_identity_margin_and_relation_exposure(
#         source_identity=identity,
#         source_deformation=variation,
#         boundary=boundary,
#     )
#     diagonal = torch.eye(ids.numel(), device=clearance.device, dtype=torch.bool)
#     floor = clearance.masked_fill(diagonal, torch.inf).min(dim=1).values
#     if not bool(torch.isfinite(clearance).all()) or not bool(torch.isfinite(floor).all()):
#         raise RuntimeError("previous historical clearance became invalid")
#     return clearance.detach(), floor.detach()


# def _effective_pair_risk(learned: Tensor, physical: Tensor) -> Tensor:
#     learned_value = torch.as_tensor(learned).clamp(0.0, 1.0)
#     physical_value = torch.as_tensor(
#         physical, device=learned_value.device, dtype=learned_value.dtype
#     ).clamp(0.0, 1.0)
#     if learned_value.shape != physical_value.shape or learned_value.ndim != 2:
#         raise ValueError("learned/physical risk must share [C,C] or [O,N] shape")
#     if not bool(torch.isfinite(learned_value).all()) or not bool(torch.isfinite(physical_value).all()):
#         raise ValueError("learned/physical risk contains NaN or Inf")
#     return (
#         learned_value
#         + (1.0 - learned_value) * learned_value * physical_value
#     ).clamp(0.0, 1.0)


# def _classwise_robust_loss(
#     *,
#     clearance: Tensor,
#     pair_weight: Tensor,
#     valid_mask: Tensor,
#     scale: Tensor,
# ) -> Tensor:
#     if clearance.shape != pair_weight.shape or clearance.shape != valid_mask.shape:
#         raise ValueError("clearance, pair_weight and valid_mask must share shape")
#     if valid_mask.dtype != torch.bool:
#         raise TypeError("valid_mask must be boolean")
#     eps = torch.finfo(clearance.dtype).eps
#     terms = pair_weight.clamp_min(eps).log() - scale * clearance
#     terms = terms.masked_fill(~valid_mask, -torch.inf)
#     if not bool(valid_mask.any(dim=1).all()):
#         raise RuntimeError("every source class must have at least one rival")
#     zero = torch.zeros((clearance.size(0), 1), device=clearance.device, dtype=clearance.dtype)
#     return torch.logsumexp(torch.cat((zero, terms), dim=1), dim=1)


# @torch.no_grad()
# def initialize_incoming_classifier_rows_from_cds_identity(
#     *,
#     classifier: CosineClassificationHead,
#     seen_cds: ClassDiscriminativeStructureState,
#     historical_class_count: int,
# ) -> IncomingCDSInitializationDiagnostics:
   
#     ids, identity, variation, old_count, count, _k, _d = _validate_seen_cds(
#         classifier=classifier,
#         seen_cds=seen_cds,
#         historical_class_count=historical_class_count,
#     )
#     eps = torch.finfo(identity.dtype).eps
#     unit_identity = F.normalize(identity.reshape(count, -1), p=2, dim=1, eps=eps)
#     new_rows = unit_identity[old_count:]
#     new_ids = ids[old_count:]
#     classifier.weight.index_copy_(
#         0, new_ids, new_rows.to(device=classifier.weight.device, dtype=classifier.weight.dtype)
#     )

#     old_rows = F.normalize(
#         classifier.class_weight(ids[:old_count]), p=2, dim=1, eps=eps
#     ).detach()
#     old_boundary = old_rows[:, None, :] - new_rows[None, :, :]
#     _, _, old_to_new = _directed_identity_margin_and_relation_exposure(
#         source_identity=identity[:old_count],
#         source_deformation=variation[:old_count],
#         boundary=old_boundary,
#     )
#     new_boundary = new_rows[:, None, :] - old_rows[None, :, :]
#     _, _, new_to_old = _directed_identity_margin_and_relation_exposure(
#         source_identity=identity[old_count:],
#         source_deformation=variation[old_count:],
#         boundary=new_boundary,
#     )
#     joint = torch.minimum(old_to_new, new_to_old.transpose(0, 1).contiguous())

#     learned_all = seen_cds.relations.learned_combined_proximity.to(
#         device=identity.device, dtype=identity.dtype
#     ).detach()
#     physical_all = seen_cds.relations.physical_combined_proximity.to(
#         device=identity.device, dtype=identity.dtype
#     ).detach()
#     effective_all = _effective_pair_risk(learned_all, physical_all)
#     cross_learned = learned_all[:old_count, old_count:]
#     cross_physical = physical_all[:old_count, old_count:]
#     cross_effective = effective_all[:old_count, old_count:]

#     if not all(bool(torch.isfinite(v).all()) for v in (new_rows, joint, cross_effective)):
#         raise RuntimeError("incoming CDS identity initialization became NaN or Inf")

#     return IncomingCDSInitializationDiagnostics(
#         joint_clearance_min=joint.min(),
#         joint_clearance_mean=joint.mean(),
#         learned_relevance_mean=cross_learned.mean(),
#         physical_relevance_mean=cross_physical.mean(),
#         effective_relevance_mean=cross_effective.mean(),
#     )


# @dataclass(frozen=True)
# class WholeCDSClassBalancedObjective:
#     loss: Tensor
#     per_class_loss: Tensor
#     clearance_mean: Tensor
#     clearance_min: Tensor
#     learned_pair_risk_mean: Tensor
#     physical_pair_risk_mean: Tensor
#     effective_pair_risk_mean: Tensor


# def whole_cds_class_balanced_classifier_loss(
#     *,
#     classifier: CosineClassificationHead,
#     cds: ClassDiscriminativeStructureState,
# ) -> WholeCDSClassBalancedObjective:
#     """Class-balanced Whole-CDS classifier supervision for a single class set.

#     This is the common classifier-side objective used in Phase 0 and incremental
#     phases.  Each source class contributes equal mass.  I_Q/V_Q define robust
#     directed cosine clearance and I_P/V_P contribute through the same learned +
#     physical inter-class relevance already defined by the CDS bank.
#     """
#     if not isinstance(classifier, CosineClassificationHead):
#         raise TypeError("classifier must be CosineClassificationHead")
#     if not isinstance(cds, ClassDiscriminativeStructureState):
#         raise TypeError("cds must be ClassDiscriminativeStructureState")
#     classes = cds.classes
#     ids = classes.class_ids.to(device=classifier.weight.device, dtype=torch.long).flatten()
#     count = int(ids.numel())
#     if count < 2 or int(torch.unique(ids).numel()) != count:
#         raise ValueError("CDS classifier objective requires at least two unique classes")
#     identity = classes.learned_identity.to(
#         device=classifier.weight.device, dtype=classifier.weight.dtype
#     ).detach()
#     variation = classes.learned_variation.to(
#         device=classifier.weight.device, dtype=classifier.weight.dtype
#     ).detach()
#     if identity.ndim != 3 or variation.shape != (count, identity.size(1), identity.size(1), identity.size(2)):
#         raise ValueError("I_Q/V_Q shape mismatch")
#     eps = torch.finfo(identity.dtype).eps
#     rows = F.normalize(classifier.class_weight(ids), p=2, dim=1, eps=eps)
#     boundary = rows[:, None, :] - rows[None, :, :]
#     _, _, clearance = _directed_identity_margin_and_relation_exposure(
#         source_identity=identity, source_deformation=variation, boundary=boundary
#     )
#     learned = cds.relations.learned_combined_proximity.to(
#         device=identity.device, dtype=identity.dtype
#     ).detach()
#     physical = cds.relations.physical_combined_proximity.to(
#         device=identity.device, dtype=identity.dtype
#     ).detach()
#     effective = _effective_pair_risk(learned, physical)
#     pair_weight = 1.0 + effective
#     valid = ~torch.eye(count, device=identity.device, dtype=torch.bool)
#     scale = classifier.logit_scale.to(device=identity.device, dtype=identity.dtype).detach()
#     per_class = _classwise_robust_loss(
#         clearance=clearance, pair_weight=pair_weight, valid_mask=valid, scale=scale
#     )
#     loss = per_class.mean()
#     offdiag = valid
#     scalars=(loss, clearance[offdiag].mean(), clearance[offdiag].min(),
#              learned[offdiag].mean(), physical[offdiag].mean(), effective[offdiag].mean())
#     if not all(bool(torch.isfinite(v)) for v in scalars):
#         raise RuntimeError("whole-CDS class-balanced objective became NaN or Inf")
#     return WholeCDSClassBalancedObjective(
#         loss=loss, per_class_loss=per_class,
#         clearance_mean=clearance[offdiag].mean(), clearance_min=clearance[offdiag].min(),
#         learned_pair_risk_mean=learned[offdiag].mean(),
#         physical_pair_risk_mean=physical[offdiag].mean(),
#         effective_pair_risk_mean=effective[offdiag].mean(),
#     )


# class HSICDSDecisionRegionRegulator(nn.Module):
#     """Class-balanced all-seen classifier reconstruction from the whole CDS."""

#     def __init__(self) -> None:
#         super().__init__()

#     @property
#     def parameter_count(self) -> int:
#         return 0

#     def forward(
#         self,
#         *,
#         classifier: CosineClassificationHead,
#         seen_cds: ClassDiscriminativeStructureState,
#         historical_class_count: int,
#         reference_classifier: CosineClassificationHead | None = None,
#         reference_historical_cds: ClassDiscriminativeStructureState | None = None,
#     ) -> WholeCDSClassifierProtectionObjective:
#         ids, identity, variation, old_count, count, _k, _d = _validate_seen_cds(
#             classifier=classifier,
#             seen_cds=seen_cds,
#             historical_class_count=historical_class_count,
#         )
#         new_count = count - old_count
#         eps = torch.finfo(identity.dtype).eps
#         rows = classifier.class_weight(ids)
#         unit_rows = F.normalize(rows, p=2, dim=1, eps=eps)
#         old_rows = unit_rows[:old_count]
#         new_rows = unit_rows[old_count:]

#         relations = seen_cds.relations
#         learned_all = relations.learned_combined_proximity.to(
#             device=identity.device, dtype=identity.dtype
#         ).detach()
#         physical_all = relations.physical_combined_proximity.to(
#             device=identity.device, dtype=identity.dtype
#         ).detach()
#         if learned_all.shape != (count, count) or physical_all.shape != (count, count):
#             raise RuntimeError("whole-CDS relation matrices must be [C,C]")
#         effective_all = _effective_pair_risk(learned_all, physical_all)
#         pair_weight_all = 1.0 + effective_all

#         scale = classifier.logit_scale.to(device=identity.device, dtype=identity.dtype).detach()
#         if scale.ndim != 0 or not bool(torch.isfinite(scale)) or not bool(scale > 0):
#             raise RuntimeError("cosine logit scale is invalid")

#         # Historical source classes provide genuine historical positive evidence.
#         # Their positive rows and all seen rivals remain live.
#         old_boundary_all = old_rows[:, None, :] - unit_rows[None, :, :]
#         old_margin_all, old_exposure_all, old_clearance_all = (
#             _directed_identity_margin_and_relation_exposure(
#                 source_identity=identity[:old_count],
#                 source_deformation=variation[:old_count],
#                 boundary=old_boundary_all,
#             )
#         )
#         old_mask = torch.ones(
#             old_count, count, device=identity.device, dtype=torch.bool
#         )
#         old_index = torch.arange(old_count, device=identity.device)
#         old_mask[old_index, old_index] = False
#         old_per_class = _classwise_robust_loss(
#             clearance=old_clearance_all,
#             pair_weight=pair_weight_all[:old_count],
#             valid_mask=old_mask,
#             scale=scale,
#         )
#         old_reconstruction_loss = old_per_class.mean()

#         # Current-source CDS is real current-class structure. It may update incoming
#         # rows, but old rival rows are detached so current-only evidence cannot
#         # rewrite historical rows. New-vs-new boundaries remain fully trainable.
#         new_old_boundary = new_rows[:, None, :] - old_rows.detach()[None, :, :]
#         new_old_margin, new_old_exposure, new_old_clearance = (
#             _directed_identity_margin_and_relation_exposure(
#                 source_identity=identity[old_count:],
#                 source_deformation=variation[old_count:],
#                 boundary=new_old_boundary,
#             )
#         )
#         new_new_boundary = new_rows[:, None, :] - new_rows[None, :, :]
#         new_new_margin, new_new_exposure, new_new_clearance = (
#             _directed_identity_margin_and_relation_exposure(
#                 source_identity=identity[old_count:],
#                 source_deformation=variation[old_count:],
#                 boundary=new_new_boundary,
#             )
#         )
#         new_margin_all = torch.cat((new_old_margin, new_new_margin), dim=1)
#         new_exposure_all = torch.cat((new_old_exposure, new_new_exposure), dim=1)
#         new_clearance_all = torch.cat((new_old_clearance, new_new_clearance), dim=1)
#         new_mask = torch.ones(
#             new_count, count, device=identity.device, dtype=torch.bool
#         )
#         new_index = torch.arange(new_count, device=identity.device)
#         new_mask[new_index, old_count + new_index] = False
#         new_per_class = _classwise_robust_loss(
#             clearance=new_clearance_all,
#             pair_weight=pair_weight_all[old_count:],
#             valid_mask=new_mask,
#             scale=scale,
#         )
#         new_reconstruction_loss = new_per_class.mean()

#         global_reconstruction_loss = (
#             old_per_class.sum() + new_per_class.sum()
#         ) / float(count)

#         # Previous old-old robust clearance is the only historical classifier target
#         # that existed before the current phase and is therefore retained exactly.
#         old_internal_clearance = old_clearance_all[:, :old_count]
#         old_internal_mask = ~torch.eye(
#             old_count, device=identity.device, dtype=torch.bool
#         )
#         if reference_classifier is not None or reference_historical_cds is not None:
#             if reference_classifier is None or reference_historical_cds is None:
#                 raise ValueError("reference classifier and historical CDS must be supplied together")
#             reference_internal, reference_floor = _previous_historical_clearance(
#                 reference_classifier=reference_classifier,
#                 reference_historical_cds=reference_historical_cds,
#                 historical_ids=ids[:old_count],
#             )
#             reference_internal = reference_internal.to(
#                 device=identity.device, dtype=identity.dtype
#             )
#             reference_floor = reference_floor.to(
#                 device=identity.device, dtype=identity.dtype
#             )
#             internal_retention = old_internal_clearance - reference_internal
#         else:
#             reference_internal = identity.new_zeros((old_count, old_count))
#             reference_floor = identity.new_zeros((old_count,))
#             internal_retention = old_internal_clearance

#         internal_violation = F.relu(-internal_retention)[old_internal_mask]
#         if internal_violation.numel():
#             old_old_retention_loss = (
#                 torch.logsumexp(scale * internal_violation, dim=0)
#                 - math.log(float(internal_violation.numel()))
#             ).clamp_min(0.0)
#         else:
#             old_old_retention_loss = old_internal_clearance.sum() * 0.0

#         loss = global_reconstruction_loss + old_old_retention_loss

#         # Cross-phase diagnostics use both directions of the same whole-CDS
#         # clearance. They are diagnostics, not a second objective.
#         old_to_new = old_clearance_all[:, old_count:]
#         new_to_old = new_old_clearance.transpose(0, 1).contiguous()
#         joint_clearance = torch.minimum(old_to_new, new_to_old)
#         cross_learned = learned_all[:old_count, old_count:]
#         cross_physical = physical_all[:old_count, old_count:]
#         cross_effective = effective_all[:old_count, old_count:]
#         cross_weight = 1.0 + cross_effective
#         old_new_violation = F.relu(-joint_clearance)
#         old_new_violation_mean = (
#             (cross_weight * old_new_violation).sum()
#             / cross_weight.sum().clamp_min(eps)
#         )
#         old_new_unsafe = joint_clearance.lt(0.0).to(identity.dtype).mean()

#         historical_margin = old_margin_all[:, old_count:]
#         historical_exposure = old_exposure_all[:, old_count:]
#         current_margin = new_old_margin.transpose(0, 1).contiguous()
#         current_exposure = new_old_exposure.transpose(0, 1).contiguous()
#         current_clearance = new_old_clearance.transpose(0, 1).contiguous()

#         reference_retention = old_to_new - reference_floor[:, None]
#         reference_violation = reference_retention.lt(0.0).to(identity.dtype).mean()
#         internal_unsafe = internal_retention[old_internal_mask].lt(0.0).to(identity.dtype).mean()

#         scalars = (
#             loss,
#             global_reconstruction_loss,
#             old_reconstruction_loss,
#             new_reconstruction_loss,
#             old_old_retention_loss,
#             historical_margin.mean(), historical_margin.min(),
#             historical_exposure.mean(), old_to_new.mean(), old_to_new.min(),
#             current_margin.mean(), current_margin.min(), current_exposure.mean(),
#             current_clearance.mean(), current_clearance.min(),
#             old_internal_clearance[old_internal_mask].mean(),
#             old_internal_clearance[old_internal_mask].min(),
#             internal_retention[old_internal_mask].mean(),
#             internal_retention[old_internal_mask].min(), internal_unsafe,
#             joint_clearance.mean(), joint_clearance.min(), cross_weight.mean(),
#             cross_learned.mean(), cross_physical.mean(), cross_effective.mean(),
#             old_new_violation_mean, old_new_unsafe,
#             reference_floor.mean(), reference_floor.min(),
#             reference_retention.mean(), reference_retention.min(), reference_violation,
#         )
#         if not all(bool(torch.isfinite(value)) for value in scalars):
#             raise RuntimeError("whole-CDS classifier reconstruction became NaN or Inf")

#         return WholeCDSClassifierProtectionObjective(
#             loss=loss,
#             global_reconstruction_loss=global_reconstruction_loss,
#             old_class_reconstruction_loss=old_reconstruction_loss,
#             new_class_reconstruction_loss=new_reconstruction_loss,
#             old_old_retention_loss=old_old_retention_loss,
#             historical_identity_margin_mean=historical_margin.mean(),
#             historical_identity_margin_min=historical_margin.min(),
#             historical_deformation_exposure_mean=historical_exposure.mean(),
#             historical_structural_clearance_mean=old_to_new.mean(),
#             historical_structural_clearance_min=old_to_new.min(),
#             current_identity_margin_mean=current_margin.mean(),
#             current_identity_margin_min=current_margin.min(),
#             current_deformation_exposure_mean=current_exposure.mean(),
#             current_structural_clearance_mean=current_clearance.mean(),
#             current_structural_clearance_min=current_clearance.min(),
#             historical_internal_clearance_mean=old_internal_clearance[old_internal_mask].mean(),
#             historical_internal_clearance_min=old_internal_clearance[old_internal_mask].min(),
#             historical_internal_retention_mean=internal_retention[old_internal_mask].mean(),
#             historical_internal_retention_min=internal_retention[old_internal_mask].min(),
#             historical_internal_violation_fraction=internal_unsafe,
#             joint_structural_clearance_mean=joint_clearance.mean(),
#             joint_structural_clearance_min=joint_clearance.min(),
#             unsafe_pair_fraction=old_new_unsafe,
#             pair_weight_mean=cross_weight.mean(),
#             learned_pair_risk_mean=cross_learned.mean(),
#             physical_pair_risk_mean=cross_physical.mean(),
#             effective_pair_risk_mean=cross_effective.mean(),
#             old_new_structural_violation_mean=old_new_violation_mean,
#             old_new_violation_fraction=old_new_unsafe,
#             reference_clearance_mean=reference_floor.mean(),
#             reference_clearance_min=reference_floor.min(),
#             reference_retention_mean=reference_retention.mean(),
#             reference_retention_min=reference_retention.min(),
#             reference_violation_fraction=reference_violation,
#         )


# __all__ = [
#     "WholeCDSClassifierProtectionObjective",
#     "IncomingCDSInitializationDiagnostics",
#     "WholeCDSClassBalancedObjective",
#     "initialize_incoming_classifier_rows_from_cds_identity",
#     "whole_cds_class_balanced_classifier_loss",
#     "HSICDSDecisionRegionRegulator",
# ]




# from __future__ import annotations

# """Whole-CDS classifier reconstruction for exemplar-free HSI CIL.

# One persistent cosine classifier is used in every phase. Persistent class memory is
# exactly {I_Q, V_Q, I_P, V_P, N}. This module never creates a second classifier,
# replay buffer, Gaussian/covariance model, or synthetic feature distribution.

# Incremental classifier evidence is separated by provenance:
#   * real current labels: handled by the trainer; all seen rows compete, but only
#     incoming rows may receive label gradients;
#   * historical CDS: old-source robust class evidence may update historical positive
#     rows and their seen rivals;
#   * current CDS: new-source structural evidence may update incoming rows, while old
#     rival rows are detached because current-only evidence must not rewrite old rows;
#   * old-old retention: previously achieved historical clearances are preserved.

# The classifier reconstruction itself is class-balanced over ALL seen CDS classes.
# I_Q and V_Q define robust directed cosine clearances. I_P and V_P define the
# phase-stable physical pair relevance that reinforces learned CDS confusability.
# """

# from dataclasses import dataclass
# import math

# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# from models.classifier import CosineClassificationHead
# from models.geometry_bank import ClassDiscriminativeStructureState

# Tensor = torch.Tensor


# @dataclass(frozen=True)
# class WholeCDSClassifierProtectionObjective:
#     loss: Tensor
#     global_reconstruction_loss: Tensor
#     old_class_reconstruction_loss: Tensor
#     new_class_reconstruction_loss: Tensor
#     old_old_retention_loss: Tensor

#     historical_identity_margin_mean: Tensor
#     historical_identity_margin_min: Tensor
#     historical_deformation_exposure_mean: Tensor
#     historical_structural_clearance_mean: Tensor
#     historical_structural_clearance_min: Tensor

#     current_identity_margin_mean: Tensor
#     current_identity_margin_min: Tensor
#     current_deformation_exposure_mean: Tensor
#     current_structural_clearance_mean: Tensor
#     current_structural_clearance_min: Tensor

#     historical_internal_clearance_mean: Tensor
#     historical_internal_clearance_min: Tensor
#     historical_internal_retention_mean: Tensor
#     historical_internal_retention_min: Tensor
#     historical_internal_violation_fraction: Tensor

#     joint_structural_clearance_mean: Tensor
#     joint_structural_clearance_min: Tensor
#     unsafe_pair_fraction: Tensor
#     pair_weight_mean: Tensor
#     learned_pair_risk_mean: Tensor
#     physical_pair_risk_mean: Tensor
#     effective_pair_risk_mean: Tensor
#     old_new_structural_violation_mean: Tensor
#     old_new_violation_fraction: Tensor

#     reference_clearance_mean: Tensor
#     reference_clearance_min: Tensor
#     reference_retention_mean: Tensor
#     reference_retention_min: Tensor
#     reference_violation_fraction: Tensor

#     @property
#     def historical_robust_classification_loss(self) -> Tensor:
#         return self.old_class_reconstruction_loss

#     @property
#     def current_robust_structural_loss(self) -> Tensor:
#         return self.new_class_reconstruction_loss


# @dataclass(frozen=True)
# class IncomingWholeCDSCalibrationDiagnostics:
#     raw_joint_clearance_min: Tensor
#     final_joint_clearance_min: Tensor
#     raw_joint_clearance_mean: Tensor
#     final_joint_clearance_mean: Tensor
#     accepted_fraction: Tensor
#     correction_norm_mean: Tensor
#     learned_relevance_mean: Tensor
#     physical_relevance_mean: Tensor
#     effective_relevance_mean: Tensor
#     support_mass_mean: Tensor


# def _validate_seen_cds(
#     *,
#     classifier: CosineClassificationHead,
#     seen_cds: ClassDiscriminativeStructureState,
#     historical_class_count: int,
# ) -> tuple[Tensor, Tensor, Tensor, int, int, int, int]:
#     if not isinstance(classifier, CosineClassificationHead):
#         raise TypeError("classifier must be CosineClassificationHead")
#     if not isinstance(seen_cds, ClassDiscriminativeStructureState):
#         raise TypeError("seen_cds must be ClassDiscriminativeStructureState")

#     classes = seen_cds.classes
#     ids = classes.class_ids.to(device=classifier.weight.device, dtype=torch.long).flatten()
#     count = int(ids.numel())
#     old_count = int(historical_class_count)
#     if count < 2 or old_count <= 0 or old_count >= count:
#         raise ValueError("historical_class_count must split old and new seen classes")
#     if int(torch.unique(ids).numel()) != count:
#         raise ValueError("seen class IDs must be unique")

#     identity = classes.learned_identity.to(
#         device=classifier.weight.device, dtype=classifier.weight.dtype
#     ).detach()
#     variation = classes.learned_variation.to(
#         device=classifier.weight.device, dtype=classifier.weight.dtype
#     ).detach()
#     if identity.ndim != 3 or identity.size(0) != count or identity.size(1) < 2:
#         raise ValueError("I_Q must be [C,K,D], K>=2")
#     c, k, d = identity.shape
#     if variation.shape != (c, k, k, d):
#         raise ValueError("V_Q must be [C,K,K,D]")
#     if not bool(torch.isfinite(identity).all()) or not bool(torch.isfinite(variation).all()):
#         raise ValueError("I_Q/V_Q contains NaN or Inf")
#     if bool((variation < 0).any()):
#         raise ValueError("V_Q must be non-negative")
#     if int(classifier.feature_dim) != k * d:
#         raise ValueError("classifier feature dimension does not match Q")
#     return ids, identity, variation, old_count, count, k, d


# def _historical_cds(
#     *,
#     reference_classifier: CosineClassificationHead,
#     reference_historical_cds: ClassDiscriminativeStructureState,
#     expected_ids: Tensor,
# ) -> tuple[Tensor, Tensor, Tensor]:
#     if not isinstance(reference_classifier, CosineClassificationHead):
#         raise TypeError("reference_classifier must be CosineClassificationHead")
#     if not isinstance(reference_historical_cds, ClassDiscriminativeStructureState):
#         raise TypeError("reference_historical_cds must be ClassDiscriminativeStructureState")
#     ids = reference_historical_cds.classes.class_ids.to(
#         device=reference_classifier.weight.device, dtype=torch.long
#     ).flatten()
#     expected = expected_ids.to(device=ids.device, dtype=torch.long)
#     if not torch.equal(ids, expected):
#         raise ValueError("reference historical CDS IDs disagree with historical prefix")
#     identity = reference_historical_cds.classes.learned_identity.to(
#         device=reference_classifier.weight.device, dtype=reference_classifier.weight.dtype
#     ).detach()
#     variation = reference_historical_cds.classes.learned_variation.to(
#         device=reference_classifier.weight.device, dtype=reference_classifier.weight.dtype
#     ).detach()
#     if identity.ndim != 3 or variation.shape != (
#         identity.size(0), identity.size(1), identity.size(1), identity.size(2)
#     ):
#         raise ValueError("reference I_Q/V_Q shape mismatch")
#     return ids, identity, variation


# def _directed_identity_margin_and_relation_exposure(
#     *,
#     source_identity: Tensor,
#     source_deformation: Tensor,
#     boundary: Tensor,
# ) -> tuple[Tensor, Tensor, Tensor]:
#     """Directed robust cosine clearance defined by complete learned CDS.

#     I_Q supplies class identity. V_Q contributes both diagonal coordinate IQR
#     sensitivity and off-diagonal ordered-token relation IQR sensitivity. V_Q is
#     never interpreted as covariance, a Gaussian, or a hard class box.
#     """
#     identity = torch.as_tensor(source_identity)
#     deformation = torch.as_tensor(source_deformation)
#     normal = torch.as_tensor(boundary)
#     if identity.ndim != 3 or identity.size(0) <= 0 or identity.size(1) < 2:
#         raise ValueError("source_identity must be [S,K,D]")
#     s, k, d = identity.shape
#     if deformation.shape != (s, k, k, d):
#         raise ValueError("source_deformation must be [S,K,K,D]")
#     if normal.ndim != 3 or normal.size(0) != s or normal.size(2) != k * d:
#         raise ValueError("boundary must be [S,R,K*D]")
#     if not all(bool(torch.isfinite(v).all()) for v in (identity, deformation, normal)):
#         raise ValueError("boundary-clearance tensors contain NaN or Inf")

#     eps = torch.finfo(identity.dtype).eps
#     flat_identity = identity.reshape(s, -1)
#     identity_norm = flat_identity.norm(dim=1, keepdim=True).clamp_min(eps)
#     identity_unit = flat_identity / identity_norm
#     margin = (normal * identity_unit[:, None, :]).sum(dim=2)

#     tangent = (
#         normal - margin[:, :, None] * identity_unit[:, None, :]
#     ) / identity_norm[:, None, :]
#     tangent_kd = tangent.reshape(s, normal.size(1), k, d)

#     index = torch.arange(k, device=identity.device)
#     absolute_scale = deformation[:, index, index, :]
#     absolute_exposure = (
#         tangent_kd.abs() * absolute_scale[:, None, :, :]
#     ).sum(dim=(2, 3))

#     relation_sensitivity = (
#         tangent_kd[:, :, :, None, :] - tangent_kd[:, :, None, :, :]
#     ).abs()
#     upper = torch.triu(
#         torch.ones(k, k, device=identity.device, dtype=torch.bool), diagonal=1
#     )
#     relational_exposure = (
#         relation_sensitivity
#         * deformation[:, None, :, :, :]
#         * upper.view(1, 1, k, k, 1)
#     ).sum(dim=(2, 3, 4)) / float(k)

#     exposure = torch.maximum(absolute_exposure, relational_exposure)
#     clearance = margin - exposure
#     if not all(bool(torch.isfinite(v).all()) for v in (margin, exposure, clearance)):
#         raise RuntimeError("whole-CDS boundary clearance became NaN or Inf")
#     return margin, exposure, clearance


# @torch.no_grad()
# def _previous_historical_clearance(
#     *,
#     reference_classifier: CosineClassificationHead,
#     reference_historical_cds: ClassDiscriminativeStructureState,
#     historical_ids: Tensor,
# ) -> tuple[Tensor, Tensor]:
#     ids, identity, variation = _historical_cds(
#         reference_classifier=reference_classifier,
#         reference_historical_cds=reference_historical_cds,
#         expected_ids=historical_ids,
#     )
#     if ids.numel() < 2:
#         raise ValueError("reference historical CDS needs at least two classes")
#     eps = torch.finfo(identity.dtype).eps
#     rows = F.normalize(reference_classifier.class_weight(ids), p=2, dim=1, eps=eps)
#     boundary = rows[:, None, :] - rows[None, :, :]
#     _, _, clearance = _directed_identity_margin_and_relation_exposure(
#         source_identity=identity,
#         source_deformation=variation,
#         boundary=boundary,
#     )
#     diagonal = torch.eye(ids.numel(), device=clearance.device, dtype=torch.bool)
#     floor = clearance.masked_fill(diagonal, torch.inf).min(dim=1).values
#     if not bool(torch.isfinite(clearance).all()) or not bool(torch.isfinite(floor).all()):
#         raise RuntimeError("previous historical clearance became invalid")
#     return clearance.detach(), floor.detach()


# def _effective_pair_risk(learned: Tensor, physical: Tensor) -> Tensor:
#     learned_value = torch.as_tensor(learned).clamp(0.0, 1.0)
#     physical_value = torch.as_tensor(
#         physical, device=learned_value.device, dtype=learned_value.dtype
#     ).clamp(0.0, 1.0)
#     if learned_value.shape != physical_value.shape or learned_value.ndim != 2:
#         raise ValueError("learned/physical risk must share [C,C] or [O,N] shape")
#     if not bool(torch.isfinite(learned_value).all()) or not bool(torch.isfinite(physical_value).all()):
#         raise ValueError("learned/physical risk contains NaN or Inf")
#     return (
#         learned_value
#         + (1.0 - learned_value) * learned_value * physical_value
#     ).clamp(0.0, 1.0)


# def _classwise_robust_loss(
#     *,
#     clearance: Tensor,
#     pair_weight: Tensor,
#     valid_mask: Tensor,
#     scale: Tensor,
# ) -> Tensor:
#     if clearance.shape != pair_weight.shape or clearance.shape != valid_mask.shape:
#         raise ValueError("clearance, pair_weight and valid_mask must share shape")
#     if valid_mask.dtype != torch.bool:
#         raise TypeError("valid_mask must be boolean")
#     eps = torch.finfo(clearance.dtype).eps
#     terms = pair_weight.clamp_min(eps).log() - scale * clearance
#     terms = terms.masked_fill(~valid_mask, -torch.inf)
#     if not bool(valid_mask.any(dim=1).all()):
#         raise RuntimeError("every source class must have at least one rival")
#     zero = torch.zeros((clearance.size(0), 1), device=clearance.device, dtype=clearance.dtype)
#     return torch.logsumexp(torch.cat((zero, terms), dim=1), dim=1)


# @torch.no_grad()
# def initialize_incoming_classifier_rows_from_whole_cds(
#     *,
#     classifier: CosineClassificationHead,
#     seen_cds: ClassDiscriminativeStructureState,
#     historical_class_count: int,
# ) -> IncomingWholeCDSCalibrationDiagnostics:
#     """One-time deterministic incoming-row imprinting from the complete CDS.

#     Raw current I_Q supplies the incoming semantic direction. The mismatch between
#     finalized historical classifier rows and historical I_Q defines a classifier-
#     coordinate correction. That correction is transferred only through whole-CDS
#     old/new relevance R(I_Q,V_Q,I_P,V_P). For each incoming class, raw I_Q and the
#     transferred row are compared using the exact bidirectional robust old/new CDS
#     criterion; the safer candidate is selected. No iterative solver is used.
#     """
#     ids, identity, variation, old_count, count, _k, _d = _validate_seen_cds(
#         classifier=classifier,
#         seen_cds=seen_cds,
#         historical_class_count=historical_class_count,
#     )
#     new_count = count - old_count
#     eps = torch.finfo(identity.dtype).eps
#     unit_identity = F.normalize(identity.reshape(count, -1), p=2, dim=1, eps=eps)
#     old_identity = unit_identity[:old_count]
#     new_identity = unit_identity[old_count:]
#     old_rows = F.normalize(
#         classifier.class_weight(ids[:old_count]), p=2, dim=1, eps=eps
#     ).detach()

#     relations = seen_cds.relations
#     learned_all = relations.learned_combined_proximity.to(
#         device=identity.device, dtype=identity.dtype
#     ).detach()
#     physical_all = relations.physical_combined_proximity.to(
#         device=identity.device, dtype=identity.dtype
#     ).detach()
#     if learned_all.shape != (count, count) or physical_all.shape != (count, count):
#         raise RuntimeError("seen CDS relation matrices have invalid shape")
#     effective_all = _effective_pair_risk(learned_all, physical_all)
#     cross_learned = learned_all[:old_count, old_count:]
#     cross_physical = physical_all[:old_count, old_count:]
#     cross_effective = effective_all[:old_count, old_count:]

#     column_sum = cross_effective.sum(dim=0, keepdim=True)
#     normalized = torch.where(
#         column_sum > eps,
#         cross_effective / column_sum.clamp_min(eps),
#         torch.zeros_like(cross_effective),
#     )
#     confidence = cross_effective.max(dim=0, keepdim=True).values
#     transfer_weight = normalized * confidence
#     support_mass = transfer_weight.sum(dim=0)

#     old_coordinate_correction = old_rows - old_identity
#     incoming_correction = transfer_weight.transpose(0, 1) @ old_coordinate_correction
#     transferred = F.normalize(new_identity + incoming_correction, p=2, dim=1, eps=eps)

#     scale = classifier.logit_scale.to(device=identity.device, dtype=identity.dtype).detach()
#     old_identity_tensor = identity[:old_count]
#     old_variation = variation[:old_count]
#     new_identity_tensor = identity[old_count:]
#     new_variation = variation[old_count:]
#     pair_weight = 1.0 + cross_effective

#     def _candidate_joint(candidate: Tensor) -> Tensor:
#         candidate = F.normalize(candidate, p=2, dim=1, eps=eps)
#         old_boundary = old_rows[:, None, :] - candidate[None, :, :]
#         _, _, old_to_new = _directed_identity_margin_and_relation_exposure(
#             source_identity=old_identity_tensor,
#             source_deformation=old_variation,
#             boundary=old_boundary,
#         )
#         new_boundary = candidate[:, None, :] - old_rows[None, :, :]
#         _, _, new_to_old = _directed_identity_margin_and_relation_exposure(
#             source_identity=new_identity_tensor,
#             source_deformation=new_variation,
#             boundary=new_boundary,
#         )
#         return torch.minimum(old_to_new, new_to_old.transpose(0, 1).contiguous())

#     raw_joint = _candidate_joint(new_identity)
#     transferred_joint = _candidate_joint(transferred)

#     def _per_new_loss(joint: Tensor) -> Tensor:
#         terms = pair_weight.clamp_min(eps).log() - scale * joint
#         zero = torch.zeros((1, new_count), device=identity.device, dtype=identity.dtype)
#         return torch.logsumexp(torch.cat((zero, terms), dim=0), dim=0)

#     raw_loss = _per_new_loss(raw_joint)
#     transferred_loss = _per_new_loss(transferred_joint)
#     use_transfer = transferred_loss <= raw_loss
#     final_rows = torch.where(use_transfer[:, None], transferred, new_identity)
#     final_joint = _candidate_joint(final_rows)

#     new_ids = ids[old_count:]
#     classifier.weight.index_copy_(
#         0,
#         new_ids,
#         final_rows.to(device=classifier.weight.device, dtype=classifier.weight.dtype),
#     )

#     if not all(
#         bool(torch.isfinite(value).all())
#         for value in (raw_joint, final_joint, final_rows, cross_effective, support_mass)
#     ):
#         raise RuntimeError("whole-CDS incoming-row initialization became NaN or Inf")

#     return IncomingWholeCDSCalibrationDiagnostics(
#         raw_joint_clearance_min=raw_joint.min(),
#         final_joint_clearance_min=final_joint.min(),
#         raw_joint_clearance_mean=raw_joint.mean(),
#         final_joint_clearance_mean=final_joint.mean(),
#         accepted_fraction=use_transfer.to(identity.dtype).mean(),
#         correction_norm_mean=(final_rows - new_identity).norm(dim=1).mean(),
#         learned_relevance_mean=cross_learned.mean(),
#         physical_relevance_mean=cross_physical.mean(),
#         effective_relevance_mean=cross_effective.mean(),
#         support_mass_mean=support_mass.mean(),
#     )


# class HSICDSDecisionRegionRegulator(nn.Module):
#     """Class-balanced all-seen classifier reconstruction from the whole CDS."""

#     def __init__(self) -> None:
#         super().__init__()

#     @property
#     def parameter_count(self) -> int:
#         return 0

#     def forward(
#         self,
#         *,
#         classifier: CosineClassificationHead,
#         seen_cds: ClassDiscriminativeStructureState,
#         historical_class_count: int,
#         reference_classifier: CosineClassificationHead | None = None,
#         reference_historical_cds: ClassDiscriminativeStructureState | None = None,
#     ) -> WholeCDSClassifierProtectionObjective:
#         ids, identity, variation, old_count, count, _k, _d = _validate_seen_cds(
#             classifier=classifier,
#             seen_cds=seen_cds,
#             historical_class_count=historical_class_count,
#         )
#         new_count = count - old_count
#         eps = torch.finfo(identity.dtype).eps
#         rows = classifier.class_weight(ids)
#         unit_rows = F.normalize(rows, p=2, dim=1, eps=eps)
#         old_rows = unit_rows[:old_count]
#         new_rows = unit_rows[old_count:]

#         relations = seen_cds.relations
#         learned_all = relations.learned_combined_proximity.to(
#             device=identity.device, dtype=identity.dtype
#         ).detach()
#         physical_all = relations.physical_combined_proximity.to(
#             device=identity.device, dtype=identity.dtype
#         ).detach()
#         if learned_all.shape != (count, count) or physical_all.shape != (count, count):
#             raise RuntimeError("whole-CDS relation matrices must be [C,C]")
#         effective_all = _effective_pair_risk(learned_all, physical_all)
#         pair_weight_all = 1.0 + effective_all

#         scale = classifier.logit_scale.to(device=identity.device, dtype=identity.dtype).detach()
#         if scale.ndim != 0 or not bool(torch.isfinite(scale)) or not bool(scale > 0):
#             raise RuntimeError("cosine logit scale is invalid")

#         # Historical source classes provide genuine historical positive evidence.
#         # Their positive rows and all seen rivals remain live.
#         old_boundary_all = old_rows[:, None, :] - unit_rows[None, :, :]
#         old_margin_all, old_exposure_all, old_clearance_all = (
#             _directed_identity_margin_and_relation_exposure(
#                 source_identity=identity[:old_count],
#                 source_deformation=variation[:old_count],
#                 boundary=old_boundary_all,
#             )
#         )
#         old_mask = torch.ones(
#             old_count, count, device=identity.device, dtype=torch.bool
#         )
#         old_index = torch.arange(old_count, device=identity.device)
#         old_mask[old_index, old_index] = False
#         old_per_class = _classwise_robust_loss(
#             clearance=old_clearance_all,
#             pair_weight=pair_weight_all[:old_count],
#             valid_mask=old_mask,
#             scale=scale,
#         )
#         old_reconstruction_loss = old_per_class.mean()

#         # Current-source CDS is real current-class structure. It may update incoming
#         # rows, but old rival rows are detached so current-only evidence cannot
#         # rewrite historical rows. New-vs-new boundaries remain fully trainable.
#         new_old_boundary = new_rows[:, None, :] - old_rows.detach()[None, :, :]
#         new_old_margin, new_old_exposure, new_old_clearance = (
#             _directed_identity_margin_and_relation_exposure(
#                 source_identity=identity[old_count:],
#                 source_deformation=variation[old_count:],
#                 boundary=new_old_boundary,
#             )
#         )
#         new_new_boundary = new_rows[:, None, :] - new_rows[None, :, :]
#         new_new_margin, new_new_exposure, new_new_clearance = (
#             _directed_identity_margin_and_relation_exposure(
#                 source_identity=identity[old_count:],
#                 source_deformation=variation[old_count:],
#                 boundary=new_new_boundary,
#             )
#         )
#         new_margin_all = torch.cat((new_old_margin, new_new_margin), dim=1)
#         new_exposure_all = torch.cat((new_old_exposure, new_new_exposure), dim=1)
#         new_clearance_all = torch.cat((new_old_clearance, new_new_clearance), dim=1)
#         new_mask = torch.ones(
#             new_count, count, device=identity.device, dtype=torch.bool
#         )
#         new_index = torch.arange(new_count, device=identity.device)
#         new_mask[new_index, old_count + new_index] = False
#         new_per_class = _classwise_robust_loss(
#             clearance=new_clearance_all,
#             pair_weight=pair_weight_all[old_count:],
#             valid_mask=new_mask,
#             scale=scale,
#         )
#         new_reconstruction_loss = new_per_class.mean()

#         global_reconstruction_loss = (
#             old_per_class.sum() + new_per_class.sum()
#         ) / float(count)

#         # Previous old-old robust clearance is the only historical classifier target
#         # that existed before the current phase and is therefore retained exactly.
#         old_internal_clearance = old_clearance_all[:, :old_count]
#         old_internal_mask = ~torch.eye(
#             old_count, device=identity.device, dtype=torch.bool
#         )
#         if reference_classifier is not None or reference_historical_cds is not None:
#             if reference_classifier is None or reference_historical_cds is None:
#                 raise ValueError("reference classifier and historical CDS must be supplied together")
#             reference_internal, reference_floor = _previous_historical_clearance(
#                 reference_classifier=reference_classifier,
#                 reference_historical_cds=reference_historical_cds,
#                 historical_ids=ids[:old_count],
#             )
#             reference_internal = reference_internal.to(
#                 device=identity.device, dtype=identity.dtype
#             )
#             reference_floor = reference_floor.to(
#                 device=identity.device, dtype=identity.dtype
#             )
#             internal_retention = old_internal_clearance - reference_internal
#         else:
#             reference_internal = identity.new_zeros((old_count, old_count))
#             reference_floor = identity.new_zeros((old_count,))
#             internal_retention = old_internal_clearance

#         internal_violation = F.relu(-internal_retention)[old_internal_mask]
#         if internal_violation.numel():
#             old_old_retention_loss = (
#                 torch.logsumexp(scale * internal_violation, dim=0)
#                 - math.log(float(internal_violation.numel()))
#             ).clamp_min(0.0)
#         else:
#             old_old_retention_loss = old_internal_clearance.sum() * 0.0

#         loss = global_reconstruction_loss + old_old_retention_loss

#         # Cross-phase diagnostics use both directions of the same whole-CDS
#         # clearance. They are diagnostics, not a second objective.
#         old_to_new = old_clearance_all[:, old_count:]
#         new_to_old = new_old_clearance.transpose(0, 1).contiguous()
#         joint_clearance = torch.minimum(old_to_new, new_to_old)
#         cross_learned = learned_all[:old_count, old_count:]
#         cross_physical = physical_all[:old_count, old_count:]
#         cross_effective = effective_all[:old_count, old_count:]
#         cross_weight = 1.0 + cross_effective
#         old_new_violation = F.relu(-joint_clearance)
#         old_new_violation_mean = (
#             (cross_weight * old_new_violation).sum()
#             / cross_weight.sum().clamp_min(eps)
#         )
#         old_new_unsafe = joint_clearance.lt(0.0).to(identity.dtype).mean()

#         historical_margin = old_margin_all[:, old_count:]
#         historical_exposure = old_exposure_all[:, old_count:]
#         current_margin = new_old_margin.transpose(0, 1).contiguous()
#         current_exposure = new_old_exposure.transpose(0, 1).contiguous()
#         current_clearance = new_old_clearance.transpose(0, 1).contiguous()

#         reference_retention = old_to_new - reference_floor[:, None]
#         reference_violation = reference_retention.lt(0.0).to(identity.dtype).mean()
#         internal_unsafe = internal_retention[old_internal_mask].lt(0.0).to(identity.dtype).mean()

#         scalars = (
#             loss,
#             global_reconstruction_loss,
#             old_reconstruction_loss,
#             new_reconstruction_loss,
#             old_old_retention_loss,
#             historical_margin.mean(), historical_margin.min(),
#             historical_exposure.mean(), old_to_new.mean(), old_to_new.min(),
#             current_margin.mean(), current_margin.min(), current_exposure.mean(),
#             current_clearance.mean(), current_clearance.min(),
#             old_internal_clearance[old_internal_mask].mean(),
#             old_internal_clearance[old_internal_mask].min(),
#             internal_retention[old_internal_mask].mean(),
#             internal_retention[old_internal_mask].min(), internal_unsafe,
#             joint_clearance.mean(), joint_clearance.min(), cross_weight.mean(),
#             cross_learned.mean(), cross_physical.mean(), cross_effective.mean(),
#             old_new_violation_mean, old_new_unsafe,
#             reference_floor.mean(), reference_floor.min(),
#             reference_retention.mean(), reference_retention.min(), reference_violation,
#         )
#         if not all(bool(torch.isfinite(value)) for value in scalars):
#             raise RuntimeError("whole-CDS classifier reconstruction became NaN or Inf")

#         return WholeCDSClassifierProtectionObjective(
#             loss=loss,
#             global_reconstruction_loss=global_reconstruction_loss,
#             old_class_reconstruction_loss=old_reconstruction_loss,
#             new_class_reconstruction_loss=new_reconstruction_loss,
#             old_old_retention_loss=old_old_retention_loss,
#             historical_identity_margin_mean=historical_margin.mean(),
#             historical_identity_margin_min=historical_margin.min(),
#             historical_deformation_exposure_mean=historical_exposure.mean(),
#             historical_structural_clearance_mean=old_to_new.mean(),
#             historical_structural_clearance_min=old_to_new.min(),
#             current_identity_margin_mean=current_margin.mean(),
#             current_identity_margin_min=current_margin.min(),
#             current_deformation_exposure_mean=current_exposure.mean(),
#             current_structural_clearance_mean=current_clearance.mean(),
#             current_structural_clearance_min=current_clearance.min(),
#             historical_internal_clearance_mean=old_internal_clearance[old_internal_mask].mean(),
#             historical_internal_clearance_min=old_internal_clearance[old_internal_mask].min(),
#             historical_internal_retention_mean=internal_retention[old_internal_mask].mean(),
#             historical_internal_retention_min=internal_retention[old_internal_mask].min(),
#             historical_internal_violation_fraction=internal_unsafe,
#             joint_structural_clearance_mean=joint_clearance.mean(),
#             joint_structural_clearance_min=joint_clearance.min(),
#             unsafe_pair_fraction=old_new_unsafe,
#             pair_weight_mean=cross_weight.mean(),
#             learned_pair_risk_mean=cross_learned.mean(),
#             physical_pair_risk_mean=cross_physical.mean(),
#             effective_pair_risk_mean=cross_effective.mean(),
#             old_new_structural_violation_mean=old_new_violation_mean,
#             old_new_violation_fraction=old_new_unsafe,
#             reference_clearance_mean=reference_floor.mean(),
#             reference_clearance_min=reference_floor.min(),
#             reference_retention_mean=reference_retention.mean(),
#             reference_retention_min=reference_retention.min(),
#             reference_violation_fraction=reference_violation,
#         )


# __all__ = [
#     "WholeCDSClassifierProtectionObjective",
#     "IncomingWholeCDSCalibrationDiagnostics",
#     "initialize_incoming_classifier_rows_from_whole_cds",
#     "HSICDSDecisionRegionRegulator",
# ]
