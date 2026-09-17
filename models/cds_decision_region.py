from __future__ import annotations

"""Classifier-side historical CDS decision-region protection for HSI NECIL.

The deployed classifier remains one shared bias-free cosine classifier over the
ordered learned representation Q[K,D].  This module does not initialize,
reconstruct, or post-hoc project classifier rows.

During an incremental phase it has one responsibility:

    keep the previously validated historical classifier rows fixed, while
    preventing newly learned classifier rows from cutting historical CDS regions.

The persistent CDS is {I_Q, V_Q, I_P, V_P, N}.  I_Q supplies historical learned
identity; V_Q supplies direct ordered-relation IQR with an exactly zero diagonal;
I_P and V_P affect only whole-CDS old/current pair risk.  Physical spectral
statistics never get projected into Q-space.

No old image, old feature row, pseudo feature, reconstructed support, second
classifier, task head, or trainable parameter is created here.
"""

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.classifier import SpectralCDSClassifier
from models.geometry_bank import ClassDiscriminativeStructureState, whole_cds_pair_risk

Tensor = torch.Tensor


@dataclass(frozen=True)
class HistoricalCDSClassifierProtectionObjective:
    """Historical-region protection loss plus detached current-side diagnostics."""

    loss: Tensor
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
    unsafe_pair_fraction: Tensor
    pair_weight_mean: Tensor
    learned_region_risk_mean: Tensor
    spectral_region_risk_mean: Tensor
    per_old_class_loss: Tensor


def _validate_seen_cds(
    *,
    classifier: SpectralCDSClassifier,
    seen_cds: ClassDiscriminativeStructureState,
    historical_class_count: int,
) -> tuple[Tensor, Tensor, Tensor, int, int, int, int]:
    """Validate the exact CDS semantics consumed by the regulator.

    The class order is required to be ``[historical..., current...]`` because the
    phase trainer constructs ``seen_cds`` by merging migrated historical CDS first
    and current CDS second.  V_Q is direct ordered-relation IQR only; its diagonal
    must therefore stay exactly zero.
    """
    if not isinstance(classifier, SpectralCDSClassifier):
        raise TypeError("classifier must be SpectralCDSClassifier")
    if not isinstance(seen_cds, ClassDiscriminativeStructureState):
        raise TypeError("seen_cds must be ClassDiscriminativeStructureState")

    classes = seen_cds.classes
    ids = classes.class_ids.to(device=classifier.weight.device, dtype=torch.long).flatten()
    if ids.ndim != 1 or ids.numel() < 2:
        raise ValueError("seen CDS must contain at least two classes")
    if int(torch.unique(ids).numel()) != int(ids.numel()):
        raise ValueError("seen CDS class_ids must be unique")

    count = int(ids.numel())
    old_count = int(historical_class_count)
    if old_count <= 0 or old_count >= count:
        raise ValueError("historical_class_count must split historical and current seen classes")

    identity = classes.identity.to(
        device=classifier.weight.device,
        dtype=classifier.weight.dtype,
    ).detach()
    deformation = classes.variation.to(
        device=classifier.weight.device,
        dtype=classifier.weight.dtype,
    ).detach()

    if identity.ndim != 3 or identity.size(0) != count or identity.size(1) < 2 or identity.size(2) <= 0:
        raise ValueError("I_Q must be [C,K,D] with C matching class_ids and K>=2")
    c, k, d = identity.shape
    if deformation.shape != (c, k, k, d):
        raise ValueError("V_Q must be [C,K,K,D] and match I_Q")
    if not bool(torch.isfinite(identity).all()) or not bool(torch.isfinite(deformation).all()):
        raise ValueError("I_Q/V_Q contains NaN or Inf")
    if bool((deformation < 0).any()):
        raise ValueError("V_Q ordered-relation IQR must be non-negative")

    diagonal = deformation.diagonal(offset=0, dim1=1, dim2=2)
    if bool(torch.count_nonzero(diagonal)):
        raise ValueError("V_Q diagonal must be exactly zero; V_Q stores ordered relations only")

    rows = classifier.class_weight(ids)
    if rows.ndim != 2 or rows.shape != (count, k * d):
        raise ValueError("classifier rows must be [C,K*D] and match the CDS representation")
    if not bool(torch.isfinite(rows).all()):
        raise ValueError("classifier rows contain NaN or Inf")

    return ids, identity, deformation, old_count, count, k, d


def _directed_identity_margin_and_relation_exposure(
    *,
    source_identity: Tensor,
    source_deformation: Tensor,
    boundary: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    """Evaluate directed CDS clearance for actual cosine classifier boundaries.

    ``source_deformation`` is V_Q and is used only in its native ordered-relation
    coordinates.  For boundary B[k,d], relation sensitivity is

        |B[k,d] - B[l,d]|,

    which is paired directly with V_Q[k,l,d].  No covariance, Gaussian support,
    or absolute-coordinate radius is inferred from V_Q.
    """
    identity = torch.as_tensor(source_identity)
    deformation = torch.as_tensor(source_deformation)
    normal = torch.as_tensor(boundary)

    if identity.ndim != 3 or identity.size(0) <= 0 or identity.size(1) < 2:
        raise ValueError("source_identity must be non-empty [S,K,D], K>=2")
    s, k, d = identity.shape
    if deformation.shape != (s, k, k, d):
        raise ValueError("source_deformation must be [S,K,K,D]")
    if normal.ndim != 3 or normal.size(0) != s or normal.size(2) != k * d:
        raise ValueError("boundary must be [S,R,K*D] and match source CDS")
    if any(value.device != identity.device or value.dtype != identity.dtype for value in (deformation, normal)):
        raise ValueError("identity/deformation/boundary must share device and dtype")
    if not all(bool(torch.isfinite(value).all()) for value in (identity, deformation, normal)):
        raise ValueError("decision-region tensors contain NaN or Inf")
    if bool((deformation < 0).any()):
        raise ValueError("V_Q deformation must be non-negative")

    eps = torch.finfo(identity.dtype).eps
    flat_identity = identity.reshape(s, -1)
    identity_norm = flat_identity.norm(dim=1).clamp_min(eps)

    margin = (normal * flat_identity[:, None, :]).sum(dim=2) / identity_norm[:, None]

    normal_kd = normal.reshape(s, normal.size(1), k, d)
    relation_sensitivity = (
        normal_kd[:, :, :, None, :] - normal_kd[:, :, None, :, :]
    ).abs()
    upper = torch.triu(
        torch.ones(k, k, device=identity.device, dtype=torch.bool),
        diagonal=1,
    )
    exposure_numerator = (
        relation_sensitivity
        * deformation[:, None, :, :, :]
        * upper.view(1, 1, k, k, 1)
    ).sum(dim=(2, 3, 4))

    # The 1/K factor is the exact pairwise decomposition factor for zero-token-
    # mean relational perturbations.  Dividing by ||I_Q|| keeps margin and
    # exposure in the same normalized classifier scale without a hand-set margin.
    exposure = exposure_numerator / (float(k) * identity_norm[:, None])
    clearance = margin - exposure

    if not all(bool(torch.isfinite(value).all()) for value in (margin, exposure, clearance)):
        raise RuntimeError("CDS decision-region clearance became NaN or Inf")
    return margin, exposure, clearance


def _priority_weights_from_cds_risk(risk: Tensor) -> Tensor:
    """Prioritize HSI-confusable rivals without changing old-class balance.

    ``whole_cds_pair_risk`` already combines learned CDS risk (I_Q,V_Q) and
    physical spectral risk (I_P,V_P).  ``1 + risk`` keeps every old/current pair
    active.  Row normalization then gives every historical class equal aggregate
    responsibility regardless of its number of rivals or absolute risk scale.
    """
    value = torch.as_tensor(risk)
    if value.ndim != 2 or value.size(0) <= 0 or value.size(1) <= 0:
        raise ValueError("risk must be a non-empty [O,N] matrix")
    if not torch.is_floating_point(value) or value.is_complex() or not bool(torch.isfinite(value).all()):
        raise ValueError("risk must be finite real floating point")
    if bool((value < 0).any()):
        raise ValueError("risk must be non-negative")

    eps = torch.finfo(value.dtype).eps
    weight = 1.0 + value
    return weight / weight.mean(dim=1, keepdim=True).clamp_min(eps)


class HSICDSDecisionRegionRegulator(nn.Module):
    """Protect historical CDS regions by moving only current classifier rows.

    The old classifier rows are normalized and detached before boundary
    construction.  Historical/current CDS tensors are detached.  Therefore the
    only trainable object reachable from ``loss`` is the set of current classifier
    rows.  The backbone is not an input to this module.
    """

    def __init__(self) -> None:
        super().__init__()

    @property
    def parameter_count(self) -> int:
        return 0

    def forward(
        self,
        *,
        classifier: SpectralCDSClassifier,
        seen_cds: ClassDiscriminativeStructureState,
        historical_class_count: int,
    ) -> HistoricalCDSClassifierProtectionObjective:
        (
            ids,
            identity,
            deformation,
            old_count,
            count,
            _k,
            _d,
        ) = _validate_seen_cds(
            classifier=classifier,
            seen_cds=seen_cds,
            historical_class_count=historical_class_count,
        )

        eps = torch.finfo(identity.dtype).eps
        rows = classifier.class_weight(ids)
        unit_rows = F.normalize(rows, p=2, dim=1, eps=eps)
        old_rows = unit_rows[:old_count].detach()
        current_rows = unit_rows[old_count:]

        # Actual deployed old-vs-current cosine boundaries.  Historical rows are
        # fixed references; only current rows are live in this graph.
        historical_boundary = old_rows[:, None, :] - current_rows[None, :, :]
        historical_margin, historical_exposure, historical_clearance = (
            _directed_identity_margin_and_relation_exposure(
                source_identity=identity[:old_count],
                source_deformation=deformation[:old_count],
                boundary=historical_boundary,
            )
        )

        # I_P/V_P never enter Q-space.  They influence classifier learning only
        # through whole-CDS pair risk, which prioritizes physically confusable
        # old/current class pairs together with learned I_Q/V_Q risk.
        risk, learned, spectral = whole_cds_pair_risk(seen_cds)
        risk = risk.to(device=identity.device, dtype=identity.dtype).detach()
        learned = learned.to(device=identity.device, dtype=identity.dtype).detach()
        spectral = spectral.to(device=identity.device, dtype=identity.dtype).detach()
        expected = (count, count)
        if risk.shape != expected or learned.shape != expected or spectral.shape != expected:
            raise RuntimeError("whole-CDS pair-risk matrices must be [C,C]")
        if not all(bool(torch.isfinite(value).all()) for value in (risk, learned, spectral)):
            raise RuntimeError("whole-CDS pair risk contains NaN or Inf")
        if bool((risk < 0).any()) or bool((learned < 0).any()) or bool((spectral < 0).any()):
            raise RuntimeError("whole-CDS pair risk must be non-negative")

        cross_risk = risk[:old_count, old_count:]
        cross_learned = learned[:old_count, old_count:]
        cross_spectral = spectral[:old_count, old_count:]
        if cross_risk.shape != historical_clearance.shape:
            raise RuntimeError("whole-CDS old/current risk disagrees with classifier boundary shape")

        pair_weight = _priority_weights_from_cds_risk(cross_risk)

        # Smooth zero-margin protection.  Real current CE determines where the
        # current row should point; this term only makes cutting a historical CDS
        # region increasingly expensive.  No current-side CDS loss is optimized.
        pair_loss = F.softplus(-historical_clearance)
        per_old = (pair_weight * pair_loss).mean(dim=1)
        loss = per_old.mean()

        # Current-side clearance is diagnostic only.  Detaching the boundary here
        # makes that ownership explicit and prevents the diagnostic path from
        # retaining or contributing classifier gradients.
        with torch.no_grad():
            current_boundary = -historical_boundary.detach().permute(1, 0, 2).contiguous()
            current_margin_raw, current_exposure_raw, current_clearance_raw = (
                _directed_identity_margin_and_relation_exposure(
                    source_identity=identity[old_count:],
                    source_deformation=deformation[old_count:],
                    boundary=current_boundary,
                )
            )
            current_margin = current_margin_raw.transpose(0, 1).contiguous()
            current_exposure = current_exposure_raw.transpose(0, 1).contiguous()
            current_clearance = current_clearance_raw.transpose(0, 1).contiguous()

        unsafe_pair_fraction = historical_clearance.le(0.0).to(identity.dtype).mean()

        scalars = (
            loss,
            historical_margin.mean(),
            historical_margin.min(),
            historical_exposure.mean(),
            historical_clearance.mean(),
            historical_clearance.min(),
            current_margin.mean(),
            current_margin.min(),
            current_exposure.mean(),
            current_clearance.mean(),
            current_clearance.min(),
            unsafe_pair_fraction,
            pair_weight.mean(),
            cross_learned.mean(),
            cross_spectral.mean(),
        )
        if not all(bool(torch.isfinite(value)) for value in scalars):
            raise RuntimeError("historical CDS decision-region objective became NaN or Inf")

        return HistoricalCDSClassifierProtectionObjective(
            loss=loss,
            historical_identity_margin_mean=historical_margin.mean(),
            historical_identity_margin_min=historical_margin.min(),
            historical_deformation_exposure_mean=historical_exposure.mean(),
            historical_structural_clearance_mean=historical_clearance.mean(),
            historical_structural_clearance_min=historical_clearance.min(),
            current_identity_margin_mean=current_margin.mean(),
            current_identity_margin_min=current_margin.min(),
            current_deformation_exposure_mean=current_exposure.mean(),
            current_structural_clearance_mean=current_clearance.mean(),
            current_structural_clearance_min=current_clearance.min(),
            unsafe_pair_fraction=unsafe_pair_fraction,
            pair_weight_mean=pair_weight.mean(),
            learned_region_risk_mean=cross_learned.mean(),
            spectral_region_risk_mean=cross_spectral.mean(),
            per_old_class_loss=per_old,
        )


__all__ = [
    "HistoricalCDSClassifierProtectionObjective",
    "HSICDSDecisionRegionRegulator",
]














# from __future__ import annotations

# """HSI-CDS decision-region regulator for non-exemplar CIL.

# This module is the classifier-side continual-learning component.  The classifier
# itself remains a plain shared cosine decision rule.  Historical HSI structure is
# provided by the persistent CDS {I_Q, V_Q, I_P, V_P, N}; the regulator reads that
# state and the classifier rows, then constrains only newly introduced rows so they
# do not cut historical decision regions.

# No old image, old feature, pseudo feature, reconstructed support, second
# classifier, or task-specific head is created here.  The module owns no trainable
# parameters and stores no class state.
# """

# from dataclasses import dataclass
# from typing import Sequence

# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# from models.classifier import SpectralCDSClassifier
# from models.geometry_bank import ClassDiscriminativeStructureState, whole_cds_pair_risk

# Tensor = torch.Tensor


# @dataclass(frozen=True)
# class HistoricalCDSClassifierProtectionObjective:
#     loss: Tensor
#     historical_side_loss: Tensor
#     current_side_loss: Tensor
#     historical_identity_margin_mean: Tensor
#     historical_identity_margin_min: Tensor
#     current_identity_margin_mean: Tensor
#     current_identity_margin_min: Tensor
#     historical_deformation_exposure_mean: Tensor
#     current_deformation_exposure_mean: Tensor
#     historical_structural_clearance_mean: Tensor
#     historical_structural_clearance_min: Tensor
#     current_structural_clearance_mean: Tensor
#     current_structural_clearance_min: Tensor
#     unsafe_pair_fraction: Tensor
#     pair_weight_mean: Tensor
#     learned_region_risk_mean: Tensor
#     spectral_region_risk_mean: Tensor
#     per_old_class_loss: Tensor

#     # Compatibility aliases for existing report code.
#     @property
#     def historical_variation_extent_mean(self) -> Tensor:
#         return self.historical_deformation_exposure_mean

#     @property
#     def current_variation_extent_mean(self) -> Tensor:
#         return self.current_deformation_exposure_mean

#     @property
#     def historical_boundary_safety_mean(self) -> Tensor:
#         return self.historical_structural_clearance_mean

#     @property
#     def historical_boundary_safety_min(self) -> Tensor:
#         return self.historical_structural_clearance_min

#     @property
#     def current_boundary_safety_mean(self) -> Tensor:
#         return self.current_structural_clearance_mean

#     @property
#     def current_boundary_safety_min(self) -> Tensor:
#         return self.current_structural_clearance_min


# @dataclass(frozen=True)
# class RegionClassifierObjective:
#     loss: Tensor
#     cross_phase_loss: Tensor
#     current_current_loss: Tensor
#     historical_historical_diagnostic: Tensor
#     cross_phase_clearance_mean: Tensor
#     cross_phase_clearance_min: Tensor
#     current_current_clearance_mean: Tensor
#     current_current_clearance_min: Tensor
#     historical_historical_clearance_min: Tensor
#     deformation_exposure_mean: Tensor
#     learned_region_risk_mean: Tensor
#     spectral_region_risk_mean: Tensor
#     pair_weight_mean: Tensor

#     # Compatibility aliases for existing report code.
#     @property
#     def cross_phase_margin_mean(self) -> Tensor:
#         return self.cross_phase_clearance_mean

#     @property
#     def cross_phase_margin_min(self) -> Tensor:
#         return self.cross_phase_clearance_min

#     @property
#     def current_current_margin_mean(self) -> Tensor:
#         return self.current_current_clearance_mean

#     @property
#     def current_current_margin_min(self) -> Tensor:
#         return self.current_current_clearance_min

#     @property
#     def historical_historical_margin_min(self) -> Tensor:
#         return self.historical_historical_clearance_min

#     @property
#     def pooled_extent_mean(self) -> Tensor:
#         return self.deformation_exposure_mean


# def _validate_class_ids(
#     class_ids: Tensor | Sequence[int],
#     *,
#     total_classes: int,
#     device: torch.device,
# ) -> Tensor:
#     ids = torch.as_tensor(class_ids, device=device, dtype=torch.long).flatten()
#     if ids.ndim != 1 or ids.numel() == 0:
#         raise ValueError("class_ids must be non-empty rank-one")
#     if bool((ids < 0).any()) or bool((ids >= int(total_classes)).any()):
#         raise ValueError("class_ids fall outside classifier capacity")
#     if int(torch.unique(ids).numel()) != int(ids.numel()):
#         raise ValueError("class_ids must be unique")
#     return ids


# def _directed_identity_margin_and_deformation_exposure(
#     *,
#     source_identity: Tensor,
#     source_deformation: Tensor,
#     boundary: Tensor,
# ) -> tuple[Tensor, Tensor, Tensor]:
#     """Measure source-class clearance for actual cosine classifier boundaries.

#     I_Q defines the historical class identity. V_Q stores robust magnitudes of
#     ordered spectral-relation deformation. Therefore the boundary is expressed in
#     the same pairwise relation coordinates:

#         G(k,l,d) = |B(k,d) - B(l,d)|

#     and the structural clearance is identity margin minus deformation exposure.
#     V_Q is never treated as covariance or a probability distribution.
#     """
#     identity = torch.as_tensor(source_identity)
#     deformation = torch.as_tensor(source_deformation)
#     normal = torch.as_tensor(boundary)
#     if identity.ndim != 3 or identity.size(0) <= 0 or identity.size(1) < 2:
#         raise ValueError("source_identity must be non-empty [S,K,D], K>=2")
#     s, k, d = identity.shape
#     if deformation.shape != (s, k, k, d):
#         raise ValueError("source_deformation must be [S,K,K,D]")
#     if normal.ndim != 3 or normal.size(0) != s or normal.size(2) != k * d:
#         raise ValueError("boundary must be [S,R,K*D] and match source CDS")
#     if any(v.device != identity.device or v.dtype != identity.dtype for v in (deformation, normal)):
#         raise ValueError("identity/deformation/boundary must share device/dtype")
#     if not all(bool(torch.isfinite(v).all()) for v in (identity, deformation, normal)):
#         raise ValueError("decision-region tensors contain NaN or Inf")
#     if bool((deformation < 0).any()):
#         raise ValueError("V_Q deformation must be non-negative")

#     eps = torch.finfo(identity.dtype).eps
#     flat_identity = identity.reshape(s, -1)
#     identity_norm = flat_identity.norm(dim=1).clamp_min(eps)
#     margin = (normal * flat_identity[:, None, :]).sum(dim=2) / identity_norm[:, None]

#     normal_kd = normal.reshape(s, normal.size(1), k, d)
#     relation_sensitivity = (
#         normal_kd[:, :, :, None, :] - normal_kd[:, :, None, :, :]
#     ).abs()
#     upper = torch.triu(
#         torch.ones(k, k, device=identity.device, dtype=torch.bool),
#         diagonal=1,
#     )
#     exposure_numerator = (
#         relation_sensitivity
#         * deformation[:, None, :, :, :]
#         * upper.view(1, 1, k, k, 1)
#     ).sum(dim=(2, 3, 4))

#     # For zero-token-mean perturbations, <b,delta> decomposes into the pairwise
#     # relation sum with factor 1/K.  Dividing by ||I_Q|| gives margin/exposure the
#     # same scale and introduces no hand-set margin.
#     exposure = exposure_numerator / (float(k) * identity_norm[:, None])
#     clearance = margin - exposure
#     if not all(bool(torch.isfinite(v).all()) for v in (margin, exposure, clearance)):
#         raise RuntimeError("decision-region clearance became NaN or Inf")
#     return margin, exposure, clearance


# def _priority_weights_from_cds_risk(risk: Tensor) -> Tensor:
#     """Keep every old/new boundary active while prioritizing HSI-confusable pairs."""
#     value = torch.as_tensor(risk)
#     if value.ndim != 2 or value.size(0) <= 0 or value.size(1) <= 0:
#         raise ValueError("risk must be a non-empty matrix")
#     if not torch.is_floating_point(value) or value.is_complex() or not bool(torch.isfinite(value).all()):
#         raise ValueError("risk must be finite real floating point")
#     if bool((value < 0).any()):
#         raise ValueError("risk must be non-negative")
#     eps = torch.finfo(value.dtype).eps
#     weight = 1.0 + value
#     return weight / weight.mean(dim=1, keepdim=True).clamp_min(eps)


# class HSICDSDecisionRegionRegulator(nn.Module):
#     """Parameter-free regulator of current classifier rows against historical CDS.

#     The module is active only during incremental training/final refinement.
#     Historical rows and CDS are detached; gradients from this module can reach
#     current classifier rows only.  The backbone is not an input, so the regulator
#     cannot move Q.
#     """

#     def __init__(self) -> None:
#         super().__init__()

#     @property
#     def parameter_count(self) -> int:
#         return 0

#     @torch.no_grad()
#     def initialize_new_rows(
#         self,
#         *,
#         classifier: SpectralCDSClassifier,
#         cds: ClassDiscriminativeStructureState,
#         class_ids: Tensor | Sequence[int],
#     ) -> None:
#         """Initialize current rows from I_Q without making CDS part of the classifier."""
#         if not isinstance(classifier, SpectralCDSClassifier):
#             raise TypeError("classifier must be SpectralCDSClassifier")
#         if not isinstance(cds, ClassDiscriminativeStructureState):
#             raise TypeError("cds must be ClassDiscriminativeStructureState")
#         selected_ids = _validate_class_ids(
#             class_ids,
#             total_classes=classifier.total_class_count,
#             device=classifier.weight.device,
#         )
#         cds_ids = _validate_class_ids(
#             cds.classes.class_ids,
#             total_classes=classifier.total_class_count,
#             device=classifier.weight.device,
#         )
#         matches = selected_ids.unsqueeze(1).eq(cds_ids.unsqueeze(0))
#         if not bool(matches.sum(dim=1).eq(1).all()):
#             raise ValueError("requested initialization class is absent from CDS")
#         rows = matches.to(torch.long).argmax(dim=1)
#         identity = cds.classes.identity.to(
#             device=classifier.weight.device,
#             dtype=classifier.weight.dtype,
#         )
#         directions = F.normalize(identity.reshape(identity.size(0), -1), p=2, dim=1)
#         classifier.weight.index_copy_(0, selected_ids, directions.index_select(0, rows))

#     def forward(
#         self,
#         *,
#         classifier: SpectralCDSClassifier,
#         seen_cds: ClassDiscriminativeStructureState,
#         historical_class_count: int,
#     ) -> HistoricalCDSClassifierProtectionObjective:
#         """Return the historical decision-region regulation objective."""
#         if not isinstance(classifier, SpectralCDSClassifier):
#             raise TypeError("classifier must be SpectralCDSClassifier")
#         if not isinstance(seen_cds, ClassDiscriminativeStructureState):
#             raise TypeError("seen_cds must be ClassDiscriminativeStructureState")

#         classes = seen_cds.classes
#         ids = classes.class_ids.to(device=classifier.weight.device, dtype=torch.long)
#         count = int(ids.numel())
#         old_count = int(historical_class_count)
#         if old_count <= 0 or old_count >= count:
#             raise ValueError("historical_class_count must split all-seen classes")

#         identity = classes.identity.to(
#             device=classifier.weight.device,
#             dtype=classifier.weight.dtype,
#         ).detach()
#         deformation = classes.variation.to(
#             device=classifier.weight.device,
#             dtype=classifier.weight.dtype,
#         ).detach()
#         if identity.ndim != 3:
#             raise ValueError("I_Q must be [C,K,D]")
#         c, k, d = identity.shape
#         if c != count or deformation.shape != (c, k, k, d):
#             raise ValueError("V_Q must be [C,K,K,D] and match I_Q")

#         eps = torch.finfo(identity.dtype).eps
#         rows = classifier.class_weight(ids)
#         unit_rows = F.normalize(rows, p=2, dim=1, eps=eps)
#         old_rows = unit_rows[:old_count].detach()
#         new_rows = unit_rows[old_count:]

#         # Actual deployed old-vs-current cosine boundaries. Only new_rows are live.
#         boundary_old = old_rows[:, None, :] - new_rows[None, :, :]
#         old_margin, old_exposure, old_clearance = _directed_identity_margin_and_deformation_exposure(
#             source_identity=identity[:old_count],
#             source_deformation=deformation[:old_count],
#             boundary=boundary_old,
#         )

#         # Current-side quantities are diagnostic only; CE + separation own current plasticity.
#         boundary_new = -boundary_old.permute(1, 0, 2).contiguous()
#         new_margin_no, new_exposure_no, new_clearance_no = _directed_identity_margin_and_deformation_exposure(
#             source_identity=identity[old_count:],
#             source_deformation=deformation[old_count:],
#             boundary=boundary_new,
#         )
#         current_margin = new_margin_no.transpose(0, 1)
#         current_exposure = new_exposure_no.transpose(0, 1)
#         current_clearance = new_clearance_no.transpose(0, 1)

#         risk, learned, spectral = whole_cds_pair_risk(seen_cds)
#         risk = risk.to(device=identity.device, dtype=identity.dtype)[:old_count, old_count:]
#         learned = learned.to(device=identity.device, dtype=identity.dtype)[:old_count, old_count:]
#         spectral = spectral.to(device=identity.device, dtype=identity.dtype)[:old_count, old_count:]
#         pair_weight = _priority_weights_from_cds_risk(risk)

#         old_pair_loss = F.softplus(-old_clearance)
#         current_pair_loss = F.softplus(-current_clearance)
#         per_old = (pair_weight * old_pair_loss).mean(dim=1)
#         historical_side_loss = per_old.mean()
#         loss = historical_side_loss
#         current_side_loss = (pair_weight * current_pair_loss).mean().detach()
#         unsafe_pair_fraction = old_clearance.le(0.0).to(identity.dtype).mean()

#         values = (
#             loss, old_margin.mean(), old_exposure.mean(), old_clearance.mean(),
#             current_margin.mean(), current_exposure.mean(), current_clearance.mean(),
#             unsafe_pair_fraction, learned.mean(), spectral.mean(), pair_weight.mean(),
#         )
#         if not all(bool(torch.isfinite(value)) for value in values):
#             raise RuntimeError("HSI-CDS decision-region regulation became NaN or Inf")

#         return HistoricalCDSClassifierProtectionObjective(
#             loss=loss,
#             historical_side_loss=historical_side_loss,
#             current_side_loss=current_side_loss,
#             historical_identity_margin_mean=old_margin.mean(),
#             historical_identity_margin_min=old_margin.min(),
#             current_identity_margin_mean=current_margin.mean(),
#             current_identity_margin_min=current_margin.min(),
#             historical_deformation_exposure_mean=old_exposure.mean(),
#             current_deformation_exposure_mean=current_exposure.mean(),
#             historical_structural_clearance_mean=old_clearance.mean(),
#             historical_structural_clearance_min=old_clearance.min(),
#             current_structural_clearance_mean=current_clearance.mean(),
#             current_structural_clearance_min=current_clearance.min(),
#             unsafe_pair_fraction=unsafe_pair_fraction,
#             pair_weight_mean=pair_weight.mean(),
#             learned_region_risk_mean=learned.mean(),
#             spectral_region_risk_mean=spectral.mean(),
#             per_old_class_loss=per_old,
#         )

#     @torch.no_grad()
#     def diagnose(
#         self,
#         *,
#         classifier: SpectralCDSClassifier,
#         cds: ClassDiscriminativeStructureState,
#         historical_class_count: int,
#         optimize_all_pairs: bool = False,
#     ) -> RegionClassifierObjective:
#         """Report the same clearance semantics across all relevant class pairs."""
#         if not isinstance(classifier, SpectralCDSClassifier):
#             raise TypeError("classifier must be SpectralCDSClassifier")
#         if not isinstance(cds, ClassDiscriminativeStructureState):
#             raise TypeError("cds must be ClassDiscriminativeStructureState")
#         classes = cds.classes
#         ids = classes.class_ids.to(device=classifier.weight.device, dtype=torch.long)
#         c = int(ids.numel())
#         old_count = int(historical_class_count)
#         if old_count <= 0 or old_count >= c:
#             raise ValueError("historical_class_count must split all-seen classes")

#         identity = classes.identity.to(device=classifier.weight.device, dtype=classifier.weight.dtype).detach()
#         deformation = classes.variation.to(device=classifier.weight.device, dtype=classifier.weight.dtype).detach()
#         _, k, d = identity.shape
#         eps = torch.finfo(identity.dtype).eps
#         rows = classifier.class_weight(ids)
#         unit_rows = F.normalize(rows, p=2, dim=1, eps=eps)
#         boundary = unit_rows[:, None, :] - unit_rows[None, :, :]
#         margin, exposure, clearance = _directed_identity_margin_and_deformation_exposure(
#             source_identity=identity,
#             source_deformation=deformation,
#             boundary=boundary,
#         )
#         pair_loss = F.softplus(-clearance)
#         risk, learned, spectral = whole_cds_pair_risk(cds)
#         risk = risk.to(device=identity.device, dtype=identity.dtype)
#         learned = learned.to(device=identity.device, dtype=identity.dtype)
#         spectral = spectral.to(device=identity.device, dtype=identity.dtype)
#         priority = 1.0 + risk

#         diagonal = torch.eye(c, device=identity.device, dtype=torch.bool)
#         old_mask = torch.arange(c, device=identity.device) < old_count
#         current_mask = ~old_mask
#         source_current = current_mask.unsqueeze(1).expand(c, c)
#         rival_current = current_mask.unsqueeze(0).expand(c, c)
#         optim_mask = ~diagonal if bool(optimize_all_pairs) else ((~diagonal) & (source_current | rival_current))
#         cross_mask = (~diagonal) & (old_mask.unsqueeze(1) ^ old_mask.unsqueeze(0))
#         current_current_mask = (~diagonal) & source_current & rival_current
#         old_old_mask = (~diagonal) & old_mask.unsqueeze(1) & old_mask.unsqueeze(0)

#         def _weighted(mask: Tensor) -> Tensor:
#             if not bool(mask.any()):
#                 return identity.new_zeros(())
#             w = priority[mask]
#             return (w * pair_loss[mask]).sum() / w.sum().clamp_min(eps)

#         def _mean(mask: Tensor, value: Tensor) -> Tensor:
#             return value[mask].mean() if bool(mask.any()) else identity.new_zeros(())

#         def _min(mask: Tensor, value: Tensor) -> Tensor:
#             return value[mask].min() if bool(mask.any()) else identity.new_zeros(())

#         loss = _weighted(optim_mask)
#         cross = _weighted(cross_mask)
#         current_current = _weighted(current_current_mask)
#         old_old = _weighted(old_old_mask)
#         return RegionClassifierObjective(
#             loss=loss,
#             cross_phase_loss=cross,
#             current_current_loss=current_current,
#             historical_historical_diagnostic=old_old,
#             cross_phase_clearance_mean=_mean(cross_mask, clearance),
#             cross_phase_clearance_min=_min(cross_mask, clearance),
#             current_current_clearance_mean=_mean(current_current_mask, clearance),
#             current_current_clearance_min=_min(current_current_mask, clearance),
#             historical_historical_clearance_min=_min(old_old_mask, clearance),
#             deformation_exposure_mean=_mean(optim_mask, exposure),
#             learned_region_risk_mean=learned[optim_mask].mean(),
#             spectral_region_risk_mean=spectral[optim_mask].mean(),
#             pair_weight_mean=priority[optim_mask].mean(),
#         )


# __all__ = [
#     "HistoricalCDSClassifierProtectionObjective",
#     "RegionClassifierObjective",
#     "HSICDSDecisionRegionRegulator",
# ]








# from __future__ import annotations

# """Classifier-side protection from the complete all-seen HSI CDS.

# Samples are still classified only from the single learned representation Q by
# one shared cosine classifier. The persistent CDS {I_Q,V_Q,I_P,V_P,N} constrains
# the same old/current classifier boundaries during incremental learning.

# For each historical class o and current class n, the same boundary is checked
# from both sides:
#   * old I_Q,V_Q must remain on the historical side;
#   * current I_Q,V_Q must remain on the current side;
#   * I_P,V_P from both classes enter whole_cds_pair_risk and prioritize HSI-
#     confusable pairs.

# Historical rows and all CDS tensors are detached. Only current classifier rows
# receive gradients. No pseudo feature, prototype sample, second head, or trainable
# regulator parameter is introduced.
# """

# from dataclasses import dataclass

# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# from models.classifier import SpectralCDSClassifier
# from models.geometry_bank import (
#     ClassDiscriminativeStructureState,
#     ordered_variation_operator,
#     whole_cds_pair_risk,
# )

# Tensor = torch.Tensor


# @dataclass(frozen=True)
# class HistoricalCDSClassifierProtectionObjective:
#     loss: Tensor
#     historical_identity_margin_mean: Tensor
#     historical_identity_margin_min: Tensor
#     current_identity_margin_mean: Tensor
#     current_identity_margin_min: Tensor
#     historical_relational_exposure_mean: Tensor
#     current_relational_exposure_mean: Tensor
#     historical_structural_clearance_mean: Tensor
#     historical_structural_clearance_min: Tensor
#     current_structural_clearance_mean: Tensor
#     current_structural_clearance_min: Tensor
#     joint_structural_clearance_mean: Tensor
#     joint_structural_clearance_min: Tensor
#     unsafe_pair_fraction: Tensor
#     pair_weight_mean: Tensor
#     learned_region_risk_mean: Tensor
#     spectral_region_risk_mean: Tensor
#     per_old_class_loss: Tensor



# def _validate_relation_variation(identity: Tensor, variation: Tensor) -> tuple[int, int, int]:
#     if identity.ndim != 3 or identity.size(0) <= 0 or identity.size(1) < 2 or identity.size(2) <= 0:
#         raise ValueError("I_Q must be non-empty [C,K,D], K>=2")
#     c, k, d = identity.shape
#     if variation.shape != (c, k, k, d):
#         raise ValueError("V_Q must be [C,K,K,D] and match I_Q")
#     if identity.device != variation.device or identity.dtype != variation.dtype:
#         raise ValueError("I_Q and V_Q must share device/dtype")
#     if not bool(torch.isfinite(identity).all()) or not bool(torch.isfinite(variation).all()):
#         raise ValueError("I_Q/V_Q contains NaN or Inf")
#     if bool((variation < 0).any()):
#         raise ValueError("V_Q must be non-negative")
#     tolerance = 32.0 * torch.finfo(variation.dtype).eps
#     if not bool(torch.allclose(variation, variation.transpose(1, 2), rtol=0.0, atol=tolerance)):
#         raise ValueError("V_Q must be symmetric across ordered spectral pairs")
#     diagonal = variation.diagonal(dim1=1, dim2=2)
#     if bool((diagonal.abs() > tolerance).any()):
#         raise ValueError("V_Q diagonal must be zero under relation-IQR semantics")
#     return c, k, d


# def _directed_relation_clearance(
#     *,
#     identity: Tensor,
#     variation: Tensor,
#     boundary: Tensor,
# ) -> tuple[Tensor, Tensor, Tensor]:
#     """Measure one CDS side against directed cosine boundaries.

#     V_Q[k,l,d] is IQR_x(Q_x[k,d]-Q_x[l,d]); therefore boundary exposure is
#     measured in the same ordered-relation coordinates
#     |B[k,d]-B[l,d]|. I_Q locates the class identity, while V_Q supplies its
#     legitimate relation deformation. V_Q is never treated as covariance or an
#     absolute-coordinate box.
#     """
#     source_count, k, d = _validate_relation_variation(identity, variation)
#     b = torch.as_tensor(boundary)
#     if b.ndim != 3 or b.size(0) != source_count or b.size(2) != k * d:
#         raise ValueError("boundary must be [S,R,K*D] and match source CDS")
#     if b.device != identity.device or b.dtype != identity.dtype:
#         raise ValueError("boundary and source CDS must share device/dtype")
#     if not bool(torch.isfinite(b).all()):
#         raise ValueError("classifier boundary contains NaN or Inf")

#     eps = torch.finfo(identity.dtype).eps
#     flat_identity = identity.reshape(source_count, -1)
#     identity_norm = flat_identity.norm(dim=1).clamp_min(eps)
#     identity_margin = (b * flat_identity[:, None, :]).sum(dim=2) / identity_norm[:, None]

#     boundary_kd = b.reshape(source_count, b.size(1), k, d)
#     sensitivity = (
#         boundary_kd[:, :, :, None, :] - boundary_kd[:, :, None, :, :]
#     ).abs()
#     upper = torch.triu(torch.ones(k, k, device=identity.device, dtype=torch.bool), diagonal=1)
#     exposure = (
#         sensitivity
#         * variation[:, None, :, :, :]
#         * upper.view(1, 1, k, k, 1)
#     ).sum(dim=(2, 3, 4)) / (float(k) * identity_norm[:, None])
#     clearance = identity_margin - exposure

#     if not all(bool(torch.isfinite(v).all()) for v in (identity_margin, exposure, clearance)):
#         raise RuntimeError("CDS classifier clearance became NaN or Inf")
#     return identity_margin, exposure, clearance


# class HSICDSDecisionRegionRegulator(nn.Module):
#     """Constrain each old/current boundary from both complete learned-CDS sides."""

#     def __init__(self) -> None:
#         super().__init__()

#     @property
#     def parameter_count(self) -> int:
#         return 0

#     def forward(
#         self,
#         *,
#         classifier: SpectralCDSClassifier,
#         seen_cds: ClassDiscriminativeStructureState,
#         historical_class_count: int,
#     ) -> HistoricalCDSClassifierProtectionObjective:
#         if not isinstance(classifier, SpectralCDSClassifier):
#             raise TypeError("classifier must be SpectralCDSClassifier")
#         if not isinstance(seen_cds, ClassDiscriminativeStructureState):
#             raise TypeError("seen_cds must be ClassDiscriminativeStructureState")

#         classes = seen_cds.classes
#         ids = classes.class_ids.to(device=classifier.weight.device, dtype=torch.long)
#         count = int(ids.numel())
#         old_count = int(historical_class_count)
#         if old_count <= 0 or old_count >= count:
#             raise ValueError("historical_class_count must split all-seen classes")

#         identity = classes.identity.to(
#             device=classifier.weight.device, dtype=classifier.weight.dtype
#         ).detach()
#         variation = ordered_variation_operator(
#             classes.variation.to(
#                 device=classifier.weight.device, dtype=classifier.weight.dtype
#             )
#         ).detach()
#         _validate_relation_variation(identity, variation)

#         eps = torch.finfo(identity.dtype).eps
#         rows = classifier.class_weight(ids)
#         unit_rows = F.normalize(rows, p=2, dim=1, eps=eps)
#         old_rows = unit_rows[:old_count].detach()
#         current_rows = unit_rows[old_count:]

#         # One deployed boundary, evaluated from both class sides. Only current
#         # rows remain live in autograd; old rows are the fixed historical anchor.
#         old_to_current_boundary = old_rows[:, None, :] - current_rows[None, :, :]
#         old_margin, old_exposure, old_clearance = _directed_relation_clearance(
#             identity=identity[:old_count],
#             variation=variation[:old_count],
#             boundary=old_to_current_boundary,
#         )

#         current_to_old_boundary = -old_to_current_boundary.permute(1, 0, 2).contiguous()
#         current_margin_no, current_exposure_no, current_clearance_no = _directed_relation_clearance(
#             identity=identity[old_count:],
#             variation=variation[old_count:],
#             boundary=current_to_old_boundary,
#         )
#         # Reorient current-side tensors to the common [old,current] pair layout.
#         current_margin = current_margin_no.transpose(0, 1)
#         current_exposure = current_exposure_no.transpose(0, 1)
#         current_clearance = current_clearance_no.transpose(0, 1)

#         risk, learned, spectral = whole_cds_pair_risk(seen_cds)
#         risk = risk.to(device=identity.device, dtype=identity.dtype)[:old_count, old_count:]
#         learned = learned.to(device=identity.device, dtype=identity.dtype)[:old_count, old_count:]
#         spectral = spectral.to(device=identity.device, dtype=identity.dtype)[:old_count, old_count:]
#         if risk.shape != old_clearance.shape or current_clearance.shape != old_clearance.shape:
#             raise RuntimeError("whole-CDS old/current risk shape disagrees with classifier boundaries")
#         if not all(bool(torch.isfinite(v).all()) for v in (risk, learned, spectral)):
#             raise RuntimeError("whole-CDS pair risk contains NaN or Inf")

#         # Every pair remains active. Whole learned+physical CDS only prioritizes
#         # classes that are structurally and spectrally confusable.
#         # Keep every pair active with unit baseline, while allowing complete
#         # learned+physical CDS risk to change the actual gradient magnitude.
#         # Do not renormalize away the absolute HSI risk: for a single old/new
#         # pair, I_P/V_P must still affect classifier protection.
#         pair_weight = 1.0 + risk

#         # One old/current boundary is safe only when BOTH CDS sides clear it.
#         # Optimizing the mean of two side-specific penalties is incorrect because
#         # it can improve the current side while sacrificing the historical side.
#         # The weaker side therefore defines the pair objective directly.
#         joint_clearance = torch.minimum(old_clearance, current_clearance)
#         pair_loss = F.softplus(-joint_clearance)
#         per_old = (pair_weight * pair_loss).mean(dim=1)
#         loss = per_old.mean()
#         unsafe = joint_clearance.le(0.0).to(identity.dtype).mean()

#         scalars = (
#             loss,
#             old_margin.mean(), old_margin.min(),
#             current_margin.mean(), current_margin.min(),
#             old_exposure.mean(), current_exposure.mean(),
#             old_clearance.mean(), old_clearance.min(),
#             current_clearance.mean(), current_clearance.min(),
#             joint_clearance.mean(), joint_clearance.min(), unsafe,
#             pair_weight.mean(), learned.mean(), spectral.mean(),
#         )
#         if not all(bool(torch.isfinite(v)) for v in scalars):
#             raise RuntimeError("joint HSI-CDS classifier objective became NaN or Inf")

#         return HistoricalCDSClassifierProtectionObjective(
#             loss=loss,
#             historical_identity_margin_mean=old_margin.mean(),
#             historical_identity_margin_min=old_margin.min(),
#             current_identity_margin_mean=current_margin.mean(),
#             current_identity_margin_min=current_margin.min(),
#             historical_relational_exposure_mean=old_exposure.mean(),
#             current_relational_exposure_mean=current_exposure.mean(),
#             historical_structural_clearance_mean=old_clearance.mean(),
#             historical_structural_clearance_min=old_clearance.min(),
#             current_structural_clearance_mean=current_clearance.mean(),
#             current_structural_clearance_min=current_clearance.min(),
#             joint_structural_clearance_mean=joint_clearance.mean(),
#             joint_structural_clearance_min=joint_clearance.min(),
#             unsafe_pair_fraction=unsafe,
#             pair_weight_mean=pair_weight.mean(),
#             learned_region_risk_mean=learned.mean(),
#             spectral_region_risk_mean=spectral.mean(),
#             per_old_class_loss=per_old,
#         )


# __all__ = [
#     "HistoricalCDSClassifierProtectionObjective",
#     "HSICDSDecisionRegionRegulator",
# ]

