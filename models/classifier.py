from __future__ import annotations
from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.geometry_bank import (
    ClassDiscriminativeStructureState,
    combine_learned_physical_relevance,
    normalize_ordered_decision_features,
    ordered_cds_support_components,
)

Tensor = torch.Tensor


@dataclass(frozen=True)
class ClassificationOutput:
    logits: Tensor
    class_ids: Tensor
    prediction: Tensor
    probabilities: Tensor
    cosine_scores: Tensor
    pairwise_margin: Tensor
    # Read-only Whole-CDS diagnostics; inference uses the pure cosine head only.
    learned_support: Tensor
    physical_support: Tensor
    whole_support: Tensor

    @property
    def decision_support(self) -> Tensor:
        return self.cosine_scores

    @property
    def complete_support(self) -> Tensor:
        return self.whole_support

    @property
    def cosine_similarity(self) -> Tensor:
        return self.cosine_scores


def _validate_class_ids(
    values: Tensor | Sequence[int],
    *,
    total_class_count: int,
    device: torch.device,
    allow_empty: bool = False,
) -> Tensor:
    ids = torch.as_tensor(values, device=device, dtype=torch.long).flatten()
    if ids.ndim != 1 or (ids.numel() == 0 and not allow_empty):
        raise ValueError("class_ids must be a non-empty rank-one sequence")
    if ids.numel() == 0:
        return ids
    if ids.unique().numel() != ids.numel():
        raise ValueError("class_ids must be unique")
    if bool((ids < 0).any()) or bool((ids >= int(total_class_count)).any()):
        raise ValueError("class_ids fall outside classifier capacity")
    return ids


def flatten_ordered_features(ordered_features: Tensor) -> Tensor:
    features = torch.as_tensor(ordered_features)
    if (
        features.ndim != 3
        or features.size(0) == 0
        or features.size(1) < 2
        or features.size(2) <= 0
    ):
        raise ValueError("ordered_features must be non-empty [N,K,D], K>=2")
    if not torch.is_floating_point(features) or features.is_complex():
        raise TypeError("ordered_features must be real floating point")
    if not bool(torch.isfinite(features).all()):
        raise ValueError("ordered_features contain NaN or Inf")
    return features.reshape(features.size(0), -1)


def _select_cds_classes(
    cds: ClassDiscriminativeStructureState,
    class_ids: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    if not isinstance(cds, ClassDiscriminativeStructureState):
        raise TypeError("cds must be ClassDiscriminativeStructureState")
    stored_ids = cds.classes.class_ids.to(
        device=class_ids.device, dtype=torch.long
    ).flatten()
    matches = class_ids.unsqueeze(1).eq(stored_ids.unsqueeze(0))
    if not bool(matches.sum(dim=1).eq(1).all()):
        raise ValueError("requested class_ids are absent or duplicated in CDS")
    index = matches.to(torch.long).argmax(dim=1)
    classes = cds.classes
    return (
        classes.learned_identity.index_select(0, index),
        classes.learned_variation.index_select(0, index),
        classes.physical_identity.index_select(0, index),
        classes.physical_variation.index_select(0, index),
    )


class CosineClassificationHead(nn.Module):
    """Single live all-seen cosine head.

    ``class_ids`` controls which rows compete in a forward pass and
    ``trainable_class_ids`` controls gradient ownership. Incremental Stage 1 keeps
    this head frozen while Q is learned; after final Whole-CDS synchronization,
    Stage 3 jointly reconstructs all seen rows from one class-balanced structural
    objective. Deployment is exactly tau*cosine(Q,W); no phase-bias state exists.
    """

    def __init__(
        self,
        *,
        spectral_token_count: int,
        spectral_token_dim: int,
        total_class_count: int,
    ) -> None:
        super().__init__()
        self.spectral_token_count = int(spectral_token_count)
        self.spectral_token_dim = int(spectral_token_dim)
        self.total_class_count = int(total_class_count)
        if self.spectral_token_count < 2 or self.spectral_token_dim <= 0:
            raise ValueError("invalid ordered Q dimensions")
        if self.total_class_count < 2:
            raise ValueError("total_class_count must be at least 2")

        self.feature_dim = self.spectral_token_count * self.spectral_token_dim
        self.weight = nn.Parameter(
            torch.empty(self.total_class_count, self.feature_dim)
        )
        self.logit_scale_parameter = nn.Parameter(torch.zeros(()))
        nn.init.trunc_normal_(self.weight, std=0.02)

    @property
    def parameter_count(self) -> int:
        return int(self.weight.numel() + self.logit_scale_parameter.numel())

    @property
    def logit_scale(self) -> Tensor:
        value = self.logit_scale_parameter.exp()
        if not bool(torch.isfinite(value)) or not bool(value > 0):
            raise RuntimeError("cosine logit scale became invalid")
        return value

    def class_weight(self, class_ids: Tensor | Sequence[int]) -> Tensor:
        ids = _validate_class_ids(
            class_ids,
            total_class_count=self.total_class_count,
            device=self.weight.device,
        )
        return self.weight.index_select(0, ids)

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ) -> None:
        # Older checkpoints may contain the removed phase_calibration_offset buffer.
        # Drop it before strict loading; the finalized head has no phase-bias state.
        state_dict.pop(prefix + "phase_calibration_offset", None)
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def cosine_scores(
        self,
        ordered_features: Tensor,
        *,
        class_ids: Tensor | Sequence[int],
        trainable_class_ids: Tensor | Sequence[int] | None = None,
    ) -> Tensor:
        features = flatten_ordered_features(ordered_features)
        ids = _validate_class_ids(
            class_ids,
            total_class_count=self.total_class_count,
            device=features.device,
        )
        if self.weight.device != features.device or self.weight.dtype != features.dtype:
            raise ValueError("classifier and Q must share device/dtype")

        selected_weight = self.weight.index_select(0, ids)
        if trainable_class_ids is not None:
            trainable_ids = _validate_class_ids(
                trainable_class_ids,
                total_class_count=self.total_class_count,
                device=features.device,
                allow_empty=True,
            )
            if trainable_ids.numel() == 0:
                selected_weight = selected_weight.detach()
            else:
                is_trainable = (
                    ids.unsqueeze(1).eq(trainable_ids.unsqueeze(0)).any(dim=1)
                )
                selected_weight = torch.where(
                    is_trainable.unsqueeze(1),
                    selected_weight,
                    selected_weight.detach(),
                )

        eps = torch.finfo(features.dtype).eps
        feature_unit = F.normalize(features, p=2, dim=1, eps=eps)
        weight_unit = F.normalize(selected_weight, p=2, dim=1, eps=eps)
        scores = F.linear(feature_unit, weight_unit).clamp(-1.0, 1.0)
        if not bool(torch.isfinite(scores).all()):
            raise RuntimeError("cosine classifier produced NaN or Inf")
        return scores

    def _diagnostic_supports(
        self,
        ordered_features: Tensor,
        *,
        class_ids: Tensor,
        physical_spectral_evidence: Tensor | None,
        cds: ClassDiscriminativeStructureState | None,
        cosine_scores: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        # If no CDS is supplied, keep compatibility diagnostics without changing
        # the decision rule.
        if cds is None or physical_spectral_evidence is None:
            learned = ((cosine_scores + 1.0) * 0.5).clamp(0.0, 1.0)
            physical = torch.ones_like(learned)
            return learned, physical, learned

        q = normalize_ordered_decision_features(torch.as_tensor(ordered_features))
        p = torch.as_tensor(physical_spectral_evidence, device=q.device, dtype=q.dtype)
        if p.ndim != 3 or p.size(0) != q.size(0) or p.size(2) != 1:
            raise ValueError("physical_spectral_evidence must be [N,B,1]")
        if not bool(torch.isfinite(p).all()):
            raise ValueError("physical_spectral_evidence contains NaN or Inf")

        i_q, v_q, i_p, v_p = _select_cds_classes(cds, class_ids)
        i_q = i_q.to(device=q.device, dtype=q.dtype)
        v_q = v_q.to(device=q.device, dtype=q.dtype)
        i_p = i_p.to(device=q.device, dtype=q.dtype)
        v_p = v_p.to(device=q.device, dtype=q.dtype)
        if p.shape[1:] != tuple(i_p.shape[1:]):
            raise ValueError("P and physical CDS have incompatible shapes")

        _, _, learned = ordered_cds_support_components(q, i_q, v_q)
        _, _, physical = ordered_cds_support_components(p, i_p, v_p)
        whole = combine_learned_physical_relevance(learned, physical)
        return learned, physical, whole

    def forward(
        self,
        ordered_features: Tensor,
        *,
        class_ids: Tensor | Sequence[int],
        trainable_class_ids: Tensor | Sequence[int] | None = None,
        physical_spectral_evidence: Tensor | None = None,
        cds: ClassDiscriminativeStructureState | None = None,
    ) -> ClassificationOutput:
        ids = _validate_class_ids(
            class_ids,
            total_class_count=self.total_class_count,
            device=torch.as_tensor(ordered_features).device,
        )
        cosine = self.cosine_scores(
            ordered_features,
            class_ids=ids,
            trainable_class_ids=trainable_class_ids,
        )
        scale = self.logit_scale.to(device=cosine.device, dtype=cosine.dtype)
        logits = cosine * scale
        if not bool(torch.isfinite(logits).all()):
            raise RuntimeError("classifier produced NaN or Inf")

        learned_support, physical_support, whole_support = self._diagnostic_supports(
            ordered_features,
            class_ids=ids,
            physical_spectral_evidence=physical_spectral_evidence,
            cds=cds,
            cosine_scores=cosine,
        )
        prediction = ids.index_select(0, logits.argmax(dim=1))
        probabilities = torch.softmax(logits, dim=1)
        pairwise_margin = logits.unsqueeze(2) - logits.unsqueeze(1)
        return ClassificationOutput(
            logits=logits,
            class_ids=ids,
            prediction=prediction,
            probabilities=probabilities,
            cosine_scores=cosine,
            pairwise_margin=pairwise_margin,
            learned_support=learned_support,
            physical_support=physical_support,
            whole_support=whole_support,
        )


__all__ = [
    "ClassificationOutput",
    "CosineClassificationHead",
    "flatten_ordered_features",
]


# from __future__ import annotations
# from dataclasses import dataclass
# from typing import Sequence

# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# from models.geometry_bank import (
#     ClassDiscriminativeStructureState,
#     combine_learned_physical_relevance,
#     normalize_ordered_decision_features,
#     ordered_cds_support_components,
# )

# Tensor = torch.Tensor


# @dataclass(frozen=True)
# class ClassificationOutput:
#     logits: Tensor
#     class_ids: Tensor
#     prediction: Tensor
#     probabilities: Tensor
#     cosine_scores: Tensor
#     pairwise_margin: Tensor
#     # Read-only Whole-CDS diagnostics; inference itself remains cosine(Q,W).
#     learned_support: Tensor
#     physical_support: Tensor
#     whole_support: Tensor

#     @property
#     def decision_support(self) -> Tensor:
#         return self.cosine_scores

#     @property
#     def complete_support(self) -> Tensor:
#         return self.whole_support

#     @property
#     def cosine_similarity(self) -> Tensor:
#         return self.cosine_scores


# def _validate_class_ids(
#     values: Tensor | Sequence[int],
#     *,
#     total_class_count: int,
#     device: torch.device,
#     allow_empty: bool = False,
# ) -> Tensor:
#     ids = torch.as_tensor(values, device=device, dtype=torch.long).flatten()
#     if ids.ndim != 1 or (ids.numel() == 0 and not allow_empty):
#         raise ValueError("class_ids must be a non-empty rank-one sequence")
#     if ids.numel() == 0:
#         return ids
#     if ids.unique().numel() != ids.numel():
#         raise ValueError("class_ids must be unique")
#     if bool((ids < 0).any()) or bool((ids >= int(total_class_count)).any()):
#         raise ValueError("class_ids fall outside classifier capacity")
#     return ids


# def flatten_ordered_features(ordered_features: Tensor) -> Tensor:
#     features = torch.as_tensor(ordered_features)
#     if (
#         features.ndim != 3
#         or features.size(0) == 0
#         or features.size(1) < 2
#         or features.size(2) <= 0
#     ):
#         raise ValueError("ordered_features must be non-empty [N,K,D], K>=2")
#     if not torch.is_floating_point(features) or features.is_complex():
#         raise TypeError("ordered_features must be real floating point")
#     if not bool(torch.isfinite(features).all()):
#         raise ValueError("ordered_features contain NaN or Inf")
#     return features.reshape(features.size(0), -1)


# def _select_cds_classes(
#     cds: ClassDiscriminativeStructureState,
#     class_ids: Tensor,
# ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
#     if not isinstance(cds, ClassDiscriminativeStructureState):
#         raise TypeError("cds must be ClassDiscriminativeStructureState")
#     stored_ids = cds.classes.class_ids.to(device=class_ids.device, dtype=torch.long).flatten()
#     matches = class_ids.unsqueeze(1).eq(stored_ids.unsqueeze(0))
#     if not bool(matches.sum(dim=1).eq(1).all()):
#         raise ValueError("requested class_ids are absent or duplicated in CDS")
#     index = matches.to(torch.long).argmax(dim=1)
#     classes = cds.classes
#     return (
#         classes.learned_identity.index_select(0, index),
#         classes.learned_variation.index_select(0, index),
#         classes.physical_identity.index_select(0, index),
#         classes.physical_variation.index_select(0, index),
#     )


# class CosineClassificationHead(nn.Module):
#     """Bias-free all-seen cosine head trained directly on joint old/new support.

#     ``class_ids`` decides which rows compete in the logits.
#     ``trainable_class_ids`` decides which of those selected rows may receive
#     gradient from this particular forward/loss path.  Detaching a row never
#     removes it from the softmax denominator.
#     """

#     def __init__(
#         self,
#         *,
#         spectral_token_count: int,
#         spectral_token_dim: int,
#         total_class_count: int,
#     ) -> None:
#         super().__init__()
#         self.spectral_token_count = int(spectral_token_count)
#         self.spectral_token_dim = int(spectral_token_dim)
#         self.total_class_count = int(total_class_count)
#         if self.spectral_token_count < 2 or self.spectral_token_dim <= 0:
#             raise ValueError("invalid ordered Q dimensions")
#         if self.total_class_count < 2:
#             raise ValueError("total_class_count must be at least 2")

#         self.feature_dim = self.spectral_token_count * self.spectral_token_dim
#         self.weight = nn.Parameter(torch.empty(self.total_class_count, self.feature_dim))
#         self.logit_scale_parameter = nn.Parameter(torch.zeros(()))
#         nn.init.trunc_normal_(self.weight, std=0.02)

#     @property
#     def parameter_count(self) -> int:
#         return int(self.weight.numel() + self.logit_scale_parameter.numel())

#     @property
#     def logit_scale(self) -> Tensor:
#         value = self.logit_scale_parameter.exp()
#         if not bool(torch.isfinite(value)) or not bool(value > 0):
#             raise RuntimeError("cosine logit scale became invalid")
#         return value

#     def class_weight(self, class_ids: Tensor | Sequence[int]) -> Tensor:
#         ids = _validate_class_ids(
#             class_ids,
#             total_class_count=self.total_class_count,
#             device=self.weight.device,
#         )
#         return self.weight.index_select(0, ids)

#     def cosine_scores(
#         self,
#         ordered_features: Tensor,
#         *,
#         class_ids: Tensor | Sequence[int],
#         trainable_class_ids: Tensor | Sequence[int] | None = None,
#     ) -> Tensor:
#         features = flatten_ordered_features(ordered_features)
#         ids = _validate_class_ids(
#             class_ids,
#             total_class_count=self.total_class_count,
#             device=features.device,
#         )
#         if self.weight.device != features.device or self.weight.dtype != features.dtype:
#             raise ValueError("classifier and Q must share device/dtype")

#         selected_weight = self.weight.index_select(0, ids)
#         if trainable_class_ids is not None:
#             trainable_ids = _validate_class_ids(
#                 trainable_class_ids,
#                 total_class_count=self.total_class_count,
#                 device=features.device,
#                 allow_empty=True,
#             )
#             if trainable_ids.numel() == 0:
#                 selected_weight = selected_weight.detach()
#             else:
#                 is_trainable = ids.unsqueeze(1).eq(trainable_ids.unsqueeze(0)).any(dim=1)
#                 selected_weight = torch.where(
#                     is_trainable.unsqueeze(1),
#                     selected_weight,
#                     selected_weight.detach(),
#                 )

#         eps = torch.finfo(features.dtype).eps
#         feature_unit = F.normalize(features, p=2, dim=1, eps=eps)
#         weight_unit = F.normalize(selected_weight, p=2, dim=1, eps=eps)
#         scores = F.linear(feature_unit, weight_unit).clamp(-1.0, 1.0)
#         if not bool(torch.isfinite(scores).all()):
#             raise RuntimeError("cosine classifier produced NaN or Inf")
#         return scores

#     def _diagnostic_supports(
#         self,
#         ordered_features: Tensor,
#         *,
#         class_ids: Tensor,
#         physical_spectral_evidence: Tensor | None,
#         cds: ClassDiscriminativeStructureState | None,
#         cosine_scores: Tensor,
#     ) -> tuple[Tensor, Tensor, Tensor]:
#         # If no CDS is supplied, keep compatibility diagnostics without changing
#         # the decision rule.
#         if cds is None or physical_spectral_evidence is None:
#             learned = ((cosine_scores + 1.0) * 0.5).clamp(0.0, 1.0)
#             physical = torch.ones_like(learned)
#             return learned, physical, learned

#         q = normalize_ordered_decision_features(torch.as_tensor(ordered_features))
#         p = torch.as_tensor(
#             physical_spectral_evidence, device=q.device, dtype=q.dtype
#         )
#         if p.ndim != 3 or p.size(0) != q.size(0) or p.size(2) != 1:
#             raise ValueError("physical_spectral_evidence must be [N,B,1]")
#         if not bool(torch.isfinite(p).all()):
#             raise ValueError("physical_spectral_evidence contains NaN or Inf")

#         i_q, v_q, i_p, v_p = _select_cds_classes(cds, class_ids)
#         i_q = i_q.to(device=q.device, dtype=q.dtype)
#         v_q = v_q.to(device=q.device, dtype=q.dtype)
#         i_p = i_p.to(device=q.device, dtype=q.dtype)
#         v_p = v_p.to(device=q.device, dtype=q.dtype)
#         if p.shape[1:] != tuple(i_p.shape[1:]):
#             raise ValueError("P and physical CDS have incompatible shapes")

#         _, _, learned = ordered_cds_support_components(q, i_q, v_q)
#         _, _, physical = ordered_cds_support_components(p, i_p, v_p)
#         whole = combine_learned_physical_relevance(learned, physical)
#         return learned, physical, whole

#     def forward(
#         self,
#         ordered_features: Tensor,
#         *,
#         class_ids: Tensor | Sequence[int],
#         trainable_class_ids: Tensor | Sequence[int] | None = None,
#         physical_spectral_evidence: Tensor | None = None,
#         cds: ClassDiscriminativeStructureState | None = None,
#     ) -> ClassificationOutput:
#         ids = _validate_class_ids(
#             class_ids,
#             total_class_count=self.total_class_count,
#             device=torch.as_tensor(ordered_features).device,
#         )
#         cosine = self.cosine_scores(
#             ordered_features,
#             class_ids=ids,
#             trainable_class_ids=trainable_class_ids,
#         )
#         scale = self.logit_scale.to(device=cosine.device, dtype=cosine.dtype)
#         logits = cosine * scale
#         if not bool(torch.isfinite(logits).all()):
#             raise RuntimeError("classifier produced NaN or Inf")

#         learned_support, physical_support, whole_support = self._diagnostic_supports(
#             ordered_features,
#             class_ids=ids,
#             physical_spectral_evidence=physical_spectral_evidence,
#             cds=cds,
#             cosine_scores=cosine,
#         )
#         prediction = ids.index_select(0, logits.argmax(dim=1))
#         probabilities = torch.softmax(logits, dim=1)
#         pairwise_margin = logits.unsqueeze(2) - logits.unsqueeze(1)
#         return ClassificationOutput(
#             logits=logits,
#             class_ids=ids,
#             prediction=prediction,
#             probabilities=probabilities,
#             cosine_scores=cosine,
#             pairwise_margin=pairwise_margin,
#             learned_support=learned_support,
#             physical_support=physical_support,
#             whole_support=whole_support,
#         )


# __all__ = [
#     "ClassificationOutput",
#     "CosineClassificationHead",
#     "flatten_ordered_features",
# ]
