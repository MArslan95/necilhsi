from __future__ import annotations
from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.geometry_bank import (
    ClassDiscriminativeStructureState,
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
    # Read-only Whole-CDS diagnostics; inference itself remains cosine(Q,W).
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
    stored_ids = cds.classes.class_ids.to(device=class_ids.device, dtype=torch.long).flatten()
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
    """Bias-free all-seen cosine head trained directly on joint old/new support.

    ``class_ids`` decides which rows compete in the logits.
    ``trainable_class_ids`` decides which of those selected rows may receive
    gradient from this particular forward/loss path.  Detaching a row never
    removes it from the softmax denominator.
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
        self.weight = nn.Parameter(torch.empty(self.total_class_count, self.feature_dim))
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
                is_trainable = ids.unsqueeze(1).eq(trainable_ids.unsqueeze(0)).any(dim=1)
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
        p = torch.as_tensor(
            physical_spectral_evidence, device=q.device, dtype=q.dtype
        )
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
        whole = torch.sqrt(
            learned.clamp_min(0.0) * physical.clamp_min(0.0)
        ).clamp(0.0, 1.0)
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
#     # Read-only Whole-CDS diagnostics.  Training reconstructs/protects this same
#     # all-seen cosine head from CDS; inference itself remains cosine(Q,W).
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
#     """Bias-free all-seen cosine head reconstructed/protected by Whole-CDS training.

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
#         whole = torch.sqrt(
#             learned.clamp_min(0.0) * physical.clamp_min(0.0)
#         ).clamp(0.0, 1.0)
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













# from __future__ import annotations
# from dataclasses import dataclass
# from typing import Sequence

# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# from models.geometry_bank import (
#     ClassDiscriminativeStructureState,
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
#     # Read-only Whole-CDS diagnostics.  Training reconstructs/protects this same
#     # all-seen cosine head from CDS; inference itself remains cosine(Q,W).
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
#     """Bias-free all-seen cosine head reconstructed/protected by Whole-CDS training.

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
#         whole = torch.sqrt(
#             learned.clamp_min(0.0) * physical.clamp_min(0.0)
#         ).clamp(0.0, 1.0)
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









# from __future__ import annotations
# from dataclasses import dataclass
# from typing import Sequence

# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# from models.geometry_bank import (
#     ClassDiscriminativeStructureState,
#     normalize_ordered_decision_features,
#     sanitize_ordered_variation,
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
#     # Read-only CDS diagnostics.  They never participate in the deployed logits.
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


# def _robust_cds_support(
#     sample: Tensor,
#     identity: Tensor,
#     variation: Tensor,
# ) -> tuple[Tensor, Tensor, Tensor]:

#     x = torch.as_tensor(sample)
#     center = torch.as_tensor(identity, device=x.device, dtype=x.dtype).detach()
#     scale = sanitize_ordered_variation(
#         torch.as_tensor(variation, device=x.device, dtype=x.dtype)
#     ).detach()
#     if x.ndim != 3 or x.size(0) == 0 or x.size(1) < 2 or x.size(2) <= 0:
#         raise ValueError("sample must be non-empty [N,L,F], L>=2")
#     if center.ndim != 3 or tuple(center.shape[1:]) != tuple(x.shape[1:]):
#         raise ValueError("CDS identity must be [C,L,F] matching sample")
#     c, length, feature_count = center.shape
#     if scale.shape != (c, length, length, feature_count):
#         raise ValueError("CDS variation must be [C,L,L,F]")
#     if not all(bool(torch.isfinite(v).all()) for v in (x, center, scale)):
#         raise ValueError("CDS support input contains NaN or Inf")

#     eps = torch.finfo(x.dtype).eps
#     diagonal = torch.arange(length, device=x.device)
#     absolute_scale = scale[:, diagonal, diagonal, :]
#     absolute_deviation = (x[:, None] - center[None]).abs()
#     denominator = absolute_scale[None] + absolute_deviation
#     absolute = absolute_scale[None] / denominator.clamp_min(eps)
#     absolute = torch.where(denominator.le(eps), torch.ones_like(absolute), absolute)
#     absolute_support = absolute.mean(dim=(2, 3)).clamp(0.0, 1.0)

#     upper = torch.triu(
#         torch.ones(length, length, device=x.device, dtype=torch.bool), diagonal=1
#     )
#     sample_relation = x[:, :, None, :] - x[:, None, :, :]
#     relation_rows: list[Tensor] = []
#     for class_index in range(c):
#         identity_relation = (
#             center[class_index, :, None, :] - center[class_index, None, :, :]
#         )
#         deviation = (sample_relation - identity_relation.unsqueeze(0)).abs()
#         class_scale = scale[class_index].unsqueeze(0)
#         rel_denominator = class_scale + deviation
#         element = class_scale / rel_denominator.clamp_min(eps)
#         element = torch.where(
#             rel_denominator.le(eps), torch.ones_like(element), element
#         )
#         relation_rows.append(
#             element[:, upper, :].mean(dim=(1, 2)).clamp(0.0, 1.0)
#         )
#     relation_support = torch.stack(relation_rows, dim=1)
#     complete_support = torch.sqrt(
#         absolute_support.clamp_min(0.0) * relation_support.clamp_min(0.0)
#     ).clamp(0.0, 1.0)
#     return absolute_support, relation_support, complete_support


# class CosineClassificationHead(nn.Module):
#     """Bias-free cosine head with one row per protocol class.

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

#         _, _, learned = _robust_cds_support(q, i_q, v_q)
#         _, _, physical = _robust_cds_support(p, i_p, v_p)
#         whole = torch.sqrt(
#             learned.clamp_min(0.0) * physical.clamp_min(0.0)
#         ).clamp(0.0, 1.0)
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



# from __future__ import annotations

# """Single all-seen cosine classifier for the HSI CDS architecture.

# Q[K,D] is the only learned discriminative representation. The classifier owns
# one bias-free row per dataset class and never stores exemplars. The HSI CDS
# {I_Q,V_Q,I_P,V_P,N} remains the persistent class structure used for transport
# and historical decision-region protection. Inter-class CDS relations are always derived
# on demand and never stored as a second classifier state.
# """

# from dataclasses import dataclass
# from typing import Sequence

# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# from models.geometry_bank import (
#     ClassDiscriminativeStructureState,
#     ordered_variation_operator,
#     whole_cds_pair_risk,
# )

# Tensor = torch.Tensor
# CDSDecisionState = ClassDiscriminativeStructureState


# @dataclass(frozen=True)
# class ClassifierOutput:
#     logits: Tensor
#     class_ids: Tensor
#     prediction: Tensor
#     decision_support: Tensor
#     cosine_support: Tensor
#     identity_support: Tensor
#     structural_support: Tensor
#     structural_reliability: Tensor
#     ordered_reliability: Tensor
#     complete_support: Tensor
#     pairwise_margin: Tensor
#     pairwise_weight: Tensor
#     spectral_confusability: Tensor


# def _validate_class_ids(
#     class_ids: Tensor | Sequence[int],
#     *,
#     total_classes: int,
#     device: torch.device,
# ) -> Tensor:
#     ids = torch.as_tensor(class_ids, device=device, dtype=torch.long).flatten()
#     if ids.ndim != 1 or ids.numel() == 0:
#         raise ValueError("class_ids must be non-empty rank-one")
#     values = [int(v) for v in ids.detach().cpu().tolist()]
#     if len(values) != len(set(values)):
#         raise ValueError("class_ids must be unique")
#     if bool((ids < 0).any()) or bool((ids >= int(total_classes)).any()):
#         raise ValueError("class_ids fall outside classifier capacity")
#     return ids


# def ordered_feature_vector(ordered_features: Tensor) -> Tensor:
#     q = torch.as_tensor(ordered_features)
#     if q.ndim != 3 or q.size(0) == 0 or q.size(1) < 2 or q.size(2) <= 0:
#         raise ValueError("ordered_features must be non-empty [N,K,D], K>=2")
#     if not torch.is_floating_point(q) or q.is_complex() or not bool(torch.isfinite(q).all()):
#         raise ValueError("ordered_features must be finite real floating point")
#     return q.reshape(q.size(0), -1)


# def ordered_identity_similarity(sample: Tensor, identity: Tensor) -> Tensor:
#     q = torch.as_tensor(sample)
#     center = torch.as_tensor(identity)
#     if q.ndim != 3 or center.ndim != 3 or center.size(0) == 0:
#         raise ValueError("Q/identity must be [N,K,D]/[C,K,D]")
#     if tuple(center.shape[1:]) != tuple(q.shape[1:]):
#         raise ValueError("identity must match Q[K,D]")
#     if center.device != q.device or center.dtype != q.dtype:
#         raise ValueError("Q and identity must share device/dtype")
#     eps = torch.finfo(q.dtype).eps
#     q_unit = F.normalize(q, p=2, dim=2, eps=eps)
#     center_unit = F.normalize(center, p=2, dim=2, eps=eps)
#     return torch.einsum("nkd,ckd->nck", q_unit, center_unit).clamp(-1.0, 1.0)


# def ordered_structural_reliability(variation: Tensor) -> Tensor:
#     """Return class/token reliability [C,K] from complete ordered V_Q.

#     Each token uses the magnitude of its complete pairwise-deformation row
#     across all other ordered tokens and learned channels. Stable spectral rows
#     receive larger weight. The result is normalized per class and is diagnostic;
#     it never changes the cosine decision rule.
#     """
#     v = torch.as_tensor(variation)
#     if v.ndim != 4 or v.size(0) == 0 or v.size(1) < 2 or v.size(1) != v.size(2):
#         raise ValueError("variation must be [C,K,K,D]")
#     if not torch.is_floating_point(v) or v.is_complex() or not bool(torch.isfinite(v).all()):
#         raise ValueError("variation must be finite real floating point")
#     row_energy = v.square().mean(dim=(2, 3)).sqrt()
#     eps = torch.finfo(v.dtype).eps
#     class_scale = row_energy.mean(dim=1, keepdim=True)
#     relative = torch.where(
#         class_scale > eps,
#         row_energy / class_scale.clamp_min(eps),
#         torch.zeros_like(row_energy),
#     )
#     raw = 1.0 / (1.0 + relative)
#     return raw / raw.sum(dim=1, keepdim=True).clamp_min(eps)




# def cds_supported_identity(
#     identity: Tensor,
#     variation: Tensor,
# ) -> Tensor:
#     """Diagnostic one-direction CDS summary; not an active preservation target.

#     The finalized method preserves complete class regions and pairwise decision
#     relations. This helper is retained only for backward-compatible diagnostics.
#     """
#     center = torch.as_tensor(identity)
#     v = torch.as_tensor(variation)
#     if center.ndim != 3 or v.ndim != 4:
#         raise ValueError("identity/variation must be [C,K,D]/[C,K,K,D]")
#     c, k, d = center.shape
#     if v.shape != (c, k, k, d):
#         raise ValueError("variation is incompatible with identity")
#     if center.device != v.device or center.dtype != v.dtype:
#         raise ValueError("identity and variation must share device/dtype")
#     reliability = ordered_structural_reliability(v)
#     supported = center * reliability.unsqueeze(2)
#     vector = supported.reshape(c, k * d)
#     eps = torch.finfo(vector.dtype).eps
#     return F.normalize(vector, p=2, dim=1, eps=eps)


# def ordered_identity_support(
#     sample: Tensor,
#     identity: Tensor,
#     variation: Tensor | None = None,
# ) -> Tensor:
#     similarity = ordered_identity_similarity(sample, identity)
#     if variation is None:
#         return similarity.mean(dim=2)
#     weight = ordered_structural_reliability(variation)
#     if weight.shape != similarity.shape[1:]:
#         raise ValueError("variation reliability is incompatible with identity")
#     return (similarity * weight.unsqueeze(0)).sum(dim=2).clamp(-1.0, 1.0)


# def ordered_intra_region_support(
#     sample: Tensor,
#     identity: Tensor,
#     variation: Tensor,
# ) -> Tensor:
#     """Return complete learned-CDS compatibility [N,C] in [0,1].

#     The diagonal of V_Q stores absolute-coordinate IQR; off-diagonal entries
#     store ordered-relation IQR.  The two supports are combined by geometric mean
#     so common-mode displacement cannot be invisible and relation deformation
#     cannot be ignored.  This remains diagnostic; logits are still pure cosine.
#     """
#     q = torch.as_tensor(sample)
#     center = torch.as_tensor(identity)
#     v = torch.as_tensor(variation)
#     if q.ndim != 3 or q.size(0) == 0 or q.size(1) < 2 or q.size(2) <= 0:
#         raise ValueError("sample must be non-empty [N,K,D], K>=2")
#     if center.ndim != 3 or tuple(center.shape[1:]) != tuple(q.shape[1:]):
#         raise ValueError("identity must be [C,K,D] matching Q")
#     c, k, d = center.shape
#     if c < 1 or v.shape != (c, k, k, d):
#         raise ValueError("variation must be [C,K,K,D] matching identity")
#     if any(value.device != q.device or value.dtype != q.dtype for value in (center, v)):
#         raise ValueError("Q/I_Q/V_Q must share device/dtype")
#     if not all(bool(torch.isfinite(value).all()) for value in (q, center, v)):
#         raise ValueError("Q/I_Q/V_Q contains NaN or Inf")
#     if bool((v < 0).any()):
#         raise ValueError("V_Q must be non-negative")

#     eps = torch.finfo(q.dtype).eps
#     index = torch.arange(k, device=q.device)
#     absolute_extent = v[:, index, index, :]
#     absolute_deviation = (q[:, None, :, :] - center[None, :, :, :]).abs()
#     absolute_support = absolute_extent[None] / (
#         absolute_extent[None] + absolute_deviation + eps
#     )
#     absolute_both_zero = (absolute_extent[None] <= eps) & (absolute_deviation <= eps)
#     absolute_support = torch.where(
#         absolute_both_zero, torch.ones_like(absolute_support), absolute_support
#     ).mean(dim=(2, 3))

#     upper = torch.triu(torch.ones(k, k, device=q.device, dtype=torch.bool), diagonal=1)
#     sample_relation = q[:, :, None, :] - q[:, None, :, :]
#     identity_relation = center[:, :, None, :] - center[:, None, :, :]
#     relation_deviation = (
#         sample_relation[:, None, :, :, :] - identity_relation[None, :, :, :, :]
#     ).abs()
#     relation_extent = v[None, :, :, :, :]
#     relation_support = relation_extent / (relation_extent + relation_deviation + eps)
#     relation_both_zero = (relation_extent <= eps) & (relation_deviation <= eps)
#     relation_support = torch.where(
#         relation_both_zero, torch.ones_like(relation_support), relation_support
#     )
#     relation_count = float(int(upper.sum().item()) * d)
#     if relation_count <= 0.0:
#         raise RuntimeError("ordered relation support has no off-diagonal entries")
#     relation_support = (
#         relation_support * upper.view(1, 1, k, k, 1)
#     ).sum(dim=(2, 3, 4)) / relation_count

#     complete = torch.sqrt(
#         absolute_support.clamp_min(0.0) * relation_support.clamp_min(0.0)
#     ).clamp(0.0, 1.0)
#     if not bool(torch.isfinite(complete).all()):
#         raise RuntimeError("complete learned CDS support became NaN or Inf")
#     return complete


# class SpectralCDSClassifier(nn.Module):
#     """One bias-free normalized cosine head shared by all phases."""

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
#         self.feature_dim = self.spectral_token_count * self.spectral_token_dim
#         if self.spectral_token_count < 2 or self.spectral_token_dim <= 0:
#             raise ValueError("invalid ordered Q dimensions")
#         if self.total_class_count < 2:
#             raise ValueError("total_class_count must be >=2")
#         self.weight = nn.Parameter(torch.empty(self.total_class_count, self.feature_dim))
#         nn.init.trunc_normal_(self.weight, std=0.02)
#         # One positive scale shared by every class. It is learned from HSI data
#         # rather than copied from another dataset/method. A common scale changes
#         # CE optimization strength but cannot create old/new argmax bias.
#         self.logit_scale = nn.Parameter(torch.zeros(()))

#     @property
#     def parameter_count(self) -> int:
#         return int(self.weight.numel() + self.logit_scale.numel())

#     @property
#     def logit_scale_value(self) -> Tensor:
#         scale = self.logit_scale.exp()
#         if not bool(torch.isfinite(scale)) or not bool(scale > 0):
#             raise RuntimeError("shared cosine logit scale became invalid")
#         return scale

#     @torch.no_grad()
#     def mask_gradient_to_classes(
#         self,
#         class_ids: Tensor | Sequence[int],
#     ) -> None:
#         """Keep classifier gradients only for the specified global class rows.

#         The classifier allocates one row for every dataset class from the start.
#         During Phase 0 only base-class rows are owned by the active task, so future
#         rows must remain bitwise unchanged. During incremental training, row-level
#         ownership is normally handled directly in ``cosine_logits`` through
#         ``trainable_class_ids``; this method is the explicit post-backward guard
#         used by the base trainer.
#         """
#         ids = _validate_class_ids(
#             class_ids,
#             total_classes=self.total_class_count,
#             device=self.weight.device,
#         )
#         gradient = self.weight.grad
#         if gradient is None:
#             return
#         if gradient.shape != self.weight.shape:
#             raise RuntimeError("classifier gradient shape disagrees with weight")
#         if gradient.device != self.weight.device or gradient.dtype != self.weight.dtype:
#             raise RuntimeError("classifier gradient disagrees with weight device/dtype")
#         keep = torch.zeros(
#             self.total_class_count,
#             device=self.weight.device,
#             dtype=torch.bool,
#         )
#         keep.index_fill_(0, ids, True)
#         gradient.mul_(keep.unsqueeze(1).to(dtype=gradient.dtype))

#     def class_weight(self, class_ids: Tensor | Sequence[int]) -> Tensor:
#         ids = _validate_class_ids(
#             class_ids,
#             total_classes=self.total_class_count,
#             device=self.weight.device,
#         )
#         return self.weight.index_select(0, ids)

#     def cosine_similarity(
#         self,
#         ordered_features: Tensor,
#         *,
#         class_ids: Tensor | Sequence[int],
#         trainable_class_ids: Tensor | Sequence[int] | None = None,
#     ) -> Tensor:
#         """Return raw normalized cosine similarities in [-1,1].

#         When ``trainable_class_ids`` is supplied, classifier rows outside that
#         subset are detached while remaining active rivals. Passing an empty
#         sequence detaches every selected row, which is used by feature-only
#         structural objectives. Q still receives gradients against every seen row.
#         """
#         q = ordered_feature_vector(ordered_features)
#         ids = _validate_class_ids(
#             class_ids,
#             total_classes=self.total_class_count,
#             device=q.device,
#         )
#         if self.weight.device != q.device or self.weight.dtype != q.dtype:
#             raise ValueError("classifier and Q must share device/dtype")

#         selected = self.weight.index_select(0, ids)
#         if trainable_class_ids is not None:
#             raw_trainable = torch.as_tensor(
#                 trainable_class_ids, device=q.device, dtype=torch.long
#             ).flatten()
#             if raw_trainable.numel() == 0:
#                 # Generic feature-only scoring mode: keep the exact cosine decision
#                 # function while preventing this objective from moving any row.
#                 selected = selected.detach()
#             else:
#                 trainable = _validate_class_ids(
#                     raw_trainable,
#                     total_classes=self.total_class_count,
#                     device=q.device,
#                 )
#                 live = ids.unsqueeze(1).eq(trainable.unsqueeze(0)).any(dim=1)
#                 selected = torch.where(live.unsqueeze(1), selected, selected.detach())

#         eps = torch.finfo(q.dtype).eps
#         q_unit = F.normalize(q, p=2, dim=1, eps=eps)
#         w_unit = F.normalize(selected, p=2, dim=1, eps=eps)
#         return F.linear(q_unit, w_unit).clamp(-1.0, 1.0)

#     def cosine_logits(
#         self,
#         ordered_features: Tensor,
#         *,
#         class_ids: Tensor | Sequence[int],
#         trainable_class_ids: Tensor | Sequence[int] | None = None,
#     ) -> Tensor:
#         """Return HSI-data-learned globally scaled cosine logits."""
#         cosine = self.cosine_similarity(
#             ordered_features,
#             class_ids=class_ids,
#             trainable_class_ids=trainable_class_ids,
#         )
#         return cosine * self.logit_scale_value.to(device=cosine.device, dtype=cosine.dtype)

#     def forward(
#         self,
#         ordered_features: Tensor,
#         *,
#         class_ids: Tensor | Sequence[int] | None = None,
#         cds: ClassDiscriminativeStructureState | None = None,
#         trainable_class_ids: Tensor | Sequence[int] | None = None,
#     ) -> ClassifierOutput:
#         q = torch.as_tensor(ordered_features)
#         if q.ndim != 3 or tuple(q.shape[1:]) != (
#             self.spectral_token_count,
#             self.spectral_token_dim,
#         ):
#             raise ValueError("ordered_features have invalid shape")

#         if cds is not None:
#             if not isinstance(cds, ClassDiscriminativeStructureState):
#                 raise TypeError("cds must be ClassDiscriminativeStructureState")
#             cds_ids = cds.classes.class_ids.to(device=q.device)
#             if class_ids is None:
#                 ids = cds_ids
#             else:
#                 ids = _validate_class_ids(
#                     class_ids,
#                     total_classes=self.total_class_count,
#                     device=q.device,
#                 )
#                 if ids.shape != cds_ids.shape or not torch.equal(ids, cds_ids):
#                     raise ValueError("class_ids must match supplied CDS order")
#         else:
#             if class_ids is None:
#                 raise ValueError("class_ids are required when cds is absent")
#             ids = _validate_class_ids(
#                 class_ids,
#                 total_classes=self.total_class_count,
#                 device=q.device,
#             )

#         cosine = self.cosine_similarity(
#             q,
#             class_ids=ids,
#             trainable_class_ids=trainable_class_ids,
#         )
#         logits = cosine * self.logit_scale_value.to(device=q.device, dtype=q.dtype)
#         local_prediction = logits.argmax(dim=1)
#         prediction = ids.index_select(0, local_prediction)
#         decision_support = torch.softmax(logits, dim=1)
#         pairwise_margin = logits.unsqueeze(2) - logits.unsqueeze(1)

#         if cds is None:
#             identity_support = cosine.detach() * 0.0
#             structural_support = cosine.detach() * 0.0
#             reliability = q.new_full(
#                 (ids.numel(), self.spectral_token_count),
#                 1.0 / self.spectral_token_count,
#             )
#             confusability = q.new_zeros((ids.numel(), ids.numel()))
#         else:
#             classes = cds.classes
#             stable_variation = ordered_variation_operator(
#                 classes.variation.to(device=q.device, dtype=q.dtype)
#             )
#             identity_tensor = classes.identity.to(device=q.device, dtype=q.dtype)
#             identity_support = ordered_identity_support(q, identity_tensor)
#             structural_support = ordered_intra_region_support(
#                 q,
#                 identity_tensor,
#                 stable_variation,
#             )
#             reliability = ordered_structural_reliability(stable_variation)
#             confusability = cds.relations.spectral_confusability.to(
#                 device=q.device,
#                 dtype=q.dtype,
#             )

#         if cds is None:
#             complete_support = cosine.detach() * 0.0
#             pairwise_weight = q.new_zeros((ids.numel(), ids.numel()))
#         else:
#             complete_support = complete_cds_pairwise_evidence(
#                 identity_support,
#                 structural_support,
#                 confusability,
#             )[0]
#             pairwise_weight, _learned_risk, _spectral_risk = whole_cds_pair_risk(cds)
#             pairwise_weight = pairwise_weight.to(device=q.device, dtype=q.dtype)

#         values = (
#             logits, cosine, decision_support, complete_support,
#             pairwise_margin, pairwise_weight
#         )
#         if not all(bool(torch.isfinite(value).all()) for value in values):
#             raise RuntimeError("cosine classifier produced NaN or Inf")

#         return ClassifierOutput(
#             logits=logits,
#             class_ids=ids,
#             prediction=prediction,
#             decision_support=decision_support,
#             cosine_support=cosine,
#             identity_support=identity_support,
#             structural_support=structural_support,
#             structural_reliability=reliability,
#             ordered_reliability=reliability,
#             complete_support=complete_support,
#             pairwise_margin=pairwise_margin,
#             pairwise_weight=pairwise_weight,
#             spectral_confusability=confusability,
#         )


# def complete_cds_pairwise_evidence(
#     identity_support: Tensor,
#     structural_support: Tensor,
#     spectral_confusability: Tensor,
# ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
#     identity = torch.as_tensor(identity_support)
#     structural = torch.as_tensor(structural_support)
#     conf = torch.as_tensor(spectral_confusability)
#     if identity.ndim != 2 or structural.shape != identity.shape:
#         raise ValueError("identity/structural support must share [N,C] shape")
#     if conf.shape != (identity.size(1), identity.size(1)):
#         raise ValueError("spectral_confusability must be [C,C]")
#     if any(v.device != identity.device or v.dtype != identity.dtype for v in (structural, conf)):
#         raise ValueError("complete CDS evidence tensors must share device/dtype")
#     if not all(bool(torch.isfinite(v).all()) for v in (identity, structural, conf)):
#         raise ValueError("complete CDS evidence contains NaN or Inf")

#     identity_01 = ((identity.clamp(-1.0, 1.0) + 1.0) * 0.5).clamp(0.0, 1.0)
#     structural_01 = structural.clamp(0.0, 1.0)
#     complete = torch.sqrt(identity_01 * structural_01).clamp(0.0, 1.0)
#     pairwise = complete.unsqueeze(2) - complete.unsqueeze(1)

#     weight = 1.0 + conf.clamp(0.0, 1.0)
#     diagonal = torch.arange(identity.size(1), device=identity.device)
#     weight = weight.clone()
#     weight[diagonal, diagonal] = 0.0
#     return complete, identity_01, pairwise, weight


# def complete_cds_pairwise_logits(
#     identity_support: Tensor,
#     structural_support: Tensor,
#     spectral_confusability: Tensor,
# ) -> Tensor:
#     return complete_cds_pairwise_evidence(
#         identity_support,
#         structural_support,
#         spectral_confusability,
#     )[0]


# __all__ = [
#     "CDSDecisionState",
#     "ClassifierOutput",
#     "SpectralCDSClassifier",
#     "ordered_feature_vector",
#     "ordered_identity_similarity",
#     "ordered_identity_support",
#     "ordered_intra_region_support",
#     "ordered_structural_reliability",
#     "cds_supported_identity",
#     "complete_cds_pairwise_evidence",
#     "complete_cds_pairwise_logits",
# ]









# from __future__ import annotations

# """Single all-seen cosine classifier for the HSI CDS architecture.

# Q[K,D] is the only learned discriminative representation. The classifier owns
# one bias-free row per dataset class and never stores exemplars. The HSI CDS
# {I_Q,V_Q,I_P,V_P,N} remains the persistent class structure used for transport
# and historical decision-region protection. Inter-class CDS relations are always derived
# on demand and never stored as a second classifier state.
# """

# from dataclasses import dataclass
# from typing import Sequence

# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# from models.geometry_bank import (
#     ClassDiscriminativeStructureState,
#     ordered_variation_operator,
#     whole_cds_pair_risk,
# )

# Tensor = torch.Tensor
# CDSDecisionState = ClassDiscriminativeStructureState


# @dataclass(frozen=True)
# class ClassifierOutput:
#     logits: Tensor
#     class_ids: Tensor
#     prediction: Tensor
#     decision_support: Tensor
#     cosine_support: Tensor
#     identity_support: Tensor
#     structural_support: Tensor
#     structural_reliability: Tensor
#     ordered_reliability: Tensor
#     complete_support: Tensor
#     pairwise_margin: Tensor
#     pairwise_weight: Tensor
#     spectral_confusability: Tensor


# def _validate_class_ids(
#     class_ids: Tensor | Sequence[int],
#     *,
#     total_classes: int,
#     device: torch.device,
# ) -> Tensor:
#     ids = torch.as_tensor(class_ids, device=device, dtype=torch.long).flatten()
#     if ids.ndim != 1 or ids.numel() == 0:
#         raise ValueError("class_ids must be non-empty rank-one")
#     values = [int(v) for v in ids.detach().cpu().tolist()]
#     if len(values) != len(set(values)):
#         raise ValueError("class_ids must be unique")
#     if bool((ids < 0).any()) or bool((ids >= int(total_classes)).any()):
#         raise ValueError("class_ids fall outside classifier capacity")
#     return ids


# def ordered_feature_vector(ordered_features: Tensor) -> Tensor:
#     q = torch.as_tensor(ordered_features)
#     if q.ndim != 3 or q.size(0) == 0 or q.size(1) < 2 or q.size(2) <= 0:
#         raise ValueError("ordered_features must be non-empty [N,K,D], K>=2")
#     if not torch.is_floating_point(q) or q.is_complex() or not bool(torch.isfinite(q).all()):
#         raise ValueError("ordered_features must be finite real floating point")
#     return q.reshape(q.size(0), -1)


# def ordered_identity_similarity(sample: Tensor, identity: Tensor) -> Tensor:
#     q = torch.as_tensor(sample)
#     center = torch.as_tensor(identity)
#     if q.ndim != 3 or center.ndim != 3 or center.size(0) == 0:
#         raise ValueError("Q/identity must be [N,K,D]/[C,K,D]")
#     if tuple(center.shape[1:]) != tuple(q.shape[1:]):
#         raise ValueError("identity must match Q[K,D]")
#     if center.device != q.device or center.dtype != q.dtype:
#         raise ValueError("Q and identity must share device/dtype")
#     eps = torch.finfo(q.dtype).eps
#     q_unit = F.normalize(q, p=2, dim=2, eps=eps)
#     center_unit = F.normalize(center, p=2, dim=2, eps=eps)
#     return torch.einsum("nkd,ckd->nck", q_unit, center_unit).clamp(-1.0, 1.0)


# def ordered_structural_reliability(variation: Tensor) -> Tensor:
#     """Return class/token reliability [C,K] from complete ordered V_Q.

#     Each token uses the magnitude of its complete pairwise-deformation row
#     across all other ordered tokens and learned channels. Stable spectral rows
#     receive larger weight. The result is normalized per class and is diagnostic;
#     it never changes the cosine decision rule.
#     """
#     v = torch.as_tensor(variation)
#     if v.ndim != 4 or v.size(0) == 0 or v.size(1) < 2 or v.size(1) != v.size(2):
#         raise ValueError("variation must be [C,K,K,D]")
#     if not torch.is_floating_point(v) or v.is_complex() or not bool(torch.isfinite(v).all()):
#         raise ValueError("variation must be finite real floating point")
#     row_energy = v.square().mean(dim=(2, 3)).sqrt()
#     eps = torch.finfo(v.dtype).eps
#     class_scale = row_energy.mean(dim=1, keepdim=True)
#     relative = torch.where(
#         class_scale > eps,
#         row_energy / class_scale.clamp_min(eps),
#         torch.zeros_like(row_energy),
#     )
#     raw = 1.0 / (1.0 + relative)
#     return raw / raw.sum(dim=1, keepdim=True).clamp_min(eps)




# def cds_supported_identity(
#     identity: Tensor,
#     variation: Tensor,
# ) -> Tensor:
#     """Diagnostic one-direction CDS summary; not an active preservation target.

#     The finalized method preserves complete class regions and pairwise decision
#     relations. This helper is retained only for backward-compatible diagnostics.
#     """
#     center = torch.as_tensor(identity)
#     v = torch.as_tensor(variation)
#     if center.ndim != 3 or v.ndim != 4:
#         raise ValueError("identity/variation must be [C,K,D]/[C,K,K,D]")
#     c, k, d = center.shape
#     if v.shape != (c, k, k, d):
#         raise ValueError("variation is incompatible with identity")
#     if center.device != v.device or center.dtype != v.dtype:
#         raise ValueError("identity and variation must share device/dtype")
#     reliability = ordered_structural_reliability(v)
#     supported = center * reliability.unsqueeze(2)
#     vector = supported.reshape(c, k * d)
#     eps = torch.finfo(vector.dtype).eps
#     return F.normalize(vector, p=2, dim=1, eps=eps)


# def ordered_identity_support(
#     sample: Tensor,
#     identity: Tensor,
#     variation: Tensor | None = None,
# ) -> Tensor:
#     similarity = ordered_identity_similarity(sample, identity)
#     if variation is None:
#         return similarity.mean(dim=2)
#     weight = ordered_structural_reliability(variation)
#     if weight.shape != similarity.shape[1:]:
#         raise ValueError("variation reliability is incompatible with identity")
#     return (similarity * weight.unsqueeze(0)).sum(dim=2).clamp(-1.0, 1.0)


# def ordered_intra_region_support(
#     sample: Tensor,
#     identity: Tensor,
#     variation: Tensor,
# ) -> Tensor:
#     """Return pairwise-deformation compatibility [N,C] in [0,1].

#     This is a diagnostic view of the finalized V_Q semantics, not a second
#     classifier. For every class and unordered spectral-token pair (k,l), it
#     compares the observed relation deformation

#         |(Q[k]-Q[l]) - (I_Q[k]-I_Q[l])|

#     with the stored robust deformation V_Q[k,l]. A zero-deformation sample has
#     support one; deformation much larger than the stored class deformation
#     approaches zero. No covariance projection or matrix inverse is used.
#     """
#     q = torch.as_tensor(sample)
#     center = torch.as_tensor(identity)
#     v = torch.as_tensor(variation)
#     if q.ndim != 3 or q.size(0) == 0 or q.size(1) < 2 or q.size(2) <= 0:
#         raise ValueError("sample must be non-empty [N,K,D], K>=2")
#     if center.ndim != 3 or tuple(center.shape[1:]) != tuple(q.shape[1:]):
#         raise ValueError("identity must be [C,K,D] matching Q")
#     c, k, d = center.shape
#     if c < 1 or v.shape != (c, k, k, d):
#         raise ValueError("variation must be [C,K,K,D] matching identity")
#     if any(value.device != q.device or value.dtype != q.dtype for value in (center, v)):
#         raise ValueError("Q/I_Q/V_Q must share device/dtype")
#     if not all(bool(torch.isfinite(value).all()) for value in (q, center, v)):
#         raise ValueError("Q/I_Q/V_Q contains NaN or Inf")
#     if bool((v < 0).any()):
#         raise ValueError("V_Q deformation must be non-negative")
#     diagonal = v.diagonal(offset=0, dim1=1, dim2=2)
#     if bool(torch.count_nonzero(diagonal)):
#         raise ValueError("V_Q diagonal must be exactly zero under relation-IQR semantics")

#     eps = torch.finfo(q.dtype).eps
#     output: list[Tensor] = []
#     row_chunk = 4

#     for class_index in range(c):
#         total = q.new_zeros((q.size(0),))
#         count = 0
#         for start in range(0, k, row_chunk):
#             stop = min(start + row_chunk, k)
#             sample_relation = (
#                 q[:, start:stop, None, :] - q[:, None, :, :]
#             )  # [N,R,K,D]
#             identity_relation = (
#                 center[class_index, start:stop, None, :]
#                 - center[class_index, None, :, :]
#             )  # [R,K,D]
#             observed = (
#                 sample_relation - identity_relation.unsqueeze(0)
#             ).abs()
#             tolerance = v[class_index, start:stop, :, :].unsqueeze(0)

#             element = tolerance / (tolerance + observed + eps)
#             both_zero = (tolerance <= eps) & (observed <= eps)
#             element = torch.where(both_zero, torch.ones_like(element), element)

#             row_index = torch.arange(start, stop, device=q.device).view(-1, 1)
#             col_index = torch.arange(k, device=q.device).view(1, -1)
#             upper = row_index < col_index
#             total = total + (
#                 element * upper.view(1, stop - start, k, 1)
#             ).sum(dim=(1, 2, 3))
#             count += int(upper.sum().item()) * d

#         if count <= 0:
#             raise RuntimeError("ordered pair support has no off-diagonal entries")
#         output.append((total / float(count)).clamp(0.0, 1.0))

#     support = torch.stack(output, dim=1)
#     if not bool(torch.isfinite(support).all()):
#         raise RuntimeError("ordered intra-class CDS support became NaN or Inf")
#     return support


# class SpectralCDSClassifier(nn.Module):
#     """One bias-free normalized cosine head shared by all phases."""

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
#         self.feature_dim = self.spectral_token_count * self.spectral_token_dim
#         if self.spectral_token_count < 2 or self.spectral_token_dim <= 0:
#             raise ValueError("invalid ordered Q dimensions")
#         if self.total_class_count < 2:
#             raise ValueError("total_class_count must be >=2")
#         self.weight = nn.Parameter(torch.empty(self.total_class_count, self.feature_dim))
#         nn.init.trunc_normal_(self.weight, std=0.02)
#         # One positive scale shared by every class. It is learned from HSI data
#         # rather than copied from another dataset/method. A common scale changes
#         # CE optimization strength but cannot create old/new argmax bias.
#         self.logit_scale = nn.Parameter(torch.zeros(()))

#     @property
#     def parameter_count(self) -> int:
#         return int(self.weight.numel() + self.logit_scale.numel())

#     @property
#     def logit_scale_value(self) -> Tensor:
#         scale = self.logit_scale.exp()
#         if not bool(torch.isfinite(scale)) or not bool(scale > 0):
#             raise RuntimeError("shared cosine logit scale became invalid")
#         return scale

#     @torch.no_grad()
#     def mask_gradient_to_classes(
#         self,
#         class_ids: Tensor | Sequence[int],
#     ) -> None:
#         """Keep classifier gradients only for the specified global class rows.

#         The classifier allocates one row for every dataset class from the start.
#         During Phase 0 only base-class rows are owned by the active task, so future
#         rows must remain bitwise unchanged. During incremental training, row-level
#         ownership is normally handled directly in ``cosine_logits`` through
#         ``trainable_class_ids``; this method is the explicit post-backward guard
#         used by the base trainer.
#         """
#         ids = _validate_class_ids(
#             class_ids,
#             total_classes=self.total_class_count,
#             device=self.weight.device,
#         )
#         gradient = self.weight.grad
#         if gradient is None:
#             return
#         if gradient.shape != self.weight.shape:
#             raise RuntimeError("classifier gradient shape disagrees with weight")
#         if gradient.device != self.weight.device or gradient.dtype != self.weight.dtype:
#             raise RuntimeError("classifier gradient disagrees with weight device/dtype")
#         keep = torch.zeros(
#             self.total_class_count,
#             device=self.weight.device,
#             dtype=torch.bool,
#         )
#         keep.index_fill_(0, ids, True)
#         gradient.mul_(keep.unsqueeze(1).to(dtype=gradient.dtype))

#     def class_weight(self, class_ids: Tensor | Sequence[int]) -> Tensor:
#         ids = _validate_class_ids(
#             class_ids,
#             total_classes=self.total_class_count,
#             device=self.weight.device,
#         )
#         return self.weight.index_select(0, ids)

#     def cosine_similarity(
#         self,
#         ordered_features: Tensor,
#         *,
#         class_ids: Tensor | Sequence[int],
#         trainable_class_ids: Tensor | Sequence[int] | None = None,
#     ) -> Tensor:
#         """Return raw normalized cosine similarities in [-1,1].

#         When ``trainable_class_ids`` is supplied, classifier rows outside that
#         subset are detached while remaining active rivals. Passing an empty
#         sequence detaches every selected row, which is used by feature-only
#         structural objectives. Q still receives gradients against every seen row.
#         """
#         q = ordered_feature_vector(ordered_features)
#         ids = _validate_class_ids(
#             class_ids,
#             total_classes=self.total_class_count,
#             device=q.device,
#         )
#         if self.weight.device != q.device or self.weight.dtype != q.dtype:
#             raise ValueError("classifier and Q must share device/dtype")

#         selected = self.weight.index_select(0, ids)
#         if trainable_class_ids is not None:
#             raw_trainable = torch.as_tensor(
#                 trainable_class_ids, device=q.device, dtype=torch.long
#             ).flatten()
#             if raw_trainable.numel() == 0:
#                 # Generic feature-only scoring mode: keep the exact cosine decision
#                 # function while preventing this objective from moving any row.
#                 selected = selected.detach()
#             else:
#                 trainable = _validate_class_ids(
#                     raw_trainable,
#                     total_classes=self.total_class_count,
#                     device=q.device,
#                 )
#                 live = ids.unsqueeze(1).eq(trainable.unsqueeze(0)).any(dim=1)
#                 selected = torch.where(live.unsqueeze(1), selected, selected.detach())

#         eps = torch.finfo(q.dtype).eps
#         q_unit = F.normalize(q, p=2, dim=1, eps=eps)
#         w_unit = F.normalize(selected, p=2, dim=1, eps=eps)
#         return F.linear(q_unit, w_unit).clamp(-1.0, 1.0)

#     def cosine_logits(
#         self,
#         ordered_features: Tensor,
#         *,
#         class_ids: Tensor | Sequence[int],
#         trainable_class_ids: Tensor | Sequence[int] | None = None,
#     ) -> Tensor:
#         """Return HSI-data-learned globally scaled cosine logits."""
#         cosine = self.cosine_similarity(
#             ordered_features,
#             class_ids=class_ids,
#             trainable_class_ids=trainable_class_ids,
#         )
#         return cosine * self.logit_scale_value.to(device=cosine.device, dtype=cosine.dtype)

#     def forward(
#         self,
#         ordered_features: Tensor,
#         *,
#         class_ids: Tensor | Sequence[int] | None = None,
#         cds: ClassDiscriminativeStructureState | None = None,
#         trainable_class_ids: Tensor | Sequence[int] | None = None,
#     ) -> ClassifierOutput:
#         q = torch.as_tensor(ordered_features)
#         if q.ndim != 3 or tuple(q.shape[1:]) != (
#             self.spectral_token_count,
#             self.spectral_token_dim,
#         ):
#             raise ValueError("ordered_features have invalid shape")

#         if cds is not None:
#             if not isinstance(cds, ClassDiscriminativeStructureState):
#                 raise TypeError("cds must be ClassDiscriminativeStructureState")
#             cds_ids = cds.classes.class_ids.to(device=q.device)
#             if class_ids is None:
#                 ids = cds_ids
#             else:
#                 ids = _validate_class_ids(
#                     class_ids,
#                     total_classes=self.total_class_count,
#                     device=q.device,
#                 )
#                 if ids.shape != cds_ids.shape or not torch.equal(ids, cds_ids):
#                     raise ValueError("class_ids must match supplied CDS order")
#         else:
#             if class_ids is None:
#                 raise ValueError("class_ids are required when cds is absent")
#             ids = _validate_class_ids(
#                 class_ids,
#                 total_classes=self.total_class_count,
#                 device=q.device,
#             )

#         cosine = self.cosine_similarity(
#             q,
#             class_ids=ids,
#             trainable_class_ids=trainable_class_ids,
#         )
#         logits = cosine * self.logit_scale_value.to(device=q.device, dtype=q.dtype)
#         local_prediction = logits.argmax(dim=1)
#         prediction = ids.index_select(0, local_prediction)
#         decision_support = torch.softmax(logits, dim=1)
#         pairwise_margin = logits.unsqueeze(2) - logits.unsqueeze(1)

#         if cds is None:
#             identity_support = cosine.detach() * 0.0
#             structural_support = cosine.detach() * 0.0
#             reliability = q.new_full(
#                 (ids.numel(), self.spectral_token_count),
#                 1.0 / self.spectral_token_count,
#             )
#             confusability = q.new_zeros((ids.numel(), ids.numel()))
#         else:
#             classes = cds.classes
#             stable_variation = ordered_variation_operator(
#                 classes.variation.to(device=q.device, dtype=q.dtype)
#             )
#             identity_tensor = classes.identity.to(device=q.device, dtype=q.dtype)
#             identity_support = ordered_identity_support(q, identity_tensor)
#             structural_support = ordered_intra_region_support(
#                 q,
#                 identity_tensor,
#                 stable_variation,
#             )
#             reliability = ordered_structural_reliability(stable_variation)
#             confusability = cds.relations.spectral_confusability.to(
#                 device=q.device,
#                 dtype=q.dtype,
#             )

#         if cds is None:
#             pairwise_weight = q.new_zeros((ids.numel(), ids.numel()))
#         else:
#             pairwise_weight, _learned_risk, _spectral_risk = whole_cds_pair_risk(cds)
#             pairwise_weight = pairwise_weight.to(device=q.device, dtype=q.dtype)

#         values = (logits, cosine, decision_support, pairwise_margin, pairwise_weight)
#         if not all(bool(torch.isfinite(value).all()) for value in values):
#             raise RuntimeError("cosine classifier produced NaN or Inf")

#         return ClassifierOutput(
#             logits=logits,
#             class_ids=ids,
#             prediction=prediction,
#             decision_support=decision_support,
#             cosine_support=cosine,
#             identity_support=identity_support,
#             structural_support=structural_support,
#             structural_reliability=reliability,
#             ordered_reliability=reliability,
#             complete_support=logits,
#             pairwise_margin=pairwise_margin,
#             pairwise_weight=pairwise_weight,
#             spectral_confusability=confusability,
#         )


# def complete_cds_pairwise_evidence(
#     identity_support: Tensor,
#     structural_support: Tensor,
#     spectral_confusability: Tensor,
# ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
#     identity = torch.as_tensor(identity_support)
#     conf = torch.as_tensor(spectral_confusability)
#     if identity.ndim != 2 or conf.shape != (identity.size(1), identity.size(1)):
#         raise ValueError("invalid compatibility tensors")
#     pairwise = identity.unsqueeze(2) - identity.unsqueeze(1)
#     weight = 1.0 + conf.clamp(0.0, 1.0)
#     diagonal = torch.arange(identity.size(1), device=identity.device)
#     weight = weight.clone()
#     weight[diagonal, diagonal] = 0.0
#     return identity, identity, pairwise, weight


# def complete_cds_pairwise_logits(
#     identity_support: Tensor,
#     structural_support: Tensor,
#     spectral_confusability: Tensor,
# ) -> Tensor:
#     return complete_cds_pairwise_evidence(
#         identity_support,
#         structural_support,
#         spectral_confusability,
#     )[0]


# __all__ = [
#     "CDSDecisionState",
#     "ClassifierOutput",
#     "SpectralCDSClassifier",
#     "ordered_feature_vector",
#     "ordered_identity_similarity",
#     "ordered_identity_support",
#     "ordered_intra_region_support",
#     "ordered_structural_reliability",
#     "cds_supported_identity",
#     "complete_cds_pairwise_evidence",
#     "complete_cds_pairwise_logits",
# ]
