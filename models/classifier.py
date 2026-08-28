from __future__ import annotations

"""Persistent scale-canonical classifier for NECIL-HSI.

Role
----
This module performs classification only.

The model provides the already-canonical representation

    Q(x) in R^{K x D},    RMS(Q)=1.

After flattening,

    z(x)=vec(Q(x)) in R^F,    F=KD,

so the canonicalizer gives the fixed sample norm

    ||z(x)||_2 = sqrt(F).

The classifier normalizes only each trainable class row,

    u_c = w_c / ||w_c||_2,

and uses

    l_c(x) = z(x)^T u_c = sqrt(F) cos(theta_c).

Thus classification has the angular equal-rule behavior of a cosine head while
using a deterministic logit scale supplied by the canonical representation
itself.  There is no learned/hand-chosen scale, temperature, margin, bias,
geometry energy, replay state, preservation state, or task head.

Prediction is argmax_c l_c(x), which is exactly the same ordering as argmax
cosine because sqrt(F) is positive and class-independent.
"""

from dataclasses import dataclass
from numbers import Integral
from typing import Sequence

import torch
import torch.nn as nn

Tensor = torch.Tensor


@dataclass(frozen=True)
class ClassifierOutput:
    logits: Tensor
    cosine_similarity: Tensor
    class_ids: Tensor
    prediction: Tensor


def _parse_class_ids(
    class_ids: Sequence[int] | Tensor,
    *,
    allow_empty: bool,
) -> tuple[int, ...]:
    if torch.is_tensor(class_ids):
        if class_ids.ndim != 1:
            raise ValueError("class_ids tensor must be rank one")
        if class_ids.dtype != torch.long:
            raise TypeError("class_ids tensor must use torch.long")
        source = class_ids.detach().cpu().tolist()
    else:
        source = list(class_ids)

    parsed: list[int] = []
    for value in source:
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise TypeError("class_ids must contain integer class IDs")
        value = int(value)
        if value < 0:
            raise ValueError("class_ids must be non-negative")
        parsed.append(value)

    result = tuple(parsed)
    if not allow_empty and not result:
        raise ValueError("class_ids cannot be empty")
    if len(result) != len(set(result)):
        raise ValueError("class_ids must be unique")
    return result


def _unit_normalize(value: Tensor, *, dim: int) -> Tensor:
    if not torch.is_floating_point(value) or value.is_complex():
        raise TypeError("normalization input must be real floating point")
    norm = torch.linalg.vector_norm(value, ord=2, dim=dim, keepdim=True)
    if not bool(torch.isfinite(norm).all()):
        raise RuntimeError("normalization norm contains NaN or Inf")
    floor = torch.as_tensor(
        torch.finfo(value.dtype).tiny,
        device=value.device,
        dtype=value.dtype,
    )
    return value / norm.clamp_min(floor)


class PersistentCosineClassifier(nn.Module):
    """One normalized classifier whose class rows expand across phases."""

    def __init__(
        self,
        *,
        spectral_token_count: int,
        spectral_token_dim: int,
        class_ids: Sequence[int] | Tensor = (),
    ) -> None:
        super().__init__()

        if (
            isinstance(spectral_token_count, bool)
            or not isinstance(spectral_token_count, Integral)
            or int(spectral_token_count) <= 0
        ):
            raise ValueError("spectral_token_count must be a positive integer")
        if (
            isinstance(spectral_token_dim, bool)
            or not isinstance(spectral_token_dim, Integral)
            or int(spectral_token_dim) <= 0
        ):
            raise ValueError("spectral_token_dim must be a positive integer")

        self.spectral_token_count = int(spectral_token_count)
        self.spectral_token_dim = int(spectral_token_dim)
        self.input_dim = self.spectral_token_count * self.spectral_token_dim

        parsed = _parse_class_ids(class_ids, allow_empty=True)
        self.register_buffer(
            "class_ids",
            torch.tensor(parsed, dtype=torch.long),
            persistent=True,
        )
        self.weight = nn.Parameter(torch.empty(len(parsed), self.input_dim))
        self._initialize_rows(self.weight)
        self.validate_state()

    @staticmethod
    def _initialize_rows(rows: Tensor) -> None:
        # Only the direction matters because rows are normalized in forward().
        if rows.numel() > 0:
            nn.init.normal_(rows, mean=0.0, std=1.0)

    @property
    def class_count(self) -> int:
        return int(self.class_ids.numel())

    @property
    def seen_class_ids(self) -> tuple[int, ...]:
        return tuple(int(v) for v in self.class_ids.detach().cpu().tolist())

    def validate_state(self) -> bool:
        if self.class_ids.ndim != 1 or self.class_ids.dtype != torch.long:
            raise RuntimeError("classifier class_ids must be rank-one torch.long")
        if bool((self.class_ids < 0).any()):
            raise RuntimeError("classifier class_ids must be non-negative")
        if self.class_ids.unique().numel() != self.class_ids.numel():
            raise RuntimeError("classifier class_ids must be unique")

        expected = (self.class_count, self.input_dim)
        if tuple(self.weight.shape) != expected:
            raise RuntimeError(
                f"classifier weight must have shape {expected}, got "
                f"{tuple(self.weight.shape)}"
            )
        if not torch.is_floating_point(self.weight) or self.weight.is_complex():
            raise RuntimeError("classifier weight must be real floating point")
        if not bool(torch.isfinite(self.weight).all()):
            raise RuntimeError("classifier weight contains NaN or Inf")
        return True

    def add_classes(self, class_ids: Sequence[int] | Tensor) -> None:
        """Append unseen rows while copying every existing classifier row exactly."""
        new_ids = _parse_class_ids(class_ids, allow_empty=False)
        existing = set(self.seen_class_ids)
        duplicates = [class_id for class_id in new_ids if class_id in existing]
        if duplicates:
            raise ValueError(f"classifier already contains class IDs {duplicates}")

        new_rows = torch.empty(
            len(new_ids),
            self.input_dim,
            device=self.weight.device,
            dtype=self.weight.dtype,
        )
        self._initialize_rows(new_rows)

        with torch.no_grad():
            combined_weight = torch.cat((self.weight.detach(), new_rows), dim=0)
            combined_ids = torch.cat(
                (
                    self.class_ids,
                    torch.tensor(
                        new_ids,
                        device=self.class_ids.device,
                        dtype=torch.long,
                    ),
                ),
                dim=0,
            )

        # Expansion must happen before creating that phase's optimizer because
        # the Parameter object is intentionally replaced here.
        self.weight = nn.Parameter(combined_weight)
        self.class_ids = combined_ids
        self.validate_state()

    def _selected_rows(
        self,
        class_ids: Sequence[int] | Tensor | None,
    ) -> tuple[Tensor, Tensor]:
        if self.class_count == 0:
            raise RuntimeError("classifier has no classes")
        if class_ids is None:
            return self.weight, self.class_ids

        requested = _parse_class_ids(class_ids, allow_empty=False)
        row_by_id = {
            int(class_id): row
            for row, class_id in enumerate(self.class_ids.detach().cpu().tolist())
        }
        missing = [class_id for class_id in requested if class_id not in row_by_id]
        if missing:
            raise KeyError(f"classifier does not contain class IDs {missing}")
        indices = torch.tensor(
            [row_by_id[class_id] for class_id in requested],
            device=self.class_ids.device,
            dtype=torch.long,
        )
        return (
            self.weight.index_select(0, indices),
            self.class_ids.index_select(0, indices),
        )

    def target_indices(self, labels: Tensor) -> Tensor:
        labels = torch.as_tensor(labels, device=self.class_ids.device)
        if labels.ndim != 1 or labels.numel() == 0:
            raise ValueError("labels must be non-empty rank-one")
        if labels.dtype != torch.long:
            raise TypeError("labels must use torch.long")
        if self.class_count == 0:
            raise RuntimeError("classifier has no classes")

        matches = labels.unsqueeze(1).eq(self.class_ids.unsqueeze(0))
        counts = matches.sum(dim=1)
        if not bool(counts.eq(1).all()):
            missing = labels[counts.eq(0)].detach().cpu().unique().tolist()
            raise ValueError(
                f"labels contain class IDs absent from classifier: {missing}"
            )
        return matches.to(torch.long).argmax(dim=1)

    def forward(
        self,
        ordered_features: Tensor,
        *,
        class_ids: Sequence[int] | Tensor | None = None,
    ) -> ClassifierOutput:
        self.validate_state()
        selected_weight, selected_ids = self._selected_rows(class_ids)

        features = torch.as_tensor(ordered_features)
        expected_tail = (self.spectral_token_count, self.spectral_token_dim)
        if (
            features.ndim != 3
            or features.size(0) == 0
            or tuple(features.shape[1:]) != expected_tail
        ):
            raise ValueError(
                "ordered_features must have shape "
                f"[N,{expected_tail[0]},{expected_tail[1]}]"
            )
        if not torch.is_floating_point(features) or features.is_complex():
            raise TypeError("ordered_features must be real floating point")
        if features.device != selected_weight.device:
            raise ValueError("ordered_features and classifier must share a device")
        if features.dtype != selected_weight.dtype:
            raise TypeError("ordered_features and classifier must share a dtype")
        if not bool(torch.isfinite(features).all()):
            raise ValueError("ordered_features contain NaN or Inf")

        flattened = features.reshape(features.size(0), self.input_dim)
        weight_direction = _unit_normalize(selected_weight, dim=1)

        # Q is already globally RMS-canonicalized by NECILModel.  Therefore
        # ||vec(Q)||_2=sqrt(KD), which is a deterministic, class-independent
        # scale.  Do not normalize Q again here: doing so would discard the
        # canonicalizer's fixed norm and compress CE logits to [-1,1].
        logits = flattened @ weight_direction.transpose(0, 1)
        feature_direction = _unit_normalize(flattened, dim=1)
        cosine = feature_direction @ weight_direction.transpose(0, 1)
        if not bool(torch.isfinite(logits).all()):
            raise RuntimeError("classifier logits contain NaN or Inf")
        if not bool(torch.isfinite(cosine).all()):
            raise RuntimeError("classifier cosine diagnostics contain NaN or Inf")

        prediction = selected_ids.index_select(0, logits.argmax(dim=1))
        return ClassifierOutput(
            logits=logits,
            cosine_similarity=cosine,
            class_ids=selected_ids,
            prediction=prediction,
        )

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        ids_key = prefix + "class_ids"
        weight_key = prefix + "weight"
        incoming_ids = state_dict.get(ids_key)
        incoming_weight = state_dict.get(weight_key)

        if incoming_ids is not None:
            if incoming_ids.ndim != 1 or incoming_ids.dtype != torch.long:
                error_msgs.append(f"{ids_key} must be rank-one torch.long")
            elif self.class_ids.shape != incoming_ids.shape:
                self.class_ids = torch.empty(
                    incoming_ids.shape,
                    device=self.class_ids.device,
                    dtype=torch.long,
                )

        if incoming_weight is not None:
            if incoming_weight.ndim != 2 or incoming_weight.size(1) != self.input_dim:
                error_msgs.append(
                    f"{weight_key} must have shape [C,{self.input_dim}]"
                )
            elif self.weight.shape != incoming_weight.shape:
                self.weight = nn.Parameter(
                    torch.empty(
                        incoming_weight.shape,
                        device=self.weight.device,
                        dtype=self.weight.dtype,
                    )
                )

        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )
        if not error_msgs:
            self.validate_state()


__all__ = ["ClassifierOutput", "PersistentCosineClassifier"]









# from __future__ import annotations

# """Persistent normalized cosine classifier for NECIL-HSI.

# Role
# ----
# This module performs classification only.

# The HSI backbone/model provides the canonical ordered representation

#     Q(x) in R^{K x D}.

# The classifier flattens Q only for the decision operation,

#     z(x) = vec(Q(x)) in R^{KD},

# and compares its direction with one trainable row per seen class:

#     s_c(x) = < z/||z||_2 , w_c/||w_c||_2 >.

# Prediction is

#     y_hat = class_ids[argmax_c s_c(x)].

# There is no bias, class-specific scale, temperature, margin, geometry energy,
# replay state, preservation state, or task head in this component.  Class rows
# persist across phases and new rows are appended before the optimizer for a new
# phase is constructed.
# """

# from dataclasses import dataclass
# from numbers import Integral
# from typing import Sequence

# import torch
# import torch.nn as nn

# Tensor = torch.Tensor


# @dataclass(frozen=True)
# class ClassifierOutput:
#     cosine_similarity: Tensor
#     class_ids: Tensor
#     prediction: Tensor


# def _parse_class_ids(
#     class_ids: Sequence[int] | Tensor,
#     *,
#     allow_empty: bool,
# ) -> tuple[int, ...]:
#     if torch.is_tensor(class_ids):
#         if class_ids.ndim != 1:
#             raise ValueError("class_ids tensor must be rank one")
#         if class_ids.dtype != torch.long:
#             raise TypeError("class_ids tensor must use torch.long")
#         source = class_ids.detach().cpu().tolist()
#     else:
#         source = list(class_ids)

#     parsed: list[int] = []
#     for value in source:
#         if isinstance(value, bool) or not isinstance(value, Integral):
#             raise TypeError("class_ids must contain integer class IDs")
#         value = int(value)
#         if value < 0:
#             raise ValueError("class_ids must be non-negative")
#         parsed.append(value)

#     result = tuple(parsed)
#     if not allow_empty and not result:
#         raise ValueError("class_ids cannot be empty")
#     if len(result) != len(set(result)):
#         raise ValueError("class_ids must be unique")
#     return result


# def _unit_normalize(value: Tensor, *, dim: int) -> Tensor:
#     if not torch.is_floating_point(value) or value.is_complex():
#         raise TypeError("normalization input must be real floating point")
#     norm = torch.linalg.vector_norm(value, ord=2, dim=dim, keepdim=True)
#     if not bool(torch.isfinite(norm).all()):
#         raise RuntimeError("normalization norm contains NaN or Inf")
#     floor = torch.as_tensor(
#         torch.finfo(value.dtype).tiny,
#         device=value.device,
#         dtype=value.dtype,
#     )
#     return value / norm.clamp_min(floor)


# class PersistentCosineClassifier(nn.Module):
#     """One normalized classifier whose class rows expand across phases."""

#     def __init__(
#         self,
#         *,
#         spectral_token_count: int,
#         spectral_token_dim: int,
#         class_ids: Sequence[int] | Tensor = (),
#     ) -> None:
#         super().__init__()

#         if (
#             isinstance(spectral_token_count, bool)
#             or not isinstance(spectral_token_count, Integral)
#             or int(spectral_token_count) <= 0
#         ):
#             raise ValueError("spectral_token_count must be a positive integer")
#         if (
#             isinstance(spectral_token_dim, bool)
#             or not isinstance(spectral_token_dim, Integral)
#             or int(spectral_token_dim) <= 0
#         ):
#             raise ValueError("spectral_token_dim must be a positive integer")

#         self.spectral_token_count = int(spectral_token_count)
#         self.spectral_token_dim = int(spectral_token_dim)
#         self.input_dim = self.spectral_token_count * self.spectral_token_dim

#         parsed = _parse_class_ids(class_ids, allow_empty=True)
#         self.register_buffer(
#             "class_ids",
#             torch.tensor(parsed, dtype=torch.long),
#             persistent=True,
#         )
#         self.weight = nn.Parameter(torch.empty(len(parsed), self.input_dim))
#         self._initialize_rows(self.weight)
#         self.validate_state()

#     @staticmethod
#     def _initialize_rows(rows: Tensor) -> None:
#         # Only the direction matters because rows are normalized in forward().
#         if rows.numel() > 0:
#             nn.init.normal_(rows, mean=0.0, std=1.0)

#     @property
#     def class_count(self) -> int:
#         return int(self.class_ids.numel())

#     @property
#     def seen_class_ids(self) -> tuple[int, ...]:
#         return tuple(int(v) for v in self.class_ids.detach().cpu().tolist())

#     def validate_state(self) -> bool:
#         if self.class_ids.ndim != 1 or self.class_ids.dtype != torch.long:
#             raise RuntimeError("classifier class_ids must be rank-one torch.long")
#         if bool((self.class_ids < 0).any()):
#             raise RuntimeError("classifier class_ids must be non-negative")
#         if self.class_ids.unique().numel() != self.class_ids.numel():
#             raise RuntimeError("classifier class_ids must be unique")

#         expected = (self.class_count, self.input_dim)
#         if tuple(self.weight.shape) != expected:
#             raise RuntimeError(
#                 f"classifier weight must have shape {expected}, got "
#                 f"{tuple(self.weight.shape)}"
#             )
#         if not torch.is_floating_point(self.weight) or self.weight.is_complex():
#             raise RuntimeError("classifier weight must be real floating point")
#         if not bool(torch.isfinite(self.weight).all()):
#             raise RuntimeError("classifier weight contains NaN or Inf")
#         return True

#     def add_classes(self, class_ids: Sequence[int] | Tensor) -> None:
#         """Append unseen rows while copying every existing classifier row exactly."""
#         new_ids = _parse_class_ids(class_ids, allow_empty=False)
#         existing = set(self.seen_class_ids)
#         duplicates = [class_id for class_id in new_ids if class_id in existing]
#         if duplicates:
#             raise ValueError(f"classifier already contains class IDs {duplicates}")

#         new_rows = torch.empty(
#             len(new_ids),
#             self.input_dim,
#             device=self.weight.device,
#             dtype=self.weight.dtype,
#         )
#         self._initialize_rows(new_rows)

#         with torch.no_grad():
#             combined_weight = torch.cat((self.weight.detach(), new_rows), dim=0)
#             combined_ids = torch.cat(
#                 (
#                     self.class_ids,
#                     torch.tensor(
#                         new_ids,
#                         device=self.class_ids.device,
#                         dtype=torch.long,
#                     ),
#                 ),
#                 dim=0,
#             )

#         # Expansion must happen before creating that phase's optimizer because
#         # the Parameter object is intentionally replaced here.
#         self.weight = nn.Parameter(combined_weight)
#         self.class_ids = combined_ids
#         self.validate_state()

#     def _selected_rows(
#         self,
#         class_ids: Sequence[int] | Tensor | None,
#     ) -> tuple[Tensor, Tensor]:
#         if self.class_count == 0:
#             raise RuntimeError("classifier has no classes")
#         if class_ids is None:
#             return self.weight, self.class_ids

#         requested = _parse_class_ids(class_ids, allow_empty=False)
#         row_by_id = {
#             int(class_id): row
#             for row, class_id in enumerate(self.class_ids.detach().cpu().tolist())
#         }
#         missing = [class_id for class_id in requested if class_id not in row_by_id]
#         if missing:
#             raise KeyError(f"classifier does not contain class IDs {missing}")
#         indices = torch.tensor(
#             [row_by_id[class_id] for class_id in requested],
#             device=self.class_ids.device,
#             dtype=torch.long,
#         )
#         return (
#             self.weight.index_select(0, indices),
#             self.class_ids.index_select(0, indices),
#         )

#     def target_indices(self, labels: Tensor) -> Tensor:
#         labels = torch.as_tensor(labels, device=self.class_ids.device)
#         if labels.ndim != 1 or labels.numel() == 0:
#             raise ValueError("labels must be non-empty rank-one")
#         if labels.dtype != torch.long:
#             raise TypeError("labels must use torch.long")
#         if self.class_count == 0:
#             raise RuntimeError("classifier has no classes")

#         matches = labels.unsqueeze(1).eq(self.class_ids.unsqueeze(0))
#         counts = matches.sum(dim=1)
#         if not bool(counts.eq(1).all()):
#             missing = labels[counts.eq(0)].detach().cpu().unique().tolist()
#             raise ValueError(
#                 f"labels contain class IDs absent from classifier: {missing}"
#             )
#         return matches.to(torch.long).argmax(dim=1)

#     def forward(
#         self,
#         ordered_features: Tensor,
#         *,
#         class_ids: Sequence[int] | Tensor | None = None,
#     ) -> ClassifierOutput:
#         self.validate_state()
#         selected_weight, selected_ids = self._selected_rows(class_ids)

#         features = torch.as_tensor(ordered_features)
#         expected_tail = (self.spectral_token_count, self.spectral_token_dim)
#         if (
#             features.ndim != 3
#             or features.size(0) == 0
#             or tuple(features.shape[1:]) != expected_tail
#         ):
#             raise ValueError(
#                 "ordered_features must have shape "
#                 f"[N,{expected_tail[0]},{expected_tail[1]}]"
#             )
#         if not torch.is_floating_point(features) or features.is_complex():
#             raise TypeError("ordered_features must be real floating point")
#         if features.device != selected_weight.device:
#             raise ValueError("ordered_features and classifier must share a device")
#         if features.dtype != selected_weight.dtype:
#             raise TypeError("ordered_features and classifier must share a dtype")
#         if not bool(torch.isfinite(features).all()):
#             raise ValueError("ordered_features contain NaN or Inf")

#         flattened = features.reshape(features.size(0), self.input_dim)
#         feature_direction = _unit_normalize(flattened, dim=1)
#         weight_direction = _unit_normalize(selected_weight, dim=1)
#         cosine = feature_direction @ weight_direction.transpose(0, 1)
#         if not bool(torch.isfinite(cosine).all()):
#             raise RuntimeError("cosine classifier produced NaN or Inf")

#         prediction = selected_ids.index_select(0, cosine.argmax(dim=1))
#         return ClassifierOutput(
#             cosine_similarity=cosine,
#             class_ids=selected_ids,
#             prediction=prediction,
#         )

#     def _load_from_state_dict(
#         self,
#         state_dict,
#         prefix,
#         local_metadata,
#         strict,
#         missing_keys,
#         unexpected_keys,
#         error_msgs,
#     ):
#         ids_key = prefix + "class_ids"
#         weight_key = prefix + "weight"
#         incoming_ids = state_dict.get(ids_key)
#         incoming_weight = state_dict.get(weight_key)

#         if incoming_ids is not None:
#             if incoming_ids.ndim != 1 or incoming_ids.dtype != torch.long:
#                 error_msgs.append(f"{ids_key} must be rank-one torch.long")
#             elif self.class_ids.shape != incoming_ids.shape:
#                 self.class_ids = torch.empty(
#                     incoming_ids.shape,
#                     device=self.class_ids.device,
#                     dtype=torch.long,
#                 )

#         if incoming_weight is not None:
#             if incoming_weight.ndim != 2 or incoming_weight.size(1) != self.input_dim:
#                 error_msgs.append(
#                     f"{weight_key} must have shape [C,{self.input_dim}]"
#                 )
#             elif self.weight.shape != incoming_weight.shape:
#                 self.weight = nn.Parameter(
#                     torch.empty(
#                         incoming_weight.shape,
#                         device=self.weight.device,
#                         dtype=self.weight.dtype,
#                     )
#                 )

#         super()._load_from_state_dict(
#             state_dict,
#             prefix,
#             local_metadata,
#             strict,
#             missing_keys,
#             unexpected_keys,
#             error_msgs,
#         )
#         if not error_msgs:
#             self.validate_state()


# __all__ = ["ClassifierOutput", "PersistentCosineClassifier"]









# from __future__ import annotations

# """Parameter-free equal-rule classifier over a rich persistent HSI structure.

# Persistent class state
# ----------------------
# The bank stores only

#     M_c = {I_c,S_c,N_c},

# where

#     I_c = E[Q|c]                                              [K,D]

#     S_c[k,l,d]
#         = E[(Q[k,d]-I_c[k,d])(Q[l,d]-I_c[l,d])|c]             [K,K,D].

# The classifier derives, without storing additional state,

#     A_c[k,d] = S_c[k,k,d]                                    [K,D]

#     C_c[k,l] = mean_d S_c[k,l,d]                             [K,K].

# Why the projections are used
# ----------------------------
# S_c is a population second-order expectation.  For one query x,
# Delta_c(x)[:,d] Delta_c(x)[:,d]^T is only one rank-one realization for each
# feature channel d.  Directly matching that [K,K,D] tensor to S_c is therefore
# not a sound membership rule.

# Instead, the equal-rule classifier uses two statistically more stable
# single-query views:

#     Delta_c(x) = Q(x) - I_c

#     E_identity(c,x)
#         = mean(Delta_c(x)^2)

#     E_feature(c,x)
#         = RMS(Delta_c(x)^2 - A_c)

#     C_x^(c)
#         = (1/D) Delta_c(x) Delta_c(x)^T

#     E_cross(c,x)
#         = RMS_offdiag(C_x^(c) - C_c).

# The diagonal of C_c is excluded because

#     C_c[k,k] = mean_d A_c[k,d],

# so including it would duplicate variation already represented by A_c.

# The deployed energy is

#     E_c(x)
#         = E_identity(c,x)
#         + E_feature(c,x)
#         + E_cross(c,x),

#     prediction = argmin_c E_c(x).

# No classifier parameters, learned/tuned component weights, temperature, margin,
# calibration head, task head, pairwise boundary state, replay state, Gaussian
# state, or exemplar state are introduced.  The rich S_c state is still retained
# for future continual-learning preservation through

#     R_cd = {I_c-I_d,S_c-S_d}.
# """

# from dataclasses import dataclass
# import math
# from typing import Sequence

# import torch
# import torch.nn as nn

# from models.geometry_bank import HSIClassDiscriminativeGeometry

# Tensor = torch.Tensor


# @dataclass(frozen=True)
# class GeometryClassifierOutput:
#     """Equal-rule classification result in ``class_ids`` column order."""

#     scores: Tensor
#     total_energy: Tensor
#     identity_energy: Tensor
#     feature_variation_energy: Tensor
#     cross_region_covariation_energy: Tensor
#     class_ids: Tensor
#     prediction: Tensor


# class GeometryClassifier(nn.Module):
#     """Stateless equal-rule classifier over derived views of persistent {I,S}."""

#     def __init__(self) -> None:
#         super().__init__()

#     @staticmethod
#     def _validate_features(
#         ordered_features: Tensor,
#         *,
#         geometry_bank: HSIClassDiscriminativeGeometry,
#     ) -> Tensor:
#         features = torch.as_tensor(ordered_features)

#         expected_tail = (
#             geometry_bank.spectral_token_count,
#             geometry_bank.spectral_token_dim,
#         )

#         if (
#             features.ndim != 3
#             or features.size(0) == 0
#             or tuple(features.shape[1:]) != expected_tail
#         ):
#             raise ValueError(
#                 "ordered_features must have shape "
#                 f"[N,{expected_tail[0]},{expected_tail[1]}]"
#             )

#         if not torch.is_floating_point(features) or features.is_complex():
#             raise TypeError(
#                 "ordered_features must be a real floating-point tensor"
#             )

#         if features.dtype != geometry_bank.dtype:
#             raise TypeError(
#                 "ordered_features and geometry bank must share a dtype"
#             )

#         if features.device != geometry_bank.device:
#             raise ValueError(
#                 "ordered_features and geometry bank must share a device"
#             )

#         if not bool(torch.isfinite(features).all()):
#             raise ValueError(
#                 "ordered_features contain NaN or Inf"
#             )

#         return features

#     @staticmethod
#     def _rms(value: Tensor) -> Tensor:
#         """RMS over all non-batch entries without an epsilon or tuned floor."""
#         if value.ndim < 2 or value.size(0) == 0:
#             raise ValueError(
#                 "RMS input must have a non-empty batch dimension and features"
#             )

#         flattened = value.reshape(value.size(0), -1)
#         if flattened.size(1) == 0:
#             raise ValueError(
#                 "RMS input must contain at least one feature entry"
#             )

#         return torch.linalg.vector_norm(
#             flattened,
#             ord=2,
#             dim=1,
#         ) / math.sqrt(float(flattened.size(1)))

#     @staticmethod
#     def _class_energy(
#         ordered_features: Tensor,
#         *,
#         identity: Tensor,
#         feature_variation: Tensor,
#         cross_region_covariation: Tensor,
#     ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
#         """Compute one candidate class discrepancy for every query sample.

#         Shapes
#         ------
#         ordered_features:             [N,K,D]
#         identity:                     [K,D]
#         feature_variation:            [K,D]
#         cross_region_covariation:     [K,K]

#         Returns
#         -------
#         total_energy:                 [N]
#         identity_energy:              [N]
#         feature_variation_energy:     [N]
#         cross_region_energy:          [N]
#         """
#         if identity.shape != ordered_features.shape[1:]:
#             raise RuntimeError(
#                 "stored class identity has the wrong shape"
#             )

#         if feature_variation.shape != ordered_features.shape[1:]:
#             raise RuntimeError(
#                 "stored feature variation has the wrong shape"
#             )

#         token_count = ordered_features.size(1)
#         feature_dim = ordered_features.size(2)

#         if token_count < 2:
#             raise RuntimeError(
#                 "cross-region classification requires at least two spectral regions"
#             )

#         if cross_region_covariation.shape != (
#             token_count,
#             token_count,
#         ):
#             raise RuntimeError(
#                 "stored cross-region co-variation has the wrong shape"
#             )

#         # Delta_c(x) = Q(x) - I_c.
#         residual = ordered_features - identity.unsqueeze(0)
#         residual_square = residual.square()

#         # Identity discrepancy.  Units: feature^2.
#         identity_energy = residual_square.mean(dim=(1, 2))

#         # A_c describes the expected feature-wise centered squared residual.
#         # The RMS difference has the same feature^2 units as identity_energy.
#         feature_difference = (
#             residual_square
#             - feature_variation.unsqueeze(0)
#         )
#         feature_variation_energy = GeometryClassifier._rms(
#             feature_difference
#         )

#         # Per-query cross-region second-order structure:
#         #     (1/D) Delta_c Delta_c^T.
#         sample_cross_region = torch.einsum(
#             "nkd,nld->nkl",
#             residual,
#             residual,
#         ) / float(feature_dim)

#         cross_difference = (
#             sample_cross_region
#             - cross_region_covariation.unsqueeze(0)
#         )

#         # diag(C_c)[k] = mean_d A_c[k,d], so the diagonal repeats aggregated
#         # feature-variation information.  Only k != l is cross-region relation.
#         off_diagonal_mask = ~torch.eye(
#             token_count,
#             device=ordered_features.device,
#             dtype=torch.bool,
#         )
#         cross_off_diagonal = cross_difference[:, off_diagonal_mask]
#         cross_region_energy = GeometryClassifier._rms(
#             cross_off_diagonal
#         )

#         # All three terms have the same physical units and use dimension-normalized
#         # reductions.  Their direct sum introduces no tunable component weights.
#         total_energy = (
#             identity_energy
#             + feature_variation_energy
#             + cross_region_energy
#         )

#         return (
#             total_energy,
#             identity_energy,
#             feature_variation_energy,
#             cross_region_energy,
#         )

#     @staticmethod
#     def _validate_output(
#         output: GeometryClassifierOutput,
#     ) -> GeometryClassifierOutput:
#         if not isinstance(output, GeometryClassifierOutput):
#             raise TypeError(
#                 "output must be GeometryClassifierOutput"
#             )

#         energy_tensors = (
#             output.total_energy,
#             output.identity_energy,
#             output.feature_variation_energy,
#             output.cross_region_covariation_energy,
#         )

#         if (
#             output.scores.ndim != 2
#             or output.scores.size(0) == 0
#             or output.scores.size(1) == 0
#         ):
#             raise RuntimeError(
#                 "classifier scores must be non-empty [N,C]"
#             )

#         expected_shape = output.scores.shape

#         if (
#             not torch.is_floating_point(output.scores)
#             or output.scores.is_complex()
#             or not bool(torch.isfinite(output.scores).all())
#         ):
#             raise RuntimeError(
#                 "classifier scores must be finite real floating point"
#             )

#         for tensor in energy_tensors:
#             if tensor.shape != expected_shape:
#                 raise RuntimeError(
#                     "all classifier energies must have shape [N,C]"
#                 )
#             if (
#                 not torch.is_floating_point(tensor)
#                 or tensor.is_complex()
#                 or not bool(torch.isfinite(tensor).all())
#             ):
#                 raise RuntimeError(
#                     "classifier energies must be finite real floating point"
#                 )
#             if bool((tensor < 0).any()):
#                 raise RuntimeError(
#                     "classifier energies cannot be negative"
#                 )

#         if not bool(torch.equal(output.scores, -output.total_energy)):
#             raise RuntimeError(
#                 "scores must equal negative total energy"
#             )

#         if not bool(
#             torch.equal(
#                 output.total_energy,
#                 output.identity_energy
#                 + output.feature_variation_energy
#                 + output.cross_region_covariation_energy,
#             )
#         ):
#             raise RuntimeError(
#                 "total_energy must equal identity + feature-variation "
#                 "+ cross-region energies"
#             )

#         class_ids = output.class_ids
#         prediction = output.prediction

#         if (
#             class_ids.ndim != 1
#             or class_ids.shape != (output.scores.size(1),)
#             or class_ids.dtype != torch.long
#             or class_ids.device != output.scores.device
#         ):
#             raise RuntimeError(
#                 "class_ids must be [C] torch.long on the score device"
#             )

#         if (
#             class_ids.unique().numel() != class_ids.numel()
#             or bool((class_ids < 0).any())
#         ):
#             raise RuntimeError(
#                 "class_ids must contain unique non-negative IDs"
#             )

#         if (
#             prediction.ndim != 1
#             or prediction.shape != (output.scores.size(0),)
#             or prediction.dtype != torch.long
#             or prediction.device != output.scores.device
#         ):
#             raise RuntimeError(
#                 "prediction must be [N] torch.long on the score device"
#             )

#         expected_prediction = class_ids.index_select(
#             0,
#             output.total_energy.argmin(dim=1),
#         )

#         if not bool(torch.equal(prediction, expected_prediction)):
#             raise RuntimeError(
#                 "prediction must equal argmin of total structural energy"
#             )

#         return output

#     def forward(
#         self,
#         ordered_features: Tensor,
#         *,
#         geometry_bank: HSIClassDiscriminativeGeometry,
#         class_ids: Sequence[int] | Tensor | None = None,
#     ) -> GeometryClassifierOutput:
#         """Classify Q(x) against requested persistent {I,S,N} class states."""
#         if not isinstance(
#             geometry_bank,
#             HSIClassDiscriminativeGeometry,
#         ):
#             raise TypeError(
#                 "geometry_bank must be HSIClassDiscriminativeGeometry"
#             )

#         features = self._validate_features(
#             ordered_features,
#             geometry_bank=geometry_bank,
#         )

#         # The bank owns class-ID validation and row lookup.  The classifier only
#         # consumes the requested class structures in the returned order.
#         geometry = geometry_bank.class_geometries(class_ids)

#         total_columns: list[Tensor] = []
#         identity_columns: list[Tensor] = []
#         feature_columns: list[Tensor] = []
#         cross_columns: list[Tensor] = []

#         for row in range(geometry.class_ids.numel()):
#             (
#                 total_energy,
#                 identity_energy,
#                 feature_energy,
#                 cross_energy,
#             ) = self._class_energy(
#                 features,
#                 identity=geometry.identity[row],
#                 feature_variation=geometry.feature_variation[row],
#                 cross_region_covariation=(
#                     geometry.cross_region_covariation[row]
#                 ),
#             )

#             total_columns.append(total_energy)
#             identity_columns.append(identity_energy)
#             feature_columns.append(feature_energy)
#             cross_columns.append(cross_energy)

#         total_energy = torch.stack(total_columns, dim=1)
#         identity_energy = torch.stack(identity_columns, dim=1)
#         feature_variation_energy = torch.stack(feature_columns, dim=1)
#         cross_region_covariation_energy = torch.stack(
#             cross_columns,
#             dim=1,
#         )

#         scores = -total_energy
#         output_class_ids = geometry.class_ids
#         prediction = output_class_ids.index_select(
#             0,
#             total_energy.argmin(dim=1),
#         )

#         return self._validate_output(
#             GeometryClassifierOutput(
#                 scores=scores,
#                 total_energy=total_energy,
#                 identity_energy=identity_energy,
#                 feature_variation_energy=feature_variation_energy,
#                 cross_region_covariation_energy=(
#                     cross_region_covariation_energy
#                 ),
#                 class_ids=output_class_ids,
#                 prediction=prediction,
#             )
#         )


# __all__ = [
#     "GeometryClassifier",
#     "GeometryClassifierOutput",
# ]
