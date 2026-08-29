from __future__ import annotations

"""Persistent angular classifier acting on the single SCOPE-HSI Q-space.

The model provides canonical

    Q(x) in R^{K x D},    RMS(Q)=1,

and the classifier uses

    z(x) = vec(Q(x)),
    u_c  = w_c / ||w_c||_2,
    l_c  = z(x)^T u_c.

There is one persistent classifier for all seen classes and one decision rule
for base and incremental phases.  Class statistics do not form a second
classifier; however, classifier rows and class statistics are decision-relevant
state in the same Q coordinate.  When the Q coordinate is propagated by one
orthogonal spectral-region operator R, historical classifier rows are moved by
that same R so their responses remain coordinate-consistent.

New incremental rows may be initialized from the current class identities
I_c measured in the same Q-space.  This is initialization only; the classifier
remains the trainable decision operator.

There is no task head, old/new inference rule, temperature, margin, bias,
replay state, or independent classifier coordinate.
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

    def add_classes(
        self,
        class_ids: Sequence[int] | Tensor,
        *,
        initial_rows: Tensor | None = None,
    ) -> None:
        """Append unseen rows while preserving every historical row exactly.

        ``initial_rows`` is optional for phase 0.  During an incremental phase
        it should be ``vec(I_c^-)`` for the new classes, where I_c^- is measured
        from the current real TRAIN samples in the phase-start Q coordinate.
        """
        new_ids = _parse_class_ids(class_ids, allow_empty=False)
        existing = set(self.seen_class_ids)
        duplicates = [class_id for class_id in new_ids if class_id in existing]
        if duplicates:
            raise ValueError(f"classifier already contains class IDs {duplicates}")

        if initial_rows is None:
            new_rows = torch.empty(
                len(new_ids),
                self.input_dim,
                device=self.weight.device,
                dtype=self.weight.dtype,
            )
            self._initialize_rows(new_rows)
        else:
            new_rows = torch.as_tensor(
                initial_rows,
                device=self.weight.device,
                dtype=self.weight.dtype,
            )
            expected = (len(new_ids), self.input_dim)
            if tuple(new_rows.shape) != expected:
                raise ValueError(
                    f"initial_rows must have shape {expected}, got "
                    f"{tuple(new_rows.shape)}"
                )
            if not bool(torch.isfinite(new_rows).all()):
                raise ValueError("initial_rows contain NaN or Inf")
            norms = torch.linalg.vector_norm(new_rows, ord=2, dim=1)
            if bool(norms.le(0).any()):
                raise ValueError("initial_rows must have non-zero class directions")
            new_rows = new_rows.detach().clone()

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

    def row_indices(
        self,
        class_ids: Sequence[int] | Tensor,
    ) -> Tensor:
        """Return parameter-row indices for requested global class IDs."""
        requested = _parse_class_ids(class_ids, allow_empty=False)
        row_by_id = {
            int(class_id): row
            for row, class_id in enumerate(self.class_ids.detach().cpu().tolist())
        }
        missing = [class_id for class_id in requested if class_id not in row_by_id]
        if missing:
            raise KeyError(f"classifier does not contain class IDs {missing}")
        return torch.tensor(
            [row_by_id[class_id] for class_id in requested],
            device=self.weight.device,
            dtype=torch.long,
        )

    @torch.no_grad()
    def transport_classes(
        self,
        transform: Tensor,
        *,
        class_ids: Sequence[int] | Tensor,
    ) -> None:
        """Propagate historical classifier rows with the common Q transform.

        For R in R^{K x K}, reshape each row w_c to W_c in R^{K x D} and apply

            W_c <- R W_c.

        R must be orthogonal.  Therefore applying the same R to Q and W keeps
        Frobenius inner products and the deployed angular decision rule
        invariant under that coordinate change.
        """
        self.validate_state()
        rows = self.row_indices(class_ids)
        r = torch.as_tensor(
            transform,
            device=self.weight.device,
            dtype=self.weight.dtype,
        )
        expected = (self.spectral_token_count, self.spectral_token_count)
        if tuple(r.shape) != expected:
            raise ValueError(f"transform must have shape {expected}")
        if not bool(torch.isfinite(r).all()):
            raise ValueError("transform contains NaN or Inf")

        eye = torch.eye(
            self.spectral_token_count,
            device=r.device,
            dtype=r.dtype,
        )
        tolerance = 5e-5 if r.dtype == torch.float32 else 1e-10
        if not bool(torch.allclose(
            r.transpose(0, 1) @ r,
            eye,
            rtol=tolerance,
            atol=tolerance,
        )):
            raise ValueError("classifier transport must be orthogonal")

        selected = self.weight.detach().index_select(0, rows).reshape(
            rows.numel(),
            self.spectral_token_count,
            self.spectral_token_dim,
        )
        moved = torch.einsum("ka,cad->ckd", r, selected).reshape(
            rows.numel(), self.input_dim
        )
        if not bool(torch.isfinite(moved).all()):
            raise RuntimeError("classifier transport produced NaN or Inf")
        self.weight.index_copy_(0, rows, moved)
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
