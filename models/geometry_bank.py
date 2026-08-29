from __future__ import annotations

"""Persistent HSI class-discriminative statistics in the single Q-space.

The model provides one canonical ordered spectral-spatial representation

    Q(x) in R^{K x D}.

For class c,

    I_c = E[Q | c]                                           [K,D]

    S_c[k,l,d]
        = E[(Q[k,d]-I_c[k,d])(Q[l,d]-I_c[l,d]) | c]          [K,K,D].

The persistent class state is

    G_c = {I_c, S_c, N_c}.

These are statistics of Q, not a second sample representation and not a second
classifier.  They retain class identity, feature-resolved intra-class spectral
variation, and cross-region relations.  Inter-class relations are derived on
demand from the same states.

During incremental learning one common orthogonal spectral-region coordinate
operator R may be inferred from paired current-class Q before/after backbone
adaptation.  Historical statistics then follow the transformation law induced
by that same Q movement:

    I_c <- R I_c,
    S_c[:,:,d] <- R S_c[:,:,d] R^T.

No independent geometry correction is estimated here.  This module only
estimates, stores, retrieves, derives, transports, and validates class
statistics.  It never classifies or performs optimization.
"""

from dataclasses import dataclass
from numbers import Integral
from typing import Iterable, Sequence

import torch
import torch.nn as nn

Tensor = torch.Tensor


# -----------------------------------------------------------------------------
# Public views
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class ClassGeometryView:
    """Detached persistent class state {I_c,S_c,N_c}."""

    class_id: int
    sample_count: int
    identity: Tensor
    structure: Tensor

    @property
    def feature_variation(self) -> Tensor:
        """Derived A_c[k,d] = S_c[k,k,d]."""
        k = int(self.structure.size(0))
        index = torch.arange(k, device=self.structure.device)
        return self.structure[index, index, :]

    @property
    def cross_region_covariation(self) -> Tensor:
        """Derived aggregate C_c[k,l] = mean_d S_c[k,l,d]."""
        return self.structure.mean(dim=2)


@dataclass(frozen=True)
class ClassGeometryBatch:
    """Stacked class states in ``class_ids`` column order."""

    class_ids: Tensor
    sample_counts: Tensor
    identity: Tensor
    structure: Tensor

    @property
    def feature_variation(self) -> Tensor:
        """Derived stacked A_c with shape [C,K,D]."""
        k = int(self.structure.size(1))
        index = torch.arange(k, device=self.structure.device)
        return self.structure[:, index, index, :]

    @property
    def cross_region_covariation(self) -> Tensor:
        """Derived stacked aggregate C_c with shape [C,K,K]."""
        return self.structure.mean(dim=3)


@dataclass(frozen=True)
class InterClassGeometryView:
    """Derived inter-class structure R_cd={Delta I_cd, Delta S_cd}."""

    left_class_id: int
    right_class_id: int
    identity_difference: Tensor
    structure_difference: Tensor

    @property
    def feature_variation_difference(self) -> Tensor:
        """Derived Delta A_cd from the diagonal of Delta S_cd."""
        k = int(self.structure_difference.size(0))
        index = torch.arange(k, device=self.structure_difference.device)
        return self.structure_difference[index, index, :]

    @property
    def cross_region_covariation_difference(self) -> Tensor:
        """Derived aggregate Delta C_cd = mean_d Delta S_cd."""
        return self.structure_difference.mean(dim=2)


# -----------------------------------------------------------------------------
# Validation helpers
# -----------------------------------------------------------------------------


def _integer(
    value: object,
    name: str,
    *,
    minimum: int,
) -> int:
    if torch.is_tensor(value):
        if value.ndim != 0:
            raise TypeError(f"{name} must be an integer scalar")
        if (
            value.dtype == torch.bool
            or value.is_complex()
            or torch.is_floating_point(value)
        ):
            raise TypeError(f"{name} must be an integer scalar")
        value = value.item()

    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")

    result = int(value)
    if result < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return result


def _class_id_tuple(
    values: Sequence[int] | Tensor | Iterable[int],
    *,
    name: str,
) -> tuple[int, ...]:
    if torch.is_tensor(values):
        if values.ndim != 1 or values.numel() == 0:
            raise ValueError(f"{name} tensor must be non-empty and rank one")
        if values.dtype != torch.long:
            raise TypeError(f"{name} tensor must use torch.long")
        source = values.detach().cpu().tolist()
    else:
        source = list(values)

    result = tuple(
        _integer(value, name, minimum=0)
        for value in source
    )

    if not result:
        raise ValueError(f"{name} cannot be empty")
    if len(result) != len(set(result)):
        raise ValueError(f"{name} must contain unique class IDs")
    return result


def _tolerances(dtype: torch.dtype) -> tuple[float, float]:
    if dtype == torch.float32:
        return 3e-5, 3e-6
    if dtype == torch.float64:
        return 1e-10, 1e-12
    raise TypeError("only float32 and float64 are supported")


# -----------------------------------------------------------------------------
# Structure bank
# -----------------------------------------------------------------------------


class HSIClassDiscriminativeGeometry(nn.Module):
    """Store one feature-resolved HSI class structure per seen class."""

    def __init__(
        self,
        *,
        spectral_token_count: int,
        spectral_token_dim: int,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()

        self.spectral_token_count = _integer(
            spectral_token_count,
            "spectral_token_count",
            minimum=2,
        )
        self.spectral_token_dim = _integer(
            spectral_token_dim,
            "spectral_token_dim",
            minimum=1,
        )

        if dtype not in (torch.float32, torch.float64):
            raise TypeError(
                "geometry dtype must be torch.float32 or torch.float64"
            )

        self.register_buffer(
            "_floating_reference",
            torch.empty(0, device=device, dtype=dtype),
            persistent=False,
        )
        self.register_buffer(
            "_token_count_contract",
            torch.tensor(
                self.spectral_token_count,
                device=device,
                dtype=torch.long,
            ),
        )
        self.register_buffer(
            "_token_dim_contract",
            torch.tensor(
                self.spectral_token_dim,
                device=device,
                dtype=torch.long,
            ),
        )

        self.register_buffer(
            "class_ids",
            torch.empty(0, device=device, dtype=torch.long),
        )
        self.register_buffer(
            "sample_counts",
            torch.empty(0, device=device, dtype=torch.long),
        )
        self.register_buffer(
            "identity_bank",
            torch.empty(
                0,
                self.spectral_token_count,
                self.spectral_token_dim,
                device=device,
                dtype=dtype,
            ),
        )
        self.register_buffer(
            "structure_bank",
            torch.empty(
                0,
                self.spectral_token_count,
                self.spectral_token_count,
                self.spectral_token_dim,
                device=device,
                dtype=dtype,
            ),
        )

    @property
    def device(self) -> torch.device:
        return self._floating_reference.device

    @property
    def dtype(self) -> torch.dtype:
        return self._floating_reference.dtype

    def __len__(self) -> int:
        return int(self.class_ids.numel())

    @property
    def seen_class_ids(self) -> tuple[int, ...]:
        return tuple(
            int(value)
            for value in self.class_ids.detach().cpu().tolist()
        )

    # ------------------------------------------------------------------
    # Inputs
    # ------------------------------------------------------------------

    def _validate_features(self, ordered_features: Tensor) -> Tensor:
        features = torch.as_tensor(ordered_features)
        expected_tail = (
            self.spectral_token_count,
            self.spectral_token_dim,
        )

        if (
            features.ndim != 3
            or features.size(0) == 0
            or tuple(features.shape[1:]) != expected_tail
        ):
            raise ValueError(
                "ordered_features must have shape "
                f"[N,{expected_tail[0]},{expected_tail[1]}]"
            )
        if (
            not torch.is_floating_point(features)
            or features.is_complex()
        ):
            raise TypeError(
                "ordered_features must be a real floating-point tensor"
            )
        if features.dtype != self.dtype:
            raise TypeError(
                "ordered_features and geometry must share a dtype"
            )
        if features.device != self.device:
            raise ValueError(
                "ordered_features and geometry must share a device"
            )
        if not bool(torch.isfinite(features).all()):
            raise ValueError("ordered_features contain NaN or Inf")
        return features

    def _validate_labels(self, labels: Tensor, *, rows: int) -> Tensor:
        value = torch.as_tensor(labels)
        if value.shape != (rows,):
            raise ValueError(f"labels must have shape [{rows}]")
        if value.dtype != torch.long:
            raise TypeError("labels must use torch.long")
        if value.device != self.device:
            raise ValueError("labels and geometry must share a device")
        if bool((value < 0).any()):
            raise ValueError("labels must contain non-negative class IDs")
        return value

    # ------------------------------------------------------------------
    # Persistent-state validation
    # ------------------------------------------------------------------

    def validate_state(self) -> bool:
        class_count = len(self)

        if (
            self._token_count_contract.ndim != 0
            or self._token_count_contract.dtype != torch.long
            or int(self._token_count_contract.item())
            != self.spectral_token_count
        ):
            raise RuntimeError(
                "spectral_token_count contract is invalid"
            )

        if (
            self._token_dim_contract.ndim != 0
            or self._token_dim_contract.dtype != torch.long
            or int(self._token_dim_contract.item())
            != self.spectral_token_dim
        ):
            raise RuntimeError(
                "spectral_token_dim contract is invalid"
            )

        if (
            self.class_ids.shape != (class_count,)
            or self.class_ids.dtype != torch.long
        ):
            raise RuntimeError("class_ids state is invalid")
        if (
            len(set(self.seen_class_ids)) != class_count
            or bool((self.class_ids < 0).any())
        ):
            raise RuntimeError(
                "stored class IDs must be unique and non-negative"
            )

        if (
            self.sample_counts.shape != (class_count,)
            or self.sample_counts.dtype != torch.long
        ):
            raise RuntimeError("sample_counts state is invalid")
        if class_count > 0 and bool((self.sample_counts <= 0).any()):
            raise RuntimeError(
                "stored sample counts must be positive"
            )

        identity_shape = (
            class_count,
            self.spectral_token_count,
            self.spectral_token_dim,
        )
        structure_shape = (
            class_count,
            self.spectral_token_count,
            self.spectral_token_count,
            self.spectral_token_dim,
        )

        if self.identity_bank.shape != identity_shape:
            raise RuntimeError("identity_bank has an invalid shape")
        if self.structure_bank.shape != structure_shape:
            raise RuntimeError("structure_bank has an invalid shape")

        for name, value in (
            ("identity_bank", self.identity_bank),
            ("structure_bank", self.structure_bank),
        ):
            if value.device != self.device:
                raise RuntimeError(f"{name} is on the wrong device")
            if value.dtype != self.dtype:
                raise RuntimeError(f"{name} has the wrong dtype")
            if not bool(torch.isfinite(value).all()):
                raise RuntimeError(f"{name} contains NaN or Inf")

        if class_count == 0:
            return True

        rtol, atol = _tolerances(self.dtype)

        # S_c[k,l,d] = S_c[l,k,d].
        if not bool(
            torch.allclose(
                self.structure_bank,
                self.structure_bank.transpose(1, 2),
                rtol=rtol,
                atol=atol,
            )
        ):
            raise RuntimeError(
                "structure_bank is not symmetric across spectral regions"
            )

        # diag_K(S_c) is A_c and must be non-negative.
        k = self.spectral_token_count
        index = torch.arange(k, device=self.device)
        diagonal = self.structure_bank[:, index, index, :]
        negative_tolerance = max(
            atol,
            rtol
            * float(
                self.structure_bank.abs().amax().detach().cpu().item()
            ),
        )
        if bool((diagonal < -negative_tolerance).any()):
            raise RuntimeError(
                "structure_bank has a negative diagonal beyond "
                "floating-point tolerance"
            )

        # For every class c and feature channel d, S_c[:,:,d] is a covariance
        # matrix over ordered spectral regions and is positive semidefinite.
        matrices = self.structure_bank.permute(0, 3, 1, 2)
        eigenvalues = torch.linalg.eigvalsh(matrices)

        matrix_scale = float(
            self.structure_bank.abs().amax().detach().cpu().item()
        )
        psd_tolerance = max(atol, rtol * matrix_scale)

        if bool((eigenvalues < -psd_tolerance).any()):
            minimum = float(
                eigenvalues.min().detach().cpu().item()
            )
            raise RuntimeError(
                "structure_bank is not positive semidefinite per feature "
                f"channel; minimum eigenvalue={minimum:.3e}"
            )

        return True

    # ------------------------------------------------------------------
    # Class estimation
    # ------------------------------------------------------------------

    def _estimate_class(
        self,
        class_features: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Compute I_c [K,D] and S_c [K,K,D] from class features [N,K,D]."""
        features = self._validate_features(class_features)

        identity = features.mean(dim=0)
        residual = features - identity.unsqueeze(0)

        # Feature-resolved second-order structure:
        # S_c[k,l,d] = E_n[Delta[n,k,d] Delta[n,l,d]].
        structure = torch.einsum(
            "nkd,nld->kld",
            residual,
            residual,
        ) / float(features.size(0))

        # Remove only floating-point asymmetry.
        structure = 0.5 * (
            structure
            + structure.transpose(0, 1)
        )

        if (
            not bool(torch.isfinite(identity).all())
            or not bool(torch.isfinite(structure).all())
        ):
            raise RuntimeError(
                "class structure estimation produced NaN or Inf"
            )

        return identity, structure

    @torch.no_grad()
    def estimate_classes(
        self,
        ordered_features: Tensor,
        labels: Tensor,
        *,
        class_ids: Sequence[int] | Tensor,
    ) -> ClassGeometryBatch:
        """Estimate {I_c,S_c,N_c} without modifying persistent state."""
        self.validate_state()
        features = self._validate_features(ordered_features)
        labels_value = self._validate_labels(labels, rows=features.size(0))
        requested = _class_id_tuple(class_ids, name="class_ids")

        observed = tuple(sorted(
            int(value)
            for value in labels_value.unique().detach().cpu().tolist()
        ))
        if observed != tuple(sorted(requested)):
            raise ValueError(
                f"labels contain class IDs {observed}, expected exactly "
                f"{tuple(sorted(requested))}"
            )

        counts: list[int] = []
        identities: list[Tensor] = []
        structures: list[Tensor] = []
        for class_id in requested:
            mask = labels_value.eq(class_id)
            count = int(mask.sum().item())
            if count <= 0:
                raise RuntimeError(f"class {class_id} has no samples")
            identity, structure = self._estimate_class(features[mask])
            counts.append(count)
            identities.append(identity)
            structures.append(structure)

        return ClassGeometryBatch(
            class_ids=torch.tensor(requested, device=self.device, dtype=torch.long),
            sample_counts=torch.tensor(counts, device=self.device, dtype=torch.long),
            identity=torch.stack(identities, dim=0).detach(),
            structure=torch.stack(structures, dim=0).detach(),
        )

    @torch.no_grad()
    def add_classes(
        self,
        ordered_features: Tensor,
        labels: Tensor,
        *,
        class_ids: Sequence[int] | Tensor,
    ) -> None:
        """Estimate and append {I_c,S_c,N_c} for previously unseen classes."""
        self.validate_state()
        requested = _class_id_tuple(class_ids, name="class_ids")
        overlap = sorted(set(requested).intersection(self.seen_class_ids))
        if overlap:
            raise ValueError(
                "class structure already exists for "
                f"class IDs {overlap}"
            )

        estimate = self.estimate_classes(
            ordered_features,
            labels,
            class_ids=requested,
        )

        # Atomic append after all requested estimates succeeded.
        self.class_ids = torch.cat((self.class_ids, estimate.class_ids), dim=0)
        self.sample_counts = torch.cat(
            (self.sample_counts, estimate.sample_counts), dim=0
        )
        self.identity_bank = torch.cat(
            (self.identity_bank, estimate.identity), dim=0
        )
        self.structure_bank = torch.cat(
            (self.structure_bank, estimate.structure), dim=0
        )
        self.validate_state()

    @torch.no_grad()
    def transport_classes(
        self,
        transform: Tensor,
        *,
        class_ids: Sequence[int] | Tensor,
    ) -> None:
        """Propagate historical {I,S} with the common Q coordinate operator.

        For orthogonal R:

            I_c <- R I_c
            S_c[:,:,d] <- R S_c[:,:,d] R^T.

        Counts and class IDs do not change.  Orthogonal congruence preserves
        symmetry and positive semidefiniteness of each S_c[:,:,d].
        """
        self.validate_state()
        requested = _class_id_tuple(class_ids, name="class_ids")
        missing = sorted(set(requested) - set(self.seen_class_ids))
        if missing:
            raise KeyError(f"class statistics are missing class IDs {missing}")

        r = torch.as_tensor(transform, device=self.device, dtype=self.dtype)
        expected = (self.spectral_token_count, self.spectral_token_count)
        if tuple(r.shape) != expected:
            raise ValueError(f"transform must have shape {expected}")
        if not bool(torch.isfinite(r).all()):
            raise ValueError("transform contains NaN or Inf")
        eye = torch.eye(
            self.spectral_token_count, device=self.device, dtype=self.dtype
        )
        tolerance = 5e-5 if self.dtype == torch.float32 else 1e-10
        if not bool(torch.allclose(
            r.transpose(0, 1) @ r,
            eye,
            rtol=tolerance,
            atol=tolerance,
        )):
            raise ValueError("class-statistic transport must be orthogonal")

        rows = torch.tensor(
            [self._row_for_class(class_id) for class_id in requested],
            device=self.device,
            dtype=torch.long,
        )
        identity = self.identity_bank.index_select(0, rows)
        structure = self.structure_bank.index_select(0, rows)
        moved_identity = torch.einsum("ka,cad->ckd", r, identity)
        moved_structure = torch.einsum(
            "ka,cabd,lb->ckld", r, structure, r
        )
        moved_structure = 0.5 * (
            moved_structure + moved_structure.transpose(1, 2)
        )
        if (
            not bool(torch.isfinite(moved_identity).all())
            or not bool(torch.isfinite(moved_structure).all())
        ):
            raise RuntimeError("class-statistic transport produced NaN or Inf")

        self.identity_bank.index_copy_(0, rows, moved_identity)
        self.structure_bank.index_copy_(0, rows, moved_structure)
        self.validate_state()

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    def _row_for_class(self, class_id: int) -> int:
        class_id = _integer(
            class_id,
            "class_id",
            minimum=0,
        )
        matches = torch.nonzero(
            self.class_ids.eq(class_id),
            as_tuple=False,
        ).flatten()

        if matches.numel() == 0:
            raise KeyError(
                "class structure is not stored for "
                f"class {class_id}"
            )
        if matches.numel() != 1:
            raise RuntimeError(
                "structure bank contains duplicate class IDs"
            )
        return int(matches.item())

    def class_geometry(self, class_id: int) -> ClassGeometryView:
        """Return one detached persistent class state."""
        self.validate_state()
        row = self._row_for_class(class_id)

        return ClassGeometryView(
            class_id=int(self.class_ids[row].item()),
            sample_count=int(self.sample_counts[row].item()),
            identity=self.identity_bank[row].detach().clone(),
            structure=self.structure_bank[row].detach().clone(),
        )

    def class_geometries(
        self,
        class_ids: Sequence[int] | Tensor | None = None,
    ) -> ClassGeometryBatch:
        """Return stacked class states in requested global-class order."""
        self.validate_state()
        if len(self) == 0:
            raise RuntimeError("geometry bank is empty")

        requested = (
            self.seen_class_ids
            if class_ids is None
            else _class_id_tuple(class_ids, name="class_ids")
        )

        rows = torch.tensor(
            [self._row_for_class(class_id) for class_id in requested],
            device=self.device,
            dtype=torch.long,
        )

        return ClassGeometryBatch(
            class_ids=self.class_ids.index_select(
                0, rows
            ).detach().clone(),
            sample_counts=self.sample_counts.index_select(
                0, rows
            ).detach().clone(),
            identity=self.identity_bank.index_select(
                0, rows
            ).detach().clone(),
            structure=self.structure_bank.index_select(
                0, rows
            ).detach().clone(),
        )

    # ------------------------------------------------------------------
    # Derived inter-class structure
    # ------------------------------------------------------------------

    def inter_class_geometry(
        self,
        left_class_id: int,
        right_class_id: int,
    ) -> InterClassGeometryView:
        left_id = _integer(
            left_class_id,
            "left_class_id",
            minimum=0,
        )
        right_id = _integer(
            right_class_id,
            "right_class_id",
            minimum=0,
        )
        if left_id == right_id:
            raise ValueError(
                "inter-class structure requires two different classes"
            )

        left = self.class_geometry(left_id)
        right = self.class_geometry(right_id)

        return InterClassGeometryView(
            left_class_id=left_id,
            right_class_id=right_id,
            identity_difference=left.identity - right.identity,
            structure_difference=left.structure - right.structure,
        )

    # ------------------------------------------------------------------
    # Checkpoint loading
    # ------------------------------------------------------------------

    def _load_from_state_dict(
        self,
        state_dict: dict[str, Tensor],
        prefix: str,
        local_metadata: dict,
        strict: bool,
        missing_keys: list[str],
        unexpected_keys: list[str],
        error_msgs: list[str],
    ) -> None:
        """Resize class-dependent buffers before standard PyTorch loading.

        Old {I,A,C,N} checkpoints are intentionally not converted.  A strict
        load must use a checkpoint created by this {I,S,N} architecture.
        """
        token_count_key = prefix + "_token_count_contract"
        token_dim_key = prefix + "_token_dim_contract"

        if token_count_key in state_dict:
            incoming = torch.as_tensor(state_dict[token_count_key])
            if (
                incoming.ndim != 0
                or incoming.dtype != torch.long
                or int(incoming.detach().cpu().item())
                != self.spectral_token_count
            ):
                error_msgs.append(
                    "checkpoint spectral_token_count does not match this geometry"
                )

        if token_dim_key in state_dict:
            incoming = torch.as_tensor(state_dict[token_dim_key])
            if (
                incoming.ndim != 0
                or incoming.dtype != torch.long
                or int(incoming.detach().cpu().item())
                != self.spectral_token_dim
            ):
                error_msgs.append(
                    "checkpoint spectral_token_dim does not match this geometry"
                )

        class_ids_key = prefix + "class_ids"
        class_count: int | None = None

        if class_ids_key in state_dict:
            incoming_ids = torch.as_tensor(state_dict[class_ids_key])
            if (
                incoming_ids.ndim != 1
                or incoming_ids.dtype != torch.long
            ):
                error_msgs.append(
                    "checkpoint class_ids must be rank-one torch.long"
                )
            else:
                class_count = int(incoming_ids.numel())

        if class_count is not None:
            specifications = {
                "class_ids": ((class_count,), torch.long),
                "sample_counts": ((class_count,), torch.long),
                "identity_bank": (
                    (
                        class_count,
                        self.spectral_token_count,
                        self.spectral_token_dim,
                    ),
                    self.dtype,
                ),
                "structure_bank": (
                    (
                        class_count,
                        self.spectral_token_count,
                        self.spectral_token_count,
                        self.spectral_token_dim,
                    ),
                    self.dtype,
                ),
            }

            for name, (expected_shape, expected_dtype) in specifications.items():
                key = prefix + name
                if key not in state_dict:
                    continue

                incoming = torch.as_tensor(state_dict[key])
                if tuple(incoming.shape) != expected_shape:
                    error_msgs.append(
                        f"checkpoint {name} has shape "
                        f"{tuple(incoming.shape)}, expected {expected_shape}"
                    )
                    continue
                if incoming.dtype != expected_dtype:
                    error_msgs.append(
                        f"checkpoint {name} has dtype {incoming.dtype}, "
                        f"expected {expected_dtype}"
                    )
                    continue

                setattr(
                    self,
                    name,
                    torch.empty(
                        expected_shape,
                        device=self.device,
                        dtype=expected_dtype,
                    ),
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

        try:
            self.validate_state()
        except (TypeError, ValueError, RuntimeError) as exc:
            error_msgs.append(
                f"{prefix}invalid HSI class-discriminative "
                f"structure state: {exc}"
            )


__all__ = [
    "ClassGeometryBatch",
    "ClassGeometryView",
    "HSIClassDiscriminativeGeometry",
    "InterClassGeometryView",
]
