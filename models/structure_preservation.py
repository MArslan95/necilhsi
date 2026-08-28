from __future__ import annotations

"""Class-discriminative structural gradient preservation for NECIL-HSI.

The persistent class statistics are

    I_c in R^{K x D}
    S_c in R^{K x K x D}
    N_c in N.

They never classify and no exemplar, old patch, old spectrum, per-sample
feature, replay support, or teacher model is retained.

This module has one job: derive a compact historical structural subspace from
{I_c,S_c} and remove the direct current-task gradient component that lies in
that subspace.

Construction
------------
Let m_c = vec(I_c - mean_j I_j).  A compact identity-discriminative basis V is
obtained from the generalized Rayleigh problem

    B v = rho (B + W) v,

where B is the between-class identity scatter and W is the class-balanced
average of the feature-resolved within-class structure.  The problem is solved
exactly inside span{m_c}, so rank(V) <= C-1 and no research rank/threshold is
introduced.

Using only V would let S_c merely reweight identity directions.  To preserve
class-specific HSI variation as well, define

    Delta S_c = S_c - mean_j S_j

and include the structural responses

    Delta S_c V.

The final protected representation space is therefore

    U = orth( [ V, Delta S_1 V, ..., Delta S_C V ] ).

This retains only class-specific variation *around discriminative identity
directions* instead of protecting the full covariance space.  Its rank is
determined solely by numerical linear dependence.

During incremental training,

    g_Q <- (I - U U^T) g_Q.

Hence the direct current-task gradient at canonical Q is orthogonal to the
historical class-discriminative structural subspace.  This is a first-order
representation-gradient constraint; it does not claim exact invariance of old
samples under a nonlinear shared backbone.

Old angular-classifier rows remain trainable.  Their updates are restricted so
their normalized responses on the span of historical class identities are not
changed to first order; new classifier rows remain fully plastic.
"""

from dataclasses import dataclass
from numbers import Integral
from typing import Sequence

import torch

from models.classifier import PersistentCosineClassifier
from models.geometry_bank import HSIClassDiscriminativeGeometry

Tensor = torch.Tensor


@dataclass(frozen=True)
class PreservationSummary:
    old_class_ids: tuple[int, ...]
    feature_dim: int
    identity_discriminative_rank: int
    representation_rank: int
    classifier_identity_rank: int
    generalized_eigenvalues: tuple[float, ...]


def _class_ids(values: Sequence[int] | Tensor) -> tuple[int, ...]:
    source = (
        values.detach().cpu().tolist()
        if torch.is_tensor(values)
        else list(values)
    )
    result: list[int] = []
    for value in source:
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise TypeError("class IDs must be integers")
        class_id = int(value)
        if class_id < 0:
            raise ValueError("class IDs must be non-negative")
        result.append(class_id)

    output = tuple(result)
    if not output:
        raise ValueError("class IDs cannot be empty")
    if len(output) != len(set(output)):
        raise ValueError("class IDs must be unique")
    return output


def _numerical_tolerance(
    scale: Tensor,
    *,
    source_dtype: torch.dtype,
    dimension: int,
) -> Tensor:
    if source_dtype not in (torch.float32, torch.float64):
        raise TypeError("only float32/float64 are supported")
    eps = torch.finfo(source_dtype).eps
    magnitude = (
        torch.max(torch.abs(scale))
        if scale.numel()
        else torch.zeros((), device=scale.device, dtype=scale.dtype)
    )
    return magnitude * float(max(1, int(dimension))) * float(eps)


def _orthonormal_column_basis(
    matrix: Tensor,
    *,
    source_dtype: torch.dtype,
) -> Tensor:
    """Return the numerical column space using only dtype-relative tolerance."""
    value = torch.as_tensor(matrix)
    if value.ndim != 2 or value.size(0) == 0 or value.size(1) == 0:
        raise ValueError("basis input must be non-empty rank-two")
    if not bool(torch.isfinite(value).all()):
        raise ValueError("basis input contains NaN/Inf")

    work = value.to(torch.float64)
    u, singular, _ = torch.linalg.svd(work, full_matrices=False)
    if singular.numel() == 0:
        raise RuntimeError("cannot construct an empty basis")

    tolerance = _numerical_tolerance(
        singular,
        source_dtype=source_dtype,
        dimension=max(value.shape),
    ).to(singular.dtype)
    keep = singular > tolerance
    if not bool(keep.any()):
        raise RuntimeError("class structure has zero numerical rank")

    basis = u[:, keep].to(device=value.device, dtype=value.dtype)
    q, _ = torch.linalg.qr(basis, mode="reduced")
    if not bool(torch.isfinite(q).all()):
        raise RuntimeError("orthonormal basis construction produced NaN/Inf")
    return q


class ClassStructurePreserver:
    """Derive and apply one historical class-structure preservation constraint."""

    def __init__(
        self,
        *,
        geometry_bank: HSIClassDiscriminativeGeometry,
        old_class_ids: Sequence[int] | Tensor,
    ) -> None:
        if not isinstance(geometry_bank, HSIClassDiscriminativeGeometry):
            raise TypeError(
                "geometry_bank must be HSIClassDiscriminativeGeometry"
            )
        geometry_bank.validate_state()

        self.old_class_ids = _class_ids(old_class_ids)
        if len(self.old_class_ids) < 2:
            raise ValueError(
                "structure preservation requires at least two historical classes"
            )

        missing = sorted(
            set(self.old_class_ids) - set(geometry_bank.seen_class_ids)
        )
        if missing:
            raise RuntimeError(f"class statistics lack old classes {missing}")

        batch = geometry_bank.class_geometries(self.old_class_ids)
        identity = batch.identity.detach()
        structure = batch.structure.detach()

        if identity.ndim != 3 or structure.ndim != 4:
            raise RuntimeError("class-statistic tensors have invalid ranks")
        if identity.size(0) != len(self.old_class_ids):
            raise RuntimeError("class-statistic count is inconsistent")
        if structure.shape[:3] != (
            identity.size(0),
            identity.size(1),
            identity.size(1),
        ) or structure.size(3) != identity.size(2):
            raise RuntimeError("identity/structure dimensions are inconsistent")
        if (
            not bool(torch.isfinite(identity).all())
            or not bool(torch.isfinite(structure).all())
        ):
            raise RuntimeError("class statistics contain NaN/Inf")

        self.device = identity.device
        self.dtype = identity.dtype
        self.k = int(identity.size(1))
        self.d = int(identity.size(2))
        self.feature_dim = self.k * self.d

        identity_basis, eigenvalues = self._build_identity_discriminative_basis(
            identity,
            structure,
        )
        self.identity_discriminative_basis = identity_basis
        self.representation_basis = self._build_structural_basis(
            identity_basis,
            structure,
        )

        raw_identity_vectors = identity.reshape(
            identity.size(0), -1
        ).transpose(0, 1)
        self.classifier_identity_basis = _orthonormal_column_basis(
            raw_identity_vectors,
            source_dtype=self.dtype,
        )

        self._generalized_eigenvalues = tuple(
            float(value)
            for value in eigenvalues.detach().cpu().tolist()
        )
        self.validate_state()

    def _within_action(
        self,
        operator: Tensor,
        vectors: Tensor,
    ) -> Tensor:
        """Apply [K,K,D] operator to [F,R] vectors without materializing F x F."""
        if operator.shape != (self.k, self.k, self.d):
            raise RuntimeError("structure operator shape is invalid")
        if vectors.ndim != 2 or vectors.size(0) != self.feature_dim:
            raise RuntimeError("feature-vector matrix shape is invalid")

        shaped = vectors.reshape(self.k, self.d, vectors.size(1))
        acted = torch.einsum("kld,ldr->kdr", operator, shaped)
        return acted.reshape(self.feature_dim, vectors.size(1))

    def _build_identity_discriminative_basis(
        self,
        identity: Tensor,
        structure: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Solve the generalized discriminative problem in centered identity span."""
        class_count = int(identity.size(0))
        centered = identity - identity.mean(dim=0, keepdim=True)
        m = centered.reshape(class_count, self.feature_dim)

        # G[a,b] = <m_a,m_b>.
        gram = m @ m.transpose(0, 1)

        # W[k,l,d] = class-balanced average S_c[k,l,d].
        within = structure.mean(dim=0)
        w_centered = torch.einsum(
            "kld,cld->ckd",
            within,
            centered,
        )
        projected_within = torch.einsum(
            "akd,bkd->ab",
            centered,
            w_centered,
        )

        left = (gram @ gram) / float(class_count)
        right = left + projected_within
        left = 0.5 * (left + left.transpose(0, 1))
        right = 0.5 * (right + right.transpose(0, 1))

        l64 = left.to(torch.float64)
        r64 = right.to(torch.float64)

        eval_r, evec_r = torch.linalg.eigh(r64)
        tol_r = _numerical_tolerance(
            eval_r,
            source_dtype=self.dtype,
            dimension=class_count,
        ).to(eval_r.dtype)
        keep_r = eval_r > tol_r
        if not bool(keep_r.any()):
            raise RuntimeError(
                "historical total discriminative scatter has zero numerical rank"
            )

        whitening = (
            evec_r[:, keep_r]
            / torch.sqrt(eval_r[keep_r]).unsqueeze(0)
        )
        reduced = whitening.transpose(0, 1) @ l64 @ whitening
        reduced = 0.5 * (reduced + reduced.transpose(0, 1))

        rho, eigenvectors = torch.linalg.eigh(reduced)
        order = torch.argsort(rho, descending=True)
        rho = rho.index_select(0, order)
        eigenvectors = eigenvectors.index_select(1, order)

        tol_rho = _numerical_tolerance(
            rho,
            source_dtype=self.dtype,
            dimension=class_count,
        ).to(rho.dtype)
        keep = rho > tol_rho
        if not bool(keep.any()):
            raise RuntimeError(
                "historical classes have no discriminative numerical mode"
            )

        selected_rho = rho[keep]
        bound_tol = _numerical_tolerance(
            selected_rho,
            source_dtype=self.dtype,
            dimension=class_count,
        ).to(selected_rho.dtype)
        if bool((selected_rho < -bound_tol).any()) or bool(
            (selected_rho > 1.0 + bound_tol).any()
        ):
            raise RuntimeError(
                "generalized relevance values violate the PSD [0,1] range"
            )
        selected_rho = selected_rho.clamp(0.0, 1.0)

        coefficients = whitening @ eigenvectors[:, keep]
        candidate = (
            m.to(torch.float64).transpose(0, 1) @ coefficients
        ).to(dtype=self.dtype)

        basis = _orthonormal_column_basis(
            candidate,
            source_dtype=self.dtype,
        )
        if basis.size(1) > class_count - 1:
            raise RuntimeError(
                "identity-discriminative basis exceeds C-1"
            )
        if basis.size(1) != selected_rho.numel():
            raise RuntimeError(
                "generalized modes lost numerical rank after feature-space mapping"
            )
        return basis, selected_rho.to(dtype=self.dtype)

    def _build_structural_basis(
        self,
        identity_basis: Tensor,
        structure: Tensor,
    ) -> Tensor:
        """Add class-specific S_c responses around discriminative identity modes."""
        mean_structure = structure.mean(dim=0)
        candidates: list[Tensor] = [identity_basis]

        for class_index in range(structure.size(0)):
            delta = structure[class_index] - mean_structure
            response = self._within_action(delta, identity_basis)
            if not bool(torch.isfinite(response).all()):
                raise RuntimeError(
                    "class-specific structural response contains NaN/Inf"
                )
            candidates.append(response)

        stacked = torch.cat(candidates, dim=1)
        basis = _orthonormal_column_basis(
            stacked,
            source_dtype=self.dtype,
        )

        theoretical_max = min(
            self.feature_dim,
            identity_basis.size(1) * (len(self.old_class_ids) + 1),
        )
        if basis.size(1) > theoretical_max:
            raise RuntimeError(
                "structural basis exceeds its construction rank bound"
            )
        return basis

    @property
    def summary(self) -> PreservationSummary:
        return PreservationSummary(
            old_class_ids=self.old_class_ids,
            feature_dim=self.feature_dim,
            identity_discriminative_rank=int(
                self.identity_discriminative_basis.size(1)
            ),
            representation_rank=int(self.representation_basis.size(1)),
            classifier_identity_rank=int(
                self.classifier_identity_basis.size(1)
            ),
            generalized_eigenvalues=self._generalized_eigenvalues,
        )

    def validate_state(self) -> bool:
        for name, basis in (
            (
                "identity_discriminative_basis",
                self.identity_discriminative_basis,
            ),
            ("representation_basis", self.representation_basis),
            (
                "classifier_identity_basis",
                self.classifier_identity_basis,
            ),
        ):
            if (
                basis.ndim != 2
                or basis.size(0) != self.feature_dim
                or basis.size(1) <= 0
            ):
                raise RuntimeError(
                    f"{name} has invalid shape {tuple(basis.shape)}"
                )
            if basis.device != self.device or basis.dtype != self.dtype:
                raise RuntimeError(f"{name} has wrong device/dtype")
            if not bool(torch.isfinite(basis).all()):
                raise RuntimeError(f"{name} contains NaN/Inf")

            gram = basis.transpose(0, 1) @ basis
            eye = torch.eye(
                gram.size(0),
                device=gram.device,
                dtype=gram.dtype,
            )
            atol = 4e-5 if self.dtype == torch.float32 else 1e-10
            if not bool(
                torch.allclose(
                    gram,
                    eye,
                    rtol=atol,
                    atol=atol,
                )
            ):
                raise RuntimeError(f"{name} is not orthonormal")

        if (
            self.identity_discriminative_basis.size(1)
            > len(self.old_class_ids) - 1
        ):
            raise RuntimeError(
                "identity-discriminative basis violates C-1 rank bound"
            )
        return True

    def project_representation_gradient(
        self,
        gradient: Tensor,
    ) -> Tensor:
        """Project dL/dQ onto the plastic complement of historical structure."""
        grad = torch.as_tensor(gradient)
        if grad.ndim != 3 or tuple(grad.shape[1:]) != (self.k, self.d):
            raise RuntimeError(
                f"representation gradient must be [N,{self.k},{self.d}]"
            )
        if grad.device != self.device or grad.dtype != self.dtype:
            raise RuntimeError(
                "representation gradient has wrong device/dtype"
            )
        if not bool(torch.isfinite(grad).all()):
            raise RuntimeError(
                "representation gradient contains NaN/Inf"
            )

        flat = grad.reshape(grad.size(0), self.feature_dim)
        basis = self.representation_basis
        protected = (flat @ basis) @ basis.transpose(0, 1)
        result = (flat - protected).reshape_as(grad)

        if not bool(torch.isfinite(result).all()):
            raise RuntimeError(
                "projected representation gradient contains NaN/Inf"
            )
        return result

    def _classifier_old_rows(
        self,
        classifier: PersistentCosineClassifier,
    ) -> Tensor:
        if not isinstance(classifier, PersistentCosineClassifier):
            raise TypeError(
                "classifier must be PersistentCosineClassifier"
            )

        row_by_id = {
            int(class_id): row
            for row, class_id in enumerate(
                classifier.class_ids.detach().cpu().tolist()
            )
        }
        missing = [
            class_id
            for class_id in self.old_class_ids
            if class_id not in row_by_id
        ]
        if missing:
            raise RuntimeError(f"classifier lacks old rows {missing}")

        return torch.tensor(
            [row_by_id[class_id] for class_id in self.old_class_ids],
            device=classifier.weight.device,
            dtype=torch.long,
        )

    def _project_classifier_rows(
        self,
        values: Tensor,
        reference_rows: Tensor,
    ) -> Tensor:
        """Keep old-row changes orthogonal to old identity span and row direction."""
        if values.ndim != 2 or reference_rows.shape != values.shape:
            raise RuntimeError(
                "classifier row projection shape mismatch"
            )

        identity_basis = self.classifier_identity_basis
        projected = values - (
            (values @ identity_basis)
            @ identity_basis.transpose(0, 1)
        )

        reference_perp = reference_rows - (
            (reference_rows @ identity_basis)
            @ identity_basis.transpose(0, 1)
        )
        norm = torch.linalg.vector_norm(
            reference_perp,
            dim=1,
            keepdim=True,
        )
        floor = torch.as_tensor(
            torch.finfo(reference_rows.dtype).tiny,
            device=reference_rows.device,
            dtype=reference_rows.dtype,
        )
        active = norm.squeeze(1) > floor

        if bool(active.any()):
            direction = reference_perp[active] / norm[active]
            active_values = projected[active]
            active_values = active_values - (
                (active_values * direction).sum(
                    dim=1,
                    keepdim=True,
                )
                * direction
            )
            projected = projected.clone()
            projected[active] = active_values

        return projected

    @torch.no_grad()
    def project_old_classifier_gradient_(
        self,
        classifier: PersistentCosineClassifier,
    ) -> None:
        """Project old-row gradients before AdamW preconditioning."""
        gradient = classifier.weight.grad
        if gradient is None:
            raise RuntimeError("classifier gradient is absent")
        if gradient.shape != classifier.weight.shape:
            raise RuntimeError(
                "classifier gradient shape is invalid"
            )

        rows = self._classifier_old_rows(classifier)
        old_gradient = gradient.index_select(0, rows)
        old_weight = classifier.weight.detach().index_select(0, rows)
        projected = self._project_classifier_rows(
            old_gradient,
            old_weight,
        )
        gradient.index_copy_(0, rows, projected)

        if not bool(torch.isfinite(gradient).all()):
            raise RuntimeError(
                "projected classifier gradient contains NaN/Inf"
            )

    @torch.no_grad()
    def project_old_classifier_update_(
        self,
        classifier: PersistentCosineClassifier,
        weight_before_step: Tensor,
    ) -> float:
        """Project the realized old-row AdamW displacement.

        AdamW's elementwise preconditioning can rotate a projected gradient.
        This correction applies the same constraint to the actual parameter
        displacement. New-class rows are untouched.
        """
        before = torch.as_tensor(
            weight_before_step,
            device=classifier.weight.device,
            dtype=classifier.weight.dtype,
        )
        if before.shape != classifier.weight.shape:
            raise RuntimeError(
                "pre-step classifier weight has wrong shape"
            )

        rows = self._classifier_old_rows(classifier)
        old_before = before.index_select(0, rows)
        old_after = classifier.weight.detach().index_select(0, rows)
        delta = old_after - old_before
        projected_delta = self._project_classifier_rows(
            delta,
            old_before,
        )

        classifier.weight.index_copy_(
            0,
            rows,
            old_before + projected_delta,
        )

        identity_basis = self.classifier_identity_basis
        identity_residual = projected_delta @ identity_basis

        old_norm = torch.linalg.vector_norm(
            old_before,
            dim=1,
            keepdim=True,
        ).clamp_min(torch.finfo(old_before.dtype).tiny)
        old_direction = old_before / old_norm
        tangent_residual = (
            projected_delta * old_direction
        ).sum(dim=1)

        maximum = 0.0
        if identity_residual.numel():
            maximum = max(
                maximum,
                float(torch.max(torch.abs(identity_residual)).item()),
            )
        if tangent_residual.numel():
            maximum = max(
                maximum,
                float(torch.max(torch.abs(tangent_residual)).item()),
            )

        if not bool(torch.isfinite(classifier.weight).all()):
            raise RuntimeError(
                "classifier update projection produced NaN/Inf"
            )
        return maximum


__all__ = [
    "ClassStructurePreserver",
    "PreservationSummary",
]
