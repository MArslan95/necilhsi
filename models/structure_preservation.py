from __future__ import annotations

"""One movement operator for the single SCOPE-HSI representation.

The current real TRAIN samples are legal observations in an incremental phase.
For the same samples, collect canonical Q immediately before and after a model
update:

    Q^- , Q^+ in R^{N x K x D}.

Every feature channel supplies a K-dimensional ordered spectral-region vector.
Stack all N*D vectors:

    X^- , X^+ in R^{K x (N D)}.

The phase/epoch movement is the orthogonal Procrustes operator

    R = argmin_{R^T R = I} ||R X^- - X^+||_F^2.

This is not a second representation.  It is an alignment from the previous
coordinate of Q to the current coordinate of the same Q.  The same R must be
used for every historical state derived from Q:

    I -> R I,
    S[:,:,d] -> R S[:,:,d] R^T,
    W -> R W.

No independent correction for I, S, or W is estimated here.
"""

from dataclasses import dataclass

import torch

Tensor = torch.Tensor


@dataclass(frozen=True)
class CoordinateTransportSummary:
    sample_count: int
    spectral_region_count: int
    feature_dimension: int
    observed_rank: int
    raw_feature_change_rms: float
    aligned_feature_residual_rms: float
    transform_distance_from_identity_rms: float
    orthogonality_residual: float


def _rms(value: Tensor) -> float:
    tensor = torch.as_tensor(value)
    if tensor.numel() == 0:
        raise ValueError("cannot compute RMS of an empty tensor")
    if not bool(torch.isfinite(tensor).all()):
        raise ValueError("RMS input contains NaN or Inf")
    return float(torch.sqrt(torch.mean(tensor.to(torch.float64).square())).item())


def _numerical_rank(
    singular_values: Tensor,
    *,
    source_dtype: torch.dtype,
    dimension: int,
) -> int:
    if source_dtype not in (torch.float32, torch.float64):
        raise TypeError("only float32 and float64 are supported")
    values = torch.as_tensor(singular_values)
    if values.ndim != 1 or not bool(torch.isfinite(values).all()):
        raise ValueError("singular values are invalid")
    if values.numel() == 0:
        return 0
    scale = torch.max(torch.abs(values))
    tolerance = (
        scale
        * float(max(1, int(dimension)))
        * float(torch.finfo(source_dtype).eps)
    )
    return int((values > tolerance).sum().item())



def _validate_canonical_q(value: Tensor, *, name: str) -> None:
    q = torch.as_tensor(value)
    if q.ndim != 3 or q.size(0) == 0 or q.size(1) == 0 or q.size(2) == 0:
        raise ValueError(f"{name} must be non-empty [N,K,D]")
    flat = q.reshape(q.size(0), -1)
    rms = torch.linalg.vector_norm(flat, ord=2, dim=1) / (float(flat.size(1)) ** 0.5)
    tolerance = 2e-5 if q.dtype == torch.float32 else 1e-10
    if not bool(torch.allclose(
        rms, torch.ones_like(rms), rtol=tolerance, atol=tolerance
    )):
        raise RuntimeError(f"{name} must be canonical global-RMS Q, not Q_raw")

def _closest_identity_procrustes(
    source: Tensor,
    target: Tensor,
) -> tuple[Tensor, int]:
    """Primary-optimal orthogonal Procrustes, closest to I if non-unique."""
    x = torch.as_tensor(source)
    y = torch.as_tensor(target)
    if x.ndim != 2 or x.size(0) == 0 or x.size(1) == 0 or y.shape != x.shape:
        raise ValueError("source and target must be aligned non-empty [K,M]")
    if x.device != y.device or x.dtype != y.dtype:
        raise ValueError("source and target must share device and dtype")
    if x.dtype not in (torch.float32, torch.float64):
        raise TypeError("source and target must use float32 or float64")
    if not bool(torch.isfinite(x).all()) or not bool(torch.isfinite(y).all()):
        raise ValueError("source or target contains NaN or Inf")

    # min ||R X-Y|| is max tr(R X Y^T).
    cross = x.to(torch.float64) @ y.to(torch.float64).transpose(0, 1)
    u, singular, vh = torch.linalg.svd(cross, full_matrices=True)
    v = vh.transpose(0, 1)
    rank = _numerical_rank(
        singular,
        source_dtype=x.dtype,
        dimension=max(cross.shape),
    )
    k = int(cross.size(0))

    if rank == 0:
        r = torch.eye(k, device=x.device, dtype=torch.float64)
    elif rank == k:
        r = v @ u.transpose(0, 1)
    else:
        # All primary-optimal solutions map U_r to V_r.  The complement is
        # free; choose the completion that maximizes tr(R), equivalently the
        # primary-optimal transform closest to identity in Frobenius norm.
        u_r, v_r = u[:, :rank], v[:, :rank]
        u_0, v_0 = u[:, rank:], v[:, rank:]
        p, _, qh = torch.linalg.svd(
            u_0.transpose(0, 1) @ v_0,
            full_matrices=True,
        )
        q = qh.transpose(0, 1)
        z = q @ p.transpose(0, 1)
        r = v_r @ u_r.transpose(0, 1) + v_0 @ z @ u_0.transpose(0, 1)

    r = r.to(device=x.device, dtype=x.dtype)
    eye = torch.eye(k, device=x.device, dtype=x.dtype)
    tolerance = 5e-5 if x.dtype == torch.float32 else 1e-10
    if not bool(torch.allclose(
        r.transpose(0, 1) @ r,
        eye,
        rtol=tolerance,
        atol=tolerance,
    )):
        raise RuntimeError("estimated coordinate transport is not orthogonal")
    return r, rank


def apply_transport_to_features(
    ordered_features: Tensor,
    transform: Tensor,
) -> Tensor:
    """Apply R to Q [N,K,D]; used for transport diagnostics."""
    q = torch.as_tensor(ordered_features)
    r = torch.as_tensor(transform, device=q.device, dtype=q.dtype)
    if q.ndim != 3 or q.size(0) == 0:
        raise ValueError("ordered_features must be non-empty [N,K,D]")
    if tuple(r.shape) != (q.size(1), q.size(1)):
        raise ValueError("transform shape disagrees with Q")
    result = torch.einsum("ka,nad->nkd", r, q)
    if not bool(torch.isfinite(result).all()):
        raise RuntimeError("feature transport produced NaN or Inf")
    return result


class HSICoordinateTransport:
    """Estimate one common Q-coordinate movement from paired current samples."""

    def __init__(self, q_before: Tensor, q_after: Tensor) -> None:
        before = torch.as_tensor(q_before)
        after = torch.as_tensor(q_after)
        if before.ndim != 3 or before.size(0) == 0 or after.shape != before.shape:
            raise ValueError("q_before/q_after must be aligned non-empty [N,K,D]")
        if before.device != after.device or before.dtype != after.dtype:
            raise ValueError("q_before/q_after must share device and dtype")
        if before.dtype not in (torch.float32, torch.float64):
            raise TypeError("q_before/q_after must use float32 or float64")
        if not bool(torch.isfinite(before).all()) or not bool(torch.isfinite(after).all()):
            raise ValueError("q_before/q_after contain NaN or Inf")
        _validate_canonical_q(before, name="q_before")
        _validate_canonical_q(after, name="q_after")

        self.sample_count = int(before.size(0))
        self.k = int(before.size(1))
        self.d = int(before.size(2))
        self.device = before.device
        self.dtype = before.dtype

        source = before.permute(1, 0, 2).reshape(self.k, self.sample_count * self.d)
        target = after.permute(1, 0, 2).reshape(self.k, self.sample_count * self.d)
        self.transform, self.observed_rank = _closest_identity_procrustes(source, target)

        aligned = apply_transport_to_features(before, self.transform)
        eye = torch.eye(self.k, device=self.device, dtype=self.dtype)
        orthogonality = self.transform.transpose(0, 1) @ self.transform - eye
        self._summary = CoordinateTransportSummary(
            sample_count=self.sample_count,
            spectral_region_count=self.k,
            feature_dimension=self.d,
            observed_rank=int(self.observed_rank),
            raw_feature_change_rms=_rms(after - before),
            aligned_feature_residual_rms=_rms(after - aligned),
            transform_distance_from_identity_rms=_rms(self.transform - eye),
            orthogonality_residual=float(torch.max(torch.abs(orthogonality)).item()),
        )
        self.validate_state()

    @property
    def summary(self) -> CoordinateTransportSummary:
        return self._summary

    def validate_state(self) -> bool:
        if tuple(self.transform.shape) != (self.k, self.k):
            raise RuntimeError("coordinate transform has invalid shape")
        if self.transform.device != self.device or self.transform.dtype != self.dtype:
            raise RuntimeError("coordinate transform has wrong device/dtype")
        if not bool(torch.isfinite(self.transform).all()):
            raise RuntimeError("coordinate transform contains NaN or Inf")
        eye = torch.eye(self.k, device=self.device, dtype=self.dtype)
        tolerance = 5e-5 if self.dtype == torch.float32 else 1e-10
        if not bool(torch.allclose(
            self.transform.transpose(0, 1) @ self.transform,
            eye,
            rtol=tolerance,
            atol=tolerance,
        )):
            raise RuntimeError("coordinate transform is not orthogonal")
        if not 0 <= self.observed_rank <= self.k:
            raise RuntimeError("coordinate transport rank is invalid")
        return True


__all__ = [
    "CoordinateTransportSummary",
    "HSICoordinateTransport",
    "apply_transport_to_features",
]
