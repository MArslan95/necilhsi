from __future__ import annotations

from numbers import Integral

import torch
import torch.nn as nn

Tensor = torch.Tensor


def _positive_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result <= 0:
        raise ValueError(f"{name} must be positive")
    return result


def _validate_spectra(values: Tensor, *, band_count: int) -> Tensor:
    spectra = torch.as_tensor(values)
    if spectra.ndim != 2 or spectra.size(0) == 0 or spectra.size(1) != band_count:
        raise ValueError(f"spectra must be non-empty [N,{band_count}]")
    if not torch.is_floating_point(spectra) or spectra.is_complex():
        raise TypeError("spectra must be real floating point")
    if not bool(torch.isfinite(spectra).all()):
        raise ValueError("spectra contain NaN or Inf")
    return spectra


def _linear_quantile(sorted_values: Tensor, quantile: float) -> Tensor:
    values = torch.as_tensor(sorted_values)
    if values.ndim != 2 or values.size(0) == 0:
        raise ValueError("sorted_values must be non-empty [N,B]")
    if not 0.0 <= float(quantile) <= 1.0:
        raise ValueError("quantile must lie in [0,1]")
    count = int(values.size(0))
    if count == 1:
        return values[0]
    position = float(quantile) * float(count - 1)
    lower_index = int(position)
    upper_index = min(lower_index + 1, count - 1)
    interpolation_weight = position - float(lower_index)
    if upper_index == lower_index or interpolation_weight == 0.0:
        return values[lower_index]
    return (
        values[lower_index] * (1.0 - interpolation_weight)
        + values[upper_index] * interpolation_weight
    )


class PhysicalSpectralEvidence(nn.Module):
    """Return fixed robust standardized native-band evidence P[N,B,1]."""

    def __init__(self, *, band_count: int) -> None:
        super().__init__()
        self.band_count = _positive_int(band_count, "band_count")
        self.register_buffer(
            "band_center",
            torch.zeros(self.band_count, dtype=torch.float32),
            persistent=True,
        )
        self.register_buffer(
            "band_scale",
            torch.ones(self.band_count, dtype=torch.float32),
            persistent=True,
        )
        self.register_buffer(
            "fitted",
            torch.tensor(False, dtype=torch.bool),
            persistent=True,
        )

    @torch.no_grad()
    def fit(self, spectra: Tensor, *, overwrite: bool = False) -> None:
        if bool(self.fitted.item()) and not overwrite:
            raise RuntimeError("physical spectral normalization is already fitted")

        values = _validate_spectra(spectra, band_count=self.band_count)
        # MPS cannot perform a float64 cast as part of a device transfer.
        # Move first, then promote on CPU for the robust statistics.
        cpu = values.detach().to(device="cpu").to(dtype=torch.float64)
        ordered = torch.sort(cpu, dim=0, stable=True).values

        center = _linear_quantile(ordered, 0.50)
        absolute_deviation = (cpu - center.unsqueeze(0)).abs()
        ordered_deviation = torch.sort(absolute_deviation, dim=0, stable=True).values
        mad = _linear_quantile(ordered_deviation, 0.50)
        # IQR-free robust phase-zero normalization; no Gaussian fit or old prototypes.
        # When MAD collapses, use empirical mean absolute deviation, then unit scale.
        mean_abs = absolute_deviation.mean(dim=0)
        eps = torch.finfo(torch.float64).eps
        scale = torch.where(
            mad > eps, mad, torch.where(mean_abs > eps, mean_abs, torch.ones_like(mad))
        )

        if not bool(torch.isfinite(center).all()) or not bool(
            torch.isfinite(scale).all()
        ):
            raise RuntimeError("physical spectral normalization produced NaN or Inf")
        if bool((scale <= 0).any()):
            raise RuntimeError(
                "physical spectral normalization produced non-positive scale"
            )

        self.band_center.copy_(center.to(self.band_center))
        self.band_scale.copy_(scale.to(self.band_scale))
        self.fitted.fill_(True)

    def inverse(self, evidence: Tensor) -> Tensor:
        """Map P[N,B,1] back to the spectral coordinate supplied to forward."""
        if not bool(self.fitted.item()):
            raise RuntimeError("physical spectral normalization must be fitted first")
        value = torch.as_tensor(evidence)
        if value.ndim == 2:
            value = value.unsqueeze(-1)
        if value.ndim != 3 or value.size(1) != self.band_count or value.size(2) != 1:
            raise ValueError(f"evidence must be [N,{self.band_count},1]")
        if (
            not torch.is_floating_point(value)
            or value.is_complex()
            or not bool(torch.isfinite(value).all())
        ):
            raise ValueError("evidence must be finite real floating point")
        center = self.band_center.to(device=value.device, dtype=value.dtype)
        scale = self.band_scale.to(device=value.device, dtype=value.dtype)
        spectra = value.squeeze(-1) * scale.unsqueeze(0) + center.unsqueeze(0)
        if not bool(torch.isfinite(spectra).all()):
            raise RuntimeError("inverse physical spectral evidence contains NaN or Inf")
        return spectra

    def forward(self, spectra: Tensor) -> Tensor:
        if not bool(self.fitted.item()):
            raise RuntimeError("physical spectral normalization must be fitted first")

        values = _validate_spectra(spectra, band_count=self.band_count)
        compute_dtype = (
            torch.float32
            if values.dtype in (torch.float16, torch.bfloat16)
            else values.dtype
        )
        center = self.band_center.to(device=values.device, dtype=compute_dtype)
        scale = self.band_scale.to(device=values.device, dtype=compute_dtype)
        normalized = (values.to(dtype=compute_dtype) - center) / scale
        normalized = normalized.to(dtype=values.dtype).unsqueeze(-1)

        if normalized.shape[1:] != (self.band_count, 1):
            raise RuntimeError("physical spectral evidence must have shape [N,B,1]")
        if not bool(torch.isfinite(normalized).all()):
            raise RuntimeError("physical spectral evidence contains NaN or Inf")
        return normalized


__all__ = ["PhysicalSpectralEvidence"]

# from __future__ import annotations

# from numbers import Integral

# import torch
# import torch.nn as nn

# Tensor = torch.Tensor


# def _positive_int(value: object, name: str) -> int:
#     if isinstance(value, bool) or not isinstance(value, Integral):
#         raise TypeError(f"{name} must be an integer")
#     result = int(value)
#     if result <= 0:
#         raise ValueError(f"{name} must be positive")
#     return result


# def _validate_spectra(values: Tensor, *, band_count: int) -> Tensor:
#     spectra = torch.as_tensor(values)
#     if spectra.ndim != 2 or spectra.size(0) == 0 or spectra.size(1) != band_count:
#         raise ValueError(f"spectra must be non-empty [N,{band_count}]")
#     if not torch.is_floating_point(spectra) or spectra.is_complex():
#         raise TypeError("spectra must be real floating point")
#     if not bool(torch.isfinite(spectra).all()):
#         raise ValueError("spectra contain NaN or Inf")
#     return spectra


# def _linear_quantile(sorted_values: Tensor, quantile: float) -> Tensor:
#     values = torch.as_tensor(sorted_values)
#     if values.ndim != 2 or values.size(0) == 0:
#         raise ValueError("sorted_values must be non-empty [N,B]")
#     if not 0.0 <= float(quantile) <= 1.0:
#         raise ValueError("quantile must lie in [0,1]")
#     count = int(values.size(0))
#     if count == 1:
#         return values[0]
#     position = float(quantile) * float(count - 1)
#     lower_index = int(position)
#     upper_index = min(lower_index + 1, count - 1)
#     interpolation_weight = position - float(lower_index)
#     if upper_index == lower_index or interpolation_weight == 0.0:
#         return values[lower_index]
#     return (
#         values[lower_index] * (1.0 - interpolation_weight)
#         + values[upper_index] * interpolation_weight
#     )


# class PhysicalSpectralEvidence(nn.Module):
#     """Return fixed robust standardized native-band evidence P[N,B,1]."""

#     def __init__(self, *, band_count: int) -> None:
#         super().__init__()
#         self.band_count = _positive_int(band_count, "band_count")
#         self.register_buffer(
#             "band_center",
#             torch.zeros(self.band_count, dtype=torch.float32),
#             persistent=True,
#         )
#         self.register_buffer(
#             "band_scale",
#             torch.ones(self.band_count, dtype=torch.float32),
#             persistent=True,
#         )
#         self.register_buffer(
#             "fitted",
#             torch.tensor(False, dtype=torch.bool),
#             persistent=True,
#         )

#     @torch.no_grad()
#     def fit(self, spectra: Tensor, *, overwrite: bool = False) -> None:
#         if bool(self.fitted.item()) and not overwrite:
#             raise RuntimeError("physical spectral normalization is already fitted")

#         values = _validate_spectra(spectra, band_count=self.band_count)
#         # MPS cannot perform a float64 cast as part of a device transfer.
#         # Move first, then promote on CPU for the robust statistics.
#         cpu = values.detach().to(device="cpu").to(dtype=torch.float64)
#         ordered = torch.sort(cpu, dim=0, stable=True).values

#         center = _linear_quantile(ordered, 0.50)
#         absolute_deviation = (cpu - center.unsqueeze(0)).abs()
#         ordered_deviation = torch.sort(absolute_deviation, dim=0, stable=True).values
#         mad = _linear_quantile(ordered_deviation, 0.50)
#         # IQR-free robust phase-zero normalization; no Gaussian fit or old prototypes.
#         # When MAD collapses, use empirical mean absolute deviation, then unit scale.
#         mean_abs = absolute_deviation.mean(dim=0)
#         eps = torch.finfo(torch.float64).eps
#         scale = torch.where(
#             mad > eps, mad, torch.where(mean_abs > eps, mean_abs, torch.ones_like(mad))
#         )

#         if not bool(torch.isfinite(center).all()) or not bool(
#             torch.isfinite(scale).all()
#         ):
#             raise RuntimeError("physical spectral normalization produced NaN or Inf")
#         if bool((scale <= 0).any()):
#             raise RuntimeError(
#                 "physical spectral normalization produced non-positive scale"
#             )

#         self.band_center.copy_(center.to(self.band_center))
#         self.band_scale.copy_(scale.to(self.band_scale))
#         self.fitted.fill_(True)

#     def inverse(self, evidence: Tensor) -> Tensor:
#         """Map P[N,B,1] back to the spectral coordinate supplied to forward."""
#         if not bool(self.fitted.item()):
#             raise RuntimeError("physical spectral normalization must be fitted first")
#         value = torch.as_tensor(evidence)
#         if value.ndim == 2:
#             value = value.unsqueeze(-1)
#         if value.ndim != 3 or value.size(1) != self.band_count or value.size(2) != 1:
#             raise ValueError(f"evidence must be [N,{self.band_count},1]")
#         if (
#             not torch.is_floating_point(value)
#             or value.is_complex()
#             or not bool(torch.isfinite(value).all())
#         ):
#             raise ValueError("evidence must be finite real floating point")
#         center = self.band_center.to(device=value.device, dtype=value.dtype)
#         scale = self.band_scale.to(device=value.device, dtype=value.dtype)
#         spectra = value.squeeze(-1) * scale.unsqueeze(0) + center.unsqueeze(0)
#         if not bool(torch.isfinite(spectra).all()):
#             raise RuntimeError("inverse physical spectral evidence contains NaN or Inf")
#         return spectra

#     def forward(self, spectra: Tensor) -> Tensor:
#         if not bool(self.fitted.item()):
#             raise RuntimeError("physical spectral normalization must be fitted first")

#         values = _validate_spectra(spectra, band_count=self.band_count)
#         compute_dtype = (
#             torch.float32
#             if values.dtype in (torch.float16, torch.bfloat16)
#             else values.dtype
#         )
#         center = self.band_center.to(device=values.device, dtype=compute_dtype)
#         scale = self.band_scale.to(device=values.device, dtype=compute_dtype)
#         normalized = (values.to(dtype=compute_dtype) - center) / scale
#         normalized = normalized.to(dtype=values.dtype).unsqueeze(-1)

#         if normalized.shape[1:] != (self.band_count, 1):
#             raise RuntimeError("physical spectral evidence must have shape [N,B,1]")
#         if not bool(torch.isfinite(normalized).all()):
#             raise RuntimeError("physical spectral evidence contains NaN or Inf")
#         return normalized


# __all__ = ["PhysicalSpectralEvidence"]
