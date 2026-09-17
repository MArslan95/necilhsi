from __future__ import annotations

"""Full-band physical spectral structure for HSI CDS.

For a dataset with B native ordered pre-PCA bands this deterministic path returns

    P(x) in R^{B x 3} = [level, spectral slope, spectral curvature].

The raw spectral shape is preserved: derivatives are computed from the original
centre spectrum using actual band/wavelength spacing when supplied, otherwise a
uniform normalized band coordinate.  Only a three-component normalization is
fitted on Phase-0 TRAIN and then frozen.  Thus B is dataset-derived and entirely
independent of the learned token count K.

P is a phase-stable physical structure, not a class-score branch.  This module
owns no class state, replay state, classifier parameter, or trainable parameter.
"""

from numbers import Integral
from typing import Sequence

import torch
import torch.nn as nn

Tensor = torch.Tensor


def _positive_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or int(value) <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _band_positions(
    values: Tensor | Sequence[float] | None,
    *,
    count: int,
) -> Tensor:
    if values is None:
        return torch.linspace(-1.0, 1.0, count, dtype=torch.float32)
    raw = torch.as_tensor(values, dtype=torch.float32).flatten()
    if raw.shape != (count,):
        raise ValueError(f"band_positions must contain {count} values")
    if not bool(torch.isfinite(raw).all()):
        raise ValueError("band_positions contain NaN or Inf")
    if count > 1 and bool((raw[1:] <= raw[:-1]).any()):
        raise ValueError("band_positions must be strictly increasing")
    if count == 1:
        return torch.zeros(1, dtype=torch.float32)
    span = raw[-1] - raw[0]
    if not bool(span > 0):
        raise ValueError("band_positions must span a positive interval")
    return 2.0 * (raw - raw[0]) / span - 1.0


def _first_derivative(values: Tensor, positions: Tensor) -> Tensor:
    if values.ndim != 2 or positions.shape != (values.size(1),):
        raise ValueError("values/positions must be [N,B] and [B]")
    b = int(values.size(1))
    if b < 2:
        return torch.zeros_like(values)
    result = torch.empty_like(values)
    result[:, 0] = (values[:, 1] - values[:, 0]) / (positions[1] - positions[0])
    result[:, -1] = (values[:, -1] - values[:, -2]) / (positions[-1] - positions[-2])
    if b > 2:
        result[:, 1:-1] = (
            (values[:, 2:] - values[:, :-2])
            / (positions[2:] - positions[:-2]).unsqueeze(0)
        )
    return result


def _second_derivative(values: Tensor, positions: Tensor) -> Tensor:
    if values.ndim != 2 or positions.shape != (values.size(1),):
        raise ValueError("values/positions must be [N,B] and [B]")
    b = int(values.size(1))
    result = torch.zeros_like(values)
    if b < 3:
        return result
    step = positions[1:] - positions[:-1]
    slope = (values[:, 1:] - values[:, :-1]) / step.unsqueeze(0)
    interior = (
        2.0
        * (slope[:, 1:] - slope[:, :-1])
        / (positions[2:] - positions[:-2]).unsqueeze(0)
    )
    result[:, 1:-1] = interior
    result[:, 0] = interior[:, 0]
    result[:, -1] = interior[:, -1]
    return result


class OrderedPhysicalSpectralEvidence(nn.Module):
    """Map the complete pre-PCA spectrum to frozen full-band physical evidence."""

    component_names = ("level", "slope", "curvature")

    def __init__(
        self,
        *,
        band_count: int,
        band_positions: Tensor | Sequence[float] | None = None,
    ) -> None:
        super().__init__()
        self.band_count = _positive_int(band_count, "band_count")
        positions = _band_positions(band_positions, count=self.band_count)
        self.register_buffer("band_positions", positions, persistent=True)
        self.register_buffer("descriptor_mean", torch.zeros(3, dtype=torch.float32), persistent=True)
        self.register_buffer("descriptor_scale", torch.ones(3, dtype=torch.float32), persistent=True)
        self.register_buffer("fitted", torch.tensor(False, dtype=torch.bool), persistent=True)

    def _validate_spectra(self, physical_spectrum: Tensor) -> Tensor:
        spectrum = torch.as_tensor(physical_spectrum)
        if spectrum.ndim != 2 or spectrum.size(0) == 0 or spectrum.size(1) != self.band_count:
            raise ValueError(f"physical_spectrum must be non-empty [N,{self.band_count}]")
        if not torch.is_floating_point(spectrum) or spectrum.is_complex():
            raise TypeError("physical_spectrum must be real floating point")
        if not bool(torch.isfinite(spectrum).all()):
            raise ValueError("physical_spectrum contains NaN or Inf")
        return spectrum

    def raw_descriptors(self, physical_spectrum: Tensor) -> Tensor:
        spectrum = self._validate_spectra(physical_spectrum)
        positions = self.band_positions.to(device=spectrum.device, dtype=spectrum.dtype)
        slope = _first_derivative(spectrum, positions)
        curvature = _second_derivative(spectrum, positions)
        result = torch.stack((spectrum, slope, curvature), dim=2)
        if not bool(torch.isfinite(result).all()):
            raise RuntimeError("full-band physical spectral descriptors contain NaN or Inf")
        return result

    @torch.no_grad()
    def fit(self, physical_spectra: Tensor, *, overwrite: bool = False) -> None:
        if bool(self.fitted.item()) and not overwrite:
            raise RuntimeError("physical spectral normalization is already fitted")
        raw = self.raw_descriptors(physical_spectra).to(torch.float32)
        flat = raw.reshape(-1, 3)
        mean = flat.mean(dim=0)
        std = flat.std(dim=0, unbiased=False)
        scale = torch.where(std > 0, std, torch.ones_like(std))
        if not bool(torch.isfinite(mean).all()) or not bool(torch.isfinite(scale).all()):
            raise RuntimeError("physical spectral normalization produced NaN or Inf")
        self.descriptor_mean.copy_(mean.to(self.descriptor_mean))
        self.descriptor_scale.copy_(scale.to(self.descriptor_scale))
        self.fitted.fill_(True)

    def forward(self, physical_spectrum: Tensor) -> Tensor:
        if not bool(self.fitted.item()):
            raise RuntimeError("physical spectral normalization must be fitted first")
        raw = self.raw_descriptors(physical_spectrum)
        mean = self.descriptor_mean.to(device=raw.device, dtype=raw.dtype)
        scale = self.descriptor_scale.to(device=raw.device, dtype=raw.dtype)
        result = (raw - mean.view(1, 1, 3)) / scale.view(1, 1, 3)
        if result.shape[1:] != (self.band_count, 3):
            raise RuntimeError("full-band physical spectral evidence has invalid shape")
        if not bool(torch.isfinite(result).all()):
            raise RuntimeError("normalized full-band physical spectral evidence contains NaN or Inf")
        return result


__all__ = ["OrderedPhysicalSpectralEvidence"]












# from __future__ import annotations

# """Full-band physical spectral structure for HSI CDS.

# For a dataset with B native ordered pre-PCA bands this deterministic path returns

#     P(x) in R^{B x 3} = [level, spectral slope, spectral curvature].

# The raw spectral shape is preserved: derivatives are computed from the original
# centre spectrum using actual band/wavelength spacing when supplied, otherwise a
# uniform normalized band coordinate.  Only a three-component normalization is
# fitted on Phase-0 TRAIN and then frozen.  Thus B is dataset-derived and entirely
# independent of the learned token count K.

# P is a phase-stable physical structure, not a class-score branch.  This module
# owns no class state, replay state, classifier parameter, or trainable parameter.
# """

# from numbers import Integral
# from typing import Sequence

# import torch
# import torch.nn as nn

# Tensor = torch.Tensor


# def _positive_int(value: object, name: str) -> int:
#     if isinstance(value, bool) or not isinstance(value, Integral) or int(value) <= 0:
#         raise ValueError(f"{name} must be a positive integer")
#     return int(value)


# def _band_positions(
#     values: Tensor | Sequence[float] | None,
#     *,
#     count: int,
# ) -> Tensor:
#     if values is None:
#         return torch.linspace(-1.0, 1.0, count, dtype=torch.float32)
#     raw = torch.as_tensor(values, dtype=torch.float32).flatten()
#     if raw.shape != (count,):
#         raise ValueError(f"band_positions must contain {count} values")
#     if not bool(torch.isfinite(raw).all()):
#         raise ValueError("band_positions contain NaN or Inf")
#     if count > 1 and bool((raw[1:] <= raw[:-1]).any()):
#         raise ValueError("band_positions must be strictly increasing")
#     if count == 1:
#         return torch.zeros(1, dtype=torch.float32)
#     span = raw[-1] - raw[0]
#     if not bool(span > 0):
#         raise ValueError("band_positions must span a positive interval")
#     return 2.0 * (raw - raw[0]) / span - 1.0


# def _first_derivative(values: Tensor, positions: Tensor) -> Tensor:
#     if values.ndim != 2 or positions.shape != (values.size(1),):
#         raise ValueError("values/positions must be [N,B] and [B]")
#     b = int(values.size(1))
#     if b < 2:
#         return torch.zeros_like(values)
#     result = torch.empty_like(values)
#     result[:, 0] = (values[:, 1] - values[:, 0]) / (positions[1] - positions[0])
#     result[:, -1] = (values[:, -1] - values[:, -2]) / (positions[-1] - positions[-2])
#     if b > 2:
#         result[:, 1:-1] = (
#             (values[:, 2:] - values[:, :-2])
#             / (positions[2:] - positions[:-2]).unsqueeze(0)
#         )
#     return result


# def _second_derivative(values: Tensor, positions: Tensor) -> Tensor:
#     if values.ndim != 2 or positions.shape != (values.size(1),):
#         raise ValueError("values/positions must be [N,B] and [B]")
#     b = int(values.size(1))
#     result = torch.zeros_like(values)
#     if b < 3:
#         return result
#     step = positions[1:] - positions[:-1]
#     slope = (values[:, 1:] - values[:, :-1]) / step.unsqueeze(0)
#     interior = (
#         2.0
#         * (slope[:, 1:] - slope[:, :-1])
#         / (positions[2:] - positions[:-2]).unsqueeze(0)
#     )
#     result[:, 1:-1] = interior
#     result[:, 0] = interior[:, 0]
#     result[:, -1] = interior[:, -1]
#     return result


# class OrderedPhysicalSpectralEvidence(nn.Module):
#     """Map the complete pre-PCA spectrum to frozen full-band physical evidence."""

#     component_names = ("level", "slope", "curvature")

#     def __init__(
#         self,
#         *,
#         band_count: int,
#         band_positions: Tensor | Sequence[float] | None = None,
#     ) -> None:
#         super().__init__()
#         self.band_count = _positive_int(band_count, "band_count")
#         positions = _band_positions(band_positions, count=self.band_count)
#         self.register_buffer("band_positions", positions, persistent=True)
#         self.register_buffer("descriptor_mean", torch.zeros(3, dtype=torch.float32), persistent=True)
#         self.register_buffer("descriptor_scale", torch.ones(3, dtype=torch.float32), persistent=True)
#         self.register_buffer("fitted", torch.tensor(False, dtype=torch.bool), persistent=True)

#     def _validate_spectra(self, physical_spectrum: Tensor) -> Tensor:
#         spectrum = torch.as_tensor(physical_spectrum)
#         if spectrum.ndim != 2 or spectrum.size(0) == 0 or spectrum.size(1) != self.band_count:
#             raise ValueError(f"physical_spectrum must be non-empty [N,{self.band_count}]")
#         if not torch.is_floating_point(spectrum) or spectrum.is_complex():
#             raise TypeError("physical_spectrum must be real floating point")
#         if not bool(torch.isfinite(spectrum).all()):
#             raise ValueError("physical_spectrum contains NaN or Inf")
#         return spectrum

#     def raw_descriptors(self, physical_spectrum: Tensor) -> Tensor:
#         spectrum = self._validate_spectra(physical_spectrum)
#         positions = self.band_positions.to(device=spectrum.device, dtype=spectrum.dtype)
#         slope = _first_derivative(spectrum, positions)
#         curvature = _second_derivative(spectrum, positions)
#         result = torch.stack((spectrum, slope, curvature), dim=2)
#         if not bool(torch.isfinite(result).all()):
#             raise RuntimeError("full-band physical spectral descriptors contain NaN or Inf")
#         return result

#     @torch.no_grad()
#     def fit(self, physical_spectra: Tensor, *, overwrite: bool = False) -> None:
#         if bool(self.fitted.item()) and not overwrite:
#             raise RuntimeError("physical spectral normalization is already fitted")
#         raw = self.raw_descriptors(physical_spectra).to(torch.float32)
#         flat = raw.reshape(-1, 3)
#         mean = flat.mean(dim=0)
#         std = flat.std(dim=0, unbiased=False)
#         scale = torch.where(std > 0, std, torch.ones_like(std))
#         if not bool(torch.isfinite(mean).all()) or not bool(torch.isfinite(scale).all()):
#             raise RuntimeError("physical spectral normalization produced NaN or Inf")
#         self.descriptor_mean.copy_(mean.to(self.descriptor_mean))
#         self.descriptor_scale.copy_(scale.to(self.descriptor_scale))
#         self.fitted.fill_(True)

#     def forward(self, physical_spectrum: Tensor) -> Tensor:
#         if not bool(self.fitted.item()):
#             raise RuntimeError("physical spectral normalization must be fitted first")
#         raw = self.raw_descriptors(physical_spectrum)
#         mean = self.descriptor_mean.to(device=raw.device, dtype=raw.dtype)
#         scale = self.descriptor_scale.to(device=raw.device, dtype=raw.dtype)
#         result = (raw - mean.view(1, 1, 3)) / scale.view(1, 1, 3)
#         if result.shape[1:] != (self.band_count, 3):
#             raise RuntimeError("full-band physical spectral evidence has invalid shape")
#         if not bool(torch.isfinite(result).all()):
#             raise RuntimeError("normalized full-band physical spectral evidence contains NaN or Inf")
#         return result


# __all__ = ["OrderedPhysicalSpectralEvidence"]










# from __future__ import annotations

# """Full-band physical spectral structure for HSI CDS.

# For a dataset with B native ordered pre-PCA bands this deterministic path returns

#     P(x) in R^{B x 3} = [level, spectral slope, spectral curvature].

# The raw spectral shape is preserved: derivatives are computed from the original
# centre spectrum using actual band/wavelength spacing when supplied, otherwise a
# uniform normalized band coordinate.  Only a three-component normalization is
# fitted on Phase-0 TRAIN and then frozen.  Thus B is dataset-derived and entirely
# independent of the learned token count K.

# P is a phase-stable physical structure, not a class-score branch.  This module
# owns no class state, replay state, classifier parameter, or trainable parameter.
# """

# from numbers import Integral
# from typing import Sequence

# import torch
# import torch.nn as nn

# Tensor = torch.Tensor


# def _positive_int(value: object, name: str) -> int:
#     if isinstance(value, bool) or not isinstance(value, Integral) or int(value) <= 0:
#         raise ValueError(f"{name} must be a positive integer")
#     return int(value)


# def _band_positions(
#     values: Tensor | Sequence[float] | None,
#     *,
#     count: int,
# ) -> Tensor:
#     if values is None:
#         return torch.linspace(-1.0, 1.0, count, dtype=torch.float32)
#     raw = torch.as_tensor(values, dtype=torch.float32).flatten()
#     if raw.shape != (count,):
#         raise ValueError(f"band_positions must contain {count} values")
#     if not bool(torch.isfinite(raw).all()):
#         raise ValueError("band_positions contain NaN or Inf")
#     if count > 1 and bool((raw[1:] <= raw[:-1]).any()):
#         raise ValueError("band_positions must be strictly increasing")
#     if count == 1:
#         return torch.zeros(1, dtype=torch.float32)
#     span = raw[-1] - raw[0]
#     if not bool(span > 0):
#         raise ValueError("band_positions must span a positive interval")
#     return 2.0 * (raw - raw[0]) / span - 1.0


# def _first_derivative(values: Tensor, positions: Tensor) -> Tensor:
#     if values.ndim != 2 or positions.shape != (values.size(1),):
#         raise ValueError("values/positions must be [N,B] and [B]")
#     b = int(values.size(1))
#     if b < 2:
#         return torch.zeros_like(values)
#     result = torch.empty_like(values)
#     result[:, 0] = (values[:, 1] - values[:, 0]) / (positions[1] - positions[0])
#     result[:, -1] = (values[:, -1] - values[:, -2]) / (positions[-1] - positions[-2])
#     if b > 2:
#         result[:, 1:-1] = (
#             (values[:, 2:] - values[:, :-2])
#             / (positions[2:] - positions[:-2]).unsqueeze(0)
#         )
#     return result


# def _second_derivative(values: Tensor, positions: Tensor) -> Tensor:
#     if values.ndim != 2 or positions.shape != (values.size(1),):
#         raise ValueError("values/positions must be [N,B] and [B]")
#     b = int(values.size(1))
#     result = torch.zeros_like(values)
#     if b < 3:
#         return result
#     step = positions[1:] - positions[:-1]
#     slope = (values[:, 1:] - values[:, :-1]) / step.unsqueeze(0)
#     interior = (
#         2.0
#         * (slope[:, 1:] - slope[:, :-1])
#         / (positions[2:] - positions[:-2]).unsqueeze(0)
#     )
#     result[:, 1:-1] = interior
#     result[:, 0] = interior[:, 0]
#     result[:, -1] = interior[:, -1]
#     return result


# class OrderedPhysicalSpectralEvidence(nn.Module):
#     """Map the complete pre-PCA spectrum to frozen full-band physical evidence."""

#     component_names = ("level", "slope", "curvature")

#     def __init__(
#         self,
#         *,
#         band_count: int,
#         band_positions: Tensor | Sequence[float] | None = None,
#     ) -> None:
#         super().__init__()
#         self.band_count = _positive_int(band_count, "band_count")
#         positions = _band_positions(band_positions, count=self.band_count)
#         self.register_buffer("band_positions", positions, persistent=True)
#         self.register_buffer("descriptor_mean", torch.zeros(3, dtype=torch.float32), persistent=True)
#         self.register_buffer("descriptor_scale", torch.ones(3, dtype=torch.float32), persistent=True)
#         self.register_buffer("fitted", torch.tensor(False, dtype=torch.bool), persistent=True)

#     def _validate_spectra(self, physical_spectrum: Tensor) -> Tensor:
#         spectrum = torch.as_tensor(physical_spectrum)
#         if spectrum.ndim != 2 or spectrum.size(0) == 0 or spectrum.size(1) != self.band_count:
#             raise ValueError(f"physical_spectrum must be non-empty [N,{self.band_count}]")
#         if not torch.is_floating_point(spectrum) or spectrum.is_complex():
#             raise TypeError("physical_spectrum must be real floating point")
#         if not bool(torch.isfinite(spectrum).all()):
#             raise ValueError("physical_spectrum contains NaN or Inf")
#         return spectrum

#     def raw_descriptors(self, physical_spectrum: Tensor) -> Tensor:
#         spectrum = self._validate_spectra(physical_spectrum)
#         positions = self.band_positions.to(device=spectrum.device, dtype=spectrum.dtype)
#         slope = _first_derivative(spectrum, positions)
#         curvature = _second_derivative(spectrum, positions)
#         result = torch.stack((spectrum, slope, curvature), dim=2)
#         if not bool(torch.isfinite(result).all()):
#             raise RuntimeError("full-band physical spectral descriptors contain NaN or Inf")
#         return result

#     @torch.no_grad()
#     def fit(self, physical_spectra: Tensor, *, overwrite: bool = False) -> None:
#         if bool(self.fitted.item()) and not overwrite:
#             raise RuntimeError("physical spectral normalization is already fitted")
#         raw = self.raw_descriptors(physical_spectra).to(torch.float32)
#         flat = raw.reshape(-1, 3)
#         mean = flat.mean(dim=0)
#         std = flat.std(dim=0, unbiased=False)
#         scale = torch.where(std > 0, std, torch.ones_like(std))
#         if not bool(torch.isfinite(mean).all()) or not bool(torch.isfinite(scale).all()):
#             raise RuntimeError("physical spectral normalization produced NaN or Inf")
#         self.descriptor_mean.copy_(mean.to(self.descriptor_mean))
#         self.descriptor_scale.copy_(scale.to(self.descriptor_scale))
#         self.fitted.fill_(True)

#     def forward(self, physical_spectrum: Tensor) -> Tensor:
#         if not bool(self.fitted.item()):
#             raise RuntimeError("physical spectral normalization must be fitted first")
#         raw = self.raw_descriptors(physical_spectrum)
#         mean = self.descriptor_mean.to(device=raw.device, dtype=raw.dtype)
#         scale = self.descriptor_scale.to(device=raw.device, dtype=raw.dtype)
#         result = (raw - mean.view(1, 1, 3)) / scale.view(1, 1, 3)
#         if result.shape[1:] != (self.band_count, 3):
#             raise RuntimeError("full-band physical spectral evidence has invalid shape")
#         if not bool(torch.isfinite(result).all()):
#             raise RuntimeError("normalized full-band physical spectral evidence contains NaN or Inf")
#         return result


# __all__ = ["OrderedPhysicalSpectralEvidence"]




# from __future__ import annotations

# """Full-band physical spectral structure for HSI CDS.

# For a dataset with B native ordered pre-PCA bands this deterministic path returns

#     P(x) in R^{B x 3} = [level, spectral slope, spectral curvature].

# The raw spectral shape is preserved: derivatives are computed from the original
# centre spectrum using actual band/wavelength spacing when supplied, otherwise a
# uniform normalized band coordinate.  Only a three-component normalization is
# fitted on Phase-0 TRAIN and then frozen.  Thus B is dataset-derived and entirely
# independent of the learned token count K.

# P is a phase-stable physical structure, not a class-score branch.  This module
# owns no class state, replay state, classifier parameter, or trainable parameter.
# """

# from numbers import Integral
# from typing import Sequence

# import torch
# import torch.nn as nn

# Tensor = torch.Tensor


# def _positive_int(value: object, name: str) -> int:
#     if isinstance(value, bool) or not isinstance(value, Integral) or int(value) <= 0:
#         raise ValueError(f"{name} must be a positive integer")
#     return int(value)


# def _band_positions(
#     values: Tensor | Sequence[float] | None,
#     *,
#     count: int,
# ) -> Tensor:
#     if values is None:
#         return torch.linspace(-1.0, 1.0, count, dtype=torch.float32)
#     raw = torch.as_tensor(values, dtype=torch.float32).flatten()
#     if raw.shape != (count,):
#         raise ValueError(f"band_positions must contain {count} values")
#     if not bool(torch.isfinite(raw).all()):
#         raise ValueError("band_positions contain NaN or Inf")
#     if count > 1 and bool((raw[1:] <= raw[:-1]).any()):
#         raise ValueError("band_positions must be strictly increasing")
#     if count == 1:
#         return torch.zeros(1, dtype=torch.float32)
#     span = raw[-1] - raw[0]
#     if not bool(span > 0):
#         raise ValueError("band_positions must span a positive interval")
#     return 2.0 * (raw - raw[0]) / span - 1.0


# def _first_derivative(values: Tensor, positions: Tensor) -> Tensor:
#     if values.ndim != 2 or positions.shape != (values.size(1),):
#         raise ValueError("values/positions must be [N,B] and [B]")
#     b = int(values.size(1))
#     if b < 2:
#         return torch.zeros_like(values)
#     result = torch.empty_like(values)
#     result[:, 0] = (values[:, 1] - values[:, 0]) / (positions[1] - positions[0])
#     result[:, -1] = (values[:, -1] - values[:, -2]) / (positions[-1] - positions[-2])
#     if b > 2:
#         result[:, 1:-1] = (
#             (values[:, 2:] - values[:, :-2])
#             / (positions[2:] - positions[:-2]).unsqueeze(0)
#         )
#     return result


# def _second_derivative(values: Tensor, positions: Tensor) -> Tensor:
#     if values.ndim != 2 or positions.shape != (values.size(1),):
#         raise ValueError("values/positions must be [N,B] and [B]")
#     b = int(values.size(1))
#     result = torch.zeros_like(values)
#     if b < 3:
#         return result
#     step = positions[1:] - positions[:-1]
#     slope = (values[:, 1:] - values[:, :-1]) / step.unsqueeze(0)
#     interior = (
#         2.0
#         * (slope[:, 1:] - slope[:, :-1])
#         / (positions[2:] - positions[:-2]).unsqueeze(0)
#     )
#     result[:, 1:-1] = interior
#     result[:, 0] = interior[:, 0]
#     result[:, -1] = interior[:, -1]
#     return result


# class OrderedPhysicalSpectralEvidence(nn.Module):
#     """Map the complete pre-PCA spectrum to frozen full-band physical evidence."""

#     component_names = ("level", "slope", "curvature")

#     def __init__(
#         self,
#         *,
#         band_count: int,
#         band_positions: Tensor | Sequence[float] | None = None,
#     ) -> None:
#         super().__init__()
#         self.band_count = _positive_int(band_count, "band_count")
#         positions = _band_positions(band_positions, count=self.band_count)
#         self.register_buffer("band_positions", positions, persistent=True)
#         self.register_buffer("descriptor_mean", torch.zeros(3, dtype=torch.float32), persistent=True)
#         self.register_buffer("descriptor_scale", torch.ones(3, dtype=torch.float32), persistent=True)
#         self.register_buffer("fitted", torch.tensor(False, dtype=torch.bool), persistent=True)

#     def _validate_spectra(self, physical_spectrum: Tensor) -> Tensor:
#         spectrum = torch.as_tensor(physical_spectrum)
#         if spectrum.ndim != 2 or spectrum.size(0) == 0 or spectrum.size(1) != self.band_count:
#             raise ValueError(f"physical_spectrum must be non-empty [N,{self.band_count}]")
#         if not torch.is_floating_point(spectrum) or spectrum.is_complex():
#             raise TypeError("physical_spectrum must be real floating point")
#         if not bool(torch.isfinite(spectrum).all()):
#             raise ValueError("physical_spectrum contains NaN or Inf")
#         return spectrum

#     def raw_descriptors(self, physical_spectrum: Tensor) -> Tensor:
#         spectrum = self._validate_spectra(physical_spectrum)
#         positions = self.band_positions.to(device=spectrum.device, dtype=spectrum.dtype)
#         slope = _first_derivative(spectrum, positions)
#         curvature = _second_derivative(spectrum, positions)
#         result = torch.stack((spectrum, slope, curvature), dim=2)
#         if not bool(torch.isfinite(result).all()):
#             raise RuntimeError("full-band physical spectral descriptors contain NaN or Inf")
#         return result

#     @torch.no_grad()
#     def fit(self, physical_spectra: Tensor, *, overwrite: bool = False) -> None:
#         if bool(self.fitted.item()) and not overwrite:
#             raise RuntimeError("physical spectral normalization is already fitted")
#         raw = self.raw_descriptors(physical_spectra).to(torch.float32)
#         flat = raw.reshape(-1, 3)
#         mean = flat.mean(dim=0)
#         std = flat.std(dim=0, unbiased=False)
#         scale = torch.where(std > 0, std, torch.ones_like(std))
#         if not bool(torch.isfinite(mean).all()) or not bool(torch.isfinite(scale).all()):
#             raise RuntimeError("physical spectral normalization produced NaN or Inf")
#         self.descriptor_mean.copy_(mean.to(self.descriptor_mean))
#         self.descriptor_scale.copy_(scale.to(self.descriptor_scale))
#         self.fitted.fill_(True)

#     def forward(self, physical_spectrum: Tensor) -> Tensor:
#         if not bool(self.fitted.item()):
#             raise RuntimeError("physical spectral normalization must be fitted first")
#         raw = self.raw_descriptors(physical_spectrum)
#         mean = self.descriptor_mean.to(device=raw.device, dtype=raw.dtype)
#         scale = self.descriptor_scale.to(device=raw.device, dtype=raw.dtype)
#         result = (raw - mean.view(1, 1, 3)) / scale.view(1, 1, 3)
#         if result.shape[1:] != (self.band_count, 3):
#             raise RuntimeError("full-band physical spectral evidence has invalid shape")
#         if not bool(torch.isfinite(result).all()):
#             raise RuntimeError("normalized full-band physical spectral evidence contains NaN or Inf")
#         return result


# __all__ = ["OrderedPhysicalSpectralEvidence"]










# from __future__ import annotations

# """Full-band physical spectral structure for HSI CDS.

# For a dataset with B native ordered pre-PCA bands this deterministic path returns

#     P(x) in R^{B x 3} = [level, spectral slope, spectral curvature].

# The raw spectral shape is preserved: derivatives are computed from the original
# centre spectrum using actual band/wavelength spacing when supplied, otherwise a
# uniform normalized band coordinate.  Only a three-component normalization is
# fitted on Phase-0 TRAIN and then frozen.  Thus B is dataset-derived and entirely
# independent of the learned token count K.

# P is a phase-stable physical structure, not a class-score branch.  This module
# owns no class state, replay state, classifier parameter, or trainable parameter.
# """

# from numbers import Integral
# from typing import Sequence

# import torch
# import torch.nn as nn

# Tensor = torch.Tensor


# def _positive_int(value: object, name: str) -> int:
#     if isinstance(value, bool) or not isinstance(value, Integral) or int(value) <= 0:
#         raise ValueError(f"{name} must be a positive integer")
#     return int(value)


# def _band_positions(
#     values: Tensor | Sequence[float] | None,
#     *,
#     count: int,
# ) -> Tensor:
#     if values is None:
#         return torch.linspace(-1.0, 1.0, count, dtype=torch.float32)
#     raw = torch.as_tensor(values, dtype=torch.float32).flatten()
#     if raw.shape != (count,):
#         raise ValueError(f"band_positions must contain {count} values")
#     if not bool(torch.isfinite(raw).all()):
#         raise ValueError("band_positions contain NaN or Inf")
#     if count > 1 and bool((raw[1:] <= raw[:-1]).any()):
#         raise ValueError("band_positions must be strictly increasing")
#     if count == 1:
#         return torch.zeros(1, dtype=torch.float32)
#     span = raw[-1] - raw[0]
#     if not bool(span > 0):
#         raise ValueError("band_positions must span a positive interval")
#     return 2.0 * (raw - raw[0]) / span - 1.0


# def _first_derivative(values: Tensor, positions: Tensor) -> Tensor:
#     if values.ndim != 2 or positions.shape != (values.size(1),):
#         raise ValueError("values/positions must be [N,B] and [B]")
#     b = int(values.size(1))
#     if b < 2:
#         return torch.zeros_like(values)
#     result = torch.empty_like(values)
#     result[:, 0] = (values[:, 1] - values[:, 0]) / (positions[1] - positions[0])
#     result[:, -1] = (values[:, -1] - values[:, -2]) / (positions[-1] - positions[-2])
#     if b > 2:
#         result[:, 1:-1] = (
#             (values[:, 2:] - values[:, :-2])
#             / (positions[2:] - positions[:-2]).unsqueeze(0)
#         )
#     return result


# def _second_derivative(values: Tensor, positions: Tensor) -> Tensor:
#     if values.ndim != 2 or positions.shape != (values.size(1),):
#         raise ValueError("values/positions must be [N,B] and [B]")
#     b = int(values.size(1))
#     result = torch.zeros_like(values)
#     if b < 3:
#         return result
#     step = positions[1:] - positions[:-1]
#     slope = (values[:, 1:] - values[:, :-1]) / step.unsqueeze(0)
#     interior = (
#         2.0
#         * (slope[:, 1:] - slope[:, :-1])
#         / (positions[2:] - positions[:-2]).unsqueeze(0)
#     )
#     result[:, 1:-1] = interior
#     result[:, 0] = interior[:, 0]
#     result[:, -1] = interior[:, -1]
#     return result


# class OrderedPhysicalSpectralEvidence(nn.Module):
#     """Map the complete pre-PCA spectrum to frozen full-band physical evidence."""

#     component_names = ("level", "slope", "curvature")

#     def __init__(
#         self,
#         *,
#         band_count: int,
#         band_positions: Tensor | Sequence[float] | None = None,
#     ) -> None:
#         super().__init__()
#         self.band_count = _positive_int(band_count, "band_count")
#         positions = _band_positions(band_positions, count=self.band_count)
#         self.register_buffer("band_positions", positions, persistent=True)
#         self.register_buffer("descriptor_mean", torch.zeros(3, dtype=torch.float32), persistent=True)
#         self.register_buffer("descriptor_scale", torch.ones(3, dtype=torch.float32), persistent=True)
#         self.register_buffer("fitted", torch.tensor(False, dtype=torch.bool), persistent=True)

#     def _validate_spectra(self, physical_spectrum: Tensor) -> Tensor:
#         spectrum = torch.as_tensor(physical_spectrum)
#         if spectrum.ndim != 2 or spectrum.size(0) == 0 or spectrum.size(1) != self.band_count:
#             raise ValueError(f"physical_spectrum must be non-empty [N,{self.band_count}]")
#         if not torch.is_floating_point(spectrum) or spectrum.is_complex():
#             raise TypeError("physical_spectrum must be real floating point")
#         if not bool(torch.isfinite(spectrum).all()):
#             raise ValueError("physical_spectrum contains NaN or Inf")
#         return spectrum

#     def raw_descriptors(self, physical_spectrum: Tensor) -> Tensor:
#         spectrum = self._validate_spectra(physical_spectrum)
#         positions = self.band_positions.to(device=spectrum.device, dtype=spectrum.dtype)
#         slope = _first_derivative(spectrum, positions)
#         curvature = _second_derivative(spectrum, positions)
#         result = torch.stack((spectrum, slope, curvature), dim=2)
#         if not bool(torch.isfinite(result).all()):
#             raise RuntimeError("full-band physical spectral descriptors contain NaN or Inf")
#         return result

#     @torch.no_grad()
#     def fit(self, physical_spectra: Tensor, *, overwrite: bool = False) -> None:
#         if bool(self.fitted.item()) and not overwrite:
#             raise RuntimeError("physical spectral normalization is already fitted")
#         raw = self.raw_descriptors(physical_spectra).to(torch.float32)
#         flat = raw.reshape(-1, 3)
#         mean = flat.mean(dim=0)
#         std = flat.std(dim=0, unbiased=False)
#         scale = torch.where(std > 0, std, torch.ones_like(std))
#         if not bool(torch.isfinite(mean).all()) or not bool(torch.isfinite(scale).all()):
#             raise RuntimeError("physical spectral normalization produced NaN or Inf")
#         self.descriptor_mean.copy_(mean.to(self.descriptor_mean))
#         self.descriptor_scale.copy_(scale.to(self.descriptor_scale))
#         self.fitted.fill_(True)

#     def forward(self, physical_spectrum: Tensor) -> Tensor:
#         if not bool(self.fitted.item()):
#             raise RuntimeError("physical spectral normalization must be fitted first")
#         raw = self.raw_descriptors(physical_spectrum)
#         mean = self.descriptor_mean.to(device=raw.device, dtype=raw.dtype)
#         scale = self.descriptor_scale.to(device=raw.device, dtype=raw.dtype)
#         result = (raw - mean.view(1, 1, 3)) / scale.view(1, 1, 3)
#         if result.shape[1:] != (self.band_count, 3):
#             raise RuntimeError("full-band physical spectral evidence has invalid shape")
#         if not bool(torch.isfinite(result).all()):
#             raise RuntimeError("normalized full-band physical spectral evidence contains NaN or Inf")
#         return result


# __all__ = ["OrderedPhysicalSpectralEvidence"]


