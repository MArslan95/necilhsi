from __future__ import annotations

"""HSI Mamba feature extractor for the single SCOPE-HSI representation.

Representation contract
-----------------------
The backbone has one job: map the complete HSI sample to one ordered
spectral-spatial feature tensor

    Q_raw(x) in R^{K x D}.

1. Spectral path
   The raw centre spectrum is standardized with phase-0 TRAIN statistics,
   encoded in physical band order by shared bidirectional Mamba, and only then
   reduced to K contiguous ordered spectral regions.

2. Spatial path
   The processed/PCA patch contributes neighbourhood context only.  It is made
   centre-relative, encoded with shared Mamba over locality-preserving scans,
   and pooled with the encoded full-spectrum summary.

3. Fusion
   Spatial context is fused into every ordered spectral-region feature while
   retaining the K-row order.  No second learned sample representation is
   created.

The backbone output itself is the only downstream sample representation Q.
No model-level global RMS canonicalization is applied.  Cosine normalization is
used only inside relation/separation computations where it is mathematically
required.  Auxiliary tensors returned with ``return_aux=True`` are diagnostics
of the computation that produced Q; they are never alternate classifier or
continual-state representations.

This module does not form class structure, classifier state, replay state, or
incremental-learning state.
"""

from dataclasses import dataclass
import math
from numbers import Integral
from typing import Any, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

Tensor = torch.Tensor


# -----------------------------------------------------------------------------
# Validation and fixed coordinates
# -----------------------------------------------------------------------------


def _positive_int(value: object, name: str) -> int:
    """Require an actual positive integer; do not silently coerce floats."""
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
    if result <= 0:
        raise ValueError(f"{name} must be positive")
    return result


def _spectral_positions(
    band_count: int,
    coordinates: Sequence[float],
) -> Tensor:
    """Validate physical band coordinates and normalize them to [-1,1]."""
    values = torch.as_tensor(
        coordinates,
        dtype=torch.float32,
    ).flatten()

    if values.numel() != band_count:
        raise ValueError(
            "spectral coordinates have "
            f"{values.numel()} values; expected {band_count}"
        )
    if not bool(torch.isfinite(values).all()):
        raise ValueError("spectral coordinates contain NaN or Inf")

    if band_count == 1:
        return torch.zeros(
            1,
            dtype=torch.float32,
        )

    if bool((values[1:] <= values[:-1]).any()):
        raise ValueError("spectral coordinates must be strictly increasing")

    span = values[-1] - values[0]
    if not bool(torch.isfinite(span)) or not bool(span > 0):
        raise ValueError("spectral coordinates must have a finite positive span")

    return 2.0 * (values - values[0]) / span - 1.0


def _spectral_region_bounds(
    band_count: int,
    region_count: int,
) -> tuple[tuple[int, int], ...]:
    """Partition all ordered bands into K contiguous non-overlapping regions."""
    band_count = _positive_int(
        band_count,
        "spectral_bands",
    )
    region_count = _positive_int(
        region_count,
        "spectral_token_count",
    )

    if region_count < 2:
        raise ValueError(
            "spectral_token_count must be at least 2 because "
            "ordered spectral structure requires more than one spectral region"
        )

    if region_count > band_count:
        raise ValueError(
            "spectral_token_count cannot exceed the number of raw spectral bands"
        )

    bounds: list[tuple[int, int]] = []
    for index in range(region_count):
        start = index * band_count // region_count
        end = (index + 1) * band_count // region_count
        if end <= start:
            raise RuntimeError("spectral-region partition produced an empty region")
        bounds.append((start, end))

    if bounds[0][0] != 0 or bounds[-1][1] != band_count:
        raise RuntimeError("spectral-region partition does not cover all bands")

    for left, right in zip(
        bounds[:-1],
        bounds[1:],
    ):
        if left[1] != right[0]:
            raise RuntimeError("spectral regions are not contiguous")

    return tuple(bounds)


def _spatial_positions(
    patch_size: int,
) -> Tensor:
    axis = torch.linspace(
        -1.0,
        1.0,
        patch_size,
        dtype=torch.float32,
    )
    yy, xx = torch.meshgrid(
        axis,
        axis,
        indexing="ij",
    )
    return torch.stack(
        (xx, yy),
        dim=0,
    ).unsqueeze(0)


def _serpentine_orders(
    patch_size: int,
) -> Tensor:
    """Horizontal/vertical forward/reverse locality-preserving scan orders."""
    grid = torch.arange(
        patch_size * patch_size,
        dtype=torch.long,
    ).reshape(
        patch_size,
        patch_size,
    )

    horizontal = torch.cat(
        [
            (grid[row] if row % 2 == 0 else grid[row].flip(0))
            for row in range(patch_size)
        ],
        dim=0,
    )

    vertical = torch.cat(
        [
            (grid[:, column] if column % 2 == 0 else grid[:, column].flip(0))
            for column in range(patch_size)
        ],
        dim=0,
    )

    return torch.stack(
        (
            horizontal,
            horizontal.flip(0),
            vertical,
            vertical.flip(0),
        ),
        dim=0,
    )


# -----------------------------------------------------------------------------
# Fixed phase-0 raw-spectrum standardization
# -----------------------------------------------------------------------------


class FrozenBandStandardizer(nn.Module):
    """Per-band raw-spectrum standardization fixed after phase-0 fitting."""

    def __init__(
        self,
        band_count: int,
    ) -> None:
        super().__init__()

        self.band_count = _positive_int(
            band_count,
            "spectral_bands",
        )

        self.register_buffer(
            "mean",
            torch.zeros(
                self.band_count,
                dtype=torch.float32,
            ),
        )
        self.register_buffer(
            "scale",
            torch.ones(
                self.band_count,
                dtype=torch.float32,
            ),
        )
        self.register_buffer(
            "fitted",
            torch.tensor(
                False,
                dtype=torch.bool,
            ),
        )

    @torch.no_grad()
    def set_statistics(
        self,
        mean: Tensor | Sequence[float],
        scale: Tensor | Sequence[float],
        *,
        overwrite: bool = False,
    ) -> None:
        if bool(self.fitted.item()) and not overwrite:
            raise RuntimeError("spectral standardization is already fitted")

        mean_value = torch.as_tensor(
            mean,
            device=self.mean.device,
            dtype=self.mean.dtype,
        ).flatten()

        scale_value = torch.as_tensor(
            scale,
            device=self.scale.device,
            dtype=self.scale.dtype,
        ).flatten()

        expected = (self.band_count,)
        if mean_value.shape != expected or scale_value.shape != expected:
            raise ValueError(
                "spectral standardization statistics must each contain "
                f"{self.band_count} values"
            )

        if not bool(torch.isfinite(mean_value).all()) or not bool(
            torch.isfinite(scale_value).all()
        ):
            raise ValueError("spectral standardization statistics contain NaN or Inf")

        if bool((scale_value <= 0).any()):
            raise ValueError("spectral standardization scales must be positive")

        self.mean.copy_(mean_value)
        self.scale.copy_(scale_value)
        self.fitted.fill_(True)

    @torch.no_grad()
    def fit(
        self,
        spectra: Tensor,
        *,
        overwrite: bool = False,
    ) -> None:
        values = torch.as_tensor(spectra)

        if values.ndim != 2 or values.size(0) == 0 or values.size(1) != self.band_count:
            raise ValueError("base spectra must have shape " f"[N,{self.band_count}]")

        if not torch.is_floating_point(values) or values.is_complex():
            raise TypeError("base spectra must be real floating point")

        if not bool(torch.isfinite(values).all()):
            raise ValueError("base spectra contain NaN or Inf")

        values = values.to(dtype=torch.float32)

        mean = values.mean(dim=0)
        standard_deviation = values.std(
            dim=0,
            unbiased=False,
        )

        # A constant band has zero empirical variance.  Mapping its scale to one
        # keeps the standardized value finite and exactly zero after centering.
        scale = torch.where(
            standard_deviation > 0,
            standard_deviation,
            torch.ones_like(standard_deviation),
        )

        self.set_statistics(
            mean,
            scale,
            overwrite=overwrite,
        )

    def forward(
        self,
        spectrum: Tensor,
    ) -> Tensor:
        if not bool(self.fitted.item()):
            raise RuntimeError(
                "spectral standardization must be fitted or loaded "
                "before encoding spectra"
            )

        if spectrum.ndim != 2 or spectrum.size(1) != self.band_count:
            raise ValueError("spectrum must have shape " f"[N,{self.band_count}]")

        compute_dtype = (
            torch.float32
            if spectrum.dtype
            in (
                torch.float16,
                torch.bfloat16,
            )
            else spectrum.dtype
        )

        mean = self.mean.to(
            device=spectrum.device,
            dtype=compute_dtype,
        )
        scale = self.scale.to(
            device=spectrum.device,
            dtype=compute_dtype,
        )

        result = (spectrum.to(dtype=compute_dtype) - mean) / scale

        result = result.to(dtype=spectrum.dtype)

        if not bool(torch.isfinite(result).all()):
            raise RuntimeError("spectral standardization produced NaN or Inf")

        return result


# -----------------------------------------------------------------------------
# Selective state-space encoder
# -----------------------------------------------------------------------------


class SelectiveStateSpace(nn.Module):
    """Self-contained selective state-space sequence mixer."""

    def __init__(
        self,
        d_model: int,
        *,
        d_state: int,
        d_conv: int,
        expand: int,
        dt_min: float = 1e-3,
        dt_max: float = 1e-1,
    ) -> None:
        super().__init__()

        self.d_model = _positive_int(
            d_model,
            "d_model",
        )
        self.d_state = _positive_int(
            d_state,
            "d_state",
        )
        self.d_conv = _positive_int(
            d_conv,
            "d_conv",
        )
        self.expand = _positive_int(
            expand,
            "expand",
        )

        if (
            not math.isfinite(dt_min)
            or not math.isfinite(dt_max)
            or not 0.0 < dt_min < dt_max
        ):
            raise ValueError("require finite 0 < dt_min < dt_max")

        self.d_inner = self.d_model * self.expand
        self.dt_rank = max(
            1,
            math.ceil(self.d_model / 16),
        )

        self.in_projection = nn.Linear(
            self.d_model,
            2 * self.d_inner,
            bias=False,
        )

        self.depthwise_convolution = nn.Conv1d(
            self.d_inner,
            self.d_inner,
            kernel_size=self.d_conv,
            groups=self.d_inner,
            padding=self.d_conv - 1,
        )

        self.state_projection = nn.Linear(
            self.d_inner,
            self.dt_rank + 2 * self.d_state,
            bias=False,
        )

        self.delta_projection = nn.Linear(
            self.dt_rank,
            self.d_inner,
        )

        state_index = torch.arange(
            1,
            self.d_state + 1,
            dtype=torch.float32,
        )

        self.A_log = nn.Parameter(
            state_index.log().repeat(
                self.d_inner,
                1,
            )
        )

        self.D = nn.Parameter(torch.ones(self.d_inner))

        self.out_projection = nn.Linear(
            self.d_inner,
            self.d_model,
            bias=False,
        )

        delta_scale = self.dt_rank**-0.5
        nn.init.uniform_(
            self.delta_projection.weight,
            -delta_scale,
            delta_scale,
        )

        initial_delta = torch.exp(
            torch.empty(self.d_inner).uniform_(
                math.log(dt_min),
                math.log(dt_max),
            )
        )

        inverse_softplus = initial_delta + torch.log(-torch.expm1(-initial_delta))

        with torch.no_grad():
            self.delta_projection.bias.copy_(inverse_softplus)

    def _scan(
        self,
        x: Tensor,
        delta: Tensor,
        b: Tensor,
        c: Tensor,
    ) -> Tensor:
        batch_size, length, _ = x.shape

        scan_dtype = (
            torch.float32
            if x.dtype
            in (
                torch.float16,
                torch.bfloat16,
            )
            else x.dtype
        )

        state = torch.zeros(
            batch_size,
            self.d_inner,
            self.d_state,
            device=x.device,
            dtype=scan_dtype,
        )

        transition_rate = -self.A_log.to(dtype=scan_dtype).exp()

        x_scan = x.to(dtype=scan_dtype)
        delta_scan = delta.to(dtype=scan_dtype)
        b_scan = b.to(dtype=scan_dtype)
        c_scan = c.to(dtype=scan_dtype)
        skip = self.D.to(dtype=scan_dtype)

        outputs: list[Tensor] = []

        for index in range(length):
            step = delta_scan[
                :,
                index,
            ]

            transition = torch.exp(step.unsqueeze(-1) * transition_rate.unsqueeze(0))

            input_state = (
                step.unsqueeze(-1)
                * b_scan[
                    :,
                    index,
                ].unsqueeze(1)
                * x_scan[
                    :,
                    index,
                ].unsqueeze(-1)
            )

            state = transition * state + input_state

            output = (
                state
                * c_scan[
                    :,
                    index,
                ].unsqueeze(1)
            ).sum(dim=-1)

            output = (
                output
                + skip
                * x_scan[
                    :,
                    index,
                ]
            )

            outputs.append(output)

        return torch.stack(
            outputs,
            dim=1,
        ).to(dtype=x.dtype)

    def forward(
        self,
        tokens: Tensor,
    ) -> Tensor:
        if (
            tokens.ndim != 3
            or tokens.size(0) == 0
            or tokens.size(1) == 0
            or tokens.size(2) != self.d_model
        ):
            raise ValueError("tokens must have shape " f"[N,L,{self.d_model}]")

        if not torch.is_floating_point(tokens) or tokens.is_complex():
            raise TypeError("tokens must be real floating point")

        if not bool(torch.isfinite(tokens).all()):
            raise ValueError("tokens contain NaN or Inf")

        length = tokens.size(1)

        x, gate = self.in_projection(tokens).chunk(
            2,
            dim=-1,
        )

        x = self.depthwise_convolution(
            x.transpose(
                1,
                2,
            )
        )[
            ...,
            :length,
        ].transpose(
            1,
            2,
        )

        x = F.silu(x)

        projected = self.state_projection(x)

        delta_low_rank, b, c = torch.split(
            projected,
            (
                self.dt_rank,
                self.d_state,
                self.d_state,
            ),
            dim=-1,
        )

        delta = F.softplus(self.delta_projection(delta_low_rank))

        mixed = self._scan(
            x,
            delta,
            b,
            c,
        ) * F.silu(gate)

        output = self.out_projection(mixed)

        if not bool(torch.isfinite(output).all()):
            raise RuntimeError("selective state-space mixer produced NaN or Inf")

        return output


class MambaBlock(nn.Module):
    """Pre-normalized residual selective-state-space block."""

    def __init__(
        self,
        width: int,
        *,
        d_state: int,
        d_conv: int,
        expand: int,
        dropout: float,
    ) -> None:
        super().__init__()

        self.normalization = nn.LayerNorm(width)
        self.mixer = SelectiveStateSpace(
            width,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        tokens: Tensor,
    ) -> Tensor:
        return tokens + self.dropout(self.mixer(self.normalization(tokens)))


# -----------------------------------------------------------------------------
# Ordered spectral encoder
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class _SpectralEncoderOutput:
    region_features: Tensor
    summary: Tensor
    band_features: Tensor


class OrderedSpectralEncoder(nn.Module):
    """Encode the complete ordered raw spectrum before K-region reduction."""

    def __init__(
        self,
        band_count: int,
        feature_dim: int,
        region_count: int,
        *,
        layers: int,
        d_state: int,
        d_conv: int,
        expand: int,
        dropout: float,
        spectral_positions: Sequence[float] | None,
    ) -> None:
        super().__init__()

        self.band_count = _positive_int(
            band_count,
            "spectral_bands",
        )
        self.feature_dim = _positive_int(
            feature_dim,
            "d_model",
        )
        self.region_count = _positive_int(
            region_count,
            "spectral_token_count",
        )

        self.region_bounds = _spectral_region_bounds(
            self.band_count,
            self.region_count,
        )

        self.use_physical_positions = spectral_positions is not None

        if self.use_physical_positions:
            positions = _spectral_positions(
                self.band_count,
                spectral_positions,
            )
        else:
            positions = torch.empty(
                0,
                dtype=torch.float32,
            )

        self.register_buffer(
            "band_positions",
            positions,
        )

        self.standardizer = FrozenBandStandardizer(self.band_count)

        embedding_dim = 2 if self.use_physical_positions else 1

        # No bias: a shared class-independent offset is unnecessary before the
        # ordered spectral class structure is formed downstream.
        self.band_embedding = nn.Linear(
            embedding_dim,
            self.feature_dim,
            bias=False,
        )

        layer_count = _positive_int(
            layers,
            "num_spectral_layers",
        )

        self.blocks = nn.ModuleList(
            [
                MambaBlock(
                    self.feature_dim,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand,
                    dropout=dropout,
                )
                for _ in range(layer_count)
            ]
        )

    def _encode(
        self,
        band_features: Tensor,
    ) -> Tensor:
        output = band_features
        for block in self.blocks:
            output = block(output)
        return output

    def _reduce_regions(
        self,
        band_features: Tensor,
    ) -> Tensor:
        if band_features.shape[1:] != (
            self.band_count,
            self.feature_dim,
        ):
            raise RuntimeError("encoded band-feature shape is invalid")

        region_features = torch.stack(
            [
                band_features[
                    :,
                    start:end,
                ].mean(dim=1)
                for start, end in self.region_bounds
            ],
            dim=1,
        )

        expected = (
            band_features.size(0),
            self.region_count,
            self.feature_dim,
        )

        if region_features.shape != expected:
            raise RuntimeError(
                "spectral-region features have shape "
                f"{tuple(region_features.shape)}, expected {expected}"
            )

        return region_features

    def forward(
        self,
        center_spectrum: Tensor,
    ) -> _SpectralEncoderOutput:
        if (
            center_spectrum.ndim != 2
            or center_spectrum.size(0) == 0
            or center_spectrum.size(1) != self.band_count
        ):
            raise ValueError(
                "center_spectrum must have shape " f"[N,{self.band_count}]"
            )

        if not torch.is_floating_point(center_spectrum) or center_spectrum.is_complex():
            raise TypeError("center_spectrum must be real floating point")

        if not bool(torch.isfinite(center_spectrum).all()):
            raise ValueError("center_spectrum contains NaN or Inf")

        normalized = self.standardizer(center_spectrum)

        if self.use_physical_positions:
            positions = (
                self.band_positions.to(
                    device=normalized.device,
                    dtype=normalized.dtype,
                )
                .unsqueeze(0)
                .expand(
                    normalized.size(0),
                    -1,
                )
            )

            embedding_input = torch.stack(
                (
                    normalized,
                    positions,
                ),
                dim=-1,
            )
        else:
            embedding_input = normalized.unsqueeze(-1)

        band_features = self.band_embedding(embedding_input)

        # HSI spectral classification is non-causal.  One shared encoder is
        # applied in both wavelength directions and both outputs are restored to
        # the same physical band order before averaging.
        paired = torch.cat(
            (
                band_features,
                band_features.flip(1),
            ),
            dim=0,
        )

        encoded = self._encode(paired)

        forward_features, reverse_features = encoded.chunk(
            2,
            dim=0,
        )

        band_features = (forward_features + reverse_features.flip(1)) * 0.5

        region_features = self._reduce_regions(band_features)

        # Use all encoded bands for the spectral query used by the spatial
        # context pool; this avoids region-size weighting when B % K != 0.
        summary = band_features.mean(dim=1)

        if (
            not bool(torch.isfinite(region_features).all())
            or not bool(torch.isfinite(summary).all())
            or not bool(torch.isfinite(band_features).all())
        ):
            raise RuntimeError("ordered spectral encoder produced NaN or Inf")

        return _SpectralEncoderOutput(
            region_features=region_features,
            summary=summary,
            band_features=band_features,
        )


# -----------------------------------------------------------------------------
# Spatial context encoder
# -----------------------------------------------------------------------------


class SpectralConditionedSpatialPool(nn.Module):
    """Pool spatial tokens using the encoded full-spectrum summary as query."""

    def __init__(
        self,
        feature_dim: int,
    ) -> None:
        super().__init__()

        self.feature_dim = _positive_int(
            feature_dim,
            "d_model",
        )

        self.spatial_normalization = nn.LayerNorm(self.feature_dim)
        self.spectral_normalization = nn.LayerNorm(self.feature_dim)

        self.query_projection = nn.Linear(
            self.feature_dim,
            self.feature_dim,
            bias=False,
        )
        self.key_projection = nn.Linear(
            self.feature_dim,
            self.feature_dim,
            bias=False,
        )

    def forward(
        self,
        spatial_features: Tensor,
        spectral_summary: Tensor,
    ) -> tuple[Tensor, Tensor]:
        if (
            spatial_features.ndim != 3
            or spatial_features.size(0) == 0
            or spatial_features.size(1) == 0
            or spatial_features.size(2) != self.feature_dim
        ):
            raise ValueError(
                "spatial_features must have shape " f"[N,L,{self.feature_dim}]"
            )

        if spectral_summary.shape != (
            spatial_features.size(0),
            self.feature_dim,
        ):
            raise ValueError(
                "spectral_summary must have shape " f"[N,{self.feature_dim}]"
            )

        if spatial_features.device != spectral_summary.device:
            raise ValueError(
                "spatial features and spectral summary must share a device"
            )

        if spatial_features.dtype != spectral_summary.dtype:
            raise ValueError("spatial features and spectral summary must share a dtype")

        if not bool(torch.isfinite(spatial_features).all()) or not bool(
            torch.isfinite(spectral_summary).all()
        ):
            raise ValueError("spatial-pooling inputs contain NaN or Inf")

        query = self.query_projection(
            self.spectral_normalization(spectral_summary)
        ).unsqueeze(1)

        keys = self.key_projection(self.spatial_normalization(spatial_features))

        scores = (query * keys).sum(dim=-1) / math.sqrt(self.feature_dim)

        weights = F.softmax(
            scores,
            dim=1,
        )

        context = (spatial_features * weights.unsqueeze(-1)).sum(dim=1)

        if not bool(torch.isfinite(context).all()) or not bool(
            torch.isfinite(weights).all()
        ):
            raise RuntimeError("spectral-conditioned spatial pool produced NaN or Inf")

        return (
            context,
            weights,
        )


@dataclass(frozen=True)
class _SpatialEncoderOutput:
    context: Tensor
    spatial_features: Tensor
    pooling_weights: Tensor
    relative_patch: Tensor


class CenterRelativeSpatialEncoder(nn.Module):
    """Encode neighborhood differences around the classified center pixel."""

    def __init__(
        self,
        input_channels: int,
        patch_size: int,
        feature_dim: int,
        *,
        layers: int,
        d_state: int,
        d_conv: int,
        expand: int,
        dropout: float,
        norm_groups: int,
    ) -> None:
        super().__init__()

        self.input_channels = _positive_int(
            input_channels,
            "num_bands",
        )
        self.patch_size = _positive_int(
            patch_size,
            "patch_size",
        )
        self.feature_dim = _positive_int(
            feature_dim,
            "d_model",
        )
        self.spatial_token_count = self.patch_size**2

        group_count = _positive_int(
            norm_groups,
            "stem_norm_groups",
        )

        if group_count > self.feature_dim or self.feature_dim % group_count != 0:
            raise ValueError(
                "stem_norm_groups must divide d_model " f"({self.feature_dim})"
            )

        self.register_buffer(
            "spatial_positions",
            _spatial_positions(self.patch_size),
            persistent=False,
        )

        scan_orders = _serpentine_orders(self.patch_size)

        self.register_buffer(
            "scan_orders",
            scan_orders,
            persistent=False,
        )
        self.register_buffer(
            "inverse_scan_orders",
            scan_orders.argsort(dim=1),
            persistent=False,
        )

        self.stem = nn.Sequential(
            nn.Conv2d(
                self.input_channels + 2,
                self.feature_dim,
                kernel_size=1,
                bias=False,
            ),
            nn.GroupNorm(
                group_count,
                self.feature_dim,
            ),
            nn.GELU(),
            nn.Conv2d(
                self.feature_dim,
                self.feature_dim,
                kernel_size=3,
                padding=1,
                groups=self.feature_dim,
                bias=False,
            ),
            nn.GELU(),
            nn.Conv2d(
                self.feature_dim,
                self.feature_dim,
                kernel_size=1,
                bias=False,
            ),
        )

        layer_count = _positive_int(
            layers,
            "num_spatial_layers",
        )

        self.blocks = nn.ModuleList(
            [
                MambaBlock(
                    self.feature_dim,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand,
                    dropout=dropout,
                )
                for _ in range(layer_count)
            ]
        )

        self.pool = SpectralConditionedSpatialPool(self.feature_dim)

    def _encode(
        self,
        spatial_features: Tensor,
    ) -> Tensor:
        output = spatial_features
        for block in self.blocks:
            output = block(output)
        return output

    def forward(
        self,
        patch: Tensor,
        spectral_summary: Tensor,
    ) -> _SpatialEncoderOutput:
        expected_patch = (
            self.input_channels,
            self.patch_size,
            self.patch_size,
        )

        if (
            patch.ndim != 4
            or patch.size(0) == 0
            or tuple(patch.shape[1:]) != expected_patch
        ):
            raise ValueError(
                "patch must have shape "
                f"[N,{self.input_channels},"
                f"{self.patch_size},{self.patch_size}]"
            )

        if spectral_summary.shape != (
            patch.size(0),
            self.feature_dim,
        ):
            raise ValueError(
                "spectral_summary must have shape " f"[N,{self.feature_dim}]"
            )

        if not torch.is_floating_point(patch) or patch.is_complex():
            raise TypeError("patch must be real floating point")

        if (
            not torch.is_floating_point(spectral_summary)
            or spectral_summary.is_complex()
        ):
            raise TypeError("spectral_summary must be real floating point")

        if patch.device != spectral_summary.device:
            raise ValueError("patch and spectral_summary must share a device")

        if patch.dtype != spectral_summary.dtype:
            raise ValueError("patch and spectral_summary must share a dtype")

        if not bool(torch.isfinite(patch).all()) or not bool(
            torch.isfinite(spectral_summary).all()
        ):
            raise ValueError("spatial encoder inputs contain NaN or Inf")

        center_index = self.patch_size // 2

        center = patch[
            :,
            :,
            center_index,
            center_index,
        ]

        # The raw center spectrum carries absolute spectral identity.  This
        # spatial branch therefore models neighborhood differences rather than
        # duplicating that center spectrum in a second representation.
        relative_patch = (
            patch
            - center[
                :,
                :,
                None,
                None,
            ]
        )

        positions = self.spatial_positions.to(
            device=patch.device,
            dtype=patch.dtype,
        ).expand(
            patch.size(0),
            -1,
            -1,
            -1,
        )

        spatial_features = self.stem(
            torch.cat(
                (
                    relative_patch,
                    positions,
                ),
                dim=1,
            )
        )

        spatial_features = spatial_features.flatten(2).transpose(
            1,
            2,
        )

        batch_size = spatial_features.size(0)

        scanned = torch.cat(
            [
                spatial_features.index_select(
                    1,
                    order,
                )
                for order in self.scan_orders
            ],
            dim=0,
        )

        encoded = self._encode(scanned).reshape(
            4,
            batch_size,
            self.spatial_token_count,
            self.feature_dim,
        )

        restored = torch.stack(
            [
                encoded[scan_index].index_select(
                    1,
                    self.inverse_scan_orders[scan_index],
                )
                for scan_index in range(4)
            ],
            dim=0,
        )

        # Each spatial location is represented by the mean response over the
        # four shared-weight scan directions after restoring the same grid order.
        spatial_features = restored.mean(dim=0)

        context, pooling_weights = self.pool(
            spatial_features,
            spectral_summary,
        )

        if not bool(torch.isfinite(spatial_features).all()):
            raise RuntimeError("spatial encoder produced NaN or Inf")

        return _SpatialEncoderOutput(
            context=context,
            spatial_features=spatial_features,
            pooling_weights=pooling_weights,
            relative_patch=relative_patch,
        )


# -----------------------------------------------------------------------------
# Direct spectral-spatial fusion
# -----------------------------------------------------------------------------


class SpectralSpatialFusion(nn.Module):
    """Fuse context into Q without destroying spectral-region order.

    Let T_k be the encoded feature for ordered spectral region k and c the
    pooled spatial context.  The fusion is

        h_k = LN(T_k) * W_c LN(c)
        q_k = T_k + W_o SiLU(h_k).

    The element-wise interaction makes the spatial correction token-specific:
    the same context does not simply add one identical vector to every spectral
    region.  The residual keeps T_k as the spectral anchor, while W_o allows the
    context to change both magnitude and feature direction.

    No softmax competition across spectral regions and no K-dependent scaling
    are used because the downstream model operates directly on the ordered Q.
    """

    def __init__(
        self,
        feature_dim: int,
    ) -> None:
        super().__init__()

        self.feature_dim = _positive_int(
            feature_dim,
            "d_model",
        )

        self.spectral_normalization = nn.LayerNorm(self.feature_dim)
        self.context_normalization = nn.LayerNorm(self.feature_dim)

        self.context_projection = nn.Linear(
            self.feature_dim,
            self.feature_dim,
            bias=False,
        )

        self.output_projection = nn.Linear(
            self.feature_dim,
            self.feature_dim,
            bias=False,
        )

    def forward(
        self,
        spectral_region_features: Tensor,
        spatial_context: Tensor,
    ) -> Tensor:
        if (
            spectral_region_features.ndim != 3
            or spectral_region_features.size(0) == 0
            or spectral_region_features.size(1) < 2
            or spectral_region_features.size(2) != self.feature_dim
        ):
            raise ValueError(
                "spectral_region_features must have shape "
                f"[N,K,{self.feature_dim}] with K>=2"
            )

        if spatial_context.shape != (
            spectral_region_features.size(0),
            self.feature_dim,
        ):
            raise ValueError(
                "spatial_context must have shape " f"[N,{self.feature_dim}]"
            )

        if spectral_region_features.device != spatial_context.device:
            raise ValueError("spectral and spatial features must share a device")

        if spectral_region_features.dtype != spatial_context.dtype:
            raise ValueError("spectral and spatial features must share a dtype")

        if not bool(torch.isfinite(spectral_region_features).all()) or not bool(
            torch.isfinite(spatial_context).all()
        ):
            raise ValueError("spectral-spatial fusion inputs contain NaN or Inf")

        normalized_spectral = self.spectral_normalization(spectral_region_features)

        aligned_context = self.context_projection(
            self.context_normalization(spatial_context)
        ).unsqueeze(1)

        interaction = normalized_spectral * aligned_context

        correction = self.output_projection(F.silu(interaction))

        ordered_features = spectral_region_features + correction

        if not bool(torch.isfinite(ordered_features).all()):
            raise RuntimeError("spectral-spatial fusion produced NaN or Inf")

        return ordered_features


# -----------------------------------------------------------------------------
# Public backbone
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class BackboneOutput:
    """Output of the HSI feature extractor.

    ``ordered_features`` is the only mandatory backbone output:

        Q_raw(x) in R^{N x K x D}.

    Its second axis follows spectral-region order and this tensor is the deployed
    Q used by the prediction head and class-structure bank.  Physical spectral
    evidence P is produced separately from the raw centre spectrum and is not a
    second learned representation.

    All other fields are optional diagnostics returned only when
    ``return_aux=True``; they are not alternate downstream representations.
    """

    ordered_features: Tensor

    spectral_region_features: Tensor | None = None
    encoded_band_features: Tensor | None = None
    spectral_summary: Tensor | None = None

    spatial_context: Tensor | None = None
    spatial_features: Tensor | None = None
    spatial_pool_weights: Tensor | None = None
    relative_patch: Tensor | None = None


class HSIMambaFeatureExtractor(nn.Module):
    """Extract one ordered spectral-spatial feature tensor Q(x) per HSI sample.

    Continual-learning contract
    --------------------------
    Base and incremental learning keep one learned representation Q[K,D].  During
    incremental learning the learned extractor remains plastic; stability is supplied
    by old-logit KD, Whole-CDS preservation, CDS-based old/new separation, and
    semantic-drift correction rather than by freezing arbitrary learned entry layers.
    The phase-0 raw-spectrum standardizer remains fixed because it defines the
    physical input coordinate.  A previous backbone copy is trainer-local/read-only.
    """

    def __init__(
        self,
        args: Any,
    ) -> None:
        super().__init__()

        self.patch_bands = _positive_int(
            args.num_bands,
            "num_bands",
        )

        self.spectral_bands = _positive_int(
            getattr(
                args,
                "spectral_bands",
                getattr(
                    args,
                    "raw_num_bands",
                    args.num_bands,
                ),
            ),
            "spectral_bands",
        )

        self.patch_size = _positive_int(
            args.patch_size,
            "patch_size",
        )

        if self.patch_size % 2 == 0:
            raise ValueError("patch_size must be odd so one center pixel is defined")

        if not hasattr(
            args,
            "spectral_token_count",
        ):
            raise ValueError("spectral_token_count must be provided explicitly")

        # Public backbone/structure contract.
        self.spectral_token_count = _positive_int(
            args.spectral_token_count,
            "spectral_token_count",
        )
        self.spectral_token_dim = _positive_int(
            args.d_model,
            "d_model",
        )

        if self.spectral_token_count < 2:
            raise ValueError(
                "spectral_token_count must be at least 2 because "
                "ordered spectral structure models cross-region interactions"
            )

        if self.spectral_token_count > self.spectral_bands:
            raise ValueError("spectral_token_count cannot exceed spectral_bands")

        d_state = _positive_int(
            getattr(
                args,
                "d_state",
                16,
            ),
            "d_state",
        )
        d_conv = _positive_int(
            getattr(
                args,
                "d_conv",
                4,
            ),
            "d_conv",
        )
        expand = _positive_int(
            getattr(
                args,
                "expand",
                2,
            ),
            "expand",
        )

        dropout = float(
            getattr(
                args,
                "mamba_dropout",
                0.0,
            )
        )

        if not math.isfinite(dropout) or not 0.0 <= dropout < 1.0:
            raise ValueError("mamba_dropout must be finite and lie in [0,1)")

        spectral_positions = getattr(
            args,
            "band_wavelengths",
            None,
        )

        if spectral_positions is None:
            spectral_positions = getattr(
                args,
                "band_positions",
                None,
            )

        self.spectral_encoder = OrderedSpectralEncoder(
            self.spectral_bands,
            self.spectral_token_dim,
            self.spectral_token_count,
            layers=getattr(
                args,
                "num_spectral_layers",
                2,
            ),
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            dropout=dropout,
            spectral_positions=spectral_positions,
        )

        normalization_mean = getattr(
            args,
            "spectral_normalization_mean",
            None,
        )
        normalization_scale = getattr(
            args,
            "spectral_normalization_std",
            None,
        )

        if (normalization_mean is None) != (normalization_scale is None):
            raise ValueError(
                "spectral_normalization_mean and "
                "spectral_normalization_std must be supplied together"
            )

        if normalization_mean is not None:
            self.spectral_encoder.standardizer.set_statistics(
                normalization_mean,
                normalization_scale,
            )

        self.spatial_encoder = CenterRelativeSpatialEncoder(
            self.patch_bands,
            self.patch_size,
            self.spectral_token_dim,
            layers=getattr(
                args,
                "num_spatial_layers",
                2,
            ),
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            dropout=dropout,
            norm_groups=getattr(
                args,
                "stem_norm_groups",
                1,
            ),
        )

        self.fusion = SpectralSpatialFusion(self.spectral_token_dim)

        # Explicit phase trainability state. The trainer should call the public
        # methods below rather than toggling arbitrary submodules.

    @staticmethod
    def _set_module_trainable(module: nn.Module, enabled: bool) -> None:
        for parameter in module.parameters():
            parameter.requires_grad_(bool(enabled))
            if not bool(enabled):
                parameter.grad = None
        if not bool(enabled):
            module.eval()

    def set_base_trainability(self) -> None:
        """Base phase: learn the complete spectral-spatial representation."""
        self._set_module_trainable(self.spectral_encoder, True)
        self._set_module_trainable(self.spatial_encoder, True)
        self._set_module_trainable(self.fusion, True)

    def set_incremental_trainability(self) -> None:
        """Incremental phase: keep the full learned representation plastic.

        The phase-0 standardizer is buffer state, not a trainable module.  Learned
        entry maps, Mamba/SSM blocks, spatial context and fusion all remain adaptive;
        continual stability is enforced by the explicit NECIL objectives.
        """
        if not self.spectral_normalization_fitted:
            raise RuntimeError(
                "spectral normalization must be fitted before incremental trainability"
            )
        self._set_module_trainable(self.spectral_encoder, True)
        self._set_module_trainable(self.spatial_encoder, True)
        self._set_module_trainable(self.fusion, True)

    def incremental_trainable_parameters(self) -> tuple[nn.Parameter, ...]:
        parameters = tuple(
            parameter for parameter in self.parameters() if parameter.requires_grad
        )
        if not parameters:
            raise RuntimeError("incremental backbone has no trainable parameters")
        return parameters

    @property
    def spectral_normalization_fitted(
        self,
    ) -> bool:
        return bool(self.spectral_encoder.standardizer.fitted.item())

    @torch.no_grad()
    def set_spectral_normalization(
        self,
        mean: Tensor | Sequence[float],
        scale: Tensor | Sequence[float],
        *,
        overwrite: bool = False,
    ) -> None:
        self.spectral_encoder.standardizer.set_statistics(
            mean,
            scale,
            overwrite=overwrite,
        )

    @torch.no_grad()
    def fit_spectral_normalization(
        self,
        base_center_spectra: Tensor,
        *,
        overwrite: bool = False,
    ) -> None:
        self.spectral_encoder.standardizer.fit(
            base_center_spectra,
            overwrite=overwrite,
        )

    def forward(
        self,
        patch: Tensor,
        *,
        center_spectrum: Tensor,
        return_aux: bool = False,
    ) -> BackboneOutput:
        if (
            not torch.is_tensor(patch)
            or not torch.is_floating_point(patch)
            or patch.is_complex()
        ):
            raise TypeError("patch must be a real floating-point tensor")

        if (
            not torch.is_tensor(center_spectrum)
            or not torch.is_floating_point(center_spectrum)
            or center_spectrum.is_complex()
        ):
            raise TypeError("center_spectrum must be a real floating-point tensor")

        expected_patch = (
            self.patch_bands,
            self.patch_size,
            self.patch_size,
        )

        if (
            patch.ndim != 4
            or patch.size(0) == 0
            or tuple(patch.shape[1:]) != expected_patch
        ):
            raise ValueError(
                "patch must have shape "
                f"[N,{self.patch_bands},"
                f"{self.patch_size},{self.patch_size}]"
            )

        if center_spectrum.shape != (
            patch.size(0),
            self.spectral_bands,
        ):
            raise ValueError(
                "center_spectrum must have shape " f"[N,{self.spectral_bands}]"
            )

        if patch.device != center_spectrum.device:
            raise ValueError("patch and center_spectrum must share a device")

        if patch.dtype != center_spectrum.dtype:
            raise ValueError("patch and center_spectrum must share a dtype")

        if not bool(torch.isfinite(patch).all()) or not bool(
            torch.isfinite(center_spectrum).all()
        ):
            raise ValueError("backbone inputs contain NaN or Inf")

        spectral = self.spectral_encoder(center_spectrum)

        spatial = self.spatial_encoder(
            patch,
            spectral.summary,
        )

        ordered_features = self.fusion(
            spectral.region_features,
            spatial.context,
        )

        expected_output = (
            patch.size(0),
            self.spectral_token_count,
            self.spectral_token_dim,
        )

        if ordered_features.shape != expected_output:
            raise RuntimeError(
                "ordered_features have shape "
                f"{tuple(ordered_features.shape)}, "
                f"expected {expected_output}"
            )

        if not bool(torch.isfinite(ordered_features).all()):
            raise RuntimeError("backbone produced NaN or Inf")

        if not return_aux:
            return BackboneOutput(
                ordered_features=ordered_features,
            )

        return BackboneOutput(
            ordered_features=ordered_features,
            spectral_region_features=(spectral.region_features),
            encoded_band_features=(spectral.band_features),
            spectral_summary=(spectral.summary),
            spatial_context=(spatial.context),
            spatial_features=(spatial.spatial_features),
            spatial_pool_weights=(spatial.pooling_weights),
            relative_patch=(spatial.relative_patch),
        )


__all__ = [
    "BackboneOutput",
    "HSIMambaFeatureExtractor",
]


# from __future__ import annotations

# """HSI Mamba feature extractor for the single SCOPE-HSI representation.

# Representation contract
# -----------------------
# The backbone has one job: map the complete HSI sample to one ordered
# spectral-spatial feature tensor

#     Q_raw(x) in R^{K x D}.

# 1. Spectral path
#    The raw centre spectrum is standardized with phase-0 TRAIN statistics,
#    encoded in physical band order by shared bidirectional Mamba, and only then
#    reduced to K contiguous ordered spectral regions.

# 2. Spatial path
#    The processed/PCA patch contributes neighbourhood context only.  It is made
#    centre-relative, encoded with shared Mamba over locality-preserving scans,
#    and pooled with the encoded full-spectrum summary.

# 3. Fusion
#    Spatial context is fused into every ordered spectral-region feature while
#    retaining the K-row order.  No second learned sample representation is
#    created.

# The backbone output itself is the only downstream sample representation Q.
# No model-level global RMS canonicalization is applied.  Cosine normalization is
# used only inside relation/separation computations where it is mathematically
# required.  Auxiliary tensors returned with ``return_aux=True`` are diagnostics
# of the computation that produced Q; they are never alternate classifier or
# continual-state representations.

# This module does not form class structure, classifier state, replay state, or
# incremental-learning state.
# """

# from dataclasses import dataclass
# import math
# from numbers import Integral
# from typing import Any, Sequence

# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# Tensor = torch.Tensor


# # -----------------------------------------------------------------------------
# # Validation and fixed coordinates
# # -----------------------------------------------------------------------------


# def _positive_int(value: object, name: str) -> int:
#     """Require an actual positive integer; do not silently coerce floats."""
#     if torch.is_tensor(value):
#         if value.ndim != 0:
#             raise TypeError(f"{name} must be an integer scalar")
#         if (
#             value.dtype == torch.bool
#             or value.is_complex()
#             or torch.is_floating_point(value)
#         ):
#             raise TypeError(f"{name} must be an integer scalar")
#         value = value.item()

#     if isinstance(value, bool) or not isinstance(value, Integral):
#         raise TypeError(f"{name} must be an integer")

#     result = int(value)
#     if result <= 0:
#         raise ValueError(f"{name} must be positive")
#     return result


# def _spectral_positions(
#     band_count: int,
#     coordinates: Sequence[float],
# ) -> Tensor:
#     """Validate physical band coordinates and normalize them to [-1,1]."""
#     values = torch.as_tensor(
#         coordinates,
#         dtype=torch.float32,
#     ).flatten()

#     if values.numel() != band_count:
#         raise ValueError(
#             "spectral coordinates have "
#             f"{values.numel()} values; expected {band_count}"
#         )
#     if not bool(torch.isfinite(values).all()):
#         raise ValueError("spectral coordinates contain NaN or Inf")

#     if band_count == 1:
#         return torch.zeros(
#             1,
#             dtype=torch.float32,
#         )

#     if bool((values[1:] <= values[:-1]).any()):
#         raise ValueError("spectral coordinates must be strictly increasing")

#     span = values[-1] - values[0]
#     if not bool(torch.isfinite(span)) or not bool(span > 0):
#         raise ValueError("spectral coordinates must have a finite positive span")

#     return 2.0 * (values - values[0]) / span - 1.0


# def _spectral_region_bounds(
#     band_count: int,
#     region_count: int,
# ) -> tuple[tuple[int, int], ...]:
#     """Partition all ordered bands into K contiguous non-overlapping regions."""
#     band_count = _positive_int(
#         band_count,
#         "spectral_bands",
#     )
#     region_count = _positive_int(
#         region_count,
#         "spectral_token_count",
#     )

#     if region_count < 2:
#         raise ValueError(
#             "spectral_token_count must be at least 2 because "
#             "ordered spectral structure requires more than one spectral region"
#         )

#     if region_count > band_count:
#         raise ValueError(
#             "spectral_token_count cannot exceed the number of raw spectral bands"
#         )

#     bounds: list[tuple[int, int]] = []
#     for index in range(region_count):
#         start = index * band_count // region_count
#         end = (index + 1) * band_count // region_count
#         if end <= start:
#             raise RuntimeError("spectral-region partition produced an empty region")
#         bounds.append((start, end))

#     if bounds[0][0] != 0 or bounds[-1][1] != band_count:
#         raise RuntimeError("spectral-region partition does not cover all bands")

#     for left, right in zip(
#         bounds[:-1],
#         bounds[1:],
#     ):
#         if left[1] != right[0]:
#             raise RuntimeError("spectral regions are not contiguous")

#     return tuple(bounds)


# def _spatial_positions(
#     patch_size: int,
# ) -> Tensor:
#     axis = torch.linspace(
#         -1.0,
#         1.0,
#         patch_size,
#         dtype=torch.float32,
#     )
#     yy, xx = torch.meshgrid(
#         axis,
#         axis,
#         indexing="ij",
#     )
#     return torch.stack(
#         (xx, yy),
#         dim=0,
#     ).unsqueeze(0)


# def _serpentine_orders(
#     patch_size: int,
# ) -> Tensor:
#     """Horizontal/vertical forward/reverse locality-preserving scan orders."""
#     grid = torch.arange(
#         patch_size * patch_size,
#         dtype=torch.long,
#     ).reshape(
#         patch_size,
#         patch_size,
#     )

#     horizontal = torch.cat(
#         [
#             (grid[row] if row % 2 == 0 else grid[row].flip(0))
#             for row in range(patch_size)
#         ],
#         dim=0,
#     )

#     vertical = torch.cat(
#         [
#             (grid[:, column] if column % 2 == 0 else grid[:, column].flip(0))
#             for column in range(patch_size)
#         ],
#         dim=0,
#     )

#     return torch.stack(
#         (
#             horizontal,
#             horizontal.flip(0),
#             vertical,
#             vertical.flip(0),
#         ),
#         dim=0,
#     )


# # -----------------------------------------------------------------------------
# # Fixed phase-0 raw-spectrum standardization
# # -----------------------------------------------------------------------------


# class FrozenBandStandardizer(nn.Module):
#     """Per-band raw-spectrum standardization fixed after phase-0 fitting."""

#     def __init__(
#         self,
#         band_count: int,
#     ) -> None:
#         super().__init__()

#         self.band_count = _positive_int(
#             band_count,
#             "spectral_bands",
#         )

#         self.register_buffer(
#             "mean",
#             torch.zeros(
#                 self.band_count,
#                 dtype=torch.float32,
#             ),
#         )
#         self.register_buffer(
#             "scale",
#             torch.ones(
#                 self.band_count,
#                 dtype=torch.float32,
#             ),
#         )
#         self.register_buffer(
#             "fitted",
#             torch.tensor(
#                 False,
#                 dtype=torch.bool,
#             ),
#         )

#     @torch.no_grad()
#     def set_statistics(
#         self,
#         mean: Tensor | Sequence[float],
#         scale: Tensor | Sequence[float],
#         *,
#         overwrite: bool = False,
#     ) -> None:
#         if bool(self.fitted.item()) and not overwrite:
#             raise RuntimeError("spectral standardization is already fitted")

#         mean_value = torch.as_tensor(
#             mean,
#             device=self.mean.device,
#             dtype=self.mean.dtype,
#         ).flatten()

#         scale_value = torch.as_tensor(
#             scale,
#             device=self.scale.device,
#             dtype=self.scale.dtype,
#         ).flatten()

#         expected = (self.band_count,)
#         if mean_value.shape != expected or scale_value.shape != expected:
#             raise ValueError(
#                 "spectral standardization statistics must each contain "
#                 f"{self.band_count} values"
#             )

#         if not bool(torch.isfinite(mean_value).all()) or not bool(
#             torch.isfinite(scale_value).all()
#         ):
#             raise ValueError("spectral standardization statistics contain NaN or Inf")

#         if bool((scale_value <= 0).any()):
#             raise ValueError("spectral standardization scales must be positive")

#         self.mean.copy_(mean_value)
#         self.scale.copy_(scale_value)
#         self.fitted.fill_(True)

#     @torch.no_grad()
#     def fit(
#         self,
#         spectra: Tensor,
#         *,
#         overwrite: bool = False,
#     ) -> None:
#         values = torch.as_tensor(spectra)

#         if values.ndim != 2 or values.size(0) == 0 or values.size(1) != self.band_count:
#             raise ValueError("base spectra must have shape " f"[N,{self.band_count}]")

#         if not torch.is_floating_point(values) or values.is_complex():
#             raise TypeError("base spectra must be real floating point")

#         if not bool(torch.isfinite(values).all()):
#             raise ValueError("base spectra contain NaN or Inf")

#         values = values.to(dtype=torch.float32)

#         mean = values.mean(dim=0)
#         standard_deviation = values.std(
#             dim=0,
#             unbiased=False,
#         )

#         # A constant band has zero empirical variance.  Mapping its scale to one
#         # keeps the standardized value finite and exactly zero after centering.
#         scale = torch.where(
#             standard_deviation > 0,
#             standard_deviation,
#             torch.ones_like(standard_deviation),
#         )

#         self.set_statistics(
#             mean,
#             scale,
#             overwrite=overwrite,
#         )

#     def forward(
#         self,
#         spectrum: Tensor,
#     ) -> Tensor:
#         if not bool(self.fitted.item()):
#             raise RuntimeError(
#                 "spectral standardization must be fitted or loaded "
#                 "before encoding spectra"
#             )

#         if spectrum.ndim != 2 or spectrum.size(1) != self.band_count:
#             raise ValueError("spectrum must have shape " f"[N,{self.band_count}]")

#         compute_dtype = (
#             torch.float32
#             if spectrum.dtype
#             in (
#                 torch.float16,
#                 torch.bfloat16,
#             )
#             else spectrum.dtype
#         )

#         mean = self.mean.to(
#             device=spectrum.device,
#             dtype=compute_dtype,
#         )
#         scale = self.scale.to(
#             device=spectrum.device,
#             dtype=compute_dtype,
#         )

#         result = (spectrum.to(dtype=compute_dtype) - mean) / scale

#         result = result.to(dtype=spectrum.dtype)

#         if not bool(torch.isfinite(result).all()):
#             raise RuntimeError("spectral standardization produced NaN or Inf")

#         return result


# # -----------------------------------------------------------------------------
# # Selective state-space encoder
# # -----------------------------------------------------------------------------


# class SelectiveStateSpace(nn.Module):
#     """Self-contained selective state-space sequence mixer."""

#     def __init__(
#         self,
#         d_model: int,
#         *,
#         d_state: int,
#         d_conv: int,
#         expand: int,
#         dt_min: float = 1e-3,
#         dt_max: float = 1e-1,
#     ) -> None:
#         super().__init__()

#         self.d_model = _positive_int(
#             d_model,
#             "d_model",
#         )
#         self.d_state = _positive_int(
#             d_state,
#             "d_state",
#         )
#         self.d_conv = _positive_int(
#             d_conv,
#             "d_conv",
#         )
#         self.expand = _positive_int(
#             expand,
#             "expand",
#         )

#         if (
#             not math.isfinite(dt_min)
#             or not math.isfinite(dt_max)
#             or not 0.0 < dt_min < dt_max
#         ):
#             raise ValueError("require finite 0 < dt_min < dt_max")

#         self.d_inner = self.d_model * self.expand
#         self.dt_rank = max(
#             1,
#             math.ceil(self.d_model / 16),
#         )

#         self.in_projection = nn.Linear(
#             self.d_model,
#             2 * self.d_inner,
#             bias=False,
#         )

#         self.depthwise_convolution = nn.Conv1d(
#             self.d_inner,
#             self.d_inner,
#             kernel_size=self.d_conv,
#             groups=self.d_inner,
#             padding=self.d_conv - 1,
#         )

#         self.state_projection = nn.Linear(
#             self.d_inner,
#             self.dt_rank + 2 * self.d_state,
#             bias=False,
#         )

#         self.delta_projection = nn.Linear(
#             self.dt_rank,
#             self.d_inner,
#         )

#         state_index = torch.arange(
#             1,
#             self.d_state + 1,
#             dtype=torch.float32,
#         )

#         self.A_log = nn.Parameter(
#             state_index.log().repeat(
#                 self.d_inner,
#                 1,
#             )
#         )

#         self.D = nn.Parameter(torch.ones(self.d_inner))

#         self.out_projection = nn.Linear(
#             self.d_inner,
#             self.d_model,
#             bias=False,
#         )

#         delta_scale = self.dt_rank**-0.5
#         nn.init.uniform_(
#             self.delta_projection.weight,
#             -delta_scale,
#             delta_scale,
#         )

#         initial_delta = torch.exp(
#             torch.empty(self.d_inner).uniform_(
#                 math.log(dt_min),
#                 math.log(dt_max),
#             )
#         )

#         inverse_softplus = initial_delta + torch.log(-torch.expm1(-initial_delta))

#         with torch.no_grad():
#             self.delta_projection.bias.copy_(inverse_softplus)

#     def _scan(
#         self,
#         x: Tensor,
#         delta: Tensor,
#         b: Tensor,
#         c: Tensor,
#     ) -> Tensor:
#         batch_size, length, _ = x.shape

#         scan_dtype = (
#             torch.float32
#             if x.dtype
#             in (
#                 torch.float16,
#                 torch.bfloat16,
#             )
#             else x.dtype
#         )

#         state = torch.zeros(
#             batch_size,
#             self.d_inner,
#             self.d_state,
#             device=x.device,
#             dtype=scan_dtype,
#         )

#         transition_rate = -self.A_log.to(dtype=scan_dtype).exp()

#         x_scan = x.to(dtype=scan_dtype)
#         delta_scan = delta.to(dtype=scan_dtype)
#         b_scan = b.to(dtype=scan_dtype)
#         c_scan = c.to(dtype=scan_dtype)
#         skip = self.D.to(dtype=scan_dtype)

#         outputs: list[Tensor] = []

#         for index in range(length):
#             step = delta_scan[
#                 :,
#                 index,
#             ]

#             transition = torch.exp(step.unsqueeze(-1) * transition_rate.unsqueeze(0))

#             input_state = (
#                 step.unsqueeze(-1)
#                 * b_scan[
#                     :,
#                     index,
#                 ].unsqueeze(1)
#                 * x_scan[
#                     :,
#                     index,
#                 ].unsqueeze(-1)
#             )

#             state = transition * state + input_state

#             output = (
#                 state
#                 * c_scan[
#                     :,
#                     index,
#                 ].unsqueeze(1)
#             ).sum(dim=-1)

#             output = (
#                 output
#                 + skip
#                 * x_scan[
#                     :,
#                     index,
#                 ]
#             )

#             outputs.append(output)

#         return torch.stack(
#             outputs,
#             dim=1,
#         ).to(dtype=x.dtype)

#     def forward(
#         self,
#         tokens: Tensor,
#     ) -> Tensor:
#         if (
#             tokens.ndim != 3
#             or tokens.size(0) == 0
#             or tokens.size(1) == 0
#             or tokens.size(2) != self.d_model
#         ):
#             raise ValueError("tokens must have shape " f"[N,L,{self.d_model}]")

#         if not torch.is_floating_point(tokens) or tokens.is_complex():
#             raise TypeError("tokens must be real floating point")

#         if not bool(torch.isfinite(tokens).all()):
#             raise ValueError("tokens contain NaN or Inf")

#         length = tokens.size(1)

#         x, gate = self.in_projection(tokens).chunk(
#             2,
#             dim=-1,
#         )

#         x = self.depthwise_convolution(
#             x.transpose(
#                 1,
#                 2,
#             )
#         )[
#             ...,
#             :length,
#         ].transpose(
#             1,
#             2,
#         )

#         x = F.silu(x)

#         projected = self.state_projection(x)

#         delta_low_rank, b, c = torch.split(
#             projected,
#             (
#                 self.dt_rank,
#                 self.d_state,
#                 self.d_state,
#             ),
#             dim=-1,
#         )

#         delta = F.softplus(self.delta_projection(delta_low_rank))

#         mixed = self._scan(
#             x,
#             delta,
#             b,
#             c,
#         ) * F.silu(gate)

#         output = self.out_projection(mixed)

#         if not bool(torch.isfinite(output).all()):
#             raise RuntimeError("selective state-space mixer produced NaN or Inf")

#         return output


# class MambaBlock(nn.Module):
#     """Pre-normalized residual selective-state-space block."""

#     def __init__(
#         self,
#         width: int,
#         *,
#         d_state: int,
#         d_conv: int,
#         expand: int,
#         dropout: float,
#     ) -> None:
#         super().__init__()

#         self.normalization = nn.LayerNorm(width)
#         self.mixer = SelectiveStateSpace(
#             width,
#             d_state=d_state,
#             d_conv=d_conv,
#             expand=expand,
#         )
#         self.dropout = nn.Dropout(dropout)

#     def forward(
#         self,
#         tokens: Tensor,
#     ) -> Tensor:
#         return tokens + self.dropout(self.mixer(self.normalization(tokens)))


# # -----------------------------------------------------------------------------
# # Ordered spectral encoder
# # -----------------------------------------------------------------------------


# @dataclass(frozen=True)
# class _SpectralEncoderOutput:
#     region_features: Tensor
#     summary: Tensor
#     band_features: Tensor


# class OrderedSpectralEncoder(nn.Module):
#     """Encode the complete ordered raw spectrum before K-region reduction."""

#     def __init__(
#         self,
#         band_count: int,
#         feature_dim: int,
#         region_count: int,
#         *,
#         layers: int,
#         d_state: int,
#         d_conv: int,
#         expand: int,
#         dropout: float,
#         spectral_positions: Sequence[float] | None,
#     ) -> None:
#         super().__init__()

#         self.band_count = _positive_int(
#             band_count,
#             "spectral_bands",
#         )
#         self.feature_dim = _positive_int(
#             feature_dim,
#             "d_model",
#         )
#         self.region_count = _positive_int(
#             region_count,
#             "spectral_token_count",
#         )

#         self.region_bounds = _spectral_region_bounds(
#             self.band_count,
#             self.region_count,
#         )

#         self.use_physical_positions = spectral_positions is not None

#         if self.use_physical_positions:
#             positions = _spectral_positions(
#                 self.band_count,
#                 spectral_positions,
#             )
#         else:
#             positions = torch.empty(
#                 0,
#                 dtype=torch.float32,
#             )

#         self.register_buffer(
#             "band_positions",
#             positions,
#         )

#         self.standardizer = FrozenBandStandardizer(self.band_count)

#         embedding_dim = 2 if self.use_physical_positions else 1

#         # No bias: a shared class-independent offset is unnecessary before the
#         # ordered spectral class structure is formed downstream.
#         self.band_embedding = nn.Linear(
#             embedding_dim,
#             self.feature_dim,
#             bias=False,
#         )

#         layer_count = _positive_int(
#             layers,
#             "num_spectral_layers",
#         )

#         self.blocks = nn.ModuleList(
#             [
#                 MambaBlock(
#                     self.feature_dim,
#                     d_state=d_state,
#                     d_conv=d_conv,
#                     expand=expand,
#                     dropout=dropout,
#                 )
#                 for _ in range(layer_count)
#             ]
#         )

#     def _encode(
#         self,
#         band_features: Tensor,
#     ) -> Tensor:
#         output = band_features
#         for block in self.blocks:
#             output = block(output)
#         return output

#     def _reduce_regions(
#         self,
#         band_features: Tensor,
#     ) -> Tensor:
#         if band_features.shape[1:] != (
#             self.band_count,
#             self.feature_dim,
#         ):
#             raise RuntimeError("encoded band-feature shape is invalid")

#         region_features = torch.stack(
#             [
#                 band_features[
#                     :,
#                     start:end,
#                 ].mean(dim=1)
#                 for start, end in self.region_bounds
#             ],
#             dim=1,
#         )

#         expected = (
#             band_features.size(0),
#             self.region_count,
#             self.feature_dim,
#         )

#         if region_features.shape != expected:
#             raise RuntimeError(
#                 "spectral-region features have shape "
#                 f"{tuple(region_features.shape)}, expected {expected}"
#             )

#         return region_features

#     def forward(
#         self,
#         center_spectrum: Tensor,
#     ) -> _SpectralEncoderOutput:
#         if (
#             center_spectrum.ndim != 2
#             or center_spectrum.size(0) == 0
#             or center_spectrum.size(1) != self.band_count
#         ):
#             raise ValueError(
#                 "center_spectrum must have shape " f"[N,{self.band_count}]"
#             )

#         if not torch.is_floating_point(center_spectrum) or center_spectrum.is_complex():
#             raise TypeError("center_spectrum must be real floating point")

#         if not bool(torch.isfinite(center_spectrum).all()):
#             raise ValueError("center_spectrum contains NaN or Inf")

#         normalized = self.standardizer(center_spectrum)

#         if self.use_physical_positions:
#             positions = (
#                 self.band_positions.to(
#                     device=normalized.device,
#                     dtype=normalized.dtype,
#                 )
#                 .unsqueeze(0)
#                 .expand(
#                     normalized.size(0),
#                     -1,
#                 )
#             )

#             embedding_input = torch.stack(
#                 (
#                     normalized,
#                     positions,
#                 ),
#                 dim=-1,
#             )
#         else:
#             embedding_input = normalized.unsqueeze(-1)

#         band_features = self.band_embedding(embedding_input)

#         # HSI spectral classification is non-causal.  One shared encoder is
#         # applied in both wavelength directions and both outputs are restored to
#         # the same physical band order before averaging.
#         paired = torch.cat(
#             (
#                 band_features,
#                 band_features.flip(1),
#             ),
#             dim=0,
#         )

#         encoded = self._encode(paired)

#         forward_features, reverse_features = encoded.chunk(
#             2,
#             dim=0,
#         )

#         band_features = (forward_features + reverse_features.flip(1)) * 0.5

#         region_features = self._reduce_regions(band_features)

#         # Use all encoded bands for the spectral query used by the spatial
#         # context pool; this avoids region-size weighting when B % K != 0.
#         summary = band_features.mean(dim=1)

#         if (
#             not bool(torch.isfinite(region_features).all())
#             or not bool(torch.isfinite(summary).all())
#             or not bool(torch.isfinite(band_features).all())
#         ):
#             raise RuntimeError("ordered spectral encoder produced NaN or Inf")

#         return _SpectralEncoderOutput(
#             region_features=region_features,
#             summary=summary,
#             band_features=band_features,
#         )


# # -----------------------------------------------------------------------------
# # Spatial context encoder
# # -----------------------------------------------------------------------------


# class SpectralConditionedSpatialPool(nn.Module):
#     """Pool spatial tokens using the encoded full-spectrum summary as query."""

#     def __init__(
#         self,
#         feature_dim: int,
#     ) -> None:
#         super().__init__()

#         self.feature_dim = _positive_int(
#             feature_dim,
#             "d_model",
#         )

#         self.spatial_normalization = nn.LayerNorm(self.feature_dim)
#         self.spectral_normalization = nn.LayerNorm(self.feature_dim)

#         self.query_projection = nn.Linear(
#             self.feature_dim,
#             self.feature_dim,
#             bias=False,
#         )
#         self.key_projection = nn.Linear(
#             self.feature_dim,
#             self.feature_dim,
#             bias=False,
#         )

#     def forward(
#         self,
#         spatial_features: Tensor,
#         spectral_summary: Tensor,
#     ) -> tuple[Tensor, Tensor]:
#         if (
#             spatial_features.ndim != 3
#             or spatial_features.size(0) == 0
#             or spatial_features.size(1) == 0
#             or spatial_features.size(2) != self.feature_dim
#         ):
#             raise ValueError(
#                 "spatial_features must have shape " f"[N,L,{self.feature_dim}]"
#             )

#         if spectral_summary.shape != (
#             spatial_features.size(0),
#             self.feature_dim,
#         ):
#             raise ValueError(
#                 "spectral_summary must have shape " f"[N,{self.feature_dim}]"
#             )

#         if spatial_features.device != spectral_summary.device:
#             raise ValueError(
#                 "spatial features and spectral summary must share a device"
#             )

#         if spatial_features.dtype != spectral_summary.dtype:
#             raise ValueError("spatial features and spectral summary must share a dtype")

#         if not bool(torch.isfinite(spatial_features).all()) or not bool(
#             torch.isfinite(spectral_summary).all()
#         ):
#             raise ValueError("spatial-pooling inputs contain NaN or Inf")

#         query = self.query_projection(
#             self.spectral_normalization(spectral_summary)
#         ).unsqueeze(1)

#         keys = self.key_projection(self.spatial_normalization(spatial_features))

#         scores = (query * keys).sum(dim=-1) / math.sqrt(self.feature_dim)

#         weights = F.softmax(
#             scores,
#             dim=1,
#         )

#         context = (spatial_features * weights.unsqueeze(-1)).sum(dim=1)

#         if not bool(torch.isfinite(context).all()) or not bool(
#             torch.isfinite(weights).all()
#         ):
#             raise RuntimeError("spectral-conditioned spatial pool produced NaN or Inf")

#         return (
#             context,
#             weights,
#         )


# @dataclass(frozen=True)
# class _SpatialEncoderOutput:
#     context: Tensor
#     spatial_features: Tensor
#     pooling_weights: Tensor
#     relative_patch: Tensor


# class CenterRelativeSpatialEncoder(nn.Module):
#     """Encode neighborhood differences around the classified center pixel."""

#     def __init__(
#         self,
#         input_channels: int,
#         patch_size: int,
#         feature_dim: int,
#         *,
#         layers: int,
#         d_state: int,
#         d_conv: int,
#         expand: int,
#         dropout: float,
#         norm_groups: int,
#     ) -> None:
#         super().__init__()

#         self.input_channels = _positive_int(
#             input_channels,
#             "num_bands",
#         )
#         self.patch_size = _positive_int(
#             patch_size,
#             "patch_size",
#         )
#         self.feature_dim = _positive_int(
#             feature_dim,
#             "d_model",
#         )
#         self.spatial_token_count = self.patch_size**2

#         group_count = _positive_int(
#             norm_groups,
#             "stem_norm_groups",
#         )

#         if group_count > self.feature_dim or self.feature_dim % group_count != 0:
#             raise ValueError(
#                 "stem_norm_groups must divide d_model " f"({self.feature_dim})"
#             )

#         self.register_buffer(
#             "spatial_positions",
#             _spatial_positions(self.patch_size),
#             persistent=False,
#         )

#         scan_orders = _serpentine_orders(self.patch_size)

#         self.register_buffer(
#             "scan_orders",
#             scan_orders,
#             persistent=False,
#         )
#         self.register_buffer(
#             "inverse_scan_orders",
#             scan_orders.argsort(dim=1),
#             persistent=False,
#         )

#         self.stem = nn.Sequential(
#             nn.Conv2d(
#                 self.input_channels + 2,
#                 self.feature_dim,
#                 kernel_size=1,
#                 bias=False,
#             ),
#             nn.GroupNorm(
#                 group_count,
#                 self.feature_dim,
#             ),
#             nn.GELU(),
#             nn.Conv2d(
#                 self.feature_dim,
#                 self.feature_dim,
#                 kernel_size=3,
#                 padding=1,
#                 groups=self.feature_dim,
#                 bias=False,
#             ),
#             nn.GELU(),
#             nn.Conv2d(
#                 self.feature_dim,
#                 self.feature_dim,
#                 kernel_size=1,
#                 bias=False,
#             ),
#         )

#         layer_count = _positive_int(
#             layers,
#             "num_spatial_layers",
#         )

#         self.blocks = nn.ModuleList(
#             [
#                 MambaBlock(
#                     self.feature_dim,
#                     d_state=d_state,
#                     d_conv=d_conv,
#                     expand=expand,
#                     dropout=dropout,
#                 )
#                 for _ in range(layer_count)
#             ]
#         )

#         self.pool = SpectralConditionedSpatialPool(self.feature_dim)

#     def _encode(
#         self,
#         spatial_features: Tensor,
#     ) -> Tensor:
#         output = spatial_features
#         for block in self.blocks:
#             output = block(output)
#         return output

#     def forward(
#         self,
#         patch: Tensor,
#         spectral_summary: Tensor,
#     ) -> _SpatialEncoderOutput:
#         expected_patch = (
#             self.input_channels,
#             self.patch_size,
#             self.patch_size,
#         )

#         if (
#             patch.ndim != 4
#             or patch.size(0) == 0
#             or tuple(patch.shape[1:]) != expected_patch
#         ):
#             raise ValueError(
#                 "patch must have shape "
#                 f"[N,{self.input_channels},"
#                 f"{self.patch_size},{self.patch_size}]"
#             )

#         if spectral_summary.shape != (
#             patch.size(0),
#             self.feature_dim,
#         ):
#             raise ValueError(
#                 "spectral_summary must have shape " f"[N,{self.feature_dim}]"
#             )

#         if not torch.is_floating_point(patch) or patch.is_complex():
#             raise TypeError("patch must be real floating point")

#         if (
#             not torch.is_floating_point(spectral_summary)
#             or spectral_summary.is_complex()
#         ):
#             raise TypeError("spectral_summary must be real floating point")

#         if patch.device != spectral_summary.device:
#             raise ValueError("patch and spectral_summary must share a device")

#         if patch.dtype != spectral_summary.dtype:
#             raise ValueError("patch and spectral_summary must share a dtype")

#         if not bool(torch.isfinite(patch).all()) or not bool(
#             torch.isfinite(spectral_summary).all()
#         ):
#             raise ValueError("spatial encoder inputs contain NaN or Inf")

#         center_index = self.patch_size // 2

#         center = patch[
#             :,
#             :,
#             center_index,
#             center_index,
#         ]

#         # The raw center spectrum carries absolute spectral identity.  This
#         # spatial branch therefore models neighborhood differences rather than
#         # duplicating that center spectrum in a second representation.
#         relative_patch = (
#             patch
#             - center[
#                 :,
#                 :,
#                 None,
#                 None,
#             ]
#         )

#         positions = self.spatial_positions.to(
#             device=patch.device,
#             dtype=patch.dtype,
#         ).expand(
#             patch.size(0),
#             -1,
#             -1,
#             -1,
#         )

#         spatial_features = self.stem(
#             torch.cat(
#                 (
#                     relative_patch,
#                     positions,
#                 ),
#                 dim=1,
#             )
#         )

#         spatial_features = spatial_features.flatten(2).transpose(
#             1,
#             2,
#         )

#         batch_size = spatial_features.size(0)

#         scanned = torch.cat(
#             [
#                 spatial_features.index_select(
#                     1,
#                     order,
#                 )
#                 for order in self.scan_orders
#             ],
#             dim=0,
#         )

#         encoded = self._encode(scanned).reshape(
#             4,
#             batch_size,
#             self.spatial_token_count,
#             self.feature_dim,
#         )

#         restored = torch.stack(
#             [
#                 encoded[scan_index].index_select(
#                     1,
#                     self.inverse_scan_orders[scan_index],
#                 )
#                 for scan_index in range(4)
#             ],
#             dim=0,
#         )

#         # Each spatial location is represented by the mean response over the
#         # four shared-weight scan directions after restoring the same grid order.
#         spatial_features = restored.mean(dim=0)

#         context, pooling_weights = self.pool(
#             spatial_features,
#             spectral_summary,
#         )

#         if not bool(torch.isfinite(spatial_features).all()):
#             raise RuntimeError("spatial encoder produced NaN or Inf")

#         return _SpatialEncoderOutput(
#             context=context,
#             spatial_features=spatial_features,
#             pooling_weights=pooling_weights,
#             relative_patch=relative_patch,
#         )


# # -----------------------------------------------------------------------------
# # Direct spectral-spatial fusion
# # -----------------------------------------------------------------------------


# class SpectralSpatialFusion(nn.Module):
#     """Fuse context into Q without destroying spectral-region order.

#     Let T_k be the encoded feature for ordered spectral region k and c the
#     pooled spatial context.  The fusion is

#         h_k = LN(T_k) * W_c LN(c)
#         q_k = T_k + W_o SiLU(h_k).

#     The element-wise interaction makes the spatial correction token-specific:
#     the same context does not simply add one identical vector to every spectral
#     region.  The residual keeps T_k as the spectral anchor, while W_o allows the
#     context to change both magnitude and feature direction.

#     No softmax competition across spectral regions and no K-dependent scaling
#     are used because the downstream model operates directly on the ordered Q.
#     """

#     def __init__(
#         self,
#         feature_dim: int,
#     ) -> None:
#         super().__init__()

#         self.feature_dim = _positive_int(
#             feature_dim,
#             "d_model",
#         )

#         self.spectral_normalization = nn.LayerNorm(self.feature_dim)
#         self.context_normalization = nn.LayerNorm(self.feature_dim)

#         self.context_projection = nn.Linear(
#             self.feature_dim,
#             self.feature_dim,
#             bias=False,
#         )

#         self.output_projection = nn.Linear(
#             self.feature_dim,
#             self.feature_dim,
#             bias=False,
#         )

#     def forward(
#         self,
#         spectral_region_features: Tensor,
#         spatial_context: Tensor,
#     ) -> Tensor:
#         if (
#             spectral_region_features.ndim != 3
#             or spectral_region_features.size(0) == 0
#             or spectral_region_features.size(1) < 2
#             or spectral_region_features.size(2) != self.feature_dim
#         ):
#             raise ValueError(
#                 "spectral_region_features must have shape "
#                 f"[N,K,{self.feature_dim}] with K>=2"
#             )

#         if spatial_context.shape != (
#             spectral_region_features.size(0),
#             self.feature_dim,
#         ):
#             raise ValueError(
#                 "spatial_context must have shape " f"[N,{self.feature_dim}]"
#             )

#         if spectral_region_features.device != spatial_context.device:
#             raise ValueError("spectral and spatial features must share a device")

#         if spectral_region_features.dtype != spatial_context.dtype:
#             raise ValueError("spectral and spatial features must share a dtype")

#         if not bool(torch.isfinite(spectral_region_features).all()) or not bool(
#             torch.isfinite(spatial_context).all()
#         ):
#             raise ValueError("spectral-spatial fusion inputs contain NaN or Inf")

#         normalized_spectral = self.spectral_normalization(spectral_region_features)

#         aligned_context = self.context_projection(
#             self.context_normalization(spatial_context)
#         ).unsqueeze(1)

#         interaction = normalized_spectral * aligned_context

#         correction = self.output_projection(F.silu(interaction))

#         ordered_features = spectral_region_features + correction

#         if not bool(torch.isfinite(ordered_features).all()):
#             raise RuntimeError("spectral-spatial fusion produced NaN or Inf")

#         return ordered_features


# # -----------------------------------------------------------------------------
# # Public backbone
# # -----------------------------------------------------------------------------


# @dataclass(frozen=True)
# class BackboneOutput:
#     """Output of the HSI feature extractor.

#     ``ordered_features`` is the only mandatory backbone output:

#         Q_raw(x) in R^{N x K x D}.

#     Its second axis follows spectral-region order and this tensor is the deployed
#     Q used by the prediction head and class-structure bank.  Physical spectral
#     evidence P is produced separately from the raw centre spectrum and is not a
#     second learned representation.

#     All other fields are optional diagnostics returned only when
#     ``return_aux=True``; they are not alternate downstream representations.
#     """

#     ordered_features: Tensor

#     spectral_region_features: Tensor | None = None
#     encoded_band_features: Tensor | None = None
#     spectral_summary: Tensor | None = None

#     spatial_context: Tensor | None = None
#     spatial_features: Tensor | None = None
#     spatial_pool_weights: Tensor | None = None
#     relative_patch: Tensor | None = None


# class HSIMambaFeatureExtractor(nn.Module):
#     """Extract one ordered spectral-spatial feature tensor Q(x) per HSI sample.

#     Continual-learning contract
#     --------------------------
#     Base and incremental learning keep one learned representation Q[K,D].  During
#     incremental learning the learned extractor remains plastic; stability is supplied
#     by old-logit KD, Whole-CDS preservation, CDS-based old/new separation, and
#     semantic-drift correction rather than by freezing arbitrary learned entry layers.
#     The phase-0 raw-spectrum standardizer remains fixed because it defines the
#     physical input coordinate.  A previous backbone copy is trainer-local/read-only.
#     """

#     def __init__(
#         self,
#         args: Any,
#     ) -> None:
#         super().__init__()

#         self.patch_bands = _positive_int(
#             args.num_bands,
#             "num_bands",
#         )

#         self.spectral_bands = _positive_int(
#             getattr(
#                 args,
#                 "spectral_bands",
#                 getattr(
#                     args,
#                     "raw_num_bands",
#                     args.num_bands,
#                 ),
#             ),
#             "spectral_bands",
#         )

#         self.patch_size = _positive_int(
#             args.patch_size,
#             "patch_size",
#         )

#         if self.patch_size % 2 == 0:
#             raise ValueError("patch_size must be odd so one center pixel is defined")

#         if not hasattr(
#             args,
#             "spectral_token_count",
#         ):
#             raise ValueError("spectral_token_count must be provided explicitly")

#         # Public backbone/structure contract.
#         self.spectral_token_count = _positive_int(
#             args.spectral_token_count,
#             "spectral_token_count",
#         )
#         self.spectral_token_dim = _positive_int(
#             args.d_model,
#             "d_model",
#         )

#         if self.spectral_token_count < 2:
#             raise ValueError(
#                 "spectral_token_count must be at least 2 because "
#                 "ordered spectral structure models cross-region interactions"
#             )

#         if self.spectral_token_count > self.spectral_bands:
#             raise ValueError("spectral_token_count cannot exceed spectral_bands")

#         d_state = _positive_int(
#             getattr(
#                 args,
#                 "d_state",
#                 16,
#             ),
#             "d_state",
#         )
#         d_conv = _positive_int(
#             getattr(
#                 args,
#                 "d_conv",
#                 4,
#             ),
#             "d_conv",
#         )
#         expand = _positive_int(
#             getattr(
#                 args,
#                 "expand",
#                 2,
#             ),
#             "expand",
#         )

#         dropout = float(
#             getattr(
#                 args,
#                 "mamba_dropout",
#                 0.0,
#             )
#         )

#         if not math.isfinite(dropout) or not 0.0 <= dropout < 1.0:
#             raise ValueError("mamba_dropout must be finite and lie in [0,1)")

#         spectral_positions = getattr(
#             args,
#             "band_wavelengths",
#             None,
#         )

#         if spectral_positions is None:
#             spectral_positions = getattr(
#                 args,
#                 "band_positions",
#                 None,
#             )

#         self.spectral_encoder = OrderedSpectralEncoder(
#             self.spectral_bands,
#             self.spectral_token_dim,
#             self.spectral_token_count,
#             layers=getattr(
#                 args,
#                 "num_spectral_layers",
#                 2,
#             ),
#             d_state=d_state,
#             d_conv=d_conv,
#             expand=expand,
#             dropout=dropout,
#             spectral_positions=spectral_positions,
#         )

#         normalization_mean = getattr(
#             args,
#             "spectral_normalization_mean",
#             None,
#         )
#         normalization_scale = getattr(
#             args,
#             "spectral_normalization_std",
#             None,
#         )

#         if (normalization_mean is None) != (normalization_scale is None):
#             raise ValueError(
#                 "spectral_normalization_mean and "
#                 "spectral_normalization_std must be supplied together"
#             )

#         if normalization_mean is not None:
#             self.spectral_encoder.standardizer.set_statistics(
#                 normalization_mean,
#                 normalization_scale,
#             )

#         self.spatial_encoder = CenterRelativeSpatialEncoder(
#             self.patch_bands,
#             self.patch_size,
#             self.spectral_token_dim,
#             layers=getattr(
#                 args,
#                 "num_spatial_layers",
#                 2,
#             ),
#             d_state=d_state,
#             d_conv=d_conv,
#             expand=expand,
#             dropout=dropout,
#             norm_groups=getattr(
#                 args,
#                 "stem_norm_groups",
#                 1,
#             ),
#         )

#         self.fusion = SpectralSpatialFusion(self.spectral_token_dim)

#         # Explicit phase trainability state. The trainer should call the public
#         # methods below rather than toggling arbitrary submodules.

#     @staticmethod
#     def _set_module_trainable(module: nn.Module, enabled: bool) -> None:
#         for parameter in module.parameters():
#             parameter.requires_grad_(bool(enabled))
#             if not bool(enabled):
#                 parameter.grad = None
#         if not bool(enabled):
#             module.eval()

#     def set_base_trainability(self) -> None:
#         """Base phase: learn the complete spectral-spatial representation."""
#         self._set_module_trainable(self.spectral_encoder, True)
#         self._set_module_trainable(self.spatial_encoder, True)
#         self._set_module_trainable(self.fusion, True)

#     def set_incremental_trainability(self) -> None:
#         """Incremental phase: keep the full learned representation plastic.

#         The phase-0 standardizer is buffer state, not a trainable module.  Learned
#         entry maps, Mamba/SSM blocks, spatial context and fusion all remain adaptive;
#         continual stability is enforced by the explicit NECIL objectives.
#         """
#         if not self.spectral_normalization_fitted:
#             raise RuntimeError(
#                 "spectral normalization must be fitted before incremental trainability"
#             )
#         self._set_module_trainable(self.spectral_encoder, True)
#         self._set_module_trainable(self.spatial_encoder, True)
#         self._set_module_trainable(self.fusion, True)

#     def incremental_trainable_parameters(self) -> tuple[nn.Parameter, ...]:
#         parameters = tuple(
#             parameter for parameter in self.parameters() if parameter.requires_grad
#         )
#         if not parameters:
#             raise RuntimeError("incremental backbone has no trainable parameters")
#         return parameters

#     @property
#     def spectral_normalization_fitted(
#         self,
#     ) -> bool:
#         return bool(self.spectral_encoder.standardizer.fitted.item())

#     @torch.no_grad()
#     def set_spectral_normalization(
#         self,
#         mean: Tensor | Sequence[float],
#         scale: Tensor | Sequence[float],
#         *,
#         overwrite: bool = False,
#     ) -> None:
#         self.spectral_encoder.standardizer.set_statistics(
#             mean,
#             scale,
#             overwrite=overwrite,
#         )

#     @torch.no_grad()
#     def fit_spectral_normalization(
#         self,
#         base_center_spectra: Tensor,
#         *,
#         overwrite: bool = False,
#     ) -> None:
#         self.spectral_encoder.standardizer.fit(
#             base_center_spectra,
#             overwrite=overwrite,
#         )

#     def forward(
#         self,
#         patch: Tensor,
#         *,
#         center_spectrum: Tensor,
#         return_aux: bool = False,
#     ) -> BackboneOutput:
#         if (
#             not torch.is_tensor(patch)
#             or not torch.is_floating_point(patch)
#             or patch.is_complex()
#         ):
#             raise TypeError("patch must be a real floating-point tensor")

#         if (
#             not torch.is_tensor(center_spectrum)
#             or not torch.is_floating_point(center_spectrum)
#             or center_spectrum.is_complex()
#         ):
#             raise TypeError("center_spectrum must be a real floating-point tensor")

#         expected_patch = (
#             self.patch_bands,
#             self.patch_size,
#             self.patch_size,
#         )

#         if (
#             patch.ndim != 4
#             or patch.size(0) == 0
#             or tuple(patch.shape[1:]) != expected_patch
#         ):
#             raise ValueError(
#                 "patch must have shape "
#                 f"[N,{self.patch_bands},"
#                 f"{self.patch_size},{self.patch_size}]"
#             )

#         if center_spectrum.shape != (
#             patch.size(0),
#             self.spectral_bands,
#         ):
#             raise ValueError(
#                 "center_spectrum must have shape " f"[N,{self.spectral_bands}]"
#             )

#         if patch.device != center_spectrum.device:
#             raise ValueError("patch and center_spectrum must share a device")

#         if patch.dtype != center_spectrum.dtype:
#             raise ValueError("patch and center_spectrum must share a dtype")

#         if not bool(torch.isfinite(patch).all()) or not bool(
#             torch.isfinite(center_spectrum).all()
#         ):
#             raise ValueError("backbone inputs contain NaN or Inf")

#         spectral = self.spectral_encoder(center_spectrum)

#         spatial = self.spatial_encoder(
#             patch,
#             spectral.summary,
#         )

#         ordered_features = self.fusion(
#             spectral.region_features,
#             spatial.context,
#         )

#         expected_output = (
#             patch.size(0),
#             self.spectral_token_count,
#             self.spectral_token_dim,
#         )

#         if ordered_features.shape != expected_output:
#             raise RuntimeError(
#                 "ordered_features have shape "
#                 f"{tuple(ordered_features.shape)}, "
#                 f"expected {expected_output}"
#             )

#         if not bool(torch.isfinite(ordered_features).all()):
#             raise RuntimeError("backbone produced NaN or Inf")

#         if not return_aux:
#             return BackboneOutput(
#                 ordered_features=ordered_features,
#             )

#         return BackboneOutput(
#             ordered_features=ordered_features,
#             spectral_region_features=(spectral.region_features),
#             encoded_band_features=(spectral.band_features),
#             spectral_summary=(spectral.summary),
#             spatial_context=(spatial.context),
#             spatial_features=(spatial.spatial_features),
#             spatial_pool_weights=(spatial.pooling_weights),
#             relative_patch=(spatial.relative_patch),
#         )


# __all__ = [
#     "BackboneOutput",
#     "HSIMambaFeatureExtractor",
# ]
