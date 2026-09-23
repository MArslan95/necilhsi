from __future__ import annotations

"""Transient HSI support reconstruction from the complete historical CDS.

Persistent memory is never expanded: the only stored historical state remains
{I_Q,V_Q,I_P,V_P,N}. This module uses that state to optimize a temporary set of
spectral-spatial HSI patches under the frozen previous model. The temporary
supports are discarded after the incremental phase.
"""

from dataclasses import dataclass
import math
from typing import Any, Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from losses.loss import cds_structure_reconstruction_loss
from models.geometry_bank import (
    ClassStructureBatch,
    estimate_structure_batch_differentiable,
)

Tensor = torch.Tensor


class FrozenSpectralPreprocessor(nn.Module):
    """Differentiable copy of the real HSI loader preprocessing.

    Input is a native/raw HSI spectral patch, exactly like ``ordered_cube``.
    The real loader first standardizes every raw band using Phase-0 TRAIN
    statistics, then optionally applies PCA (and optional whitening).  Transient
    historical HSI must follow the identical map before entering the backbone.
    """

    def __init__(
        self,
        *,
        active: bool,
        raw_bands: int,
        model_bands: int,
        normalization_mean: Sequence[float] | Tensor,
        normalization_std: Sequence[float] | Tensor,
        mean: Sequence[float] | Tensor | None,
        components: Sequence[Sequence[float]] | Tensor | None,
        explained_variance: Sequence[float] | Tensor | None,
        whiten: bool,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        self.active = bool(active)
        self.raw_bands = int(raw_bands)
        self.model_bands = int(model_bands)
        self.whiten = bool(whiten)
        if self.raw_bands <= 1 or self.model_bands <= 0:
            raise ValueError("invalid spectral dimensions")

        norm_mean_t = torch.as_tensor(
            normalization_mean, device=device, dtype=dtype
        ).flatten()
        norm_std_t = torch.as_tensor(
            normalization_std, device=device, dtype=dtype
        ).flatten()
        if norm_mean_t.shape != (self.raw_bands,) or norm_std_t.shape != (self.raw_bands,):
            raise ValueError("raw spectral normalization has wrong shape")
        if not bool(torch.isfinite(norm_mean_t).all()) or not bool(torch.isfinite(norm_std_t).all()):
            raise ValueError("raw spectral normalization contains NaN or Inf")
        if bool((norm_std_t <= 0).any()):
            raise ValueError("raw spectral normalization std must be positive")

        if self.active:
            if mean is None or components is None:
                raise ValueError("active PCA reconstruction requires mean/components")
            mean_t = torch.as_tensor(mean, device=device, dtype=dtype).flatten()
            comp_t = torch.as_tensor(components, device=device, dtype=dtype)
            if mean_t.shape != (self.raw_bands,):
                raise ValueError("reduction mean has wrong shape")
            if comp_t.shape != (self.model_bands, self.raw_bands):
                raise ValueError("reduction components have wrong shape")
            if self.whiten:
                if explained_variance is None:
                    raise ValueError("whitened PCA requires explained variance")
                exp_t = torch.as_tensor(
                    explained_variance, device=device, dtype=dtype
                ).flatten()
                if exp_t.shape != (self.model_bands,):
                    raise ValueError("explained variance has wrong shape")
            else:
                exp_t = torch.ones(self.model_bands, device=device, dtype=dtype)
        else:
            if self.model_bands != self.raw_bands:
                raise ValueError("identity preprocessing requires model_bands == raw_bands")
            mean_t = torch.zeros(self.raw_bands, device=device, dtype=dtype)
            comp_t = torch.eye(self.raw_bands, device=device, dtype=dtype)
            exp_t = torch.ones(self.raw_bands, device=device, dtype=dtype)

        self.register_buffer("normalization_mean", norm_mean_t, persistent=False)
        self.register_buffer("normalization_std", norm_std_t, persistent=False)
        self.register_buffer("mean", mean_t, persistent=False)
        self.register_buffer("components", comp_t, persistent=False)
        self.register_buffer("explained_variance", exp_t, persistent=False)

    def forward(self, spectral_patch: Tensor) -> Tensor:
        values = torch.as_tensor(spectral_patch)
        if (
            values.ndim != 4
            or values.size(0) <= 0
            or values.size(1) != self.raw_bands
        ):
            raise ValueError(
                f"spectral_patch must be non-empty [N,{self.raw_bands},H,W]"
            )
        if not torch.is_floating_point(values) or values.is_complex():
            raise TypeError("spectral_patch must be real floating point")
        if not bool(torch.isfinite(values).all()):
            raise ValueError("spectral_patch contains NaN or Inf")
        n, _, h, w = values.shape
        flat_raw = values.permute(0, 2, 3, 1).reshape(-1, self.raw_bands)
        normalized = (
            flat_raw - self.normalization_mean.unsqueeze(0)
        ) / self.normalization_std.unsqueeze(0)

        if self.active:
            reduced = (normalized - self.mean.unsqueeze(0)) @ self.components.t()
            if self.whiten:
                reduced = reduced / self.explained_variance.clamp_min(1e-12).sqrt().unsqueeze(0)
        else:
            reduced = normalized

        result = reduced.reshape(n, h, w, self.model_bands).permute(0, 3, 1, 2).contiguous()
        if not bool(torch.isfinite(result).all()):
            raise RuntimeError("frozen spectral preprocessing produced NaN or Inf")
        return result


@dataclass(frozen=True)
class TransientHistoricalSupportPool:
    patches: Tensor
    spectra: Tensor
    labels: Tensor
    class_ids: Tensor
    supports_per_class: int
    diagnostics: Mapping[str, float]

    @property
    def sample_count(self) -> int:
        return int(self.labels.numel())

    def balanced_batch(self, *, per_class: int, step: int) -> tuple[Tensor, Tensor, Tensor]:
        count = int(per_class)
        if count <= 0:
            raise ValueError("per_class must be positive")
        cycle = int(step)
        indices = []
        for class_id in self.class_ids.tolist():
            class_rows = torch.nonzero(self.labels.eq(int(class_id)), as_tuple=False).flatten()
            if class_rows.numel() != self.supports_per_class:
                raise RuntimeError("transient support pool is not class-balanced")
            local = (
                torch.arange(count, device=class_rows.device, dtype=torch.long)
                + cycle * count
            ) % class_rows.numel()
            indices.append(class_rows.index_select(0, local))
        selected = torch.cat(indices, dim=0)
        return (
            self.patches.index_select(0, selected),
            self.spectra.index_select(0, selected),
            self.labels.index_select(0, selected),
        )


class CDSHistoricalSupportReconstructor:
    """Reconstruct balanced transient old HSI support from complete CDS."""

    def __init__(
        self,
        *,
        model: Any,
        reference_backbone: nn.Module,
        reference_classifier: nn.Module,
        historical_structure: ClassStructureBatch,
        patch_size: int,
        supports_per_class: int,
        generation_steps: int,
        generation_lr: float,
        reduction_active: bool,
        model_bands: int,
        preprocessing_normalization_mean: Sequence[float] | Tensor,
        preprocessing_normalization_std: Sequence[float] | Tensor,
        reduction_mean: Sequence[float] | Tensor | None,
        reduction_components: Sequence[Sequence[float]] | Tensor | None,
        reduction_explained_variance: Sequence[float] | Tensor | None,
        reduction_whiten: bool,
    ) -> None:
        self.model = model
        self.reference_backbone = reference_backbone
        self.reference_classifier = reference_classifier
        self.target = historical_structure
        self.patch_size = int(patch_size)
        self.supports_per_class = int(supports_per_class)
        self.generation_steps = int(generation_steps)
        self.generation_lr = float(generation_lr)
        if self.patch_size <= 0 or self.patch_size % 2 == 0:
            raise ValueError("patch_size must be a positive odd integer")
        if self.supports_per_class < 4:
            raise ValueError("supports_per_class must be >=4 to reconstruct median/IQR")
        if self.generation_steps <= 0:
            raise ValueError("generation_steps must be positive")
        if not math.isfinite(self.generation_lr) or self.generation_lr <= 0:
            raise ValueError("generation_lr must be finite and positive")

        self.device = model.device
        self.dtype = model.dtype
        self.class_ids = self.target.class_ids.to(device=self.device, dtype=torch.long)
        self.preprocessor = FrozenSpectralPreprocessor(
            active=bool(reduction_active),
            raw_bands=int(model.structure_bank.spectral_bands),
            model_bands=int(model_bands),
            normalization_mean=preprocessing_normalization_mean,
            normalization_std=preprocessing_normalization_std,
            mean=reduction_mean,
            components=reduction_components,
            explained_variance=reduction_explained_variance,
            whiten=bool(reduction_whiten),
            device=self.device,
            dtype=self.dtype,
        )

    @staticmethod
    def _local_targets(labels: Tensor, class_ids: Tensor) -> Tensor:
        matches = labels.unsqueeze(1).eq(class_ids.unsqueeze(0))
        if not bool(matches.sum(dim=1).eq(1).all()):
            raise ValueError("historical label is absent from class_ids")
        return matches.to(torch.long).argmax(dim=1)

    def _physical_spectral_module(self) -> nn.Module:
        """Return the fitted PhysicalSpectralEvidence module, not its wrapper method."""
        physical = getattr(self.model, "physical_spectral_extractor", None)
        if physical is None or not isinstance(physical, nn.Module):
            raise RuntimeError(
                "model must expose the fitted physical spectral module as "
                "physical_spectral_extractor"
            )
        if not hasattr(physical, "inverse"):
            raise RuntimeError(
                "physical_spectral_extractor must provide inverse(P) for "
                "transient CDS support reconstruction"
            )
        fitted = getattr(physical, "fitted", None)
        if fitted is not None and not bool(torch.as_tensor(fitted).item()):
            raise RuntimeError("physical spectral normalization must be fitted first")
        return physical

    def _initial_spectral_patch(self) -> tuple[Tensor, Tensor]:
        # `model.physical_spectral_evidence(...)` is a wrapper function.  The
        # actual fitted full-band P <-> native-spectrum coordinate transform lives
        # in `physical_spectral_extractor`; use that module for the inverse map.
        physical = self._physical_spectral_module()
        identity_spectrum = physical.inverse(self.target.physical_identity).to(
            device=self.device, dtype=self.dtype
        )
        c, b = identity_spectrum.shape
        m = self.supports_per_class
        center = identity_spectrum[:, None, :].expand(c, m, b).reshape(c * m, b)

        # V_P is used only to seed diversity; the optimized set is judged by the
        # exact reconstructed CDS and V_P is never treated as a hard radius.
        variation = self.target.physical_variation.to(device=self.device, dtype=self.dtype)
        diagonal = torch.arange(b, device=self.device)
        p_scale = variation[:, diagonal, diagonal, 0]
        physical_band_scale = physical.band_scale.to(device=self.device, dtype=self.dtype)
        spectral_scale = (p_scale * physical_band_scale.unsqueeze(0)).clamp_min(1e-4)
        noise = torch.randn(c, m, b, device=self.device, dtype=self.dtype)
        noise = noise * spectral_scale[:, None, :] * 0.25
        center = center + noise.reshape(c * m, b)

        patch = center[:, :, None, None].expand(-1, -1, self.patch_size, self.patch_size).clone()
        spatial_scale = spectral_scale[:, None, :].expand(c, m, b).reshape(c * m, b, 1, 1)
        spatial_noise = torch.randn_like(patch) * spatial_scale * 0.05
        patch = patch + spatial_noise
        labels = self.class_ids[:, None].expand(c, m).reshape(-1).contiguous()
        return patch, labels

    def generate(self) -> TransientHistoricalSupportPool:
        self.model.eval()
        self.reference_backbone.eval()
        self.reference_classifier.eval()
        if any(p.requires_grad for p in self.reference_backbone.parameters()):
            raise RuntimeError("reference backbone must be frozen during support reconstruction")
        if any(p.requires_grad for p in self.reference_classifier.parameters()):
            raise RuntimeError("reference classifier must be frozen during support reconstruction")

        spectral_patch, labels = self._initial_spectral_patch()
        spectral_patch = nn.Parameter(spectral_patch)
        optimizer = torch.optim.Adam([spectral_patch], lr=self.generation_lr)
        center_index = self.patch_size // 2
        target = ClassStructureBatch(
            class_ids=self.target.class_ids.to(self.device),
            sample_counts=self.target.sample_counts.to(self.device),
            learned_identity=self.target.learned_identity.to(self.device, self.dtype),
            learned_variation=self.target.learned_variation.to(self.device, self.dtype),
            physical_identity=self.target.physical_identity.to(self.device, self.dtype),
            physical_variation=self.target.physical_variation.to(self.device, self.dtype),
        )

        last = None
        for _ in range(self.generation_steps):
            optimizer.zero_grad(set_to_none=True)
            processed_patch = self.preprocessor(spectral_patch)
            center_spectrum = spectral_patch[:, :, center_index, center_index]
            teacher_features = self.model.extract_features_with_backbone(
                self.reference_backbone,
                processed_patch,
                center_spectrum=center_spectrum,
                return_aux=False,
            ).ordered_features
            teacher_output = self.reference_classifier(
                teacher_features,
                class_ids=self.class_ids,
                trainable_class_ids=(),
            )
            local_targets = self._local_targets(labels, self.class_ids)
            teacher_ce = F.cross_entropy(teacher_output.logits, local_targets)
            physical_evidence = self.model.physical_spectral_evidence(center_spectrum)
            reconstructed = estimate_structure_batch_differentiable(
                teacher_features,
                physical_evidence,
                labels,
                class_ids=self.class_ids,
            )
            cds = cds_structure_reconstruction_loss(
                reconstructed=reconstructed,
                target=target,
            )
            loss = teacher_ce + cds.loss
            if not bool(torch.isfinite(loss)):
                raise RuntimeError("transient historical support reconstruction became NaN or Inf")
            loss.backward()
            if spectral_patch.grad is None or not bool(torch.isfinite(spectral_patch.grad).all()):
                raise RuntimeError("support reconstruction gradient is missing or non-finite")
            torch.nn.utils.clip_grad_norm_([spectral_patch], 5.0, error_if_nonfinite=True)
            optimizer.step()
            last = (teacher_ce.detach(), cds)

        with torch.no_grad():
            processed_patch = self.preprocessor(spectral_patch.detach())
            center_spectrum = spectral_patch.detach()[:, :, center_index, center_index].contiguous()
            teacher_features = self.model.extract_features_with_backbone(
                self.reference_backbone,
                processed_patch,
                center_spectrum=center_spectrum,
                return_aux=False,
            ).ordered_features
            teacher_output = self.reference_classifier(
                teacher_features,
                class_ids=self.class_ids,
                trainable_class_ids=(),
            )
            local_targets = self._local_targets(labels, self.class_ids)
            teacher_accuracy = teacher_output.logits.argmax(dim=1).eq(local_targets).to(self.dtype).mean()
            physical_evidence = self.model.physical_spectral_evidence(center_spectrum)
            reconstructed = estimate_structure_batch_differentiable(
                teacher_features,
                physical_evidence,
                labels,
                class_ids=self.class_ids,
            )
            cds = cds_structure_reconstruction_loss(reconstructed=reconstructed, target=target)
            teacher_ce = F.cross_entropy(teacher_output.logits, local_targets)

        diagnostics = {
            "teacher_ce": float(teacher_ce.item()),
            "teacher_accuracy": float(teacher_accuracy.item()),
            "cds_reconstruction": float(cds.loss.item()),
            "cds_intra": float(cds.intra_loss.item()),
            "cds_inter": float(cds.inter_loss.item()),
            "i_q_error": float(cds.learned_identity_loss.item()),
            "v_q_error": float(cds.learned_variation_loss.item()),
            "i_p_error": float(cds.physical_identity_loss.item()),
            "v_p_error": float(cds.physical_variation_loss.item()),
            "learned_inter_error": float(cds.learned_inter_loss.item()),
            "physical_inter_error": float(cds.physical_inter_loss.item()),
        }
        return TransientHistoricalSupportPool(
            patches=processed_patch.detach(),
            spectra=center_spectrum.detach(),
            labels=labels.detach(),
            class_ids=self.class_ids.detach().clone(),
            supports_per_class=self.supports_per_class,
            diagnostics=diagnostics,
        )


__all__ = [
    "FrozenSpectralPreprocessor",
    "TransientHistoricalSupportPool",
    "CDSHistoricalSupportReconstructor",
]
