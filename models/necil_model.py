from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Sequence

import torch
import torch.nn as nn

from models.backbone import BackboneOutput, HSIMambaFeatureExtractor
from models.classifier import ClassificationOutput, CosineClassificationHead
from models.geometry_bank import (
    ClassDiscriminativeStructureState,
    ClassStructureBatch,
    HSIClassDiscriminativeStructureBank,
    PhysicalSpectralStructureBatch,
)
from models.spectral_evidence import PhysicalSpectralEvidence

Tensor = torch.Tensor


@dataclass(frozen=True)
class NECILOutput:
    backbone: BackboneOutput
    physical_spectral_evidence: Tensor
    classification: ClassificationOutput


class NECILModel(nn.Module):

    def __init__(self, args: Any) -> None:
        super().__init__()
        self.backbone = HSIMambaFeatureExtractor(args)
        reference = next(self.backbone.parameters(), None)
        if reference is None:
            raise RuntimeError("backbone must contain parameters")
        if reference.dtype not in (torch.float32, torch.float64):
            raise TypeError("backbone parameters must use float32 or float64")

        self.physical_spectral_extractor = PhysicalSpectralEvidence(
            band_count=int(self.backbone.spectral_bands)
        )
        positions = self.backbone.spectral_encoder.band_positions.detach().cpu()
        band_positions = (
            positions
            if positions.numel() == int(self.backbone.spectral_bands)
            else None
        )
        self.structure_bank = HSIClassDiscriminativeStructureBank(
            spectral_token_count=int(self.backbone.spectral_token_count),
            spectral_token_dim=int(self.backbone.spectral_token_dim),
            spectral_bands=int(self.backbone.spectral_bands),
            spectral_band_positions=band_positions,
            device=reference.device,
            dtype=reference.dtype,
        )

        total_class_count = int(getattr(args, "class_count", 0))
        if total_class_count < 2:
            raise ValueError(
                "args.class_count must be set before NECILModel construction"
            )
        self.classifier = CosineClassificationHead(
            spectral_token_count=int(self.backbone.spectral_token_count),
            spectral_token_dim=int(self.backbone.spectral_token_dim),
            total_class_count=total_class_count,
        )
        self.validate_model_state()

    def _reference_parameter(self) -> Tensor:
        parameter = next(self.backbone.parameters(), None)
        if parameter is None:
            raise RuntimeError("backbone has no parameters")
        return parameter

    @property
    def device(self) -> torch.device:
        return self._reference_parameter().device

    @property
    def dtype(self) -> torch.dtype:
        return self._reference_parameter().dtype

    @property
    def spectral_token_count(self) -> int:
        return int(self.backbone.spectral_token_count)

    @property
    def spectral_token_dim(self) -> int:
        return int(self.backbone.spectral_token_dim)

    @property
    def spectral_bands(self) -> int:
        return int(self.backbone.spectral_bands)

    @property
    def committed_class_ids(self) -> tuple[int, ...]:
        return self.structure_bank.seen_class_ids

    @property
    def spectral_normalization_fitted(self) -> bool:
        return bool(self.backbone.spectral_normalization_fitted) and bool(
            self.physical_spectral_extractor.fitted.item()
        )

    def validate_model_state(self) -> bool:
        if not isinstance(self.classifier, CosineClassificationHead):
            raise RuntimeError("classifier must be CosineClassificationHead")
        if not bool(torch.isfinite(self.classifier.logit_scale)):
            raise RuntimeError("classifier logit scale is invalid")
        if not isinstance(self.structure_bank, HSIClassDiscriminativeStructureBank):
            raise RuntimeError(
                "structure_bank must be HSIClassDiscriminativeStructureBank"
            )
        self.structure_bank.validate_state()
        if (
            self.structure_bank.device != self.device
            or self.structure_bank.dtype != self.dtype
        ):
            raise RuntimeError("backbone and CDS bank use different device/dtype")
        if self.structure_bank.spectral_token_count != self.spectral_token_count:
            raise RuntimeError("backbone and CDS bank disagree on K")
        if self.structure_bank.spectral_token_dim != self.spectral_token_dim:
            raise RuntimeError("backbone and CDS bank disagree on D")
        if self.structure_bank.spectral_bands != self.spectral_bands:
            raise RuntimeError("backbone and CDS bank disagree on B")
        if len(self.structure_bank) and not self.spectral_normalization_fitted:
            raise RuntimeError(
                "stored CDS requires fitted Phase-0 spectral normalization"
            )
        return True

    @torch.no_grad()
    def fit_spectral_normalization(
        self,
        base_center_spectra: Tensor,
        *,
        overwrite: bool = False,
    ) -> None:
        if len(self.structure_bank):
            raise RuntimeError("spectral normalization cannot change after Phase 0")
        # A configured backbone normalizer and the physical normalizer may be
        # fitted independently. Complete the missing fit without changing an
        # existing coordinate system unless overwrite was explicitly requested.
        if overwrite or not self.backbone.spectral_normalization_fitted:
            self.backbone.fit_spectral_normalization(
                base_center_spectra, overwrite=overwrite
            )
        if overwrite or not bool(self.physical_spectral_extractor.fitted.item()):
            self.physical_spectral_extractor.fit(
                base_center_spectra, overwrite=overwrite
            )
        if not self.spectral_normalization_fitted:
            raise RuntimeError("spectral normalization was not fitted")

    def set_base_trainability(self) -> None:
        self.backbone.set_base_trainability()

    def set_incremental_trainability(self) -> None:
        self.validate_finalized_state()
        self.backbone.set_incremental_trainability()
        if not tuple(self.backbone.incremental_trainable_parameters()):
            raise RuntimeError("incremental representation has no plastic parameters")

    def extract_features_with_backbone(
        self,
        backbone: nn.Module,
        patch: Tensor,
        *,
        center_spectrum: Tensor,
        return_aux: bool = False,
    ) -> BackboneOutput:
        output = backbone(
            patch,
            center_spectrum=center_spectrum,
            return_aux=return_aux,
        )
        expected = (
            patch.size(0),
            self.spectral_token_count,
            self.spectral_token_dim,
        )
        if output.ordered_features.shape != expected or not bool(
            torch.isfinite(output.ordered_features).all()
        ):
            raise RuntimeError("backbone produced invalid ordered Q")
        return output

    def extract_features(
        self,
        patch: Tensor,
        *,
        center_spectrum: Tensor,
        return_aux: bool = False,
    ) -> BackboneOutput:
        if not self.spectral_normalization_fitted:
            raise RuntimeError(
                "spectral normalization must be fitted before feature extraction"
            )
        return self.extract_features_with_backbone(
            self.backbone,
            patch,
            center_spectrum=center_spectrum,
            return_aux=return_aux,
        )

    def physical_spectral_evidence(self, center_spectrum: Tensor) -> Tensor:
        if not self.spectral_normalization_fitted:
            raise RuntimeError(
                "spectral normalization must be fitted before physical evidence"
            )
        evidence = self.physical_spectral_extractor(center_spectrum).to(
            device=self.device, dtype=self.dtype
        )
        if evidence.shape[1:] != (self.spectral_bands, 1):
            raise RuntimeError("physical spectral evidence must have shape [N,B,1]")
        return evidence

    def _resolve_cds(
        self,
        class_ids: Sequence[int] | Tensor,
        cds: ClassDiscriminativeStructureState | None,
    ) -> ClassDiscriminativeStructureState | None:
        """Resolve CDS only for read-only diagnostics/continual constraints.

        The classifier logits do not depend on this object.  If all requested
        classes are already committed we expose their CDS so evaluation can report
        learned/physical support diagnostics alongside the cosine decision.
        """
        if cds is not None:
            return cds
        ids = tuple(int(v) for v in torch.as_tensor(class_ids).flatten().cpu().tolist())
        if not ids:
            raise ValueError("class_ids must be non-empty")
        committed = set(self.committed_class_ids)
        if all(class_id in committed for class_id in ids):
            return self.class_discriminative_structure(ids)
        return None

    def classify_features(
        self,
        ordered_features: Tensor,
        *,
        physical_spectral_evidence: Tensor | None = None,
        class_ids: Sequence[int] | Tensor,
        cds: ClassDiscriminativeStructureState | None = None,
        trainable_class_ids: Sequence[int] | Tensor | None = None,
    ) -> ClassificationOutput:
        resolved = self._resolve_cds(class_ids, cds)
        # Deployed inference is the single calibrated cosine head over Q.
        # Whole-CDS shapes/guards continual learning and remains diagnostic-only
        # at inference; it is not a second decision score.
        return self.classifier(
            ordered_features,
            class_ids=class_ids,
            trainable_class_ids=trainable_class_ids,
            physical_spectral_evidence=physical_spectral_evidence,
            cds=resolved,
        )

    @torch.no_grad()
    def estimate_physical_class_structure(
        self,
        raw_center_spectra: Tensor,
        labels: Tensor,
        *,
        class_ids: Sequence[int] | Tensor,
    ) -> PhysicalSpectralStructureBatch:
        evidence = self.physical_spectral_evidence(raw_center_spectra)
        return self.structure_bank.estimate_physical(
            evidence, labels, class_ids=class_ids
        )

    @torch.no_grad()
    def estimate_class_structure(
        self,
        ordered_features: Tensor,
        raw_center_spectra: Tensor | None,
        labels: Tensor,
        *,
        class_ids: Sequence[int] | Tensor,
        physical_structure: PhysicalSpectralStructureBatch | None = None,
    ) -> ClassStructureBatch:
        if physical_structure is None:
            if raw_center_spectra is None:
                raise ValueError(
                    "raw_center_spectra is required when physical_structure is absent"
                )
            evidence = self.physical_spectral_evidence(raw_center_spectra)
        else:
            evidence = None
        return self.structure_bank.estimate_classes(
            ordered_features,
            evidence,
            labels,
            class_ids=class_ids,
            physical_structure=physical_structure,
        )

    @torch.no_grad()
    def commit_class_structure(
        self,
        structure: ClassStructureBatch,
        *,
        ordered_features: Tensor,
        physical_evidence: Tensor,
        labels: Tensor,
    ) -> None:
        self.structure_bank.replace_all_with_moments(
            structure, ordered_features, physical_evidence, labels
        )
        self.validate_model_state()

    @torch.no_grad()
    def replace_class_structure(
        self,
        structure: ClassStructureBatch,
        *,
        ordered_features: Tensor,
        physical_evidence: Tensor,
        labels: Tensor,
    ) -> None:
        self.structure_bank.replace_all_with_moments(
            structure, ordered_features, physical_evidence, labels
        )
        self.validate_model_state()

    def class_discriminative_structure(
        self,
        class_ids: Sequence[int] | Tensor | None = None,
    ) -> ClassDiscriminativeStructureState:
        return self.structure_bank.build_cds(
            self.structure_bank.class_structures(class_ids)
        )

    def validate_finalized_state(self) -> bool:
        self.validate_model_state()
        if len(self.structure_bank) == 0:
            raise RuntimeError("finalized phase requires non-empty CDS")
        if not self.spectral_normalization_fitted:
            raise RuntimeError("finalized phase requires fitted spectral normalization")
        return True

    def forward(
        self,
        patch: Tensor,
        *,
        center_spectrum: Tensor,
        class_ids: Sequence[int] | Tensor,
        cds: ClassDiscriminativeStructureState | None = None,
        trainable_class_ids: Sequence[int] | Tensor | None = None,
        return_aux: bool = False,
    ) -> NECILOutput:
        backbone_output = self.extract_features(
            patch,
            center_spectrum=center_spectrum,
            return_aux=return_aux,
        )
        physical = self.physical_spectral_evidence(center_spectrum)
        classification = self.classify_features(
            backbone_output.ordered_features,
            physical_spectral_evidence=physical,
            class_ids=class_ids,
            cds=cds,
            trainable_class_ids=trainable_class_ids,
        )
        return NECILOutput(
            backbone=backbone_output,
            physical_spectral_evidence=physical,
            classification=classification,
        )


__all__ = ["NECILModel", "NECILOutput"]


# from __future__ import annotations
# from dataclasses import dataclass
# from typing import Any, Sequence

# import torch
# import torch.nn as nn

# from models.backbone import BackboneOutput, HSIMambaFeatureExtractor
# from models.classifier import ClassificationOutput, CosineClassificationHead
# from models.geometry_bank import (
#     ClassDiscriminativeStructureState,
#     ClassStructureBatch,
#     HSIClassDiscriminativeStructureBank,
#     PhysicalSpectralStructureBatch,
# )
# from models.spectral_evidence import PhysicalSpectralEvidence

# Tensor = torch.Tensor


# @dataclass(frozen=True)
# class NECILOutput:
#     backbone: BackboneOutput
#     physical_spectral_evidence: Tensor
#     classification: ClassificationOutput


# class NECILModel(nn.Module):

#     def __init__(self, args: Any) -> None:
#         super().__init__()
#         self.backbone = HSIMambaFeatureExtractor(args)
#         reference = next(self.backbone.parameters(), None)
#         if reference is None:
#             raise RuntimeError("backbone must contain parameters")
#         if reference.dtype not in (torch.float32, torch.float64):
#             raise TypeError("backbone parameters must use float32 or float64")

#         self.physical_spectral_extractor = PhysicalSpectralEvidence(
#             band_count=int(self.backbone.spectral_bands)
#         )
#         positions = self.backbone.spectral_encoder.band_positions.detach().cpu()
#         band_positions = (
#             positions
#             if positions.numel() == int(self.backbone.spectral_bands)
#             else None
#         )
#         self.structure_bank = HSIClassDiscriminativeStructureBank(
#             spectral_token_count=int(self.backbone.spectral_token_count),
#             spectral_token_dim=int(self.backbone.spectral_token_dim),
#             spectral_bands=int(self.backbone.spectral_bands),
#             spectral_band_positions=band_positions,
#             device=reference.device,
#             dtype=reference.dtype,
#         )

#         total_class_count = int(getattr(args, "class_count", 0))
#         if total_class_count < 2:
#             raise ValueError(
#                 "args.class_count must be set before NECILModel construction"
#             )
#         self.classifier = CosineClassificationHead(
#             spectral_token_count=int(self.backbone.spectral_token_count),
#             spectral_token_dim=int(self.backbone.spectral_token_dim),
#             total_class_count=total_class_count,
#         )
#         self.validate_model_state()

#     def _reference_parameter(self) -> Tensor:
#         parameter = next(self.backbone.parameters(), None)
#         if parameter is None:
#             raise RuntimeError("backbone has no parameters")
#         return parameter

#     @property
#     def device(self) -> torch.device:
#         return self._reference_parameter().device

#     @property
#     def dtype(self) -> torch.dtype:
#         return self._reference_parameter().dtype

#     @property
#     def spectral_token_count(self) -> int:
#         return int(self.backbone.spectral_token_count)

#     @property
#     def spectral_token_dim(self) -> int:
#         return int(self.backbone.spectral_token_dim)

#     @property
#     def spectral_bands(self) -> int:
#         return int(self.backbone.spectral_bands)

#     @property
#     def committed_class_ids(self) -> tuple[int, ...]:
#         return self.structure_bank.seen_class_ids

#     @property
#     def spectral_normalization_fitted(self) -> bool:
#         return bool(self.backbone.spectral_normalization_fitted) and bool(
#             self.physical_spectral_extractor.fitted.item()
#         )

#     def validate_model_state(self) -> bool:
#         if not isinstance(self.classifier, CosineClassificationHead):
#             raise RuntimeError("classifier must be CosineClassificationHead")
#         if not bool(torch.isfinite(self.classifier.logit_scale)):
#             raise RuntimeError("classifier logit scale is invalid")
#         if not isinstance(self.structure_bank, HSIClassDiscriminativeStructureBank):
#             raise RuntimeError(
#                 "structure_bank must be HSIClassDiscriminativeStructureBank"
#             )
#         self.structure_bank.validate_state()
#         if (
#             self.structure_bank.device != self.device
#             or self.structure_bank.dtype != self.dtype
#         ):
#             raise RuntimeError("backbone and CDS bank use different device/dtype")
#         if self.structure_bank.spectral_token_count != self.spectral_token_count:
#             raise RuntimeError("backbone and CDS bank disagree on K")
#         if self.structure_bank.spectral_token_dim != self.spectral_token_dim:
#             raise RuntimeError("backbone and CDS bank disagree on D")
#         if self.structure_bank.spectral_bands != self.spectral_bands:
#             raise RuntimeError("backbone and CDS bank disagree on B")
#         if len(self.structure_bank) and not self.spectral_normalization_fitted:
#             raise RuntimeError(
#                 "stored CDS requires fitted Phase-0 spectral normalization"
#             )
#         return True

#     @torch.no_grad()
#     def fit_spectral_normalization(
#         self,
#         base_center_spectra: Tensor,
#         *,
#         overwrite: bool = False,
#     ) -> None:
#         if len(self.structure_bank):
#             raise RuntimeError("spectral normalization cannot change after Phase 0")
#         self.backbone.fit_spectral_normalization(
#             base_center_spectra, overwrite=overwrite
#         )
#         self.physical_spectral_extractor.fit(base_center_spectra, overwrite=overwrite)
#         if not self.spectral_normalization_fitted:
#             raise RuntimeError("spectral normalization was not fitted")

#     def set_base_trainability(self) -> None:
#         self.backbone.set_base_trainability()

#     def set_incremental_trainability(self) -> None:
#         self.validate_finalized_state()
#         self.backbone.set_incremental_trainability()
#         if not tuple(self.backbone.incremental_trainable_parameters()):
#             raise RuntimeError("incremental representation has no plastic parameters")

#     def extract_features_with_backbone(
#         self,
#         backbone: nn.Module,
#         patch: Tensor,
#         *,
#         center_spectrum: Tensor,
#         return_aux: bool = False,
#     ) -> BackboneOutput:
#         output = backbone(
#             patch,
#             center_spectrum=center_spectrum,
#             return_aux=return_aux,
#         )
#         expected = (
#             patch.size(0),
#             self.spectral_token_count,
#             self.spectral_token_dim,
#         )
#         if output.ordered_features.shape != expected or not bool(
#             torch.isfinite(output.ordered_features).all()
#         ):
#             raise RuntimeError("backbone produced invalid ordered Q")
#         return output

#     def extract_features(
#         self,
#         patch: Tensor,
#         *,
#         center_spectrum: Tensor,
#         return_aux: bool = False,
#     ) -> BackboneOutput:
#         if not self.spectral_normalization_fitted:
#             raise RuntimeError(
#                 "spectral normalization must be fitted before feature extraction"
#             )
#         return self.extract_features_with_backbone(
#             self.backbone,
#             patch,
#             center_spectrum=center_spectrum,
#             return_aux=return_aux,
#         )

#     def physical_spectral_evidence(self, center_spectrum: Tensor) -> Tensor:
#         if not self.spectral_normalization_fitted:
#             raise RuntimeError(
#                 "spectral normalization must be fitted before physical evidence"
#             )
#         evidence = self.physical_spectral_extractor(center_spectrum).to(
#             device=self.device, dtype=self.dtype
#         )
#         if evidence.shape[1:] != (self.spectral_bands, 1):
#             raise RuntimeError("physical spectral evidence must have shape [N,B,1]")
#         return evidence

#     def _resolve_cds(
#         self,
#         class_ids: Sequence[int] | Tensor,
#         cds: ClassDiscriminativeStructureState | None,
#     ) -> ClassDiscriminativeStructureState | None:
#         """Resolve CDS only for read-only diagnostics/continual constraints.

#         The classifier logits do not depend on this object.  If all requested
#         classes are already committed we expose their CDS so evaluation can report
#         learned/physical support diagnostics alongside the cosine decision.
#         """
#         if cds is not None:
#             return cds
#         ids = tuple(int(v) for v in torch.as_tensor(class_ids).flatten().cpu().tolist())
#         if not ids:
#             raise ValueError("class_ids must be non-empty")
#         committed = set(self.committed_class_ids)
#         if all(class_id in committed for class_id in ids):
#             return self.class_discriminative_structure(ids)
#         return None

#     def classify_features(
#         self,
#         ordered_features: Tensor,
#         *,
#         physical_spectral_evidence: Tensor | None = None,
#         class_ids: Sequence[int] | Tensor,
#         cds: ClassDiscriminativeStructureState | None = None,
#         trainable_class_ids: Sequence[int] | Tensor | None = None,
#     ) -> ClassificationOutput:
#         resolved = self._resolve_cds(class_ids, cds)
#         # Deployed inference is the single calibrated cosine head over Q.
#         # Whole-CDS shapes/guards continual learning and remains diagnostic-only
#         # at inference; it is not a second decision score.
#         return self.classifier(
#             ordered_features,
#             class_ids=class_ids,
#             trainable_class_ids=trainable_class_ids,
#             physical_spectral_evidence=physical_spectral_evidence,
#             cds=resolved,
#         )

#     @torch.no_grad()
#     def estimate_physical_class_structure(
#         self,
#         raw_center_spectra: Tensor,
#         labels: Tensor,
#         *,
#         class_ids: Sequence[int] | Tensor,
#     ) -> PhysicalSpectralStructureBatch:
#         evidence = self.physical_spectral_evidence(raw_center_spectra)
#         return self.structure_bank.estimate_physical(
#             evidence, labels, class_ids=class_ids
#         )

#     @torch.no_grad()
#     def estimate_class_structure(
#         self,
#         ordered_features: Tensor,
#         raw_center_spectra: Tensor | None,
#         labels: Tensor,
#         *,
#         class_ids: Sequence[int] | Tensor,
#         physical_structure: PhysicalSpectralStructureBatch | None = None,
#     ) -> ClassStructureBatch:
#         if physical_structure is None:
#             if raw_center_spectra is None:
#                 raise ValueError(
#                     "raw_center_spectra is required when physical_structure is absent"
#                 )
#             evidence = self.physical_spectral_evidence(raw_center_spectra)
#         else:
#             evidence = None
#         return self.structure_bank.estimate_classes(
#             ordered_features,
#             evidence,
#             labels,
#             class_ids=class_ids,
#             physical_structure=physical_structure,
#         )

#     @torch.no_grad()
#     def commit_class_structure(
#         self,
#         structure: ClassStructureBatch,
#         *,
#         ordered_features: Tensor,
#         physical_evidence: Tensor,
#         labels: Tensor,
#     ) -> None:
#         self.structure_bank.replace_all_with_moments(
#             structure, ordered_features, physical_evidence, labels
#         )
#         self.validate_model_state()

#     @torch.no_grad()
#     def replace_class_structure(
#         self,
#         structure: ClassStructureBatch,
#         *,
#         ordered_features: Tensor,
#         physical_evidence: Tensor,
#         labels: Tensor,
#     ) -> None:
#         self.structure_bank.replace_all_with_moments(
#             structure, ordered_features, physical_evidence, labels
#         )
#         self.validate_model_state()

#     def class_discriminative_structure(
#         self,
#         class_ids: Sequence[int] | Tensor | None = None,
#     ) -> ClassDiscriminativeStructureState:
#         return self.structure_bank.build_cds(
#             self.structure_bank.class_structures(class_ids)
#         )

#     def validate_finalized_state(self) -> bool:
#         self.validate_model_state()
#         if len(self.structure_bank) == 0:
#             raise RuntimeError("finalized phase requires non-empty CDS")
#         if not self.spectral_normalization_fitted:
#             raise RuntimeError("finalized phase requires fitted spectral normalization")
#         return True

#     def forward(
#         self,
#         patch: Tensor,
#         *,
#         center_spectrum: Tensor,
#         class_ids: Sequence[int] | Tensor,
#         cds: ClassDiscriminativeStructureState | None = None,
#         trainable_class_ids: Sequence[int] | Tensor | None = None,
#         return_aux: bool = False,
#     ) -> NECILOutput:
#         backbone_output = self.extract_features(
#             patch,
#             center_spectrum=center_spectrum,
#             return_aux=return_aux,
#         )
#         physical = self.physical_spectral_evidence(center_spectrum)
#         classification = self.classify_features(
#             backbone_output.ordered_features,
#             physical_spectral_evidence=physical,
#             class_ids=class_ids,
#             cds=cds,
#             trainable_class_ids=trainable_class_ids,
#         )
#         return NECILOutput(
#             backbone=backbone_output,
#             physical_spectral_evidence=physical,
#             classification=classification,
#         )


# __all__ = ["NECILModel", "NECILOutput"]
