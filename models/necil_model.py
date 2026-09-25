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
    """NECIL-HSI model: one Q, one fixed P, one Whole-CDS bank, one head.

    Persistent method memory is {I_Q,V_Q,I_P,V_P}. Whole-CDS is the
    prototype-like historical class state used for transient historical support,
    intra/inter preservation and risk-aware separation. The only deployed
    classifier is the all-seen cosine head over Q.
    """

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
            raise ValueError("args.class_count must be set before NECILModel construction")
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
            raise RuntimeError("structure_bank must be HSIClassDiscriminativeStructureBank")
        self.structure_bank.validate_state()
        if self.structure_bank.device != self.device or self.structure_bank.dtype != self.dtype:
            raise RuntimeError("backbone and CDS bank use different device/dtype")
        if self.structure_bank.spectral_token_count != self.spectral_token_count:
            raise RuntimeError("backbone and CDS bank disagree on K")
        if self.structure_bank.spectral_token_dim != self.spectral_token_dim:
            raise RuntimeError("backbone and CDS bank disagree on D")
        if self.structure_bank.spectral_bands != self.spectral_bands:
            raise RuntimeError("backbone and CDS bank disagree on B")
        if len(self.structure_bank) and not self.spectral_normalization_fitted:
            raise RuntimeError("stored CDS requires fitted Phase-0 spectral normalization")
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
        self.backbone.fit_spectral_normalization(base_center_spectra, overwrite=overwrite)
        self.physical_spectral_extractor.fit(base_center_spectra, overwrite=overwrite)
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
            raise RuntimeError("spectral normalization must be fitted before feature extraction")
        return self.extract_features_with_backbone(
            self.backbone,
            patch,
            center_spectrum=center_spectrum,
            return_aux=return_aux,
        )

    def physical_spectral_evidence(self, center_spectrum: Tensor) -> Tensor:
        if not self.spectral_normalization_fitted:
            raise RuntimeError("spectral normalization must be fitted before physical evidence")
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
        # Deployed inference is cosine(Q,W). Whole-CDS shapes Q during continual
        # training and supplies read-only structural diagnostics at inference.
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
                raise ValueError("raw_center_spectra is required when physical_structure is absent")
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
    def commit_class_structure(self, structure: ClassStructureBatch) -> None:
        self.structure_bank.append_estimate(structure)

    @torch.no_grad()
    def replace_class_structure(self, structure: ClassStructureBatch) -> None:
        self.structure_bank.replace_all(structure)
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

# """NECIL-HSI model: one learned Q, one full-band physical P, one CDS bank.

# Q[K,D] is the only learned discriminative sample representation. P[B,3] is a
# deterministic phase-stable physical spectral structure built from the complete
# pre-PCA centre spectrum; B is dataset-derived and independent of K.

# Persistent class state is exactly
#     {I_Q[K,D], V_Q[K,K,D], I_P[B,3], V_P[B,B,3], N}.
# A single bias-free normalized cosine classifier supplies the all-seen decision
# rule from Q only. The *whole* HSI CDS constrains incremental learning: I_Q and
# V_Q define learned decision-region structure, while I_P and V_P define native
# full-band physical spectral confusability used to prioritize preservation. Physical
# spectral evidence is not concatenated with Q and is not a second classifier input.
# """

# from dataclasses import dataclass
# from numbers import Integral
# from typing import Any, Sequence

# import torch
# import torch.nn as nn

# from models.backbone import BackboneOutput, HSIMambaFeatureExtractor
# from models.classifier import ClassifierOutput, SpectralCDSClassifier
# from models.geometry_bank import (
#     ClassDiscriminativeStructureState,
#     PhysicalSpectralStructureBatch,
#     ClassStructureBatch,
#     HSIClassDiscriminativeStructureBank,
# )
# from models.spectral_evidence import OrderedPhysicalSpectralEvidence

# Tensor = torch.Tensor


# def _class_id_tuple(values: Sequence[int] | Tensor) -> tuple[int, ...]:
#     source = values.detach().cpu().tolist() if torch.is_tensor(values) else list(values)
#     result: list[int] = []
#     for value in source:
#         if isinstance(value, bool) or not isinstance(value, Integral) or int(value) < 0:
#             raise ValueError("class_ids must contain non-negative integers")
#         result.append(int(value))
#     output = tuple(result)
#     if not output or len(output) != len(set(output)):
#         raise ValueError("class_ids must be non-empty and unique")
#     return output


# def _physical_band_positions(backbone: HSIMambaFeatureExtractor) -> Tensor | None:
#     positions = backbone.spectral_encoder.band_positions.detach().cpu()
#     return positions if positions.numel() == int(backbone.spectral_bands) else None


# @dataclass(frozen=True)
# class NECILOutput:
#     backbone: BackboneOutput
#     spectral_evidence: Tensor
#     classification: ClassifierOutput


# class NECILModel(nn.Module):
#     """One plastic HSI representation, one physical spectral reference, one CDS."""

#     def __init__(self, args: Any) -> None:
#         super().__init__()
#         self.backbone = HSIMambaFeatureExtractor(args)
#         reference = next(self.backbone.parameters(), None)
#         if reference is None:
#             raise RuntimeError("backbone must contain parameters")
#         if reference.dtype not in (torch.float32, torch.float64):
#             raise TypeError("backbone parameters must use float32 or float64")

#         band_positions = _physical_band_positions(self.backbone)
#         self.spectral_evidence_extractor = OrderedPhysicalSpectralEvidence(
#             band_count=int(self.backbone.spectral_bands),
#             band_positions=band_positions,
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
#             raise ValueError("args.class_count must be set before NECILModel construction")
#         self.classifier = SpectralCDSClassifier(
#             spectral_token_count=int(self.backbone.spectral_token_count),
#             spectral_token_dim=int(self.backbone.spectral_token_dim),
#             total_class_count=total_class_count,
#         )
#         self.validate_model_state()

#     def _reference_parameter(self) -> Tensor:
#         reference = next(self.backbone.parameters(), None)
#         if reference is None:
#             raise RuntimeError("backbone has no parameters")
#         return reference

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
#             self.spectral_evidence_extractor.fitted.item()
#         )

#     def validate_model_state(self) -> bool:
#         if not isinstance(self.classifier, SpectralCDSClassifier):
#             raise RuntimeError("classifier must be SpectralCDSClassifier")
#         if self.classifier.parameter_count <= 0:
#             raise RuntimeError("cosine classifier must contain trainable class rows")
#         scale = self.classifier.logit_scale_value
#         if not bool(torch.isfinite(scale)) or not bool(scale > 0):
#             raise RuntimeError("shared cosine classifier scale is invalid")
#         if not isinstance(self.structure_bank, HSIClassDiscriminativeStructureBank):
#             raise RuntimeError("structure_bank must be HSIClassDiscriminativeStructureBank")
#         self.structure_bank.validate_state()
#         if self.structure_bank.device != self.device or self.structure_bank.dtype != self.dtype:
#             raise RuntimeError("backbone and structure bank use different device/dtype")
#         if self.structure_bank.spectral_token_count != self.spectral_token_count:
#             raise RuntimeError("backbone and bank disagree on K")
#         if self.structure_bank.spectral_token_dim != self.spectral_token_dim:
#             raise RuntimeError("backbone and bank disagree on D")
#         if self.structure_bank.spectral_bands != self.spectral_bands:
#             raise RuntimeError("backbone and bank disagree on native band count B")
#         if len(self.structure_bank) and not self.spectral_normalization_fitted:
#             raise RuntimeError("stored CDS requires fitted Phase-0 spectral normalization")
#         return True

#     @torch.no_grad()
#     def fit_spectral_normalization(self, base_center_spectra: Tensor, *, overwrite: bool = False) -> None:
#         if len(self.structure_bank):
#             raise RuntimeError("spectral normalization cannot change after Phase 0")
#         self.backbone.fit_spectral_normalization(base_center_spectra, overwrite=overwrite)
#         self.spectral_evidence_extractor.fit(base_center_spectra, overwrite=overwrite)
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
#         output = backbone(patch, center_spectrum=center_spectrum, return_aux=return_aux)
#         q = output.ordered_features
#         expected = (patch.size(0), self.spectral_token_count, self.spectral_token_dim)
#         if q.shape != expected or not bool(torch.isfinite(q).all()):
#             raise RuntimeError("backbone produced invalid ordered features")
#         return output

#     def extract_features(
#         self,
#         patch: Tensor,
#         *,
#         center_spectrum: Tensor,
#         return_aux: bool = False,
#     ) -> BackboneOutput:
#         if not self.spectral_normalization_fitted:
#             raise RuntimeError("spectral normalization must be fitted before feature extraction")
#         return self.extract_features_with_backbone(
#             self.backbone,
#             patch,
#             center_spectrum=center_spectrum,
#             return_aux=return_aux,
#         )

#     def spectral_evidence(self, center_spectrum: Tensor) -> Tensor:
#         if not self.spectral_normalization_fitted:
#             raise RuntimeError("spectral normalization must be fitted before spectral evidence")
#         evidence = self.spectral_evidence_extractor(center_spectrum)
#         if evidence.device != self.device or evidence.dtype != self.dtype:
#             evidence = evidence.to(device=self.device, dtype=self.dtype)
#         if evidence.ndim != 3 or tuple(evidence.shape[1:]) != (self.spectral_bands, 3):
#             raise RuntimeError("full-band physical spectral evidence has invalid shape")
#         return evidence

#     def _resolve_cds(
#         self,
#         *,
#         cds: ClassDiscriminativeStructureState | None,
#         structures: ClassStructureBatch | None,
#         class_ids: Sequence[int] | Tensor | None,
#     ) -> ClassDiscriminativeStructureState:
#         if cds is not None:
#             if structures is not None or class_ids is not None:
#                 raise ValueError("cds cannot be combined with structures/class_ids")
#             return cds
#         if structures is not None:
#             if class_ids is not None:
#                 requested = _class_id_tuple(class_ids)
#                 actual = tuple(int(v) for v in structures.class_ids.detach().cpu().tolist())
#                 if requested != actual:
#                     raise ValueError("class_ids must match supplied structures")
#             return self.structure_bank.build_cds(structures)
#         if len(self.structure_bank) == 0:
#             raise RuntimeError("classification requires a working CDS")
#         return self.structure_bank.build_cds(self.structure_bank.class_structures(class_ids))

#     def build_class_discriminative_structure(
#         self,
#         structures: ClassStructureBatch,
#     ) -> ClassDiscriminativeStructureState:
#         return self.structure_bank.build_cds(structures)

#     def classify_features(
#         self,
#         ordered_features: Tensor,
#         *,
#         cds: ClassDiscriminativeStructureState | None = None,
#         structures: ClassStructureBatch | None = None,
#         class_ids: Sequence[int] | Tensor | None = None,
#     ) -> ClassifierOutput:
#         q = torch.as_tensor(ordered_features)
#         if q.ndim != 3 or tuple(q.shape[1:]) != (self.spectral_token_count, self.spectral_token_dim):
#             raise ValueError("ordered_features have invalid shape")
#         resolved = self._resolve_cds(cds=cds, structures=structures, class_ids=class_ids)
#         return self.classifier(q, cds=resolved)

#     @torch.no_grad()
#     def estimate_physical_class_structure(
#         self,
#         raw_center_spectra: Tensor,
#         labels: Tensor,
#         *,
#         class_ids: Sequence[int] | Tensor,
#     ) -> PhysicalSpectralStructureBatch:
#         p = self.spectral_evidence(raw_center_spectra)
#         return self.structure_bank.estimate_physical(p, labels, class_ids=class_ids)

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
#         q = torch.as_tensor(ordered_features)
#         p = None if physical_structure is not None else self.spectral_evidence(raw_center_spectra)
#         return self.structure_bank.estimate_classes(
#             q,
#             p,
#             labels,
#             class_ids=class_ids,
#             physical_structure=physical_structure,
#         )

#     @torch.no_grad()
#     def commit_class_structure(self, estimated: ClassStructureBatch) -> None:
#         self.structure_bank.append_estimate(estimated)

#     @torch.no_grad()
#     def replace_class_structure(self, structures: ClassStructureBatch) -> None:
#         self.structure_bank.replace_all(structures)
#         self.validate_model_state()

#     def class_discriminative_structure(
#         self,
#         class_ids: Sequence[int] | Tensor | None = None,
#     ) -> ClassDiscriminativeStructureState:
#         return self.structure_bank.build_cds(self.structure_bank.class_structures(class_ids))

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
#         cds: ClassDiscriminativeStructureState | None = None,
#         structures: ClassStructureBatch | None = None,
#         class_ids: Sequence[int] | Tensor | None = None,
#         return_aux: bool = False,
#     ) -> NECILOutput:
#         backbone_output = self.extract_features(
#             patch,
#             center_spectrum=center_spectrum,
#             return_aux=return_aux,
#         )
#         p = self.spectral_evidence(center_spectrum)
#         resolved = self._resolve_cds(cds=cds, structures=structures, class_ids=class_ids)
#         classification = self.classifier(backbone_output.ordered_features, cds=resolved)
#         return NECILOutput(
#             backbone=backbone_output,
#             spectral_evidence=p,
#             classification=classification,
#         )


# __all__ = ["NECILModel", "NECILOutput"]

