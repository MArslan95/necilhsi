from __future__ import annotations

"""Clean NECIL-HSI model integration.

Each component has one job:

    HSIMambaFeatureExtractor
        extracts ordered spectral-spatial HSI features Q_raw(x) in R^{K x D}.

    structural_coordinates()
        removes the already-observed arbitrary global feature-scale degree of
        freedom with one scalar RMS normalization per sample:

            Q(x) = Q_raw(x) / RMS(Q_raw(x)).

        It does not learn class information and it does not classify.

    PersistentCosineClassifier
        performs classification only from Q(x).

    HSIClassDiscriminativeGeometry
        stores historical class statistics only:

            M_c = {I_c, S_c, N_c}.

Geometry is deliberately absent from the classifier forward path.  It is
retained for the later non-exemplar preservation mechanism and for read-only
structural diagnostics.
"""

from dataclasses import dataclass, replace
import math
from typing import Any, Sequence

import torch
import torch.nn as nn

from models.backbone import BackboneOutput, HSIMambaFeatureExtractor
from models.classifier import ClassifierOutput, PersistentCosineClassifier
from models.geometry_bank import HSIClassDiscriminativeGeometry

Tensor = torch.Tensor


@dataclass(frozen=True)
class NECILOutput:
    backbone: BackboneOutput
    classification: ClassifierOutput


class NECILModel(nn.Module):
    """HSI feature extractor + classifier + independent statistical memory."""

    def __init__(self, args: Any) -> None:
        super().__init__()

        self.backbone = HSIMambaFeatureExtractor(args)
        for name in ("spectral_token_count", "spectral_token_dim"):
            if not hasattr(self.backbone, name):
                raise RuntimeError(f"backbone must expose {name}")

        reference = next(self.backbone.parameters(), None)
        if reference is None:
            raise RuntimeError("backbone must contain trainable parameters")
        if reference.dtype not in (torch.float32, torch.float64):
            raise TypeError("backbone parameters must use float32 or float64")

        self.classifier = PersistentCosineClassifier(
            spectral_token_count=int(self.backbone.spectral_token_count),
            spectral_token_dim=int(self.backbone.spectral_token_dim),
        )
        self.geometry_bank = HSIClassDiscriminativeGeometry(
            spectral_token_count=int(self.backbone.spectral_token_count),
            spectral_token_dim=int(self.backbone.spectral_token_dim),
            device=reference.device,
            dtype=reference.dtype,
        )
        self.validate_model_state()

    # ------------------------------------------------------------------
    # Basic model contract
    # ------------------------------------------------------------------

    def _backbone_reference(self) -> Tensor:
        reference = next(self.backbone.parameters(), None)
        if reference is None:
            raise RuntimeError("backbone has no parameters")
        return reference

    @property
    def device(self) -> torch.device:
        return self._backbone_reference().device

    @property
    def dtype(self) -> torch.dtype:
        dtype = self._backbone_reference().dtype
        if dtype not in (torch.float32, torch.float64):
            raise RuntimeError("backbone parameters must use float32 or float64")
        return dtype

    @property
    def spectral_token_count(self) -> int:
        return int(self.backbone.spectral_token_count)

    @property
    def spectral_token_dim(self) -> int:
        return int(self.backbone.spectral_token_dim)

    @property
    def classifier_class_ids(self) -> tuple[int, ...]:
        return tuple(self.classifier.seen_class_ids)

    @property
    def committed_class_ids(self) -> tuple[int, ...]:
        return tuple(self.geometry_bank.seen_class_ids)

    @property
    def spectral_normalization_fitted(self) -> bool:
        return bool(self.backbone.spectral_normalization_fitted)

    def _validate_geometry_contract(
        self,
        geometry_bank: HSIClassDiscriminativeGeometry,
    ) -> None:
        if not isinstance(geometry_bank, HSIClassDiscriminativeGeometry):
            raise TypeError(
                "geometry_bank must be HSIClassDiscriminativeGeometry"
            )
        if geometry_bank.spectral_token_count != self.spectral_token_count:
            raise RuntimeError(
                "backbone and geometry bank disagree on spectral_token_count"
            )
        if geometry_bank.spectral_token_dim != self.spectral_token_dim:
            raise RuntimeError(
                "backbone and geometry bank disagree on spectral_token_dim"
            )

    def validate_model_state(self) -> bool:
        if not isinstance(self.classifier, PersistentCosineClassifier):
            raise RuntimeError("classifier must be PersistentCosineClassifier")
        self.classifier.validate_state()
        self._validate_geometry_contract(self.geometry_bank)
        self.geometry_bank.validate_state()

        if self.classifier.spectral_token_count != self.spectral_token_count:
            raise RuntimeError(
                "backbone and classifier disagree on spectral_token_count"
            )
        if self.classifier.spectral_token_dim != self.spectral_token_dim:
            raise RuntimeError(
                "backbone and classifier disagree on spectral_token_dim"
            )
        if self.classifier.weight.device != self.device:
            raise RuntimeError("classifier and backbone use different devices")
        if self.classifier.weight.dtype != self.dtype:
            raise RuntimeError("classifier and backbone use different dtypes")
        if self.geometry_bank.device != self.device:
            raise RuntimeError("geometry bank and backbone use different devices")
        if self.geometry_bank.dtype != self.dtype:
            raise RuntimeError("geometry bank and backbone use different dtypes")

        # A committed historical class must always have a decision row.  During
        # an open incremental phase the classifier may legitimately contain new
        # rows whose final geometry has not yet been committed.
        committed = set(self.committed_class_ids)
        classified = set(self.classifier_class_ids)
        if not committed.issubset(classified):
            missing = sorted(committed - classified)
            raise RuntimeError(
                "committed geometry lacks classifier rows for classes "
                f"{missing}"
            )

        if len(self.geometry_bank) > 0 and not self.spectral_normalization_fitted:
            raise RuntimeError(
                "stored class geometry requires fitted spectral normalization"
            )
        return True

    def load_state_dict(
        self,
        state_dict: Any,
        strict: bool = True,
        **kwargs: Any,
    ) -> Any:
        result = super().load_state_dict(
            state_dict,
            strict=strict,
            **kwargs,
        )
        self.validate_model_state()
        return result

    # ------------------------------------------------------------------
    # Fixed HSI spectral normalization owned by the backbone
    # ------------------------------------------------------------------

    @torch.no_grad()
    def set_spectral_normalization(
        self,
        mean: Tensor | Sequence[float],
        scale: Tensor | Sequence[float],
        *,
        overwrite: bool = False,
    ) -> None:
        if len(self.geometry_bank) != 0:
            raise RuntimeError(
                "spectral normalization cannot change after historical geometry "
                "has been committed"
            )
        self.backbone.set_spectral_normalization(
            mean,
            scale,
            overwrite=overwrite,
        )
        if not self.spectral_normalization_fitted:
            raise RuntimeError("spectral normalization did not enter fitted state")

    @torch.no_grad()
    def fit_spectral_normalization(
        self,
        base_center_spectra: Tensor,
        *,
        overwrite: bool = False,
    ) -> None:
        if len(self.geometry_bank) != 0:
            raise RuntimeError(
                "spectral normalization cannot change after historical geometry "
                "has been committed"
            )
        self.backbone.fit_spectral_normalization(
            base_center_spectra,
            overwrite=overwrite,
        )
        if not self.spectral_normalization_fitted:
            raise RuntimeError("spectral normalization did not enter fitted state")

    # ------------------------------------------------------------------
    # Canonical structural coordinate
    # ------------------------------------------------------------------

    @staticmethod
    def structural_coordinates(ordered_features_raw: Tensor) -> Tensor:
        """Remove one global positive scale degree of freedom per HSI sample.

        The normalization uses a single scalar for the whole K x D tensor.  It
        therefore preserves spectral-region order and all within-sample entry
        ratios.  The only floor is the dtype's numerical tiny value; there is
        no research threshold or tunable normalization constant.
        """
        features = torch.as_tensor(ordered_features_raw)
        if features.ndim != 3 or features.size(0) == 0:
            raise ValueError("ordered_features_raw must be non-empty [N,K,D]")
        if not torch.is_floating_point(features) or features.is_complex():
            raise TypeError("ordered_features_raw must be real floating point")
        if not bool(torch.isfinite(features).all()):
            raise ValueError("ordered_features_raw contains NaN or Inf")

        flat = features.reshape(features.size(0), -1)
        if flat.size(1) == 0:
            raise ValueError("ordered_features_raw contains no feature entries")
        rms = torch.linalg.vector_norm(flat, ord=2, dim=1, keepdim=True)
        rms = rms / math.sqrt(float(flat.size(1)))
        floor = torch.as_tensor(
            torch.finfo(features.dtype).tiny,
            device=features.device,
            dtype=features.dtype,
        )
        normalized = features / rms.clamp_min(floor).view(-1, 1, 1)
        if not bool(torch.isfinite(normalized).all()):
            raise RuntimeError("structural RMS normalization produced NaN or Inf")
        return normalized

    def extract_features(
        self,
        patch: Tensor,
        *,
        center_spectrum: Tensor,
        return_aux: bool = False,
    ) -> BackboneOutput:
        """Return the one canonical Q(x) used by classifier and geometry."""
        if not self.spectral_normalization_fitted:
            raise RuntimeError(
                "spectral normalization must be fitted before feature extraction"
            )

        raw_output = self.backbone(
            patch,
            center_spectrum=center_spectrum,
            return_aux=return_aux,
        )
        raw = raw_output.ordered_features
        expected = (
            patch.size(0),
            self.spectral_token_count,
            self.spectral_token_dim,
        )
        if tuple(raw.shape) != expected:
            raise RuntimeError(
                "backbone ordered_features have shape "
                f"{tuple(raw.shape)}, expected {expected}"
            )
        if raw.device != self.device:
            raise RuntimeError("backbone output and model use different devices")
        if raw.dtype != self.dtype:
            raise RuntimeError("backbone output and model use different dtypes")

        canonical = self.structural_coordinates(raw)
        return replace(raw_output, ordered_features=canonical)

    # ------------------------------------------------------------------
    # Classifier: decision only
    # ------------------------------------------------------------------

    def add_classifier_classes(
        self,
        class_ids: Sequence[int] | Tensor,
    ) -> None:
        """Append new decision rows before constructing that phase optimizer."""
        self.classifier.add_classes(class_ids)
        self.validate_model_state()

    def classify_features(
        self,
        ordered_features: Tensor,
        *,
        class_ids: Sequence[int] | Tensor | None = None,
    ) -> ClassifierOutput:
        return self.classifier(
            ordered_features,
            class_ids=class_ids,
        )

    # ------------------------------------------------------------------
    # Geometry: historical statistics only
    # ------------------------------------------------------------------

    @torch.no_grad()
    def add_classes(
        self,
        ordered_features: Tensor,
        labels: Tensor,
        *,
        class_ids: Sequence[int] | Tensor,
    ) -> None:
        """Commit {I_c,S_c,N_c} from canonical TRAIN Q(x)."""
        if not self.spectral_normalization_fitted:
            raise RuntimeError(
                "spectral normalization must be fitted before geometry commit"
            )

        requested = tuple(
            int(v)
            for v in (
                class_ids.detach().cpu().tolist()
                if torch.is_tensor(class_ids)
                else list(class_ids)
            )
        )
        classifier_ids = set(self.classifier_class_ids)
        missing = [class_id for class_id in requested if class_id not in classifier_ids]
        if missing:
            raise RuntimeError(
                "cannot commit geometry before classifier rows exist for classes "
                f"{missing}"
            )

        features = torch.as_tensor(ordered_features)
        expected = (
            int(torch.as_tensor(labels).numel()),
            self.spectral_token_count,
            self.spectral_token_dim,
        )
        if tuple(features.shape) != expected:
            raise ValueError(
                f"geometry features must have shape {expected}, got "
                f"{tuple(features.shape)}"
            )

        # `extract_features()` already returns canonical Q.  This invariant
        # catches accidental raw-Q commits rather than silently renormalizing a
        # second time and hiding a caller bug.
        flat = features.reshape(features.size(0), -1)
        rms = torch.linalg.vector_norm(flat, ord=2, dim=1) / math.sqrt(
            float(flat.size(1))
        )
        tolerance = 2e-5 if features.dtype == torch.float32 else 1e-10
        if not bool(
            torch.allclose(
                rms,
                torch.ones_like(rms),
                rtol=tolerance,
                atol=tolerance,
            )
        ):
            raise RuntimeError(
                "geometry commit requires canonical global-RMS-normalized Q(x)"
            )

        self.geometry_bank.add_classes(
            features,
            labels,
            class_ids=class_ids,
        )
        self.validate_model_state()

    # ------------------------------------------------------------------
    # End-to-end prediction path
    # ------------------------------------------------------------------

    def forward(
        self,
        patch: Tensor,
        *,
        center_spectrum: Tensor,
        class_ids: Sequence[int] | Tensor | None = None,
        return_aux: bool = False,
    ) -> NECILOutput:
        backbone_output = self.extract_features(
            patch,
            center_spectrum=center_spectrum,
            return_aux=return_aux,
        )
        classification = self.classify_features(
            backbone_output.ordered_features,
            class_ids=class_ids,
        )
        return NECILOutput(
            backbone=backbone_output,
            classification=classification,
        )


__all__ = ["NECILModel", "NECILOutput"]







# from __future__ import annotations

# """NECIL-HSI model integration for feature-resolved class structure.

# The model connects one representation contract:

#     HSI sample
#         -> HSIMambaFeatureExtractor
#         -> Q_raw(x) in R^{K x D}
#         -> global-RMS-normalized Q(x) in R^{K x D}
#         -> HSIClassDiscriminativeGeometry
#         -> GeometryClassifier using derived A_c/C_c views
#         -> prediction.

# The persistent class state is

#     M_c = {I_c,S_c,N_c},

# where

#     Q(x) = Q_raw(x) / RMS(Q_raw(x)),

#     I_c = E[Q | c],

#     S_c[k,l,d]
#         = E[(Q[k,d]-I_c[k,d])(Q[l,d]-I_c[l,d]) | c].

# A_c=diag_K(S_c) and aggregate C_c=mean_D(S_c) are derived diagnostic views,
# not separately stored classifier state.

# During phase-0 training the model-owned persistent bank remains empty.  The
# trainer estimates a detached temporary {I_c,S_c,N_c} bank from complete
# phase-0 TRAIN normalized Q(x) at the current backbone state.  Mini-batches are
# classified against that fixed temporary bank with the same deployed energy

#     E_c = E_identity + E_feature + E_cross,

# where A_c=diag_K(S_c) and C_c=mean_D(S_c) are derived from the temporary bank.

# After each epoch the temporary bank is refreshed from the updated backbone for
# TRAIN/VAL diagnostics.  After the final requested epoch, complete phase-0
# TRAIN is encoded again and final {I,S,N} is committed once to the persistent bank.

# The integration layer creates no second representation, auxiliary classifier,
# pairwise boundary bank, replay state, Gaussian state, calibration state, or
# incremental preservation mechanism.
# """

# from dataclasses import dataclass
# import math
# from typing import Any, Sequence

# import torch
# import torch.nn as nn

# from models.backbone import (
#     BackboneOutput,
#     HSIMambaFeatureExtractor,
# )
# from models.classifier import (
#     GeometryClassifier,
#     GeometryClassifierOutput,
# )
# from models.geometry_bank import (
#     HSIClassDiscriminativeGeometry,
# )

# Tensor = torch.Tensor


# @dataclass(frozen=True)
# class NECILOutput:
#     """End-to-end model output."""

#     backbone: BackboneOutput
#     classification: GeometryClassifierOutput


# class NECILModel(nn.Module):
#     """HSI feature extractor + persistent class structure + equal-rule classifier."""

#     def __init__(
#         self,
#         args: Any,
#     ) -> None:
#         super().__init__()

#         self.backbone = HSIMambaFeatureExtractor(
#             args
#         )

#         if not hasattr(
#             self.backbone,
#             "spectral_token_count",
#         ):
#             raise RuntimeError(
#                 "backbone must expose spectral_token_count"
#             )

#         if not hasattr(
#             self.backbone,
#             "spectral_token_dim",
#         ):
#             raise RuntimeError(
#                 "backbone must expose spectral_token_dim"
#             )

#         reference = next(
#             self.backbone.parameters(),
#             None,
#         )

#         if reference is None:
#             raise RuntimeError(
#                 "backbone must contain trainable parameters"
#             )

#         if reference.dtype not in (
#             torch.float32,
#             torch.float64,
#         ):
#             raise TypeError(
#                 "backbone parameters must use float32 or float64"
#             )

#         self.geometry_bank = (
#             HSIClassDiscriminativeGeometry(
#                 spectral_token_count=(
#                     self.backbone
#                     .spectral_token_count
#                 ),
#                 spectral_token_dim=(
#                     self.backbone
#                     .spectral_token_dim
#                 ),
#                 device=reference.device,
#                 dtype=reference.dtype,
#             )
#         )

#         self.classifier = GeometryClassifier()

#         self.validate_model_state()

#     # ------------------------------------------------------------------
#     # Model contract
#     # ------------------------------------------------------------------

#     def _backbone_reference(
#         self,
#     ) -> Tensor:
#         reference = next(
#             self.backbone.parameters(),
#             None,
#         )
#         if reference is None:
#             raise RuntimeError(
#                 "backbone has no parameters"
#             )
#         return reference

#     @property
#     def device(
#         self,
#     ) -> torch.device:
#         return self._backbone_reference().device

#     @property
#     def dtype(
#         self,
#     ) -> torch.dtype:
#         dtype = self._backbone_reference().dtype
#         if dtype not in (
#             torch.float32,
#             torch.float64,
#         ):
#             raise RuntimeError(
#                 "backbone parameters must use float32 or float64"
#             )
#         return dtype

#     @property
#     def spectral_token_count(
#         self,
#     ) -> int:
#         return int(
#             self.backbone
#             .spectral_token_count
#         )

#     @property
#     def spectral_token_dim(
#         self,
#     ) -> int:
#         return int(
#             self.backbone
#             .spectral_token_dim
#         )

#     @property
#     def committed_class_ids(
#         self,
#     ) -> tuple[int, ...]:
#         return tuple(
#             self.geometry_bank
#             .seen_class_ids
#         )

#     @property
#     def spectral_normalization_fitted(
#         self,
#     ) -> bool:
#         return bool(
#             self.backbone
#             .spectral_normalization_fitted
#         )

#     def _validate_geometry_contract(
#         self,
#         geometry_bank: HSIClassDiscriminativeGeometry,
#     ) -> None:
#         if not isinstance(
#             geometry_bank,
#             HSIClassDiscriminativeGeometry,
#         ):
#             raise TypeError(
#                 "geometry_bank must be "
#                 "HSIClassDiscriminativeGeometry"
#             )

#         geometry_bank.validate_state()

#         if (
#             geometry_bank.device
#             != self.device
#         ):
#             raise RuntimeError(
#                 "backbone and geometry bank use different devices"
#             )

#         if (
#             geometry_bank.dtype
#             != self.dtype
#         ):
#             raise RuntimeError(
#                 "backbone and geometry bank use different dtypes"
#             )

#         if (
#             geometry_bank
#             .spectral_token_count
#             != self.spectral_token_count
#         ):
#             raise RuntimeError(
#                 "backbone and geometry bank disagree on "
#                 "spectral_token_count"
#             )

#         if (
#             geometry_bank
#             .spectral_token_dim
#             != self.spectral_token_dim
#         ):
#             raise RuntimeError(
#                 "backbone and geometry bank disagree on "
#                 "spectral_token_dim"
#             )

#     def validate_model_state(
#         self,
#     ) -> bool:
#         self._validate_geometry_contract(
#             self.geometry_bank
#         )

#         if not isinstance(
#             self.classifier,
#             GeometryClassifier,
#         ):
#             raise RuntimeError(
#                 "classifier must be GeometryClassifier"
#             )

#         if any(
#             True
#             for _ in self.classifier.parameters()
#         ):
#             raise RuntimeError(
#                 "GeometryClassifier must remain parameter-free"
#             )

#         if (
#             len(self.geometry_bank) > 0
#             and not self.spectral_normalization_fitted
#         ):
#             raise RuntimeError(
#                 "stored class structure requires fitted "
#                 "spectral normalization"
#             )

#         return True

#     def load_state_dict(
#         self,
#         state_dict: Any,
#         strict: bool = True,
#         **kwargs: Any,
#     ) -> Any:
#         result = super().load_state_dict(
#             state_dict,
#             strict=strict,
#             **kwargs,
#         )

#         self.validate_model_state()

#         return result

#     # ------------------------------------------------------------------
#     # Fixed spectral input normalization
#     # ------------------------------------------------------------------

#     @torch.no_grad()
#     def set_spectral_normalization(
#         self,
#         mean: Tensor | Sequence[float],
#         scale: Tensor | Sequence[float],
#         *,
#         overwrite: bool = False,
#     ) -> None:
#         """Set raw-band normalization before persistent class structure exists."""
#         if len(
#             self.geometry_bank
#         ) != 0:
#             raise RuntimeError(
#                 "spectral normalization cannot change after "
#                 "class geometry has been committed"
#             )

#         self.backbone.set_spectral_normalization(
#             mean,
#             scale,
#             overwrite=overwrite,
#         )

#         if not self.spectral_normalization_fitted:
#             raise RuntimeError(
#                 "spectral normalization did not enter fitted state"
#             )

#     @torch.no_grad()
#     def fit_spectral_normalization(
#         self,
#         base_center_spectra: Tensor,
#         *,
#         overwrite: bool = False,
#     ) -> None:
#         """Fit raw-band normalization before persistent class structure exists."""
#         if len(
#             self.geometry_bank
#         ) != 0:
#             raise RuntimeError(
#                 "spectral normalization cannot change after "
#                 "class geometry has been committed"
#             )

#         self.backbone.fit_spectral_normalization(
#             base_center_spectra,
#             overwrite=overwrite,
#         )

#         if not self.spectral_normalization_fitted:
#             raise RuntimeError(
#                 "spectral normalization did not enter fitted state"
#             )

#     # ------------------------------------------------------------------
#     # HSI feature extraction
#     # ------------------------------------------------------------------

#     def extract_features(
#         self,
#         patch: Tensor,
#         *,
#         center_spectrum: Tensor,
#         return_aux: bool = False,
#     ) -> BackboneOutput:
#         """Return the raw backbone representation Q_raw(x)."""
#         if not self.spectral_normalization_fitted:
#             raise RuntimeError(
#                 "spectral normalization must be fitted "
#                 "before feature extraction"
#             )

#         output = self.backbone(
#             patch,
#             center_spectrum=center_spectrum,
#             return_aux=return_aux,
#         )

#         ordered_features = (
#             output.ordered_features
#         )

#         expected_shape = (
#             patch.size(0),
#             self.spectral_token_count,
#             self.spectral_token_dim,
#         )

#         if (
#             ordered_features.shape
#             != expected_shape
#         ):
#             raise RuntimeError(
#                 "backbone ordered_features have shape "
#                 f"{tuple(ordered_features.shape)}, "
#                 f"expected {expected_shape}"
#             )

#         if (
#             ordered_features.device
#             != self.device
#         ):
#             raise RuntimeError(
#                 "backbone output and model use different devices"
#             )

#         if (
#             ordered_features.dtype
#             != self.dtype
#         ):
#             raise RuntimeError(
#                 "backbone output and model use different dtypes"
#             )

#         if not bool(
#             torch.isfinite(
#                 ordered_features
#             ).all()
#         ):
#             raise RuntimeError(
#                 "backbone ordered_features contain NaN or Inf"
#             )

#         return output

#     def normalize_ordered_features(
#         self,
#         ordered_features: Tensor,
#     ) -> Tensor:
#         """Remove only the arbitrary global feature magnitude per sample.

#         The normalization uses one scalar RMS over the complete ordered [K,D]
#         representation.  It therefore preserves every relative spectral-region
#         and feature-channel value while making the class-structure coordinate
#         system invariant to Q_raw -> alpha * Q_raw for alpha > 0.
#         """
#         if not torch.is_tensor(ordered_features):
#             raise TypeError("ordered_features must be a tensor")

#         expected_shape = (
#             ordered_features.size(0) if ordered_features.ndim >= 1 else 0,
#             self.spectral_token_count,
#             self.spectral_token_dim,
#         )
#         if ordered_features.ndim != 3 or tuple(ordered_features.shape) != expected_shape:
#             raise RuntimeError(
#                 "ordered_features must have shape [N,K,D]; "
#                 f"got {tuple(ordered_features.shape)}, expected {expected_shape}"
#             )
#         if ordered_features.size(0) == 0:
#             raise RuntimeError("ordered_features cannot be empty")
#         if ordered_features.device != self.device:
#             raise RuntimeError("ordered_features are on the wrong device")
#         if ordered_features.dtype != self.dtype:
#             raise RuntimeError("ordered_features use the wrong dtype")
#         if not bool(torch.isfinite(ordered_features).all()):
#             raise RuntimeError("ordered_features contain NaN or Inf")

#         flat = ordered_features.reshape(ordered_features.size(0), -1)
#         rms = torch.linalg.vector_norm(flat, ord=2, dim=1) / math.sqrt(float(flat.size(1)))
#         if not bool(torch.isfinite(rms).all()):
#             raise RuntimeError("ordered-feature RMS contains NaN or Inf")
#         if bool(rms.le(0.0).any()):
#             raise RuntimeError(
#                 "global RMS normalization is undefined for an all-zero ordered feature"
#             )

#         normalized = ordered_features / rms.view(-1, 1, 1)
#         if not bool(torch.isfinite(normalized).all()):
#             raise RuntimeError("normalized ordered features contain NaN or Inf")
#         return normalized

#     # ------------------------------------------------------------------
#     # Persistent class structure
#     # ------------------------------------------------------------------

#     @torch.no_grad()
#     def add_classes(
#         self,
#         ordered_features: Tensor,
#         labels: Tensor,
#         *,
#         class_ids: Sequence[int] | Tensor,
#     ) -> None:
#         """Commit class structure in the normalized Q(x) coordinate system.

#         The trainer controls which data split and backbone state produced
#         ``ordered_features``.  This method only performs persistent storage.
#         """
#         if not self.spectral_normalization_fitted:
#             raise RuntimeError(
#                 "spectral normalization must be fitted "
#                 "before class structure is committed"
#             )

#         self._validate_geometry_contract(
#             self.geometry_bank
#         )

#         structural_features = self.normalize_ordered_features(ordered_features)

#         self.geometry_bank.add_classes(
#             structural_features,
#             labels,
#             class_ids=class_ids,
#         )

#         self.validate_model_state()

#     # ------------------------------------------------------------------
#     # Classification
#     # ------------------------------------------------------------------

#     def classify_features(
#         self,
#         ordered_features: Tensor,
#         *,
#         class_ids: Sequence[int] | Tensor | None = None,
#         geometry_bank: HSIClassDiscriminativeGeometry | None = None,
#     ) -> GeometryClassifierOutput:
#         """Normalize Q_raw(x), then apply the persistent structural classifier."""
#         bank = (
#             self.geometry_bank
#             if geometry_bank is None
#             else geometry_bank
#         )

#         self._validate_geometry_contract(
#             bank
#         )

#         structural_features = self.normalize_ordered_features(ordered_features)

#         return self.classifier(
#             structural_features,
#             geometry_bank=bank,
#             class_ids=class_ids,
#         )

#     # ------------------------------------------------------------------
#     # End-to-end path
#     # ------------------------------------------------------------------

#     def forward(
#         self,
#         patch: Tensor,
#         *,
#         center_spectrum: Tensor,
#         class_ids: Sequence[int] | Tensor | None = None,
#         geometry_bank: HSIClassDiscriminativeGeometry | None = None,
#         return_aux: bool = False,
#     ) -> NECILOutput:
#         """Extract Q_raw(x) and classify normalized Q(x) with one equal rule."""
#         backbone_output = self.extract_features(
#             patch,
#             center_spectrum=center_spectrum,
#             return_aux=return_aux,
#         )

#         classification = self.classify_features(
#             backbone_output.ordered_features,
#             class_ids=class_ids,
#             geometry_bank=geometry_bank,
#         )

#         return NECILOutput(
#             backbone=backbone_output,
#             classification=classification,
#         )


# __all__ = [
#     "NECILModel",
#     "NECILOutput",
# ]
