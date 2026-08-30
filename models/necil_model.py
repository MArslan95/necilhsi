from __future__ import annotations

"""SCOPE-HSI model integration with one canonical representation.

The architecture has one learned sample representation:

    HSI -> backbone -> Q_raw -> global RMS canonicalization -> Q in R^{K x D}.

The persistent angular classifier W and historical class state {I,S,N} are
defined in this same Q coordinate. Phase 0 learns Q and W. After a class is
finalized, I/S/N are estimated from its real TRAIN Q. During incremental
learning the backbone is frozen, so historical state remains valid. Real
current-class Q and retained historical I/S jointly supervise the same
persistent classifier, including historical rows.

There is no feature transport, second representation, geometry prediction head,
phase-specific classifier, replay, or analytic classifier.
"""

from dataclasses import dataclass, replace
import math
from numbers import Integral
from typing import Any, Sequence

import torch
import torch.nn as nn

from models.backbone import BackboneOutput, HSIMambaFeatureExtractor
from models.classifier import ClassifierOutput, PersistentCosineClassifier
from models.geometry_bank import HSIClassDiscriminativeGeometry

Tensor = torch.Tensor


def _class_id_tuple(values: Sequence[int] | Tensor) -> tuple[int, ...]:
    if torch.is_tensor(values):
        if values.ndim != 1 or values.numel() == 0:
            raise ValueError("class_ids tensor must be non-empty and rank one")
        if values.dtype != torch.long:
            raise TypeError("class_ids tensor must use torch.long")
        source = values.detach().cpu().tolist()
    else:
        source = list(values)
    result: list[int] = []
    for value in source:
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise TypeError("class_ids must contain integer IDs")
        class_id = int(value)
        if class_id < 0:
            raise ValueError("class_ids must be non-negative")
        result.append(class_id)
    output = tuple(result)
    if not output:
        raise ValueError("class_ids cannot be empty")
    if len(output) != len(set(output)):
        raise ValueError("class_ids must be unique")
    return output


@dataclass(frozen=True)
class NECILOutput:
    backbone: BackboneOutput
    classification: ClassifierOutput

@dataclass(frozen=True)
class BoundarySafetyOutput:
    historical_class_ids: tuple[int, ...]
    rival_class_ids: tuple[int, ...]
    mean_gap: Tensor
    dispersion_upper_bound: Tensor
    score: Tensor



class NECILModel(nn.Module):
    """One Q-space + persistent angular classifier + Q-derived class statistics."""

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
        *,
        initial_identity: Tensor | None = None,
    ) -> None:
        """Append decision rows in the same Q-space used by class statistics.

        For an incremental phase, pass the phase-start current-class identities
        I_c^- with shape [C,K,D].  They are flattened only to initialize the new
        classifier directions; no alternate representation is formed.
        """
        initial_rows = None
        if initial_identity is not None:
            identity = torch.as_tensor(
                initial_identity,
                device=self.device,
                dtype=self.dtype,
            )
            expected_tail = (self.spectral_token_count, self.spectral_token_dim)
            if identity.ndim != 3 or tuple(identity.shape[1:]) != expected_tail:
                raise ValueError(
                    "initial_identity must have shape "
                    f"[C,{expected_tail[0]},{expected_tail[1]}]"
                )
            initial_rows = identity.reshape(identity.size(0), -1)
        self.classifier.add_classes(class_ids, initial_rows=initial_rows)
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
    def estimate_class_statistics(
        self,
        ordered_features: Tensor,
        labels: Tensor,
        *,
        class_ids: Sequence[int] | Tensor,
    ):
        """Estimate current-phase class statistics without committing state."""
        features = torch.as_tensor(ordered_features)
        self._validate_canonical_q(features, name="class-statistic estimation")
        return self.geometry_bank.estimate_classes(
            features, labels, class_ids=class_ids
        )

    def _validate_canonical_q(self, features: Tensor, *, name: str) -> None:
        expected_tail = (self.spectral_token_count, self.spectral_token_dim)
        if (
            features.ndim != 3
            or features.size(0) == 0
            or tuple(features.shape[1:]) != expected_tail
        ):
            raise ValueError(
                f"{name} requires non-empty Q with shape "
                f"[N,{expected_tail[0]},{expected_tail[1]}]"
            )
        if features.device != self.device or features.dtype != self.dtype:
            raise ValueError(f"{name} Q has wrong device/dtype")
        if not bool(torch.isfinite(features).all()):
            raise ValueError(f"{name} Q contains NaN or Inf")
        flat = features.reshape(features.size(0), -1)
        rms = torch.linalg.vector_norm(flat, ord=2, dim=1) / math.sqrt(
            float(flat.size(1))
        )
        tolerance = 2e-5 if features.dtype == torch.float32 else 1e-10
        if not bool(torch.allclose(
            rms, torch.ones_like(rms), rtol=tolerance, atol=tolerance
        )):
            raise RuntimeError(f"{name} requires canonical global-RMS Q")

    def boundary_safety_scores(
        self,
        *,
        historical_class_ids: Sequence[int] | Tensor,
        rival_class_ids: Sequence[int] | Tensor,
    ) -> BoundarySafetyOutput:
        """Evaluate historical class safety against deployed classifier rivals.

        For historical class c and rival j, with normalized classifier
        directions u_c and u_j,

            d_cj = u_c - u_j,
            mu_cj = <I_c, d_cj>_F.

        The geometry bank uses the full S_c[K,K,D] to compute a conservative
        upper bound sigma_bar_cj on the decision-gap standard deviation. For
        non-zero spread the score is the dimensionless standardized gap

            score_cj = mu_cj / sigma_bar_cj.

        For numerical zero spread, score_cj = mu_cj because there is no
        distributional variation to standardize. Positive values place the
        historical identity on its own side of the boundary; larger values
        indicate more separation relative to retained intra-class spectral
        structure. No distributional assumption is made.
        """
        historical_ids = _class_id_tuple(historical_class_ids)
        rival_ids = _class_id_tuple(rival_class_ids)
        missing_geometry = sorted(set(historical_ids) - set(self.committed_class_ids))
        if missing_geometry:
            raise RuntimeError(
                f"historical class structure is missing classes {missing_geometry}"
            )
        missing_classifier = sorted(
            (set(historical_ids) | set(rival_ids)) - set(self.classifier_class_ids)
        )
        if missing_classifier:
            raise RuntimeError(
                f"classifier is missing boundary classes {missing_classifier}"
            )

        historical_u, historical_returned = self.classifier.normalized_rows(
            historical_ids
        )
        rival_u, rival_returned = self.classifier.normalized_rows(rival_ids)
        if tuple(int(v) for v in historical_returned.detach().cpu().tolist()) != historical_ids:
            raise RuntimeError("historical classifier row order changed")
        if tuple(int(v) for v in rival_returned.detach().cpu().tolist()) != rival_ids:
            raise RuntimeError("rival classifier row order changed")

        direction = (
            historical_u[:, None, :] - rival_u[None, :, :]
        ).reshape(
            len(historical_ids),
            len(rival_ids),
            self.spectral_token_count,
            self.spectral_token_dim,
        )
        geometry = self.geometry_bank.class_geometries(historical_ids)
        mean_gap = torch.einsum("crkd,ckd->cr", direction, geometry.identity)
        dispersion = self.geometry_bank.boundary_standard_deviation_upper_bound(
            historical_ids,
            direction,
        )
        # Standardize only when historical structural dispersion is
        # numerically non-zero. If the retained spread is zero, the class is
        # deterministic in this boundary direction and the raw mean gap is the
        # correct decision quantity. This is a numerical branch, not a method
        # threshold, and avoids gap/tiny overflow for zero-variance classes.
        scale = torch.maximum(
            dispersion.detach().amax(),
            torch.ones((), device=self.device, dtype=self.dtype),
        )
        zero_tol = 64.0 * torch.finfo(self.dtype).eps * scale
        standardized = dispersion.detach() > zero_tol
        denominator = torch.where(
            standardized, dispersion, torch.ones_like(dispersion)
        )
        score = torch.where(
            standardized, mean_gap / denominator, mean_gap
        )
        if not bool(torch.isfinite(score).all()):
            raise RuntimeError("boundary safety score contains NaN or Inf")

        return BoundarySafetyOutput(
            historical_class_ids=historical_ids,
            rival_class_ids=rival_ids,
            mean_gap=mean_gap,
            dispersion_upper_bound=dispersion,
            score=score,
        )

    def validate_finalized_state(self) -> bool:
        """Require exact classifier/statistic class alignment at phase boundary."""
        self.validate_model_state()
        if self.classifier_class_ids != self.committed_class_ids:
            raise RuntimeError(
                "finalized phase requires classifier rows and class statistics "
                "for exactly the same classes in the same order"
            )
        return True

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

        requested = _class_id_tuple(class_ids)
        classifier_ids = set(self.classifier_class_ids)
        missing = [class_id for class_id in requested if class_id not in classifier_ids]
        if missing:
            raise RuntimeError(
                "cannot commit geometry before classifier rows exist for classes "
                f"{missing}"
            )

        features = torch.as_tensor(ordered_features)
        if features.size(0) != int(torch.as_tensor(labels).numel()):
            raise ValueError("class-statistic features/labels have different rows")
        self._validate_canonical_q(features, name="class-statistic commit")

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


__all__ = ["BoundarySafetyOutput", "NECILModel", "NECILOutput"]

















# from __future__ import annotations

# """SCOPE-HSI model integration with one canonical representation.

# The architecture has one learned sample representation:

#     HSI -> backbone -> Q_raw -> global RMS canonicalization -> Q in R^{K x D}.

# The persistent angular classifier W and historical class state {I,S,N} are
# defined in this same Q coordinate. Phase 0 learns Q and W. After a class is
# finalized, I/S/N are estimated from its real TRAIN Q. During incremental
# learning the backbone is frozen, so historical state remains valid. Real
# current-class Q and retained historical I/S jointly supervise the same
# persistent classifier, including historical rows.

# There is no feature transport, second representation, geometry prediction head,
# phase-specific classifier, replay, or analytic classifier.
# """

# from dataclasses import dataclass, replace
# import math
# from numbers import Integral
# from typing import Any, Sequence

# import torch
# import torch.nn as nn

# from models.backbone import BackboneOutput, HSIMambaFeatureExtractor
# from models.classifier import ClassifierOutput, PersistentCosineClassifier
# from models.geometry_bank import HSIClassDiscriminativeGeometry

# Tensor = torch.Tensor


# def _class_id_tuple(values: Sequence[int] | Tensor) -> tuple[int, ...]:
#     if torch.is_tensor(values):
#         if values.ndim != 1 or values.numel() == 0:
#             raise ValueError("class_ids tensor must be non-empty and rank one")
#         if values.dtype != torch.long:
#             raise TypeError("class_ids tensor must use torch.long")
#         source = values.detach().cpu().tolist()
#     else:
#         source = list(values)
#     result: list[int] = []
#     for value in source:
#         if isinstance(value, bool) or not isinstance(value, Integral):
#             raise TypeError("class_ids must contain integer IDs")
#         class_id = int(value)
#         if class_id < 0:
#             raise ValueError("class_ids must be non-negative")
#         result.append(class_id)
#     output = tuple(result)
#     if not output:
#         raise ValueError("class_ids cannot be empty")
#     if len(output) != len(set(output)):
#         raise ValueError("class_ids must be unique")
#     return output


# @dataclass(frozen=True)
# class NECILOutput:
#     backbone: BackboneOutput
#     classification: ClassifierOutput

# @dataclass(frozen=True)
# class BoundarySafetyOutput:
#     historical_class_ids: tuple[int, ...]
#     rival_class_ids: tuple[int, ...]
#     mean_gap: Tensor
#     dispersion_upper_bound: Tensor
#     score: Tensor



# class NECILModel(nn.Module):
#     """One Q-space + persistent angular classifier + Q-derived class statistics."""

#     def __init__(self, args: Any) -> None:
#         super().__init__()

#         self.backbone = HSIMambaFeatureExtractor(args)
#         for name in ("spectral_token_count", "spectral_token_dim"):
#             if not hasattr(self.backbone, name):
#                 raise RuntimeError(f"backbone must expose {name}")

#         reference = next(self.backbone.parameters(), None)
#         if reference is None:
#             raise RuntimeError("backbone must contain trainable parameters")
#         if reference.dtype not in (torch.float32, torch.float64):
#             raise TypeError("backbone parameters must use float32 or float64")

#         self.classifier = PersistentCosineClassifier(
#             spectral_token_count=int(self.backbone.spectral_token_count),
#             spectral_token_dim=int(self.backbone.spectral_token_dim),
#         )
#         self.geometry_bank = HSIClassDiscriminativeGeometry(
#             spectral_token_count=int(self.backbone.spectral_token_count),
#             spectral_token_dim=int(self.backbone.spectral_token_dim),
#             device=reference.device,
#             dtype=reference.dtype,
#         )
#         self.validate_model_state()

#     # ------------------------------------------------------------------
#     # Basic model contract
#     # ------------------------------------------------------------------

#     def _backbone_reference(self) -> Tensor:
#         reference = next(self.backbone.parameters(), None)
#         if reference is None:
#             raise RuntimeError("backbone has no parameters")
#         return reference

#     @property
#     def device(self) -> torch.device:
#         return self._backbone_reference().device

#     @property
#     def dtype(self) -> torch.dtype:
#         dtype = self._backbone_reference().dtype
#         if dtype not in (torch.float32, torch.float64):
#             raise RuntimeError("backbone parameters must use float32 or float64")
#         return dtype

#     @property
#     def spectral_token_count(self) -> int:
#         return int(self.backbone.spectral_token_count)

#     @property
#     def spectral_token_dim(self) -> int:
#         return int(self.backbone.spectral_token_dim)

#     @property
#     def classifier_class_ids(self) -> tuple[int, ...]:
#         return tuple(self.classifier.seen_class_ids)

#     @property
#     def committed_class_ids(self) -> tuple[int, ...]:
#         return tuple(self.geometry_bank.seen_class_ids)

#     @property
#     def spectral_normalization_fitted(self) -> bool:
#         return bool(self.backbone.spectral_normalization_fitted)

#     def _validate_geometry_contract(
#         self,
#         geometry_bank: HSIClassDiscriminativeGeometry,
#     ) -> None:
#         if not isinstance(geometry_bank, HSIClassDiscriminativeGeometry):
#             raise TypeError(
#                 "geometry_bank must be HSIClassDiscriminativeGeometry"
#             )
#         if geometry_bank.spectral_token_count != self.spectral_token_count:
#             raise RuntimeError(
#                 "backbone and geometry bank disagree on spectral_token_count"
#             )
#         if geometry_bank.spectral_token_dim != self.spectral_token_dim:
#             raise RuntimeError(
#                 "backbone and geometry bank disagree on spectral_token_dim"
#             )

#     def validate_model_state(self) -> bool:
#         if not isinstance(self.classifier, PersistentCosineClassifier):
#             raise RuntimeError("classifier must be PersistentCosineClassifier")
#         self.classifier.validate_state()
#         self._validate_geometry_contract(self.geometry_bank)
#         self.geometry_bank.validate_state()

#         if self.classifier.spectral_token_count != self.spectral_token_count:
#             raise RuntimeError(
#                 "backbone and classifier disagree on spectral_token_count"
#             )
#         if self.classifier.spectral_token_dim != self.spectral_token_dim:
#             raise RuntimeError(
#                 "backbone and classifier disagree on spectral_token_dim"
#             )
#         if self.classifier.weight.device != self.device:
#             raise RuntimeError("classifier and backbone use different devices")
#         if self.classifier.weight.dtype != self.dtype:
#             raise RuntimeError("classifier and backbone use different dtypes")
#         if self.geometry_bank.device != self.device:
#             raise RuntimeError("geometry bank and backbone use different devices")
#         if self.geometry_bank.dtype != self.dtype:
#             raise RuntimeError("geometry bank and backbone use different dtypes")

#         # A committed historical class must always have a decision row.  During
#         # an open incremental phase the classifier may legitimately contain new
#         # rows whose final geometry has not yet been committed.
#         committed = set(self.committed_class_ids)
#         classified = set(self.classifier_class_ids)
#         if not committed.issubset(classified):
#             missing = sorted(committed - classified)
#             raise RuntimeError(
#                 "committed geometry lacks classifier rows for classes "
#                 f"{missing}"
#             )

#         if len(self.geometry_bank) > 0 and not self.spectral_normalization_fitted:
#             raise RuntimeError(
#                 "stored class geometry requires fitted spectral normalization"
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
#     # Fixed HSI spectral normalization owned by the backbone
#     # ------------------------------------------------------------------

#     @torch.no_grad()
#     def set_spectral_normalization(
#         self,
#         mean: Tensor | Sequence[float],
#         scale: Tensor | Sequence[float],
#         *,
#         overwrite: bool = False,
#     ) -> None:
#         if len(self.geometry_bank) != 0:
#             raise RuntimeError(
#                 "spectral normalization cannot change after historical geometry "
#                 "has been committed"
#             )
#         self.backbone.set_spectral_normalization(
#             mean,
#             scale,
#             overwrite=overwrite,
#         )
#         if not self.spectral_normalization_fitted:
#             raise RuntimeError("spectral normalization did not enter fitted state")

#     @torch.no_grad()
#     def fit_spectral_normalization(
#         self,
#         base_center_spectra: Tensor,
#         *,
#         overwrite: bool = False,
#     ) -> None:
#         if len(self.geometry_bank) != 0:
#             raise RuntimeError(
#                 "spectral normalization cannot change after historical geometry "
#                 "has been committed"
#             )
#         self.backbone.fit_spectral_normalization(
#             base_center_spectra,
#             overwrite=overwrite,
#         )
#         if not self.spectral_normalization_fitted:
#             raise RuntimeError("spectral normalization did not enter fitted state")

#     # ------------------------------------------------------------------
#     # Canonical structural coordinate
#     # ------------------------------------------------------------------

#     @staticmethod
#     def structural_coordinates(ordered_features_raw: Tensor) -> Tensor:
#         """Remove one global positive scale degree of freedom per HSI sample.

#         The normalization uses a single scalar for the whole K x D tensor.  It
#         therefore preserves spectral-region order and all within-sample entry
#         ratios.  The only floor is the dtype's numerical tiny value; there is
#         no research threshold or tunable normalization constant.
#         """
#         features = torch.as_tensor(ordered_features_raw)
#         if features.ndim != 3 or features.size(0) == 0:
#             raise ValueError("ordered_features_raw must be non-empty [N,K,D]")
#         if not torch.is_floating_point(features) or features.is_complex():
#             raise TypeError("ordered_features_raw must be real floating point")
#         if not bool(torch.isfinite(features).all()):
#             raise ValueError("ordered_features_raw contains NaN or Inf")

#         flat = features.reshape(features.size(0), -1)
#         if flat.size(1) == 0:
#             raise ValueError("ordered_features_raw contains no feature entries")
#         rms = torch.linalg.vector_norm(flat, ord=2, dim=1, keepdim=True)
#         rms = rms / math.sqrt(float(flat.size(1)))
#         floor = torch.as_tensor(
#             torch.finfo(features.dtype).tiny,
#             device=features.device,
#             dtype=features.dtype,
#         )
#         normalized = features / rms.clamp_min(floor).view(-1, 1, 1)
#         if not bool(torch.isfinite(normalized).all()):
#             raise RuntimeError("structural RMS normalization produced NaN or Inf")
#         return normalized

#     def extract_features(
#         self,
#         patch: Tensor,
#         *,
#         center_spectrum: Tensor,
#         return_aux: bool = False,
#     ) -> BackboneOutput:
#         """Return the one canonical Q(x) used by classifier and geometry."""
#         if not self.spectral_normalization_fitted:
#             raise RuntimeError(
#                 "spectral normalization must be fitted before feature extraction"
#             )

#         raw_output = self.backbone(
#             patch,
#             center_spectrum=center_spectrum,
#             return_aux=return_aux,
#         )
#         raw = raw_output.ordered_features
#         expected = (
#             patch.size(0),
#             self.spectral_token_count,
#             self.spectral_token_dim,
#         )
#         if tuple(raw.shape) != expected:
#             raise RuntimeError(
#                 "backbone ordered_features have shape "
#                 f"{tuple(raw.shape)}, expected {expected}"
#             )
#         if raw.device != self.device:
#             raise RuntimeError("backbone output and model use different devices")
#         if raw.dtype != self.dtype:
#             raise RuntimeError("backbone output and model use different dtypes")

#         canonical = self.structural_coordinates(raw)
#         return replace(raw_output, ordered_features=canonical)

#     # ------------------------------------------------------------------
#     # Classifier: decision only
#     # ------------------------------------------------------------------

#     def add_classifier_classes(
#         self,
#         class_ids: Sequence[int] | Tensor,
#         *,
#         initial_identity: Tensor | None = None,
#     ) -> None:
#         """Append decision rows in the same Q-space used by class statistics.

#         For an incremental phase, pass the phase-start current-class identities
#         I_c^- with shape [C,K,D].  They are flattened only to initialize the new
#         classifier directions; no alternate representation is formed.
#         """
#         initial_rows = None
#         if initial_identity is not None:
#             identity = torch.as_tensor(
#                 initial_identity,
#                 device=self.device,
#                 dtype=self.dtype,
#             )
#             expected_tail = (self.spectral_token_count, self.spectral_token_dim)
#             if identity.ndim != 3 or tuple(identity.shape[1:]) != expected_tail:
#                 raise ValueError(
#                     "initial_identity must have shape "
#                     f"[C,{expected_tail[0]},{expected_tail[1]}]"
#                 )
#             initial_rows = identity.reshape(identity.size(0), -1)
#         self.classifier.add_classes(class_ids, initial_rows=initial_rows)
#         self.validate_model_state()

#     def classify_features(
#         self,
#         ordered_features: Tensor,
#         *,
#         class_ids: Sequence[int] | Tensor | None = None,
#     ) -> ClassifierOutput:
#         return self.classifier(
#             ordered_features,
#             class_ids=class_ids,
#         )

#     # ------------------------------------------------------------------
#     # Geometry: historical statistics only
#     # ------------------------------------------------------------------

#     @torch.no_grad()
#     def estimate_class_statistics(
#         self,
#         ordered_features: Tensor,
#         labels: Tensor,
#         *,
#         class_ids: Sequence[int] | Tensor,
#     ):
#         """Estimate current-phase class statistics without committing state."""
#         features = torch.as_tensor(ordered_features)
#         self._validate_canonical_q(features, name="class-statistic estimation")
#         return self.geometry_bank.estimate_classes(
#             features, labels, class_ids=class_ids
#         )

#     def _validate_canonical_q(self, features: Tensor, *, name: str) -> None:
#         expected_tail = (self.spectral_token_count, self.spectral_token_dim)
#         if (
#             features.ndim != 3
#             or features.size(0) == 0
#             or tuple(features.shape[1:]) != expected_tail
#         ):
#             raise ValueError(
#                 f"{name} requires non-empty Q with shape "
#                 f"[N,{expected_tail[0]},{expected_tail[1]}]"
#             )
#         if features.device != self.device or features.dtype != self.dtype:
#             raise ValueError(f"{name} Q has wrong device/dtype")
#         if not bool(torch.isfinite(features).all()):
#             raise ValueError(f"{name} Q contains NaN or Inf")
#         flat = features.reshape(features.size(0), -1)
#         rms = torch.linalg.vector_norm(flat, ord=2, dim=1) / math.sqrt(
#             float(flat.size(1))
#         )
#         tolerance = 2e-5 if features.dtype == torch.float32 else 1e-10
#         if not bool(torch.allclose(
#             rms, torch.ones_like(rms), rtol=tolerance, atol=tolerance
#         )):
#             raise RuntimeError(f"{name} requires canonical global-RMS Q")

#     def boundary_safety_scores(
#         self,
#         *,
#         historical_class_ids: Sequence[int] | Tensor,
#         rival_class_ids: Sequence[int] | Tensor,
#     ) -> BoundarySafetyOutput:
#         """Evaluate historical class safety against deployed classifier rivals.

#         For historical class c and rival j, with normalized classifier
#         directions u_c and u_j,

#             d_cj = u_c - u_j,
#             mu_cj = <I_c, d_cj>_F.

#         The geometry bank uses the full S_c[K,K,D] to compute a conservative
#         upper bound sigma_bar_cj on the decision-gap standard deviation. For
#         non-zero spread the score is the dimensionless standardized gap

#             score_cj = mu_cj / sigma_bar_cj.

#         For numerical zero spread, score_cj = mu_cj because there is no
#         distributional variation to standardize. Positive values place the
#         historical identity on its own side of the boundary; larger values
#         indicate more separation relative to retained intra-class spectral
#         structure. No distributional assumption is made.
#         """
#         historical_ids = _class_id_tuple(historical_class_ids)
#         rival_ids = _class_id_tuple(rival_class_ids)
#         missing_geometry = sorted(set(historical_ids) - set(self.committed_class_ids))
#         if missing_geometry:
#             raise RuntimeError(
#                 f"historical class structure is missing classes {missing_geometry}"
#             )
#         missing_classifier = sorted(
#             (set(historical_ids) | set(rival_ids)) - set(self.classifier_class_ids)
#         )
#         if missing_classifier:
#             raise RuntimeError(
#                 f"classifier is missing boundary classes {missing_classifier}"
#             )

#         historical_u, historical_returned = self.classifier.normalized_rows(
#             historical_ids
#         )
#         rival_u, rival_returned = self.classifier.normalized_rows(rival_ids)
#         if tuple(int(v) for v in historical_returned.detach().cpu().tolist()) != historical_ids:
#             raise RuntimeError("historical classifier row order changed")
#         if tuple(int(v) for v in rival_returned.detach().cpu().tolist()) != rival_ids:
#             raise RuntimeError("rival classifier row order changed")

#         direction = (
#             historical_u[:, None, :] - rival_u[None, :, :]
#         ).reshape(
#             len(historical_ids),
#             len(rival_ids),
#             self.spectral_token_count,
#             self.spectral_token_dim,
#         )
#         geometry = self.geometry_bank.class_geometries(historical_ids)
#         mean_gap = torch.einsum("crkd,ckd->cr", direction, geometry.identity)
#         dispersion = self.geometry_bank.boundary_standard_deviation_upper_bound(
#             historical_ids,
#             direction,
#         )
#         # Standardize only when historical structural dispersion is
#         # numerically non-zero. If the retained spread is zero, the class is
#         # deterministic in this boundary direction and the raw mean gap is the
#         # correct decision quantity. This is a numerical branch, not a method
#         # threshold, and avoids gap/tiny overflow for zero-variance classes.
#         scale = torch.maximum(
#             dispersion.detach().amax(),
#             torch.ones((), device=self.device, dtype=self.dtype),
#         )
#         zero_tol = 64.0 * torch.finfo(self.dtype).eps * scale
#         standardized = dispersion.detach() > zero_tol
#         denominator = torch.where(
#             standardized, dispersion, torch.ones_like(dispersion)
#         )
#         score = torch.where(
#             standardized, mean_gap / denominator, mean_gap
#         )
#         if not bool(torch.isfinite(score).all()):
#             raise RuntimeError("boundary safety score contains NaN or Inf")

#         return BoundarySafetyOutput(
#             historical_class_ids=historical_ids,
#             rival_class_ids=rival_ids,
#             mean_gap=mean_gap,
#             dispersion_upper_bound=dispersion,
#             score=score,
#         )

#     def validate_finalized_state(self) -> bool:
#         """Require exact classifier/statistic class alignment at phase boundary."""
#         self.validate_model_state()
#         if self.classifier_class_ids != self.committed_class_ids:
#             raise RuntimeError(
#                 "finalized phase requires classifier rows and class statistics "
#                 "for exactly the same classes in the same order"
#             )
#         return True

#     @torch.no_grad()
#     def add_classes(
#         self,
#         ordered_features: Tensor,
#         labels: Tensor,
#         *,
#         class_ids: Sequence[int] | Tensor,
#     ) -> None:
#         """Commit {I_c,S_c,N_c} from canonical TRAIN Q(x)."""
#         if not self.spectral_normalization_fitted:
#             raise RuntimeError(
#                 "spectral normalization must be fitted before geometry commit"
#             )

#         requested = _class_id_tuple(class_ids)
#         classifier_ids = set(self.classifier_class_ids)
#         missing = [class_id for class_id in requested if class_id not in classifier_ids]
#         if missing:
#             raise RuntimeError(
#                 "cannot commit geometry before classifier rows exist for classes "
#                 f"{missing}"
#             )

#         features = torch.as_tensor(ordered_features)
#         if features.size(0) != int(torch.as_tensor(labels).numel()):
#             raise ValueError("class-statistic features/labels have different rows")
#         self._validate_canonical_q(features, name="class-statistic commit")

#         self.geometry_bank.add_classes(
#             features,
#             labels,
#             class_ids=class_ids,
#         )
#         self.validate_model_state()

#     # ------------------------------------------------------------------
#     # End-to-end prediction path
#     # ------------------------------------------------------------------

#     def forward(
#         self,
#         patch: Tensor,
#         *,
#         center_spectrum: Tensor,
#         class_ids: Sequence[int] | Tensor | None = None,
#         return_aux: bool = False,
#     ) -> NECILOutput:
#         backbone_output = self.extract_features(
#             patch,
#             center_spectrum=center_spectrum,
#             return_aux=return_aux,
#         )
#         classification = self.classify_features(
#             backbone_output.ordered_features,
#             class_ids=class_ids,
#         )
#         return NECILOutput(
#             backbone=backbone_output,
#             classification=classification,
#         )


# __all__ = ["BoundarySafetyOutput", "NECILModel", "NECILOutput"]



# from __future__ import annotations

# """SCOPE-HSI model integration with one representation and one coordinate law.

# The architecture has exactly one learned sample representation:

#     HSI -> backbone -> Q_raw -> global RMS canonicalization -> Q in R^{K x D}.

# Both downstream states are defined in that same Q coordinate:

#     classifier row W_c     : decision direction acting on Q,
#     class statistics G_c   : {I_c,S_c,N_c} estimated from Q.

# They have different roles but are not independent coordinate systems.  During
# incremental learning, if paired current-class observations estimate one common
# orthogonal movement R from Q_old to Q_new, all historical Q-derived state is
# propagated atomically with that same movement:

#     I_c -> R I_c,
#     S_c[:,:,d] -> R S_c[:,:,d] R^T,
#     W_c -> R W_c.

# This keeps historical class structure and its decision direction mutually
# consistent in the current Q-space.  No second representation, geometry head,
# or independent preservation coordinate exists.
# """

# from dataclasses import dataclass, replace
# import math
# from numbers import Integral
# from typing import Any, Sequence

# import torch
# import torch.nn as nn

# from models.backbone import BackboneOutput, HSIMambaFeatureExtractor
# from models.classifier import ClassifierOutput, PersistentCosineClassifier
# from models.geometry_bank import HSIClassDiscriminativeGeometry

# Tensor = torch.Tensor


# def _class_id_tuple(values: Sequence[int] | Tensor) -> tuple[int, ...]:
#     if torch.is_tensor(values):
#         if values.ndim != 1 or values.numel() == 0:
#             raise ValueError("class_ids tensor must be non-empty and rank one")
#         if values.dtype != torch.long:
#             raise TypeError("class_ids tensor must use torch.long")
#         source = values.detach().cpu().tolist()
#     else:
#         source = list(values)
#     result: list[int] = []
#     for value in source:
#         if isinstance(value, bool) or not isinstance(value, Integral):
#             raise TypeError("class_ids must contain integer IDs")
#         class_id = int(value)
#         if class_id < 0:
#             raise ValueError("class_ids must be non-negative")
#         result.append(class_id)
#     output = tuple(result)
#     if not output:
#         raise ValueError("class_ids cannot be empty")
#     if len(output) != len(set(output)):
#         raise ValueError("class_ids must be unique")
#     return output


# @dataclass(frozen=True)
# class NECILOutput:
#     backbone: BackboneOutput
#     classification: ClassifierOutput


# class NECILModel(nn.Module):
#     """One Q-space + persistent angular classifier + Q-derived class statistics."""

#     def __init__(self, args: Any) -> None:
#         super().__init__()

#         self.backbone = HSIMambaFeatureExtractor(args)
#         for name in ("spectral_token_count", "spectral_token_dim"):
#             if not hasattr(self.backbone, name):
#                 raise RuntimeError(f"backbone must expose {name}")

#         reference = next(self.backbone.parameters(), None)
#         if reference is None:
#             raise RuntimeError("backbone must contain trainable parameters")
#         if reference.dtype not in (torch.float32, torch.float64):
#             raise TypeError("backbone parameters must use float32 or float64")

#         self.classifier = PersistentCosineClassifier(
#             spectral_token_count=int(self.backbone.spectral_token_count),
#             spectral_token_dim=int(self.backbone.spectral_token_dim),
#         )
#         self.geometry_bank = HSIClassDiscriminativeGeometry(
#             spectral_token_count=int(self.backbone.spectral_token_count),
#             spectral_token_dim=int(self.backbone.spectral_token_dim),
#             device=reference.device,
#             dtype=reference.dtype,
#         )
#         self.validate_model_state()

#     # ------------------------------------------------------------------
#     # Basic model contract
#     # ------------------------------------------------------------------

#     def _backbone_reference(self) -> Tensor:
#         reference = next(self.backbone.parameters(), None)
#         if reference is None:
#             raise RuntimeError("backbone has no parameters")
#         return reference

#     @property
#     def device(self) -> torch.device:
#         return self._backbone_reference().device

#     @property
#     def dtype(self) -> torch.dtype:
#         dtype = self._backbone_reference().dtype
#         if dtype not in (torch.float32, torch.float64):
#             raise RuntimeError("backbone parameters must use float32 or float64")
#         return dtype

#     @property
#     def spectral_token_count(self) -> int:
#         return int(self.backbone.spectral_token_count)

#     @property
#     def spectral_token_dim(self) -> int:
#         return int(self.backbone.spectral_token_dim)

#     @property
#     def classifier_class_ids(self) -> tuple[int, ...]:
#         return tuple(self.classifier.seen_class_ids)

#     @property
#     def committed_class_ids(self) -> tuple[int, ...]:
#         return tuple(self.geometry_bank.seen_class_ids)

#     @property
#     def spectral_normalization_fitted(self) -> bool:
#         return bool(self.backbone.spectral_normalization_fitted)

#     def _validate_geometry_contract(
#         self,
#         geometry_bank: HSIClassDiscriminativeGeometry,
#     ) -> None:
#         if not isinstance(geometry_bank, HSIClassDiscriminativeGeometry):
#             raise TypeError(
#                 "geometry_bank must be HSIClassDiscriminativeGeometry"
#             )
#         if geometry_bank.spectral_token_count != self.spectral_token_count:
#             raise RuntimeError(
#                 "backbone and geometry bank disagree on spectral_token_count"
#             )
#         if geometry_bank.spectral_token_dim != self.spectral_token_dim:
#             raise RuntimeError(
#                 "backbone and geometry bank disagree on spectral_token_dim"
#             )

#     def validate_model_state(self) -> bool:
#         if not isinstance(self.classifier, PersistentCosineClassifier):
#             raise RuntimeError("classifier must be PersistentCosineClassifier")
#         self.classifier.validate_state()
#         self._validate_geometry_contract(self.geometry_bank)
#         self.geometry_bank.validate_state()

#         if self.classifier.spectral_token_count != self.spectral_token_count:
#             raise RuntimeError(
#                 "backbone and classifier disagree on spectral_token_count"
#             )
#         if self.classifier.spectral_token_dim != self.spectral_token_dim:
#             raise RuntimeError(
#                 "backbone and classifier disagree on spectral_token_dim"
#             )
#         if self.classifier.weight.device != self.device:
#             raise RuntimeError("classifier and backbone use different devices")
#         if self.classifier.weight.dtype != self.dtype:
#             raise RuntimeError("classifier and backbone use different dtypes")
#         if self.geometry_bank.device != self.device:
#             raise RuntimeError("geometry bank and backbone use different devices")
#         if self.geometry_bank.dtype != self.dtype:
#             raise RuntimeError("geometry bank and backbone use different dtypes")

#         # A committed historical class must always have a decision row.  During
#         # an open incremental phase the classifier may legitimately contain new
#         # rows whose final geometry has not yet been committed.
#         committed = set(self.committed_class_ids)
#         classified = set(self.classifier_class_ids)
#         if not committed.issubset(classified):
#             missing = sorted(committed - classified)
#             raise RuntimeError(
#                 "committed geometry lacks classifier rows for classes "
#                 f"{missing}"
#             )

#         if len(self.geometry_bank) > 0 and not self.spectral_normalization_fitted:
#             raise RuntimeError(
#                 "stored class geometry requires fitted spectral normalization"
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
#     # Fixed HSI spectral normalization owned by the backbone
#     # ------------------------------------------------------------------

#     @torch.no_grad()
#     def set_spectral_normalization(
#         self,
#         mean: Tensor | Sequence[float],
#         scale: Tensor | Sequence[float],
#         *,
#         overwrite: bool = False,
#     ) -> None:
#         if len(self.geometry_bank) != 0:
#             raise RuntimeError(
#                 "spectral normalization cannot change after historical geometry "
#                 "has been committed"
#             )
#         self.backbone.set_spectral_normalization(
#             mean,
#             scale,
#             overwrite=overwrite,
#         )
#         if not self.spectral_normalization_fitted:
#             raise RuntimeError("spectral normalization did not enter fitted state")

#     @torch.no_grad()
#     def fit_spectral_normalization(
#         self,
#         base_center_spectra: Tensor,
#         *,
#         overwrite: bool = False,
#     ) -> None:
#         if len(self.geometry_bank) != 0:
#             raise RuntimeError(
#                 "spectral normalization cannot change after historical geometry "
#                 "has been committed"
#             )
#         self.backbone.fit_spectral_normalization(
#             base_center_spectra,
#             overwrite=overwrite,
#         )
#         if not self.spectral_normalization_fitted:
#             raise RuntimeError("spectral normalization did not enter fitted state")

#     # ------------------------------------------------------------------
#     # Canonical structural coordinate
#     # ------------------------------------------------------------------

#     @staticmethod
#     def structural_coordinates(ordered_features_raw: Tensor) -> Tensor:
#         """Remove one global positive scale degree of freedom per HSI sample.

#         The normalization uses a single scalar for the whole K x D tensor.  It
#         therefore preserves spectral-region order and all within-sample entry
#         ratios.  The only floor is the dtype's numerical tiny value; there is
#         no research threshold or tunable normalization constant.
#         """
#         features = torch.as_tensor(ordered_features_raw)
#         if features.ndim != 3 or features.size(0) == 0:
#             raise ValueError("ordered_features_raw must be non-empty [N,K,D]")
#         if not torch.is_floating_point(features) or features.is_complex():
#             raise TypeError("ordered_features_raw must be real floating point")
#         if not bool(torch.isfinite(features).all()):
#             raise ValueError("ordered_features_raw contains NaN or Inf")

#         flat = features.reshape(features.size(0), -1)
#         if flat.size(1) == 0:
#             raise ValueError("ordered_features_raw contains no feature entries")
#         rms = torch.linalg.vector_norm(flat, ord=2, dim=1, keepdim=True)
#         rms = rms / math.sqrt(float(flat.size(1)))
#         floor = torch.as_tensor(
#             torch.finfo(features.dtype).tiny,
#             device=features.device,
#             dtype=features.dtype,
#         )
#         normalized = features / rms.clamp_min(floor).view(-1, 1, 1)
#         if not bool(torch.isfinite(normalized).all()):
#             raise RuntimeError("structural RMS normalization produced NaN or Inf")
#         return normalized

#     def extract_features(
#         self,
#         patch: Tensor,
#         *,
#         center_spectrum: Tensor,
#         return_aux: bool = False,
#     ) -> BackboneOutput:
#         """Return the one canonical Q(x) used by classifier and geometry."""
#         if not self.spectral_normalization_fitted:
#             raise RuntimeError(
#                 "spectral normalization must be fitted before feature extraction"
#             )

#         raw_output = self.backbone(
#             patch,
#             center_spectrum=center_spectrum,
#             return_aux=return_aux,
#         )
#         raw = raw_output.ordered_features
#         expected = (
#             patch.size(0),
#             self.spectral_token_count,
#             self.spectral_token_dim,
#         )
#         if tuple(raw.shape) != expected:
#             raise RuntimeError(
#                 "backbone ordered_features have shape "
#                 f"{tuple(raw.shape)}, expected {expected}"
#             )
#         if raw.device != self.device:
#             raise RuntimeError("backbone output and model use different devices")
#         if raw.dtype != self.dtype:
#             raise RuntimeError("backbone output and model use different dtypes")

#         canonical = self.structural_coordinates(raw)
#         return replace(raw_output, ordered_features=canonical)

#     # ------------------------------------------------------------------
#     # Classifier: decision only
#     # ------------------------------------------------------------------

#     def add_classifier_classes(
#         self,
#         class_ids: Sequence[int] | Tensor,
#         *,
#         initial_identity: Tensor | None = None,
#     ) -> None:
#         """Append decision rows in the same Q-space used by class statistics.

#         For an incremental phase, pass the phase-start current-class identities
#         I_c^- with shape [C,K,D].  They are flattened only to initialize the new
#         classifier directions; no alternate representation is formed.
#         """
#         initial_rows = None
#         if initial_identity is not None:
#             identity = torch.as_tensor(
#                 initial_identity,
#                 device=self.device,
#                 dtype=self.dtype,
#             )
#             expected_tail = (self.spectral_token_count, self.spectral_token_dim)
#             if identity.ndim != 3 or tuple(identity.shape[1:]) != expected_tail:
#                 raise ValueError(
#                     "initial_identity must have shape "
#                     f"[C,{expected_tail[0]},{expected_tail[1]}]"
#                 )
#             initial_rows = identity.reshape(identity.size(0), -1)
#         self.classifier.add_classes(class_ids, initial_rows=initial_rows)
#         self.validate_model_state()

#     def classify_features(
#         self,
#         ordered_features: Tensor,
#         *,
#         class_ids: Sequence[int] | Tensor | None = None,
#     ) -> ClassifierOutput:
#         return self.classifier(
#             ordered_features,
#             class_ids=class_ids,
#         )

#     # ------------------------------------------------------------------
#     # Geometry: historical statistics only
#     # ------------------------------------------------------------------

#     @torch.no_grad()
#     def estimate_class_statistics(
#         self,
#         ordered_features: Tensor,
#         labels: Tensor,
#         *,
#         class_ids: Sequence[int] | Tensor,
#     ):
#         """Estimate current-phase class statistics without committing state."""
#         features = torch.as_tensor(ordered_features)
#         self._validate_canonical_q(features, name="class-statistic estimation")
#         return self.geometry_bank.estimate_classes(
#             features, labels, class_ids=class_ids
#         )

#     def _validate_canonical_q(self, features: Tensor, *, name: str) -> None:
#         expected_tail = (self.spectral_token_count, self.spectral_token_dim)
#         if (
#             features.ndim != 3
#             or features.size(0) == 0
#             or tuple(features.shape[1:]) != expected_tail
#         ):
#             raise ValueError(
#                 f"{name} requires non-empty Q with shape "
#                 f"[N,{expected_tail[0]},{expected_tail[1]}]"
#             )
#         if features.device != self.device or features.dtype != self.dtype:
#             raise ValueError(f"{name} Q has wrong device/dtype")
#         if not bool(torch.isfinite(features).all()):
#             raise ValueError(f"{name} Q contains NaN or Inf")
#         flat = features.reshape(features.size(0), -1)
#         rms = torch.linalg.vector_norm(flat, ord=2, dim=1) / math.sqrt(
#             float(flat.size(1))
#         )
#         tolerance = 2e-5 if features.dtype == torch.float32 else 1e-10
#         if not bool(torch.allclose(
#             rms, torch.ones_like(rms), rtol=tolerance, atol=tolerance
#         )):
#             raise RuntimeError(f"{name} requires canonical global-RMS Q")

#     @torch.no_grad()
#     def transport_historical_state(
#         self,
#         transform: Tensor,
#         *,
#         class_ids: Sequence[int] | Tensor,
#     ) -> None:
#         """Atomically propagate historical {I,S,W} with one common R.

#         Geometry/classifier roles stay separate, but their coordinate movement
#         cannot diverge.  If either update fails, both are restored.
#         """
#         requested = _class_id_tuple(class_ids)
#         if not set(requested).issubset(set(self.committed_class_ids)):
#             raise RuntimeError("transport requires committed historical class statistics")
#         if not set(requested).issubset(set(self.classifier_class_ids)):
#             raise RuntimeError("transport requires historical classifier rows")

#         identity_before = self.geometry_bank.identity_bank.detach().clone()
#         structure_before = self.geometry_bank.structure_bank.detach().clone()
#         classifier_before = self.classifier.weight.detach().clone()
#         try:
#             self.geometry_bank.transport_classes(transform, class_ids=requested)
#             self.classifier.transport_classes(transform, class_ids=requested)
#             self.validate_model_state()
#         except Exception:
#             self.geometry_bank.identity_bank.copy_(identity_before)
#             self.geometry_bank.structure_bank.copy_(structure_before)
#             self.classifier.weight.copy_(classifier_before)
#             self.validate_model_state()
#             raise

#     def validate_finalized_state(self) -> bool:
#         """Require exact classifier/statistic class alignment at phase boundary."""
#         self.validate_model_state()
#         if self.classifier_class_ids != self.committed_class_ids:
#             raise RuntimeError(
#                 "finalized phase requires classifier rows and class statistics "
#                 "for exactly the same classes in the same order"
#             )
#         return True

#     @torch.no_grad()
#     def add_classes(
#         self,
#         ordered_features: Tensor,
#         labels: Tensor,
#         *,
#         class_ids: Sequence[int] | Tensor,
#     ) -> None:
#         """Commit {I_c,S_c,N_c} from canonical TRAIN Q(x)."""
#         if not self.spectral_normalization_fitted:
#             raise RuntimeError(
#                 "spectral normalization must be fitted before geometry commit"
#             )

#         requested = _class_id_tuple(class_ids)
#         classifier_ids = set(self.classifier_class_ids)
#         missing = [class_id for class_id in requested if class_id not in classifier_ids]
#         if missing:
#             raise RuntimeError(
#                 "cannot commit geometry before classifier rows exist for classes "
#                 f"{missing}"
#             )

#         features = torch.as_tensor(ordered_features)
#         if features.size(0) != int(torch.as_tensor(labels).numel()):
#             raise ValueError("class-statistic features/labels have different rows")
#         self._validate_canonical_q(features, name="class-statistic commit")

#         self.geometry_bank.add_classes(
#             features,
#             labels,
#             class_ids=class_ids,
#         )
#         self.validate_model_state()

#     # ------------------------------------------------------------------
#     # End-to-end prediction path
#     # ------------------------------------------------------------------

#     def forward(
#         self,
#         patch: Tensor,
#         *,
#         center_spectrum: Tensor,
#         class_ids: Sequence[int] | Tensor | None = None,
#         return_aux: bool = False,
#     ) -> NECILOutput:
#         backbone_output = self.extract_features(
#             patch,
#             center_spectrum=center_spectrum,
#             return_aux=return_aux,
#         )
#         classification = self.classify_features(
#             backbone_output.ordered_features,
#             class_ids=class_ids,
#         )
#         return NECILOutput(
#             backbone=backbone_output,
#             classification=classification,
#         )


# __all__ = ["NECILModel", "NECILOutput"]

