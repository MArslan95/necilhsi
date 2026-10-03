from __future__ import annotations

"""Shared trainer infrastructure for the HSI CDS NECIL pipeline."""

from contextlib import contextmanager
import math
from numbers import Integral, Real
from typing import Any, Dict, Iterable, Mapping, Sequence

import torch

from models.classifier import CosineClassificationHead
from models.geometry_bank import HSIClassDiscriminativeStructureBank
from utils.eval import evaluate_loader as evaluate_model_loader

Tensor = torch.Tensor


class TrainerHelper:
    model: Any
    dataset: Any
    args: Any
    device: torch.device

    def cfg(self, name: str, default: Any) -> Any:
        return getattr(self.args, name, default)

    @staticmethod
    def _exact_nonnegative_int(value: object, *, name: str) -> int:
        if torch.is_tensor(value):
            if (
                value.numel() != 1
                or value.dtype == torch.bool
                or value.is_complex()
                or torch.is_floating_point(value)
            ):
                raise RuntimeError(f"{name} must be an integer scalar")
            value = value.item()
        if isinstance(value, bool):
            raise RuntimeError(f"{name} must be an integer")
        if isinstance(value, Integral):
            result = int(value)
        elif isinstance(value, Real):
            number = float(value)
            if not math.isfinite(number) or not number.is_integer():
                raise RuntimeError(f"{name} must be a finite integer")
            result = int(number)
        else:
            raise RuntimeError(f"{name} must be an integer")
        if result < 0:
            raise RuntimeError(f"{name} must be non-negative")
        return result

    @classmethod
    def _validated_class_ids(cls, values: Iterable[int], *, name: str) -> list[int]:
        ids = [
            cls._exact_nonnegative_int(value, name=f"{name} value") for value in values
        ]
        if not ids or len(ids) != len(set(ids)):
            raise RuntimeError(f"{name} must contain unique class IDs")
        return ids

    def assert_dataset_contract(self) -> Dict[str, Any]:
        phase_map = getattr(self.dataset, "phase_to_classes", None)
        if not isinstance(phase_map, Mapping) or 0 not in phase_map:
            raise RuntimeError("dataset.phase_to_classes must define Phase 0")

        schedule: Dict[int, list[int]] = {}
        used: set[int] = set()
        for raw_phase, raw_classes in phase_map.items():
            phase = self._exact_nonnegative_int(raw_phase, name="phase_id")
            ids = self._validated_class_ids(raw_classes, name=f"phase_{phase}_classes")
            overlap = used.intersection(ids)
            if overlap:
                raise RuntimeError(
                    f"classes occur in multiple phases: {sorted(overlap)}"
                )
            schedule[phase] = ids
            used.update(ids)

        if sorted(schedule) != list(range(len(schedule))):
            raise RuntimeError("phase IDs must be contiguous from zero")

        required = (
            "start_phase",
            "finalize_phase",
            "get_phase_dataloader",
            "get_reporting_dataloader",
            "get_new_classes",
            "get_old_classes",
            "get_seen_classes",
            "assert_exemplar_free_contract",
        )
        missing = [
            name for name in required if not callable(getattr(self.dataset, name, None))
        ]
        if missing:
            raise RuntimeError(f"dataset lacks required APIs: {missing}")

        if self.dataset.assert_exemplar_free_contract() is not True:
            raise RuntimeError("dataset reports an exemplar-free protocol violation")

        processed = getattr(self.dataset, "processed_cube", None)
        ordered = getattr(self.dataset, "ordered_spectral_cube", None)
        if processed is None or ordered is None:
            raise RuntimeError("dataset must expose processed and ordered HSI cubes")

        if int(processed.shape[2]) != int(self.model.backbone.patch_bands):
            raise RuntimeError("processed cube channels disagree with backbone")
        if int(ordered.shape[2]) != int(self.model.backbone.spectral_bands):
            raise RuntimeError("ordered cube bands disagree with backbone")
        if int(self.dataset.patch_size) != int(self.model.backbone.patch_size):
            raise RuntimeError("dataset/backbone patch sizes disagree")

        return {"schedule": dict(sorted(schedule.items()))}

    def assert_model_contract(self) -> Dict[str, Any]:
        required = (
            "extract_features",
            "extract_features_with_backbone",
            "set_base_trainability",
            "set_incremental_trainability",
            "physical_spectral_evidence",
            "classify_features",
            "estimate_physical_class_structure",
            "estimate_class_structure",
            "commit_class_structure",
            "replace_class_structure",
            "class_discriminative_structure",
            "validate_model_state",
            "validate_finalized_state",
            "fit_spectral_normalization",
            "forward",
        )
        missing = [
            name for name in required if not callable(getattr(self.model, name, None))
        ]
        if missing:
            raise RuntimeError(f"model lacks required NECIL-HSI APIs: {missing}")

        if not isinstance(
            self.model.structure_bank,
            HSIClassDiscriminativeStructureBank,
        ):
            raise RuntimeError(
                "structure_bank must be HSIClassDiscriminativeStructureBank"
            )

        classifier = self.model.classifier
        if not isinstance(classifier, CosineClassificationHead):
            raise RuntimeError("classifier must be CosineClassificationHead")
        if classifier.parameter_count <= 0:
            raise RuntimeError("classifier must contain trainable parameters")
        if not bool(torch.isfinite(classifier.logit_scale)) or not bool(
            classifier.logit_scale > 0
        ):
            raise RuntimeError("classifier logit scale is invalid")

        expected_feature_dim = int(self.model.backbone.spectral_token_count) * int(
            self.model.backbone.spectral_token_dim
        )
        if int(classifier.feature_dim) != expected_feature_dim:
            raise RuntimeError("classifier feature dimension disagrees with Q[K,D]")

        scheduled_classes = {
            int(class_id)
            for classes in self.dataset.phase_to_classes.values()
            for class_id in classes
        }
        expected_total_classes = len(scheduled_classes)
        if int(classifier.total_class_count) != expected_total_classes:
            raise RuntimeError(
                "classifier capacity disagrees with the incremental protocol"
            )

        if tuple(classifier.weight.shape) != (
            expected_total_classes,
            expected_feature_dim,
        ):
            raise RuntimeError(
                "classifier weight shape is incompatible with the protocol and Q"
            )

        self.model.validate_model_state()
        return {
            "representation": "one ordered learned Q[K,D]",
            "physical_spectral_evidence": "direct robustly normalized native spectrum P[B,1]",
            "persistent_cds": "{I_Q,V_Q,I_P,V_P}; learned/physical inter-class relations are derived on demand",
            "prediction_head": "one all-seen cosine head; inference is exactly tau*cosine(Q,W)",
            "base_training": "real base CE + unified Whole-CDS contrastive separation in the single ordered Q",
            "incremental_reference": "frozen previous backbone/classifier are temporary teacher references; the current backbone remains plastic",
            "incremental_feature_stability": "Feature KD plus historical Whole-CDS intra/inter preservation; optional old-logit KD",
            "incremental_cds": "one observable phase map plus bounded Whole-CDS-conditioned residual correction; physical I_P/V_P remain fixed",
            "classifier_finalization": "Stage 3 freezes E_t and jointly optimizes all seen W using corrected historical Whole-CDS positive supervision plus real-current all-seen CE; no replay, reconstruction state, phase bias or second head is used",
            "phase_commit": "commit the existing Whole-CDS {I_Q,V_Q,I_P,V_P} and single deployed W only after Stage 1 representation, Stage 2 coordinate correction/real-new CDS, and Stage 3 balanced decision learning",
        }

    def unpack_batch(
        self,
        batch: Mapping[str, Any],
    ) -> tuple[Tensor, Tensor, Tensor]:
        if not isinstance(batch, Mapping):
            raise TypeError("HSI batches must be mappings")

        required = {"image", "raw_center_spectrum", "label"}
        missing = required - set(batch)
        if missing:
            raise KeyError(f"batch missing {sorted(missing)}")

        patch = torch.as_tensor(
            batch["image"],
            device=self.device,
            dtype=self.model.dtype,
        )
        spectrum = torch.as_tensor(
            batch["raw_center_spectrum"],
            device=self.device,
            dtype=self.model.dtype,
        )

        raw_labels = torch.as_tensor(batch["label"])
        if (
            raw_labels.dtype == torch.bool
            or raw_labels.is_complex()
            or torch.is_floating_point(raw_labels)
        ):
            raise RuntimeError("labels must use an integer dtype")

        labels = raw_labels.to(
            device=self.device,
            dtype=torch.long,
        ).flatten()

        expected_patch = (
            int(self.model.backbone.patch_bands),
            int(self.model.backbone.patch_size),
            int(self.model.backbone.patch_size),
        )
        if (
            patch.ndim != 4
            or patch.size(0) == 0
            or tuple(patch.shape[1:]) != expected_patch
        ):
            raise RuntimeError("processed HSI patch shape is invalid")

        if spectrum.shape != (
            patch.size(0),
            int(self.model.backbone.spectral_bands),
        ):
            raise RuntimeError("raw center spectrum shape is invalid")

        if labels.shape != (patch.size(0),):
            raise RuntimeError("labels are batch-misaligned")

        if not bool(torch.isfinite(patch).all()) or not bool(
            torch.isfinite(spectrum).all()
        ):
            raise RuntimeError("HSI batch contains NaN or Inf")

        return patch, spectrum, labels

    @contextmanager
    def temporary_eval_state(self):
        states = [(module, bool(module.training)) for module in self.model.modules()]
        try:
            self.model.eval()
            yield
        finally:
            for module, state in states:
                module.training = state

    @torch.no_grad()
    def collect_ordered_features(
        self,
        loader: Any,
        *,
        backbone: Any | None = None,
        require_coordinates: bool = False,
    ) -> Dict[str, Tensor]:
        feature_rows: list[Tensor] = []
        spectrum_rows: list[Tensor] = []
        label_rows: list[Tensor] = []
        coordinate_rows: list[Tensor] = []
        coordinates_present: bool | None = None

        with self.temporary_eval_state():
            if backbone is not None:
                backbone.eval()

            for batch in loader:
                patch, spectrum, labels = self.unpack_batch(batch)
                output = (
                    self.model.extract_features(
                        patch,
                        center_spectrum=spectrum,
                        return_aux=False,
                    )
                    if backbone is None
                    else self.model.extract_features_with_backbone(
                        backbone,
                        patch,
                        center_spectrum=spectrum,
                        return_aux=False,
                    )
                )

                feature_rows.append(output.ordered_features.detach())
                spectrum_rows.append(spectrum.detach())
                label_rows.append(labels.detach())

                has_coords = "coord" in batch
                if coordinates_present is None:
                    coordinates_present = has_coords
                elif coordinates_present != has_coords:
                    raise RuntimeError("coord presence must be consistent")

                if require_coordinates and not has_coords:
                    raise RuntimeError("coordinates are required but absent")

                if has_coords:
                    coords = torch.as_tensor(batch["coord"])
                    if (
                        coords.shape != (labels.numel(), 2)
                        or coords.dtype == torch.bool
                        or coords.is_complex()
                        or torch.is_floating_point(coords)
                    ):
                        raise RuntimeError("coord must be integer [N,2]")
                    coordinate_rows.append(coords.to(torch.long).cpu())

        if not feature_rows:
            raise RuntimeError("cannot encode an empty loader")

        q = torch.cat(feature_rows, dim=0)
        spectra = torch.cat(spectrum_rows, dim=0)
        labels = torch.cat(label_rows, dim=0).flatten()

        result: Dict[str, Tensor] = {
            "ordered_features": q,
            "raw_center_spectra": spectra,
            "labels": labels,
        }

        if coordinates_present:
            coords = torch.cat(coordinate_rows, dim=0)
            if coords.shape != (labels.numel(), 2):
                raise RuntimeError("collected coordinates are misaligned")
            result["coords"] = coords

        return result

    @torch.no_grad()
    def evaluate_loader(
        self,
        loader: Any,
        *,
        class_ids: Sequence[int],
        target_class_ids: Sequence[int] | None = None,
    ) -> Dict[str, Any]:
        return evaluate_model_loader(
            self.model,
            loader,
            class_ids=class_ids,
            target_class_ids=target_class_ids,
            device=self.device,
        )


__all__ = ["TrainerHelper"]



############ Before the Orcal Version TrainerHelper Starts ############

# from __future__ import annotations

# """Shared trainer infrastructure for the HSI CDS NECIL pipeline."""

# from contextlib import contextmanager
# import math
# from numbers import Integral, Real
# from typing import Any, Dict, Iterable, Mapping, Sequence

# import torch

# from models.classifier import CosineClassificationHead
# from models.geometry_bank import HSIClassDiscriminativeStructureBank
# from utils.eval import evaluate_loader as evaluate_model_loader

# Tensor = torch.Tensor


# class TrainerHelper:
#     model: Any
#     dataset: Any
#     args: Any
#     device: torch.device

#     def cfg(self, name: str, default: Any) -> Any:
#         return getattr(self.args, name, default)

#     @staticmethod
#     def _exact_nonnegative_int(value: object, *, name: str) -> int:
#         if torch.is_tensor(value):
#             if (
#                 value.numel() != 1
#                 or value.dtype == torch.bool
#                 or value.is_complex()
#                 or torch.is_floating_point(value)
#             ):
#                 raise RuntimeError(f"{name} must be an integer scalar")
#             value = value.item()
#         if isinstance(value, bool):
#             raise RuntimeError(f"{name} must be an integer")
#         if isinstance(value, Integral):
#             result = int(value)
#         elif isinstance(value, Real):
#             number = float(value)
#             if not math.isfinite(number) or not number.is_integer():
#                 raise RuntimeError(f"{name} must be a finite integer")
#             result = int(number)
#         else:
#             raise RuntimeError(f"{name} must be an integer")
#         if result < 0:
#             raise RuntimeError(f"{name} must be non-negative")
#         return result

#     @classmethod
#     def _validated_class_ids(cls, values: Iterable[int], *, name: str) -> list[int]:
#         ids = [
#             cls._exact_nonnegative_int(value, name=f"{name} value")
#             for value in values
#         ]
#         if not ids or len(ids) != len(set(ids)):
#             raise RuntimeError(f"{name} must contain unique class IDs")
#         return ids

#     def assert_dataset_contract(self) -> Dict[str, Any]:
#         phase_map = getattr(self.dataset, "phase_to_classes", None)
#         if not isinstance(phase_map, Mapping) or 0 not in phase_map:
#             raise RuntimeError("dataset.phase_to_classes must define Phase 0")

#         schedule: Dict[int, list[int]] = {}
#         used: set[int] = set()
#         for raw_phase, raw_classes in phase_map.items():
#             phase = self._exact_nonnegative_int(raw_phase, name="phase_id")
#             ids = self._validated_class_ids(
#                 raw_classes, name=f"phase_{phase}_classes"
#             )
#             overlap = used.intersection(ids)
#             if overlap:
#                 raise RuntimeError(
#                     f"classes occur in multiple phases: {sorted(overlap)}"
#                 )
#             schedule[phase] = ids
#             used.update(ids)

#         if sorted(schedule) != list(range(len(schedule))):
#             raise RuntimeError("phase IDs must be contiguous from zero")

#         required = (
#             "start_phase",
#             "finalize_phase",
#             "get_phase_dataloader",
#             "get_reporting_dataloader",
#             "get_new_classes",
#             "get_old_classes",
#             "get_seen_classes",
#             "assert_exemplar_free_contract",
#         )
#         missing = [
#             name
#             for name in required
#             if not callable(getattr(self.dataset, name, None))
#         ]
#         if missing:
#             raise RuntimeError(f"dataset lacks required APIs: {missing}")

#         if self.dataset.assert_exemplar_free_contract() is not True:
#             raise RuntimeError(
#                 "dataset reports an exemplar-free protocol violation"
#             )

#         processed = getattr(self.dataset, "processed_cube", None)
#         ordered = getattr(self.dataset, "ordered_spectral_cube", None)
#         if processed is None or ordered is None:
#             raise RuntimeError(
#                 "dataset must expose processed and ordered HSI cubes"
#             )

#         if int(processed.shape[2]) != int(self.model.backbone.patch_bands):
#             raise RuntimeError(
#                 "processed cube channels disagree with backbone"
#             )
#         if int(ordered.shape[2]) != int(self.model.backbone.spectral_bands):
#             raise RuntimeError(
#                 "ordered cube bands disagree with backbone"
#             )
#         if int(self.dataset.patch_size) != int(self.model.backbone.patch_size):
#             raise RuntimeError(
#                 "dataset/backbone patch sizes disagree"
#             )

#         return {"schedule": dict(sorted(schedule.items()))}

#     def assert_model_contract(self) -> Dict[str, Any]:
#         required = (
#             "extract_features",
#             "extract_features_with_backbone",
#             "set_base_trainability",
#             "set_incremental_trainability",
#             "physical_spectral_evidence",
#             "classify_features",
#             "estimate_physical_class_structure",
#             "estimate_class_structure",
#             "commit_class_structure",
#             "replace_class_structure",
#             "class_discriminative_structure",
#             "validate_model_state",
#             "validate_finalized_state",
#             "fit_spectral_normalization",
#             "forward",
#         )
#         missing = [
#             name
#             for name in required
#             if not callable(getattr(self.model, name, None))
#         ]
#         if missing:
#             raise RuntimeError(f"model lacks required NECIL-HSI APIs: {missing}")

#         if not isinstance(
#             self.model.structure_bank,
#             HSIClassDiscriminativeStructureBank,
#         ):
#             raise RuntimeError(
#                 "structure_bank must be HSIClassDiscriminativeStructureBank"
#             )

#         classifier = self.model.classifier
#         if not isinstance(classifier, CosineClassificationHead):
#             raise RuntimeError("classifier must be CosineClassificationHead")
#         if classifier.parameter_count <= 0:
#             raise RuntimeError("classifier must contain trainable parameters")
#         if not bool(torch.isfinite(classifier.logit_scale)) or not bool(classifier.logit_scale > 0):
#             raise RuntimeError("classifier logit scale is invalid")

#         expected_feature_dim = (
#             int(self.model.backbone.spectral_token_count)
#             * int(self.model.backbone.spectral_token_dim)
#         )
#         if int(classifier.feature_dim) != expected_feature_dim:
#             raise RuntimeError(
#                 "classifier feature dimension disagrees with Q[K,D]"
#             )

#         scheduled_classes = {
#             int(class_id)
#             for classes in self.dataset.phase_to_classes.values()
#             for class_id in classes
#         }
#         expected_total_classes = len(scheduled_classes)
#         if int(classifier.total_class_count) != expected_total_classes:
#             raise RuntimeError(
#                 "classifier capacity disagrees with the incremental protocol"
#             )

#         if tuple(classifier.weight.shape) != (
#             expected_total_classes,
#             expected_feature_dim,
#         ):
#             raise RuntimeError(
#                 "classifier weight shape is incompatible with the protocol and Q"
#             )

#         self.model.validate_model_state()
#         return {
#             "representation": "one ordered learned Q[K,D]",
#             "physical_spectral_evidence": "direct robustly normalized native spectrum P[B,1]",
#             "persistent_cds": "{I_Q,V_Q,I_P,V_P}; learned/physical inter-class relations are derived on demand",
#             "prediction_head": "one calibrated all-seen cosine head; inference is tau*cosine(Q,W)+committed phase offset",
#             "base_training": "real base CE + unified Whole-CDS contrastive separation in the single ordered Q",
#             "incremental_reference": "frozen previous backbone/classifier are temporary teacher references; the current backbone remains plastic",
#             "incremental_feature_stability": "Feature KD plus historical Whole-CDS intra/inter preservation; optional old-logit KD",
#             "incremental_cds": "one observable phase map plus bounded Whole-CDS-conditioned residual correction; physical I_P/V_P remain fixed",
#             "classifier_finalization": "after E+ and all-seen CDS are finalized, one deterministic mixed-evidence solver calibrates incoming W direction plus one shared phase offset from real-new HSI and historical Whole-CDS; no classifier epoch schedule",
#             "phase_commit": "commit only after representation, coordinate correction, real-new CDS construction and unified decision calibration; no replay or second classifier",
#         }

#     def unpack_batch(
#         self,
#         batch: Mapping[str, Any],
#     ) -> tuple[Tensor, Tensor, Tensor]:
#         if not isinstance(batch, Mapping):
#             raise TypeError("HSI batches must be mappings")

#         required = {"image", "raw_center_spectrum", "label"}
#         missing = required - set(batch)
#         if missing:
#             raise KeyError(f"batch missing {sorted(missing)}")

#         patch = torch.as_tensor(
#             batch["image"],
#             device=self.device,
#             dtype=self.model.dtype,
#         )
#         spectrum = torch.as_tensor(
#             batch["raw_center_spectrum"],
#             device=self.device,
#             dtype=self.model.dtype,
#         )

#         raw_labels = torch.as_tensor(batch["label"])
#         if (
#             raw_labels.dtype == torch.bool
#             or raw_labels.is_complex()
#             or torch.is_floating_point(raw_labels)
#         ):
#             raise RuntimeError("labels must use an integer dtype")

#         labels = raw_labels.to(
#             device=self.device,
#             dtype=torch.long,
#         ).flatten()

#         expected_patch = (
#             int(self.model.backbone.patch_bands),
#             int(self.model.backbone.patch_size),
#             int(self.model.backbone.patch_size),
#         )
#         if (
#             patch.ndim != 4
#             or patch.size(0) == 0
#             or tuple(patch.shape[1:]) != expected_patch
#         ):
#             raise RuntimeError(
#                 "processed HSI patch shape is invalid"
#             )

#         if spectrum.shape != (
#             patch.size(0),
#             int(self.model.backbone.spectral_bands),
#         ):
#             raise RuntimeError(
#                 "raw center spectrum shape is invalid"
#             )

#         if labels.shape != (patch.size(0),):
#             raise RuntimeError("labels are batch-misaligned")

#         if (
#             not bool(torch.isfinite(patch).all())
#             or not bool(torch.isfinite(spectrum).all())
#         ):
#             raise RuntimeError("HSI batch contains NaN or Inf")

#         return patch, spectrum, labels

#     @contextmanager
#     def temporary_eval_state(self):
#         states = [
#             (module, bool(module.training))
#             for module in self.model.modules()
#         ]
#         try:
#             self.model.eval()
#             yield
#         finally:
#             for module, state in states:
#                 module.training = state

#     @torch.no_grad()
#     def collect_ordered_features(
#         self,
#         loader: Any,
#         *,
#         backbone: Any | None = None,
#         require_coordinates: bool = False,
#     ) -> Dict[str, Tensor]:
#         feature_rows: list[Tensor] = []
#         spectrum_rows: list[Tensor] = []
#         label_rows: list[Tensor] = []
#         coordinate_rows: list[Tensor] = []
#         coordinates_present: bool | None = None

#         with self.temporary_eval_state():
#             if backbone is not None:
#                 backbone.eval()

#             for batch in loader:
#                 patch, spectrum, labels = self.unpack_batch(batch)
#                 output = (
#                     self.model.extract_features(
#                         patch,
#                         center_spectrum=spectrum,
#                         return_aux=False,
#                     )
#                     if backbone is None
#                     else self.model.extract_features_with_backbone(
#                         backbone,
#                         patch,
#                         center_spectrum=spectrum,
#                         return_aux=False,
#                     )
#                 )

#                 feature_rows.append(output.ordered_features.detach())
#                 spectrum_rows.append(spectrum.detach())
#                 label_rows.append(labels.detach())

#                 has_coords = "coord" in batch
#                 if coordinates_present is None:
#                     coordinates_present = has_coords
#                 elif coordinates_present != has_coords:
#                     raise RuntimeError(
#                         "coord presence must be consistent"
#                     )

#                 if require_coordinates and not has_coords:
#                     raise RuntimeError(
#                         "coordinates are required but absent"
#                     )

#                 if has_coords:
#                     coords = torch.as_tensor(batch["coord"])
#                     if (
#                         coords.shape != (labels.numel(), 2)
#                         or coords.dtype == torch.bool
#                         or coords.is_complex()
#                         or torch.is_floating_point(coords)
#                     ):
#                         raise RuntimeError(
#                             "coord must be integer [N,2]"
#                         )
#                     coordinate_rows.append(
#                         coords.to(torch.long).cpu()
#                     )

#         if not feature_rows:
#             raise RuntimeError("cannot encode an empty loader")

#         q = torch.cat(feature_rows, dim=0)
#         spectra = torch.cat(spectrum_rows, dim=0)
#         labels = torch.cat(label_rows, dim=0).flatten()

#         result: Dict[str, Tensor] = {
#             "ordered_features": q,
#             "raw_center_spectra": spectra,
#             "labels": labels,
#         }

#         if coordinates_present:
#             coords = torch.cat(coordinate_rows, dim=0)
#             if coords.shape != (labels.numel(), 2):
#                 raise RuntimeError(
#                     "collected coordinates are misaligned"
#                 )
#             result["coords"] = coords

#         return result

#     @torch.no_grad()
#     def evaluate_loader(
#         self,
#         loader: Any,
#         *,
#         class_ids: Sequence[int],
#         target_class_ids: Sequence[int] | None = None,
#     ) -> Dict[str, Any]:
#         return evaluate_model_loader(
#             self.model,
#             loader,
#             class_ids=class_ids,
#             target_class_ids=target_class_ids,
#             device=self.device,
#         )


# __all__ = ["TrainerHelper"]


############ ENDs Here ############
