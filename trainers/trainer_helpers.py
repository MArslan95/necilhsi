from __future__ import annotations

"""Shared trainer utilities for the clean NECIL-HSI architecture.

Responsibilities are intentionally limited to protocol validation, batch
handling, deterministic canonical-Q collection, read-only evaluation, and
serialization.  This helper does not define a classifier, geometry objective,
or preservation mechanism.
"""

from contextlib import contextmanager
import json
import math
import os
from numbers import Integral, Real
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence

import torch

from models.classifier import PersistentCosineClassifier
from models.geometry_bank import HSIClassDiscriminativeGeometry
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
            if value.numel() != 1:
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
    def _validated_class_ids(
        cls,
        values: Iterable[int],
        *,
        name: str,
    ) -> list[int]:
        ids = [
            cls._exact_nonnegative_int(value, name=f"{name} value")
            for value in values
        ]
        if not ids:
            raise RuntimeError(f"{name} cannot be empty")
        if len(ids) != len(set(ids)):
            raise RuntimeError(f"{name} must contain unique class IDs")
        return ids

    def assert_dataset_contract(self) -> Dict[str, Any]:
        phase_map = getattr(self.dataset, "phase_to_classes", None)
        if not isinstance(phase_map, Mapping) or 0 not in phase_map:
            raise RuntimeError("dataset.phase_to_classes must define phase 0")

        schedule: Dict[int, list[int]] = {}
        used: set[int] = set()
        for raw_phase, raw_classes in phase_map.items():
            phase = self._exact_nonnegative_int(raw_phase, name="phase_id")
            ids = self._validated_class_ids(
                raw_classes,
                name=f"phase_{phase}_classes",
            )
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
            "get_cumulative_dataloader",
            "get_reporting_dataloader",
            "get_new_classes",
            "get_old_classes",
            "get_seen_classes",
            "assert_exemplar_free_contract",
        )
        missing = [name for name in required if not callable(getattr(self.dataset, name, None))]
        if missing:
            raise RuntimeError(f"dataset lacks required APIs: {missing}")
        if self.dataset.assert_exemplar_free_contract() is not True:
            raise RuntimeError("dataset reports an exemplar-free protocol violation")

        validation_enabled = getattr(self.dataset, "validation_enabled", None)
        if not isinstance(validation_enabled, bool):
            raise RuntimeError("dataset.validation_enabled must be bool")

        base = self._validated_class_ids(
            self.dataset.get_new_classes(0),
            name="base_classes",
        )
        if base != schedule[0]:
            raise RuntimeError("phase map and get_new_classes(0) disagree")
        if len(base) < 2:
            raise RuntimeError("phase 0 requires at least two classes")

        processed = getattr(self.dataset, "processed_cube", None)
        ordered = getattr(self.dataset, "ordered_spectral_cube", None)
        backbone = getattr(self.model, "backbone", None)
        if processed is None or ordered is None or backbone is None:
            raise RuntimeError("dataset/model lacks HSI cube or backbone state")
        if getattr(processed, "ndim", None) != 3 or getattr(ordered, "ndim", None) != 3:
            raise RuntimeError("processed and ordered HSI cubes must be [H,W,C/B]")
        if tuple(processed.shape[:2]) != tuple(ordered.shape[:2]):
            raise RuntimeError("processed and ordered cubes disagree spatially")
        if int(processed.shape[2]) != int(backbone.patch_bands):
            raise RuntimeError("processed cube channels disagree with backbone")
        if int(ordered.shape[2]) != int(backbone.spectral_bands):
            raise RuntimeError("ordered cube bands disagree with backbone")
        if int(getattr(self.dataset, "patch_size", 0)) != int(backbone.patch_size):
            raise RuntimeError("dataset/backbone patch sizes disagree")

        leakage_report = None
        reporter = getattr(self.dataset, "spatial_leakage_report", None)
        if callable(reporter):
            leakage_report = reporter()
            if not isinstance(leakage_report, Mapping):
                raise RuntimeError("spatial_leakage_report must return a mapping")

        return {
            "schedule": dict(sorted(schedule.items())),
            "base_classes": list(base),
            "spatial_leakage": leakage_report,
        }

    def assert_model_contract(self) -> Dict[str, Any]:
        required = (
            "extract_features",
            "structural_coordinates",
            "add_classes",
            "add_classifier_classes",
            "classify_features",
            "forward",
            "validate_model_state",
            "fit_spectral_normalization",
        )
        missing = [name for name in required if not callable(getattr(self.model, name, None))]
        if missing:
            raise RuntimeError(f"model lacks required APIs: {missing}")

        backbone = getattr(self.model, "backbone", None)
        bank = getattr(self.model, "geometry_bank", None)
        classifier = getattr(self.model, "classifier", None)
        if backbone is None or bank is None or classifier is None:
            raise RuntimeError("model must own backbone, classifier, and geometry_bank")
        if not isinstance(bank, HSIClassDiscriminativeGeometry):
            raise RuntimeError("geometry_bank must be HSIClassDiscriminativeGeometry")
        if not isinstance(classifier, PersistentCosineClassifier):
            raise RuntimeError("classifier must be PersistentCosineClassifier")

        for name in (
            "patch_bands",
            "spectral_bands",
            "patch_size",
            "spectral_token_count",
            "spectral_token_dim",
        ):
            if int(getattr(backbone, name, 0)) <= 0:
                raise RuntimeError(f"backbone.{name} must be positive")

        if int(bank.spectral_token_count) != int(self.model.spectral_token_count):
            raise RuntimeError("model/geometry spectral token counts disagree")
        if int(bank.spectral_token_dim) != int(self.model.spectral_token_dim):
            raise RuntimeError("model/geometry spectral token dimensions disagree")
        if torch.device(self.model.device) != self.device:
            raise RuntimeError("model is not on trainer device")
        if bank.device != self.device:
            raise RuntimeError("geometry bank is not on trainer device")
        if classifier.weight.device != self.device:
            raise RuntimeError("classifier is not on trainer device")
        if self.model.dtype != bank.dtype or self.model.dtype != classifier.weight.dtype:
            raise RuntimeError("model/classifier/geometry dtypes disagree")

        self.model.validate_model_state()
        return {
            "architecture": "clean_role_separated_necil_hsi",
            "patch_bands": int(backbone.patch_bands),
            "spectral_bands": int(backbone.spectral_bands),
            "patch_size": int(backbone.patch_size),
            "spectral_token_count": int(self.model.spectral_token_count),
            "spectral_token_dim": int(self.model.spectral_token_dim),
            "classifier_parameter_count": int(
                sum(parameter.numel() for parameter in classifier.parameters())
            ),
            "geometry_role": "historical statistics only",
            "spectral_normalization_fitted": bool(self.model.spectral_normalization_fitted),
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

        patch = torch.as_tensor(batch["image"], device=self.device, dtype=self.model.dtype)
        spectrum = torch.as_tensor(
            batch["raw_center_spectrum"],
            device=self.device,
            dtype=self.model.dtype,
        )
        labels = torch.as_tensor(batch["label"], device=self.device).flatten()

        backbone = self.model.backbone
        expected_patch = (
            int(backbone.patch_bands),
            int(backbone.patch_size),
            int(backbone.patch_size),
        )
        if patch.ndim != 4 or patch.size(0) == 0 or tuple(patch.shape[1:]) != expected_patch:
            raise RuntimeError(f"processed HSI patch shape {tuple(patch.shape)} is invalid")
        expected_spectrum = (patch.size(0), int(backbone.spectral_bands))
        if spectrum.ndim != 2 or tuple(spectrum.shape) != expected_spectrum:
            raise RuntimeError(
                f"raw center spectrum shape {tuple(spectrum.shape)} is invalid; "
                f"expected {expected_spectrum}"
            )
        if not bool(torch.isfinite(patch).all()) or not bool(torch.isfinite(spectrum).all()):
            raise RuntimeError("HSI batch contains NaN or Inf")
        if labels.numel() != patch.size(0) or labels.dtype == torch.bool or labels.is_complex():
            raise RuntimeError("labels are invalid or batch-misaligned")
        if torch.is_floating_point(labels):
            if not bool(torch.isfinite(labels).all()) or not bool(labels.eq(labels.round()).all()):
                raise RuntimeError("labels must be finite integers")
        labels = labels.to(torch.long)
        if bool((labels < 0).any()):
            raise RuntimeError("labels must be non-negative")
        return patch, spectrum, labels

    @contextmanager
    def temporary_eval_state(self):
        states = {module: bool(module.training) for module in self.model.modules()}
        try:
            self.model.eval()
            yield
        finally:
            for module, state in states.items():
                module.training = state

    @torch.no_grad()
    def collect_ordered_features(self, loader: Any) -> Dict[str, Tensor]:
        """Collect canonical Q(x), never raw Q_raw(x)."""
        feature_rows: list[Tensor] = []
        label_rows: list[Tensor] = []
        with self.temporary_eval_state():
            for batch in loader:
                patch, spectrum, labels = self.unpack_batch(batch)
                output = self.model.extract_features(
                    patch,
                    center_spectrum=spectrum,
                    return_aux=False,
                )
                feature_rows.append(output.ordered_features.detach())
                label_rows.append(labels.detach())
        if not feature_rows:
            raise RuntimeError("cannot encode an empty loader")

        ordered_features = torch.cat(feature_rows, dim=0)
        labels = torch.cat(label_rows, dim=0).flatten()
        expected_shape = (
            labels.numel(),
            int(self.model.spectral_token_count),
            int(self.model.spectral_token_dim),
        )
        if tuple(ordered_features.shape) != expected_shape:
            raise RuntimeError(
                f"collected ordered features have {tuple(ordered_features.shape)}, "
                f"expected {expected_shape}"
            )
        if ordered_features.device != self.model.geometry_bank.device:
            raise RuntimeError("collected features are on the wrong device")
        if ordered_features.dtype != self.model.geometry_bank.dtype:
            raise RuntimeError("collected features use the wrong dtype")
        if not bool(torch.isfinite(ordered_features).all()):
            raise RuntimeError("collected features contain NaN or Inf")

        flat = ordered_features.reshape(ordered_features.size(0), -1)
        qrms = torch.linalg.vector_norm(flat, ord=2, dim=1) / math.sqrt(float(flat.size(1)))
        tolerance = 2e-5 if ordered_features.dtype == torch.float32 else 1e-10
        if not bool(
            torch.allclose(
                qrms,
                torch.ones_like(qrms),
                rtol=tolerance,
                atol=tolerance,
            )
        ):
            raise RuntimeError("collected Q(x) bypassed structural RMS normalization")

        return {"ordered_features": ordered_features, "labels": labels}

    @torch.no_grad()
    def evaluate_loader(
        self,
        loader: Any,
        *,
        class_ids: Sequence[int],
        target_class_ids: Optional[Sequence[int]] = None,
        geometry_bank: Optional[HSIClassDiscriminativeGeometry] = None,
    ) -> Dict[str, Any]:
        return evaluate_model_loader(
            self.model,
            loader,
            class_ids=class_ids,
            target_class_ids=target_class_ids,
            geometry_bank=geometry_bank,
            device=self.device,
        )

    def geometry_state_summary(self) -> Dict[str, Any]:
        bank = self.model.geometry_bank
        bank.validate_state()
        classes: Dict[int, Dict[str, Any]] = {}
        for class_id in self.model.committed_class_ids:
            view = bank.class_geometry(class_id)
            k = int(view.structure.size(0))
            upper = torch.triu_indices(k, k, device=view.structure.device)
            unique_structure = view.structure[upper[0], upper[1], :]
            cross = view.cross_region_covariation
            offdiag = ~torch.eye(k, device=cross.device, dtype=torch.bool)
            classes[int(class_id)] = {
                "sample_count": int(view.sample_count),
                "identity_rms": float(torch.sqrt(view.identity.square().mean()).item()),
                "structure_unique_rms": float(
                    torch.sqrt(unique_structure.square().mean()).item()
                ),
                "feature_variation_mean": float(view.feature_variation.mean().item()),
                "cross_region_offdiagonal_rms": float(
                    torch.sqrt(cross[offdiag].square().mean()).item()
                    if bool(offdiag.any())
                    else 0.0
                ),
            }
        return {
            "class_ids": list(self.model.committed_class_ids),
            "class_count": len(bank),
            "spectral_token_count": int(bank.spectral_token_count),
            "spectral_token_dim": int(bank.spectral_token_dim),
            "stored_state": "{I_c,S_c,N_c} per class",
            "geometry_role": "historical statistics only; not classification",
            "classes": classes,
        }

    @staticmethod
    def save_json(
        path: str | os.PathLike[str],
        payload: Mapping[str, Any],
    ) -> str:
        destination = Path(path).resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)

        def convert(value: Any) -> Any:
            if torch.is_tensor(value):
                if value.ndim == 0:
                    return value.item()
                return value.detach().cpu().tolist()
            if isinstance(value, Mapping):
                return {str(k): convert(v) for k, v in value.items()}
            if isinstance(value, (list, tuple)):
                return [convert(v) for v in value]
            if isinstance(value, Path):
                return str(value)
            return value

        with destination.open("w", encoding="utf-8") as stream:
            json.dump(convert(payload), stream, indent=2, sort_keys=True)
        return str(destination)


__all__ = ["TrainerHelper"]


