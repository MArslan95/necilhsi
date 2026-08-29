from __future__ import annotations

"""Shared trainer utilities for the locked SCOPE-HSI architecture.

Role
----
This helper owns only infrastructure shared by base/incremental trainers:

* protocol and model-contract validation,
* strict HSI batch unpacking,
* temporary read-only evaluation state,
* deterministic collection of the one canonical representation Q,
* row-alignment checks for before/after transport observations,
* read-only classifier evaluation wrappers,
* class-structure summaries and JSON serialization.

It does NOT define classification, class statistics, coordinate transport,
preservation losses, replay, or optimizer mathematics.

Representation contract
-----------------------
Every optimization/evaluation path must use the model-owned canonical

    Q(x) in R^{K x D}.

The helper never constructs a second feature representation.  For incremental
coordinate transport, the same current HSI pixels must be observed before and
after an update; their pixel coordinates are therefore required as row identity.
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
            if value.dtype == torch.bool or value.is_complex() or torch.is_floating_point(value):
                raise RuntimeError(f"{name} must be an integer scalar")
            value = value.item()

        if isinstance(value, bool):
            raise RuntimeError(f"{name} must be an integer")

        if isinstance(value, Integral):
            result = int(value)
        elif isinstance(value, Real):
            # Configuration values occasionally arrive from argparse/yaml as 1.0.
            # Accept only mathematically exact integers; never round.
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

    # ------------------------------------------------------------------
    # Protocol/model contracts
    # ------------------------------------------------------------------

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
        missing = [
            name for name in required
            if not callable(getattr(self.dataset, name, None))
        ]
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
        # These operations are the locked SCOPE model boundary.  The trainer is
        # not allowed to re-implement them independently.
        required = (
            "extract_features",
            "structural_coordinates",
            "add_classes",
            "add_classifier_classes",
            "classify_features",
            "estimate_class_statistics",
            "transport_historical_state",
            "validate_model_state",
            "validate_finalized_state",
            "fit_spectral_normalization",
            "forward",
        )
        missing = [
            name for name in required
            if not callable(getattr(self.model, name, None))
        ]
        if missing:
            raise RuntimeError(f"model lacks required SCOPE APIs: {missing}")

        backbone = getattr(self.model, "backbone", None)
        bank = getattr(self.model, "geometry_bank", None)
        classifier = getattr(self.model, "classifier", None)
        if backbone is None or bank is None or classifier is None:
            raise RuntimeError(
                "model must own backbone, classifier, and class-structure bank"
            )
        if not isinstance(bank, HSIClassDiscriminativeGeometry):
            raise RuntimeError(
                "geometry_bank must be HSIClassDiscriminativeGeometry"
            )
        if not isinstance(classifier, PersistentCosineClassifier):
            raise RuntimeError(
                "classifier must be PersistentCosineClassifier"
            )

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
            raise RuntimeError("model/class-structure spectral token counts disagree")
        if int(bank.spectral_token_dim) != int(self.model.spectral_token_dim):
            raise RuntimeError("model/class-structure spectral token dimensions disagree")
        if int(classifier.spectral_token_count) != int(self.model.spectral_token_count):
            raise RuntimeError("model/classifier spectral token counts disagree")
        if int(classifier.spectral_token_dim) != int(self.model.spectral_token_dim):
            raise RuntimeError("model/classifier spectral token dimensions disagree")

        if torch.device(self.model.device) != self.device:
            raise RuntimeError("model is not on trainer device")
        if bank.device != self.device:
            raise RuntimeError("class-structure bank is not on trainer device")
        if classifier.weight.device != self.device:
            raise RuntimeError("classifier is not on trainer device")
        if self.model.dtype != bank.dtype or self.model.dtype != classifier.weight.dtype:
            raise RuntimeError("model/classifier/class-structure dtypes disagree")

        self.model.validate_model_state()
        return {
            "architecture": "SCOPE_HSI_one_Q_one_movement",
            "patch_bands": int(backbone.patch_bands),
            "spectral_bands": int(backbone.spectral_bands),
            "patch_size": int(backbone.patch_size),
            "spectral_region_count": int(self.model.spectral_token_count),
            "feature_dimension": int(self.model.spectral_token_dim),
            "classifier_parameter_count": int(
                sum(parameter.numel() for parameter in classifier.parameters())
            ),
            "representation_role": "single canonical ordered spectral-spatial Q",
            "class_structure_role": "historical decision-relevant statistics {I,S,N}",
            "classifier_role": "one persistent all-seen decision operator W on Q",
            "stability_role": "common Q-coordinate transport of historical {I,S,W}",
            "spectral_normalization_fitted": bool(
                self.model.spectral_normalization_fitted
            ),
        }

    # ------------------------------------------------------------------
    # Batch handling
    # ------------------------------------------------------------------

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
        if raw_labels.dtype == torch.bool or raw_labels.is_complex():
            raise RuntimeError("labels must use an integer dtype")
        if torch.is_floating_point(raw_labels):
            # Labels are semantic IDs, not numeric measurements.  Reject float
            # labels rather than silently truncating/casting them.
            raise RuntimeError("labels must use an integer dtype")
        labels = raw_labels.to(device=self.device, dtype=torch.long).flatten()

        backbone = self.model.backbone
        expected_patch = (
            int(backbone.patch_bands),
            int(backbone.patch_size),
            int(backbone.patch_size),
        )
        if (
            patch.ndim != 4
            or patch.size(0) == 0
            or tuple(patch.shape[1:]) != expected_patch
        ):
            raise RuntimeError(
                f"processed HSI patch shape {tuple(patch.shape)} is invalid; "
                f"expected [N,{expected_patch[0]},{expected_patch[1]},"
                f"{expected_patch[2]}]"
            )

        expected_spectrum = (
            patch.size(0),
            int(backbone.spectral_bands),
        )
        if spectrum.ndim != 2 or tuple(spectrum.shape) != expected_spectrum:
            raise RuntimeError(
                f"raw center spectrum shape {tuple(spectrum.shape)} is invalid; "
                f"expected {expected_spectrum}"
            )

        if (
            not bool(torch.isfinite(patch).all())
            or not bool(torch.isfinite(spectrum).all())
        ):
            raise RuntimeError("HSI batch contains NaN or Inf")
        if labels.shape != (patch.size(0),):
            raise RuntimeError("labels are batch-misaligned")
        if bool((labels < 0).any()):
            raise RuntimeError("labels must be non-negative")
        return patch, spectrum, labels

    @staticmethod
    def _batch_coordinates(
        batch: Mapping[str, Any],
        *,
        rows: int,
    ) -> Optional[Tensor]:
        if "coord" not in batch:
            return None

        coords = torch.as_tensor(batch["coord"])
        if coords.dtype == torch.bool or coords.is_complex() or torch.is_floating_point(coords):
            raise RuntimeError("coord must use an integer dtype")
        coords = coords.to(dtype=torch.long)
        if coords.shape != (rows, 2):
            raise RuntimeError("coord must have shape [N,2]")
        return coords

    # ------------------------------------------------------------------
    # Read-only state and canonical-Q collection
    # ------------------------------------------------------------------

    @contextmanager
    def temporary_eval_state(self):
        # Preserve exact per-module training flags because a caller may have a
        # deliberately mixed state.  `training` is PyTorch's canonical flag.
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
        require_coordinates: bool = False,
    ) -> Dict[str, Tensor]:
        """Collect the one canonical Q(x), never Q_raw or auxiliary features.

        When ``require_coordinates=True`` every row must include its HSI pixel
        coordinate.  Incremental before/after transport fitting should use this
        mode so row identity is proven rather than inferred from loader order.
        """
        feature_rows: list[Tensor] = []
        label_rows: list[Tensor] = []
        coordinate_rows: list[Tensor] = []
        coordinate_presence: Optional[bool] = None

        with self.temporary_eval_state():
            for batch in loader:
                patch, spectrum, labels = self.unpack_batch(batch)
                coords = self._batch_coordinates(batch, rows=labels.numel())
                has_coords = coords is not None

                if coordinate_presence is None:
                    coordinate_presence = has_coords
                elif coordinate_presence != has_coords:
                    raise RuntimeError(
                        "coord must be present in every collection batch or none"
                    )
                if require_coordinates and coords is None:
                    raise RuntimeError(
                        "coordinate transport requires HSI pixel coordinates "
                        "for every current TRAIN sample"
                    )

                output = self.model.extract_features(
                    patch,
                    center_spectrum=spectrum,
                    return_aux=False,
                )
                feature_rows.append(output.ordered_features.detach())
                label_rows.append(labels.detach())
                if coords is not None:
                    coordinate_rows.append(coords.detach().cpu())

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
                f"collected Q has {tuple(ordered_features.shape)}, "
                f"expected {expected_shape}"
            )
        if ordered_features.device != self.device:
            raise RuntimeError("collected Q is on the wrong device")
        if ordered_features.dtype != self.model.dtype:
            raise RuntimeError("collected Q uses the wrong dtype")
        if not bool(torch.isfinite(ordered_features).all()):
            raise RuntimeError("collected Q contains NaN or Inf")

        flat = ordered_features.reshape(ordered_features.size(0), -1)
        qrms = torch.linalg.vector_norm(flat, ord=2, dim=1) / math.sqrt(
            float(flat.size(1))
        )
        tolerance = 2e-5 if ordered_features.dtype == torch.float32 else 1e-10
        if not bool(
            torch.allclose(
                qrms,
                torch.ones_like(qrms),
                rtol=tolerance,
                atol=tolerance,
            )
        ):
            raise RuntimeError(
                "collected features bypassed model structural RMS canonicalization"
            )

        result: Dict[str, Tensor] = {
            "ordered_features": ordered_features,
            "labels": labels,
        }
        if coordinate_presence:
            coords = torch.cat(coordinate_rows, dim=0)
            if coords.shape != (labels.numel(), 2):
                raise RuntimeError("collected coordinates are row-misaligned")
            result["coords"] = coords
        elif require_coordinates:
            raise RuntimeError("coordinate transport collection lacks coordinates")
        return result

    @staticmethod
    def assert_aligned_feature_collections(
        before: Mapping[str, Tensor],
        after: Mapping[str, Tensor],
    ) -> None:
        """Prove that two Q collections refer to the same HSI pixels in same rows."""
        for name in ("ordered_features", "labels", "coords"):
            if name not in before or name not in after:
                raise RuntimeError(
                    f"paired coordinate-transport collections require '{name}'"
                )

        q_before = torch.as_tensor(before["ordered_features"])
        q_after = torch.as_tensor(after["ordered_features"])
        if q_before.shape != q_after.shape:
            raise RuntimeError("before/after Q collections have different shapes")

        labels_before = torch.as_tensor(before["labels"]).detach().cpu()
        labels_after = torch.as_tensor(after["labels"]).detach().cpu()
        if not torch.equal(labels_before, labels_after):
            raise RuntimeError("before/after transport labels are not row-aligned")

        coords_before = torch.as_tensor(before["coords"]).detach().cpu()
        coords_after = torch.as_tensor(after["coords"]).detach().cpu()
        if not torch.equal(coords_before, coords_after):
            raise RuntimeError(
                "before/after transport observations are not the same HSI pixels"
            )

        # A repeated coordinate would make row identity ambiguous and usually
        # indicates a sampler/dataset bug for deterministic transport fitting.
        if torch.unique(coords_before, dim=0).size(0) != coords_before.size(0):
            raise RuntimeError(
                "coordinate-transport collection contains duplicate HSI pixels"
            )

    @torch.no_grad()
    def evaluate_loader(
        self,
        loader: Any,
        *,
        class_ids: Sequence[int],
        target_class_ids: Optional[Sequence[int]] = None,
    ) -> Dict[str, Any]:
        return evaluate_model_loader(
            self.model,
            loader,
            class_ids=class_ids,
            target_class_ids=target_class_ids,
            device=self.device,
        )

    # ------------------------------------------------------------------
    # Read-only class-structure reporting
    # ------------------------------------------------------------------

    def class_structure_state_summary(self) -> Dict[str, Any]:
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
                "identity_rms": float(
                    torch.sqrt(view.identity.square().mean()).item()
                ),
                "structure_unique_rms": float(
                    torch.sqrt(unique_structure.square().mean()).item()
                ),
                "feature_variation_mean": float(
                    view.feature_variation.mean().item()
                ),
                "cross_region_offdiagonal_rms": float(
                    torch.sqrt(cross[offdiag].square().mean()).item()
                    if bool(offdiag.any())
                    else 0.0
                ),
            }

        return {
            "class_ids": list(self.model.committed_class_ids),
            "class_count": len(bank),
            "spectral_region_count": int(bank.spectral_token_count),
            "feature_dimension": int(bank.spectral_token_dim),
            "stored_state": "{I_c,S_c,N_c} per class",
            "role": (
                "historical decision-relevant class structure in the same Q "
                "coordinate as the persistent classifier"
            ),
            "classes": classes,
        }

    # Backward-compatible reporting name only.  It does not restore the old
    # geometry-as-classifier architecture.
    def geometry_state_summary(self) -> Dict[str, Any]:
        return self.class_structure_state_summary()

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
