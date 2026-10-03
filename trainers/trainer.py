from __future__ import annotations
import copy
import hashlib
import os
from pathlib import Path
from typing import Any, Dict, Mapping

import numpy as np
import torch

from trainers.base_phase_trainer import BasePhaseTrainer
from trainers.incremental_phase_trainer import *
from trainers.trainer_helpers import TrainerHelper


class Trainer(TrainerHelper, BasePhaseTrainer, IncrementalPhaseTrainer):
    def __init__(self, model: Any, dataset: Any, args: Any) -> None:
        self.model = model
        self.dataset = dataset
        self.args = args
        self.device = torch.device(getattr(args, "device"))
        self.save_dir = str(getattr(args, "save_dir"))
        self.history: Dict[int, Dict[str, Any]] = {}
        self._phase_reference_backbone: Any | None = None
        self._phase_reference_classifier: Any | None = None
        self._phase_reference_phase: int | None = None

        self.model.to(self.device)
        contract = self.assert_dataset_contract()
        self.phase_schedule = {
            int(phase): [int(value) for value in classes]
            for phase, classes in contract["schedule"].items()
        }
        self.base_class_ids = list(self.phase_schedule[0])
        self.assert_model_contract()

    def _freeze_finalized_phase_snapshot(self, phase: int) -> None:
        phase_id = int(phase)
        latest = self._assert_finalized_state()
        if latest != phase_id:
            raise RuntimeError("can only snapshot the latest finalized phase")

        reference_backbone = copy.deepcopy(self.model.backbone).to(self.device)
        reference_backbone.eval()
        for parameter in reference_backbone.parameters():
            parameter.requires_grad_(False)
            parameter.grad = None

        reference_classifier = copy.deepcopy(self.model.classifier).to(self.device)
        reference_classifier.eval()
        for parameter in reference_classifier.parameters():
            parameter.requires_grad_(False)
            parameter.grad = None

        self._phase_reference_backbone = reference_backbone
        self._phase_reference_classifier = reference_classifier
        self._phase_reference_phase = phase_id

    def _require_phase_reference(self, phase: int) -> tuple[Any, Any]:
        phase_id = int(phase)
        if (
            self._phase_reference_backbone is None
            or self._phase_reference_classifier is None
            or self._phase_reference_phase != phase_id
        ):
            raise RuntimeError(
                f"missing frozen finalized model snapshot for phase {phase_id}"
            )
        return self._phase_reference_backbone, self._phase_reference_classifier

    @staticmethod
    def _array_fingerprint(value: Any) -> str:
        array = np.ascontiguousarray(np.asarray(value))
        digest = hashlib.sha256()
        digest.update(str(array.dtype).encode("utf-8"))
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(memoryview(array).cast("B"))
        return digest.hexdigest()

    def _split_fingerprint(self) -> str:
        protocol = getattr(self.dataset, "protocol", None)
        if not isinstance(protocol, Mapping):
            return ""
        split_by_class = protocol.get("split_by_class")
        if not isinstance(split_by_class, Mapping):
            return ""
        digest = hashlib.sha256()
        for raw_key in sorted(split_by_class, key=lambda key: int(key)):
            class_id = int(raw_key)
            row = split_by_class[raw_key]
            if not isinstance(row, Mapping):
                raise RuntimeError(
                    f"split membership for class {class_id} must be a mapping"
                )
            digest.update(np.asarray([class_id], dtype=np.int64).tobytes())
            for split in ("train", "val", "test"):
                values = np.ascontiguousarray(
                    np.asarray(row.get(split, ()), dtype=np.int64).reshape(-1)
                )
                digest.update(split.encode("ascii"))
                digest.update(np.asarray([values.size], dtype=np.int64).tobytes())
                digest.update(memoryview(values).cast("B"))
        return digest.hexdigest()

    def _dataset_identity(self) -> Dict[str, Any]:
        processed = getattr(self.dataset, "processed_cube", None)
        ordered = getattr(self.dataset, "ordered_spectral_cube", None)
        labels = getattr(self.dataset, "labels", None)
        coords = getattr(self.dataset, "coords", None)
        if any(value is None for value in (processed, ordered, labels, coords)):
            raise RuntimeError(
                "dataset must expose processed/ordered cubes, labels, and coordinates"
            )
        return {
            "protocol_seed": int(
                getattr(self.dataset, "protocol_seed", getattr(self.dataset, "seed", 0))
            ),
            "optimization_dataloader_seed": int(getattr(self.dataset, "seed", 0)),
            "phase_schedule": self.phase_schedule,
            "class_order_original_ids": [
                int(value)
                for value in getattr(
                    self.dataset,
                    "class_order_original_ids",
                    [
                        class_id
                        for phase in self.phase_schedule.values()
                        for class_id in phase
                    ],
                )
            ],
            "split_strategy": str(getattr(self.dataset, "split_strategy", "")),
            "patch_size": int(getattr(self.dataset, "patch_size")),
            "processed_cube_shape": [int(value) for value in processed.shape],
            "ordered_spectral_cube_shape": [int(value) for value in ordered.shape],
            "processed_cube_sha256": self._array_fingerprint(processed),
            "ordered_spectral_cube_sha256": self._array_fingerprint(ordered),
            "labels_sha256": self._array_fingerprint(labels),
            "coordinates_sha256": self._array_fingerprint(coords),
            "split_membership_sha256": self._split_fingerprint(),
        }

    @torch.no_grad()
    def _finalized_evaluation_fingerprint(self, phase: int) -> Dict[str, Any]:
        phase_id = int(phase)
        class_ids = self._seen_through_phase(phase_id)
        loader = self.dataset.get_reporting_dataloader(
            phase_id, batch_size=int(getattr(self.args, "eval_batch_size", 256))
        )
        metrics = self.evaluate_loader(
            loader, class_ids=class_ids, target_class_ids=class_ids
        )
        return {
            "phase": phase_id,
            "class_ids": [int(v) for v in class_ids],
            "confusion_matrix": [
                [int(v) for v in row] for row in metrics["confusion_matrix"]
            ],
            "overall_accuracy": float(metrics["overall_accuracy"]),
            "balanced_accuracy": float(metrics["balanced_accuracy"]),
            "minimum_class_accuracy": float(metrics["minimum_class_accuracy"]),
        }

    def _finalized_phases(self) -> list[int]:
        raw = getattr(self.dataset, "finalized_phases", None)
        if raw is None:
            raise RuntimeError("dataset must expose finalized_phases")
        phases = [int(value) for value in raw]
        if phases != list(range(len(phases))):
            raise RuntimeError("finalized phases must be contiguous from zero")
        return phases

    def _seen_through_phase(self, phase: int) -> list[int]:
        phase_id = int(phase)
        if phase_id not in self.phase_schedule:
            raise ValueError(f"unknown phase {phase_id}")
        return [
            class_id
            for index in range(phase_id + 1)
            for class_id in self.phase_schedule[index]
        ]

    def _assert_finalized_state(self) -> int:
        finalized = self._finalized_phases()
        if not finalized:
            raise RuntimeError("no phase is finalized")
        if getattr(self.dataset, "current_phase", None) is not None:
            raise RuntimeError("finalized state cannot retain an active phase")
        latest = finalized[-1]
        expected = self._seen_through_phase(latest)
        if list(self.model.committed_class_ids) != expected:
            raise RuntimeError("retained CDS disagrees with finalized protocol")
        self.model.validate_finalized_state()
        return latest

    def run_base_only(
        self, *, epochs: int, batch_size: int = 64, lr: float = 1e-4
    ) -> Dict[str, Any]:
        if (
            self._finalized_phases()
            or getattr(self.dataset, "current_phase", None) is not None
        ):
            raise RuntimeError("Phase 0 requires a fresh unfinalized dataset")
        if len(self.model.structure_bank) != 0:
            raise RuntimeError("Phase 0 requires an empty CDS bank")
        result = dict(
            self.train_base_phase(phase=0, epochs=epochs, batch_size=batch_size, lr=lr)
        )
        missing = set(self.BASE_RESULT_FIELDS) - set(result)
        if missing:
            raise RuntimeError(
                f"BasePhaseTrainer returned incomplete state: {sorted(missing)}"
            )
        if [int(value) for value in result["class_ids"]] != self.base_class_ids:
            raise RuntimeError("Phase-0 class IDs disagree with schedule")
        latest = self._assert_finalized_state()
        if latest != 0:
            raise RuntimeError("base run finalized an unexpected phase")
        self._freeze_finalized_phase_snapshot(0)
        self.history[0] = {
            "final_epoch": int(result["final_epoch"]),
            "epochs": list(result["history"]),
        }
        phase_dir = Path(self.save_dir) / "phase_0"
        phase_dir.mkdir(parents=True, exist_ok=True)
        result["checkpoint"] = self.save_checkpoint(phase_dir / "checkpoint.pth")
        return result

    def run_incremental_phase(
        self, *, phase: int, epochs: int, batch_size: int, lr: float
    ) -> Dict[str, Any]:
        phase_id = int(phase)
        if phase_id <= 0 or phase_id not in self.phase_schedule:
            raise ValueError("incremental phase must be a known phase > 0")
        if self._finalized_phases() != list(range(phase_id)):
            raise RuntimeError(
                f"phase {phase_id} requires finalized phases {list(range(phase_id))}"
            )
        if getattr(self.dataset, "current_phase", None) is not None:
            raise RuntimeError("dataset already has an active phase")
        previous = self._assert_finalized_state()
        if previous != phase_id - 1:
            raise RuntimeError(
                "incremental phase must start from the immediately preceding phase"
            )

        reference_backbone, reference_classifier = self._require_phase_reference(
            phase_id - 1
        )
        result = dict(
            self.train_incremental_phase(
                phase=phase_id,
                epochs=epochs,
                batch_size=batch_size,
                lr=lr,
                reference_backbone=reference_backbone,
                reference_classifier=reference_classifier,
            )
        )
        missing = set(self.INCREMENTAL_RESULT_FIELDS) - set(result)
        if missing:
            raise RuntimeError(
                f"IncrementalPhaseTrainer returned incomplete state: {sorted(missing)}"
            )
        latest = self._assert_finalized_state()
        if latest != phase_id:
            raise RuntimeError("incremental phase finalized the wrong protocol state")
        self._freeze_finalized_phase_snapshot(phase_id)
        self.history[phase_id] = {
            "final_epoch": int(result["final_epoch"]),
            "epochs": list(result["history"]),
            "stage3_whole_cds_decision": list(result["classifier_history"]),
            "method_state": dict(result["method_state"]),
        }
        phase_dir = Path(self.save_dir) / f"phase_{phase_id}"
        phase_dir.mkdir(parents=True, exist_ok=True)
        result["checkpoint"] = self.save_checkpoint(phase_dir / "checkpoint.pth")
        return result

    def save_checkpoint(self, path: str | os.PathLike[str]) -> str:
        latest = self._assert_finalized_state()
        destination = Path(path).resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(destination.name + ".tmp")
        payload = {
            "model_state": self.model.state_dict(),
            "dataset_identity": self._dataset_identity(),
            "finalized_phases": list(range(latest + 1)),
            "trainer_history": self.history,
            "evaluation_fingerprint": self._finalized_evaluation_fingerprint(latest),
        }
        try:
            torch.save(payload, temporary)
            os.replace(temporary, destination)
        except Exception:
            if temporary.exists():
                temporary.unlink()
            raise
        return str(destination)

    def load_checkpoint(self, path: str | os.PathLike[str]) -> Dict[str, Any]:
        source = Path(path).resolve()
        if not source.is_file():
            raise FileNotFoundError(source)
        if (
            getattr(self.dataset, "current_phase", None) is not None
            or self._finalized_phases()
        ):
            raise RuntimeError(
                "load checkpoint into a fresh unfinalized dataset manager"
            )
        if len(self.model.structure_bank) != 0:
            raise RuntimeError("load checkpoint into a model with empty CDS state")

        payload = torch.load(source, map_location=self.device)
        if not isinstance(payload, Mapping):
            raise RuntimeError("checkpoint payload must be a mapping")
        required = {"model_state", "dataset_identity", "finalized_phases"}
        missing = required - set(payload)
        if missing:
            raise RuntimeError(f"checkpoint is missing fields {sorted(missing)}")
        if payload["dataset_identity"] != self._dataset_identity():
            raise RuntimeError(
                "checkpoint dataset/split identity does not match this run"
            )

        finalized = [int(value) for value in payload["finalized_phases"]]
        if not finalized or finalized != list(range(len(finalized))):
            raise RuntimeError(
                "checkpoint finalized phases must be contiguous from zero"
            )
        model_state = payload["model_state"]
        if not isinstance(model_state, Mapping):
            raise RuntimeError("checkpoint model_state must be a mapping")
        # Loading is strict so the checkpoint must match the current model,
        # single deployed classifier W and persistent Whole-CDS schema.
        self.model.load_state_dict(model_state, strict=True)

        restore = getattr(self.dataset, "restore_finalized_phases", None)
        if not callable(restore):
            raise RuntimeError("dataset manager must expose restore_finalized_phases()")
        restore(finalized)
        raw_history = payload.get("trainer_history", {})
        if not isinstance(raw_history, Mapping):
            raise RuntimeError("checkpoint trainer_history must be a mapping")
        self.history = {int(phase): dict(value) for phase, value in raw_history.items()}

        self.model.eval()
        latest = self._assert_finalized_state()
        if latest != finalized[-1]:
            raise RuntimeError("restored model/protocol phase mismatch")
        # Behavioral resume gate: protocol hashes alone cannot detect a model/eval
        # mismatch. New checkpoints therefore require exact confusion reproduction.
        actual_fingerprint = self._finalized_evaluation_fingerprint(latest)
        stored_fingerprint = payload.get("evaluation_fingerprint")
        evaluation_verified = False
        if stored_fingerprint is not None:
            if not isinstance(stored_fingerprint, Mapping):
                raise RuntimeError(
                    "checkpoint evaluation_fingerprint must be a mapping"
                )
            expected_ids = [int(v) for v in stored_fingerprint.get("class_ids", ())]
            expected_confusion = stored_fingerprint.get("confusion_matrix")
            if expected_ids != actual_fingerprint["class_ids"]:
                raise RuntimeError("checkpoint evaluation class IDs do not reproduce")
            if expected_confusion != actual_fingerprint["confusion_matrix"]:
                raise RuntimeError(
                    "checkpoint model does not reproduce its finalized confusion matrix before incremental setup"
                )
            evaluation_verified = True
            print(
                f"[Checkpoint verified] phase={latest} "
                f"OA={100*actual_fingerprint['overall_accuracy']:.2f}% "
                f"BA={100*actual_fingerprint['balanced_accuracy']:.2f}% "
                f"Min={100*actual_fingerprint['minimum_class_accuracy']:.2f}%"
            )
        else:
            print(
                f"[Legacy checkpoint loaded] phase={latest} "
                f"OA={100*actual_fingerprint['overall_accuracy']:.2f}% "
                f"BA={100*actual_fingerprint['balanced_accuracy']:.2f}% "
                f"Min={100*actual_fingerprint['minimum_class_accuracy']:.2f}% "
            )

        self._freeze_finalized_phase_snapshot(latest)
        return {
            "checkpoint": str(source),
            "finalized_phases": list(finalized),
            "committed_class_ids": list(self.model.committed_class_ids),
            "next_phase": latest + 1 if latest + 1 in self.phase_schedule else None,
            "evaluation_verified": evaluation_verified,
            "resume_evaluation": actual_fingerprint,
        }


__all__ = ["Trainer"]


# from __future__ import annotations

# """Top-level orchestration and checkpointing for HSI CDS NECIL."""

# import copy
# import hashlib
# import os
# from pathlib import Path
# from typing import Any, Dict, Mapping

# import numpy as np
# import torch

# from trainers.base_phase_trainer import BasePhaseTrainer
# from trainers.incremental_phase_trainer import IncrementalPhaseTrainer
# from trainers.trainer_helpers import TrainerHelper


# class Trainer(TrainerHelper, BasePhaseTrainer, IncrementalPhaseTrainer):
#     def __init__(self, model: Any, dataset: Any, args: Any) -> None:
#         self.model = model
#         self.dataset = dataset
#         self.args = args
#         self.device = torch.device(getattr(args, "device"))
#         self.save_dir = str(getattr(args, "save_dir"))
#         self.history: Dict[int, Dict[str, Any]] = {}
#         self._phase_reference_backbone: Any | None = None
#         self._phase_reference_classifier: Any | None = None
#         self._phase_reference_phase: int | None = None

#         self.model.to(self.device)
#         contract = self.assert_dataset_contract()
#         self.phase_schedule = {
#             int(phase): [int(value) for value in classes]
#             for phase, classes in contract["schedule"].items()
#         }
#         self.base_class_ids = list(self.phase_schedule[0])
#         self.assert_model_contract()

#     def _freeze_finalized_phase_snapshot(self, phase: int) -> None:
#         """Freeze the finalized previous backbone and classifier for the next phase.

#         The snapshot is temporary process state, not persistent exemplar memory.
#         Checkpoints continue to store only the model and persistent CDS state.
#         """
#         phase_id = int(phase)
#         latest = self._assert_finalized_state()
#         if latest != phase_id:
#             raise RuntimeError("can only snapshot the latest finalized phase")

#         reference_backbone = copy.deepcopy(self.model.backbone).to(self.device)
#         reference_backbone.eval()
#         for parameter in reference_backbone.parameters():
#             parameter.requires_grad_(False)
#             parameter.grad = None

#         reference_classifier = copy.deepcopy(self.model.classifier).to(self.device)
#         reference_classifier.eval()
#         for parameter in reference_classifier.parameters():
#             parameter.requires_grad_(False)
#             parameter.grad = None

#         self._phase_reference_backbone = reference_backbone
#         self._phase_reference_classifier = reference_classifier
#         self._phase_reference_phase = phase_id

#     def _require_phase_reference(self, phase: int) -> tuple[Any, Any]:
#         phase_id = int(phase)
#         if (
#             self._phase_reference_backbone is None
#             or self._phase_reference_classifier is None
#             or self._phase_reference_phase != phase_id
#         ):
#             raise RuntimeError(f"missing frozen finalized model snapshot for phase {phase_id}")
#         return self._phase_reference_backbone, self._phase_reference_classifier

#     @staticmethod
#     def _array_fingerprint(value: Any) -> str:
#         array = np.ascontiguousarray(np.asarray(value))
#         digest = hashlib.sha256()
#         digest.update(str(array.dtype).encode("utf-8"))
#         digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
#         digest.update(memoryview(array).cast("B"))
#         return digest.hexdigest()

#     def _split_fingerprint(self) -> str:
#         protocol = getattr(self.dataset, "protocol", None)
#         if not isinstance(protocol, Mapping):
#             return ""
#         split_by_class = protocol.get("split_by_class")
#         if not isinstance(split_by_class, Mapping):
#             return ""
#         digest = hashlib.sha256()
#         for raw_key in sorted(split_by_class, key=lambda key: int(key)):
#             class_id = int(raw_key)
#             row = split_by_class[raw_key]
#             if not isinstance(row, Mapping):
#                 raise RuntimeError(f"split membership for class {class_id} must be a mapping")
#             digest.update(np.asarray([class_id], dtype=np.int64).tobytes())
#             for split in ("train", "val", "test"):
#                 values = np.ascontiguousarray(np.asarray(row.get(split, ()), dtype=np.int64).reshape(-1))
#                 digest.update(split.encode("ascii"))
#                 digest.update(np.asarray([values.size], dtype=np.int64).tobytes())
#                 digest.update(memoryview(values).cast("B"))
#         return digest.hexdigest()

#     def _dataset_identity(self) -> Dict[str, Any]:
#         processed = getattr(self.dataset, "processed_cube", None)
#         ordered = getattr(self.dataset, "ordered_spectral_cube", None)
#         labels = getattr(self.dataset, "labels", None)
#         coords = getattr(self.dataset, "coords", None)
#         if any(value is None for value in (processed, ordered, labels, coords)):
#             raise RuntimeError("dataset must expose processed/ordered cubes, labels, and coordinates")
#         return {
#             "protocol_seed": int(getattr(self.dataset, "protocol_seed", getattr(self.dataset, "seed", 0))),
#             "optimization_dataloader_seed": int(getattr(self.dataset, "seed", 0)),
#             "phase_schedule": self.phase_schedule,
#             "class_order_original_ids": [
#                 int(value)
#                 for value in getattr(
#                     self.dataset,
#                     "class_order_original_ids",
#                     [class_id for phase in self.phase_schedule.values() for class_id in phase],
#                 )
#             ],
#             "split_strategy": str(getattr(self.dataset, "split_strategy", "")),
#             "patch_size": int(getattr(self.dataset, "patch_size")),
#             "processed_cube_shape": [int(value) for value in processed.shape],
#             "ordered_spectral_cube_shape": [int(value) for value in ordered.shape],
#             "processed_cube_sha256": self._array_fingerprint(processed),
#             "ordered_spectral_cube_sha256": self._array_fingerprint(ordered),
#             "labels_sha256": self._array_fingerprint(labels),
#             "coordinates_sha256": self._array_fingerprint(coords),
#             "split_membership_sha256": self._split_fingerprint(),
#         }

#     def _finalized_phases(self) -> list[int]:
#         raw = getattr(self.dataset, "finalized_phases", None)
#         if raw is None:
#             raise RuntimeError("dataset must expose finalized_phases")
#         phases = [int(value) for value in raw]
#         if phases != list(range(len(phases))):
#             raise RuntimeError("finalized phases must be contiguous from zero")
#         return phases

#     def _seen_through_phase(self, phase: int) -> list[int]:
#         phase_id = int(phase)
#         if phase_id not in self.phase_schedule:
#             raise ValueError(f"unknown phase {phase_id}")
#         return [
#             class_id
#             for index in range(phase_id + 1)
#             for class_id in self.phase_schedule[index]
#         ]

#     def _assert_finalized_state(self) -> int:
#         finalized = self._finalized_phases()
#         if not finalized:
#             raise RuntimeError("no phase is finalized")
#         if getattr(self.dataset, "current_phase", None) is not None:
#             raise RuntimeError("finalized state cannot retain an active phase")
#         latest = finalized[-1]
#         expected = self._seen_through_phase(latest)
#         if list(self.model.committed_class_ids) != expected:
#             raise RuntimeError("retained CDS disagrees with finalized protocol")
#         self.model.validate_finalized_state()
#         return latest

#     def run_base_only(self, *, epochs: int, batch_size: int = 64, lr: float = 1e-4) -> Dict[str, Any]:
#         if self._finalized_phases() or getattr(self.dataset, "current_phase", None) is not None:
#             raise RuntimeError("Phase 0 requires a fresh unfinalized dataset")
#         if len(self.model.structure_bank) != 0:
#             raise RuntimeError("Phase 0 requires an empty CDS bank")
#         result = dict(self.train_base_phase(phase=0, epochs=epochs, batch_size=batch_size, lr=lr))
#         missing = set(self.BASE_RESULT_FIELDS) - set(result)
#         if missing:
#             raise RuntimeError(f"BasePhaseTrainer returned incomplete state: {sorted(missing)}")
#         if [int(value) for value in result["class_ids"]] != self.base_class_ids:
#             raise RuntimeError("Phase-0 class IDs disagree with schedule")
#         latest = self._assert_finalized_state()
#         if latest != 0:
#             raise RuntimeError("base run finalized an unexpected phase")
#         self._freeze_finalized_phase_snapshot(0)
#         self.history[0] = {"final_epoch": int(result["final_epoch"]), "epochs": list(result["history"])}
#         phase_dir = Path(self.save_dir) / "phase_0"
#         phase_dir.mkdir(parents=True, exist_ok=True)
#         result["checkpoint"] = self.save_checkpoint(phase_dir / "checkpoint.pth")
#         return result

#     def run_incremental_phase(self, *, phase: int, epochs: int, batch_size: int, lr: float) -> Dict[str, Any]:
#         phase_id = int(phase)
#         if phase_id <= 0 or phase_id not in self.phase_schedule:
#             raise ValueError("incremental phase must be a known phase > 0")
#         if self._finalized_phases() != list(range(phase_id)):
#             raise RuntimeError(f"phase {phase_id} requires finalized phases {list(range(phase_id))}")
#         if getattr(self.dataset, "current_phase", None) is not None:
#             raise RuntimeError("dataset already has an active phase")
#         previous = self._assert_finalized_state()
#         if previous != phase_id - 1:
#             raise RuntimeError("incremental phase must start from the immediately preceding phase")

#         reference_backbone, reference_classifier = self._require_phase_reference(phase_id - 1)
#         result = dict(
#             self.train_incremental_phase(
#                 phase=phase_id,
#                 epochs=epochs,
#                 batch_size=batch_size,
#                 lr=lr,
#                 reference_backbone=reference_backbone,
#                 reference_classifier=reference_classifier,
#             )
#         )
#         missing = set(self.INCREMENTAL_RESULT_FIELDS) - set(result)
#         if missing:
#             raise RuntimeError(f"IncrementalPhaseTrainer returned incomplete state: {sorted(missing)}")
#         latest = self._assert_finalized_state()
#         if latest != phase_id:
#             raise RuntimeError("incremental phase finalized the wrong protocol state")
#         self._freeze_finalized_phase_snapshot(phase_id)
#         self.history[phase_id] = {
#             "final_epoch": int(result["final_epoch"]),
#             "epochs": list(result["history"]),
#             "method_state": dict(result["method_state"]),
#         }
#         phase_dir = Path(self.save_dir) / f"phase_{phase_id}"
#         phase_dir.mkdir(parents=True, exist_ok=True)
#         result["checkpoint"] = self.save_checkpoint(phase_dir / "checkpoint.pth")
#         return result

#     def save_checkpoint(self, path: str | os.PathLike[str]) -> str:
#         latest = self._assert_finalized_state()
#         destination = Path(path).resolve()
#         destination.parent.mkdir(parents=True, exist_ok=True)
#         temporary = destination.with_name(destination.name + ".tmp")
#         payload = {
#             "model_state": self.model.state_dict(),
#             "dataset_identity": self._dataset_identity(),
#             "finalized_phases": list(range(latest + 1)),
#             "trainer_history": self.history,
#         }
#         try:
#             torch.save(payload, temporary)
#             os.replace(temporary, destination)
#         except Exception:
#             if temporary.exists():
#                 temporary.unlink()
#             raise
#         return str(destination)

#     def load_checkpoint(self, path: str | os.PathLike[str]) -> Dict[str, Any]:
#         source = Path(path).resolve()
#         if not source.is_file():
#             raise FileNotFoundError(source)
#         if getattr(self.dataset, "current_phase", None) is not None or self._finalized_phases():
#             raise RuntimeError("load checkpoint into a fresh unfinalized dataset manager")
#         if len(self.model.structure_bank) != 0:
#             raise RuntimeError("load checkpoint into a model with empty CDS state")

#         payload = torch.load(source, map_location=self.device)
#         if not isinstance(payload, Mapping):
#             raise RuntimeError("checkpoint payload must be a mapping")
#         required = {"model_state", "dataset_identity", "finalized_phases"}
#         missing = required - set(payload)
#         if missing:
#             raise RuntimeError(f"checkpoint is missing fields {sorted(missing)}")
#         if payload["dataset_identity"] != self._dataset_identity():
#             raise RuntimeError("checkpoint dataset/split identity does not match this run")

#         finalized = [int(value) for value in payload["finalized_phases"]]
#         if not finalized or finalized != list(range(len(finalized))):
#             raise RuntimeError("checkpoint finalized phases must be contiguous from zero")
#         model_state = payload["model_state"]
#         if not isinstance(model_state, Mapping):
#             raise RuntimeError("checkpoint model_state must be a mapping")
#         # Loading is strict so the checkpoint must match the current model and CDS schema.
#         self.model.load_state_dict(model_state, strict=True)

#         restore = getattr(self.dataset, "restore_finalized_phases", None)
#         if not callable(restore):
#             raise RuntimeError("dataset manager must expose restore_finalized_phases()")
#         restore(finalized)
#         raw_history = payload.get("trainer_history", {})
#         if not isinstance(raw_history, Mapping):
#             raise RuntimeError("checkpoint trainer_history must be a mapping")
#         self.history = {int(phase): dict(value) for phase, value in raw_history.items()}

#         self.model.eval()
#         latest = self._assert_finalized_state()
#         if latest != finalized[-1]:
#             raise RuntimeError("restored model/protocol phase mismatch")
#         self._freeze_finalized_phase_snapshot(latest)
#         return {
#             "checkpoint": str(source),
#             "finalized_phases": list(finalized),
#             "committed_class_ids": list(self.model.committed_class_ids),
#             "next_phase": latest + 1 if latest + 1 in self.phase_schedule else None,
#         }


# __all__ = ["Trainer"]
