from __future__ import annotations

"""Top-level orchestration for the locked SCOPE-HSI training protocol.

Role
----
The top-level Trainer owns phase ordering and finalized checkpoint state only.
It does not implement feature learning, class statistics, coordinate transport,
or classifier update mathematics.  Those stay in the phase trainers/model.

A finalized SCOPE phase must satisfy one common-coordinate invariant:

    classifier class IDs == committed {I,S,N} class IDs == seen protocol IDs.

Only finalized phase-boundary checkpoints are saved.  Therefore no optimizer or
temporary before/after current-sample Q collection belongs in a checkpoint.
"""

import hashlib
import os
from pathlib import Path
from typing import Any, Dict, Mapping

import numpy as np
import torch

from trainers.base_phase_trainer import BasePhaseTrainer
# from trainers.incremental_phase_trainer import IncrementalPhaseTrainer
from trainers.trainer_helpers import TrainerHelper


class Trainer(TrainerHelper, BasePhaseTrainer,):
    """Coordinate base/incremental phases around one persistent SCOPE model."""

    def __init__(
        self,
        model: torch.nn.Module,
        dataset: Any,
        args: Any,
    ) -> None:
        self.model = model
        self.dataset = dataset
        self.args = args

        configured_device = getattr(args, "device", None)
        requested = (
            str(configured_device)
            if configured_device is not None
            else ("cuda:0" if torch.cuda.is_available() else "cpu")
        )
        try:
            self.device = torch.device(requested)
        except (TypeError, RuntimeError) as exc:
            raise ValueError(f"invalid training device {requested!r}") from exc

        if self.device.type == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError(
                    f"CUDA device {self.device} was requested but unavailable"
                )
            index = 0 if self.device.index is None else int(self.device.index)
            if not 0 <= index < torch.cuda.device_count():
                raise RuntimeError(
                    f"CUDA device index {index} is unavailable; "
                    f"device_count={torch.cuda.device_count()}"
                )

        self.model.to(self.device)

        self.save_dir = os.path.abspath(
            str(getattr(args, "save_dir", "./outputs"))
        )
        os.makedirs(self.save_dir, exist_ok=True)

        dataset_contract = self.assert_dataset_contract()
        self.assert_model_contract()

        self.phase_schedule = {
            int(phase): [int(v) for v in classes]
            for phase, classes in dataset_contract["schedule"].items()
        }
        self.base_class_ids = [
            int(v) for v in dataset_contract["base_classes"]
        ]
        self.history: Dict[int, Dict[str, Any]] = {}

    # ------------------------------------------------------------------
    # Experiment identity
    # ------------------------------------------------------------------

    @staticmethod
    def _array_fingerprint(value: Any) -> str:
        array = np.ascontiguousarray(np.asarray(value))
        digest = hashlib.sha256()
        digest.update(array.dtype.str.encode("ascii"))
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(memoryview(array).cast("B"))
        return digest.hexdigest()

    def _split_fingerprint(self) -> str:
        split_by_class = getattr(self.dataset, "_split_by_class", None)
        if not isinstance(split_by_class, Mapping):
            raise RuntimeError(
                "dataset must expose validated split membership in _split_by_class"
            )

        digest = hashlib.sha256()
        normalized_keys = []
        for raw_key in split_by_class:
            class_id = self._exact_nonnegative_int(raw_key, name="split class_id")
            normalized_keys.append((class_id, raw_key))
        normalized_keys.sort(key=lambda item: item[0])
        if len({class_id for class_id, _ in normalized_keys}) != len(normalized_keys):
            raise RuntimeError("split membership contains duplicate normalized class IDs")

        for class_id, raw_key in normalized_keys:
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
                digest.update(
                    np.asarray([values.size], dtype=np.int64).tobytes()
                )
                digest.update(memoryview(values).cast("B"))
        return digest.hexdigest()

    def _dataset_identity(self) -> Dict[str, Any]:
        processed = getattr(self.dataset, "processed_cube", None)
        ordered = getattr(self.dataset, "ordered_spectral_cube", None)
        labels = getattr(self.dataset, "labels", None)
        coords = getattr(self.dataset, "coords", None)
        if (
            processed is None
            or ordered is None
            or labels is None
            or coords is None
        ):
            raise RuntimeError(
                "dataset must expose processed/ordered cubes, labels, and coords"
            )

        return {
            "seed": int(getattr(self.dataset, "seed", 0)),
            "phase_schedule": self.phase_schedule,
            "class_order_original_ids": [
                int(v)
                for v in getattr(
                    self.dataset,
                    "class_order_original_ids",
                    [
                        c
                        for phase in self.phase_schedule.values()
                        for c in phase
                    ],
                )
            ],
            "split_strategy": str(
                getattr(self.dataset, "split_strategy", "")
            ),
            "spatial_partition_mode": str(
                getattr(self.dataset, "spatial_partition_mode", "")
            ),
            "context_policy": str(
                getattr(self.dataset, "context_policy", "")
            ),
            "patch_size": int(getattr(self.dataset, "patch_size")),
            "validation_enabled": bool(
                getattr(self.dataset, "validation_enabled", False)
            ),
            "require_patch_disjoint": bool(
                getattr(self.dataset, "require_patch_disjoint", False)
            ),
            "require_phase_context_isolation": bool(
                getattr(self.dataset, "require_phase_context_isolation", False)
            ),
            "processed_cube_shape": [int(v) for v in processed.shape],
            "ordered_spectral_cube_shape": [int(v) for v in ordered.shape],
            "processed_cube_sha256": self._array_fingerprint(processed),
            "ordered_spectral_cube_sha256": self._array_fingerprint(ordered),
            "labels_sha256": self._array_fingerprint(labels),
            "coordinates_sha256": self._array_fingerprint(coords),
            "split_membership_sha256": self._split_fingerprint(),
        }

    # ------------------------------------------------------------------
    # Phase state
    # ------------------------------------------------------------------

    def _finalized_phases(self) -> list[int]:
        raw = getattr(self.dataset, "finalized_phases", None)
        if raw is None:
            raise RuntimeError("dataset must expose finalized_phases")

        phases = [int(v) for v in raw]
        if phases != list(range(len(phases))):
            raise RuntimeError(
                "finalized phases must be contiguous from zero"
            )
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
            raise RuntimeError(
                "finalized state cannot retain an active phase"
            )

        latest = finalized[-1]
        expected = self._seen_through_phase(latest)

        if list(self.model.committed_class_ids) != expected:
            raise RuntimeError(
                "retained class structure disagrees with finalized protocol"
            )
        if list(self.model.classifier_class_ids) != expected:
            raise RuntimeError(
                "persistent classifier rows disagree with finalized protocol"
            )

        # Stronger than validate_model_state(): at a phase boundary classifier
        # and class statistics must contain exactly the same classes/order.
        self.model.validate_finalized_state()
        return latest

    # ------------------------------------------------------------------
    # Training dispatch
    # ------------------------------------------------------------------

    def run_base_only(
        self,
        *,
        epochs: int,
        batch_size: int = 64,
        lr: float = 1e-4,
    ) -> Dict[str, Any]:
        if self._finalized_phases():
            raise RuntimeError("phase 0 requires an unfinalized dataset")
        if getattr(self.dataset, "current_phase", None) is not None:
            raise RuntimeError("dataset already has an active phase")
        if len(self.model.geometry_bank) != 0:
            raise RuntimeError("phase 0 requires empty class statistics")
        if self.model.classifier.class_count != 0:
            raise RuntimeError("phase 0 requires an empty persistent classifier")

        result = dict(
            self.train_base_phase(
                phase=0,
                epochs=epochs,
                batch_size=batch_size,
                lr=lr,
            )
        )

        missing = set(self.BASE_RESULT_FIELDS) - set(result)
        if missing:
            raise RuntimeError(
                f"BasePhaseTrainer returned incomplete state: {sorted(missing)}"
            )
        if [int(v) for v in result["class_ids"]] != self.base_class_ids:
            raise RuntimeError(
                "phase-0 trainer class IDs disagree with schedule"
            )
        if (
            result["geometry_committed"] is not True
            or result["phase_finalized"] is not True
        ):
            raise RuntimeError("phase 0 did not finalize cleanly")

        latest = self._assert_finalized_state()
        if latest != 0:
            raise RuntimeError("base run finalized an unexpected phase")

        self.history[0] = {
            "final_epoch": int(result["final_epoch"]),
            "epochs": list(result["history"]),
        }

        phase_dir = Path(self.save_dir) / "phase_0"
        phase_dir.mkdir(parents=True, exist_ok=True)
        result["checkpoint"] = self.save_checkpoint(
            phase_dir / "checkpoint.pth"
        )
        return result

    def run_incremental_phase(
        self,
        *,
        phase: int,
        epochs: int,
        batch_size: int,
        lr: float,
    ) -> Dict[str, Any]:
        phase_id = int(phase)
        if phase_id <= 0 or phase_id not in self.phase_schedule:
            raise ValueError(
                "incremental phase must be a known phase > 0"
            )
        if self._finalized_phases() != list(range(phase_id)):
            raise RuntimeError(
                f"phase {phase_id} requires finalized phases "
                f"{list(range(phase_id))}"
            )
        if getattr(self.dataset, "current_phase", None) is not None:
            raise RuntimeError("dataset already has an active phase")

        previous_latest = self._assert_finalized_state()
        if previous_latest != phase_id - 1:
            raise RuntimeError(
                "incremental phase does not start from the immediately "
                "preceding finalized phase"
            )

        old_ids = self._seen_through_phase(phase_id - 1)
        new_ids = list(self.phase_schedule[phase_id])

        result = dict(
            self.train_incremental_phase(
                phase=phase_id,
                epochs=epochs,
                batch_size=batch_size,
                lr=lr,
            )
        )

        missing = set(self.INCREMENTAL_RESULT_FIELDS) - set(result)
        if missing:
            raise RuntimeError(
                "IncrementalPhaseTrainer returned incomplete state: "
                f"{sorted(missing)}"
            )
        if (
            result["geometry_committed"] is not True
            or result["phase_finalized"] is not True
        ):
            raise RuntimeError(
                "incremental phase did not finalize cleanly"
            )

        # Validate class identity if the incremental result exposes these fields.
        if "old_class_ids" in result and [
            int(v) for v in result["old_class_ids"]
        ] != old_ids:
            raise RuntimeError("incremental trainer old class IDs are incorrect")
        if "new_class_ids" in result and [
            int(v) for v in result["new_class_ids"]
        ] != new_ids:
            raise RuntimeError("incremental trainer new class IDs are incorrect")

        latest = self._assert_finalized_state()
        if latest != phase_id:
            raise RuntimeError(
                "incremental phase finalized the wrong protocol state"
            )

        self.history[phase_id] = {
            "final_epoch": int(result["final_epoch"]),
            "epochs": list(result["history"]),
        }

        phase_dir = Path(self.save_dir) / f"phase_{phase_id}"
        phase_dir.mkdir(parents=True, exist_ok=True)
        result["checkpoint"] = self.save_checkpoint(
            phase_dir / "checkpoint.pth"
        )
        return result

    # ------------------------------------------------------------------
    # Checkpoint I/O
    # ------------------------------------------------------------------

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
        }

        try:
            torch.save(payload, temporary)
            os.replace(temporary, destination)
        except Exception:
            if temporary.exists():
                temporary.unlink()
            raise
        return str(destination)

    def load_checkpoint(
        self,
        path: str | os.PathLike[str],
    ) -> Dict[str, Any]:
        source = Path(path).resolve()
        if not source.is_file():
            raise FileNotFoundError(source)

        if getattr(self.dataset, "current_phase", None) is not None:
            raise RuntimeError(
                "cannot load a checkpoint while a phase is active"
            )
        if self._finalized_phases():
            raise RuntimeError(
                "load checkpoint into a fresh unfinalized dataset manager"
            )
        if len(self.model.geometry_bank) != 0:
            raise RuntimeError(
                "load checkpoint into a model with empty class statistics"
            )
        if self.model.classifier.class_count != 0:
            raise RuntimeError(
                "load checkpoint into a model with an empty classifier"
            )

        payload = torch.load(source, map_location=self.device)
        if not isinstance(payload, Mapping):
            raise RuntimeError("checkpoint payload must be a mapping")

        required = {"model_state", "dataset_identity", "finalized_phases"}
        missing = required - set(payload)
        if missing:
            raise RuntimeError(
                f"checkpoint is missing fields {sorted(missing)}"
            )

        if payload["dataset_identity"] != self._dataset_identity():
            raise RuntimeError(
                "checkpoint dataset/split identity does not match this run"
            )

        finalized = [int(v) for v in payload["finalized_phases"]]
        if not finalized or finalized != list(range(len(finalized))):
            raise RuntimeError(
                "checkpoint finalized phases must be contiguous from zero"
            )
        if finalized[-1] not in self.phase_schedule:
            raise RuntimeError(
                "checkpoint references a phase outside this schedule"
            )

        try:
            self.model.load_state_dict(
                payload["model_state"],
                strict=True,
            )
        except RuntimeError as exc:
            raise RuntimeError(
                "checkpoint model state is incompatible with the current "
                "locked SCOPE model"
            ) from exc

        restore_finalized = getattr(
            self.dataset,
            "restore_finalized_phases",
            None,
        )
        if not callable(restore_finalized):
            raise RuntimeError(
                "dataset manager must expose restore_finalized_phases() "
                "for checkpoint restore"
            )
        restore_finalized(finalized)

        raw_history = payload.get("trainer_history", {})
        if not isinstance(raw_history, Mapping):
            raise RuntimeError("checkpoint trainer_history must be a mapping")
        self.history = {
            int(phase): dict(value)
            for phase, value in raw_history.items()
        }

        self.model.eval()
        latest = self._assert_finalized_state()
        if latest != finalized[-1]:
            raise RuntimeError(
                "restored model/protocol phase mismatch"
            )

        return {
            "checkpoint": str(source),
            "finalized_phases": list(finalized),
            "committed_class_ids": list(self.model.committed_class_ids),
            "classifier_class_ids": list(self.model.classifier_class_ids),
            "next_phase": (
                latest + 1
                if latest + 1 in self.phase_schedule
                else None
            ),
        }


__all__ = ["Trainer"]
