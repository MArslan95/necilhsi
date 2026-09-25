from __future__ import annotations

"""Reporting-only qualitative maps for finalized full-CDS NECIL-HSI phases."""

from contextlib import contextmanager
from numbers import Integral, Real
from typing import Any, Dict, Mapping
import math
import os

import numpy as np
import torch

from models.classifier import CosineClassificationHead
from models.geometry_bank import HSIClassDiscriminativeStructureBank
from utils.visualize import save_full_phase_qualitative_maps


def _nonnegative_int(value: object, *, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer")
    if isinstance(value, Integral):
        result = int(value)
    elif isinstance(value, Real):
        number = float(value)
        if not math.isfinite(number) or not number.is_integer():
            raise ValueError(f"{name} must be a finite integer")
        result = int(number)
    else:
        raise ValueError(f"{name} must be an integer")
    if result < 0:
        raise ValueError(f"{name} must be non-negative")
    return result


@contextmanager
def _temporary_eval_state(model: Any):
    states = [(module, bool(module.training)) for module in model.modules()]
    try:
        model.eval()
        yield
    finally:
        for module, state in states:
            module.training = state


@torch.no_grad()
def generate_phase_qualitative_maps(
    *, model: Any, dataset: Any, phase: int, output_dir: str,
    device: str | torch.device, batch_size: int = 256,
    cmap_name: str = "nipy_spectral", dpi: int = 300,
) -> Dict[str, str]:
    phase_id = _nonnegative_int(phase, name="phase")
    batch_size_value = _nonnegative_int(batch_size, name="batch_size")
    dpi_value = _nonnegative_int(dpi, name="dpi")
    if batch_size_value == 0 or dpi_value == 0:
        raise ValueError("batch_size and dpi must be positive")

    evaluation_device = torch.device(device)
    if torch.device(model.device) != evaluation_device:
        raise RuntimeError("model and reporting device disagree")

    model.validate_finalized_state()
    if not isinstance(model.structure_bank, HSIClassDiscriminativeStructureBank):
        raise RuntimeError("model.structure_bank has wrong type")
    if not isinstance(model.classifier, CosineClassificationHead):
        raise RuntimeError("model.classifier has wrong type")

    finalized = [int(value) for value in getattr(dataset, "finalized_phases", [])]
    if phase_id not in finalized or getattr(dataset, "current_phase", None) is not None:
        raise RuntimeError("qualitative reporting requires a finalized inactive phase")

    class_ids = [int(value) for value in dataset.get_seen_classes(phase_id)]
    if list(model.committed_class_ids) != class_ids:
        raise RuntimeError("committed CDS does not match phase classes")

    phase_map = getattr(dataset, "phase_to_classes", None)
    if not isinstance(phase_map, Mapping):
        raise RuntimeError("dataset must expose phase_to_classes")
    phase_groups = {
        current_phase: [int(value) for value in phase_map[current_phase]]
        for current_phase in range(phase_id + 1)
    }
    if [class_id for current_phase in range(phase_id + 1) for class_id in phase_groups[current_phase]] != class_ids:
        raise RuntimeError("phase_to_classes does not reproduce visible class order")

    loader = dataset.get_map_dataloader(phase_id, batch_size=batch_size_value)
    coordinate_rows: list[torch.Tensor] = []
    target_rows: list[torch.Tensor] = []
    prediction_rows: list[torch.Tensor] = []
    margin_rows: list[torch.Tensor] = []

    with _temporary_eval_state(model):
        for batch in loader:
            if not isinstance(batch, Mapping):
                raise TypeError("map-reporting batches must be mappings")
            missing = {"image", "raw_center_spectrum", "label", "coord"} - set(batch)
            if missing:
                raise KeyError(f"map-reporting batch missing {sorted(missing)}")

            patch = torch.as_tensor(
                batch["image"],
                device=evaluation_device,
                dtype=model.dtype,
            )
            spectrum = torch.as_tensor(
                batch["raw_center_spectrum"],
                device=evaluation_device,
                dtype=model.dtype,
            )
            labels = torch.as_tensor(
                batch["label"],
                device=evaluation_device,
                dtype=torch.long,
            ).flatten()
            coordinates = torch.as_tensor(
                batch["coord"],
                dtype=torch.long,
            ).detach().cpu()

            if labels.shape != (patch.size(0),) or coordinates.shape != (patch.size(0), 2):
                raise RuntimeError("map-reporting rows are batch-misaligned")

            ordered_features = model.extract_features(
                patch,
                center_spectrum=spectrum,
                return_aux=False,
            ).ordered_features
            output = model.classify_features(
                ordered_features,
                class_ids=class_ids,
            )

            matches = labels.unsqueeze(1).eq(output.class_ids.unsqueeze(0))
            if not bool(matches.sum(dim=1).eq(1).all()):
                raise RuntimeError("could not map labels to classifier columns")

            local_targets = matches.to(torch.long).argmax(dim=1)
            rows = torch.arange(labels.numel(), device=labels.device)
            target_logits = output.logits[rows, local_targets]
            rival_logits = output.logits.clone()
            rival_logits[rows, local_targets] = -torch.inf
            decision_margin = target_logits - rival_logits.max(dim=1).values
            if not bool(torch.isfinite(decision_margin).all()):
                raise RuntimeError("qualitative-map decision margin is NaN or Inf")

            coordinate_rows.append(coordinates)
            target_rows.append(labels.detach().cpu())
            prediction_rows.append(output.prediction.detach().cpu())
            margin_rows.append(decision_margin.detach().cpu())

    if not target_rows:
        raise RuntimeError("map-reporting loader is empty")

    return save_full_phase_qualitative_maps(
        output_dir=os.path.abspath(output_dir),
        phase=phase_id,
        gt_shape=dataset.gt_shape,
        coords=torch.cat(coordinate_rows).numpy(),
        targets=torch.cat(target_rows).numpy(),
        predictions=torch.cat(prediction_rows).numpy(),
        decision_margins=torch.cat(margin_rows).numpy(),
        class_ids=class_ids,
        phase_to_class_ids=phase_groups,
        total_classes=len(dataset.target_names),
        target_names=dataset.target_names,
        cmap_name=str(cmap_name),
        dpi=dpi_value,
    )


@torch.no_grad()
def generate_base_qualitative_maps(**kwargs: Any) -> Dict[str, str]:
    kwargs["phase"] = 0
    return generate_phase_qualitative_maps(**kwargs)


__all__ = ["generate_phase_qualitative_maps", "generate_base_qualitative_maps"]





# from __future__ import annotations

# """Reporting-only qualitative maps for finalized full-CDS NECIL-HSI phases."""

# from contextlib import contextmanager
# from numbers import Integral, Real
# from typing import Any, Dict, Mapping
# import math
# import os

# import numpy as np
# import torch

# from models.classifier import CosineClassificationHead
# from models.geometry_bank import HSIClassDiscriminativeStructureBank
# from utils.visualize import save_full_phase_qualitative_maps


# def _nonnegative_int(value: object, *, name: str) -> int:
#     if isinstance(value, bool):
#         raise ValueError(f"{name} must be an integer")
#     if isinstance(value, Integral):
#         result = int(value)
#     elif isinstance(value, Real):
#         number = float(value)
#         if not math.isfinite(number) or not number.is_integer():
#             raise ValueError(f"{name} must be a finite integer")
#         result = int(number)
#     else:
#         raise ValueError(f"{name} must be an integer")
#     if result < 0:
#         raise ValueError(f"{name} must be non-negative")
#     return result


# @contextmanager
# def _temporary_eval_state(model: Any):
#     states = [(module, bool(module.training)) for module in model.modules()]
#     try:
#         model.eval()
#         yield
#     finally:
#         for module, state in states:
#             module.training = state


# @torch.no_grad()
# def generate_phase_qualitative_maps(
#     *, model: Any, dataset: Any, phase: int, output_dir: str,
#     device: str | torch.device, batch_size: int = 256,
#     cmap_name: str = "nipy_spectral", dpi: int = 300,
# ) -> Dict[str, str]:
#     phase_id = _nonnegative_int(phase, name="phase")
#     batch_size_value = _nonnegative_int(batch_size, name="batch_size")
#     dpi_value = _nonnegative_int(dpi, name="dpi")
#     if batch_size_value == 0 or dpi_value == 0:
#         raise ValueError("batch_size and dpi must be positive")

#     evaluation_device = torch.device(device)
#     if torch.device(model.device) != evaluation_device:
#         raise RuntimeError("model and reporting device disagree")

#     model.validate_finalized_state()
#     if not isinstance(model.structure_bank, HSIClassDiscriminativeStructureBank):
#         raise RuntimeError("model.structure_bank has wrong type")
#     if not isinstance(model.classifier, CosineClassificationHead):
#         raise RuntimeError("model.classifier has wrong type")

#     finalized = [int(value) for value in getattr(dataset, "finalized_phases", [])]
#     if phase_id not in finalized or getattr(dataset, "current_phase", None) is not None:
#         raise RuntimeError("qualitative reporting requires a finalized inactive phase")

#     class_ids = [int(value) for value in dataset.get_seen_classes(phase_id)]
#     if list(model.committed_class_ids) != class_ids:
#         raise RuntimeError("committed CDS does not match phase classes")

#     phase_map = getattr(dataset, "phase_to_classes", None)
#     if not isinstance(phase_map, Mapping):
#         raise RuntimeError("dataset must expose phase_to_classes")
#     phase_groups = {
#         current_phase: [int(value) for value in phase_map[current_phase]]
#         for current_phase in range(phase_id + 1)
#     }
#     if [class_id for current_phase in range(phase_id + 1) for class_id in phase_groups[current_phase]] != class_ids:
#         raise RuntimeError("phase_to_classes does not reproduce visible class order")

#     loader = dataset.get_map_dataloader(phase_id, batch_size=batch_size_value)
#     coordinate_rows: list[torch.Tensor] = []
#     target_rows: list[torch.Tensor] = []
#     prediction_rows: list[torch.Tensor] = []
#     margin_rows: list[torch.Tensor] = []

#     with _temporary_eval_state(model):
#         for batch in loader:
#             if not isinstance(batch, Mapping):
#                 raise TypeError("map-reporting batches must be mappings")
#             missing = {"image", "raw_center_spectrum", "label", "coord"} - set(batch)
#             if missing:
#                 raise KeyError(f"map-reporting batch missing {sorted(missing)}")

#             patch = torch.as_tensor(
#                 batch["image"],
#                 device=evaluation_device,
#                 dtype=model.dtype,
#             )
#             spectrum = torch.as_tensor(
#                 batch["raw_center_spectrum"],
#                 device=evaluation_device,
#                 dtype=model.dtype,
#             )
#             labels = torch.as_tensor(
#                 batch["label"],
#                 device=evaluation_device,
#                 dtype=torch.long,
#             ).flatten()
#             coordinates = torch.as_tensor(
#                 batch["coord"],
#                 dtype=torch.long,
#             ).detach().cpu()

#             if labels.shape != (patch.size(0),) or coordinates.shape != (patch.size(0), 2):
#                 raise RuntimeError("map-reporting rows are batch-misaligned")

#             ordered_features = model.extract_features(
#                 patch,
#                 center_spectrum=spectrum,
#                 return_aux=False,
#             ).ordered_features
#             output = model.classify_features(
#                 ordered_features,
#                 class_ids=class_ids,
#             )

#             matches = labels.unsqueeze(1).eq(output.class_ids.unsqueeze(0))
#             if not bool(matches.sum(dim=1).eq(1).all()):
#                 raise RuntimeError("could not map labels to classifier columns")

#             local_targets = matches.to(torch.long).argmax(dim=1)
#             rows = torch.arange(labels.numel(), device=labels.device)
#             target_logits = output.logits[rows, local_targets]
#             rival_logits = output.logits.clone()
#             rival_logits[rows, local_targets] = -torch.inf
#             decision_margin = target_logits - rival_logits.max(dim=1).values
#             if not bool(torch.isfinite(decision_margin).all()):
#                 raise RuntimeError("qualitative-map decision margin is NaN or Inf")

#             coordinate_rows.append(coordinates)
#             target_rows.append(labels.detach().cpu())
#             prediction_rows.append(output.prediction.detach().cpu())
#             margin_rows.append(decision_margin.detach().cpu())

#     if not target_rows:
#         raise RuntimeError("map-reporting loader is empty")

#     return save_full_phase_qualitative_maps(
#         output_dir=os.path.abspath(output_dir),
#         phase=phase_id,
#         gt_shape=dataset.gt_shape,
#         coords=torch.cat(coordinate_rows).numpy(),
#         targets=torch.cat(target_rows).numpy(),
#         predictions=torch.cat(prediction_rows).numpy(),
#         decision_margins=torch.cat(margin_rows).numpy(),
#         class_ids=class_ids,
#         phase_to_class_ids=phase_groups,
#         total_classes=len(dataset.target_names),
#         target_names=dataset.target_names,
#         cmap_name=str(cmap_name),
#         dpi=dpi_value,
#     )


# @torch.no_grad()
# def generate_base_qualitative_maps(**kwargs: Any) -> Dict[str, str]:
#     kwargs["phase"] = 0
#     return generate_phase_qualitative_maps(**kwargs)


# __all__ = ["generate_phase_qualitative_maps", "generate_base_qualitative_maps"]
