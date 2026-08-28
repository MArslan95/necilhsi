from __future__ import annotations

"""Reporting-only qualitative maps for finalized NECIL-HSI phases.

Prediction uses only the persistent normalized cosine classifier.  The
committed {I_c,S_c,N_c} geometry is historical statistical memory and never
produces map labels.
"""

from contextlib import contextmanager
import math
import os
from numbers import Integral, Real
from typing import Any, Dict, Mapping

import numpy as np
import torch

from models.classifier import PersistentCosineClassifier
from models.geometry_bank import HSIClassDiscriminativeGeometry
from utils.visualize import save_full_phase_qualitative_maps



def _as_nonnegative_int(value: object, *, name: str) -> int:
    """Convert an exact non-negative integer scalar without silent truncation."""
    if torch.is_tensor(value):
        if value.numel() != 1:
            raise ValueError(f"{name} must be an integer scalar")
        value = value.item()
    if isinstance(value, (bool, np.bool_)):
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
    if result > np.iinfo(np.int64).max:
        raise ValueError(f"{name} lies outside the supported int64 range")
    return result


@contextmanager
def _temporary_eval_state(model: Any):
    states = {
        module: bool(module.training)
        for module in model.modules()
    }
    try:
        model.eval()
        yield
    finally:
        for module, state in states.items():
            module.training = state


@torch.no_grad()
def generate_phase_qualitative_maps(
    *,
    model: Any,
    dataset: Any,
    phase: int,
    output_dir: str,
    device: str | torch.device,
    batch_size: int = 256,
    cmap_name: str = "nipy_spectral",
    dpi: int = 300,
) -> Dict[str, str]:
    """Generate separate full-scene GT and prediction maps for a finalized phase."""
    phase_id = _as_nonnegative_int(phase, name="phase")
    batch_size_value = _as_nonnegative_int(batch_size, name="batch_size")
    dpi_value = _as_nonnegative_int(dpi, name="dpi")
    if batch_size_value == 0:
        raise ValueError("batch_size must be positive")
    if dpi_value == 0:
        raise ValueError("dpi must be positive")

    evaluation_device = torch.device(device)
    if torch.device(model.device) != evaluation_device:
        raise RuntimeError("model and reporting device disagree")

    validate_model = getattr(model, "validate_model_state", None)
    if not callable(validate_model):
        raise RuntimeError("model must expose validate_model_state()")
    validate_model()

    bank = getattr(model, "geometry_bank", None)
    classifier = getattr(model, "classifier", None)
    if not isinstance(bank, HSIClassDiscriminativeGeometry):
        raise RuntimeError(
            "model.geometry_bank must be HSIClassDiscriminativeGeometry"
        )
    if not isinstance(classifier, PersistentCosineClassifier):
        raise RuntimeError("model.classifier must be PersistentCosineClassifier")

    finalized = [
        _as_nonnegative_int(value, name="finalized phase")
        for value in getattr(dataset, "finalized_phases", [])
    ]
    if phase_id not in finalized:
        raise RuntimeError(
            f"qualitative reporting requires finalized phase {phase_id}"
        )
    if getattr(dataset, "current_phase", None) is not None:
        raise RuntimeError(
            "qualitative reporting cannot run during an active phase"
        )

    get_seen = getattr(dataset, "get_seen_classes", None)
    get_map_loader = getattr(dataset, "get_map_dataloader", None)
    if not callable(get_seen) or not callable(get_map_loader):
        raise RuntimeError(
            "dataset must expose get_seen_classes() and get_map_dataloader()"
        )

    class_ids = [
        _as_nonnegative_int(value, name="seen class ID")
        for value in get_seen(phase_id)
    ]
    if not class_ids or len(class_ids) != len(set(class_ids)):
        raise RuntimeError("dataset returned invalid seen class IDs")

    committed = [
        _as_nonnegative_int(value, name="committed class ID")
        for value in model.committed_class_ids
    ]
    if committed != class_ids:
        raise RuntimeError(
            f"committed class structure {committed} does not match "
            f"phase-{phase_id} seen classes {class_ids}"
        )

    target_names = list(getattr(dataset, "target_names", []))
    if not target_names:
        raise RuntimeError("dataset must expose target_names")
    total_classes = len(target_names)
    if max(class_ids) >= total_classes:
        raise RuntimeError("seen class IDs exceed dataset class count")

    gt_shape = getattr(dataset, "gt_shape", None)
    if gt_shape is None or len(gt_shape) != 2:
        raise RuntimeError("dataset must expose two-dimensional gt_shape")

    loader = get_map_loader(
        phase_id,
        batch_size=batch_size_value,
    )

    coords_parts: list[torch.Tensor] = []
    target_parts: list[torch.Tensor] = []
    prediction_parts: list[torch.Tensor] = []

    with _temporary_eval_state(model):
        for batch in loader:
            if not isinstance(batch, Mapping):
                raise TypeError("map-reporting batches must be mappings")

            required = {
                "image",
                "raw_center_spectrum",
                "label",
                "coord",
            }
            missing = required - set(batch)
            if missing:
                raise KeyError(
                    f"map-reporting batch missing {sorted(missing)}"
                )

            dtype = model.dtype
            patch = torch.as_tensor(
                batch["image"],
                device=evaluation_device,
                dtype=dtype,
            )
            spectrum = torch.as_tensor(
                batch["raw_center_spectrum"],
                device=evaluation_device,
                dtype=dtype,
            )
            labels = torch.as_tensor(
                batch["label"],
                device=evaluation_device,
            ).flatten()
            coords = torch.as_tensor(batch["coord"]).detach().cpu()

            backbone = model.backbone
            expected_patch = (
                patch.size(0) if patch.ndim > 0 else 0,
                int(backbone.patch_bands),
                int(backbone.patch_size),
                int(backbone.patch_size),
            )
            expected_spectrum = (
                patch.size(0) if patch.ndim > 0 else 0,
                int(backbone.spectral_bands),
            )

            if (
                patch.ndim != 4
                or patch.size(0) == 0
                or tuple(patch.shape) != expected_patch
            ):
                raise RuntimeError(
                    "map-reporting patch shape disagrees with the backbone"
                )
            if (
                spectrum.ndim != 2
                or tuple(spectrum.shape) != expected_spectrum
            ):
                raise RuntimeError(
                    "map-reporting spectrum shape disagrees with the backbone"
                )
            if (
                not bool(torch.isfinite(patch).all())
                or not bool(torch.isfinite(spectrum).all())
            ):
                raise RuntimeError(
                    "map-reporting input contains NaN or Inf"
                )

            if (
                labels.numel() != patch.size(0)
                or labels.dtype == torch.bool
                or labels.is_complex()
            ):
                raise RuntimeError(
                    "map-reporting labels are invalid or batch-misaligned"
                )
            if torch.is_floating_point(labels):
                if (
                    not bool(torch.isfinite(labels).all())
                    or not bool(labels.eq(labels.round()).all())
                ):
                    raise RuntimeError(
                        "map-reporting labels must be finite integers"
                    )
            labels = labels.to(torch.long)

            if bool((labels < 0).any()):
                raise RuntimeError(
                    "map-reporting labels must be non-negative"
                )

            allowed = torch.tensor(
                class_ids,
                device=labels.device,
                dtype=torch.long,
            )
            if not bool(torch.isin(labels, allowed).all()):
                raise RuntimeError(
                    "map loader returned a phase-invisible class"
                )

            if coords.dtype == torch.bool or coords.is_complex():
                raise RuntimeError(
                    "map-reporting coordinates must contain integers"
                )
            if torch.is_floating_point(coords):
                if (
                    not bool(torch.isfinite(coords).all())
                    or not bool(coords.eq(coords.round()).all())
                ):
                    raise RuntimeError(
                        "map-reporting coordinates must contain finite integers"
                    )
            coords = coords.long()
            if tuple(coords.shape) != (labels.numel(), 2):
                raise RuntimeError(
                    "map-reporting coordinates must have shape [B,2]"
                )

            backbone_output = model.extract_features(
                patch,
                center_spectrum=spectrum,
                return_aux=False,
            )
            output = model.classify_features(
                backbone_output.ordered_features,
                class_ids=class_ids,
            )

            returned_ids = [
                int(value)
                for value in output.class_ids.detach().cpu().tolist()
            ]
            if returned_ids != class_ids:
                raise RuntimeError(
                    "persistent classifier class order changed during map reporting"
                )

            expected_prediction = output.class_ids.index_select(
                0,
                output.cosine_similarity.argmax(dim=1),
            )
            if not torch.equal(output.prediction, expected_prediction):
                raise RuntimeError(
                    "map prediction is not argmax of cosine classifier scores"
                )

            coords_parts.append(coords)
            target_parts.append(labels.detach().cpu())
            prediction_parts.append(
                output.prediction.detach().cpu()
            )

    if not target_parts:
        raise RuntimeError("map-reporting loader is empty")

    coords = torch.cat(coords_parts, dim=0).numpy()
    targets = torch.cat(target_parts, dim=0).numpy()
    predictions = torch.cat(prediction_parts, dim=0).numpy()

    if np.unique(coords, axis=0).shape[0] != coords.shape[0]:
        raise RuntimeError(
            "map-reporting coordinates contain duplicate image locations"
        )

    observed_targets = set(int(v) for v in np.unique(targets).tolist())
    if observed_targets != set(class_ids):
        missing = sorted(set(class_ids) - observed_targets)
        raise RuntimeError(
            f"map loader is missing seen classes {missing}"
        )

    return save_full_phase_qualitative_maps(
        output_dir=os.path.abspath(output_dir),
        phase=phase_id,
        gt_shape=gt_shape,
        coords=coords,
        targets=targets,
        predictions=predictions,
        class_ids=class_ids,
        total_classes=total_classes,
        target_names=target_names,
        cmap_name=str(cmap_name),
        dpi=dpi_value,
    )


@torch.no_grad()
def generate_base_qualitative_maps(
    *,
    model: Any,
    dataset: Any,
    output_dir: str,
    device: str | torch.device,
    batch_size: int = 256,
    cmap_name: str = "nipy_spectral",
    dpi: int = 300,
) -> Dict[str, str]:
    """Phase-0 convenience wrapper."""
    return generate_phase_qualitative_maps(
        model=model,
        dataset=dataset,
        phase=0,
        output_dir=output_dir,
        device=device,
        batch_size=batch_size,
        cmap_name=cmap_name,
        dpi=dpi,
    )


__all__ = [
    "generate_phase_qualitative_maps",
    "generate_base_qualitative_maps",
]







# from __future__ import annotations

# """Reporting-only qualitative maps for finalized NECIL-HSI phases.

# Deployed architecture
# ---------------------
#     HSI
#       -> ordered contextual spectral-region features Q(x)
#       -> committed structured class state {I_c,S_c,N_c}
#       -> GeometryClassifier energy
#            E_c=E_identity+E_feature+E_cross, with A_c/C_c derived from S_c
#       -> argmin_c E_c(x).

# This module is reporting-only.  It may use every labeled pixel belonging to
# the classes visible at a finalized phase through the dataset's dedicated map
# loader, but those rows never enter optimization, model selection, temporary
# class-structure estimation, or persistent class-state estimation.
# """

# from contextlib import contextmanager
# import math
# import os
# from numbers import Integral, Real
# from typing import Any, Dict, Mapping

# import numpy as np
# import torch

# from models.classifier import GeometryClassifier
# from models.geometry_bank import HSIClassDiscriminativeGeometry
# from utils.visualize import save_full_phase_qualitative_maps



# def _as_nonnegative_int(value: object, *, name: str) -> int:
#     """Convert an exact non-negative integer scalar without silent truncation."""
#     if torch.is_tensor(value):
#         if value.numel() != 1:
#             raise ValueError(f"{name} must be an integer scalar")
#         value = value.item()
#     if isinstance(value, (bool, np.bool_)):
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
#     if result > np.iinfo(np.int64).max:
#         raise ValueError(f"{name} lies outside the supported int64 range")
#     return result


# @contextmanager
# def _temporary_eval_state(model: Any):
#     states = {
#         module: bool(module.training)
#         for module in model.modules()
#     }
#     try:
#         model.eval()
#         yield
#     finally:
#         for module, state in states.items():
#             module.training = state


# @torch.no_grad()
# def generate_phase_qualitative_maps(
#     *,
#     model: Any,
#     dataset: Any,
#     phase: int,
#     output_dir: str,
#     device: str | torch.device,
#     batch_size: int = 256,
#     cmap_name: str = "nipy_spectral",
#     dpi: int = 300,
# ) -> Dict[str, str]:
#     """Generate separate full-scene GT and prediction maps for a finalized phase."""
#     phase_id = _as_nonnegative_int(phase, name="phase")
#     batch_size_value = _as_nonnegative_int(batch_size, name="batch_size")
#     dpi_value = _as_nonnegative_int(dpi, name="dpi")
#     if batch_size_value == 0:
#         raise ValueError("batch_size must be positive")
#     if dpi_value == 0:
#         raise ValueError("dpi must be positive")

#     evaluation_device = torch.device(device)
#     if torch.device(model.device) != evaluation_device:
#         raise RuntimeError("model and reporting device disagree")

#     validate_model = getattr(model, "validate_model_state", None)
#     if not callable(validate_model):
#         raise RuntimeError("model must expose validate_model_state()")
#     validate_model()

#     bank = getattr(model, "geometry_bank", None)
#     classifier = getattr(model, "classifier", None)
#     if not isinstance(bank, HSIClassDiscriminativeGeometry):
#         raise RuntimeError(
#             "model.geometry_bank must be HSIClassDiscriminativeGeometry"
#         )
#     if not isinstance(classifier, GeometryClassifier):
#         raise RuntimeError("model.classifier must be GeometryClassifier")

#     finalized = [
#         _as_nonnegative_int(value, name="finalized phase")
#         for value in getattr(dataset, "finalized_phases", [])
#     ]
#     if phase_id not in finalized:
#         raise RuntimeError(
#             f"qualitative reporting requires finalized phase {phase_id}"
#         )
#     if getattr(dataset, "current_phase", None) is not None:
#         raise RuntimeError(
#             "qualitative reporting cannot run during an active phase"
#         )

#     get_seen = getattr(dataset, "get_seen_classes", None)
#     get_map_loader = getattr(dataset, "get_map_dataloader", None)
#     if not callable(get_seen) or not callable(get_map_loader):
#         raise RuntimeError(
#             "dataset must expose get_seen_classes() and get_map_dataloader()"
#         )

#     class_ids = [
#         _as_nonnegative_int(value, name="seen class ID")
#         for value in get_seen(phase_id)
#     ]
#     if not class_ids or len(class_ids) != len(set(class_ids)):
#         raise RuntimeError("dataset returned invalid seen class IDs")

#     committed = [
#         _as_nonnegative_int(value, name="committed class ID")
#         for value in model.committed_class_ids
#     ]
#     if committed != class_ids:
#         raise RuntimeError(
#             f"committed class structure {committed} does not match "
#             f"phase-{phase_id} seen classes {class_ids}"
#         )

#     target_names = list(getattr(dataset, "target_names", []))
#     if not target_names:
#         raise RuntimeError("dataset must expose target_names")
#     total_classes = len(target_names)
#     if max(class_ids) >= total_classes:
#         raise RuntimeError("seen class IDs exceed dataset class count")

#     gt_shape = getattr(dataset, "gt_shape", None)
#     if gt_shape is None or len(gt_shape) != 2:
#         raise RuntimeError("dataset must expose two-dimensional gt_shape")

#     loader = get_map_loader(
#         phase_id,
#         batch_size=batch_size_value,
#     )

#     coords_parts: list[torch.Tensor] = []
#     target_parts: list[torch.Tensor] = []
#     prediction_parts: list[torch.Tensor] = []

#     with _temporary_eval_state(model):
#         for batch in loader:
#             if not isinstance(batch, Mapping):
#                 raise TypeError("map-reporting batches must be mappings")

#             required = {
#                 "image",
#                 "raw_center_spectrum",
#                 "label",
#                 "coord",
#             }
#             missing = required - set(batch)
#             if missing:
#                 raise KeyError(
#                     f"map-reporting batch missing {sorted(missing)}"
#                 )

#             dtype = model.dtype
#             patch = torch.as_tensor(
#                 batch["image"],
#                 device=evaluation_device,
#                 dtype=dtype,
#             )
#             spectrum = torch.as_tensor(
#                 batch["raw_center_spectrum"],
#                 device=evaluation_device,
#                 dtype=dtype,
#             )
#             labels = torch.as_tensor(
#                 batch["label"],
#                 device=evaluation_device,
#             ).flatten()
#             coords = torch.as_tensor(batch["coord"]).detach().cpu()

#             backbone = model.backbone
#             expected_patch = (
#                 patch.size(0) if patch.ndim > 0 else 0,
#                 int(backbone.patch_bands),
#                 int(backbone.patch_size),
#                 int(backbone.patch_size),
#             )
#             expected_spectrum = (
#                 patch.size(0) if patch.ndim > 0 else 0,
#                 int(backbone.spectral_bands),
#             )

#             if (
#                 patch.ndim != 4
#                 or patch.size(0) == 0
#                 or tuple(patch.shape) != expected_patch
#             ):
#                 raise RuntimeError(
#                     "map-reporting patch shape disagrees with the backbone"
#                 )
#             if (
#                 spectrum.ndim != 2
#                 or tuple(spectrum.shape) != expected_spectrum
#             ):
#                 raise RuntimeError(
#                     "map-reporting spectrum shape disagrees with the backbone"
#                 )
#             if (
#                 not bool(torch.isfinite(patch).all())
#                 or not bool(torch.isfinite(spectrum).all())
#             ):
#                 raise RuntimeError(
#                     "map-reporting input contains NaN or Inf"
#                 )

#             if (
#                 labels.numel() != patch.size(0)
#                 or labels.dtype == torch.bool
#                 or labels.is_complex()
#             ):
#                 raise RuntimeError(
#                     "map-reporting labels are invalid or batch-misaligned"
#                 )
#             if torch.is_floating_point(labels):
#                 if (
#                     not bool(torch.isfinite(labels).all())
#                     or not bool(labels.eq(labels.round()).all())
#                 ):
#                     raise RuntimeError(
#                         "map-reporting labels must be finite integers"
#                     )
#             labels = labels.to(torch.long)

#             if bool((labels < 0).any()):
#                 raise RuntimeError(
#                     "map-reporting labels must be non-negative"
#                 )

#             allowed = torch.tensor(
#                 class_ids,
#                 device=labels.device,
#                 dtype=torch.long,
#             )
#             if not bool(torch.isin(labels, allowed).all()):
#                 raise RuntimeError(
#                     "map loader returned a phase-invisible class"
#                 )

#             if coords.dtype == torch.bool or coords.is_complex():
#                 raise RuntimeError(
#                     "map-reporting coordinates must contain integers"
#                 )
#             if torch.is_floating_point(coords):
#                 if (
#                     not bool(torch.isfinite(coords).all())
#                     or not bool(coords.eq(coords.round()).all())
#                 ):
#                     raise RuntimeError(
#                         "map-reporting coordinates must contain finite integers"
#                     )
#             coords = coords.long()
#             if tuple(coords.shape) != (labels.numel(), 2):
#                 raise RuntimeError(
#                     "map-reporting coordinates must have shape [B,2]"
#                 )

#             backbone_output = model.extract_features(
#                 patch,
#                 center_spectrum=spectrum,
#                 return_aux=False,
#             )
#             output = model.classify_features(
#                 backbone_output.ordered_features,
#                 class_ids=class_ids,
#             )

#             returned_ids = [
#                 int(value)
#                 for value in output.class_ids.detach().cpu().tolist()
#             ]
#             if returned_ids != class_ids:
#                 raise RuntimeError(
#                     "GeometryClassifier class order changed during map reporting"
#                 )

#             expected_prediction = output.class_ids.index_select(
#                 0,
#                 output.total_energy.argmin(dim=1),
#             )
#             if not torch.equal(
#                 output.prediction,
#                 expected_prediction,
#             ):
#                 raise RuntimeError(
#                     "map prediction is not argmin of total structural energy"
#                 )

#             coords_parts.append(coords)
#             target_parts.append(labels.detach().cpu())
#             prediction_parts.append(
#                 output.prediction.detach().cpu()
#             )

#     if not target_parts:
#         raise RuntimeError("map-reporting loader is empty")

#     coords = torch.cat(coords_parts, dim=0).numpy()
#     targets = torch.cat(target_parts, dim=0).numpy()
#     predictions = torch.cat(prediction_parts, dim=0).numpy()

#     if np.unique(coords, axis=0).shape[0] != coords.shape[0]:
#         raise RuntimeError(
#             "map-reporting coordinates contain duplicate image locations"
#         )

#     observed_targets = set(int(v) for v in np.unique(targets).tolist())
#     if observed_targets != set(class_ids):
#         missing = sorted(set(class_ids) - observed_targets)
#         raise RuntimeError(
#             f"map loader is missing seen classes {missing}"
#         )

#     return save_full_phase_qualitative_maps(
#         output_dir=os.path.abspath(output_dir),
#         phase=phase_id,
#         gt_shape=gt_shape,
#         coords=coords,
#         targets=targets,
#         predictions=predictions,
#         class_ids=class_ids,
#         total_classes=total_classes,
#         target_names=target_names,
#         cmap_name=str(cmap_name),
#         dpi=dpi_value,
#     )


# @torch.no_grad()
# def generate_base_qualitative_maps(
#     *,
#     model: Any,
#     dataset: Any,
#     output_dir: str,
#     device: str | torch.device,
#     batch_size: int = 256,
#     cmap_name: str = "nipy_spectral",
#     dpi: int = 300,
# ) -> Dict[str, str]:
#     """Phase-0 convenience wrapper."""
#     return generate_phase_qualitative_maps(
#         model=model,
#         dataset=dataset,
#         phase=0,
#         output_dir=output_dir,
#         device=device,
#         batch_size=batch_size,
#         cmap_name=cmap_name,
#         dpi=dpi,
#     )


# __all__ = [
#     "generate_phase_qualitative_maps",
#     "generate_base_qualitative_maps",
# ]

