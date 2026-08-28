from __future__ import annotations

"""Phase-0 optimization for the role-separated NECIL-HSI architecture.

Phase 0 trains only the HSI backbone and classifier. Class statistics
{I_c,S_c,N_c} are estimated once from deterministic TRAIN features after the
final requested epoch. They do not participate in optimization or prediction.
"""

import math
import sys
from numbers import Integral, Real
from typing import Any, Dict, Sequence

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from losses.loss import cosine_classification_objective
from models.classifier import PersistentCosineClassifier

Tensor = torch.Tensor


def _positive_int(value: object, *, name: str) -> int:
    if torch.is_tensor(value):
        if value.numel() != 1:
            raise ValueError(f"{name} must be an integer scalar")
        value = value.item()
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
    if result <= 0:
        raise ValueError(f"{name} must be positive")
    return result


def _finite_nonnegative(name: str, value: object) -> float:
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return result


def _finite_positive(name: str, value: object) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return result


class BasePhaseTrainer:
    """Fixed-epoch base training with independent classifier and geometry roles."""

    BASE_RESULT_FIELDS = frozenset(
        {
            "class_ids",
            "history",
            "final_epoch",
            "train",
            "validation",
            "test",
            "geometry_committed",
            "phase_finalized",
        }
    )

    def _base_class_ids(self, values: Sequence[int]) -> tuple[int, ...]:
        ids = tuple(self._validated_class_ids(values, name="phase_0_classes"))
        if len(ids) < 2:
            raise ValueError("phase-0 classification requires at least two classes")
        return ids

    def _require_classifier(self) -> PersistentCosineClassifier:
        classifier = getattr(self.model, "classifier", None)
        if not isinstance(classifier, PersistentCosineClassifier):
            raise RuntimeError(
                "NECILModel.classifier must be PersistentCosineClassifier"
            )
        return classifier

    @staticmethod
    def _parameters_finite(parameters: Sequence[nn.Parameter]) -> bool:
        return all(bool(torch.isfinite(parameter).all()) for parameter in parameters)

    @staticmethod
    def _spatial_symmetry_augment(
        patch: Tensor,
        *,
        generator: torch.Generator,
    ) -> Tensor:
        """Apply one D4 square symmetry per sample, spatial axes only."""
        if patch.ndim != 4 or patch.size(-1) != patch.size(-2):
            raise RuntimeError(
                "spatial symmetry augmentation requires [N,C,H,W] square patches"
            )
        codes = torch.randint(
            0,
            8,
            (patch.size(0),),
            generator=generator,
            device="cpu",
        )
        rows: list[Tensor] = []
        for sample, code in zip(patch, codes.tolist()):
            value = sample
            if int(code) >= 4:
                value = torch.flip(value, dims=(-1,))
            rotation = int(code) % 4
            if rotation:
                value = torch.rot90(value, k=rotation, dims=(-2, -1))
            rows.append(value)
        result = torch.stack(rows, dim=0)
        if result.shape != patch.shape or not bool(torch.isfinite(result).all()):
            raise RuntimeError("spatial symmetry augmentation produced invalid patches")
        return result

    @torch.no_grad()
    def _ensure_spectral_normalization(self, loader: Any) -> None:
        if bool(self.model.spectral_normalization_fitted):
            return
        spectra: list[Tensor] = []
        for batch in loader:
            _, center_spectrum, _ = self.unpack_batch(batch)
            spectra.append(center_spectrum.detach())
        if not spectra:
            raise RuntimeError("phase-0 spectral-normalization loader is empty")
        self.model.fit_spectral_normalization(
            torch.cat(spectra, dim=0),
            overwrite=False,
        )
        if not bool(self.model.spectral_normalization_fitted):
            raise RuntimeError("spectral normalization was not fitted")

    def _validation_enabled(self) -> bool:
        value = getattr(self.dataset, "validation_enabled", None)
        if not isinstance(value, bool):
            raise RuntimeError("dataset.validation_enabled must be bool")
        return value

    def _phase_train_loader(
        self,
        *,
        batch_size: int,
        shuffle: bool,
    ) -> DataLoader:
        return self.dataset.get_phase_dataloader(
            0,
            split="train",
            batch_size=batch_size,
            shuffle=shuffle,
        )

    def train_base_phase(
        self,
        phase: int,
        epochs: int,
        batch_size: int = 64,
        lr: float = 1e-4,
    ) -> Dict[str, Any]:
        if int(phase) != 0:
            raise ValueError("BasePhaseTrainer handles phase 0 only")

        epoch_count = _positive_int(epochs, name="epochs")
        train_batch_size = _positive_int(batch_size, name="batch_size")
        learning_rate = _finite_positive("lr", lr)

        self.model.validate_model_state()
        classifier = self._require_classifier()
        if len(self.model.geometry_bank) != 0:
            raise RuntimeError("phase 0 requires an empty persistent geometry bank")
        if classifier.class_count != 0:
            raise RuntimeError("phase 0 requires an empty persistent classifier")
        if getattr(self.dataset, "current_phase", None) is not None:
            raise RuntimeError("dataset already has an active phase")

        self.dataset.start_phase(0)
        class_ids = self._base_class_ids(self.dataset.get_new_classes(0))

        eval_batch_size = _positive_int(
            self.cfg("eval_batch_size", 256),
            name="eval_batch_size",
        )
        train_eval_loader = self._phase_train_loader(
            batch_size=eval_batch_size,
            shuffle=False,
        )
        self._ensure_spectral_normalization(train_eval_loader)

        # Expansion replaces the classifier Parameter object, so it must happen
        # before optimizer construction.
        self.model.add_classifier_classes(class_ids)
        classifier = self._require_classifier()
        if tuple(classifier.seen_class_ids) != class_ids:
            raise RuntimeError("phase-0 classifier class IDs are incorrect")

        train_loader = self._phase_train_loader(
            batch_size=train_batch_size,
            shuffle=True,
        )
        val_loader = None
        if self._validation_enabled():
            val_loader = self.dataset.get_phase_dataloader(
                0,
                split="val",
                batch_size=eval_batch_size,
                shuffle=False,
            )

        backbone_parameters = [
            parameter
            for parameter in self.model.backbone.parameters()
            if parameter.requires_grad
        ]
        classifier_parameters = [
            parameter
            for parameter in classifier.parameters()
            if parameter.requires_grad
        ]
        if not backbone_parameters or not classifier_parameters:
            raise RuntimeError(
                "phase-0 training requires trainable backbone and classifier parameters"
            )
        trainable = backbone_parameters + classifier_parameters

        backbone_weight_decay = _finite_nonnegative(
            "weight_decay",
            self.cfg("weight_decay", 1e-4),
        )
        optimizer = torch.optim.AdamW(
            [
                {
                    "params": backbone_parameters,
                    "weight_decay": backbone_weight_decay,
                },
                {
                    # Classifier rows are direction-only because forward()
                    # normalizes w_c.  Weight decay on their raw norm has no
                    # decision meaning, so do not regularize that nuisance scale.
                    "params": classifier_parameters,
                    "weight_decay": 0.0,
                },
            ],
            lr=learning_rate,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=epoch_count,
            eta_min=0.0,
        )
        gradient_clip = _finite_nonnegative(
            "gradient_clip",
            self.cfg("gradient_clip", 0.0),
        )

        augmentation_generator = torch.Generator(device="cpu")
        augmentation_generator.manual_seed(int(self.cfg("seed", 0)))
        class_tensor = torch.tensor(class_ids, device=self.device, dtype=torch.long)
        history: list[Dict[str, Any]] = []

        progress = tqdm(
            range(epoch_count),
            desc="Phase 0 classifier representation learning",
            unit="epoch",
            dynamic_ncols=True,
            file=sys.stdout,
        )

        for epoch in progress:
            self.model.train()
            epoch_lr = float(optimizer.param_groups[0]["lr"])
            if not math.isfinite(epoch_lr) or epoch_lr < 0.0:
                raise RuntimeError("optimizer learning rate is invalid")

            weighted_loss = 0.0
            sample_count = 0

            for batch in train_loader:
                patch, spectrum, labels = self.unpack_batch(batch)
                patch = self._spatial_symmetry_augment(
                    patch,
                    generator=augmentation_generator,
                )
                if not set(labels.detach().cpu().tolist()).issubset(set(class_ids)):
                    raise RuntimeError("phase-0 batch contains a non-base class")

                optimizer.zero_grad(set_to_none=True)
                backbone_output = self.model.extract_features(
                    patch,
                    center_spectrum=spectrum,
                    return_aux=False,
                )
                classifier_output = self.model.classify_features(
                    backbone_output.ordered_features,
                    class_ids=class_ids,
                )
                objective = cosine_classification_objective(
                    output=classifier_output,
                    labels_global=labels,
                )
                objective.loss.backward()

                if any(
                    parameter.grad is not None
                    and not bool(torch.isfinite(parameter.grad).all())
                    for parameter in trainable
                ):
                    raise RuntimeError("phase-0 optimization produced NaN/Inf gradients")

                if gradient_clip > 0.0:
                    torch.nn.utils.clip_grad_norm_(
                        trainable,
                        gradient_clip,
                        error_if_nonfinite=True,
                    )
                optimizer.step()
                if not self._parameters_finite(trainable):
                    raise RuntimeError("phase-0 optimization produced NaN/Inf parameters")

                n = int(labels.numel())
                weighted_loss += float(objective.loss.detach().item()) * n
                sample_count += n
            if sample_count <= 0:
                raise RuntimeError("phase-0 train loader yielded no samples")
            train_loss = weighted_loss / float(sample_count)

            # Reporting uses deterministic unaugmented data and the deployed
            # classifier. Geometry does not yet exist and cannot influence it.
            train_metrics = self.evaluate_loader(
                train_eval_loader,
                class_ids=class_ids,
            )
            validation_metrics = (
                None
                if val_loader is None
                else self.evaluate_loader(val_loader, class_ids=class_ids)
            )

            scheduler.step()
            next_lr = float(optimizer.param_groups[0]["lr"])
            if not math.isfinite(next_lr) or next_lr < 0.0:
                raise RuntimeError("scheduled learning rate is invalid")

            record: Dict[str, Any] = {
                "epoch": epoch + 1,
                "learning_rate": epoch_lr,
                "classification_loss": float(train_loss),
                "train_oa": float(train_metrics["overall_accuracy"]),
                "train_ba": float(train_metrics["balanced_accuracy"]),
                "validation_oa": (
                    None
                    if validation_metrics is None
                    else float(validation_metrics["overall_accuracy"])
                ),
                "validation_ba": (
                    None
                    if validation_metrics is None
                    else float(validation_metrics["balanced_accuracy"])
                ),
            }
            history.append(record)

            message = (
                f"[Epoch {epoch + 1:03d}/{epoch_count:03d}] "
                f"LR={epoch_lr:.3e} | CE={train_loss:.4f} | "
                f"train_OA={100.0 * train_metrics['overall_accuracy']:.2f}% | "
                f"train_BA={100.0 * train_metrics['balanced_accuracy']:.2f}% | "
                f"CosGap={train_metrics['mean_cosine_margin']:.4f}"
            )
            if validation_metrics is not None:
                message += (
                    f" | val_OA={100.0 * validation_metrics['overall_accuracy']:.2f}%"
                    f" | val_BA={100.0 * validation_metrics['balanced_accuracy']:.2f}%"
                    f" | val_CosGap={validation_metrics['mean_cosine_margin']:.4f}"
                )
            tqdm.write(message, file=sys.stdout)
            progress.set_postfix(
                lr=f"{epoch_lr:.2e}",
                ce=f"{train_loss:.4f}",
                val_ba=(
                    f"{100.0 * validation_metrics['balanced_accuracy']:.2f}%"
                    if validation_metrics is not None
                    else "-"
                ),
            )

        progress.close()
        if len(history) != epoch_count:
            raise RuntimeError("phase-0 training did not complete all epochs")

        # Fixed-epoch contract: no validation checkpoint is restored.  Geometry
        # is constructed once, here, from deterministic complete TRAIN Q(x).
        self.model.eval()
        final_train = self.collect_ordered_features(train_eval_loader)
        if len(self.model.geometry_bank) != 0:
            raise RuntimeError("persistent geometry changed before phase-0 commit")
        self.model.add_classes(
            final_train["ordered_features"],
            final_train["labels"],
            class_ids=class_ids,
        )
        if tuple(self.model.committed_class_ids) != class_ids:
            raise RuntimeError("phase-0 geometry commit produced incorrect class IDs")
        if tuple(self.model.classifier_class_ids) != class_ids:
            raise RuntimeError("phase-0 classifier class IDs changed unexpectedly")

        # Final classifier metrics remain classifier-only even though geometry is
        # now present in the model.
        train_metrics = self.evaluate_loader(
            train_eval_loader,
            class_ids=class_ids,
        )
        validation_metrics = (
            None
            if val_loader is None
            else self.evaluate_loader(val_loader, class_ids=class_ids)
        )

        self.dataset.finalize_phase(0)
        test_loader = self.dataset.get_reporting_dataloader(
            0,
            batch_size=eval_batch_size,
        )
        test_metrics = self.evaluate_loader(
            test_loader,
            class_ids=class_ids,
        )

        return {
            "phase": 0,
            "class_ids": list(class_ids),
            "history": history,
            "final_epoch": epoch_count,
            "train": train_metrics,
            "validation": validation_metrics,
            "test": test_metrics,
            "geometry_committed": True,
            "phase_finalized": True,
        }


__all__ = ["BasePhaseTrainer"]












# from __future__ import annotations

# """Phase-0 optimization for the clean role-separated NECIL-HSI architecture.

# Roles during phase 0
# --------------------
# Backbone:
#     learn the ordered HSI representation Q_raw(x).

# Canonicalizer (inside NECILModel):
#     remove arbitrary per-sample global feature scale and produce Q(x).

# PersistentCosineClassifier:
#     learn class decision directions from Q(x).

# Geometry bank:
#     does not participate in optimization.  After the final requested epoch it
#     receives complete deterministic TRAIN Q(x) once and stores {I_c,S_c,N_c}.

# Training-only D4 augmentation has one job: reduce orientation-specific spatial
# nuisance while preserving every processed spectral/PCA channel and the raw
# center spectrum.  Cosine LR has one job: reduce late fixed-epoch optimization
# movement.  Neither is persistent model state.
# """

# import math
# import sys
# from numbers import Integral, Real
# from typing import Any, Dict, Sequence

# import torch
# import torch.nn as nn
# from torch.utils.data import DataLoader
# from tqdm import tqdm

# from losses.loss import cosine_classification_objective
# from models.classifier import PersistentCosineClassifier

# Tensor = torch.Tensor


# def _positive_int(value: object, *, name: str) -> int:
#     if torch.is_tensor(value):
#         if value.numel() != 1:
#             raise ValueError(f"{name} must be an integer scalar")
#         value = value.item()
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
#     if result <= 0:
#         raise ValueError(f"{name} must be positive")
#     return result


# def _finite_nonnegative(name: str, value: object) -> float:
#     result = float(value)
#     if not math.isfinite(result) or result < 0.0:
#         raise ValueError(f"{name} must be finite and non-negative")
#     return result


# def _finite_positive(name: str, value: object) -> float:
#     result = float(value)
#     if not math.isfinite(result) or result <= 0.0:
#         raise ValueError(f"{name} must be finite and positive")
#     return result


# class BasePhaseTrainer:
#     """Fixed-epoch base training with independent classifier and geometry roles."""

#     BASE_RESULT_FIELDS = frozenset(
#         {
#             "class_ids",
#             "history",
#             "final_epoch",
#             "train",
#             "validation",
#             "test",
#             "geometry_committed",
#             "phase_finalized",
#         }
#     )

#     def _base_class_ids(self, values: Sequence[int]) -> tuple[int, ...]:
#         ids = tuple(self._validated_class_ids(values, name="phase_0_classes"))
#         if len(ids) < 2:
#             raise ValueError("phase-0 classification requires at least two classes")
#         return ids

#     def _require_classifier(self) -> PersistentCosineClassifier:
#         classifier = getattr(self.model, "classifier", None)
#         if not isinstance(classifier, PersistentCosineClassifier):
#             raise RuntimeError(
#                 "NECILModel.classifier must be PersistentCosineClassifier"
#             )
#         return classifier

#     @staticmethod
#     def _parameters_finite(parameters: Sequence[nn.Parameter]) -> bool:
#         return all(bool(torch.isfinite(parameter).all()) for parameter in parameters)

#     @staticmethod
#     def _spatial_symmetry_augment(
#         patch: Tensor,
#         *,
#         generator: torch.Generator,
#     ) -> Tensor:
#         """Apply one D4 square symmetry per sample, spatial axes only."""
#         if patch.ndim != 4 or patch.size(-1) != patch.size(-2):
#             raise RuntimeError(
#                 "spatial symmetry augmentation requires [N,C,H,W] square patches"
#             )
#         codes = torch.randint(
#             0,
#             8,
#             (patch.size(0),),
#             generator=generator,
#             device="cpu",
#         )
#         rows: list[Tensor] = []
#         for sample, code in zip(patch, codes.tolist()):
#             value = sample
#             if int(code) >= 4:
#                 value = torch.flip(value, dims=(-1,))
#             rotation = int(code) % 4
#             if rotation:
#                 value = torch.rot90(value, k=rotation, dims=(-2, -1))
#             rows.append(value)
#         result = torch.stack(rows, dim=0)
#         if result.shape != patch.shape or not bool(torch.isfinite(result).all()):
#             raise RuntimeError("spatial symmetry augmentation produced invalid patches")
#         return result

#     @torch.no_grad()
#     def _ensure_spectral_normalization(self, loader: Any) -> None:
#         if bool(self.model.spectral_normalization_fitted):
#             return
#         spectra: list[Tensor] = []
#         for batch in loader:
#             _, center_spectrum, _ = self.unpack_batch(batch)
#             spectra.append(center_spectrum.detach())
#         if not spectra:
#             raise RuntimeError("phase-0 spectral-normalization loader is empty")
#         self.model.fit_spectral_normalization(
#             torch.cat(spectra, dim=0),
#             overwrite=False,
#         )
#         if not bool(self.model.spectral_normalization_fitted):
#             raise RuntimeError("spectral normalization was not fitted")

#     def _validation_enabled(self) -> bool:
#         value = getattr(self.dataset, "validation_enabled", None)
#         if not isinstance(value, bool):
#             raise RuntimeError("dataset.validation_enabled must be bool")
#         return value

#     def _phase_train_loader(
#         self,
#         *,
#         batch_size: int,
#         shuffle: bool,
#     ) -> DataLoader:
#         return self.dataset.get_phase_dataloader(
#             0,
#             split="train",
#             batch_size=batch_size,
#             shuffle=shuffle,
#         )

#     def train_base_phase(
#         self,
#         phase: int,
#         epochs: int,
#         batch_size: int = 64,
#         lr: float = 1e-4,
#     ) -> Dict[str, Any]:
#         if int(phase) != 0:
#             raise ValueError("BasePhaseTrainer handles phase 0 only")

#         epoch_count = _positive_int(epochs, name="epochs")
#         train_batch_size = _positive_int(batch_size, name="batch_size")
#         learning_rate = _finite_positive("lr", lr)

#         self.model.validate_model_state()
#         classifier = self._require_classifier()
#         if len(self.model.geometry_bank) != 0:
#             raise RuntimeError("phase 0 requires an empty persistent geometry bank")
#         if classifier.class_count != 0:
#             raise RuntimeError("phase 0 requires an empty persistent classifier")
#         if getattr(self.dataset, "current_phase", None) is not None:
#             raise RuntimeError("dataset already has an active phase")

#         self.dataset.start_phase(0)
#         class_ids = self._base_class_ids(self.dataset.get_new_classes(0))

#         eval_batch_size = _positive_int(
#             self.cfg("eval_batch_size", 256),
#             name="eval_batch_size",
#         )
#         train_eval_loader = self._phase_train_loader(
#             batch_size=eval_batch_size,
#             shuffle=False,
#         )
#         self._ensure_spectral_normalization(train_eval_loader)

#         # Expansion replaces the classifier Parameter object, so it must happen
#         # before optimizer construction.
#         self.model.add_classifier_classes(class_ids)
#         classifier = self._require_classifier()
#         if tuple(classifier.seen_class_ids) != class_ids:
#             raise RuntimeError("phase-0 classifier class IDs are incorrect")

#         train_loader = self._phase_train_loader(
#             batch_size=train_batch_size,
#             shuffle=True,
#         )
#         val_loader = None
#         if self._validation_enabled():
#             val_loader = self.dataset.get_phase_dataloader(
#                 0,
#                 split="val",
#                 batch_size=eval_batch_size,
#                 shuffle=False,
#             )

#         backbone_parameters = [
#             parameter
#             for parameter in self.model.backbone.parameters()
#             if parameter.requires_grad
#         ]
#         classifier_parameters = [
#             parameter
#             for parameter in classifier.parameters()
#             if parameter.requires_grad
#         ]
#         if not backbone_parameters or not classifier_parameters:
#             raise RuntimeError(
#                 "phase-0 training requires trainable backbone and classifier parameters"
#             )
#         trainable = backbone_parameters + classifier_parameters

#         backbone_weight_decay = _finite_nonnegative(
#             "weight_decay",
#             self.cfg("weight_decay", 1e-4),
#         )
#         optimizer = torch.optim.AdamW(
#             [
#                 {
#                     "params": backbone_parameters,
#                     "weight_decay": backbone_weight_decay,
#                 },
#                 {
#                     # Classifier rows are direction-only because forward()
#                     # normalizes w_c.  Weight decay on their raw norm has no
#                     # decision meaning, so do not regularize that nuisance scale.
#                     "params": classifier_parameters,
#                     "weight_decay": 0.0,
#                 },
#             ],
#             lr=learning_rate,
#         )
#         scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
#             optimizer,
#             T_max=epoch_count,
#             eta_min=0.0,
#         )
#         gradient_clip = _finite_nonnegative(
#             "gradient_clip",
#             self.cfg("gradient_clip", 0.0),
#         )

#         augmentation_generator = torch.Generator(device="cpu")
#         augmentation_generator.manual_seed(int(self.cfg("seed", 0)))
#         class_tensor = torch.tensor(class_ids, device=self.device, dtype=torch.long)
#         history: list[Dict[str, Any]] = []

#         progress = tqdm(
#             range(epoch_count),
#             desc="Phase 0 classifier representation learning",
#             unit="epoch",
#             dynamic_ncols=True,
#             file=sys.stdout,
#         )

#         for epoch in progress:
#             self.model.train()
#             epoch_lr = float(optimizer.param_groups[0]["lr"])
#             if not math.isfinite(epoch_lr) or epoch_lr < 0.0:
#                 raise RuntimeError("optimizer learning rate is invalid")

#             weighted_loss = 0.0
#             sample_count = 0
#             correct = 0
#             class_correct = {class_id: 0 for class_id in class_ids}
#             class_total = {class_id: 0 for class_id in class_ids}

#             for batch in train_loader:
#                 patch, spectrum, labels = self.unpack_batch(batch)
#                 patch = self._spatial_symmetry_augment(
#                     patch,
#                     generator=augmentation_generator,
#                 )
#                 if not set(labels.detach().cpu().tolist()).issubset(set(class_ids)):
#                     raise RuntimeError("phase-0 batch contains a non-base class")

#                 optimizer.zero_grad(set_to_none=True)
#                 backbone_output = self.model.extract_features(
#                     patch,
#                     center_spectrum=spectrum,
#                     return_aux=False,
#                 )
#                 classifier_output = self.model.classify_features(
#                     backbone_output.ordered_features,
#                     class_ids=class_ids,
#                 )
#                 objective = cosine_classification_objective(
#                     output=classifier_output,
#                     labels_global=labels,
#                 )
#                 objective.loss.backward()

#                 if any(
#                     parameter.grad is not None
#                     and not bool(torch.isfinite(parameter.grad).all())
#                     for parameter in trainable
#                 ):
#                     raise RuntimeError("phase-0 optimization produced NaN/Inf gradients")

#                 if gradient_clip > 0.0:
#                     torch.nn.utils.clip_grad_norm_(
#                         trainable,
#                         gradient_clip,
#                         error_if_nonfinite=True,
#                     )
#                 optimizer.step()
#                 if not self._parameters_finite(trainable):
#                     raise RuntimeError("phase-0 optimization produced NaN/Inf parameters")

#                 n = int(labels.numel())
#                 weighted_loss += float(objective.loss.detach().item()) * n
#                 sample_count += n
#                 pred = classifier_output.prediction.detach()
#                 correct += int(pred.eq(labels).sum().item())
#                 for class_id in class_ids:
#                     mask = labels.eq(class_id)
#                     count = int(mask.sum().item())
#                     if count:
#                         class_total[class_id] += count
#                         class_correct[class_id] += int(pred[mask].eq(labels[mask]).sum().item())

#             if sample_count <= 0:
#                 raise RuntimeError("phase-0 train loader yielded no samples")
#             missing_classes = [class_id for class_id in class_ids if class_total[class_id] == 0]
#             if missing_classes:
#                 raise RuntimeError(
#                     f"phase-0 epoch did not observe classes {missing_classes}"
#                 )

#             train_loss = weighted_loss / float(sample_count)
#             train_oa_aug = correct / float(sample_count)
#             train_ba_aug = sum(
#                 class_correct[class_id] / float(class_total[class_id])
#                 for class_id in class_ids
#             ) / float(len(class_ids))

#             # Reporting uses deterministic unaugmented data and the deployed
#             # classifier. Geometry does not yet exist and cannot influence it.
#             train_metrics = self.evaluate_loader(
#                 train_eval_loader,
#                 class_ids=class_ids,
#             )
#             validation_metrics = (
#                 None
#                 if val_loader is None
#                 else self.evaluate_loader(val_loader, class_ids=class_ids)
#             )

#             scheduler.step()
#             next_lr = float(optimizer.param_groups[0]["lr"])
#             if not math.isfinite(next_lr) or next_lr < 0.0:
#                 raise RuntimeError("scheduled learning rate is invalid")

#             record: Dict[str, Any] = {
#                 "epoch": epoch + 1,
#                 "classification_cross_entropy": float(train_loss),
#                 "learning_rate": epoch_lr,
#                 "next_learning_rate": next_lr,
#                 "augmented_training_accuracy": float(train_oa_aug),
#                 "augmented_training_balanced_accuracy": float(train_ba_aug),
#                 "train": train_metrics,
#                 "validation": validation_metrics,
#                 "geometry_used_for_optimization": False,
#                 "spatial_augmentation": (
#                     "TRAIN-only D4 square-patch symmetry; channel order and raw "
#                     "center spectrum unchanged"
#                 ),
#             }
#             history.append(record)

#             message = (
#                 f"[Epoch {epoch + 1:03d}/{epoch_count:03d}] "
#                 f"LR={epoch_lr:.3e} | CE={train_loss:.4f} | "
#                 f"train_OA={100.0 * train_metrics['overall_accuracy']:.2f}% | "
#                 f"train_BA={100.0 * train_metrics['balanced_accuracy']:.2f}% | "
#                 f"CosGap={train_metrics['mean_cosine_margin']:.4f}"
#             )
#             if validation_metrics is not None:
#                 message += (
#                     f" | val_OA={100.0 * validation_metrics['overall_accuracy']:.2f}%"
#                     f" | val_BA={100.0 * validation_metrics['balanced_accuracy']:.2f}%"
#                     f" | val_CosGap={validation_metrics['mean_cosine_margin']:.4f}"
#                 )
#             tqdm.write(message, file=sys.stdout)
#             progress.set_postfix(
#                 lr=f"{epoch_lr:.2e}",
#                 ce=f"{train_loss:.4f}",
#                 val_ba=(
#                     f"{100.0 * validation_metrics['balanced_accuracy']:.2f}%"
#                     if validation_metrics is not None
#                     else "-"
#                 ),
#             )

#         progress.close()
#         if len(history) != epoch_count:
#             raise RuntimeError("phase-0 training did not complete all epochs")

#         # Fixed-epoch contract: no validation checkpoint is restored.  Geometry
#         # is constructed once, here, from deterministic complete TRAIN Q(x).
#         self.model.eval()
#         final_train = self.collect_ordered_features(train_eval_loader)
#         if len(self.model.geometry_bank) != 0:
#             raise RuntimeError("persistent geometry changed before phase-0 commit")
#         self.model.add_classes(
#             final_train["ordered_features"],
#             final_train["labels"],
#             class_ids=class_ids,
#         )
#         if tuple(self.model.committed_class_ids) != class_ids:
#             raise RuntimeError("phase-0 geometry commit produced incorrect class IDs")
#         if tuple(self.model.classifier_class_ids) != class_ids:
#             raise RuntimeError("phase-0 classifier class IDs changed unexpectedly")

#         # Final classifier metrics remain classifier-only even though geometry is
#         # now present in the model.
#         train_metrics = self.evaluate_loader(
#             train_eval_loader,
#             class_ids=class_ids,
#         )
#         validation_metrics = (
#             None
#             if val_loader is None
#             else self.evaluate_loader(val_loader, class_ids=class_ids)
#         )

#         self.dataset.finalize_phase(0)
#         test_loader = self.dataset.get_reporting_dataloader(
#             0,
#             batch_size=eval_batch_size,
#         )
#         test_metrics = self.evaluate_loader(
#             test_loader,
#             class_ids=class_ids,
#         )

#         return {
#             "phase": 0,
#             "class_ids": list(class_ids),
#             "history": history,
#             "final_epoch": epoch_count,
#             "train": train_metrics,
#             "validation": validation_metrics,
#             "test": test_metrics,
#             "geometry_committed": True,
#             "phase_finalized": True,
#         }


# __all__ = ["BasePhaseTrainer"]

