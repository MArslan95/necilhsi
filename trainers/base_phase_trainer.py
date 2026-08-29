from __future__ import annotations

"""Phase-0 optimization for the locked SCOPE-HSI architecture.

Phase-0 roles
-------------
1. Backbone: learn the single ordered spectral-spatial representation Q.
2. Persistent angular classifier: learn base-class decision directions on Q.
3. Historical class statistics {I_c,S_c,N_c}: absent from optimization and
   prediction during phase 0; estimated once from deterministic complete TRAIN
   Q after the final requested epoch.

The base phase therefore does not contain transport, preservation projection,
replay, teacher/KD, geometry loss, margin, temperature, or a second feature
representation.
"""

from contextlib import contextmanager
import math
import sys
from numbers import Integral, Real
from typing import Any, Dict, Mapping, Sequence

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


def _explicit_bool(name: str, value: object) -> bool:
    if isinstance(value, bool):
        return value
    raise TypeError(f"{name} must be bool")


class BasePhaseTrainer:
    """Fixed-epoch phase-0 learning for one Q-space and one classifier."""

    BASE_RESULT_FIELDS = frozenset(
        {
            "phase",
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
        """Apply one D4 transform per HSI patch; spectral/channel order is untouched."""
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

    @torch.no_grad()
    def _prepare_base_train_statistics(
        self,
        loader: Any,
        *,
        class_ids: tuple[int, ...],
    ) -> tuple[Tensor, int]:
        """Fit phase-0 spectral normalization and derive fixed CE class weights.

        Only phase-0 TRAIN labels/spectra are used.  For class c with N_c samples,

            weight_c = N / (C N_c).

        These weights make sample-mean weighted CE estimate the macro-average
        empirical class CE without a tunable balancing coefficient.
        """
        counts = {class_id: 0 for class_id in class_ids}
        spectra: list[Tensor] = []
        total = 0

        for batch in loader:
            _, center_spectrum, labels = self.unpack_batch(batch)
            observed = set(int(v) for v in labels.detach().cpu().unique().tolist())
            outside = sorted(observed - set(class_ids))
            if outside:
                raise RuntimeError(
                    f"phase-0 TRAIN loader contains non-base classes {outside}"
                )
            for class_id in class_ids:
                count = int(labels.eq(class_id).sum().item())
                counts[class_id] += count
                total += count
            if not bool(self.model.spectral_normalization_fitted):
                spectra.append(center_spectrum.detach())

        if total <= 0:
            raise RuntimeError("phase-0 TRAIN loader is empty")
        missing = [class_id for class_id in class_ids if counts[class_id] <= 0]
        if missing:
            raise RuntimeError(f"phase-0 TRAIN split lacks classes {missing}")

        if not bool(self.model.spectral_normalization_fitted):
            if not spectra:
                raise RuntimeError("phase-0 spectral-normalization data are empty")
            self.model.fit_spectral_normalization(
                torch.cat(spectra, dim=0),
                overwrite=False,
            )
        if not bool(self.model.spectral_normalization_fitted):
            raise RuntimeError("spectral normalization was not fitted")

        class_count = len(class_ids)
        weights = torch.tensor(
            [total / float(class_count * counts[class_id]) for class_id in class_ids],
            device=self.device,
            dtype=self.model.dtype,
        )
        if not bool(torch.isfinite(weights).all()) or bool((weights <= 0).any()):
            raise RuntimeError("derived class-balance weights are invalid")
        return weights, total

    @contextmanager
    def _temporary_eval_state(self):
        states = {module: bool(module.training) for module in self.model.modules()}
        try:
            self.model.eval()
            yield
        finally:
            for module, state in states.items():
                module.training = state

    @torch.no_grad()
    def _evaluate_classifier_only(
        self,
        loader: Any,
        *,
        class_ids: tuple[int, ...],
    ) -> Dict[str, Any]:
        """Evaluate the deployed classifier without requiring committed geometry."""
        cosine_rows: list[Tensor] = []
        prediction_rows: list[Tensor] = []
        label_rows: list[Tensor] = []
        feature_rms_rows: list[Tensor] = []
        ids_reference: Tensor | None = None

        with self._temporary_eval_state():
            for batch in loader:
                patch, spectrum, labels = self.unpack_batch(batch)
                outside = sorted(
                    set(int(v) for v in labels.detach().cpu().unique().tolist())
                    - set(class_ids)
                )
                if outside:
                    raise RuntimeError(
                        f"evaluation loader contains classes outside phase 0: {outside}"
                    )

                features = self.model.extract_features(
                    patch,
                    center_spectrum=spectrum,
                    return_aux=False,
                ).ordered_features
                output = self.model.classify_features(
                    features,
                    class_ids=class_ids,
                )

                if ids_reference is None:
                    ids_reference = output.class_ids.detach().clone()
                elif not bool(torch.equal(ids_reference, output.class_ids)):
                    raise RuntimeError("classifier column order changed across batches")

                flat = features.reshape(features.size(0), -1)
                feature_rms = torch.linalg.vector_norm(flat, ord=2, dim=1) / math.sqrt(
                    float(flat.size(1))
                )
                cosine_rows.append(output.cosine_similarity.detach())
                prediction_rows.append(output.prediction.detach())
                label_rows.append(labels.detach())
                feature_rms_rows.append(feature_rms.detach())

        if not cosine_rows or ids_reference is None:
            raise RuntimeError("cannot evaluate an empty loader")

        cosine = torch.cat(cosine_rows, dim=0)
        prediction = torch.cat(prediction_rows, dim=0)
        labels = torch.cat(label_rows, dim=0)
        feature_rms = torch.cat(feature_rms_rows, dim=0)

        expected_ids = torch.tensor(class_ids, device=self.device, dtype=torch.long)
        if not bool(torch.equal(ids_reference, expected_ids)):
            raise RuntimeError("classifier column order disagrees with requested class IDs")

        matches = labels.unsqueeze(1).eq(ids_reference.unsqueeze(0))
        counts = matches.sum(dim=1)
        if not bool(counts.eq(1).all()):
            raise RuntimeError("evaluation labels do not map uniquely to classifier columns")
        targets_local = matches.to(torch.long).argmax(dim=1)
        pred_local = cosine.argmax(dim=1)
        expected_prediction = ids_reference.index_select(0, pred_local)
        if not bool(torch.equal(prediction, expected_prediction)):
            raise RuntimeError("classifier prediction disagrees with cosine argmax")

        rows = torch.arange(labels.numel(), device=self.device)
        target_cosine = cosine[rows, targets_local]
        rival = cosine.clone()
        rival[rows, targets_local] = -torch.inf
        nearest_rival = rival.max(dim=1).values
        margin = target_cosine - nearest_rival

        per_class_accuracy: Dict[int, float] = {}
        per_class_margin: Dict[int, float] = {}
        class_counts: Dict[int, int] = {}
        confusion = torch.zeros(
            len(class_ids),
            len(class_ids),
            device=self.device,
            dtype=torch.long,
        )

        for target_index, predicted_index in zip(
            targets_local.detach().cpu().tolist(),
            pred_local.detach().cpu().tolist(),
        ):
            confusion[int(target_index), int(predicted_index)] += 1

        for local_index, class_id in enumerate(class_ids):
            mask = targets_local.eq(local_index)
            count = int(mask.sum().item())
            if count <= 0:
                raise RuntimeError(f"evaluation split lacks class {class_id}")
            class_counts[class_id] = count
            per_class_accuracy[class_id] = float(
                prediction[mask].eq(labels[mask]).to(torch.float32).mean().item()
            )
            per_class_margin[class_id] = float(margin[mask].mean().item())

        overall_accuracy = float(prediction.eq(labels).to(torch.float32).mean().item())
        balanced_accuracy = sum(per_class_accuracy.values()) / float(len(class_ids))
        minimum_class_accuracy = min(per_class_accuracy.values())

        confusion_float = confusion.to(torch.float64)
        total = float(confusion_float.sum().item())
        observed = float(torch.trace(confusion_float).item()) / total
        row_sum = confusion_float.sum(dim=1)
        col_sum = confusion_float.sum(dim=0)
        expected = float((row_sum * col_sum).sum().item()) / (total * total)
        denominator = 1.0 - expected
        kappa = 0.0 if denominator == 0.0 else (observed - expected) / denominator

        return {
            "overall_accuracy": overall_accuracy,
            "accuracy": overall_accuracy,
            "balanced_accuracy": float(balanced_accuracy),
            "minimum_class_accuracy": float(minimum_class_accuracy),
            "kappa": float(kappa),
            "confusion_matrix": confusion.detach().cpu().tolist(),
            "evaluated_class_ids": list(class_ids),
            "target_class_ids": list(class_ids),
            "class_counts": class_counts,
            "per_class_accuracy": per_class_accuracy,
            "mean_cosine_margin": float(margin.mean().item()),
            "per_class_mean_cosine_margin": per_class_margin,
            "mean_target_cosine_similarity": float(target_cosine.mean().item()),
            "mean_nearest_rival_cosine_similarity": float(nearest_rival.mean().item()),
            "mean_feature_rms": float(feature_rms.mean().item()),
            "mean_structural_feature_rms": float(feature_rms.mean().item()),
            "classifier_rule": (
                "persistent normalized angular classifier; "
                "prediction=argmax deployed logits/cosine"
            ),
        }

    def train_base_phase(
        self,
        phase: int,
        epochs: int,
        batch_size: int = 64,
        lr: float = 1e-4,
    ) -> Dict[str, Any]:
        if isinstance(phase, bool) or int(phase) != 0:
            raise ValueError("BasePhaseTrainer handles phase 0 only")

        epoch_count = _positive_int(epochs, name="epochs")
        train_batch_size = _positive_int(batch_size, name="batch_size")
        learning_rate = _finite_positive("lr", lr)

        self.model.validate_model_state()
        classifier = self._require_classifier()
        if len(self.model.geometry_bank) != 0:
            raise RuntimeError("phase 0 requires empty historical class statistics")
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
        class_balance_weights, expected_train_samples = self._prepare_base_train_statistics(
            train_eval_loader,
            class_ids=class_ids,
        )

        # Expansion replaces the classifier Parameter, therefore it must happen
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
                    # Only classifier direction affects the deployed decision.
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

        use_d4 = _explicit_bool(
            "spatial_d4_augmentation",
            self.cfg("spatial_d4_augmentation", True),
        )
        augmentation_generator = torch.Generator(device="cpu")
        augmentation_generator.manual_seed(int(self.cfg("seed", 0)))

        history: list[Dict[str, Any]] = []
        progress = tqdm(
            range(epoch_count),
            desc="Phase 0 SCOPE-HSI representation learning",
            unit="epoch",
            dynamic_ncols=True,
            file=sys.stdout,
        )

        for epoch in progress:
            self.model.train()
            epoch_lr = float(optimizer.param_groups[0]["lr"])
            if not math.isfinite(epoch_lr) or epoch_lr < 0.0:
                raise RuntimeError("optimizer learning rate is invalid")

            weighted_loss_sum = 0.0
            sample_count = 0

            for batch in train_loader:
                patch, spectrum, labels = self.unpack_batch(batch)
                if use_d4:
                    patch = self._spatial_symmetry_augment(
                        patch,
                        generator=augmentation_generator,
                    )

                optimizer.zero_grad(set_to_none=True)
                ordered_features = self.model.extract_features(
                    patch,
                    center_spectrum=spectrum,
                    return_aux=False,
                ).ordered_features
                classifier_output = self.model.classify_features(
                    ordered_features,
                    class_ids=class_ids,
                )
                objective = cosine_classification_objective(
                    output=classifier_output,
                    labels_global=labels,
                    class_balance_weights=class_balance_weights,
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
                weighted_loss_sum += float(objective.loss.detach().item()) * n
                sample_count += n

            if sample_count <= 0:
                raise RuntimeError("phase-0 train loader yielded no samples")
            if sample_count != expected_train_samples:
                raise RuntimeError(
                    "phase-0 train loader did not traverse the complete TRAIN split: "
                    f"observed={sample_count}, expected={expected_train_samples}"
                )
            train_loss = weighted_loss_sum / float(sample_count)

            # Reporting is deterministic, unaugmented, and classifier-only.
            # Historical statistics still do not exist at this point.
            train_metrics = self._evaluate_classifier_only(
                train_eval_loader,
                class_ids=class_ids,
            )
            validation_metrics = (
                None
                if val_loader is None
                else self._evaluate_classifier_only(
                    val_loader,
                    class_ids=class_ids,
                )
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

        # Fixed-epoch contract: no best-validation state is restored.  The final
        # unaugmented TRAIN Q is the sole source of committed base statistics.
        self.model.eval()
        final_train = self.collect_ordered_features(train_eval_loader)
        if len(self.model.geometry_bank) != 0:
            raise RuntimeError("historical class statistics changed before phase-0 commit")
        self.model.add_classes(
            final_train["ordered_features"],
            final_train["labels"],
            class_ids=class_ids,
        )
        if tuple(self.model.committed_class_ids) != class_ids:
            raise RuntimeError("phase-0 class-statistic commit produced incorrect IDs")
        if tuple(self.model.classifier_class_ids) != class_ids:
            raise RuntimeError("phase-0 classifier class IDs changed unexpectedly")
        self.model.validate_finalized_state()

        # Final predictive reporting remains classifier-only.  Geometry/class
        # structure is reported separately and never changes prediction here.
        train_metrics = self._evaluate_classifier_only(
            train_eval_loader,
            class_ids=class_ids,
        )
        validation_metrics = (
            None
            if val_loader is None
            else self._evaluate_classifier_only(
                val_loader,
                class_ids=class_ids,
            )
        )

        self.dataset.finalize_phase(0)
        test_loader = self.dataset.get_reporting_dataloader(
            0,
            batch_size=eval_batch_size,
        )
        test_metrics = self._evaluate_classifier_only(
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

