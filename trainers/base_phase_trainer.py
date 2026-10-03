from __future__ import annotations
import math
import sys
from numbers import Integral, Real
from typing import Any, Dict, Mapping

import torch
from tqdm import tqdm

from losses.loss import classification_objective, whole_cds_contrastive_loss
from models.classifier import CosineClassificationHead
from models.geometry_bank import ClassStructureBatch, PhysicalSpectralStructureBatch


def _positive_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (Integral, Real)):
        raise ValueError(f"{name} must be a positive integer")
    number = float(value)
    if not math.isfinite(number) or not number.is_integer() or number <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(number)


def _finite_positive(name: str, value: object) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return result


def _finite_nonnegative(name: str, value: object) -> float:
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise ValueError(f"{name} must be finite and non-negative")
    return result


def _worst_class_accuracy(metrics: Mapping[str, Any]) -> tuple[int, float]:
    values = metrics.get("per_class_accuracy")
    if not isinstance(values, Mapping) or not values:
        raise RuntimeError("evaluation metrics lack per_class_accuracy")
    parsed = [(int(key), float(value)) for key, value in values.items()]
    return min(parsed, key=lambda item: (item[1], item[0]))


class BasePhaseTrainer:
    BASE_RESULT_FIELDS = frozenset(
        {
            "phase",
            "class_ids",
            "history",
            "final_epoch",
            "train",
            "validation",
            "test",
            "structure_committed",
            "phase_finalized",
        }
    )

    def _require_classifier(self) -> CosineClassificationHead:
        classifier = getattr(self.model, "classifier", None)
        if (
            not isinstance(classifier, CosineClassificationHead)
            or classifier.parameter_count <= 0
        ):
            raise RuntimeError("model.classifier must be the CosineClassificationHead")
        return classifier

    @torch.no_grad()
    def _prepare_train_state(
        self,
        loader: Any,
        *,
        class_ids: tuple[int, ...],
    ) -> tuple[torch.Tensor, int]:
        counts = {class_id: 0 for class_id in class_ids}
        spectra: list[torch.Tensor] = []
        total = 0
        for batch in loader:
            _, center_spectrum, labels = self.unpack_batch(batch)
            observed = set(
                int(value) for value in labels.unique().detach().cpu().tolist()
            )
            outside = observed - set(class_ids)
            if outside:
                raise RuntimeError(
                    f"Phase-0 TRAIN loader contains non-base classes {sorted(outside)}"
                )
            if not self.model.spectral_normalization_fitted:
                spectra.append(center_spectrum.detach())
            for class_id in class_ids:
                count = int(labels.eq(class_id).sum().item())
                counts[class_id] += count
                total += count

        if total <= 0 or any(counts[class_id] <= 0 for class_id in class_ids):
            raise RuntimeError("Phase-0 TRAIN split is empty or misses a base class")
        if not self.model.spectral_normalization_fitted:
            if not spectra:
                raise RuntimeError("Phase-0 spectra are unavailable for normalization")
            self.model.fit_spectral_normalization(
                torch.cat(spectra, dim=0), overwrite=False
            )

        class_count = len(class_ids)
        weights = torch.tensor(
            [total / float(class_count * counts[class_id]) for class_id in class_ids],
            device=self.device,
            dtype=self.model.dtype,
        )
        return weights, total

    @torch.no_grad()
    def _fixed_physical_structure(
        self,
        loader: Any,
        *,
        class_ids: tuple[int, ...],
    ) -> PhysicalSpectralStructureBatch:
        spectra: list[torch.Tensor] = []
        labels_rows: list[torch.Tensor] = []
        for batch in loader:
            _, spectrum, labels = self.unpack_batch(batch)
            spectra.append(spectrum.detach())
            labels_rows.append(labels.detach())
        if not spectra:
            raise RuntimeError(
                "cannot estimate physical CDS from an empty Phase-0 loader"
            )
        return self.model.estimate_physical_class_structure(
            torch.cat(spectra, dim=0),
            torch.cat(labels_rows, dim=0),
            class_ids=class_ids,
        )

    @torch.no_grad()
    def _working_structure(
        self,
        loader: Any,
        *,
        class_ids: tuple[int, ...],
        physical: PhysicalSpectralStructureBatch,
    ) -> ClassStructureBatch:
        rows = self.collect_ordered_features(loader)
        observed = tuple(
            sorted(
                int(value) for value in rows["labels"].unique().detach().cpu().tolist()
            )
        )
        if observed != tuple(sorted(class_ids)):
            raise RuntimeError("Phase-0 TRAIN rows do not match the base class set")
        return self.model.estimate_class_structure(
            rows["ordered_features"],
            None,
            rows["labels"],
            class_ids=class_ids,
            physical_structure=physical,
        )

    def train_base_phase(
        self,
        phase: int,
        epochs: int,
        batch_size: int = 64,
        lr: float = 1e-4,
    ) -> Dict[str, Any]:
        if isinstance(phase, bool) or int(phase) != 0:
            raise ValueError("BasePhaseTrainer handles Phase 0 only")

        epoch_count = _positive_int(epochs, name="epochs")
        train_batch_size = _positive_int(batch_size, name="batch_size")
        learning_rate = _finite_positive("lr", lr)
        eval_batch_size = _positive_int(
            self.cfg("eval_batch_size", 256), name="eval_batch_size"
        )
        weight_decay = _finite_nonnegative(
            "weight_decay", self.cfg("weight_decay", 1e-4)
        )
        gradient_clip = _finite_nonnegative(
            "gradient_clip", self.cfg("gradient_clip", 0.0)
        )
        contrastive_weight = _finite_nonnegative(
            "contrastive_weight", self.cfg("contrastive_weight", 1.0)
        )

        self.model.validate_model_state()
        self.model.set_base_trainability()
        classifier = self._require_classifier()
        if len(self.model.structure_bank) != 0:
            raise RuntimeError("Phase 0 requires an empty CDS bank")
        if getattr(self.dataset, "current_phase", None) is not None:
            raise RuntimeError("dataset already has an active phase")

        self.dataset.start_phase(0)
        class_ids = tuple(
            self._validated_class_ids(
                self.dataset.get_new_classes(0), name="phase_0_classes"
            )
        )
        if len(class_ids) < 2:
            raise RuntimeError("Phase 0 requires at least two classes")
        class_ids_tensor = torch.tensor(class_ids, device=self.device, dtype=torch.long)

        train_eval_loader = self.dataset.get_phase_dataloader(
            0, split="train", batch_size=eval_batch_size, shuffle=False
        )
        class_weights, expected_samples = self._prepare_train_state(
            train_eval_loader, class_ids=class_ids
        )
        physical_structure = self._fixed_physical_structure(
            train_eval_loader, class_ids=class_ids
        )

        train_loader = self.dataset.get_phase_dataloader(
            0, split="train", batch_size=train_batch_size, shuffle=True
        )
        val_loader = (
            self.dataset.get_phase_dataloader(
                0, split="val", batch_size=eval_batch_size, shuffle=False
            )
            if self.dataset.validation_enabled
            else None
        )

        backbone_trainable = [
            p for p in self.model.backbone.parameters() if p.requires_grad
        ]
        if not backbone_trainable:
            raise RuntimeError("Phase-0 backbone has no trainable parameters")

        backbone_optimizer = torch.optim.AdamW(
            [{"params": backbone_trainable, "weight_decay": weight_decay}],
            lr=learning_rate,
        )
        # Future classifier rows are intentionally allocated but inactive in
        # Phase 0. Decoupled AdamW weight decay would move those rows even when
        # their gradients are masked, so classifier weight decay must be zero.
        classifier_optimizer = torch.optim.AdamW(
            list(classifier.parameters()),
            lr=learning_rate,
            weight_decay=0.0,
        )
        backbone_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            backbone_optimizer, T_max=epoch_count, eta_min=0.0
        )
        classifier_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            classifier_optimizer, T_max=epoch_count, eta_min=0.0
        )

        history: list[dict[str, Any]] = []
        progress = tqdm(
            range(epoch_count),
            desc="Phase 0 base representation",
            unit="epoch",
            dynamic_ncols=True,
            file=sys.stdout,
        )

        for epoch in progress:
            # Refresh Whole-CDS in the current Q coordinate before every epoch.
            # Its intra structure, inter-class risk and native spectral structure
            # directly shape the risk-aware contrastive separation objective.
            self.model.eval()
            epoch_structure = self._working_structure(
                train_eval_loader, class_ids=class_ids, physical=physical_structure
            )
            epoch_cds = self.model.structure_bank.build_cds(epoch_structure)
            self.model.train()
            epoch_lr = float(backbone_optimizer.param_groups[0]["lr"])
            epoch_classifier_lr = float(classifier_optimizer.param_groups[0]["lr"])
            classification_sum = 0.0
            contrastive_sum = 0.0
            contrastive_positive_sum = 0.0
            contrastive_negative_sum = 0.0
            contrastive_gap_sum = 0.0
            contrastive_anchor_fraction_sum = 0.0
            total_sum = 0.0
            correct = 0
            sample_count = 0
            per_class_correct = {class_id: 0 for class_id in class_ids}
            per_class_total = {class_id: 0 for class_id in class_ids}

            for batch in train_loader:
                patch, spectrum, labels = self.unpack_batch(batch)
                backbone_optimizer.zero_grad(set_to_none=True)
                classifier_optimizer.zero_grad(set_to_none=True)

                q = self.model.extract_features(
                    patch, center_spectrum=spectrum, return_aux=False
                ).ordered_features
                output = classifier(
                    q, class_ids=class_ids_tensor, trainable_class_ids=class_ids_tensor
                )
                targets_local = (
                    labels.unsqueeze(1)
                    .eq(class_ids_tensor.unsqueeze(0))
                    .to(torch.long)
                    .argmax(dim=1)
                )
                sample_weights = class_weights.index_select(0, targets_local)
                classification = classification_objective(
                    logits=output.logits,
                    class_ids=output.class_ids,
                    labels_global=labels,
                    sample_weights=sample_weights,
                )
                physical_evidence = self.model.physical_spectral_evidence(
                    spectrum
                ).detach()
                contrastive = whole_cds_contrastive_loss(
                    current_features=q,
                    labels_global=labels,
                    physical_spectral_evidence=physical_evidence,
                    seen_cds=epoch_cds,
                    historical_class_count=0,
                    logit_scale=classifier.logit_scale,
                    sample_weights=sample_weights,
                )
                # With no historical classes, the unified continual separation
                # objective reduces to ordinary supervised contrastive learning.
                # W is trained directly from real base samples by deployed cosine CE.
                loss = classification.loss + contrastive_weight * contrastive.loss
                if not bool(torch.isfinite(loss)):
                    raise RuntimeError(
                        "Phase-0 CE + Whole-CDS contrastive objective became NaN or Inf"
                    )
                loss.backward()

                for parameter in backbone_trainable:
                    if parameter.grad is not None and not bool(
                        torch.isfinite(parameter.grad).all()
                    ):
                        raise RuntimeError(
                            "Phase-0 backbone gradient contains NaN or Inf"
                        )
                for parameter in classifier.parameters():
                    if parameter.grad is not None and not bool(
                        torch.isfinite(parameter.grad).all()
                    ):
                        raise RuntimeError(
                            "Phase-0 classifier gradient contains NaN or Inf"
                        )
                if gradient_clip > 0:
                    torch.nn.utils.clip_grad_norm_(
                        backbone_trainable, gradient_clip, error_if_nonfinite=True
                    )

                backbone_optimizer.step()
                classifier_optimizer.step()

                n = int(labels.numel())
                classification_sum += float(classification.loss.detach().item()) * n
                contrastive_sum += float(contrastive.loss.detach().item()) * n
                contrastive_positive_sum += (
                    float(contrastive.positive_similarity_mean.detach().item()) * n
                )
                contrastive_negative_sum += (
                    float(contrastive.negative_similarity_mean.detach().item()) * n
                )
                contrastive_gap_sum += (
                    float(contrastive.similarity_gap.detach().item()) * n
                )
                contrastive_anchor_fraction_sum += (
                    float(contrastive.valid_anchor_fraction.detach().item()) * n
                )
                total_sum += float(loss.detach().item()) * n
                prediction = output.prediction.detach()
                correct += int(prediction.eq(labels).sum().item())
                sample_count += n
                for class_id in class_ids:
                    mask = labels.eq(class_id)
                    count = int(mask.sum().item())
                    per_class_total[class_id] += count
                    if count:
                        per_class_correct[class_id] += int(
                            prediction[mask].eq(labels[mask]).sum().item()
                        )

            if sample_count != expected_samples:
                raise RuntimeError(
                    "Phase-0 loader did not traverse the complete TRAIN split"
                )
            backbone_scheduler.step()
            classifier_scheduler.step()

            self.model.eval()
            refreshed = self._working_structure(
                train_eval_loader, class_ids=class_ids, physical=physical_structure
            )
            train_metrics = self.evaluate_loader(
                train_eval_loader, class_ids=class_ids, target_class_ids=class_ids
            )
            validation_metrics = (
                None
                if val_loader is None
                else self.evaluate_loader(
                    val_loader, class_ids=class_ids, target_class_ids=class_ids
                )
            )

            online_per_class = {
                cid: per_class_correct[cid] / float(per_class_total[cid])
                for cid in class_ids
            }
            train_worst_id, train_worst_acc = _worst_class_accuracy(train_metrics)
            if validation_metrics is None:
                val_worst_id, val_worst_acc = None, None
            else:
                val_worst_id, val_worst_acc = _worst_class_accuracy(validation_metrics)

            record = {
                "epoch": epoch + 1,
                "learning_rate": epoch_lr,
                "classifier_learning_rate": epoch_classifier_lr,
                "classification_loss": classification_sum / sample_count,
                "contrastive_loss": contrastive_sum / sample_count,
                "contrastive_positive_similarity_mean": contrastive_positive_sum
                / sample_count,
                "contrastive_negative_similarity_mean": contrastive_negative_sum
                / sample_count,
                "contrastive_similarity_gap": contrastive_gap_sum / sample_count,
                "contrastive_valid_anchor_fraction": contrastive_anchor_fraction_sum
                / sample_count,
                "total_loss": total_sum / sample_count,
                "logit_scale": float(classifier.logit_scale.detach().item()),
                "online_cosine_oa": correct / float(sample_count),
                "online_cosine_ba": sum(online_per_class.values())
                / float(len(online_per_class)),
                "online_cosine_min_class_accuracy": min(online_per_class.values()),
                "train_oa": float(train_metrics["overall_accuracy"]),
                "train_ba": float(train_metrics["balanced_accuracy"]),
                "train_min_class_accuracy": float(
                    train_metrics["minimum_class_accuracy"]
                ),
                "train_logit_margin": float(train_metrics["mean_logit_margin"]),
                "train_worst_class_id": int(train_worst_id),
                "train_worst_class_accuracy": float(train_worst_acc),
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
                "validation_min_class_accuracy": (
                    None
                    if validation_metrics is None
                    else float(validation_metrics["minimum_class_accuracy"])
                ),
                "validation_logit_margin": (
                    None
                    if validation_metrics is None
                    else float(validation_metrics["mean_logit_margin"])
                ),
                "validation_worst_class_id": val_worst_id,
                "validation_worst_class_accuracy": val_worst_acc,
            }
            history.append(record)

            message = (
                f"[Epoch {epoch + 1:03d}/{epoch_count:03d}] "
                f"CE={record['classification_loss']:.4f} "
                f"Con={record['contrastive_loss']:.4f} "
                f"ConGap={record['contrastive_similarity_gap']:.4f} "
                f"Scale={record['logit_scale']:.3f} "
                f"LR={record['learning_rate']:.2e} HeadLR={record['classifier_learning_rate']:.2e} | "
                f"Online[OA={100*record['online_cosine_oa']:.2f}%,"
                f"BA={100*record['online_cosine_ba']:.2f}%,"
                f"Min={100*record['online_cosine_min_class_accuracy']:.2f}%] | "
                f"Train[OA={100*record['train_oa']:.2f}%,BA={100*record['train_ba']:.2f}%,"
                f"Min={100*record['train_min_class_accuracy']:.2f}%]"
            )
            if validation_metrics is not None:
                message += (
                    f" | Val[OA={100*record['validation_oa']:.2f}%,"
                    f"BA={100*record['validation_ba']:.2f}%,"
                    f"Min={100*record['validation_min_class_accuracy']:.2f}%]"
                )
            tqdm.write(message, file=sys.stdout)

        progress.close()

        self.model.eval()
        final_structure = self._working_structure(
            train_eval_loader, class_ids=class_ids, physical=physical_structure
        )

        self.model.commit_class_structure(final_structure)
        self.model.validate_finalized_state()

        train_metrics = self.evaluate_loader(
            train_eval_loader, class_ids=class_ids, target_class_ids=class_ids
        )
        validation_metrics = (
            None
            if val_loader is None
            else self.evaluate_loader(
                val_loader, class_ids=class_ids, target_class_ids=class_ids
            )
        )
        self.dataset.finalize_phase(0)
        reporting_loader = self.dataset.get_reporting_dataloader(
            0, batch_size=eval_batch_size
        )
        test_metrics = self.evaluate_loader(
            reporting_loader, class_ids=class_ids, target_class_ids=class_ids
        )

        return {
            "phase": 0,
            "class_ids": list(class_ids),
            "history": history,
            "final_epoch": epoch_count,
            "train": train_metrics,
            "validation": validation_metrics,
            "test": test_metrics,
            "structure_committed": True,
            "phase_finalized": True,
            "contrastive_weight": contrastive_weight,
        }


__all__ = ["BasePhaseTrainer"]


# from __future__ import annotations
# import math
# import sys
# from numbers import Integral, Real
# from typing import Any, Dict, Mapping

# import torch
# from tqdm import tqdm

# from losses.loss import classification_objective, whole_cds_contrastive_loss
# from models.classifier import CosineClassificationHead
# from models.geometry_bank import ClassStructureBatch, PhysicalSpectralStructureBatch


# def _positive_int(value: object, *, name: str) -> int:
#     if isinstance(value, bool) or not isinstance(value, (Integral, Real)):
#         raise ValueError(f"{name} must be a positive integer")
#     number = float(value)
#     if not math.isfinite(number) or not number.is_integer() or number <= 0:
#         raise ValueError(f"{name} must be a positive integer")
#     return int(number)


# def _finite_positive(name: str, value: object) -> float:
#     result = float(value)
#     if not math.isfinite(result) or result <= 0:
#         raise ValueError(f"{name} must be finite and positive")
#     return result


# def _finite_nonnegative(name: str, value: object) -> float:
#     result = float(value)
#     if not math.isfinite(result) or result < 0:
#         raise ValueError(f"{name} must be finite and non-negative")
#     return result


# def _worst_class_accuracy(metrics: Mapping[str, Any]) -> tuple[int, float]:
#     values = metrics.get("per_class_accuracy")
#     if not isinstance(values, Mapping) or not values:
#         raise RuntimeError("evaluation metrics lack per_class_accuracy")
#     parsed = [(int(key), float(value)) for key, value in values.items()]
#     return min(parsed, key=lambda item: (item[1], item[0]))


# class BasePhaseTrainer:
#     BASE_RESULT_FIELDS = frozenset(
#         {
#             "phase",
#             "class_ids",
#             "history",
#             "final_epoch",
#             "train",
#             "validation",
#             "test",
#             "structure_committed",
#             "phase_finalized",
#         }
#     )

#     def _require_classifier(self) -> CosineClassificationHead:
#         classifier = getattr(self.model, "classifier", None)
#         if (
#             not isinstance(classifier, CosineClassificationHead)
#             or classifier.parameter_count <= 0
#         ):
#             raise RuntimeError("model.classifier must be the CosineClassificationHead")
#         return classifier

#     @torch.no_grad()
#     def _prepare_train_state(
#         self,
#         loader: Any,
#         *,
#         class_ids: tuple[int, ...],
#     ) -> tuple[torch.Tensor, int]:
#         counts = {class_id: 0 for class_id in class_ids}
#         spectra: list[torch.Tensor] = []
#         total = 0
#         for batch in loader:
#             _, center_spectrum, labels = self.unpack_batch(batch)
#             observed = set(
#                 int(value) for value in labels.unique().detach().cpu().tolist()
#             )
#             outside = observed - set(class_ids)
#             if outside:
#                 raise RuntimeError(
#                     f"Phase-0 TRAIN loader contains non-base classes {sorted(outside)}"
#                 )
#             if not self.model.spectral_normalization_fitted:
#                 spectra.append(center_spectrum.detach())
#             for class_id in class_ids:
#                 count = int(labels.eq(class_id).sum().item())
#                 counts[class_id] += count
#                 total += count

#         if total <= 0 or any(counts[class_id] <= 0 for class_id in class_ids):
#             raise RuntimeError("Phase-0 TRAIN split is empty or misses a base class")
#         if not self.model.spectral_normalization_fitted:
#             if not spectra:
#                 raise RuntimeError("Phase-0 spectra are unavailable for normalization")
#             self.model.fit_spectral_normalization(
#                 torch.cat(spectra, dim=0), overwrite=False
#             )

#         class_count = len(class_ids)
#         weights = torch.tensor(
#             [total / float(class_count * counts[class_id]) for class_id in class_ids],
#             device=self.device,
#             dtype=self.model.dtype,
#         )
#         return weights, total

#     @torch.no_grad()
#     def _fixed_physical_structure(
#         self,
#         loader: Any,
#         *,
#         class_ids: tuple[int, ...],
#     ) -> PhysicalSpectralStructureBatch:
#         spectra: list[torch.Tensor] = []
#         labels_rows: list[torch.Tensor] = []
#         for batch in loader:
#             _, spectrum, labels = self.unpack_batch(batch)
#             spectra.append(spectrum.detach())
#             labels_rows.append(labels.detach())
#         if not spectra:
#             raise RuntimeError(
#                 "cannot estimate physical CDS from an empty Phase-0 loader"
#             )
#         return self.model.estimate_physical_class_structure(
#             torch.cat(spectra, dim=0),
#             torch.cat(labels_rows, dim=0),
#             class_ids=class_ids,
#         )

#     @torch.no_grad()
#     def _working_structure(
#         self,
#         loader: Any,
#         *,
#         class_ids: tuple[int, ...],
#         physical: PhysicalSpectralStructureBatch,
#     ) -> ClassStructureBatch:
#         rows = self.collect_ordered_features(loader)
#         observed = tuple(
#             sorted(
#                 int(value) for value in rows["labels"].unique().detach().cpu().tolist()
#             )
#         )
#         if observed != tuple(sorted(class_ids)):
#             raise RuntimeError("Phase-0 TRAIN rows do not match the base class set")
#         return self.model.estimate_class_structure(
#             rows["ordered_features"],
#             None,
#             rows["labels"],
#             class_ids=class_ids,
#             physical_structure=physical,
#         )

#     def train_base_phase(
#         self,
#         phase: int,
#         epochs: int,
#         batch_size: int = 64,
#         lr: float = 1e-4,
#     ) -> Dict[str, Any]:
#         if isinstance(phase, bool) or int(phase) != 0:
#             raise ValueError("BasePhaseTrainer handles Phase 0 only")

#         epoch_count = _positive_int(epochs, name="epochs")
#         train_batch_size = _positive_int(batch_size, name="batch_size")
#         learning_rate = _finite_positive("lr", lr)
#         eval_batch_size = _positive_int(
#             self.cfg("eval_batch_size", 256), name="eval_batch_size"
#         )
#         weight_decay = _finite_nonnegative(
#             "weight_decay", self.cfg("weight_decay", 1e-4)
#         )
#         gradient_clip = _finite_nonnegative(
#             "gradient_clip", self.cfg("gradient_clip", 0.0)
#         )
#         contrastive_weight = _finite_nonnegative(
#             "contrastive_weight", self.cfg("contrastive_weight", 1.0)
#         )

#         self.model.validate_model_state()
#         self.model.set_base_trainability()
#         classifier = self._require_classifier()
#         if len(self.model.structure_bank) != 0:
#             raise RuntimeError("Phase 0 requires an empty CDS bank")
#         if getattr(self.dataset, "current_phase", None) is not None:
#             raise RuntimeError("dataset already has an active phase")

#         self.dataset.start_phase(0)
#         class_ids = tuple(
#             self._validated_class_ids(
#                 self.dataset.get_new_classes(0), name="phase_0_classes"
#             )
#         )
#         if len(class_ids) < 2:
#             raise RuntimeError("Phase 0 requires at least two classes")
#         class_ids_tensor = torch.tensor(class_ids, device=self.device, dtype=torch.long)

#         train_eval_loader = self.dataset.get_phase_dataloader(
#             0, split="train", batch_size=eval_batch_size, shuffle=False
#         )
#         class_weights, expected_samples = self._prepare_train_state(
#             train_eval_loader, class_ids=class_ids
#         )
#         physical_structure = self._fixed_physical_structure(
#             train_eval_loader, class_ids=class_ids
#         )

#         train_loader = self.dataset.get_phase_dataloader(
#             0, split="train", batch_size=train_batch_size, shuffle=True
#         )
#         val_loader = (
#             self.dataset.get_phase_dataloader(
#                 0, split="val", batch_size=eval_batch_size, shuffle=False
#             )
#             if self.dataset.validation_enabled
#             else None
#         )

#         backbone_trainable = [
#             p for p in self.model.backbone.parameters() if p.requires_grad
#         ]
#         if not backbone_trainable:
#             raise RuntimeError("Phase-0 backbone has no trainable parameters")

#         backbone_optimizer = torch.optim.AdamW(
#             [{"params": backbone_trainable, "weight_decay": weight_decay}],
#             lr=learning_rate,
#         )
#         # Future classifier rows are intentionally allocated but inactive in
#         # Phase 0. Decoupled AdamW weight decay would move those rows even when
#         # their gradients are masked, so classifier weight decay must be zero.
#         classifier_optimizer = torch.optim.AdamW(
#             list(classifier.parameters()),
#             lr=learning_rate,
#             weight_decay=0.0,
#         )
#         backbone_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
#             backbone_optimizer, T_max=epoch_count, eta_min=0.0
#         )
#         classifier_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
#             classifier_optimizer, T_max=epoch_count, eta_min=0.0
#         )

#         history: list[dict[str, Any]] = []
#         progress = tqdm(
#             range(epoch_count),
#             desc="Phase 0 base representation",
#             unit="epoch",
#             dynamic_ncols=True,
#             file=sys.stdout,
#         )

#         for epoch in progress:
#             # Refresh Whole-CDS in the current Q coordinate before every epoch.
#             # Its intra structure, inter-class risk and native spectral structure
#             # directly shape the risk-aware contrastive separation objective.
#             self.model.eval()
#             epoch_structure = self._working_structure(
#                 train_eval_loader, class_ids=class_ids, physical=physical_structure
#             )
#             epoch_cds = self.model.structure_bank.build_cds(epoch_structure)
#             self.model.train()
#             epoch_lr = float(backbone_optimizer.param_groups[0]["lr"])
#             epoch_classifier_lr = float(classifier_optimizer.param_groups[0]["lr"])
#             classification_sum = 0.0
#             contrastive_sum = 0.0
#             contrastive_positive_sum = 0.0
#             contrastive_negative_sum = 0.0
#             contrastive_gap_sum = 0.0
#             contrastive_anchor_fraction_sum = 0.0
#             total_sum = 0.0
#             correct = 0
#             sample_count = 0
#             per_class_correct = {class_id: 0 for class_id in class_ids}
#             per_class_total = {class_id: 0 for class_id in class_ids}

#             for batch in train_loader:
#                 patch, spectrum, labels = self.unpack_batch(batch)
#                 backbone_optimizer.zero_grad(set_to_none=True)
#                 classifier_optimizer.zero_grad(set_to_none=True)

#                 q = self.model.extract_features(
#                     patch, center_spectrum=spectrum, return_aux=False
#                 ).ordered_features
#                 output = classifier(
#                     q, class_ids=class_ids_tensor, trainable_class_ids=class_ids_tensor
#                 )
#                 targets_local = (
#                     labels.unsqueeze(1)
#                     .eq(class_ids_tensor.unsqueeze(0))
#                     .to(torch.long)
#                     .argmax(dim=1)
#                 )
#                 sample_weights = class_weights.index_select(0, targets_local)
#                 classification = classification_objective(
#                     logits=output.logits,
#                     class_ids=output.class_ids,
#                     labels_global=labels,
#                     sample_weights=sample_weights,
#                 )
#                 physical_evidence = self.model.physical_spectral_evidence(
#                     spectrum
#                 ).detach()
#                 contrastive = whole_cds_contrastive_loss(
#                     current_features=q,
#                     labels_global=labels,
#                     physical_spectral_evidence=physical_evidence,
#                     seen_cds=epoch_cds,
#                     historical_class_count=0,
#                     logit_scale=classifier.logit_scale,
#                     sample_weights=sample_weights,
#                 )
#                 # With no historical classes, the unified continual separation
#                 # objective reduces to ordinary supervised contrastive learning.
#                 # W is trained directly from real base samples by deployed cosine CE.
#                 loss = classification.loss + contrastive_weight * contrastive.loss
#                 if not bool(torch.isfinite(loss)):
#                     raise RuntimeError(
#                         "Phase-0 CE + Whole-CDS contrastive objective became NaN or Inf"
#                     )
#                 loss.backward()

#                 for parameter in backbone_trainable:
#                     if parameter.grad is not None and not bool(
#                         torch.isfinite(parameter.grad).all()
#                     ):
#                         raise RuntimeError(
#                             "Phase-0 backbone gradient contains NaN or Inf"
#                         )
#                 for parameter in classifier.parameters():
#                     if parameter.grad is not None and not bool(
#                         torch.isfinite(parameter.grad).all()
#                     ):
#                         raise RuntimeError(
#                             "Phase-0 classifier gradient contains NaN or Inf"
#                         )
#                 if gradient_clip > 0:
#                     torch.nn.utils.clip_grad_norm_(
#                         backbone_trainable, gradient_clip, error_if_nonfinite=True
#                     )

#                 backbone_optimizer.step()
#                 classifier_optimizer.step()

#                 n = int(labels.numel())
#                 classification_sum += float(classification.loss.detach().item()) * n
#                 contrastive_sum += float(contrastive.loss.detach().item()) * n
#                 contrastive_positive_sum += (
#                     float(contrastive.positive_similarity_mean.detach().item()) * n
#                 )
#                 contrastive_negative_sum += (
#                     float(contrastive.negative_similarity_mean.detach().item()) * n
#                 )
#                 contrastive_gap_sum += (
#                     float(contrastive.similarity_gap.detach().item()) * n
#                 )
#                 contrastive_anchor_fraction_sum += (
#                     float(contrastive.valid_anchor_fraction.detach().item()) * n
#                 )
#                 total_sum += float(loss.detach().item()) * n
#                 prediction = output.prediction.detach()
#                 correct += int(prediction.eq(labels).sum().item())
#                 sample_count += n
#                 for class_id in class_ids:
#                     mask = labels.eq(class_id)
#                     count = int(mask.sum().item())
#                     per_class_total[class_id] += count
#                     if count:
#                         per_class_correct[class_id] += int(
#                             prediction[mask].eq(labels[mask]).sum().item()
#                         )

#             if sample_count != expected_samples:
#                 raise RuntimeError(
#                     "Phase-0 loader did not traverse the complete TRAIN split"
#                 )
#             backbone_scheduler.step()
#             classifier_scheduler.step()

#             self.model.eval()
#             refreshed = self._working_structure(
#                 train_eval_loader, class_ids=class_ids, physical=physical_structure
#             )
#             train_metrics = self.evaluate_loader(
#                 train_eval_loader, class_ids=class_ids, target_class_ids=class_ids
#             )
#             validation_metrics = (
#                 None
#                 if val_loader is None
#                 else self.evaluate_loader(
#                     val_loader, class_ids=class_ids, target_class_ids=class_ids
#                 )
#             )

#             online_per_class = {
#                 cid: per_class_correct[cid] / float(per_class_total[cid])
#                 for cid in class_ids
#             }
#             train_worst_id, train_worst_acc = _worst_class_accuracy(train_metrics)
#             if validation_metrics is None:
#                 val_worst_id, val_worst_acc = None, None
#             else:
#                 val_worst_id, val_worst_acc = _worst_class_accuracy(validation_metrics)

#             record = {
#                 "epoch": epoch + 1,
#                 "learning_rate": epoch_lr,
#                 "classifier_learning_rate": epoch_classifier_lr,
#                 "classification_loss": classification_sum / sample_count,
#                 "contrastive_loss": contrastive_sum / sample_count,
#                 "contrastive_positive_similarity_mean": contrastive_positive_sum
#                 / sample_count,
#                 "contrastive_negative_similarity_mean": contrastive_negative_sum
#                 / sample_count,
#                 "contrastive_similarity_gap": contrastive_gap_sum / sample_count,
#                 "contrastive_valid_anchor_fraction": contrastive_anchor_fraction_sum
#                 / sample_count,
#                 "total_loss": total_sum / sample_count,
#                 "logit_scale": float(classifier.logit_scale.detach().item()),
#                 "online_cosine_oa": correct / float(sample_count),
#                 "online_cosine_ba": sum(online_per_class.values())
#                 / float(len(online_per_class)),
#                 "online_cosine_min_class_accuracy": min(online_per_class.values()),
#                 "train_oa": float(train_metrics["overall_accuracy"]),
#                 "train_ba": float(train_metrics["balanced_accuracy"]),
#                 "train_min_class_accuracy": float(
#                     train_metrics["minimum_class_accuracy"]
#                 ),
#                 "train_logit_margin": float(train_metrics["mean_logit_margin"]),
#                 "train_worst_class_id": int(train_worst_id),
#                 "train_worst_class_accuracy": float(train_worst_acc),
#                 "validation_oa": (
#                     None
#                     if validation_metrics is None
#                     else float(validation_metrics["overall_accuracy"])
#                 ),
#                 "validation_ba": (
#                     None
#                     if validation_metrics is None
#                     else float(validation_metrics["balanced_accuracy"])
#                 ),
#                 "validation_min_class_accuracy": (
#                     None
#                     if validation_metrics is None
#                     else float(validation_metrics["minimum_class_accuracy"])
#                 ),
#                 "validation_logit_margin": (
#                     None
#                     if validation_metrics is None
#                     else float(validation_metrics["mean_logit_margin"])
#                 ),
#                 "validation_worst_class_id": val_worst_id,
#                 "validation_worst_class_accuracy": val_worst_acc,
#             }
#             history.append(record)

#             message = (
#                 f"[Epoch {epoch + 1:03d}/{epoch_count:03d}] "
#                 f"CE={record['classification_loss']:.4f} "
#                 f"Con={record['contrastive_loss']:.4f} "
#                 f"ConGap={record['contrastive_similarity_gap']:.4f} "
#                 f"Scale={record['logit_scale']:.3f} "
#                 f"LR={record['learning_rate']:.2e} HeadLR={record['classifier_learning_rate']:.2e} | "
#                 f"Online[OA={100*record['online_cosine_oa']:.2f}%,"
#                 f"BA={100*record['online_cosine_ba']:.2f}%,"
#                 f"Min={100*record['online_cosine_min_class_accuracy']:.2f}%] | "
#                 f"Train[OA={100*record['train_oa']:.2f}%,BA={100*record['train_ba']:.2f}%,"
#                 f"Min={100*record['train_min_class_accuracy']:.2f}%]"
#             )
#             if validation_metrics is not None:
#                 message += (
#                     f" | Val[OA={100*record['validation_oa']:.2f}%,"
#                     f"BA={100*record['validation_ba']:.2f}%,"
#                     f"Min={100*record['validation_min_class_accuracy']:.2f}%]"
#                 )
#             tqdm.write(message, file=sys.stdout)

#         progress.close()

#         self.model.eval()
#         final_structure = self._working_structure(
#             train_eval_loader, class_ids=class_ids, physical=physical_structure
#         )

#         self.model.commit_class_structure(final_structure)
#         self.model.validate_finalized_state()

#         train_metrics = self.evaluate_loader(
#             train_eval_loader, class_ids=class_ids, target_class_ids=class_ids
#         )
#         validation_metrics = (
#             None
#             if val_loader is None
#             else self.evaluate_loader(
#                 val_loader, class_ids=class_ids, target_class_ids=class_ids
#             )
#         )
#         self.dataset.finalize_phase(0)
#         reporting_loader = self.dataset.get_reporting_dataloader(
#             0, batch_size=eval_batch_size
#         )
#         test_metrics = self.evaluate_loader(
#             reporting_loader, class_ids=class_ids, target_class_ids=class_ids
#         )

#         return {
#             "phase": 0,
#             "class_ids": list(class_ids),
#             "history": history,
#             "final_epoch": epoch_count,
#             "train": train_metrics,
#             "validation": validation_metrics,
#             "test": test_metrics,
#             "structure_committed": True,
#             "phase_finalized": True,
#             "contrastive_weight": contrastive_weight,
#         }


# __all__ = ["BasePhaseTrainer"]


# from __future__ import annotations
# import math
# import sys
# from numbers import Integral, Real
# from typing import Any, Dict, Mapping

# import torch
# from tqdm import tqdm

# from losses.loss import classification_objective, whole_cds_contrastive_loss
# from models.classifier import CosineClassificationHead
# from models.geometry_bank import ClassStructureBatch, PhysicalSpectralStructureBatch


# def _positive_int(value: object, *, name: str) -> int:
#     if isinstance(value, bool) or not isinstance(value, (Integral, Real)):
#         raise ValueError(f"{name} must be a positive integer")
#     number = float(value)
#     if not math.isfinite(number) or not number.is_integer() or number <= 0:
#         raise ValueError(f"{name} must be a positive integer")
#     return int(number)


# def _finite_positive(name: str, value: object) -> float:
#     result = float(value)
#     if not math.isfinite(result) or result <= 0:
#         raise ValueError(f"{name} must be finite and positive")
#     return result


# def _finite_nonnegative(name: str, value: object) -> float:
#     result = float(value)
#     if not math.isfinite(result) or result < 0:
#         raise ValueError(f"{name} must be finite and non-negative")
#     return result


# def _worst_class_accuracy(metrics: Mapping[str, Any]) -> tuple[int, float]:
#     values = metrics.get("per_class_accuracy")
#     if not isinstance(values, Mapping) or not values:
#         raise RuntimeError("evaluation metrics lack per_class_accuracy")
#     parsed = [(int(key), float(value)) for key, value in values.items()]
#     return min(parsed, key=lambda item: (item[1], item[0]))


# class BasePhaseTrainer:
#     BASE_RESULT_FIELDS = frozenset(
#         {
#             "phase",
#             "class_ids",
#             "history",
#             "final_epoch",
#             "train",
#             "validation",
#             "test",
#             "structure_committed",
#             "phase_finalized",
#         }
#     )

#     def _require_classifier(self) -> CosineClassificationHead:
#         classifier = getattr(self.model, "classifier", None)
#         if not isinstance(classifier, CosineClassificationHead) or classifier.parameter_count <= 0:
#             raise RuntimeError("model.classifier must be the CosineClassificationHead")
#         return classifier

#     @torch.no_grad()
#     def _prepare_train_state(
#         self,
#         loader: Any,
#         *,
#         class_ids: tuple[int, ...],
#     ) -> tuple[torch.Tensor, int]:
#         counts = {class_id: 0 for class_id in class_ids}
#         spectra: list[torch.Tensor] = []
#         total = 0
#         for batch in loader:
#             _, center_spectrum, labels = self.unpack_batch(batch)
#             observed = set(int(value) for value in labels.unique().detach().cpu().tolist())
#             outside = observed - set(class_ids)
#             if outside:
#                 raise RuntimeError(f"Phase-0 TRAIN loader contains non-base classes {sorted(outside)}")
#             if not self.model.spectral_normalization_fitted:
#                 spectra.append(center_spectrum.detach())
#             for class_id in class_ids:
#                 count = int(labels.eq(class_id).sum().item())
#                 counts[class_id] += count
#                 total += count

#         if total <= 0 or any(counts[class_id] <= 0 for class_id in class_ids):
#             raise RuntimeError("Phase-0 TRAIN split is empty or misses a base class")
#         if not self.model.spectral_normalization_fitted:
#             if not spectra:
#                 raise RuntimeError("Phase-0 spectra are unavailable for normalization")
#             self.model.fit_spectral_normalization(torch.cat(spectra, dim=0), overwrite=False)

#         class_count = len(class_ids)
#         weights = torch.tensor(
#             [total / float(class_count * counts[class_id]) for class_id in class_ids],
#             device=self.device,
#             dtype=self.model.dtype,
#         )
#         return weights, total

#     @torch.no_grad()
#     def _fixed_physical_structure(
#         self,
#         loader: Any,
#         *,
#         class_ids: tuple[int, ...],
#     ) -> PhysicalSpectralStructureBatch:
#         spectra: list[torch.Tensor] = []
#         labels_rows: list[torch.Tensor] = []
#         for batch in loader:
#             _, spectrum, labels = self.unpack_batch(batch)
#             spectra.append(spectrum.detach())
#             labels_rows.append(labels.detach())
#         if not spectra:
#             raise RuntimeError("cannot estimate physical CDS from an empty Phase-0 loader")
#         return self.model.estimate_physical_class_structure(
#             torch.cat(spectra, dim=0),
#             torch.cat(labels_rows, dim=0),
#             class_ids=class_ids,
#         )

#     @torch.no_grad()
#     def _working_structure(
#         self,
#         loader: Any,
#         *,
#         class_ids: tuple[int, ...],
#         physical: PhysicalSpectralStructureBatch,
#     ) -> ClassStructureBatch:
#         rows = self.collect_ordered_features(loader)
#         observed = tuple(sorted(int(value) for value in rows["labels"].unique().detach().cpu().tolist()))
#         if observed != tuple(sorted(class_ids)):
#             raise RuntimeError("Phase-0 TRAIN rows do not match the base class set")
#         return self.model.estimate_class_structure(
#             rows["ordered_features"],
#             None,
#             rows["labels"],
#             class_ids=class_ids,
#             physical_structure=physical,
#         )

#     def train_base_phase(
#         self,
#         phase: int,
#         epochs: int,
#         batch_size: int = 64,
#         lr: float = 1e-4,
#     ) -> Dict[str, Any]:
#         if isinstance(phase, bool) or int(phase) != 0:
#             raise ValueError("BasePhaseTrainer handles Phase 0 only")

#         epoch_count = _positive_int(epochs, name="epochs")
#         train_batch_size = _positive_int(batch_size, name="batch_size")
#         learning_rate = _finite_positive("lr", lr)
#         eval_batch_size = _positive_int(self.cfg("eval_batch_size", 256), name="eval_batch_size")
#         weight_decay = _finite_nonnegative("weight_decay", self.cfg("weight_decay", 1e-4))
#         gradient_clip = _finite_nonnegative("gradient_clip", self.cfg("gradient_clip", 0.0))
#         contrastive_weight = _finite_nonnegative(
#             "contrastive_weight", self.cfg("contrastive_weight", 1.0)
#         )

#         self.model.validate_model_state()
#         self.model.set_base_trainability()
#         classifier = self._require_classifier()
#         if len(self.model.structure_bank) != 0:
#             raise RuntimeError("Phase 0 requires an empty CDS bank")
#         if getattr(self.dataset, "current_phase", None) is not None:
#             raise RuntimeError("dataset already has an active phase")

#         self.dataset.start_phase(0)
#         class_ids = tuple(
#             self._validated_class_ids(self.dataset.get_new_classes(0), name="phase_0_classes")
#         )
#         if len(class_ids) < 2:
#             raise RuntimeError("Phase 0 requires at least two classes")
#         class_ids_tensor = torch.tensor(class_ids, device=self.device, dtype=torch.long)

#         train_eval_loader = self.dataset.get_phase_dataloader(
#             0, split="train", batch_size=eval_batch_size, shuffle=False
#         )
#         class_weights, expected_samples = self._prepare_train_state(
#             train_eval_loader, class_ids=class_ids
#         )
#         physical_structure = self._fixed_physical_structure(
#             train_eval_loader, class_ids=class_ids
#         )

#         train_loader = self.dataset.get_phase_dataloader(
#             0, split="train", batch_size=train_batch_size, shuffle=True
#         )
#         val_loader = (
#             self.dataset.get_phase_dataloader(
#                 0, split="val", batch_size=eval_batch_size, shuffle=False
#             )
#             if self.dataset.validation_enabled
#             else None
#         )

#         backbone_trainable = [p for p in self.model.backbone.parameters() if p.requires_grad]
#         if not backbone_trainable:
#             raise RuntimeError("Phase-0 backbone has no trainable parameters")

#         backbone_optimizer = torch.optim.AdamW(
#             [{"params": backbone_trainable, "weight_decay": weight_decay}],
#             lr=learning_rate,
#         )
#         # Future classifier rows are intentionally allocated but inactive in
#         # Phase 0. Decoupled AdamW weight decay would move those rows even when
#         # their gradients are masked, so classifier weight decay must be zero.
#         classifier_optimizer = torch.optim.AdamW(
#             list(classifier.parameters()),
#             lr=learning_rate,
#             weight_decay=0.0,
#         )
#         backbone_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
#             backbone_optimizer, T_max=epoch_count, eta_min=0.0
#         )
#         classifier_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
#             classifier_optimizer, T_max=epoch_count, eta_min=0.0
#         )

#         history: list[dict[str, Any]] = []
#         progress = tqdm(
#             range(epoch_count),
#             desc="Phase 0 base representation",
#             unit="epoch",
#             dynamic_ncols=True,
#             file=sys.stdout,
#         )

#         for epoch in progress:
#             # Refresh Whole-CDS in the current Q coordinate before every epoch.
#             # Its intra structure, inter-class risk and native spectral structure
#             # directly shape the risk-aware contrastive separation objective.
#             self.model.eval()
#             epoch_structure = self._working_structure(
#                 train_eval_loader, class_ids=class_ids, physical=physical_structure
#             )
#             epoch_cds = self.model.structure_bank.build_cds(epoch_structure)
#             self.model.train()
#             epoch_lr = float(backbone_optimizer.param_groups[0]["lr"])
#             epoch_classifier_lr = float(classifier_optimizer.param_groups[0]["lr"])
#             classification_sum = 0.0
#             contrastive_sum = 0.0
#             contrastive_positive_sum = 0.0
#             contrastive_negative_sum = 0.0
#             contrastive_gap_sum = 0.0
#             contrastive_anchor_fraction_sum = 0.0
#             total_sum = 0.0
#             correct = 0
#             sample_count = 0
#             per_class_correct = {class_id: 0 for class_id in class_ids}
#             per_class_total = {class_id: 0 for class_id in class_ids}

#             for batch in train_loader:
#                 patch, spectrum, labels = self.unpack_batch(batch)
#                 backbone_optimizer.zero_grad(set_to_none=True)
#                 classifier_optimizer.zero_grad(set_to_none=True)

#                 q = self.model.extract_features(
#                     patch, center_spectrum=spectrum, return_aux=False
#                 ).ordered_features
#                 output = classifier(q, class_ids=class_ids_tensor, trainable_class_ids=class_ids_tensor)
#                 targets_local = labels.unsqueeze(1).eq(class_ids_tensor.unsqueeze(0)).to(torch.long).argmax(dim=1)
#                 sample_weights = class_weights.index_select(0, targets_local)
#                 classification = classification_objective(
#                     logits=output.logits,
#                     class_ids=output.class_ids,
#                     labels_global=labels,
#                     sample_weights=sample_weights,
#                 )
#                 physical_evidence = self.model.physical_spectral_evidence(spectrum).detach()
#                 contrastive = whole_cds_contrastive_loss(
#                     current_features=q,
#                     labels_global=labels,
#                     physical_spectral_evidence=physical_evidence,
#                     seen_cds=epoch_cds,
#                     historical_class_count=0,
#                     logit_scale=classifier.logit_scale,
#                     sample_weights=sample_weights,
#                 )
#                 # With no historical classes, the unified continual separation
#                 # objective reduces to ordinary supervised contrastive learning.
#                 # W is trained directly from real base samples by deployed cosine CE.
#                 loss = classification.loss + contrastive_weight * contrastive.loss
#                 if not bool(torch.isfinite(loss)):
#                     raise RuntimeError("Phase-0 CE + Whole-CDS contrastive objective became NaN or Inf")
#                 loss.backward()

#                 for parameter in backbone_trainable:
#                     if parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all()):
#                         raise RuntimeError("Phase-0 backbone gradient contains NaN or Inf")
#                 for parameter in classifier.parameters():
#                     if parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all()):
#                         raise RuntimeError("Phase-0 classifier gradient contains NaN or Inf")
#                 if gradient_clip > 0:
#                     torch.nn.utils.clip_grad_norm_(backbone_trainable, gradient_clip, error_if_nonfinite=True)

#                 backbone_optimizer.step()
#                 classifier_optimizer.step()

#                 n = int(labels.numel())
#                 classification_sum += float(classification.loss.detach().item()) * n
#                 contrastive_sum += float(contrastive.loss.detach().item()) * n
#                 contrastive_positive_sum += float(contrastive.positive_similarity_mean.detach().item()) * n
#                 contrastive_negative_sum += float(contrastive.negative_similarity_mean.detach().item()) * n
#                 contrastive_gap_sum += float(contrastive.similarity_gap.detach().item()) * n
#                 contrastive_anchor_fraction_sum += float(contrastive.valid_anchor_fraction.detach().item()) * n
#                 total_sum += float(loss.detach().item()) * n
#                 prediction = output.prediction.detach()
#                 correct += int(prediction.eq(labels).sum().item())
#                 sample_count += n
#                 for class_id in class_ids:
#                     mask = labels.eq(class_id)
#                     count = int(mask.sum().item())
#                     per_class_total[class_id] += count
#                     if count:
#                         per_class_correct[class_id] += int(prediction[mask].eq(labels[mask]).sum().item())

#             if sample_count != expected_samples:
#                 raise RuntimeError("Phase-0 loader did not traverse the complete TRAIN split")
#             backbone_scheduler.step()
#             classifier_scheduler.step()

#             self.model.eval()
#             refreshed = self._working_structure(
#                 train_eval_loader, class_ids=class_ids, physical=physical_structure
#             )
#             train_metrics = self.evaluate_loader(
#                 train_eval_loader,
#                 class_ids=class_ids,
#                 target_class_ids=class_ids
#             )
#             validation_metrics = (
#                 None
#                 if val_loader is None
#                 else self.evaluate_loader(
#                     val_loader,
#                     class_ids=class_ids,
#                     target_class_ids=class_ids
#                 )
#             )

#             online_per_class = {
#                 cid: per_class_correct[cid] / float(per_class_total[cid]) for cid in class_ids
#             }
#             train_worst_id, train_worst_acc = _worst_class_accuracy(train_metrics)
#             if validation_metrics is None:
#                 val_worst_id, val_worst_acc = None, None
#             else:
#                 val_worst_id, val_worst_acc = _worst_class_accuracy(validation_metrics)

#             record = {
#                 "epoch": epoch + 1,
#                 "learning_rate": epoch_lr,
#                 "classifier_learning_rate": epoch_classifier_lr,
#                 "classification_loss": classification_sum / sample_count,
#                 "contrastive_loss": contrastive_sum / sample_count,
#                 "contrastive_positive_similarity_mean": contrastive_positive_sum / sample_count,
#                 "contrastive_negative_similarity_mean": contrastive_negative_sum / sample_count,
#                 "contrastive_similarity_gap": contrastive_gap_sum / sample_count,
#                 "contrastive_valid_anchor_fraction": contrastive_anchor_fraction_sum / sample_count,
#                 "total_loss": total_sum / sample_count,
#                 "logit_scale": float(classifier.logit_scale.detach().item()),
#                 "online_cosine_oa": correct / float(sample_count),
#                 "online_cosine_ba": sum(online_per_class.values()) / float(len(online_per_class)),
#                 "online_cosine_min_class_accuracy": min(online_per_class.values()),
#                 "train_oa": float(train_metrics["overall_accuracy"]),
#                 "train_ba": float(train_metrics["balanced_accuracy"]),
#                 "train_min_class_accuracy": float(train_metrics["minimum_class_accuracy"]),
#                 "train_logit_margin": float(train_metrics["mean_logit_margin"]),
#                 "train_worst_class_id": int(train_worst_id),
#                 "train_worst_class_accuracy": float(train_worst_acc),
#                 "validation_oa": None if validation_metrics is None else float(validation_metrics["overall_accuracy"]),
#                 "validation_ba": None if validation_metrics is None else float(validation_metrics["balanced_accuracy"]),
#                 "validation_min_class_accuracy": None if validation_metrics is None else float(validation_metrics["minimum_class_accuracy"]),
#                 "validation_logit_margin": None if validation_metrics is None else float(validation_metrics["mean_logit_margin"]),
#                 "validation_worst_class_id": val_worst_id,
#                 "validation_worst_class_accuracy": val_worst_acc,
#             }
#             history.append(record)

#             message = (
#                 f"[Epoch {epoch + 1:03d}/{epoch_count:03d}] "
#                 f"CE={record['classification_loss']:.4f} "
#                 f"Con={record['contrastive_loss']:.4f} "
#                 f"ConGap={record['contrastive_similarity_gap']:.4f} "
#                 f"Scale={record['logit_scale']:.3f} "
#                 f"LR={record['learning_rate']:.2e} HeadLR={record['classifier_learning_rate']:.2e} | "
#                 f"Online[OA={100*record['online_cosine_oa']:.2f}%,"
#                 f"BA={100*record['online_cosine_ba']:.2f}%,"
#                 f"Min={100*record['online_cosine_min_class_accuracy']:.2f}%] | "
#                 f"Train[OA={100*record['train_oa']:.2f}%,BA={100*record['train_ba']:.2f}%,"
#                 f"Min={100*record['train_min_class_accuracy']:.2f}%]"
#             )
#             if validation_metrics is not None:
#                 message += (
#                     f" | Val[OA={100*record['validation_oa']:.2f}%,"
#                     f"BA={100*record['validation_ba']:.2f}%,"
#                     f"Min={100*record['validation_min_class_accuracy']:.2f}%]"
#                 )
#             tqdm.write(message, file=sys.stdout)

#         progress.close()

#         self.model.eval()
#         final_structure = self._working_structure(
#             train_eval_loader, class_ids=class_ids, physical=physical_structure
#         )
#         self.model.commit_class_structure(final_structure)
#         self.model.validate_finalized_state()

#         train_metrics = self.evaluate_loader(
#             train_eval_loader, class_ids=class_ids, target_class_ids=class_ids
#         )
#         validation_metrics = (
#             None
#             if val_loader is None
#             else self.evaluate_loader(val_loader, class_ids=class_ids, target_class_ids=class_ids)
#         )
#         self.dataset.finalize_phase(0)
#         reporting_loader = self.dataset.get_reporting_dataloader(0, batch_size=eval_batch_size)
#         test_metrics = self.evaluate_loader(
#             reporting_loader, class_ids=class_ids, target_class_ids=class_ids
#         )

#         return {
#             "phase": 0,
#             "class_ids": list(class_ids),
#             "history": history,
#             "final_epoch": epoch_count,
#             "train": train_metrics,
#             "validation": validation_metrics,
#             "test": test_metrics,
#             "structure_committed": True,
#             "phase_finalized": True,
#             "contrastive_weight": contrastive_weight,
#         }


# __all__ = ["BasePhaseTrainer"]
