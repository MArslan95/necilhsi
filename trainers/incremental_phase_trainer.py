from __future__ import annotations

"""Incremental optimization for the role-separated NECIL-HSI architecture.

This file intentionally replaces the previous frozen-encoder/replay/pairwise-
boundary implementation. The incremental phase now follows the same deployed
architecture as phase 0:

    HSI -> trainable Mamba backbone -> canonical Q -> angular classifier.

Historical class statistics ``{I_c,S_c,N_c}`` never classify and no historical
sample/support/replay stream is created. They are used only by one dedicated
``ClassStructurePreserver`` that constrains incremental gradients.

For current-phase HSI only:

    L_inc = CE(all-seen angular logits, current labels).

Before the CE representation gradient enters the backbone,
``ClassStructurePreserver`` removes its component in a compact historical
class-discriminative structural subspace.  The basis contains identity-
discriminative modes and the class-specific ``(S_c - mean S)`` responses of
those modes.  The same component constrains old classifier-row updates while
leaving new rows fully plastic. There is no preservation loss, no KD, no teacher, no replay, no old
feature cache, no frozen backbone, no pairwise boundary candidate, and no
separate incremental head.
"""

import math
import sys
from numbers import Integral, Real
from typing import Any, Dict, Mapping, Optional, Sequence

import torch
import torch.nn as nn
from tqdm import tqdm

from losses.loss import cosine_classification_objective
from models.classifier import PersistentCosineClassifier
from models.structure_preservation import ClassStructurePreserver
from utils.eval import old_new_summary

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


def _finite_positive(name: str, value: object) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return result


def _finite_nonnegative(name: str, value: object) -> float:
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return result


class IncrementalPhaseTrainer:
    """Train one strict non-exemplar incremental phase."""

    INCREMENTAL_RESULT_FIELDS = frozenset(
        {
            "phase",
            "old_class_ids",
            "new_class_ids",
            "seen_class_ids",
            "history",
            "final_epoch",
            "old_before",
            "cumulative_test",
            "old_new",
            "forgetting",
            "preservation",
            "geometry_committed",
            "phase_finalized",
        }
    )

    def _require_classifier(self) -> PersistentCosineClassifier:
        classifier = getattr(self.model, "classifier", None)
        if not isinstance(classifier, PersistentCosineClassifier):
            raise RuntimeError("model.classifier must be PersistentCosineClassifier")
        return classifier

    def _phase_classes(
        self,
        phase: int,
    ) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
        old_ids = tuple(
            self._validated_class_ids(
                self.dataset.get_old_classes(phase),
                name=f"phase_{phase}_old_classes",
            )
        )
        new_ids = tuple(
            self._validated_class_ids(
                self.dataset.get_new_classes(phase),
                name=f"phase_{phase}_new_classes",
            )
        )
        seen_ids = tuple(
            self._validated_class_ids(
                self.dataset.get_seen_classes(phase),
                name=f"phase_{phase}_seen_classes",
            )
        )
        if not old_ids or not new_ids:
            raise RuntimeError("incremental phase requires old and new classes")
        if set(old_ids).intersection(new_ids):
            raise RuntimeError("old and new class IDs overlap")
        if seen_ids != old_ids + new_ids:
            raise RuntimeError("seen classes must equal old classes followed by new classes")

        if tuple(self.model.committed_class_ids) != old_ids:
            raise RuntimeError(
                "retained class statistics do not match the historical phase: "
                f"geometry={self.model.committed_class_ids}, old={old_ids}"
            )
        if tuple(self.model.classifier_class_ids) != old_ids:
            raise RuntimeError(
                "classifier must contain exactly the historical rows before phase expansion"
            )
        return old_ids, new_ids, seen_ids

    @staticmethod
    def _parameters_finite(parameters: Sequence[nn.Parameter]) -> bool:
        return all(bool(torch.isfinite(parameter).all()) for parameter in parameters)

    @staticmethod
    def _spatial_symmetry_augment(
        patch: Tensor,
        *,
        generator: torch.Generator,
    ) -> Tensor:
        """Apply one D4 symmetry per current TRAIN sample, spatial axes only."""
        if patch.ndim != 4 or patch.size(-1) != patch.size(-2):
            raise RuntimeError("D4 augmentation requires square [N,C,H,W] patches")
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
            raise RuntimeError("D4 augmentation produced invalid patches")
        return result

    def _validation_loader(
        self,
        phase: int,
        *,
        batch_size: int,
    ) -> Optional[Any]:
        enabled = getattr(self.dataset, "validation_enabled", None)
        if not isinstance(enabled, bool):
            raise RuntimeError("dataset.validation_enabled must be bool")
        if not enabled:
            return None
        return self.dataset.get_phase_dataloader(
            phase,
            split="val",
            batch_size=batch_size,
            shuffle=False,
        )

    @torch.no_grad()
    def _collect_labels_only(self, loader: Any) -> Tensor:
        """Collect current-phase labels without materializing Q."""
        rows: list[Tensor] = []
        for batch in loader:
            _, _, labels = self.unpack_batch(batch)
            rows.append(labels.detach().cpu())
        if not rows:
            raise RuntimeError("cannot collect labels from an empty loader")
        return torch.cat(rows, dim=0).flatten()

    def train_incremental_phase(
        self,
        *,
        phase: int,
        epochs: int,
        batch_size: int,
        lr: float,
    ) -> Dict[str, Any]:
        phase_id = _positive_int(phase, name="phase")
        epoch_count = _positive_int(epochs, name="epochs")
        train_batch_size = _positive_int(batch_size, name="batch_size")
        learning_rate = _finite_positive("lr", lr)

        self.model.validate_model_state()
        if getattr(self.dataset, "current_phase", None) is not None:
            raise RuntimeError("dataset already has an active phase")

        # Reporting-only pre-update old performance. It is never used by the
        # optimizer, scheduler, epoch selection, or preservation construction.
        previous_phase = phase_id - 1
        previous_seen = tuple(
            self._validated_class_ids(
                self.dataset.get_seen_classes(previous_phase),
                name="previous_seen_classes",
            )
        )
        if tuple(self.model.classifier_class_ids) != previous_seen:
            raise RuntimeError("checkpoint classifier does not match previous finalized phase")
        old_before_loader = self.dataset.get_reporting_dataloader(
            previous_phase,
            batch_size=_positive_int(
                self.cfg("eval_batch_size", 256),
                name="eval_batch_size",
            ),
        )
        old_before = self.evaluate_loader(
            old_before_loader,
            class_ids=previous_seen,
            target_class_ids=previous_seen,
        )

        self.dataset.start_phase(phase_id)
        old_ids, new_ids, seen_ids = self._phase_classes(phase_id)

        # Build the preservation constraint from class statistics only. No old
        # sample or per-sample feature is materialized.
        preserver = ClassStructurePreserver(
            geometry_bank=self.model.geometry_bank,
            old_class_ids=old_ids,
        )

        # Classifier expansion replaces its Parameter object and therefore must
        # happen before creating the optimizer.
        self.model.add_classifier_classes(new_ids)
        classifier = self._require_classifier()
        if tuple(classifier.seen_class_ids) != seen_ids:
            raise RuntimeError("classifier expansion produced incorrect seen-class order")

        eval_batch_size = _positive_int(
            self.cfg("eval_batch_size", 256),
            name="eval_batch_size",
        )
        train_loader = self.dataset.get_phase_dataloader(
            phase_id,
            split="train",
            batch_size=train_batch_size,
            shuffle=True,
        )
        train_eval_loader = self.dataset.get_phase_dataloader(
            phase_id,
            split="train",
            batch_size=eval_batch_size,
            shuffle=False,
        )
        validation_loader = self._validation_loader(
            phase_id,
            batch_size=eval_batch_size,
        )

        # Current TRAIN loader must contain only the phase's new classes.
        # Validate labels only; Q is collected once, after the final epoch, when
        # the new classes' {I_c,S_c,N_c} statistics are committed.
        current_labels = self._collect_labels_only(train_eval_loader)
        observed = tuple(
            sorted(int(v) for v in current_labels.unique().tolist())
        )
        if observed != tuple(sorted(new_ids)):
            raise RuntimeError(
                f"phase-{phase_id} TRAIN labels {observed} do not match new classes "
                f"{tuple(sorted(new_ids))}"
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
            raise RuntimeError("incremental learning requires trainable backbone and classifier")

        # Incremental preservation constrains the current-task CE gradient.
        # Decoupled weight decay would add an independent backbone displacement
        # outside that constraint, so the incremental backbone uses zero decay.
        optimizer = torch.optim.AdamW(
            [
                {
                    "params": backbone_parameters,
                    "weight_decay": 0.0,
                },
                {
                    # Classifier rows are direction-only. Their raw norm has no
                    # decision meaning, so keep the same zero-decay policy as base.
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
        optimized_parameters = backbone_parameters + classifier_parameters

        augmentation_generator = torch.Generator(device="cpu")
        # Make every phase deterministic but distinct from phase 0.
        augmentation_generator.manual_seed(
            int(self.cfg("seed", 0)) + phase_id
        )

        history: list[Dict[str, Any]] = []
        progress = tqdm(
            range(epoch_count),
            desc=f"Incremental phase {phase_id}",
            unit="epoch",
            dynamic_ncols=True,
            file=sys.stdout,
        )

        for epoch in progress:
            self.model.train()
            epoch_lr = float(optimizer.param_groups[0]["lr"])
            if not math.isfinite(epoch_lr) or epoch_lr < 0.0:
                raise RuntimeError("optimizer learning rate is invalid")

            loss_sum = 0.0
            sample_count = 0
            correct = 0
            class_correct = {class_id: 0 for class_id in new_ids}
            class_total = {class_id: 0 for class_id in new_ids}
            maximum_classifier_constraint_residual = 0.0

            for batch in train_loader:
                patch, spectrum, labels = self.unpack_batch(batch)
                observed_batch = set(int(v) for v in labels.detach().cpu().tolist())
                if not observed_batch.issubset(set(new_ids)):
                    raise RuntimeError("incremental TRAIN batch contains a historical/future class")

                patch = self._spatial_symmetry_augment(
                    patch,
                    generator=augmentation_generator,
                )
                optimizer.zero_grad(set_to_none=True)

                output = self.model.extract_features(
                    patch,
                    center_spectrum=spectrum,
                    return_aux=False,
                )
                q = output.ordered_features
                if not q.requires_grad:
                    raise RuntimeError("incremental canonical Q must require gradients")

                # One preservation component, one responsibility: remove the
                # direct current-task Q-gradient component lying in the retained
                # historical identity/structure subspace before backpropagation
                # continues into the shared backbone.
                q.register_hook(preserver.project_representation_gradient)

                classification = self.model.classify_features(
                    q,
                    class_ids=seen_ids,
                )
                objective = cosine_classification_objective(
                    output=classification,
                    labels_global=labels,
                )
                objective.loss.backward()

                # Old classifier rows remain trainable, but their gradient is
                # restricted so stored old-identity responses are not directly
                # altered. New rows are never projected.
                preserver.project_old_classifier_gradient_(classifier)

                if any(
                    parameter.grad is not None
                    and not bool(torch.isfinite(parameter.grad).all())
                    for parameter in optimized_parameters
                ):
                    raise RuntimeError("incremental optimization produced NaN/Inf gradients")

                if gradient_clip > 0.0:
                    torch.nn.utils.clip_grad_norm_(
                        optimized_parameters,
                        gradient_clip,
                        error_if_nonfinite=True,
                    )

                classifier_before = classifier.weight.detach().clone()
                optimizer.step()
                residual = preserver.project_old_classifier_update_(
                    classifier,
                    classifier_before,
                )
                maximum_classifier_constraint_residual = max(
                    maximum_classifier_constraint_residual,
                    residual,
                )

                if not self._parameters_finite(optimized_parameters):
                    raise RuntimeError("incremental optimization produced NaN/Inf parameters")
                self.model.validate_model_state()

                n = int(labels.numel())
                loss_sum += float(objective.loss.detach().item()) * n
                sample_count += n
                prediction = classification.prediction.detach()
                correct += int(prediction.eq(labels).sum().item())
                for class_id in new_ids:
                    mask = labels.eq(class_id)
                    count = int(mask.sum().item())
                    if count:
                        class_total[class_id] += count
                        class_correct[class_id] += int(
                            prediction[mask].eq(labels[mask]).sum().item()
                        )

            if sample_count <= 0:
                raise RuntimeError("incremental TRAIN loader yielded no samples")
            missing_classes = [
                class_id for class_id in new_ids if class_total[class_id] == 0
            ]
            if missing_classes:
                raise RuntimeError(
                    f"incremental epoch did not observe new classes {missing_classes}"
                )

            train_loss = loss_sum / float(sample_count)
            train_oa = correct / float(sample_count)
            train_ba = sum(
                class_correct[class_id] / float(class_total[class_id])
                for class_id in new_ids
            ) / float(len(new_ids))

            validation_metrics = (
                None
                if validation_loader is None
                else self.evaluate_loader(
                    validation_loader,
                    class_ids=seen_ids,
                    target_class_ids=new_ids,
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
                "new_train_oa": float(train_oa),
                "new_train_ba": float(train_ba),
                "validation_new_ba": (
                    None
                    if validation_metrics is None
                    else float(validation_metrics["balanced_accuracy"])
                ),
                "classifier_constraint_residual": float(
                    maximum_classifier_constraint_residual
                ),
            }
            history.append(record)

            message = (
                f"[Phase {phase_id} | Epoch {epoch + 1:03d}/{epoch_count:03d}] "
                f"LR={epoch_lr:.3e} | CE={train_loss:.4f} | "
                f"newTrain_BA={100.0 * train_ba:.2f}%"
            )
            if validation_metrics is not None:
                message += (
                    f" | newVal_BA="
                    f"{100.0 * validation_metrics['balanced_accuracy']:.2f}%"
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
            raise RuntimeError("incremental training did not complete all epochs")

        # Fixed-epoch contract: retain the final requested state. Estimate and
        # append geometry only for classes whose real TRAIN data are available
        # in this phase. Historical {I,S,N} are never re-fit from new samples.
        self.model.eval()
        final_new_train = self.collect_ordered_features(train_eval_loader)
        self.model.add_classes(
            final_new_train["ordered_features"],
            final_new_train["labels"],
            class_ids=new_ids,
        )
        if tuple(self.model.committed_class_ids) != seen_ids:
            raise RuntimeError("incremental geometry commit produced incorrect seen classes")
        if tuple(self.model.classifier_class_ids) != seen_ids:
            raise RuntimeError("classifier class IDs changed unexpectedly")

        self.dataset.finalize_phase(phase_id)
        reporting_loader = self.dataset.get_reporting_dataloader(
            phase_id,
            batch_size=eval_batch_size,
        )
        cumulative_test = self.evaluate_loader(
            reporting_loader,
            class_ids=seen_ids,
            target_class_ids=seen_ids,
        )
        groups = old_new_summary(
            cumulative_test,
            old_class_ids=old_ids,
            new_class_ids=new_ids,
        )
        old_after_ba = float(groups["old"]["balanced_accuracy"])
        old_before_ba = float(old_before["balanced_accuracy"])
        forgetting = old_before_ba - old_after_ba

        summary = preserver.summary
        preservation = {
            "old_class_ids": list(summary.old_class_ids),
            "feature_dim": int(summary.feature_dim),
            "identity_discriminative_rank": int(
                summary.identity_discriminative_rank
            ),
            "representation_rank": int(summary.representation_rank),
            "classifier_identity_rank": int(summary.classifier_identity_rank),
            "generalized_eigenvalues": list(summary.generalized_eigenvalues),
        }

        return {
            "phase": phase_id,
            "old_class_ids": list(old_ids),
            "new_class_ids": list(new_ids),
            "seen_class_ids": list(seen_ids),
            "history": history,
            "final_epoch": epoch_count,
            "old_before": old_before,
            "cumulative_test": cumulative_test,
            "old_new": groups,
            "forgetting": {
                "old_balanced_accuracy_before": old_before_ba,
                "old_balanced_accuracy_after": old_after_ba,
                "old_balanced_accuracy_drop": float(forgetting),
            },
            "preservation": preservation,
            "geometry_committed": True,
            "phase_finalized": True,
        }


__all__ = ["IncrementalPhaseTrainer"]
