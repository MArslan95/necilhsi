from __future__ import annotations

"""Stable-Q structural classifier adaptation for SCOPE-HSI.

Incremental phase t keeps the canonical HSI representation Q fixed.  The single
persistent angular classifier is expanded with current-class rows initialized
from current class identities and then ALL seen classifier rows are optimized
from a phase-balanced classification risk:

    current classes    -> real current TRAIN Q and labels,
    historical classes -> retained {I_c,S_c}.

For each historical class c and every seen rival j != c, the retained class
identity and full feature-resolved structure define

    mu_cj       = <I_c, U_c-U_j>_F,
    sigma_bar_cj = structural decision-gap spread from S_c[K,K,D],
    r_cj        = mu_cj / sigma_bar_cj   (when spread is non-zero).

At phase start, old-old scores are captured from the previous finalized
classifier. During optimization, old-old relations are not allowed to become
less safe than that reference (failed reference relations are repaired toward
zero), while old-new relations use the same retained spectral-region score to
exclude new rows from historical regions. Per class, the worst old-old rival
and worst old-new rival are retained separately; their two mandatory role risks
are averaged so neither can lose gradient. Real current CE supplies the new side
of old-new boundaries and all new-new boundaries.

There is no replay, teacher/KD, generated feature, transport, phase-specific
head, SSS training penalty, old-row gradient mask, or analytic classifier.
"""

import math
import sys
from numbers import Integral, Real
from typing import Any, Dict, Mapping, Optional, Sequence

import torch
from tqdm import tqdm

from losses.loss import (
    cosine_classification_objective,
    reference_anchored_historical_decision_objective,
    unified_incremental_classification_objective,
)
from models.classifier import PersistentCosineClassifier
from utils.eval import (
    old_new_summary,
    phase_origin_confusion_diagnostics,
)

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

    # trainers/trainer.py validates this high-level result contract.
    INCREMENTAL_RESULT_FIELDS = frozenset(
        {
            "phase",
            "old_class_ids",
            "new_class_ids",
            "seen_class_ids",
            "history",
            "final_epoch",
            "cumulative_test",
            "old_new",
            "forgetting",
            "method_state",
            "geometry_committed",
            "phase_finalized",
        }
    )

    def _require_classifier(self) -> PersistentCosineClassifier:
        classifier = getattr(self.model, "classifier", None)
        if not isinstance(classifier, PersistentCosineClassifier):
            raise RuntimeError(
                "model.classifier must be PersistentCosineClassifier"
            )
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
            raise RuntimeError(
                "seen classes must equal old classes followed by new classes"
            )
        if tuple(self.model.committed_class_ids) != old_ids:
            raise RuntimeError(
                "historical class structure does not match the phase"
            )
        if tuple(self.model.classifier_class_ids) != old_ids:
            raise RuntimeError(
                "classifier must contain exactly historical rows at phase start"
            )
        return old_ids, new_ids, seen_ids

    def _validation_loader(
        self,
        phase: int,
        *,
        batch_size: int,
    ) -> Optional[Any]:
        if not isinstance(self.dataset.validation_enabled, bool):
            raise RuntimeError("dataset.validation_enabled must be bool")
        if not self.dataset.validation_enabled:
            return None
        return self.dataset.get_phase_dataloader(
            phase,
            split="val",
            batch_size=batch_size,
            shuffle=False,
        )

    @staticmethod
    def _snapshot_geometry(
        bank: Any,
        class_ids: Sequence[int],
    ) -> Dict[str, Tensor]:
        rows = bank.class_geometries(class_ids)
        return {
            "class_ids": rows.class_ids.detach().clone(),
            "sample_counts": rows.sample_counts.detach().clone(),
            "identity": rows.identity.detach().clone(),
            "structure": rows.structure.detach().clone(),
        }

    @staticmethod
    def _assert_geometry_unchanged(
        bank: Any,
        class_ids: Sequence[int],
        reference: Mapping[str, Tensor],
    ) -> None:
        current = bank.class_geometries(class_ids)
        checks = {
            "class_ids": current.class_ids,
            "sample_counts": current.sample_counts,
            "identity": current.identity,
            "structure": current.structure,
        }
        for name, value in checks.items():
            expected = reference[name]
            if value.shape != expected.shape or not torch.equal(value, expected):
                raise RuntimeError(f"historical geometry changed: {name}")

    @staticmethod
    @torch.no_grad()
    def _snapshot_backbone(
        backbone: torch.nn.Module,
    ) -> Dict[str, Tensor]:
        return {
            name: value.detach().cpu().clone()
            for name, value in backbone.state_dict().items()
        }

    @staticmethod
    @torch.no_grad()
    def _assert_backbone_unchanged(
        backbone: torch.nn.Module,
        reference: Mapping[str, Tensor],
    ) -> None:
        current = backbone.state_dict()
        if set(current) != set(reference):
            raise RuntimeError("backbone state keys changed")
        for name, expected in reference.items():
            value = current[name].detach().cpu()
            if value.shape != expected.shape or not torch.equal(value, expected):
                raise RuntimeError(f"frozen backbone changed: {name}")

    @staticmethod
    def _assert_unit_classifier_rows(
        classifier: PersistentCosineClassifier,
    ) -> None:
        norms = classifier.parameter_row_norms()
        if norms.numel() == 0 or not bool(torch.isfinite(norms).all()):
            raise RuntimeError("classifier row norms are invalid")
        tolerance = 3e-6 if norms.dtype == torch.float32 else 1e-12
        if not bool(
            torch.allclose(
                norms,
                torch.ones_like(norms),
                rtol=tolerance,
                atol=tolerance,
            )
        ):
            minimum = float(norms.detach().amin().cpu().item())
            maximum = float(norms.detach().amax().cpu().item())
            raise RuntimeError(
                "classifier parameter rows left the unit-norm gauge; "
                f"min={minimum:.8f}, max={maximum:.8f}"
            )

    @staticmethod
    def _current_class_balance_weights(
        classifier: PersistentCosineClassifier,
        *,
        new_class_ids: Sequence[int],
        sample_counts: Tensor,
    ) -> Tensor:
        new_ids = tuple(int(v) for v in new_class_ids)
        counts = torch.as_tensor(
            sample_counts,
            device=classifier.weight.device,
        )
        if counts.shape != (len(new_ids),) or counts.dtype != torch.long:
            raise ValueError(
                "sample_counts must be torch.long with one value per new class"
            )
        if bool((counts <= 0).any()):
            raise ValueError("new-class sample counts must be positive")

        # cosine_classification_objective only indexes the weight belonging to
        # the current target. Historical entries therefore remain one and are
        # never consumed by this current-only loader.
        weights = torch.ones(
            classifier.class_count,
            device=classifier.weight.device,
            dtype=classifier.weight.dtype,
        )
        total = float(counts.sum().item())
        class_count = float(len(new_ids))
        for position, class_id in enumerate(new_ids):
            row = int(classifier.row_indices((class_id,)).item())
            weights[row] = total / (
                class_count * float(counts[position].item())
            )
        return weights

    def _historical_objective(
        self,
        *,
        seen_ids: Sequence[int],
        new_ids: Sequence[int],
        historical_class_ids: Tensor,
        reference_old_old_score: Tensor,
    ):
        old_ids = tuple(
            int(v) for v in historical_class_ids.detach().cpu().tolist()
        )
        safety = self.model.boundary_safety_scores(
            historical_class_ids=old_ids,
            rival_class_ids=seen_ids,
        )
        returned_old = tuple(int(v) for v in safety.historical_class_ids)
        returned_rivals = tuple(int(v) for v in safety.rival_class_ids)
        if returned_old != old_ids:
            raise RuntimeError("historical boundary class order changed")
        if returned_rivals != tuple(int(v) for v in seen_ids):
            raise RuntimeError("historical rival class order changed")
        return reference_anchored_historical_decision_objective(
            mean_gap=safety.mean_gap,
            dispersion_upper_bound=safety.dispersion_upper_bound,
            historical_class_ids=historical_class_ids,
            seen_class_ids=self.model.classifier.class_ids,
            new_class_ids=torch.tensor(
                tuple(int(v) for v in new_ids),
                device=self.model.device,
                dtype=torch.long,
            ),
            reference_old_old_score=reference_old_old_score,
        )

    def _phase_groups(
        self,
        phase: int,
    ) -> Dict[int, list[int]]:
        groups: Dict[int, list[int]] = {}
        if not hasattr(self.dataset, "phase_to_classes"):
            raise RuntimeError(
                "dataset must expose phase_to_classes for phase-origin reporting"
            )
        for phase_id in range(phase + 1):
            if phase_id not in self.dataset.phase_to_classes:
                raise RuntimeError(
                    f"dataset phase_to_classes lacks phase {phase_id}"
                )
            groups[phase_id] = [
                int(v) for v in self.dataset.phase_to_classes[phase_id]
            ]
        return groups

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
        batch_size = _positive_int(batch_size, name="batch_size")
        learning_rate = _finite_positive("lr", lr)

        self.model.validate_model_state()
        if getattr(self.dataset, "current_phase", None) is not None:
            raise RuntimeError("dataset already has an active phase")

        previous_phase = phase_id - 1
        previous_seen = tuple(
            self._validated_class_ids(
                self.dataset.get_seen_classes(previous_phase),
                name="previous_seen_classes",
            )
        )
        if tuple(self.model.classifier_class_ids) != previous_seen:
            raise RuntimeError(
                "checkpoint classifier does not match previous phase"
            )

        eval_batch_size = _positive_int(
            self.cfg("eval_batch_size", 256),
            name="eval_batch_size",
        )
        old_before = self.evaluate_loader(
            self.dataset.get_reporting_dataloader(
                previous_phase,
                batch_size=eval_batch_size,
            ),
            class_ids=previous_seen,
            target_class_ids=previous_seen,
        )

        self.dataset.start_phase(phase_id)
        old_ids, new_ids, seen_ids = self._phase_classes(phase_id)

        # The representation is the stable knowledge carrier. No incremental
        # optimizer receives a backbone parameter.
        for parameter in self.model.backbone.parameters():
            parameter.requires_grad_(False)
            parameter.grad = None
        self.model.backbone.eval()

        backbone_snapshot = self._snapshot_backbone(self.model.backbone)
        old_geometry_snapshot = self._snapshot_geometry(
            self.model.geometry_bank,
            old_ids,
        )
        historical_class_ids_tensor = torch.tensor(
            old_ids,
            device=self.model.device,
            dtype=torch.long,
        )

        train_loader = self.dataset.get_phase_dataloader(
            phase_id,
            split="train",
            batch_size=batch_size,
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

        # Q is fixed, so one deterministic full-TRAIN extraction is valid for
        # initialization, class-count balancing, and the final new-class commit.
        current_train = self.collect_ordered_features(train_eval_loader)
        current_labels = torch.as_tensor(current_train["labels"])
        observed = tuple(
            sorted(
                int(v)
                for v in current_labels.detach().cpu().unique().tolist()
            )
        )
        if observed != tuple(sorted(new_ids)):
            raise RuntimeError(
                "incremental TRAIN labels do not match new classes"
            )

        current_statistics = self.model.estimate_class_statistics(
            current_train["ordered_features"],
            current_train["labels"],
            class_ids=new_ids,
        )
        if tuple(
            int(v)
            for v in current_statistics.class_ids.detach().cpu().tolist()
        ) != new_ids:
            raise RuntimeError(
                "current class-statistic order changed"
            )

        classifier = self._require_classifier()

        # Fix the irrelevant scale gauge of historical rows before adaptation.
        # This changes no deployed logit or prediction because forward normalizes
        # each row already.
        classifier.canonicalize_parameter_rows_(old_ids)
        old_phase_start_directions, _ = classifier.normalized_rows(old_ids)
        old_phase_start_directions = (
            old_phase_start_directions.detach().clone()
        )

        # The previous finalized classifier defines the historical old-old
        # decision regions that this phase is allowed to adapt but not degrade.
        # The reference is phase-local and derived entirely from existing W,
        # I_c and S_c; it is not a replay buffer or additional persistent memory.
        with torch.no_grad():
            reference_safety = self.model.boundary_safety_scores(
                historical_class_ids=old_ids,
                rival_class_ids=old_ids,
            )
            if tuple(reference_safety.historical_class_ids) != old_ids:
                raise RuntimeError("old-old reference row order changed")
            if tuple(reference_safety.rival_class_ids) != old_ids:
                raise RuntimeError("old-old reference column order changed")
            reference_old_old_score = reference_safety.score.detach().clone()
            if not bool(torch.isfinite(reference_old_old_score).all()):
                raise RuntimeError("old-old reference score contains NaN/Inf")

        self.model.add_classifier_classes(
            new_ids,
            initial_identity=current_statistics.identity,
        )
        classifier = self._require_classifier()
        if tuple(classifier.seen_class_ids) != seen_ids:
            raise RuntimeError(
                "classifier expansion produced wrong class order"
            )

        # All rows use one phase-independent parameter gauge.
        classifier.canonicalize_parameter_rows_()
        self._assert_unit_classifier_rows(classifier)

        current_balance_weights = self._current_class_balance_weights(
            classifier,
            new_class_ids=new_ids,
            sample_counts=current_statistics.sample_counts,
        )

        optimizer = torch.optim.AdamW(
            [classifier.weight],
            lr=learning_rate,
            weight_decay=0.0,
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

        old_count = len(old_ids)
        new_count = len(new_ids)

        history: list[Dict[str, Any]] = []
        progress = tqdm(
            range(epoch_count),
            desc=(
                f"Incremental phase {phase_id} | "
                "stable Q + reference-anchored spectral decision regions"
            ),
            unit="epoch",
            dynamic_ncols=True,
            file=sys.stdout,
        )

        for epoch in progress:
            self.model.train()
            self.model.backbone.eval()
            epoch_lr = float(optimizer.param_groups[0]["lr"])

            current_sum = 0.0
            historical_sum = 0.0
            identity_ce_sum = 0.0
            spectral_score_mean_sum = 0.0
            spectral_score_min = math.inf
            dispersion_mean_sum = 0.0
            zero_dispersion_fraction_sum = 0.0
            total_sum = 0.0
            sample_count = 0
            batch_count = 0
            class_correct = {class_id: 0 for class_id in new_ids}
            class_total = {class_id: 0 for class_id in new_ids}

            for batch in train_loader:
                patch, spectrum, batch_labels = self.unpack_batch(batch)
                batch_classes = set(
                    int(v)
                    for v in batch_labels.detach().cpu().tolist()
                )
                if not batch_classes.issubset(set(new_ids)):
                    raise RuntimeError(
                        "incremental TRAIN batch contains historical/future class"
                    )

                optimizer.zero_grad(set_to_none=True)

                with torch.no_grad():
                    q = self.model.extract_features(
                        patch,
                        center_spectrum=spectrum,
                        return_aux=False,
                    ).ordered_features

                classification = self.model.classify_features(
                    q,
                    class_ids=seen_ids,
                )
                current_objective = cosine_classification_objective(
                    output=classification,
                    labels_global=batch_labels,
                    class_balance_weights=current_balance_weights,
                )
                historical_objective = self._historical_objective(
                    seen_ids=seen_ids,
                    new_ids=new_ids,
                    historical_class_ids=historical_class_ids_tensor,
                    reference_old_old_score=reference_old_old_score,
                )
                unified = unified_incremental_classification_objective(
                    current_real_risk=current_objective.loss,
                    historical_spectral_decision_risk=historical_objective.loss,
                    current_class_count=new_count,
                    historical_class_count=old_count,
                )

                if not bool(torch.isfinite(unified.loss)):
                    raise RuntimeError(
                        "unified incremental loss became NaN/Inf"
                    )

                unified.loss.backward()
                gradient = classifier.weight.grad
                if gradient is None:
                    raise RuntimeError("classifier received no gradient")
                if not bool(torch.isfinite(gradient).all()):
                    raise RuntimeError(
                        "classifier gradient contains NaN/Inf"
                    )

                if gradient_clip > 0.0:
                    torch.nn.utils.clip_grad_norm_(
                        [classifier.weight],
                        gradient_clip,
                        error_if_nonfinite=True,
                    )

                optimizer.step()
                classifier.canonicalize_parameter_rows_()
                self._assert_unit_classifier_rows(classifier)

                n = int(batch_labels.numel())
                sample_count += n
                batch_count += 1
                current_sum += (
                    float(current_objective.loss.detach().item()) * n
                )
                # Historical risk is the exact macro old-class term evaluated
                # once per stochastic current batch; average it by batch count,
                # not by the number of current pixels.
                historical_sum += float(
                    historical_objective.loss.detach().item()
                )
                identity_ce_sum += float(
                    historical_objective.identity_cross_entropy.detach().item()
                )
                spectral_score_mean_sum += float(
                    historical_objective.mean_spectral_decision_score.detach().item()
                )
                spectral_score_min = min(
                    spectral_score_min,
                    float(
                        historical_objective.minimum_spectral_decision_score
                        .detach()
                        .item()
                    ),
                )
                dispersion_mean_sum += float(
                    historical_objective.mean_dispersion_upper_bound.detach().item()
                )
                zero_dispersion_fraction_sum += float(
                    historical_objective.zero_dispersion_fraction.detach().item()
                )
                total_sum += float(unified.loss.detach().item())

                prediction = classification.prediction.detach()
                for class_id in new_ids:
                    mask = batch_labels.eq(class_id)
                    count = int(mask.sum().item())
                    if count:
                        class_total[class_id] += count
                        class_correct[class_id] += int(
                            prediction[mask]
                            .eq(batch_labels[mask])
                            .sum()
                            .item()
                        )

            if sample_count <= 0 or batch_count <= 0:
                raise RuntimeError(
                    "incremental TRAIN loader yielded no samples"
                )
            missing = [
                class_id
                for class_id in new_ids
                if class_total[class_id] == 0
            ]
            if missing:
                raise RuntimeError(
                    f"incremental epoch missed new classes {missing}"
                )

            train_ba = sum(
                class_correct[class_id]
                / float(class_total[class_id])
                for class_id in new_ids
            ) / float(new_count)

            validation = (
                None
                if validation_loader is None
                else self.evaluate_loader(
                    validation_loader,
                    class_ids=seen_ids,
                    target_class_ids=new_ids,
                )
            )

            # Re-evaluate the deterministic historical spectral state using
            # the post-step classifier, so the logged diagnostic corresponds to
            # the actual end-of-epoch decision state.
            with torch.no_grad():
                post_historical = self._historical_objective(
                    seen_ids=seen_ids,
                    new_ids=new_ids,
                    historical_class_ids=historical_class_ids_tensor,
                    reference_old_old_score=reference_old_old_score,
                )

            record = {
                "epoch": epoch + 1,
                "learning_rate": epoch_lr,
                "current_real_risk": current_sum / float(sample_count),
                "historical_spectral_decision_risk": historical_sum / float(batch_count),
                "historical_identity_ce": identity_ce_sum / float(batch_count),
                "historical_spectral_score_mean": (
                    spectral_score_mean_sum / float(batch_count)
                ),
                "historical_spectral_score_min": spectral_score_min,
                "historical_dispersion_mean": (
                    dispersion_mean_sum / float(batch_count)
                ),
                "historical_zero_dispersion_fraction": (
                    zero_dispersion_fraction_sum / float(batch_count)
                ),
                "total_loss": total_sum / float(batch_count),
                "new_train_ba": train_ba,
                "validation_new_ba": (
                    None
                    if validation is None
                    else float(validation["balanced_accuracy"])
                ),
                "historical_identity_accuracy": float(
                    post_historical.identity_accuracy.item()
                ),
                "historical_identity_margin": float(
                    post_historical.mean_identity_margin.item()
                ),
                "old_old_retention_risk": float(
                    post_historical.old_old_retention_risk.item()
                ),
                "old_new_exclusion_risk": float(
                    post_historical.old_new_exclusion_risk.item()
                ),
                "old_old_reference_min_score": float(
                    post_historical.old_old_reference_min_score.item()
                ),
                "old_old_current_min_score": float(
                    post_historical.old_old_current_min_score.item()
                ),
                "old_old_max_degradation": float(
                    post_historical.old_old_max_degradation.item()
                ),
                "old_old_violation_fraction": float(
                    post_historical.old_old_violation_fraction.item()
                ),
                "old_new_min_score": float(
                    post_historical.old_new_min_score.item()
                ),
                "old_new_violation_fraction": float(
                    post_historical.old_new_violation_fraction.item()
                ),
            }
            history.append(record)

            message = (
                f"[Phase {phase_id} | Epoch {epoch + 1:03d}/{epoch_count:03d}] "
                f"LR={epoch_lr:.3e} | "
                f"Current={record['current_real_risk']:.4f} | "
                f"Hist={record['historical_spectral_decision_risk']:.4f} | "
                f"I-CE={record['historical_identity_ce']:.4f} | "
                f"Rmean={record['historical_spectral_score_mean']:.4f} | "
                f"Rmin={record['historical_spectral_score_min']:.4f} | "
                f"OO={record['old_old_retention_risk']:.4f} | "
                f"ON={record['old_new_exclusion_risk']:.4f} | "
                f"OOdeg={record['old_old_max_degradation']:.4f} | "
                f"ONmin={record['old_new_min_score']:.4f} | "
                f"Total={record['total_loss']:.4f} | "
                f"newTrain_BA={100.0 * train_ba:.2f}% | "
                f"HistIAcc={100.0 * record['historical_identity_accuracy']:.2f}%"
            )
            if validation is not None:
                message += (
                    f" | newVal_BA="
                    f"{100.0 * validation['balanced_accuracy']:.2f}%"
                )
            tqdm.write(message, file=sys.stdout)
            scheduler.step()

        progress.close()

        self._assert_backbone_unchanged(
            self.model.backbone,
            backbone_snapshot,
        )
        self._assert_geometry_unchanged(
            self.model.geometry_bank,
            old_ids,
            old_geometry_snapshot,
        )
        self._assert_unit_classifier_rows(classifier)

        # Q never changed, so the phase-start deterministic current statistics
        # remain exactly the statistics of the final representation.
        self.model.add_classes(
            current_train["ordered_features"],
            current_train["labels"],
            class_ids=new_ids,
        )
        del current_train, current_statistics

        if tuple(self.model.committed_class_ids) != seen_ids:
            raise RuntimeError(
                "geometry commit produced wrong class order"
            )
        if tuple(self.model.classifier_class_ids) != seen_ids:
            raise RuntimeError(
                "classifier class order changed unexpectedly"
            )

        self._assert_geometry_unchanged(
            self.model.geometry_bank,
            old_ids,
            old_geometry_snapshot,
        )
        self._assert_backbone_unchanged(
            self.model.backbone,
            backbone_snapshot,
        )
        self._assert_unit_classifier_rows(classifier)

        with torch.no_grad():
            final_historical = self._historical_objective(
                seen_ids=seen_ids,
                new_ids=new_ids,
                historical_class_ids=historical_class_ids_tensor,
                reference_old_old_score=reference_old_old_score,
            )
            final_old_directions, _ = classifier.normalized_rows(old_ids)
            old_direction_cosine = torch.sum(
                old_phase_start_directions * final_old_directions,
                dim=1,
            )
            if not bool(torch.isfinite(old_direction_cosine).all()):
                raise RuntimeError(
                    "historical classifier movement diagnostic is invalid"
                )

        self.dataset.finalize_phase(phase_id)
        cumulative_test = self.evaluate_loader(
            self.dataset.get_reporting_dataloader(
                phase_id,
                batch_size=eval_batch_size,
            ),
            class_ids=seen_ids,
            target_class_ids=seen_ids,
        )
        groups = old_new_summary(
            cumulative_test,
            old_class_ids=old_ids,
            new_class_ids=new_ids,
        )
        phase_origin = phase_origin_confusion_diagnostics(
            cumulative_test,
            phase_groups=self._phase_groups(phase_id),
        )

        old_before_ba = float(old_before["balanced_accuracy"])
        old_after_ba = float(groups["old"]["balanced_accuracy"])

        method_state = {
            "strategy": "stable_Q_reference_anchored_spectral_decision_regions",
            "backbone_unchanged": True,
            "historical_geometry_unchanged": True,
            "classifier_rows_jointly_optimized": True,
            "classifier_rows_unit_norm": True,
            "current_class_count": new_count,
            "historical_class_count": old_count,
            "current_class_fraction": new_count / float(len(seen_ids)),
            "historical_class_fraction": old_count / float(len(seen_ids)),
            "final_historical_spectral_decision_risk": float(
                final_historical.loss.item()
            ),
            "final_historical_identity_ce": float(
                final_historical.identity_cross_entropy.item()
            ),
            "final_historical_spectral_score_mean": float(
                final_historical.mean_spectral_decision_score.item()
            ),
            "final_historical_spectral_score_min": float(
                final_historical.minimum_spectral_decision_score.item()
            ),
            "final_historical_dispersion_mean": float(
                final_historical.mean_dispersion_upper_bound.item()
            ),
            "final_historical_zero_dispersion_fraction": float(
                final_historical.zero_dispersion_fraction.item()
            ),
            "final_historical_identity_accuracy": float(
                final_historical.identity_accuracy.item()
            ),
            "final_historical_identity_margin": float(
                final_historical.mean_identity_margin.item()
            ),
            "final_old_old_retention_risk": float(
                final_historical.old_old_retention_risk.item()
            ),
            "final_old_new_exclusion_risk": float(
                final_historical.old_new_exclusion_risk.item()
            ),
            "final_old_old_reference_min_score": float(
                final_historical.old_old_reference_min_score.item()
            ),
            "final_old_old_current_min_score": float(
                final_historical.old_old_current_min_score.item()
            ),
            "final_old_old_max_degradation": float(
                final_historical.old_old_max_degradation.item()
            ),
            "final_old_old_violation_fraction": float(
                final_historical.old_old_violation_fraction.item()
            ),
            "final_old_new_min_score": float(
                final_historical.old_new_min_score.item()
            ),
            "final_old_new_violation_fraction": float(
                final_historical.old_new_violation_fraction.item()
            ),
            "reference_old_old_class_ids": list(old_ids),
            "reference_old_old_score_matrix": (
                reference_old_old_score.detach().cpu().tolist()
            ),
            "historical_classifier_direction_cosine_to_phase_start_mean": float(
                old_direction_cosine.mean().item()
            ),
            "historical_classifier_direction_cosine_to_phase_start_min": float(
                old_direction_cosine.min().item()
            ),
            "replay": False,
            "teacher": False,
            "knowledge_distillation": False,
            "stored_old_samples": 0,
            "stored_old_features": 0,
        }

        return {
            "phase": phase_id,
            "old_class_ids": list(old_ids),
            "new_class_ids": list(new_ids),
            "seen_class_ids": list(seen_ids),
            "history": history,
            "final_epoch": epoch_count,
            "cumulative_test": cumulative_test,
            "old_new": groups,
            "phase_origin_confusion": phase_origin,
            "forgetting": {
                "old_balanced_accuracy_before": old_before_ba,
                "old_balanced_accuracy_after": old_after_ba,
                "old_balanced_accuracy_drop": (
                    old_before_ba - old_after_ba
                ),
            },
            "method_state": method_state,
            "geometry_committed": True,
            "phase_finalized": True,
        }


__all__ = ["IncrementalPhaseTrainer"]




















# from __future__ import annotations

# """Stable-Q structural classifier adaptation for SCOPE-HSI.

# Incremental phase t keeps the canonical HSI representation Q fixed.  The single
# persistent angular classifier is expanded with current-class rows initialized
# from current class identities and then ALL seen classifier rows are optimized
# from a phase-balanced classification risk:

#     current classes    -> real current TRAIN Q and labels,
#     historical classes -> retained {I_c,S_c}.

# For each historical class c and every seen rival j != c, the retained class
# identity and full feature-resolved structure define

#     mu_cj       = <I_c, U_c-U_j>_F,
#     sigma_bar_cj = structural decision-gap spread from S_c[K,K,D],
#     r_cj        = mu_cj / sigma_bar_cj   (when spread is non-zero).

# The historical per-class risk is

#     R_c^hist = log(1 + sum_{j != c} exp(-r_cj)).

# For numerical zero spread, r_cj is the exact deterministic mean gap.  Thus
# historical state directly preserves the all-seen decision region of the same
# persistent classifier used at inference.

# There is no replay, teacher/KD, generated feature, transport, phase-specific
# head, SSS training penalty, old-row gradient mask, or analytic classifier.
# """

# import math
# import sys
# from numbers import Integral, Real
# from typing import Any, Dict, Mapping, Optional, Sequence

# import torch
# from tqdm import tqdm

# from losses.loss import (
#     cosine_classification_objective,
#     historical_spectral_decision_objective,
#     unified_incremental_classification_objective,
# )
# from models.classifier import PersistentCosineClassifier
# from utils.eval import (
#     old_new_summary,
#     phase_origin_confusion_diagnostics,
# )

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


# def _finite_positive(name: str, value: object) -> float:
#     result = float(value)
#     if not math.isfinite(result) or result <= 0.0:
#         raise ValueError(f"{name} must be finite and positive")
#     return result


# def _finite_nonnegative(name: str, value: object) -> float:
#     result = float(value)
#     if not math.isfinite(result) or result < 0.0:
#         raise ValueError(f"{name} must be finite and non-negative")
#     return result


# class IncrementalPhaseTrainer:
#     """Train one strict non-exemplar incremental phase."""

#     # trainers/trainer.py validates this high-level result contract.
#     INCREMENTAL_RESULT_FIELDS = frozenset(
#         {
#             "phase",
#             "old_class_ids",
#             "new_class_ids",
#             "seen_class_ids",
#             "history",
#             "final_epoch",
#             "cumulative_test",
#             "old_new",
#             "forgetting",
#             "method_state",
#             "geometry_committed",
#             "phase_finalized",
#         }
#     )

#     def _require_classifier(self) -> PersistentCosineClassifier:
#         classifier = getattr(self.model, "classifier", None)
#         if not isinstance(classifier, PersistentCosineClassifier):
#             raise RuntimeError(
#                 "model.classifier must be PersistentCosineClassifier"
#             )
#         return classifier

#     def _phase_classes(
#         self,
#         phase: int,
#     ) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
#         old_ids = tuple(
#             self._validated_class_ids(
#                 self.dataset.get_old_classes(phase),
#                 name=f"phase_{phase}_old_classes",
#             )
#         )
#         new_ids = tuple(
#             self._validated_class_ids(
#                 self.dataset.get_new_classes(phase),
#                 name=f"phase_{phase}_new_classes",
#             )
#         )
#         seen_ids = tuple(
#             self._validated_class_ids(
#                 self.dataset.get_seen_classes(phase),
#                 name=f"phase_{phase}_seen_classes",
#             )
#         )
#         if not old_ids or not new_ids:
#             raise RuntimeError("incremental phase requires old and new classes")
#         if set(old_ids).intersection(new_ids):
#             raise RuntimeError("old and new class IDs overlap")
#         if seen_ids != old_ids + new_ids:
#             raise RuntimeError(
#                 "seen classes must equal old classes followed by new classes"
#             )
#         if tuple(self.model.committed_class_ids) != old_ids:
#             raise RuntimeError(
#                 "historical class structure does not match the phase"
#             )
#         if tuple(self.model.classifier_class_ids) != old_ids:
#             raise RuntimeError(
#                 "classifier must contain exactly historical rows at phase start"
#             )
#         return old_ids, new_ids, seen_ids

#     def _validation_loader(
#         self,
#         phase: int,
#         *,
#         batch_size: int,
#     ) -> Optional[Any]:
#         if not isinstance(self.dataset.validation_enabled, bool):
#             raise RuntimeError("dataset.validation_enabled must be bool")
#         if not self.dataset.validation_enabled:
#             return None
#         return self.dataset.get_phase_dataloader(
#             phase,
#             split="val",
#             batch_size=batch_size,
#             shuffle=False,
#         )

#     @staticmethod
#     def _snapshot_geometry(
#         bank: Any,
#         class_ids: Sequence[int],
#     ) -> Dict[str, Tensor]:
#         rows = bank.class_geometries(class_ids)
#         return {
#             "class_ids": rows.class_ids.detach().clone(),
#             "sample_counts": rows.sample_counts.detach().clone(),
#             "identity": rows.identity.detach().clone(),
#             "structure": rows.structure.detach().clone(),
#         }

#     @staticmethod
#     def _assert_geometry_unchanged(
#         bank: Any,
#         class_ids: Sequence[int],
#         reference: Mapping[str, Tensor],
#     ) -> None:
#         current = bank.class_geometries(class_ids)
#         checks = {
#             "class_ids": current.class_ids,
#             "sample_counts": current.sample_counts,
#             "identity": current.identity,
#             "structure": current.structure,
#         }
#         for name, value in checks.items():
#             expected = reference[name]
#             if value.shape != expected.shape or not torch.equal(value, expected):
#                 raise RuntimeError(f"historical geometry changed: {name}")

#     @staticmethod
#     @torch.no_grad()
#     def _snapshot_backbone(
#         backbone: torch.nn.Module,
#     ) -> Dict[str, Tensor]:
#         return {
#             name: value.detach().cpu().clone()
#             for name, value in backbone.state_dict().items()
#         }

#     @staticmethod
#     @torch.no_grad()
#     def _assert_backbone_unchanged(
#         backbone: torch.nn.Module,
#         reference: Mapping[str, Tensor],
#     ) -> None:
#         current = backbone.state_dict()
#         if set(current) != set(reference):
#             raise RuntimeError("backbone state keys changed")
#         for name, expected in reference.items():
#             value = current[name].detach().cpu()
#             if value.shape != expected.shape or not torch.equal(value, expected):
#                 raise RuntimeError(f"frozen backbone changed: {name}")

#     @staticmethod
#     def _assert_unit_classifier_rows(
#         classifier: PersistentCosineClassifier,
#     ) -> None:
#         norms = classifier.parameter_row_norms()
#         if norms.numel() == 0 or not bool(torch.isfinite(norms).all()):
#             raise RuntimeError("classifier row norms are invalid")
#         tolerance = 3e-6 if norms.dtype == torch.float32 else 1e-12
#         if not bool(
#             torch.allclose(
#                 norms,
#                 torch.ones_like(norms),
#                 rtol=tolerance,
#                 atol=tolerance,
#             )
#         ):
#             minimum = float(norms.detach().amin().cpu().item())
#             maximum = float(norms.detach().amax().cpu().item())
#             raise RuntimeError(
#                 "classifier parameter rows left the unit-norm gauge; "
#                 f"min={minimum:.8f}, max={maximum:.8f}"
#             )

#     @staticmethod
#     def _current_class_balance_weights(
#         classifier: PersistentCosineClassifier,
#         *,
#         new_class_ids: Sequence[int],
#         sample_counts: Tensor,
#     ) -> Tensor:
#         new_ids = tuple(int(v) for v in new_class_ids)
#         counts = torch.as_tensor(
#             sample_counts,
#             device=classifier.weight.device,
#         )
#         if counts.shape != (len(new_ids),) or counts.dtype != torch.long:
#             raise ValueError(
#                 "sample_counts must be torch.long with one value per new class"
#             )
#         if bool((counts <= 0).any()):
#             raise ValueError("new-class sample counts must be positive")

#         # cosine_classification_objective only indexes the weight belonging to
#         # the current target. Historical entries therefore remain one and are
#         # never consumed by this current-only loader.
#         weights = torch.ones(
#             classifier.class_count,
#             device=classifier.weight.device,
#             dtype=classifier.weight.dtype,
#         )
#         total = float(counts.sum().item())
#         class_count = float(len(new_ids))
#         for position, class_id in enumerate(new_ids):
#             row = int(classifier.row_indices((class_id,)).item())
#             weights[row] = total / (
#                 class_count * float(counts[position].item())
#             )
#         return weights

#     def _historical_objective(
#         self,
#         *,
#         seen_ids: Sequence[int],
#         historical_class_ids: Tensor,
#     ):
#         old_ids = tuple(
#             int(v) for v in historical_class_ids.detach().cpu().tolist()
#         )
#         safety = self.model.boundary_safety_scores(
#             historical_class_ids=old_ids,
#             rival_class_ids=seen_ids,
#         )
#         returned_old = tuple(int(v) for v in safety.historical_class_ids)
#         returned_rivals = tuple(int(v) for v in safety.rival_class_ids)
#         if returned_old != old_ids:
#             raise RuntimeError("historical boundary class order changed")
#         if returned_rivals != tuple(int(v) for v in seen_ids):
#             raise RuntimeError("historical rival class order changed")
#         return historical_spectral_decision_objective(
#             mean_gap=safety.mean_gap,
#             dispersion_upper_bound=safety.dispersion_upper_bound,
#             historical_class_ids=historical_class_ids,
#             rival_class_ids=self.model.classifier.class_ids,
#         )

#     def _phase_groups(
#         self,
#         phase: int,
#     ) -> Dict[int, list[int]]:
#         groups: Dict[int, list[int]] = {}
#         if not hasattr(self.dataset, "phase_to_classes"):
#             raise RuntimeError(
#                 "dataset must expose phase_to_classes for phase-origin reporting"
#             )
#         for phase_id in range(phase + 1):
#             if phase_id not in self.dataset.phase_to_classes:
#                 raise RuntimeError(
#                     f"dataset phase_to_classes lacks phase {phase_id}"
#                 )
#             groups[phase_id] = [
#                 int(v) for v in self.dataset.phase_to_classes[phase_id]
#             ]
#         return groups

#     def train_incremental_phase(
#         self,
#         *,
#         phase: int,
#         epochs: int,
#         batch_size: int,
#         lr: float,
#     ) -> Dict[str, Any]:
#         phase_id = _positive_int(phase, name="phase")
#         epoch_count = _positive_int(epochs, name="epochs")
#         batch_size = _positive_int(batch_size, name="batch_size")
#         learning_rate = _finite_positive("lr", lr)

#         self.model.validate_model_state()
#         if getattr(self.dataset, "current_phase", None) is not None:
#             raise RuntimeError("dataset already has an active phase")

#         previous_phase = phase_id - 1
#         previous_seen = tuple(
#             self._validated_class_ids(
#                 self.dataset.get_seen_classes(previous_phase),
#                 name="previous_seen_classes",
#             )
#         )
#         if tuple(self.model.classifier_class_ids) != previous_seen:
#             raise RuntimeError(
#                 "checkpoint classifier does not match previous phase"
#             )

#         eval_batch_size = _positive_int(
#             self.cfg("eval_batch_size", 256),
#             name="eval_batch_size",
#         )
#         old_before = self.evaluate_loader(
#             self.dataset.get_reporting_dataloader(
#                 previous_phase,
#                 batch_size=eval_batch_size,
#             ),
#             class_ids=previous_seen,
#             target_class_ids=previous_seen,
#         )

#         self.dataset.start_phase(phase_id)
#         old_ids, new_ids, seen_ids = self._phase_classes(phase_id)

#         # The representation is the stable knowledge carrier. No incremental
#         # optimizer receives a backbone parameter.
#         for parameter in self.model.backbone.parameters():
#             parameter.requires_grad_(False)
#             parameter.grad = None
#         self.model.backbone.eval()

#         backbone_snapshot = self._snapshot_backbone(self.model.backbone)
#         old_geometry_snapshot = self._snapshot_geometry(
#             self.model.geometry_bank,
#             old_ids,
#         )
#         historical_class_ids_tensor = torch.tensor(
#             old_ids,
#             device=self.model.device,
#             dtype=torch.long,
#         )

#         train_loader = self.dataset.get_phase_dataloader(
#             phase_id,
#             split="train",
#             batch_size=batch_size,
#             shuffle=True,
#         )
#         train_eval_loader = self.dataset.get_phase_dataloader(
#             phase_id,
#             split="train",
#             batch_size=eval_batch_size,
#             shuffle=False,
#         )
#         validation_loader = self._validation_loader(
#             phase_id,
#             batch_size=eval_batch_size,
#         )

#         # Q is fixed, so one deterministic full-TRAIN extraction is valid for
#         # initialization, class-count balancing, and the final new-class commit.
#         current_train = self.collect_ordered_features(train_eval_loader)
#         current_labels = torch.as_tensor(current_train["labels"])
#         observed = tuple(
#             sorted(
#                 int(v)
#                 for v in current_labels.detach().cpu().unique().tolist()
#             )
#         )
#         if observed != tuple(sorted(new_ids)):
#             raise RuntimeError(
#                 "incremental TRAIN labels do not match new classes"
#             )

#         current_statistics = self.model.estimate_class_statistics(
#             current_train["ordered_features"],
#             current_train["labels"],
#             class_ids=new_ids,
#         )
#         if tuple(
#             int(v)
#             for v in current_statistics.class_ids.detach().cpu().tolist()
#         ) != new_ids:
#             raise RuntimeError(
#                 "current class-statistic order changed"
#             )

#         classifier = self._require_classifier()

#         # Fix the irrelevant scale gauge of historical rows before adaptation.
#         # This changes no deployed logit or prediction because forward normalizes
#         # each row already.
#         classifier.canonicalize_parameter_rows_(old_ids)
#         old_phase_start_directions, _ = classifier.normalized_rows(old_ids)
#         old_phase_start_directions = (
#             old_phase_start_directions.detach().clone()
#         )

#         self.model.add_classifier_classes(
#             new_ids,
#             initial_identity=current_statistics.identity,
#         )
#         classifier = self._require_classifier()
#         if tuple(classifier.seen_class_ids) != seen_ids:
#             raise RuntimeError(
#                 "classifier expansion produced wrong class order"
#             )

#         # All rows use one phase-independent parameter gauge.
#         classifier.canonicalize_parameter_rows_()
#         self._assert_unit_classifier_rows(classifier)

#         current_balance_weights = self._current_class_balance_weights(
#             classifier,
#             new_class_ids=new_ids,
#             sample_counts=current_statistics.sample_counts,
#         )

#         optimizer = torch.optim.AdamW(
#             [classifier.weight],
#             lr=learning_rate,
#             weight_decay=0.0,
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

#         old_count = len(old_ids)
#         new_count = len(new_ids)

#         history: list[Dict[str, Any]] = []
#         progress = tqdm(
#             range(epoch_count),
#             desc=(
#                 f"Incremental phase {phase_id} | "
#                 "stable Q + all-seen spectral decision risk"
#             ),
#             unit="epoch",
#             dynamic_ncols=True,
#             file=sys.stdout,
#         )

#         for epoch in progress:
#             self.model.train()
#             self.model.backbone.eval()
#             epoch_lr = float(optimizer.param_groups[0]["lr"])

#             current_sum = 0.0
#             historical_sum = 0.0
#             identity_ce_sum = 0.0
#             spectral_score_mean_sum = 0.0
#             spectral_score_min = math.inf
#             dispersion_mean_sum = 0.0
#             zero_dispersion_fraction_sum = 0.0
#             total_sum = 0.0
#             sample_count = 0
#             batch_count = 0
#             class_correct = {class_id: 0 for class_id in new_ids}
#             class_total = {class_id: 0 for class_id in new_ids}

#             for batch in train_loader:
#                 patch, spectrum, batch_labels = self.unpack_batch(batch)
#                 batch_classes = set(
#                     int(v)
#                     for v in batch_labels.detach().cpu().tolist()
#                 )
#                 if not batch_classes.issubset(set(new_ids)):
#                     raise RuntimeError(
#                         "incremental TRAIN batch contains historical/future class"
#                     )

#                 optimizer.zero_grad(set_to_none=True)

#                 with torch.no_grad():
#                     q = self.model.extract_features(
#                         patch,
#                         center_spectrum=spectrum,
#                         return_aux=False,
#                     ).ordered_features

#                 classification = self.model.classify_features(
#                     q,
#                     class_ids=seen_ids,
#                 )
#                 current_objective = cosine_classification_objective(
#                     output=classification,
#                     labels_global=batch_labels,
#                     class_balance_weights=current_balance_weights,
#                 )
#                 historical_objective = self._historical_objective(
#                     seen_ids=seen_ids,
#                     historical_class_ids=historical_class_ids_tensor,
#                 )
#                 unified = unified_incremental_classification_objective(
#                     current_real_risk=current_objective.loss,
#                     historical_spectral_decision_risk=historical_objective.loss,
#                     current_class_count=new_count,
#                     historical_class_count=old_count,
#                 )

#                 if not bool(torch.isfinite(unified.loss)):
#                     raise RuntimeError(
#                         "unified incremental loss became NaN/Inf"
#                     )

#                 unified.loss.backward()
#                 gradient = classifier.weight.grad
#                 if gradient is None:
#                     raise RuntimeError("classifier received no gradient")
#                 if not bool(torch.isfinite(gradient).all()):
#                     raise RuntimeError(
#                         "classifier gradient contains NaN/Inf"
#                     )

#                 if gradient_clip > 0.0:
#                     torch.nn.utils.clip_grad_norm_(
#                         [classifier.weight],
#                         gradient_clip,
#                         error_if_nonfinite=True,
#                     )

#                 optimizer.step()
#                 classifier.canonicalize_parameter_rows_()
#                 self._assert_unit_classifier_rows(classifier)

#                 n = int(batch_labels.numel())
#                 sample_count += n
#                 batch_count += 1
#                 current_sum += (
#                     float(current_objective.loss.detach().item()) * n
#                 )
#                 # Historical risk is the exact macro old-class term evaluated
#                 # once per stochastic current batch; average it by batch count,
#                 # not by the number of current pixels.
#                 historical_sum += float(
#                     historical_objective.loss.detach().item()
#                 )
#                 identity_ce_sum += float(
#                     historical_objective.identity_cross_entropy.detach().item()
#                 )
#                 spectral_score_mean_sum += float(
#                     historical_objective.mean_spectral_decision_score.detach().item()
#                 )
#                 spectral_score_min = min(
#                     spectral_score_min,
#                     float(
#                         historical_objective.minimum_spectral_decision_score
#                         .detach()
#                         .item()
#                     ),
#                 )
#                 dispersion_mean_sum += float(
#                     historical_objective.mean_dispersion_upper_bound.detach().item()
#                 )
#                 zero_dispersion_fraction_sum += float(
#                     historical_objective.zero_dispersion_fraction.detach().item()
#                 )
#                 total_sum += float(unified.loss.detach().item())

#                 prediction = classification.prediction.detach()
#                 for class_id in new_ids:
#                     mask = batch_labels.eq(class_id)
#                     count = int(mask.sum().item())
#                     if count:
#                         class_total[class_id] += count
#                         class_correct[class_id] += int(
#                             prediction[mask]
#                             .eq(batch_labels[mask])
#                             .sum()
#                             .item()
#                         )

#             if sample_count <= 0 or batch_count <= 0:
#                 raise RuntimeError(
#                     "incremental TRAIN loader yielded no samples"
#                 )
#             missing = [
#                 class_id
#                 for class_id in new_ids
#                 if class_total[class_id] == 0
#             ]
#             if missing:
#                 raise RuntimeError(
#                     f"incremental epoch missed new classes {missing}"
#                 )

#             train_ba = sum(
#                 class_correct[class_id]
#                 / float(class_total[class_id])
#                 for class_id in new_ids
#             ) / float(new_count)

#             validation = (
#                 None
#                 if validation_loader is None
#                 else self.evaluate_loader(
#                     validation_loader,
#                     class_ids=seen_ids,
#                     target_class_ids=new_ids,
#                 )
#             )

#             # Re-evaluate the deterministic historical spectral state using
#             # the post-step classifier, so the logged diagnostic corresponds to
#             # the actual end-of-epoch decision state.
#             with torch.no_grad():
#                 post_historical = self._historical_objective(
#                     seen_ids=seen_ids,
#                     historical_class_ids=historical_class_ids_tensor,
#                 )

#             record = {
#                 "epoch": epoch + 1,
#                 "learning_rate": epoch_lr,
#                 "current_real_risk": current_sum / float(sample_count),
#                 "historical_spectral_decision_risk": historical_sum / float(batch_count),
#                 "historical_identity_ce": identity_ce_sum / float(batch_count),
#                 "historical_spectral_score_mean": (
#                     spectral_score_mean_sum / float(batch_count)
#                 ),
#                 "historical_spectral_score_min": spectral_score_min,
#                 "historical_dispersion_mean": (
#                     dispersion_mean_sum / float(batch_count)
#                 ),
#                 "historical_zero_dispersion_fraction": (
#                     zero_dispersion_fraction_sum / float(batch_count)
#                 ),
#                 "total_loss": total_sum / float(batch_count),
#                 "new_train_ba": train_ba,
#                 "validation_new_ba": (
#                     None
#                     if validation is None
#                     else float(validation["balanced_accuracy"])
#                 ),
#                 "historical_identity_accuracy": float(
#                     post_historical.identity_accuracy.item()
#                 ),
#                 "historical_identity_margin": float(
#                     post_historical.mean_identity_margin.item()
#                 ),
#             }
#             history.append(record)

#             message = (
#                 f"[Phase {phase_id} | Epoch {epoch + 1:03d}/{epoch_count:03d}] "
#                 f"LR={epoch_lr:.3e} | "
#                 f"Current={record['current_real_risk']:.4f} | "
#                 f"Hist={record['historical_spectral_decision_risk']:.4f} | "
#                 f"I-CE={record['historical_identity_ce']:.4f} | "
#                 f"Rmean={record['historical_spectral_score_mean']:.4f} | "
#                 f"Rmin={record['historical_spectral_score_min']:.4f} | "
#                 f"Total={record['total_loss']:.4f} | "
#                 f"newTrain_BA={100.0 * train_ba:.2f}% | "
#                 f"HistIAcc={100.0 * record['historical_identity_accuracy']:.2f}%"
#             )
#             if validation is not None:
#                 message += (
#                     f" | newVal_BA="
#                     f"{100.0 * validation['balanced_accuracy']:.2f}%"
#                 )
#             tqdm.write(message, file=sys.stdout)
#             scheduler.step()

#         progress.close()

#         self._assert_backbone_unchanged(
#             self.model.backbone,
#             backbone_snapshot,
#         )
#         self._assert_geometry_unchanged(
#             self.model.geometry_bank,
#             old_ids,
#             old_geometry_snapshot,
#         )
#         self._assert_unit_classifier_rows(classifier)

#         # Q never changed, so the phase-start deterministic current statistics
#         # remain exactly the statistics of the final representation.
#         self.model.add_classes(
#             current_train["ordered_features"],
#             current_train["labels"],
#             class_ids=new_ids,
#         )
#         del current_train, current_statistics

#         if tuple(self.model.committed_class_ids) != seen_ids:
#             raise RuntimeError(
#                 "geometry commit produced wrong class order"
#             )
#         if tuple(self.model.classifier_class_ids) != seen_ids:
#             raise RuntimeError(
#                 "classifier class order changed unexpectedly"
#             )

#         self._assert_geometry_unchanged(
#             self.model.geometry_bank,
#             old_ids,
#             old_geometry_snapshot,
#         )
#         self._assert_backbone_unchanged(
#             self.model.backbone,
#             backbone_snapshot,
#         )
#         self._assert_unit_classifier_rows(classifier)

#         with torch.no_grad():
#             final_historical = self._historical_objective(
#                 seen_ids=seen_ids,
#                 historical_class_ids=historical_class_ids_tensor,
#             )
#             final_old_directions, _ = classifier.normalized_rows(old_ids)
#             old_direction_cosine = torch.sum(
#                 old_phase_start_directions * final_old_directions,
#                 dim=1,
#             )
#             if not bool(torch.isfinite(old_direction_cosine).all()):
#                 raise RuntimeError(
#                     "historical classifier movement diagnostic is invalid"
#                 )

#         self.dataset.finalize_phase(phase_id)
#         cumulative_test = self.evaluate_loader(
#             self.dataset.get_reporting_dataloader(
#                 phase_id,
#                 batch_size=eval_batch_size,
#             ),
#             class_ids=seen_ids,
#             target_class_ids=seen_ids,
#         )
#         groups = old_new_summary(
#             cumulative_test,
#             old_class_ids=old_ids,
#             new_class_ids=new_ids,
#         )
#         phase_origin = phase_origin_confusion_diagnostics(
#             cumulative_test,
#             phase_groups=self._phase_groups(phase_id),
#         )

#         old_before_ba = float(old_before["balanced_accuracy"])
#         old_after_ba = float(groups["old"]["balanced_accuracy"])

#         method_state = {
#             "strategy": "stable_Q_all_seen_spectral_decision_risk",
#             "backbone_unchanged": True,
#             "historical_geometry_unchanged": True,
#             "classifier_rows_jointly_optimized": True,
#             "classifier_rows_unit_norm": True,
#             "current_class_count": new_count,
#             "historical_class_count": old_count,
#             "current_class_fraction": new_count / float(len(seen_ids)),
#             "historical_class_fraction": old_count / float(len(seen_ids)),
#             "final_historical_spectral_decision_risk": float(
#                 final_historical.loss.item()
#             ),
#             "final_historical_identity_ce": float(
#                 final_historical.identity_cross_entropy.item()
#             ),
#             "final_historical_spectral_score_mean": float(
#                 final_historical.mean_spectral_decision_score.item()
#             ),
#             "final_historical_spectral_score_min": float(
#                 final_historical.minimum_spectral_decision_score.item()
#             ),
#             "final_historical_dispersion_mean": float(
#                 final_historical.mean_dispersion_upper_bound.item()
#             ),
#             "final_historical_zero_dispersion_fraction": float(
#                 final_historical.zero_dispersion_fraction.item()
#             ),
#             "final_historical_identity_accuracy": float(
#                 final_historical.identity_accuracy.item()
#             ),
#             "final_historical_identity_margin": float(
#                 final_historical.mean_identity_margin.item()
#             ),
#             "historical_classifier_direction_cosine_to_phase_start_mean": float(
#                 old_direction_cosine.mean().item()
#             ),
#             "historical_classifier_direction_cosine_to_phase_start_min": float(
#                 old_direction_cosine.min().item()
#             ),
#             "replay": False,
#             "teacher": False,
#             "knowledge_distillation": False,
#             "stored_old_samples": 0,
#             "stored_old_features": 0,
#         }

#         return {
#             "phase": phase_id,
#             "old_class_ids": list(old_ids),
#             "new_class_ids": list(new_ids),
#             "seen_class_ids": list(seen_ids),
#             "history": history,
#             "final_epoch": epoch_count,
#             "cumulative_test": cumulative_test,
#             "old_new": groups,
#             "phase_origin_confusion": phase_origin,
#             "forgetting": {
#                 "old_balanced_accuracy_before": old_before_ba,
#                 "old_balanced_accuracy_after": old_after_ba,
#                 "old_balanced_accuracy_drop": (
#                     old_before_ba - old_after_ba
#                 ),
#             },
#             "method_state": method_state,
#             "geometry_committed": True,
#             "phase_finalized": True,
#         }


# __all__ = ["IncrementalPhaseTrainer"]
