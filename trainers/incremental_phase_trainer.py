from __future__ import annotations

"""Incremental training with direct complete-HSI-CDS historical supervision.

The architecture remains:
    real current HSI -> HSIMamba -> Q[K,D]
    raw centre spectrum -> deterministic P[B,3]
    persistent memory -> {I_Q,V_Q,I_P,V_P,N}
    deployed decision -> one bias-free normalized cosine classifier

The fix is intentionally narrow. The failed temporary-support/Stage-B path is
removed. The same real-sample direct-Q contrastive separation used in Phase 0 is
retained in every incremental phase. During every incremental update, the
transported historical CDS itself supervises the same classifier:

    current CE        : real new HSI; only current rows are live;
    HDC               : preserve phase-start historical decision relations;
    CDS separation    : real new HSI stays out of risky historical regions;
    historical CDS CE : aligned old I_Q supplies old positive supervision while
                        V_Q directly constrains old-vs-seen boundary clearance and
                        I_P/V_P prioritize spectrally dangerous competition.

There is no pseudo-feature, inversion, Gaussian/covariance model, second head,
post-hoc calibration stage, L-BFGS solve, or decision-region regulator.
"""

from dataclasses import dataclass
import math
import sys
from numbers import Integral, Real
from typing import Any, Dict, Mapping

import torch
from tqdm import tqdm

from losses.loss import (
    classification_objective,
    supervised_contrastive_q_loss,
    historical_decision_consistency_loss,
    hsi_cds_separation_loss,
    historical_full_cds_supervision_loss,
)
from models.classifier import SpectralCDSClassifier
from models.geometry_bank import (
    ClassDiscriminativeStructureState,
    PhysicalSpectralStructureBatch,
    ClassStructureBatch,
    merge_class_structure_batches,
)
from models.phase_coordinate_alignment import (
    CDSAffinityState,
    CurrentClassResidual,
    PhaseAlignmentDiagnostics,
    PhaseCoordinateTransform,
    align_historical_structure,
    derive_cds_affinity,
)
from utils.eval import old_new_summary, phase_origin_confusion_diagnostics

Tensor = torch.Tensor


def _positive_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (Integral, Real)):
        raise ValueError(f"{name} must be a positive integer")
    number = float(value)
    if not math.isfinite(number) or not number.is_integer() or number <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(number)


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


def _clone_structure(batch: ClassStructureBatch) -> ClassStructureBatch:
    if not isinstance(batch, ClassStructureBatch) or batch.class_count <= 0:
        raise ValueError("cannot clone an empty class structure")
    return ClassStructureBatch(
        class_ids=batch.class_ids.detach().clone(),
        sample_counts=batch.sample_counts.detach().clone(),
        identity=batch.identity.detach().clone(),
        variation=batch.variation.detach().clone(),
        spectral_identity=batch.spectral_identity.detach().clone(),
        spectral_variation=batch.spectral_variation.detach().clone(),
    )


def _targets_local(labels: Tensor, class_ids: Tensor) -> Tensor:
    value = torch.as_tensor(labels)
    ids = torch.as_tensor(class_ids)
    if value.ndim != 1 or value.numel() == 0 or value.dtype != torch.long:
        raise ValueError("labels must be non-empty rank-one torch.long")
    if ids.ndim != 1 or ids.numel() < 1 or ids.dtype != torch.long:
        raise ValueError("class_ids must be non-empty rank-one torch.long")
    if value.device != ids.device:
        raise ValueError("labels and class_ids must share device")
    matches = value.unsqueeze(1).eq(ids.unsqueeze(0))
    if not bool(matches.sum(dim=1).eq(1).all()):
        raise ValueError("labels contain a class absent from class_ids")
    return matches.to(torch.long).argmax(dim=1)


def _batch_coordinates(batch: Mapping[str, Any], *, expected_rows: int) -> Tensor:
    if "coord" not in batch:
        raise RuntimeError("incremental TRAIN batches must expose coord for phase-start reference lookup")
    coords = torch.as_tensor(batch["coord"])
    if coords.shape != (expected_rows, 2):
        raise RuntimeError("coord must be [N,2] and batch-aligned")
    if coords.dtype == torch.bool or coords.is_complex() or torch.is_floating_point(coords):
        raise RuntimeError("coord must use an integer dtype")
    return coords.to(dtype=torch.long, device="cpu")


@dataclass(frozen=True)
class CurrentTrainingState:
    task_mapped_historical: ClassStructureBatch
    migrated_historical: ClassStructureBatch
    current: ClassStructureBatch
    seen: ClassStructureBatch
    seen_cds: ClassDiscriminativeStructureState
    current_rows: Dict[str, Tensor]
    transform: PhaseCoordinateTransform
    affinity: CDSAffinityState
    residual: CurrentClassResidual
    alignment: PhaseAlignmentDiagnostics


class IncrementalPhaseTrainer:
    INCREMENTAL_RESULT_FIELDS = frozenset(
        {
            "phase", "old_class_ids", "new_class_ids", "seen_class_ids", "history",
            "final_epoch", "cumulative_test", "old_new", "forgetting", "method_state",
            "structure_committed", "phase_finalized",
        }
    )

    def _require_classifier(self) -> SpectralCDSClassifier:
        classifier = getattr(self.model, "classifier", None)
        if not isinstance(classifier, SpectralCDSClassifier) or classifier.parameter_count <= 0:
            raise RuntimeError("model.classifier must be the trainable cosine SpectralCDSClassifier")
        return classifier

    def _phase_classes(self, phase: int) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
        old_ids = tuple(self._validated_class_ids(self.dataset.get_old_classes(phase), name=f"phase_{phase}_old_classes"))
        new_ids = tuple(self._validated_class_ids(self.dataset.get_new_classes(phase), name=f"phase_{phase}_new_classes"))
        seen_ids = tuple(self._validated_class_ids(self.dataset.get_seen_classes(phase), name=f"phase_{phase}_seen_classes"))
        if not old_ids or not new_ids or set(old_ids).intersection(new_ids):
            raise RuntimeError("incremental phase has invalid old/new class partition")
        if seen_ids != old_ids + new_ids:
            raise RuntimeError("seen classes must equal old classes followed by new classes")
        if tuple(self.model.committed_class_ids) != old_ids:
            raise RuntimeError("historical CDS does not match the old class set")
        return old_ids, new_ids, seen_ids

    @staticmethod
    def _assert_frozen_reference(reference_backbone: torch.nn.Module) -> None:
        reference_backbone.eval()
        if any(parameter.requires_grad for parameter in reference_backbone.parameters()):
            raise RuntimeError("previous backbone reference must remain fully frozen")
        if any(parameter.grad is not None for parameter in reference_backbone.parameters()):
            raise RuntimeError("previous backbone reference must not carry gradients")

    @staticmethod
    @torch.no_grad()
    def _assert_reference_matches_phase_start(reference_backbone: torch.nn.Module, current_backbone: torch.nn.Module) -> None:
        left, right = reference_backbone.state_dict(), current_backbone.state_dict()
        if tuple(left.keys()) != tuple(right.keys()):
            raise RuntimeError("previous/current backbone schemas differ at phase start")
        for name in left:
            if left[name].shape != right[name].shape or left[name].dtype != right[name].dtype:
                raise RuntimeError(f"phase-start backbone state differs at {name}")
            if not torch.equal(left[name].detach().cpu(), right[name].detach().cpu()):
                raise RuntimeError(f"previous backbone is not the finalized phase-start state ({name})")

    @torch.no_grad()
    def _current_class_weights(self, loader: Any, *, new_ids: tuple[int, ...]) -> tuple[Tensor, int]:
        counts = {class_id: 0 for class_id in new_ids}
        total = 0
        valid = set(new_ids)
        for batch in loader:
            _, _, labels = self.unpack_batch(batch)
            observed = set(int(v) for v in labels.unique().detach().cpu().tolist())
            outside = observed - valid
            if outside:
                raise RuntimeError(f"incremental TRAIN loader contains non-current classes {sorted(outside)}")
            for class_id in new_ids:
                count = int(labels.eq(class_id).sum().item())
                counts[class_id] += count
                total += count
        if total <= 0 or any(counts[c] <= 0 for c in new_ids):
            raise RuntimeError("incremental TRAIN split is empty or misses a current class")
        class_count = len(new_ids)
        weights = torch.tensor(
            [total / float(class_count * counts[c]) for c in new_ids],
            device=self.device,
            dtype=self.model.dtype,
        )
        return weights, total

    @staticmethod
    def _row_lookup(reference_coords: Tensor) -> dict[tuple[int, int], int]:
        if reference_coords.ndim != 2 or reference_coords.size(1) != 2:
            raise ValueError("reference_coords must be [N,2]")
        lookup: dict[tuple[int, int], int] = {}
        for row, pair in enumerate(reference_coords.detach().cpu().tolist()):
            key = (int(pair[0]), int(pair[1]))
            if key in lookup:
                raise RuntimeError(f"duplicate current TRAIN coordinate {key}")
            lookup[key] = row
        return lookup

    @staticmethod
    def _cached_rows_for_batch(batch_coords: Tensor, *, lookup: dict[tuple[int, int], int], cached: Tensor) -> Tensor:
        rows: list[int] = []
        for pair in batch_coords.tolist():
            key = (int(pair[0]), int(pair[1]))
            if key not in lookup:
                raise RuntimeError(f"current TRAIN batch coordinate {key} is absent from phase-start cache")
            rows.append(lookup[key])
        return cached.index_select(0, torch.tensor(rows, device=cached.device, dtype=torch.long))

    @staticmethod
    def _mean_sample_rms(features: Tensor) -> Tensor:
        value = torch.as_tensor(features)
        if value.ndim != 3 or value.size(0) == 0:
            raise ValueError("features must be non-empty [N,K,D]")
        return value.square().mean(dim=(1, 2)).sqrt().mean()

    @torch.no_grad()
    def _refresh_training_state(
        self,
        *,
        train_eval_loader: Any,
        phase_start_historical: ClassStructureBatch,
        current_reference: ClassStructureBatch,
        phase_affinity: CDSAffinityState,
        reference_q_full: Tensor,
        reference_labels: Tensor,
        reference_spectra: Tensor,
        reference_coords: Tensor,
        current_physical: PhysicalSpectralStructureBatch,
        new_ids: tuple[int, ...],
        seen_ids: tuple[int, ...],
    ) -> CurrentTrainingState:
        current_rows = self.collect_ordered_features(train_eval_loader, require_coordinates=True)
        if not torch.equal(current_rows["labels"], reference_labels):
            raise RuntimeError("current TRAIN labels/order changed within phase")
        if not torch.equal(current_rows["coords"], reference_coords):
            raise RuntimeError("current TRAIN coordinates/order changed within phase")
        if not torch.equal(current_rows["raw_center_spectra"], reference_spectra):
            raise RuntimeError("current TRAIN spectra/order changed within phase")

        current_structure = self.model.estimate_class_structure(
            current_rows["ordered_features"],
            None,
            current_rows["labels"],
            class_ids=new_ids,
            physical_structure=current_physical,
        )
        task_mapped_old, migrated_old, transform, affinity, residual, alignment = align_historical_structure(
            historical=phase_start_historical,
            current_reference=current_reference,
            current_structure=current_structure,
            reference_features=reference_q_full,
            current_features=current_rows["ordered_features"],
            affinity=phase_affinity,
        )
        seen_structure = merge_class_structure_batches((migrated_old, current_structure))
        actual = tuple(int(v) for v in seen_structure.class_ids.detach().cpu().tolist())
        if actual != seen_ids:
            raise RuntimeError("all-seen CDS class order changed")
        seen_cds = self.model.build_class_discriminative_structure(seen_structure)
        return CurrentTrainingState(
            task_mapped_historical=task_mapped_old,
            migrated_historical=migrated_old,
            current=current_structure,
            seen=seen_structure,
            seen_cds=seen_cds,
            current_rows=current_rows,
            transform=transform,
            affinity=affinity,
            residual=residual,
            alignment=alignment,
        )

    def train_incremental_phase(
        self,
        *,
        phase: int,
        epochs: int,
        batch_size: int,
        lr: float,
        reference_backbone: torch.nn.Module,
    ) -> Dict[str, Any]:
        phase_id = _positive_int(phase, name="phase")
        epoch_count = _positive_int(epochs, name="epochs")
        train_batch_size = _positive_int(batch_size, name="batch_size")
        backbone_learning_rate = _finite_positive("lr_inc", lr)
        classifier_learning_rate = _finite_positive("lr", self.cfg("lr", backbone_learning_rate))
        eval_batch_size = _positive_int(self.cfg("eval_batch_size", 256), name="eval_batch_size")
        gradient_clip = _finite_nonnegative("gradient_clip", self.cfg("gradient_clip", 0.0))
        weight_decay = _finite_nonnegative("weight_decay", self.cfg("weight_decay", 1e-4))

        self.model.validate_finalized_state()
        classifier = self._require_classifier()
        self._assert_frozen_reference(reference_backbone)
        self._assert_reference_matches_phase_start(reference_backbone, self.model.backbone)

        previous_phase = phase_id - 1
        previous_seen = tuple(self._validated_class_ids(self.dataset.get_seen_classes(previous_phase), name="previous_seen_classes"))
        old_reporting_loader = self.dataset.get_reporting_dataloader(previous_phase, batch_size=eval_batch_size)
        old_before = self.evaluate_loader(old_reporting_loader, class_ids=previous_seen, target_class_ids=previous_seen)

        self.dataset.start_phase(phase_id)
        old_ids, new_ids, seen_ids = self._phase_classes(phase_id)
        old_ids_tensor = torch.tensor(old_ids, device=self.device, dtype=torch.long)
        new_ids_tensor = torch.tensor(new_ids, device=self.device, dtype=torch.long)
        seen_ids_tensor = torch.tensor(seen_ids, device=self.device, dtype=torch.long)
        phase_start_historical = _clone_structure(self.model.structure_bank.class_structures(old_ids))

        self.model.set_incremental_trainability()
        backbone_trainable = [p for p in self.model.backbone.parameters() if p.requires_grad]
        backbone_frozen = [p for p in self.model.backbone.parameters() if not p.requires_grad]
        if not backbone_trainable:
            raise RuntimeError("incremental backbone has no trainable parameters")

        train_eval_loader = self.dataset.get_phase_dataloader(
            phase_id, split="train", batch_size=eval_batch_size, shuffle=False
        )
        class_weights, expected_samples = self._current_class_weights(train_eval_loader, new_ids=new_ids)

        reference_train = self.collect_ordered_features(
            train_eval_loader, backbone=reference_backbone, require_coordinates=True
        )
        reference_q_full = reference_train["ordered_features"].detach()
        reference_spectra = reference_train["raw_center_spectra"].detach()
        reference_labels = reference_train["labels"].detach()
        reference_coords = reference_train["coords"]
        observed = tuple(sorted(int(v) for v in reference_labels.unique().detach().cpu().tolist()))
        if observed != tuple(sorted(new_ids)):
            raise RuntimeError("incremental TRAIN rows do not match current phase classes")
        current_physical = self.model.estimate_physical_class_structure(
            reference_spectra, reference_labels, class_ids=new_ids
        )
        cache_lookup = self._row_lookup(reference_coords)
        reference_q_rms = self._mean_sample_rms(reference_q_full).detach()

        # The previous tested old rows are a frozen reference for HDC only. They
        # are NOT frozen in the deployed classifier: historical full-CDS
        # supervision is now the only path allowed to move old rows.
        historical_classifier_rows = classifier.class_weight(old_ids_tensor).detach().clone()

        phase_start_current = self.model.estimate_class_structure(
            reference_q_full,
            None,
            reference_labels,
            class_ids=new_ids,
            physical_structure=current_physical,
        )
        phase_start_seen = merge_class_structure_batches((phase_start_historical, phase_start_current))
        phase_affinity = derive_cds_affinity(
            historical=phase_start_historical,
            current_reference=phase_start_current,
        )

        # Do NOT imprint new classifier rows from current I_Q.  Future rows were
        # allocated with the cosine head at model construction and remain untouched
        # during previous phases (masked gradient + zero classifier weight decay).
        # Keeping that neutral initialization matches incremental cosine-head
        # practice: old rows retain their tested state, while new rows acquire
        # class semantics only through real-current all-seen CE plus CDS protection.
        # Imprinting from I_Q creates cross-phase decisions before any optimization
        # and can immediately make new rows win on historical samples.
        phase_start_new_rows = classifier.class_weight(new_ids_tensor).detach().clone()
        if not torch.equal(classifier.class_weight(old_ids_tensor), historical_classifier_rows):
            raise RuntimeError("phase-start setup changed historical classifier rows")
        if not torch.equal(classifier.class_weight(new_ids_tensor), phase_start_new_rows):
            raise RuntimeError("phase-start setup changed neutral current classifier rows")

        phase_start_old_all_seen = self.evaluate_loader(
            old_reporting_loader,
            class_ids=seen_ids,
            target_class_ids=old_ids,
            structures=phase_start_seen,
        )

        print(
            f"[Phase {phase_id} start] PhaseStartOldBA={100*float(old_before['balanced_accuracy']):.2f}% "
            f"OldAllSeen={100*float(phase_start_old_all_seen['balanced_accuracy']):.2f}% | "
            f"Old={list(old_ids)} New={list(new_ids)} Seen={list(seen_ids)}",
            file=sys.stdout,
        )

        backbone_optimizer = torch.optim.AdamW(
            [{"params": backbone_trainable, "weight_decay": weight_decay}],
            lr=backbone_learning_rate,
        )
        classifier_optimizer = torch.optim.AdamW(
            list(classifier.parameters()), lr=classifier_learning_rate, weight_decay=0.0
        )
        backbone_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            backbone_optimizer, T_max=epoch_count, eta_min=0.0
        )
        classifier_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            classifier_optimizer, T_max=epoch_count, eta_min=0.0
        )

        train_loader = self.dataset.get_phase_dataloader(
            phase_id, split="train", batch_size=train_batch_size, shuffle=True
        )
        val_loader = (
            self.dataset.get_phase_dataloader(phase_id, split="val", batch_size=eval_batch_size, shuffle=False)
            if self.dataset.validation_enabled else None
        )

        history: list[dict[str, Any]] = []
        progress = tqdm(
            range(epoch_count),
            desc=f"Phase {phase_id} full-CDS historical supervision",
            unit="epoch",
            dynamic_ncols=True,
            file=sys.stdout,
        )

        final_working: CurrentTrainingState | None = None
        for epoch in progress:
            self.model.eval()
            working = self._refresh_training_state(
                train_eval_loader=train_eval_loader,
                phase_start_historical=phase_start_historical,
                current_reference=phase_start_current,
                phase_affinity=phase_affinity,
                reference_q_full=reference_q_full,
                reference_labels=reference_labels,
                reference_spectra=reference_spectra,
                reference_coords=reference_coords,
                current_physical=current_physical,
                new_ids=new_ids,
                seen_ids=seen_ids,
            )

            self.model.train()
            epoch_lr = float(backbone_optimizer.param_groups[0]["lr"])
            epoch_head_lr = float(classifier_optimizer.param_groups[0]["lr"])
            classification_sum = hdc_sum = hdc_rmse_sum = 0.0
            contrastive_sum = contrastive_gap_sum = contrastive_anchor_fraction_sum = 0.0
            separation_sum = separation_margin_sum = 0.0
            historical_sum = historical_ce_sum = historical_anchor_ce_sum = historical_relation_sum = 0.0
            historical_accuracy_sum = historical_anchor_accuracy_sum = historical_margin_sum = 0.0
            historical_exposure_sum = historical_clearance_sum = 0.0
            historical_pair_weight_sum = historical_learned_risk_sum = historical_spectral_risk_sum = 0.0
            total_sum = 0.0
            correct = sample_count = 0
            per_class_correct = {class_id: 0 for class_id in new_ids}
            per_class_total = {class_id: 0 for class_id in new_ids}

            for batch in train_loader:
                patch, spectrum, labels = self.unpack_batch(batch)
                batch_coords = _batch_coordinates(batch, expected_rows=int(labels.numel()))
                backbone_optimizer.zero_grad(set_to_none=True)
                classifier_optimizer.zero_grad(set_to_none=True)

                q = self.model.extract_features(patch, center_spectrum=spectrum, return_aux=False).ordered_features
                reference_q = self._cached_rows_for_batch(batch_coords, lookup=cache_lookup, cached=reference_q_full)

                # Real current labels are allowed to move only current classifier
                # rows. Historical rows remain active rivals but are detached here.
                all_seen_output = classifier(
                    q,
                    class_ids=seen_ids_tensor,
                    trainable_class_ids=new_ids_tensor,
                )
                target_new_local = _targets_local(labels, new_ids_tensor)
                sample_weights = class_weights.index_select(0, target_new_local)
                classification = classification_objective(
                    logits=all_seen_output.logits,
                    class_ids=all_seen_output.class_ids,
                    labels_global=labels,
                    sample_weights=sample_weights,
                )

                hdc = historical_decision_consistency_loss(
                    reference_features=reference_q,
                    current_features=q,
                    historical_classifier_rows=historical_classifier_rows,
                    labels_global=labels,
                    seen_cds=working.seen_cds,
                    historical_class_count=len(old_ids),
                )

                # Same direct-Q supervised contrast used in Phase 0. Only real
                # current samples participate. The classifier scale is detached,
                # so this path changes Q only and introduces no second head/state.
                contrastive = supervised_contrastive_q_loss(
                    ordered_features=q,
                    labels_global=labels,
                    logit_scale=classifier.logit_scale_value,
                    sample_weights=sample_weights,
                )

                # Feature-side new->old protection. Classifier rows and scale are
                # detached here so this objective changes only the representation.
                separation_cosine = classifier.cosine_similarity(
                    q,
                    class_ids=seen_ids_tensor,
                    trainable_class_ids=(),
                )
                separation_logits = separation_cosine * classifier.logit_scale_value.detach().to(
                    device=q.device, dtype=q.dtype
                )
                separation = hsi_cds_separation_loss(
                    logits=separation_logits,
                    class_ids=seen_ids_tensor,
                    labels_global=labels,
                    seen_cds=working.seen_cds,
                    historical_class_count=len(old_ids),
                    sample_weights=sample_weights,
                )

                # Old->seen direct classifier supervision. The aligned historical
                # I_Q provides classifier-shaped anchors, while COMPLETE intra/inter
                # CDS {I_Q,V_Q,I_P,V_P} derives the pairwise risk that controls old
                # vs old/new competition. This objective has no backbone path.
                historical = historical_full_cds_supervision_loss(
                    classifier=classifier,
                    seen_cds=working.seen_cds,
                    historical_class_count=len(old_ids),
                    historical_classifier_rows=historical_classifier_rows,
                )

                loss = (
                    classification.loss
                    + contrastive.loss
                    + hdc.loss
                    + separation.loss
                    + historical.loss
                )
                if not bool(torch.isfinite(loss)):
                    raise RuntimeError(
                        "incremental CE + direct-Q contrast + HDC + CDS separation + historical full-CDS supervision became NaN or Inf"
                    )
                loss.backward()

                if any(parameter.grad is not None for parameter in reference_backbone.parameters()):
                    raise RuntimeError("previous backbone reference accumulated gradients")
                if any(parameter.grad is not None for parameter in backbone_frozen):
                    raise RuntimeError("a frozen incremental backbone parameter accumulated gradients")
                for parameter in backbone_trainable:
                    if parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all()):
                        raise RuntimeError("incremental backbone gradient contains NaN or Inf")
                for parameter in classifier.parameters():
                    if parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all()):
                        raise RuntimeError("incremental classifier gradient contains NaN or Inf")
                if gradient_clip > 0.0:
                    torch.nn.utils.clip_grad_norm_(backbone_trainable, gradient_clip, error_if_nonfinite=True)

                backbone_optimizer.step()
                classifier_optimizer.step()

                n = int(labels.numel())
                classification_sum += float(classification.loss.detach().item()) * n
                hdc_sum += float(hdc.loss.detach().item()) * n
                hdc_rmse_sum += float(hdc.relation_rmse.detach().item()) * n
                contrastive_sum += float(contrastive.loss.detach().item()) * n
                contrastive_gap_sum += float(contrastive.similarity_gap.detach().item()) * n
                contrastive_anchor_fraction_sum += float(contrastive.valid_anchor_fraction.detach().item()) * n
                separation_sum += float(separation.loss.detach().item()) * n
                separation_margin_sum += float(separation.margin_mean.detach().item()) * n
                historical_sum += float(historical.loss.detach().item()) * n
                historical_ce_sum += float(historical.positive_ce.detach().item()) * n
                historical_anchor_ce_sum += float(historical.decision_anchor_ce.detach().item()) * n
                historical_relation_sum += float(historical.relation_loss.detach().item()) * n
                historical_accuracy_sum += float(historical.identity_accuracy.detach().item()) * n
                historical_anchor_accuracy_sum += float(historical.decision_anchor_accuracy.detach().item()) * n
                historical_margin_sum += float(historical.identity_margin_mean.detach().item()) * n
                historical_exposure_sum += float(historical.deformation_exposure_mean.detach().item()) * n
                historical_clearance_sum += float(historical.structural_clearance_mean.detach().item()) * n
                historical_pair_weight_sum += float(historical.pair_weight_mean.detach().item()) * n
                historical_learned_risk_sum += float(historical.learned_region_risk_mean.detach().item()) * n
                historical_spectral_risk_sum += float(historical.spectral_region_risk_mean.detach().item()) * n
                total_sum += float(loss.detach().item()) * n

                prediction = all_seen_output.prediction.detach()
                correct += int(prediction.eq(labels).sum().item())
                sample_count += n
                for class_id in new_ids:
                    mask = labels.eq(class_id)
                    count = int(mask.sum().item())
                    per_class_total[class_id] += count
                    if count:
                        per_class_correct[class_id] += int(prediction[mask].eq(labels[mask]).sum().item())

            if sample_count != expected_samples:
                raise RuntimeError("incremental TRAIN loader did not traverse the complete current split")

            backbone_scheduler.step()
            classifier_scheduler.step()

            self.model.eval()
            eval_working = self._refresh_training_state(
                train_eval_loader=train_eval_loader,
                phase_start_historical=phase_start_historical,
                current_reference=phase_start_current,
                phase_affinity=phase_affinity,
                reference_q_full=reference_q_full,
                reference_labels=reference_labels,
                reference_spectra=reference_spectra,
                reference_coords=reference_coords,
                current_physical=current_physical,
                new_ids=new_ids,
                seen_ids=seen_ids,
            )
            final_working = eval_working

            new_train_all_seen = self.evaluate_loader(
                train_eval_loader, class_ids=seen_ids, target_class_ids=new_ids, structures=eval_working.seen
            )
            new_train_only = self.evaluate_loader(
                train_eval_loader, class_ids=new_ids, target_class_ids=new_ids, structures=eval_working.current
            )
            validation_all_seen = None if val_loader is None else self.evaluate_loader(
                val_loader, class_ids=seen_ids, target_class_ids=new_ids, structures=eval_working.seen
            )
            validation_new_only = None if val_loader is None else self.evaluate_loader(
                val_loader, class_ids=new_ids, target_class_ids=new_ids, structures=eval_working.current
            )

            online_per_class = {cid: per_class_correct[cid] / float(per_class_total[cid]) for cid in new_ids}
            diag = eval_working.alignment
            record = {
                "epoch": epoch + 1,
                "learning_rate": epoch_lr,
                "classifier_learning_rate": epoch_head_lr,
                "classification_loss": classification_sum / sample_count,
                "historical_decision_consistency_loss": hdc_sum / sample_count,
                "historical_decision_relation_rmse": hdc_rmse_sum / sample_count,
                "contrastive_loss": contrastive_sum / sample_count,
                "contrastive_similarity_gap": contrastive_gap_sum / sample_count,
                "contrastive_valid_anchor_fraction": contrastive_anchor_fraction_sum / sample_count,
                "hsi_cds_separation_loss": separation_sum / sample_count,
                "hsi_cds_separation_margin_mean": separation_margin_sum / sample_count,
                "historical_full_cds_loss": historical_sum / sample_count,
                "historical_full_cds_positive_ce": historical_ce_sum / sample_count,
                "historical_decision_anchor_ce": historical_anchor_ce_sum / sample_count,
                "historical_full_cds_relation_loss": historical_relation_sum / sample_count,
                "historical_identity_accuracy": historical_accuracy_sum / sample_count,
                "historical_decision_anchor_accuracy": historical_anchor_accuracy_sum / sample_count,
                "historical_identity_margin_mean": historical_margin_sum / sample_count,
                "historical_deformation_exposure_mean": historical_exposure_sum / sample_count,
                "historical_structural_clearance_mean": historical_clearance_sum / sample_count,
                "historical_pair_weight_mean": historical_pair_weight_sum / sample_count,
                "historical_learned_risk_mean": historical_learned_risk_sum / sample_count,
                "historical_spectral_risk_mean": historical_spectral_risk_sum / sample_count,
                "total_loss": total_sum / sample_count,
                "new_train_oa_online": correct / float(sample_count),
                "new_train_ba_online": sum(online_per_class.values()) / float(len(online_per_class)),
                "new_train_min_online": min(online_per_class.values()),
                "new_train_ba_all_seen": float(new_train_all_seen["balanced_accuracy"]),
                "new_train_ba_new_only": float(new_train_only["balanced_accuracy"]),
                "validation_new_ba_all_seen": None if validation_all_seen is None else float(validation_all_seen["balanced_accuracy"]),
                "validation_new_ba_new_only": None if validation_new_only is None else float(validation_new_only["balanced_accuracy"]),
                "semantic_shift_mapping_error": float(diag.task_mapping_error.item()),
                "feature_drift_rms": float(diag.representation_drift_rms.item()),
                "reference_q_rms": float(reference_q_rms.item()),
                "current_q_rms": float(self._mean_sample_rms(eval_working.current_rows["ordered_features"]).item()),
                "classifier_logit_scale": float(classifier.logit_scale_value.detach().item()),
            }
            history.append(record)

            message = (
                f"[Epoch {epoch + 1:03d}/{epoch_count:03d}] "
                f"CE={record['classification_loss']:.4f} "
                f"Con={record['contrastive_loss']:.4f} ConGap={record['contrastive_similarity_gap']:.4f} "
                f"HDC={record['historical_decision_consistency_loss']:.4f} "
                f"Sep={record['hsi_cds_separation_loss']:.4f} HistCDS={record['historical_full_cds_loss']:.4f} "
                f"HistAcc={100*record['historical_identity_accuracy']:.2f}% "
                f"AnchorAcc={100*record['historical_decision_anchor_accuracy']:.2f}% "
                f"HistClr={record['historical_structural_clearance_mean']:.4f} "
                f"LR={record['learning_rate']:.2e} HeadLR={record['classifier_learning_rate']:.2e} | "
                f"NewOnlyBA={100*record['new_train_ba_new_only']:.2f}% "
                f"AllSeenBA={100*record['new_train_ba_all_seen']:.2f}% "
                f"MapErr={record['semantic_shift_mapping_error']:.4f}"
            )
            if validation_all_seen is not None:
                message += (
                    f" | ValNewOnlyBA={100*record['validation_new_ba_new_only']:.2f}%"
                    f" ValAllSeenBA={100*record['validation_new_ba_all_seen']:.2f}%"
                )
            tqdm.write(message, file=sys.stdout)

        progress.close()
        if final_working is None:
            raise RuntimeError("incremental phase produced no working CDS")

        self.model.eval()
        final_state = self._refresh_training_state(
            train_eval_loader=train_eval_loader,
            phase_start_historical=phase_start_historical,
            current_reference=phase_start_current,
            phase_affinity=phase_affinity,
            reference_q_full=reference_q_full,
            reference_labels=reference_labels,
            reference_spectra=reference_spectra,
            reference_coords=reference_coords,
            current_physical=current_physical,
            new_ids=new_ids,
            seen_ids=seen_ids,
        )

        final_old_only = self.evaluate_loader(
            old_reporting_loader, class_ids=old_ids, target_class_ids=old_ids, structures=final_state.migrated_historical
        )
        final_old_all_seen = self.evaluate_loader(
            old_reporting_loader, class_ids=seen_ids, target_class_ids=old_ids, structures=final_state.seen
        )
        final_new_all_seen = self.evaluate_loader(
            train_eval_loader, class_ids=seen_ids, target_class_ids=new_ids, structures=final_state.seen
        )
        final_new_only = self.evaluate_loader(
            train_eval_loader, class_ids=new_ids, target_class_ids=new_ids, structures=final_state.current
        )

        with torch.no_grad():
            final_historical_objective = historical_full_cds_supervision_loss(
                classifier=classifier,
                seen_cds=final_state.seen_cds,
                historical_class_count=len(old_ids),
                historical_classifier_rows=historical_classifier_rows,
            )

        self.model.replace_class_structure(final_state.seen)
        self.model.validate_finalized_state()
        self.dataset.finalize_phase(phase_id)

        cumulative_loader = self.dataset.get_reporting_dataloader(phase_id, batch_size=eval_batch_size)
        cumulative = self.evaluate_loader(cumulative_loader, class_ids=seen_ids, target_class_ids=seen_ids)
        groups = old_new_summary(cumulative, old_class_ids=old_ids, new_class_ids=new_ids)
        forgetting = float(old_before["balanced_accuracy"]) - float(groups["old"]["balanced_accuracy"])
        phase_origin = phase_origin_confusion_diagnostics(
            cumulative,
            phase_to_classes={index: self.dataset.phase_to_classes[index] for index in range(phase_id + 1)},
        )

        diag = final_state.alignment
        method_state = {
            "classifier": "single_bias_free_scaled_cosine_head",
            "new_classifier_row_initialization": "neutral_preallocated_cosine_rows; no_I_Q_imprinting",
            "persistent_cds": "{I_Q,V_Q,I_P,V_P,N}; all inter-class CDS is derived on demand",
            "historical_supervision": "aligned_old_I_Q_plus_phase_start_classifier_decision_anchors_plus_V_Q_region_clearance_over_all_seen_boundaries_with_I_P_V_P_risk_priority",
            "historical_cds_information": "intra={I_Q,V_Q,I_P,V_P}; inter={R_IQ,R_VQ,R_IP,R_VP}; physical_spectral_information_is_active_in_classifier_risk",
            "representation_objective": "real_current_CE_plus_same_direct_Q_supervised_contrast_as_base_plus_HDC_plus_HSI_CDS_separation",
            "stores_pseudo_features": False,
            "reconstructs_historical_features": False,
            "uses_input_inversion": False,
            "uses_stage_B": False,
            "uses_covariance_classifier": False,
            "semantic_shift_mapping_error": float(diag.task_mapping_error.item()),
            "semantic_shift_channel_error_max": float(diag.task_mapping_channel_error_max.item()),
            "feature_drift_rms": float(diag.representation_drift_rms.item()),
            "task_translation_rms": float(diag.task_translation_rms.item()),
            "task_scale_deviation_rms": float(diag.task_scale_deviation_rms.item()),
            "reference_q_rms": float(reference_q_rms.item()),
            "current_q_rms": float(self._mean_sample_rms(final_state.current_rows["ordered_features"]).item()),
            "phase_start_old_ba": float(old_before["balanced_accuracy"]),
            "phase_start_old_all_seen_ba": float(phase_start_old_all_seen["balanced_accuracy"]),
            "final_old_only_ba": float(final_old_only["balanced_accuracy"]),
            "final_old_all_seen_ba": float(final_old_all_seen["balanced_accuracy"]),
            "representation_forgetting": float(old_before["balanced_accuracy"]) - float(final_old_only["balanced_accuracy"]),
            "decision_interference_start": float(old_before["balanced_accuracy"]) - float(phase_start_old_all_seen["balanced_accuracy"]),
            "decision_interference_final": float(final_old_only["balanced_accuracy"]) - float(final_old_all_seen["balanced_accuracy"]),
            "final_new_train_all_seen_ba": float(final_new_all_seen["balanced_accuracy"]),
            "final_new_train_only_ba": float(final_new_only["balanced_accuracy"]),
            "classifier_logit_scale": float(classifier.logit_scale_value.detach().item()),
            "final_classification_loss": float(history[-1]["classification_loss"]),
            "final_historical_decision_consistency_loss": float(history[-1]["historical_decision_consistency_loss"]),
            "final_contrastive_loss": float(history[-1]["contrastive_loss"]),
            "final_contrastive_similarity_gap": float(history[-1]["contrastive_similarity_gap"]),
            "final_contrastive_valid_anchor_fraction": float(history[-1]["contrastive_valid_anchor_fraction"]),
            "final_hsi_cds_separation_loss": float(history[-1]["hsi_cds_separation_loss"]),
            "final_historical_full_cds_loss": float(final_historical_objective.loss.item()),
            "final_historical_full_cds_positive_ce": float(final_historical_objective.positive_ce.item()),
            "final_historical_decision_anchor_ce": float(final_historical_objective.decision_anchor_ce.item()),
            "final_historical_full_cds_relation_loss": float(final_historical_objective.relation_loss.item()),
            "final_historical_identity_accuracy": float(final_historical_objective.identity_accuracy.item()),
            "final_historical_decision_anchor_accuracy": float(final_historical_objective.decision_anchor_accuracy.item()),
            "final_historical_identity_margin_mean": float(final_historical_objective.identity_margin_mean.item()),
            "final_historical_identity_margin_min": float(final_historical_objective.identity_margin_min.item()),
            "final_historical_deformation_exposure_mean": float(final_historical_objective.deformation_exposure_mean.item()),
            "final_historical_structural_clearance_mean": float(final_historical_objective.structural_clearance_mean.item()),
            "final_historical_structural_clearance_min": float(final_historical_objective.structural_clearance_min.item()),
            "final_historical_pair_weight_mean": float(final_historical_objective.pair_weight_mean.item()),
            "final_historical_learned_risk_mean": float(final_historical_objective.learned_region_risk_mean.item()),
            "final_historical_spectral_risk_mean": float(final_historical_objective.spectral_region_risk_mean.item()),
        }

        return {
            "phase": phase_id,
            "old_class_ids": list(old_ids),
            "new_class_ids": list(new_ids),
            "seen_class_ids": list(seen_ids),
            "history": history,
            "final_epoch": epoch_count,
            "cumulative_test": cumulative,
            "old_new": groups,
            "forgetting": forgetting,
            "method_state": method_state,
            "phase_origin_confusion": phase_origin,
            "structure_committed": True,
            "phase_finalized": True,
        }


__all__ = ["IncrementalPhaseTrainer"]










# from __future__ import annotations

# """Incremental NECIL-HSI training with one Q, one all-seen CDS and one cosine head.

# Current real HSI supplies CE and direct-Q supervised contrast. HDC stabilizes the
# representation against the frozen previous backbone/old classifier rows. A
# feature-only old-rival separation term keeps current Q away from historical
# decisions. The complete all-seen CDS then protects the SAME classifier: I_Q,V_Q
# define historical decision regions and I_P,V_P prioritize spectrally confusable
# historical/seen boundaries. Current CE keeps historical rows detached, while the
# whole migrated CDS is the only objective allowed to realign historical classifier rows.
# """

# from dataclasses import dataclass
# import math
# import sys
# from numbers import Integral, Real
# from typing import Any, Dict, Mapping

# import torch
# from tqdm import tqdm

# from losses.loss import (
#     classification_objective,
#     supervised_contrastive_q_loss,
#     historical_decision_consistency_loss,
#     hsi_cds_separation_loss,
# )
# from models.cds_decision_region import HSICDSDecisionRegionRegulator
# from models.classifier import SpectralCDSClassifier
# from models.geometry_bank import (
#     ClassDiscriminativeStructureState,
#     PhysicalSpectralStructureBatch,
#     ClassStructureBatch,
#     merge_class_structure_batches,
# )
# from models.phase_coordinate_alignment import (
#     CDSAffinityState,
#     CurrentClassResidual,
#     PhaseAlignmentDiagnostics,
#     PhaseCoordinateTransform,
#     align_historical_structure,
#     derive_cds_affinity,
# )
# from utils.eval import old_new_summary, phase_origin_confusion_diagnostics

# Tensor = torch.Tensor


# def _positive_int(value: object, *, name: str) -> int:
#     if isinstance(value, bool) or not isinstance(value, (Integral, Real)):
#         raise ValueError(f"{name} must be a positive integer")
#     number = float(value)
#     if not math.isfinite(number) or not number.is_integer() or number <= 0:
#         raise ValueError(f"{name} must be a positive integer")
#     return int(number)


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


# def _clone_structure(batch: ClassStructureBatch) -> ClassStructureBatch:
#     if not isinstance(batch, ClassStructureBatch) or batch.class_count <= 0:
#         raise ValueError("cannot clone an empty class structure")
#     return ClassStructureBatch(
#         class_ids=batch.class_ids.detach().clone(),
#         sample_counts=batch.sample_counts.detach().clone(),
#         identity=batch.identity.detach().clone(),
#         variation=batch.variation.detach().clone(),
#         spectral_identity=batch.spectral_identity.detach().clone(),
#         spectral_variation=batch.spectral_variation.detach().clone(),
#     )


# def _targets_local(labels: Tensor, class_ids: Tensor) -> Tensor:
#     value = torch.as_tensor(labels)
#     ids = torch.as_tensor(class_ids)
#     if value.ndim != 1 or value.numel() == 0 or value.dtype != torch.long:
#         raise ValueError("labels must be non-empty rank-one torch.long")
#     if ids.ndim != 1 or ids.numel() < 1 or ids.dtype != torch.long:
#         raise ValueError("class_ids must be non-empty rank-one torch.long")
#     if value.device != ids.device:
#         raise ValueError("labels and class_ids must share device")
#     matches = value.unsqueeze(1).eq(ids.unsqueeze(0))
#     if not bool(matches.sum(dim=1).eq(1).all()):
#         raise ValueError("labels contain a class absent from class_ids")
#     return matches.to(torch.long).argmax(dim=1)


# def _batch_coordinates(batch: Mapping[str, Any], *, expected_rows: int) -> Tensor:
#     if "coord" not in batch:
#         raise RuntimeError("incremental TRAIN batches must expose coord for phase-start reference lookup")
#     coords = torch.as_tensor(batch["coord"])
#     if coords.shape != (expected_rows, 2):
#         raise RuntimeError("coord must be [N,2] and batch-aligned")
#     if coords.dtype == torch.bool or coords.is_complex() or torch.is_floating_point(coords):
#         raise RuntimeError("coord must use an integer dtype")
#     return coords.to(dtype=torch.long, device="cpu")


# @dataclass(frozen=True)
# class CurrentTrainingState:
#     task_mapped_historical: ClassStructureBatch
#     migrated_historical: ClassStructureBatch
#     current: ClassStructureBatch
#     seen: ClassStructureBatch
#     seen_cds: ClassDiscriminativeStructureState
#     current_rows: Dict[str, Tensor]
#     transform: PhaseCoordinateTransform
#     affinity: CDSAffinityState
#     residual: CurrentClassResidual
#     alignment: PhaseAlignmentDiagnostics


# class IncrementalPhaseTrainer:
#     INCREMENTAL_RESULT_FIELDS = frozenset(
#         {
#             "phase", "old_class_ids", "new_class_ids", "seen_class_ids", "history",
#             "final_epoch", "cumulative_test", "old_new", "forgetting", "method_state",
#             "structure_committed", "phase_finalized",
#         }
#     )

#     def _require_classifier(self) -> SpectralCDSClassifier:
#         classifier = getattr(self.model, "classifier", None)
#         if not isinstance(classifier, SpectralCDSClassifier) or classifier.parameter_count <= 0:
#             raise RuntimeError("model.classifier must be the trainable cosine SpectralCDSClassifier")
#         return classifier

#     def _phase_classes(self, phase: int) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
#         old_ids = tuple(self._validated_class_ids(self.dataset.get_old_classes(phase), name=f"phase_{phase}_old_classes"))
#         new_ids = tuple(self._validated_class_ids(self.dataset.get_new_classes(phase), name=f"phase_{phase}_new_classes"))
#         seen_ids = tuple(self._validated_class_ids(self.dataset.get_seen_classes(phase), name=f"phase_{phase}_seen_classes"))
#         if not old_ids or not new_ids or set(old_ids).intersection(new_ids):
#             raise RuntimeError("incremental phase has invalid old/new class partition")
#         if seen_ids != old_ids + new_ids:
#             raise RuntimeError("seen classes must equal old classes followed by new classes")
#         if tuple(self.model.committed_class_ids) != old_ids:
#             raise RuntimeError("historical CDS does not match the old class set")
#         return old_ids, new_ids, seen_ids

#     @staticmethod
#     def _assert_frozen_reference(reference_backbone: torch.nn.Module) -> None:
#         reference_backbone.eval()
#         if any(parameter.requires_grad for parameter in reference_backbone.parameters()):
#             raise RuntimeError("previous backbone reference must remain fully frozen")
#         if any(parameter.grad is not None for parameter in reference_backbone.parameters()):
#             raise RuntimeError("previous backbone reference must not carry gradients")

#     @staticmethod
#     @torch.no_grad()
#     def _assert_reference_matches_phase_start(reference_backbone: torch.nn.Module, current_backbone: torch.nn.Module) -> None:
#         left, right = reference_backbone.state_dict(), current_backbone.state_dict()
#         if tuple(left.keys()) != tuple(right.keys()):
#             raise RuntimeError("previous/current backbone schemas differ at phase start")
#         for name in left:
#             if left[name].shape != right[name].shape or left[name].dtype != right[name].dtype:
#                 raise RuntimeError(f"phase-start backbone state differs at {name}")
#             if not torch.equal(left[name].detach().cpu(), right[name].detach().cpu()):
#                 raise RuntimeError(f"previous backbone is not the finalized phase-start state ({name})")

#     @torch.no_grad()
#     def _current_class_weights(self, loader: Any, *, new_ids: tuple[int, ...]) -> tuple[Tensor, int]:
#         counts = {class_id: 0 for class_id in new_ids}
#         total = 0
#         valid = set(new_ids)
#         for batch in loader:
#             _, _, labels = self.unpack_batch(batch)
#             observed = set(int(v) for v in labels.unique().detach().cpu().tolist())
#             outside = observed - valid
#             if outside:
#                 raise RuntimeError(f"incremental TRAIN loader contains non-current classes {sorted(outside)}")
#             for class_id in new_ids:
#                 count = int(labels.eq(class_id).sum().item())
#                 counts[class_id] += count
#                 total += count
#         if total <= 0 or any(counts[c] <= 0 for c in new_ids):
#             raise RuntimeError("incremental TRAIN split is empty or misses a current class")
#         class_count = len(new_ids)
#         weights = torch.tensor(
#             [total / float(class_count * counts[c]) for c in new_ids],
#             device=self.device,
#             dtype=self.model.dtype,
#         )
#         return weights, total

#     @staticmethod
#     def _row_lookup(reference_coords: Tensor) -> dict[tuple[int, int], int]:
#         if reference_coords.ndim != 2 or reference_coords.size(1) != 2:
#             raise ValueError("reference_coords must be [N,2]")
#         lookup: dict[tuple[int, int], int] = {}
#         for row, pair in enumerate(reference_coords.detach().cpu().tolist()):
#             key = (int(pair[0]), int(pair[1]))
#             if key in lookup:
#                 raise RuntimeError(f"duplicate current TRAIN coordinate {key}")
#             lookup[key] = row
#         return lookup

#     @staticmethod
#     def _cached_rows_for_batch(batch_coords: Tensor, *, lookup: dict[tuple[int, int], int], cached: Tensor) -> Tensor:
#         rows: list[int] = []
#         for pair in batch_coords.tolist():
#             key = (int(pair[0]), int(pair[1]))
#             if key not in lookup:
#                 raise RuntimeError(f"current TRAIN batch coordinate {key} is absent from phase-start cache")
#             rows.append(lookup[key])
#         return cached.index_select(0, torch.tensor(rows, device=cached.device, dtype=torch.long))

#     @staticmethod
#     def _mean_sample_rms(features: Tensor) -> Tensor:
#         value = torch.as_tensor(features)
#         if value.ndim != 3 or value.size(0) == 0:
#             raise ValueError("features must be non-empty [N,K,D]")
#         return value.square().mean(dim=(1, 2)).sqrt().mean()

#     @torch.no_grad()
#     def _refresh_training_state(
#         self,
#         *,
#         train_eval_loader: Any,
#         phase_start_historical: ClassStructureBatch,
#         current_reference: ClassStructureBatch,
#         phase_affinity: CDSAffinityState,
#         reference_q_full: Tensor,
#         reference_labels: Tensor,
#         reference_spectra: Tensor,
#         reference_coords: Tensor,
#         current_physical: PhysicalSpectralStructureBatch,
#         new_ids: tuple[int, ...],
#         seen_ids: tuple[int, ...],
#     ) -> CurrentTrainingState:
#         current_rows = self.collect_ordered_features(train_eval_loader, require_coordinates=True)
#         if not torch.equal(current_rows["labels"], reference_labels):
#             raise RuntimeError("current TRAIN labels/order changed within phase")
#         if not torch.equal(current_rows["coords"], reference_coords):
#             raise RuntimeError("current TRAIN coordinates/order changed within phase")
#         if not torch.equal(current_rows["raw_center_spectra"], reference_spectra):
#             raise RuntimeError("current TRAIN spectra/order changed within phase")

#         current_structure = self.model.estimate_class_structure(
#             current_rows["ordered_features"],
#             None,
#             current_rows["labels"],
#             class_ids=new_ids,
#             physical_structure=current_physical,
#         )
#         task_mapped_old, migrated_old, transform, affinity, residual, alignment = align_historical_structure(
#             historical=phase_start_historical,
#             current_reference=current_reference,
#             current_structure=current_structure,
#             reference_features=reference_q_full,
#             current_features=current_rows["ordered_features"],
#             affinity=phase_affinity,
#         )
#         seen_structure = merge_class_structure_batches((migrated_old, current_structure))
#         actual = tuple(int(v) for v in seen_structure.class_ids.detach().cpu().tolist())
#         if actual != seen_ids:
#             raise RuntimeError("all-seen CDS class order changed")
#         seen_cds = self.model.build_class_discriminative_structure(seen_structure)
#         return CurrentTrainingState(
#             task_mapped_historical=task_mapped_old,
#             migrated_historical=migrated_old,
#             current=current_structure,
#             seen=seen_structure,
#             seen_cds=seen_cds,
#             current_rows=current_rows,
#             transform=transform,
#             affinity=affinity,
#             residual=residual,
#             alignment=alignment,
#         )

#     def train_incremental_phase(
#         self,
#         *,
#         phase: int,
#         epochs: int,
#         batch_size: int,
#         lr: float,
#         reference_backbone: torch.nn.Module,
#     ) -> Dict[str, Any]:
#         phase_id = _positive_int(phase, name="phase")
#         epoch_count = _positive_int(epochs, name="epochs")
#         train_batch_size = _positive_int(batch_size, name="batch_size")
#         backbone_learning_rate = _finite_positive("lr_inc", lr)
#         classifier_learning_rate = _finite_positive("lr", self.cfg("lr", backbone_learning_rate))
#         eval_batch_size = _positive_int(self.cfg("eval_batch_size", 256), name="eval_batch_size")
#         gradient_clip = _finite_nonnegative("gradient_clip", self.cfg("gradient_clip", 0.0))
#         weight_decay = _finite_nonnegative("weight_decay", self.cfg("weight_decay", 1e-4))

#         self.model.validate_finalized_state()
#         classifier = self._require_classifier()
#         self._assert_frozen_reference(reference_backbone)
#         self._assert_reference_matches_phase_start(reference_backbone, self.model.backbone)

#         previous_phase = phase_id - 1
#         previous_seen = tuple(self._validated_class_ids(self.dataset.get_seen_classes(previous_phase), name="previous_seen_classes"))
#         old_reporting_loader = self.dataset.get_reporting_dataloader(previous_phase, batch_size=eval_batch_size)
#         old_before = self.evaluate_loader(old_reporting_loader, class_ids=previous_seen, target_class_ids=previous_seen)

#         self.dataset.start_phase(phase_id)
#         old_ids, new_ids, seen_ids = self._phase_classes(phase_id)
#         old_ids_tensor = torch.tensor(old_ids, device=self.device, dtype=torch.long)
#         new_ids_tensor = torch.tensor(new_ids, device=self.device, dtype=torch.long)
#         seen_ids_tensor = torch.tensor(seen_ids, device=self.device, dtype=torch.long)
#         phase_start_historical = _clone_structure(self.model.structure_bank.class_structures(old_ids))

#         self.model.set_incremental_trainability()
#         backbone_trainable = [p for p in self.model.backbone.parameters() if p.requires_grad]
#         backbone_frozen = [p for p in self.model.backbone.parameters() if not p.requires_grad]
#         if not backbone_trainable:
#             raise RuntimeError("incremental backbone has no trainable parameters")

#         structure_loader_factory = getattr(self.dataset, "get_phase_structure_dataloader", None)
#         train_eval_loader = (
#             structure_loader_factory(phase_id, batch_size=eval_batch_size)
#             if callable(structure_loader_factory)
#             else self.dataset.get_phase_dataloader(
#                 phase_id, split="train", batch_size=eval_batch_size, shuffle=False
#             )
#         )
#         class_weights, expected_samples = self._current_class_weights(train_eval_loader, new_ids=new_ids)

#         reference_train = self.collect_ordered_features(
#             train_eval_loader, backbone=reference_backbone, require_coordinates=True
#         )
#         reference_q_full = reference_train["ordered_features"].detach()
#         reference_spectra = reference_train["raw_center_spectra"].detach()
#         reference_labels = reference_train["labels"].detach()
#         reference_coords = reference_train["coords"]
#         observed = tuple(sorted(int(v) for v in reference_labels.unique().detach().cpu().tolist()))
#         if observed != tuple(sorted(new_ids)):
#             raise RuntimeError("incremental TRAIN rows do not match current phase classes")
#         current_physical = self.model.estimate_physical_class_structure(
#             reference_spectra, reference_labels, class_ids=new_ids
#         )
#         cache_lookup = self._row_lookup(reference_coords)
#         reference_q_rms = self._mean_sample_rms(reference_q_full).detach()

#         # Snapshot the previous tested historical rows only as the frozen HDC
#         # teacher. The live historical rows may later realign, but only through
#         # classifier-side supervision from the migrated historical CDS.
#         historical_classifier_rows = classifier.class_weight(old_ids_tensor).detach().clone()

#         phase_start_current = self.model.estimate_class_structure(
#             reference_q_full,
#             None,
#             reference_labels,
#             class_ids=new_ids,
#             physical_structure=current_physical,
#         )
#         phase_start_seen = merge_class_structure_batches((phase_start_historical, phase_start_current))
#         phase_affinity = derive_cds_affinity(
#             historical=phase_start_historical,
#             current_reference=phase_start_current,
#         )

#         # New classifier rows are NOT initialized from I_Q or any CDS center.
#         # They retain their deterministic model initialization and are learned
#         # from real current samples. The complete CDS constrains them through
#         # whole-CDS contrast and boundary clearance; historical rows stay fixed.
#         if not torch.equal(classifier.class_weight(old_ids_tensor), historical_classifier_rows):
#             raise RuntimeError("historical classifier rows changed before incremental optimization")

#         phase_start_old_all_seen = self.evaluate_loader(
#             old_reporting_loader,
#             class_ids=seen_ids,
#             target_class_ids=old_ids,
#             structures=phase_start_seen,
#         )

#         print(
#             f"[Phase {phase_id} start] PhaseStartOldBA={100*float(old_before['balanced_accuracy']):.2f}% "
#             f"OldAllSeen={100*float(phase_start_old_all_seen['balanced_accuracy']):.2f}% | "
#             f"Old={list(old_ids)} New={list(new_ids)} Seen={list(seen_ids)}",
#             file=sys.stdout,
#         )

#         backbone_optimizer = torch.optim.AdamW(
#             [{"params": backbone_trainable, "weight_decay": weight_decay}],
#             lr=backbone_learning_rate,
#         )
#         classifier_optimizer = torch.optim.AdamW(
#             list(classifier.parameters()), lr=classifier_learning_rate, weight_decay=0.0
#         )
#         backbone_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
#             backbone_optimizer, T_max=epoch_count, eta_min=0.0
#         )
#         classifier_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
#             classifier_optimizer, T_max=epoch_count, eta_min=0.0
#         )

#         train_loader = self.dataset.get_phase_dataloader(
#             phase_id, split="train", batch_size=train_batch_size, shuffle=True
#         )
#         val_loader = (
#             self.dataset.get_phase_dataloader(phase_id, split="val", batch_size=eval_batch_size, shuffle=False)
#             if self.dataset.validation_enabled else None
#         )
#         cds_regulator = HSICDSDecisionRegionRegulator().to(self.device)
#         if cds_regulator.parameter_count != 0:
#             raise RuntimeError("CDS decision-region regulator must remain parameter-free")

#         history: list[dict[str, Any]] = []
#         progress = tqdm(
#             range(epoch_count),
#             desc=f"Phase {phase_id} all-seen CDS continual learning",
#             unit="epoch",
#             dynamic_ncols=True,
#             file=sys.stdout,
#         )

#         final_working: CurrentTrainingState | None = None
#         for epoch in progress:
#             self.model.eval()
#             working = self._refresh_training_state(
#                 train_eval_loader=train_eval_loader,
#                 phase_start_historical=phase_start_historical,
#                 current_reference=phase_start_current,
#                 phase_affinity=phase_affinity,
#                 reference_q_full=reference_q_full,
#                 reference_labels=reference_labels,
#                 reference_spectra=reference_spectra,
#                 reference_coords=reference_coords,
#                 current_physical=current_physical,
#                 new_ids=new_ids,
#                 seen_ids=seen_ids,
#             )

#             self.model.train()
#             epoch_lr = float(backbone_optimizer.param_groups[0]["lr"])
#             epoch_head_lr = float(classifier_optimizer.param_groups[0]["lr"])
#             classification_sum = hdc_sum = hdc_rmse_sum = 0.0
#             contrastive_sum = contrastive_gap_sum = contrastive_anchor_fraction_sum = 0.0
#             separation_sum = separation_margin_sum = 0.0
#             boundary_sum = boundary_clearance_sum = 0.0
#             boundary_clearance_min_value = float("inf")
#             boundary_absolute_clearance_sum = boundary_relational_clearance_sum = 0.0
#             boundary_unsafe_sum = 0.0
#             boundary_pair_weight_sum = boundary_learned_risk_sum = boundary_spectral_risk_sum = 0.0
#             total_sum = 0.0
#             correct = sample_count = 0
#             per_class_correct = {class_id: 0 for class_id in new_ids}
#             per_class_total = {class_id: 0 for class_id in new_ids}

#             for batch in train_loader:
#                 patch, spectrum, labels = self.unpack_batch(batch)
#                 batch_coords = _batch_coordinates(batch, expected_rows=int(labels.numel()))
#                 backbone_optimizer.zero_grad(set_to_none=True)
#                 classifier_optimizer.zero_grad(set_to_none=True)

#                 q = self.model.extract_features(patch, center_spectrum=spectrum, return_aux=False).ordered_features
#                 reference_q = self._cached_rows_for_batch(batch_coords, lookup=cache_lookup, cached=reference_q_full)

#                 # Real current labels are allowed to move only current classifier
#                 # rows. Historical rows remain active rivals but are detached here.
#                 all_seen_output = classifier(
#                     q,
#                     class_ids=seen_ids_tensor,
#                     trainable_class_ids=new_ids_tensor,
#                 )
#                 target_new_local = _targets_local(labels, new_ids_tensor)
#                 sample_weights = class_weights.index_select(0, target_new_local)
#                 classification = classification_objective(
#                     logits=all_seen_output.logits,
#                     class_ids=all_seen_output.class_ids,
#                     labels_global=labels,
#                     sample_weights=sample_weights,
#                 )

#                 hdc = historical_decision_consistency_loss(
#                     reference_features=reference_q,
#                     current_features=q,
#                     historical_classifier_rows=historical_classifier_rows,
#                     labels_global=labels,
#                     seen_cds=working.seen_cds,
#                     historical_class_count=len(old_ids),
#                 )

#                 # Same direct-Q supervised contrast used in Phase 0. Only real
#                 # current samples participate; no CDS center, pseudo feature, or
#                 # second representation is introduced.
#                 contrastive = supervised_contrastive_q_loss(
#                     ordered_features=q,
#                     labels_global=labels,
#                     logit_scale=classifier.logit_scale_value,
#                     sample_weights=sample_weights,
#                 )

#                 # Feature-side new->old separation. Classifier rows and scale are
#                 # detached, so this objective can move Q but cannot move W. The
#                 # complete all-seen CDS only prioritizes old/current conflicts.
#                 separation_cosine = classifier.cosine_similarity(
#                     q,
#                     class_ids=seen_ids_tensor,
#                     trainable_class_ids=(),
#                 )
#                 separation_logits = separation_cosine * classifier.logit_scale_value.detach().to(
#                     device=q.device, dtype=q.dtype
#                 )
#                 separation = hsi_cds_separation_loss(
#                     logits=separation_logits,
#                     class_ids=seen_ids_tensor,
#                     labels_global=labels,
#                     seen_cds=working.seen_cds,
#                     historical_class_count=len(old_ids),
#                     sample_weights=sample_weights,
#                 )

#                 # Classifier-side historical-region protection. Migrated I_Q locates
#                 # each historical region, diagonal/off-diagonal V_Q bound its
#                 # legitimate extent, and I_P/V_P prioritize historical/seen
#                 # conflicts. This is the only loss allowed to realign old rows.
#                 boundary = cds_regulator(
#                     classifier=classifier,
#                     seen_cds=working.seen_cds,
#                     historical_class_count=len(old_ids),
#                 )

#                 loss = (
#                     classification.loss
#                     + contrastive.loss
#                     + hdc.loss
#                     + separation.loss
#                     + boundary.loss
#                 )
#                 if not bool(torch.isfinite(loss)):
#                     raise RuntimeError(
#                         "incremental CE + direct-Q contrast + HDC + CDS separation + classifier protection became NaN or Inf"
#                     )
#                 loss.backward()

#                 if any(parameter.grad is not None for parameter in reference_backbone.parameters()):
#                     raise RuntimeError("previous backbone reference accumulated gradients")
#                 if any(parameter.grad is not None for parameter in backbone_frozen):
#                     raise RuntimeError("a frozen incremental backbone parameter accumulated gradients")
#                 for parameter in backbone_trainable:
#                     if parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all()):
#                         raise RuntimeError("incremental backbone gradient contains NaN or Inf")
#                 for parameter in classifier.parameters():
#                     if parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all()):
#                         raise RuntimeError("incremental classifier gradient contains NaN or Inf")
#                 if gradient_clip > 0.0:
#                     torch.nn.utils.clip_grad_norm_(backbone_trainable, gradient_clip, error_if_nonfinite=True)

#                 backbone_optimizer.step()
#                 classifier_optimizer.step()

#                 n = int(labels.numel())
#                 classification_sum += float(classification.loss.detach().item()) * n
#                 hdc_sum += float(hdc.loss.detach().item()) * n
#                 hdc_rmse_sum += float(hdc.relation_rmse.detach().item()) * n
#                 contrastive_sum += float(contrastive.loss.detach().item()) * n
#                 contrastive_gap_sum += float(contrastive.similarity_gap.detach().item()) * n
#                 contrastive_anchor_fraction_sum += float(contrastive.valid_anchor_fraction.detach().item()) * n
#                 separation_sum += float(separation.loss.detach().item()) * n
#                 separation_margin_sum += float(separation.margin_mean.detach().item()) * n
#                 boundary_sum += float(boundary.loss.detach().item()) * n
#                 boundary_clearance_sum += float(boundary.structural_clearance_mean.detach().item()) * n
#                 boundary_clearance_min_value = min(
#                     boundary_clearance_min_value,
#                     float(boundary.structural_clearance_min.detach().item()),
#                 )
#                 boundary_absolute_clearance_sum += float(boundary.absolute_clearance_mean.detach().item()) * n
#                 boundary_relational_clearance_sum += float(boundary.relational_clearance_mean.detach().item()) * n
#                 boundary_unsafe_sum += float(boundary.unsafe_pair_fraction.detach().item()) * n
#                 boundary_pair_weight_sum += float(boundary.pair_weight_mean.detach().item()) * n
#                 boundary_learned_risk_sum += float(boundary.learned_region_risk_mean.detach().item()) * n
#                 boundary_spectral_risk_sum += float(boundary.spectral_region_risk_mean.detach().item()) * n
#                 total_sum += float(loss.detach().item()) * n

#                 prediction = all_seen_output.prediction.detach()
#                 correct += int(prediction.eq(labels).sum().item())
#                 sample_count += n
#                 for class_id in new_ids:
#                     mask = labels.eq(class_id)
#                     count = int(mask.sum().item())
#                     per_class_total[class_id] += count
#                     if count:
#                         per_class_correct[class_id] += int(prediction[mask].eq(labels[mask]).sum().item())

#             if sample_count != expected_samples:
#                 raise RuntimeError("incremental TRAIN loader did not traverse the complete current split")

#             backbone_scheduler.step()
#             classifier_scheduler.step()

#             self.model.eval()
#             eval_working = self._refresh_training_state(
#                 train_eval_loader=train_eval_loader,
#                 phase_start_historical=phase_start_historical,
#                 current_reference=phase_start_current,
#                 phase_affinity=phase_affinity,
#                 reference_q_full=reference_q_full,
#                 reference_labels=reference_labels,
#                 reference_spectra=reference_spectra,
#                 reference_coords=reference_coords,
#                 current_physical=current_physical,
#                 new_ids=new_ids,
#                 seen_ids=seen_ids,
#             )
#             final_working = eval_working

#             new_train_all_seen = self.evaluate_loader(
#                 train_eval_loader, class_ids=seen_ids, target_class_ids=new_ids, structures=eval_working.seen
#             )
#             new_train_only = self.evaluate_loader(
#                 train_eval_loader, class_ids=new_ids, target_class_ids=new_ids, structures=eval_working.current
#             )
#             validation_all_seen = None if val_loader is None else self.evaluate_loader(
#                 val_loader, class_ids=seen_ids, target_class_ids=new_ids, structures=eval_working.seen
#             )
#             validation_new_only = None if val_loader is None else self.evaluate_loader(
#                 val_loader, class_ids=new_ids, target_class_ids=new_ids, structures=eval_working.current
#             )

#             online_per_class = {cid: per_class_correct[cid] / float(per_class_total[cid]) for cid in new_ids}
#             diag = eval_working.alignment
#             record = {
#                 "epoch": epoch + 1,
#                 "learning_rate": epoch_lr,
#                 "classifier_learning_rate": epoch_head_lr,
#                 "classification_loss": classification_sum / sample_count,
#                 "historical_decision_consistency_loss": hdc_sum / sample_count,
#                 "historical_decision_relation_rmse": hdc_rmse_sum / sample_count,
#                 "contrastive_loss": contrastive_sum / sample_count,
#                 "contrastive_similarity_gap": contrastive_gap_sum / sample_count,
#                 "contrastive_valid_anchor_fraction": contrastive_anchor_fraction_sum / sample_count,
#                 "hsi_cds_separation_loss": separation_sum / sample_count,
#                 "hsi_cds_separation_margin_mean": separation_margin_sum / sample_count,
#                 "cds_boundary_loss": boundary_sum / sample_count,
#                 "cds_boundary_clearance_mean": boundary_clearance_sum / sample_count,
#                 "cds_boundary_clearance_min": boundary_clearance_min_value,
#                 "cds_boundary_absolute_clearance_mean": boundary_absolute_clearance_sum / sample_count,
#                 "cds_boundary_relational_clearance_mean": boundary_relational_clearance_sum / sample_count,
#                 "cds_boundary_unsafe_pair_fraction": boundary_unsafe_sum / sample_count,
#                 "cds_boundary_pair_weight_mean": boundary_pair_weight_sum / sample_count,
#                 "cds_boundary_learned_risk_mean": boundary_learned_risk_sum / sample_count,
#                 "cds_boundary_spectral_risk_mean": boundary_spectral_risk_sum / sample_count,
#                 "total_loss": total_sum / sample_count,
#                 "new_train_oa_online": correct / float(sample_count),
#                 "new_train_ba_online": sum(online_per_class.values()) / float(len(online_per_class)),
#                 "new_train_min_online": min(online_per_class.values()),
#                 "new_train_ba_all_seen": float(new_train_all_seen["balanced_accuracy"]),
#                 "new_train_ba_new_only": float(new_train_only["balanced_accuracy"]),
#                 "validation_new_ba_all_seen": None if validation_all_seen is None else float(validation_all_seen["balanced_accuracy"]),
#                 "validation_new_ba_new_only": None if validation_new_only is None else float(validation_new_only["balanced_accuracy"]),
#                 "semantic_shift_mapping_error": float(diag.task_mapping_error.item()),
#                 "feature_drift_rms": float(diag.representation_drift_rms.item()),
#                 "reference_q_rms": float(reference_q_rms.item()),
#                 "current_q_rms": float(self._mean_sample_rms(eval_working.current_rows["ordered_features"]).item()),
#                 "classifier_logit_scale": float(classifier.logit_scale_value.detach().item()),
#             }
#             history.append(record)

#             message = (
#                 f"[Epoch {epoch + 1:03d}/{epoch_count:03d}] "
#                 f"CE={record['classification_loss']:.4f} Con={record['contrastive_loss']:.4f} "
#                 f"ConGap={record['contrastive_similarity_gap']:.4f} "
#                 f"HDC={record['historical_decision_consistency_loss']:.4f} "
#                 f"Sep={record['hsi_cds_separation_loss']:.4f} "
#                 f"Boundary={record['cds_boundary_loss']:.4f} "
#                 f"ClearMin={record['cds_boundary_clearance_min']:.4f} "
#                 f"LR={record['learning_rate']:.2e} HeadLR={record['classifier_learning_rate']:.2e} | "
#                 f"NewOnlyBA={100*record['new_train_ba_new_only']:.2f}% "
#                 f"AllSeenBA={100*record['new_train_ba_all_seen']:.2f}% "
#                 f"MapErr={record['semantic_shift_mapping_error']:.4f}"
#             )
#             if validation_all_seen is not None:
#                 message += (
#                     f" | ValNewOnlyBA={100*record['validation_new_ba_new_only']:.2f}%"
#                     f" ValAllSeenBA={100*record['validation_new_ba_all_seen']:.2f}%"
#                 )
#             tqdm.write(message, file=sys.stdout)

#         progress.close()
#         if final_working is None:
#             raise RuntimeError("incremental phase produced no working CDS")

#         self.model.eval()
#         final_state = self._refresh_training_state(
#             train_eval_loader=train_eval_loader,
#             phase_start_historical=phase_start_historical,
#             current_reference=phase_start_current,
#             phase_affinity=phase_affinity,
#             reference_q_full=reference_q_full,
#             reference_labels=reference_labels,
#             reference_spectra=reference_spectra,
#             reference_coords=reference_coords,
#             current_physical=current_physical,
#             new_ids=new_ids,
#             seen_ids=seen_ids,
#         )

#         final_old_only = self.evaluate_loader(
#             old_reporting_loader, class_ids=old_ids, target_class_ids=old_ids, structures=final_state.migrated_historical
#         )
#         final_old_all_seen = self.evaluate_loader(
#             old_reporting_loader, class_ids=seen_ids, target_class_ids=old_ids, structures=final_state.seen
#         )
#         final_new_all_seen = self.evaluate_loader(
#             train_eval_loader, class_ids=seen_ids, target_class_ids=new_ids, structures=final_state.seen
#         )
#         final_new_only = self.evaluate_loader(
#             train_eval_loader, class_ids=new_ids, target_class_ids=new_ids, structures=final_state.current
#         )

#         with torch.no_grad():
#             final_boundary_objective = cds_regulator(
#                 classifier=classifier,
#                 seen_cds=final_state.seen_cds,
#                 historical_class_count=len(old_ids),
#             )
#         self.model.replace_class_structure(final_state.seen)
#         self.model.validate_finalized_state()
#         self.dataset.finalize_phase(phase_id)

#         cumulative_loader = self.dataset.get_reporting_dataloader(phase_id, batch_size=eval_batch_size)
#         cumulative = self.evaluate_loader(cumulative_loader, class_ids=seen_ids, target_class_ids=seen_ids)
#         groups = old_new_summary(cumulative, old_class_ids=old_ids, new_class_ids=new_ids)
#         forgetting = float(old_before["balanced_accuracy"]) - float(groups["old"]["balanced_accuracy"])
#         phase_origin = phase_origin_confusion_diagnostics(
#             cumulative,
#             phase_to_classes={index: self.dataset.phase_to_classes[index] for index in range(phase_id + 1)},
#         )

#         diag = final_state.alignment
#         method_state = {
#             "classifier": "single_bias_free_scaled_cosine_head",
#             "persistent_cds": "{I_Q,V_Q,I_P,V_P,N}; all inter-class CDS is derived on demand",
#             "historical_supervision": "HDC_plus_feature_side_old_rival_separation_plus_whole_CDS_classifier_region_protection",
#             "historical_cds_information": "I_Q locates migrated historical Q regions; V_Q bounds absolute/relational extent; I_P,V_P prioritize historical/seen classifier conflicts through whole-CDS risk",
#             "representation_objective": "real_current_CE_plus_direct_Q_supervised_contrast_plus_HDC_plus_feature_only_old_rival_separation",
#             "classifier_objective": "migrated_old_CDS_absolute_and_relational_clearance_against_all_seen_rows; current_CE_detaches_old_rows",
#             "stores_pseudo_features": False,
#             "reconstructs_historical_features": False,
#             "uses_input_inversion": False,
#             "uses_stage_B": False,
#             "uses_covariance_classifier": False,
#             "semantic_shift_mapping_error": float(diag.task_mapping_error.item()),
#             "semantic_shift_channel_error_max": float(diag.task_mapping_channel_error_max.item()),
#             "feature_drift_rms": float(diag.representation_drift_rms.item()),
#             "task_translation_rms": float(diag.task_translation_rms.item()),
#             "task_scale_deviation_rms": float(diag.task_scale_deviation_rms.item()),
#             "reference_q_rms": float(reference_q_rms.item()),
#             "current_q_rms": float(self._mean_sample_rms(final_state.current_rows["ordered_features"]).item()),
#             "phase_start_old_ba": float(old_before["balanced_accuracy"]),
#             "phase_start_old_all_seen_ba": float(phase_start_old_all_seen["balanced_accuracy"]),
#             "final_old_only_ba": float(final_old_only["balanced_accuracy"]),
#             "final_old_all_seen_ba": float(final_old_all_seen["balanced_accuracy"]),
#             "representation_forgetting": float(old_before["balanced_accuracy"]) - float(final_old_only["balanced_accuracy"]),
#             "decision_interference_start": float(old_before["balanced_accuracy"]) - float(phase_start_old_all_seen["balanced_accuracy"]),
#             "decision_interference_final": float(final_old_only["balanced_accuracy"]) - float(final_old_all_seen["balanced_accuracy"]),
#             "final_new_train_all_seen_ba": float(final_new_all_seen["balanced_accuracy"]),
#             "final_new_train_only_ba": float(final_new_only["balanced_accuracy"]),
#             "classifier_logit_scale": float(classifier.logit_scale_value.detach().item()),
#             "final_classification_loss": float(history[-1]["classification_loss"]),
#             "final_contrastive_loss": float(history[-1]["contrastive_loss"]),
#             "final_contrastive_similarity_gap": float(history[-1]["contrastive_similarity_gap"]),
#             "final_hsi_cds_separation_loss": float(history[-1]["hsi_cds_separation_loss"]),
#             "final_hsi_cds_separation_margin_mean": float(history[-1]["hsi_cds_separation_margin_mean"]),
#             "final_historical_decision_consistency_loss": float(history[-1]["historical_decision_consistency_loss"]),
#             "final_cds_boundary_loss": float(final_boundary_objective.loss.item()),
#             "final_cds_boundary_clearance_mean": float(final_boundary_objective.structural_clearance_mean.item()),
#             "final_cds_boundary_clearance_min": float(final_boundary_objective.structural_clearance_min.item()),
#             "final_cds_boundary_absolute_clearance_mean": float(final_boundary_objective.absolute_clearance_mean.item()),
#             "final_cds_boundary_relational_clearance_mean": float(final_boundary_objective.relational_clearance_mean.item()),
#             "final_cds_boundary_unsafe_pair_fraction": float(final_boundary_objective.unsafe_pair_fraction.item()),
#             "final_cds_boundary_pair_weight_mean": float(final_boundary_objective.pair_weight_mean.item()),
#             "final_cds_boundary_learned_risk_mean": float(final_boundary_objective.learned_region_risk_mean.item()),
#             "final_cds_boundary_spectral_risk_mean": float(final_boundary_objective.spectral_region_risk_mean.item()),
#             "historical_classifier_rows_frozen_for_current_CE": True,
#             "historical_classifier_rows_realign_via_migrated_CDS": True,
#         }

#         return {
#             "phase": phase_id,
#             "old_class_ids": list(old_ids),
#             "new_class_ids": list(new_ids),
#             "seen_class_ids": list(seen_ids),
#             "history": history,
#             "final_epoch": epoch_count,
#             "cumulative_test": cumulative,
#             "old_new": groups,
#             "forgetting": forgetting,
#             "method_state": method_state,
#             "phase_origin_confusion": phase_origin,
#             "structure_committed": True,
#             "phase_finalized": True,
#         }


# __all__ = ["IncrementalPhaseTrainer"]






























# from __future__ import annotations

# """Incremental NECIL-HSI training with one Q, one all-seen CDS and one cosine head.

# Current real HSI supplies CE and direct-Q supervised contrast. HDC stabilizes the
# representation against the frozen previous backbone/old classifier rows. A
# feature-only old-rival separation term keeps current Q away from historical
# decisions. The complete all-seen CDS then protects the SAME classifier: I_Q,V_Q
# define historical decision regions and I_P,V_P prioritize spectrally confusable
# old/current boundaries. Historical classifier rows remain bitwise frozen.
# """

# from dataclasses import dataclass
# import math
# import sys
# from numbers import Integral, Real
# from typing import Any, Dict, Mapping

# import torch
# from tqdm import tqdm

# from losses.loss import (
#     classification_objective,
#     supervised_contrastive_q_loss,
#     historical_decision_consistency_loss,
#     hsi_cds_separation_loss,
# )
# from models.cds_decision_region import HSICDSDecisionRegionRegulator
# from models.classifier import SpectralCDSClassifier
# from models.geometry_bank import (
#     ClassDiscriminativeStructureState,
#     PhysicalSpectralStructureBatch,
#     ClassStructureBatch,
#     merge_class_structure_batches,
# )
# from models.phase_coordinate_alignment import (
#     CDSAffinityState,
#     CurrentClassResidual,
#     PhaseAlignmentDiagnostics,
#     PhaseCoordinateTransform,
#     align_historical_structure,
#     derive_cds_affinity,
# )
# from utils.eval import old_new_summary, phase_origin_confusion_diagnostics

# Tensor = torch.Tensor


# def _positive_int(value: object, *, name: str) -> int:
#     if isinstance(value, bool) or not isinstance(value, (Integral, Real)):
#         raise ValueError(f"{name} must be a positive integer")
#     number = float(value)
#     if not math.isfinite(number) or not number.is_integer() or number <= 0:
#         raise ValueError(f"{name} must be a positive integer")
#     return int(number)


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


# def _clone_structure(batch: ClassStructureBatch) -> ClassStructureBatch:
#     if not isinstance(batch, ClassStructureBatch) or batch.class_count <= 0:
#         raise ValueError("cannot clone an empty class structure")
#     return ClassStructureBatch(
#         class_ids=batch.class_ids.detach().clone(),
#         sample_counts=batch.sample_counts.detach().clone(),
#         identity=batch.identity.detach().clone(),
#         variation=batch.variation.detach().clone(),
#         spectral_identity=batch.spectral_identity.detach().clone(),
#         spectral_variation=batch.spectral_variation.detach().clone(),
#     )


# def _targets_local(labels: Tensor, class_ids: Tensor) -> Tensor:
#     value = torch.as_tensor(labels)
#     ids = torch.as_tensor(class_ids)
#     if value.ndim != 1 or value.numel() == 0 or value.dtype != torch.long:
#         raise ValueError("labels must be non-empty rank-one torch.long")
#     if ids.ndim != 1 or ids.numel() < 1 or ids.dtype != torch.long:
#         raise ValueError("class_ids must be non-empty rank-one torch.long")
#     if value.device != ids.device:
#         raise ValueError("labels and class_ids must share device")
#     matches = value.unsqueeze(1).eq(ids.unsqueeze(0))
#     if not bool(matches.sum(dim=1).eq(1).all()):
#         raise ValueError("labels contain a class absent from class_ids")
#     return matches.to(torch.long).argmax(dim=1)


# def _batch_coordinates(batch: Mapping[str, Any], *, expected_rows: int) -> Tensor:
#     if "coord" not in batch:
#         raise RuntimeError("incremental TRAIN batches must expose coord for phase-start reference lookup")
#     coords = torch.as_tensor(batch["coord"])
#     if coords.shape != (expected_rows, 2):
#         raise RuntimeError("coord must be [N,2] and batch-aligned")
#     if coords.dtype == torch.bool or coords.is_complex() or torch.is_floating_point(coords):
#         raise RuntimeError("coord must use an integer dtype")
#     return coords.to(dtype=torch.long, device="cpu")


# @dataclass(frozen=True)
# class CurrentTrainingState:
#     task_mapped_historical: ClassStructureBatch
#     migrated_historical: ClassStructureBatch
#     current: ClassStructureBatch
#     seen: ClassStructureBatch
#     seen_cds: ClassDiscriminativeStructureState
#     current_rows: Dict[str, Tensor]
#     transform: PhaseCoordinateTransform
#     affinity: CDSAffinityState
#     residual: CurrentClassResidual
#     alignment: PhaseAlignmentDiagnostics


# class IncrementalPhaseTrainer:
#     INCREMENTAL_RESULT_FIELDS = frozenset(
#         {
#             "phase", "old_class_ids", "new_class_ids", "seen_class_ids", "history",
#             "final_epoch", "cumulative_test", "old_new", "forgetting", "method_state",
#             "structure_committed", "phase_finalized",
#         }
#     )

#     def _require_classifier(self) -> SpectralCDSClassifier:
#         classifier = getattr(self.model, "classifier", None)
#         if not isinstance(classifier, SpectralCDSClassifier) or classifier.parameter_count <= 0:
#             raise RuntimeError("model.classifier must be the trainable cosine SpectralCDSClassifier")
#         return classifier

#     def _phase_classes(self, phase: int) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
#         old_ids = tuple(self._validated_class_ids(self.dataset.get_old_classes(phase), name=f"phase_{phase}_old_classes"))
#         new_ids = tuple(self._validated_class_ids(self.dataset.get_new_classes(phase), name=f"phase_{phase}_new_classes"))
#         seen_ids = tuple(self._validated_class_ids(self.dataset.get_seen_classes(phase), name=f"phase_{phase}_seen_classes"))
#         if not old_ids or not new_ids or set(old_ids).intersection(new_ids):
#             raise RuntimeError("incremental phase has invalid old/new class partition")
#         if seen_ids != old_ids + new_ids:
#             raise RuntimeError("seen classes must equal old classes followed by new classes")
#         if tuple(self.model.committed_class_ids) != old_ids:
#             raise RuntimeError("historical CDS does not match the old class set")
#         return old_ids, new_ids, seen_ids

#     @staticmethod
#     def _assert_frozen_reference(reference_backbone: torch.nn.Module) -> None:
#         reference_backbone.eval()
#         if any(parameter.requires_grad for parameter in reference_backbone.parameters()):
#             raise RuntimeError("previous backbone reference must remain fully frozen")
#         if any(parameter.grad is not None for parameter in reference_backbone.parameters()):
#             raise RuntimeError("previous backbone reference must not carry gradients")

#     @staticmethod
#     @torch.no_grad()
#     def _assert_reference_matches_phase_start(reference_backbone: torch.nn.Module, current_backbone: torch.nn.Module) -> None:
#         left, right = reference_backbone.state_dict(), current_backbone.state_dict()
#         if tuple(left.keys()) != tuple(right.keys()):
#             raise RuntimeError("previous/current backbone schemas differ at phase start")
#         for name in left:
#             if left[name].shape != right[name].shape or left[name].dtype != right[name].dtype:
#                 raise RuntimeError(f"phase-start backbone state differs at {name}")
#             if not torch.equal(left[name].detach().cpu(), right[name].detach().cpu()):
#                 raise RuntimeError(f"previous backbone is not the finalized phase-start state ({name})")

#     @torch.no_grad()
#     def _current_class_weights(self, loader: Any, *, new_ids: tuple[int, ...]) -> tuple[Tensor, int]:
#         counts = {class_id: 0 for class_id in new_ids}
#         total = 0
#         valid = set(new_ids)
#         for batch in loader:
#             _, _, labels = self.unpack_batch(batch)
#             observed = set(int(v) for v in labels.unique().detach().cpu().tolist())
#             outside = observed - valid
#             if outside:
#                 raise RuntimeError(f"incremental TRAIN loader contains non-current classes {sorted(outside)}")
#             for class_id in new_ids:
#                 count = int(labels.eq(class_id).sum().item())
#                 counts[class_id] += count
#                 total += count
#         if total <= 0 or any(counts[c] <= 0 for c in new_ids):
#             raise RuntimeError("incremental TRAIN split is empty or misses a current class")
#         class_count = len(new_ids)
#         weights = torch.tensor(
#             [total / float(class_count * counts[c]) for c in new_ids],
#             device=self.device,
#             dtype=self.model.dtype,
#         )
#         return weights, total

#     @staticmethod
#     def _row_lookup(reference_coords: Tensor) -> dict[tuple[int, int], int]:
#         if reference_coords.ndim != 2 or reference_coords.size(1) != 2:
#             raise ValueError("reference_coords must be [N,2]")
#         lookup: dict[tuple[int, int], int] = {}
#         for row, pair in enumerate(reference_coords.detach().cpu().tolist()):
#             key = (int(pair[0]), int(pair[1]))
#             if key in lookup:
#                 raise RuntimeError(f"duplicate current TRAIN coordinate {key}")
#             lookup[key] = row
#         return lookup

#     @staticmethod
#     def _cached_rows_for_batch(batch_coords: Tensor, *, lookup: dict[tuple[int, int], int], cached: Tensor) -> Tensor:
#         rows: list[int] = []
#         for pair in batch_coords.tolist():
#             key = (int(pair[0]), int(pair[1]))
#             if key not in lookup:
#                 raise RuntimeError(f"current TRAIN batch coordinate {key} is absent from phase-start cache")
#             rows.append(lookup[key])
#         return cached.index_select(0, torch.tensor(rows, device=cached.device, dtype=torch.long))

#     @staticmethod
#     def _mean_sample_rms(features: Tensor) -> Tensor:
#         value = torch.as_tensor(features)
#         if value.ndim != 3 or value.size(0) == 0:
#             raise ValueError("features must be non-empty [N,K,D]")
#         return value.square().mean(dim=(1, 2)).sqrt().mean()

#     @torch.no_grad()
#     def _refresh_training_state(
#         self,
#         *,
#         train_eval_loader: Any,
#         phase_start_historical: ClassStructureBatch,
#         current_reference: ClassStructureBatch,
#         phase_affinity: CDSAffinityState,
#         reference_q_full: Tensor,
#         reference_labels: Tensor,
#         reference_spectra: Tensor,
#         reference_coords: Tensor,
#         current_physical: PhysicalSpectralStructureBatch,
#         new_ids: tuple[int, ...],
#         seen_ids: tuple[int, ...],
#     ) -> CurrentTrainingState:
#         current_rows = self.collect_ordered_features(train_eval_loader, require_coordinates=True)
#         if not torch.equal(current_rows["labels"], reference_labels):
#             raise RuntimeError("current TRAIN labels/order changed within phase")
#         if not torch.equal(current_rows["coords"], reference_coords):
#             raise RuntimeError("current TRAIN coordinates/order changed within phase")
#         if not torch.equal(current_rows["raw_center_spectra"], reference_spectra):
#             raise RuntimeError("current TRAIN spectra/order changed within phase")

#         current_structure = self.model.estimate_class_structure(
#             current_rows["ordered_features"],
#             None,
#             current_rows["labels"],
#             class_ids=new_ids,
#             physical_structure=current_physical,
#         )
#         task_mapped_old, migrated_old, transform, affinity, residual, alignment = align_historical_structure(
#             historical=phase_start_historical,
#             current_reference=current_reference,
#             current_structure=current_structure,
#             reference_features=reference_q_full,
#             current_features=current_rows["ordered_features"],
#             affinity=phase_affinity,
#         )
#         seen_structure = merge_class_structure_batches((migrated_old, current_structure))
#         actual = tuple(int(v) for v in seen_structure.class_ids.detach().cpu().tolist())
#         if actual != seen_ids:
#             raise RuntimeError("all-seen CDS class order changed")
#         seen_cds = self.model.build_class_discriminative_structure(seen_structure)
#         return CurrentTrainingState(
#             task_mapped_historical=task_mapped_old,
#             migrated_historical=migrated_old,
#             current=current_structure,
#             seen=seen_structure,
#             seen_cds=seen_cds,
#             current_rows=current_rows,
#             transform=transform,
#             affinity=affinity,
#             residual=residual,
#             alignment=alignment,
#         )

#     def train_incremental_phase(
#         self,
#         *,
#         phase: int,
#         epochs: int,
#         batch_size: int,
#         lr: float,
#         reference_backbone: torch.nn.Module,
#     ) -> Dict[str, Any]:
#         phase_id = _positive_int(phase, name="phase")
#         epoch_count = _positive_int(epochs, name="epochs")
#         train_batch_size = _positive_int(batch_size, name="batch_size")
#         backbone_learning_rate = _finite_positive("lr_inc", lr)
#         classifier_learning_rate = _finite_positive("lr", self.cfg("lr", backbone_learning_rate))
#         eval_batch_size = _positive_int(self.cfg("eval_batch_size", 256), name="eval_batch_size")
#         gradient_clip = _finite_nonnegative("gradient_clip", self.cfg("gradient_clip", 0.0))
#         weight_decay = _finite_nonnegative("weight_decay", self.cfg("weight_decay", 1e-4))

#         self.model.validate_finalized_state()
#         classifier = self._require_classifier()
#         self._assert_frozen_reference(reference_backbone)
#         self._assert_reference_matches_phase_start(reference_backbone, self.model.backbone)

#         previous_phase = phase_id - 1
#         previous_seen = tuple(self._validated_class_ids(self.dataset.get_seen_classes(previous_phase), name="previous_seen_classes"))
#         old_reporting_loader = self.dataset.get_reporting_dataloader(previous_phase, batch_size=eval_batch_size)
#         old_before = self.evaluate_loader(old_reporting_loader, class_ids=previous_seen, target_class_ids=previous_seen)

#         self.dataset.start_phase(phase_id)
#         old_ids, new_ids, seen_ids = self._phase_classes(phase_id)
#         old_ids_tensor = torch.tensor(old_ids, device=self.device, dtype=torch.long)
#         new_ids_tensor = torch.tensor(new_ids, device=self.device, dtype=torch.long)
#         seen_ids_tensor = torch.tensor(seen_ids, device=self.device, dtype=torch.long)
#         phase_start_historical = _clone_structure(self.model.structure_bank.class_structures(old_ids))

#         self.model.set_incremental_trainability()
#         backbone_trainable = [p for p in self.model.backbone.parameters() if p.requires_grad]
#         backbone_frozen = [p for p in self.model.backbone.parameters() if not p.requires_grad]
#         if not backbone_trainable:
#             raise RuntimeError("incremental backbone has no trainable parameters")

#         structure_loader_factory = getattr(self.dataset, "get_phase_structure_dataloader", None)
#         train_eval_loader = (
#             structure_loader_factory(phase_id, batch_size=eval_batch_size)
#             if callable(structure_loader_factory)
#             else self.dataset.get_phase_dataloader(
#                 phase_id, split="train", batch_size=eval_batch_size, shuffle=False
#             )
#         )
#         class_weights, expected_samples = self._current_class_weights(train_eval_loader, new_ids=new_ids)

#         reference_train = self.collect_ordered_features(
#             train_eval_loader, backbone=reference_backbone, require_coordinates=True
#         )
#         reference_q_full = reference_train["ordered_features"].detach()
#         reference_spectra = reference_train["raw_center_spectra"].detach()
#         reference_labels = reference_train["labels"].detach()
#         reference_coords = reference_train["coords"]
#         observed = tuple(sorted(int(v) for v in reference_labels.unique().detach().cpu().tolist()))
#         if observed != tuple(sorted(new_ids)):
#             raise RuntimeError("incremental TRAIN rows do not match current phase classes")
#         current_physical = self.model.estimate_physical_class_structure(
#             reference_spectra, reference_labels, class_ids=new_ids
#         )
#         cache_lookup = self._row_lookup(reference_coords)
#         reference_q_rms = self._mean_sample_rms(reference_q_full).detach()

#         # The previous tested old rows are the deployed historical decision rows
#         # and remain bitwise frozen for the entire phase. They are also the HDC
#         # reference used on the same legal current TRAIN samples.
#         historical_classifier_rows = classifier.class_weight(old_ids_tensor).detach().clone()

#         phase_start_current = self.model.estimate_class_structure(
#             reference_q_full,
#             None,
#             reference_labels,
#             class_ids=new_ids,
#             physical_structure=current_physical,
#         )
#         phase_start_seen = merge_class_structure_batches((phase_start_historical, phase_start_current))
#         phase_affinity = derive_cds_affinity(
#             historical=phase_start_historical,
#             current_reference=phase_start_current,
#         )

#         # New classifier rows are NOT initialized from I_Q or any CDS center.
#         # They retain their deterministic model initialization and are learned
#         # from real current samples. The complete CDS constrains them through
#         # whole-CDS contrast and boundary clearance; historical rows stay fixed.
#         if not torch.equal(classifier.class_weight(old_ids_tensor), historical_classifier_rows):
#             raise RuntimeError("historical classifier rows changed before incremental optimization")

#         phase_start_old_all_seen = self.evaluate_loader(
#             old_reporting_loader,
#             class_ids=seen_ids,
#             target_class_ids=old_ids,
#             structures=phase_start_seen,
#         )

#         print(
#             f"[Phase {phase_id} start] PhaseStartOldBA={100*float(old_before['balanced_accuracy']):.2f}% "
#             f"OldAllSeen={100*float(phase_start_old_all_seen['balanced_accuracy']):.2f}% | "
#             f"Old={list(old_ids)} New={list(new_ids)} Seen={list(seen_ids)}",
#             file=sys.stdout,
#         )

#         backbone_optimizer = torch.optim.AdamW(
#             [{"params": backbone_trainable, "weight_decay": weight_decay}],
#             lr=backbone_learning_rate,
#         )
#         classifier_optimizer = torch.optim.AdamW(
#             list(classifier.parameters()), lr=classifier_learning_rate, weight_decay=0.0
#         )
#         backbone_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
#             backbone_optimizer, T_max=epoch_count, eta_min=0.0
#         )
#         classifier_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
#             classifier_optimizer, T_max=epoch_count, eta_min=0.0
#         )

#         train_loader = self.dataset.get_phase_dataloader(
#             phase_id, split="train", batch_size=train_batch_size, shuffle=True
#         )
#         val_loader = (
#             self.dataset.get_phase_dataloader(phase_id, split="val", batch_size=eval_batch_size, shuffle=False)
#             if self.dataset.validation_enabled else None
#         )
#         cds_regulator = HSICDSDecisionRegionRegulator().to(self.device)
#         if cds_regulator.parameter_count != 0:
#             raise RuntimeError("CDS decision-region regulator must remain parameter-free")

#         history: list[dict[str, Any]] = []
#         progress = tqdm(
#             range(epoch_count),
#             desc=f"Phase {phase_id} all-seen CDS continual learning",
#             unit="epoch",
#             dynamic_ncols=True,
#             file=sys.stdout,
#         )

#         final_working: CurrentTrainingState | None = None
#         for epoch in progress:
#             self.model.eval()
#             working = self._refresh_training_state(
#                 train_eval_loader=train_eval_loader,
#                 phase_start_historical=phase_start_historical,
#                 current_reference=phase_start_current,
#                 phase_affinity=phase_affinity,
#                 reference_q_full=reference_q_full,
#                 reference_labels=reference_labels,
#                 reference_spectra=reference_spectra,
#                 reference_coords=reference_coords,
#                 current_physical=current_physical,
#                 new_ids=new_ids,
#                 seen_ids=seen_ids,
#             )

#             self.model.train()
#             epoch_lr = float(backbone_optimizer.param_groups[0]["lr"])
#             epoch_head_lr = float(classifier_optimizer.param_groups[0]["lr"])
#             classification_sum = hdc_sum = hdc_rmse_sum = 0.0
#             contrastive_sum = contrastive_gap_sum = contrastive_anchor_fraction_sum = 0.0
#             separation_sum = separation_margin_sum = 0.0
#             boundary_sum = boundary_clearance_sum = 0.0
#             boundary_clearance_min_value = float("inf")
#             boundary_absolute_clearance_sum = boundary_relational_clearance_sum = 0.0
#             boundary_unsafe_sum = 0.0
#             boundary_pair_weight_sum = boundary_learned_risk_sum = boundary_spectral_risk_sum = 0.0
#             total_sum = 0.0
#             correct = sample_count = 0
#             per_class_correct = {class_id: 0 for class_id in new_ids}
#             per_class_total = {class_id: 0 for class_id in new_ids}

#             for batch in train_loader:
#                 patch, spectrum, labels = self.unpack_batch(batch)
#                 batch_coords = _batch_coordinates(batch, expected_rows=int(labels.numel()))
#                 backbone_optimizer.zero_grad(set_to_none=True)
#                 classifier_optimizer.zero_grad(set_to_none=True)

#                 q = self.model.extract_features(patch, center_spectrum=spectrum, return_aux=False).ordered_features
#                 reference_q = self._cached_rows_for_batch(batch_coords, lookup=cache_lookup, cached=reference_q_full)

#                 # Real current labels are allowed to move only current classifier
#                 # rows. Historical rows remain active rivals but are detached here.
#                 all_seen_output = classifier(
#                     q,
#                     class_ids=seen_ids_tensor,
#                     trainable_class_ids=new_ids_tensor,
#                 )
#                 target_new_local = _targets_local(labels, new_ids_tensor)
#                 sample_weights = class_weights.index_select(0, target_new_local)
#                 classification = classification_objective(
#                     logits=all_seen_output.logits,
#                     class_ids=all_seen_output.class_ids,
#                     labels_global=labels,
#                     sample_weights=sample_weights,
#                 )

#                 hdc = historical_decision_consistency_loss(
#                     reference_features=reference_q,
#                     current_features=q,
#                     historical_classifier_rows=historical_classifier_rows,
#                     labels_global=labels,
#                     seen_cds=working.seen_cds,
#                     historical_class_count=len(old_ids),
#                 )

#                 # Same direct-Q supervised contrast used in Phase 0. Only real
#                 # current samples participate; no CDS center, pseudo feature, or
#                 # second representation is introduced.
#                 contrastive = supervised_contrastive_q_loss(
#                     ordered_features=q,
#                     labels_global=labels,
#                     logit_scale=classifier.logit_scale_value,
#                     sample_weights=sample_weights,
#                 )

#                 # Feature-side new->old separation. Classifier rows and scale are
#                 # detached, so this objective can move Q but cannot move W. The
#                 # complete all-seen CDS only prioritizes old/current conflicts.
#                 separation_cosine = classifier.cosine_similarity(
#                     q,
#                     class_ids=seen_ids_tensor,
#                     trainable_class_ids=(),
#                 )
#                 separation_logits = separation_cosine * classifier.logit_scale_value.detach().to(
#                     device=q.device, dtype=q.dtype
#                 )
#                 separation = hsi_cds_separation_loss(
#                     logits=separation_logits,
#                     class_ids=seen_ids_tensor,
#                     labels_global=labels,
#                     seen_cds=working.seen_cds,
#                     historical_class_count=len(old_ids),
#                     sample_weights=sample_weights,
#                 )

#                 # Classifier-side old->new protection. I_Q locates each historical
#                 # region, diagonal/off-diagonal V_Q bound its legitimate extent,
#                 # and I_P/V_P enter whole-CDS old/current risk. Only current rows
#                 # are live; historical classifier rows are detached and frozen.
#                 boundary = cds_regulator(
#                     classifier=classifier,
#                     seen_cds=working.seen_cds,
#                     historical_class_count=len(old_ids),
#                 )

#                 loss = (
#                     classification.loss
#                     + contrastive.loss
#                     + hdc.loss
#                     + separation.loss
#                     + boundary.loss
#                 )
#                 if not bool(torch.isfinite(loss)):
#                     raise RuntimeError(
#                         "incremental CE + direct-Q contrast + HDC + CDS separation + classifier protection became NaN or Inf"
#                     )
#                 loss.backward()
#                 classifier.mask_gradient_to_classes(new_ids_tensor)

#                 if any(parameter.grad is not None for parameter in reference_backbone.parameters()):
#                     raise RuntimeError("previous backbone reference accumulated gradients")
#                 if any(parameter.grad is not None for parameter in backbone_frozen):
#                     raise RuntimeError("a frozen incremental backbone parameter accumulated gradients")
#                 for parameter in backbone_trainable:
#                     if parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all()):
#                         raise RuntimeError("incremental backbone gradient contains NaN or Inf")
#                 for parameter in classifier.parameters():
#                     if parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all()):
#                         raise RuntimeError("incremental classifier gradient contains NaN or Inf")
#                 weight_grad = classifier.weight.grad
#                 if weight_grad is not None and not torch.equal(
#                     weight_grad.index_select(0, old_ids_tensor),
#                     torch.zeros_like(weight_grad.index_select(0, old_ids_tensor)),
#                 ):
#                     raise RuntimeError("historical classifier rows received incremental gradients")
#                 if gradient_clip > 0.0:
#                     torch.nn.utils.clip_grad_norm_(backbone_trainable, gradient_clip, error_if_nonfinite=True)

#                 backbone_optimizer.step()
#                 classifier_optimizer.step()
#                 if not torch.equal(classifier.class_weight(old_ids_tensor), historical_classifier_rows):
#                     raise RuntimeError("historical classifier rows changed during incremental optimization")

#                 n = int(labels.numel())
#                 classification_sum += float(classification.loss.detach().item()) * n
#                 hdc_sum += float(hdc.loss.detach().item()) * n
#                 hdc_rmse_sum += float(hdc.relation_rmse.detach().item()) * n
#                 contrastive_sum += float(contrastive.loss.detach().item()) * n
#                 contrastive_gap_sum += float(contrastive.similarity_gap.detach().item()) * n
#                 contrastive_anchor_fraction_sum += float(contrastive.valid_anchor_fraction.detach().item()) * n
#                 separation_sum += float(separation.loss.detach().item()) * n
#                 separation_margin_sum += float(separation.margin_mean.detach().item()) * n
#                 boundary_sum += float(boundary.loss.detach().item()) * n
#                 boundary_clearance_sum += float(boundary.structural_clearance_mean.detach().item()) * n
#                 boundary_clearance_min_value = min(
#                     boundary_clearance_min_value,
#                     float(boundary.structural_clearance_min.detach().item()),
#                 )
#                 boundary_absolute_clearance_sum += float(boundary.absolute_clearance_mean.detach().item()) * n
#                 boundary_relational_clearance_sum += float(boundary.relational_clearance_mean.detach().item()) * n
#                 boundary_unsafe_sum += float(boundary.unsafe_pair_fraction.detach().item()) * n
#                 boundary_pair_weight_sum += float(boundary.pair_weight_mean.detach().item()) * n
#                 boundary_learned_risk_sum += float(boundary.learned_region_risk_mean.detach().item()) * n
#                 boundary_spectral_risk_sum += float(boundary.spectral_region_risk_mean.detach().item()) * n
#                 total_sum += float(loss.detach().item()) * n

#                 prediction = all_seen_output.prediction.detach()
#                 correct += int(prediction.eq(labels).sum().item())
#                 sample_count += n
#                 for class_id in new_ids:
#                     mask = labels.eq(class_id)
#                     count = int(mask.sum().item())
#                     per_class_total[class_id] += count
#                     if count:
#                         per_class_correct[class_id] += int(prediction[mask].eq(labels[mask]).sum().item())

#             if sample_count != expected_samples:
#                 raise RuntimeError("incremental TRAIN loader did not traverse the complete current split")

#             backbone_scheduler.step()
#             classifier_scheduler.step()

#             self.model.eval()
#             eval_working = self._refresh_training_state(
#                 train_eval_loader=train_eval_loader,
#                 phase_start_historical=phase_start_historical,
#                 current_reference=phase_start_current,
#                 phase_affinity=phase_affinity,
#                 reference_q_full=reference_q_full,
#                 reference_labels=reference_labels,
#                 reference_spectra=reference_spectra,
#                 reference_coords=reference_coords,
#                 current_physical=current_physical,
#                 new_ids=new_ids,
#                 seen_ids=seen_ids,
#             )
#             final_working = eval_working

#             new_train_all_seen = self.evaluate_loader(
#                 train_eval_loader, class_ids=seen_ids, target_class_ids=new_ids, structures=eval_working.seen
#             )
#             new_train_only = self.evaluate_loader(
#                 train_eval_loader, class_ids=new_ids, target_class_ids=new_ids, structures=eval_working.current
#             )
#             validation_all_seen = None if val_loader is None else self.evaluate_loader(
#                 val_loader, class_ids=seen_ids, target_class_ids=new_ids, structures=eval_working.seen
#             )
#             validation_new_only = None if val_loader is None else self.evaluate_loader(
#                 val_loader, class_ids=new_ids, target_class_ids=new_ids, structures=eval_working.current
#             )

#             online_per_class = {cid: per_class_correct[cid] / float(per_class_total[cid]) for cid in new_ids}
#             diag = eval_working.alignment
#             record = {
#                 "epoch": epoch + 1,
#                 "learning_rate": epoch_lr,
#                 "classifier_learning_rate": epoch_head_lr,
#                 "classification_loss": classification_sum / sample_count,
#                 "historical_decision_consistency_loss": hdc_sum / sample_count,
#                 "historical_decision_relation_rmse": hdc_rmse_sum / sample_count,
#                 "contrastive_loss": contrastive_sum / sample_count,
#                 "contrastive_similarity_gap": contrastive_gap_sum / sample_count,
#                 "contrastive_valid_anchor_fraction": contrastive_anchor_fraction_sum / sample_count,
#                 "hsi_cds_separation_loss": separation_sum / sample_count,
#                 "hsi_cds_separation_margin_mean": separation_margin_sum / sample_count,
#                 "cds_boundary_loss": boundary_sum / sample_count,
#                 "cds_boundary_clearance_mean": boundary_clearance_sum / sample_count,
#                 "cds_boundary_clearance_min": boundary_clearance_min_value,
#                 "cds_boundary_absolute_clearance_mean": boundary_absolute_clearance_sum / sample_count,
#                 "cds_boundary_relational_clearance_mean": boundary_relational_clearance_sum / sample_count,
#                 "cds_boundary_unsafe_pair_fraction": boundary_unsafe_sum / sample_count,
#                 "cds_boundary_pair_weight_mean": boundary_pair_weight_sum / sample_count,
#                 "cds_boundary_learned_risk_mean": boundary_learned_risk_sum / sample_count,
#                 "cds_boundary_spectral_risk_mean": boundary_spectral_risk_sum / sample_count,
#                 "total_loss": total_sum / sample_count,
#                 "new_train_oa_online": correct / float(sample_count),
#                 "new_train_ba_online": sum(online_per_class.values()) / float(len(online_per_class)),
#                 "new_train_min_online": min(online_per_class.values()),
#                 "new_train_ba_all_seen": float(new_train_all_seen["balanced_accuracy"]),
#                 "new_train_ba_new_only": float(new_train_only["balanced_accuracy"]),
#                 "validation_new_ba_all_seen": None if validation_all_seen is None else float(validation_all_seen["balanced_accuracy"]),
#                 "validation_new_ba_new_only": None if validation_new_only is None else float(validation_new_only["balanced_accuracy"]),
#                 "semantic_shift_mapping_error": float(diag.task_mapping_error.item()),
#                 "feature_drift_rms": float(diag.representation_drift_rms.item()),
#                 "reference_q_rms": float(reference_q_rms.item()),
#                 "current_q_rms": float(self._mean_sample_rms(eval_working.current_rows["ordered_features"]).item()),
#                 "classifier_logit_scale": float(classifier.logit_scale_value.detach().item()),
#             }
#             history.append(record)

#             message = (
#                 f"[Epoch {epoch + 1:03d}/{epoch_count:03d}] "
#                 f"CE={record['classification_loss']:.4f} Con={record['contrastive_loss']:.4f} "
#                 f"ConGap={record['contrastive_similarity_gap']:.4f} "
#                 f"HDC={record['historical_decision_consistency_loss']:.4f} "
#                 f"Sep={record['hsi_cds_separation_loss']:.4f} "
#                 f"Boundary={record['cds_boundary_loss']:.4f} "
#                 f"ClearMin={record['cds_boundary_clearance_min']:.4f} "
#                 f"LR={record['learning_rate']:.2e} HeadLR={record['classifier_learning_rate']:.2e} | "
#                 f"NewOnlyBA={100*record['new_train_ba_new_only']:.2f}% "
#                 f"AllSeenBA={100*record['new_train_ba_all_seen']:.2f}% "
#                 f"MapErr={record['semantic_shift_mapping_error']:.4f}"
#             )
#             if validation_all_seen is not None:
#                 message += (
#                     f" | ValNewOnlyBA={100*record['validation_new_ba_new_only']:.2f}%"
#                     f" ValAllSeenBA={100*record['validation_new_ba_all_seen']:.2f}%"
#                 )
#             tqdm.write(message, file=sys.stdout)

#         progress.close()
#         if final_working is None:
#             raise RuntimeError("incremental phase produced no working CDS")

#         self.model.eval()
#         final_state = self._refresh_training_state(
#             train_eval_loader=train_eval_loader,
#             phase_start_historical=phase_start_historical,
#             current_reference=phase_start_current,
#             phase_affinity=phase_affinity,
#             reference_q_full=reference_q_full,
#             reference_labels=reference_labels,
#             reference_spectra=reference_spectra,
#             reference_coords=reference_coords,
#             current_physical=current_physical,
#             new_ids=new_ids,
#             seen_ids=seen_ids,
#         )

#         final_old_only = self.evaluate_loader(
#             old_reporting_loader, class_ids=old_ids, target_class_ids=old_ids, structures=final_state.migrated_historical
#         )
#         final_old_all_seen = self.evaluate_loader(
#             old_reporting_loader, class_ids=seen_ids, target_class_ids=old_ids, structures=final_state.seen
#         )
#         final_new_all_seen = self.evaluate_loader(
#             train_eval_loader, class_ids=seen_ids, target_class_ids=new_ids, structures=final_state.seen
#         )
#         final_new_only = self.evaluate_loader(
#             train_eval_loader, class_ids=new_ids, target_class_ids=new_ids, structures=final_state.current
#         )

#         with torch.no_grad():
#             final_boundary_objective = cds_regulator(
#                 classifier=classifier,
#                 seen_cds=final_state.seen_cds,
#                 historical_class_count=len(old_ids),
#             )
#         if not torch.equal(classifier.class_weight(old_ids_tensor), historical_classifier_rows):
#             raise RuntimeError("historical classifier rows changed during incremental optimization")

#         self.model.replace_class_structure(final_state.seen)
#         self.model.validate_finalized_state()
#         self.dataset.finalize_phase(phase_id)

#         cumulative_loader = self.dataset.get_reporting_dataloader(phase_id, batch_size=eval_batch_size)
#         cumulative = self.evaluate_loader(cumulative_loader, class_ids=seen_ids, target_class_ids=seen_ids)
#         groups = old_new_summary(cumulative, old_class_ids=old_ids, new_class_ids=new_ids)
#         forgetting = float(old_before["balanced_accuracy"]) - float(groups["old"]["balanced_accuracy"])
#         phase_origin = phase_origin_confusion_diagnostics(
#             cumulative,
#             phase_to_classes={index: self.dataset.phase_to_classes[index] for index in range(phase_id + 1)},
#         )

#         diag = final_state.alignment
#         method_state = {
#             "classifier": "single_bias_free_scaled_cosine_head",
#             "persistent_cds": "{I_Q,V_Q,I_P,V_P,N}; all inter-class CDS is derived on demand",
#             "historical_supervision": "HDC_plus_feature_side_old_rival_separation_plus_whole_CDS_classifier_region_protection",
#             "historical_cds_information": "I_Q locates historical Q regions; V_Q bounds absolute/relational extent; I_P,V_P prioritize old/current classifier conflicts through whole-CDS risk",
#             "representation_objective": "real_current_CE_plus_direct_Q_supervised_contrast_plus_HDC_plus_feature_only_old_rival_separation",
#             "classifier_objective": "old_CDS_to_current_row_absolute_and_relational_clearance_with_old_rows_bitwise_frozen",
#             "stores_pseudo_features": False,
#             "reconstructs_historical_features": False,
#             "uses_input_inversion": False,
#             "uses_stage_B": False,
#             "uses_covariance_classifier": False,
#             "semantic_shift_mapping_error": float(diag.task_mapping_error.item()),
#             "semantic_shift_channel_error_max": float(diag.task_mapping_channel_error_max.item()),
#             "feature_drift_rms": float(diag.representation_drift_rms.item()),
#             "task_translation_rms": float(diag.task_translation_rms.item()),
#             "task_scale_deviation_rms": float(diag.task_scale_deviation_rms.item()),
#             "reference_q_rms": float(reference_q_rms.item()),
#             "current_q_rms": float(self._mean_sample_rms(final_state.current_rows["ordered_features"]).item()),
#             "phase_start_old_ba": float(old_before["balanced_accuracy"]),
#             "phase_start_old_all_seen_ba": float(phase_start_old_all_seen["balanced_accuracy"]),
#             "final_old_only_ba": float(final_old_only["balanced_accuracy"]),
#             "final_old_all_seen_ba": float(final_old_all_seen["balanced_accuracy"]),
#             "representation_forgetting": float(old_before["balanced_accuracy"]) - float(final_old_only["balanced_accuracy"]),
#             "decision_interference_start": float(old_before["balanced_accuracy"]) - float(phase_start_old_all_seen["balanced_accuracy"]),
#             "decision_interference_final": float(final_old_only["balanced_accuracy"]) - float(final_old_all_seen["balanced_accuracy"]),
#             "final_new_train_all_seen_ba": float(final_new_all_seen["balanced_accuracy"]),
#             "final_new_train_only_ba": float(final_new_only["balanced_accuracy"]),
#             "classifier_logit_scale": float(classifier.logit_scale_value.detach().item()),
#             "final_classification_loss": float(history[-1]["classification_loss"]),
#             "final_contrastive_loss": float(history[-1]["contrastive_loss"]),
#             "final_contrastive_similarity_gap": float(history[-1]["contrastive_similarity_gap"]),
#             "final_hsi_cds_separation_loss": float(history[-1]["hsi_cds_separation_loss"]),
#             "final_hsi_cds_separation_margin_mean": float(history[-1]["hsi_cds_separation_margin_mean"]),
#             "final_historical_decision_consistency_loss": float(history[-1]["historical_decision_consistency_loss"]),
#             "final_cds_boundary_loss": float(final_boundary_objective.loss.item()),
#             "final_cds_boundary_clearance_mean": float(final_boundary_objective.structural_clearance_mean.item()),
#             "final_cds_boundary_clearance_min": float(final_boundary_objective.structural_clearance_min.item()),
#             "final_cds_boundary_absolute_clearance_mean": float(final_boundary_objective.absolute_clearance_mean.item()),
#             "final_cds_boundary_relational_clearance_mean": float(final_boundary_objective.relational_clearance_mean.item()),
#             "final_cds_boundary_unsafe_pair_fraction": float(final_boundary_objective.unsafe_pair_fraction.item()),
#             "final_cds_boundary_pair_weight_mean": float(final_boundary_objective.pair_weight_mean.item()),
#             "final_cds_boundary_learned_risk_mean": float(final_boundary_objective.learned_region_risk_mean.item()),
#             "final_cds_boundary_spectral_risk_mean": float(final_boundary_objective.spectral_region_risk_mean.item()),
#             "historical_classifier_rows_frozen": True,
#         }

#         return {
#             "phase": phase_id,
#             "old_class_ids": list(old_ids),
#             "new_class_ids": list(new_ids),
#             "seen_class_ids": list(seen_ids),
#             "history": history,
#             "final_epoch": epoch_count,
#             "cumulative_test": cumulative,
#             "old_new": groups,
#             "forgetting": forgetting,
#             "method_state": method_state,
#             "phase_origin_confusion": phase_origin,
#             "structure_committed": True,
#             "phase_finalized": True,
#         }


# __all__ = ["IncrementalPhaseTrainer"]
