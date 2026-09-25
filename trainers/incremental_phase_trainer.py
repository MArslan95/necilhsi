from __future__ import annotations

"""One-stage incremental training for the coherent Whole-CDS NECIL-HSI architecture.

Problem-to-mechanism contract
-----------------------------
* semantic shift: immutable historical CDS is the reference; the existing phase
  alignment builds a detached Whole-CDS-conditioned working historical CDS in
  the current Q coordinate;
* feature stability: ordered feature KD + old-logit KD + fixed-reference
  Whole-CDS preservation;
* feature plasticity: real-current CE on incoming classes only;
* cross-phase separation: one risk-aware Whole-CDS contrastive objective uses
  I_Q/V_Q and I_P/V_P directly;
* classifier bias: finalized historical classifier rows remain fixed while the
  existing Whole-CDS decision-region regulator positions incoming rows against
  those anchors using the SAME cosine decision function used at inference.

Persistent method memory is only {I_Q,V_Q,I_P,V_P}. No old images, pseudo HSI,
model inversion, pseudo-feature replay, second representation, or second
classifier is introduced.
"""
import math
import sys
from numbers import Integral, Real
from typing import Any, Dict

import torch
from tqdm import tqdm

from losses.loss import (
    classification_objective,
    whole_cds_contrastive_loss,
    historical_cds_preservation_loss,
    logit_distillation_loss,
    ordered_feature_distillation_loss,
)
from models.classifier import CosineClassificationHead
from models.cds_decision_region import (
    HSICDSDecisionRegionRegulator,
)
from models.geometry_bank import ClassStructureBatch, merge_class_structure_batches
from models.phase_coordinate_alignment import correct_historical_structure
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


def _clone_structure(structure: ClassStructureBatch) -> ClassStructureBatch:
    if not isinstance(structure, ClassStructureBatch) or structure.class_count <= 0:
        raise ValueError("cannot clone an empty class structure")
    return ClassStructureBatch(
        class_ids=structure.class_ids.detach().clone(),
        sample_counts=structure.sample_counts.detach().clone(),
        learned_identity=structure.learned_identity.detach().clone(),
        learned_variation=structure.learned_variation.detach().clone(),
        physical_identity=structure.physical_identity.detach().clone(),
        physical_variation=structure.physical_variation.detach().clone(),
    )


def _targets_local(labels: Tensor, class_ids: Tensor) -> Tensor:
    matches = labels.unsqueeze(1).eq(class_ids.unsqueeze(0))
    if not bool(matches.sum(dim=1).eq(1).all()):
        raise ValueError("a current label is absent from current_class_ids")
    return matches.to(torch.long).argmax(dim=1)


class IncrementalPhaseTrainer:
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
            "structure_committed",
            "phase_finalized",
        }
    )


    def _require_classifier(self) -> CosineClassificationHead:
        classifier = getattr(self.model, "classifier", None)
        if not isinstance(classifier, CosineClassificationHead):
            raise RuntimeError("model.classifier must be CosineClassificationHead")
        return classifier

    def _phase_classes(
        self,
        phase: int,
    ) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
        old_class_ids = tuple(
            self._validated_class_ids(
                self.dataset.get_old_classes(phase), name=f"phase_{phase}_old_classes"
            )
        )
        new_class_ids = tuple(
            self._validated_class_ids(
                self.dataset.get_new_classes(phase), name=f"phase_{phase}_new_classes"
            )
        )
        seen_class_ids = tuple(
            self._validated_class_ids(
                self.dataset.get_seen_classes(phase), name=f"phase_{phase}_seen_classes"
            )
        )
        if not old_class_ids or not new_class_ids:
            raise RuntimeError("incremental phase requires old and new classes")
        if set(old_class_ids).intersection(new_class_ids):
            raise RuntimeError("old and new classes overlap")
        if seen_class_ids != old_class_ids + new_class_ids:
            raise RuntimeError("seen classes must equal old classes followed by new classes")
        if tuple(self.model.committed_class_ids) != old_class_ids:
            raise RuntimeError("committed historical CDS does not match old classes")
        return old_class_ids, new_class_ids, seen_class_ids

    @staticmethod
    def _assert_frozen_reference(reference_backbone: torch.nn.Module) -> None:
        reference_backbone.eval()
        if any(parameter.requires_grad for parameter in reference_backbone.parameters()):
            raise RuntimeError("previous backbone reference must remain frozen")
        if any(parameter.grad is not None for parameter in reference_backbone.parameters()):
            raise RuntimeError("previous backbone reference must not carry gradients")

    @staticmethod
    def _assert_frozen_reference_classifier(
        reference_classifier: CosineClassificationHead,
    ) -> None:
        reference_classifier.eval()
        if any(parameter.requires_grad for parameter in reference_classifier.parameters()):
            raise RuntimeError("previous classifier reference must remain frozen")
        if any(parameter.grad is not None for parameter in reference_classifier.parameters()):
            raise RuntimeError("previous classifier reference must not carry gradients")

    @staticmethod
    @torch.no_grad()
    def _assert_reference_matches_phase_start(
        reference: torch.nn.Module,
        current: torch.nn.Module,
        *,
        name: str,
    ) -> None:
        ref_state = reference.state_dict()
        cur_state = current.state_dict()
        if tuple(ref_state.keys()) != tuple(cur_state.keys()):
            raise RuntimeError(f"previous/current {name} schemas differ at phase start")
        for key in ref_state:
            if not torch.equal(ref_state[key].detach().cpu(), cur_state[key].detach().cpu()):
                raise RuntimeError(
                    f"previous {name} does not match finalized phase-start state at {key}"
                )

    @torch.no_grad()
    def _current_class_weights(
        self,
        loader: Any,
        *,
        new_class_ids: tuple[int, ...],
    ) -> tuple[Tensor, int]:
        counts = {class_id: 0 for class_id in new_class_ids}
        total = 0
        valid = set(new_class_ids)
        for batch in loader:
            _, _, labels = self.unpack_batch(batch)
            observed = set(int(value) for value in labels.unique().detach().cpu().tolist())
            outside = observed - valid
            if outside:
                raise RuntimeError(
                    f"incremental TRAIN loader contains non-current classes {sorted(outside)}"
                )
            for class_id in new_class_ids:
                count = int(labels.eq(class_id).sum().item())
                counts[class_id] += count
                total += count
        if total <= 0 or any(counts[class_id] <= 0 for class_id in new_class_ids):
            raise RuntimeError("incremental TRAIN split is empty or misses a current class")
        class_count = len(new_class_ids)
        weights = torch.tensor(
            [total / float(class_count * counts[class_id]) for class_id in new_class_ids],
            device=self.device,
            dtype=self.model.dtype,
        )
        return weights, total

    @staticmethod
    def _max_parameter_change(before: Tensor, after: Tensor) -> float:
        if before.shape != after.shape:
            raise ValueError("parameter snapshots have different shapes")
        if before.numel() == 0:
            return 0.0
        return float((after - before).abs().max().item())

    @staticmethod
    def _assert_future_rows_unchanged(
        classifier: CosineClassificationHead,
        future_class_tensor: Tensor,
        future_weight_before: Tensor | None,
    ) -> None:
        if future_weight_before is None or future_class_tensor.numel() == 0:
            return
        current = classifier.class_weight(future_class_tensor).detach()
        if not torch.equal(current, future_weight_before):
            raise RuntimeError("future classifier rows changed before their phase")

    @staticmethod
    def _row_grad_max(gradient: Tensor | None, class_ids: Tensor) -> float:
        if gradient is None or class_ids.numel() == 0:
            return 0.0
        rows = gradient.index_select(0, class_ids)
        return float(rows.abs().max().detach().item()) if rows.numel() else 0.0

    def train_incremental_phase(
        self,
        *,
        phase: int,
        epochs: int,
        batch_size: int,
        lr: float,
        reference_backbone: torch.nn.Module,
        reference_classifier: CosineClassificationHead,
    ) -> Dict[str, Any]:
        phase_id = _positive_int(phase, name="phase")
        epoch_count = _positive_int(epochs, name="epochs")
        train_batch_size = _positive_int(batch_size, name="batch_size")
        learning_rate = _finite_positive("lr_inc", lr)
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
        kd_logit_weight = _finite_nonnegative(
            "kd_logit_weight", self.cfg("kd_logit_weight", 1.0)
        )
        kd_temperature = _finite_positive(
            "kd_temperature", self.cfg("kd_temperature", 2.0)
        )
        # One CDS weight owns both structural preservation and classifier-region
        # protection.  This removes the previous parser/trainer inconsistency where
        # a deprecated ``cds_classifier_weight=0`` could silently disable the very
        # classifier protection the architecture relied on.
        cds_weight = _finite_nonnegative(
            "cds_weight", self.cfg("cds_weight", 1.0)
        )
        feature_kd_weight = _finite_nonnegative(
            "feature_kd_weight", self.cfg("feature_kd_weight", 10.0)
        )
        
        self.model.validate_finalized_state()
        classifier = self._require_classifier()
        self._assert_frozen_reference(reference_backbone)
        self._assert_frozen_reference_classifier(reference_classifier)
        self._assert_reference_matches_phase_start(
            reference_backbone, self.model.backbone, name="backbone"
        )
        self._assert_reference_matches_phase_start(
            reference_classifier, classifier, name="classifier"
        )

        previous_phase = phase_id - 1
        previous_seen_class_ids = tuple(
            self._validated_class_ids(
                self.dataset.get_seen_classes(previous_phase),
                name="previous_seen_classes",
            )
        )
        old_reporting_loader = self.dataset.get_reporting_dataloader(
            previous_phase, batch_size=eval_batch_size
        )
        old_before = self.evaluate_loader(
            old_reporting_loader,
            class_ids=previous_seen_class_ids,
            target_class_ids=previous_seen_class_ids,
        )

        self.dataset.start_phase(phase_id)
        old_class_ids, new_class_ids, seen_class_ids = self._phase_classes(phase_id)
        old_class_tensor = torch.tensor(old_class_ids, device=self.device, dtype=torch.long)
        new_class_tensor = torch.tensor(new_class_ids, device=self.device, dtype=torch.long)
        seen_class_tensor = torch.tensor(seen_class_ids, device=self.device, dtype=torch.long)

        historical_structure = _clone_structure(
            self.model.structure_bank.class_structures(old_class_ids)
        )
        reference_historical_cds = self.model.structure_bank.build_cds(historical_structure)

        self.model.set_incremental_trainability()
        backbone_parameters = [
            parameter
            for parameter in self.model.backbone.parameters()
            if parameter.requires_grad
        ]
        if not backbone_parameters:
            raise RuntimeError("incremental backbone has no trainable parameters")

        train_eval_loader = self.dataset.get_phase_dataloader(
            phase_id, split="train", batch_size=eval_batch_size, shuffle=False
        )
        current_class_weights, expected_samples = self._current_class_weights(
            train_eval_loader, new_class_ids=new_class_ids
        )
        reference_train = self.collect_ordered_features(
            train_eval_loader, backbone=reference_backbone
        )
        reference_features_full = reference_train["ordered_features"].detach()
        reference_spectra = reference_train["raw_center_spectra"].detach()
        reference_labels = reference_train["labels"].detach()
        observed_new = tuple(
            sorted(int(value) for value in reference_labels.unique().detach().cpu().tolist())
        )
        if observed_new != tuple(sorted(new_class_ids)):
            raise RuntimeError("incremental TRAIN rows do not match current phase classes")

        current_physical_structure = self.model.estimate_physical_class_structure(
            reference_spectra, reference_labels, class_ids=new_class_ids
        )
        current_reference_structure = self.model.estimate_class_structure(
            reference_features_full,
            None,
            reference_labels,
            class_ids=new_class_ids,
            physical_structure=current_physical_structure,
        )
        # Snapshot classifier ownership before Phase-t optimization.  Incoming
        # rows retain their native classifier initialization from model creation;
        # Whole-CDS is supervision, never an initializer or a pre-SGD repair.
        old_weight_before = classifier.class_weight(old_class_tensor).detach().clone()
        new_weight_phase_start = classifier.class_weight(new_class_tensor).detach().clone()
        seen_set = set(seen_class_ids)
        future_class_ids = tuple(
            class_id
            for class_id in range(classifier.total_class_count)
            if class_id not in seen_set
        )
        future_class_tensor = torch.tensor(
            future_class_ids, device=self.device, dtype=torch.long
        )
        future_weight_before = (
            classifier.class_weight(future_class_tensor).detach().clone()
            if future_class_ids
            else None
        )

        scale_requires_grad = bool(classifier.logit_scale_parameter.requires_grad)
        classifier.logit_scale_parameter.requires_grad_(False)
        classifier.logit_scale_parameter.grad = None
        classifier.weight.requires_grad_(True)
        phase_start_scale = float(classifier.logit_scale.detach().item())

        train_loader = self.dataset.get_phase_dataloader(
            phase_id, split="train", batch_size=train_batch_size, shuffle=True
        )
        val_loader = (
            self.dataset.get_phase_dataloader(
                phase_id, split="val", batch_size=eval_batch_size, shuffle=False
            )
            if self.dataset.validation_enabled
            else None
        )

        # Phase-setup invariance gate.  Starting a phase, toggling trainability,
        # constructing CDS views, or exposing future classifier capacity must not
        # change the already-finalized old-only decision rule before optimization.
        phase_setup_old_only = self.evaluate_loader(
            old_reporting_loader,
            class_ids=old_class_ids,
            target_class_ids=old_class_ids,
        )
        if phase_setup_old_only["confusion_matrix"] != old_before["confusion_matrix"]:
            raise RuntimeError(
                "Phase setup changed old-only predictions before optimization"
            )

        phase_start_new_only = self.evaluate_loader(
            train_eval_loader, class_ids=new_class_ids, target_class_ids=new_class_ids
        )
        phase_start_new_all_seen = self.evaluate_loader(
            train_eval_loader, class_ids=seen_class_ids, target_class_ids=new_class_ids
        )
        phase_start_old_all_seen = self.evaluate_loader(
            old_reporting_loader, class_ids=seen_class_ids, target_class_ids=old_class_ids
        )
        print(
            f"[Phase {phase_id} start] "
            f"OldBA(previous-only)={100*float(old_before['balanced_accuracy']):.2f}% "
            f"OldBA(all-seen)={100*float(phase_start_old_all_seen['balanced_accuracy']):.2f}% "
            f"NewBA(new-only)={100*float(phase_start_new_only['balanced_accuracy']):.2f}% "
            f"NewBA(all-seen)={100*float(phase_start_new_all_seen['balanced_accuracy']):.2f}% | "
            "Init=native-classifier-rows; historical CDS enters only through joint losses",
            file=sys.stdout,
        )

        @torch.no_grad()
        def refresh_current_structure():
            """Build one coherent all-seen CDS in the CURRENT Q coordinate.

            The finalized Phase-(t-1) ``historical_structure`` remains immutable
            and is the preservation reference.  ``correct_historical_structure``
            uses only paired real current-class Q^-/Q^+ observations plus the
            already-stored Whole-CDS relevance to construct a detached working
            historical CDS in Q_t.  Current-class CDS is estimated from real
            current TRAIN data.  The two are then merged in one coordinate for
            separation and classifier supervision.
            """
            self.model.eval()
            rows = self.collect_ordered_features(train_eval_loader)
            if not torch.equal(rows["labels"], reference_labels):
                raise RuntimeError("current TRAIN label order changed during CDS refresh")
            if not torch.equal(rows["raw_center_spectra"], reference_spectra):
                raise RuntimeError("current TRAIN spectra changed during CDS refresh")

            current_features_full = rows["ordered_features"].detach()
            current_structure = self.model.estimate_class_structure(
                current_features_full,
                None,
                reference_labels,
                class_ids=new_class_ids,
                physical_structure=current_physical_structure,
            )

            working_historical_structure, _transform, alignment = (
                correct_historical_structure(
                    historical_structure=historical_structure,
                    current_reference_structure=current_reference_structure,
                    current_structure=current_structure,
                    reference_features=reference_features_full,
                    current_features=current_features_full,
                )
            )
            if not torch.equal(
                working_historical_structure.physical_identity,
                historical_structure.physical_identity,
            ) or not torch.equal(
                working_historical_structure.physical_variation,
                historical_structure.physical_variation,
            ):
                raise RuntimeError("working historical CDS changed physical I_P/V_P")

            merged = merge_class_structure_batches(
                (working_historical_structure, current_structure)
            )
            merged_ids = tuple(
                int(v) for v in merged.class_ids.detach().cpu().tolist()
            )
            if merged_ids != seen_class_ids:
                raise RuntimeError("working CDS produced incorrect seen-class order")
            seen_cds = self.model.structure_bank.build_cds(merged)
            return (
                current_features_full,
                current_structure,
                working_historical_structure,
                merged,
                seen_cds,
                alignment,
            )

        # One optimizer, one scheduler, one phase loop.  Classifier weight decay is
        # zero so masked old/future rows are not silently changed by AdamW decay.
        optimizer = torch.optim.AdamW(
            [
                {"params": backbone_parameters, "weight_decay": weight_decay},
                {"params": [classifier.weight], "weight_decay": 0.0},
            ],
            lr=learning_rate,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epoch_count, eta_min=0.0
        )
        regulator = HSICDSDecisionRegionRegulator().to(self.device)

        joint_state = refresh_current_structure()
        pre_structural = regulator(
            classifier=classifier,
            seen_cds=joint_state[4],
            historical_class_count=len(old_class_ids),
            reference_classifier=reference_classifier,
            reference_historical_cds=reference_historical_cds,
            trainable_class_ids=new_class_tensor,
        )

        history: list[dict[str, Any]] = []
        gradient_contract_checked = False
        progress = tqdm(
            range(epoch_count),
            desc=f"Phase {phase_id} joint CDS-cosine training",
            unit="epoch",
            dynamic_ncols=True,
            file=sys.stdout,
        )

        for epoch in progress:
            (
                _epoch_features,
                _epoch_current_structure,
                epoch_historical_structure,
                _epoch_merged,
                epoch_seen_cds,
                _epoch_alignment,
            ) = joint_state

            self.model.train()
            sums = {name: 0.0 for name in (
                "classification", "feature_kd", "feature_rms",
                "cds_classifier", "cds_classifier_old", "cds_classifier_new", "pic", "kd",
                "cds", "intra", "inter", "physical_abs",
                "physical_rel", "physical_joint", "pic_margin",
                "pic_violation", "pic_positive", "pic_current_negative",
                "pic_historical_negative", "pic_valid", "pic_physical_confusability",
                "pic_pair_risk", "structural", "old_old", "total"
            )}
            sample_count = 0
            correct_all_seen = 0

            for batch in train_loader:
                patch, spectrum, labels = self.unpack_batch(batch)
                if not bool(torch.isin(labels, new_class_tensor).all()):
                    raise RuntimeError("incremental TRAIN batch contains an old or future class")

                optimizer.zero_grad(set_to_none=True)
                classifier.logit_scale_parameter.grad = None

                with torch.no_grad():
                    reference_features = self.model.extract_features_with_backbone(
                        reference_backbone,
                        patch,
                        center_spectrum=spectrum,
                        return_aux=False,
                    ).ordered_features.detach()
                    physical_evidence = self.model.physical_spectral_evidence(spectrum).detach()
                    teacher_old_logits = reference_classifier(
                        reference_features,
                        class_ids=old_class_tensor,
                        trainable_class_ids=(),
                    ).logits.detach()

                current_features = self.model.extract_features(
                    patch, center_spectrum=spectrum, return_aux=False
                ).ordered_features
                current_local = _targets_local(labels, new_class_tensor)
                sample_weights = current_class_weights.index_select(0, current_local)

                # Real current labels train ONLY the incoming classifier rows.
                # Historical rows have no real positive samples in this phase and
                # therefore must not be rewritten by current-only CE.  Cross-phase
                # old/new competition is handled by the existing Whole-CDS
                # decision-region regulator with frozen historical anchors.
                new_output = classifier(
                    current_features,
                    class_ids=new_class_tensor,
                    trainable_class_ids=new_class_tensor,
                )
                classification = classification_objective(
                    logits=new_output.logits,
                    class_ids=new_output.class_ids,
                    labels_global=labels,
                    sample_weights=sample_weights,
                )

                # Read-only deployed decision rule for online all-seen accuracy.
                all_seen_output = classifier(
                    current_features,
                    class_ids=seen_class_tensor,
                    trainable_class_ids=(),
                )
                # Direct feature KD is the missing stability mechanism exposed by
                # the IP run: old-only BA fell 91.70 -> 73.47 even though old/new
                # separation improved.  This constrains the one active Q space,
                # exactly as FCS constrains semantic feature drift.
                feature_kd = ordered_feature_distillation_loss(
                    reference_features=reference_features,
                    current_features=current_features,
                    sample_weights=sample_weights,
                )

                student_old_logits = classifier(
                    current_features,
                    class_ids=old_class_tensor,
                    trainable_class_ids=(),
                ).logits
                kd = logit_distillation_loss(
                    student_old_logits=student_old_logits,
                    teacher_old_logits=teacher_old_logits,
                    temperature=kd_temperature,
                    sample_weights=sample_weights,
                )

                cds = historical_cds_preservation_loss(
                    reference_features=reference_features,
                    current_features=current_features,
                    physical_spectral_evidence=physical_evidence,
                    reference_historical_structure=historical_structure,
                    current_historical_structure=epoch_historical_structure,
                )
                # One continual separation objective: current/current SupCon plus
                # direct all-seen Whole-CDS discrimination. I_Q, V_Q, I_P and V_P
                # are all inside the optimized class score, not only diagnostics.
                pic = whole_cds_contrastive_loss(
                    current_features=current_features,
                    labels_global=labels,
                    physical_spectral_evidence=physical_evidence,
                    seen_cds=epoch_seen_cds,
                    historical_class_count=len(old_class_ids),
                    logit_scale=classifier.logit_scale,
                    sample_weights=sample_weights,
                )

                # Symmetric all-seen Whole-CDS decision-region protection.  No
                # source/rival row is detached and old-old clearance is retained
                # relative to the finalized previous classifier.
                structural = regulator(
                    classifier=classifier,
                    seen_cds=epoch_seen_cds,
                    historical_class_count=len(old_class_ids),
                    reference_classifier=reference_classifier,
                    reference_historical_cds=reference_historical_cds,
                    trainable_class_ids=new_class_tensor,
                )
                if not bool(torch.isfinite(structural.loss)):
                    raise RuntimeError(
                        "symmetric Whole-CDS classifier supervision became NaN or Inf"
                    )

                if not gradient_contract_checked:
                    tolerance = 64.0 * torch.finfo(classifier.weight.dtype).eps

                    ce_grad = torch.autograd.grad(
                        classification.loss, classifier.weight,
                        retain_graph=True, allow_unused=True,
                    )[0]
                    ce_old = self._row_grad_max(ce_grad, old_class_tensor)
                    ce_new = self._row_grad_max(ce_grad, new_class_tensor)
                    ce_future = self._row_grad_max(ce_grad, future_class_tensor)
                    if ce_old > tolerance:
                        raise RuntimeError(
                            "real-current CE leaked gradient into finalized historical rows"
                        )
                    if ce_new <= tolerance:
                        raise RuntimeError(
                            "real-current CE produced no incoming-row gradient"
                        )
                    if ce_future > tolerance:
                        raise RuntimeError("real-current CE leaked into future rows")

                    cds_grad = torch.autograd.grad(
                        structural.loss, classifier.weight,
                        retain_graph=True, allow_unused=True,
                    )[0]
                    cds_old = self._row_grad_max(cds_grad, old_class_tensor)
                    cds_new = self._row_grad_max(cds_grad, new_class_tensor)
                    cds_future = self._row_grad_max(cds_grad, future_class_tensor)
                    if cds_old > tolerance:
                        raise RuntimeError(
                            "Whole-CDS classifier protection leaked into finalized historical rows"
                        )
                    if cds_new <= tolerance:
                        raise RuntimeError(
                            "Whole-CDS classifier protection produced no incoming-row gradient"
                        )
                    if cds_future > tolerance:
                        raise RuntimeError("Whole-CDS supervision leaked into future rows")

                    for name, objective in (
                        ("feature KD", feature_kd.loss),
                        ("Whole-CDS contrastive", pic.loss),
                    ):
                        grad = torch.autograd.grad(
                            objective, classifier.weight, retain_graph=True, allow_unused=True
                        )[0]
                        if grad is not None and bool(torch.count_nonzero(grad)):
                            raise RuntimeError(f"{name} must not update classifier rows")

                    # First-batch numerical contract.  A finite scalar loss can still
                    # have an undefined derivative (the original sqrt(MSE) feature-KD
                    # bug did exactly that at zero drift).  Check every objective that
                    # acts on current Q before allowing the combined backward pass.
                    for name, objective in (
                        ("current new-class CE", classification.loss),
                        ("feature KD", feature_kd.loss),
                        ("old-logit KD", kd.loss),
                        ("intra/inter CDS preservation", cds.loss),
                        ("Whole-CDS contrastive", pic.loss),
                    ):
                        q_grad = torch.autograd.grad(
                            objective, current_features, retain_graph=True, allow_unused=True
                        )[0]
                        if q_grad is None:
                            raise RuntimeError(f"{name} produced no current-Q gradient")
                        if not bool(torch.isfinite(q_grad).all()):
                            raise RuntimeError(
                                f"{name} produced NaN/Inf current-Q gradient before joint backward"
                            )

                    print(
                        "[Gradient ownership] "
                        f"NewCE old={ce_old:.3e} new={ce_new:.3e} future={ce_future:.3e} | "
                        f"WholeCDS old={cds_old:.3e} new={cds_new:.3e} future={cds_future:.3e} | "
                        "HistoricalW=frozen | FeatureKD classifier=0 | WholeCDSContrast classifier=0",
                        file=sys.stdout,
                    )
                    gradient_contract_checked = True

                loss = (
                    classification.loss
                    + contrastive_weight * pic.loss
                    + feature_kd_weight * feature_kd.loss
                    + kd_logit_weight * kd.loss
                    + cds_weight * (cds.loss + structural.loss)
                )
                if not bool(torch.isfinite(loss)):
                    raise RuntimeError("joint incremental objective became NaN or Inf")
                loss.backward()

                if classifier.logit_scale_parameter.grad is not None:
                    raise RuntimeError("incremental training must not update cosine logit scale")
                if classifier.weight.grad is None or not bool(
                    torch.isfinite(classifier.weight.grad).all()
                ):
                    raise RuntimeError("joint classifier gradient is missing/non-finite")
                if bool(
                    torch.count_nonzero(
                        classifier.weight.grad.index_select(0, old_class_tensor)
                    )
                ):
                    raise RuntimeError(
                        "finalized historical classifier rows received incremental gradients"
                    )
                if future_class_tensor.numel() and bool(
                    torch.count_nonzero(
                        classifier.weight.grad.index_select(0, future_class_tensor)
                    )
                ):
                    raise RuntimeError("future classifier rows received gradients")

                backbone_grad_seen = False
                for parameter in backbone_parameters:
                    if parameter.grad is None:
                        continue
                    if not bool(torch.isfinite(parameter.grad).all()):
                        raise RuntimeError("incremental backbone gradient contains NaN or Inf")
                    if bool(torch.count_nonzero(parameter.grad)):
                        backbone_grad_seen = True
                if not backbone_grad_seen:
                    raise RuntimeError("joint objective produced zero backbone gradient")

                if gradient_clip > 0:
                    torch.nn.utils.clip_grad_norm_(
                        backbone_parameters + [classifier.weight],
                        gradient_clip,
                        error_if_nonfinite=True,
                    )
                optimizer.step()
                if not torch.equal(
                    classifier.class_weight(old_class_tensor).detach(), old_weight_before
                ):
                    raise RuntimeError(
                        "finalized historical classifier rows changed during incremental training"
                    )
                self._assert_future_rows_unchanged(
                    classifier, future_class_tensor, future_weight_before
                )

                n = int(labels.numel())
                sums["classification"] += float(classification.loss.detach().item()) * n
                sums["feature_kd"] += float(feature_kd.loss.detach().item()) * n
                sums["feature_rms"] += float(feature_kd.rms_drift.detach().item()) * n
                sums["cds_classifier"] += float(structural.loss.detach().item()) * n
                sums["cds_classifier_old"] += float(
                    structural.old_class_reconstruction_loss.detach().item()
                ) * n
                sums["cds_classifier_new"] += float(
                    structural.new_class_reconstruction_loss.detach().item()
                ) * n
                sums["pic"] += float(pic.loss.detach().item()) * n
                sums["kd"] += float(kd.loss.detach().item()) * n
                sums["cds"] += float(cds.loss.detach().item()) * n
                sums["intra"] += float(cds.intra_loss.detach().item()) * n
                sums["inter"] += float(cds.inter_loss.detach().item()) * n
                sums["physical_abs"] += float(
                    cds.physical_absolute_support_mean.detach().item()
                ) * n
                sums["physical_rel"] += float(
                    cds.physical_relation_support_mean.detach().item()
                ) * n
                sums["physical_joint"] += float(
                    cds.physical_relevance_mean.detach().item()
                ) * n
                sums["pic_margin"] += float(
                    pic.learned_support_margin_mean.detach().item()
                ) * n
                sums["pic_violation"] += float(
                    pic.violation_fraction.detach().item()
                ) * n
                sums["pic_positive"] += float(
                    pic.positive_similarity_mean.detach().item()
                ) * n
                sums["pic_current_negative"] += float(
                    pic.current_negative_similarity_mean.detach().item()
                ) * n
                sums["pic_historical_negative"] += float(
                    pic.historical_negative_similarity_mean.detach().item()
                ) * n
                sums["pic_valid"] += float(
                    pic.valid_anchor_fraction.detach().item()
                ) * n
                sums["pic_physical_confusability"] += float(
                    pic.physical_confusability_mean.detach().item()
                ) * n
                sums["pic_pair_risk"] += float(
                    pic.effective_pair_risk_mean.detach().item()
                ) * n
                sums["structural"] += float(structural.loss.detach().item()) * n
                sums["old_old"] += float(structural.old_old_retention_loss.detach().item()) * n
                sums["total"] += float(loss.detach().item()) * n
                correct_all_seen += int(all_seen_output.prediction.detach().eq(labels).sum().item())
                sample_count += n

            if sample_count != expected_samples:
                raise RuntimeError("incremental TRAIN loader did not traverse the complete split")
            scheduler.step()
            if float(classifier.logit_scale.detach().item()) != phase_start_scale:
                raise RuntimeError("cosine logit scale changed during incremental training")
            self._assert_future_rows_unchanged(
                classifier, future_class_tensor, future_weight_before
            )

            # Refresh current-class CDS and its detached working historical
            # alignment for the next epoch. The persistent historical reference
            # itself remains immutable.
            joint_state = refresh_current_structure()
            epoch_structural = regulator(
                classifier=classifier,
                seen_cds=joint_state[4],
                historical_class_count=len(old_class_ids),
                reference_classifier=reference_classifier,
                reference_historical_cds=reference_historical_cds,
                trainable_class_ids=new_class_tensor,
            )

            self.model.eval()
            old_only = self.evaluate_loader(
                old_reporting_loader,
                class_ids=old_class_ids,
                target_class_ids=old_class_ids,
            )
            old_all = self.evaluate_loader(
                old_reporting_loader,
                class_ids=seen_class_ids,
                target_class_ids=old_class_ids,
            )
            new_only = self.evaluate_loader(
                train_eval_loader,
                class_ids=new_class_ids,
                target_class_ids=new_class_ids,
            )
            new_all = self.evaluate_loader(
                train_eval_loader,
                class_ids=seen_class_ids,
                target_class_ids=new_class_ids,
            )
            validation_all = (
                None
                if val_loader is None
                else self.evaluate_loader(
                    val_loader,
                    class_ids=seen_class_ids,
                    target_class_ids=new_class_ids,
                )
            )

            record = {
                "stage": "joint",
                "epoch": epoch + 1,
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "classification_loss": sums["classification"] / sample_count,
                "feature_kd_loss": sums["feature_kd"] / sample_count,
                "whole_cds_classifier_loss": sums["cds_classifier"] / sample_count,
                "old_cds_classifier_loss": sums["cds_classifier_old"] / sample_count,
                "new_cds_classifier_loss": sums["cds_classifier_new"] / sample_count,
                "current_all_seen_accuracy": correct_all_seen / float(sample_count),
                "pic_loss": sums["pic"] / sample_count,
                "contrastive_loss": sums["pic"] / sample_count,
                "kd_logit_loss": sums["kd"] / sample_count,
                "cds_preservation_loss": sums["cds"] / sample_count,
                "cds_separation_loss": sums["pic"] / sample_count,
                "cds_separation_margin": sums["pic_margin"] / sample_count,
                "cds_separation_violation_fraction": sums["pic_violation"] / sample_count,
                "pic_positive_similarity": sums["pic_positive"] / sample_count,
                "pic_current_negative_similarity": sums["pic_current_negative"] / sample_count,
                "pic_historical_negative_similarity": sums["pic_historical_negative"] / sample_count,
                "pic_valid_anchor_fraction": sums["pic_valid"] / sample_count,
                "pic_physical_confusability_mean": sums["pic_physical_confusability"] / sample_count,
                "pic_effective_pair_risk_mean": sums["pic_pair_risk"] / sample_count,
                "cds_intra_loss": sums["intra"] / sample_count,
                "cds_inter_loss": sums["inter"] / sample_count,
                "physical_absolute_support_mean": sums["physical_abs"] / sample_count,
                "physical_relation_support_mean": sums["physical_rel"] / sample_count,
                "physical_relevance_mean": sums["physical_joint"] / sample_count,
                "structural_loss": sums["structural"] / sample_count,
                "old_old_retention_loss": sums["old_old"] / sample_count,
                "total_loss": sums["total"] / sample_count,
                "old_only_ba": float(old_only["balanced_accuracy"]),
                "old_all_seen_ba": float(old_all["balanced_accuracy"]),
                "new_only_ba": float(new_only["balanced_accuracy"]),
                "new_all_seen_ba": float(new_all["balanced_accuracy"]),
                "validation_new_all_seen_ba": None
                if validation_all is None
                else float(validation_all["balanced_accuracy"]),
                "old_new_violation_fraction": float(
                    epoch_structural.old_new_violation_fraction.detach().item()
                ),
                "old_old_violation_fraction": float(
                    epoch_structural.historical_internal_violation_fraction.detach().item()
                ),
                "joint_clearance_mean": float(
                    epoch_structural.joint_structural_clearance_mean.detach().item()
                ),
                "joint_clearance_min": float(
                    epoch_structural.joint_structural_clearance_min.detach().item()
                ),
                "epoch_representation_drift_rms": sums["feature_rms"] / sample_count,
            }
            history.append(record)
            tqdm.write(
                f"[Joint {epoch + 1:03d}/{epoch_count:03d}] "
                f"CEnew={record['classification_loss']:.4f} "
                f"FKD={record['feature_kd_loss']:.4f} "
                f"CDSCls={record['whole_cds_classifier_loss']:.4f} "
                f"PIC={record['pic_loss']:.4f} "
                f"KD={record['kd_logit_loss']:.4f} "
                f"CDS={record['cds_preservation_loss']:.4f} "
                f"Sep={record['cds_separation_loss']:.4f}"
                f"[I={record['cds_intra_loss']:.4f},R={record['cds_inter_loss']:.4f}] "
                f" SepMargin={record['cds_separation_margin']:.4f} "
                f"SepViol={100*record['cds_separation_violation_fraction']:.1f}% "
                f"Struct={record['structural_loss']:.4f} | "
                f"OldOnly/All={100*record['old_only_ba']:.2f}%/"
                f"{100*record['old_all_seen_ba']:.2f}% "
                f"NewOnly/All={100*record['new_only_ba']:.2f}%/"
                f"{100*record['new_all_seen_ba']:.2f}% "
                f"OldNewViol={100*record['old_new_violation_fraction']:.1f}% "
                f"OldOldViol={100*record['old_old_violation_fraction']:.1f}%",
                file=sys.stdout,
            )

        progress.close()
        if not history:
            raise RuntimeError("joint incremental training produced no epochs")

        self.model.eval()
        (
            final_features_full,
            final_current_structure,
            final_historical_structure,
            committed_structure,
            committed_seen_cds,
            final_alignment,
        ) = joint_state

        if not torch.equal(
            final_historical_structure.physical_identity,
            historical_structure.physical_identity,
        ) or not torch.equal(
            final_historical_structure.physical_variation,
            historical_structure.physical_variation,
        ):
            raise RuntimeError("final working CDS changed physical I_P/V_P")

        # Final diagnostics use the exact committed current-coordinate CDS, the
        # same semantic state used during the final joint epoch.
        final_structural = regulator(
            classifier=classifier,
            seen_cds=committed_seen_cds,
            historical_class_count=len(old_class_ids),
            reference_classifier=reference_classifier,
            reference_historical_cds=reference_historical_cds,
            trainable_class_ids=new_class_tensor,
        )

        # Commit only corrected/real CDS.  The cosine classifier is already the
        # same classifier optimized during the one joint stage; no reset follows.
        self.model.replace_class_structure(committed_structure)
        self.model.validate_finalized_state()
        self.dataset.finalize_phase(phase_id)
        classifier.logit_scale_parameter.requires_grad_(scale_requires_grad)
        classifier.logit_scale_parameter.grad = None

        cumulative_loader = self.dataset.get_reporting_dataloader(
            phase_id, batch_size=eval_batch_size
        )
        cumulative = self.evaluate_loader(
            cumulative_loader,
            class_ids=seen_class_ids,
            target_class_ids=seen_class_ids,
        )
        groups = old_new_summary(
            cumulative, old_class_ids=old_class_ids, new_class_ids=new_class_ids
        )
        forgetting = float(old_before["balanced_accuracy"]) - float(
            groups["old"]["balanced_accuracy"]
        )
        phase_origin = phase_origin_confusion_diagnostics(
            cumulative,
            phase_to_classes={
                index: self.dataset.phase_to_classes[index]
                for index in range(phase_id + 1)
            },
        )

        final_record = history[-1]
        final_old_weight = classifier.class_weight(old_class_tensor).detach()
        final_new_weight = classifier.class_weight(new_class_tensor).detach()
        if not torch.equal(final_old_weight, old_weight_before):
            raise RuntimeError(
                "historical classifier rows changed across the incremental phase"
            )
        method_state = {
            "persistent_cds": "{I_Q,V_Q,I_P,V_P}",
            "decision_rule": "one shared bias-free all-seen cosine classifier over Q",
            "physical_evidence": "I_P/V_P is phase-stable spectral evidence used in intra/inter preservation, cross-phase separation and Whole-CDS pair risk",
            "training_chain": "single joint incremental optimization; native incoming rows; new-class CE + ordered feature KD + old-logit KD + immutable-reference/working-coordinate Whole-CDS preservation + risk-aware Whole-CDS contrastive separation + frozen-old Whole-CDS classifier protection",
            "gradient_ownership": "historical classifier rows are immutable; real-current CE updates incoming rows only; Whole-CDS classifier protection updates incoming rows against fixed historical anchors; future rows never update",
            "semantic_drift": "persistent historical CDS is immutable; a detached current-coordinate working CDS is refreshed from observable current-class Q drift using Whole-CDS relevance; I_P/V_P stay unchanged",
            "stores_old_images": False,
            "stores_old_feature_rows": False,
            "stores_synthetic_support": False,
            "uses_covariance_or_gaussian_memory": False,
            "uses_second_classifier": False,
            "uses_post_training_classifier_stage": False,
            "uses_phase_start_classifier_repair": False,
            "uses_cds_row_initialization": False,
            "joint_epochs": epoch_count,
            "joint_lr": learning_rate,
            "contrastive_weight": contrastive_weight,
            "pic_weight": contrastive_weight,
            "feature_kd_weight": feature_kd_weight,
            "kd_logit_weight": kd_logit_weight,
            "kd_temperature": kd_temperature,
            "cds_weight": cds_weight,
            "feature_stability": "ordered feature KD + old-logit KD + immutable-reference/working-coordinate Whole-CDS preservation",
            "cross_phase_separation": "risk-aware Whole-CDS contrastive separation in one current Q coordinate; I_Q/V_Q/I_P/V_P and inter-class pair risk all affect the optimized loss",
            "phase_mapping_error": float(final_alignment.mapping_error.item()),
            "representation_drift_rms": float(final_alignment.representation_drift_rms.item()),
            "phase_scale_deviation_rms": float(final_alignment.scale_deviation_rms.item()),
            "phase_translation_rms": float(final_alignment.translation_rms.item()),
            "drift_learned_relevance_mean": float(final_alignment.learned_relevance_mean.item()),
            "drift_physical_relevance_mean": float(final_alignment.physical_relevance_mean.item()),
            "drift_effective_relevance_mean": float(final_alignment.effective_relevance_mean.item()),
            "drift_support_mass_mean": float(final_alignment.class_specific_support_mass_mean.item()),
            "drift_support_mass_min": float(final_alignment.class_specific_support_mass_min.item()),
            "drift_support_mass_max": float(final_alignment.class_specific_support_mass_max.item()),
            "drift_identity_correction_relative_mean": float(final_alignment.identity_correction_relative_mean.item()),
            "drift_identity_correction_relative_max": float(final_alignment.identity_correction_relative_max.item()),
            "drift_variation_correction_relative_mean": float(final_alignment.variation_correction_relative_mean.item()),
            "drift_variation_correction_relative_max": float(final_alignment.variation_correction_relative_max.item()),
            "drift_variation_negative_fraction_before_sanitize": float(final_alignment.variation_negative_fraction_before_sanitize.item()),
            "final_joint_classification_loss": float(final_record["classification_loss"]),
            "final_feature_kd_loss": float(final_record["feature_kd_loss"]),
            "final_whole_cds_classifier_loss": float(final_record["whole_cds_classifier_loss"]),
            "final_old_cds_classifier_loss": float(final_record["old_cds_classifier_loss"]),
            "final_new_cds_classifier_loss": float(final_record["new_cds_classifier_loss"]),
            "final_cds_separation_loss": float(final_record["cds_separation_loss"]),
            "final_cds_separation_margin": float(final_record["cds_separation_margin"]),
            "final_cds_separation_violation_fraction": float(final_record["cds_separation_violation_fraction"]),
            "classifier_balance": "real current CE trains incoming rows only; finalized historical rows stay fixed; the existing Whole-CDS regulator positions incoming rows against fixed old/new structural boundaries using the same deployed cosine logits",
            "final_contrastive_loss": float(final_record["contrastive_loss"]),
            "final_pic_loss": float(final_record["pic_loss"]),
            "final_pic_positive_similarity": float(final_record["pic_positive_similarity"]),
            "final_pic_current_negative_similarity": float(final_record["pic_current_negative_similarity"]),
            "final_pic_historical_negative_similarity": float(final_record["pic_historical_negative_similarity"]),
            "final_pic_valid_anchor_fraction": float(final_record["pic_valid_anchor_fraction"]),
            "final_pic_physical_confusability_mean": float(final_record["pic_physical_confusability_mean"]),
            "final_pic_effective_pair_risk_mean": float(final_record["pic_effective_pair_risk_mean"]),
            "final_kd_logit_loss": float(final_record["kd_logit_loss"]),
            "final_cds_preservation_loss": float(final_record["cds_preservation_loss"]),
            "final_cds_intra_loss": float(final_record["cds_intra_loss"]),
            "final_cds_inter_loss": float(final_record["cds_inter_loss"]),
            "final_classifier_structural_loss": float(final_structural.loss.detach().item()),
            "final_old_old_retention_loss": float(final_record["old_old_retention_loss"]),
            "initial_old_new_violation_fraction": float(pre_structural.old_new_violation_fraction.detach().item()),
            "final_old_new_violation_fraction": float(final_structural.old_new_violation_fraction.detach().item()),
            "initial_old_old_violation_fraction": float(pre_structural.historical_internal_violation_fraction.detach().item()),
            "final_old_old_violation_fraction": float(final_structural.historical_internal_violation_fraction.detach().item()),
            "final_old_old_clearance_mean": float(final_structural.historical_internal_clearance_mean.detach().item()),
            "final_old_old_clearance_min": float(final_structural.historical_internal_clearance_min.detach().item()),
            "final_old_old_retention_mean": float(final_structural.historical_internal_retention_mean.detach().item()),
            "final_old_old_retention_min": float(final_structural.historical_internal_retention_min.detach().item()),
            "final_whole_cds_learned_risk_mean": float(final_structural.learned_pair_risk_mean.detach().item()),
            "final_whole_cds_physical_risk_mean": float(final_structural.physical_pair_risk_mean.detach().item()),
            "final_physical_absolute_support_mean": float(final_record["physical_absolute_support_mean"]),
            "final_physical_relation_support_mean": float(final_record["physical_relation_support_mean"]),
            "final_physical_relevance_mean": float(final_record["physical_relevance_mean"]),
            "historical_classifier_row_max_change": self._max_parameter_change(old_weight_before, final_old_weight),
            "incoming_classifier_initialization_max_change": 0.0,
            "incoming_classifier_training_max_change": self._max_parameter_change(new_weight_phase_start, final_new_weight),
            "phase_start_old_ba": float(old_before["balanced_accuracy"]),
            "phase_start_old_all_seen_ba": float(phase_start_old_all_seen["balanced_accuracy"]),
            "phase_start_new_only_ba": float(phase_start_new_only["balanced_accuracy"]),
            "phase_start_new_all_seen_ba": float(phase_start_new_all_seen["balanced_accuracy"]),
            "final_old_only_ba": float(final_record["old_only_ba"]),
            "final_old_all_seen_ba": float(final_record["old_all_seen_ba"]),
            "final_new_only_ba": float(final_record["new_only_ba"]),
            "final_new_all_seen_ba": float(final_record["new_all_seen_ba"]),
            "classifier_logit_scale_frozen_incremental": True,
            "second_classifier_stage": False,
        }


        return {
            "phase": phase_id,
            "old_class_ids": list(old_class_ids),
            "new_class_ids": list(new_class_ids),
            "seen_class_ids": list(seen_class_ids),
            "history": history,
            "final_epoch": epoch_count,
            "cumulative_test": cumulative,
            "old_new": groups,
            "forgetting": forgetting,
            "phase_origin_confusion": phase_origin,
            "method_state": method_state,
            "structure_committed": True,
            "phase_finalized": True,
        }


__all__ = ["IncrementalPhaseTrainer"]











# from __future__ import annotations
# import math
# import sys
# from numbers import Integral, Real
# from typing import Any, Dict

# import torch
# from tqdm import tqdm

# from losses.loss import (
#     classification_objective,
#     historical_cds_preservation_loss,
#     logit_distillation_loss,
#     reference_q_consistency_loss,
#     supervised_contrastive_q_loss,
# )
# from models.classifier import CosineClassificationHead
# from models.cds_decision_region import (
#     HSICDSDecisionRegionRegulator,
#     initialize_incoming_classifier_rows_from_cds_identity,
# )
# from models.geometry_bank import ClassStructureBatch, merge_class_structure_batches
# from models.phase_coordinate_alignment import correct_historical_structure
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


# def _clone_structure(structure: ClassStructureBatch) -> ClassStructureBatch:
#     if not isinstance(structure, ClassStructureBatch) or structure.class_count <= 0:
#         raise ValueError("cannot clone an empty class structure")
#     return ClassStructureBatch(
#         class_ids=structure.class_ids.detach().clone(),
#         sample_counts=structure.sample_counts.detach().clone(),
#         learned_identity=structure.learned_identity.detach().clone(),
#         learned_variation=structure.learned_variation.detach().clone(),
#         physical_identity=structure.physical_identity.detach().clone(),
#         physical_variation=structure.physical_variation.detach().clone(),
#     )


# def _targets_local(labels: Tensor, class_ids: Tensor) -> Tensor:
#     matches = labels.unsqueeze(1).eq(class_ids.unsqueeze(0))
#     if not bool(matches.sum(dim=1).eq(1).all()):
#         raise ValueError("a current label is absent from current_class_ids")
#     return matches.to(torch.long).argmax(dim=1)


# class IncrementalPhaseTrainer:
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
#             "structure_committed",
#             "phase_finalized",
#         }
#     )

#     def _require_classifier(self) -> CosineClassificationHead:
#         classifier = getattr(self.model, "classifier", None)
#         if not isinstance(classifier, CosineClassificationHead):
#             raise RuntimeError("model.classifier must be CosineClassificationHead")
#         return classifier

#     def _phase_classes(
#         self,
#         phase: int,
#     ) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
#         old_class_ids = tuple(
#             self._validated_class_ids(
#                 self.dataset.get_old_classes(phase), name=f"phase_{phase}_old_classes"
#             )
#         )
#         new_class_ids = tuple(
#             self._validated_class_ids(
#                 self.dataset.get_new_classes(phase), name=f"phase_{phase}_new_classes"
#             )
#         )
#         seen_class_ids = tuple(
#             self._validated_class_ids(
#                 self.dataset.get_seen_classes(phase), name=f"phase_{phase}_seen_classes"
#             )
#         )
#         if not old_class_ids or not new_class_ids:
#             raise RuntimeError("incremental phase requires old and new classes")
#         if set(old_class_ids).intersection(new_class_ids):
#             raise RuntimeError("old and new classes overlap")
#         if seen_class_ids != old_class_ids + new_class_ids:
#             raise RuntimeError("seen classes must equal old classes followed by new classes")
#         if tuple(self.model.committed_class_ids) != old_class_ids:
#             raise RuntimeError("committed historical CDS does not match old classes")
#         return old_class_ids, new_class_ids, seen_class_ids

#     @staticmethod
#     def _assert_frozen_reference(reference_backbone: torch.nn.Module) -> None:
#         reference_backbone.eval()
#         if any(parameter.requires_grad for parameter in reference_backbone.parameters()):
#             raise RuntimeError("previous backbone reference must remain frozen")
#         if any(parameter.grad is not None for parameter in reference_backbone.parameters()):
#             raise RuntimeError("previous backbone reference must not carry gradients")

#     @staticmethod
#     def _assert_frozen_reference_classifier(
#         reference_classifier: CosineClassificationHead,
#     ) -> None:
#         reference_classifier.eval()
#         if any(parameter.requires_grad for parameter in reference_classifier.parameters()):
#             raise RuntimeError("previous classifier reference must remain frozen")
#         if any(parameter.grad is not None for parameter in reference_classifier.parameters()):
#             raise RuntimeError("previous classifier reference must not carry gradients")

#     @staticmethod
#     @torch.no_grad()
#     def _assert_reference_matches_phase_start(
#         reference: torch.nn.Module,
#         current: torch.nn.Module,
#         *,
#         name: str,
#     ) -> None:
#         ref_state = reference.state_dict()
#         cur_state = current.state_dict()
#         if tuple(ref_state.keys()) != tuple(cur_state.keys()):
#             raise RuntimeError(f"previous/current {name} schemas differ at phase start")
#         for key in ref_state:
#             if not torch.equal(ref_state[key].detach().cpu(), cur_state[key].detach().cpu()):
#                 raise RuntimeError(
#                     f"previous {name} does not match finalized phase-start state at {key}"
#                 )

#     @torch.no_grad()
#     def _current_class_weights(
#         self,
#         loader: Any,
#         *,
#         new_class_ids: tuple[int, ...],
#     ) -> tuple[Tensor, int]:
#         counts = {class_id: 0 for class_id in new_class_ids}
#         total = 0
#         valid = set(new_class_ids)
#         for batch in loader:
#             _, _, labels = self.unpack_batch(batch)
#             observed = set(int(value) for value in labels.unique().detach().cpu().tolist())
#             outside = observed - valid
#             if outside:
#                 raise RuntimeError(
#                     f"incremental TRAIN loader contains non-current classes {sorted(outside)}"
#                 )
#             for class_id in new_class_ids:
#                 count = int(labels.eq(class_id).sum().item())
#                 counts[class_id] += count
#                 total += count
#         if total <= 0 or any(counts[class_id] <= 0 for class_id in new_class_ids):
#             raise RuntimeError("incremental TRAIN split is empty or misses a current class")
#         class_count = len(new_class_ids)
#         weights = torch.tensor(
#             [total / float(class_count * counts[class_id]) for class_id in new_class_ids],
#             device=self.device,
#             dtype=self.model.dtype,
#         )
#         return weights, total

#     @staticmethod
#     def _max_parameter_change(before: Tensor, after: Tensor) -> float:
#         if before.shape != after.shape:
#             raise ValueError("parameter snapshots have different shapes")
#         if before.numel() == 0:
#             return 0.0
#         return float((after - before).abs().max().item())

#     @staticmethod
#     def _assert_future_rows_unchanged(
#         classifier: CosineClassificationHead,
#         future_class_tensor: Tensor,
#         future_weight_before: Tensor | None,
#     ) -> None:
#         if future_weight_before is None or future_class_tensor.numel() == 0:
#             return
#         current = classifier.class_weight(future_class_tensor).detach()
#         if not torch.equal(current, future_weight_before):
#             raise RuntimeError("future classifier rows changed before their phase")

#     @staticmethod
#     def _row_grad_max(gradient: Tensor | None, class_ids: Tensor) -> float:
#         if gradient is None or class_ids.numel() == 0:
#             return 0.0
#         rows = gradient.index_select(0, class_ids)
#         return float(rows.abs().max().detach().item()) if rows.numel() else 0.0

#     def train_incremental_phase(
#         self,
#         *,
#         phase: int,
#         epochs: int,
#         batch_size: int,
#         lr: float,
#         reference_backbone: torch.nn.Module,
#         reference_classifier: CosineClassificationHead,
#     ) -> Dict[str, Any]:
#         phase_id = _positive_int(phase, name="phase")
#         epoch_count = _positive_int(epochs, name="epochs")
#         train_batch_size = _positive_int(batch_size, name="batch_size")
#         learning_rate = _finite_positive("lr_inc", lr)
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
#         q_consistency_weight = _finite_nonnegative(
#             "q_consistency_weight", self.cfg("q_consistency_weight", 1.0)
#         )
#         kd_logit_weight = _finite_nonnegative(
#             "kd_logit_weight", self.cfg("kd_logit_weight", 1.0)
#         )
#         kd_temperature = _finite_positive(
#             "kd_temperature", self.cfg("kd_temperature", 2.0)
#         )
#         cds_weight = _finite_nonnegative("cds_weight", self.cfg("cds_weight", 1.0))
        
#         self.model.validate_finalized_state()
#         classifier = self._require_classifier()
#         self._assert_frozen_reference(reference_backbone)
#         self._assert_frozen_reference_classifier(reference_classifier)
#         self._assert_reference_matches_phase_start(
#             reference_backbone, self.model.backbone, name="backbone"
#         )
#         self._assert_reference_matches_phase_start(
#             reference_classifier, classifier, name="classifier"
#         )

#         previous_phase = phase_id - 1
#         previous_seen_class_ids = tuple(
#             self._validated_class_ids(
#                 self.dataset.get_seen_classes(previous_phase),
#                 name="previous_seen_classes",
#             )
#         )
#         old_reporting_loader = self.dataset.get_reporting_dataloader(
#             previous_phase, batch_size=eval_batch_size
#         )
#         old_before = self.evaluate_loader(
#             old_reporting_loader,
#             class_ids=previous_seen_class_ids,
#             target_class_ids=previous_seen_class_ids,
#         )

#         self.dataset.start_phase(phase_id)
#         old_class_ids, new_class_ids, seen_class_ids = self._phase_classes(phase_id)
#         old_class_tensor = torch.tensor(old_class_ids, device=self.device, dtype=torch.long)
#         new_class_tensor = torch.tensor(new_class_ids, device=self.device, dtype=torch.long)
#         seen_class_tensor = torch.tensor(seen_class_ids, device=self.device, dtype=torch.long)

#         historical_structure = _clone_structure(
#             self.model.structure_bank.class_structures(old_class_ids)
#         )
#         reference_historical_cds = self.model.structure_bank.build_cds(historical_structure)

#         self.model.set_incremental_trainability()
#         backbone_parameters = [
#             parameter
#             for parameter in self.model.backbone.parameters()
#             if parameter.requires_grad
#         ]
#         if not backbone_parameters:
#             raise RuntimeError("incremental backbone has no trainable parameters")

#         train_eval_loader = self.dataset.get_phase_dataloader(
#             phase_id, split="train", batch_size=eval_batch_size, shuffle=False
#         )
#         current_class_weights, expected_samples = self._current_class_weights(
#             train_eval_loader, new_class_ids=new_class_ids
#         )
#         reference_train = self.collect_ordered_features(
#             train_eval_loader, backbone=reference_backbone
#         )
#         reference_features_full = reference_train["ordered_features"].detach()
#         reference_spectra = reference_train["raw_center_spectra"].detach()
#         reference_labels = reference_train["labels"].detach()
#         observed_new = tuple(
#             sorted(int(value) for value in reference_labels.unique().detach().cpu().tolist())
#         )
#         if observed_new != tuple(sorted(new_class_ids)):
#             raise RuntimeError("incremental TRAIN rows do not match current phase classes")

#         current_physical_structure = self.model.estimate_physical_class_structure(
#             reference_spectra, reference_labels, class_ids=new_class_ids
#         )
#         current_reference_structure = self.model.estimate_class_structure(
#             reference_features_full,
#             None,
#             reference_labels,
#             class_ids=new_class_ids,
#             physical_structure=current_physical_structure,
#         )
#         phase_start_seen_structure = merge_class_structure_batches(
#             (historical_structure, current_reference_structure)
#         )
#         phase_start_seen_cds = self.model.structure_bank.build_cds(
#             phase_start_seen_structure
#         )

#         # Snapshot ownership before incoming-row admission.
#         old_weight_before = classifier.class_weight(old_class_tensor).detach().clone()
#         new_weight_random_before = classifier.class_weight(new_class_tensor).detach().clone()
#         seen_set = set(seen_class_ids)
#         future_class_ids = tuple(
#             class_id
#             for class_id in range(classifier.total_class_count)
#             if class_id not in seen_set
#         )
#         future_class_tensor = torch.tensor(
#             future_class_ids, device=self.device, dtype=torch.long
#         )
#         future_weight_before = (
#             classifier.class_weight(future_class_tensor).detach().clone()
#             if future_class_ids
#             else None
#         )

#         # Deterministic, one-time incoming-row admission from the complete CDS.
#         # Historical rows are never changed by this operation.
#         initialization = initialize_incoming_classifier_rows_from_cds_identity(
#             classifier=classifier,
#             seen_cds=phase_start_seen_cds,
#             historical_class_count=len(old_class_ids),
#         )
#         phase_start_new_weight = classifier.class_weight(new_class_tensor).detach().clone()
#         if not torch.equal(classifier.class_weight(old_class_tensor).detach(), old_weight_before):
#             raise RuntimeError("incoming-row admission changed a historical classifier row")
#         self._assert_future_rows_unchanged(
#             classifier, future_class_tensor, future_weight_before
#         )

#         scale_requires_grad = bool(classifier.logit_scale_parameter.requires_grad)
#         classifier.logit_scale_parameter.requires_grad_(False)
#         classifier.logit_scale_parameter.grad = None
#         classifier.weight.requires_grad_(True)
#         phase_start_scale = float(classifier.logit_scale.detach().item())

#         train_loader = self.dataset.get_phase_dataloader(
#             phase_id, split="train", batch_size=train_batch_size, shuffle=True
#         )
#         val_loader = (
#             self.dataset.get_phase_dataloader(
#                 phase_id, split="val", batch_size=eval_batch_size, shuffle=False
#             )
#             if self.dataset.validation_enabled
#             else None
#         )

#         phase_start_new_only = self.evaluate_loader(
#             train_eval_loader, class_ids=new_class_ids, target_class_ids=new_class_ids
#         )
#         phase_start_new_all_seen = self.evaluate_loader(
#             train_eval_loader, class_ids=seen_class_ids, target_class_ids=new_class_ids
#         )
#         phase_start_old_all_seen = self.evaluate_loader(
#             old_reporting_loader, class_ids=seen_class_ids, target_class_ids=old_class_ids
#         )
#         print(
#             f"[Phase {phase_id} start] "
#             f"OldBA(previous-only)={100*float(old_before['balanced_accuracy']):.2f}% "
#             f"OldBA(all-seen)={100*float(phase_start_old_all_seen['balanced_accuracy']):.2f}% "
#             f"NewBA(new-only)={100*float(phase_start_new_only['balanced_accuracy']):.2f}% "
#             f"NewBA(all-seen)={100*float(phase_start_new_all_seen['balanced_accuracy']):.2f}% | "
#             f"InitClear={float(initialization.joint_clearance_min.item()):.4f} "
#             f"InitClearMean={float(initialization.joint_clearance_mean.item()):.4f}",
#             file=sys.stdout,
#         )

#         @torch.no_grad()
#         def refresh_joint_structure():
#             self.model.eval()
#             rows = self.collect_ordered_features(train_eval_loader)
#             if not torch.equal(rows["labels"], reference_labels):
#                 raise RuntimeError("current TRAIN label order changed during CDS refresh")
#             if not torch.equal(rows["raw_center_spectra"], reference_spectra):
#                 raise RuntimeError("current TRAIN spectra changed during CDS refresh")
#             current_features_full = rows["ordered_features"].detach()
#             current_structure = self.model.estimate_class_structure(
#                 current_features_full,
#                 None,
#                 reference_labels,
#                 class_ids=new_class_ids,
#                 physical_structure=current_physical_structure,
#             )
#             corrected_historical, transform, alignment = correct_historical_structure(
#                 historical_structure=historical_structure,
#                 current_reference_structure=current_reference_structure,
#                 current_structure=current_structure,
#                 reference_features=reference_features_full,
#                 current_features=current_features_full,
#             )
#             if not torch.equal(
#                 corrected_historical.physical_identity,
#                 historical_structure.physical_identity,
#             ):
#                 raise RuntimeError("historical I_P changed during semantic correction")
#             if not torch.equal(
#                 corrected_historical.physical_variation,
#                 historical_structure.physical_variation,
#             ):
#                 raise RuntimeError("historical V_P changed during semantic correction")
#             merged = merge_class_structure_batches(
#                 (corrected_historical, current_structure)
#             )
#             merged_ids = tuple(
#                 int(value) for value in merged.class_ids.detach().cpu().tolist()
#             )
#             if merged_ids != seen_class_ids:
#                 raise RuntimeError("CDS refresh produced incorrect seen-class order")
#             seen_cds = self.model.structure_bank.build_cds(merged)
#             return (
#                 current_features_full,
#                 corrected_historical,
#                 current_structure,
#                 merged,
#                 seen_cds,
#                 transform,
#                 alignment,
#             )

#         # One optimizer, one scheduler, one phase loop.  Classifier weight decay is
#         # zero so masked old/future rows are not silently changed by AdamW decay.
#         optimizer = torch.optim.AdamW(
#             [
#                 {"params": backbone_parameters, "weight_decay": weight_decay},
#                 {"params": [classifier.weight], "weight_decay": 0.0},
#             ],
#             lr=learning_rate,
#         )
#         scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
#             optimizer, T_max=epoch_count, eta_min=0.0
#         )
#         regulator = HSICDSDecisionRegionRegulator().to(self.device)

#         joint_state = refresh_joint_structure()
#         pre_structural = regulator(
#             classifier=classifier,
#             seen_cds=joint_state[4],
#             historical_class_count=len(old_class_ids),
#             reference_classifier=reference_classifier,
#             reference_historical_cds=reference_historical_cds,
#         )

#         history: list[dict[str, Any]] = []
#         gradient_contract_checked = False
#         progress = tqdm(
#             range(epoch_count),
#             desc=f"Phase {phase_id} joint CDS-cosine training",
#             unit="epoch",
#             dynamic_ncols=True,
#             file=sys.stdout,
#         )

#         for epoch in progress:
#             (
#                 _epoch_features,
#                 epoch_historical_structure,
#                 _epoch_current_structure,
#                 _epoch_merged,
#                 epoch_seen_cds,
#                 _epoch_transform,
#                 _epoch_alignment,
#             ) = joint_state

#             self.model.train()
#             sums = {name: 0.0 for name in (
#                 "classification", "cds_classifier", "cds_classifier_old",
#                 "cds_classifier_new", "contrastive", "q_consistency", "kd",
#                 "cds", "intra", "inter", "physical_abs", "physical_rel",
#                 "physical_joint", "old_old", "structural", "total"
#             )}
#             sample_count = 0
#             correct_all_seen = 0

#             for batch in train_loader:
#                 patch, spectrum, labels = self.unpack_batch(batch)
#                 if not bool(torch.isin(labels, new_class_tensor).all()):
#                     raise RuntimeError("incremental TRAIN batch contains an old or future class")

#                 optimizer.zero_grad(set_to_none=True)
#                 classifier.logit_scale_parameter.grad = None

#                 with torch.no_grad():
#                     reference_features = self.model.extract_features_with_backbone(
#                         reference_backbone,
#                         patch,
#                         center_spectrum=spectrum,
#                         return_aux=False,
#                     ).ordered_features.detach()
#                     physical_evidence = self.model.physical_spectral_evidence(spectrum).detach()
#                     teacher_old_logits = reference_classifier(
#                         reference_features,
#                         class_ids=old_class_tensor,
#                         trainable_class_ids=(),
#                     ).logits.detach()

#                 current_features = self.model.extract_features(
#                     patch, center_spectrum=spectrum, return_aux=False
#                 ).ordered_features
#                 current_local = _targets_local(labels, new_class_tensor)
#                 sample_weights = current_class_weights.index_select(0, current_local)

#                 # CRITICAL ownership rule: all seen classes compete, only incoming
#                 # rows receive gradient from labels that contain current classes only.
#                 all_seen_output = classifier(
#                     current_features,
#                     class_ids=seen_class_tensor,
#                     trainable_class_ids=new_class_tensor,
#                 )
#                 classification = classification_objective(
#                     logits=all_seen_output.logits,
#                     class_ids=all_seen_output.class_ids,
#                     labels_global=labels,
#                     sample_weights=sample_weights,
#                 )
#                 contrastive = supervised_contrastive_q_loss(
#                     ordered_features=current_features,
#                     labels_global=labels,
#                     logit_scale=classifier.logit_scale,
#                     sample_weights=sample_weights,
#                 )
#                 q_consistency = reference_q_consistency_loss(
#                     reference_features=reference_features,
#                     current_features=current_features,
#                 )
#                 student_old_logits = classifier(
#                     current_features,
#                     class_ids=old_class_tensor,
#                     trainable_class_ids=(),
#                 ).logits
#                 kd = logit_distillation_loss(
#                     student_old_logits=student_old_logits,
#                     teacher_old_logits=teacher_old_logits,
#                     temperature=kd_temperature,
#                     sample_weights=sample_weights,
#                 )
#                 cds = historical_cds_preservation_loss(
#                     reference_features=reference_features,
#                     current_features=current_features,
#                     physical_spectral_evidence=physical_evidence,
#                     reference_historical_structure=historical_structure,
#                     current_historical_structure=epoch_historical_structure,
#                 )
#                 structural = regulator(
#                     classifier=classifier,
#                     seen_cds=epoch_seen_cds,
#                     historical_class_count=len(old_class_ids),
#                     reference_classifier=reference_classifier,
#                     reference_historical_cds=reference_historical_cds,
#                 )

#                 # One coherent classifier-reconstruction rule is used for every
#                 # incremental phase. Real current labels use the deployed all-seen
#                 # cosine head but can update incoming rows only. In parallel, the
#                 # whole seen-class CDS reconstructs the same classifier with equal
#                 # per-class mass: I_Q/V_Q define directed robust class support and
#                 # I_P/V_P define pair relevance. Old-source CDS may update old
#                 # positives and seen rivals; new-source CDS keeps old rival rows
#                 # detached. No prototype-only anchor or second classifier exists.
#                 if not bool(torch.isfinite(structural.loss)):
#                     raise RuntimeError("whole-CDS classifier reconstruction became NaN or Inf")

#                 if not gradient_contract_checked:
#                     tolerance = 64.0 * torch.finfo(classifier.weight.dtype).eps

#                     ce_grad = torch.autograd.grad(
#                         classification.loss,
#                         classifier.weight,
#                         retain_graph=True,
#                         allow_unused=True,
#                     )[0]
#                     ce_old = self._row_grad_max(ce_grad, old_class_tensor)
#                     ce_new = self._row_grad_max(ce_grad, new_class_tensor)
#                     ce_future = self._row_grad_max(ce_grad, future_class_tensor)
#                     if ce_old > tolerance:
#                         raise RuntimeError("new-only CE leaked gradient into historical rows")
#                     if ce_future > tolerance:
#                         raise RuntimeError("new-only CE leaked gradient into future rows")
#                     if ce_new <= tolerance:
#                         raise RuntimeError("incoming rows received no real-current CE gradient")

#                     old_cds_grad = torch.autograd.grad(
#                         structural.old_class_reconstruction_loss,
#                         classifier.weight,
#                         retain_graph=True,
#                         allow_unused=True,
#                     )[0]
#                     old_cds_old = self._row_grad_max(old_cds_grad, old_class_tensor)
#                     old_cds_new = self._row_grad_max(old_cds_grad, new_class_tensor)
#                     old_cds_future = self._row_grad_max(old_cds_grad, future_class_tensor)
#                     if old_cds_old <= tolerance or old_cds_new <= tolerance:
#                         raise RuntimeError(
#                             "historical whole-CDS reconstruction must update old positives and seen rivals"
#                         )
#                     if old_cds_future > tolerance:
#                         raise RuntimeError("historical whole-CDS reconstruction leaked into future rows")

#                     new_cds_grad = torch.autograd.grad(
#                         structural.new_class_reconstruction_loss,
#                         classifier.weight,
#                         retain_graph=True,
#                         allow_unused=True,
#                     )[0]
#                     new_cds_old = self._row_grad_max(new_cds_grad, old_class_tensor)
#                     new_cds_new = self._row_grad_max(new_cds_grad, new_class_tensor)
#                     new_cds_future = self._row_grad_max(new_cds_grad, future_class_tensor)
#                     if new_cds_old > tolerance:
#                         raise RuntimeError("current-class CDS leaked gradient into historical rows")
#                     if new_cds_new <= tolerance:
#                         raise RuntimeError("current-class CDS produced no incoming-row gradient")
#                     if new_cds_future > tolerance:
#                         raise RuntimeError("current-class CDS leaked into future rows")

#                     global_cds_grad = torch.autograd.grad(
#                         structural.global_reconstruction_loss,
#                         classifier.weight,
#                         retain_graph=True,
#                         allow_unused=True,
#                     )[0]
#                     global_old = self._row_grad_max(global_cds_grad, old_class_tensor)
#                     global_new = self._row_grad_max(global_cds_grad, new_class_tensor)
#                     global_future = self._row_grad_max(global_cds_grad, future_class_tensor)
#                     if global_old <= tolerance or global_new <= tolerance:
#                         raise RuntimeError("global whole-CDS reconstruction must update all seen groups")
#                     if global_future > tolerance:
#                         raise RuntimeError("global whole-CDS reconstruction leaked into future rows")

#                     retention_grad = torch.autograd.grad(
#                         structural.old_old_retention_loss,
#                         classifier.weight,
#                         retain_graph=True,
#                         allow_unused=True,
#                     )[0]
#                     ret_old = self._row_grad_max(retention_grad, old_class_tensor)
#                     ret_new = self._row_grad_max(retention_grad, new_class_tensor)
#                     ret_future = self._row_grad_max(retention_grad, future_class_tensor)
#                     if ret_new > tolerance or ret_future > tolerance:
#                         raise RuntimeError("old-old retention changed a non-historical row")

#                     print(
#                         "[Gradient ownership] "
#                         f"NewCE old={ce_old:.3e} new={ce_new:.3e} future={ce_future:.3e} | "
#                         f"OldCDS old={old_cds_old:.3e} new={old_cds_new:.3e} | "
#                         f"NewCDS old={new_cds_old:.3e} new={new_cds_new:.3e} | "
#                         f"GlobalCDS old={global_old:.3e} new={global_new:.3e} future={global_future:.3e} | "
#                         f"Retention old={ret_old:.3e} new={ret_new:.3e} future={ret_future:.3e}",
#                         file=sys.stdout,
#                     )
#                     gradient_contract_checked = True

#                 loss = (
#                     classification.loss
#                     + contrastive_weight * contrastive.loss
#                     + q_consistency_weight * q_consistency.loss
#                     + kd_logit_weight * kd.loss
#                     + cds_weight * cds.loss
#                     + structural.loss
#                 )
#                 if not bool(torch.isfinite(loss)):
#                     raise RuntimeError("joint incremental objective became NaN or Inf")
#                 loss.backward()

#                 if classifier.logit_scale_parameter.grad is not None:
#                     raise RuntimeError("incremental training must not update cosine logit scale")
#                 if classifier.weight.grad is None or not bool(
#                     torch.isfinite(classifier.weight.grad).all()
#                 ):
#                     raise RuntimeError("joint classifier gradient is missing/non-finite")
#                 if future_class_tensor.numel() and bool(
#                     torch.count_nonzero(
#                         classifier.weight.grad.index_select(0, future_class_tensor)
#                     )
#                 ):
#                     raise RuntimeError("future classifier rows received gradients")

#                 backbone_grad_seen = False
#                 for parameter in backbone_parameters:
#                     if parameter.grad is None:
#                         continue
#                     if not bool(torch.isfinite(parameter.grad).all()):
#                         raise RuntimeError("incremental backbone gradient contains NaN or Inf")
#                     if bool(torch.count_nonzero(parameter.grad)):
#                         backbone_grad_seen = True
#                 if not backbone_grad_seen:
#                     raise RuntimeError("joint objective produced zero backbone gradient")

#                 if gradient_clip > 0:
#                     torch.nn.utils.clip_grad_norm_(
#                         backbone_parameters + [classifier.weight],
#                         gradient_clip,
#                         error_if_nonfinite=True,
#                     )
#                 optimizer.step()
#                 self._assert_future_rows_unchanged(
#                     classifier, future_class_tensor, future_weight_before
#                 )

#                 n = int(labels.numel())
#                 sums["classification"] += float(classification.loss.detach().item()) * n
#                 sums["cds_classifier"] += float(
#                     structural.global_reconstruction_loss.detach().item()
#                 ) * n
#                 sums["cds_classifier_old"] += float(
#                     structural.old_class_reconstruction_loss.detach().item()
#                 ) * n
#                 sums["cds_classifier_new"] += float(
#                     structural.new_class_reconstruction_loss.detach().item()
#                 ) * n
#                 sums["contrastive"] += float(contrastive.loss.detach().item()) * n
#                 sums["q_consistency"] += float(q_consistency.loss.detach().item()) * n
#                 sums["kd"] += float(kd.loss.detach().item()) * n
#                 sums["cds"] += float(cds.loss.detach().item()) * n
#                 sums["intra"] += float(cds.intra_loss.detach().item()) * n
#                 sums["inter"] += float(cds.inter_loss.detach().item()) * n
#                 sums["physical_abs"] += float(
#                     cds.physical_absolute_support_mean.detach().item()
#                 ) * n
#                 sums["physical_rel"] += float(
#                     cds.physical_relation_support_mean.detach().item()
#                 ) * n
#                 sums["physical_joint"] += float(
#                     cds.physical_relevance_mean.detach().item()
#                 ) * n
#                 sums["structural"] += float(structural.loss.detach().item()) * n
#                 sums["old_old"] += float(
#                     structural.old_old_retention_loss.detach().item()
#                 ) * n
#                 sums["total"] += float(loss.detach().item()) * n
#                 correct_all_seen += int(all_seen_output.prediction.detach().eq(labels).sum().item())
#                 sample_count += n

#             if sample_count != expected_samples:
#                 raise RuntimeError("incremental TRAIN loader did not traverse the complete split")
#             scheduler.step()
#             if float(classifier.logit_scale.detach().item()) != phase_start_scale:
#                 raise RuntimeError("cosine logit scale changed during incremental training")
#             self._assert_future_rows_unchanged(
#                 classifier, future_class_tensor, future_weight_before
#             )

#             # Refresh CDS coordinates for the NEXT epoch.  This performs no
#             # optimizer step and is therefore not a second training stage.
#             joint_state = refresh_joint_structure()
#             epoch_structural = regulator(
#                 classifier=classifier,
#                 seen_cds=joint_state[4],
#                 historical_class_count=len(old_class_ids),
#                 reference_classifier=reference_classifier,
#                 reference_historical_cds=reference_historical_cds,
#             )
#             alignment = joint_state[6]

#             self.model.eval()
#             old_only = self.evaluate_loader(
#                 old_reporting_loader,
#                 class_ids=old_class_ids,
#                 target_class_ids=old_class_ids,
#             )
#             old_all = self.evaluate_loader(
#                 old_reporting_loader,
#                 class_ids=seen_class_ids,
#                 target_class_ids=old_class_ids,
#             )
#             new_only = self.evaluate_loader(
#                 train_eval_loader,
#                 class_ids=new_class_ids,
#                 target_class_ids=new_class_ids,
#             )
#             new_all = self.evaluate_loader(
#                 train_eval_loader,
#                 class_ids=seen_class_ids,
#                 target_class_ids=new_class_ids,
#             )
#             validation_all = (
#                 None
#                 if val_loader is None
#                 else self.evaluate_loader(
#                     val_loader,
#                     class_ids=seen_class_ids,
#                     target_class_ids=new_class_ids,
#                 )
#             )

#             record = {
#                 "stage": "joint",
#                 "epoch": epoch + 1,
#                 "learning_rate": float(optimizer.param_groups[0]["lr"]),
#                 "classification_loss": sums["classification"] / sample_count,
#                 "whole_cds_classifier_loss": sums["cds_classifier"] / sample_count,
#                 "old_cds_classifier_loss": sums["cds_classifier_old"] / sample_count,
#                 "new_cds_classifier_loss": sums["cds_classifier_new"] / sample_count,
#                 "current_all_seen_accuracy": correct_all_seen / float(sample_count),
#                 "contrastive_loss": sums["contrastive"] / sample_count,
#                 "q_consistency_loss": sums["q_consistency"] / sample_count,
#                 "kd_logit_loss": sums["kd"] / sample_count,
#                 "cds_preservation_loss": sums["cds"] / sample_count,
#                 "cds_intra_loss": sums["intra"] / sample_count,
#                 "cds_inter_loss": sums["inter"] / sample_count,
#                 "physical_absolute_support_mean": sums["physical_abs"] / sample_count,
#                 "physical_relation_support_mean": sums["physical_rel"] / sample_count,
#                 "physical_relevance_mean": sums["physical_joint"] / sample_count,
#                 "structural_loss": sums["structural"] / sample_count,
#                 "old_old_retention_loss": sums["old_old"] / sample_count,
#                 "total_loss": sums["total"] / sample_count,
#                 "old_only_ba": float(old_only["balanced_accuracy"]),
#                 "old_all_seen_ba": float(old_all["balanced_accuracy"]),
#                 "new_only_ba": float(new_only["balanced_accuracy"]),
#                 "new_all_seen_ba": float(new_all["balanced_accuracy"]),
#                 "validation_new_all_seen_ba": None
#                 if validation_all is None
#                 else float(validation_all["balanced_accuracy"]),
#                 "old_new_violation_fraction": float(
#                     epoch_structural.old_new_violation_fraction.detach().item()
#                 ),
#                 "old_old_violation_fraction": float(
#                     epoch_structural.historical_internal_violation_fraction.detach().item()
#                 ),
#                 "joint_clearance_mean": float(
#                     epoch_structural.joint_structural_clearance_mean.detach().item()
#                 ),
#                 "joint_clearance_min": float(
#                     epoch_structural.joint_structural_clearance_min.detach().item()
#                 ),
#                 "epoch_mapping_error": float(alignment.mapping_error.item()),
#                 "epoch_representation_drift_rms": float(
#                     alignment.representation_drift_rms.item()
#                 ),
#             }
#             history.append(record)
#             tqdm.write(
#                 f"[Joint {epoch + 1:03d}/{epoch_count:03d}] "
#                 f"CEnew-all={record['classification_loss']:.4f} "
#                 f"CDSCls={record['whole_cds_classifier_loss']:.4f}"
#                 f"[Old={record['old_cds_classifier_loss']:.4f},"
#                 f"New={record['new_cds_classifier_loss']:.4f}] "
#                 f"Con={record['contrastive_loss']:.4f} "
#                 f"Q={record['q_consistency_loss']:.4f} "
#                 f"KD={record['kd_logit_loss']:.4f} "
#                 f"CDS={record['cds_preservation_loss']:.4f}"
#                 f"[I={record['cds_intra_loss']:.4f},R={record['cds_inter_loss']:.4f}] "
#                 f"Struct={record['structural_loss']:.4f}"
#                 f"[Ret={record['old_old_retention_loss']:.4f}] | "
#                 f"OldOnly/All={100*record['old_only_ba']:.2f}%/"
#                 f"{100*record['old_all_seen_ba']:.2f}% "
#                 f"NewOnly/All={100*record['new_only_ba']:.2f}%/"
#                 f"{100*record['new_all_seen_ba']:.2f}% "
#                 f"OldNewViol={100*record['old_new_violation_fraction']:.1f}% "
#                 f"OldOldViol={100*record['old_old_violation_fraction']:.1f}%",
#                 file=sys.stdout,
#             )

#         progress.close()
#         if not history:
#             raise RuntimeError("joint incremental training produced no epochs")

#         self.model.eval()
#         (
#             _final_features,
#             final_historical_structure,
#             final_current_structure,
#             committed_structure,
#             final_seen_cds,
#             _final_transform,
#             final_alignment,
#         ) = joint_state
#         final_structural = regulator(
#             classifier=classifier,
#             seen_cds=final_seen_cds,
#             historical_class_count=len(old_class_ids),
#             reference_classifier=reference_classifier,
#             reference_historical_cds=reference_historical_cds,
#         )

#         # Commit only corrected/real CDS.  The cosine classifier is already the
#         # same classifier optimized during the one joint stage; no reset follows.
#         self.model.replace_class_structure(committed_structure)
#         self.model.validate_finalized_state()
#         self.dataset.finalize_phase(phase_id)
#         classifier.logit_scale_parameter.requires_grad_(scale_requires_grad)
#         classifier.logit_scale_parameter.grad = None

#         cumulative_loader = self.dataset.get_reporting_dataloader(
#             phase_id, batch_size=eval_batch_size
#         )
#         cumulative = self.evaluate_loader(
#             cumulative_loader,
#             class_ids=seen_class_ids,
#             target_class_ids=seen_class_ids,
#         )
#         groups = old_new_summary(
#             cumulative, old_class_ids=old_class_ids, new_class_ids=new_class_ids
#         )
#         forgetting = float(old_before["balanced_accuracy"]) - float(
#             groups["old"]["balanced_accuracy"]
#         )
#         phase_origin = phase_origin_confusion_diagnostics(
#             cumulative,
#             phase_to_classes={
#                 index: self.dataset.phase_to_classes[index]
#                 for index in range(phase_id + 1)
#             },
#         )

#         final_record = history[-1]
#         final_old_weight = classifier.class_weight(old_class_tensor).detach()
#         final_new_weight = classifier.class_weight(new_class_tensor).detach()
#         method_state = {
#             "persistent_cds": "{I_Q,V_Q,I_P,V_P,N}",
#             "stores_old_images": False,
#             "stores_old_feature_rows": False,
#             "stores_synthetic_support": False,
#             "uses_covariance_or_gaussian_memory": False,
#             "uses_second_classifier": False,
#             "uses_post_training_classifier_stage": False,
#             "joint_epochs": epoch_count,
#             "joint_lr": learning_rate,
#             "contrastive_weight": contrastive_weight,
#             "q_consistency_weight": q_consistency_weight,
#             "kd_logit_weight": kd_logit_weight,
#             "kd_temperature": kd_temperature,
#             "cds_weight": cds_weight,
#                         "phase_mapping_error": float(final_alignment.mapping_error.item()),
#             "representation_drift_rms": float(
#                 final_alignment.representation_drift_rms.item()
#             ),
#             "phase_scale_deviation_rms": float(
#                 final_alignment.scale_deviation_rms.item()
#             ),
#             "phase_translation_rms": float(final_alignment.translation_rms.item()),
#             "drift_learned_relevance_mean": float(
#                 final_alignment.learned_relevance_mean.item()
#             ),
#             "drift_physical_relevance_mean": float(
#                 final_alignment.physical_relevance_mean.item()
#             ),
#             "drift_effective_relevance_mean": float(
#                 final_alignment.effective_relevance_mean.item()
#             ),
#             "drift_support_mass_mean": float(
#                 final_alignment.class_specific_support_mass_mean.item()
#             ),
#             "drift_support_mass_min": float(
#                 final_alignment.class_specific_support_mass_min.item()
#             ),
#             "drift_support_mass_max": float(
#                 final_alignment.class_specific_support_mass_max.item()
#             ),
#             "drift_identity_correction_relative_mean": float(
#                 final_alignment.identity_correction_relative_mean.item()
#             ),
#             "drift_identity_correction_relative_max": float(
#                 final_alignment.identity_correction_relative_max.item()
#             ),
#             "drift_variation_correction_relative_mean": float(
#                 final_alignment.variation_correction_relative_mean.item()
#             ),
#             "drift_variation_correction_relative_max": float(
#                 final_alignment.variation_correction_relative_max.item()
#             ),
#             "drift_variation_negative_fraction_before_sanitize": float(
#                 final_alignment.variation_negative_fraction_before_sanitize.item()
#             ),
#             "final_joint_classification_loss": float(final_record["classification_loss"]),
#             "final_whole_cds_classifier_loss": float(final_record["whole_cds_classifier_loss"]),
#             "final_old_cds_classifier_loss": float(final_record["old_cds_classifier_loss"]),
#             "final_new_cds_classifier_loss": float(final_record["new_cds_classifier_loss"]),
#             "final_contrastive_loss": float(final_record["contrastive_loss"]),
#             "final_q_consistency_loss": float(final_record["q_consistency_loss"]),
#             "final_kd_logit_loss": float(final_record["kd_logit_loss"]),
#             "final_cds_preservation_loss": float(final_record["cds_preservation_loss"]),
#             "final_cds_intra_loss": float(final_record["cds_intra_loss"]),
#             "final_cds_inter_loss": float(final_record["cds_inter_loss"]),
#             "final_classifier_structural_loss": float(
#                 final_structural.loss.detach().item()
#             ),
#             "final_global_cds_classifier_loss": float(
#                 final_structural.global_reconstruction_loss.detach().item()
#             ),
#             "final_old_cds_classifier_component": float(
#                 final_structural.old_class_reconstruction_loss.detach().item()
#             ),
#             "final_new_cds_classifier_component": float(
#                 final_structural.new_class_reconstruction_loss.detach().item()
#             ),
#             "final_old_old_retention_loss": float(
#                 final_structural.old_old_retention_loss.detach().item()
#             ),
#             "initial_old_new_violation_fraction": float(
#                 pre_structural.old_new_violation_fraction.detach().item()
#             ),
#             "final_old_new_violation_fraction": float(
#                 final_structural.old_new_violation_fraction.detach().item()
#             ),
#             "initial_old_old_violation_fraction": float(
#                 pre_structural.historical_internal_violation_fraction.detach().item()
#             ),
#             "final_old_old_violation_fraction": float(
#                 final_structural.historical_internal_violation_fraction.detach().item()
#             ),
#             "final_old_old_clearance_mean": float(
#                 final_structural.historical_internal_clearance_mean.detach().item()
#             ),
#             "final_old_old_clearance_min": float(
#                 final_structural.historical_internal_clearance_min.detach().item()
#             ),
#             "final_old_old_retention_mean": float(
#                 final_structural.historical_internal_retention_mean.detach().item()
#             ),
#             "final_old_old_retention_min": float(
#                 final_structural.historical_internal_retention_min.detach().item()
#             ),
#             "final_whole_cds_learned_risk_mean": float(
#                 final_structural.learned_pair_risk_mean.detach().item()
#             ),
#             "final_whole_cds_physical_risk_mean": float(
#                 final_structural.physical_pair_risk_mean.detach().item()
#             ),
#             "final_physical_absolute_support_mean": float(
#                 final_record["physical_absolute_support_mean"]
#             ),
#             "final_physical_relation_support_mean": float(
#                 final_record["physical_relation_support_mean"]
#             ),
#             "final_physical_relevance_mean": float(
#                 final_record["physical_relevance_mean"]
#             ),
#             "historical_classifier_row_max_change": self._max_parameter_change(
#                 old_weight_before, final_old_weight
#             ),
#             "incoming_classifier_initialization_max_change": self._max_parameter_change(
#                 new_weight_random_before, phase_start_new_weight
#             ),
#             "incoming_classifier_training_max_change": self._max_parameter_change(
#                 phase_start_new_weight, final_new_weight
#             ),
#             "incoming_initial_clearance_min": float(
#                 initialization.joint_clearance_min.item()
#             ),
#             "incoming_initial_clearance_mean": float(
#                 initialization.joint_clearance_mean.item()
#             ),
#             "phase_start_old_ba": float(old_before["balanced_accuracy"]),
#             "phase_start_old_all_seen_ba": float(
#                 phase_start_old_all_seen["balanced_accuracy"]
#             ),
#             "phase_start_new_only_ba": float(
#                 phase_start_new_only["balanced_accuracy"]
#             ),
#             "phase_start_new_all_seen_ba": float(
#                 phase_start_new_all_seen["balanced_accuracy"]
#             ),
#             "final_old_only_ba": float(final_record["old_only_ba"]),
#             "final_old_all_seen_ba": float(final_record["old_all_seen_ba"]),
#             "final_new_only_ba": float(final_record["new_only_ba"]),
#             "final_new_all_seen_ba": float(final_record["new_all_seen_ba"]),
#             "classifier_logit_scale_frozen_incremental": True,
#             "second_classifier_stage": False,
#         }

#         return {
#             "phase": phase_id,
#             "old_class_ids": list(old_class_ids),
#             "new_class_ids": list(new_class_ids),
#             "seen_class_ids": list(seen_class_ids),
#             "history": history,
#             "final_epoch": epoch_count,
#             "cumulative_test": cumulative,
#             "old_new": groups,
#             "forgetting": forgetting,
#             "phase_origin_confusion": phase_origin,
#             "method_state": method_state,
#             "structure_committed": True,
#             "phase_finalized": True,
#         }
# __all__ = ["IncrementalPhaseTrainer"]
