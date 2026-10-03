from __future__ import annotations

import math

import sys

from numbers import Integral, Real

from typing import Any, Dict


import torch

from tqdm import tqdm


from losses.loss import (
    classification_objective,
    cross_phase_cds_separation_loss,
    historical_cds_preservation_loss,
    logit_distillation_loss,
    ordered_feature_distillation_loss,
)


from models.classifier import CosineClassificationHead


from models.cds_decision_region import whole_cds_class_balanced_classifier_loss


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
            "classifier_history",
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

            raise RuntimeError(
                "seen classes must equal old classes followed by new classes"
            )

        if tuple(self.model.committed_class_ids) != old_class_ids:

            raise RuntimeError("committed historical CDS does not match old classes")

        return old_class_ids, new_class_ids, seen_class_ids

    @staticmethod
    def _assert_frozen_reference(reference_backbone: torch.nn.Module) -> None:

        reference_backbone.eval()

        if any(
            parameter.requires_grad for parameter in reference_backbone.parameters()
        ):

            raise RuntimeError("previous backbone reference must remain frozen")

        if any(
            parameter.grad is not None for parameter in reference_backbone.parameters()
        ):

            raise RuntimeError("previous backbone reference must not carry gradients")

    @staticmethod
    def _assert_frozen_reference_classifier(
        reference_classifier: CosineClassificationHead,
    ) -> None:

        reference_classifier.eval()

        if any(
            parameter.requires_grad for parameter in reference_classifier.parameters()
        ):

            raise RuntimeError("previous classifier reference must remain frozen")

        if any(
            parameter.grad is not None
            for parameter in reference_classifier.parameters()
        ):

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

            if not torch.equal(
                ref_state[key].detach().cpu(), cur_state[key].detach().cpu()
            ):

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

            observed = set(
                int(value) for value in labels.unique().detach().cpu().tolist()
            )

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

            raise RuntimeError(
                "incremental TRAIN split is empty or misses a current class"
            )

        class_count = len(new_class_ids)

        weights = torch.tensor(
            [
                total / float(class_count * counts[class_id])
                for class_id in new_class_ids
            ],
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
    def _max_direction_change(before: Tensor, after: Tensor) -> float:

        if before.shape != after.shape:

            raise ValueError("parameter snapshots have different shapes")

        if before.numel() == 0:

            return 0.0

        eps = torch.finfo(before.dtype).eps

        before_unit = torch.nn.functional.normalize(before, p=2, dim=1, eps=eps)

        after_unit = torch.nn.functional.normalize(after, p=2, dim=1, eps=eps)

        return float((after_unit - before_unit).abs().max().item())

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
            "kd_logit_weight", self.cfg("kd_logit_weight", 0.0)
        )

        kd_temperature = _finite_positive(
            "kd_temperature", self.cfg("kd_temperature", 2.0)
        )

        cds_weight = _finite_nonnegative("cds_weight", self.cfg("cds_weight", 1.0))

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

        old_class_tensor = torch.tensor(
            old_class_ids, device=self.device, dtype=torch.long
        )

        new_class_tensor = torch.tensor(
            new_class_ids, device=self.device, dtype=torch.long
        )

        seen_class_tensor = torch.tensor(
            seen_class_ids, device=self.device, dtype=torch.long
        )

        historical_structure = _clone_structure(
            self.model.structure_bank.class_structures(old_class_ids)
        )

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
            sorted(
                int(value)
                for value in reference_labels.unique().detach().cpu().tolist()
            )
        )

        if observed_new != tuple(sorted(new_class_ids)):

            raise RuntimeError(
                "incremental TRAIN rows do not match current phase classes"
            )

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

        old_weight_before = classifier.class_weight(old_class_tensor).detach().clone()

        new_weight_phase_start = (
            classifier.class_weight(new_class_tensor).detach().clone()
        )

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
            old_reporting_loader,
            class_ids=seen_class_ids,
            target_class_ids=old_class_ids,
        )

        print(
            f"[Phase {phase_id} start] "
            f"OldBA(previous-only)={100*float(old_before['balanced_accuracy']):.2f}% "
            f"OldBA(all-seen)={100*float(phase_start_old_all_seen['balanced_accuracy']):.2f}% "
            f"NewBA(new-only)={100*float(phase_start_new_only['balanced_accuracy']):.2f}% "
            f"NewBA(all-seen)={100*float(phase_start_new_all_seen['balanced_accuracy']):.2f}% | "
            "Stage1=real-new all-seen CE (old rows frozen) + FeatureKD + cross-phase Whole-CDS separation/preservation; Stage2=one-time CDS coordinate correction; no post-hoc classifier reconstruction",
            file=sys.stdout,
        )

        @torch.no_grad()
        def collect_current_structure():

            self.model.eval()

            rows = self.collect_ordered_features(train_eval_loader)

            if not torch.equal(rows["labels"], reference_labels):

                raise RuntimeError("current TRAIN label order changed")

            if not torch.equal(rows["raw_center_spectra"], reference_spectra):

                raise RuntimeError("current TRAIN spectra changed")

            current_features_full = rows["ordered_features"].detach()

            current_structure = self.model.estimate_class_structure(
                current_features_full,
                None,
                reference_labels,
                class_ids=new_class_ids,
                physical_structure=current_physical_structure,
            )

            return current_features_full, current_structure

        # Stage 1 uses one fixed previous-coordinate Whole-CDS reference. Historical

        # CDS is not transported while E_t is still moving.

        stage1_structure = merge_class_structure_batches(
            (historical_structure, current_reference_structure)
        )

        stage1_ids = tuple(
            int(v) for v in stage1_structure.class_ids.detach().cpu().tolist()
        )

        if stage1_ids != seen_class_ids:

            raise RuntimeError("Stage-1 CDS produced incorrect seen-class order")

        stage1_seen_cds = self.model.structure_bank.build_cds(stage1_structure)

        # Stage 1: representation learning. Historical/future classifier rows are

        # fixed; only incoming W rows receive real-current CE gradients.

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

        history: list[dict[str, Any]] = []

        gradient_contract_checked = False

        progress = tqdm(
            range(epoch_count),
            desc=f"Phase {phase_id} representation training",
            unit="epoch",
            dynamic_ncols=True,
            file=sys.stdout,
        )

        for epoch in progress:

            epoch_historical_structure = historical_structure

            epoch_seen_cds = stage1_seen_cds

            self.model.train()

            sums = {
                name: 0.0
                for name in (
                    "classification",
                    "feature_kd",
                    "feature_rms",
                    "contrastive",
                    "kd",
                    "cds",
                    "intra",
                    "inter",
                    "physical_abs",
                    "physical_rel",
                    "physical_joint",
                    "sep_margin",
                    "sep_violation",
                    "sep_positive",
                    "sep_current_negative",
                    "sep_historical_negative",
                    "sep_valid",
                    "sep_physical_confusability",
                    "sep_pair_risk",
                    "total",
                )
            }

            sample_count = 0

            correct_new = 0

            for batch in train_loader:

                patch, spectrum, labels = self.unpack_batch(batch)

                if not bool(torch.isin(labels, new_class_tensor).all()):

                    raise RuntimeError(
                        "incremental TRAIN batch contains an old or future class"
                    )

                optimizer.zero_grad(set_to_none=True)

                classifier.logit_scale_parameter.grad = None

                with torch.no_grad():

                    reference_features = self.model.extract_features_with_backbone(
                        reference_backbone,
                        patch,
                        center_spectrum=spectrum,
                        return_aux=False,
                    ).ordered_features.detach()

                    physical_evidence = self.model.physical_spectral_evidence(
                        spectrum
                    ).detach()

                    teacher_old_logits = (
                        reference_classifier(
                            reference_features,
                            class_ids=old_class_tensor,
                            trainable_class_ids=(),
                        ).logits.detach()
                        if kd_logit_weight > 0
                        else None
                    )

                current_features = self.model.extract_features(
                    patch, center_spectrum=spectrum, return_aux=False
                ).ordered_features

                current_local = _targets_local(labels, new_class_tensor)

                sample_weights = current_class_weights.index_select(0, current_local)

                # Real current HSI is optimized in the deployed all-seen decision

                # space. Historical rows are fixed rivals: CE may move Q_new and

                # W_new, but current-only labels can never move W_old.

                current_output = classifier(
                    current_features,
                    class_ids=seen_class_tensor,
                    trainable_class_ids=new_class_tensor,
                )

                classification = classification_objective(
                    logits=current_output.logits,
                    class_ids=current_output.class_ids,
                    labels_global=labels,
                    sample_weights=sample_weights,
                )

                # Explicitly retained by design: paired-current feature KD keeps the

                # single ordered Q coordinate stable while allowing plasticity.

                feature_kd = ordered_feature_distillation_loss(
                    reference_features=reference_features,
                    current_features=current_features,
                    sample_weights=sample_weights,
                )

                if kd_logit_weight > 0:

                    if teacher_old_logits is None:

                        raise RuntimeError(
                            "teacher old logits are missing while logit KD is enabled"
                        )

                    student_old_logits = classifier(
                        current_features,
                        class_ids=old_class_tensor,
                        trainable_class_ids=(),
                    ).logits

                    kd_loss = logit_distillation_loss(
                        student_old_logits=student_old_logits,
                        teacher_old_logits=teacher_old_logits,
                        temperature=kd_temperature,
                        sample_weights=sample_weights,
                    ).loss

                else:

                    kd_loss = current_features.sum() * 0.0

                cds = historical_cds_preservation_loss(
                    reference_features=reference_features,
                    current_features=current_features,
                    physical_spectral_evidence=physical_evidence,
                    reference_historical_structure=historical_structure,
                    current_historical_structure=epoch_historical_structure,
                )

                separation = cross_phase_cds_separation_loss(
                    current_features=current_features,
                    labels_global=labels,
                    physical_spectral_evidence=physical_evidence,
                    seen_cds=epoch_seen_cds,
                    historical_class_count=len(old_class_ids),
                    logit_scale=classifier.logit_scale,
                    sample_weights=sample_weights,
                )

                if not gradient_contract_checked:

                    tolerance = 64.0 * torch.finfo(classifier.weight.dtype).eps

                    ce_grad = torch.autograd.grad(
                        classification.loss,
                        classifier.weight,
                        retain_graph=True,
                        allow_unused=True,
                    )[0]

                    ce_old = self._row_grad_max(ce_grad, old_class_tensor)

                    ce_new = self._row_grad_max(ce_grad, new_class_tensor)

                    ce_future = self._row_grad_max(ce_grad, future_class_tensor)

                    if ce_old > tolerance:

                        raise RuntimeError(
                            "Stage-1 all-seen current CE leaked into historical classifier rows"
                        )

                    if ce_new <= tolerance:

                        raise RuntimeError(
                            "Stage-1 all-seen current CE produced no incoming-row gradient"
                        )

                    if ce_future > tolerance:

                        raise RuntimeError(
                            "Stage-1 all-seen current CE leaked into future classifier rows"
                        )

                    for name, objective in (
                        ("Feature KD", feature_kd.loss),
                        ("Cross-phase Whole-CDS separation", separation.loss),
                        ("Whole-CDS preservation", cds.loss),
                    ):

                        grad = torch.autograd.grad(
                            objective,
                            classifier.weight,
                            retain_graph=True,
                            allow_unused=True,
                        )[0]

                        if grad is not None and bool(torch.count_nonzero(grad)):

                            raise RuntimeError(
                                f"{name} must not update classifier rows"
                            )

                    q_objectives = [
                        ("real-new all-seen CE", classification.loss),
                        ("Feature KD", feature_kd.loss),
                        ("Cross-phase Whole-CDS separation", separation.loss),
                        ("Whole-CDS preservation", cds.loss),
                    ]

                    if kd_logit_weight > 0:

                        q_objectives.append(("old-logit KD", kd_loss))

                    for name, objective in q_objectives:

                        q_grad = torch.autograd.grad(
                            objective,
                            current_features,
                            retain_graph=True,
                            allow_unused=True,
                        )[0]

                        if q_grad is None or not bool(torch.isfinite(q_grad).all()):

                            raise RuntimeError(
                                f"{name} produced invalid current-Q gradient"
                            )

                    print(
                        "[Incremental gradient ownership] "
                        f"CE old={ce_old:.3e} new={ce_new:.3e} future={ce_future:.3e} | "
                        "FeatureKD/CDS-separation/CDS-preservation classifier=0",
                        file=sys.stdout,
                    )

                    gradient_contract_checked = True

                loss = (
                    classification.loss
                    + contrastive_weight * separation.loss
                    + feature_kd_weight * feature_kd.loss
                    + kd_logit_weight * kd_loss
                    + cds_weight * cds.loss
                )

                if not bool(torch.isfinite(loss)):

                    raise RuntimeError(
                        "incremental representation objective became NaN or Inf"
                    )

                loss.backward()

                if classifier.logit_scale_parameter.grad is not None:

                    raise RuntimeError(
                        "incremental training must not update cosine logit scale"
                    )

                if classifier.weight.grad is None or not bool(
                    torch.isfinite(classifier.weight.grad).all()
                ):

                    raise RuntimeError(
                        "incoming classifier gradient is missing/non-finite"
                    )

                if self._row_grad_max(classifier.weight.grad, old_class_tensor) > 0.0:

                    raise RuntimeError(
                        "representation stage changed historical classifier rows"
                    )

                if (
                    self._row_grad_max(classifier.weight.grad, future_class_tensor)
                    > 0.0
                ):

                    raise RuntimeError(
                        "representation stage changed future classifier rows"
                    )

                backbone_grad_seen = False

                for parameter in backbone_parameters:

                    if parameter.grad is None:

                        continue

                    if not bool(torch.isfinite(parameter.grad).all()):

                        raise RuntimeError(
                            "incremental backbone gradient contains NaN or Inf"
                        )

                    backbone_grad_seen = backbone_grad_seen or bool(
                        torch.count_nonzero(parameter.grad)
                    )

                if not backbone_grad_seen:

                    raise RuntimeError(
                        "representation objective produced zero backbone gradient"
                    )

                if gradient_clip > 0:

                    torch.nn.utils.clip_grad_norm_(
                        backbone_parameters + [classifier.weight],
                        gradient_clip,
                        error_if_nonfinite=True,
                    )

                optimizer.step()

                self._assert_future_rows_unchanged(
                    classifier, future_class_tensor, future_weight_before
                )

                if not torch.equal(
                    classifier.class_weight(old_class_tensor).detach(),
                    old_weight_before,
                ):

                    raise RuntimeError(
                        "historical classifier rows moved during representation training"
                    )

                n = int(labels.numel())

                sums["classification"] += float(classification.loss.detach().item()) * n

                sums["feature_kd"] += float(feature_kd.loss.detach().item()) * n

                sums["feature_rms"] += float(feature_kd.rms_drift.detach().item()) * n

                sums["contrastive"] += float(separation.loss.detach().item()) * n

                sums["kd"] += float(kd_loss.detach().item()) * n

                sums["cds"] += float(cds.loss.detach().item()) * n

                sums["intra"] += float(cds.intra_loss.detach().item()) * n

                sums["inter"] += float(cds.inter_loss.detach().item()) * n

                sums["physical_abs"] += (
                    float(cds.physical_absolute_support_mean.detach().item()) * n
                )

                sums["physical_rel"] += (
                    float(cds.physical_relation_support_mean.detach().item()) * n
                )

                sums["physical_joint"] += (
                    float(cds.physical_relevance_mean.detach().item()) * n
                )

                sums["sep_margin"] += (
                    float(separation.learned_support_margin_mean.detach().item()) * n
                )

                sums["sep_violation"] += (
                    float(separation.violation_fraction.detach().item()) * n
                )

                sums["sep_positive"] += (
                    float(separation.learned_positive_support_mean.detach().item()) * n
                )

                # New/new competition is already handled by real all-seen CE;

                # this CDS loss contains only current-positive vs historical rivals.

                sums["sep_current_negative"] += 0.0

                sums["sep_historical_negative"] += (
                    float(
                        separation.learned_historical_rival_support_mean.detach().item()
                    )
                    * n
                )

                sums["sep_valid"] += 1.0 * n

                sums["sep_physical_confusability"] += (
                    float(separation.physical_confusability_mean.detach().item()) * n
                )

                sums["sep_pair_risk"] += (
                    float(separation.effective_pair_risk_mean.detach().item()) * n
                )

                sums["total"] += float(loss.detach().item()) * n

                correct_new += int(
                    current_output.prediction.detach().eq(labels).sum().item()
                )

                sample_count += n

            if sample_count != expected_samples:

                raise RuntimeError(
                    "incremental TRAIN loader did not traverse the complete split"
                )

            scheduler.step()

            if float(classifier.logit_scale.detach().item()) != phase_start_scale:

                raise RuntimeError(
                    "cosine logit scale changed during representation training"
                )

            self._assert_future_rows_unchanged(
                classifier, future_class_tensor, future_weight_before
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
                "stage": "representation",
                "epoch": epoch + 1,
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "classification_loss": sums["classification"] / sample_count,
                "feature_kd_loss": sums["feature_kd"] / sample_count,
                "coordinate_stability_loss": sums["feature_kd"] / sample_count,
                "epoch_representation_drift_rms": sums["feature_rms"] / sample_count,
                "contrastive_loss": sums["contrastive"] / sample_count,
                "cds_separation_loss": sums["contrastive"] / sample_count,
                "cds_separation_margin": sums["sep_margin"] / sample_count,
                "cds_separation_violation_fraction": sums["sep_violation"]
                / sample_count,
                "contrastive_positive_similarity": sums["sep_positive"] / sample_count,
                "contrastive_current_negative_similarity": sums["sep_current_negative"]
                / sample_count,
                "contrastive_historical_negative_similarity": sums[
                    "sep_historical_negative"
                ]
                / sample_count,
                "contrastive_valid_anchor_fraction": sums["sep_valid"] / sample_count,
                "physical_confusability_mean": sums["sep_physical_confusability"]
                / sample_count,
                "effective_pair_risk_mean": sums["sep_pair_risk"] / sample_count,
                "kd_logit_loss": sums["kd"] / sample_count,
                "cds_preservation_loss": sums["cds"] / sample_count,
                "cds_intra_loss": sums["intra"] / sample_count,
                "cds_inter_loss": sums["inter"] / sample_count,
                "physical_absolute_support_mean": sums["physical_abs"] / sample_count,
                "physical_relation_support_mean": sums["physical_rel"] / sample_count,
                "physical_relevance_mean": sums["physical_joint"] / sample_count,
                "total_loss": sums["total"] / sample_count,
                "current_new_all_seen_accuracy": correct_new / float(sample_count),
                "old_only_ba": float(old_only["balanced_accuracy"]),
                "old_all_seen_ba": float(old_all["balanced_accuracy"]),
                "new_only_ba": float(new_only["balanced_accuracy"]),
                "new_all_seen_ba": float(new_all["balanced_accuracy"]),
                "validation_new_all_seen_ba": (
                    None
                    if validation_all is None
                    else float(validation_all["balanced_accuracy"])
                ),
            }

            history.append(record)

            tqdm.write(
                f"[Representation {epoch + 1:03d}/{epoch_count:03d}] "
                f"CEall={record['classification_loss']:.4f} "
                f"Sep={record['cds_separation_loss']:.4f} "
                f"FKD={record['feature_kd_loss']:.4f} "
                f"KD={record['kd_logit_loss']:.4f} "
                f"CDS={record['cds_preservation_loss']:.4f}"
                f"[I={record['cds_intra_loss']:.4f},R={record['cds_inter_loss']:.4f}] "
                f"SepMargin={record['cds_separation_margin']:.4f} "
                f"SepViol={100*record['cds_separation_violation_fraction']:.1f}% | "
                f"OldOnly/All={100*record['old_only_ba']:.2f}%/"
                f"{100*record['old_all_seen_ba']:.2f}% "
                f"NewOnly/All={100*record['new_only_ba']:.2f}%/"
                f"{100*record['new_all_seen_ba']:.2f}%",
                file=sys.stdout,
            )

        progress.close()

        if not history:

            raise RuntimeError("incremental representation training produced no epochs")

        # Stage 2: freeze E_t, then synchronize Whole-CDS exactly once.

        self.model.eval()

        for parameter in self.model.backbone.parameters():

            parameter.requires_grad_(False)

            parameter.grad = None

        final_features_full, final_current_structure = collect_current_structure()

        final_historical_structure, _final_transform, final_alignment = (
            correct_historical_structure(
                historical_structure=historical_structure,
                current_reference_structure=current_reference_structure,
                current_structure=final_current_structure,
                reference_features=reference_features_full,
                current_features=final_features_full,
            )
        )

        committed_structure = merge_class_structure_batches(
            (final_historical_structure, final_current_structure)
        )

        committed_ids = tuple(
            int(v) for v in committed_structure.class_ids.detach().cpu().tolist()
        )

        if committed_ids != seen_class_ids:

            raise RuntimeError("final Whole-CDS produced incorrect seen-class order")

        committed_seen_cds = self.model.structure_bank.build_cds(committed_structure)

        if not torch.equal(
            final_historical_structure.physical_identity,
            historical_structure.physical_identity,
        ) or not torch.equal(
            final_historical_structure.physical_variation,
            historical_structure.physical_variation,
        ):

            raise RuntimeError("final historical correction changed physical I_P/V_P")

        post_rep_old_only = self.evaluate_loader(
            old_reporting_loader,
            class_ids=old_class_ids,
            target_class_ids=old_class_ids,
        )

        post_rep_old_all = self.evaluate_loader(
            old_reporting_loader,
            class_ids=seen_class_ids,
            target_class_ids=old_class_ids,
        )

        post_rep_new_only = self.evaluate_loader(
            train_eval_loader, class_ids=new_class_ids, target_class_ids=new_class_ids
        )

        post_rep_new_all = self.evaluate_loader(
            train_eval_loader, class_ids=seen_class_ids, target_class_ids=new_class_ids
        )

        # Final classifier state: no post-hoc reconstruction. Stage 1 already
        # learned W_new against fixed W_old in the deployed all-seen softmax.
        # Stage 2 only synchronizes persistent Whole-CDS for future phases.
        old_weight_after_training = (
            classifier.class_weight(old_class_tensor).detach().clone()
        )
        new_weight_after_training = (
            classifier.class_weight(new_class_tensor).detach().clone()
        )
        if not torch.equal(old_weight_after_training, old_weight_before):
            raise RuntimeError(
                "historical classifier rows changed during incremental admission"
            )
        self._assert_future_rows_unchanged(
            classifier, future_class_tensor, future_weight_before
        )
        if float(classifier.logit_scale.detach().item()) != phase_start_scale:
            raise RuntimeError("incremental admission changed cosine logit scale")

        # Whole-CDS remains a read-only structural diagnostic. It does not rebuild W.
        final_classifier_objective = whole_cds_class_balanced_classifier_loss(
            classifier=classifier,
            cds=committed_seen_cds,
            historical_class_count=len(old_class_ids),
        )
        pre_finalize = final_classifier_objective
        classifier_history: list[dict[str, float]] = []
        stage3_solver_iterations = 0

        # Backward-compatible report scalars. Initial == final because there is no
        # post-hoc classifier mutation. These keep existing main.py/reporting alive.
        old_count = len(old_class_ids)
        per_class_cds = final_classifier_objective.per_class_loss.detach()
        diagnostic_old_cds = per_class_cds[:old_count].mean()
        diagnostic_new_cds = per_class_cds[old_count:].mean()
        with torch.no_grad():
            diagnostic_seen = classifier(
                final_features_full,
                class_ids=seen_class_tensor,
                trainable_class_ids=(),
            )
            diagnostic_new_ce = classification_objective(
                logits=diagnostic_seen.logits,
                class_ids=diagnostic_seen.class_ids,
                labels_global=reference_labels,
            ).loss
            diagnostic_new_all_accuracy = (
                diagnostic_seen.prediction.eq(reference_labels)
                .to(diagnostic_seen.logits.dtype)
                .mean()
            )
            diagnostic_new_only = classifier(
                final_features_full,
                class_ids=new_class_tensor,
                trainable_class_ids=(),
            )
            diagnostic_new_only_accuracy = (
                diagnostic_new_only.prediction.eq(reference_labels)
                .to(diagnostic_new_only.logits.dtype)
                .mean()
            )

        print(
            "[Final all-seen admission] "
            f"OldOnly/All={100*float(post_rep_old_only['balanced_accuracy']):.2f}%/"
            f"{100*float(post_rep_old_all['balanced_accuracy']):.2f}% "
            f"NewOnly/All={100*float(post_rep_new_only['balanced_accuracy']):.2f}%/"
            f"{100*float(post_rep_new_all['balanced_accuracy']):.2f}% "
            f"NewCE={float(diagnostic_new_ce.detach().item()):.4f} "
            f"CDSViol={100*float(final_classifier_objective.violation_fraction.detach().item()):.1f}% "
            f"OldNewViol={100*float(final_classifier_objective.old_new_violation_fraction.detach().item()):.1f}%",
            file=sys.stdout,
        )

        # Step 11: commit only after representation, coordinate correction, CDS

        # construction and classifier finalization are all complete.

        self.model.replace_class_structure(committed_structure)

        self.model.validate_finalized_state()

        self.dataset.finalize_phase(phase_id)

        classifier.logit_scale_parameter.requires_grad_(scale_requires_grad)

        classifier.logit_scale_parameter.grad = None

        # Step 12: all-seen evaluation uses exactly tau*cosine(Q,W).

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

        final_old_only = self.evaluate_loader(
            old_reporting_loader,
            class_ids=old_class_ids,
            target_class_ids=old_class_ids,
        )

        final_old_all = self.evaluate_loader(
            old_reporting_loader,
            class_ids=seen_class_ids,
            target_class_ids=old_class_ids,
        )

        final_new_only = self.evaluate_loader(
            train_eval_loader, class_ids=new_class_ids, target_class_ids=new_class_ids
        )

        final_new_all = self.evaluate_loader(
            train_eval_loader, class_ids=seen_class_ids, target_class_ids=new_class_ids
        )

        final_record = history[-1]

        final_old_weight = classifier.class_weight(old_class_tensor).detach()

        final_new_weight = classifier.class_weight(new_class_tensor).detach()

        method_state = {
            "stores_old_images": False,
            "stores_old_feature_rows": False,
            "stores_synthetic_support": False,
            "uses_covariance_or_gaussian_memory": False,
            "uses_mean_or_second_moment_classifier_memory": False,
            "uses_second_classifier": False,
            "uses_phase_bias": False,
            "uses_all_seen_competition_during_representation": True,
            "historical_classifier_rows_frozen_during_representation": True,
            "uses_posthoc_classifier_reconstruction": False,
            "uses_new_only_representation_ce": False,
            "uses_one_time_post_representation_cds_synchronization": True,
            "uses_whole_cds_classifier_reconstruction": False,
            "uses_real_new_all_seen_classification_in_stage3": False,
            "uses_real_new_all_seen_classification_diagnostic_in_stage3": True,
            "uses_homogeneous_all_seen_whole_cds_stage3": False,
            "uses_finalized_iq_stage3_initialization": False,
            "uses_stage1_classifier_rows_as_stage3_initialization": False,
            "uses_unit_sphere_stage3_solver": False,
            "stage3_solver": "none",
            "prediction_rule": "tau*cosine(Q,W)",
            "representation_epochs": epoch_count,
            "representation_lr": learning_rate,
            "stage3_steps": stage3_solver_iterations,
            "stage3_historical_cds_to_old_grad_norm": 0.0,
            "stage3_historical_cds_to_new_grad_norm": 0.0,
            "stage3_current_cds_to_old_grad_norm": 0.0,
            "stage3_current_cds_to_new_grad_norm": 0.0,
            "contrastive_weight": contrastive_weight,
            "feature_kd_weight": feature_kd_weight,
            "kd_logit_weight": kd_logit_weight,
            "kd_temperature": kd_temperature,
            "cds_weight": cds_weight,
            "final_representation_classification_loss": float(
                final_record["classification_loss"]
            ),
            "final_feature_kd_loss": float(final_record["feature_kd_loss"]),
            "final_contrastive_loss": float(final_record["contrastive_loss"]),
            "final_cds_separation_loss": float(final_record["cds_separation_loss"]),
            "final_cds_separation_margin": float(final_record["cds_separation_margin"]),
            "final_cds_separation_violation_fraction": float(
                final_record["cds_separation_violation_fraction"]
            ),
            "final_kd_logit_loss": float(final_record["kd_logit_loss"]),
            "final_cds_preservation_loss": float(final_record["cds_preservation_loss"]),
            "final_cds_intra_loss": float(final_record["cds_intra_loss"]),
            "final_cds_inter_loss": float(final_record["cds_inter_loss"]),
            "final_physical_absolute_support_mean": float(
                final_record["physical_absolute_support_mean"]
            ),
            "final_physical_relation_support_mean": float(
                final_record["physical_relation_support_mean"]
            ),
            "final_physical_relevance_mean": float(
                final_record["physical_relevance_mean"]
            ),
            "phase_mapping_error": float(final_alignment.mapping_error.item()),
            "representation_drift_rms": float(
                final_alignment.representation_drift_rms.item()
            ),
            "phase_scale_deviation_rms": float(
                final_alignment.scale_deviation_rms.item()
            ),
            "phase_translation_rms": float(final_alignment.translation_rms.item()),
            "drift_learned_relevance_mean": float(
                final_alignment.learned_relevance_mean.item()
            ),
            "drift_physical_relevance_mean": float(
                final_alignment.physical_relevance_mean.item()
            ),
            "drift_effective_relevance_mean": float(
                final_alignment.effective_relevance_mean.item()
            ),
            "drift_support_mass_mean": float(
                final_alignment.class_specific_support_mass_mean.item()
            ),
            "drift_identity_correction_relative_mean": float(
                final_alignment.identity_correction_relative_mean.item()
            ),
            "drift_variation_correction_relative_mean": float(
                final_alignment.variation_correction_relative_mean.item()
            ),
            "stage3_inherited_classifier_cds_loss": float(
                final_classifier_objective.loss.detach().item()
            ),
            "stage3_iq_initialization_cds_loss": float(
                final_classifier_objective.loss.detach().item()
            ),
            "stage3_initial_balanced_loss": float(
                final_classifier_objective.loss.detach().item()
            ),
            "stage3_final_balanced_loss": float(
                final_classifier_objective.loss.detach().item()
            ),
            "stage3_initial_historical_positive_loss": float(
                diagnostic_old_cds.detach().item()
            ),
            "stage3_final_historical_positive_loss": float(
                diagnostic_old_cds.detach().item()
            ),
            "stage3_initial_new_real_ce": float(diagnostic_new_ce.detach().item()),
            "stage3_final_new_real_ce": float(diagnostic_new_ce.detach().item()),
            "stage3_initial_current_cds_positive_loss": float(
                diagnostic_new_cds.detach().item()
            ),
            "stage3_final_current_cds_positive_loss": float(
                diagnostic_new_cds.detach().item()
            ),
            "stage3_initial_new_all_seen_accuracy": float(
                diagnostic_new_all_accuracy.detach().item()
            ),
            "stage3_final_new_all_seen_accuracy": float(
                diagnostic_new_all_accuracy.detach().item()
            ),
            "stage3_initial_new_only_accuracy": float(
                diagnostic_new_only_accuracy.detach().item()
            ),
            "stage3_final_new_only_accuracy": float(
                diagnostic_new_only_accuracy.detach().item()
            ),
            "stage3_initial_cds_loss": float(
                final_classifier_objective.loss.detach().item()
            ),
            "stage3_final_cds_loss": float(
                final_classifier_objective.loss.detach().item()
            ),
            "stage3_initial_old_new_violation_fraction": float(
                final_classifier_objective.old_new_violation_fraction.detach().item()
            ),
            "stage3_final_old_new_violation_fraction": float(
                final_classifier_objective.old_new_violation_fraction.detach().item()
            ),
            "stage3_final_old_old_violation_fraction": float(
                final_classifier_objective.old_old_violation_fraction.detach().item()
            ),
            "stage3_final_new_new_violation_fraction": float(
                final_classifier_objective.new_new_violation_fraction.detach().item()
            ),
            "classifier_pre_stage3_cds_loss": float(
                final_classifier_objective.loss.detach().item()
            ),
            "classifier_post_stage3_cds_loss": float(
                final_classifier_objective.loss.detach().item()
            ),
            "classifier_pre_stage3_violation_fraction": float(
                final_classifier_objective.violation_fraction.detach().item()
            ),
            "classifier_post_stage3_violation_fraction": float(
                final_classifier_objective.violation_fraction.detach().item()
            ),
            "classifier_learned_pair_risk_mean": float(
                final_classifier_objective.learned_pair_risk_mean.detach().item()
            ),
            "classifier_physical_pair_risk_mean": float(
                final_classifier_objective.physical_pair_risk_mean.detach().item()
            ),
            "classifier_effective_pair_risk_mean": float(
                final_classifier_objective.effective_pair_risk_mean.detach().item()
            ),
            "historical_classifier_training_max_change": self._max_parameter_change(
                old_weight_before, final_old_weight
            ),
            "historical_classifier_stage3_change": 0.0,
            "incoming_classifier_training_max_change": self._max_parameter_change(
                new_weight_phase_start, final_new_weight
            ),
            "incoming_classifier_stage3_change": 0.0,
            "phase_start_old_ba": float(old_before["balanced_accuracy"]),
            "phase_start_old_all_seen_ba": float(
                phase_start_old_all_seen["balanced_accuracy"]
            ),
            "phase_start_new_only_ba": float(phase_start_new_only["balanced_accuracy"]),
            "phase_start_new_all_seen_ba": float(
                phase_start_new_all_seen["balanced_accuracy"]
            ),
            "post_rep_old_only_ba": float(post_rep_old_only["balanced_accuracy"]),
            "post_rep_old_all_seen_ba": float(post_rep_old_all["balanced_accuracy"]),
            "post_rep_new_only_ba": float(post_rep_new_only["balanced_accuracy"]),
            "post_rep_new_all_seen_ba": float(post_rep_new_all["balanced_accuracy"]),
            "final_old_only_ba": float(final_old_only["balanced_accuracy"]),
            "final_old_all_seen_ba": float(final_old_all["balanced_accuracy"]),
            "final_new_only_ba": float(final_new_only["balanced_accuracy"]),
            "final_new_all_seen_ba": float(final_new_all["balanced_accuracy"]),
            "classifier_logit_scale_frozen_incremental": True,
        }

        return {
            "phase": phase_id,
            "old_class_ids": list(old_class_ids),
            "new_class_ids": list(new_class_ids),
            "seen_class_ids": list(seen_class_ids),
            "history": history,
            "classifier_history": classifier_history,
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
