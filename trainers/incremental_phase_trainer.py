from __future__ import annotations
from copy import deepcopy
from dataclasses import replace
import math
import sys
import warnings
from numbers import Integral, Real
from typing import Any, Dict
import torch
from tqdm import tqdm

from losses.loss import (
    classification_objective,
    cross_phase_cds_separation_loss,
    whole_cds_contrastive_loss,
    historical_cds_preservation_loss,
    logit_distillation_loss,
    ordered_feature_distillation_loss,
    feature_overlap_audit,
    fixed_q_classifier_probe,
    deterministic_class_audit_indices,
)

from models.classifier import CosineClassificationHead
from models.cds_decision_region import (
    whole_cds_class_balanced_classifier_loss,
    whole_cds_joint_classifier_state,
    feasible_directional_head_objective,
    current_class_feasible_radii,
    resolve_classifier_gradient_conflict,
)

from models.geometry_bank import (
    ClassStructureBatch,
    merge_class_structure_batches,
    whole_cds_pair_risk,
    normalize_ordered_decision_features,
)
from models.phase_coordinate_alignment import (
    correct_historical_structure,
    rotate_historical_flat_rows,
    evaluate_fixed_phase_coordinate_transform,
)
from models.historical_cds_support import HistoricalCDSSupportBank
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
    """Clone the complete committed class structure without altering Q or P."""
    if not isinstance(structure, ClassStructureBatch) or structure.class_count <= 0:
        raise ValueError("cannot clone an empty class structure")
    from dataclasses import fields

    return ClassStructureBatch(
        **{
            field.name: getattr(structure, field.name).detach().clone()
            for field in fields(ClassStructureBatch)
        }
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

    @staticmethod
    def parameter_gradient_audit(objectives, backbone_parameters):
        """Detached parameter-space gradient norms, without optimizer mutation.

        Q-space gradient cosines alone do not characterize backbone parameter
        interference. This audit measures the real trainable parameter groups.
        """
        result = {}
        for name, objective in objectives.items():
            if not objective.requires_grad:
                result[name] = {"grad_norm": 0.0, "active_parameters": 0}
                continue
            grads = torch.autograd.grad(
                objective,
                backbone_parameters,
                retain_graph=True,
                allow_unused=True,
            )
            norms = [
                grad.detach().float().square().sum()
                for grad in grads
                if grad is not None
            ]
            total = torch.stack(norms).sum().sqrt().item() if norms else 0.0
            result[name] = {
                "grad_norm": float(total),
                "active_parameters": sum(g is not None for g in grads),
            }
        return result

    @torch.no_grad()
    def audit_current_classifier_risk(self, loader, *, class_ids, cds):
        """Report moment approximation error on available current VAL only.

        Do not use old test examples to tune historical risk or certify drift.
        """
        if loader is None:
            return {"status": "skipped_no_validation", "used_for_selection": False}
        observed = self.collect_ordered_features(loader)
        return self.historical_support.audit_expected_ce(
            self._require_classifier(),
            class_ids,
            cds,
            observed["ordered_features"],
            observed["labels"],
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

            raise RuntimeError(
                f"previous/current {name} state keys differ at phase start"
            )

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

    def _run_new_class_head_admission(
        self,
        *,
        classifier: CosineClassificationHead,
        train_loader: Any,
        train_eval_loader: Any,
        old_class_ids: tuple[int, ...],
        new_class_ids: tuple[int, ...],
        seen_class_ids: tuple[int, ...],
        current_class_weights: Tensor,
        epochs: int,
        learning_rate: float,
        gradient_clip: float,
    ) -> dict[str, float | int]:
        """Train only incoming rows against ALL seen rows with real current HSI.

        Backbone Q, old W, future W and tau remain frozen. No old images,
        feature replay or pseudo-samples enter this stage. The caller must
        invoke it before the joint incremental optimizer is created.
        """
        if epochs < 0:
            raise ValueError("head admission epochs must be nonnegative")
        if epochs == 0:
            return {"epochs": 0, "steps": 0}
        if not math.isfinite(learning_rate) or learning_rate <= 0:
            raise ValueError("head admission learning rate must be positive")

        old_ids = torch.tensor(old_class_ids, dtype=torch.long, device=self.device)
        new_ids = torch.tensor(new_class_ids, dtype=torch.long, device=self.device)
        seen_ids = torch.tensor(seen_class_ids, dtype=torch.long, device=self.device)
        future_ids = torch.tensor(
            sorted(set(range(classifier.total_class_count)) - set(seen_class_ids)),
            dtype=torch.long,
            device=self.device,
        )
        old_snapshot = classifier.class_weight(old_ids).detach().clone()
        future_snapshot = (
            classifier.class_weight(future_ids).detach().clone()
            if future_ids.numel()
            else None
        )
        tau_snapshot = classifier.logit_scale_parameter.detach().clone()
        backbone_snapshot = {
            k: v.detach().clone() for k, v in self.model.backbone.state_dict().items()
        }

        @torch.no_grad()
        def margin_diagnostics() -> dict[str, float]:
            """Observe TRUE current-class logit relative to strongest old logit."""
            self.model.eval()
            correct_margin_sum = 0.0
            loss_sum = 0.0
            count = 0
            for batch in train_eval_loader:
                patch, spectrum, labels = self.unpack_batch(batch)
                if not bool(torch.isin(labels, new_ids).all()):
                    raise RuntimeError(
                        "admission diagnostics include old/future labels"
                    )
                features = self.model.extract_features(
                    patch, center_spectrum=spectrum, return_aux=False
                ).ordered_features
                scores = classifier(
                    features,
                    class_ids=seen_ids,
                    trainable_class_ids=(),
                ).logits
                local = _targets_local(labels, seen_ids)
                true_scores = scores.gather(1, local[:, None]).squeeze(1)
                rival_old = scores[:, : len(old_class_ids)].max(dim=1).values
                correct_margin_sum += float((true_scores - rival_old).sum().item())
                loss_sum += float(
                    torch.nn.functional.cross_entropy(
                        scores, local, reduction="sum"
                    ).item()
                )
                count += int(labels.numel())
            return {
                "new_vs_old_logit_margin": correct_margin_sum / max(1, count),
                "new_all_seen_ce": loss_sum / max(1, count),
            }

        before = margin_diagnostics()
        # The previous teacher is deliberately not needed during head admission:
        # incoming features come from the fixed phase-start backbone.
        self.model.eval()
        optimizer = torch.optim.AdamW(
            [classifier.weight], lr=learning_rate, weight_decay=0.0
        )
        total_loss, total_correct, total_examples, steps = 0.0, 0, 0, 0
        for _ in range(epochs):
            for batch in train_loader:
                patch, spectrum, labels = self.unpack_batch(batch)
                if not bool(torch.isin(labels, new_ids).all()):
                    raise RuntimeError(
                        "head admission saw a non-current training class"
                    )
                optimizer.zero_grad(set_to_none=True)
                with torch.no_grad():
                    features = self.model.extract_features(
                        patch, center_spectrum=spectrum, return_aux=False
                    ).ordered_features.detach()
                logits = classifier(
                    features,
                    class_ids=seen_ids,
                    trainable_class_ids=new_ids,
                ).logits
                local = _targets_local(labels, seen_ids)
                new_local = _targets_local(labels, new_ids)
                weights = current_class_weights.index_select(0, new_local)
                per_sample = torch.nn.functional.cross_entropy(
                    logits, local, reduction="none"
                )
                loss = (per_sample * weights).mean()
                if not bool(torch.isfinite(loss)):
                    raise RuntimeError("nonfinite new-class admission loss")
                loss.backward()
                gradient = classifier.weight.grad
                if gradient is None or not bool(torch.isfinite(gradient).all()):
                    raise RuntimeError("nonfinite or absent admission gradient")
                if self._row_grad_max(gradient, old_ids) != 0.0 or (
                    future_ids.numel()
                    and self._row_grad_max(gradient, future_ids) != 0.0
                ):
                    raise RuntimeError("head admission must update incoming rows only")
                if gradient_clip > 0:
                    torch.nn.utils.clip_grad_norm_([classifier.weight], gradient_clip)
                optimizer.step()
                steps += 1
                n = int(labels.numel())
                total_examples += n
                total_loss += float(loss.detach().item()) * n
                total_correct += int(logits.argmax(dim=1).eq(local).sum().item())
        classifier.weight.grad = None
        if not torch.equal(classifier.class_weight(old_ids).detach(), old_snapshot):
            raise RuntimeError("head admission modified a historical classifier row")
        if future_snapshot is not None and not torch.equal(
            classifier.class_weight(future_ids).detach(), future_snapshot
        ):
            raise RuntimeError("head admission modified a future classifier row")
        if not torch.equal(classifier.logit_scale_parameter.detach(), tau_snapshot):
            raise RuntimeError("head admission modified the shared cosine temperature")
        if any(
            not torch.equal(value, backbone_snapshot[key])
            for key, value in self.model.backbone.state_dict().items()
        ):
            raise RuntimeError("head admission modified the feature backbone")

        after = self.evaluate_loader(
            train_eval_loader,
            class_ids=seen_class_ids,
            target_class_ids=new_class_ids,
        )
        after_margin = margin_diagnostics()
        result = {
            "new_vs_old_logit_margin_before": before["new_vs_old_logit_margin"],
            "new_vs_old_logit_margin_after": after_margin["new_vs_old_logit_margin"],
            "new_all_seen_ce_before": before["new_all_seen_ce"],
            "new_all_seen_ce_after": after_margin["new_all_seen_ce"],
            "epochs": epochs,
            "steps": steps,
            "train_loss": total_loss / max(total_examples, 1),
            "train_all_seen_accuracy": total_correct / max(total_examples, 1),
            "new_all_seen_ba_after": float(after["balanced_accuracy"]),
        }
        print(
            f"[New-class head admission] epochs={epochs} steps={steps} "
            f"all-seen-new-BA={100*result['new_all_seen_ba_after']:.2f}% "
            f"new-vs-old margin={before['new_vs_old_logit_margin']:.4f}"
            f" -> {after_margin['new_vs_old_logit_margin']:.4f} "
            f"CE={before['new_all_seen_ce']:.4f} -> {after_margin['new_all_seen_ce']:.4f} "
            "| only W_new updated; Q/W_old/W_future/tau unchanged",
            file=sys.stdout,
        )
        return result

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

        historical_negative_mode = self.cfg("historical_negative_mode", "distribution")
        if historical_negative_mode not in ("classifier", "mean", "distribution"):
            raise ValueError("historical_negative_mode must be classifier, mean or distribution")
        historical_negative_samples = _positive_int(
            self.cfg("historical_negative_samples_per_class", 32),
            name="historical_negative_samples_per_class",
        )
        # Independent stream: sampling must not perturb dropout or loader order.
        # Finalized-phase resume reconstructs the same stream for that phase.
        negative_generator = torch.Generator(device=self.device)
        negative_generator.manual_seed(int(self.cfg("seed", 0)) + phase_id)
        effective_negative_samples = (
            historical_negative_samples if historical_negative_mode == "distribution"
            else (1 if historical_negative_mode == "mean" else 0)
        )

        kd_logit_weight = _finite_nonnegative(
            "kd_logit_weight", self.cfg("kd_logit_weight", 0.0)
        )

        kd_temperature = _finite_positive(
            "kd_temperature", self.cfg("kd_temperature", 2.0)
        )

        cds_weight = _finite_nonnegative("cds_weight", self.cfg("cds_weight", 1.0))

        # Zero is an explicit preservation ablation, consistent with the CLI.
        feature_kd_weight = _finite_nonnegative(
            "feature_kd_weight", self.cfg("feature_kd_weight", 10.0)
        )
        representation_ce_scope = self.cfg("representation_ce_scope", "all_seen")
        if representation_ce_scope not in ("all_seen", "new_only"):
            raise ValueError("representation_ce_scope must be all_seen or new_only")
        configured_joint = self.cfg("representation_train_new_head", True)
        if not isinstance(configured_joint, bool):
            raise ValueError("representation_train_new_head must be boolean")
        if not configured_joint or representation_ce_scope != "all_seen":
            raise ValueError("unified CDS requires all_seen CE and joint seen-head training")
        representation_train_new_head = configured_joint
        raw_head_epochs = self.cfg("support_head_epochs", 0)
        if isinstance(raw_head_epochs, bool) or not isinstance(raw_head_epochs, (Integral, Real)):
            raise ValueError("support_head_epochs must be a non-negative integer")
        if not math.isfinite(float(raw_head_epochs)) or float(raw_head_epochs) < 0 or not float(raw_head_epochs).is_integer():
            raise ValueError("support_head_epochs must be a non-negative integer")
        recon_epochs = int(raw_head_epochs)

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

        self.historical_support.require(old_class_ids)

        historical_structure = _clone_structure(
            self.model.structure_bank.class_structures(old_class_ids)
        )
        historical_pair_risk = whole_cds_pair_risk(
            self.model.class_discriminative_structure(old_class_ids)
        )[0].detach()

        self.model.set_incremental_trainability()
        backbone_trainability = [(p, p.requires_grad) for p in self.model.backbone.parameters()]

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
        # The before-statistics are measured once in the PREVIOUS model space,
        # exactly as required by the paired incoming-feature observation protocol. They are
        # not old-class samples, and are not saved across phases.
        reference_feature_norm = float(
            reference_features_full.reshape(reference_features_full.size(0), -1)
            .norm(dim=1)
            .mean()
            .item()
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

        phase_start_new_only = self.evaluate_loader(
            train_eval_loader, class_ids=new_class_ids, target_class_ids=new_class_ids
        )

        phase_start_new_all_seen = self.evaluate_loader(
            train_eval_loader, class_ids=seen_class_ids, target_class_ids=new_class_ids
        )

        print(
            f"[Phase {phase_id} start] "
            f"NewBA(new-only)={100*float(phase_start_new_only['balanced_accuracy']):.2f}% "
            f"NewBA(all-seen)={100*float(phase_start_new_all_seen['balanced_accuracy']):.2f}% | "
            f"Joint Q/W + FeatureKD/CDS/contrast; fixed-Q same-head epochs={recon_epochs}; "
            f"historical negatives={historical_negative_mode} "
            f"draws_per_class={effective_negative_samples}; no stored old examples",
            file=sys.stdout,
        )

        @torch.no_grad()
        def collect_current_state():

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

        # Prototype initialization on CURRENT TRAIN features only. No old
        # exemplars, extra head, or trainable module are introduced.
        # The cosine head compares unit directions, so an incoming class's
        # observed direction is a better starting point than a random row.
        with torch.no_grad():
            incoming_means = current_reference_structure.learned_identity.flatten(1)
            mean_norms = incoming_means.norm(p=2, dim=1)
            if not bool((mean_norms > torch.finfo(incoming_means.dtype).eps).all()):
                raise RuntimeError("incoming class has a degenerate mean direction")
            new_prototypes = torch.nn.functional.normalize(
                incoming_means, p=2, dim=1
            ).to(device=classifier.weight.device, dtype=classifier.weight.dtype)
            if new_prototypes.shape != (
                len(new_class_ids),
                classifier.feature_dim,
            ) or not bool(torch.isfinite(new_prototypes).all()):
                raise RuntimeError("invalid incoming class prototypes")
            classifier.weight.index_copy_(0, new_class_tensor, new_prototypes)

        # Diagnostic classifier admission: learn W_new against all historical
        # rows before the more restrictive historical robust loss is activated.
        # This addresses zero new all-seen accuracy without modifying Q or W_old.
        admission_epochs = int(self.cfg("head_admission_epochs", 0))
        if admission_epochs < 0:
            raise ValueError("head_admission_epochs must be nonnegative")
        admission_lr = _finite_positive(
            "head_admission_lr", self.cfg("head_admission_lr", 1e-3)
        )
        admission_state = self._run_new_class_head_admission(
            classifier=classifier,
            train_loader=train_loader,
            train_eval_loader=train_eval_loader,
            old_class_ids=old_class_ids,
            new_class_ids=new_class_ids,
            seen_class_ids=seen_class_ids,
            current_class_weights=current_class_weights,
            epochs=admission_epochs,
            learning_rate=admission_lr,
            gradient_clip=gradient_clip,
        )
        admission_new_weight = (
            classifier.class_weight(new_class_tensor).detach().clone()
        )
        # One detached orthogonal frame per epoch, fitted only on paired current
        # TRAIN inputs. Rejection retains identity; native spectra never rotate.
        historical_transport = bool(self.cfg("historical_transport", True))

        # Stage 1: COORDINATED Q + one-head W training. The current-data CE
        # updates both Q and active seen W, while the EXISTING historical CDS
        # directional historical CDS objective simultaneously protects old rows.
        # Whole-CDS inter/intra/native-spectrum preservation and contrast train Q.
        # No extra head or old exemplar. Optional temporary Gaussian Q draws
        # affect contrastive separation only; directional W supervision stays.

        representation_parameters = list(backbone_parameters)
        optimizer_groups = [
            {"params": backbone_parameters, "weight_decay": weight_decay}
        ]
        if representation_train_new_head:
            # Gradient masking alone cannot protect inactive rows from AdamW
            # decay. Start a fresh optimizer and disable decay for the head.
            optimizer_groups.append(
                {"params": [classifier.weight], "weight_decay": 0.0}
            )
            representation_parameters.append(classifier.weight)
        optimizer = torch.optim.AdamW(optimizer_groups, lr=learning_rate)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epoch_count, eta_min=0.0
        )

        # Explicit final-epoch contract. All scheduled optimizer steps are kept.
        # Validation remains read-only reporting and never restores prior Q/W.
        selected_representation_epoch = epoch_count  # compatibility metadata only
        selected_representation_score = None

        history: list[dict[str, Any]] = []

        gradient_contract_checked = False
        parameter_gradient_audit = {}
        parameter_gradient_audits = []

        progress = tqdm(
            range(epoch_count),
            desc=f"Phase {phase_id} representation training",
            unit="epoch",
            dynamic_ncols=True,
            file=sys.stdout,
        )

        for epoch in progress:
            self.model.eval()
            epoch_features_full, epoch_current_structure = collect_current_state()
            epoch_historical_structure, epoch_transform, _epoch_alignment = (
                correct_historical_structure(
                    historical_structure=historical_structure,
                    current_reference_structure=current_reference_structure,
                    current_structure=epoch_current_structure,
                    reference_features=reference_features_full,
                    current_features=epoch_features_full,
                    use_transport=historical_transport,
                )
            )
            epoch_support = HistoricalCDSSupportBank(
                self.model.structure_bank,
                margin_std=self.historical_support.margin_std,
                risk_mode=self.historical_support.risk_mode,
                coordinate_transform=epoch_transform,
            )
            epoch_seen_structure = merge_class_structure_batches(
                (epoch_historical_structure, epoch_current_structure)
            )
            epoch_seen_cds = self.model.structure_bank.build_cds(epoch_seen_structure)
            epoch_pair_risk = whole_cds_pair_risk(epoch_seen_cds)[0].detach()
            epoch_radii, epoch_intrusion = current_class_feasible_radii(
                current_features=epoch_features_full,
                current_labels=reference_labels,
                cds=epoch_seen_cds,
                historical_class_count=len(old_class_ids),
            )

            self.model.train()

            sums = {
                name: 0.0
                for name in (
                    "classification",
                    "real_classification",
                    "direct_head_classification",
                    "historical_cds_head_loss",
                    "joint_historical_risk",
                    "current_real_head_loss",
                    "old_all_seen_accuracy",
                    "current_new_only_accuracy",
                    "feature_kd",
                    "teacher_hist_relevance",
                    "feature_rms",
                    "contrastive",
                    "diagnostic_sep_loss",
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
                    "core_radius",
                    "core_violation",
                    "historical_intrusion",
                    "spectral_interval_overlap",
                    "variation_confidence",
                    "representation_gradient_conflict",
                    "representation_gradient_cosine",
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

                    # The teacher remains frozen. Its normalized features and
                    # historical moments use the same detached epoch frame.
                    reference_features_in_frame = rotate_historical_flat_rows(
                        normalize_ordered_decision_features(reference_features).flatten(1),
                        epoch_transform,
                    ).reshape_as(reference_features)

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

                # The deployed all-seen boundary receives real rejection evidence
                # on old rows and positive evidence on incoming rows.
                current_output = classifier(
                    current_features,
                    class_ids=seen_class_tensor,
                    trainable_class_ids=(),
                )
                representation_output = classifier(
                    current_features,
                    class_ids=(
                        new_class_tensor
                        if representation_ce_scope == "new_only"
                        else seen_class_tensor
                    ),
                    trainable_class_ids=(
                        seen_class_tensor if representation_train_new_head else ()
                    ),
                )

                real_classification = classification_objective(
                    logits=representation_output.logits,
                    class_ids=representation_output.class_ids,
                    labels_global=labels,
                    sample_weights=sample_weights,
                    unbiased_sample_weighting=True,
                )

                # Direct historical CDS boundary evidence and real current labels
                # update the deployed all-seen head. Q is detached on this branch;
                # no support examples are synthesized or retained.
                # Legacy feasible-core calculation is diagnostic ONLY; no
                # gradients or model updates use this surrogate.
                with torch.no_grad():
                    feasible_state = feasible_directional_head_objective(
                        classifier=classifier,
                        current_features=current_features.detach(),
                        current_labels=labels,
                        seen_cds=epoch_seen_cds,
                        historical_class_count=len(old_class_ids),
                        feasible_radii=epoch_radii,
                        intrusion_fraction=epoch_intrusion,
                        current_sample_weights=sample_weights,
                    )
                support_classification = feasible_state.state
                # Real CE acts on the deployed all-seen decision boundary. The
                # same historical CDS risk is optimized in this SAME step when
                # the seen head is jointly trainable, instead of arriving only
                # after Q has stopped moving.
                if representation_train_new_head:
                    joint_old_risk = epoch_support.historical_risk(
                        classifier,
                        old_class_ids,
                        epoch_seen_cds,
                        class_pair_risk=epoch_pair_risk,
                    )
                    classification_loss = (
                        len(new_class_ids) * real_classification.loss
                        + len(old_class_ids) * joint_old_risk.loss
                    ) / len(seen_class_ids)
                else:
                    joint_old_risk = None
                    classification_loss = real_classification.loss

                # Both sides use the same fixed epoch frame. Native spectral
                # support participates in intra/inter preservation and also
                # weights teacher relevance; no old inputs are accessed.
                cds = historical_cds_preservation_loss(
                    reference_features=reference_features_in_frame,
                    current_features=current_features,
                    physical_spectral_evidence=physical_evidence,
                    reference_historical_structure=epoch_historical_structure,
                    current_historical_structure=epoch_historical_structure,
                    historical_pair_risk=historical_pair_risk,
                )
                feature_kd = ordered_feature_distillation_loss(
                    reference_features=reference_features_in_frame,
                    current_features=current_features,
                    sample_weights=sample_weights,
                    historical_relevance=cds.teacher_historical_relevance,
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

                separation = cross_phase_cds_separation_loss(
                    current_cosine_scores=current_output.cosine_scores,
                    class_ids=current_output.class_ids,
                    labels_global=labels,
                    physical_spectral_evidence=physical_evidence,
                    seen_cds=epoch_seen_cds,
                    historical_class_count=len(old_class_ids),
                    sample_weights=sample_weights,
                )
                # The reader validates the same fixed epoch frame used by CDS
                # preservation. Draws are transient/detached and never acquire
                # invented native spectra. Mean mode isolates covariance's role.
                negative_features = negative_labels = None
                if historical_negative_mode != "classifier":
                    negative_features, negative_labels = epoch_support.sample_historical_features(
                        old_class_ids, epoch_seen_cds, effective_negative_samples,
                        generator=negative_generator,
                        use_covariance=historical_negative_mode == "distribution",
                    )
                contrastive = whole_cds_contrastive_loss(
                    current_features=current_features,
                    labels_global=labels,
                    physical_spectral_evidence=physical_evidence,
                    seen_cds=epoch_seen_cds,
                    historical_class_count=len(old_class_ids),
                    logit_scale=classifier.logit_scale,
                    sample_weights=sample_weights,
                    historical_support=epoch_support,
                    historical_classifier=classifier,
                    class_pair_risk=epoch_pair_risk,
                    historical_negative_mode=historical_negative_mode,
                    historical_negative_features=negative_features,
                    historical_negative_labels=negative_labels,
                )

                if not gradient_contract_checked:

                    tolerance = 64.0 * torch.finfo(classifier.weight.dtype).eps

                    real_ce_grad = torch.autograd.grad(
                        real_classification.loss,
                        classifier.weight,
                        retain_graph=True,
                        allow_unused=True,
                    )[0]
                    real_ce_old = self._row_grad_max(real_ce_grad, old_class_tensor)
                    real_ce_new = self._row_grad_max(real_ce_grad, new_class_tensor)
                    real_ce_future = self._row_grad_max(
                        real_ce_grad, future_class_tensor
                    )
                    if (
                        real_ce_future > tolerance
                        or real_ce_grad is None
                        or not bool(torch.isfinite(real_ce_grad).all())
                    ):
                        raise RuntimeError(
                            "real all-seen CE must have finite seen-row gradients and zero future-row gradients"
                        )
                    if representation_train_new_head:
                        risk_grad = torch.autograd.grad(
                            joint_old_risk.loss,
                            classifier.weight,
                            retain_graph=True,
                            allow_unused=True,
                        )[0]
                        if (
                            risk_grad is None
                            or not bool(torch.isfinite(risk_grad).all())
                            or self._row_grad_max(risk_grad, future_class_tensor)
                            > tolerance
                        ):
                            raise RuntimeError(
                                "joint historical CDS risk must train old/new W only"
                            )

                    for name, objective in (
                        ("Feature KD", feature_kd.loss),
                        ("Historical Whole-CDS contrastive", contrastive.loss),
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
                        ("real-current all-seen CE", real_classification.loss),
                        ("Feature KD", feature_kd.loss),
                        ("Historical Whole-CDS contrastive", contrastive.loss),
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
                        "[Whole-CDS representation gradient ownership] "
                        f"RealCE Wold={real_ce_old:.3e} Wnew={real_ce_new:.3e} "
                        f"| CE scope={representation_ce_scope}; "
                        f"joint W trainable={representation_train_new_head}; "
                        "real CE -> W_old/W_new; historical CDS -> W_old/W_new; W_future/tau fixed",
                        file=sys.stdout,
                    )

                    gradient_contract_checked = True

                loss = (
                    classification_loss
                    + contrastive_weight * contrastive.loss
                    + feature_kd_weight * feature_kd.loss
                    + kd_logit_weight * kd_loss
                    + cds_weight * cds.loss
                )

                if not bool(torch.isfinite(loss)):

                    raise RuntimeError(
                        "incremental representation objective became NaN or Inf"
                    )

                if sample_count == 0 and epoch in (0, epoch_count - 1):
                    parameter_gradient_audit = self.parameter_gradient_audit(
                        {
                            "classification_joint": classification_loss,
                            "real_current_ce": real_classification.loss,
                            "contrastive": contrastive_weight * contrastive.loss,
                            "feature_kd": feature_kd_weight * feature_kd.loss,
                            "cds_preservation": cds_weight * cds.loss,
                            "preservation_total": (
                                feature_kd_weight * feature_kd.loss
                                + cds_weight * cds.loss
                                + kd_logit_weight * kd_loss
                            ),
                        },
                        backbone_parameters,
                    )
                    parameter_gradient_audit["epoch"] = epoch + 1
                    parameter_gradient_audits.append(parameter_gradient_audit)

                # Observe gradient interference in the actual learned Q output.
                # This is a diagnostic, NOT a projected-gradient optimizer. No
                # extra distillation or trainable components are introduced.
                current_gradient = torch.autograd.grad(
                    classification_loss, current_features, retain_graph=True
                )[0].detach()
                preservation_gradient = torch.autograd.grad(
                    feature_kd_weight * feature_kd.loss
                    + cds_weight * cds.loss
                    + kd_logit_weight * kd_loss,
                    current_features,
                    retain_graph=True,
                )[0].detach()
                gradient_dot = (
                    (current_gradient * preservation_gradient).flatten(1).sum(1)
                )
                current_norm = current_gradient.flatten(1).norm(dim=1)
                preservation_norm = preservation_gradient.flatten(1).norm(dim=1)
                gradient_scale = current_norm * preservation_norm
                valid_gradient = gradient_scale > 1e-12
                gradient_cosine = torch.where(
                    valid_gradient,
                    gradient_dot / gradient_scale.clamp_min(1e-12),
                    torch.zeros_like(gradient_dot),
                ).clamp(-1.0, 1.0)
                if not bool(torch.isfinite(gradient_cosine).all()):
                    raise RuntimeError(
                        "nonfinite Q classification/preservation gradient similarity"
                    )
                sums["representation_gradient_conflict"] += float(
                    ((gradient_dot < 0) & valid_gradient).sum().item()
                )
                sums["representation_gradient_cosine"] += float(
                    gradient_cosine.sum().item()
                )

                # Interference remains an observation, not a projected update.
                loss.backward()

                if classifier.logit_scale_parameter.grad is not None:

                    raise RuntimeError(
                        "incremental training must not update cosine logit scale"
                    )

                if (
                    not representation_train_new_head
                    and classifier.weight.grad is not None
                ):
                    raise RuntimeError("representation step must not train W")

                if (
                    not representation_train_new_head
                    and self._row_grad_max(classifier.weight.grad, old_class_tensor) > 0
                ):
                    raise RuntimeError("frozen-head representation step updated W_old")

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
                        representation_parameters,
                        gradient_clip,
                        error_if_nonfinite=True,
                    )

                optimizer.step()

                self._assert_future_rows_unchanged(
                    classifier, future_class_tensor, future_weight_before
                )

                n = int(labels.numel())
                sums["core_radius"] += float(feasible_state.mean_core_radius) * n
                sums["core_violation"] += (
                    float(feasible_state.effective_core_violation) * n
                )
                sums["historical_intrusion"] += (
                    float(feasible_state.historical_intrusion_fraction) * n
                )
                sums["spectral_interval_overlap"] += (
                    float(feasible_state.physical_interval_overlap) * n
                )
                sums["variation_confidence"] += (
                    float(feasible_state.ordered_variation_confidence) * n
                )

                sums["classification"] += float(classification_loss.detach().item()) * n
                if joint_old_risk is not None:
                    sums["joint_historical_risk"] += (
                        float(joint_old_risk.loss.detach().item()) * n
                    )
                sums["real_classification"] += (
                    float(real_classification.loss.detach().item()) * n
                )
                sums["direct_head_classification"] += (
                    float(support_classification.balanced_loss.detach().item()) * n
                )
                sums["historical_cds_head_loss"] += (
                    float(
                        support_classification.historical_positive_loss.detach().item()
                    )
                    * n
                )
                sums["current_real_head_loss"] += (
                    float(support_classification.current_positive_loss.detach().item())
                    * n
                )
                sums["old_all_seen_accuracy"] += (
                    float(support_classification.real_all_seen_accuracy.detach().item())
                    * n
                )
                sums["current_new_only_accuracy"] += (
                    float(support_classification.new_only_accuracy.detach().item()) * n
                )

                sums["feature_kd"] += float(feature_kd.loss.detach().item()) * n
                sums["teacher_hist_relevance"] += (
                    float(cds.teacher_historical_relevance.detach().mean().item()) * n
                )

                sums["feature_rms"] += float(feature_kd.rms_drift.detach().item()) * n

                sums["contrastive"] += float(contrastive.loss.detach().item()) * n
                sums["diagnostic_sep_loss"] += float(separation.loss.detach()) * n

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
                    float(separation.decision_margin_mean.detach().item()) * n
                )

                sums["sep_violation"] += (
                    float(separation.violation_fraction.detach().item()) * n
                )

                sums["sep_positive"] += (
                    float(separation.positive_decision_score_mean.detach().item()) * n
                )

                # New/new competition is already handled by real all-seen CE;

                # this CDS loss contains only current-positive vs historical rivals.

                sums["sep_current_negative"] += 0.0

                sums["sep_historical_negative"] += (
                    float(
                        separation.historical_rival_decision_score_mean.detach().item()
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
                "joint_historical_risk_loss": sums["joint_historical_risk"]
                / sample_count,
                "real_current_classification_loss": sums["real_classification"]
                / sample_count,
                "direct_cds_classifier_loss": sums["direct_head_classification"]
                / sample_count,
                "direct_cds_historical_loss": sums["historical_cds_head_loss"]
                / sample_count,
                "direct_cds_current_loss": sums["current_real_head_loss"]
                / sample_count,
                "direct_old_all_seen_accuracy": sums["old_all_seen_accuracy"]
                / sample_count,
                "direct_current_new_only_accuracy": sums["current_new_only_accuracy"]
                / sample_count,
                "historical_core_radius": sums["core_radius"] / sample_count,
                "historical_core_violation": sums["core_violation"] / sample_count,
                "current_in_old_support_fraction": sums["historical_intrusion"]
                / sample_count,
                "spectral_band_interval_overlap": sums["spectral_interval_overlap"]
                / sample_count,
                "ordered_variation_confidence": sums["variation_confidence"]
                / sample_count,
                "representation_gradient_conflict_fraction": sums[
                    "representation_gradient_conflict"
                ]
                / sample_count,
                "representation_gradient_cosine": sums["representation_gradient_cosine"]
                / sample_count,
                "feature_kd_loss": sums["feature_kd"] / sample_count,
                "teacher_historical_relevance_mean": sums["teacher_hist_relevance"]
                / sample_count,
                "coordinate_stability_loss": sums["feature_kd"] / sample_count,
                "epoch_representation_drift_rms": sums["feature_rms"] / sample_count,
                "contrastive_loss": sums["contrastive"] / sample_count,
                "historical_negative_mode": historical_negative_mode,
                "historical_negative_samples_per_class": effective_negative_samples,
                "cds_separation_loss": sums["diagnostic_sep_loss"]
                / sample_count,  # read-only diagnostic
                "cds_separation_margin": sums["sep_margin"] / sample_count,
                "cds_separation_violation_fraction": sums["sep_violation"]
                / sample_count,
                "contrastive_positive_similarity": float(
                    contrastive.positive_similarity_mean.detach()
                ),
                "contrastive_current_negative_similarity": float(
                    contrastive.current_negative_similarity_mean.detach()
                ),
                "contrastive_historical_negative_similarity": float(
                    contrastive.historical_negative_similarity_mean.detach()
                ),
                "contrastive_valid_anchor_fraction": float(
                    contrastive.valid_anchor_fraction.detach()
                ),
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
                "new_only_ba": float(new_only["balanced_accuracy"]),
                "new_all_seen_ba": float(new_all["balanced_accuracy"]),
                "validation_new_all_seen_ba": (
                    None
                    if validation_all is None
                    else float(validation_all["balanced_accuracy"])
                ),
            }

            # Pairwise parameter gradients are sampled before the optimizer step,
            # not projected and not computed from reporting/test examples.
            if parameter_gradient_audits and parameter_gradient_audits[-1]["epoch"] == epoch + 1:
                record["backbone_objective_gradient_audit"] = parameter_gradient_audits[-1]
            # paired-feature evidence quality, observed ONLY on incoming classes.
            record["alignment_holdout_candidate_error"] = float(
                _epoch_alignment.holdout_candidate_error.item()
            )
            record["alignment_holdout_identity_error"] = float(
                _epoch_alignment.holdout_identity_error.item()
            )
            record["alignment_transport_accepted"] = bool(
                _epoch_alignment.transport_accepted.item()
            )
            # Legacy report fields are preserved without checkpoint ranking.
            record["validation_joint_selection_objective"] = None
            record["validation_historical_cds_risk"] = None
            record["validation_real_new_ce"] = None
            history.append(record)

            tqdm.write(
                f"[Representation {epoch + 1:03d}/{epoch_count:03d}] "
                f"JointObj={record['classification_loss']:.4f} "
                f"CEreal={record['real_current_classification_loss']:.4f} "
                f"LegacyCoreDiag={record['direct_cds_classifier_loss']:.4f} "
                f"Con={record['contrastive_loss']:.4f} "
                f"CoreR={record['historical_core_radius']:.3f} "
                f"CoreViol={100*record['historical_core_violation']:.1f}% "
                f"GradConflict={100*record['representation_gradient_conflict_fraction']:.1f}% "
                f"QGradientCos={record['representation_gradient_cosine']:.3f} "
                f"FKD={record['feature_kd_loss']:.4f} "
                # f"KD={record['kd_logit_loss']:.4f} "
                f"CDS={record['cds_preservation_loss']:.4f}"
                f"[I={record['cds_intra_loss']:.4f},R={record['cds_inter_loss']:.4f}] "
                f"DecisionMargin={record['cds_separation_margin']:.4f} "
                f"DecisionViol={100*record['cds_separation_violation_fraction']:.1f}% | "
                f"NewOnly/All={100*record['new_only_ba']:.2f}%/"
                f"{100*record['new_all_seen_ba']:.2f}%",
                file=sys.stdout,
            )

        progress.close()

        if not history:

            raise RuntimeError("incremental representation training produced no epochs")

        # The deployed representation and cosine rows are exactly those
        # produced by the final optimizer step, not a validation-selected copy.
        self._assert_future_rows_unchanged(
            classifier, future_class_tensor, future_weight_before
        )
        print(
            f"[Joint Q+W final epoch] retained epoch {epoch_count}/{epoch_count}; "
            "no checkpoint ranking, no Q/W rollback",
            file=sys.stdout,
        )

        # Phase-end statistical synchronization. This is NOT another training stage.

        self.model.eval()

        for parameter in self.model.backbone.parameters():

            parameter.requires_grad_(False)

            parameter.grad = None

        final_features_full, final_current_structure = collect_current_state()
        # Before/after migration is measured on the same current TRAIN samples.
        # This is an observable signal for historical Q-distribution alignment,
        # not proof of how unavailable historical samples migrated.
        with torch.no_grad():
            before_q = torch.nn.functional.normalize(
                reference_features_full.reshape(reference_features_full.size(0), -1),
                dim=1,
            )
            after_q = torch.nn.functional.normalize(
                final_features_full.reshape(final_features_full.size(0), -1), dim=1
            )
            paired_new_feature_cosine = float(
                (before_q * after_q).sum(dim=1).mean().item()
            )
            paired_new_feature_chord = float(
                (before_q - after_q).square().sum(dim=1).mean().sqrt().item()
            )

        alignment_spectral_evidence = self.model.physical_spectral_evidence(
            reference_spectra
        )
        if recon_epochs:
            # A final refit is allowed only when the same head will subsequently
            # train against this exact fixed bank coordinate.
            final_historical_structure, final_transform, final_alignment = correct_historical_structure(
                historical_structure=historical_structure,
                current_reference_structure=current_reference_structure,
                current_structure=final_current_structure,
                reference_features=reference_features_full,
                current_features=final_features_full,
                use_transport=historical_transport,
            )
            historical_frame_source = "final_alignment_then_head_training"
        else:
            # Commit the last frame that actually supervised the deployed head.
            # Measure its final residual without fitting an unused replacement.
            final_historical_structure = epoch_historical_structure
            final_transform = epoch_transform
            final_metrics = evaluate_fixed_phase_coordinate_transform(
                reference_features_full, final_features_full, final_transform
            )
            final_alignment = replace(_epoch_alignment, **final_metrics)
            historical_frame_source = "last_representation_epoch"
        committed_structure = merge_class_structure_batches(
            (final_historical_structure, final_current_structure)
        )

        committed_ids = tuple(
            int(v) for v in committed_structure.class_ids.detach().cpu().tolist()
        )
        if committed_ids != seen_class_ids:
            raise RuntimeError(
                "final Whole-CDS structure produced incorrect seen-class order"
            )

        # The live bank stays immutable until the candidate and any requested
        # head training succeed. Old means/bases rotate together; native P stays fixed.
        candidate_bank = deepcopy(self.model.structure_bank)
        # The reference->student map is a validated orthogonal rotation. Means
        # and low-rank covariance bases must commit in exactly the same frame.
        candidate_bank.apply_historical_orthogonal_transform(
            old_class_ids, final_transform
        )
        candidate_bank.replace_all_with_moments(
            committed_structure,
            final_features_full,
            alignment_spectral_evidence,
            reference_labels,
        )
        candidate_bank.validate_state()
        committed_seen_cds = candidate_bank.build_cds(candidate_bank.class_structures())
        reconstruction_support = HistoricalCDSSupportBank(
            candidate_bank,
            margin_std=self.historical_support.margin_std,
            risk_mode=self.historical_support.risk_mode,
        )
        reconstruction_pair_risk = whole_cds_pair_risk(committed_seen_cds)[0].detach()

        if not torch.equal(
            final_historical_structure.physical_identity,
            historical_structure.physical_identity,
        ) or not torch.equal(
            final_historical_structure.physical_variation,
            historical_structure.physical_variation,
        ):
            raise RuntimeError("final historical correction changed physical I_P/V_P")

        # Preserve only model parameters for later reporting. No reporting/test
        # example is opened until reconstruction and phase finalization finish.
        post_rep_classifier = deepcopy(classifier).requires_grad_(False).eval()
        post_rep_new_only = self.evaluate_loader(
            train_eval_loader, class_ids=new_class_ids, target_class_ids=new_class_ids
        )
        post_rep_new_all = self.evaluate_loader(
            train_eval_loader, class_ids=seen_class_ids, target_class_ids=new_class_ids
        )

        # Current-task learned-feature movement report: both structures were fitted to EXACTLY
        # the same current TRAIN class IDs and spectra, before/after backbone
        # optimization. Only per-class aggregate diagnostics are persisted.
        classwise_new_migration = {}
        classwise_old_transport = {}
        with torch.no_grad():
            for row, cls_id in enumerate(new_class_ids):
                before = current_reference_structure.learned_identity[row].reshape(-1)
                after = final_current_structure.learned_identity[row].reshape(-1)
                before_rel = current_reference_structure.learned_variation[row]
                after_rel = final_current_structure.learned_variation[row]
                before_disp = current_reference_structure.learned_absolute_deviation[
                    row
                ]
                after_disp = final_current_structure.learned_absolute_deviation[row]
                classwise_new_migration[str(cls_id)] = {
                    "identity_delta_l2": float((after - before).norm().item()),
                    "identity_cosine": float(
                        torch.nn.functional.cosine_similarity(
                            before[None], after[None]
                        ).item()
                    ),
                    "relational_interval_change_l2": float(
                        (after_rel - before_rel).norm().item()
                    ),
                    "learned_absolute_deviation_delta_l2": float(
                        (after_disp - before_disp).norm().item()
                    ),
                    "learned_absolute_deviation_mean_before": float(
                        before_disp.mean().item()
                    ),
                    "learned_absolute_deviation_mean_after": float(
                        after_disp.mean().item()
                    ),
                }
            for row, cls_id in enumerate(old_class_ids):
                # Measured only on the compact stored class means; neither
                # accepted nor rejected transport certifies unavailable old drift.
                classwise_old_transport[str(cls_id)] = {
                    "stored_identity_delta_l2": float(
                        (
                            final_historical_structure.learned_identity[row]
                            - historical_structure.learned_identity[row]
                        )
                        .norm()
                        .item()
                    ),
                    "stored_dispersion_delta_l2": float(
                        (
                            final_historical_structure.learned_absolute_deviation[row]
                            - historical_structure.learned_absolute_deviation[row]
                        )
                        .norm()
                        .item()
                    ),
                    "unobserved_old_feature_drift_certified": False,
                }

        old_weight_after_training = (
            classifier.class_weight(old_class_tensor).detach().clone()
        )
        if not representation_train_new_head and not torch.equal(
            old_weight_after_training, old_weight_before
        ):
            raise RuntimeError(
                "frozen-head ablation must preserve all historical W rows"
            )
        new_weight_after_training = (
            classifier.class_weight(new_class_tensor).detach().clone()
        )
        self._assert_future_rows_unchanged(
            classifier, future_class_tensor, future_weight_before
        )
        if float(classifier.logit_scale.detach().item()) != phase_start_scale:
            raise RuntimeError("incremental representation changed cosine logit scale")

        # Read-only final diagnostics before actual frozen-Q reconstruction.
        final_classifier_objective = whole_cds_class_balanced_classifier_loss(
            classifier=classifier,
            cds=committed_seen_cds,
            historical_class_count=len(old_class_ids),
        )
        pre_classifier_objective = final_classifier_objective
        pre_finalize = final_classifier_objective

        final_radii, final_intrusion = current_class_feasible_radii(
            current_features=final_features_full,
            current_labels=reference_labels,
            cds=committed_seen_cds,
            historical_class_count=len(old_class_ids),
        )
        final_feasible_state = feasible_directional_head_objective(
            classifier=classifier,
            current_features=final_features_full.detach(),
            current_labels=reference_labels,
            seen_cds=committed_seen_cds,
            historical_class_count=len(old_class_ids),
            feasible_radii=final_radii,
            intrusion_fraction=final_intrusion,
        )
        final_classifier_state = final_feasible_state.state
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
            diagnostic_new_only_output = classifier(
                final_features_full,
                class_ids=new_class_tensor,
                trainable_class_ids=(),
            )
            diagnostic_new_only_accuracy = (
                diagnostic_new_only_output.prediction.eq(reference_labels)
                .to(diagnostic_new_only_output.logits.dtype)
                .mean()
            )

        print(
            "[Before Whole-CDS support reconstruction: legacy diagnostic ONLY] "
            f"JointLoss={float(final_classifier_state.balanced_loss.detach().item()):.4f} "
            f"OldCDSLoss={float(final_classifier_state.historical_positive_loss.detach().item()):.4f} "
            f"NewRealLoss={float(final_classifier_state.current_positive_loss.detach().item()):.4f} "
            f"RealNewCE={float(diagnostic_new_ce.detach().item()):.4f} "
            f"NewAllAcc={100*float(diagnostic_new_all_accuracy.detach().item()):.2f}% "
            f"NewOnlyAcc={100*float(diagnostic_new_only_accuracy.detach().item()):.2f}% "
            f"OldNewViol={100*float(final_classifier_objective.old_new_violation_fraction.detach().item()):.1f}%",
            file=sys.stdout,
        )

        # Optional existing same-head budget: fixed final Q and fixed final CDS.
        # Keep all configured optimizer steps; validation scores are audit only.
        recon_lr = _finite_positive(
            "support_head_lr", self.cfg("support_head_lr", 1e-3)
        )

        self.model.eval()
        # Zero-epoch diagnostics must not freeze the deployed student backbone.
        if recon_epochs:
            for parameter in self.model.backbone.parameters():
                parameter.requires_grad_(False)
                parameter.grad = None
        classifier.weight.grad = None
        fixed_q = final_features_full.detach()
        fixed_targets = reference_labels.detach()
        # The features are transient and are discarded after reconstruction.
        # Each configured epoch now traverses minibatches (not one full-batch
        # gradient step). This matches the meaning of training epochs and
        # permits classifier adaptation against old + new classes.
        targets_seen = _targets_local(fixed_targets, seen_class_tensor)
        recon_opt = torch.optim.AdamW(
            [classifier.weight], lr=recon_lr, weight_decay=0.0
        )
        history_reconstruction = []
        selection_source = "current_train_only"
        selection_q, selection_y = fixed_q, fixed_targets
        if val_loader is not None:
            with torch.no_grad():
                valid_rows = self.collect_ordered_features(val_loader)
                valid_q = valid_rows["ordered_features"].detach()
                valid_y = valid_rows["labels"].detach()
                if all(bool((valid_y == cid).any()) for cid in new_class_ids):
                    selection_q, selection_y = valid_q, valid_y
                    selection_source = "current_validation"
                else:
                    selection_source = "current_train_only_missing_validation_classes"

        @torch.no_grad()
        def reconstruction_objective():
            historical = reconstruction_support.historical_risk(
                classifier,
                old_class_ids,
                committed_seen_cds,
                class_pair_risk=reconstruction_pair_risk,
            )
            outputs = classifier(
                selection_q,
                class_ids=seen_class_tensor,
                trainable_class_ids=(),
            )
            targets = _targets_local(selection_y, seen_class_tensor)
            sample_ce = torch.nn.functional.cross_entropy(
                outputs.logits,
                targets,
                reduction="none",
            )
            new_ce = torch.stack(
                [sample_ce[selection_y.eq(cid)].mean() for cid in new_class_ids]
            ).mean()
            criterion = (
                len(old_class_ids) * historical.loss + len(new_class_ids) * new_ce
            ) / len(seen_class_ids)
            return (
                float(criterion.item()),
                float(historical.loss.item()),
                float(new_ce.item()),
            )

        initial_score, initial_old_risk, initial_new_ce = reconstruction_objective()
        selected_score = initial_score
        selected_epoch = 0
        history_reconstruction.append(
            {
                "epoch": 0,
                "optimizer_steps": 0,
                "selection_objective": initial_score,
                "historical_support_loss": initial_old_risk,
                "selection_new_ce": initial_new_ce,
                "selection_source": selection_source,
            }
        )
        for rec_epoch in range(recon_epochs):
            permutation = torch.randperm(
                fixed_targets.numel(), device=fixed_targets.device
            )
            steps = 0
            for rows in permutation.split(train_batch_size):
                batch_q = fixed_q.index_select(0, rows)
                batch_labels = fixed_targets.index_select(0, rows)
                batch_targets = targets_seen.index_select(0, rows)
                local_new = _targets_local(batch_labels, new_class_tensor)
                weights = current_class_weights.index_select(0, local_new)

                recon_opt.zero_grad(set_to_none=True)
                risk = reconstruction_support.historical_risk(
                    classifier,
                    old_class_ids,
                    committed_seen_cds,
                    class_pair_risk=reconstruction_pair_risk,
                )
                out = classifier(
                    batch_q,
                    class_ids=seen_class_tensor,
                    trainable_class_ids=seen_class_tensor,
                )
                per_sample = torch.nn.functional.cross_entropy(
                    out.logits, batch_targets, reduction="none"
                )
                # A minibatch need not contain every incoming class.
                # Class weights estimated from the entire incoming TRAIN split
                # approximate one equal vote per incoming class.
                new_ce = (per_sample * weights).mean()
                # Each seen class has equal weight, not each TASK.
                joint = (
                    len(old_class_ids) * risk.loss + len(new_class_ids) * new_ce
                ) / len(seen_class_ids)
                if not bool(torch.isfinite(joint)):
                    raise RuntimeError("nonfinite joint-moment classifier objective")
                joint.backward()
                if classifier.weight.grad is None or not bool(
                    torch.isfinite(classifier.weight.grad).all()
                ):
                    raise RuntimeError(
                        "historical/current moment reconstruction gradient invalid"
                    )
                if (
                    future_class_ids
                    and self._row_grad_max(classifier.weight.grad, future_class_tensor)
                    > 0
                ):
                    raise RuntimeError(
                        "moment reconstruction gradient leaked into future rows"
                    )
                if gradient_clip > 0:
                    torch.nn.utils.clip_grad_norm_(
                        [classifier.weight], gradient_clip, error_if_nonfinite=True
                    )
                recon_opt.step()
                self._assert_future_rows_unchanged(
                    classifier, future_class_tensor, future_weight_before
                )
                steps += 1

            # Log POST-epoch scores on the same fixed incoming TRAIN features.
            with torch.no_grad():
                risk = reconstruction_support.historical_risk(
                    classifier,
                    old_class_ids,
                    committed_seen_cds,
                    class_pair_risk=reconstruction_pair_risk,
                )
                out = classifier(
                    fixed_q,
                    class_ids=seen_class_tensor,
                    trainable_class_ids=(),
                )
                ce = torch.nn.functional.cross_entropy(
                    out.logits, targets_seen, reduction="none"
                )
                new_ce = torch.stack(
                    [ce[fixed_targets.eq(cid)].mean() for cid in new_class_ids]
                ).mean()
                joint = (
                    len(old_class_ids) * risk.loss + len(new_class_ids) * new_ce
                ) / len(seen_class_ids)
                true_new_logits = out.logits.gather(1, targets_seen[:, None]).squeeze(1)
                strongest_old_logit = out.logits[:, : len(old_class_ids)].max(1).values
                row = {
                    "epoch": rec_epoch + 1,
                    "optimizer_steps": steps,
                    "historical_support_loss": float(risk.loss),
                    "historical_objective": risk.objective,
                    "historical_mean_ce": (
                        None if risk.mean_ce_loss is None else float(risk.mean_ce_loss)
                    ),
                    "historical_curvature": (
                        None
                        if risk.curvature_correction is None
                        else float(risk.curvature_correction)
                    ),
                    "current_real_ce": float(new_ce),
                    "balanced_loss": float(joint),
                    "new_vs_old_logit_margin": float(
                        (true_new_logits - strongest_old_logit).mean()
                    ),
                    "new_all_seen_train_acc": float(
                        out.prediction.eq(fixed_targets).float().mean()
                    ),
                    "hardest_historical_rival": risk.per_class_worst_rival.cpu().tolist(),
                    "historical_classwise_moment_margin_penalty": risk.per_class_loss.cpu().tolist(),
                }
            score, held_old_risk, held_new_ce = reconstruction_objective()
            row["selection_objective"] = score
            row["selection_new_ce"] = held_new_ce
            row["selection_old_risk"] = held_old_risk
            row["selection_source"] = selection_source
            history_reconstruction.append(row)
            selected_score = score
            selected_epoch = rec_epoch + 1
        classifier.weight.grad = None
        # Refresh all diagnostics after the actual deployed W has been fitted.
        final_feasible_state = feasible_directional_head_objective(
            classifier=classifier,
            current_features=final_features_full.detach(),
            current_labels=reference_labels,
            seen_cds=committed_seen_cds,
            historical_class_count=len(old_class_ids),
            feasible_radii=final_radii,
            intrusion_fraction=final_intrusion,
        )
        final_classifier_state = final_feasible_state.state
        final_classifier_objective = whole_cds_class_balanced_classifier_loss(
            classifier=classifier,
            cds=committed_seen_cds,
            historical_class_count=len(old_class_ids),
        )
        final_head_accepted = sum(row["optimizer_steps"] for row in history_reconstruction)
        print(
            f"[Joint Q+W final-state diagnostics] same_head_epochs={recon_epochs}; "
            f"initial_old_surrogate={initial_old_risk:.4f} initial_new_CE={initial_new_ce:.4f}; "
            "final optimizer state retained; validation is audit only",
            file=sys.stdout,
        )
        # Commit the complete, validated candidate after reconstruction.
        # Failed reconstruction leaves the historical CDS bank unchanged.
        self.model.structure_bank.load_state_dict(
            candidate_bank.state_dict(), strict=True
        )
        self.historical_support.require(seen_class_ids)
        self.model.validate_finalized_state()
        risk_audit = self.audit_current_classifier_risk(
            val_loader, class_ids=new_class_ids, cds=committed_seen_cds
        )

        # Read-only integrated diagnosis on current-class TRAIN / current VAL only.
        # No old images/labels are accessed and no deployed model weight is updated.
        integrated_audit: dict[str, Any] = {
            "status": "disabled",
            "used_for_selection": False,
            "historical_overlap_scope": "modeled_old_means_not_real_old_examples",
        }
        if bool(self.cfg("audit_enabled", False)):
            audit_val_features = (
                self.collect_ordered_features(val_loader) if val_loader is not None else None
            )
            audit_source = audit_val_features
            audit_split = "current_validation"
            if audit_source is None:
                audit_source = self.collect_ordered_features(train_eval_loader)
                audit_split = "current_train_no_validation"
            audit_indices = deterministic_class_audit_indices(
                audit_source["labels"],
                max_per_class=int(self.cfg("audit_max_per_class", 48)),
            )
            overlap = feature_overlap_audit(
                audit_source["ordered_features"].index_select(0, audit_indices),
                audit_source["labels"].index_select(0, audit_indices),
                historical_means=final_historical_structure.learned_identity,
            )
            overlap["split"] = audit_split
            # A new-only probe is a diagnostic, not a replacement deployed head.
            if audit_val_features is None:
                probe: dict[str, Any] = {
                    "status": "skipped_validation_disabled", "used_for_selection": False
                }
            else:
                audit_train = self.collect_ordered_features(train_eval_loader)
                probe = fixed_q_classifier_probe(
                    audit_train["ordered_features"],
                    audit_train["labels"],
                    audit_val_features["ordered_features"],
                    audit_val_features["labels"],
                    class_ids=new_class_ids,
                    deployed_weights=classifier.class_weight(new_class_tensor),
                    logit_scale=classifier.logit_scale,
                    steps=int(self.cfg("audit_probe_steps", 60)),
                )
            integrated_audit = {
                "status": "evaluated",
                "used_for_selection": False,
                "historical_overlap_scope": "modeled_old_means_not_real_old_examples",
                "feature_overlap": overlap,
                "fixed_q_new_only_probe": probe,
                "backbone_gradient_snapshots": parameter_gradient_audits,
                "transport_comparison": "run separate identical-seed experiments with --historical_transport true/false; no old test labels enter alignment",
                "active_historical_transport": historical_transport,
                "historical_new_pair_holdout_accepted": bool(final_alignment.transport_accepted.item()),
                "historical_risk_method": "directional_moment_margin_surrogate_not_old_empirical_ce",
            }
            print(
                f"[Read-only audit] {audit_split}: "
                f"positive-pair fraction={overlap['real_positive_anchor_fraction']:.3f}, "
                f"hard-negative gap={overlap['real_positive_vs_hard_negative_gap']}, "
                f"probe={probe['status']}",
                file=sys.stdout,
            )

        self.dataset.finalize_phase(phase_id)

        for parameter, trainable in backbone_trainability:
            parameter.requires_grad_(trainable)
        classifier.logit_scale_parameter.requires_grad_(scale_requires_grad)

        classifier.logit_scale_parameter.grad = None

        old_reporting_loader = self.dataset.get_reporting_dataloader(
            previous_phase, batch_size=eval_batch_size
        )
        live_backbone, live_classifier = self.model.backbone, self.model.classifier
        try:
            self.model.backbone = reference_backbone
            self.model.classifier = reference_classifier
            old_before = self.evaluate_loader(
                old_reporting_loader,
                class_ids=previous_seen_class_ids,
                target_class_ids=previous_seen_class_ids,
            )
            # Q did not change during head reconstruction; the saved W therefore
            # reproduces the post-representation decision on this reporting split.
            self.model.backbone = live_backbone
            self.model.classifier = post_rep_classifier
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
        finally:
            self.model.backbone, self.model.classifier = live_backbone, live_classifier
        del post_rep_classifier

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
            cumulative_loader, class_ids=new_class_ids, target_class_ids=new_class_ids
        )

        final_new_all = self.evaluate_loader(
            cumulative_loader, class_ids=seen_class_ids, target_class_ids=new_class_ids
        )

        final_record = history[-1]
        final_old_weight = classifier.class_weight(old_class_tensor).detach()
        final_new_weight = classifier.class_weight(new_class_tensor).detach()
        method_state = {
            "stores_old_images": False,
            "stores_old_feature_rows": False,
            "stores_synthetic_support": False,
            "stores_transient_distribution_support": False,
            "uses_transient_historical_distribution_support": historical_negative_mode == "distribution",
            "historical_negative_mode": historical_negative_mode,
            "historical_negative_samples_per_class": effective_negative_samples,
            "historical_negative_aggregation": "per_class_log_mean_exp" if historical_negative_mode != "classifier" else "single_direction",
            "historical_negative_sampling_scope": "contrastive_only; epoch_frame; fresh_per_minibatch",
            "historical_negative_generator_seed": int(self.cfg("seed", 0)) + phase_id,
            "historical_classifier_supervision": "unchanged_directional_moment_risk",
            "generates_native_spectra": False,
            "uses_persistent_whole_cds_support": True,
            "feasible_core_objective_active": False,
            "support_reconstruction_active": bool(recon_epochs > 0),
            "uses_real_current_positive_support": True,
            "uses_covariance_or_gaussian_memory": True,
            "uses_gaussian_density_or_sampling": historical_negative_mode == "distribution",
            "uses_mean_or_second_moment_classifier_memory": True,
            "uses_second_classifier": False,
            "uses_phase_bias": False,
            "uses_posthoc_classifier_reconstruction": bool(recon_epochs > 0),
            "final_synchronized_head_steps_accepted": final_head_accepted,
            "selected_support_head_epoch": selected_epoch,
            "reconstruction_selection_source": selection_source,
            "reconstruction_selected_objective": selected_score,
            "reconstruction_initial_objective": initial_score,
            "reconstruction_initial_new_ce": initial_new_ce,
            "reconstruction_new_ce_guard_active": False,
            "historical_support_reconstruction": history_reconstruction,
            # The active joint representation loop uses real-new CE plus an
            # analytic old-class directional risk, NOT old positive replay CE.
            "uses_balanced_old_new_positive_classifier_support": False,
            "historical_classifier_positive_supervision": "real_current_ce_only",
            "historical_classifier_old_supervision": "analytical_directional_cds_risk",
            "integrated_audit": integrated_audit,
            "uses_signed_whole_cds_distribution": True,
            "historical_support_audit": self.historical_support.diagnostics(),
            "uses_per_epoch_direct_historical_coordinate_refresh": historical_transport,
            "historical_frame_source": historical_frame_source,
            "historical_frame_fit_scope": "current_train_pairs_at_epoch_start" if not recon_epochs else "current_train_pairs_after_representation",
            "historical_frame_diagnostics_scope": "fixed_frame_final_current_train_pairs",
            "contrastive_historical_carrier": {
                "classifier": "detached_deployed_cosine_rows",
                "mean": "detached_normalized_historical_means",
                "distribution": "detached_normalized_compact_gaussian_draws",
            }[historical_negative_mode],
            "historical_negative_support_gate": False,
            "real_ce_first_batch_row_gradient_max": {
                "old": real_ce_old, "new": real_ce_new, "future": real_ce_future,
            },
            "classifier_audit_selects_weights": False,
            "historical_risk_mode": self.historical_support.risk_mode,
            "representation_ce_scope": representation_ce_scope,
            "representation_train_new_head": representation_train_new_head,
            "joint_seen_cosine_head_during_representation": representation_train_new_head,
            "historical_cds_head_risk_in_representation": representation_train_new_head,
            "risk_approximation_audit": risk_audit,
            "competition_splits": {
                "old": "reporting", "new": "reporting",
                "phase_start_and_post_rep_new": "train",
            },
            "parameter_gradient_audit": parameter_gradient_audit,
            "parameter_gradient_audits": parameter_gradient_audits,
            "representation_incoming_direction_change": self._max_direction_change(
                admission_new_weight, new_weight_after_training
            ),
            "head_admission": admission_state,
            "historical_transport_enabled": historical_transport,
            "historical_feasible_radius_mean": float(
                final_feasible_state.mean_core_radius
            ),
            "historical_current_intrusion_fraction": float(
                final_feasible_state.historical_intrusion_fraction
            ),
            "historical_core_violation": float(
                final_feasible_state.effective_core_violation
            ),
            "historical_spectral_interval_overlap": float(
                final_feasible_state.physical_interval_overlap
            ),
            "historical_vq_directional_confidence": float(
                final_feasible_state.ordered_variation_confidence
            ),
            "representation_gradient_conflict_fraction": float(
                final_record["representation_gradient_conflict_fraction"]
            ),
            "representation_gradient_cosine": float(
                final_record["representation_gradient_cosine"]
            ),
            "uses_feature_kd": feature_kd_weight > 0,
            "new_class_before_feature_mean_norm": reference_feature_norm,
            "new_class_before_after_mean_cosine": paired_new_feature_cosine,
            "new_class_before_after_chord_rms": paired_new_feature_chord,
            "new_class_before_after_structure": classwise_new_migration,
            "historical_cds_estimated_transport": classwise_old_transport,
            "final_teacher_historical_relevance_mean": float(
                final_record["teacher_historical_relevance_mean"]
            ),
            "uses_logit_kd": bool(kd_logit_weight > 0),
            "prediction_rule": "tau*cosine(Q,W)",
            "representation_epochs": epoch_count,
            "selected_representation_epoch": selected_representation_epoch,
            "representation_selection_source": "final_epoch_no_selection",
            "selected_representation_objective": selected_representation_score,
            "representation_lr": learning_rate,
            "contrastive_weight": contrastive_weight,
            "feature_kd_weight": feature_kd_weight,
            "kd_logit_weight": kd_logit_weight,
            "kd_temperature": kd_temperature,
            "cds_weight": cds_weight,
            "final_joint_classification_loss": float(
                final_record["classification_loss"]
            ),
            "final_joint_historical_risk_loss": float(
                final_record["joint_historical_risk_loss"]
            ),
            "final_real_current_classification_loss": float(
                final_record["real_current_classification_loss"]
            ),
            "final_feature_kd_loss": float(final_record["feature_kd_loss"]),
            "final_cds_separation_loss": float(final_record["cds_separation_loss"]),
            "final_cds_separation_margin": float(final_record["cds_separation_margin"]),
            "final_cds_separation_violation_fraction": float(
                final_record["cds_separation_violation_fraction"]
            ),
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
            "phase_transport_accepted_on_new_class_holdout": bool(
                final_alignment.transport_accepted.item()
            ),
            "phase_holdout_affine_error": float(
                final_alignment.holdout_candidate_error.item()
            ),
            "phase_holdout_identity_error": float(
                final_alignment.holdout_identity_error.item()
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
            "final_direct_cds_classifier_loss": float(
                final_classifier_state.balanced_loss.detach().item()
            ),
            "final_direct_cds_historical_loss": float(
                final_classifier_state.historical_positive_loss.detach().item()
            ),
            "final_direct_cds_current_loss": float(
                final_classifier_state.current_positive_loss.detach().item()
            ),
            "final_whole_cds_old_new_violation_fraction": float(
                final_classifier_objective.old_new_violation_fraction.detach().item()
            ),
            "classifier_learned_pair_risk_mean": float(
                final_classifier_state.learned_pair_risk_mean.detach().item()
            ),
            "classifier_physical_pair_risk_mean": float(
                final_classifier_state.physical_pair_risk_mean.detach().item()
            ),
            "classifier_effective_pair_risk_mean": float(
                final_classifier_state.effective_pair_risk_mean.detach().item()
            ),
            "historical_classifier_training_max_change": self._max_parameter_change(
                old_weight_before, final_old_weight
            ),
            "historical_classifier_training_direction_change": self._max_direction_change(
                old_weight_before, final_old_weight
            ),
            "incoming_classifier_training_max_change": self._max_parameter_change(
                new_weight_phase_start, final_new_weight
            ),
            "incoming_classifier_training_direction_change": self._max_direction_change(
                new_weight_phase_start, final_new_weight
            ),
            "phase_start_old_ba": float(old_before["balanced_accuracy"]),
            "phase_start_old_all_seen_ba": None,  # No true all-seen phase-start measurement.
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
            "historical_support_reconstruction": history_reconstruction,
            "historical_support_classes": list(self.historical_support.class_ids),
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
# from copy import deepcopy
# from dataclasses import replace
# import math
# import sys
# import warnings
# from numbers import Integral, Real
# from typing import Any, Dict
# import torch
# from tqdm import tqdm

# from losses.loss import (
#     classification_objective,
#     cross_phase_cds_separation_loss,
#     whole_cds_contrastive_loss,
#     historical_cds_preservation_loss,
#     logit_distillation_loss,
#     ordered_feature_distillation_loss,
# )

# from models.classifier import CosineClassificationHead
# from models.cds_decision_region import (
#     whole_cds_class_balanced_classifier_loss,
#     whole_cds_joint_classifier_state,
#     feasible_directional_head_objective,
#     current_class_feasible_radii,
#     resolve_classifier_gradient_conflict,
# )

# from models.geometry_bank import (
#     ClassStructureBatch,
#     merge_class_structure_batches,
#     whole_cds_pair_risk,
#     normalize_ordered_decision_features,
# )
# from models.phase_coordinate_alignment import (
#     correct_historical_structure,
#     rotate_historical_flat_rows,
#     evaluate_fixed_phase_coordinate_transform,
# )
# from models.historical_cds_support import HistoricalCDSSupportBank
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
#     """Clone the complete committed class structure without altering Q or P."""
#     if not isinstance(structure, ClassStructureBatch) or structure.class_count <= 0:
#         raise ValueError("cannot clone an empty class structure")
#     from dataclasses import fields

#     return ClassStructureBatch(
#         **{
#             field.name: getattr(structure, field.name).detach().clone()
#             for field in fields(ClassStructureBatch)
#         }
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

#     @staticmethod
#     def parameter_gradient_audit(objectives, backbone_parameters):
#         """Detached parameter-space gradient norms, without optimizer mutation.

#         Q-space gradient cosines alone do not characterize backbone parameter
#         interference. This audit measures the real trainable parameter groups.
#         """
#         result = {}
#         for name, objective in objectives.items():
#             if not objective.requires_grad:
#                 result[name] = {"grad_norm": 0.0, "active_parameters": 0}
#                 continue
#             grads = torch.autograd.grad(
#                 objective,
#                 backbone_parameters,
#                 retain_graph=True,
#                 allow_unused=True,
#             )
#             norms = [
#                 grad.detach().float().square().sum()
#                 for grad in grads
#                 if grad is not None
#             ]
#             total = torch.stack(norms).sum().sqrt().item() if norms else 0.0
#             result[name] = {
#                 "grad_norm": float(total),
#                 "active_parameters": sum(g is not None for g in grads),
#             }
#         return result

#     @torch.no_grad()
#     def audit_current_classifier_risk(self, loader, *, class_ids, cds):
#         """Report moment approximation error on available current VAL only.

#         Do not use old test examples to tune historical risk or certify drift.
#         """
#         if loader is None:
#             return {"status": "skipped_no_validation", "used_for_selection": False}
#         observed = self.collect_ordered_features(loader)
#         return self.historical_support.audit_expected_ce(
#             self._require_classifier(),
#             class_ids,
#             cds,
#             observed["ordered_features"],
#             observed["labels"],
#         )

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

#             raise RuntimeError(
#                 "seen classes must equal old classes followed by new classes"
#             )

#         if tuple(self.model.committed_class_ids) != old_class_ids:

#             raise RuntimeError("committed historical CDS does not match old classes")

#         return old_class_ids, new_class_ids, seen_class_ids

#     @staticmethod
#     def _assert_frozen_reference(reference_backbone: torch.nn.Module) -> None:

#         reference_backbone.eval()

#         if any(
#             parameter.requires_grad for parameter in reference_backbone.parameters()
#         ):
#             raise RuntimeError("previous backbone reference must remain frozen")
#         if any(
#             parameter.grad is not None for parameter in reference_backbone.parameters()
#         ):
#             raise RuntimeError("previous backbone reference must not carry gradients")

#     @staticmethod
#     def _assert_frozen_reference_classifier(
#         reference_classifier: CosineClassificationHead,
#     ) -> None:

#         reference_classifier.eval()

#         if any(
#             parameter.requires_grad for parameter in reference_classifier.parameters()
#         ):

#             raise RuntimeError("previous classifier reference must remain frozen")

#         if any(
#             parameter.grad is not None
#             for parameter in reference_classifier.parameters()
#         ):

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

#             raise RuntimeError(
#                 f"previous/current {name} state keys differ at phase start"
#             )

#         for key in ref_state:

#             if not torch.equal(
#                 ref_state[key].detach().cpu(), cur_state[key].detach().cpu()
#             ):

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

#             observed = set(
#                 int(value) for value in labels.unique().detach().cpu().tolist()
#             )

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

#             raise RuntimeError(
#                 "incremental TRAIN split is empty or misses a current class"
#             )

#         class_count = len(new_class_ids)

#         weights = torch.tensor(
#             [
#                 total / float(class_count * counts[class_id])
#                 for class_id in new_class_ids
#             ],
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
#     def _max_direction_change(before: Tensor, after: Tensor) -> float:

#         if before.shape != after.shape:

#             raise ValueError("parameter snapshots have different shapes")

#         if before.numel() == 0:

#             return 0.0

#         eps = torch.finfo(before.dtype).eps

#         before_unit = torch.nn.functional.normalize(before, p=2, dim=1, eps=eps)

#         after_unit = torch.nn.functional.normalize(after, p=2, dim=1, eps=eps)

#         return float((after_unit - before_unit).abs().max().item())

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

#     def _run_new_class_head_admission(
#         self,
#         *,
#         classifier: CosineClassificationHead,
#         train_loader: Any,
#         train_eval_loader: Any,
#         old_class_ids: tuple[int, ...],
#         new_class_ids: tuple[int, ...],
#         seen_class_ids: tuple[int, ...],
#         current_class_weights: Tensor,
#         epochs: int,
#         learning_rate: float,
#         gradient_clip: float,
#     ) -> dict[str, float | int]:
#         """Train only incoming rows against ALL seen rows with real current HSI.

#         Backbone Q, old W, future W and tau remain frozen. No old images,
#         feature replay or pseudo-samples enter this stage. The caller must
#         invoke it before the joint incremental optimizer is created.
#         """
#         if epochs < 0:
#             raise ValueError("head admission epochs must be nonnegative")
#         if epochs == 0:
#             return {"epochs": 0, "steps": 0}
#         if not math.isfinite(learning_rate) or learning_rate <= 0:
#             raise ValueError("head admission learning rate must be positive")

#         old_ids = torch.tensor(old_class_ids, dtype=torch.long, device=self.device)
#         new_ids = torch.tensor(new_class_ids, dtype=torch.long, device=self.device)
#         seen_ids = torch.tensor(seen_class_ids, dtype=torch.long, device=self.device)
#         future_ids = torch.tensor(
#             sorted(set(range(classifier.total_class_count)) - set(seen_class_ids)),
#             dtype=torch.long,
#             device=self.device,
#         )
#         old_snapshot = classifier.class_weight(old_ids).detach().clone()
#         future_snapshot = (
#             classifier.class_weight(future_ids).detach().clone()
#             if future_ids.numel()
#             else None
#         )
#         tau_snapshot = classifier.logit_scale_parameter.detach().clone()
#         backbone_snapshot = {
#             k: v.detach().clone() for k, v in self.model.backbone.state_dict().items()
#         }

#         @torch.no_grad()
#         def margin_diagnostics() -> dict[str, float]:
#             """Observe TRUE current-class logit relative to strongest old logit."""
#             self.model.eval()
#             correct_margin_sum = 0.0
#             loss_sum = 0.0
#             count = 0
#             for batch in train_eval_loader:
#                 patch, spectrum, labels = self.unpack_batch(batch)
#                 if not bool(torch.isin(labels, new_ids).all()):
#                     raise RuntimeError(
#                         "admission diagnostics include old/future labels"
#                     )
#                 features = self.model.extract_features(
#                     patch, center_spectrum=spectrum, return_aux=False
#                 ).ordered_features
#                 scores = classifier(
#                     features,
#                     class_ids=seen_ids,
#                     trainable_class_ids=(),
#                 ).logits
#                 local = _targets_local(labels, seen_ids)
#                 true_scores = scores.gather(1, local[:, None]).squeeze(1)
#                 rival_old = scores[:, : len(old_class_ids)].max(dim=1).values
#                 correct_margin_sum += float((true_scores - rival_old).sum().item())
#                 loss_sum += float(
#                     torch.nn.functional.cross_entropy(
#                         scores, local, reduction="sum"
#                     ).item()
#                 )
#                 count += int(labels.numel())
#             return {
#                 "new_vs_old_logit_margin": correct_margin_sum / max(1, count),
#                 "new_all_seen_ce": loss_sum / max(1, count),
#             }

#         before = margin_diagnostics()
#         # The previous teacher is deliberately not needed during head admission:
#         # incoming features come from the fixed phase-start backbone.
#         self.model.eval()
#         optimizer = torch.optim.AdamW(
#             [classifier.weight], lr=learning_rate, weight_decay=0.0
#         )
#         total_loss, total_correct, total_examples, steps = 0.0, 0, 0, 0
#         for _ in range(epochs):
#             for batch in train_loader:
#                 patch, spectrum, labels = self.unpack_batch(batch)
#                 if not bool(torch.isin(labels, new_ids).all()):
#                     raise RuntimeError(
#                         "head admission saw a non-current training class"
#                     )
#                 optimizer.zero_grad(set_to_none=True)
#                 with torch.no_grad():
#                     features = self.model.extract_features(
#                         patch, center_spectrum=spectrum, return_aux=False
#                     ).ordered_features.detach()
#                 logits = classifier(
#                     features,
#                     class_ids=seen_ids,
#                     trainable_class_ids=new_ids,
#                 ).logits
#                 local = _targets_local(labels, seen_ids)
#                 new_local = _targets_local(labels, new_ids)
#                 weights = current_class_weights.index_select(0, new_local)
#                 per_sample = torch.nn.functional.cross_entropy(
#                     logits, local, reduction="none"
#                 )
#                 loss = (per_sample * weights).mean()
#                 if not bool(torch.isfinite(loss)):
#                     raise RuntimeError("nonfinite new-class admission loss")
#                 loss.backward()
#                 gradient = classifier.weight.grad
#                 if gradient is None or not bool(torch.isfinite(gradient).all()):
#                     raise RuntimeError("nonfinite or absent admission gradient")
#                 if self._row_grad_max(gradient, old_ids) != 0.0 or (
#                     future_ids.numel()
#                     and self._row_grad_max(gradient, future_ids) != 0.0
#                 ):
#                     raise RuntimeError("head admission must update incoming rows only")
#                 if gradient_clip > 0:
#                     torch.nn.utils.clip_grad_norm_([classifier.weight], gradient_clip)
#                 optimizer.step()
#                 steps += 1
#                 n = int(labels.numel())
#                 total_examples += n
#                 total_loss += float(loss.detach().item()) * n
#                 total_correct += int(logits.argmax(dim=1).eq(local).sum().item())
#         classifier.weight.grad = None
#         if not torch.equal(classifier.class_weight(old_ids).detach(), old_snapshot):
#             raise RuntimeError("head admission modified a historical classifier row")
#         if future_snapshot is not None and not torch.equal(
#             classifier.class_weight(future_ids).detach(), future_snapshot
#         ):
#             raise RuntimeError("head admission modified a future classifier row")
#         if not torch.equal(classifier.logit_scale_parameter.detach(), tau_snapshot):
#             raise RuntimeError("head admission modified the shared cosine temperature")
#         if any(
#             not torch.equal(value, backbone_snapshot[key])
#             for key, value in self.model.backbone.state_dict().items()
#         ):
#             raise RuntimeError("head admission modified the feature backbone")

#         after = self.evaluate_loader(
#             train_eval_loader,
#             class_ids=seen_class_ids,
#             target_class_ids=new_class_ids,
#         )
#         after_margin = margin_diagnostics()
#         result = {
#             "new_vs_old_logit_margin_before": before["new_vs_old_logit_margin"],
#             "new_vs_old_logit_margin_after": after_margin["new_vs_old_logit_margin"],
#             "new_all_seen_ce_before": before["new_all_seen_ce"],
#             "new_all_seen_ce_after": after_margin["new_all_seen_ce"],
#             "epochs": epochs,
#             "steps": steps,
#             "train_loss": total_loss / max(total_examples, 1),
#             "train_all_seen_accuracy": total_correct / max(total_examples, 1),
#             "new_all_seen_ba_after": float(after["balanced_accuracy"]),
#         }
#         print(
#             f"[New-class head admission] epochs={epochs} steps={steps} "
#             f"all-seen-new-BA={100*result['new_all_seen_ba_after']:.2f}% "
#             f"new-vs-old margin={before['new_vs_old_logit_margin']:.4f}"
#             f" -> {after_margin['new_vs_old_logit_margin']:.4f} "
#             f"CE={before['new_all_seen_ce']:.4f} -> {after_margin['new_all_seen_ce']:.4f} "
#             "| only W_new updated; Q/W_old/W_future/tau unchanged",
#             file=sys.stdout,
#         )
#         return result

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

#         historical_negative_mode = self.cfg("historical_negative_mode", "distribution")
#         if historical_negative_mode not in ("classifier", "mean", "distribution"):
#             raise ValueError("historical_negative_mode must be classifier, mean or distribution")
#         historical_negative_samples = _positive_int(
#             self.cfg("historical_negative_samples_per_class", 32),
#             name="historical_negative_samples_per_class",
#         )
#         # Independent stream: sampling must not perturb dropout or loader order.
#         # Finalized-phase resume reconstructs the same stream for that phase.
#         negative_generator = torch.Generator(device=self.device)
#         negative_generator.manual_seed(int(self.cfg("seed", 0)) + phase_id)
#         effective_negative_samples = (
#             historical_negative_samples if historical_negative_mode == "distribution"
#             else (1 if historical_negative_mode == "mean" else 0)
#         )

#         kd_logit_weight = _finite_nonnegative(
#             "kd_logit_weight", self.cfg("kd_logit_weight", 0.0)
#         )

#         kd_temperature = _finite_positive(
#             "kd_temperature", self.cfg("kd_temperature", 2.0)
#         )

#         cds_weight = _finite_nonnegative("cds_weight", self.cfg("cds_weight", 1.0))

#         # Zero is an explicit preservation ablation, consistent with the CLI.
#         feature_kd_weight = _finite_nonnegative(
#             "feature_kd_weight", self.cfg("feature_kd_weight", 10.0)
#         )
#         representation_ce_scope = self.cfg("representation_ce_scope", "all_seen")
#         if representation_ce_scope not in ("all_seen", "new_only"):
#             raise ValueError("representation_ce_scope must be all_seen or new_only")
#         configured_joint = self.cfg("representation_train_new_head", True)
#         if not isinstance(configured_joint, bool):
#             raise ValueError("representation_train_new_head must be boolean")
#         if not configured_joint or representation_ce_scope != "all_seen":
#             raise ValueError("unified CDS requires all_seen CE and joint seen-head training")
#         representation_train_new_head = configured_joint
#         raw_head_epochs = self.cfg("support_head_epochs", 0)
#         if isinstance(raw_head_epochs, bool) or not isinstance(raw_head_epochs, (Integral, Real)):
#             raise ValueError("support_head_epochs must be a non-negative integer")
#         if not math.isfinite(float(raw_head_epochs)) or float(raw_head_epochs) < 0 or not float(raw_head_epochs).is_integer():
#             raise ValueError("support_head_epochs must be a non-negative integer")
#         recon_epochs = int(raw_head_epochs)

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

#         self.dataset.start_phase(phase_id)

#         old_class_ids, new_class_ids, seen_class_ids = self._phase_classes(phase_id)

#         old_class_tensor = torch.tensor(
#             old_class_ids, device=self.device, dtype=torch.long
#         )

#         new_class_tensor = torch.tensor(
#             new_class_ids, device=self.device, dtype=torch.long
#         )

#         seen_class_tensor = torch.tensor(
#             seen_class_ids, device=self.device, dtype=torch.long
#         )

#         self.historical_support.require(old_class_ids)

#         historical_structure = _clone_structure(
#             self.model.structure_bank.class_structures(old_class_ids)
#         )
#         historical_pair_risk = whole_cds_pair_risk(
#             self.model.class_discriminative_structure(old_class_ids)
#         )[0].detach()

#         self.model.set_incremental_trainability()
#         backbone_trainability = [(p, p.requires_grad) for p in self.model.backbone.parameters()]

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
#             sorted(
#                 int(value)
#                 for value in reference_labels.unique().detach().cpu().tolist()
#             )
#         )

#         if observed_new != tuple(sorted(new_class_ids)):

#             raise RuntimeError(
#                 "incremental TRAIN rows do not match current phase classes"
#             )

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
#         # The before-statistics are measured once in the PREVIOUS model space,
#         # exactly as required by the paired incoming-feature observation protocol. They are
#         # not old-class samples, and are not saved across phases.
#         reference_feature_norm = float(
#             reference_features_full.reshape(reference_features_full.size(0), -1)
#             .norm(dim=1)
#             .mean()
#             .item()
#         )

#         old_weight_before = classifier.class_weight(old_class_tensor).detach().clone()

#         new_weight_phase_start = (
#             classifier.class_weight(new_class_tensor).detach().clone()
#         )

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

#         print(
#             f"[Phase {phase_id} start] "
#             f"NewBA(new-only)={100*float(phase_start_new_only['balanced_accuracy']):.2f}% "
#             f"NewBA(all-seen)={100*float(phase_start_new_all_seen['balanced_accuracy']):.2f}% | "
#             f"Joint Q/W + FeatureKD/CDS/contrast; fixed-Q same-head epochs={recon_epochs}; "
#             f"historical negatives={historical_negative_mode} "
#             f"draws_per_class={effective_negative_samples}; no stored old examples",
#             file=sys.stdout,
#         )

#         @torch.no_grad()
#         def collect_current_state():

#             self.model.eval()

#             rows = self.collect_ordered_features(train_eval_loader)

#             if not torch.equal(rows["labels"], reference_labels):
#                 raise RuntimeError("current TRAIN label order changed")

#             if not torch.equal(rows["raw_center_spectra"], reference_spectra):
#                 raise RuntimeError("current TRAIN spectra changed")

#             current_features_full = rows["ordered_features"].detach()

#             current_structure = self.model.estimate_class_structure(
#                 current_features_full,
#                 None,
#                 reference_labels,
#                 class_ids=new_class_ids,
#                 physical_structure=current_physical_structure,
#             )
#             return current_features_full, current_structure

#         # Prototype initialization on CURRENT TRAIN features only. No old
#         # exemplars, extra head, or trainable module are introduced.
#         # The cosine head compares unit directions, so an incoming class's
#         # observed direction is a better starting point than a random row.
#         with torch.no_grad():
#             incoming_means = current_reference_structure.learned_identity.flatten(1)
#             mean_norms = incoming_means.norm(p=2, dim=1)
#             if not bool((mean_norms > torch.finfo(incoming_means.dtype).eps).all()):
#                 raise RuntimeError("incoming class has a degenerate mean direction")
#             new_prototypes = torch.nn.functional.normalize(
#                 incoming_means, p=2, dim=1
#             ).to(device=classifier.weight.device, dtype=classifier.weight.dtype)
#             if new_prototypes.shape != (
#                 len(new_class_ids),
#                 classifier.feature_dim,
#             ) or not bool(torch.isfinite(new_prototypes).all()):
#                 raise RuntimeError("invalid incoming class prototypes")
#             classifier.weight.index_copy_(0, new_class_tensor, new_prototypes)

#         # Diagnostic classifier admission: learn W_new against all historical
#         # rows before the more restrictive historical robust loss is activated.
#         # This addresses zero new all-seen accuracy without modifying Q or W_old.
#         admission_epochs = int(self.cfg("head_admission_epochs", 0))
#         if admission_epochs < 0:
#             raise ValueError("head_admission_epochs must be nonnegative")
#         admission_lr = _finite_positive(
#             "head_admission_lr", self.cfg("head_admission_lr", 1e-3)
#         )
#         admission_state = self._run_new_class_head_admission(
#             classifier=classifier,
#             train_loader=train_loader,
#             train_eval_loader=train_eval_loader,
#             old_class_ids=old_class_ids,
#             new_class_ids=new_class_ids,
#             seen_class_ids=seen_class_ids,
#             current_class_weights=current_class_weights,
#             epochs=admission_epochs,
#             learning_rate=admission_lr,
#             gradient_clip=gradient_clip,
#         )
#         admission_new_weight = (
#             classifier.class_weight(new_class_tensor).detach().clone()
#         )
#         # One detached orthogonal frame per epoch, fitted only on paired current
#         # TRAIN inputs. Rejection retains identity; native spectra never rotate.
#         historical_transport = bool(self.cfg("historical_transport", True))

#         # Stage 1: COORDINATED Q + one-head W training. The current-data CE
#         # updates both Q and active seen W, while the EXISTING historical CDS
#         # directional historical CDS objective simultaneously protects old rows.
#         # Whole-CDS inter/intra/native-spectrum preservation and contrast train Q.
#         # No extra head or old exemplar. Optional temporary Gaussian Q draws
#         # affect contrastive separation only; directional W supervision stays.

#         representation_parameters = list(backbone_parameters)
#         optimizer_groups = [
#             {"params": backbone_parameters, "weight_decay": weight_decay}
#         ]
#         if representation_train_new_head:
#             # Gradient masking alone cannot protect inactive rows from AdamW
#             # decay. Start a fresh optimizer and disable decay for the head.
#             optimizer_groups.append(
#                 {"params": [classifier.weight], "weight_decay": 0.0}
#             )
#             representation_parameters.append(classifier.weight)
#         optimizer = torch.optim.AdamW(optimizer_groups, lr=learning_rate)

#         scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
#             optimizer, T_max=epoch_count, eta_min=0.0
#         )

#         # Explicit final-epoch contract. All scheduled optimizer steps are kept.
#         # Validation remains read-only reporting and never restores prior Q/W.
#         selected_representation_epoch = epoch_count  # compatibility metadata only
#         selected_representation_score = None

#         history: list[dict[str, Any]] = []

#         gradient_contract_checked = False
#         parameter_gradient_audit = {}
#         parameter_gradient_audits = []

#         progress = tqdm(
#             range(epoch_count),
#             desc=f"Phase {phase_id} representation training",
#             unit="epoch",
#             dynamic_ncols=True,
#             file=sys.stdout,
#         )

#         for epoch in progress:
#             self.model.eval()
#             epoch_features_full, epoch_current_structure = collect_current_state()
#             epoch_historical_structure, epoch_transform, _epoch_alignment = (
#                 correct_historical_structure(
#                     historical_structure=historical_structure,
#                     current_reference_structure=current_reference_structure,
#                     current_structure=epoch_current_structure,
#                     reference_features=reference_features_full,
#                     current_features=epoch_features_full,
#                     use_transport=historical_transport,
#                 )
#             )
#             epoch_support = HistoricalCDSSupportBank(
#                 self.model.structure_bank,
#                 margin_std=self.historical_support.margin_std,
#                 risk_mode=self.historical_support.risk_mode,
#                 coordinate_transform=epoch_transform,
#             )
#             epoch_seen_structure = merge_class_structure_batches(
#                 (epoch_historical_structure, epoch_current_structure)
#             )
#             epoch_seen_cds = self.model.structure_bank.build_cds(epoch_seen_structure)
#             epoch_pair_risk = whole_cds_pair_risk(epoch_seen_cds)[0].detach()
#             epoch_radii, epoch_intrusion = current_class_feasible_radii(
#                 current_features=epoch_features_full,
#                 current_labels=reference_labels,
#                 cds=epoch_seen_cds,
#                 historical_class_count=len(old_class_ids),
#             )

#             self.model.train()

#             sums = {
#                 name: 0.0
#                 for name in (
#                     "classification",
#                     "real_classification",
#                     "direct_head_classification",
#                     "historical_cds_head_loss",
#                     "joint_historical_risk",
#                     "current_real_head_loss",
#                     "old_all_seen_accuracy",
#                     "current_new_only_accuracy",
#                     "feature_kd",
#                     "teacher_hist_relevance",
#                     "feature_rms",
#                     "contrastive",
#                     "diagnostic_sep_loss",
#                     "kd",
#                     "cds",
#                     "intra",
#                     "inter",
#                     "physical_abs",
#                     "physical_rel",
#                     "physical_joint",
#                     "sep_margin",
#                     "sep_violation",
#                     "sep_positive",
#                     "sep_current_negative",
#                     "sep_historical_negative",
#                     "sep_valid",
#                     "sep_physical_confusability",
#                     "sep_pair_risk",
#                     "total",
#                     "core_radius",
#                     "core_violation",
#                     "historical_intrusion",
#                     "spectral_interval_overlap",
#                     "variation_confidence",
#                     "representation_gradient_conflict",
#                     "representation_gradient_cosine",
#                 )
#             }

#             sample_count = 0

#             correct_new = 0

#             for batch in train_loader:

#                 patch, spectrum, labels = self.unpack_batch(batch)

#                 if not bool(torch.isin(labels, new_class_tensor).all()):

#                     raise RuntimeError(
#                         "incremental TRAIN batch contains an old or future class"
#                     )

#                 optimizer.zero_grad(set_to_none=True)

#                 classifier.logit_scale_parameter.grad = None

#                 with torch.no_grad():

#                     reference_features = self.model.extract_features_with_backbone(
#                         reference_backbone,
#                         patch,
#                         center_spectrum=spectrum,
#                         return_aux=False,
#                     ).ordered_features.detach()

#                     physical_evidence = self.model.physical_spectral_evidence(
#                         spectrum
#                     ).detach()

#                     # The teacher remains frozen. Its normalized features and
#                     # historical moments use the same detached epoch frame.
#                     reference_features_in_frame = rotate_historical_flat_rows(
#                         normalize_ordered_decision_features(reference_features).flatten(1),
#                         epoch_transform,
#                     ).reshape_as(reference_features)

#                     teacher_old_logits = (
#                         reference_classifier(
#                             reference_features,
#                             class_ids=old_class_tensor,
#                             trainable_class_ids=(),
#                         ).logits.detach()
#                         if kd_logit_weight > 0
#                         else None
#                     )

#                 current_features = self.model.extract_features(
#                     patch, center_spectrum=spectrum, return_aux=False
#                 ).ordered_features

#                 current_local = _targets_local(labels, new_class_tensor)

#                 sample_weights = current_class_weights.index_select(0, current_local)

#                 # The deployed all-seen boundary receives real rejection evidence
#                 # on old rows and positive evidence on incoming rows.
#                 current_output = classifier(
#                     current_features,
#                     class_ids=seen_class_tensor,
#                     trainable_class_ids=(),
#                 )
#                 representation_output = classifier(
#                     current_features,
#                     class_ids=(
#                         new_class_tensor
#                         if representation_ce_scope == "new_only"
#                         else seen_class_tensor
#                     ),
#                     trainable_class_ids=(
#                         seen_class_tensor if representation_train_new_head else ()
#                     ),
#                 )

#                 real_classification = classification_objective(
#                     logits=representation_output.logits,
#                     class_ids=representation_output.class_ids,
#                     labels_global=labels,
#                     sample_weights=sample_weights,
#                     unbiased_sample_weighting=True,
#                 )

#                 # Direct historical CDS boundary evidence and real current labels
#                 # update the deployed all-seen head. Q is detached on this branch;
#                 # no support examples are synthesized or retained.
#                 # Legacy feasible-core calculation is diagnostic ONLY; no
#                 # gradients or model updates use this surrogate.
#                 with torch.no_grad():
#                     feasible_state = feasible_directional_head_objective(
#                         classifier=classifier,
#                         current_features=current_features.detach(),
#                         current_labels=labels,
#                         seen_cds=epoch_seen_cds,
#                         historical_class_count=len(old_class_ids),
#                         feasible_radii=epoch_radii,
#                         intrusion_fraction=epoch_intrusion,
#                         current_sample_weights=sample_weights,
#                     )
#                 support_classification = feasible_state.state
#                 # Real CE acts on the deployed all-seen decision boundary. The
#                 # same historical CDS risk is optimized in this SAME step when
#                 # the seen head is jointly trainable, instead of arriving only
#                 # after Q has stopped moving.
#                 if representation_train_new_head:
#                     joint_old_risk = epoch_support.historical_risk(
#                         classifier,
#                         old_class_ids,
#                         epoch_seen_cds,
#                         class_pair_risk=epoch_pair_risk,
#                     )
#                     classification_loss = (
#                         len(new_class_ids) * real_classification.loss
#                         + len(old_class_ids) * joint_old_risk.loss
#                     ) / len(seen_class_ids)
#                 else:
#                     joint_old_risk = None
#                     classification_loss = real_classification.loss

#                 # Both sides use the same fixed epoch frame. Native spectral
#                 # support participates in intra/inter preservation and also
#                 # weights teacher relevance; no old inputs are accessed.
#                 cds = historical_cds_preservation_loss(
#                     reference_features=reference_features_in_frame,
#                     current_features=current_features,
#                     physical_spectral_evidence=physical_evidence,
#                     reference_historical_structure=epoch_historical_structure,
#                     current_historical_structure=epoch_historical_structure,
#                     historical_pair_risk=historical_pair_risk,
#                 )
#                 feature_kd = ordered_feature_distillation_loss(
#                     reference_features=reference_features_in_frame,
#                     current_features=current_features,
#                     sample_weights=sample_weights,
#                     historical_relevance=cds.teacher_historical_relevance,
#                 )

#                 if kd_logit_weight > 0:

#                     if teacher_old_logits is None:

#                         raise RuntimeError(
#                             "teacher old logits are missing while logit KD is enabled"
#                         )

#                     student_old_logits = classifier(
#                         current_features,
#                         class_ids=old_class_tensor,
#                         trainable_class_ids=(),
#                     ).logits

#                     kd_loss = logit_distillation_loss(
#                         student_old_logits=student_old_logits,
#                         teacher_old_logits=teacher_old_logits,
#                         temperature=kd_temperature,
#                         sample_weights=sample_weights,
#                     ).loss

#                 else:

#                     kd_loss = current_features.sum() * 0.0

#                 separation = cross_phase_cds_separation_loss(
#                     current_cosine_scores=current_output.cosine_scores,
#                     class_ids=current_output.class_ids,
#                     labels_global=labels,
#                     physical_spectral_evidence=physical_evidence,
#                     seen_cds=epoch_seen_cds,
#                     historical_class_count=len(old_class_ids),
#                     sample_weights=sample_weights,
#                 )
#                 # The reader validates the same fixed epoch frame used by CDS
#                 # preservation. Draws are transient/detached and never acquire
#                 # invented native spectra. Mean mode isolates covariance's role.
#                 negative_features = negative_labels = None
#                 if historical_negative_mode != "classifier":
#                     negative_features, negative_labels = epoch_support.sample_historical_features(
#                         old_class_ids, epoch_seen_cds, effective_negative_samples,
#                         generator=negative_generator,
#                         use_covariance=historical_negative_mode == "distribution",
#                     )
#                 contrastive = whole_cds_contrastive_loss(
#                     current_features=current_features,
#                     labels_global=labels,
#                     physical_spectral_evidence=physical_evidence,
#                     seen_cds=epoch_seen_cds,
#                     historical_class_count=len(old_class_ids),
#                     logit_scale=classifier.logit_scale,
#                     sample_weights=sample_weights,
#                     historical_support=epoch_support,
#                     historical_classifier=classifier,
#                     class_pair_risk=epoch_pair_risk,
#                     historical_negative_mode=historical_negative_mode,
#                     historical_negative_features=negative_features,
#                     historical_negative_labels=negative_labels,
#                 )

#                 if not gradient_contract_checked:

#                     tolerance = 64.0 * torch.finfo(classifier.weight.dtype).eps

#                     real_ce_grad = torch.autograd.grad(
#                         real_classification.loss,
#                         classifier.weight,
#                         retain_graph=True,
#                         allow_unused=True,
#                     )[0]
#                     real_ce_old = self._row_grad_max(real_ce_grad, old_class_tensor)
#                     real_ce_new = self._row_grad_max(real_ce_grad, new_class_tensor)
#                     real_ce_future = self._row_grad_max(
#                         real_ce_grad, future_class_tensor
#                     )
#                     if (
#                         real_ce_future > tolerance
#                         or real_ce_grad is None
#                         or not bool(torch.isfinite(real_ce_grad).all())
#                     ):
#                         raise RuntimeError(
#                             "real all-seen CE must have finite seen-row gradients and zero future-row gradients"
#                         )
#                     if representation_train_new_head:
#                         risk_grad = torch.autograd.grad(
#                             joint_old_risk.loss,
#                             classifier.weight,
#                             retain_graph=True,
#                             allow_unused=True,
#                         )[0]
#                         if (
#                             risk_grad is None
#                             or not bool(torch.isfinite(risk_grad).all())
#                             or self._row_grad_max(risk_grad, future_class_tensor)
#                             > tolerance
#                         ):
#                             raise RuntimeError(
#                                 "joint historical CDS risk must train old/new W only"
#                             )

#                     for name, objective in (
#                         ("Feature KD", feature_kd.loss),
#                         ("Historical Whole-CDS contrastive", contrastive.loss),
#                         ("Whole-CDS preservation", cds.loss),
#                     ):
#                         grad = torch.autograd.grad(
#                             objective,
#                             classifier.weight,
#                             retain_graph=True,
#                             allow_unused=True,
#                         )[0]
#                         if grad is not None and bool(torch.count_nonzero(grad)):
#                             raise RuntimeError(
#                                 f"{name} must not update classifier rows"
#                             )

#                     q_objectives = [
#                         ("real-current all-seen CE", real_classification.loss),
#                         ("Feature KD", feature_kd.loss),
#                         ("Historical Whole-CDS contrastive", contrastive.loss),
#                         ("Whole-CDS preservation", cds.loss),
#                     ]
#                     if kd_logit_weight > 0:
#                         q_objectives.append(("old-logit KD", kd_loss))

#                     for name, objective in q_objectives:
#                         q_grad = torch.autograd.grad(
#                             objective,
#                             current_features,
#                             retain_graph=True,
#                             allow_unused=True,
#                         )[0]
#                         if q_grad is None or not bool(torch.isfinite(q_grad).all()):
#                             raise RuntimeError(
#                                 f"{name} produced invalid current-Q gradient"
#                             )

#                     print(
#                         "[Whole-CDS representation gradient ownership] "
#                         f"RealCE Wold={real_ce_old:.3e} Wnew={real_ce_new:.3e} "
#                         f"| CE scope={representation_ce_scope}; "
#                         f"joint W trainable={representation_train_new_head}; "
#                         "real CE -> W_old/W_new; historical CDS -> W_old/W_new; W_future/tau fixed",
#                         file=sys.stdout,
#                     )

#                     gradient_contract_checked = True

#                 loss = (
#                     classification_loss
#                     + contrastive_weight * contrastive.loss
#                     + feature_kd_weight * feature_kd.loss
#                     + kd_logit_weight * kd_loss
#                     + cds_weight * cds.loss
#                 )

#                 if not bool(torch.isfinite(loss)):

#                     raise RuntimeError(
#                         "incremental representation objective became NaN or Inf"
#                     )

#                 if sample_count == 0 and epoch in (0, epoch_count - 1):
#                     parameter_gradient_audit = self.parameter_gradient_audit(
#                         {
#                             "classification": classification_loss,
#                             "contrastive": contrastive_weight * contrastive.loss,
#                             "feature_kd": feature_kd_weight * feature_kd.loss,
#                             "cds": cds_weight * cds.loss,
#                             "logit_kd": kd_logit_weight * kd_loss,
#                         },
#                         backbone_parameters,
#                     )
#                     parameter_gradient_audit["epoch"] = epoch + 1
#                     parameter_gradient_audits.append(parameter_gradient_audit)

#                 # Observe gradient interference in the actual learned Q output.
#                 # This is a diagnostic, NOT a projected-gradient optimizer. No
#                 # extra distillation or trainable components are introduced.
#                 current_gradient = torch.autograd.grad(
#                     classification_loss, current_features, retain_graph=True
#                 )[0].detach()
#                 preservation_gradient = torch.autograd.grad(
#                     feature_kd_weight * feature_kd.loss
#                     + cds_weight * cds.loss
#                     + kd_logit_weight * kd_loss,
#                     current_features,
#                     retain_graph=True,
#                 )[0].detach()
#                 gradient_dot = (
#                     (current_gradient * preservation_gradient).flatten(1).sum(1)
#                 )
#                 current_norm = current_gradient.flatten(1).norm(dim=1)
#                 preservation_norm = preservation_gradient.flatten(1).norm(dim=1)
#                 gradient_scale = current_norm * preservation_norm
#                 valid_gradient = gradient_scale > 1e-12
#                 gradient_cosine = torch.where(
#                     valid_gradient,
#                     gradient_dot / gradient_scale.clamp_min(1e-12),
#                     torch.zeros_like(gradient_dot),
#                 ).clamp(-1.0, 1.0)
#                 if not bool(torch.isfinite(gradient_cosine).all()):
#                     raise RuntimeError(
#                         "nonfinite Q classification/preservation gradient similarity"
#                     )
#                 sums["representation_gradient_conflict"] += float(
#                     ((gradient_dot < 0) & valid_gradient).sum().item()
#                 )
#                 sums["representation_gradient_cosine"] += float(
#                     gradient_cosine.sum().item()
#                 )

#                 # Interference remains an observation, not a projected update.
#                 loss.backward()

#                 if classifier.logit_scale_parameter.grad is not None:

#                     raise RuntimeError(
#                         "incremental training must not update cosine logit scale"
#                     )

#                 if (
#                     not representation_train_new_head
#                     and classifier.weight.grad is not None
#                 ):
#                     raise RuntimeError("representation step must not train W")

#                 if (
#                     not representation_train_new_head
#                     and self._row_grad_max(classifier.weight.grad, old_class_tensor) > 0
#                 ):
#                     raise RuntimeError("frozen-head representation step updated W_old")

#                 if (
#                     self._row_grad_max(classifier.weight.grad, future_class_tensor)
#                     > 0.0
#                 ):

#                     raise RuntimeError(
#                         "representation stage changed future classifier rows"
#                     )

#                 backbone_grad_seen = False

#                 for parameter in backbone_parameters:

#                     if parameter.grad is None:

#                         continue

#                     if not bool(torch.isfinite(parameter.grad).all()):

#                         raise RuntimeError(
#                             "incremental backbone gradient contains NaN or Inf"
#                         )

#                     backbone_grad_seen = backbone_grad_seen or bool(
#                         torch.count_nonzero(parameter.grad)
#                     )

#                 if not backbone_grad_seen:

#                     raise RuntimeError(
#                         "representation objective produced zero backbone gradient"
#                     )

#                 if gradient_clip > 0:

#                     torch.nn.utils.clip_grad_norm_(
#                         representation_parameters,
#                         gradient_clip,
#                         error_if_nonfinite=True,
#                     )

#                 optimizer.step()

#                 self._assert_future_rows_unchanged(
#                     classifier, future_class_tensor, future_weight_before
#                 )

#                 n = int(labels.numel())
#                 sums["core_radius"] += float(feasible_state.mean_core_radius) * n
#                 sums["core_violation"] += (
#                     float(feasible_state.effective_core_violation) * n
#                 )
#                 sums["historical_intrusion"] += (
#                     float(feasible_state.historical_intrusion_fraction) * n
#                 )
#                 sums["spectral_interval_overlap"] += (
#                     float(feasible_state.physical_interval_overlap) * n
#                 )
#                 sums["variation_confidence"] += (
#                     float(feasible_state.ordered_variation_confidence) * n
#                 )

#                 sums["classification"] += float(classification_loss.detach().item()) * n
#                 if joint_old_risk is not None:
#                     sums["joint_historical_risk"] += (
#                         float(joint_old_risk.loss.detach().item()) * n
#                     )
#                 sums["real_classification"] += (
#                     float(real_classification.loss.detach().item()) * n
#                 )
#                 sums["direct_head_classification"] += (
#                     float(support_classification.balanced_loss.detach().item()) * n
#                 )
#                 sums["historical_cds_head_loss"] += (
#                     float(
#                         support_classification.historical_positive_loss.detach().item()
#                     )
#                     * n
#                 )
#                 sums["current_real_head_loss"] += (
#                     float(support_classification.current_positive_loss.detach().item())
#                     * n
#                 )
#                 sums["old_all_seen_accuracy"] += (
#                     float(support_classification.real_all_seen_accuracy.detach().item())
#                     * n
#                 )
#                 sums["current_new_only_accuracy"] += (
#                     float(support_classification.new_only_accuracy.detach().item()) * n
#                 )

#                 sums["feature_kd"] += float(feature_kd.loss.detach().item()) * n
#                 sums["teacher_hist_relevance"] += (
#                     float(cds.teacher_historical_relevance.detach().mean().item()) * n
#                 )

#                 sums["feature_rms"] += float(feature_kd.rms_drift.detach().item()) * n

#                 sums["contrastive"] += float(contrastive.loss.detach().item()) * n
#                 sums["diagnostic_sep_loss"] += float(separation.loss.detach()) * n

#                 sums["kd"] += float(kd_loss.detach().item()) * n

#                 sums["cds"] += float(cds.loss.detach().item()) * n

#                 sums["intra"] += float(cds.intra_loss.detach().item()) * n

#                 sums["inter"] += float(cds.inter_loss.detach().item()) * n

#                 sums["physical_abs"] += (
#                     float(cds.physical_absolute_support_mean.detach().item()) * n
#                 )

#                 sums["physical_rel"] += (
#                     float(cds.physical_relation_support_mean.detach().item()) * n
#                 )

#                 sums["physical_joint"] += (
#                     float(cds.physical_relevance_mean.detach().item()) * n
#                 )

#                 sums["sep_margin"] += (
#                     float(separation.decision_margin_mean.detach().item()) * n
#                 )

#                 sums["sep_violation"] += (
#                     float(separation.violation_fraction.detach().item()) * n
#                 )

#                 sums["sep_positive"] += (
#                     float(separation.positive_decision_score_mean.detach().item()) * n
#                 )

#                 # New/new competition is already handled by real all-seen CE;

#                 # this CDS loss contains only current-positive vs historical rivals.

#                 sums["sep_current_negative"] += 0.0

#                 sums["sep_historical_negative"] += (
#                     float(
#                         separation.historical_rival_decision_score_mean.detach().item()
#                     )
#                     * n
#                 )

#                 sums["sep_valid"] += 1.0 * n

#                 sums["sep_physical_confusability"] += (
#                     float(separation.physical_confusability_mean.detach().item()) * n
#                 )

#                 sums["sep_pair_risk"] += (
#                     float(separation.effective_pair_risk_mean.detach().item()) * n
#                 )

#                 sums["total"] += float(loss.detach().item()) * n

#                 correct_new += int(
#                     current_output.prediction.detach().eq(labels).sum().item()
#                 )

#                 sample_count += n

#             if sample_count != expected_samples:

#                 raise RuntimeError(
#                     "incremental TRAIN loader did not traverse the complete split"
#                 )

#             scheduler.step()

#             if float(classifier.logit_scale.detach().item()) != phase_start_scale:

#                 raise RuntimeError(
#                     "cosine logit scale changed during representation training"
#                 )

#             self._assert_future_rows_unchanged(
#                 classifier, future_class_tensor, future_weight_before
#             )

#             self.model.eval()

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
#                 "stage": "representation",
#                 "epoch": epoch + 1,
#                 "learning_rate": float(optimizer.param_groups[0]["lr"]),
#                 "classification_loss": sums["classification"] / sample_count,
#                 "joint_historical_risk_loss": sums["joint_historical_risk"]
#                 / sample_count,
#                 "real_current_classification_loss": sums["real_classification"]
#                 / sample_count,
#                 "direct_cds_classifier_loss": sums["direct_head_classification"]
#                 / sample_count,
#                 "direct_cds_historical_loss": sums["historical_cds_head_loss"]
#                 / sample_count,
#                 "direct_cds_current_loss": sums["current_real_head_loss"]
#                 / sample_count,
#                 "direct_old_all_seen_accuracy": sums["old_all_seen_accuracy"]
#                 / sample_count,
#                 "direct_current_new_only_accuracy": sums["current_new_only_accuracy"]
#                 / sample_count,
#                 "historical_core_radius": sums["core_radius"] / sample_count,
#                 "historical_core_violation": sums["core_violation"] / sample_count,
#                 "current_in_old_support_fraction": sums["historical_intrusion"]
#                 / sample_count,
#                 "spectral_band_interval_overlap": sums["spectral_interval_overlap"]
#                 / sample_count,
#                 "ordered_variation_confidence": sums["variation_confidence"]
#                 / sample_count,
#                 "representation_gradient_conflict_fraction": sums[
#                     "representation_gradient_conflict"
#                 ]
#                 / sample_count,
#                 "representation_gradient_cosine": sums["representation_gradient_cosine"]
#                 / sample_count,
#                 "feature_kd_loss": sums["feature_kd"] / sample_count,
#                 "teacher_historical_relevance_mean": sums["teacher_hist_relevance"]
#                 / sample_count,
#                 "coordinate_stability_loss": sums["feature_kd"] / sample_count,
#                 "epoch_representation_drift_rms": sums["feature_rms"] / sample_count,
#                 "contrastive_loss": sums["contrastive"] / sample_count,
#                 "historical_negative_mode": historical_negative_mode,
#                 "historical_negative_samples_per_class": effective_negative_samples,
#                 "cds_separation_loss": sums["diagnostic_sep_loss"]
#                 / sample_count,  # read-only diagnostic
#                 "cds_separation_margin": sums["sep_margin"] / sample_count,
#                 "cds_separation_violation_fraction": sums["sep_violation"]
#                 / sample_count,
#                 "contrastive_positive_similarity": float(
#                     contrastive.positive_similarity_mean.detach()
#                 ),
#                 "contrastive_current_negative_similarity": float(
#                     contrastive.current_negative_similarity_mean.detach()
#                 ),
#                 "contrastive_historical_negative_similarity": float(
#                     contrastive.historical_negative_similarity_mean.detach()
#                 ),
#                 "contrastive_valid_anchor_fraction": float(
#                     contrastive.valid_anchor_fraction.detach()
#                 ),
#                 "physical_confusability_mean": sums["sep_physical_confusability"]
#                 / sample_count,
#                 "effective_pair_risk_mean": sums["sep_pair_risk"] / sample_count,
#                 "kd_logit_loss": sums["kd"] / sample_count,
#                 "cds_preservation_loss": sums["cds"] / sample_count,
#                 "cds_intra_loss": sums["intra"] / sample_count,
#                 "cds_inter_loss": sums["inter"] / sample_count,
#                 "physical_absolute_support_mean": sums["physical_abs"] / sample_count,
#                 "physical_relation_support_mean": sums["physical_rel"] / sample_count,
#                 "physical_relevance_mean": sums["physical_joint"] / sample_count,
#                 "total_loss": sums["total"] / sample_count,
#                 "current_new_all_seen_accuracy": correct_new / float(sample_count),
#                 "new_only_ba": float(new_only["balanced_accuracy"]),
#                 "new_all_seen_ba": float(new_all["balanced_accuracy"]),
#                 "validation_new_all_seen_ba": (
#                     None
#                     if validation_all is None
#                     else float(validation_all["balanced_accuracy"])
#                 ),
#             }

#             # paired-feature evidence quality, observed ONLY on incoming classes.
#             record["alignment_holdout_candidate_error"] = float(
#                 _epoch_alignment.holdout_candidate_error.item()
#             )
#             record["alignment_holdout_identity_error"] = float(
#                 _epoch_alignment.holdout_identity_error.item()
#             )
#             record["alignment_transport_accepted"] = bool(
#                 _epoch_alignment.transport_accepted.item()
#             )
#             # Legacy report fields are preserved without checkpoint ranking.
#             record["validation_joint_selection_objective"] = None
#             record["validation_historical_cds_risk"] = None
#             record["validation_real_new_ce"] = None
#             history.append(record)

#             tqdm.write(
#                 f"[Representation {epoch + 1:03d}/{epoch_count:03d}] "
#                 f"JointObj={record['classification_loss']:.4f} "
#                 f"CEreal={record['real_current_classification_loss']:.4f} "
#                 f"LegacyCoreDiag={record['direct_cds_classifier_loss']:.4f} "
#                 f"Con={record['contrastive_loss']:.4f} "
#                 f"CoreR={record['historical_core_radius']:.3f} "
#                 f"CoreViol={100*record['historical_core_violation']:.1f}% "
#                 f"GradConflict={100*record['representation_gradient_conflict_fraction']:.1f}% "
#                 f"QGradientCos={record['representation_gradient_cosine']:.3f} "
#                 f"FKD={record['feature_kd_loss']:.4f} "
#                 # f"KD={record['kd_logit_loss']:.4f} "
#                 f"CDS={record['cds_preservation_loss']:.4f}"
#                 f"[I={record['cds_intra_loss']:.4f},R={record['cds_inter_loss']:.4f}] "
#                 f"DecisionMargin={record['cds_separation_margin']:.4f} "
#                 f"DecisionViol={100*record['cds_separation_violation_fraction']:.1f}% | "
#                 f"NewOnly/All={100*record['new_only_ba']:.2f}%/"
#                 f"{100*record['new_all_seen_ba']:.2f}%",
#                 file=sys.stdout,
#             )

#         progress.close()

#         if not history:

#             raise RuntimeError("incremental representation training produced no epochs")

#         # The deployed representation and cosine rows are exactly those
#         # produced by the final optimizer step, not a validation-selected copy.
#         self._assert_future_rows_unchanged(
#             classifier, future_class_tensor, future_weight_before
#         )
#         print(
#             f"[Joint Q+W final epoch] retained epoch {epoch_count}/{epoch_count}; "
#             "no checkpoint ranking, no Q/W rollback",
#             file=sys.stdout,
#         )

#         # Phase-end statistical synchronization. This is NOT another training stage.

#         self.model.eval()

#         for parameter in self.model.backbone.parameters():

#             parameter.requires_grad_(False)

#             parameter.grad = None

#         final_features_full, final_current_structure = collect_current_state()
#         # Before/after migration is measured on the same current TRAIN samples.
#         # This is an observable signal for historical Q-distribution alignment,
#         # not proof of how unavailable historical samples migrated.
#         with torch.no_grad():
#             before_q = torch.nn.functional.normalize(
#                 reference_features_full.reshape(reference_features_full.size(0), -1),
#                 dim=1,
#             )
#             after_q = torch.nn.functional.normalize(
#                 final_features_full.reshape(final_features_full.size(0), -1), dim=1
#             )
#             paired_new_feature_cosine = float(
#                 (before_q * after_q).sum(dim=1).mean().item()
#             )
#             paired_new_feature_chord = float(
#                 (before_q - after_q).square().sum(dim=1).mean().sqrt().item()
#             )

#         alignment_spectral_evidence = self.model.physical_spectral_evidence(
#             reference_spectra
#         )
#         if recon_epochs:
#             # A final refit is allowed only when the same head will subsequently
#             # train against this exact fixed bank coordinate.
#             final_historical_structure, final_transform, final_alignment = correct_historical_structure(
#                 historical_structure=historical_structure,
#                 current_reference_structure=current_reference_structure,
#                 current_structure=final_current_structure,
#                 reference_features=reference_features_full,
#                 current_features=final_features_full,
#                 use_transport=historical_transport,
#             )
#             historical_frame_source = "final_alignment_then_head_training"
#         else:
#             # Commit the last frame that actually supervised the deployed head.
#             # Measure its final residual without fitting an unused replacement.
#             final_historical_structure = epoch_historical_structure
#             final_transform = epoch_transform
#             final_metrics = evaluate_fixed_phase_coordinate_transform(
#                 reference_features_full, final_features_full, final_transform
#             )
#             final_alignment = replace(_epoch_alignment, **final_metrics)
#             historical_frame_source = "last_representation_epoch"
#         committed_structure = merge_class_structure_batches(
#             (final_historical_structure, final_current_structure)
#         )

#         committed_ids = tuple(
#             int(v) for v in committed_structure.class_ids.detach().cpu().tolist()
#         )
#         if committed_ids != seen_class_ids:
#             raise RuntimeError(
#                 "final Whole-CDS structure produced incorrect seen-class order"
#             )

#         # The live bank stays immutable until the candidate and any requested
#         # head training succeed. Old means/bases rotate together; native P stays fixed.
#         candidate_bank = deepcopy(self.model.structure_bank)
#         # The reference->student map is a validated orthogonal rotation. Means
#         # and low-rank covariance bases must commit in exactly the same frame.
#         candidate_bank.apply_historical_orthogonal_transform(
#             old_class_ids, final_transform
#         )
#         candidate_bank.replace_all_with_moments(
#             committed_structure,
#             final_features_full,
#             alignment_spectral_evidence,
#             reference_labels,
#         )
#         candidate_bank.validate_state()
#         committed_seen_cds = candidate_bank.build_cds(candidate_bank.class_structures())
#         reconstruction_support = HistoricalCDSSupportBank(
#             candidate_bank,
#             margin_std=self.historical_support.margin_std,
#             risk_mode=self.historical_support.risk_mode,
#         )
#         reconstruction_pair_risk = whole_cds_pair_risk(committed_seen_cds)[0].detach()

#         if not torch.equal(
#             final_historical_structure.physical_identity,
#             historical_structure.physical_identity,
#         ) or not torch.equal(
#             final_historical_structure.physical_variation,
#             historical_structure.physical_variation,
#         ):
#             raise RuntimeError("final historical correction changed physical I_P/V_P")

#         # Preserve only model parameters for later reporting. No reporting/test
#         # example is opened until reconstruction and phase finalization finish.
#         post_rep_classifier = deepcopy(classifier).requires_grad_(False).eval()
#         post_rep_new_only = self.evaluate_loader(
#             train_eval_loader, class_ids=new_class_ids, target_class_ids=new_class_ids
#         )
#         post_rep_new_all = self.evaluate_loader(
#             train_eval_loader, class_ids=seen_class_ids, target_class_ids=new_class_ids
#         )

#         # Current-task learned-feature movement report: both structures were fitted to EXACTLY
#         # the same current TRAIN class IDs and spectra, before/after backbone
#         # optimization. Only per-class aggregate diagnostics are persisted.
#         classwise_new_migration = {}
#         classwise_old_transport = {}
#         with torch.no_grad():
#             for row, cls_id in enumerate(new_class_ids):
#                 before = current_reference_structure.learned_identity[row].reshape(-1)
#                 after = final_current_structure.learned_identity[row].reshape(-1)
#                 before_rel = current_reference_structure.learned_variation[row]
#                 after_rel = final_current_structure.learned_variation[row]
#                 before_disp = current_reference_structure.learned_absolute_deviation[
#                     row
#                 ]
#                 after_disp = final_current_structure.learned_absolute_deviation[row]
#                 classwise_new_migration[str(cls_id)] = {
#                     "identity_delta_l2": float((after - before).norm().item()),
#                     "identity_cosine": float(
#                         torch.nn.functional.cosine_similarity(
#                             before[None], after[None]
#                         ).item()
#                     ),
#                     "relational_interval_change_l2": float(
#                         (after_rel - before_rel).norm().item()
#                     ),
#                     "learned_absolute_deviation_delta_l2": float(
#                         (after_disp - before_disp).norm().item()
#                     ),
#                     "learned_absolute_deviation_mean_before": float(
#                         before_disp.mean().item()
#                     ),
#                     "learned_absolute_deviation_mean_after": float(
#                         after_disp.mean().item()
#                     ),
#                 }
#             for row, cls_id in enumerate(old_class_ids):
#                 # Measured only on the compact stored class means; neither
#                 # accepted nor rejected transport certifies unavailable old drift.
#                 classwise_old_transport[str(cls_id)] = {
#                     "stored_identity_delta_l2": float(
#                         (
#                             final_historical_structure.learned_identity[row]
#                             - historical_structure.learned_identity[row]
#                         )
#                         .norm()
#                         .item()
#                     ),
#                     "stored_dispersion_delta_l2": float(
#                         (
#                             final_historical_structure.learned_absolute_deviation[row]
#                             - historical_structure.learned_absolute_deviation[row]
#                         )
#                         .norm()
#                         .item()
#                     ),
#                     "unobserved_old_feature_drift_certified": False,
#                 }

#         old_weight_after_training = (
#             classifier.class_weight(old_class_tensor).detach().clone()
#         )
#         if not representation_train_new_head and not torch.equal(
#             old_weight_after_training, old_weight_before
#         ):
#             raise RuntimeError(
#                 "frozen-head ablation must preserve all historical W rows"
#             )
#         new_weight_after_training = (
#             classifier.class_weight(new_class_tensor).detach().clone()
#         )
#         self._assert_future_rows_unchanged(
#             classifier, future_class_tensor, future_weight_before
#         )
#         if float(classifier.logit_scale.detach().item()) != phase_start_scale:
#             raise RuntimeError("incremental representation changed cosine logit scale")

#         # Read-only final diagnostics before actual frozen-Q reconstruction.
#         final_classifier_objective = whole_cds_class_balanced_classifier_loss(
#             classifier=classifier,
#             cds=committed_seen_cds,
#             historical_class_count=len(old_class_ids),
#         )
#         pre_classifier_objective = final_classifier_objective
#         pre_finalize = final_classifier_objective

#         final_radii, final_intrusion = current_class_feasible_radii(
#             current_features=final_features_full,
#             current_labels=reference_labels,
#             cds=committed_seen_cds,
#             historical_class_count=len(old_class_ids),
#         )
#         final_feasible_state = feasible_directional_head_objective(
#             classifier=classifier,
#             current_features=final_features_full.detach(),
#             current_labels=reference_labels,
#             seen_cds=committed_seen_cds,
#             historical_class_count=len(old_class_ids),
#             feasible_radii=final_radii,
#             intrusion_fraction=final_intrusion,
#         )
#         final_classifier_state = final_feasible_state.state
#         with torch.no_grad():
#             diagnostic_seen = classifier(
#                 final_features_full,
#                 class_ids=seen_class_tensor,
#                 trainable_class_ids=(),
#             )
#             diagnostic_new_ce = classification_objective(
#                 logits=diagnostic_seen.logits,
#                 class_ids=diagnostic_seen.class_ids,
#                 labels_global=reference_labels,
#             ).loss
#             diagnostic_new_all_accuracy = (
#                 diagnostic_seen.prediction.eq(reference_labels)
#                 .to(diagnostic_seen.logits.dtype)
#                 .mean()
#             )
#             diagnostic_new_only_output = classifier(
#                 final_features_full,
#                 class_ids=new_class_tensor,
#                 trainable_class_ids=(),
#             )
#             diagnostic_new_only_accuracy = (
#                 diagnostic_new_only_output.prediction.eq(reference_labels)
#                 .to(diagnostic_new_only_output.logits.dtype)
#                 .mean()
#             )

#         print(
#             "[Before Whole-CDS support reconstruction: legacy diagnostic ONLY] "
#             f"JointLoss={float(final_classifier_state.balanced_loss.detach().item()):.4f} "
#             f"OldCDSLoss={float(final_classifier_state.historical_positive_loss.detach().item()):.4f} "
#             f"NewRealLoss={float(final_classifier_state.current_positive_loss.detach().item()):.4f} "
#             f"RealNewCE={float(diagnostic_new_ce.detach().item()):.4f} "
#             f"NewAllAcc={100*float(diagnostic_new_all_accuracy.detach().item()):.2f}% "
#             f"NewOnlyAcc={100*float(diagnostic_new_only_accuracy.detach().item()):.2f}% "
#             f"OldNewViol={100*float(final_classifier_objective.old_new_violation_fraction.detach().item()):.1f}%",
#             file=sys.stdout,
#         )

#         # Optional existing same-head budget: fixed final Q and fixed final CDS.
#         # Keep all configured optimizer steps; validation scores are audit only.
#         recon_lr = _finite_positive(
#             "support_head_lr", self.cfg("support_head_lr", 1e-3)
#         )

#         self.model.eval()
#         # Zero-epoch diagnostics must not freeze the deployed student backbone.
#         if recon_epochs:
#             for parameter in self.model.backbone.parameters():
#                 parameter.requires_grad_(False)
#                 parameter.grad = None
#         classifier.weight.grad = None
#         fixed_q = final_features_full.detach()
#         fixed_targets = reference_labels.detach()
#         # The features are transient and are discarded after reconstruction.
#         # Each configured epoch now traverses minibatches (not one full-batch
#         # gradient step). This matches the meaning of training epochs and
#         # permits classifier adaptation against old + new classes.
#         targets_seen = _targets_local(fixed_targets, seen_class_tensor)
#         recon_opt = torch.optim.AdamW(
#             [classifier.weight], lr=recon_lr, weight_decay=0.0
#         )
#         history_reconstruction = []
#         selection_source = "current_train_only"
#         selection_q, selection_y = fixed_q, fixed_targets
#         if val_loader is not None:
#             with torch.no_grad():
#                 valid_rows = self.collect_ordered_features(val_loader)
#                 valid_q = valid_rows["ordered_features"].detach()
#                 valid_y = valid_rows["labels"].detach()
#                 if all(bool((valid_y == cid).any()) for cid in new_class_ids):
#                     selection_q, selection_y = valid_q, valid_y
#                     selection_source = "current_validation"
#                 else:
#                     selection_source = "current_train_only_missing_validation_classes"

#         @torch.no_grad()
#         def reconstruction_objective():
#             historical = reconstruction_support.historical_risk(
#                 classifier,
#                 old_class_ids,
#                 committed_seen_cds,
#                 class_pair_risk=reconstruction_pair_risk,
#             )
#             outputs = classifier(
#                 selection_q,
#                 class_ids=seen_class_tensor,
#                 trainable_class_ids=(),
#             )
#             targets = _targets_local(selection_y, seen_class_tensor)
#             sample_ce = torch.nn.functional.cross_entropy(
#                 outputs.logits,
#                 targets,
#                 reduction="none",
#             )
#             new_ce = torch.stack(
#                 [sample_ce[selection_y.eq(cid)].mean() for cid in new_class_ids]
#             ).mean()
#             criterion = (
#                 len(old_class_ids) * historical.loss + len(new_class_ids) * new_ce
#             ) / len(seen_class_ids)
#             return (
#                 float(criterion.item()),
#                 float(historical.loss.item()),
#                 float(new_ce.item()),
#             )

#         initial_score, initial_old_risk, initial_new_ce = reconstruction_objective()
#         selected_score = initial_score
#         selected_epoch = 0
#         history_reconstruction.append(
#             {
#                 "epoch": 0,
#                 "optimizer_steps": 0,
#                 "selection_objective": initial_score,
#                 "historical_support_loss": initial_old_risk,
#                 "selection_new_ce": initial_new_ce,
#                 "selection_source": selection_source,
#             }
#         )
#         for rec_epoch in range(recon_epochs):
#             permutation = torch.randperm(
#                 fixed_targets.numel(), device=fixed_targets.device
#             )
#             steps = 0
#             for rows in permutation.split(train_batch_size):
#                 batch_q = fixed_q.index_select(0, rows)
#                 batch_labels = fixed_targets.index_select(0, rows)
#                 batch_targets = targets_seen.index_select(0, rows)
#                 local_new = _targets_local(batch_labels, new_class_tensor)
#                 weights = current_class_weights.index_select(0, local_new)

#                 recon_opt.zero_grad(set_to_none=True)
#                 risk = reconstruction_support.historical_risk(
#                     classifier,
#                     old_class_ids,
#                     committed_seen_cds,
#                     class_pair_risk=reconstruction_pair_risk,
#                 )
#                 out = classifier(
#                     batch_q,
#                     class_ids=seen_class_tensor,
#                     trainable_class_ids=seen_class_tensor,
#                 )
#                 per_sample = torch.nn.functional.cross_entropy(
#                     out.logits, batch_targets, reduction="none"
#                 )
#                 # A minibatch need not contain every incoming class.
#                 # Class weights estimated from the entire incoming TRAIN split
#                 # approximate one equal vote per incoming class.
#                 new_ce = (per_sample * weights).mean()
#                 # Each seen class has equal weight, not each TASK.
#                 joint = (
#                     len(old_class_ids) * risk.loss + len(new_class_ids) * new_ce
#                 ) / len(seen_class_ids)
#                 if not bool(torch.isfinite(joint)):
#                     raise RuntimeError("nonfinite joint-moment classifier objective")
#                 joint.backward()
#                 if classifier.weight.grad is None or not bool(
#                     torch.isfinite(classifier.weight.grad).all()
#                 ):
#                     raise RuntimeError(
#                         "historical/current moment reconstruction gradient invalid"
#                     )
#                 if (
#                     future_class_ids
#                     and self._row_grad_max(classifier.weight.grad, future_class_tensor)
#                     > 0
#                 ):
#                     raise RuntimeError(
#                         "moment reconstruction gradient leaked into future rows"
#                     )
#                 if gradient_clip > 0:
#                     torch.nn.utils.clip_grad_norm_(
#                         [classifier.weight], gradient_clip, error_if_nonfinite=True
#                     )
#                 recon_opt.step()
#                 self._assert_future_rows_unchanged(
#                     classifier, future_class_tensor, future_weight_before
#                 )
#                 steps += 1

#             # Log POST-epoch scores on the same fixed incoming TRAIN features.
#             with torch.no_grad():
#                 risk = reconstruction_support.historical_risk(
#                     classifier,
#                     old_class_ids,
#                     committed_seen_cds,
#                     class_pair_risk=reconstruction_pair_risk,
#                 )
#                 out = classifier(
#                     fixed_q,
#                     class_ids=seen_class_tensor,
#                     trainable_class_ids=(),
#                 )
#                 ce = torch.nn.functional.cross_entropy(
#                     out.logits, targets_seen, reduction="none"
#                 )
#                 new_ce = torch.stack(
#                     [ce[fixed_targets.eq(cid)].mean() for cid in new_class_ids]
#                 ).mean()
#                 joint = (
#                     len(old_class_ids) * risk.loss + len(new_class_ids) * new_ce
#                 ) / len(seen_class_ids)
#                 true_new_logits = out.logits.gather(1, targets_seen[:, None]).squeeze(1)
#                 strongest_old_logit = out.logits[:, : len(old_class_ids)].max(1).values
#                 row = {
#                     "epoch": rec_epoch + 1,
#                     "optimizer_steps": steps,
#                     "historical_support_loss": float(risk.loss),
#                     "historical_objective": risk.objective,
#                     "historical_mean_ce": (
#                         None if risk.mean_ce_loss is None else float(risk.mean_ce_loss)
#                     ),
#                     "historical_curvature": (
#                         None
#                         if risk.curvature_correction is None
#                         else float(risk.curvature_correction)
#                     ),
#                     "current_real_ce": float(new_ce),
#                     "balanced_loss": float(joint),
#                     "new_vs_old_logit_margin": float(
#                         (true_new_logits - strongest_old_logit).mean()
#                     ),
#                     "new_all_seen_train_acc": float(
#                         out.prediction.eq(fixed_targets).float().mean()
#                     ),
#                     "hardest_historical_rival": risk.per_class_worst_rival.cpu().tolist(),
#                     "historical_classwise_moment_margin_penalty": risk.per_class_loss.cpu().tolist(),
#                 }
#             score, held_old_risk, held_new_ce = reconstruction_objective()
#             row["selection_objective"] = score
#             row["selection_new_ce"] = held_new_ce
#             row["selection_old_risk"] = held_old_risk
#             row["selection_source"] = selection_source
#             history_reconstruction.append(row)
#             selected_score = score
#             selected_epoch = rec_epoch + 1
#         classifier.weight.grad = None
#         # Refresh all diagnostics after the actual deployed W has been fitted.
#         final_feasible_state = feasible_directional_head_objective(
#             classifier=classifier,
#             current_features=final_features_full.detach(),
#             current_labels=reference_labels,
#             seen_cds=committed_seen_cds,
#             historical_class_count=len(old_class_ids),
#             feasible_radii=final_radii,
#             intrusion_fraction=final_intrusion,
#         )
#         final_classifier_state = final_feasible_state.state
#         final_classifier_objective = whole_cds_class_balanced_classifier_loss(
#             classifier=classifier,
#             cds=committed_seen_cds,
#             historical_class_count=len(old_class_ids),
#         )
#         final_head_accepted = sum(row["optimizer_steps"] for row in history_reconstruction)
#         print(
#             f"[Joint Q+W final-state diagnostics] same_head_epochs={recon_epochs}; "
#             f"initial_old_surrogate={initial_old_risk:.4f} initial_new_CE={initial_new_ce:.4f}; "
#             "final optimizer state retained; validation is audit only",
#             file=sys.stdout,
#         )
#         # Commit the complete, validated candidate after reconstruction.
#         # Failed reconstruction leaves the historical CDS bank unchanged.
#         self.model.structure_bank.load_state_dict(
#             candidate_bank.state_dict(), strict=True
#         )
#         self.historical_support.require(seen_class_ids)
#         self.model.validate_finalized_state()
#         risk_audit = self.audit_current_classifier_risk(
#             val_loader, class_ids=new_class_ids, cds=committed_seen_cds
#         )

#         self.dataset.finalize_phase(phase_id)

#         for parameter, trainable in backbone_trainability:
#             parameter.requires_grad_(trainable)
#         classifier.logit_scale_parameter.requires_grad_(scale_requires_grad)

#         classifier.logit_scale_parameter.grad = None

#         old_reporting_loader = self.dataset.get_reporting_dataloader(
#             previous_phase, batch_size=eval_batch_size
#         )
#         live_backbone, live_classifier = self.model.backbone, self.model.classifier
#         try:
#             self.model.backbone = reference_backbone
#             self.model.classifier = reference_classifier
#             old_before = self.evaluate_loader(
#                 old_reporting_loader,
#                 class_ids=previous_seen_class_ids,
#                 target_class_ids=previous_seen_class_ids,
#             )
#             # Q did not change during head reconstruction; the saved W therefore
#             # reproduces the post-representation decision on this reporting split.
#             self.model.backbone = live_backbone
#             self.model.classifier = post_rep_classifier
#             post_rep_old_only = self.evaluate_loader(
#                 old_reporting_loader,
#                 class_ids=old_class_ids,
#                 target_class_ids=old_class_ids,
#             )
#             post_rep_old_all = self.evaluate_loader(
#                 old_reporting_loader,
#                 class_ids=seen_class_ids,
#                 target_class_ids=old_class_ids,
#             )
#         finally:
#             self.model.backbone, self.model.classifier = live_backbone, live_classifier
#         del post_rep_classifier

#         # Step 12: all-seen evaluation uses exactly tau*cosine(Q,W).

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

#         final_old_only = self.evaluate_loader(
#             old_reporting_loader,
#             class_ids=old_class_ids,
#             target_class_ids=old_class_ids,
#         )

#         final_old_all = self.evaluate_loader(
#             old_reporting_loader,
#             class_ids=seen_class_ids,
#             target_class_ids=old_class_ids,
#         )

#         final_new_only = self.evaluate_loader(
#             cumulative_loader, class_ids=new_class_ids, target_class_ids=new_class_ids
#         )

#         final_new_all = self.evaluate_loader(
#             cumulative_loader, class_ids=seen_class_ids, target_class_ids=new_class_ids
#         )

#         final_record = history[-1]
#         final_old_weight = classifier.class_weight(old_class_tensor).detach()
#         final_new_weight = classifier.class_weight(new_class_tensor).detach()
#         method_state = {
#             "stores_old_images": False,
#             "stores_old_feature_rows": False,
#             "stores_synthetic_support": False,
#             "stores_transient_distribution_support": False,
#             "uses_transient_historical_distribution_support": historical_negative_mode == "distribution",
#             "historical_negative_mode": historical_negative_mode,
#             "historical_negative_samples_per_class": effective_negative_samples,
#             "historical_negative_aggregation": "per_class_log_mean_exp" if historical_negative_mode != "classifier" else "single_direction",
#             "historical_negative_sampling_scope": "contrastive_only; epoch_frame; fresh_per_minibatch",
#             "historical_negative_generator_seed": int(self.cfg("seed", 0)) + phase_id,
#             "historical_classifier_supervision": "unchanged_directional_moment_risk",
#             "generates_native_spectra": False,
#             "uses_persistent_whole_cds_support": True,
#             "feasible_core_objective_active": False,
#             "support_reconstruction_active": bool(recon_epochs > 0),
#             "uses_real_current_positive_support": True,
#             "uses_covariance_or_gaussian_memory": True,
#             "uses_gaussian_density_or_sampling": historical_negative_mode == "distribution",
#             "uses_mean_or_second_moment_classifier_memory": True,
#             "uses_second_classifier": False,
#             "uses_phase_bias": False,
#             "uses_posthoc_classifier_reconstruction": bool(recon_epochs > 0),
#             "final_synchronized_head_steps_accepted": final_head_accepted,
#             "selected_support_head_epoch": selected_epoch,
#             "reconstruction_selection_source": selection_source,
#             "reconstruction_selected_objective": selected_score,
#             "reconstruction_initial_objective": initial_score,
#             "reconstruction_initial_new_ce": initial_new_ce,
#             "reconstruction_new_ce_guard_active": False,
#             "historical_support_reconstruction": history_reconstruction,
#             "uses_balanced_old_new_positive_classifier_support": True,
#             "uses_signed_whole_cds_distribution": True,
#             "historical_support_audit": self.historical_support.diagnostics(),
#             "uses_per_epoch_direct_historical_coordinate_refresh": historical_transport,
#             "historical_frame_source": historical_frame_source,
#             "historical_frame_fit_scope": "current_train_pairs_at_epoch_start" if not recon_epochs else "current_train_pairs_after_representation",
#             "historical_frame_diagnostics_scope": "fixed_frame_final_current_train_pairs",
#             "contrastive_historical_carrier": {
#                 "classifier": "detached_deployed_cosine_rows",
#                 "mean": "detached_normalized_historical_means",
#                 "distribution": "detached_normalized_compact_gaussian_draws",
#             }[historical_negative_mode],
#             "historical_negative_support_gate": False,
#             "real_ce_first_batch_row_gradient_max": {
#                 "old": real_ce_old, "new": real_ce_new, "future": real_ce_future,
#             },
#             "classifier_audit_selects_weights": False,
#             "historical_risk_mode": self.historical_support.risk_mode,
#             "representation_ce_scope": representation_ce_scope,
#             "representation_train_new_head": representation_train_new_head,
#             "joint_seen_cosine_head_during_representation": representation_train_new_head,
#             "historical_cds_head_risk_in_representation": representation_train_new_head,
#             "risk_approximation_audit": risk_audit,
#             "competition_splits": {
#                 "old": "reporting", "new": "reporting",
#                 "phase_start_and_post_rep_new": "train",
#             },
#             "parameter_gradient_audit": parameter_gradient_audit,
#             "parameter_gradient_audits": parameter_gradient_audits,
#             "representation_incoming_direction_change": self._max_direction_change(
#                 admission_new_weight, new_weight_after_training
#             ),
#             "head_admission": admission_state,
#             "historical_transport_enabled": historical_transport,
#             "historical_feasible_radius_mean": float(
#                 final_feasible_state.mean_core_radius
#             ),
#             "historical_current_intrusion_fraction": float(
#                 final_feasible_state.historical_intrusion_fraction
#             ),
#             "historical_core_violation": float(
#                 final_feasible_state.effective_core_violation
#             ),
#             "historical_spectral_interval_overlap": float(
#                 final_feasible_state.physical_interval_overlap
#             ),
#             "historical_vq_directional_confidence": float(
#                 final_feasible_state.ordered_variation_confidence
#             ),
#             "representation_gradient_conflict_fraction": float(
#                 final_record["representation_gradient_conflict_fraction"]
#             ),
#             "representation_gradient_cosine": float(
#                 final_record["representation_gradient_cosine"]
#             ),
#             "uses_feature_kd": feature_kd_weight > 0,
#             "new_class_before_feature_mean_norm": reference_feature_norm,
#             "new_class_before_after_mean_cosine": paired_new_feature_cosine,
#             "new_class_before_after_chord_rms": paired_new_feature_chord,
#             "new_class_before_after_structure": classwise_new_migration,
#             "historical_cds_estimated_transport": classwise_old_transport,
#             "final_teacher_historical_relevance_mean": float(
#                 final_record["teacher_historical_relevance_mean"]
#             ),
#             "uses_logit_kd": bool(kd_logit_weight > 0),
#             "prediction_rule": "tau*cosine(Q,W)",
#             "representation_epochs": epoch_count,
#             "selected_representation_epoch": selected_representation_epoch,
#             "representation_selection_source": "final_epoch_no_selection",
#             "selected_representation_objective": selected_representation_score,
#             "representation_lr": learning_rate,
#             "contrastive_weight": contrastive_weight,
#             "feature_kd_weight": feature_kd_weight,
#             "kd_logit_weight": kd_logit_weight,
#             "kd_temperature": kd_temperature,
#             "cds_weight": cds_weight,
#             "final_joint_classification_loss": float(
#                 final_record["classification_loss"]
#             ),
#             "final_joint_historical_risk_loss": float(
#                 final_record["joint_historical_risk_loss"]
#             ),
#             "final_real_current_classification_loss": float(
#                 final_record["real_current_classification_loss"]
#             ),
#             "final_feature_kd_loss": float(final_record["feature_kd_loss"]),
#             "final_cds_separation_loss": float(final_record["cds_separation_loss"]),
#             "final_cds_separation_margin": float(final_record["cds_separation_margin"]),
#             "final_cds_separation_violation_fraction": float(
#                 final_record["cds_separation_violation_fraction"]
#             ),
#             "final_cds_preservation_loss": float(final_record["cds_preservation_loss"]),
#             "final_cds_intra_loss": float(final_record["cds_intra_loss"]),
#             "final_cds_inter_loss": float(final_record["cds_inter_loss"]),
#             "final_physical_absolute_support_mean": float(
#                 final_record["physical_absolute_support_mean"]
#             ),
#             "final_physical_relation_support_mean": float(
#                 final_record["physical_relation_support_mean"]
#             ),
#             "final_physical_relevance_mean": float(
#                 final_record["physical_relevance_mean"]
#             ),
#             "phase_transport_accepted_on_new_class_holdout": bool(
#                 final_alignment.transport_accepted.item()
#             ),
#             "phase_holdout_affine_error": float(
#                 final_alignment.holdout_candidate_error.item()
#             ),
#             "phase_holdout_identity_error": float(
#                 final_alignment.holdout_identity_error.item()
#             ),
#             "phase_mapping_error": float(final_alignment.mapping_error.item()),
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
#             "final_direct_cds_classifier_loss": float(
#                 final_classifier_state.balanced_loss.detach().item()
#             ),
#             "final_direct_cds_historical_loss": float(
#                 final_classifier_state.historical_positive_loss.detach().item()
#             ),
#             "final_direct_cds_current_loss": float(
#                 final_classifier_state.current_positive_loss.detach().item()
#             ),
#             "final_whole_cds_old_new_violation_fraction": float(
#                 final_classifier_objective.old_new_violation_fraction.detach().item()
#             ),
#             "classifier_learned_pair_risk_mean": float(
#                 final_classifier_state.learned_pair_risk_mean.detach().item()
#             ),
#             "classifier_physical_pair_risk_mean": float(
#                 final_classifier_state.physical_pair_risk_mean.detach().item()
#             ),
#             "classifier_effective_pair_risk_mean": float(
#                 final_classifier_state.effective_pair_risk_mean.detach().item()
#             ),
#             "historical_classifier_training_max_change": self._max_parameter_change(
#                 old_weight_before, final_old_weight
#             ),
#             "historical_classifier_training_direction_change": self._max_direction_change(
#                 old_weight_before, final_old_weight
#             ),
#             "incoming_classifier_training_max_change": self._max_parameter_change(
#                 new_weight_phase_start, final_new_weight
#             ),
#             "incoming_classifier_training_direction_change": self._max_direction_change(
#                 new_weight_phase_start, final_new_weight
#             ),
#             "phase_start_old_ba": float(old_before["balanced_accuracy"]),
#             "phase_start_old_all_seen_ba": None,  # No true all-seen phase-start measurement.
#             "phase_start_new_only_ba": float(phase_start_new_only["balanced_accuracy"]),
#             "phase_start_new_all_seen_ba": float(
#                 phase_start_new_all_seen["balanced_accuracy"]
#             ),
#             "post_rep_old_only_ba": float(post_rep_old_only["balanced_accuracy"]),
#             "post_rep_old_all_seen_ba": float(post_rep_old_all["balanced_accuracy"]),
#             "post_rep_new_only_ba": float(post_rep_new_only["balanced_accuracy"]),
#             "post_rep_new_all_seen_ba": float(post_rep_new_all["balanced_accuracy"]),
#             "final_old_only_ba": float(final_old_only["balanced_accuracy"]),
#             "final_old_all_seen_ba": float(final_old_all["balanced_accuracy"]),
#             "final_new_only_ba": float(final_new_only["balanced_accuracy"]),
#             "final_new_all_seen_ba": float(final_new_all["balanced_accuracy"]),
#             "classifier_logit_scale_frozen_incremental": True,
#             "historical_support_reconstruction": history_reconstruction,
#             "historical_support_classes": list(self.historical_support.class_ids),
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



# from __future__ import annotations
# from copy import deepcopy
# import math
# import sys
# import warnings
# from numbers import Integral, Real
# from typing import Any, Dict
# import torch
# from tqdm import tqdm

# from losses.loss import (
#     classification_objective,
#     cross_phase_cds_separation_loss,
#     whole_cds_contrastive_loss,
#     historical_cds_preservation_loss,
#     logit_distillation_loss,
#     ordered_feature_distillation_loss,
# )

# from models.classifier import CosineClassificationHead
# from models.cds_decision_region import (
#     whole_cds_class_balanced_classifier_loss,
#     whole_cds_joint_classifier_state,
#     feasible_directional_head_objective,
#     current_class_feasible_radii,
#     resolve_classifier_gradient_conflict,
# )

# from models.geometry_bank import (
#     ClassStructureBatch,
#     merge_class_structure_batches,
#     whole_cds_pair_risk,
# )
# from models.phase_coordinate_alignment import correct_historical_structure
# from models.historical_cds_support import HistoricalCDSSupportBank
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
#     """Clone the complete committed class structure without altering Q or P."""
#     if not isinstance(structure, ClassStructureBatch) or structure.class_count <= 0:
#         raise ValueError("cannot clone an empty class structure")
#     from dataclasses import fields

#     return ClassStructureBatch(
#         **{
#             field.name: getattr(structure, field.name).detach().clone()
#             for field in fields(ClassStructureBatch)
#         }
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

#     @staticmethod
#     def parameter_gradient_audit(objectives, backbone_parameters):
#         """Detached parameter-space gradient norms, without optimizer mutation.

#         Q-space gradient cosines alone do not characterize backbone parameter
#         interference. This audit measures the real trainable parameter groups.
#         """
#         result = {}
#         for name, objective in objectives.items():
#             if not objective.requires_grad:
#                 result[name] = {"grad_norm": 0.0, "active_parameters": 0}
#                 continue
#             grads = torch.autograd.grad(
#                 objective,
#                 backbone_parameters,
#                 retain_graph=True,
#                 allow_unused=True,
#             )
#             norms = [
#                 grad.detach().float().square().sum()
#                 for grad in grads
#                 if grad is not None
#             ]
#             total = torch.stack(norms).sum().sqrt().item() if norms else 0.0
#             result[name] = {
#                 "grad_norm": float(total),
#                 "active_parameters": sum(g is not None for g in grads),
#             }
#         return result

#     @torch.no_grad()
#     def audit_current_classifier_risk(self, loader, *, class_ids, cds):
#         """Report moment approximation error on available current VAL only.

#         Do not use old test examples to tune historical risk or certify drift.
#         """
#         if loader is None:
#             return {"status": "skipped_no_validation", "used_for_selection": False}
#         observed = self.collect_ordered_features(loader)
#         return self.historical_support.audit_expected_ce(
#             self._require_classifier(),
#             class_ids,
#             cds,
#             observed["ordered_features"],
#             observed["labels"],
#         )

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

#             raise RuntimeError(
#                 "seen classes must equal old classes followed by new classes"
#             )

#         if tuple(self.model.committed_class_ids) != old_class_ids:

#             raise RuntimeError("committed historical CDS does not match old classes")

#         return old_class_ids, new_class_ids, seen_class_ids

#     @staticmethod
#     def _assert_frozen_reference(reference_backbone: torch.nn.Module) -> None:

#         reference_backbone.eval()

#         if any(
#             parameter.requires_grad for parameter in reference_backbone.parameters()
#         ):
#             raise RuntimeError("previous backbone reference must remain frozen")
#         if any(
#             parameter.grad is not None for parameter in reference_backbone.parameters()
#         ):
#             raise RuntimeError("previous backbone reference must not carry gradients")

#     @staticmethod
#     def _assert_frozen_reference_classifier(
#         reference_classifier: CosineClassificationHead,
#     ) -> None:

#         reference_classifier.eval()

#         if any(
#             parameter.requires_grad for parameter in reference_classifier.parameters()
#         ):

#             raise RuntimeError("previous classifier reference must remain frozen")

#         if any(
#             parameter.grad is not None
#             for parameter in reference_classifier.parameters()
#         ):

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

#             raise RuntimeError(
#                 f"previous/current {name} state keys differ at phase start"
#             )

#         for key in ref_state:

#             if not torch.equal(
#                 ref_state[key].detach().cpu(), cur_state[key].detach().cpu()
#             ):

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

#             observed = set(
#                 int(value) for value in labels.unique().detach().cpu().tolist()
#             )

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

#             raise RuntimeError(
#                 "incremental TRAIN split is empty or misses a current class"
#             )

#         class_count = len(new_class_ids)

#         weights = torch.tensor(
#             [
#                 total / float(class_count * counts[class_id])
#                 for class_id in new_class_ids
#             ],
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
#     def _max_direction_change(before: Tensor, after: Tensor) -> float:

#         if before.shape != after.shape:

#             raise ValueError("parameter snapshots have different shapes")

#         if before.numel() == 0:

#             return 0.0

#         eps = torch.finfo(before.dtype).eps

#         before_unit = torch.nn.functional.normalize(before, p=2, dim=1, eps=eps)

#         after_unit = torch.nn.functional.normalize(after, p=2, dim=1, eps=eps)

#         return float((after_unit - before_unit).abs().max().item())

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

#     def _run_new_class_head_admission(
#         self,
#         *,
#         classifier: CosineClassificationHead,
#         train_loader: Any,
#         train_eval_loader: Any,
#         old_class_ids: tuple[int, ...],
#         new_class_ids: tuple[int, ...],
#         seen_class_ids: tuple[int, ...],
#         current_class_weights: Tensor,
#         epochs: int,
#         learning_rate: float,
#         gradient_clip: float,
#     ) -> dict[str, float | int]:
#         """Train only incoming rows against ALL seen rows with real current HSI.

#         Backbone Q, old W, future W and tau remain frozen. No old images,
#         feature replay or pseudo-samples enter this stage. The caller must
#         invoke it before the joint incremental optimizer is created.
#         """
#         if epochs < 0:
#             raise ValueError("head admission epochs must be nonnegative")
#         if epochs == 0:
#             return {"epochs": 0, "steps": 0}
#         if not math.isfinite(learning_rate) or learning_rate <= 0:
#             raise ValueError("head admission learning rate must be positive")

#         old_ids = torch.tensor(old_class_ids, dtype=torch.long, device=self.device)
#         new_ids = torch.tensor(new_class_ids, dtype=torch.long, device=self.device)
#         seen_ids = torch.tensor(seen_class_ids, dtype=torch.long, device=self.device)
#         future_ids = torch.tensor(
#             sorted(set(range(classifier.total_class_count)) - set(seen_class_ids)),
#             dtype=torch.long,
#             device=self.device,
#         )
#         old_snapshot = classifier.class_weight(old_ids).detach().clone()
#         future_snapshot = (
#             classifier.class_weight(future_ids).detach().clone()
#             if future_ids.numel()
#             else None
#         )
#         tau_snapshot = classifier.logit_scale_parameter.detach().clone()
#         backbone_snapshot = {
#             k: v.detach().clone() for k, v in self.model.backbone.state_dict().items()
#         }

#         @torch.no_grad()
#         def margin_diagnostics() -> dict[str, float]:
#             """Observe TRUE current-class logit relative to strongest old logit."""
#             self.model.eval()
#             correct_margin_sum = 0.0
#             loss_sum = 0.0
#             count = 0
#             for batch in train_eval_loader:
#                 patch, spectrum, labels = self.unpack_batch(batch)
#                 if not bool(torch.isin(labels, new_ids).all()):
#                     raise RuntimeError(
#                         "admission diagnostics include old/future labels"
#                     )
#                 features = self.model.extract_features(
#                     patch, center_spectrum=spectrum, return_aux=False
#                 ).ordered_features
#                 scores = classifier(
#                     features,
#                     class_ids=seen_ids,
#                     trainable_class_ids=(),
#                 ).logits
#                 local = _targets_local(labels, seen_ids)
#                 true_scores = scores.gather(1, local[:, None]).squeeze(1)
#                 rival_old = scores[:, : len(old_class_ids)].max(dim=1).values
#                 correct_margin_sum += float((true_scores - rival_old).sum().item())
#                 loss_sum += float(
#                     torch.nn.functional.cross_entropy(
#                         scores, local, reduction="sum"
#                     ).item()
#                 )
#                 count += int(labels.numel())
#             return {
#                 "new_vs_old_logit_margin": correct_margin_sum / max(1, count),
#                 "new_all_seen_ce": loss_sum / max(1, count),
#             }

#         before = margin_diagnostics()
#         # The previous teacher is deliberately not needed during head admission:
#         # incoming features come from the fixed phase-start backbone.
#         self.model.eval()
#         optimizer = torch.optim.AdamW(
#             [classifier.weight], lr=learning_rate, weight_decay=0.0
#         )
#         total_loss, total_correct, total_examples, steps = 0.0, 0, 0, 0
#         for _ in range(epochs):
#             for batch in train_loader:
#                 patch, spectrum, labels = self.unpack_batch(batch)
#                 if not bool(torch.isin(labels, new_ids).all()):
#                     raise RuntimeError(
#                         "head admission saw a non-current training class"
#                     )
#                 optimizer.zero_grad(set_to_none=True)
#                 with torch.no_grad():
#                     features = self.model.extract_features(
#                         patch, center_spectrum=spectrum, return_aux=False
#                     ).ordered_features.detach()
#                 logits = classifier(
#                     features,
#                     class_ids=seen_ids,
#                     trainable_class_ids=new_ids,
#                 ).logits
#                 local = _targets_local(labels, seen_ids)
#                 new_local = _targets_local(labels, new_ids)
#                 weights = current_class_weights.index_select(0, new_local)
#                 per_sample = torch.nn.functional.cross_entropy(
#                     logits, local, reduction="none"
#                 )
#                 loss = (per_sample * weights).mean()
#                 if not bool(torch.isfinite(loss)):
#                     raise RuntimeError("nonfinite new-class admission loss")
#                 loss.backward()
#                 gradient = classifier.weight.grad
#                 if gradient is None or not bool(torch.isfinite(gradient).all()):
#                     raise RuntimeError("nonfinite or absent admission gradient")
#                 if self._row_grad_max(gradient, old_ids) != 0.0 or (
#                     future_ids.numel()
#                     and self._row_grad_max(gradient, future_ids) != 0.0
#                 ):
#                     raise RuntimeError("head admission must update incoming rows only")
#                 if gradient_clip > 0:
#                     torch.nn.utils.clip_grad_norm_([classifier.weight], gradient_clip)
#                 optimizer.step()
#                 steps += 1
#                 n = int(labels.numel())
#                 total_examples += n
#                 total_loss += float(loss.detach().item()) * n
#                 total_correct += int(logits.argmax(dim=1).eq(local).sum().item())
#         classifier.weight.grad = None
#         if not torch.equal(classifier.class_weight(old_ids).detach(), old_snapshot):
#             raise RuntimeError("head admission modified a historical classifier row")
#         if future_snapshot is not None and not torch.equal(
#             classifier.class_weight(future_ids).detach(), future_snapshot
#         ):
#             raise RuntimeError("head admission modified a future classifier row")
#         if not torch.equal(classifier.logit_scale_parameter.detach(), tau_snapshot):
#             raise RuntimeError("head admission modified the shared cosine temperature")
#         if any(
#             not torch.equal(value, backbone_snapshot[key])
#             for key, value in self.model.backbone.state_dict().items()
#         ):
#             raise RuntimeError("head admission modified the feature backbone")

#         after = self.evaluate_loader(
#             train_eval_loader,
#             class_ids=seen_class_ids,
#             target_class_ids=new_class_ids,
#         )
#         after_margin = margin_diagnostics()
#         result = {
#             "new_vs_old_logit_margin_before": before["new_vs_old_logit_margin"],
#             "new_vs_old_logit_margin_after": after_margin["new_vs_old_logit_margin"],
#             "new_all_seen_ce_before": before["new_all_seen_ce"],
#             "new_all_seen_ce_after": after_margin["new_all_seen_ce"],
#             "epochs": epochs,
#             "steps": steps,
#             "train_loss": total_loss / max(total_examples, 1),
#             "train_all_seen_accuracy": total_correct / max(total_examples, 1),
#             "new_all_seen_ba_after": float(after["balanced_accuracy"]),
#         }
#         print(
#             f"[New-class head admission] epochs={epochs} steps={steps} "
#             f"all-seen-new-BA={100*result['new_all_seen_ba_after']:.2f}% "
#             f"new-vs-old margin={before['new_vs_old_logit_margin']:.4f}"
#             f" -> {after_margin['new_vs_old_logit_margin']:.4f} "
#             f"CE={before['new_all_seen_ce']:.4f} -> {after_margin['new_all_seen_ce']:.4f} "
#             "| only W_new updated; Q/W_old/W_future/tau unchanged",
#             file=sys.stdout,
#         )
#         return result

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

#         kd_logit_weight = _finite_nonnegative(
#             "kd_logit_weight", self.cfg("kd_logit_weight", 0.0)
#         )

#         kd_temperature = _finite_positive(
#             "kd_temperature", self.cfg("kd_temperature", 2.0)
#         )

#         cds_weight = _finite_nonnegative("cds_weight", self.cfg("cds_weight", 1.0))

#         # Zero is an explicit preservation ablation, consistent with the CLI.
#         feature_kd_weight = _finite_nonnegative(
#             "feature_kd_weight", self.cfg("feature_kd_weight", 10.0)
#         )
#         representation_ce_scope = self.cfg("representation_ce_scope", "all_seen")
#         if representation_ce_scope not in ("all_seen", "new_only"):
#             raise ValueError("representation_ce_scope must be all_seen or new_only")
#         configured_joint = self.cfg("representation_train_new_head", True)
#         if not isinstance(configured_joint, bool):
#             raise ValueError("representation_train_new_head must be boolean")
#         # The unified method requires coordinated Q and W training. Old CLI
#         # defaults supported a frozen-W ablation: warn, but enforce the contract.
#         if not configured_joint:
#             warnings.warn(
#                 "Unified CDS requires joint Q/W training: overriding the legacy "
#                 "representation_train_new_head=false setting",
#                 RuntimeWarning,
#             )
#         representation_train_new_head = True
#         if representation_ce_scope != "all_seen":
#             warnings.warn(
#                 "Unified CDS uses all-seen real CE: overriding new_only CE ablation",
#                 RuntimeWarning,
#             )
#             representation_ce_scope = "all_seen"

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

#         self.dataset.start_phase(phase_id)

#         old_class_ids, new_class_ids, seen_class_ids = self._phase_classes(phase_id)

#         old_class_tensor = torch.tensor(
#             old_class_ids, device=self.device, dtype=torch.long
#         )

#         new_class_tensor = torch.tensor(
#             new_class_ids, device=self.device, dtype=torch.long
#         )

#         seen_class_tensor = torch.tensor(
#             seen_class_ids, device=self.device, dtype=torch.long
#         )

#         self.historical_support.require(old_class_ids)

#         historical_structure = _clone_structure(
#             self.model.structure_bank.class_structures(old_class_ids)
#         )
#         historical_moments = self.model.structure_bank.class_moments(old_class_ids)
#         historical_pair_risk = whole_cds_pair_risk(
#             self.model.class_discriminative_structure(old_class_ids)
#         )[0].detach()

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
#             sorted(
#                 int(value)
#                 for value in reference_labels.unique().detach().cpu().tolist()
#             )
#         )

#         if observed_new != tuple(sorted(new_class_ids)):

#             raise RuntimeError(
#                 "incremental TRAIN rows do not match current phase classes"
#             )

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
#         # The before-statistics are measured once in the PREVIOUS model space,
#         # exactly as required by the paired incoming-feature observation protocol. They are
#         # not old-class samples, and are not saved across phases.
#         reference_feature_norm = float(
#             reference_features_full.reshape(reference_features_full.size(0), -1)
#             .norm(dim=1)
#             .mean()
#             .item()
#         )

#         old_weight_before = classifier.class_weight(old_class_tensor).detach().clone()

#         new_weight_phase_start = (
#             classifier.class_weight(new_class_tensor).detach().clone()
#         )

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

#         print(
#             f"[Phase {phase_id} start] "
#             f"NewBA(new-only)={100*float(phase_start_new_only['balanced_accuracy']):.2f}% "
#             f"NewBA(all-seen)={100*float(phase_start_new_all_seen['balanced_accuracy']):.2f}% | "
#             "Two-stage: Q/FeatureKD/Whole-CDS support contrast then shared-head support reconstruction; no replay",
#             file=sys.stdout,
#         )

#         @torch.no_grad()
#         def collect_current_state():

#             self.model.eval()

#             rows = self.collect_ordered_features(train_eval_loader)

#             if not torch.equal(rows["labels"], reference_labels):
#                 raise RuntimeError("current TRAIN label order changed")

#             if not torch.equal(rows["raw_center_spectra"], reference_spectra):
#                 raise RuntimeError("current TRAIN spectra changed")

#             current_features_full = rows["ordered_features"].detach()

#             current_structure = self.model.estimate_class_structure(
#                 current_features_full,
#                 None,
#                 reference_labels,
#                 class_ids=new_class_ids,
#                 physical_structure=current_physical_structure,
#             )
#             return current_features_full, current_structure

#         # Prototype initialization on CURRENT TRAIN features only. No old
#         # exemplars, extra head, or trainable module are introduced.
#         # The cosine head compares unit directions, so an incoming class's
#         # observed direction is a better starting point than a random row.
#         with torch.no_grad():
#             incoming_means = current_reference_structure.learned_identity.flatten(1)
#             mean_norms = incoming_means.norm(p=2, dim=1)
#             if not bool((mean_norms > torch.finfo(incoming_means.dtype).eps).all()):
#                 raise RuntimeError("incoming class has a degenerate mean direction")
#             new_prototypes = torch.nn.functional.normalize(
#                 incoming_means, p=2, dim=1
#             ).to(device=classifier.weight.device, dtype=classifier.weight.dtype)
#             if new_prototypes.shape != (
#                 len(new_class_ids),
#                 classifier.feature_dim,
#             ) or not bool(torch.isfinite(new_prototypes).all()):
#                 raise RuntimeError("invalid incoming class prototypes")
#             classifier.weight.index_copy_(0, new_class_tensor, new_prototypes)

#         # Diagnostic classifier admission: learn W_new against all historical
#         # rows before the more restrictive historical robust loss is activated.
#         # This addresses zero new all-seen accuracy without modifying Q or W_old.
#         admission_epochs = int(self.cfg("head_admission_epochs", 0))
#         if admission_epochs < 0:
#             raise ValueError("head_admission_epochs must be nonnegative")
#         admission_lr = _finite_positive(
#             "head_admission_lr", self.cfg("head_admission_lr", 1e-3)
#         )
#         admission_state = self._run_new_class_head_admission(
#             classifier=classifier,
#             train_loader=train_loader,
#             train_eval_loader=train_eval_loader,
#             old_class_ids=old_class_ids,
#             new_class_ids=new_class_ids,
#             seen_class_ids=seen_class_ids,
#             current_class_weights=current_class_weights,
#             epochs=admission_epochs,
#             learning_rate=admission_lr,
#             gradient_clip=gradient_clip,
#         )
#         admission_new_weight = (
#             classifier.class_weight(new_class_tensor).detach().clone()
#         )
#         # Empirically supported moment translations are attempted after, not
#         # during, representation adaptation. Unsupported classes stay unchanged.
#         historical_transport = bool(self.cfg("historical_transport", True))

#         # Stage 1: COORDINATED Q + one-head W training. The current-data CE
#         # updates both Q and active seen W, while the EXISTING historical CDS
#         # directional historical CDS objective simultaneously protects old rows.
#         # Whole-CDS inter/intra/native-spectrum preservation and contrast train Q.
#         # No extra head, old sample, generation, or additional KD module.

#         representation_parameters = list(backbone_parameters)
#         optimizer_groups = [
#             {"params": backbone_parameters, "weight_decay": weight_decay}
#         ]
#         if representation_train_new_head:
#             # Gradient masking alone cannot protect inactive rows from AdamW
#             # decay. Start a fresh optimizer and disable decay for the head.
#             optimizer_groups.append(
#                 {"params": [classifier.weight], "weight_decay": 0.0}
#             )
#             representation_parameters.append(classifier.weight)
#         optimizer = torch.optim.AdamW(optimizer_groups, lr=learning_rate)

#         scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
#             optimizer, T_max=epoch_count, eta_min=0.0
#         )

#         # Explicit final-epoch contract. All scheduled optimizer steps are kept.
#         # Validation remains read-only reporting and never restores prior Q/W.
#         selected_representation_epoch = epoch_count  # compatibility metadata only
#         selected_representation_score = None

#         history: list[dict[str, Any]] = []

#         gradient_contract_checked = False
#         parameter_gradient_audit = {}
#         parameter_gradient_audits = []

#         progress = tqdm(
#             range(epoch_count),
#             desc=f"Phase {phase_id} representation training",
#             unit="epoch",
#             dynamic_ncols=True,
#             file=sys.stdout,
#         )

#         for epoch in progress:
#             self.model.eval()
#             epoch_features_full, epoch_current_structure = collect_current_state()
#             epoch_historical_structure, epoch_transform, _epoch_alignment = (
#                 correct_historical_structure(
#                     historical_structure=historical_structure,
#                     current_reference_structure=current_reference_structure,
#                     current_structure=epoch_current_structure,
#                     reference_features=reference_features_full,
#                     current_features=epoch_features_full,
#                     use_transport=False,
#                 )
#             )
#             epoch_seen_structure = merge_class_structure_batches(
#                 (epoch_historical_structure, epoch_current_structure)
#             )
#             epoch_seen_cds = self.model.structure_bank.build_cds(epoch_seen_structure)
#             epoch_pair_risk = whole_cds_pair_risk(epoch_seen_cds)[0].detach()
#             epoch_radii, epoch_intrusion = current_class_feasible_radii(
#                 current_features=epoch_features_full,
#                 current_labels=reference_labels,
#                 cds=epoch_seen_cds,
#                 historical_class_count=len(old_class_ids),
#             )

#             self.model.train()

#             sums = {
#                 name: 0.0
#                 for name in (
#                     "classification",
#                     "real_classification",
#                     "direct_head_classification",
#                     "historical_cds_head_loss",
#                     "joint_historical_risk",
#                     "current_real_head_loss",
#                     "old_all_seen_accuracy",
#                     "current_new_only_accuracy",
#                     "feature_kd",
#                     "teacher_hist_relevance",
#                     "feature_rms",
#                     "contrastive",
#                     "diagnostic_sep_loss",
#                     "kd",
#                     "cds",
#                     "intra",
#                     "inter",
#                     "physical_abs",
#                     "physical_rel",
#                     "physical_joint",
#                     "sep_margin",
#                     "sep_violation",
#                     "sep_positive",
#                     "sep_current_negative",
#                     "sep_historical_negative",
#                     "sep_valid",
#                     "sep_physical_confusability",
#                     "sep_pair_risk",
#                     "total",
#                     "core_radius",
#                     "core_violation",
#                     "historical_intrusion",
#                     "spectral_interval_overlap",
#                     "variation_confidence",
#                     "representation_gradient_conflict",
#                     "representation_gradient_cosine",
#                 )
#             }

#             sample_count = 0

#             correct_new = 0

#             for batch in train_loader:

#                 patch, spectrum, labels = self.unpack_batch(batch)

#                 if not bool(torch.isin(labels, new_class_tensor).all()):

#                     raise RuntimeError(
#                         "incremental TRAIN batch contains an old or future class"
#                     )

#                 optimizer.zero_grad(set_to_none=True)

#                 classifier.logit_scale_parameter.grad = None

#                 with torch.no_grad():

#                     reference_features = self.model.extract_features_with_backbone(
#                         reference_backbone,
#                         patch,
#                         center_spectrum=spectrum,
#                         return_aux=False,
#                     ).ordered_features.detach()

#                     physical_evidence = self.model.physical_spectral_evidence(
#                         spectrum
#                     ).detach()

#                     teacher_old_logits = (
#                         reference_classifier(
#                             reference_features,
#                             class_ids=old_class_tensor,
#                             trainable_class_ids=(),
#                         ).logits.detach()
#                         if kd_logit_weight > 0
#                         else None
#                     )

#                 current_features = self.model.extract_features(
#                     patch, center_spectrum=spectrum, return_aux=False
#                 ).ordered_features

#                 current_local = _targets_local(labels, new_class_tensor)

#                 sample_weights = current_class_weights.index_select(0, current_local)

#                 # All-seen output remains available for competition diagnostics.
#                 # Only the explicitly selected representation CE may train W_new.
#                 current_output = classifier(
#                     current_features,
#                     class_ids=seen_class_tensor,
#                     trainable_class_ids=(),
#                 )
#                 representation_output = classifier(
#                     current_features,
#                     class_ids=(
#                         new_class_tensor
#                         if representation_ce_scope == "new_only"
#                         else seen_class_tensor
#                     ),
#                     trainable_class_ids=(
#                         new_class_tensor if representation_train_new_head else ()
#                     ),
#                 )

#                 real_classification = classification_objective(
#                     logits=representation_output.logits,
#                     class_ids=representation_output.class_ids,
#                     labels_global=labels,
#                     sample_weights=sample_weights,
#                     unbiased_sample_weighting=True,
#                 )

#                 # Direct historical CDS boundary evidence and real current labels
#                 # update the deployed all-seen head. Q is detached on this branch;
#                 # no support examples are synthesized or retained.
#                 # Legacy feasible-core calculation is diagnostic ONLY; no
#                 # gradients or model updates use this surrogate.
#                 with torch.no_grad():
#                     feasible_state = feasible_directional_head_objective(
#                         classifier=classifier,
#                         current_features=current_features.detach(),
#                         current_labels=labels,
#                         seen_cds=epoch_seen_cds,
#                         historical_class_count=len(old_class_ids),
#                         feasible_radii=epoch_radii,
#                         intrusion_fraction=epoch_intrusion,
#                         current_sample_weights=sample_weights,
#                     )
#                 support_classification = feasible_state.state
#                 # Real CE acts on the deployed all-seen decision boundary. The
#                 # same historical CDS risk is optimized in this SAME step when
#                 # the seen head is jointly trainable, instead of arriving only
#                 # after Q has stopped moving.
#                 if representation_train_new_head:
#                     joint_old_risk = self.historical_support.historical_risk(
#                         classifier,
#                         old_class_ids,
#                         epoch_seen_cds,
#                         class_pair_risk=epoch_pair_risk,
#                     )
#                     classification_loss = (
#                         len(new_class_ids) * real_classification.loss
#                         + len(old_class_ids) * joint_old_risk.loss
#                     ) / len(seen_class_ids)
#                 else:
#                     joint_old_risk = None
#                     classification_loss = real_classification.loss

#                 # The frozen teacher is
#                 # evaluated on CURRENT samples. Historical CDS compatibility is
#                 # computed with the teacher's OLD-CLASS bank, so incoming pixels
#                 # that resemble vulnerable historical classes receive stronger
#                 # feature stability pressure, without any old sample replay.
#                 cds = historical_cds_preservation_loss(
#                     reference_features=reference_features,
#                     current_features=current_features,
#                     physical_spectral_evidence=physical_evidence,
#                     reference_historical_structure=historical_structure,
#                     current_historical_structure=epoch_historical_structure,
#                     reference_moments=historical_moments,
#                     current_moments=historical_moments,
#                     historical_pair_risk=historical_pair_risk,
#                 )
#                 feature_kd = ordered_feature_distillation_loss(
#                     reference_features=reference_features,
#                     current_features=current_features,
#                     sample_weights=sample_weights,
#                     historical_relevance=cds.teacher_historical_relevance,
#                 )

#                 if kd_logit_weight > 0:

#                     if teacher_old_logits is None:

#                         raise RuntimeError(
#                             "teacher old logits are missing while logit KD is enabled"
#                         )

#                     student_old_logits = classifier(
#                         current_features,
#                         class_ids=old_class_tensor,
#                         trainable_class_ids=(),
#                     ).logits

#                     kd_loss = logit_distillation_loss(
#                         student_old_logits=student_old_logits,
#                         teacher_old_logits=teacher_old_logits,
#                         temperature=kd_temperature,
#                         sample_weights=sample_weights,
#                     ).loss

#                 else:

#                     kd_loss = current_features.sum() * 0.0

#                 separation = cross_phase_cds_separation_loss(
#                     current_cosine_scores=current_output.cosine_scores,
#                     class_ids=current_output.class_ids,
#                     labels_global=labels,
#                     physical_spectral_evidence=physical_evidence,
#                     seen_cds=epoch_seen_cds,
#                     historical_class_count=len(old_class_ids),
#                     sample_weights=sample_weights,
#                 )
#                 # Active FCS-style structural contrast: old support is used as
#                 # cosine-space negatives; the same support supervises the head.
#                 contrastive = whole_cds_contrastive_loss(
#                     current_features=current_features,
#                     labels_global=labels,
#                     physical_spectral_evidence=physical_evidence,
#                     seen_cds=epoch_seen_cds,
#                     historical_class_count=len(old_class_ids),
#                     logit_scale=classifier.logit_scale,
#                     sample_weights=sample_weights,
#                     historical_support=self.historical_support,
#                     class_pair_risk=epoch_pair_risk,
#                 )

#                 if not gradient_contract_checked:

#                     tolerance = 64.0 * torch.finfo(classifier.weight.dtype).eps

#                     real_ce_grad = torch.autograd.grad(
#                         real_classification.loss,
#                         classifier.weight,
#                         retain_graph=True,
#                         allow_unused=True,
#                     )[0]
#                     real_ce_old = self._row_grad_max(real_ce_grad, old_class_tensor)
#                     real_ce_new = self._row_grad_max(real_ce_grad, new_class_tensor)
#                     real_ce_future = self._row_grad_max(
#                         real_ce_grad, future_class_tensor
#                     )
#                     if (
#                         real_ce_future > tolerance
#                         or real_ce_old > tolerance
#                         or (representation_train_new_head and real_ce_new <= tolerance)
#                     ):
#                         raise RuntimeError(
#                             "real new-class CE must train Q and W_new only; W_old/W_future are reference competitors"
#                         )
#                     if representation_train_new_head:
#                         risk_grad = torch.autograd.grad(
#                             joint_old_risk.loss,
#                             classifier.weight,
#                             retain_graph=True,
#                             allow_unused=True,
#                         )[0]
#                         if (
#                             risk_grad is None
#                             or not bool(torch.isfinite(risk_grad).all())
#                             or self._row_grad_max(risk_grad, future_class_tensor)
#                             > tolerance
#                         ):
#                             raise RuntimeError(
#                                 "joint historical CDS risk must train old/new W only"
#                             )

#                     for name, objective in (
#                         ("Feature KD", feature_kd.loss),
#                         ("Historical Whole-CDS contrastive", contrastive.loss),
#                         ("Whole-CDS preservation", cds.loss),
#                     ):
#                         grad = torch.autograd.grad(
#                             objective,
#                             classifier.weight,
#                             retain_graph=True,
#                             allow_unused=True,
#                         )[0]
#                         if grad is not None and bool(torch.count_nonzero(grad)):
#                             raise RuntimeError(
#                                 f"{name} must not update classifier rows"
#                             )

#                     q_objectives = [
#                         ("real-current all-seen CE", real_classification.loss),
#                         ("Feature KD", feature_kd.loss),
#                         ("Historical Whole-CDS contrastive", contrastive.loss),
#                         ("Whole-CDS preservation", cds.loss),
#                     ]
#                     if kd_logit_weight > 0:
#                         q_objectives.append(("old-logit KD", kd_loss))

#                     for name, objective in q_objectives:
#                         q_grad = torch.autograd.grad(
#                             objective,
#                             current_features,
#                             retain_graph=True,
#                             allow_unused=True,
#                         )[0]
#                         if q_grad is None or not bool(torch.isfinite(q_grad).all()):
#                             raise RuntimeError(
#                                 f"{name} produced invalid current-Q gradient"
#                             )

#                     print(
#                         "[Whole-CDS representation gradient ownership] "
#                         f"RealCE Wold={real_ce_old:.3e} Wnew={real_ce_new:.3e} "
#                         f"| CE scope={representation_ce_scope}; "
#                         f"joint W trainable={representation_train_new_head}; "
#                         "real CE -> W_new only; historical CDS -> W_old/W_new; W_future/tau fixed",
#                         file=sys.stdout,
#                     )

#                     gradient_contract_checked = True

#                 loss = (
#                     classification_loss
#                     + contrastive_weight * contrastive.loss
#                     + feature_kd_weight * feature_kd.loss
#                     + kd_logit_weight * kd_loss
#                     + cds_weight * cds.loss
#                 )

#                 if not bool(torch.isfinite(loss)):

#                     raise RuntimeError(
#                         "incremental representation objective became NaN or Inf"
#                     )

#                 if sample_count == 0 and epoch in (0, epoch_count - 1):
#                     parameter_gradient_audit = self.parameter_gradient_audit(
#                         {
#                             "classification": classification_loss,
#                             "contrastive": contrastive_weight * contrastive.loss,
#                             "feature_kd": feature_kd_weight * feature_kd.loss,
#                             "cds": cds_weight * cds.loss,
#                             "logit_kd": kd_logit_weight * kd_loss,
#                         },
#                         backbone_parameters,
#                     )
#                     parameter_gradient_audit["epoch"] = epoch + 1
#                     parameter_gradient_audits.append(parameter_gradient_audit)

#                 # Observe gradient interference in the actual learned Q output.
#                 # This is a diagnostic, NOT a projected-gradient optimizer. No
#                 # extra distillation or trainable components are introduced.
#                 current_gradient = torch.autograd.grad(
#                     classification_loss, current_features, retain_graph=True
#                 )[0].detach()
#                 preservation_gradient = torch.autograd.grad(
#                     feature_kd_weight * feature_kd.loss
#                     + cds_weight * cds.loss
#                     + kd_logit_weight * kd_loss,
#                     current_features,
#                     retain_graph=True,
#                 )[0].detach()
#                 gradient_dot = (
#                     (current_gradient * preservation_gradient).flatten(1).sum(1)
#                 )
#                 current_norm = current_gradient.flatten(1).norm(dim=1)
#                 preservation_norm = preservation_gradient.flatten(1).norm(dim=1)
#                 gradient_scale = current_norm * preservation_norm
#                 valid_gradient = gradient_scale > 1e-12
#                 gradient_cosine = torch.where(
#                     valid_gradient,
#                     gradient_dot / gradient_scale.clamp_min(1e-12),
#                     torch.zeros_like(gradient_dot),
#                 ).clamp(-1.0, 1.0)
#                 if not bool(torch.isfinite(gradient_cosine).all()):
#                     raise RuntimeError(
#                         "nonfinite Q classification/preservation gradient similarity"
#                     )
#                 sums["representation_gradient_conflict"] += float(
#                     ((gradient_dot < 0) & valid_gradient).sum().item()
#                 )
#                 sums["representation_gradient_cosine"] += float(
#                     gradient_cosine.sum().item()
#                 )

#                 # Interference remains an observation, not a projected update.
#                 loss.backward()

#                 if classifier.logit_scale_parameter.grad is not None:

#                     raise RuntimeError(
#                         "incremental training must not update cosine logit scale"
#                     )

#                 if (
#                     not representation_train_new_head
#                     and classifier.weight.grad is not None
#                 ):
#                     raise RuntimeError("representation step must not train W")

#                 if (
#                     not representation_train_new_head
#                     and self._row_grad_max(classifier.weight.grad, old_class_tensor) > 0
#                 ):
#                     raise RuntimeError("frozen-head representation step updated W_old")

#                 if (
#                     self._row_grad_max(classifier.weight.grad, future_class_tensor)
#                     > 0.0
#                 ):

#                     raise RuntimeError(
#                         "representation stage changed future classifier rows"
#                     )

#                 backbone_grad_seen = False

#                 for parameter in backbone_parameters:

#                     if parameter.grad is None:

#                         continue

#                     if not bool(torch.isfinite(parameter.grad).all()):

#                         raise RuntimeError(
#                             "incremental backbone gradient contains NaN or Inf"
#                         )

#                     backbone_grad_seen = backbone_grad_seen or bool(
#                         torch.count_nonzero(parameter.grad)
#                     )

#                 if not backbone_grad_seen:

#                     raise RuntimeError(
#                         "representation objective produced zero backbone gradient"
#                     )

#                 if gradient_clip > 0:

#                     torch.nn.utils.clip_grad_norm_(
#                         representation_parameters,
#                         gradient_clip,
#                         error_if_nonfinite=True,
#                     )

#                 optimizer.step()

#                 self._assert_future_rows_unchanged(
#                     classifier, future_class_tensor, future_weight_before
#                 )

#                 n = int(labels.numel())
#                 sums["core_radius"] += float(feasible_state.mean_core_radius) * n
#                 sums["core_violation"] += (
#                     float(feasible_state.effective_core_violation) * n
#                 )
#                 sums["historical_intrusion"] += (
#                     float(feasible_state.historical_intrusion_fraction) * n
#                 )
#                 sums["spectral_interval_overlap"] += (
#                     float(feasible_state.physical_interval_overlap) * n
#                 )
#                 sums["variation_confidence"] += (
#                     float(feasible_state.ordered_variation_confidence) * n
#                 )

#                 sums["classification"] += float(classification_loss.detach().item()) * n
#                 if joint_old_risk is not None:
#                     sums["joint_historical_risk"] += (
#                         float(joint_old_risk.loss.detach().item()) * n
#                     )
#                 sums["real_classification"] += (
#                     float(real_classification.loss.detach().item()) * n
#                 )
#                 sums["direct_head_classification"] += (
#                     float(support_classification.balanced_loss.detach().item()) * n
#                 )
#                 sums["historical_cds_head_loss"] += (
#                     float(
#                         support_classification.historical_positive_loss.detach().item()
#                     )
#                     * n
#                 )
#                 sums["current_real_head_loss"] += (
#                     float(support_classification.current_positive_loss.detach().item())
#                     * n
#                 )
#                 sums["old_all_seen_accuracy"] += (
#                     float(support_classification.real_all_seen_accuracy.detach().item())
#                     * n
#                 )
#                 sums["current_new_only_accuracy"] += (
#                     float(support_classification.new_only_accuracy.detach().item()) * n
#                 )

#                 sums["feature_kd"] += float(feature_kd.loss.detach().item()) * n
#                 sums["teacher_hist_relevance"] += (
#                     float(cds.teacher_historical_relevance.detach().mean().item()) * n
#                 )

#                 sums["feature_rms"] += float(feature_kd.rms_drift.detach().item()) * n

#                 sums["contrastive"] += float(contrastive.loss.detach().item()) * n
#                 sums["diagnostic_sep_loss"] += float(separation.loss.detach()) * n

#                 sums["kd"] += float(kd_loss.detach().item()) * n

#                 sums["cds"] += float(cds.loss.detach().item()) * n

#                 sums["intra"] += float(cds.intra_loss.detach().item()) * n

#                 sums["inter"] += float(cds.inter_loss.detach().item()) * n

#                 sums["physical_abs"] += (
#                     float(cds.physical_absolute_support_mean.detach().item()) * n
#                 )

#                 sums["physical_rel"] += (
#                     float(cds.physical_relation_support_mean.detach().item()) * n
#                 )

#                 sums["physical_joint"] += (
#                     float(cds.physical_relevance_mean.detach().item()) * n
#                 )

#                 sums["sep_margin"] += (
#                     float(separation.decision_margin_mean.detach().item()) * n
#                 )

#                 sums["sep_violation"] += (
#                     float(separation.violation_fraction.detach().item()) * n
#                 )

#                 sums["sep_positive"] += (
#                     float(separation.positive_decision_score_mean.detach().item()) * n
#                 )

#                 # New/new competition is already handled by real all-seen CE;

#                 # this CDS loss contains only current-positive vs historical rivals.

#                 sums["sep_current_negative"] += 0.0

#                 sums["sep_historical_negative"] += (
#                     float(
#                         separation.historical_rival_decision_score_mean.detach().item()
#                     )
#                     * n
#                 )

#                 sums["sep_valid"] += 1.0 * n

#                 sums["sep_physical_confusability"] += (
#                     float(separation.physical_confusability_mean.detach().item()) * n
#                 )

#                 sums["sep_pair_risk"] += (
#                     float(separation.effective_pair_risk_mean.detach().item()) * n
#                 )

#                 sums["total"] += float(loss.detach().item()) * n

#                 correct_new += int(
#                     current_output.prediction.detach().eq(labels).sum().item()
#                 )

#                 sample_count += n

#             if sample_count != expected_samples:

#                 raise RuntimeError(
#                     "incremental TRAIN loader did not traverse the complete split"
#                 )

#             scheduler.step()

#             if float(classifier.logit_scale.detach().item()) != phase_start_scale:

#                 raise RuntimeError(
#                     "cosine logit scale changed during representation training"
#                 )

#             self._assert_future_rows_unchanged(
#                 classifier, future_class_tensor, future_weight_before
#             )

#             self.model.eval()

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
#                 "stage": "representation",
#                 "epoch": epoch + 1,
#                 "learning_rate": float(optimizer.param_groups[0]["lr"]),
#                 "classification_loss": sums["classification"] / sample_count,
#                 "joint_historical_risk_loss": sums["joint_historical_risk"]
#                 / sample_count,
#                 "real_current_classification_loss": sums["real_classification"]
#                 / sample_count,
#                 "direct_cds_classifier_loss": sums["direct_head_classification"]
#                 / sample_count,
#                 "direct_cds_historical_loss": sums["historical_cds_head_loss"]
#                 / sample_count,
#                 "direct_cds_current_loss": sums["current_real_head_loss"]
#                 / sample_count,
#                 "direct_old_all_seen_accuracy": sums["old_all_seen_accuracy"]
#                 / sample_count,
#                 "direct_current_new_only_accuracy": sums["current_new_only_accuracy"]
#                 / sample_count,
#                 "historical_core_radius": sums["core_radius"] / sample_count,
#                 "historical_core_violation": sums["core_violation"] / sample_count,
#                 "current_in_old_support_fraction": sums["historical_intrusion"]
#                 / sample_count,
#                 "spectral_band_interval_overlap": sums["spectral_interval_overlap"]
#                 / sample_count,
#                 "ordered_variation_confidence": sums["variation_confidence"]
#                 / sample_count,
#                 "representation_gradient_conflict_fraction": sums[
#                     "representation_gradient_conflict"
#                 ]
#                 / sample_count,
#                 "representation_gradient_cosine": sums["representation_gradient_cosine"]
#                 / sample_count,
#                 "feature_kd_loss": sums["feature_kd"] / sample_count,
#                 "teacher_historical_relevance_mean": sums["teacher_hist_relevance"]
#                 / sample_count,
#                 "coordinate_stability_loss": sums["feature_kd"] / sample_count,
#                 "epoch_representation_drift_rms": sums["feature_rms"] / sample_count,
#                 "contrastive_loss": sums["contrastive"] / sample_count,
#                 "cds_separation_loss": sums["diagnostic_sep_loss"]
#                 / sample_count,  # read-only diagnostic
#                 "cds_separation_margin": sums["sep_margin"] / sample_count,
#                 "cds_separation_violation_fraction": sums["sep_violation"]
#                 / sample_count,
#                 "contrastive_positive_similarity": float(
#                     contrastive.positive_similarity_mean.detach()
#                 ),
#                 "contrastive_current_negative_similarity": float(
#                     contrastive.current_negative_similarity_mean.detach()
#                 ),
#                 "contrastive_historical_negative_similarity": float(
#                     contrastive.historical_negative_similarity_mean.detach()
#                 ),
#                 "contrastive_valid_anchor_fraction": float(
#                     contrastive.valid_anchor_fraction.detach()
#                 ),
#                 "physical_confusability_mean": sums["sep_physical_confusability"]
#                 / sample_count,
#                 "effective_pair_risk_mean": sums["sep_pair_risk"] / sample_count,
#                 "kd_logit_loss": sums["kd"] / sample_count,
#                 "cds_preservation_loss": sums["cds"] / sample_count,
#                 "cds_intra_loss": sums["intra"] / sample_count,
#                 "cds_inter_loss": sums["inter"] / sample_count,
#                 "physical_absolute_support_mean": sums["physical_abs"] / sample_count,
#                 "physical_relation_support_mean": sums["physical_rel"] / sample_count,
#                 "physical_relevance_mean": sums["physical_joint"] / sample_count,
#                 "total_loss": sums["total"] / sample_count,
#                 "current_new_all_seen_accuracy": correct_new / float(sample_count),
#                 "new_only_ba": float(new_only["balanced_accuracy"]),
#                 "new_all_seen_ba": float(new_all["balanced_accuracy"]),
#                 "validation_new_all_seen_ba": (
#                     None
#                     if validation_all is None
#                     else float(validation_all["balanced_accuracy"])
#                 ),
#             }

#             # paired-feature evidence quality, observed ONLY on incoming classes.
#             record["alignment_holdout_candidate_error"] = float(
#                 _epoch_alignment.holdout_candidate_error.item()
#             )
#             record["alignment_holdout_identity_error"] = float(
#                 _epoch_alignment.holdout_identity_error.item()
#             )
#             record["alignment_transport_accepted"] = bool(
#                 _epoch_alignment.transport_accepted.item()
#             )
#             # Legacy report fields are preserved without checkpoint ranking.
#             record["validation_joint_selection_objective"] = None
#             record["validation_historical_cds_risk"] = None
#             record["validation_real_new_ce"] = None
#             history.append(record)

#             tqdm.write(
#                 f"[Representation {epoch + 1:03d}/{epoch_count:03d}] "
#                 f"CEjoint={record['classification_loss']:.4f} "
#                 f"CEreal={record['real_current_classification_loss']:.4f} "
#                 f"LegacyCoreDiag={record['direct_cds_classifier_loss']:.4f} "
#                 f"Con={record['contrastive_loss']:.4f} "
#                 f"CoreR={record['historical_core_radius']:.3f} "
#                 f"CoreViol={100*record['historical_core_violation']:.1f}% "
#                 f"GradConflict={100*record['representation_gradient_conflict_fraction']:.1f}% "
#                 f"QGradientCos={record['representation_gradient_cosine']:.3f} "
#                 f"FKD={record['feature_kd_loss']:.4f} "
#                 # f"KD={record['kd_logit_loss']:.4f} "
#                 f"CDS={record['cds_preservation_loss']:.4f}"
#                 f"[I={record['cds_intra_loss']:.4f},R={record['cds_inter_loss']:.4f}] "
#                 f"DecisionMargin={record['cds_separation_margin']:.4f} "
#                 f"DecisionViol={100*record['cds_separation_violation_fraction']:.1f}% | "
#                 f"NewOnly/All={100*record['new_only_ba']:.2f}%/"
#                 f"{100*record['new_all_seen_ba']:.2f}%",
#                 file=sys.stdout,
#             )

#         progress.close()

#         if not history:

#             raise RuntimeError("incremental representation training produced no epochs")

#         # The deployed representation and cosine rows are exactly those
#         # produced by the final optimizer step, not a validation-selected copy.
#         self._assert_future_rows_unchanged(
#             classifier, future_class_tensor, future_weight_before
#         )
#         print(
#             f"[Joint Q+W final epoch] retained epoch {epoch_count}/{epoch_count}; "
#             "no checkpoint ranking, no Q/W rollback",
#             file=sys.stdout,
#         )

#         # Phase-end statistical synchronization. This is NOT another training stage.

#         self.model.eval()

#         for parameter in self.model.backbone.parameters():

#             parameter.requires_grad_(False)

#             parameter.grad = None

#         final_features_full, final_current_structure = collect_current_state()
#         # Before/after migration is measured on the same current TRAIN samples.
#         # This is an observable signal for historical Q-distribution alignment,
#         # not proof of how unavailable historical samples migrated.
#         with torch.no_grad():
#             before_q = torch.nn.functional.normalize(
#                 reference_features_full.reshape(reference_features_full.size(0), -1),
#                 dim=1,
#             )
#             after_q = torch.nn.functional.normalize(
#                 final_features_full.reshape(final_features_full.size(0), -1), dim=1
#             )
#             paired_new_feature_cosine = float(
#                 (before_q * after_q).sum(dim=1).mean().item()
#             )
#             paired_new_feature_chord = float(
#                 (before_q - after_q).square().sum(dim=1).mean().sqrt().item()
#             )

#         alignment_spectral_evidence = self.model.physical_spectral_evidence(
#             reference_spectra
#         )
#         final_historical_structure, final_transform, final_alignment = (
#             correct_historical_structure(
#                 historical_structure=historical_structure,
#                 current_reference_structure=current_reference_structure,
#                 current_structure=final_current_structure,
#                 reference_features=reference_features_full,
#                 current_features=final_features_full,
#                 use_transport=historical_transport,
#                 historical_moments=historical_moments,
#                 physical_spectral_evidence=alignment_spectral_evidence,
#             )
#         )
#         committed_structure = merge_class_structure_batches(
#             (final_historical_structure, final_current_structure)
#         )

#         committed_ids = tuple(
#             int(v) for v in committed_structure.class_ids.detach().cpu().tolist()
#         )
#         if committed_ids != seen_class_ids:
#             raise RuntimeError(
#                 "final Whole-CDS structure produced incorrect seen-class order"
#             )

#         # Prepare and validate a complete candidate CDS; the live bank stays
#         # immutable until classifier reconstruction succeeds. Historical Q
#         # translations, conditional Q moments and native P move coherently.
#         candidate_bank = deepcopy(self.model.structure_bank)
#         # The reference->student map is a validated orthogonal rotation. Means
#         # and low-rank covariance bases must commit in exactly the same frame.
#         candidate_bank.apply_historical_orthogonal_transform(
#             old_class_ids, final_transform
#         )
#         candidate_bank.replace_all_with_moments(
#             committed_structure,
#             final_features_full,
#             alignment_spectral_evidence,
#             reference_labels,
#         )
#         candidate_bank.validate_state()
#         committed_seen_cds = candidate_bank.build_cds(candidate_bank.class_structures())
#         reconstruction_support = HistoricalCDSSupportBank(
#             candidate_bank,
#             margin_std=self.historical_support.margin_std,
#             risk_mode=self.historical_support.risk_mode,
#         )
#         reconstruction_pair_risk = whole_cds_pair_risk(committed_seen_cds)[0].detach()

#         if not torch.equal(
#             final_historical_structure.physical_identity,
#             historical_structure.physical_identity,
#         ) or not torch.equal(
#             final_historical_structure.physical_variation,
#             historical_structure.physical_variation,
#         ):
#             raise RuntimeError("final historical correction changed physical I_P/V_P")

#         # Preserve only model parameters for later reporting. No reporting/test
#         # example is opened until reconstruction and phase finalization finish.
#         post_rep_classifier = deepcopy(classifier).requires_grad_(False).eval()
#         post_rep_new_only = self.evaluate_loader(
#             train_eval_loader, class_ids=new_class_ids, target_class_ids=new_class_ids
#         )
#         post_rep_new_all = self.evaluate_loader(
#             train_eval_loader, class_ids=seen_class_ids, target_class_ids=new_class_ids
#         )

#         # Current-task learned-feature movement report: both structures were fitted to EXACTLY
#         # the same current TRAIN class IDs and spectra, before/after backbone
#         # optimization. Only per-class aggregate diagnostics are persisted.
#         classwise_new_migration = {}
#         classwise_old_transport = {}
#         with torch.no_grad():
#             for row, cls_id in enumerate(new_class_ids):
#                 before = current_reference_structure.learned_identity[row].reshape(-1)
#                 after = final_current_structure.learned_identity[row].reshape(-1)
#                 before_rel = current_reference_structure.learned_variation[row]
#                 after_rel = final_current_structure.learned_variation[row]
#                 before_disp = current_reference_structure.learned_absolute_deviation[
#                     row
#                 ]
#                 after_disp = final_current_structure.learned_absolute_deviation[row]
#                 classwise_new_migration[str(cls_id)] = {
#                     "identity_delta_l2": float((after - before).norm().item()),
#                     "identity_cosine": float(
#                         torch.nn.functional.cosine_similarity(
#                             before[None], after[None]
#                         ).item()
#                     ),
#                     "relational_interval_change_l2": float(
#                         (after_rel - before_rel).norm().item()
#                     ),
#                     "learned_absolute_deviation_delta_l2": float(
#                         (after_disp - before_disp).norm().item()
#                     ),
#                     "learned_absolute_deviation_mean_before": float(
#                         before_disp.mean().item()
#                     ),
#                     "learned_absolute_deviation_mean_after": float(
#                         after_disp.mean().item()
#                     ),
#                 }
#             for row, cls_id in enumerate(old_class_ids):
#                 # Measured only on the compact stored class means; neither
#                 # accepted nor rejected transport certifies unavailable old drift.
#                 classwise_old_transport[str(cls_id)] = {
#                     "stored_identity_delta_l2": float(
#                         (
#                             final_historical_structure.learned_identity[row]
#                             - historical_structure.learned_identity[row]
#                         )
#                         .norm()
#                         .item()
#                     ),
#                     "stored_dispersion_delta_l2": float(
#                         (
#                             final_historical_structure.learned_absolute_deviation[row]
#                             - historical_structure.learned_absolute_deviation[row]
#                         )
#                         .norm()
#                         .item()
#                     ),
#                     "unobserved_old_feature_drift_certified": False,
#                 }

#         old_weight_after_training = (
#             classifier.class_weight(old_class_tensor).detach().clone()
#         )
#         if not representation_train_new_head and not torch.equal(
#             old_weight_after_training, old_weight_before
#         ):
#             raise RuntimeError(
#                 "frozen-head ablation must preserve all historical W rows"
#             )
#         new_weight_after_training = (
#             classifier.class_weight(new_class_tensor).detach().clone()
#         )
#         self._assert_future_rows_unchanged(
#             classifier, future_class_tensor, future_weight_before
#         )
#         if float(classifier.logit_scale.detach().item()) != phase_start_scale:
#             raise RuntimeError("incremental representation changed cosine logit scale")

#         # Read-only final diagnostics before actual frozen-Q reconstruction.
#         final_classifier_objective = whole_cds_class_balanced_classifier_loss(
#             classifier=classifier,
#             cds=committed_seen_cds,
#             historical_class_count=len(old_class_ids),
#         )
#         pre_classifier_objective = final_classifier_objective
#         pre_finalize = final_classifier_objective

#         final_radii, final_intrusion = current_class_feasible_radii(
#             current_features=final_features_full,
#             current_labels=reference_labels,
#             cds=committed_seen_cds,
#             historical_class_count=len(old_class_ids),
#         )
#         final_feasible_state = feasible_directional_head_objective(
#             classifier=classifier,
#             current_features=final_features_full.detach(),
#             current_labels=reference_labels,
#             seen_cds=committed_seen_cds,
#             historical_class_count=len(old_class_ids),
#             feasible_radii=final_radii,
#             intrusion_fraction=final_intrusion,
#         )
#         final_classifier_state = final_feasible_state.state
#         with torch.no_grad():
#             diagnostic_seen = classifier(
#                 final_features_full,
#                 class_ids=seen_class_tensor,
#                 trainable_class_ids=(),
#             )
#             diagnostic_new_ce = classification_objective(
#                 logits=diagnostic_seen.logits,
#                 class_ids=diagnostic_seen.class_ids,
#                 labels_global=reference_labels,
#             ).loss
#             diagnostic_new_all_accuracy = (
#                 diagnostic_seen.prediction.eq(reference_labels)
#                 .to(diagnostic_seen.logits.dtype)
#                 .mean()
#             )
#             diagnostic_new_only_output = classifier(
#                 final_features_full,
#                 class_ids=new_class_tensor,
#                 trainable_class_ids=(),
#             )
#             diagnostic_new_only_accuracy = (
#                 diagnostic_new_only_output.prediction.eq(reference_labels)
#                 .to(diagnostic_new_only_output.logits.dtype)
#                 .mean()
#             )

#         print(
#             "[Before Whole-CDS support reconstruction: legacy diagnostic ONLY] "
#             f"JointLoss={float(final_classifier_state.balanced_loss.detach().item()):.4f} "
#             f"OldCDSLoss={float(final_classifier_state.historical_positive_loss.detach().item()):.4f} "
#             f"NewRealLoss={float(final_classifier_state.current_positive_loss.detach().item()):.4f} "
#             f"RealNewCE={float(diagnostic_new_ce.detach().item()):.4f} "
#             f"NewAllAcc={100*float(diagnostic_new_all_accuracy.detach().item()):.2f}% "
#             f"NewOnlyAcc={100*float(diagnostic_new_only_accuracy.detach().item()):.2f}% "
#             f"OldNewViol={100*float(final_classifier_objective.old_new_violation_fraction.detach().item()):.1f}%",
#             file=sys.stdout,
#         )

#         # Read-only historical/new-class classifier audit; post-hoc head training
#         # is disabled, and the final-epoch W must remain the deployed W.
#         requested_recon_epochs = int(self.cfg("support_head_epochs", 0))
#         if requested_recon_epochs < 0:
#             raise ValueError("support_head_epochs must be non-negative")
#         if requested_recon_epochs:
#             warnings.warn(
#                 "Unified CDS already trains historical and current cosine rows jointly; "
#                 "the legacy post-hoc head reconstruction is disabled",
#                 RuntimeWarning,
#             )
#         recon_epochs = 0
#         recon_lr = _finite_positive(
#             "support_head_lr", self.cfg("support_head_lr", 1e-3)
#         )

#         self.model.eval()
#         # Zero-epoch diagnostics must not freeze the deployed student backbone.
#         if recon_epochs:
#             for parameter in self.model.backbone.parameters():
#                 parameter.requires_grad_(False)
#                 parameter.grad = None
#         classifier.weight.grad = None
#         fixed_q = final_features_full.detach()
#         fixed_targets = reference_labels.detach()
#         # The features are transient and are discarded after reconstruction.
#         # Each configured epoch now traverses minibatches (not one full-batch
#         # gradient step). This matches the meaning of training epochs and
#         # permits classifier adaptation against old + new classes.
#         targets_seen = _targets_local(fixed_targets, seen_class_tensor)
#         recon_opt = torch.optim.AdamW(
#             [classifier.weight], lr=recon_lr, weight_decay=0.0
#         )
#         history_reconstruction = []
#         selection_source = "current_train_only"
#         selection_q, selection_y = fixed_q, fixed_targets
#         if val_loader is not None:
#             with torch.no_grad():
#                 valid_rows = self.collect_ordered_features(val_loader)
#                 valid_q = valid_rows["ordered_features"].detach()
#                 valid_y = valid_rows["labels"].detach()
#                 if all(bool((valid_y == cid).any()) for cid in new_class_ids):
#                     selection_q, selection_y = valid_q, valid_y
#                     selection_source = "current_validation"
#                 else:
#                     selection_source = "current_train_only_missing_validation_classes"

#         @torch.no_grad()
#         def reconstruction_objective():
#             historical = reconstruction_support.historical_risk(
#                 classifier,
#                 old_class_ids,
#                 committed_seen_cds,
#                 class_pair_risk=reconstruction_pair_risk,
#             )
#             outputs = classifier(
#                 selection_q,
#                 class_ids=seen_class_tensor,
#                 trainable_class_ids=(),
#             )
#             targets = _targets_local(selection_y, seen_class_tensor)
#             sample_ce = torch.nn.functional.cross_entropy(
#                 outputs.logits,
#                 targets,
#                 reduction="none",
#             )
#             new_ce = torch.stack(
#                 [sample_ce[selection_y.eq(cid)].mean() for cid in new_class_ids]
#             ).mean()
#             criterion = (
#                 len(old_class_ids) * historical.loss + len(new_class_ids) * new_ce
#             ) / len(seen_class_ids)
#             return (
#                 float(criterion.item()),
#                 float(historical.loss.item()),
#                 float(new_ce.item()),
#             )

#         initial_score, initial_old_risk, initial_new_ce = reconstruction_objective()
#         selected_score = initial_score
#         selected_epoch = 0
#         selected_head = classifier.weight.detach().clone()
#         history_reconstruction.append(
#             {
#                 "epoch": 0,
#                 "optimizer_steps": 0,
#                 "selection_objective": initial_score,
#                 "historical_support_loss": initial_old_risk,
#                 "selection_new_ce": initial_new_ce,
#                 "selection_source": selection_source,
#             }
#         )
#         for rec_epoch in range(recon_epochs):
#             permutation = torch.randperm(
#                 fixed_targets.numel(), device=fixed_targets.device
#             )
#             steps = 0
#             for rows in permutation.split(train_batch_size):
#                 batch_q = fixed_q.index_select(0, rows)
#                 batch_labels = fixed_targets.index_select(0, rows)
#                 batch_targets = targets_seen.index_select(0, rows)
#                 local_new = _targets_local(batch_labels, new_class_tensor)
#                 weights = current_class_weights.index_select(0, local_new)

#                 recon_opt.zero_grad(set_to_none=True)
#                 risk = reconstruction_support.historical_risk(
#                     classifier,
#                     old_class_ids,
#                     committed_seen_cds,
#                     class_pair_risk=reconstruction_pair_risk,
#                 )
#                 out = classifier(
#                     batch_q,
#                     class_ids=seen_class_tensor,
#                     trainable_class_ids=seen_class_tensor,
#                 )
#                 per_sample = torch.nn.functional.cross_entropy(
#                     out.logits, batch_targets, reduction="none"
#                 )
#                 # A minibatch need not contain every incoming class.
#                 # Class weights estimated from the entire incoming TRAIN split
#                 # approximate one equal vote per incoming class.
#                 new_ce = (per_sample * weights).mean()
#                 # Each seen class has equal weight, not each TASK.
#                 joint = (
#                     len(old_class_ids) * risk.loss + len(new_class_ids) * new_ce
#                 ) / len(seen_class_ids)
#                 if not bool(torch.isfinite(joint)):
#                     raise RuntimeError("nonfinite joint-moment classifier objective")
#                 joint.backward()
#                 if classifier.weight.grad is None or not bool(
#                     torch.isfinite(classifier.weight.grad).all()
#                 ):
#                     raise RuntimeError(
#                         "historical/current moment reconstruction gradient invalid"
#                     )
#                 if (
#                     future_class_ids
#                     and self._row_grad_max(classifier.weight.grad, future_class_tensor)
#                     > 0
#                 ):
#                     raise RuntimeError(
#                         "moment reconstruction gradient leaked into future rows"
#                     )
#                 if gradient_clip > 0:
#                     torch.nn.utils.clip_grad_norm_(
#                         [classifier.weight], gradient_clip, error_if_nonfinite=True
#                     )
#                 recon_opt.step()
#                 self._assert_future_rows_unchanged(
#                     classifier, future_class_tensor, future_weight_before
#                 )
#                 steps += 1

#             # Log POST-epoch scores on the same fixed incoming TRAIN features.
#             with torch.no_grad():
#                 risk = reconstruction_support.historical_risk(
#                     classifier,
#                     old_class_ids,
#                     committed_seen_cds,
#                     class_pair_risk=reconstruction_pair_risk,
#                 )
#                 out = classifier(
#                     fixed_q,
#                     class_ids=seen_class_tensor,
#                     trainable_class_ids=(),
#                 )
#                 ce = torch.nn.functional.cross_entropy(
#                     out.logits, targets_seen, reduction="none"
#                 )
#                 new_ce = torch.stack(
#                     [ce[fixed_targets.eq(cid)].mean() for cid in new_class_ids]
#                 ).mean()
#                 joint = (
#                     len(old_class_ids) * risk.loss + len(new_class_ids) * new_ce
#                 ) / len(seen_class_ids)
#                 true_new_logits = out.logits.gather(1, targets_seen[:, None]).squeeze(1)
#                 strongest_old_logit = out.logits[:, : len(old_class_ids)].max(1).values
#                 row = {
#                     "epoch": rec_epoch + 1,
#                     "optimizer_steps": steps,
#                     "historical_support_loss": float(risk.loss),
#                     "historical_objective": risk.objective,
#                     "historical_mean_ce": (
#                         None if risk.mean_ce_loss is None else float(risk.mean_ce_loss)
#                     ),
#                     "historical_curvature": (
#                         None
#                         if risk.curvature_correction is None
#                         else float(risk.curvature_correction)
#                     ),
#                     "current_real_ce": float(new_ce),
#                     "balanced_loss": float(joint),
#                     "new_vs_old_logit_margin": float(
#                         (true_new_logits - strongest_old_logit).mean()
#                     ),
#                     "new_all_seen_train_acc": float(
#                         out.prediction.eq(fixed_targets).float().mean()
#                     ),
#                     "hardest_historical_rival": risk.per_class_worst_rival.cpu().tolist(),
#                     "historical_classwise_moment_margin_penalty": risk.per_class_loss.cpu().tolist(),
#                 }
#             score, held_old_risk, held_new_ce = reconstruction_objective()
#             row["selection_objective"] = score
#             row["selection_new_ce"] = held_new_ce
#             row["selection_old_risk"] = held_old_risk
#             row["selection_source"] = selection_source
#             history_reconstruction.append(row)
#             # Safeguard the incoming-class validation CE against the starting
#             # joint-trained checkpoint. A better historical surrogate alone
#             # must not justify sacrificing current real-class recognition.
#             if held_new_ce <= initial_new_ce + 1e-8 and score < selected_score:
#                 selected_score = score
#                 selected_epoch = rec_epoch + 1
#                 selected_head = classifier.weight.detach().clone()
#         # Post-hoc reconstruction is disabled in unified CDS. Never restore
#         # a selected W state after joint training.
#         if recon_epochs:
#             with torch.no_grad():
#                 classifier.weight.copy_(selected_head)
#         classifier.weight.grad = None
#         # Refresh all diagnostics after the actual deployed W has been fitted.
#         final_feasible_state = feasible_directional_head_objective(
#             classifier=classifier,
#             current_features=final_features_full.detach(),
#             current_labels=reference_labels,
#             seen_cds=committed_seen_cds,
#             historical_class_count=len(old_class_ids),
#             feasible_radii=final_radii,
#             intrusion_fraction=final_intrusion,
#         )
#         final_classifier_state = final_feasible_state.state
#         final_classifier_objective = whole_cds_class_balanced_classifier_loss(
#             classifier=classifier,
#             cds=committed_seen_cds,
#             historical_class_count=len(old_class_ids),
#         )
#         final_head_accepted = selected_epoch
#         print(
#             "[Joint Q+W final-state diagnostics] calibration=disabled; "
#             f"old_risk={initial_old_risk:.4f} new_CE={initial_new_ce:.4f}; "
#             "classifier weights unchanged since last training step",
#             file=sys.stdout,
#         )
#         # Commit the complete, validated candidate after reconstruction.
#         # Failed reconstruction leaves the historical CDS bank unchanged.
#         self.model.structure_bank.load_state_dict(
#             candidate_bank.state_dict(), strict=True
#         )
#         self.historical_support.require(seen_class_ids)
#         self.model.validate_finalized_state()
#         risk_audit = self.audit_current_classifier_risk(
#             val_loader, class_ids=new_class_ids, cds=committed_seen_cds
#         )

#         self.dataset.finalize_phase(phase_id)

#         classifier.logit_scale_parameter.requires_grad_(scale_requires_grad)

#         classifier.logit_scale_parameter.grad = None

#         old_reporting_loader = self.dataset.get_reporting_dataloader(
#             previous_phase, batch_size=eval_batch_size
#         )
#         live_backbone, live_classifier = self.model.backbone, self.model.classifier
#         try:
#             self.model.backbone = reference_backbone
#             self.model.classifier = reference_classifier
#             old_before = self.evaluate_loader(
#                 old_reporting_loader,
#                 class_ids=previous_seen_class_ids,
#                 target_class_ids=previous_seen_class_ids,
#             )
#             # Q did not change during head reconstruction; the saved W therefore
#             # reproduces the post-representation decision on this reporting split.
#             self.model.backbone = live_backbone
#             self.model.classifier = post_rep_classifier
#             post_rep_old_only = self.evaluate_loader(
#                 old_reporting_loader,
#                 class_ids=old_class_ids,
#                 target_class_ids=old_class_ids,
#             )
#             post_rep_old_all = self.evaluate_loader(
#                 old_reporting_loader,
#                 class_ids=seen_class_ids,
#                 target_class_ids=old_class_ids,
#             )
#         finally:
#             self.model.backbone, self.model.classifier = live_backbone, live_classifier
#         del post_rep_classifier

#         # Step 12: all-seen evaluation uses exactly tau*cosine(Q,W).

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

#         final_old_only = self.evaluate_loader(
#             old_reporting_loader,
#             class_ids=old_class_ids,
#             target_class_ids=old_class_ids,
#         )

#         final_old_all = self.evaluate_loader(
#             old_reporting_loader,
#             class_ids=seen_class_ids,
#             target_class_ids=old_class_ids,
#         )

#         final_new_only = self.evaluate_loader(
#             train_eval_loader, class_ids=new_class_ids, target_class_ids=new_class_ids
#         )

#         final_new_all = self.evaluate_loader(
#             train_eval_loader, class_ids=seen_class_ids, target_class_ids=new_class_ids
#         )

#         final_record = history[-1]
#         final_old_weight = classifier.class_weight(old_class_tensor).detach()
#         final_new_weight = classifier.class_weight(new_class_tensor).detach()
#         method_state = {
#             "stores_old_images": False,
#             "stores_old_feature_rows": False,
#             "stores_synthetic_support": False,
#             "stores_transient_distribution_support": False,
#             "uses_transient_historical_distribution_support": False,
#             "uses_persistent_whole_cds_support": True,
#             "feasible_core_objective_active": False,
#             "support_reconstruction_active": bool(recon_epochs > 0),
#             "uses_real_current_positive_support": True,
#             "uses_covariance_or_gaussian_memory": True,
#             "uses_gaussian_density_or_sampling": False,
#             "uses_mean_or_second_moment_classifier_memory": True,
#             "uses_second_classifier": False,
#             "uses_phase_bias": False,
#             "uses_posthoc_classifier_reconstruction": bool(recon_epochs > 0),
#             "final_synchronized_head_steps_accepted": final_head_accepted,
#             "selected_support_head_epoch": selected_epoch,
#             "reconstruction_selection_source": selection_source,
#             "reconstruction_selected_objective": selected_score,
#             "reconstruction_initial_objective": initial_score,
#             "reconstruction_initial_new_ce": initial_new_ce,
#             "reconstruction_new_ce_guard_active": True,
#             "historical_support_reconstruction": history_reconstruction,
#             "uses_balanced_old_new_positive_classifier_support": True,
#             "uses_signed_whole_cds_distribution": True,
#             "historical_support_audit": self.historical_support.diagnostics(),
#             "uses_per_epoch_direct_historical_coordinate_refresh": False,
#             "historical_risk_mode": self.historical_support.risk_mode,
#             "representation_ce_scope": representation_ce_scope,
#             "representation_train_new_head": representation_train_new_head,
#             "joint_seen_cosine_head_during_representation": representation_train_new_head,
#             "historical_cds_head_risk_in_representation": representation_train_new_head,
#             "risk_approximation_audit": risk_audit,
#             "competition_splits": {"old": "reporting", "new": "train"},
#             "parameter_gradient_audit": parameter_gradient_audit,
#             "parameter_gradient_audits": parameter_gradient_audits,
#             "representation_incoming_direction_change": self._max_direction_change(
#                 admission_new_weight, new_weight_after_training
#             ),
#             "head_admission": admission_state,
#             "historical_transport_enabled": historical_transport,
#             "historical_feasible_radius_mean": float(
#                 final_feasible_state.mean_core_radius
#             ),
#             "historical_current_intrusion_fraction": float(
#                 final_feasible_state.historical_intrusion_fraction
#             ),
#             "historical_core_violation": float(
#                 final_feasible_state.effective_core_violation
#             ),
#             "historical_spectral_interval_overlap": float(
#                 final_feasible_state.physical_interval_overlap
#             ),
#             "historical_vq_directional_confidence": float(
#                 final_feasible_state.ordered_variation_confidence
#             ),
#             "representation_gradient_conflict_fraction": float(
#                 final_record["representation_gradient_conflict_fraction"]
#             ),
#             "representation_gradient_cosine": float(
#                 final_record["representation_gradient_cosine"]
#             ),
#             "uses_feature_kd": feature_kd_weight > 0,
#             "new_class_before_feature_mean_norm": reference_feature_norm,
#             "new_class_before_after_mean_cosine": paired_new_feature_cosine,
#             "new_class_before_after_chord_rms": paired_new_feature_chord,
#             "new_class_before_after_structure": classwise_new_migration,
#             "historical_cds_estimated_transport": classwise_old_transport,
#             "final_teacher_historical_relevance_mean": float(
#                 final_record["teacher_historical_relevance_mean"]
#             ),
#             "uses_logit_kd": bool(kd_logit_weight > 0),
#             "prediction_rule": "tau*cosine(Q,W)",
#             "representation_epochs": epoch_count,
#             "selected_representation_epoch": selected_representation_epoch,
#             "representation_selection_source": "final_epoch_no_selection",
#             "selected_representation_objective": selected_representation_score,
#             "representation_lr": learning_rate,
#             "contrastive_weight": contrastive_weight,
#             "feature_kd_weight": feature_kd_weight,
#             "kd_logit_weight": kd_logit_weight,
#             "kd_temperature": kd_temperature,
#             "cds_weight": cds_weight,
#             "final_joint_classification_loss": float(
#                 final_record["classification_loss"]
#             ),
#             "final_joint_historical_risk_loss": float(
#                 final_record["joint_historical_risk_loss"]
#             ),
#             "final_real_current_classification_loss": float(
#                 final_record["real_current_classification_loss"]
#             ),
#             "final_feature_kd_loss": float(final_record["feature_kd_loss"]),
#             "final_cds_separation_loss": float(final_record["cds_separation_loss"]),
#             "final_cds_separation_margin": float(final_record["cds_separation_margin"]),
#             "final_cds_separation_violation_fraction": float(
#                 final_record["cds_separation_violation_fraction"]
#             ),
#             "final_cds_preservation_loss": float(final_record["cds_preservation_loss"]),
#             "final_cds_intra_loss": float(final_record["cds_intra_loss"]),
#             "final_cds_inter_loss": float(final_record["cds_inter_loss"]),
#             "final_physical_absolute_support_mean": float(
#                 final_record["physical_absolute_support_mean"]
#             ),
#             "final_physical_relation_support_mean": float(
#                 final_record["physical_relation_support_mean"]
#             ),
#             "final_physical_relevance_mean": float(
#                 final_record["physical_relevance_mean"]
#             ),
#             "phase_transport_accepted_on_new_class_holdout": bool(
#                 final_alignment.transport_accepted.item()
#             ),
#             "phase_holdout_affine_error": float(
#                 final_alignment.holdout_candidate_error.item()
#             ),
#             "phase_holdout_identity_error": float(
#                 final_alignment.holdout_identity_error.item()
#             ),
#             "phase_mapping_error": float(final_alignment.mapping_error.item()),
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
#             "final_direct_cds_classifier_loss": float(
#                 final_classifier_state.balanced_loss.detach().item()
#             ),
#             "final_direct_cds_historical_loss": float(
#                 final_classifier_state.historical_positive_loss.detach().item()
#             ),
#             "final_direct_cds_current_loss": float(
#                 final_classifier_state.current_positive_loss.detach().item()
#             ),
#             "final_whole_cds_old_new_violation_fraction": float(
#                 final_classifier_objective.old_new_violation_fraction.detach().item()
#             ),
#             "classifier_learned_pair_risk_mean": float(
#                 final_classifier_state.learned_pair_risk_mean.detach().item()
#             ),
#             "classifier_physical_pair_risk_mean": float(
#                 final_classifier_state.physical_pair_risk_mean.detach().item()
#             ),
#             "classifier_effective_pair_risk_mean": float(
#                 final_classifier_state.effective_pair_risk_mean.detach().item()
#             ),
#             "historical_classifier_training_max_change": self._max_parameter_change(
#                 old_weight_before, final_old_weight
#             ),
#             "historical_classifier_training_direction_change": self._max_direction_change(
#                 old_weight_before, final_old_weight
#             ),
#             "incoming_classifier_training_max_change": self._max_parameter_change(
#                 new_weight_phase_start, final_new_weight
#             ),
#             "incoming_classifier_training_direction_change": self._max_direction_change(
#                 new_weight_phase_start, final_new_weight
#             ),
#             "phase_start_old_ba": float(old_before["balanced_accuracy"]),
#             "phase_start_old_all_seen_ba": None,  # No true all-seen phase-start measurement.
#             "phase_start_new_only_ba": float(phase_start_new_only["balanced_accuracy"]),
#             "phase_start_new_all_seen_ba": float(
#                 phase_start_new_all_seen["balanced_accuracy"]
#             ),
#             "post_rep_old_only_ba": float(post_rep_old_only["balanced_accuracy"]),
#             "post_rep_old_all_seen_ba": float(post_rep_old_all["balanced_accuracy"]),
#             "post_rep_new_only_ba": float(post_rep_new_only["balanced_accuracy"]),
#             "post_rep_new_all_seen_ba": float(post_rep_new_all["balanced_accuracy"]),
#             "final_old_only_ba": float(final_old_only["balanced_accuracy"]),
#             "final_old_all_seen_ba": float(final_old_all["balanced_accuracy"]),
#             "final_new_only_ba": float(final_new_only["balanced_accuracy"]),
#             "final_new_all_seen_ba": float(final_new_all["balanced_accuracy"]),
#             "classifier_logit_scale_frozen_incremental": True,
#             "historical_support_reconstruction": history_reconstruction,
#             "historical_support_classes": list(self.historical_support.class_ids),
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
