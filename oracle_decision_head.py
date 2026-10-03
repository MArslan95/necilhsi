from __future__ import annotations

"""Read-only oracle diagnostics for the NECIL-HSI decision head.

This module is intentionally *not* part of training.  It uses historical TRAIN
rows only after a phase has been finalized in order to falsify two hypotheses:

Test 1: with the successful Phase-0 representation frozen, can the exact
        classifier-space features reconstruct a strong analytic base head?
Test 2: with the learned Phase-1 representation frozen, can an oracle all-seen
        analytic head built from real old+new TRAIN features close the
        NewOnly -> NewAll decision gap?

Oracle results are diagnostic only and are invalid as reported NECIL method
results because Test 2 deliberately accesses historical training rows.
"""

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import torch
import torch.nn.functional as F

Tensor = torch.Tensor
ORACLE_AUTHORIZATION = "ORACLE_DIAGNOSTIC_ONLY"


@dataclass(frozen=True)
class OracleHeadSolution:
    class_ids: tuple[int, ...]
    weight: Tensor                 # [C,D], row-normalized
    counts: Tensor                 # [C]
    class_sums: Tensor             # [C,D] = S_c
    gram: Tensor                   # [D,D] = weighted all-seen G
    cross: Tensor                  # [D,C] = weighted all-seen B
    ridge: float
    class_balanced: bool
    relative_residual: float


def _ids(values: Sequence[int], *, name: str) -> tuple[int, ...]:
    result = tuple(int(v) for v in values)
    if not result or len(result) != len(set(result)) or any(v < 0 for v in result):
        raise ValueError(f"{name} must contain unique non-negative class IDs")
    return result


def _validate_feature_rows(features: Tensor, labels: Tensor) -> tuple[Tensor, Tensor]:
    z = torch.as_tensor(features)
    y = torch.as_tensor(labels, device=z.device, dtype=torch.long).flatten()
    if z.ndim != 2 or z.size(0) == 0 or z.size(1) == 0:
        raise ValueError("decision_features must be non-empty [N,D]")
    if y.shape != (z.size(0),):
        raise ValueError("labels must align with decision_features")
    if not torch.is_floating_point(z) or z.is_complex():
        raise TypeError("decision_features must be real floating point")
    if not bool(torch.isfinite(z).all()):
        raise ValueError("decision_features contain NaN/Inf")
    # TrainerHelper already creates exact cosine coordinates.  Re-normalize only
    # as a numerical assertion/cleanup, using the same dtype epsilon.
    eps = torch.finfo(z.dtype).eps
    norms = z.norm(p=2, dim=1)
    if bool((norms <= eps).any()):
        raise RuntimeError("decision_features contain zero-norm rows")
    z = F.normalize(z, p=2, dim=1, eps=eps)
    return z, y


def _class_columns(labels: Tensor, class_ids: tuple[int, ...]) -> Tensor:
    columns = torch.full_like(labels, -1)
    for column, class_id in enumerate(class_ids):
        columns[labels.eq(class_id)] = int(column)
    if bool((columns < 0).any()):
        observed = sorted(int(v) for v in labels.unique().detach().cpu().tolist())
        raise ValueError(
            f"labels contain classes outside requested class_ids; observed={observed}, "
            f"requested={list(class_ids)}"
        )
    return columns


def accumulate_class_conditioned_information(
    decision_features: Tensor,
    labels: Tensor,
    *,
    class_ids: Sequence[int],
) -> Dict[str, Tensor]:
    """Return exact class-conditioned decision information from real feature rows.

    S_c = sum_i z_i for y_i=c.
    G_c = sum_i z_i z_i^T is not materialized for every class here because the
    diagnostic ridge system can accumulate it directly from Z without retaining
    C copies of a D x D matrix.  Per-class G_c is nevertheless exactly the
    information used when ``class_balanced=True`` via per-class sample weights.
    """
    ids = _ids(class_ids, name="class_ids")
    z, y = _validate_feature_rows(decision_features, labels)
    columns = _class_columns(y, ids)
    c = len(ids)
    counts = torch.bincount(columns, minlength=c)
    if bool((counts <= 0).any()):
        raise RuntimeError("every requested class must have at least one TRAIN row")
    sums = torch.zeros(c, z.size(1), device=z.device, dtype=z.dtype)
    sums.index_add_(0, columns, z)
    return {
        "counts": counts,
        "class_sums": sums,
        "class_columns": columns,
    }


def solve_ridge_oracle_head(
    decision_features: Tensor,
    labels: Tensor,
    *,
    class_ids: Sequence[int],
    ridge_scale: float = 1.0e-3,
    class_balanced: bool = True,
) -> OracleHeadSolution:
    """Solve one all-seen linear head from exact frozen-representation statistics.

    The primary diagnostic uses equal class mass because HSI class frequencies are
    strongly imbalanced and the project's principal metric is balanced accuracy.
    The weighting remains reconstructable from per-class (S_c, G_c, N_c): each
    class contributes the same total mass N/(C*N_c).
    """
    scale = float(ridge_scale)
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError("ridge_scale must be finite and > 0")

    ids = _ids(class_ids, name="class_ids")
    z, y = _validate_feature_rows(decision_features, labels)
    info = accumulate_class_conditioned_information(z, y, class_ids=ids)
    counts = info["counts"]
    sums = info["class_sums"]
    columns = info["class_columns"]

    n, d = z.shape
    c = len(ids)
    if class_balanced:
        # Equal total mass per class while preserving total effective mass N.
        class_mass = float(n) / float(c)
        sample_weight = class_mass / counts.index_select(0, columns).to(z.dtype)
    else:
        sample_weight = torch.ones(n, device=z.device, dtype=z.dtype)

    weighted_z = z * sample_weight.unsqueeze(1)
    gram = z.transpose(0, 1).matmul(weighted_z)

    one_hot = F.one_hot(columns, num_classes=c).to(dtype=z.dtype)
    cross = z.transpose(0, 1).matmul(one_hot * sample_weight.unsqueeze(1))

    # Scale the ridge to the observed system magnitude instead of hard-coding a
    # dataset-dependent absolute lambda.  No test labels are used to tune it.
    diag_mean = gram.diagonal().mean().abs().clamp_min(torch.finfo(z.dtype).eps)
    ridge = float((diag_mean * scale).detach().item())
    system = gram + diag_mean * scale * torch.eye(d, device=z.device, dtype=z.dtype)

    # Cholesky is stable for the positive-definite ridge system.  Fallback to a
    # generic solve if numerical round-off prevents factorization.
    chol, info_code = torch.linalg.cholesky_ex(system)
    if int(info_code.max().item()) == 0:
        coefficients = torch.cholesky_solve(cross, chol)
    else:
        coefficients = torch.linalg.solve(system, cross)

    residual = system.matmul(coefficients) - cross
    relative_residual = float(
        (residual.norm() / cross.norm().clamp_min(torch.finfo(z.dtype).eps)).detach().item()
    )
    weight = coefficients.transpose(0, 1).contiguous()
    eps = torch.finfo(weight.dtype).eps
    if bool((weight.norm(p=2, dim=1) <= eps).any()):
        raise RuntimeError("ridge reconstruction produced a zero-norm class row")
    weight = F.normalize(weight, p=2, dim=1, eps=eps)
    if not bool(torch.isfinite(weight).all()):
        raise RuntimeError("ridge reconstruction produced NaN/Inf")

    return OracleHeadSolution(
        class_ids=ids,
        weight=weight.detach(),
        counts=counts.detach(),
        class_sums=sums.detach(),
        gram=gram.detach(),
        cross=cross.detach(),
        ridge=ridge,
        class_balanced=bool(class_balanced),
        relative_residual=relative_residual,
    )


def _subset_metrics(
    prediction: Tensor,
    labels: Tensor,
    *,
    target_class_ids: Sequence[int],
) -> Dict[str, Any]:
    ids = _ids(target_class_ids, name="target_class_ids")
    pred = torch.as_tensor(prediction).detach().cpu().to(torch.long).flatten()
    y = torch.as_tensor(labels).detach().cpu().to(torch.long).flatten()
    if pred.shape != y.shape or pred.numel() == 0:
        raise ValueError("prediction/labels must be aligned non-empty vectors")

    mask = torch.zeros_like(y, dtype=torch.bool)
    for class_id in ids:
        mask |= y.eq(class_id)
    if not bool(mask.any()):
        raise RuntimeError("target subset has no evaluation rows")
    pred = pred[mask]
    y = y[mask]

    recalls: Dict[int, float] = {}
    support: Dict[int, int] = {}
    for class_id in ids:
        class_mask = y.eq(class_id)
        count = int(class_mask.sum().item())
        if count <= 0:
            raise RuntimeError(f"evaluation subset lacks class {class_id}")
        support[class_id] = count
        recalls[class_id] = float(pred[class_mask].eq(class_id).float().mean().item())

    values = list(recalls.values())
    return {
        "overall_accuracy": float(pred.eq(y).float().mean().item()),
        "balanced_accuracy": float(sum(values) / len(values)),
        "minimum_class_accuracy": float(min(values)),
        "per_class_accuracy": {str(k): v for k, v in recalls.items()},
        "support": {str(k): v for k, v in support.items()},
    }


def evaluate_oracle_solution(
    solution: OracleHeadSolution,
    decision_features: Tensor,
    labels: Tensor,
    *,
    old_class_ids: Sequence[int] | None = None,
    new_class_ids: Sequence[int] | None = None,
) -> Dict[str, Any]:
    z, y = _validate_feature_rows(decision_features, labels)
    if z.size(1) != solution.weight.size(1):
        raise ValueError("oracle head and decision feature dimensions disagree")
    scores = z.matmul(solution.weight.transpose(0, 1))
    column_prediction = scores.argmax(dim=1)
    ids = torch.tensor(solution.class_ids, device=z.device, dtype=torch.long)
    prediction = ids.index_select(0, column_prediction)

    report: Dict[str, Any] = {
        "all_seen": _subset_metrics(prediction, y, target_class_ids=solution.class_ids),
        "score_margin_mean": None,
        "score_margin_min": None,
    }
    true_columns = _class_columns(y, solution.class_ids)
    true_scores = scores.gather(1, true_columns.unsqueeze(1)).squeeze(1)
    masked = scores.clone()
    masked.scatter_(1, true_columns.unsqueeze(1), float("-inf"))
    rival = masked.max(dim=1).values
    margin = true_scores - rival
    report["score_margin_mean"] = float(margin.mean().item())
    report["score_margin_min"] = float(margin.min().item())
    report["positive_margin_fraction"] = float(margin.gt(0).float().mean().item())

    if old_class_ids:
        report["old"] = _subset_metrics(prediction, y, target_class_ids=old_class_ids)
    if new_class_ids:
        report["new"] = _subset_metrics(prediction, y, target_class_ids=new_class_ids)
    if old_class_ids and new_class_ids:
        old_ba = float(report["old"]["balanced_accuracy"])
        new_ba = float(report["new"]["balanced_accuracy"])
        report["old_new_harmonic_mean"] = (
            0.0 if old_ba + new_ba <= 0.0 else 2.0 * old_ba * new_ba / (old_ba + new_ba)
        )
    return report


@torch.no_grad()
def _current_new_only_from_rows(
    trainer: Any,
    rows: Mapping[str, Tensor],
    *,
    new_class_ids: Sequence[int],
) -> Dict[str, Any]:
    """Evaluate the deployed head on new-class rows with new classes competing only.

    The cumulative reporting loader deliberately contains every seen class.
    ``utils.eval.evaluate_loader`` correctly rejects such a loader when
    ``class_ids`` contains only the new classes, because old labels would be
    outside the requested classifier domain.  For the NewOnly diagnostic we
    therefore filter the *already collected, read-only test rows* to new targets
    first and only then run the exact deployed classifier over the new-class
    columns.  No model state is modified.
    """
    ids = _ids(new_class_ids, name="new_class_ids")
    required = {"ordered_features", "raw_center_spectra", "labels"}
    missing = required - set(rows)
    if missing:
        raise KeyError(f"current-head rows missing {sorted(missing)}")

    q = torch.as_tensor(rows["ordered_features"])
    spectra = torch.as_tensor(
        rows["raw_center_spectra"], device=q.device, dtype=q.dtype
    )
    labels = torch.as_tensor(rows["labels"], device=q.device, dtype=torch.long).flatten()
    if q.ndim != 3 or q.size(0) == 0:
        raise ValueError("ordered_features must be non-empty [N,K,D]")
    if spectra.ndim != 2 or spectra.size(0) != q.size(0):
        raise ValueError("raw_center_spectra must align with ordered_features")
    if labels.shape != (q.size(0),):
        raise ValueError("labels must align with ordered_features")

    new_tensor = torch.tensor(ids, device=labels.device, dtype=torch.long)
    mask = torch.isin(labels, new_tensor)
    if not bool(mask.any()):
        raise RuntimeError("cumulative test rows contain no new-class samples")
    selected_labels = labels[mask]
    observed = tuple(
        sorted(int(v) for v in selected_labels.unique().detach().cpu().tolist())
    )
    if observed != tuple(sorted(ids)):
        raise RuntimeError(
            f"new-only test subset has classes {observed}, expected {tuple(sorted(ids))}"
        )

    model = trainer.model
    q_new = q[mask]
    spectra_new = spectra[mask]
    physical = model.physical_spectral_evidence(spectra_new)
    output = model.classify_features(
        q_new,
        physical_spectral_evidence=physical,
        class_ids=ids,
        trainable_class_ids=(),
    )
    return _subset_metrics(
        output.prediction,
        selected_labels,
        target_class_ids=ids,
    )


def _current_head_report(
    trainer: Any,
    loader: Any,
    rows: Mapping[str, Tensor],
    *,
    phase: int,
) -> Dict[str, Any]:
    dataset = trainer.dataset
    seen = _ids(dataset.get_seen_classes(phase), name="seen_classes")
    old = tuple(int(v) for v in dataset.get_old_classes(phase))
    new = _ids(dataset.get_new_classes(phase), name="new_classes")

    # AllSeen/OldAll/NewAll legitimately use the cumulative loader because every
    # target label is inside ``seen``.  NewOnly is different: the classifier
    # domain is restricted to ``new``, so old rows must be excluded first.
    report: Dict[str, Any] = {
        "all_seen": trainer.evaluate_loader(
            loader, class_ids=seen, target_class_ids=seen
        )
    }
    if old:
        report["old_all_seen"] = trainer.evaluate_loader(
            loader, class_ids=seen, target_class_ids=old
        )
    report["new_all_seen"] = trainer.evaluate_loader(
        loader, class_ids=seen, target_class_ids=new
    )
    report["new_only"] = _current_new_only_from_rows(
        trainer,
        rows,
        new_class_ids=new,
    )
    return report


def _compact_model_metrics(metrics: Mapping[str, Any]) -> Dict[str, Any]:
    keep = (
        "overall_accuracy",
        "balanced_accuracy",
        "minimum_class_accuracy",
        "per_class_accuracy",
        "confusion_matrix",
    )
    return {name: metrics[name] for name in keep if name in metrics}


def run_oracle_phase_test(
    trainer: Any,
    *,
    phase: int,
    batch_size: int = 256,
    ridge_scale: float = 1.0e-3,
) -> Dict[str, Any]:
    """Run Test 1 for phase=0 or Test 2 for phase>0 on a finalized checkpoint."""
    phase_id = int(phase)
    dataset = trainer.dataset
    if getattr(dataset, "current_phase", None) is not None:
        raise RuntimeError("oracle test requires a finalized, non-training state")
    finalized = [int(v) for v in getattr(dataset, "finalized_phases", [])]
    if phase_id not in finalized:
        raise RuntimeError(
            f"phase {phase_id} is not finalized in this trainer; finalized={finalized}"
        )

    train_loader = dataset.get_oracle_diagnostic_train_dataloader(
        phase_id,
        batch_size=batch_size,
        authorization=ORACLE_AUTHORIZATION,
    )
    test_loader = dataset.get_reporting_dataloader(
        phase_id,
        batch_size=batch_size,
    )

    train_rows = trainer.collect_decision_features(train_loader)
    test_rows = trainer.collect_decision_features(test_loader)
    seen = _ids(dataset.get_seen_classes(phase_id), name="seen_classes")
    old = tuple(int(v) for v in dataset.get_old_classes(phase_id))
    new = _ids(dataset.get_new_classes(phase_id), name="new_classes")

    current = _current_head_report(
        trainer, test_loader, test_rows, phase=phase_id
    )

    solutions: Dict[str, Any] = {}
    for name, balanced in (("class_balanced", True), ("raw_frequency", False)):
        solution = solve_ridge_oracle_head(
            train_rows["decision_features"],
            train_rows["labels"],
            class_ids=seen,
            ridge_scale=ridge_scale,
            class_balanced=balanced,
        )
        metrics = evaluate_oracle_solution(
            solution,
            test_rows["decision_features"],
            test_rows["labels"],
            old_class_ids=old or None,
            new_class_ids=new,
        )
        solutions[name] = {
            "ridge": float(solution.ridge),
            "relative_linear_system_residual": float(solution.relative_residual),
            "class_balanced": bool(solution.class_balanced),
            "class_counts": {
                str(class_id): int(solution.counts[index].item())
                for index, class_id in enumerate(solution.class_ids)
            },
            "class_sum_norms": {
                str(class_id): float(solution.class_sums[index].norm().item())
                for index, class_id in enumerate(solution.class_ids)
            },
            "metrics": metrics,
        }

    compact_current: Dict[str, Any] = {}
    for key, value in current.items():
        compact_current[key] = _compact_model_metrics(value)

    primary = solutions["class_balanced"]["metrics"]
    comparison: Dict[str, Any] = {
        "oracle_minus_current_all_seen_ba": float(
            primary["all_seen"]["balanced_accuracy"]
            - compact_current["all_seen"]["balanced_accuracy"]
        ),
    }
    if phase_id > 0 and old:
        current_new_all = float(compact_current["new_all_seen"]["balanced_accuracy"])
        current_new_only = float(compact_current["new_only"]["balanced_accuracy"])
        oracle_new = float(primary["new"]["balanced_accuracy"])
        denominator = current_new_only - current_new_all
        comparison.update({
            "oracle_minus_current_old_all_seen_ba": float(
                primary["old"]["balanced_accuracy"]
                - compact_current["old_all_seen"]["balanced_accuracy"]
            ),
            "oracle_minus_current_new_all_seen_ba": float(
                oracle_new - current_new_all
            ),
            "current_new_only_minus_all_seen_gap": float(denominator),
            "oracle_new_gap_closure_fraction": (
                None
                if denominator <= 0.0
                else float((oracle_new - current_new_all) / denominator)
            ),
        })

    return {
        "test": "phase0_base_analytic_head" if phase_id == 0 else "phase1_oracle_all_seen_head",
        "phase": phase_id,
        "diagnostic_only": True,
        "oracle_uses_historical_training_data": bool(phase_id > 0),
        "valid_for_method_reporting": False,
        "purpose": (
            "falsify whether the frozen representation supports a globally balanced "
            "analytic head before adding persistent decision statistics"
        ),
        "seen_class_ids": list(seen),
        "old_class_ids": list(old),
        "new_class_ids": list(new),
        "train_rows": int(train_rows["labels"].numel()),
        "test_rows": int(test_rows["labels"].numel()),
        "feature_dimension": int(train_rows["decision_features"].size(1)),
        "ridge_scale": float(ridge_scale),
        "current_head": compact_current,
        "oracle": solutions,
        "comparison": comparison,
    }


def run_test1_test2_suite(
    *,
    base_trainer: Any,
    phase1_trainer: Any,
    batch_size: int = 256,
    ridge_scale: float = 1.0e-3,
    output_json: str | None = None,
) -> Dict[str, Any]:
    """Run Test 1 and Test 2 together using separately loaded finalized trainers."""
    test1 = run_oracle_phase_test(
        base_trainer,
        phase=0,
        batch_size=batch_size,
        ridge_scale=ridge_scale,
    )
    test2 = run_oracle_phase_test(
        phase1_trainer,
        phase=1,
        batch_size=batch_size,
        ridge_scale=ridge_scale,
    )
    report = {
        "diagnostic_only": True,
        "valid_for_method_reporting": False,
        "test1": test1,
        "test2": test2,
    }
    if output_json:
        path = Path(output_json).expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as stream:
            json.dump(report, stream, indent=2, sort_keys=True)
        report["output_json"] = str(path)
    return report


__all__ = [
    "ORACLE_AUTHORIZATION",
    "OracleHeadSolution",
    "accumulate_class_conditioned_information",
    "solve_ridge_oracle_head",
    "evaluate_oracle_solution",
    "run_oracle_phase_test",
    "run_test1_test2_suite",
]
