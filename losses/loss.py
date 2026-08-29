from __future__ import annotations

"""Classification objective for the SCOPE-HSI base phase.

Role
----
Phase 0 optimizes only the single HSI representation Q and the persistent
angular classifier.  Historical class statistics {I_c,S_c,N_c} are absent from
this objective and are committed only after the final requested epoch.

The model supplies canonical

    Q(x) in R^{K x D},  RMS(Q)=1,

and the classifier produces

    l_c(x) = vec(Q)^T normalize(w_c) = sqrt(KD) cos(theta_c).

The deployed logits are used directly by cross entropy.  There is no margin,
temperature, extra logit scale, geometry energy, preservation term, replay, or
teacher term.

For an imbalanced HSI training set, ``class_balance_weights`` may contain the
fixed phase-level empirical weights

    a_c = N / (C N_c),

in classifier-column order.  Then the minibatch loss

    mean_i a_{y_i} CE_i

is an unbiased minibatch estimator of the empirical macro-average class CE
under ordinary uniform sample shuffling.  No tunable balancing coefficient is
introduced.
"""

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from models.classifier import ClassifierOutput

Tensor = torch.Tensor


@dataclass(frozen=True)
class ClassificationObjective:
    loss: Tensor
    cross_entropy: Tensor
    accuracy: Tensor
    balanced_accuracy: Tensor
    targets_local: Tensor
    mean_cosine_margin: Tensor


def _validate_class_ids(class_ids: Tensor, *, columns: int, device: torch.device) -> Tensor:
    ids = torch.as_tensor(class_ids)
    if ids.ndim != 1 or ids.shape != (columns,):
        raise ValueError(f"class_ids must have shape [{columns}]")
    if ids.dtype != torch.long:
        raise TypeError("class_ids must use torch.long")
    if ids.device != device:
        raise ValueError("class_ids and logits must share a device")
    if bool((ids < 0).any()) or ids.unique().numel() != ids.numel():
        raise ValueError("class_ids must contain unique non-negative IDs")
    return ids


def _targets_local(labels_global: Tensor, class_ids: Tensor) -> Tensor:
    labels = torch.as_tensor(labels_global)
    if labels.ndim != 1 or labels.numel() == 0:
        raise ValueError("labels_global must be non-empty rank-one")
    if labels.dtype != torch.long:
        raise TypeError("labels_global must use torch.long")
    if labels.device != class_ids.device:
        raise ValueError("labels_global and class_ids must share a device")

    matches = labels.unsqueeze(1).eq(class_ids.unsqueeze(0))
    counts = matches.sum(dim=1)
    if not bool(counts.eq(1).all()):
        missing = labels[counts.eq(0)].detach().cpu().unique().tolist()
        raise ValueError(f"labels contain class IDs absent from classifier: {missing}")
    return matches.to(torch.long).argmax(dim=1)


def _balanced_accuracy(pred_local: Tensor, targets_local: Tensor) -> Tensor:
    represented = targets_local.unique(sorted=True)
    if represented.numel() == 0:
        raise RuntimeError("no represented target classes")
    return torch.stack(
        [
            pred_local[targets_local.eq(class_index)]
            .eq(targets_local[targets_local.eq(class_index)])
            .to(torch.float32)
            .mean()
            for class_index in represented
        ]
    ).mean()


def cosine_classification_objective(
    *,
    output: ClassifierOutput,
    labels_global: Tensor,
    class_balance_weights: Tensor | None = None,
) -> ClassificationObjective:
    """Compute deployed-logit CE and read-only angular diagnostics.

    ``class_balance_weights`` must be one positive weight per classifier column.
    The base trainer derives these once from the complete phase-0 TRAIN split.
    Passing ``None`` gives ordinary sample-mean cross entropy.
    """
    if not isinstance(output, ClassifierOutput):
        raise TypeError("output must be ClassifierOutput")

    logits = torch.as_tensor(output.logits)
    cosine = torch.as_tensor(output.cosine_similarity)
    prediction = torch.as_tensor(output.prediction)

    if logits.ndim != 2 or logits.size(0) == 0 or logits.size(1) < 2:
        raise ValueError("logits must be non-empty [N,C] with C>=2")
    if not torch.is_floating_point(logits) or logits.is_complex():
        raise TypeError("logits must be real floating point")
    if not bool(torch.isfinite(logits).all()):
        raise ValueError("logits contain NaN or Inf")

    if cosine.shape != logits.shape:
        raise RuntimeError("cosine_similarity must match logits shape")
    if cosine.device != logits.device or cosine.dtype != logits.dtype:
        raise RuntimeError("cosine_similarity must share logits device/dtype")
    if not torch.is_floating_point(cosine) or cosine.is_complex():
        raise TypeError("cosine_similarity must be real floating point")
    if not bool(torch.isfinite(cosine).all()):
        raise ValueError("cosine_similarity contains NaN or Inf")

    class_ids = _validate_class_ids(
        output.class_ids,
        columns=logits.size(1),
        device=logits.device,
    )

    if prediction.shape != (logits.size(0),) or prediction.dtype != torch.long:
        raise RuntimeError("prediction must contain one torch.long class ID per row")
    if prediction.device != logits.device:
        raise RuntimeError("prediction and logits must share a device")

    labels = torch.as_tensor(labels_global)
    if labels.shape != (logits.size(0),):
        raise ValueError("labels_global must contain one rank-one label per sample")
    if labels.dtype != torch.long:
        raise TypeError("labels_global must use torch.long")
    if labels.device != logits.device:
        raise ValueError("labels_global and logits must share a device")

    targets_local = _targets_local(labels, class_ids)

    # The classifier contract is: logits are a positive class-independent scale
    # of cosine similarity.  Their class ordering must therefore be identical.
    pred_local = logits.argmax(dim=1)
    cosine_pred_local = cosine.argmax(dim=1)
    if not bool(torch.equal(pred_local, cosine_pred_local)):
        raise RuntimeError("logit and cosine class orderings disagree")
    expected_prediction = class_ids.index_select(0, pred_local)
    if not bool(torch.equal(prediction, expected_prediction)):
        raise RuntimeError("ClassifierOutput.prediction disagrees with deployed logits")

    per_sample_ce = F.cross_entropy(logits, targets_local, reduction="none")

    if class_balance_weights is None:
        per_sample_objective = per_sample_ce
    else:
        weights = torch.as_tensor(
            class_balance_weights,
            device=logits.device,
            dtype=logits.dtype,
        )
        if weights.shape != (logits.size(1),):
            raise ValueError(
                "class_balance_weights must contain one value per classifier column"
            )
        if not bool(torch.isfinite(weights).all()) or bool((weights <= 0).any()):
            raise ValueError("class_balance_weights must be finite and positive")
        per_sample_objective = per_sample_ce * weights.index_select(0, targets_local)

    cross_entropy = per_sample_objective.mean()
    if not bool(torch.isfinite(cross_entropy)):
        raise RuntimeError("classification cross entropy became NaN or Inf")

    accuracy = prediction.eq(labels).to(torch.float32).mean()
    balanced_accuracy = _balanced_accuracy(pred_local, targets_local)

    rows = torch.arange(cosine.size(0), device=cosine.device)
    target_score = cosine[rows, targets_local]
    rival_score = cosine.clone()
    rival_score[rows, targets_local] = -torch.inf
    mean_margin = (target_score - rival_score.max(dim=1).values).mean()
    if not bool(torch.isfinite(mean_margin)):
        raise RuntimeError("cosine margin became NaN or Inf")

    return ClassificationObjective(
        loss=cross_entropy,
        cross_entropy=cross_entropy,
        accuracy=accuracy,
        balanced_accuracy=balanced_accuracy,
        targets_local=targets_local,
        mean_cosine_margin=mean_margin,
    )


__all__ = ["ClassificationObjective", "cosine_classification_objective"]
