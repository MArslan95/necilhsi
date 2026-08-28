from __future__ import annotations

"""Classification objective for the clean NECIL-HSI base phase.

The geometry bank does not participate in this loss.  The persistent classifier
receives canonical Q with RMS(Q)=1 and produces

    l_c(x)=vec(Q)^T normalize(w_c)=sqrt(KD) cos(theta_c).

The base objective is class-balanced multiclass cross entropy on these deployed
logits:

    L_cls = CE(l(x), y).

There is no margin, temperature, learned/hand-chosen logit scale, structural
energy, geometry loss, replay term, or preservation term in phase 0.  Because
sqrt(KD) is positive and class-independent, training and inference use the same
class ordering as cosine similarity.
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


def _targets_local(labels_global: Tensor, class_ids: Tensor) -> Tensor:
    labels = torch.as_tensor(labels_global, device=class_ids.device).flatten()
    if labels.numel() == 0 or labels.dtype != torch.long:
        raise TypeError("labels_global must be non-empty rank-one torch.long")
    matches = labels.unsqueeze(1).eq(class_ids.unsqueeze(0))
    counts = matches.sum(dim=1)
    if not bool(counts.eq(1).all()):
        missing = labels[counts.eq(0)].detach().cpu().unique().tolist()
        raise ValueError(f"labels contain class IDs absent from classifier: {missing}")
    return matches.to(torch.long).argmax(dim=1)


def _class_balanced_mean(per_sample: Tensor, targets_local: Tensor) -> Tensor:
    if per_sample.ndim != 1 or per_sample.shape != targets_local.shape:
        raise ValueError("per_sample and targets_local must be matching [N]")
    represented = targets_local.unique(sorted=True)
    if represented.numel() == 0:
        raise RuntimeError("no represented target classes")
    return torch.stack(
        [per_sample[targets_local.eq(class_id)].mean() for class_id in represented]
    ).mean()


def cosine_classification_objective(
    *,
    output: ClassifierOutput,
    labels_global: Tensor,
) -> ClassificationObjective:
    if not isinstance(output, ClassifierOutput):
        raise TypeError("output must be ClassifierOutput")

    logits = torch.as_tensor(output.logits)
    cosine = torch.as_tensor(output.cosine_similarity)
    class_ids = torch.as_tensor(output.class_ids)
    prediction = torch.as_tensor(output.prediction)
    if logits.ndim != 2 or logits.size(0) == 0 or logits.size(1) < 2:
        raise ValueError("logits must be non-empty [N,C] with C>=2")
    if cosine.shape != logits.shape:
        raise RuntimeError("cosine_similarity must match logits shape")
    if not torch.is_floating_point(logits) or logits.is_complex():
        raise TypeError("logits must be real floating point")
    if not torch.is_floating_point(cosine) or cosine.is_complex():
        raise TypeError("cosine_similarity must be real floating point")
    if not bool(torch.isfinite(logits).all()):
        raise ValueError("logits contain NaN or Inf")
    if not bool(torch.isfinite(cosine).all()):
        raise ValueError("cosine_similarity contains NaN or Inf")
    if class_ids.shape != (logits.size(1),) or class_ids.dtype != torch.long:
        raise RuntimeError("class_ids must contain one torch.long ID per column")
    if prediction.shape != (logits.size(0),) or prediction.dtype != torch.long:
        raise RuntimeError("prediction must contain one torch.long class ID per row")

    labels = torch.as_tensor(
        labels_global,
        device=logits.device,
    ).flatten().to(torch.long)
    if labels.shape != (logits.size(0),):
        raise ValueError("labels_global must contain one label per sample")

    targets_local = _targets_local(labels, class_ids)
    per_sample_ce = F.cross_entropy(
        logits,
        targets_local,
        reduction="none",
    )
    cross_entropy = _class_balanced_mean(per_sample_ce, targets_local)
    if not bool(torch.isfinite(cross_entropy)):
        raise RuntimeError("classification cross entropy became NaN or Inf")

    pred_local = cosine.argmax(dim=1)
    accuracy = pred_local.eq(targets_local).to(torch.float32).mean()
    represented = targets_local.unique(sorted=True)
    balanced_accuracy = torch.stack(
        [
            pred_local[targets_local.eq(class_id)]
            .eq(targets_local[targets_local.eq(class_id)])
            .to(torch.float32)
            .mean()
            for class_id in represented
        ]
    ).mean()

    rows = torch.arange(cosine.size(0), device=cosine.device)
    target_score = cosine[rows, targets_local]
    rival_score = cosine.clone()
    rival_score[rows, targets_local] = -torch.inf
    mean_margin = (target_score - rival_score.max(dim=1).values).mean()

    return ClassificationObjective(
        loss=cross_entropy,
        cross_entropy=cross_entropy,
        accuracy=accuracy,
        balanced_accuracy=balanced_accuracy,
        targets_local=targets_local,
        mean_cosine_margin=mean_margin,
    )


__all__ = ["ClassificationObjective", "cosine_classification_objective"]







# from __future__ import annotations

# """Base training objective for the HSI class-discriminative classifier.

# The classifier provides, for every query x and class c,

#     E_c(x)
#       = E_identity(c,x)
#       + E_feature(c,x)
#       + E_cross(c,x)
#       >= 0,

# with deployed prediction

#     y_hat = argmin_c E_c(x).

# The classifier currently exposes

#     score_c(x) = -E_c(x),

# but ordinary CE(-E, y) is sensitive to a common positive rescaling of all
# energies.  If the backbone inflates the global feature scale, all energies can
# grow together while the class ordering stays unchanged, yet CE(-E, y) can
# become artificially sharper.

# Base training therefore uses the strictly monotonic relative-energy logits

#     l_c(x) = -log E_c(x).

# Because -log(.) is strictly decreasing,

#     argmax_c l_c(x) = argmin_c E_c(x),

# so the training rule preserves the deployed classifier ordering.

# For strictly positive energies and exact arithmetic, one sample with target y
# has

#     L_i
#       = CE(-log E, y)
#       = log(1 + sum_{d != y} E_y / E_d).

# The implementation uses the dtype-relative floor described below only to define
# the logarithm numerically at zero energy.

# Thus the objective directly reduces the target energy relative to every rival
# energy.  A common positive rescaling

#     E_c -> a E_c

# adds the same constant -log(a) to all logits and leaves the per-sample loss
# unchanged.

# HSI class frequencies can be strongly imbalanced.  The batch objective therefore
# averages the per-sample relative-energy loss within each represented target class
# first, then averages those class means:

#     L_base
#       = (1 / |C_B|) sum_{c in C_B}
#           (1 / N_{B,c}) sum_{i:y_i=c} L_i.

# This changes no class score and introduces no class-specific learned or tuned
# weight; it only prevents a frequent class from dominating the base
# discrimination objective.

# No margin, temperature, class-specific scale, auxiliary head, prototype loss,
# pairwise boundary loss, replay term, calibration term, or manually weighted
# classifier component is introduced.

# Decision overlap is measured by the exact deployed ordering:

#     E_nearest_rival <= E_target.

# The scale-invariant separation diagnostic is

#     log(E_nearest_rival / E_target).

# To evaluate log(E) when one candidate energy is exactly zero, the code adds
# the same dtype-relative numerical floor to every class energy in that sample:

#     E_safe = E + eps_dtype * mean_c(E_c).

# The floor scales with the sample energies, so E -> aE implies
# E_safe -> a E_safe.  It therefore preserves both class ordering and the exact
# common-scale invariance of the objective.  ``eps_dtype`` is machine precision,
# not a model hyperparameter.  A row for which every class energy is exactly zero
# is rejected because the classifier itself contains no decision information.
# """

# from dataclasses import dataclass

# import torch
# import torch.nn.functional as F

# from models.classifier import GeometryClassifierOutput

# Tensor = torch.Tensor


# @dataclass(frozen=True)
# class ClassDiscriminativeStructureObjective:
#     """Base classification loss and structural-overlap diagnostics."""

#     total: Tensor
#     cross_entropy: Tensor
#     accuracy: Tensor
#     balanced_accuracy: Tensor
#     relative_energy_probabilities: Tensor
#     log_energy_margin: Tensor
#     decision_overlap: Tensor
#     mean_log_energy_margin: Tensor
#     decision_overlap_rate: Tensor
#     targets_local: Tensor
#     class_ids: Tensor


# def _validate_classifier_output(
#     output: GeometryClassifierOutput,
# ) -> GeometryClassifierOutput:
#     """Validate the classifier contract required by this objective."""
#     if not isinstance(
#         output,
#         GeometryClassifierOutput,
#     ):
#         raise TypeError(
#             "output must be GeometryClassifierOutput produced by GeometryClassifier"
#         )

#     scores = output.scores

#     if (
#         scores.ndim != 2
#         or scores.size(0) == 0
#         or scores.size(1) < 2
#     ):
#         raise ValueError(
#             "classifier scores must be non-empty [N,C] with C>=2"
#         )

#     if (
#         not torch.is_floating_point(scores)
#         or scores.is_complex()
#         or scores.dtype not in (
#             torch.float32,
#             torch.float64,
#         )
#         or not bool(
#             torch.isfinite(scores).all()
#         )
#     ):
#         raise TypeError(
#             "classifier scores must be finite float32 or float64"
#         )

#     expected_shape = scores.shape

#     energy_tensors = (
#         output.total_energy,
#         output.identity_energy,
#         output.feature_variation_energy,
#         output.cross_region_covariation_energy,
#     )

#     for value in energy_tensors:
#         if value.shape != expected_shape:
#             raise ValueError(
#                 "all classifier energy tensors must have shape [N,C]"
#             )

#         if (
#             not torch.is_floating_point(value)
#             or value.is_complex()
#             or value.dtype != scores.dtype
#             or value.device != scores.device
#             or not bool(
#                 torch.isfinite(value).all()
#             )
#         ):
#             raise ValueError(
#                 "classifier energies must be finite real tensors "
#                 "with the same dtype/device as scores"
#             )

#         if bool(
#             (value < 0).any()
#         ):
#             raise ValueError(
#                 "classifier energies cannot be negative"
#             )

#     if not bool(
#         torch.equal(
#             scores,
#             -output.total_energy,
#         )
#     ):
#         raise ValueError(
#             "classifier scores must equal negative total energy"
#         )

#     class_ids = output.class_ids

#     if (
#         class_ids.ndim != 1
#         or class_ids.shape != (
#             scores.size(1),
#         )
#         or class_ids.dtype != torch.long
#         or class_ids.device != scores.device
#         or class_ids.unique().numel()
#         != class_ids.numel()
#         or bool(
#             (class_ids < 0).any()
#         )
#     ):
#         raise ValueError(
#             "class_ids must contain one unique non-negative torch.long "
#             "ID per classifier column"
#         )

#     prediction = output.prediction

#     if (
#         prediction.ndim != 1
#         or prediction.shape != (
#             scores.size(0),
#         )
#         or prediction.dtype != torch.long
#         or prediction.device != scores.device
#     ):
#         raise ValueError(
#             "prediction must be [N] torch.long on the classifier device"
#         )

#     expected_prediction = class_ids.index_select(
#         0,
#         output.total_energy.argmin(
#             dim=1
#         ),
#     )

#     if not bool(
#         torch.equal(
#             prediction,
#             expected_prediction,
#         )
#     ):
#         raise ValueError(
#             "prediction must equal argmin of total structural energy"
#         )

#     return output


# def _validate_labels(
#     labels_global: Tensor,
#     *,
#     rows: int,
#     device: torch.device,
# ) -> Tensor:
#     labels = torch.as_tensor(
#         labels_global
#     )

#     if (
#         labels.ndim != 1
#         or labels.shape != (
#             rows,
#         )
#     ):
#         raise ValueError(
#             f"labels_global must have shape [{rows}]"
#         )

#     if labels.dtype != torch.long:
#         raise TypeError(
#             "labels_global must use torch.long"
#         )

#     if labels.device != device:
#         raise ValueError(
#             "labels_global and classifier output must share a device"
#         )

#     if bool(
#         (labels < 0).any()
#     ):
#         raise ValueError(
#             "labels_global must contain non-negative class IDs"
#         )

#     return labels


# def _targets_local(
#     labels_global: Tensor,
#     class_ids: Tensor,
# ) -> Tensor:
#     """Map global labels to classifier columns."""
#     matches = labels_global.unsqueeze(
#         1
#     ).eq(
#         class_ids.unsqueeze(
#             0
#         )
#     )

#     match_count = matches.sum(
#         dim=1
#     )

#     if not bool(
#         match_count.eq(
#             1
#         ).all()
#     ):
#         missing = (
#             labels_global[
#                 match_count.eq(
#                     0
#                 )
#             ]
#             .unique()
#             .detach()
#             .cpu()
#             .tolist()
#         )

#         raise ValueError(
#             "every target label must occur exactly once in classifier class_ids; "
#             f"unmapped labels={missing}"
#         )

#     return (
#         matches
#         .to(
#             dtype=torch.long
#         )
#         .argmax(
#             dim=1
#         )
#     )


# def _balanced_accuracy(
#     prediction_local: Tensor,
#     targets_local: Tensor,
# ) -> Tensor:
#     """Mean recall over target classes represented in this batch."""
#     observed_classes = targets_local.unique(
#         sorted=True
#     )

#     if observed_classes.numel() == 0:
#         raise RuntimeError(
#             "no target classes are represented"
#         )

#     recalls: list[Tensor] = []

#     for class_index in observed_classes:
#         mask = targets_local.eq(
#             class_index
#         )

#         recalls.append(
#             prediction_local[
#                 mask
#             ]
#             .eq(
#                 targets_local[
#                     mask
#                 ]
#             )
#             .to(
#                 dtype=torch.float32
#             )
#             .mean()
#         )

#     return torch.stack(
#         recalls
#     ).mean()


# def _class_balanced_mean(
#     per_sample_loss: Tensor,
#     targets_local: Tensor,
# ) -> Tensor:
#     """Average loss per represented target class, then average classes."""
#     if (
#         per_sample_loss.ndim != 1
#         or targets_local.ndim != 1
#         or per_sample_loss.shape != targets_local.shape
#         or per_sample_loss.numel() == 0
#     ):
#         raise ValueError(
#             "per_sample_loss and targets_local must be matching non-empty [N]"
#         )

#     observed_classes = targets_local.unique(
#         sorted=True
#     )

#     class_means: list[Tensor] = []

#     for class_index in observed_classes:
#         mask = targets_local.eq(
#             class_index
#         )

#         class_means.append(
#             per_sample_loss[
#                 mask
#             ].mean()
#         )

#     return torch.stack(
#         class_means
#     ).mean()


# def _safe_log_energy(
#     energy: Tensor,
# ) -> Tensor:
#     """Return a scale-invariant numerical log of non-negative energies.

#     The same relative floor is added to every class in one sample, so neither
#     the argmin class nor invariance to E -> aE is changed.
#     """
#     if (
#         energy.ndim != 2
#         or energy.size(0) == 0
#         or energy.size(1) < 2
#     ):
#         raise ValueError(
#             "energy must be non-empty [N,C] with C>=2"
#         )

#     if bool(
#         (energy < 0).any()
#     ):
#         raise ValueError(
#             "energy cannot contain negative values"
#         )

#     row_scale = energy.mean(
#         dim=1,
#         keepdim=True,
#     )

#     zero_rows = row_scale.eq(
#         0
#     )

#     if bool(
#         zero_rows.any()
#     ):
#         rows = (
#             torch.nonzero(
#                 zero_rows.squeeze(1),
#                 as_tuple=False,
#             )
#             .flatten()
#             .detach()
#             .cpu()
#             .tolist()
#         )

#         raise RuntimeError(
#             "all class energies are zero for classifier rows "
#             f"{rows}; relative structural classification is undefined"
#         )

#     # Numerical-only relative floor.  Detaching row_scale keeps the floor out
#     # of the optimization path while preserving exact forward scale invariance.
#     floor = (
#         torch.finfo(
#             energy.dtype
#         ).eps
#         * row_scale.detach()
#     )

#     return torch.log(
#         energy
#         + floor
#     )


# def _target_and_nearest_rival(
#     energy: Tensor,
#     targets_local: Tensor,
# ) -> tuple[Tensor, Tensor, Tensor]:
#     target_energy = energy.gather(
#         1,
#         targets_local.unsqueeze(
#             1
#         ),
#     ).squeeze(
#         1
#     )

#     rivals = energy.clone()

#     rivals.scatter_(
#         1,
#         targets_local.unsqueeze(
#             1
#         ),
#         torch.inf,
#     )

#     nearest_rival = rivals.min(
#         dim=1
#     )
#     nearest_rival_energy = nearest_rival.values
#     nearest_rival_local = nearest_rival.indices

#     if (
#         not bool(
#             torch.isfinite(
#                 target_energy
#             ).all()
#         )
#         or not bool(
#             torch.isfinite(
#                 nearest_rival_energy
#             ).all()
#         )
#     ):
#         raise RuntimeError(
#             "target or nearest-rival energy is invalid"
#         )

#     return (
#         target_energy,
#         nearest_rival_energy,
#         nearest_rival_local,
#     )


# def class_discriminative_structure_objective(
#     *,
#     output: GeometryClassifierOutput,
#     labels_global: Tensor,
# ) -> ClassDiscriminativeStructureObjective:
#     """Optimize relative structural separation with the deployed energy order."""
#     validated = _validate_classifier_output(
#         output
#     )

#     energy = validated.total_energy
#     class_ids = validated.class_ids

#     labels = _validate_labels(
#         labels_global,
#         rows=energy.size(
#             0
#         ),
#         device=energy.device,
#     )

#     targets_local = _targets_local(
#         labels,
#         class_ids,
#     )

#     # Scale-invariant structural logits.  The monotonic transformation preserves
#     # exactly the deployed argmin-energy class ordering.
#     safe_log_energy = _safe_log_energy(
#         energy
#     )
#     training_logits = -safe_log_energy

#     per_sample_cross_entropy = F.cross_entropy(
#         training_logits,
#         targets_local,
#         reduction="none",
#     )

#     cross_entropy = _class_balanced_mean(
#         per_sample_cross_entropy,
#         targets_local,
#     )

#     if not bool(
#         torch.isfinite(
#             cross_entropy
#         )
#     ):
#         raise RuntimeError(
#             "class-discriminative cross-entropy is NaN or Inf"
#         )

#     relative_energy_probabilities = torch.softmax(
#         training_logits,
#         dim=1,
#     )

#     prediction_local = energy.argmin(
#         dim=1
#     )

#     accuracy = (
#         validated.prediction
#         .eq(
#             labels
#         )
#         .to(
#             dtype=torch.float32
#         )
#         .mean()
#     )

#     balanced_accuracy = _balanced_accuracy(
#         prediction_local,
#         targets_local,
#     )

#     (
#         target_energy,
#         nearest_rival_energy,
#         nearest_rival_local,
#     ) = _target_and_nearest_rival(
#         energy,
#         targets_local,
#     )

#     # Exact classifier-boundary diagnostic; no hand-chosen margin.
#     decision_overlap = (
#         nearest_rival_energy
#         .le(
#             target_energy
#         )
#     )

#     decision_overlap_rate = (
#         decision_overlap
#         .to(
#             dtype=torch.float32
#         )
#         .mean()
#     )

#     # Scale-invariant separation under the exact same safe-log values used
#     # to construct the training logits:
#     #
#     #     log(E_rival_safe) - log(E_target_safe).
#     #
#     # Using the already-computed full-class safe_log_energy avoids introducing
#     # a second numerical floor based on a different two-column row mean.
#     rows = torch.arange(
#         energy.size(0),
#         device=energy.device,
#     )
#     log_energy_margin = (
#         safe_log_energy[
#             rows,
#             nearest_rival_local,
#         ]
#         - safe_log_energy[
#             rows,
#             targets_local,
#         ]
#     )

#     if not bool(
#         torch.isfinite(
#             log_energy_margin
#         ).all()
#     ):
#         raise RuntimeError(
#             "log-energy margin contains NaN or Inf"
#         )

#     return ClassDiscriminativeStructureObjective(
#         total=cross_entropy,
#         cross_entropy=cross_entropy,
#         accuracy=accuracy,
#         balanced_accuracy=balanced_accuracy,
#         relative_energy_probabilities=(
#             relative_energy_probabilities
#         ),
#         log_energy_margin=(
#             log_energy_margin
#         ),
#         decision_overlap=(
#             decision_overlap
#         ),
#         mean_log_energy_margin=(
#             log_energy_margin.mean()
#         ),
#         decision_overlap_rate=(
#             decision_overlap_rate
#         ),
#         targets_local=targets_local,
#         class_ids=class_ids,
#     )


# __all__ = [
#     "ClassDiscriminativeStructureObjective",
#     "class_discriminative_structure_objective",
# ]
