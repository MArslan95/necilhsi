"""Stateless historical cosine-decision constraints from one canonical CDS bank.

No historical images or feature exemplars are stored. Optional transient
Gaussian Q draws provide contrastive negatives; no native spectra are generated.
The classifier objective remains the existing directional moment constraint.
"""

from __future__ import annotations
from dataclasses import dataclass
import math
import warnings
import torch
import torch.nn.functional as F
from models.geometry_bank import (
    HSIClassDiscriminativeStructureBank,
    normalize_ordered_decision_features,
    whole_cds_pair_risk,
)
from models.phase_coordinate_alignment import (
    rotate_historical_flat_rows,
    transport_historical_structure,
)

Tensor = torch.Tensor


@dataclass(frozen=True)
class HistoricalSupportRisk:
    loss: Tensor
    per_class_loss: Tensor
    per_class_worst_rival: Tensor
    class_ids: Tensor
    margin_surrogate: Tensor
    per_class_worst_group: Tensor
    positive_margin_fraction: Tensor
    objective: str = "directional"
    mean_ce_loss: Tensor | None = None
    curvature_correction: Tensor | None = None

    @property
    def upper_margin(self):
        return self.margin_surrogate


class HistoricalCDSSupportBank:
    """Compatibility façade; CDS storage belongs exclusively to geometry_bank."""

    def __init__(
        self,
        cds_bank: HSIClassDiscriminativeStructureBank,
        *,
        margin_std: float = 1.0,
        risk_mode: str = "directional",
        coordinate_transform=None,
    ):
        if not isinstance(cds_bank, HSIClassDiscriminativeStructureBank):
            raise TypeError("historical reader requires the canonical CDS bank")
        if not math.isfinite(float(margin_std)) or float(margin_std) < 0:
            raise ValueError("margin_std must be finite and nonnegative")
        # Older main.py versions still supply 'moment_ce'. Do not silently
        # continue that incompatible historical objective.
        if risk_mode != "directional":
            if risk_mode not in ("moment_ce", "mean_ce", "worst_margin"):
                raise ValueError("historical_risk_mode must be directional")
            warnings.warn(
                f"Legacy historical_risk_mode={risk_mode!r} now uses directional "
                "cosine-boundary protection; update main.py CLI to 'directional'",
                RuntimeWarning,
                stacklevel=2,
            )
        self.cds_bank = cds_bank
        self.margin_std = float(margin_std)
        self.risk_mode = "directional"
        # A transient read frame, never a second persistent statistics bank.
        self.coordinate_transform = coordinate_transform

    @property
    def class_ids(self):
        return self.cds_bank.seen_class_ids

    def require(self, ids):
        if self.class_ids != tuple(map(int, ids)):
            raise RuntimeError("historical bank classes disagree with phase order")

    def validate_against_cds(self, cds):
        ids = tuple(map(int, cds.classes.class_ids.tolist()))
        self.require(ids)
        self._selected(ids, cds, self.cds_bank.device, self.cds_bank.dtype)
        return True

    @torch.no_grad()
    def _selected(self, ids, cds, device, dtype):
        ids = tuple(map(int, ids))
        seen = tuple(map(int, cds.classes.class_ids.tolist()))
        if not ids or not set(ids).issubset(seen):
            raise ValueError("requested historical class not present in CDS")
        for cid in ids:
            if cid not in self.class_ids:
                raise ValueError(f"no committed moments for class {cid}")
        expected = self.cds_bank.class_structures(ids)
        state = self.cds_bank.class_moments(ids)
        if self.coordinate_transform is not None:
            expected = transport_historical_structure(
                expected, self.coordinate_transform
            )
            state["mean"] = rotate_historical_flat_rows(
                state["mean"], self.coordinate_transform
            )
            state["basis"] = rotate_historical_flat_rows(
                state["basis"].transpose(1, 2), self.coordinate_transform
            ).transpose(1, 2)
        for i, cid in enumerate(ids):
            j = seen.index(cid)
            for field in expected.__dataclass_fields__:
                if not torch.equal(
                    getattr(cds.classes, field)[j], getattr(expected, field)[i]
                ):
                    raise RuntimeError(
                        f"historical class {cid} has inconsistent bank coordinates: {field}"
                    )
        return {
            key: value.detach().to(
                device=device, dtype=dtype if value.is_floating_point() else value.dtype
            )
            for key, value in state.items()
        }

    @torch.no_grad()
    def sample_historical_features(
        self,
        old_ids,
        cds,
        samples_per_class,
        *,
        generator=None,
        normalize=True,
        use_covariance=True,
    ):
        """Detached transient Q draws from canonical moments in this reader's frame.

        Raw covariance is (n-1)/n times the bank's low-rank sample covariance.
        L2 normalization then changes the moments. These are model draws, not
        recovered observations; native spectra are never generated or paired.
        Mean-only mode uses no random numbers. Nothing is cached or committed.
        """
        from numbers import Integral

        if (
            isinstance(samples_per_class, bool)
            or not isinstance(samples_per_class, Integral)
            or samples_per_class < 1
        ):
            raise ValueError("samples_per_class must be a positive integer")
        if not isinstance(normalize, bool) or not isinstance(use_covariance, bool):
            raise ValueError("normalize and use_covariance must be boolean")
        ids = tuple(map(int, old_ids))
        if not ids or len(set(ids)) != len(ids):
            raise ValueError("historical class IDs must be non-empty and unique")
        state = self._selected(ids, cds, self.cds_bank.device, self.cds_bank.dtype)
        mean, basis = state["mean"], state["basis"]
        count = int(samples_per_class)
        c, f = mean.shape
        flat = mean[:, None, :].expand(c, count, f).clone()
        if use_covariance:
            # Padded zero basis columns are allowed. Orthogonal nonzero columns
            # project the isotropic noise out of the retained covariance axes.
            epsilon = torch.randn(
                c, count, f, device=mean.device, dtype=mean.dtype, generator=generator
            )
            coefficients = torch.randn(
                c,
                count,
                basis.size(-1),
                device=mean.device,
                dtype=mean.dtype,
                generator=generator,
            )
            projected = torch.einsum("cmf,cfr->cmr", epsilon, basis)
            complement = epsilon - torch.einsum("cmr,cfr->cmf", projected, basis)
            low_rank = torch.einsum(
                "cmr,cfr->cmf",
                coefficients * state["eigenvalues"].clamp_min(0).sqrt()[:, None, :],
                basis,
            )
            noise = (
                low_rank
                + state["residual_variance"].clamp_min(0).sqrt()[:, None, None]
                * complement
            )
            n = state["counts"].to(mean)
            population_scale = ((n - 1) / n.clamp_min(1)).clamp_min(0).sqrt()
            flat = flat + population_scale[:, None, None] * noise
        if not bool(torch.isfinite(flat).all()):
            raise RuntimeError("nonfinite historical model draw")
        if normalize:
            norms = flat.norm(dim=-1, keepdim=True)
            if not bool((norms > torch.finfo(flat.dtype).eps).all()):
                raise RuntimeError("historical feature has no stable cosine direction")
            flat = flat / norms
        labels = torch.tensor(
            ids, device=mean.device, dtype=torch.long
        ).repeat_interleave(count)
        return (
            flat.reshape(
                c * count,
                self.cds_bank.spectral_token_count,
                self.cds_bank.spectral_token_dim,
            ).detach(),
            labels,
        )

    def historical_negative_logits(self, q, old_ids, cds, tau, *, classifier=None):
        """Detached deployed old W negatives; mean carriers support legacy callers."""
        z = normalize_ordered_decision_features(q).flatten(1)
        st = self._selected(tuple(map(int, old_ids)), cds, z.device, z.dtype)
        carrier = (
            st["mean"]
            if classifier is None
            else F.normalize(classifier.class_weight(old_ids).detach().to(z), dim=1)
        )
        return (
            torch.as_tensor(tau, device=z.device, dtype=z.dtype).detach()
            * z
            @ carrier.T
        )

    def historical_risk(
        self,
        classifier,
        old_ids,
        cds,
        *,
        pair_risk_weight=1.0,
        class_pair_risk=None,
        objective=None,
    ):
        """Old-vs-all-seen directional margin/variance on deployed cosine W.

        Not old-data CE and not a guarantee on an unknown distribution. All
        historical first/second moments are detached. Gradients affect only
        authorized seen classifier rows; never Q, teacher, or stored CDS.
        """
        if objective is not None and objective not in (
            "directional",
            "moment_ce",
            "mean_ce",
            "worst_margin",
        ):
            raise ValueError("historical risk must use the directional CDS objective")
        ids = tuple(map(int, old_ids))
        seen = tuple(map(int, cds.classes.class_ids.tolist()))
        if not ids or not set(ids).issubset(seen):
            raise ValueError("old classes absent from all-seen class order")
        if not math.isfinite(float(pair_risk_weight)) or not 0 <= pair_risk_weight <= 1:
            raise ValueError("pair risk weight must be in [0,1]")
        w = F.normalize(classifier.class_weight(cds.classes.class_ids), dim=1)
        state = self._selected(ids, cds, w.device, w.dtype)
        target = torch.tensor(
            [seen.index(cid) for cid in ids], device=w.device, dtype=torch.long
        )
        d = w.index_select(0, target)[:, None, :] - w[None, :, :]
        mean_margin = torch.einsum("cjf,cf->cj", d, state["mean"])
        projected = torch.einsum("cjf,cfr->cjr", d, state["basis"])
        directional = (projected.square() * state["eigenvalues"][:, None, :]).sum(-1)
        remaining = (d.square().sum(-1) - projected.square().sum(-1)).clamp_min(0)
        directional += state["residual_variance"][:, None] * remaining
        # Bank holds unbiased sample covariance; prediction risk uses its
        # population-scale estimate. A PSD low-rank approximation is retained.
        n = state["counts"].to(w).clamp_min(1)
        directional *= ((n - 1) / n)[:, None]
        std = (directional.clamp_min(0) + 1e-8).sqrt()
        tau = classifier.logit_scale.detach()
        conservative = tau * (mean_margin - self.margin_std * std)
        risk = (
            whole_cds_pair_risk(cds)[0]
            if class_pair_risk is None
            else torch.as_tensor(class_pair_risk, device=w.device, dtype=w.dtype)
        )
        if (
            risk.shape != (len(seen), len(seen))
            or not bool(torch.isfinite(risk).all())
            or bool(((risk < 0) | (risk > 1)).any())
        ):
            raise ValueError(
                "CDS pair risk must be finite [all seen, all seen] in [0,1]"
            )
        relevance = risk.to(w).index_select(0, target).detach()
        own = F.one_hot(target, len(seen)).bool()
        # Spectral rivalry increases difficulty but NEVER removes a competing
        # class. All cross-phase historical decisions remain supervised.
        penalty = -conservative + torch.log1p(float(pair_risk_weight) * relevance)
        penalty = penalty.masked_fill(own, -torch.inf)
        losses = torch.logsumexp(
            torch.cat(
                (penalty, torch.zeros(len(ids), 1, device=w.device, dtype=w.dtype)),
                dim=1,
            ),
            dim=1,
        )
        worst = penalty.argmax(dim=1)
        valid = ~own
        positive = ((conservative > 0) & valid).sum() / valid.sum().clamp_min(1)
        if not bool(torch.isfinite(losses).all()):
            raise RuntimeError("nonfinite historical directional loss")
        return HistoricalSupportRisk(
            losses.mean(),
            losses,
            cds.classes.class_ids.index_select(0, worst),
            torch.tensor(ids, device=w.device, dtype=torch.long),
            conservative[:, None, :],
            torch.zeros(len(ids), device=w.device, dtype=torch.long),
            positive,
            objective="directional",
        )

    @torch.no_grad()
    def audit_expected_ce(self, classifier, class_ids, cds, features, labels):
        # A directional margin surrogate cannot be assessed by treating its
        # value as empirical cross-entropy. No false CE-approximation metrics.
        return {
            "status": "not_applicable_directional_margin_not_CE",
            "class_ids": list(map(int, class_ids)),
            "sample_count": int(len(features)),
            "used_for_selection": False,
            "scope": "Current validation only; no old-class oracle observations",
        }

    def historical_upper_risk(self, *_args, **_kwargs):
        raise RuntimeError("historical moments do not certify old-class error")

    def fit_decision_calibration(self, *_args, **_kwargs):
        return self.diagnostics()

    def diagnostics(self):
        if not self.class_ids:
            return {}
        st = self.cds_bank.class_moments(self.class_ids)
        return {
            "class_ids": self.class_ids,
            "sample_counts": st["counts"].cpu().tolist(),
            "stores_covariance": True,
            "covariance": "one low-rank Q covariance per class",
            "stores_old_examples": False,
            "stores_feature_exemplars": False,
            "spectral_grouping": "none",
            "spectral_statistics": "native band and adjacent-difference population variances",
            "classifier_objective": "directional",
            "expected_ce_is_approximate": False,
        }


# """Stateless historical cosine-decision constraints from one canonical CDS bank.

# No historical images, feature samples, pseudo-features, classifier surrogates,
# Gaussian likelihoods, or learned calibration networks are constructed here.
# """

# from __future__ import annotations
# from dataclasses import dataclass
# import math
# import warnings
# import torch
# import torch.nn.functional as F
# from models.geometry_bank import (
#     HSIClassDiscriminativeStructureBank,
#     normalize_ordered_decision_features,
#     whole_cds_pair_risk,
# )

# Tensor = torch.Tensor


# @dataclass(frozen=True)
# class HistoricalSupportRisk:
#     loss: Tensor
#     per_class_loss: Tensor
#     per_class_worst_rival: Tensor
#     class_ids: Tensor
#     margin_surrogate: Tensor
#     per_class_worst_group: Tensor
#     positive_margin_fraction: Tensor
#     objective: str = "directional"
#     mean_ce_loss: Tensor | None = None
#     curvature_correction: Tensor | None = None

#     @property
#     def upper_margin(self):
#         return self.margin_surrogate


# class HistoricalCDSSupportBank:
#     """Compatibility façade; CDS storage belongs exclusively to geometry_bank."""

#     def __init__(
#         self,
#         cds_bank: HSIClassDiscriminativeStructureBank,
#         *,
#         margin_std: float = 1.0,
#         risk_mode: str = "directional",
#     ):
#         if not isinstance(cds_bank, HSIClassDiscriminativeStructureBank):
#             raise TypeError("historical reader requires the canonical CDS bank")
#         if not math.isfinite(float(margin_std)) or float(margin_std) < 0:
#             raise ValueError("margin_std must be finite and nonnegative")
#         # Older main.py versions still supply 'moment_ce'. Do not silently
#         # continue that incompatible historical objective.
#         if risk_mode != "directional":
#             if risk_mode not in ("moment_ce", "mean_ce", "worst_margin"):
#                 raise ValueError("historical_risk_mode must be directional")
#             warnings.warn(
#                 f"Legacy historical_risk_mode={risk_mode!r} now uses directional "
#                 "cosine-boundary protection; update main.py CLI to 'directional'",
#                 RuntimeWarning,
#                 stacklevel=2,
#             )
#         self.cds_bank = cds_bank
#         self.margin_std = float(margin_std)
#         self.risk_mode = "directional"

#     @property
#     def class_ids(self):
#         return self.cds_bank.seen_class_ids

#     def require(self, ids):
#         if self.class_ids != tuple(map(int, ids)):
#             raise RuntimeError("historical bank classes disagree with phase order")

#     def validate_against_cds(self, cds):
#         ids = tuple(map(int, cds.classes.class_ids.tolist()))
#         self.require(ids)
#         self._selected(ids, cds, self.cds_bank.device, self.cds_bank.dtype)
#         return True

#     def _selected(self, ids, cds, device, dtype):
#         ids = tuple(map(int, ids))
#         seen = tuple(map(int, cds.classes.class_ids.tolist()))
#         if not ids or not set(ids).issubset(seen):
#             raise ValueError("requested historical class not present in CDS")
#         for cid in ids:
#             if cid not in self.class_ids:
#                 raise ValueError(f"no committed moments for class {cid}")
#             i, j = self.cds_bank._row_for_class(cid), seen.index(cid)
#             for source, target in (
#                 ("learned_identity", "learned_identity_bank"),
#                 ("physical_identity", "physical_identity_bank"),
#                 ("physical_variance", "physical_variance_bank"),
#                 ("physical_delta_variance", "physical_delta_variance_bank"),
#                 ("learned_covariance_basis", "moment_basis_bank"),
#                 ("learned_covariance_eigenvalues", "moment_eigenvalues_bank"),
#                 ("learned_covariance_residual", "moment_residual_variance_bank"),
#             ):
#                 if not torch.equal(
#                     getattr(cds.classes, source)[j], getattr(self.cds_bank, target)[i]
#                 ):
#                     raise RuntimeError(
#                         f"historical class {cid} has inconsistent bank coordinates"
#                     )
#         return self.cds_bank.class_moments(ids, device=device, dtype=dtype)

#     def historical_negative_logits(self, q, old_ids, cds, tau):
#         """Old means as real-sample contrastive negatives, never generated data."""
#         z = normalize_ordered_decision_features(q).flatten(1)
#         st = self._selected(tuple(map(int, old_ids)), cds, z.device, z.dtype)
#         return (
#             torch.as_tensor(tau, device=z.device, dtype=z.dtype).detach()
#             * z
#             @ st["mean"].T
#         )

#     def historical_risk(
#         self,
#         classifier,
#         old_ids,
#         cds,
#         *,
#         pair_risk_weight=1.0,
#         class_pair_risk=None,
#         objective=None,
#     ):
#         """Old-vs-all-seen directional margin/variance on deployed cosine W.

#         Not old-data CE and not a guarantee on an unknown distribution. All
#         historical first/second moments are detached. Gradients affect only
#         authorized seen classifier rows; never Q, teacher, or stored CDS.
#         """
#         if objective is not None and objective not in (
#             "directional",
#             "moment_ce",
#             "mean_ce",
#             "worst_margin",
#         ):
#             raise ValueError("historical risk must use the directional CDS objective")
#         ids = tuple(map(int, old_ids))
#         seen = tuple(map(int, cds.classes.class_ids.tolist()))
#         if not ids or not set(ids).issubset(seen):
#             raise ValueError("old classes absent from all-seen class order")
#         if not math.isfinite(float(pair_risk_weight)) or not 0 <= pair_risk_weight <= 1:
#             raise ValueError("pair risk weight must be in [0,1]")
#         w = F.normalize(classifier.class_weight(cds.classes.class_ids), dim=1)
#         state = self._selected(ids, cds, w.device, w.dtype)
#         target = torch.tensor(
#             [seen.index(cid) for cid in ids], device=w.device, dtype=torch.long
#         )
#         d = w.index_select(0, target)[:, None, :] - w[None, :, :]
#         mean_margin = torch.einsum("cjf,cf->cj", d, state["mean"])
#         projected = torch.einsum("cjf,cfr->cjr", d, state["basis"])
#         directional = (projected.square() * state["eigenvalues"][:, None, :]).sum(-1)
#         remaining = (d.square().sum(-1) - projected.square().sum(-1)).clamp_min(0)
#         directional += state["residual_variance"][:, None] * remaining
#         # Bank holds unbiased sample covariance; prediction risk uses its
#         # population-scale estimate. A PSD low-rank approximation is retained.
#         n = state["counts"].to(w).clamp_min(1)
#         directional *= ((n - 1) / n)[:, None]
#         std = (directional.clamp_min(0) + 1e-8).sqrt()
#         tau = classifier.logit_scale.detach()
#         conservative = tau * (mean_margin - self.margin_std * std)
#         risk = (
#             whole_cds_pair_risk(cds)[0]
#             if class_pair_risk is None
#             else torch.as_tensor(class_pair_risk, device=w.device, dtype=w.dtype)
#         )
#         if (
#             risk.shape != (len(seen), len(seen))
#             or not bool(torch.isfinite(risk).all())
#             or bool(((risk < 0) | (risk > 1)).any())
#         ):
#             raise ValueError(
#                 "CDS pair risk must be finite [all seen, all seen] in [0,1]"
#             )
#         relevance = risk.to(w).index_select(0, target).detach()
#         own = F.one_hot(target, len(seen)).bool()
#         # Spectral rivalry increases difficulty but NEVER removes a competing
#         # class. All cross-phase historical decisions remain supervised.
#         penalty = -conservative + torch.log1p(float(pair_risk_weight) * relevance)
#         penalty = penalty.masked_fill(own, -torch.inf)
#         losses = torch.logsumexp(
#             torch.cat(
#                 (penalty, torch.zeros(len(ids), 1, device=w.device, dtype=w.dtype)),
#                 dim=1,
#             ),
#             dim=1,
#         )
#         worst = penalty.argmax(dim=1)
#         valid = ~own
#         positive = ((conservative > 0) & valid).sum() / valid.sum().clamp_min(1)
#         if not bool(torch.isfinite(losses).all()):
#             raise RuntimeError("nonfinite historical directional loss")
#         return HistoricalSupportRisk(
#             losses.mean(),
#             losses,
#             cds.classes.class_ids.index_select(0, worst),
#             torch.tensor(ids, device=w.device, dtype=torch.long),
#             conservative[:, None, :],
#             torch.zeros(len(ids), device=w.device, dtype=torch.long),
#             positive,
#             objective="directional",
#         )

#     @torch.no_grad()
#     def audit_expected_ce(self, classifier, class_ids, cds, features, labels):
#         # A directional margin surrogate cannot be assessed by treating its
#         # value as empirical cross-entropy. No false CE-approximation metrics.
#         return {
#             "status": "not_applicable_directional_margin_not_CE",
#             "class_ids": list(map(int, class_ids)),
#             "sample_count": int(len(features)),
#             "used_for_selection": False,
#             "scope": "Current validation only; no old-class oracle observations",
#         }

#     def historical_upper_risk(self, *_args, **_kwargs):
#         raise RuntimeError("historical moments do not certify old-class error")

#     def fit_decision_calibration(self, *_args, **_kwargs):
#         return self.diagnostics()

#     def diagnostics(self):
#         if not self.class_ids:
#             return {}
#         st = self.cds_bank.class_moments(self.class_ids)
#         return {
#             "class_ids": self.class_ids,
#             "sample_counts": st["counts"].cpu().tolist(),
#             "stores_covariance": True,
#             "covariance": "one low-rank Q covariance per class",
#             "stores_old_examples": False,
#             "stores_feature_exemplars": False,
#             "spectral_grouping": "none",
#             "spectral_statistics": "native band and adjacent-difference population variances",
#             "classifier_objective": "directional",
#             "expected_ce_is_approximate": False,
#         }
