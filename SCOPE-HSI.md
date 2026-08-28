# SCOPE-HSI Architecture


## Method Name

\[
oxed{	extbf{SCOPE-HSI}}
\]

**SCOPE-HSI: Structural Class-Discriminative Orthogonal Plasticity for Exemplar-Free Hyperspectral Class-Incremental Learning**

The name reflects the central mechanism of the framework:

- **Structural** — historical classes are represented by \(\{I_c,S_c,N_c\}\);
- **Class-Discriminative** — preservation focuses on directions that distinguish classes;
- **Orthogonal** — destructive current-task gradient components are removed by orthogonal projection;
- **Plasticity** — the remaining representation space stays trainable for new classes;
- **Exemplar-Free** — no old HSI samples, old per-sample features, replay, or teacher model are retained.

The complete framework is called **SCOPE-HSI**.  
The incremental preservation component is called **Class-Structural Orthogonal Plasticity (CSOP)**.

## 1. Research Goal

We address **non-exemplar class-incremental learning (NECIL) for hyperspectral image classification (HSI)**.

At phase \(t\), the model receives training samples only from the current new classes. It must learn those classes while retaining performance on all previously learned classes.

The protocol is:

- no stored old HSI patches;
- no stored old spectra;
- no exemplar replay;
- no stored per-sample old features;
- no teacher model or knowledge distillation;
- no synthetic replay;
- one persistent classifier for all seen classes;
- one shared feature space across all phases.

The central problem is therefore:

\[
\boxed{
\text{learn new classes without destructively changing old class-discriminative structure}
}
\]

---

# 2. Complete Architecture

```text
                               CLASS DECISION
                         ┌─────────────────────────┐
                         │ Persistent Angular      │
                         │ Classifier              │
                         │                         │
                         │ all seen classes        │
                         └───────────▲─────────────┘
                                     │
                                     │ Q
                                     │
HSI patch + raw spectrum             │
          │                          │
          ▼                          │
┌───────────────────────┐            │
│ HSI Mamba Backbone    │            │
│                       │            │
│ spectral processing   │            │
│ spatial processing    │            │
│ spectral-spatial      │            │
│ fusion                │            │
└───────────┬───────────┘            │
            │                        │
            ▼                        │
       Q_raw(x) ∈ R^(K×D)            │
            │                        │
            ▼                        │
┌───────────────────────┐            │
│ Global RMS            │            │
│ Canonicalization      │            │
└───────────┬───────────┘            │
            │                        │
            ▼                        │
      Q(x) ∈ R^(K×D) ────────────────┘
            │
            │
            ▼
┌──────────────────────────────┐
│ Class Statistics             │
│                              │
│ G_c = {I_c, S_c, N_c}        │
│                              │
│ I_c : class identity         │
│ S_c : intra-class structure │
│ N_c : sample count           │
└──────────────┬───────────────┘
               │
               │ old classes only
               ▼
┌──────────────────────────────┐
│ Class-Structural Orthogonal Plasticity (CSOP)       │
│                              │
│ derive protected historical │
│ discriminative subspace U   │
└──────────────┬───────────────┘
               │
               ▼
      incremental gradient
         projection only
```

The architecture is intentionally role-separated:

| Component | One specific job |
|---|---|
| HSI Mamba backbone | Extract ordered spectral-spatial features |
| RMS canonicalizer | Remove global representation-scale instability |
| Angular classifier | Make the class decision |
| \(\{I_c,S_c,N_c\}\) | Retain class-level discriminative statistics |
| Structure preservation | Restrict destructive incremental adaptation |
| Reporting | Measure performance and structural drift only |

The class statistics **never produce logits or predictions**.

---

# 3. HSI Feature Representation

The backbone produces

\[
Q_{\text{raw}}(x)\in\mathbb{R}^{K\times D},
\]

where:

- \(K\) = number of ordered spectral regions;
- \(D\) = feature dimension.

For the current configuration:

\[
K=16,\qquad D=128.
\]

Therefore:

\[
Q_{\text{raw}}(x)\in\mathbb{R}^{16\times128}.
\]

The feature extractor combines:

1. ordered spectral processing;
2. spatial-context processing;
3. spectral-conditioned spatial fusion.

The spectral order is retained because neighboring HSI bands/regions carry meaningful spectral relationships.

---

# 4. Global RMS Canonicalization

We convert the raw representation to a stable coordinate:

\[
\boxed{
Q(x)=
\frac{Q_{\text{raw}}(x)}
{\operatorname{RMS}(Q_{\text{raw}}(x))}
}
\]

with

\[
\operatorname{RMS}(Q_{\text{raw}})
=
\sqrt{
\frac{1}{KD}
\sum_{k=1}^{K}
\sum_{d=1}^{D}
Q_{\text{raw}}[k,d]^2
}.
\]

Therefore:

\[
\operatorname{RMS}(Q)=1
\]

and

\[
\|\operatorname{vec}(Q)\|_2=\sqrt{KD}.
\]

For \(K=16,D=128\):

\[
\sqrt{KD}=\sqrt{2048}\approx45.25.
\]

### Why we use one global scalar

We do **not** independently normalize every spectral region.

Independent token-wise normalization would remove relative magnitude information between spectral regions.

The global RMS normalization stabilizes scale while preserving the internal spectral-region relationships.

---

# 5. Persistent Angular Classifier

Flatten the canonical representation:

\[
z=\operatorname{vec}(Q)\in\mathbb{R}^{KD}.
\]

For class \(c\), normalize only the classifier row:

\[
u_c=\frac{w_c}{\|w_c\|_2}.
\]

The class logit is

\[
\boxed{
\ell_c=z^\top u_c
}
\]

and prediction is

\[
\boxed{
\hat y=\arg\max_{c\in\mathcal C_{\text{seen}}}\ell_c.
}
\]

Because

\[
\|z\|_2=\sqrt{KD},
\]

the logit can also be written as

\[
\ell_c=\sqrt{KD}\cos\theta_c.
\]

There is:

- no temperature;
- no manually selected scale;
- no angular margin;
- no old/new classifier heads;
- no post-hoc calibration.

The same rule is used for every class in every phase.

---

# 6. Class-Discriminative HSI Statistics

After a phase is finalized, we estimate class-level statistics from the **real TRAIN samples available in that phase**.

For class \(c\),

\[
\boxed{
I_c=E[Q(x)\mid y=c]
}
\]

with

\[
I_c\in\mathbb{R}^{K\times D}.
\]

\(I_c\) describes the class identity/location in the ordered HSI representation.

Define the residual

\[
\Delta_c(x)=Q(x)-I_c.
\]

Then

\[
\boxed{
S_c[k,l,d]
=
E[
\Delta_c[k,d]\Delta_c[l,d]
\mid y=c
]
}
\]

with

\[
S_c\in\mathbb{R}^{K\times K\times D}.
\]

\(S_c\) preserves:

- within-class variation;
- relationships between spectral regions \(k\) and \(l\);
- feature-specific variation through dimension \(d\).

The persistent class state is

\[
\boxed{
\mathcal G_c=\{I_c,S_c,N_c\}.
}
\]

\(N_c\) is the number of class-\(c\) TRAIN samples used to estimate the statistics.

---

# 7. Inter-Class Structure

We do not store an \(O(C^2)\) pair bank.

Inter-class relationships are derived when needed:

\[
\boxed{
R_{cd}
=
\{I_c-I_d,\;S_c-S_d\}.
}
\]

This provides:

- relative class identity;
- relative class-specific variation.

Thus the stored statistics already contain the information required for both intra-class and inter-class analysis.

---

# 8. Base Phase

Suppose the base phase contains classes

\[
\mathcal C_0.
\]

The training path is

```text
base HSI TRAIN samples
        ↓
HSI Mamba backbone
        ↓
canonical Q
        ↓
angular classifier
        ↓
cross-entropy
        ↓
update backbone + classifier
```

The base objective is simply

\[
\boxed{
L_{\text{base}}
=
CE(\ell(x),y).
}
\]

Training runs for the fixed requested number of epochs.

There is:

- no early stopping;
- no best-validation checkpoint;
- no geometry loss;
- no replay;
- no teacher.

After the final epoch, deterministic unaugmented TRAIN samples are passed through the finalized model and we commit

\[
\boxed{
\{I_c,S_c,N_c\},
\qquad c\in\mathcal C_0.
}
\]

The class statistics are therefore a description of the **finalized** base representation.

---

# 9. Why Incremental Learning Is Difficult

At the next phase, only new-class samples are available.

If the backbone changes from

\[
\theta_{t-1}
\rightarrow
\theta_t,
\]

the old representation may also change.

But old training samples are unavailable, so during incremental optimization we cannot directly recompute

\[
I_c^t=E[Q_{\theta_t}(x_{\text{old}})]
\]

or

\[
S_c^t.
\]

Therefore the stored statistics tell us

\[
\boxed{\text{what historical class information must remain stable}}
\]

but we need a separate mechanism that determines

\[
\boxed{\text{how current updates are allowed to change the shared representation}}.
\]

That is the job of **Class-Structural Orthogonal Plasticity (CSOP)**.

---

# 10. Class-Structural Orthogonal Plasticity (CSOP)

## 10.1 Identity-discriminative modes

For each historical class, flatten the class identity:

\[
\mu_c=\operatorname{vec}(I_c).
\]

Compute the class-balanced identity mean:

\[
\bar\mu=
\frac1C\sum_c\mu_c.
\]

The between-class identity operator is

\[
\boxed{
B=
\frac1C
\sum_c
(\mu_c-\bar\mu)
(\mu_c-\bar\mu)^T.
}
\]

The class structures \(S_c\) provide the class-balanced within-class operator

\[
W.
\]

We identify discriminative modes from

\[
\boxed{
Bv=\rho(B+W)v.
}
\]

A mode is important when it has strong between-class separation relative to historical within-class variation.

Because

\[
\operatorname{rank}(B)\le C-1,
\]

the number of identity-discriminative modes is naturally bounded:

\[
\boxed{
r_I\le C_{\text{old}}-1.
}
\]

No manually selected preservation rank is introduced.

Let

\[
V=[v_1,\ldots,v_{r_I}].
\]

---

## 10.2 Make \(S_c\) an explicit preservation signal

Preserving only \(V\) would protect class-identity directions but would not fully use class-specific variation.

Compute

\[
\bar S=
\frac1C\sum_cS_c
\]

and

\[
\boxed{
\Delta S_c=S_c-\bar S.
}
\]

For each historical class, apply its class-specific structural deviation to the discriminative modes:

\[
\boxed{
R_c=\Delta S_cV.
}
\]

The protected historical representation space is

\[
\boxed{
U=
\operatorname{orth}
\left(
[V,\Delta S_1V,\ldots,\Delta S_CV]
\right).
}
\]

Therefore \(U\) contains:

1. historical class-identity discrimination;
2. class-specific HSI variation around those discriminative modes.

This is the information we protect during incremental adaptation.

---

# 11. Incremental Training

At incremental phase \(t\):

\[
\mathcal C_{\text{old}}
=
\bigcup_{j<t}\mathcal C_j
\]

and the current classes are

\[
\mathcal C_t.
\]

Only samples with

\[
y\in\mathcal C_t
\]

are used for training.

---

## 11.1 Classifier expansion

The persistent classifier is expanded:

\[
W_{\text{old}}
\rightarrow
W_{\text{old+new}}.
\]

There is still one classifier.

All classes compete under the same rule:

\[
\hat y
=
\arg\max_{c\in\mathcal C_{\text{seen}}}
\ell_c.
\]

---

## 11.2 Incremental objective

For current-phase samples,

\[
\boxed{
L_{\text{inc}}
=
CE(\ell(x),y).
}
\]

There is no explicit preservation loss.

In particular, we do not use

\[
L_{\text{inc}}
=
L_{\text{cls}}
+
\lambda L_{\text{preserve}}.
\]

Therefore there is no preservation weight \(\lambda\).

---

# 12. Gradient-Level Preservation

The current CE objective gives a representation gradient

\[
g_Q
=
\frac{\partial L_{\text{inc}}}{\partial Q}.
\]

Before this gradient enters the shared backbone, we remove the component lying in the old protected structural subspace:

\[
\boxed{
g_Q^{\text{plastic}}
=
(I-UU^T)g_Q.
}
\]

Therefore

\[
\boxed{
U^Tg_Q^{\text{plastic}}=0.
}
\]

### Meaning

The current-task gradient is allowed to use the remaining representation space but cannot directly push \(Q\) along the historical class-discriminative structural directions represented by \(U\).

So the design is:

```text
new-class CE gradient
        │
        ├── historical structural component ── remove
        │
        └── plastic component ──────────────── keep
                                                │
                                                ▼
                                          backbone update
```

The backbone is **not frozen**.

---

# 13. Old Classifier Rows

Old classifier rows are also not frozen.

They remain trainable, but their updates are constrained so that their normalized responses on the historical class-identity span are not directly altered to first order.

New classifier rows remain fully plastic.

This avoids using:

- immutable old classifier rows;
- separate old/new heads;
- classifier replay.

---

# 14. Why Incremental Backbone Weight Decay Is Zero

During incremental learning, preservation constrains the task-driven gradient.

AdamW weight decay would add an independent update

\[
\Delta\theta_{\text{wd}}
=
-\eta\lambda\theta
\]

that does not pass through the structural gradient projection.

Therefore the incremental backbone uses

\[
\boxed{
\lambda_{\text{wd}}^{\text{inc}}=0.
}
\]

Phase 0 can still use normal weight decay because no historical structure must yet be preserved.

---

# 15. End of an Incremental Phase

After the final epoch of phase \(t\), the real current-phase TRAIN samples are still available.

We compute only the new classes' statistics:

\[
I_c,\quad S_c,\quad N_c,
\qquad c\in\mathcal C_t.
\]

They are appended to the previously retained class statistics:

\[
\boxed{
\mathcal G^{t}
=
\mathcal G^{t-1}
\cup
\{\mathcal G_c:c\in\mathcal C_t\}.
}
\]

Historical class statistics are **not re-estimated using new-class samples**.

At the next phase, the preservation space is reconstructed from statistics of all previously learned classes.

---

# 16. Complete Phase Flow

```text
PHASE 0
────────────────────────────────────────────
base-class TRAIN HSI
        ↓
Mamba backbone
        ↓
canonical Q
        ↓
angular classifier
        ↓
CE
        ↓
final epoch
        ↓
estimate {I,S,N} for base classes
        ↓
save checkpoint


PHASE t > 0
────────────────────────────────────────────
historical {I,S,N}
        ↓
derive V
        ↓
derive (S_c - mean S)V
        ↓
build protected structural basis U


current new-class TRAIN HSI only
        ↓
same trainable Mamba backbone
        ↓
canonical Q
        ↓
same all-seen angular classifier
        ↓
CE
        ↓
dL/dQ
        ↓
(I - UUᵀ)dL/dQ
        ↓
update shared backbone

old classifier rows:
constrained update

new classifier rows:
fully plastic

        ↓
final epoch
        ↓
estimate {I,S,N} for new classes only
        ↓
append class statistics
        ↓
save finalized checkpoint
```

---

# 17. What the Preservation Mechanism Guarantees

The implemented constraint guarantees

\[
\boxed{
U^Tg_Q^{\text{plastic}}=0.
}
\]

This means the **direct current-task representation gradient** has zero component in the protected historical structural subspace.

It does **not** claim the stronger statement

\[
Q_{\theta_t}(x_{\text{old}})
=
Q_{\theta_{t-1}}(x_{\text{old}})
\]

for every old sample.

The shared backbone is nonlinear, and old samples are unavailable during training.

Therefore actual historical drift must be measured experimentally.

This distinction is important.

---

# 18. Reporting and Validation

Reporting never participates in training.

After phase \(t\) is finalized, we compare the immediately previous checkpoint

\[
\theta_{t-1}
\]

against the current checkpoint

\[
\theta_t
\]

using the **same held-out old TEST samples**.

We compute empirical old-class statistics under both checkpoints.

### Identity drift

\[
\boxed{
\Delta I_c
=
RMS
\left(
I_c^{t,\text{test}}
-
I_c^{t-1,\text{test}}
\right).
}
\]

### Structure drift

\[
\boxed{
\Delta S_c
=
RMS
\left(
S_c^{t,\text{test}}
-
S_c^{t-1,\text{test}}
\right).
}
\]

These quantities test whether the preservation mechanism actually reduces old-representation damage.

They are never fed back into training.

---

# 19. Incremental Metrics

For every incremental phase we report:

### Old-class balanced accuracy

\[
BA_{\text{old}}
\]

### New-class balanced accuracy

\[
BA_{\text{new}}
\]

### Harmonic stability-plasticity score

\[
\boxed{
H=
\frac{
2BA_{\text{old}}BA_{\text{new}}
}{
BA_{\text{old}}+BA_{\text{new}}
}.
}
\]

### Old-class forgetting

\[
\boxed{
F_{\text{old}}
=
BA_{\text{old}}^{\text{before}}
-
BA_{\text{old}}^{\text{after}}.
}
\]

We also retain:

- OA;
- BA;
- minimum class accuracy;
- kappa;
- angular/cosine margin;
- confusion matrix;
- per-class accuracy.

---

# 20. Joint t-SNE Visualization

For representation visualization, we do **not** fit separate t-SNE embeddings for different phases.

Instead:

\[
Q_{t-1}
\quad\text{and}\quad
Q_t
\]

are concatenated and embedded with one joint t-SNE model.

```text
previous finalized checkpoint ──> old TEST Q
                                      │
                                      ├──────┐
current finalized checkpoint ───> old/new Q │
                                      │      │
                                      └── concatenate
                                              │
                                              ▼
                                         one t-SNE
```

This produces comparable 2-D coordinates for the two panels.

The figure uses:

- same class color across phases;
- circle = historical class;
- triangle = new class;
- cross = embedded class center;
- ellipse = qualitative spread in t-SNE space.

The t-SNE ellipse is only a visualization and is **not** the stored \(S_c\).

---

# 21. What Is Stored Between Phases

The checkpoint contains the persistent model state and class-level statistics required to continue learning.

The continual class information is

\[
\boxed{
\{I_c,S_c,N_c\}.
}
\]

We do not retain:

- old raw HSI patches;
- old spectra;
- old sample indices for replay;
- old per-sample features;
- replay batches;
- teacher outputs;
- synthetic old samples.

The preservation basis itself is reconstructed from the class statistics at the beginning of each incremental phase.

---

# 22. Why the Architecture Is Coherent

The design follows one principle:

\[
\boxed{
\text{each component solves one problem}
}
\]

### Feature extraction problem

Solved by:

\[
\text{HSI Mamba backbone}.
\]

### Representation-scale problem

Solved by:

\[
\text{global RMS canonicalization}.
\]

### Classification problem

Solved by:

\[
\text{persistent angular classifier}.
\]

### Historical class-information problem

Solved by:

\[
\{I_c,S_c,N_c\}.
\]

### Catastrophic-forgetting problem

Solved by:

\[
\text{class-structural gradient preservation}.
\]

The geometry/statistics do not become a second classifier.

The classifier does not become a preservation mechanism.

The preservation component does not introduce a second task loss.

---

# 23. Main Research Idea in One Paragraph

**SCOPE-HSI** learns a shared ordered spectral-spatial representation using an HSI Mamba backbone and performs all-seen classification with one persistent angular classifier. After each finalized phase, every learned class is represented only by class-level statistics \(\{I_c,S_c,N_c\}\), where \(I_c\) captures class identity and \(S_c\) captures feature-resolved relationships among ordered spectral regions. During subsequent phases, these statistics are converted into a compact historical class-discriminative structural subspace. New-class learning still uses ordinary cross-entropy, but the representation gradient is projected onto the orthogonal plastic complement of the protected subspace before entering the shared backbone. The model therefore remains trainable for new classes while direct gradient pressure along historically discriminative HSI directions is removed, without replay, stored exemplars, synthetic old data, teacher distillation, or separate classifiers.

---

# 24. Short Explanation for a Presentation

A concise way to explain the method is:

> **First, we learn an HSI spectral-spatial representation and a single classifier. After a phase finishes, we do not keep old samples; instead we retain only each class's identity \(I_c\) and structured intra-class variation \(S_c\). At the next phase, these statistics tell us which representation directions were important for distinguishing the old classes. We construct a protected structural subspace from them. New classes are then trained normally with cross-entropy, but before the gradient updates the shared backbone, we remove the component that points through this protected old-class subspace. The remaining orthogonal space stays plastic for learning the new classes. At the end of the phase we append statistics for the new classes and repeat the same process.**

---

# 25. What to Emphasize When Explaining the Novelty

Do not describe the method simply as:

> "we use geometry"  
> or  
> "we apply orthogonal gradient projection."

The important distinction is:

\[
\boxed{
\text{HSI class-discriminative statistics}
\rightarrow
\text{protected class-structural subspace}
\rightarrow
\text{controlled incremental adaptation}
}
\]

The preserved information is not generic feature covariance.

It explicitly contains:

\[
\boxed{
\text{class identity}
+
\text{class-specific feature-resolved spectral-region variation}.
}
\]

The research question is therefore:

> **Can historical HSI class-discriminative structure define which representation directions should remain stable and which directions should remain plastic during strict non-exemplar incremental learning?**

---


# 26. Framework Naming

The complete method is:

\[
\boxed{
\textbf{SCOPE-HSI}
}
\]

**Structural Class-Discriminative Orthogonal Plasticity for Exemplar-Free Hyperspectral Class-Incremental Learning**

The internal incremental mechanism is:

\[
\boxed{
\textbf{CSOP}
}
\]

**Class-Structural Orthogonal Plasticity**

The naming hierarchy is therefore:

```text
SCOPE-HSI
│
├── HSI Mamba Feature Extractor
├── Global RMS Canonicalizer
├── Persistent Angular Classifier
├── HSI Class Structure {I,S,N}
└── CSOP
    └── Class-Structural Orthogonal Plasticity
```

SCOPE-HSI is the full architecture. CSOP is only the historical-structure preservation mechanism used during incremental optimization.

# 27. Final Method Summary

\[
\boxed{
\begin{aligned}
\text{Representation:}\quad
&Q=f_\theta(x)/RMS(f_\theta(x))
\\[2mm]
\text{Classifier:}\quad
&\ell_c=
\operatorname{vec}(Q)^T
\frac{w_c}{\|w_c\|}
\\[2mm]
\text{Class identity:}\quad
&I_c=E[Q|c]
\\[2mm]
\text{Class structure:}\quad
&S_c[k,l,d]
=
E[\Delta_c[k,d]\Delta_c[l,d]|c]
\\[2mm]
\text{Historical modes:}\quad
&Bv=\rho(B+W)v
\\[2mm]
\text{Structural basis:}\quad
&U=
\operatorname{orth}
[V,(S_1-\bar S)V,\ldots,(S_C-\bar S)V]
\\[2mm]
\text{Incremental loss:}\quad
&L_{\mathrm{inc}}=CE
\\[2mm]
\text{Allowed gradient:}\quad
&g_Q^{\mathrm{plastic}}
=
(I-UU^T)g_Q
\end{aligned}
}
\]

The complete principle is:

\[
\boxed{
\textbf{SCOPE-HSI preserves old class-discriminative HSI structure while keeping the remaining representation space plastic for new-class learning.}
}
