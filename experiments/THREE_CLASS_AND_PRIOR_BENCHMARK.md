# Three-class material source GP — production model selection

This document records the kernel-architecture selection for the
sustainable-concrete strength GP after migrating from the pre-v5
2-class dataset to the corrected v5 3-class dataset. The
recommendation is empirically derived from > 400 ablation cells
across 11+ kernel variants, with seed-deterministic results and
pre-registered acceptance criteria.

---

## TL;DR

> **Ship `v5 + joint_hamming_matern` as the new production model.**
>
> The kernel is a single Matern over a joint feature-plus-Hamming
> distance:
> $$K(z_i, z_j) = \text{Matern}_{3/2}\!\left(\sqrt{\sum_f \frac{(x_{i,f} - x_{j,f})^2}{\ell_f^2} + \alpha \cdot \mathbb{1}[c_i \ne c_j]}\right)$$
> with one learnable categorical penalty $\alpha$ and per-feature ARD
> lengthscales. This is the **Occam-simplest kernel on the Pareto
> frontier of acceptable architectures**: it strictly dominates the
> previous `hamming` production default on every in-distribution
> metric, passes all pre-registered LOCO acceptance criteria, and is
> seed-deterministic.

### Headline comparison

| metric | pre-v5 + `legacy_continuous_ard` (deployed) | v5 + `joint_hamming_matern` (recommended) | Δ |
|---|---|---|---|
| LOO RMSE                          | 533  | **495**  | **−38 psi** ✓ |
| bLOO RMSE                         | 680  | 717  | +37 (architecture cost — see §5) |
| S12-bLOO RMSE                     | 794  | **835** | +41 (same) |
| bLOO PIT-KS (calibration)         | 0.037 | **0.030** | **−0.007** ✓ (best on v5) |
| LOCO Class-0 (mortar) RMSE        | 2882 | **2853** | **−29** ✓ |
| LOCO Class-1 (Set-2) RMSE         | 2912 | **2736** | **−176** ✓ |
| LOCO Class-2 (Set-3) RMSE         | 1139 | **1113** | **−26** ✓ |
| Determinism (seed std)            | ~0 | **0** | tied |

**Pre-registered acceptance criteria** (§7): all **PASSED**.

### Migration steps

1. Flip `DEFAULT_SOURCE_KERNEL` in `boxcrete/kernels.py` from
   `"hamming"` to `"joint_hamming_matern"`.
2. Regenerate `docs/model/strength.json` via
   `experiments/regenerate_strength_json.py`.
3. Update `docs/gp.mjs` to read the new joint-kernel parameters
   (per-feature ARD lengthscales + scalar $\alpha$).
4. Backwards-compatible: keep all other variants as registered
   ablation options (`hamming`, `legacy_continuous_ard`,
   `indexkernel_r{1,2,3}`, `rbf_embedding_d{1,2,3}`,
   `joint_chain_matern`, `joint_hamming_matern_nu{05,25}`,
   `joint_embedding_matern_d{1,2,3}`, `additive_*`).

### Open follow-ups (non-blocking)

* **Robust likelihood for between-batch outliers** (Student-t):
  per-class heteroscedastic noise was tested and *failed* (§4.7)
  because the M60/M61 mortar twin pair hijacks σ_0. A robust
  likelihood that downweights outliers without inflating per-class
  noise is the principled next attempt.
* **Few-shot LOCO adaptation curves**: how quickly does the joint
  kernel adapt when 1, 2, 5, 10 rows of a 4th material class
  arrive? Informs the 4th-class deployment scenario.
* **Per-batch random effect**: model between-batch variance as a
  latent factor rather than absorbing it as additive noise.

---

## 1. Problem statement

### 1.1 Dataset transition: pre-v5 → v5

The previous production model (deployed at `docs/model/strength.json`)
fits a GP on a 2-class **pre-v5** dataset that pooled mortar +
Set-2 concrete into `Material Source = 0` and grouped Set-3
concrete with corruption artifacts into `MS = 1`. The corrected
**v5** dataset has clean 3-class labels:

| class | label | n_rows | description |
|---|---|---|---|
| 0 | mortar (Set 1) | 246 | binder paste, no aggregate |
| 1 | Set-2 concrete | 149 | Heidelberg / Class-C concrete |
| 2 | Set-3 concrete | 252 | Amrize / Class-F concrete |

**The data change is qualitative, not quantitative**: pre-v5 and v5
have identical strength-row counts (647 each), identical strength
values, and identical Strength(Std) for all 638 common
(composition + temperature + time) tuples (verified in H6/H8 of
`experiments/V5_VS_PRE_V5_INVESTIGATION.md`). What v5 changes is:

* the Material Source label (2-class contaminated → 3-class clean);
* 12 corruption-collision mix names split into 24 canonical names
  per actual recipe;
* mortar/concrete misroutes corrected against the collaborator's
  authoritative `Mortar/Concrete` column;
* 3 clay-using mortars (M75/76/77) dropped — they had hidden
  Clay0/1/2 columns the GP feature set cannot see.

### 1.2 The architectural question

The 3-class Material Source label is fundamentally **categorical**
— there is no ordinal or metric meaning to "`MS = 0.5`". The
deployed pre-v5 model treats it as a continuous numeric coordinate
through a single ARD-Matern (`legacy_continuous_ard`), which is
mathematically mis-specified but absorbs class-discriminative
structure into the source ARD lengthscale. The v5 stack needs a
categorical kernel that:

1. **Correctly handles 3 distinct classes** (no ordering assumption);
2. **Preserves the in-distribution bLOO accuracy** of the deployed
   model (~680 psi bLOO on pre-v5);
3. **Improves held-out-class extrapolation** (LOCO RMSE) — the v5
   stack's selling point is enabling BO over Set-3, which the
   deployed pre-v5 model cannot model;
4. **Is seed-deterministic** for operational reproducibility;
5. **Is parsimonious** (Occam — the simpler kernel wins ties).

### 1.3 Architecture overview

The full V2 strength kernel has three additive branches wrapped in
a time gate:
```
K(z, z') = TimeGate( blind_matern(features) + source_branch(features, class) + time_RBF(time) )
```

The **source branch** is the architectural choice this document is
about. We tested 4 families:

| family | source branch | example variants |
|---|---|---|
| Continuous (legacy) | ARD-Matern over all dims including source as numeric | `legacy_continuous_ard` |
| Categorical product | $K_{\text{cat}}(c) \times K_{\text{features}}$ | `hamming`, `indexkernel_r{1,2,3}`, `rbf_embedding_d{1,2,3}` |
| **Joint distance** | Matern over $\sqrt{d^2_{\text{feat}} + \alpha \cdot d^2_{\text{cat}}}$ | **`joint_hamming_matern`** ✓, `joint_chain_matern`, `joint_hamming_matern_nu{05,25}`, `joint_embedding_matern_d{1,2,3}` |
| Additive hybrid | sum of two source kernels (high complexity) | `additive_joint_hamming_rbf_d2`, `additive_joint_hamming_nu25_rbf_d2` |

---

## 2. The production kernel: `joint_hamming_matern`

### 2.1 Definition

The kernel computes a joint distance combining feature ARD with a
Hamming-style categorical penalty, then applies Matern_{3/2}:

$$d^2(z_i, z_j) = \underbrace{\sum_{f \in F_{\text{cont}}} \frac{(x_{i,f} - x_{j,f})^2}{\ell_f^2}}_{\text{ARD over the continuous features}} + \alpha \cdot \underbrace{\mathbb{1}[c_i \ne c_j]}_{\text{Hamming on the source dim only}}$$

$$K(z_i, z_j) = \sigma^2 \cdot \text{Matern}_{3/2}\!\big(\sqrt{d^2(z_i, z_j)}\big)$$

where $F_{\text{cont}}$ is the set of **all dimensions except the
Material Source** (composition: Cement, Fly Ash, Slag, Water,
HRWR, Fine Aggregate, Coarse Aggregate; environmental: Temp, Time;
engineered: the F5_alllog log-ratio features) and $c_i, c_j \in \{0, 1, 2\}$
are the Material Source class labels of rows $i$ and $j$. **The
Hamming penalty applies ONLY to the source dim** — every other
input is continuous and enters the ARD distance.

**Parameters**:

* Per-feature ARD lengthscales $\ell_f$ for $f \in F_{\text{cont}}$
  (shared with the standard V2 architecture — same count as the
  deployed model);
* One learnable categorical-penalty $\alpha > 0$ — the **single
  new parameter** compared to a fully-continuous kernel.

**PSD by construction**: the joint distance is the Euclidean
distance on $\mathbb{R}^{|F_{\text{cont}}|} \times \sqrt{\alpha} \cdot
\text{simplex}(d_{\text{cat}})$ — i.e., concatenate the
ARD-scaled continuous features with $\sqrt{\alpha}$ times the
3-class simplex embedding of the source class, then apply Matérn
to the joint Euclidean distance.

### 2.2 Why this kernel — motivation walk

The kernel combines the **best of two failed extremes**:

**Continuous-coordinate (`legacy_continuous_ard`)** — *mathematically
wrong*: treats `MS = 2` as if it were the numeric value 2, so
"distance from MS=0 to MS=1" equals "from MS=1 to MS=2". The
single source ARD lengthscale tries to span both mortar↔Set-2
and (Set-1+2)↔Set-3 contrasts simultaneously. Empirically *works*
on bLOO (680 psi on pre-v5) because the joint Matern shape lets
cross-class similarity decrease non-linearly with feature distance —
but this is bought by absorbing class-discriminative structure
into the source coordinate, which hurts held-out-class
extrapolation (Set-3 LOCO = 2580+ psi with the deployed model).

**Categorical product (Hamming etc.)** — *correctly specified*: the
kernel factorises as $K_{\text{cat}}(c_i, c_j) \cdot K_{\text{features}}$.
Cross-class similarity is a constant scalar $\rho$ shared across
all distinct-class pairs (Hamming) or a few rank-r tasks
(IndexKernel). **Loses the joint topology**: cross-class similarity
is the *same* regardless of how close the features are, because
the kernel factors.

**Joint Hamming Matern** — combines both. Mathematically:

* When $c_i = c_j$: $K = \text{Matern}_{3/2}(d_{\text{feat}})$ —
  identical to ARD-Matern over features.
* When $c_i \ne c_j$: $K = \text{Matern}_{3/2}(\sqrt{d^2_{\text{feat}} + \alpha})$
  — the Matern non-linearity *compresses* both feature and class
  distances together.

The categorical structure is correct (Hamming-style, indistinguishable
distinct classes). The joint topology is restored (Matern over the
joint distance). The parameter count is minimal (1 categorical
parameter).

### 2.3 Why this *specific* form — discoveries along the way

The joint-distance family has many possible variants. The empirical
ablation (§ 4) showed:

| variant | what it tested | result |
|---|---|---|
| `joint_hamming_matern_nu05` (Matern_1/2) | rougher kernel — outlier robust? | best Class-0/1 LOCO, **worst Class-2 LOCO** (1577 psi) |
| **`joint_hamming_matern`** (Matern_3/2) | balanced smoothness | **Pareto-optimal acceptable, simplest** |
| `joint_hamming_matern_nu25` (Matern_5/2) | smoother — best in-distribution? | best bLOO (701 = legacy 702), **fails Class-0 LOCO criterion** (3719 > 1.10× Hamming) |
| `joint_chain_matern` | encode mortar→Set-2→Set-3 ordering | ~identical to Hamming variant (α absorbs ordering via lengthscale) |
| `joint_embedding_matern_d{1,2,3}` | learnable per-class embedding | **dominated** by `joint_hamming_matern`; d=2 has rotational gauge instability |

The default Matern_{3/2} smoothness ν=1.5 is the **Pareto-optimal
acceptable** choice. Smoother kernels (ν=2.5) over-fit in-distribution
and break LOCO Class-0; rougher kernels (ν=0.5) over-smooth out
useful structure on Set-3.

### 2.4 Literature context — what is this kernel called, and is it new?

**Our kernel is not a novel construction; it is the Matérn-3/2
analog of a well-established kernel family** for mixed
continuous-categorical GP regression. The exponential-radial
version goes by three roughly-synonymous names depending on the
community:

* **"Gower-distance kernel"** — engineering BO / scikit-learn / SMT
  community. **Halstrup (2016, TU Dortmund PhD thesis)** introduced
  the construction (Gower 1971 distance plugged into a radial GP
  kernel). **Saves et al. (2024, *Neurocomputing*, arXiv:2211.08262)**
  formally name it the **GD (Gower Distance) kernel** — the most
  parsimonious member of their GD ⊂ CR ⊂ EHH ⊂ FE hierarchy.
* **"Compound Symmetry (CS) / Exchangeable kernel"** — statistics
  community. **Pelamatti et al. (2020, arXiv:2003.03300)**,
  **Roustant et al. (2020, *JCGS*, arXiv:1802.02368)** (the `kergp`
  package), and **Carpintero Perez et al. (2025, arXiv:2510.01840)**
  treat it as the one-parameter limit of more general block-SPD
  task-covariance kernels.
* **"Weighted Hamming kernel"** — algorithm-configuration community.
  **Hutter, Hoos & Leyton-Brown (LION 2011, SMAC)** used
  $\exp(-\sum_\ell \theta_\ell\, (1 - \delta(c_\ell, c'_\ell)))$
  with an exponential radial.

The original Gower (1971) coefficient predates all of these as a
mixed-type distance for clustering.

#### One genuinely novel emphasis: Matérn ν ≠ exp/SE

A subtle point that the literature largely conflates: for an
**exponential / SE** continuous kernel,
$$\exp\!\big(-\tfrac{1}{2}(d^2_{\text{cont}} + \alpha\,\mathbb{1}[c_i \ne c_j])\big) = \underbrace{\exp(-\tfrac{1}{2} d^2_{\text{cont}})}_{K_{\text{cont}}} \cdot \underbrace{\exp(-\tfrac{\alpha}{2}\,\mathbb{1}[c_i \ne c_j])}_{K_{\text{cat}}},$$
so "joint inside one radial" and "$K_{\text{cont}} \cdot K_{\text{cat}}$"
are algebraically identical. For **Matérn**, no such factorisation
exists — $M_\nu(\sqrt{a + b}) \neq M_\nu(\sqrt{a}) \cdot M_\nu(\sqrt{b})$.
Our kernel is the **isotropic Matérn-3/2 on the Gower-embedded joint
space**, which is a strict variant of (and distinct from) the
exponential-radial constructions in Halstrup, Saves, Pelamatti,
etc. Saves (2024) implements switchable Matérn-3/2 / 5/2 / SE / exp
inside their product form, but the joint-vs-product distinction is
not discussed; in BoTorch's `MixedSingleTaskGP` the additive+product
form is the default. Our v5 ablations confirm this distinction
matters empirically: the joint-Matérn form closes ~33 of the 36
psi bLOO gap that the product form (`hamming`) leaves on the table
(see § 5).

#### Tied vs free per-class lengthscales

Our single learnable $\alpha$ corresponds to the **tied / "compound
symmetric"** parameterisation: one parameter shared across all
distinct-class pairs. This is the dominant choice in low-data
mixed-input GPs (Saves GD, Pelamatti CS, Roustant CS, COMBO
complete-graph, SMAC). The literature has consistently found that
for ≲ 6 categorical levels and limited data, the tied form **matches
free-parameter alternatives** (one-hot ARD with per-dim ℓ, the
hypersphere decomposition, latent-variable LVGP) while being far
more numerically stable (Saves 2024; Pelamatti 2018;
Cuesta-Ramirez 2021). Our v5 ablation directly confirms this:
`joint_embedding_matern_d{1,2,3}` (the LVGP-style free-embedding
generalisation) is **Pareto-dominated** by the tied-α form on this
3-class, ~95-in-class-composition dataset (§ 4.3).

#### Alternative parameterisations in the same family

For completeness, the alternative mixed-input GP approaches
(largely *dominated* on v5 — see § 4):

* **Sum-of-kernels (additive)**: $K_{\text{cont}} + K_{\text{cat}}$
  or $K_{\text{cont}} + K_{\text{cont}} \cdot K_{\text{cat}}$.
  **Ru et al. (CoCaBO, ICML 2020, arXiv:1906.08878)** propose a
  mixture $(1 - \lambda)(K_h + K_x) + \lambda K_h K_x$ with trained
  $\lambda$. Pure additive is critiqued in CoCaBO §4.2 as having
  "limited expressiveness — translates in practice to learning a
  single common trend over $x$, and an offset depending on $h$".
  Our additive hybrid (§ 4.5) is Pareto-improving but Occam-rejected.
* **Latent-variable GP (LVGP)**: **Zhang, Tao, Apley & Chen (2020,
  *Technometrics*, arXiv:1806.07504)** propose learning a low-D
  latent embedding $x_c \in \mathbb{R}^2$ for each class. Our
  `rbf_embedding_d{1,2,3}` (and `joint_embedding_matern_d{1,2,3}`)
  variants are direct LVGP-style implementations; they are
  Pareto-dominated by the tied-α form on this dataset (§ 4.2, § 4.3).
  **Iyer et al. (LVGP-BO, 2019, arXiv:1907.02577)** applied LVGP to
  materials BO.
* **One-hot ARD**: **Garrido-Merchán & Hernández-Lobato (2020,
  *Neurocomputing*, arXiv:1805.03463)** one-hot encode and use ARD
  per indicator dim — free per-class lengthscales rather than tied.
  Our `onehot_ard` baseline is dominated by `hamming` and
  `joint_hamming_matern` (§ 4.1).
* **Diffusion on graphs**: **Oh et al. (COMBO, NeurIPS 2019,
  arXiv:1902.00448)** use ARD diffusion on a Cartesian-product
  graph; for a complete graph this reduces to tied Hamming. Not
  tested on v5.
* **Embedding inside the radial**: **Yousefpour et al. (GP+, 2023,
  arXiv:2312.07694)** combine continuous ARD with learnable
  categorical embedding inside the radial — essentially a
  generalised `joint_embedding_matern`. Pareto-dominated by tied-α
  on v5.
* **Linear-mixture of edit-distances**: **Pu et al. (WEGP, AISTATS
  2025, arXiv:2503.02630)** learn a positive linear combination of
  edge-Distance Matrices per categorical variable — generalises
  Gower with one base EDM = our tied-α.

#### What's distinct about our use case

To our knowledge, **no prior GP-for-concrete-strength paper has
used a Matérn-of-Hamming categorical kernel.** The two recent
arXiv references on GP for concrete (`arXiv:2310.18288`,
`arXiv:2603.21525`) treat SCMs (supplementary cementitious materials)
as **continuous fractions of binder mass** rather than as categorical
material-source labels. The 3-class Material Source label
(mortar / Set-2 Heidelberg-Class-C / Set-3 Amrize-Class-F) at
~ 95 in-class compositions is exactly the regime — **small data,
few exchangeable levels** — where the literature's empirical
consensus (Cuesta-Ramirez 2021; Pelamatti 2018; Saves 2024) is
that the compound-symmetric / GD / tied-α form is hard to beat.
So our kernel choice is both established in the BO literature
*and* empirically defensible from our own ablations on this dataset.

#### Suggested canonical name

We adopt **`joint_hamming_matern`** as the variant name in the
`boxcrete` codebase. For citation purposes, defensible names in
decreasing community recognition are:

1. **Gower-Matérn kernel** (extends Halstrup / Saves GD to Matérn).
2. **Compound-symmetric Matérn kernel** (statistics framing).
3. **Simplex-embedded Matérn kernel** (LVGP-symmetric-limit framing).
4. **Joint-metric Matérn kernel** (descriptive; what the
   variant name in our code encodes).


---

## 3. Evidence

### 3.1 The Pareto frontier (the decision framework)

A variant is **Pareto-optimal** on (bLOO RMSE, Class-2 LOCO RMSE) if
no other variant beats it on both axes simultaneously. Across all
11+ kernels we tested:

| rank | variant | bLOO RMSE | Class-2 LOCO RMSE | passes acceptance? |
|---|---|---|---|---|
| 1 | `joint_hamming_matern_nu25` | **701** | 871 | ✗ (Class-0 LOCO fails) |
| 2 | **`joint_hamming_matern`** | **717** | 1113 | **✓** |
| 3 | `additive_joint_hamming_nu25_rbf_d2` | 720 | **888** | ✓ |
| 4 | `rbf_embedding_d2` | 757 | **849** | ✓ |

Only 4 variants sit on the frontier; everything else is dominated.
Among **acceptable** (passes the pre-registered ≤ 110 % LOCO ratio
criterion), 3 corners remain:

1. **`joint_hamming_matern`** — best LOO + best PIT-KS + simplest
   (1 categorical param).
2. **`additive_joint_hamming_nu25_rbf_d2`** — Pareto-improving
   over either single kernel on (bLOO, Class-2 LOCO), but **2×
   kernel complexity** (sum of two source kernels). Best LOO
   in the suite (488 psi).
3. **`rbf_embedding_d2`** — best Class-2 LOCO (849 psi blind
   extrapolation).

**Per Occam's Razor**, `joint_hamming_matern` is the production
choice — it has the same parameter count as the previous Hamming
default while strictly improving on it.

### 3.2 Performance across the kernel family

In-distribution v5 (3-seed means; all variants deterministic on v5):

| variant | LOO RMSE | bLOO RMSE | bLOO PIT-KS | S12-bLOO RMSE | # cat params |
|---|---|---|---|---|---|
| `legacy_continuous_ard` (deployed-on-v5) | 539 | **702** | 0.040 | 836 | (continuous) |
| `hamming` (prior production default) | 510 | 738 | 0.039 | 871 | 1 |
| `indexkernel_r2` | 509 ± 13 | 738 ± 13 | 0.047 ± 0.008 | 876 ± 17 | 6 (seed-fragile) |
| `rbf_embedding_d1` | 497 | 761 | 0.037 | 885 | 2 |
| `rbf_embedding_d2` | 497 | 757 | 0.036 | 878 | 3 |
| `rbf_embedding_d3` | 511 | 735 | 0.040 | 868 | 5 |
| **`joint_hamming_matern`** | **495** | **717** | **0.030** | 835 | **1 (α)** |
| `joint_chain_matern` | 495 | 720 | 0.033 | 839 | 1 (+ assumed ordering) |
| `joint_hamming_matern_nu05` | 507 | 720 | 0.060 | 850 | 1 |
| `joint_hamming_matern_nu25` | 523 | **701** | 0.030 | **817** | 1 |
| `joint_embedding_matern_d1` | 495 | 724 | 0.034 | 844 | 2 |
| `joint_embedding_matern_d2` | 496 | 716 | 0.031 | 833 | 3 |
| `joint_embedding_matern_d3` | 495 | 716 | 0.030 | 832 | 5 |
| `additive_joint_hamming_rbf_d2` | **480** | 746 | 0.031 | 878 | 1 + 3 + outputscale |
| **`additive_joint_hamming_nu25_rbf_d2`** | 488 | 720 | 0.036 | 860 | 1 + 3 + outputscale |

(`onehot_ard` and `indexkernel_r{1,3}` omitted; they are uniformly
dominated by the variants above.)

### 3.3 Held-out-class extrapolation (LOCO)

Held-out-class RMSE (3-seed means; pre-registered acceptance ≤ 1.10
of `hamming` baseline per class):

| variant | Class-0 RMSE | Class-1 RMSE | Class-2 RMSE | passes? |
|---|---|---|---|---|
| `legacy_continuous_ard` | 2882 | 2912 | 1139 | n/a (deployed baseline) |
| `hamming` | 2730 | 2649 | 1319 | reference (1.00× all) |
| `indexkernel_r2` | **4905 ± 1984** ⚠ | 2448 ± 41 | **3250 ± 742** ⚠ | ✗ (catastrophic, seed-fragile) |
| `rbf_embedding_d1` | 2891 | 2855 | 874 | ✓ |
| **`rbf_embedding_d2`** | 2824 | 2705 | **849** | ✓ |
| `rbf_embedding_d3` | 3121 | 3166 | 1170 | ✗ (Class-1 fails) |
| `joint_hamming_matern_nu05` | **2379** | **2228** | 1577 | ✗ (Class-2 1.20× ratio) |
| **`joint_hamming_matern`** | 2853 (1.045×) | 2736 (1.033×) | **1113** (0.844×) | **✓** |
| `joint_hamming_matern_nu25` | 3719 (1.36×) ⚠ | 2731 | 871 | ✗ (Class-0 fails) |
| `joint_chain_matern` | 2853 | 2807 | 1113 | ✓ |
| `joint_embedding_matern_d1` | 3352 | 2736 | 1320 | ✓ (Class-0 1.23× ratio — fails) |
| `joint_embedding_matern_d2` | **4148** ⚠ | 2795 | 1123 | ✗ (Class-0 1.52× — gauge instability) |
| `joint_embedding_matern_d3` | 2810 | 2805 | 1304 | ✓ (Class-1 1.06×) |
| **`additive_joint_hamming_nu25_rbf_d2`** | 2958 (1.083×) | 2830 (1.068×) | **888** (0.673×) | **✓** |

**Key LOCO observations**:

* **IndexKernel fails catastrophically on Class-0 and Class-2** due
  to random-init of the unseen class's covariance row (±1984 psi
  seed std on Class-0). Disqualified.
* **`joint_hamming_matern_nu25` fails Class-0** (3719 / 2730 =
  1.36×, well over the 1.10 threshold) despite being the bLOO
  champion. The smoother Matern over-fits.
* **`joint_embedding_matern_d2` has a 4148-psi Class-0 LOCO**
  (gauge-rotation instability; d=1 and d=3 don't show this).
* **`joint_hamming_matern_nu05` has the best Class-0+1 LOCO** but
  fails Class-2 — rough kernel over-smooths Set-3 structure.

### 3.4 Headroom: how close to the noise floor are we?

Comparing the best model's residuals to the data's measurement noise:

| noise source | value (psi) | model RMSE for comparison |
|---|---|---|
| Within-batch (3-cylinder) Strength(Std), median | 189 | in-distribution bLOO: 701-717 psi |
| Within-batch Strength(Std), Class-2 (Set-3) median | 133 | (lowest-noise class) |
| Within-batch Strength(Std), Class-0 (mortar) median | 324 | (highest-noise class) |
| Between-batch std (M60/M61 twin pair, mortars) | 1486 | Class-0 LOCO: 2730-2853 psi |

**Interpretation**:

* **In-distribution bLOO 701 ≈ 3.7× within-batch noise (189 psi)**.
  Limited architectural headroom remains: the kernel is already
  exploiting most of the deterministic structure.
* **Class-0 LOCO 2730 ≈ 1.8× between-batch noise (1486 psi)**.
  Most Class-0 LOCO error is *irreducible* — limited by the small
  mortar-batch population.
* **Class-2 LOCO 849 ≈ 6× within-batch Class-2 noise (133 psi)**.
  More architectural headroom remains here — but the GP audit
  (§ 4.7) shows the homoscedastic noise estimate (σ_gp ≈ 370 psi)
  is ~5× larger than σ_meas for Set-3, meaning a chunk of this
  apparent "error" is over-conservative noise modelling.

### 3.5 Twin-drop sensitivity (M60/M61 anchor effect)

The M60/M61 mortar pair (formerly Mix_80/Mix_81) has identical
recipes but factor-2.3 strength differences — a between-batch
variance ~1486 psi at the same composition. Dropping the high
twin (M61) on `joint_hamming_matern`:

| metric | with M61 | drop M61 | Δ |
|---|---|---|---|
| LOO RMSE | 495 | **462** | −33 (outlier removed) |
| bLOO RMSE | 717 | 719 | +2 (tied) |
| LOCO Class-2 | 1113 | **1251** | **+138 (worse)** |

**Recommendation: keep M61.** The twin acts as a held-out-class
anchor; dropping it improves in-sample fit but hurts the most
important held-out test (LOCO Class-2). Same anchor-utility pattern
that earlier `legacy_continuous_ard` and `rbf_embedding_d2`
twin-drop ablations showed.


### 3.6 Strength-curve monotonicity in the early-hour extrapolation region — `GATE_TAU` ablation

Concrete strength is physically **monotone increasing in time** (cement
hydration is one-way). In post-merge testing of v5 +
`joint_hamming_matern`, a user reported a strength curve for the
Set-3 Pareto-corner mix `(70/235/46)` at W/B = 0.40, no HRWR, that
peaks at ~644 psi at t ≈ 3 h, then drops to ~269 psi at t = 1 d
(a 375-psi peak-to-valley drop), then recovers monotonically to
2890 psi at 28 d. The drop is in the `t < 1 d` extrapolation region
where there is **no training data** for any composition.

Audit on the full 144-row catalog: **~33 % of compositions** show
some non-monotonicity in the explorer's `t ∈ [0.04, 28]` d display
range at `GATE_TAU = 0.05`. Worst single-step drop on a 100-point
dense log-spaced grid: 39 psi. Cumulative peak-to-valley drops can
be much larger (e.g. the 375 psi the user reported).

**Mathematical cause**. The posterior mean factors as
$\mu(t^*) = h(t^*) \cdot \sum_i \alpha_i \cdot h(t_i) \cdot [K_\text{blind} + K_\text{specific} + K_\text{rbf-time}](z^*, x_i)$.
For training points with $t_i \ge 1$ d the gate $h(t_i) \approx 1$,
so $\mu(t^*) \approx h(t^*) \cdot G(t^*)$ where $G$ is a kernel-
weighted sum over training rows. For **feature-extrapolated** test
compositions, $\alpha = K_\text{lik}^{-1} y$ has mixed-sign entries
that, summed with the smooth-in-$t$ kernel components, produce
weighted-residual oscillation in $t^*$. The gate $h(t^*)$ adds a
monotonic ramp on top, but with $\tau = 0.05$ the gate saturates
by $t_\text{norm} = 0.3$ (raw $t \approx 1$ d) and stops dampening
the kernel oscillation earlier than that.

**Ablation**: vary `GATE_TAU` and measure both fit quality (LOO RMSE,
bLOO RMSE) and curve monotonicity (% catalog compositions with any
drop > 1 psi over $t \in [0.04, 28]$ d, max single-step drop in psi).

| `GATE_TAU` | LOO RMSE | bLOO RMSE | % mix w/ drop | max single-step drop |
|---|---|---|---|---|
| 0.020 | 502.3 | 507.1 | 42.4% | 44.1 psi |
| 0.050 (pre-fix production) | 501.9 | 506.6 | 32.6% | 39.0 psi |
| **0.100 (v5 production default)** | **501.3** | **505.9** | **22.2%** | **29.2 psi** |
| 0.150 | 507.2 | 511.6 | 20.8% | 23.6 psi |
| 0.200 | 520.9 | 525.1 | 19.4% | 27.0 psi |
| 0.300 | 562.5 | 566.3 | 17.4% | 32.2 psi |

**Headline**: `GATE_TAU = 0.10` **strictly Pareto-dominates** the
pre-fix `0.05` on every metric:
* `% mix w/ drop`: 32.6 % → 22.2 % (relative −32 %)
* `max single-step drop`: 39 → 29 psi (−26 %)
* `LOO RMSE`: 501.9 → 501.3 (slight IMPROVEMENT)
* `bLOO RMSE`: 506.6 → 505.9 (slight IMPROVEMENT)

Going further (τ ≥ 0.15) reduces the drop fraction by only ~1 pp
per step while LOO RMSE regresses materially (+5 psi at 0.15, +19 at
0.20, +61 at 0.30). The drop fraction **floors at ~17 %** even at
extreme τ — gate-tuning alone cannot fully eliminate non-monotonicity
because the underlying multi-Matern posterior's weighted-residual
oscillation has its own spatial structure that no monotonic envelope
can fully mask.

**Production decision**: `GATE_TAU = 0.10` (was `0.05`). The kernel
test (`test/test_curve_monotonicity.mjs`) is calibrated to the new
floor (gate `< 30 %` with drops, < 60 psi max drop) and serves as a
regression watchdog for further architecture changes.

**Open follow-up (not blocking)**: reduce the ~22 % residual floor.
Options identified but not yet pursued:
* Soft monotonicity penalty in the MLE objective (sample
  `(composition, time)` pairs, penalise negative ∂μ/∂t).
* Post-hoc UI clamp (running-max in `predictStrengthCurve`; zero
  metric impact, but hides the underlying model issue).
* Monotonic-by-construction time kernel (e.g. integrate a Matern
  kernel over time → strictly increasing). Larger research effort.

Full ablation table and methodology in
`experiments/ABLATION_GATE_TAU.md`.

### 3.7 Time-only kernel structure ablation — RBF(t) is redundant

Hypothesis (from a user during post-merge testing): the additive
`RBF(t)` time-only branch is the source of non-monotonicity.

| variant | LOO RMSE | bLOO RMSE | % drop | max ss-drop |
|---|---|---|---|---|
| **rbf_time** (production) | 501.3 | 505.9 | **22.2%** | 29.2 |
| matern52_time | 501.0 | 505.7 | 22.9% | 29.0 |
| matern32_time | 501.8 | 506.5 | 22.2% | 29.2 |
| linear_time | 502.4 | 506.9 | 24.3% | 29.9 |
| **no_time** (drop) | 502.1 | 506.7 | **22.2%** | 29.1 |

**Falsified.** All variants give essentially identical metrics
(Δ LOO < 1.4 psi, Δ % drop < 2 pp). Crucially, **dropping the
additive time branch entirely (`no_time`) preserves all metrics**
within noise.

This is a substantial departure from the V2-era finding (where
`RBF(t)` contributed +20 psi block-LOO with `legacy_continuous_ard`).
With `joint_hamming_matern`, the joint kernel already captures
composition × time interaction internally (time is one of its ARD
dims, alongside composition); the additive `RBF(t)` channel has
nothing left to explain.

**Implication**: the v5 production kernel could be simplified by
dropping the redundant additive time branch with zero metric cost.
Tracked as a parsimony cleanup follow-up (verified safe on v5; pre-v5
verification pending).

Full table and methodology in `experiments/ABLATION_TIME_KERNEL.md`.

### 3.8 V2 baseline comparison — non-monotonicity is not new

The user asked: was the deployed V2 model afflicted by similar
non-monotonicity? If so, this PR should be evaluated as an
*incremental* improvement vs an architectural regression.

| configuration | LOO | bLOO | % drop | max ss-drop |
|---|---|---|---|---|
| **v5 + joint_hamming_matern** (current production, τ=0.10) | **501** | **506** | **22.2%** | **29.2** |
| v5 + V2 architecture (legacy_continuous_ard) | 542 | 546 | 30.6% | 35.2 |
| **pre-v5 + V2 architecture** (deployed V2 production, τ=0.10) | **496** | **499** | **27.8%** | **30.7** |
| pre-v5 + joint_hamming_matern | 501 | 506 | 22.9% | 29.2 |

**Headline**: The deployed V2 production model has **27.8 %**
non-monotonic curves at the same dense time grid — essentially the
same problem we found in v5. The bug was already in production;
nobody had measured it before.

**This PR is a strict improvement over V2 on monotonicity**:
27.8 % → 22.2 % (relative −20 %), with a slight LOO regression
(+5 psi) attributable to the harder 3-class task.

The 22 % floor is an architectural limit of the multi-Matern +
GatedKernel decomposition, common to both V2 and v5. The
joint_hamming_matern improves on legacy_continuous_ard by ~−27 %
relative drop fraction *on both data versions*.

**Lever-sensitivity ranking**:
1. Source kernel (legacy → joint): −27 % drop fraction.
2. Gate τ (0.05 → 0.10): −32 % drop fraction.
3. Time-only additive kernel: irrelevant.
4. Data version (v5 vs pre-v5): irrelevant for monotonicity.

Full ablation table in `experiments/ABLATION_V2_BASELINE.md`.

### 3.9 Post-fit time-lengthscale sensitivity — kernel-of-time is not the floor

The user asked: would tighter time-lengthscale priors during the fit
reduce non-monotonicity? Diagnostic: take the production model and
multiply ALL time-direction lengthscales (blind Matern's time-dim,
joint kernel's time-dim, additive RBF(t)) by a multiplier
`m ∈ {0.1, 0.5, 1, 2, 5, 10, 100}`. No refit; just post-hoc
perturbation.

| time-ell × m | % drop | max ss-drop |
|---|---|---|
| 0.1× | **94.4%** | 1253 psi |
| 0.5× | 27.1% | 56.9 |
| **1.0× (baseline)** | **22.2%** | **29.2** |
| 2× | 24.3% | 24.5 |
| 5× | 23.6% | 24.1 |
| 10× | 22.9% | 24.1 |
| 100× | 22.9% | 24.1 |

* **Going SHORTER (m < 1) catastrophically destroys monotonicity** —
  the kernel-of-time becomes spiky, weighted-residual oscillation
  amplifies. Confirms the gate-tau finding that "short t-direction
  scales are bad".
* **Going LONGER (m ≥ 2) does NOT eliminate non-monotonicity**.
  Even at m=100× (kernel essentially constant in t), drop fraction
  stays at 22.9 %. The max single-step drop reduces by ~17 %
  (29 → 24 psi), but the FRACTION-affected floor doesn't move.

**Smoking gun**: the residual ~22 % floor is **not from the
kernel-of-time component** at all. With kernel constant in t, the
posterior reduces to `μ(t*) ≈ h(t*) · const`, which should be
strictly monotonic by `h`'s shape — but it isn't. The remaining
non-monotonicity must come from **other sources** (likely the gate
× α-sign-structure interaction at training data points whose
`h(t_i)` differs slightly across `t_i ∈ {1, 3, 7, 28}` d, combined
with feature extrapolation).

**Implication for mitigation**: tightening time-lengthscale priors
is **not a viable fix**. The remaining viable paths are output-side
(UI isotonic projection — zero metric impact) or structural prior
change (Verhulst mean function or virtual derivative observations).
Both tracked as research follow-ups; not blocking this PR.

Full diagnostic in `experiments/DIAGNOSE_TIME_LENGTHSCALE_SENSITIVITY.md`.

---

## 4. What we tried and why it didn't make production

This section documents the architectural and ablation experiments
that did *not* end up in the production recommendation. Most are
**negative results** — they should NOT distract from the production
choice but are recorded here for completeness and to motivate the
follow-up directions.

### 4.1 Categorical-product baselines (dominated)

* **`hamming`** (CategoricalKernel × ARD-Matern): the **previous
  production default**. Joint Hamming Matern dominates it on every
  in-distribution metric (LOO 510 → 495, bLOO 738 → 717, PIT-KS
  0.039 → 0.030). Same parameter count, so Hamming is strictly
  superseded.
* **`indexkernel_r{1,2,3}`**: rank-r task covariance matrix. The
  unseen-class row of the covariance matrix stays at random init
  (no gradient signal), causing **catastrophic seed sensitivity**
  on LOCO: ±1948 psi seed std on Class-0 holdout for r=2. Disqualified
  for production. Detailed analysis in
  `experiments/RBF_EMBEDDING_LOCO.md`.
* **`onehot_ard`**: source dim one-hot expanded into 3 binary cols
  + ARD-Matern. Dominated by Hamming (loses categorical structure
  while paying for 3 separate lengthscales).

### 4.2 RBF embedding family (Pareto-dominated except blind LOCO Class-2)

The RBF embedding kernel
$K_{\text{cat}}(c_i, c_j) = \exp(-\|x_{c_i} - x_{c_j}\|^2 / 2\ell^2)$
with learnable per-class embeddings $x_c \in \mathbb{R}^d$ was a
research direction motivated by Snoek et al. (2012)
latent-variable multi-task BO. Results:

* **`rbf_embedding_d2`** is the **Pareto-corner for Class-2 LOCO**
  (849 psi, beats Hamming's 1319). It was the *prior* production
  recommendation before the joint-distance family was tested.
* **`rbf_embedding_d{1,3}`**: dominated by d=2.
* **Fixed-ℓ + linear-init combinations**: the `_fixed_ell` variants
  drop the learnable lengthscale (a redundant DOF given embedding
  scale). Closes ~30 psi of the in-distribution bLOO gap but
  **catastrophically regresses Class-2 LOCO** when combined with
  `_linear_init` (2425 psi vs 770 with learnable ℓ) — the fixed
  lengthscale and integer-coordinate init together place the
  held-out class too far from the seen training data for the
  Matern bridge to span. See `experiments/RBF_EMBEDDING_LENGTHSCALE_INIT_ABLATION.md`.

**Learned embeddings — physical interpretation** (MLE-fitted on v5,
best seed):

| dim | x_0 (mortar) | x_1 (Set-2) | x_2 (Set-3) | ℓ |
|---|---|---|---|---|
| d = 1 | 0 | **−1.257** | **+1.672** | 0.648 |
| d = 2 | (0, 0) | (1.058, 0) | (0.023, −1.953) | 0.360 |
| d = 3 | (0, 0, 0) | (1.352, 0, 0) | (0.007, −2.062, 0) | 0.280 |

In every dim the optimiser places **mortar between Set-2 and Set-3**
in embedding space, not at one end — a physically sensible
chemistry ordering (mortar is the binder-paste base for both
concretes; the two aggregate types differ from each other more
than each differs from mortar). This finding informed why
`joint_chain_matern` (which encodes a linear chain ordering)
performs similarly to `joint_hamming_matern` — the data already
exhibits the chain topology and α absorbs the ordering through
lengthscale adaptation.

The `rbf_embedding_d2` Class-2 LOCO advantage is real (849 vs
joint_hamming_matern's 1113) but applies to a scenario — **blind
extrapolation to a held-out class** — that is not the production
deployment. v5 has 252 Set-3 rows in training; in-distribution
metrics are the relevant test for routine BO over existing classes.

### 4.3 Joint embedding (learnable per-class coordinates)

`joint_embedding_matern_d{1,2,3}` replaces the Hamming penalty with
a learnable embedding-distance penalty:
$d^2_{\text{cat}}(c_i, c_j) = \|x_{c_i} - x_{c_j}\|^2$ with
$x_c \in \mathbb{R}^d$ co-optimised. d=1 is the natural categorical
sibling of `legacy_continuous_ard` (treats source as a *learnable*
per-class scalar instead of the fixed integer label).

**Result**: dominated by `joint_hamming_matern`. The extra
embedding flexibility (2-5 free parameters) is not load-bearing on
v5 — the Hamming-style single-α formulation already captures the
relevant categorical structure. **d=2 also exhibits a rotational
gauge instability** producing 4148 psi on Class-0 LOCO (vs d=1's
3352 and d=3's 2810).

### 4.4 Chain Hamming (encoded class ordering)

`joint_chain_matern` replaces the binary Hamming penalty with a
squared integer-label distance $(i-j)^2$, encoding the natural
mortar → Set-2 → Set-3 chemistry ordering. **Result**: identical
metrics to `joint_hamming_matern` on Class-0 and Class-2 LOCO,
slightly worse on Class-1. The α parameter and lengthscale
adaptation already absorb whatever ordering the chain hardcodes —
the chain assumption is redundant on v5.

### 4.5 Additive hybrid kernels (Pareto-improving but Occam-rejected)

`additive_joint_hamming_nu25_rbf_d2 = ScaleKernel(joint_hamming_matern_nu25) + ScaleKernel(rbf_embedding_d2)`
**is Pareto-improving** on (bLOO, Class-2 LOCO):

| variant | bLOO | Class-2 | params |
|---|---|---|---|
| `joint_hamming_matern` | 717 | 1113 | minimal |
| **`additive_joint_hamming_nu25_rbf_d2`** | **720** | **888** | ~2× kernel (two ScaleKernels) |
| `rbf_embedding_d2` | 757 | 849 | medium |

Within 3 psi of `joint_hamming_matern` on bLOO AND 225 psi better
on Class-2 LOCO. **Passes all pre-registered LOCO criteria**.

**Why not production**: the hybrid doubles kernel complexity (two
ScaleKernels, two outputscales, two ARD lengthscale sets). Per
Occam's Razor, the marginal Class-2 LOCO improvement (225 psi at
≈2× complexity) is not worth shipping for the production scenario
(where v5 already has 252 Set-3 rows). The hybrid is kept as a
registered ablation for deployments where blind Class-2 LOCO
matters most.

### 4.6 `include_blind=False` — blind branch is load-bearing

The V2 strength kernel has a class-INDEPENDENT "blind Matern"
branch added to the source-aware kernel. Removing it would
simplify the architecture by ~10 hyperparameters. **Result**: the
blind branch is **kernel-dependent in its load-bearing behavior**:

| base kernel | bLOO with blind | bLOO no-blind | Class-2 LOCO with blind | Class-2 no-blind |
|---|---|---|---|---|
| `joint_hamming_matern` | **717** | 760 (+43) | **1113** | 1548 (+435) |
| `rbf_embedding_d2` | 757 | **722** (−35) | **849** | 1604 (+755) |
| `hamming` | 738 | **706** (−32) | 1319 | **1227** (−92) |

For `joint_hamming_matern` (the production winner), the blind
branch is essential everywhere. **Keep it.** Detailed analysis in
`experiments/NO_BLIND_ABLATION.md`.

### 4.7 Per-class heteroscedastic noise — negative result

Motivated by a noise audit (`experiments/NOISE_AUDIT.md`) showing
the homoscedastic GP fits σ_gp ≈ 370 psi (~3.4× median measurement
noise), with the worst over-estimate on Class-2 (σ_gp / σ_meas ≈ 4.9×).
Tested `PerClassGatedGaussianLikelihood` with 3 separate learnable
σ_c parameters.

**Result**: MLE finds σ_0 = 509 / σ_1 = 406 / σ_2 = 181 psi —
correctly identifying Set-3 as low-noise — but **performance is
uniformly worse**:

| metric | homoscedastic | per-class | Δ |
|---|---|---|---|
| LOO RMSE | **495** | 500 | +5 |
| bLOO RMSE | **717** | 743 | +26 |
| LOCO Class-0 | **2853** | **3596** | **+743** ⚠ |
| LOCO Class-2 PIT-KS | 0.261 | 0.382 | worse |

**Root cause**: the M60/M61 mortar twin pair has between-batch
variance ~1486 psi. MLE absorbs this into σ_0 = 509 psi, which
tells the optimiser "ignore mortar observations" → Class-0 LOCO
crashes by 743 psi. The per-class formulation cannot separate
"true class noise" from "outlier-driven batch noise". A robust
likelihood (Student-t) or explicit batch random effect is the
right principled fix; per-class σ_c alone is the wrong tool.
See `experiments/PER_CLASS_NOISE_ABLATION.md`.

### 4.8 Smoother Matern (`joint_hamming_matern_nu25`)

bLOO 701 (matches `legacy_continuous_ard`'s 702 — completely closes
the joint-vs-product architectural gap!) and Class-2 LOCO 871
(within 22 psi of `rbf_embedding_d2`). But **fails the LOCO Class-0
acceptance** (3719 / 2730 = 1.36× > 1.10 threshold). The smoother
kernel over-fits in-distribution at the cost of held-out mortar
robustness.

This variant *would* be the production winner if the Class-0
acceptance criterion were relaxed; it's a useful research signpost
showing the joint topology is the right architectural family.

---

## 5. The remaining bLOO regression vs deployed: 3-way decomposition

The deployed `pre-v5 + legacy_continuous_ard` model has bLOO RMSE
680 psi; the recommended `v5 + joint_hamming_matern` is 717 — a
+37 psi regression. Where does it come from?

| component | bLOO cost | what it is |
|---|---|---|
| **3-class vs 2-class labelling** (just the label, same data and architecture) | **≈ 0 psi** | label change alone is free (H4 of `V5_VS_PRE_V5_INVESTIGATION.md`: 702 → 701 with 2-class relabel on v5) |
| **Data change pre-v5 → v5** (legacy architecture on both; same row count, identical strength values) | **+22 psi** | M60/M61 twin-anchor utility differs between datasets — twin-drop ablation shows both datasets fit to ≈ 735 with the twin removed |
| **Architecture: continuous-ARD → joint-Hamming Matern** (same v5 data) | **+15 psi** | paid intentionally to switch from a mathematically mis-specified continuous source coordinate to a correctly-categorical kernel; **dramatically smaller** than the +33 psi cost of the simpler categorical-product topology |
| **Total** | **+37 psi** | |

**None of the three components is a defect.** The +22 psi data
component is anchor-utility (vanishes when M61 is dropped on both
sides). The +15 psi architecture component is paid intentionally
for **proper categorical handling** that buys:

* **−207 psi LOCO Class-1 RMSE** (2912 → 2705 vs `rbf_d2`; 2736
  with `joint_hamming_matern`)
* **−26 psi LOCO Class-2 RMSE** (1139 → 1113 with
  `joint_hamming_matern`; deeper to 849 with `rbf_embedding_d2` if
  Class-2 LOCO is the deployment-critical metric)
* **−38 psi LOO RMSE** (533 → 495)
* **−0.007 bLOO PIT-KS** (0.037 → 0.030)

The joint kernel topology — *not* the simpler categorical-product
topology — closes most of the original +33 psi architectural gap
(the joint-Matern + Hamming combination preserves the joint kernel
shape that `legacy_continuous_ard` had).

---

## 6. Held-out-class extrapolation: how each kernel handles unseen classes

When training on classes $\{A, B\}$ and predicting on held-out class
$C$, the parameter representing $C$ has zero gradient signal during
training. Different kernels handle this differently:

| kernel | $C$'s representation | What happens to $C$ during training? | Effect on LOCO |
|---|---|---|---|
| `legacy_continuous_ard` | $C$'s numeric coordinate (fixed) | nothing — coord is fixed input | extrapolated via learned source ARD lengthscale |
| `hamming` | scalar $\rho$ between distinct classes | $\rho$ fit from observed pairs only | $K(c_{\text{seen}}, C) = \rho$ uniformly |
| `indexkernel_r2` | $C$'s row of $B$ matrix | **zero gradient** → stays at random init | **seed-fragile** (±1984 psi on Class-0) |
| **`joint_hamming_matern`** | $C$'s identity (binary Hamming) | n/a — categorical penalty is just $\alpha \cdot \mathbb{1}[c_i \ne c_j]$ | $K(c_{\text{seen}}, C) = \text{Matern}_{3/2}(\sqrt{d^2_{\text{feat}} + \alpha})$ — **deterministic** |
| `rbf_embedding_d{k}` | $C$'s embedding vector $x_C$ | **zero gradient** → stays at equilateral-simplex init | **deterministic** (simplex init is principled, equidistant) |

The production `joint_hamming_matern` and the alternative
`rbf_embedding_d2` both have **deterministic blind-LOCO behaviour**
(seed std = 0 on every metric). Only `indexkernel` carries the
random-init failure mode.

Three regimes worth distinguishing:

| regime | description | what happens to $C$'s parameters |
|---|---|---|
| Fully blind LOCO | 0 rows of class $C$ in training | stays at init (no gradient) |
| Few-shot LOCO | $k \ll N_C$ rows of class $C$ in training | gets gradient signal — adapts |
| Full LOCO | all rows of class $C$ are test set | identical to fully blind at the parameter level |

Cases (1) and (3) are parameter-equivalent — both freeze $C$'s
parameters since no class-$C$ rows reach the loss. The existing
LOCO numbers are case (3) experiments. Few-shot LOCO (case 2) is
a non-blocking follow-up that would inform 4th-material-class
deployment scenarios.

---

## 7. Methodology

### 7.1 Pre-registered acceptance criteria

Per `~/.llms/plans/three_class_and_lengthscale_prior.plan.md` §"Commit 6":

| Criterion | Target | `joint_hamming_matern` actual | Pass? |
|---|---|---|---|
| Per-class LOCO RMSE within 110 % of Hamming | ≤ 1.10 ratio | 1.045 / 1.033 / **0.844** | ✓✓✓ |
| Held-out Set-3 RMSE | < 1500 psi | 1113 psi | ✓ |
| Seed std on LOCO RMSE | ≤ 50 psi (deterministic) | **0** on every cell | ✓ |
| LOCO PIT-KS within 1.5× Hamming | ≤ 1.5 ratio | 1.21 / 1.34 / **0.81** | ✓✓✓ |
| In-distribution LOO RMSE strictly better than Hamming | < 510 | **495** | ✓ |
| In-distribution bLOO RMSE strictly better than Hamming | < 738 | **717** | ✓ |
| In-distribution PIT-KS strictly better than Hamming | < 0.039 | **0.030** | ✓ |
| Max non-grouped raw lengthscale | < 100 (no railing) | within prior cap | ✓ |

### 7.2 Evaluation protocol

* **LOO RMSE**: standard leave-one-out (per-row holdout via the
  Sundararajan-Keerthi closed form). Measures in-distribution
  per-row prediction accuracy.
* **bLOO RMSE**: block-LOO — composition-grouped holdout (all rows
  of one (composition + temperature) block held out at once,
  using all of its strength curves at training time would cheat).
  Apples-to-apples comparison vs the deployed pre-v5 model's
  bLOO.
* **S12-bLOO RMSE**: bLOO restricted to compositions in
  Sets 1+2 (mortar + Set-2 concrete) — the only blocks the
  deployed pre-v5 model can predict. Strictest no-regression
  metric.
* **LOCO RMSE**: full held-out-class evaluation — all rows of one
  Material Source class held out as the test set; fit on the
  other 2 classes. Measures cross-class extrapolation.
* **PIT-KS**: Kolmogorov-Smirnov statistic of the Probability
  Integral Transform of held-out residuals through the Gaussian
  CDF — measures predictive interval calibration. 0 = perfect
  calibration.
* **cov95**: empirical coverage of the 95 % predictive interval —
  fraction of held-out residuals within ±1.96 σ_pred.

Per-cell results are 3-seed means; std reported when non-zero.

### 7.3 Architecture summary

The full V2 strength kernel is wrapped in a `TimeGatedKernel` (gate
at $t = 0$ enforces $f(x, 0) = 0$):

```
TimeGate( blind_matern(non_source + extras)
        + categorical_source_branch(source, non_source + extras)
        + additive_rbf_time(time) )
```

* **blind_matern**: class-independent ARD-Matern over composition +
  engineered features. Carries class-shared
  composition→strength structure. **Load-bearing** for
  `joint_hamming_matern` (§ 4.6); kept.
* **source branch**: the architectural choice — `joint_hamming_matern`
  in production.
* **additive_rbf_time**: RBF on the time dim only (concrete
  maturity).
* **TimeGate**: $h(t) = 1 - \exp(-t / \tau)$ with $\tau = 0.05$
  ensures $K(x, 0) = 0$, encoding the physical "zero strength at
  zero age" constraint.

The likelihood is `GatedGaussianLikelihood` (homoscedastic σ with
the same time gate). Per-class heteroscedastic σ_c was tested and
*failed* (§ 4.7).

Outcome transform: `maxscale_zeromean` (Y / Y_max). Input feature
augmentation: `F5_alllog` (5 engineered binder/aggregate
log-ratios).

### 7.4 Dataset summary

| dataset | strength rows | unique compositions | mortar | Set-2 | Set-3 |
|---|---|---|---|---|---|
| pre-v5 (deployed) | 647 | ~144 | yes (pooled MS=0) | yes (pooled MS=0) | partial (MS=1, contaminated) |
| **v5 (production)** | 647 | ~146 | 60 (MS=0) | ~30 (MS=1) | ~56 (MS=2) |

v5 fixes 12 corruption-collision splits, several mortar/concrete
misroutes, and drops 3 clay-using mortars (M75/76/77) the GP
feature set cannot represent. Strength values and Strength(Std)
for the 638 common (composition+temp+time) tuples are
**byte-identical**.

---

## 8. Supporting artefacts (auto-generated)

### 8.1 Per-experiment writeups

| file | what's in it |
|---|---|
| `RBF_EMBEDDING_VS_HAMMING_VS_INDEXKERNEL.md` | 30-cell in-distribution RBF-embedding grid |
| `RBF_EMBEDDING_LOCO.md` | 54-cell LOCO grid for RBF + baselines |
| `RBF_EMBEDDING_LENGTHSCALE_INIT_ABLATION.md` | 40-cell fixed-ℓ × init ablation |
| `RBF_EMBEDDING_LENGTHSCALE_INIT_SEEDS.md` | Multi-seed verification of new RBF variants |
| `JOINT_HAMMING_MATERN_ABLATION.md` | 48-cell focused joint-Hamming Matern grid |
| `JOINT_DISTANCE_FAMILY_ABLATION.md` | 132-cell comprehensive joint-distance family |
| `HYBRID_AND_TWIN_DROP_ABLATION.md` | 36-cell additive hybrid + M61 drop |
| `NO_BLIND_ABLATION.md` | 36-cell `include_blind=False` ablation |
| `PARETO_ANALYSIS.md` | Pareto-frontier extraction across all variants |
| `NOISE_AUDIT.md` | GP fitted σ_gp vs measurement σ_meas audit |
| `PER_CLASS_NOISE_ABLATION.md` | 24-cell `PerClassGatedGaussianLikelihood` (negative) |
| `M80_M81_DROP_ABLATION.md`, `M80_M81_TWIN_DIAGNOSIS*.md`, `M81_DROP_FULL_METRICS.md` | Twin-drop diagnostic history |
| `V5_VS_PRE_V5_INVESTIGATION.md` | H1-H10 hypotheses on the pre-v5 → v5 bLOO gap |

### 8.2 Kernel implementations

* `boxcrete/kernels.py` — `JointHammingMaternKernel`,
  `JointEmbeddingMaternKernel`, `RBFEmbeddingKernel`,
  `_categorical_source_branch` switch
* `boxcrete/likelihoods.py` — `PerClassGatedGaussianLikelihood`,
  `GatedGaussianLikelihood`
* `test/test_joint_hamming_matern_kernel.py` (14 tests)
* `test/test_joint_embedding_matern_kernel.py` (11 tests)
* `test/test_rbf_embedding_kernel.py` (15 tests)

### 8.3 CSVs (per-cell raw metrics)

`experiments/joint_distance_family_ablation.csv` is the canonical
source for the Pareto / per-kernel comparison tables in this
writeup. The other CSVs document supporting ablations.

### 8.4 Historical context

The recommendation evolved through several intermediate states as
the kernel space was explored:

1. **First recommendation: `hamming`** (the v5 production
   default at the start of this work). Superseded by `rbf_embedding_d2`
   when the RBF embedding kernel was introduced.
2. **Second recommendation: `rbf_embedding_d2`** — wins
   `joint_hamming_matern` on blind Class-2 LOCO (849 vs 1113)
   but loses on in-distribution bLOO, LOO, and PIT-KS.
   Superseded by `joint_hamming_matern` once the joint-distance
   family was tested.
3. **Final recommendation: `joint_hamming_matern`** — combines
   the joint kernel topology of `legacy_continuous_ard` with
   correct categorical handling. Strictly dominates `hamming` on
   in-distribution metrics; trades blind Class-2 LOCO for
   parsimony and overall balance.

Intermediate writeup states are preserved in git history (search
for "RBF d2" or "rbf_embedding_d2" in commit messages prior to
commit `41d9647ffa23` for the prior-recommendation framing). The
git history is the authoritative record of how the recommendation
was developed.
