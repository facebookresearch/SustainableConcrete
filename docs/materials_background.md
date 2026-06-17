# Materials background — BOxCrete dataset

This document captures the three distinct material sets that make up the
BOxCrete strength dataset, as shared by our collaborator. The class labels
`{0, 1, 2}` referenced throughout the codebase (in the `Material Source`
column of `data/boxcrete_data.csv`, in `boxcrete.mix_naming`, and in the
strength-GP kernel's source-aware branch) correspond to the three sets
below.

The defining physical-realism rule is:

> **Mortar mixes have zero coarse aggregate. Concrete mixes have nonzero
> coarse aggregate.** This invariant is enforced as a unit test
> (`test/test_mix_naming.py::test_coarse_aggregate_invariant_against_data_csv`)
> and is also the simplest safety check at data-load time.

---

## Class 0 — Set 1 — Mortar

**Naming**: `M<int>` (e.g., `M1`, `M2`, …, `M65`).
**Definition**: zero coarse aggregate (`Coarse Aggregates (kg/m3) = 0`).

| Component | Source |
|---|---|
| Cement | 1L — Amrize — Ste. Genevieve Plant, Bloomsdale, Missouri |
| Fly Ash | Class C — Ozinga — Elm Road Station, Oak Creek, Wisconsin |
| Slag | Grade 100 — Ozinga — Anshan City, Liaoning, China |
| Fine Aggregate | Masonary Sand — Prairie Material — Sand Valley, Danville, Illinois |
| Coarse Aggregate | (none) |
| High Range Water Reducer | Chryso®Adva Cast 530 |

Notes:
- The original collaborator file (`data/BOxCrete_data.csv`, commit
  `f1aa7e4`) named this set `M1`–`M69` (69 mortars). The v5 canonical
  table (`boxcrete/_mix_naming_table.csv`) has 65 mortars after
  dropping 15 strength-less new-file mortars (`M60..M74` in the
  collaborator's updated master, all-NaN strength rows useless for
  GP fitting); the surviving 65 are renumbered consecutively as
  `M1..M65` in the v5 design (no overflow prefixes).

---

## Class 1 — Set 2 — Concrete (Heidelberg cement, Class C fly ash)

**Naming**: `C<int>` for concretes from Sets 2 and 3 collectively, with
Set 2 occupying the lower range (`C1`, …, `C30`). The class is encoded in
the `Material Source` column (`= 1`). Set 2 contains 30 unique mixes.
**Definition**: nonzero coarse aggregate; cementitious chemistry as below.

| Component | Source |
|---|---|
| Cement | 1L — Heidelberg Materials — Mitchell Plant, Mitchell, Indiana |
| Fly Ash | Class C — Eco Material Technologies — Labadie Station, Labadie, Missouri |
| Slag | Grade 100 — Ozinga — Anshan City, Liaoning, China |
| Fine Aggregate | Concrete Sand — Prairie Material — Sand Valley, Danville, Illinois |
| Coarse Aggregate | Limestone — Prairie Material — Vulcan Materials Kankakee Facility, Kankakee, Illinois |
| High Range Water Reducer | Chryso®Adva Cast 593 |

Notes:
- Original collaborator names: `C1`–`C27`. Now extended to `C1`–`C30` after
  follow-up additions (3 new Set 2 mixes). The v5 canonical numbering
  is consecutive (`C1..C30`); the rejected overflow-prefix design
  (`C2_28`, `C2_29`, `C2_30`) is removed.

---

## Class 2 — Set 3 — Concrete (Amrize cement, Class F fly ash)

**Naming**: `C<int>` continuation past Set 2's range (`C31`, …, `C84`).
Class encoded in `Material Source = 2`. Set 3 contains 54 unique mixes.
**Definition**: nonzero coarse aggregate; cementitious chemistry as below.

| Component | Source |
|---|---|
| Cement | 1L — Amrize — Ste. Genevieve Plant, Bloomsdale, Missouri |
| Fly Ash | **Class F** — Eco Material Technologies — Coal Creek Station, McLean County, North Dakota |
| Slag | Grade 100 — Amrize — South Chicago Plant, Chicago, Illinois |
| Fine Aggregate | Concrete Sand — Amrize — Elk River Plant, Elk River, Minnesota |
| Coarse Aggregate (1) | #6 Gravel — Amrize — Empire Plant, Farmington, Minnesota |
| Coarse Aggregate (2) | #89 Gravel — Amrize — Empire Plant, Farmington, Minnesota |
| High Range Water Reducer | Sika® ViscoCrete® 1000 |

Notes:
- Original collaborator names: `C28`–`C54`. After splitting the 12
  corruption-collision mixes (`Mix_126..Mix_137` in the legacy
  intermediate naming, each containing two physically distinct
  compositions filed under one name) and adding follow-up Set-3
  entries from the new collaborator file, the v5 canonical numbering
  is consecutive `C31..C84` (54 concretes total). The rejected
  overflow-prefix design (`C3_55`, `C3_56`, ...) is removed.
- Set 3 differs from Set 2 in *every* materials component except slag
  grade: the cement plant is different (Amrize vs Heidelberg), the fly
  ash class is different (F vs C), the slag plant is different, the fine
  aggregate origin is different, the coarse aggregate is gravel rather
  than crushed limestone (and is binary-blended), and the HRWR is from a
  different chemistry vendor (Sika vs Chryso). Treating Sets 2 and 3 as
  pooled "concrete" data — as the model did before this branch — was a
  meaningful misspecification of the response surface.

---

## Why the model needs three classes (not two)

Prior to the `three-class-source` branch, `data/boxcrete_data.csv` carried
a single `Material Source ∈ {0, 1}` column with the semantic
`0 = Mortar ∪ Set 2 Concrete`, `1 = Set 3 Concrete`. This was a regression
introduced by the data-cleanup commit (`b4f34b9`) which dropped the
`Mortar or Concrete` column on the assumption that `Material Source` alone
carried the categorical signal. It did not — it pooled (Mortar, MS=0) and
(Concrete-Set-2, MS=0) under a single `MS=0` bucket and erased the Set 1
↔ Set 2 distinction.

The `three-class-source` branch restores the lost distinction by:

1. Splitting `Material Source` to three integer levels `{0, 1, 2}`
   matching the sets above.
2. Renaming mixes to a `M<int>` / `C<int>` convention where the prefix
   makes mortar-vs-concrete unambiguous from the name alone.
3. Adding a categorical (Hamming/Index-style) kernel branch in
   `boxcrete.kernels` so the strength GP treats the source dimension as a
   categorical with three levels rather than a continuous real with two
   levels.

See `boxcrete/mix_naming.py` for the canonical name → class API and
`/Users/sebastianament/.llms/plans/three_class_material_source.plan.md`
for the broader implementation plan.
