"""Generate LLM-quality mix analysis descriptions for each observed composition.

⚠️  WARNING — DO NOT RUN THIS CASUALLY  ⚠️

The committed ``docs/model/mix_analyses.json`` was authored by Claude
Opus with full domain context (specific psi values, dataset
comparisons, qualitative assessments like "among the top performers in
the Source A mortar series"). This script's output is a deterministic
template-based fallback that produces *objectively less detailed*
descriptions ("A blended binder system with 47% cement replacement by
SCMs.") and does NOT reproduce the LLM-authored content.

Running this script will OVERWRITE ``docs/model/mix_analyses.json``
with the inferior templated descriptions. Don't.

When to actually run it:
  * When ``compositions.json`` adds new mixes and you need *some*
    description for them — run the script, then hand-edit / re-LLM
    the new entries up to the existing quality bar before committing.
  * Never invoke it from CI or from
    ``experiments/regenerate_all_artifacts.sh`` (which deliberately
    does not call this script).

To regenerate from scratch with LLM quality, hand-prompt an LLM with
the composition CSV plus the current ``mix_analyses.json`` as a style
exemplar; do NOT just run this script.
"""

import json
import os
import sys

# Load compositions data
with open(os.path.join(os.path.dirname(__file__), "model/compositions.json")) as f:
    data = json.load(f)

cols = data["column_names"]
comps = data["compositions"]
obs = data["observations"]
gwp_preds = data["gwp_predictions"]
cost_preds = data["cost_predictions"]
strength_preds = data["strength_predictions"]
if "pareto_mask" not in data:
    print(
        "[generate_mix_analyses] WARNING: input compositions.json has no "
        "'pareto_mask' field; no mix will be labelled Pareto-optimal.",
        file=sys.stderr,
    )
pareto_mask = data.get("pareto_mask", [0] * len(comps))
# Strength predictions must include the day keys we read below.
# Fail loudly here rather than producing nonsense "inf\u00d7 the 1-day" prose.
for required_day in ("1", "28"):
    if required_day not in strength_preds:
        raise KeyError(
            f"compositions.json::strength_predictions is missing the "
            f"required key {required_day!r}; got {list(strength_preds)}."
        )


def get_col(comp, name):
    idx = cols.index(name)
    return comp[idx]


def generate_description(idx, comp):
    cement = get_col(comp, "Cement (kg/m3)")
    fly_ash = get_col(comp, "Fly Ash (kg/m3)")
    slag = get_col(comp, "Slag (kg/m3)")
    water = get_col(comp, "Water (kg/m3)")
    hrwr = get_col(comp, "HRWR (kg/m3)")
    fine_agg = get_col(comp, "Fine Aggregate (kg/m3)")
    coarse_agg = get_col(comp, "Coarse Aggregates (kg/m3)")
    mat_source = get_col(comp, "Material Source")
    temp = get_col(comp, "Temp (C)")

    binder = cement + fly_ash + slag
    has_binder = binder > 0
    wb = water / binder if has_binder else None
    scm_pct = ((fly_ash + slag) / binder * 100) if has_binder else 0
    cement_pct = (cement / binder * 100) if has_binder else 0
    gwp = abs(gwp_preds[idx])
    str_28 = strength_preds["28"][idx]
    is_pareto = pareto_mask[idx] > 0.5

    # Observations
    obs_data = obs.get(str(idx))
    obs_str = ""
    if obs_data:
        day_vals = {d: s for d, s in obs_data}
        if 28 in day_vals and 1 in day_vals and day_vals[1] > 0:
            ratio = day_vals[28] / day_vals[1]
            if ratio > 5:
                obs_str = (
                    f" Observed data shows remarkable late-age development"
                    f" \u2014 28-day strength is {ratio:.1f}\u00d7 the"
                    f" 1-day value, indicating significant pozzolanic"
                    f" reaction over time."
                )
            elif ratio < 1.5 and day_vals[1] > 3000:
                obs_str = (
                    f" This mix achieves most of its strength early"
                    f" \u2014 the 28/1-day ratio is only {ratio:.1f}\u00d7,"
                    f" suggesting rapid hydration dominates."
                )
            else:
                obs_str = (
                    f" Strength develops steadily from "
                    f"{day_vals[1]:,.0f} psi at day 1 to "
                    f"{day_vals[28]:,.0f} psi at day 28."
                )

    parts = []

    # Title/summary
    if is_pareto:
        parts.append(
            "**Pareto-optimal mix** \u2014 this formulation achieves an "
            "exceptional balance of strength and sustainability that no "
            "other tested mix dominates."
        )

    # Binder system description
    if scm_pct > 80:
        parts.append(
            f"An ultra-high SCM replacement mix ({scm_pct:.0f}% "
            f"supplementary cementitious materials), with only "
            f"{cement:.0f} kg/m\u00b3 of Portland cement."
        )
    elif scm_pct > 60:
        parts.append(
            f"A high-replacement mix using {scm_pct:.0f}% SCMs (fly ash "
            f"+ slag), significantly reducing clinker demand."
        )
    elif scm_pct > 30:
        parts.append(
            f"A blended binder system with {scm_pct:.0f}% cement "
            f"replacement by SCMs."
        )
    elif cement > 600:
        parts.append(
            f"A high-cement mix ({cement:.0f} kg/m\u00b3) with minimal "
            f"SCM replacement \u2014 prioritizing early strength over "
            f"sustainability."
        )
    else:
        parts.append(
            f"A Portland cement\u2013dominant mix ({cement_pct:.0f}% OPC) "
            f"with {binder:.0f} kg/m\u00b3 total binder."
        )

    # SCM specifics
    if slag > 200 and fly_ash < 50:
        parts.append(
            f"Slag dominates the SCM fraction at {slag:.0f} kg/m\u00b3, "
            f"which contributes to denser pore structure and superior "
            f"long-term strength development through latent hydraulic "
            f"reaction."
        )
    elif fly_ash > 200 and slag < 50:
        parts.append(
            f"High fly ash content ({fly_ash:.0f} kg/m\u00b3) provides "
            f"pozzolanic reactivity \u2014 consuming Ca(OH)\u2082 to form "
            f"additional C-S-H gel. This delays early strength but "
            f"improves workability and long-term durability."
        )
    elif fly_ash > 100 and slag > 100:
        parts.append(
            f"A ternary blend ({cement:.0f}/{fly_ash:.0f}/{slag:.0f} "
            f"cement/fly ash/slag) leveraging synergies: slag provides "
            f"hydraulic strength while fly ash improves particle "
            f"packing and workability."
        )

    # W/B ratio
    if not has_binder:
        parts.append(
            "Mix has zero binder (cement + fly ash + slag) \u2014 W/B ratio "
            "undefined; this is a degenerate composition that the "
            "strength model would not be expected to handle meaningfully."
        )
    elif wb < 0.25:
        parts.append(
            f"Ultra-low W/B ratio ({wb:.3f}) enabled by {hrwr:.1f} "
            f"kg/m\u00b3 of superplasticizer (HRWR). This produces an "
            f"extremely dense matrix with minimal capillary porosity."
        )
    elif wb < 0.35:
        parts.append(
            f"Low W/B ratio ({wb:.3f})"
            f"{' with HRWR for workability' if hrwr > 1 else ''} \u2014 "
            f"targeting high strength through reduced porosity."
        )
    elif wb > 0.5:
        parts.append(
            f"Relatively high W/B ratio ({wb:.3f}) \u2014 this increases "
            f"workability but introduces more capillary pores, limiting "
            f"ultimate strength."
        )
    else:
        parts.append(
            f"Moderate W/B ratio ({wb:.3f}) balancing workability and strength."
        )

    # Temperature
    if temp < 0:
        parts.append(
            f"Cured at {temp:.0f}\u00b0C \u2014 cold curing dramatically "
            f"slows cement hydration and pozzolanic reactions. Early-age "
            f"strength is severely compromised, though long-term "
            f"strength may partially recover if curing conditions improve."
        )
    elif temp == 10:
        parts.append(
            f"Cured at {temp:.0f}\u00b0C \u2014 below the standard "
            f"22\u00b0C reference, which slows hydration kinetics and "
            f"delays strength development, particularly for SCM-rich "
            f"formulations."
        )

    # Aggregate
    if coarse_agg > 0 and fine_agg > 0:
        ca_ratio = coarse_agg / (fine_agg + coarse_agg) * 100
        descriptor = (
            "coarse-dominant for structural applications"
            if ca_ratio > 55
            else "balanced gradation for workability"
        )
        parts.append(
            f"Aggregate blend: {ca_ratio:.0f}% coarse / "
            f"{100 - ca_ratio:.0f}% fine \u2014 {descriptor}."
        )
    elif coarse_agg == 0:
        parts.append(
            "Fine-aggregate-only mix (no coarse aggregate) \u2014 a "
            "mortar-type formulation, common in laboratory screening "
            "studies."
        )

    # Material source
    if mat_source == 1:
        parts.append(
            "Uses **Material Source B** \u2014 a different raw material "
            "supplier, which may affect reactivity and particle size "
            "distribution."
        )

    # Performance
    perf_parts = []
    if str_28 > 10000:
        perf_parts.append(f"exceptional 28-day strength ({str_28:,.0f} psi)")
    elif str_28 > 7000:
        perf_parts.append(f"strong 28-day performance ({str_28:,.0f} psi)")
    elif str_28 < 4000:
        perf_parts.append(f"modest 28-day strength ({str_28:,.0f} psi)")

    if gwp < 120:
        perf_parts.append(f"very low GWP ({gwp:.0f} kg CO₂e/m³)")
    elif gwp > 400:
        perf_parts.append(f"high GWP ({gwp:.0f} kg CO₂e/m³)")

    if perf_parts:
        parts.append("Predicted performance: " + ", ".join(perf_parts) + ".")

    if obs_str:
        parts.append(obs_str.strip())

    return " ".join(parts)


# Generate all descriptions
analyses = {}
for i in range(len(comps)):
    if str(i) in obs:  # Only generate for compositions with observations
        analyses[str(i)] = generate_description(i, comps[i])

# Write output
out_path = os.path.join(os.path.dirname(__file__), "model/mix_analyses.json")
with open(out_path, "w") as f:
    json.dump(analyses, f, indent=2)

print(f"Generated {len(analyses)} mix descriptions → {out_path}")
