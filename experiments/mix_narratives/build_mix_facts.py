#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Build the per-mix fact base used to author docs/model/mix_analyses.json.

Joins docs/model/compositions.json (the explorer catalog, index-keyed)
back to data/boxcrete_data.csv so every catalog entry carries its
canonical mix name, plus derived mix-design metrics and the measured
strength trajectory. Writes experiments/mix_narratives/mix_facts.json.

Run from the repo root:
    python experiments/mix_narratives/build_mix_facts.py
"""

import csv
import json
from collections import defaultdict

COMP = "docs/model/compositions.json"
DATA = "data/boxcrete_data.csv"
OUT = "experiments/mix_narratives/mix_facts.json"

CLASS_NAME = {
    0: "Set 1 mortar (Amrize 1L cement, Class C fly ash, no coarse aggregate)",
    1: "Set 2 concrete (Heidelberg 1L cement, Class C fly ash, limestone)",
    2: "Set 3 concrete (Amrize 1L cement, Class F fly ash, gravel)",
}
SOURCE_LETTER = {0: "A", 1: "B", 2: "C"}


def main():
    co = json.load(open(COMP))
    comps = co["compositions"]
    s1 = co["strength_predictions"]["1"]
    s28 = co["strength_predictions"]["28"]
    gwp = co["gwp_predictions"]
    cost = co["cost_predictions"]
    pareto = co["pareto_mask"]
    obs = co["observations"]

    rows = list(csv.DictReader(open(DATA)))
    # Group CSV rows by mix name; each mix has one composition, many times.
    by_mix = defaultdict(list)
    for r in rows:
        by_mix[r["Mix Name"]].append(r)

    def comp_of(mix_rows):
        r = mix_rows[0]
        return [
            float(r["Cement (kg/m3)"]),
            float(r["Fly Ash (kg/m3)"]),
            float(r["Slag (kg/m3)"]),
            float(r["Water (kg/m3)"]),
            float(r["HRWR (kg/m3)"]),
            float(r["Fine Aggregate (kg/m3)"]),
            float(r["Coarse Aggregates (kg/m3)"]),
            float(r["Material Source"]),
            float(r["Temp (C)"]),
        ]

    mix_comps = {m: comp_of(rs) for m, rs in by_mix.items()}

    # Match each catalog index to a mix name by nearest composition vector.
    unused = set(mix_comps)
    index_to_mix = {}
    for i, c in enumerate(comps):
        best, bestd = None, None
        for m in unused:
            mc = mix_comps[m]
            d = sum(abs(a - b) for a, b in zip(c, mc))
            if bestd is None or d < bestd:
                best, bestd = m, d
        index_to_mix[i] = (best, bestd)
        unused.discard(best)

    worst = max(d for _, d in index_to_mix.values())
    print(f"matched {len(index_to_mix)} indices; worst L1 mismatch = {worst:.6f}")
    print(f"unmatched mixes remaining: {sorted(unused)}")

    facts = {}
    for i, c in enumerate(comps):
        mix, dist = index_to_mix[i]
        cement, fa, slag, water, hrwr, fine, coarse, src, temp = c
        src = int(src)
        binder = cement + fa + slag
        scm = fa + slag
        agg = fine + coarse
        meas = sorted(([int(t), float(v)] for t, v in obs[str(i)]), key=lambda p: p[0])
        facts[str(i)] = {
            "mix": mix,
            "match_l1": round(dist, 6),
            "material_source": src,
            "source_letter": SOURCE_LETTER[src],
            "class_desc": CLASS_NAME[src],
            "cement": cement,
            "fly_ash": fa,
            "slag": slag,
            "water": water,
            "hrwr": hrwr,
            "fine": fine,
            "coarse": coarse,
            "temp": temp,
            "binder": round(binder, 1),
            "scm": round(scm, 1),
            "scm_frac": round(scm / binder, 4) if binder else None,
            "slag_frac_of_scm": round(slag / scm, 4) if scm else None,
            "wb": round(water / binder, 4) if binder else None,
            "wc": round(water / cement, 4) if cement else None,
            "hrwr_pct_binder": round(100 * hrwr / binder, 3) if binder else None,
            "agg_total": round(agg, 1),
            "coarse_frac": round(coarse / agg, 4) if agg else 0.0,
            "paste_frac_mass": round(binder + water, 1),
            "observations": meas,
            "obs_days": [p[0] for p in meas],
            "s28_meas": next((v for t, v in meas if t == 28), None),
            "s1_meas": next((v for t, v in meas if t == 1), None),
            "s7_meas": next((v for t, v in meas if t == 7), None),
            "pred_1d": round(s1[i], 1),
            "pred_28d": round(s28[i], 1),
            "gwp": round(-gwp[i], 2),
            "cost": round(-cost[i], 2),
            "on_pareto": bool(pareto[i]),
        }

    # Class-relative percentiles for the headline quantities.
    for src in (0, 1, 2):
        peers = [k for k, f in facts.items() if f["material_source"] == src]
        for key in ("wb", "scm_frac", "binder", "gwp", "pred_28d", "cost"):
            vals = sorted(facts[k][key] for k in peers if facts[k][key] is not None)
            for k in peers:
                v = facts[k][key]
                if v is None:
                    continue
                rank = sum(1 for x in vals if x < v)
                facts[k][f"pct_{key}"] = round(100 * rank / max(1, len(vals) - 1), 1)
        facts_src = [facts[k] for k in peers]
        for k in peers:
            facts[k]["class_n"] = len(peers)
            facts[k]["class_median_28d"] = round(
                sorted(f["pred_28d"] for f in facts_src)[len(facts_src) // 2], 1
            )
            facts[k]["class_median_gwp"] = round(
                sorted(f["gwp"] for f in facts_src)[len(facts_src) // 2], 1
            )

    json.dump(facts, open(OUT, "w"), indent=1)
    print(f"wrote {OUT} with {len(facts)} entries")

    n_pareto = sum(1 for f in facts.values() if f["on_pareto"])
    print(f"pareto-optimal: {n_pareto}")
    for src in (0, 1, 2):
        peers = [f for f in facts.values() if f["material_source"] == src]
        names = sorted(f["mix"] for f in peers)
        print(f"  class {src}: n={len(peers)}  {names[0]}..{names[-1]}")


if __name__ == "__main__":
    main()
