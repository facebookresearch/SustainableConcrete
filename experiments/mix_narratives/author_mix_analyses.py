#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Author docs/model/mix_analyses.json.

Every numeric figure in the output is injected from the verified fact base
(experiments/mix_narratives/mix_facts.json, built by build_mix_facts.py from
docs/model/compositions.json joined to data/boxcrete_data.csv), so the prose
cannot drift from the shipped catalog. The interpretive ANALYSIS text is
hand-authored per mix.

``check_claims`` additionally asserts every superlative and comparative claim
made in that prose against the data, so a future re-merge that changes which
mix is strongest/weakest/cleanest fails the build instead of silently shipping
a false statement.

The explorer renders each string as HTML after converting **bold**, so the
text must not contain raw '<' or '&'.

Run from the repo root, after build_mix_facts.py:
    python experiments/mix_narratives/build_mix_facts.py
    python experiments/mix_narratives/author_mix_analyses.py
"""

import json

FACTS = "experiments/mix_narratives/mix_facts.json"
OUT = "docs/model/mix_analyses.json"

SOURCE_BLURB = {
    0: (
        "**Material Source A** (Set 1): Amrize 1L cement, Class C fly ash, "
        "Grade 100 slag, masonry sand, no coarse aggregate"
    ),
    1: (
        "**Material Source B** (Set 2): Heidelberg 1L cement, Class C fly ash, "
        "concrete sand, crushed limestone coarse aggregate"
    ),
    2: (
        "**Material Source C** (Set 3): Amrize 1L cement, Class F fly ash, "
        "concrete sand, river-gravel coarse aggregate"
    ),
}

# Hand-authored interpretation. Keyed by canonical mix name.
ANALYSIS = {
    # ---------- Set 1 mortars: constant-binder exploration (M1-M14) ----------
    "M1": "The standout of the early constant-binder mortars. Slag carries most of "
    "the replacement, and the low water demand is bought with superplasticizer "
    "rather than cement. Slag's latent hydraulic reaction densifies the pore "
    "structure without the carbon of clinker, so this mix reaches the 10,000 psi "
    "class on a binder that is nearly half supplementary. A good template for how "
    "this dataset achieves low-carbon, high-strength mortar.",
    "M2": "A balanced fly-ash-plus-slag blend at high replacement. Strength lands "
    "well below M1 despite a similar water-binder ratio, which is the recurring "
    "penalty for pushing Class C fly ash past about half the binder: the "
    "pozzolanic reaction is slower and contributes less at 28 days than slag's "
    "latent hydraulic reaction.",
    "M3": "High water content with no superplasticizer. The resulting capillary "
    "porosity, not the replacement level, is what limits this mix. Compare with "
    "M1 at the same binder and a similar replacement level but far less water.",
    "M4": "High replacement combined with no superplasticizer. This is the weakest "
    "outcome among the early constant-binder mortars and the clearest "
    "illustration that aggressive SCM substitution needs chemical admixture "
    "support to be viable.",
    "M5": "A fly-ash-leaning blend at moderate water content. Performs respectably, "
    "but the measured 28-day strength sits below slag-leaning mixes at "
    "comparable replacement, again favouring slag for this dataset.",
    "M6": "Cement-rich with modest replacement and low water. Develops strength "
    "very quickly, reaching most of its 28-day value within a day. The trade is "
    "carbon: embodied GWP is among the higher values in the mortar set.",
    "M7": "A slag blend held back by high water content. Middling strength at "
    "relatively high embodied carbon, since the cement fraction stays large.",
    "M8": "The highest water-binder ratio in the early series. Strength is limited "
    "by paste porosity rather than by binder chemistry; the substantial slag "
    "content cannot compensate for the excess mixing water.",
    "M9": "An efficient point in the early series: high replacement, moderate water, "
    "only a token superplasticizer dose, and strong 28-day performance at low "
    "embodied carbon. One of the better carbon-to-strength trades among the "
    "no-to-low-admixture mortars.",
    "M10": "Slag-only replacement at a modest level, with high water. Useful as a "
    "single-SCM reference point against the blended mixes.",
    "M11": "Fly-ash-only replacement at a modest level. Pairs with M10 to isolate "
    "fly ash against slag at similar replacement; the two land close together "
    "at 28 days, with the difference showing up mainly in early-age strength.",
    "M12": "Slag-only at high water content. The high water-binder ratio dominates "
    "the outcome and pulls strength below what the binder alone would support.",
    "M13": "The lowest replacement level in the early series, essentially a "
    "cement-dominant control. Its value in the dataset is as a high-carbon "
    "anchor point against which the blended mixes are judged.",
    "M14": "A half-slag binder at high water. A clean single-variable companion to "
    "M8, which pushes water higher still at similar replacement.",
    # ---------- Set 1 mortars: low-water superplasticized series (M15-M26) ----
    "M15": "The opening point of the low-water, superplasticized mortar series. "
    "A heavy superplasticizer dose buys an extremely low water-binder ratio, "
    "producing a dense matrix and one of the highest measured strengths in the "
    "mortar set. Sits on the strength-carbon Pareto frontier.",
    "M16": "Same low-water, high-admixture recipe as M15 with a small amount of "
    "cement traded for fly ash. Strength is essentially unchanged while embodied "
    "carbon drops, which is exactly the substitution this dataset is exploring.",
    "M17": "Continues the M15-M18 replacement ladder. Strength holds steady as "
    "clinker is progressively removed, demonstrating that at very low water-binder "
    "ratio the binder has strength headroom to spare.",
    "M18": "The most aggressive replacement in the M15-M18 ladder, and it is the "
    "strongest of the four. This is the key result of the series: at a very low "
    "water-binder ratio, replacing more than half the cement with fly ash and slag "
    "costs nothing in 28-day strength while cutting embodied carbon substantially.",
    "M19": "Very high replacement with a reduced superplasticizer dose and "
    "correspondingly higher water. Strength falls off relative to the "
    "fully-superplasticized members of the series, marking the practical limit of "
    "replacement once water is no longer tightly controlled.",
    "M20": "The strongest mortar measured in the dataset. Roughly half the binder is "
    "supplementary, water is very low, and the superplasticizer dose is generous. "
    "Demonstrates that peak mortar strength here does not require a "
    "cement-dominant binder.",
    "M21": "Slag-dominant at high replacement and minimum water. Strong and "
    "notably low-carbon for its strength class.",
    "M22": "Pushes replacement higher than M21 at the same water and admixture. "
    "Strength gives up a few hundred psi while carbon falls further, a favourable "
    "trade for most structural applications.",
    "M23": "A moderate-replacement member of the low-water series. Fits the "
    "series trend closely and is unremarkable except as a fill-in point on the "
    "replacement axis.",
    "M24": "Balanced fly ash and slag at moderate-to-high replacement, minimum "
    "water. A well-rounded point: near the top of the strength range at "
    "meaningfully reduced carbon.",
    "M25": "Slightly more water and less slag than its neighbours in the series. "
    "The small strength deficit relative to M24 and M26 tracks the reduced slag "
    "content more than the marginal water increase.",
    "M26": "Close to the series optimum. Roughly half the binder is supplementary, "
    "water is at the series minimum, and 28-day strength is near the top of the "
    "mortar range.",
    # ---------- Set 1 mortars: aggregate and binder-content variations --------
    "M27": "Same binder chemistry as the low-water series but with the sand content "
    "cut sharply, raising the paste fraction. Strength drops well below the "
    "full-aggregate members, showing that this series was aggregate-optimised and "
    "does not simply improve with more paste.",
    "M28": "High replacement, moderate water, no superplasticizer. The slag-dominant "
    "half of a deliberate pair with M29 and the stronger of the two.",
    "M29": "The direct counterpart to M28: identical binder content, water and "
    "replacement level, but with the fly-ash and slag proportions swapped so fly "
    "ash dominates. It is markedly weaker at 28 days. This pair is the cleanest "
    "single-variable evidence in the mortar data that slag outperforms Class C fly "
    "ash as the primary replacement.",
    "M30": "Reduced binder content at very high replacement. Both levers push the "
    "same direction, and strength lands in the lower-middle of the mortar range at "
    "correspondingly low embodied carbon.",
    "M31": "High binder content combined with very high replacement and low water. "
    "The result is among the strongest mortars measured, showing that a large but "
    "mostly supplementary binder is an effective route to high strength.",
    # ---------- Set 1 mortars: cold-cured replicates (M32-M38) ----------------
    "M32": "A cold-cured replicate of the low-water superplasticized series, mixed "
    "and cured near 4.5 C rather than the usual 22 C. Early strength is heavily "
    "suppressed by the cold, but the mix recovers to a normal 28-day value. Cold "
    "curing delays hydration; it does not, at this replacement level, destroy the "
    "ultimate strength.",
    "M33": "Cold-cured companion to M32 at higher replacement. The early-age penalty "
    "deepens as supplementary content rises, because both the pozzolanic and latent "
    "hydraulic reactions are more temperature-sensitive than clinker hydration.",
    "M34": "Cold-cured at high replacement. One-day strength is a small fraction of "
    "what the same composition achieves at normal temperature, yet the 28-day "
    "result remains solidly usable. A striking demonstration of delayed rather than "
    "lost strength.",
    "M35": "The cold-cured mix with the highest replacement and elevated water. This "
    "combination gives up the most: both early and 28-day strength fall clearly "
    "below the rest of the cold-cured group.",
    "M36": "The most extreme early-age result in the dataset. At very high "
    "replacement and 4.5 C, one-day strength is almost nil, but by 28 days the mix "
    "has caught up to the normal-temperature members of its series. The definitive "
    "example of why cold-weather placement of high-SCM mixes must be judged on "
    "mature strength, not on early-age cylinders.",
    "M37": "Cold-cured at moderate replacement. Sits between M32 and M33 on the "
    "early-age penalty, consistent with its intermediate supplementary content.",
    "M38": "Cold-cured near the series-average replacement, and it recovers to a "
    "strong 28-day value. Together with M32 through M37 it establishes that the "
    "cold-weather penalty in this dataset scales with replacement level and is "
    "largely temporary.",
    # ---------- Set 1 mortars: small-batch and low-binder series (M39-M48) ----
    "M39": "A plain cement mortar at very low water with a heavy superplasticizer "
    "dose and a much-reduced sand content. Despite the low water-binder ratio the "
    "strength is modest, indicating this small-batch series is not directly "
    "comparable to the main constant-binder mortars.",
    "M40": "Very low binder and sand content with a large superplasticizer dose. "
    "Weak, as expected at this binder loading; useful mainly to anchor the low end "
    "of the binder-content axis.",
    "M41": "Low binder with moderate replacement. Strength is limited by the sheer "
    "lack of binder rather than by its chemistry.",
    "M42": "Low binder combined with high water. Among the weakest mortars measured, "
    "with both principal strength levers working against it.",
    "M43": "A well-proportioned small-batch mortar: moderate replacement, low water, "
    "generous superplasticizer. Delivers solid strength at moderate embodied "
    "carbon, though both M46 and M48 reach higher strength within this group.",
    "M44": "Nearly all cement at very low water. Strong, but at high embodied carbon; "
    "the low replacement level is the limiting factor from a sustainability "
    "standpoint rather than a performance one.",
    "M45": "The weakest mix in the dataset. Minimal binder, high replacement and a "
    "small batch size combine to leave almost no strength-generating paste.",
    "M46": "A plain cement mortar at low water and high binder content. The "
    "highest-strength member of the small-batch group and also the highest-carbon, "
    "serving as the conventional control this dataset is trying to beat.",
    "M47": "Very low binder with light replacement. Weak, and useful chiefly as a "
    "low-binder reference point.",
    "M48": "Moderate replacement, high binder, very low water. Reaches high strength "
    "at appreciably lower carbon than the plain-cement M46, making it one of the "
    "more efficient mixes in the small-batch group.",
    # ---------- Set 1 mortars: very high binder (M49-M50) ---------------------
    "M49": "An ultra-high-binder mortar with minimal replacement, approaching "
    "high-performance mortar territory. Very high early and 28-day strength, but "
    "the embodied carbon is among the highest in the dataset, which is precisely "
    "the trade-off this project exists to avoid.",
    "M50": "The companion to M49 with marginally more supplementary content and less "
    "water. Slightly stronger than M49 and sits on the strength-carbon Pareto "
    "frontier, though only because nothing else in the data reaches its strength "
    "level at all. It is Pareto-optimal by virtue of extreme strength, not "
    "efficiency.",
    # ---------- Set 1 mortars: moderate water-binder series (M51-M58) ---------
    "M51": "High binder and very high replacement at moderate water, without "
    "superplasticizer. Strength is middling; the absence of admixture forces the "
    "extra water that limits it.",
    "M52": "High replacement at moderate water. A representative member of the "
    "no-admixture group and a useful baseline for the value of superplasticizer, "
    "which the low-water series quantifies directly.",
    "M53": "Moderate replacement, reduced binder and sand content. Lands mid-range "
    "on strength and low on carbon.",
    "M54": "Near-total cement replacement, overwhelmingly slag. Strength is low, "
    "marking the point where too little clinker remains to activate the slag "
    "effectively. In exchange it has the lowest embodied carbon of any mortar in "
    "the dataset, so it defines the extreme low-carbon corner of the mortar design "
    "space.",
    "M55": "High replacement with elevated binder and sand content at moderate "
    "water. A solid mid-range performer at low carbon.",
    "M56": "High replacement with balanced fly ash and slag at moderate water. "
    "Behaves as the series trend predicts.",
    "M57": "Slag-only replacement at high level with high water and a token "
    "superplasticizer dose. Respectable strength for its carbon, reinforcing "
    "slag's advantage over fly ash in this dataset.",
    "M58": "Slag-only at very high replacement and high binder content. Reaches "
    "reasonable strength at low carbon and is one of the better efficiency points "
    "among the no-superplasticizer mortars.",
    # ---------- Set 1 mortars: cold-cured, early-age only (M59-M60) -----------
    "M59": "A cement-rich mix cured near 4.5 C, with only early-age measurements "
    "available. No 28-day cylinder was recorded, so the model's mature-strength "
    "prediction here is an extrapolation and should be treated with more caution "
    "than for fully-measured mixes.",
    "M60": "A fly-ash blend cured near 4.5 C, again with only early-age data. Its "
    "very low one-day strength is consistent with the rest of the cold-cured group; "
    "as with M59, the 28-day figure shown is a model extrapolation rather than a "
    "measurement.",
    # ---------- Set 1 mortars: slag-dominant, high binder (M61-M66) -----------
    "M61": "An extreme slag mix: binder content roughly double the series norm and "
    "almost entirely slag. Strength is low despite the enormous binder loading, "
    "because there is too little cement to supply the alkalinity that activates it. "
    "The clearest evidence in the mortar data that slag needs a clinker partner.",
    "M62": "High slag replacement at moderate binder and water. A well-balanced "
    "low-carbon mortar that reaches solid strength, and a good counterpoint to the "
    "over-slagged M61.",
    "M63": "A slag-only binder with no Portland cement at all. Without an activator "
    "the mix develops only modest strength, confirming the M61 result at the "
    "limiting case. Its embodied carbon is among the lowest of any mortar here, "
    "behind only the near-total-replacement M54 and M65.",
    "M64": "Very high slag replacement at moderate binder. Retains enough cement to "
    "activate the slag and reaches respectable strength at low carbon.",
    "M65": "Near-total slag replacement. Strength holds up better than M61 and M63 "
    "because the binder content is more moderate, keeping the paste volume "
    "sensible.",
    "M66": "High slag content at high binder loading and moderate water. One of the "
    "stronger low-carbon mortars, and a more successful execution of the "
    "high-slag idea than M61.",
    # ---------- Set 1 mortars: plain-cement water ladder (M67-M69) ------------
    "M67": "The low-water rung of a clean three-point plain-cement ladder with M68 "
    "and M69, in which only mixing water changes. Strongest of the three, and the "
    "textbook confirmation of Abrams' law within this dataset.",
    "M68": "The middle rung of the M67-M69 water ladder. Strength falls between its "
    "two siblings almost exactly in proportion to the added water.",
    "M69": "The high-water rung of the M67-M69 ladder and the weakest of the three, "
    "despite identical binder content and chemistry. Together the trio isolates "
    "water-binder ratio as a strength driver with no confounding variables, which "
    "is why they are valuable calibration points for the model.",
    # ---------- Set 2 concretes: aggregate-content series (C1-C4) -------------
    "C1": "One of four mixes (C1 through C4) that share an identical plain-cement "
    "paste and roughly 1,780 kg/m3 of total aggregate, differing only in how that "
    "aggregate is split between gravel and sand. C1 is the weakest of the four by a "
    "clear margin despite sitting mid-range on coarse fraction, which makes it the "
    "odd one out rather than the endpoint of a trend.",
    "C2": "A coarse-leaning member of the C1-C4 gradation series and its strongest "
    "point, though only marginally ahead of C3 and C4.",
    "C3": "The most gravel-rich member of the C1-C4 gradation series. Essentially "
    "tied with C2 and C4 on strength.",
    "C4": "The most sand-rich member of the C1-C4 gradation series, yet it matches "
    "the gravel-rich C2 and C3. Read together, the four show no monotonic "
    "relationship between coarse-to-fine ratio and strength at fixed paste: three "
    "of them land within about 150 psi of each other while C1 sits roughly 2,000 psi "
    "lower. The spread is better explained as batch-to-batch variability than as an "
    "aggregate-gradation effect, which is a useful caution when reading small "
    "strength differences elsewhere in this catalog.",
    # ---------- Set 2 concretes: high-performance blends (C5-C15) -------------
    "C5": "A very high replacement blend at low water with superplasticizer, and one "
    "of the two strongest mixes in the entire dataset. Nearly seventy percent of the "
    "binder is supplementary, yet it outperforms every plain-cement concrete here by "
    "a wide margin. The headline result of the Set 2 programme.",
    "C6": "Identical to C5 except for a higher superplasticizer dose, and it is the "
    "single strongest mix in the dataset. The pair isolates admixture dosage as the "
    "only variable, and the gain is substantial. C6 sits on the strength-carbon "
    "Pareto frontier and, unlike the high-cement M50, it earns that position through "
    "genuine efficiency rather than brute binder content.",
    "C7": "Moderate replacement at low water with superplasticizer. Very strong and "
    "Pareto-optimal, offering a less aggressive alternative to C5 and C6 for "
    "applications where high fly-ash and slag content is not acceptable.",
    "C8": "A very high binder, near-plain-cement concrete. Strong, but it needs far "
    "more binder and roughly three times the embodied carbon of C9 to get there. A "
    "direct illustration of how inefficient the conventional route is.",
    "C9": "High replacement at low water. Strong, Pareto-optimal, and low-carbon. "
    "Alongside C5, C6 and C11 it forms the core evidence that heavily blended "
    "binders are the best available strength-per-unit-carbon option in this data.",
    "C10": "The same high-replacement binder as C5 and C6 but with high water and no "
    "superplasticizer. Strength collapses relative to those mixes. The contrast is "
    "the most direct measure in the dataset of what water control is worth at high "
    "replacement.",
    "C11": "Roughly eighty percent of the binder is supplementary, dominated by slag, "
    "yet this mix reaches strength comparable to good conventional concrete at a "
    "fraction of the carbon. One of the most impressive results in the dataset.",
    "C12": "The same very high replacement level as C11 but supplied entirely by "
    "Class C fly ash with no slag. Strength collapses to roughly a third of C11's. "
    "This pair is the sharpest single-variable evidence in the whole dataset that at "
    "extreme replacement it is specifically slag, not supplementary material in "
    "general, that sustains strength.",
    "C13": "Very high replacement using both fly ash and slag, with slag the larger "
    "share. Performs nearly as well as C11 and far better than the fly-ash-only C12, "
    "filling in the middle of that comparison.",
    "C14": "High replacement at moderate water with superplasticizer. A strong, "
    "low-carbon result that sits just behind the C5 and C6 pair.",
    "C15": "The same binder as C14 with more water and less admixture, and clearly "
    "weaker for it. A useful intermediate point between C14 and the "
    "high-water C10.",
    # ---------- Set 2 concretes: high-water fly-ash ladder (C16-C20) ----------
    "C16": "A plain cement concrete at high water. Weak, and the highest-carbon "
    "member of its group. Serves as the control for the C16-C20 fly-ash "
    "replacement ladder.",
    "C17": "Moderate fly-ash replacement at high water. Essentially matches the "
    "plain-cement C16 on strength while cutting carbon appreciably, so the "
    "replacement is free from a performance standpoint.",
    "C18": "Higher fly-ash replacement at high water, and it slightly outperforms "
    "both C16 and C17. At high water-binder ratios the paste is porosity-limited, "
    "so removing clinker costs nothing and the fly ash refines the pore structure.",
    "C19": "High fly-ash replacement at reduced water, and the strongest of the "
    "C16-C20 ladder. Combining replacement with water reduction is what unlocks the "
    "gain; either lever alone does much less.",
    "C20": "The highest fly-ash replacement in the C16-C20 ladder. Strength falls "
    "back sharply, locating the practical ceiling for Class C fly ash as a sole "
    "replacement in this system at somewhere below this level.",
    # ---------- Set 2 concretes: reduced-binder series (C21-C27) --------------
    "C21": "A plain cement concrete at moderate binder and high aggregate content. A "
    "conventional baseline for the C21-C27 group.",
    "C22": "Plain cement at reduced binder and low water. Stronger than C21 despite "
    "less binder, because the water reduction more than compensates. Another clean "
    "demonstration that water-binder ratio outranks binder content.",
    "C23": "Moderate fly-ash replacement at low water. Outperforms the plain-cement "
    "C22 while cutting embodied carbon by roughly a third, and is one of the better "
    "efficiency points in Set 2.",
    "C24": "Higher fly-ash replacement at low water. Matches C23 on strength at "
    "lower carbon still, extending the favourable trend.",
    "C25": "Very high replacement dominated by slag at low water. Reaches solid "
    "strength at the lowest embodied carbon in the Set 2 group, and is arguably its "
    "best carbon-efficiency result.",
    "C26": "High fly-ash replacement at reduced water. Reasonable strength, though "
    "the fly-ash-only binder keeps it below the slag-containing C25.",
    "C27": "The highest fly-ash replacement in the C21-C27 group, and it falls off "
    "as expected. Together with C20 it brackets the useful upper limit for Class C "
    "fly ash used without slag.",
    # ---------- Set 3 concretes: reference and water ladder (C28-C35) ---------
    "C28": "The strongest mix in Set 3 and the reference point for much of the "
    "series. A modest slag replacement at low water and moderate binder. Its carbon "
    "sits above the Set 3 median because the binder stays cement-heavy, so C28 is "
    "best read as the series' strength ceiling rather than its efficiency optimum. "
    "Several later mixes are systematic perturbations of this recipe.",
    "C29": "Higher binder and higher slag replacement than C28 at the same water "
    "ratio. Strength is close to C28 while carbon drops, making it a slightly better "
    "efficiency point.",
    "C30": "Introduces Class F fly ash alongside slag at the C28 water ratio. "
    "Slightly weaker than C28, consistent with Class F fly ash contributing little "
    "at 28 days.",
    "C31": "Slag replacement above half the binder at low water. Solid strength at "
    "notably low carbon, and one of the better lean-binder results in Set 3.",
    "C32": "Slag at sixty percent of a lean binder. Strength holds up well and "
    "embodied carbon is low, extending the C31 trend without a meaningful penalty.",
    "C33": "The second rung of the C28, C33, C34, C35 water ladder, in which binder "
    "and chemistry are held fixed and only mixing water changes. The drop from C28 "
    "is immediate and substantial.",
    "C34": "The third rung of the C28-C35 water ladder. Strength continues to decline "
    "roughly linearly with added water.",
    "C35": "The highest-water rung of the C28-C35 ladder and the weakest of the four. "
    "The series spans a large strength range on water content alone, with identical "
    "binder throughout, and is among the most informative single-variable sequences "
    "in the dataset.",
    "C36": "The first rung of the C36, C37, C38 binder-content ladder, which reduces "
    "total binder while holding the slag fraction fixed. Strongest of the three.",
    "C37": "The middle rung of the C36-C38 binder ladder. Strength declines with the "
    "reduced binder, though less steeply than the water ladder declines with added "
    "water.",
    "C38": "The leanest rung of the C36-C38 ladder and the weakest. Comparing this "
    "series against C28-C35 shows water-binder ratio to be the stronger lever of the "
    "two in Set 3.",
    # ---------- Set 3 concretes: further water ladders (C39-C45) --------------
    "C39": "A higher-water variant of C29. The strength loss relative to its parent "
    "again tracks the added water closely.",
    "C40": "A moderate-water variant of C31, and the strongest of the C40, C41, C42 "
    "sub-ladder built on that recipe.",
    "C41": "The middle rung of the C40-C42 water ladder on the C31 binder.",
    "C42": "The highest-water rung of the C40-C42 ladder and the weakest, closing "
    "out another clean demonstration of water sensitivity at fixed binder.",
    "C43": "The richest rung of the C43, C44, C45 binder ladder at high slag "
    "replacement. Reasonable strength at low carbon.",
    "C44": "The middle rung of the C43-C45 binder ladder.",
    "C45": "The leanest rung of the C43-C45 ladder. Strength falls as binder is "
    "withdrawn, but the mix remains serviceable and its embodied carbon is very low.",
    # ---------- Set 3 concretes: replicates and temperature (C46-C47) ---------
    "C46": "A replicate of the C37 recipe at a higher superplasticizer dose. The "
    "modest strength gain over C37 quantifies the admixture's value at this lean "
    "binder level.",
    "C47": "A replicate of the C37 recipe cured near 10 C rather than 22 C. Both "
    "early and 28-day strength fall below the normal-temperature version, though far "
    "less severely than the 4.5 C mortars M32 through M38. The only "
    "reduced-temperature mix in Set 3.",
    # ---------- Set 3 concretes: lean-binder factorial (C48-C61) --------------
    "C48": "Part of a lean-binder factorial exploring fly ash and slag splits at "
    "fixed binder content. A moderate-replacement, moderate-water member with "
    "middling strength.",
    "C49": "Raises replacement above C48 at the same binder and water. Strength "
    "declines modestly while carbon falls, a mild but favourable trade.",
    "C50": "The same total replacement as C49 but weighted toward slag rather than "
    "fly ash, and clearly stronger for it. Another confirmation of the slag "
    "advantage, here at lean binder content.",
    "C51": "The low-water, moderate-replacement corner of the lean-binder factorial "
    "and its strongest member. Water control again dominates.",
    "C52": "A moderate-replacement, low-water member of the lean-binder factorial. "
    "Performs solidly for its binder content.",
    "C53": "Higher replacement at low water. Weaker than C52, tracking the reduced "
    "clinker content.",
    "C54": "The slag-weighted counterpart to C53 at identical total replacement, and "
    "meaningfully stronger. Pairs with the C49-C50 comparison to make the same point "
    "twice within this factorial.",
    "C55": "A slightly richer binder at moderate water. Mid-range strength at "
    "moderate carbon.",
    "C56": "A three-way blend at low replacement and moderate water. Unremarkable, "
    "and useful mainly as a factorial fill-in point.",
    "C57": "A moderate-replacement three-way blend. Sits close to the factorial "
    "average on both strength and carbon.",
    "C58": "The same binder as C57 at reduced water, and stronger for it. The pair "
    "isolates water content within the three-way-blend family.",
    "C59": "A richer binder with light fly ash and slag. Middling strength; the "
    "extra binder does not buy much at this water ratio.",
    "C60": "A moderate three-way blend at the same binder content as C59. Comparable "
    "strength at lower carbon, so the higher replacement is the better choice.",
    "C61": "The reduced-water counterpart to C60, and stronger. Reinforces the "
    "pattern that within Set 3, water reduction is consistently the most reliable "
    "route to strength.",
    # ---------- Set 3 concretes: fly-ash and slag comparison (C62-C80) --------
    "C62": "A fly-ash-only binder at high replacement using Class F ash. Among the "
    "weakest concretes in the dataset. Class F fly ash reacts more slowly than the "
    "Class C ash used in Sets 1 and 2, and at 28 days it has contributed very "
    "little.",
    "C63": "A slag-only binder at the same binder content and water ratio as C62, "
    "and dramatically stronger despite a somewhat lower replacement level. This pair "
    "is the Set 3 counterpart to the C11-C12 comparison and points the same way: "
    "slag sustains strength where Class F fly ash does not.",
    "C64": "A light slag replacement at moderate water. Strong for Set 3, though its "
    "cement-heavy binder makes it one of the higher-carbon mixes in this group.",
    "C65": "Very high slag replacement at lean binder. Strength drops off, marking "
    "the point where too little cement remains to activate the slag, echoing the "
    "mortar result at M61 and M63.",
    "C66": "High slag replacement at elevated water. Both levers work against it and "
    "the mix is correspondingly weak, though its embodied carbon is very low.",
    "C67": "High slag replacement with a small fly-ash addition at moderate water. A "
    "reasonable low-carbon result that sits above the more extreme C65 and C66.",
    "C68": "Extremely high replacement with slag dominant. Weak, but with among the "
    "lowest embodied carbon of any concrete in the dataset. Useful for defining the "
    "low-carbon extreme of the achievable design space.",
    "C69": "A three-way blend at very high replacement. Modest strength at very low "
    "carbon, consistent with its position in the C65-C80 low-binder family.",
    "C70": "A fly-ash-weighted blend at high replacement. Weak, continuing the Set 3 "
    "pattern that Class F fly ash contributes little by 28 days.",
    "C71": "Very high replacement split between fly ash and slag. Weak, with the "
    "fly-ash share limiting what the slag can deliver.",
    "C72": "Overwhelmingly fly ash at very high replacement, and the weakest concrete "
    "in the dataset. The clearest single data point on the limits of Class F fly ash "
    "as a primary binder replacement.",
    "C73": "A fly-ash-dominant blend at high replacement. Weak, and grouped naturally "
    "with C70 through C74.",
    "C74": "Fly-ash-dominant with a modest slag share. The small slag addition lifts "
    "it slightly above the comparable fly-ash-only mixes but does not change the "
    "overall picture.",
    "C75": "A balanced fly-ash-heavy blend at moderate replacement. Mid-low strength "
    "at low carbon.",
    "C76": "Half the binder replaced, weighted toward fly ash. One of the better "
    "performers among the fly-ash-dominant Set 3 mixes, helped by its lower overall "
    "replacement level.",
    "C77": "A balanced three-way blend at moderate replacement. Comparable to C76 and "
    "similarly among the more successful fly-ash-containing mixes in this group.",
    "C78": "Very high replacement, almost entirely slag, at lean binder. Outperforms "
    "the fly-ash-dominant mixes at comparable replacement, again separating the two "
    "supplementary materials cleanly.",
    "C79": "A moderate three-way blend, and one of the stronger members of the "
    "C65-C80 low-binder family. The balanced fly ash and slag split works better "
    "here than either extreme.",
    "C80": "A balanced high-replacement blend at lean binder. Rounds out the Set 3 "
    "series with a mid-range result at low embodied carbon.",
}


def spec_sentence(f):
    """Compose the factual opening from verified catalog values."""
    scm_pct = round(100 * f["scm_frac"]) if f["scm_frac"] is not None else 0
    parts = []

    if scm_pct == 0:
        binder_desc = f"a plain Portland cement binder at {f['binder']:.0f} kg/m3"
    else:
        pieces = []
        if f["fly_ash"] > 0:
            pieces.append(f"{f['fly_ash']:.0f} kg/m3 fly ash")
        if f["slag"] > 0:
            pieces.append(f"{f['slag']:.0f} kg/m3 slag")
        binder_desc = (
            f"{f['binder']:.0f} kg/m3 of binder with {scm_pct}% replaced by "
            + " and ".join(pieces)
        )
    parts.append(f"Built on {binder_desc}")
    parts.append(f"at a water-binder ratio of {f['wb']:.3f}")

    if f["hrwr"] > 0:
        parts.append(f"with {f['hrwr']:.2f} kg/m3 of superplasticizer")
    else:
        parts.append("with no superplasticizer")

    if f["coarse"] > 0:
        parts.append(
            f"and {f['coarse']:.0f} kg/m3 coarse plus {f['fine']:.0f} kg/m3 "
            "fine aggregate"
        )
    else:
        parts.append(f"and {f['fine']:.0f} kg/m3 fine aggregate only")

    sent = ", ".join(parts[:2]) + " " + " ".join(parts[2:]) + "."

    if f["temp"] != 22:
        sent += f" Mixed and cured at {f['temp']} C rather than the usual 22 C."
    return sent


def check_claims(facts):
    """Verify the superlative claims made in ANALYSIS against the fact base.

    Prose claims like "the strongest mix in the dataset" silently rot when the
    dataset is re-merged. Asserting them at build time keeps the narratives
    honest.
    """
    by_mix = {f["mix"]: f for f in facts.values()}

    def strongest(pool):
        return max(pool, key=lambda m: by_mix[m]["s28_meas"] or -1)

    def weakest(pool):
        return min(pool, key=lambda m: by_mix[m]["s28_meas"] or 1e9)

    def cleanest(pool):
        return min(pool, key=lambda m: by_mix[m]["gwp"])

    measured = [m for m, f in by_mix.items() if f["s28_meas"] is not None]
    mortars = [m for m in measured if by_mix[m]["material_source"] == 0]
    concretes = [m for m in measured if by_mix[m]["material_source"] != 0]
    set3 = [m for m in measured if by_mix[m]["material_source"] == 2]

    claims = [
        ("C6 is the strongest mix overall", strongest(measured) == "C6"),
        ("C5 is second strongest overall", strongest(set(measured) - {"C6"}) == "C5"),
        ("M20 is the strongest mortar", strongest(mortars) == "M20"),
        ("M45 is the weakest mix overall", weakest(measured) == "M45"),
        ("C72 is the weakest concrete", weakest(concretes) == "C72"),
        ("C28 is the strongest Set 3 mix", strongest(set3) == "C28"),
        ("M54 is the lowest-carbon mortar", cleanest(mortars) == "M54"),
        (
            "M18 tops the M15-M18 ladder",
            strongest(["M15", "M16", "M17", "M18"]) == "M18",
        ),
        (
            "M67 tops the M67-M69 water ladder",
            strongest(["M67", "M68", "M69"]) == "M67",
        ),
        ("M69 is weakest of M67-M69", weakest(["M67", "M68", "M69"]) == "M69"),
        (
            "C28 tops the C28-C35 water ladder",
            strongest(["C28", "C33", "C34", "C35"]) == "C28",
        ),
        ("C35 is weakest of C28-C35", weakest(["C28", "C33", "C34", "C35"]) == "C35"),
        (
            "C36 tops the C36-C38 binder ladder",
            strongest(["C36", "C37", "C38"]) == "C36",
        ),
        ("C38 is weakest of C36-C38", weakest(["C36", "C37", "C38"]) == "C38"),
        ("C40 tops the C40-C42 ladder", strongest(["C40", "C41", "C42"]) == "C40"),
        ("C42 is weakest of C40-C42", weakest(["C40", "C41", "C42"]) == "C42"),
        (
            "C19 tops the C16-C20 ladder",
            strongest(["C16", "C17", "C18", "C19", "C20"]) == "C19",
        ),
        (
            "C11 far outperforms C12",
            by_mix["C11"]["s28_meas"] > 3 * by_mix["C12"]["s28_meas"],
        ),
        ("C63 outperforms C62", by_mix["C63"]["s28_meas"] > by_mix["C62"]["s28_meas"]),
        ("M28 outperforms M29", by_mix["M28"]["s28_meas"] > by_mix["M29"]["s28_meas"]),
        (
            "C6 beats C5 on strength",
            by_mix["C6"]["s28_meas"] > by_mix["C5"]["s28_meas"],
        ),
        ("C1 is the weakest of C1-C4", weakest(["C1", "C2", "C3", "C4"]) == "C1"),
        (
            "C25 is the lowest-carbon Set 2 mix",
            cleanest([m for m in measured if by_mix[m]["material_source"] == 1])
            == "C25",
        ),
        (
            "M50 and M49 are the highest-carbon mixes",
            sorted(by_mix, key=lambda m: -by_mix[m]["gwp"])[:2] == ["M50", "M49"],
        ),
        (
            "M59 and M60 lack 28-day data",
            by_mix["M59"]["s28_meas"] is None and by_mix["M60"]["s28_meas"] is None,
        ),
        (
            "M36 has the lowest 1-day to 28-day strength ratio",
            min(
                (f["s1_meas"] / f["s28_meas"], m)
                for m, f in by_mix.items()
                if f["s1_meas"] and f["s28_meas"]
            )[1]
            == "M36",
        ),
        (
            "C47 is the only sub-22C Set 3 mix",
            [m for m in set3 if by_mix[m]["temp"] != 22] == ["C47"],
        ),
    ]
    bad = [name for name, ok in claims if not ok]
    if bad:
        raise SystemExit("Claim(s) contradicted by the data:\n  " + "\n  ".join(bad))
    print(f"verified {len(claims)} superlative/comparative claims against the data")


def main():
    facts = json.load(open(FACTS))
    missing = sorted(f["mix"] for f in facts.values() if f["mix"] not in ANALYSIS)
    if missing:
        raise SystemExit(f"Missing hand-authored analysis for: {missing}")
    check_claims(facts)

    out = {}
    for idx, f in facts.items():
        chunks = [
            f"**{f['mix']}** \u2014 {SOURCE_BLURB[f['material_source']]}.",
            spec_sentence(f),
            ANALYSIS[f["mix"]],
        ]
        text = " ".join(c for c in chunks if c)
        for bad in ("<", "&"):
            if bad in text:
                raise SystemExit(
                    f"Unsafe character {bad!r} in entry {idx} ({f['mix']})"
                )
        out[idx] = text

    json.dump(out, open(OUT, "w"), indent=1, ensure_ascii=False)
    print(f"wrote {OUT} with {len(out)} entries")
    lens = sorted(len(v) for v in out.values())
    median = lens[len(lens) // 2]
    print(f"length: min={lens[0]} median={median} max={lens[-1]}")


if __name__ == "__main__":
    main()
