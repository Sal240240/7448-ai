"""
Phase 3 step 2: is transfer failure uniform, or structured by biochemistry?

The cross-cohort result (median r = 0.002) has two competing explanations, and
they imply very different projects.

  Technical.  Cohorts ran different untargeted mass-spec platforms in arbitrary
              units, so pooling them teaches the model which platform produced a
              row rather than any biology. If this is the whole story, transfer
              failure should be roughly UNIFORM across compounds -- the units
              problem does not care what molecule it is attached to.

  Biological. Only some stool metabolites are microbial products. Butyrate is
              made by bacteria fermenting fibre; a phosphatidylcholine is mostly
              host membrane lipid that bacteria barely touch; a drug metabolite
              is whatever the patient swallowed. Compounds whose levels are set
              by diet or host physiology are not predictable from taxonomy in
              ANY cohort, however clean the units. If this is the story,
              transfer should be STRUCTURED -- high for fermentation products,
              zero for host and dietary compounds -- and the pooled median hides
              that by averaging the two populations together.

These are distinguishable, which is the point of this script. It groups each
target by biochemical family and asks whether cross-cohort transfer separates by
family. A flat profile supports the technical story; a profile where short-chain
fatty acids and bile acids sit well above complex lipids and xenobiotics
supports the biological one.

The practical stake: under the technical story the fix is normalization and the
product predicts a broad panel. Under the biological story no amount of
normalization helps the host-derived compounds, and the honest product is a
NARROW panel of genuinely microbial metabolites -- which is a smaller claim but
a defensible one, and the right target list for a focused assay.

Family assignment is keyword matching over compound names, which is a heuristic
and not a substitute for curation: it is reported with per-family counts so thin
or suspect groups are visible, and unmatched compounds are kept as "unassigned"
rather than silently dropped into a bucket.

Output:
    experiments/transfer_by_chemistry.csv      per-family summary
    experiments/transfer_per_metabolite.csv    per-compound detail, named
    experiments/run_log.jsonl                  (appended)

Usage:
    python scripts/transfer_chemistry.py
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from _paths import REFERENCE_DIR, ROOT

sys.path.insert(0, str(ROOT / "src"))
from runlog import log_run  # noqa: E402

EXPERIMENTS_DIR = ROOT / "experiments"

# Ordered most-to-least microbially-derived, which is the hypothesis under test.
# Patterns are lowercase substrings matched against the compound name. Order
# matters: the first family whose pattern matches wins, so the specific families
# are listed before the broad ones ("butyrate" before the generic "acid").
FAMILIES: list[tuple[str, tuple[str, ...]]] = [
    # Direct fermentation products: bacteria synthesize these from dietary fibre.
    ("scfa", ("butyrate", "butyric", "propionate", "propionic", "acetate", "acetic",
              "valerate", "valeric", "caproate", "hexanoate")),
    # Protein fermentation -- branched-chain, from amino acid catabolism.
    ("branched_scfa", ("isobutyr", "isovaler", "2-methylbutyr", "methylbutyrate")),
    # Bacterial transformation of a HOST product (bile acids are host-synthesized,
    # then deconjugated and 7alpha-dehydroxylated by gut bacteria).
    ("bile_acid", ("cholate", "cholic", "deoxycholic", "lithocholic", "ursodeoxy",
                   "chenodeoxy", "glycochol", "taurochol", "glycodeoxy", "taurodeoxy",
                   "bile acid", "muricholic")),
    # Tryptophan route: bacterial tryptophanase and reductases.
    ("indole_tryptophan", ("indole", "indol-", "indoxyl", "skatole", "tryptamine",
                           "tryptophan", "kynuren")),
    # Tyrosine/phenylalanine route: bacterial decarboxylation and dehydroxylation.
    ("phenolic", ("cresol", "cresyl", "phenylacet", "phenylpropion", "hydroxyphenyl",
                  "phenol", "hippurate", "hippuric", "benzoate", "phenyllact")),
    # Choline/carnitine route: cutC-carrying taxa, then host FMO3 to TMAO.
    ("tma_tmao", ("trimethylamine", "tmao", "methylamine", "betaine", "carnitine", "choline")),
    # Bacterial decarboxylation of amino acids.
    ("polyamine", ("putrescine", "cadaverine", "spermidine", "spermine", "agmatine")),
    ("vitamin", ("cobalamin", "folate", "folic", "menaquinone", "riboflavin", "thiamine",
                 "biotin", "pantothen", "pyridox", "niacin", "ascorb", "tocopherol")),
    # Host/diet dominated: expected to transfer poorly if the hypothesis holds.
    ("lipid_complex", ("phosphatidyl", "sphingomyelin", "ceramide", "lysopc", "lysope",
                       "triglyceride", "diacylglycerol", "plasmalogen", "cholesterol",
                       "sterol", "cardiolipin")),
    ("xenobiotic_drug", ("caffeine", "theobromine", "paraxanthine", "cotinine", "nicotine",
                         "ibuprofen", "acetaminophen", "salicyl", "metformin", "sulfamethox",
                         "warfarin", "atenolol", "drug", "medication")),
    ("nucleotide", ("adenosine", "guanosine", "cytidine", "uridine", "thymidine", "inosine",
                    "xanthine", "hypoxanthine", "urate", "uric acid", "purine", "pyrimidine")),
    ("amino_acid", ("alanine", "glycine", "serine", "threonine", "cysteine", "methionine",
                    "valine", "leucine", "isoleucine", "proline", "arginine", "histidine",
                    "lysine", "aspart", "glutam", "asparagine", "tyrosine", "phenylalanine",
                    "taurine", "ornithine", "citrulline")),
    ("bile_pigment", ("bilirubin", "urobilin", "stercobilin", "biliverdin")),
    ("fatty_acid_long", ("palmit", "stearat", "stearic", "oleat", "oleic", "linole",
                         "arachidon", "myrist", "laur", "docosa", "eicosa")),
    ("sugar_carbohydrate", ("glucose", "fructose", "galactose", "mannose", "sucrose",
                            "lactose", "maltose", "ribose", "xylose", "sorbitol",
                            "mannitol", "inositol")),
]

# Families expected to be microbially driven, for the headline contrast.
# bile_pigment belongs here, not with host/diet: urobilinogen and stercobilin are
# produced by gut bacteria reducing host-secreted bilirubin (the responsible
# enzyme, BilR, was characterized in 2024). It was miscategorized in the first
# run of this script, which diluted the contrast it was meant to measure.
MICROBIAL_FAMILIES = {"scfa", "branched_scfa", "bile_acid", "indole_tryptophan",
                      "phenolic", "tma_tmao", "polyamine", "bile_pigment"}
HOST_DIET_FAMILIES = {"lipid_complex", "xenobiotic_drug", "fatty_acid_long",
                      "sugar_carbohydrate"}

# A family represented by one or two compounds says nothing about that chemistry;
# it is reported for completeness but excluded from the headline contrast.
MIN_FAMILY_N = 5


def assign_family(name: str) -> str:
    """First matching family wins; specific patterns are ordered before broad ones."""
    if not isinstance(name, str):
        return "unassigned"
    lowered = name.lower()
    for family, patterns in FAMILIES:
        if any(p in lowered for p in patterns):
            return family
    return "unassigned"


def load_reference() -> pd.DataFrame:
    ref = pd.read_csv(REFERENCE_DIR / "metabolite_reference.csv")
    kegg_path = REFERENCE_DIR / "metabolite_kegg.csv"
    if kegg_path.exists():
        kegg = pd.read_csv(kegg_path)
        ref = ref.merge(kegg[["kegg_id", "brite_classes"]], on="kegg_id", how="left")
    else:
        ref["brite_classes"] = np.nan
    ref["family"] = ref["name"].map(assign_family)
    return ref


def load_transfer() -> pd.DataFrame:
    """Per-target cross-cohort scores, averaged over held-out cohorts, per variant."""
    path = EXPERIMENTS_DIR / "generalization_loco_variants.csv"
    if not path.exists():
        raise SystemExit(f"missing {path} -- run scripts/generalization.py first")
    df = pd.read_csv(path)
    return (
        df.groupby(["variant", "target"])
        .agg(cross_cohort_rho=("spearman", "median"),
             cross_cohort_pearson=("pearson", "median"),
             n_folds=("spearman", "size"))
        .reset_index()
    )


def load_within_cohort() -> pd.DataFrame:
    """The Phase 1 within-cohort number, for the in-lab vs out-of-lab contrast."""
    path = EXPERIMENTS_DIR / "elastic_net_baseline_metrics.csv"
    if not path.exists():
        return pd.DataFrame(columns=["target", "within_cohort_r"])
    df = pd.read_csv(path)
    return df[["target", "pearson_test"]].rename(columns={"pearson_test": "within_cohort_r"})


def summarize_by_family(merged: pd.DataFrame, variant: str) -> pd.DataFrame:
    subset = merged[merged["variant"] == variant]
    grouped = (
        subset.groupby("family")
        .agg(n_metabolites=("target", "size"),
             median_cross_cohort_rho=("cross_cohort_rho", "median"),
             median_within_cohort_r=("within_cohort_r", "median"),
             frac_rho_above_0_2=("cross_cohort_rho", lambda s: float((s > 0.2).mean())))
        .round(3)
        .sort_values("median_cross_cohort_rho", ascending=False)
    )
    grouped["expected_microbial"] = [
        "yes" if f in MICROBIAL_FAMILIES else "no" if f in HOST_DIET_FAMILIES else "-"
        for f in grouped.index
    ]
    return grouped.reset_index()


def main() -> None:
    ref = load_reference()
    transfer = load_transfer()
    within = load_within_cohort()

    merged = (
        transfer.merge(ref[["hmdb_id", "name", "family", "brite_classes"]],
                       left_on="target", right_on="hmdb_id", how="left")
        .merge(within, on="target", how="left")
    )
    merged["family"] = merged["family"].fillna("unassigned")

    variants = sorted(merged["variant"].unique())
    best_variant = (
        merged.groupby("variant")["cross_cohort_rho"].median().idxmax()
        if variants else None
    )

    with log_run("transfer_chemistry", params={"variants": variants}) as run:
        print(f"{merged['target'].nunique()} targets scored, variants: {variants}\n")

        for variant in variants:
            summary = summarize_by_family(merged, variant)
            print(f"=== transfer by biochemical family [{variant}] ===")
            print(summary.to_string(index=False))
            print()
            if variant == best_variant:
                summary.to_csv(EXPERIMENTS_DIR / "transfer_by_chemistry.csv", index=False)

        # The headline contrast: do the families we expect to be microbial
        # actually transfer better than the host/diet ones?
        best = merged[merged["variant"] == best_variant]
        family_n = best["family"].value_counts()
        well_sampled = set(family_n[family_n >= MIN_FAMILY_N].index)
        scored = best[best["family"].isin(well_sampled)]
        microbial = scored[scored["family"].isin(MICROBIAL_FAMILIES)]["cross_cohort_rho"].dropna()
        host_diet = scored[scored["family"].isin(HOST_DIET_FAMILIES)]["cross_cohort_rho"].dropna()

        # The primary signal is the spread ACROSS families, not the binary
        # contrast. The binary split groups chemically unlike families together
        # and, once thin families are excluded, can leave one side represented by
        # a single family -- in which case its "gap" measures that one family,
        # not the hypothesis. Report the spread first and caveat the contrast.
        summary_best = summarize_by_family(merged, best_variant)
        sampled = summary_best[summary_best["n_metabolites"] >= MIN_FAMILY_N]
        spread = float(sampled["median_cross_cohort_rho"].max()
                       - sampled["median_cross_cohort_rho"].min()) if len(sampled) >= 3 else np.nan
        n_host_families = sampled["family"].isin(HOST_DIET_FAMILIES).sum()
        run.record(family_spread=round(spread, 4) if np.isfinite(spread) else None,
                   n_well_sampled_families=int(len(sampled)))
        print(f"=== family spread [{best_variant}] ===")
        print(f"  best family  : {sampled.iloc[0]['family']} rho = {sampled.iloc[0]['median_cross_cohort_rho']:.3f}")
        print(f"  worst family : {sampled.iloc[-1]['family']} rho = {sampled.iloc[-1]['median_cross_cohort_rho']:.3f}")
        print(f"  spread       : {spread:.3f}\n")

        verdict = "inconclusive -- too few assigned compounds"
        if len(microbial) >= 5 and len(host_diet) >= 5:
            gap = float(microbial.median() - host_diet.median())
            verdict = ("STRUCTURED: microbial families transfer better" if gap > 0.05
                       else "FLAT: no separation by biochemistry" if abs(gap) <= 0.05
                       else "INVERTED: host/diet families transfer better")
            if n_host_families < 2:
                verdict += (f" -- but UNDERPOWERED: only {n_host_families} host/diet family "
                            f"clears n>={MIN_FAMILY_N}, so this contrast reflects that one "
                            f"family rather than the hypothesis. Read the family spread "
                            f"({spread:.3f}) instead.")
            run.record(microbial_median_rho=round(float(microbial.median()), 4),
                       host_diet_median_rho=round(float(host_diet.median()), 4),
                       gap=round(gap, 4),
                       n_microbial=len(microbial), n_host_diet=len(host_diet))
            print(f"=== hypothesis test [{best_variant}] ===")
            print(f"  microbially-derived families : median rho = {microbial.median():6.3f}  (n={len(microbial)})")
            print(f"  host/diet-derived families   : median rho = {host_diet.median():6.3f}  (n={len(host_diet)})")
            print(f"  gap                          : {gap:+.3f}")
        run.note(verdict)
        print(f"  -> {verdict}\n")

        # Named winners: what actually transfers, for the biological plausibility check.
        top = best.nlargest(25, "cross_cohort_rho")[
            ["name", "family", "cross_cohort_rho", "within_cohort_r", "n_folds"]
        ]
        print("=== top 25 transferring compounds ===")
        print(top.to_string(index=False))

        out = EXPERIMENTS_DIR / "transfer_per_metabolite.csv"
        merged.sort_values(["variant", "cross_cohort_rho"], ascending=[True, False]).to_csv(out, index=False)
        run.artifact(out)
        run.record(best_variant=str(best_variant),
                   n_assigned=int((merged["family"] != "unassigned").sum()))
        print(f"\n  -> {out}")


if __name__ == "__main__":
    main()
