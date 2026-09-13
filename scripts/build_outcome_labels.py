"""
Phase 2 step 5: harmonize the 14 cohorts' study_group strings into one
condition vocabulary.

Each cohort labels its subjects in its own dialect: "Control", "Healthy",
"Normal", "nonIBD", "control", "H", and "0" all mean "this person is the
comparison group," and no model can use them until they agree. This produces
one harmonized label per sample plus a case/control flag, and links each
harmonized condition to the Disbiome disease name so the literature layer
(fetch_disbiome.py) joins cleanly.

**The trap this file exists to flag.** Every cohort studies one disease on one
sequencing+metabolomics platform. So "IBD" is perfectly correlated with "was
run by the FRANZOSA/iHMP labs," and a classifier trained across cohorts can
score near-perfect AUROC by learning each cohort's batch signature instead of
any biology -- and would then be useless on a new person's sample. That is the
single easiest way to produce an impressive-looking and completely worthless
result here.

The defense, applied downstream in train_outcome_model.py, is that the honest
evaluation is **case vs. control within the same cohort**, where batch is held
constant by construction. `comparison_group` below marks which cohorts support
that (they contain both cases and controls on the same platform); cohorts
without an internal control arm can't answer the question and are excluded from
the headline metric rather than quietly inflating it.

Judgment calls made here, all reviewable in the output CSV:
  - HE_INFANTS_MFGM_2019's arms are feeding regimens and timepoints, not
    disease states -> labeled `not_a_condition`, excluded from outcome modeling.
  - YACHIDA's Stage_0/I_II/III_IV/MP/HS collapse to colorectal cancer stages;
    MP (mucosal polyp) and HS (history of surgery?) are ambiguous in the source
    metadata, so they're kept as their own labels rather than guessed into
    "cancer."
  - KIM's Adenoma is precancerous, not cancer -> its own condition.
  - WANDRO's preterm infants: sepsis and NEC are distinct conditions, and the
    "GI" arm has a single sample so it's dropped as unusable.

Output:
    data/reference/outcome_label_map.csv   (checked in; the mapping itself, for review)
    data/processed/outcome_labels.parquet  (per-sample harmonized labels)

Usage:
    python scripts/build_outcome_labels.py
"""
import sys

import pandas as pd

from _paths import PROCESSED_DIR, REFERENCE_DIR, ROOT

sys.path.insert(0, str(ROOT / "src"))
from runlog import log_run  # noqa: E402

# (cohort, raw study_group) -> harmonized condition, case/control role, Disbiome disease name.
# role: "control" = comparison arm, "case" = has the condition, "excluded" = unusable.
LABEL_MAP = {
    "ERAWIJANTARI_GASTRIC_CANCER_2020": {
        "Healthy": ("healthy", "control", ""),
        "Gastrectomy": ("post_gastrectomy", "case", "Gastric cancer"),
    },
    "FRANZOSA_IBD_2019": {
        "Control": ("healthy", "control", ""),
        "CD": ("crohns_disease", "case", "Crohn's Disease"),
        "UC": ("ulcerative_colitis", "case", "Ulcerative Colitis"),
    },
    "HE_INFANTS_MFGM_2019": {
        # Feeding-trial arms and timepoints, not disease states.
        "Baseline": ("not_a_condition", "excluded", ""),
        "Month12": ("not_a_condition", "excluded", ""),
        "With.comp.food": ("not_a_condition", "excluded", ""),
        "Without.comp.food": ("not_a_condition", "excluded", ""),
    },
    "JACOBS_IBD_FAMILIES_2016": {
        "Normal": ("healthy", "control", ""),
        "CD": ("crohns_disease", "case", "Crohn's Disease"),
        "UC": ("ulcerative_colitis", "case", "Ulcerative Colitis"),
    },
    "KANG_AUTISM_2017": {
        # Trailing space is in the source data, not a typo here.
        "Autistic ": ("autism", "case", "Autism"),
        "Neurotypical": ("healthy", "control", ""),
    },
    "KIM_ADENOMAS_2020": {
        "Control": ("healthy", "control", ""),
        "Adenoma": ("colorectal_adenoma", "case", "Colorectal cancer"),
        "Carcinoma": ("colorectal_cancer", "case", "Colorectal cancer"),
    },
    "KOSTIC_INFANTS_DIABETES_2015": {
        "case": ("type_1_diabetes", "case", "Type 1 Diabetes"),
        "control": ("healthy", "control", ""),
    },
    "MARS_IBS_2020": {
        "H": ("healthy", "control", ""),
        "D": ("ibs_diarrhea", "case", "Irritable Bowel Syndrome"),
        "C": ("ibs_constipation", "case", "Irritable Bowel Syndrome"),
    },
    "POYET_BIO_ML_2019": {},  # no study_group recorded at all
    "SINHA_CRC_2016": {
        "0": ("healthy", "control", ""),
        "1": ("colorectal_cancer", "case", "Colorectal cancer"),
    },
    "WANDRO_PRETERMS_2018": {
        "control": ("preterm_stable", "control", ""),
        "septic": ("neonatal_sepsis", "case", "Sepsis"),
        "nec": ("necrotizing_enterocolitis", "case", "Necrotizing enterocolitis"),
        "GI": ("not_a_condition", "excluded", ""),  # n=1, unusable
    },
    "WANG_ESRD_2020": {
        "Control": ("healthy", "control", ""),
        "ESRD": ("end_stage_renal_disease", "case", "Chronic kidney disease"),
    },
    "YACHIDA_CRC_2019": {
        "Healthy": ("healthy", "control", ""),
        "Stage_0": ("colorectal_cancer_stage_0", "case", "Colorectal cancer"),
        "Stage_I_II": ("colorectal_cancer", "case", "Colorectal cancer"),
        "Stage_III_IV": ("colorectal_cancer", "case", "Colorectal cancer"),
        # Ambiguous in the source metadata -- kept separate rather than guessed.
        "MP": ("colorectal_polyp", "case", "Colorectal cancer"),
        "HS": ("colorectal_other", "excluded", ""),
    },
    "iHMP_IBDMDB_2019": {
        "nonIBD": ("healthy", "control", ""),
        "CD": ("crohns_disease", "case", "Crohn's Disease"),
        "UC": ("ulcerative_colitis", "case", "Ulcerative Colitis"),
    },
}


def main() -> None:
    with log_run("build_outcome_labels") as run:
        manifest = pd.read_parquet(PROCESSED_DIR / "manifest.parquet")
        df = manifest.reset_index()

        rows, unmapped = [], []
        for _, r in df.iterrows():
            cohort, raw = r["cohort"], r["study_group"]
            entry = LABEL_MAP.get(cohort, {}).get(raw)
            if entry is None:
                condition, role, disbiome = ("unlabeled", "excluded", "")
                if isinstance(raw, str) and raw.strip():
                    unmapped.append((cohort, raw))
            else:
                condition, role, disbiome = entry
            rows.append({
                "cohort": cohort,
                "sample_id": r["sample_id"],
                "subject_id": r["subject_id"],
                "study_group_raw": raw,
                "condition": condition,
                "role": role,
                "disbiome_disease": disbiome,
            })

        labels = pd.DataFrame(rows)

        if unmapped:
            unique_unmapped = sorted(set(unmapped))
            print(f"  ! {len(unique_unmapped)} unmapped (cohort, study_group) pairs:")
            for cohort, raw in unique_unmapped[:20]:
                print(f"      {cohort}: {raw!r}")
            run.note(f"{len(unique_unmapped)} unmapped study_group values")

        # A cohort can support within-cohort case/control comparison only if it
        # has both arms on the same platform. Everything else is batch-confounded.
        usable = labels[labels["role"].isin(["case", "control"])]
        roles_per_cohort = usable.groupby("cohort")["role"].nunique()
        comparison_cohorts = set(roles_per_cohort[roles_per_cohort == 2].index)
        labels["comparison_group"] = labels["cohort"].isin(comparison_cohorts)

        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        out_labels = PROCESSED_DIR / "outcome_labels.parquet"
        labels.set_index(["cohort", "sample_id"]).to_parquet(out_labels)
        run.artifact(out_labels)

        # The mapping itself, flattened for review in a diff.
        map_rows = [
            {"cohort": cohort, "study_group_raw": raw, "condition": cond,
             "role": role, "disbiome_disease": disbiome,
             "n_samples": int(((labels["cohort"] == cohort) & (labels["study_group_raw"] == raw)).sum())}
            for cohort, mapping in LABEL_MAP.items()
            for raw, (cond, role, disbiome) in mapping.items()
        ]
        REFERENCE_DIR.mkdir(parents=True, exist_ok=True)
        out_map = REFERENCE_DIR / "outcome_label_map.csv"
        pd.DataFrame(map_rows).sort_values(["cohort", "study_group_raw"]).to_csv(out_map, index=False)
        run.artifact(out_map)

        n_case = int((labels["role"] == "case").sum())
        n_control = int((labels["role"] == "control").sum())
        run.record(
            n_samples=len(labels),
            n_case=n_case,
            n_control=n_control,
            n_excluded=int((labels["role"] == "excluded").sum()),
            n_conditions=int(labels.loc[labels["role"] == "case", "condition"].nunique()),
            n_comparison_cohorts=len(comparison_cohorts),
        )

        print(f"\n-> {out_map}")
        print(f"-> {out_labels}")
        print(f"\nSamples: {len(labels)}  (case {n_case}, control {n_control}, "
              f"excluded {(labels['role'] == 'excluded').sum()})")
        print(f"\nCohorts with BOTH case and control arms (usable for within-cohort "
              f"comparison): {len(comparison_cohorts)}/14")
        for c in sorted(comparison_cohorts):
            sub = usable[usable["cohort"] == c]
            print(f"  {c:36s} case={int((sub['role'] == 'case').sum()):4d} "
                  f"control={int((sub['role'] == 'control').sum()):4d}")
        excluded_cohorts = sorted(set(labels["cohort"]) - comparison_cohorts)
        print(f"\nExcluded from within-cohort comparison: {', '.join(excluded_cohorts)}")

        print("\nHarmonized condition counts:")
        print(labels[labels["role"] == "case"]["condition"].value_counts().to_string())


if __name__ == "__main__":
    main()
