"""
Phase 3 step 3: score a blind biochemical prior against measured transfer.

A biochemistry review was commissioned WITHOUT access to any results file. It
read only the metabolite annotation table (names, HMDB IDs, cohort counts) and
then committed, in advance, to: a five-tier ranking of compound classes by
expected cross-cohort transferability, a named list of compounds predicted to
transfer, a named list predicted not to, and explicit conditions under which its
own framework should be judged wrong. Those predictions are transcribed here
verbatim as HMDB IDs and scored against `generalization_loco_variants.csv`.

This is the one test in the project that cannot be explained by the mass-spec
units problem. Platform confounding is indifferent to which molecule it is
attached to, so it cannot produce a correlation between measured transfer and a
biochemist's prior about enzymes. If the prior scores, the residual structure in
the transfer results is mechanism. If it does not, the units story is the whole
story and the biology framing should be dropped.

The prior's own stated framework, which is sharper than "microbial vs host":
transferability is high when (a) substrate supply is constant and NOT set by
diet, (b) the microbial step is phylogenetically narrow rather than redundant,
and (c) the measured pool is a terminal product rather than a high-flux
intermediate. That predicts the counterintuitive call scored below -- that bile
acids transfer better than butyrate, despite butyrate being the more famously
"microbial" compound -- because butyrate is made redundantly by dozens of genera
and ~95% of it is absorbed before reaching stool.

Scored predictions:
  tier_correlation  Spearman between predicted tier (A=1..E=5) and measured
                    transfer. The prior called -0.35 to -0.55, and nominated
                    > -0.2 as falsification.
  named_lists       Hit rate on the explicit DO-transfer and DO-NOT-transfer
                    lists, against the measured median as the decision boundary.
  sharp_bet         Does deoxycholate beat butyrate? The prior staked its
                    framework on this one.
  falsifiers        (1) flat profile across tiers, (2) xenobiotics transferring
                    as well as bile acids -- which would indicate leakage and
                    void the comparison, (3) bile acids failing to beat the
                    global median.

Output:
    experiments/biochem_prior_scored.csv
    experiments/run_log.jsonl  (appended)

Usage:
    python scripts/score_biochem_prior.py
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, spearmanr

from _paths import REFERENCE_DIR, ROOT

sys.path.insert(0, str(ROOT / "src"))
from runlog import log_run  # noqa: E402

EXPERIMENTS_DIR = ROOT / "experiments"

# --- the prior, transcribed verbatim as HMDB IDs -----------------------------
# Tier A: constant host-supplied substrate, narrow microbial gate, terminal product
# Tier B: microbial but redundant and/or absorption-confounded
# Tier C: microbial transformation of a DIETARY substrate (substrate varies)
# Tier D: microbial step not rate-limiting, or pool dominated by non-microbial supply
# Tier E: essentially unpredictable (exposure-dominated) -- negative controls
TIERS: dict[str, int] = {}
for _ids, _tier in [
    # A
    (["HMDB0000626", "HMDB0000761", "HMDB0000631", "HMDB0000391", "HMDB0000946",
      "HMDB0000036", "HMDB0000951", "HMDB0000637", "HMDB0000251", "HMDB0004160",
      "HMDB0000054", "HMDB0000056"], 1),
    # B
    (["HMDB0000039", "HMDB0000237", "HMDB0000042", "HMDB0000190", "HMDB0000254",
      "HMDB0000784"], 2),
    # C
    (["HMDB0000718", "HMDB0001873", "HMDB0002302", "HMDB0000738", "HMDB0000197",
      "HMDB0000671", "HMDB0000682", "HMDB0011635", "HMDB0000020", "HMDB0001858",
      "HMDB0000209", "HMDB0000306", "HMDB0002322", "HMDB0001414", "HMDB0001257"], 3),
    # D
    (["HMDB0000158", "HMDB0000929", "HMDB0000906", "HMDB0000925", "HMDB0000244",
      "HMDB0000235", "HMDB0000210", "HMDB0000030", "HMDB0001545", "HMDB0000067",
      "HMDB0000292", "HMDB0000300", "HMDB0000262", "HMDB0000294", "HMDB0000562"], 4),
    # E
    (["HMDB0001847", "HMDB0001859", "HMDB0001921", "HMDB0000840", "HMDB0001924",
      "HMDB0003072", "HMDB0000714"], 5),
]:
    TIERS.update(dict.fromkeys(_ids, _tier))

PREDICTED_TRANSFERS = [
    "HMDB0000626", "HMDB0000251", "HMDB0000631", "HMDB0000056", "HMDB0000784",
    "HMDB0000951", "HMDB0002302", "HMDB0000761", "HMDB0000391", "HMDB0001414",
    "HMDB0002322", "HMDB0000020",
]
PREDICTED_FAILS = [
    "HMDB0000158", "HMDB0000292", "HMDB0000562", "HMDB0000294", "HMDB0000925",
    "HMDB0000906", "HMDB0000244", "HMDB0000235", "HMDB0000210", "HMDB0000030",
    "HMDB0000067", "HMDB0000929", "HMDB0003072", "HMDB0001847", "HMDB0001859",
    "HMDB0000840", "HMDB0001921", "HMDB0001924", "HMDB0000190", "HMDB0000254",
]
# The prior staked its framework on deoxycholate strictly beating butyrate.
SHARP_BET = {"deoxycholate": "HMDB0000626", "butyrate": "HMDB0000039"}
BILE_ACIDS = ["HMDB0000626", "HMDB0000761", "HMDB0000631", "HMDB0000391",
              "HMDB0000946", "HMDB0000036", "HMDB0000951", "HMDB0000637"]
XENOBIOTICS = ["HMDB0001847", "HMDB0001859", "HMDB0001921", "HMDB0000840", "HMDB0001924"]

TIER_NAMES = {1: "A", 2: "B", 3: "C", 4: "D", 5: "E"}


def load_scored() -> tuple[pd.DataFrame, str]:
    path = EXPERIMENTS_DIR / "generalization_loco_variants.csv"
    if not path.exists():
        raise SystemExit(f"missing {path} -- run scripts/generalization.py first")
    raw = pd.read_csv(path)
    best = raw.groupby("variant")["spearman"].median().idxmax()
    per_target = (
        raw[raw["variant"] == best]
        .groupby("target")
        .agg(rho=("spearman", "median"), n_folds=("spearman", "size"))
        .reset_index()
    )
    ref = pd.read_csv(REFERENCE_DIR / "metabolite_reference.csv")
    merged = per_target.merge(ref[["hmdb_id", "name"]], left_on="target",
                              right_on="hmdb_id", how="left")
    merged["tier"] = merged["target"].map(TIERS)
    return merged, str(best)


def _median_of(df: pd.DataFrame, ids: list[str]) -> tuple[float, int]:
    hit = df[df["target"].isin(ids)]["rho"].dropna()
    return (float(hit.median()) if len(hit) else np.nan), len(hit)


def main() -> None:
    df, variant = load_scored()
    overall = float(df["rho"].median())
    tiered = df.dropna(subset=["tier"])

    print(f"variant scored: {variant}")
    print(f"{len(df)} targets measured; {len(tiered)} carry a tier assignment "
          f"({len(TIERS)} compounds in the prior, the rest are outside the "
          f"densest-{len(df)} panel)")
    print(f"global median rho = {overall:.3f}  (the decision boundary below)\n")

    with log_run("score_biochem_prior", params={"variant": variant}) as run:
        # --- tier correlation ------------------------------------------------
        print("=== transfer by predicted tier ===")
        by_tier = (tiered.groupby("tier")
                   .agg(n=("rho", "size"), median_rho=("rho", "median"))
                   .round(3))
        by_tier.index = [f"{TIER_NAMES[t]} ({t})" for t in by_tier.index]
        print(by_tier.to_string())

        rho_corr, p_corr = (spearmanr(tiered["tier"], tiered["rho"])
                            if len(tiered) >= 8 else (np.nan, np.nan))
        print(f"\n  Spearman(tier, measured rho) = {rho_corr:+.3f}  (p = {p_corr:.4f}, n = {len(tiered)})")
        print(f"  prior predicted -0.35 to -0.55; it nominated > -0.20 as falsification")
        tier_verdict = ("CONFIRMED - within predicted range" if -0.55 <= rho_corr <= -0.35
                        else "DIRECTIONALLY CORRECT - stronger than predicted" if rho_corr < -0.55
                        else "DIRECTIONALLY CORRECT - weaker than predicted" if -0.35 < rho_corr <= -0.20
                        else "FALSIFIED by the prior's own criterion")
        print(f"  -> {tier_verdict}\n")

        # --- named lists -----------------------------------------------------
        hits = df[df["target"].isin(PREDICTED_TRANSFERS)].dropna(subset=["rho"])
        fails = df[df["target"].isin(PREDICTED_FAILS)].dropna(subset=["rho"])
        hit_rate = float((hits["rho"] > overall).mean()) if len(hits) else np.nan
        fail_rate = float((fails["rho"] <= overall).mean()) if len(fails) else np.nan

        print("=== named compound lists (boundary = global median) ===")
        print(f"  predicted TO transfer    : {len(hits):2d} testable, "
              f"{hit_rate:.0%} above median (median rho {hits['rho'].median():.3f})")
        print(f"  predicted NOT to transfer: {len(fails):2d} testable, "
              f"{fail_rate:.0%} at/below median (median rho {fails['rho'].median():.3f})")
        if len(hits) >= 3 and len(fails) >= 3:
            u_p = float(mannwhitneyu(hits["rho"], fails["rho"], alternative="greater").pvalue)
            print(f"  separation: Mann-Whitney one-sided p = {u_p:.4f}")
        else:
            u_p = np.nan
        for label, frame in (("predicted TO transfer", hits), ("predicted NOT to", fails)):
            if len(frame):
                shown = frame.sort_values("rho", ascending=False)[["name", "rho", "n_folds"]]
                print(f"\n  -- {label} --")
                print(shown.to_string(index=False))
        print()

        # --- the sharp bet ---------------------------------------------------
        print("=== the sharp bet: deoxycholate > butyrate ===")
        bet = {k: _median_of(df, [v])[0] for k, v in SHARP_BET.items()}
        for name, value in bet.items():
            print(f"  {name:14s} rho = {value:.3f}" if np.isfinite(value)
                  else f"  {name:14s} not in the measured panel")
        if all(np.isfinite(v) for v in bet.values()):
            won = bet["deoxycholate"] > bet["butyrate"]
            print(f"  -> {'PRIOR WINS' if won else 'PRIOR LOSES - redundancy/absorption argument is wrong'}")
        else:
            won = None
            print("  -> untestable in this panel")
        print()

        # --- falsifiers ------------------------------------------------------
        bile_med, n_bile = _median_of(df, BILE_ACIDS)
        xeno_med, n_xeno = _median_of(df, XENOBIOTICS)
        print("=== falsifier checks ===")
        print(f"  bile acids   : median rho = {bile_med:.3f} (n={n_bile}) vs global {overall:.3f}")
        print(f"  xenobiotics  : median rho = {xeno_med:.3f} (n={n_xeno})")
        f2 = np.isfinite(xeno_med) and np.isfinite(bile_med) and xeno_med >= bile_med
        f3 = np.isfinite(bile_med) and bile_med <= overall
        print(f"  F2 xenobiotics >= bile acids (would indicate LEAKAGE): {'TRIGGERED' if f2 else 'not triggered'}")
        print(f"  F3 bile acids fail to beat global median            : {'TRIGGERED' if f3 else 'not triggered'}")

        run.record(variant=variant, tier_spearman=round(float(rho_corr), 4),
                   tier_p=round(float(p_corr), 5), n_tiered=len(tiered),
                   hit_rate_predicted_transfers=None if np.isnan(hit_rate) else round(hit_rate, 3),
                   hit_rate_predicted_fails=None if np.isnan(fail_rate) else round(fail_rate, 3),
                   list_separation_p=None if np.isnan(u_p) else round(u_p, 5),
                   sharp_bet_won=won, bile_acid_rho=round(bile_med, 4) if np.isfinite(bile_med) else None,
                   xenobiotic_rho=round(xeno_med, 4) if np.isfinite(xeno_med) else None,
                   falsifier_leakage=bool(f2), falsifier_bile_acids=bool(f3))
        run.note(f"tier correlation {rho_corr:+.3f}: {tier_verdict}")

        out = EXPERIMENTS_DIR / "biochem_prior_scored.csv"
        tiered.sort_values(["tier", "rho"], ascending=[True, False]).to_csv(out, index=False)
        run.artifact(out)
        print(f"\n  -> {out}")


if __name__ == "__main__":
    main()
