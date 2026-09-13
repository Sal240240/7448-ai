"""
Phase 2 step 9: the backtests that decide whether any of this should be
believed.

The headline number in the README (median test r = 0.458) comes from one
70/15/15 split of patients drawn from the same 14 cohorts the model trained on.
That is the standard way to report this, and it answers a narrower question
than it appears to. Four tests here probe the parts it doesn't cover:

  negative_control  Shuffle each metabolite's values across training patients,
                    destroying any real relationship, and refit. A correctly
                    built pipeline scores ~0. Anything meaningfully above 0
                    means information is leaking from target to features --
                    which would invalidate every other number in the project.
                    This is a correctness gate, not a metric.

  cv_stability      Redraw the patient split under different random seeds and
                    refit. Reports the spread. A median r that swings widely
                    across seeds is a property of the split, not the model.

  cross_cohort      Train on 13 cohorts, test on the 14th, for every cohort.
                    This is the question that actually matters for using the
                    model on a new dataset: does it transfer to a lab it has
                    never seen? Within-cohort held-out patients share platform,
                    population and protocol, so they flatter the model.

  chemical_class    Break accuracy down by compound class. A single median
                    hides that some chemistry is predictable from microbial
                    composition and some is mostly diet or host metabolism.

Runtime note: the refit-based tests are restricted to a sample of targets
(--n-targets, default 80, chosen as the densest-measured ones) because refitting
1,098 elastic nets x 14 folds is hours of compute for a conclusion that a
well-chosen sample establishes. The sampling rule is stated in the output so
the numbers aren't mistaken for a full sweep.

Output:
    experiments/backtest_<test>.csv
    experiments/run_log.jsonl  (appended)

Usage:
    python scripts/backtest.py --test all
    python scripts/backtest.py --test cross_cohort --n-targets 40
"""
import argparse
import sys

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.linear_model import ElasticNet

from _paths import PROCESSED_DIR, REFERENCE_DIR, ROOT

sys.path.insert(0, str(ROOT / "src"))
from data import Dataset  # noqa: E402
from runlog import log_run  # noqa: E402

EXPERIMENTS_DIR = ROOT / "experiments"
DEFAULT_ALPHA, DEFAULT_L1 = 0.01, 0.5


def _hyperparams() -> dict[str, tuple[float, float]]:
    """Per-target (alpha, l1_ratio) chosen by the baseline's own CV search."""
    path = EXPERIMENTS_DIR / "elastic_net_baseline_metrics.csv"
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    return {r["target"]: (r["alpha"], r["l1_ratio"]) for _, r in df.iterrows()}


def _pearson(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) < 3 or np.std(y_true) == 0 or np.std(y_pred) == 0:
        return np.nan
    return float(np.corrcoef(y_true, y_pred)[0, 1])


def fit_predict(X_tr, y_tr, X_te, alpha: float, l1_ratio: float) -> np.ndarray:
    model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=5000)
    model.fit(X_tr, y_tr)
    return model.predict(X_te)


def select_targets(ds: Dataset, n_targets: int) -> list[str]:
    """Densest-measured targets: the ones with enough data for a fold to be meaningful."""
    counts = ds.mask.sum().sort_values(ascending=False)
    return counts.head(n_targets).index.tolist()


# ------------------------------------------------------------------ tests ---

def run_negative_control(ds: Dataset, targets: list[str], params: dict, n_jobs: int, seed: int) -> pd.DataFrame:
    """Refit on permuted targets. Real skill must collapse to ~0."""
    rng = np.random.default_rng(seed)
    tr_idx = ds.split_index("train")
    te_idx = ds.split_index("test")

    def one(target: str) -> dict | None:
        alpha, l1 = params.get(target, (DEFAULT_ALPHA, DEFAULT_L1))
        m_tr = ds.mask.loc[tr_idx, target].values
        m_te = ds.mask.loc[te_idx, target].values
        if m_tr.sum() < 30 or m_te.sum() < 5:
            return None
        y_tr = ds.Y.loc[tr_idx, target].values[m_tr]
        y_te = ds.Y.loc[te_idx, target].values[m_te]
        X_tr = ds.X.loc[tr_idx].values[m_tr]
        X_te = ds.X.loc[te_idx].values[m_te]

        real = _pearson(y_te, fit_predict(X_tr, y_tr, X_te, alpha, l1))
        # Permuting only the training labels severs the feature-target link
        # while leaving every other pipeline step identical.
        shuffled = fit_predict(X_tr, rng.permutation(y_tr), X_te, alpha, l1)
        return {
            "target": target,
            "n_train": int(m_tr.sum()),
            "n_test": int(m_te.sum()),
            "pearson_real": real,
            "pearson_shuffled": _pearson(y_te, shuffled),
        }

    rows = Parallel(n_jobs=n_jobs, verbose=1)(delayed(one)(t) for t in targets)
    return pd.DataFrame([r for r in rows if r])


def run_cv_stability(ds: Dataset, targets: list[str], params: dict, n_jobs: int, seeds: list[int]) -> pd.DataFrame:
    """Redraw the patient-level split per seed and refit.

    Subjects are keyed by (cohort, subject_id), not subject_id alone. The raw
    IDs are only unique within a cohort -- JACOBS_IBD_FAMILIES_2016 and
    KIM_ADENOMAS_2020 both number their participants A001, A002, ... and share
    90 such IDs. Splitting on the bare ID would collapse 90 pairs of unrelated
    people into single units and force both cohorts' rows onto the same side of
    every split.
    """
    manifest = pd.read_parquet(PROCESSED_DIR / "manifest.parquet")
    subjects = (
        manifest.index.get_level_values("cohort").astype(str)
        + "::"
        + manifest["subject_id"].astype(str)
    )
    subjects.index = manifest.index
    results = []

    for seed in seeds:
        rng = np.random.default_rng(seed)
        unique_subjects = np.array(sorted(subjects.unique()), dtype=object)
        rng.shuffle(unique_subjects)
        n_test = int(len(unique_subjects) * 0.15)
        test_subjects = set(unique_subjects[:n_test])
        is_test = subjects.isin(test_subjects)
        te_idx = subjects.index[is_test]
        tr_idx = subjects.index[~is_test]

        def one(target: str) -> dict | None:
            alpha, l1 = params.get(target, (DEFAULT_ALPHA, DEFAULT_L1))
            m_tr = ds.mask.loc[tr_idx, target].values
            m_te = ds.mask.loc[te_idx, target].values
            if m_tr.sum() < 30 or m_te.sum() < 5:
                return None
            pred = fit_predict(
                ds.X.loc[tr_idx].values[m_tr], ds.Y.loc[tr_idx, target].values[m_tr],
                ds.X.loc[te_idx].values[m_te], alpha, l1,
            )
            return {"seed": seed, "target": target,
                    "pearson_test": _pearson(ds.Y.loc[te_idx, target].values[m_te], pred),
                    "n_test": int(m_te.sum())}

        rows = Parallel(n_jobs=n_jobs, verbose=0)(delayed(one)(t) for t in targets)
        results.extend([r for r in rows if r])
        seed_scores = [r["pearson_test"] for r in results if r["seed"] == seed and not np.isnan(r["pearson_test"])]
        print(f"  seed {seed}: median r = {np.median(seed_scores):.3f} over {len(seed_scores)} targets")

    return pd.DataFrame(results)


def run_cross_cohort(ds: Dataset, targets: list[str], params: dict, n_jobs: int) -> pd.DataFrame:
    """Leave-one-cohort-out: the transfer question."""
    cohorts = sorted(ds.X.index.get_level_values("cohort").unique())
    cohort_of = ds.X.index.get_level_values("cohort")
    results = []

    for cohort in cohorts:
        held_out = cohort_of == cohort
        tr_idx = ds.X.index[~held_out]
        te_idx = ds.X.index[held_out]

        def one(target: str) -> dict | None:
            alpha, l1 = params.get(target, (DEFAULT_ALPHA, DEFAULT_L1))
            m_tr = ds.mask.loc[tr_idx, target].values
            m_te = ds.mask.loc[te_idx, target].values
            # A cohort that never measured this metabolite can't test it.
            if m_tr.sum() < 30 or m_te.sum() < 10:
                return None
            pred = fit_predict(
                ds.X.loc[tr_idx].values[m_tr], ds.Y.loc[tr_idx, target].values[m_tr],
                ds.X.loc[te_idx].values[m_te], alpha, l1,
            )
            return {"held_out_cohort": cohort, "target": target,
                    "pearson_test": _pearson(ds.Y.loc[te_idx, target].values[m_te], pred),
                    "n_train": int(m_tr.sum()), "n_test": int(m_te.sum())}

        rows = Parallel(n_jobs=n_jobs, verbose=0)(delayed(one)(t) for t in targets)
        rows = [r for r in rows if r]
        results.extend(rows)
        scores = [r["pearson_test"] for r in rows if not np.isnan(r["pearson_test"])]
        if scores:
            print(f"  {cohort:36s} median r = {np.median(scores):6.3f}  ({len(scores)} testable targets)")
        else:
            print(f"  {cohort:36s} no testable targets (metabolite panel doesn't overlap)")

    return pd.DataFrame(results)


def run_chemical_class() -> pd.DataFrame:
    """Break the existing baseline accuracy down by compound class. No refit needed."""
    baseline = pd.read_csv(EXPERIMENTS_DIR / "elastic_net_baseline_metrics.csv").rename(
        columns={"target": "hmdb_id"}
    )
    ref = pd.read_csv(REFERENCE_DIR / "metabolite_reference.csv")
    merged = baseline.merge(ref[["hmdb_id", "name", "class_hint"]], on="hmdb_id", how="left")

    kegg_path = REFERENCE_DIR / "metabolite_kegg.csv"
    if kegg_path.exists():
        kegg = pd.read_csv(kegg_path)
        ref_kegg = ref[["hmdb_id", "kegg_id"]].merge(
            kegg[["kegg_id", "brite_classes"]], on="kegg_id", how="left"
        )
        merged = merged.merge(ref_kegg[["hmdb_id", "brite_classes"]], on="hmdb_id", how="left")
        # KEGG BRITE covers more targets than any single cohort's class column.
        merged["chemical_class"] = (
            merged["brite_classes"].fillna("").str.split(";").str[0].str.strip()
            .replace("", np.nan).fillna(merged["class_hint"]).fillna("unclassified")
        )
    else:
        merged["chemical_class"] = merged["class_hint"].fillna("unclassified")

    grouped = (
        merged.groupby("chemical_class")["pearson_test"]
        .agg(["median", "mean", "count"])
        .sort_values("count", ascending=False)
    )
    return grouped[grouped["count"] >= 5].round(3).reset_index()


# ------------------------------------------------------------------- main ---

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--test", default="all",
                        choices=["all", "negative_control", "cv_stability", "cross_cohort", "chemical_class"])
    parser.add_argument("--n-targets", type=int, default=80, help="Targets sampled for refit-based tests")
    parser.add_argument("--n-jobs", type=int, default=3)
    parser.add_argument("--seeds", type=int, nargs="*", default=[1, 2, 3])
    args = parser.parse_args()

    EXPERIMENTS_DIR.mkdir(exist_ok=True)
    wanted = (["negative_control", "cv_stability", "cross_cohort", "chemical_class"]
              if args.test == "all" else [args.test])

    needs_data = any(t != "chemical_class" for t in wanted)
    ds = Dataset() if needs_data else None
    params = _hyperparams()
    targets = select_targets(ds, args.n_targets) if ds is not None else []
    if ds is not None:
        print(f"Sampled {len(targets)} densest-measured targets of {ds.n_targets} for refit-based tests\n")

    for test in wanted:
        print(f"=== {test} ===")
        with log_run(f"backtest_{test}", params={"n_targets": args.n_targets, "seeds": args.seeds}) as run:
            if test == "negative_control":
                df = run_negative_control(ds, targets, params, args.n_jobs, seed=0)
                real = df["pearson_real"].median()
                shuffled = df["pearson_shuffled"].median()
                run.record(median_real_r=round(float(real), 4),
                           median_shuffled_r=round(float(shuffled), 4),
                           n_targets=len(df))
                passed = abs(shuffled) < 0.1
                run.note("PASS: shuffled labels score ~0" if passed
                         else "FAIL: shuffled labels score well above 0 -- investigate leakage")
                print(f"  real median r     = {real:.4f}")
                print(f"  shuffled median r = {shuffled:.4f}")
                print(f"  -> {'PASS' if passed else 'FAIL -- possible leakage'}")

            elif test == "cv_stability":
                df = run_cv_stability(ds, targets, params, args.n_jobs, args.seeds)
                per_seed = df.groupby("seed")["pearson_test"].median()
                run.record(median_r_per_seed=per_seed.round(4).to_dict(),
                           spread=round(float(per_seed.max() - per_seed.min()), 4))
                print(f"  spread across seeds: {per_seed.max() - per_seed.min():.4f}")

            elif test == "cross_cohort":
                df = run_cross_cohort(ds, targets, params, args.n_jobs)
                per_cohort = df.groupby("held_out_cohort")["pearson_test"].median()
                within = pd.read_csv(EXPERIMENTS_DIR / "elastic_net_baseline_metrics.csv")["pearson_test"].median()
                run.record(median_cross_cohort_r=round(float(df["pearson_test"].median()), 4),
                           median_within_cohort_r=round(float(within), 4),
                           n_cohorts_testable=int(per_cohort.notna().sum()))
                print(f"\n  cross-cohort median r  = {df['pearson_test'].median():.4f}")
                print(f"  within-cohort median r = {within:.4f}  (the README number, for contrast)")

            elif test == "chemical_class":
                df = run_chemical_class()
                run.record(n_classes=len(df))
                print(df.head(18).to_string(index=False))

            out = EXPERIMENTS_DIR / f"backtest_{test}.csv"
            df.to_csv(out, index=False)
            run.artifact(out)
            print(f"  -> {out}\n")


if __name__ == "__main__":
    main()
