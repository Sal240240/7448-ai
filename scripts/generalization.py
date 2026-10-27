"""
Phase 3 step 1: attacking the cross-cohort generalization gap.

Phase 2's verdict was that this model does not transfer. Train on 13 cohorts,
predict the 14th, and the median Pearson r is 0.002 -- against 0.458 on
held-out patients from cohorts the model trained on. Nothing downstream
(clinical use, a second opinion on a lab result, anything involving a sample
from a lab the model has not seen) is defensible until that changes, so this
script is about one question: is the gap fixable, and by what.

The working hypothesis is units, not biology. Each cohort ran its own
untargeted metabolomics panel, and the abundances are in platform-specific
arbitrary units. Pooling 13 of them and fitting log-abundance means most of
the variance the model sees is *between* cohorts -- which platform produced
the row -- rather than within-subject biological variation. A model that fits
between-cohort offsets has learned something real about the training set and
nothing that survives contact with a 15th lab.

If that is right, normalizing each metabolite within its own cohort before
training should recover transferable signal, because it removes the
platform-specific location and scale and leaves the relative ordering of
subjects, which is the part that should generalize.

Four experiments, in the order they answer the question:

  loco_variants   Leave-one-cohort-out under an escalating ladder of
                  normalizations (below). The head-to-head that says whether
                  the hypothesis holds. Run this first.

  transfer_matrix Train on one cohort, test on one other, for all 182 ordered
                  pairs. Phase 2 only measured 13-vs-1, which averages over
                  the possibility that transfer works fine between similar
                  cohorts and fails between a preterm infant gut and an ESRD
                  adult. A pooled median of zero is consistent with both
                  "nothing transfers" and "some things transfer and the
                  average hides it," and those imply very different projects.

  n_cohorts       Transfer as a function of how many cohorts are in the
                  training set (k = 1, 2, 4, 8, 13). The slope decides
                  whether acquiring more cohorts is the fix. If transfer is
                  flat in k, more data of the same kind will not help and the
                  problem is representational.

  regularization  Transfer across an alpha sweep. A model that memorizes
                  cohort-specific structure should transfer better when
                  regularized harder, and the within-cohort-optimal alpha is
                  probably too permissive for transfer.

Metric: Spearman rho on the held-out cohort's raw values, not Pearson. Every
intervention here is a monotone per-cohort transform of the targets, and
Spearman is invariant to those, so the comparison measures the intervention
rather than the metric. Pearson is reported alongside for continuity with the
Phase 1/2 numbers.

One distinction the output keeps separate, because it decides what the product
can do: feature normalization computed within each cohort *including the test
cohort* is transductive -- it needs a batch of samples from the new lab to
normalize against. Feature normalization from training statistics alone works
on a single incoming sample. If only the transductive variant transfers, then
single-sample inference from an unseen lab is not supported, and that is a
constraint on every clinical use case, not a footnote.

Output:
    experiments/generalization_<experiment>.csv
    experiments/run_log.jsonl  (appended)

Usage:
    python scripts/generalization.py --experiment loco_variants
    python scripts/generalization.py --experiment all --n-targets 40
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import norm, rankdata, spearmanr
from sklearn.linear_model import ElasticNet

from _paths import ROOT

sys.path.insert(0, str(ROOT / "src"))
from data import Dataset  # noqa: E402
from runlog import log_run  # noqa: E402

EXPERIMENTS_DIR = ROOT / "experiments"

DEFAULT_ALPHA = 0.01
DEFAULT_L1 = 0.5

# (normalize targets within cohort, feature mode). Feature modes:
#   "raw"    leave features alone
#   "z"      z-score with training-cohort statistics only (single-sample safe)
#   "rank"   inverse-normal within each cohort separately (needs a batch from
#            the new lab -- transductive)
VARIANTS: dict[str, tuple[bool, str]] = {
    "raw": (False, "raw"),
    "target_rank": (True, "raw"),
    "target_rank_feat_z": (True, "z"),
    "target_rank_feat_rank": (True, "rank"),
    "feat_rank_only": (False, "rank"),
}

MIN_TRAIN = 30
MIN_TEST = 10


@dataclass(frozen=True)
class Panel:
    """The dataset narrowed to the targets under test, with transforms precomputed.

    Both normalizations are computed once, per cohort, rather than per fold.
    That is only valid because every fold in this script splits on whole
    cohorts: a per-cohort transform of a cohort's rows is the same whichever
    side of the split that cohort lands on, so precomputing leaks nothing.
    A fold that cut *within* a cohort would break that and would have to
    recompute from its own training rows.
    """

    X_raw: pd.DataFrame
    X_rank: pd.DataFrame
    Y_raw: pd.DataFrame
    Y_rank: pd.DataFrame
    mask: pd.DataFrame
    targets: tuple[str, ...]

    @property
    def cohort_of(self) -> np.ndarray:
        return self.X_raw.index.get_level_values("cohort").to_numpy()

    @property
    def cohorts(self) -> list[str]:
        return sorted(set(self.cohort_of))

    def features(self, feature_mode: str) -> pd.DataFrame:
        return self.X_rank if feature_mode == "rank" else self.X_raw

    def train_labels(self, rank_targets: bool) -> pd.DataFrame:
        return self.Y_rank if rank_targets else self.Y_raw


# -------------------------------------------------------------- transforms ---

def inverse_normal(values: np.ndarray) -> np.ndarray:
    """Rank-based inverse normal transform. Ties share a rank, so the all-zero
    columns common in sparse taxonomy tables collapse to one value rather than
    acquiring a spurious ordering."""
    n = len(values)
    if n == 0:
        return values
    ranks = rankdata(values)
    return norm.ppf((ranks - 0.5) / n)


def rank_targets_per_cohort(Y: pd.DataFrame, mask: pd.DataFrame) -> pd.DataFrame:
    """Inverse-normal each (cohort, target) over that cohort's measured rows.

    Unmeasured entries stay NaN: a cohort that never ran a metabolite on its
    panel carries no information about it, and ranking it against the measured
    values would invent some.
    """
    values = Y.to_numpy(dtype=np.float64)
    measured = mask.to_numpy(dtype=bool)
    out = np.full(values.shape, np.nan)
    cohorts = Y.index.get_level_values("cohort").to_numpy()

    for cohort in pd.unique(cohorts):
        in_cohort = cohorts == cohort
        for j in range(values.shape[1]):
            rows = in_cohort & measured[:, j]
            if rows.sum() < 3:
                continue
            out[rows, j] = inverse_normal(values[rows, j])
    return pd.DataFrame(out, index=Y.index, columns=Y.columns)


def rank_features_per_cohort(X: pd.DataFrame) -> pd.DataFrame:
    """Inverse-normal each (cohort, feature) across that cohort's samples.

    Transductive when the test cohort is included -- see module docstring.
    """
    values = X.to_numpy(dtype=np.float64)
    out = np.empty_like(values)
    cohorts = X.index.get_level_values("cohort").to_numpy()

    for cohort in pd.unique(cohorts):
        rows = cohorts == cohort
        out[rows] = np.apply_along_axis(inverse_normal, 0, values[rows])
    return pd.DataFrame(out, index=X.index, columns=X.columns)


def zscore_from_train(X_train: np.ndarray, X_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Standardize both sides using training statistics only."""
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    std[std == 0] = 1.0
    return (X_train - mean) / std, (X_test - mean) / std


# ------------------------------------------------------------------ fitting ---

def _scores(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float]:
    """(spearman, pearson), NaN where undefined."""
    if len(y_true) < 3 or np.std(y_true) == 0 or np.std(y_pred) == 0:
        return np.nan, np.nan
    return float(spearmanr(y_true, y_pred).statistic), float(np.corrcoef(y_true, y_pred)[0, 1])


def fit_eval_target(
    target: str,
    X: pd.DataFrame,
    Y_train_source: pd.DataFrame,
    Y_eval: pd.DataFrame,
    mask: pd.DataFrame,
    tr_idx: pd.Index,
    te_idx: pd.Index,
    feature_mode: str,
    alpha: float,
    l1_ratio: float,
) -> dict | None:
    """Fit one metabolite on tr_idx, score on te_idx against raw held-out values.

    Y_train_source supplies training labels (possibly normalized); Y_eval always
    supplies the evaluation truth, so no transform can flatter the metric.
    """
    m_tr = mask.loc[tr_idx, target].to_numpy(dtype=bool)
    m_te = mask.loc[te_idx, target].to_numpy(dtype=bool)
    if m_tr.sum() < MIN_TRAIN or m_te.sum() < MIN_TEST:
        return None

    y_tr = Y_train_source.loc[tr_idx, target].to_numpy(dtype=np.float64)[m_tr]
    finite = np.isfinite(y_tr)
    if finite.sum() < MIN_TRAIN:
        return None

    X_tr = X.loc[tr_idx].to_numpy(dtype=np.float64)[m_tr][finite]
    y_tr = y_tr[finite]
    X_te = X.loc[te_idx].to_numpy(dtype=np.float64)[m_te]
    y_te = Y_eval.loc[te_idx, target].to_numpy(dtype=np.float64)[m_te]

    if feature_mode == "z":
        X_tr, X_te = zscore_from_train(X_tr, X_te)

    model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=5000)
    model.fit(X_tr, y_tr)
    rho, r = _scores(y_te, model.predict(X_te))
    return {
        "target": target,
        "spearman": rho,
        "pearson": r,
        "n_train": int(len(y_tr)),
        "n_test": int(m_te.sum()),
        "n_nonzero_coef": int((model.coef_ != 0).sum()),
    }


def _fit_fold(
    panel: Panel, tr_idx: pd.Index, te_idx: pd.Index, variant: str, alpha: float, n_jobs: int
) -> list[dict]:
    rank_targets, feature_mode = VARIANTS[variant]
    X = panel.features(feature_mode)
    Y_src = panel.train_labels(rank_targets)
    # "rank" features are already normalized per cohort, so no train-stat z-scoring on top.
    fit_mode = "raw" if feature_mode == "rank" else feature_mode
    results = Parallel(n_jobs=n_jobs, verbose=0)(
        delayed(fit_eval_target)(
            t, X, Y_src, panel.Y_raw, panel.mask, tr_idx, te_idx, fit_mode, alpha, DEFAULT_L1
        )
        for t in panel.targets
    )
    return [r for r in results if r]


def loco_folds(panel: Panel) -> list[tuple[str, pd.Index, pd.Index]]:
    cohort_of = panel.cohort_of
    index = panel.X_raw.index
    return [(c, index[cohort_of != c], index[cohort_of == c]) for c in panel.cohorts]


def _median_rho(rows: list[dict]) -> float:
    scores = [r["spearman"] for r in rows if np.isfinite(r["spearman"])]
    return float(np.median(scores)) if scores else np.nan


# -------------------------------------------------------------- experiments ---

def run_loco_variants(panel: Panel, n_jobs: int, variants: list[str]) -> pd.DataFrame:
    """The head-to-head: each rung of the normalization ladder, same folds, same targets."""
    rows: list[dict] = []
    for variant in variants:
        variant_rows: list[dict] = []
        for cohort, tr_idx, te_idx in loco_folds(panel):
            variant_rows += [
                {"variant": variant, "held_out_cohort": cohort, **r}
                for r in _fit_fold(panel, tr_idx, te_idx, variant, DEFAULT_ALPHA, n_jobs)
            ]
        rows += variant_rows
        print(f"  {variant:24s} median rho = {_median_rho(variant_rows):6.3f}  "
              f"({len(variant_rows)} cohort-target fits)")
    return pd.DataFrame(rows)


def run_transfer_matrix(panel: Panel, n_jobs: int, variant: str) -> pd.DataFrame:
    """All ordered single-cohort pairs: does transfer work between similar cohorts?"""
    index = panel.X_raw.index
    cohort_of = panel.cohort_of
    rows: list[dict] = []

    for train_cohort in panel.cohorts:
        tr_idx = index[cohort_of == train_cohort]
        for test_cohort in panel.cohorts:
            if test_cohort == train_cohort:
                continue
            pair = _fit_fold(panel, tr_idx, index[cohort_of == test_cohort], variant, DEFAULT_ALPHA, n_jobs)
            if not pair:
                continue
            rows += [{"train_cohort": train_cohort, "test_cohort": test_cohort, **r} for r in pair]
            print(f"  {train_cohort[:30]:30s} -> {test_cohort[:30]:30s} "
                  f"rho = {_median_rho(pair):6.3f}  (n={len(pair)})")
    return pd.DataFrame(rows)


def run_n_cohorts(
    panel: Panel, n_jobs: int, variant: str, k_values: list[int], n_draws: int, seed: int
) -> pd.DataFrame:
    """Transfer vs. number of training cohorts -- the 'will more data fix it' curve."""
    rng = np.random.default_rng(seed)
    index = panel.X_raw.index
    cohort_of = panel.cohort_of
    rows: list[dict] = []

    for held_out in panel.cohorts:
        pool = [c for c in panel.cohorts if c != held_out]
        te_idx = index[cohort_of == held_out]
        for k in k_values:
            if k > len(pool):
                continue
            # k == len(pool) has only one possible draw.
            for draw in range(1 if k == len(pool) else n_draws):
                chosen = pool if k == len(pool) else list(rng.choice(pool, size=k, replace=False))
                tr_idx = index[np.isin(cohort_of, chosen)]
                rows += [
                    {"held_out_cohort": held_out, "k_train_cohorts": k, "draw": draw, **r}
                    for r in _fit_fold(panel, tr_idx, te_idx, variant, DEFAULT_ALPHA, n_jobs)
                ]
        print(f"  {held_out:36s} done")
    return pd.DataFrame(rows)


def run_regularization(panel: Panel, n_jobs: int, variant: str, alphas: list[float]) -> pd.DataFrame:
    """Does transfer improve when the model is not allowed to memorize?"""
    rows: list[dict] = []
    for alpha in alphas:
        alpha_rows: list[dict] = []
        for cohort, tr_idx, te_idx in loco_folds(panel):
            alpha_rows += [
                {"alpha": alpha, "held_out_cohort": cohort, **r}
                for r in _fit_fold(panel, tr_idx, te_idx, variant, alpha, n_jobs)
            ]
        rows += alpha_rows
        sparsity = np.median([r["n_nonzero_coef"] for r in alpha_rows]) if alpha_rows else np.nan
        print(f"  alpha {alpha:<8g} median rho = {_median_rho(alpha_rows):6.3f}  "
              f"median nonzero coefs = {sparsity:.0f}")
    return pd.DataFrame(rows)


# --------------------------------------------------------------------- main ---

def select_targets(ds: Dataset, n_targets: int, min_cohorts: int = 4) -> list[str]:
    """Densest-measured targets measured by at least `min_cohorts` cohorts.

    Cross-cohort work needs targets several cohorts actually ran; a metabolite
    on one panel only cannot be transferred anywhere.
    """
    cohort_of = ds.mask.index.get_level_values("cohort")
    per_cohort = ds.mask.groupby(cohort_of).sum()
    n_measuring = (per_cohort >= MIN_TEST).sum()
    eligible = n_measuring[n_measuring >= min_cohorts].index
    return ds.mask[eligible].sum().sort_values(ascending=False).head(n_targets).index.tolist()


def build_panel(ds: Dataset, targets: list[str], need_feature_rank: bool) -> Panel:
    Y = ds.Y[targets]
    mask = ds.mask[targets]
    print("  normalizing targets per cohort...")
    Y_rank = rank_targets_per_cohort(Y, mask)
    if need_feature_rank:
        print("  normalizing features per cohort...")
        X_rank = rank_features_per_cohort(ds.X)
    else:
        X_rank = ds.X
    return Panel(X_raw=ds.X, X_rank=X_rank, Y_raw=Y, Y_rank=Y_rank, mask=mask, targets=tuple(targets))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--experiment", default="loco_variants",
                        choices=["all", "loco_variants", "transfer_matrix", "n_cohorts", "regularization"])
    parser.add_argument("--n-targets", type=int, default=30)
    parser.add_argument("--n-jobs", type=int, default=3)
    parser.add_argument("--variants", nargs="*", default=list(VARIANTS), choices=list(VARIANTS))
    parser.add_argument("--variant", default="target_rank",
                        help="Normalization used by transfer_matrix/n_cohorts/regularization")
    parser.add_argument("--alphas", type=float, nargs="*", default=[0.001, 0.01, 0.05, 0.2, 1.0])
    parser.add_argument("--k-values", type=int, nargs="*", default=[1, 2, 4, 8, 13])
    parser.add_argument("--n-draws", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    EXPERIMENTS_DIR.mkdir(exist_ok=True)
    wanted = (["loco_variants", "transfer_matrix", "n_cohorts", "regularization"]
              if args.experiment == "all" else [args.experiment])

    ds = Dataset()
    targets = select_targets(ds, args.n_targets)
    print(f"{ds.n_features} features, {len(targets)} of {ds.n_targets} targets selected "
          f"(densest, measured in >= 4 cohorts)")

    uses = set(args.variants) if "loco_variants" in wanted else set()
    uses.add(args.variant)
    need_feature_rank = any(VARIANTS[v][1] == "rank" for v in uses)
    panel = build_panel(ds, targets, need_feature_rank)
    print()

    for experiment in wanted:
        print(f"=== {experiment} ===")
        params = {"n_targets": len(targets), "variant": args.variant, "alpha": DEFAULT_ALPHA}
        with log_run(f"generalization_{experiment}", params=params) as run:
            if experiment == "loco_variants":
                df = run_loco_variants(panel, args.n_jobs, args.variants)
                summary = df.groupby("variant")["spearman"].median().sort_values(ascending=False)
                baseline = float(summary.get("raw", np.nan))
                run.record(median_rho_by_variant=summary.round(4).to_dict(),
                           best_variant=str(summary.index[0]),
                           best_rho=round(float(summary.iloc[0]), 4),
                           baseline_raw_rho=round(baseline, 4))
                run.note(f"best = {summary.index[0]} at rho {summary.iloc[0]:.3f} vs raw {baseline:.3f}")

            elif experiment == "transfer_matrix":
                df = run_transfer_matrix(panel, args.n_jobs, args.variant)
                pair_medians = df.groupby(["train_cohort", "test_cohort"])["spearman"].median()
                run.record(n_pairs=int(len(pair_medians)),
                           median_pair_rho=round(float(pair_medians.median()), 4),
                           frac_pairs_above_0_2=round(float((pair_medians > 0.2).mean()), 4),
                           best_pairs={f"{a} -> {b}": round(float(v), 3)
                                       for (a, b), v in pair_medians.nlargest(5).items()})

            elif experiment == "n_cohorts":
                df = run_n_cohorts(panel, args.n_jobs, args.variant,
                                   args.k_values, args.n_draws, args.seed)
                curve = df.groupby("k_train_cohorts")["spearman"].median()
                run.record(median_rho_by_k=curve.round(4).to_dict())
                print("\n  transfer vs training cohorts:")
                for k, rho in curve.items():
                    print(f"    k = {k:2d}  median rho = {rho:6.3f}")

            elif experiment == "regularization":
                df = run_regularization(panel, args.n_jobs, args.variant, args.alphas)
                curve = df.groupby("alpha")["spearman"].median()
                run.record(median_rho_by_alpha=curve.round(4).to_dict(),
                           best_alpha=float(curve.idxmax()))

            out = EXPERIMENTS_DIR / f"generalization_{experiment}.csv"
            df.to_csv(out, index=False)
            run.artifact(out)
            print(f"  -> {out}\n")


if __name__ == "__main__":
    main()
