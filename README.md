# 7448 AI — Gut Microbiome → Metabolome

Predicting a gut metabolomic profile from sequencing data (taxonomic +
functional composition), without running mass spectrometry. Full technical
rationale, prior-art positioning, and phased roadmap are in the project spec
(not checked into this repo — see your own copy).

This repo covers **Phase 0 (data)**, **Phase 1 (baseline models)**, and
**Phase 2 (outcome layer, rigorous backtesting, and a public-facing webapp)**.

> ## Read this before quoting any accuracy number
>
> The headline result — median test Pearson **r = 0.458** — is measured on
> held-out *patients* from the same 14 cohorts the model trained on. Two
> Phase 2 backtests qualify it sharply:
>
> - **Cross-cohort transfer is nil.** Train on 13 cohorts, predict the 14th, for
>   every cohort: median **r = 0.002**. Not degraded — absent. Pearson r is
>   scale-invariant, so this is not an artefact of differing mass-spec units.
>   The model does not currently generalise to a laboratory it has not seen.
> - **Apparent disease-classification skill is mostly batch effect.** A pooled
>   condition classifier scores median **AUROC 0.958**. A classifier given
>   *only which cohort a sample came from* — no biology at all — scores
>   **0.915**. Restricted to case-vs-control within a single cohort, where batch
>   is constant, the honest figure is **0.643**, and only 6 of 17
>   condition×cohort models clear chance at a 95% CI.
>
> The pipeline does pass its leakage check (shuffled labels score r = −0.002
> against 0.599 on real data), so the within-cohort signal is real. It is the
> *generalisation* that is unproven. Closing that gap is the project's central
> open problem, and the webapp states it on the "Inside the model" page rather
> than burying it.

**Current dataset:** 2,900 paired samples, 1,759 unique subjects, across all
14 Borenstein-collection cohorts. 12,263 genus-level taxonomic features (5,038
after a low-prevalence filter); 1,384 unique HMDB-annotated metabolite
targets. Patient-level 70/15/15 train/val/test split already computed.

## Status

| Step | Status |
|---|---|
| Pull Borenstein Lab curated collection (14 cohorts) | Done — `scripts/fetch_borenstein.py` |
| Standardize taxonomy (arcsine-sqrt transform, cross-cohort column union) | Done — `scripts/standardize_taxonomy.py` |
| Map metabolites to HMDB IDs, log-transform | Done — `scripts/standardize_metabolites.py` |
| Join into modeling-ready tables + subject/cohort manifest | Done — `scripts/build_dataset.py` |
| Patient-level train/val/test split | Done — `scripts/make_splits.py` |
| Elastic net baseline (spec section 5.1 comparison point) | Done — `scripts/train_elastic_net.py` |
| MLP baseline (spec section 5.1) | Done — `scripts/train_mlp.py` — **does not yet beat the elastic net**, see below |
| Functional (gene/pathway) profiling via HUMAnN3 | **Not done** — see below |
| HMP2 / PRISM / PROTECT as separate sources | **Not needed** — already included in the Borenstein collection (`iHMP_IBDMDB_2019`, `FRANZOSA_IBD_2019`) |
| Paired Omics Data Platform (PoDP) expansion | Deferred to Phase 3 per spec |

### Phase 2

| Step | Status |
|---|---|
| Cross-cohort metabolite reference (names, KEGG IDs, classes) | Done — `scripts/build_metabolite_reference.py` |
| KEGG pathway/class enrichment (967 of 1,384 targets) | Done — `scripts/fetch_kegg.py` |
| Disbiome microbe↔disease associations (7,163 genus-disease pairs, 1,179 papers) | Done — `scripts/fetch_disbiome.py` |
| Harmonized disease labels across 14 cohorts | Done — `scripts/build_outcome_labels.py` |
| Outcome classifier with batch-effect controls | Done — `scripts/train_outcome_model.py` |
| Deployable metabolite predictor (sparse coefficients) | Done — `scripts/train_production_model.py` |
| Backtests: leakage control, cross-cohort, CV stability, per-class | Done — `scripts/backtest.py` |
| Longitudinal trajectories (331 subjects, 6 cohorts) | Done — `scripts/build_trajectories.py` |
| Webapp (FastAPI + React), hosting-ready | Done — `webapp/`, see `webapp/DEPLOYMENT.md` |
| GMrepo as an additional labeled source | **Blocked** — their API returns server-side `RuntimeError` on every endpoint (tried `getAllPhenotypes`, `countAssociatedRunsByPhenotypeMeshID`, GET and POST). Not worked around. |
| HMDB disease annotations | **Not pursued** — hmdb.ca returns 403 to scripted requests. KEGG covers the same enrichment through a documented API, so that was used instead. |

## Phase 1 results: elastic net vs. MLP

Trained and evaluated on the same patient-level splits, taxonomy-only
features (no HUMAnN3 functional features yet — see below), 1,098 metabolites
with enough measured samples to fit reliably (`scripts/compare_baselines.py`):

| Metric | Elastic net | MLP |
|---|---:|---:|
| Median test Pearson r | **0.458** | 0.375 |
| Mean test Pearson r | 0.485 | 0.460 |
| Median test MAE | 1.073 | **0.975** |
| Targets where it wins | 367/1098 (33%) | 545/1098 (50%) |

**Reading this straight, per the spec's own rule** ("if you don't beat the
2019 linear baseline, the deep model isn't earning its complexity yet"): **the
MLP has not cleared that bar.** It wins on the plurality of individual
metabolites and has a competitive mean, but the elastic net's *median*
performance — the more representative number given the spread — is still
ahead, and it trains in a fraction of the time.

This isn't a bug to fix so much as the exact risk the spec calls out in
section 9: *"public paired cohorts are small relative to the feature
dimensionality — overfitting risk is real."* ~2,000 training samples against
5,038 input features and 1,384 joint outputs is a hard regime for a 24M-parameter
network; the elastic net's per-target regularization is a better fit for this
data volume specifically. The architecture upgrades in spec section 5.2
(encoder-decoder bottleneck, multi-task learning) exist partly to address
this by sharing statistical strength across metabolites more efficiently than
a plain wide MLP — worth trying before concluding deep learning doesn't help
here. Full per-target numbers: `experiments/baseline_comparison.csv`.

## Why HUMAnN3 functional profiling isn't wired up yet

The Borenstein curated collection ships taxonomic profiles (genus/species
relative abundance) and metabolomics, but not raw reads or gene/pathway
abundance tables. Producing HUMAnN3 output means downloading raw FASTQ for
each cohort (much larger, and not uniformly available for every cohort) and
running MetaPhlAn + HUMAnN3 ourselves, which needs a Linux/WSL + conda
(bioconda) environment — those tools don't run natively on Windows. That's a
real chunk of infrastructure work, not a script tweak, so it's split out as
its own next step rather than silently skipped.

**Until then, the baseline model (spec section 5.1) trains on taxonomy alone.**
The spec notes functional features are "often more predictive than taxonomy
alone" per MelonnPan/MIMOSA2 — so this is a known gap to close before or
during Phase 1, not a permanent design choice.

## Data pipeline

```
scripts/fetch_borenstein.py        # download raw per-cohort .tsv files from GitHub
scripts/standardize_taxonomy.py    # -> data/processed/taxonomy_genus.parquet, taxonomy_species.parquet
scripts/standardize_metabolites.py # -> data/processed/metabolites.parquet, metabolites_measured.parquet
scripts/build_dataset.py           # -> data/processed/{features,targets,targets_mask,manifest}.parquet
```

Run in that order. Each script is idempotent and safe to re-run.

### Setup

```
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
python scripts/fetch_borenstein.py
python scripts/standardize_taxonomy.py
python scripts/standardize_metabolites.py
python scripts/build_dataset.py
```

### Source data

[github.com/borenstein-lab/microbiome-metabolome-curated-data](https://github.com/borenstein-lab/microbiome-metabolome-curated-data) —
14 independent paired fecal microbiome-metabolome cohorts, already cleaned
and harmonized by the Borenstein Lab. `fetch_borenstein.py` pulls only the
`.tsv` files we need (~480MB) rather than the full ~1.3GB repo (which also
ships `.RData` versions of everything).

Two important format quirks this pipeline accounts for:

- **Taxonomy resolution varies by cohort.** Only 6 of 14 cohorts have
  species-level profiles (the shotgun-metagenomic ones); all 14 have
  genus-level. `taxonomy_genus.parquet` is therefore the common feature
  space used for cross-cohort work; `taxonomy_species.parquet` is a
  finer-grained option for the subset that supports it.
- **A missing metabolite value means "not on this cohort's panel," not
  "zero."** Different cohorts ran different (and differently-sized)
  untargeted metabolomics panels. `metabolites.parquet` leaves those gaps
  as `NaN`; `metabolites_measured.parquet` is a same-shaped boolean mask
  so training can distinguish "not measured" from "measured as absent."
  Don't `fillna(0)` on the targets without checking the mask first.

### Output tables (`data/processed/`)

All indexed by `(cohort, sample_id)`.

- `features.parquet` — genus-level relative abundance, arcsine-sqrt
  transformed, unioned across cohorts (missing taxa = 0, since absence
  from a taxonomic profile legitimately means "not detected").
- `targets.parquet` — log1p metabolite abundance, columns = HMDB IDs.
- `targets_mask.parquet` — boolean, same shape as `targets.parquet`.
- `manifest.parquet` — `subject_id`, `study_group` per sample. **Use
  `subject_id` for any train/val/test split** (spec section 5.3) —
  splitting by `sample_id` leaks patient identity across the split
  wherever a subject contributed multiple samples.

## Next steps

Reordered after Phase 2. The cross-cohort result changes the priorities: there
is no point tuning architecture for another few points of within-cohort r while
out-of-cohort transfer sits at zero.

1. **Attack the generalization gap first.** Options worth trying, roughly in
   order of expected value: per-cohort target standardisation (rank-normalise
   metabolites within cohort before training, so the model learns ordering
   rather than platform-specific scale); explicit batch correction (ComBat,
   MMUPHin); domain-adversarial training; and leave-one-cohort-out model
   selection rather than random-split selection. Each is testable with the
   existing `scripts/backtest.py --test cross_cohort` harness.
2. **Functional profiling (HUMAnN3).** Still the clearest data-side upgrade —
   gene/pathway abundance is mechanistically closer to metabolite production
   than taxonomy is, and it may transfer across cohorts better precisely
   because gene content is less lab-specific than 16S composition. Needs a
   Linux/WSL + bioconda environment, so it is real infrastructure work.
3. **Architecture upgrades (spec 5.2)** — encoder-decoder bottleneck,
   multi-task learning. Worth doing *after* (1), and worth judging on
   cross-cohort transfer, not just held-out patients.
4. **More labeled cohorts.** curatedMetagenomicData is the obvious next source;
   GMrepo would have been, but their API is currently returning server errors.

### Done in Phase 2 (was items 3 and 4)

Cross-cohort generalization testing and the per-metabolite-class breakdown are
now implemented in `scripts/backtest.py`. The class breakdown uses KEGG BRITE
classification rather than `Putative.Chemical.Class`, because that column
exists in only 2 of 14 cohorts while KEGG covers 967 targets.

## Reproducing the Phase 1 baselines

```
python scripts/train_elastic_net.py --min-train-measured 100
python scripts/train_mlp.py --epochs 200 --patience 15
python scripts/compare_baselines.py
```

## Phase 2: outcome layer, backtests, and the webapp

```
python scripts/build_metabolite_reference.py   # names + KEGG IDs for all 1,384 targets
python scripts/fetch_kegg.py                   # pathway/class enrichment
python scripts/fetch_disbiome.py               # microbe-disease literature + citations
python scripts/build_outcome_labels.py         # harmonize 14 cohorts' disease labels
python scripts/train_outcome_model.py          # condition classifier + batch controls
python scripts/train_production_model.py       # the deployable sparse predictor
python scripts/build_trajectories.py           # longitudinal structure
python scripts/build_app_data.py               # everything the webapp reads
python scripts/backtest.py --test all          # the tests that qualify the numbers
```

`python scripts/status.py` prints current pipeline state — which tables exist,
what is trained, and the recent run history. Every step appends to
`experiments/run_log.jsonl`, including failures.

Then see [`webapp/DEPLOYMENT.md`](webapp/DEPLOYMENT.md) to run the app.

### Where the new data comes from

| Source | What it provides | Access |
|---|---|---|
| [Borenstein Lab collection](https://github.com/borenstein-lab/microbiome-metabolome-curated-data) | 14 paired cohorts (training data) | Public repo |
| [Disbiome](https://disbiome.ugent.be) | 10,866 curated microbe-disease observations, 1,179 papers, per-study quality flags | Open API (port 8080) |
| [KEGG](https://www.genome.jp/kegg/) | Compound pathways and BRITE classification | Documented REST API |
| [PubMed](https://pubmed.ncbi.nlm.nih.gov/) | Citation resolution | NCBI E-utilities |

All are public and were accessed through documented interfaces. No source was
scraped around a technical restriction: HMDB blocks scripted access and was
therefore not used directly (KEGG supplies equivalent enrichment), and GMrepo's
API is returning server errors, which is recorded rather than circumvented.
