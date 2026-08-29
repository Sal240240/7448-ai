# 7448 AI — Gut Microbiome → Metabolome

Predicting a gut metabolomic profile from sequencing data (taxonomic +
functional composition), without running mass spectrometry. Full technical
rationale, prior-art positioning, and phased roadmap are in the project spec
(not checked into this repo — see your own copy).

This repo currently covers **Phase 0: data acquisition and standardization**.

**Current dataset:** 2,900 paired samples, 1,759 unique subjects, across all
14 Borenstein-collection cohorts. 12,263 genus-level taxonomic features;
1,384 unique HMDB-annotated metabolite targets. Patient-level 70/15/15
train/val/test split already computed.

## Status

| Step | Status |
|---|---|
| Pull Borenstein Lab curated collection (14 cohorts) | Done — `scripts/fetch_borenstein.py` |
| Standardize taxonomy (arcsine-sqrt transform, cross-cohort column union) | Done — `scripts/standardize_taxonomy.py` |
| Map metabolites to HMDB IDs, log-transform | Done — `scripts/standardize_metabolites.py` |
| Join into modeling-ready tables + subject/cohort manifest | Done — `scripts/build_dataset.py` |
| Functional (gene/pathway) profiling via HUMAnN3 | **Not done** — see below |
| HMP2 / PRISM / PROTECT as separate sources | **Not needed** — already included in the Borenstein collection (`iHMP_IBDMDB_2019`, `FRANZOSA_IBD_2019`) |
| Paired Omics Data Platform (PoDP) expansion | Deferred to Phase 3 per spec |

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

## Next steps (Phase 1)

1. Patient-level train/val/test split script keyed on `manifest.subject_id`.
2. MelonnPan-style elastic net baseline (`scikit-learn`) on `features.parquet` → `targets.parquet`.
3. MLP baseline (spec section 5.1, PyTorch) — must beat the elastic net on
   held-out data before any architecture upgrades are worth pursuing.
4. Decide on the HUMAnN3 functional-profiling investment before or after
   the first baseline, depending on how far taxonomy-only gets you.
