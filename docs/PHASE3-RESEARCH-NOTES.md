# Phase 3 research notes — data sources, prior art, and commercial positioning

Gathered 2026-10-26. Two parallel research passes: new data sources, and clinical
use cases / monetization.

**Verification status is marked on every claim.** Anything marked UNVERIFIED was
not independently confirmed and must be checked before it goes into the README,
a grant application, or anything a reader might act on. The single most
important item here (Muller et al. 2021) was being verified when the session
ended — see the flag on it.

---

## 1. The prior-art finding that reframes the project

**VERIFIED 2026-10-27** (fetched PMC8507343 directly; an earlier session left
this flagged as unconfirmed). Muller, Algavi & Borenstein (2021),
"A meta-analysis study of the robustness and universality of gut
microbiome-metabolome associations", *Microbiome* 9:203 — the same lab whose
data this project trains on — ran essentially our cross-cohort experiment
across 10 datasets (1,733 samples). Confirmed figures: within-dataset, 418 of
1,255 metabolite models were "well-predicted" (Spearman rho > 0.3, FDR < 0.1),
29.1% on real data vs 0.3% shuffled. Across datasets, **97 metabolites robustly
well-predicted against 87.7 ± 5.5 expected by chance, P = 0.053** — the authors
describe this as only "somewhat higher than expected by chance," and it does not
clear the conventional 0.05 threshold.

One further detail, not in the original note and directly useful: their
cross-predictability analysis is **metabolite-specific**, not uniformly
negative. Beta-alanine, sebacic acid and taurine transferred well between
dataset pairs; l-tyrosine and xanthine transferred between *no* pair. That is
the published form of the result we independently reproduced in
`transfer_chemistry.py` — sebacate at rho 0.537, taurine 0.368, and the
nucleotide family (xanthine's) second-lowest at 0.117. Note they used random
forest at genus level, so the agreement is across method as well as data.

**Why this matters more than any other line in this document:** if it holds, our
median cross-cohort r = 0.002 is not a bug in our pipeline. It is a cleaner
replication of the field's own negative result — we measured leave-one-cohort-out
across all 14 cohorts with a shuffled-label control at r = -0.002, which is a
stronger design than a cross-dataset robustness count. That converts an
apparent failure into a publishable contribution.

It also caps the within-cohort claim: MelonnPan (Mallick et al. 2019,
*Nat Commun* 10:3136) is itself an elastic net, so our finding that the elastic
net beats a 24M-parameter MLP reproduces the field's standing architecture
choice rather than advancing past it. **Do not position the model as a
methodological advance.**

**Status: confirmed, safe to cite.** The figures above were checked against the
paper itself and may be quoted. The one framing correction: this is not a flat
negative result for the field, it is a *metabolite-specific* one — a minority of
compounds transfer and most do not, which is exactly what our own tier analysis
found. Describing it as "the field got chance-level transfer" overstates it;
"cross-study robustness is confined to a small, chemically coherent subset" is
accurate.

---

## 2. New data sources

Tested live 2026-10-26. VERIFIED = endpoint was fetched and returned real data.

### Tier 1 — paired microbiome + metabolome

**Metabolomics Workbench (NIH) — VERIFIED, highest value.**
Base `https://www.metabolomicsworkbench.org/rest/`. The `metstat` context:

```
https://www.metabolomicsworkbench.org/rest/metstat/;;;Human;Feces;;;
```

Returns 72 records / 70 unique study IDs. Slot order is
`ANALYSIS_TYPE;POLARITY;CHROMATOGRAPHY;SPECIES;SAMPLE SOURCE;DISEASE;KEGG_ID;REFMET_NAME`
— empty slots still need their semicolons. The only stool term in MW's
vocabulary is `Feces` (not "Stool" or "Fecal").

Download endpoints, both VERIFIED HTTP 200:
- `rest/study/study_id/ST002018/metabolites` — per-metabolite `refmet_name`/`refmet_id`
- `rest/study/study_id/ST002018/data` — ~1 MB JSON, `{"1":{metabolite_name, refmet_name, units, DATA:{sample_id: value}}}`, directly parseable
- `rest/study/analysis_id/AN000001/datatable/file` — flat file

**Critical caveat (VERIFIED by checking two studies' metadata pages): MW hosts
the metabolome only.** ST002247 and ST002018 carry zero 16S/shotgun/SRA/ENA/
PRJNA accessions — even ST002018, titled "Multi-omic analysis of the microbiome
and metabolome." Pairing requires manual per-study linkage: publication →
BioProject → SRA/ENA. **That linkage work, not the download, is the real cost.**

Highest-value adult candidates (sample counts VERIFIED via `/summary`):

| Study | Title | Samples |
|---|---|---|
| ST002247 | Microbiota and Health Study (Dhaka, Bangladesh) | 1368 (222 subjects metabolomics) |
| ST002337 | Guangzhou Nutrition and Health Study (GNHS) | 1092 |
| ST003161 | Diet-omics, SOURCE urban/rural Crohn's | 37 |
| ST002018 | Multi-omic microbiome+metabolome, healthy | 75 |
| ST002535 | Gut microbiome & cognitive impairment, aged care | 47 |
| ST002262 | Metabolomics + metagenomics, pediatric obesity | 58 |

Also triage: ST005010 (MS), ST003313 (Crohn's flare), ST002471, ST002862,
ST001729 (metformin), ST001813, ST003206/07/08 (early-life maturation).
Already in our set, skip: ST001192 (POYET), ST001000/ST000923 (iHMP IBDMDB).

Dhaka and GNHS are non-Western adult populations on different MS platforms —
exactly the transfer-test material we lack.

**MetaboLights (EBI) — VERIFIED.** Keyword search via EBI Search REST (the
MetaboLights WS has no keyword search):

```
https://www.ebi.ac.uk/ebisearch/ws/rest/metabolights?query=feces%20OR%20faeces%20OR%20stool&size=100&format=json&fields=id,name,description
```

`hitCount: 345`; `query=stool%20AND%20microbiome` gives 50. Mixes in mouse
studies — filter on organism. Public FTP is open (VERIFIED listing):

```
https://ftp.ebi.ac.uk/pub/databases/metabolights/studies/public/MTBLS13494/
```

`m_*_v2_maf.tsv` is the abundance matrix and its `database_identifier` column
**carries HMDB IDs natively — no mapping step needed**, unlike MW. Full index:
`https://www.ebi.ac.uk/metabolights/ws/studies` → 3,589 studies.

**Paired Omics Data Platform — DEAD END (VERIFIED).** Natural-products focused
(strain genomes ↔ MS spectra), not human cohorts. Latest Zenodo snapshot
(record 7497442) is 272 KB from 2023-01-01, no live REST API.

**Named large cohorts — GATED (not individually tested).** TwinsUK, FINRISK/
METSIM, Rotterdam, Lifelines-DEEP, Milieu Interieur, ZOE/PREDICT, Estonian
Microbiome Cohort, GEM, SPARC IBD, HELIUS all need data-access-committee
application or a signed DTA. Exception: GNHS fecal metabolome is already open
via MW ST002337. American Gut/Microsetta has open 16S but no matched stool
metabolome at scale.

### Tier 2 — outcome-labeled taxonomic cohorts

**curatedMetagenomicData — VERIFIED (partial non-R path).** 22,588 samples /
93 datasets, 141 metadata fields. Metadata over plain HTTPS, no R needed:

```
https://raw.githubusercontent.com/waldronlab/curatedMetagenomicData/devel/inst/extdata/metadata.csv
```

Abundance matrices have **no pure-HTTP route**. Practical bridge:
`curatedMetagenomicDataTerminal` prints TSV to stdout, so it is callable from
Python via `subprocess`. Install
`Rscript -e 'BiocManager::install("waldronlab/curatedMetagenomicDataTerminal")'`,
then `curatedMetagenomicData "AsnicarF_20.+.relative_abundance" --counts`.
Taxonomy is MetaPhlAn, genus-extractable. License not stated on the vignette —
check before redistribution.

**GMrepo — CONFIRMED STILL BROKEN (VERIFIED).** Retested; Django `RuntimeError`
on GET and POST, six weeks on. `/home` returns 200 (site up, API dead). Write
it off unless the maintainers are emailed.

**MGnify — VERIFIED.** 13 pages of fecal studies:

```
https://www.ebi.ac.uk/metagenomics/api/v1/studies?biome_name=root:Host-associated:Human:Digestive%20system:Large%20intestine:Fecal&format=json
```

No metabolomics. **MicrobiomeHD — UNVERIFIED**, not tested.

### Tier 3 — metabolite annotation

**HMDB — still blocked (VERIFIED 403).** `https://hmdb.ca/downloads` returns 403
to scripted requests; no legitimate scripted path found. Browser download is the
compliant route. The research agent attempted one browser-User-Agent request,
correctly treated that as bot-check circumvention, and stopped — the right call,
and the position this project keeps.

**Use Metabolomics Workbench instead — VERIFIED.** Largely removes the HMDB
dependency for ID and class annotation:

```
rest/compound/pubchem_cid/5997/name,hmdb_id,kegg_id,formula,exactmass
→ {"name":"Cholesterol","hmdb_id":"HMDB0000067","kegg_id":"C00187", ...}

rest/refmet/name/Cholesterol/all
→ super_class "Sterol Lipids", main_class "Sterols", sub_class "Cholesterols", ...
```

Gives HMDB↔KEGG↔PubChem↔ChEBI crosswalks plus a chemical class hierarchy for
all 1,384 targets. Free, no auth.

**VMH/AGORA — THIN.** Documents `/apiDocs`, `/allMetabolites`, SPARQL, but
homepage gives relative paths only, no concrete URLs or license beyond "for
research purposes only." **gutMGene / GMMAD2 — UNVERIFIED**, conflicting hosts,
neither reached. **MetaCyc / MASI — UNVERIFIED**, not tested.

### Realistic yield

MW (70 studies) + MetaboLights (50 stool+microbiome) plausibly yields **15–30
genuinely new paired cohorts**, several non-Western adult, on different MS
platforms — a 2–3× increase over our 14. The gating factor is not access or
licensing but the **per-study publication→SRA linkage**, since neither
repository cross-links the sequencing side.

---

## 3. Commercial positioning

All of this is downstream of two numbers: within-cohort median r = 0.458
(**R² ≈ 0.21** — a fifth of the variance, in a lab the model has seen) and
cross-cohort median r = 0.002 (**R² ≈ 0**).

### Who can actually buy today

**Academic/translational researchers — serviceable now.** Two workflows work at
r = 0.458, both *within*-cohort by construction:
- **Within-study imputation**: cohort has sequencing on 400 samples, MS on 120;
  train on the paired 120, impute the other 280. This is the most defensible
  technical claim the project owns.
- **Panel triage**: rank metabolites by predictability to decide what goes on a
  targeted panel next.

Economics that make this real (pricing VERIFIED from core-facility pages):
untargeted LC-MS runs ~$92–400/sample (Illinois $92; NIEHS $135–175; MD Anderson
$200 internal / $399 non-academic; UC Davis ~$149 at n=12 → ~$52 at n=500). 16S
is $55–75/sample, shotgun $120–285. So **MS is 1.5–5× the sequencing cost on a
cohort that already has sequencing** — that gap is the value proposition.
Barrier: academics pay in citations, not cash.

**Clinicians — not serviceable at any specialty.** Within-cohort case-vs-control
AUROC 0.643, with 6 of 17 condition×cohort models clearing chance at 95% CI, is
below every incumbent: fecal calprotectin (IBD), FIT/Cologuard (CRC). Notable:
**no FDA-cleared microbiome-profile or dysbiosis diagnostic exists in the US** —
the cleared stool tests are all infectious-pathogen panels (BioFire, Applied
BioCode, BD, Diasorin). That is a difficulty signal, not an open lane.
Nephrology (ESRD) is the most scientifically interesting niche — indoxyl sulfate
and p-cresyl sulfate are microbially derived and mechanistically tied to the
microbiome — but research-only. Neonatology has the highest unmet need and the
lowest error tolerance; it is a device from day one.

**Dietitians / functional medicine — where the money is, and where not to go
yet.** Practitioner-channel stool tests list ~AU$380–445 (GI-MAP via one
distributor); Rupa Health resells at $179–249. The practitioner monetizes
*interpretation* — the one deliverable we could ship today with zero validation.
That is the trap: FTC §5 substantiation applies to us even when a practitioner
says it to the patient, and a 2024 University of Maryland group publicly called
for federal regulation of DTC microbiome tests precisely because none have had
claims evaluated by FDA. The defensible version, post-validation, is a
**bibliographic layer with no predictive claim** — "these taxa appear in N
Disbiome-indexed papers associated with condition X." Our 7,163 genus–disease
pairs across 1,179 papers is a real asset that does not depend on the model
working.

**Pharma/biotech R&D — the first segment that pays cash.** A microbiome-
therapeutics company in Phase 1/2 has sequencing on everyone and MS on a subset
because MS at $200–400 × 300 samples is a real line item. Imputing the rest is a
within-study problem — **the only commercial framing where r = 0.458 is the
relevant number and transfer is not required.** They will not buy trial
stratification or surrogate endpoints; both need the r = 0.002 case.

**Diagnostic labs / CROs — natural channel, hardest sell.** A licensee's lab is
by definition an unseen cohort. Realistic entry is selling the report/literature
layer, or a **paid validation study** where we train on their platform — which
converts the problem into a service engagement.

### Regulatory (US; EU noted)

- **RUO** is where we are and should stay. It is a labeling-and-conduct status,
  not a loophole — marketing an RUO product for clinical use makes it an
  unapproved device regardless of label.
- **General wellness** is narrower than marketing teams assume. "Supports
  digestive wellness" may fit; "indicates IBD" does not.
- **Device/SaMD**: no predicate exists for this, so 510(k) is unlikely; De Novo
  at the moderate-risk end.
- **CDS carve-out does not apply.** FDA's final CDS guidance (Sept 2022,
  §520(o)(1)(E)) requires all four criteria. Two kill us: software must not
  analyze IVD signals (a sequencing-derived profile is exactly that), and
  patient-facing recommendation software is a device, period. **Any 7448 output
  aimed at a clinician or consumer and keyed to a condition is a device.**
- **LDT rule vacated** in its entirety on 2025-03-31 (E.D. Tex., Judge Sean D.
  Jordan) — a device is a tangible product, not a professional service. The LDT
  route is currently wider than FDA intended, but it was remanded to HHS and
  appeal/legislation is live. **Do not build a model whose only defense is that
  vacatur.**
- **EU IVDR 2017/746**: RUO is explicitly not a compliance route. Interpretation
  and reporting software is in scope, including in-house. (Class C software 2028
  / in-house 2030 transition dates are UNVERIFIED — check the official
  transition regulation.)

**Can say today:** "predicts stool metabolite profiles from taxonomic
composition, for research use"; the actual numbers including the cross-cohort
one; "For Research Use Only."

**Cannot say, any channel:** "metabolome test"; "without mass spectrometry" as a
clinical substitution claim; "detects/screens/indicates/monitors" any named
condition; **"AUROC 0.958" without the adjacent 0.915 cohort-only baseline in
the same sentence** — that is the single most dangerous sentence available to
us; "clinically validated."

**Cautionary set.** uBiome: pivoted to physician-ordered reimbursed tests; FBI
raid 2019, Chapter 7 Oct 2019, founders indicted March 2021 on 47 counts
(SEC alleging ~$60M fraud) — VERIFIED via DOJ. The operative lesson is not
"avoid DTC" but that **billing insurers for unvalidated tests is the specific
act that turns a weak product into a federal crime.** 23andMe's 2013 warning
letter is the precedent for consumer health-risk reporting without
authorization. Viome markets CancerDetect as "clinically validated" while
GenomeWeb reported the published classifier at 83% sensitivity / 97.9%
specificity against press claims of 90%/95%; its Breakthrough Device
*Designation* (2022) is a review-process designation, not an authorization.
Survivable posture: Thorne, Tiny Health (defensibly narrow infant-gut
population), Zoe — wellness framing, no diagnostic claims.

### Monetization, ranked by risk-adjusted EV

1. **Non-dilutive grants — highest EV, and the only route where our weakest
   result is the asset.** Sell the research program *including* the negative
   result. SBIR/STTR lapsed 2025-09-30, NIH closed all 23 open SBIR NOFOs
   2025-11-17 (NOT-OD-26-006), reauthorization (S. 3971) took effect
   **2026-04-13** through 2031-09-30. Targets: **NIDDK** (GI microbiology),
   **NCI** (contract topic 458, "Microbiome-Based Tests for Cancer Research,
   Diagnosis, Prognosis and/or Patient Management"), NIAID, plus R21/R01 for
   methods work. 9–15 months to revenue, no regulatory exposure. *Phase I
   figures UNVERIFIED and conflicting across secondary sources — check the NIH
   Guide directly for live NOFOs and ceilings.*
2. **Pharma/biotech within-study imputation services.** 3–9 months, no
   regulatory exposure, low transfer-dependency *if scoped to within-study*.
   WTP anchors are thin: closest public comparables are academic core rates
   (UMN $2,122/dataset statistical analysis, $4,244 custom bioinformatics,
   $12,233 expedited shotgun ≤100 samples; UCSF $191.10/hr for-profit).
   *A $25K–100K/engagement estimate is UNVERIFIED — no public comparables found.*
3. **Dataset/benchmark authority.** The LOCO harness with a shuffled-label
   control is a real contribution to a field whose own flagship meta-analysis
   got P = 0.053. Near-zero direct revenue, high strategic value, low cost.
4. **Licensing — reject the strong version, take the weak one.** Licensing the
   *predictor* means a licensee validates it, finds r ≈ 0, and we're done — or
   nobody validates and unsupported numbers reach consumers under someone
   else's brand (uBiome failure mode with our model inside it). Licensing the
   **association/report layer** has no predictive claim and no transfer
   dependency; price as software/content. The strong version becomes the best
   route in the portfolio the moment M1 clears, and not before.
5. **Open source + consulting.** Complement, not a business.
6. **Researcher API/SaaS.** Build it because pharma engagements need it, not
   because seats will sell — a competing elastic net ships free in bioBakery.
7. **Practitioner SaaS.** Defer past validation; enter with the bibliographic
   layer only if ever.
8. **DTC — don't.** Benchmarks for completeness: Viome $129–259 + $69.95/mo;
   Zoe ~$294 + $29–39/mo US (£149 + £9.99–24.99/mo UK); Tiny Health $249 kit,
   $399/yr membership; Jona $385 kit. Requires CLIA access, consumer support,
   real CAC, and wellness-adjacent claims on R² ≈ 0.21. The one route where a
   bad outcome is career-ending rather than merely unprofitable.

### The 12-month path

**Be a methods group that solves cross-cohort transfer, funded by NIH and 2–3
pharma imputation pilots. Nothing else.** Every revenue route except grants and
within-study services is gated on the same technical milestone, so treat them
as a sequence, not parallel options.

Build, in order: (1) per-cohort rank normalization judged on LOCO — cheapest
intervention, highest prior, attacks platform-scale confounding directly;
(2) batch correction (MMUPHin/ComBat), domain-adversarial training, and
critically **model selection on LOCO rather than random split** — selecting on
random split is itself a source of the gap; (3) HUMAnN3 functional profiling
(real infrastructure work — do it after the cheap tests say whether the gap is
normalization or biology); (4) **freeze the webapp** as a credibility artifact.

Three falsifiable milestones:

- **M1 (unlocks licensing/stratification).** Pre-registered LOCO: median
  Pearson r ≥ 0.20 on ≥ 200 metabolites, holding across ≥ 10 of 14 held-out
  cohorts, shuffled control still ~0. Deliberately modest — it is the threshold
  where "transfers weakly but non-trivially" becomes defensible versus 0.002.
- **M2 (unlocks pharma revenue; does NOT require M1 — do this first, fastest
  path to a dollar).** Within-cohort imputation utility: train on 30% of a
  cohort's MS-measured samples, impute the other 70%, show the imputed data
  reproduces the cohort's published differential-abundance findings at a
  concordance stated in advance (e.g. ≥ 70% directional agreement at matched
  FDR). Measurable with data already on disk.
- **M3 (credibility/grant scoring).** Submitted paper combining the LOCO
  negative result, the batch-effect decomposition (0.958 pooled / 0.915
  cohort-only / 0.643 within-cohort), and a public benchmark, with **≥ 2
  external groups submitting to the leaderboard** — the external submissions
  are the falsifiable part.

Sell to, in order: one NIH application (NIDDK or NCI); 2–3 microbiome-
therapeutics companies for M2-shaped pilots; nobody else.

**Do not:** quote AUROC 0.958 without 0.915 beside it; build DTC or practitioner
SaaS; pursue CRC screening; take equity on current numbers; tune architecture
for within-cohort gains while LOCO sits at zero; or let the association layer
drift into recommending interventions, which makes it an unapproved device.

**Positioning that is both true and sellable today:** *a reproducible benchmark
showing microbiome-to-metabolome prediction is real within a cohort (median
r = 0.458) and absent across cohorts (median r = 0.002), plus a working
imputation tool for studies with paired mass spec on a subset of samples.*
