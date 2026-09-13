import { useEffect, useState } from 'react'
import { api, type ModelSummary } from '../api'
import { Disclaimer, Eyebrow } from '../components/primitives'

const SOURCES = [
  {
    name: 'Borenstein Lab curated microbiome-metabolome collection',
    role: 'Training data — 14 cohorts where the same stool samples were both sequenced and run through mass spectrometry.',
    detail: '2,900 paired samples from 1,759 people, harmonised by the original curators.',
    url: 'https://github.com/borenstein-lab/microbiome-metabolome-curated-data',
  },
  {
    name: 'Disbiome',
    role: 'Literature associations — curated records of which microbes published studies report elevated or reduced in which condition.',
    detail: '10,866 experimental observations across 375 conditions and 1,179 publications, with methodological-quality flags per study.',
    url: 'https://disbiome.ugent.be',
  },
  {
    name: 'KEGG',
    role: 'Metabolite annotation — pathway membership and chemical classification.',
    detail: 'Reached through the documented REST API for the 967 target metabolites carrying a KEGG compound identifier.',
    url: 'https://www.genome.jp/kegg/',
  },
  {
    name: 'HMDB',
    role: 'Metabolite identifiers — the naming scheme every target in this project is keyed on.',
    detail: 'Identifiers arrive via the Borenstein annotation files. HMDB\'s own site refuses scripted requests, so it is not queried directly; KEGG supplies the enrichment instead.',
    url: 'https://hmdb.ca',
  },
  {
    name: 'PubMed / NCBI E-utilities',
    role: 'Citations — resolving the publications behind each association.',
    detail: 'Queried through the official E-utilities API within NCBI\'s documented rate limits.',
    url: 'https://pubmed.ncbi.nlm.nih.gov/',
  },
]

/**
 * Provenance, limitations, and the things that would change the conclusions.
 *
 * The limitations section is deliberately specific. "This has limitations" is
 * noise; "the disease labels are perfectly confounded with sequencing batch,
 * and here is what we did about it" is information a reader can act on.
 */
export function Methodology() {
  const [summary, setSummary] = useState<ModelSummary | null>(null)

  useEffect(() => {
    api.modelInfo().then(setSummary).catch(() => setSummary(null))
  }, [])

  return (
    <>
      <section className="section">
        <div className="wrap wrap--narrow">
          <Eyebrow>Methodology & sources</Eyebrow>
          <h1>Where all of this comes from</h1>
          <p className="lede" style={{ marginTop: 'var(--space-4)' }}>
            Every number on this site traces to a public dataset or a published paper. Nothing is
            proprietary, and nothing was scraped from a source that did not intend to be read
            programmatically.
          </p>
        </div>
      </section>

      <section className="section">
        <div className="wrap wrap--narrow">
          <h2>Data sources</h2>
          <div style={{ marginTop: 'var(--space-5)' }}>
            {SOURCES.map((source) => (
              <div key={source.name} style={{ padding: 'var(--space-4) 0', borderBottom: '1px solid var(--rule)' }}>
                <a href={source.url} target="_blank" rel="noopener noreferrer" style={{ fontWeight: 600 }}>
                  {source.name}
                </a>
                <p className="small muted" style={{ margin: '6px 0 4px' }}>
                  {source.role}
                </p>
                <p className="tiny faint" style={{ margin: 0 }}>
                  {source.detail}
                </p>
              </div>
            ))}
          </div>
        </div>
      </section>

      <section className="section">
        <div className="wrap wrap--narrow">
          <h2>How the model is evaluated</h2>
          <div className="prose" style={{ marginTop: 'var(--space-4)' }}>
            <p>
              <strong>Split by person, not by sample.</strong> Six of the fourteen cohorts sampled
              the same individual repeatedly. Splitting at the sample level would put one person's
              Monday and Thursday samples on opposite sides of the train/test boundary, and the
              model would score well by recognising the person. Every split here assigns whole
              individuals.
            </p>
            <p>
              <strong>Missing is not zero.</strong> Different cohorts ran different metabolomics
              panels, so a blank is usually "this lab did not measure that compound," not "the
              level was zero." Treating blanks as zeros would train the model to predict absence
              for anything unfashionable to measure. Each metabolite is therefore trained only on
              the samples where it was genuinely measured, tracked by an explicit mask.
            </p>
            <p>
              <strong>Accuracy travels with every prediction.</strong> Each metabolite carries the
              correlation it achieved on held-out patients. The interface shows it next to the
              value, always, because{' '}
              {summary
                ? `${summary.metabolite_model['n_targets_r_below_0.2']} of ${summary.dataset.n_metabolite_targets} `
                : 'a substantial share of '}
              metabolites are predicted poorly, and rendering those identically to the good ones
              would be the single most misleading thing this interface could do.
            </p>
          </div>
        </div>
      </section>

      <section className="section">
        <div className="wrap wrap--narrow">
          <h2>Known limitations</h2>
          <div className="prose" style={{ marginTop: 'var(--space-4)' }}>
            <p>
              <strong>Batch confounding is severe.</strong> Each cohort studied one disease on one
              platform, so disease status and laboratory are nearly inseparable. The consequence
              is quantified on the{' '}
              <em>Inside the model</em> page: a classifier using only cohort identity nearly
              matches one using the actual biology.
            </p>
            <p>
              <strong>Association is not causation, and group is not individual.</strong> A
              microbe reported reduced in a condition across a population says nothing about why,
              and nothing about any one person. Both directions of causality are plausible for
              most of these findings, and many are likely downstream of diet or medication.
            </p>
            <p>
              <strong>Composition only, no function.</strong> The model sees which bacteria are
              present, not which genes they are expressing. Two communities with identical
              membership can behave differently. Functional profiling would likely improve
              accuracy and is the clearest next step.
            </p>
            <p>
              <strong>Genus-level resolution.</strong> Only six cohorts have species-level data, so
              the common feature space stops at genus. Meaningful differences exist below that
              level — <em>E. coli</em> strains range from harmless to dangerous.
            </p>
            <p>
              <strong>Cohorts are not representative.</strong> The source studies skew toward
              particular geographies, ages, and diets. A profile from an under-represented
              population will be compared against a reference that does not describe it well.
            </p>
          </div>
        </div>
      </section>

      <section className="section">
        <div className="wrap wrap--narrow">
          <h2>Privacy</h2>
          <div className="prose" style={{ marginTop: 'var(--space-4)' }}>
            <p>
              Nothing you enter is stored. The API holds no database, sets no cookies, creates no
              account, and writes no log of submitted compositions. Each request is a pure
              function of its input and a fixed model snapshot, and the response is computed and
              discarded.
            </p>
            <p>
              The model is also fixed: it does not learn from anything visitors do here. That is
              what makes the transparency page meaningful — the model you can inspect is exactly
              the model that answered you.
            </p>
          </div>
        </div>
      </section>

      <section className="section">
        <div className="wrap wrap--narrow">
          <Disclaimer />
        </div>
      </section>
    </>
  )
}
