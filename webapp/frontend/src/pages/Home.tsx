import { useEffect, useState } from 'react'
import { api, type ModelSummary } from '../api'
import { Disclaimer, Eyebrow, Stat } from '../components/primitives'

/**
 * The explainer.
 *
 * Written for a reader who does not know what a metabolite is. The structure
 * is deliberately a chain — what lives there, what it makes, why anyone would
 * predict it, how well that works — because the product is that chain, and a
 * reader who follows it can then use the simulator without further explanation.
 *
 * Numbers come from the API, which generates them from the actual experiment
 * outputs. Nothing here is a hand-typed performance claim that could drift away
 * from what the model really does.
 */
export function Home({ onNavigate }: { onNavigate: (page: string) => void }) {
  const [summary, setSummary] = useState<ModelSummary | null>(null)

  useEffect(() => {
    api.modelInfo().then(setSummary).catch(() => setSummary(null))
  }, [])

  return (
    <>
      <section className="section">
        <div className="wrap">
          <Eyebrow>Gut microbiome → metabolome</Eyebrow>
          <h1 style={{ maxWidth: '17ch' }}>
            Your gut is a chemical factory. This predicts what it makes.
          </h1>
          <p className="lede" style={{ marginTop: 'var(--space-5)' }}>
            Roughly thirty trillion bacteria live in your large intestine. They are not
            passengers — they digest the fibre you cannot, and in doing so they manufacture
            hundreds of chemicals that enter your bloodstream and reach your organs. This model
            predicts which chemicals a given community of bacteria will produce, from the
            bacteria alone.
          </p>

          <div className="row" style={{ marginTop: 'var(--space-6)', flexWrap: 'wrap' }}>
            <button className="btn" onClick={() => onNavigate('simulator')}>
              Try the simulator
            </button>
            <button className="btn btn--ghost" onClick={() => onNavigate('model')}>
              See how the model works
            </button>
          </div>
        </div>
      </section>

      <section className="section">
        <div className="wrap">
          <h2>Start from zero</h2>
          <div className="prose" style={{ marginTop: 'var(--space-4)' }}>
            <p>
              <strong>Microbiome</strong> — the community of bacteria living in your gut. Everyone
              has one. They differ from person to person more than human genomes do: two people
              might share only a third of the same bacterial species.
            </p>
            <p>
              <strong>Metabolite</strong> — a small molecule produced by metabolism. When gut
              bacteria break down food, the leftovers are metabolites. Some are waste. Some are
              fuel for your own cells. Some act as signals that influence your immune system,
              your metabolism, and the nerves lining your gut.
            </p>
            <p>
              <strong>Metabolome</strong> — all the metabolites present, taken together. It is the
              chemical output of the whole system.
            </p>
            <p>
              A concrete example: when the bacterium <em>Faecalibacterium</em> ferments fibre, it
              produces <strong>butyrate</strong>. The cells lining your colon take most of their
              energy directly from butyrate rather than from blood sugar. Fewer of those bacteria
              generally means less butyrate, and low butyrate is one of the most consistently
              reported findings in inflammatory bowel disease.
            </p>
          </div>
        </div>
      </section>

      <section className="section">
        <div className="wrap">
          <h2>Why predict it instead of measuring it</h2>
          <div className="prose" style={{ marginTop: 'var(--space-4)' }}>
            <p>
              Both things can be measured directly. Sequencing tells you which bacteria are
              present; mass spectrometry tells you which chemicals are present. The problem is
              that sequencing is cheap and routine, while metabolomics is expensive, slow, and
              run by relatively few labs.
            </p>
            <p>
              So the great majority of microbiome data in the world has the bacteria and not the
              chemistry. If the chemistry can be predicted from the bacteria with known accuracy,
              every one of those existing datasets gains a chemical readout it never paid for —
              and that is the point of this project.
            </p>
          </div>

          <div className="flow" style={{ marginTop: 'var(--space-6)' }}>
            <div className="flow__step">
              <div className="flow__index">01</div>
              <div className="flow__title">Bacteria</div>
              <p className="flow__body">
                A stool sample is sequenced, giving the relative abundance of each bacterial
                genus present.
              </p>
            </div>
            <div className="flow__step">
              <div className="flow__index">02</div>
              <div className="flow__title">Model</div>
              <p className="flow__body">
                A separate regression model per metabolite weighs each bacterium's contribution.
                The weights are readable, so every prediction can be traced back to specific
                microbes.
              </p>
            </div>
            <div className="flow__step">
              <div className="flow__index">03</div>
              <div className="flow__title">Chemistry</div>
              <p className="flow__body">
                Out comes a predicted level for each of {summary?.dataset.n_metabolite_targets ?? '1,098'}{' '}
                metabolites, each with the accuracy it achieved on patients the model never saw.
              </p>
            </div>
            <div className="flow__step">
              <div className="flow__index">04</div>
              <div className="flow__title">Context</div>
              <p className="flow__body">
                Where taxa deviate from a healthy reference, published studies reporting the same
                shift in a condition are surfaced — with citations, not conclusions.
              </p>
            </div>
          </div>
        </div>
      </section>

      {summary && (
        <section className="section">
          <div className="wrap">
            <h2>What it was built on</h2>
            <p className="muted" style={{ marginTop: 'var(--space-3)', maxWidth: '62ch' }}>
              Fourteen independent published cohorts where the same stool samples were both
              sequenced and run through mass spectrometry — the rare case where both halves of
              the answer exist for the same person.
            </p>
            <div className="stats" style={{ marginTop: 'var(--space-5)' }}>
              <Stat value={summary.dataset.n_samples.toLocaleString()} label="paired samples" />
              <Stat value={summary.dataset.n_subjects.toLocaleString()} label="individual people" />
              <Stat value={String(summary.dataset.n_cohorts)} label="independent studies" />
              <Stat
                value={summary.dataset.n_metabolite_targets.toLocaleString()}
                label="metabolites predicted"
              />
              <Stat
                value={summary.metabolite_model.median_test_pearson_r.toFixed(2)}
                label="median accuracy (r)"
              />
            </div>

            <div className="note" style={{ marginTop: 'var(--space-5)', maxWidth: '64ch' }}>
              That median of{' '}
              <span className="num">
                r = {summary.metabolite_model.median_test_pearson_r.toFixed(2)}
              </span>{' '}
              is a real but moderate correlation — worth being plain about.{' '}
              <span className="num">{summary.metabolite_model['n_targets_r_above_0.5']}</span>{' '}
              metabolites are predicted well (r ≥ 0.5), while{' '}
              <span className="num">{summary.metabolite_model['n_targets_r_below_0.2']}</span>{' '}
              are predicted poorly (r &lt; 0.2). The interface marks which is which on every
              single value rather than presenting them as equivalent.
            </div>
          </div>
        </section>
      )}

      <section className="section">
        <div className="wrap wrap--narrow">
          <Disclaimer />
        </div>
      </section>
    </>
  )
}
