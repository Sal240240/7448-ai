import { useEffect, useState } from 'react'
import { api, type ModelSummary, type RunRecord } from '../api'
import { Eyebrow, Stat } from '../components/primitives'

/**
 * The transparency view.
 *
 * This page exists to make the model inspectable rather than to advertise it,
 * so it leads with the least flattering result in the project: that the
 * obvious way to build the outcome classifier produces an excellent-looking
 * number which is almost entirely an artefact. A reader who understands that
 * comparison understands what this tool is and is not.
 */
export function InsideModel() {
  const [summary, setSummary] = useState<ModelSummary | null>(null)
  const [runs, setRuns] = useState<RunRecord[]>([])
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    api.modelInfo().then(setSummary).catch((e: Error) => setError(e.message))
    api.runs(30).then(setRuns).catch(() => setRuns([]))
  }, [])

  if (error) {
    return (
      <section className="section">
        <div className="wrap">
          <div className="error">Could not load model information: {error}</div>
        </div>
      </section>
    )
  }

  const outcome = summary?.outcome_model

  return (
    <>
      <section className="section">
        <div className="wrap">
          <Eyebrow>Inside the model</Eyebrow>
          <h1 style={{ maxWidth: '19ch' }}>What it does, and where it fails</h1>
          <p className="lede" style={{ marginTop: 'var(--space-4)' }}>
            Every claim on this site is generated from the experiment outputs in the repository,
            not written by hand. If a number here looks unimpressive, that is because it is the
            number the model actually produced.
          </p>
        </div>
      </section>

      {outcome && (
        <section className="section">
          <div className="wrap">
            <h2>The result that shaped this whole tool</h2>
            <div className="prose" style={{ marginTop: 'var(--space-4)' }}>
              <p>
                The obvious way to predict "which condition does this profile resemble" is to
                pool all fourteen cohorts and train one classifier. Done that way, it reaches a
                median{' '}
                <span className="num">AUROC {outcome.median_auroc_pooled?.toFixed(3)}</span> —
                which would be an excellent result, and is the kind of number that gets
                published and productised.
              </p>
              <p>
                It is almost entirely an artefact. Each cohort studied one disease, on one
                sequencing platform, in one lab. So "has Crohn's disease" is nearly perfectly
                correlated with "was processed by the lab that studied Crohn's disease," and a
                model can score brilliantly by recognising the lab's technical signature instead
                of anything biological.
              </p>
              <p>
                The control that proves it: train a classifier on{' '}
                <strong>nothing but which cohort a sample came from</strong> — no bacteria at
                all, just a one-hot label. It scores a median{' '}
                <span className="num">AUROC {outcome.median_auroc_cohort_identity_only?.toFixed(3)}</span>.
                A model with zero biological information nearly matches the impressive one.
              </p>
            </div>

            <div className="table-scroll" style={{ marginTop: 'var(--space-5)' }}>
              <table className="table">
                <thead>
                  <tr>
                    <th>Model</th>
                    <th>What it can see</th>
                    <th className="num">Median AUROC</th>
                  </tr>
                </thead>
                <tbody>
                  <tr>
                    <td>Pooled across cohorts</td>
                    <td className="muted">Bacteria + (implicitly) which lab ran the sample</td>
                    <td className="num">{outcome.median_auroc_pooled?.toFixed(3)}</td>
                  </tr>
                  <tr>
                    <td>Cohort identity only</td>
                    <td className="muted">
                      <em>Only</em> which lab ran the sample. No biology whatsoever.
                    </td>
                    <td className="num">{outcome.median_auroc_cohort_identity_only?.toFixed(3)}</td>
                  </tr>
                  <tr>
                    <td>
                      <strong>Within cohort</strong>
                    </td>
                    <td className="muted">
                      Bacteria only — cases vs controls from the same lab, same platform
                    </td>
                    <td className="num">
                      <strong>{outcome.median_auroc_within_cohort?.toFixed(3)}</strong>
                    </td>
                  </tr>
                </tbody>
              </table>
            </div>

            <div className="note note--caution" style={{ marginTop: 'var(--space-5)', maxWidth: '66ch' }}>
              The honest number is the last one:{' '}
              <span className="num">{outcome.median_auroc_within_cohort?.toFixed(3)}</span>. Of{' '}
              <span className="num">{outcome.n_condition_cohort_models}</span> condition-by-cohort
              models tested, only{' '}
              <span className="num">{outcome.n_validated_conditions}</span> clear chance once a
              95% confidence interval is applied. That is why this site shows literature
              citations rather than risk scores, and says so on every signal it displays.
            </div>
          </div>
        </section>
      )}

      {summary?.backtests?.cross_cohort && (
        <section className="section">
          <div className="wrap">
            <h2>It does not transfer to a lab it has never seen</h2>
            <div className="prose" style={{ marginTop: 'var(--space-4)' }}>
              <p>
                The headline accuracy — median{' '}
                <span className="num">
                  r = {summary.metabolite_model.median_test_pearson_r.toFixed(3)}
                </span>{' '}
                — is measured on held-out <em>patients</em> drawn from the same fourteen cohorts
                the model trained on. A stricter test is to hold out an entire cohort: train on
                thirteen, predict the fourteenth, and repeat for all of them.
              </p>
              <p>
                Under that test the median correlation is{' '}
                <span className="num">
                  {summary.backtests.cross_cohort.median_r.toFixed(4)}
                </span>
                . Not reduced — <strong>gone</strong>. Across all{' '}
                <span className="num">{summary.backtests.cross_cohort.n_cohorts}</span> held-out
                cohorts the best result was{' '}
                <span className="num">{summary.backtests.cross_cohort.best_cohort_r.toFixed(3)}</span>{' '}
                and the worst was{' '}
                <span className="num">{summary.backtests.cross_cohort.worst_cohort_r.toFixed(3)}</span>.
              </p>
              <p>
                Since Pearson correlation is unaffected by rescaling, this cannot be explained
                away as differing measurement units between mass-spectrometry platforms. The
                model genuinely fails to rank samples correctly in a cohort it was not trained
                on.
              </p>
            </div>

            <div className="disclaimer" style={{ marginTop: 'var(--space-5)' }}>
              <strong>What this means for how you should read this tool.</strong> The predictions
              are informative about relationships <em>within</em> this dataset — which is a real
              and useful thing for understanding how microbial composition and gut chemistry
              relate. They should not be treated as a validated instrument for a new sample from
              a new laboratory. Closing this gap is the central open problem for the project, not
              a caveat at the bottom of it.
            </div>
          </div>
        </section>
      )}

      {summary?.backtests?.negative_control && (
        <section className="section">
          <div className="wrap">
            <h2>Leakage check</h2>
            <div className="prose" style={{ marginTop: 'var(--space-4)' }}>
              <p>
                A pipeline can produce excellent scores purely by accidentally leaking the answer
                into the inputs. The test: shuffle each metabolite's values across training
                patients, destroying any real relationship, and refit everything unchanged. A
                correct pipeline should then score zero.
              </p>
              <p>
                Real data scored{' '}
                <span className="num">
                  r = {summary.backtests.negative_control.median_real_r.toFixed(3)}
                </span>
                ; shuffled data scored{' '}
                <span className="num">
                  r = {summary.backtests.negative_control.median_shuffled_r.toFixed(4)}
                </span>{' '}
                across{' '}
                <span className="num">{summary.backtests.negative_control.n_targets}</span>{' '}
                metabolites.{' '}
                {summary.backtests.negative_control.passed
                  ? 'The pipeline passes — the within-cohort signal is real, whatever its limits.'
                  : 'The pipeline FAILS this check and every other number here is suspect.'}
              </p>
            </div>
          </div>
        </section>
      )}

      {summary && (
        <section className="section">
          <div className="wrap">
            <h2>Metabolite prediction accuracy</h2>
            <p className="muted" style={{ marginTop: 'var(--space-3)', maxWidth: '64ch' }}>
              Measured on held-out patients — people whose samples the model never saw during
              training, split by individual rather than by sample so that repeat visits from one
              person cannot leak across the boundary.
            </p>

            <div className="stats" style={{ marginTop: 'var(--space-5)' }}>
              <Stat
                value={summary.metabolite_model.median_test_pearson_r.toFixed(3)}
                label="median correlation (r)"
              />
              <Stat
                value={String(summary.metabolite_model['n_targets_r_above_0.5'])}
                label="well predicted (r ≥ 0.5)"
              />
              <Stat
                value={String(summary.metabolite_model['n_targets_r_above_0.3'])}
                label="usable (r ≥ 0.3)"
              />
              <Stat
                value={String(summary.metabolite_model['n_targets_r_below_0.2'])}
                label="poorly predicted (r < 0.2)"
              />
            </div>

            {summary.mlp_comparison && (
              <div className="prose" style={{ marginTop: 'var(--space-6)' }}>
                <h3 style={{ marginBottom: 'var(--space-3)' }}>
                  The neural network lost to the linear model
                </h3>
                <p>
                  A deep network was trained on the same data and the same splits. It reached a
                  median{' '}
                  <span className="num">
                    r = {summary.mlp_comparison.median_test_r_mlp.toFixed(3)}
                  </span>{' '}
                  against the elastic net's{' '}
                  <span className="num">
                    r = {summary.mlp_comparison.median_test_r_elastic_net?.toFixed(3)}
                  </span>
                  . With roughly 2,000 training samples against 5,038 input features, there is
                  not enough data for the deep model to earn its complexity, so the simpler
                  model ships.
                </p>
                <p>
                  That turns out to be a feature rather than a consolation. Because the served
                  model is linear, each prediction decomposes exactly into per-microbe
                  contributions that sum to the answer. The "why this number" breakdown in the
                  simulator is not an approximation of the model's reasoning — it <em>is</em> the
                  model's reasoning, read directly off the weights.
                </p>
              </div>
            )}
          </div>
        </section>
      )}

      <section className="section">
        <div className="wrap">
          <h2>Pipeline run log</h2>
          <p className="muted" style={{ marginTop: 'var(--space-3)', maxWidth: '64ch' }}>
            Every data pull, training run, and backtest appends a record here — including the
            ones that failed. This is the same log the project's own tooling reads; nothing is
            curated for presentation.
          </p>

          <div style={{ marginTop: 'var(--space-5)' }}>
            {runs.map((run, i) => (
              <div className="run" key={`${run.step}-${run.started_at}-${i}`}>
                <span className="run__step">{run.step}</span>
                <span className="tiny faint">
                  {new Date(run.started_at).toLocaleString()} ·{' '}
                  <span className="num">{run.duration_s.toFixed(1)}s</span>
                  {run.artifacts.length > 0 && ` · ${run.artifacts.length} artifacts`}
                </span>
                <span className={`run__status run__status--${run.status}`}>{run.status}</span>
                {Object.keys(run.metrics).length > 0 && (
                  <div className="run__metrics">
                    {Object.entries(run.metrics)
                      .slice(0, 6)
                      .map(([key, value]) => (
                        <span key={key}>
                          {key}={typeof value === 'number' ? Number(value.toFixed(4)) : String(value)}
                        </span>
                      ))}
                  </div>
                )}
                {run.error && (
                  <div className="run__metrics" style={{ color: 'var(--counter)' }}>
                    {run.error}
                  </div>
                )}
              </div>
            ))}
            {runs.length === 0 && <div className="loading">No runs recorded yet.</div>}
          </div>
        </div>
      </section>
    </>
  )
}
