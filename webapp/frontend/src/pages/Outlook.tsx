import { useEffect, useState } from 'react'
import { api, type OutlookResult, type Taxon } from '../api'
import { TaxonSlider } from '../components/TaxonSlider'
import { Disclaimer, Eyebrow } from '../components/primitives'

/**
 * Forward outlook.
 *
 * Deliberately not a forecast chart. The underlying data is 331 people sampled
 * irregularly across six cohorts, which is enough to say "here is what happened
 * to people who started somewhere similar" and nowhere near enough to draw a
 * projected curve. A line going forward in time would imply a precision that
 * would be invented, so the visual is a spread of observed outcomes with the
 * sample size stated on every row.
 */

function ChangeBar({
  median,
  p25,
  p75,
  scale,
}: {
  median: number
  p25: number
  p75: number
  scale: number
}) {
  const toPct = (v: number) => 50 + (v / scale) * 50
  const left = Math.max(0, Math.min(100, toPct(p25)))
  const right = Math.max(0, Math.min(100, toPct(p75)))
  const mid = Math.max(0, Math.min(100, toPct(median)))

  return (
    <span className="contrib__track" style={{ height: 18 }}>
      <span className="contrib__axis" aria-hidden="true" />
      {/* Interquartile range: where the middle half of matched people landed. */}
      <span
        style={{
          position: 'absolute',
          left: `${Math.min(left, right)}%`,
          width: `${Math.abs(right - left)}%`,
          top: 4,
          bottom: 4,
          background: 'var(--rule-strong)',
          borderRadius: 2,
        }}
      />
      <span
        style={{
          position: 'absolute',
          left: `calc(${mid}% - 1px)`,
          top: 1,
          bottom: 1,
          width: 2,
          background: median >= 0 ? 'var(--signal)' : 'var(--counter)',
        }}
      />
    </span>
  )
}

export function Outlook() {
  const [taxa, setTaxa] = useState<Taxon[]>([])
  const [adjustments, setAdjustments] = useState<Record<string, number>>({})
  const [result, setResult] = useState<OutlookResult | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [loading, setLoading] = useState(false)

  useEffect(() => {
    api
      .taxa()
      .then((list) => {
        setTaxa(list)
        setAdjustments(Object.fromEntries(list.map((t) => [t.feature, t.healthy_abundance])))
      })
      .catch((e: Error) => setError(e.message))
  }, [])

  const run = () => {
    setLoading(true)
    api
      .outlook(adjustments)
      .then((res) => {
        setResult(res)
        setError(null)
      })
      .catch((e: Error) => setError(e.message))
      .finally(() => setLoading(false))
  }

  useEffect(() => {
    if (taxa.length > 0 && result === null) run()
    // Runs once after taxa load; the outlook is an on-demand panel, not a live
    // recompute, because the neighbour search reads the full feature table.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [taxa.length])

  const scale = result
    ? Math.max(
        ...result.metabolites.flatMap((m) => [Math.abs(m.p25_change), Math.abs(m.p75_change)]),
        0.1,
      )
    : 1

  return (
    <section className="section">
      <div className="wrap">
        <Eyebrow>Forward outlook</Eyebrow>
        <h1 style={{ maxWidth: '21ch' }}>What happened to people who started here</h1>
        <p className="lede" style={{ marginTop: 'var(--space-4)' }}>
          Six of the fourteen cohorts sampled the same people repeatedly — some weekly for a
          year, some from infancy to age three. This finds the real individuals whose starting
          composition most resembles the profile below, and reports what was actually measured in
          them later.
        </p>

        <div className="note note--caution" style={{ marginTop: 'var(--space-5)', maxWidth: '66ch' }}>
          This is not a prediction of the future. No model of change is fitted. It is a summary of
          measurements from a small number of real people, and with a few hundred subjects across
          six very different studies, the honest output is a spread rather than a trajectory.
        </div>

        <div className="grid-2" style={{ marginTop: 'var(--space-6)' }}>
          <div>
            <div className="row" style={{ justifyContent: 'space-between', marginBottom: 10 }}>
              <h3>Starting composition</h3>
              <button className="btn btn--small" onClick={run} disabled={loading}>
                {loading ? 'Matching…' : 'Find similar people'}
              </button>
            </div>
            <div className="panel" style={{ padding: 'var(--space-3) var(--space-4)' }}>
              {taxa.slice(0, 12).map((taxon) => (
                <TaxonSlider
                  key={taxon.feature}
                  taxon={taxon}
                  value={adjustments[taxon.feature] ?? taxon.healthy_abundance}
                  onChange={(abundance) =>
                    setAdjustments((prev) => ({ ...prev, [taxon.feature]: abundance }))
                  }
                />
              ))}
              {taxa.length === 0 && !error && <div className="loading">Loading…</div>}
            </div>
          </div>

          <div>
            <h3>What was measured in them later</h3>

            {error && <div className="error" style={{ marginTop: 12 }}>{error}</div>}

            {result && (
              <>
                <p className="small muted" style={{ margin: '8px 0 4px' }}>
                  Matched <span className="num">{result.n_neighbours}</span> people
                  {result.median_follow_up_days !== null && (
                    <>
                      {' '}
                      followed for a median of{' '}
                      <span className="num">{result.median_follow_up_days}</span> days
                    </>
                  )}
                  {result.cohorts.length > 0 && (
                    <span className="faint"> · from {result.cohorts.join(', ')}</span>
                  )}
                </p>

                <div className="panel" style={{ padding: 'var(--space-4)' }}>
                  <div className="tiny faint" style={{ marginBottom: 10 }}>
                    Bar spans the middle half of observed changes; the line is the median. Left of
                    centre is a decrease.
                  </div>
                  {result.metabolites.map((m) => (
                    <div key={m.hmdb_id} style={{ marginBottom: 12 }}>
                      <div
                        className="row"
                        style={{ justifyContent: 'space-between', marginBottom: 3 }}
                      >
                        <span className="small">{m.name}</span>
                        <span className="tiny faint num">
                          n={m.n_subjects} · {(m.share_increasing * 100).toFixed(0)}% rose
                        </span>
                      </div>
                      <ChangeBar
                        median={m.median_change}
                        p25={m.p25_change}
                        p75={m.p75_change}
                        scale={scale}
                      />
                    </div>
                  ))}
                  {result.metabolites.length === 0 && (
                    <div className="tiny muted">
                      No metabolite was measured at two timepoints in enough matched people to
                      summarise.
                    </div>
                  )}
                </div>

                {result.notes.map((note, i) => (
                  <div className="note tiny" key={i} style={{ marginTop: 'var(--space-3)' }}>
                    {note}
                  </div>
                ))}
              </>
            )}

            {!result && !error && <div className="loading" style={{ marginTop: 16 }}>Matching…</div>}
          </div>
        </div>

        <div style={{ marginTop: 'var(--space-7)' }}>
          <Disclaimer />
        </div>
      </div>
    </section>
  )
}
