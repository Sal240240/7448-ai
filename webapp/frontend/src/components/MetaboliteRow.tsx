import { useState } from 'react'
import type { Metabolite } from '../api'
import { ConfidenceBadge, ContributionBar } from './primitives'

function deltaClass(delta: number | null | undefined): string {
  if (delta === null || delta === undefined || Math.abs(delta) < 0.01) return 'flat'
  return delta > 0 ? 'up' : 'down'
}

/**
 * One predicted metabolite.
 *
 * The delta against the base profile is the headline rather than the absolute
 * value: systematic model error largely cancels in a difference, so "this
 * change moved butyrate down 0.38" is a far better-supported statement than
 * "butyrate is 9.15". The absolute value and its population percentile stay
 * available underneath for anyone who wants them.
 */
export function MetaboliteRow({ metabolite }: { metabolite: Metabolite }) {
  const [open, setOpen] = useState(false)
  const { delta, population_percentile: percentile } = metabolite
  const hasDelta = delta !== null && delta !== undefined
  const maxContribution = Math.max(
    ...metabolite.drivers.map((d) => Math.abs(d.contribution)),
    0.0001,
  )

  return (
    <div className="metabolite">
      <div className="metabolite__head">
        <div>
          <span className="metabolite__name">{metabolite.name}</span>
          {metabolite.family && (
            <span className="metabolite__family"> · {metabolite.family}</span>
          )}
        </div>
        <div className="row">
          <ConfidenceBadge confidence={metabolite.confidence} r={metabolite.test_pearson_r} />
          {hasDelta && (
            <span className={`metabolite__delta metabolite__delta--${deltaClass(delta)}`}>
              {delta > 0 ? '+' : ''}
              {delta.toFixed(2)}
            </span>
          )}
        </div>
      </div>

      {metabolite.explanation && <p className="metabolite__desc">{metabolite.explanation}</p>}

      {percentile !== null && (
        <>
          <div className="percentile" role="img"
               aria-label={`Predicted level is higher than ${percentile.toFixed(0)} percent of real samples`}>
            <span className="percentile__fill" style={{ width: `${percentile}%` }} />
            <span className="percentile__marker" style={{ left: `calc(${percentile}% - 1px)` }} />
          </div>
          <div className="tiny faint" style={{ marginTop: 4 }}>
            Higher than <span className="num">{percentile.toFixed(0)}%</span> of the 2,900 real
            samples measured
          </div>
        </>
      )}

      <button
        className="btn btn--ghost btn--small"
        style={{ marginTop: 12 }}
        onClick={() => setOpen((v) => !v)}
        aria-expanded={open}
      >
        {open ? 'Hide' : 'Why this number'}
      </button>

      {open && (
        <div style={{ marginTop: 12 }}>
          {metabolite.why_it_matters && (
            <p className="small muted" style={{ maxWidth: '62ch' }}>
              {metabolite.why_it_matters}
            </p>
          )}
          <div className="tiny faint" style={{ margin: '10px 0 6px' }}>
            Microbes driving this prediction — bar length is each one's exact contribution,
            left of centre pushes the level down, right pushes it up.
          </div>
          {metabolite.drivers.map((driver) => (
            <ContributionBar
              key={driver.taxon}
              name={driver.taxon}
              value={driver.contribution}
              max={maxContribution}
            />
          ))}
          {metabolite.drivers.length === 0 && (
            <div className="tiny faint">
              The model uses no microbial features for this metabolite — its prediction is just
              the population average.
            </div>
          )}
        </div>
      )}
    </div>
  )
}
