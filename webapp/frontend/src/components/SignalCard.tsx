import { useState } from 'react'
import type { Signal } from '../api'

/**
 * One literature-backed condition signal.
 *
 * The design goal is that a reader cannot walk away with the conclusion but
 * without the caveat. So the card leads with what actually matched and how
 * many studies back it, states plainly when no validated model exists for the
 * condition, and puts real citations one click away rather than behind a
 * "learn more" link to nowhere.
 */
export function SignalCard({ signal }: { signal: Signal }) {
  const [open, setOpen] = useState(false)

  return (
    <div className="signal-card">
      <div className="signal-card__head">
        <span className="signal-card__condition">{signal.condition}</span>
        <span className="tiny faint num">
          {signal.matching_taxa} matching {signal.matching_taxa === 1 ? 'taxon' : 'taxa'}
        </span>
      </div>

      <p className="small muted" style={{ margin: '8px 0 10px', maxWidth: '60ch' }}>
        {signal.summary}
      </p>

      {signal.model_auroc !== null ? (
        <div className={`note ${signal.model_validated ? '' : 'note--caution'} tiny`}>
          {signal.model_validated ? (
            <>
              A classifier trained on this condition reaches{' '}
              <span className="num">AUROC {signal.model_auroc.toFixed(2)}</span>
              {signal.model_auroc_ci && (
                <>
                  {' '}
                  (95% CI{' '}
                  <span className="num">
                    {signal.model_auroc_ci[0].toFixed(2)}–{signal.model_auroc_ci[1].toFixed(2)}
                  </span>
                  )
                </>
              )}{' '}
              on patients it never saw — above chance.
            </>
          ) : (
            <>
              A classifier for this condition scored{' '}
              <span className="num">AUROC {signal.model_auroc.toFixed(2)}</span>
              {signal.model_auroc_ci && (
                <>
                  {' '}
                  (95% CI{' '}
                  <span className="num">
                    {signal.model_auroc_ci[0].toFixed(2)}–{signal.model_auroc_ci[1].toFixed(2)}
                  </span>
                  )
                </>
              )}
              , which does not exclude chance. Treat the match below as literature context
              only.
            </>
          )}
        </div>
      ) : (
        <div className="note note--caution tiny">
          No validated classifier exists for this condition in our cohorts — this is a
          literature match only.
        </div>
      )}

      <button
        className="btn btn--ghost btn--small"
        style={{ marginTop: 12 }}
        onClick={() => setOpen((v) => !v)}
        aria-expanded={open}
      >
        {open ? 'Hide evidence' : `Evidence & sources`}
      </button>

      {open && (
        <div style={{ marginTop: 10 }}>
          {signal.evidence.map((item) => (
            <div className="evidence-row" key={item.taxon}>
              <span className="evidence-row__taxon">{item.taxon}</span>
              <span className={`tag tag--${item.observed === 'elevated' ? 'up' : 'down'}`}>
                {item.observed} {item.deviation_sd > 0 ? '+' : ''}
                {item.deviation_sd}σ
              </span>
              <span className="tiny muted">
                <span className="num">{item.n_reports}</span> studies ·{' '}
                <span className="num">{(item.consistency * 100).toFixed(0)}%</span> agree
                {item.study_quality !== null && (
                  <>
                    {' '}· quality <span className="num">{(item.study_quality * 100).toFixed(0)}%</span>
                  </>
                )}
                {item.latest_year && <> · latest <span className="num">{item.latest_year}</span></>}
              </span>
            </div>
          ))}

          {signal.citations.length > 0 && (
            <div style={{ marginTop: 12 }}>
              <div className="tiny faint" style={{ marginBottom: 4 }}>
                Sources
              </div>
              {signal.citations.map((c) => (
                <div className="citation" key={c.pmid}>
                  <a href={c.url} target="_blank" rel="noopener noreferrer">
                    {c.first_author} {c.year ? `(${c.year})` : ''} — {c.title}
                  </a>
                  {c.journal && <span className="faint"> · {c.journal}</span>}
                </div>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  )
}
