import type { Confidence } from '../api'

const CONFIDENCE_COPY: Record<Confidence, string> = {
  good: 'well predicted',
  moderate: 'moderate',
  weak: 'weak',
  very_weak: 'very weak',
  unvalidated: 'not validated',
}

/**
 * A prediction's accuracy, rendered inseparably from the prediction itself.
 *
 * The tooltip carries the actual held-out correlation rather than only the
 * band, so a reader who wants the number can get it without a detour into the
 * methodology page.
 */
export function ConfidenceBadge({ confidence, r }: { confidence: Confidence; r: number | null }) {
  const label = CONFIDENCE_COPY[confidence] ?? confidence
  const title =
    r === null
      ? 'No held-out accuracy estimate available for this metabolite.'
      : `Held-out correlation r = ${r.toFixed(2)} against measured values in patients the model never saw.`
  return (
    <span className={`conf conf--${confidence}`} title={title}>
      <span className="conf__dot" aria-hidden="true" />
      {label}
      {r !== null && <span className="num"> r={r.toFixed(2)}</span>}
    </span>
  )
}

export function Stat({ value, label }: { value: string; label: string }) {
  return (
    <div className="stat">
      <div className="stat__value">{value}</div>
      <div className="stat__label">{label}</div>
    </div>
  )
}

export function Eyebrow({ children }: { children: React.ReactNode }) {
  return <div className="eyebrow">{children}</div>
}

/**
 * Signed contribution bar with zero pinned to the centre.
 *
 * Direction is readable without reading the number — which matters because
 * these update live under a slider, faster than anyone reads digits.
 */
export function ContributionBar({
  name,
  value,
  max,
}: {
  name: string
  value: number
  max: number
}) {
  const scale = max > 0 ? Math.min(Math.abs(value) / max, 1) : 0
  const halfWidth = scale * 50
  const positive = value > 0
  return (
    <div className="contrib">
      <span className="contrib__name" title={name}>
        {name}
      </span>
      <span className="contrib__track">
        <span className="contrib__axis" aria-hidden="true" />
        <span
          className={`contrib__bar ${positive ? 'contrib__bar--pos' : 'contrib__bar--neg'}`}
          style={{
            left: positive ? '50%' : `${50 - halfWidth}%`,
            width: `${halfWidth}%`,
          }}
        />
      </span>
      <span className="contrib__value">
        {value >= 0 ? '+' : ''}
        {value.toFixed(3)}
      </span>
    </div>
  )
}

export function Disclaimer() {
  return (
    <div className="disclaimer">
      <strong>This is a research tool, not a medical one.</strong> It reports statistical
      associations found in published studies of groups of people. It does not diagnose,
      screen for, or rule out any condition, and it cannot tell you anything about your own
      health. Talk to a clinician about symptoms.
    </div>
  )
}
