import type { Taxon } from '../api'

/**
 * Abundance control for one genus.
 *
 * Relative abundances span several orders of magnitude — Faecalibacterium sits
 * around 4.5% while Fusobacterium sits near 0.01% — so a linear slider would
 * spend 99% of its travel in a range where nothing changes. The control is
 * therefore logarithmic in abundance, with the healthy-population value marked
 * so "normal" is visible rather than something you have to know.
 */

const MIN_LOG = -6 // 0.0001%
const MAX_LOG = -0.7 // ~20%

function toSlider(abundance: number): number {
  if (abundance <= 0) return MIN_LOG
  return Math.min(Math.max(Math.log10(abundance), MIN_LOG), MAX_LOG)
}

function fromSlider(value: number): number {
  return 10 ** value
}

function formatPercent(abundance: number): string {
  const pct = abundance * 100
  if (pct >= 1) return `${pct.toFixed(1)}%`
  if (pct >= 0.01) return `${pct.toFixed(2)}%`
  return `${pct.toExponential(1)}%`
}

export function TaxonSlider({
  taxon,
  value,
  onChange,
}: {
  taxon: Taxon
  value: number
  onChange: (abundance: number) => void
}) {
  const healthy = taxon.healthy_abundance
  const changed = Math.abs(Math.log10(Math.max(value, 1e-9)) - Math.log10(Math.max(healthy, 1e-9))) > 0.05
  const healthyPosition = ((toSlider(healthy) - MIN_LOG) / (MAX_LOG - MIN_LOG)) * 100
  const ratio = healthy > 0 ? value / healthy : 1

  return (
    <div className={`slider ${changed ? 'slider--changed' : ''}`}>
      <span className="slider__name" title={taxon.feature}>{taxon.genus}</span>
      <span className="slider__value">{formatPercent(value)}</span>

      <input
        className="slider__input"
        type="range"
        min={MIN_LOG}
        max={MAX_LOG}
        step={0.01}
        value={toSlider(value)}
        onChange={(e) => onChange(fromSlider(Number(e.target.value)))}
        aria-label={`${taxon.genus} relative abundance`}
      />

      <div className="slider__meta">
        <span title={`Healthy-population average: ${formatPercent(healthy)}`}>
          typical <span className="num">{formatPercent(healthy)}</span>
          <span
            aria-hidden="true"
            style={{
              display: 'inline-block',
              width: 1,
              height: 7,
              background: 'var(--rule-strong)',
              margin: '0 0 0 6px',
              verticalAlign: 'middle',
              opacity: healthyPosition >= 0 ? 1 : 0,
            }}
          />
        </span>
        <span>
          {changed ? (
            <span className="num">
              {ratio >= 1 ? `${ratio.toFixed(1)}× typical` : `${(1 / ratio).toFixed(1)}× below typical`}
            </span>
          ) : (
            <span className="faint">{taxon.n_disease_associations} studied associations</span>
          )}
        </span>
      </div>
    </div>
  )
}
