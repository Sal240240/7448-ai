import { memo, useMemo } from 'react'
import type { MapPoint } from '../api'

const WIDTH = 720
const HEIGHT = 460
const PADDING = 28

interface Bounds {
  minX: number
  maxX: number
  minY: number
  maxY: number
}

function computeBounds(points: MapPoint[], extra: { x: number; y: number } | null): Bounds {
  const xs = points.map((p) => p.x)
  const ys = points.map((p) => p.y)
  if (extra) {
    xs.push(extra.x)
    ys.push(extra.y)
  }
  const pad = 0.04
  const minX = Math.min(...xs)
  const maxX = Math.max(...xs)
  const minY = Math.min(...ys)
  const maxY = Math.max(...ys)
  const spanX = maxX - minX || 1
  const spanY = maxY - minY || 1
  return {
    minX: minX - spanX * pad,
    maxX: maxX + spanX * pad,
    minY: minY - spanY * pad,
    maxY: maxY + spanY * pad,
  }
}

/**
 * Static sample cloud, memoized separately from the live marker.
 *
 * 2,900 circles is enough DOM that re-rendering them on every slider frame
 * would visibly stutter. The marker moves; the cloud does not, so they render
 * independently.
 */
const PointCloud = memo(function PointCloud({
  points,
  bounds,
}: {
  points: MapPoint[]
  bounds: Bounds
}) {
  const sx = (x: number) =>
    PADDING + ((x - bounds.minX) / (bounds.maxX - bounds.minX)) * (WIDTH - PADDING * 2)
  const sy = (y: number) =>
    HEIGHT - PADDING - ((y - bounds.minY) / (bounds.maxY - bounds.minY)) * (HEIGHT - PADDING * 2)

  return (
    <g>
      {points.map((p, i) => (
        <circle
          key={i}
          cx={sx(p.x)}
          cy={sy(p.y)}
          r={2.1}
          fill={p.role === 'control' ? 'var(--signal)' : p.role === 'case' ? 'var(--counter)' : 'var(--ink-faint)'}
          opacity={p.role === 'excluded' ? 0.18 : 0.32}
        />
      ))}
    </g>
  )
})

/**
 * Where a simulated profile sits among every real sample in the dataset.
 *
 * Deliberately honest about what it is: the first two principal components
 * capture only ~28% of the variation, so proximity here is suggestive, not a
 * verdict. The caption says so rather than letting the picture imply more
 * precision than the projection has.
 */
export function PopulationMap({
  points,
  position,
  explainedVariance,
}: {
  points: MapPoint[]
  position: { x: number; y: number } | null
  explainedVariance: number[]
}) {
  const bounds = useMemo(() => computeBounds(points, position), [points, position])

  if (points.length === 0) return <div className="loading">Loading population map…</div>

  const sx = (x: number) =>
    PADDING + ((x - bounds.minX) / (bounds.maxX - bounds.minX)) * (WIDTH - PADDING * 2)
  const sy = (y: number) =>
    HEIGHT - PADDING - ((y - bounds.minY) / (bounds.maxY - bounds.minY)) * (HEIGHT - PADDING * 2)

  const totalVariance = explainedVariance.reduce((a, b) => a + b, 0)

  return (
    <div>
      <svg
        className="map"
        viewBox={`0 0 ${WIDTH} ${HEIGHT}`}
        role="img"
        aria-label="Scatter plot of 2,900 real gut microbiome samples with the simulated profile marked"
      >
        <PointCloud points={points} bounds={bounds} />

        {position && (
          <g>
            <line
              x1={sx(position.x)}
              y1={PADDING}
              x2={sx(position.x)}
              y2={HEIGHT - PADDING}
              stroke="var(--ink)"
              strokeWidth={0.5}
              strokeDasharray="3 4"
              opacity={0.45}
            />
            <line
              x1={PADDING}
              y1={sy(position.y)}
              x2={WIDTH - PADDING}
              y2={sy(position.y)}
              stroke="var(--ink)"
              strokeWidth={0.5}
              strokeDasharray="3 4"
              opacity={0.45}
            />
            <circle
              cx={sx(position.x)}
              cy={sy(position.y)}
              r={7}
              fill="none"
              stroke="var(--ink)"
              strokeWidth={1.6}
            />
            <circle cx={sx(position.x)} cy={sy(position.y)} r={2.6} fill="var(--ink)" />
            <text
              x={sx(position.x) + 12}
              y={sy(position.y) - 9}
              fill="var(--ink)"
              fontSize={11}
              fontFamily="var(--font-mono)"
            >
              your profile
            </text>
          </g>
        )}
      </svg>

      <div className="legend" style={{ marginTop: 10 }}>
        <span className="legend__item">
          <span className="legend__swatch" style={{ background: 'var(--signal)' }} />
          healthy / control samples
        </span>
        <span className="legend__item">
          <span className="legend__swatch" style={{ background: 'var(--counter)' }} />
          samples from people with a diagnosed condition
        </span>
        <span className="legend__item">
          <span className="legend__swatch" style={{ background: 'var(--ink-faint)' }} />
          unlabelled
        </span>
      </div>

      <p className="tiny faint" style={{ marginTop: 10, maxWidth: '64ch' }}>
        Each dot is one real stool sample, positioned by overall microbial composition
        (principal components 1 and 2). These two axes capture{' '}
        <span className="num">{(totalVariance * 100).toFixed(0)}%</span> of the total variation —
        so being near a cluster is suggestive, not conclusive. Cases and controls overlap heavily,
        which is the honest picture of how much a microbiome profile alone distinguishes them.
      </p>
    </div>
  )
}
