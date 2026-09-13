import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import {
  api,
  type ExampleProfile,
  type MapPoint,
  type SimulationResult,
  type Taxon,
} from '../api'
import { MetaboliteRow } from '../components/MetaboliteRow'
import { PopulationMap } from '../components/PopulationMap'
import { SignalCard } from '../components/SignalCard'
import { TaxonSlider } from '../components/TaxonSlider'
import { Disclaimer, Eyebrow } from '../components/primitives'

const DEBOUNCE_MS = 180

function conditionLabel(condition: string): string {
  return condition.replace(/_/g, ' ')
}

/**
 * The tool itself — the first thing on the page, not buried under marketing.
 *
 * Requests are debounced rather than fired per slider frame: the model is fast,
 * but a request per pixel of drag would be wasteful and would make results
 * arrive out of order. An in-flight sequence number guards against a slow early
 * response overwriting a fast later one, which is the classic way a live
 * control panel ends up displaying stale numbers.
 */
export function Simulator() {
  const [taxa, setTaxa] = useState<Taxon[]>([])
  const [examples, setExamples] = useState<ExampleProfile[]>([])
  const [adjustments, setAdjustments] = useState<Record<string, number>>({})
  const [exampleId, setExampleId] = useState<string | undefined>(undefined)
  const [result, setResult] = useState<SimulationResult | null>(null)
  const [mapPoints, setMapPoints] = useState<MapPoint[]>([])
  const [explainedVariance, setExplainedVariance] = useState<number[]>([])
  const [error, setError] = useState<string | null>(null)
  const [pending, setPending] = useState(false)

  const requestSeq = useRef(0)
  const debounceTimer = useRef<number | undefined>(undefined)

  useEffect(() => {
    Promise.all([api.taxa(), api.examples(), api.populationMap()])
      .then(([taxaList, exampleList, map]) => {
        setTaxa(taxaList)
        setExamples(exampleList)
        setMapPoints(map.points)
        setExplainedVariance(map.explained_variance)
        // Keyed by the full GTDB lineage, not the genus name: several lineages
        // can share a genus, so a genus key resolves to whichever column the
        // backend happens to index first -- which may not be the column this
        // slider's healthy-reference default was computed from.
        setAdjustments(
          Object.fromEntries(taxaList.map((t) => [t.feature, t.healthy_abundance])),
        )
      })
      .catch((e: Error) => setError(e.message))
  }, [])

  const runSimulation = useCallback(
    (current: Record<string, number>, example: string | undefined) => {
      const seq = ++requestSeq.current
      setPending(true)
      api
        .simulate(current, example)
        .then((res) => {
          // Drop responses superseded by a newer request.
          if (seq !== requestSeq.current) return
          setResult(res)
          setError(null)
        })
        .catch((e: Error) => {
          if (seq !== requestSeq.current) return
          setError(e.message)
        })
        .finally(() => {
          if (seq === requestSeq.current) setPending(false)
        })
    },
    [],
  )

  useEffect(() => {
    if (taxa.length === 0) return
    window.clearTimeout(debounceTimer.current)
    debounceTimer.current = window.setTimeout(
      () => runSimulation(adjustments, exampleId),
      DEBOUNCE_MS,
    )
    return () => window.clearTimeout(debounceTimer.current)
  }, [adjustments, exampleId, taxa.length, runSimulation])

  const reset = () => {
    setExampleId(undefined)
    setAdjustments(Object.fromEntries(taxa.map((t) => [t.feature, t.healthy_abundance])))
  }

  const changedCount = useMemo(
    () =>
      taxa.filter(
        (t) =>
          Math.abs(
            Math.log10(Math.max(adjustments[t.feature] ?? t.healthy_abundance, 1e-9)) -
              Math.log10(Math.max(t.healthy_abundance, 1e-9)),
          ) > 0.05,
      ).length,
    [taxa, adjustments],
  )

  if (error && taxa.length === 0) {
    return (
      <section className="section">
        <div className="wrap">
          <div className="error">
            Could not reach the model API: {error}
            <div className="tiny muted" style={{ marginTop: 8 }}>
              Start it with <code>uvicorn app.main:app</code> from <code>webapp/backend</code>.
            </div>
          </div>
        </div>
      </section>
    )
  }

  return (
    <section className="section">
      <div className="wrap">
        <Eyebrow>Simulator</Eyebrow>
        <h1 style={{ maxWidth: '20ch' }}>Change the bacteria, watch the chemistry move</h1>
        <p className="lede" style={{ marginTop: 'var(--space-4)' }}>
          Every slider starts at the average level found in healthy people. Move one, and the
          model re-predicts the whole chemical profile. Everything you are not adjusting stays at
          its real healthy-population value — because a profile of five known bacteria and 5,033
          zeros is not a gut that exists.
        </p>

        <div style={{ marginTop: 'var(--space-6)' }} className="grid-2">
          {/* ---------------------------------------------------- controls --- */}
          <div>
            <div className="row" style={{ justifyContent: 'space-between', marginBottom: 8 }}>
              <h3>Bacterial composition</h3>
              <button className="btn btn--ghost btn--small" onClick={reset}>
                Reset
              </button>
            </div>

            <div className="tiny faint" style={{ marginBottom: 12 }}>
              {changedCount === 0
                ? 'Currently showing a typical healthy profile.'
                : `${changedCount} ${changedCount === 1 ? 'taxon' : 'taxa'} changed from typical.`}
              {pending && <span className="loading"> · recomputing…</span>}
            </div>

            <div className="panel" style={{ padding: 'var(--space-3) var(--space-4)' }}>
              {taxa.map((taxon) => (
                <TaxonSlider
                  key={taxon.genus}
                  taxon={taxon}
                  value={adjustments[taxon.feature] ?? taxon.healthy_abundance}
                  onChange={(abundance) =>
                    setAdjustments((prev) => ({ ...prev, [taxon.feature]: abundance }))
                  }
                />
              ))}
              {taxa.length === 0 && <div className="loading">Loading taxa…</div>}
            </div>

            {examples.length > 0 && (
              <div style={{ marginTop: 'var(--space-5)' }}>
                <h3>Or start from a real person's sample</h3>
                <p className="tiny faint" style={{ margin: '6px 0 10px' }}>
                  Published, de-identified research samples from the 14 source cohorts.
                </p>
                <div className="row" style={{ flexWrap: 'wrap', gap: 6 }}>
                  {examples.slice(0, 14).map((ex) => (
                    <button
                      key={ex.id}
                      className="chip"
                      aria-pressed={exampleId === ex.id}
                      onClick={() => setExampleId(exampleId === ex.id ? undefined : ex.id)}
                      title={`${ex.cohort} — ${conditionLabel(ex.condition)}`}
                    >
                      {conditionLabel(ex.condition)}
                    </button>
                  ))}
                </div>
                {exampleId && (
                  <p className="tiny muted" style={{ marginTop: 10 }}>
                    Sliders now apply on top of this real sample rather than the healthy average.
                  </p>
                )}
              </div>
            )}
          </div>

          {/* ------------------------------------------------- predictions --- */}
          <div>
            <h3>Predicted chemical output</h3>
            <p className="tiny faint" style={{ margin: '6px 0 4px' }}>
              The number beside each metabolite is the change from the starting profile. Changes
              are more reliable than absolute levels, because the model's systematic error
              largely cancels in a difference.
            </p>

            <div className="panel" style={{ padding: 'var(--space-2) var(--space-4)' }}>
              {result ? (
                result.metabolites.map((m) => <MetaboliteRow key={m.hmdb_id} metabolite={m} />)
              ) : (
                <div className="loading" style={{ padding: '20px 0' }}>
                  Running model…
                </div>
              )}
            </div>
          </div>
        </div>

        {/* ---------------------------------------------------- signals --- */}
        <div style={{ marginTop: 'var(--space-8)' }}>
          <h2>What the literature says about profiles like this</h2>
          <p className="muted" style={{ marginTop: 'var(--space-3)', maxWidth: '64ch' }}>
            These are not predictions about you or anyone. Where a bacterium in this profile sits
            well away from the healthy average, we look up published studies that reported the
            same shift in the same direction in a given condition, and show you those studies.
          </p>

          <div className="grid-2" style={{ marginTop: 'var(--space-5)' }}>
            {result?.signals.map((signal) => (
              <SignalCard key={signal.condition} signal={signal} />
            ))}
          </div>

          {result && result.signals.length === 0 && (
            <div className="note" style={{ marginTop: 'var(--space-4)' }}>
              Nothing in this profile deviates far enough from the healthy reference to match a
              documented association. That is the expected result for a typical profile.
            </div>
          )}

          {result?.notes.map((note, i) => (
            <div className="note note--caution tiny" key={i} style={{ marginTop: 'var(--space-3)' }}>
              {note}
            </div>
          ))}
        </div>

        {/* -------------------------------------------------------- map --- */}
        <div style={{ marginTop: 'var(--space-8)' }}>
          <h2>Where this profile sits among real people</h2>
          <p className="muted" style={{ margin: 'var(--space-3) 0 var(--space-5)', maxWidth: '64ch' }}>
            Every dot is one real stool sample from the source cohorts. The marker updates live as
            you move the sliders.
          </p>
          <PopulationMap
            points={mapPoints}
            position={result?.map_position ?? null}
            explainedVariance={explainedVariance}
          />
        </div>

        <div style={{ marginTop: 'var(--space-7)' }}>
          <Disclaimer />
        </div>
      </div>
    </section>
  )
}
