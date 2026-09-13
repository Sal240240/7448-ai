/**
 * API client.
 *
 * Base URL comes from VITE_API_BASE at build time and defaults to a relative
 * path, which the Vite dev proxy handles — so development is same-origin and
 * needs no CORS, while a hosted build can point at a separate API host.
 */

const API_BASE = import.meta.env.VITE_API_BASE ?? ''

export type Confidence = 'good' | 'moderate' | 'weak' | 'very_weak' | 'unvalidated'

export interface Driver {
  taxon: string
  contribution: number
  direction: 'increases' | 'decreases'
}

export interface Metabolite {
  hmdb_id: string
  name: string
  family: string | null
  value: number
  baseline_value?: number | null
  delta?: number | null
  test_pearson_r: number | null
  confidence: Confidence
  population_percentile: number | null
  explanation: string | null
  why_it_matters: string | null
  drivers: Driver[]
}

export interface Evidence {
  taxon: string
  observed: 'elevated' | 'reduced'
  reported: 'elevated' | 'reduced'
  deviation_sd: number
  n_reports: number
  n_gut_reports: number
  consistency: number
  study_quality: number | null
  latest_year: string | number
}

export interface Citation {
  pmid: string
  title: string
  first_author: string
  year: number | null
  journal: string
  url: string
}

export interface Signal {
  condition: string
  matching_taxa: number
  evidence_weight: number
  summary: string
  model_auroc: number | null
  model_auroc_ci: [number, number] | null
  model_validated: boolean
  evidence: Evidence[]
  citations: Citation[]
}

export interface SimulationResult {
  metabolites: Metabolite[]
  signals: Signal[]
  deviations: Record<string, number>
  map_position: { x: number; y: number }
  notes: string[]
}

export interface Taxon {
  genus: string
  display_name: string
  feature: string
  healthy_abundance: number
  prevalence: number
  n_disease_associations: number
}

export interface ExampleProfile {
  id: string
  cohort: string
  condition: string
  role: string
}

export interface MapPoint {
  x: number
  y: number
  cohort: string
  condition: string
  role: string
}

export interface ModelSummary {
  dataset: {
    n_samples: number
    n_subjects: number
    n_cohorts: number
    n_taxa_features: number
    n_metabolite_targets: number
  }
  metabolite_model: {
    type: string
    median_test_pearson_r: number
    mean_test_pearson_r: number
    'n_targets_r_above_0.5': number
    'n_targets_r_above_0.3': number
    'n_targets_r_below_0.2': number
    trained_on_samples: number
  }
  mlp_comparison?: {
    median_test_r_elastic_net: number | null
    median_test_r_mlp: number
  }
  outcome_model?: {
    median_auroc_within_cohort: number
    median_auroc_pooled: number
    median_auroc_cohort_identity_only: number
    n_validated_conditions: number
    n_condition_cohort_models: number
  }
  backtests?: {
    cross_cohort?: {
      median_r: number
      n_cohorts: number
      n_targets_sampled: number
      best_cohort_r: number
      worst_cohort_r: number
    }
    negative_control?: {
      median_real_r: number
      median_shuffled_r: number
      n_targets: number
      passed: boolean
    }
  }
}

export interface MetaboliteOutlook {
  hmdb_id: string
  name: string
  n_subjects: number
  median_change: number
  p25_change: number
  p75_change: number
  share_increasing: number
  median_days: number | null
}

export interface OutlookResult {
  n_neighbours: number
  median_follow_up_days: number | null
  cohorts: string[]
  metabolites: MetaboliteOutlook[]
  notes: string[]
}

export interface RunRecord {
  step: string
  status: string
  started_at: string
  duration_s: number
  params: Record<string, unknown>
  metrics: Record<string, number | string | null>
  artifacts: string[]
  notes: string[]
  error: string | null
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, {
    headers: { 'Content-Type': 'application/json' },
    ...init,
  })
  if (!response.ok) {
    let detail = `Request failed (${response.status})`
    try {
      const body = await response.json()
      if (body?.detail) detail = typeof body.detail === 'string' ? body.detail : detail
    } catch {
      /* response wasn't JSON; the status-based message is the best we have */
    }
    throw new Error(detail)
  }
  return response.json() as Promise<T>
}

export const api = {
  simulate: (adjustments: Record<string, number>, exampleId?: string) =>
    request<SimulationResult>('/api/simulate', {
      method: 'POST',
      body: JSON.stringify({
        adjustments,
        example_id: exampleId ?? null,
        featured_only: true,
      }),
    }),
  outlook: (adjustments: Record<string, number>, exampleId?: string) =>
    request<OutlookResult>('/api/outlook', {
      method: 'POST',
      body: JSON.stringify({
        adjustments,
        example_id: exampleId ?? null,
        featured_only: true,
      }),
    }),
  taxa: () => request<Taxon[]>('/api/taxa'),
  examples: () => request<ExampleProfile[]>('/api/examples'),
  modelInfo: () => request<ModelSummary>('/api/model-info'),
  populationMap: () =>
    request<{ points: MapPoint[]; explained_variance: number[] }>('/api/population-map'),
  runs: (limit = 25) => request<RunRecord[]>(`/api/runs?limit=${limit}`),
}
