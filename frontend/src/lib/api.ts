// Typed client for the SUNGRAZER AI FastAPI backend (proxied at /api).

export interface Sequence {
  id: number
  instrument: string
  start_time: string | null
  end_time: string | null
  frame_count: number
  status: string
  n_candidates: number
}

export interface Candidate {
  id: number
  track_id: string
  sequence_id: number
  n_frames: number
  start_time: string | null
  end_time: string | null
  label: string | null
  status: string
  review: string | null
  reviewer_notes: string | null
  reviewed_at: string | null
  fusion_score: number | null
}

export interface CandidateDetail extends Candidate {
  features: Record<string, unknown> | null
  predictions: Prediction[]
}

export interface Prediction {
  run_id: string
  model_version: string
  score: number
}

export interface CandidatePage {
  total: number
  limit: number
  offset: number
  items: Candidate[]
}

export interface Statistics {
  candidates_total: number
  candidates_by_status: Record<string, number>
  candidates_by_label: Record<string, number>
  sequences_total: number
  runs: { run_id: string; model_version: string; n_predictions: number }[]
}

export interface Evidence {
  track_id: string
  candidate: {
    sequence_id: number
    instrument: string | null
    start_time: string | null
    end_time: string | null
    n_frames: number
    label: string | null
    status: string
  }
  track: Record<string, unknown>
  motion_dna: Record<string, number | string | string[]> | null
  predictions: Prediction[]
  event: {
    soho_number: number
    relation: string
    day: string
    group: string | null
    source_line: string | null
    notes: string | null
  } | null
  provenance: Record<string, unknown>
  caveats: string[]
}

export interface AgentReport {
  track_id: string
  generated_by: string
  report: string
  event: Evidence['event']
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`/api${path}`, init)
  if (!res.ok) {
    const body = await res.json().catch(() => null)
    const detail = body?.detail
    throw new Error(
      typeof detail === 'string' ? detail : `${res.status} ${res.statusText}`,
    )
  }
  return res.json()
}

const get = request

export const api = {
  health: () => get<{ status: string; database: string }>('/health'),
  sequences: () => get<Sequence[]>('/sequences'),
  statistics: () => get<Statistics>('/statistics'),
  candidates: (params: {
    status?: string
    label?: string
    sequence_id?: number
    q?: string
    limit?: number
    offset?: number
  }) => {
    const search = new URLSearchParams()
    for (const [key, value] of Object.entries(params)) {
      if (value !== undefined && value !== '') search.set(key, String(value))
    }
    return get<CandidatePage>(`/candidates?${search}`)
  },
  candidate: (trackId: string) => get<CandidateDetail>(`/candidates/${trackId}`),
  evidence: (trackId: string) => get<Evidence>(`/candidates/${trackId}/evidence`),
  report: (trackId: string) => get<AgentReport>(`/candidates/${trackId}/report`),
  review: (trackId: string, body: { review?: string | null; reviewer_notes?: string | null }) =>
    request<Candidate>(`/candidates/${trackId}/review`, {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    }),
}

export const REVIEW_VERDICTS = [
  { value: 'approved', label: 'Approve' },
  { value: 'rejected', label: 'Reject' },
  { value: 'artifact', label: 'Artifact' },
  { value: 'known_object', label: 'Known object' },
  { value: 'uncertain', label: 'Uncertain' },
]

export const REVIEW_LABELS: Record<string, string> = Object.fromEntries(
  REVIEW_VERDICTS.map((v) => [v.value, v.label]),
)

export const STATUS_LABELS: Record<string, string> = {
  HIGH_PRIORITY: 'High priority',
  MEDIUM_PRIORITY: 'Medium priority',
  LOW_PRIORITY: 'Low priority',
  KNOWN_COMET: 'Known comet',
  LIKELY_ARTIFACT: 'Likely artifact',
  UNRESOLVED: 'Unresolved',
}

export const STATUS_ORDER = [
  'KNOWN_COMET',
  'HIGH_PRIORITY',
  'MEDIUM_PRIORITY',
  'LOW_PRIORITY',
  'UNRESOLVED',
  'LIKELY_ARTIFACT',
]

export function formatTime(iso: string | null): string {
  if (!iso) return '—'
  return iso.replace('T', ' ').slice(0, 16) + ' UT'
}

export function formatScore(score: number | null): string {
  return score === null ? '—' : score.toFixed(4)
}
