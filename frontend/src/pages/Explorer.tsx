import { useEffect, useState } from 'react'
import { useNavigate, useSearchParams } from 'react-router-dom'
import { api, REVIEW_LABELS, STATUS_LABELS, STATUS_ORDER, formatScore, formatTime } from '../lib/api'
import type { CandidatePage } from '../lib/api'
import StatusBadge from '../components/StatusBadge'

const PAGE_SIZE = 50

// Explorer doubles as the Review queue and Unknown-object views via presets.
export interface ExplorerPreset {
  title: string
  sub: string
  status?: string
  lockStatus?: boolean
}

const DEFAULT_PRESET: ExplorerPreset = {
  title: 'Candidate explorer',
  sub: 'All persistent moving-object tracks, ranked by the fusion score (mean of model checkpoint sigmoids). Click a row for full evidence.',
}

export default function Explorer({ preset = DEFAULT_PRESET }: { preset?: ExplorerPreset }) {
  const [params, setParams] = useSearchParams()
  const [page, setPage] = useState<CandidatePage | null>(null)
  const [error, setError] = useState<string | null>(null)
  const navigate = useNavigate()

  const status = preset.lockStatus ? (preset.status ?? '') : (params.get('status') ?? preset.status ?? '')
  const label = params.get('label') ?? ''
  const sequenceId = params.get('sequence_id') ?? ''
  const q = params.get('q') ?? ''
  const offset = Number(params.get('offset') ?? 0)

  useEffect(() => {
    setPage(null)
    setError(null)
    api.candidates({
      status: status || undefined,
      label: label || undefined,
      sequence_id: sequenceId ? Number(sequenceId) : undefined,
      q: q || undefined,
      limit: PAGE_SIZE,
      offset,
    })
      .then(setPage)
      .catch((e: Error) => setError(e.message))
  }, [status, label, sequenceId, q, offset])

  function update(key: string, value: string) {
    const next = new URLSearchParams(params)
    if (value) next.set(key, value)
    else next.delete(key)
    if (key !== 'offset') next.delete('offset') // filter change resets paging
    setParams(next, { replace: true })
  }

  return (
    <>
      <h1 className="page-title">{preset.title}</h1>
      <p className="page-sub">{preset.sub}</p>

      <div className="filters">
        {!preset.lockStatus && (
          <select value={status} onChange={(e) => update('status', e.target.value)}>
            <option value="">All statuses</option>
            {STATUS_ORDER.map((s) => (
              <option key={s} value={s}>{STATUS_LABELS[s]}</option>
            ))}
          </select>
        )}
        <select value={label} onChange={(e) => update('label', e.target.value)}>
          <option value="">All labels</option>
          <option value="comet">Labeled comet</option>
          <option value="excluded">Excluded day</option>
        </select>
        <input type="number" placeholder="Sequence #" value={sequenceId} min={1}
          style={{ width: 110 }}
          onChange={(e) => update('sequence_id', e.target.value)} />
        <input type="search" placeholder="Search track ID…" value={q}
          style={{ width: 220 }} className="mono"
          onChange={(e) => update('q', e.target.value)} />
        <div className="spacer" />
        {page && <span style={{ fontSize: 13, color: 'var(--text-secondary)' }}>
          {page.total.toLocaleString()} tracks
        </span>}
      </div>

      {error && <div className="error-note">Backend unreachable: {error}</div>}
      {!error && !page && <div className="loading">Querying candidates…</div>}

      {page && (
        <div className="panel">
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th className="num">#</th>
                  <th>Track</th>
                  <th className="num">Fusion score</th>
                  <th>Status</th>
                  <th>Review</th>
                  <th>Label</th>
                  <th className="num">Seq</th>
                  <th className="num">Frames</th>
                  <th>Start</th>
                </tr>
              </thead>
              <tbody>
                {page.items.map((c, i) => (
                  <tr key={c.track_id} className="clickable"
                    onClick={() => navigate(`/candidates/${c.track_id}`)}>
                    <td className="num" style={{ color: 'var(--text-muted)' }}>{offset + i + 1}</td>
                    <td className="mono">{c.track_id}</td>
                    <td className="num mono">{formatScore(c.fusion_score)}</td>
                    <td><StatusBadge status={c.status} /></td>
                    <td>{c.review ? REVIEW_LABELS[c.review] ?? c.review : '—'}</td>
                    <td>{c.label ?? '—'}</td>
                    <td className="num">{c.sequence_id}</td>
                    <td className="num">{c.n_frames}</td>
                    <td>{formatTime(c.start_time)}</td>
                  </tr>
                ))}
                {page.items.length === 0 && (
                  <tr><td colSpan={9} style={{ color: 'var(--text-muted)' }}>No tracks match these filters.</td></tr>
                )}
              </tbody>
            </table>
          </div>
          <div className="pager">
            <span>
              {page.total === 0 ? '0' : `${offset + 1}–${Math.min(offset + PAGE_SIZE, page.total)}`}
              &nbsp;of {page.total.toLocaleString()}
            </span>
            <button disabled={offset === 0}
              onClick={() => update('offset', String(Math.max(0, offset - PAGE_SIZE)))}>
              ← Prev
            </button>
            <button disabled={offset + PAGE_SIZE >= page.total}
              onClick={() => update('offset', String(offset + PAGE_SIZE))}>
              Next →
            </button>
          </div>
        </div>
      )}
    </>
  )
}
