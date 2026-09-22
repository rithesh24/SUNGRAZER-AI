import { useEffect, useState } from 'react'
import { Link, useNavigate, useSearchParams } from 'react-router-dom'
import { REVIEW_LABELS, formatScore, formatTime } from '../lib/api'
import type { Candidate } from '../lib/api'
import StatusBadge from '../components/StatusBadge'

interface Match extends Candidate {
  dna_distance: number
}

export default function Archaeology() {
  const [params, setParams] = useSearchParams()
  const track = params.get('track') ?? ''
  const [input, setInput] = useState(track)
  const [matches, setMatches] = useState<Match[] | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [loading, setLoading] = useState(false)
  const navigate = useNavigate()

  useEffect(() => {
    setInput(track)
    setMatches(null)
    setError(null)
    if (!track) return
    setLoading(true)
    fetch(`/api/candidates/${track}/similar?limit=25`)
      .then(async (res) => {
        if (!res.ok) throw new Error((await res.json()).detail ?? res.statusText)
        return res.json()
      })
      .then((body) => setMatches(body.matches))
      .catch((e: Error) => setError(e.message))
      .finally(() => setLoading(false))
  }, [track])

  const refSeq = matches?.length ? Number(track.match(/^seq(\d+)_/)?.[1]) : null

  return (
    <>
      <h1 className="page-title">Trajectory archaeology</h1>
      <p className="page-sub">
        Search the processed archive for tracks whose Motion DNA — speed,
        direction consistency, curvature, brightness behavior — is closest to
        a reference track. Matches in <em>other</em> sequences are candidate
        re-appearances or siblings from the same comet group.
      </p>

      <form className="filters" onSubmit={(e) => {
        e.preventDefault()
        if (input.trim()) setParams({ track: input.trim() }, { replace: true })
      }}>
        <input type="search" className="mono" placeholder="Reference track ID…"
          value={input} style={{ width: 320 }}
          onChange={(e) => setInput(e.target.value)} />
        <button type="submit" disabled={!input.trim()}>Search archive</button>
        {!track && (
          <span style={{ fontSize: 13, color: 'var(--text-muted)' }}>
            Tip: open any candidate and use “Find similar motion”, or paste a
            track ID from the <Link to="/candidates" style={{ color: 'var(--accent)' }}>explorer</Link>.
          </span>
        )}
      </form>

      {error && <div className="error-note">{error}</div>}
      {loading && <div className="loading">Measuring the archive…</div>}

      {matches && (
        <div className="panel">
          <h2>
            25 nearest tracks to <span className="mono">{track}</span>
            {' '}· {matches.filter((m) => m.sequence_id !== refSeq).length} in other sequences
          </h2>
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>Track</th>
                  <th className="num">DNA distance</th>
                  <th className="num">Fusion score</th>
                  <th>Status</th>
                  <th>Review</th>
                  <th className="num">Seq</th>
                  <th>Start</th>
                </tr>
              </thead>
              <tbody>
                {matches.map((m) => (
                  <tr key={m.track_id} className="clickable"
                    onClick={() => navigate(`/candidates/${m.track_id}`)}
                    style={m.sequence_id !== refSeq ? {} : { opacity: 0.65 }}>
                    <td className="mono">{m.track_id}</td>
                    <td className="num mono">{m.dna_distance.toFixed(3)}</td>
                    <td className="num mono">{formatScore(m.fusion_score)}</td>
                    <td><StatusBadge status={m.status} /></td>
                    <td>{m.review ? REVIEW_LABELS[m.review] ?? m.review : '—'}</td>
                    <td className="num">{m.sequence_id}</td>
                    <td>{formatTime(m.start_time)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p className="page-sub" style={{ margin: '12px 0 0', fontSize: 12 }}>
            Distance is L2 over z-scored Motion DNA features — lower is more
            similar. Same-sequence rows are dimmed; cross-sequence matches are
            the archaeologically interesting ones. Similar motion is a lead,
            not an identification.
          </p>
        </div>
      )}
    </>
  )
}
