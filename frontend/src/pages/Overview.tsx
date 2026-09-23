import { useEffect, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { api, STATUS_LABELS, STATUS_ORDER, formatTime } from '../lib/api'
import type { Sequence, Statistics } from '../lib/api'
import { statusColor } from '../components/StatusBadge'

export default function Overview() {
  const [stats, setStats] = useState<Statistics | null>(null)
  const [sequences, setSequences] = useState<Sequence[] | null>(null)
  const [error, setError] = useState<string | null>(null)
  const navigate = useNavigate()

  useEffect(() => {
    Promise.all([api.statistics(), api.sequences()])
      .then(([s, q]) => { setStats(s); setSequences(q) })
      .catch((e: Error) => setError(e.message))
  }, [])

  if (error) return <div className="error-note" style={{ marginTop: 30 }}>Backend unreachable: {error}</div>
  if (!stats || !sequences) return <div className="loading">Loading mission state…</div>

  const byStatus = stats.candidates_by_status
  const maxCount = Math.max(...Object.values(byStatus), 1)
  const frames = sequences.reduce((n, s) => n + s.frame_count, 0)

  return (
    <>
      <h1 className="page-title">Mission overview</h1>
      <p className="page-sub">
        Autonomous detection of faint sungrazing-comet candidates in SOHO/LASCO
        coronagraph imagery. Candidates are ranking signals for human review —
        not confirmed discoveries.
      </p>

      <div className="panel">
        <h2>What this software does</h2>
        <p style={{ margin: '0 0 10px' }}>
          The SOHO spacecraft photographs the Sun's surroundings every ~12
          minutes. Sungrazing comets — fragments of ancient broken-up comets —
          cross that field of view, evaporate against the Sun, and are gone
          within hours. Most are found by volunteers eyeballing the images;
          faint ones get missed.
        </p>
        <p style={{ margin: '0 0 10px' }}>
          SUNGRAZER AI automates the search: it downloads the raw imagery,
          aligns the frames, extracts every persistent moving object, computes
          a behavioral fingerprint (Motion&nbsp;DNA) for each, and ranks them
          with a temporal neural network so the most comet-like tracks surface
          first. A discovery agent then narrates the structured evidence for
          each candidate.
        </p>
        <p className="page-sub" style={{ margin: 0, fontSize: 13 }}>
          Validated on 41 days of 2024 archive data: 10 of 15 confirmed comets
          recovered, each ranked #1 within its own day. A high score means
          "worth a human's two minutes" — confirmation always requires human
          review, and verified positions can be submitted to the Sungrazer
          Project for official designation.
        </p>
      </div>

      <div className="tiles">
        <div className="tile">
          <div className="label">Candidate tracks</div>
          <div className="value">{stats.candidates_total.toLocaleString()}</div>
          <div className="hint">from {frames.toLocaleString()} frames</div>
        </div>
        <div className="tile">
          <div className="label">High priority</div>
          <div className="value" style={{ color: 'var(--status-high)' }}>
            {(byStatus.HIGH_PRIORITY ?? 0).toLocaleString()}
          </div>
          <div className="hint">top of the review queue</div>
        </div>
        <div className="tile">
          <div className="label">Known comets recovered</div>
          <div className="value" style={{ color: 'var(--status-known-comet)' }}>
            {(byStatus.KNOWN_COMET ?? 0).toLocaleString()}
          </div>
          <div className="hint">labeled validation tracks</div>
        </div>
        <div className="tile">
          <div className="label">Sequences processed</div>
          <div className="value">{stats.sequences_total.toLocaleString()}</div>
          <div className="hint">LASCO C3 day-sequences</div>
        </div>
      </div>

      <div className="panel">
        <h2>Candidates by status</h2>
        {STATUS_ORDER.filter((s) => byStatus[s]).map((status) => (
          <div key={status} className="status-row"
            style={{ '--badge-color': statusColor(status) } as React.CSSProperties}>
            <span className="badge">{STATUS_LABELS[status]}</span>
            <span className="count">{byStatus[status].toLocaleString()}</span>
            <div className="bar" style={{ width: `${(byStatus[status] / maxCount) * 100}%` }} />
          </div>
        ))}
        <p className="page-sub" style={{ margin: '12px 0 0', fontSize: 13 }}>
          Bars are scaled to the largest class. Unresolved tracks sit on
          excluded event days and are kept out of automatic prioritization.
        </p>
      </div>

      <div className="panel">
        <h2>Model runs</h2>
        <div className="table-wrap">
          <table>
            <thead>
              <tr><th>Run</th><th>Model</th><th className="num">Predictions</th></tr>
            </thead>
            <tbody>
              {stats.runs.map((r) => (
                <tr key={r.run_id}>
                  <td className="mono">{r.run_id}</td>
                  <td>{r.model_version}</td>
                  <td className="num">{r.n_predictions.toLocaleString()}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      <div className="panel">
        <h2>Image sequences</h2>
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th>Seq</th><th>Instrument</th><th>Window start</th>
                <th className="num">Frames</th><th className="num">Candidates</th>
              </tr>
            </thead>
            <tbody>
              {sequences.map((s) => (
                <tr key={s.id} className="clickable"
                  onClick={() => navigate(`/candidates?sequence_id=${s.id}`)}>
                  <td className="mono">{s.id}</td>
                  <td>{s.instrument}</td>
                  <td>{formatTime(s.start_time)}</td>
                  <td className="num">{s.frame_count}</td>
                  <td className="num">{s.n_candidates.toLocaleString()}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </>
  )
}
