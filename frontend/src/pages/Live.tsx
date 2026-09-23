import { useEffect, useRef, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { api, formatScore, formatTime } from '../lib/api'
import type { Candidate, LiveDay, LiveStatus } from '../lib/api'
import StatusBadge from '../components/StatusBadge'

const POLL_MS = 3000

export default function Live() {
  const [days, setDays] = useState<LiveDay[] | null>(null)
  const [status, setStatus] = useState<LiveStatus | null>(null)
  const [top, setTop] = useState<Candidate[] | null>(null)
  const [error, setError] = useState<string | null>(null)
  const navigate = useNavigate()
  const poll = useRef<number>(0)

  function refreshAvailable() {
    api.liveAvailable().then((b) => setDays(b.days)).catch((e: Error) => setError(e.message))
  }

  useEffect(() => {
    refreshAvailable()
    api.liveStatus().then(setStatus).catch((e: Error) => setError(e.message))
  }, [])

  // Poll while a run is in progress; on completion refresh the day list and
  // load the finished sequence's top candidates.
  useEffect(() => {
    if (status?.state !== 'running') return
    poll.current = window.setTimeout(() => {
      api.liveStatus().then((s) => {
        setStatus(s)
        if (s.state !== 'running') refreshAvailable()
      }).catch(() => {})
    }, POLL_MS)
    return () => clearTimeout(poll.current)
  }, [status])

  useEffect(() => {
    if (status?.state === 'done' && status.sequence_ids.length) {
      api.candidates({ sequence_id: status.sequence_ids[0], limit: 10 })
        .then((page) => setTop(page.items))
        .catch(() => {})
    } else {
      setTop(null)
    }
  }, [status?.state, status?.sequence_ids.join(',')])

  const running = status?.state === 'running'

  return (
    <>
      <h1 className="page-title">Live analysis</h1>
      <p className="page-sub">
        NASA's rolling archive keeps the last ~2 weeks of LASCO C3 imagery
        (science-grade files arrive a few hours behind the spacecraft). Pick a
        day, pull it, and the full pipeline runs: download → align → motion
        extraction → tracking → Motion DNA → model scoring. New candidates
        appear ranked below and in the explorer, with full per-track evidence.
      </p>

      {error && <div className="error-note">Backend unreachable: {error}</div>}

      {status && status.state !== 'idle' && (
        <div className="panel">
          <h2>
            {running ? `Analyzing ${status.date}…`
              : status.state === 'done' ? `Analysis of ${status.date} complete`
              : `Analysis of ${status.date} failed`}
          </h2>
          {running && (
            <>
              <p style={{ margin: '0 0 8px' }}>
                Stage {status.stages_done + 1}/{status.stages_total}:{' '}
                <span className="mono">{status.stage}</span>
              </p>
              <div className="bar-track">
                <div className="bar" style={{
                  width: `${(status.stages_done / status.stages_total) * 100}%`,
                  height: 6,
                }} />
              </div>
            </>
          )}
          {status.state === 'error' && (
            <div className="error-note">{status.error}</div>
          )}
          {status.log_tail.length > 0 && (
            <pre className="mono" style={{
              fontSize: 11, color: 'var(--text-muted)', maxHeight: 160,
              overflow: 'auto', margin: '10px 0 0', whiteSpace: 'pre-wrap',
            }}>{status.log_tail.join('\n')}</pre>
          )}
          {top && (
            <>
              <h2 style={{ marginTop: 18 }}>Top candidates from this run</h2>
              <div className="table-wrap">
                <table>
                  <thead>
                    <tr>
                      <th className="num">#</th><th>Track</th>
                      <th className="num">Fusion score</th><th>Status</th>
                      <th className="num">Frames</th><th>Start</th>
                    </tr>
                  </thead>
                  <tbody>
                    {top.map((c, i) => (
                      <tr key={c.track_id} className="clickable"
                        onClick={() => navigate(`/candidates/${c.track_id}`)}>
                        <td className="num" style={{ color: 'var(--text-muted)' }}>{i + 1}</td>
                        <td className="mono">{c.track_id}</td>
                        <td className="num mono">{formatScore(c.fusion_score)}</td>
                        <td><StatusBadge status={c.status} /></td>
                        <td className="num">{c.n_frames}</td>
                        <td>{formatTime(c.start_time)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              <p className="page-sub" style={{ margin: '10px 0 0', fontSize: 13 }}>
                Click a row for detection crops, trajectory, Motion DNA and the
                agent report — or open the full ranked list in the{' '}
                <a style={{ color: 'var(--accent)', cursor: 'pointer' }}
                  onClick={() => navigate(`/candidates?sequence_id=${status.sequence_ids[0]}`)}>
                  candidate explorer
                </a>.
              </p>
            </>
          )}
        </div>
      )}

      <div className="panel">
        <h2>Recent days available to pull</h2>
        {!days && <div className="loading">Checking the archive…</div>}
        {days && days.length === 0 && (
          <p className="page-sub" style={{ margin: 0 }}>
            The archive listing returned nothing — it may be temporarily
            unreachable. Try again in a minute.
          </p>
        )}
        {days && days.length > 0 && (
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>Day (UT)</th>
                  <th className="num">C3 frames on archive</th>
                  <th>Local state</th>
                  <th></th>
                </tr>
              </thead>
              <tbody>
                {days.map((d) => (
                  <tr key={d.date}>
                    <td className="mono">{d.date}</td>
                    <td className="num">{d.frames}</td>
                    <td>
                      {d.ingested
                        ? <span style={{ color: 'var(--status-known-comet)' }}>analyzed</span>
                        : <span style={{ color: 'var(--text-muted)' }}>not pulled</span>}
                    </td>
                    <td style={{ textAlign: 'right' }}>
                      {d.ingested && d.sequence_ids.length > 0 && (
                        <button onClick={() => navigate(`/candidates?sequence_id=${d.sequence_ids[0]}`)}>
                          View results
                        </button>
                      )}{' '}
                      <button disabled={running || d.frames === 0}
                        onClick={() => {
                          setError(null)
                          api.liveRun(d.date).then(setStatus).catch((e: Error) => setError(e.message))
                        }}>
                        {d.ingested ? 'Re-analyze' : 'Pull & analyze'}
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
        <p className="page-sub" style={{ margin: '10px 0 0', fontSize: 13 }}>
          A full day is ~120 frames and the complete analysis typically takes
          several minutes. Days still being downlinked have fewer frames;
          re-analyzing later picks up the newly arrived ones.
        </p>
      </div>
    </>
  )
}
