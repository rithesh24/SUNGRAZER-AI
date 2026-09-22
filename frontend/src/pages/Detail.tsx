import { useEffect, useRef, useState } from 'react'
import { Link, useParams } from 'react-router-dom'
import { api, formatScore, formatTime } from '../lib/api'
import type { AgentReport, CandidateDetail, Evidence } from '../lib/api'
import StatusBadge from '../components/StatusBadge'
import ReviewPanel from '../components/ReviewPanel'

const FRAME_PX = 128 // 32px crop * backend scale 4

function CropViewer({ trackId }: { trackId: string }) {
  const [frames, setFrames] = useState(0)
  const [frame, setFrame] = useState(0)
  const [playing, setPlaying] = useState(true)
  const [missing, setMissing] = useState(false)
  const url = `/api/candidates/${trackId}/crops.png`

  useEffect(() => {
    const img = new Image()
    img.onload = () => setFrames(Math.round(img.naturalWidth / FRAME_PX))
    img.onerror = () => setMissing(true)
    img.src = url
  }, [url])

  useEffect(() => {
    if (!playing || frames === 0) return
    const t = setInterval(() => setFrame((f) => (f + 1) % frames), 180)
    return () => clearInterval(t)
  }, [playing, frames])

  if (missing) return <p className="page-sub">No crop stack on disk for this track.</p>
  if (frames === 0) return <div className="loading">Loading crops…</div>

  return (
    <div className="crop-viewer">
      <div className="crop-frame" style={{
        backgroundImage: `url(${url})`,
        backgroundPosition: `-${frame * FRAME_PX}px 0`,
      }} />
      <div className="crop-controls">
        <button onClick={() => setPlaying(!playing)}>{playing ? 'Pause' : 'Play'}</button>
        <input type="range" min={0} max={frames - 1} value={frame}
          onChange={(e) => { setPlaying(false); setFrame(Number(e.target.value)) }} />
        <span className="mono">{frame + 1}/{frames}</span>
      </div>
      <p className="page-sub" style={{ margin: '8px 0 0', fontSize: 12 }}>
        32×32 px cutouts from background-subtracted registered frames,
        centered on the detection. The moving source should stay centered
        while residual noise changes frame to frame.
      </p>
    </div>
  )
}

function Trajectory({ evidence }: { evidence: Evidence }) {
  const positions = (evidence.track.positions ?? []) as [number, number][]
  const sun = (evidence.provenance.sun_center_xy ?? null) as [number, number] | null
  if (positions.length < 2) return null

  const xs = positions.map((p) => p[0])
  const ys = positions.map((p) => p[1])
  const pad = 60
  const minX = Math.min(...xs) - pad
  const maxX = Math.max(...xs) + pad
  const minY = Math.min(...ys) - pad
  const maxY = Math.max(...ys) + pad
  const points = positions.map((p) => `${p[0]},${p[1]}`).join(' ')
  const first = positions[0]
  const last = positions[positions.length - 1]

  return (
    <svg className="trajectory" viewBox={`${minX} ${minY} ${maxX - minX} ${maxY - minY}`}
      preserveAspectRatio="xMidYMid meet">
      {sun && (
        <line x1={last[0]} y1={last[1]} x2={sun[0]} y2={sun[1]}
          stroke="var(--status-medium)" strokeWidth={1} strokeDasharray="4 5"
          opacity={0.5} vectorEffect="non-scaling-stroke" />
      )}
      <polyline points={points} fill="none" stroke="var(--accent)"
        strokeWidth={2} vectorEffect="non-scaling-stroke" />
      {positions.map((p, i) => (
        <circle key={i} cx={p[0]} cy={p[1]} r={2.5} fill="var(--accent)" opacity={0.7} />
      ))}
      <circle cx={first[0]} cy={first[1]} r={5} fill="none"
        stroke="var(--text-secondary)" strokeWidth={1.5} vectorEffect="non-scaling-stroke" />
      <circle cx={last[0]} cy={last[1]} r={5} fill="var(--status-high)" />
      <text x={first[0] + 9} y={first[1] + 4} className="svg-label">start</text>
      <text x={last[0] + 9} y={last[1] + 4} className="svg-label">end</text>
    </svg>
  )
}

// Just enough markdown for the agent's ##-and-bullets reports.
function ReportBody({ text }: { text: string }) {
  const blocks: React.ReactNode[] = []
  let list: string[] = []
  const flush = () => {
    if (list.length) {
      blocks.push(<ul key={blocks.length}>{list.map((t, i) => <li key={i}>{inline(t)}</li>)}</ul>)
      list = []
    }
  }
  const inline = (t: string) =>
    t.split(/\*\*(.+?)\*\*/g).map((part, i) => (i % 2 ? <strong key={i}>{part}</strong> : part))
  for (const raw of text.split('\n')) {
    const line = raw.trim()
    if (line.startsWith('## ')) { flush(); blocks.push(<h3 key={blocks.length}>{line.slice(3)}</h3>) }
    else if (line.startsWith('- ')) list.push(line.slice(2))
    else if (line) { flush(); blocks.push(<p key={blocks.length}>{inline(line)}</p>) }
  }
  flush()
  return <div className="report-body">{blocks}</div>
}

export default function Detail() {
  const { trackId = '' } = useParams()
  const [candidate, setCandidate] = useState<CandidateDetail | null>(null)
  const [evidence, setEvidence] = useState<Evidence | null>(null)
  const [report, setReport] = useState<AgentReport | null>(null)
  const [reportError, setReportError] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)
  const reportRequested = useRef(false)

  useEffect(() => {
    setCandidate(null); setEvidence(null); setReport(null); setError(null)
    reportRequested.current = false
    api.candidate(trackId).then(setCandidate).catch((e: Error) => setError(e.message))
    api.evidence(trackId).then(setEvidence).catch((e: Error) => setError(e.message))
  }, [trackId])

  useEffect(() => {
    // The report triggers an LLM call; request it once, after basics load.
    if (candidate && !reportRequested.current) {
      reportRequested.current = true
      api.report(trackId).then(setReport).catch((e: Error) => setReportError(e.message))
    }
  }, [candidate, trackId])

  if (error) return <div className="error-note" style={{ marginTop: 30 }}>{error}</div>
  if (!candidate) return <div className="loading">Loading candidate…</div>

  const dna = evidence?.motion_dna ?? null

  return (
    <>
      <p style={{ margin: '22px 0 0' }}>
        <Link to="/candidates" style={{ color: 'var(--text-muted)', fontSize: 13 }}>← Candidates</Link>
      </p>
      <h1 className="page-title mono" style={{ marginTop: 6 }}>{candidate.track_id}</h1>
      <p className="page-sub" style={{ display: 'flex', gap: 16, alignItems: 'center', flexWrap: 'wrap' }}>
        <StatusBadge status={candidate.status} />
        <span>{candidate.n_frames} frames</span>
        <span>{formatTime(candidate.start_time)} → {formatTime(candidate.end_time)}</span>
        <span>sequence {candidate.sequence_id}</span>
        <Link to={`/archaeology?track=${candidate.track_id}`}
          style={{ color: 'var(--accent)', fontSize: 13 }}>
          Find similar motion →
        </Link>
      </p>

      <div className="detail-grid">
        <div>
          <div className="panel">
            <h2>Detection crops</h2>
            <CropViewer trackId={candidate.track_id} />
          </div>

          {evidence && (
            <div className="panel">
              <h2>Trajectory · registered-frame pixels</h2>
              <Trajectory evidence={evidence} />
              <p className="page-sub" style={{ margin: '10px 0 0', fontSize: 12 }}>
                Dashed line points toward the Sun's center. Sunward, consistent
                motion is the classic sungrazer signature.
              </p>
            </div>
          )}

          <div className="panel">
            <h2>Discovery agent report</h2>
            {report && (
              <>
                <ReportBody text={report.report} />
                <p className="mono" style={{ fontSize: 11, color: 'var(--text-muted)', marginBottom: 0 }}>
                  generated_by: {report.generated_by}
                  {report.generated_by === 'fallback' && ' (LLM unavailable — deterministic summary)'}
                </p>
              </>
            )}
            {!report && !reportError && <div className="loading">Drafting report…</div>}
            {reportError && <div className="error-note">Report failed: {reportError}</div>}
          </div>
        </div>

        <div>
          <ReviewPanel key={candidate.track_id + (candidate.reviewed_at ?? '')}
            candidate={candidate}
            onSaved={(updated) => setCandidate({ ...candidate, ...updated })} />

          <div className="panel">
            <h2>Model scores</h2>
            <table>
              <tbody>
                {candidate.predictions.map((p) => (
                  <tr key={p.run_id} style={p.run_id === 'fusion_mean_v1' ? { color: 'var(--text-primary)' } : {}}>
                    <td className="mono" style={{ maxWidth: 220, overflow: 'hidden', textOverflow: 'ellipsis' }}>
                      {p.run_id === 'fusion_mean_v1' ? 'fusion (primary)' : p.run_id}
                    </td>
                    <td className="num mono">{formatScore(p.score)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {evidence?.event && (
            <div className="panel">
              <h2>Known-object check</h2>
              <p style={{ margin: '0 0 6px' }}>
                SOHO-{evidence.event.soho_number}
                {evidence.event.group ? ` · ${evidence.event.group}` : ''}
              </p>
              <p className="page-sub" style={{ margin: 0, fontSize: 13 }}>
                {evidence.event.relation === 'confirmed_positive'
                  ? `Confirmed positive track for this event (${evidence.event.day}).`
                  : `On a registered event day (${evidence.event.day}); the event itself was not recovered.`}
              </p>
              {evidence.event.source_line && (
                <p className="mono" style={{ fontSize: 11, color: 'var(--text-muted)', margin: '8px 0 0' }}>
                  {evidence.event.source_line}
                </p>
              )}
            </div>
          )}
          {evidence && !evidence.event && (
            <div className="panel">
              <h2>Known-object check</h2>
              <p className="page-sub" style={{ margin: 0 }}>
                No attribution to a catalogued Sungrazer event.
              </p>
            </div>
          )}

          {dna && (
            <div className="panel">
              <h2>Motion DNA</h2>
              <table>
                <tbody>
                  {Object.entries(dna)
                    .filter(([, v]) => typeof v === 'number')
                    .map(([key, value]) => (
                      <tr key={key}>
                        <td>{key}</td>
                        <td className="num mono">{Number(value).toPrecision(4)}</td>
                      </tr>
                    ))}
                </tbody>
              </table>
            </div>
          )}

          {evidence && (
            <div className="panel">
              <h2>Provenance</h2>
              <table>
                <tbody>
                  {Object.entries(evidence.provenance)
                    .filter(([, v]) => typeof v === 'string')
                    .map(([key, value]) => (
                      <tr key={key}>
                        <td>{key}</td>
                        <td className="mono" style={{ maxWidth: 200, overflow: 'hidden', textOverflow: 'ellipsis' }}>
                          {String(value)}
                        </td>
                      </tr>
                    ))}
                </tbody>
              </table>
              <ul className="page-sub" style={{ fontSize: 12, margin: '10px 0 0', paddingLeft: 18 }}>
                {evidence.caveats.map((c, i) => <li key={i}>{c}</li>)}
              </ul>
            </div>
          )}
        </div>
      </div>
    </>
  )
}
