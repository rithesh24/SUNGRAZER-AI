import { useState } from 'react'
import { api, REVIEW_VERDICTS, formatTime } from '../lib/api'
import type { Candidate } from '../lib/api'

export default function ReviewPanel({ candidate, onSaved }: {
  candidate: Candidate
  onSaved: (updated: Candidate) => void
}) {
  const [verdict, setVerdict] = useState(candidate.review)
  const [notes, setNotes] = useState(candidate.reviewer_notes ?? '')
  const [saving, setSaving] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const dirty = verdict !== candidate.review || notes !== (candidate.reviewer_notes ?? '')

  async function save() {
    setSaving(true)
    setError(null)
    try {
      const updated = await api.review(candidate.track_id, {
        review: verdict ?? null,
        reviewer_notes: notes || null,
      })
      onSaved(updated)
    } catch (e) {
      setError((e as Error).message)
    } finally {
      setSaving(false)
    }
  }

  return (
    <div className="panel">
      <h2>Human review</h2>
      <div className="verdicts">
        {REVIEW_VERDICTS.map((v) => (
          <button key={v.value}
            className={verdict === v.value ? 'verdict active' : 'verdict'}
            onClick={() => setVerdict(verdict === v.value ? null : v.value)}>
            {v.label}
          </button>
        ))}
      </div>
      <textarea rows={3} placeholder="Reviewer notes…" value={notes}
        onChange={(e) => setNotes(e.target.value)} />
      <div className="pager" style={{ justifyContent: 'space-between' }}>
        <span style={{ fontSize: 12, color: 'var(--text-muted)' }}>
          {candidate.reviewed_at ? `last reviewed ${formatTime(candidate.reviewed_at)}` : 'not yet reviewed'}
        </span>
        <button disabled={!dirty || saving} onClick={save}>
          {saving ? 'Saving…' : 'Save review'}
        </button>
      </div>
      {error && <div className="error-note" style={{ marginTop: 8 }}>{error}</div>}
    </div>
  )
}
