import { STATUS_LABELS } from '../lib/api'

const STATUS_COLORS: Record<string, string> = {
  KNOWN_COMET: 'var(--status-known-comet)',
  HIGH_PRIORITY: 'var(--status-high)',
  MEDIUM_PRIORITY: 'var(--status-medium)',
  LOW_PRIORITY: 'var(--status-low)',
  UNRESOLVED: 'var(--status-unresolved)',
  LIKELY_ARTIFACT: 'var(--status-artifact)',
}

export function statusColor(status: string): string {
  return STATUS_COLORS[status] ?? 'var(--text-muted)'
}

export default function StatusBadge({ status }: { status: string }) {
  return (
    <span className="badge" style={{ '--badge-color': statusColor(status) } as React.CSSProperties}>
      {STATUS_LABELS[status] ?? status}
    </span>
  )
}
