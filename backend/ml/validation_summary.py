"""Scientific validation summary (ship plan S9, tracker section 31).

One reproducible report over the frozen dataset + fusion scores:

1. Per-event recovery — every labeled Sungrazer event: was at least one
   confirmed track recovered by the CV pipeline, and where does its best
   track rank (globally and within its own sequence) under the primary
   fusion score?
2. Ranking quality per split — AP / P@10 / R@50 / R@250 on the exact
   populations the training evaluation used (``ml.dataset.resolve_split``),
   scored with the DB's ``fusion_mean_v1`` predictions.

Usage:
    python -m ml.validation_summary [--data-root PATH] [--json OUT.json]

Markdown goes to stdout; --json adds a machine-readable copy.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
from sqlalchemy import select

from db.models import Candidate, ModelPrediction
from db.session import SessionLocal
from ml.dataset import resolve_split
from ml.metrics import average_precision, precision_at_k, recall_at_k

FUSION_RUN_ID = "fusion_mean_v1"
SPLITS = ("train", "val", "test")


def _fusion_scores(session) -> dict[str, float]:
    """track_id -> fusion score for every scored candidate."""
    rows = session.execute(
        select(Candidate.track_id, ModelPrediction.score)
        .join(ModelPrediction, ModelPrediction.candidate_id == Candidate.id)
        .where(ModelPrediction.run_id == FUSION_RUN_ID)
    ).all()
    return dict(rows)


def _sequence_of(track_id: str) -> int:
    return int(track_id.split("_", 1)[0][3:])


def build_summary(labels: dict, scores: dict[str, float],
                  data_root: Path) -> dict:
    ranked = sorted(scores.items(), key=lambda kv: -kv[1])
    global_rank = {t: i + 1 for i, (t, _) in enumerate(ranked)}
    seq_tracks: dict[int, list[str]] = {}
    for track in scores:
        seq_tracks.setdefault(_sequence_of(track), []).append(track)
    for tracks in seq_tracks.values():
        tracks.sort(key=lambda t: -scores[t])

    events = []
    day_split = {d: s for s in SPLITS for d in labels["splits"][s]}
    for event in sorted(labels["events"], key=lambda e: e["soho_number"]):
        positives = [t for rec in event["days"].values()
                     for t in rec.get("positive_track_ids", [])]
        # A day absent from every split list was never used for ML
        # (e.g. observation days outside the ingested sequences).
        splits = sorted({day_split[d] for d in event["days"]
                         if d in day_split}) or ["unsplit"]
        entry = {
            "soho_number": event["soho_number"],
            "group": event.get("group"),
            "splits": splits,
            "days": len(event["days"]),
            "recovered": bool(positives),
            "n_positive_tracks": len(positives),
        }
        if positives:
            best = max(positives, key=lambda t: scores.get(t, float("-inf")))
            seq = _sequence_of(best)
            entry.update({
                "best_track": best,
                "best_fusion": round(scores.get(best, float("nan")), 4),
                "global_rank": global_rank.get(best),
                "global_pool": len(ranked),
                "sequence_rank": seq_tracks[seq].index(best) + 1,
                "sequence_pool": len(seq_tracks[seq]),
            })
        events.append(entry)

    split_metrics = {}
    for split in SPLITS:
        tracks = resolve_split(labels, split, data_root)
        ids = tracks.positives + tracks.negatives
        missing = [t for t in ids if t not in scores]
        y = np.array([1.0] * len(tracks.positives)
                     + [0.0] * len(tracks.negatives))
        s = np.array([scores.get(t, float("-inf")) for t in ids])
        split_metrics[split] = {
            "n_positives": len(tracks.positives),
            "n_negatives": len(tracks.negatives),
            "n_missing_scores": len(missing),
            "ap": round(average_precision(y, s), 4),
            "p_at_10": round(precision_at_k(y, s, 10), 4),
            "r_at_50": round(recall_at_k(y, s, 50), 4),
            "r_at_250": round(recall_at_k(y, s, 250), 4),
        }

    recovered = [e for e in events if e["recovered"]]
    return {
        "run_id": FUSION_RUN_ID,
        "n_scored_tracks": len(scores),
        "events": events,
        "recovery": {
            "n_events": len(events),
            "n_recovered": len(recovered),
            "n_top10_in_sequence": sum(
                1 for e in recovered if e["sequence_rank"] <= 10),
        },
        "split_metrics": split_metrics,
    }


def to_markdown(summary: dict) -> str:
    rec = summary["recovery"]
    lines = [
        "# Scientific validation summary",
        "",
        f"Primary score: `{summary['run_id']}` over "
        f"{summary['n_scored_tracks']:,} scored tracks.",
        "",
        "## Known-event recovery",
        "",
        f"{rec['n_recovered']}/{rec['n_events']} labeled Sungrazer events "
        f"recovered by the CV pipeline; of those, "
        f"{rec['n_top10_in_sequence']}/{rec['n_recovered']} rank in the "
        "top 10 of their own sequence under the fusion score.",
        "",
        "| SOHO # | Group | Split(s) | Recovered | Best fusion | "
        "Seq rank | Global rank |",
        "|---|---|---|---|---|---|---|",
    ]
    for e in summary["events"]:
        if e["recovered"]:
            lines.append(
                f"| {e['soho_number']} | {e['group']} | "
                f"{'/'.join(e['splits'])} | yes "
                f"({e['n_positive_tracks']} track"
                f"{'s' if e['n_positive_tracks'] != 1 else ''}) | "
                f"{e['best_fusion']:.4f} | "
                f"{e['sequence_rank']}/{e['sequence_pool']} | "
                f"{e['global_rank']}/{e['global_pool']} |")
        else:
            lines.append(
                f"| {e['soho_number']} | {e['group']} | "
                f"{'/'.join(e['splits'])} | no | — | — | — |")
    lines += [
        "",
        "## Ranking quality by split",
        "",
        "| Split | Positives | Negatives | AP | P@10 | R@50 | R@250 |",
        "|---|---|---|---|---|---|---|",
    ]
    for split, m in summary["split_metrics"].items():
        lines.append(
            f"| {split} | {m['n_positives']} | {m['n_negatives']:,} | "
            f"{m['ap']:.4f} | {m['p_at_10']:.4f} | {m['r_at_50']:.4f} | "
            f"{m['r_at_250']:.4f} |")
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-root",
                        default=os.environ.get("SOHO_DATA_ROOT", "data"))
    parser.add_argument("--json", help="also write the summary as JSON")
    args = parser.parse_args(argv)
    logging.basicConfig(level=os.environ.get("LOG_LEVEL", "WARNING"))
    labels = json.loads(
        (Path(args.data_root) / "dataset" / "v1" / "labels.json")
        .read_text(encoding="utf-8"))
    with SessionLocal() as session:
        scores = _fusion_scores(session)
    summary = build_summary(labels, scores, Path(args.data_root))
    print(to_markdown(summary))
    if args.json:
        Path(args.json).write_text(json.dumps(summary, indent=2),
                                   encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
