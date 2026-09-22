"""Discovery agent (ship plan S4): evidence package -> written report.

A two-node LangGraph: fetch the evidence package, then draft a report with
Groq. Failure-safe by construction — no GROQ_API_KEY, a network error, or
any LLM failure degrades to a deterministic report assembled from the same
evidence, never an exception. Only a missing candidate raises (KeyError),
so the API can 404.

Usage (one-off):
    python -m agent.discovery --track-id seq36_e91e65277cbe11c1_00104
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import TypedDict

from langgraph.graph import END, StateGraph
from sqlalchemy.orm import Session

from dataset.evidence import build_evidence
from db.session import SessionLocal

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "llama-3.3-70b-versatile"

SYSTEM_PROMPT = """\
You are the discovery assistant of SUNGRAZER AI, a pipeline that surfaces
candidate sungrazing comets in SOHO/LASCO imagery. You receive one
candidate's evidence package (motion track summary, Motion DNA features,
model scores, and any known-event attribution).

Write a concise report in markdown with exactly these sections:
## Assessment — one short paragraph: how comet-like is this track and why.
## Key evidence — 3-6 bullets citing concrete numbers from the package.
## Caveats — bullets; always include that scores are ranking signals, not
detection probabilities, and that confirmation requires human review and
astrometric verification.

Rules: never claim a discovery; never invent numbers not present in the
evidence; if the candidate is attributed to a known SOHO event, say so
plainly and treat it as a validation case, not a find."""


class AgentState(TypedDict, total=False):
    track_id: str
    evidence: dict
    report: str
    generated_by: str


def _compact(evidence: dict) -> dict:
    """Evidence trimmed for the prompt: drop verbatim per-frame arrays."""
    track = evidence.get("track") or {}
    positions = track.get("positions") or []
    out = {
        "track_id": evidence["track_id"],
        "candidate": evidence["candidate"],
        "track_summary": {
            "n_points": len(positions),
            "first_position": positions[0] if positions else None,
            "last_position": positions[-1] if positions else None,
        },
        "motion_dna": evidence.get("motion_dna"),
        "predictions": evidence.get("predictions"),
        "event": evidence.get("event"),
        "caveats": evidence.get("caveats"),
    }
    return out


def _fallback_report(evidence: dict) -> str:
    """Deterministic report from the evidence alone (no LLM)."""
    cand = evidence["candidate"]
    preds = evidence.get("predictions") or []
    fusion = next((p["score"] for p in preds
                   if p["run_id"] == "fusion_mean_v1"), None)
    event = evidence.get("event")
    lines = ["## Assessment",
             f"Automated summary for `{evidence['track_id']}` "
             f"(status {cand['status']}, {cand['n_frames']} frames, "
             f"{cand['instrument'] or 'unknown instrument'}). "
             "LLM narration unavailable; facts below are quoted directly "
             "from the evidence package.",
             "", "## Key evidence"]
    if fusion is not None:
        lines.append(f"- Fusion score (fusion_mean_v1): {fusion:.4f}")
    for p in preds:
        if p["run_id"] != "fusion_mean_v1":
            lines.append(f"- {p['run_id']}: {p['score']:.4f}")
    dna = evidence.get("motion_dna") or {}
    for key in sorted(dna):
        value = dna[key]
        if isinstance(value, (int, float)):
            lines.append(f"- Motion DNA {key}: {value:.4g}")
    if event:
        lines.append(f"- Attribution: SOHO-{event['soho_number']} "
                     f"({event['relation']}, {event.get('day')})")
    else:
        lines.append("- No known-event attribution.")
    lines += ["", "## Caveats"]
    lines += [f"- {c}" for c in evidence.get("caveats", [])]
    return "\n".join(lines)


def _call_llm(evidence: dict) -> str:
    """One Groq chat completion. Raises on any failure; caller degrades."""
    from groq import Groq  # imported lazily; module works without the key

    client = Groq()  # reads GROQ_API_KEY; raises if unset
    response = client.chat.completions.create(
        model=os.environ.get("GROQ_MODEL", DEFAULT_MODEL),
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",
             "content": json.dumps(_compact(evidence), default=str)},
        ],
        temperature=0.2,
        max_tokens=1024,
    )
    text = response.choices[0].message.content
    if not text or not text.strip():
        raise ValueError("empty completion")
    return text


def _make_graph(session: Session, data_root: Path):
    def fetch(state: AgentState) -> AgentState:
        return {"evidence": build_evidence(session, data_root,
                                           state["track_id"])}

    def draft(state: AgentState) -> AgentState:
        try:
            return {"report": _call_llm(state["evidence"]),
                    "generated_by": "groq:" + os.environ.get(
                        "GROQ_MODEL", DEFAULT_MODEL)}
        except Exception as exc:  # failure-safe: any LLM problem degrades
            logger.warning("LLM unavailable (%s); using fallback report", exc)
            return {"report": _fallback_report(state["evidence"]),
                    "generated_by": "fallback"}

    graph = StateGraph(AgentState)
    graph.add_node("fetch_evidence", fetch)
    graph.add_node("draft_report", draft)
    graph.set_entry_point("fetch_evidence")
    graph.add_edge("fetch_evidence", "draft_report")
    graph.add_edge("draft_report", END)
    return graph.compile()


def run_agent(session: Session, data_root: Path, track_id: str) -> dict:
    """Report package for one candidate. Raises KeyError if unknown track."""
    state = _make_graph(session, data_root).invoke({"track_id": track_id})
    return {"track_id": track_id,
            "generated_by": state["generated_by"],
            "report": state["report"],
            "event": state["evidence"].get("event")}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Draft one discovery report")
    parser.add_argument("--data-root",
                        default=os.environ.get("SOHO_DATA_ROOT", "data"))
    parser.add_argument("--track-id", required=True)
    args = parser.parse_args(argv)
    logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))
    with SessionLocal() as session:
        result = run_agent(session, Path(args.data_root), args.track_id)
    print(result["report"])
    print(f"\n[generated_by: {result['generated_by']}]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
