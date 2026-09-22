"""Score candidates with frozen checkpoints and persist model_predictions.

Usage:
    python -m ml.score --all --run run_f7ded5030483e421_dataa3de09e1_seed0 \
        --run run_744ca82263542eda_dataaa66d7b4_seed0
    python -m ml.score --sequence-id 36 --run run_f7ded503... --no-status

With two or more runs a mean-of-sigmoid fusion row is also written
(``run_id`` = ``--fusion-id``, default ``fusion_mean_v1``). After scoring,
unlabeled candidates get a priority status from the primary score (fusion
when present, else the single run): global top 50 → HIGH_PRIORITY, top 250
→ MEDIUM_PRIORITY, rest → LOW_PRIORITY (matches the review budgets in the
frozen-model evaluation). ``KNOWN_COMET`` rows are never touched.

Idempotent: predictions upsert on (candidate_id, run_id).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import torch
from sqlalchemy import select, update
from sqlalchemy.dialects.postgresql import insert
from torch.utils.data import DataLoader

from db.models import Candidate, ModelPrediction
from db.session import SessionLocal
from ml.dataset import TrackCropDataset
from ml.net import TemporalRanker
from ml.train import TrainConfig, _loader_kwargs, pick_device, to_device

logger = logging.getLogger(__name__)

MODEL_VERSION = "temporal_ranker_v1"
HIGH_K, MEDIUM_K = 50, 250


def load_run(data_root: Path, run_name: str,
             device: torch.device) -> tuple[TemporalRanker, TrainConfig]:
    run_dir = data_root / "models" / MODEL_VERSION / run_name
    raw = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    fields = set(TrainConfig.__dataclass_fields__)
    config = TrainConfig(**{k: v for k, v in raw.items() if k in fields})
    checkpoint = torch.load(run_dir / "checkpoint.pt", weights_only=True,
                            map_location=device)
    model = TemporalRanker(config.embed_dim, config.hidden_dim,
                           dna_dim=config.dna_dim,
                           dropout=config.dropout).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, config


@torch.no_grad()
def score_tracks(model: TemporalRanker, config: TrainConfig, data_root: Path,
                 track_ids: list[str], device: torch.device) -> np.ndarray:
    dataset = TrackCropDataset(data_root, None, max_frames=config.max_frames,
                               with_dna=config.use_dna,
                               explicit_tracks=track_ids)
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=False,
                        **_loader_kwargs())
    scores = []
    for batch in loader:
        batch = to_device(batch, device)
        logits = model(batch["crops"], batch["length"], batch.get("dna"))
        scores.append(torch.sigmoid(logits).cpu())
    return torch.cat(scores).numpy()


def upsert_predictions(session, candidate_ids: list[int], run_id: str,
                       scores: np.ndarray) -> None:
    rows = [{"candidate_id": cid, "model_version": MODEL_VERSION,
             "run_id": run_id, "score": float(s)}
            for cid, s in zip(candidate_ids, scores)]
    for start in range(0, len(rows), 5000):
        stmt = insert(ModelPrediction).values(rows[start:start + 5000])
        stmt = stmt.on_conflict_do_update(
            constraint="uq_prediction_run",
            set_={"score": stmt.excluded.score,
                  "model_version": stmt.excluded.model_version})
        session.execute(stmt)


def assign_status(session, primary_run_id: str) -> dict:
    """Priority statuses for unlabeled candidates from the primary score."""
    ranked = session.execute(
        select(Candidate.id)
        .join(ModelPrediction)
        .where(ModelPrediction.run_id == primary_run_id,
               Candidate.label.is_(None))
        .order_by(ModelPrediction.score.desc())
    ).scalars().all()
    high, medium = ranked[:HIGH_K], ranked[HIGH_K:MEDIUM_K]
    rest = ranked[MEDIUM_K:]
    for ids, status in ((high, "HIGH_PRIORITY"), (medium, "MEDIUM_PRIORITY"),
                        (rest, "LOW_PRIORITY")):
        if ids:
            session.execute(update(Candidate)
                            .where(Candidate.id.in_(ids))
                            .values(status=status))
    return {"HIGH_PRIORITY": len(high), "MEDIUM_PRIORITY": len(medium),
            "LOW_PRIORITY": len(rest)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Score candidates")
    parser.add_argument("--data-root",
                        default=os.environ.get("SOHO_DATA_ROOT", "data"))
    parser.add_argument("--sequence-id", type=int, action="append", default=[])
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--run", action="append", required=True,
                        help="run dir name under models/temporal_ranker_v1")
    parser.add_argument("--fusion-id", default="fusion_mean_v1")
    parser.add_argument("--no-status", action="store_true",
                        help="skip priority-status assignment")
    args = parser.parse_args(argv)
    logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"),
                        format="%(asctime)s %(levelname)s %(name)s %(message)s")
    data_root = Path(args.data_root)
    device = pick_device()

    with SessionLocal() as session:
        query = select(Candidate.id, Candidate.track_id)
        if not args.all:
            if not args.sequence_id:
                parser.error("give --sequence-id or --all")
            query = query.where(Candidate.sequence_id.in_(args.sequence_id))
        pairs = session.execute(query.order_by(Candidate.id)).all()
    if not pairs:
        logger.error("no candidates in DB — run dataset.candidates first")
        return 1
    candidate_ids = [p[0] for p in pairs]
    track_ids = [p[1] for p in pairs]
    logger.info("scoring %d candidates on %s", len(track_ids), device)

    per_run: dict[str, np.ndarray] = {}
    for run_name in args.run:
        model, config = load_run(data_root, run_name, device)
        per_run[run_name] = score_tracks(model, config, data_root,
                                         track_ids, device)
        logger.info("run %s: scored (max %.4f)", run_name,
                    per_run[run_name].max())

    with SessionLocal() as session:
        for run_name, scores in per_run.items():
            upsert_predictions(session, candidate_ids, run_name, scores)
        primary = args.run[0]
        if len(per_run) >= 2:
            fused = np.mean(list(per_run.values()), axis=0)
            upsert_predictions(session, candidate_ids, args.fusion_id, fused)
            primary = args.fusion_id
        if not args.no_status:
            counts = assign_status(session, primary)
            logger.info("status assignment (primary=%s): %s", primary, counts)
        session.commit()
    logger.info("Scoring summary: %d candidates x %d runs%s",
                len(candidate_ids), len(per_run),
                " + fusion" if len(per_run) >= 2 else "")
    return 0


if __name__ == "__main__":
    sys.exit(main())
