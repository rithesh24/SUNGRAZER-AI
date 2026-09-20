"""Train the baseline temporal ranking model (tracker section 14).

Usage:
    python -m ml.train [--data-root PATH] [--epochs N] [--seed N]
    python -m ml.train --evaluate-test   # deliberate, logged test-set run

Trains ``temporal_ranker_v1`` (ml.net) on the ``train`` split and selects
the checkpoint by average precision on ``val``. The ``test`` split is
NEVER touched unless ``--evaluate-test`` is passed explicitly — it is
reserved for model selection across architectures (tracker section 15).

Class imbalance (single-digit positives vs thousands of negatives) is
handled by a WeightedRandomSampler that draws positives at a configurable
expected fraction of each epoch (default 10%). The loss stays plain
BCE-with-logits: sampler oversampling and pos_weight together would
double-count the correction.

Every run writes an experiment record (claude.md section 19) under
``data/models/temporal_ranker_v1/run_<confighash>_data<labelshash>_seed<seed>/``:
``config.json``, ``metrics.json`` (per-epoch history + best), and
``checkpoint.pt`` (best-val-AP weights). The run directory is keyed by
config hash + a labels.json content hash + seed, so retraining after the
dataset grows lands in a NEW directory instead of overwriting the prior
experiment (claude.md section 19). Same config + dataset + seed
reproduces the same run and overwrites its own directory only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import random
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler

from ml.dataset import DNA_FEATURE_COUNT, TrackCropDataset
from ml.metrics import ranking_report
from ml.net import MODEL_VERSION, TemporalRanker
from pipeline.fileio import atomic_write_text

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TrainConfig:
    """Deterministic configuration for one training run."""

    model_version: str = MODEL_VERSION
    epochs: int = 20
    batch_size: int = 32
    learning_rate: float = 1e-3
    max_frames: int = 16
    embed_dim: int = 64
    hidden_dim: int = 64
    positive_fraction: float = 0.1  # expected positive share per epoch
    use_dna: bool = False  # concat Motion DNA features before the head
    dropout: float = 0.0  # on frame embeddings + pooled features (ml.net)
    weight_decay: float = 0.0  # Adam L2 regularization
    seed: int = 0

    @property
    def dna_dim(self) -> int:
        return DNA_FEATURE_COUNT if self.use_dna else 0

    def to_dict(self) -> dict:
        return asdict(self)

    def config_hash(self) -> str:
        payload = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()[:16]


def dataset_hash(data_root: Path) -> str:
    """Content hash of labels.json — the dataset identity of a run."""
    labels_file = data_root / "dataset" / "v1" / "labels.json"
    return hashlib.sha256(labels_file.read_bytes()).hexdigest()[:8]


def run_dir_for(data_root: Path, config: TrainConfig) -> Path:
    return (data_root / "models" / MODEL_VERSION
            / f"run_{config.config_hash()}_data{dataset_hash(data_root)}"
              f"_seed{config.seed}")


def pick_device() -> torch.device:
    """CUDA when available, else CPU. Recorded in every run's metrics."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def seed_everything(seed: int) -> torch.Generator:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # Reproducibility over raw speed on GPU (claude.md section 19); the
    # convolutions here are tiny, the deterministic algorithms cost little.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def to_device(batch: dict, device: torch.device) -> dict:
    """Move a loader batch's tensors to the device (lengths stay CPU-safe:
    pack_padded_sequence calls .cpu() on them in ml.net)."""
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()}


NUM_WORKERS = int(os.environ.get("ML_DATALOADER_WORKERS", "4"))


def _loader_kwargs() -> dict:
    """Parallel data loading so disk/CPU work overlaps model compute.

    Deterministic here: the dataset applies no random transforms and the
    sampler runs (seeded) in the main process, so worker count does not
    affect results.
    """
    if NUM_WORKERS <= 0:
        return {}
    return {"num_workers": NUM_WORKERS, "persistent_workers": True,
            "pin_memory": torch.cuda.is_available()}


def make_train_loader(dataset: TrackCropDataset, config: TrainConfig,
                      generator: torch.Generator) -> DataLoader:
    """Oversample positives to ``positive_fraction`` of expected draws."""
    labels = dataset.labels.numpy()
    n_pos, n_neg = int(labels.sum()), int((labels == 0).sum())
    if n_pos == 0:
        raise ValueError("train split has no positives")
    pos_weight = (config.positive_fraction / n_pos) if n_pos else 0.0
    neg_weight = (1.0 - config.positive_fraction) / n_neg if n_neg else 0.0
    weights = np.where(labels == 1, pos_weight, neg_weight)
    sampler = WeightedRandomSampler(
        torch.as_tensor(weights, dtype=torch.double), num_samples=len(dataset),
        replacement=True, generator=generator)
    return DataLoader(dataset, batch_size=config.batch_size, sampler=sampler,
                      **_loader_kwargs())


@torch.no_grad()
def evaluate(model: nn.Module, dataset: TrackCropDataset,
             batch_size: int, return_scores: bool = False,
             device: torch.device | None = None):
    """Score every track of a split and compute the ranking report."""
    device = device or pick_device()
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        **_loader_kwargs())
    scores, labels = [], []
    for batch in loader:
        batch = to_device(batch, device)
        logits = model(batch["crops"], batch["length"], batch.get("dna"))
        scores.append(torch.sigmoid(logits).cpu())
        labels.append(batch["label"].cpu())
    scores = torch.cat(scores).numpy()
    labels = torch.cat(labels).numpy()
    report = ranking_report(labels, scores)
    if return_scores:
        return report, scores, labels
    return report


def train(data_root: Path, config: TrainConfig) -> dict:
    generator = seed_everything(config.seed)
    train_set = TrackCropDataset(data_root, "train",
                                 max_frames=config.max_frames,
                                 with_dna=config.use_dna)
    val_set = TrackCropDataset(data_root, "val", max_frames=config.max_frames,
                               with_dna=config.use_dna)
    loader = make_train_loader(train_set, config, generator)

    device = pick_device()
    logger.info("training on %s", device)
    model = TemporalRanker(config.embed_dim, config.hidden_dim,
                           dna_dim=config.dna_dim,
                           dropout=config.dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate,
                                 weight_decay=config.weight_decay)
    loss_fn = nn.BCEWithLogitsLoss()

    run_dir = run_dir_for(data_root, config)
    run_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_text(run_dir / "config.json",
                      json.dumps(config.to_dict(), indent=2))

    history, best = [], {"val_average_precision": -1.0, "epoch": None}
    for epoch in range(1, config.epochs + 1):
        model.train()
        epoch_loss, n_batches = 0.0, 0
        for batch in loader:
            batch = to_device(batch, device)
            optimizer.zero_grad()
            logits = model(batch["crops"], batch["length"], batch.get("dna"))
            loss = loss_fn(logits, batch["label"])
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.detach())
            n_batches += 1
        val_report = evaluate(model, val_set, config.batch_size,
                              device=device)
        entry = {"epoch": epoch, "train_loss": epoch_loss / max(n_batches, 1),
                 "val": val_report}
        history.append(entry)
        logger.info("epoch %d: loss %.4f  val AP %.4f  val R@50 %.2f",
                    epoch, entry["train_loss"],
                    val_report["average_precision"], val_report["recall_at_50"])
        if val_report["average_precision"] > best["val_average_precision"]:
            best = {"val_average_precision": val_report["average_precision"],
                    "epoch": epoch}
            torch.save({"model_version": MODEL_VERSION,
                        "config": config.to_dict(),
                        "epoch": epoch,
                        "state_dict": model.state_dict()},
                       run_dir / "checkpoint.pt")

    train_report = evaluate(model, train_set, config.batch_size,
                            device=device)
    record = {
        "model_version": MODEL_VERSION,
        "config_hash": config.config_hash(),
        "dataset_hash": dataset_hash(data_root),
        "device": str(device),
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "best": best,
        "final_train": train_report,
        "history": history,
    }
    atomic_write_text(run_dir / "metrics.json", json.dumps(record, indent=2))
    logger.info("run complete: best val AP %.4f (epoch %s) -> %s",
                best["val_average_precision"], best["epoch"], run_dir)
    return record


def evaluate_test(data_root: Path, config: TrainConfig) -> dict:
    """Deliberate test-split evaluation of a finished run's checkpoint."""
    run_dir = run_dir_for(data_root, config)
    device = pick_device()
    checkpoint = torch.load(run_dir / "checkpoint.pt", weights_only=True,
                            map_location=device)
    model = TemporalRanker(config.embed_dim, config.hidden_dim,
                           dna_dim=config.dna_dim,
                           dropout=config.dropout).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    test_set = TrackCropDataset(data_root, "test",
                                max_frames=config.max_frames,
                                with_dna=config.use_dna)
    report, scores, labels = evaluate(model, test_set, config.batch_size,
                                      return_scores=True, device=device)
    order = np.argsort(-scores)
    rank = np.empty(len(scores), dtype=int)
    rank[order] = np.arange(1, len(scores) + 1)
    positive_ranks = {test_set.track_ids[i]: int(rank[i])
                      for i in np.where(labels == 1)[0]}
    record = {"model_version": MODEL_VERSION,
              "dataset_hash": dataset_hash(data_root),
              "checkpoint_epoch": checkpoint["epoch"],
              "evaluated_at": datetime.now(timezone.utc).isoformat(),
              "test": report,
              "positive_ranks": positive_ranks}
    atomic_write_text(run_dir / "test_metrics.json",
                      json.dumps(record, indent=2))
    logger.info("test: AP %.4f  R@50 %.2f  (%d pos / %d tracks)",
                report["average_precision"], report["recall_at_50"],
                report["n_pos"], report["n_total"])
    for track_id, track_rank in sorted(positive_ranks.items(),
                                       key=lambda kv: kv[1]):
        logger.info("  positive %s rank %d/%d",
                    track_id, track_rank, report["n_total"])
    return record


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train the temporal ranker")
    parser.add_argument("--data-root",
                        default=os.environ.get("SOHO_DATA_ROOT", "data"))
    parser.add_argument("--epochs", type=int, default=TrainConfig.epochs)
    parser.add_argument("--seed", type=int, default=TrainConfig.seed)
    parser.add_argument("--max-frames", type=int,
                        default=TrainConfig.max_frames)
    parser.add_argument("--use-dna", action="store_true",
                        help="concatenate Motion DNA features to the GRU "
                             "state before the scoring head")
    parser.add_argument("--dropout", type=float,
                        default=TrainConfig.dropout,
                        help="dropout on frame embeddings + pooled features")
    parser.add_argument("--weight-decay", type=float,
                        default=TrainConfig.weight_decay,
                        help="Adam weight decay (L2)")
    parser.add_argument("--evaluate-test", action="store_true",
                        help="evaluate an existing checkpoint on the test "
                             "split (deliberate, logged)")
    args = parser.parse_args(argv)
    logging.basicConfig(level="INFO", format="%(levelname)s %(message)s")
    config = TrainConfig(epochs=args.epochs, seed=args.seed,
                         max_frames=args.max_frames, use_dna=args.use_dna,
                         dropout=args.dropout,
                         weight_decay=args.weight_decay)
    if args.evaluate_test:
        evaluate_test(Path(args.data_root), config)
    else:
        train(Path(args.data_root), config)
    return 0


if __name__ == "__main__":
    sys.exit(main())
