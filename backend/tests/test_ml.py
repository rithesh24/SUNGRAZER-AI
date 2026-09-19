"""Offline unit tests for ml.crops, ml.metrics, ml.dataset, ml.net."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ml.crops import CropConfig, cut_crop  # noqa: E402
from ml.dataset import TrackCropDataset, resolve_split  # noqa: E402
from ml.metrics import average_precision, precision_at_k, ranking_report  # noqa: E402
from ml.net import TemporalRanker  # noqa: E402


def test_cut_crop_centering_and_edges():
    frame = np.zeros((100, 100), dtype=np.float32)
    frame[50, 60] = 7.0
    mask = np.ones_like(frame, dtype=bool)
    crop, fraction = cut_crop(frame, mask, x=60, y=50, size=8)
    assert crop.shape == (8, 8) and fraction == 1.0
    assert crop[4, 4] == 7.0  # source lands on the center pixel
    # Corner position: crop extends off-frame, off-frame part is zero.
    frame[0, 0] = 5.0
    crop, fraction = cut_crop(frame, mask, x=0, y=0, size=8)
    assert crop[4, 4] == 5.0 and fraction == 16 / 64
    assert crop[:4].sum() == 0 and crop[:, :4].sum() == 0
    # Invalid-mask pixels are zeroed but tracked in the fraction.
    mask[50, 60] = False
    crop, _ = cut_crop(frame, mask, x=60, y=50, size=8)
    assert crop[4, 4] == 0.0


def test_crop_config_hash_stable():
    assert CropConfig().config_hash() == CropConfig().config_hash()
    assert CropConfig().config_hash() != CropConfig(crop_size_px=64).config_hash()


def test_ranking_metrics_known_case():
    labels = np.array([1, 0, 1, 0, 0])
    scores = np.array([0.9, 0.8, 0.7, 0.6, 0.1])
    # positives at ranks 1 and 3 -> AP = (1/1 + 2/3) / 2
    assert abs(average_precision(labels, scores) - (1 + 2 / 3) / 2) < 1e-9
    assert precision_at_k(labels, scores, 1) == 1.0
    assert precision_at_k(labels, scores, 4) == 0.5
    report = ranking_report(labels, scores, ks=(2,))
    assert report["n_pos"] == 2 and report["recall_at_2"] == 0.5
    assert report["precision"] == 0.5  # threshold 0.5: predicts ranks 1-4


def _write_crop_stack(root: Path, seq_id: int, track_id: str, n_frames: int):
    seq_dir = root / "processed" / "candidate_crops" / f"seq_{seq_id}"
    seq_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    np.savez(seq_dir / f"{track_id}.npz",
             crops=rng.normal(size=(n_frames, 8, 8)).astype(np.float32),
             positions=np.zeros((n_frames, 2), dtype=np.float32),
             frames=np.asarray([str(i) for i in range(n_frames)]),
             timestamps=np.asarray(["2024-01-01T00:00:00+00:00"] * n_frames),
             valid_fractions=np.ones(n_frames, dtype=np.float32))
    manifest = seq_dir / "crops_manifest.json"
    files = sorted(p.name for p in seq_dir.glob("*.npz"))
    manifest.write_text(json.dumps({"sequence_id": seq_id,
                                    "track_files": files}), encoding="utf-8")


def _synthetic_labels() -> dict:
    return {
        "label_format": "labels_v1",
        "events": [{
            "soho_number": 1, "days": {
                "2024-01-03": {"sequence_id": 3,
                               "positive_track_ids": ["seq3_h_00001"]},
                "2024-01-02": {"sequence_id": 4, "positive_track_ids": []},
            }}],
        "negative_days": [{"day": "2024-01-01", "camera": "C3",
                           "sequence_id": 1}],
        "splits": {"train": ["2024-01-01", "2024-01-02", "2024-01-03"]},
    }


def test_resolve_split_excludes_below_floor_day():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _write_crop_stack(root, 3, "seq3_h_00001", 5)   # positive
        _write_crop_stack(root, 3, "seq3_h_00002", 5)   # negative
        _write_crop_stack(root, 4, "seq4_h_00001", 5)   # below-floor day
        _write_crop_stack(root, 1, "seq1_h_00001", 5)   # negative day
        tracks = resolve_split(_synthetic_labels(), "train", root)
        assert tracks.positives == ["seq3_h_00001"]
        # seq 4 (comet below floor, unlabeled) contributes NO negatives.
        assert sorted(tracks.negatives) == ["seq1_h_00001", "seq3_h_00002"]


def test_dataset_item_shapes_and_padding():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _write_crop_stack(root, 3, "seq3_h_00001", 5)    # shorter than max
        _write_crop_stack(root, 3, "seq3_h_00002", 40)   # longer than max
        _write_crop_stack(root, 1, "seq1_h_00001", 16)
        (root / "dataset" / "v1").mkdir(parents=True)
        (root / "dataset" / "v1" / "labels.json").write_text(
            json.dumps(_synthetic_labels()), encoding="utf-8")
        ds = TrackCropDataset(root, "train", max_frames=16)
        assert len(ds) == 3 and float(ds.labels.sum()) == 1.0
        by_id = {ds[i]["track_id"]: ds[i] for i in range(len(ds))}
        short = by_id["seq3_h_00001"]
        assert short["crops"].shape == (16, 1, 8, 8)
        assert int(short["length"]) == 5
        assert float(short["crops"][5:].abs().sum()) == 0.0  # padded tail
        long = by_id["seq3_h_00002"]
        assert int(long["length"]) == 16 and long["crops"].shape == (16, 1, 8, 8)
        assert float(by_id["seq3_h_00001"]["label"]) == 1.0


def test_model_forward_and_padding_mask():
    torch.manual_seed(0)
    model = TemporalRanker(embed_dim=16, hidden_dim=16)
    crops = torch.randn(3, 10, 1, 8, 8)
    lengths = torch.tensor([10, 4, 7])
    logits = model(crops, lengths)
    assert logits.shape == (3,)
    # Padding must not influence the score: same real frames, garbage tail.
    crops2 = crops.clone()
    crops2[1, 4:] = 99.0
    logits2 = model(crops2, lengths)
    assert torch.allclose(logits[1], logits2[1], atol=1e-6)


def test_model_dna_path_and_null_features():
    torch.manual_seed(0)
    from ml.dataset import DNA_FEATURE_COUNT, _dna_vector
    # Null features map to 0.0 and never NaN.
    vec = _dna_vector({"radial_speed_px_s": None, "n_frames": 12})
    assert vec.shape == (DNA_FEATURE_COUNT,)
    assert np.isfinite(vec).all() and vec[0] == 0.0
    model = TemporalRanker(embed_dim=16, hidden_dim=16,
                           dna_dim=DNA_FEATURE_COUNT)
    crops = torch.randn(2, 5, 1, 8, 8)
    lengths = torch.tensor([5, 3])
    dna = torch.randn(2, DNA_FEATURE_COUNT)
    assert model(crops, lengths, dna).shape == (2,)
    # DNA must actually influence the score.
    assert not torch.allclose(model(crops, lengths, dna),
                              model(crops, lengths, dna + 1.0))
    # dna_dim > 0 without features is an explicit error, not a silent skip.
    try:
        model(crops, lengths)
        assert False, "expected ValueError"
    except ValueError:
        pass


def main() -> int:
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except AssertionError as exc:
                failures += 1
                print(f"FAIL {name}: {exc}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
