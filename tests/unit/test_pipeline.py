"""Pipeline: config -> ingestion -> task inference -> split -> artifacts on disk."""

import json
from pathlib import Path

import numpy as np

from quantautoml.pipeline import run_pipeline


def test_run_pipeline_writes_reports_and_segments(tmp_path: Path, make_frames, build_config) -> None:
    feats, labels = make_frames(n=3000)
    cfg = build_config(feats, labels, horizon_h=21)
    out = tmp_path / "out"

    result = run_pipeline(cfg, out)

    assert result.split.train.X.dtype == np.float32
    for name in ("ingestion_report.json", "split_report.json", "task.json"):
        assert (out / name).is_file()
    assert json.loads((out / "task.json").read_text())["task_type"] == "binary_classification"
    for seg in ("train", "val", "oos"):
        with np.load(out / f"{seg}.npz") as z:
            assert z["X"].dtype == np.float32
            assert z["X"].shape[0] == getattr(result.split, seg).X.shape[0]
