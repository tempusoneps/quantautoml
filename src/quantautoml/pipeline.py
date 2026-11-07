"""Pipeline: runs every implemented stage (currently ingestion -> task inference -> split); later stages are skipped until implemented."""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from quantautoml.common.config import PipelineConfig, load_config
from quantautoml.common.types import Dataset, Segment, SplitResult
from quantautoml.data.ingestion import DataIngestion
from quantautoml.data.splitter import AdaptiveTimeSeriesSplitter
from quantautoml.data.task_inference import infer_task_type


@dataclass(frozen=True)
class PipelineResult:
    dataset: Dataset
    split: SplitResult


def run_pipeline(config: PipelineConfig | str | Path, out_dir: str | Path) -> PipelineResult:
    cfg = load_config(config) if isinstance(config, (str, Path)) else config
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    dataset = DataIngestion(cfg).load()
    task = infer_task_type(dataset.y)
    dataset = replace(dataset, y=task.y, task_type=task.task_type, classes_=task.classes)
    split = AdaptiveTimeSeriesSplitter().split(dataset, cfg.split.horizon_h)

    _dump_json(out / "ingestion_report.json", dataset.report.to_dict())
    _dump_json(out / "split_report.json", split.report.to_dict())
    _dump_json(
        out / "task.json",
        {
            "task_type": task.task_type.value,
            "classes": None if task.classes is None else task.classes.tolist(),
            "feature_names": list(dataset.feature_names),
        },
    )
    segments = {"train": split.train, "val": split.val, "oos": split.oos}
    if split.cal_a is not None and split.cal_b is not None:
        segments |= {"cal_a": split.cal_a, "cal_b": split.cal_b}
    for name, seg in segments.items():
        _save_segment(out / f"{name}.npz", seg)
    return PipelineResult(dataset, split)


def _dump_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _save_segment(path: Path, seg: Segment) -> None:
    arrays: dict[str, np.ndarray] = {
        "X": seg.X,
        "timestamps": seg.timestamps,
        "index": seg.index,
    }
    if seg.y is not None:
        arrays["y"] = seg.y
    np.savez(path, **arrays)
