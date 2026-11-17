"""Pipeline: runs every implemented stage; later stages are skipped until implemented.

Stages: ingestion -> task inference -> split -> (only when ``cfg.features`` is set)
FeatureFunnel fitted on Train, which adds ``funnel_report.json`` and
``selected_features.json`` to ``out_dir``. Segment ``.npz`` files keep the unfiltered X.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from quantautoml.common.config import PipelineConfig, load_config
from quantautoml.common.exceptions import InvalidDataError
from quantautoml.common.types import Dataset, Segment, SplitResult
from quantautoml.data.ingestion import DataIngestion
from quantautoml.data.splitter import AdaptiveTimeSeriesSplitter
from quantautoml.features.funnel import FeatureFunnel
from quantautoml.models.decision_engine import DecisionReport, ModelDecisionEngine
from quantautoml.models.tournament import ModelTournament, TournamentResult
from quantautoml.models.tuner import DeepOptunaTuner, TuningResult


@dataclass(frozen=True)
class PipelineResult:
    dataset: Dataset
    split: SplitResult
    funnel: FeatureFunnel | None = None
    tournament_result: TournamentResult | None = None
    decision_report: DecisionReport | None = None
    tuning_result: TuningResult | None = None


def run_pipeline(config: PipelineConfig | str | Path, out_dir: str | Path) -> PipelineResult:
    cfg = load_config(config) if isinstance(config, (str, Path)) else config
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    dataset = DataIngestion(cfg).load()
    split = AdaptiveTimeSeriesSplitter().split(dataset, cfg.split.horizon_h)

    _dump_json(out / "ingestion_report.json", dataset.report.to_dict())
    _dump_json(out / "split_report.json", split.report.to_dict())
    _dump_json(
        out / "task.json",
        {
            "task_type": dataset.task_type.value,
            "classes": None if dataset.classes_ is None else dataset.classes_.tolist(),
            "feature_names": list(dataset.feature_names),
        },
    )
    segments = {"train": split.train, "val": split.val, "oos": split.oos}
    if split.cal_a is not None and split.cal_b is not None:
        segments |= {"cal_a": split.cal_a, "cal_b": split.cal_b}
    for name, seg in segments.items():
        _save_segment(out / f"{name}.npz", seg)

    funnel: FeatureFunnel | None = None
    if cfg.features is not None:
        funnel = FeatureFunnel(cfg.features).fit(split, dataset)
        _dump_json(out / "funnel_report.json", [asdict(r) for r in funnel.reports_])
        _dump_json(
            out / "selected_features.json",
            {"selected_features": list(funnel.selected_features_)},
        )

    tournament_result: TournamentResult | None = None
    decision_report: DecisionReport | None = None
    tuning_result: TuningResult | None = None

    if cfg.models is not None:
        if split.train.y is None:
            raise InvalidDataError("Cannot train models without labels (UNSUPERVISED task)")

        train_seg = funnel.transform(split.train) if funnel is not None else split.train

        tournament = ModelTournament(cfg.models, dataset.task_type)
        tournament_result = tournament.run(
            train_seg.X, train_seg.y, embargo_bars=cfg.split.horizon_h
        )
        _dump_json(out / "tournament_report.json", tournament_result.model_dump())

        decision_engine = ModelDecisionEngine(cfg.models)
        decision_report = decision_engine.evaluate(tournament_result, train_seg.y)
        _dump_json(out / "decision_report.json", decision_report.model_dump())

        tuner = DeepOptunaTuner(cfg.models, dataset.task_type)
        tuning_result = tuner.tune(
            decision_report.selected_models,
            train_seg.X,
            train_seg.y,
            embargo_bars=cfg.split.horizon_h,
        )
        tuning_payload = {
            "tuned_candidates": [
                {
                    "model_name": c.model_name,
                    "best_params": c.best_params,
                    "best_score": c.best_score,
                    "best_ic_sharpe": c.best_ic_sharpe,
                    "complexity_score": c.complexity_score,
                }
                for c in tuning_result.tuned_candidates
            ],
            "study_summaries": tuning_result.study_summaries,
        }
        _dump_json(out / "tuning_report.json", tuning_payload)

    return PipelineResult(
        dataset=dataset,
        split=split,
        funnel=funnel,
        tournament_result=tournament_result,
        decision_report=decision_report,
        tuning_result=tuning_result,
    )


def _dump_json(path: Path, payload: dict[str, object] | list[dict[str, object]]) -> None:
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
