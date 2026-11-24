"""Pipeline: runs every implemented stage and writes phase-ordered artifacts.

Output layout under ``out_dir``::

    00_run_manifest.json
    01_data/       (01_ingestion_report.json, 02_task.json, 03_split_report.json, segments/*.npz)
    02_features/   (01_funnel_report.json, 02_selected_features.json)
    03_models/     (01_tournament_report.json, 02_decision_report.json, 03_tuning_report.json)
    04_ensemble/   (01_ensemble_report.json, 02_calibration_report.json)
    05_packaging/  (bundle/)
    06_backtest/   (01_backtest_report.json, 02_realworld_report.json, backtest_report.html, trades.csv)

Skipped phase directories are not created. Segment ``.npz`` files keep the unfiltered X.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from quantautoml.backtest import OOSBacktestRunner, OOSMetrics, RealWorldMetrics
from quantautoml.common.config import PipelineConfig, load_config
from quantautoml.common.exceptions import InvalidDataError
from quantautoml.common.types import Dataset, PreprocessorSpec, Segment, SplitResult, TaskType
from quantautoml.data.ingestion import DataIngestion
from quantautoml.data.splitter import AdaptiveTimeSeriesSplitter
from quantautoml.ensemble import (
    BootstrapShrinkageEngine,
    CalibrationResult,
    EnsembleWeightsResult,
    TemperatureCalibrator,
)
from quantautoml.features.funnel import FeatureFunnel
from quantautoml.models.candidates import create_candidate
from quantautoml.models.decision_engine import DecisionReport, ModelDecisionEngine
from quantautoml.models.tournament import ModelTournament, TournamentResult
from quantautoml.models.tuner import DeepOptunaTuner, TunedCandidate, TuningResult
from quantautoml.packaging import BundleManifest, ONNXBundleExporter


@dataclass(frozen=True)
class PipelineResult:
    dataset: Dataset
    split: SplitResult
    funnel: FeatureFunnel | None = None
    tournament_result: TournamentResult | None = None
    decision_report: DecisionReport | None = None
    tuning_result: TuningResult | None = None
    ensemble_result: EnsembleWeightsResult | None = None
    calibration_result: CalibrationResult | None = None
    bundle_manifest: BundleManifest | None = None
    backtest_metrics: OOSMetrics | None = None
    realworld_metrics: RealWorldMetrics | None = None


def run_pipeline(config: PipelineConfig | str | Path, out_dir: str | Path) -> PipelineResult:
    cfg = load_config(config) if isinstance(config, (str, Path)) else config
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    phases: list[dict[str, object]] = []

    # --- Phase 01: Data Ingestion, Task Inference & Adaptive Temporal Split ---
    data_dir = out / "01_data"
    segments_dir = data_dir / "segments"
    dataset = DataIngestion(cfg).load()
    split = AdaptiveTimeSeriesSplitter().split(dataset, cfg.split.horizon_h)

    ingestion_path = data_dir / "01_ingestion_report.json"
    task_path = data_dir / "02_task.json"
    split_path = data_dir / "03_split_report.json"
    _dump_json(ingestion_path, dataset.report.to_dict())
    _dump_json(
        task_path,
        {
            "task_type": dataset.task_type.value,
            "classes": None if dataset.classes_ is None else dataset.classes_.tolist(),
            "feature_names": list(dataset.feature_names),
        },
    )
    _dump_json(split_path, split.report.to_dict())
    segments = {"train": split.train, "val": split.val, "oos": split.oos}
    if split.cal_a is not None and split.cal_b is not None:
        segments |= {"cal_a": split.cal_a, "cal_b": split.cal_b}
    for name, seg in segments.items():
        _save_segment(segments_dir / f"{name}.npz", seg)
    phases.append(
        {
            "phase": "01_data",
            "status": "completed",
            "artifacts": {
                "ingestion_report": _rel(ingestion_path, out),
                "task": _rel(task_path, out),
                "split_report": _rel(split_path, out),
                "segments_dir": _rel(segments_dir, out),
            },
        }
    )

    # --- Phase 02: Leakage-Free Feature Funnel ---
    funnel: FeatureFunnel | None = None
    if cfg.features is not None:
        feat_dir = out / "02_features"
        funnel = FeatureFunnel(cfg.features).fit(split, dataset)
        funnel_path = feat_dir / "01_funnel_report.json"
        selected_path = feat_dir / "02_selected_features.json"
        _dump_json(funnel_path, [asdict(r) for r in funnel.reports_])
        _dump_json(
            selected_path,
            {"selected_features": list(funnel.selected_features_)},
        )
        phases.append(
            {
                "phase": "02_features",
                "status": "completed",
                "artifacts": {
                    "funnel_report": _rel(funnel_path, out),
                    "selected_features": _rel(selected_path, out),
                },
            }
        )
    else:
        phases.append({"phase": "02_features", "status": "skipped", "artifacts": {}})

    # --- Phase 03: Model Tournament, Decision Engine & Deep Optuna Tuning ---
    tournament_result: TournamentResult | None = None
    decision_report: DecisionReport | None = None
    tuning_result: TuningResult | None = None

    if cfg.models is not None:
        if split.train.y is None:
            raise InvalidDataError("Cannot train models without labels (UNSUPERVISED task)")

        models_dir = out / "03_models"
        train_seg = funnel.transform(split.train) if funnel is not None else split.train

        tournament = ModelTournament(cfg.models, dataset.task_type)
        tournament_result = tournament.run(
            train_seg.X, train_seg.y, embargo_bars=cfg.split.horizon_h
        )
        tournament_path = models_dir / "01_tournament_report.json"
        _dump_json(tournament_path, tournament_result.model_dump())

        decision_engine = ModelDecisionEngine(cfg.models)
        decision_report = decision_engine.evaluate(tournament_result, train_seg.y)
        decision_path = models_dir / "02_decision_report.json"
        _dump_json(decision_path, decision_report.model_dump())

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
        tuning_path = models_dir / "03_tuning_report.json"
        _dump_json(tuning_path, tuning_payload)
        phases.append(
            {
                "phase": "03_models",
                "status": "completed",
                "artifacts": {
                    "tournament_report": _rel(tournament_path, out),
                    "decision_report": _rel(decision_path, out),
                    "tuning_report": _rel(tuning_path, out),
                },
            }
        )
    else:
        phases.append({"phase": "03_models", "status": "skipped", "artifacts": {}})

    # --- Phase 04: Convex Ensemble Optimization & Temperature Calibration ---
    ensemble_result: EnsembleWeightsResult | None = None
    calibration_result: CalibrationResult | None = None

    if cfg.ensemble is not None and tuning_result is not None:
        cal_b_seg: Segment | None = None
        cal_a_seg: Segment | None = None
        cal_a_cands: list[TunedCandidate] = tuning_result.tuned_candidates
        cal_b_cands: list[TunedCandidate] = tuning_result.tuned_candidates

        if split.cal_b is not None and split.cal_a is not None:
            cal_b_seg = funnel.transform(split.cal_b) if funnel is not None else split.cal_b
            cal_a_seg = funnel.transform(split.cal_a) if funnel is not None else split.cal_a
        elif (
            split.oof_folds is not None
            and len(split.oof_folds) >= 2
            and split.train.y is not None
            and dataset.task_type is not None
        ):
            # True out-of-fold models for Cal-A and Cal-B (RULE-01: never use refitted_model on Train).
            tr_a, cal_a_idx = split.oof_folds[-2]
            tr_b, cal_b_idx = split.oof_folds[-1]
            train_transformed = funnel.transform(split.train) if funnel is not None else split.train
            assert train_transformed.y is not None
            seed = cfg.models.random_state if cfg.models is not None else cfg.ensemble.random_state
            cal_a_cands = _fit_oof_candidates(
                tuning_result.tuned_candidates,
                train_transformed.X,
                train_transformed.y,
                tr_a,
                dataset.task_type,
                seed,
            )
            cal_b_cands = _fit_oof_candidates(
                tuning_result.tuned_candidates,
                train_transformed.X,
                train_transformed.y,
                tr_b,
                dataset.task_type,
                seed,
            )
            cal_a_seg = Segment(
                X=train_transformed.X[cal_a_idx],
                y=train_transformed.y[cal_a_idx],
                labels_aux={k: v[cal_a_idx] for k, v in train_transformed.labels_aux.items()},
                timestamps=train_transformed.timestamps[cal_a_idx],
                index=train_transformed.index[cal_a_idx],
            )
            cal_b_seg = Segment(
                X=train_transformed.X[cal_b_idx],
                y=train_transformed.y[cal_b_idx],
                labels_aux={k: v[cal_b_idx] for k, v in train_transformed.labels_aux.items()},
                timestamps=train_transformed.timestamps[cal_b_idx],
                index=train_transformed.index[cal_b_idx],
            )

        if (
            cal_b_seg is not None
            and cal_b_seg.y is not None
            and cal_a_seg is not None
            and cal_a_seg.y is not None
        ):
            ensemble_dir = out / "04_ensemble"
            # 1. Collect candidate predictions on Cal-A (earlier chronological slice)
            cal_a_preds = np.column_stack([
                _get_candidate_scores(cand, cal_a_seg.X, dataset.task_type)
                for cand in cal_a_cands
            ])

            # 2. Fit Bootstrap SLSQP Ensemble on Cal-A
            ensemble_engine = BootstrapShrinkageEngine(cfg.ensemble)
            ensemble_result = ensemble_engine.fit(cal_a_preds, cal_a_seg.y)
            ensemble_path = ensemble_dir / "01_ensemble_report.json"
            _dump_json(ensemble_path, ensemble_result.model_dump())

            # 3. Combine predictions on Cal-B (later chronological slice, closest to Val/OOS)
            weights_arr = np.array(ensemble_result.weights, dtype=np.float32)
            if dataset.task_type == TaskType.MULTICLASS_CLASSIFICATION:
                all_probs = [
                    _get_candidate_probs(cand, cal_b_seg.X, dataset.task_type)
                    for cand in cal_b_cands
                ]
                cal_b_comb = np.zeros_like(all_probs[0], dtype=np.float32)
                for w_k, p_k in zip(weights_arr, all_probs):
                    cal_b_comb += w_k * p_k
            else:
                cal_b_preds = np.column_stack([
                    _get_candidate_scores(cand, cal_b_seg.X, dataset.task_type)
                    for cand in cal_b_cands
                ])
                cal_b_comb = (cal_b_preds @ weights_arr).astype(np.float32)

            # 4. Temperature Calibration on Cal-B
            calibrator = TemperatureCalibrator(
                task_type=dataset.task_type,
                random_state=cfg.ensemble.random_state,
            )
            calibrator.fit(cal_b_comb, cal_b_seg.y)
            calibration_result = calibrator.get_result()
            calibration_path = ensemble_dir / "02_calibration_report.json"
            _dump_json(calibration_path, calibration_result.model_dump())
            phases.append(
                {
                    "phase": "04_ensemble",
                    "status": "completed",
                    "artifacts": {
                        "ensemble_report": _rel(ensemble_path, out),
                        "calibration_report": _rel(calibration_path, out),
                    },
                }
            )
        else:
            phases.append({"phase": "04_ensemble", "status": "skipped", "artifacts": {}})
    else:
        phases.append({"phase": "04_ensemble", "status": "skipped", "artifacts": {}})

    # --- Phase 05: ONNX Model Bundle Packaging ---
    bundle_dir = out / "05_packaging" / "bundle"
    bundle_manifest: BundleManifest | None = None
    if (
        cfg.packaging is not None
        and tuning_result is not None
        and ensemble_result is not None
        and calibration_result is not None
    ):
        # Packaging only reads Train (statistics) and Val (precision check); it never fits.
        bundle_manifest = ONNXBundleExporter(cfg.packaging).export(
            out_dir=bundle_dir,
            preprocessor_spec=(
                funnel.preprocessor_spec()
                if funnel is not None
                else _identity_spec(dataset.feature_names)
            ),
            candidates=tuning_result.tuned_candidates,
            ensemble=ensemble_result,
            calibration=calibration_result,
            task_type=dataset.task_type,
            input_features=dataset.feature_names,
            classes=_model_classes(tuning_result.tuned_candidates[0]),
            X_train_raw=split.train.X,
            X_val_raw=split.val.X,
            X_val_model=funnel.transform(split.val).X if funnel is not None else split.val.X,
        )
        phases.append(
            {
                "phase": "05_packaging",
                "status": "completed",
                "artifacts": {
                    "bundle_dir": _rel(bundle_dir, out),
                    "bundle_manifest": _rel(bundle_dir / "bundle_manifest.json", out),
                },
            }
        )
    else:
        phases.append({"phase": "05_packaging", "status": "skipped", "artifacts": {}})

    # --- Phase 06: Friction-Aware Blind OOS Backtesting ---
    backtest_metrics: OOSMetrics | None = None
    realworld_metrics: RealWorldMetrics | None = None
    if cfg.backtest is not None and cfg.packaging is not None and bundle_manifest is not None:
        # Blind OOS acceptance: the only stage allowed to read split.oos for evaluation.
        backtest_dir = out / "06_backtest"
        backtest_dir.mkdir(parents=True, exist_ok=True)
        backtest_path = backtest_dir / "01_backtest_report.json"
        others = [split.train, split.val] + [s for s in (split.cal_a, split.cal_b) if s is not None]
        ohlcv_df = None
        rw_cfg = cfg.realworld
        if rw_cfg is not None and rw_cfg.enabled:
            ohlcv_df = _extract_ohlcv_df(split.oos, dataset.feature_names, rw_cfg)
            if ohlcv_df is not None:
                rw_cfg = rw_cfg.model_copy(
                    update={
                        "out_html": rw_cfg.out_html or str(backtest_dir / "backtest_report.html"),
                        "out_trades_csv": rw_cfg.out_trades_csv or str(backtest_dir / "trades.csv"),
                    }
                )
        backtest_metrics, realworld_metrics = OOSBacktestRunner(cfg.backtest).run(
            bundle_dir=bundle_dir,
            oos=split.oos,
            feature_names=dataset.feature_names,
            other_segments=others,
            out_path=backtest_path,
            latency_repeats=cfg.packaging.latency_repeats,
            latency_budget_ms=cfg.packaging.latency_budget_ms,
            ohlcv_df=ohlcv_df,
            realworld_cfg=rw_cfg,
        )
        bt_artifacts: dict[str, str] = {"backtest_report": _rel(backtest_path, out)}
        if realworld_metrics is not None:
            realworld_path = backtest_dir / "02_realworld_report.json"
            _dump_json(realworld_path, realworld_metrics.model_dump())
            bt_artifacts["realworld_report"] = _rel(realworld_path, out)
        phases.append({"phase": "06_backtest", "status": "completed", "artifacts": bt_artifacts})
    else:
        phases.append({"phase": "06_backtest", "status": "skipped", "artifacts": {}})

    _dump_json(out / "00_run_manifest.json", {"phases": phases})

    return PipelineResult(
        dataset=dataset,
        split=split,
        funnel=funnel,
        tournament_result=tournament_result,
        decision_report=decision_report,
        tuning_result=tuning_result,
        ensemble_result=ensemble_result,
        calibration_result=calibration_result,
        bundle_manifest=bundle_manifest,
        backtest_metrics=backtest_metrics,
        realworld_metrics=realworld_metrics,
    )


def _extract_ohlcv_df(
    oos: Segment, feature_names: tuple[str, ...], rw_cfg: object
) -> object | None:
    """Build an OHLCV DataFrame from raw OOS features when all 5 columns are present."""
    import pandas as pd

    cols = (
        rw_cfg.ohlcv_open_col,  # type: ignore[attr-defined]
        rw_cfg.ohlcv_high_col,  # type: ignore[attr-defined]
        rw_cfg.ohlcv_low_col,  # type: ignore[attr-defined]
        rw_cfg.ohlcv_close_col,  # type: ignore[attr-defined]
        rw_cfg.ohlcv_volume_col,  # type: ignore[attr-defined]
    )
    if not all(c in feature_names for c in cols):
        return None
    name_to_idx = {name: i for i, name in enumerate(feature_names)}
    return pd.DataFrame(
        {c: oos.X[:, name_to_idx[c]].astype(np.float64) for c in cols},
        index=pd.to_datetime(oos.timestamps),
    )


def _identity_spec(feature_names: tuple[str, ...]) -> PreprocessorSpec:
    """Preprocessor for a pipeline without a feature funnel: models see every raw column."""
    idx = tuple(range(len(feature_names)))
    return PreprocessorSpec(
        n_in=len(idx), bool_fill=(), indicator_idx=(), keep_idx=idx, final_idx=idx,
        output_names=tuple(feature_names),
    )


def _fit_oof_candidates(
    candidates: list[TunedCandidate],
    X_train: np.ndarray,
    y_train: np.ndarray,
    tr_idx: np.ndarray,
    task_type: TaskType,
    random_state: int,
) -> list[TunedCandidate]:
    """Fit fresh clones of tuned candidates strictly on ``tr_idx`` for true OOF predictions."""
    oof_list: list[TunedCandidate] = []
    for cand in candidates:
        model = create_candidate(
            name=cand.model_name,
            task_type=task_type,
            params=cand.best_params,
            random_state=random_state,
        )
        model.fit(X_train[tr_idx], y_train[tr_idx])
        oof_list.append(cand.model_copy(update={"fitted_model": model}))
    return oof_list


def _model_classes(cand: TunedCandidate) -> list[int]:
    classes = cand.fitted_model.classes_
    return [] if classes is None else [int(c) for c in classes]


def _get_candidate_scores(
    cand: TunedCandidate,
    X: np.ndarray,
    task_type: TaskType | None,
) -> np.ndarray:
    """Unified continuous ranking score matching BaseGBDCandidate.predict & Predictor.predict."""
    return cand.fitted_model.predict(X).astype(np.float32)


def _get_candidate_probs(
    cand: TunedCandidate,
    X: np.ndarray,
    task_type: TaskType | None,
) -> np.ndarray:
    if task_type in (TaskType.BINARY_CLASSIFICATION, TaskType.MULTICLASS_CLASSIFICATION):
        probs = cand.fitted_model.predict_proba(X)
        if task_type == TaskType.BINARY_CLASSIFICATION and probs.ndim == 2 and probs.shape[1] >= 2:
            return probs[:, 1].astype(np.float32)
        return probs.astype(np.float32)
    return cand.fitted_model.predict(X).astype(np.float32)


def _rel(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def _dump_json(path: Path, payload: dict[str, object] | list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _save_segment(path: Path, seg: Segment) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, np.ndarray] = {
        "X": seg.X,
        "timestamps": seg.timestamps,
        "index": seg.index,
    }
    if seg.y is not None:
        arrays["y"] = seg.y
    np.savez(path, **arrays)
