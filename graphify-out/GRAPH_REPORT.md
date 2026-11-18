# Graph Report - quantautoml  (2026-11-18)

## Corpus Check
- 136 files · ~94,275 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 1323 nodes · 4106 edges · 63 communities (53 shown, 10 thin omitted)
- Extraction: 64% EXTRACTED · 36% INFERRED · 0% AMBIGUOUS · INFERRED: 1471 edges (avg confidence: 0.65)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `b1f87a22`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- InvalidDataError
- FitContext
- types.py
- .fit
- Core Hard Gates RULE-01..10
- calc_rank_ic
- make_frames
- BaseGBDCandidate
- load_config
- test_regime_synergy.py
- 7-Stage Feature Funnel (F1-F8)
- .transform
- test_clustering.py
- test_collinearity.py
- test_jmi.py
- test_variance.py
- infer_task_type
- test_ic_stability.py
- ModelsConfig
- Temperature Scaling (RULE-03, rank invariant)
- What You Must Do When Invoked
- EnsembleConfig
- TemperatureCalibrator
- exceptions.py
- Git Conventions & Branching
- Detailed Task Breakdown
- create_candidate
- float32
- TaskType
- calc_ece
- DeepOptunaTuner
- SmoothSLSQPOptimizer
- Detailed Task Breakdown
- 2. Đặc Tả Chi Tiết 7 Tầng Lọc Tính Năng
- Detailed Task Breakdown
- 2. Chuẩn Commit Message (Conventional Commits)
- config.py
- graphify reference: extra exports and benchmark
- AI Agent Guidelines — Quant AutoML Pipeline
- _calc_series_ic_sharpe
- graphify reference: query, path, explain
- Step 04 — Design: Deep Optuna Tuning
- test_zero_leakage_ensemble.py
- .optimize
- graphify reference: add a URL and watch a folder
- graphify reference: commit hook and native CLAUDE.md integration
- graphify reference: incremental update and cluster-only
- Project Rules & Invariants — Quant AutoML Pipeline
- Step 05 — Scope: Convex Ensemble & Probability Calibration
- Step 06 — Scope: ONNX Model Bundle Packaging
- Self
- graphify reference: GitHub clone and cross-repo merge
- graphify reference: transcribe video and audio
- Step 03 — Scope: Model Tournament & Decision Engine
- rules/graphify.md
- extraction-spec.md
- workflows/graphify.md
- generate_agents_markdown.sh
- quantautoml

## God Nodes (most connected - your core abstractions)
1. `InvalidDataError` - 217 edges
2. `FitContext` - 100 edges
3. `FeaturesConfig` - 91 edges
4. `TaskType` - 86 edges
5. `BaseFeatureFilter` - 67 edges
6. `FeatureSelectionError` - 65 edges
7. `FilterReport` - 62 edges
8. `ModelsConfig` - 54 edges
9. `FeatureFunnel` - 47 edges
10. `Dataset` - 39 edges

## Surprising Connections (you probably didn't know these)
- `Six-Layer Unidirectional Architecture` --semantically_similar_to--> `System Architecture`  [INFERRED] [semantically similar]
  CLAUDE.md → docs/plan/ARCHITECTURE.md
- `FeaturesConfig` --uses--> `Pipeline: config -> ingestion -> task inference -> split -> artifacts on disk.`  [INFERRED]
  src/quantautoml/common/config.py → tests/unit/test_pipeline.py
- `ModelsConfig` --calls--> `test_invalid_diversity_threshold_rejected()`  [INFERRED]
  src/quantautoml/common/config.py → tests/unit/test_models_config.py
- `ModelsConfig` --calls--> `test_invalid_enabled_models_rejected()`  [INFERRED]
  src/quantautoml/common/config.py → tests/unit/test_models_config.py
- `ModelsConfig` --calls--> `test_invalid_trials_rejected()`  [INFERRED]
  src/quantautoml/common/config.py → tests/unit/test_models_config.py

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **Leakage and rank-safety gates** — claude_rule01, claude_rule03, claude_rule04, claude_rule08 [INFERRED 0.75]
- **Five-phase pipeline** — docs_plan_phases_phase1, docs_plan_phases_phase2, docs_plan_phases_phase3, docs_plan_phases_phase4, docs_plan_phases_phase5 [EXTRACTED 0.90]
- **Feature funnel stages F1-F8** — s02_f1_triage, s02_f2_variance, s02_f3_ic_stability, s02_f4_fast_dcor, s02_f5_clustering, s02_f6_regime_synergy, s02_f8_jmi [EXTRACTED 1.00]
- **SLSQP, bootstrap shrinkage and temperature calibration flow** — s05_slsqp_optimizer, s05_bootstrap_shrinkage, s05_temperature_scaling, s05_rank_pearson_objective [EXTRACTED 0.95]
- **Friction-aware OOS acceptance** — s07_backtest_runner, s07_friction_cost_model, s07_hysteresis_buffer, s07_oos_acceptance_gate [EXTRACTED 0.95]

## Communities (63 total, 10 thin omitted)

### Community 0 - "InvalidDataError"
Cohesion: 0.11
Nodes (29): InvalidDataError, Input data or configuration violates a pipeline data contract., ModelTournament, Any, BaseModel, float32, int32, NDArray (+21 more)

### Community 1 - "FitContext"
Cohesion: 0.05
Nodes (99): BaseFeatureFilter, MissingKind, FeaturesConfig, FeatureSelectionError, Feature funnel cannot be fit or applied (e.g. unfitted filter, empty result)., FilterReport, BaseFeatureFilter, FitContext (+91 more)

### Community 2 - "types.py"
Cohesion: 0.06
Nodes (96): Enum, IndexArray, PipelineConfig, DataLeakageError, Timestamps are non-monotonic, duplicated or cannot be aligned., A look-ahead leakage invariant was violated (RULE-01)., TimestampAlignmentError, Dataset (+88 more)

### Community 3 - ".fit"
Cohesion: 0.08
Nodes (63): _fit(), int32, Self, FeatureFunnel, Self, TempPathFactory, Case, _load() (+55 more)

### Community 4 - "Core Hard Gates RULE-01..10"
Cohesion: 0.06
Nodes (56): AGENTS.md Agent Guide, Agent TDD Workflow & Session Onboarding, CLAUDE.md Agent Guide, Six-Layer Unidirectional Architecture, Senior Quant Developer Persona, RULE-01 Zero Look-Ahead Leakage, RULE-02 Rank Metric Consistency, RULE-03 No Isotonic; Temperature Scaling (+48 more)

### Community 5 - "calc_rank_ic"
Cohesion: 0.09
Nodes (40): calc_ic_sharpe(), calc_icir(), calc_rank_ic(), calc_rolling_ic(), calc_rolling_ic_matrix(), float32, float64, int32 (+32 more)

### Community 6 - "make_frames"
Cohesion: 0.07
Nodes (69): ArgumentParser, datetime64, build_config_content(), detect_file_info(), main(), any, Path, _build_parser() (+61 more)

### Community 7 - "BaseGBDCandidate"
Cohesion: 0.13
Nodes (23): ABC, ModelError, Base class for model training, tournament, and tuning failures., Quantitative metrics shared across pipeline layers., BaseGBDCandidate, CatBoostCandidate, _clip01(), _complexity() (+15 more)

### Community 8 - "load_config"
Cohesion: 0.14
Nodes (30): load_config(), Path, Path, write_toml(), Path, d = 1 - |corr| <= 1 always, so a cut at 1.0 would merge every feature., test_cluster_distance_open_unit_interval(), test_defaults_are_applied() (+22 more)

### Community 9 - "test_regime_synergy.py"
Cohesion: 0.14
Nodes (39): label_regimes(), int8, _cfg(), _ctx(), _fit(), _noise(), float32, float64 (+31 more)

### Community 10 - "7-Stage Feature Funnel (F1-F8)"
Cohesion: 0.06
Nodes (34): AdaptiveTimeSeriesSplitter (Purging & Embargo), UD-02 Boolean to Float32 auto-cast, DataIngestion (UTC us, float32, inner join), horizon_h Purging for forward-looking labels, infer_task_type (Binary/Multiclass/Unsupervised), NEED_UPDATE: design corrections (timestamp precision, bool/string, mid-NaN labels), Adaptive OOF-Assisted Split (N<10000), PipelineConfig (Pydantic TOML) (+26 more)

### Community 11 - ".transform"
Cohesion: 0.17
Nodes (34): float32, NDArray, _transform(), test_transform_all_segments_same_columns_float32(), test_transform_wrong_width_raises(), _cfg(), _ctx(), _fit() (+26 more)

### Community 12 - "test_clustering.py"
Cohesion: 0.14
Nodes (36): ic_behavior_distance(), bool_, float32, float64, int32, intp, NDArray, _cfg() (+28 more)

### Community 13 - "test_collinearity.py"
Cohesion: 0.15
Nodes (28): fast_dcor(), _cfg(), _f32(), _fit(), float32, float64, int32, MonkeyPatch (+20 more)

### Community 14 - "test_jmi.py"
Cohesion: 0.21
Nodes (26): _cfg(), _ctx(), _fit(), _noise(), float32, float64, int32, MonkeyPatch (+18 more)

### Community 15 - "test_variance.py"
Cohesion: 0.21
Nodes (22): _cfg(), _ctx(), _fit(), _normal(), float32, int32, NDArray, test_all_nan_column_dropped_no_warning() (+14 more)

### Community 16 - "infer_task_type"
Cohesion: 0.20
Nodes (17): generic, infer_task_type(), NDArray, test_binary_minus_one_one_is_remapped_to_zero_one(), test_binary_string_labels(), test_binary_zero_one_float_labels(), test_empty_labels_are_rejected(), test_inf_labels_are_rejected() (+9 more)

### Community 17 - "test_ic_stability.py"
Cohesion: 0.10
Nodes (35): apply_fdr_correction(), bool_, float64, NDArray, Benjamini-Hochberg FDR control (RULE-08)., Return a mask of accepted hypotheses, in the original order.      k = max{i : p_, _cfg(), _ctx() (+27 more)

### Community 18 - "ModelsConfig"
Cohesion: 0.15
Nodes (27): BaseModel, ModelsConfig, DecisionReport, DecisionType, ModelDecisionEngine, BaseModel, float32, int32 (+19 more)

### Community 19 - "Temperature Scaling (RULE-03, rank invariant)"
Cohesion: 0.17
Nodes (13): Isotonic Regression (rejected: ties), Platt Scaling, Temperature Scaling (RULE-03, rank invariant), Rationale: Temperature Scaling preserves 100% Rank IC, DistributionShiftWarning (z>3), ONNXBundleExporter, PrecisionValidator (|y_py-y_onnx|<1e-4, RULE-06), predictor.py decoupled inference (numpy+onnxruntime, RULE-07) (+5 more)

### Community 20 - "What You Must Do When Invoked"
Cohesion: 0.08
Nodes (24): For /graphify add and --watch, For /graphify query, For the commit hook and native CLAUDE.md integration, For --update and --cluster-only, /graphify, Honesty Rules, Interpreter guard for subcommands, Part A - Structural extraction for code files (+16 more)

### Community 21 - "EnsembleConfig"
Cohesion: 0.14
Nodes (21): EnsembleConfig, BootstrapShrinkageEngine, EnsembleWeightsResult, BaseModel, Bootstrap Shrinkage Engine and Hurdle Fallback Gate (RULE-05)., Result contract of the ensemble weight optimization and hurdle evaluation., Bootstrap weight distribution estimator with shrinkage and hurdle guard rail., RULE-01: Mutating split.val or split.oos MUST NOT affect ensemble weights or cal (+13 more)

### Community 22 - "TemperatureCalibrator"
Cohesion: 0.12
Nodes (19): CalibrationResult, BaseModel, float32, int32, NDArray, Self, Apply learned temperature scaling to input probabilities.          Parameters, Verify that temperature calibration strictly preserved 100% of ranking order. (+11 more)

### Community 23 - "exceptions.py"
Cohesion: 0.12
Nodes (21): Exception, CalibrationError, EnsembleError, OptimizationConvergenceError, Exception, QuantAutoMLError, Custom exceptions shared across the quantautoml pipeline., Tournament evaluation or candidate selection failed. (+13 more)

### Community 25 - "Detailed Task Breakdown"
Cohesion: 0.10
Nodes (19): Detailed Task Breakdown, Execution Order Overview, Final Verification Checklist, Global Constraints, Implementation Plan: Quant AutoML Pipeline, Quyết định thiết kế (cần bạn xác nhận), Review Focus, [x] Task 10: F6 — Regime-Aware MI & Synergy (+11 more)

### Community 26 - "create_candidate"
Cohesion: 0.16
Nodes (14): create_candidate(), Any, Factory creating a unified GBDT candidate wrapper., Predict continuous scores suitable for Spearman Rank IC evaluation.          For, Return complexity penalty score in (0, 1] based on tree depth and count., Early walk-forward folds may lack a class; labels {0, 2} must still fit and scor, test_candidates_fit_predict_binary(), test_candidates_fit_predict_multiclass() (+6 more)

### Community 27 - "float32"
Cohesion: 0.28
Nodes (7): float32, int32, NDArray, Self, Fit model on training segment. Both X and y must be numpy arrays., Predict class probabilities of shape (n_samples, n_classes) as float32., Map labels to contiguous 0..k-1 (required by XGBoost) and remember the originals

### Community 28 - "TaskType"
Cohesion: 0.24
Nodes (13): TaskType, Infer the learning task (binary / multiclass / unsupervised) from the label vect, Infer the task. Missing labels (NaN / None) are unlabeled rows: they are     ign, TaskInference, Set `output_names_` and `report_`., Apply the fitted selection without mutating state., BaseModel, A candidate model that has been fine-tuned and refitted on the full training set (+5 more)

### Community 29 - "calc_ece"
Cohesion: 0.19
Nodes (12): calc_ece(), float32, int32, NDArray, Metrics for evaluating probability calibration error (ECE)., Calculate the Expected Calibration Error (ECE) for binary or multiclass probabil, Unit tests for Expected Calibration Error (ECE) metric., test_calc_ece_empty_or_invalid() (+4 more)

### Community 30 - "DeepOptunaTuner"
Cohesion: 0.20
Nodes (11): DeepOptunaTuner, float32, int32, NDArray, Deep Bayesian hyperparameter tuner using Multivariate TPE and HyperbandPruner., Run deep tuning for each model in models_to_tune.          Evaluates on walk-for, test_tuner_empty_models_rejected(), test_tuner_hyperband_prunes_weak_trials() (+3 more)

### Community 31 - "SmoothSLSQPOptimizer"
Cohesion: 0.21
Nodes (10): Convex ensemble optimization and probability calibration., Smooth SLSQP Convex Optimization on Rank-Transformed Data (C2 Differentiable Obj, SLSQP convex weight optimizer with C2 differentiable objective.      Solves for, SmoothSLSQPOptimizer, Unit tests for SmoothSLSQPOptimizer (C2 Objective)., test_slsqp_convergence_speed(), test_slsqp_fallback_on_constant_data(), test_slsqp_input_validation() (+2 more)

### Community 32 - "Detailed Task Breakdown"
Cohesion: 0.15
Nodes (12): Detailed Task Breakdown, Execution Order Overview, Final Verification Checklist, Global Constraints, Implementation Plan: Quant AutoML Pipeline — Phase 3, [x] Task 1: Dependencies, Exceptions & ModelsConfig, [x] Task 2: Quantitative Metrics — IC Sharpe & Walk-Forward Validation Helper, [x] Task 3: Modern GBDT Trio Pool (Candidates Wrappers) (+4 more)

### Community 33 - "2. Đặc Tả Chi Tiết 7 Tầng Lọc Tính Năng"
Cohesion: 0.17
Nodes (11): 1. Kiến Trúc Tổng Thể & Luồng Dữ Liệu (Funnel Architecture), 2.1. Tầng F1: NaN/Inf Triage & MI Score (`triage.py`), 2.2. Tầng F2: Multi-Metric Variance Check (`variance.py`), 2.3. Tầng F3+7: IC Stability, FDR Gate & Term Structure (`ic_stability.py`), 2.4. Tầng F4: Fast Nonlinear Collinearity (`collinearity.py`), 2.5. Tầng F5: IC-Behavior Hierarchical Clustering (`clustering.py`), 2.6. Tầng F6: Regime-Aware MI & Synergy Detection (`regime_synergy.py`), 2.7. Tầng F8: Joint Mutual Information Selector (`jmi.py`) (+3 more)

### Community 34 - "Detailed Task Breakdown"
Cohesion: 0.17
Nodes (11): Detailed Task Breakdown, Execution Order Overview, Final Verification Checklist, Global Constraints & Hard Gates, Implementation Plan: Quant AutoML Pipeline — Phase 4, [x] Task 1: Ensemble Config & Custom Exceptions, [x] Task 2: Quantitative Metrics — Expected Calibration Error (ECE), [x] Task 3: Smooth SLSQP Optimizer ($C^2$ Objective) (+3 more)

### Community 35 - "2. Chuẩn Commit Message (Conventional Commits)"
Cohesion: 0.20
Nodes (9): 1.1. Các Nhánh Chính Của Dự Án (Core Branches), 1.2. Nhánh Dành Riêng Cho Contributor (Contributor Branches Only), 1. Cấu Trúc Nhánh (Branching Strategy), 2.1. Danh mục Types:, 2.2. Danh mục Scopes chuẩn của dự án:, 2.3. Ví dụ chuẩn:, 2. Chuẩn Commit Message (Conventional Commits), 3. Tần Suất Commit & Quy Tắc Vệ Sinh (Git Hygiene) (+1 more)

### Community 36 - "config.py"
Cohesion: 0.27
Nodes (6): DataConfig, LabelsConfig, BaseModel, Pipeline configuration loaded from a TOML file and validated with Pydantic., SplitConfig, _Strict

### Community 37 - "graphify reference: extra exports and benchmark"
Cohesion: 0.22
Nodes (8): graphify reference: extra exports and benchmark, Step 6b - Wiki (only if --wiki flag), Step 7 - Neo4j export (only if --neo4j or --neo4j-push flag), Step 7a - FalkorDB export (only if --falkordb or --falkordb-push flag), Step 7b - SVG export (only if --svg flag), Step 7c - GraphML export (only if --graphml flag), Step 7d - MCP server (only if --mcp flag), Step 8 - Token reduction benchmark (only if total_words > 5000)

### Community 38 - "AI Agent Guidelines — Quant AutoML Pipeline"
Cohesion: 0.22
Nodes (8): 1. Vai Trò & Tôn Chỉ Kỹ Thuật (Persona & Core Philosophy), 2.1. Giao Thức Khởi Động Phiên Làm Việc (Session Onboarding & State Discovery Protocol), 2.2. Giao Thức Quản Lý Vòng Đời Trạng Thái (State Lifecycle Management), 2.3. Các bước bắt buộc khi thực hiện task:, 2. Quy Trình Làm Việc Tiêu Chuẩn Cho Agent (Workflow), 3. Nguyên Tắc An Toàn Định Lượng (Quant Safety Directives), 4. Hướng Dẫn Debug & Xử Lý Sự Cố (Root-Cause Debugging), AI Agent Guidelines — Quant AutoML Pipeline

### Community 39 - "_calc_series_ic_sharpe"
Cohesion: 0.43
Nodes (6): _calc_series_ic_sharpe(), float32, int32, NDArray, Helper to evaluate IC Sharpe or fallback to Rank IC on small sample slices., Fit ensemble weights on Cal-B using Bootstrap and Shrinkage.          Parameters

### Community 40 - "graphify reference: query, path, explain"
Cohesion: 0.33
Nodes (5): For /graphify explain, For /graphify path, graphify reference: query, path, explain, Step 0 — Constrained query expansion (REQUIRED before traversal), Step 1 — Traversal

### Community 41 - "Step 04 — Design: Deep Optuna Tuning"
Cohesion: 0.33
Nodes (5): Công nghệ, Data flow, Ràng buộc, Source layout, Step 04 — Design: Deep Optuna Tuning

### Community 42 - "test_zero_leakage_ensemble.py"
Cohesion: 0.33
Nodes (5): _make_dummy_dataset(), Path, Integration test: Zero Look-Ahead Leakage for Phase 4 Ensemble & Calibration (RU, Test full pipeline end-to-end integration with ensemble & calibration reports., test_pipeline_with_ensemble_and_calibration()

### Community 43 - ".optimize"
Cohesion: 0.40
Nodes (4): float32, int32, NDArray, Optimize ensemble weights on Cal-B predictions.          Parameters         ----

### Community 44 - "graphify reference: add a URL and watch a folder"
Cohesion: 0.50
Nodes (3): For /graphify add, For --watch, graphify reference: add a URL and watch a folder

### Community 45 - "graphify reference: commit hook and native CLAUDE.md integration"
Cohesion: 0.50
Nodes (3): For git commit hook, For native CLAUDE.md integration, graphify reference: commit hook and native CLAUDE.md integration

### Community 46 - "graphify reference: incremental update and cluster-only"
Cohesion: 0.50
Nodes (3): For --cluster-only, For --update (incremental re-extraction), graphify reference: incremental update and cluster-only

### Community 47 - "Project Rules & Invariants — Quant AutoML Pipeline"
Cohesion: 0.50
Nodes (3): 2. Tiêu Chuẩn Lập Trình & Chất Lượng Mã Nguồn (Code Quality Standards), 🚨 BẢNG QUY TẮC CỐT LÕI (CORE HARD GATES), Project Rules & Invariants — Quant AutoML Pipeline

### Community 48 - "Step 05 — Scope: Convex Ensemble & Probability Calibration"
Cohesion: 0.50
Nodes (3): In-Scope (Must Have), Ràng buộc liên quan, Step 05 — Scope: Convex Ensemble & Probability Calibration

### Community 49 - "Step 06 — Scope: ONNX Model Bundle Packaging"
Cohesion: 0.50
Nodes (3): In-Scope (Must Have), Ràng buộc liên quan, Step 06 — Scope: ONNX Model Bundle Packaging

## Knowledge Gaps
- **145 isolated node(s):** `quantautoml`, `generate_agents_markdown.sh script`, `graphify`, `Usage`, `What graphify is for` (+140 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **10 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `InvalidDataError` connect `InvalidDataError` to `FitContext`, `types.py`, `.fit`, `calc_rank_ic`, `make_frames`, `BaseGBDCandidate`, `test_regime_synergy.py`, `test_clustering.py`, `test_collinearity.py`, `test_variance.py`, `infer_task_type`, `test_ic_stability.py`, `ModelsConfig`, `exceptions.py`, `create_candidate`, `float32`, `TaskType`, `DeepOptunaTuner`?**
  _High betweenness centrality (0.199) - this node is a cross-community bridge._
- **Why does `run_pipeline()` connect `make_frames` to `InvalidDataError`, `types.py`, `.fit`, `load_config`, `test_zero_leakage_ensemble.py`, `ModelsConfig`, `EnsembleConfig`, `TemperatureCalibrator`, `TaskType`, `DeepOptunaTuner`?**
  _High betweenness centrality (0.039) - this node is a cross-community bridge._
- **Why does `FitContext` connect `FitContext` to `InvalidDataError`, `types.py`, `.fit`, `BaseGBDCandidate`, `test_regime_synergy.py`, `.transform`, `test_clustering.py`, `test_collinearity.py`, `test_jmi.py`, `test_variance.py`, `test_ic_stability.py`, `TaskType`?**
  _High betweenness centrality (0.036) - this node is a cross-community bridge._
- **Are the 191 inferred relationships involving `InvalidDataError` (e.g. with `DataIngestion` and `.load()`) actually correct?**
  _`InvalidDataError` has 191 INFERRED edges - model-reasoned connections that need verification._
- **Are the 68 inferred relationships involving `FitContext` (e.g. with `FeatureSelectionError` and `InvalidDataError`) actually correct?**
  _`FitContext` has 68 INFERRED edges - model-reasoned connections that need verification._
- **Are the 58 inferred relationships involving `FeaturesConfig` (e.g. with `ICBehaviorClusterFilter` and `F5: IC-behavior hierarchical clustering with a Max-MI representative per cluster`) actually correct?**
  _`FeaturesConfig` has 58 INFERRED edges - model-reasoned connections that need verification._
- **Are the 67 inferred relationships involving `TaskType` (e.g. with `Infer the learning task (binary / multiclass / unsupervised) from the label vect` and `Infer the task. Missing labels (NaN / None) are unlabeled rows: they are     ign`) actually correct?**
  _`TaskType` has 67 INFERRED edges - model-reasoned connections that need verification._