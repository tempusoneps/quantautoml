# Graph Report - quantautoml  (2026-11-24)

## Corpus Check
- 276 files · ~870,193 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 3541 nodes · 7816 edges · 213 communities (201 shown, 12 thin omitted)
- Extraction: 81% EXTRACTED · 19% INFERRED · 0% AMBIGUOUS · INFERRED: 1460 edges (avg confidence: 0.68)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `ce22aae7`
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
- bootstrap.py
- Detailed Task Breakdown
- 2. Đặc Tả Chi Tiết 7 Tầng Lọc Tính Năng
- Detailed Task Breakdown
- 2. Chuẩn Commit Message (Conventional Commits)
- config.py
- graphify reference: extra exports and benchmark
- AI Agent Guidelines — Quant AutoML Pipeline
- Quantitative metrics shared across pipeline layers.
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
- README.md
- verify_geometry.py
- DeepOptunaTuner
- drawio_extract.py
- BacktestError
- Step 08-fix1 — Design: Dual-Tier Realistic Backtest (backtesting.py Integration)
- exceptions.py
- FeatureFunnel
- contingency_mi
- Onboarding — generate your skin from a design source
- Ridgeline
- 2. Layout formulas — deterministic geometry
- excalidraw_extract.py
- evaluate_oos_performance
- DataIngestion
- Process
- runner.py
- FeatureSchemaContract
- Data Flow
- IT current-state
- Beeswarm
- DP security matrix
- Diagram Design
- plantuml_extract.py
- HysteresisBuffer
- validate_numerical_precision
- apply_fdr_correction
- Brand
- DP integration
- Medallion
- export_svg_document
- str
- export_svg.py
- digest
- Diagram
- OOSBacktestRunner
- test_zero_leakage_packaging.py
- _edge_split
- Client profiles
- Style Guide
- test_packaging_bundle.py
- Optional animation
- css_escape
- .predict_proba
- test_zero_leakage.py
- Sequence
- Marimekko
- Export to PNG / SVG
- layout-budget.md
- Draw.io import output spec — format × size × detail × audience
- Architecture delta
- Loop
- _fail
- build_preprocessor_onnx
- test_runner_tier2_integration.py
- Import from PlantUML
- _rewrite_start_tags
- Import from draw.io
- Import from Mermaid
- Data stack
- Semantic patterns
- Exploded axonometric
- Polar Chart
- Steps
- System Architecture & Technical Specifications: Quant AutoML Pipeline
- ._fit
- .transform
- ._fit
- test_zero_leakage_models.py
- Import from Excalidraw
- Icons (primitive)
- Arc diagram
- _CssRule
- .run
- test_backtest_strategy.py
- test_packaging_converters.py
- Axonometric plan
- Consultant special (2×2 scenario matrix)
- RegimeSynergyFilter
- Data
- Network
- Bar / Column Chart
- Database Schema
- User Journey Map
- Action
- Core SVG primitives
- Architecture
- Radar / Spider
- Sankey / Flow-Quantity
- Service Blueprint
- Unit Grid
- ._fit
- Environment doctor
- DevOps
- Compute
- Kubernetes
- Sketchy Filter (hand-drawn variant)
- Terminal Window (CLI-chrome variant)
- test_pipeline_packaging.py
- Heatmap
- Segment
- User Story Map
- UML Class Diagram
- Waterfall Chart
- _parse_declaration_tail
- Node
- CalibrationError
- Statistical tools
- Event modeling
- Gantt Chart
- Org Chart / Responsibility Map
- Pyramid / Funnel
- Requirement traceability
- Wardley Map
- ._fit
- Annotation Callout (italic-serif aside)
- People
- PipelineConfig
- State Machine
- Venn / Set Overlap
- export_png.py
- ER / Data Model
- Flowchart
- Layer Stack
- Nested Containment
- Swimlane
- Timeline
- Tree / Hierarchy
- merge_style_into_defs
- Configuration (`config.toml`)
- calc_rolling_ic
- 2. Sinh File Cấu Hình `config.toml` Tự Động (`scripts/generate_config.py`)
- ic.py
- Export block registry (`--registry`)
- System Architecture & Technical Specifications: Quant AutoML Pipeline
- People
- Danh Mục Báo Cáo & Artifacts Đầu Ra (`OUTPUTS`)
- Repository & Codebase Structure: Quant AutoML Pipeline
- 4. Mathematical Formulations & Optimization Engines
- Quant AutoML Pipeline
- generate_readme.sh
- extract_first_svg
- Terminal Window (CLI-chrome variant)
- Project / Feature Scope — `v1.1-fix1` (Quant Audit Remediation)
- _sample_params
- System Architecture & Data Flow Changes — `v1.1-fix1`
- Phase Breakdown
- v1.1-fix1/PLAN.md
- test_oof_assisted_phase4_uses_true_oof_not_refitted_in_sample
- Directory & File Structure — `v1.1-fix1`

## God Nodes (most connected - your core abstractions)
1. `InvalidDataError` - 199 edges
2. `FitContext` - 95 edges
3. `FeaturesConfig` - 93 edges
4. `TaskType` - 71 edges
5. `FeatureSelectionError` - 62 edges
6. `BaseFeatureFilter` - 62 edges
7. `FilterReport` - 59 edges
8. `FeatureFunnel` - 56 edges
9. `make_frames()` - 52 edges
10. `build_config()` - 52 edges

## Surprising Connections (you probably didn't know these)
- `Six-Layer Unidirectional Architecture` --semantically_similar_to--> `System Architecture`  [INFERRED] [semantically similar]
  CLAUDE.md → docs/plan/ARCHITECTURE.md
- `FeaturesConfig` --uses--> `Pipeline: config -> ingestion -> task inference -> split -> artifacts on disk.`  [INFERRED]
  src/quantautoml/common/config.py → tests/unit/test_pipeline.py
- `InvalidDataError` --uses--> `Unit tests for quantautoml.metrics.fdr.`  [INFERRED]
  src/quantautoml/common/exceptions.py → tests/unit/test_metrics_fdr.py
- `InvalidDataError` --uses--> `Unit tests for quantautoml.metrics.ic.`  [INFERRED]
  src/quantautoml/common/exceptions.py → tests/unit/test_metrics_ic.py
- `FilterReport` --calls--> `test_filter_report_is_frozen()`  [INFERRED]
  src/quantautoml/common/types.py → tests/unit/test_features_base.py

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **Leakage and rank-safety gates** — claude_rule01, claude_rule03, claude_rule04, claude_rule08 [INFERRED 0.75]
- **Five-phase pipeline** — docs_plan_phases_phase1, docs_plan_phases_phase2, docs_plan_phases_phase3, docs_plan_phases_phase4, docs_plan_phases_phase5 [EXTRACTED 0.90]
- **Feature funnel stages F1-F8** — s02_f1_triage, s02_f2_variance, s02_f3_ic_stability, s02_f4_fast_dcor, s02_f5_clustering, s02_f6_regime_synergy, s02_f8_jmi [EXTRACTED 1.00]
- **SLSQP, bootstrap shrinkage and temperature calibration flow** — s05_slsqp_optimizer, s05_bootstrap_shrinkage, s05_temperature_scaling, s05_rank_pearson_objective [EXTRACTED 0.95]
- **Friction-aware OOS acceptance** — s07_backtest_runner, s07_friction_cost_model, s07_hysteresis_buffer, s07_oos_acceptance_gate [EXTRACTED 0.95]

## Communities (213 total, 12 thin omitted)

### Community 0 - "InvalidDataError"
Cohesion: 0.14
Nodes (22): _lag1_autocorr(), bool_, float32, float64, int32, NDArray, Pearson correlation of contiguous finite adjacent pairs (col[:-1], col[1:])., Range of contiguous chronological block means normalized by full-Train std. (+14 more)

### Community 1 - "FitContext"
Cohesion: 0.05
Nodes (36): 1. Kiến Trúc Không Gian Tìm Kiếm Siêu Tham Số Chống Overfit (Phase 1), 2. Tăng Tốc GPU (`models.device`), Bảng Không Gian Tìm Kiếm Mới (Tuner & Tournament), System Architecture & Search-Space Design — `v1.1-fix2`, Phase 1: GBDT Regularization, Search-Space Hardening & GPU Acceleration *(Done)*, Phase 2: Regime Freshness & Multi-Window Walk-Forward OOS *(Active)*, Phase Breakdown, Roadmap Overview (+28 more)

### Community 2 - "types.py"
Cohesion: 0.12
Nodes (51): FeaturesConfig, FeatureSelectionError, Feature funnel cannot be fit or applied (e.g. unfitted filter, empty result)., FilterReport, BaseFeatureFilter, FitContext, ABC, Base contract for every feature filter (fit on Train only, transform anywhere). (+43 more)

### Community 3 - ".fit"
Cohesion: 0.20
Nodes (24): _fit(), int32, Self, Set `output_names_` and `report_`., _ctx(), _data(), KeepFirstFilter, float32 (+16 more)

### Community 4 - "Core Hard Gates RULE-01..10"
Cohesion: 0.06
Nodes (56): AGENTS.md Agent Guide, Agent TDD Workflow & Session Onboarding, CLAUDE.md Agent Guide, Six-Layer Unidirectional Architecture, Senior Quant Developer Persona, RULE-01 Zero Look-Ahead Leakage, RULE-02 Rank Metric Consistency, RULE-03 No Isotonic; Temperature Scaling (+48 more)

### Community 5 - "calc_rank_ic"
Cohesion: 0.29
Nodes (16): _calib(), _ensemble(), _export(), fx(), Path, ONNXBundleExporter: files, precision gate (RULE-06), train-only stats, no fit()., test_export_writes_required_files(), test_exporter_never_calls_fit() (+8 more)

### Community 6 - "make_frames"
Cohesion: 0.08
Nodes (73): PipelineConfig, build_config(), make_frames(), DataFrame, Shared pytest fixtures for the quantautoml test-suite., Build (features, labels) frames with forward-looking labels.      Labels: ``y_di, Write frames to disk and return a PipelineConfig pointing at them., cfg() (+65 more)

### Community 7 - "BaseGBDCandidate"
Cohesion: 0.06
Nodes (50): ABC, GraphProto, ModelProto, ModelError, PackagingError, Base class for model training, tournament, and tuning failures., Tournament evaluation or candidate selection failed., Hyperparameter optimization failed or produced invalid results. (+42 more)

### Community 8 - "load_config"
Cohesion: 0.11
Nodes (42): load_config(), Path, Path, write_toml(), Path, d = 1 - |corr| <= 1 always, so a cut at 1.0 would merge every feature., test_cluster_distance_open_unit_interval(), test_data_config_date_range_validation() (+34 more)

### Community 9 - "test_regime_synergy.py"
Cohesion: 0.14
Nodes (39): label_regimes(), int8, _cfg(), _ctx(), _fit(), _noise(), float32, float64 (+31 more)

### Community 10 - "7-Stage Feature Funnel (F1-F8)"
Cohesion: 0.06
Nodes (34): AdaptiveTimeSeriesSplitter (Purging & Embargo), UD-02 Boolean to Float32 auto-cast, DataIngestion (UTC us, float32, inner join), horizon_h Purging for forward-looking labels, infer_task_type (Binary/Multiclass/Unsupervised), NEED_UPDATE: design corrections (timestamp precision, bool/string, mid-NaN labels), Adaptive OOF-Assisted Split (N<10000), PipelineConfig (Pydantic TOML) (+26 more)

### Community 11 - ".transform"
Cohesion: 0.21
Nodes (29): _cfg(), _ctx(), _fit(), _labels(), float32, int32, MonkeyPatch, NDArray (+21 more)

### Community 12 - "test_clustering.py"
Cohesion: 0.23
Nodes (29): _cfg(), _ctx(), _fit(), _label(), _loading(), _noise(), float32, float64 (+21 more)

### Community 13 - "test_collinearity.py"
Cohesion: 0.10
Nodes (37): fast_dcor(), FastNonlinearCollinearityFilter, float32, float64, int32, NDArray, F4: fast nonlinear collinearity filter based on distance correlation (dCor, AVL), _standardize() (+29 more)

### Community 14 - "test_jmi.py"
Cohesion: 0.21
Nodes (10): Bootstrap Shrinkage Engine and Hurdle Fallback Gate (RULE-05)., Smooth SLSQP Convex Optimization on Rank-Transformed Data (C2 Differentiable Obj, SLSQP convex weight optimizer with C2 differentiable objective.      Solves for, SmoothSLSQPOptimizer, Unit tests for SmoothSLSQPOptimizer (C2 Objective)., test_slsqp_convergence_speed(), test_slsqp_fallback_on_constant_data(), test_slsqp_input_validation() (+2 more)

### Community 15 - "test_variance.py"
Cohesion: 0.20
Nodes (25): _cfg(), _ctx(), _fit(), _normal(), float32, int32, NDArray, test_all_nan_column_dropped_no_warning() (+17 more)

### Community 16 - "infer_task_type"
Cohesion: 0.18
Nodes (18): generic, infer_task_type(), NDArray, Infer the task. Missing labels (NaN / None) are unlabeled rows: they are     ign, test_binary_minus_one_one_is_remapped_to_zero_one(), test_binary_string_labels(), test_binary_zero_one_float_labels(), test_empty_labels_are_rejected() (+10 more)

### Community 17 - "test_ic_stability.py"
Cohesion: 0.16
Nodes (28): _cfg(), _ctx(), _fit(), _market(), float32, float64, int32, MonkeyPatch (+20 more)

### Community 18 - "ModelsConfig"
Cohesion: 0.08
Nodes (50): BaseModel, ModelsConfig, Quantitative metrics shared across pipeline layers., DecisionReport, DecisionType, ModelDecisionEngine, BaseModel, float32 (+42 more)

### Community 19 - "Temperature Scaling (RULE-03, rank invariant)"
Cohesion: 0.17
Nodes (13): Isotonic Regression (rejected: ties), Platt Scaling, Temperature Scaling (RULE-03, rank invariant), Rationale: Temperature Scaling preserves 100% Rank IC, DistributionShiftWarning (z>3), ONNXBundleExporter, PrecisionValidator (|y_py-y_onnx|<1e-4, RULE-06), predictor.py decoupled inference (numpy+onnxruntime, RULE-07) (+5 more)

### Community 20 - "What You Must Do When Invoked"
Cohesion: 0.08
Nodes (24): For /graphify add and --watch, For /graphify query, For the commit hook and native CLAUDE.md integration, For --update and --cluster-only, /graphify, Honesty Rules, Interpreter guard for subcommands, Part A - Structural extraction for code files (+16 more)

### Community 21 - "EnsembleConfig"
Cohesion: 0.16
Nodes (18): EnsembleConfig, BootstrapShrinkageEngine, Bootstrap weight distribution estimator with shrinkage and hurdle guard rail., RULE-01: Mutating split.val or split.oos MUST NOT affect ensemble weights or cal, test_zero_lookahead_leakage_val_and_oos_independence(), Unit tests for BootstrapShrinkageEngine and Hurdle Fallback (RULE-05)., test_bootstrap_shrinkage_calculation(), test_bootstrap_shrinkage_single_model() (+10 more)

### Community 22 - "TemperatureCalibrator"
Cohesion: 0.07
Nodes (39): Enum, CalibrationError, EnsembleError, OptimizationConvergenceError, Base class for ensemble optimization and probability calibration failures., SLSQP convex optimization failed to converge to a feasible solution., Probability calibration failed or violated rank-invariance., Core enums, data containers and report dataclasses for Phase 1. (+31 more)

### Community 23 - "exceptions.py"
Cohesion: 0.22
Nodes (18): PrecisionValidationError, Python vs ONNX outputs differ by >= the numerical precision tolerance (RULE-06)., float32, NDArray, Numerical Precision Contract (RULE-06): max|y_py - y_onnx| < tol on float32 outp, Return ``max|y_py - y_onnx|``; raise ``PrecisionValidationError`` unless it is `, validate_numerical_precision(), _f32() (+10 more)

### Community 25 - "Detailed Task Breakdown"
Cohesion: 0.10
Nodes (19): Detailed Task Breakdown, Execution Order Overview, Final Verification Checklist, Global Constraints, Implementation Plan: Quant AutoML Pipeline, Quyết định thiết kế (cần bạn xác nhận), Review Focus, [x] Task 10: F6 — Regime-Aware MI & Synergy (+11 more)

### Community 26 - "create_candidate"
Cohesion: 0.20
Nodes (17): calc_rank_ic(), Return (Spearman rho, p-value); degenerate inputs yield (0.0, 1.0).      Pairs w, ndarray, _rand(), Unit tests for quantautoml.metrics.ic., test_degenerate_inputs(), test_icir(), test_ignore_value_equals_row_drop() (+9 more)

### Community 27 - "float32"
Cohesion: 0.21
Nodes (26): _cfg(), _ctx(), _fit(), _noise(), float32, float64, int32, MonkeyPatch (+18 more)

### Community 28 - "TaskType"
Cohesion: 0.11
Nodes (33): PreprocessorSpec, Composed description of ``FeatureFunnel.transform`` for ONNX export.      Semant, A candidate model that has been fine-tuned and refitted on the full training set, TunedCandidate, _dict_to_toml(), _dump_json(), _extract_ohlcv_df(), _fit_oof_candidates() (+25 more)

### Community 29 - "calc_ece"
Cohesion: 0.67
Nodes (3): 1. True OOF Predictions trong `OOF_ASSISTED` Mode (Audit #1), Các phương án cân nhắc:, Kết luận:

### Community 30 - "DeepOptunaTuner"
Cohesion: 0.06
Nodes (77): analyze(), _block_summary(), classify_shape(), clean_label(), _configure_stdout_utf8(), Diagram, digest(), _discard_nonsemantic() (+69 more)

### Community 31 - "bootstrap.py"
Cohesion: 0.40
Nodes (4): float32, int32, NDArray, Optimize ensemble weights on Cal-A predictions (chronologically before Calibrati

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
Cohesion: 0.14
Nodes (23): calc_ic_sharpe(), calc_icir(), calc_rolling_ic(), calc_rolling_ic_matrix(), float32, float64, int32, NDArray (+15 more)

### Community 37 - "graphify reference: extra exports and benchmark"
Cohesion: 0.22
Nodes (8): graphify reference: extra exports and benchmark, Step 6b - Wiki (only if --wiki flag), Step 7 - Neo4j export (only if --neo4j or --neo4j-push flag), Step 7a - FalkorDB export (only if --falkordb or --falkordb-push flag), Step 7b - SVG export (only if --svg flag), Step 7c - GraphML export (only if --graphml flag), Step 7d - MCP server (only if --mcp flag), Step 8 - Token reduction benchmark (only if total_words > 5000)

### Community 38 - "AI Agent Guidelines — Quant AutoML Pipeline"
Cohesion: 0.22
Nodes (8): 1. Vai Trò & Tôn Chỉ Kỹ Thuật (Persona & Core Philosophy), 2.1. Giao Thức Khởi Động Phiên Làm Việc (Session Onboarding & State Discovery Protocol), 2.2. Giao Thức Quản Lý Vòng Đời Trạng Thái (State Lifecycle Management), 2.3. Các bước bắt buộc khi thực hiện task:, 2. Quy Trình Làm Việc Tiêu Chuẩn Cho Agent (Workflow), 3. Nguyên Tắc An Toàn Định Lượng (Quant Safety Directives), 4. Hướng Dẫn Debug & Xử Lý Sự Cố (Root-Cause Debugging), AI Agent Guidelines — Quant AutoML Pipeline

### Community 39 - "Quantitative metrics shared across pipeline layers."
Cohesion: 0.09
Nodes (30): create_candidate(), LightGBMCandidate, Any, float32, int32, NDArray, Self, Map labels to contiguous 0..k-1 (required by XGBoost) and remember the originals (+22 more)

### Community 40 - "graphify reference: query, path, explain"
Cohesion: 0.33
Nodes (5): For /graphify explain, For /graphify path, graphify reference: query, path, explain, Step 0 — Constrained query expansion (REQUIRED before traversal), Step 1 — Traversal

### Community 41 - "Step 04 — Design: Deep Optuna Tuning"
Cohesion: 0.33
Nodes (5): Công nghệ, Data flow, Ràng buộc, Source layout, Step 04 — Design: Deep Optuna Tuning

### Community 42 - "test_zero_leakage_ensemble.py"
Cohesion: 0.05
Nodes (51): BalanceParser, canonical_controller(), check_css_references(), check_document_completeness(), check_motion(), check_scripts(), check_svgs(), check_text_fit() (+43 more)

### Community 43 - ".optimize"
Cohesion: 0.19
Nodes (12): calc_ece(), float32, int32, NDArray, Metrics for evaluating probability calibration error (ECE)., Calculate the Expected Calibration Error (ECE) for binary or multiclass probabil, Unit tests for Expected Calibration Error (ECE) metric., test_calc_ece_empty_or_invalid() (+4 more)

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

### Community 50 - "Self"
Cohesion: 0.18
Nodes (8): _css_blocks(), Make HTML comments valid XML comments; CDATA is copied unchanged.      HTML allo, Yield whole balanced rules without flattening nested conditional blocks., xml_safe_comments(), parse_iso_bound(), Self, Parse an ISO-8601 date or datetime string into a UTC-aware ``datetime``.      A, ValueError

### Community 59 - "README.md"
Cohesion: 0.50
Nodes (4): 1. Cấu Trúc Thư Mục Kết Quả (`runs/default/`), 2. File Nào Lưu Các Đặc Trưng Cuối Cùng Được Chọn?, 3. Cách Đọc Báo Cáo Nghiệm Thu `backtest_report.json`, Danh Mục Báo Cáo & Artifacts Đầu Ra (`OUTPUTS`)

### Community 63 - "verify_geometry.py"
Cohesion: 0.08
Nodes (44): arc(), attached(), attribute(), bezier(), bounds(), check(), check_connectors(), chord_count() (+36 more)

### Community 64 - "DeepOptunaTuner"
Cohesion: 0.10
Nodes (26): Custom exceptions shared across the quantautoml pipeline., DeepOptunaTuner, BaseModel, float32, int32, NDArray, Deep hyperparameter optimization using Optuna Multivariate TPE and Hyperband pru, Run deep tuning for each model in models_to_tune.          Evaluates on walk-for (+18 more)

### Community 65 - "drawio_extract.py"
Cohesion: 0.11
Nodes (43): _aligned(), analyze(), classify_shape(), clean_label(), _configure_stdout_utf8(), _decompress_limited(), digest(), Edge (+35 more)

### Community 67 - "BacktestError"
Cohesion: 0.22
Nodes (11): Series, _num(), BaseModel, DataFrame, float64, NDArray, Tier 2 realistic execution: replays Tier 1 positions through backtesting.py.  ``, backtesting.py reports undefined ratios as NaN; surface them as 0.0. (+3 more)

### Community 68 - "Step 08-fix1 — Design: Dual-Tier Realistic Backtest (backtesting.py Integration)"
Cohesion: 0.06
Nodes (32): 1. Kiến Trúc Tổng Thể (Dual-Tier), 2. Data Flow Chi Tiết, 3. Source Layout, 4. Interface Contracts, 5. Non-Functional Requirements, `BacktestingPyStrategy` (`backtest/strategy.py`), `RealWorldConfig` (thêm vào `common/config.py`), `RealWorldMetrics` (Pydantic, frozen) (+24 more)

### Community 69 - "exceptions.py"
Cohesion: 0.18
Nodes (17): OOSBacktestRunner, BacktestConfig, bundle(), _ohlcv(), _oos(), DataFrame, Tier 2 (backtesting.py) end-to-end on a real bundle: HTML + trades.csv, gated by, Tier 2 replays the same positions: it never trades while Tier 1 is flat for the (+9 more)

### Community 70 - "FeatureFunnel"
Cohesion: 0.15
Nodes (32): FeatureFunnel, Fit the eight feature filters in order on Train only, then transform any segment, _cfg(), _dataset(), fitted(), Signal ``sig`` drives the next 1-bar return; ``miss`` has MNAR NaN (on y == 1)., _report(), _split() (+24 more)

### Community 71 - "contingency_mi"
Cohesion: 0.12
Nodes (19): contingency_mi(), float32, intp, NDArray, quantile_bin_codes(), Private helpers shared by F6 and F8: quantile bin codes and contingency-table MI, Per-column ``n_bins``-quantile bin codes and the code range of each column., Plug-in MI (nats) of codes ``a`` in ``[0, na)`` vs class codes ``yc`` in ``[0, n (+11 more)

### Community 72 - "Onboarding — generate your skin from a design source"
Cohesion: 0.07
Nodes (30): Colors, Constraint checks, Exact-font gate for brand-matched output, Final step — user verifies the output, § Folder, Fonts, Invocation, Invocation (+22 more)

### Community 73 - "Ridgeline"
Cohesion: 0.07
Nodes (29): Anti-patterns, Anti-patterns, Anti-patterns, Anti-patterns, Bump chart, Colour, Colour, Colour (+21 more)

### Community 74 - "2. Layout formulas — deterministic geometry"
Cohesion: 0.07
Nodes (29): 1. Inputs — the parameter contract, 2.1 Canvas, 2.2 Horizontal chevron banner, 2.3 Source zone (dashed, external), 2.4 Cluster boundary (solid), 2.5 Cross-cutting bars (identity, observability, …), 2.6 Orchestration bar component (inside cluster), 2.7 Component nodes (inside cluster) (+21 more)

### Community 75 - "excalidraw_extract.py"
Cohesion: 0.17
Nodes (27): analyze(), clean_label(), _configure_stdout_utf8(), digest(), Edge, _escape_inline(), _escape_markdown(), _escape_table() (+19 more)

### Community 76 - "evaluate_oos_performance"
Cohesion: 0.11
Nodes (39): _daily_compounded_returns(), _deflated_sharpe_ratio(), evaluate_oos_performance(), _max_drawdown(), OOSMetrics, BaseModel, bool_, datetime64 (+31 more)

### Community 77 - "DataIngestion"
Cohesion: 0.05
Nodes (95): ArgumentParser, datetime, Exception, IndexArray, build_config_content(), detect_file_info(), main(), Any (+87 more)

### Community 78 - "Process"
Cohesion: 0.07
Nodes (28): 10. Complexity budget, 11. Anti-patterns, 12.1 What this YAML proves, 12.2 Adapting this YAML to a different process, 12. Worked example — full YAML for an extended process variant, 13. Examples, 1. Inputs — the parameter contract, 2.1 Background structure (+20 more)

### Community 79 - "runner.py"
Cohesion: 0.13
Nodes (16): Hysteresis Buffer Bands: enter on the top/bottom ``top_pct`` of the signal, exit, FrictionCostModel, float64, NDArray, Friction cost model (RULE-09): R_net = R_gross - Turnover * (fee + slippage)., Taker fee + slippage in basis points, charged on ``Turnover_t = 0.5 * sum|w_t -, Friction-aware blind OOS backtesting., Blind OOS backtest: the bundle's predictor scores OOS, Hysteresis sizes it, fric (+8 more)

### Community 80 - "FeatureSchemaContract"
Cohesion: 0.13
Nodes (22): Input columns/dtype do not match the features_schema.json data contract., SchemaMismatchError, FeatureSchemaContract, _is_binary(), BaseModel, float32, float64, NDArray (+14 more)

### Community 81 - "Data Flow"
Cohesion: 0.08
Nodes (24): 10. Complexity budget, 11. Anti-patterns, 12. Examples, 1. Inputs — the parameter contract, 2.1 Background structure, 2.2 Step header chip, 2.3 Lane labels, 2.4 Node content layout (inside the 100×64 rect) (+16 more)

### Community 82 - "IT current-state"
Cohesion: 0.08
Nodes (23): 10.1 What this YAML proves, 10. Worked YAML — full inputs for `example-it-state.html`, 1. Inputs — the parameter contract, 2.1 Background and zone frame, 2.2 Component box, 2.3 Connector geometry (§3 holds the routing rules), 2.4 Footer bar, 2.5 Legend strip (+15 more)

### Community 83 - "Beeswarm"
Cohesion: 0.08
Nodes (23): Anti-patterns, Anti-patterns, Anti-patterns, Anti-patterns, Beeswarm, Box plot, Bubble, Colour (+15 more)

### Community 84 - "DP security matrix"
Cohesion: 0.09
Nodes (22): 10.1 What this YAML proves, 10. Worked YAML — full inputs for `example-dp-security-matrix.html`, 1. Inputs — the parameter contract, 2.1 Background, 2.2 Header row (`y = 72, h = 52`), 2.3 Data row (`y = row_y(k), h = 36`), 2.4 Cell style table, 2.5 Legend (`y_top = legend_y_top, h ≈ 30`) (+14 more)

### Community 85 - "Diagram Design"
Cohesion: 0.09
Nodes (23): 0. First-time setup — style guide gate, 10. Templates & Variants, 11. Importing an Existing Diagram (draw.io), Mermaid, Excalidraw, and PlantUML, 12. Output, 1. Philosophy, 2. When to Use, 3. Selection: semantic pattern, then visual type, 4. Universal Anti-patterns (+15 more)

### Community 86 - "plantuml_extract.py"
Cohesion: 0.10
Nodes (19): _activation_shortcut(), _configure_stdout_utf8(), ExtractError, _finalize_degrees(), _note_has_inline_body(), _parse_note(), Exception, _quoted_or_token() (+11 more)

### Community 87 - "HysteresisBuffer"
Cohesion: 0.15
Nodes (18): HysteresisBuffer, float32, float64, NDArray, Percentile in [0, 1] of ``scores[t]`` within ``scores[t-w+1 : t+1]``; NaN before, State machine over percentiles: +1 long, -1 short, 0 flat (also while pct is NaN, _buf(), HysteresisBuffer: causal rolling percentile + enter/exit bands on the signal ran (+10 more)

### Community 88 - "validate_numerical_precision"
Cohesion: 0.14
Nodes (22): TempPathFactory, Case, _load(), _noise_segment(), float64, MonkeyPatch, NDArray, Path (+14 more)

### Community 89 - "apply_fdr_correction"
Cohesion: 0.17
Nodes (13): apply_fdr_correction(), bool_, float64, NDArray, Benjamini-Hochberg FDR control (RULE-08)., Return a mask of accepted hypotheses, in the original order.      k = max{i : p_, Unit tests for quantautoml.metrics.fdr., test_empty() (+5 more)

### Community 90 - "Brand"
Cohesion: 0.10
Nodes (21): active-directory, aws, azure, Brand, docker, gcp, gitea, github (+13 more)

### Community 91 - "DP integration"
Cohesion: 0.10
Nodes (21): 10. Budget — this type exceeds the default, 11. Anti-patterns, 12. Examples, 1. Inputs — the parameter contract, 2.1 Row placement (cursor algorithm), 2.2 Node placement inside a `row` entry, 2.3 Bar (full-zone-width) placement, 2.4 Source / consumer placement (side columns) (+13 more)

### Community 92 - "Medallion"
Cohesion: 0.10
Nodes (20): 10. Worked YAML, 1. Inputs — the parameter contract, 2.1 Background, 2.2 Tier card (172 × 380), 2.3 Tier styles, 2.4 Promotion arcs (over the top of the tiers), 2.5 Path row (bottom, optional), 2. Layout formulas — deterministic geometry (+12 more)

### Community 93 - "export_svg_document"
Cohesion: 0.12
Nodes (20): assert_export_gate(), decode_xml_references(), ensure_viewbox(), ensure_xmlns(), export_svg_document(), find_defs_ids(), main(), namespace_defs_ids() (+12 more)

### Community 94 - "str"
Cohesion: 0.15
Nodes (23): InvalidDataError, Timestamps are non-monotonic, duplicated or cannot be aligned., Input data or configuration violates a pipeline data contract., TimestampAlignmentError, IngestionReport, DataIngestion, _encode_features(), _extract_selected_label() (+15 more)

### Community 95 - "export_svg.py"
Cohesion: 0.25
Nodes (11): _animation_component(), _animation_declarations(), _animation_property(), _css_segments(), _diagram_css_leaf(), escape_css_for_xml(), has_diagram_stylesheet(), True when a <style> block contains CSS rules beyond the fonts @import. (+3 more)

### Community 96 - "digest"
Cohesion: 0.19
Nodes (19): analyze(), _block_summary(), digest(), _escape_markdown(), _escape_table(), _fragment_summary(), load_blocks(), main() (+11 more)

### Community 97 - "Diagram"
Cohesion: 0.16
Nodes (10): Diagram, _parse_class(), Consume multi-line styling: `skinparam x {` ... `}` and     `legend` / `header`, Return True when *text* is styling that the caller must skip., Store a note's text plus where it sits. A note on exactly one     participant or, Resolve a name exactly (PlantUML names are case-sensitive).          Class names, Return statement text, skipping note, legend, styling, and class bodies.      Bo, _record_note() (+2 more)

### Community 98 - "OOSBacktestRunner"
Cohesion: 0.09
Nodes (24): ModuleType, DataConfig, LabelsConfig, PackagingConfig, BaseModel, Pipeline configuration loaded from a TOML file and validated with Pydantic., SplitConfig, _Strict (+16 more)

### Community 99 - "test_zero_leakage_packaging.py"
Cohesion: 0.23
Nodes (22): BacktestError, Out-of-sample backtest cannot be run on the supplied data., _oos(), _other(), Path, OOSBacktestRunner: blind OOS run through the standalone predictor with friction, Audit #8: execution_price='next_open' computes returns from Open[t+1] -> Open[t+, _run() (+14 more)

### Community 100 - "_edge_split"
Cohesion: 0.12
Nodes (18): _class_relationship(), _edge_split(), _endpoint_code(), _is_skin(), _is_structural_link(), Match, Split *text* around its first whole arrow outside double quotes.      The arrow, True for an inheritance, composition, or aggregation head, which only     a clas (+10 more)

### Community 101 - "Client profiles"
Cohesion: 0.12
Nodes (17): 1. Inspect the project marker, 2. Resolve without a valid marker, Built-in `default`, Client profiles, Current-schema structural check, `delete [slug]`, Failure and recovery cases, `list` (+9 more)

### Community 102 - "Style Guide"
Cohesion: 0.12
Nodes (17): Constraints (don't break these), Contrast contract, Customizing the skin, Cyrillic labels, Font source, Font stack, Inversion rule (light → dark), Korean labels (+9 more)

### Community 103 - "test_packaging_bundle.py"
Cohesion: 0.13
Nodes (28): RealWorldConfig, Unit tests for RealWorldConfig., test_cash_zero_rejected(), test_defaults_match_spec(), test_extra_field_forbidden(), test_invalid_size_rejected(), test_negative_commission_rejected(), test_valid_size_modes_accepted() (+20 more)

### Community 104 - "Optional animation"
Cohesion: 0.22
Nodes (9): Anti-patterns, Complexity and deterministic timing, Export and verification, Interactive controls and keyboard, Modes, Optional animation, Reduced motion, color, and accessibility, Semantic primitives (+1 more)

### Community 105 - "css_escape"
Cohesion: 0.12
Nodes (19): body_inherited_css(), css_escape(), css_id_token(), _css_paint_may_target_descendants(), decoded_css_url_fragment(), is_chrome_selector(), Element, Exclude provably unrelated simple selectors; preserve unknown CSS cascades. (+11 more)

### Community 106 - ".predict_proba"
Cohesion: 0.18
Nodes (11): InferenceSession, DistributionShiftWarning, Predictor, ndarray, Path, Standalone inference engine for a quantautoml model bundle (numpy + onnxruntime, An input feature lies more than ``drift_z_threshold`` Train std-devs from its Tr, Calibrated probabilities ``[N, C]`` (binary: ``[1 - p, p]``), float32. (+3 more)

### Community 107 - "test_zero_leakage.py"
Cohesion: 0.29
Nodes (7): Anti-patterns, Complexity budget, Examples, Fishbone / Ishikawa (root-cause), Layout conventions, Math, Pre-computed reference (5-bone layout, HEAD=1200, CY=320)

### Community 108 - "Sequence"
Cohesion: 0.13
Nodes (15): Activation bar primitive, Anti-patterns, Combined fragments (`alt` / `opt` / `loop`), Complexity budget (sequence-specific), Examples, Fragment layout rules, Frame primitive (shared), Guard + divider primitives (+7 more)

### Community 109 - "Marimekko"
Cohesion: 0.13
Nodes (14): Anti-patterns, Anti-patterns, Cell element pattern, Colour, Declaring the share, Declaring the values, Examples, Honest-data rule (+6 more)

### Community 110 - "Export to PNG / SVG"
Cohesion: 0.14
Nodes (14): Caveat to surface to the user, Detection, Edge cases, Export to PNG / SVG, Hitting an exact pixel size, Manual algorithm (what the helper does), Output naming, PNG export procedure (+6 more)

### Community 111 - "layout-budget.md"
Cohesion: 0.33
Nodes (6): 4px grid, Complexity budget (per diagram), Layout grid, per-type budgets, page layout, and summary cards, Layout & Spacing, Page layout, Summary Card Pattern

### Community 112 - "Draw.io import output spec — format × size × detail × audience"
Cohesion: 0.14
Nodes (14): 1. Format, 2. Size, 3. Detail level, 4. Audience level, 5. Fidelity ledger, 6. Checklist, Degrade ladder, Deriving `fit` (+6 more)

### Community 113 - "Architecture delta"
Cohesion: 0.14
Nodes (13): Anti-patterns, Architecture delta, Change vocabulary and redundant encoding, Components, Diagram and snapshots, Examples, Layout conventions, Ledger (+5 more)

### Community 114 - "Loop"
Cohesion: 0.14
Nodes (13): 1. Inputs — the parameter contract, 2.1 Station centers, 2.2 Solid ring-flow endpoints, 2.3 Dashed write-back spoke endpoints, 2.4 ViewBox sizing, 2. Layout math — deterministic geometry, 3. Visual grammar, 4. Connector rules (mandatory) (+5 more)

### Community 115 - "_fail"
Cohesion: 0.25
Nodes (12): _blank_comments(), _detect_kind(), _fail(), parse_block(), _parse_participant(), _parse_sequence(), NoReturn, Record an activation change on the message timeline.          ``after_message`` (+4 more)

### Community 116 - "build_preprocessor_onnx"
Cohesion: 0.23
Nodes (9): BaseFeatureFilter, ICStabilityFilter, float32, float64, int32, NDArray, Two-sided 1-sample t-test p-value of non-overlapping rolling Rank ICs vs 0., Keep features with stable, significant, monotonically decaying Rank IC (Train on (+1 more)

### Community 117 - "test_runner_tier2_integration.py"
Cohesion: 0.22
Nodes (14): bundle(), _ohlcv(), _oos(), DataFrame, OOSBacktestRunner Tier 2 trigger: realistic execution runs only on accepted Tier, Tier 2 must see all n positions (not the n-1 slice used for Tier 1 returns)., _run(), test_backward_compat_old_caller() (+6 more)

### Community 118 - "Import from PlantUML"
Cohesion: 0.15
Nodes (13): Anti-patterns, Edge cases, Import from PlantUML, Multi-block files, Step 1 — Extract the IR, Step 2 — Set the four dials, Step 3 — Pick the target type, Step 4 — Build the semantic model (+5 more)

### Community 119 - "_rewrite_start_tags"
Cohesion: 0.18
Nodes (17): _decoded_css_property_text(), extract_first_svg(), _has_animation_property(), html_tokens(), mask_opaque_html(), namespace_inline_animations(), normalize_rgba_presentation_attrs(), Match (+9 more)

### Community 120 - "Import from draw.io"
Cohesion: 0.17
Nodes (12): Anti-patterns, Edge cases, Import from draw.io, Multi-page files, Step 1 — Extract the IR, Step 2 — Set the four dials, Step 3 — Pick the target type, Step 4 — Build the semantic model (+4 more)

### Community 121 - "Import from Mermaid"
Cohesion: 0.17
Nodes (12): Anti-patterns, Edge cases, Import from Mermaid, Multi-block files, Step 1 — Extract the IR, Step 2 — Set the four dials, Step 3 — Pick the target type, Step 4 — Build the semantic model (+4 more)

### Community 122 - "Data stack"
Cohesion: 0.17
Nodes (12): airflow, dagster, Data stack, hop, jupyter, nifi, pentaho, powerbi (+4 more)

### Community 123 - "Semantic patterns"
Cohesion: 0.17
Nodes (12): 1. Fan-in queue / bottleneck, 2. Stage framework with semantic slots, 3. Unstructured input → structured artifact, 4. Paired policy-evaluation traces, 5. Secure paved road, 6. Governance / control catalog, 7. Compensating security layers, 8. Traceable block decomposition (+4 more)

### Community 124 - "Exploded axonometric"
Cohesion: 0.17
Nodes (11): Anti-patterns, Detail, Examples, Exploded axonometric, Faces and lines, Focal part, Labels, Layout conventions (+3 more)

### Community 125 - "Polar Chart"
Cohesion: 0.17
Nodes (11): Anti-patterns, Complexity budget, Examples, Geometry, Input contract, Layout conventions, Polar Chart, Quantitative encoding (+3 more)

### Community 126 - "Steps"
Cohesion: 0.17
Nodes (12): 1. Inputs, 2. Layout grammar, 3. Focal and color rules, 4. Legend, 5. Reproducibility checklist, 6. What `verify-steps.py` requires, 7. Anti-patterns, Copy block (+4 more)

### Community 127 - "System Architecture & Technical Specifications: Quant AutoML Pipeline"
Cohesion: 0.17
Nodes (11): 1. System Overview & Core Philosophy, 2. End-to-End Pipeline Architecture Diagram, 3. Data Flow & Temporal Isolation Architecture, 4.1. Smooth SLSQP Objective ($C^2$ Differentiability), 4.2. Temperature Scaling & 100% Rank Invariance Contract (RULE-03), 4.3. Multi-Horizon Term Structure & FDR Control (RULE-08), 4.4. Friction Cost Matrix & Hysteresis Buffer (RULE-09), 4. Mathematical Formulations & Optimization Engines (+3 more)

### Community 128 - "._fit"
Cohesion: 0.18
Nodes (10): MissingKind, _preprocessor_spec(), Return the composed, ONNX-exportable description of ``transform``., NaNInfTriage, bool_, float32, float64, int32 (+2 more)

### Community 129 - ".transform"
Cohesion: 0.18
Nodes (17): bundle(), Friction + Hysteresis behave as RULE-09 requires on a real bundle (blind OOS run, Slow signal + noise: the percentile rank keeps wobbling around the entry boundar, _run(), test_buffer_lowers_turnover(), test_friction_lowers_sharpe(), test_higher_friction_never_improves_net_sharpe(), _wobbly_oos() (+9 more)

### Community 130 - "._fit"
Cohesion: 0.29
Nodes (8): ic_behavior_distance(), ICBehaviorClusterFilter, bool_, float32, float64, int32, intp, NDArray

### Community 131 - "test_zero_leakage_models.py"
Cohesion: 0.18
Nodes (10): 1. Yêu Cầu Hệ Thống & Cài Đặt, 2.1. Sinh cấu hình mặc định (nhãn nhị phân `is_peak_10`), 2.2. Đổi nhãn mục tiêu sang nhãn giao dịch có hướng (`allow_entry`), 2.3. Tùy chỉnh kỳ hạn (`horizon-h`, `target-horizon`) và số lượng Trials, 2.4. Sinh cấu hình cho bài toán Không giám sát (Unsupervised Mode), 2.5. Bảng tham số CLI của `scripts/generate_config.py`, 2. Sinh File Cấu Hình `config.toml` Tự Động (`scripts/generate_config.py`), 3. Chạy Toàn Bộ Pipeline (+2 more)

### Community 132 - "Import from Excalidraw"
Cohesion: 0.18
Nodes (11): Anti-patterns, Edge cases, Import from Excalidraw, Step 1 — Extract the IR, Step 2 — Set the four dials, Step 3 — Pick the target type, Step 4 — Build the semantic model, Step 5 — Redraw (+3 more)

### Community 133 - "Icons (primitive)"
Cohesion: 0.18
Nodes (11): csv, excel, File formats, Icons (primitive), Language, License attribution, python, r (+3 more)

### Community 134 - "Arc diagram"
Cohesion: 0.18
Nodes (11): Anti-patterns, Anti-patterns, Arc diagram, Colour, Complexity budget, Declaring the values, Dependency Graph, Examples (+3 more)

### Community 135 - "_CssRule"
Cohesion: 0.27
Nodes (10): _css_rule_nodes(), _css_rule_tree(), _CssRule, diagram_css_from_html(), _diagram_css_rules(), _html_css_rules(), _keyframe_names(), Carry scoped diagram CSS while retaining its existing media conditions. (+2 more)

### Community 136 - ".run"
Cohesion: 0.23
Nodes (11): _calc_series_ic_sharpe(), float32, int32, intp, NDArray, Moving Block Bootstrap: sample contiguous time-series blocks of length ``block_s, Helper to evaluate IC Sharpe or fallback to Rank IC on small sample slices., Fit ensemble weights on Cal-B using Bootstrap and Shrinkage.          Parameters (+3 more)

### Community 137 - "test_backtest_strategy.py"
Cohesion: 0.16
Nodes (15): BacktestingPyStrategy, Thin backtesting.py adapter: replays pre-computed positions (Tier 2)., Issues Market orders following the pre-computed ``_position`` column.      The o, Strategy, _ohlcv(), DataFrame, Unit tests for BacktestingPyStrategy (adapter for pre-computed positions)., With size_mode='start_cash', trade size is computed from initial_cash rather tha (+7 more)

### Community 138 - "test_packaging_converters.py"
Cohesion: 0.27
Nodes (9): _count_gaps(), _latency_ms(), _periods_per_year(), DataFrame, datetime64, float32, float64, NDArray (+1 more)

### Community 139 - "Axonometric plan"
Cohesion: 0.20
Nodes (9): Anti-patterns, Axonometric plan, Examples, Focal element, Layout conventions, Metadata contract, Optional motion, Projection (+1 more)

### Community 140 - "Consultant special (2×2 scenario matrix)"
Cohesion: 0.20
Nodes (9): Anti-patterns, Anti-patterns (variant-specific), Consultant special (2×2 scenario matrix), Examples, Layout conventions, Layout conventions, Quadrant, Style tokens (in-house) (+1 more)

### Community 141 - "RegimeSynergyFilter"
Cohesion: 0.29
Nodes (7): float32, int32, NDArray, Self, Apply learned temperature scaling to input probabilities.          Parameters, Verify that temperature calibration strictly preserved 100% of ranking order., Fit optimal temperature scalar T > 0 on Cal-B (chronologically after Ensemble).

### Community 142 - "Data"
Cohesion: 0.22
Nodes (9): backup, bucket, cache, Data, database, file, log, queue (+1 more)

### Community 143 - "Network"
Cohesion: 0.22
Nodes (9): cdn, cloud, dns, firewall, gateway, internet, load-balancer, Network (+1 more)

### Community 144 - "Bar / Column Chart"
Cohesion: 0.20
Nodes (9): Anti-patterns, Bar / Column Chart, Bar element pattern, Dumbbell element pattern, Dumbbell honesty rules, Dumbbell layout, Examples, Layout conventions (+1 more)

### Community 145 - "Database Schema"
Cohesion: 0.22
Nodes (9): Anti-patterns, Complexity budget, Database Schema, Examples, Focal rule, Foreign-key connectors — the defining rule, Layout conventions, Schema grouping (+1 more)

### Community 146 - "User Journey Map"
Cohesion: 0.22
Nodes (8): Anti-patterns, Complexity budget, Connector note, Examples, Focal rule, Geometry, Layout conventions, User Journey Map

### Community 147 - "Action"
Cohesion: 0.25
Nodes (8): Action, alert, api, key, lock, request, response, sync

### Community 148 - "Core SVG primitives"
Cohesion: 0.25
Nodes (8): Accessible SVG contract, Arrow labels — always mask, always with margin, Arrow markers (define all three, always), Background, Core SVG primitives, Legend — horizontal strip at the bottom, Mandatory connector rules, Node box — full pattern

### Community 149 - "Architecture"
Cohesion: 0.25
Nodes (8): Anti-patterns, Architecture, Connector style, Containment, Crossing arrows — bridge / hop, Examples, Layout conventions, Zone grouping

### Community 150 - "Radar / Spider"
Cohesion: 0.25
Nodes (8): Anti-patterns, Examples, Layout conventions, Math, Pre-computed reference (N=5, cx=500, cy=240, R=160, S=10, integer-rounded), Radar / Spider, Series palette, Worked example (N=5)

### Community 151 - "Sankey / Flow-Quantity"
Cohesion: 0.25
Nodes (7): Anti-patterns, Complexity budget, Examples, Layout conventions, Sankey / Flow-Quantity, Scale rule, Worked reference (k = 0.02 px/unit, budget = 12,000 CI minutes)

### Community 152 - "Service Blueprint"
Cohesion: 0.25
Nodes (7): Anti-patterns, Examples, Honesty rules, Layout conventions, Machine-readable data contract, Service Blueprint, Service-blueprint element pattern

### Community 153 - "Unit Grid"
Cohesion: 0.25
Nodes (7): Anti-patterns, Complexity budget, Examples, Honest-data rule, Input contract, Layout conventions, Unit Grid

### Community 154 - "._fit"
Cohesion: 0.27
Nodes (7): MultiMetricVarianceFilter, _peak_var_ratio(), float32, float64, int32, NDArray, P90 of rolling variance / full variance per column, via cumulative sums.

### Community 155 - "Environment doctor"
Cohesion: 0.29
Nodes (6): Environment doctor, Example result, Inputs, Output contract, Required checks, Safety and behavior rules

### Community 156 - "DevOps"
Cohesion: 0.29
Nodes (7): bug, DevOps, git-branch, monitoring, pipeline, terminal, test

### Community 157 - "Compute"
Cohesion: 0.29
Nodes (7): Compute, container, desktop, laptop, phone, server, vm

### Community 158 - "Kubernetes"
Cohesion: 0.29
Nodes (7): deployment, ingress, Kubernetes, node, pod, service, volume

### Community 159 - "Sketchy Filter (hand-drawn variant)"
Cohesion: 0.33
Nodes (6): Critical rule, Grammar, Sketchy Filter (hand-drawn variant), Tuning, When not to use, When to use

### Community 161 - "test_pipeline_packaging.py"
Cohesion: 0.15
Nodes (28): build_preprocessor_onnx(), float32, NDArray, Path, Write an ONNX graph ``X[N, n_in] float32 -> Xp[N, len(final_idx)] float32``., Run ``preprocessor.onnx`` (helper for tests and the precision validator)., run_preprocessor_onnx(), make_bundle() (+20 more)

### Community 162 - "Heatmap"
Cohesion: 0.29
Nodes (7): Anti-patterns, Cell pattern, Declaring the values, Heatmap, Honest-data rule, Layout conventions, The fill ramp

### Community 163 - "Segment"
Cohesion: 0.20
Nodes (8): float32, intp, NDArray, Indices (into this stage's input columns) of the columns it keeps, in output ord, Apply the fitted selection without mutating state., _transform(), test_transform_all_segments_same_columns_float32(), test_transform_wrong_width_raises()

### Community 164 - "User Story Map"
Cohesion: 0.29
Nodes (6): Anti-patterns, Complexity budget, Examples, Focal rule, Layout conventions, User Story Map

### Community 165 - "UML Class Diagram"
Cohesion: 0.29
Nodes (7): Anti-patterns, Complexity budget, Connector rules, Examples, Layout conventions, Relationship vocabulary, UML Class Diagram

### Community 166 - "Waterfall Chart"
Cohesion: 0.29
Nodes (6): Anti-patterns, Examples, Layout conventions, Waterfall Chart, Waterfall element pattern, Waterfall honesty rules

### Community 167 - "_parse_declaration_tail"
Cohesion: 0.29
Nodes (7): clean_label(), _parse_classifier_header(), _parse_declaration_tail(), Flatten PlantUML / Creole / HTML label markup without interpreting it.      Enti, The one declaration tokenizer: participants, classifiers, and packages.      Rea, Split a balanced ``<...>`` type parameter list from the front of *text*., _split_generic()

### Community 168 - "Node"
Cohesion: 0.29
Nodes (5): Edge, _has_cycle(), Node, node_ref(), Name a node unambiguously: its id, plus its label when that differs.      Two cl

### Community 169 - "CalibrationError"
Cohesion: 0.29
Nodes (7): Edge cases, Export block registry (`--registry`), JSON schema, Procedure, Scope, Trigger, What this never does

### Community 170 - "Statistical tools"
Cohesion: 0.33
Nodes (6): qgis, rstudio, sas, spss, stata, Statistical tools

### Community 171 - "Event modeling"
Cohesion: 0.33
Nodes (6): Anti-patterns, Complexity budget, Event modeling, Examples, Layout grammar, Machine-readable contract

### Community 172 - "Gantt Chart"
Cohesion: 0.33
Nodes (5): Anti-patterns, Examples, Gantt Chart, Layout conventions, Task bar pattern

### Community 173 - "Org Chart / Responsibility Map"
Cohesion: 0.33
Nodes (6): Anti-patterns, Complexity budget, Examples, Layout conventions, Node treatments, Org Chart / Responsibility Map

### Community 174 - "Pyramid / Funnel"
Cohesion: 0.33
Nodes (5): Anti-patterns, Examples, Layout conventions, Pyramid / Funnel, Two orientations — pick one

### Community 175 - "Requirement traceability"
Cohesion: 0.33
Nodes (6): Anti-patterns, Complexity budget, Examples, Layout conventions, Machine-readable contract, Requirement traceability

### Community 176 - "Wardley Map"
Cohesion: 0.33
Nodes (5): Anti-patterns, Complexity budget, Examples, Layout conventions, Wardley Map

### Community 177 - "._fit"
Cohesion: 0.25
Nodes (8): 2.1. Tầng F1: NaN/Inf Triage & MI Score (`triage.py`), 2.2. Tầng F2: Multi-Metric Variance Check (`variance.py`), 2.3. Tầng F3+7: IC Stability, FDR Gate & Term Structure (`ic_stability.py`), 2.4. Tầng F4: Fast Nonlinear Collinearity (`collinearity.py`), 2.5. Tầng F5: IC-Behavior Hierarchical Clustering (`clustering.py`), 2.6. Tầng F6: Regime-Aware MI & Synergy Detection (`regime_synergy.py`), 2.7. Tầng F8: Joint Mutual Information Selector (`jmi.py`), 2. Đặc Tả Chi Tiết 7 Tầng Lọc Tính Năng

### Community 178 - "Annotation Callout (italic-serif aside)"
Cohesion: 0.40
Nodes (5): Annotation Callout (italic-serif aside), Anti-patterns, Colors, Grammar, Rules

### Community 179 - "People"
Cohesion: 0.33
Nodes (6): Anti-patterns, Budget, Examples, Glyphs (PD subset), Layout conventions, SBGN

### Community 180 - "PipelineConfig"
Cohesion: 0.40
Nodes (3): _check_train_index(), Self, RULE-01: Train rows are chronological and disjoint from every other segment.

### Community 181 - "State Machine"
Cohesion: 0.40
Nodes (5): Anti-patterns, Examples, Layout conventions, Routing distinctions, State Machine

### Community 182 - "Venn / Set Overlap"
Cohesion: 0.40
Nodes (4): Anti-patterns, Examples, Layout conventions, Venn / Set Overlap

### Community 183 - "export_png.py"
Cohesion: 0.60
Nodes (4): exact_size_scale(), main(), Path, rasterize()

### Community 184 - "ER / Data Model"
Cohesion: 0.50
Nodes (4): Anti-patterns, ER / Data Model, Examples, Layout conventions

### Community 185 - "Flowchart"
Cohesion: 0.50
Nodes (4): Anti-patterns, Examples, Flowchart, Layout conventions

### Community 186 - "Layer Stack"
Cohesion: 0.50
Nodes (4): Anti-patterns, Examples, Layer Stack, Layout conventions

### Community 187 - "Nested Containment"
Cohesion: 0.50
Nodes (4): Anti-patterns, Examples, Layout conventions, Nested Containment

### Community 188 - "Swimlane"
Cohesion: 0.50
Nodes (4): Anti-patterns, Examples, Layout conventions, Swimlane

### Community 189 - "Timeline"
Cohesion: 0.50
Nodes (4): Anti-patterns, Examples, Layout conventions, Timeline

### Community 190 - "Tree / Hierarchy"
Cohesion: 0.50
Nodes (4): Anti-patterns, Examples, Layout conventions, Tree / Hierarchy

### Community 191 - "merge_style_into_defs"
Cohesion: 0.33
Nodes (5): 1. Kiến Trúc Tổng Thể & Luồng Dữ Liệu (Funnel Architecture), 3. Báo Cáo Phễu Lọc (Funnel Reporting), Feature Funnel Specification & Policies: Quant AutoML Pipeline, Quant AutoML Pipeline, Điểm Nổi Bật & Quy Tắc Bất Biến (Core Hard Gates)

### Community 192 - "Configuration (`config.toml`)"
Cohesion: 0.17
Nodes (12): 1. Tham số Đóng gói ONNX (`[packaging]`), 2. Tham số Kiểm thử mù OOS & Chi phí ma sát (`[backtest]`), 3. Tham số Mô phỏng Khớp lệnh Thực chiến Tier 2 (`[realworld]` — `backtesting.py`), Configuration (`config.toml`), Nhãn có hướng (`[labels.mapping]`), Phase 1 — Ingestion, Labels & Adaptive Temporal Split (`[data]`, `[labels]`, `[split]`), Phase 2 — 7-Stage Leakage-Free Feature Funnel (`[features]`), Phase 3 — Modern GBDT Tournament & Deep Optuna Tuning (`[models]`) (+4 more)

### Community 193 - "calc_rolling_ic"
Cohesion: 0.17
Nodes (11): 2. Thứ tự `Cal-A` và `Cal-B` (Audit #2), 3. Thay thế `unique_ratio` trong F2 `MultiMetricVarianceFilter` (Audit #6), 4. P-value cho BH-FDR trong F3+7 (Audit #3 — Phase 2), 5. Bar-level Sharpe vs Daily-Aggregated Net Sharpe (Audit #8 — Phase 3), Architectural & Quantitative Trade-offs — `v1.1-fix1`, Các phương án cân nhắc:, Các phương án cân nhắc:, Các phương án cân nhắc: (+3 more)

### Community 194 - "2. Sinh File Cấu Hình `config.toml` Tự Động (`scripts/generate_config.py`)"
Cohesion: 0.20
Nodes (10): 1. Yêu Cầu Hệ Thống & Cài Đặt, 2.1. Sinh cấu hình mặc định (nhãn nhị phân `is_peak_10`), 2.2. Đổi nhãn mục tiêu sang nhãn giao dịch có hướng (`allow_entry`), 2.3. Tùy chỉnh kỳ hạn (`horizon-h`, `target-horizon`) và số lượng Trials, 2.4. Sinh cấu hình cho bài toán Không giám sát (Unsupervised Mode), 2.5. Bảng tham số CLI của `scripts/generate_config.py`, 2. Sinh File Cấu Hình `config.toml` Tự Động (`scripts/generate_config.py`), 3. Chạy Toàn Bộ Pipeline (+2 more)

### Community 195 - "ic.py"
Cohesion: 0.29
Nodes (7): Anti-patterns, Card states — the type's semantic vocabulary, Complexity budget, Examples, Kanban Board, Layout conventions, Over-limit column

### Community 196 - "Export block registry (`--registry`)"
Cohesion: 0.40
Nodes (5): _fwd(), float32, float64, NDArray, Forward k-bar return at t = r[t+1] + ... + r[t+k] (``r[t+1]`` realized at t+1).

### Community 198 - "System Architecture & Technical Specifications: Quant AutoML Pipeline"
Cohesion: 0.18
Nodes (11): 1. System Overview & Core Philosophy, 2. End-to-End Pipeline Architecture Diagram, 3. Data Flow & Temporal Isolation Architecture, 4.1. Smooth SLSQP Objective ($C^2$ Differentiability), 4.2. Temperature Scaling & 100% Rank Invariance Contract (RULE-03), 4.3. Multi-Horizon Term Structure & FDR Control (RULE-08), 4.4. Friction Cost Matrix & Hysteresis Buffer (RULE-09), 4. Mathematical Formulations & Optimization Engines (+3 more)

### Community 199 - "People"
Cohesion: 0.40
Nodes (5): admin, People, robot, user, users

### Community 200 - "Danh Mục Báo Cáo & Artifacts Đầu Ra (`OUTPUTS`)"
Cohesion: 0.40
Nodes (4): 1. Cấu Trúc Thư Mục Kết Quả (`runs/default/`), 2. File Nào Lưu Các Đặc Trưng Cuối Cùng Được Chọn?, 3. Cách Đọc Báo Cáo Nghiệm Thu `backtest_report.json`, Danh Mục Báo Cáo & Artifacts Đầu Ra (`OUTPUTS`)

### Community 201 - "Repository & Codebase Structure: Quant AutoML Pipeline"
Cohesion: 0.40
Nodes (5): 1. Cây Thư Mục Tổng Thể (Repository Layout), 2. Các Tầng Kiến Trúc & Luồng Dữ Liệu (Layer Architecture), 3. Ma Trận Phân Định Trách Nhiệm Module (Responsibility Matrix), 4. Quy Tắc Ranh Giới & Phụ Thuộc (Boundaries & Coupling Rules), Repository & Codebase Structure: Quant AutoML Pipeline

### Community 202 - "4. Mathematical Formulations & Optimization Engines"
Cohesion: 0.24
Nodes (9): MonkeyPatch, Path, Integration test: Zero Look-Ahead Leakage for Phase 4 Ensemble & Calibration (RU, Audit #2: Temporal causality — Ensemble fits on Cal-A (60-70%), Calibration on C, Audit #1: In OOF_ASSISTED mode (N < 10000), Phase 4 MUST use true out-of-fold pr, Test full pipeline end-to-end integration with ensemble & calibration reports., test_oof_assisted_phase4_uses_true_oof_not_refitted_in_sample(), test_pipeline_with_ensemble_and_calibration() (+1 more)

### Community 206 - "extract_first_svg"
Cohesion: 0.40
Nodes (5): Anti-patterns, Complexity budget, Deployment, Examples, Layout conventions

### Community 208 - "Terminal Window (CLI-chrome variant)"
Cohesion: 0.29
Nodes (7): Critical rules, Grammar, Terminal Window (CLI-chrome variant), Titlebar dots, Typography, When not to use, When to use

### Community 209 - "Project / Feature Scope — `v1.1-fix1` (Quant Audit Remediation)"
Cohesion: 0.22
Nodes (8): 1. Objectives & Goals, 2. In-Scope (Must Have), 3. Out-of-Scope (Non-Goals), 4. Constraints & Assumptions, Phase 1 — Ingestion Date Slicing, Causality, Leakage & Feature Filter Rigor *(Done)*, Phase 2 — Statistical Rigor & Objective Alignment *(Done)*, Phase 3 — Time-Series Bootstrap & Production Backtest Realism *(Done)*, Project / Feature Scope — `v1.1-fix1` (Quant Audit Remediation)

### Community 210 - "_sample_params"
Cohesion: 0.18
Nodes (13): adaptive_child_weight_bounds(), adaptive_leaf_bounds(), Any, Trial, Sample-size-aware LightGBM ``min_child_samples`` bounds (prevents tiny overfit l, Sample-size-aware XGBoost ``min_child_weight`` bounds., Sample balanced, regularized hyperparameters using Optuna trial suggestions., _sample_params() (+5 more)

### Community 211 - "System Architecture & Data Flow Changes — `v1.1-fix1`"
Cohesion: 0.29
Nodes (6): 1. Tổng Quan Kiến Trúc Cải Tiến, 2.1. Chế độ `STATIC_5WAY` ($N \ge 10{,}000$), 2.2. Chế độ `OOF_ASSISTED` ($N < 10{,}000$), 2. Luồng Dữ Liệu Nhân Quả Mới (Phase 1 Fix — Audit #1 & #2), 3. Công Thức Toán Học Cải Tiến, System Architecture & Data Flow Changes — `v1.1-fix1`

### Community 212 - "Phase Breakdown"
Cohesion: 0.29
Nodes (6): Phase 1: Ingestion Date Slicing, Causality, Leakage & Feature Filter Rigor *(Active)*, Phase 2: Statistical Rigor & Objective Alignment *(Done)*, Phase 3: Time-Series Bootstrap & Production Backtest Realism *(Done)*, Phase Breakdown, Roadmap Overview, System Phases & Decomposition — `v1.1-fix1`

### Community 213 - "v1.1-fix1/PLAN.md"
Cohesion: 0.25
Nodes (8): Detailed Task Breakdown, Execution Order Overview, Final Verification Checklist (Phase 3), Implementation Plan — `v1.1-fix1` (Phase 3: Time-Series Bootstrap & Production Backtest Realism), [ ] Task 1: Moving Block Bootstrap & OOB Hurdle Gate (Audit #7), [ ] Task 2: Daily-Aggregated Net Sharpe & Deflated Sharpe Ratio (Audit #8a), [ ] Task 3: Tier 1 `next_open` Execution Alignment (Audit #8b), [ ] Task 4: Full Verification, Docs, Graphify & Single Phase 3 Commit

### Community 214 - "test_oof_assisted_phase4_uses_true_oof_not_refitted_in_sample"
Cohesion: 0.50
Nodes (4): leading_accessibility_end(), merge_style_into_defs(), Keep leading title/desc and comments ahead of a newly inserted defs., Ensure one <defs> and place a <style> with fonts + diagram CSS first.      With

### Community 216 - "Directory & File Structure — `v1.1-fix1`"
Cohesion: 0.33
Nodes (4): Component & File Responsibilities (Phase 1 Active), Design Rules & Boundaries, Directory & File Structure — `v1.1-fix1`, Directory Tree (Chỉ liệt kê các vùng bị tác động)

## Knowledge Gaps
- **974 isolated node(s):** `quantautoml`, `generate_agents_markdown.sh script`, `generate_readme.sh script`, `graphify`, `0. First-time setup — style guide gate` (+969 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **12 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `InvalidDataError` connect `str` to `InvalidDataError`, `types.py`, `.fit`, `._fit`, `make_frames`, `BaseGBDCandidate`, `test_regime_synergy.py`, `test_clustering.py`, `test_collinearity.py`, `test_variance.py`, `infer_task_type`, `test_ic_stability.py`, `ModelsConfig`, `TemperatureCalibrator`, `create_candidate`, `._fit`, `TaskType`, `config.py`, `Quantitative metrics shared across pipeline layers.`, `PipelineConfig`, `DeepOptunaTuner`, `FeatureFunnel`, `DataIngestion`, `_sample_params`, `apply_fdr_correction`, `build_preprocessor_onnx`?**
  _High betweenness centrality (0.048) - this node is a cross-community bridge._
- **Why does `evaluate_oos_performance()` connect `evaluate_oos_performance` to `config.py`, `exceptions.py`, `test_packaging_converters.py`, `runner.py`, `Self`, `create_candidate`?**
  _High betweenness centrality (0.020) - this node is a cross-community bridge._
- **Why does `FeaturesConfig` connect `types.py` to `._fit`, `InvalidDataError`, `._fit`, `make_frames`, `test_regime_synergy.py`, `.transform`, `test_clustering.py`, `test_collinearity.py`, `test_variance.py`, `test_ic_stability.py`, `._fit`, `float32`, `Self`, `FeatureFunnel`, `DataIngestion`, `validate_numerical_precision`, `str`, `OOSBacktestRunner`, `build_preprocessor_onnx`?**
  _High betweenness centrality (0.015) - this node is a cross-community bridge._
- **Are the 169 inferred relationships involving `InvalidDataError` (e.g. with `DataIngestion` and `.load()`) actually correct?**
  _`InvalidDataError` has 169 INFERRED edges - model-reasoned connections that need verification._
- **Are the 58 inferred relationships involving `FitContext` (e.g. with `FeatureSelectionError` and `InvalidDataError`) actually correct?**
  _`FitContext` has 58 INFERRED edges - model-reasoned connections that need verification._
- **Are the 52 inferred relationships involving `FeaturesConfig` (e.g. with `ICBehaviorClusterFilter` and `F5: IC-behavior hierarchical clustering with a Max-MI representative per cluster`) actually correct?**
  _`FeaturesConfig` has 52 INFERRED edges - model-reasoned connections that need verification._
- **Are the 46 inferred relationships involving `TaskType` (e.g. with `Infer the learning task (binary / multiclass / unsupervised) from the label vect` and `Infer the task. Missing labels (NaN / None) are unlabeled rows: they are     ign`) actually correct?**
  _`TaskType` has 46 INFERRED edges - model-reasoned connections that need verification._