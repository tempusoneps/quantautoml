# Graph Report - .  (2026-11-11)

## Corpus Check
- Corpus is ~36,800 words - fits in a single context window. You may not need a graph.

## Summary
- 302 nodes · 674 edges · 13 communities detected
- Extraction: 58% EXTRACTED · 42% INFERRED · 0% AMBIGUOUS · INFERRED: 285 edges (avg confidence: 0.71)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- [[_COMMUNITY_Config & Exceptions|Config & Exceptions]]
- [[_COMMUNITY_Architecture & Rules|Architecture & Rules]]
- [[_COMMUNITY_Ingestion Tests|Ingestion Tests]]
- [[_COMMUNITY_Ingestion & Splitting Design|Ingestion & Splitting Design]]
- [[_COMMUNITY_Splitter & Tests|Splitter & Tests]]
- [[_COMMUNITY_Task Inference|Task Inference]]
- [[_COMMUNITY_Config Loading Tests|Config Loading Tests]]
- [[_COMMUNITY_Calibration & Packaging|Calibration & Packaging]]
- [[_COMMUNITY_Config Generator & CLI Tests|Config Generator & CLI Tests]]
- [[_COMMUNITY_Leakage Tests|Leakage Tests]]
- [[_COMMUNITY_Pydantic Config Models|Pydantic Config Models]]
- [[_COMMUNITY_CLI Entrypoint|CLI Entrypoint]]
- [[_COMMUNITY_Git Conventions|Git Conventions]]

## God Nodes (most connected - your core abstractions)
1. `make_frames()` - 31 edges
2. `build_config()` - 29 edges
3. `_load()` - 29 edges
4. `InvalidDataError` - 24 edges
5. `make_dataset()` - 23 edges
6. `infer_task_type()` - 19 edges
7. `AdaptiveTimeSeriesSplitter` - 17 edges
8. `Dataset` - 16 edges
9. `load_config()` - 15 edges
10. `DataIngestion` - 15 edges

## Surprising Connections (you probably didn't know these)
- `System Architecture` --semantically_similar_to--> `Six-Layer Unidirectional Architecture`  [INFERRED] [semantically similar]
  docs/plan/ARCHITECTURE.md → CLAUDE.md
- `Shared pytest fixtures for the quantautoml test-suite.` --uses--> `PipelineConfig`  [INFERRED]
  tests/conftest.py → src/quantautoml/common/config.py
- `Build (features, labels) frames with forward-looking labels.      Labels: ``y_di` --uses--> `PipelineConfig`  [INFERRED]
  tests/conftest.py → src/quantautoml/common/config.py
- `Write frames to disk and return a PipelineConfig pointing at them.` --uses--> `PipelineConfig`  [INFERRED]
  tests/conftest.py → src/quantautoml/common/config.py
- `test_split_api_has_no_shuffle_parameter()` --calls--> `AdaptiveTimeSeriesSplitter`  [INFERRED]
  tests/unit/test_splitter.py → src/quantautoml/data/splitter.py

## Hyperedges (group relationships)
- **Leakage and rank-safety gates** — claude_rule01, claude_rule03, claude_rule04, claude_rule08 [INFERRED 0.75]
- **Five-phase pipeline** — phases_phase1, phases_phase2, phases_phase3, phases_phase4, phases_phase5 [EXTRACTED 0.90]
- **Feature funnel stages F1-F8** — s02_f1_triage, s02_f2_variance, s02_f3_ic_stability, s02_f4_fast_dcor, s02_f5_clustering, s02_f6_regime_synergy, s02_f8_jmi [EXTRACTED 1.00]
- **SLSQP, bootstrap shrinkage and temperature calibration flow** — s05_slsqp_optimizer, s05_bootstrap_shrinkage, s05_temperature_scaling, s05_rank_pearson_objective [EXTRACTED 0.95]
- **Friction-aware OOS acceptance** — s07_backtest_runner, s07_friction_cost_model, s07_hysteresis_buffer, s07_oos_acceptance_gate [EXTRACTED 0.95]

## Communities

### Community 0 - "Config & Exceptions"
Cohesion: 0.09
Nodes (53): PipelineConfig, Enum, Exception, DataLeakageError, InvalidDataError, QuantAutoMLError, Custom exceptions shared across the quantautoml pipeline., Timestamps are non-monotonic, duplicated or cannot be aligned. (+45 more)

### Community 1 - "Architecture & Rules"
Cohesion: 0.06
Nodes (56): AGENTS.md Agent Guide, System Architecture, Multi-Horizon Term Structure & FDR, Friction Cost Model & Net Sharpe, Non-Functional Requirements & Safety Guards, Smooth SLSQP Objective C2, Temperature Scaling Rank Invariance, Agent TDD Workflow & Session Onboarding (+48 more)

### Community 2 - "Ingestion Tests"
Cohesion: 0.19
Nodes (33): build_config(), make_frames(), Build (features, labels) frames with forward-looking labels.      Labels: ``y_di, Write frames to disk and return a PipelineConfig pointing at them., _load(), test_auxiliary_labels_keep_their_nan(), test_boolean_features_are_cast_to_float32(), test_duplicate_timestamps_are_rejected() (+25 more)

### Community 3 - "Ingestion & Splitting Design"
Cohesion: 0.06
Nodes (34): AdaptiveTimeSeriesSplitter (Purging & Embargo), UD-02 Boolean to Float32 auto-cast, DataIngestion (UTC us, float32, inner join), horizon_h Purging for forward-looking labels, infer_task_type (Binary/Multiclass/Unsupervised), NEED_UPDATE: design corrections (timestamp precision, bool/string, mid-NaN labels), Adaptive OOF-Assisted Split (N<10000), PipelineConfig (Pydantic TOML) (+26 more)

### Community 4 - "Splitter & Tests"
Cohesion: 0.23
Nodes (23): verify_split(), make_dataset(), test_gap_between_every_adjacent_segment_equals_horizon(), test_horizon_too_large_for_oof_blocks_is_rejected(), test_horizon_too_large_for_static_segments_is_rejected(), test_missing_class_in_a_segment_raises_a_warning(), test_mode_is_chosen_from_row_count(), test_non_positive_horizon_is_rejected() (+15 more)

### Community 5 - "Task Inference"
Cohesion: 0.17
Nodes (19): infer_task_type(), Infer the learning task (binary / multiclass / unsupervised) from the label vect, Infer the task. Missing labels (NaN / None) are unlabeled rows: they are     ign, TaskInference, test_binary_minus_one_one_is_remapped_to_zero_one(), test_binary_string_labels(), test_binary_zero_one_float_labels(), test_empty_labels_are_rejected() (+11 more)

### Community 6 - "Config Loading Tests"
Cohesion: 0.27
Nodes (13): load_config(), Shared pytest fixtures for the quantautoml test-suite., write_toml(), test_defaults_are_applied(), test_horizon_below_one_is_rejected(), test_labels_path_requires_selected(), test_load_config_parses_valid_toml(), test_min_match_ratio_must_be_in_unit_interval() (+5 more)

### Community 7 - "Calibration & Packaging"
Cohesion: 0.17
Nodes (13): Isotonic Regression (rejected: ties), Platt Scaling, Temperature Scaling (RULE-03, rank invariant), Rationale: Temperature Scaling preserves 100% Rank IC, DistributionShiftWarning (z>3), ONNXBundleExporter, PrecisionValidator (|y_py-y_onnx|<1e-4, RULE-06), predictor.py decoupled inference (numpy+onnxruntime, RULE-07) (+5 more)

### Community 8 - "Config Generator & CLI Tests"
Cohesion: 0.33
Nodes (8): build_config_content(), detect_file_info(), main(), CLI: ``quantautoml run --config X --out DIR``., test_missing_subcommand_exits_with_usage_error(), test_run_command_returns_nonzero_on_pipeline_error(), test_run_command_writes_artifacts(), _toml()

### Community 9 - "Leakage Tests"
Cohesion: 0.51
Nodes (9): _fwd(), _labels_unchanged_when_future_masked(), _run(), test_horizon_smaller_than_longest_label_is_blocked_before_splitting(), test_no_segment_label_depends_on_a_later_segment(), test_oof_folds_do_not_leak_validation_prices_into_training_labels(), test_reports_are_json_serialisable(), test_training_segment_never_contains_a_later_timestamp() (+1 more)

### Community 10 - "Pydantic Config Models"
Cohesion: 0.36
Nodes (6): BaseModel, DataConfig, LabelsConfig, Pipeline configuration loaded from a TOML file and validated with Pydantic., SplitConfig, _Strict

### Community 11 - "CLI Entrypoint"
Cohesion: 0.67
Nodes (3): _build_parser(), main(), Command-line entrypoint: ``quantautoml run --config config.toml --out out/``.

### Community 16 - "Git Conventions"
Cohesion: 1.0
Nodes (1): Git Conventions & Branching

## Knowledge Gaps
- **41 isolated node(s):** `Core enums, data containers and report dataclasses for Phase 1.`, `Pipeline configuration loaded from a TOML file and validated with Pydantic.`, `Custom exceptions shared across the quantautoml pipeline.`, `Base class for every error raised by quantautoml.`, `Input data or configuration violates a pipeline data contract.` (+36 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **Thin community `Git Conventions`** (1 nodes): `Git Conventions & Branching`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `InvalidDataError` connect `Config & Exceptions` to `Splitter & Tests`, `Task Inference`?**
  _High betweenness centrality (0.071) - this node is a cross-community bridge._
- **Why does `_load()` connect `Ingestion Tests` to `Config & Exceptions`?**
  _High betweenness centrality (0.070) - this node is a cross-community bridge._
- **Why does `infer_task_type()` connect `Task Inference` to `Config & Exceptions`?**
  _High betweenness centrality (0.064) - this node is a cross-community bridge._
- **Are the 29 inferred relationships involving `make_frames()` (e.g. with `test_run_command_writes_artifacts()` and `test_run_command_returns_nonzero_on_pipeline_error()`) actually correct?**
  _`make_frames()` has 29 INFERRED edges - model-reasoned connections that need verification._
- **Are the 27 inferred relationships involving `build_config()` (e.g. with `test_run_pipeline_writes_reports_and_segments()` and `test_loads_parquet_and_csv()`) actually correct?**
  _`build_config()` has 27 INFERRED edges - model-reasoned connections that need verification._
- **Are the 2 inferred relationships involving `_load()` (e.g. with `.load()` and `DataIngestion`) actually correct?**
  _`_load()` has 2 INFERRED edges - model-reasoned connections that need verification._
- **Are the 21 inferred relationships involving `InvalidDataError` (e.g. with `_SelectedLabel` and `DataIngestion`) actually correct?**
  _`InvalidDataError` has 21 INFERRED edges - model-reasoned connections that need verification._