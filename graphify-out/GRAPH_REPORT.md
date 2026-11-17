# Graph Report - quantautoml  (2026-11-17)

## Corpus Check
- 60 files · ~148,818 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 838 nodes · 2661 edges · 21 communities detected
- Extraction: 50% EXTRACTED · 50% INFERRED · 0% AMBIGUOUS · INFERRED: 1318 edges (avg confidence: 0.65)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- [[_COMMUNITY_Community 0|Community 0]]
- [[_COMMUNITY_Community 1|Community 1]]
- [[_COMMUNITY_Community 2|Community 2]]
- [[_COMMUNITY_Community 3|Community 3]]
- [[_COMMUNITY_Community 4|Community 4]]
- [[_COMMUNITY_Community 5|Community 5]]
- [[_COMMUNITY_Community 6|Community 6]]
- [[_COMMUNITY_Community 7|Community 7]]
- [[_COMMUNITY_Community 8|Community 8]]
- [[_COMMUNITY_Community 9|Community 9]]
- [[_COMMUNITY_Community 10|Community 10]]
- [[_COMMUNITY_Community 11|Community 11]]
- [[_COMMUNITY_Community 12|Community 12]]
- [[_COMMUNITY_Community 13|Community 13]]
- [[_COMMUNITY_Community 14|Community 14]]
- [[_COMMUNITY_Community 15|Community 15]]
- [[_COMMUNITY_Community 16|Community 16]]
- [[_COMMUNITY_Community 17|Community 17]]
- [[_COMMUNITY_Community 18|Community 18]]
- [[_COMMUNITY_Community 19|Community 19]]
- [[_COMMUNITY_Community 24|Community 24]]

## God Nodes (most connected - your core abstractions)
1. `InvalidDataError` - 133 edges
2. `FitContext` - 76 edges
3. `FeaturesConfig` - 68 edges
4. `TaskType` - 61 edges
5. `BaseFeatureFilter` - 55 edges
6. `FilterReport` - 53 edges
7. `FeatureSelectionError` - 47 edges
8. `FeatureFunnel` - 45 edges
9. `make_frames()` - 33 edges
10. `ModelsConfig` - 31 edges

## Surprising Connections (you probably didn't know these)
- `Six-Layer Unidirectional Architecture` --semantically_similar_to--> `System Architecture`  [INFERRED] [semantically similar]
  CLAUDE.md → docs/plan/ARCHITECTURE.md
- `FilterReport` --calls--> `test_filter_report_is_frozen()`  [INFERRED]
  src/quantautoml/common/types.py → tests/unit/test_features_base.py
- `FeaturesConfig` --uses--> `Pipeline: config -> ingestion -> task inference -> split -> artifacts on disk.`  [INFERRED]
  src/quantautoml/common/config.py → tests/unit/test_pipeline.py
- `ModelsConfig` --calls--> `test_invalid_trials_rejected()`  [INFERRED]
  src/quantautoml/common/config.py → tests/unit/test_models_config.py
- `ModelsConfig` --calls--> `test_invalid_diversity_threshold_rejected()`  [INFERRED]
  src/quantautoml/common/config.py → tests/unit/test_models_config.py

## Hyperedges (group relationships)
- **Leakage and rank-safety gates** — claude_rule01, claude_rule03, claude_rule04, claude_rule08 [INFERRED 0.75]
- **Five-phase pipeline** — phases_phase1, phases_phase2, phases_phase3, phases_phase4, phases_phase5 [EXTRACTED 0.90]
- **Feature funnel stages F1-F8** — s02_f1_triage, s02_f2_variance, s02_f3_ic_stability, s02_f4_fast_dcor, s02_f5_clustering, s02_f6_regime_synergy, s02_f8_jmi [EXTRACTED 1.00]
- **SLSQP, bootstrap shrinkage and temperature calibration flow** — s05_slsqp_optimizer, s05_bootstrap_shrinkage, s05_temperature_scaling, s05_rank_pearson_objective [EXTRACTED 0.95]
- **Friction-aware OOS acceptance** — s07_backtest_runner, s07_friction_cost_model, s07_hysteresis_buffer, s07_oos_acceptance_gate [EXTRACTED 0.95]

## Communities

### Community 0 - "Community 0"
Cohesion: 0.06
Nodes (73): BaseFeatureFilter, FitContext, Base contract for every feature filter (fit on Train only, transform anywhere)., Template-method base: validation lives here, logic in `_fit`/`_transform`., Set `output_names_` and `report_`., Apply the fitted selection without mutating state., BaseFeatureFilter, Fit model on training segment. Both X and y must be numpy arrays. (+65 more)

### Community 1 - "Community 1"
Cohesion: 0.08
Nodes (73): ic_behavior_distance(), FeatureFunnel, _cfg(), _ctx(), _fit(), _label(), _loading(), _noise() (+65 more)

### Community 2 - "Community 2"
Cohesion: 0.08
Nodes (68): _build_parser(), main(), Command-line entrypoint: ``quantautoml run --config config.toml --out out/``., PipelineConfig, Enum, Exception, DataLeakageError, QuantAutoMLError (+60 more)

### Community 3 - "Community 3"
Cohesion: 0.06
Nodes (56): AGENTS.md Agent Guide, System Architecture, Multi-Horizon Term Structure & FDR, Friction Cost Model & Net Sharpe, Non-Functional Requirements & Safety Guards, Smooth SLSQP Objective C2, Temperature Scaling Rank Invariance, Agent TDD Workflow & Session Onboarding (+48 more)

### Community 4 - "Community 4"
Cohesion: 0.08
Nodes (39): BaseModel, DataConfig, LabelsConfig, ModelsConfig, Pipeline configuration loaded from a TOML file and validated with Pydantic., SplitConfig, _Strict, DecisionReport (+31 more)

### Community 5 - "Community 5"
Cohesion: 0.08
Nodes (48): calc_ic_sharpe(), calc_icir(), calc_rank_ic(), calc_rolling_ic(), calc_rolling_ic_matrix(), Spearman Rank IC, non-overlapping rolling IC (per column / matrix) and ICIR (RUL, Return mean/std (ddof=1, NaN ignored); 0.0 if std == 0 or < 2 points., Calculate IC Sharpe = ICIR * sqrt(annualization_factor).      Returns 0.0 if std (+40 more)

### Community 6 - "Community 6"
Cohesion: 0.08
Nodes (26): ABC, _fit(), _transform(), BaseGBDCandidate, CatBoostCandidate, create_candidate(), LightGBMCandidate, Modern GBDT Trio model wrappers (LightGBM, CatBoost, XGBoost) with unified contr (+18 more)

### Community 7 - "Community 7"
Cohesion: 0.13
Nodes (37): build_config_content(), detect_file_info(), main(), _build_report(), _fmt(), _oof_folds(), _take(), verify_split() (+29 more)

### Community 8 - "Community 8"
Cohesion: 0.16
Nodes (36): build_config(), make_frames(), Shared pytest fixtures for the quantautoml test-suite., Build (features, labels) frames with forward-looking labels.      Labels: ``y_di, Write frames to disk and return a PipelineConfig pointing at them., _load(), test_auxiliary_labels_keep_their_nan(), test_boolean_features_are_cast_to_float32() (+28 more)

### Community 9 - "Community 9"
Cohesion: 0.18
Nodes (32): label_regimes(), _cfg(), _ctx(), _fit(), _noise(), mi_floor = 0.0 keeps only features whose corrected MI is > 0 (strict inequality), _ret(), _small_regime() (+24 more)

### Community 10 - "Community 10"
Cohesion: 0.06
Nodes (34): AdaptiveTimeSeriesSplitter (Purging & Embargo), UD-02 Boolean to Float32 auto-cast, DataIngestion (UTC us, float32, inner join), horizon_h Purging for forward-looking labels, infer_task_type (Binary/Multiclass/Unsupervised), NEED_UPDATE: design corrections (timestamp precision, bool/string, mid-NaN labels), Adaptive OOF-Assisted Split (N<10000), PipelineConfig (Pydantic TOML) (+26 more)

### Community 11 - "Community 11"
Cohesion: 0.15
Nodes (26): load_config(), write_toml(), test_cluster_distance_open_unit_interval(), test_defaults_are_applied(), test_f6_thresholds_defaults(), test_f6_thresholds_must_be_non_negative(), test_f6_thresholds_parse(), test_fdr_alpha_must_be_in_zero_to_005() (+18 more)

### Community 12 - "Community 12"
Cohesion: 0.15
Nodes (26): fast_dcor(), F4: fast nonlinear collinearity filter based on distance correlation (dCor, AVL), _standardize(), _cfg(), _f32(), _fit(), test_constant_column_is_never_collinear(), test_default_threshold_drops_monotone_duplicate_tie_keeps_earlier() (+18 more)

### Community 13 - "Community 13"
Cohesion: 0.25
Nodes (25): _cfg(), _ctx(), _fit(), _labels(), _signal(), test_all_columns_gated_yields_empty_output(), test_all_nan_and_high_nan_ratio_columns_dropped(), test_bool_imputation_uses_train_state_only() (+17 more)

### Community 14 - "Community 14"
Cohesion: 0.26
Nodes (21): _cfg(), _ctx(), _fit(), _noise(), test_all_labels_masked_no_crash(), test_constant_nan_and_inf_columns_no_warnings(), test_deterministic_order_and_transform_keeps_original_order(), test_first_selected_is_max_mi() (+13 more)

### Community 15 - "Community 15"
Cohesion: 0.25
Nodes (19): _cfg(), _ctx(), _fit(), _normal(), test_all_nan_column_dropped_no_warning(), test_balanced_binary_kept(), test_constant_column_dropped_good_kept(), test_low_cv_dropped() (+11 more)

### Community 16 - "Community 16"
Cohesion: 0.23
Nodes (15): infer_task_type(), test_binary_minus_one_one_is_remapped_to_zero_one(), test_binary_string_labels(), test_binary_zero_one_float_labels(), test_empty_labels_are_rejected(), test_inf_labels_are_rejected(), test_more_than_ten_classes_is_rejected_as_regression(), test_multiclass_between_three_and_ten_integer_classes() (+7 more)

### Community 17 - "Community 17"
Cohesion: 0.19
Nodes (10): contingency_mi(), quantile_bin_codes(), Private helpers shared by F6 and F8: quantile bin codes and contingency-table MI, Per-column ``n_bins``-quantile bin codes and the code range of each column., Plug-in MI (nats) of codes ``a`` in ``[0, na)`` vs class codes ``yc`` in ``[0, n, Shared quantile binning and contingency-table MI used by F6 and F8., test_contingency_mi_matches_sklearn(), test_contingency_mi_single_class_is_zero() (+2 more)

### Community 18 - "Community 18"
Cohesion: 0.22
Nodes (10): apply_fdr_correction(), Benjamini-Hochberg FDR control (RULE-08)., Return a mask of accepted hypotheses, in the original order.      k = max{i : p_, Unit tests for quantautoml.metrics.fdr., test_empty(), test_hand_example(), test_invalid_p_values(), test_matches_scipy_reference() (+2 more)

### Community 19 - "Community 19"
Cohesion: 0.17
Nodes (13): Isotonic Regression (rejected: ties), Platt Scaling, Temperature Scaling (RULE-03, rank invariant), Rationale: Temperature Scaling preserves 100% Rank IC, DistributionShiftWarning (z>3), ONNXBundleExporter, PrecisionValidator (|y_py-y_onnx|<1e-4, RULE-06), predictor.py decoupled inference (numpy+onnxruntime, RULE-07) (+5 more)

### Community 24 - "Community 24"
Cohesion: 1.0
Nodes (1): Git Conventions & Branching

## Knowledge Gaps
- **50 isolated node(s):** `Core enums, data containers and report dataclasses for Phase 1.`, `Pipeline configuration loaded from a TOML file and validated with Pydantic.`, `Custom exceptions shared across the quantautoml pipeline.`, `Base class for every error raised by quantautoml.`, `Input data or configuration violates a pipeline data contract.` (+45 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **Thin community `Community 24`** (1 nodes): `Git Conventions & Branching`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `InvalidDataError` connect `Community 0` to `Community 1`, `Community 2`, `Community 4`, `Community 5`, `Community 6`, `Community 7`, `Community 9`, `Community 12`, `Community 16`, `Community 18`?**
  _High betweenness centrality (0.301) - this node is a cross-community bridge._
- **Why does `FeaturesConfig` connect `Community 0` to `Community 1`, `Community 2`, `Community 4`, `Community 5`, `Community 8`, `Community 9`, `Community 12`, `Community 13`, `Community 14`, `Community 15`?**
  _High betweenness centrality (0.056) - this node is a cross-community bridge._
- **Why does `run_pipeline()` connect `Community 2` to `Community 8`, `Community 1`, `Community 11`, `Community 7`?**
  _High betweenness centrality (0.055) - this node is a cross-community bridge._
- **Are the 130 inferred relationships involving `InvalidDataError` (e.g. with `_SelectedLabel` and `DataIngestion`) actually correct?**
  _`InvalidDataError` has 130 INFERRED edges - model-reasoned connections that need verification._
- **Are the 75 inferred relationships involving `FitContext` (e.g. with `RegimeSynergyFilter` and `F6: regime-aware mutual information and synergistic-pair protection.`) actually correct?**
  _`FitContext` has 75 INFERRED edges - model-reasoned connections that need verification._
- **Are the 66 inferred relationships involving `FeaturesConfig` (e.g. with `RegimeSynergyFilter` and `F6: regime-aware mutual information and synergistic-pair protection.`) actually correct?**
  _`FeaturesConfig` has 66 INFERRED edges - model-reasoned connections that need verification._
- **Are the 58 inferred relationships involving `TaskType` (e.g. with `TaskInference` and `Infer the learning task (binary / multiclass / unsupervised) from the label vect`) actually correct?**
  _`TaskType` has 58 INFERRED edges - model-reasoned connections that need verification._