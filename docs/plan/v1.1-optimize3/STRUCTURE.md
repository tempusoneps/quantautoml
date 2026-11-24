# Repository & Module Structure — `v1.1-optimize3`

Tài liệu này xác định chính xác các tệp nguồn, tệp kiểm thử và tài liệu bị tác động trong `v1.1-optimize3`, tuân thủ quy tắc phụ thuộc một chiều trong [`docs/STRUCTURE.md`](../../STRUCTURE.md).

---

## 1. Cây Thư Mục Tác Động (Affected Files)

```text
quantautoml/
├── docs/
│   ├── CONFIGURATION.md                       # [Modify - P1, P2] Cập nhật mô tả ic_window, signal_ema_span, min_hold_bars
│   ├── OUTPUTS.md                             # [Modify - P3] Đồng bộ cây 01_data..06_backtest & bổ sung 06_backtest/03_regime_diagnostics.json
│   ├── STRUCTURE.md                           # [Modify - P3] Bổ sung backtest/diagnostics.py vào cây nguồn
│   └── plan/
│       └── v1.1-optimize3/                    # [Create] Bộ tài liệu kế hoạch cho v1.1-optimize3
│           ├── SCOPE.md
│           ├── PHASES.md
│           ├── STRUCTURE.md
│           ├── ARCHITECTURE.md
│           ├── TRADE-OFF.md
│           └── PLAN.md
├── scripts/
│   └── generate_config.py                     # [Modify - P1, P2] Đồng bộ tham số mới vào template config.toml
├── src/quantautoml/
│   ├── cli.py                                 # [Modify - P3] Thêm subcommand `quantautoml diagnose --run-dir <dir>`
│   ├── common/
│   │   └── config.py                          # [Modify - P1, P2] Thêm ic_window (ModelsConfig, EnsembleConfig), signal_ema_span & min_hold_bars (BacktestConfig)
│   ├── models/
│   │   ├── tournament.py                      # [Modify - P1] Nâng cấp fold_subwindow_ics nhận ic_window tùy chọn + cận trên thích ứng 500
│   │   └── tuner.py                           # [Modify - P1] Truyền self.cfg.ic_window vào fold_subwindow_ics
│   ├── ensemble/
│   │   └── bootstrap.py                       # [Modify - P1] Thêm _resolve_ic_window, nâng cấp _calc_series_ic_sharpe & bổ sung field chẩn đoán vào EnsembleWeightsResult
│   └── backtest/
│       ├── __init__.py                        # [Modify - P3] Export evaluate_bundle_segments & SegmentDiagnostic
│       ├── buffer.py                          # [Modify - P2] Thêm smooth_scores (NaN-safe causal EMA), NaN-safe rolling_percentile, và min_hold_bars vào HysteresisBuffer
│       ├── runner.py                          # [Modify - P2, P3] Truyền signal_ema_span, min_hold_bars vào HysteresisBuffer; trích helper đánh giá dùng chung cho diagnostics
│       └── diagnostics.py                     # [Create - P3] Đánh giá read-only đa segment (cal_a/cal_b/val/oos) & so sánh close vs next_open
└── tests/
    └── unit/
        ├── test_models_config.py              # [Modify - P1] Kiểm thử validation cho ModelsConfig.ic_window
        ├── test_ensemble_config.py            # [Modify - P1] Kiểm thử validation cho EnsembleConfig.ic_window
        ├── test_backtest_config.py            # [Modify - P2] Kiểm thử validation cho signal_ema_span & min_hold_bars
        ├── test_tournament.py                 # [Modify - P1] Kiểm thử fold_subwindow_ics với n lớn, mẫu nhỏ, ic_window tường minh & ic_window < 3
        ├── test_tuner.py                      # [Modify - P1] Kiểm thử DeepOptunaTuner truyền cfg.ic_window xuống fold_subwindow_ics
        ├── test_bootstrap_shrinkage.py        # [Modify - P1] Kiểm thử cửa sổ thích ứng, nhánh K=1, clamp mẫu nhỏ & field chẩn đoán mới
        ├── test_buffer.py                     # [Modify - P2] Kiểm thử causality của EMA, NaN safety (smooth_scores + rolling_percentile), min_hold_bars và giảm turnover
        ├── test_runner.py                     # [Modify - P2] Kiểm thử OOSBacktestRunner truyền signal_ema_span & min_hold_bars vào HysteresisBuffer
        ├── test_cli.py                        # [Modify - P3] Kiểm thử subcommand `quantautoml diagnose --run-dir <dir>`
        └── test_backtest_diagnostics.py       # [Create - P3] Kiểm thử báo cáo đa regime read-only & xử lý segment skipped
```

---

## 2. Hợp Đồng Giao Diện & Trách Nhiệm (Interfaces & Boundaries)

| Module | Thay đổi Interface | Ràng buộc phụ thuộc |
| :--- | :--- | :--- |
| `common/config.py` | `ModelsConfig.ic_window: int \| None = Field(default=None, ge=20)`; `EnsembleConfig.ic_window: int \| None = Field(default=None, ge=20)`; `BacktestConfig.signal_ema_span: int = Field(default=1, ge=1)`; `BacktestConfig.min_hold_bars: int = Field(default=0, ge=0)`. | Không import tầng trên (`models`, `ensemble`, `backtest`). |
| `models/tournament.py` | `fold_subwindow_ics(pred, y_fold, *, ic_window: int \| None = None) -> list[float]` (ném `InvalidDataError` nếu truyền trực tiếp `ic_window < 3`). | Chỉ dùng `numpy`, `common`, `metrics`. |
| `models/tuner.py` | Truyền `ic_window=self.cfg.ic_window` khi gọi `fold_subwindow_ics(...)`. | Không đổi signature public `DeepOptunaTuner.tune()`. |
| `ensemble/bootstrap.py` | `_resolve_ic_window(n: int, ic_window: int \| None) -> int` (validate `ic_window >= 3`); `_calc_series_ic_sharpe(pred, target, *, ic_window: int \| None = None) -> float`; `EnsembleWeightsResult` thêm `ic_window_used: int = 0`, `single_model_scores: list[float] = Field(default_factory=list)`, `overall_rank_ic: float = 0.0` (điền cho cả `K == 1` và `K > 1`). | Không đổi logic SLSQP (`RULE-04`) và ngưỡng hurdle (`RULE-05`). |
| `backtest/buffer.py` | `HysteresisBuffer(top_pct, exit_pct, rank_window, *, signal_ema_span: int = 1, min_hold_bars: int = 0)` (validate `signal_ema_span >= 1`, `min_hold_bars >= 0`); thêm `smooth_scores(scores) -> NDArray[np.float32]`; `rolling_percentile` NaN-safe. | Thuần `numpy`, hoàn toàn nhân quả (`RULE-01`). |
| `backtest/runner.py` & `backtest/diagnostics.py` | Tách helper dùng chung đánh giá 1 vector `positions` theo `close` / `next_open`; `evaluate_bundle_segments(bundle_dir, segments, feature_names, cfg) -> dict[str, SegmentDiagnostic]` (tách `SegmentDiagnostic` khỏi gate `"latency"`). | Chỉ đọc bundle qua `predictor_loader`, tuyệt đối không `fit()`. |
| `cli.py` | Thêm subcommand `quantautoml diagnose --run-dir <dir>` gọi `evaluate_bundle_segments` (fallback `BacktestConfig()` nếu thiếu `[backtest]`) và ghi `06_backtest/03_regime_diagnostics.json`. | Không thay đổi hành vi `quantautoml run`. |
