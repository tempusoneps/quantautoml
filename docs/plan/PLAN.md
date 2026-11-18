# Implementation Plan: Quant AutoML Pipeline — Phase 5

> **For agentic workers:** REQUIRED SUB-SKILL: dùng `superpowers:subagent-driven-development` (khuyến nghị) hoặc `superpowers:executing-plans` để thực thi plan theo từng task. TDD bắt buộc (RED → GREEN → REFACTOR); cập nhật `- [x]` trong plan này sau mỗi task theo RULE-10.

**Active Phase:** Phase 5 — ONNX Model Bundle Packaging & Friction-Aware Blind OOS Backtesting
**Branch:** `develop/v1.0-phase5`
**Goal:** Đóng gói pipeline đã huấn luyện (funnel → GBDT → ensemble → temperature) thành ONNX Model Bundle độc lập (`numpy` + `onnxruntime`), chặn xuất xưởng nếu sai số số học ≥ $10^{-4}$, rồi nghiệm thu mù trên OOS với phí + trượt giá + Hysteresis Buffer.
**Architecture:** `packaging/` nhận `FeatureFunnel` (qua `PreprocessorSpec` định nghĩa ở `common`), các `TunedCandidate`, `EnsembleWeightsResult`, `CalibrationResult`; sinh `preprocessor.onnx`, `model_*.onnx`, `ensemble_meta.json`, `features_schema.json`, `predictor.py`. `backtest/` chỉ tiêu thụ bundle (qua `predictor.py`) + `split.oos`, tính tín hiệu → vị thế (Hysteresis) → lợi nhuận ròng (Friction) → chỉ số nghiệm thu, **không fit bất cứ thứ gì**. Backtest viết bằng `numpy` vector hóa (xem Quyết định D1).
**Tech Stack:** `onnx`, `onnxruntime`, `onnxmltools`, `skl2onnx` (chỉ ở phía export), `numpy`, `scipy.stats`, `pydantic` v2, `pytest`.
**Spec:** [`PHASES.md`](PHASES.md) (Phase 5), [`ARCHITECTURE.md`](ARCHITECTURE.md) §3.4 & §5, [`TRADE-OFF.md`](TRADE-OFF.md) §6, [`STRUCTURE.md`](STRUCTURE.md), [`06-packaging/`](06-packaging/), [`07-backtest/`](07-backtest/).

---

## Quyết định cần duyệt (đã giả định trong plan — sửa nếu không đồng ý)

- **D1 — Không dùng thư viện `backtesting.py`.** Docs ghi "qua `backtesting.py`", nhưng `STRUCTURE.md` đã định nghĩa `friction.py/buffer.py/evaluator.py/runner.py` và công thức $R_{net}$ cần kiểm soát từng bước; thư viện đó tính phí theo lệnh, chậm trên 95k nến 5 phút, và không có hook cho Hysteresis theo phân vị. Plan dùng mô phỏng `numpy` vector hóa.
- **D2 — Dữ liệu là 1 tài sản, nến 5 phút** (`datasets/feature.parquet`: 95.768 dòng × 825 cột, có cột `Close`). Vì không có cắt ngang tài sản, "Top 20% / ngưỡng 30%" được hiểu là **phân vị cuốn chiếu nhân quả (causal rolling percentile) của điểm tín hiệu trong cửa sổ `rank_window` nến quá khứ**: vào Long khi pct ≥ $1-\text{top\_pct}$ (0.80), giữ cho tới khi pct < $1-\text{exit\_pct}$ (0.70); Short đối xứng (pct ≤ 0.20, thoát khi > 0.30).
- **D3 — Công thức chi phí theo đúng `ARCHITECTURE.md` §3.4:** $\text{Turnover}_t=\tfrac12\sum_j|w_{j,t}-w_{j,t^-}|$, $R_{net}=R_{gross}-\text{Turnover}\times(\text{fee}+\text{slip})$. Với 1 tài sản, đảo vị thế $-1\to+1$ chỉ tính turnover = 1 (không phải 2). Nếu muốn bảo thủ hơn (tính $\sum|\Delta w|$), đổi 1 dòng ở Task 9.
- **D4 — Lợi nhuận gộp** $R_{gross,t}=\text{pos}_t\cdot(\text{Close}_{t+1}/\text{Close}_t-1)$, lấy `Close` từ **X thô** của `split.oos` (cột `price_col`, mặc định `"Close"`), không qua funnel. Annualization suy ra từ timestamp (`n_bars / số năm`), không hard-code.
- **D5 — Fail gate nghiệm thu OOS không ném exception**: ghi `backtest_report.json` với `accepted=false` và danh sách gate trượt. Chỉ `PrecisionValidationError` (RULE-06) chặn cứng.

---

## Global Constraints & Hard Gates

- **RULE-01:** Packaging và Backtest **không gọi `fit()`**. Thống kê z-score cho drift chỉ tính trên **Train**. `split.oos` chỉ được đọc bởi `OOSBacktestRunner`; index OOS phải rời Train/Val/Cal, vi phạm → `DataLeakageError`.
- **RULE-02:** Rank IC (Spearman) + ICIR làm thước đo OOS; không dùng MSE/Accuracy làm tiêu chí nghiệm thu.
- **RULE-03:** Calibration trong predictor chỉ là Temperature Scaling (đơn điệu) — không Isotonic.
- **RULE-06:** Mọi mảng `np.float32`; $\max|\hat y_{py}-\hat y_{onnx}|<10^{-4}$ trên Val, nếu không `PrecisionValidationError` và **không ghi bundle hoàn chỉnh**.
- **RULE-07:** `predictor.py` chỉ import `numpy`, `onnxruntime` và stdlib (`json`, `warnings`, `pathlib`, `__future__`); tuyệt đối không `quantautoml`/`sklearn`/`lightgbm`/`xgboost`/`catboost`/`optuna`/`torch`. Latency batch 1 dòng $<5.0$ ms.
- **RULE-09:** Nghiệm thu OOS dùng **Net Sharpe sau Fee + Slippage** và Hysteresis Buffer.
- **Ngưỡng nghiệm thu (verbatim):** OOS Rank IC $>0.035$, ICIR $>0.40$, Net Sharpe $>1.20$, MDD $<18\%$, Latency $<5$ ms/row, Downside IC ổn định (không âm) trong kỳ sụt giảm. `DistributionShiftWarning` khi $|z|>3.0$.
- **Dependency:** `common`/`metrics` không import tầng trên; `packaging` chỉ phụ thuộc `onnx`, `onnxruntime`, `numpy`, `common`; `backtest` chỉ phụ thuộc `numpy`, `common`, `metrics`.
- Python ≥ 3.12, type hints 100%, exception kế thừa `QuantAutoMLError`, không `except Exception: pass`.

## Review Focus (đầu vào/điều kiện dễ vỡ mà spec không nói rõ — mỗi dòng có test ở task sở hữu)

1. **NaN/±Inf ở feature lúc suy luận** — `predictor` phải xử lý giống hệt `funnel.transform` (inf→NaN, bool fill, indicator `is_nan_*`), không trả NaN ra ngoài. → Task 3, 6.
2. **Feature có std = 0 trên Train** — z-score không được chia 0 (sàn std). → Task 2.
3. **Sai thứ tự/thiếu/thừa cột đầu vào** — lỗi rõ ràng (`SchemaMismatchError`), không trả kết quả sai im lặng. → Task 2, 6.
4. **Tín hiệu hằng số / OOS ngắn hơn `rank_window` / `Close` NaN hoặc ≤ 0** — không ra NaN/inf trong Sharpe; `accepted=false`, lý do rõ ràng. → Task 10, 11, 12.
5. **Khoảng nghỉ qua đêm giữa các phiên 5 phút** — return qua gap vẫn tính vào P&L (không lọc ngầm); số nến gap được báo trong report. → Task 12.

---

## Execution Order Overview

- [ ] **Task 1: Dependencies, PackagingConfig/BacktestConfig & Exceptions**
- [ ] **Task 2: `FeatureSchemaContract` (`packaging/schema.py`)**
- [ ] **Task 3: `PreprocessorSpec` + `preprocessor.onnx` (`FeatureFunnel.preprocessor_spec()`, `packaging/preprocessor.py`)**
- [ ] **Task 4: GBDT → ONNX converters (`packaging/converters.py`)**
- [ ] **Task 5: `validate_numerical_precision` (`packaging/validator.py`, RULE-06)**
- [ ] **Task 6: Standalone `predictor_template.py` (RULE-07, latency, DistributionShiftWarning)**
- [ ] **Task 7: `ONNXBundleExporter` (`packaging/bundle.py`)**
- [ ] **Task 8: Pipeline integration — packaging**
- [ ] **Task 9: `FrictionCostModel` (`backtest/friction.py`)**
- [ ] **Task 10: `HysteresisBuffer` (`backtest/buffer.py`)**
- [ ] **Task 11: OOS Evaluator (`backtest/evaluator.py`)**
- [ ] **Task 12: `OOSBacktestRunner` (`backtest/runner.py`)**
- [ ] **Task 13: Pipeline integration — backtest, E2E & Zero-Leakage tests, docs, Phase handoff**

Quy ước mỗi task: Step 1 viết test (RED) → Step 2 chạy xác nhận FAIL → Step 3 hiện thực tối thiểu → Step 4 chạy PASS (+ `uv run pytest tests/` không hồi quy) → Step 5 commit (Conventional Commits, scope theo `GIT_CONVENTIONS`).

---

## Detailed Task Breakdown

### [ ] Task 1: Dependencies, Config & Exceptions

**Files:**
- Modify: `pyproject.toml` (thêm `onnx`, `onnxruntime`, `onnxmltools`, `skl2onnx` — chạy `uv add ...`)
- Modify: `src/quantautoml/common/config.py` — thêm `PackagingConfig`, `BacktestConfig`; `PipelineConfig.packaging: PackagingConfig | None = None`, `PipelineConfig.backtest: BacktestConfig | None = None`
- Modify: `src/quantautoml/common/exceptions.py`
- Test: `tests/unit/test_packaging_config.py`, `tests/unit/test_backtest_config.py`

**Interfaces — Produces:**
- `PackagingConfig(precision_tol: float = 1e-4 (gt 0), drift_z_threshold: float = 3.0 (gt 0), latency_budget_ms: float = 5.0 (gt 0), std_floor: float = 1e-8 (gt 0), latency_repeats: int = 200 (ge 1))`
- `BacktestConfig(price_col: str = "Close", fee_bps: float = 4.0 (ge 0), slippage_bps: float = 1.0 (ge 0), top_pct: float = 0.20 (gt 0, lt 0.5), exit_pct: float = 0.30 (gt 0, lt 0.5), rank_window: int = 2000 (ge 20), ic_window: int = 500 (ge 20), min_rank_ic: float = 0.035, min_icir: float = 0.40, min_net_sharpe: float = 1.20, max_mdd: float = 0.18, downside_quantile: float = 0.20 (gt 0, lt 0.5), min_downside_ic: float = 0.0)`; validator: `exit_pct >= top_pct` (nếu không → `ValueError`).
- `PackagingError(QuantAutoMLError)`, `PrecisionValidationError(PackagingError)`, `SchemaMismatchError(PackagingError)`, `BacktestError(QuantAutoMLError)`.

- [ ] Step 1: Test: `test_defaults_match_spec` (tol `1e-4`, drift `3.0`, latency `5.0`, fee/slip/top/exit như trên, ngưỡng 0.035/0.40/1.20/0.18); `test_exit_pct_lt_top_pct_rejected` (`pytest.raises(ValueError)`); `test_extra_field_forbidden`; `test_pipeline_config_accepts_optional_sections` (TOML có `[packaging]`/`[backtest]`); `test_exception_hierarchy` (`issubclass(PrecisionValidationError, QuantAutoMLError)`).
- [ ] Step 2: Chạy → FAIL (import error).
- [ ] Step 3: `uv add onnx onnxruntime onnxmltools skl2onnx`; hiện thực config (kế thừa `_Strict`) + exceptions.
- [ ] Step 4: `uv run pytest tests/unit/test_packaging_config.py tests/unit/test_backtest_config.py tests/unit/test_config.py -v` → PASS.
- [ ] Step 5: Commit `feat(setup): add onnx deps, packaging/backtest config and exceptions`.

---

### [ ] Task 2: `FeatureSchemaContract`

**Files:** Create `src/quantautoml/packaging/__init__.py`, `src/quantautoml/packaging/schema.py`; Test `tests/unit/test_packaging_schema.py`.

**Interfaces — Produces:**
```python
class FeatureSchemaContract(BaseModel):   # frozen, extra="forbid"
    input_features: list[str]            # tên cột thô, đúng thứ tự
    model_features: list[str]            # tên cột sau preprocessor
    dtype: Literal["float32"] = "float32"
    train_mean: list[float]              # theo model_features
    train_std: list[float]               # đã áp sàn std_floor
    task_type: str
    classes: list[int] | None
    @classmethod
    def from_train(cls, input_features, model_features, X_model_train: NDArray[np.float32],
                   task_type: TaskType, classes, std_floor: float) -> Self
    def check_input(self, X: NDArray[np.float32], names: Sequence[str] | None = None) -> None  # SchemaMismatchError
    def to_json(self) -> str ;  @classmethod from_json(cls, s: str) -> Self
```

- [ ] Step 1: Test: `test_from_train_uses_only_given_train_matrix` (mean/std khớp `np.nanmean/nanstd` của ma trận truyền vào); `test_zero_std_feature_floored` (cột hằng → `train_std == std_floor`, không 0); `test_check_input_wrong_width_raises` / `test_check_input_wrong_names_order_raises` (`SchemaMismatchError`); `test_check_input_wrong_dtype_raises` (float64); `test_json_roundtrip_equal`.
- [ ] Step 2: FAIL. Step 3: hiện thực (thống kê bằng `nanmean/nanstd` float64 rồi cast list float). Step 4: PASS. Step 5: Commit `feat(packaging): add feature schema contract`.

---

### [ ] Task 3: `PreprocessorSpec` + `preprocessor.onnx`

**Files:**
- Modify: `src/quantautoml/common/types.py` — thêm `PreprocessorSpec`
- Modify: `src/quantautoml/features/funnel.py` — thêm `FeatureFunnel.preprocessor_spec(self) -> PreprocessorSpec`; Modify `src/quantautoml/features/triage.py` chỉ nếu cần thuộc tính đọc công khai (`keep_idx_`, `indicator_idx_`, `bool_fill_`) — không đổi hành vi.
- Create: `src/quantautoml/packaging/preprocessor.py`
- Test: `tests/unit/test_preprocessor_spec.py`, `tests/unit/test_preprocessor_onnx.py`

**Interfaces — Produces:**
```python
@dataclass(frozen=True)
class PreprocessorSpec:
    n_in: int
    bool_fill: tuple[tuple[int, float], ...]   # (cột thô, giá trị fill) — áp sau inf→NaN
    indicator_idx: tuple[int, ...]             # cột thô sinh is_nan_*
    keep_idx: tuple[int, ...]                  # cột thô giữ lại từ F1
    final_idx: tuple[int, ...]                 # chỉ số vào [keep ++ indicators] sau khi gộp F2..F8
    output_names: tuple[str, ...]
def build_preprocessor_onnx(spec: PreprocessorSpec, path: Path) -> None   # input "X" float32 [N, n_in] -> output "Xp" float32 [N, len(final_idx)]
def run_preprocessor_onnx(path: Path, X: NDArray[np.float32]) -> NDArray[np.float32]  # helper chỉ dùng trong test/validator
```
Gộp F2..F8: mỗi stage chỉ là `X[:, keep_idx]`, nên `final_idx` là phép hợp thành các `keep_idx` (compose).

- [ ] Step 1: Test: `test_spec_reproduces_funnel_transform` — fit funnel trên dữ liệu tổng hợp có NaN/±inf/cột bool/cột NaN-nhiều (dùng fixture `make_frames`), so `apply_spec_numpy(spec, X)` (hàm tham chiếu trong test) với `funnel.transform(seg).X` bằng `np.array_equal`; `test_onnx_matches_funnel_transform_with_inf_nan` (`max|Δ| == 0` trên cả hàng chứa ±inf/NaN); `test_empty_selection_gives_n_by_0` (funnel loại hết cột).
- [ ] Step 2: FAIL. Step 3: hiện thực spec (compose) + dựng graph bằng `onnx.helper` (Where/IsInf/IsNaN/Gather/Concat/Cast; float32 xuyên suốt) và `onnx.checker.check_model`. Step 4: PASS. Step 5: Commit `feat(packaging): export funnel preprocessing as ONNX`.

---

### [ ] Task 4: GBDT → ONNX converters

**Files:** Create `src/quantautoml/packaging/converters.py`; Test `tests/unit/test_packaging_converters.py`.

**Interfaces — Produces:**
```python
def export_candidate_onnx(model: BaseGBDCandidate, n_features: int, path: Path) -> None
# ONNX: input "X" float32 [N, n_features] -> output "probabilities" float32 [N, n_classes] (không ZipMap)
```
Dispatch theo `model.name` ∈ {`lightgbm`, `xgboost`, `catboost`}: LightGBM/XGBoost qua `onnxmltools.convert_*` (`zipmap` tắt / `target_opset` cố định, ghi trong code), CatBoost qua `estimator_.save_model(path, format="onnx")` + chuẩn hóa tên I/O. Tên không hỗ trợ → `PackagingError`.

- [ ] Step 1: Test (tham số hóa 3 model, binary + multiclass 3 lớp, 300×8 dữ liệu tổng hợp): `test_onnx_probs_close_to_python_predict_proba` (`max|Δ| < 1e-4`); `test_output_shape_and_dtype` (`(N, C)`, float32); `test_unknown_model_name_raises`. **Nếu một cặp (model, task) không đạt < 1e-4, DỪNG và báo user** (rủi ro tương thích numpy 2.5/onnxmltools — xem Rủi ro).
- [ ] Step 2: FAIL. Step 3: hiện thực. Step 4: PASS. Step 5: Commit `feat(packaging): convert GBDT trio to ONNX`.

---

### [ ] Task 5: `validate_numerical_precision` (RULE-06)

**Files:** Create `src/quantautoml/packaging/validator.py`; Test `tests/unit/test_packaging_validator.py`.

**Interfaces — Produces:**
```python
def validate_numerical_precision(y_py: NDArray[np.float32], y_onnx: NDArray[np.float32], *, tol: float = 1e-4) -> float
# trả max|Δ|; ném PrecisionValidationError nếu max|Δ| >= tol, shape khác, dtype != float32 hoặc có NaN/inf ở một bên
```

- [ ] Step 1: Test: `test_pass_returns_max_abs_diff` (`Δ=5e-5` → trả `≈5e-5`); `test_at_tolerance_fails` (`Δ == 1e-4` → raise, vì điều kiện là `<`); `test_shape_mismatch_raises`; `test_float64_input_raises`; `test_nan_on_one_side_raises` (NaN so với số hữu hạn không được coi là "khớp").
- [ ] Step 2–5: FAIL → hiện thực → PASS → Commit `feat(packaging): add float32 precision validator`.

---

### [ ] Task 6: Standalone `predictor_template.py`

**Files:** Create `src/quantautoml/packaging/templates/predictor_template.py`; Test `tests/unit/test_predictor_template.py`.

**Interfaces — Produces** (trong template, không import `quantautoml`):
```python
class DistributionShiftWarning(UserWarning): ...
class Predictor:
    def __init__(self, bundle_dir: str | Path) -> None   # nạp features_schema.json, ensemble_meta.json, các .onnx
    def predict(self, X: np.ndarray) -> np.ndarray       # [N, n_in] float32 -> [N] float32 (binary: P(y=1) đã calibrate; multiclass: kỳ vọng lớp Σ c·p đã calibrate)
    def predict_proba(self, X: np.ndarray) -> np.ndarray # [N, C] đã calibrate (binary: [N,2])
```
Luồng: kiểm tra schema (`ValueError` rõ ràng) → `preprocessor.onnx` → mỗi `model_k.onnx` → trộn `Σ w_k·p_k` → Temperature (binary: `sigmoid(logit(clip(p,1e-7,1-1e-7))/T)`; multiclass: `softmax(log(clip(P,1e-7,1))/T)`, khớp `TemperatureCalibrator.calibrate`) → `warnings.warn(DistributionShiftWarning)` nếu có `|z| > drift_z_threshold` (z theo `train_mean/train_std` của `features_schema.json`, tính trên đầu ra preprocessor). Thân file ~50–80 dòng.

- [ ] Step 1: Test dùng bundle tối thiểu dựng bằng helper trong test (preprocessor + 2 model ONNX nhỏ + meta): `test_imports_only_numpy_onnxruntime_stdlib` (parse `ast`, tập import ⊆ `{numpy, onnxruntime, json, warnings, pathlib, __future__}`); `test_runs_in_isolated_subprocess_without_quantautoml` (`python -I`, `sys.modules` không chứa `quantautoml`/`sklearn`/`lightgbm`/`xgboost`/`catboost`/`optuna`/`torch`); `test_matches_python_reference_pipeline` (sai số < 1e-4 so với funnel→models→weights→`TemperatureCalibrator.calibrate`); `test_nan_inf_input_never_returns_nan` (Review Focus 1); `test_drift_warning_when_z_gt_3` (`pytest.warns(DistributionShiftWarning)`) và `test_no_warning_in_distribution`; `test_wrong_column_count_raises_valueerror` (Review Focus 3); `test_single_row_latency_under_5ms` (median qua 200 lần gọi 1 dòng, < 5.0 ms).
- [ ] Step 2: FAIL. Step 3: hiện thực template; `bundle.py` (Task 7) sẽ copy nguyên văn thành `predictor.py`. Step 4: PASS. Step 5: Commit `feat(packaging): add decoupled numpy/onnxruntime predictor template`.

---

### [ ] Task 7: `ONNXBundleExporter`

**Files:** Create `src/quantautoml/packaging/bundle.py`; Modify `src/quantautoml/packaging/__init__.py` (export `ONNXBundleExporter`, `BundleManifest`, `FeatureSchemaContract`, `validate_numerical_precision`); Test `tests/unit/test_packaging_bundle.py`.

**Interfaces — Consumes:** Task 2–6. **Produces:**
```python
class BundleManifest(BaseModel):   # ghi vào bundle_manifest.json
    files: list[str]; max_abs_diff: float; n_models: int; task_type: str
class ONNXBundleExporter:
    def __init__(self, cfg: PackagingConfig) -> None
    def export(self, *, out_dir: Path, preprocessor_spec: PreprocessorSpec,
               candidates: Sequence[TunedCandidate], ensemble: EnsembleWeightsResult,
               calibration: CalibrationResult, task_type: TaskType,
               input_features: Sequence[str], classes: Sequence[int] | None,
               X_train_raw: NDArray[np.float32], X_val_raw: NDArray[np.float32]) -> BundleManifest
```
Quy trình: build preprocessor → thống kê z-score từ `X_train_raw` qua preprocessor → export từng model → `ensemble_meta.json` (weights, temperature, classes, task_type, model files) → copy `predictor_template.py` → `predictor.py` → chạy `Predictor` trên `X_val_raw` và so với đường tham chiếu Python (`validate_numerical_precision`) → ghi `bundle_manifest.json`. **Nếu validator ném, dọn thư mục `out_dir` (không để bundle dở dang) rồi re-raise.**

- [ ] Step 1: Test: `test_export_writes_required_files` (`preprocessor.onnx`, `model_0.onnx…`, `ensemble_meta.json`, `features_schema.json`, `predictor.py`, `bundle_manifest.json`); `test_manifest_max_abs_diff_below_tol`; `test_precision_failure_raises_and_leaves_no_bundle` (monkeypatch làm 1 model lệch > 1e-4 → `PrecisionValidationError`, `out_dir` không còn `predictor.py`); `test_exporter_never_calls_fit` (mock `fit` ném lỗi → không bị gọi); `test_schema_stats_use_train_only` (đổi `X_val_raw` không làm đổi `train_mean`); `test_fallback_weights_exported` (ensemble bị từ chối → `weights == [1/K,…]` vẫn được ghi đúng).
- [ ] Step 2–5: FAIL → hiện thực → PASS → Commit `feat(packaging): implement ONNX model bundle exporter`.

---

### [ ] Task 8: Pipeline integration — packaging

**Files:** Modify `src/quantautoml/pipeline.py` (thêm `bundle_manifest: BundleManifest | None = None` vào `PipelineResult`); Test `tests/unit/test_pipeline_packaging.py`.

Khi `cfg.packaging` được đặt **và** `ensemble_result`, `calibration_result`, `funnel`... đã có: gọi `ONNXBundleExporter.export(out_dir=out / "bundle", ..., X_val_raw=split.val.X)` (Val chỉ để **suy luận** đối chiếu, không fit). Khi thiếu thành phần tiên quyết → bỏ qua (giống các stage trước), không báo lỗi giả. Với OOF mode (không có `cal_a/cal_b` riêng) vẫn dùng `split.val`.

- [ ] Step 1: Test: `test_pipeline_writes_bundle_when_packaging_configured`; `test_pipeline_skips_bundle_without_packaging_section`; `test_pipeline_bundle_precision_ok` (đọc `bundle_manifest.json`, `max_abs_diff < 1e-4`); `test_pipeline_does_not_touch_oos_for_packaging` (spy: `split.oos.X` không được truy cập trong stage này).
- [ ] Step 2–5: FAIL → tích hợp → PASS (+ toàn bộ `tests/`) → Commit `feat(pipeline): wire ONNX bundle packaging into pipeline`.

---

### [ ] Task 9: `FrictionCostModel`

**Files:** Create `src/quantautoml/backtest/__init__.py`, `src/quantautoml/backtest/friction.py`; Test `tests/unit/test_friction.py`.

**Interfaces — Produces:**
```python
class FrictionCostModel:
    def __init__(self, fee_bps: float, slippage_bps: float) -> None   # ValueError nếu < 0
    def turnover(self, positions: NDArray[np.float64]) -> NDArray[np.float64]   # [T]; bar 0 so với vị thế 0; = 0.5*|Δw| (D3)
    def net_returns(self, gross: NDArray[np.float64], positions: NDArray[np.float64]) -> NDArray[np.float64]
    # R_net = gross - turnover * (fee_bps + slippage_bps) / 1e4
```

- [ ] Step 1: Test: `test_zero_cost_equals_gross`; `test_known_turnover_and_net` (positions `[0,1,1,-1,0]`, fee 4 + slip 1 → turnover `[0.5,0,1,0.5,…]`, net đúng tới `1e-12`); `test_first_bar_entry_pays_cost`; `test_net_never_exceeds_gross_when_costs_positive` (property, vị thế ngẫu nhiên); `test_length_mismatch_raises`.
- [ ] Step 2–5: FAIL → hiện thực → PASS → Commit `feat(backtest): add friction cost model`.

---

### [ ] Task 10: `HysteresisBuffer`

**Files:** Create `src/quantautoml/backtest/buffer.py`; Test `tests/unit/test_buffer.py`.

**Interfaces — Produces:**
```python
class HysteresisBuffer:
    def __init__(self, top_pct: float, exit_pct: float, rank_window: int) -> None
    def rolling_percentile(self, scores: NDArray[np.float32]) -> NDArray[np.float64]  # [T] ∈ [0,1]; NaN cho t < rank_window-1; chỉ dùng scores[t-rank_window+1 : t+1]
    def positions(self, scores: NDArray[np.float32]) -> NDArray[np.float64]           # ∈ {-1,0,+1}; 0 khi percentile chưa xác định
```
Luật (D2): vào Long khi pct ≥ `1-top_pct`, giữ tới khi pct < `1-exit_pct`; vào Short khi pct ≤ `top_pct`, giữ tới khi pct > `exit_pct`; không bao giờ đảo trực tiếp mà không đi qua vùng thoát theo luật trên.

- [ ] Step 1: Test: `test_percentile_is_causal` (đổi `scores[t+1:]` không làm đổi `percentile[:t+1]`); `test_nan_before_window_full`; `test_zero_position_before_window_full`; `test_hysteresis_holds_between_thresholds` (pct đi 0.85 → 0.75 → 0.65: vẫn Long ở 0.75, thoát ở 0.65); `test_buffer_reduces_trades_vs_no_buffer` (tín hiệu nhiễu quanh ngưỡng: số lần đổi vị thế khi `exit_pct=0.30` < khi `exit_pct==top_pct`, giảm ≥ 20%); `test_exit_equals_top_is_memoryless` ; `test_constant_scores_never_trade` (Review Focus 4: không NaN, vị thế toàn 0).
- [ ] Step 2: FAIL. Step 3: percentile cuốn chiếu bằng `numpy.lib.stride_tricks.sliding_window_view` (rank của phần tử cuối trong cửa sổ, ties → mid-rank); vòng lặp trạng thái. Step 4: PASS. Step 5: Commit `feat(backtest): add hysteresis buffer bands`.

---

### [ ] Task 11: OOS Evaluator

**Files:** Create `src/quantautoml/backtest/evaluator.py`; Test `tests/unit/test_evaluator.py`.

**Interfaces — Consumes:** `calc_rank_ic`, `calc_rolling_ic`, `calc_icir` (`metrics/ic.py`), `FrictionCostModel`, `BacktestConfig`. **Produces:**
```python
class OOSMetrics(BaseModel):
    rank_ic: float; icir: float; gross_sharpe: float; net_sharpe: float
    max_drawdown: float; downside_ic: float; mean_turnover: float
    n_bars: int; latency_ms: float | None
    gates: dict[str, bool]      # keys: rank_ic, icir, net_sharpe, mdd, latency, downside_ic
    accepted: bool; failed_gates: list[str]
def evaluate_oos_performance(*, scores: NDArray[np.float32], y: NDArray[np.int32],
        gross_returns: NDArray[np.float64], net_returns: NDArray[np.float64],
        turnover: NDArray[np.float64], periods_per_year: float,
        latency_ms: float | None, cfg: BacktestConfig) -> OOSMetrics
```
Định nghĩa: Sharpe = `mean/std(ddof=1)·sqrt(periods_per_year)` (std=0 hoặc <2 điểm → 0.0); MDD trên đường vốn cộng dồn `cumprod(1+r)` (dương, ví dụ 0.12); Rank IC = `calc_rank_ic(scores, y)`; ICIR = `calc_icir(calc_rolling_ic(scores, y, cfg.ic_window))`; **Downside IC** = Rank IC trên các nến có `gross_market_return` rơi vào `downside_quantile` thấp nhất của cửa sổ OOS — nếu không đủ cặp → `0.0` và gate `downside_ic` đánh trượt; gate `latency` = `latency_ms is not None and < 5.0`.

- [ ] Step 1: Test: `test_sharpe_matches_manual` (chuỗi cố định); `test_mdd_known_curve` (`[+10%, -20%]` → `0.20`); `test_flat_returns_sharpe_zero_not_nan` (Review Focus 4); `test_gates_use_strict_inequality` (đúng bằng ngưỡng `0.035` → trượt); `test_accepted_only_if_all_gates_pass`; `test_failed_gates_listed`; `test_missing_latency_fails_latency_gate`; `test_net_sharpe_used_for_gate_not_gross` (gross > 1.2 nhưng net < 1.2 → `accepted=False`, RULE-09).
- [ ] Step 2–5: FAIL → hiện thực → PASS → Commit `feat(backtest): add OOS acceptance evaluator`.

---

### [ ] Task 12: `OOSBacktestRunner`

**Files:** Create `src/quantautoml/backtest/runner.py`; Modify `src/quantautoml/backtest/__init__.py` (export `FrictionCostModel`, `HysteresisBuffer`, `evaluate_oos_performance`, `OOSBacktestRunner`); Test `tests/unit/test_runner.py`.

**Interfaces — Consumes:** Task 9–11; `predictor.py` trong bundle (nạp bằng `importlib` từ đường dẫn — **không** import qua `quantautoml`). **Produces:**
```python
class OOSBacktestRunner:
    def __init__(self, cfg: BacktestConfig) -> None
    def run(self, *, bundle_dir: Path, oos: Segment, feature_names: Sequence[str],
            other_segments: Sequence[Segment], out_path: Path | None = None) -> OOSMetrics
```
Luồng: kiểm tra `oos.index` không giao `other_segments[*].index` (`DataLeakageError`) → lấy `Close` từ `oos.X[:, feature_names.index(cfg.price_col)]` (thiếu cột / NaN / ≤ 0 → `BacktestError`) → `scores = Predictor.predict(oos.X)` → đo latency batch 1 dòng (median) → `HysteresisBuffer.positions` → `gross_t = pos_t·(Close_{t+1}/Close_t − 1)` (bar cuối bỏ) → `FrictionCostModel.net_returns` → `periods_per_year = n_bars / năm(timestamps)` → `evaluate_oos_performance` → ghi `backtest_report.json` (kèm `n_overnight_gaps`: số khoảng cách timestamp > 3× median — Review Focus 5). `n_bars < rank_window + 2` → `BacktestError`.

- [ ] Step 1: Test (bundle tổng hợp từ helper Task 6, tín hiệu nhân tạo có alpha rõ): `test_runner_end_to_end_accepts_strong_signal`; `test_runner_rejects_pure_noise_signal` (`accepted=False`, `failed_gates` không rỗng); `test_oos_overlap_raises_dataleakage` ; `test_missing_price_col_raises`; `test_nonpositive_close_raises`; `test_oos_shorter_than_window_raises`; `test_report_written_and_has_gates`; `test_gap_count_reported`; `test_runner_does_not_modify_bundle_files` (hash trước/sau bằng nhau — Blind: không fit/không ghi).
- [ ] Step 2–5: FAIL → hiện thực → PASS → Commit `feat(backtest): add blind OOS backtest runner`.

---

### [ ] Task 13: Pipeline integration (backtest), E2E & Zero-Leakage tests, docs, Phase handoff

**Files:**
- Modify: `src/quantautoml/pipeline.py` (`PipelineResult.backtest_metrics: OOSMetrics | None = None`; chạy `OOSBacktestRunner` khi `cfg.backtest` và `bundle_manifest` tồn tại, ghi `backtest_report.json`)
- Create: `tests/integration/test_end_to_end.py`, `tests/integration/test_friction_backtest.py`, `tests/integration/test_zero_leakage_packaging.py`
- Modify: `docs/plan/STRUCTURE.md`, `docs/STRUCTURE.md` (thêm `converters.py`, `preprocessor.py`), `docs/CONFIGURATION.md` (`[packaging]`, `[backtest]`), `docs/plan/PHASES.md`, `docs/plan/PLAN.md`

- [ ] Step 1: Test: `test_end_to_end_raw_data_to_bundle_and_report` (dữ liệu tổng hợp → `run_pipeline` → có `bundle/` + `backtest_report.json`); `test_friction_lowers_sharpe` (cùng tín hiệu: `net_sharpe < gross_sharpe` khi phí > 0; bằng nhau khi phí = 0); `test_buffer_lowers_turnover_in_pipeline`; `test_zero_leakage_packaging` (mock `fit` ở mọi filter/model/calibrator/optimizer trong giai đoạn packaging + backtest → không bị gọi; `split.oos` chỉ được đọc trong runner); `test_pipeline_without_backtest_section_still_ok`.
- [ ] Step 2–4: FAIL → tích hợp → `uv run pytest tests/` PASS 100%.
- [ ] Step 5: Chạy thử thật trên 1 mẫu con của `datasets/` (không commit dữ liệu/bundle): xác nhận bundle sinh ra, `max_abs_diff < 1e-4`, latency < 5 ms, đọc `backtest_report.json`, **báo cáo trung thực kết quả kể cả khi `accepted=false`**.
- [ ] Step 6: Cập nhật docs; Phase Handoff theo CLAUDE.md §2.2: Phase 5 → `Status: Done` trong `PHASES.md`; vì đây là phase cuối của v1.0, **hỏi user trước khi** archive vào `docs/plan/archive/v1.0/` và chuyển sang `develop/v1.1`.
- [ ] Step 7: Commit `feat(pipeline): wire friction-aware blind OOS backtest (Phase 5 Done)`.

---

## Rủi ro đã biết

- **Tương thích converter:** `numpy 2.5.4` + `onnxmltools`/`skl2onnx` có thể chưa hỗ trợ; Task 4 là "cổng" sớm — nếu lỗi, dừng và hỏi user (phương án: pin phiên bản, hoặc tự dựng graph cây bằng `TreeEnsembleClassifier` op).
- **Sai khác ngưỡng cây float32 vs float64** có thể làm vài mẫu sát ngưỡng lệch > 1e-4 → validator sẽ chặn; đây là hành vi đúng của RULE-06, không nới `tol`.
- **Net Sharpe > 1.20 / MDD < 18%** là tiêu chí kết quả thực nghiệm, không đảm bảo được bằng code; plan chỉ đảm bảo *đo đúng* và *báo cáo trung thực*.

## Final Verification Checklist

- [ ] `uv run pytest tests/` đạt 100% PASS (Phase 1–5).
- [ ] `max|ŷ_py − ŷ_onnx| < 1e-4` trên Val, `PrecisionValidationError` chặn bundle khi vi phạm.
- [ ] `predictor.py` chạy độc lập (`python -I`), import chỉ `numpy`/`onnxruntime`/stdlib; latency 1 dòng < 5 ms.
- [ ] `DistributionShiftWarning` kích hoạt khi $|z|>3.0$.
- [ ] Backtest báo cáo Net Sharpe sau phí + slippage, có Hysteresis Buffer, Downside IC.
- [ ] Zero Leakage: không `fit()` ở packaging/backtest; OOS chỉ được đọc bởi runner.
- [ ] `PHASES.md` Phase 5 → Done; xác nhận với user việc archive v1.0.
