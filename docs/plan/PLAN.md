# Implementation Plan: Quant AutoML Pipeline — Phase 3

> **For agentic workers:** REQUIRED SUB-SKILL: dùng `superpowers:subagent-driven-development` (khuyến nghị) hoặc `superpowers:executing-plans` để thực thi plan theo từng task. Quy trình TDD bắt buộc (RED → GREEN → REFACTOR), cập nhật `- [x]` sau mỗi task theo RULE-10.

**Active Phase:** Phase 3 — Modern GBDT Tournament & Deep Hyperparameter Optimization  
**Branch:** `develop/v1.0-phase3`  
**Goal:** Xây dựng hệ thống đấu loại công bằng giữa 3 mô hình GBDT hiện đại (LightGBM, CatBoost, XGBoost), tự động quyết định Single Best Model vs. Ensemble Pool (Diversity < 0.88, Marginal Gain ≥ 1.5%, WF Consistency ≥ 60%), và tinh chỉnh siêu tham số chuyên sâu bằng Optuna Multivariate TPE + HyperbandPruner tối ưu theo hàm mục tiêu $\text{IC Sharpe}_{WF} - 0.05 \times \text{Complexity\_Score}$.  
**Architecture:** `models/` là module độc lập tiếp nhận đầu ra của `FeatureFunnel` trên `split.train`. Các mô hình GBDT được đóng gói qua giao diện chuẩn `GBDTTrioPool` trong `candidates.py`. `tournament.py` tổ chức 30 trials `RandomSampler` trên 5-fold Expanding Window Walk-Forward. `decision_engine.py` thẩm định 3 điều kiện chọn Ensemble ($K \le 3$) hay Single Model. `tuner.py` chạy Bayesian Optimization chuyên sâu. Các metric định lượng (IC Sharpe, Walk-Forward IC) nằm tại `metrics/ic.py`.  
**Tech Stack:** `lightgbm`, `xgboost`, `catboost`, `optuna`, `scipy`, `numpy`, `pydantic` v2, `pytest`.  
**Spec:** [`PHASES.md`](PHASES.md) (Phase 3), [`ARCHITECTURE.md`](ARCHITECTURE.md) §3, [`TRADE-OFF.md`](TRADE-OFF.md) §3, [`STRUCTURE.md`](STRUCTURE.md), [`03-model-selection/`](03-model-selection/), [`04-tuning/`](04-tuning/).

---

## Global Constraints

- **RULE-01:** Mọi thao tác huấn luyện mô hình, cross-validation Walk-Forward chỉ diễn ra trên `split.train`. Tuyệt đối cấm lọt mẫu từ Cal-A, Cal-B, Val, OOS. Vi phạm $\to$ `DataLeakageError`.
- **RULE-02:** Xuyên suốt Tournament và Deep Tuning, chỉ số đánh giá tối ưu bắt buộc là **Spearman Rank IC & IC Sharpe** (không dùng raw Accuracy hay MSE đơn thuần).
- **RULE-05:** Decision Engine chỉ chấp thuận cấu hình Ensemble khi vượt qua cả 3 điều kiện:
  1. Diversity correlation giữa các mô hình $< 0.88$ (độ đa dạng $> 0.12$).
  2. Marginal Gain của Equal-Weight ensemble $\ge +1.5\%$ Rank IC so với Single Best Model.
  3. Walk-Forward Consistency $\ge 60\%$ (tối thiểu 3/5 fold ensemble vượt trội hơn single model).
  Nếu không đạt, tự động fallback về Single Best Model.
- **RULE-06:** Dữ liệu ma trận số học đầu vào và đầu ra xác suất/dự đoán phải dùng chuẩn `np.float32`.
- **Python $\ge 3.12$**, type hints 100%, không dùng `Any` mơ hồ, ngoại lệ kế thừa `QuantAutoMLError`, không dùng `try ... except Exception: pass`.
- Phụ thuộc một chiều: `common` và `metrics` không import `models`; `models` chỉ import `common`, `metrics`, `numpy`, `scipy`, `lightgbm`, `xgboost`, `catboost`, `optuna`.

---

## Execution Order Overview

- [x] **Task 1: Dependencies, Exceptions & ModelsConfig** — Bổ sung thư viện GBDT/Optuna vào `pyproject.toml`, định nghĩa custom exceptions và Pydantic `ModelsConfig`.
- [x] **Task 2: Quantitative Metrics — IC Sharpe & Walk-Forward Validation Helper** — Hiện thực `calc_ic_sharpe()` và expanding window walk-forward fold splitter kèm purging/embargo.
- [x] **Task 3: Modern GBDT Trio Pool (Candidates Wrappers)** — Gói `LightGBM`, `CatBoost`, `XGBoost` theo giao diện chuẩn `BaseGBDCandidate`, hỗ trợ Binary/Multiclass/Continuous và hàm tính `complexity_score()`.
- [x] **Task 4: ModelTournament (30-Trial Walk-Forward Tournament)** — Cơ chế đấu loại 30 trials `RandomSampler` trên 5-fold Walk-Forward, xếp hạng bằng Out-of-Fold IC Sharpe.
- [x] **Task 5: ModelDecisionEngine (Single Model vs. Ensemble Gate)** — Thẩm định 3 điều kiện (Diversity < 0.88, Marginal Gain ≥ 1.5%, WF Consistency ≥ 60%) và chọn lọc Ensemble Pool ($K \le 3$) hoặc fallback về Single Best.
- [x] **Task 6: DeepOptunaTuner (Multivariate TPE + HyperbandPruner)** — Tối ưu hóa siêu tham số chuyên sâu với hàm mục tiêu $\text{IC Sharpe}_{WF} - 0.05 \times \text{Complexity\_Score}$ và cơ chế tỉa sớm Hyperband.
- [x] **Task 7: Pipeline Integration, Reporting & Zero Leakage Verification** — Nối `models` vào `PipelineResult` trong `pipeline.py`, xuất báo cáo `tournament_report.json`, `decision_report.json`, `tuning_report.json`, kiểm tra toàn diện Zero Leakage.

---

## Detailed Task Breakdown

### [x] Task 1: Dependencies, Exceptions & ModelsConfig

**Files:**
- Modify: `pyproject.toml`, `src/quantautoml/common/exceptions.py`, `src/quantautoml/common/config.py`, `docs/CONFIGURATION.md`
- Test: `tests/unit/test_models_config.py`

**Interfaces:**
- Produces:
  - Exceptions: `class ModelError(QuantAutoMLError)`, `class TournamentError(ModelError)`, `class TuningError(ModelError)`.
  - Pydantic Schema: `class ModelsConfig(_Strict)`:
    - `tournament_trials: int = 30` (ngân sách mỗi model)
    - `tuning_trials: int = 50` (ngân sách Optuna tuning)
    - `n_wf_folds: int = 5` (số fold Expanding Walk-Forward)
    - `diversity_threshold: float = 0.88` (ngưỡng tương quan tối đa)
    - `marginal_gain_threshold: float = 0.015` (ngưỡng tăng trưởng tối thiểu 1.5% Rank IC)
    - `wf_consistency_threshold: float = 0.60` (ngưỡng nhất quán Walk-Forward 60%)
    - `complexity_penalty: float = 0.05`
    - `random_state: int = 42`
    - `enabled_models: list[str] = ["lightgbm", "xgboost", "catboost"]`
  - Modify `PipelineConfig`: thêm `models: ModelsConfig | None = None`.

**Action Steps:**
1. Viết test thất bại `tests/unit/test_models_config.py`:
   - Parse TOML có section `[models]` hợp lệ.
   - Bắt lỗi khi `tournament_trials < 1`, `diversity_threshold <= 0` hoặc `> 1.0`.
   - Bắt lỗi khi `enabled_models` chứa model không hợp lệ ngoài trio.
2. Thêm vào `pyproject.toml`:
   - `lightgbm>=4.3.0`
   - `xgboost>=2.0.0`
   - `catboost>=1.2.0`
   - `optuna>=3.6.0`
   Chạy `uv sync` để đồng bộ môi trường ảo.
3. Hiện thực exceptions trong `exceptions.py` và `ModelsConfig` trong `config.py`.
4. Cập nhật `docs/CONFIGURATION.md` phần `[models]`.

- **Verification Command:** `uv run pytest tests/unit/test_models_config.py -v`
- **Commit Message:** `chore(models): add gbdt dependencies and models config`

---

### [x] Task 2: Quantitative Metrics — IC Sharpe & Walk-Forward Validation Helper

**Files:**
- Modify: `src/quantautoml/metrics/ic.py`, `src/quantautoml/metrics/__init__.py`
- Create: `src/quantautoml/models/validation.py`
- Test: `tests/unit/test_metrics_ic_sharpe.py`, `tests/unit/test_walk_forward.py`

**Interfaces:**
- Produces:
  - `def calc_ic_sharpe(ic_series: NDArray[np.float64], *, annualization_factor: float = 1.0) -> float`
    - Tính $\frac{\text{mean}(\text{IC})}{\text{std}(\text{IC}, ddof=1)} \times \sqrt{\text{annualization\_factor}}$.
    - Trả $0.0$ nếu $\text{std} == 0$, mảng có ít hơn 2 phần tử hợp lệ, hoặc toàn NaN.
  - `class ExpandingWalkForwardSplitter`:
    - `__init__(n_folds: int = 5, min_train_ratio: float = 0.5, embargo_bars: int = 0)`
    - `split(n_samples: int) -> Iterator[tuple[NDArray[np.int32], NDArray[np.int32]]]`
    - Tạo các fold expanding window theo thời gian: Train mở rộng dần, Test fold đi tiếp theo sau embargo, đảm bảo không có khoảng cách âm và không xáo trộn thứ tự thời gian.

**Action Steps:**
1. Viết test thất bại `tests/unit/test_metrics_ic_sharpe.py`:
   - Kiểm tra công thức chuẩn, kiểm tra mảng hằng số trả về 0.0, kiểm tra annualization.
2. Viết test thất bại `tests/unit/test_walk_forward.py`:
   - Kiểm tra 5 folds mở rộng liên tục: $\text{Train}_k \subset \text{Train}_{k+1}$.
   - Kiểm tra `max(Train_k) + embargo < min(Test_k)`.
   - Kiểm tra không chồng lấn giữa các tập test.
3. Hiện thực mã nguồn trong `metrics/ic.py` và `models/validation.py`.

- **Verification Command:** `uv run pytest tests/unit/test_metrics_ic_sharpe.py tests/unit/test_walk_forward.py -v`
- **Commit Message:** `feat(models): implement ic sharpe metric and expanding walk-forward splitter`

---

### [x] Task 3: Modern GBDT Trio Pool (Candidates Wrappers)

**Files:**
- Create: `src/quantautoml/models/candidates.py`
- Test: `tests/unit/test_candidates.py`

**Interfaces:**
- Produces:
  - `class BaseGBDCandidate(ABC)`:
    - Giao diện thống nhất: `fit(X: NDArray[np.float32], y: NDArray[np.float32 | np.int32]) -> Self`, `predict(X: NDArray[np.float32]) -> NDArray[np.float32]`, `predict_proba(X: NDArray[np.float32]) -> NDArray[np.float32]`, `complexity_score() -> float`.
  - `class LightGBMCandidate(BaseGBDCandidate)`
  - `class CatBoostCandidate(BaseGBDCandidate)`
  - `class XGBoostCandidate(BaseGBDCandidate)`
  - `def create_candidate(name: str, task_type: TaskType, params: dict[str, Any] | None = None, random_state: int = 42) -> BaseGBDCandidate`
  - **Ghi chú phạm vi:** `TaskType` hiện chỉ có Binary/Multiclass/Unsupervised nên wrapper là classifier; hồi quy (Continuous) chưa hỗ trợ và nằm ngoài phạm vi v1.0.
  - `complexity_score()`:
    - Chuẩn hóa thang điểm $[0, 1]$ dựa trên `n_estimators`, `max_depth` (hoặc `num_leaves`), `learning_rate` nhằm phạt mô hình quá phức tạp dễ overfit.

**Action Steps:**
1. Viết test thất bại `tests/unit/test_candidates.py`:
   - Cả 3 candidates khởi tạo đúng theo `TaskType.BINARY` và `TaskType.MULTICLASS`.
   - Đầu vào nhận `float32`, trả dự đoán `float32`.
   - Kiểm tra tính tái lập với cùng `random_state`.
   - Tính toán `complexity_score()` trả về giá trị dương xác định.
2. Hiện thực 3 wrappers trong `candidates.py`, cấu hình tham số mặc định tối ưu hiệu năng và tắt log console thừa (`verbose=-1`, `silent=True`).

- **Verification Command:** `uv run pytest tests/unit/test_candidates.py -v`
- **Commit Message:** `feat(models): implement modern gbdt trio wrappers with unified interface`

---

### [x] Task 4: ModelTournament (30-Trial Walk-Forward Tournament)

**Files:**
- Create: `src/quantautoml/models/tournament.py`
- Test: `tests/unit/test_tournament.py`

**Interfaces:**
- Produces:
  - `class TrialScore(BaseModel)`: `trial_id: int, model_name: str, params: dict, fold_ics: list[float], mean_ic: float, ic_sharpe: float`
  - `class TournamentResult(BaseModel)`: `rankings: list[TrialScore], best_per_model: dict[str, TrialScore], oof_predictions: dict[str, list[float]]`
  - `class ModelTournament`:
    - `__init__(cfg: ModelsConfig, task_type: TaskType)`
    - `run(X: NDArray[np.float32], y: NDArray[np.float32 | np.int32], embargo_bars: int = 0) -> TournamentResult`
    - Cấp 30 trials `optuna.samplers.RandomSampler` cho mỗi candidate trong trio trên 5-fold Walk-Forward.
    - Xếp hạng theo out-of-fold `ic_sharpe`.

**Action Steps:**
1. Viết test thất bại `tests/unit/test_tournament.py`:
   - Chạy tournament giả lập (budget nhỏ trong test: 2 trials/model) trên dữ liệu mẫu.
   - Kiểm tra tổng số trial khớp với cấu hình, ranking được sắp xếp giảm dần theo `ic_sharpe`.
   - Kiểm tra OOF predictions được ghi nhận đầy đủ cho các fold test.
2. Hiện thực `tournament.py` sử dụng `ExpandingWalkForwardSplitter` từ Task 2 và `create_candidate` từ Task 3.

- **Verification Command:** `uv run pytest tests/unit/test_tournament.py -v`
- **Commit Message:** `feat(models): implement walk-forward model tournament with random sampling`

---

### [x] Task 5: ModelDecisionEngine (Single Model vs. Ensemble Gate)

**Files:**
- Create: `src/quantautoml/models/decision_engine.py`
- Test: `tests/unit/test_decision_engine.py`

**Interfaces:**
- Produces:
  - `class DecisionType(StrEnum)`: `SINGLE_MODEL = "single_model"`, `ENSEMBLE = "ensemble"`
  - `class DecisionReport(BaseModel)`:
    - `decision: DecisionType`
    - `selected_models: list[str]` (tối đa $K \le 3$)
    - `pairwise_correlations: dict[str, float]`
    - `diversity_score: float` ($1 - \max(\rho)$)
    - `marginal_gain: float` (Rank IC chênh lệch giữa Ensemble vs Best Single)
    - `wf_consistency: float` (tỷ lệ fold ensemble vượt single best)
    - `rationale: str`
  - `class ModelDecisionEngine`:
    - `__init__(cfg: ModelsConfig)`
    - `evaluate(tournament_result: TournamentResult, y_true: NDArray[np.float32 | np.int32]) -> DecisionReport`
    - Kiểm tra 3 điều kiện:
      1. Tương quan cặp dự đoán $< 0.88$ (Diversity $> 0.12$).
      2. Marginal Gain của Equal Weights ensemble $\ge 1.5\%$ Rank IC.
      3. WF Consistency $\ge 60\%$.
    - Nếu vi phạm bất kỳ điều kiện nào $\to$ fallback về `SINGLE_MODEL` với mô hình có IC Sharpe cao nhất.

**Action Steps:**
1. Viết test thất bại `tests/unit/test_decision_engine.py`:
   - Test case 1: 3 mô hình tương quan cao ($> 0.88$) $\to$ fallback về `SINGLE_MODEL`.
   - Test case 2: Đa dạng tốt nhưng marginal gain $< 1.5\%$ $\to$ fallback về `SINGLE_MODEL`.
   - Test case 3: Cả 3 điều kiện thỏa mãn $\to$ chọn `ENSEMBLE` với $K \le 3$.
2. Hiện thực logic kiểm tra nghiêm ngặt trong `decision_engine.py`.

- **Verification Command:** `uv run pytest tests/unit/test_decision_engine.py -v`
- **Commit Message:** `feat(models): implement model decision engine with hurdle ensemble conditions`

---

### [x] Task 6: DeepOptunaTuner (Multivariate TPE + HyperbandPruner)

**Files:**
- Create: `src/quantautoml/models/tuner.py`
- Test: `tests/unit/test_tuner.py`

**Interfaces:**
- Produces:
  - `class TunedCandidate(BaseModel)`: `model_name: str, best_params: dict, best_score: float, fitted_model: Any`
  - `class TuningResult(BaseModel)`: `tuned_candidates: list[TunedCandidate], study_summaries: dict[str, Any]`
  - `class DeepOptunaTuner`:
    - `__init__(cfg: ModelsConfig, task_type: TaskType)`
    - `tune(models_to_tune: list[str], X: NDArray[np.float32], y: NDArray[np.float32 | np.int32], embargo_bars: int = 0) -> TuningResult`
    - Sử dụng `optuna.samplers.TPESampler(multivariate=True)` kết hợp `optuna.pruners.HyperbandPruner`.
    - Hàm mục tiêu: $\text{IC Sharpe}_{WF} - \text{complexity\_penalty} \times \text{Complexity\_Score}$.

**Action Steps:**
1. Viết test thất bại `tests/unit/test_tuner.py`:
   - Chạy tuning với 3 trials trên model giả lập, kiểm tra hàm mục tiêu trừ điểm phạt complexity.
   - Kiểm tra HyperbandPruner ngắt các trial kém hiệu quả.
   - Kiểm tra mô hình cuối cùng được fit lại (refit) trên toàn bộ `split.train`.
2. Hiện thực `DeepOptunaTuner` với không gian tham số phân bổ chuẩn xác cho từng họ GBDT.

- **Verification Command:** `uv run pytest tests/unit/test_tuner.py -v`
- **Commit Message:** `feat(models): implement deep optuna tuner with multivariate tpe and hyperband`

---

### [x] Task 7: Pipeline Integration, Reporting & Zero Leakage Verification

**Files:**
- Create: `src/quantautoml/models/__init__.py`, `tests/integration/test_zero_leakage_models.py`
- Modify: `src/quantautoml/pipeline.py`, `tests/unit/test_pipeline.py`

**Interfaces:**
- Consumes: `FeatureFunnel`, `TournamentResult`, `DecisionReport`, `TuningResult`.
- Produces:
  - `PipelineResult` mở rộng:
    - `tournament_result: TournamentResult | None = None`
    - `decision_report: DecisionReport | None = None`
    - `tuning_result: TuningResult | None = None`
  - Ghi các file báo cáo JSON khi `cfg.models is not None`:
    - `tournament_report.json`
    - `decision_report.json`
    - `tuning_report.json`
  - Xuất các class chính qua `quantautoml.models`.

**Action Steps:**
1. Viết test thất bại `tests/integration/test_zero_leakage_models.py`:
   - Xác nhận mọi fold Walk-Forward, Tournament, và Optuna Tuning **chỉ đọc `split.train`**.
   - Spy bọc dữ liệu: Nếu thay đổi giá trị trong `split.val`, `split.cal_a`, `split.oos`, kết quả Tournament & Tuning trên Train không đổi 1 bit.
   - Kiểm tra kiểm soát kiểu `np.float32` trên mọi output dự đoán (RULE-06).
2. Nối luồng huấn luyện vào `run_pipeline` trong `src/quantautoml/pipeline.py`.
3. Chạy toàn bộ test suite `uv run pytest tests/ -v` xác nhận 100% PASS không hồi quy.
4. Cập nhật checkbox checklist trong `docs/plan/PLAN.md`.

- **Verification Command:** `uv run pytest tests/ -v`
- **Commit Message:** `feat(pipeline): wire gbdt tournament, decision engine and tuner into pipeline`

---

## Final Verification Checklist

- [x] `uv run pytest tests/` đạt 100% PASS (bao gồm cả Phase 1 và Phase 2).
- [x] Huấn luyện chỉ thực hiện trên `split.train` (Zero Leakage Test PASS).
- [x] GBDT Trio hỗ trợ đầy đủ LightGBM, CatBoost, XGBoost.
- [x] Tournament tôn trọng ngân sách 30 trials và xếp hạng bằng IC Sharpe.
- [x] Decision Engine fallback về Single Model khi Diversity $< 0.12$ (tương quan $> 0.88$).
- [x] Deep Optuna Tuner phạt độ phức tạp $\text{IC Sharpe} - 0.05 \times \text{Complexity}$.
- [x] Sẵn sàng tích hợp sang Phase 4: Output của Phase 3 (`TuningResult` và candidate models) là đầu vào trực tiếp cho SLSQP Convex Ensemble trên Cal-B và Temperature Calibration trên Cal-A.
