# Implementation Plan: Quant AutoML Pipeline — Phase 4

> **For agentic workers:** REQUIRED SUB-SKILL: dùng `superpowers:subagent-driven-development` (khuyến nghị) hoặc `superpowers:executing-plans` để thực thi plan theo từng task. Quy trình TDD bắt buộc (RED → GREEN → REFACTOR), cập nhật `- [x]` sau mỗi task theo RULE-10.

**Active Phase:** Phase 4 — Convex Ensemble Optimization ($C^2$ Objective) & Temperature Calibration  
**Branch:** `develop/v1.0-phase4`  
**Goal:** Tối ưu hóa trọng số kết hợp mô hình theo phương pháp quy hoạch phi tuyến SLSQP có hàm mục tiêu khả vi bậc hai liên tục ($C^2$) trên tập Cal-B, kiểm soát tính vững bằng Bootstrap Shrinkage về $1/K$ (kèm 5% Hurdle Rule), và hiệu chuẩn xác suất bằng Temperature Scaling trên tập Cal-A đảm bảo bảo toàn 100% Rank IC và $\text{ECE} < 0.05$.  
**Architecture:** `ensemble/` là module độc lập tiếp nhận đầu ra của Phase 3 (`TuningResult` hoặc `DecisionReport` với $K \le 3$ models). `slsqp.py` giải bài toán tối ưu với hàm mục tiêu Pearson trên ma trận dự đoán đã Rank-Transform ($\tilde{Y} \in [0, 1]$), ràng buộc $0.05 \le w_i \le 0.60, \sum w_i = 1$. `bootstrap.py` lấy mẫu $80\%$ Cal-B ($N=200$), tính median weight, co cụm shrinkage về $1/K$ và áp dụng hurdle threshold $+5\%$ IC Sharpe. `calibration.py` tối ưu scalar $T > 0$ trên Cal-A để cực tiểu hóa NLL/ECE mà không làm thay đổi thứ tự phân hạng. Metric $\text{ECE}$ nằm tại `metrics/calibration.py`.  
**Tech Stack:** `scipy.optimize`, `numpy`, `pydantic` v2, `pytest`.  
**Spec:** [`PHASES.md`](PHASES.md) (Phase 4), [`ARCHITECTURE.md`](ARCHITECTURE.md) §4, [`TRADE-OFF.md`](TRADE-OFF.md) §4, [`STRUCTURE.md`](STRUCTURE.md), [`05-ensemble-calibration/`](05-ensemble-calibration/).

---

## Global Constraints & Hard Gates

- **RULE-01 (Zero Look-Ahead Leakage):** SLSQP Optimization và Bootstrap Shrinkage chỉ được fit trên tập `split.cal_b`. Temperature Scaling chỉ được fit trên tập `split.cal_a`. Tuyệt đối không đọc `split.val` hoặc `split.oos`. Vi phạm $\to$ `DataLeakageError`.
- **RULE-02 (Rank Metric Consistency):** Mọi đánh giá tối ưu hóa và kiểm định chất lượng mô hình ensemble phải dựa trên Spearman Rank IC & IC Sharpe.
- **RULE-03 (Rank Invariance trong Calibration):** Calibration bắt buộc dùng Temperature Scaling ($T > 0$) hoặc Platt Scaling. Tuyệt đối **CẤM dùng Isotonic Regression** vì tạo điểm bằng nhau (ties) làm méo mó thứ tự phân hạng alpha. Kiểm định: $\text{Rank}(y_{calibrated}) \equiv \text{Rank}(y_{raw})$ với $100\%$ mẫu.
- **RULE-04 (Tính Khả Vi $C^2$ trong SLSQP):** Thuật toán `SLSQP` trên Cal-B bắt buộc nhận hàm mục tiêu là **Pearson Correlation trên dữ liệu đã Rank-Transform** ($\tilde{Y} \in [0, 1]$). Cấm truyền trực tiếp hàm Spearman Rank IC rời rạc vào SLSQP để tránh bế tắc đạo hàm (Zero-Gradient trap).
- **RULE-05 (Earn Its Place / Hurdle Ensemble):** Trọng số Ensemble từ SLSQP chỉ được chấp thuận kích hoạt khi vượt qua Single Best Model và baseline Equal Weights ($1/K$) ít nhất $5\%$ IC Sharpe trên tập đánh giá. Ngược lại, tự động fallback về Single Best Model hoặc $W = [1/K, \dots, 1/K]$.
- **RULE-06 (Numerical Precision Contract):** Mọi mảng số học dùng chuẩn `np.float32`.
- **Python $\ge 3.12$**, type hints 100%, Pydantic schemas cho data contract, custom exceptions kế thừa `QuantAutoMLError`, không dùng `try ... except Exception: pass`.

---

## Execution Order Overview

- [x] **Task 1: Ensemble Config & Custom Exceptions** — Định nghĩa Pydantic `EnsembleConfig`, bổ sung vào `PipelineConfig`, định nghĩa custom exceptions (`EnsembleError`, `OptimizationConvergenceError`, `CalibrationError`).
- [x] **Task 2: Quantitative Metrics — Expected Calibration Error (ECE)** — Hiện thực hàm đo lường `calc_ece()` chuẩn hóa tại `metrics/calibration.py` để lượng hóa sai số calibration cho binary & multiclass.
- [ ] **Task 3: Smooth SLSQP Optimizer ($C^2$ Objective)** — Hiện thực `SmoothSLSQPOptimizer` trên `cal_b`: Rank-transform $\tilde{Y} \in [0, 1]$, tối ưu hàm Pearson tương quan khả vi $C^2$ với ràng buộc $0.05 \le w_i \le 0.60$ và $\sum w_i = 1$.
- [ ] **Task 4: Bootstrap Shrinkage Engine & Hurdle Fallback (RULE-05)** — Hiện thực `BootstrapShrinkageEngine`: $N=200$ lần bootstrap (subsample $80\%$), tính vector trọng số trung vị, áp dụng Shrinkage về $1/K$, và thẩm định ngưỡng vượt trội $\ge 5\%$ IC Sharpe.
- [ ] **Task 5: Temperature Scaling Calibrator (RULE-03 Rank Invariance)** — Hiện thực `TemperatureCalibrator` trên `cal_a`: Tối ưu scalar $T > 0$ cực tiểu hóa Negative Log-Likelihood, đảm bảo $\text{ECE} < 0.05$ và bảo toàn $100\%$ Rank IC gốc.
- [ ] **Task 6: Pipeline Integration, Reporting & Zero Leakage Verification** — Tích hợp module `ensemble` vào `run_pipeline()`, xuất báo cáo `ensemble_report.json` và `calibration_report.json`, kiểm tra toàn diện Zero Leakage.

---

## Detailed Task Breakdown

### [x] Task 1: Ensemble Config & Custom Exceptions

**Files:**
- Modify: `src/quantautoml/common/exceptions.py`, `src/quantautoml/common/config.py`, `docs/CONFIGURATION.md`
- Test: `tests/unit/test_ensemble_config.py`

**Interfaces:**
- Produces:
  - Exceptions in `exceptions.py`:
    - `class EnsembleError(QuantAutoMLError)`
    - `class OptimizationConvergenceError(EnsembleError)`
    - `class CalibrationError(EnsembleError)`
  - Pydantic Schema in `config.py`:
    - `class EnsembleConfig(_Strict)`:
      - `min_weight: float = 0.05` (trọng số tối thiểu mỗi model)
      - `max_weight: float = 0.60` (trọng số tối đa mỗi model)
      - `n_bootstrap: int = 200` (số lần lấy mẫu bootstrap)
      - `bootstrap_sample_ratio: float = 0.80` (tỷ lệ mẫu bootstrap mỗi lần)
      - `shrinkage_factor: float = 0.10` (hệ số co cụm về $1/K$)
      - `hurdle_ratio: float = 0.05` (ngưỡng vượt trội tối thiểu 5% IC Sharpe)
      - `max_ece_threshold: float = 0.05` (ngưỡng ECE tối đa chấp nhận sau calibration)
      - `random_state: int = 42`
  - Modify `PipelineConfig`:
    - Thêm `ensemble: EnsembleConfig | None = None`
  - Documentation:
    - Cập nhật `docs/CONFIGURATION.md` phần `[ensemble]`.

**Action Steps:**
1. Viết unit test thất bại `tests/unit/test_ensemble_config.py`:
   - Parse config hợp lệ với section `[ensemble]`.
   - Validate lỗi khi `min_weight < 0`, `max_weight > 1.0`, `min_weight >= max_weight`.
   - Validate lỗi khi `n_bootstrap < 10`, `shrinkage_factor < 0` hoặc `> 1.0`.
   - Validate lỗi khi `hurdle_ratio < 0`.
2. Hiện thực các lớp ngoại lệ trong `exceptions.py` và `EnsembleConfig` trong `config.py`.
3. Cập nhật `docs/CONFIGURATION.md`.

- **Verification Command:** `uv run pytest tests/unit/test_ensemble_config.py -v`
- **Commit Message:** `feat(ensemble): add ensemble config schemas and custom exceptions`

---

### [x] Task 2: Quantitative Metrics — Expected Calibration Error (ECE)

**Files:**
- Create: `src/quantautoml/metrics/calibration.py`
- Modify: `src/quantautoml/metrics/__init__.py`
- Test: `tests/unit/test_metrics_calibration.py`

**Interfaces:**
- Produces:
  - `def calc_ece(y_true: NDArray[np.int32 | np.float32], y_prob: NDArray[np.float32], *, n_bins: int = 10) -> float`
    - Chia khoảng xác suất $[0, 1]$ thành `n_bins` khoảng đều nhau.
    - Với mỗi bin $B_m$: tính độ chính xác $\text{acc}(B_m) = \frac{1}{|B_m|} \sum_{i \in B_m} \mathbf{1}(\hat{y}_i = y_i)$ và độ tự tin trung bình $\text{conf}(B_m) = \frac{1}{|B_m|} \sum_{i \in B_m} \hat{p}_i$.
    - $\text{ECE} = \sum_{m=1}^{M} \frac{|B_m|}{N} |\text{acc}(B_m) - \text{conf}(B_m)|$.
    - Hỗ trợ cả nhị phân (`y_prob` 1D $[0, 1]$) và đa lớp (`y_prob` 2D $[N, C]$).
    - Trả về `float` chuẩn $[0, 1]$, ném `ValueError` nếu dữ liệu rỗng hoặc shape không tương thích.

**Action Steps:**
1. Viết test thất bại `tests/unit/test_metrics_calibration.py`:
   - Test mô hình hiệu chuẩn hoàn hảo ($\hat{p} = y$) $\to \text{ECE} \approx 0.0$.
   - Test mô hình tự tin thái quá ($\hat{p} = 0.99$ nhưng $y = 0$) $\to \text{ECE} \approx 0.99$.
   - Test bài toán Binary (1D) và Multiclass (2D).
   - Test mảng rỗng hoặc có NaN ném lỗi phù hợp.
2. Hiện thực `calc_ece` trong `src/quantautoml/metrics/calibration.py` và export qua `src/quantautoml/metrics/__init__.py`.

- **Verification Command:** `uv run pytest tests/unit/test_metrics_calibration.py -v`
- **Commit Message:** `feat(metrics): implement expected calibration error metric for probability calibration`

---

### [ ] Task 3: Smooth SLSQP Optimizer ($C^2$ Objective)

**Files:**
- Create: `src/quantautoml/ensemble/slsqp.py`
- Test: `tests/unit/test_slsqp_smooth.py`

**Interfaces:**
- Produces:
  - `class SmoothSLSQPOptimizer`:
    - `__init__(min_weight: float = 0.05, max_weight: float = 0.60, random_state: int = 42)`
    - `optimize(predictions: NDArray[np.float32], y_true: NDArray[np.float32 | np.int32]) -> NDArray[np.float32]`
      - `predictions` có shape $(N, K)$, đại diện cho dự đoán của $K$ mô hình trên Cal-B.
      - **Bước 1 (Rank-Transformation):** Chuẩn hóa từng cột $\tilde{Y}_j = \frac{\text{Rank}(\hat{Y}_j)}{N} \in [0, 1]$ và nhãn đích $\tilde{Y}_{target} = \frac{\text{Rank}(Y_{true})}{N} \in [0, 1]$.
      - **Bước 2 (Hàm mục tiêu khả vi $C^2$):**
        $$f(w) = - \frac{\text{Cov}\left(\sum_{j=1}^K w_j \tilde{Y}_j, \, \tilde{Y}_{target}\right)}{\sigma\left(\sum_{j=1}^K w_j \tilde{Y}_j\right) \cdot \sigma\left(\tilde{Y}_{target}\right)}$$
        (Tối thiểu hóa âm của tương quan Pearson giữa tổ hợp tuyến tính và nhãn rank-transformed).
      - **Bước 3 (Ràng buộc lồi):**
        - Ràng buộc đẳng thức: $\sum_{j=1}^K w_j = 1$.
        - Ràng buộc hộp: $w_j \in [\text{min\_weight}, \text{max\_weight}]$ (với cơ chế adaptive bounds nếu $K < 2$ hoặc $K \times \text{min\_weight} > 1$).
      - **Bước 4:** Khởi tạo $w_0 = [1/K, \dots, 1/K]$, gọi `scipy.optimize.minimize(method="SLSQP")`.
      - Ném `OptimizationConvergenceError` nếu tối ưu thất bại và fallback về $w = [1/K, \dots, 1/K]$.
      - Trả về mảng numpy trọng số kiểu `np.float32`.

**Action Steps:**
1. Viết test thất bại `tests/unit/test_slsqp_smooth.py`:
   - Test ràng buộc tổng trọng số $\sum w_i = 1$ và $0.05 \le w_i \le 0.60$.
   - Test tính hội tụ nhanh ($< 50\text{ms}$) trên mẫu $N = 2,000, K = 3$.
   - Test mô hình vượt trội hơn hẳn được gán trọng số cao hơn tiệm cận trần `max_weight`.
   - Test $K=1$ tự động trả về $w = [1.0]$.
   - Test dữ liệu hằng số hoặc bất thường kích hoạt fallback an toàn.
2. Hiện thực `SmoothSLSQPOptimizer` trong `slsqp.py`.

- **Verification Command:** `uv run pytest tests/unit/test_slsqp_smooth.py -v`
- **Commit Message:** `feat(ensemble): implement smooth slsqp optimizer with c2 rank-transformed objective`

---

### [ ] Task 4: Bootstrap Shrinkage Engine & Hurdle Fallback (RULE-05)

**Files:**
- Create: `src/quantautoml/ensemble/bootstrap.py`
- Test: `tests/unit/test_bootstrap_shrinkage.py`

**Interfaces:**
- Produces:
  - `class EnsembleWeightsResult(BaseModel)`:
    - `weights: list[float]` (trọng số cuối cùng cho $K$ models)
    - `raw_weights: list[float]` (trọng số trung vị bootstrap)
    - `bootstrap_std: list[float]` (độ lệch chuẩn trọng số qua 200 trials)
    - `is_ensemble_accepted: bool` (đạt hay vi phạm 5% Hurdle Rule)
    - `ensemble_ic_sharpe: float`
    - `best_single_ic_sharpe: float`
    - `equal_weights_ic_sharpe: float`
    - `hurdle_ratio_achieved: float`
    - `fallback_reason: str | None = None`
  - `class BootstrapShrinkageEngine`:
    - `__init__(cfg: EnsembleConfig)`
    - `fit(predictions: NDArray[np.float32], y_true: NDArray[np.float32 | np.int32], single_model_scores: list[float] | None = None) -> EnsembleWeightsResult`
    - **Thuật toán:**
      1. Lặp $N = \text{n\_bootstrap}$ (200 lần): Lấy mẫu ngẫu nhiên không hoàn lại $80\%$ quan sát từ Cal-B, chạy `SmoothSLSQPOptimizer` thu được $w^{(b)}$.
      2. Tính vector trọng số trung vị: $w_{\text{med}} = \text{median}(W, \text{axis}=0)$.
      3. Chuẩn hóa lại tổng: $w_{\text{med}} = w_{\text{med}} / \sum w_{\text{med}}$.
      4. Áp dụng Co cụm (Shrinkage):
         $$w_{\text{shrunk}} = (1 - \lambda) w_{\text{med}} + \lambda \left[\frac{1}{K}, \dots, \frac{1}{K}\right]$$
         với $\lambda = \text{shrinkage\_factor}$ ($0.10$).
      5. Đánh giá **Hurdle Rule (RULE-05):**
         - Tính IC Sharpe của tổ hợp $w_{\text{shrunk}}$ trên toàn bộ Cal-B.
         - So sánh với IC Sharpe của Single Best Model và Equal Weights ($1/K$).
         - Nếu $\text{IC Sharpe}_{ens} < (1 + \text{hurdle\_ratio}) \times \max(\text{SingleBest}, \text{EqualWeights})$:
           $\to$ Từ chối ensemble phức tạp, fallback về Single Best Model (hoặc Equal Weights nếu single model kém hơn).
         - Đánh dấu `is_ensemble_accepted = False` và ghi rõ `fallback_reason`.

**Action Steps:**
1. Viết test thất bại `tests/unit/test_bootstrap_shrinkage.py`:
   - Kiểm tra Bootstrap 200 lần sinh ra phân phối trọng số ổn định.
   - Kiểm tra hiệu ứng Shrinkage: Trọng số lệch xa $1/K$ bị kéo về $1/K$ theo hệ số $\lambda$.
   - Test case Hurdle không đạt: Fallback về Single Best Model hoặc Equal Weights, `is_ensemble_accepted == False`.
   - Test case Hurdle đạt: Chấp thuận trọng số shrunk, `is_ensemble_accepted == True`.
2. Hiện thực `BootstrapShrinkageEngine` trong `bootstrap.py`.

- **Verification Command:** `uv run pytest tests/unit/test_bootstrap_shrinkage.py -v`
- **Commit Message:** `feat(ensemble): implement bootstrap shrinkage engine with hurdle fallback gate`

---

### [ ] Task 5: Temperature Scaling Calibrator (RULE-03 Rank Invariance)

**Files:**
- Create: `src/quantautoml/ensemble/calibration.py`
- Test: `tests/unit/test_temperature_calibration.py`

**Interfaces:**
- Produces:
  - `class CalibrationResult(BaseModel)`:
    - `temperature: float` ($T > 0$)
    - `uncalibrated_ece: float`
    - `calibrated_ece: float`
    - `rank_ic_preserved: bool` (bắt buộc True)
    - `method: str = "temperature_scaling"`
  - `class TemperatureCalibrator`:
    - `__init__(task_type: TaskType, random_state: int = 42)`
    - `fit(logits_or_probs: NDArray[np.float32], y_true: NDArray[np.int32 | np.float32]) -> Self`
      - Tối ưu hóa biến vô hướng $T > 0$ trên tập **Cal-A** bằng `scipy.optimize.minimize_scalar(method="bounded", bounds=(0.01, 10.0))`.
      - Hàm tổn thất: Negative Log-Likelihood (Cross-Entropy).
    - `calibrate(logits_or_probs: NDArray[np.float32]) -> NDArray[np.float32]`
      - Tính $\sigma(z / T)$ (hoặc softmax với Multiclass).
      - Ép kiểu kết quả trả về `np.float32`.
    - `verify_rank_invariance(y_raw: NDArray[np.float32], y_calibrated: NDArray[np.float32]) -> bool`
      - Kiểm tra nghiêm ngặt: Thứ tự phân hạng của các mẫu trước và sau hiệu chuẩn phải đồng nhất $100\%$ (Spearman correlation giữa raw và calibrated bằng $1.0 - 10^{-7}$).

**Action Steps:**
1. Viết test thất bại `tests/unit/test_temperature_calibration.py`:
   - Test $T > 0$ được tìm thấy làm giảm ECE trên tập kiểm thử phân phối lệch.
   - **Test Rank Invariance (Bắt buộc theo RULE-03):** Xác nhận $\text{Rank}(y_{\text{calibrated}}) \equiv \text{Rank}(y_{\text{raw}})$ không sai lệch bất kỳ vị trí nào.
   - Test bài toán Binary và Multiclass.
   - Test kiểm tra nếu dữ liệu đầu vào là xác suất $[0, 1]$, calibrator tự động chuyển đổi sang log-odds (logit) ổn định số học trước khi scale nhiệt độ.
2. Hiện thực `TemperatureCalibrator` trong `src/quantautoml/ensemble/calibration.py`.

- **Verification Command:** `uv run pytest tests/unit/test_temperature_calibration.py -v`
- **Commit Message:** `feat(ensemble): implement temperature scaling calibrator with strict rank invariance`

---

### [ ] Task 6: Pipeline Integration, Reporting & Zero Leakage Verification

**Files:**
- Create: `src/quantautoml/ensemble/__init__.py`, `tests/integration/test_zero_leakage_ensemble.py`
- Modify: `src/quantautoml/pipeline.py`, `tests/unit/test_pipeline.py`

**Interfaces:**
- Produces:
  - `PipelineResult` mở rộng:
    - `ensemble_result: EnsembleWeightsResult | None = None`
    - `calibration_result: CalibrationResult | None = None`
  - Ghi báo cáo JSON vào artifacts/output:
    - `ensemble_report.json`: Trọng số $w$, kết quả bootstrap, quyết định hurdle, so sánh IC Sharpe.
    - `calibration_report.json`: Giá trị nhiệt độ $T$, ECE trước/sau hiệu chuẩn, chứng nhận bảo toàn Rank IC.
  - Export public API qua `quantautoml.ensemble`.

**Action Steps:**
1. Viết test thất bại `tests/integration/test_zero_leakage_ensemble.py`:
   - **Zero Look-Ahead Leakage (RULE-01):**
     - Đảm bảo `BootstrapShrinkageEngine.fit()` chỉ nhận `split.cal_b`.
     - Đảm bảo `TemperatureCalibrator.fit()` chỉ nhận `split.cal_a`.
     - Tuyệt đối không đọc hoặc sử dụng bất kỳ mẫu nào từ `split.val` hoặc `split.oos`.
     - Spy data test: Thay đổi giá trị trên `split.val` và `split.oos` không làm thay đổi trọng số ensemble $w$ hoặc nhiệt độ $T$ dù chỉ 1 bit.
   - **Hợp đồng độ chính xác số học (RULE-06):** Toàn bộ ma trận trung gian và đầu ra dự đoán cuối cùng đều là `np.float32`.
2. Nối luồng `ensemble` và `calibration` vào hàm `run_pipeline()` trong `pipeline.py`.
3. Chạy toàn bộ test suite `uv run pytest tests/ -v` xác nhận 100% PASS không hồi quy.
4. Cập nhật checkbox checklist trong `docs/plan/PLAN.md`.

- **Verification Command:** `uv run pytest tests/ -v`
- **Commit Message:** `feat(pipeline): wire convex ensemble and temperature calibration into pipeline`

---

## Final Verification Checklist

- [ ] `uv run pytest tests/` đạt 100% PASS (bao gồm Phase 1, Phase 2, Phase 3, Phase 4).
- [ ] Ràng buộc trọng số SLSQP thỏa mãn: $\sum w_i = 1$ và $0.05 \le w_i \le 0.60$.
- [ ] Bootstrap $N=200$ kết hợp co cụm Shrinkage về $1/K$ ổn định số học.
- [ ] Hurdle Rule (RULE-05) tự động fallback khi không vượt trội $\ge 5\%$ IC Sharpe.
- [ ] Temperature Scaling bảo toàn $100\%$ Rank IC gốc (RULE-03 Rank Invariance Test PASS) và đạt $\text{ECE} < 0.05$.
- [ ] Zero Leakage Test PASS: Cal-B dùng cho ensemble, Cal-A dùng cho calibration, cấm chạm vào Val/OOS.
- [ ] Sẵn sàng chuyển giao sang Phase 5: Gói gọn Ensemble weights và Calibrator vào ONNX Model Bundle và Friction Backtest.
