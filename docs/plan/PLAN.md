# Implementation Plan: Quant AutoML Pipeline

> **For agentic workers:** REQUIRED SUB-SKILL: dùng `superpowers:subagent-driven-development` (khuyến nghị) hoặc `superpowers:executing-plans` để thực thi plan theo từng task. Quy trình TDD bắt buộc (RED → GREEN → REFACTOR), cập nhật `- [x]` sau mỗi task theo RULE-10.

**Active Phase:** Phase 2 — Leakage-Free Feature Funnel (Fast dCor & FDR Control)
**Branch:** `develop/v1.0-phase2`
**Goal:** Xây phễu lọc tính năng 7 tầng (F1 → F2 → F3+7 → F4 → F5 → F6 → F8) chỉ `fit()` trên Train, kiểm soát FDR ≤ 0.05, dCor chạy được với N > 10,000 mà không tràn RAM.
**Architecture:** Mỗi tầng là một `BaseFeatureFilter` độc lập (SRP), nhận mảng `float32` của segment Train để `fit()` và chỉ `transform()` trên Val/Cal/OOS. `FeatureFunnel` nối tuần tự các tầng, ghi lại báo cáo loại bỏ theo từng tầng. Các hàm thống kê dùng chung (Rank IC, ICIR, BH-FDR) nằm ở `metrics/` (tầng nền, không import ngược).
**Tech Stack:** `numpy`, `scipy` (Spearman, `cluster.hierarchy`), `scikit-learn` (mutual information), `dcor` (Fast dCor, phương thức `AVL`), `pydantic` v2, `pytest`.
**Spec:** [`SCOPE.md`](SCOPE.md) (Module 2), [`PHASES.md`](PHASES.md) (Phase 2), [`ARCHITECTURE.md`](ARCHITECTURE.md) §3.3, [`TRADE-OFF.md`](TRADE-OFF.md) §4, [`STRUCTURE.md`](STRUCTURE.md).

## Global Constraints

- **RULE-01:** `fit()` chỉ nhận `Segment` Train; Val/Cal/OOS chỉ `transform()`. Vi phạm → `DataLeakageError`.
- **RULE-02:** mọi tiêu chí chọn feature dựa trên Spearman Rank IC / ICIR (MI chỉ dùng cho F1/F5/F6/F8 như spec quy định).
- **RULE-06:** đầu vào/đầu ra của mọi filter là `np.float32`. Ngoại lệ có chủ đích: dCor cast tạm sang `float64` ngay trước khi gọi `dcor` (float32 làm `dcor` rơi về bản chưa biên dịch, đã kiểm chứng), kết quả không lưu lại.
- **RULE-08:** BH-FDR ≤ 0.05 và đơn điệu kỳ hạn trên `{1, 5, 10, 21}`.
- Ngưỡng cố định từ spec: ICIR(h_target) > 0.30; dCor > 0.85; rolling window = 63; IC horizons = {1, 5, 10, 21}; Max-MI / Rank-IC chọn đại diện cụm; khoảng cách cụm = `1 − |corr(IC_i, IC_j)|`.
- Python ≥ 3.12, type hints 100%, không `Any` mơ hồ, exception kế thừa `QuantAutoMLError`, không `except Exception: pass`.
- Phụ thuộc một chiều: `common`/`metrics` không import `features`; `features` chỉ import `common`, `metrics`, `numpy`, `scipy`, `sklearn`, `dcor`.
- Zero Scope Creep: `calc_ic_sharpe`, Downside IC, `calibration.py` (ECE) **không** thuộc Phase 2 (để Phase 3/4/5).

## Quyết định thiết kế (cần bạn xác nhận)

Spec không chốt các điểm dưới đây; plan chọn mặc định hợp lý, đều chỉnh được qua `[features]` trong config:

1. **Ánh xạ horizon → cột nhãn:** `y` là nhãn phân loại, nên IC đa kỳ hạn lấy từ `labels_aux` (cột forward-return). Config khai báo `features.ic_horizons = {1 = "fwd_ret_1", 5 = "fwd_ret_5", ...}` và `features.target_horizon`.
2. **F1 MCAR/MAR/MNAR:** quy ước vận hành (không phải kiểm định thống kê chính thức): `MNAR` nếu MI(indicator NaN, y) > `indicator_mi_min`; `MAR` nếu |Spearman(indicator, feature khác)| lớn nhất > 0.3; còn lại `MCAR`. Chỉ `MNAR`/`MAR` sinh cột `is_nan_<tên>`. Hard Gate: cột toàn NaN hoặc NaN ratio > `max_nan_ratio` (0.6) bị loại.
3. **F2 ngưỡng:** `min_unique_ratio=0.01`, `min_cv=1e-6`, `max_peak_var_ratio=50` (phân vị 90 của rolling variance w=63 chia phương sai toàn Train). Đây là giá trị khởi điểm, cần hiệu chỉnh trên dữ liệu thật.
4. **F6 regime:** Bull/Bear/Sideways suy ra *nhân quả* từ nhãn h=1 đã realized (dịch 1 nến), tổng 63 nến, ngưỡng ±0.5σ của Train. Thiếu cột h=1 → F6 bỏ qua kèm cảnh báo trong báo cáo.
5. **F8:** `max_features=30` (mặc định); MI rời rạc hóa bằng 10 quantile bin.
6. **dCor:** `dcor.distance_correlation(..., method="AVL")` trên toàn Train (đã đo: N=20,000 ≈ 0.04 s/cặp). Chỉ tính cặp giữa các feature còn sống, không bao giờ tạo ma trận N×N. Quy tắc loại trong cặp: giữ feature có |Rank IC| cao hơn.

## Review Focus

Các đầu vào mà spec không nói tới nhưng dễ gây hỏng, mỗi dòng có test ở task sở hữu:

1. Nhãn `y = -1` (mid-NaN từ Phase 1) lọt vào tính IC/MI làm lệch kết quả → phải mask (Task 2, 5).
2. Cột hằng số / toàn NaN / chỉ 1 giá trị → không được ném `ZeroDivisionError` hay trả NaN IC (Task 2, 5, 6, 8).
3. Train quá ngắn (N < `rolling_window` × 2) → báo `InvalidDataError` rõ ràng thay vì mảng rỗng (Task 2, 7).
4. Không feature nào sống sau một tầng → funnel trả tập rỗng kèm báo cáo, không crash ở tầng sau (Task 12).
5. Thay đổi giá trị Val/OOS không được làm thay đổi tập feature chọn (kiểm tra rò rỉ ở mức hành vi) (Task 12).

---

## Execution Order Overview

- [ ] **Task 1:** Dependencies (`scipy`, `scikit-learn`, `dcor`), exceptions & `FeaturesConfig`.
- [ ] **Task 2:** `metrics/ic.py` — Rank IC, Rolling IC, ICIR.
- [ ] **Task 3:** `metrics/fdr.py` — Benjamini-Hochberg.
- [ ] **Task 4:** `features/base.py` — `BaseFeatureFilter` & `FilterReport`.
- [ ] **Task 5:** F1 `triage.py` — NaN/Inf Triage & MI score.
- [ ] **Task 6:** F2 `variance.py` — Multi-Metric Variance.
- [ ] **Task 7:** F3+7 `ic_stability.py` — IC Stability + FDR + Term Structure.
- [ ] **Task 8:** F4 `collinearity.py` — Fast dCor.
- [ ] **Task 9:** F5 `clustering.py` — IC-Behavior Clustering.
- [ ] **Task 10:** F6 `regime_synergy.py` — Regime MI & Synergy.
- [ ] **Task 11:** F8 `jmi.py` — Joint Mutual Information.
- [ ] **Task 12:** `funnel.py` + wiring vào `pipeline.py` + integration test chống rò rỉ & scalability.

---

## Detailed Task Breakdown

### [ ] Task 1: Dependencies, Exceptions & FeaturesConfig

**Files:**
- Modify: `pyproject.toml`, `src/quantautoml/common/exceptions.py`, `src/quantautoml/common/config.py`, `docs/CONFIGURATION.md`
- Test: `tests/unit/test_config.py` (mở rộng)

**Interfaces:**
- Produces: `class FeatureSelectionError(QuantAutoMLError)`; `class FeaturesConfig(_Strict)` với các trường: `ic_horizons: dict[int, str]`, `target_horizon: int`, `icir_min: float = 0.30`, `fdr_alpha: float = 0.05`, `rolling_window: int = 63`, `dcor_threshold: float = 0.85`, `cluster_distance: float = 0.30`, `max_nan_ratio: float = 0.6`, `min_unique_ratio: float = 0.01`, `min_cv: float = 1e-6`, `max_peak_var_ratio: float = 50.0`, `max_features: int = 30`, `monotonic_tolerance: float = 0.05`; `PipelineConfig.features: FeaturesConfig | None = None`.

**Action Steps:**
1. Test thất bại trong `test_config.py`: (a) TOML có `[features]` hợp lệ parse đúng; (b) `target_horizon` không nằm trong `ic_horizons` → `ValidationError`; (c) thiếu `[features]` → `features is None` (tương thích ngược Phase 1); (d) `fdr_alpha` ngoài `(0, 0.05]` → `ValidationError` (RULE-08).
2. Thêm vào `pyproject.toml`: `scipy>=1.13.0`, `scikit-learn>=1.4.0`, `dcor>=0.7`, rồi chạy `uv sync`.
3. Hiện thực `FeatureSelectionError`, `FeaturesConfig` (validator `target_horizon in ic_horizons`), trường `features` của `PipelineConfig`.
4. Cập nhật `docs/CONFIGURATION.md` thêm mục "Phase 2 — Feature Funnel" với ví dụ TOML.

- **Verification Command:** `uv run pytest tests/unit/test_config.py -v` → PASS.
- **Commit Message:** `chore(setup): add phase 2 dependencies and features config`

---

### [ ] Task 2: Rank IC, Rolling IC & ICIR

**Files:**
- Create: `src/quantautoml/metrics/__init__.py`, `src/quantautoml/metrics/ic.py`
- Test: `tests/unit/test_metrics_ic.py`

**Interfaces:**
- Produces:
  - `calc_rank_ic(x: NDArray[np.float32], y: NDArray[np.float32 | np.int32], *, ignore_value: float | None = None) -> tuple[float, float]` → `(rho, p_value)`; bỏ cặp NaN và cặp `y == ignore_value`; < 3 cặp hợp lệ hoặc `x`/`y` hằng → `(0.0, 1.0)`.
  - `calc_rolling_ic(x, y, window: int, *, ignore_value: float | None = None) -> NDArray[np.float64]` → IC trên cửa sổ không chồng lấp (bước = `window`), độ dài `n // window`; cửa sổ suy biến → `NaN`.
  - `calc_icir(ic_series: NDArray[np.float64]) -> float` → `mean/std` (bỏ NaN, `ddof=1`); std = 0 hoặc < 2 điểm → `0.0`.

**Action Steps:**
1. Test thất bại: `rho == 1.0` khi `y = x`; `rho == -1.0` khi `y = -x`; khớp `scipy.stats.spearmanr` trên dữ liệu ngẫu nhiên (`atol=1e-9`); `y=-1` bị loại (kết quả bằng kết quả khi cắt các dòng đó); cột hằng → `(0.0, 1.0)`; toàn NaN → `(0.0, 1.0)`; `calc_rolling_ic` trả đúng độ dài và NaN ở cửa sổ hằng; `calc_icir` của chuỗi hằng = `0.0`.
2. Hiện thực bằng `scipy.stats.spearmanr` (kiểm tra suy biến trước khi gọi để tránh cảnh báo).

- **Verification Command:** `uv run pytest tests/unit/test_metrics_ic.py -v`
- **Commit Message:** `feat(metrics): implement rank IC, rolling IC and ICIR`

---

### [ ] Task 3: Benjamini-Hochberg FDR

**Files:**
- Create: `src/quantautoml/metrics/fdr.py`
- Test: `tests/unit/test_metrics_fdr.py`

**Interfaces:**
- Produces: `apply_fdr_correction(p_values: NDArray[np.float64], alpha: float = 0.05) -> NDArray[np.bool_]` → mặt nạ "được chấp nhận" theo công thức $k = \max\{i : p_{(i)} \le \frac{i}{M}\alpha\}$ (ARCHITECTURE §3.3).

**Action Steps:**
1. Test thất bại: ví dụ tay `p=[0.001, 0.008, 0.039, 0.041, 0.6]`, `alpha=0.05` → `[T, T, T, T, F]`; khớp `scipy.stats.false_discovery_control(p) <= alpha` trên 1,000 p-value ngẫu nhiên; mảng rỗng → mảng rỗng; p-value NaN hoặc ngoài [0,1] → `InvalidDataError`; 1,000 p-value đều U(0,1) thuần nhiễu → tỷ lệ chấp nhận ≤ 0.05 trên 200 lần lặp seed cố định (kiểm chứng chống data-snooping).
2. Hiện thực thủ công theo công thức (sort, so ngưỡng `i/M·α`, lấy `k` lớn nhất, chấp nhận mọi p ≤ `p_(k)`), trả mặt nạ theo thứ tự gốc.

- **Verification Command:** `uv run pytest tests/unit/test_metrics_fdr.py -v`
- **Commit Message:** `feat(metrics): implement Benjamini-Hochberg FDR correction`

---

### [ ] Task 4: BaseFeatureFilter & FilterReport

**Files:**
- Create: `src/quantautoml/features/__init__.py`, `src/quantautoml/features/base.py`
- Modify: `src/quantautoml/common/types.py` (thêm `FilterReport`)
- Test: `tests/unit/test_features_base.py`

**Interfaces:**
- Produces:
  - `@dataclass(frozen=True) FilterReport(stage: str, n_in: int, n_out: int, dropped: dict[str, str], notes: tuple[str, ...] = ())` — `dropped` ánh xạ tên feature → lý do.
  - `@dataclass(frozen=True) FitContext(feature_names: tuple[str, ...], labels_aux: dict[str, NDArray[np.float32]], task_type: TaskType)`.
  - `class BaseFeatureFilter(ABC)`: `name: ClassVar[str]`; `fit(self, X: NDArray[np.float32], y: NDArray[np.int32], ctx: FitContext) -> Self` (đặt `is_fitted_`, `output_names_`, `report_`); `transform(self, X: NDArray[np.float32]) -> NDArray[np.float32]` (ném `FeatureSelectionError` nếu chưa fit hoặc số cột ≠ lúc fit; không bao giờ cập nhật trạng thái đã fit); abstract `_fit(...)` và `_transform(...)`.
  - `y` có thể `None`/`UNSUPERVISED` → `fit` ném `InvalidDataError` (phễu cần nhãn).

**Action Steps:**
1. Test thất bại bằng một filter giả: `transform` trước `fit` → `FeatureSelectionError`; `transform` sai số cột → lỗi; `transform` gọi 2 lần trên dữ liệu khác nhau không thay đổi `output_names_`; `fit` với `y=None` → `InvalidDataError`; `fit` trả `self`; đầu ra luôn `float32` (filter trả `float64` → bị ép/ném lỗi theo RULE-06).
2. Hiện thực lớp cơ sở (template method: kiểm tra → `_fit`/`_transform` → kiểm tra dtype).

- **Verification Command:** `uv run pytest tests/unit/test_features_base.py -v`
- **Commit Message:** `feat(features): add BaseFeatureFilter contract and FilterReport`

---

### [ ] Task 5: F1 — NaN/Inf Triage & MI Score

**Files:**
- Create: `src/quantautoml/features/triage.py`
- Test: `tests/unit/test_triage.py`

**Interfaces:**
- Consumes: `BaseFeatureFilter`, `FitContext`, `FeaturesConfig`.
- Produces: `class NaNInfTriage(BaseFeatureFilter)` với `__init__(self, cfg: FeaturesConfig)`; sau `fit`: `scores_: dict[str, float]` = `MI(X, y) × (1 − nan_ratio)`, `missing_kind_: dict[str, Literal["none", "MCAR", "MAR", "MNAR"]]`. `transform` thay `±Inf` bằng `NaN`, loại cột đã bị Hard Gate, **thêm cột `is_nan_<tên>`** (float32, 0/1) cho cột `MAR`/`MNAR`.

**Action Steps:**
1. Test thất bại: cột toàn NaN và cột NaN ratio 0.7 bị loại với lý do rõ; `±Inf` → `NaN`; cột không NaN → `"none"` và không sinh indicator; NaN rải ngẫu nhiên độc lập với `y` → `"MCAR"`; NaN chỉ xuất hiện khi `y == 1` → `"MNAR"` và có cột `is_nan_*` (MI > 0); `score` giảm khi NaN ratio tăng với cùng MI; `y = -1` bị mask trước khi tính MI; cột hằng → score `0.0`, không crash; `fit` hai lần với Val bị trộn vào thì `missing_kind_` không đổi khi chỉ `transform` Val (không fit lại).
2. Hiện thực: impute trung vị (Train) chỉ để tính MI, dùng `sklearn.feature_selection.mutual_info_classif` với `random_state=0`; phân loại theo quy ước ở mục "Quyết định thiết kế" #2.

- **Verification Command:** `uv run pytest tests/unit/test_triage.py -v`
- **Commit Message:** `feat(features): implement F1 NaN/Inf triage with MI scoring`

---

### [ ] Task 6: F2 — Multi-Metric Variance Filter

**Files:**
- Create: `src/quantautoml/features/variance.py`
- Test: `tests/unit/test_variance.py`

**Interfaces:**
- Produces: `class MultiMetricVarianceFilter(BaseFeatureFilter)` với `__init__(self, cfg: FeaturesConfig)`; `metrics_: dict[str, tuple[float, float, float]]` = `(cv, peak_var_ratio, unique_ratio)`. Loại feature nếu `n_unique < 2` **hoặc** (`n_unique > 2` và `unique_ratio < min_unique_ratio`) **hoặc** `|cv| < min_cv` **hoặc** `peak_var_ratio > max_peak_var_ratio`, với `peak_var_ratio` = phân vị 90 của phương sai cuộn (w = `rolling_window`, bỏ NaN) / phương sai toàn Train.

**Action Steps:**
1. Test thất bại: cột hằng bị loại (`unique_ratio`); cột nhị phân 0/1 cân bằng được giữ (quy tắc `unique_ratio` chỉ áp khi `n_unique > 2`); feature có burst phương sai cục bộ lớn bị loại bởi `peak_var_ratio`; mean ≈ 0 không gây chia cho 0 (dùng `std/ max(|mean|, eps)`); Train ngắn hơn `rolling_window` → `InvalidDataError`; toàn NaN → bị loại, không crash.
2. Hiện thực bằng `numpy.lib.stride_tricks.sliding_window_view` (không tạo bản sao lớn) hoặc tích lũy cumsum.

- **Verification Command:** `uv run pytest tests/unit/test_variance.py -v`
- **Commit Message:** `feat(features): implement F2 multi-metric variance filter`

---

### [ ] Task 7: F3+7 — IC Stability, FDR & Term Structure

**Files:**
- Create: `src/quantautoml/features/ic_stability.py`
- Test: `tests/unit/test_ic_stability.py`

**Interfaces:**
- Consumes: `calc_rank_ic`, `calc_rolling_ic`, `calc_icir`, `apply_fdr_correction`, `FitContext.labels_aux`, `FeaturesConfig.ic_horizons / target_horizon`.
- Produces: `class ICStabilityFilter(BaseFeatureFilter)`; sau `fit`: `icir_by_horizon_: dict[str, dict[int, float]]`, `p_values_: dict[str, float]` (p-value của Rank IC tại `h_target`). Một feature được giữ khi **đồng thời**: (1) `ICIR(h_target) > icir_min` (theo giá trị tuyệt đối; dấu IC phải nhất quán qua mọi horizon), (2) qua BH-FDR trên toàn bộ feature thử nghiệm, (3) đơn điệu kỳ hạn: `|ICIR(h_i)| ≥ |ICIR(h_{i+1})| − monotonic_tolerance` với các horizon sắp tăng dần.

**Action Steps:**
1. Test thất bại (dữ liệu tổng hợp seed cố định): feature có tín hiệu thật suy giảm mượt theo horizon → được giữ; feature nhiễu thuần (500 cột ngẫu nhiên) → **tỷ lệ giữ ≤ 5%** (kiểm chứng RULE-08 / TRADE-OFF §4); feature có IC `h=1` âm, `h=5` dương mạnh, `h=10` sụp đổ → bị loại vì vi phạm đơn điệu/đổi dấu; feature đạt ICIR nhưng p-value không qua BH → bị loại; thiếu cột nhãn của một horizon trong `labels_aux` → `InvalidDataError` nêu tên cột; Train < `2 × rolling_window` → `InvalidDataError`; nhãn aux chứa NaN đuôi không làm lệch IC (mask cặp NaN); `transform` chỉ chọn cột, không đọc `labels_aux`.
2. Hiện thực: với mỗi horizon, `calc_rolling_ic` → `calc_icir`; p-value từ `calc_rank_ic` trên toàn Train tại `h_target`; gom p-value của mọi feature rồi gọi `apply_fdr_correction` một lần.

- **Verification Command:** `uv run pytest tests/unit/test_ic_stability.py -v`
- **Commit Message:** `feat(features): implement F3+7 IC stability with FDR and term-structure gate`

---

### [ ] Task 8: F4 — Fast Nonlinear Collinearity (dCor)

**Files:**
- Create: `src/quantautoml/features/collinearity.py`
- Test: `tests/unit/test_collinearity.py`

**Interfaces:**
- Produces: `class FastNonlinearCollinearityFilter(BaseFeatureFilter)`; hàm công khai `fast_dcor(x: NDArray[np.float32], y: NDArray[np.float32]) -> float` (bỏ cặp NaN, cast `float64`, gọi `dcor.distance_correlation(x, y, method="AVL")`; < 3 cặp hoặc hằng → `0.0`). Cặp có `dCor > dcor_threshold` → loại feature có `|Rank IC(h_target)|` thấp hơn (hòa → giữ cột đứng trước).

**Action Steps:**
1. Test thất bại: `fast_dcor(x, x**2)` > 0.3 trong khi Pearson ≈ 0 (bắt được phụ thuộc phi tuyến); `fast_dcor(x, x) ≈ 1.0` (`atol=1e-6`); hai nhiễu độc lập < 0.1 (N = 5,000); cột hằng → `0.0`; không có cảnh báo `UserWarning` của `dcor` (test chạy với `pytest.warns` ngược/`-W error::UserWarning`, chứng minh đã cast `float64`); cặp `Y = X²` → giữ cột có IC cao hơn; **scalability:** N = 20,000 × 30 feature hoàn thành dưới 60 s và đỉnh bộ nhớ (`tracemalloc`) < 300 MB, chứng minh không có ma trận N×N; thứ tự loại xác định (chạy 2 lần cho cùng kết quả).
2. Hiện thực: sắp feature theo `|IC|` giảm dần, duyệt tham lam, chỉ so mỗi feature với các feature đã được giữ (O(F·K) cặp).

- **Verification Command:** `uv run pytest tests/unit/test_collinearity.py -v`
- **Commit Message:** `feat(features): implement F4 fast distance-correlation collinearity filter`

---

### [ ] Task 9: F5 — IC-Behavior Hierarchical Clustering

**Files:**
- Create: `src/quantautoml/features/clustering.py`
- Test: `tests/unit/test_clustering.py`

**Interfaces:**
- Produces: `class ICBehaviorClusterFilter(BaseFeatureFilter)`; sau `fit`: `clusters_: dict[int, tuple[str, ...]]`, `representatives_: dict[int, str]`. Ma trận đặc trưng = chuỗi rolling IC (w = `rolling_window`, theo `h_target`) của mỗi feature; khoảng cách `1 − |corr(IC_i, IC_j)|`; `scipy.cluster.hierarchy.linkage(method="average")` + `fcluster(criterion="distance", t=cluster_distance)`; đại diện = feature có MI cao nhất với `y` (hòa → `|Rank IC|` cao hơn).

**Action Steps:**
1. Test thất bại: 2 feature có chuỗi IC cùng dạng (một là bản nhân đôi có nhiễu nhỏ) → cùng cụm, chỉ 1 đại diện sống; 2 feature IC độc lập → khác cụm, đều giữ; 1 feature duy nhất → pass-through; chuỗi IC hằng (std = 0) → khoảng cách = 1 tới mọi feature, không sinh NaN/`ValueError` trong `linkage`; đại diện đúng là Max-MI; kết quả xác định giữa các lần chạy.
2. Hiện thực; kiểm tra `np.isfinite` ma trận khoảng cách trước `squareform`.

- **Verification Command:** `uv run pytest tests/unit/test_clustering.py -v`
- **Commit Message:** `feat(features): implement F5 IC-behavior hierarchical clustering`

---

### [ ] Task 10: F6 — Regime-Aware MI & Synergy

**Files:**
- Create: `src/quantautoml/features/regime_synergy.py`
- Test: `tests/unit/test_regime_synergy.py`

**Interfaces:**
- Produces:
  - `def label_regimes(ret_h1: NDArray[np.float32], window: int = 63, band: float = 0.5) -> NDArray[np.int8]` → `{1: Bull, -1: Bear, 0: Sideways}`; regime tại t chỉ dùng `ret_h1[:t]` đã realized (dịch 1 nến), ngưỡng `±band·σ` tính trên Train.
  - `class RegimeSynergyFilter(BaseFeatureFilter)` với `mi_by_regime_: dict[str, dict[int, float]]`, `synergy_pairs_: tuple[tuple[str, str], ...]`. Giữ feature nếu `max_regime MI ≥ mi_floor` **hoặc** thuộc cặp synergy: `I(Xi,Xj;Y) − I(Xi;Y) − I(Xj;Y) > synergy_delta` (rời rạc hóa 5 quantile bin, chỉ xét top-30 feature theo MI để giới hạn O(K²)).

**Action Steps:**
1. Test thất bại: `label_regimes(ret)[t]` **không đổi** khi sửa `ret[t:]` (kiểm chứng nhân quả — Review Focus #5); feature chỉ có tín hiệu trong Bear (nhiễu ở các regime khác) vẫn được giữ; cặp XOR (`y = sign(x1·x2)`, mỗi cột đơn lẻ MI ≈ 0) → cả hai được bảo vệ qua `synergy_pairs_`; feature nhiễu thuần bị loại; regime có < 50 mẫu bị bỏ qua (không tính MI) và ghi vào `report_.notes`; thiếu cột nhãn h=1 → filter pass-through + ghi chú, không crash.
2. Hiện thực; MI rời rạc bằng `sklearn.metrics.mutual_info_score` trên nhãn bin kết hợp.

- **Verification Command:** `uv run pytest tests/unit/test_regime_synergy.py -v`
- **Commit Message:** `feat(features): implement F6 regime-aware MI and synergy gate`

---

### [ ] Task 11: F8 — Joint Mutual Information Selector

**Files:**
- Create: `src/quantautoml/features/jmi.py`
- Test: `tests/unit/test_jmi.py`

**Interfaces:**
- Produces: `class JMIFeatureSelector(BaseFeatureFilter)`; sau `fit`: `selected_order_: tuple[str, ...]`. Chọn tham lam: feature đầu = MI cao nhất; các feature tiếp theo tối đa hóa $J(f) = \sum_{s \in S} I(f, s; Y)$ (rời rạc hóa 10 quantile bin), dừng khi đạt `min(max_features, n_features)`.

**Action Steps:**
1. Test thất bại: `max_features ≥ n_features` → giữ tất cả; feature thứ nhất luôn là Max-MI; bản sao (gần như trùng) của feature đã chọn được xếp sau một feature bổ sung thông tin mới (JMI phạt dư thừa); cặp XOR được chọn cùng nhau khi `max_features=2`; thứ tự `selected_order_` xác định; `transform` trả cột theo đúng thứ tự gốc của dữ liệu đầu vào (không đổi thứ tự cột).
2. Hiện thực; cache nhãn bin theo cột để tránh tính lại.

- **Verification Command:** `uv run pytest tests/unit/test_jmi.py -v`
- **Commit Message:** `feat(features): implement F8 JMI feature selector`

---

### [ ] Task 12: FeatureFunnel, Pipeline Wiring & Integration Tests

**Files:**
- Create: `src/quantautoml/features/funnel.py`, `tests/integration/test_zero_leakage_funnel.py`
- Modify: `src/quantautoml/pipeline.py`, `src/quantautoml/features/__init__.py` (export), `tests/unit/test_pipeline.py`
- Test: `tests/unit/test_funnel.py`

**Interfaces:**
- Consumes: 7 filter ở Task 5–11, `SplitResult`, `Dataset`, `FeaturesConfig`.
- Produces: `class FeatureFunnel` với `__init__(self, cfg: FeaturesConfig)`; `fit(self, split: SplitResult, dataset: Dataset) -> Self` (**chỉ đọc `split.train`**); `transform(self, seg: Segment) -> Segment` (trả `Segment` mới với `X` đã lọc); thuộc tính `selected_features_: tuple[str, ...]`, `reports_: tuple[FilterReport, ...]`; `PipelineResult` thêm `funnel: FeatureFunnel | None`; `run_pipeline` ghi thêm `funnel_report.json` (và `selected_features.json`) chỉ khi `cfg.features is not None`.

**Action Steps:**
1. Test thất bại — `tests/unit/test_funnel.py`: thứ tự tầng đúng F1→F2→F3+7→F4→F5→F6→F8; số cột giảm đơn điệu qua các tầng (`n_out` của tầng k = `n_in` của tầng k+1, tính cả indicator do F1 sinh); báo cáo ghi đủ 7 tầng kèm lý do; một tầng loại hết feature → các tầng sau pass-through, `selected_features_ == ()`, không ném lỗi (Review Focus #4); `fit` với `UNSUPERVISED` → `InvalidDataError`; `transform` trên Val/OOS/Cal-A/Cal-B trả cùng danh sách cột, `float32`.
2. Test thất bại — `tests/integration/test_zero_leakage_funnel.py` (dùng dữ liệu 15,000 dòng từ fixture Phase 1): (a) **Hành vi:** thay toàn bộ `Val`, `OOS`, `Cal-A`, `Cal-B` bằng nhiễu → `selected_features_` **giống hệt** (Review Focus #5); (b) **Spy:** bọc `fit` của từng filter, khẳng định độ dài `X` truyền vào bằng `len(split.train.X)` (RULE-01); (c) `FeatureFunnel.fit` ném `DataLeakageError` nếu `split.train.index` không đơn điệu hoặc giao với `index` của Val/OOS/Cal; (d) **BH end-to-end:** 300 feature nhiễu + 5 tín hiệu thật → số feature nhiễu sống sót ≤ 5% và ≥ 4/5 tín hiệu thật được giữ; (e) **Scalability:** N = 20,000, 60 feature chạy trọn funnel < 120 s.
3. Hiện thực `FeatureFunnel` (khởi tạo filter từ `cfg`, truyền `FitContext` dựng từ `dataset.labels_aux` **cắt theo `split.train.index`**, không dùng nhãn ngoài Train); nối vào `run_pipeline` + `test_pipeline.py` kiểm tra file báo cáo.
4. Chạy toàn bộ `uv run pytest tests/ -v` PASS 100%; kiểm tra RULE-06 (mọi mảng `float32`), cập nhật checkbox, bàn giao sang Phase 3.

- **Verification Command:** `uv run pytest tests/ -v`
- **Commit Message:** `feat(features): implement FeatureFunnel and verify zero leakage end-to-end`

---

## Final Verification Checklist
- [ ] `uv run pytest tests/` đạt 100% PASS (gồm Phase 1 không hồi quy).
- [ ] Mọi `fit()` chỉ nhận dữ liệu Train (spy test PASS).
- [ ] BH-FDR: tỷ lệ nhiễu sống sót ≤ 5% (Task 3, 7, 12).
- [ ] dCor N = 20,000 chạy trong giới hạn thời gian/bộ nhớ, không ma trận N×N (Task 8, 12).
- [ ] Cấu trúc file đúng [`STRUCTURE.md`](STRUCTURE.md); không import ngược tầng.
- [ ] Sẵn sàng tích hợp sang Phase 3 (GBDT Tournament): `FeatureFunnel.transform` là đầu vào của `ModelTournament`.
