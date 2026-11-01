# Git Conventions — Quant AutoML Pipeline

Tài liệu này quy định các tiêu chuẩn làm việc với Git, cấu trúc commit message, quy ước đặt tên nhánh và vệ sinh repository cho dự án **Quant AutoML Pipeline**.

---

## 1. Cấu Trúc Nhánh (Branching Strategy)

Dự án duy trì cấu trúc nhánh tinh gọn theo các phiên bản phát hành chính:

### 1.1. Các Nhánh Chính Của Dự Án (Core Branches)
- **`master`**: Nhánh sản xuất (production-ready). Chứa mã nguồn ổn định nhất của các phiên bản đã nghiệm thu và phát hành chính thức.
- **`develop/v1.0`**: Nhánh phát triển chính cho phiên bản **v1.0** (Quant AutoML Pipeline cốt lõi hiện tại). Mọi công việc triển khai các giai đoạn/task của v1.0 được tích hợp trực tiếp vào nhánh này.
- **`develop/v1.1`**: Nhánh phát triển dành cho phiên bản tiếp theo **v1.1** (các tính năng mở rộng, nâng cấp thuật toán và tối ưu hóa hạ tầng sau v1.0).

### 1.2. Nhánh Dành Riêng Cho Contributor (Contributor Branches Only)
Các nhà đóng góp bên ngoài (contributors) khi thực hiện pull request bắt buộc phải phân nhánh từ nhánh develop tương ứng theo quy ước:
- **`feature/<phase>-<task>`**: Dành cho contributor phát triển tính năng hoặc sub-system mới (Ví dụ: `feature/p1-ingestion`, `feature/p2-fast-dcor`).
- **`fix/<issue-name>`**: Dành cho contributor sửa lỗi hoặc khắc phục bug.
- **`refactor/<component-name>`**: Dành cho contributor tái cấu trúc module mà không đổi logic.

---

## 2. Chuẩn Commit Message (Conventional Commits)

Định dạng bắt buộc:
```text
<type>(<scope>): <mô tả ngắn gọn bằng thể mệnh lệnh (imperative)>

[Phần thân (body) mô tả chi tiết lý do và sự thay đổi - Tùy chọn]

[Footer tham chiếu issue hoặc PR - Tùy chọn]
```

### 2.1. Danh mục Types:
- **`feat`**: Tính năng mới hoặc sub-system mới được hiện thực.
- **`fix`**: Sửa lỗi logic, sửa bug leakage, sửa sai số số học.
- **`test`**: Bổ sung hoặc chỉnh sửa unit test, integration test, fixtures.
- **`refactor`**: Tái cấu trúc code (tách module, tối ưu hóa class) mà không thay đổi chức năng.
- **`perf`**: Cải tiến hiệu năng tính toán (ví dụ: chuyển từ dCor $\mathcal{O}(N^2)$ sang Fast dCor $\mathcal{O}(N \log N)$).
- **`docs`**: Cập nhật tài liệu kỹ thuật, kế hoạch, guide hoặc docstring.
- **`chore`**: Cập nhật dependencies, cấu hình CI, cấu hình linters/formatters.

### 2.2. Danh mục Scopes chuẩn của dự án:
- `(data)`: Ingestion, task inference, temporal splitter, purging & embargo.
- `(features)`: Các bộ lọc tính năng F1 đến F8, Feature Funnel.
- `(metrics)`: Rank IC, ICIR, IC Sharpe, Benjamini-Hochberg FDR, ECE.
- `(models)`: Tournament, Decision Engine, Modern GBDT Trio, Optuna Tuner.
- `(ensemble)`: SLSQP Optimizer, Bootstrap Shrinkage, Temperature Calibration.
- `(packaging)`: Export ONNX Bundle, schema data contract, precision validator.
- `(backtest)`: Blind OOS Runner, friction cost model, hysteresis buffer bands.
- `(common)`: Shared types, exceptions, base classes.
- `(setup)`: `pyproject.toml`, Git config, linters.

### 2.3. Ví dụ chuẩn:
```text
feat(data): implement AdaptiveTimeSeriesSplitter with purging and embargo
test(data): add tests for OOF split mode when sample size < 5000
fix(ensemble): use rank-transformed pearson in SLSQP to fix zero-gradient issue
perf(features): optimize distance correlation using fast dcor algorithm
docs(plan): update TRADE-OFF.md with mathematical justification for temperature scaling
```

---

## 3. Tần Suất Commit & Quy Tắc Vệ Sinh (Git Hygiene)

1. **Commit theo từng chu kỳ TDD nhỏ:**
   - Commit ngay sau khi hoàn thành một task có kiểm thử PASS hoàn chỉnh (Atomic Commits).
   - Không dồn nhiều ngày công việc hay nhiều sub-system độc lập vào một commit khổng lồ.
2. **Những thứ TUYỆT ĐỐI KHÔNG commit:**
   - Không commit file dữ liệu lớn (`*.parquet`, `*.csv` trong `datasets/`).
   - Không commit các model bundle lớn đã train (`*.onnx` dung lượng lớn; chỉ chấp nhận mock fixture giả lập $< 50\text{ KB}$ trong `tests/fixtures/`).
   - Không commit cache của Python/IDE (`__pycache__/`, `.pytest_cache/`, `.mypy_cache/`, `.idea/`, `.vscode/`).
   - Không commit API keys, credentials bí mật.
