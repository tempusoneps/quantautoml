# Project Rules & Invariants — Quant AutoML Pipeline

Tài liệu này tập hợp các quy tắc bất biến (Invariants / Hard Gates) bắt buộc mọi lập trình viên và AI Agent phải tuân thủ tuyệt đối khi phát triển mã nguồn trong dự án **Quant AutoML Pipeline**.

---

## 🚨 BẢNG QUY TẮC CỐT LÕI (CORE HARD GATES)

| Mã Quy Tắc | Tên Quy Tắc | Bản Chất Kỹ Thuật / Ngưỡng Ràng Buộc | Chế Tài / Hậu Quả Vi Phạm |
| :--- | :--- | :--- | :--- |
| **RULE-01** | **Zero Look-Ahead Leakage** | `fit()` chỉ được gọi trên tập Train. Cấm xáo trộn dữ liệu (`shuffle=False`). Áp dụng Purging & Embargo $h$ nến giữa các ranh giới thời gian. | Pipeline lập tức dừng, ném `DataLeakageError`. Code không được merge. |
| **RULE-02** | **Rank Metric Consistency** | Mọi đánh giá tối ưu xuyên suốt các giai đoạn (Feature Selection, Tournament, Optuna, Ensemble) phải căn cứ vào **Spearman Rank IC & IC Sharpe**. | Không chấp nhận đánh giá bằng MSE hay Raw Accuracy đơn thuần. |
| **RULE-03** | **Rank Invariance trong Calibration** | Hiệu chuẩn xác suất phải dùng **Temperature Scaling** ($T > 0$) hoặc Platt Scaling. Tuyệt đối **CẤM dùng Isotonic Regression** vì gây hiện tượng bậc thang làm mất độ phân giải thứ hạng (ties). | Code vi phạm sẽ bị từ chối; test kiểm tra tính đơn điệu thứ hạng sẽ fail. |
| **RULE-04** | **Tính Khả Vi $C^2$ trong SLSQP** | Thuật toán `SLSQP` trên Cal-B bắt buộc nhận hàm mục tiêu là **Pearson Correlation trên dữ liệu đã Rank-Transform** ($\tilde{Y} \in [0, 1]$). Cấm truyền trực tiếp hàm Spearman Rank IC vào SLSQP. | Tránh bế tắc đạo hàm (Zero Gradient) và lỗi không hội tụ của solver. |
| **RULE-05** | **Earn Its Place (Hurdle Ensemble)** | Ensemble chỉ được kích hoạt khi vượt qua Single Best Model và baseline Equal Weights ($1/K$) ít nhất $5\%$ IC Sharpe. Ngược lại, tự động fallback về Single Best Model hoặc $W = [1/K, \dots, 1/K]$. | Ngăn chặn hiện tượng lạm phát độ phức tạp vô ích. |
| **RULE-06** | **Numerical Precision Contract** | Mọi mảng số học dùng chuẩn `np.float32`. Sai số giữa Python gốc và ONNX Runtime phải thỏa mãn: $\max \|\hat{y}_{py} - \hat{y}_{onnx}\| < 10^{-4}$ trên tập Validation. | Ném lỗi `PrecisionValidationError` và chặn xuất xưởng model bundle. |
| **RULE-07** | **Decoupled Inference Engine** | Script suy luận `predictor.py` xuất ra trong Model Bundle **không được phép `import quantautoml`**, PyTorch, LightGBM, Scikit-Learn hay Optuna. Chỉ dùng `numpy` và `onnxruntime`. | Đảm bảo tính siêu nhẹ, độc lập nền tảng và độ trễ suy luận $< 5.0\text{ ms/row}$. |
| **RULE-08** | **Kiểm Soát Đa Giả Thuyết (FDR)** | Bộ lọc F3+7 bắt buộc kiểm soát **Benjamini-Hochberg (FDR $\le 0.05$)** và kiểm tra tính đơn điệu suy giảm của IC qua các kỳ hạn $\{1, 5, 10, 21\}$. | Ngăn chặn hiện tượng Data Snooping Bias / False Discovery. |
| **RULE-09** | **Nghiệm Thu Sau Chi Phí Ma Sát** | Nghiệm thu OOS (Giai đoạn 6) bắt buộc tính toán **Net Sharpe sau khi trừ chi phí giao dịch (Fees + Slippage)** và tích hợp dải đệm trễ **Hysteresis Buffer Bands** để kiểm soát vòng quay danh mục. | Loại bỏ các chiến lược "lãi ảo trên giấy" nhưng lỗ nặng trong thực tế vì phí sàn. |
| **RULE-10** | **Duy Trì Trạng Thái (State Continuity)** | Mọi Agent khi bắt đầu phiên bắt buộc phải kiểm tra `git branch`, `PHASES.md` và `PLAN.md`. Khi hoàn thành task, BẮT BUỘC cập nhật `- [x]` vào `docs/plan/PLAN.md`. | Ngăn ngừa đứt gãy ngữ cảnh giữa các phiên làm việc; đảm bảo agent sau luôn tiếp nối chính xác agent trước. |

---

## 2. Tiêu Chuẩn Lập Trình & Chất Lượng Mã Nguồn (Code Quality Standards)

1. **Python $\ge 3.12$ & Static Typing:**
   - 100% các hàm, phương thức và lớp phải có Type Hints đầy đủ.
   - Sử dụng `Pydantic` v2 hoặc `@dataclass` cho các data contract / cấu hình.
2. **Xử lý ngoại lệ minh bạch:**
   - Sử dụng các lớp ngoại lệ chuyên biệt kế thừa từ `quantautoml.common.exceptions.QuantAutoMLError`.
   - Không dùng `try ... except Exception: pass` nuốt lỗi.
3. **Tuân thủ ranh giới thư mục:**
   - Tuân thủ nghiêm ngặt bảng phân chia trách nhiệm trong [`docs/STRUCTURE.md`](file:///mnt/Shares/GIT/the-new-algo/quantautoml/docs/STRUCTURE.md).
   - Module tầng dưới không được phép phụ thuộc ngược (reverse dependency) lên module tầng trên.
