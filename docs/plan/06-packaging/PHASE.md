# Step 06 — Phase 5 (phần 1/2): ONNX Model Bundle Packaging

> Trạng thái phase được theo dõi tại [`../PHASES.md`](../PHASES.md). Phần 2/2 (backtest): [`../07-backtest/PHASE.md`](../07-backtest/PHASE.md).

- **Mục tiêu:** Đóng gói toàn bộ pipeline thành gói phân phối độc lập chuẩn ONNX Model Bundle.

## Sub-systems & Deliverables

1. `ONNXBundleExporter`: Xuất `preprocessor.onnx`, `model_*.onnx`, `ensemble_meta.json`, và `features_schema.json` (kiểu dữ liệu `np.float32`).
2. `Predictor`: Module runtime độc lập `predictor.py` (~50 dòng code thuần `numpy` & `onnxruntime`), hỗ trợ kiểm tra `DistributionShiftWarning` ($z\text{-score} > 3.0$).
3. `PrecisionValidator`: Kiểm tra sai số số học bắt buộc $\max |\hat{y}_{py} - \hat{y}_{onnx}| < 10^{-4}$ trên tập Validation.

## Tiêu chí kiểm thử độc lập (Independent Verification)

- Chạy `predictor.py` không cần import bất kỳ module huấn luyện nào.
