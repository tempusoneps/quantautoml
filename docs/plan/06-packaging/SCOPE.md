# Step 06 — Scope: ONNX Model Bundle Packaging

> Thuộc [Phase 5](../PHASES.md). Phần nghiệm thu backtest nằm ở [`../07-backtest/`](../07-backtest/SCOPE.md).

## In-Scope (Must Have)

- [ ] Đóng gói dạng Model Bundle: `preprocessor.onnx`, `model_*.onnx`, `ensemble_meta.json`, `features_schema.json`, và `predictor.py`.
- [ ] Kiểm tra sai số số học bắt buộc: $\max |\hat{y}_{py} - \hat{y}_{onnx}| < 10^{-4}$ trên chuẩn `float32`.
- [ ] Cảnh báo phân phối dịch chuyển (`DistributionShiftWarning` khi $z\text{-score} > 3.0$).

## Ràng buộc liên quan

- RULE-06 (Numerical Precision Contract), RULE-07 (Decoupled Inference Engine: `predictor.py` chỉ dùng `numpy` và `onnxruntime`).
- Độ trễ suy luận $< 5\text{ ms/row}$.
