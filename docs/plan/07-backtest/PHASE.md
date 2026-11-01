# Step 07 — Phase 5 (phần 2/2): Friction-Aware Blind OOS Backtesting

> Trạng thái phase được theo dõi tại [`../PHASES.md`](../PHASES.md). Phần 1/2: [`../06-packaging/PHASE.md`](../06-packaging/PHASE.md).

- **Mục tiêu:** Tổ chức kiểm thử mù (Blind Test) nghiệm thu hiệu suất thực tế có tính chi phí ma sát và giới hạn vòng quay.

## Sub-systems & Deliverables

1. `BacktestRunner`: Nghiệm thu độc lập trên tập **OOS** (Blind Test) qua `backtesting.py`:
   - Áp dụng **Ma trận phí giao dịch + Trượt giá** thực tế.
   - Cơ chế **Hysteresis Buffer Bands** để kiểm soát vòng quay danh mục.
   - Kiểm tra tiêu chí: Out-of-Sample Rank IC $> 0.035$, ICIR $> 0.40$, **Net Sharpe $> 1.20$**, MDD $< 18\%$, Latency $< 5\text{ ms}$, và Downside IC ổn định trong các kỳ sụt giảm thị trường.

## Tiêu chí kiểm thử độc lập (Independent Verification)

- Báo cáo backtest phản ánh lợi nhuận thực tế sau chi phí giao dịch.
