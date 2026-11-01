# Step 07 — Scope: Friction-Aware Blind OOS Backtesting

> Thuộc [Phase 5](../PHASES.md). Phần đóng gói nằm ở [`../06-packaging/`](../06-packaging/SCOPE.md).

## In-Scope (Must Have)

- [ ] Runner kiểm thử độc lập qua `backtesting.py` trên tập OOS phong tỏa:
  - Tích hợp **Ma trận chi phí ma sát (Taker fee + Slippage bps)**.
  - Cơ chế **Hysteresis Buffer Bands** để kiểm soát vòng quay (Turnover Cap).
  - Kiểm tra bổ sung chỉ số **Downside / Stress Period IC** trong các chu kỳ sụt giảm mạnh.

## OOS Acceptance Gate

Chiến lược kiểm thử mù (Blind Test) trên tập OOS đạt: Out-of-Sample Rank IC $> 0.035$, ICIR $> 0.40$, Net Annualized Sharpe $> 1.20$ **(sau khi trừ phí giao dịch & trượt giá với cơ chế kiểm soát vòng quay Turnover)**, MDD $< 18\%$, độ trễ suy luận $< 5\text{ ms/row}$.
