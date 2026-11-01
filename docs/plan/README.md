# docs/plan — Index

Kế hoạch triển khai Quant AutoML Pipeline, tổ chức theo **step** của pipeline.

## Cấu trúc

```
docs/plan/
├── README.md             # Index này
├── REQUIREMENT.md        # Yêu cầu gốc
├── SCOPE.md              # Phạm vi tổng thể, tiêu chí thành công, out-of-scope, ràng buộc
├── PHASES.md             # Roadmap + TRẠNG THÁI từng phase (agent cập nhật tại đây)
├── PLAN.md               # Checklist task của phase đang active (agent cập nhật [x] tại đây)
├── ARCHITECTURE.md       # Kiến trúc tổng thể, end-to-end diagram, tech stack, NFR
├── STRUCTURE.md          # Cây thư mục source và ma trận trách nhiệm
├── TRADE-OFF.md          # Index các quyết định đánh đổi
├── 01-ingestion-split/       (Phase 1)
├── 02-feature-selection/     (Phase 2)
├── 03-model-selection/       (Phase 3, phần 1)
├── 04-tuning/                (Phase 3, phần 2)
├── 05-ensemble-calibration/  (Phase 4)
├── 06-packaging/             (Phase 5, phần 1)
└── 07-backtest/              (Phase 5, phần 2)
```

Mỗi thư mục step có các file:

| File | Nội dung |
| :--- | :--- |
| `SCOPE.md` | In-Scope checklist của step |
| `PHASE.md` | Mục tiêu, deliverables, tiêu chí kiểm thử độc lập (không chứa trạng thái) |
| `DESIGN.md` | Data flow, công thức toán, layout source, trách nhiệm module |
| `TRADE-OFF.md` | Phân tích đánh đổi (chỉ có ở step có quyết định kiến trúc; step 04 và 06 không có) |

## Quy ước

- **Trạng thái phase** chỉ cập nhật trong `PHASES.md`.
- **Checkbox task** chỉ cập nhật trong `PLAN.md` (RULE-10). Khi sang phase mới, nạp task của phase đó vào `PLAN.md`.
- Nội dung chi tiết của một step chỉ nằm trong thư mục step đó, không lặp lại ở file gốc.
