# Phân công A / B / C — Midterm

**Nguyên tắc:** chia module nghiên cứu + code + slide, không để C làm mọi thí nghiệm, B làm mọi slide, hoặc A chỉ làm toán.

## A — Phạm Thái Minh (24127081) — Core Algorithm & Theory

- [ ] Đọc đầy đủ paper; phụ trách Hoeffding Tree/HAT, MLHAT node statistics, multi-label split criterion, phương pháp và độ phức tạp.
- [ ] Thiết kế thống kê ở node và logic split dựa trên công thức trong paper, có ví dụ tính tay nhỏ.
- [ ] Triển khai `src/mlhat/tree.py`, `src/mlhat/splitting.py`, viết unit tests minh bạch.
- [ ] Cùng C kiểm thử việc kết nối split/tree/drift; cùng B xác minh leaf interface.
- [ ] Làm slides **7–13 và 18–20**, hình tự vẽ, đầy đủ notes và IEEE citations.
- [ ] Chủ trì EXTRA-01 trên SynHPGrad, bàn giao bảng độ nhạy và phần diễn giải cho slide 29.

**Bàn giao:** sơ đồ thuật toán gốc nhóm, tài liệu công thức, source + tests, 10 slide editable/notes.

## B — Vòng Bảo Thắng (24127539) — Multi-label, Data, Evaluation & Slide Lead

- [ ] Đọc paper; phụ trách multi-label target, leaf prediction, datasets, metrics, baseline protocol.
- [ ] Implement `src/mlhat/leaves.py`, `src/mlhat/data.py` và `evaluation.py` theo giao thức paper đã thống nhất.
- [ ] Dataset phải tự tải qua Hugging Face/Kaggle, không copy local thủ công.
- [ ] Phụ trách W&B config/run mapping (W&B project phải public), chủ trì EXP-01 trên Yeast với A/C.
- [ ] Làm slides **1–6, 14–15 và 21–22** (10 slide).
- [ ] **Slide lead:** chuẩn hóa template, ghép `ID.pptx`, kiểm tra 20–40 slide, <20 MB, notes và citations.

**Bàn giao:** pipeline data+metric+W&B, kết quả Exp1 có run thật, 10 slide và PPTX chung.

## C — Đàng Trung Hiện (24127367) — Drift Adaptation & Reproduction Lead

- [ ] Đọc paper; phụ trách concept drift, ADWIN, alternate/background branches và điều kiện thay thế.
- [ ] Implement `src/mlhat/drift.py` dựa trên paper, viết tests cho kịch bản drift.
- [ ] Đồng bộ interface để tích hợp A/B, chủ trì EXP-02 trên SynHPSud và SynHPGrad với A/B.
- [ ] Đối chiếu paper vs nhóm: relative deviation, nguyên nhân do cấu hình/seed/version/logic.
- [ ] Làm slides **16–17, 23–30** (10 slide), kết quả dựa trên W&B runs có thật.

**Bàn giao:** drift component+tests, Exp2 run thật, bảng sai lệch và 10 slide editable/notes.

## Quy định review chéo

- [ ] Mọi PR có ít nhất 1 reviewer khác vai trò chính.
- [ ] A review lý thuyết/công thức, B review số liệu/metrics/W&B, C review tái tạo/drift.
- [ ] Cả nhóm đọc và xác nhận file PPTX cuối cùng.
- [ ] Không đánh dấu hoàn thành thực nghiệm nếu chưa có W&B public run và lệnh tái lập.
