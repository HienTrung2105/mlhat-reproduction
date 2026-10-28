# Experimental reproduction protocol — TO BE APPROVED

**Trạng thái: chưa chốt Core/Full, chưa xác minh thí nghiệm trung tâm từ Section 4.** Không coi các gợi ý trước đó là thông số chính thức của paper.

Core Scope yêu cầu tối thiểu **2 thí nghiệm quan trọng nhất, gồm phép so sánh chính của paper**, cộng một **phân tích bổ sung do nhóm thiết kế**.

| ID | Câu hỏi nghiên cứu / paper table-figure | Dataset ID + nguồn HF/Kaggle | Label schema | Baseline | Metric / averaging | Seed và số lần chạy | Settings paper / nhóm | Owner | Status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| EXP-01 | **TODO: tìm phép so sánh trung tâm** | TODO | TODO | TODO | TODO | TODO | TODO | B (A,C hỗ trợ) | PLANNED |
| EXP-02 | **TODO: chọn thí nghiệm chính thứ hai** | TODO | TODO | TODO | TODO | TODO | TODO | C (A,B hỗ trợ) | PLANNED |
| EXTRA-01 | **TODO: phân tích bổ sung khi chọn Core** | TODO | TODO | TODO | TODO | TODO | TODO | A (B,C hỗ trợ) | PLANNED |

## Những điều phải kiểm tra

- [ ] Đọc mục Experiments và bảng hình chính bài báo, ghi rõ nguồn trang/section.
- [ ] So đúng metric và hướng tối ưu, phương pháp đánh giá prequential, biến thể và preprocessing.
- [ ] Dataset tải tự động bằng script từ HF/Kaggle (nếu không có public mirror, nhóm phải giải quyết theo hướng dẫn thầy).
- [ ] Paper vs nhóm: cùng config/schema/seed nếu paper cung cấp; giải thích điểm khác.
- [ ] Bản triển khai từ paper phải độc lập, không copy code tác giả.
- [ ] W&B run public với config, metrics, liên kết README và số liệu cuối slide.
- [ ] Relative deviation chỉ tính ở trường hợp số liệu cùng định nghĩa.
