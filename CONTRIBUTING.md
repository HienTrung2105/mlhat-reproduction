# Cách làm việc nhóm

> Mục tiêu: commit thật, lịch sử dễ xem, tránh 3 người cùng sửa một file.

## Branch gợi ý

- `main`: phiên bản ổn định nhất nhóm chấp nhận.
- `feat/a-tree-*`: A triển khai cây và splitting.
- `feat/b-evaluation-*`: B triển khai lá, data/evaluation/W&B.
- `feat/c-drift-*`: C triển khai drift, tích hợp thí nghiệm.
- `docs/*`: chỉnh tài liệu, slide plan, biểu đồ.

1. `git checkout -b feat/a-tree-node` rồi chỉnh module phụ trách.
2. Chạy `python -m pytest -q` và kiểm tra file có chứa secrets không.
3. Commit gọn, có ý nghĩa; push branch và mở Pull Request lên `main`.
4. **Ít nhất một thành viên khác review**, kiểm tra logic + kết quả test trước khi merge.
5. Không force-push lên `main`, không đổi interface chung mà không báo A/B/C.

## Hợp đồng tích hợp

Xem `src/mlhat/interfaces.py` cho phương thức `predict_one`/`learn_one` và `docs/integration-contracts.md` cho cấu trúc dữ liệu và việc cần thống nhất. Interface là khung đề xuất, được phép sửa sau khi cả nhóm thống nhất từ paper.

## Những thứ KHÔNG được commit

API key, `.env`, dataset tải về, log W&B local, file model nặng, kết quả chưa chạy, nội dung sao chép từ code/hình gốc paper.
