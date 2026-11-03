# Quyết định dự án MLHAT — nhóm 10

## Thông tin đã xác nhận

Chủ đề MLHAT đã được duyệt theo thông tin của nhóm. Mã nhóm là **10**, tên file Midterm là **10.pptx**.

- A — Phạm Thái Minh (24127081).
- B — Vòng Bảo Thắng (24127539).
- C — Đàng Trung Hiện (24127367).

## Quyết định triển khai

- **D-01 — Core Scope.** Tái tạo một phần hai phép so sánh của paper và thực hiện một phân tích bổ sung.
- **D-02 — EXP-01.** MLHAT và BR+HAT trên Yeast, bám RQ2, đồng thời đối chiếu bối cảnh benchmark RQ1.
- **D-03 — EXP-02.** MLHAT và BR+HAT trên SynHPSud và SynHPGrad, bám RQ3. Thêm biến thể MLHAT tắt drift để giải thích đóng góp của cơ chế này.
- **D-04 — EXTRA-01.** Trên SynHPGrad, khảo sát kappa_alt với 100, 200 và 500. Giá trị 200 lấy từ Table 5 của paper v1.
- **D-05 — Đánh giá.** Test-then-train, giữ thứ tự stream nguồn. Metric chính là example-based F1. Báo thêm Hamming loss và subset accuracy. Seed của nhóm là 1, 2, 3, áp dụng cho ngẫu nhiên của mô hình.
- **D-06 — Tích hợp.** Dùng giao diện trong [integration-contracts.md](integration-contracts.md). A sở hữu cấu trúc cây, B sở hữu lá và đánh giá, C sở hữu quyết định drift.
- **D-07 — Bằng chứng.** Kết quả phải truy về script, cấu hình, revision dữ liệu và W&B run công khai. So sánh số liệu với paper khi cùng dataset và giao thức.

Chi tiết chạy nằm trong [experiment-protocol.md](experiment-protocol.md). Căn cứ phương pháp nằm trong [paper-notes.md](paper-notes.md).

## Điều kiện còn phụ thuộc nguồn bên ngoài

- Đại diện nhóm điền hai hạn Moodle từ thông báo chính thức. Kế hoạch nội bộ dùng mốc tương đối trong [timeline.md](timeline.md).
- B và C ghi ID, revision và giấy phép của bản Hugging Face/Kaggle trước khi chạy. Dataset chuẩn đã được chọn. Nếu nguồn hiện có thiếu quyền phân phối lại, dùng nguồn được môn chấp nhận sau khi có căn cứ về quyền sử dụng.
- Giấy phép mã nhóm cần sự đồng ý của các đồng tác giả trước khi thêm LICENSE. Giấy phép dữ liệu được quản lý riêng.
