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

## Điểm implementation C cần nhóm rà, chưa phải quyết định khoa học đã chốt

- Paper §3.5 gọi tín hiệu ADWIN là *warning* nhưng không cho ngưỡng warning riêng. River ADWIN 0.26.1 chỉ có tín hiệu `drift_detected`; code C dùng tín hiệu này khi ước lượng Hamming loss tăng. `adwin_delta=0.002` là mặc định River, đã ghi tường minh trong config, không phải giá trị Table 5 của paper.
- Algorithm 1 ghi cập nhật detector cây nền bằng `Y_i, Z_i` với `Z_i` là dự đoán chung, trong khi §3.5 so lỗi riêng `e` và `e'` của hai cây. Code C dùng dự đoán trước cập nhật của từng nhánh để so lỗi; cần A/C rà lại cách hiểu này bằng ví dụ nhỏ trước run chính thức.
- Algorithm 3 yêu cầu cân dự đoán theo lỗi nhưng không cho công thức trọng số cụ thể. Code C dùng `1 − e` rồi chuẩn hóa, và chỉ đưa cây nền vào khi số mẫu có trọng số vượt `kappa_alt`. Đây là lựa chọn triển khai cần A/B/C xác nhận trước khi diễn giải số liệu so với paper.
- §3.5 có điều kiện so lỗi ghi `delta_alt`, còn Eq. (16) và Algorithm 1 dùng `epsilon_alt`. Code C theo Eq. (16) và Algorithm 1 như ghi trong [paper-notes.md](paper-notes.md).
