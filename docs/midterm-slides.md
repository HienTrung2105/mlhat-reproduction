# Kịch bản Midterm — 10.pptx

Kế hoạch 30 slide. Giới hạn 20–40 slide tính cả phụ lục và dưới 20 MB theo hướng dẫn dự án trong repo. Mỗi slide có speaker notes tối đa 500 từ. Trích dẫn IEEE ở chân slide liên quan. Dùng hình do nhóm vẽ và đặt số liệu run cạnh nguồn của chúng.

## B — Vòng Bảo Thắng

- **Slide 1** — Tên paper, nhóm 10 và thành viên.
- **Slide 2** — Ví dụ một mẫu có nhiều nhãn.
- **Slide 3** — Luồng dữ liệu online và thách thức về drift.
- **Slide 4** — Hoeffding Tree và thống kê tích lũy.
- **Slide 5** — HAT và Binary Relevance.
- **Slide 6** — MLHT và khoảng trống paper đặt ra.
- **Slide 14** — Sơ đồ ba chế độ dự đoán ở lá.
- **Slide 15** — So sánh kNN + LP với bagging LR + BR.
- **Slide 21** — Ba câu hỏi nghiên cứu và dữ liệu tác giả sử dụng.
- **Slide 22** — Một bước test-then-train và ví dụ cách tính metric.

## A — Phạm Thái Minh

- **Slide 7** — Khoảng trống của các phương pháp trước.
- **Slide 8** — Ba đóng góp của MLHAT.
- **Slide 9** — Ký hiệu x, y, W và W_l.
- **Slide 10** — Sơ đồ predict → metric → learn.
- **Slide 11** — Cây từ lá gốc đến nút chia và hai lá con.
- **Slide 12** — Entropy từng nhãn và gain qua ví dụ tính tay.
- **Slide 13** — Ước lượng ứng viên và kiểm tra Hoeffding.
- **Slide 18** — Thống kê online và trọng số Poisson.
- **Slide 19** — Giả định và giới hạn của việc diễn giải Eq. (5) và Eq. (7).
- **Slide 20** — Chi phí theo đường đi, phép chia, bộ dự đoán và cây nền.

## C — Đàng Trung Hiện

- **Slide 16** — Cửa sổ ADWIN và cảnh báo theo Hamming loss.
- **Slide 17** — Vòng đời nhánh nền, điều kiện thay/bỏ và phối hợp dự đoán.
- **Slide 23** — Sơ đồ hai thí nghiệm chính và một phân tích bổ sung.
- **Slide 24** — Phiên bản dữ liệu, seed, tham số và cách đối chiếu paper.
- **Slide 25** — EXP-01, cấu hình Yeast và BR+HAT.
- **Slide 26** — EXP-01, bảng kết quả và diễn giải khác biệt.
- **Slide 27** — EXP-02, cấu hình và mốc sudden/gradual.
- **Slide 28** — EXP-02, đường cong F1 và sự kiện drift.
- **Slide 29** — EXTRA-01, độ nhạy kappa_alt. A cung cấp số liệu và diễn giải.
- **Slide 30** — Kết luận theo câu hỏi nghiên cứu và giới hạn của phạm vi Core.

## Kiểm tra trước khi nộp

B ghép deck và kiểm tra ghi chú. A đối chiếu phương trình và ký hiệu. C kiểm tra mỗi số liệu theo run registry và public W&B. Với kết quả chưa chạy, giữ mục tương ứng ở bản nháp và điền từ run thật trước khi chốt file. Khi bảng paper chỉ có số trung bình toàn benchmark, ghi rõ phạm vi thay vì tính chênh trực tiếp với Yeast. Dùng [paper v1](https://arxiv.org/html/2410.20242v1) làm nguồn gốc cho phần đọc hiểu.
