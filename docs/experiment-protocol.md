# Giao thức tái tạo — nhóm 10

Giao thức của hai thí nghiệm chính và phân tích bổ sung trong phạm vi Core của Midterm.

## EXP-01 — So sánh trên dữ liệu thật

B chủ trì, A kiểm tra cây và C đối chiếu baseline. Chạy MLHAT nhóm và BR+HAT trên **Yeast** (2.417 mẫu, 103 đặc trưng, 14 nhãn theo Table 3). BR+HAT dùng một Hoeffding Adaptive Tree của River cho mỗi nhãn. Cố định phiên bản River và ghi toàn bộ tham số của baseline. Tham chiếu §4.3–4.4, Tables 6–8 và Figure 5. Đây là so sánh hai mô hình hoàn chỉnh. Mức chênh không thể quy hết cho split vì cấu trúc và bộ dự đoán cũng khác nhau.

Đã hoàn thành 6 run, gồm hai mô hình × ba seed. W&B ghi commit `344cf97`, giao thức `test_then_train` và 2.417 mẫu mỗi run. Bảng F1 và link từng seed nằm trong [README](../README.md#exp-01--kết-quả-yeast).

## EXP-02 — Drift đột ngột và gradual

C chủ trì với B xác thực dữ liệu và A rà thao tác thay cây. MLHAT, BR+HAT và MLHAT tắt drift chạy trên SynHPSud/SynHPGrad: 50.000 mẫu/stream, 30 feature số, 8 nhãn; width lần lượt 1 và 500. Predict → metric → learn, thứ tự gốc, seed 1/2/3; mốc drift một-based 12.500/25.000/37.500. Theo dõi cảnh báo/mở, promote và discard; ablation giữ các thành phần còn lại.

Curve nhóm: cửa sổ 1.000 mẫu/điểm mỗi 250 mẫu; bổ sung fading alpha=0.995 theo từng mẫu/điểm mỗi 50 mẫu. Chưa xác minh phép tổng hợp fading trùng evaluator tác giả, không lấy hai loại metric khác nhau để tính relative deviation. Tham chiếu §4.5, Table 4 và Figure 6(e–f) của paper.

Đã hoàn tất 18 run và kiểm tra W&B Public. [Báo cáo EXP-02 chính](exp2-results.md) giữ toàn bộ nguồn HF/commit/checksum, config, kết quả mean ± SD, 18 URL, kiểm chứng public, lệnh tái lập/upload và giới hạn khoa học. Các file gốc và snapshot được bảo toàn; báo cáo dẫn xuất mới ghi ra `artifacts/exp2_reports/`, không ghi đè batch `artifacts/exp2_hyperplane/`.

## Phân tích bổ sung — Ablation bật/tắt drift

So MLHAT bật drift và MLHAT tắt drift trên SynHPSud và SynHPGrad. Ghép theo stream và seed, giữ nguyên split, leaf, bootstrap và các tham số chung. Phân tích dùng 12 run có sẵn trong EXP-02, tức sáu cặp. Báo Example F1 toàn stream và quan sát sự kiện mở, thay, bỏ cây dự phòng. Xem [số liệu và diễn giải](exp2-results.md#phân-tích-bổ-sung--ablation-bậttắt-drift).

Khảo sát độ nhạy với `kappa_alt = 100, 200, 500` là hướng nghiên cứu tiếp theo. Các run MLHAT bật drift trong EXP-02 dùng `kappa_alt = 200`.

## Chuẩn chung cho các lần chạy

1. Đối chiếu Yeast với [UCO MLL Resources](https://www.uco.es/kdis/mllresources/) và hai stream Hyperplane với thư mục synthdata của [repo tác giả](https://github.com/aestebant/mlhat). B/C ghim HF/Kaggle ID, revision, checksum, quyền sử dụng, thứ tự dòng, thứ tự feature và nhãn theo metadata nguồn. Loader tự tải và kiểm tra số dòng, số cột, nhãn 0/1 và giá trị hữu hạn. Chỉ chạy chính thức sau khi có bản HF/Kaggle tương ứng với nguồn tham chiếu. Cache dữ liệu ngoài Git.
2. Mỗi mẫu chạy predict → cập nhật metric → learn. Tất cả mô hình dùng cùng stream, không shuffle và không fit preprocessing trên dữ liệu tương lai. Giữ cả mẫu đầu tiên trong phép đo và ghi quy ước khởi tạo. Nhóm dùng dự đoán toàn 0 khi lá trống.
3. Với mỗi mẫu, F1 = 2TP/(2TP+FP+FN), quy ước bằng 0 khi mẫu số bằng 0. Hamming loss là tỷ lệ vị trí nhãn sai. Subset accuracy bằng 1 nếu toàn bộ vector khớp. Lấy trung bình theo mẫu và báo mean ± SD mẫu qua ba seed 1, 2, 3. Ghi quy ước F1 của nhóm khi đối chiếu số liệu paper.
4. MLHAT dùng cấu hình Table 5 của v1 làm tham chiếu. delta_spl=1e-7, kappa_spl=200, delta_alt=0.05, kappa_alt=200, eta=750, lambda=1. Leaf nhỏ là kNN + Label Powerset. Leaf lớn là bagging Logistic Regression + Binary Relevance. Drift dùng Hamming loss. Paper có thông số khác ở phần văn bản, gồm delta_spl tại §4.1 và kappa_alt/eta tại §4.2. Ghi nguồn từng giá trị trong config và ghi mọi khác biệt của baseline.
5. Mỗi run lưu commit, config, seed, phiên bản môi trường, fingerprint dữ liệu, số mẫu, metrics, thời gian, chuỗi metric và URL W&B công khai vào run registry. Thời gian tính phần predict/learn và đánh giá, tách thời gian tải dữ liệu. Ghi thiết bị và phiên bản thư viện.
6. Bảng kết quả gồm dataset, baseline, metric, paper/reference, nhóm, chênh lệch tuyệt đối và URL run. Chênh tương đối = 100 × (nhóm − tham chiếu)/|tham chiếu| khi cùng giao thức và mẫu số khác 0. Dùng kết quả theo dataset từ artifacts tác giả nếu khớp phiên bản. Nếu paper chỉ có đường cong hoặc trung bình toàn bộ benchmark, trình bày tham chiếu định tính và so sánh các run nhóm trên cùng stream.

## Ghi chú implementation của runner EXP-01

Các giá trị dưới đây mô tả cách code hiện tại triển khai protocol; chúng bổ sung cho tham số paper, không thay thế lựa chọn đã ghi ở trên.

- Với feature số, `numeric_splitter.n_bins=10`: code tạo 10 ngưỡng nội bộ cách đều giữa giá trị nhỏ nhất và lớn nhất đã quan sát tại node.
- Leaf dùng `n_neighbors=5` cho LP-kNN khi chưa đủ `eta=750` mẫu; khi đạt ngưỡng, dùng bagging Logistic Regression với `n_estimators=10` cho mỗi nhãn. Prediction threshold là `0.5`; trọng số online của cây dùng Poisson `lambda=1`.
- Splitter chọn ngưỡng tốt nhất cho từng feature trước khi so hai feature tốt nhất, theo Eq. (9). Số bin bằng 10 là lựa chọn implementation của nhóm.
- Evaluator hiện cộng dồn metric trên toàn bộ stream và trả example-F1, Hamming loss, subset accuracy, cùng micro-F1. Runner không tạo chuỗi metric theo từng mốc và không áp dụng fading/forgetting cho metric.
- Sáu run EXP-01 được báo cáo lấy từ project `mlhat-team10-exp1-drift`. Ba run MLHAT ghi cấu hình bật drift. Các giả định của detector và nhánh nền được ghi trong [decisions.md](decisions.md).

Nguồn phương pháp và đối chiếu. [Paper v1](https://arxiv.org/html/2410.20242v1).
