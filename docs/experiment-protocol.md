# Giao thức tái tạo — nhóm 10

Phạm vi Core. Ba mục dưới đây là kế hoạch chạy. Bản thảo slide và README lấy số liệu từ run registry sau khi chạy.

## EXP-01 — So sánh trên dữ liệu thật

B chủ trì, A kiểm tra cây và C đối chiếu baseline. Chạy MLHAT nhóm và BR+HAT trên **Yeast** (2.417 mẫu, 103 đặc trưng, 14 nhãn theo Table 3). BR+HAT dùng một Hoeffding Adaptive Tree của River cho mỗi nhãn. Cố định phiên bản River và ghi toàn bộ tham số của baseline. Tham chiếu §4.3–4.4, Tables 6–8 và Figure 5. Đây là so sánh hai mô hình hoàn chỉnh. Mức chênh không thể quy hết cho split vì cấu trúc và bộ dự đoán cũng khác nhau.

## EXP-02 — Drift đột ngột và gradual

C chủ trì, B xác thực dữ liệu và A rà thao tác thay cây con. Chạy MLHAT, BR+HAT và MLHAT tắt drift trên **SynHPSud** và **SynHPGrad**. Mỗi stream có 50.000 mẫu, 30 đặc trưng số và 8 nhãn. Độ rộng drift lần lượt là 1 và 500. Đối chiếu §4.5, Table 4 và Figure 6(e–f). Giữ thứ tự nguồn, đánh dấu các mốc 12.500, 25.000 và 37.500 theo quy ước của paper, ghi rõ chỉ số bắt đầu từ đâu. Vẽ F1 theo cửa sổ 1.000 mẫu, ghi một điểm mỗi 250 mẫu theo quy ước của nhóm. Log riêng cảnh báo, promote và discard. Biến thể tắt drift giữ split, leaf, bootstrap và seed, đồng thời tắt detector và nhánh nền.

**Nguồn và schema đã kiểm tra (15/11/2026).** File gốc ở commit `8578941fcae1f9d55e95d6bc41b771a171a9c48d` của [`aestebant/mlhat/synthdata`](https://github.com/aestebant/mlhat/tree/8578941fcae1f9d55e95d6bc41b771a171a9c48d/synthdata), giấy phép repo CC0-1.0. Header ARFF ghi `-C 8`: tám nhãn `class0`–`class7` **đứng đầu**, sau đó là 30 đặc trưng `att1`–`att30`; mỗi dòng dữ liệu có dấu phẩy cuối. Hai checksum gốc và drift width được ghi ở [`data/manifests/hyperplane-reference.json`](../data/manifests/hyperplane-reference.json). Parser đã đọc cả hai file gốc và xác nhận shape `(50000, 30)` / `(50000, 8)`. Chưa có HF/Kaggle ID đã ghim và chưa có run EXP-02.

Theo hướng dẫn giảng viên do nhóm cung cấp, nếu không có bản trên HF/Kaggle phù hợp thì nhóm tạo dataset HF/Kaggle của mình từ file gốc. B xác nhận tài khoản/dataset ID và upload hai file ARFF nguyên byte cùng dataset card; C đối chiếu SHA-256 của file HF với checksum gốc. Script [`prepare_exp2_dataset.py`](../experiments/prepare_exp2_dataset.py) chuẩn bị hai file vào `data/raw/hyperplane-publish/` (bị Git ignore), không tự đăng. Runner [`reproduce_exp2.py`](../experiments/reproduce_exp2.py) chỉ tải từ HF ở revision 40 ký tự và từ chối file khác checksum gốc.

**Cập nhật 16/11/2026:** C đã đăng hai file lên [hanggio2105/mlhat-hyperplane](https://huggingface.co/datasets/hanggio2105/mlhat-hyperplane), public. Revision `7171b3ab7dc6b33aeea2723174f5ff9c32dc026f` được ghim trong [hyperplane-hf.json](../data/manifests/hyperplane-hf.json). Loader đã tải cả hai file qua HF và kiểm checksum/schema khớp bản gốc. Những ghi chú chưa có HF phía trên mô tả trạng thái ngày 15/11, đã được thay thế bởi cập nhật này.

Chạy từ repo root: `python -u -m experiments.reproduce_exp2 --manifest data/manifests/hyperplane-hf.json --resume`. Lệnh chạy hai stream × ba mô hình × ba seed và lưu từng run hoàn tất vào `artifacts/exp2_hyperplane/`. Ghi trực tiếp W&B bằng `--wandb --wandb-project TEN_PROJECT_CHUNG` sau khi commit code và đăng nhập W&B. Có thể đăng bằng chứng đã đo cục bộ với `python -m experiments.report_exp2 --wandb-project TEN_PROJECT_CHUNG --wandb-entity TEN_TAI_KHOAN`; script ghi rõ đây là upload kết quả đã đo, đính kèm measurement và snapshot mã nguồn, không giả lập run. W&B project phải public; cần kiểm tra URL khi không đăng nhập trước khi dùng trong deck. Không coi run cục bộ là đủ yêu cầu W&B.

Runner ghi F1 cửa sổ 1.000/250 và F1 trung bình có trọng số suy giảm mỗi mẫu `alpha=0.995`, lấy điểm mỗi 50 mẫu. Đây là định nghĩa tường minh của nhóm; phải kiểm chứng cách fading của evaluator tác giả trước khi gọi đường cong cùng giao thức với paper. Không tính chênh phần trăm giữa metric cả stream và điểm fading. Hiện chưa có đối chiếu evaluator gốc, chưa có W&B public. Snapshot nguồn của đợt chạy cục bộ ở `artifacts/exp2_hyperplane/source_snapshot.zip`; record ghi trung thực trạng thái Git còn thay đổi.

Hướng dẫn môn còn yêu cầu **chạy mã gốc tác giả để đối chiếu độc lập**. A/B/C cần lưu lệnh, phiên bản môi trường, stream, metric và output của run gốc riêng; không nhập hoặc sao chép code lõi gốc vào repo nhóm. Chỉ so số khi cùng dữ liệu và cùng cách tính metric.

## EXTRA-01 — Độ nhạy với kappa_alt

A chủ trì, C kiểm tra drift và B tổng hợp W&B. Chạy lại SynHPGrad với kappa_alt = 100, 200, 500 và giữ nguyên mọi tham số khác. Dùng lại run EXP-02 có cùng cấu hình tại 200. Báo F1 cả stream, F1 trong 2.000 mẫu sau từng mốc drift, số lần promote và thời gian xử lý. Mốc 200 là cấu hình tham chiếu đã chọn cho nhóm.

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
- Trên nhánh C, code đã tích hợp detector và nhánh nền; các giả định chưa chốt được ghi trong [decisions.md](decisions.md). Chưa có run EXP-01 được xác thực lại với code này, nên không dùng số cũ của biến thể chỉ có cây chính để báo cáo MLHAT hoàn chỉnh.

Nguồn phương pháp và đối chiếu. [Paper v1](https://arxiv.org/html/2410.20242v1).
