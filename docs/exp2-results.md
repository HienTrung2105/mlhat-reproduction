# EXP-02 — kết quả cục bộ ngày 16/11/2026

**Đã chạy thật đủ 18 run**, mỗi run 50.000 mẫu: hai stream × ba mô hình × seed 1, 2, 3. Chưa có W&B public; các số này là bằng chứng cục bộ, chưa đáp ứng đầy đủ yêu cầu báo cáo của môn. Chưa chạy đối chiếu implementation gốc tác giả, chưa kết luận tái tạo thành công.

## Nguồn và lệnh

- Dữ liệu tải bằng code từ [HF của nhóm](https://huggingface.co/datasets/hanggio2105/mlhat-hyperplane), revision `7171b3ab7dc6b33aeea2723174f5ff9c32dc026f`, checksum trong [manifest](../data/manifests/hyperplane-hf.json) đã kiểm khớp file gốc tác giả.
- [Cấu hình](../configs/exp2_hyperplane.json), [runner](../experiments/reproduce_exp2.py), [script tổng hợp](../experiments/report_exp2.py).
- Lệnh đo: `python -u -m experiments.reproduce_exp2 --manifest data/manifests/hyperplane-hf.json --resume`.
- Lệnh tổng hợp: `python -m experiments.report_exp2`.
- Đợt đo bắt đầu khi HEAD là `6d5c122` và code mới chưa commit; record ghi `git_dirty=true`. Snapshot chính xác nguồn đo nằm tại `artifacts/exp2_hyperplane/source_snapshot.zip`, SHA-256 `d229a956f7b7a25c07d0bac05f62121db27f911e9fc69e4ac4278562d438965b`. Các file trong snapshot đã đối chiếu khớp nguồn được commit trong `16bea75`; không sửa commit của record hồi tố.
- Môi trường: Python 3.13.5, NumPy 2.3.4, SciPy 1.16.3, scikit-learn 1.7.2, River 0.26.1. Metadata hệ điều hành/phần cứng và thời gian từng run nằm trong JSON. Thời gian không phải benchmark máy chuyên dụng có kiểm soát tải nền.

## Metric trung bình cả stream

Mỗi ô là mean ± SD mẫu của ba seed. F1 lấy trung bình theo mẫu, F1 khi cả nhãn thật/dự đoán rỗng bằng 0; dự đoán trước khi học, không shuffle. Giá trị trong [summary.csv](../artifacts/exp2_hyperplane/summary.csv) có đầy đủ độ chính xác.

| Stream | Mô hình | Example F1 ↑ | Hamming loss ↓ | Subset accuracy ↑ |
| --- | --- | --- | --- | --- |
| SynHPSud | MLHAT | 0.184405 ± 0.000254 | 0.268576 ± 0.000214 | 0.026113 ± 0.000168 |
| SynHPSud | BR+HAT | 0.173472 ± 0.005833 | 0.269277 ± 0.001015 | 0.022740 ± 0.000764 |
| SynHPSud | MLHAT tắt drift | 0.184651 ± 0.000169 | 0.268407 ± 0.000121 | 0.026233 ± 0.000223 |
| SynHPGrad | MLHAT | 0.121807 ± 0.000175 | 0.257334 ± 0.000117 | 0.015727 ± 0.000160 |
| SynHPGrad | BR+HAT | 0.116550 ± 0.012620 | 0.258973 ± 0.000798 | 0.016487 ± 0.001916 |
| SynHPGrad | MLHAT tắt drift | 0.121401 ± 0.000405 | 0.257253 ± 0.000091 | 0.015600 ± 0.000211 |

MLHAT có F1 trung bình cao hơn BR+HAT ở hai stream, nhưng không thắng mọi metric: subset accuracy trên gradual thấp hơn baseline. Chưa kiểm định ý nghĩa thống kê; không suy rộng thành ưu thế tổng quát.

## Cơ chế drift thực sự làm gì trong các run này?

Mỗi run MLHAT bật drift có đúng một START_BACKGROUND, một DISCARD_BACKGROUND, **không có PROMOTE_BACKGROUND**. Tất cả xảy ra ở root, sau mốc 25.000. Sáu run tắt drift không có sự kiện; sự kiện nội bộ River của BR+HAT chưa được instrument, nên không báo rằng baseline có 0 cảnh báo.

| Stream | Seed | Mở cây nền tại mẫu | Bỏ cây nền tại mẫu |
| --- | --- | --- | --- |
| SynHPSud | 1 | 25312 | 27425 |
| SynHPSud | 2 | 25344 | 25983 |
| SynHPSud | 3 | 25408 | 25905 |
| SynHPGrad | 1 | 25152 | 25815 |
| SynHPGrad | 2 | 25152 | 25450 |
| SynHPGrad | 3 | 25152 | 25624 |

Bật drift đổi F1 trung bình khoảng −0.000246 trên sudden và +0.000405 trên gradual so với tắt drift. Quan sát này chưa cho thấy lợi ích rõ từ thay cây trong cấu hình nhóm: không có lần thay cây nào. Đường F1 tăng sau mốc 25.000 không được quy cho promote; cả biến thể tắt drift cũng tăng. Cần xem ảnh hưởng của dự đoán kết hợp, lựa chọn leaf và giả định drift trước khi giải thích nhân quả.

## Hình cho slide và giới hạn đối chiếu

- `artifacts/exp2_hyperplane/SynHPSud_rolling_f1.png` và `SynHPGrad_rolling_f1.png`: cửa sổ 1.000 mẫu, điểm mỗi 250 mẫu, trung bình ba seed. Các đường gần trùng nhau có thể che nhau; bảng trên vẫn phải đi kèm để thấy sai khác.
- Hai file `*_faded_f1.png`: fading alpha 0.995 theo từng mẫu, điểm mỗi 50 mẫu. Đây là định nghĩa của evaluator nhóm, chưa xác minh trùng cách tổng hợp của tác giả.
- [Run registry](../artifacts/exp2_hyperplane/run_registry.csv): số liệu từng run, thời gian, số sự kiện và ô URL W&B còn trống. JSON từng run chứa toàn bộ điểm đường cong và config.
- [Bảng F1 của tác giả](https://github.com/aestebant/mlhat/blob/8578941fcae1f9d55e95d6bc41b771a171a9c48d/results/per_dataset/example_f1.csv) có MLHAT SynHPSud = 0.3269752095776777, SynHPGrad = 0.2892550611489483. Đây là số tham chiếu của tác giả; **chưa xác minh phép tổng hợp tương đương F1 toàn stream ở bảng nhóm**, nên chưa tính relative deviation hay kết luận khoảng cách định lượng với paper.

## Việc còn lại

1. Đăng nhập W&B bằng terminal, thống nhất một project public chung với B. Upload phép đo và snapshot bằng `experiments.report_exp2`; kiểm tra các URL ở chế độ không đăng nhập, rồi cập nhật README và citation cho slide.
2. A/C rà các giả định ở [decisions.md](decisions.md), chạy code gốc trong môi trường riêng và đối chiếu evaluator/output. Giải thích sai lệch dựa trên kiểm tra đó; không quy toàn bộ khoảng cách cho seed hay thư viện khi chưa có bằng chứng.
3. C đưa cấu hình, bảng và hình vào slide 27–28; A làm EXTRA-01, B ghép deck. Các file artifacts là local và bị Git ignore; người clone repo chạy lại từ script hoặc lấy artifact W&B khi đã đăng public.
