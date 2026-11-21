# MLHAT — tái triển khai và kiểm chứng bài báo

Nhóm 10 đang tự triển khai phương pháp **Multi-Label Hoeffding Adaptive Tree (MLHAT)** từ bài báo [*Hoeffding adaptive trees for multi-label classification on data streams*](https://arxiv.org/abs/2410.20242) của Esteban và cộng sự (2024, [DOI](https://doi.org/10.1016/j.knosys.2024.112561)). Mục tiêu là hiểu thuật toán, tái tạo một phần thí nghiệm và giải thích sai lệch giữa kết quả của nhóm với bài báo.

## Trạng thái

Repo đã có cây chính và phép chia trên đặc trưng số, bộ dự đoán nhiều nhãn tại lá, cơ chế phát hiện drift và cây con dự phòng, bộ tải Yeast, baseline BR+HAT và vòng đánh giá test-then-train. Phần drift đã qua kiểm thử trên các tình huống nhỏ nhưng còn những [giả định cần nhóm rà lại](docs/decisions.md) trước khi dùng để kết luận về MLHAT.

Nhóm đã hoàn thành **6 run EXP-01 trên Yeast** và **18 run EXP-02 trên hai stream Hyperplane**, đều có W&B công khai. Phân tích bổ sung của Midterm dùng các run bật/tắt drift đã có trong EXP-02. Phép đối chiếu với tác giả vẫn cần xác minh giao thức đánh giá. Dữ liệu Hugging Face được ghim revision và checksum trong cấu hình hoặc manifest.

## Cài đặt và kiểm thử

Repo yêu cầu Python 3.11 trở lên. Trong thư mục gốc:

```powershell
python -m pip install -e ".[dev,online]"
python -m pytest -q
python -m experiments.smoke_baseline
```

Lệnh cuối chỉ kiểm tra pipeline demo. Khi cần dữ liệu Hugging Face hoặc ghi W&B, cài thêm `.[data]` hoặc `.[tracking]`. Không đưa dữ liệu tải về, khóa API, checkpoint hay log W&B local vào Git.

## Các phần trong repo

| Phần | Tệp chính | Nội dung |
| --- | --- | --- |
| Cây và chia nhánh | [`tree.py`](src/mlhat/tree.py), [`splitting.py`](src/mlhat/splitting.py) | Thống kê node, chọn split và cập nhật cây online |
| Drift | [`drift.py`](src/mlhat/drift.py) | ADWIN, so lỗi hai cây và quyết định mở, thay hoặc bỏ cây con dự phòng |
| Dự đoán tại lá | [`leaves.py`](src/mlhat/leaves.py) | Label Powerset + kNN; Binary Relevance + bagging Logistic Regression |
| Dữ liệu và đánh giá | [`data.py`](src/mlhat/data.py), [`evaluation.py`](src/mlhat/evaluation.py) | Tải Yeast, kiểm tra schema, dự đoán trước khi học và tính metric |
| Thí nghiệm | [`experiments/`](experiments/), [`configs/`](configs/) | Baseline BR+HAT, runner EXP-01/EXP-02 và báo cáo/upload EXP-02 |
| Kiểm thử | [`tests/`](tests/) | Kiểm tra giao diện, metric, ADWIN và vòng đời cây con dự phòng |

Chi tiết về giao diện A/B/C nằm trong [integration-contracts.md](docs/integration-contracts.md). Những chỗ paper chưa rõ và lựa chọn triển khai cần xác nhận được ghi trong [decisions.md](docs/decisions.md).

## Phạm vi tái tạo đã chọn

| Mục | So sánh hoặc phân tích | Tình trạng |
| --- | --- | --- |
| EXP-01 | MLHAT và BR+HAT trên Yeast | [6 run công khai](#exp-01--kết-quả-yeast), mỗi run 2.417 mẫu |
| EXP-02 | MLHAT, BR+HAT và MLHAT tắt drift trên SynHPSud, SynHPGrad | [18 run công khai](docs/exp2-results.md), mỗi run 50.000 mẫu |
| Phân tích bổ sung | Ablation bật/tắt drift, ghép theo stream và seed | Dùng 12 trong 18 run EXP-02, gồm 6 run bật và 6 run tắt |

Khảo sát độ nhạy `kappa_alt = 100, 200, 500` được giữ làm hướng nghiên cứu tiếp theo, ngoài phạm vi kết quả Midterm.

Nhóm dùng thứ tự stream gốc và predict → metric → learn. Metric tổng kết là trung bình cả stream. EXP-02 có đường rolling và fading riêng, chưa xác minh phép tổng hợp tương đương paper. Tham số và giới hạn nằm trong [giao thức chung](docs/experiment-protocol.md), [báo cáo EXP-02](docs/exp2-results.md) và [quyết định của nhóm](docs/decisions.md).

Kiểm tra/xuất báo cáo từ **batch đã có**, không train hoặc upload:

```powershell
python -m pip install -e ".[dev,online,data,tracking,plots]"
python -m experiments.report_exp2 --validate-only
python -m experiments.report_exp2
python -m experiments.report_exp2 --plan-upload
```

Input bằng chứng `artifacts/exp2_hyperplane/` được giữ nguyên; đầu ra dẫn xuất ở `artifacts/exp2_reports/`. Dữ liệu tải/cache và outputs đều bị Git ignore. Tái chạy một đợt mới phải chọn output mới:

```powershell
python -u -m experiments.reproduce_exp2 --manifest data/manifests/hyperplane-hf.json --output-dir artifacts/exp2_rerun
```

Lệnh này chỉ dùng khi chủ động muốn train lại. Runner từ chối ghi đè JSON đã có; `--resume` chỉ skip đúng commit/config/checksum, không phải lệnh upload. Batch 18 run hiện đã public; xem báo cáo chính cho lệnh W&B và cách nhận diện run ID cố định. Không cần upload lại để tạo bằng chứng.

## Nhóm và bài nộp

| Thành viên | Phần chính |
| --- | --- |
| A — Phạm Thái Minh | Cây, thống kê node, split và lý thuyết |
| B — Vòng Bảo Thắng | Lá, dữ liệu, metric, W&B và EXP-01 |
| C — Đàng Trung Hiện | Drift, EXP-02 và phân tích sai lệch |

Midterm gồm **URL repo GitHub công khai** và file **`10.pptx`** riêng. Deck cần 20–40 slide, dưới 20 MB, có speaker notes trên mọi slide và trích dẫn IEEE tại slide dùng nguồn. Số liệu thực nghiệm phải truy được về code, config và W&B run công khai. Xem [kế hoạch slide](docs/midterm-slides.md) và [timeline](docs/timeline.md); hạn nộp cần đối chiếu Moodle.

Mỗi phần việc nên ở một nhánh để thành viên khác review trước khi merge. Khi sửa code, chạy test liên quan và ghi lệnh cùng kết quả trong Pull Request.

## Nguồn và phần tái sử dụng

Code MLHAT của nhóm dựa trên mô tả và công thức trong [paper](https://arxiv.org/html/2410.20242v1). [Repo tác giả](https://github.com/aestebant/mlhat) chỉ dùng để kiểm chứng và đối chiếu độc lập; nhóm không sao chép code lõi.

Repo dùng NumPy và scikit-learn cho hạ tầng; River 0.26.1 cung cấp ADWIN, các bộ học ở lá và baseline BR+HAT. Hugging Face datasets dùng để tải dữ liệu bằng code, còn W&B dùng để lưu bằng chứng run khi thí nghiệm được thực hiện.

## EXP-01 — Kết quả Yeast

Nhóm chạy MLHAT và BR+HAT trên cùng 2.417 mẫu Yeast theo thứ tự nguồn. Sáu run `Finished` ghi commit [`344cf97`](https://github.com/HienTrung2105/mlhat-reproduction/commit/344cf9733882530d43ad0bebf30b9f3ac743fa8d). Các tham số, revision và checksum trong W&B khớp cấu hình tại commit đó. Xem [script](experiments/reproduce_exp1.py), [cấu hình](configs/exp1_yeast.json) và [project W&B công khai](https://forge.coreweave.com/wandb/ml-group10-24khmt1-2026/mlhat-team10-exp1-drift).

| Model | Seed 1 | Seed 2 | Seed 3 | Example F1 (mean ± SD mẫu) |
| --- | --- | --- | --- | --- |
| MLHAT | [run](https://forge.coreweave.com/wandb/ml-group10-24khmt1-2026/mlhat-team10-exp1-drift/runs/d2gaujzt) | [run](https://forge.coreweave.com/wandb/ml-group10-24khmt1-2026/mlhat-team10-exp1-drift/runs/ngdas8ng) | [run](https://forge.coreweave.com/wandb/ml-group10-24khmt1-2026/mlhat-team10-exp1-drift/runs/fgvhzyga) | 0.493146 ± 0.003487 |
| BR+HAT | [run](https://forge.coreweave.com/wandb/ml-group10-24khmt1-2026/mlhat-team10-exp1-drift/runs/1xcadmvz) | [run](https://forge.coreweave.com/wandb/ml-group10-24khmt1-2026/mlhat-team10-exp1-drift/runs/4me7j1ep) | [run](https://forge.coreweave.com/wandb/ml-group10-24khmt1-2026/mlhat-team10-exp1-drift/runs/2zfdhtoq) | 0.576288 ± 0.012768 |

## EXP-02: script/config → W&B runs

[Project public](https://forge.coreweave.com/wandb/ml-group10-24khmt1-2026/mlhat-team10-exp2). Script đo: `experiments/reproduce_exp2.py`; config: `configs/exp2_hyperplane.json`; uploader: `experiments/report_exp2.py --upload`. Tất cả 18 run Finished đã đối chiếu config/metrics/ảnh/bảng. Bảng là Example F1 mean ± SD mẫu qua ba seed; phần so với paper vẫn cần xác minh giao thức.

| Dataset | Model | Seed 1 | Seed 2 | Seed 3 | Example F1 |
| --- | --- | --- | --- | --- | --- |
| SynHPSud | MLHAT | [run](https://forge.coreweave.com/wandb/ml-group10-24khmt1-2026/mlhat-team10-exp2/runs/exp2c83ed081d41cb0fb) | [run](https://forge.coreweave.com/wandb/ml-group10-24khmt1-2026/mlhat-team10-exp2/runs/exp2aac5e879609efb7e) | [run](https://forge.coreweave.com/wandb/ml-group10-24khmt1-2026/mlhat-team10-exp2/runs/exp2bd953105936de12f) | 0.184405 ± 0.000254 |
| SynHPSud | BR+HAT | [run](https://forge.coreweave.com/wandb/ml-group10-24khmt1-2026/mlhat-team10-exp2/runs/exp2c9a24d42f57ad173) | [run](https://forge.coreweave.com/wandb/ml-group10-24khmt1-2026/mlhat-team10-exp2/runs/exp280a2ce2de88730a1) | [run](https://forge.coreweave.com/wandb/ml-group10-24khmt1-2026/mlhat-team10-exp2/runs/exp2a21738073805368b) | 0.173472 ± 0.005833 |
| SynHPSud | MLHAT-no-drift | [run](https://forge.coreweave.com/wandb/ml-group10-24khmt1-2026/mlhat-team10-exp2/runs/exp2175bed53ee351153) | [run](https://forge.coreweave.com/wandb/ml-group10-24khmt1-2026/mlhat-team10-exp2/runs/exp2d2df3fd1d2c56694) | [run](https://forge.coreweave.com/wandb/ml-group10-24khmt1-2026/mlhat-team10-exp2/runs/exp204287f774ab08ca5) | 0.184651 ± 0.000169 |
| SynHPGrad | MLHAT | [run](https://forge.coreweave.com/wandb/ml-group10-24khmt1-2026/mlhat-team10-exp2/runs/exp2830df16218a63401) | [run](https://forge.coreweave.com/wandb/ml-group10-24khmt1-2026/mlhat-team10-exp2/runs/exp2cb829f55bf517e90) | [run](https://forge.coreweave.com/wandb/ml-group10-24khmt1-2026/mlhat-team10-exp2/runs/exp26e278b8c0356113a) | 0.121807 ± 0.000175 |
| SynHPGrad | BR+HAT | [run](https://forge.coreweave.com/wandb/ml-group10-24khmt1-2026/mlhat-team10-exp2/runs/exp27528bfc743bd8336) | [run](https://forge.coreweave.com/wandb/ml-group10-24khmt1-2026/mlhat-team10-exp2/runs/exp22e3847a7d4a9630d) | [run](https://forge.coreweave.com/wandb/ml-group10-24khmt1-2026/mlhat-team10-exp2/runs/exp2d5742212f6d5001f) | 0.116550 ± 0.012620 |
| SynHPGrad | MLHAT-no-drift | [run](https://forge.coreweave.com/wandb/ml-group10-24khmt1-2026/mlhat-team10-exp2/runs/exp2da20c25df6e0577d) | [run](https://forge.coreweave.com/wandb/ml-group10-24khmt1-2026/mlhat-team10-exp2/runs/exp2eb47fc013fa2e81e) | [run](https://forge.coreweave.com/wandb/ml-group10-24khmt1-2026/mlhat-team10-exp2/runs/exp2c636191998a5e176) | 0.121401 ± 0.000405 |
