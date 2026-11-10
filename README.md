# MLHAT — tái triển khai và kiểm chứng bài báo

Nhóm 10 đang tự triển khai phương pháp **Multi-Label Hoeffding Adaptive Tree (MLHAT)** từ bài báo [*Hoeffding adaptive trees for multi-label classification on data streams*](https://arxiv.org/abs/2410.20242) của Esteban và cộng sự (2024, [DOI](https://doi.org/10.1016/j.knosys.2024.112561)). Mục tiêu là hiểu thuật toán, tái tạo một phần thí nghiệm và giải thích sai lệch giữa kết quả của nhóm với bài báo.

## Trạng thái

Repo đã có cây chính và phép chia trên đặc trưng số, bộ dự đoán nhiều nhãn tại lá, cơ chế phát hiện drift và cây con dự phòng, bộ tải Yeast, baseline BR+HAT và vòng đánh giá test-then-train. Phần drift đã qua kiểm thử trên các tình huống nhỏ nhưng còn những [giả định cần nhóm rà lại](docs/decisions.md) trước khi dùng để kết luận về MLHAT.

**Chưa có run tái tạo được xác thực hoặc W&B run công khai.** Yeast đã được ghim phiên bản và checksum; nguồn cho SynHPSud và SynHPGrad chưa được ghim. Script EXP-01 hiện có là hạ tầng chạy, chưa phải bằng chứng tái tạo. Script `smoke_baseline` chỉ là demo synthetic với mô hình SGD theo từng nhãn.

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
| Thí nghiệm | [`experiments/`](experiments/), [`configs/`](configs/) | Baseline BR+HAT, cấu hình và script EXP-01 |
| Kiểm thử | [`tests/`](tests/) | Kiểm tra giao diện, metric, ADWIN và vòng đời cây con dự phòng |

Chi tiết về giao diện A/B/C nằm trong [integration-contracts.md](docs/integration-contracts.md). Những chỗ paper chưa rõ và lựa chọn triển khai cần xác nhận được ghi trong [decisions.md](docs/decisions.md).

## Phạm vi tái tạo đã chọn

| Mục | So sánh hoặc phân tích | Tình trạng |
| --- | --- | --- |
| EXP-01 | MLHAT và BR+HAT trên Yeast | Có loader, config và runner; chưa có run báo cáo |
| EXP-02 | MLHAT, BR+HAT và MLHAT tắt drift trên SynHPSud, SynHPGrad | Chưa ghim nguồn stream và chưa có runner |
| EXTRA-01 | Độ nhạy theo `kappa_alt = 100, 200, 500` trên SynHPGrad | Chưa chạy |

Nhóm dùng thứ tự stream gốc, dự đoán trước khi học từng mẫu và báo example-based F1, Hamming loss, subset accuracy. Giao thức, seed, tham số, nguồn dữ liệu và giới hạn khi đối chiếu số liệu paper nằm trong [experiment-protocol.md](docs/experiment-protocol.md). Evaluator hiện tính trung bình toàn stream; paper dùng forgetting factor `0.995` và đánh giá mỗi 50 mẫu, nên chưa thể xem hai giá trị là cùng giao thức.

Khi có run hợp lệ, README sẽ ghi ánh xạ **thí nghiệm → script/config → W&B run public → kết quả**, rồi đối chiếu các giá trị so sánh được với paper. Không dùng số của demo synthetic làm kết quả tái tạo.

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
