# MLHAT — tái triển khai và kiểm chứng bài báo

Nhóm 3 sinh viên đang chuẩn bị tái triển khai độc lập phương pháp **Multi-Label Hoeffding Adaptive Tree (MLHAT)** trong bài báo [*Hoeffding adaptive trees for multi-label classification on data streams*](https://arxiv.org/abs/2410.20242) (2024, DOI: [10.1016/j.knosys.2024.112561](https://doi.org/10.1016/j.knosys.2024.112561)). Mã nguồn của [tác giả](https://github.com/aestebant/mlhat) chỉ dùng để đối chiếu, không sao chép thuật toán.

**Trạng thái:** code hiện có cây chính, bộ dự đoán lá, pipeline EXP-01 và phần drift đang được tích hợp trên nhánh C. **Chưa có run tái tạo được xác thực hay kết quả W&B công khai.** Các giả định drift cần nhóm review trước khi xem đây là MLHAT hoàn chỉnh. Nhóm đã ghi nhận mã nhóm **10**, phân công A/B/C và phạm vi Core trong [quyết định dự án](docs/decisions.md). Hạn nộp ghi trong [timeline](docs/timeline.md) cần đối chiếu thông báo Moodle.

## Bắt đầu

Cần Python 3.11 trở lên theo `pyproject.toml`. Trong thư mục gốc của repo:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev,online]"
python -m pytest -q
python -m experiments.smoke_baseline
```

Trên macOS/Linux, dùng `python3.11 -m venv .venv`, `source .venv/bin/activate`, rồi chạy các lệnh `pip`, `pytest` và demo tương ứng. Nếu PowerShell chặn activate, có thể gọi trực tiếp `.venv\Scripts\python.exe`.

`smoke_baseline` chỉ kiểm tra pipeline bằng dữ liệu synthetic và mô hình SGD theo từng nhãn; kết quả **không phải** kết quả MLHAT hoặc paper. Có thể cài riêng `.[data]` để dùng Hugging Face datasets và `.[tracking]` để ghi W&B khi thực sự cần.

## Phần hiện có

| Đường dẫn | Nội dung |
| --- | --- |
| `src/mlhat/interfaces.py` | Giao diện `predict_one`/`learn_one` và kiểm tra định dạng mẫu |
| `src/mlhat/evaluation.py` | Vòng test-then-train, example-F1, Hamming loss, subset accuracy và micro-F1; chưa có forgetting factor của paper |
| `src/mlhat/data.py` | Loader Yeast đã ghim revision và checksum; hai stream Hyperplane chưa ghim nguồn |
| `src/mlhat/tree.py`, `src/mlhat/splitting.py` | Cây chính, thống kê node và split số; nhánh C tích hợp cây nền và sự kiện drift |
| `src/mlhat/drift.py` | ADWIN và chính sách mở, thay, bỏ nhánh nền theo Eq. (16) |
| `src/mlhat/leaves.py` | Dự đoán tại lá bằng LP+kNN và BR+bagging LR |
| `experiments/smoke_baseline.py` | Demo baseline synthetic, không thuộc thí nghiệm paper |
| `tests/` | Kiểm tra giao diện và vòng đánh giá hiện có |
| `docs/decisions.md` | Mã nhóm, phân công và quyết định phạm vi Core |
| `docs/experiment-protocol.md` | Kế hoạch EXP-01, EXP-02 và EXTRA-01; còn phải xác minh nguồn dữ liệu |
| `docs/integration-contracts.md` | Giao diện tích hợp A/B/C đã được nhóm cập nhật |
| `docs/paper-notes.md` | Ghi chú phương pháp và các điểm cần đối chiếu paper |

Chỉ tạo thêm script, cấu hình, hình, slide và thư mục kết quả khi có nội dung thật. Không commit dữ liệu tải về, API key, log W&B hoặc số đo chưa chạy.

## Việc tiếp theo của nhóm

1. Đối chiếu hạn nộp trong `docs/timeline.md` với Moodle và cập nhật khi có thông báo chính thức.
2. Cả ba tiếp tục kiểm chứng `docs/paper-notes.md` và `docs/experiment-protocol.md` theo paper; ghim nguồn, revision và quyền sử dụng dataset trước khi chạy.
3. Thực hiện phạm vi Core đã ghi: hai thí nghiệm chính và một phân tích bổ sung, chỉ báo cáo số liệu từ run thật.
4. A phụ trách tree/splitting và lý thuyết; B phụ trách leaf prediction, data, evaluation và W&B; C phụ trách drift và phối hợp tái tạo. Mỗi người làm slide phần mình và review phần của người khác.
5. Triển khai MLHAT độc lập từ paper, kiểm tra bằng trường hợp nhỏ, rồi chạy thí nghiệm với dữ liệu tải tự động từ Hugging Face/Kaggle. Ghi config, seed, môi trường và W&B run **public** cho từng kết quả.

Midterm có **hai bài nộp riêng**: URL repo GitHub công khai và file `10.pptx`. Deck cần 20–40 slide, dưới 20 MB, speaker notes ở mọi slide (tối đa 500 từ/slide), hình nhóm tự làm và trích dẫn IEEE trên slide sử dụng nguồn. Chỉ đưa số liệu thực nghiệm khi truy được về run W&B công khai và code/config tương ứng. Phân công cụ thể ở [kế hoạch slide](docs/midterm-slides.md).

## Kết quả tái tạo

**Chưa có run tái tạo.** Khi có kết quả thật, thêm bảng `experiment → script/config → W&B run URL → kết quả một dòng` tại đây và phân tích sai lệch so với paper. Không dùng số liệu của demo synthetic làm kết quả tái tạo.

## Cách cộng tác

Tạo nhánh từ `main` cho từng phần việc. Trước khi push hoặc mở Pull Request, mỗi người tự chạy trong môi trường Python 3.11:

```powershell
python -m pip install -e ".[dev,online]"
python -m pytest -q
```

Nếu sửa pipeline hoặc demo, chạy thêm `python -m experiments.smoke_baseline`. Trong Pull Request, ghi thay đổi, phần A/B/C phụ trách, căn cứ từ paper nếu sửa MLHAT, lệnh kiểm tra cùng kết quả thật, và phần còn chưa rõ. Nhờ ít nhất một thành viên khác review trước khi merge; không force-push `main` và cần thống nhất khi đổi giao diện chung. Repo hiện không chạy test tự động trên GitHub.

Các thư viện hạ tầng hiện dùng gồm NumPy, scikit-learn và River 0.26.1 (ADWIN, bộ dự đoán tại lá và baseline BR+HAT). Hugging Face datasets/W&B là phụ thuộc tùy chọn cho dữ liệu và tracking. Phần chính sách cây MLHAT, split và thay nhánh nền do nhóm tự triển khai; không dùng package MLHAT có sẵn.
