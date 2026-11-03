# MLHAT — tái triển khai và kiểm chứng bài báo

Nhóm 3 sinh viên đang chuẩn bị tái triển khai độc lập phương pháp **Multi-Label Hoeffding Adaptive Tree (MLHAT)** trong bài báo [*Hoeffding adaptive trees for multi-label classification on data streams*](https://arxiv.org/abs/2410.20242) (2024, DOI: [10.1016/j.knosys.2024.112561](https://doi.org/10.1016/j.knosys.2024.112561)). Mã nguồn của [tác giả](https://github.com/aestebant/mlhat) chỉ dùng để đối chiếu, không sao chép thuật toán.

**Trạng thái:** repo mới có hạ tầng nhỏ, vòng đánh giá test-then-train và demo baseline trên dữ liệu synthetic. **Chưa có MLHAT hoàn chỉnh, thí nghiệm tái tạo hay kết quả W&B.** Việc paper được duyệt, mã nhóm, deadline và phạm vi thực nghiệm vẫn cần xác nhận; xem [quyết định còn mở](docs/decisions.md).

## Bắt đầu

Cần Python 3.10 hoặc 3.11. Trong thư mục gốc của repo:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
python -m pytest -q
python -m experiments.smoke_baseline
```

Trên macOS/Linux, dùng `python3.11 -m venv .venv`, `source .venv/bin/activate`, rồi chạy các lệnh `pip`, `pytest` và demo tương ứng. Nếu PowerShell chặn activate, có thể gọi trực tiếp `.venv\Scripts\python.exe`.

`smoke_baseline` chỉ kiểm tra pipeline bằng dữ liệu synthetic và mô hình SGD theo từng nhãn; kết quả **không phải** kết quả MLHAT hoặc paper. Có thể cài riêng `.[data]` để dùng Hugging Face datasets và `.[tracking]` để ghi W&B khi thực sự cần.

## Phần hiện có

| Đường dẫn | Nội dung |
| --- | --- |
| `src/mlhat/interfaces.py` | Giao diện `predict_one`/`learn_one` và kiểm tra định dạng mẫu |
| `src/mlhat/evaluation.py` | Vòng test-then-train và metric demo; cần đối chiếu metric paper trước khi báo cáo |
| `src/mlhat/data.py` | Hàm tải dataset Hugging Face chung; chưa có dataset ID/schema đã duyệt |
| `src/mlhat/tree.py` | Khung giao diện cây; phương pháp MLHAT chưa được triển khai |
| `experiments/smoke_baseline.py` | Demo baseline synthetic, không thuộc thí nghiệm paper |
| `tests/` | Kiểm tra giao diện và vòng đánh giá hiện có |
| `docs/decisions.md` | Các quyết định nhóm còn cần xác nhận |
| `docs/experiment-protocol.md` | Bảng cần điền từ phần thực nghiệm của paper trước khi viết script tái tạo |

Chỉ tạo thêm script, cấu hình, hình, slide và thư mục kết quả khi có nội dung thật. Không commit dữ liệu tải về, API key, log W&B hoặc số đo chưa chạy.

## Việc tiếp theo của nhóm

1. Xác nhận paper, mã nhóm và hạn nộp từ Moodle; cập nhật `docs/decisions.md`.
2. Cả ba đọc paper, ghi nhận công thức và các điểm chưa rõ. Điền `docs/experiment-protocol.md` bằng dataset, baseline, metric, cấu hình, seed, hình/bảng paper và nguồn tải dữ liệu.
3. Chọn Core hay Full scope. Nếu chọn Core, ưu tiên ít nhất hai thí nghiệm quan trọng, gồm so sánh trung tâm, và một phân tích bổ sung do nhóm thiết kế.
4. A phụ trách tree/splitting và lý thuyết; B phụ trách leaf prediction, data, evaluation và W&B; C phụ trách drift và phối hợp tái tạo. Mỗi người làm slide phần mình và review phần của người khác.
5. Triển khai MLHAT độc lập từ paper, kiểm tra bằng trường hợp nhỏ, rồi chạy thí nghiệm với dữ liệu tải tự động từ Hugging Face/Kaggle. Ghi config, seed, môi trường và W&B run **public** cho từng kết quả.

Midterm có **hai bài nộp riêng**: URL repo GitHub công khai và file `ID.pptx` với `ID` là mã nhóm thật. Deck cần 20–40 slide, dưới 20 MB, speaker notes ở mọi slide (tối đa 500 từ/slide), hình nhóm tự làm và trích dẫn IEEE trên slide sử dụng nguồn. Chỉ đưa số liệu thực nghiệm khi truy được về run W&B công khai và code/config tương ứng. Yêu cầu chi tiết và phân công slide đề xuất nằm trong [AGENTS.md](AGENTS.md).

## Kết quả tái tạo

**Chưa có run tái tạo.** Khi có kết quả thật, thêm bảng `experiment → script/config → W&B run URL → kết quả một dòng` tại đây và phân tích sai lệch so với paper. Không dùng số liệu của demo synthetic làm kết quả tái tạo.

## Cách cộng tác

Tạo nhánh từ `main` cho từng phần việc. Trước khi push hoặc mở Pull Request, mỗi người tự chạy trong môi trường Python 3.11:

```powershell
python -m pip install -e ".[dev]"
python -m pytest -q
```

Nếu sửa pipeline hoặc demo, chạy thêm `python -m experiments.smoke_baseline`. Trong Pull Request, ghi thay đổi, phần A/B/C phụ trách, căn cứ từ paper nếu sửa MLHAT, lệnh kiểm tra cùng kết quả thật, và phần còn chưa rõ. Nhờ ít nhất một thành viên khác review trước khi merge; không force-push `main` và cần thống nhất khi đổi giao diện chung. Repo hiện không chạy test tự động trên GitHub.

Các thư viện hạ tầng hiện dùng gồm NumPy, scikit-learn và, nếu cài tùy chọn, Hugging Face datasets/W&B; chúng không cung cấp sẵn MLHAT.
