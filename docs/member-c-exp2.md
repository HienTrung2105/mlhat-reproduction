# Bàn giao EXP-02 của C

## Nguồn đã xác minh

- HF public: https://huggingface.co/datasets/hanggio2105/mlhat-hyperplane
- Revision: `7171b3ab7dc6b33aeea2723174f5ff9c32dc026f`.
- Manifest chạy: `data/manifests/hyperplane-hf.json`.
- Hai file tải bằng loader từ HF có SHA-256 khớp file gốc tác giả. Mỗi stream có 50.000 mẫu, 8 nhãn đầu dòng, 30 đặc trưng sau; giữ nguyên thứ tự dòng, không preprocessing hoặc shuffle.

## Chạy và tổng hợp

Trong PowerShell tại repo root:

```powershell
python -m pip install -e ".[dev,online,data,tracking,plots]"
python -m pytest -q
python -u -m experiments.reproduce_exp2 --manifest data/manifests/hyperplane-hf.json --resume
python -m experiments.report_exp2
```

Runner xử lý 18 run: hai stream × ba mô hình × seed 1, 2, 3. Biến thể tắt drift giữ cùng split, leaf, Poisson và seed. Mỗi run hoàn tất ghi JSON vào `artifacts/exp2_hyperplane/`; `--resume` chỉ bỏ qua record khớp commit, config, checksum và đủ 50.000 mẫu. Script tổng hợp chỉ đọc run hoàn tất, ghi `summary.csv` (mean, SD mẫu), `run_registry.csv` (từng run, sự kiện drift, URL), và bốn PNG của hai loại đường F1. Nếu chỉ một seed hoàn tất, SD để trống, không trình bày như kết quả ba seed.

## W&B là nơi lưu kết quả chạy

HF giữ dữ liệu đầu vào. W&B giữ config, metrics, đường cong và bằng chứng run để thầy mở link kiểm tra. Môn yêu cầu một project public chung cho EXP-01, EXP-02 và EXTRA-01.

Máy chưa đăng nhập W&B lúc bắt đầu đợt chạy. Các phép đo được lưu cục bộ trước; chưa phải bằng chứng W&B public. Đăng nhập bằng `python -m wandb login`, nhập key trực tiếp trong terminal, không đưa key vào chat hay Git. Sau khi thống nhất tên project với B:

```powershell
python -m experiments.report_exp2 --wandb-entity TEN_TAI_KHOAN --wandb-project TEN_PROJECT_CHUNG
```

Lệnh trên upload phép đo đã hoàn tất, không chạy lại hoặc tạo số mới. Nó ghi rõ chế độ upload, trạng thái Git lúc đo và SHA-256 của `source_snapshot.zip`, đồng thời đính kèm record và mã nguồn. Snapshot nguồn được tạo lúc bắt đầu đợt chạy cục bộ vì code mới chưa commit; không được ghi những run này là chạy trên một commit sạch. Bổ sung URL và số liệu vào README sau khi mở được run khi không đăng nhập. Không ghi “public” chỉ từ việc có URL.

## Nội dung cho slide C

- Slide 27: sơ đồ EXP-02, nguồn HF/revision, ba mô hình, ba seed và mốc drift một-based 12.500/25.000/37.500; width 1 và 500 từ paper.
- Slide 28: dùng PNG nhóm tự tạo từ run đã hoàn tất, thêm citation W&B cho mỗi đường. Phân biệt cửa sổ 1.000 mẫu/điểm mỗi 250 mẫu với fading mỗi mẫu alpha 0.995/điểm mỗi 50 mẫu.
- Trình bày sự kiện mở/thay/bỏ cây từ record. Nếu không có promote hoặc biến thể tắt drift tương đương, báo đúng quan sát; không coi đó tự động là bằng chứng phát hiện drift thành công hay thất bại trên mọi stream.
- Table kết quả chỉ báo mean ± SD khi đủ ba seed. Số paper chỉ so lệch tương đối sau khi xác minh evaluator/giao thức trùng khớp.

## Những việc vẫn cần trước kết luận tái tạo

- A/C rà lại warning ADWIN, lỗi cây nền và trọng số kết hợp đang ghi ở `docs/decisions.md`. Đây là giả định của implementation nhóm, chưa phải lựa chọn được nhóm xác nhận.
- Chạy implementation tác giả trong môi trường riêng, ghi rõ lệnh/config/seed và đối chiếu evaluator. Không sao chép code lõi vào repo nhóm.
- Bảng F1 tác giả nằm ở `results/per_dataset/example_f1.csv` của repo tác giả. Có dòng SynHPSud/SynHPGrad nhưng chưa xác minh cách tổng hợp, nên không lấy trực tiếp số đó trừ với F1 toàn stream của nhóm.
- EXTRA-01 thuộc A; EXP-01 và ghép deck thuộc B. Chưa có đủ bằng chứng cho toàn bộ bài nộp chỉ từ việc hoàn thành runner EXP-02.
