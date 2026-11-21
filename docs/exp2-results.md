# EXP-02 — kết quả và tái lập

## Trạng thái

Đã đo thật **18 run**, mỗi run **50.000 mẫu**, và upload đủ 18 run **Finished** vào [W&B của Team](https://forge.coreweave.com/wandb/ml-group10-24khmt1-2026/mlhat-team10-exp2). **Public đã được kiểm tra ngày 16/11/2026 bằng HTTP/API không có cookie hoặc key.** Đây là báo cáo EXP-02 chính; protocol chung và các quyết định chưa chốt vẫn nằm ở [experiment-protocol.md](experiment-protocol.md) và [decisions.md](decisions.md).

Phần chạy và bằng chứng public đã có. **Chưa xác thực phép so với evaluator/code gốc tác giả**, nên chưa tuyên bố tái tạo thành công hay tính relative deviation với paper.

## Phạm vi và protocol đã dùng

- Hai stream SynHPSud/SynHPGrad, ba mô hình MLHAT, BR+HAT, MLHAT tắt drift; seed 1, 2, 3. Ablation giữ split, leaf, bootstrap và seed, chỉ tắt detector/nhánh nền.
- Predict → metric → learn, giữ thứ tự dòng, không shuffle, không fit preprocessing trên tương lai. Lá trống dự đoán vector 0.
- Example F1 từng mẫu = 2TP/(2TP+FP+FN), mẫu số 0 thì F1=0. Hamming loss là tỷ lệ nhãn sai; subset accuracy là toàn vector khớp. Bảng dưới lấy trung bình toàn stream, rồi mean ± **SD mẫu** qua ba seed (`ddof=1`).
- Đường rolling F1 dùng cửa sổ 1.000 mẫu, điểm mỗi 250 mẫu (197 điểm/run). Đường fading là trung bình có trọng số suy giảm mỗi mẫu alpha=0.995, điểm mỗi 50 mẫu (1.000 điểm/run). Hai loại curve có định nghĩa khác nhau; chưa xác minh fading của nhóm trùng evaluator tác giả.
- Mốc drift một-based 12.500, 25.000, 37.500; width sudden=1, gradual=500, theo paper §4.1.2/Table 4. Tham chiếu kết quả drift: §4.5/Figure 6(e–f).
- Tham số giữ nguyên [exp2_hyperplane.json](../configs/exp2_hyperplane.json): delta_spl=1e-7, kappa_spl=200, delta_alt=0.05, kappa_alt=200, eta=750, lambda=1, ADWIN delta=0.002, 10 bin; leaf LP+kNN(5) và BR+bagging LR(10). ADWIN delta và cách cân dự đoán là lựa chọn implementation cần review, không tự gọi là thông số paper.

## Dữ liệu, mã nguồn và môi trường

- HF: [hanggio2105/mlhat-hyperplane](https://huggingface.co/datasets/hanggio2105/mlhat-hyperplane), revision `7171b3ab7dc6b33aeea2723174f5ff9c32dc026f`. Hai ARFF tải bằng code khớp nguyên byte với file tác giả.
- [Manifest HF chạy](../data/manifests/hyperplane-hf.json) và [tham chiếu tác giả](../data/manifests/hyperplane-reference.json) chứa checksum, commit gốc `8578941fcae1f9d55e95d6bc41b771a171a9c48d`, giấy phép CC0-1.0. Mỗi file có 50.000 dòng: **8 nhãn class0–class7 đứng đầu**, tiếp theo 30 feature att1–att30; export có dấu phẩy cuối.
- Đợt đo bắt đầu tại HEAD `6d5c122`, với code chưa commit; JSON ghi trung thực `git_dirty=true`. Snapshot chính xác nguồn đo: `artifacts/exp2_hyperplane/source_snapshot.zip`, SHA-256 `d229a956f7b7a25c07d0bac05f62121db27f911e9fc69e4ac4278562d438965b`. Nguồn đã được đối chiếu khớp commit `16bea75`; không ghi lại commit đo hồi tố.
- Môi trường đo: Python 3.13.5, NumPy 2.3.4, SciPy 1.16.3, scikit-learn 1.7.2, River 0.26.1. JSON từng run lưu môi trường/phần cứng và thời gian.
- Tổng `duration_seconds_excluding_download` của 18 run là **2443.55 giây ≈ 40.73 phút**, gồm predict/metric/learn; từng run 66.88–189.55 giây. `Total compute`/`_runtime` W&B có thể chỉ tính phiên upload số cũ; không dùng nó làm thời gian train. Máy có tải nền, không phải benchmark phần cứng được kiểm soát.

## Kết quả mean ± SD của ba seed

| Stream | Model | Example F1 ↑ | Hamming loss ↓ | Subset accuracy ↑ |
| --- | --- | --- | --- | --- |
| SynHPSud | MLHAT | 0.184405 ± 0.000254 | 0.268576 ± 0.000214 | 0.026113 ± 0.000168 |
| SynHPSud | BR+HAT | 0.173472 ± 0.005833 | 0.269277 ± 0.001015 | 0.022740 ± 0.000764 |
| SynHPSud | MLHAT-no-drift | 0.184651 ± 0.000169 | 0.268407 ± 0.000121 | 0.026233 ± 0.000223 |
| SynHPGrad | MLHAT | 0.121807 ± 0.000175 | 0.257334 ± 0.000117 | 0.015727 ± 0.000160 |
| SynHPGrad | BR+HAT | 0.116550 ± 0.012620 | 0.258973 ± 0.000798 | 0.016487 ± 0.001916 |
| SynHPGrad | MLHAT-no-drift | 0.121401 ± 0.000405 | 0.257253 ± 0.000091 | 0.015600 ± 0.000211 |

MLHAT có F1 trung bình cao hơn BR+HAT trên hai stream nhưng không thắng mọi metric: subset accuracy gradual thấp hơn baseline. Chưa kiểm định ý nghĩa thống kê, không suy rộng thành ưu thế tổng quát.

## Phân tích bổ sung — Ablation bật/tắt drift

Phân tích bổ sung của Midterm dùng **sáu cặp run** trong batch EXP-02, ghép theo stream và seed. Mỗi cặp so MLHAT bật drift với MLHAT tắt drift. Tổng số run EXP-02 vẫn là 18.

| Stream | Seed | Mở cây nền tại mẫu | Bỏ cây nền tại mẫu |
| --- | --- | --- | --- |
| SynHPGrad | 1 | 25152 | 25815 |
| SynHPGrad | 2 | 25152 | 25450 |
| SynHPGrad | 3 | 25152 | 25624 |
| SynHPSud | 1 | 25312 | 27425 |
| SynHPSud | 2 | 25344 | 25983 |
| SynHPSud | 3 | 25408 | 25905 |

Cả sáu run MLHAT có đúng một lần mở rồi bỏ cây nền ở root; **không có promote**. Run tắt drift không có sự kiện; sự kiện nội bộ River của BR+HAT chưa instrument, nên không báo count 0 cho baseline.

Bật drift đổi F1 trung bình khoảng −0.000246 (sudden) và +0.000405 (gradual) so với tắt drift. Chưa thấy lợi ích rõ từ thay cây trong cấu hình nhóm. Không quy F1 tăng sau mốc 25.000 cho promote, vì ablation cũng tăng và không run nào promote. Ảnh hưởng của dự đoán kết hợp, leaf và giả định detector còn cần kiểm chứng.

Các run bật drift đều dùng `kappa_alt = 200`. Khảo sát `kappa_alt = 100, 200, 500` nằm ngoài phạm vi kết quả Midterm và được giữ làm hướng nghiên cứu tiếp theo.

## Đối chiếu với tác giả

[CSV tham chiếu của tác giả](https://github.com/aestebant/mlhat/blob/8578941fcae1f9d55e95d6bc41b771a171a9c48d/results/per_dataset/example_f1.csv) ghi MLHAT SynHPSud=0.3269752095776777 và SynHPGrad=0.2892550611489483. **Chưa xác minh cách tổng hợp đó tương đương F1 toàn stream của nhóm**; không tính relative deviation giữa các giá trị chưa so sánh được. Chênh cấu hình/thư viện hiện là giả thuyết giải thích, chưa là nguyên nhân đã chứng minh.

## W&B links và bằng chứng public

Script đo: [reproduce_exp2.py](../experiments/reproduce_exp2.py); config: [exp2_hyperplane.json](../configs/exp2_hyperplane.json); upload số đã có: [report_exp2.py](../experiments/report_exp2.py) với cờ `--upload`. Các URL sau là run thật đã Finished:

| Dataset | Model | Seed 1 | Seed 2 | Seed 3 |
| --- | --- | --- | --- | --- |
| SynHPSud | MLHAT | [run](https://forge.coreweave.com/wandb/ml-group10-24khmt1-2026/mlhat-team10-exp2/runs/exp2c83ed081d41cb0fb) | [run](https://forge.coreweave.com/wandb/ml-group10-24khmt1-2026/mlhat-team10-exp2/runs/exp2aac5e879609efb7e) | [run](https://forge.coreweave.com/wandb/ml-group10-24khmt1-2026/mlhat-team10-exp2/runs/exp2bd953105936de12f) |
| SynHPSud | BR+HAT | [run](https://forge.coreweave.com/wandb/ml-group10-24khmt1-2026/mlhat-team10-exp2/runs/exp2c9a24d42f57ad173) | [run](https://forge.coreweave.com/wandb/ml-group10-24khmt1-2026/mlhat-team10-exp2/runs/exp280a2ce2de88730a1) | [run](https://forge.coreweave.com/wandb/ml-group10-24khmt1-2026/mlhat-team10-exp2/runs/exp2a21738073805368b) |
| SynHPSud | MLHAT-no-drift | [run](https://forge.coreweave.com/wandb/ml-group10-24khmt1-2026/mlhat-team10-exp2/runs/exp2175bed53ee351153) | [run](https://forge.coreweave.com/wandb/ml-group10-24khmt1-2026/mlhat-team10-exp2/runs/exp2d2df3fd1d2c56694) | [run](https://forge.coreweave.com/wandb/ml-group10-24khmt1-2026/mlhat-team10-exp2/runs/exp204287f774ab08ca5) |
| SynHPGrad | MLHAT | [run](https://forge.coreweave.com/wandb/ml-group10-24khmt1-2026/mlhat-team10-exp2/runs/exp2830df16218a63401) | [run](https://forge.coreweave.com/wandb/ml-group10-24khmt1-2026/mlhat-team10-exp2/runs/exp2cb829f55bf517e90) | [run](https://forge.coreweave.com/wandb/ml-group10-24khmt1-2026/mlhat-team10-exp2/runs/exp26e278b8c0356113a) |
| SynHPGrad | BR+HAT | [run](https://forge.coreweave.com/wandb/ml-group10-24khmt1-2026/mlhat-team10-exp2/runs/exp27528bfc743bd8336) | [run](https://forge.coreweave.com/wandb/ml-group10-24khmt1-2026/mlhat-team10-exp2/runs/exp22e3847a7d4a9630d) | [run](https://forge.coreweave.com/wandb/ml-group10-24khmt1-2026/mlhat-team10-exp2/runs/exp2d5742212f6d5001f) |
| SynHPGrad | MLHAT-no-drift | [run](https://forge.coreweave.com/wandb/ml-group10-24khmt1-2026/mlhat-team10-exp2/runs/exp2da20c25df6e0577d) | [run](https://forge.coreweave.com/wandb/ml-group10-24khmt1-2026/mlhat-team10-exp2/runs/exp2eb47fc013fa2e81e) | [run](https://forge.coreweave.com/wandb/ml-group10-24khmt1-2026/mlhat-team10-exp2/runs/exp2c636191998a5e176) |

Kiểm tra không đăng nhập đọc được config, metrics và dataset/model/seed của cả 18 run; tải **72 PNG** và **36 bảng** stream/drift khớp dữ liệu gốc. **66 manifest artifact** đọc được, digest measurement/snapshot/summary khớp. Blob artifact đã kiểm byte bằng SDK đăng nhập ở bước trước; public audit kiểm manifest, không tải lại mọi blob. HTML trả HTTP 200 với user-agent trình duyệt thông thường; không dùng trình duyệt tương tác để kiểm UI.

Audit được giữ nguyên ở `artifacts/exp2_hyperplane/`: `online-verification.json`, `public-verification.json`, `wandb_uploads.json`, `run_registry.csv`. 18 JSON đo, hình gốc, dữ liệu tải, source snapshot và W&B local đều giữ nguyên, bị Git ignore. Báo cáo refactor tạo mới ở thư mục riêng, không ghi đè bằng chứng này.

Các tiện ích `.py` cũ trong thư mục bằng chứng là audit lịch sử, không phải entrypoint được bảo trì. Workflow hiện hành dùng các module trong `experiments/`. Trong đợt dọn repo, SHA-256 của 400 file bằng chứng, dữ liệu, config, core và phần A/B đã được đối chiếu trước–sau; giá trị không đổi.

Kiểm tra số liệu: đủ 18 identity; 50.000 mẫu/run; grid 197/1.000 điểm; F1 cả stream khớp 50 cửa sổ 1.000 mẫu không chồng lấn; mean/SD được đối chiếu với NumPy; 7.182 điểm curve trung bình khớp ba seed. **Không lưu prediction từng mẫu**, nên không thể tính lại độc lập mọi metric/fading từ prediction gốc mà không chạy lại.

## Lệnh sử dụng

Chỉ có **hai entrypoint** cần chạy: `reproduce_exp2.py` để đo và `report_exp2.py` để kiểm tra/xuất báo cáo/upload. Ba module còn lại là dependency nội bộ: `exp2_data.py` đọc và kiểm ARFF từ HF, `exp2_evidence.py` kiểm chứng batch/thống kê, `exp2_upload.py` quản lý W&B và run ID. Giữ chúng riêng vì có trách nhiệm độc lập. Script chuẩn bị mirror HF một lần đã được bỏ sau khi dataset public được ghim; dữ liệu gốc, dataset card và snapshot lịch sử vẫn giữ nguyên.

Chạy từ repo root, Python >=3.11:

```powershell
python -m pip install -e ".[dev,online,data,tracking,plots]"
python -m experiments.report_exp2 --validate-only
python -m experiments.report_exp2
python -m experiments.report_exp2 --plan-upload
```

Hai lệnh report đầu đọc batch cũ và ghi kết quả dẫn xuất vào `artifacts/exp2_reports/`; `--plan-upload` chỉ đọc ledger, in ID cố định và trạng thái đã ghi, **không gọi W&B**. Report/upload không import core hoặc runner train. Input mặc định là `artifacts/exp2_hyperplane/`; có thể chọn `--directory`, nhưng `--output-dir` phải tách khỏi input.

Batch đã online đủ 18 run nên **không cần upload lại**. Khi thực sự cần upload/tiếp tục upload:

```powershell
python -m wandb login
python -m experiments.report_exp2 --upload --wandb-entity ml-group10-24khmt1-2026 --wandb-project mlhat-team10-exp2
```

Team/project được cố định, không tạo Team mới. Run ID dựa trên JSON gốc; retry dùng lại ID, kiểm server, không tạo run trùng. JSON gốc không sửa; ledger/log upload mới ở `artifacts/exp2_reports/`, kế thừa ledger cũ. Public đã xác minh không tự bị đổi thành false khi chỉ xác nhận lại run đã xong. Không đưa key vào chat/Git; nhập key trực tiếp trong terminal.

Chỉ khi chủ động cần một **đợt đo mới**, dùng output mới chưa tồn tại (không chạy các lệnh này để upload):

```powershell
python -u -m experiments.reproduce_exp2 --manifest data/manifests/hyperplane-hf.json --output-dir artifacts/exp2_rerun
```

Chọn một tổ hợp khi cần:

```powershell
python -u -m experiments.reproduce_exp2 --manifest data/manifests/hyperplane-hf.json --output-dir artifacts/exp2_rerun --dataset SynHPSud --model MLHAT --seed 1
```

Runner từ chối ghi đè JSON đã có trước khi tải dữ liệu hoặc tạo model. `--resume` chỉ skip nếu commit/config/checksum/độ dài khớp; không ép batch cũ chạy tiếp ở commit mới. Validator report đang ghim snapshot của **batch 18 run này**; một batch mới cần provenance/snapshot và kiểm định riêng trước khi báo cáo, không trộn số vào batch cũ.

## Bàn giao slide và việc còn thiếu

- Các slide về drift và thực nghiệm sử dụng [quyết định về cơ chế drift](decisions.md), [giao thức chung](experiment-protocol.md), [6 run EXP-01 trên Yeast](../README.md#exp-01--kết-quả-yeast) và bằng chứng EXP-02 trong báo cáo này. Phần phân tích bổ sung trình bày ablation và quan sát về cây dự phòng.
- Hình gốc để trích dẫn vẫn ở `artifacts/exp2_hyperplane/*_f1.png`; bản xuất mới ở `artifacts/exp2_reports/`. Mỗi đường là trung bình ba seed; hai đường gần trùng có thể che nhau, cần kèm bảng. Mỗi hình/số trên deck có URL W&B cụ thể; hình do nhóm tạo, notes mỗi slide và trích dẫn IEEE tại slide.
- **Cùng A/B xác minh evaluator và phép tổng hợp paper**, rồi mới so relative deviation; thực hiện phép chạy code tác giả độc lập theo yêu cầu môn, không sao chép core.
- Rà warning ADWIN, lỗi cây nền, trọng số kết hợp và giải thích sai lệch bằng bằng chứng; chưa quy cho seed/version khi chưa kiểm tra. Kết quả 0 promote là quan sát phải báo trung thực.
- EXP-01 và EXP-02 hiện có hai project công khai trong cùng Team. Nhóm cần đối chiếu cách tổ chức này với yêu cầu “one public W&B project” trong hướng dẫn môn.
