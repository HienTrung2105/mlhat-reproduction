# MLHAT — Independent Paper Reproduction

> **Trạng thái:** Repository skeleton / khung khởi tạo. **Chưa có MLHAT được triển khai và chưa tái tạo kết quả paper.**
>
> **Nhóm:** 3 thành viên (A/B/C là vai trò tạm). **Môn:** Nhập môn Máy học.

Bài báo: **Hoeffding adaptive trees for multi-label classification on data streams** (2024).
- Paper: https://arxiv.org/abs/2410.20242
- DOI: https://doi.org/10.1016/j.knosys.2024.112561
- Repo của tác giả (chỉ dùng để kiểm chứng/đối chiếu; **không sao chép lõi thuật toán**): https://github.com/aestebant/mlhat

## 1. Hai bài nộp ở Midterm

| Moodle | Nộp gì? | Yêu cầu chính |
| --- | --- | --- |
| **GitHub repository** | **Một URL GitHub công khai** | Code nhóm tự viết, script thực nghiệm, README, commit thực tế; có thể tiếp tục commit sau khi nộp URL theo hướng dẫn Moodle |
| **Midterm Slide** | **`ID.pptx`** (`ID` là mã nhóm thật) | **20–40 slide**, **<20 MB**, ghi chú speaker notes **mỗi slide (<=500 từ)**, citations IEEE ngay trên slide, sơ đồ tự vẽ; đọc hiểu + thực nghiệm tái tạo có kết quả thật |

Phần **Final Poster** làm sau; không phải mục tiêu của repo skeleton hiện tại.

## 2. Bắt đầu nhanh (Windows / VS Code)

**Bước 1 — Tạo môi trường:** cần Python **3.10 hoặc 3.11** (khuyên dùng 3.11).

```powershell
cd mlhat-reproduction
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Nếu PowerShell chặn việc activate, dùng `cmd.exe` và `.venv\Scripts\activate.bat`, hoặc chạy trực tiếp `.venv\Scripts\python.exe`.

**Bước 2 — Kiểm tra khung có chạy:**

```powershell
python -m pytest -q
python -m experiments.smoke_baseline
```

`smoke_baseline` chạy **baseline demo trên dữ liệu synthetic**, chỉ để kiểm tra pipeline/metric. **Nó không phải MLHAT, không phải thí nghiệm của paper và kết quả không được dùng để tuyên bố tái tạo.**

**Bước 3 — Chỉ khi cần kết nối HF datasets/W&B:**

```powershell
python -m pip install -e ".[data,tracking]"
# Có thể đăng nhập W&B bằng lệnh `wandb login` trên máy của nhóm.
# Không commit token hoặc .env thật.
```

Để chạy demo và ghi lên W&B, cần tài khoản đã login và project được cấu hình phù hợp:

```powershell
python -m experiments.smoke_baseline --log-wandb --wandb-project mlhat-team-demo
```

**Kết quả demo không phải kết quả paper.** Các thí nghiệm chính hiện vẫn là placeholder đợi nhóm xác minh giao thức Section 4.

Trên macOS/Linux thay tạo venv và activation bằng:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
```

## 3. Ai chịu trách nhiệm phần nào?

| Vai trò | Nghiên cứu / code chính | Midterm slide (kế hoạch đề xuất) | Phối hợp |
| --- | --- | --- | --- |
| **A — Tree & Theory** | Hoeffding Tree/MLHAT nodes, statistics, multi-label split, toán, độ phức tạp | **7–13, 18–20** | Review thuật toán, thiết kế phân tích bổ sung |
| **B — Multi-label & Evaluation** | Leaf prediction, data loading, metrics, prequential evaluation, W&B | **1–6, 14–15, 21–22** | Chủ trì thí nghiệm 1; **ghép file PPTX**, kiểm tra notes/citations |
| **C — Drift & Reproduction** | ADWIN/adaptation, nhánh dự phòng, tích hợp, thí nghiệm drift | **16–17, 23–30** | Chủ trì thí nghiệm 2; tổng hợp sai lệch, **không làm toàn bộ thực nghiệm một mình** |

**Cả ba đọc toàn bộ paper và review chéo.** Xem checklist chi tiết tại [`docs/team-roles.md`](docs/team-roles.md), slide plan tại [`docs/midterm-slides.md`](docs/midterm-slides.md).

## 4. Cấu trúc repo

```text
mlhat-reproduction/
├── AGENTS.md                     # Context + quy tắc AI coding agent
├── README.md                     # Hướng dẫn nhóm và người đánh giá
├── CONTRIBUTING.md               # Branch / PR / commit convention
├── pyproject.toml                # Cài package + dependencies
├── src/mlhat/                   # Core algorithm (hiện là skeleton có TODO)
│   ├── interfaces.py            # Interface dùng chung giữa A/B/C
│   ├── tree.py                  # A: cây MLHAT
│   ├── splitting.py             # A: tiêu chí chia nhánh
│   ├── leaves.py                # B: multi-label leaf predictor
│   ├── drift.py                 # C: drift/adaptive branches
│   ├── evaluation.py            # B: vòng test-then-train + metric
│   └── data.py                  # B: loader dữ liệu HF (generic)
├── experiments/
│   ├── smoke_baseline.py        # DEMO kiểm tra hệ thống, KHÔNG PHẢI MLHAT
│   ├── reproduce_exp1.py        # TODO: thí nghiệm 1 từ paper
│   ├── reproduce_exp2.py        # TODO: thí nghiệm 2 từ paper
│   └── sensitivity.py           # TODO: phân tích thêm tự thiết kế
├── configs/                    # Config đã chọn, không bịa paper defaults
├── tests/                      # Test cho logic đã được triển khai
├── docs/                       # Paper notes, kế hoạch, giao thức, quyết định
├── results/                    # README + run registry, chưa có kết quả giả
├── slides/                     # Slide plan / file PPTX khi nhóm tạo
├── figures/                    # Hình do nhóm tự tạo
├── data/                       # Hướng dẫn nguồn, không commit dataset lớn
└── .github/workflows/          # CI chạy pytest
```

## 5. Việc đầu tiên khi nhóm vừa chọn paper

1. Xác minh paper được duyệt trên Moodle, team ID, deadline. Ghi vào `docs/decisions.md`.
2. Cả 3 đọc **Abstract -> Introduction -> Conclusion -> Related Work -> Experiments -> Method**; viết `docs/paper-notes.md`.
3. Điền **giao thức thí nghiệm có căn cứ từ bài báo** vào `docs/experiment-protocol.md`: dataset, baseline, metric, tham số, hình/bảng đối chiếu, số lần chạy/seed.
4. Chốt **Core Scope** hoặc Full Scope; Core phải có **ít nhất hai thí nghiệm quan trọng (bao gồm phép so sánh chính)** và **phân tích bổ sung**.
5. Thiết kế interface A/B/C, làm unit tests rồi mới triển khai lõi từ paper, không copy code tác giả.
6. Chạy thật và theo dõi trên **W&B public**; chỉ thêm kết quả vào [`results/run-registry.csv`](results/run-registry.csv) sau khi run có thật.
7. Hoàn thiện slide và speaker notes; số liệu trong slide phải truy ngược được về W&B run công khai.

## 6. Hướng dẫn W&B / dữ liệu

- Mọi thí nghiệm paper phải tải dữ liệu **tự động từ Hugging Face hoặc Kaggle** trong lần chạy. `src/mlhat/data.py` có generic HF loader, **chưa xác định dataset ID và schema cụ thể**.
- Đăng nhập W&B bằng `wandb login` (không viết API key vào repo). Config, metrics và run URL cần truy xuất được.
- W&B project phải **public** theo yêu cầu môn. Repo GitHub public không tự làm W&B public.
- Điền bảng ánh xạ script -> W&B run -> mô tả kết quả trong README **khi có run thật**. Hiện tại: **chưa có**.

### Reproduction runs (chưa có — chỉ cập nhật sau khi chạy thành công)

| Experiment | Script | W&B run (public) | Kết quả |
| --- | --- | --- | --- |
| _Chưa thực hiện_ | — | — | — |

## 7. Nguồn và quy tắc đạo đức học thuật

Đóng góp lõi MLHAT phải do nhóm tự triển khai từ paper. Mã của tác giả chỉ đọc/chạy để **kiểm chứng và so sánh** chứ **không sao chép/port**. Các thư viện tiêu chuẩn sử dụng trong hạ tầng và baseline phải được ghi nguồn trong README. Không bịa số liệu, không chụp lại hình paper làm hình nhóm, không nhét token vào GitHub.

**Cho AI agent:** trước khi sửa code, đọc `AGENTS.md` và các `docs/` liên quan. Mỗi thay đổi cần giải thích giả định, test đã chạy, những gì chưa được triển khai.

## 8. Tạo repo GitHub và đẩy code lên

Tạo một repository **trống** trên GitHub, ví dụ `mlhat-reproduction` (đừng chọn tự tạo README để tránh conflict), sau đó mở terminal trong folder này:

```bash
git init
git add .
git commit -m "chore: initialize MLHAT reproduction scaffold"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/mlhat-reproduction.git
git push -u origin main
```

Thay `YOUR_USERNAME` bằng tài khoản/organization thật. **GitHub clone là thao tác tải repo về máy; còn tình huống này là tạo folder local rồi push lên GitHub.** Từ máy thành viên khác mới dùng `git clone URL`.
