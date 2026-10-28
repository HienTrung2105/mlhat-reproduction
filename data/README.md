# Dataset handling

Thầy yêu cầu tải dữ liệu **tự động qua Hugging Face hoặc Kaggle** mỗi khi chạy; không nộp dữ liệu chuẩn bị thủ công làm điều kiện bắt buộc trước đó.

- Chọn dataset ID và license/schema dựa vào paper và nguồn thực có.
- `src/mlhat/data.py` có một generic HF loader, chưa có bộ chuyển đổi multi-label cho dataset MLHAT đã chốt.
- Tất cả dataset thật phải document nguồn, phiên bản, đặc trưng, thứ tự stream, labels, preprocess và khả năng tái lập.
- Thư mục `data/raw` và `data/processed` được `.gitignore`, không commit tập dữ liệu lớn.
