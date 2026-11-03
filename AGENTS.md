# AGENTS.md — MLHAT reproduction

Quy tắc làm việc cho coding agent trong repo này. Đọc [README.md](README.md) và cây file thực tế trước khi sửa. Giao tiếp với nhóm bằng tiếng Việt.

## Bối cảnh và nguồn ưu tiên

- Nhóm 3 sinh viên môn Nhập môn Máy học đã chọn bài báo *Hoeffding adaptive trees for multi-label classification on data streams* (Esteban et al., 2024): [paper](https://arxiv.org/abs/2410.20242), [DOI](https://doi.org/10.1016/j.knosys.2024.112561).
- Mục tiêu: hiểu và tự triển khai đóng góp MLHAT, tái tạo một phần thí nghiệm, định lượng và giải thích sai lệch so với paper.
- **Trạng thái hiện tại:** repo có hạ tầng đánh giá và baseline demo synthetic; chưa có MLHAT hoàn chỉnh hay run tái tạo. Paper được duyệt, mã nhóm, deadline, vai trò thực tế, dataset và scope chưa được xác nhận trong repo. Xem [docs/decisions.md](docs/decisions.md).
- Thứ tự nguồn: **hướng dẫn Moodle/giảng viên mới nhất → paper → quyết định đã ghi của nhóm → [repo tác giả](https://github.com/aestebant/mlhat) để kiểm chứng**. Báo lại nếu nguồn mâu thuẫn; đừng tự coi kế hoạch đề xuất là quyết định chính thức.

## Hai bài nộp Midterm

1. **URL repo GitHub công khai.** Repo tiếp tục phát triển sau khi nộp link; commit phải thể hiện công việc thật của nhóm.
2. **Một file `ID.pptx`** với mã nhóm đã đăng ký: 20–40 slide tính cả appendix, dưới 20 MB, speaker notes trên **mọi** slide (tối đa 500 từ/slide). Deck phải gồm đọc hiểu paper **và** tái tạo thực nghiệm. Dùng hình/bảng nhóm tự tạo; trích dẫn IEEE ở cuối từng slide sử dụng nguồn, không làm slide tài liệu tham khảo riêng. Đối chiếu số liệu thực với paper và giải thích sai lệch; mỗi số liệu/hình thực nghiệm phải truy về code, config và W&B run công khai.

Final poster là giai đoạn sau. Không tuyên bố đã nộp, đã chạy hay đã tái tạo khi chưa có bằng chứng.

## Liêm chính và tái lập

- **Tự triển khai lõi MLHAT từ mô tả, công thức của paper.** Không sao chép, port, dịch hay viết lại code lõi của tác giả hoặc nhóm khác. Repo tác giả chỉ để đọc/chạy/đối chiếu độc lập; ghi rõ mọi phép so sánh đó.
- Có thể dùng thư viện thông dụng và hạ tầng không mới, nhưng ghi nguồn/phần tái sử dụng trong README. Course yêu cầu Python >=3.9; cấu hình repo hiện yêu cầu >=3.10. Không phụ thuộc vào package có sẵn MLHAT.
- Ghi seed, phiên bản thư viện, môi trường/phần cứng cần thiết, hyperparameter và giao thức đánh giá. Dùng docstring cho hàm quan trọng, nêu mục đích, tham số, giá trị trả về.
- Tải dataset thí nghiệm **bằng code từ Hugging Face hoặc Kaggle khi chạy**; ghi nguồn, phiên bản, schema nhãn, thứ tự stream và tiền xử lý. Không yêu cầu chép dataset thủ công.
- Mỗi run báo cáo phải có W&B **public** ghi config và metric; README ánh xạ `experiment → script/config → run URL → kết quả một dòng`. Không bịa số đo, hình, citation hay run.
- Không commit secret, API key, dataset lớn, cache, checkpoint hoặc log W&B local.

## Trước khi viết thí nghiệm

- Đọc phần Experiments của paper và điền [docs/experiment-protocol.md](docs/experiment-protocol.md): claim, dataset, baseline, metric chính xác, config, seed/số lần chạy, output và hình/bảng paper cần kiểm chứng.
- Nhóm phải chốt **Full** hoặc **Core**. Nếu chọn Core, kế hoạch cần ít nhất hai thí nghiệm quan trọng, gồm so sánh trung tâm của paper, và một phân tích bổ sung do nhóm thiết kế. Hiện đây là **phương án**, chưa phải scope được duyệt.
- Nếu paper yêu cầu online/prequential, dự đoán trước khi học từ mẫu đó. Kiểm tra schema, split, định nghĩa và thang metric trước khi so số liệu; chỉ tính relative deviation cho các giá trị so sánh được.
- Khi công thức hoặc thứ tự cập nhật chưa rõ, nêu rõ mục/định nghĩa cần xác minh và đề xuất kiểm tra nhỏ; không tự sáng tạo chi tiết khoa học.

## Phối hợp A/B/C (đề xuất, chưa gán tên)

| Vai trò | Phần chính | Midterm slides đề xuất |
| --- | --- | --- |
| A | Tree, node statistics, split, lý thuyết; hỗ trợ phân tích bổ sung | 7–13, 18–20 |
| B | Leaf prediction, data, metric, W&B, thí nghiệm 1; ghép deck | 1–6, 14–15, 21–22 |
| C | Drift adaptation, thí nghiệm 2, đối chiếu sai lệch | 16–17, 23–30 |

Cả ba đọc paper, review chéo và hiểu toàn bộ hệ thống. C không ôm hết tích hợp/thí nghiệm; mỗi người làm cả code/nghiên cứu lẫn slide của mình.

## Cách agent làm việc

1. Kiểm tra repo, README và trạng thái Git; phân biệt **yêu cầu môn**, **fact từ paper**, **quyết định nhóm** và **đề xuất agent**. Ghi việc chưa chốt vào `docs/decisions.md`.
2. Sửa từng phần nhỏ, giữ công việc của thành viên khác, dùng nhánh/PR để review. Không force-push hay tự đổi giao diện chung khi chưa thống nhất.
3. Viết code rõ ràng với type hints khi hữu ích, seed/config tường minh và test có ý nghĩa. Sau thay đổi code, chạy test liên quan và báo kết quả thực.
4. Chỉ tạo module, script, thư mục, hình hoặc slide khi có nội dung thật. Không biến placeholder hay demo thành tuyên bố MLHAT đã hoàn thành.
5. Khi bàn giao việc lớn, nêu file đã đổi, lệnh/test đã chạy, kết quả, phần còn rủi ro và việc tiếp theo của A/B/C.
