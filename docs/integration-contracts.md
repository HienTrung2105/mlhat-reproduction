# Giao diện tích hợp A / B / C

## Sơ đồ luồng

```text
[B] Dataset adapter -> (x_t, y_t) -> prequential evaluator
                                         |
                                  predict_one(x_t)
                                         v
                              [A] MLHAT tree traversal
                                         |
                               [B] leaf predictor
                                         |
                              prediction y_hat_t
                                         v
                              [B] update metrics
                                         |
                               learn_one(x_t, y_t)
                                         v
                         [A] node statistics and split
                                <-> [C] drift adaptation
```

## 1. Dạng dữ liệu thống nhất

- `x`: một mẫu dưới dạng `numpy.ndarray`, một chiều, số thực `float64`, không rỗng và không chứa `NaN`/vô cực. Số chiều và thứ tự đặc trưng phải giữ ổn định trong một lần chạy.
- `y`: vector nhãn nhị phân `numpy.ndarray`, một chiều, kiểu `int64`, shape `(n_labels,)`. Số nhãn và thứ tự nhãn phải giữ ổn định trong một lần chạy.
- `predict_one(x)` trả về vector nhị phân kiểu `int64`, shape `(n_labels,)`.
- Kiểm tra mẫu đầu vào dùng `validate_sample` trong `src/mlhat/interfaces.py`; dữ liệu sai shape hoặc nhãn không phải 0/1 bị từ chối bằng `ValueError`.
- B chịu trách nhiệm biến dữ liệu nguồn thành schema này. Nếu có đặc trưng phân loại hoặc thiếu dữ liệu, B phải chọn và ghi rõ cách mã hóa/xử lý trong protocol; không âm thầm biến nhãn phân loại thành số có thứ tự.

## 2. API công khai của mô hình online

Giữ API đã khai báo trong `src/mlhat/interfaces.py`:

```python
model = MLHATTree(n_labels=n_labels)
y_hat = model.predict_one(x)  # NDArray[np.int64], shape (n_labels,)
model.learn_one(x, y)          # cập nhật một mẫu; trả về None
```

`predict_one` chỉ dự đoán, không cập nhật thống kê học. `learn_one` nhận đúng một mẫu đã có nhãn. Không đổi sang dictionary hoặc thêm đối số bắt buộc vào API này nếu chưa cập nhật hợp đồng và các caller cùng lúc.

### Constructor được runner EXP-01 gọi hiện tại

Đoạn trên mô tả interface online tối thiểu. Lớp cụ thể trong `src/mlhat/tree.py` hiện cần thêm số feature, config và seed:

```python
model = MLHATTree(
    n_labels=Y.shape[1],
    n_features=X.shape[1],
    config=config["mlhat"],
    seed=seed,
)
```

Phân biệt interface của `predict_one`/`learn_one` với constructor cụ thể này khi viết caller mới.

Bộ dự đoán tại leaf cũng nhận `x`, trả vector `y_hat` cùng shape/kiểu dữ liệu trên, và cập nhật bằng `learn_one(x, y)`. B hiện sở hữu implementation của leaf predictor; A chỉ gọi qua interface.

## 3. Thứ tự test-then-train

Với từng mẫu trong stream, evaluator thực hiện đúng thứ tự:

```python
y_hat = model.predict_one(x)
metric.update(y, y_hat)
model.learn_one(x, y)
```

Metric của mẫu `t` dùng dự đoán từ trạng thái mô hình trước khi học mẫu `t`. Mọi tín hiệu drift cũng phải được tính từ dự đoán trước cập nhật, theo đúng định nghĩa lỗi và thứ tự cập nhật trong paper.

## 4. Hợp đồng nội bộ cho drift giữa A và C

- Mỗi node thích nghi có một `AdaptiveDriftManager` do C triển khai.
- Trong lúc xử lý `learn_one`, A cung cấp cho manager lỗi dự đoán trước cập nhật của cây đang hoạt động và, nếu đã tồn tại, nhánh nền.
- Manager trả về một `DriftAction`: `NONE`, `START_BACKGROUND`, `PROMOTE_BACKGROUND`, hoặc `DISCARD_BACKGROUND`.
- C chịu trách nhiệm detector, điều kiện và chọn action theo paper. A chịu trách nhiệm áp dụng action vào cấu trúc node/cây. Nhánh hoạt động và nhánh nền phải được cập nhật theo thứ tự paper; chi tiết này được ghi trong code/test của hai module khi tích hợp.
- Enum/action ở trên chỉ là hợp đồng phần mềm. Nó không thay thế thuật toán drift, không tự quy định ngưỡng ADWIN hoặc định nghĩa lỗi.

## 5. Ranh giới trách nhiệm

- **B — dữ liệu/evaluation:** tải và chuyển đổi dữ liệu, giữ schema ổn định, chạy vòng test-then-train, tính metric, ghi W&B; triển khai leaf predictor.
- **A — cây/splitting:** đi qua cây, duy trì thống kê tại node, đánh giá phép chia và cập nhật cấu trúc cây theo tiêu chí trong paper; kiểm thử các trường hợp split/no-split.
- **C — drift:** triển khai bộ theo dõi drift và chính sách nhánh dự phòng/thay thế theo paper; trả `DriftAction` cho A áp dụng.

## 6. Cách tích hợp

1. A triển khai và kiểm tra cây/splitting với các vector NumPy nhỏ, tự tạo trong test.
2. B làm leaf predictor và evaluator tuân theo API/schema ở trên.
3. C làm drift manager tuân theo `DriftAction`; A/C kiểm tra luồng tạo, giữ, thay thế hoặc bỏ nhánh nền.
4. Cả nhóm kiểm tra stream nhỏ theo đúng thứ tự predict -> metric -> learn.
5. Chỉ sau đó mới chạy dataset paper, thí nghiệm và W&B. Dữ liệu thô/tải về không commit; xem [README gốc](../README.md) về quy tắc lưu dữ liệu.

Repo tác giả chỉ dùng để đọc/chạy/đối chiếu theo quy tắc học thuật của nhóm; không sao chép lõi thuật toán vào implementation nhóm.

## 7. API xác suất dùng nội bộ

- `predict_proba_one(x)` trả vector `float64` shape `(n_labels,)`, mỗi phần tử nằm trong `[0, 1]`. Dùng xác suất này để kết hợp dự đoán của cây chính và cây nền.
- `predict_one(x)` vẫn là API nhị phân mà evaluator dùng; ngưỡng hiện tại là `0.5` theo cấu hình.
- Trong `learn_one(x, y)`, tính tín hiệu lỗi từ dự đoán trước cập nhật. Không dùng nhãn hiện tại để dự đoán chính mẫu đó.
- Đây là phần bổ sung cho giao diện xác suất trong `src/mlhat/interfaces.py`; phần drift và cây nền vẫn do C triển khai.
