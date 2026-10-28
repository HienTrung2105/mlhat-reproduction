# Interface A / B / C (khung đề xuất, phải thống nhất từ paper)

```text
         Input x_t, truth y_t
                  |
                  v
        [B] Dataset adapter
                  |
                  v
         prequential loop
         predict(x_t)  FIRST
                  |
                  v
    [A] MLHAT tree traversal <----> [B] leaf predictor
                  |
                  v
        model.learn(x_t, y_t)
                  |
                  v
    [A] node update / split <-----> [C] drift monitor / adaptation
                  |
                  v
    [B] Metrics / W&B logging
```

## Dữ liệu thống nhất tạm thời

- `x`: 1-D `numpy.float64`, một mẫu.
- `y`: 1-D binary vector `numpy.int64`, `shape=(n_labels,)`.
- `predict_one(x)` -> 1-D binary vector n_labels.
- `learn_one(x,y)` -> update một mẫu.
- **Interface demo không thay thế đặc tả của paper** (ví dụ categorical handling, criterion, drift updates cần nhóm quyết định).

## Ghép code

1. A/B/C review giấy thiết kế interface trước khi implement.
2. B kiểm tra dữ liệu, metric và test-then-train loop.
3. A chịu trách nhiệm split/node stats; C báo trước nếu drift cần giữ thống kê/bản sao cây.
4. C cùng A test trường hợp drift; B log và kiểm tra reproducibility.
5. Tạo tests trên stream rất nhỏ, sau đó mới chạy paper dataset và W&B.
