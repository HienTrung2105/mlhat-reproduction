# Ghi chú paper MLHAT

## Bài toán và tiền nhiệm

Mỗi bước nhận đặc trưng x và vector nhiều nhãn nhị phân y. Ví dụ một bài báo có thể đồng thời có nhãn AI và y tế. Mô hình dự đoán rồi học từ nhãn thật. Phân phối nhãn và quan hệ giữa đặc trưng với nhãn có thể đổi theo thời gian.

Hoeffding Tree tăng trưởng từ thống kê online và giữ quyết định chia đã tạo. HAT thêm theo dõi drift và thay cây con. Binary Relevance + HAT dùng một cây cho mỗi nhãn. MLHT hỗ trợ nhiều nhãn ở lá nhưng thiếu cơ chế thích nghi drift của MLHAT.

## Cây và phép chia (§3.1–3.4)

Cây bắt đầu bằng một lá gốc. Một mẫu đi theo điều kiện trên các đặc trưng đến lá, nơi bộ dự đoán và thống kê được cập nhật sau khi biết nhãn thật. Khi đủ bằng chứng, thay lá bằng nút chia và hai lá con. Mỗi lá biểu diễn một vùng đặc trưng và học từ nhiều mẫu.

Với W là tổng trọng số và W_l là trọng số mẫu có nhãn l, đặt p_l = W_l/W. Entropy Eq. (5) cộng entropy Bernoulli theo từng nhãn. Dù paper giới thiệu phân phối Bernoulli đa biến, công thức này là tổng entropy biên. Gain của một phép chia bằng entropy cha trừ trung bình có trọng số của entropy hai con. Ví dụ entropy cha 1.0 bit, hai con có entropy 0.3 và 0.5 bit, mỗi con nhận nửa trọng số thì gain là 0.6 bit. Đặc trưng số dùng thống kê Gaussian theo feature và trạng thái nhãn để ước lượng phân bố quanh ngưỡng. Paper còn mô tả đếm tần suất cho đặc trưng phân loại.

Sau grace period, so ứng viên tốt nhất và nhì theo gain, lấy hai ứng viên từ hai feature khác nhau như §3.4. Chấp nhận nếu G1 − G2 > epsilon_spl của Eq. (7). Ghi rõ cách biểu diễn tập labelset đã quan sát khi tính biên. Nếu chưa đủ hai ứng viên, tiếp tục tích lũy trong phạm vi benchmark đã chọn. Biên giảm khi W tăng. Trình bày đây là kiểm tra trên ước lượng gain từ stream.

## Lá, bootstrap và drift (§3.3, §3.5–3.6)

Lá thuần dự đoán labelset duy nhất đã quan sát. Lá chưa thuần và còn ít mẫu dùng kNN + Label Powerset. Lá nhiều mẫu dùng bagging Logistic Regression + Binary Relevance khi vẫn chưa chia. Bộ dự đoán phức tạp được cập nhật sớm khi lá đủ điều kiện học. Trọng số Poisson dùng cho học online và thống kê cây. Bộ bagging có cơ chế Poisson riêng, tránh áp trọng số bootstrap hai lần cho bộ đó.

ADWIN theo dõi Hamming loss từ dự đoán trước khi học. Cảnh báo mở cây con dự phòng. Khi đủ dữ liệu, so lỗi hai cây với epsilon_alt trong Eq. (16) và Algorithm 1. Cây mới tốt hơn đủ mức thì thay, kém hơn đủ mức thì bỏ, còn sát nhau thì giữ cả hai. Lá nền đủ dữ liệu có thể góp xác suất theo lỗi theo Algorithm 3. Phần văn bản §3.5 dùng delta_alt ở một số điều kiện, trong khi Algorithm 1 dùng epsilon_alt. Nhóm theo thuật toán và Eq. (16).

## Chi phí và bằng chứng (§4)

Dự đoán cần định tuyến trên đường đi và chạy bộ phân loại ở lá. Học thêm chi phí cập nhật thống kê và duyệt ứng viên khi thử chia. Nhiều cây nền đồng thời làm tăng thời gian và bộ nhớ. Khi trình bày biểu thức độ phức tạp của tác giả ở §4.6, định nghĩa rõ độ sâu, số nhãn, số đặc trưng, số ứng viên và chi phí học ở lá.

- RQ1 đánh giá cạnh tranh trên nhiều bộ dữ liệu và baseline.
- RQ2 so với các cây Hoeffding, đặc biệt BR+HAT. Phép so sánh đồng thời thay nhiều thành phần.
- RQ3 xem quá trình phục hồi theo thời gian với drift. Drift incremental là trường hợp khó trong kết quả tác giả.
- Phạm vi nhóm dùng Yeast và hai stream Hyperplane. Kết quả trên các bộ đã chọn không đại diện cho trung bình toàn benchmark.

Nguồn. [Esteban et al., arXiv v1](https://arxiv.org/html/2410.20242v1). Kế hoạch nhóm nằm trong [experiment-protocol.md](experiment-protocol.md).
