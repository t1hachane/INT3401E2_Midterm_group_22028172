# INT3401E2_Midterm_group_22028172

Đây là repo của bài tập lớn giữa kì Dự báo lượng mưa của lớp Trí tuệ nhân tạo INT3401E2. Báo cáo của nhóm có thể tìm thấy tại: https://drive.google.com/file/d/1r7KZrQ73_2dzyn5Dw8MtALSJDeWCpgVa/view?usp=sharing. Nhóm gồm những thành viên như sau: 
- Lê Xuân Hùng - MSV: 22028172 (nhóm trưởng)
- Tăng Vĩnh Hà - MSV: 22028129
- Ngô Tùng Lâm - MSV: 22028092
- Vũ Nguyệt Hằng - MSV: 22028079

Nhóm dự đoán lượng mưa 3h tới với đầu vào là lượng mưa 6h trước, train riêng biệt giữa 4 tháng dữ liệu đã cung cấp. Cấu trúc repo như sau
- Folder `EDA`: gồm các notebook nhóm đã sử dụng cho việc phân tích dữ liệu nhằm hiểu dữ liệu và có insight cho việc chọn, xử lý dữ liệu sau này. Gồm 2 file là `aws-eda` dùng để phân tích biến AWS cần dự đoán và `era5-eda` để phân tích dữ liệu biến các biến ERA5
- Folder `Preprocess`: gồm các notebook cho phần xử lý dữ liệu:
  - `convert_image_to_tab_data`: chuyển dữ liệu ảnh sang dạng bảng
  - `expanding-window-train-test-val`: chia dữ liệu thành tập train, val, test theo phương pháp expanding window
  - `handle-missing-value-knn-impute`: xử lý dữ liệu AWS bị thiếu bằng phương pháp KNN
  - `handle-missing-value-kriging-impute`: xử lý dữ liệu AWS bị thiếu bằng phương pháp Kriging
  - `clustering-input-data`: phân cụm dữ liệu đầu vào cho việc thử nghiệm cải tiến mô hình sau này
- Folder `Models`: gồm các notebook cho việc train model
  - Subfolder `ML`: gồm các mô hình machine learning là `RandomForest` và `XGBoost`. Do trong cài đặt hồi quy chỉ có thể có 1 đầu ra nên nhóm sử dụng wrapper `MultiOutputRegressor` bao ngoài 2 mô  hình để phù hợp với bài toán dự đoán
      - `random-forest_multi-output-regressor`: mô hình `RandomForest`
      - `xgboost_multi-output-regressor`: mô hình `XGBoost`
  - Subfolder `DL`: gồm các mô hình deep learning là `LSTM` và `ConvLSTM`, ứng với 2 notebook là `lstm` và `convlstm`
  - Subfolder `DL_Enhanced`: gồm các mô hình deep learning được cải tiến
      - Cải tiến LSTM thành bài toán dự đoán theo dữ liệu đầu vào được phân cụm: `Clustered_LSTM`
      - Cải tiến LSTM thành mô hình 2 bước: `HybridLSTMModel`
- Folder `Plot_rain`: script vẽ bản đồ mưa từ một output có format cố định.

Link các dataset được tiền xử lý được đặt trên kaggle:
- Dataset chỉ có AWS: https://www.kaggle.com/datasets/hunggg123/ai-original-data-only-aws/data
- Dataset có AWS và các ERA5 feature được xử lý missing bằng KNN: https://www.kaggle.com/datasets/hunggg123/ai-dataknnimputedataset-k-fold
- Dataset có AWS và các ERA5 feature được xử lý missing bằng Kriging: https://www.kaggle.com/datasets/eunjungham/dataset-k-fold
- Dict lưu số cụm được phân tương ứng với từng điểm: https://www.kaggle.com/datasets/eunjungham/ai-data4clustering-dict/data

Contribution: các thành viên trong nhóm có đóng góp như sau
- Lê Xuân Hùng:
    - Tìm hiểu các phương pháp impute 
    - Cài đặt LSTM
    - Code phân cụm dữ liệu đầu vào cho việc cải tiến
    - Tìm hiểu các phương pháp phân cụm và lên ý tưởng cho mô hình cải tiến
    - Code 2 mô hình cải tiến clustered LSTM và hybrid LSTM
    - Viết báo cáo
- Tăng Vĩnh Hà:
    - Viết script để chia dữ liệu thành các tập train, val, test
    - Tìm hiểu kiến trúc các mô hình và các siêu tham số cần được tinh chỉnh
    - Cài đặt ConvLSTM
    - Viết script để vẽ bản đồ mưa
    - Viết báo cáo
- Ngô Tùng Lâm:
    - Code chuyển đổi dữ liệu sang dạng bảng
    - Phân tích dữ liệu
    - Code xử lý dữ liệu bị missing
    - Cài đặt XGBoost
    - Viết báo cáo
- Vũ Nguyệt Hằng
    - Phân tích dữ liệu
    - Code xử lý dữ liệu bị missing
    - Code xử lý dữ liệu: xử lý giá trị ngoại lai, bổ sung một số đặc trưng, scale dữ liệu
    - Cài đặt RandomForest
    - Viết báo cáo
