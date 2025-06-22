# Dự báo Lượt Truy cập Bài viết trên Wikipedia
Đây là dự án bài tập lớn môn Phân tích Chuỗi Thời gian thuộc Khoa Công nghệ Thông tin, Trường Đại học Thủy Lợi. Dự án tập trung vào việc xây dựng hệ thống dự báo lượt truy cập bài viết trên Wikipedia dựa trên dữ liệu lịch sử và nội dung bài viết, sử dụng các mô hình chuỗi thời gian hiện đại.
# Thông tin dự án

Đề tài: Dự báo lượt truy cập bài viết trên Wikipedia dựa trên Logs và nội dung bài viết
## Giảng viên hướng dẫn: 
Trần Anh Đạt
## Nhóm sinh viên thực hiện:
### Bùi Anh Tuấn, 64 TTNT1
### Phạm Đình Bảo, 64 TTNT1
### Nguyễn Đức Anh, 63 TTNT
### Trần Quốc Huy, 64 TTNT1


# Thời gian: Tháng 6, 2025
### Ngôn ngữ lập trình: Python
### Dữ liệu: Bộ dữ liệu Wikipedia Structured Contents (Kaggle) https://www.kaggle.com/datasets/wikimedia-foundation/wikipedia-structured-contents

# Mục tiêu

### Tổng quát: Xây dựng hệ thống dự báo lượt truy cập bài viết trên Wikipedia với độ chính xác cao, sử dụng các mô hình phân tích chuỗi thời gian.
### Cụ thể:
#### Áp dụng các mô hình ARIMA, Temporal Fusion Transformer (TFT), và Informer.
#### So sánh hiệu suất các mô hình trên dữ liệu thực tế.
#### Đánh giá độ chính xác và khả năng ứng dụng thực tế.
#### Đề xuất mô hình phù hợp nhất cho bài toán.


# Cấu trúc dự án
project/
|-- .devcontainer/
|   |-- devcontainer.json                # Cấu hình container phát triển
|-- Tien_xu_ly/
|   |-- Crawl.ipynb                      # Notebook thu thập dữ liệu
|   |-- Tien_xu_ly.ipynb                 # Notebook tiền xử lý dữ liệu
|-- data/
|   |-- raw/                             # Dữ liệu thô chưa xử lý
|   |-- processed/                       # Dữ liệu đã được xử lý
|-- models/
|   |-- Informer2020/                    # Thư mục chứa mô hình Informer
|   |-- representative_arima.pkl         # Mô hình ARIMA đã huấn luyện
|   |-- representative_informer.pt       # Mô hình Informer đã huấn luyện
|   |-- representative_tft.pt            # Mô hình TFT đã huấn luyện
|-- results/
|   |-- arima_results.csv                # Kết quả dự báo của ARIMA
|   |-- informer_results.csv             # Kết quả dự báo của Informer
|   |-- tft_results.csv                  # Kết quả dự báo của TFT
|-- .gitignore                           # Danh sách file/thư mục bỏ qua git
|-- app.py                               # Ứng dụng web chính
|-- requirements.txt                     # Danh sách thư viện Python
|-- system_requirements.txt              # Yêu cầu hệ thống

# Công cụ và thư viện

#### Ngôn ngữ: Python
#### Môi trường phát triển: VS Code
#### Thư viện chính:
#### pandas, numpy: Xử lý dữ liệu
#### statsmodels: Mô hình ARIMA
#### sklearn: Chuẩn hóa dữ liệu, tính TF-IDF, đánh giá
#### pytorch, pytorch_forecasting: Mô hình TFT
#### Informer2020: Mô hình Informer
#### streamlit, plotly: Giao diện và trực quan hóa
#### tqdm, joblib, multiprocessing, logging: Tăng tốc và quản lý quy trình



# Quy trình thực hiện

### Thu thập dữ liệu:

#### Tải bộ dữ liệu Wikipedia Structured Contents từ Kaggle và đặt vào data/raw/.
#### Crawl dữ liệu lượt truy cập qua API Wikipedia (xem Tien_xu_ly/Crawl.ipynb).


### Tiền xử lý dữ liệu:

#### Chuyển đổi file JSONL thành CSV (xem Tien_xu_ly/Tien_xu_ly.ipynb).
#### Lọc các cột quan trọng (title, abstract, date, view).
#### Tính điểm TF-IDF cho nội dung bài viết.
#### Lưu kết quả vào data/processed/ (các file CSV thô, lọc, crawl, long).


### Xây dựng mô hình:

#### ARIMA: Mô hình cổ điển, hiệu quả với dữ liệu tuyến tính.
#### TFT: Mô hình học sâu, xử lý đa biến, sử dụng cơ chế attention.
#### Informer: Mô hình Transformer cải tiến cho chuỗi thời gian dài.


### Đánh giá mô hình:

#### Sử dụng các độ đo MAE và RMSE.
#### Lưu kết quả vào results/.


### Triển khai ứng dụng:

#### Phát triển ứng dụng Streamlit (app.py) cho phép người dùng tải file CSV và dự báo lượt truy cập (1-30 ngày).
#### Trực quan hóa kết quả bằng biểu đồ.



# Hướng dẫn cài đặt

Clone repository:
git clone <repository_url>
cd project


Cài đặt môi trường:
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows


Cài đặt thư viện:
pip install -r requirements.txt


Tải dữ liệu:

Tải file enwiki_namespace_0_0.jsonl từ Kaggle và đặt vào thư mục data/raw/.
Chạy Tien_xu_ly/Tien_xu_ly.ipynb và Tien_xu_ly/Crawl.ipynb để tạo các file CSV trong data/processed/. Lưu ý: Các file CSV không được đẩy lên Git, bạn cần tự tạo chúng.


Chạy tiền xử lý dữ liệu:
jupyter notebook Tien_xu_ly/Tien_xu_ly.ipynb
jupyter notebook Tien_xu_ly/Crawl.ipynb


Huấn luyện mô hình:

Các mô hình đã được huấn luyện và lưu trong models/. Nếu cần huấn luyện lại, hãy tham khảo mã nguồn trong các notebook hoặc script tương ứng.


Chạy ứng dụng Streamlit:
streamlit run app.py



# Kết quả

Hiệu suất mô hình:

Informer: Độ chính xác cao nhất, đặc biệt với dự báo dài hạn (15-30 ngày).
TFT: Tốt cho dự báo ngắn hạn (1-7 ngày), nhưng kém ổn định hơn khi thời gian dài.
ARIMA: Phù hợp với dữ liệu đơn giản, nhưng không hiệu quả với dữ liệu phức tạp.


Ứng dụng Streamlit:

Giao diện thân thiện, hỗ trợ dự báo và trực quan hóa kết quả.
Ví dụ: Dự báo lượt truy cập bài viết “Bill McCarren” và “1902 French legislative election”.


Đánh giá:

Informer được đề xuất là mô hình tối ưu cho bài toán.
Cần cải thiện TFT và ARIMA về độ chính xác và khả năng nắm bắt xu hướng.



Hạn chế và hướng phát triển

Hạn chế:

Mô hình học sâu (TFT, Informer) yêu cầu tài nguyên tính toán lớn.
Dữ liệu phức tạp gây khó khăn trong tiền xử lý.
Các file CSV từ crawl và tiền xử lý không được lưu trên Git, yêu cầu người dùng tự tạo.


Hướng phát triển:

Tích hợp thêm biến ngoại sinh (sự kiện xã hội, xu hướng tìm kiếm).
Tối ưu hóa mô hình để giảm thời gian huấn luyện.
Mở rộng sang dữ liệu đa ngôn ngữ.



# Liên hệ
Nếu có câu hỏi hoặc cần hỗ trợ, vui lòng liên hệ nhóm thực hiện qua email:  anhnguyen080123@gmail.com


Lưu ý: Dự án này được thực hiện với mục đích học thuật. 
