import streamlit as st
import pandas as pd
import numpy as np
import joblib
import torch
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer
import plotly.graph_objects as go
import os
import sys
import re
from sklearn.preprocessing import StandardScaler
from datetime import timedelta
import logging
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller

# Thiết lập logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Hàm làm sạch tên file
def clean_filename(filename):
    cleaned = re.sub(r'[^\w\s-]', '', filename)
    cleaned = re.sub(r'\s+', '_', cleaned.strip())
    return cleaned

# Thêm đường dẫn Informer2020 và import
try:
    logger.info("Kiểm tra và thêm đường dẫn Informer2020...")
    sys.path.append(os.path.abspath("Informer2020"))
    from Informer2020.models.model import Informer
    logger.info("Đã import thành công mô hình Informer")
except Exception as e:
    logger.error(f"Lỗi khi import Informer: {str(e)}")
    st.error(f"Lỗi khi import Informer: {str(e)}")
    Informer = None

# Hàm kiểm tra tính dừng
def check_stationarity(series):
    result = adfuller(series)
    return result[1] <= 0.05  # p-value <= 0.05 -> chuỗi dừng

# Hàm làm dữ liệu liên tục
def make_data_continuous(data):
    logger.info("Bắt đầu làm dữ liệu liên tục...")
    try:
        # Đảm bảo cột date ở dạng datetime
        data['date'] = pd.to_datetime(data['date'], errors='coerce')
        if data['date'].isna().any():
            logger.error("Cột 'date' chứa giá trị không hợp lệ.")
            st.error("Cột 'date' chứa giá trị không hợp lệ.")
            return None

        # Tạo chuỗi ngày liên tục từ ngày nhỏ nhất đến ngày lớn nhất
        start_date = data['date'].min()
        end_date = data['date'].max()
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')

        # Tạo DataFrame mới với tất cả các ngày
        df_continuous = pd.DataFrame({'date': date_range})

        # Gộp dữ liệu gốc với chuỗi ngày liên tục
        df_continuous = df_continuous.merge(data, on='date', how='left')

        # Điền giá trị thiếu
        df_continuous['title'] = df_continuous['title'].fillna(data['title'].iloc[0])
        df_continuous['views'] = df_continuous['views'].interpolate(method='linear', limit_direction='both')
        df_continuous['day_of_week'] = df_continuous['date'].dt.dayofweek
        df_continuous['month'] = df_continuous['date'].dt.month
        df_continuous['quarter'] = df_continuous['date'].dt.quarter
        df_continuous['tfidf_score'] = df_continuous['tfidf_score'].fillna(method='ffill').fillna(method='bfill')
        df_continuous['time_idx'] = (df_continuous['date'] - start_date).dt.days

        # Đảm bảo các cột số ở đúng định dạng
        for col in ['views', 'day_of_week', 'month', 'quarter', 'tfidf_score']:
            df_continuous[col] = pd.to_numeric(df_continuous[col], errors='coerce').astype(float)

        logger.info("Đã làm dữ liệu liên tục thành công")
        return df_continuous
    except Exception as e:
        logger.error(f"Lỗi khi làm dữ liệu liên tục: {str(e)}")
        st.error(f"Lỗi khi làm dữ liệu liên tục: {str(e)}")
        return None

# Hàm kiểm tra tính hợp lệ của dataset mới
def validate_new_dataset(data):
    logger.info("Kiểm tra tính hợp lệ của dataset mới...")
    required_columns = ['date', 'title', 'views', 'day_of_week', 'month', 'quarter', 'tfidf_score', 'time_idx']
    if not all(col in data.columns for col in required_columns):
        missing_cols = [col for col in required_columns if col not in data.columns]
        st.error(f"Dataset mới thiếu các cột bắt buộc: {missing_cols}. Các cột có trong file: {list(data.columns)}")
        return False
    if len(data['title'].unique()) != 1:
        st.error("Dataset mới chỉ được chứa đúng một title duy nhất.")
        return False
    if data['date'].isna().any() or data['time_idx'].isna().any():
        st.error("Cột 'date' hoặc 'time_idx' chứa giá trị NaN.")
        return False
    try:
        data['date'] = pd.to_datetime(data['date'], errors='coerce')
        if data['date'].isna().any():
            st.error("Một số giá trị trong cột 'date' không hợp lệ và đã bị chuyển thành NaT.")
            return False
        if not pd.api.types.is_numeric_dtype(data['time_idx']) or (data['time_idx'] < 0).any():
            st.error("Cột 'time_idx' phải là số nguyên không âm.")
            return False
        for col in ['views', 'day_of_week', 'month', 'quarter', 'tfidf_score']:
            data[col] = pd.to_numeric(data[col], errors='coerce').fillna(0).astype(float)
        logger.info("Dataset mới hợp lệ")
        return True
    except Exception as e:
        st.error(f"Lỗi khi xử lý cột 'date' hoặc 'time_idx': {str(e)}")
        return False

# Hàm tải file kết quả đánh giá
@st.cache_data
def load_results_file(file_name):
    logger.info(f"Tải file kết quả: {file_name}")
    try:
        return pd.read_csv(file_name)
    except Exception as e:
        logger.error(f"Lỗi khi đọc file {file_name}: {str(e)}")
        st.error(f"Lỗi khi đọc file {file_name}: {str(e)}")
        return None

# Hàm tính MAE và RMSE
def calculate_metrics(actual, predicted):
    if len(actual) == len(predicted):
        mae = mean_absolute_error(actual, predicted)
        rmse = np.sqrt(mean_squared_error(actual, predicted))
        return mae, rmse
    return None, None

# Hàm dự báo ARIMA
def forecast_arima(data, forecast_days):
    logger.info(f"Dự báo ARIMA cho {forecast_days} ngày")
    try:
        model_path = "representative_arima.pkl"
        series = data['views'].values
        # Kiểm tra dữ liệu đầu vào
        if len(series) < 10:
            logger.warning(f"Dữ liệu quá ngắn ({len(series)} bản ghi), sử dụng dự báo mặc định.")
            st.warning(f"Dữ liệu quá ngắn ({len(series)} bản ghi), không thể dự báo ARIMA chính xác.")
            return np.zeros(forecast_days)

        # Kiểm tra tính dừng và chọn order
        order = (1, 1, 1)  # Mặc định
        if not check_stationarity(series):
            series_diff = np.diff(series)
            series = series_diff[~np.isnan(series_diff)]
            order = (1, 1, 1)  # D=1
            if len(series) < 10:
                logger.warning(f"Dữ liệu sau diff lần 1 quá ngắn ({len(series)} bản ghi), sử dụng dự báo mặc định.")
                return np.zeros(forecast_days)
            if not check_stationarity(series):
                series_diff = np.diff(series)
                series = series_diff[~np.isnan(series_diff)]
                order = (1, 2, 1)  # D=2
                if len(series) < 10:
                    logger.warning(f"Dữ liệu sau diff lần 2 quá ngắn ({len(series)} bản ghi), sử dụng dự báo mặc định.")
                    return np.zeros(forecast_days)

        # Thử tải mô hình đại diện
        if os.path.exists(model_path):
            try:
                model = joblib.load(model_path)
                forecast = model.forecast(steps=forecast_days)
                forecast = np.maximum(forecast, 0)
                logger.info("Đã sử dụng mô hình ARIMA đại diện từ representative_arima.pkl")
                st.info("Sử dụng mô hình ARIMA đại diện từ representative_arima.pkl")
                return forecast
            except Exception as e:
                logger.warning(f"Lỗi khi tải mô hình đại diện: {str(e)}. Huấn luyện lại mô hình ARIMA.")
                st.warning(f"Lỗi khi tải mô hình đại diện: {str(e)}. Huấn luyện lại mô hình ARIMA.")

        # Huấn luyện lại mô hình nếu không có mô hình đại diện hoặc lỗi
        logger.info("Huấn luyện lại mô hình ARIMA trên dữ liệu mới...")
        model = ARIMA(data['views'].values, order=order).fit()
        forecast = model.forecast(steps=forecast_days)
        forecast = np.maximum(forecast, 0)
        logger.info(f"Dự báo ARIMA thành công với {forecast_days} ngày")
        return forecast
    except Exception as e:
        logger.error(f"Lỗi khi dự báo ARIMA: {str(e)}")
        st.error(f"Lỗi khi dự báo ARIMA: {str(e)}. Sử dụng dự báo mặc định (0).")
        return np.zeros(forecast_days)

# Hàm tải mô hình TFT
@st.cache_resource
def load_tft_model(title, data, forecast_days):
    logger.info(f"Tải mô hình TFT cho title: {title} với {forecast_days} ngày dự báo")
    model_path = "representative_tft.pt"
    if len(data) < 15:
        logger.error(f"Dữ liệu cho {title} không đủ (yêu cầu tối thiểu 15 dòng)")
        st.error(f"Dữ liệu cho {title} không đủ để tạo TimeSeriesDataSet (yêu cầu tối thiểu 15 dòng).")
        return None, None
    for col in ['views', 'day_of_week', 'month', 'quarter', 'tfidf_score']:
        data[col] = pd.to_numeric(data[col], errors='coerce').fillna(0).astype(float)
    data = data.sort_values('time_idx')
    if data['time_idx'].isna().any():
        logger.error("Cột 'time_idx' chứa giá trị NaN.")
        st.error("Cột 'time_idx' chứa giá trị NaN. Vui lòng kiểm tra dữ liệu.")
        return None, None
    training = TimeSeriesDataSet(
        data,
        time_idx="time_idx",
        target="views",
        group_ids=["title"],
        allow_missing_timesteps=True,
        min_encoder_length=15,
        max_encoder_length=30,
        max_prediction_length=forecast_days,
        static_categoricals=["title"],
        time_varying_known_reals=["day_of_week", "month", "quarter", "tfidf_score"],
        time_varying_unknown_reals=["views"],
        target_normalizer=GroupNormalizer(groups=["title"], transformation="softplus")
    )
    tft = TemporalFusionTransformer.from_dataset(
        training,
        hidden_size=32,
        attention_head_size=2,
        dropout=0.2,
        hidden_continuous_size=16,
        output_size=1
    )
    if os.path.exists(model_path):
        try:
            state_dict = torch.load(model_path, map_location=torch.device('cpu'))
            filtered_state_dict = {k: v for k, v in state_dict.items() if k in tft.state_dict() and v.shape == tft.state_dict()[k].shape}
            if len(filtered_state_dict) < len(state_dict):
                logger.warning(f"Có {len(state_dict) - len(filtered_state_dict)} tham số không khớp cấu trúc, đã bỏ qua.")
            tft.load_state_dict(filtered_state_dict, strict=False)
            logger.info(f"Đã tải mô hình TFT đại diện thành công cho {title}")
            st.warning(f"Sử dụng mô hình TFT đại diện cho {title}")
        except Exception as e:
            logger.error(f"Lỗi khi tải mô hình TFT: {str(e)}")
            st.error(f"Lỗi khi tải mô hình TFT: {str(e)}. Sử dụng mô hình mặc định.")
            tft = TemporalFusionTransformer.from_dataset(training, hidden_size=32, attention_head_size=2, dropout=0.2, hidden_continuous_size=16, output_size=1)
    else:
        logger.error("Không tìm thấy mô hình TFT đại diện")
        st.error("Không tìm thấy mô hình TFT đại diện. Sử dụng mô hình mặc định.")
    return tft, training

# Hàm tải mô hình Informer
@st.cache_resource
def load_informer_model(title, forecast_days):
    logger.info(f"Tải mô hình Informer cho title: {title} với {forecast_days} ngày dự báo")
    if Informer is None:
        logger.error("Không thể sử dụng mô hình Informer do lỗi import")
        st.error("Không thể sử dụng mô hình Informer do lỗi import")
        return None
    model_path = "representative_informer.pt"
    model = Informer(
        enc_in=5,
        dec_in=5,
        c_out=1,
        seq_len=30,
        label_len=15,
        out_len=forecast_days,
        d_model=256,
        n_heads=4,
        e_layers=1,
        d_layers=1,
        d_ff=1024,
        dropout=0.1,
        attn='prob',
        embed='timeF',
        freq='d'
    )
    if os.path.exists(model_path):
        try:
            model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
            logger.info("Đã tải mô hình Informer đại diện thành công")
            st.warning(f"Sử dụng mô hình Informer đại diện cho {title}")
        except Exception as e:
            logger.error(f"Lỗi khi tải mô hình Informer: {str(e)}")
            st.error(f"Lỗi khi tải mô hình Informer: {str(e)}. Sử dụng mô hình mặc định.")
            model = Informer(
                enc_in=5, dec_in=5, c_out=1, seq_len=30, label_len=15, out_len=forecast_days,
                d_model=256, n_heads=4, e_layers=1, d_layers=1, d_ff=1024, dropout=0.1,
                attn='prob', embed='timeF', freq='d'
            )
    else:
        logger.error("Không tìm thấy mô hình Informer đại diện")
        st.error("Không tìm thấy mô hình Informer đại diện. Sử dụng mô hình mặc định.")
    return model

# Giao diện Streamlit
st.title("Dự báo lượt truy cập Website")
logger.info("Khởi tạo giao diện Streamlit thành công")
st.write("Tải lên dataset mới (chứa 1 title) để dự báo lượt truy cập.")

# Tải lên dataset mới
uploaded_file = st.file_uploader("Tải lên file CSV (chứa 1 title)", type=["csv"])
if uploaded_file is not None:
    try:
        data = pd.read_csv(uploaded_file)
        logger.info(f"Các cột trong file CSV: {list(data.columns)}")
        if validate_new_dataset(data):
            data = make_data_continuous(data)
            if data is None:
                st.error("Không thể làm dữ liệu liên tục. Vui lòng kiểm tra lại.")
                data = None
                title = None
            else:
                data['time_idx'] = data['time_idx'].astype(int)
                data.fillna(0, inplace=True)
                title = data['title'].iloc[0]
                st.success(f"Dataset mới đã được tải với title: {title}")
                st.write("Dữ liệu đầu vào (đã làm liên tục):", data.head())
        else:
            st.error("Dataset mới không hợp lệ. Vui lòng kiểm tra lại.")
            data = None
            title = None
    except Exception as e:
        logger.error(f"Lỗi khi đọc file CSV mới: {str(e)}")
        st.error(f"Lỗi khi đọc file CSV mới: {str(e)}. Vui lòng kiểm tra file CSV (các cột yêu cầu: date, title, views, day_of_week, month, quarter, tfidf_score, time_idx).")
        data = None
        title = None
else:
    data = None
    title = None
    st.warning("Vui lòng tải lên file CSV để tiếp tục.")

# Chọn số ngày dự báo
forecast_days = st.slider("Chọn số ngày dự báo (tối đa 30 ngày):", min_value=1, max_value=30, value=7, step=1)
if forecast_days > 7:
    st.warning("Dự báo ARIMA cho hơn 7 ngày có thể dẫn đến sai số lớn. Hãy cân nhắc sử dụng TFT hoặc Informer cho dự báo dài hạn.")

# Dự báo
if st.button("Dự báo") and data is not None and title is not None:
    logger.info(f"Bắt đầu dự báo cho title: {title} với {forecast_days} ngày")
    df_title = data.copy()
    df_title = df_title.sort_values('time_idx')
    
    end_date = pd.to_datetime('2024-12-31')
    last_month_start = pd.to_datetime('2024-11-01')
    two_months_ago_start = pd.to_datetime('2024-10-01')
    df_three_months = df_title[df_title['date'] >= two_months_ago_start]
    df_last_month = df_title[df_title['date'] >= last_month_start]

    # ARIMA
    arima_forecast = forecast_arima(df_three_months, forecast_days)

    # TFT
    try:
        tft_model, training = load_tft_model(title, df_three_months, forecast_days)
        if tft_model and training:
            tft_forecast = []
            df_input = df_three_months.copy()
            steps = (forecast_days // 7) + (1 if forecast_days % 7 else 0)

            for _ in range(steps):
                for col in ['views', 'day_of_week', 'month', 'quarter', 'tfidf_score']:
                    df_input[col] = pd.to_numeric(df_input[col], errors='coerce').fillna(0).astype(float)
                df_input = df_input.sort_values('time_idx')
                if df_input['time_idx'].isna().any():
                    logger.error("Cột 'time_idx' chứa giá trị NaN trong dự báo TFT.")
                    st.error("Cột 'time_idx' chứa giá trị NaN. Vui lòng kiểm tra dữ liệu.")
                    break
                temp_training = TimeSeriesDataSet(
                    df_input,
                    time_idx="time_idx",
                    target="views",
                    group_ids=["title"],
                    allow_missing_timesteps=True,
                    min_encoder_length=15,
                    max_encoder_length=30,
                    max_prediction_length=forecast_days,
                    static_categoricals=["title"],
                    time_varying_known_reals=["day_of_week", "month", "quarter", "tfidf_score"],
                    time_varying_unknown_reals=["views"],
                    target_normalizer=GroupNormalizer(groups=["title
