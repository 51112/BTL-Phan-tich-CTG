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
import pmdarima as pm

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
        # Điền title (giá trị duy nhất trong tập test)
        df_continuous['title'] = df_continuous['title'].fillna(data['title'].iloc[0])

        # Nội suy tuyến tính cho cột views
        df_continuous['views'] = df_continuous['views'].interpolate(method='linear', limit_direction='both')

        # Điền các cột khác
        df_continuous['day_of_week'] = df_continuous['date'].dt.dayofweek
        df_continuous['month'] = df_continuous['date'].dt.month
        df_continuous['quarter'] = df_continuous['date'].dt.quarter

        # Điền tfidf_score bằng giá trị gần nhất (forward fill, rồi backward fill nếu cần)
        df_continuous['tfidf_score'] = df_continuous['tfidf_score'].fillna(method='ffill')
        df_continuous['tfidf_score'] = df_continuous['tfidf_score'].fillna(method='bfill')

        # Tạo lại time_idx liên tục
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
            st.error("Một số giá trị trong cột 'date' không hợp lệ và đã bị chuyển thành NaT. Vui lòng kiểm tra định dạng ngày tháng.")
            return False
        # Kiểm tra time_idx là số nguyên không âm
        if not pd.api.types.is_numeric_dtype(data['time_idx']) or (data['time_idx'] < 0).any():
            st.error("Cột 'time_idx' phải là số nguyên không âm.")
            return False
    except Exception as e:
        st.error(f"Lỗi khi xử lý cột 'date' hoặc 'time_idx': {str(e)}")
        return False
    # Chuyển đổi các cột số sang kiểu dữ liệu phù hợp
    for col in ['views', 'day_of_week', 'month', 'quarter', 'tfidf_score']:
        data[col] = pd.to_numeric(data[col], errors='coerce').fillna(0).astype(float)
    logger.info("Dataset mới hợp lệ")
    return True

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
        # Lấy chuỗi views của title duy nhất
        series = data['views'].values
        # Xây dựng và huấn luyện mô hình ARIMA tự động
        model = pm.auto_arima(series, seasonal=True, m=7, trace=True, error_action='ignore', suppress_warnings=True)
        # Dự báo
        forecast, conf_int = model.predict(n_periods=forecast_days, return_conf_int=True)
        forecast = np.maximum(forecast, 0)  # Đảm bảo giá trị không âm
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
    # Chuyển đổi kiểu dữ liệu, đảm bảo tất cả cột là số
    for col in ['views', 'day_of_week', 'month', 'quarter', 'tfidf_score']:
        data[col] = pd.to_numeric(data[col], errors='coerce').fillna(0).astype(float)
    # Sử dụng time_idx đã có sẵn, không tính lại từ date
    data = data.sort_values('time_idx')
    if data['time_idx'].isna().any():
        logger.error("Cột 'time_idx' chứa giá trị NaN.")
        st.error("Cột 'time_idx' chứa giá trị NaN. Vui lòng kiểm tra dữ liệu.")
        return None, None
    training = TimeSeriesDataSet(
        data,
        time_idx="time_idx",  # Sử dụng cột time_idx đã có
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
            # Lọc các tham số khớp với cấu trúc hiện tại
            model_dict = tft.state_dict()
            filtered_state_dict = {k: v for k, v in state_dict.items() if k in model_dict and v.shape == model_dict[k].shape}
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
st.write("Tải lên dataset mới (chỉ chứa 1 title) để dự báo lượt truy cập.")

# Tải lên dataset mới
uploaded_file = st.file_uploader("Tải lên file CSV (chứa 1 title)", type=["csv"])
if uploaded_file is not None:
    try:
        data = pd.read_csv(uploaded_file)
        logger.info(f"Các cột trong file CSV: {list(data.columns)}")
        if validate_new_dataset(data):
            # Làm dữ liệu liên tục trước khi xử lý
            data = make_data_continuous(data)
            if data is None:
                st.error("Không thể làm dữ liệu liên tục. Vui lòng kiểm tra lại.")
                data = None
                title = None
            else:
                data['time_idx'] = data['time_idx'].astype(int)  # Đảm bảo time_idx là integer
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

# Dự báo
if st.button("Dự báo") and data is not None and title is not None:
    logger.info(f"Bắt đầu dự báo cho title: {title} với {forecast_days} ngày")
    # Sử dụng dữ liệu đã có time_idx, không cần tổng hợp lại
    df_title = data.copy()
    df_title = df_title.sort_values('time_idx')
    
    # Lấy 3 tháng cuối năm 2024 (dựa trên time_idx)
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
                # Sử dụng time_idx đã có, không cần tạo lại
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
                    target_normalizer=GroupNormalizer(groups=["title"], transformation="softplus")
                )
                dataloader = temp_training.to_dataloader(train=False, batch_size=32, num_workers=0)
                predictions = []
                tft_model.eval()
                with torch.no_grad():
                    for batch in dataloader:
                        x, _ = batch
                        x = {k: v.to(torch.device('cpu')) for k, v in x.items() if v is not None}
                        output = tft_model(x)
                        pred = output.prediction.squeeze(-1)
                        predictions.append(pred)
                predictions = torch.cat(predictions, dim=0).detach().numpy()
                if predictions.ndim > 1:
                    predictions = predictions.flatten()
                predictions = np.maximum(predictions, 0)[:forecast_days]
                tft_forecast.extend(predictions)

                if len(tft_forecast) < forecast_days:
                    last_time_idx = df_input['time_idx'].max()
                    new_time_idx = np.arange(last_time_idx + 1, last_time_idx + 1 + min(7, forecast_days - len(tft_forecast)), dtype=np.int64)
                    last_date = pd.to_datetime(df_input['date'].iloc[-1])
                    new_dates = pd.date_range(start=last_date + timedelta(days=1), periods=min(7, forecast_days - len(tft_forecast)), freq='D')
                    last_tfidf = float(df_input['tfidf_score'].iloc[-1])
                    new_data = pd.DataFrame({
                        'date': new_dates,
                        'views': predictions[:len(new_dates)],
                        'day_of_week': new_dates.dayofweek.to_numpy(),
                        'month': new_dates.month.to_numpy(),
                        'quarter': new_dates.quarter.to_numpy(),
                        'tfidf_score': [last_tfidf] * len(new_dates),
                        'title': [title] * len(new_dates),
                        'time_idx': new_time_idx
                    })
                    df_input = pd.concat([df_input, new_data], ignore_index=True)

            tft_forecast = np.array(tft_forecast[:forecast_days])
            logger.info(f"Dự báo TFT thành công với {forecast_days} ngày")
        else:
            tft_forecast = np.zeros(forecast_days)
            logger.warning("Không có mô hình TFT, sử dụng dự báo mặc định (0)")
    except Exception as e:
        logger.error(f"Lỗi khi dự báo TFT: {str(e)}")
        st.error(f"Lỗi khi dự báo TFT: {str(e)}.")
        tft_forecast = np.zeros(forecast_days)

    # Informer
    try:
        informer_model = load_informer_model(title, forecast_days)
        if informer_model:
            scaler = StandardScaler()
            views_scaler = StandardScaler()
            df_informer = df_three_months[['views', 'day_of_week', 'month', 'quarter', 'tfidf_score']].tail(30)
            data_scaled = scaler.fit_transform(df_informer)
            views_scaled = views_scaler.fit_transform(df_informer[['views']])

            x_enc = torch.tensor(data_scaled, dtype=torch.float32).unsqueeze(0)
            x_mark_enc = torch.tensor(df_informer[['day_of_week', 'month', 'quarter']].values, dtype=torch.float32).unsqueeze(0)
            x_dec = torch.zeros((15 + forecast_days, 5), dtype=torch.float32).unsqueeze(0)
            x_dec[:, :15] = torch.tensor(data_scaled[-15:], dtype=torch.float32)
            x_mark_dec = torch.tensor(np.zeros((15 + forecast_days, 3)), dtype=torch.float32).unsqueeze(0)
            x_mark_dec[:, :15] = x_mark_enc[:, -15:]

            informer_model.eval()
            pred_scaled = informer_model(x_enc, x_mark_enc, x_dec, x_mark_dec).squeeze(-1).detach().numpy().squeeze()
            informer_forecast = views_scaler.inverse_transform(pred_scaled.reshape(-1, 1)).flatten()[:forecast_days]
            informer_forecast = np.maximum(informer_forecast, 0)
            logger.info(f"Dự báo Informer thành công với {forecast_days} ngày")
        else:
            informer_forecast = np.zeros(forecast_days)
            logger.warning("Không có mô hình Informer, sử dụng dự báo mặc định (0)")
    except Exception as e:
        logger.error(f"Lỗi khi dự báo Informer: {str(e)}")
        st.error(f"Lỗi khi dự báo Informer: {str(e)}")
        informer_forecast = np.zeros(forecast_days)

    # Trực quan hóa
    st.subheader(f"So sánh dữ liệu thực tế 3 tháng cuối 2024 và dự báo {forecast_days} ngày")
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df_three_months['date'],
        y=df_three_months['views'],
        name="Thực tế (3 tháng cuối 2024)",
        mode="lines+markers",
        line=dict(color="blue"),
        marker=dict(size=4)
    ))

    forecast_dates = pd.date_range(start=df_three_months['date'].max() + timedelta(days=1), periods=forecast_days, freq='D')
    forecast_dates_numeric = (forecast_dates - pd.Timestamp("1970-01-01")) // pd.Timedelta('1s')

    fig.add_trace(go.Scatter(
        x=[df_three_months['date'].max()] + list(forecast_dates),
        y=[df_last_month['views'].iloc[-1]] + list(arima_forecast),
        name="ARIMA",
        mode="lines+markers",
        line=dict(color="green", dash="dash"),
        marker=dict(size=4)
    ))

    fig.add_trace(go.Scatter(
        x=[df_three_months['date'].max()] + list(forecast_dates),
        y=[df_last_month['views'].iloc[-1]] + list(tft_forecast),
        name="TFT",
        mode="lines+markers",
        line=dict(color="red", dash="dot"),
        marker=dict(size=4)
    ))

    fig.add_trace(go.Scatter(
        x=[df_three_months['date'].max()] + list(forecast_dates),
        y=[df_last_month['views'].iloc[-1]] + list(informer_forecast),
        name="Informer",
        mode="lines+markers",
        line=dict(color="purple", dash="longdash"),
        marker=dict(size=4)
    ))

    if forecast_days >= 30:
        fig.add_vline(x=float(forecast_dates_numeric[0]), line_dash="dash", line_color="gray", annotation_text="1 ngày")
        fig.add_vline(x=float(forecast_dates_numeric[6]), line_dash="dash", line_color="gray", annotation_text="7 ngày")
        fig.add_vline(x=float(forecast_dates_numeric[29]), line_dash="dash", line_color="gray", annotation_text="30 ngày")

    fig.update_layout(
        title=f"Dữ liệu thực tế 3 tháng cuối 2024 và dự báo {forecast_days} ngày cho '{title}'",
        xaxis_title="Ngày",
        yaxis_title="Lượt truy cập",
        template="plotly_white",
        hovermode="x unified",
        showlegend=True,
        xaxis=dict(gridcolor="lightgray", showgrid=True, tickformat="%Y-%m-%d"),
        yaxis=dict(gridcolor="lightgray", showgrid=True, zeroline=True),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5)
    )

    st.plotly_chart(fig, use_container_width=True)

    # Hiển thị kết quả đánh giá
    st.subheader("Kết quả đánh giá mô hình (dựa trên dữ liệu đại diện)")
    try:
        # Tải kết quả từ file (nếu có)
        arima_results = load_results_file("arima_results.csv")
        tft_results = load_results_file("tft_results.csv")
        informer_results = load_results_file("informer_results.csv")

        # Lấy dữ liệu thực tế cuối cùng để so sánh (giả sử dùng dữ liệu đã biết)
        actual_values = df_three_months['views'].tail(forecast_days).values
        if len(actual_values) < forecast_days:
            actual_values = np.pad(actual_values, (0, forecast_days - len(actual_values)), mode='constant', constant_values=0)

        # Tính MAE và RMSE cho từng mô hình
        arima_mae, arima_rmse = calculate_metrics(actual_values, arima_forecast)
        tft_mae, tft_rmse = calculate_metrics(actual_values, tft_forecast)
        informer_mae, informer_rmse = calculate_metrics(actual_values, informer_forecast)

        # Nếu không tính được (do dữ liệu không đủ), dùng giá trị từ file
        if arima_mae is None or arima_rmse is None:
            arima_mae = arima_results['mae'].mean() if arima_results is not None else "N/A"
            arima_rmse = arima_results['rmse'].mean() if arima_results is not None else "N/A"
        if tft_mae is None or tft_rmse is None:
            tft_mae = tft_results['mae'].mean() if tft_results is not None else "N/A"
            tft_rmse = tft_results['rmse'].mean() if tft_results is not None else "N/A"
        if informer_mae is None or informer_rmse is None:
            informer_mae = informer_results['mae'].mean() if informer_results is not None else "N/A"
            informer_rmse = informer_results['rmse'].mean() if informer_results is not None else "N/A"

        st.write(f"**ARIMA** - MAE: {arima_mae:.2f}, RMSE: {arima_rmse:.2f}" if arima_mae != "N/A" else "**ARIMA** - Không có kết quả đánh giá")
        st.write(f"**TFT** - MAE: {tft_mae:.2f}, RMSE: {tft_rmse:.2f}" if tft_mae != "N/A" else "**TFT** - Không có kết quả đánh giá")
        st.write(f"**Informer** - MAE: {informer_mae:.2f}, RMSE: {informer_rmse:.2f}" if informer_mae != "N/A" else "**Informer** - Không có kết quả đánh giá")
    except Exception as e:
        logger.error(f"Lỗi khi tải kết quả đánh giá: {str(e)}")
        st.error(f"Lỗi khi tải kết quả đánh giá: {str(e)}")
