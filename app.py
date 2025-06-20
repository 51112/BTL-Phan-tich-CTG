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

# Hàm kiểm tra tính hợp lệ của dataset mới
def validate_new_dataset(data, required_columns=['date', 'title', 'views', 'day_of_week', 'month', 'quarter', 'tfidf_score']):
    logger.info("Kiểm tra tính hợp lệ của dataset mới...")
    if not all(col in data.columns for col in required_columns):
        missing_cols = [col for col in required_columns if col not in data.columns]
        st.error(f"Dataset mới thiếu các cột bắt buộc: {missing_cols}")
        return False
    if len(data['title'].unique()) != 1:
        st.error("Dataset mới chỉ được chứa đúng một title duy nhất.")
        return False
    if data['date'].isna().any():
        st.error("Cột 'date' trong dataset mới chứa giá trị NaN.")
        return False
    try:
        data['date'] = pd.to_datetime(data['date'])
    except Exception as e:
        st.error(f"Lỗi khi chuyển đổi cột 'date' sang định dạng datetime: {str(e)}")
        return False
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

# Hàm tải mô hình TFT
@st.cache_resource
def load_tft_model(title, data):
    logger.info(f"Tải mô hình TFT cho title: {title}")
    model_path = "representative_tft.pt"
    if len(data) < 15:
        logger.error(f"Dữ liệu cho {title} không đủ (yêu cầu tối thiểu 15 dòng)")
        st.error(f"Dữ liệu cho {title} không đủ để tạo TimeSeriesDataSet (yêu cầu tối thiểu 15 dòng).")
        return None, None
    # Chuyển đổi kiểu dữ liệu
    for col in ['views', 'day_of_week', 'month', 'quarter', 'tfidf_score', 'time_idx']:
        data[col] = pd.to_numeric(data[col], errors='coerce').fillna(0).astype(float)
    training = TimeSeriesDataSet(
        data,
        time_idx="time_idx",
        target="views",
        group_ids=["title"],
        allow_missing_timesteps=True,
        min_encoder_length=15,
        max_encoder_length=30,
        max_prediction_length=31,  # Dự báo 31 ngày
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
            tft.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
            logger.info("Đã tải mô hình TFT đại diện thành công")
            st.warning(f"Sử dụng mô hình TFT đại diện cho {title}.")
            return tft, training
        except Exception as e:
            logger.error(f"Lỗi khi tải mô hình TFT: {str(e)}")
            st.error(f"Lỗi khi tải mô hình TFT: {str(e)}")
            return None, None
    logger.error("Không tìm thấy mô hình TFT đại diện")
    st.error("Không tìm thấy mô hình TFT đại diện")
    return None, None

# Hàm tải mô hình Informer
@st.cache_resource
def load_informer_model(title):
    logger.info(f"Tải mô hình Informer cho title: {title}")
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
        out_len=31,  # Dự báo 31 ngày
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
            st.warning(f"Sử dụng mô hình Informer đại diện cho {title}.")
            return model
        except Exception as e:
            logger.error(f"Lỗi khi tải mô hình Informer: {str(e)}")
            st.error(f"Lỗi khi tải mô hình Informer: {str(e)}")
            return None
    logger.error("Không tìm thấy mô hình Informer đại diện")
    st.error("Không tìm thấy mô hình Informer đại diện")
    return None

# Giao diện Streamlit
st.title("Dự báo lượt truy cập Website")
logger.info("Khởi tạo giao diện Streamlit thành công")
st.write("Tải lên dataset mới (chỉ chứa 1 title) để dự báo lượt truy cập trong 31 ngày của tháng 12/2024, so sánh với 2 tháng cuối (11-12/2024) và xu hướng tháng 10/2024.")

# Tải lên dataset mới
uploaded_file = st.file_uploader("Tải lên file CSV (chứa 1 title)", type=["csv"])
if uploaded_file is not None:
    try:
        data = pd.read_csv(uploaded_file)
        if validate_new_dataset(data):
            data['time_idx'] = (pd.to_datetime(data['date']) - pd.to_datetime(data['date']).min()).dt.days
            data.fillna(0, inplace=True)
            title = data['title'].iloc[0]
            st.success(f"Dataset mới đã được tải với title: {title}")
        else:
            st.error("Dataset mới không hợp lệ. Vui lòng kiểm tra lại.")
            data = None
            title = None
    except Exception as e:
        logger.error(f"Lỗi khi đọc file CSV mới: {str(e)}")
        st.error(f"Lỗi khi đọc file CSV mới: {str(e)}")
        data = None
        title = None
else:
    data = None
    title = None
    st.warning("Vui lòng tải lên file CSV để tiếp tục.")

# Dự báo
if st.button("Dự báo") and data is not None and title is not None:
    logger.info(f"Bắt đầu dự báo cho title: {title}")
    # Tổng hợp dữ liệu theo ngày để loại bỏ trùng lặp
    df_title = data.groupby('date').agg({
        'views': 'sum',
        'day_of_week': 'mean',
        'month': 'mean',
        'quarter': 'mean',
        'tfidf_score': 'mean',
        'time_idx': 'first',  # Giữ time_idx đầu tiên
        'title': 'first'      # Giữ title đầu tiên
    }).reset_index()
    df_title.set_index('date', inplace=True)
    
    # Kiểm tra trùng lặp trong index
    if df_title.index.duplicated().any():
        logger.error("Index 'date' vẫn chứa giá trị trùng lặp sau khi tổng hợp")
        st.error("Dữ liệu có ngày trùng lặp không thể xử lý. Vui lòng kiểm tra file CSV.")
    else:
        # Đặt tần suất hàng ngày và điền giá trị thiếu
        df_title = df_title.asfreq('D', fill_value=0)

        # Lấy 3 tháng cuối năm 2024 (10/2024, 11/2024, 12/2024)
        end_date = pd.to_datetime('2024-12-31')
        last_month_start = pd.to_datetime('2024-11-01')
        two_months_ago_start = pd.to_datetime('2024-10-01')
        df_three_months = df_title[(df_title.index >= two_months_ago_start) & (df_title.index <= end_date)]
        df_last_month = df_title[(df_title.index >= last_month_start) & (df_title.index <= end_date)]

        # ARIMA (tạm thời bỏ qua, sử dụng giá trị mặc định)
        arima_forecast = np.zeros(31)
        logger.warning("Mô hình ARIMA bị vô hiệu hóa do lỗi pmdarima, sử dụng dự báo mặc định (0)")

        # TFT
        try:
            tft_model, training = load_tft_model(title, df_three_months)
            if tft_model and training:
                tft_forecast = []
                df_input = df_three_months.reset_index()
                steps = (31 // 7) + (1 if 31 % 7 else 0)

                for _ in range(steps):
                    for col in ['views', 'day_of_week', 'month', 'quarter', 'tfidf_score', 'time_idx']:
                        df_input[col] = pd.to_numeric(df_input[col], errors='coerce').fillna(0).astype(float)
                    temp_training = TimeSeriesDataSet(
                        df_input,
                        time_idx="time_idx",
                        target="views",
                        group_ids=["title"],
                        allow_missing_timesteps=True,
                        min_encoder_length=15,
                        max_encoder_length=30,
                        max_prediction_length=31,
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
                    predictions = np.maximum(predictions, 0)[:31]
                    tft_forecast.extend(predictions)

                    if len(tft_forecast) < 31:
                        last_date = df_input['date'].max()
                        new_dates = pd.date_range(start=last_date + timedelta(days=1), periods=min(7, 31 - len(tft_forecast)), freq='D')
                        last_tfidf = float(df_input['tfidf_score'].iloc[-1])
                        last_time_idx = df_input['time_idx'].max()
                        new_time_idx = np.arange(last_time_idx + 1, last_time_idx + 1 + len(new_dates), dtype=np.int64)
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

                tft_forecast = np.array(tft_forecast[:31])
                logger.info("Dự báo TFT thành công")
            else:
                tft_forecast = np.zeros(31)
                logger.warning("Không có mô hình TFT, sử dụng dự báo mặc định (0)")
        except Exception as e:
            logger.error(f"Lỗi khi dự báo TFT: {str(e)}")
            st.error(f"Lỗi khi dự báo TFT: {str(e)}")
            tft_forecast = np.zeros(31)

        # Informer
        try:
            informer_model = load_informer_model(title)
            if informer_model:
                scaler = StandardScaler()
                views_scaler = StandardScaler()
                df_informer = df_three_months[['views', 'day_of_week', 'month', 'quarter', 'tfidf_score']].tail(30)
                data_scaled = scaler.fit_transform(df_informer)
                views_scaled = views_scaler.fit_transform(df_informer[['views']])

                x_enc = torch.tensor(data_scaled, dtype=torch.float32).unsqueeze(0)
                x_mark_enc = torch.tensor(df_informer[['day_of_week', 'month', 'quarter']].values, dtype=torch.float32).unsqueeze(0)
                x_dec = torch.zeros((15 + 31, 5), dtype=torch.float32).unsqueeze(0)
                x_dec[:, :15] = torch.tensor(data_scaled[-15:], dtype=torch.float32)
                x_mark_dec = torch.tensor(np.zeros((15 + 31, 3)), dtype=torch.float32).unsqueeze(0)
                x_mark_dec[:, :15] = x_mark_enc[:, -15:]

                informer_model.eval()
                pred_scaled = informer_model(x_enc, x_mark_enc, x_dec, x_mark_dec).squeeze(-1).detach().numpy().squeeze()
                informer_forecast = views_scaler.inverse_transform(pred_scaled.reshape(-1, 1)).flatten()[:31]
                informer_forecast = np.maximum(informer_forecast, 0)
                logger.info("Dự báo Informer thành công")
            else:
                informer_forecast = np.zeros(31)
                logger.warning("Không có mô hình Informer, sử dụng dự báo mặc định (0)")
        except Exception as e:
            logger.error(f"Lỗi khi dự báo Informer: {str(e)}")
            st.error(f"Lỗi khi dự báo Informer: {str(e)}")
            informer_forecast = np.zeros(31)

        # Trực quan hóa 3 tháng cuối và dự báo 31 ngày
        st.subheader("So sánh dữ liệu thực tế 3 tháng cuối 2024 và dự báo 31 ngày tháng 12/2024")
        fig = go.Figure()

        # Dữ liệu thực tế 3 tháng cuối (10/2024 - 12/2024)
        fig.add_trace(go.Scatter(
            x=df_three_months.index,
            y=df_three_months['views'],
            name="Thực tế (3 tháng cuối 2024)",
            mode="lines+markers",
            line=dict(color="blue"),
            marker=dict(size=4)
        ))

        # Dự báo 31 ngày tháng 12/2024 (nối với điểm cuối của tháng 12)
        forecast_dates = pd.date_range(start=end_date + timedelta(days=1), periods=31, freq='D')

        # ARIMA
        fig.add_trace(go.Scatter(
            x=[end_date] + list(forecast_dates),
            y=[df_last_month['views'].iloc[-1]] + list(arima_forecast),
            name="ARIMA",
            mode="lines+markers",
            line=dict(color="green", dash="dash"),
            marker=dict(size=4)
        ))

        # TFT
        fig.add_trace(go.Scatter(
            x=[end_date] + list(forecast_dates),
            y=[df_last_month['views'].iloc[-1]] + list(tft_forecast),
            name="TFT",
            mode="lines+markers",
            line=dict(color="red", dash="dot"),
            marker=dict(size=4)
        ))

        # Informer
        fig.add_trace(go.Scatter(
            x=[end_date] + list(forecast_dates),
            y=[df_last_month['views'].iloc[-1]] + list(informer_forecast),
            name="Informer",
            mode="lines+markers",
            line=dict(color="purple", dash="longdash"),
            marker=dict(size=4)
        ))

        # Cấu hình biểu đồ
        fig.update_layout(
            title=f"Dữ liệu thực tế 3 tháng cuối 2024 và dự báo 31 ngày cho '{title}'",
            xaxis_title="Ngày",
            yaxis_title="Lượt truy cập",
            template="plotly_white",
            hovermode="x unified",
            showlegend=True,
            xaxis=dict(
                gridcolor="lightgray",
                showgrid=True,
                tickformat="%Y-%m-%d"
            ),
            yaxis=dict(
                gridcolor="lightgray",
                showgrid=True,
                zeroline=True
            ),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="center",
                x=0.5
            )
        )

        st.plotly_chart(fig, use_container_width=True)

        # Hiển thị kết quả đánh giá (dựa trên mô hình đại diện)
        st.subheader("Kết quả đánh giá mô hình (dựa trên mô hình đại diện)")
        try:
            arima_results = load_results_file("arima_results.csv")
            tft_results = load_results_file("tft_results.csv")
            informer_results = load_results_file("informer_results.csv")

            arima_mae = arima_results['mae'].mean() if arima_results is not None else "N/A"
            arima_rmse = arima_results['rmse'].mean() if arima_results is not None else "N/A"
            tft_mae = tft_results['mae'].mean() if tft_results is not None else "N/A"
            tft_rmse = tft_results['rmse'].mean() if tft_results is not None else "N/A"
            informer_mae = informer_results['mae'].mean() if informer_results is not None else "N/A"
            informer_rmse = informer_results['rmse'].mean() if informer_results is not None else "N/A"

            st.write(f"**ARIMA** - MAE: {arima_mae:.2f}, RMSE: {arima_rmse:.2f}" if arima_mae != "N/A" else "**ARIMA** - Không có kết quả đánh giá")
            st.write(f"**TFT** - MAE: {tft_mae:.2f}, RMSE: {tft_rmse:.2f}" if tft_mae != "N/A" else "**TFT** - Không có kết quả đánh giá")
            st.write(f"**Informer** - MAE: {informer_mae:.2f}, RMSE: {informer_rmse:.2f}" if informer_mae != "N/A" else "**Informer** - Không có kết quả đánh giá")
        except Exception as e:
            logger.error(f"Lỗi khi tải kết quả đánh giá: {str(e)}")
            st.error(f"Lỗi khi tải kết quả đánh giá: {str(e)}")
