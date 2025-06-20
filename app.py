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

# Thêm thư mục Informer2020 vào sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'Informer2020')))
from Informer2020.models.model import Informer

# Hàm làm sạch tên file
def clean_filename(filename):
    cleaned = re.sub(r'[^\w\s-]', '', filename)
    cleaned = re.sub(r'\s+', '_', cleaned.strip())
    return cleaned

# Hàm kiểm tra tính hợp lệ của dataset mới
def validate_new_dataset(data, required_columns=['date', 'title', 'views', 'day_of_week', 'month', 'quarter', 'tfidf_score']):
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
    return True

import streamlit as st
import pandas as pd
import gdown
import os
import time
import io

@st.cache_data
def load_train_data():
    data_path = "Data/Crawl_ca_nam_long/Crawl_full_views_ca_nam_batch_1.parquet"
    google_drive_url = "https://drive.google.com/file/d/1bnzkIs3kTMaIKx7w7HKtaLIruZ4R49_Q/view?usp=sharing"  # Thay bằng ID của file Parquet
    
    # Kiểm tra file tồn tại cục bộ
    if not os.path.exists(data_path):
        st.info(f"File {data_path} không tồn tại cục bộ. Đang tải từ Google Drive...")
        for attempt in range(3):  # Thử tối đa 3 lần
            try:
                os.makedirs(os.path.dirname(data_path), exist_ok=True)
                gdown.download(google_drive_url, data_path, quiet=False)
                st.success(f"Đã tải file Parquet từ Google Drive và lưu tại {data_path}")
                break
            except gdown.exceptions.FileURLRetrievalError as e:
                st.warning(f"Lỗi tải file từ Google Drive, thử lại lần {attempt+1}/3...")
                time.sleep(60)  # Chờ 60 giây trước khi thử lại
                if attempt == 2:
                    st.error("Không thể tải file sau 3 lần thử. Vui lòng kiểm tra URL hoặc liên hệ quản trị viên.")
                    return None
            except Exception as e:
                st.error(f"Lỗi không xác định khi tải file: {str(e)}")
                return None
    
    # Đọc file Parquet
    try:
        data = pd.read_parquet(data_path)
        data['date'] = pd.to_datetime(data['date'])
        data['time_idx'] = (data['date'] - data['date'].min()).dt.days
        data.fillna(0, inplace=True)
        return data
    except Exception as e:
        st.error(f"Lỗi khi đọc file Parquet: {str(e)}")
        return None

# Giao diện Streamlit
st.title("Dự báo lượt truy cập Website")
st.write("Chọn bài viết từ dữ liệu đã train hoặc tải lên dataset mới (chỉ chứa 1 title) để dự báo lượt truy cập trong 30 ngày cuối, so sánh với dữ liệu thực tế và xu hướng với tháng trước.")

# Tùy chọn giữa dữ liệu train và dữ liệu mới
data_option = st.radio("Chọn nguồn dữ liệu:", ("Dữ liệu đã train", "Tải lên dataset mới"))

if data_option == "Dữ liệu đã train":
    data = load_train_data()
    titles = data['title'].unique()[:200]
    title = st.selectbox("Chọn bài viết", titles)
    new_data = None
else:
    uploaded_file = st.file_uploader("Tải lên file CSV (chứa 1 title)", type=["csv"])
    if uploaded_file is not None:
        try:
            new_data = pd.read_csv(uploaded_file)
            if validate_new_dataset(new_data):
                new_data['time_idx'] = (pd.to_datetime(new_data['date']) - pd.to_datetime(new_data['date']).min()).dt.days
                new_data.fillna(0, inplace=True)
                title = new_data['title'].iloc[0]
                data = new_data
                st.success(f"Dataset mới đã được tải với title: {title}")
            else:
                st.error("Dataset mới không hợp lệ. Vui lòng kiểm tra lại.")
                new_data = None
                title = None
        except Exception as e:
            st.error(f"Lỗi khi đọc file CSV mới: {str(e)}")
            new_data = None
            title = None
    else:
        new_data = None
        title = None
        st.warning("Vui lòng tải lên file CSV để tiếp tục.")

forecast_days = 30  # Cố định dự báo 30 ngày

# Hàm tải mô hình
@st.cache_resource
def load_arima_model(title, data):
    safe_title = clean_filename(title)
    # Tìm index dựa trên dữ liệu train (nếu có)
    idx = data[data['title'] == title].index[0] if data_option == "Dữ liệu đã train" else 0
    model_path = f"models/arima_{idx+1}_{safe_title}.pkl"
    if os.path.exists(model_path):
        try:
            model = joblib.load(model_path)
            return model
        except Exception as e:
            st.error(f"Lỗi khi tải mô hình ARIMA cho {title}: {str(e)}")
            return None
    else:
        # Sử dụng mô hình đại diện nếu không tìm thấy mô hình riêng
        rep_model_path = "representative_arima.pkl"
        if os.path.exists(rep_model_path):
            try:
                model = joblib.load(rep_model_path)
                st.warning(f"Sử dụng mô hình ARIMA đại diện cho {title} vì không tìm thấy mô hình riêng.")
                return model
            except Exception as e:
                st.error(f"Lỗi khi tải mô hình ARIMA đại diện cho {title}: {str(e)}")
                return None
        st.error(f"Không tìm thấy mô hình ARIMA cho {title} hoặc mô hình đại diện.")
        return None

@st.cache_resource
def load_tft_model(title, data):
    safe_title = clean_filename(title)
    # Tìm index dựa trên dữ liệu train (nếu có)
    idx = data[data['title'] == title].index[0] if data_option == "Dữ liệu đã train" else 0
    model_path = f"models/tft_{idx+1}_{safe_title}.pt"
    df_title = data[data['title'] == title].copy()
    if len(df_title) < 15:
        st.error(f"Dữ liệu cho {title} không đủ để tạo TimeSeriesDataSet (yêu cầu tối thiểu 15 dòng).")
        return None, None
    training = TimeSeriesDataSet(
        df_title,
        time_idx="time_idx",
        target="views",
        group_ids=["title"],
        allow_missing_timesteps=True,
        min_encoder_length=15,
        max_encoder_length=30,
        max_prediction_length=7,
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
            return tft, training
        except Exception as e:
            st.error(f"Lỗi khi tải mô hình TFT cho {title}: {str(e)}")
            return None, None
    else:
        # Sử dụng mô hình đại diện nếu không tìm thấy mô hình riêng
        rep_model_path = "representative_tft.pt"
        if os.path.exists(rep_model_path):
            try:
                tft.load_state_dict(torch.load(rep_model_path, map_location=torch.device('cpu')))
                st.warning(f"Sử dụng mô hình TFT đại diện cho {title} vì không tìm thấy mô hình riêng.")
                return tft, training
            except Exception as e:
                st.error(f"Lỗi khi tải mô hình TFT đại diện cho {title}: {str(e)}")
                return None, None
        st.error(f"Không tìm thấy mô hình TFT cho {title} hoặc mô hình đại diện.")
        return None, None

@st.cache_resource
def load_informer_model(title, data):
    safe_title = clean_filename(title)
    # Tìm index dựa trên dữ liệu train (nếu có)
    idx = data[data['title'] == title].index[0] if data_option == "Dữ liệu đã train" else 0
    model_path = f"models/informer_{idx+1}_{safe_title}.pt"
    model = Informer(
        enc_in=5,
        dec_in=5,
        c_out=1,
        seq_len=30,
        label_len=15,
        out_len=7,
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
            return model
        except Exception as e:
            st.error(f"Lỗi khi tải mô hình Informer cho {title}: {str(e)}")
            return None
    else:
        # Sử dụng mô hình đại diện nếu không tìm thấy mô hình riêng
        rep_model_path = "representative_informer.pt"
        if os.path.exists(rep_model_path):
            try:
                model.load_state_dict(torch.load(rep_model_path, map_location=torch.device('cpu')))
                st.warning(f"Sử dụng mô hình Informer đại diện cho {title} vì không tìm thấy mô hình riêng.")
                return model
            except Exception as e:
                st.error(f"Lỗi khi tải mô hình Informer đại diện cho {title}: {str(e)}")
                return None
        st.error(f"Không tìm thấy mô hình Informer cho {title} hoặc mô hình đại diện.")
        return None

# Dự báo
if st.button("Dự báo") and title is not None:
    # Lọc dữ liệu theo title
    df_title = data[data['title'] == title][['date', 'views', 'day_of_week', 'month', 'quarter', 'tfidf_score']].copy()
    
    # Tổng hợp dữ liệu trùng lặp theo ngày, thêm cột title
    df_title = df_title.groupby('date').agg({
        'views': 'sum',
        'day_of_week': 'mean',
        'month': 'mean',
        'quarter': 'mean',
        'tfidf_score': 'mean'
    }).reset_index()
    df_title['title'] = title
    
    # Đặt lại index là date
    df_title.set_index('date', inplace=True)
    
    # Đặt tần suất hàng ngày và điền giá trị thiếu
    df_title = df_title.asfreq('D', fill_value=0)
    df_title['title'] = title
    
    # Chia dữ liệu: trước tháng cuối (train) và tháng cuối (test)
    last_month_start = df_title.index[-1] - pd.Timedelta(days=30)
    df_train = df_title[df_title.index < last_month_start]
    df_test = df_title[df_title.index >= last_month_start]
    
    # ARIMA
    try:
        arima_model = load_arima_model(title, data)
        if arima_model:
            arima_forecast = arima_model.predict(n_periods=forecast_days)
            arima_forecast = np.maximum(arima_forecast, 0)
        else:
            arima_forecast = np.zeros(forecast_days)
    except Exception as e:
        st.error(f"Lỗi khi dự báo ARIMA: {str(e)}")
        arima_forecast = np.zeros(forecast_days)
    
    # TFT
    try:
        tft_model, training = load_tft_model(title, data)
        if tft_model and training:
            tft_forecast = []
            df_input = df_train.copy().reset_index()
            df_input['time_idx'] = (df_input['date'] - df_input['date'].min()).dt.days.astype(np.int64)
            steps = (forecast_days // 7) + (1 if forecast_days % 7 else 0)
            
            for _ in range(steps):
                if 'title' not in df_input.columns:
                    df_input['title'] = title
                for col in ['views', 'day_of_week', 'month', 'quarter', 'tfidf_score']:
                    df_input[col] = df_input[col].astype(float)
                if df_input['time_idx'].isna().any():
                    raise ValueError("time_idx contains NaN values")
                
                temp_training = TimeSeriesDataSet(
                    df_input,
                    time_idx="time_idx",
                    target="views",
                    group_ids=["title"],
                    allow_missing_timesteps=True,
                    min_encoder_length=15,
                    max_encoder_length=30,
                    max_prediction_length=7,
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
                predictions = np.maximum(predictions, 0)[:7]
                tft_forecast.extend(predictions)
                
                if len(tft_forecast) < forecast_days:
                    last_date = df_input['date'].max()
                    new_dates = pd.date_range(start=last_date + timedelta(days=1), periods=min(7, forecast_days - len(tft_forecast)), freq='D')
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
            
            tft_forecast = np.array(tft_forecast[:forecast_days])
        else:
            tft_forecast = np.zeros(forecast_days)
    except Exception as e:
        st.error(f"Lỗi khi dự báo TFT: {str(e)}")
        tft_forecast = np.zeros(forecast_days)
    
    # Informer
    try:
        informer_model = load_informer_model(title, data)
        if informer_model:
            scaler = StandardScaler()
            views_scaler = StandardScaler()
            df_informer = df_title[['views', 'day_of_week', 'month', 'quarter', 'tfidf_score']].tail(30)
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
        else:
            informer_forecast = np.zeros(forecast_days)
    except Exception as e:
        st.error(f"Lỗi khi dự báo Informer: {str(e)}")
        informer_forecast = np.zeros(forecast_days)
    
    # Trực quan hóa tháng cuối (dữ liệu thực tế + dự đoán)
    st.subheader("So sánh dữ liệu thực tế và dự đoán trong 30 ngày cuối")
    fig1 = go.Figure()
    
    # Dữ liệu thực tế tháng cuối
    fig1.add_trace(go.Scatter(
        x=df_test.index,
        y=df_test['views'],
        name="Thực tế",
        mode="lines+markers",
        line=dict(color="blue"),
        marker=dict(size=4)
    ))
    
    # Dữ liệu thực tế trước tháng cuối (để nối liền)
    last_data_point = df_train.tail(1)
    if not last_data_point.empty:
        fig1.add_trace(go.Scatter(
            x=[last_data_point.index[0], df_test.index[0]],
            y=[last_data_point['views'].iloc[0], df_test['views'].iloc[0]],
            name="Kết nối thực tế",
            mode="lines",
            line=dict(color="blue", width=1),
            showlegend=False
        ))
    
    # Dự báo (nối với điểm cuối của train)
    forecast_dates = df_test.index[:forecast_days]
    
    # ARIMA
    fig1.add_trace(go.Scatter(
        x=[df_train.index[-1]] + list(forecast_dates),
        y=[df_train['views'].iloc[-1]] + list(arima_forecast),
        name="ARIMA",
        mode="lines+markers",
        line=dict(color="green", dash="dash"),
        marker=dict(size=4)
    ))
    
    # TFT
    fig1.add_trace(go.Scatter(
        x=[df_train.index[-1]] + list(forecast_dates),
        y=[df_train['views'].iloc[-1]] + list(tft_forecast),
        name="TFT",
        mode="lines+markers",
        line=dict(color="red", dash="dot"),
        marker=dict(size=4)
    ))
    
    # Informer
    fig1.add_trace(go.Scatter(
        x=[df_train.index[-1]] + list(forecast_dates),
        y=[df_train['views'].iloc[-1]] + list(informer_forecast),
        name="Informer",
        mode="lines+markers",
        line=dict(color="purple", dash="longdash"),
        marker=dict(size=4)
    ))
    
    # Cấu hình biểu đồ
    fig1.update_layout(
        title=f"Dữ liệu thực tế và dự đoán cho '{title}' (30 ngày cuối)",
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
    
    st.plotly_chart(fig1, use_container_width=True)
    
    # Trực quan hóa 30 ngày cuối và 30 ngày trước đó
    st.subheader("So sánh 30 ngày cuối và 30 ngày trước đó")
    second_last_month_start = df_title.index[-1] - pd.Timedelta(days=60)
    second_last_month_end = last_month_start - pd.Timedelta(days=1)
    df_second_last_month = df_title[(df_title.index >= second_last_month_start) & (df_title.index <= second_last_month_end)]
    
    fig2 = go.Figure()
    
    # Dữ liệu 30 ngày trước đó
    fig2.add_trace(go.Scatter(
        x=df_second_last_month.index,
        y=df_second_last_month['views'],
        name="30 ngày trước",
        mode="lines+markers",
        line=dict(color="orange"),
        marker=dict(size=4)
    ))
    
    # Dữ liệu 30 ngày cuối
    fig2.add_trace(go.Scatter(
        x=df_test.index,
        y=df_test['views'],
        name="30 ngày cuối",
        mode="lines+markers",
        line=dict(color="blue"),
        marker=dict(size=4)
    ))
    
    # Cấu hình biểu đồ
    fig2.update_layout(
        title=f"So sánh lượt truy cập giữa 30 ngày cuối và 30 ngày trước đó cho '{title}'",
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
    
    st.plotly_chart(fig2, use_container_width=True)

# Hiển thị kết quả đánh giá (chỉ cho dữ liệu train)
if data_option == "Dữ liệu đã train" and title is not None:
    st.subheader("Kết quả đánh giá mô hình")
    try:
        arima_results = pd.read_csv("arima_results.csv")
        tft_results = pd.read_csv("tft_results.csv")
        informer_results = pd.read_csv("informer_results.csv")
        
        arima_mae = arima_results[arima_results['title'] == title]['mae'].values[0] if title in arima_results['title'].values else "N/A"
        arima_mse = arima_results[arima_results['title'] == title]['rmse'].values[0] if title in arima_results['title'].values else "N/A"  # Sửa từ 'mse' sang 'rmse' theo mã ARIMA
        tft_mae = tft_results[tft_results['title'] == title]['mae'].values[0] if title in tft_results['title'].values else "N/A"
        tft_rmse = tft_results[tft_results['title'] == title]['rmse'].values[0] if title in tft_results['title'].values else "N/A"
        informer_mae = informer_results[informer_results['title'] == title]['mae'].values[0] if title in informer_results['title'].values else "N/A"
        informer_mse = informer_results[informer_results['title'] == title]['rmse'].values[0] if title in informer_results['title'].values else "N/A"  # Sửa từ 'mse' sang 'rmse' theo mã Informer
        
        st.write(f"**ARIMA** - MAE: {arima_mae:.2f}, RMSE: {arima_mse:.2f}" if arima_mae != "N/A" else "**ARIMA** - Không có kết quả đánh giá")
        st.write(f"**TFT** - MAE: {tft_mae:.2f}, RMSE: {tft_rmse:.2f}" if tft_mae != "N/A" else "**TFT** - Không có kết quả đánh giá")
        st.write(f"**Informer** - MAE: {informer_mae:.2f}, RMSE: {informer_mse:.2f}" if informer_mae != "N/A" else "**Informer** - Không có kết quả đánh giá")
    except Exception as e:
        st.error(f"Lỗi khi tải kết quả đánh giá: {str(e)}")
else:
    st.info("Kết quả đánh giá mô hình không khả dụng cho dataset mới.")
