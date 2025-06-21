import pandas as pd
import torch
import torch.multiprocessing as mp
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
from pytorch_forecasting.metrics import MAE, RMSE
from pytorch_forecasting.data import GroupNormalizer
from tqdm import tqdm
import os
import re
import logging
import warnings
import numpy as np
from torch.cuda.amp import GradScaler, autocast
import gc

# Tắt cảnh báo
warnings.filterwarnings("ignore")

# Cấu hình logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Hàm làm sạch tên file
def clean_filename(filename):
    cleaned = re.sub(r'[^\w\s-]', '', filename)
    cleaned = re.sub(r'\s+', '_', cleaned.strip())
    return cleaned

# Hàm làm sạch bộ nhớ
def clear_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    logger.info("Đã làm sạch bộ nhớ")

# Hàm dự báo thủ công
def manual_predict(model, dataloader, device):
    try:
        model.eval()
        predictions = []
        targets = []
        with torch.no_grad():
            for batch in dataloader:
                x, y = batch
                if x is None or y is None or not x:
                    logger.warning("Batch dữ liệu rỗng trong dự báo")
                    continue
                x = {k: v.to(device) for k, v in x.items() if v is not None}
                output = model(x)
                pred = output.prediction.squeeze(-1)
                predictions.append(pred.cpu())
                targets.append(y[0].cpu())
        if not predictions or not targets:
            logger.warning("Không có dự báo hoặc mục tiêu hợp lệ")
            return None, None
        predictions = torch.cat(predictions, dim=0)
        targets = torch.cat(targets, dim=0)
        return predictions, targets
    except Exception as e:
        logger.error(f"Lỗi trong dự báo: {str(e)}")
        return None, None

# Hàm huấn luyện mô hình cho một title
def train_model_for_title(args):
    idx, title, df_chunk, chunk_idx, device = args
    try:
        logger.info(f"Đang huấn luyện mô hình cho title: {title} trong chunk {chunk_idx}")
        
        # Kiểm tra dữ liệu chunk
        if df_chunk.empty or len(df_chunk) < 15:
            logger.warning(f"Chunk {chunk_idx} cho title {title} không đủ dữ liệu (số dòng: {len(df_chunk)})")
            return None
        
        # Kiểm tra các cột bắt buộc
        required_columns = ['time_idx', 'views', 'title', 'day_of_week', 'month', 'quarter', 'tfidf_score']
        missing_columns = [col for col in required_columns if col not in df_chunk.columns]
        if missing_columns:
            logger.warning(f"Chunk {chunk_idx} cho title {title} thiếu cột: {missing_columns}")
            return None
        
        # Tạo TimeSeriesDataSet
        try:
            training = TimeSeriesDataSet(
                df_chunk,
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
        except Exception as e:
            logger.error(f"Lỗi khi tạo TimeSeriesDataSet cho title {title} trong chunk {chunk_idx}: {str(e)}")
            return None
        
        # Kiểm tra dataset
        if len(training) == 0:
            logger.warning(f"Dataset trống cho title {title} trong chunk {chunk_idx}")
            return None
        
        # Tạo dataloader
        try:
            train_dataloader = training.to_dataloader(train=True, batch_size=32, num_workers=0)
        except Exception as e:
            logger.error(f"Lỗi khi tạo dataloader cho title {title} trong chunk {chunk_idx}: {str(e)}")
            return None
        
        # Khởi tạo mô hình TFT
        tft = TemporalFusionTransformer.from_dataset(
            training,
            learning_rate=0.001,
            hidden_size=32,
            attention_head_size=2,
            dropout=0.2,
            hidden_continuous_size=16,
            output_size=1,
            loss=MAE()
        ).to(device)
        
        # Optimizer và scheduler
        optimizer = torch.optim.AdamW(tft.parameters(), lr=0.001, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=1)
        scaler = GradScaler() if device.type == 'cuda' else None
        
        # Vòng lặp huấn luyện
        tft.train()
        prev_lr = optimizer.param_groups[0]['lr']
        for epoch in range(5):
            total_loss = 0
            batch_count = 0
            for batch in train_dataloader:
                optimizer.zero_grad()
                x, y = batch
                if x is None or y is None or not x or not y[0].numel():
                    logger.warning(f"Batch rỗng hoặc target không hợp lệ cho title {title} trong chunk {chunk_idx}")
                    continue
                x = {k: v.to(device) for k, v in x.items() if v is not None}
                y = tuple(t.to(device) for t in y if t is not None)
                if not y or not y[0].numel():
                    logger.warning(f"Target rỗng cho title {title} trong chunk {chunk_idx}")
                    continue
                
                if device.type == 'cuda':
                    with autocast():
                        output = tft(x)
                        pred = output.prediction.squeeze(-1)
                        target = y[0]
                        loss = MAE()(pred, target)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    output = tft(x)
                    pred = output.prediction.squeeze(-1)
                    target = y[0]
                    loss = MAE()(pred, target)
                    loss.backward()
                    optimizer.step()
                
                total_loss += loss.item()
                batch_count += 1
            if batch_count == 0:
                logger.warning(f"Không có batch hợp lệ cho title {title} trong chunk {chunk_idx}, epoch {epoch+1}")
                return None
            avg_loss = total_loss / batch_count
            logger.info(f"Title: {title}, Chunk: {chunk_idx}, Epoch: {epoch+1}, Loss: {avg_loss:.4f}")
            scheduler.step(avg_loss)
            
            # Kiểm tra learning rate
            current_lr = optimizer.param_groups[0]['lr']
            if current_lr < prev_lr:
                logger.info(f"Learning rate giảm xuống {current_lr:.6f} cho title {title}")
            prev_lr = current_lr
        
        # Lưu mô hình với tiền tố tft_{idx+1}_
        os.makedirs("models", exist_ok=True)
        model_path = f"models/tft_{idx+1}_{clean_filename(title)}.pt"
        torch.save(tft.state_dict(), model_path)
        
        # Dự báo thủ công và đánh giá
        predictions, targets = manual_predict(tft, train_dataloader, device)
        if predictions is None or targets is None:
            logger.warning(f"Dự báo thất bại cho title {title} trong chunk {chunk_idx}")
            return None
        mae = MAE()(predictions, targets)
        rmse = RMSE()(predictions, targets)
        
        return {
            'title': title,
            'chunk_idx': chunk_idx,
            'mae': mae.item(),
            'rmse': rmse.item(),
            'forecast': predictions.detach().numpy(),
            'model_path': model_path
        }
    except Exception as e:
        logger.error(f"Lỗi huấn luyện title {title} trong chunk {chunk_idx}: {str(e)}")
        return None

# Hàm huấn luyện mô hình đại diện cho Streamlit
def train_representative_model(data, device):
    try:
        logger.info("Đang huấn luyện mô hình đại diện cho triển khai Streamlit")
        
        # Lấy mẫu tập con dữ liệu
        sample_titles = data['title'].unique()[:50]
        df_sample = data[data['title'].isin(sample_titles)].copy()
        
        if df_sample.empty or len(df_sample) < 15:
            logger.error(f"Dữ liệu mẫu cho mô hình đại diện không đủ (số dòng: {len(df_sample)})")
            return None
        
        # Kiểm tra các cột bắt buộc
        required_columns = ['time_idx', 'views', 'title', 'day_of_week', 'month', 'quarter', 'tfidf_score']
        missing_columns = [col for col in required_columns if col not in df_sample.columns]
        if missing_columns:
            logger.error(f"Dữ liệu mẫu thiếu cột: {missing_columns}")
            return None
        
        # Tạo TimeSeriesDataSet
        try:
            training = TimeSeriesDataSet(
                df_sample,
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
        except Exception as e:
            logger.error(f"Lỗi khi tạo TimeSeriesDataSet cho mô hình đại diện: {str(e)}")
            return None
        
        if len(training) == 0:
            logger.error("Dataset đại diện trống")
            return None
        
        # Tạo dataloader
        try:
            train_dataloader = training.to_dataloader(train=True, batch_size=64, num_workers=1)
        except Exception as e:
            logger.error(f"Lỗi khi tạo dataloader cho mô hình đại diện: {str(e)}")
            return None
        
        # Khởi tạo mô hình TFT
        tft = TemporalFusionTransformer.from_dataset(
            training,
            learning_rate=0.0005,
            hidden_size=64,
            attention_head_size=4,
            dropout=0.1,
            hidden_continuous_size=32,
            output_size=1,
            loss=MAE()
        ).to(device)
        
        # Optimizer và scheduler
        optimizer = torch.optim.AdamW(tft.parameters(), lr=0.0005, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
        scaler = GradScaler() if device.type == 'cuda' else None
        
        # Vòng lặp huấn luyện
        tft.train()
        for epoch in range(7):
            total_loss = 0
            batch_count = 0
            for batch in train_dataloader:
                optimizer.zero_grad()
                x, y = batch
                if x is None or y is None or not x or not y[0].numel():
                    logger.warning(f"Batch rỗng hoặc target không hợp lệ trong mô hình đại diện, epoch {epoch+1}")
                    continue
                x = {k: v.to(device) for k, v in x.items() if v is not None}
                y = tuple(t.to(device) for t in y if t is not None)
                if not y or not y[0].numel():
                    logger.warning(f"Target rỗng trong mô hình đại diện, epoch {epoch+1}")
                    continue
                
                if device.type == 'cuda':
                    with autocast():
                        output = tft(x)
                        pred = output.prediction.squeeze(-1)
                        target = y[0]
                        loss = MAE()(pred, target)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    output = tft(x)
                    pred = output.prediction.squeeze(-1)
                    target = y[0]
                    loss = MAE()(pred, target)
                    loss.backward()
                    optimizer.step()
                
                total_loss += loss.item()
                batch_count += 1
            if batch_count == 0:
                logger.warning(f"Không có batch hợp lệ cho mô hình đại diện, epoch {epoch+1}")
                return None
            avg_loss = total_loss / batch_count
            logger.info(f"Mô hình đại diện, Epoch: {epoch+1}, Loss: {avg_loss:.4f}")
            scheduler.step(avg_loss)
        
        # Lưu mô hình đại diện trong thư mục gốc
        rep_model_path = "representative_tft.pt"
        state_dict = tft.state_dict()
        torch.save(state_dict, rep_model_path)
        logger.info(f"Mô hình đại diện được lưu tại {rep_model_path}")
        
        return rep_model_path
    except Exception as e:
        logger.error(f"Lỗi huấn luyện mô hình đại diện: {str(e)}")
        return None

# Hàm chính
def main():
    try:
        # Kiểm tra thiết bị
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Sử dụng thiết bị: {device}")
        
        # Đọc dữ liệu
        data = pd.read_csv("Crawl(2)/Crawl_ca_nam_long/Crawl_full_views_ca_nam_batch_1.csv")
        data['date'] = pd.to_datetime(data['date'], errors='coerce')
        data['time_idx'] = (data['date'] - data['date'].min()).dt.days
        
        # Kiểm tra dữ liệu
        logger.info("Kiểm tra NaN trong dữ liệu:")
        logger.info(data[['date', 'title', 'views', 'day_of_week', 'month', 'quarter', 'tfidf_score']].isna().sum())
        
        # Xử lý NaN
        data.fillna({'views': 0, 'tfidf_score': 0, 'day_of_week': 0, 'month': 0, 'quarter': 0}, inplace=True)
        data.dropna(subset=['date', 'title'], inplace=True)
        
        # Lấy danh sách title
        titles = data['title'].unique()[:200]  # Giới hạn 200 title
        
        # Chia title thành các chunk, mỗi chunk tối đa 20 title
        titles_per_chunk = 20
        chunks = []
        for i in range(0, len(titles), titles_per_chunk):
            chunk_titles = titles[i:i+titles_per_chunk]
            chunk_data = data[data['title'].isin(chunk_titles)].copy()
            if not chunk_data.empty:
                chunks.append(chunk_data)
                logger.info(f"Chunk {len(chunks)}: {len(chunk_titles)} titles, {len(chunk_data)} rows")
        
        # Kiểm tra tính toàn vẹn của chunk
        all_chunk_titles = []
        for chunk in chunks:
            chunk_titles = set(chunk['title'].unique())
            for prev_titles in all_chunk_titles:
                if not chunk_titles.isdisjoint(prev_titles):
                    logger.error("Có title bị chia nhỏ giữa các chunk!")
            all_chunk_titles.append(chunk_titles)
        
        # Chuẩn bị tham số cho multiprocessing
        mp_args = []
        for chunk_idx, chunk in enumerate(chunks):
            chunk_grouped = chunk.groupby('title')
            for idx, title in enumerate(chunk['title'].unique()):
                df_title = chunk_grouped.get_group(title).copy()
                if not df_title.empty:
                    mp_args.append((idx, title, df_title, chunk_idx, device))
        
        # Sử dụng multiprocessing và làm sạch bộ nhớ sau mỗi chunk
        results = []
        for chunk_idx, chunk_args in enumerate([mp_args[i:i+titles_per_chunk] for i in range(0, len(mp_args), titles_per_chunk)]):
            logger.info(f"Xử lý chunk {chunk_idx + 1}/{len(chunks)}")
            with mp.Pool(processes=max(1, mp.cpu_count() // 2)) as pool:
                chunk_results = list(tqdm(pool.imap(train_model_for_title, chunk_args), total=len(chunk_args), desc=f"Huấn luyện chunk {chunk_idx + 1}"))
                results.extend([r for r in chunk_results if r is not None])
            
            # Làm sạch bộ nhớ sau mỗi chunk
            clear_memory()
        
        # Lưu kết quả đánh giá
        if results:
            results_df = pd.DataFrame(results)
            results_df.to_csv("tft_results.csv", index=False)
            logger.info("Kết quả đánh giá được lưu tại tft_results.csv")
        
        # Huấn luyện mô hình đại diện
        rep_model_path = train_representative_model(data, device)
        if rep_model_path:
            logger.info("Hoàn thành huấn luyện mô hình đại diện")
        
        # Làm sạch bộ nhớ lần cuối
        clear_memory()
        
        logger.info("Hoàn thành huấn luyện TFT. Mô hình được lưu trong thư mục 'models/' với tiền tố từ tft_1_ đến tft_200_. Mô hình đại diện: 'representative_tft.pt'")
    except Exception as e:
        logger.error(f"Lỗi trong quá trình chính: {str(e)}")
        clear_memory()

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    main()
