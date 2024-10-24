import lightning.pytorch as pl
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
from pytorch_forecasting.metrics import RMSE
import optuna
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import logging
import warnings
import torch
import os
import random

# نمایش هشدارها
warnings.filterwarnings("ignore")

# تنظیمات logging
logging.basicConfig(
    filename='model_training.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# تنظیم پورت DDP به صورت تصادفی برای جلوگیری از خطای EADDRINUSE
os.environ['MASTER_PORT'] = str(random.randint(10000, 20000))
print(f"MASTER_PORT set to: {os.environ['MASTER_PORT']}")

# تعریف تابع encode_time_of_day
def encode_time_of_day(time_str):
    time_in_seconds = pd.to_timedelta(time_str).total_seconds()
    max_time = 24 * 60 * 60  # تعداد ثانیه‌های یک روز
    time_sin = np.sin(2 * np.pi * time_in_seconds / max_time)
    time_cos = np.cos(2 * np.pi * time_in_seconds / max_time)
    return time_sin, time_cos

def objective(trial):
    # انتخاب هایپرپارامترهای مهم برای بهینه‌سازی
    window_size = trial.suggest_int('max_encoder_length', 10, 100)  # سایز پنجره زمانی
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])  # اندازه پچ
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-2)  # نرخ یادگیری
    dropout = trial.suggest_float('dropout', 0.1, 0.5)  # دراپ‌اوت برای جلوگیری از اورفیت
    hidden_size = trial.suggest_int('hidden_size', 16, 128)  # تعداد نودهای لایه مخفی

    # بارگذاری داده‌ها و تنظیمات DataLoader
    df = pd.read_csv("df_1_years.csv")  # مطمئن شوید فایل csv شما در مسیر درست است
    df = df.sample(frac=0.1, random_state=42)  # استفاده از ۱۰٪ داده‌ها برای تست

    # پیش‌پردازش داده‌ها
    df['time_sin'], df['time_cos'] = zip(*df['Time of Day'].apply(encode_time_of_day))
    df.drop(columns=['Unnamed: 0', 'Time of Day'], inplace=True)
    df['Sequence_10'] = df['Sequence_10'].shift(-10)
    df.dropna(inplace=True)
    df = df.sort_index()
    df['time_idx'] = np.arange(len(df))

    scaler = StandardScaler()
    feature_columns = df.columns.difference(['Sequence_10', 'time_idx', 'ID'])
    df[feature_columns] = scaler.fit_transform(df[feature_columns])

    if 'ID' not in df.columns:
        df['ID'] = 1
    df['ID'] = df['ID'].astype(str)

    training_cutoff = int(0.8 * len(df))

    # تعریف TimeSeriesDataSet برای آموزش
    training = TimeSeriesDataSet(
        df[lambda x: x.time_idx <= training_cutoff],
        time_idx="time_idx",
        target="Sequence_10",
        group_ids=["ID"],
        max_encoder_length=window_size,
        max_prediction_length=1,
        static_categoricals=["ID"],
        static_reals=[],
        time_varying_known_reals=["time_idx", "time_sin", "time_cos"],
        time_varying_unknown_reals=feature_columns.tolist(),
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True
    )

    # ایجاد مجموعه داده اعتبارسنجی
    validation = TimeSeriesDataSet.from_dataset(
        training, 
        df, 
        min_prediction_idx=training_cutoff + 1, 
        stop_randomization=True
    )

    # تنظیم DataLoaderها
    train_dataloader = training.to_dataloader(train=True, batch_size=batch_size, num_workers=2)
    val_dataloader = validation.to_dataloader(train=False, batch_size=batch_size, num_workers=2)

    # تعریف مدل TFT
    tft = TemporalFusionTransformer.from_dataset(
        training,
        learning_rate=learning_rate,
        hidden_size=hidden_size,
        dropout=dropout,
        loss=RMSE(),
        log_interval=10
    )

    # تنظیم Trainer با استراتژی DDP و استفاده از یک GPU
    trainer = pl.Trainer(
        max_epochs=100,
        accelerator="gpu",
        devices=1,  # استفاده از یک GPU
        strategy="ddp",  # استفاده از ddp به جای ddp_spawn
        precision=32,  # استفاده از دقت ۳۲ بیتی
        gradient_clip_val=0.1,
        callbacks=[
            EarlyStopping(monitor="val_loss", patience=10),
            LearningRateMonitor(logging_interval='step')
        ],
        logger=TensorBoardLogger("lightning_logs"),
        num_sanity_val_steps=0  # غیرفعال کردن sanity check
    )

    # آموزش مدل
    trainer.fit(
        tft, 
        train_dataloaders=train_dataloader, 
        val_dataloaders=val_dataloader
    )

    # محاسبه RMSE
    predictions = tft.predict(val_dataloader)
    true_values = []
    pred_values = []

    for batch, pred in zip(val_dataloader, predictions):
        target = batch[1]
        
        # اگر target یک tuple است، اولین عنصر آن را دریافت کنید
        if isinstance(target, tuple):
            target = target[0]
        
        # استخراج مقادیر از Tensor و تبدیل به numpy
        if isinstance(target, torch.Tensor):
            true_values.append(target.detach().cpu())
            pred_values.append(pred.detach().cpu())
    
    # تبدیل مقادیر به numpy با torch.cat
    true_values = torch.cat(true_values).numpy()
    pred_values = torch.cat(pred_values).numpy()

    # بررسی اینکه آیا اندازه true_values و pred_values برابر هستند
    if len(true_values) != len(pred_values):
        print(f"Warning: Inconsistent lengths, true_values: {len(true_values)}, pred_values: {len(pred_values)}")
        # برش دادن true_values و pred_values به کوچکترین طول
        min_len = min(len(true_values), len(pred_values))
        true_values = true_values[:min_len]
        pred_values = pred_values[:min_len]
    
    # محاسبه RMSE
    rmse = np.sqrt(mean_squared_error(true_values, pred_values))

    # فقط فرآیند اصلی (rank 0) لاگ را می‌نویسد
    if trainer.global_rank == 0:
        with open("optuna_logs.txt", "a") as f:
            f.write(f"Trial {trial.number} - Parameters: Window Size: {window_size}, Batch Size: {batch_size}, "
                    f"Learning Rate: {learning_rate}, Dropout: {dropout}, Hidden Size: {hidden_size} - RMSE: {rmse}\n")

    return rmse

# اجرای کد اصلی درون بلوک if __name__ == "__main__":
if __name__ == "__main__":
    # تنظیمات Optuna
    study = optuna.create_study(direction="minimize")
    # شروع بهینه‌سازی
    study.optimize(objective, n_trials=50)

    # بهترین هایپرپارامترها و RMSE
    best_trial = study.best_trial
    print(f"Best trial: {best_trial.number}")
    print(f"Best parameters: {best_trial.params}")
    print(f"Best RMSE: {best_trial.value}")
