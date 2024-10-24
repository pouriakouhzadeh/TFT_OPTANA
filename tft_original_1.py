# imports for training
import lightning.pytorch as pl
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
from pytorch_forecasting.metrics import RMSE
from lightning.pytorch.tuner import Tuner
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import logging
import warnings

# نمایش هشدارها
warnings.filterwarnings("ignore")

# تنظیمات logging
logging.basicConfig(
    filename='model_training.log', 
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# بارگذاری داده‌ها
df_cleaned = pd.read_csv("aggregated_filtered_data.csv")

# پیش‌پردازش داده‌ها
def encode_time_of_day(time_str):
    time_in_seconds = pd.to_timedelta(time_str).total_seconds()
    max_time = 24 * 60 * 60  # تعداد ثانیه‌های یک روز
    time_sin = np.sin(2 * np.pi * time_in_seconds / max_time)
    time_cos = np.cos(2 * np.pi * time_in_seconds / max_time)
    return time_sin, time_cos

# ایجاد ویژگی‌های sin و cos برای زمان روز
df_cleaned['time_sin'], df_cleaned['time_cos'] = zip(*df_cleaned['Time of Day'].apply(encode_time_of_day))

# حذف ستون‌های غیر مرتبط
df_cleaned.drop(columns=['Unnamed: 0', 'Time of Day'], inplace=True)

# شیفت -10 برای ستون Sequence_10 برای پیش‌بینی 10 گام آینده
# df_cleaned['Sequence_10'] = df_cleaned['Sequence_10'].shift(-10)
# df_cleaned.dropna(inplace=True)

# ایجاد ستون زمانی time_idx
df_cleaned = df_cleaned.sort_index()  # مرتب‌سازی بر اساس ترتیب ردیف‌ها
df_cleaned['time_idx'] = np.arange(len(df_cleaned))  # ایجاد ستون زمانی به عنوان شاخص زمانی پیوسته

# نرمال‌سازی داده‌ها
scaler = StandardScaler()
feature_columns = df_cleaned.columns.difference(['Sequence_10', 'time_idx'])  # ستون‌های زمان و هدف حذف می‌شوند
df_cleaned[feature_columns] = scaler.fit_transform(df_cleaned[feature_columns])

# ذخیره نام ویژگی‌ها
saved_feature_columns = feature_columns

# تعریف نقطه قطع برای آموزش
training_cutoff = df_cleaned["time_idx"].max() - 30  # فرض می‌کنیم 30 روز آخر برای اعتبارسنجی است

# تعریف TimeSeriesDataSet برای آموزش
training = TimeSeriesDataSet(
    df_cleaned[lambda x: x.time_idx <= training_cutoff],
    time_idx="time_idx",
    target="Sequence_10",
    max_encoder_length=36,  # میزان تاریخچه‌ای که مدل می‌تواند استفاده کند
    max_prediction_length=1,  # میزان پیش‌بینی آینده، یک گام بعدی
    time_varying_known_reals=["time_idx", "time_sin", "time_cos"],  # ویژگی‌های عددی شناخته شده در آینده
    time_varying_unknown_reals=feature_columns.tolist(),  # ویژگی‌های عددی ناشناخته در آینده
    add_relative_time_idx=True,
    add_target_scales=True,
    add_encoder_length=True
)

# ایجاد مجموعه داده اعتبارسنجی
validation = TimeSeriesDataSet.from_dataset(training, df_cleaned, min_prediction_idx=training_cutoff + 1, stop_randomization=True)

# ایجاد DataLoaderها
batch_size = 16
train_dataloader = training.to_dataloader(train=True, batch_size=batch_size, num_workers=4)
val_dataloader = validation.to_dataloader(train=False, batch_size=batch_size, num_workers=4)

# ایجاد مدل Temporal Fusion Transformer
tft = TemporalFusionTransformer.from_dataset(
    training,
    hidden_size=64,  # افزایش اندازه مخفی برای قدرت بیشتر مدل
    attention_head_size=4,  # افزایش تعداد سرهای توجه
    dropout=0.1,
    hidden_continuous_size=36,  # افزایش اندازه ویژگی‌های مخفی پیوسته
    output_size=1,  # تعداد گام‌های پیش‌بینی، یک گام
    loss=RMSE(),  # استفاده از RMSE به عنوان تابع تلفات
    log_interval=10,
    learning_rate=0.03,
    reduce_on_plateau_patience=4
)

print(f"Number of parameters in network: {tft.size()/1e3:.1f}k")

# تنظیمات Trainer با Early Stopping و Learning Rate Monitoring
early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=1e-4, patience=10, verbose=False, mode="min")
lr_logger = LearningRateMonitor(logging_interval='step')

trainer = pl.Trainer(
    max_epochs=100,
    accelerator="auto",  # استفاده از GPU در صورت وجود
    gradient_clip_val=0.1,
    callbacks=[lr_logger, early_stop_callback],
    logger=TensorBoardLogger("lightning_logs")
)

# یافتن نرخ یادگیری بهینه
tuner = Tuner(trainer)
lr_finder = tuner.lr_find(
    tft, 
    train_dataloaders=train_dataloader, 
    val_dataloaders=val_dataloader,
    early_stop_threshold=1000.0,
    max_lr=0.3
)

# چاپ و نمایش نرخ یادگیری پیشنهادی
suggested_lr = lr_finder.suggestion()
print(f"Suggested learning rate: {suggested_lr}")

# تنظیم نرخ یادگیری مدل با نرخ یادگیری پیشنهادی
tft.learning_rate = suggested_lr

# آموزش مدل
trainer.fit(
    tft,
    train_dataloaders=train_dataloader,
    val_dataloaders=val_dataloader
)

# محاسبه RMSE پس از پایان آموزش با استفاده از tft.predict
predictions = tft.predict(val_dataloader)

# بررسی شکل پیش‌بینی‌ها
print(f"Shape of predictions: {predictions.shape}")

# استخراج مقادیر حقیقی از داده‌های اعتبارسنجی
true_values = np.concatenate([batch[1].cpu().numpy() for batch in val_dataloader])

# تبدیل پیش‌بینی‌ها به numpy و انتقال به CPU
pred_values = np.concatenate([pred.cpu().numpy() for pred in predictions])

# بررسی شکل مقادیر حقیقی
print(f"Shape of true values: {true_values.shape}")

# محاسبه RMSE
rmse = np.sqrt(mean_squared_error(true_values, pred_values))
print(f"RMSE: {rmse}")
