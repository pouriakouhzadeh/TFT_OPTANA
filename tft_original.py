import lightning.pytorch as pl
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
from torch.nn import MSELoss
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from pytorch_forecasting.metrics import RMSE
import logging
import warnings
import torch

# نمایش هشدارها
warnings.filterwarnings("ignore")

# تنظیمات logging
logging.basicConfig(
    filename='model_training.log', 
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# بارگذاری داده‌ها

df = pd.read_csv("df_1_years.csv")
# df = df[-(round(len(df)/4)):]


# پیش‌پردازش داده‌ها
def encode_time_of_day(time_str):
    time_in_seconds = pd.to_timedelta(time_str).total_seconds()
    max_time = 24 * 60 * 60  # تعداد ثانیه‌های یک روز
    time_sin = np.sin(2 * np.pi * time_in_seconds / max_time)
    time_cos = np.cos(2 * np.pi * time_in_seconds / max_time)
    return time_sin, time_cos

# ایجاد ویژگی‌های sin و cos برای زمان روز
df['time_sin'], df['time_cos'] = zip(*df['Time of Day'].apply(encode_time_of_day))

# حذف ستون‌های غیر مرتبط
df.drop(columns=['Unnamed: 0', 'Time of Day'], inplace=True)
# df.drop(columns=['Time of Day'], inplace=True)
# شیفت -10 برای ستون Sequence_10 برای پیش‌بینی 10 گام آینده
df['Sequence_10'] = df['Sequence_10'].shift(-10)
df.dropna(inplace=True)

# ایجاد ستون زمانی time_idx
df = df.sort_index()  # مرتب‌سازی بر اساس ترتیب ردیف‌ها
df['time_idx'] = np.arange(len(df))  # ایجاد ستون زمانی به عنوان شاخص زمانی پیوسته

# نرمال‌سازی داده‌ها
scaler = StandardScaler()
feature_columns = df.columns.difference(['Sequence_10', 'time_idx'])  # ستون‌های زمان و هدف حذف می‌شوند

df[feature_columns] = scaler.fit_transform(df[feature_columns])

# ذخیره نام ویژگی‌ها
saved_feature_columns = feature_columns

# ایجاد ستون 'ID' اگر وجود ندارد و تبدیل به نوع رشته
if 'ID' not in df.columns:
    df['ID'] = 1  # ایجاد شناسه یکتا اگر فقط یک سری زمانی دارید

df['ID'] = df['ID'].astype(str)  # تبدیل ستون ID به رشته

# تعریف نقطه قطع برای آموزش
training_cutoff = df["time_idx"].max() - 3000  # فرض می‌کنیم 30 روز آخر برای اعتبارسنجی است

# تعریف TimeSeriesDataSet برای آموزش
training = TimeSeriesDataSet(
    df[lambda x: x.time_idx <= training_cutoff],
    time_idx="time_idx",
    target="Sequence_10",
    group_ids=["ID"],  # استفاده از ستون 'ID' به عنوان شناسه سری زمانی
    max_encoder_length=30,  # میزان تاریخچه‌ای که مدل می‌تواند استفاده کند
    max_prediction_length=1,  # میزان پیش‌بینی آینده، یک گام بعدی
    static_categoricals=["ID"],
    static_reals=[],  # ویژگی‌های ثابت عددی (در صورت وجود)
    time_varying_known_categoricals=[],  # ویژگی‌های دسته‌ای شناخته شده در آینده
    time_varying_known_reals=["time_idx", "time_sin", "time_cos"],  # ویژگی‌های عددی شناخته شده در آینده
    time_varying_unknown_categoricals=[],  # ویژگی‌های دسته‌ای ناشناخته در آینده
    time_varying_unknown_reals=feature_columns.tolist(),  # ویژگی‌های عددی ناشناخته در آینده
    add_relative_time_idx=True,
    add_target_scales=True,
    add_encoder_length=True
)

# ایجاد مجموعه داده اعتبارسنجی
validation = TimeSeriesDataSet.from_dataset(training, df, min_prediction_idx=training_cutoff + 1, stop_randomization=True)

# ایجاد DataLoaderها
batch_size = 64
train_dataloader = training.to_dataloader(train=True, batch_size=batch_size, num_workers=4)
val_dataloader = validation.to_dataloader(train=False, batch_size=batch_size, num_workers=4)

# بازگرداندن پارامترها به حالت پیش‌فرض
tft = TemporalFusionTransformer.from_dataset(
    training,
    loss=RMSE(),  # استفاده از RMSE به عنوان تابع تلفات
    log_interval=10,
    learning_rate=1e-3  # نرخ یادگیری پیش‌فرض
)

# تنظیمات Trainer با Early Stopping و Learning Rate Monitoring
early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=1e-4, patience=10, verbose=False, mode="min")
lr_logger = LearningRateMonitor(logging_interval='step')

trainer = pl.Trainer(
    max_epochs=100,  # افزایش تعداد epoch ها برای بهبود آموزش
    accelerator="gpu",  # مشخص کردن استفاده از GPU
    devices=[1],  # استفاده از GPU شماره 1
    gradient_clip_val=0.1,
    callbacks=[lr_logger, early_stop_callback],
    logger=TensorBoardLogger("lightning_logs")
)


# آموزش مدل
trainer.fit(
    tft,
    train_dataloaders=train_dataloader,
    val_dataloaders=val_dataloader
)

# پیش‌بینی با مدل
predictions = tft.predict(val_dataloader)

# استخراج مقادیر حقیقی از داده‌های اعتبارسنجی با بررسی ساختار batch[1]
true_values = []
for batch in val_dataloader:
    target = batch[1]  # فرض می‌کنیم target در مکان دوم تاپل قرار دارد
    if target is not None:  # حذف مقادیر None
        if isinstance(target, tuple):  # اگر target یک تاپل است
            target = target[0]  # استخراج اولین عنصر تاپل
        if isinstance(target, torch.Tensor):
            target = target.detach().cpu().numpy()
        # بررسی اینکه target دارای ابعاد یکنواخت است
        if target.ndim == 2 and target.shape[1] == 1:  # چک کردن اینکه target یک آرایه 2 بعدی با یک ستون است
            true_values.append(target.flatten())  # تبدیل به آرایه 1 بعدی
        else:
            print(f"Shape mismatch in batch: {target.shape}")

# اطمینان از اینکه true_values یک آرایه numpy یکدست است
if len(true_values) > 0:
    true_values = np.concatenate([x for x in true_values if x is not None], axis=0)
    print(f"Shape of true values: {true_values.shape}")
else:
    print("No valid true values found!")
    true_values = None

# تبدیل پیش‌بینی‌ها به numpy و انتقال به CPU
pred_values = np.concatenate([pred.detach().cpu().numpy() for pred in predictions])

# بررسی و تطبیق اندازه داده‌ها
min_len = min(len(true_values), len(pred_values))  # یافتن کوچکترین تعداد داده
true_values = true_values[:min_len]  # برش true_values به اندازه min_len
pred_values = pred_values[:min_len]  # برش pred_values به اندازه min_len

# بررسی تطابق اندازه‌ها
print(f"Shape of true values: {true_values.shape}")
print(f"Shape of predictions: {pred_values.shape}")

# Save predictions to CSV
df_out = pd.DataFrame({
    'true_values': true_values,
    'predictions': pred_values
})
df_out.to_csv("output.csv", index=False)

if pred_values.shape == true_values.shape:
    # محاسبه RMSE اگر شکل‌ها مطابق باشند
    rmse = np.sqrt(mean_squared_error(true_values, pred_values))
    print(f"RMSE: {rmse}")
else:
    print("Error: Cannot calculate RMSE due to shape mismatch.")
