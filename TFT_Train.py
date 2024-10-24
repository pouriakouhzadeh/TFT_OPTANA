import pandas as pd
import torch
from pytorch_lightning import LightningModule
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer, Baseline
from pytorch_forecasting.data.encoders import NaNLabelEncoder
from pytorch_forecasting.metrics import MAE
from pytorch_lightning import Trainer

# بارگذاری داده‌ها
df = pd.read_csv('df.csv')

# قدم اول: پیش‌پردازش داده‌ها
df['Time of Day'] = pd.to_datetime(df['Time of Day'], format='%H:%M:%S')

# ایجاد شاخص زمانی (time_idx) برای هر دقیقه
df['time_idx'] = (df['Time of Day'] - df['Time of Day'].min()).dt.total_seconds() // 60

# تعریف طول داده‌های تاریخی و طول پیش‌بینی
max_encoder_length = 60  # طول تاریخچه ۶۰ دقیقه
max_prediction_length = 10  # پیش‌بینی برای ۱۰ دقیقه بعد

# قدم دوم: ایجاد مجموعه داده‌ها برای TFT
training_cutoff = df['time_idx'].max() - max_prediction_length

training = TimeSeriesDataSet(
    df[df.time_idx <= training_cutoff],
    time_idx='time_idx',
    target='Sequence_10',
    group_ids=['Index'],  # فرض بر این است که ستون 'Index' سری زمانی را مشخص می‌کند
    min_encoder_length=max_encoder_length // 2,  # حداقل طول تاریخچه
    max_encoder_length=max_encoder_length,
    min_prediction_length=max_prediction_length,
    max_prediction_length=max_prediction_length,
    static_categoricals=[],  # اضافه کردن ویژگی‌های ثابت اگر لازم باشد
    time_varying_known_reals=['time_idx'],  # ویژگی‌های شناخته شده (مثلاً شاخص زمانی)
    time_varying_unknown_reals=[col for col in df.columns if 'Sequence_10' in col or 'p_' in col or 'hl_' in col],  # ویژگی‌های واقعی که ناشناخته هستند
    target_normalizer=NaNLabelEncoder()  # نرمال‌سازی هدف برای یادگیری بهتر
)

# قدم سوم: ایجاد مجموعه داده برای اعتبارسنجی
validation = TimeSeriesDataSet.from_dataset(training, df, min_prediction_idx=training_cutoff + 1)

# تعریف دیتالودرها
batch_size = 64
train_dataloader = training.to_dataloader(train=True, batch_size=batch_size)
val_dataloader = validation.to_dataloader(train=False, batch_size=batch_size)

# قدم چهارم: ساخت و آموزش مدل TFT
tft = TemporalFusionTransformer.from_dataset(
    training,
    learning_rate=0.03,
    hidden_size=16,  # تعداد لایه‌های مخفی
    attention_head_size=1,
    dropout=0.1,
    hidden_continuous_size=8,
    output_size=7,  # ۷ کوانتیل پیش‌فرض
    loss=MAE(),
    reduce_on_plateau_patience=4  # کاهش نرخ یادگیری در صورت عدم بهبود
)

# آموزش
trainer = Trainer(
    max_epochs=30,
    gpus=0  # اگر GPU دارید، مقدار را به 1 تغییر دهید
)

trainer.fit(
    tft, 
    train_dataloaders=train_dataloader, 
    val_dataloaders=val_dataloader
)

# قدم پنجم: پیش‌بینی
best_model = tft.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)

# پیش‌بینی بر روی مجموعه اعتبارسنجی
raw_predictions, x = best_model.predict(val_dataloader, mode="raw", return_x=True)

# نمایش مقادیر پیش‌بینی شده برای Sequence_10
print("Predicted values for Sequence_10:", raw_predictions)
