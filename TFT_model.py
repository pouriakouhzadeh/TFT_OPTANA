import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import logging

# تنظیمات logging
logging.basicConfig(
    filename='model_training.log', 
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s'
)

df = pd.read_csv("filtered_df_7_years.csv")

# Data Preprocessing
def encode_time_of_day(time_str):
    time_in_seconds = pd.to_timedelta(time_str).total_seconds()
    max_time = 24 * 60 * 60  # seconds in a day
    time_sin = np.sin(2 * np.pi * time_in_seconds / max_time)
    time_cos = np.cos(2 * np.pi * time_in_seconds / max_time)
    return time_sin, time_cos

# ایجاد ویژگی‌های sin و cos برای time of day
df['time_sin'], df['time_cos'] = zip(*df['Time of Day'].apply(encode_time_of_day))

# حذف ستون‌های غیر مرتبط
df.drop(columns=['Unnamed: 0', 'Time of Day'], inplace=True)

# نرمال‌سازی داده‌ها
scaler = StandardScaler()
feature_columns = df.columns.difference(['Sequence_10'])
df[feature_columns] = scaler.fit_transform(df[feature_columns])

# تقسیم داده‌ها به train و validation
df_train, df_val = train_test_split(df, test_size=0.2, random_state=42, shuffle=True)

# Custom Dataset class
class TimeSeriesDataset(Dataset):
    def __init__(self, df, sequence_length):
        self.features = df[feature_columns].values
        self.target = df['Sequence_10'].values
        self.sequence_length = sequence_length

    def __len__(self):
        return len(self.features) - self.sequence_length

    def __getitem__(self, idx):
        x = self.features[idx:idx + self.sequence_length]
        y = self.target[idx + self.sequence_length]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

# ساخت دیتاست‌های train و validation
train_dataset = TimeSeriesDataset(df_train, 100)
val_dataset = TimeSeriesDataset(df_val, 100)

# DataLoader برای train و validation
train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=4, pin_memory=True)
val_dataloader = DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=4, pin_memory=True)

# Define the TFT model
class TFTModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_heads, sequence_length, num_layers=2, dropout=0.3):
        super(TFTModel, self).__init__()
        # افزایش تعداد لایه‌های LSTM به 2
        self.encoder = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True, dropout=dropout)
        self.positional_embedding = nn.Parameter(torch.randn(1, sequence_length, input_dim))
        self.attention = nn.MultiheadAttention(hidden_dim, n_heads, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        positional_embedding = self.positional_embedding[:, :seq_len, :].clone()
        x = x + positional_embedding
        x, _ = self.encoder(x)
        x, _ = self.attention(x, x, x)
        x = self.fc(x[:, -1, :])
        return x

# Hyperparameters
input_dim = len(feature_columns)
hidden_dim = 128  # افزایش hidden_dim برای بهبود قدرت مدل
output_dim = 1
n_heads = 4
sequence_length = 100
batch_size = 128
learning_rate = 1e-4
epochs = 120
patience = 10  # patience برای early stopping

# Model, optimizer, and loss function
model = TFTModel(input_dim, hidden_dim, output_dim, n_heads, sequence_length, num_layers=2, dropout=0.3).cuda()
optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
loss_fn = nn.MSELoss()

# استفاده از Scheduler برای کاهش نرخ یادگیری به صورت پویا
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)

# Multi-GPU setup
device_ids = [0, 1]
model = nn.DataParallel(model, device_ids=device_ids)

# Early Stopping class
class EarlyStopping:
    def __init__(self, patience=10, delta=0.01):
        self.patience = patience
        self.delta = delta
        self.best_loss = None
        self.counter = 0
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

# Training loop with validation, gradient clipping, and early stopping
def train_model(model, train_dataloader, val_dataloader, optimizer, loss_fn, epochs, scheduler, early_stopping):
    for epoch in range(epochs):
        model.train()
        total_train_loss = 0

        # آموزش روی داده‌های training
        for batch_x, batch_y in train_dataloader:
            batch_x, batch_y = batch_x.cuda(), batch_y.cuda()
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = loss_fn(outputs.squeeze(), batch_y)
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_dataloader)

        # اعتبارسنجی روی داده‌های validation
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for batch_x, batch_y in val_dataloader:
                batch_x, batch_y = batch_x.cuda(), batch_y.cuda()
                outputs = model(batch_x)
                val_loss = loss_fn(outputs.squeeze(), batch_y)
                total_val_loss += val_loss.item()

        avg_val_loss = total_val_loss / len(val_dataloader)

        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss}, Validation Loss: {avg_val_loss}, LR: {optimizer.param_groups[0]['lr']}")
        logging.info(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss}, Validation Loss: {avg_val_loss}, LR: {optimizer.param_groups[0]['lr']}")

        # به‌روزرسانی نرخ یادگیری توسط Scheduler
        scheduler.step(avg_val_loss)

        # بررسی Early Stopping
        early_stopping(avg_val_loss)
        if early_stopping.early_stop:
            print(f"Early stopping triggered at epoch {epoch+1}")
            logging.info(f"Early stopping triggered at epoch {epoch+1}")
            break

# تعریف Early Stopping
early_stopping = EarlyStopping(patience=patience, delta=0.01)

# Train the model
train_model(model, train_dataloader, val_dataloader, optimizer, loss_fn, epochs, scheduler, early_stopping)

# ذخیره مدل آموزش دیده
torch.save(model.state_dict(), 'TFT_model.pth')

# لود دیتاست جدید برای پیش‌بینی
df_test = pd.read_csv('df.csv')

# تولید ویژگی‌های time_sin و time_cos برای دیتاست تست
df_test['time_sin'], df_test['time_cos'] = zip(*df_test['Time of Day'].apply(encode_time_of_day))

# شیفت -10 برای ستون Sequence_10
df_test['Sequence_10'] = df_test['Sequence_10'].shift(-10)
df_test.dropna(inplace=True)

# نرمال‌سازی داده‌ها
df_test[feature_columns] = scaler.transform(df_test[feature_columns])

# ساخت دیتاست مشابه برای تست
test_dataset = TimeSeriesDataset(df_test, sequence_length)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

# پیش‌بینی با مدل آموزش دیده
model.eval()
predictions = []
true_values = []

with torch.no_grad():
    for batch_x, batch_y in test_dataloader:
        batch_x = batch_x.cuda()
        outputs = model(batch_x)
        predictions.extend(outputs.cpu().numpy())
        true_values.extend(batch_y.cpu().numpy())

# تبدیل لیست‌ها به آرایه‌های numpy
predictions = np.array(predictions).flatten()
true_values = np.array(true_values).flatten()

# محاسبه RMSE
rmse = np.sqrt(mean_squared_error(true_values, predictions))
print(f"RMSE: {rmse}")

logging.info(f"RMSE: {rmse}")
