import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
import copy

# Use cuDNN benchmark for optimal GPU performance
torch.backends.cudnn.benchmark = True

# Load the dataset
file_path = "df_1_years.csv"
df = pd.read_csv(file_path)

# Shift the 'Sequence_10' column by -10 to predict 10 steps ahead
df['Target'] = df['Sequence_10'].shift(-10)

# Drop the rows with NaN values (created by the shift)
df = df.dropna()

# Extract the shifted sequence_10 column and the target
data = df['Sequence_10'].values
target = df['Target'].values

# Normalize data using StandardScaler
scaler_data = StandardScaler()
scaler_target = StandardScaler()
data = scaler_data.fit_transform(data.reshape(-1, 1)).flatten()
target = scaler_target.fit_transform(target.reshape(-1, 1)).flatten()

# Split the data into train and test sets (90% train, 10% test)
train_data, test_data, train_target, test_target = train_test_split(data, target, test_size=0.1, shuffle=False)

# Sample Dataset class for time series data
class TimeSeriesDataset(Dataset):
    def __init__(self, data, target, seq_len):
        self.data = data
        self.target = target
        self.seq_len = seq_len

    def __len__(self):
        return len(self.data) - self.seq_len

    def __getitem__(self, index):
        return (
            self.data[index : index + self.seq_len],
            self.target[index + self.seq_len],
        )

# Defining the Fedformer model with Dropout
class Fedformer(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=4):  # Increased to 4 layers
        super(Fedformer, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(p=0.3)  # Dropout with 30%

    def forward(self, x):
        h0 = torch.zeros(4, x.size(0), hidden_size).to(x.device).float()  # Adjusted for num_layers=4
        c0 = torch.zeros(4, x.size(0), hidden_size).to(x.device).float()  # Adjusted for num_layers=4
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        out = self.dropout(out)  # Apply dropout
        return out

# Configurations
input_size = 1
hidden_size = 512  # Increased hidden size for better learning
output_size = 1
seq_len = 50  # Increased sequence length for capturing more context
num_layers = 4  # Increased to 4 layers
epochs = 10  # Increased epochs
lr_base = 0.0005  # Reduced learning rate for better convergence
batch_size = 512  # Increased batch size for better GPU utilization

# Early Stopping Setup
early_stopping_patience = 10
best_loss = float('inf')
early_stopping_counter = 0
best_model_state = None

# Check if GPU is available and use multiple GPUs
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs.")
    model = nn.DataParallel(Fedformer(input_size, hidden_size, output_size, num_layers))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Move the model to the available GPU(s)
model = model.to(device)

# Create training dataset and dataloader
train_dataset = TimeSeriesDataset(torch.FloatTensor(train_data).unsqueeze(-1), torch.FloatTensor(train_target).unsqueeze(-1), seq_len)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=8)  # Increased num_workers to 8

# Create test dataset and dataloader
test_dataset = TimeSeriesDataset(torch.FloatTensor(test_data).unsqueeze(-1), torch.FloatTensor(test_target).unsqueeze(-1), seq_len)
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, pin_memory=True)

# Define learning rate based on sequence length
lr = lr_base * (30 / seq_len)  # Scale learning rate by seq_len

# Optimizer with weight decay for regularization
optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)

# Scheduler using ReduceLROnPlateau
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

# Use Huber Loss
criterion = nn.SmoothL1Loss()  # Huber loss is also called SmoothL1Loss

# Mixed Precision Training
scaler = torch.amp.GradScaler()

# Training Loop with Early Stopping
for epoch in range(epochs):
    epoch_loss = 0.0
    model.train()  # Set the model to training mode
    for x, y in train_dataloader:
        x, y = x.to(device).float(), y.to(device).float()  # Move data to GPU and ensure float32
        
        # Zero the parameter gradients
        optimizer.zero_grad()

        # Mixed Precision with autocast
        with torch.amp.autocast(device_type='cuda'):
            outputs = model(x)
            loss = criterion(outputs, y)

        # Backward pass
        scaler.scale(loss).backward()

        # Optimizer step
        scaler.step(optimizer)
        scaler.update()
        
        epoch_loss += loss.item()
    
    # Calculate average loss for the epoch
    avg_epoch_loss = epoch_loss / len(train_dataloader)
    
    # Step the scheduler based on the validation loss
    scheduler.step(avg_epoch_loss)

    print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_epoch_loss:.4f}")

    # Early stopping logic
    if avg_epoch_loss < best_loss:
        best_loss = avg_epoch_loss
        best_model_state = copy.deepcopy(model.state_dict())  # Save the best model
        early_stopping_counter = 0  # Reset the early stopping counter
    else:
        early_stopping_counter += 1
        if early_stopping_counter >= early_stopping_patience:
            print("Early stopping triggered.")
            break

# Load the best model if early stopping was triggered
if best_model_state:
    model.load_state_dict(best_model_state)

print("Training complete!")

# Testing and evaluation
model.eval()  # Set the model to evaluation mode
predictions = []
actuals = []

with torch.no_grad():
    for x, y in test_dataloader:
        x = x.to(device).float()
        y = y.to(device).float()

        # Predict using the model
        with torch.amp.autocast(device_type='cuda'):
            output = model(x)
        predictions.append(output.item())
        actuals.append(y.item())

# Calculate Mean Squared Error as the evaluation metric
mse = mean_squared_error(actuals, predictions)
rmse = np.sqrt(mse)
print(f"Test MSE: {mse:.4f}")
print(f"Test RMSE: {rmse:.4f}")