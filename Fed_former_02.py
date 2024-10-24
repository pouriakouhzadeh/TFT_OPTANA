import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np

# Load the dataset
file_path = "df.csv"
df = pd.read_csv(file_path)

# Shift the 'Sequence_10' column by -10 to predict 10 steps ahead
df['Target'] = df['Sequence_10'].shift(-10)

# Drop the rows with NaN values (created by the shift)
df = df.dropna()

# Extract the shifted sequence_10 column and the target
data = df['Sequence_10'].values
target = df['Target'].values

# Sample Dataset class for time series data (Reduced sample to 20%)
class TimeSeriesDataset(Dataset):
    def __init__(self, data, target, seq_len):
        self.data = data[:int(len(data) * 0.2)]  # Use 20% of the data
        self.target = target[:int(len(target) * 0.2)]
        self.seq_len = seq_len

    def __len__(self):
        return len(self.data) - self.seq_len

    def __getitem__(self, index):
        return (
            self.data[index : index + self.seq_len],
            self.target[index + self.seq_len],
        )

# Defining the Fedformer model (with reduced hidden size and layers)
class Fedformer(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):  # Reduced num_layers to 1
        super(Fedformer, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(1, x.size(0), hidden_size).to(x.device).float()  # Ensure h0 is float32
        c0 = torch.zeros(1, x.size(0), hidden_size).to(x.device).float()  # Ensure c0 is float32
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

# Configurations
input_size = 1
hidden_size = 32  # Reduced hidden size to reduce memory consumption
output_size = 1
seq_len = 10
num_layers = 1  # Reduced number of layers
epochs = 100
lr = 0.001
batch_size = 4  # Further reduced batch size

# Check if GPU is available and use it if possible
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Convert data and target to PyTorch tensors and create dataset
dataset = TimeSeriesDataset(torch.FloatTensor(data).unsqueeze(-1), torch.FloatTensor(target).unsqueeze(-1), seq_len)  # Use float32
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Initialize model, loss, and optimizer
model = Fedformer(input_size, hidden_size, output_size, num_layers).to(device).float()  # Use float32 for the model
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

# Training Loop
for epoch in range(epochs):
    for x, y in dataloader:
        x, y = x.to(device).float(), y.to(device).float()  # Move data to GPU and ensure float32
        optimizer.zero_grad()
        outputs = model(x)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()

    print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

print("Training complete!")
