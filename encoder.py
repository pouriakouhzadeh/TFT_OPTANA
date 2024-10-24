import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
from torch.cuda.amp import GradScaler, autocast
from torch.utils.checkpoint import checkpoint
import math

# Positional Embedding
class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # Add a batch dimension
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]  # Add positional encoding to the input
        return x

# Load the dataset
df = pd.read_csv('/mnt/data/df.csv')

# Shift the 'Sequence_10' column by -10 to predict 10 steps ahead
df['Target'] = df['Sequence_10'].shift(-10)
df = df.dropna()

data = df['Sequence_10'].values
target = df['Target'].values

scaler_data = StandardScaler()
scaler_target = StandardScaler()

data = scaler_data.fit_transform(data.reshape(-1, 1)).flatten()
target = scaler_target.fit_transform(target.reshape(-1, 1)).flatten()

def create_sequences(data, target, seq_len):
    sequences = []
    targets = []
    for i in range(len(data) - seq_len):
        sequences.append(data[i:i + seq_len])
        targets.append(target[i + seq_len])
    return np.array(sequences), np.array(targets)

# Model configuration
input_size = 1
hidden_size = 16  # Further reduced hidden size to save memory
output_size = 1
num_layers = 1  # Reduced number of LSTM layers
d_model = 16  # Further reduced model size
seq_len = 30  # Reduced sequence length to save memory
n_heads = 1  # Reduced attention heads
target_len = 10
dropout_prob = 0.3

# Encoder with LSTM and Positional Embedding
class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, d_model, seq_len, n_heads, dropout_prob):
        super(Encoder, self).__init__()
        self.pos_embedding = PositionalEmbedding(d_model, max_len=seq_len)
        self.attention = AttentionBlock(d_model, n_heads)
        self.reduce_dim = nn.Linear(d_model, input_size)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout_prob)

    def forward(self, x):
        x = self.pos_embedding(x)
        x = checkpoint(self.attention, x)  # Use checkpointing to reduce memory usage
        x = self.reduce_dim(x)
        h0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size).to(x.device)
        c0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size).to(x.device)
        output, (hn, cn) = self.lstm(x, (h0, c0))
        return output, hn, cn

# Attention Block (used inside the encoder)
class AttentionBlock(nn.Module):
    def __init__(self, d_model, n_heads):
        super(AttentionBlock, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, batch_first=True)
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, x):
        attn_output, _ = self.multihead_attn(x, x, x)
        x = self.layer_norm(x + self.dropout(attn_output))
        return x

# Decoder with LSTM
class Decoder(nn.Module):
    def __init__(self, hidden_size, output_size, num_layers, dropout_prob):
        super(Decoder, self).__init__()
        self.lstm = nn.LSTM(output_size, hidden_size, num_layers, batch_first=True, dropout=dropout_prob)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden, cell):
        output, (hn, cn) = self.lstm(x, (hidden, cell))
        output = self.fc(output)
        return output, hn, cn

# Complete Encoder-Decoder Model
class EncoderDecoderModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, d_model, seq_len, n_heads, dropout_prob):
        super(EncoderDecoderModel, self).__init__()
        self.encoder = Encoder(input_size, hidden_size, num_layers, d_model, seq_len, n_heads, dropout_prob)
        self.decoder = Decoder(hidden_size, output_size, num_layers, dropout_prob)

    def forward(self, x, target_len):
        encoded_output, hn, cn = self.encoder(x)
        decoder_input = torch.zeros(x.size(0), 1, x.size(2)).to(x.device)
        outputs = []
        for t in range(target_len):
            decoder_output, hn, cn = self.decoder(decoder_input, hn, cn)
            outputs.append(decoder_output)
            decoder_input = decoder_output
        outputs = torch.cat(outputs, dim=1)
        return outputs

# Preparing data for training and testing
train_data_seq, train_target_seq = create_sequences(data, target, seq_len)

# Convert to torch tensors
train_data_seq = torch.tensor(train_data_seq, dtype=torch.float32).unsqueeze(-1)
train_target_seq = torch.tensor(train_target_seq, dtype=torch.float32).unsqueeze(-1)

# Split into train and test sets
train_data_seq, test_data_seq, train_target_seq, test_target_seq = train_test_split(
    train_data_seq, train_target_seq, test_size=0.1, shuffle=False
)

# Move data to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_data_seq, train_target_seq = train_data_seq.to(device), train_target_seq.to(device)
test_data_seq, test_target_seq = test_data_seq.to(device), test_target_seq.to(device)

# Create the model
model = EncoderDecoderModel(input_size, hidden_size, output_size, num_layers, d_model, seq_len, n_heads, dropout_prob).to(device)

# Mixed Precision Training
scaler = GradScaler()

# Optimizer, scheduler, and loss function
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
criterion = nn.MSELoss()

# Training loop with mixed precision and early stopping
best_loss = float('inf')
early_stopping_counter = 0
early_stopping_patience = 10

for epoch in range(100):  # Set a max epoch limit
    model.train()
    epoch_loss = 0.0
    
    for i in range(0, len(train_data_seq), 32):  # Mini-batch training (batch size 32)
        inputs = train_data_seq[i:i + 32]
        targets = train_target_seq[i:i + 32]

        optimizer.zero_grad()

        # Use mixed precision during forward pass
        with autocast():
            outputs = model(inputs, target_len)
            loss = criterion(outputs, targets)

        # Backward pass with mixed precision
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        epoch_loss += loss.item()

    # Scheduler step after epoch
    scheduler.step(epoch_loss)

    # Calculate average loss for this epoch
    avg_loss = epoch_loss / (len(train_data_seq) // 32)

    print(f"Epoch {epoch + 1}, Loss: {avg_loss:.6f}")

    # Early stopping logic
    if avg_loss < best_loss:
        best_loss = avg_loss
        early_stopping_counter = 0
        torch.save(model.state_dict(), 'best_model.pth')  # Save the best model
    else:
        early_stopping_counter += 1

    if early_stopping_counter >= early_stopping_patience:
        print("Early stopping triggered")
        break

# Testing the model
model.eval()
with torch.no_grad():
    test_loss = 0.0
    for i in range(0, len(test_data_seq), 32):
        inputs = test_data_seq[i:i + 32]
        targets = test_target_seq[i:i + 32]
        with autocast():
            outputs = model(inputs, target_len)
            loss = criterion(outputs, targets)
        test_loss += loss.item()

    avg_test_loss = test_loss / (len(test_data_seq) // 32)
    print(f"Test Loss: {avg_test_loss:.6f}")

# Inverse scale the predictions for real values (optional)
predictions = scaler_target.inverse_transform(outputs.cpu().numpy().reshape(-1,
