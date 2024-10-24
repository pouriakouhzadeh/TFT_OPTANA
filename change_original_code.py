import pandas as pd
import numpy as np
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
from pytorch_forecasting.data import GroupNormalizer
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_forecasting.metrics import RMSE
import pickle
from pytorch_forecasting.models.temporal_fusion_transformer.tuning import optimize_hyperparameters

# 1. Load the dataset
df = pd.read_csv("df_1_years.csv")

# Check for NA and infinite values in the target column
print(f"NA count in target column: {df['Sequence_10'].isna().sum()}")
print(f"Infinite count in target column: {np.isinf(df['Sequence_10']).sum()}")

# 2. Fill NA values and remove infinite values
df['Sequence_10'] = df['Sequence_10'].replace([np.inf, -np.inf], np.nan)
df['Sequence_10'] = df['Sequence_10'].interpolate(method='linear', limit_direction='both')
df['Sequence_10'] = df['Sequence_10'].fillna(df['Sequence_10'].mean())

# Re-check target column values after processing
print("Target column values after processing:")
print(df['Sequence_10'].describe())

# Check if there are still any invalid values
invalid_values = df[df['Sequence_10'].isna() | ~np.isfinite(df['Sequence_10'])]
if len(invalid_values) > 0:
    print(f"Invalid rows: {invalid_values}")
else:
    print("No invalid values found.")

# 3. Convert 'Time of Day' to minutes
df['time_idx'] = pd.to_datetime(df['Time of Day'], format='%H:%M:%S', errors='coerce')
df['time_idx'] = (df['time_idx'] - df['time_idx'].min()).dt.total_seconds() // 60
df['time_idx'] = df['time_idx'].astype(int)

# Check for duplicates in 'time_idx'
duplicates = df[df.duplicated(subset=['time_idx'], keep=False)]
print(f"Number of duplicate values in time index: {len(duplicates)}")

# Handle duplicates by making time_idx unique
df['time_idx'] = df.groupby('time_idx').cumcount() + df['time_idx']

# 4. Drop non-numeric columns
df = df.select_dtypes(include=[np.number])

# 5. Fill missing values in other columns
for col in df.columns:
    if col != 'Sequence_10':
        df[col] = df[col].replace([np.inf, -np.inf], np.nan)
        df[col] = df[col].interpolate(method='linear', limit_direction='both')
        df[col] = df[col].fillna(df[col].mean())

# 6. Add group_id column for uniform grouping
df['group_id'] = 0

target = 'Sequence_10'
feature_columns = df.columns.difference(['time_idx', target, 'group_id']).tolist()

# 7. Create TimeSeriesDataSet
max_encoder_length = 100
max_prediction_length = 10

training_cutoff = df['time_idx'].max() - max_prediction_length

# Ensure no missing values in target
df['Sequence_10'] = df['Sequence_10'].replace([np.inf, -np.inf], np.nan)
df['Sequence_10'] = df['Sequence_10'].interpolate(method='linear', limit_direction='both')
df['Sequence_10'] = df['Sequence_10'].fillna(df['Sequence_10'].mean())

# Create TimeSeriesDataSet for training
training = TimeSeriesDataSet(
    df[lambda x: x.time_idx <= training_cutoff],
    time_idx="time_idx",
    target="Sequence_10",
    group_ids=["group_id"],
    min_encoder_length=max_encoder_length // 2,
    max_encoder_length=max_encoder_length,
    min_prediction_length=1,
    max_prediction_length=max_prediction_length,
    time_varying_known_reals=["time_idx"] + feature_columns,
    time_varying_unknown_reals=["Sequence_10"],
    target_normalizer=GroupNormalizer(
        groups=["group_id"], transformation="softplus"
    ),
    add_relative_time_idx=True,
    add_target_scales=True,
    add_encoder_length=True,
    allow_missing_timesteps=True,  # Allow missing timesteps to handle irregular time indices
)

# Validation dataset
validation = TimeSeriesDataSet.from_dataset(training, df, predict=True, stop_randomization=True)

# 8. Create DataLoaders
batch_size = 64
train_dataloader = DataLoader(training, batch_size=batch_size, shuffle=True, num_workers=8)
val_dataloader = DataLoader(validation, batch_size=batch_size * 10, shuffle=False, num_workers=8)

# Set seed for reproducibility
pl.seed_everything(42)

# 9. Define and train Temporal Fusion Transformer (TFT) model
tft = TemporalFusionTransformer.from_dataset(
    training,
    learning_rate=4.78e-05,
    hidden_size=16,
    attention_head_size=8,
    dropout=0.1,
    hidden_continuous_size=8,
    output_size=1,
    loss=RMSE(),
    log_interval=10,
    reduce_on_plateau_patience=10,
)

# Trainer configuration
trainer = pl.Trainer(
    max_epochs=15,
    accelerator="gpu",
    devices=1,  # Adjusting to use 1 GPU
    gradient_clip_val=0.1,
    log_every_n_steps=10,
    callbacks=[
        pl.callbacks.EarlyStopping(monitor="val_loss", patience=10),
        pl.callbacks.LearningRateMonitor(logging_interval="step")
    ],
)

# Train the model
trainer.fit(tft, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)

# 10. Evaluate the model
def evaluate_model(tft, val_dataloader):
    predictions = tft.predict(val_dataloader, return_y=True, trainer_kwargs=dict(accelerator="gpu"))
    rmse_score = RMSE()(predictions.output, predictions.y)
    print(f"RMSE score: {rmse_score}")
    return rmse_score

# Run evaluation
rmse_result = evaluate_model(tft, val_dataloader)

# 11. Hyperparameter tuning with Optuna
def tune_hyperparameters(train_dataloader, val_dataloader):
    study = optimize_hyperparameters(
        train_dataloader,
        val_dataloader,
        model_path="optuna_study",
        n_trials=200,
        max_epochs=50,
        gradient_clip_val_range=(0.01, 1.0),
        hidden_size_range=(8, 128),
        hidden_continuous_size_range=(8, 128),
        attention_head_size_range=(1, 4),
        learning_rate_range=(0.001, 0.1),
        dropout_range=(0.1, 0.3),
        trainer_kwargs=dict(limit_train_batches=30),
        reduce_on_plateau_patience=4,
        use_learning_rate_finder=False,
    )
    
    with open("study_results.pkl", "wb") as f:
        pickle.dump(study, f)
    
    print("Best hyperparameters: ", study.best_trial.params)
    return study.best_trial.params

# Call hyperparameter tuning
best_hyperparameters = tune_hyperparameters(train_dataloader, val_dataloader)
