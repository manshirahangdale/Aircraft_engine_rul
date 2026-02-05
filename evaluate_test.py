import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path

TIME_STEPS = 30
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

PROCESSED_PATH = Path("data/processed")
RAW_PATH = Path("data/raw")

print(f"Using device: {DEVICE}")

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        return self.fc(out).squeeze()


# loading data
print("Loading data...")
test_df = pd.read_csv(PROCESSED_PATH / "test.csv")

true_rul = pd.read_csv(
    RAW_PATH / "RUL_FD001.txt", header=None
)[0].values

feature_cols = np.load(
    PROCESSED_PATH / "feature_cols.npy",
    allow_pickle=True
).tolist()

# loading model
input_size = len(feature_cols)
model = LSTMModel(input_size=input_size).to(DEVICE)
model.load_state_dict(torch.load("lstm_rul_model.pt", map_location=DEVICE))
model.eval()

print(f"Model loaded. Input size: {input_size}")

print("Making predictions...")
predictions = []

with torch.no_grad():
    for engine_id in sorted(test_df["engine_id"].unique()):
        engine_df = test_df[test_df["engine_id"] == engine_id]

        # Take LAST 30 cycles only
        if len(engine_df) < TIME_STEPS:
            seq = engine_df[feature_cols].values
            padding = np.zeros((TIME_STEPS - len(seq), len(feature_cols)))
            seq = np.vstack([padding, seq])
        else:
            seq = engine_df[feature_cols].values[-TIME_STEPS:]

        seq = torch.tensor(
            seq, dtype=torch.float32
        ).unsqueeze(0).to(DEVICE)

        pred = model(seq).item()
        predictions.append(pred)

predictions = np.array(predictions)

rmse = np.sqrt(np.mean((predictions - true_rul) ** 2))
mae = np.mean(np.abs(predictions - true_rul))

print("\n" + "="*50)
print("TEST RESULTS")
print("="*50)
print(f"Test RMSE: {rmse:.4f}")
print(f"Test MAE:  {mae:.4f}")
print("="*50)

results = pd.DataFrame({
    "engine_id": np.arange(1, len(predictions) + 1),
    "true_rul": true_rul,
    "predicted_rul": predictions,
    "error": predictions - true_rul,
    "abs_error": np.abs(predictions - true_rul)
})

results.to_csv("test_predictions.csv", index=False)
print("\nSaved test_predictions.csv")
print(f"\nFirst 10 predictions:")
print(results.head(10))