# Aircraft Engine Remaining Useful Life (RUL) Prediction

Deep Learning project for predicting the Remaining Useful Life (RUL) of aircraft turbofan engines using LSTM networks on NASA's C-MAPSS dataset.

## Project Overview

This project implements a Long Short-Term Memory (LSTM) neural network to predict the remaining useful life of aircraft engines based on sensor readings. The model achieves a **Test RMSE of 22.09** and **MAE of 16.38** on the NASA Turbofan Engine Degradation Simulation dataset (FD001).

## Key Features
- Time-series sequence modeling using LSTM
- Automated data preprocessing pipeline
- Train/Validation/Test split by engine ID
- Feature engineering with low-variance sensor removal
- Standardized sensor readings using StandardScaler
- Early stopping with best model checkpointing

## Results

| Metric | Value |
|--------|-------|
| **Test RMSE** | 22.09 |
| **Test MAE** | 16.38 |
| **Best Val MSE** | 706.98 |

### Model Performance
- Training samples: 14,241 sequences
- Validation samples: 3,490 sequences
- Test engines: 100
- Sequence length: 30 time steps
- Features: 16 (after removing low-variance sensors)

**LSTM Model:**
- Input: (batch_size, 30, 16) - 30 timesteps with 16 features
- LSTM layers: 2 layers with 64 hidden units
- Dropout: 0.2 (between LSTM layers)
- Output: Single RUL value (regression)
- Loss function: Mean Squared Error (MSE)
- Optimizer: Adam (lr=0.001)

## Dataset
-Source: NASA C-MAPSS (FD001)
-Engines: 100 train, 100 test
-Sensors: 21 (16 after removing low-variance)
-Timesteps per sequence: 30

**Dataset characteristics:**
- Training engines: 100
- Test engines: 100
- Operating conditions: 1 (Sea level)
- Fault modes: 1 (HPC Degradation)
- Sensor readings: 21 sensors
- Operating settings: 3

# 1. Preprocess data
python preprocess.py
This script:
- Loads raw sensor data
- Calculates RUL (Remaining Useful Life) for each cycle
- Removes low-variance sensors
- Splits data into train/validation sets
- Applies StandardScaler normalization

# 2. check features
python checkdata.py

# 3. Create sequences
python make_sequences.py
Creates sliding window sequences of 30 timesteps for LSTM input.

# 4. Train model
python train_lstm.py
Trains the LSTM model for 25 epochs with early stopping.

# 5. Evaluate
python evaluate_test.py
Generates predictions and calculates RMSE/MAE metrics.

## Key Learnings

- Handling time-series data with sequential dependencies
- Implementing LSTM networks in PyTorch
- Proper train/validation/test splitting for time-series data
- Feature engineering for sensor data
- Hyperparameter tuning for optimal performance

## Future Improvements

- Implement attention mechanisms
-  Try bidirectional LSTM
-  Experiment with GRU layers
-  Add ensemble methods
-  Implement RUL clipping (common in literature)
-  Extend to FD002, FD003, FD004 datasets
-  Add visualization dashboard
