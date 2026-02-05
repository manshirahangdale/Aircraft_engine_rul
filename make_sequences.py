import numpy as np
import pandas as pd
from pathlib import Path

TIME_STEPS = 30
TARGET = "RUL"

PROCESSED_PATH = Path("data/processed")


def load_data():
    train = pd.read_csv(PROCESSED_PATH / "train.csv")
    val = pd.read_csv(PROCESSED_PATH / "val.csv")
    test = pd.read_csv(PROCESSED_PATH / "test.csv")
    return train, val, test


def create_sequences(df, time_steps, target_col=None):
    X, y = [], []

    feature_cols = df.columns.tolist()
    # Remove non-feature columns
    for col in ["engine_id", "cycle", target_col]:
        if col in feature_cols:
            feature_cols.remove(col)

    for engine_id in df["engine_id"].unique():
        engine_df = df[df["engine_id"] == engine_id]
        features = engine_df[feature_cols].values

        if target_col:
            targets = engine_df[target_col].values

        for i in range(len(engine_df) - time_steps + 1):
            X.append(features[i:i + time_steps])

            if target_col:
                y.append(targets[i + time_steps - 1])

    X = np.array(X)

    if target_col:
        y = np.array(y)
        return X, y

    return X


def main():
    print("Loading processed data...")
    train, val, test = load_data()

    print("Creating train sequences...")
    X_train, y_train = create_sequences(train, TIME_STEPS, TARGET)

    print("Creating validation sequences...")
    X_val, y_val = create_sequences(val, TIME_STEPS, TARGET)

    print("Creating test sequences...")
    X_test = []
    test_engine_ids = []

    feature_cols = test.columns.tolist()
    # Remove non-feature columns
    for col in ["engine_id", "cycle"]:
        if col in feature_cols:
            feature_cols.remove(col)

    for engine_id in test["engine_id"].unique():
        engine_df = test[test["engine_id"] == engine_id]
        features = engine_df[feature_cols].values

        for i in range(len(engine_df) - TIME_STEPS + 1):
            X_test.append(features[i:i + TIME_STEPS])
            test_engine_ids.append(engine_id)

    X_test = np.array(X_test)
    test_engine_ids = np.array(test_engine_ids)

    print("Saving NumPy arrays...")
    np.save(PROCESSED_PATH / "X_train.npy", X_train)
    np.save(PROCESSED_PATH / "y_train.npy", y_train)
    np.save(PROCESSED_PATH / "X_val.npy", X_val)
    np.save(PROCESSED_PATH / "y_val.npy", y_val)
    np.save(PROCESSED_PATH / "X_test.npy", X_test)
    np.save(PROCESSED_PATH / "test_engine_ids.npy", test_engine_ids)

    print("Done.")
    print("X_train:", X_train.shape)
    print("y_train:", y_train.shape)
    print("X_val:", X_val.shape)
    print("y_val:", y_val.shape)
    print("X_test:", X_test.shape)
    print("test_engine_ids:", test_engine_ids.shape)


# ADD THIS LINE - this was missing!
if __name__ == "__main__":
    main()
