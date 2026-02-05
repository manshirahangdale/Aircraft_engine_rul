import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from pathlib import Path

# setting paths
raw_data = Path("data/raw")
processed_data = Path("data/processed")
processed_data.mkdir(parents=True, exist_ok=True)

columns = (
    ['engine_id', 'cycle']
    + [f'setting{i}' for i in range(1,4)]
    + [f's{i}' for i in range(1,22)]
)

# loading data (train,test)
def load_data():
    train = pd.read_csv(
        raw_data / "train_FD001.txt",
        sep=r"\s+",
        header=None
    )

    test = pd.read_csv(
        raw_data / "test_FD001.txt",
        sep=r"\s+",
        header=None
    )

    train.columns = columns
    test.columns = columns

    return train, test

def add_rul(train_df):
    max_cycles = train_df.groupby("engine_id")["cycle"].max()
    train_df["RUL"] = train_df.apply(
        lambda row: max_cycles[row["engine_id"]] - row["cycle"],
        axis=1
    )
    return train_df

def drop_low_variance_sensors(train_df, test_df, threshold=1e-3):
    sensor_cols = [c for c in train_df.columns if c.startswith("s")]
    low_var = train_df[sensor_cols].std()
    drop_cols = low_var[low_var < threshold].index.tolist()

    train_df = train_df.drop(columns=drop_cols)
    test_df = test_df.drop(columns=drop_cols)

    feature_cols = [
        c for c in train_df.columns
        if c not in ["engine_id", "cycle", "RUL"]
    ]

    np.save("data/processed/feature_cols.npy", feature_cols)

    return train_df, test_df

def split_and_scale(train_df, test_df):
    features = [c for c in train_df.columns 
                if c not in ["engine_id", "cycle", "RUL"]]

    engine_ids = train_df["engine_id"].unique()
    train_ids, val_ids = train_test_split(
        engine_ids, test_size=0.2, random_state=42
    )

    # Use .loc to avoid SettingWithCopyWarning
    train_split = train_df.loc[train_df["engine_id"].isin(train_ids)].copy()
    val_split = train_df.loc[train_df["engine_id"].isin(val_ids)].copy()

    scaler = StandardScaler()
    train_split.loc[:, features] = scaler.fit_transform(train_split[features])
    val_split.loc[:, features] = scaler.transform(val_split[features])
    test_df.loc[:, features] = scaler.transform(test_df[features])

    return train_split, val_split, test_df

def main():
    print("Loading data...")
    train, test = load_data()

    print("Creating RUL...")
    train = add_rul(train)

    print("Dropping low-variance sensors...")
    train, test = drop_low_variance_sensors(train, test)

    print("Splitting and scaling...")
    train_df, val_df, test_df = split_and_scale(train, test)

    train_df.to_csv(processed_data / "train.csv", index=False)
    val_df.to_csv(processed_data / "val.csv", index=False)
    test_df.to_csv(processed_data / "test.csv", index=False)

    print("Preprocessing complete.")
    print(f"Train shape: {train_df.shape}")
    print(f"Val shape: {val_df.shape}")
    print(f"Test shape: {test_df.shape}")

if __name__ == "__main__":
    main()