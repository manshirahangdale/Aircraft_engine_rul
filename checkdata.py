import numpy as np

features = np.load("data/processed/feature_cols.npy", allow_pickle=True)
print("Number of features:", len(features))
print(features)
