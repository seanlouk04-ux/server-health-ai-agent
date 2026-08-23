import os
import pandas as pd
import numpy as np

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

def generate_broken_data():
    print("🧪 [TEST SUITE] Generating intentionally corrupted data files...")

    # Case 1: Corrupted Schema (Missing the 'target' column entirely)
    broken_schema_data = {
        "feature_1": [12.4, 9.8, 11.1],
        "feature_2": [44.1, 21.3, 33.5]
        # 'target' column is missing!
    }
    df1 = pd.DataFrame(broken_schema_data)
    path1 = os.path.join(DATA_DIR, "corrupted_schema_test.csv")
    df1.to_csv(path1, index=False)
    print(f"⚠️ Dropped bad schema file to: {path1}")

    # Case 2: Missing Data Values (Contains Nulls/NaNs)
    null_value_data = {
        "feature_1": [10.2, np.nan, 11.5], # Row 2 is missing a value!
        "feature_2": [14.2, 22.8, np.nan], # Row 3 is missing a value!
        "target": [0, 1, 0]
    }
    df2 = pd.DataFrame(null_value_data)
    path2 = os.path.join(DATA_DIR, "null_values_test.csv")
    df2.to_csv(path2, index=False)
    print(f"⚠️ Dropped null value file to: {path2}")

if __name__ == "__main__":
    generate_broken_data()