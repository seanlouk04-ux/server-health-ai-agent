import os
import glob
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)

TRAINING_DIR = os.path.join(PROJECT_ROOT, "training_set")
MODEL_SAVE_PATH = os.path.join(PROJECT_ROOT, "models", "edge_model.pkl")

def retrain_model():
    print("📈 [TRAINER] Gathering accumulated telemetry from 'training_set/'...")
    csv_files = glob.glob(os.path.join(TRAINING_DIR, "*.csv"))
    
    if not csv_files:
        print("⚠️ [TRAINER] No telemetry files found in training_set/. Aborting.")
        return

    full_df = pd.concat([pd.read_csv(f) for f in csv_files], ignore_index=True)
    print(f"📊 [TRAINER] Total dataset size: {len(full_df)} records across {len(csv_files)} batches.")

    X = full_df[['feature_1', 'feature_2']]
    y = full_df['target']

    # =========================================================================
    # GUARDRAIL: Skip training if there's only 1 unique class (all 0s or all 1s)
    # =========================================================================
    unique_classes = y.unique()
    if len(unique_classes) <= 1:
        print(f"⚠️ [TRAINER] Data contains only class {unique_classes[0]}. Skipping fit until both classes (0 and 1) exist.")
        return

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    if len(y_train.unique()) <= 1:
        print("⚠️ [TRAINER] Train split ended up with only a single class. Skipping fit.")
        return

    model = LogisticRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    print(f"🎯 [MODEL METRICS] Test Set Evaluation:")
    print(f"   - Accuracy:  {acc:.2%}")
    print(f"   - Precision: {prec:.2%}")
    print(f"   - Recall:    {rec:.2%}")
    print(f"   - F1-Score:  {f1:.2f}")

    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    joblib.dump(model, MODEL_SAVE_PATH)
    print(f"✅ [TRAINER] Model successfully saved to '{MODEL_SAVE_PATH}'")

if __name__ == "__main__":
    retrain_model()