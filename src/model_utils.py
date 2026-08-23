import os
import time
import argparse
import numpy as np
import pandas as pd

# ==========================================
# FIX: Enforce Absolute Pathing
# ==========================================
# Step up one level from 'src/' to the project root
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)

# Anchor the data directory to the project root
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
os.makedirs(DATA_DIR, exist_ok=True)

PROFILES = {
    "workstation": {"cpu_base": 20.0, "ram_base": 30.0, "noise": 3.0},
    "edge_server": {"cpu_base": 45.0, "ram_base": 50.0, "noise": 4.0},
    "iot_node":    {"cpu_base": 75.0, "ram_base": 70.0, "noise": 5.0},
}

def generate_telemetry(profile_name="edge_server", spike=False, num_samples=100):
    cfg = PROFILES.get(profile_name, PROFILES["edge_server"])
    
    if spike:
        # Simulate a 40%+ sudden resource spike (true anomaly)
        cpu = np.clip(np.random.normal(cfg["cpu_base"] + 40, cfg["noise"], num_samples), 0, 100)
        ram = np.clip(np.random.normal(cfg["ram_base"] + 35, cfg["noise"], num_samples), 0, 100)
        ground_truth = "SPIKE"
    else:
        cpu = np.clip(np.random.normal(cfg["cpu_base"], cfg["noise"], num_samples), 0, 100)
        ram = np.clip(np.random.normal(cfg["ram_base"], cfg["noise"], num_samples), 0, 100)
        ground_truth = "STABLE"

    # Target is binary classification (1 = system overload, 0 = healthy)
    target = (cpu * 0.6 + ram * 0.4 > 70).astype(int)

    df = pd.DataFrame({
        "feature_1": np.round(cpu, 2),
        "feature_2": np.round(ram, 2),
        "target": target
    })

    timestamp = int(time.time())
    filename = f"telemetry_{profile_name}_{ground_truth}_{timestamp}.csv"
    filepath = os.path.join(DATA_DIR, filename)
    
    df.to_csv(filepath, index=False)
    print(f"📡 [SIMULATOR] Generated '{filename}' (Profile: {profile_name}, Truth: {ground_truth})")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", type=str, default="edge_server", choices=["workstation", "edge_server", "iot_node"])
    parser.add_argument("--spike", action="store_true", help="Generate an anomalous spike file")
    args = parser.parse_args()
    
    generate_telemetry(profile_name=args.profile, spike=args.spike)