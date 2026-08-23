import os
import json
import sys
import pandas as pd
import httpx
import asyncio
from audit_log import log_audit_event

# ==========================================
# FIX 1: Enforce Absolute Pathing
# ==========================================
# Step up one level from 'src/' (where main.py lives) to the project root
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)

# Anchor all directories to the project root
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
TRAINING_DIR = os.path.join(PROJECT_ROOT, "training_set")
QUARANTINE_DIR = os.path.join(PROJECT_ROOT, "quarantine")
ERROR_DIR = os.path.join(PROJECT_ROOT, "errors")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
HISTORICAL_FILE = os.path.join(MODELS_DIR, "historical_baseline.json")

# Ensure all absolute paths exist before the loop starts
for folder in [DATA_DIR, TRAINING_DIR, QUARANTINE_DIR, ERROR_DIR, MODELS_DIR]:
    os.makedirs(folder, exist_ok=True)

def load_historical_baseline():
    if os.path.exists(HISTORICAL_FILE):
        try:
            with open(HISTORICAL_FILE, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError:
            pass
    return {"cpu_mean": None, "ram_mean": None, "total_files_analyzed": 0}

def update_historical_baseline(new_cpu_mean, new_ram_mean):
    baseline = load_historical_baseline()
    n = baseline["total_files_analyzed"]
    
    if baseline["cpu_mean"] is None or baseline["ram_mean"] is None:
        baseline["cpu_mean"] = round(new_cpu_mean, 1)
        baseline["ram_mean"] = round(new_ram_mean, 1)
    else:
        baseline["cpu_mean"] = round(((baseline["cpu_mean"] * n) + new_cpu_mean) / (n + 1), 1)
        baseline["ram_mean"] = round(((baseline["ram_mean"] * n) + new_ram_mean) / (n + 1), 1)
        
    baseline["total_files_analyzed"] += 1
    
    with open(HISTORICAL_FILE, 'w') as f:
        json.dump(baseline, f, indent=4)
    return baseline

async def query_local_agent(prompt):
    """Queries local Llama 3.1 with timeout and resilience fallback."""
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(
                # Included the Docker environment variable setup here for later
                os.getenv("LLM_URL", "http://localhost:1234/v1/chat/completions"),
                json={
                    "model": "meta-llama-3.1-8b-instruct",
                    "messages": [
                        {
                            "role": "system",
                            "content": (
                                "You are an Edge MLOps Data Drift Guardian.\n"
                                "Rule: Check 'Calculated Drift Statistics (Python verified)'.\n"
                                "If CPU Drift > 25.0% OR RAM Drift > 25.0%, output 'QUARANTINE - Data drift detected'.\n"
                                "Otherwise, output 'PROCEED TO TRAINING'."
                            )
                        },
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0.0
                }
            )
            result = response.json()
            return result['choices'][0]['message']['content'].strip(), "ACTIVE_LLM"
    except (httpx.ConnectError, httpx.TimeoutException) as e:
        print("⚠️ [RESILIENCE] Local LLM server offline/unreachable. Falling back to Python threshold rule.")
        return None, "FALLBACK_RULE"

async def process_incoming_file(filepath):
    filename = os.path.basename(filepath)
    print(f"\n📥 [ORCHESTRATOR] Auditing file: {filename}...")
    
    try:
        df = pd.read_csv(filepath)
        
        # ==========================================
        # FIX 2: Strict Schema & Null Validation
        # ==========================================
        required_cols = {'feature_1', 'feature_2', 'target'}
        if not required_cols.issubset(df.columns) or df[list(required_cols)].isnull().values.any():
            print(f"⚠️ [VALIDATION FAILED] Schema error or nulls detected in {filename}. Routing to errors.")
            dest_path = os.path.join(ERROR_DIR, filename)
            os.rename(filepath, dest_path)
            return

        ground_truth = "SPIKE" if "SPIKE" in filename else "STABLE"
        profile_name = filename.split("_")[1] if "telemetry_" in filename and len(filename.split("_")) > 2 else "default"

        cpu_mean = float(df['feature_1'].mean())
        ram_mean = float(df['feature_2'].mean())
        
        history = load_historical_baseline()

        cpu_drift_pct = 0.0
        ram_drift_pct = 0.0
        if history['cpu_mean'] is not None and history['cpu_mean'] > 0:
            cpu_drift_pct = abs(cpu_mean - history['cpu_mean']) / history['cpu_mean'] * 100
        if history['ram_mean'] is not None and history['ram_mean'] > 0:
            ram_drift_pct = abs(ram_mean - history['ram_mean']) / history['ram_mean'] * 100

        naive_decision = "QUARANTINE" if (cpu_drift_pct > 25.0 or ram_drift_pct > 25.0) else "PROCEED TO TRAINING"

        if history['total_files_analyzed'] < 3:
            agent_decision = "PROCEED TO TRAINING"
            mode = "WARM_UP"
            print(f"🤖 [LOCAL AGENT] Warm-Up Mode: Learning baseline ({history['total_files_analyzed'] + 1}/3)...")
        else:
            prompt = (
                f"Calculated Drift Statistics (Python verified):\n"
                f"- CPU Drift: {cpu_drift_pct:.1f}%\n"
                f"- RAM Drift: {ram_drift_pct:.1f}%\n"
            )
            llm_result, mode = await query_local_agent(prompt)
            
            if mode == "FALLBACK_RULE":
                agent_decision = naive_decision
            else:
                agent_decision = "PROCEED TO TRAINING" if "PROCEED TO TRAINING" in llm_result else "QUARANTINE"

            print(f"🤖 [LOCAL AGENT] Audit Decision [{mode}]: {agent_decision}")
            if agent_decision != naive_decision:
                print(f"⚡ [DIVERGENCE DETECTED] LLM Decision ({agent_decision}) != Naive Rule ({naive_decision})")

        log_audit_event(
            file_name=filename,
            cpu_mean=cpu_mean,
            ram_mean=ram_mean,
            cpu_drift_pct=cpu_drift_pct,
            ram_drift_pct=ram_drift_pct,
            naive_decision=naive_decision,
            agent_decision=agent_decision,
            ground_truth=ground_truth,
            mode=mode,
            profile_name=profile_name
        )

        if "PROCEED TO TRAINING" in agent_decision:
            updated_history = update_historical_baseline(cpu_mean, ram_mean)
            num_accumulated = updated_history["total_files_analyzed"]
            
            dest_path = os.path.join(TRAINING_DIR, filename)
            os.rename(filepath, dest_path)
            print(f"✅ Approved! Moved to: '{dest_path}'")

            if num_accumulated >= 3:
                # ==========================================
                # FIX 3: Subprocess Working Directory Match
                # ==========================================
                train_script_path = os.path.join(BASE_DIR, "train.py")
                print(f"📈 [MLOPS] Triggering retrainer: {train_script_path}")
                
                process = await asyncio.create_subprocess_exec(
                    sys.executable, train_script_path,
                    cwd=PROJECT_ROOT 
                )
                await process.wait()
        else:
            dest_path = os.path.join(QUARANTINE_DIR, filename)
            os.rename(filepath, dest_path)
            print(f"🚨 Anomalous Telemetry Quarantined: '{dest_path}'")

    except Exception as e:
        print(f"❌ Error auditing file {filename}: {e}")
        dest_path = os.path.join(ERROR_DIR, filename)
        if os.path.exists(filepath):
            os.rename(filepath, dest_path)

async def main_loop():
    print(f"🚀 [STARTUP] Running Edge Orchestration Loop. Watching '{DATA_DIR}' directory...")
    while True:
        csv_files = [os.path.join(DATA_DIR, f) for f in os.listdir(DATA_DIR) if f.endswith('.csv')]
        for file in csv_files:
            await process_incoming_file(file)
            await asyncio.sleep(0.5)
        await asyncio.sleep(2)

if __name__ == "__main__":
    asyncio.run(main_loop())