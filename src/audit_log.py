import json
import os
from datetime import datetime

AUDIT_LOG_PATH = "models/audit_log.jsonl"

def log_audit_event(
    file_name: str,
    cpu_mean: float,
    ram_mean: float,
    cpu_drift_pct: float,
    ram_drift_pct: float,
    naive_decision: str,
    agent_decision: str,
    ground_truth: str,
    mode: str,
    profile_name: str = "default"
):
    os.makedirs("models", exist_ok=True)
    
    event = {
        "timestamp": datetime.utcnow().isoformat(),
        "file_name": file_name,
        "hardware_profile": profile_name,
        "cpu_mean": round(cpu_mean, 2),
        "ram_mean": round(ram_mean, 2),
        "cpu_drift_pct": round(cpu_drift_pct, 2),
        "ram_drift_pct": round(ram_drift_pct, 2),
        "naive_rule_decision": naive_decision,
        "agent_decision": agent_decision,
        "ground_truth": ground_truth,  # "STABLE" or "SPIKE"
        "mode": mode,                  # "WARM_UP", "ACTIVE_LLM", "FALLBACK_RULE"
        "divergence": naive_decision != agent_decision
    }
    
    with open(AUDIT_LOG_PATH, "a") as f:
        f.write(json.dumps(event) + "\n")