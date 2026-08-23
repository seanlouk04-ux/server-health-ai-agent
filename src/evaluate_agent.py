import json
import os
import pandas as pd

AUDIT_LOG_PATH = "models/audit_log.jsonl"

def evaluate_performance():
    if not os.path.exists(AUDIT_LOG_PATH):
        print("⚠️ No audit log found at 'models/audit_log.jsonl'. Run some files through main.py first!")
        return

    records = []
    with open(AUDIT_LOG_PATH, "r") as f:
        for line in f:
            records.append(json.loads(line.strip()))

    df = pd.DataFrame(records)
    active_df = df[df["mode"] == "ACTIVE_LLM"]

    if len(active_df) == 0:
        print("⏳ No ACTIVE_LLM audit decisions logged yet (still in warm-up or fallback).")
        return

    # Map decisions to binary
    active_df["agent_pred"] = active_df["agent_decision"].apply(lambda x: 1 if "QUARANTINE" in x else 0)
    active_df["naive_pred"] = active_df["naive_rule_decision"].apply(lambda x: 1 if "QUARANTINE" in x else 0)
    active_df["truth_label"] = active_df["ground_truth"].apply(lambda x: 1 if x == "SPIKE" else 0)

    # Calculate agreement metrics
    llm_accuracy = (active_df["agent_pred"] == active_df["truth_label"]).mean()
    naive_accuracy = (active_df["naive_pred"] == active_df["truth_label"]).mean()
    divergence_rate = active_df["divergence"].mean()

    print("\n==================================================")
    print("      EDGE MLOPS DRIFT POLICY EVALUATION          ")
    print("==================================================")
    print(f"Total Audits Evaluated:      {len(active_df)}")
    print(f"LLM Policy Accuracy vs Truth: {llm_accuracy:.2%}")
    print(f"Naive Threshold Accuracy:     {naive_accuracy:.2%}")
    print(f"Divergence Rate (LLM vs Rule): {divergence_rate:.2%}")
    print("==================================================\n")

if __name__ == "__main__":
    evaluate_performance()