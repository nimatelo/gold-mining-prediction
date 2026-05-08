import os
import subprocess
import sys

STEPS = [
    ("Step 1: Generate Simulated Data",     "src/generate_data.py"),
    ("Step 2: Exploratory Data Analysis",   "src/eda.py"),
    ("Step 3: Preprocess Data",             "src/preprocess.py"),
    ("Step 4a: Train LSTM (Gold Price)",    "src/train_lstm.py"),
    ("Step 4b: Train Random Forest (Ore)",  "src/train_rf.py"),
    ("Step 4c: Train XGBoost (Production)", "src/train_xgboost.py"),
]

def run_step(label, script_path):
    print(f"\n{'='*55}")
    print(f"  {label}")
    print(f"{'='*55}")
    if not os.path.exists(script_path):
        print(f"  Script not found — skipping")
        return
    result = subprocess.run([sys.executable, script_path])
    if result.returncode == 0:
        print(f"  Completed successfully")
    else:
        print(f"  Failed with error code {result.returncode}")

if __name__ == "__main__":
    print("\n" + "="*55)
    print("  GOLD MINING PREDICTION SYSTEM — FULL PIPELINE")
    print("="*55)
    for label, script in STEPS:
        run_step(label, script)
    print("\n  PIPELINE COMPLETE!")
    print("  Launch dashboard: streamlit run app/dashboard.py")