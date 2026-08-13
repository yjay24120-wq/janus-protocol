import os
import glob
import json
import pandas as pd
import subprocess

def get_latest_telemetry_file():
    """Scans the working directory for the most recent telemetry export CSV based on modification time."""
    csv_files = glob.glob("*_export.csv") + glob.glob("*export*.csv")
    csv_files = list(set(csv_files))  # Remove duplicates
    if not csv_files:
        raise FileNotFoundError("No telemetry export CSV files found in the directory.")
    latest_file = max(csv_files, key=os.path.getmtime)
    return latest_file

def parse_optimal_dna(csv_path):
    """Parses the telemetry CSV and extracts the elite DNA vector based on Sharpe Ratio."""
    df = pd.read_csv(csv_path)
    
    # Locate the strategy with the highest Sharpe ratio
    best_row = df.loc[df['sharpe_ratio'].idxmax()]
    
    strategy_id = int(best_row['id'])
    sharpe_ratio = float(best_row['sharpe_ratio'])
    max_drawdown = float(best_row['max_drawdown'])
    
    # Parse DNA vector string representation into list
    dna_vector = eval(best_row['dna_vector'])
    
    # Extract parameters
    optimal_params = {
        "source_file": os.path.basename(csv_path),
        "strategy_id": strategy_id,
        "sharpe_ratio": round(sharpe_ratio, 4),
        "max_drawdown_pct": round(max_drawdown, 4),
        "lookback_window": round(float(dna_vector[1]), 2),
        "breakout_threshold": round(float(dna_vector[2]), 5),
        "channel_weight": round(float(dna_vector[3]), 4)
    }
    return optimal_params

def write_slider_config(params, config_filename="slider_config.json"):
    """Saves the extracted parameters to a JSON file for the dashboard."""
    with open(config_filename, "w") as f:
        json.dump(params, f, indent=4)
    print(f"[+] Successfully exported optimal parameters to {config_filename}:")
    print(json.dumps(params, indent=4))

def git_commit_and_push(config_filename="slider_config.json"):
    """Automatically stages, commits, and pushes the updated config to GitHub."""
    try:
        print("[*] Staging configuration file in Git...")
        subprocess.run(["git", "add", config_filename], check=True)
        
        print("[*] Committing changes...")
        commit_msg = f"Auto-update slider config to Strategy ID parameters (Sharpe Elite)"
        subprocess.run(["git", "commit", "-m", commit_msg], check=True)
        
        print("[*] Pushing to GitHub (origin main)...")
        subprocess.run(["git", "push", "origin", "main"], check=True)
        
        print("[+] Successfully pushed updates to GitHub!")
    except subprocess.CalledProcessError as e:
        print(f"[!] Git operation failed: {e}")

if __name__ == "__main__":
    try:
        latest_csv = get_latest_telemetry_file()
        print(f"[*] Analyzing telemetry export: {latest_csv}")
        
        params = parse_optimal_dna(latest_csv)
        write_slider_config(params)
        git_commit_and_push()
        
    except Exception as e:
        print(f"[!] Error during execution: {e}")
