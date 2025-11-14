# examples/run_batch_experiments.py
"""
Batch experiment runner for TBPSA variants on the Heteroskedastic Quadratic problem.
- Run both algorithms (baseline and sigma-aware) for multiple seeds.
- Save their CSV histories in separate folders
- Later load these CSVs in notebooks for plotting and comparison 
"""
from __future__ import annotations
import csv
from pathlib import Path

from noisy_tbpsa_baseline import run_baseline
from noisy_tbpsa_weighed import run_sigma_aware

# Saves the sigma-aware history to a CSV file in the specified output directory.
# history: list of dictionaries (one per iteration)
# out_dir: output folder path (string)
def save_history_csv(history, out_dir: str):
    out_dir_path = Path(out_dir)
    out_dir_path.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir_path / "history.csv"

    with open(csv_path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            ["iter", "evals", "noisy_best", "true_best",
             "sigma_best", "step", "x_best"]
        )
        
        # Write each iteration
        for i, rec in enumerate(history, start=1):
            writer.writerow([
                i,
                rec["evals"],
                rec["noisy_best"],
                rec["true_best"],
                rec["sigma_best"],
                rec["step"],
                # Join vector x_best into a clean "v1|v2|...|vd" format
                "|".join(f"{v:.6g}" for v in rec["x_best"]),
            ])
    print(f"Saved {csv_path}")

# Runs baseline and sigma-aware TBPSA for seeds 0 to 4
# Provides 10 experiment folders in total, with each folder containing history.csv (+ baseline plots)
def main():
    dim = 5
    budget = 200
    seeds = range(5)   # 5 runs per algo: 0,1,2,3,4

    # ---- Baseline runs ----
    for seed in seeds:
        out_dir = f"tbpsa_baseline_out_seed{seed}"
        print(f"Running BASELINE, seed={seed}")
        run_baseline(
            dim=dim,
            budget=budget,
            seed=seed,
            out_dir=out_dir,
        )

    # ---- Sigma-aware runs ----
    for seed in seeds:
        out_dir = f"tbpsa_sigma_weighted_out_seed{seed}"
        print(f"Running SIGMA-AWARE, seed={seed}")
        history = run_sigma_aware(
            dim=dim,
            budget=budget,
            seed=seed,
        )
        save_history_csv(history, out_dir)


if __name__ == "__main__":
    main()
