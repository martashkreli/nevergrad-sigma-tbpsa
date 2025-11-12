"""examples/noisy_tbpsa_baseline.py

Baseline TBPSA-style experiment runner (sigma-unaware).

This script provides a simple baseline optimizer that DOES NOT use the
per-location sigma(x) information. It's intended for the TBPSA developer
to test behaviour when noise varies with distance. The sigma-aware
developer can later implement a variant that uses sigma(x) to weight or
resample evaluations.

What this baseline does (simple, clear behavior):
 - Maintain a current best solution x_best.
 - At each iteration, sample `n_candidates` points from N(x_best, step_scale^2 I).
 - Evaluate each candidate once (no resampling, noise-aware info ignored for selection).
 - Accept the candidate with lowest observed noisy value if it improves best noisy value.
 - Record per-iteration: best noisy value, best true value (deterministic), sigma(x_best).

Outputs:
 - CSV file with history (iteration, evals, x_best, noisy_best, true_best, sigma)
 - Two simple plots: best value vs iteration, sigma profile along best norm.

Usage:
 python examples/noisy_tbpsa_baseline.py --dim 5 --budget 200 --seed 0

This file intentionally keeps the baseline simple; it's easy to adapt to
other acquisition/evaluation policies to compare against a sigma-aware
implementation.
"""

from __future__ import annotations
import argparse
import csv
import os
from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt

# import the example problem (load by path so the examples/ folder need not be a package)
import importlib.util
from pathlib import Path

_this_dir = Path(__file__).resolve().parent
_mod_path = _this_dir / "noisy_function.py"
spec = importlib.util.spec_from_file_location("examples.noisy_function", str(_mod_path))
noisy_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(noisy_mod)  # type: ignore[attr-defined]
HeteroskedasticQuadratic = noisy_mod.HeteroskedasticQuadratic


def propose_candidates(x_center: np.ndarray, n: int, step: float, rng: np.random.Generator) -> np.ndarray:
    """Sample n candidates from a multivariate normal around x_center."""
    dim = x_center.size
    return rng.normal(loc=x_center, scale=step, size=(n, dim))


def run_baseline(
    dim: int = 5,
    budget: int = 200,
    n_candidates: int = 10,
    init_step: float = 1.0,
    step_decay: float = 0.995,
    seed: int = 0,
    out_dir: str | None = None,
):
    rng = np.random.default_rng(seed)
    prob = HeteroskedasticQuadratic(dim=dim, seed=seed)

    # initialize at random
    x_best = rng.normal(0.0, 1.0, size=(dim,))
    y_best_noisy, _ = prob.noisy_f(x_best)
    y_best_true, sigma_best = prob.noisy_f(x_best, deterministic=True)

    history = []

    step = init_step
    evals = 0

    while evals < budget:
        # propose candidates (we'll produce at most n_candidates evaluations this iteration)
        cand = propose_candidates(x_best, n_candidates, step, rng)
        noisy_vals = []
        sigmas = []
        trues = []
        for c in cand:
            y_noisy, sig = prob.noisy_f(c, deterministic=False)
            y_true, _ = prob.noisy_f(c, deterministic=True)
            noisy_vals.append(y_noisy)
            sigmas.append(sig)
            trues.append(y_true)
            evals += 1
            if evals >= budget:
                break

        # pick best by noisy observation (baseline: unaware of sigma)
        noisy_vals = np.array(noisy_vals)
        trues = np.array(trues)
        sigmas = np.array(sigmas)
        best_idx = int(np.argmin(noisy_vals))
        if noisy_vals[best_idx] < y_best_noisy:
            x_best = cand[best_idx]
            y_best_noisy = float(noisy_vals[best_idx])
            y_best_true = float(trues[best_idx])
            sigma_best = float(sigmas[best_idx])

        # record snapshot
        history.append(
            {
                "evals": evals,
                "x_best": x_best.copy(),
                "noisy_best": float(y_best_noisy),
                "true_best": float(y_best_true),
                "sigma_best": float(sigma_best),
                "step": float(step),
            }
        )

        # decay step
        step *= step_decay

    # save results
    if out_dir is None:
        out_dir = os.path.join(os.getcwd(), "tbpsa_baseline_out")
    os.makedirs(out_dir, exist_ok=True)

    csv_path = os.path.join(out_dir, "history.csv")
    with open(csv_path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["iter", "evals", "noisy_best", "true_best", "sigma_best", "step", "x_best"])
        for i, rec in enumerate(history, start=1):
            writer.writerow(
                [
                    i,
                    rec["evals"],
                    rec["noisy_best"],
                    rec["true_best"],
                    rec["sigma_best"],
                    rec["step"],
                    "|".join([f"{v:.6g}" for v in rec["x_best"]]),
                ]
            )

    # simple plots
    iters = np.arange(1, len(history) + 1)
    noisy_best_vals = np.array([h["noisy_best"] for h in history])
    true_best_vals = np.array([h["true_best"] for h in history])
    sigma_vals = np.array([h["sigma_best"] for h in history])

    plt.figure(figsize=(8, 4))
    plt.plot(iters, noisy_best_vals, label="best noisy (observed)")
    plt.plot(iters, true_best_vals, label="best true (deterministic)")
    plt.xlabel("iteration")
    plt.ylabel("objective")
    plt.title("Baseline TBPSA: best objective over iterations (sigma-unaware)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "best_values.png"))

    plt.figure(figsize=(8, 3))
    plt.plot(iters, sigma_vals)
    plt.xlabel("iteration")
    plt.ylabel("sigma at x_best")
    plt.title("sigma(x_best) over iterations")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "sigma_over_iters.png"))

    # a quick scatter of final best location coordinates (norm vs sigma)
    norms = [float(np.linalg.norm(h["x_best"])) for h in history]
    plt.figure(figsize=(6, 4))
    plt.scatter(norms, sigma_vals, c=iters, cmap="viridis", s=20)
    plt.xlabel("||x_best||")
    plt.ylabel("sigma(x_best)")
    plt.title("sigma vs norm of best solution (iterations colored)")
    plt.colorbar(label="iteration")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "sigma_vs_norm.png"))

    print(f"Saved history to: {csv_path}")
    print(f"Saved plots to: {out_dir}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Baseline TBPSA runner (sigma-unaware)")
    p.add_argument("--dim", type=int, default=5)
    p.add_argument("--budget", type=int, default=200)
    p.add_argument("--n_candidates", type=int, default=10)
    p.add_argument("--init_step", type=float, default=1.0)
    p.add_argument("--step_decay", type=float, default=0.995)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out_dir", type=str, default=None)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_baseline(
        dim=args.dim,
        budget=args.budget,
        n_candidates=args.n_candidates,
        init_step=args.init_step,
        step_decay=args.step_decay,
        seed=args.seed,
        out_dir=args.out_dir,
    )
