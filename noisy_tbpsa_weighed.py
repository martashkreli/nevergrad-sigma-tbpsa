import numpy as np
import csv, os
import matplotlib.pyplot as plt
from noisy_function import HeteroskedasticQuadratic


# -----------------------------------------------------------
#  Generate candidates (same as Task 2)
# -----------------------------------------------------------
def propose_candidates(x_center, n, step, rng):
    return rng.normal(loc=x_center, scale=step, size=(n, x_center.size))


# -----------------------------------------------------------
#  Sigma-aware evaluation via RESAMPLING (Step 3 requirement)
# -----------------------------------------------------------
def evaluate_sigma_aware(prob, x):
    """
    Return (mean_y, sigma, k) where
    - k is number of evaluations based on sigma
    - mean_y is the average noisy result
    """
    _, sigma = prob.noisy_f(x, deterministic=True)

    if sigma < 0.1:
        k = 1
    elif sigma < 0.2:
        k = 2
    elif sigma < 0.3:
        k = 4
    else:
        k = 6

    vals = [prob.noisy_f(x)[0] for _ in range(k)]
    return float(np.mean(vals)), sigma, k


# -----------------------------------------------------------
#  TASK 3: TBPSA with sigma-aware evaluation
#  (Structure identical to Task 2)
# -----------------------------------------------------------
def run_sigma_aware(
    dim=5,
    budget=200,           # <-- YES: hard-coded just like Task 2
    n_candidates=10,
    init_step=1.0,
    step_decay=0.995,
    seed=0,
):
    rng = np.random.default_rng(seed)
    prob = HeteroskedasticQuadratic(dim=dim, seed=seed)

    # Initial point
    x_best = rng.normal(0.0, 1.0, size=(dim,))
    y_best, sigma_best, _ = evaluate_sigma_aware(prob, x_best)
    true_best, _ = prob.noisy_f(x_best, deterministic=True)

    history = []
    evals = 0
    step = init_step

    while evals < budget:
        cand = propose_candidates(x_best, n_candidates, step, rng)

        for c in cand:
            y_mean, sigma, k = evaluate_sigma_aware(prob, c)
            y_true, _ = prob.noisy_f(c, deterministic=True)

            evals += k
            if evals >= budget:
                break

            # SAME DECISION RULE AS TASK 2
            # (only difference: y_mean comes from resampling)
            if y_mean < y_best:
                x_best = c
                y_best = y_mean
                sigma_best = sigma
                true_best = y_true

        history.append({
            "evals": evals,
            "x_best": x_best.copy(),
            "noisy_best": float(y_best),
            "true_best": float(true_best),
            "sigma_best": float(sigma_best),
            "step": float(step),
        })

        step *= step_decay

    return history


# -----------------------------------------------------------
#  RUN + PLOTS  (same structure as Task 2)
# -----------------------------------------------------------
if __name__ == "__main__":

    history = run_sigma_aware()

    out_dir = "tbpsa_sigma_weighted_out"
    os.makedirs(out_dir, exist_ok=True)

    # Save CSV
    csv_path = os.path.join(out_dir, "history.csv")
    with open(csv_path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            ["iter", "evals", "noisy_best", "true_best",
             "sigma_best", "x_best"]
        )
        for i, rec in enumerate(history, start=1):
            writer.writerow([
                i,
                rec["evals"],
                rec["noisy_best"],
                rec["true_best"],
                rec["sigma_best"],
                "|".join(f"{v:.6g}" for v in rec["x_best"]),
            ])

    # PLOTS (identical to Task 2)
    iters = np.arange(1, len(history) + 1)
    noisy_vals = [h["noisy_best"] for h in history]
    true_vals  = [h["true_best"] for h in history]
    sigma_vals = [h["sigma_best"] for h in history]
    norms      = [np.linalg.norm(h["x_best"]) for h in history]

    # PLOT 1
    plt.figure(figsize=(8,4))
    plt.plot(iters, noisy_vals, label="best noisy")
    plt.plot(iters, true_vals, label="best true")
    plt.xlabel("iteration")
    plt.ylabel("objective")
    plt.title("Sigma-aware TBPSA (resampling): best objective")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "best_values.png"))

    # PLOT 2
    plt.figure(figsize=(8,3))
    plt.plot(iters, sigma_vals)
    plt.xlabel("iteration")
    plt.ylabel("sigma")
    plt.title("sigma(x_best) over iterations")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "sigma_over_iters.png"))

    # PLOT 3
    plt.figure(figsize=(6,4))
    plt.scatter(norms, sigma_vals, c=iters, cmap="viridis", s=20)
    plt.xlabel("||x_best||")
    plt.ylabel("sigma(x_best)")
    plt.title("Sigma-aware TBPSA: sigma vs norm")
    plt.colorbar(label="iteration")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "sigma_vs_norm.png"))

    print(f"Sigma-aware results saved in: {out_dir}")
