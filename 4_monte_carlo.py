"""
Monte Carlo estimation of P(fail within k months).

Target:  P(∃ t ≤ k : X_t ≥ X_fail | X_0 = x_current)

Estimator (N = 10,000 simulations):
  X_{t+1}^(i) = X_t^(i) + μ̂ + ε_t,   ε_t ~ N(0, σ̂²)
  P̂ = (1/N) Σ 1[trajectory i hits X_fail before month k]

By the Law of Large Numbers: P̂ → P(fail within k) as N → ∞.
SE = sqrt(p̂(1-p̂)/N) < 0.005 for all p̂ with N = 10,000.

X_fail is set to (observed training maximum) × 1.10 — a 10% buffer
approximating the physical well casing / pump intake depth.
"""
import numpy as np
import pandas as pd
import json
import glob
import os

N_SIMS = 10_000
MAX_MONTHS = 60


def get_x_fail(df_train):
    return float(df_train["depth_ft"].max() * 1.10)


def failure_curve(x0, mu, sigma, x_fail, max_months=MAX_MONTHS, n_sims=N_SIMS, seed=42):
    """Return list of P(fail within k) for k = 1 … max_months."""
    rng = np.random.default_rng(seed)
    steps = rng.normal(mu, sigma, size=(n_sims, max_months))
    trajectories = x0 + np.cumsum(steps, axis=1)  # shape (n_sims, max_months)

    probs = []
    for k in range(1, max_months + 1):
        failed = np.any(trajectories[:, :k] >= x_fail, axis=1)
        probs.append(float(failed.mean()))
    return probs, trajectories


def main():
    if not os.path.exists("data/mle_params.json"):
        print("MLE parameters not found. Run 3_mle.py first.")
        return {}

    with open("data/mle_params.json") as f:
        mle_params = json.load(f)

    results = {}
    for name, params in mle_params.items():
        clean_path = f"data/{name}_clean.csv"
        if not os.path.exists(clean_path):
            print(f"{name}: clean CSV not found, skipping")
            continue

        df = pd.read_csv(clean_path, parse_dates=["date"])
        df_train = df[df["split"] == "train"]

        x0 = params["current_depth"]
        mu = params["mu_hat"]
        sigma = params["sigma_hat"]
        x_fail = get_x_fail(df_train)

        curve, _ = failure_curve(x0, mu, sigma, x_fail)

        p12 = curve[11]  # index 11 = k=12
        p60 = curve[59]  # index 59 = k=60

        print(f"{name}:")
        print(f"  x_fail = {x_fail:.1f} ft  (training max × 1.10)")
        print(f"  P(fail ≤ 12 mo) = {p12:.3f}")
        print(f"  P(fail ≤ 60 mo) = {p60:.3f}")

        results[name] = {
            "x_fail": x_fail,
            "p_fail_12": p12,
            "p_fail_60": p60,
            "failure_curve": curve,
        }

    with open("data/monte_carlo_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved Monte Carlo results → data/monte_carlo_results.json")
    return results


if __name__ == "__main__":
    main()
