"""
Sequential Bayesian updating of drift parameter μ via Gaussian conjugate model.

Prior:       μ ~ N(μ₀, τ²)          where μ₀ = μ̂_MLE, τ² = σ̂²/n_train
Likelihood:  Δ_new | μ ~ N(μ, σ²)   (σ² fixed at MLE estimate)

Posterior after m new observations with sample mean Δ̄_new:
  μ_post  = (μ₀/τ² + m·Δ̄_new/σ²) / (1/τ² + m/σ²)
  σ²_post = 1 / (1/τ² + m/σ²)

Each month we replay a held-out 2022–2024 observation, update the posterior,
and re-run Monte Carlo with μ_post as the drift to get an updated P(fail).
"""
import numpy as np
import pandas as pd
import json
import os

N_SIMS = 5_000   # faster for sequential updates; still SE < 0.007


def monte_carlo_pfail(x_current, mu, sigma, x_fail, k_months=24, n_sims=N_SIMS, seed=0):
    rng = np.random.default_rng(seed)
    steps = rng.normal(mu, sigma, size=(n_sims, k_months))
    trajectories = x_current + np.cumsum(steps, axis=1)
    return float(np.any(trajectories >= x_fail, axis=1).mean())


def bayesian_update_sequence(df, mle_params, x_fail, k_months=24):
    mu0 = mle_params["mu_hat"]
    sigma = mle_params["sigma_hat"]
    n_train = mle_params["n_train"]

    # Prior: τ² = standard error of the MLE estimate squared
    tau2 = (sigma ** 2) / n_train
    sigma2 = sigma ** 2
    precision_prior = 1.0 / tau2

    test_rows = df[df["split"] == "test"].dropna(subset=["delta"]).reset_index(drop=True)
    if test_rows.empty:
        print("  No test rows found.")
        return []

    updates = []
    observed_deltas = []

    for i, row in test_rows.iterrows():
        observed_deltas.append(row["delta"])
        m = len(observed_deltas)
        delta_bar = float(np.mean(observed_deltas))
        precision_likelihood = m / sigma2

        # Gaussian conjugate posterior
        mu_post = (mu0 / tau2 + m * delta_bar / sigma2) / (precision_prior + precision_likelihood)
        sigma2_post = 1.0 / (precision_prior + precision_likelihood)

        x_current = float(row["depth_ft"])
        p_fail = monte_carlo_pfail(x_current, mu_post, sigma, x_fail, k_months, seed=i)

        updates.append({
            "date": str(row["date"])[:10],
            "month_idx": int(i + 1),
            "delta_observed": float(row["delta"]),
            "mu_post": float(mu_post),
            "sigma_post": float(np.sqrt(sigma2_post)),
            "p_fail": p_fail,
            "x_current": x_current,
        })

        print(f"  month {i+1:3d} ({updates[-1]['date'][:7]}): "
              f"μ_post={mu_post:+.4f}  σ_post={np.sqrt(sigma2_post):.4f}  "
              f"P(fail)={p_fail:.3f}")

    return updates


def main():
    for req in ["data/mle_params.json", "data/monte_carlo_results.json"]:
        if not os.path.exists(req):
            print(f"Missing {req}. Run prior pipeline steps first.")
            return {}

    with open("data/mle_params.json") as f:
        mle_params = json.load(f)
    with open("data/monte_carlo_results.json") as f:
        mc_results = json.load(f)

    all_updates = {}
    for name in mle_params:
        clean_path = f"data/{name}_clean.csv"
        if not os.path.exists(clean_path):
            continue

        print(f"\n{name} — sequential Bayesian updates:")
        df = pd.read_csv(clean_path, parse_dates=["date"])
        df["date"] = df["date"].astype(str)

        x_fail = mc_results[name]["x_fail"]
        updates = bayesian_update_sequence(df, mle_params[name], x_fail)
        all_updates[name] = updates
        print(f"  → {len(updates)} update steps")

    with open("data/bayesian_updates.json", "w") as f:
        json.dump(all_updates, f, indent=2)
    print("\nSaved → data/bayesian_updates.json")
    return all_updates


if __name__ == "__main__":
    main()
