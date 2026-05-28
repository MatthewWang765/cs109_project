"""
MLE fitting for the Gaussian drift model.

Model:  Δt = X_t - X_{t-1}  ~  N(μ, σ²)

MLE:
  μ̂  = (1/n) Σ Δt          (sample mean)
  σ̂² = (1/n) Σ (Δt - μ̂)²  (biased MLE variance)
"""
import numpy as np
import pandas as pd
import glob
import json
import os


def fit_mle(df):
    train = df[df["split"] == "train"].dropna(subset=["delta"])
    deltas = train["delta"].values
    n = len(deltas)
    if n < 2:
        return None

    mu_hat = float(np.mean(deltas))
    sigma_hat = float(np.sqrt(np.mean((deltas - mu_hat) ** 2)))  # biased MLE

    return {
        "mu_hat": mu_hat,
        "sigma_hat": sigma_hat,
        "n_train": int(n),
        "current_depth": float(df["depth_ft"].iloc[-1]),
        "max_depth_train": float(train["depth_ft"].max()),
    }


def main():
    clean_files = sorted(glob.glob("data/*_clean.csv"))
    if not clean_files:
        print("No clean data found. Run 2_clean_data.py first.")
        return {}

    results = {}
    for path in clean_files:
        name = os.path.basename(path).replace("_clean.csv", "")
        df = pd.read_csv(path, parse_dates=["date"])
        params = fit_mle(df)
        if params is None:
            print(f"{name}: insufficient data for MLE, skipping")
            continue

        mu, sigma = params["mu_hat"], params["sigma_hat"]
        direction = "depleting" if mu > 0.005 else "recharging" if mu < -0.005 else "stable"
        print(f"{name}:")
        print(f"  μ̂  = {mu:+.4f} ft/month  ({direction})")
        print(f"  σ̂  = {sigma:.4f} ft/month")
        print(f"  n  = {params['n_train']} training months")
        print(f"  current depth = {params['current_depth']:.1f} ft")
        results[name] = params

    os.makedirs("data", exist_ok=True)
    with open("data/mle_params.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved MLE parameters for {len(results)} wells → data/mle_params.json")
    return results


if __name__ == "__main__":
    main()
