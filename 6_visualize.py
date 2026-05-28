"""
Generate all static figures for the writeup.
Saves PNGs to data/figures/.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json
import os
import glob

os.makedirs("data/figures", exist_ok=True)

N_DISPLAY = 200   # trajectories shown in fan plots
N_SIMS    = 10_000


def rng_trajectories(x0, mu, sigma, k_months, n_sims, seed=0):
    rng = np.random.default_rng(seed)
    steps = rng.normal(mu, sigma, size=(n_sims, k_months))
    return x0 + np.cumsum(steps, axis=1)


# ── Figure 1: Water-level history ───────────────────────────────────────────

def fig_history(name, df):
    train = df[df["split"] == "train"]
    test  = df[df["split"] == "test"]

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(train["date"], train["depth_ft"], lw=1.5, color="steelblue", label="Training (2000–2021)")
    ax.plot(test["date"],  test["depth_ft"],  lw=1.5, color="darkorange", label="Held-out replay (2022+)")
    ax.axvline(pd.Timestamp("2022-01-01"), color="gray", ls="--", lw=1)
    ax.set_xlabel("Date")
    ax.set_ylabel("Depth to water (ft below surface)")
    ax.set_title(f"{name} — Historical groundwater depth (USGS)")
    ax.legend()
    plt.tight_layout()
    path = f"data/figures/{name}_history.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"  Saved {path}")


# ── Figure 2: Monte Carlo trajectory fan ────────────────────────────────────

def fig_trajectories(name, df, mle_params, mc_results, k_months=60):
    mu     = mle_params["mu_hat"]
    sigma  = mle_params["sigma_hat"]
    x_fail = mc_results["x_fail"]
    x0     = df["depth_ft"].iloc[-1]

    trajs = rng_trajectories(x0, mu, sigma, k_months, N_SIMS, seed=1)
    idx   = np.random.default_rng(2).choice(N_SIMS, N_DISPLAY, replace=False)

    fig, ax = plt.subplots(figsize=(10, 5))
    months = np.arange(1, k_months + 1)

    for i in idx:
        failed = trajs[i].max() >= x_fail
        ax.plot(months, trajs[i], alpha=0.09, lw=0.5, color="red" if failed else "steelblue")

    ax.axhline(x_fail, color="red", lw=2, ls="--", label=f"Failure depth ({x_fail:.0f} ft)")
    ax.axhline(x0,     color="black", lw=1, ls=":", label=f"Current ({x0:.0f} ft)")

    p60 = mc_results["p_fail_60"]
    ax.set_xlabel("Months from now")
    ax.set_ylabel("Depth to water (ft)")
    ax.set_title(f"{name} — Monte Carlo trajectories (N={N_SIMS:,})\n"
                 f"P(fail within 60 months) = {p60:.1%}")
    ax.legend()
    plt.tight_layout()
    path = f"data/figures/{name}_trajectories.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"  Saved {path}")


# ── Figure 3: Failure-probability curve P(fail ≤ k) ─────────────────────────

def fig_failure_curve(name, mc_results):
    curve  = mc_results["failure_curve"]
    months = np.arange(1, len(curve) + 1)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(months, curve, color="steelblue", lw=2)
    ax.fill_between(months, curve, alpha=0.15, color="steelblue")
    ax.axhline(0.5, color="orange", ls="--", lw=1.2, label="50% threshold")
    ax.set_xlabel("Months from now (k)")
    ax.set_ylabel("P(fail within k months)")
    ax.set_ylim(0, 1)
    ax.set_title(f"{name} — Cumulative failure probability curve")
    ax.legend()
    plt.tight_layout()
    path = f"data/figures/{name}_failure_curve.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"  Saved {path}")


# ── Figure 4: Bayesian update time series ───────────────────────────────────

def fig_bayesian_update(name, updates, mu_hat):
    if not updates:
        print(f"  No Bayesian updates for {name}, skipping")
        return

    dates      = [u["date"][:7] for u in updates]
    mu_posts   = [u["mu_post"]  for u in updates]
    sigmas_p   = [u["sigma_post"] for u in updates]
    p_fails    = [u["p_fail"]   for u in updates]
    x = np.arange(len(dates))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    # Posterior drift μ_post with ±2σ band
    ax1.plot(x, mu_posts, color="steelblue", lw=2, label="μ_post")
    ax1.fill_between(x,
                     [m - 2*s for m, s in zip(mu_posts, sigmas_p)],
                     [m + 2*s for m, s in zip(mu_posts, sigmas_p)],
                     alpha=0.2, color="steelblue", label="±2σ_post")
    ax1.axhline(mu_hat, color="gray", ls="--", lw=1, label=f"μ̂ MLE = {mu_hat:.4f}")
    ax1.axhline(0, color="lightgray", ls=":", lw=1)
    ax1.set_ylabel("μ_post (ft/month)")
    ax1.set_title(f"{name} — Sequential Bayesian updating (2022–2024 replay)")
    ax1.legend(fontsize=8)

    # P(fail within 24 months) over time
    ax2.plot(x, p_fails, color="crimson", lw=2, label="P(fail ≤ 24 months)")
    ax2.fill_between(x, p_fails, alpha=0.15, color="crimson")
    ax2.set_ylim(0, 1)
    ax2.set_ylabel("P(fail)")
    ax2.set_xlabel("Months of new data observed")
    ax2.legend(fontsize=8)

    step = max(1, len(dates) // 8)
    ax2.set_xticks(x[::step])
    ax2.set_xticklabels(dates[::step], rotation=45, fontsize=8)

    plt.tight_layout()
    path = f"data/figures/{name}_bayesian_update.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"  Saved {path}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    for req in ["data/mle_params.json", "data/monte_carlo_results.json", "data/bayesian_updates.json"]:
        if not os.path.exists(req):
            print(f"Missing {req}. Run earlier pipeline steps first.")
            return

    with open("data/mle_params.json") as f:
        mle_params = json.load(f)
    with open("data/monte_carlo_results.json") as f:
        mc_results = json.load(f)
    with open("data/bayesian_updates.json") as f:
        bayesian_updates = json.load(f)

    for name in mle_params:
        clean_path = f"data/{name}_clean.csv"
        if not os.path.exists(clean_path):
            continue
        df = pd.read_csv(clean_path, parse_dates=["date"])
        print(f"\n{name}:")
        fig_history(name, df)
        fig_trajectories(name, df, mle_params[name], mc_results[name])
        fig_failure_curve(name, mc_results[name])
        fig_bayesian_update(name, bayesian_updates.get(name, []), mle_params[name]["mu_hat"])

    print("\nAll figures saved to data/figures/")


if __name__ == "__main__":
    main()
