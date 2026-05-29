"""
main.py — Train the Gaussian HMM on SJV data and decode regimes.

Usage:
    python main.py                                      # real NetCDF data (default)
    python main.py --precip data/gpm_sjv_subset.nc --et data/openet_sjv_subset.nc
    python main.py --synthetic                          # synthetic test
    python main.py --data data/sjv_daily.csv            # CSV fallback
"""

import argparse
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from hmm import GaussianHMM
from data_loader import load_csv, load_netcdf, normalize, make_synthetic_data


def plot_log_likelihood(log_likelihoods, path="outputs/convergence.png"):
    plt.figure(figsize=(8, 4))
    plt.plot(log_likelihoods, marker="o", markersize=3)
    plt.xlabel("Baum-Welch Iteration")
    plt.ylabel("Log-likelihood")
    plt.title("Baum-Welch Convergence")
    plt.tight_layout()
    plt.savefig(path)
    print(f"Saved convergence plot to {path}")


def plot_transition_matrix(A, path="outputs/transition_matrix.png"):
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(A, cmap="Blues", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax)
    ax.set_xticks(range(len(A)))
    ax.set_yticks(range(len(A)))
    ax.set_xlabel("To state")
    ax.set_ylabel("From state")
    ax.set_title("Learned Transition Matrix A")
    for i in range(len(A)):
        for j in range(len(A)):
            ax.text(j, i, f"{A[i,j]:.2f}", ha="center", va="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(path)
    print(f"Saved transition matrix to {path}")


def plot_decoded_sequence(states, dates=None, path="outputs/decoded_states.png"):
    plt.figure(figsize=(14, 3))
    x = np.arange(len(states)) if dates is None else dates
    plt.scatter(x, states, c=states, cmap="tab10", s=2, alpha=0.7)
    plt.yticks([0, 1, 2, 3], ["State 0", "State 1", "State 2", "State 3"])
    plt.xlabel("Time")
    plt.title("Decoded Drought Regime Sequence (Viterbi)")
    plt.tight_layout()
    plt.savefig(path)
    print(f"Saved decoded sequence to {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--precip", type=str, default="data/gpm_sjv_subset.nc")
    parser.add_argument("--et",     type=str, default="data/openet_sjv_subset.nc")
    parser.add_argument("--data",   type=str, default=None, help="CSV fallback")
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--K", type=int, default=4)
    args = parser.parse_args()

    os.makedirs("outputs", exist_ok=True)

    # --- Load data ---
    if args.synthetic:
        print("Using synthetic data for testing...")
        X, true_states, true_params = make_synthetic_data()
        dates = None
    elif args.data:
        print(f"Loading CSV from {args.data}...")
        X, dates = load_csv(args.data)
    else:
        print(f"Loading NetCDF: precip={args.precip}, ET={args.et}")
        X, dates = load_netcdf(args.precip, args.et)

    print(f"Data shape: {X.shape}  (T={X.shape[0]} days, D={X.shape[1]} features)")
    print(f"Precip range: {X[:,0].min():.2f} – {X[:,0].max():.2f} mm/day")
    print(f"ET range:     {X[:,1].min():.2f} – {X[:,1].max():.2f} mm/day")

    # Normalize
    X_norm, mean, std = normalize(X)
    print(f"Observation mean: {mean}, std: {std}")

    # --- Train ---
    model = GaussianHMM(K=args.K, max_iter=200, tol=1e-4)
    model.fit(X_norm)

    # --- Inspect ---
    model.print_params()

    print("\n=== Emission means (original scale) ===")
    for k in range(args.K):
        mu_orig = model.mus[k] * std + mean
        print(f"  State {k}: precip={mu_orig[0]:.2f} mm/day, ET={mu_orig[1]:.2f} mm/day")

    # --- Decode ---
    states, log_prob = model.decode(X_norm)
    print(f"\nViterbi log-prob: {log_prob:.4f}")

    print("\n=== Regime duration statistics ===")
    for k in range(args.K):
        days_in_k = (states == k).sum()
        pct = 100 * days_in_k / len(states)
        print(f"  State {k}: {days_in_k} days ({pct:.1f}%)")

    # --- Plots ---
    plot_log_likelihood(model.log_likelihoods)
    plot_transition_matrix(model.A)
    plot_decoded_sequence(states, dates)

    np.save("outputs/decoded_states.npy", states)
    np.save("outputs/mus.npy", model.mus)
    np.save("outputs/sigmas.npy", model.sigmas)
    np.save("outputs/A.npy", model.A)
    np.save("outputs/log_likelihoods.npy", np.array(model.log_likelihoods))
    if dates is not None:
        import pandas as pd
        pd.Series(states, index=dates).to_csv("outputs/decoded_states_dated.csv", header=["state"])
    print("\nOutputs saved to outputs/")


if __name__ == "__main__":
    main()
