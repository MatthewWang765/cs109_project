"""
Streamlit app — Groundwater Well Failure Prediction
CS109 Stanford Challenge Project

Layout:
  Sidebar: well selector, forecast horizon slider
  Top:     Monte Carlo trajectory fan (last 24 months history + future sims)
  Bottom-left:  P(fail) big number + MLE params + failure curve
  Bottom-right: Bayesian updating chart (μ_post and P(fail) over time)
"""
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import os

st.set_page_config(
    page_title="Groundwater Failure Prediction",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ── Data loading ─────────────────────────────────────────────────────────────

@st.cache_data
def load_all():
    with open("data/mle_params.json") as f:
        mle = json.load(f)
    with open("data/monte_carlo_results.json") as f:
        mc = json.load(f)
    with open("data/bayesian_updates.json") as f:
        bayes = json.load(f)
    return mle, mc, bayes


@st.cache_data
def load_df(name):
    df = pd.read_csv(f"data/{name}_clean.csv", parse_dates=["date"])
    return df


def run_mc_live(x0, mu, sigma, x_fail, k_months, n_sims=5_000, seed=42):
    rng = np.random.default_rng(seed)
    steps = rng.normal(mu, sigma, size=(n_sims, k_months))
    trajs = x0 + np.cumsum(steps, axis=1)
    failed = np.any(trajs >= x_fail, axis=1)
    return float(failed.mean()), trajs


def compute_curve_live(x0, mu, sigma, x_fail, max_months=60, n_sims=5_000, seed=42):
    rng = np.random.default_rng(seed)
    steps = rng.normal(mu, sigma, size=(n_sims, max_months))
    trajs = x0 + np.cumsum(steps, axis=1)
    return [float(np.any(trajs[:, :k] >= x_fail, axis=1).mean()) for k in range(1, max_months + 1)]


# ── Guard: check data exists ──────────────────────────────────────────────────

if not os.path.exists("data/mle_params.json"):
    st.error(
        "Data files not found. Run the pipeline first:\n\n"
        "```\npython 1_fetch_data.py\npython 2_clean_data.py\n"
        "python 3_mle.py\npython 4_monte_carlo.py\npython 5_bayesian_update.py\n```"
    )
    st.stop()

mle_params, mc_results, bayesian_updates = load_all()
well_names = list(mle_params.keys())

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("Controls")
    selected = st.selectbox("Select Well", well_names)
    k_months = st.slider("Forecast horizon (months)", min_value=1, max_value=60, value=24)
    n_display = st.slider("Trajectories to display", 50, 500, 150, step=50)
    st.divider()
    st.caption(
        "**Model:** Gaussian drift MLE + Monte Carlo first-passage simulation "
        "+ Gaussian conjugate Bayesian updating\n\n"
        "**Data:** USGS National Groundwater Monitoring Network"
    )

params  = mle_params[selected]
mc      = mc_results[selected]
updates = bayesian_updates.get(selected, [])
df      = load_df(selected)

mu      = params["mu_hat"]
sigma   = params["sigma_hat"]
x_fail  = mc["x_fail"]
x0      = float(df["depth_ft"].iloc[-1])

# ── Header ────────────────────────────────────────────────────────────────────

st.title("Groundwater Well Failure Prediction")
st.caption("CS109 Stanford · MLE + Monte Carlo + Bayesian Updating · Real USGS Data")

# ── Top: Trajectory fan ───────────────────────────────────────────────────────

st.subheader(f"Monte Carlo Trajectory Fan — {selected}")

p_fail, trajs = run_mc_live(x0, mu, sigma, x_fail, k_months, n_sims=5_000)

fig1, ax1 = plt.subplots(figsize=(13, 5))
months = np.arange(1, k_months + 1)
rng = np.random.default_rng(7)
idx = rng.choice(5_000, min(n_display, 5_000), replace=False)

for i in idx:
    traj = trajs[i]
    failed = traj.max() >= x_fail
    ax1.plot(months, traj, alpha=0.07, lw=0.5, color="crimson" if failed else "steelblue")

# Show last 24 months of history for context
hist = df.tail(24)
hist_x = np.arange(-(len(hist) - 1), 1)
ax1.plot(hist_x, hist["depth_ft"].values, color="black", lw=2.5, label="Historical (last 24 mo)", zorder=5)
ax1.axvline(0, color="gray", lw=1, ls=":", label="Now")
ax1.axhline(x_fail, color="crimson", lw=2.5, ls="--", label=f"Failure depth ({x_fail:.0f} ft)")

n_failed = int(p_fail * 5_000)
ax1.set_xlabel("Months from now")
ax1.set_ylabel("Depth to water (ft below surface)")
ax1.set_title(
    f"N = 5,000 simulated trajectories   |   red = failure before month {k_months}   |   "
    f"{n_failed:,} / 5,000 fail  →  P̂ = {p_fail:.1%}"
)
ax1.legend(loc="upper left")
plt.tight_layout()
st.pyplot(fig1)
plt.close(fig1)

# ── Bottom row ────────────────────────────────────────────────────────────────

col_left, col_right = st.columns([1, 2])

# ── Left: P(fail) number + params + curve ─────────────────────────────────────

with col_left:
    st.subheader("Failure Probability")

    color = "red" if p_fail > 0.5 else "darkorange" if p_fail > 0.2 else "green"
    st.markdown(
        f"<div style='text-align:center; font-size:80px; font-weight:bold; color:{color}'>"
        f"{p_fail:.1%}</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        f"<div style='text-align:center; margin-bottom:12px'>P(fail within "
        f"<b>{k_months} months</b>)</div>",
        unsafe_allow_html=True,
    )
    st.divider()

    col_a, col_b = st.columns(2)
    col_a.metric("μ̂ drift (ft/mo)", f"{mu:+.4f}")
    col_b.metric("σ̂ noise (ft/mo)", f"{sigma:.4f}")
    col_a.metric("Current depth", f"{x0:.1f} ft")
    col_b.metric("Failure depth", f"{x_fail:.1f} ft")

    st.divider()
    st.caption("**P(fail within k months)**")
    curve = compute_curve_live(x0, mu, sigma, x_fail)
    fig2, ax2 = plt.subplots(figsize=(4, 2.8))
    ax2.plot(range(1, 61), curve, color="steelblue", lw=2)
    ax2.axvline(k_months, color="crimson", ls="--", lw=1.5)
    ax2.axhline(p_fail, color="crimson", ls=":", lw=1)
    ax2.set_xlim(1, 60)
    ax2.set_ylim(0, 1)
    ax2.set_xlabel("Months (k)")
    ax2.set_ylabel("P(fail)")
    plt.tight_layout()
    st.pyplot(fig2)
    plt.close(fig2)

# ── Right: Bayesian updating ──────────────────────────────────────────────────

with col_right:
    st.subheader("Bayesian Updating — 2022–2024 Held-out Replay")

    if updates:
        dates     = [u["date"][:7]   for u in updates]
        mu_posts  = [u["mu_post"]    for u in updates]
        sig_posts = [u["sigma_post"] for u in updates]
        p_fails_b = [u["p_fail"]     for u in updates]
        x = np.arange(len(dates))

        fig3, (ax3, ax4) = plt.subplots(2, 1, figsize=(8, 5), sharex=True)

        # μ_post with ±2σ uncertainty band
        ax3.plot(x, mu_posts, color="steelblue", lw=2, label="μ_post")
        ax3.fill_between(
            x,
            [m - 2*s for m, s in zip(mu_posts, sig_posts)],
            [m + 2*s for m, s in zip(mu_posts, sig_posts)],
            alpha=0.2, color="steelblue", label="±2σ_post",
        )
        ax3.axhline(mu, color="gray", ls="--", lw=1.2, label=f"μ̂_MLE = {mu:.4f}")
        ax3.axhline(0,  color="lightgray", ls=":", lw=1)
        ax3.set_ylabel("μ_post (ft/month)")
        ax3.set_title("Posterior drift estimate narrows as new data arrives")
        ax3.legend(fontsize=8)

        # P(fail) over time
        ax4.plot(x, p_fails_b, color="crimson", lw=2, label="P(fail ≤ 24 months)")
        ax4.fill_between(x, p_fails_b, alpha=0.15, color="crimson")
        ax4.set_ylim(0, 1)
        ax4.set_ylabel("P(fail)")
        ax4.set_xlabel("Months of new data observed")
        ax4.legend(fontsize=8)

        step = max(1, len(dates) // 8)
        ax4.set_xticks(x[::step])
        ax4.set_xticklabels(dates[::step], rotation=45, fontsize=8)

        plt.tight_layout()
        st.pyplot(fig3)
        plt.close(fig3)

        st.caption(
            f"{len(updates)} sequential updates replayed. "
            "Each month: observe Δ_new → update μ_post via Gaussian conjugate → "
            "re-run Monte Carlo with μ_post."
        )
    else:
        st.info("Run `5_bayesian_update.py` to generate the Bayesian update sequence.")

# ── Footer ────────────────────────────────────────────────────────────────────

st.divider()
st.caption(
    "Source: USGS National Groundwater Monitoring Network (parameter code 72019) · "
    "Model: N(μ,σ²) drift MLE → Monte Carlo first-passage time → Gaussian conjugate Bayesian updating · "
    "CS109 Stanford Challenge · June 2026"
)
