"""
app.py — Streamlit dashboard for the SJV Drought Regime HMM.

Loads pre-computed model outputs from outputs/ and visualizes:
  1. Regime timeline (interactive, zoomable)
  2. Raw observations overlaid with decoded states
  3. Transition matrix heatmap
  4. Regime duration & seasonal breakdown
  5. Baum-Welch convergence

Run with:  streamlit run app.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap

# ── page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SJV Drought Regimes",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── design tokens ─────────────────────────────────────────────────────────────
# Earthy, drought-appropriate palette — one color per regime
REGIME_COLORS = {
    0: "#7A8C7E",   # sage-grey  → cool-dry (suppressed precip + ET)
    1: "#5B8DB8",   # steel-blue → wet (precip surplus)
    2: "#D4A843",   # gold       → warm-wet (precip + ET both above normal)
    3: "#B94040",   # brick-red  → hot drought (precip deficit + high ET demand)
}

REGIME_LABELS = {
    0: "Cool-Dry",
    1: "Wet",
    2: "Warm-Wet",
    3: "Hot Drought",
}

PLOTLY_LAYOUT = dict(
    font_family="Inter, system-ui, sans-serif",
    font_color="#1a1a1a",
    paper_bgcolor="white",
    plot_bgcolor="white",
    margin=dict(l=48, r=24, t=40, b=48),
)

# ── minimal CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
  /* tighten default Streamlit padding */
  .block-container { padding-top: 2rem; padding-bottom: 2rem; }
  /* metric label */
  [data-testid="stMetricLabel"] { font-size: 0.75rem; color: #666; }
  /* section headers */
  h3 { font-weight: 600; letter-spacing: -0.02em; margin-top: 0; }
</style>
""", unsafe_allow_html=True)


# ── data loading ──────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Loading model outputs…")
def load_outputs():
    states = np.load("outputs/decoded_states.npy")
    mus    = np.load("outputs/mus.npy")
    sigmas = np.load("outputs/sigmas.npy")
    A      = np.load("outputs/A.npy")

    df_dated = pd.read_csv(
        "outputs/decoded_states_dated.csv",
        index_col=0, parse_dates=True,
    )
    dates = df_dated.index

    log_likelihoods = None
    # try to load convergence data if main.py saved it
    ll_path = "outputs/log_likelihoods.npy"
    if os.path.exists(ll_path):
        log_likelihoods = np.load(ll_path)

    return states, mus, sigmas, A, dates, log_likelihoods


@st.cache_data(show_spinner="Loading observations…")
def load_obs():
    from data_loader import load_netcdf, normalize
    X, dates = load_netcdf("data/gpm_sjv_subset.nc", "data/openet_sjv_subset.nc")
    X_norm, mean, std = normalize(X)
    return X, X_norm, dates, mean, std


def run_and_cache_model():
    """Train model and cache outputs if outputs are missing."""
    from hmm import GaussianHMM
    from data_loader import normalize
    import json

    X, X_norm, obs_dates, mean, std = load_obs()
    model = GaussianHMM(K=4, max_iter=200, tol=1e-4)

    progress = st.progress(0, text="Running Baum-Welch…")
    _orig_fit = model.fit

    def fit_with_progress(X_):
        model._initialize(X_)
        from emissions import compute_log_emission_matrix
        from forward_backward import forward_backward
        import math
        prev_ll = -math.inf
        for i in range(model.max_iter):
            from emissions import compute_log_emission_matrix
            log_B = compute_log_emission_matrix(X_, model.mus, model.sigmas)
            gamma, xi, ll = forward_backward(log_B, model.A, model.pi)
            model.log_likelihoods.append(ll)
            pct = min(int((i + 1) / model.max_iter * 100), 99)
            progress.progress(pct, text=f"Baum-Welch iteration {i+1}  (LL={ll:.1f})")
            if abs(ll - prev_ll) < model.tol:
                break
            prev_ll = ll
            model._m_step(X_, gamma, xi)
        progress.progress(100, text="Done.")
        return model

    model = fit_with_progress(X_norm)

    os.makedirs("outputs", exist_ok=True)
    states, log_prob = model.decode(X_norm)
    np.save("outputs/decoded_states.npy", states)
    np.save("outputs/mus.npy", model.mus)
    np.save("outputs/sigmas.npy", model.sigmas)
    np.save("outputs/A.npy", model.A)
    np.save("outputs/log_likelihoods.npy", np.array(model.log_likelihoods))
    pd.Series(states, index=obs_dates).to_csv(
        "outputs/decoded_states_dated.csv", header=["state"]
    )
    st.cache_data.clear()
    st.rerun()


# ── helpers ───────────────────────────────────────────────────────────────────
def smooth_states(s_arr, window):
    """Rolling majority vote over a centred window (display only, model unchanged)."""
    if window <= 1:
        return s_arr
    half = window // 2
    out = s_arr.copy()
    for i in range(len(s_arr)):
        lo, hi = max(0, i - half), min(len(s_arr), i + half + 1)
        vals, counts = np.unique(s_arr[lo:hi], return_counts=True)
        out[i] = vals[np.argmax(counts)]
    return out


def regime_color_list(states):
    return [REGIME_COLORS[s] for s in states]


def run_length_stats(states):
    """Return dict of {state: list_of_run_lengths}."""
    runs = {k: [] for k in REGIME_COLORS}
    if len(states) == 0:
        return runs
    cur, length = states[0], 1
    for s in states[1:]:
        if s == cur:
            length += 1
        else:
            runs[cur].append(length)
            cur, length = s, 1
    runs[cur].append(length)
    return runs


# ── main ──────────────────────────────────────────────────────────────────────
outputs_exist = (
    os.path.exists("outputs/decoded_states.npy")
    and os.path.exists("outputs/decoded_states_dated.csv")
)

# ── sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## San Joaquin Valley\nDrought Regime HMM")
    st.divider()

    st.markdown("**Model**")
    st.markdown("4-state Gaussian HMM  \nBaum-Welch (EM), from scratch  \nViterbi decoding")
    st.divider()

    st.markdown("**Data**")
    st.markdown("GPM IMERG precip → monthly totals  \nOpenET monthly ET  \nAnomaly from climatological mean  \n252 months · 2000-01 – 2020-12")
    st.divider()

    if not outputs_exist:
        st.warning("No model outputs found.")
        if st.button("Train model now", type="primary"):
            run_and_cache_model()
        st.stop()
    else:
        if st.button("Retrain model", type="secondary"):
            run_and_cache_model()

    st.divider()
    st.markdown("**Regime legend**")
    for k, label in REGIME_LABELS.items():
        color = REGIME_COLORS[k]
        st.markdown(
            f'<span style="display:inline-block;width:12px;height:12px;'
            f'border-radius:2px;background:{color};margin-right:6px"></span>{label}',
            unsafe_allow_html=True,
        )

# ── load data ─────────────────────────────────────────────────────────────────
states, mus, sigmas, A, dates, log_likelihoods = load_outputs()
X, X_norm, obs_dates, obs_mean, obs_std = load_obs()

# Align observations to decoded-state dates
obs_df = pd.DataFrame(X, index=obs_dates, columns=["precip_anom", "et_anom"])
obs_df = obs_df.loc[dates]
obs_df["state"] = states
obs_df["regime"] = obs_df["state"].map(REGIME_LABELS)
obs_df["color"] = obs_df["state"].map(REGIME_COLORS)

# Unscale emission means back to physical units
mus_phys = mus * obs_std + obs_mean

# ── title row ─────────────────────────────────────────────────────────────────
st.markdown("## SJV Drought Regime Analysis  \n###### Hidden Markov Model · 2000–2020")
st.divider()

# ── summary metrics ───────────────────────────────────────────────────────────
cols = st.columns(4)
for k, col in enumerate(cols):
    days = (states == k).sum()
    pct  = 100 * days / len(states)
    col.metric(
        label=REGIME_LABELS[k],
        value=f"{pct:.1f}%",
        delta=f"{days:,} days",
        delta_color="off",
    )

st.divider()

# ── Tab layout ────────────────────────────────────────────────────────────────
tab_timeline, tab_obs, tab_params, tab_duration, tab_convergence = st.tabs([
    "Regime Timeline", "Observations", "Model Parameters", "Duration Analysis", "Convergence"
])

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — REGIME TIMELINE
# ═══════════════════════════════════════════════════════════════════════════════
with tab_timeline:
    st.markdown("### Decoded Drought Regimes — Daily, 2000–2020")

    smooth_days = st.slider(
        "Smoothing window (days) — majority vote, display only",
        min_value=1, max_value=30, value=14, step=1,
        help="Suppresses isolated 1–3 day state flips. Does not change the model.",
    )

    display_states = smooth_states(states, smooth_days)
    display_df = obs_df.copy()
    display_df["state"] = display_states
    display_df["regime"] = display_df["state"].map(REGIME_LABELS)

    st.caption("Coloured ribbon — each row shows when that regime was active. Drag to zoom, double-click to reset.")

    # ── Build run-length encoded segments for px.timeline (Gantt ribbon) ──
    def rle_segments(df):
        rows = []
        cur_state = int(df["state"].iloc[0])
        cur_start = df.index[0]
        for ts, row in df.iterrows():
            s = int(row["state"])
            if s != cur_state:
                rows.append({"start": cur_start, "end": ts, "state": cur_state})
                cur_state, cur_start = s, ts
        rows.append({"start": cur_start, "end": df.index[-1] + pd.Timedelta(days=1), "state": cur_state})
        segs = pd.DataFrame(rows)
        segs["regime"] = segs["state"].map(REGIME_LABELS)
        segs["dur"] = (segs["end"] - segs["start"]).dt.days
        return segs

    segs = rle_segments(display_df)

    color_map = {REGIME_LABELS[k]: REGIME_COLORS[k] for k in range(4)}
    # Row order: arrange regimes from wettest to driest top-to-bottom
    row_order = [REGIME_LABELS[k] for k in range(4)]

    fig = px.timeline(
        segs,
        x_start="start", x_end="end",
        y="regime",          # each regime on its own row
        color="regime",
        color_discrete_map=color_map,
        category_orders={"regime": row_order},
        hover_data={"start": "|%b %d, %Y", "end": "|%b %d, %Y",
                    "dur": True, "regime": False},
        labels={"dur": "days", "start": "Start", "end": "End"},
    )
    fig.update_layout(
        **{**PLOTLY_LAYOUT, "margin": dict(l=16, r=16, t=16, b=16)},
        height=220,
        xaxis=dict(showgrid=False, title=None,
                   rangeslider=dict(visible=True, thickness=0.12)),
        yaxis=dict(title=None, showgrid=False, tickfont=dict(size=12)),
        showlegend=False,
    )
    # Make bars tall enough to look like solid bands
    fig.update_traces(marker_line_width=0)
    st.plotly_chart(fig, use_container_width=True)

    # ── annual regime composition bar chart ──────────────────────────────────
    st.markdown("### Annual Regime Composition")
    obs_df["year"] = obs_df.index.year
    annual = (
        obs_df.groupby(["year", "state"])
        .size()
        .reset_index(name="days")
    )
    annual["pct"] = annual.groupby("year")["days"].transform(lambda x: 100 * x / x.sum())
    annual["regime"] = annual["state"].map(REGIME_LABELS)

    fig2 = go.Figure()
    for k in range(4):
        sub = annual[annual["state"] == k]
        fig2.add_trace(go.Bar(
            x=sub["year"], y=sub["pct"],
            name=REGIME_LABELS[k],
            marker_color=REGIME_COLORS[k],
            hovertemplate="%{x}: %{y:.1f}%<extra>" + REGIME_LABELS[k] + "</extra>",
        ))

    fig2.update_layout(
        **PLOTLY_LAYOUT,
        barmode="stack",
        height=300,
        xaxis=dict(title=None, dtick=2),
        yaxis=dict(title="% of days", ticksuffix="%", showgrid=True,
                   gridcolor="#f0f0f0"),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02,
            xanchor="left", x=0, itemsizing="constant",
        ),
    )
    st.plotly_chart(fig2, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — OBSERVATIONS
# ═══════════════════════════════════════════════════════════════════════════════
with tab_obs:
    st.markdown("### Raw Observations Coloured by Decoded Regime")

    fig3 = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=("Precip anomaly (mm/mo)", "ET anomaly (mm/mo)"),
    )

    for k in range(4):
        mask = obs_df["state"] == k
        # precip
        fig3.add_trace(go.Scatter(
            x=obs_df.index[mask], y=obs_df.loc[mask, "precip_anom"],
            mode="markers",
            name=REGIME_LABELS[k],
            legendgroup=str(k),
            marker=dict(color=REGIME_COLORS[k], size=2.5, opacity=0.6),
            showlegend=True,
            hovertemplate="%{x|%Y-%m-%d}: %{y:.1f} mm<extra>" + REGIME_LABELS[k] + "</extra>",
        ), row=1, col=1)
        # ET
        fig3.add_trace(go.Scatter(
            x=obs_df.index[mask], y=obs_df.loc[mask, "et_anom"],
            mode="markers",
            name=REGIME_LABELS[k],
            legendgroup=str(k),
            marker=dict(color=REGIME_COLORS[k], size=2.5, opacity=0.6),
            showlegend=False,
            hovertemplate="%{x|%Y-%m-%d}: %{y:.1f} mm<extra>" + REGIME_LABELS[k] + "</extra>",
        ), row=2, col=1)

    fig3.update_layout(
        **PLOTLY_LAYOUT,
        height=500,
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02,
            xanchor="left", x=0, itemsizing="constant",
        ),
    )
    fig3.update_xaxes(showgrid=False)
    fig3.update_yaxes(showgrid=True, gridcolor="#f0f0f0")
    st.plotly_chart(fig3, use_container_width=True)

    # ── scatter: precip vs ET coloured by regime ──────────────────────────────
    st.markdown("### Emission Space — Precip vs. ET")
    st.caption("Each point is one day. ✕ marks the learned emission mean for each state.")

    fig4 = go.Figure()
    for k in range(4):
        mask = obs_df["state"] == k
        fig4.add_trace(go.Scatter(
            x=obs_df.loc[mask, "precip_anom"],
            y=obs_df.loc[mask, "et_anom"],
            mode="markers",
            name=REGIME_LABELS[k],
            marker=dict(color=REGIME_COLORS[k], size=3, opacity=0.4),
            hovertemplate=(
                "precip: %{x:.1f} mm<br>ET: %{y:.1f} mm"
                "<extra>" + REGIME_LABELS[k] + "</extra>"
            ),
        ))
        # emission mean
        fig4.add_trace(go.Scatter(
            x=[mus_phys[k, 0]], y=[mus_phys[k, 1]],
            mode="markers+text",
            marker=dict(
                symbol="x", color=REGIME_COLORS[k],
                size=14, line=dict(width=2.5),
            ),
            text=[REGIME_LABELS[k]],
            textposition="top center",
            textfont=dict(size=11, color=REGIME_COLORS[k]),
            showlegend=False,
            hovertemplate=(
                f"<b>μ — {REGIME_LABELS[k]}</b><br>"
                f"precip: {mus_phys[k,0]:.2f} mm<br>"
                f"ET: {mus_phys[k,1]:.2f} mm<extra></extra>"
            ),
        ))

    fig4.update_layout(
        **PLOTLY_LAYOUT,
        height=420,
        xaxis=dict(title="Precip anomaly (mm/mo)", showgrid=True, gridcolor="#f0f0f0"),
        yaxis=dict(title="ET anomaly (mm/mo)", showgrid=True, gridcolor="#f0f0f0"),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02,
            xanchor="left", x=0, itemsizing="constant",
        ),
    )
    st.plotly_chart(fig4, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — MODEL PARAMETERS
# ═══════════════════════════════════════════════════════════════════════════════
with tab_params:
    col_A, col_mus = st.columns([1, 1])

    with col_A:
        st.markdown("### Transition Matrix")
        st.caption("Row → column: probability of moving from regime *i* to regime *j*.")

        labels = [REGIME_LABELS[k] for k in range(4)]
        fig5 = go.Figure(go.Heatmap(
            z=A,
            x=labels,
            y=labels,
            colorscale=[
                [0.0, "#f7f7f7"],
                [0.5, "#9ecae1"],
                [1.0, "#084594"],
            ],
            zmin=0, zmax=1,
            text=[[f"{A[i,j]:.3f}" for j in range(4)] for i in range(4)],
            texttemplate="%{text}",
            textfont=dict(size=13),
            hovertemplate=(
                "%{y} → %{x}<br>Prob: %{z:.4f}<extra></extra>"
            ),
            showscale=True,
        ))
        fig5.update_layout(
            **PLOTLY_LAYOUT,
            height=360,
            xaxis=dict(title="To", side="bottom"),
            yaxis=dict(title="From", autorange="reversed"),
        )
        st.plotly_chart(fig5, use_container_width=True)

    with col_mus:
        st.markdown("### Emission Parameters (original scale)")

        rows = []
        for k in range(4):
            rows.append({
                "Regime": REGIME_LABELS[k],
                "Precip μ anom (mm/mo)": f"{mus_phys[k,0]:.1f}",
                "ET μ anom (mm/mo)": f"{mus_phys[k,1]:.1f}",
                "Precip σ": f"{np.sqrt(sigmas[k,0,0]) * obs_std[0]:.3f}",
                "ET σ": f"{np.sqrt(sigmas[k,1,1]) * obs_std[1]:.2f}",
            })
        param_df = pd.DataFrame(rows).set_index("Regime")
        st.dataframe(param_df, use_container_width=True)

        st.divider()
        st.markdown("**Initial distribution π**")
        pi_rows = [
            {"Regime": REGIME_LABELS[k], "π": f"{A[0,k]:.4f}"}  # approx stationary
            for k in range(4)
        ]
        # Actually compute stationary distribution from A
        # eigendecomposition: find left eigenvector for eigenvalue 1
        eigvals, eigvecs = np.linalg.eig(A.T)
        idx = np.argmin(np.abs(eigvals - 1.0))
        stat = np.real(eigvecs[:, idx])
        stat = np.abs(stat) / np.abs(stat).sum()

        pi_df = pd.DataFrame({
            "Regime": [REGIME_LABELS[k] for k in range(4)],
            "Stationary π": [f"{stat[k]:.3f}" for k in range(4)],
        }).set_index("Regime")
        st.dataframe(pi_df, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 — DURATION ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════
with tab_duration:
    runs = run_length_stats(states)

    st.markdown("### Regime Duration Distributions")
    st.caption(
        "Distribution of consecutive-day run lengths within each regime. "
        "Geometric if the HMM is well-specified."
    )

    fig6 = make_subplots(
        rows=2, cols=2,
        subplot_titles=[REGIME_LABELS[k] for k in range(4)],
        vertical_spacing=0.14,
        horizontal_spacing=0.10,
    )
    positions = [(1,1),(1,2),(2,1),(2,2)]
    for k, (r, c) in zip(range(4), positions):
        rl = runs[k]
        if not rl:
            continue
        mean_dur = np.mean(rl)
        fig6.add_trace(go.Histogram(
            x=rl,
            nbinsx=min(40, max(rl)),
            marker_color=REGIME_COLORS[k],
            opacity=0.85,
            showlegend=False,
            hovertemplate="Duration: %{x} days<br>Count: %{y}<extra></extra>",
        ), row=r, col=c)
        fig6.add_vline(
            x=mean_dur,
            line_dash="dash",
            line_color="#333",
            line_width=1.5,
            annotation_text=f"μ={mean_dur:.1f}d",
            annotation_font_size=10,
            row=r, col=c,
        )

    fig6.update_layout(
        **PLOTLY_LAYOUT,
        height=460,
    )
    fig6.update_xaxes(title_text="Days", showgrid=False)
    fig6.update_yaxes(title_text="Count", showgrid=True, gridcolor="#f0f0f0")
    st.plotly_chart(fig6, use_container_width=True)

    # ── seasonal heatmap ─────────────────────────────────────────────────────
    st.markdown("### Seasonal Regime Frequency")
    st.caption("Fraction of days in each regime, broken out by calendar month.")

    obs_df["month"] = obs_df.index.month
    seasonal = (
        obs_df.groupby(["month", "state"])
        .size()
        .unstack(fill_value=0)
    )
    seasonal_pct = seasonal.div(seasonal.sum(axis=1), axis=0) * 100

    month_names = ["Jan","Feb","Mar","Apr","May","Jun",
                   "Jul","Aug","Sep","Oct","Nov","Dec"]

    fig7 = go.Figure()
    for k in range(4):
        vals = [seasonal_pct.loc[m, k] if k in seasonal_pct.columns else 0
                for m in range(1, 13)]
        fig7.add_trace(go.Bar(
            x=month_names,
            y=vals,
            name=REGIME_LABELS[k],
            marker_color=REGIME_COLORS[k],
            hovertemplate="%{x}: %{y:.1f}%<extra>" + REGIME_LABELS[k] + "</extra>",
        ))

    fig7.update_layout(
        **PLOTLY_LAYOUT,
        barmode="stack",
        height=320,
        xaxis=dict(title=None),
        yaxis=dict(title="% of days", ticksuffix="%", showgrid=True, gridcolor="#f0f0f0"),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02,
            xanchor="left", x=0, itemsizing="constant",
        ),
    )
    st.plotly_chart(fig7, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 5 — CONVERGENCE
# ═══════════════════════════════════════════════════════════════════════════════
with tab_convergence:
    st.markdown("### Baum-Welch Convergence")

    if log_likelihoods is not None and len(log_likelihoods) > 0:
        fig8 = go.Figure(go.Scatter(
            x=list(range(1, len(log_likelihoods) + 1)),
            y=log_likelihoods,
            mode="lines+markers",
            marker=dict(size=5, color="#5B8DB8"),
            line=dict(color="#5B8DB8", width=2),
            hovertemplate="Iter %{x}: LL = %{y:.2f}<extra></extra>",
        ))
        fig8.update_layout(
            **PLOTLY_LAYOUT,
            height=340,
            xaxis=dict(title="Baum-Welch Iteration", showgrid=True, gridcolor="#f0f0f0"),
            yaxis=dict(title="Log-likelihood", showgrid=True, gridcolor="#f0f0f0"),
        )
        st.plotly_chart(fig8, use_container_width=True)

        n_iter = len(log_likelihoods)
        final_ll = log_likelihoods[-1]
        delta = abs(log_likelihoods[-1] - log_likelihoods[-2]) if n_iter > 1 else float("nan")
        c1, c2, c3 = st.columns(3)
        c1.metric("Iterations", n_iter)
        c2.metric("Final log-likelihood", f"{final_ll:.2f}")
        c3.metric("|ΔLL| at convergence", f"{delta:.2e}")
    else:
        st.info(
            "Convergence data not found. Retrain the model using the sidebar button "
            "to generate `outputs/log_likelihoods.npy`."
        )
        # show the saved convergence plot as fallback
        if os.path.exists("outputs/convergence.png"):
            st.image("outputs/convergence.png", use_column_width=True)
