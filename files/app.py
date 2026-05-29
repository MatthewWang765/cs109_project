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
    0: "#3978AE",   # deep blue   → pluvial (12-mo wet)
    1: "#8B2E2A",   # dark red    → persistent drought
    2: "#E08E3C",   # amber       → hot regime (warming-driven)
    3: "#9CA89C",   # sage-grey   → near-normal
}

REGIME_LABELS = {
    0: "Pluvial (Wet)",
    1: "Drought",
    2: "Hot Regime",
    3: "Near-Normal",
}

# Map state → which drought category it contributes to.
# Used to build the continuous drought-intensity score (Σ γ over dry states).
DROUGHT_STATES = {1: 1.0, 2: 0.4}    # full weight on persistent drought, partial on hot

# U.S. Drought Monitor severity scale (mapped to P(drought | data))
DROUGHT_MONITOR_SCALE = [
    (0.00, 0.20, "#FFFFFF", "None"),
    (0.20, 0.40, "#FFFF00", "D0 — Abnormally Dry"),
    (0.40, 0.60, "#FCD37F", "D1 — Moderate"),
    (0.60, 0.80, "#FFAA00", "D2 — Severe"),
    (0.80, 0.95, "#E60000", "D3 — Extreme"),
    (0.95, 1.01, "#730000", "D4 — Exceptional"),
]

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
    ll_path = "outputs/log_likelihoods.npy"
    if os.path.exists(ll_path):
        log_likelihoods = np.load(ll_path)

    gamma = None
    g_path = "outputs/gamma.npy"
    if os.path.exists(g_path):
        gamma = np.load(g_path)

    return states, mus, sigmas, A, dates, log_likelihoods, gamma


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
    gamma = model.posterior(X_norm)
    np.save("outputs/decoded_states.npy", states)
    np.save("outputs/mus.npy", model.mus)
    np.save("outputs/sigmas.npy", model.sigmas)
    np.save("outputs/A.npy", model.A)
    np.save("outputs/log_likelihoods.npy", np.array(model.log_likelihoods))
    np.save("outputs/gamma.npy", gamma)
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
    st.markdown(
        "GPM IMERG precip  \nOpenET monthly ET  \n"
        "12-month rolling sums  \nz-scored by calendar month  \n"
        "(equivalent to SPI-12 / ETI-12)  \n"
        "241 months · 2000-12 – 2020-12"
    )
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
states, mus, sigmas, A, dates, log_likelihoods, gamma = load_outputs()
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
tab_severity, tab_timeline, tab_obs, tab_params, tab_duration, tab_convergence = st.tabs([
    "Drought Severity", "Regime Timeline", "Observations",
    "Model Parameters", "Duration Analysis", "Convergence",
])

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 0 — DROUGHT SEVERITY (continuous posterior)
# ═══════════════════════════════════════════════════════════════════════════════
with tab_severity:
    st.markdown("### Drought Intensity — Bayesian Posterior  P(drought | data)")
    st.caption(
        "Continuous drought-severity score derived from the forward–backward smoothed posterior γₜ(k). "
        "Unlike Viterbi (which forces a single hard label per month), this preserves the full probability "
        "distribution over states — exactly what the U.S. Drought Monitor expresses as a severity gradient. "
        "Score = γₜ(Drought) + 0.4·γₜ(Hot Regime)."
    )

    if gamma is None:
        st.error("Posterior gamma not found. Run `python main.py` to regenerate outputs.")
    else:
        # Build drought intensity = weighted sum of dry-state posteriors
        intensity = np.zeros(len(states))
        for k, w in DROUGHT_STATES.items():
            intensity += w * gamma[:, k]
        intensity = np.clip(intensity, 0.0, 1.0)

        sev_df = pd.DataFrame({"intensity": intensity}, index=dates)

        # ── Categorise into Drought Monitor classes ──────────────────────────
        def classify(val):
            for lo, hi, color, label in DROUGHT_MONITOR_SCALE:
                if lo <= val < hi:
                    return label, color
            return DROUGHT_MONITOR_SCALE[-1][3], DROUGHT_MONITOR_SCALE[-1][2]

        sev_df["category"] = sev_df["intensity"].apply(lambda v: classify(v)[0])
        sev_df["color"]    = sev_df["intensity"].apply(lambda v: classify(v)[1])

        # ── Headline severity-strip plot ─────────────────────────────────────
        # Build a continuous filled scatter that looks like the Drought Monitor.
        fig_strip = go.Figure()
        # Background fill — shaded bands per severity category for the legend
        for lo, hi, color, label in DROUGHT_MONITOR_SCALE[1:]:   # skip "None"
            fig_strip.add_trace(go.Scatter(
                x=[None], y=[None],
                mode="markers",
                marker=dict(color=color, size=12, symbol="square"),
                name=label,
                showlegend=True,
            ))

        # The intensity line itself, coloured by category at each point.
        # Use a filled area under the curve — this gives the Drought Monitor look.
        fig_strip.add_trace(go.Scatter(
            x=sev_df.index, y=sev_df["intensity"],
            mode="lines",
            line=dict(color="#222", width=1.5),
            fill="tozeroy",
            fillcolor="rgba(139, 46, 42, 0.08)",
            showlegend=False,
            hovertemplate="<b>%{x|%b %Y}</b><br>P(drought) = %{y:.2f}<extra></extra>",
        ))

        # Horizontal threshold bands as shaded rectangles
        for lo, hi, color, label in DROUGHT_MONITOR_SCALE[1:]:
            fig_strip.add_hrect(
                y0=lo, y1=hi,
                fillcolor=color, opacity=0.18,
                line_width=0, layer="below",
            )

        fig_strip.update_layout(
            **PLOTLY_LAYOUT,
            height=380,
            yaxis=dict(
                title="P(drought | data)",
                range=[0, 1], showgrid=False,
                tickformat=".0%",
            ),
            xaxis=dict(showgrid=False, title=None),
            legend=dict(
                orientation="h", yanchor="bottom", y=1.04,
                xanchor="left", x=0,
                itemsizing="constant",
                title=None,
            ),
        )

        # Annotate the famous droughts
        annotations = [
            ("2008-07-01", "2007–09 drought"),
            ("2014-09-01", "2012–17 mega-drought"),
            ("2020-09-01", "2020 drought"),
        ]
        for date_str, label in annotations:
            ts = pd.Timestamp(date_str)
            if ts in sev_df.index or (sev_df.index[0] <= ts <= sev_df.index[-1]):
                # interpolate intensity
                yv = float(sev_df["intensity"].asof(ts))
                fig_strip.add_annotation(
                    x=ts, y=min(yv + 0.08, 0.98),
                    text=label, showarrow=True, arrowhead=0,
                    ay=-30, ax=0,
                    font=dict(size=11, color="#222"),
                    bgcolor="rgba(255,255,255,0.85)",
                    bordercolor="#888", borderwidth=0.5, borderpad=3,
                )

        st.plotly_chart(fig_strip, use_container_width=True)

        # ── Summary metrics ──────────────────────────────────────────────────
        annual = sev_df["intensity"].groupby(sev_df.index.year).mean()
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Driest year (annual mean)",  annual.idxmax(),
                  delta=f"P = {annual.max():.2f}", delta_color="off")
        c2.metric("Wettest year",               annual.idxmin(),
                  delta=f"P = {annual.min():.2f}", delta_color="off")
        c3.metric("Months in D3+ Extreme",      int((intensity >= 0.80).sum()),
                  delta=f"{(intensity >= 0.80).mean()*100:.0f}% of record",
                  delta_color="off")
        c4.metric("Months in D4 Exceptional",   int((intensity >= 0.95).sum()),
                  delta=f"{(intensity >= 0.95).mean()*100:.0f}% of record",
                  delta_color="off")

        # ── Posterior probabilities per state (stacked area) ─────────────────
        st.markdown("### Posterior State Probabilities  γₜ(k)")
        st.caption(
            "Full Bayesian smoothed posterior from forward–backward. "
            "Each column sums to 1; the stack shows the probability mass on each regime at every month."
        )

        fig_post = go.Figure()
        # Stack ordering: Wet at bottom, then Near-Normal, then Hot, then Drought on top
        stack_order = [0, 3, 2, 1]
        for k in stack_order:
            fig_post.add_trace(go.Scatter(
                x=dates, y=gamma[:, k],
                mode="lines",
                stackgroup="post",
                name=REGIME_LABELS[k],
                line=dict(width=0),
                fillcolor=REGIME_COLORS[k],
                hovertemplate=f"<b>%{{x|%b %Y}}</b><br>γ({REGIME_LABELS[k]}) = %{{y:.2f}}<extra></extra>",
            ))
        fig_post.update_layout(
            **PLOTLY_LAYOUT,
            height=320,
            yaxis=dict(title="Posterior probability",
                       range=[0, 1], tickformat=".0%",
                       showgrid=True, gridcolor="#f0f0f0"),
            xaxis=dict(showgrid=False, title=None),
            legend=dict(
                orientation="h", yanchor="bottom", y=1.04,
                xanchor="left", x=0, itemsizing="constant",
            ),
        )
        st.plotly_chart(fig_post, use_container_width=True)

        # ── Annual ranking table ─────────────────────────────────────────────
        st.markdown("### Annual Drought Ranking")
        rank_df = pd.DataFrame({
            "Year":          annual.index,
            "P(drought)":    annual.values.round(2),
            "Severity":      [classify(v)[0] for v in annual.values],
        }).set_index("Year")
        rank_df = rank_df.sort_values("P(drought)", ascending=False)
        st.dataframe(rank_df, use_container_width=True, height=420)

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
    st.markdown("### Standardised Anomalies — What the Model Sees")
    st.caption(
        "Inputs to the HMM: 3-month rolling sums of precipitation and ET, "
        "expressed as standardised anomalies (σ from the climatological mean for that calendar month). "
        "By construction these have no seasonal cycle — the dashed line at 0 is normal, "
        "positive = wetter / higher ET than typical for that month."
    )

    fig3 = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.10,
        subplot_titles=("SPI-3  (precipitation, σ)", "ETI-3  (evapotranspiration, σ)"),
    )

    for k in range(4):
        mask = obs_df["state"] == k
        fig3.add_trace(go.Scatter(
            x=obs_df.index[mask], y=obs_df.loc[mask, "precip_anom"],
            mode="markers",
            name=REGIME_LABELS[k],
            legendgroup=str(k),
            marker=dict(color=REGIME_COLORS[k], size=6, opacity=0.85,
                        line=dict(width=0.5, color="white")),
            showlegend=True,
            hovertemplate="%{x|%b %Y}: %{y:+.2f}σ<extra>" + REGIME_LABELS[k] + "</extra>",
        ), row=1, col=1)
        fig3.add_trace(go.Scatter(
            x=obs_df.index[mask], y=obs_df.loc[mask, "et_anom"],
            mode="markers",
            name=REGIME_LABELS[k],
            legendgroup=str(k),
            marker=dict(color=REGIME_COLORS[k], size=6, opacity=0.85,
                        line=dict(width=0.5, color="white")),
            showlegend=False,
            hovertemplate="%{x|%b %Y}: %{y:+.2f}σ<extra>" + REGIME_LABELS[k] + "</extra>",
        ), row=2, col=1)

    # Zero reference lines
    fig3.add_hline(y=0, line_dash="dash", line_color="#999", line_width=1, row=1, col=1)
    fig3.add_hline(y=0, line_dash="dash", line_color="#999", line_width=1, row=2, col=1)

    fig3.update_layout(
        **PLOTLY_LAYOUT,
        height=520,
        legend=dict(
            orientation="h", yanchor="bottom", y=1.04,
            xanchor="left", x=0, itemsizing="constant",
        ),
    )
    fig3.update_xaxes(showgrid=False, title=None)
    fig3.update_yaxes(showgrid=True, gridcolor="#f0f0f0",
                      zeroline=False, title="σ from monthly mean")
    st.plotly_chart(fig3, use_container_width=True)

    # ── scatter: precip vs ET coloured by regime ──────────────────────────────
    st.markdown("### Emission Space — SPI-3 vs. ETI-3")
    st.caption(
        "Each point is one month. ✕ marks the learned emission mean (μ_k) for each state. "
        "Drought states sit in the left half (precip deficit), heat-driven states in the upper half."
    )

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
                "SPI-3: %{x:+.2f}σ<br>ETI-3: %{y:+.2f}σ"
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
                f"SPI-3: {mus_phys[k,0]:+.2f}σ<br>"
                f"ETI-3: {mus_phys[k,1]:+.2f}σ<extra></extra>"
            ),
        ))

    # quadrant guides
    fig4.add_hline(y=0, line_dash="dash", line_color="#bbb", line_width=1)
    fig4.add_vline(x=0, line_dash="dash", line_color="#bbb", line_width=1)

    fig4.update_layout(
        **PLOTLY_LAYOUT,
        height=420,
        xaxis=dict(title="SPI-3  (precipitation anomaly, σ)", showgrid=True, gridcolor="#f0f0f0", zeroline=False),
        yaxis=dict(title="ETI-3  (ET anomaly, σ)", showgrid=True, gridcolor="#f0f0f0", zeroline=False),
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
                "SPI-3 μ (σ)": f"{mus_phys[k,0]:+.2f}",
                "ETI-3 μ (σ)": f"{mus_phys[k,1]:+.2f}",
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
