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
import time
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


@st.cache_data(show_spinner="Computing per-cell drought percentiles…")
def load_spatial():
    """Returns (lat, lon, dates, pct) — drought percentile per cell per month."""
    from data_loader import load_spatial_percentile
    return load_spatial_percentile("data/gpm_sjv_subset.nc", window=12)


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
tab_map, tab_severity, tab_regimes, tab_model = st.tabs([
    "Drought Map",
    "Drought Severity",
    "Latent Regimes",
    "Model Internals",
])

# Aliases so existing tab bodies don't have to be rewritten in bulk
tab_timeline = tab_regimes
tab_obs      = tab_regimes
tab_params   = tab_model
tab_duration = tab_model     # cut entirely below
tab_convergence = tab_model

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 0 — SPATIAL DROUGHT MAP (SPI-12 across the SJV grid)
# ═══════════════════════════════════════════════════════════════════════════════
with tab_map:
    st.markdown("### San Joaquin Valley Drought Map")
    st.caption(
        "Each cell shows where it ranks against the same calendar month over the 21-year record. "
        "A cell shaded **dark red** had its driest year for that month; **dark blue** had its wettest. "
        "Same percentile-rank method the U.S. Drought Monitor uses — distribution-free, "
        "directly interpretable, no σ scaling. Press ▶ to animate; drag the slider to scrub."
    )

    sp_lat, sp_lon, sp_dates, sp_grid = load_spatial()
    T = len(sp_dates)

    # ── USDM-style categorical bands tuned for a 21-year sample ──────────────
    # Five dry / Normal / five wet — fully symmetric.
    #   rank 0  (driest of 21):  pct ≈ 0.024  →  D4 Exceptional
    #   rank 1  (2nd driest):    pct ≈ 0.071  →  D3 Extreme
    #   rank 2:                  pct ≈ 0.119  →  D2 Severe
    #   ranks 3-4:               pct ≤ 0.214  →  D1 Moderate
    #   ranks 5-6:               pct ≤ 0.310  →  D0 Abnormally Dry
    USDM = [
        ("D4 Exceptional Drought", "#5C0000", 0.00, 0.05),
        ("D3 Extreme Drought",     "#A60000", 0.05, 0.10),
        ("D2 Severe Drought",      "#E66B00", 0.10, 0.15),
        ("D1 Moderate Drought",    "#FFA94D", 0.15, 0.24),
        ("D0 Abnormally Dry",      "#FFE099", 0.24, 0.33),
        ("Normal",                 "#F2F2F2", 0.33, 0.67),
        ("W0 Abnormally Wet",      "#B3D9E6", 0.67, 0.76),
        ("W1 Moderately Wet",      "#7FB8D9", 0.76, 0.81),
        ("W2 Very Wet",            "#4D94CC", 0.81, 0.86),
        ("W3 Extremely Wet",       "#1F5F99", 0.86, 0.95),
        ("W4 Exceptionally Wet",   "#0A3D66", 0.95, 1.01),
    ]

    # Build a colorscale with sharp transitions at category boundaries
    colorscale = []
    for label, color, lo, hi in USDM:
        colorscale.append([lo, color])
        colorscale.append([min(hi, 1.0), color])

    cat_grid = np.empty(sp_grid.shape, dtype=object)
    for label, _, lo, hi in USDM:
        cat_grid[(sp_grid >= lo) & (sp_grid < hi)] = label
    cat_grid[sp_grid >= 1.0] = USDM[-1][0]
    cat_grid[np.isnan(sp_grid)] = "—"

    AXIS_DARK = "#222"
    min_dt = sp_dates[0].to_pydatetime().date()
    max_dt = sp_dates[-1].to_pydatetime().date()
    month_labels = [d.strftime("%b %Y") for d in sp_dates]

    # ── Single source of truth for the displayed month ───────────────────────
    if "map_idx" not in st.session_state:
        st.session_state["map_idx"] = len(sp_dates) - 1
    if "map_playing" not in st.session_state:
        st.session_state["map_playing"] = False

    # If something queued a new index since last run, apply it BEFORE the
    # slider widget is created so the widget picks it up as its current value.
    if "_map_target" in st.session_state:
        st.session_state["map_idx"] = st.session_state.pop("_map_target")

    # ── Date picker (jumps to nearest month, stops animation) ────────────────
    def _on_date_change():
        d = st.session_state["_date_jump"]
        ts = pd.Timestamp(d)
        diffs = np.array([abs((sp_dates[i] - ts).total_seconds()) for i in range(T)])
        st.session_state["_map_target"] = int(np.argmin(diffs))
        st.session_state["map_playing"] = False

    st.date_input(
        "🔍 Jump to date",
        value=sp_dates[st.session_state["map_idx"]].to_pydatetime().date(),
        min_value=min_dt, max_value=max_dt,
        key="_date_jump", on_change=_on_date_change,
        help="Pick any date — the map snaps to the nearest available SPI-12 month.",
    )

    # ── Static heatmap for the current month ─────────────────────────────────
    idx = st.session_state["map_idx"]
    fig_map = go.Figure(go.Heatmap(
        z=sp_grid[idx], x=sp_lon, y=sp_lat,
        zmin=0, zmax=1, colorscale=colorscale,
        colorbar=dict(
            tickmode="array",
            tickvals=[0.025, 0.075, 0.125, 0.195, 0.285, 0.50,
                      0.715, 0.785, 0.835, 0.905, 0.975],
            ticktext=["D4", "D3", "D2", "D1", "D0", "Normal",
                      "W0", "W1", "W2", "W3", "W4"],
            tickfont=dict(color=AXIS_DARK, size=11),
            len=0.92, thickness=18, outlinewidth=0,
        ),
        customdata=cat_grid[idx],
        hovertemplate=(
            "lat %{y:.2f}°  lon %{x:.2f}°<br>"
            "Percentile rank: %{z:.0%}<br>"
            "<b>%{customdata}</b><extra></extra>"
        ),
    ))
    fig_map.update_layout(
        **{**PLOTLY_LAYOUT, "margin": dict(l=60, r=24, t=80, b=60)},
        height=580,
        title=dict(
            text=f"<b>{sp_dates[idx].strftime('%B %Y')}</b>",
            x=0.5, xanchor="center", y=0.97,
            font=dict(size=22, color=AXIS_DARK),
        ),
        xaxis=dict(
            title=dict(text="Longitude (°W)",
                       font=dict(color=AXIS_DARK, size=13), standoff=20),
            showgrid=False, showline=True, linecolor=AXIS_DARK, linewidth=1,
            tickfont=dict(color=AXIS_DARK, size=11), ticks="outside",
            tickcolor=AXIS_DARK,
            scaleanchor="y", scaleratio=1.0, tickformat=".1f", zeroline=False,
        ),
        yaxis=dict(
            title=dict(text="Latitude (°N)", font=dict(color=AXIS_DARK, size=13)),
            showgrid=False, showline=True, linecolor=AXIS_DARK, linewidth=1,
            tickfont=dict(color=AXIS_DARK, size=11), ticks="outside",
            tickcolor=AXIS_DARK,
            tickformat=".1f", zeroline=False,
        ),
    )
    # `displayModeBar: False` hides Plotly's modebar — no fullscreen button,
    # no broken state to enter.
    st.plotly_chart(fig_map, use_container_width=True, theme=None,
                    config={"displayModeBar": False})

    # ── Streamlit slider + Play/Pause (live drag, single-click, always works)
    st.select_slider(
        "Month",
        options=list(range(T)),
        format_func=lambda i: month_labels[i],
        key="map_idx",
    )

    def _on_play():  st.session_state["map_playing"] = True
    def _on_pause(): st.session_state["map_playing"] = False

    bc1, bc2, _ = st.columns([1, 1, 6])
    bc1.button("▶ Play",   on_click=_on_play,
               disabled=st.session_state["map_playing"],
               use_container_width=True)
    bc2.button("❚❚ Pause", on_click=_on_pause,
               disabled=not st.session_state["map_playing"],
               use_container_width=True)

    # ── Time-series stack: % of SJV in each USDM category over time ──────────
    st.markdown("#### % of SJV in each USDM category, over time")
    st.caption(
        "Stacked area — the dark-red band thickening through 2008–09 and 2014–15 is the "
        "core mega-drought signal. Look for thick blue bands in 2011, 2016–17, 2019."
    )

    pct_in_cat = {}
    for label, color, lo, hi in USDM:
        m = (sp_grid >= lo) & (sp_grid < hi)
        pct_in_cat[label] = 100 * np.nanmean(m, axis=(1, 2))   # (T,)

    fig_ts = go.Figure()
    # Stack from driest (bottom) to wettest (top) for a clean visual narrative
    for label, color, lo, hi in USDM:
        fig_ts.add_trace(go.Scatter(
            x=sp_dates, y=pct_in_cat[label],
            mode="lines",
            stackgroup="usdm",
            name=label,
            line=dict(width=0),
            fillcolor=color,
            hovertemplate=("<b>%{x|%b %Y}</b><br>" + label + ": %{y:.0f}% of SJV<extra></extra>"),
        ))
    fig_ts.update_layout(
        **PLOTLY_LAYOUT,
        height=380,
        yaxis=dict(title="% of SJV", range=[0, 100], ticksuffix="%",
                   showgrid=True, gridcolor="#f0f0f0"),
        xaxis=dict(showgrid=False, title=None),
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="left", x=0, itemsizing="constant",
                    font=dict(size=10)),
    )
    st.plotly_chart(fig_ts, use_container_width=True, theme=None)

    # ── Auto-advance when playing ────────────────────────────────────────────
    # Streamlit-driven: write to _map_target, rerun. Next run applies it to
    # the slider before any widget renders, so map_idx advances cleanly.
    if st.session_state["map_playing"]:
        if st.session_state["map_idx"] < T - 1:
            time.sleep(0.15)
            st.session_state["_map_target"] = st.session_state["map_idx"] + 1
            st.rerun()
        else:
            st.session_state["map_playing"] = False


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
        fig_strip = go.Figure()

        # Horizontal threshold bands behind the curve
        for lo, hi, color, label in DROUGHT_MONITOR_SCALE[1:]:
            fig_strip.add_hrect(
                y0=lo, y1=min(hi, 1.0),
                fillcolor=color, opacity=0.22,
                line_width=0, layer="below",
                annotation_text=label, annotation_position="top right",
                annotation_font_size=10,
            )

        # The main intensity line — solid, no transparency issues
        fig_strip.add_trace(go.Scatter(
            x=sev_df.index, y=sev_df["intensity"],
            mode="lines",
            line=dict(color="#222", width=2.2),
            fill="tozeroy",
            fillcolor="rgba(50,50,50,0.18)",
            name="P(drought | data)",
            hovertemplate="<b>%{x|%b %Y}</b><br>P(drought) = %{y:.0%}<extra></extra>",
        ))

        fig_strip.update_layout(
            **PLOTLY_LAYOUT,
            height=360,
            yaxis=dict(
                title="P(drought | data)",
                range=[0, 1], showgrid=False,
                tickformat=".0%",
            ),
            xaxis=dict(showgrid=False, title=None),
            showlegend=False,
        )

        # Annotate the famous droughts
        annotations = [
            ("2008-09-01", "2007–09 drought"),
            ("2014-09-01", "Mega-drought peak"),
            ("2020-09-01", "2020 drought"),
        ]
        for date_str, label in annotations:
            ts = pd.Timestamp(date_str)
            if sev_df.index[0] <= ts <= sev_df.index[-1]:
                yv = float(sev_df["intensity"].asof(ts))
                fig_strip.add_annotation(
                    x=ts, y=min(yv + 0.05, 0.96),
                    text=label, showarrow=True, arrowhead=2,
                    ay=-28, ax=0,
                    font=dict(size=11, color="#222"),
                    bgcolor="rgba(255,255,255,0.92)",
                    bordercolor="#666", borderwidth=0.5, borderpad=3,
                )

        st.plotly_chart(fig_strip, use_container_width=True, theme=None)

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
    st.markdown("### Regime Probability Over Time")
    st.caption(
        "Each row shows the posterior probability γₜ(k) = P(regime = k | data) for that regime. "
        "Solid colour = ~100% probability, faded = lower probability. "
        "Unlike a Viterbi MAP ribbon, this reveals **intensity** and **uncertainty** — "
        "in the lead-up to the 2014 mega-drought peak you can see Drought probability "
        "rising gradually from 2012 while Near-Normal fades."
    )

    # Stack order: wet/normal on top, drought/hot at bottom for visual narrative
    row_order = [0, 3, 2, 1]    # Pluvial → Near-Normal → Hot → Drought

    fig = go.Figure()
    for k in row_order:
        base = REGIME_COLORS[k]
        # Build a colorscale from transparent to the regime's full colour
        # so colour intensity directly encodes γ.
        colorscale = [
            [0.00, "rgba(255,255,255,0)"],
            [0.10, "rgba(255,255,255,0)"],   # below 10% reads as empty
            [1.00, base],
        ]
        fig.add_trace(go.Heatmap(
            z=[gamma[:, k]],
            x=dates,
            y=[REGIME_LABELS[k]],
            colorscale=colorscale,
            zmin=0, zmax=1,
            showscale=False,
            xgap=0, ygap=2,
            hovertemplate=(
                "<b>%{x|%b %Y}</b><br>"
                f"γ({REGIME_LABELS[k]}) = %{{z:.0%}}<extra></extra>"
            ),
        ))

    fig.update_layout(
        **{**PLOTLY_LAYOUT, "margin": dict(l=160, r=24, t=30, b=50)},
        height=280,
        xaxis=dict(
            showgrid=False, title=None,
            tickfont=dict(size=11, color="#1a1a1a"),
            ticks="outside", tickcolor="#1a1a1a",
            showline=True, linecolor="#1a1a1a",
        ),
        yaxis=dict(
            title=None,
            showgrid=False,
            tickfont=dict(size=13, color="#1a1a1a"),
            categoryorder="array",
            categoryarray=[REGIME_LABELS[k] for k in row_order],
        ),
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True, theme=None)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — OBSERVATIONS
# ═══════════════════════════════════════════════════════════════════════════════
with tab_obs:
    # ── scatter: precip vs ET coloured by regime (emission space) ────────────
    st.markdown("### Where each regime lives in observation space")
    st.caption(
        "Every month is a point at its (SPI-12, ETI-12) coordinates — coloured by the regime "
        "the HMM assigned it. The ✕ marks the learned Gaussian mean μₖ for each regime. "
        "Quadrants: **left half** = below-normal precipitation, **upper half** = above-normal ET. "
        "Drought (red) clusters bottom-left (dry + suppressed ET); Hot Regime (amber) sits upper-right; "
        "Pluvial (blue) is top-right; Near-Normal (grey) hugs the origin."
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
        height=460,
        xaxis=dict(
            title=dict(text="SPI-12 — precipitation anomaly (σ)",
                       font=dict(color="#1a1a1a", size=13)),
            tickfont=dict(color="#1a1a1a", size=11),
            showgrid=True, gridcolor="#e8e8e8", zeroline=False,
            showline=True, linecolor="#1a1a1a",
        ),
        yaxis=dict(
            title=dict(text="ETI-12 — ET anomaly (σ)",
                       font=dict(color="#1a1a1a", size=13)),
            tickfont=dict(color="#1a1a1a", size=11),
            showgrid=True, gridcolor="#e8e8e8", zeroline=False,
            showline=True, linecolor="#1a1a1a",
        ),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02,
            xanchor="left", x=0, itemsizing="constant",
            font=dict(color="#1a1a1a", size=12),
        ),
    )
    st.plotly_chart(fig4, use_container_width=True, theme=None)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — MODEL PARAMETERS
# ═══════════════════════════════════════════════════════════════════════════════
with tab_params:
    st.markdown("### Learned HMM Parameters")
    st.caption(
        "These are the parameters Baum-Welch converged on. "
        "**Transition matrix A** tells you how likely each regime is to switch to another next month. "
        "**Emission means μₖ** locate each regime in (SPI-12, ETI-12) space. "
        "**Stationary distribution π̄** is the long-run fraction of time the chain spends in each regime."
    )

    col_A, col_mus = st.columns([1, 1])
    DARK = "#1a1a1a"

    with col_A:
        st.markdown("#### Transition matrix  A")
        st.caption("Aⱼₖ = P(regime next month = k | regime this month = j). Rows sum to 1.")

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
            textfont=dict(size=14, color=DARK),
            hovertemplate="%{y} → %{x}<br>Prob: %{z:.4f}<extra></extra>",
            colorbar=dict(tickfont=dict(color=DARK), thickness=12, len=0.8),
            showscale=True,
        ))
        fig5.update_layout(
            **PLOTLY_LAYOUT,
            height=380,
            xaxis=dict(title=dict(text="To  →", font=dict(color=DARK, size=13)),
                       side="bottom", tickfont=dict(color=DARK, size=11)),
            yaxis=dict(title=dict(text="From", font=dict(color=DARK, size=13)),
                       autorange="reversed", tickfont=dict(color=DARK, size=11)),
        )
        st.plotly_chart(fig5, use_container_width=True, theme=None)

    with col_mus:
        st.markdown("#### Emission means and standard deviations")
        st.caption("μₖ in σ units (input is z-scored); σ is the regime's spread along each axis.")

        rows = []
        for k in range(4):
            rows.append({
                "Regime":         REGIME_LABELS[k],
                "μ SPI-12 (σ)":   f"{mus_phys[k,0]:+.2f}",
                "μ ETI-12 (σ)":   f"{mus_phys[k,1]:+.2f}",
                "σ precip":       f"{np.sqrt(sigmas[k,0,0]) * obs_std[0]:.3f}",
                "σ ET":           f"{np.sqrt(sigmas[k,1,1]) * obs_std[1]:.3f}",
            })
        st.dataframe(pd.DataFrame(rows).set_index("Regime"),
                     use_container_width=True)

        st.markdown("#### Stationary distribution  π̄")
        st.caption("Left eigenvector of A with eigenvalue 1 — the long-run fraction of months in each regime.")
        eigvals, eigvecs = np.linalg.eig(A.T)
        idx = np.argmin(np.abs(eigvals - 1.0))
        stat = np.abs(np.real(eigvecs[:, idx]))
        stat /= stat.sum()
        st.dataframe(pd.DataFrame({
            "Regime":        [REGIME_LABELS[k] for k in range(4)],
            "π̄  (long-run)": [f"{stat[k]*100:.1f}%" for k in range(4)],
        }).set_index("Regime"), use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 — BAUM-WELCH CONVERGENCE (part of Model Internals)
# ═══════════════════════════════════════════════════════════════════════════════
with tab_convergence:
    st.markdown("#### Baum-Welch convergence")
    st.caption(
        "Log-likelihood log P(X | θ) at each EM iteration. "
        "Should be monotonically increasing — that's the EM guarantee. "
        "Flattening means the parameters have settled at a local optimum."
    )

    if log_likelihoods is not None and len(log_likelihoods) > 0:
        fig8 = go.Figure(go.Scatter(
            x=list(range(1, len(log_likelihoods) + 1)),
            y=log_likelihoods,
            mode="lines+markers",
            marker=dict(size=5, color="#5B8DB8"),
            line=dict(color="#5B8DB8", width=2),
            hovertemplate="Iter %{x}: LL = %{y:.2f}<extra></extra>",
        ))
        DARK = "#1a1a1a"
        fig8.update_layout(
            **PLOTLY_LAYOUT,
            height=340,
            xaxis=dict(title=dict(text="Baum-Welch iteration", font=dict(color=DARK, size=13)),
                       tickfont=dict(color=DARK, size=11),
                       showgrid=True, gridcolor="#e8e8e8",
                       showline=True, linecolor=DARK),
            yaxis=dict(title=dict(text="log P(X | θ)", font=dict(color=DARK, size=13)),
                       tickfont=dict(color=DARK, size=11),
                       showgrid=True, gridcolor="#e8e8e8",
                       showline=True, linecolor=DARK),
        )
        st.plotly_chart(fig8, use_container_width=True, theme=None)

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
