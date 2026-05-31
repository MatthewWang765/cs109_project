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
  /* hide the top-right "Running…" status widget (sport-icon animation +
     stop button) so the animation loop doesn't trigger it every frame. */
  [data-testid="stStatusWidget"] { display: none !important; }
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
def load_spatial(_v=2):     # bump _v to invalidate cache after loader changes
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
    st.markdown("## Drought Regime Analysis")
    st.markdown(
        "<div style='color:#888; font-size:0.85rem; line-height:1.35; margin-top:-0.4rem;'>"
        "A 4-state Gaussian Hidden Markov Model that detects San Joaquin Valley drought "
        "from precipitation and evapotranspiration alone."
        "</div>",
        unsafe_allow_html=True,
    )
    st.divider()

    st.markdown("**Data**")
    st.markdown(
        "NASA GPM IMERG Late-Run precipitation  \n"
        "OpenET Monthly Ensemble evapotranspiration  \n"
        "<span style='color:#888'>241 months · Dec 2000 – Dec 2020</span>",
        unsafe_allow_html=True,
    )
    st.divider()

    if not outputs_exist:
        st.warning("No model outputs found. Run `python main.py` to generate them.")
        st.stop()

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
        "Each square is a 10 km patch of the SJV. Its colour shows how dry or wet that patch was, "
        "compared to all other Januaries (or Julys, etc.) in our 21-year record. "
        "**Dark red** = driest version of that month ever recorded; "
        "**dark blue** = wettest. Use the date picker, slider, or ▶ Play to step through time."
    )
    with st.expander("📐 What this is, statistically"):
        st.markdown(
            "- Each cell's value is its **percentile rank** of the 12-month rolling precipitation "
            "total against the 20 other instances of that same calendar month in the record.\n"
            "- A rank of 1/21 ≈ 4.8% → driest such month ever → **D4 Exceptional** "
            "(this is the same definition the U.S. Drought Monitor uses).\n"
            "- The method is **distribution-free** — no Gaussian assumption, no σ scaling, no "
            "deflation when the drought is part of the baseline. Each cell is compared only to "
            "itself across years.\n"
            "- Non-SJV pixels are masked NaN (white); a single black outline marks the "
            "valley boundary."
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

    # ── Precompute SJV outline ONCE — same mask every month ──────────────────
    # Emits any cell edge that borders a NaN neighbour. A single Scatter
    # polyline overlay draws them all on top of the heatmap → one black
    # outline around the perimeter of the SJV (no per-pixel grid).
    @st.cache_data(show_spinner=False)
    def _sjv_outline(lat_arr, lon_arr, mask_bytes):
        mask = np.frombuffer(mask_bytes, dtype=bool).reshape(len(lat_arr), len(lon_arr))
        Nlat, Nlon = mask.shape
        dh = float(lat_arr[1] - lat_arr[0]) if Nlat > 1 else 0.1
        dw = float(lon_arr[1] - lon_arr[0]) if Nlon > 1 else 0.1
        xs, ys = [], []
        for i in range(Nlat):
            for j in range(Nlon):
                if not mask[i, j]:
                    continue
                yt = float(lat_arr[i]) + dh / 2
                yb = float(lat_arr[i]) - dh / 2
                xl = float(lon_arr[j]) - dw / 2
                xr = float(lon_arr[j]) + dw / 2
                if i + 1 >= Nlat or not mask[i + 1, j]:
                    xs += [xl, xr, None]; ys += [yt, yt, None]
                if i - 1 < 0 or not mask[i - 1, j]:
                    xs += [xl, xr, None]; ys += [yb, yb, None]
                if j - 1 < 0 or not mask[i, j - 1]:
                    xs += [xl, xl, None]; ys += [yb, yt, None]
                if j + 1 >= Nlon or not mask[i, j + 1]:
                    xs += [xr, xr, None]; ys += [yb, yt, None]
        return xs, ys

    _sjv_mask = (~np.isnan(sp_grid[0])).astype(bool)
    outline_x, outline_y = _sjv_outline(sp_lat, sp_lon, _sjv_mask.tobytes())

    # ── Single source of truth for the displayed month ───────────────────────
    if "map_idx" not in st.session_state:
        st.session_state["map_idx"] = 0       # ← start at first month (Dec 2000)
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

    # ── Static heatmap for the current month + SJV outline overlay ──────────
    idx = st.session_state["map_idx"]
    fig_map = go.Figure()
    fig_map.add_trace(go.Heatmap(
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
    fig_map.add_trace(go.Scatter(
        x=outline_x, y=outline_y,
        mode="lines",
        line=dict(color="black", width=1.4),
        hoverinfo="skip", showlegend=False,
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

    # ── Drought Calendar: months × years heatmap ─────────────────────────────
    # Each cell = one month. Colour = % of the SJV in moderate-or-worse drought
    # (D1+) for that month. Read-at-a-glance: dark red columns are drought
    # years; pale columns are wet years; bands across rows show seasonality.
    st.markdown("#### Drought Calendar — % of SJV in moderate-or-worse drought")
    st.caption(
        "A 12 × 21 calendar of the entire record. Each tile is one month; "
        "**darker red = more of the valley was in drought that month**. "
        "Vertical red streaks are drought *years* (2007–09, 2014–15, 2020); "
        "pale-blue columns are wet years (2005, 2011, 2017). "
        "Horizontal patterns would show seasonality."
    )
    with st.expander("📐 What this is, statistically"):
        st.markdown(
            "- For every month and every SJV cell we have a percentile rank (from the map above).\n"
            "- A cell is **in drought** if its rank is below the 33rd percentile of its "
            "calendar-month history — that's the U.S. Drought Monitor's D0 threshold.\n"
            "- Tile colour = the fraction of the 357 SJV cells in drought that month, in [0%, 100%].\n"
            "- Pale blue at 0% = the entire valley was at or above normal; "
            "dark red at 100% = every SJV cell was in drought."
        )

    # Use the SJV mask (non-NaN cells in any frame) as the denominator —
    # ignoring the ~973 non-SJV cells fixes the "max 25%" display bug.
    sjv_mask_full = ~np.isnan(sp_grid[0])
    sjv_n = max(int(sjv_mask_full.sum()), 1)
    pct_in_drought = np.zeros(T, dtype=float)
    for t in range(T):
        in_drought = (sp_grid[t] < 0.33) & sjv_mask_full      # D0 or worse
        pct_in_drought[t] = 100 * int(in_drought.sum()) / sjv_n

    years = sorted({d.year for d in sp_dates})
    year_idx = {y: i for i, y in enumerate(years)}
    cal = np.full((12, len(years)), np.nan)
    for t, d in enumerate(sp_dates):
        cal[d.month - 1, year_idx[d.year]] = pct_in_drought[t]

    fig_cal = go.Figure(go.Heatmap(
        z=cal,
        x=years,
        y=["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"],
        zmin=0, zmax=100,
        colorscale=[
            [0.00, "#DCEEF6"],   # 0%   — pale blue (no drought, distinct from page bg)
            [0.20, "#FFF2B3"],   # 20%  — pale yellow
            [0.40, "#FFCB73"],   # 40%  — orange
            [0.60, "#E66B00"],   # 60%  — deep orange
            [0.80, "#A60000"],   # 80%  — red
            [1.00, "#5C0000"],   # 100% — dark red
        ],
        xgap=2, ygap=2,
        hovertemplate=(
            "<b>%{y} %{x}</b><br>"
            "%{z:.0f}% of SJV in drought (D1+)<extra></extra>"
        ),
        colorbar=dict(
            title=dict(text="% of SJV<br>in drought", side="right",
                       font=dict(color=AXIS_DARK, size=12)),
            tickvals=[0, 25, 50, 75, 100],
            ticktext=["0%", "25%", "50%", "75%", "100%"],
            tickfont=dict(color=AXIS_DARK, size=11),
            len=0.85, thickness=14, outlinewidth=0,
        ),
    ))
    fig_cal.update_layout(
        **{**PLOTLY_LAYOUT, "margin": dict(l=44, r=24, t=24, b=40)},
        height=380,
        xaxis=dict(title=None, type="category",
                   tickfont=dict(color=AXIS_DARK, size=11),
                   showgrid=False, showline=False),
        yaxis=dict(title=None, autorange="reversed",
                   tickfont=dict(color=AXIS_DARK, size=11),
                   showgrid=False, showline=False),
    )
    st.plotly_chart(fig_cal, use_container_width=True, theme=None)

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
    st.markdown("### Drought Intensity — how confident is the model that we're in drought?")
    st.caption(
        "A continuous score from 0% to 100% for every month, showing how strongly the model "
        "believes the SJV was in drought conditions. Higher = deeper drought. "
        "The coloured background bands match the U.S. Drought Monitor's "
        "D0 → D4 severity scale, so you can read off what 'category' each month was in. "
        "Annotated peaks line up with the real-world drought events."
    )
    with st.expander("📐 What this is, statistically"):
        st.markdown(
            "- The HMM produces a smoothed posterior probability for every regime at every month:  \n"
            "  $\\gamma_t(k) = \\Pr(Z_t = k \\mid X_{1:T})$  — computed by the forward–backward algorithm.\n"
            "- Unlike **Viterbi decoding** (which picks the single most likely state per month), the "
            "posterior preserves the full probability *distribution* across all four regimes. "
            "That's what gives this view a smooth gradient instead of a step function.\n"
            "- We summarise it into a drought-intensity score:  \n"
            "  $\\text{DI}(t) = \\gamma_t(\\text{Drought}) + 0.4 \\cdot \\gamma_t(\\text{Hot Regime})$  \n"
            "  — full weight on the persistent drought regime, partial weight on the hot/warming "
            "regime since it's also dry but with different physical drivers.\n"
            "- The USDM-style D0–D4 bands map this score to standard drought categories so the "
            "result is directly comparable to published drought reports."
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
        "The HMM discovered four climate 'archetypes' — Pluvial, Near-Normal, Hot, and Drought. "
        "Each panel below tracks one archetype over time: the line shows how strongly the model "
        "believes the SJV was in *that* archetype each month. 100% = the model is certain; "
        "values in between mean it's split between this archetype and a neighbour."
    )
    st.info(
        "💡 The 'Drought' panel only lights up when conditions are *deeply* anomalous on both "
        "precipitation and ET. The famous 2012–2017 California drought was a multi-year stretch "
        "of varying severity that peaked sharply in 2014–2015; 2012–13 were dry but not yet "
        "at the Drought archetype's centroid, so they classify as Near-Normal here. "
        "For partial-credit drought during the buildup years, see the **Drought Severity** tab."
    )
    with st.expander("📐 What this is, statistically"):
        st.markdown(
            "- The HMM has 4 hidden states (regimes). For every month it computes the smoothed "
            "posterior probability of being in each:  \n"
            "  $\\gamma_t(k) = \\Pr(Z_t = k \\mid X_{1:T})$\n"
            "- Computed by the **forward–backward algorithm** — uses both past *and* future "
            "observations to refine the probability at each time step.\n"
            "- For every month, the four panels sum to 100% (the regime must be exactly one of the four).\n"
            "- Compare to **Viterbi MAP decoding**, which would pick the single argmax regime per "
            "month — fine, but loses uncertainty. The posterior shows when the model was on the "
            "fence between two regimes (look for plateaus around 30–70% instead of 0/100% spikes)."
        )

    # Small-multiples: one row per regime — every row reads independently
    # as a normal time series, far more interpretable than a colour-density
    # heatmap. Order from wet → dry top-to-bottom for narrative flow.
    row_order = [0, 3, 2, 1]   # Pluvial → Near-Normal → Hot → Drought

    def _hex_to_rgba(hex_color, alpha):
        h = hex_color.lstrip("#")
        r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
        return f"rgba({r},{g},{b},{alpha})"

    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.035,
        subplot_titles=[REGIME_LABELS[k] for k in row_order],
    )
    for row_i, k in enumerate(row_order, start=1):
        color = REGIME_COLORS[k]
        fig.add_trace(go.Scatter(
            x=dates, y=gamma[:, k],
            mode="lines",
            line=dict(color=color, width=1.6),
            fill="tozeroy",
            fillcolor=_hex_to_rgba(color, 0.30),
            hovertemplate=(
                f"<b>{REGIME_LABELS[k]}</b><br>"
                "%{x|%b %Y}<br>γ = %{y:.0%}<extra></extra>"
            ),
            showlegend=False,
        ), row=row_i, col=1)
        # Subtle reference lines at 50% and 100%
        fig.add_hline(y=0.5, line_dash="dot", line_color="#cccccc", line_width=1, row=row_i, col=1)

    DARK = "#1a1a1a"
    fig.update_layout(
        **{**PLOTLY_LAYOUT, "margin": dict(l=48, r=24, t=40, b=40)},
        height=520,
        showlegend=False,
    )
    fig.update_xaxes(showgrid=False, showline=True, linecolor=DARK,
                     tickfont=dict(color=DARK, size=10),
                     ticks="outside", tickcolor=DARK)
    fig.update_yaxes(range=[0, 1.02], tickformat=".0%",
                     showgrid=True, gridcolor="#eaeaea",
                     showline=True, linecolor=DARK,
                     tickfont=dict(color=DARK, size=10),
                     tickvals=[0, 0.5, 1.0])
    # Style the subplot titles (regime labels) to be left-aligned, bold, in regime colour
    for i, k in enumerate(row_order):
        fig.layout.annotations[i].update(
            font=dict(size=13, color=REGIME_COLORS[k]),
            x=0.0, xanchor="left",
        )
    st.plotly_chart(fig, use_container_width=True, theme=None)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — OBSERVATIONS
# ═══════════════════════════════════════════════════════════════════════════════
with tab_obs:
    # ── Emission space + Gaussian ellipses ───────────────────────────────────
    st.markdown("### Where each regime lives — observations + learned ellipses")
    st.caption(
        "Each dot is one month, positioned by **how anomalous its precipitation was** (horizontal) "
        "and **how anomalous its ET was** (vertical). The dashed lines are 'normal'. "
        "Dots are coloured by the regime the model assigned them. "
        "Around each regime's centre (◆) you see two ellipses — the model thinks "
        "**~40% of that regime's months land inside the inner ellipse**, **~86% inside the outer one**.\n\n"
        "Read the quadrants: left = drier than normal, right = wetter; top = higher ET, bottom = lower ET. "
        "**Drought** (red) clusters bottom-left, **Hot Regime** (amber) top-right, "
        "**Pluvial** (blue) top-right with high precip, **Near-Normal** (grey) hugs the origin."
    )
    with st.expander("📐 What this is, statistically"):
        st.markdown(
            "- The HMM uses a **multivariate Gaussian emission model**: given regime $k$, the "
            "observation is distributed as  \n"
            "  $X_t \\mid Z_t = k \\;\\sim\\; \\mathcal{N}(\\boldsymbol{\\mu}_k, \\boldsymbol{\\Sigma}_k)$\n"
            "- Each ellipse is a **level set** of that Gaussian's probability density — the locus "
            "of points satisfying  $(x - \\mu_k)^\\top \\Sigma_k^{-1} (x - \\mu_k) = c^2$.\n"
            "- For $c = 1$ the ellipse encloses ~39% of the probability mass (the 2-D analogue of "
            "'1σ'); for $c = 2$ it encloses ~86%. (Note: these are *not* the familiar 68% / 95% — "
            "those are the 1-D values; in 2D the same $c$ encloses less mass.)\n"
            "- Ellipses are computed by **eigendecomposition** of $\\Sigma_k$: the eigenvectors give "
            "the principal axes (rotation), the eigenvalues give the variance along each axis "
            "(stretching).\n"
            "- The orientation/tilt of an ellipse reveals **covariance** — a tilted ellipse means "
            "precip and ET are correlated within that regime; an axis-aligned ellipse means they "
            "vary independently."
        )

    def _ellipse(mu, cov, k_sigma, n=120):
        """(x,y) coords of the k_sigma confidence ellipse for 𝒩(mu, cov)."""
        eigvals, eigvecs = np.linalg.eigh(cov)
        eigvals = np.clip(eigvals, 0, None)
        theta = np.linspace(0, 2 * np.pi, n)
        circle = np.stack([np.cos(theta), np.sin(theta)], axis=0)
        scaled = np.diag(np.sqrt(eigvals) * k_sigma) @ circle
        rot = eigvecs @ scaled
        return mu[0] + rot[0], mu[1] + rot[1]

    def _hex_rgba(hex_color, alpha):
        h = hex_color.lstrip("#")
        r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
        return f"rgba({r},{g},{b},{alpha})"

    # Σ in observation space  (z-score units, same as obs_df columns)
    # mus_phys is already in obs units; rescale Σ similarly:
    #   Σ_obs[k] = D · Σ_norm[k] · D,  where D = diag(obs_std)
    D = np.diag(obs_std)
    sigmas_obs = np.array([D @ sigmas[k] @ D for k in range(4)])

    fig4 = go.Figure()

    # 1. Data points
    for k in range(4):
        mask = obs_df["state"] == k
        fig4.add_trace(go.Scatter(
            x=obs_df.loc[mask, "precip_anom"],
            y=obs_df.loc[mask, "et_anom"],
            mode="markers",
            name=REGIME_LABELS[k],
            marker=dict(color=REGIME_COLORS[k], size=7,
                        opacity=0.65,
                        line=dict(color="white", width=0.5)),
            hovertemplate=(
                "<b>" + REGIME_LABELS[k] + "</b><br>"
                "SPI-12: %{x:+.2f}σ<br>ETI-12: %{y:+.2f}σ<extra></extra>"
            ),
        ))

    # 2. Confidence ellipses (2σ first so 1σ draws on top)
    for k in range(4):
        mu = mus_phys[k]
        for k_sig, alpha_fill, alpha_line in [(2.0, 0.06, 0.55),
                                              (1.0, 0.14, 0.95)]:
            ex, ey = _ellipse(mu, sigmas_obs[k], k_sig)
            fig4.add_trace(go.Scatter(
                x=ex, y=ey,
                mode="lines",
                line=dict(color=REGIME_COLORS[k], width=1.6),
                opacity=alpha_line,
                fill="toself",
                fillcolor=_hex_rgba(REGIME_COLORS[k], alpha_fill),
                showlegend=False, hoverinfo="skip",
            ))

    # 3. Means
    for k in range(4):
        mu = mus_phys[k]
        fig4.add_trace(go.Scatter(
            x=[mu[0]], y=[mu[1]],
            mode="markers+text",
            marker=dict(symbol="diamond", color=REGIME_COLORS[k],
                        size=18, line=dict(color="white", width=2.5)),
            text=[f"  μ — {REGIME_LABELS[k]}"],
            textposition="middle right",
            textfont=dict(size=12, color=REGIME_COLORS[k]),
            showlegend=False,
            hovertemplate=(
                f"<b>μ — {REGIME_LABELS[k]}</b><br>"
                f"SPI-12: {mu[0]:+.2f}σ<br>ETI-12: {mu[1]:+.2f}σ<extra></extra>"
            ),
        ))

    fig4.add_hline(y=0, line_dash="dash", line_color="#bbb", line_width=1)
    fig4.add_vline(x=0, line_dash="dash", line_color="#bbb", line_width=1)

    fig4.update_layout(
        **PLOTLY_LAYOUT,
        height=560,
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
        "The numbers the model converged on after training. Three sections:\n\n"
        "1. **Transition matrix** — how the model expects the SJV to *switch* between regimes month to month.\n"
        "2. **Emission means** — what 'typical' precipitation and ET look like inside each regime.\n"
        "3. **Stationary distribution** — if the SJV ran forever under this model, what fraction "
        "of months would it spend in each regime?"
    )
    with st.expander("📐 What this is, statistically"):
        st.markdown(
            "An HMM is fully specified by three pieces:\n\n"
            "- **Initial distribution**  $\\pi$  — probabilities of starting in each regime.\n"
            "- **Transition matrix**  $A_{jk} = \\Pr(Z_t = k \\mid Z_{t-1} = j)$ — a $K \\times K$ "
            "stochastic matrix; each row sums to 1.\n"
            "- **Emission parameters**  $\\boldsymbol{\\mu}_k, \\boldsymbol{\\Sigma}_k$ — the mean and "
            "covariance of the multivariate Gaussian that generates observations when the chain is "
            "in regime $k$.\n\n"
            "All three are learned from the data via **Baum-Welch** (the EM algorithm for HMMs), "
            "starting from k-means initialisation of the means and an identity-matrix covariance.\n\n"
            "The **stationary distribution** $\\bar{\\pi}$ is the unique probability vector "
            "satisfying $\\bar{\\pi}^\\top A = \\bar{\\pi}^\\top$ — i.e., the left eigenvector of "
            "$A$ with eigenvalue 1. It tells you the long-run fraction of time the Markov chain "
            "spends in each regime."
        )

    col_A, col_mus = st.columns([1, 1])
    DARK = "#1a1a1a"

    with col_A:
        st.markdown("#### Transition matrix")
        st.caption(
            "Read row → column: 'if I'm in regime X this month, what's the chance I'm in "
            "regime Y next month?' Diagonal values are *self-transitions* — how persistent each "
            "regime is. High diagonals (e.g. 0.9) mean the regime tends to last several months."
        )

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
        st.caption(
            "For each regime: where its 'centre' sits in precipitation/ET space (μ) and how "
            "wide a spread of months it covers (σ). Values are in standard-deviation units, so "
            "+1.0 = one σ above the long-term mean, −1.0 = one σ below."
        )

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

        st.markdown("#### Stationary distribution")
        st.caption(
            "If the SJV ran forever under this model, what fraction of months would it spend "
            "in each regime? Mathematically: the long-run balance of the Markov chain."
        )
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
        "How well the model fit improved with each training iteration. "
        "The line should always go up (better fit) and flatten out — that's the "
        "model finding its best possible parameters and stopping."
    )
    with st.expander("📐 What this is, statistically"):
        st.markdown(
            "- We train the HMM by maximising the **log-likelihood** of the observed data:  \n"
            "  $\\log \\Pr(X_{1:T} \\mid \\theta)$  where $\\theta = (\\pi, A, \\mu_k, \\Sigma_k)$.\n"
            "- This is done with **Baum-Welch** — the Expectation-Maximisation algorithm specialised "
            "for HMMs. Each iteration alternates:\n"
            "  - **E-step**: compute posterior probabilities of being in each regime at each time, "
            "given current parameters (the forward-backward algorithm).\n"
            "  - **M-step**: update $\\theta$ to maximise the expected complete-data log-likelihood "
            "under those posteriors (closed-form updates for $\\pi$, $A$, $\\mu_k$, $\\Sigma_k$).\n"
            "- A core theorem of EM: **each iteration is guaranteed to increase the log-likelihood "
            "(or leave it unchanged)** — that's why the curve is monotone.\n"
            "- We stop when the iteration-over-iteration change drops below $10^{-4}$, i.e. when "
            "the optimisation has converged to a (local) optimum."
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
