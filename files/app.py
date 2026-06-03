"""
app.py — SJV Drought Analysis Dashboard (CS109 Spring 2026 Challenge Project)

Five tabs:
  1. Drought Map       — percentile-rank spatial map (unchanged)
  2. Modeling Precip   — Continuous RVs + MLE (Gamma vs Normal)
  3. Predicting Drought — Logistic Regression from scratch
  4. Uncertainty       — Bootstrap + CLT + Bayesian Beta-Binomial
  5. Information Content — Shannon entropy + KL divergence + mutual information

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
from plotly.subplots import make_subplots
from scipy import stats, special

# ── page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SJV Drought Analysis",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── design tokens ─────────────────────────────────────────────────────────────
PLOTLY_LAYOUT = dict(
    font_family="Inter, system-ui, sans-serif",
    font_color="#1a1a1a",
    paper_bgcolor="white",
    plot_bgcolor="white",
    margin=dict(l=48, r=24, t=40, b=48),
)
DARK = "#1a1a1a"

# U.S. Drought Monitor palette
USDM_COLORS = {
    "D4": "#5C0000", "D3": "#A60000", "D2": "#E66B00",
    "D1": "#FFA94D", "D0": "#FFE099", "Normal": "#F2F2F2",
    "W0": "#B3D9E6", "W1": "#7FB8D9", "W2": "#4D94CC",
    "W3": "#1F5F99", "W4": "#0A3D66",
}

# ── minimal CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
  .block-container { padding-top: 2rem; padding-bottom: 2rem; }
  [data-testid="stMetricLabel"] { font-size: 0.75rem; color: #666; }
  h3 { font-weight: 600; letter-spacing: -0.02em; margin-top: 0; }
  [data-testid="stStatusWidget"] { display: none !important; }
</style>
""", unsafe_allow_html=True)


# ── data loaders ──────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Computing per-cell drought percentiles…")
def load_spatial(_v=2):
    from data_loader import load_spatial_percentile
    return load_spatial_percentile("data/gpm_sjv_subset.nc", window=12)


@st.cache_data(show_spinner="Loading SPI-12 anomalies…")
def load_anomalies():
    from data_loader import load_netcdf
    X, dates = load_netcdf("data/gpm_sjv_subset.nc", "data/openet_sjv_subset.nc")
    return X, dates


@st.cache_data(show_spinner="Loading raw monthly totals…")
def load_raw():
    from data_loader import load_raw_monthly
    p, e, dates = load_raw_monthly("data/gpm_sjv_subset.nc", "data/openet_sjv_subset.nc")
    return p, e, dates


# ── sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## San Joaquin Valley Drought Analysis")
    st.markdown(
        "<div style='color:#888; font-size:0.85rem; line-height:1.35; margin-top:-0.4rem;'>"
        "A probabilistic study of San Joaquin Valley drought from 2000 to 2020 using "
        "precipitation alone."
        "</div>",
        unsafe_allow_html=True,
    )
    st.divider()
    st.markdown("**Data**")
    st.markdown(
        "NASA GPM IMERG Late-Run precipitation  \n"
        "<span style='color:#888'>252 months · Jan 2000 – Dec 2020</span>",
        unsafe_allow_html=True,
    )
    st.divider()
    st.markdown("**CS109 methods**")
    st.markdown(
        "- Gamma / Normal MLE\n"
        "- Logistic regression\n"
        "- Bootstrap + CLT\n"
        "- Beta-Binomial Bayesian\n"
        "- Shannon entropy & KL divergence",
    )

# ── load shared data ──────────────────────────────────────────────────────────
X_anom, anom_dates = load_anomalies()
precip_raw, _et_raw, raw_dates = load_raw()

spi12 = X_anom[:, 0].astype(np.float64)
DROUGHT_THRESH = -0.5                   # SPI-12 < -0.5 → drought month

# ── bootstrap CI for drought rate (used in quick stats) ──────────────────────
@st.cache_data(show_spinner=False)
def _bootstrap_drought_rate(spi_bytes, n_boot=10_000, seed=42):
    spi = np.frombuffer(spi_bytes, dtype=np.float64)
    y = (spi < DROUGHT_THRESH).astype(float)
    rng = np.random.default_rng(seed)
    boot = rng.choice(y, size=(n_boot, len(y)), replace=True).mean(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return float(y.mean()), lo, hi

drought_rate, dr_lo, dr_hi = _bootstrap_drought_rate(spi12.tobytes())

# ── title + quick stats ───────────────────────────────────────────────────────
st.markdown("## San Joaquin Valley Drought Analysis  \n###### 21 years of rainfall data · 2000–2020")
st.markdown("##### Quick Statistics")
st.markdown(
    "<div style='color:#888; font-size:0.85rem; margin-top:-0.4rem; margin-bottom:0.6rem;'>"
    "Key numbers from 241 months of San Joaquin Valley rainfall data."
    "</div>",
    unsafe_allow_html=True,
)

annual_spi = pd.Series(spi12, index=anom_dates).resample("YS").mean()
col1, col2, col3, col4 = st.columns(4)
col1.markdown(
    f"**Driest year**  \n"
    f"<span style='font-size:1.8rem;font-weight:600;'>{int(annual_spi.idxmin().year)}</span>  \n"
    f"<span style='color:#888;font-size:0.85rem;'>On average, each month in 2014 had the lowest amount of rainfall</span>",
    unsafe_allow_html=True,
)
col2.markdown(
    f"**Wettest year**  \n"
    f"<span style='font-size:1.8rem;font-weight:600;'>{int(annual_spi.idxmax().year)}</span>  \n"
    f"<span style='color:#888;font-size:0.85rem;'>On average, each month in 2017 had the highest amount of rainfall</span>",
    unsafe_allow_html=True,
)
col3.markdown(
    f"**Months in drought**  \n"
    f"<span style='font-size:1.8rem;font-weight:600;'>{drought_rate*100:.0f}%</span>  \n"
    f"<span style='color:#888;font-size:0.85rem;'>95% Confidence Interval: {dr_lo*100:.0f}% to {dr_hi*100:.0f}%</span>",
    unsafe_allow_html=True,
)
col4.markdown(
    f"**Sample size**  \n"
    f"<span style='font-size:1.8rem;font-weight:600;'>241 months</span>  \n"
    f"<span style='color:#888;font-size:0.85rem;'>One observation per month</span>",
    unsafe_allow_html=True,
)

# ── Tab layout ────────────────────────────────────────────────────────────────
tab_map, tab_precip, tab_logit, tab_boot, tab_info = st.tabs([
    "Drought Map",
    "Measuring Drought",
    "Predicting Drought",
    "Uncertainty",
    "Predictability",
])


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — SPATIAL DROUGHT MAP  (preserved verbatim)
# ═══════════════════════════════════════════════════════════════════════════════
with tab_map:
    st.markdown("### San Joaquin Valley Drought Map")
    st.caption(
        "Each square is a 10 km patch of the San Joaquin Valley. The colour shows how dry or wet "
        "that patch was compared to all other Januaries (or Julys, Decembers, etc.) in our 21-year record. "
        "**Dark red** is the driest version of that month on record, and "
        "**dark blue** is the wettest. "
        "The plots below show which years were drought years at a glance. "
        "**See the Measuring Drought tab for how drought severity is defined.**"
    )
    with st.expander("Statistical Details"):
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

    if "map_idx" not in st.session_state:
        st.session_state["map_idx"] = 0
    if "map_playing" not in st.session_state:
        st.session_state["map_playing"] = False

    if "_map_target" in st.session_state:
        st.session_state["map_idx"] = st.session_state.pop("_map_target")

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
        help="Pick any date. The map will jump to the nearest available month.",
    )

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
            title=dict(text="Longitude (°W)", font=dict(color=AXIS_DARK, size=13), standoff=20),
            showgrid=False, showline=True, linecolor=AXIS_DARK, linewidth=1,
            tickfont=dict(color=AXIS_DARK, size=11), ticks="outside", tickcolor=AXIS_DARK,
            scaleanchor="y", scaleratio=1.0, tickformat=".1f", zeroline=False,
        ),
        yaxis=dict(
            title=dict(text="Latitude (°N)", font=dict(color=AXIS_DARK, size=13)),
            showgrid=False, showline=True, linecolor=AXIS_DARK, linewidth=1,
            tickfont=dict(color=AXIS_DARK, size=11), ticks="outside", tickcolor=AXIS_DARK,
            tickformat=".1f", zeroline=False,
        ),
    )
    st.plotly_chart(fig_map, use_container_width=True, theme=None, config={"displayModeBar": False})

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
               disabled=st.session_state["map_playing"], use_container_width=True)
    bc2.button("❚❚ Pause", on_click=_on_pause,
               disabled=not st.session_state["map_playing"], use_container_width=True)

    st.markdown("#### Drought Calendar — % of SJV in moderate-or-worse drought")
    st.caption(
        "A 12 × 21 calendar of the entire record. Each tile is one month; "
        "**darker red = more of the valley was in drought that month**. "
        "Vertical red streaks are drought years (2007–09, 2014–15, 2020); "
        "pale-blue columns are wet years (2005, 2011, 2017). "
        "Horizontal patterns show seasonality."
    )
    with st.expander("Statistical details"):
        st.markdown(
            "- For every month and every SJV cell we have a percentile rank (from the map above).\n"
            "- A cell is **in drought** if its rank is below the 33rd percentile of its "
            "calendar-month history — that's the U.S. Drought Monitor's D0 threshold.\n"
            "- Tile colour = the fraction of the 357 SJV cells in drought that month, in [0%, 100%].\n"
            "- Pale blue at 0% = the entire valley was at or above normal; "
            "dark red at 100% = every SJV cell was in drought."
        )

    sjv_mask_full = ~np.isnan(sp_grid[0])
    sjv_n = max(int(sjv_mask_full.sum()), 1)
    pct_in_drought = np.zeros(T, dtype=float)
    for t in range(T):
        in_drought = (sp_grid[t] < 0.33) & sjv_mask_full
        pct_in_drought[t] = 100 * int(in_drought.sum()) / sjv_n

    years = sorted({d.year for d in sp_dates})
    year_idx = {y: i for i, y in enumerate(years)}
    cal = np.full((12, len(years)), np.nan)
    for t, d in enumerate(sp_dates):
        cal[d.month - 1, year_idx[d.year]] = pct_in_drought[t]

    fig_cal = go.Figure(go.Heatmap(
        z=cal, x=years,
        y=["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"],
        zmin=0, zmax=100,
        colorscale=[
            [0.00, "#DCEEF6"], [0.20, "#FFF2B3"], [0.40, "#FFCB73"],
            [0.60, "#E66B00"], [0.80, "#A60000"], [1.00, "#5C0000"],
        ],
        xgap=2, ygap=2,
        hovertemplate="<b>%{y} %{x}</b><br>%{z:.0f}% of SJV in drought (D1+)<extra></extra>",
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
                   tickfont=dict(color=AXIS_DARK, size=11), showgrid=False, showline=False),
        yaxis=dict(title=None, autorange="reversed",
                   tickfont=dict(color=AXIS_DARK, size=11), showgrid=False, showline=False),
    )
    st.plotly_chart(fig_cal, use_container_width=True, theme=None)

    if st.session_state["map_playing"]:
        if st.session_state["map_idx"] < T - 1:
            time.sleep(0.15)
            st.session_state["_map_target"] = st.session_state["map_idx"] + 1
            st.rerun()
        else:
            st.session_state["map_playing"] = False


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — MODELING PRECIPITATION  (Continuous RVs + MLE)
# ═══════════════════════════════════════════════════════════════════════════════
with tab_precip:
    st.markdown("### Measuring Drought")
    st.caption(
        "Before we can predict drought, we need a way to measure it. "
        "The standard tool is the **Standardised Precipitation Index (SPI-12)**: "
        "compare this year's 12-month rainfall total to historical averages for the same time of year, "
        "and express the difference as a single number. "
        "Zero means exactly average. Below -0.5 means drier than normal, which we call drought. "
        "This score is used as the definition of drought in every tab that follows. "
        "To compute it, we first need to fit a statistical curve to historical rainfall. "
        "The chart below asks: which curve fits best?"
    )
    with st.expander("Statistical details"):
        st.markdown(r"""
**Gamma distribution**

$$f(x;\, k,\theta) = \frac{x^{k-1}\,e^{-x/\theta}}{\theta^k\,\Gamma(k)}, \quad x > 0$$

Parameters: shape $k > 0$, scale $\theta > 0$.

**MLE objective**

$$\hat\theta_{\text{MLE}} = \underset{\theta}{\arg\max}\; \sum_i \log f(x_i;\theta)$$

- **Normal MLE**: closed form — $\hat\mu = \bar x$, $\hat\sigma^2 = \tfrac{1}{n}\sum(x_i-\bar x)^2$.
- **Gamma MLE**: no closed form for $k$. `scipy.stats.gamma.fit` uses Newton-Raphson on the digamma function.

**Why this matters**: the U.S. Drought Monitor's Standardized Precipitation Index (SPI) is defined as:
$$\text{SPI}(x) = \Phi^{-1}(F_{\text{Gamma}}(x))$$
i.e. fit a Gamma to the data, transform to a standard Normal via the CDF.
""")

    MONTH_NAMES = ["January","February","March","April","May","June",
                   "July","August","September","October","November","December"]
    selected_month_name = st.selectbox(
        "Calendar month",
        MONTH_NAMES,
        index=0,
        key="mle_month",
    )
    sel_m = MONTH_NAMES.index(selected_month_name) + 1

    # Pull raw monthly precip for the selected calendar month
    precip_series = precip_raw.copy()
    month_data = precip_series[precip_series.index.month == sel_m].values
    month_data = month_data[month_data > 0]   # Gamma requires x > 0; drop exact zeros

    n = len(month_data)
    x_lo, x_hi = 0.0, month_data.max() * 1.35
    x_fit = np.linspace(max(x_lo, 1e-3), x_hi, 400)

    # Gamma MLE
    gamma_fit = stats.gamma.fit(month_data, floc=0)   # fix location=0
    k_hat, loc_hat, theta_hat = gamma_fit
    gamma_ll = float(stats.gamma.logpdf(month_data, *gamma_fit).sum())

    # Normal MLE (closed form)
    mu_hat  = float(month_data.mean())
    sig_hat = float(month_data.std(ddof=0))
    norm_ll = float(stats.norm.logpdf(month_data, mu_hat, sig_hat).sum())

    gamma_pdf = stats.gamma.pdf(x_fit, k_hat, loc_hat, theta_hat)
    norm_pdf  = stats.norm.pdf(x_fit, mu_hat, sig_hat)

    # ── Histogram + fitted PDFs ──────────────────────────────────────────────
    fig_mle = go.Figure()
    fig_mle.add_trace(go.Histogram(
        x=month_data, nbinsx=10,
        histnorm="probability density",
        marker=dict(color="#B3D9E6", line=dict(color="white", width=1)),
        name=f"{selected_month_name} data (n={n})",
        hovertemplate="bin centre %{x:.0f} mm<br>density %{y:.4f}<extra></extra>",
    ))
    fig_mle.add_trace(go.Scatter(
        x=x_fit, y=gamma_pdf,
        mode="lines",
        line=dict(color="#A60000", width=2.5),
        name=f"Gamma MLE  (k={k_hat:.2f}, θ={theta_hat:.1f})",
    ))
    fig_mle.add_trace(go.Scatter(
        x=x_fit, y=norm_pdf,
        mode="lines",
        line=dict(color="#888", width=2, dash="dash"),
        name=f"Normal MLE  (μ={mu_hat:.1f}, σ={sig_hat:.1f})",
    ))
    fig_mle.update_layout(
        **PLOTLY_LAYOUT, height=400,
        xaxis=dict(title="Monthly precipitation (mm)",
                   showgrid=False, showline=True, linecolor=DARK,
                   tickfont=dict(color=DARK)),
        yaxis=dict(title="Probability density",
                   showgrid=True, gridcolor="#eee",
                   showline=True, linecolor=DARK,
                   tickfont=dict(color=DARK)),
        legend=dict(orientation="h", yanchor="bottom", y=1.01, x=0),
    )
    st.plotly_chart(fig_mle, use_container_width=True, theme=None)

    # ── Parameter summary cards ──────────────────────────────────────────────
    c1, c2, c3 = st.columns(3)
    c1.metric("Gamma fit score", f"{gamma_ll:.1f}")
    c2.metric("Normal fit score", f"{norm_ll:.1f}",
              delta=f"{norm_ll - gamma_ll:.1f} vs Gamma", delta_color="inverse")
    c3.metric("Gamma wins?", "Yes ✓" if gamma_ll > norm_ll else "No")

    st.caption(
        f"Higher score = better fit to the data. "
        f"The Gamma scores {gamma_ll:.1f} vs the Normal's {norm_ll:.1f} for {selected_month_name}. "
        "The Normal curve also has another problem: it allows negative rainfall, which is impossible."
    )

    # ── Empirical CDF vs theoretical CDFs ────────────────────────────────────
    st.markdown("#### Does the Gamma curve actually match the data?")
    st.caption(
        "The grey steps show the actual data: for any given rainfall amount, what fraction of "
        "historical months were below it? The red and dashed lines are what the Gamma and Normal "
        "curves predict. A good fit means the line closely follows the steps. "
        "Notice the Normal curve extends past zero to the left, where rain cannot go."
    )

    x_sorted = np.sort(month_data)
    ecdf_y   = np.arange(1, len(x_sorted) + 1) / len(x_sorted)

    fig_cdf = go.Figure()
    # Empirical CDF as a step function
    fig_cdf.add_trace(go.Scatter(
        x=np.repeat(x_sorted, 2)[1:],
        y=np.repeat(ecdf_y,   2)[:-1],
        mode="lines",
        line=dict(color="#aaa", width=2, shape="hv"),
        name="Empirical CDF",
        hovertemplate="x=%{x:.0f} mm<br>F=%{y:.2f}<extra></extra>",
    ))
    # Gamma CDF
    fig_cdf.add_trace(go.Scatter(
        x=x_fit, y=stats.gamma.cdf(x_fit, k_hat, loc_hat, theta_hat),
        mode="lines", line=dict(color="#A60000", width=2.5),
        name=f"Gamma MLE  (k={k_hat:.2f}, θ={theta_hat:.1f})",
    ))
    # Normal CDF (extends below zero)
    x_fit_wide = np.linspace(min(x_fit[0], -20), x_hi, 400)
    fig_cdf.add_trace(go.Scatter(
        x=x_fit_wide, y=stats.norm.cdf(x_fit_wide, mu_hat, sig_hat),
        mode="lines", line=dict(color="#888", width=2, dash="dash"),
        name=f"Normal MLE  (μ={mu_hat:.1f})",
    ))
    # Mark zero
    fig_cdf.add_vline(x=0, line_color="#bbb", line_width=1,
                      annotation_text="x=0 (no rain possible below here)",
                      annotation_position="top right",
                      annotation_font=dict(size=10, color="#999"))
    fig_cdf.update_layout(
        **PLOTLY_LAYOUT, height=340,
        xaxis=dict(title="Monthly precipitation (mm)",
                   showgrid=False, showline=True, linecolor=DARK,
                   tickfont=dict(color=DARK)),
        yaxis=dict(title="Cumulative probability", range=[0, 1.05],
                   showgrid=True, gridcolor="#eee",
                   showline=True, linecolor=DARK, tickfont=dict(color=DARK)),
        legend=dict(orientation="h", yanchor="bottom", y=1.01, x=0),
    )
    st.plotly_chart(fig_cdf, use_container_width=True, theme=None)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — PREDICTING DROUGHT  (Logistic Regression from scratch)
# ═══════════════════════════════════════════════════════════════════════════════
with tab_logit:
    st.markdown("### Predicting Drought: Logistic Regression")
    st.caption(
        "The previous tab defined drought as a rainfall score below -0.5. "
        "Now we ask: **can we predict a month in advance** whether next month will be a drought month? "
        "We train a model on data from 2000 to 2015, then test it on 2016 to 2020."
    )
    with st.expander("Statistical details"):
        st.markdown(r"""
**Logistic regression as a probabilistic classifier**

$$\Pr(y=1 \mid x) = \sigma(\beta^\top x) = \frac{1}{1+e^{-\beta^\top x}}$$

**Log-likelihood (MLE objective)**

$$\ell(\beta) = \sum_i \bigl[y_i \log\sigma(\beta^\top x_i) + (1-y_i)\log(1-\sigma(\beta^\top x_i))\bigr]$$

**Gradient ascent update** (no closed form — log-likelihood is concave but not quadratic):

$$\beta \leftarrow \beta + \eta \sum_i (y_i - \sigma(\beta^\top x_i))\, x_i$$

Key CS109 result: the gradient $\nabla_\beta\ell = X^\top(y - \hat{p})$ has a beautifully clean form.

**Features**: SPI-12 at month $t$, sin/cos of calendar month (seasonality).
**Label**: $y_t = 1$ if SPI-12 at month $t+1 < -0.5$ (drought), else 0.
**Split**: train 2000–2015, test 2016–2020.
""")

    # ── Build features and labels ────────────────────────────────────────────
    T_anom = len(spi12)
    # .to_numpy() forces numpy array — pandas 3.0 removed .mean() from Index
    months_sin = np.sin(2 * np.pi * anom_dates.month.to_numpy() / 12)
    months_cos = np.cos(2 * np.pi * anom_dates.month.to_numpy() / 12)

    # Features at t, label is drought at t+1
    feat = np.column_stack([
        np.ones(T_anom - 1),          # bias
        spi12[:-1],
        months_sin[:-1],
        months_cos[:-1],
    ])
    labels = (spi12[1:] < DROUGHT_THRESH).astype(float)
    feat_dates = anom_dates[:-1]

    # Train/test split on year boundary
    train_mask = feat_dates.year.to_numpy() <= 2015
    test_mask  = feat_dates.year.to_numpy() >= 2016

    X_tr, y_tr = feat[train_mask], labels[train_mask]
    X_te, y_te = feat[test_mask],  labels[test_mask]

    # ── Gradient ascent ──────────────────────────────────────────────────────
    @st.cache_data(show_spinner=False)
    def _fit_logistic(X_bytes, y_bytes, n_iter=2000, lr=0.05):
        X = np.frombuffer(X_bytes, dtype=np.float64).reshape(-1, 4)
        y = np.frombuffer(y_bytes, dtype=np.float64)
        beta = np.zeros(X.shape[1])
        lls  = []
        for _ in range(n_iter):
            logits = X @ beta
            p = 1.0 / (1.0 + np.exp(-np.clip(logits, -30, 30)))
            ll = float(np.sum(y * np.log(p + 1e-15) + (1 - y) * np.log(1 - p + 1e-15)))
            lls.append(ll)
            grad = X.T @ (y - p)
            beta += lr * grad
        return beta, np.array(lls)

    beta, train_lls = _fit_logistic(X_tr.tobytes(), y_tr.tobytes())

    def sigmoid(z): return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))

    p_tr = sigmoid(X_tr @ beta)
    p_te = sigmoid(X_te @ beta)
    pred_te = (p_te >= 0.5).astype(int)

    # ── Metrics ──────────────────────────────────────────────────────────────
    TP = int(((pred_te == 1) & (y_te == 1)).sum())
    FP = int(((pred_te == 1) & (y_te == 0)).sum())
    TN = int(((pred_te == 0) & (y_te == 0)).sum())
    FN = int(((pred_te == 0) & (y_te == 1)).sum())
    acc  = (TP + TN) / len(y_te)
    prec = TP / (TP + FP) if TP + FP > 0 else 0.0
    rec  = TP / (TP + FN) if TP + FN > 0 else 0.0
    f1   = 2 * prec * rec / (prec + rec) if prec + rec > 0 else 0.0

    # ── Layout: 2 columns ────────────────────────────────────────────────────
    left, right = st.columns(2)

    # Training log-likelihood curve
    with left:
        st.markdown("##### Training log-likelihood (gradient ascent)")
        fig_ll = go.Figure(go.Scatter(
            x=list(range(1, len(train_lls) + 1)), y=train_lls,
            mode="lines", line=dict(color="#3978AE", width=2),
            hovertemplate="Iter %{x}<br>LL = %{y:.1f}<extra></extra>",
        ))
        fig_ll.update_layout(
            **PLOTLY_LAYOUT, height=280,
            xaxis=dict(title="Gradient ascent iteration",
                       showgrid=False, showline=True, linecolor=DARK),
            yaxis=dict(title="Log-likelihood",
                       showgrid=True, gridcolor="#eee", showline=True, linecolor=DARK),
        )
        st.plotly_chart(fig_ll, use_container_width=True, theme=None)

    # Metrics in the right column
    with right:
        st.markdown("##### Test-set performance (2016–2020)")
        st.markdown("<br>", unsafe_allow_html=True)
        m1, m2 = st.columns(2)
        m1.metric("Accuracy",  f"{acc:.0%}")
        m2.metric("F1 score",  f"{f1:.0%}")
        m3, m4 = st.columns(2)
        m3.metric("Precision", f"{prec:.0%}")
        m4.metric("Recall",    f"{rec:.0%}")
        st.caption(
            f"Trained on {int(train_mask.sum())} months (2000–2015), "
            f"tested on {int(test_mask.sum())} months (2016–2020). "
            f"A month is labelled drought if its rainfall score drops below {DROUGHT_THRESH}."
        )

    # ── Predicted probability timeline ───────────────────────────────────────
    st.markdown("##### Did the model sense droughts coming?")
    st.caption(
        "Grey filled area = model's predicted P(drought next month) for every month in the record. "
        "**Red triangles along the bottom = months that actually became droughts.** "
        "When the grey curve peaks above the orange dashed line and a red triangle sits below it, "
        "the model called it correctly."
    )

    p_all_dates = anom_dates[:-1]
    p_all = sigmoid(feat @ beta)
    y_all = labels

    drought_dates  = p_all_dates[y_all == 1]
    drought_marker = np.full(int((y_all == 1).sum()), -0.03)   # just below zero

    fig_pred = go.Figure()
    # Filled predicted probability
    fig_pred.add_trace(go.Scatter(
        x=p_all_dates, y=p_all,
        mode="lines", fill="tozeroy",
        line=dict(color="#555", width=1.4),
        fillcolor="rgba(80,80,80,0.18)",
        name="P(drought next month)",
        hovertemplate="%{x|%b %Y}<br>P = %{y:.0%}<extra></extra>",
    ))
    # Actual drought months as red tick marks at the bottom
    fig_pred.add_trace(go.Scatter(
        x=drought_dates, y=drought_marker,
        mode="markers",
        marker=dict(symbol="triangle-up", color="#A60000", size=8),
        name="Actual drought month",
        hovertemplate="%{x|%b %Y} — drought<extra></extra>",
    ))
    fig_pred.add_hline(y=0.5, line_dash="dot", line_color="#E66B00", line_width=1.5,
                       annotation_text="50% decision threshold",
                       annotation_position="top right",
                       annotation_font=dict(color="#E66B00", size=11))
    fig_pred.update_layout(
        **PLOTLY_LAYOUT, height=340,
        xaxis=dict(title=None, showgrid=False, showline=True, linecolor=DARK),
        yaxis=dict(title="P(drought next month)",
                   range=[-0.08, 1.02], tickformat=".0%",
                   showgrid=True, gridcolor="#eee",
                   showline=True, linecolor=DARK),
        legend=dict(x=0.01, y=0.97),
    )
    st.plotly_chart(fig_pred, use_container_width=True, theme=None)

    # ── Calibration curve (reliability diagram) ──────────────────────────────
    st.markdown("##### Is the model's confidence well-calibrated?")
    st.caption(
        "When the model says there is a 60% chance of drought, does drought actually happen 60% of the time? "
        "Each dot represents a group of months with similar predicted probabilities. "
        "**Dots on the diagonal line mean the model is perfectly calibrated.** "
        "Dot size shows how many months are in each group."
    )

    N_CAL_BINS = 8
    bin_edges_cal = np.linspace(0, 1, N_CAL_BINS + 1)
    bin_centers_cal, frac_pos, bin_counts = [], [], []
    for lo, hi in zip(bin_edges_cal[:-1], bin_edges_cal[1:]):
        mask_bin = (p_all >= lo) & (p_all < hi)
        if mask_bin.sum() >= 3:
            bin_centers_cal.append(float(p_all[mask_bin].mean()))
            frac_pos.append(float(y_all[mask_bin].mean()))
            bin_counts.append(int(mask_bin.sum()))

    fig_cal = go.Figure()
    fig_cal.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], mode="lines",
        line=dict(color="#bbb", dash="dash", width=1.5),
        name="Perfect calibration (y = x)",
    ))
    fig_cal.add_trace(go.Scatter(
        x=bin_centers_cal, y=frac_pos, mode="lines+markers",
        line=dict(color="#A60000", width=2.5),
        marker=dict(size=[max(8, c // 3) for c in bin_counts],
                    color="#A60000", line=dict(color="white", width=1.5)),
        name="Model",
        hovertemplate="Predicted P = %{x:.2f}<br>Actual fraction = %{y:.2f}<extra></extra>",
    ))
    fig_cal.update_layout(
        **PLOTLY_LAYOUT, height=320,
        xaxis=dict(title="Predicted P(drought)",
                   range=[0, 1], showgrid=True, gridcolor="#eee",
                   showline=True, linecolor=DARK),
        yaxis=dict(title="Observed drought fraction",
                   range=[0, 1], showgrid=True, gridcolor="#eee",
                   showline=True, linecolor=DARK),
        legend=dict(x=0.02, y=0.95),
    )
    st.plotly_chart(fig_cal, use_container_width=True, theme=None)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 — QUANTIFYING UNCERTAINTY  (Bootstrap + CLT + Bayesian)
# ═══════════════════════════════════════════════════════════════════════════════
with tab_boot:
    st.markdown("### Uncertainty: How Confident Are We in the Drought Rate?")
    st.caption(
        f"We observed drought in about {drought_rate:.0%} of months. "
        "But that is based on just 241 months of data, so there is some uncertainty. "
        "The chart below shows the full range of plausible true drought rates. "
        "The peak is our best estimate; the shaded region is where the true rate almost certainly falls."
    )
    with st.expander("Statistical details"):
        st.markdown(r"""
**Beta-Binomial conjugacy**:

Treat each of the $n = 241$ months as a Bernoulli trial: drought ($y=1$) or not ($y=0$).
With $s$ drought months observed, the likelihood is Binomial($n$, $p$).

Choose a flat (uninformative) prior: $p \sim \text{Beta}(1, 1)$ — every value of $p$ equally likely.

Bayes' theorem gives a closed-form posterior:
$$p \mid \text{data} \;\sim\; \text{Beta}(\alpha + s,\; \beta + n - s)$$
with $\alpha = \beta = 1$ (the prior pseudo-counts). The posterior mean is $\frac{s+1}{n+2} \approx \hat p$.

The **95% credible interval** is the central 95% area under this curve — it has a direct
probability interpretation: *"There is a 95% probability that the true long-run drought rate
lies in this interval,"* given the data and the flat prior.
""")

    N_BOOT = 10_000

    @st.cache_data(show_spinner=False)
    def _bootstrap_prop(spi_bytes, thr, n_boot=N_BOOT, seed=2):
        spi = np.frombuffer(spi_bytes, dtype=np.float64)
        y = (spi < thr).astype(float)
        rng = np.random.default_rng(seed)
        boot = rng.choice(y, size=(n_boot, len(y)), replace=True).mean(axis=1)
        return boot, y

    precip_vals = precip_raw.values.astype(np.float64)
    boot_prop, y_drought = _bootstrap_prop(spi12.tobytes(), DROUGHT_THRESH)

    # ── Bayesian Beta-Binomial posterior on drought rate ─────────────────────
    n2   = len(y_drought)
    s2   = int(y_drought.sum())
    phat = s2 / n2

    # Flat Beta(1,1) prior → posterior Beta(s+1, n-s+1)
    alpha_post = 1 + s2
    beta_post  = 1 + n2 - s2
    bayes_lo, bayes_hi = stats.beta.ppf([0.025, 0.975], alpha_post, beta_post)

    p_grid = np.linspace(0, 1, 500)
    posterior_pdf = stats.beta.pdf(p_grid, alpha_post, beta_post)

    # Shade the 95% credible interval under the curve
    ci_mask = (p_grid >= bayes_lo) & (p_grid <= bayes_hi)

    fig_bayes = go.Figure()
    # Shaded CI region
    fig_bayes.add_trace(go.Scatter(
        x=np.concatenate([[bayes_lo], p_grid[ci_mask], [bayes_hi]]),
        y=np.concatenate([[0], posterior_pdf[ci_mask], [0]]),
        fill="tozeroy", mode="none",
        fillcolor="rgba(166,0,0,0.15)",
        showlegend=False, hoverinfo="skip",
    ))
    # Posterior curve
    fig_bayes.add_trace(go.Scatter(
        x=p_grid, y=posterior_pdf,
        mode="lines", line=dict(color="#A60000", width=2.5),
        name=f"Posterior  Beta({alpha_post}, {beta_post})",
        hovertemplate="p = %{x:.3f}<br>density = %{y:.1f}<extra></extra>",
    ))
    # Observed proportion
    fig_bayes.add_vline(x=phat, line_color="#222", line_width=2,
                        annotation_text=f"Observed: {phat:.0%}  ({s2} of {n2} months)",
                        annotation_position="top right",
                        annotation_font=dict(size=12, color="#222"))
    fig_bayes.update_layout(
        **PLOTLY_LAYOUT, height=380,
        xaxis=dict(title="Possible true drought rate",
                   range=[0, 1], showgrid=False, showline=True, linecolor=DARK,
                   tickformat=".0%", tickfont=dict(color=DARK)),
        yaxis=dict(title="Probability of this being the true rate",
                   showgrid=True, gridcolor="#eee",
                   showline=True, linecolor=DARK, tickfont=dict(color=DARK)),
        showlegend=False,
        annotations=[dict(
            x=(bayes_lo + bayes_hi) / 2, y=posterior_pdf[ci_mask].max() * 0.45,
            text=f"95% credible interval<br>[{bayes_lo:.2f}, {bayes_hi:.2f}]",
            showarrow=False, bgcolor="rgba(255,255,255,0.85)",
            bordercolor="#ccc", borderwidth=1, borderpad=5,
            font=dict(size=12, color="#A60000"),
        )],
    )
    st.plotly_chart(fig_bayes, use_container_width=True, theme=None)
    st.caption(
        f"We saw drought in {s2} out of {n2} months. "
        f"Starting from no prior assumption, the shaded region captures 95% of plausible true rates. "
        f"We are 95% confident the true long-run drought rate falls between "
        f"**{bayes_lo:.0%}** and **{bayes_hi:.0%}**."
    )


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 5 — INFORMATION CONTENT  (Shannon entropy + KL divergence)
# ═══════════════════════════════════════════════════════════════════════════════
with tab_info:
    st.markdown("### Predictability: Which Years Were Surprising?")
    st.caption(
        "The previous tabs looked at drought month by month. "
        "Here we zoom out to the year level and ask: **how varied was each year's drought pattern?** "
        "A year stuck in severe drought all twelve months is actually very predictable — no surprises. "
        "A year that swings from drought to normal to wet and back is much harder to anticipate. "
        "The bars below measure that variety. Shorter bar = more predictable year."
    )
    with st.expander("Statistical details"):
        st.markdown(r"""
**Shannon entropy** (expected log-surprise):
$$H(X) = -\sum_i p_i \log p_i = -\mathbb{E}[\log p(X)]$$
$H = 0$ when all probability mass is on one category (perfectly predictable);
$H$ is maximised at $\ln 6 \approx 1.79$ nats when all six USDM categories are equally likely.

For each year we count how many months fell in each of six drought categories (D0–D4 + Normal),
compute the empirical probability $p_i = \text{count}_i / 12$, and plug into the formula above.

**Connection to MLE**: maximising likelihood is equivalent to minimising KL divergence from
the empirical distribution to the model — entropy is the self-information lower bound on that cost.
""")

    # ── USDM category labels ─────────────────────────────────────────────────
    # Map SPI-12 values to 6 drought categories using USDM thresholds
    # (approximately matching percentile thresholds to z-scores)
    def spi_to_category(spi_val):
        if spi_val < -2.0:   return "D4"
        elif spi_val < -1.6: return "D3"
        elif spi_val < -1.3: return "D2"
        elif spi_val < -0.8: return "D1"
        elif spi_val < -0.5: return "D0"
        else:                return "Normal"

    CAT_ORDER = ["D4", "D3", "D2", "D1", "D0", "Normal"]
    CAT_COLORS = ["#5C0000", "#A60000", "#E66B00", "#FFA94D", "#FFE099", "#F2F2F2"]

    spi_series = pd.Series(spi12, index=anom_dates)
    cat_series = spi_series.apply(spi_to_category)

    # ── Analysis 1: Entropy per year ─────────────────────────────────────────
    st.markdown("#### Variety of drought conditions, by year")

    def entropy(counts):
        total = counts.sum()
        if total == 0:
            return 0.0
        p = counts / total
        p = p[p > 0]
        return float(-np.sum(p * np.log(p)))

    years_list = sorted(cat_series.index.year.unique())
    H_per_year = []
    for yr in years_list:
        yr_cats = cat_series[cat_series.index.year == yr]
        counts = np.array([int((yr_cats == c).sum()) for c in CAT_ORDER], dtype=float)
        H_per_year.append(entropy(counts))

    H_max = np.log(6)   # maximum entropy over 6 categories

    fig_ent = go.Figure()
    fig_ent.add_trace(go.Bar(
        x=years_list, y=H_per_year,
        marker=dict(
            color=["#A60000" if h < H_max * 0.6 else "#4D94CC" for h in H_per_year],
            line=dict(width=0),
        ),
        hovertemplate="Year %{x}<br>Variety score: %{y:.2f}<extra></extra>",
    ))
    fig_ent.add_hline(y=H_max, line_dash="dot", line_color="#888",
                      annotation_text="Maximum possible variety",
                      annotation_position="top right",
                      annotation_font=dict(size=11, color="#888"))
    fig_ent.update_layout(
        **PLOTLY_LAYOUT, height=320,
        xaxis=dict(title="Year", type="category",
                   showgrid=False, showline=True, linecolor=DARK,
                   tickfont=dict(color=DARK)),
        yaxis=dict(title="Variety score",
                   showgrid=True, gridcolor="#eee",
                   showline=True, linecolor=DARK,
                   range=[0, H_max * 1.1]),
    )
    st.plotly_chart(fig_ent, use_container_width=True, theme=None)

    driest_yr  = years_list[int(np.argmin(H_per_year))]
    highest_yr = years_list[int(np.argmax(H_per_year))]
    st.caption(
        f"**{driest_yr}** is the most predictable year: the valley was in severe drought nearly "
        f"every single month, so there was little variety. "
        f"**{highest_yr}** is the least predictable: conditions shifted across many different "
        f"drought categories throughout the year. "
        "The surprising takeaway: bad drought years are actually easier to anticipate than transition years."
    )
