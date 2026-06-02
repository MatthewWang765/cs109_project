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


@st.cache_data(show_spinner="Loading SPI-12 / ETI-12 anomalies…")
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
        "precipitation and evapotranspiration alone."
        "</div>",
        unsafe_allow_html=True,
    )
    st.divider()
    st.markdown("**Data**")
    st.markdown(
        "NASA GPM IMERG Late-Run precipitation  \n"
        "OpenET Monthly Ensemble evapotranspiration  \n"
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
X_anom, anom_dates = load_anomalies()   # (241, 2) SPI-12, ETI-12
precip_raw, et_raw, raw_dates = load_raw()

spi12  = X_anom[:, 0]
eti12  = X_anom[:, 1]
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
st.markdown("## San Joaquin Valley Drought Analysis  \n###### Probabilistic Methods · 2000–2020")
st.markdown("##### Quick Statistics")
st.markdown(
    "<div style='color:#888; font-size:0.85rem; margin-top:-0.4rem; margin-bottom:0.6rem;'>"
    "Climatological highlights from the 241-month SPI-12 record."
    "</div>",
    unsafe_allow_html=True,
)

annual_spi = pd.Series(spi12, index=anom_dates).resample("YS").mean()
col1, col2, col3, col4 = st.columns(4)
col1.markdown(
    f"**Driest year**  \n"
    f"<span style='font-size:1.8rem;font-weight:600;'>{int(annual_spi.idxmin().year)}</span>  \n"
    f"<span style='color:#888;font-size:0.85rem;'>lowest mean SPI-12</span>",
    unsafe_allow_html=True,
)
col2.markdown(
    f"**Wettest year**  \n"
    f"<span style='font-size:1.8rem;font-weight:600;'>{int(annual_spi.idxmax().year)}</span>  \n"
    f"<span style='color:#888;font-size:0.85rem;'>highest mean SPI-12</span>",
    unsafe_allow_html=True,
)
col3.markdown(
    f"**Bootstrap drought rate**  \n"
    f"<span style='font-size:1.8rem;font-weight:600;'>{drought_rate*100:.0f}%</span>  \n"
    f"<span style='color:#888;font-size:0.85rem;'>95% CI: [{dr_lo*100:.0f}%, {dr_hi*100:.0f}%]</span>",
    unsafe_allow_html=True,
)
col4.markdown(
    f"**Sample size**  \n"
    f"<span style='font-size:1.8rem;font-weight:600;'>241 months</span>  \n"
    f"<span style='color:#888;font-size:0.85rem;'>one observation per month</span>",
    unsafe_allow_html=True,
)

# ── Tab layout ────────────────────────────────────────────────────────────────
tab_map, tab_precip, tab_logit, tab_boot, tab_info = st.tabs([
    "Drought Map",
    "Modeling Precipitation",
    "Predicting Drought",
    "Quantifying Uncertainty",
    "Information Content",
])


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — SPATIAL DROUGHT MAP  (preserved verbatim)
# ═══════════════════════════════════════════════════════════════════════════════
with tab_map:
    st.markdown("### San Joaquin Valley Drought Map")
    st.caption(
        "Each square is a 10 km patch of the SJV. Its colour shows how dry or wet that patch was, "
        "compared to all other Januaries (or Julys, etc.) in our 21-year record. "
        "**Dark red** = driest version of that month ever recorded; "
        "**dark blue** = wettest. Use the date picker, slider, or ▶ Play to step through time."
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
        help="Pick any date — the map snaps to the nearest available SPI-12 month.",
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
    st.markdown("### Modeling Monthly Precipitation — Gamma vs Normal MLE")
    st.caption(
        "Monthly precipitation in the SJV is right-skewed and can't go below zero — "
        "properties the Normal distribution handles poorly. "
        "The **Gamma distribution** is the climatological standard for monthly rainfall, "
        "and its parameters can be estimated by maximum-likelihood, exactly as in CS109."
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
    c1.metric("Gamma log-likelihood", f"{gamma_ll:.1f}")
    c2.metric("Normal log-likelihood", f"{norm_ll:.1f}",
              delta=f"{norm_ll - gamma_ll:.1f} vs Gamma", delta_color="inverse")
    c3.metric("Gamma wins?", "Yes ✓" if gamma_ll > norm_ll else "No")

    st.caption(
        f"The Gamma fit (log-ℓ = {gamma_ll:.1f}) beats the Normal fit "
        f"(log-ℓ = {norm_ll:.1f}) by {gamma_ll - norm_ll:.1f} nats for {selected_month_name}. "
        "Higher log-likelihood = better fit. "
        "The Normal also extends below zero, which is physically impossible for precipitation."
    )

    # ── SPI bonus: show CDF transformation ──────────────────────────────────
    st.markdown("#### Bonus: Standardised Precipitation Index (SPI)")
    st.caption(
        "The SPI converts each month's precipitation total to a z-score by way of the fitted "
        "Gamma CDF. Values below −0.5 indicate drought; below −1.5 indicate severe drought."
    )

    spi_vals = stats.norm.ppf(stats.gamma.cdf(month_data, k_hat, loc_hat, theta_hat))
    spi_vals = spi_vals[np.isfinite(spi_vals)]

    fig_spi = go.Figure()
    fig_spi.add_trace(go.Bar(
        x=list(range(len(spi_vals))),
        y=spi_vals,
        marker=dict(
            color=["#A60000" if v < -0.5 else "#4D94CC" for v in spi_vals],
            line=dict(width=0),
        ),
        hovertemplate="Year offset %{x}<br>SPI = %{y:.2f}<extra></extra>",
        showlegend=False,
    ))
    fig_spi.add_hline(y=-0.5, line_dash="dot", line_color="#E66B00",
                      annotation_text="Drought threshold (−0.5)",
                      annotation_position="bottom right",
                      annotation_font=dict(color="#E66B00", size=11))
    fig_spi.update_layout(
        **PLOTLY_LAYOUT, height=280,
        xaxis=dict(title=f"{selected_month_name} index (across 21-yr record)",
                   showgrid=False, showline=True, linecolor=DARK),
        yaxis=dict(title="SPI", showgrid=True, gridcolor="#eee",
                   showline=True, linecolor=DARK),
    )
    st.plotly_chart(fig_spi, use_container_width=True, theme=None)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — PREDICTING DROUGHT  (Logistic Regression from scratch)
# ═══════════════════════════════════════════════════════════════════════════════
with tab_logit:
    st.markdown("### Predicting Next-Month Drought — Logistic Regression")
    st.caption(
        "Can this month's precipitation and ET anomalies predict whether **next month** will be "
        "a drought month? We fit a logistic regression from scratch using gradient ascent on the "
        "log-likelihood, exactly as derived in CS109."
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

**Features**: SPI-12 at month $t$, ETI-12 at month $t$, sin/cos of calendar month.
**Label**: $y_t = 1$ if SPI-12 at month $t+1 < -0.5$ (drought), else 0.
**Split**: train 2000–2015, test 2016–2020.
""")

    # ── Build features and labels ────────────────────────────────────────────
    T_anom = len(spi12)
    months_sin = np.sin(2 * np.pi * anom_dates.month / 12)
    months_cos = np.cos(2 * np.pi * anom_dates.month / 12)

    # Features at t, label is drought at t+1
    feat = np.column_stack([
        np.ones(T_anom - 1),          # bias
        spi12[:-1],
        eti12[:-1],
        months_sin[:-1],
        months_cos[:-1],
    ])
    labels = (spi12[1:] < DROUGHT_THRESH).astype(float)
    feat_dates = anom_dates[:-1]

    # Train/test split on year boundary
    train_mask = feat_dates.year <= 2015
    test_mask  = feat_dates.year >= 2016

    X_tr, y_tr = feat[train_mask], labels[train_mask]
    X_te, y_te = feat[test_mask],  labels[test_mask]

    # ── Gradient ascent ──────────────────────────────────────────────────────
    @st.cache_data(show_spinner=False)
    def _fit_logistic(X_bytes, y_bytes, n_iter=2000, lr=0.05):
        X = np.frombuffer(X_bytes, dtype=np.float64).reshape(-1, 5)
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

    # Confusion matrix
    with right:
        st.markdown("##### Confusion matrix (test set 2016–2020)")
        cm = np.array([[TN, FP], [FN, TP]])
        fig_cm = go.Figure(go.Heatmap(
            z=cm, x=["Pred: No Drought", "Pred: Drought"],
            y=["True: No Drought", "True: Drought"],
            colorscale=[[0, "#F2F2F2"], [1, "#3978AE"]],
            showscale=False, zmin=0,
            text=[[str(v) for v in row] for row in cm],
            texttemplate="<b>%{text}</b>",
            textfont=dict(size=22, color=DARK),
            hovertemplate="%{y} / %{x}<br>Count: %{z}<extra></extra>",
        ))
        fig_cm.update_layout(
            **PLOTLY_LAYOUT, height=280,
            xaxis=dict(side="bottom", tickfont=dict(color=DARK, size=12)),
            yaxis=dict(autorange="reversed", tickfont=dict(color=DARK, size=12)),
        )
        st.plotly_chart(fig_cm, use_container_width=True, theme=None)

    # Metrics row
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Accuracy",  f"{acc:.0%}")
    m2.metric("Precision", f"{prec:.0%}")
    m3.metric("Recall",    f"{rec:.0%}")
    m4.metric("F1",        f"{f1:.0%}")

    # ROC curve
    thresholds = np.linspace(0, 1, 200)
    tprs, fprs = [], []
    for thr in thresholds:
        pp = (p_te >= thr).astype(int)
        tp_ = int(((pp == 1) & (y_te == 1)).sum())
        fp_ = int(((pp == 1) & (y_te == 0)).sum())
        tn_ = int(((pp == 0) & (y_te == 0)).sum())
        fn_ = int(((pp == 0) & (y_te == 1)).sum())
        tprs.append(tp_ / (tp_ + fn_) if tp_ + fn_ > 0 else 0.0)
        fprs.append(fp_ / (fp_ + tn_) if fp_ + tn_ > 0 else 0.0)

    auc = float(np.trapz(tprs[::-1], fprs[::-1]))

    left2, right2 = st.columns(2)

    with left2:
        st.markdown("##### ROC curve (test set)")
        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(
            x=fprs, y=tprs, mode="lines",
            line=dict(color="#A60000", width=2),
            name=f"Logistic (AUC = {auc:.3f})",
        ))
        fig_roc.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1], mode="lines",
            line=dict(color="#bbb", dash="dash"), showlegend=False,
        ))
        fig_roc.update_layout(
            **PLOTLY_LAYOUT, height=300,
            xaxis=dict(title="False positive rate", range=[0, 1],
                       showgrid=False, showline=True, linecolor=DARK),
            yaxis=dict(title="True positive rate", range=[0, 1],
                       showgrid=True, gridcolor="#eee", showline=True, linecolor=DARK),
            legend=dict(x=0.5, y=0.05),
        )
        st.plotly_chart(fig_roc, use_container_width=True, theme=None)

    # Coefficient bar chart
    with right2:
        st.markdown("##### Learned coefficients β")
        feat_names = ["Bias", "SPI-12", "ETI-12", "Month sin", "Month cos"]
        fig_beta = go.Figure(go.Bar(
            x=beta,
            y=feat_names,
            orientation="h",
            marker=dict(
                color=["#3978AE" if v >= 0 else "#A60000" for v in beta],
            ),
            hovertemplate="%{y}: β = %{x:.3f}<extra></extra>",
        ))
        fig_beta.add_vline(x=0, line_color=DARK, line_width=1)
        fig_beta.update_layout(
            **PLOTLY_LAYOUT, height=300,
            xaxis=dict(title="Coefficient value",
                       showgrid=True, gridcolor="#eee", showline=True, linecolor=DARK),
            yaxis=dict(showgrid=False, showline=True, linecolor=DARK),
        )
        st.plotly_chart(fig_beta, use_container_width=True, theme=None)

    st.caption(
        "Negative SPI-12 (drier than normal) **increases** the probability of drought next month "
        "(negative β → logit decreases → lower P(drought) when precip is above normal, "
        "higher when below). "
        "The sign of the SPI-12 coefficient should be negative: more precipitation now → "
        "less drought next month."
    )


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 — QUANTIFYING UNCERTAINTY  (Bootstrap + CLT + Bayesian)
# ═══════════════════════════════════════════════════════════════════════════════
with tab_boot:
    st.markdown("### Quantifying Uncertainty — Bootstrap, CLT, and Bayesian Inference")
    st.caption(
        "How confident should we be in statistics like 'the SJV is in drought 32% of the time'? "
        "We compare three methods for building confidence intervals: "
        "**bootstrap** (nonparametric), **CLT** (Normal approximation), and "
        "**Bayesian Beta-Binomial conjugacy** — all from CS109."
    )
    with st.expander("Statistical details"):
        st.markdown(r"""
**Bootstrap** (nonparametric, no distributional assumption):
Resample the data with replacement $B$ times. The 2.5th and 97.5th percentiles of the
bootstrap distribution form the 95% CI.

**Central Limit Theorem**:
$$\bar X \xrightarrow{d} \mathcal{N}(\mu,\, \sigma^2/n) \text{ as } n\to\infty$$
For a proportion $\hat p$ (Wald interval): $\hat p \pm 1.96\sqrt{\hat p(1-\hat p)/n}$.

**Beta-Binomial conjugacy**:
If prior is Beta($\alpha, \beta$) and likelihood is Binomial($n, p$) with $s$ successes,
the posterior is Beta($\alpha + s$, $\beta + n - s$). With a flat Beta(1,1) prior this
gives Beta($s+1$, $n-s+1$). The central 95% credible interval is the 2.5th–97.5th
percentile of this distribution.
""")

    N_BOOT = 10_000

    @st.cache_data(show_spinner=False)
    def _bootstrap_mean(precip_bytes, n_boot=N_BOOT, seed=1):
        p = np.frombuffer(precip_bytes, dtype=np.float64)
        rng = np.random.default_rng(seed)
        boot = rng.choice(p, size=(n_boot, len(p)), replace=True).mean(axis=1)
        return boot

    @st.cache_data(show_spinner=False)
    def _bootstrap_prop(spi_bytes, thr, n_boot=N_BOOT, seed=2):
        spi = np.frombuffer(spi_bytes, dtype=np.float64)
        y = (spi < thr).astype(float)
        rng = np.random.default_rng(seed)
        boot = rng.choice(y, size=(n_boot, len(y)), replace=True).mean(axis=1)
        return boot, y

    @st.cache_data(show_spinner=False)
    def _bootstrap_diff(spi_bytes, raw_bytes, thr, n_boot=N_BOOT, seed=3):
        spi = np.frombuffer(spi_bytes, dtype=np.float64)
        raw = np.frombuffer(raw_bytes, dtype=np.float64)
        drought = raw[spi < thr]
        normal  = raw[spi >= thr]
        rng = np.random.default_rng(seed)
        d_boot = rng.choice(drought, size=(n_boot, len(drought)), replace=True).mean(axis=1)
        n_boot_arr = rng.choice(normal, size=(n_boot, len(normal)), replace=True).mean(axis=1)
        return d_boot - n_boot_arr, drought, normal

    precip_vals = precip_raw.values

    boot_mean   = _bootstrap_mean(precip_vals.tobytes())
    boot_prop, y_drought = _bootstrap_prop(spi12.tobytes(), DROUGHT_THRESH)
    boot_diff, drought_precip, normal_precip = _bootstrap_diff(
        spi12.tobytes(), precip_vals[:len(spi12)].tobytes(), DROUGHT_THRESH
    )

    # ── Analysis 1: Mean monthly precipitation ───────────────────────────────
    st.markdown("#### 1. Mean monthly precipitation across all 241 months")

    n = len(precip_vals)
    xbar = precip_vals.mean()
    s    = precip_vals.std(ddof=1)
    clt_lo = xbar - 1.96 * s / np.sqrt(n)
    clt_hi = xbar + 1.96 * s / np.sqrt(n)
    boot_lo1, boot_hi1 = np.percentile(boot_mean, [2.5, 97.5])

    fig1 = go.Figure()
    fig1.add_trace(go.Histogram(
        x=boot_mean, nbinsx=60,
        histnorm="probability density",
        marker=dict(color="#B3D9E6", line=dict(color="white", width=0.5)),
        name="Bootstrap distribution",
    ))
    # Normal overlay (CLT)
    x_norm = np.linspace(boot_mean.min(), boot_mean.max(), 300)
    fig1.add_trace(go.Scatter(
        x=x_norm,
        y=stats.norm.pdf(x_norm, xbar, s / np.sqrt(n)),
        mode="lines", line=dict(color="#A60000", width=2.5, dash="dot"),
        name=f"CLT Normal  [{clt_lo:.1f}, {clt_hi:.1f}]",
    ))
    for lo, hi, label, col in [
        (boot_lo1, boot_hi1, "Bootstrap 95% CI", "#3978AE"),
        (clt_lo,  clt_hi,   "CLT 95% CI",       "#A60000"),
    ]:
        for v, side in [(lo, "left"), (hi, "right")]:
            fig1.add_vline(x=v, line_dash="dash", line_color=col, line_width=1.5)
    fig1.add_vline(x=xbar, line_color="#222", line_width=2,
                   annotation_text=f"x̄ = {xbar:.1f} mm",
                   annotation_position="top right")
    fig1.update_layout(
        **PLOTLY_LAYOUT, height=320,
        xaxis=dict(title="Bootstrap mean monthly precipitation (mm)",
                   showgrid=False, showline=True, linecolor=DARK),
        yaxis=dict(title="Density", showgrid=True, gridcolor="#eee",
                   showline=True, linecolor=DARK),
        legend=dict(x=0.65, y=0.95),
    )
    st.plotly_chart(fig1, use_container_width=True, theme=None)

    r1a, r1b = st.columns(2)
    r1a.metric("Bootstrap 95% CI", f"[{boot_lo1:.1f}, {boot_hi1:.1f}] mm")
    r1b.metric("CLT 95% CI",       f"[{clt_lo:.1f}, {clt_hi:.1f}] mm")

    # ── Analysis 2: Drought probability ─────────────────────────────────────
    st.markdown("#### 2. Probability of drought in a random month")

    n2  = len(y_drought)
    s2  = int(y_drought.sum())
    phat = s2 / n2
    wald_lo = phat - 1.96 * np.sqrt(phat * (1 - phat) / n2)
    wald_hi = phat + 1.96 * np.sqrt(phat * (1 - phat) / n2)
    boot_lo2, boot_hi2 = np.percentile(boot_prop, [2.5, 97.5])

    # Bayesian Beta-Binomial posterior
    alpha_post = 1 + s2
    beta_post  = 1 + n2 - s2
    bayes_lo, bayes_hi = stats.beta.ppf([0.025, 0.975], alpha_post, beta_post)
    p_grid = np.linspace(0, 1, 400)
    posterior_pdf = stats.beta.pdf(p_grid, alpha_post, beta_post)

    fig2 = go.Figure()
    fig2.add_trace(go.Histogram(
        x=boot_prop, nbinsx=60,
        histnorm="probability density",
        marker=dict(color="#FFE099", line=dict(color="white", width=0.5)),
        name="Bootstrap distribution",
    ))
    fig2.add_trace(go.Scatter(
        x=p_grid, y=posterior_pdf,
        mode="lines", line=dict(color="#A60000", width=2.5),
        name=f"Bayesian posterior  Beta({alpha_post},{beta_post})",
    ))
    for lo, hi, col in [
        (boot_lo2, boot_hi2, "#3978AE"),
        (wald_lo,  wald_hi,  "#888888"),
        (bayes_lo, bayes_hi, "#A60000"),
    ]:
        fig2.add_vrect(x0=lo, x1=hi, fillcolor=col, opacity=0.10, line_width=0)
        fig2.add_vline(x=lo, line_dash="dash", line_color=col, line_width=1.5)
        fig2.add_vline(x=hi, line_dash="dash", line_color=col, line_width=1.5)
    fig2.add_vline(x=phat, line_color="#222", line_width=2,
                   annotation_text=f"p̂ = {phat:.2f}",
                   annotation_position="top right")
    fig2.update_layout(
        **PLOTLY_LAYOUT, height=320,
        xaxis=dict(title="Drought probability", range=[0, 1],
                   showgrid=False, showline=True, linecolor=DARK),
        yaxis=dict(title="Density", showgrid=True, gridcolor="#eee",
                   showline=True, linecolor=DARK),
        legend=dict(x=0.02, y=0.95),
    )
    st.plotly_chart(fig2, use_container_width=True, theme=None)

    t2a, t2b, t2c = st.columns(3)
    t2a.metric("Bootstrap 95% CI", f"[{boot_lo2:.2f}, {boot_hi2:.2f}]")
    t2b.metric("CLT Wald 95% CI",  f"[{wald_lo:.2f}, {wald_hi:.2f}]")
    t2c.metric("Bayesian 95% CrI", f"[{bayes_lo:.2f}, {bayes_hi:.2f}]")

    st.caption(
        f"All three methods agree closely: the SJV is in drought (SPI-12 < −0.5) roughly "
        f"{phat:.0%} of months. The Bayesian credible interval uses a flat Beta(1,1) prior "
        f"(no prior knowledge), so it's nearly identical to the frequentist intervals."
    )

    # ── Analysis 3: Difference in mean precipitation ─────────────────────────
    st.markdown("#### 3. Difference in mean precipitation: drought vs non-drought months")

    diff_obs = drought_precip.mean() - normal_precip.mean()
    clt_diff_lo = diff_obs - 1.96 * np.sqrt(
        drought_precip.std(ddof=1)**2 / len(drought_precip) +
        normal_precip.std(ddof=1)**2  / len(normal_precip)
    )
    clt_diff_hi = diff_obs + 1.96 * np.sqrt(
        drought_precip.std(ddof=1)**2 / len(drought_precip) +
        normal_precip.std(ddof=1)**2  / len(normal_precip)
    )
    boot_lo3, boot_hi3 = np.percentile(boot_diff, [2.5, 97.5])

    fig3 = go.Figure()
    fig3.add_trace(go.Histogram(
        x=boot_diff, nbinsx=60,
        histnorm="probability density",
        marker=dict(color="#FFA94D", line=dict(color="white", width=0.5)),
        name="Bootstrap distribution of difference",
    ))
    for lo, hi, col, label in [
        (boot_lo3, boot_hi3, "#3978AE", "Bootstrap"),
        (clt_diff_lo, clt_diff_hi, "#A60000", "CLT"),
    ]:
        fig3.add_vline(x=lo, line_dash="dash", line_color=col, line_width=1.5,
                       annotation_text=f"{label} lo", annotation_position="top left",
                       annotation_font=dict(color=col, size=10))
        fig3.add_vline(x=hi, line_dash="dash", line_color=col, line_width=1.5)
    fig3.add_vline(x=diff_obs, line_color="#222", line_width=2,
                   annotation_text=f"Δ = {diff_obs:.0f} mm",
                   annotation_position="top right")
    fig3.add_vline(x=0, line_color="#bbb", line_width=1)
    fig3.update_layout(
        **PLOTLY_LAYOUT, height=300,
        xaxis=dict(title="Mean precip (drought) − mean precip (non-drought), mm",
                   showgrid=False, showline=True, linecolor=DARK),
        yaxis=dict(title="Density", showgrid=True, gridcolor="#eee",
                   showline=True, linecolor=DARK),
    )
    st.plotly_chart(fig3, use_container_width=True, theme=None)

    t3a, t3b = st.columns(2)
    t3a.metric("Bootstrap 95% CI", f"[{boot_lo3:.0f}, {boot_hi3:.0f}] mm")
    t3b.metric("CLT 95% CI",       f"[{clt_diff_lo:.0f}, {clt_diff_hi:.0f}] mm")

    st.caption(
        f"Drought months receive {abs(diff_obs):.0f} mm less precipitation than non-drought months "
        f"on average. The entire 95% CI is {'below' if boot_hi3 < 0 else 'above or crossing'} zero, "
        f"meaning the difference is statistically significant."
    )


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 5 — INFORMATION CONTENT  (Shannon entropy + KL divergence)
# ═══════════════════════════════════════════════════════════════════════════════
with tab_info:
    st.markdown("### Information Content — Entropy, KL Divergence, Mutual Information")
    st.caption(
        "How much information does precipitation carry about drought? "
        "We use Shannon entropy to measure how 'predictable' each year's drought pattern was, "
        "KL divergence to quantify how different drought-year precipitation looks from normal years, "
        "and mutual information to measure how much precipitation and ET tell us about each other."
    )
    with st.expander("Statistical details"):
        st.markdown(r"""
**Shannon entropy** (expected log-surprise):
$$H(X) = -\sum_i p_i \log p_i = -\mathbb{E}[\log p(X)]$$
$H = 0$ when all probability mass is on one category (perfectly predictable);
$H$ is maximised when all categories are equally likely (maximum uncertainty).

**KL divergence** (information gain from P to Q):
$$D_{KL}(P \| Q) = \sum_i P(i) \log \frac{P(i)}{Q(i)}$$
Higher KL = drought precipitation looks very different from normal.

**Mutual information** (KL divergence between joint and product of marginals):
$$I(X;\, Y) = \sum_{x,y} p(x,y) \log \frac{p(x,y)}{p(x)\,p(y)}$$
$I = 0$ → completely independent; $I > 0$ → knowing one variable tells you something about the other.

**Connection to MLE** (CS109 link): maximising likelihood equals minimising KL divergence from
the empirical distribution to the model — so MLE and information theory are two sides of the same coin.
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
    st.markdown("#### 1. Shannon entropy of drought-category distribution, per year")

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
        hovertemplate="Year %{x}<br>H = %{y:.3f} nats<extra></extra>",
    ))
    fig_ent.add_hline(y=H_max, line_dash="dot", line_color="#888",
                      annotation_text="Max entropy (uniform over 6 categories)",
                      annotation_position="top right",
                      annotation_font=dict(size=11, color="#888"))
    fig_ent.update_layout(
        **PLOTLY_LAYOUT, height=320,
        xaxis=dict(title="Year", type="category",
                   showgrid=False, showline=True, linecolor=DARK,
                   tickfont=dict(color=DARK)),
        yaxis=dict(title="Shannon entropy (nats)",
                   showgrid=True, gridcolor="#eee",
                   showline=True, linecolor=DARK,
                   range=[0, H_max * 1.1]),
    )
    st.plotly_chart(fig_ent, use_container_width=True, theme=None)

    driest_yr  = years_list[int(np.argmin(H_per_year))]
    wettest_yr = years_list[int(np.argmax(H_per_year))]
    st.caption(
        f"Severe drought years like **{driest_yr}** have *low* entropy — the valley was stuck in "
        f"one category almost all year (H ≈ {min(H_per_year):.2f} nats). "
        f"Transition years like **{wettest_yr}** have *high* entropy — the SJV moved through "
        f"many different drought categories (H ≈ {max(H_per_year):.2f} nats, max possible = {H_max:.2f}). "
        "Paradoxically, drought years are more *predictable* (lower entropy) than average years."
    )

    # ── Analysis 2: KL divergence, drought vs normal precipitation ───────────
    st.markdown("#### 2. KL divergence between drought-year and normal-year precipitation")

    N_BINS = 20
    precip_aligned = precip_raw.values[:len(spi12)]   # align lengths
    drought_mask_arr = spi12 < DROUGHT_THRESH
    p_drought = precip_aligned[drought_mask_arr]
    p_normal  = precip_aligned[~drought_mask_arr]

    bin_edges = np.linspace(
        min(p_drought.min(), p_normal.min()),
        max(p_drought.max(), p_normal.max()),
        N_BINS + 1,
    )
    P, _ = np.histogram(p_drought, bins=bin_edges, density=True)
    Q, _ = np.histogram(p_normal,  bins=bin_edges, density=True)
    bin_w = bin_edges[1] - bin_edges[0]
    P = P * bin_w + 1e-10   # convert density → probability, add eps
    Q = Q * bin_w + 1e-10
    P /= P.sum(); Q /= Q.sum()
    kl_pq = float(np.sum(P * np.log(P / Q)))
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    fig_kl = go.Figure()
    fig_kl.add_trace(go.Bar(
        x=bin_centers, y=P, name="Drought months",
        marker=dict(color="rgba(166,0,0,0.55)", line=dict(color="#A60000", width=1)),
        width=bin_w * 0.45,
    ))
    fig_kl.add_trace(go.Bar(
        x=bin_centers + bin_w * 0.46, y=Q, name="Non-drought months",
        marker=dict(color="rgba(57,120,174,0.55)", line=dict(color="#3978AE", width=1)),
        width=bin_w * 0.45,
    ))
    fig_kl.update_layout(
        **PLOTLY_LAYOUT, height=320,
        xaxis=dict(title="Monthly precipitation (mm)",
                   showgrid=False, showline=True, linecolor=DARK),
        yaxis=dict(title="Probability",
                   showgrid=True, gridcolor="#eee", showline=True, linecolor=DARK),
        legend=dict(x=0.65, y=0.95),
        barmode="overlay",
        annotations=[dict(
            x=0.97, y=0.92, xref="paper", yref="paper",
            text=f"KL(Drought ‖ Normal) = {kl_pq:.3f} nats",
            showarrow=False, bgcolor="rgba(255,255,255,0.85)",
            bordercolor="#ccc", borderwidth=1, borderpad=6,
            font=dict(size=13, color=DARK),
        )],
    )
    st.plotly_chart(fig_kl, use_container_width=True, theme=None)

    st.caption(
        f"KL divergence = {kl_pq:.3f} nats. The precipitation distributions for drought months "
        "and non-drought months are shifted apart — drought months cluster at lower totals. "
        "Higher KL means the two distributions look more different."
    )

    # ── Analysis 3: Mutual information between SPI-12 and ETI-12 ────────────
    st.markdown("#### 3. Mutual information between SPI-12 and ETI-12")

    N_BINS_MI = 12
    spi_edges = np.linspace(spi12.min(), spi12.max(), N_BINS_MI + 1)
    eti_edges = np.linspace(eti12.min(), eti12.max(), N_BINS_MI + 1)

    joint, _, _ = np.histogram2d(spi12, eti12, bins=[spi_edges, eti_edges])
    joint = joint / joint.sum()
    p_spi = joint.sum(axis=1, keepdims=True)
    p_eti = joint.sum(axis=0, keepdims=True)

    # MI = sum p(x,y) log p(x,y) / (p(x)p(y))
    with np.errstate(divide="ignore", invalid="ignore"):
        mi_mat = np.where(
            joint > 0,
            joint * np.log(joint / (p_spi * p_eti + 1e-15)),
            0.0,
        )
    MI = float(mi_mat.sum())

    fig_mi = go.Figure(go.Heatmap(
        z=joint.T,
        x=0.5 * (spi_edges[:-1] + spi_edges[1:]),
        y=0.5 * (eti_edges[:-1] + eti_edges[1:]),
        colorscale=[[0, "#F2F2F2"], [1, "#0A3D66"]],
        colorbar=dict(title=dict(text="Joint prob", side="right"),
                      tickfont=dict(color=DARK, size=11)),
        hovertemplate="SPI-12 %{x:.2f}<br>ETI-12 %{y:.2f}<br>p = %{z:.4f}<extra></extra>",
    ))
    fig_mi.update_layout(
        **PLOTLY_LAYOUT, height=360,
        xaxis=dict(title="SPI-12 (precipitation anomaly, σ)",
                   showgrid=False, showline=True, linecolor=DARK),
        yaxis=dict(title="ETI-12 (ET anomaly, σ)",
                   showgrid=False, showline=True, linecolor=DARK),
        annotations=[dict(
            x=0.97, y=0.05, xref="paper", yref="paper",
            text=f"I(SPI-12; ETI-12) = {MI:.3f} nats",
            showarrow=False, bgcolor="rgba(255,255,255,0.85)",
            bordercolor="#ccc", borderwidth=1, borderpad=6,
            font=dict(size=13, color=DARK),
        )],
    )
    st.plotly_chart(fig_mi, use_container_width=True, theme=None)

    st.caption(
        f"Mutual information I(SPI-12; ETI-12) = {MI:.3f} nats. "
        "The joint distribution is concentrated along the diagonal — when precipitation is anomalously "
        "low, ET is also low (less water available to evaporate). The positive MI confirms that "
        "precipitation and ET are not independent: knowing one tells us something about the other."
    )
