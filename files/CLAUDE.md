# Statistical Drought Analysis — San Joaquin Valley

## Project goal

Build a Streamlit dashboard that analyses 21 years of San Joaquin Valley precipitation
and evapotranspiration data using **probabilistic methods drawn directly from the
CS109 syllabus**. This is the CS109 Spring 2026 Challenge Project.

CS109 topics that MUST appear in the analysis (this drives everything else):

- Continuous random variables (Normal, Exponential, **Gamma**)
- **MLE** (closed-form for Normal/Exp, numerical for Gamma)
- **Logistic regression** (binary drought classification)
- **Linear regression** (drought-severity prediction)
- **Information theory** — Shannon entropy, KL divergence
- **Bootstrap sampling** + Central Limit Theorem (confidence intervals)
- **Bayesian inference** — Beta-Binomial conjugacy

CS109 topics that are explicitly OUT OF SCOPE:

- Hidden Markov Models, Baum-Welch, Viterbi, forward-backward — these are NOT
  in CS109 and were the basis of a previous (now-abandoned) version of this
  project. If you find HMM references in old code, treat them as legacy and
  remove or replace.
- Multivariate Gaussian, EM algorithm, Markov chains — out of scope.

## Repository layout

```
files/
├── CLAUDE.md            ← you are here
├── PROJECT_PLAN.md      ← the analyses to build (read second)
├── app.py               ← Streamlit dashboard (legacy HMM logic to be replaced)
├── data_loader.py       ← NetCDF loaders (KEEP — these are good)
├── data/
│   ├── gpm_sjv_subset.nc        ← NASA GPM precipitation, daily, 38×35 grid
│   └── openet_sjv_subset.nc     ← OpenET ET, monthly, ~800×700 grid
├── outputs/             ← legacy HMM outputs — safe to overwrite
└── (hmm.py, viterbi.py, forward_backward.py, emissions.py, main.py)
                         ← legacy HMM code. Remove or repurpose.
```

## Data

Two NetCDF files, both already spatially subset to the SJV:

**`data/gpm_sjv_subset.nc`** — NASA GPM IMERG Late-Run precipitation
- Variables: `precipitation` (time, lat, lon) daily mm; `precip_mean` (time) SJV-averaged daily mm
- Grid: 38 lat × 35 lon at 10 km, with ~973 NaN cells outside the SJV polygon
- Coverage: 2000-01-01 → 2020-12-31 (7,671 days)

**`data/openet_sjv_subset.nc`** — OpenET Monthly Ensemble evapotranspiration
- Variables: `et` (time, lat, lon) mm/month; `et_mean` (time) SJV-averaged mm/month
- Grid: 801 lat × 731 lon at 500 m (mismatched with GPM — use the `_mean` variables for spatial-mean work)
- Coverage: 2000-01 → 2020-12 (252 months)

The existing `data_loader.py` already implements:
- `load_netcdf(precip_path, et_path)` — returns monthly SPI-12 / ETI-12 z-score anomalies (T=241 months, 2 features). Useful for any temporal analysis at the monthly level.
- `load_spatial_percentile(precip_path)` — returns per-cell drought percentile ranks (T=241, lat=38, lon=35) with non-SJV cells as NaN. Used by the map.

For the new analyses you will also need raw monthly totals (no z-scoring), which means
adding a simpler loader. Easy lift.

## The Streamlit app to preserve

`app.py` already implements an interactive dashboard with 4 tabs and good UX scaffolding.
**Most of this scaffolding is reusable**; the analyses inside the tabs are what changes.

Reusable infrastructure in `app.py`:

- `PLOTLY_LAYOUT` design tokens for consistent styling
- Page-level CSS (hides Streamlit's running widget during animation, etc.)
- `@st.cache_data` loader wrappers
- Drought Map tab — the spatial heatmap with animation, slider, date picker, calendar
  view of drought %. **This entire tab is CS109-agnostic and should stay as-is.**
- The 4-tab layout pattern, sidebar layout, "Quick Statistics" cards at top
- The `_sjv_outline` cell-edge-tracing helper used for the SJV map boundary
- The `@st.expander("Statistical details")` pattern for showing math under each chart

Reusable utility files:

- `data_loader.py` — keep
- (Anything else in the repo that's HMM-specific can be removed)

## Visual identity

The existing app uses a clean, white-background, professional style. Keep it.

- Font: Inter / system-ui
- Plot backgrounds: white; axes dark `#1a1a1a` / `#222`
- Colour palette for drought categories matches the U.S. Drought Monitor:
  D4 `#5C0000`, D3 `#A60000`, D2 `#E66B00`, D1 `#FFA94D`, D0 `#FFE099`,
  Normal `#F2F2F2`, then wet end blue → navy `#0A3D66`
- Captions are plain English; mathematical details live in `st.expander("Statistical details")` blocks

## How to run

```bash
streamlit run app.py
```

Python deps: `streamlit, plotly, pandas, numpy, xarray, netCDF4, scipy`

## What you (Claude) should do next

Read **`PROJECT_PLAN.md`** for the full pivot plan. In short:

1. Remove or repurpose the HMM tabs (`Drought Severity`, `Latent Regimes`, `Model Internals`).
2. Keep `Drought Map` exactly as-is — it's pure climatology, no HMM.
3. Build the new tabs with CS109 analyses as specified in `PROJECT_PLAN.md`.
4. Each new chart needs a plain-English caption + a `Statistical details` expander
   citing the relevant CS109 concept by name.
5. Rewrite `WRITEUP.md` to reflect the new CS109-grounded analyses.

The end goal: a dashboard where every tab can be pointed at and labelled with a
specific CS109 lecture.
