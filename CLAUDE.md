# CLAUDE.md — Groundwater Well Failure Prediction
## CS109 Stanford Challenge Project

This file gives Claude Code full context to continue building this project.
Read this + all files in `/docs` before writing any code.

---

## Project Goal

Build a probabilistic well failure prediction system using real USGS groundwater data.
This is a submission for the Stanford CS109 Probability for Computer Scientists Challenge (due June 3, 2026).

The system:
1. Fetches real historical groundwater depth data from the USGS API
2. Fits a Gaussian drift model to each well via MLE
3. Estimates P(fail within k months) via Monte Carlo simulation
4. Demonstrates Bayesian updating by replaying held-out months sequentially
5. Visualizes everything in a Streamlit app

---

## CS109 Concepts Used

| Concept | Role in project |
|---|---|
| MLE (Gaussian) | Fit drift rate μ̂ and noise σ̂ from historical Δt values |
| Gaussian distributions | Model month-over-month water level changes |
| Monte Carlo | Estimate first-passage-time probability (no closed form) |
| Bayesian updating | Gaussian conjugate prior on μ, updated with new monthly readings |
| Law of Large Numbers | Justifies Monte Carlo convergence in writeup |
| CLT | Justifies Gaussian noise model in writeup |

Monte Carlo is not explicitly in CS109 but is justified as empirical probability
estimation from Gaussian samples — explainable entirely in CS109 terms.

---

## The Math

See `/docs/math.md` for full derivations. Summary:

**Model:**
Δt = X_t - X_{t-1} ~ N(μ, σ²)

**MLE:**
μ̂ = mean(Δt)
σ̂² = var(Δt)

**Failure probability (Monte Carlo):**
P(fail within k months) ≈ (1/N) * Σ 1[simulation i hits X_fail before month k]

**Bayesian posterior on μ after m new observations:**
μ_post = (μ₀/τ² + m*Δ̄_new/σ²) / (1/τ² + m/σ²)
σ²_post = 1 / (1/τ² + m/σ²)

---

## Project Structure

```
project/
│
├── CLAUDE.md                   # this file
├── docs/
│   ├── math.md                 # full probability derivations
│   ├── data.md                 # USGS API details and well selection
│   └── decisions.md            # key design decisions made in planning
│
├── data/
│   └── raw/                    # downloaded USGS csv files go here
│
├── 1_fetch_data.py             # pull USGS API → save to data/raw/
├── 2_clean_data.py             # parse dates, handle gaps, resample monthly
├── 3_mle.py                    # compute deltas, fit (μ̂, σ̂) per well
├── 4_monte_carlo.py            # simulate trajectories → P(fail within k months)
├── 5_bayesian_update.py        # replay held-out months, update μ_post sequentially
├── 6_visualize.py              # all matplotlib figures
└── app.py                      # Streamlit app tying it all together
```

---

## Tech Stack

- Python
- numpy — simulation and math
- scipy.stats — MLE fitting, distributions
- pandas — time series cleaning
- matplotlib — trajectory plots, probability charts
- streamlit — interactive demo app
- requests — USGS API calls
- folium (optional) — map of wells colored by P(fail)

Do NOT use tensorflow, pytorch, or other advanced ML libraries
(per CS109 challenge rules — would not count as demonstration of CS109 concepts).

---

## Streamlit App Layout

Three panels:

```
┌─────────────────────────────────────────┐
│  SELECT WELL        │  HORIZON SLIDER   │
│  [dropdown]         │  [1–60 months]    │
├─────────────────────────────────────────┤
│                                         │
│   TRAJECTORY PLOT (Monte Carlo fans)    │
│                                         │
├──────────────────┬──────────────────────┤
│  P(fail) = 0.73  │  BAYESIAN UPDATE     │
│  [big number]    │  CHART over time     │
└──────────────────┴──────────────────────┘
```

---

## Submission Requirements

- Demonstration video: ~5 minutes, screen recorded from Streamlit app
- Written writeup: max 3 pages PDF, include math derivations
- Submit on Gradescope by June 3, 2026 11:59pm
- Include how LLMs were used (Claude helped plan and scaffold this project)
- Individual work only, no groups

## Scoring Formula
score = sophistication × (1 + creativity + impact)

This project targets all three: MLE+Bayes+MC = high sophistication,
real USGS data + live updating = creativity, water scarcity = impact.
