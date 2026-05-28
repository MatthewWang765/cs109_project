# decisions.md — Key Design Decisions

Record of important choices made during planning, and why.
Don't relitigate these unless there's a good new reason.

---

## 1. No Logistic Regression

**Decision:** Do not use logistic regression as the core model.

**Why considered:** Natural classifier for binary outcome (fail/not fail).

**Why rejected:**
- Requires labeled training data (wells where outcome is known) — hard to assemble cleanly
- Treats each well as a static snapshot, losing temporal dynamics
- The trajectory story (water table path over time) is more compelling and more CS109

**Alternative considered:** Use logistic regression as a second layer on top of
Monte Carlo output. Rejected as unnecessary complexity for marginal gain.

---

## 2. Monte Carlo Despite Not Being Explicit CS109 Topic

**Decision:** Use Monte Carlo simulation as the core inference engine.

**Justification:** The CS109 challenge rules say concepts not covered should be
"close enough to the material that you could explain it in terms of things we
have learned." Monte Carlo is just:
- Sample from the fitted Gaussian model N times (CS109: sampling from distributions)
- Count how many samples hit the threshold (CS109: empirical probability)
- Average converges by LLN (CS109: Law of Large Numbers)

This is entirely explainable in CS109 terms. Include this justification in writeup.

**Why not analytical:** First passage time for Gaussian random walk with absorbing
barrier has no clean closed form. Monte Carlo is the standard approach.

---

## 3. Train/Test Replay for Bayesian Updating Demo

**Decision:** Demonstrate Bayesian updating by replaying held-out historical data,
not by waiting for live USGS updates.

**Why:** USGS groundwater data updates monthly at best. In a one-week project
timeline, you would get 0–1 new real data points — not enough to show updating.

**How:** Split data at 2021/2022 boundary. Train MLE on 2000–2021.
Replay 2022–2024 month by month as "incoming" data for Bayesian update demo.

**Writeup framing:** "We simulate sequential Bayesian updating by replaying
held-out observations chronologically, demonstrating how the system would perform
in a real deployment scenario."

---

## 4. No Deep Learning

**Decision:** Do not use neural networks despite CS109 covering deep learning.

**Why:** Challenge rules explicitly say advanced libraries (tensorflow, pytorch)
won't be counted as demonstration of CS109 concepts. A pure numpy/scipy
implementation of MLE + Bayes is cleaner, more defensible, and faster to build.

---

## 5. Streamlit for Demo App

**Decision:** Use Streamlit for the interactive demo, not React or Jupyter.

**Why Streamlit over React:** Much faster to build, Python-native, easy screen record.
**Why Streamlit over Jupyter:** More polished for video demo, cleaner UI with sliders.

**Jupyter still used:** For exploratory analysis and figures in the writeup.

---

## 6. California Central Valley as Primary Wells

**Decision:** Focus on California Central Valley aquifer for primary demo wells.

**Why:** Most heavily documented groundwater depletion crisis in the US.
Strong trend signal in the data (μ̂ >> 0). High impact narrative.
Good data availability on USGS.

Include 1–2 wells from healthier regions for contrast.

---

## 7. N = 10,000 Simulations

**Decision:** Use N = 10,000 Monte Carlo simulations per estimate.

**Why:** Standard error of P̂ = sqrt(p(1-p)/N) < 0.005 for all p.
Runs in < 1 second on modern hardware. Stable enough for visualization.
Increase to 100,000 only if final figures look noisy.
