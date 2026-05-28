# math.md — Probability Derivations

Full mathematical story for the writeup. Three stages: MLE → Monte Carlo → Bayesian Updating.

---

## Setup

Observe well water table depth over time (monthly readings, in feet):
X_0, X_1, X_2, ..., X_n

Deeper = worse. Well **fails** when X_t >= X_fail (the well casing depth).
X_fail is a fixed physical threshold per well (known from well construction records or USGS metadata).

---

## Stage 1 — MLE: Learn the Model

### Model
Month-over-month changes are i.i.d. Gaussian:

  Δt = X_t - X_{t-1} ~ N(μ, σ²)

Where:
- μ = average monthly drift (positive = water table dropping = depleting)
- σ² = variance of fluctuations (seasonal noise, measurement error)

### MLE Derivation
Log-likelihood of observing differences Δ_1, ..., Δ_n:

  ℓ(μ, σ²) = Σ log p(Δt | μ, σ²)
            = Σ [ -½log(2πσ²) - (Δt - μ)² / (2σ²) ]

Take partial derivatives and set to zero:

  ∂ℓ/∂μ = 0  →  μ̂ = (1/n) Σ Δt  =  Δ̄  (sample mean)

  ∂ℓ/∂σ² = 0  →  σ̂² = (1/n) Σ (Δt - μ̂)²  (biased MLE variance)

These are closed form — include this derivation in the writeup.

### Interpretation
- μ̂ > 0: well is depleting on average
- μ̂ ≈ 0: well is stable
- μ̂ < 0: well is recharging
- σ̂ captures seasonal swings and measurement noise

Fit one (μ̂, σ̂) pair per well from its full historical USGS record.

---

## Stage 2 — Monte Carlo: Estimate Future Failure Probability

### Target Quantity
We want:

  P(fail within k months) = P(∃ t ≤ k : X_t >= X_fail | X_0 = x_current)

This is a **first passage time probability**. No closed-form solution exists
for the general case (absorbing barrier problem with drift).

### Monte Carlo Estimator
Simulate N independent trajectories. For each simulation i:

  X_{t+1}^(i) = X_t^(i) + μ̂ + ε_t,   ε_t ~ N(0, σ̂²)

Count how many hit X_fail before month k:

  P̂(fail within k) = (1/N) Σ_{i=1}^N 1[simulation i crosses X_fail before month k]

### Justification (CS109)
By the Law of Large Numbers:
  P̂ → P(fail within k)  as N → ∞

Each indicator 1[...] is a Bernoulli(p) random variable. The sample mean
is an unbiased estimator of p. Standard error = sqrt(p(1-p)/N).

Use N = 10,000 for stable estimates (SE < 0.005 for most p values).

### Why Not Analytical?
The exact distribution of the first passage time T = min{t : X_t >= X_fail}
for a Gaussian random walk with drift involves the inverse Gaussian distribution
and is complex. Monte Carlo is the standard engineering approach.

---

## Stage 3 — Bayesian Updating: Revise as New Data Arrives

### Motivation
μ̂ was fit on historical data. As new monthly readings arrive, we should
update our belief about the true drift rate — especially if the well's
behavior seems to be changing (e.g. accelerating depletion).

### Prior
Before seeing new data, our belief about μ:

  μ ~ N(μ₀, τ²)

Set μ₀ = μ̂ (from MLE) and τ² reflects uncertainty in that estimate.
A reasonable choice: τ² = σ̂²/n (standard error of the MLE estimate squared).

### Likelihood
Each new observation Δ_new:

  Δ_new | μ ~ N(μ, σ²)

### Posterior (Gaussian Conjugate Update)
After observing m new differences with sample mean Δ̄_new:

  μ | data ~ N(μ_post, σ²_post)

Where:

  μ_post = (μ₀/τ² + m*Δ̄_new/σ²) / (1/τ² + m/σ²)

  σ²_post = 1 / (1/τ² + m/σ²)

### Interpretation
- μ_post is a precision-weighted average of prior belief and new data
- As m → ∞, μ_post → Δ̄_new (data dominates)
- As τ² → ∞ (weak prior), μ_post → Δ̄_new immediately
- σ²_post shrinks as more data arrives (uncertainty decreases)

### Plug Back Into Monte Carlo
After each new monthly reading, update μ_post and σ²_post, then re-run
Monte Carlo with the posterior mean as the drift parameter:

  X_{t+1} = X_t + μ_post + ε_t,   ε_t ~ N(0, σ̂²)

This gives an updated P(fail within k months) every month.

---

## Simulation Strategy for Demo (Key Design Decision)

Since USGS data updates too slowly (monthly at best) to demonstrate live
Bayesian updating within a one-week project timeline, we use a train/test split:

- **Train:** 2000–2021 historical data → fit (μ̂, σ̂) via MLE
- **Test (replay):** 2022–2024 data → feed in month by month as "new arrivals"

This demonstrates the full pipeline honestly. In the writeup, describe as:
"We simulate sequential Bayesian updating by replaying held-out observations
chronologically, demonstrating how the system would perform in deployment."

---

## Summary of CS109 Concepts Used

| Concept | Where |
|---|---|
| Gaussian distribution | Core noise model for Δt |
| MLE | Closed-form estimates of μ̂ and σ̂² |
| Law of Large Numbers | Justifies Monte Carlo convergence |
| CLT | Justifies Gaussian noise model for aggregated errors |
| Bayesian inference | Posterior update on μ |
| Gaussian conjugate prior | Clean closed-form posterior derivation |
