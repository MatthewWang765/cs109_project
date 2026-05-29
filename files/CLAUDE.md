# San Joaquin Valley Drought Regime HMM — CS109 Challenge Project

## Project Goal
Fit a 4-state Gaussian HMM to daily gridded precipitation + evapotranspiration data
for the San Joaquin Valley (2000–2020) to discover hidden climate/drought regimes.
This is a CS109 Challenge submission — all probability must be implemented from scratch.

## Academic Constraint (CRITICAL)
This is submitted to Stanford CS109. Do NOT use sklearn's HMM, hmmlearn, or any library
that implements Baum-Welch or Viterbi internally. We implement everything ourselves:
- Forward-backward algorithm
- Baum-Welch (EM) for parameter estimation
- Viterbi for decoding
- Multivariate Gaussian emission likelihoods

numpy and scipy.stats.multivariate_normal are allowed (for numerical stability only).
matplotlib and pandas are allowed for data loading and visualization.

## Model Specification
- K = 4 hidden states (climate regimes, labels learned post-hoc)
- Observations: X_t = [precip_t, ET_t] in R^2, one per day
- Initial distribution π: uniform [0.25, 0.25, 0.25, 0.25]
- Transition matrix A: 4x4, rows sum to 1
- Emission: X_t | Z_t=k ~ N(μ_k, Σ_k), multivariate Gaussian
- Parameters learned via Baum-Welch (EM)

## Project Structure
```
sjv_drought_hmm/
├── CLAUDE.md               ← you are here
├── data/
│   └── README.md           ← data download instructions
├── src/
│   ├── hmm.py              ← GaussianHMM class (core model)
│   ├── forward_backward.py ← E-step: alpha, beta, gamma, xi
│   ├── viterbi.py          ← decoding most likely state sequence
│   ├── emissions.py        ← multivariate Gaussian log-likelihood
│   └── data_loader.py      ← load/preprocess NetCDF or CSV data
├── notebooks/
│   └── exploration.ipynb   ← EDA and results visualization
├── outputs/
│   └── (figures, learned params, decoded sequences)
├── tests/
│   └── test_hmm.py         ← unit tests for each algorithm
└── main.py                 ← train model, decode, save outputs
```

## Key Equations to Implement (do not deviate)

### Emission log-likelihood
log P(X_t | Z_t=k) = log N(X_t; μ_k, Σ_k)
Use log-space throughout to avoid underflow.

### Forward pass (α)
α_1(k) = π_k * N(X_1; μ_k, Σ_k)
α_t(k) = N(X_t; μ_k, Σ_k) * Σ_j [α_{t-1}(j) * A_{jk}]
Scale at each step (Rabiner scaling) to avoid underflow.

### Backward pass (β)
β_T(k) = 1
β_t(k) = Σ_j [A_{kj} * N(X_{t+1}; μ_j, Σ_j) * β_{t+1}(j)]
Use same scaling factors from forward pass.

### E-step posteriors
γ_t(k) = α_t(k) * β_t(k) / Σ_j [α_t(j) * β_t(j)]
ξ_t(j,k) = α_t(j) * A_{jk} * N(X_{t+1}; μ_k, Σ_k) * β_{t+1}(k) / normalizer

### M-step updates
π_k       = γ_1(k)
A_{jk}    = Σ_t ξ_t(j,k) / Σ_t Σ_k ξ_t(j,k)
μ_k       = Σ_t γ_t(k) * X_t / Σ_t γ_t(k)
Σ_k       = Σ_t γ_t(k) * (X_t - μ_k)(X_t - μ_k)^T / Σ_t γ_t(k)

### Viterbi
delta_1(k) = log π_k + log N(X_1; μ_k, Σ_k)
delta_t(k) = log N(X_t; μ_k, Σ_k) + max_j [delta_{t-1}(j) + log A_{jk}]
Backtrack psi pointers to recover Z*_{1:T}.

## Convergence
Run Baum-Welch until |log-likelihood change| < 1e-4 or max 200 iterations.
Log the log-likelihood at each iteration (should monotonically increase).

## Initialization Strategy
Initialize μ_k with k-means on the raw observations (avoids label-switching).
Initialize Σ_k as identity matrices.
Initialize A as near-uniform with small random noise (rows must sum to 1).

## Numerical Notes
- Always work in log-space for likelihoods
- Use Rabiner scaling in forward-backward (not log-sum-exp, simpler to implement)
- Add 1e-6 * I regularization to Σ_k estimates to prevent singular matrices
- Clip γ values away from zero before division

## Output Goals
1. Learned parameters: π, A, {μ_k, Σ_k} for k=1..4
2. Decoded regime sequence Z*_{1:T} over 2000–2020
3. Regime interpretation: which state = wet / dry / hot-dry / moderate?
4. Transition matrix heatmap
5. Regime duration statistics (mean days per regime)
6. Seasonal regime distribution (which regimes dominate which months?)
