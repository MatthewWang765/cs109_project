# Detecting California Drought with a Hidden Markov Model

**CS109 Challenge Project · Spring 2026 · Matthew Wang**

---

## 1 · Introduction

Between 2012 and 2017, California suffered its worst drought in 1,200 years. $2.7 billion in agricultural losses, mass tree die-off across the Sierra, and the deepest reservoir drawdowns in modern record. The hardest-hit region was the San Joaquin Valley (SJV), which produces a quarter of America's food.

**This project asks a deliberately constrained question: can we detect droughts like this *from scratch*, using only precipitation and evapotranspiration data, by building a Hidden Markov Model on top of the Bayesian inference foundations from CS109?**

I built an interactive Streamlit dashboard that fits a 4-state Gaussian HMM to monthly SJV climate data (2000–2020), discovers four latent climate regimes, and visualises a continuous Bayesian drought-severity score against the historical record. The result is climatologically faithful — the model independently identifies the 2007–09 drought, the 2014–15 mega-drought peak, the 2011 La Niña pluvial, and the 2017 atmospheric-river relief year, all from a 2-D observation per month.

The coolest part: every probabilistic primitive — Baum-Welch (EM for HMMs), forward-backward, Viterbi, Gaussian emissions — is implemented from scratch in NumPy. No `hmmlearn`, no `sklearn`. Pure CS109 fundamentals plus the minimum extensions needed for time-series structure.

## 2 · Data

Two NASA / OpenET satellite products, both gridded across a 38 × 35 cell rectangle covering the SJV:

| Source | Variable | Native Resolution | Coverage |
|---|---|---|---|
| NASA GPM IMERG Late-Run | Daily precipitation (mm) | 10 km, daily | Jan 2000 – Dec 2020 |
| OpenET Monthly Ensemble | Evapotranspiration (mm) | 500 m, monthly | Jan 2000 – Dec 2020 |

**Preprocessing.** Daily GPM precipitation is aggregated to monthly totals. Both series are then converted into 12-month rolling sums (the SPI-12 and ETI-12 standard climatological indices) and z-scored *within each calendar month* — i.e., every January is standardised against the 21 other Januaries. This removes the seasonal cycle so the HMM doesn't waste states modelling "summer vs. winter." Final input: 241 monthly observations in $\mathbb{R}^2$.

## 3 · Methods

The HMM extends three CS109 primitives — **Bayes' rule, MLE, and the 1-D normal distribution** — into a time-series setting.

### 3.1 · Model specification

For each month $t = 1, \dots, T$, the SJV is in a hidden regime $Z_t \in \{1, 2, 3, 4\}$ and emits an observation $X_t = (\text{SPI-12}_t, \text{ETI-12}_t) \in \mathbb{R}^2$. The joint factors as

$$
\Pr(X_{1:T}, Z_{1:T} \mid \theta) = \Pr(Z_1) \prod_{t=2}^{T} \underbrace{\Pr(Z_t \mid Z_{t-1})}_{A_{Z_{t-1}, Z_t}} \prod_{t=1}^{T} \underbrace{\Pr(X_t \mid Z_t)}_{\mathcal{N}(X_t; \mu_{Z_t}, \Sigma_{Z_t})}
$$

with parameters $\theta = (\pi, A, \{\mu_k, \Sigma_k\}_{k=1}^4)$:
- $\pi$: initial distribution (uniform).
- $A \in \mathbb{R}^{4 \times 4}$: transition matrix. $A_{jk} = \Pr(Z_t = k \mid Z_{t-1} = j)$. Each row sums to 1 — this is the **Markov assumption**, an iterated chain of conditional probabilities directly extending CS109 conditional probability.
- $\mu_k, \Sigma_k$: emission mean and covariance of a **2-D multivariate Gaussian** — the natural extension of the 1-D normal we learned in class to two correlated variables.

### 3.2 · Why this model

Drought is a *latent* state — it's not directly observed, but it leaves an observable trace in precipitation and ET. HMMs are the canonical generative model for this exact setting: discrete unobserved state, continuous noisy observations, temporal persistence. I considered three alternatives:

1. **GMM (no time)** — clustering each month independently. Tried first; produced incoherent month-to-month classifications because it ignores that drought *persists*. Scrapped.
2. **Logistic regression on hand-engineered features** — requires labelled drought months, which is what we're trying to discover. Scrapped.
3. **Continuous SPEI threshold** — a fixed cutoff with no statistical model. Doesn't capture multi-month structure or give us uncertainty.

The HMM uniquely gives us all three: latent structure, temporal persistence (via $A$), and full Bayesian uncertainty (via the posterior $\gamma_t$).

### 3.3 · Implementation (everything from scratch)

I implemented three classical algorithms in NumPy, each derived from CS109 primitives:

**Baum-Welch (parameter learning).** The EM algorithm for HMMs. EM is **MLE under latent variables** — directly building on the maximum-likelihood estimation from class. Each iteration:
- **E-step**: compute posterior $\gamma_t(k) = \Pr(Z_t = k \mid X_{1:T}, \theta)$ via forward-backward.
- **M-step**: closed-form weighted updates: $\mu_k = \sum_t \gamma_t(k) X_t / \sum_t \gamma_t(k)$, similarly for $\Sigma_k$, and $A_{jk} \propto \sum_t \xi_t(j, k)$.

**Forward-backward (inference).** Computes the smoothed posterior using the *entire* observation sequence — applying Bayes' rule iteratively forward and backward:
$$
\alpha_t(k) = \Pr(X_{1:t}, Z_t = k), \quad \beta_t(k) = \Pr(X_{t+1:T} \mid Z_t = k), \quad \gamma_t(k) \propto \alpha_t(k) \beta_t(k)
$$
Implemented with Rabiner scaling for numerical stability (the unscaled values underflow within a few timesteps).

**Viterbi (decoding).** Maximum a posteriori state sequence, $z^*_{1:T} = \arg\max_{z_{1:T}} \Pr(z_{1:T} \mid X_{1:T})$. Computed in log-space via the recursion $\delta_t(k) = \max_j [\delta_{t-1}(j) + \log A_{jk}] + \log \mathcal{N}(X_t; \mu_k, \Sigma_k)$. **MAP estimation** is a CS109 concept; Viterbi is its dynamic-programming realisation for chains.

**Initialisation**: k-means on $X_t$ for $\mu_k$, identity for $\Sigma_k$, near-uniform $A$. Convergence: $|\Delta \log \mathcal{L}| < 10^{-4}$ or 200 iterations.

### 3.4 · The CS109 / extension boundary

| CS109 primitive | How it's used here |
|---|---|
| Bayes' rule, conditional probability | Posterior $\Pr(Z_t \mid X_{1:T})$, transitions $\Pr(Z_t \mid Z_{t-1})$ |
| 1-D normal distribution | Generalised to 2-D multivariate normal for emissions |
| MLE | Foundation of EM / Baum-Welch parameter learning |
| MAP estimation | Realised by the Viterbi algorithm |
| Joint and marginal probabilities | Forward / backward recursions |
| Independence (here: conditional independence of $X_t$ given $Z_t$) | Lets emissions factor across time |

**Extensions I had to learn**: Markov chains (iterated conditional probability), the multivariate Gaussian, the EM algorithm in general, and its specialisation to HMMs (forward-backward + Baum-Welch).

## 4 · Results

The interactive dashboard (Streamlit) presents four views; all results below are from the converged HMM.

### 4.1 · Validation against ground-truth drought events

The model's Bayesian drought-intensity score $\text{DI}(t) = \gamma_t(\text{Drought}) + 0.4 \cdot \gamma_t(\text{Hot})$ matches the U.S. Drought Monitor's historical record:

| Year | $\overline{\text{DI}}$ | USDM category | Real-world event |
|---|---|---|---|
| 2008 | **0.99** | D4 Exceptional | 2007–09 California drought peak ✓ |
| 2009 | **1.00** | D4 Exceptional | 2007–09 drought ✓ |
| 2011 | 0.00 | None | La Niña pluvial; biggest wet year in decade ✓ |
| 2014 | **0.99** | D4 Exceptional | Mega-drought peak; driest 12-month period on record ✓ |
| 2015 | 0.71 | D2 Severe | Mega-drought tail ✓ |
| 2017 | 0.03 | None | Atmospheric rivers ended mega-drought ✓ |
| 2020 | 0.40 | D1 Moderate | Onset of 2020–22 drought (truncated by data window) |

No drought labels were given to the model. It discovered these regimes from precipitation and ET alone.

### 4.2 · Spatial structure

I also compute a per-cell percentile-rank drought map using the same methodology as the U.S. Drought Monitor (non-parametric ranking of each cell's 12-month precip total against its own historical record). This is a complementary, non-HMM view that provides spatial localisation. Animated across 241 months it cleanly reproduces the spatial drought patterns documented in NOAA's archives.

### 4.3 · Model diagnostics

- **Convergence**: Baum-Welch's log-likelihood is monotonically increasing across 123 iterations (consistent with the EM theorem) before convergence at $|\Delta \log \mathcal{L}| < 10^{-4}$.
- **Persistence**: diagonal of $A$ averages 0.74 — regimes persist for several months on average, matching climatological reality.
- **Stationary distribution**: $\bar{\pi} = (\text{Wet} \, 15\%, \text{Drought} \, 32\%, \text{Hot} \, 23\%, \text{Normal} \, 30\%)$ — left eigenvector of $A$ with eigenvalue 1.

### 4.4 · An unexpected finding

The HMM's "Hot" regime (high ET, near-normal precipitation) dominates 2018–2020 with probability ≈ 1.0 — a regime that's distinct from both classical drought and normal conditions. This is the climatological signature of recent warming: precipitation hasn't fallen further, but evapotranspirative demand has risen sharply. The model identified this *climate-change-induced regime shift* without being told to look for it — arguably its most interesting result.

## 5 · Discussion & Future Work

**Limitations.** The 21-year baseline includes the drought itself, so per-cell SPI z-scores are partially deflated (a known SPI limitation). The HMM also assumes regime changes are Markov — real climate has longer-memory dependencies (ENSO, PDO). The model can't extrapolate into ungauged regions because it's trained on this specific SJV grid.

**Hurdles.** The biggest was numerical underflow in forward-backward — scaled $\alpha_t$ values shrink rapidly, and naive multiplication underflows by month 30. I implemented Rabiner scaling: rescale $\alpha_t$ to sum to 1 at each step and accumulate $\log$ of the scale factors to recover the true log-likelihood. A second hurdle was an early bug where my model mistook seasonal variance for inter-annual drought — fixed by z-scoring features *within each calendar month*.

**Future work.** A hierarchical HMM with monthly and annual states could model both seasonal-scale and decade-scale persistence. Extending the emission model to include temperature would let the model distinguish "hot drought" from "cool drought" more cleanly.

## 6 · Usage of Generative AI

I used Claude (Anthropic) throughout for: pair-programming Streamlit dashboard scaffolding, debugging Plotly rendering issues, and writing prose. All probability theory, model design, and the from-scratch algorithm implementations (Baum-Welch, forward-backward, Viterbi) I derived from CS109 fundamentals and the Rabiner (1989) tutorial, and reviewed each line line-by-line. AI was a productivity tool, not a substitute for understanding.

## 7 · References

1. Rabiner, L. R. (1989). *A tutorial on hidden Markov models and selected applications in speech recognition*. Proc. IEEE 77(2), 257–286.
2. McKee, T. B., Doesken, N. J., & Kleist, J. (1993). *The relationship of drought frequency and duration to time scales*. AMS 8th Conf. Applied Climatology.
3. Vicente-Serrano, S. M., Beguería, S., & López-Moreno, J. I. (2010). *A multiscalar drought index sensitive to global warming: SPEI*. J. Climate 23(7).
4. NASA GPM IMERG Late-Run V06 — https://gpm.nasa.gov/data/imerg
5. OpenET Ensemble v2.0 — https://etdata.org
6. U.S. Drought Monitor — https://droughtmonitor.unl.edu
7. Williams, A. P. et al. (2020). *Large contribution from anthropogenic warming to an emerging North American megadrought*. Science 368, 314–318.

---

*Code and dashboard: included with submission. Run `streamlit run app.py` to view the interactive results.*
