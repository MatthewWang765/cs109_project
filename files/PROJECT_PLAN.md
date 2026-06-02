# Project Plan — CS109 Drought Analysis Dashboard

This file describes exactly what to build. Read `CLAUDE.md` first for context.

## Design principle

Every tab in the dashboard maps to one (or two) CS109 lectures. A grader should be
able to look at any chart and say *"that's MLE"* or *"that's logistic regression"*.

Five tabs total. The first one is preserved verbatim from the previous version;
the other four are new and replace the HMM analyses.

---

## Tab 1 — Drought Map  *(keep as-is)*

This tab is CS109-agnostic and visually impressive. Do not change.

It shows a per-cell **percentile-rank** drought map of the SJV with animation,
date picker, slider, Play/Pause, and a "Drought Calendar" heatmap below.

Already implemented in `app.py` using `load_spatial_percentile()` from
`data_loader.py`. Keep all code in this tab intact.

---

## Tab 2 — Modeling Precipitation  *(NEW — Continuous RVs + MLE)*

**CS109 lectures**: Continuous Random Variables; MLE.

### Story

Monthly precipitation totals in the SJV are right-skewed and bounded below by
zero — they are not Normal-distributed. The Gamma distribution is the
canonical fit for monthly precipitation in climatology, and its parameters
can be estimated by maximum-likelihood estimation, exactly as covered in
CS109.

### What to build

For one selected calendar month (default: January, with a `st.selectbox` to choose),
take all 21 instances of that month's precipitation total across 2000–2020 and:

1. **Empirical histogram** of the data.
2. **Fit a Gamma distribution by MLE**.
   - Use `scipy.stats.gamma.fit(...)` for the numerical MLE.
   - Show the fitted shape $k$ and scale $\theta$ parameters.
   - Overlay the fitted PDF on the histogram.
3. **Compare to a Normal MLE fit** (closed-form: $\hat\mu = \bar x$, $\hat\sigma^2 = \frac{1}{n}\sum (x_i - \bar x)^2$).
   - Visually show how the Normal fits worse (extends below zero, ignores skew).
4. **Goodness-of-fit**: report log-likelihood for both fits side by side.
   - Higher log-likelihood → better fit.
   - Gamma should win for winter months especially.

### Statistical details expander

Cover:

- The Gamma PDF: $f(x; k, \theta) = \frac{x^{k-1} e^{-x/\theta}}{\theta^k \Gamma(k)}$ for $x > 0$.
- The MLE objective: $\hat\theta_{\text{MLE}} = \arg\max_\theta \prod_i f(x_i; \theta) = \arg\max_\theta \sum_i \log f(x_i; \theta)$.
- For the Normal, the MLE has closed form; for the Gamma, there is no
  closed form — it requires numerical optimisation (Newton-Raphson on the
  digamma function). Note that `scipy.stats.gamma.fit` does this for you.
- Why this matters climatologically: the Standardised Precipitation Index
  (SPI) used by the U.S. Drought Monitor is literally defined as
  "fit a Gamma, transform to standard Normal via the CDF."

### Optional bonus: SPI computation

If time allows, add a small section showing how the fitted Gamma is used
to compute SPI:
$$
\text{SPI}(x) = \Phi^{-1}(F_{\text{Gamma}}(x))
$$
where $F_{\text{Gamma}}$ is the fitted CDF and $\Phi^{-1}$ is the standard
Normal quantile function. This is the actual operational definition of SPI.

---

## Tab 3 — Predicting Drought  *(NEW — Logistic Regression)*

**CS109 lectures**: Logistic Regression; Comparing Classifiers.

### Story

Can we predict whether next month will be a drought month from this month's
precipitation and ET?

### What to build

1. **Construct features**: for each month $t$, features are
   - $x_1$ = SPI-12 anomaly at month $t$ (precipitation z-score)
   - $x_2$ = ETI-12 anomaly at month $t$ (ET z-score)
   - $x_3$ = calendar month sin (or one-hot encoding)
2. **Construct labels**: $y_t = 1$ if month $t+1$ is in drought (SPI-12 < some threshold, e.g. $-0.5$), else 0.
3. **Train/test split**: e.g. train on 2000–2015, test on 2016–2020.
4. **Fit logistic regression** — implement from scratch using gradient ascent on the log-likelihood:
   $$
   \ell(\beta) = \sum_i \big[ y_i \log \sigma(\beta^\top x_i) + (1 - y_i) \log (1 - \sigma(\beta^\top x_i)) \big]
   $$
   Update rule: $\beta \leftarrow \beta + \eta \sum_i (y_i - \sigma(\beta^\top x_i)) x_i$.
5. **Evaluation**:
   - Confusion matrix on test set
   - Accuracy, precision, recall, F1
   - ROC curve and AUC
6. **Coefficient interpretation**: which feature was most predictive? Show $\beta$ values with sign.

### Visualisations

- Training loss curve (log-likelihood per iteration during gradient ascent)
- Decision boundary plot in 2D (SPI-12 vs. ETI-12)
- Confusion matrix heatmap
- ROC curve

### Statistical details expander

- Logistic regression as a probabilistic classifier:
  $\Pr(y = 1 \mid x) = \sigma(\beta^\top x) = \frac{1}{1 + e^{-\beta^\top x}}$.
- MLE for logistic regression has no closed form → gradient ascent.
- The gradient $\nabla_\beta \ell = \sum_i (y_i - \sigma(\beta^\top x_i)) x_i$ is a key CS109 derivation.
- Why no closed form: the log-likelihood is concave but not quadratic.

---

## Tab 4 — Quantifying Uncertainty  *(NEW — Bootstrap + CLT)*

**CS109 lectures**: Sampling & Bootstrapping; Central Limit Theorem.

### Story

How confident should we be in claims like *"the SJV is in drought 32% of the
time"* or *"average January precipitation is 60 mm"*? Use bootstrap and CLT
to put confidence intervals on every statistic in the dashboard.

### What to build

Three side-by-side analyses, all with 95% confidence intervals computed
two ways (bootstrap + CLT) for direct comparison:

1. **Mean monthly precipitation** (overall)
   - Bootstrap: resample 241 months with replacement 10,000 times, take the mean of each. The 2.5th and 97.5th percentiles of the bootstrap distribution form the 95% CI.
   - CLT: $\bar x \pm 1.96 \cdot s / \sqrt{n}$.
   - Show both intervals overlaid on a histogram of bootstrap means.
2. **Probability of drought in a random month** (a Bernoulli proportion)
   - Bootstrap: resample binary drought indicators, compute proportion.
   - CLT: Normal approximation to the Binomial (Wald interval).
   - **Bayesian alternative**: use Beta-Binomial conjugacy with a Beta(1,1) prior:
     posterior is Beta($s + 1$, $n - s + 1$). Plot the posterior PDF and read off
     the central 95% credible interval.
3. **Difference in mean precipitation between drought and non-drought years**
   - Bootstrap a difference-of-means.
   - CLT-based difference-of-means: $(\bar x_1 - \bar x_2) \pm 1.96 \sqrt{s_1^2/n_1 + s_2^2/n_2}$.

### Visualisations

- Bootstrap distribution histogram with CI bars (matches the CS109 lecture style)
- Normal approximation overlay showing CLT in action
- Comparison table: bootstrap CI vs. CLT CI vs. Bayesian credible interval (where applicable)

### Statistical details expander

- Bootstrap: nonparametric, makes no distributional assumption.
- CLT: $\bar X \xrightarrow{d} \mathcal{N}(\mu, \sigma^2/n)$ as $n \to \infty$.
- Beta-Binomial conjugacy: if prior is Beta($\alpha, \beta$) and likelihood is
  Binomial($n, p$) with $s$ successes, posterior is Beta($\alpha + s$, $\beta + n - s$).
- When each method is appropriate, and what their assumptions are.

---

## Tab 5 — Information Content  *(NEW — Information Theory)*

**CS109 lectures**: Information Theory + MLE.

### Story

How much *information* does precipitation alone give us about drought? How does
the predictability of drought change year to year? Use Shannon entropy and KL
divergence to quantify this.

### What to build

1. **Entropy of monthly drought-category distribution per year**.
   - For each year, count how many months fell in each of the 6 USDM categories (None, D0, D1, D2, D3, D4).
   - Compute $H(Y) = -\sum_i p_i \log p_i$ where $p_i$ is the fraction of months in category $i$.
   - Plot $H$ over time. A year all in one category has $H = 0$; a year evenly split has high $H$.
   - **Surprising claim**: drought years are *more predictable* (lower entropy) than
     transition years. The graph should show this.
2. **KL divergence between drought-year and normal-year precipitation distributions**.
   - $D_{KL}(P \| Q) = \sum_i P(i) \log \frac{P(i)}{Q(i)}$ over precipitation-amount bins.
   - Higher KL = drought precipitation looks very different from normal.
3. **Mutual information between SPI-12 and ETI-12**.
   - $I(X; Y) = \sum_{x, y} p(x, y) \log \frac{p(x, y)}{p(x) p(y)}$
   - Discretise into bins, estimate joint and marginal histograms, compute $I$.
   - Interpretation: how much does precipitation tell us about ET, beyond their marginals?

### Statistical details expander

- Shannon entropy as expected log-surprise: $H(X) = -\mathbb{E}[\log p(X)]$.
- Mutual information as KL divergence between joint and product of marginals.
- Why $\log_2$ vs. $\ln$: just a unit choice (bits vs. nats).
- Connection to MLE: maximising likelihood is equivalent to minimising KL divergence
  from the empirical distribution to the model — this is the CS109 link between
  information theory and MLE.

---

## Sidebar

Keep the existing sidebar layout. Update the subtitle to remove HMM language:

> **Drought Analysis**
>
> A probabilistic study of San Joaquin Valley drought from 2000 to 2020 using
> precipitation and evapotranspiration alone.

Remove the "Regime legend" section (no more HMM regimes).

---

## Quick statistics at top of page

Keep the layout but switch from regime percentages to climatologically-grounded
numbers, with CS109 framings:

| Card | Value | Caption |
|------|-------|---------|
| Driest year | 2014 | (lowest mean SPI-12 across the record) |
| Wettest year | 2017 | (highest mean SPI-12) |
| Bootstrap-CI on drought rate | e.g. 30% ± 4% | (95% CI from 10k bootstraps) |
| Sample size | 241 months | (one observation per month) |

---

## Implementation order (suggested)

1. **Clean up the codebase first**. Delete `hmm.py`, `viterbi.py`,
   `forward_backward.py`, `emissions.py`, `main.py`. Remove HMM references
   from `data_loader.py` if any remain (they should be minimal).
2. Update `CLAUDE.md` references to those files (this file already
   anticipates their removal).
3. Build the tabs in this order: 1 (already done) → 5 → 4 → 2 → 3.
   The reason: entropy and bootstrap are simpler and let you nail the
   UI pattern first; MLE-Gamma and from-scratch logistic regression
   are heavier and benefit from a settled UI.
4. After each tab works, write its corresponding section in `WRITEUP.md`.
5. Run a final pass through every chart and ensure the plain-English caption
   doesn't mention `γ`, `Σ`, `Pr(Z=k|X)`, or any HMM artifact. All math lives
   inside the `st.expander("Statistical details")` block.

---

## Writeup framing

The CS109 grader will read `WRITEUP.md` (max 3 pages) and watch a 5-minute
video. The writeup should make the CS109-concept mapping explicit:

| Section | CS109 lecture(s) it covers |
|---------|----------------------------|
| Modeling precipitation distributions | Continuous RVs (Gamma, Normal); MLE |
| Predicting drought | Logistic Regression; Comparing Classifiers |
| Quantifying uncertainty | Sampling & Bootstrap; CLT; Beta-Binomial |
| Information content of climate signals | Information Theory |

A grader pointing at any chart should be able to say which CS109 lecture
it draws from. That's the goal.
