"""
emissions.py — Multivariate Gaussian log-likelihood for HMM emissions.
"""

import numpy as np
from scipy.stats import multivariate_normal


def log_gaussian(x, mu, sigma):
    """log N(x; mu, Sigma) for a single observation."""
    return multivariate_normal.logpdf(x, mean=mu, cov=sigma, allow_singular=False)


def compute_log_emission_matrix(X, mus, sigmas):
    """
    Compute the full (T, K) matrix of log emission probabilities.

    log_B[t, k] = log N(X_t; mu_k, Sigma_k)

    Parameters
    ----------
    X      : (T, D) observation matrix
    mus    : (K, D) mean vectors
    sigmas : (K, D, D) covariance matrices

    Returns
    -------
    log_B : (T, K) log-emission matrix
    """
    K = len(mus)
    log_B = np.column_stack([
        multivariate_normal.logpdf(X, mean=mus[k], cov=sigmas[k])
        for k in range(K)
    ])
    return log_B
