"""
forward_backward.py — E-step of Baum-Welch.

Scaled forward-backward (Rabiner scaling) to avoid underflow.
"""

import numpy as np


def forward(log_B, A, pi):
    """
    Scaled forward algorithm.

    Returns
    -------
    alpha  : (T, K) scaled forward variables
    scales : (T,)   per-step scaling factors
    """
    T, K = log_B.shape
    alpha = np.zeros((T, K))
    scales = np.zeros(T)

    # t = 0
    alpha[0] = pi * np.exp(log_B[0] - log_B[0].max())   # shift for stability
    # undo the shift: this is fine because we normalize immediately
    alpha[0] = pi * np.exp(log_B[0])
    scales[0] = alpha[0].sum()
    if scales[0] == 0:
        scales[0] = 1e-300
    alpha[0] /= scales[0]

    # t = 1 .. T-1
    for t in range(1, T):
        b = np.exp(log_B[t])                           # (K,)
        alpha[t] = b * (alpha[t - 1] @ A)             # (K,)
        scales[t] = alpha[t].sum()
        if scales[t] == 0:
            scales[t] = 1e-300
        alpha[t] /= scales[t]

    return alpha, scales


def backward(log_B, A, scales):
    """
    Scaled backward algorithm using the same scale factors as forward.

    Returns
    -------
    beta : (T, K) scaled backward variables
    """
    T, K = log_B.shape
    beta = np.zeros((T, K))

    beta[T - 1] = 1.0 / scales[T - 1]

    for t in range(T - 2, -1, -1):
        b_next = np.exp(log_B[t + 1])                 # (K,)
        beta[t] = A @ (b_next * beta[t + 1])          # (K,)
        beta[t] /= scales[t]

    return beta


def compute_gamma_xi(alpha, beta, log_B, A):
    """
    Compute gamma and xi from scaled alpha and beta.

    Returns
    -------
    gamma : (T, K)
    xi    : (T-1, K, K)
    """
    T, K = alpha.shape

    # gamma: normalize alpha * beta row-wise
    gamma = alpha * beta
    row_sums = gamma.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1e-300, row_sums)
    gamma /= row_sums

    # xi[t, j, k] ∝ alpha[t,j] * A[j,k] * B[t+1,k] * beta[t+1,k]
    xi = np.zeros((T - 1, K, K))
    for t in range(T - 1):
        b_next = np.exp(log_B[t + 1])                 # (K,)
        # outer: alpha[t][:,None] * A  then multiply each row by b_next*beta[t+1]
        mat = alpha[t][:, None] * A * (b_next * beta[t + 1])[None, :]
        total = mat.sum()
        xi[t] = mat / (total if total > 0 else 1e-300)

    return gamma, xi


def forward_backward(log_B, A, pi):
    """Full E-step: returns gamma, xi, and log-likelihood."""
    alpha, scales = forward(log_B, A, pi)
    beta = backward(log_B, A, scales)
    gamma, xi = compute_gamma_xi(alpha, beta, log_B, A)

    log_likelihood = np.sum(np.log(scales + 1e-300))

    return gamma, xi, log_likelihood
