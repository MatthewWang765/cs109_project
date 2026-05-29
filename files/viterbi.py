"""
viterbi.py — Viterbi algorithm for MAP decoding of the hidden state sequence.
"""

import numpy as np


def viterbi(log_B, A, pi):
    """
    Parameters
    ----------
    log_B : (T, K) log-emission matrix
    A     : (K, K) transition matrix
    pi    : (K,)   initial distribution

    Returns
    -------
    states   : (T,) most likely state sequence (0-indexed)
    log_prob : float
    """
    T, K = log_B.shape
    log_A = np.log(A + 1e-300)
    log_pi = np.log(pi + 1e-300)

    delta = np.full((T, K), -np.inf)
    psi = np.zeros((T, K), dtype=int)

    # Initialization
    delta[0] = log_pi + log_B[0]

    # Recursion
    for t in range(1, T):
        # scores[j, k] = delta[t-1, j] + log_A[j, k]
        scores = delta[t - 1][:, None] + log_A          # (K, K)
        psi[t] = np.argmax(scores, axis=0)              # (K,)
        delta[t] = scores[psi[t], np.arange(K)] + log_B[t]

    # Termination
    best_last = int(np.argmax(delta[T - 1]))
    log_prob = delta[T - 1, best_last]

    # Backtrack
    states = np.zeros(T, dtype=int)
    states[T - 1] = best_last
    for t in range(T - 2, -1, -1):
        states[t] = psi[t + 1, states[t + 1]]

    return states, log_prob
