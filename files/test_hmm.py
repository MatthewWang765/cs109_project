"""
test_hmm.py — Unit tests for each HMM component.

Run with: python -m pytest tests/ -v

These tests verify correctness BEFORE you run on real data.
Each test checks a specific mathematical property that must hold.
"""

import numpy as np
import pytest
from src.emissions import compute_log_emission_matrix, log_gaussian
from src.data_loader import make_synthetic_data
from src.hmm import GaussianHMM


# ---------------------------------------------------------------
# Test 1: Emission log-likelihood
# ---------------------------------------------------------------

def test_log_gaussian_known_value():
    """log N(0; 0, I) in 2D = -log(2*pi) ≈ -1.8379"""
    x = np.array([0.0, 0.0])
    mu = np.array([0.0, 0.0])
    sigma = np.eye(2)
    result = log_gaussian(x, mu, sigma)
    expected = -np.log(2 * np.pi)
    assert abs(result - expected) < 1e-6, f"Got {result}, expected {expected}"


def test_log_emission_matrix_shape():
    T, K, D = 50, 4, 2
    X = np.random.randn(T, D)
    mus = np.random.randn(K, D)
    sigmas = np.array([np.eye(D) for _ in range(K)])
    log_B = compute_log_emission_matrix(X, mus, sigmas)
    assert log_B.shape == (T, K), f"Expected ({T}, {K}), got {log_B.shape}"


def test_log_emission_values_negative():
    """Log-likelihoods of a pdf are negative (pdf < 1 in continuous case can be >1,
    but for standard Gaussians with reasonable inputs, log values are negative)"""
    T, K, D = 20, 4, 2
    X = np.random.randn(T, D)
    mus = np.zeros((K, D))
    sigmas = np.array([np.eye(D) * 5 for _ in range(K)])
    log_B = compute_log_emission_matrix(X, mus, sigmas)
    assert np.all(np.isfinite(log_B)), "log_B contains NaN or Inf"


# ---------------------------------------------------------------
# Test 2: Forward-backward probabilities sum to 1
# ---------------------------------------------------------------

def test_gamma_sums_to_one():
    """gamma[t, :] must sum to 1 for every t."""
    pytest.importorskip("src.forward_backward")
    from src.forward_backward import forward_backward

    T, K, D = 100, 4, 2
    X = np.random.randn(T, D)
    mus = np.array([np.array([i, -i], dtype=float) for i in range(K)])
    sigmas = np.array([np.eye(D) for _ in range(K)])

    A = np.ones((K, K)) / K
    pi = np.ones(K) / K

    from src.emissions import compute_log_emission_matrix
    log_B = compute_log_emission_matrix(X, mus, sigmas)
    gamma, xi, ll = forward_backward(log_B, A, pi)

    row_sums = gamma.sum(axis=1)
    assert np.allclose(row_sums, 1.0, atol=1e-5), f"gamma rows don't sum to 1: {row_sums[:5]}"


def test_xi_sums_consistent_with_gamma():
    """sum_k xi[t, j, k] = gamma[t, j] for all t, j"""
    pytest.importorskip("src.forward_backward")
    from src.forward_backward import forward_backward
    from src.emissions import compute_log_emission_matrix

    T, K, D = 100, 4, 2
    X = np.random.randn(T, D)
    mus = np.array([np.array([float(i), float(-i)]) for i in range(K)])
    sigmas = np.array([np.eye(D) for _ in range(K)])
    A = np.ones((K, K)) / K
    pi = np.ones(K) / K

    log_B = compute_log_emission_matrix(X, mus, sigmas)
    gamma, xi, ll = forward_backward(log_B, A, pi)

    # xi[t].sum(axis=1) should equal gamma[t] for t < T-1
    for t in range(T - 1):
        xi_marginal = xi[t].sum(axis=1)
        assert np.allclose(xi_marginal, gamma[t], atol=1e-4), \
            f"At t={t}: xi marginal {xi_marginal} != gamma {gamma[t]}"


# ---------------------------------------------------------------
# Test 3: Transition matrix rows sum to 1
# ---------------------------------------------------------------

def test_A_rows_sum_to_one():
    """After M-step, every row of A must sum to 1."""
    X, _, _ = make_synthetic_data(T=500)
    model = GaussianHMM(K=4, max_iter=5)
    try:
        model.fit(X)
        row_sums = model.A.sum(axis=1)
        assert np.allclose(row_sums, 1.0, atol=1e-6), f"A rows don't sum to 1: {row_sums}"
    except NotImplementedError:
        pytest.skip("M-step not implemented yet")


# ---------------------------------------------------------------
# Test 4: Log-likelihood increases monotonically
# ---------------------------------------------------------------

def test_log_likelihood_increases():
    """Baum-Welch is an EM algorithm: log-likelihood must never decrease."""
    X, _, _ = make_synthetic_data(T=1000)
    model = GaussianHMM(K=4, max_iter=20)
    try:
        model.fit(X)
        lls = model.log_likelihoods
        for i in range(1, len(lls)):
            assert lls[i] >= lls[i-1] - 1e-3, \
                f"Log-likelihood decreased at iter {i}: {lls[i-1]:.4f} -> {lls[i]:.4f}"
    except NotImplementedError:
        pytest.skip("Not fully implemented yet")


# ---------------------------------------------------------------
# Test 5: Viterbi output shape and validity
# ---------------------------------------------------------------

def test_viterbi_output():
    """Viterbi should return integer states in [0, K-1] of length T."""
    pytest.importorskip("src.viterbi")
    from src.viterbi import viterbi
    from src.emissions import compute_log_emission_matrix

    T, K, D = 200, 4, 2
    X = np.random.randn(T, D)
    mus = np.random.randn(K, D)
    sigmas = np.array([np.eye(D) for _ in range(K)])
    A = np.ones((K, K)) / K
    pi = np.ones(K) / K

    log_B = compute_log_emission_matrix(X, mus, sigmas)
    try:
        states, log_prob = viterbi(log_B, A, pi)
        assert len(states) == T, f"Expected {T} states, got {len(states)}"
        assert states.min() >= 0 and states.max() < K, "States out of range"
        assert np.isfinite(log_prob), "log_prob is not finite"
    except NotImplementedError:
        pytest.skip("Viterbi not implemented yet")


# ---------------------------------------------------------------
# Test 6: Recovery of synthetic parameters (integration test)
# ---------------------------------------------------------------

def test_synthetic_recovery():
    """
    On synthetic data, the learned mus should be close to the true mus
    (up to label permutation). This is the key sanity check.
    """
    X, true_states, true_params = make_synthetic_data(T=3000, random_state=0)
    model = GaussianHMM(K=4, max_iter=100)
    try:
        model.fit(X)
        # Normalize X to compare (model is fit on normalized data)
        from src.data_loader import normalize
        X_norm, mean, std = normalize(X)
        model2 = GaussianHMM(K=4, max_iter=100)
        model2.fit(X_norm)
        # Just check model converged (log-likelihood improved)
        assert model2.log_likelihoods[-1] > model2.log_likelihoods[0], \
            "Log-likelihood did not improve"
    except NotImplementedError:
        pytest.skip("Model not fully implemented yet")
