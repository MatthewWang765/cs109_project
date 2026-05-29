"""
hmm.py — GaussianHMM: the top-level model class.

Ties together:
  - Initialization (k-means for mu, identity for Sigma, near-uniform for A)
  - Baum-Welch training loop (calls forward_backward + M-step)
  - Viterbi decoding
  - Parameter inspection

All probability implemented from scratch (no hmmlearn).
"""

import numpy as np
from emissions import compute_log_emission_matrix
from forward_backward import forward_backward
from viterbi import viterbi


class GaussianHMM:
    """
    K-state Hidden Markov Model with multivariate Gaussian emissions.

    Parameters learned via Baum-Welch (EM).
    """

    def __init__(self, K=4, max_iter=200, tol=1e-4, reg_covar=0.01, random_state=42):
        """
        Parameters
        ----------
        K            : number of hidden states
        max_iter     : maximum Baum-Welch iterations
        tol          : convergence threshold on log-likelihood change
        reg_covar    : covariance regularization (added to diagonal)
        random_state : for reproducibility
        """
        self.K = K
        self.max_iter = max_iter
        self.tol = tol
        self.reg_covar = reg_covar
        self.rng = np.random.default_rng(random_state)

        # Parameters (set during fit)
        self.pi = None       # (K,)    initial distribution
        self.A = None        # (K, K)  transition matrix
        self.mus = None      # (K, D)  emission means
        self.sigmas = None   # (K, D, D) emission covariances

        self.log_likelihoods = []   # track convergence

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def _initialize(self, X):
        """
        Initialize parameters before Baum-Welch.

        Strategy:
          - mu_k     : k-means centroids on X  (avoids label-switching)
          - Sigma_k  : identity matrices
          - A        : near-uniform with small noise, rows normalized
          - pi       : uniform
        """
        T, D = X.shape
        K = self.K

        # --- pi: uniform ---
        self.pi = np.ones(K) / K

        # --- A: near-uniform ---
        A = np.ones((K, K)) + self.rng.uniform(0, 0.1, size=(K, K))
        self.A = A / A.sum(axis=1, keepdims=True)

        # --- mu: k-means (simple implementation) ---
        # TODO: you can replace with sklearn.cluster.KMeans if desired,
        # but implement a simple version here first.
        indices = self.rng.choice(T, K, replace=False)
        self.mus = X[indices].copy().astype(float)   # (K, D)

        # Run simple k-means for a few iterations
        for _ in range(20):
            # Assign each point to nearest centroid
            dists = np.array([np.linalg.norm(X - self.mus[k], axis=1) for k in range(K)]).T
            labels = np.argmin(dists, axis=1)
            for k in range(K):
                mask = labels == k
                if mask.sum() > 0:
                    self.mus[k] = X[mask].mean(axis=0)

        # --- Sigma: identity matrices ---
        self.sigmas = np.array([np.eye(D) for _ in range(K)])

    # ------------------------------------------------------------------
    # M-step
    # ------------------------------------------------------------------

    def _m_step(self, X, gamma, xi):
        """
        Update parameters given E-step posteriors.

        Parameters
        ----------
        X     : (T, D) observations
        gamma : (T, K) posterior state probabilities
        xi    : (T-1, K, K) pairwise posteriors
        """
        T, D = X.shape
        K = self.K

        # --- pi ---
        self.pi = gamma[0] / gamma[0].sum()

        # --- A ---
        xi_sum = xi.sum(axis=0)                        # (K, K)
        row_sum = xi_sum.sum(axis=1, keepdims=True)    # (K, 1)
        self.A = xi_sum / np.where(row_sum == 0, 1e-300, row_sum)

        # --- mu and Sigma ---
        for k in range(K):
            weight = gamma[:, k]                       # (T,)
            total_weight = weight.sum() + 1e-300

            self.mus[k] = (weight[:, None] * X).sum(axis=0) / total_weight

            diff = X - self.mus[k]                     # (T, D)
            self.sigmas[k] = (
                (weight[:, None, None] * diff[:, :, None] * diff[:, None, :]).sum(axis=0)
                / total_weight
                + self.reg_covar * np.eye(D)
            )

    # ------------------------------------------------------------------
    # Baum-Welch training loop
    # ------------------------------------------------------------------

    def fit(self, X):
        """
        Fit the HMM to observations X via Baum-Welch (EM).

        Parameters
        ----------
        X : (T, D) observation matrix

        Returns
        -------
        self
        """
        self._initialize(X)
        prev_ll = -np.inf

        for iteration in range(self.max_iter):
            # E-step
            log_B = compute_log_emission_matrix(X, self.mus, self.sigmas)
            gamma, xi, log_likelihood = forward_backward(log_B, self.A, self.pi)

            self.log_likelihoods.append(log_likelihood)
            print(f"Iter {iteration+1:3d} | log-likelihood: {log_likelihood:.4f}")

            # Check convergence
            if abs(log_likelihood - prev_ll) < self.tol:
                print(f"Converged at iteration {iteration+1}")
                break
            prev_ll = log_likelihood

            # M-step
            self._m_step(X, gamma, xi)

        return self

    # ------------------------------------------------------------------
    # Decoding
    # ------------------------------------------------------------------

    def decode(self, X):
        """
        Run Viterbi to find the most likely state sequence.

        Parameters
        ----------
        X : (T, D)

        Returns
        -------
        states   : (T,) decoded state sequence
        log_prob : float
        """
        log_B = compute_log_emission_matrix(X, self.mus, self.sigmas)
        return viterbi(log_B, self.A, self.pi)

    def posterior(self, X):
        """
        Smoothed posterior P(Z_t = k | X_{1:T}) via forward-backward.

        Unlike Viterbi (which returns one hard label per time step), this
        returns the full Bayesian posterior — a continuous probability over
        every state at every time step. This is what lets us produce a
        drought-intensity gradient instead of discrete state labels.

        Returns
        -------
        gamma : (T, K)  posterior state probabilities, rows sum to 1
        """
        log_B = compute_log_emission_matrix(X, self.mus, self.sigmas)
        gamma, _, _ = forward_backward(log_B, self.A, self.pi)
        return gamma

    # ------------------------------------------------------------------
    # Inspection helpers
    # ------------------------------------------------------------------

    def print_params(self):
        print("=== Initial distribution (pi) ===")
        print(self.pi)
        print("\n=== Transition matrix (A) ===")
        print(np.round(self.A, 3))
        print("\n=== Emission means (mu_k) ===")
        for k in range(self.K):
            print(f"  State {k}: {self.mus[k]}")
        print("\n=== Emission covariances (Sigma_k diagonal) ===")
        for k in range(self.K):
            print(f"  State {k}: diag = {np.diag(self.sigmas[k])}")
