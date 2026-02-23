"""
Gaussian copula for joint distribution of fight outcomes.
Layer 2 of the framework.
"""
import numpy as np
from scipy.stats import norm
import pandas as pd

class GaussianCopula:
    """
    Models dependence between fights on the same card.
    rho_matrix: correlation matrix (n_fights x n_fights)
    """
    def __init__(self, n_fights, rho_matrix=None):
        self.n = n_fights
        if rho_matrix is None:
            self.R = np.eye(n_fights)
        else:
            self.R = np.array(rho_matrix, dtype=float)
            if self.R.shape != (n_fights, n_fights):
                raise ValueError(
                    f"rho_matrix must be ({n_fights}, {n_fights}), "
                    f"got {self.R.shape}"
                )
            if not np.allclose(self.R, self.R.T):
                raise ValueError("rho_matrix must be symmetric.")
            if not np.allclose(np.diag(self.R), 1.0):
                raise ValueError("rho_matrix diagonal entries must all be 1.0.")

    def simulate_card(self, fight_probs, n_simulations=10000):
        try:
            L = np.linalg.cholesky(self.R)
        except np.linalg.LinAlgError as exc:
            raise ValueError(
                "Correlation matrix is not positive-definite; "
                "cannot perform Cholesky decomposition. "
                "Ensure all off-diagonal entries satisfy |rho| < 1 "
                "and the matrix is valid."
            ) from exc
        Z = np.random.randn(n_simulations, self.n) @ L.T
        U = norm.cdf(Z)
        thresholds = np.array(fight_probs)
        outcomes = (U < thresholds).astype(int)
        return outcomes

    def joint_probability(self, fight_probs, target_outcome, n_simulations=50000):
        outcomes = self.simulate_card(fight_probs, n_simulations)
        matches = np.all(outcomes == np.array(target_outcome), axis=1)
        return matches.mean()

    def favorites_sweep_probability(self, fight_probs, n_simulations=50000):
        return self.joint_probability(fight_probs, [1]*len(fight_probs), n_simulations)

    def expected_correct(self, fight_probs, n_simulations=10000):
        outcomes = self.simulate_card(fight_probs, n_simulations)
        return outcomes.mean(axis=0)

    def correlation_effect(self, fight_probs, rho_values=None, n_simulations=30000):
        if rho_values is None:
            rho_values = [0.0, 0.1, 0.2, 0.3, 0.5]
        n = len(fight_probs)
        results = []
        for rho in rho_values:
            R = np.full((n, n), rho)
            np.fill_diagonal(R, 1.0)
            copula = GaussianCopula(n, R)
            p_sweep = copula.favorites_sweep_probability(fight_probs, n_simulations)
            p_naive = np.prod(fight_probs)
            results.append({
                'rho': rho,
                'P(all favorites win) copula': round(p_sweep, 4),
                'P(all favorites win) naive': round(p_naive, 4),
                'ratio': round(p_sweep / p_naive, 3)
            })
        return pd.DataFrame(results)
