"""
Bradley-Terry model for fighter ratings.
Layer 1 of the framework.
"""
import warnings
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit

class BradleyTerry:
    """
    Each fighter has a latent skill beta (log-skill).
    P(i beats j) = sigmoid(beta_i - beta_j)
    """
    def __init__(self):
        self.ratings = {}
        self.fighters = []

    def _sigmoid(self, x):
        # expit is numerically stable for extreme values, unlike 1/(1+exp(-x))
        return expit(x)

    def win_probability(self, fighter_a, fighter_b):
        beta_a = self.ratings.get(fighter_a, 0.0)
        beta_b = self.ratings.get(fighter_b, 0.0)
        return self._sigmoid(beta_a - beta_b)

    def fit(self, fight_results, time_weights=None):
        """
        fight_results: list of (winner, loser) tuples
        time_weights: optional array of weights for each fight (e.g., time decay)
        """
        self.fighters = list({f for pair in fight_results for f in pair})
        n = len(self.fighters)
        idx = {f: i for i, f in enumerate(self.fighters)}

        if time_weights is None:
            time_weights = np.ones(len(fight_results))

        def neg_log_likelihood(betas):
            nll = 0.0
            for (winner, loser), w in zip(fight_results, time_weights):
                i, j = idx[winner], idx[loser]
                p = self._sigmoid(betas[i] - betas[j])
                nll -= w * np.log(p + 1e-10)
            # L2 regularization
            nll += 0.01 * np.sum(betas**2)
            return nll

        beta_init = np.zeros(n)
        result = minimize(neg_log_likelihood, beta_init, method='L-BFGS-B')
        if not result.success:
            warnings.warn(
                f"Bradley-Terry optimization did not converge: {result.message}. "
                "Ratings may be unreliable."
            )
        betas = result.x
        # Center ratings
        betas -= betas.mean()

        self.ratings = {f: betas[idx[f]] for f in self.fighters}
        return self

    def leaderboard(self):
        df = pd.DataFrame(
            [(name, rating) for name, rating in self.ratings.items()],
            columns=['Fighter', 'Beta']
        ).sort_values('Beta', ascending=False).reset_index(drop=True)
        df['Win% vs avg'] = self._sigmoid(df['Beta']) * 100
        return df
