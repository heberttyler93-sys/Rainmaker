"""
Proper scoring rules for evaluation.
Layer 3 of the framework.
"""
import numpy as np
import pandas as pd

class ScoringRules:
    @staticmethod
    def brier_score(predictions, outcomes):
        p = np.array(predictions)
        o = np.array(outcomes)
        return np.mean((p - o) ** 2)

    @staticmethod
    def log_loss(predictions, outcomes, eps=1e-10):
        p = np.clip(np.array(predictions), eps, 1 - eps)
        o = np.array(outcomes)
        return -np.mean(o * np.log(p) + (1 - o) * np.log(1 - p))

    @staticmethod
    def calibration_curve(predictions, outcomes, n_bins=5):
        p = np.array(predictions)
        o = np.array(outcomes)
        bins = np.linspace(0, 1, n_bins + 1)
        rows = []
        for i in range(n_bins):
            mask = (p >= bins[i]) & (p < bins[i+1])
            if mask.sum() > 0:
                rows.append({
                    'bin_center': (bins[i] + bins[i+1]) / 2,
                    'mean_predicted': p[mask].mean(),
                    'actual_win_rate': o[mask].mean(),
                    'n_fights': mask.sum()
                })
        return pd.DataFrame(rows)

    @staticmethod
    def baseline_brier(outcomes):
        return np.mean((0.5 - np.array(outcomes)) ** 2)
