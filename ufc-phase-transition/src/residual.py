"""
Residual analysis to find systematic errors.
Layer 4 of the framework.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from .scoring import ScoringRules
from .visualization import plot_residual_diagnostics

class ResidualAnalyzer:
    def __init__(self, predictions, outcomes, metadata=None):
        self.p = np.array(predictions)
        self.o = np.array(outcomes)
        if len(self.p) != len(self.o):
            raise ValueError(
                f"predictions and outcomes must have the same length, "
                f"got {len(self.p)} and {len(self.o)}."
            )
        self.residuals = self.p - self.o
        self.meta = metadata

    def summary(self):
        print("=" * 50)
        print("RESIDUAL SUMMARY")
        print("=" * 50)
        print(f"  Mean residual:    {self.residuals.mean():.4f}")
        print(f"  Std of residuals: {self.residuals.std():.4f}")
        print(f"  Max overconfidence:  {self.residuals.max():.4f}")
        print(f"  Max underconfidence: {self.residuals.min():.4f}")
        bs = ScoringRules.brier_score(self.p, self.o)
        bs_baseline = ScoringRules.baseline_brier(self.o)
        ll = ScoringRules.log_loss(self.p, self.o)
        print(f"\n  Brier Score:      {bs:.4f}")
        print(f"  Baseline Brier:   {bs_baseline:.4f}")
        print(f"  Skill vs baseline: {bs_baseline - bs:.4f}")
        print(f"  Log-Loss:         {ll:.4f}")

    def plot(self, save_path=None):
        plot_residual_diagnostics(self.p, self.o, self.residuals, save_path)

    def group_by(self, column):
        if self.meta is None:
            raise ValueError("No metadata provided")
        df = self.meta.copy()
        df['residual'] = self.residuals
        return df.groupby(column)['residual'].agg(['mean', 'std', 'count'])
