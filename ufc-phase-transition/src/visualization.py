"""
Reusable plotting functions for residual diagnostics and phase transition.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from .scoring import ScoringRules

def plot_residual_diagnostics(p, o, residuals, save_path=None):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle("Residual Diagnostics", fontsize=13, fontweight='bold')

    # Residuals vs predicted
    axes[0].scatter(p, residuals, alpha=0.6, color='steelblue')
    axes[0].axhline(0, color='red', linestyle='--')
    axes[0].set_xlabel("Predicted P(favorite wins)")
    axes[0].set_ylabel("Residual")
    axes[0].set_title("Residuals vs Predicted")

    # Histogram
    axes[1].hist(residuals, bins=15, color='steelblue', edgecolor='white')
    axes[1].axvline(0, color='red', linestyle='--')
    axes[1].set_xlabel("Residual")
    axes[1].set_title("Residual Distribution")

    # Calibration curve
    cal = ScoringRules.calibration_curve(p, o)
    axes[2].plot([0,1], [0,1], 'r--', label='Perfect')
    axes[2].scatter(cal['mean_predicted'], cal['actual_win_rate'],
                    s=cal['n_fights']*5, color='steelblue', label='Model')
    axes[2].plot(cal['mean_predicted'], cal['actual_win_rate'], color='steelblue')
    axes[2].set_xlabel("Mean predicted probability")
    axes[2].set_ylabel("Actual win rate")
    axes[2].set_title("Calibration Curve")
    axes[2].legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()

def plot_variance_vs_fav_prob(event_stats, save_path=None):
    """Phase transition plot: variance of upsets vs average favorite prob."""
    bins = np.linspace(0.5, 1.0, 11)
    event_stats = event_stats.copy()
    event_stats['prob_bin'] = pd.cut(event_stats['avg_fav_prob'], bins)
    bin_summary = event_stats.groupby('prob_bin', observed=False).agg(
        obs_variance=('actual_upsets', 'var'),
        exp_variance=('expected_var', 'mean'),
        n_events=('EVENT', 'size')
    ).reset_index()
    # Compute bin center from each interval to keep alignment correct
    bin_summary['bin_center'] = bin_summary['prob_bin'].apply(
        lambda x: x.mid if pd.notna(x) else np.nan
    )
    # Drop bins with no data
    bin_summary = bin_summary[bin_summary['n_events'] > 0]

    plt.figure(figsize=(8,5))
    plt.plot(bin_summary['bin_center'], bin_summary['obs_variance'], 'o-', label='Observed')
    plt.plot(bin_summary['bin_center'], bin_summary['exp_variance'], 's--', label='Expected (independence)')
    plt.xlabel('Average favorite win probability')
    plt.ylabel('Variance of number of upsets')
    plt.legend()
    plt.title('Variance of upsets vs. favorite strength')
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()

def plot_conditional_upset(results, save_path=None):
    """Cascade plot: P(upset | number of prior upsets)."""
    results_sorted = results.sort_values(['EVENT', 'bout_num']).copy()
    results_sorted['prior_upsets'] = results_sorted.groupby('EVENT')['upset'].transform(
        lambda x: x.shift(1).fillna(0).cumsum()
    )
    cond = results_sorted.groupby('prior_upsets')['upset'].agg(['mean', 'count', 'sem'])
    cond = cond[cond['count'] > 10]  # require at least 10 observations

    plt.figure(figsize=(8,5))
    plt.errorbar(cond.index, cond['mean'], yerr=1.96*cond['sem'], fmt='o-', capsize=3)
    plt.axhline(results['upset'].mean(), color='red', linestyle='--', label='Overall upset rate')
    plt.xlabel('Number of prior upsets on the card')
    plt.ylabel('Probability of upset in current fight')
    plt.title('Cascade effect: does one upset lead to another?')
    plt.legend()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
