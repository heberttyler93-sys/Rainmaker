"""
Compute residuals and generate phase-transition plots.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from src.residual import ResidualAnalyzer
from src.visualization import plot_variance_vs_fav_prob, plot_conditional_upset
import os

PROCESSED_DIR = 'data/processed/'
os.makedirs('outputs', exist_ok=True)
fights = pd.read_csv(os.path.join(PROCESSED_DIR, 'fights_with_probs.csv'))
fights['DATE'] = pd.to_datetime(fights['DATE'])

# Determine favorite and underdog
fights['favorite'] = fights.apply(lambda row: row['fighter1'] if row['p_pred'] > 0.5 else row['fighter2'], axis=1)
fights['favorite_win_prob'] = fights.apply(lambda row: row['p_pred'] if row['p_pred'] > 0.5 else 1 - row['p_pred'], axis=1)
fights['favorite_won'] = fights['winner'] == fights['favorite']
fights['upset'] = ~fights['favorite_won']
fights['p_upset'] = 1 - fights['favorite_win_prob']

# Aggregate by event
event_stats = fights.groupby('EVENT').agg(
    n_fights=('upset', 'size'),
    expected_upsets=('p_upset', 'sum'),
    actual_upsets=('upset', 'sum'),
    avg_fav_prob=('favorite_win_prob', 'mean'),
    date=('DATE', 'first')
).reset_index()
event_stats['residual'] = event_stats['actual_upsets'] - event_stats['expected_upsets']

# Add expected variance per event (under independence)
event_stats['expected_var'] = fights.groupby('EVENT').apply(
    lambda df: (df['p_upset'] * (1 - df['p_upset'])).sum()
).values

# Bout order (if not present, assume row order)
fights['bout_num'] = fights.groupby('EVENT').cumcount() + 1

# Plots
plt.figure(figsize=(8,5))
plt.hist(event_stats['residual'], bins=15, edgecolor='black')
plt.axvline(0, color='red', linestyle='--')
plt.xlabel('Residual (actual - expected upsets)')
plt.ylabel('Number of events')
plt.title('Distribution of event-level residuals')
plt.savefig('outputs/residual_histogram.png', dpi=150, bbox_inches='tight')
plt.show()

plot_variance_vs_fav_prob(event_stats, save_path='outputs/variance_vs_fav_prob.png')
plot_conditional_upset(fights, save_path='outputs/conditional_upset.png')

print("Analysis complete. Plots saved in outputs/")
