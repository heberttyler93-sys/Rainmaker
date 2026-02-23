"""
Train Bradley-Terry model with rolling window.
"""
import pandas as pd
import numpy as np
from src.bradley_terry import BradleyTerry
from src.utils import time_decay_weights
import os

PROCESSED_DIR = 'data/processed/'
fights = pd.read_csv(os.path.join(PROCESSED_DIR, 'fights_clean.csv'))
fights['DATE'] = pd.to_datetime(fights['DATE'])
fights = fights.sort_values('DATE').reset_index(drop=True)

# Rolling window: for each event, train on all previous fights
unique_events = fights['EVENT'].unique()
fights['p_pred'] = np.nan

for event in unique_events:
    event_mask = fights['EVENT'] == event
    # Training data: all fights before this event
    train_mask = fights['DATE'] < fights.loc[event_mask, 'DATE'].min()
    train_df = fights[train_mask]
    if len(train_df) == 0:
        fights.loc[event_mask, 'p_pred'] = 0.5
    else:
        # Optionally apply time decay
        weights = time_decay_weights(train_df['DATE'])
        fight_list = list(zip(train_df['winner'], train_df['loser']))
        bt = BradleyTerry()
        bt.fit(fight_list, time_weights=weights)
        # Predict each fight in current event
        for idx in fights[event_mask].index:
            f1 = fights.loc[idx, 'fighter1']
            f2 = fights.loc[idx, 'fighter2']
            prob = bt.win_probability(f1, f2)
            fights.loc[idx, 'p_pred'] = prob

# Save with predictions
fights.to_csv(os.path.join(PROCESSED_DIR, 'fights_with_probs.csv'), index=False)
print("Predictions saved.")
