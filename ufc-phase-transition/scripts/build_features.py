"""
Load raw data, clean, merge, and save to processed/.
"""
import pandas as pd
from src.data_loader import load_fight_results, load_event_details, load_fighter_stats, load_round_stats, merge_all
import os

RAW_DIR = 'data/raw/'
PROCESSED_DIR = 'data/processed/'
os.makedirs(PROCESSED_DIR, exist_ok=True)

# Load
fight_results = load_fight_results(os.path.join(RAW_DIR, 'fight_results.csv'))
event_details = load_event_details(os.path.join(RAW_DIR, 'event_details.csv'))
fighter_stats = load_fighter_stats(os.path.join(RAW_DIR, 'fighter_stats.csv'))
round_stats = load_round_stats(os.path.join(RAW_DIR, 'round_stats.csv'))  # optional

# Merge
merged = merge_all(fight_results, event_details, fighter_stats)

# Add any additional cleaning (e.g., normalize fighter names, handle missing values)

# Save
merged.to_csv(os.path.join(PROCESSED_DIR, 'fights_clean.csv'), index=False)

# Also create event-level summary (will be computed later, but we can save merged fights)
print("Feature building complete.")
