"""
Feature engineering utilities for tennis upset prediction.

This module handles creating engineered features like ranking differences,
head-to-head records, and recent form metrics.
"""

import pandas as pd
import numpy as np
from collections import defaultdict
from sklearn.preprocessing import OrdinalEncoder
from typing import Tuple


def calculate_ranking_difference(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate ranking difference between underdog and favorite.
    
    Args:
        df: DataFrame with fav_rank and under_rank columns
        
    Returns:
        DataFrame with rank_diff column added
    """
    df = df.copy()
    df['rank_diff'] = df['under_rank'] - df['fav_rank']
    return df


def calculate_head_to_head(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate head-to-head record between players.
    
    Args:
        df: DataFrame with fav_id, under_id, and upset columns
        
    Returns:
        DataFrame with h2h and h2h_count columns added
    """
    df = df.copy()
    h2h = defaultdict(list)
    h2h_ratios = []
    h2h_counts = []
    
    for _, row in df.iterrows():
        fav = row['fav_id']
        under = row['under_id']
        key = tuple(sorted([fav, under]))
        record = h2h[key]
        
        last_5 = record[-5:]
        count = len(last_5)
        
        fav_wins = sum(1 for winner in last_5 if winner == fav)
        ratio = fav_wins / count if count > 0 else np.nan
        
        h2h_ratios.append(ratio)
        h2h_counts.append(count)
        h2h[key].append(row['fav_id'] if row['upset'] == 0 else row['under_id'])
    
    df['h2h'] = h2h_ratios
    df['h2h_count'] = h2h_counts
    
    return df


def calculate_recent_form(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate recent form for both players.
    
    Args:
        df: DataFrame with fav_id, under_id, and upset columns
        
    Returns:
        DataFrame with fav_form, under_form, and form_diff columns added
    """
    df = df.copy()
    player_history = defaultdict(list)
    fav_form = []
    under_form = []
    
    for _, row in df.iterrows():
        fav = row['fav_id']
        under = row['under_id']
        
        fav_recent = player_history[fav][-5:]
        under_recent = player_history[under][-5:]
        fav_streak = sum(fav_recent) / len(fav_recent) if fav_recent else np.nan
        under_streak = sum(under_recent) / len(under_recent) if under_recent else np.nan
        
        fav_form.append(fav_streak)
        under_form.append(under_streak)
        
        if row['upset'] == 0:
            player_history[fav].append(1)
            player_history[under].append(0)
        else:
            player_history[fav].append(0)
            player_history[under].append(1)
    
    df['fav_form'] = fav_form
    df['under_form'] = under_form
    df['form_diff'] = df['under_form'] - df['fav_form']
    
    return df


def encode_categorical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode categorical features for modeling.
    
    Args:
        df: DataFrame with categorical columns
        
    Returns:
        DataFrame with encoded features
    """
    df = df.copy()
    
    # Group tournament levels
    level_group_map = {
        'G': 'G',
        'PM': 'PM_M',
        'M': 'PM_M',
        'P': 'P_T1_F',
        'T1': 'P_T1_F',
        'F': 'P_T1_F',
        'A': 'A_I',
        'I': 'A_I',
        'T2': 'T2',
        'T3': 'T3',
        'T4': 'T4_D',
        'D': 'T4_D',
        'T5': 'T5',
        'CC': 'Other',
        'O': 'Other',
        'W': 'Other'
    }
    
    df['tourney_level_grouped'] = df['tourney_level'].map(level_group_map)
    
    # Encode tournament level
    level_order = [['Other', 'T5', 'T4_D', 'T3', 'T2', 'A_I', 'P_T1_F', 'PM_M', 'G']]
    encoder = OrdinalEncoder(categories=level_order)
    df['tourney_level_enc'] = encoder.fit_transform(df[['tourney_level_grouped']])
    
    # Group rounds
    round_group_map = {
        'ER': 'Early',
        'BR': 'Early',
        'R128': 'Early',
        'RR': 'Early',
        'R64': 'R64',
        'R32': 'R32',
        'R16': 'R16',
        'QF': 'QF',
        'SF': 'SF',
        'F': 'F'
    }
    
    df['round_grouped'] = df['round'].map(round_group_map)
    
    # Encode rounds
    round_order = [['Early', 'R64', 'R32', 'R16', 'QF', 'SF', 'F']]
    encoder = OrdinalEncoder(categories=round_order)
    df['round_enc'] = encoder.fit_transform(df[['round_grouped']])
    
    # One-hot encode surface
    df = pd.get_dummies(df, columns=['surface'], prefix='surface', drop_first=False)
    
    return df


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Complete feature engineering pipeline.
    
    Args:
        df: Raw tennis match data
        
    Returns:
        DataFrame with all engineered features
    """
    print("Calculating ranking differences...")
    df = calculate_ranking_difference(df)
    
    print("Calculating head-to-head records...")
    df = calculate_head_to_head(df)
    
    print("Calculating recent form...")
    df = calculate_recent_form(df)
    
    print("Encoding categorical features...")
    df = encode_categorical_features(df)
    
    # Fill missing values
    df['h2h'] = df['h2h'].fillna(0.5)
    df['fav_form'] = df['fav_form'].fillna(0.5)
    df['under_form'] = df['under_form'].fillna(0.5)
    df['form_diff'] = df['form_diff'].fillna(0.0)
    df['h2h_count'] = df['h2h_count'].fillna(0.5)
    
    # Drop unused columns
    to_drop = ['tourney_id', 'tourney_date', 'match_num', 'fav_id', 'under_id',
               'tourney_level_grouped', 'round_grouped']
    df = df.drop(columns=to_drop, errors='ignore')
    
    # Remove rows with missing data
    df = df.dropna()
    
    print(f"Final dataset shape: {df.shape}")
    return df


def get_feature_columns() -> list:
    """
    Get list of feature columns for modeling.
    
    Returns:
        List of feature column names
    """
    return [
        'rank_diff', 'h2h', 'h2h_count',
        'fav_form', 'under_form', 'form_diff',
        'fav_age', 'under_age', 'tourney_level_enc',
        'fav_rank', 'under_rank', 'round_enc',
        'surface_Clay', 'surface_Grass', 'surface_Hard', 'surface_Carpet'
    ]


if __name__ == "__main__":
    # Example usage
    from data_loader import load_tennis_data, restructure_data
    
    print("Loading and processing data...")
    data = load_tennis_data(range(2020, 2022))  # Small sample for testing
    restructured = restructure_data(data)
    features = prepare_features(restructured)
    
    print("Feature engineering complete!")
    print(f"Features shape: {features.shape}")
    print(f"Upset rate: {features['upset'].mean():.3f}")
