"""
Data loading utilities for tennis upset prediction.

This module handles downloading and preprocessing tennis match data from
Jeff Sackmann's repository.
"""

import pandas as pd
import numpy as np
from typing import Tuple, List
import warnings
warnings.filterwarnings('ignore')


def load_tennis_data(years: range = range(2000, 2025)) -> pd.DataFrame:
    """
    Load and combine ATP and WTA tennis match data.
    
    Args:
        years: Range of years to load data for
        
    Returns:
        Combined DataFrame with ATP and WTA match data
    """
    # Columns to exclude from the model
    columns_to_drop = [
        'winner_ht', 'loser_ht', 'loser_hand', 'winner_hand', 'tourney_name', 
        'winner_ioc', 'loser_ioc', 'loser_seed', 'winner_seed', 'minutes',
        'w_ace', 'w_df', 'w_svpt', 'w_1stIn', 'w_1stWon', 'w_2ndWon', 
        'w_SvGms', 'w_bpSaved', 'w_bpFaced', 'l_ace', 'l_df', 'l_svpt', 
        'l_1stIn', 'l_1stWon', 'l_2ndWon', 'l_SvGms', 'l_bpSaved', 
        'l_bpFaced', 'winner_entry', 'loser_entry', 'score', 'draw_size', 
        'best_of', 'winner_rank_points', 'loser_rank_points'
    ]
    
    combined_data = []
    
    # Base URLs for ATP and WTA datasets
    base_url_atp = "https://raw.githubusercontent.com/JeffSackmann/tennis_atp/master/atp_matches_{}.csv"
    base_url_wta = "https://raw.githubusercontent.com/JeffSackmann/tennis_wta/master/wta_matches_{}.csv"
    
    # Load WTA data
    for year in years:
        wta_url = base_url_wta.format(year)
        try:
            print(f'Reading WTA file: {wta_url}')
            wta_data = pd.read_csv(wta_url)
            wta_data['tour'] = 'WTA'
            wta_data.drop(columns=columns_to_drop, errors='ignore', inplace=True)
            combined_data.append(wta_data)
        except Exception as e:
            print(f'Error reading WTA file for year {year}: {e}')
    
    # Load ATP data
    for year in years:
        atp_url = base_url_atp.format(year)
        try:
            print(f'Reading ATP file: {atp_url}')
            atp_data = pd.read_csv(atp_url)
            atp_data['tour'] = 'ATP'
            atp_data.drop(columns=columns_to_drop, errors='ignore', inplace=True)
            combined_data.append(atp_data)
        except Exception as e:
            print(f'Error reading ATP file for year {year}: {e}')
    
    if not combined_data:
        raise ValueError('No data was loaded. Please check the file paths and filenames.')
    
    # Combine all data
    final_data = pd.concat(combined_data, ignore_index=True)
    print(f"Combined data shape: {final_data.shape}")
    
    return final_data


def restructure_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Restructure data to eliminate temporal leakage and create favorite vs underdog schema.
    
    Args:
        df: Raw tennis match data
        
    Returns:
        Restructured DataFrame with favorite/underdog columns
    """
    # Sort by date to ensure no temporal leakage
    df = df.sort_values(by=['tourney_date', 'match_num']).reset_index(drop=True)
    
    def restructure_match(row):
        # In tennis, lower ranking value means better
        if row['winner_rank'] <= row['loser_rank']:
            # Winner is the favourite
            fav_id = row['winner_id']
            fav_name = row['winner_name']
            fav_age = row['winner_age']
            fav_rank = row['winner_rank']
            
            under_id = row['loser_id']
            under_name = row['loser_name']
            under_age = row['loser_age']
            under_rank = row['loser_rank']
            upset = 0  # Favourite won
        else:
            # Winner is the underdog (upset)
            fav_id = row['loser_id']
            fav_name = row['loser_name']
            fav_age = row['loser_age']
            fav_rank = row['loser_rank']
            
            under_id = row['winner_id']
            under_name = row['winner_name']
            under_age = row['winner_age']
            under_rank = row['winner_rank']
            upset = 1  # Underdog won
        
        return pd.Series({
            'tourney_id': row['tourney_id'],
            'surface': row['surface'],
            'tourney_level': row['tourney_level'],
            'tourney_date': row['tourney_date'],
            'match_num': row['match_num'],
            'round': row['round'],
            'tour': row['tour'],
            'fav_id': fav_id,
            'fav_name': fav_name,
            'fav_age': fav_age,
            'fav_rank': fav_rank,
            'under_id': under_id,
            'under_name': under_name,
            'under_age': under_age,
            'under_rank': under_rank,
            'upset': upset,
        })
    
    return df.apply(restructure_match, axis=1)


def save_data(df: pd.DataFrame, filename: str = 'tennis_data.csv') -> None:
    """
    Save processed data to CSV file.
    
    Args:
        df: Processed DataFrame
        filename: Output filename
    """
    df.to_csv(filename, index=False)
    print(f"Data saved to '{filename}'")


if __name__ == "__main__":
    # Example usage
    print("Loading tennis data...")
    data = load_tennis_data()
    
    print("Restructuring data...")
    restructured = restructure_data(data)
    
    print("Saving data...")
    save_data(restructured, 'data/tennis_data_processed.csv')
