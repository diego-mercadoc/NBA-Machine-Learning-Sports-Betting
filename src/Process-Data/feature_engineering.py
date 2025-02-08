"""
Feature engineering module for NBA game prediction models.
Contains functions for creating advanced features from raw game data.
"""

from typing import List, Dict, Any
import numpy as np
import pandas as pd


def compute_relative_metrics(frame: pd.DataFrame, exclude_cols: List[str] = None) -> pd.DataFrame:
    """
    Compute relative metrics between home and away teams.
    
    Args:
        frame: DataFrame containing team statistics
        exclude_cols: List of columns to exclude from relative metric computation
        
    Returns:
        DataFrame with added relative metrics
    """
    if exclude_cols is None:
        exclude_cols = ['Score', 'Home-Team-Win', 'OU', 'OU-Cover', 'Days-Rest-Home', 'Days-Rest-Away', 
                       'TEAM_NAME', 'Date', 'TEAM_ID']
    
    # Get all base columns (without .1 suffix for away team stats)
    base_cols = [col for col in frame.columns if not col.endswith('.1') and col not in exclude_cols]
    
    for col in base_cols:
        away_col = f"{col}.1"
        if col in frame.columns and away_col in frame.columns:
            # Compute differences (home - away)
            frame[f"{col}_diff"] = frame[col] - frame[away_col]
            
            # Compute ratios where meaningful (avoid division by zero)
            if frame[away_col].abs().mean() > 0:
                frame[f"{col}_ratio"] = frame[col] / frame[away_col].replace(0, np.nan)
    
    return frame


def compute_rolling_metrics(frame: pd.DataFrame, 
                          metrics: List[str] = None,
                          window_sizes: List[int] = None) -> pd.DataFrame:
    """
    Compute rolling averages for specified metrics.
    
    Args:
        frame: DataFrame containing game data
        metrics: List of metrics to compute rolling averages for
        window_sizes: List of window sizes for rolling averages
        
    Returns:
        DataFrame with added rolling metrics
    """
    if metrics is None:
        metrics = ['Score', 'Home-Team-Win', 'OU-Cover']
    if window_sizes is None:
        window_sizes = [3, 5, 10]
    
    frame = frame.copy()
    frame['Date'] = pd.to_datetime(frame['Date'])
    frame.sort_values('Date', inplace=True)
    
    for metric in metrics:
        if metric in frame.columns:
            for window in window_sizes:
                frame[f"{metric}_rolling_{window}"] = frame[metric].rolling(
                    window=window, min_periods=1).mean()
    
    return frame


def compute_streak_features(frame: pd.DataFrame) -> pd.DataFrame:
    """
    Compute winning/losing streak features.
    
    Args:
        frame: DataFrame containing game data
        
    Returns:
        DataFrame with added streak features
    """
    frame = frame.copy()
    frame['Date'] = pd.to_datetime(frame['Date'])
    frame.sort_values('Date', inplace=True)
    
    # Initialize streak columns
    frame['win_streak'] = 0
    frame['loss_streak'] = 0
    
    # Compute streaks
    streak = 0
    for i in range(1, len(frame)):
        if frame.iloc[i]['Home-Team-Win'] == frame.iloc[i-1]['Home-Team-Win']:
            streak = streak + 1 if frame.iloc[i]['Home-Team-Win'] == 1 else streak - 1
        else:
            streak = 1 if frame.iloc[i]['Home-Team-Win'] == 1 else -1
        
        frame.iloc[i, frame.columns.get_loc('win_streak' if streak > 0 else 'loss_streak')] = abs(streak)
    
    return frame


def compute_head_to_head_features(frame: pd.DataFrame, lookback_days: int = 365) -> pd.DataFrame:
    """
    Compute head-to-head statistics between teams.
    
    Args:
        frame: DataFrame containing game data
        lookback_days: Number of days to look back for head-to-head stats
        
    Returns:
        DataFrame with added head-to-head features
    """
    frame = frame.copy()
    frame['Date'] = pd.to_datetime(frame['Date'])
    frame.sort_values('Date', inplace=True)
    
    # Initialize head-to-head columns
    frame['h2h_wins'] = 0
    frame['h2h_points_diff'] = 0
    frame['h2h_games_count'] = 0
    
    for i in range(len(frame)):
        current_date = frame.iloc[i]['Date']
        home_team = frame.iloc[i]['TEAM_NAME']
        away_team = frame.iloc[i]['TEAM_NAME.1']
        
        # Get previous matchups within lookback period
        mask = (
            (frame['Date'] < current_date) & 
            (frame['Date'] >= current_date - pd.Timedelta(days=lookback_days)) &
            (
                ((frame['TEAM_NAME'] == home_team) & (frame['TEAM_NAME.1'] == away_team)) |
                ((frame['TEAM_NAME'] == away_team) & (frame['TEAM_NAME.1'] == home_team))
            )
        )
        
        h2h_games = frame[mask]
        
        if len(h2h_games) > 0:
            # Count games
            frame.iloc[i, frame.columns.get_loc('h2h_games_count')] = len(h2h_games)
            
            # Calculate win rate for current home team
            home_wins = h2h_games[
                ((h2h_games['TEAM_NAME'] == home_team) & (h2h_games['Home-Team-Win'] == 1)) |
                ((h2h_games['TEAM_NAME.1'] == home_team) & (h2h_games['Home-Team-Win'] == 0))
            ]
            frame.iloc[i, frame.columns.get_loc('h2h_wins')] = len(home_wins) / len(h2h_games)
            
            # Calculate average point differential for current home team
            h2h_games['points_diff'] = np.where(
                h2h_games['TEAM_NAME'] == home_team,
                h2h_games['Score'],
                -h2h_games['Score']
            )
            frame.iloc[i, frame.columns.get_loc('h2h_points_diff')] = h2h_games['points_diff'].mean()
    
    return frame


def apply_all_features(frame: pd.DataFrame, config: Dict[str, Any] = None) -> pd.DataFrame:
    """
    Apply all feature engineering steps based on configuration.
    
    Args:
        frame: DataFrame containing raw game data
        config: Feature engineering configuration dictionary
        
    Returns:
        DataFrame with all engineered features
    """
    if config is None:
        from src.Train_Models.model_config import FEATURE_CONFIG
        config = FEATURE_CONFIG
    
    print("Computing relative metrics...")
    if config.get('relative_metrics', True):
        frame = compute_relative_metrics(frame)
    
    print("Computing rolling metrics...")
    if config.get('rolling_windows'):
        frame = compute_rolling_metrics(frame, window_sizes=config['rolling_windows'])
    
    print("Computing streak features...")
    if config.get('streak_features', True):
        frame = compute_streak_features(frame)
    
    print("Computing head-to-head features...")
    if config.get('head_to_head', True):
        frame = compute_head_to_head_features(frame)
    
    return frame 