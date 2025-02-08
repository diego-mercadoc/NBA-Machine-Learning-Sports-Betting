import os
import sqlite3
import sys
from datetime import datetime

import numpy as np
import pandas as pd
import toml

sys.path.insert(1, os.path.join(sys.path[0], '../..'))
from src.Utils.Dictionaries import team_index_07, team_index_08, team_index_12, team_index_13, team_index_14, \
    team_index_current

def compute_relative_metrics(frame):
    """Compute relative metrics between home and away teams."""
    # Get all base columns (without .1 suffix for away team stats)
    base_cols = [col for col in frame.columns if not col.endswith('.1') and not col in 
                ['Score', 'Home-Team-Win', 'OU', 'OU-Cover', 'Days-Rest-Home', 'Days-Rest-Away']]
    
    for col in base_cols:
        if col in frame.columns and f"{col}.1" in frame.columns:
            # Compute differences (home - away)
            frame[f"{col}_diff"] = frame[col] - frame[f"{col}.1"]
            # Compute ratios where meaningful (avoid division by zero)
            if frame[f"{col}.1"].abs().mean() > 0:
                frame[f"{col}_ratio"] = frame[col] / frame[f"{col}.1"].replace(0, np.nan)
    
    return frame

def compute_rolling_metrics(frame, window_sizes=[3, 5, 10]):
    """Compute rolling averages for key metrics."""
    frame = frame.copy()
    frame['Date'] = pd.to_datetime(frame['Date'])
    frame.sort_values('Date', inplace=True)
    
    # Key metrics to compute rolling averages for
    metrics = ['Score', 'Home-Team-Win', 'OU-Cover']
    
    for metric in metrics:
        if metric in frame.columns:
            for window in window_sizes:
                frame[f"{metric}_rolling_{window}"] = frame[metric].rolling(
                    window=window, min_periods=1).mean()
    
    return frame

def compute_streak_features(frame):
    """Compute winning/losing streak features."""
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

config = toml.load("../../config.toml")

df = pd.DataFrame
scores = []
win_margin = []
OU = []
OU_Cover = []
games = []
days_rest_away = []
days_rest_home = []
teams_con = sqlite3.connect("../../Data/TeamData.sqlite")
odds_con = sqlite3.connect("../../Data/OddsData.sqlite")

for key, value in config['create-games'].items():
    print(key)
    odds_df = pd.read_sql_query(f"select * from \"odds_{key}_new\"", odds_con, index_col="index")
    team_table_str = key
    year_count = 0
    season = key

    for row in odds_df.itertuples():
        home_team = row[2]
        away_team = row[3]
        date = row[1]

        team_df = pd.read_sql_query(f"select * from \"{date}\"", teams_con, index_col="index")
        if len(team_df.index) == 30:
            scores.append(row[8])
            OU.append(row[4])
            days_rest_home.append(row[10])
            days_rest_away.append(row[11])
            if row[9] > 0:
                win_margin.append(1)
            else:
                win_margin.append(0)

            if row[8] < row[4]:
                OU_Cover.append(0)
            elif row[8] > row[4]:
                OU_Cover.append(1)
            elif row[8] == row[4]:
                OU_Cover.append(2)

            if season == '2007-08':
                home_team_series = team_df.iloc[team_index_07.get(home_team)]
                away_team_series = team_df.iloc[team_index_07.get(away_team)]
            elif season == '2008-09' or season == "2009-10" or season == "2010-11" or season == "2011-12":
                home_team_series = team_df.iloc[team_index_08.get(home_team)]
                away_team_series = team_df.iloc[team_index_08.get(away_team)]
            elif season == "2012-13":
                home_team_series = team_df.iloc[team_index_12.get(home_team)]
                away_team_series = team_df.iloc[team_index_12.get(away_team)]
            elif season == '2013-14':
                home_team_series = team_df.iloc[team_index_13.get(home_team)]
                away_team_series = team_df.iloc[team_index_13.get(away_team)]
            elif season == '2022-23' or season == '2023-24':
                home_team_series = team_df.iloc[team_index_current.get(home_team)]
                away_team_series = team_df.iloc[team_index_current.get(away_team)]
            else:
                try:
                    home_team_series = team_df.iloc[team_index_14.get(home_team)]
                    away_team_series = team_df.iloc[team_index_14.get(away_team)]
                except Exception as e:
                    print(home_team)
                    raise e
            game = pd.concat([home_team_series, away_team_series.rename(
                index={col: f"{col}.1" for col in team_df.columns.values}
            )])
            games.append(game)

odds_con.close()
teams_con.close()

# Create initial DataFrame
season = pd.concat(games, ignore_index=True, axis=1)
season = season.T
frame = season.drop(columns=['TEAM_ID', 'TEAM_ID.1'])
frame['Score'] = np.asarray(scores)
frame['Home-Team-Win'] = np.asarray(win_margin)
frame['OU'] = np.asarray(OU)
frame['OU-Cover'] = np.asarray(OU_Cover)
frame['Days-Rest-Home'] = np.asarray(days_rest_home)
frame['Days-Rest-Away'] = np.asarray(days_rest_away)

# Add enhanced features
print("Computing relative metrics...")
frame = compute_relative_metrics(frame)

print("Computing rolling metrics...")
frame = compute_rolling_metrics(frame)

print("Computing streak features...")
frame = compute_streak_features(frame)

# fix types
for field in frame.columns.values:
    if 'TEAM_' in field or 'Date' in field or field not in frame:
        continue
    frame[field] = frame[field].astype(float)

print("Writing enhanced dataset to database...")
con = sqlite3.connect("../../Data/dataset.sqlite")
frame.to_sql("dataset_2012-24_new", con, if_exists="replace")
con.close()

print("Feature engineering complete!")
