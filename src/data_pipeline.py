import pandas as pd
import numpy as np
import os

# Team ID to name mapping (you'll need to expand this based on your data)
TEAM_ID_TO_NAME = {
    1610612740: 'New Orleans',  # Pelicans
    1610612759: 'San Antonio',  # Spurs
    1610612744: 'Golden State',
    1610612762: 'Utah',
    # Add all other team IDs from your games.csv
    1610612737: 'Atlanta',
    1610612738: 'Boston',
    1610612751: 'Brooklyn',
    1610612766: 'Charlotte',
    1610612741: 'Chicago',
    1610612739: 'Cleveland',
    1610612742: 'Dallas',
    1610612743: 'Denver',
    1610612765: 'Detroit',
    1610612744: 'Golden State',
    1610612745: 'Houston',
    1610612754: 'Indiana',
    1610612746: 'LA Clippers',
    1610612747: 'LA Lakers',
    1610612763: 'Memphis',
    1610612748: 'Miami',
    1610612749: 'Milwaukee',
    1610612750: 'Minnesota',
    1610612740: 'New Orleans',
    1610612752: 'New York',
    1610612760: 'Oklahoma City',
    1610612753: 'Orlando',
    1610612755: 'Philadelphia',
    1610612756: 'Phoenix',
    1610612757: 'Portland',
    1610612758: 'Sacramento',
    1610612759: 'San Antonio',
    1610612761: 'Toronto',
    1610612762: 'Utah',
    1610612764: 'Washington'
}

def load_and_clean_games(path):
    print(f"📂 Loading games data from {path}")
    df = pd.read_csv(path)
    df['GAME_DATE_EST'] = pd.to_datetime(df['GAME_DATE_EST'])
    df['Date'] = df['GAME_DATE_EST'].dt.strftime('%Y-%m-%d')

    # Keep essential columns
    cols = [
        'Date', 'GAME_ID', 'SEASON', 'HOME_TEAM_ID', 'VISITOR_TEAM_ID',
        'PTS_home', 'PTS_away', 'FG_PCT_home', 'FG_PCT_away',
        'FT_PCT_home', 'FT_PCT_away', 'FG3_PCT_home', 'FG3_PCT_away',
        'AST_home', 'AST_away', 'REB_home', 'REB_away', 'HOME_TEAM_WINS'
    ]
    df = df[cols].dropna()
    
    # Add team names for merging with odds data
    df['HOME_TEAM_NAME'] = df['HOME_TEAM_ID'].map(TEAM_ID_TO_NAME)
    df['VISITOR_TEAM_NAME'] = df['VISITOR_TEAM_ID'].map(TEAM_ID_TO_NAME)
    
    # Drop rows where team names couldn't be mapped
    df = df.dropna(subset=['HOME_TEAM_NAME', 'VISITOR_TEAM_NAME'])
    
    print(f"✅ Games data cleaned: {len(df)} rows")
    return df

def load_and_clean_odds(path):
    print(f"📘 Loading odds data from {path}")
    df = pd.read_csv(path)
    df['Date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
    
    # Clean team names and handle home/visitor notation
    df['home_team'] = None
    df['away_team'] = None
    
    for idx, row in df.iterrows():
        if row['home/visitor'] == '@':
            # Current team is visitor, opponent is home
            df.at[idx, 'away_team'] = row['team']
            df.at[idx, 'home_team'] = row['opponent']
        else:
            # Current team is home, opponent is visitor
            df.at[idx, 'home_team'] = row['team']
            df.at[idx, 'away_team'] = row['opponent']
    
    # Keep useful columns
    keep_cols = ['Date', 'home_team', 'away_team', 'moneyLine', 'opponentMoneyLine', 'total', 'spread']
    df = df[[col for col in keep_cols if col in df.columns]].dropna()
    
    print(f"✅ Odds data cleaned: {len(df)} rows")
    return df



def merge_and_featurize(games_path, odds_path, output_path):
    games = load_and_clean_games(games_path)
    odds = load_and_clean_odds(odds_path)

    print("🔄 Merging datasets...")
    # Merge on date and team names
    merged = pd.merge(
        games, 
        odds, 
        left_on=['Date', 'HOME_TEAM_NAME', 'VISITOR_TEAM_NAME'],
        right_on=['Date', 'home_team', 'away_team'],
        how='inner'
    )
    
    if merged.empty:
        print("❌ No matches found after merging! Check team name mappings.")
        # Let's try a more flexible merge - just on date to see what we get
        merged = pd.merge(games, odds, on='Date', how='inner')
        print(f"⚠️ Using date-only merge: {len(merged)} rows")
    
    # Feature engineering - only create features that make sense
    # Basic game stats differences
    if all(col in merged.columns for col in ['PTS_home', 'PTS_away']):
        merged['point_diff'] = merged['PTS_home'] - merged['PTS_away']
    
    if all(col in merged.columns for col in ['FG_PCT_home', 'FG_PCT_away']):
        merged['fg_pct_diff'] = merged['FG_PCT_home'] - merged['FG_PCT_away']
    
    if all(col in merged.columns for col in ['FT_PCT_home', 'FT_PCT_away']):
        merged['ft_pct_diff'] = merged['FT_PCT_home'] - merged['FT_PCT_away']
    
    if all(col in merged.columns for col in ['FG3_PCT_home', 'FG3_PCT_away']):
        merged['fg3_pct_diff'] = merged['FG3_PCT_home'] - merged['FG3_PCT_away']
    
    if all(col in merged.columns for col in ['AST_home', 'AST_away']):
        merged['ast_diff'] = merged['AST_home'] - merged['AST_away']
    
    if all(col in merged.columns for col in ['REB_home', 'REB_away']):
        merged['reb_diff'] = merged['REB_home'] - merged['REB_away']
    
    # Odds features
    if all(col in merged.columns for col in ['moneyLine', 'opponentMoneyLine']):
        def moneyline_to_prob(moneyline):
            if pd.isna(moneyline):
                return np.nan
            if moneyline > 0:
                return 100 / (moneyline + 100)
            else:
                return abs(moneyline) / (abs(moneyline) + 100)
        
        merged['home_win_prob'] = merged['moneyLine'].apply(moneyline_to_prob)
        merged['away_win_prob'] = merged['opponentMoneyLine'].apply(moneyline_to_prob)
        merged['implied_prob_diff'] = merged['home_win_prob'] - merged['away_win_prob']
    
    # Drop rows with missing target or essential features
    essential_cols = ['HOME_TEAM_WINS'] + [col for col in merged.columns if 'diff' in col]
    merged = merged.dropna(subset=essential_cols)
    
    print(f"📊 Final dataset columns: {list(merged.columns)}")
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    merged.to_csv(output_path, index=False)
    print(f"✅ Merged dataset saved to {output_path} with {len(merged)} rows")
    
    return merged

if __name__ == "__main__":
    merged_data = merge_and_featurize(
        games_path="../data/raw/games.csv",
        odds_path="../data/raw/oddsData.csv",
        output_path="../data/processed/merged_features.csv"
    )