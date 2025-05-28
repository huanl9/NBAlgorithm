import pandas as pd
import numpy as np
from flask import Flask, request, render_template
import joblib
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

# Define the Neural Network class (must match training architecture)
class Net(nn.Module):
    def __init__(self, input_size=36):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x

# Load historical data
games = pd.read_csv('./historical-nba-data-and-player-box-scores/versions/166/Games.csv')
team_stats = pd.read_csv('./historical-nba-data-and-player-box-scores/versions/166/TeamStatistics.csv')
player_stats = pd.read_csv('./historical-nba-data-and-player-box-scores/versions/166/PlayerStatistics.csv', low_memory=False)

# Convert dates to datetime
games['gameDate'] = pd.to_datetime(games['gameDate'])
team_stats['gameDate'] = pd.to_datetime(team_stats['gameDate'])
player_stats['gameDate'] = pd.to_datetime(player_stats['gameDate'])

# Add season column
def get_season(date):
    year = date.year
    return year if date.month >= 10 else year - 1
games['season'] = games['gameDate'].apply(get_season)
team_stats['season'] = team_stats['gameDate'].apply(get_season)

# Filter data to post-2010 for consistency
games = games[games['gameDate'].dt.year >= 2010]
team_stats = team_stats[team_stats['gameDate'].dt.year >= 2010]
player_stats = player_stats[player_stats['gameId'].isin(games['gameId'])]

# Merge team_stats with games to determine home/away
team_stats = pd.merge(team_stats, games[['gameId', 'hometeamId', 'awayteamId']], on='gameId', how='left')
team_stats['homeOrAway'] = np.where(team_stats['teamId'] == team_stats['hometeamId'], 'HOME', 'AWAY')

# Assign teamId to player_stats
player_stats = pd.merge(player_stats, games[['gameId', 'hometeamName', 'awayteamName', 'hometeamId', 'awayteamId']], on='gameId', how='left')
player_stats['teamId'] = np.where(player_stats['playerteamName'] == player_stats['hometeamName'], player_stats['hometeamId'], player_stats['awayteamId'])
player_stats = player_stats.dropna(subset=['teamId'])

# Compute top 3 average points per game
top3_avg_points_df = player_stats.groupby(['gameId', 'teamId']).apply(
    lambda x: x.nlargest(3, 'points')['points'].mean() if len(x) >= 3 else x['points'].mean(),
    include_groups=False
).reset_index(name='top3_avg_points')

# Feature engineering functions
def calculate_advanced_stats(team_stats_df):
    def calc_poss(row):
        return row['fieldGoalsAttempted'] - row['reboundsOffensive'] + row['turnovers'] + 0.4 * row['freeThrowsAttempted']
    team_stats_df['Poss_team'] = team_stats_df.apply(calc_poss, axis=1)
    game_poss = team_stats_df.groupby('gameId')['Poss_team'].mean().reset_index()
    game_poss.columns = ['gameId', 'Poss']
    team_stats_df = pd.merge(team_stats_df, game_poss, on='gameId')
    team_stats_df['ORtg'] = (team_stats_df['teamScore'] / team_stats_df['Poss']) * 100
    team_stats_df['DRtg'] = (team_stats_df['opponentScore'] / team_stats_df['Poss']) * 100
    team_stats_df['Pace'] = team_stats_df['Poss']
    return team_stats_df

def calculate_win_rate(team_id, date, team_stats_df, home_away):
    past_games = team_stats_df[
        (team_stats_df['teamId'] == team_id) &
        (team_stats_df['gameDate'] < date) &
        (team_stats_df['homeOrAway'] == home_away)
    ]
    return past_games['win'].mean() if not past_games.empty else 0.5

def calculate_current_season_win_rate(team_id, date, season, team_stats_df):
    past_games = team_stats_df[
        (team_stats_df['teamId'] == team_id) &
        (team_stats_df['gameDate'] < date) &
        (team_stats_df['season'] == season)
    ]
    return past_games['win'].mean() if not past_games.empty else 0.5

def calculate_opponent_features(team_id, date, team_stats_df, N=5):
    past_games = team_stats_df[
        (team_stats_df['opponentTeamId'] == team_id) &
        (team_stats_df['gameDate'] < date)
    ].sort_values('gameDate', ascending=False).head(N)
    if past_games.empty:
        return np.zeros(3)
    return np.array([past_games['teamScore'].mean(), past_games['ORtg'].mean(), past_games['DRtg'].mean()])

def calculate_features(team_id, date, team_stats_df, top3_avg_points_df, N=5, home_away=None):
    if home_away:
        past_games = team_stats_df[
            (team_stats_df['teamId'] == team_id) &
            (team_stats_df['gameDate'] < date) &
            (team_stats_df['homeOrAway'] == home_away)
        ].sort_values('gameDate', ascending=False).head(N)
    else:
        past_games = team_stats_df[
            (team_stats_df['teamId'] == team_id) &
            (team_stats_df['gameDate'] < date)
        ].sort_values('gameDate', ascending=False).head(N)
    if past_games.empty:
        return np.zeros(13)
    past_games = pd.merge(past_games, top3_avg_points_df, on=['gameId', 'teamId'], how='left')
    opponent_features = calculate_opponent_features(team_id, date, team_stats_df, N)
    features = [
        past_games['teamScore'].mean(),
        past_games['opponentScore'].mean(),
        past_games['win'].mean(),
        past_games['assists'].mean(),
        past_games['reboundsTotal'].mean(),
        past_games['turnovers'].mean(),
        past_games['ORtg'].mean(),
        past_games['DRtg'].mean(),
        past_games['Pace'].mean(),
        past_games['top3_avg_points'].mean()
    ]
    features = np.concatenate([features, opponent_features])
    return np.array([0 if np.isnan(f) else f for f in features])

# Apply advanced stats
team_stats = calculate_advanced_stats(team_stats)

# Load models and scaler
rf_model = joblib.load('random_forest_model.joblib')
lr_model = joblib.load('logistic_regression_model.joblib')
svm_model = joblib.load('svm_model.joblib')
xgb_model = joblib.load('xgboost_model.joblib')
voting_model = joblib.load('voting_classifier_model.joblib')
scaler = joblib.load('scaler.joblib')

# Load Neural Network
nn_model = Net(input_size=36)  # Assuming 36 features as per training
nn_model.load_state_dict(torch.load('neural_network_model.pth'))
nn_model.eval()

# Team name to ID mapping
home_teams = games[['hometeamId', 'hometeamName']].rename(columns={'hometeamId': 'teamId', 'hometeamName': 'teamName'})
away_teams = games[['awayteamId', 'awayteamName']].rename(columns={'awayteamId': 'teamId', 'awayteamName': 'teamName'})
all_teams = pd.concat([home_teams, away_teams]).drop_duplicates()
team_name_to_id = {row['teamName']: row['teamId'] for _, row in all_teams.iterrows()}

# Initialize Flask app
app = Flask(__name__)

@app.route('/')
def home():
    team_names = sorted(team_name_to_id.keys())
    return render_template('index.html', team_names=team_names)

@app.route('/predict', methods=['POST'])
def predict():
    home_team_name = request.form['home_team']
    away_team_name = request.form['away_team']
    date_str = request.form['date']
    date = pd.to_datetime(date_str)
    home_id = team_name_to_id[home_team_name]
    away_id = team_name_to_id[away_team_name]

    # Filter data to games before the input date
    past_team_stats = team_stats[team_stats['gameDate'] < date]

    # Calculate features
    home_features = calculate_features(home_id, date, past_team_stats, top3_avg_points_df, N=5, home_away='HOME')
    away_features = calculate_features(away_id, date, past_team_stats, top3_avg_points_df, N=5, home_away='AWAY')

    # Additional features
    season = get_season(date)
    home_win_rate = calculate_win_rate(home_id, date, past_team_stats, 'HOME')
    away_win_rate = calculate_win_rate(away_id, date, past_team_stats, 'AWAY')
    home_current_win_rate = calculate_current_season_win_rate(home_id, date, season, past_team_stats)
    away_current_win_rate = calculate_current_season_win_rate(away_id, date, season, past_team_stats)
    last_game_home = past_team_stats[past_team_stats['teamId'] == home_id].sort_values('gameDate', ascending=False).head(1)
    rest_days_home = (date - last_game_home['gameDate'].iloc[0]).days if not last_game_home.empty else 10
    last_game_away = past_team_stats[past_team_stats['teamId'] == away_id].sort_values('gameDate', ascending=False).head(1)
    rest_days_away = (date - last_game_away['gameDate'].iloc[0]).days if not last_game_away.empty else 10
    past_h2h_games = games[((games['hometeamId'] == home_id) & (games['awayteamId'] == away_id)) |
                           ((games['hometeamId'] == away_id) & (games['awayteamId'] == home_id)) &
                           (games['gameDate'] < date)]
    h2h_win_rate = (past_h2h_games['winner'] == home_id).mean() if not past_h2h_games.empty else 0.5
    is_playoff = 0  # Assuming regular season
    ortg_diff = home_features[6] - away_features[7]
    drtg_diff = home_features[7] - away_features[6]

    # Construct feature vector
    features = np.concatenate([
        home_features, away_features,
        [home_win_rate, away_win_rate, home_current_win_rate, away_current_win_rate,
         rest_days_home, rest_days_away, h2h_win_rate, is_playoff, ortg_diff, drtg_diff]
    ])

    # Scale features
    scaled_features = scaler.transform([features])

    # Get predictions from all models
    probs = []
    for model in [rf_model, lr_model, svm_model, xgb_model, voting_model]:
        prob = model.predict_proba(scaled_features)[0, 1]  # Probability of home team winning
        probs.append(prob)
    input_tensor = torch.tensor(scaled_features, dtype=torch.float32)
    prob_nn = nn_model(input_tensor).item()
    probs.append(prob_nn)

    # Average probabilities
    average_prob = np.mean(probs)
    winner = home_team_name if average_prob > 0.5 else away_team_name
    prediction = f"The {winner} are predicted to win."

    team_names = sorted(team_name_to_id.keys())
    return render_template('index.html', prediction=prediction, team_names=team_names)

if __name__ == '__main__':
    app.run(debug=True)