import os
import requests
import streamlit as st
import pandas as pd
import numpy as np
import logging
import json
from datetime import datetime, timedelta
from typing import Dict, List
from config import GEMINI_API_KEY
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import google.generativeai as genai

# Set page configuration
st.set_page_config(
    page_title="MLB Data Explorer",
    page_icon="⚾",
    layout="wide",
    initial_sidebar_state="expanded"
)



# Custom CSS for purple theme and improved text visibility
def set_advanced_theme():
    st.markdown("""
    <style>
    /* Advanced Gradient Background */
    .stApp {
        background: linear-gradient(
            -45deg, 
            #ee7752, 
            #e73c7e, 
            #23a6d5, 
            #23d5ab
        );
        background-size: 400% 400%;
        animation: gradientBG 15s ease infinite;
        min-height: 100vh;
    }
    @keyframes gradientBG {
        0% {background-position: 0% 50%;}
        50% {background-position: 100% 50%;}
        100% {background-position: 0% 50%;}
    }
    /* Improved Text and Element Visibility */
    .css-1544g2n { color: #000 !important; }
    .st-emotion-cache-1vzeuhh, .st-emotion-cache-16idsys p, .st-emotion-cache-pkbazv {
        color: #000 !important;
        font-weight: 500 !important;
    }
    .stSelectbox div div div, .stTextInput div div {
        color: #000 !important;
        background-color: rgba(255, 255, 255, 0.9) !important;
    }
    .stSelectbox div div div:hover {
        background-color: rgba(255, 255, 255, 0.95) !important;
    }
    .stTextInput input, .stSelectbox select {
        color: #000 !important;
    }
    .stunning-container {
        background: rgba(255, 255, 255, 0.25);
        backdrop-filter: blur(15px);
        border-radius: 20px;
        border: 2px solid rgba(255, 255, 255, 0.3);
        padding: 30px;
        margin: 20px 0;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
    }
    h1, h2, h3, h4, h5, h6, .streamlit-expanderHeader {
        color: #000 !important;
        text-shadow: 1px 1px 2px rgba(255,255,255,0.5);
    }
    .css-1avcm0n { color: #000 !important; }
    .st-emotion-cache-1y4p8pa { color: #000 !important; }
    .element-container, .stMarkdown { color: #000 !important; }
    .stButton > button {
        background: linear-gradient(135deg, #6a11cb 0%, #2575fc 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 50px !important;
        padding: 10px 20px !important;
        font-weight: bold !important;
        transition: all 0.3s ease !important;
    }
    .stButton > button:hover {
        transform: scale(1.05) !important;
        box-shadow: 0 10px 20px rgba(0,0,0,0.2) !important;
    }
    .dataframe { background-color: rgba(255, 255, 255, 0.9) !important; }
    .dataframe th { background-color: #4a4a4a !important; color: white !important; }
    .dataframe td { color: #000 !important; }
    .stSuccess, .stError, .stInfo {
        color: #000 !important;
        background-color: rgba(255, 255, 255, 0.9) !important;
    }
    input[type="text"], select, textarea {
        color: #000 !important;
        background-color: rgba(255, 255, 255, 0.9) !important;
    }
    .css-1d391kg { background-color: rgba(255, 255, 255, 0.1) !important; }
    .css-1d391kg .streamlit-expanderHeader { color: #000 !important; }
    .streamlit-expanderHeader {
        background-color: rgba(255, 255, 255, 0.1) !important;
        color: #000 !important;
    }
    </style>
    """, unsafe_allow_html=True)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =============================
# MLB Data Retrieval Functionality
# =============================
class MLBDataRetriever:
    """
    Comprehensive MLB Data Retriever with multiple API endpoints
    """
    BASE_URL = "https://statsapi.mlb.com/api/v1.1"

    def __init__(self, api_key: str):
        self.api_key = api_key
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-pro')

    def get_game_data(self, game_pk: str, game_type: str = "R") -> Dict:
        endpoints = [
            f"{self.BASE_URL}/game/{game_pk}/feed/live",
            f"https://statsapi.mlb.com/api/v1/game/{game_pk}",
            f"https://statsapi.mlb.com/api/v1.1/game/{game_pk}/feed/live"
        ]
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)',
            'Accept': 'application/json'
        }
        for endpoint in endpoints:
            try:
                response = requests.get(endpoint, headers=headers, timeout=10)
                if response.status_code == 200:
                    game_data = response.json()
                    parsed_data = self._parse_game_data(game_data, game_pk)
                    if parsed_data:
                        return parsed_data
                else:
                    logger.error(f"Failed to retrieve game data from {endpoint}. Status code: {response.status_code}")
            except requests.RequestException as e:
                logger.error(f"Error accessing {endpoint}: {e}")
            except Exception as e:
                logger.error(f"Unexpected error with {endpoint}: {e}")
        return None

    def _parse_game_data(self, game_data: Dict, game_pk: str) -> Dict:
        try:
            parsing_strategies = [
                lambda: {
                    "Game ID": game_pk,
                    "Home Team": game_data.get('gameData', {}).get('teams', {}).get('home', {}).get('name'),
                    "Away Team": game_data.get('gameData', {}).get('teams', {}).get('away', {}).get('name'),
                    "Game Type": game_data.get('gameData', {}).get('game', {}).get('type')
                },
                lambda: {
                    "Game ID": game_pk,
                    "Home Team": game_data.get('teams', {}).get('home', {}).get('team', {}).get('name'),
                    "Away Team": game_data.get('teams', {}).get('away', {}).get('team', {}).get('name'),
                    "Game Type": game_data.get('gameType')
                },
                lambda: {
                    "Game ID": game_pk,
                    "Home Team": game_data.get('dates', [{}])[0].get('games', [{}])[0].get('teams', {}).get('home', {}).get('team', {}).get('name'),
                    "Away Team": game_data.get('dates', [{}])[0].get('games', [{}])[0].get('teams', {}).get('away', {}).get('team', {}).get('name'),
                    "Game Type": game_data.get('dates', [{}])[0].get('games', [{}])[0].get('gameType')
                }
            ]
            for strategy in parsing_strategies:
                game_info = strategy()
                if game_info.get('Home Team') and game_info.get('Away Team'):
                    return game_info
            return None
        except Exception as e:
            logger.error(f"Error parsing game data: {e}")
            return None

    def get_team_roster(self, team_id: int, season: int = None) -> List:
        if not season:
            season = datetime.now().year
        url = f"https://statsapi.mlb.com/api/v1/teams/{team_id}/roster?season={season}"
        try:
            response = requests.get(url)
            data = response.json()
            roster = []
            for player in data.get('roster', []):
                player_info = {
                    "Name": player.get('person', {}).get('fullName'),
                    "Position": player.get('position', {}).get('name'),
                    "Number": player.get('jerseyNumber')
                }
                roster.append(player_info)
            return roster
        except Exception as e:
            logger.error(f"Error retrieving team roster: {e}")
            return []

    def get_player_details(self, player_id: int, season: int = None) -> Dict:
        if not season:
            season = datetime.now().year
        url = f"https://statsapi.mlb.com/api/v1/people/{player_id}?season={season}"
        try:
            response = requests.get(url)
            data = response.json()
            player_info = {
                "Full Name": data.get('people', [{}])[0].get('fullName'),
                "Primary Position": data.get('people', [{}])[0].get('primaryPosition', {}).get('name'),
                "Birth Date": data.get('people', [{}])[0].get('birthDate'),
                "Height": data.get('people', [{}])[0].get('height'),
                "Weight": data.get('people', [{}])[0].get('weight'),
                "Batting Side": data.get('people', [{}])[0].get('batSide', {}).get('description'),
                "Throwing Side": data.get('people', [{}])[0].get('pitchHand', {}).get('description')
            }
            return player_info
        except Exception as e:
            logger.error(f"Error retrieving player details: {e}")
            return None

    def get_recent_games(self, game_type: str = "R") -> List:
        url = f"https://statsapi.mlb.com/api/v1/schedule?sportId=1&gameType={game_type}"
        try:
            response = requests.get(url)
            data = response.json()
            if data.get('dates') and data['dates'][0].get('games'):
                games = []
                for game in data['dates'][0]['games']:
                    game_info = {
                        "Game ID": game['gamePk'],
                        "Home Team": game['teams']['home']['team']['name'],
                        "Away Team": game['teams']['away']['team']['name'],
                        "Game Type": game_type,
                        "Game Status": game.get('status', {}).get('detailedState', ' Unknown')
                    }
                    games.append(game_info)
                return games
            else:
                return []
        except Exception as e:
            logger.error(f"Error retrieving games: {e}")
            return []

    def find_recent_games(self, days_range: int = 7, game_type: str = "R") -> List:
        all_games = []
        for i in range(-days_range, days_range + 1):
            check_date = datetime.now() + timedelta(days=i)
            formatted_date = check_date.strftime('%Y-%m-%d')
            try:
                url = f"https://statsapi.mlb.com/api/v1/schedule?sportId=1&date={formatted_date}&gameType={game_type}"
                response = requests.get(url)
                data = response.json()
                if data.get('dates') and data['dates'][0].get('games'):
                    for game in data['dates'][0]['games']:
                        game_info = {
                            "Game ID": game['gamePk'],
                            "Home Team": game['teams']['home']['team']['name'],
                            "Away Team": game['teams']['away']['team']['name'],
                            "Game Type": game_type,
                            "Game Status": game.get('status', {}).get('detailedState', 'Unknown')
                        }
                        all_games.append(game_info)
            except Exception as e:
                logger.warning(f"Error checking date {formatted_date}: {e}")
        return all_games

# =============================
# MLB Predictor Functionality
# =============================
class MLBPredictor:
    def __init__(self):
        # Load and preprocess historical MLB dataset
        self.historical_data = self.load_historical_data()
        self.model = None
    
    def load_historical_data(self):
        """
        Load and preprocess historical MLB game data.
        Here we generate mock data for demonstration.
        """
        columns = [
            'home_team_win_rate', 
            'away_team_win_rate', 
            'home_team_recent_performance',
            'away_team_recent_performance',
            'home_team_player_stats',
            'away_team_player_stats',
            'result'
        ]
        data = pd.DataFrame(np.random.rand(1000, len(columns)), columns=columns)
        return data
    
    def train_win_probability_model(self):
        """
        Train model to predict game win probability.
        """
        X = self.historical_data.drop('result', axis=1)
        y = (self.historical_data['result'] > 0.5).astype(int) 
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        self.model = RandomForestClassifier(n_estimators=100)
        self.model.fit(X_train_scaled, y_train)
    
    def predict_win_probability(self, game_data):
        """
        Predict win probability for a specific game.
        Expects game_data as a list of features in the same order as training.
        """
        if self.model is None:
            self.train_win_probability_model()
        # Note: In a full implementation, you should persist and reuse your scaler.
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform([game_data])
        probabilities = self.model.predict_proba(scaled_data)[0]
        return {
            'home_team_win_prob': probabilities[1] * 100,
            'away_team_win_prob': probabilities[0] * 100
        }

# Visualization Integration for Predictor
def create_win_probability_chart(probabilities):
    import plotly.express as px
    df = pd.DataFrame({
        'Team': ['Home Team', 'Away Team'],
        'Win Probability': [
            probabilities['home_team_win_prob'], 
            probabilities['away_team_win_prob']
        ]
    })
    fig = px.bar(
        df, 
        x='Team', 
        y='Win Probability', 
        title='Game Win Probability',
        color='Team'
    )
    return fig

def display_win_prediction(game_id):
    """
    Display win prediction for a given game.
    Uses a dummy fetch_game_data for predictor input.
    """
    predictor = MLBPredictor()
    # Dummy function to simulate game data fetching for predictor
    def fetch_game_data_for_predictor(game_id):
        # Return a list of 6 random features in the expected order.
        return list(np.random.rand(6))
    game_data = fetch_game_data_for_predictor(game_id)
    probabilities = predictor.predict_win_probability(game_data)
    st.subheader("Win Probability Prediction")
    fig = create_win_probability_chart(probabilities)
    st.plotly_chart(fig)
    st.markdown(f"""
    ### Game Prediction Insights
    - **Home Team Win Probability**: {probabilities['home_team_win_prob']:.2f}%
    - **Away Team Win Probability**: {probabilities['away_team_win_prob']:.2f}%
    """)

def create_player_performance_trend(player_id):
    """
    Create interactive player performance trend chart.
    Uses a dummy data function for demonstration.
    """
    def fetch_player_historical_data(player_id):
        dates = pd.date_range(end=datetime.today(), periods=30)
        data = pd.DataFrame({
            'date': dates,
            'batting_average': np.random.rand(30),
            'home_runs': np.random.randint(0, 5, 30),
            'rbi': np.random.randint(0, 10, 30)
        })
        return data
    player_data = fetch_player_historical_data(player_id)
    import plotly.express as px
    fig = px.line(
        player_data, 
        x='date', 
        y=['batting_average', 'home_runs', 'rbi'],
        title=f'Player Performance Trend'
    )
    return fig

def team_comparative_analytics(team1, team2):
    """
    Create team comparative analytics visualization.
    Uses a dummy data function for demonstration.
    """
    def fetch_team_comparative_data(team1, team2):
        data = pd.DataFrame({
            'metric': ['Offense', 'Defense', 'Pitching'],
            'team1_value': np.random.rand(3) * 100,
            'team2_value': np.random.rand(3) * 100
        })
        return data
    team_data = fetch_team_comparative_data(team1, team2)
    import plotly.express as px
    fig = px.bar(
        team_data, 
        x='metric', 
        y=['team1_value', 'team2_value'],
        title='Team Comparative Analytics',
        barmode='group'
    )
    return fig

# =============================
# Streamlit Page Functions
# =============================
def home_page():
    st.title("⚾ MLB Fan Companion - AI-Powered Baseball Insights App")
    st.markdown("""
    ## Welcome to the MLB Data Explorer!
    
    ### 🏆 Project Overview
    This interactive application provides comprehensive insights into Major League Baseball (MLB) data, 
    offering a deep dive into game statistics, team rosters, player details, recent game information,
    and predictive analytics.
    
    ### 🌟 Key Features
    - **Game Data**: Retrieve detailed information about specific MLB games
    - **Team Roster**: Explore current rosters for various MLB teams
    - **Player Details**: Get in-depth information about individual players
    - **Recent Games**: Stay updated with the latest MLB game information
    - **Win Prediction**: Predict game win probabilities using historical data
    - **Player & Team Analytics**: Visualize performance trends and comparative analytics
    
    ### 🛠 Technologies Used
    - Streamlit
    - MLB Stats API
    - Python (scikit-learn, Pandas, Plotly)
    
    ### 📊 How to Use
    1. Select a feature from the sidebar
    2. Choose your desired options
    3. Click the action button to retrieve data or predictions
    
    ### 🚀 Explore and Enjoy!
    """)

def game_data_page(mlb_data_retriever):
    st.subheader("Game ID Lookup")
    sample_ids = ["662025", "661975", "661980", "663142", "663141"]
    game_id = st.selectbox("Select a Game ID", ["Sample IDs"] + sample_ids + ["Custom ID"])
    if game_id == "Custom ID":
        game_id = st.text_input("Enter Custom Game ID")
    elif game_id == "Sample IDs":
        game_id = sample_ids[0]
    game_type = st.selectbox("Game Type", ["Regular Season (R)", "Postseason (P)"])
    api_game_type = "R" if game_type == "Regular Season (R)" else "P"
    if st.button("Fetch Game Data"):
        game_data = mlb_data_retriever.get_game_data(game_id, game_type=api_game_type)
        if game_data:
            st.success("Game Data Retrieved Successfully!")
            for key, value in game_data.items():
                st.write(f"**{key}**: {value}")
        else:
            st.error(f"No data found for Game ID: {game_id}")

def team_roster_page(mlb_data_retriever):
    st.subheader("MLB Team Rosters")
    team_ids = {
        "Los Angeles Dodgers": 119,
        "New York Yankees": 147,
        "Boston Red Sox": 111,
        "Chicago Cubs": 112,
        "Houston Astros": 117
    }
    col1, col2 = st.columns(2)
    with col1:
        selected_team = st.selectbox("Select a Team", list(team_ids.keys()))
    with col2:
        selected_season = st.number_input(
            "Select Season",
            min_value=2000,
            max_value=datetime.now().year,
            value=datetime.now().year
        )
    if st.button("Get Roster"):
        roster = mlb_data_retriever.get_team_roster(team_ids[selected_team], season=selected_season)
        if roster:
            st.success("Roster Details Retrieved Successfully!")
            roster_df = pd.DataFrame(roster)
            st.dataframe(roster_df)
            st.subheader("Roster Statistics")
            st.write(f"Total Players: {len(roster)}")
            position_counts = roster_df['Position'].value_counts()
            st.bar_chart(position_counts)
        else:
            st.warning("No roster data available")

def player_details_page(mlb_data_retriever):
    st.subheader("Player Information")
    sample_player_ids = {
        "Shohei Ohtani": 660271,
        "Mike Trout": 545361,
        "Mookie Betts": 605141
    }
    col1, col2 = st.columns(2)
    with col1:
        selected_player = st.selectbox("Select a Sample Player", list(sample_player_ids.keys()))
    with col2:
        player_id = st.number_input("Or Enter Custom Player ID", value=sample_player_ids[selected_player])
    selected_season = st.number_input(
        "Select Season",
        min_value=2000,
        max_value=datetime.now().year,
        value=datetime.now().year
    )
    if st.button("Get Player Details"):
        player_details = mlb_data_retriever.get_player_details(player_id, season=selected_season)
        if player_details:
            st.success("Player Details Retrieved Successfully!")
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Basic Information")
                for key in ["Full Name", "Primary Position", "Birth Date"]:
                    st.write(f"**{key}**: {player_details.get(key, 'N/A')}")
            with col2:
                st.subheader("Physical Attributes")
                for key in ["Height", "Weight", "Batting Side", "Throwing Side"]:
                    st.write(f"**{key}**: {player_details.get(key, 'N/A')}")
        else:
            st.error(f"No details found for Player ID: {player_id}")

def recent_games_page(mlb_data_retriever):
    st.subheader("Recent MLB Games")
    col1, col2 = st.columns(2)
    with col1:
        days_range = st.slider("Days to Check", min_value=1, max_value=30, value=7)
    with col2:
        game_type = st.selectbox("Game Type", ["Regular Season (R)", "Postseason (P)"])
    api_game_type = "R" if game_type == "Regular Season (R)" else "P"
    if st.button("Get Games"):
        try:
            recent_games = mlb_data_retriever.find_recent_games(days_range=days_range, game_type=api_game_type)
            if recent_games:
                games_df = pd.DataFrame(recent_games)
                st.dataframe(games_df)
                st.subheader("Game Summary")
                st.write(f"Total Games Found: {len(recent_games)}")
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("Games by Team")
                    team_counts = pd.concat([games_df['Home Team'].value_counts(), games_df['Away Team'].value_counts()])
                    st.bar_chart(team_counts)
            else:
                st.info("No recent games found within the selected date range.")
        except Exception as e:
            st.error(f"An error occurred: {e}")
            logger.error(f"Error in finding recent games: {e}")

# New pages for Predictor and Analytics
def win_prediction_page():
    st.subheader("Win Prediction")
    game_id = st.text_input("Enter Game ID for Prediction", value="662025")
    if st.button("Predict Win Probability"):
        display_win_prediction(game_id)

def player_trend_page():
    st.subheader("Player Performance Trend")
    player_id = st.number_input("Enter Player ID", value=660271)
    fig = create_player_performance_trend(player_id)
    st.plotly_chart(fig)

def team_analytics_page():
    st.subheader("Team Comparative Analytics")
    col1, col2 = st.columns(2)
    with col1:
        team1 = st.text_input("Enter Team 1 Name", value="Los Angeles Dodgers")
    with col2:
        team2 = st.text_input("Enter Team 2 Name", value="New York Yankees")
    if st.button("Compare Teams"):
        fig = team_comparative_analytics(team1, team2)
        st.plotly_chart(fig)

# =============================
# Main Application Routing
# =============================
def main():
    set_advanced_theme()
    mlb_data_retriever = MLBDataRetriever(api_key=GEMINI_API_KEY)
    
    # Sidebar navigation including new predictor/analytics pages
    page = st.sidebar.radio(
        "Navigate",
        [
            "Home 🏠",
            "Game Data 🎯",
            "Team Roster 👥",
            "Player Details 👤",
            "Recent Games 📊",
            "Win Prediction 🔮",
            "Player Trend 📈",
            "Team Analytics ⚖️"
        ]
    )
    
    if page == "Home 🏠":
        home_page()
    elif page == "Game Data 🎯":
        game_data_page(mlb_data_retriever)
    elif page == "Team Roster 👥":
        team_roster_page(mlb_data_retriever)
    elif page == "Player Details 👤":
        player_details_page(mlb_data_retriever)
    elif page == "Recent Games 📊":
        recent_games_page(mlb_data_retriever)
    elif page == "Win Prediction 🔮":
        win_prediction_page()
    elif page == "Player Trend 📈":
        player_trend_page()
    elif page == "Team Analytics ⚖️":
        team_analytics_page()

if __name__ == "__main__":
    main()
