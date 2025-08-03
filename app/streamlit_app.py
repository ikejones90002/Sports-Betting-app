import streamlit as st
import numpy as np
import re

# Set page title and layout
st.set_page_config(page_title="🏈 Advanced Sports Predictor", layout="wide")

# Title
st.title("🏈 Advanced Sports Predictor")

# Tabs for team and player predictions
tab1, tab2 = st.tabs(["Team Game Prediction", "Player Prop Bets"])

# Function to parse stats from text_area
def parse_stats(stats_text):
    try:
        points = float(re.search(r"Points: (\d+\.?\d*)", stats_text).group(1)) if re.search(r"Points: (\d+\.?\d*)", stats_text) else 0
        yards = float(re.search(r"Yards: (\d+\.?\d*)", stats_text).group(1)) if re.search(r"Yards: (\d+\.?\d*)", stats_text) else 0
        turnovers = float(re.search(r"Turnovers: (\d+\.?\d*)", stats_text).group(1)) if re.search(r"Turnovers: (\d+\.?\d*)", stats_text) else 0
        return points, yards, turnovers
    except:
        return 0, 0, 0

# Function to parse recent performance (e.g., "3-2" → 0.6 win rate)
def parse_recent_performance(record):
    try:
        wins, losses = map(int, record.split("-"))
        return wins / (wins + losses) if (wins + losses) > 0 else 0.5
    except:
        return 0.5

# Team Prediction Algorithm
def predict_team_outcome(team1_data, team2_data):
    # Parse inputs
    t1_points, t1_yards, t1_turnovers = parse_stats(team1_data["stats"])
    t2_points, t2_yards, t2_turnovers = parse_stats(team2_data["stats"])
    t1_recent = parse_recent_performance(team1_data["recent"])
    t2_recent = parse_recent_performance(team2_data["recent"])
    
    # Calculate team strength (weights: points=0.4, yards=0.3, turnovers=0.2, recent=0.1, home/away=0.05)
    t1_score = (t1_points * 0.4 + t1_yards/10 * 0.3 - t1_turnovers * 10 * 0.2 + t1_recent * 20 * 0.1)
    t2_score = (t2_points * 0.4 + t2_yards/10 * 0.3 - t2_turnovers * 10 * 0.2 + t2_recent * 20 * 0.1)
    
    # Home/away bonus
    if team1_data["home_away"] == "Home":
        t1_score *= 1.05
    elif team1_data["home_away"] == "Away":
        t1_score *= 0.95
    if team2_data["home_away"] == "Home":
        t2_score *= 1.05
    elif team2_data["home_away"] == "Away":
        t2_score *= 0.95
    
    # Injury penalty (simplified: -5% per key player out)
    t1_injury_penalty = 0.95 if "out" in team1_data["injuries"].lower() else 1.0
    t2_injury_penalty = 0.95 if "out" in team2_data["injuries"].lower() else 1.0
    t1_score *= t1_injury_penalty
    t2_score *= t2_injury_penalty
    
    # Rest days adjustment (+2% per extra rest day)
    rest_diff = team1_data["rest_days"] - team2_data["rest_days"]
    t1_score *= (1 + 0.02 * rest_diff)
    t2_score *= (1 - 0.02 * rest_diff)
    
    # Predict winner and score
    score_diff = abs(t1_score - t2_score) / 2
    predicted_score1 = round(t1_points + score_diff if t1_score > t2_score else t1_points - score_diff)
    predicted_score2 = round(t2_points - score_diff if t1_score > t2_score else t2_points + score_diff)
    
    winner = team1_data["name"] if t1_score > t2_score else team2_data["name"]
    factors = [
        f"Points (Team 1: {t1_points}, Team 2: {t2_points})",
        f"Yards (Team 1: {t1_yards}, Team 2: {t2_yards})",
        f"Turnovers (Team 1: {t1_turnovers}, Team 2: {t2_turnovers})",
        f"Recent Performance (Team 1: {t1_recent:.2f}, Team 2: {t2_recent:.2f})",
        f"Injuries: {team1_data['injuries']} vs. {team2_data['injuries']}",
        f"Home/Away: {team1_data['home_away']} vs. {team2_data['home_away']}",
        f"Rest Days: {team1_data['rest_days']} vs. {team2_data['rest_days']}"
    ]
    return f"{winner} wins {predicted_score1}-{predicted_score2}", factors

# Player Prop Prediction Algorithm
def predict_player_prop(player):
    # Parse recent stats (e.g., "Passing Yards: 250\nTDs: 2")
    try:
        stat_value = float(re.search(r"(\d+\.?\d*)", player["recent_stats"]).group(1))
    except:
        stat_value = 0
    
    # Parse opposing defense rank (e.g., "Rank: 15" → 15)
    try:
        def_rank = float(re.search(r"Rank: (\d+)", player["opp_defense"]).group(1))
    except:
        def_rank = 16  # Assume average defense
    
    # Calculate likelihood (weights: stats=0.5, defense=0.3, injury=0.2)
    likelihood = stat_value * 0.5 + (32 - def_rank) * 0.3  # Higher rank (weaker defense) increases likelihood
    if player["injury_status"] == "Out":
        likelihood *= 0.0
    elif player["injury_status"] == "Questionable":
        likelihood *= 0.7
    
    # Compare to prop value
    outcome = "Over" if likelihood > player["prop_value"] else "Under"
    confidence = abs(likelihood - player["prop_value"]) / player["prop_value"] * 100
    
    factors = [
        f"Recent Stats: {player['recent_stats']}",
        f"Opposing Defense: {player['opp_defense']}",
        f"Injury Status: {player['injury_status']}"
    ]
    return f"{player['name']} likely to hit {outcome} {player['prop_value']} ({confidence:.1f}% confidence)", factors

# Team Prediction Tab
with tab1:
    st.header("Team Game Prediction")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Team 1")
        team1_name = st.text_input("Team 1 Name", "Team A", key="team1_name")
        team1_stats = st.text_area("Team 1 Stats (e.g., Points, Yards, Turnovers)", 
                                  "Points: 27\nYards: 360\nTurnovers: 1.2", key="team1_stats")
        team1_recent = st.text_input("Recent Performance (e.g., W-L last 5 games)", "3-2", key="team1_recent")
        team1_injuries = st.text_area("Key Injuries", "None", key="team1_injuries")
        team1_home_away = st.selectbox("Home/Away", ["Home", "Away", "Neutral"], key="team1_home_away")

    with col2:
        st.subheader("Team 2")
        team2_name = st.text_input("Team 2 Name", "Team B", key="team2_name")
        team2_stats = st.text_area("Team 2 Stats (e.g., Points, Yards, Turnovers)", 
                                  "Points: 21\nYards: 310\nTurnovers: 1.8", key="team2_stats")
        team2_recent = st.text_input("Recent Performance (e.g., W-L last 5 games)", "2-3", key="team2_recent")
        team2_injuries = st.text_area("Key Injuries", "None", key="team2_injuries")
        team2_home_away = st.selectbox("Home/Away", ["Home", "Away", "Neutral"], key="team2_home_away")

    with st.expander("Game Context"):
        game_type = st.selectbox("Game Type", ["Regular Season", "Playoffs", "Preseason"], key="game_type")
        weather = st.selectbox("Weather", ["Clear", "Rain", "Snow", "Windy"], key="weather")
        rest_days_team1 = st.slider("Days Since Last Game (Team 1)", 1, 14, 7, key="rest_days_team1")
        rest_days_team2 = st.slider("Days Since Last Game (Team 2)", 1, 14, 7, key="rest_days_team2")

    if st.button("Predict Game Outcome", key="predict_team"):
        team1_data = {
            "name": team1_name, "stats": team1_stats, "recent": team1_recent,
            "injuries": team1_injuries, "home_away": team1_home_away, "rest_days": rest_days_team1
        }
        team2_data = {
            "name": team2_name, "stats": team2_stats, "recent": team2_recent,
            "injuries": team2_injuries, "home_away": team2_home_away, "rest_days": rest_days_team2
        }
        prediction, factors = predict_team_outcome(team1_data, team2_data)
        st.success(f"Prediction: {prediction}")
        st.write("Key Factors:")
        for factor in factors:
            st.write(f"- {factor}")

    if st.button("Clear Inputs", key="clear_team"):
        st.rerun()

# Player Prop Bets Tab
with tab2:
    st.header("Player Prop Bet Prediction")
    
    # Initialize session state for players
    if 'players' not in st.session_state:
        st.session_state.players = [{}]

    def add_player():
        st.session_state.players.append({})

    for i, player in enumerate(st.session_state.players):
        with st.expander(f"Player {i+1}", expanded=True):
            col1, col2 = st.columns(2)
            with col1:
                player_name = st.text_input("Player Name", "Player A", key=f"player_name_{i}")
                position = st.selectbox("Position", ["QB", "RB", "WR", "TE", "K"], key=f"position_{i}")
                recent_stats = st.text_area("Recent Stats (e.g., Passing Yards, TDs)", 
                                           "Passing Yards: 250\nTDs: 2", key=f"recent_stats_{i}")
            with col2:
                prop_type = st.selectbox("Prop Type", ["Over/Under Yards", "Touchdowns", "Receptions"], key=f"prop_type_{i}")
                prop_value = st.number_input("Prop Value (e.g., 250.5 yards)", 0.0, 1000.0, 250.0, key=f"prop_value_{i}")
                opp_defense = st.text_input("Opposing Defense (e.g., Pass Defense Rank)", "Rank: 15", key=f"opp_defense_{i}")
                injury_status = st.selectbox("Injury Status", ["Healthy", "Questionable", "Out"], key=f"injury_status_{i}")

            st.session_state.players[i] = {
                "name": player_name, "position": position, "recent_stats": recent_stats,
                "prop_type": prop_type, "prop_value": prop_value, "opp_defense": opp_defense,
                "injury_status": injury_status
            }

    if st.button("Add Player", key="add_player"):
        add_player()

    if st.button("Predict Player Props", key="predict_props"):
        results = []
        for player in st.session_state.players:
            prediction, factors = predict_player_prop(player)
            results.append(f"{prediction}\nKey Factors:\n" + "\n".join([f"- {f}" for f in factors]))
        st.success("\n\n".join(results))

    if st.button("Clear Players", key="clear_players"):
        st.session_state.players = [{}]
        st.rerun()

# Footer
st.markdown("---")
st.markdown("**Built for Fun** | Created by ijones90002")
st.write("Powered by xAI Grok 3 | Statistical Algorithm")