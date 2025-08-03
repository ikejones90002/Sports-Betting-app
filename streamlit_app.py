import streamlit as st
import numpy as np
import re

# Set page title and layout
st.set_page_config(page_title="🏀⚾🏒🏈 Multi-Sport Predictor", layout="wide")

# Title
st.title("🏀⚾🏒🏈 Multi-Sport Predictor")

# Sport selection
sport = st.selectbox("Select Sport", ["Football", "Baseball", "Basketball", "Hockey"], key="sport")

# Define sport-specific stats
SPORT_STATS = {
    "Football": {"team_stats": ["Points", "Yards", "Turnovers"], "player_stats": ["Passing Yards", "Rushing Yards", "Receptions", "Touchdowns"]},
    "Baseball": {"team_stats": ["Runs", "Hits", "Errors"], "player_stats": ["Batting Average", "Home Runs", "RBIs", "Pitcher ERA"]},
    "Basketball": {"team_stats": ["Points", "Rebounds", "Assists", "Field Goal %"], "player_stats": ["Points", "Rebounds", "Assists", "Field Goal %"]},
    "Hockey": {"team_stats": ["Goals", "Shots on Goal", "Save %"], "player_stats": ["Goals", "Assists", "Shots on Goal", "Save %"]}
}

# Tabs for team and player predictions
tab1, tab2 = st.tabs(["Team Game Prediction", "Player Prop Bets"])

# Function to parse stats from text_area
def parse_stats(stats_text, sport):
    stats = {}
    try:
        for stat in SPORT_STATS[sport]["team_stats"]:
            match = re.search(rf"{stat}: (\d+\.?\d*)", stats_text)
            stats[stat] = float(match.group(1)) if match else 0
        return stats
    except:
        return {stat: 0 for stat in SPORT_STATS[sport]["team_stats"]}

# Function to parse recent performance (e.g., "3-2" → 0.6 win rate)
def parse_recent_performance(record):
    try:
        wins, losses = map(int, record.split("-"))
        return wins / (wins + losses) if (wins + losses) > 0 else 0.5
    except:
        return 0.5

# Function to parse player stats
def parse_player_stats(stats_text, sport):
    try:
        # Extract the first numerical value from player stats
        match = re.search(r"(\d+\.?\d*)", stats_text)
        return float(match.group(1)) if match else 0
    except:
        return 0

# Team Prediction Algorithm
def predict_team_outcome(team1_data, team2_data, sport):
    # Parse stats
    t1_stats = parse_stats(team1_data["stats"], sport)
    t2_stats = parse_stats(team2_data["stats"], sport)
    t1_recent = parse_recent_performance(team1_data["recent"])
    t2_recent = parse_recent_performance(team2_data["recent"])
    
    # Calculate team strength (weights: main stat=0.4, secondary stat=0.3, tertiary stat=0.2, recent=0.1, home/away=0.05)
    weights = {"main": 0.4, "secondary": 0.3, "tertiary": 0.2, "recent": 0.1, "home_away": 0.05}
    stat_keys = SPORT_STATS[sport]["team_stats"]
    
    t1_score = t1_stats[stat_keys[0]] * weights["main"]  # Main stat (e.g., Points, Runs, Goals)
    t2_score = t2_stats[stat_keys[0]] * weights["main"]
    
    if len(stat_keys) > 1:
        t1_score += t1_stats[stat_keys[1]] / (10 if sport == "Football" else 1) * weights["secondary"]
        t2_score += t2_stats[stat_keys[1]] / (10 if sport == "Football" else 1) * weights["secondary"]
    
    if len(stat_keys) > 2:
        t1_score -= t1_stats[stat_keys[2]] * (10 if sport == "Football" else 1) * weights["tertiary"]  # Negative for turnovers/errors
        t2_score -= t2_stats[stat_keys[2]] * (10 if sport == "Football" else 1) * weights["tertiary"]
    
    t1_score += t1_recent * 20 * weights["recent"]
    t2_score += t2_recent * 20 * weights["recent"]
    
    # Home/away bonus
    if team1_data["home_away"] == "Home":
        t1_score *= 1 + weights["home_away"]
    elif team1_data["home_away"] == "Away":
        t1_score *= 1 - weights["home_away"]
    if team2_data["home_away"] == "Home":
        t2_score *= 1 + weights["home_away"]
    elif team2_data["home_away"] == "Away":
        t2_score *= 1 - weights["home_away"]
    
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
    main_stat = stat_keys[0]
    predicted_score1 = round(t1_stats[main_stat] + score_diff if t1_score > t2_score else t1_stats[main_stat] - score_diff)
    predicted_score2 = round(t2_stats[main_stat] - score_diff if t1_score > t2_score else t2_stats[main_stat] + score_diff)
    
    winner = team1_data["name"] if t1_score > t2_score else team2_data["name"]
    factors = [
        f"{main_stat} (Team 1: {t1_stats[main_stat]}, Team 2: {t2_stats[main_stat]})",
        f"Recent Performance (Team 1: {t1_recent:.2f}, Team 2: {t2_recent:.2f})",
        f"Injuries: {team1_data['injuries']} vs. {team2_data['injuries']}",
        f"Home/Away: {team1_data['home_away']} vs. {team2_data['home_away']}",
        f"Rest Days: {team1_data['rest_days']} vs. {team2_data['rest_days']}"
    ]
    for stat in stat_keys[1:]:
        factors.append(f"{stat} (Team 1: {t1_stats[stat]}, Team 2: {t2_stats[stat]})")
    
    return f"{winner} wins {predicted_score1}-{predicted_score2}", factors

# Player Prop Prediction Algorithm
def predict_player_prop(player, sport):
    stat_value = parse_player_stats(player["recent_stats"], sport)
    try:
        def_rank = float(re.search(r"Rank: (\d+)", player["opp_defense"]).group(1))
    except:
        def_rank = 16  # Assume average defense
    
    # Calculate likelihood (weights: stats=0.5, defense=0.3, injury=0.2)
    likelihood = stat_value * 0.5 + (32 - def_rank) * 0.3
    if player["injury_status"] == "Out":
        likelihood *= 0.0
    elif player["injury_status"] == "Questionable":
        likelihood *= 0.7
    
    outcome = "Over" if likelihood > player["prop_value"] else "Under"
    confidence = abs(likelihood - player["prop_value"]) / player["prop_value"] * 100
    
    factors = [
        f"Recent Stats: {player['recent_stats']}",
        f"Opposing Defense: {player['opp_defense']}",
        f"Injury Status: {player['injury_status']}"
    ]
    return f"{player['name']} likely to hit {outcome} {player['prop_value']} ({confidence:.1f}% confidence)", factors

# Reset function for team inputs
def reset_team_inputs():
    st.session_state["team1_name"] = "Team A"
    st.session_state["team2_name"] = "Team B"
    st.session_state["team1_stats"] = "\n".join([f"{stat}: 0" for stat in SPORT_STATS[sport]["team_stats"]])
    st.session_state["team2_stats"] = "\n".join([f"{stat}: 0" for stat in SPORT_STATS[sport]["team_stats"]])
    st.session_state["team1_recent"] = "0-0"
    st.session_state["team2_recent"] = "0-0"
    st.session_state["team1_injuries"] = "None"
    st.session_state["team2_injuries"] = "None"
    st.session_state["team1_home_away"] = "Home"
    st.session_state["team2_home_away"] = "Away"
    st.session_state["game_type"] = "Regular Season"
    st.session_state["weather"] = "Clear"
    st.session_state["rest_days_team1"] = 7
    st.session_state["rest_days_team2"] = 7

# Team Prediction Tab
with tab1:
    st.header("Team Game Prediction")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Team 1")
        team1_name = st.text_input("Team 1 Name", "", key="team1_name")
        team1_stats = st.text_area(f"Team 1 Stats (e.g., {', '.join(SPORT_STATS[sport]['team_stats'])})", 
                                   "\n".join([f"{stat}: 0" for stat in SPORT_STATS[sport]["team_stats"]]), key="team1_stats")
        team1_recent = st.text_input("Recent Performance (e.g., W-L last 5 games)", "", key="team1_recent")
        team1_injuries = st.text_area("Key Injuries", "None", key="team1_injuries")
        team1_home_away = st.selectbox("Home/Away", ["Home", "Away", "Neutral"], key="team1_home_away")

    with col2:
        st.subheader("Team 2")
        team2_name = st.text_input("Team 2 Name", "", key="team2_name")
        team2_stats = st.text_area(f"Team 2 Stats (e.g., {', '.join(SPORT_STATS[sport]['team_stats'])})", 
                                   "\n".join([f"{stat}: 0" for stat in SPORT_STATS[sport]["team_stats"]]), key="team2_stats")
        team2_recent = st.text_input("Recent Performance (e.g., W-L last 5 games)", "", key="team2_recent")
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
        prediction, factors = predict_team_outcome(team1_data, team2_data, sport)
        st.success(f"Prediction: {prediction}")
        st.write("Key Factors:")
        for factor in factors:
            st.write(f"- {factor}")

    if st.button("Clear Inputs", key="clear_team"):
        reset_team_inputs()

# Player Prop Bets Tab
with tab2:
    st.header("Player Prop Bet Prediction")
    
    # Initialize session state for players
    if 'players' not in st.session_state:
        st.session_state.players = [{}]

    def add_player():
        st.session_state.players.append({})
    
    def reset_players():
        st.session_state.players = [{}]
        for i in range(len(st.session_state.players)):
            st.session_state[f"player_name_{i}"] = "Player A"
            st.session_state[f"position_{i}"] = SPORT_STATS[sport]["player_stats"][0]
            st.session_state[f"recent_stats_{i}"] = f"{SPORT_STATS[sport]['player_stats'][0]}: 0"
            st.session_state[f"prop_type_{i}"] = SPORT_STATS[sport]["player_stats"][0]
            st.session_state[f"prop_value_{i}"] = 0.0
            st.session_state[f"opp_defense_{i}"] = "Rank: 16"
            st.session_state[f"injury_status_{i}"] = "Healthy"

    for i, player in enumerate(st.session_state.players):
        with st.expander(f"Player {i+1}", expanded=True):
            col1, col2 = st.columns(2)
            with col1:
                player_name = st.text_input("Player Name", "", key=f"player_name_{i}")
                position = st.selectbox("Position", SPORT_STATS[sport]["player_stats"], key=f"position_{i}")
                recent_stats = st.text_area(f"Recent Stats (e.g., {', '.join(SPORT_STATS[sport]['player_stats'])})", 
                                           f"{SPORT_STATS[sport]['player_stats'][0]}: 0", key=f"recent_stats_{i}")
            with col2:
                prop_type = st.selectbox("Prop Type", SPORT_STATS[sport]["player_stats"], key=f"prop_type_{i}")
                prop_value = st.number_input("Prop Value (e.g., 0.5)", 0.0, 1000.0, 0.0, key=f"prop_value_{i}")
                opp_defense = st.text_input("Opposing Defense (e.g., Rank)", "", key=f"opp_defense_{i}")
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
            prediction, factors = predict_player_prop(player, sport)
            results.append(f"{prediction}\nKey Factors:\n" + "\n".join([f"- {f}" for f in factors]))
        st.success("\n\n".join(results))

    if st.button("Clear Players", key="clear_players"):
        st.session_state.players = [{}]
        st.rerun()

# Footer
st.markdown("---")
st.write("Made with ❤️ by ijones90002")
st.write("This app is for educational purposes only. Please gamble responsibly.")
st.write("Powered by xAI Grok 3 | Statistical Algorithm")