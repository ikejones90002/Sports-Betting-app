import streamlit as st
import numpy as np
import re
from achievements import update_streak
from avery import avery_commentary
from data import nba_players, nfl_players, mlb_players, nhl_players
from modules import basketball_predictor, football_predictor, baseball_predictor, hockey_predictor

# Set page title and layout
st.set_page_config(page_title="Sports BetTracker", layout="wide")

# Display logo
st.image("sports-bettracker-logo.png", width=200, caption="Sports BetTracker Logo")

# Title
st.title("🏀⚾🏒🏈 Sports BetTracker")

# Sport selection
sport = st.selectbox("Select Sport", ["Football", "Baseball", "Basketball", "Hockey"], key="sport")

# Define sport-specific stats, positions, and players
SPORT_STATS = {
    "Football": {
        "team_stats": ["Points", "Yards", "Turnovers"],
        "player_positions": ["QB", "RB", "WR", "TE", "K"],
        "player_stats": ["Passing Yards", "Rushing Yards", "Receptions", "Touchdowns"],
        "players": nfl_players
    },
    "Baseball": {
        "team_stats": ["Runs", "Hits", "Errors"],
        "player_positions": ["P", "C", "1B", "2B", "3B", "SS", "LF", "CF", "RF"],
        "player_stats": ["Batting Average", "Home Runs", "RBIs", "Pitcher ERA"],
        "players": mlb_players
    },
    "Basketball": {
        "team_stats": ["Points", "Rebounds", "Assists", "Field Goal %"],
        "player_positions": ["PG", "SG", "SF", "PF", "C"],
        "player_stats": ["Points", "Rebounds", "Assists", "Field Goal %"],
        "players": nba_players
    },
    "Hockey": {
        "team_stats": ["Goals", "Shots on Goal", "Save %"],
        "player_positions": ["C", "LW", "RW", "D", "G"],
        "player_stats": ["Goals", "Assists", "Shots on Goal", "Save %"],
        "players": nhl_players
    }
}

# Initialize session state
if "team_form_reset_key" not in st.session_state:
    st.session_state.team_form_reset_key = 0
if "player_form_reset_key" not in st.session_state:
    st.session_state.player_form_reset_key = 0
if "players" not in st.session_state:
    st.session_state.players = [{}]
if "streaks" not in st.session_state:
    st.session_state.streaks = {}

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

# Function to parse recent performance
def parse_recent_performance(record):
    try:
        wins, losses = map(int, record.split("-"))
        return wins / (wins + losses) if (wins + losses) > 0 else 0.5
    except:
        return 0.5

# Team Prediction Algorithm
def predict_team_outcome(team1_data, team2_data, sport):
    t1_stats = parse_stats(team1_data["stats"], sport)
    t2_stats = parse_stats(team2_data["stats"], sport)
    t1_recent = parse_recent_performance(team1_data["recent"])
    t2_recent = parse_recent_performance(team2_data["recent"])
    
    weights = {"main": 0.4, "secondary": 0.3, "tertiary": 0.2, "recent": 0.1, "home_away": 0.05}
    stat_keys = SPORT_STATS[sport]["team_stats"]
    
    t1_score = t1_stats[stat_keys[0]] * weights["main"]
    t2_score = t2_stats[stat_keys[0]] * weights["main"]
    
    if len(stat_keys) > 1:
        t1_score += t1_stats[stat_keys[1]] / (10 if sport == "Football" else 1) * weights["secondary"]
        t2_score += t2_stats[stat_keys[1]] / (10 if sport == "Football" else 1) * weights["secondary"]
    
    if len(stat_keys) > 2:
        t1_score -= t1_stats[stat_keys[2]] * (10 if sport == "Football" else 1) * weights["tertiary"]
        t2_score -= t2_stats[stat_keys[2]] * (10 if sport == "Football" else 1) * weights["tertiary"]
    
    t1_score += t1_recent * 20 * weights["recent"]
    t2_score += t2_recent * 20 * weights["recent"]
    
    if team1_data["home_away"] == "Home":
        t1_score *= 1 + weights["home_away"]
    elif team1_data["home_away"] == "Away":
        t1_score *= 1 - weights["home_away"]
    if team2_data["home_away"] == "Home":
        t2_score *= 1 + weights["home_away"]
    elif team2_data["home_away"] == "Away":
        t2_score *= 1 - weights["home_away"]
    
    t1_injury_penalty = 0.95 if "out" in team1_data["injuries"].lower() else 1.0
    t2_injury_penalty = 0.95 if "out" in team2_data["injuries"].lower() else 1.0
    t1_score *= t1_injury_penalty
    t2_score *= t2_injury_penalty
    
    rest_diff = team1_data["rest_days"] - team2_data["rest_days"]
    t1_score *= (1 + 0.02 * rest_diff)
    t2_score *= (1 - 0.02 * rest_diff)
    
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

# Betting calculation
def calculate_bet_outcome(bet_type, stake, odds):
    if stake <= 0:
        return {"error": "Stake must be greater than 0", "profit": 0, "total_return": 0}
    if bet_type == "Moneyline":
        if odds == 0:
            return {"error": "Odds cannot be zero", "profit": 0, "total_return": 0}
        if odds > 0:
            profit = stake * (odds / 100)
        else:
            profit = stake / (abs(odds) / 100)
    else:  # Point Spread or Over/Under
        if odds == 0:
            return {"error": "Odds cannot be zero", "profit": 0, "total_return": 0}
        profit = stake * (100 / abs(odds))
    profit = round(profit, 2)
    total_return = round(stake + profit, 2)
    return {
        "profit": profit,
        "total_return": total_return,
        "details": f"{bet_type} bet: ${stake} at {odds:+} odds"
    }

# Define tabs
tab1, tab2 = st.tabs(["Team Game Prediction", "Player Prop Bets"])

# Team Prediction Tab
with tab1:
    st.header("Team Game Prediction")
    
    with st.form(key=f"team_form_{st.session_state.team_form_reset_key}"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Team 1")
            team1_name = st.text_input("Team 1 Name", "", key=f"team1_name_{st.session_state.team_form_reset_key}")
            team1_stats = st.text_area(f"Team 1 Stats (e.g., {', '.join(SPORT_STATS[sport]['team_stats'])})", 
                                      "\n".join([f"{stat}: 0" for stat in SPORT_STATS[sport]["team_stats"]]), 
                                      key=f"team1_stats_{st.session_state.team_form_reset_key}")
            team1_recent = st.text_input("Recent Performance (e.g., W-L last 5 games)", "", key=f"team1_recent_{st.session_state.team_form_reset_key}")
            team1_injuries = st.text_area("Key Injuries", "", key=f"team1_injuries_{st.session_state.team_form_reset_key}")
            team1_home_away = st.selectbox("Home/Away", ["", "Home", "Away", "Neutral"], key=f"team1_home_away_{st.session_state.team_form_reset_key}")
        
        with col2:
            st.subheader("Team 2")
            team2_name = st.text_input("Team 2 Name", "", key=f"team2_name_{st.session_state.team_form_reset_key}")
            team2_stats = st.text_area(f"Team 2 Stats (e.g., {', '.join(SPORT_STATS[sport]['team_stats'])})", 
                                      "\n".join([f"{stat}: 0" for stat in SPORT_STATS[sport]["team_stats"]]), 
                                      key=f"team2_stats_{st.session_state.team_form_reset_key}")
            team2_recent = st.text_input("Recent Performance (e.g., W-L last 5 games)", "", key=f"team2_recent_{st.session_state.team_form_reset_key}")
            team2_injuries = st.text_area("Key Injuries", "", key=f"team2_injuries_{st.session_state.team_form_reset_key}")
            team2_home_away = st.selectbox("Home/Away", ["", "Home", "Away", "Neutral"], key=f"team2_home_away_{st.session_state.team_form_reset_key}")
        
        with st.expander("Game Context"):
            game_type = st.selectbox("Game Type", ["", "Regular Season", "Playoffs", "Preseason"], key=f"game_type_{st.session_state.team_form_reset_key}")
            weather = st.selectbox("Weather", ["", "Clear", "Rain", "Snow", "Windy"], key=f"weather_{st.session_state.team_form_reset_key}")
            rest_days_team1 = st.slider("Days Since Last Game (Team 1)", 0, 14, 0, key=f"rest_days_team1_{st.session_state.team_form_reset_key}")
            rest_days_team2 = st.slider("Days Since Last Game (Team 2)", 0, 14, 0, key=f"rest_days_team2_{st.session_state.team_form_reset_key}")
        
        predict_team = st.form_submit_button("Predict Game Outcome")
        clear_team = st.form_submit_button("Clear Inputs")
    
    if predict_team:
        if not team1_name or not team2_name:
            st.error("Please enter names for both teams.")
        elif team1_home_away == "" or team2_home_away == "":
            st.error("Please select Home/Away status for both teams.")
        else:
            team1_data = {
                "name": team1_name, "stats": team1_stats, "recent": team1_recent or "0-0",
                "injuries": team1_injuries or "None", "home_away": team1_home_away or "Neutral", "rest_days": rest_days_team1
            }
            team2_data = {
                "name": team2_name, "stats": team2_stats, "recent": team2_recent or "0-0",
                "injuries": team2_injuries or "None", "home_away": team2_home_away or "Neutral", "rest_days": rest_days_team2
            }
            prediction, factors = predict_team_outcome(team1_data, team2_data, sport)
            st.success(f"Prediction: {prediction}")
            st.write("Key Factors:")
            for factor in factors:
                st.write(f"- {factor}")
    
    if clear_team:
        st.session_state.team_form_reset_key += 1
        st.rerun()

# Player Prop Bets Tab
with tab2:
    st.header("Player Prop Bet Prediction")
    
    def add_player():
        st.session_state.players.append({})
    
    with st.form(key=f"player_form_{st.session_state.player_form_reset_key}"):
        for i in range(len(st.session_state.players)):
            with st.expander(f"Player {i+1}", expanded=True):
                col1, col2 = st.columns(2)
                with col1:
                    player_name = st.selectbox("Player Name", [""] + SPORT_STATS[sport]["players"], key=f"player_name_{i}_{st.session_state.player_form_reset_key}")
                    position = st.selectbox("Position", [""] + SPORT_STATS[sport]["player_positions"], key=f"position_{i}_{st.session_state.player_form_reset_key}")
                    recent_stats = st.text_area(f"Recent Stats (e.g., {', '.join(SPORT_STATS[sport]['player_stats'])})", 
                                               "", key=f"recent_stats_{i}_{st.session_state.player_form_reset_key}")
                with col2:
                    prop_type = st.selectbox("Prop Type", [""] + SPORT_STATS[sport]["player_stats"], key=f"prop_type_{i}_{st.session_state.player_form_reset_key}")
                    bet_type = st.selectbox("Bet Type", ["", "Moneyline", "Point Spread", "Over/Under"], key=f"bet_type_{i}_{st.session_state.player_form_reset_key}")
                    odds = st.number_input("Odds (e.g., +150 or -110)", min_value=-10000, max_value=10000, value=0, key=f"odds_{i}_{st.session_state.player_form_reset_key}")
                    stake = st.number_input("Stake ($)", min_value=0.0, value=0.0, step=0.01, key=f"stake_{i}_{st.session_state.player_form_reset_key}")
                    opp_defense = st.text_input("Opposing Defense (e.g., Rank)", "Rank: 0", key=f"opp_defense_{i}_{st.session_state.player_form_reset_key}")
                    injury_status = st.selectbox("Injury Status", ["", "Healthy", "Questionable", "Out"], key=f"injury_status_{i}_{st.session_state.player_form_reset_key}")
                    
                    # Sport-specific predictor
                    if sport == "Basketball":
                        projected_key = f"projected_points_{i}_{st.session_state.player_form_reset_key}"
                        line_key = f"set_line_{i}_{st.session_state.player_form_reset_key}"
                        projected = st.slider("Projected Points", 0, 50, 0, key=projected_key)
                        line = st.slider("Set Line", 0, 50, 0, key=line_key)
                    elif sport == "Football":
                        projected_key = f"projected_yards_{i}_{st.session_state.player_form_reset_key}"
                        line_key = f"set_line_{i}_{st.session_state.player_form_reset_key}"
                        projected = st.slider("Projected Yards", 0, 400, 0, key=projected_key)
                        line = st.slider("Set Line", 0, 400, 0, key=line_key)
                    elif sport == "Baseball":
                        projected_key = f"projected_hr_{i}_{st.session_state.player_form_reset_key}"
                        line_key = f"set_line_{i}_{st.session_state.player_form_reset_key}"
                        projected = st.slider("Projected Home Runs", 0, 5, 0, key=projected_key)
                        line = st.slider("Set Line", 0, 5, 0, key=line_key)
                    else:  # Hockey
                        projected_key = f"projected_goals_{i}_{st.session_state.player_form_reset_key}"
                        line_key = f"set_line_{i}_{st.session_state.player_form_reset_key}"
                        projected = st.slider("Projected Goals", 0, 5, 0, key=projected_key)
                        line = st.slider("Set Line", 0, 5, 0, key=line_key)
                
                st.session_state.players[i] = {
                    "name": player_name, "position": position, "recent_stats": recent_stats,
                    "prop_type": prop_type, "bet_type": bet_type, "odds": odds, "stake": stake,
                    "opp_defense": opp_defense, "injury_status": injury_status or "Healthy",
                    "projected": projected, "line": line
                }
        
        add_player_btn = st.form_submit_button("Add Player")
        predict_props = st.form_submit_button("Predict Player Props")
        clear_players = st.form_submit_button("Clear Players")
    
    if add_player_btn:
        add_player()
        st.rerun()
    
    if predict_props:
        results = []
        for player in st.session_state.players:
            if not player["name"] or player["prop_type"] == "" or player["position"] == "" or player["bet_type"] == "":
                st.error(f"Please fill in all fields for Player {st.session_state.players.index(player) + 1} (Name, Position, Prop Type, Bet Type).")
                break
            # Call sport-specific predictor
            if sport == "Basketball":
                outcome = basketball_predictor(player)
            elif sport == "Football":
                outcome = football_predictor(player)
            elif sport == "Baseball":
                outcome = baseball_predictor(player)
            else:  # Hockey
                outcome = hockey_predictor(player)
            
            # Update streak
            update_streak(sport, outcome)
            
            # Get Avery commentary
            commentary = avery_commentary(sport, outcome)
            
            # Calculate bet outcome
            bet_result = calculate_bet_outcome(player["bet_type"], player["stake"], player["odds"])
            if "error" in bet_result:
                bet_info = bet_result["error"]
            else:
                bet_info = f"{bet_result['details']}, Profit: ${bet_result['profit']}, Total Return: ${bet_result['total_return']}"
            
            factors = [
                f"Recent Stats: {player['recent_stats'] or 'None'}",
                f"Opposing Defense: {player['opp_defense']}",
                f"Injury Status: {player['injury_status']}",
                f"Bet: {bet_info}",
                f"Projected {player['prop_type']}: {player['projected']}",
                f"Set Line: {player['line']}"
            ]
            results.append(f"{player['name']} likely to hit {outcome} {player['line']} ({commentary})\nKey Factors:\n" + "\n".join([f"- {f}" for f in factors]))
        
        if results:
            st.success("\n\n".join(results))
    
    if clear_players:
        st.session_state.players = [{}]
        st.session_state.player_form_reset_key += 1
        st.rerun()

# Footer
st.markdown("---")
st.write("Developed with ❤️ by the Sports BetTracker Team")
st.write("For educational purposes only. Please gamble responsibly.")
st.write("Powered by xAI Grok 3 | Statistical Algorithm")