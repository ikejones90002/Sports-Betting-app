import streamlit as st
import re
from achievements import update_streak
from avery import avery_commentary
from modules import basketball_predictor, football_predictor, baseball_predictor, hockey_predictor
from typing import Dict, List, Tuple, Union

# Set page title and layout
st.set_page_config(page_title="Sports BetTracker", layout="wide")

# Display logo with error handling
try:
    st.image("sports-bettracker-logo.png", width=200, caption="Sports BetTracker Logo")
except FileNotFoundError:
    st.warning("Logo file 'sports-bettracker-logo.png' not found. Please add it to the project directory.")

# Title
st.title("🏀⚾🏒🏈 Sports BetTracker")

# Sport selection
sport = st.selectbox("Select Sport", ["Basketball", "Football", "Baseball", "Hockey"], key="sport")

# Define sport-specific stats and positions (no players from data.py)
SPORT_STATS = {
    "Basketball": {
        "team_stats": ["Points", "Rebounds", "Assists", "Field Goal %"],
        "player_positions": ["PG", "SG", "SF", "PF", "C"],
        "player_stats": ["Points", "Rebounds", "Assists", "Field Goal %"],
        "favorite_players": ["LeBron James", "Stephen Curry", "Kevin Durant"]
    },
    "Football": {
        "team_stats": ["Points", "Yards", "Turnovers"],
        "player_positions": ["QB", "RB", "WR", "TE", "K"],
        "player_stats": ["Passing Yards", "Rushing Yards", "Receptions", "Touchdowns"],
        "favorite_players": ["Patrick Mahomes", "Christian McCaffrey", "Tyreek Hill"]
    },
    "Baseball": {
        "team_stats": ["Runs", "Hits", "Errors"],
        "player_positions": ["P", "C", "1B", "2B", "3B", "SS", "LF", "CF", "RF"],
        "player_stats": ["Batting Average", "Home Runs", "RBIs", "Pitcher ERA"],
        "favorite_players": ["Shohei Ohtani", "Aaron Judge", "Mookie Betts"]
    },
    "Hockey": {
        "team_stats": ["Goals", "Shots on Goal", "Save %"],
        "player_positions": ["C", "LW", "RW", "D", "G"],
        "player_stats": ["Goals", "Assists", "Shots on Goal", "Save %"],
        "favorite_players": ["Connor McDavid", "Auston Matthews", "Nathan MacKinnon"]
    }
}

# Initialize session state
if "team_form_reset_key" not in st.session_state:
    st.session_state.team_form_reset_key = 0
if "player_form_reset_key" not in st.session_state:
    st.session_state.player_form_reset_key = 0
if "players" not in st.session_state:
    st.session_state.players = []
if "streaks" not in st.session_state:
    st.session_state.streaks = {}

# Function to parse stats from text_area
def parse_stats(stats_text: str, sport: str) -> Dict[str, float]:
    """
    Parse team stats from text input.

    Args:
        stats_text (str): Text input containing stats (e.g., "Points: 100").
        sport (str): The selected sport.

    Returns:
        Dict[str, float]: Parsed stats dictionary.
    """
    stats = {}
    try:
        for stat in SPORT_STATS[sport]["team_stats"]:
            match = re.search(rf"{stat}:\s*(\d+\.?\d*)", stats_text, re.IGNORECASE)
            stats[stat] = float(match.group(1)) if match else 0.0
        return stats
    except Exception as e:
        st.warning(f"Invalid stats format: {e}. Using default values (0).")
        return {stat: 0.0 for stat in SPORT_STATS[sport]["team_stats"]}

# Function to parse recent performance
def parse_recent_performance(record: str) -> float:
    """
    Parse win-loss record to a win percentage.

    Args:
        record (str): Win-loss record (e.g., "3-2").

    Returns:
        float: Win percentage (0.0 to 1.0).
    """
    try:
        wins, losses = map(int, record.split("-"))
        return wins / (wins + losses) if (wins + losses) > 0 else 0.5
    except Exception as e:
        st.warning(f"Invalid record format: {e}. Defaulting to 0.5.")
        return 0.5

# Team Prediction Algorithm
def predict_team_outcome(team1_data: Dict, team2_data: Dict, sport: str) -> Tuple[str, List[str]]:
    """
    Predict team game outcome based on stats, recent performance, and other factors.

    Args:
        team1_data (Dict): Data for Team 1 (name, stats, recent, injuries, home_away, rest_days).
        team2_data (Dict): Data for Team 2.
        sport (str): The selected sport.

    Returns:
        Tuple[str, List[str]]: Prediction string and list of key factors.
    """
    try:
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
    except Exception as e:
        raise ValueError(f"Error in team prediction: {e}")

# Betting calculation
def calculate_bet_outcome(bet_type: str, stake: float, odds: int) -> Dict[str, Union[str, float]]:
    """
    Calculate betting outcome based on bet type, stake, and odds.

    Args:
        bet_type (str): Type of bet ("Moneyline", "Point Spread", "Over/Under").
        stake (float): Betting stake in dollars.
        odds (int): American odds (e.g., +150, -110).

    Returns:
        Dict[str, Union[str, float]]: Bet result with profit, total return, or error.
    """
    try:
        if stake <= 0:
            return {"error": "Stake must be greater than 0", "profit": 0.0, "total_return": 0.0}
        if odds == 0:
            return {"error": "Odds cannot be zero", "profit": 0.0, "total_return": 0.0}
        if bet_type == "Moneyline":
            if odds > 0:
                profit = stake * (odds / 100)
            else:
                profit = stake / (abs(odds) / 100)
        else:  # Point Spread or Over/Under
            profit = stake * (100 / abs(odds))
        profit = round(profit, 2)
        total_return = round(stake + profit, 2)
        return {
            "profit": profit,
            "total_return": total_return,
            "details": f"{bet_type} bet: ${stake} at {odds:+} odds"
        }
    except Exception as e:
        return {"error": f"Invalid bet calculation: {e}", "profit": 0.0, "total_return": 0.0}

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
            team1_stats = st.text_area(
                f"Team 1 Stats (e.g., {', '.join(SPORT_STATS[sport]['team_stats'])})", 
                "\n".join([f"{stat}: 0" for stat in SPORT_STATS[sport]["team_stats"]]), 
                key=f"team1_stats_{st.session_state.team_form_reset_key}"
            )
            team1_recent = st.text_input("Recent Performance (e.g., W-L last 5 games)", "0-0", key=f"team1_recent_{st.session_state.team_form_reset_key}")
            team1_injuries = st.text_area("Key Injuries", "None", key=f"team1_injuries_{st.session_state.team_form_reset_key}")
            team1_home_away = st.selectbox("Home/Away", ["Home", "Away", "Neutral"], key=f"team1_home_away_{st.session_state.team_form_reset_key}")
        
        with col2:
            st.subheader("Team 2")
            team2_name = st.text_input("Team 2 Name", "", key=f"team2_name_{st.session_state.team_form_reset_key}")
            team2_stats = st.text_area(
                f"Team 2 Stats (e.g., {', '.join(SPORT_STATS[sport]['team_stats'])})", 
                "\n".join([f"{stat}: 0" for stat in SPORT_STATS[sport]["team_stats"]]), 
                key=f"team2_stats_{st.session_state.team_form_reset_key}"
            )
            team2_recent = st.text_input("Recent Performance (e.g., W-L last 5 games)", "0-0", key=f"team2_recent_{st.session_state.team_form_reset_key}")
            team2_injuries = st.text_area("Key Injuries", "None", key=f"team2_injuries_{st.session_state.team_form_reset_key}")
            team2_home_away = st.selectbox("Home/Away", ["Home", "Away", "Neutral"], key=f"team2_home_away_{st.session_state.team_form_reset_key}")
        
        with st.expander("Game Context", expanded=False):
            game_type = st.selectbox("Game Type", ["Regular Season", "Playoffs", "Preseason"], key=f"game_type_{st.session_state.team_form_reset_key}")
            weather = st.selectbox("Weather", ["Clear", "Rain", "Snow", "Windy"], key=f"weather_{st.session_state.team_form_reset_key}")
            rest_days_team1 = st.slider("Days Since Last Game (Team 1)", 0, 14, 0, key=f"rest_days_team1_{st.session_state.team_form_reset_key}")
            rest_days_team2 = st.slider("Days Since Last Game (Team 2)", 0, 14, 0, key=f"rest_days_team2_{st.session_state.team_form_reset_key}")
        
        col1, col2 = st.columns([1, 1])
        with col1:
            predict_team = st.form_submit_button("Predict Game Outcome")
        with col2:
            clear_team = st.form_submit_button("Clear Inputs")
    
    if predict_team:
        if not team1_name or not team2_name:
            st.error("Please enter names for both teams.")
        elif team1_home_away == team2_home_away and team1_home_away in ["Home", "Away"]:
            st.error("Teams cannot both be Home or both be Away.")
        else:
            team1_data = {
                "name": team1_name, "stats": team1_stats, "recent": team1_recent,
                "injuries": team1_injuries, "home_away": team1_home_away, "rest_days": rest_days_team1
            }
            team2_data = {
                "name": team2_name, "stats": team2_stats, "recent": team2_recent,
                "injuries": team2_injuries, "home_away": team2_home_away, "rest_days": rest_days_team2
            }
            try:
                prediction, factors = predict_team_outcome(team1_data, team2_data, sport)
                st.success(f"Prediction: {prediction}")
                st.write("Key Factors:")
                for factor in factors:
                    st.write(f"- {factor}")
            except ValueError as e:
                st.error(str(e))
    
    if clear_team:
        st.session_state.team_form_reset_key += 1
        st.rerun()

# Player Prop Bets Tab
with tab2:
    st.header("Player Prop Bet Prediction")
    
    def add_player():
        if len(st.session_state.players) < 5:
            st.session_state.players.append({})
        else:
            st.warning("Maximum 5 players allowed.")
    
    with st.form(key=f"player_form_{st.session_state.player_form_reset_key}"):
        for i in range(len(st.session_state.players) or 1):  # Ensure at least one player
            with st.expander(f"Player {i+1}", expanded=True):
                col1, col2 = st.columns(2)
                with col1:
                    # Favorite players dropdown
                    favorite_player = st.selectbox(
                        "Select Favorite Player (Optional)",
                        [""] + SPORT_STATS[sport]["favorite_players"],
                        key=f"favorite_player_{i}_{st.session_state.player_form_reset_key}"
                    )
                    # Manual player name input, pre-filled if favorite selected
                    player_name = st.text_input(
                        "Player Name",
                        value=favorite_player if favorite_player else "",
                        key=f"player_name_{i}_{st.session_state.player_form_reset_key}"
                    )
                    position = st.selectbox(
                        "Position",
                        SPORT_STATS[sport]["player_positions"],
                        key=f"position_{i}_{st.session_state.player_form_reset_key}"
                    )
                    recent_stats = st.text_area(
                        f"Recent Stats (e.g., {', '.join(SPORT_STATS[sport]['player_stats'])})", 
                        "", key=f"recent_stats_{i}_{st.session_state.player_form_reset_key}"
                    )
                with col2:
                    prop_type = st.selectbox(
                        "Prop Type",
                        SPORT_STATS[sport]["player_stats"],
                        key=f"prop_type_{i}_{st.session_state.player_form_reset_key}"
                    )
                    bet_type = st.selectbox(
                        "Bet Type",
                        ["Moneyline", "Point Spread", "Over/Under"],
                        key=f"bet_type_{i}_{st.session_state.player_form_reset_key}"
                    )
                    odds = st.number_input(
                        "Odds (e.g., +150 or -110)",
                        min_value=-10000, max_value=10000, value=0,
                        key=f"odds_{i}_{st.session_state.player_form_reset_key}"
                    )
                    stake = st.number_input(
                        "Stake ($)",
                        min_value=0.0, value=0.0, step=0.01,
                        key=f"stake_{i}_{st.session_state.player_form_reset_key}"
                    )
                    opp_defense = st.text_input(
                        "Opposing Defense (e.g., Rank)",
                        "Rank: 0",
                        key=f"opp_defense_{i}_{st.session_state.player_form_reset_key}"
                    )
                    injury_status = st.selectbox(
                        "Injury Status",
                        ["Healthy", "Questionable", "Out"],
                        key=f"injury_status_{i}_{st.session_state.player_form_reset_key}"
                    )
                    
                    # Sport-specific sliders with decimal support
                    if sport == "Basketball":
                        projected = st.slider(
                            "Projected Points", 0.0, 50.0, 0.0, step=0.5,
                            key=f"projected_points_{i}_{st.session_state.player_form_reset_key}"
                        )
                        line = st.slider(
                            "Set Line", 0.0, 50.0, 0.0, step=0.5,
                            key=f"set_line_{i}_{st.session_state.player_form_reset_key}"
                        )
                    elif sport == "Football":
                        projected = st.slider(
                            "Projected Yards", 0.0, 300.0, 0.0, step=0.5,
                            key=f"projected_yards_{i}_{st.session_state.player_form_reset_key}"
                        )
                        line = st.slider(
                            "Set Line", 0.0, 300.0, 0.0, step=0.5,
                            key=f"set_line_{i}_{st.session_state.player_form_reset_key}"
                        )
                    elif sport == "Baseball":
                        projected = st.slider(
                            "Projected Home Runs", 0.0, 2.0, 0.0, step=0.1,
                            key=f"projected_hr_{i}_{st.session_state.player_form_reset_key}"
                        )
                        line = st.slider(
                            "Set Line", 0.0, 2.0, 0.0, step=0.1,
                            key=f"set_line_{i}_{st.session_state.player_form_reset_key}"
                        )
                    else:  # Hockey
                        projected = st.slider(
                            "Projected Goals", 0.0, 2.0, 0.0, step=0.1,
                            key=f"projected_goals_{i}_{st.session_state.player_form_reset_key}"
                        )
                        line = st.slider(
                            "Set Line", 0.0, 2.0, 0.0, step=0.1,
                            key=f"set_line_{i}_{st.session_state.player_form_reset_key}"
                        )
                
                st.session_state.players[i] = {
                    "name": player_name, "position": position, "recent_stats": recent_stats,
                    "prop_type": prop_type, "bet_type": bet_type, "odds": odds, "stake": stake,
                    "opp_defense": opp_defense, "injury_status": injury_status,
                    "projected": projected, "line": line
                }
        
        col1, col2, col3 = st.columns([1, 1, 1])
        with col1:
            add_player_btn = st.form_submit_button("Add Player")
        with col2:
            predict_props = st.form_submit_button("Predict Player Props")
        with col3:
            clear_players = st.form_submit_button("Clear Players")
    
    if add_player_btn:
        add_player()
        st.rerun()
    
    if predict_props:
        if not st.session_state.players:
            st.error("Please add at least one player.")
        else:
            results = []
            for i, player in enumerate(st.session_state.players):
                if not all([player.get("name"), player.get("prop_type"), player.get("position"), player.get("bet_type")]):
                    st.error(f"Please fill in all fields for Player {i + 1} (Name, Position, Prop Type, Bet Type).")
                    break
                try:
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
                    streaks = update_streak(sport, outcome, reset_on_push=True)
                    
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
                except Exception as e:
                    st.error(f"Error predicting for {player.get('name', 'Player ' + str(i + 1))}: {e}")
                    break
            
            if results:
                st.success("\n\n".join(results))
                st.write(f"Current {sport} Streak: {st.session_state.streaks.get(sport, 0)} OVERs")
    
    if clear_players:
        st.session_state.players = []
        st.session_state.player_form_reset_key += 1
        st.rerun()

# Footer
st.markdown("---")
st.write("Developed with ❤️ by the Sports BetTracker Team | &copy; 103 Software Solutions, LLC")
st.write("For educational purposes only. Please gamble responsibly.")
st.write("Built with xAI technology | Statistical Algorithm")