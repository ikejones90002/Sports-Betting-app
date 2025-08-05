import streamlit as st # type: ignore
import numpy as np # type: ignore
import re

# Set page title and layout
st.set_page_config(page_title="Sports BetTracker", layout="wide")

# Display logo
st.image("sports-bettracker-logo.png", width=200, caption="Sports BetTracker - Track the Action. Bet Smarter.")

# Title
st.title("🏀⚾🏒🏈 Sports BetTracker")

# How to Use Guide
with st.expander("How to Use Sports BetTracker"):

 # Display logo
    st.image("sports-bettracker-logo.png", width=200, caption="Sports BetTracker - Track the Action. Bet Smarter.")

st.markdown("""
    **Quick Guide to Using Sports BetTracker**

    - **Team Game Prediction**:
      - Enter team names, stats (e.g., Points: 24), recent performance (e.g., 3-2 for 3 wins, 2 losses), injuries, and home/away status.
      - Optionally add betting details (Bet Type, Odds, Stake) for one or both teams. Stakes can differ or be left as 0 if no bet is placed.
      - Click "Predict Game Outcome" to see the predicted winner, score, and win probability with a 95% confidence interval (CI).

    - **Player Prop Bets**:
      - Add players and enter their name, position, recent stats, prop type, over/under value (e.g., 25.5), odds, opposing defense rank, and injury status.
      - Over/Under must be a number. Click "Predict Player Props" for individual predictions.

    - **Same-Game Parlay**:
      - Requires at least two players with complete prop data (Name, Position, Prop Type, Over/Under) and non-zero odds.
      - Enter a single stake for the combined parlay bet. All legs must succeed to win; if any leg loses, the parlay loses.
      - Click "Predict Same Game Parlay" for combined probability and payout. Payout grows with more legs due to multiplied odds.
      - **Parlay Assumption**: The parlay probability assumes independent outcomes for simplicity. If you want to account for correlations between players, contact support for a future enhancement.

    - **Understanding Output**:
      - **Win Probability**: The % chance a team or parlay wins, based on 10,000 simulations (e.g., 62.3%).
      - **95% CI**: The range where the true probability likely falls (e.g., 58.1% - 66.4%), showing prediction reliability.
      - **Bet Outcome**: Shows profit and total return for bets, if stakes and odds are provided.

    Ensure all required fields (marked with *) are filled. Clear inputs with "Clear Inputs" or "Clear Players" to start over.
    """)

# Sport selection
sport = st.selectbox("Select Sport", ["Football", "Baseball", "Basketball", "Hockey"], key="sport")

# Define sport-specific stats and positions
SPORT_STATS = {
    "Football": {
        "team_stats": ["Points", "Yards", "Turnovers"],
        "player_positions": ["QB", "RB", "WR", "TE", "K"],
        "player_stats": ["Passing Yards", "Rushing Yards", "Receptions", "Touchdowns"]
    },
    "Baseball": {
        "team_stats": ["Runs", "Hits", "Errors"],
        "player_positions": ["P", "C", "1B", "2B", "3B", "SS", "LF", "CF", "RF"],
        "player_stats": ["Batting Average", "Home Runs", "RBIs", "Pitcher ERA"]
    },
    "Basketball": {
        "team_stats": ["Points", "Rebounds", "Assists", "Field Goal %"],
        "player_positions": ["PG", "SG", "SF", "PF", "C"],
        "player_stats": ["Points", "Rebounds", "Assists", "Field Goal %"]
    },
    "Hockey": {
        "team_stats": ["Goals", "Shots on Goal", "Save %"],
        "player_positions": ["C", "LW", "RW", "D", "G"],
        "player_stats": ["Goals", "Assists", "Shots on Goal", "Save %"]
    }
}

# Initialize session state for form resets
if "team_form_reset_key" not in st.session_state:
    st.session_state.team_form_reset_key = 0
if "player_form_reset_key" not in st.session_state:
    st.session_state.player_form_reset_key = 0
if "players" not in st.session_state:
    st.session_state.players = [{}]

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
        match = re.search(r"(\d+\.?\d*)", stats_text)
        return float(match.group(1)) if match else 0
    except:
        return 0

# Betting calculation for team bets
def calculate_bet_outcome(bet_type, stake, odds):
    if stake <= 0:
        return {"error": "Stake must be greater than 0", "profit": 0, "total_return": 0}
    if odds == 0:
        return {"error": "Odds cannot be zero", "profit": 0, "total_return": 0}
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

# SGP payout calculation
def american_to_decimal(odds):
    """Convert American odds to decimal odds."""
    if odds > 0:
        return (odds / 100) + 1
    else:
        return (100 / abs(odds)) + 1

def sgp_payout(odds_list, stake):
    """Calculate SGP total return and profit."""
    if stake <= 0:
        return {"error": "Stake must be greater than 0", "profit": 0, "total_return": 0}
    if any(o == 0 for o in odds_list):
        return {"error": "Odds cannot be zero", "profit": 0, "total_return": 0}
    decimal_total = 1
    for o in odds_list:
        decimal_total *= american_to_decimal(o)
    total_return = round(stake * decimal_total, 2)
    profit = round(total_return - stake, 2)
    combined_odds = round((decimal_total - 1) * 100) if decimal_total > 1 else round(-100 / (decimal_total - 1))
    return {
        "profit": profit,
        "total_return": total_return,
        "details": f"Parlay bet: ${stake} at {combined_odds:+} odds"
    }

# Team Prediction Algorithm
def predict_team_outcome(team1_data, team2_data, sport):
    t1_stats = parse_stats(team1_data["stats"], sport)
    t2_stats = parse_stats(team2_data["stats"], sport)
    t1_recent = parse_recent_performance(team1_data["recent"])
    t2_recent = parse_recent_performance(team2_data["recent"])
    
    weights = {"main": 0.4, "secondary": 0.3, "tertiary": 0.2, "recent": 0.1, "home_away": 0.05}
    stat_keys = SPORT_STATS[sport]["team_stats"]
    
    # Base scores
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
    
    # Monte Carlo simulation for win probability
    simulations = 10000
    t1_wins = 0
    score_diffs = []
    main_stat = stat_keys[0]
    
    for _ in range(simulations):
        # Add randomness to simulate variability
        t1_sim_score = np.random.normal(t1_score, t1_score * 0.1)  # 10% standard deviation
        t2_sim_score = np.random.normal(t2_score, t2_score * 0.1)
        score_diffs.append(t1_sim_score - t2_sim_score)
        if t1_sim_score > t2_sim_score:
            t1_wins += 1
    
    # Calculate win probability
    pwp_team1 = (t1_wins / simulations) * 100
    pwp_team2 = 100 - pwp_team1
    
    # Calculate 95% confidence interval
    mean_diff = np.mean(score_diffs)
    std_diff = np.std(score_diffs)
    ci_lower = mean_diff - 1.96 * std_diff / np.sqrt(simulations)
    ci_upper = mean_diff + 1.96 * std_diff / np.sqrt(simulations)
    ci_lower_pwp = (1 / (1 + np.exp(-ci_lower))) * 100  # Convert to probability
    ci_upper_pwp = (1 / (1 + np.exp(-ci_upper))) * 100
    
    # Predicted score
    score_diff = abs(t1_score - t2_score) / 2
    predicted_score1 = round(t1_stats[main_stat] + score_diff if t1_score > t2_score else t1_stats[main_stat] - score_diff)
    predicted_score2 = round(t2_stats[main_stat] - score_diff if t1_score > t2_score else t2_stats[main_stat] + score_diff)
    
    winner = team1_data["name"] if t1_score > t2_score else team2_data["name"]
    win_prob = pwp_team1 if t1_score > t2_score else pwp_team2
    win_team = team1_data["name"] if t1_score > t2_score else team2_data["name"]
    
    factors = [
        f"{main_stat} (Team 1: {t1_stats[main_stat]}, Team 2: {t2_stats[main_stat]})",
        f"Recent Performance (Team 1: {t1_recent:.2f}, Team 2: {t2_recent:.2f})",
        f"Injuries: {team1_data['injuries']} vs. {team2_data['injuries']}",
        f"Home/Away: {team1_data['home_away']} vs. {team2_data['home_away']}",
        f"Rest Days: {team1_data['rest_days']} vs. {team2_data['rest_days']}",
        f"Win Probability: {win_team} {win_prob:.1f}% (95% CI: {min(ci_lower_pwp, ci_upper_pwp):.1f}% - {max(ci_lower_pwp, ci_upper_pwp):.1f}%)"
    ]
    for stat in stat_keys[1:]:
        factors.append(f"{stat} (Team 1: {t1_stats[stat]}, Team 2: {t2_stats[stat]})")
    
    # Calculate bet outcome for the predicted winner if betting data is provided
    bet_info = ""
    winner_data = team1_data if t1_score > t2_score else team2_data
    if winner_data.get("bet_type") and winner_data["bet_type"] != "" and winner_data["stake"] > 0 and winner_data["odds"] != 0:
        bet_result = calculate_bet_outcome(winner_data["bet_type"], winner_data["stake"], winner_data["odds"])
        if "error" in bet_result:
            bet_info = bet_result["error"]
        else:
            bet_info = f"{bet_result['details']}, Profit: ${bet_result['profit']}, Total Return: ${bet_result['total_return']}"
        factors.append(f"Bet for {winner}: {bet_info}")
    
    return f"{winner} wins {predicted_score1}-{predicted_score2}", factors

# Player Prop Prediction Algorithm
def predict_player_prop(player, sport):
    stat_value = parse_player_stats(player["recent_stats"], sport)
    try:
        def_rank = float(re.search(r"Rank: (\d+)", player["opp_defense"]).group(1)) # type: ignore
    except:
        def_rank = 16
    
    likelihood = stat_value * 0.5 + (32 - def_rank) * 0.3
    if player["injury_status"] == "Out":
        likelihood *= 0.0
    elif player["injury_status"] == "Questionable":
        likelihood *= 0.7
    
    try:
        prop_value = float(player["over_under"])
    except:
        prop_value = 0
    
    outcome = "Over" if likelihood > prop_value else "Under"
    confidence = abs(likelihood - prop_value) / prop_value * 100 if prop_value != 0 else 0
    
    return f"{player['name']} likely to hit {outcome} {prop_value} ({confidence:.1f}% confidence)", confidence / 100

# Same Game Parlay Prediction
def predict_same_game_parlay(players, sport, stake):
    simulations = 10000
    successes = 0
    individual_confidences = []
    odds_list = []
    
    for player in players:
        prediction, confidence = predict_player_prop(player, sport)
        individual_confidences.append(confidence)
        odds_list.append(player["odds"])
    
    for _ in range(simulations):
        success = True
        for confidence in individual_confidences:
            if np.random.random() > confidence:
                success = False
                break
        if success:
            successes += 1
    
    parlay_prob = (successes / simulations) * 100
    # Calculate 95% confidence interval
    mean_prob = parlay_prob / 100
    std_prob = np.sqrt(mean_prob * (1 - mean_prob) / simulations)
    ci_lower = (mean_prob - 1.96 * std_prob) * 100
    ci_upper = (mean_prob + 1.96 * std_prob) * 100
    ci_lower = max(0, ci_lower)
    ci_upper = min(100, ci_upper)
    
    # Calculate SGP payout
    bet_info = ""
    if stake > 0:
        bet_result = sgp_payout(odds_list, stake)
        if "error" in bet_result:
            bet_info = bet_result["error"]
        else:
            bet_info = f"{bet_result['details']}, Profit: ${bet_result['profit']}, Total Return: ${bet_result['total_return']}"
    
    factors = [
        f"Individual Player Probabilities: {[f'{player['name']}: {conf*100:.1f}%' for player, conf in zip(players, individual_confidences)]}",
        f"Combined Parlay Probability: {parlay_prob:.1f}% (95% CI: {ci_lower:.1f}% - {ci_upper:.1f}%)"
    ]
    if bet_info:
        factors.append(f"Bet: {bet_info}")
    
    return f"Same Game Parlay: {parlay_prob:.1f}% chance of winning", factors

# Define tabs
tab1, tab2 = st.tabs(["Team Game Prediction", "Player Prop Bets"])

# Team Prediction Tab
with tab1:
    st.header("Team Game Prediction")
    
    with st.form(key=f"team_form_{st.session_state.team_form_reset_key}"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Team 1")
            team1_name = st.text_input("Team 1 Name *", "", key=f"team1_name_{st.session_state.team_form_reset_key}")
            team1_stats = st.text_area(f"Team 1 Stats (e.g., {', '.join(SPORT_STATS[sport]['team_stats'])})", 
                                      "\n".join([f"{stat}: 0" for stat in SPORT_STATS[sport]["team_stats"]]), 
                                      key=f"team1_stats_{st.session_state.team_form_reset_key}")
            team1_recent = st.text_input("Recent Performance (e.g., W-L last 5 games)", "", key=f"team1_recent_{st.session_state.team_form_reset_key}")
            team1_injuries = st.text_area("Key Injuries", "", key=f"team1_injuries_{st.session_state.team_form_reset_key}")
            team1_home_away = st.selectbox("Home/Away *", ["", "Home", "Away", "Neutral"], key=f"team1_home_away_{st.session_state.team_form_reset_key}")
            team1_bet_type = st.selectbox("Bet Type (Team 1)", ["", "Moneyline", "Point Spread", "Over/Under"], key=f"team1_bet_type_{st.session_state.team_form_reset_key}")
            team1_odds = st.number_input("Odds (Team 1, e.g., +150 or -110)", min_value=-10000, max_value=10000, value=0, key=f"team1_odds_{st.session_state.team_form_reset_key}")
            team1_stake = st.number_input("Stake (Team 1, $)", min_value=0.0, value=0.0, step=0.01, key=f"team1_stake_{st.session_state.team_form_reset_key}")
        
        with col2:
            st.subheader("Team 2")
            team2_name = st.text_input("Team 2 Name *", "", key=f"team2_name_{st.session_state.team_form_reset_key}")
            team2_stats = st.text_area(f"Team 2 Stats (e.g., {', '.join(SPORT_STATS[sport]['team_stats'])})", 
                                      "\n".join([f"{stat}: 0" for stat in SPORT_STATS[sport]["team_stats"]]), 
                                      key=f"team2_stats_{st.session_state.team_form_reset_key}")
            team2_recent = st.text_input("Recent Performance (e.g., W-L last 5 games)", "", key=f"team2_recent_{st.session_state.team_form_reset_key}")
            team2_injuries = st.text_area("Key Injuries", "", key=f"team2_injuries_{st.session_state.team_form_reset_key}")
            team2_home_away = st.selectbox("Home/Away *", ["", "Home", "Away", "Neutral"], key=f"team2_home_away_{st.session_state.team_form_reset_key}")
            team2_bet_type = st.selectbox("Bet Type (Team 2)", ["", "Moneyline", "Point Spread", "Over/Under"], key=f"team2_bet_type_{st.session_state.team_form_reset_key}")
            team2_odds = st.number_input("Odds (Team 2, e.g., +150 or -110)", min_value=-10000, max_value=10000, value=0, key=f"team2_odds_{st.session_state.team_form_reset_key}")
            team2_stake = st.number_input("Stake (Team 2, $)", min_value=0.0, value=0.0, step=0.01, key=f"team2_stake_{st.session_state.team_form_reset_key}")
        
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
            # Validate betting inputs
            betting_errors = []
            for team, data in [("Team 1", {"bet_type": team1_bet_type, "odds": team1_odds, "stake": team1_stake}),
                              ("Team 2", {"bet_type": team2_bet_type, "odds": team2_odds, "stake": team2_stake})]:
                if data["stake"] > 0 and (data["bet_type"] == "" or data["odds"] == 0):
                    betting_errors.append(f"Please provide Bet Type and non-zero Odds for {team} if Stake is entered.")
            if betting_errors:
                for error in betting_errors:
                    st.error(error)
            else:
                team1_data = {
                    "name": team1_name, "stats": team1_stats, "recent": team1_recent or "0-0",
                    "injuries": team1_injuries or "None", "home_away": team1_home_away or "Neutral", "rest_days": rest_days_team1,
                    "bet_type": team1_bet_type, "odds": team1_odds, "stake": team1_stake
                }
                team2_data = {
                    "name": team2_name, "stats": team2_stats, "recent": team2_recent or "0-0",
                    "injuries": team2_injuries or "None", "home_away": team2_home_away or "Neutral", "rest_days": rest_days_team2,
                    "bet_type": team2_bet_type, "odds": team2_odds, "stake": team2_stake
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
                    player_name = st.text_input("Player Name *", "", key=f"player_name_{i}_{st.session_state.player_form_reset_key}")
                    position = st.selectbox("Position *", [""] + SPORT_STATS[sport]["player_positions"], key=f"position_{i}_{st.session_state.player_form_reset_key}")
                    recent_stats = st.text_area(f"Recent Stats (e.g., {', '.join(SPORT_STATS[sport]['player_stats'])})", 
                                               "", key=f"recent_stats_{i}_{st.session_state.player_form_reset_key}")
                with col2:
                    prop_type = st.selectbox("Prop Type *", [""] + SPORT_STATS[sport]["player_stats"], key=f"prop_type_{i}_{st.session_state.player_form_reset_key}")
                    over_under = st.text_input("Over/Under * (e.g., 25.5)", "", key=f"over_under_{i}_{st.session_state.player_form_reset_key}")
                    odds = st.number_input("Odds * (e.g., +150 or -110)", min_value=-10000, max_value=10000, value=0, key=f"odds_{i}_{st.session_state.player_form_reset_key}")
                    opp_defense = st.text_input("Opposing Defense (e.g., Rank)", "Rank: 0", key=f"opp_defense_{i}_{st.session_state.player_form_reset_key}")
                    injury_status = st.selectbox("Injury Status", ["", "Healthy", "Questionable", "Out"], key=f"injury_status_{i}_{st.session_state.player_form_reset_key}")
                
                st.session_state.players[i] = {
                    "name": player_name, "position": position, "recent_stats": recent_stats,
                    "prop_type": prop_type, "over_under": over_under, "odds": odds,
                    "opp_defense": opp_defense, "injury_status": injury_status or "Healthy",
                    "prop_value": over_under
                }
        
        # Single stake input for SGP
        parlay_stake = st.number_input("Parlay Stake ($)", min_value=0.0, value=0.0, step=0.01, key=f"parlay_stake_{st.session_state.player_form_reset_key}")
        
        add_player_btn = st.form_submit_button("Add Player")
        predict_props = st.form_submit_button("Predict Player Props")
        predict_parlay = st.form_submit_button("Predict Same Game Parlay")
        clear_players = st.form_submit_button("Clear Players")
    
    if add_player_btn:
        add_player()
        st.rerun()
    
    if predict_props:
        results = []
        for player in st.session_state.players:
            if not player["name"] or player["prop_type"] == "" or player["position"] == "" or player["over_under"] == "":
                st.error(f"Please fill in all required fields for Player {st.session_state.players.index(player) + 1} (Name, Position, Prop Type, Over/Under).")
                break
            try:
                float(player["over_under"])
            except:
                st.error(f"Over/Under for Player {st.session_state.players.index(player) + 1} must be a number (e.g., 25.5).")
                break
            if player["odds"] == 0:
                st.error(f"Please provide non-zero Odds for Player {st.session_state.players.index(player) + 1}.")
                break
            prediction, confidence = predict_player_prop(player, sport)
            results.append(f"{prediction}")
        if results:
            st.success("\n\n".join(results))
    
    if predict_parlay:
        if len(st.session_state.players) < 2:
            st.error("Please add at least two players for a same-game parlay.")
        else:
            valid_players = []
            for player in st.session_state.players:
                if not player["name"] or player["prop_type"] == "" or player["position"] == "" or player["over_under"] == "":
                    continue
                try:
                    float(player["over_under"])
                except:
                    st.error(f"Over/Under for Player {st.session_state.players.index(player) + 1} must be a number (e.g., 25.5).")
                    break
                if player["odds"] == 0:
                    st.error(f"Please provide non-zero Odds for Player {st.session_state.players.index(player) + 1}.")
                    break
                valid_players.append(player)
            else:
                if len(valid_players) < 2:
                    st.error("Please fill in all required fields (Name, Position, Prop Type, Over/Under, Odds) for at least two players.")
                else:
                    if parlay_stake > 0 and len(valid_players) < 2:
                        st.error("Parlay with a stake requires at least two players with complete betting data.")
                    else:
                        prediction, factors = predict_same_game_parlay(valid_players, sport, parlay_stake)
                        st.success(f"Prediction: {prediction}")
                        st.write("Key Factors:")
                        for factor in factors:
                            st.write(f"- {factor}")
    
    if clear_players:
        st.session_state.players = [{}]
        st.session_state.player_form_reset_key += 1
        st.rerun()

# Footer
st.markdown("---")
st.write("Made with ❤️ by The Sports BetTracker Team | &copy; 2025 103 Software Solutions LLC")
st.write("Powered by xAI Grok 3 | Statistical Algorithm")