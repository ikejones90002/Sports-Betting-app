# -------------------------------------------------------------
# Sports BetTracker – Full script (with realistic per‑period scores,
# MMA score‑cards, scoring‑type selector and winner‑first display)
# -------------------------------------------------------------
import re
import numpy as np               # type: ignore
import streamlit as st           # type: ignore
import matplotlib.pyplot as plt  # type: ignore

# -------------------------------------------------------------
# 0️⃣ Page configuration & branding
# -------------------------------------------------------------
st.set_page_config(
    page_title="Sports BetTracker",
    layout="wide",
)

st.image(
    "sports-bettracker-logo.png",
    width=200,
    caption="Sports BetTracker – Track the Action. Bet Smarter.",
)
st.title("🏀⚾🏒🏈👊 Sports BetTracker")

# -------------------------------------------------------------
# 1️⃣ How‑to guide
# -------------------------------------------------------------
with st.expander("How to Use Sports BetTracker"):
    st.markdown(
        """
**Quick Guide**

* **Team Game Prediction** – Fill in two teams, their stats, recent form, injuries, home/away, and (optionally) betting details. Click **Predict** for:

  - Winner & realistic per‑period scores (quarters, halves, periods)  
  - First‑to‑score (or first significant strike for MMA)  
  - Winning margin (points or rounds)  
  - Win probability with 95 % CI  
  - Betting outcome if odds/stake supplied  

* **Player Prop Bets** – Add players, recent stats, prop type, line/option, odds & opponent info. Click **Predict Player Props**.

* **Same‑Game Parlay** – Add ≥ 2 players, set a combined stake, click **Predict Same‑Game Parlay**.

Enjoy smarter betting! 🎯
        """
    )

# -------------------------------------------------------------
# 3️⃣ Detailed sport definitions (periods, positions, props…)
# -------------------------------------------------------------
SPORT_STATS = {
    "Football": {
        "team_stats": ["Points", "Yards", "Turnovers"],
        "team_periods": ["Quarter 1", "Quarter 2", "Quarter 3", "Quarter 4"],
        "player_positions": ["QB", "RB", "WR", "TE", "K"],
        "player_stats": ["Passing Yards", "Rushing Yards", "Receptions", "Touchdowns"],
        "prop_types": ["Passing Yards", "Rushing Yards", "Receptions", "Touchdowns"],
        "score_types": ["Touchdown", "Field Goal", "Safety", "PAT"],
    },
    "Baseball": {
        "team_stats": ["Runs", "Hits", "Errors"],
        "team_periods": [
            "Inning 1",
            "Inning 2",
            "Inning 3",
            "Inning 4",
            "Inning 5",
            "Inning 6",
            "Inning 7",
            "Inning 8",
            "Inning 9",
        ],
        "player_positions": [
            "P",
            "C",
            "1B",
            "2B",
            "3B",
            "SS",
            "LF",
            "CF",
            "RF",
        ],
        "player_stats": ["Batting Average", "Home Runs", "RBIs", "Pitcher ERA"],
        "prop_types": ["Batting Average", "Home Runs", "RBIs", "Pitcher ERA"],
        "score_types": ["Single", "Double", "Triple", "Home Run"],
    },
    "Basketball": {
        "team_stats": ["Points", "Rebounds", "Assists", "Field Goal %"],
        "team_periods": ["Quarter 1", "Quarter 2", "Quarter 3", "Quarter 4"],
        "player_positions": ["PG", "SG", "SF", "PF", "C"],
        "player_stats": ["Points", "Rebounds", "Assists", "Field Goal %"],
        "prop_types": ["Points", "Rebounds", "Assists", "Field Goal %"],
        "score_types": ["2-pt FG", "3-pt FG", "Free Throw"],
    },
    "Hockey": {
        "team_stats": ["Goals", "Shots on Goal", "Save %"],
        "team_periods": ["Period 1", "Period 2", "Period 3", "Overtime"],
        "player_positions": ["C", "LW", "RW", "D", "G"],
        "player_stats": ["Goals", "Assists", "Shots on Goal", "Save %"],
        "prop_types": ["Goals", "Assists", "Shots on Goal", "Save %"],
        "score_types": ["Even Strength", "Power Play", "Short-Handed", "Empty Net"],
    },
    "MMA": {
        "team_stats": ["Strikes", "Takedowns", "Submissions"],
        "team_periods": ["Round 1", "Round 2", "Round 3", "Round 4", "Round 5"],
        "player_positions": ["Fighter"],
        "player_stats": [
            "SLpM",
            "Str Acc",
            "TD Avg",
            "TD Acc",
            "Sub Avg",
            "KO %",
            "Sub %",
            "Dec %",
            "Avg Rounds",
        ],
        "prop_types": [
            "Method of Victory",
            "Fight Goes the Distance",
            "Inside the Distance",
            "Round Betting",
            "Total Rounds Over/Under",
        ],
        "score_types": ["KO/TKO", "Submission", "Decision", "DQ"],
    },
}

# -------------------------------------------------------------
# 2️⃣ Sport selector + scoring‑type selector
# -------------------------------------------------------------
sport = st.selectbox(
    "Select Sport",
    ["Football", "Baseball", "Basketball", "Hockey", "MMA"],
    key="sport",
)

if sport != "MMA":
    score_type = st.selectbox(
        "Scoring type (main stat for this sport)",
        SPORT_STATS[sport]["team_stats"],
        key="score_type",
    )
else:
    score_type = "Round points"          # fixed for MMA

# -------------------------------------------------------------
# 4️⃣ Session‑state init
# -------------------------------------------------------------
if "team_form_reset_key" not in st.session_state:
    st.session_state.team_form_reset_key = 0
if "player_form_reset_key" not in st.session_state:
    st.session_state.player_form_reset_key = 0
if "players" not in st.session_state:
    st.session_state.players = [{}]

# -------------------------------------------------------------
# 5️⃣ Helper functions (parsing, betting, etc.)
# -------------------------------------------------------------
def parse_stats(stats_text: str, sport_key: str) -> dict:
    out = {}
    for stat in SPORT_STATS[sport_key]["team_stats"]:
        m = re.search(rf"{stat}: (\d+\.?\d*)", stats_text)
        out[stat] = float(m.group(1)) if m else 0.0
    return out


def parse_recent_performance(record: str) -> float:
    try:
        w, l = map(int, record.split("-"))
        tot = w + l
        return w / tot if tot else 0.5
    except Exception:
        return 0.5


def parse_player_stats(txt: str) -> dict:
    d = {}
    for k, v in re.findall(r"([\w %]+): (\d+\.?\d*)", txt):
        d[k.strip().replace(" %", "")] = float(v)
    if not d:
        m = re.search(r"(\d+\.?\d*)", txt)
        d["default"] = float(m.group(1)) if m else 0.0
    return d


def calculate_bet_outcome(bet_type: str, stake: float, odds: int) -> dict:
    if stake <= 0:
        return {"error": "Stake must be > 0", "profit": 0, "total_return": 0}
    if odds == 0:
        return {"error": "Odds cannot be zero", "profit": 0, "total_return": 0}
    if bet_type == "Moneyline":
        profit = stake * (odds / 100) if odds > 0 else stake / (abs(odds) / 100)
    else:
        profit = stake * (100 / abs(odds))
    profit = round(profit, 2)
    return {
        "profit": profit,
        "total_return": round(stake + profit, 2),
        "details": f"{bet_type} @ {odds:+}",
    }


def american_to_decimal(odds: int) -> float:
    return (odds / 100) + 1 if odds > 0 else (100 / abs(odds)) + 1


def sgp_payout(odds_list: list[int], stake: float) -> dict:
    if stake <= 0:
        return {"error": "Stake must be > 0", "profit": 0, "total_return": 0}
    if any(o == 0 for o in odds_list):
        return {"error": "Odds cannot be zero", "profit": 0, "total_return": 0}
    dec = 1.0
    for o in odds_list:
        dec *= american_to_decimal(o)
    total = round(stake * dec, 2)
    return {
        "profit": round(total - stake, 2),
        "total_return": total,
        "details": f"Parlay @ {round((dec - 1) * 100, 0):+}",
    }

# -------------------------------------------------------------
# 6️⃣ Prediction routine (now MMA‑aware and realistic per‑period)
# -------------------------------------------------------------
def predict_team_outcome(team1: dict, team2: dict, sport_key: str, main_score_type: str) -> tuple[str, list]:
    """
    Returns:
        - Human‑readable prediction (winner first)
        - List of explanatory factors
    Handles:
        * regular team sports – realistic per‑period scores (Poisson)
        * MMA – score‑card (10‑9, KO/TKO, etc.)
    """
    # ----- common parsing -------------------------------------------------
    t1_stats = parse_stats(team1["stats"], sport_key)
    t2_stats = parse_stats(team2["stats"], sport_key)
    t1_recent = parse_recent_performance(team1["recent"])
    t2_recent = parse_recent_performance(team2["recent"])

    # ----- weighted core score -------------------------------------------
    w = {"main": 0.4, "secondary": 0.3, "tertiary": 0.2, "recent": 0.1, "home_away": 0.05}
    keys = SPORT_STATS[sport_key]["team_stats"]

    # main stat is the one the user selected (e.g., Points, Runs, Goals)
    main = main_score_type

    # basic weighted score
    t1_score = t1_stats[main] * w["main"]
    t2_score = t2_stats[main] * w["main"]

    # secondary / tertiary when they exist
    if len(keys) > 1:
        factor = 10 if sport_key == "Football" else 1
        sec = keys[1]
        t1_score += t1_stats[sec] / factor * w["secondary"]
        t2_score += t2_stats[sec] / factor * w["secondary"]
    if len(keys) > 2:
        factor = 10 if sport_key == "Football" else 1
        ter = keys[2]
        t1_score -= t1_stats[ter] * factor * w["tertiary"]
        t2_score -= t2_stats[ter] * factor * w["tertiary"]

    # recent, home/away, injuries, rest‑day
    t1_score += t1_recent * 20 * w["recent"]
    t2_score += t2_recent * 20 * w["recent"]
    if team1["home_away"] == "Home":
        t1_score *= 1 + w["home_away"]
    elif team1["home_away"] == "Away":
        t1_score *= 1 - w["home_away"]
    if team2["home_away"] == "Home":
        t2_score *= 1 + w["home_away"]
    elif team2["home_away"] == "Away":
        t2_score *= 1 - w["home_away"]
    t1_score *= 0.95 if "out" in team1["injuries"].lower() else 1.0
    t2_score *= 0.95 if "out" in team2["injuries"].lower() else 1.0
    rd = team1["rest_days"] - team2["rest_days"]
    t1_score *= 1 + 0.02 * rd
    t2_score *= 1 - 0.02 * rd

    # ----- Monte‑Carlo win probability ------------------------------------
    sims = 10_000
    t1_wins = 0
    diffs = []
    for _ in range(sims):
        s1 = np.random.normal(t1_score, t1_score * 0.1)
        s2 = np.random.normal(t2_score, t2_score * 0.1)
        diffs.append(s1 - s2)
        if s1 > s2:
            t1_wins += 1
    win_p1 = t1_wins / sims * 100
    win_p2 = 100 - win_p1

    # 95 % CI (logistic conversion)
    mean_diff = np.mean(diffs)
    std_diff = np.std(diffs)
    ci_low = mean_diff - 1.96 * std_diff / np.sqrt(sims)
    ci_up = mean_diff + 1.96 * std_diff / np.sqrt(sims)
    ci_low_p = (1 / (1 + np.exp(-ci_low))) * 100
    ci_up_p = (1 / (1 + np.exp(-ci_up))) * 100

    # -----------------------------------------------------------------
    # 7️⃣  Build sport‑specific output
    # -----------------------------------------------------------------
    if sport_key != "MMA":
        # ----- REALISTIC PER‑PERIOD SCORES (Poisson) -----
        periods = SPORT_STATS[sport_key]["team_periods"]
        n_periods = len(periods)

        # expected points per period (average of main stat)
        t1_avg = t1_stats[main] / n_periods if n_periods else t1_stats[main]
        t2_avg = t2_stats[main] / n_periods if n_periods else t2_stats[main]

        # generate Poisson scores, then scale to match the *rounded* totals we will predict
        np.random.seed(0)    # reproducible within a single run
        t1_period_raw = np.random.poisson(lam=t1_avg, size=n_periods)
        t2_period_raw = np.random.poisson(lam=t2_avg, size=n_periods)

        # totals we intend to display (rounded version of the weighted scores)
        total_pred = round(t1_score) if t1_score > t2_score else round(t2_score)
        loser_pred = round(t2_score) if t1_score > t2_score else round(t1_score)

        # scale raw Poisson arrays to hit the totals (simple proportional scaling)
        def scale_to_total(arr, target):
            if arr.sum() == 0:
                return np.full_like(arr, int(target / len(arr)))
            factor = target / arr.sum()
            scaled = np.round(arr * factor).astype(int)
            # adjust for rounding error
            diff = target - scaled.sum()
            while diff != 0:
                idx = np.argmax(arr) if diff > 0 else np.argmin(arr)
                scaled[idx] += 1 if diff > 0 else -1
                diff = target - scaled.sum()
            return scaled

        t1_per = scale_to_total(t1_period_raw, total_pred) if t1_score > t2_score else scale_to_total(t1_period_raw, loser_pred)
        t2_per = scale_to_total(t2_period_raw, loser_pred) if t1_score > t2_score else scale_to_total(t2_period_raw, total_pred)

        # ----- Assemble prediction string (winner first) -----
        winner = team1["name"] if t1_score > t2_score else team2["name"]
        loser = team2["name"] if t1_score > t2_score else team1["name"]
        win_score = total_pred
        lose_score = loser_pred
        pred_str = f"{winner} {win_score} - {loser} {lose_score}"

        # winning margin
        margin = win_score - lose_score

        # first‑to‑score heuristic (same as before)
        if t1_recent > t2_recent:
            first = team1["name"]
            first_type = "Touchdown / Run / Goal"
        elif t2_recent > t1_recent:
            first = team2["name"]
            first_type = "Touchdown / Run / Goal"
        else:
            first = team1["name"] if team1["home_away"] == "Home" else team2["name"]
            first_type = "Touchdown / Run / Goal"

        # factor list
        factors = [
            f"Main stat ({main}) – {team1['name']}: {t1_stats[main]}, {team2['name']}: {t2_stats[main]}",
            f"Recent win% – {team1['name']}: {t1_recent:.2f}, {team2['name']}: {t2_recent:.2f}",
            f"Injuries – {team1['injuries']} vs {team2['injuries']}",
            f"Home/Away – {team1['home_away']} vs {team2['home_away']}",
            f"Rest days – {team1['rest_days']} vs {team2['rest_days']}",
            f"Win probability – {winner} {win_p1 if winner == team1['name'] else win_p2:.1f}% (95 % CI {min(ci_low_p,ci_up_p):.1f}%–{max(ci_low_p,ci_up_p):.1f}%)",
            f"Winning margin – {margin}",
            f"First to score – {first} ({first_type})",
            "Score breakdown by period:",
        ]
        for p, w, l in zip(periods, t1_per, t2_per):
            factors.append(f"  • {p}: {w}-{l}")

        # betting outcome (if supplied)
        bet_info = ""
        win_data = team1 if winner == team1["name"] else team2
        if win_data.get("bet_type") and win_data["bet_type"] != "" and win_data["stake"] > 0:
            bet_res = calculate_bet_outcome(win_data["bet_type"], win_data["stake"], win_data["odds"])
            if "error" in bet_res:
                bet_info = bet_res["error"]
            else:
                bet_info = f"{bet_res['details']}, Profit ${bet_res['profit']}, Return ${bet_res['total_return']}"
            factors.append(f"Bet for {winner}: {bet_info}")

        return pred_str, factors

    else:
        # -------------------- MMA SECTION --------------------
        max_rounds = 5

        # Determine finishing round (use the Monte‑Carlo scores as a proxy)
        min_sc, max_sc = min(t1_score, t2_score), max(t1_score, t2_score)
        if max_sc == min_sc:
            finish_round = 3
        else:
            prop = (max(t1_score, t2_score) - min_sc) / (max_sc - min_sc)
            finish_round = int(np.clip(round(prop * max_rounds), 1, max_rounds))

        winner = team1["name"] if t1_score > t2_score else team2["name"]
        loser = team2["name"] if t1_score > t2_score else team1["name"]
        win_prob = win_p1 if t1_score > t2_score else win_p2

        # Simple method‑of‑victory heuristic (same as player‑prop)
        t1_method = (
            t1_recent * 0.5
            + (t1_stats.get("Strikes", 0) + t1_stats.get("Takedowns", 0)) * 0.3
        )
        t2_method = (
            t2_recent * 0.5
            + (t2_stats.get("Strikes", 0) + t2_stats.get("Takedowns", 0)) * 0.3
        )
        method = "KO/TKO" if (t1_method if winner == team1["name"] else t2_method) > 0.6 else "Decision"

        # Build score‑card
        if method == "KO/TKO":
            scorecard = f"KO/TKO in round {finish_round}"
        else:
            # decision – show round‑by‑round 10‑9 scores (winner gets 10 each round)
            scores = ["10-9"] * finish_round
            scorecard = ", ".join(scores)

        pred_str = f"{winner} wins via {method} ({scorecard})"

        # Winning margin expressed in rounds (how many rounds the winner lasted)
        margin_rounds = max_rounds - finish_round + 1

        # First significant strike (same heuristic as before)
        if t1_recent > t2_recent:
            first_striker = team1["name"]
        elif t2_recent > t1_recent:
            first_striker = team2["name"]
        else:
            first_striker = team1["name"] if team1["home_away"] == "Home" else team2["name"]

        # factors list
        factors = [
            f"Recent win% – {team1['name']}: {t1_recent:.2f}, {team2['name']}: {t2_recent:.2f}",
            f"Injuries – {team1['injuries']} vs {team2['injuries']}",
            f"Home/Away – {team1['home_away']} vs {team2['home_away']}",
            f"Rest days – {team1['rest_days']} vs {team2['rest_days']}",
            f"Win probability – {winner} {win_prob:.1f}% (95 % CI {min(ci_low_p,ci_up_p):.1f}%–{max(ci_low_p,ci_up_p):.1f}%)",
            f"Method of victory – {method}",
            f"Finishing round – {finish_round}",
            f"Margin – {margin_rounds} round{'s' if margin_rounds > 1 else ''}",
            f"First significant strike – {first_striker}",
            "Scorecard:",
            f"  {scorecard}",
        ]

        # betting outcome (unchanged)
        bet_info = ""
        win_data = team1 if winner == team1["name"] else team2
        if win_data.get("bet_type") and win_data["bet_type"] != "" and win_data["stake"] > 0:
            bet_res = calculate_bet_outcome(win_data["bet_type"], win_data["stake"], win_data["odds"])
            if "error" in bet_res:
                bet_info = bet_res["error"]
            else:
                bet_info = f"{bet_res['details']}, Profit ${bet_res['profit']}, Return ${bet_res['total_return']}"
            factors.append(f"Bet for {winner}: {bet_info}")

        return pred_str, factors


# -------------------------------------------------------------
# 9️⃣  PLAYER‑PROP & SAME‑GAME PARLAY (unchanged)
# -------------------------------------------------------------
def predict_player_prop(player: dict, sport_key: str) -> tuple[str, float]:
    p_stats = parse_player_stats(player["recent_stats"])
    opp = parse_player_stats(player["opp_defense"])
    def_rank = opp.get("Rank", 16)
    injury_adj = (
        0.0
        if player["injury_status"] == "Out"
        else 0.7
        if player["injury_status"] == "Questionable"
        else 1.0
    )

    if sport_key != "MMA":
        mean = p_stats.get(player["prop_type"], p_stats.get("default", 0))
        sd = max(mean * 0.15, 1)
        mean *= (1 + (def_rank - 16) * 0.02) * injury_adj
        try:
            line_val = float(player["over_under"])
            numeric = True
        except ValueError:
            numeric = False

        if numeric:
            sims = np.random.normal(mean, sd, 10_000)
            prob_over = (sims > line_val).mean()
            if player["bet_side"] == "Over":
                prob = prob_over
                outcome = f"Over {line_val}"
            elif player["bet_side"] == "Under":
                prob = 1 - prob_over
                outcome = f"Under {line_val}"
            else:
                prob = max(prob_over, 1 - prob_over)
                outcome = f"{'Over' if prob_over > 0.5 else 'Under'} {line_val}"
        else:
            prob = min(max(mean / 100 * injury_adj, 0), 1)
            outcome = player["over_under"]
    else:
        ko = p_stats.get("KO %", 0) / 100 * injury_adj
        sub = p_stats.get("Sub %", 0) / 100 * injury_adj
        dec = p_stats.get("Dec %", 0) / 100 * injury_adj
        total = ko + sub + dec
        if total:
            ko, sub, dec = ko / total, sub / total, dec / total
        ko += (opp.get("KO Vuln", 20) / 100 - 0.2)
        sub += (opp.get("Sub Vuln", 20) / 100 - 0.2)
        dec += (opp.get("Dec Vuln", 20) / 100 - 0.2)
        total = ko + sub + dec
        if total:
            ko, sub, dec = ko / total, sub / total, dec / total

        pt = player["prop_type"]
        line = player["over_under"]
        if pt == "Method of Victory":
            if "KO" in line or "TKO" in line:
                prob = ko
            elif "Submission" in line:
                prob = sub
            else:
                prob = dec
            outcome = line
        elif pt == "Fight Goes the Distance":
            prob = dec if line.lower() == "yes" else 1 - dec
            outcome = f"Go the Distance: {line}"
        elif pt == "Inside the Distance":
            prob = (ko + sub) if line.lower() == "yes" else 1 - (ko + sub)
            outcome = f"Inside Distance: {line}"
        elif pt == "Total Rounds Over/Under":
            try:
                target = float(line)
            except ValueError:
                target = 2.5
            mean_r = p_stats.get("Avg Rounds", 2.5)
            sims = np.random.normal(mean_r, 1.0, 10_000)
            sims = np.clip(sims, 0.5, 5.0)
            prob_over = (sims > target).mean()
            if player["bet_side"] == "Over":
                prob = prob_over
                outcome = f"Over {target}"
            elif player["bet_side"] == "Under":
                prob = 1 - prob_over
                outcome = f"Under {target}"
            else:
                prob = max(prob_over, 1 - prob_over)
                outcome = f"{'Over' if prob_over > 0.5 else 'Under'} {target}"
        elif pt == "Round Betting":
            try:
                match = re.search(r"\d+", line)
                if match:
                    r = float(match.group())
                else:
                    r = 1.0
            except Exception:
                r = 1.0
            per_round = (ko + sub) / 3.0
            prob = (1 - per_round) ** (r - 1) * per_round
            outcome = line
        else:
            prob = 0.5
            outcome = line

    if prob >= 0.5:
        txt = f"{player['name']} likely to {outcome} ({prob*100:.1f}% prob.)"
    else:
        txt = f"{player['name']} likely NOT to {outcome} ({(1-prob)*100:.1f}% prob.)"
    return txt, prob


def predict_same_game_parlay(players: list[dict], sport_key: str, stake: float) -> tuple[str, list]:
    sims = 10_000
    successes = 0
    confidences = []
    odds = []
    for pl in players:
        _, c = predict_player_prop(pl, sport_key)
        confidences.append(c)
        odds.append(pl["odds"])

    for _ in range(sims):
        if all(np.random.random() <= c for c in confidences):
            successes += 1

    parlay_prob = successes / sims * 100
    mean = parlay_prob / 100
    std = np.sqrt(mean * (1 - mean) / sims)
    ci_low = max(0, (mean - 1.96 * std) * 100)
    ci_up = min(100, (mean + 1.96 * std) * 100)

    bet_msg = ""
    if stake > 0:
        bet_res = sgp_payout(odds, stake)
        if "error" not in bet_res:
            bet_msg = f"{bet_res['details']}, Profit ${bet_res['profit']}, Return ${bet_res['total_return']}"

    player_win_percents = [f"{pl['name']}: {c*100:.1f}%" for pl, c in zip(players, confidences)]
    factors = [
        f"Individual player win %: {player_win_percents}",
        f"Combined parlay win %: {parlay_prob:.1f}% (95 % CI {ci_low:.1f}%–{ci_up:.1f}%)",
    ]
    if bet_msg:
        factors.append(f"Bet: {bet_msg}")

    return f"Same‑Game Parlay win %: {parlay_prob:.1f}%", factors


# -------------------------------------------------------------
# 10️⃣  UI – two tabs (Team & Player)
# -------------------------------------------------------------
tab_game, tab_props = st.tabs(["Team Game Prediction", "Player Prop Bets"])

# -----------------------------------------------------------------
#   TAB 1 – TEAM GAME PREDICTION
# -----------------------------------------------------------------
with tab_game:
    st.header("Team Game Prediction")

    with st.form(key=f"team_form_{st.session_state.team_form_reset_key}"):
        c1, c2 = st.columns(2)

        # ---------- TEAM 1 ----------
        with c1:
            st.subheader("Team 1")
            t1_name = st.text_input("Name *", key=f"t1_name_{st.session_state.team_form_reset_key}")
            t1_stats = st.text_area(
                f"Stats (e.g., {', '.join(SPORT_STATS[sport]['team_stats'])})",
                "\n".join([f"{s}: 0" for s in SPORT_STATS[sport]["team_stats"]]),
                key=f"t1_stats_{st.session_state.team_form_reset_key}",
            )
            t1_recent = st.text_input("Recent (W‑L)", key=f"t1_recent_{st.session_state.team_form_reset_key}")
            t1_inj = st.text_area("Key Injuries", key=f"t1_inj_{st.session_state.team_form_reset_key}")
            t1_home = st.selectbox("Home/Away *", ["", "Home", "Away", "Neutral"],
                                   key=f"t1_home_{st.session_state.team_form_reset_key}")
            t1_bet_type = st.selectbox("Bet Type", ["", "Moneyline", "Point Spread", "Over/Under"],
                                       key=f"t1_bet_type_{st.session_state.team_form_reset_key}")
            t1_odds = st.number_input("Odds", -10_000, 10_000, 0,
                                      key=f"t1_odds_{st.session_state.team_form_reset_key}")
            t1_stake = st.number_input("Stake $", 0.0, step=0.01,
                                       key=f"t1_stake_{st.session_state.team_form_reset_key}")

        # ---------- TEAM 2 ----------
        with c2:
            st.subheader("Team 2")
            t2_name = st.text_input("Name *", key=f"t2_name_{st.session_state.team_form_reset_key}")
            t2_stats = st.text_area(
                f"Stats (e.g., {', '.join(SPORT_STATS[sport]['team_stats'])})",
                "\n".join([f"{s}: 0" for s in SPORT_STATS[sport]["team_stats"]]),
                key=f"t2_stats_{st.session_state.team_form_reset_key}",
            )
            t2_recent = st.text_input("Recent (W‑L)", key=f"t2_recent_{st.session_state.team_form_reset_key}")
            t2_inj = st.text_area("Key Injuries", key=f"t2_inj_{st.session_state.team_form_reset_key}")
            t2_home = st.selectbox("Home/Away *", ["", "Home", "Away", "Neutral"],
                                   key=f"t2_home_{st.session_state.team_form_reset_key}")
            t2_bet_type = st.selectbox("Bet Type", ["", "Moneyline", "Point Spread", "Over/Under"],
                                       key=f"t2_bet_type_{st.session_state.team_form_reset_key}")
            t2_odds = st.number_input("Odds", -10_000, 10_000, 0,
                                      key=f"t2_odds_{st.session_state.team_form_reset_key}")
            t2_stake = st.number_input("Stake $", 0.0, step=0.01,
                                       key=f"t2_stake_{st.session_state.team_form_reset_key}")

        # ---------- GAME CONTEXT ----------
        with st.expander("Game Context"):
            game_type = st.selectbox(
                "Game Type", ["", "Regular", "Playoffs", "Preseason"],
                key=f"game_type_{st.session_state.team_form_reset_key}",
            )
            weather = st.selectbox(
                "Weather", ["", "Clear", "Rain", "Snow", "Windy"],
                key=f"weather_{st.session_state.team_form_reset_key}",
            )
            r1 = st.slider("Rest days – Team 1", 0, 14, 0,
                           key=f"rest1_{st.session_state.team_form_reset_key}")
            r2 = st.slider("Rest days – Team 2", 0, 14, 0,
                           key=f"rest2_{st.session_state.team_form_reset_key}")

        submit = st.form_submit_button("Predict Game Outcome")
        clear = st.form_submit_button("Clear Inputs")

    # -----------------------------------------------------------------
    #   Handle submit / clear
    # -----------------------------------------------------------------
    if submit:
        if not t1_name or not t2_name:
            st.error("Both team names are required.")
        elif t1_home == "" or t2_home == "":
            st.error("Select Home/Away for both teams.")
        else:
            # betting validation
            bet_err = []
            for lbl, d in [
                ("Team 1", {"bet_type": t1_bet_type, "odds": t1_odds, "stake": t1_stake}),
                ("Team 2", {"bet_type": t2_bet_type, "odds": t2_odds, "stake": t2_stake}),
            ]:
                if d["stake"] > 0 and (d["bet_type"] == "" or d["odds"] == 0):
                    bet_err.append(f"{lbl}: provide Bet Type and non‑zero Odds if Stake > 0.")
            if bet_err:
                for e in bet_err:
                    st.error(e)
            else:
                t1_dict = {
                    "name": t1_name,
                    "stats": t1_stats,
                    "recent": t1_recent or "0-0",
                    "injuries": t1_inj or "None",
                    "home_away": t1_home,
                    "rest_days": r1,
                    "bet_type": t1_bet_type,
                    "odds": t1_odds,
                    "stake": t1_stake,
                }
                t2_dict = {
                    "name": t2_name,
                    "stats": t2_stats,
                    "recent": t2_recent or "0-0",
                    "injuries": t2_inj or "None",
                    "home_away": t2_home,
                    "rest_days": r2,
                    "bet_type": t2_bet_type,
                    "odds": t2_odds,
                    "stake": t2_stake,
                }

                with st.spinner("Running 10 000 simulations…"):
                    pred, factors = predict_team_outcome(t1_dict, t2_dict, sport, score_type)

                st.success(f"**Prediction** – {pred}")
                st.subheader("Key factors")
                for f in factors:
                    st.write(f"- {f}")

    if clear:
        st.session_state.team_form_reset_key += 1
        st.rerun()


# -----------------------------------------------------------------
#   TAB 2 – PLAYER PROP BETS (unchanged)
# -----------------------------------------------------------------
with tab_props:
    st.header("Player Prop Bet Prediction")

    def add_player():
        st.session_state.players.append({})

    with st.form(key=f"p_form_{st.session_state.player_form_reset_key}"):
        for i in range(len(st.session_state.players)):
            with st.expander(f"Player {i+1}", expanded=True):
                c1, c2 = st.columns(2)
                with c1:
                    pn = st.text_input("Name *", key=f"pn_{i}_{st.session_state.player_form_reset_key}")
                    pos = st.selectbox(
                        "Position *",
                        [""] + SPORT_STATS[sport]["player_positions"],
                        key=f"pos_{i}_{st.session_state.player_form_reset_key}",
                    )
                    recent = st.text_area(
                        f"Recent Stats (e.g., {', '.join(SPORT_STATS[sport]['player_stats'])})",
                        key=f"rec_{i}_{st.session_state.player_form_reset_key}",
                    )
                with c2:
                    pt = st.selectbox(
                        "Prop type *",
                        [""] + SPORT_STATS[sport]["prop_types"],
                        key=f"pt_{i}_{st.session_state.player_form_reset_key}",
                    )
                    line_lbl = "Prop line/option * (e.g., 25.5, KO/TKO, Yes, 2.5, 3)"
                    line = st.text_input(line_lbl, key=f"line_{i}_{st.session_state.player_form_reset_key}")

                    if pt and ("Over/Under" in pt or "Total" in pt):
                        side = st.selectbox("Bet side *", ["Over", "Under"],
                                            key=f"side_{i}_{st.session_state.player_form_reset_key}")
                    else:
                        side = ""

                    odds = st.number_input("Odds *", -10_000, 10_000, 0,
                                           key=f"odds_{i}_{st.session_state.player_form_reset_key}")
                    opp = st.text_area(
                        "Opponent defence stats (e.g., Rank: 16, KO Vuln: 20)",
                        key=f"opp_{i}_{st.session_state.player_form_reset_key}",
                    )
                    inj = st.selectbox("Injury status", ["", "Healthy", "Questionable", "Out"],
                                      key=f"inj_{i}_{st.session_state.player_form_reset_key}")

                st.session_state.players[i] = {
                    "name": pn,
                    "position": pos,
                    "recent_stats": recent,
                    "prop_type": pt,
                    "over_under": line,
                    "bet_side": side,
                    "odds": odds,
                    "opp_defense": opp,
                    "injury_status": inj or "Healthy",
                }

        parlay_stake = st.number_input("Parlay stake $", 0.0, step=0.01,
                                       key=f"parlay_stake_{st.session_state.player_form_reset_key}")

        add_btn = st.form_submit_button("Add Player")
        pred_btn = st.form_submit_button("Predict Player Props")
        parlay_btn = st.form_submit_button("Predict Same‑Game Parlay")
        clear_btn = st.form_submit_button("Clear Players")

    if add_btn:
        add_player()
        st.rerun()

    if pred_btn:
        msgs = []
        for pl in st.session_state.players:
            if not pl["name"] or not pl["position"] or not pl["prop_type"] or not pl["over_under"]:
                st.error("All required fields (Name, Position, Prop Type, Line) must be filled.")
                break
            if ("Over/Under" in pl["prop_type"] or "Total" in pl["prop_type"]) and pl["bet_side"] not in ["Over", "Under"]:
                st.error(f"Select Over/Under side for player {pl['name']}.")
                break
            if pl["odds"] == 0:
                st.error(f"Odds required for player {pl['name']}.")
                break
            txt, _ = predict_player_prop(pl, sport)
            msgs.append(txt)
        if msgs:
            st.success("\n\n".join(msgs))

    if parlay_btn:
        if len(st.session_state.players) < 2:
            st.error("Add at least two players for a parlay.")
        else:
            good = []
            for pl in st.session_state.players:
                if not (pl["name"] and pl["prop_type"] and pl["position"] and pl["over_under"]):
                    continue
                if ("Over/Under" in pl["prop_type"] or "Total" in pl["prop_type"]) and pl["bet_side"] not in ["Over", "Under"]:
                    st.error(f"Select Over/Under side for player {pl['name']}.")
                    break
                if pl["odds"] == 0:
                    st.error(f"Odds missing for player {pl['name']}.")
                    break
                good.append(pl)
            else:
                pred, facts = predict_same_game_parlay(good, sport, parlay_stake)
                st.success(pred)
                for f in facts:
                    st.write(f"- {f}")

    if clear_btn:
        st.session_state.players = [{}]
        st.session_state.player_form_reset_key += 1
        st.rerun()


# -------------------------------------------------------------
# 11️⃣ Footer
# -------------------------------------------------------------
st.markdown("---")
st.write("Made with ❤️ by The Sports BetTracker Team | © 2025 103 Software Solutions LLC")
st.write("Powered by xAI Grok 3 | Statistical Algorithm")