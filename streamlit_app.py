# streamlit_app.py
# Sports BetTracker – Single-file app with:
#  - negative spread convention (negative = Team1 favorite)
#  - auto-derivation of matchup spread from team inputs
#  - customizable period/inning/round templates
#  - SGP payout calculator (with push handling)
#  - preserved wizard sidebar layout and logo/footer
# Updated: 2025

import re
import streamlit as st  # type: ignore
import numpy as np      # type: ignore
import math

# -------------------------------------------------------------
# 0️⃣ GLOBAL CONFIG
# -------------------------------------------------------------
st.set_page_config(
    page_title="Sports BetTracker",
    layout="wide"
)

# If you have a logo file in the app folder uncomment the next line and provide the file.
# st.image("sports-bettracker-logo.png", width=200, caption="Sports BetTracker – Track the Action. Bet Smarter.")
st.title("🏀⚾🏒🏈👊 Sports BetTracker (Period Scoring, Spread Conventions & SGP Payouts)")

# -------------------------------------------------------------
# Sport definitions (extended with period labels and score types)
# -------------------------------------------------------------
SPORT_STATS = {
    "Football": {
        "team_stats": ["Points", "Yards", "Turnovers"],
        "player_positions": ["QB", "RB", "WR", "TE", "K"],
        "player_stats": ["Passing Yards", "Rushing Yards", "Receptions", "Touchdowns"],
        "periods": ["Q1", "Q2", "Q3", "Q4"],
        "score_types": ["Touchdown (6)", "Field Goal (3)", "Safety (2)", "PAT (1/2)"],
    },
    "Baseball": {
        "team_stats": ["Runs", "Hits", "Errors"],
        "player_positions": ["P", "C", "1B", "2B", "3B", "SS", "LF", "CF", "RF"],
        "player_stats": ["Batting Average", "Home Runs", "RBIs", "Pitcher ERA"],
        "periods": [f"Inning {i}" for i in range(1,10)],
        "score_types": ["Single", "Double", "Triple", "Home Run", "Run (earned/unearned)"],
    },
    "Basketball": {
        "team_stats": ["Points", "Rebounds", "Assists", "Field Goal %"],
        "player_positions": ["PG", "SG", "SF", "PF", "C"],
        "player_stats": ["Points", "Rebounds", "Assists", "Field Goal %"],
        "periods": ["Q1", "Q2", "Q3", "Q4"],
        "score_types": ["2-pt Field Goal", "3-pt Field Goal", "Free Throw"],
    },
    "Hockey": {
        "team_stats": ["Goals", "Shots on Goal", "Save %"],
        "player_positions": ["C", "LW", "RW", "D", "G"],
        "player_stats": ["Goals", "Assists", "Shots on Goal", "Save %"],
        "periods": ["P1", "P2", "P3"],
        "score_types": ["Even Strength Goal", "Power Play Goal", "Short-Handed Goal", "Empty Net"],
    },
    "MMA": {
        "team_stats": ["Strikes", "Takedowns", "Submissions"],
        "player_positions": ["Fighter"],
        "player_stats": ["Method of Victory", "Fight Goes the Distance", "Inside the Distance", "Round Betting", "Total Rounds Over/Under"],
        "periods": ["R1", "R2", "R3"],  # generic; user can edit number of rounds
        "score_types": ["KO/TKO", "Submission", "Decision", "DQ"],
    },
}

# -------------------------------------------------------------
# 1️⃣ STATE & HELPERS
# -------------------------------------------------------------
if "wizard_step" not in st.session_state:
    st.session_state.wizard_step = 1

if "team1" not in st.session_state:
    st.session_state.team1 = {}
if "team2" not in st.session_state:
    st.session_state.team2 = {}
if "players" not in st.session_state:
    st.session_state.players = [{}]

# Utility: American -> decimal conversion
def american_to_decimal(odds):
    try:
        odds = float(odds)
    except Exception:
        return 1.0
    if odds > 0:
        return (odds / 100.0) + 1.0
    else:
        return (100.0 / abs(odds)) + 1.0

# SGP payout: handles push legs (push legs are treated as multiplier 1)
def sgp_payout(odds_list, stake, pushes=None):
    """
    odds_list: list of American odds (ints/floats)
    stake: stake amount (float)
    pushes: list of booleans same length as odds_list; True means that leg pushed
    Returns: total_return, profit, effective_legs_count
    """
    if pushes is None:
        pushes = [False] * len(odds_list)
    decimal_total = 1.0
    effective_legs = 0
    for o, is_push in zip(odds_list, pushes):
        if is_push:
            # push -> this leg is removed from parlay (multiplier *= 1)
            continue
        dec = american_to_decimal(o)
        decimal_total *= dec
        effective_legs += 1

    # if all legs are pushes -> refund (total_return = stake)
    if effective_legs == 0:
        return stake, 0.0, effective_legs

    total_return = stake * decimal_total
    profit = total_return - stake
    return total_return, profit, effective_legs

# Data parsing helpers (kept minimal)
def parse_recent_performance(record):
    try:
        wins, losses = map(int, record.split("-"))
        return wins / (wins + losses) if (wins + losses) > 0 else 0.5
    except Exception:
        return 0.5

# -------------------------------------------------------------
# 2️⃣ SIDEBAR – WIZARD STEPS (with spread & period templates)
# -------------------------------------------------------------
with st.sidebar:
    st.markdown("### ⚙️ Settings")

    # ---------- STEP 1 – TEAM 1 ----------
    if st.session_state.wizard_step == 1:
        st.subheader("Team 1")
        st.session_state.team1["name"] = st.text_input("Name *", value=st.session_state.team1.get("name", ""))
        st.session_state.team1["home_away"] = st.selectbox("Home/Away *", ["", "Home", "Away", "Neutral"], index=3, key="t1_home_away")
        st.session_state.team1["rest_days"] = st.slider("Days Since Last Game", 0, 14, 0, key="t1_rest_days")

        st.write("**Stats**")
        col_stats = st.columns(len(SPORT_STATS.get(st.session_state.get("sport","Football"), {}).get("team_stats", ["Points"])))
        st.session_state.team1["stats"] = st.session_state.team1.get("stats", {})
        for i, stat in enumerate(SPORT_STATS["Football"]["team_stats"]):
            # default to Football layout here; actual sport-specific stats placed in Game Context
            with col_stats[i % max(1, len(col_stats))]:
                val = st.number_input(f"{stat}", min_value=0, value=st.session_state.team1["stats"].get(stat, 0), key=f"t1_{stat}")
                st.session_state.team1["stats"][stat] = val

        st.session_state.team1["recent"] = st.text_input("Recent Record (e.g., 3-2)", value=st.session_state.team1.get("recent", ""))
        st.session_state.team1["injuries"] = st.text_area("Key Injuries", value=st.session_state.team1.get("injuries", ""))

        # Betting fields - allow Point Spread to have its own spread_value field
        st.session_state.team1["bet_type"] = st.selectbox("Bet Type", ["", "Moneyline", "Point Spread", "Over/Under"], key="t1_bet_type")
        st.session_state.team1["odds"] = st.number_input("Odds (American) for bet", min_value=-10000, max_value=10000, value=st.session_state.team1.get("odds", 0), key="t1_odds")
        # separate numeric spread value field if they choose point spread
        if st.session_state.team1["bet_type"] == "Point Spread":
            st.session_state.team1["spread_value"] = st.number_input("Team1 Spread (negative = Team1 favorite)", value=st.session_state.team1.get("spread_value", 0.0), step=0.5, key="t1_spread_val")
        else:
            st.session_state.team1["spread_value"] = st.session_state.team1.get("spread_value", None)

        st.session_state.team1["stake"] = st.number_input("Stake $", min_value=0.0, value=st.session_state.team1.get("stake", 0.0), step=0.01, key="t1_stake")

        if st.button("Next →"):
            st.session_state.wizard_step = 2

    # ---------- STEP 2 – TEAM 2 ----------
    elif st.session_state.wizard_step == 2:
        st.subheader("Team 2")
        st.session_state.team2["name"] = st.text_input("Name *", value=st.session_state.team2.get("name", ""))
        st.session_state.team2["home_away"] = st.selectbox("Home/Away *", ["", "Home", "Away", "Neutral"], index=3, key="t2_home_away")
        st.session_state.team2["rest_days"] = st.slider("Days Since Last Game", 0, 14, 0, key="t2_rest_days")

        st.write("**Stats**")
        col_stats2 = st.columns(len(SPORT_STATS.get(st.session_state.get("sport","Football"), {}).get("team_stats", ["Points"])))
        st.session_state.team2["stats"] = st.session_state.team2.get("stats", {})
        for i, stat in enumerate(SPORT_STATS["Football"]["team_stats"]):
            with col_stats2[i % max(1, len(col_stats2))]:
                val = st.number_input(f"{stat}", min_value=0, value=st.session_state.team2["stats"].get(stat, 0), key=f"t2_{stat}")
                st.session_state.team2["stats"][stat] = val

        st.session_state.team2["recent"] = st.text_input("Recent Record", value=st.session_state.team2.get("recent", ""))
        st.session_state.team2["injuries"] = st.text_area("Key Injuries", value=st.session_state.team2.get("injuries", ""))

        st.session_state.team2["bet_type"] = st.selectbox("Bet Type", ["", "Moneyline", "Point Spread", "Over/Under"], key="t2_bet_type")
        st.session_state.team2["odds"] = st.number_input("Odds (American) for bet", min_value=-10000, max_value=10000, value=st.session_state.team2.get("odds", 0), key="t2_odds")
        if st.session_state.team2["bet_type"] == "Point Spread":
            # For team2, we store spread_value but auto-convention: team2's spread_value should be positive
            # When auto-deriving matchup spread we will invert sign appropriately.
            st.session_state.team2["spread_value"] = st.number_input("Team2 Spread (positive -> Team2 favored amount)", value=st.session_state.team2.get("spread_value", 0.0), step=0.5, key="t2_spread_val")
        else:
            st.session_state.team2["spread_value"] = st.session_state.team2.get("spread_value", None)

        st.session_state.team2["stake"] = st.number_input("Stake $", min_value=0.0, value=st.session_state.team2.get("stake", 0.0), step=0.01, key="t2_stake")

        if st.button("Next →"):
            st.session_state.wizard_step = 3

    # ---------- STEP 3 – GAME CONTEXT (with auto-spread derivation and period templates) ----------
    elif st.session_state.wizard_step == 3:
        st.subheader("Game Context")
        # Sport selector (moved here so we can adapt period templates)
        st.session_state.sport = st.selectbox(
            "Select Sport", ["Football", "Baseball", "Basketball", "Hockey", "MMA"],
            index=["Football", "Baseball", "Basketball", "Hockey", "MMA"].index(st.session_state.get("sport","Football"))
        )
        sport = st.session_state.sport

        st.session_state.game_type = st.selectbox(
            "Game Type", ["Regular Season", "Playoffs", "Preseason"], index=0, key="game_type"
        )
        st.session_state.weather = st.selectbox(
            "Weather", ["", "Clear", "Rain", "Snow", "Windy"], index=0, key="weather"
        )
        st.session_state.prediction_confidence = st.slider(
            "Confidence Threshold (%)", 50, 100, 75, key="confidence_slider"
        )

        # ---------- Auto-derive matchup spread ----------
        st.markdown("**Point Spread (matchup)**")
        st.markdown("""
        Convention in this app: **negative = Team 1 favorite** (e.g., -3.5 means Team1 must win by 4+ to cover).
        If either team specified a Point Spread in their team inputs, that value will auto-fill here.
        Team1's spread_value takes priority if provided.
        """)

        # Determine auto_spread:
        auto_spread = st.session_state.get("matchup_spread", 0.0)
        # Team1 provided a spread_value (should be negative for favorite by convention)
        t1_sv = st.session_state.team1.get("spread_value", None)
        if st.session_state.team1.get("bet_type") == "Point Spread" and t1_sv is not None:
            # accept whatever user entered for Team1 as canonical matchup spread
            auto_spread = float(t1_sv)
        else:
            # fallback: if team2 provided a spread (positive by its field), invert sign to get Team1 convention
            t2_sv = st.session_state.team2.get("spread_value", None)
            if st.session_state.team2.get("bet_type") == "Point Spread" and t2_sv is not None:
                # user-entered Team2 spread_value presumably indicates Team2 is favored by +X,
                # but our convention wants negative if Team1 favored, so invert:
                # e.g., team2 entered 3.5 meaning Team2 favored by 3.5 -> matchup_spread should be +3.5
                # but to represent Team1 favorite negative sign, we set matchup_spread = +3.5 (Team1 is underdog)
                auto_spread = float(-t2_sv)  # invert sign so that positive means Team1 underdog
        # Display and allow override
        st.session_state.matchup_spread = st.number_input(
            "Matchup Spread (negative = Team1 favorite)",
            value=st.session_state.get("matchup_spread", auto_spread),
            step=0.5,
            key="matchup_spread_input"
        )

        # ---------- Period scoring templates (customizable) ----------
        st.markdown("**Per-Period Scoring (customizable)**")
        periods_default = SPORT_STATS[sport]["periods"]
        # Default count to the default template length, stored in session if previously modified
        default_count = st.session_state.get("custom_period_count", len(periods_default))
        custom_period_count = st.number_input(
            "Number of periods/innings/rounds", 
            min_value=1, 
            max_value=20, 
            value=default_count,
            key="custom_period_count_input"
        )
        st.session_state.custom_period_count = custom_period_count

        # Allow custom label template or auto-generated labels
        label_mode = st.radio("Period label mode", ["Auto-generated labels", "Custom labels"], index=0, key="label_mode")

        if label_mode == "Auto-generated labels":
            if sport == "Baseball":
                periods = [f"Inning {i}" for i in range(1, int(custom_period_count) + 1)]
            elif sport == "Basketball":
                # If count > 4, treat extras as OT
                if custom_period_count <= 4:
                    periods = [f"Q{i}" for i in range(1, int(custom_period_count) + 1)]
                else:
                    # Q1..Q4 + OT1..OTn
                    periods = [f"Q{i}" for i in range(1, 5)]
                    ot_count = int(custom_period_count) - 4
                    periods += [f"OT{j}" for j in range(1, ot_count + 1)]
            elif sport == "Football":
                periods = [f"Q{i}" for i in range(1, int(custom_period_count) + 1)]
            elif sport == "Hockey":
                # Hockey usual 3 periods; extras call them OT1...
                if custom_period_count <= 3:
                    periods = [f"P{i}" for i in range(1, int(custom_period_count) + 1)]
                else:
                    periods = [f"P{i}" for i in range(1, 4)] + [f"OT{j}" for j in range(1, int(custom_period_count) - 3 + 1)]
            elif sport == "MMA":
                periods = [f"R{i}" for i in range(1, int(custom_period_count) + 1)]
            else:
                periods = [f"P{i}" for i in range(1, int(custom_period_count) + 1)]
        else:
            # Custom labels input
            custom_labels_text = st.text_area("Enter labels separated by commas (e.g., Q1,Q2,Q3,Q4,OT1)", value=",".join(periods_default))
            labels = [lab.strip() for lab in custom_labels_text.split(",") if lab.strip()]
            # If user provided fewer labels than count, auto-fill remaining
            if len(labels) < int(custom_period_count):
                extra = int(custom_period_count) - len(labels)
                labels += [f"P{i}" for i in range(1, extra + 1)]
            periods = labels[:int(custom_period_count)]

        # store periods in session for later UI
        st.session_state.periods = periods

        # Period scores stored as nested dict in session_state
        if "period_scores" not in st.session_state:
            st.session_state.period_scores = {"team1": {}, "team2": {}, "types": {}}

        st.session_state.period_scores = st.session_state.period_scores  # ensure exists
        # Render per-period inputs
        for p in periods:
            cols = st.columns(3)
            with cols[0]:
                v1 = st.number_input(f"Team1 {p}", min_value=0, value=int(st.session_state.period_scores["team1"].get(p, 0)), key=f"ps_t1_{p}")
                st.session_state.period_scores["team1"][p] = int(v1)
            with cols[1]:
                v2 = st.number_input(f"Team2 {p}", min_value=0, value=int(st.session_state.period_scores["team2"].get(p, 0)), key=f"ps_t2_{p}")
                st.session_state.period_scores["team2"][p] = int(v2)
            with cols[2]:
                st.session_state.period_scores["types"][p] = st.selectbox(
                    f"{p} Scoring Type",
                    [""] + SPORT_STATS[sport]["score_types"],
                    index=0,
                    key=f"ps_type_{p}"
                )

        st.caption("⚠️ If a *Stake* is entered for either team, both Bet Type and Odds should be provided in team forms above.")
        if st.button("Predict Game Outcome"):
            st.session_state.run_prediction = True

    # RESET button at bottom
    if st.button("Clear All Inputs"):
        st.session_state.wizard_step = 1
        st.session_state.team1 = {}
        st.session_state.team2 = {}
        st.session_state.period_scores = {"team1": {}, "team2": {}, "types": {}}
        st.session_state.players = [{}]
        st.session_state.matchup_spread = 0.0
        st.experimental_rerun()

# -------------------------------------------------------------
# 3️⃣ MAIN PANEL – RESULTS (after prediction)
# -------------------------------------------------------------
def validate_team_data(team, team_name):
    if not team.get("name"):
        st.error(f"Missing name for {team_name}.")
        return False
    if team.get("stake", 0) > 0 and (team.get("bet_type", "") == "" or team.get("odds", 0) == 0):
        st.error(f"Stake entered for {team_name} but Bet Type / Odds missing.")
        return False
    return True

def predict_team_outcome(team1, team2, sport, period_scores, matchup_spread, periods):
    """
    Produces human-readable prediction including:
     - per-period breakdown with scoring type
     - half breakdowns for applicable sports (Football, Basketball)
     - first to score and type of score
     - final score totals
     - winning margin
     - whether point spread was covered (convention: negative = Team1 favorite)
    """
    # Sum period totals if available
    team1_total = sum(period_scores.get("team1", {}).get(p, 0) for p in periods)
    team2_total = sum(period_scores.get("team2", {}).get(p, 0) for p in periods)

    # Fallback if totals are zero (no input): use team 'Points' or 'Runs' if available
    if team1_total == 0 and team2_total == 0:
        t1_pts = team1.get("stats", {}).get("Points", 0) or team1.get("stats", {}).get("Runs", 0)
        t2_pts = team2.get("stats", {}).get("Points", 0) or team2.get("stats", {}).get("Runs", 0)
        team1_total = int(t1_pts)
        team2_total = int(t2_pts)

    # Decide winner
    if team1_total > team2_total:
        winner = team1.get("name", "Team 1")
    elif team2_total > team1_total:
        winner = team2.get("name", "Team 2")
    else:
        winner = "Draw / Push"

    # Winning margin
    margin = abs(team1_total - team2_total)

    # Spread coverage using convention: negative means Team1 favored.
    # Team1 covers if (team1_total + spread) > team2_total
    cover_text = "No spread provided."
    covered = None
    try:
        sp = float(matchup_spread)
        team1_adj = team1_total + sp
        if team1_adj > team2_total:
            covered = team1.get("name", "Team 1")
            cover_text = f"{covered} covers the spread (Team1 {team1_total} + spread {sp} => {team1_adj:.1f} vs Team2 {team2_total})."
        elif team1_adj < team2_total:
            covered = team2.get("name", "Team 2")
            cover_text = f"{covered} covers the spread (Team1 {team1_total} + spread {sp} => {team1_adj:.1f} vs Team2 {team2_total})."
        else:
            covered = "Push"
            cover_text = f"Push — spread results in a tie (Team1 adjusted {team1_adj:.1f} = Team2 {team2_total})."
    except Exception:
        cover_text = "Spread not interpretable."

    # Per-period breakdown lines
    breakdown_lines = []
    for p in periods:
        s1 = period_scores.get("team1", {}).get(p, 0)
        s2 = period_scores.get("team2", {}).get(p, 0)
        stype = period_scores.get("types", {}).get(p, "")
        if stype:
            breakdown_lines.append(f"{p}: {team1.get('name','T1')} {s1} — {team2.get('name','T2')} {s2}  ({stype})")
        else:
            breakdown_lines.append(f"{p}: {team1.get('name','T1')} {s1} — {team2.get('name','T2')} {s2}")

    # Half breakdowns for Football and Basketball (assuming 4 quarters)
    half_lines = []
    if sport in ["Football", "Basketball"] and len(periods) >= 4 and all("Q" in p for p in periods[:4]):
        h1_t1 = period_scores["team1"].get(periods[0], 0) + period_scores["team1"].get(periods[1], 0)
        h1_t2 = period_scores["team2"].get(periods[0], 0) + period_scores["team2"].get(periods[1], 0)
        h2_t1 = period_scores["team1"].get(periods[2], 0) + period_scores["team1"].get(periods[3], 0)
        h2_t2 = period_scores["team2"].get(periods[2], 0) + period_scores["team2"].get(periods[3], 0)
        half_lines = [
            f"H1: {team1.get('name','T1')} {h1_t1} — {team2.get('name','T2')} {h1_t2}",
            f"H2: {team1.get('name','T1')} {h2_t1} — {team2.get('name','T2')} {h2_t2}"
        ]
        # If more periods (OT), add them as is

    # First to score and type
    first_score_text = "No scoring entered."
    for p in periods:
        s1 = period_scores.get("team1", {}).get(p, 0)
        s2 = period_scores.get("team2", {}).get(p, 0)
        if s1 > 0 or s2 > 0:
            stype = period_scores.get("types", {}).get(p, "Unknown type")
            if s1 > 0 and s2 == 0:
                first_scorer = team1.get('name', 'Team 1')
            elif s2 > 0 and s1 == 0:
                first_scorer = team2.get('name', 'Team 2')
            else:
                first_scorer = "Both teams"
            first_score_text = f"First to score: {first_scorer} in {p} ({stype})"
            break

    prediction = f"{team1.get('name','Team 1')} {team1_total} — {team2.get('name','Team 2')} {team2_total} | Winner: {winner} | Margin: {margin}"
    factors = [
        first_score_text,
        "Per-period breakdown:"
    ] + breakdown_lines
    if half_lines:
        factors += ["Half breakdowns:"] + half_lines
    factors += [
        f"Spread (negative = Team1 favored): {matchup_spread}",
        cover_text,
    ]
    return prediction, factors

# Run prediction when requested
if st.session_state.get("run_prediction"):
    sport = st.session_state.get("sport", "Football")
    if validate_team_data(st.session_state.team1, "Team 1") and validate_team_data(st.session_state.team2, "Team 2"):
        with st.spinner("Running prediction & assembling period breakdown..."):
            try:
                prediction, factors = predict_team_outcome(
                    st.session_state.team1,
                    st.session_state.team2,
                    sport,
                    st.session_state.get("period_scores", {"team1": {}, "team2": {}, "types": {}}),
                    st.session_state.get("matchup_spread", 0.0),
                    st.session_state.get("periods", SPORT_STATS[sport]["periods"])
                )
            except Exception as e:
                st.error(f"Prediction failed: {e}")
                st.session_state.run_prediction = False
            else:
                st.success("**Predicted Match Result (with Period Breakdown)**")
                st.subheader(prediction)

                st.markdown("### 🧾 Per-Period Breakdown")
                for line in factors:
                    st.write(f"• {line}")

                # Show a small summary metric
                margin_match = re.search(r'Margin: (\d+)', prediction)
                margin_value = f"{abs(int(margin_match.group(1)))} pts" if margin_match else "N/A"
                st.metric(label="Margin", value=margin_value)

        st.session_state.run_prediction = False  # reset flag

# -------------------------------------------------------------
# 4️⃣ PLAYER PROP AREA (sidebar-driven) — with SGP payout wiring
# -------------------------------------------------------------
with st.sidebar:
    st.markdown("## 🏀 Player Prop Bets")

    if st.button("Add Player"):
        st.session_state.players.append({})
        st.experimental_rerun()

    # Render player prop inputs
    for idx, _ in enumerate(st.session_state.players):
        p = st.session_state.players[idx]
        with st.expander(f"Player {idx+1}", expanded=(idx == len(st.session_state.players) - 1)):
            p["name"] = st.text_input("Name *", value=p.get("name", ""), key=f"p{idx}_name")
            p["position"] = st.selectbox("Position *", [""] + SPORT_STATS[st.session_state.get("sport","Football")]["player_positions"], key=f"p{idx}_pos")
            p["recent_stats"] = st.text_area(
                f"Recent Stats (e.g., {', '.join(SPORT_STATS[st.session_state.get('sport','Football')]['player_stats'])})",
                value=p.get("recent_stats", ""),
                key=f"p{idx}_recent_stats",
            )
            p["prop_type"] = st.selectbox(
                "Prop Type *", [""] + SPORT_STATS[st.session_state.get("sport","Football")]["player_stats"], key=f"p{idx}_prop_type"
            )
            p["over_under"] = st.text_input("Over/Under or Option (e.g., 25.5 or KO/TKO or Yes/No)", value=p.get("over_under",""), key=f"p{idx}_ou")
            p["odds"] = st.number_input("Odds * (American)", min_value=-10000, max_value=10000, value=p.get("odds", 0), key=f"p{idx}_odds")
            p["opp_defense"] = st.text_input("Opposing Defense (Rank: X)", value=p.get("opp_defense","Rank: 0"), key=f"p{idx}_def")
            p["injury_status"] = st.selectbox("Injury Status", ["", "Healthy", "Questionable", "Out"], key=f"p{idx}_injury")
            # Allow the user to mark this leg as a push (for manual simulation / in-play handling)
            p["is_push"] = st.checkbox("Mark as Push (leg refunded)", value=p.get("is_push", False), key=f"p{idx}_push")

    st.markdown("## 🧩 Same-Game Parlay")
    parlay_stake = st.number_input("Parlay Stake $", min_value=0.0, value=st.session_state.get("parlay_stake", 0.0), step=0.01, key="parlay_stake_input")
    st.session_state.parlay_stake = parlay_stake

    if st.button("Predict Player Props"):
        all_valid = True
        results = []
        for i, pr in enumerate(st.session_state.players):
            if not pr.get("name") or not pr.get("prop_type") or not pr.get("over_under"):
                st.error(f"Missing required field(s) for player {i+1}.")
                all_valid = False
                break
            if pr.get("odds", 0) == 0:
                st.error(f"Odds required for player {i+1}.")
                all_valid = False
                break
            # Placeholder: simple predicted value text
            results.append(f"{pr.get('name')} — {pr.get('prop_type')} {pr.get('over_under')} @ {pr.get('odds')} (placeholder prediction)")
        if all_valid:
            st.success("🎯 Player Prop Predictions")
            st.write("\n\n".join(results))

    if st.button("Predict Same-Game Parlay"):
        if len(st.session_state.players) < 2:
            st.error("Add at least two players for a parlay.")
        else:
            ok = True
            valid_players = []
            for i, pr in enumerate(st.session_state.players):
                if not pr.get("name") or not pr.get("over_under") or pr.get("odds", 0) == 0:
                    st.error(f"Player {i+1} missing required data.")
                    ok = False
                    break
                valid_players.append(pr)
            if ok:
                # Compute payout (handles pushes)
                odds_list = [vp["odds"] for vp in valid_players]
                pushes = [bool(vp.get("is_push", False)) for vp in valid_players]
                total_return, profit, effective_legs = sgp_payout(odds_list, parlay_stake, pushes)
                if parlay_stake <= 0:
                    st.warning("Stake not entered – calculation uses zero stake. Enter a stake to see dollar returns.")
                st.success("🎉 Same-Game Parlay Payout")
                st.write(f"Legs entered: {len(valid_players)} — Effective legs (non-push): {effective_legs}")
                st.metric(label="Total Return", value=f"${total_return:.2f}")
                st.metric(label="Profit", value=f"${profit:.2f}")
                # Show breakdown of legs and which pushed
                st.markdown("#### Parlay Legs")
                for i, vp in enumerate(valid_players):
                    push_text = " (PUSH)" if pushes[i] else ""
                    st.write(f"- {vp['name']}: {vp['prop_type']} {vp['over_under']} @ {vp['odds']}{push_text}")

    if st.button("Clear Players"):
        st.session_state.players = [{}]
        st.experimental_rerun()

# -------------------------------------------------------------
# 5️⃣ FOOTER
# -------------------------------------------------------------
st.markdown("---")
st.write("Made with ❤️ by The Sports BetTracker Team | © 2025 103 Software Solutions LLC")
st.write("Notes: Spread convention in this app is negative = Team1 favorite. Team1 spread entries take priority for the matchup spread. SGP payout treats marked push legs as removed from the parlay (refund for that leg).")