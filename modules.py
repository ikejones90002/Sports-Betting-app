import streamlit as st
from utils import show_bar_chart
from data import nba_players, nfl_players, mlb_players, nhl_players

def compare_projection(projected, line):
    if projected > line:
        st.success("✅ OVER Projected")
        return "OVER"
    elif projected < line:
        st.error("❌ UNDER Projected")
        return "UNDER"
    else:
        st.warning("🟨 Push")
        return "PUSH"

def basketball_predictor():
    st.subheader("🏀 Basketball Predictor")
    player = st.selectbox("Player", nba_players)
    projected = st.slider("Projected Points", 0, 50)
    line = st.slider("Set Line", 0, 50)
    outcome = compare_projection(projected, line)
    show_bar_chart([projected], ["Points"], f"{player}")
    return outcome

def football_predictor():
    st.subheader("🏈 Football Predictor")
    player = st.selectbox("Player", nfl_players)
    projected = st.slider("Projected Yards", 0, 400)
    line = st.slider("Set Line", 0, 400)
    outcome = compare_projection(projected, line)
    show_bar_chart([projected], ["Yards"], f"{player}")
    return outcome

def baseball_predictor():
    st.subheader("⚾ Baseball Predictor")
    player = st.selectbox("Player", mlb_players)
    projected = st.slider("Projected Home Runs", 0, 5)
    line = st.slider("Set Line", 0, 5)
    outcome = compare_projection(projected, line)
    show_bar_chart([projected], ["Home Runs"], f"{player}")
    return outcome

def hockey_predictor():
    st.subheader("🏒 Hockey Predictor")
    player = st.selectbox("Player", nhl_players)
    projected = st.slider("Projected Goals", 0, 5)
    line = st.slider("Set Line", 0, 5)
    outcome = compare_projection(projected, line)
    show_bar_chart([projected], ["Goals"], f"{player}")
    return outcome

def compare_projection(projected, line):
    if projected > line:
        st.success("✅ OVER Projected")
        return "OVER"
    elif projected < line:
        st.error("❌ UNDER Projected")
        return "UNDER"
    else:
        st.warning("🟨 Push")
        return "PUSH"
