import streamlit as st
from typing import Dict, Optional

def update_streak(sport: str, outcome: str, reset_on_push: bool = True) -> Optional[Dict[str, int]]:
    """
    Update and track prediction streaks for a sport in a Streamlit app.

    Args:
        sport (str): The sport (e.g., 'Basketball', 'Football', 'Baseball', 'Hockey').
        outcome (str): The prediction outcome ('OVER', 'UNDER', 'PUSH').
        reset_on_push (bool, optional): Whether to reset streak on 'PUSH'. Defaults to True.

    Returns:
        Optional[Dict[str, int]]: Current streaks dictionary or None if no update.

    Raises:
        ValueError: If sport or outcome is invalid.
    """
    # Validate inputs
    valid_sports = ["Basketball", "Football", "Baseball", "Hockey"]
    valid_outcomes = ["OVER", "UNDER", "PUSH"]
    sport = sport.capitalize()
    outcome = outcome.upper()

    if sport not in valid_sports:
        raise ValueError(f"Invalid sport: {sport}. Must be one of {valid_sports}.")
    if outcome not in valid_outcomes:
        raise ValueError(f"Invalid outcome: {outcome}. Must be one of {valid_outcomes}.")

    # Initialize streaks in session state
    if "streaks" not in st.session_state:
        st.session_state.streaks = {}

    # Update streak
    if outcome == "OVER":
        st.session_state.streaks[sport] = st.session_state.streaks.get(sport, 0) + 1
        if st.session_state.streaks[sport] == 3:
            st.balloons()
            st.success(f"🏆 Streak unlocked in {sport}: 3 OVERs in a row!")
    elif outcome == "UNDER" or (outcome == "PUSH" and reset_on_push):
        st.session_state.streaks[sport] = 0

    return st.session_state.streaks