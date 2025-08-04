import streamlit as st

def update_streak(sport, outcome):
    if "streaks" not in st.session_state:
        st.session_state.streaks = {}
    if outcome == "OVER":
        st.session_state.streaks[sport] = st.session_state.streaks.get(sport, 0) + 1
        if st.session_state.streaks[sport] == 3:
            st.balloons()
            st.success(f"🏆 Streak unlocked in {sport}: 3 OVERs in a row!")
    else:
        st.session_state.streaks[sport] = 0
