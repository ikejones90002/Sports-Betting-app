# Same as we discussed previously, shortened here
import streamlit as st
import requests

st.title("🏈 LLM Sports Predictor (Local)")

team1_name = st.text_input("Team 1", "Team A")
team1_stats = st.text_area("Stats 1", "Points: 27\nYards: 360\nTurnovers: 1.2")

team2_name = st.text_input("Team 2", "Team B")
team2_stats = st.text_area("Stats 2", "Points: 21\nYards: 310\nTurnovers: 1.8")

if st.button("Predict Outcome"):
    prompt = f"""
You are a sports analyst. Predict the winner based on the stats.

### {team1_name}
{team1_stats}

### {team2_name}
{team2_stats}
"""

    response = requests.post(
        "http://ollama:11434/api/generate",  # uses container name
        json={"model": "llama3", "prompt": prompt, "stream": False}
    )
    result = response.json()
    st.success(result["response"])

