import streamlit as st
import matplotlib.pyplot as plt

def show_bar_chart(values, labels, title):
    fig, ax = plt.subplots()
    ax.bar(labels, values, color="skyblue")
    ax.set_title(title)
    ax.set_ylabel("Projected Value")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    fig.patch.set_alpha(0.0)
    st.pyplot(fig)
