import streamlit as st
import matplotlib.pyplot as plt
from typing import List, Union

def show_bar_chart(values: List[Union[int, float]], labels: List[str], title: str) -> None:
    """
    Display a bar chart in a Streamlit app using Matplotlib.

    Args:
        values (List[Union[int, float]]): List of numeric values to plot.
        labels (List[str]): List of labels for the x-axis.
        title (str): Title of the chart.

    Raises:
        ValueError: If values or labels are empty, or if their lengths don't match.
        TypeError: If values contains non-numeric elements.
    """
    if not values or not labels:
        raise ValueError("Values and labels lists must not be empty.")
    if len(values) != len(labels):
        raise ValueError("Values and labels must have the same length.")
    if not all(isinstance(v, (int, float)) for v in values):
        raise TypeError("All values must be numeric.")

    try:
        fig, ax = plt.subplots()
        ax.bar(labels, values, color="skyblue")
        ax.set_title(title)
        ax.set_ylabel("Projected Value")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        st.pyplot(fig)
    except Exception as e:
        st.error(f"Error rendering chart: {e}")
    finally:
        plt.close(fig)  # Close figure to free memory