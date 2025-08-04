from utils import show_bar_chart
from typing import Dict, Union

def compare_projection(projected: float, line: float) -> str:
    """
    Compare projected value to line and return outcome.

    Args:
        projected (float): The projected value for the player.
        line (float): The line to compare against.

    Returns:
        str: "OVER", "UNDER", or "PUSH" based on comparison.
    """
    if not isinstance(projected, (int, float)) or not isinstance(line, (int, float)):
        raise TypeError("Projected and line must be numeric.")
    if projected > line:
        return "OVER"
    elif projected < line:
        return "UNDER"
    return "PUSH"

def predictor(player_data: Dict[str, Union[str, float]], metric: str) -> str:
    """
    Predict outcome for a player's performance and display a bar chart.

    Args:
        player_data (dict): Dictionary with 'name', 'projected', and 'line' keys.
        metric (str): The metric being predicted (e.g., 'Points', 'Yards').

    Returns:
        str: The predicted outcome ("OVER", "UNDER", "PUSH").

    Raises:
        KeyError: If required keys are missing from player_data.
        TypeError: If projected or line are not numeric.
    """
    required_keys = ["name", "projected", "line"]
    if not all(key in player_data for key in required_keys):
        raise KeyError(f"player_data must contain {required_keys} keys.")

    projected = player_data["projected"]
    line = player_data["line"]
    outcome = compare_projection(projected, line)
    show_bar_chart([projected], [metric], f"{player_data['name']}")
    return outcome

def basketball_predictor(player_data: Dict[str, Union[str, float]]) -> str:
    """Predict basketball player's points outcome."""
    return predictor(player_data, "Points")

def football_predictor(player_data: Dict[str, Union[str, float]]) -> str:
    """Predict football player's yards outcome."""
    return predictor(player_data, "Yards")

def baseball_predictor(player_data: Dict[str, Union[str, float]]) -> str:
    """Predict baseball player's home runs outcome."""
    return predictor(player_data, "Home Runs")

def hockey_predictor(player_data: Dict[str, Union[str, float]]) -> str:
    """Predict hockey player's goals outcome."""
    return predictor(player_data, "Goals")