import random
from typing import List, Dict, Union

def get_avery_mood() -> str:
    """
    Select a random mood for Avery's commentary.

    Returns:
        str: One of 'playful', 'serious', 'mysterious', or 'spiritual'.
    """
    moods: List[str] = ["playful", "serious", "mysterious", "spiritual"]
    return random.choice(moods)

def avery_commentary(sport: str, outcome: str) -> str:
    """
    Generate Avery's commentary based on sport, outcome, and random mood.

    Args:
        sport (str): The sport (e.g., 'Basketball', 'Football', 'Baseball', 'Hockey').
        outcome (str): The prediction outcome ('OVER', 'UNDER', 'PUSH').

    Returns:
        str: A commentary string based on the sport, outcome, and Avery's mood.

    Raises:
        ValueError: If sport or outcome is invalid.
    """
    valid_sports = ["Basketball", "Football", "Baseball", "Hockey"]
    valid_outcomes = ["OVER", "UNDER", "PUSH"]

    # Normalize inputs
    sport = sport.capitalize()
    outcome = outcome.upper()

    # Validate inputs
    if sport not in valid_sports:
        raise ValueError(f"Invalid sport: {sport}. Must be one of {valid_sports}.")
    if outcome not in valid_outcomes:
        raise ValueError(f"Invalid outcome: {outcome}. Must be one of {valid_outcomes}.")

    mood = get_avery_mood()

    commentary: Dict[str, Dict[str, Dict[str, str]]] = {
        "Basketball": {
            "OVER": {
                "playful": "🔥 Avery laughs: 'Buckets on buckets—should’ve doubled the line!'",
                "serious": "⛹️‍♂️ Avery analyzes: 'Tempo, spacing, volume—this was no accident.'",
                "mysterious": "🌀 Avery whispers: 'They scored before they even stepped on the court…'",
                "spiritual": "🙏 Avery nods: 'That shot had faith behind it—fate met form.'"
            },
            "UNDER": {
                "playful": "😴 Avery yawns: 'Defense wins championships... and unders.'",
                "serious": "🔒 Avery states: 'Perimeter defense was suffocating. Every shot was contested.'",
                "mysterious": "🤫 Avery whispers: 'The rim had a lid on it tonight... a supernatural one.'",
                "spiritual": "🧘 Avery reflects: 'Patience and discipline triumphed over aggression.'"
            },
            "PUSH": {
                "playful": "🤷‍♂️ Avery shrugs: 'A perfect balance. The universe loves equilibrium.'",
                "serious": "⚖️ Avery notes: 'The line was sharp. A true coin flip.'",
                "mysterious": "🌀 Avery ponders: 'Did both teams conspire for this outcome?'",
                "spiritual": "☯️ Avery smiles: 'Yin and yang in perfect harmony.'"
            }
        },
        "Football": {
            "OVER": {
                "playful": "🏈 Avery smirks: 'Someone hit turbo mode—defenders still trying to catch up!'",
                "serious": "📊 Avery notes: 'Clean reads, high efficiency—this QB didn’t blink.'",
                "mysterious": "🌪️ Avery hints: 'The wind whispered routes before the snap…'",
                "spiritual": "📖 Avery reflects: 'Scripted like destiny. The playbook was divine.'"
            },
            "UNDER": {
                "playful": "🧱 Avery says: 'That defense was a brick wall. No one was getting through.'",
                "serious": "📉 Avery observes: 'Third-down conversion rate was abysmal. Offense couldn't find a rhythm.'",
                "mysterious": "👻 Avery murmurs: 'It's like the offense saw ghosts out there.'",
                "spiritual": "🙏 Avery nods: 'A testament to defensive preparation and faith in the system.'"
            },
            "PUSH": {
                "playful": "🤝 Avery chuckles: 'A gentleman's agreement. No winners, no losers.'",
                "serious": "📈 Avery points out: 'The spread was dead on. Oddsmakers earned their keep.'",
                "mysterious": "❓ Avery questions: 'Was this a statistical anomaly or something more?'",
                "spiritual": "🌌 Avery muses: 'The cosmos demanded balance on the gridiron today.'"
            }
        },
        "Baseball": {
            "OVER": {
                "playful": "⚾ Avery chuckles: 'Boom! That ball’s still traveling—maybe to another dimension.'",
                "serious": "🧠 Avery reports: 'Launch angle, barrel rate—textbook slugfest.'",
                "mysterious": "🌒 Avery murmurs: 'He swung like he’d already seen the pitch.'",
                "spiritual": "✝️ Avery smiles: 'Grace met grit. That home run had purpose.'"
            },
            "UNDER": {
                "playful": "😴 Avery sighs: 'A classic pitcher's duel. Hope you like strikeouts.'",
                "serious": "🎯 Avery analyzes: 'Command, control, and a devastating off-speed pitch. Unhittable.'",
                "mysterious": "✨ Avery whispers: 'The ball was dancing to a tune only the pitcher could hear.'",
                "spiritual": "🧘 Avery reflects: 'In the stillness of the mound, he found his power.'"
            },
            "PUSH": {
                "playful": "😐 Avery says: 'Right down the middle. Perfectly average.'",
                "serious": "📊 Avery notes: 'The run-line was a mathematical masterpiece.'",
                "mysterious": "🤔 Avery wonders: 'Did the baseball gods declare a truce?'",
                "spiritual": "⚖️ Avery observes: 'A game of inches, balanced on the edge of a razor.'"
            }
        },
        "Hockey": {
            "OVER": {
                "playful": "🥅 Avery shouts: 'Skates on fire! Goalie’s still trying to find his mask.'",
                "serious": "📉 Avery observes: 'Zone entries, puck movement, conversion rate—all elite.'",
                "mysterious": "❄️ Avery muses: 'The ice cracked before the goal happened… prophecy?'",
                "spiritual": "🕊️ Avery whispers: 'Even in chaos, there was intention. That goal was born in silence.'"
            },
            "UNDER": {
                "playful": "🧤 Avery says: 'That goalie was a human highlight reel. Nothing got past him.'",
                "serious": "🛡️ Avery states: 'Defensive structure was impeccable. They clogged every shooting lane.'",
                "mysterious": "🔮 Avery hints: 'He knew where the puck was going before the shot was even taken.'",
                "spiritual": "🙏 Avery nods: 'A performance of pure focus and unwavering belief.'"
            },
            "PUSH": {
                "playful": "🤷 Avery shrugs: 'A stalemate on ice. Guess we'll call it a draw.'",
                "serious": "📈 Avery analyzes: 'The puck-line was a work of art. Perfectly balanced.'",
                "mysterious": "🌀 Avery whispers: 'The puck seemed to have a mind of its own, seeking equilibrium.'",
                "spiritual": "☯️ Avery smiles: 'A dance of fire and ice, ending in perfect harmony.'"
            }
        }
    }

    return commentary[sport][outcome][mood]