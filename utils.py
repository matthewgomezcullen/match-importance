import pandas as pd
import numpy as np
from typing import cast
from scipy.stats import norm


def calculate_standings(data: pd.DataFrame, tstart: int, t):
    # standings = pd.DataFrame(columns=["Team", "Points", "Goal Difference"])
    standings = pd.DataFrame(columns=["Team", "Points"])
    standings["Team"] = data["HomeTeam"].unique()
    standings["Points"] = 0
    # standings["Goal Difference"] = 0

    for i in range(tstart, t):
        home_team = data.loc[i, "HomeTeam"]
        away_team = data.loc[i, "AwayTeam"]
        home_goals = cast(int, data.loc[i, "FTHG"])
        away_goals = cast(int, data.loc[i, "FTAG"])

        if home_goals > away_goals:
            standings.loc[standings["Team"] == home_team, "Points"] += 3
        elif home_goals < away_goals:
            standings.loc[standings["Team"] == away_team, "Points"] += 3
        else:
            standings.loc[standings["Team"] == home_team, "Points"] += 1
            standings.loc[standings["Team"] == away_team, "Points"] += 1

        # standings.loc[standings["Team"] == home_team, "Goal Difference"] += (
        #     home_goals - away_goals
        # )
        # standings.loc[standings["Team"] == away_team, "Goal Difference"] += (
        #     away_goals - home_goals
        # )

    return standings.sort_values(
        by=["Points"], ascending=False
    ).reset_index(drop=True)


def generate_match_outcome(c_1: float, c_neg1: float, p: float):
    p_1 = norm.cdf(p - c_1)
    p_0 = norm.cdf(p - c_neg1) - norm.cdf(p - c_1)
    p_neg1 = norm.cdf(c_neg1 - p)

    probs = [p_neg1, p_0, p_1]

    outcome = np.random.choice([-1, 0, 1], p=probs)
    return outcome
