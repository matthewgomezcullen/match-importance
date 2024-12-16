import itertools
import numpy as np
import pandas as pd
from scipy.stats import norm
from typing import cast
import logging


logger = logging.getLogger(__name__)


def _get_home_points(row):
    if row["FTR"] == "H":
        return 3
    elif row["FTR"] == "A":
        return 0
    else:
        return 1


def _get_away_points(row):
    if row["FTR"] == "A":
        return 3
    elif row["FTR"] == "H":
        return 0
    else:
        return 1


def calculate_standings(
    data: pd.DataFrame,
    start: int = 0,
    end: int | None = None,
    ignore: int | None = None,
) -> pd.DataFrame:
    """
    Calculate the standings of a given season

    Args:
    data: DataFrame containing the match data
    start: int representing the start of the season (default 0)
    end: int representing the end of the season (default None)
    ignore: int representing the match to ignore (default None)

    Returns:
    A DataFrame containing the standings
    """
    if end is None:
        end = len(data)
    season = data.loc[start:end].copy()
    teams = set(season["HomeTeam"]) | set(season["AwayTeam"])
    if ignore:
        season.drop(ignore, inplace=True)
    season["HomePoints"] = season.apply(_get_home_points, axis=1)
    season["AwayPoints"] = season.apply(_get_away_points, axis=1)
    standings = (
        season.groupby("HomeTeam")["HomePoints"].sum().reindex(teams, fill_value=0) # type: ignore
        + season.groupby("AwayTeam")["AwayPoints"].sum().reindex(teams, fill_value=0) # type: ignore
    )
    standings = standings.sort_values(ascending=False).reset_index()
    standings.columns = ["Team", "Points"]
    return standings


def generate_match_outcome(row):
    p = row["TeamEloWinProb"] - 0.5
    p_1 = norm.cdf(p - 0.5)
    p_0 = norm.cdf(p + 0.5) - norm.cdf(p - 0.5)
    p_neg1 = norm.cdf(-0.5 - p)
    probs = [p_neg1, p_0, p_1]
    outcome = np.random.choice([-1, 0, 1], p=probs)
    if outcome == 1:
        return "H"
    if outcome == -1:
        return "A"
    return "D"


def calculate_outcomes(
    standings, home: str, away: str
) -> tuple[int, int, int, int, int, int, int, int, int, int, int, int]:
    """
    Calculate the outcomes of a tournamnet based on the standings

    Args:
    standings: DataFrame containing the standings
    home: str representing the home team
    away: str representing the away team

    Returns:
    A tuple representing if the home team won, relegated, and qualified for the champions league
    1 if true, 0 if false
    """
    outcomes = ()
    standings.loc[standings["Team"] == home, "Points"] += 3
    standings = standings.sort_values(by=["Points"], ascending=False).reset_index(
        drop=True
    )
    home_position = standings[standings["Team"] == home].index[0] + 1
    away_position = standings[standings["Team"] == away].index[0] + 1
    outcomes += (
        1 if home_position == 1 else 0,
        1 if home_position > 16 else 0,
        1 if home_position < 5 else 0,
        1 if away_position == 1 else 0,
        1 if away_position > 16 else 0,
        1 if away_position < 5 else 0,
    )
    standings.loc[standings["Team"] == home, "Points"] -= 3
    standings.loc[standings["Team"] == away, "Points"] += 3
    standings = standings.sort_values(by=["Points"], ascending=False).reset_index(
        drop=True
    )
    home_position = standings[standings["Team"] == home].index[0] + 1
    away_position = standings[standings["Team"] == away].index[0] + 1
    outcomes += (
        1 if home_position == 1 else 0,
        1 if home_position > 16 else 0,
        1 if home_position < 5 else 0,
        1 if away_position == 1 else 0,
        1 if away_position > 16 else 0,
        1 if away_position < 5 else 0,
    )
    return outcomes



def generate_simulations(
    matches: pd.DataFrame,
    start: int=0,
    end: int | None=None,
    nruns: int=1000,
    current: int=-1
) -> list[pd.DataFrame]:
    """
    Simulate seasons

    Args:
    matches: DataFrame containing the match data
    start: int representing the start of the season (default 0)
    end: int representing the end of the season (default None)
    nruns: int representing the number of simulations to run per match (default 1000)
    current: int representing the current matchday (default -1)

    Returns:
    A DataFrame containing the simulated seasons
    """
    simulations = []
    if end is None:
        end = len(matches)
    for _ in range(nruns):
        simulation = matches.loc[start:end].copy()
        simulation.loc[current+1:, "FTR"] = simulation.apply(generate_match_outcome, axis=1)
        simulations.append(simulation)
    return simulations