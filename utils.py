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
    tstart: int = 0,
    tend: int | None = None,
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
    if tend is None:
        tend = len(data)
    season = data.loc[tstart:tend].copy()
    teams = set(season["HomeTeam"].unique()).union(set(season["AwayTeam"].unique()))
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


def adjust_standings(
    standings: pd.DataFrame, match: pd.Series, outcome: str
) -> pd.DataFrame:
    home_team = match["HomeTeam"]
    away_team = match["AwayTeam"]

    if outcome == "H":
        standings.loc[standings["Team"] == home_team, "Points"] += 3  # type: ignore
    elif outcome == "D":
        standings.loc[standings["Team"] == home_team, "Points"] += 1  # type: ignore
        standings.loc[standings["Team"] == away_team, "Points"] += 1  # type: ignore
    else:
        standings.loc[standings["Team"] == away_team, "Points"] += 3  # type: ignore
    return standings


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


def run_simulation(
    data,
    standings,
    t,
    t_k,
    tend: int | None = None,
) -> pd.DataFrame:
    """
    Run a simulation of the remaining matches in the season, exempting the match of interest

    Args:
    data: DataFrame containing the match data
    standings: DataFrame containing the standings
    t: int representing the current matchday
    t_k: int representing the match of interest
    tend: int representing the end of the season (default None)

    Returns:
    A DataFrame representing the updated standings
    """
    tend = tend or len(data)
    standings_copy = standings.copy()
    for i in itertools.chain(range(t + 1, t_k), range(t_k + 1, tend)):
        to_simulate = data.loc[i]
        outcome = generate_match_outcome(to_simulate)
        adjust_standings(standings_copy, to_simulate, outcome)
        logger.debug(f"Simulated match {i}")
    return standings_copy


def calculate_match_importance(
    data: pd.DataFrame,
    standings: pd.DataFrame | None,
    t_k: int,
    t: int,
    tstart=0,
    nruns=50,
    tend: int | None = None,
) -> tuple[float, float]:
    """
    Calculate the importance of a match based on the standings after simulation

    Args:
    data: DataFrame containing the match data
    standings: DataFrame containing the standings. If None, it will be calculated
    t_k: int representing the index of the match of interest
    t: int representing the current matchday
    tstart: int representing the start of the season (default 0)
    nruns: int representing the number of simulations to run (default 50)
    tend: int representing the end of the season (default None)

    Returns:
    A float representing the importance of the match
    """
    match = data.loc[t_k]
    if standings is None:
        standings = calculate_standings(data)
    else:
        standings = standings.copy()
    outcomes = []
    for _ in range(nruns):
        standings = run_simulation(data, standings, t, t_k, tend)
        outcomes.append(
            calculate_outcomes(standings, match["HomeTeam"], match["AwayTeam"]) # type: ignore
        )
        logger.debug(f"Tournament simulation for {t_k} done")
    outcomes = np.mean(outcomes, axis=0)
    home_importance = max([outcomes[j] - outcomes[j + 6] for j in range(3)])
    away_importance = max([outcomes[j + 6] - outcomes[j] for j in range(3, 6)])
    return home_importance, away_importance
