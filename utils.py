import itertools
import numpy as np
import pandas as pd
from scipy.stats import norm
from typing import cast
import logging


logger = logging.getLogger(__name__)


def _initialise_standings(data: pd.DataFrame) -> pd.DataFrame:
    standings = pd.DataFrame(columns=["Team", "Points"])
    standings["Team"] = data["HomeTeam"].unique()
    standings["Points"] = 0
    return standings


def calculate_standings(data: pd.DataFrame, tstart: int, t):
    """
    Calculate the standings after a given matchday

    Args:
    data: DataFrame containing the match data
    tstart: int representing the start of the season
    t: int representing the current matchday

    Returns:
    A DataFrame representing the standings
    """
    standings = _initialise_standings(data)

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

    return standings.sort_values(by=["Points"], ascending=False).reset_index(drop=True)


def _generate_match_outcome(c_1: float, c_neg1: float, p: float):
    p_1 = norm.cdf(p - c_1)
    p_0 = norm.cdf(p - c_neg1) - norm.cdf(p - c_1)
    p_neg1 = norm.cdf(c_neg1 - p)
    probs = [p_neg1, p_0, p_1]
    outcome = np.random.choice([-1, 0, 1], p=probs)
    return outcome


def _adjust_standings(standings, match, outcome):
    home_team = match["HomeTeam"]
    away_team = match["AwayTeam"]

    if outcome == 1:
        standings.loc[standings["Team"] == home_team, "Points"] += 3
    elif outcome == 0:
        standings.loc[standings["Team"] == home_team, "Points"] += 1
        standings.loc[standings["Team"] == away_team, "Points"] += 1
    else:
        standings.loc[standings["Team"] == away_team, "Points"] += 3
    return standings


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
        to_simulate = data.iloc[i]
        outcome = _generate_match_outcome(
            0.5, -0.5, to_simulate["TeamEloWinProb"] - 0.5
        )
        _adjust_standings(standings_copy, to_simulate, outcome)
        logger.debug(f"Simulated match {i}")
    return standings_copy


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
    match = data.iloc[t_k]
    if standings is None:
        standings = calculate_standings(data, tstart, t)
    else:
        standings = standings.copy()
    outcomes = []
    for _ in range(nruns):
        standings = run_simulation(data, standings, t, t_k, tend)
        outcomes.append(
            calculate_outcomes(standings, match["HomeTeam"], match["AwayTeam"])
        )
        logger.debug(f"Tournament simulation for {t_k} done")
    outcomes = np.mean(outcomes, axis=0)
    home_importance = max([outcomes[j] - outcomes[j + 6] for j in range(3)])
    away_importance = max([outcomes[j + 6] - outcomes[j] for j in range(3, 6)])
    return home_importance, away_importance


def backfill_szn_match_importance(
    data: pd.DataFrame,
    tstart: int,
    tend: int,
    nruns: int = 50,
    home_col: str = "HI",
    away_col: str = "AI",
) -> pd.DataFrame:
    """
    Backfill the match importance for an entire season

    Args:
    data: DataFrame containing the match data
    tstart: int representing the start of the season
    tend: int representing the end of the season
    nruns: int representing the number of simulations to run (default 50)
    home_col: str representing the home team column (default "HI")
    away_col: str representing the away team column (default "AI")

    Returns:
    A DataFrame containing the match importance
    """
    standings = _initialise_standings(data)
    for t_k in range(tstart, tend):
        data.loc[t_k, [home_col, away_col]] = calculate_match_importance(  # type: ignore
            data, standings, t_k, t_k, tstart, nruns, tend
        )
        match = data.iloc[t_k]
        if match["HTR"] == "H":
            outcome = 1
        elif match["HTR"] == "A":
            outcome = -1
        else:
            outcome = 0
        _adjust_standings(standings, match, outcome)
        logger.info(
            f"Match {t_k} done: {match['HomeTeam']} vs {match['AwayTeam']}, {match[home_col]}, {match[away_col]}"
        )
    return data
