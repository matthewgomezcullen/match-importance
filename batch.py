import pandas as pd
import numpy as np

from utils import generate_match_outcome, calculate_outcomes, calculate_standings


def _generate_simulations(
    data: pd.DataFrame,
    tstart: int,
    tend: int,
    nruns: int = 1000,
) -> list[pd.DataFrame]:
    """
    Simulate seasons

    Args:
    data: DataFrame containing the match data
    tstart: int representing the start of the season
    tend: int representing the end of the season
    nruns: int representing the number of simulations to run per match (default 1000)

    Returns:
    A DataFrame containing the simulated seasons
    """
    simulations = []
    for _ in range(nruns):
        simulation = data.loc[tstart:tend].copy()
        simulation.loc[:, "FTR"] = simulation.apply(generate_match_outcome, axis=1)
        simulations.append(simulation)
    return simulations


def calculate_match_importance(
    data: pd.DataFrame,
    tstart: int,
    tend: int,
    nruns: int = 1000,
    home_col: str = "HI",
    away_col: str = "AI",
) -> pd.DataFrame:
    """
    Efficiently backfill the match importance for an entire season.
    All matches are simulated 1000 times, rather than 100 times for each match before it. The 1000
    simulations form 1000 different standings, which are used to calculate the match importance.

    Args:
    data: DataFrame containing the match data
    tstart: int representing the start of the season
    tend: int representing the end of the season
    nruns: int representing the number of simulations to run (default 1000)
    home_col: str representing the home team column (default "HI")
    away_col: str representing the away team column (default "AI")

    Returns:
    A DataFrame containing the match importance
    """
    print(f"Calculating match importance for the season, {tstart} to {tend}")
    simulations = _generate_simulations(data, tstart, tend, nruns)
    for t_k in range(tstart, tend+1):
        outcomes = []
        match = data.loc[t_k]
        for simulation in simulations:
            standings = calculate_standings(
                simulation, tstart=tstart, tend=tend, ignore=t_k
            )
            outcomes.append(calculate_outcomes(standings, match["HomeTeam"], match["AwayTeam"]))  # type: ignore
            simulation.loc[t_k] = match
        outcomes = np.mean(outcomes, axis=0)
        data.loc[t_k, home_col] = max([outcomes[j] - outcomes[j + 6] for j in range(3)])
        data.loc[t_k, away_col] = max(
            [outcomes[j + 6] - outcomes[j] for j in range(3, 6)]
        )
        print(
            f"Match {t_k} done: {match['HomeTeam']} vs {match['AwayTeam']}, "
            f"{data.loc[t_k, home_col]}, {data.loc[t_k, away_col]}"
        )
    return data
