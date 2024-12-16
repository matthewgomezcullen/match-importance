import pandas as pd
import numpy as np

from utils import generate_simulations, calculate_outcomes, calculate_standings


def backfill(
    data: pd.DataFrame,
    start: int=0,
    end: int | None=None,
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
    start: int representing the start of the season (default 0)
    end: int representing the end of the season (default None)
    nruns: int representing the number of simulations to run (default 1000)
    home_col: str representing the home team column (default "HI")
    away_col: str representing the away team column (default "AI")

    Returns:
    A DataFrame containing the match importance
    """
    if end is None:
        end = len(data) - 1
    print(f"Calculating match importance for the season, {start} to {end}")
    simulations = generate_simulations(data, start, end, nruns)
    for t_k in range(start, end+1):
        outcomes = []
        match = data.loc[t_k]
        for simulation in simulations:
            standings = calculate_standings(
                simulation, start=start, end=end, ignore=t_k
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


def predict(
    data: pd.DataFrame,
    start: int,
    end: int,
    current: int,
    targets: list[int],
    nruns: int = 1000,
    home_col: str = "HI",
    away_col: str = "AI",
):
    """
    Predict the outcomes of future matches
    
    Args:
    data: DataFrame containing the match data
    start: int representing the start of the season
    end: int representing the end of the season
    current: int representing the current matchday
    targets: list of matchdays to predict
    nruns: int representing the number of simulations to run (default 1000)
    home_col: str representing the home team column (default "HI")
    away_col: str representing the away team column (default "AI")
    """
    simulations = generate_simulations(data, start, end, nruns, current)
    for t_k in targets:
        outcomes = []
        match = data.loc[t_k]
        for simulation in simulations:
            standings = calculate_standings(
                simulation, start=start, end=end, ignore=t_k
            )
            outcomes.append(calculate_outcomes(standings, match["HomeTeam"], match["AwayTeam"]))  # type: ignore
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