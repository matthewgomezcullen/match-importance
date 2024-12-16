import pandas as pd
import numpy as np

from importance import backfill, calculate_standings


EVEN = pd.DataFrame({
    'Date': ['01-Jan-21', '01-Jan-21'],
    'HomeTeam': ['C', 'B'],
    'AwayTeam': ['D', 'A'],
    "FTR": ["D", "H"],
    "HI": [np.nan, np.nan],
    "AI": [np.nan, np.nan],
    "TeamEloWinProb": [0.5, 0.5]
})


UNEVEN = pd.DataFrame({
    'Date': ['01-Jan-21', '01-Jan-21', '02-Jan-21'],
    'HomeTeam': ['B', 'B', 'B'],
    'AwayTeam': ['A', 'A', 'C'],
    "FTR": ["H", "H", "A"],
    "HI": [np.nan, np.nan, np.nan],
    "AI": [np.nan, np.nan, np.nan],
    "TeamEloWinProb": [0.5, 0.5, 0.5]
})

def test_standings_of_single_match():
    standings = calculate_standings(
        data=EVEN.loc[0:0]
    )
    assert (standings["Points"] == 1).all()


def test_standings_of_matches():
    standings = calculate_standings(
        data=EVEN
    )
    assert standings.loc[0, "Team"] == "B"
    assert standings.loc[0, "Points"] == 3


def test_match_importance_should_equal_one():
    match_importance = backfill(
        data=EVEN,
        start=0,
        end=1,
        nruns=100
    )
    assert match_importance.loc[1, "HI"] == 1
    assert match_importance.loc[1, "AI"] == 1


def test_match_importance_should_equal_zero():
    match_importance = backfill(
        data=UNEVEN,
        start=0,
        end=2,
        nruns=100
    )
    assert match_importance.loc[2, "HI"] == 0
    assert match_importance.loc[2, "AI"] == 0