import pandas as pd
import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join('src')))

from importance import backfill
import elo


SEASON_DIR = "data/seasons"
OUTPUT_DIR = "output"
PREFIX = "importance"

if __name__ == "__main__":

    if len(sys.argv) != 2:
        raise ValueError("Please provide a season argument in the format 'YY_YY'")

    season = sys.argv[1]

    matches = pd.read_csv(f"{SEASON_DIR}/{season}.csv")
    matches["Date"] = pd.to_datetime(matches["Date"], dayfirst=True).dt.date
    elos = elo.get(matches)
    elos["Date"] = pd.to_datetime(elos["Date"]).dt.date
    elos["HomeTeam"] = elos["HomeTeam"].replace("Forest", "Nott'm Forest")
    elos["AwayTeam"] = elos["AwayTeam"].replace("Forest", "Nott'm Forest")

    match_importance = matches.merge(
        elos,
        on=["Date", "HomeTeam", "AwayTeam"],
        suffixes=("", "_master"),
    )
    match_importance[["HI", "AI"]] = np.nan
    match_importance = match_importance.sort_values("Date")

    match_importance = backfill(
        data=match_importance,
        nruns=1000
    )[["Date", "HomeTeam", "AwayTeam", "HomeElo", "AwayElo", "HI", "AI"]]

    match_importance.to_csv(f"{OUTPUT_DIR}/{PREFIX}_{season}.csv", index=False)