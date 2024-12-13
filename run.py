import pandas as pd
import numpy as np

from batch import calculate_match_importance

master = pd.read_csv('data/michal-master.csv').drop_duplicates()
master["Date"] = pd.to_datetime(master["Date"], format="%Y-%m-%d").dt.strftime("%d-%b-%y")

all_years = pd.concat(
    [
        pd.read_csv(f"data/all_years/{season}.csv")
        for season in [
            f"{i}_{i+1}"
            for i in range(17, 24)
        ]
    ]
)
all_years["Date"] = pd.to_datetime(all_years["Date"], format="%d/%m/%Y").dt.strftime("%d-%b-%y")
all_years = all_years.sort_values(by=["Date"]).reset_index(drop=True)

match_importance = all_years.merge(
    master,
    on=["Date", "HomeTeam", "AwayTeam"],
    suffixes=("", "_master"),
)
match_importance[["HI", "AI"]] = np.nan

for i in range(380, len(match_importance)+1, 380):
    match_importance = calculate_match_importance(
        data=match_importance,
        tstart=i-380,
        tend=i-1,
        nruns=1000
    )

match_importance.to_csv("data/match_importance.csv", index=False)