import pandas as pd
import soccerdata as sd

from tqdm import tqdm

tqdm.pandas()

def get(data: pd.DataFrame):
    elo = sd.ClubElo()

    # Map for team names
    CLUB_ELO_MAP = {
        "Nott'm Forest": "Forest",
    }

    def get_proper_name(name):
        """Standardize team names using a mapping."""
        return CLUB_ELO_MAP.get(name, name)

    def get_elo_per_team(df):
        """Retrieve Elo ratings for all teams."""
        teams = list(set(df["HomeTeam"]) | set(df["AwayTeam"]))
        elos = pd.concat([
            elo.read_team_history(team)
            for team in teams
        ], ignore_index=True)

        elos = elos[elos['to'] >= '2009-01-01']
        elos.reset_index(drop=True, inplace=True)

        def get_team_elos(row):
            match_date = row["Date"]

            home_elo = elos[
                (elos["team"] == row["HomeTeam"]) &
                (elos["to"] < match_date)
            ]["elo"].iloc[-1]

            away_elo = elos[
                (elos["team"] == row["AwayTeam"]) &
                (elos["to"] < match_date)
            ]["elo"].iloc[-1]

            return home_elo, away_elo

        return df.progress_apply(get_team_elos, axis=1)

    data = data[["Date", "HomeTeam", "AwayTeam"]].copy()
    data["HomeTeam"] = data["HomeTeam"].apply(get_proper_name)
    data["AwayTeam"] = data["AwayTeam"].apply(get_proper_name)
    data["Date"] = pd.to_datetime(data["Date"], format="%d/%m/%Y")
    data.reset_index(drop=True, inplace=True)

    elo_ratings = get_elo_per_team(data)
    data[["HomeElo", "AwayElo"]] = pd.DataFrame(elo_ratings.tolist(), index=data.index)

    # Calculate TeamEloWinProb
    data["TeamEloWinProb"] = 1 / (10 ** (-(data["HomeElo"] - data["AwayElo"]) / 400) + 1)

    return data