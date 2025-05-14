import pandas as pd
import numpy as np
import math


def calculate_win_probability(elo_home, elo_away):
    """
    Calculate win probability using the Elo rating system.

    Parameters:
        elo_home (float): Elo rating of team 1
        elo_away (float): Elo rating of team 2

    Returns:
        float: Probability of team 1 winning
    """
    return 1 / (1 + 10 ** ((elo_away - elo_home) / 400))


def simulate_match(proba):
    """
    Simulate a match outcome based on the win probability.

    Parameters:
        proba (float): Probability of team 1 winning

    Returns:
        str: H if team 1 wins, A if team 2 wins
    """

    ExpGH = np.where(
        proba < 0.5,
        0.2 + 1.1 * math.sqrt(proba / 0.5),
        1.69 / (1.12 * math.sqrt(2 - proba / 0.5) + 0.18),
    )
    ExpGA = np.where(
        (1 - proba) < 0.5,
        0.2 + 1.1 * math.sqrt((1 - proba) / 0.5),
        1.69 / (1.12 * math.sqrt(2 - (1 - proba) / 0.5) + 0.18),
    )
    Base = np.random.poisson(0.18 * min(ExpGA, ExpGH))
    GH = np.random.poisson(ExpGH - 0.18 * min(ExpGA, ExpGH)) + Base
    GA = np.random.poisson(ExpGA - 0.18 * min(ExpGA, ExpGH)) + Base
    #Diff = abs(GH - GA)
    #K = 30 * np.where(Diff <= 1, 1, np.where(Diff == 2, 1.25, 1.25 + (Diff - 2) / 8))
    #Res = np.where(GH > GA, 1, np.where(GH == GA, 0.5, 0))
    #EloDiff = (proba - Res) * K
    #EloH = EloH - EloDiff
    #EloA = EloA + EloDiff

    return (GH, GA)


def simulate_matches(matches_df):
    """
    Simulate matches and determine winners.

    Parameters:
        matches_df (pd.DataFrame): DataFrame containing matches to simulate

    Returns:
        pd.DataFrame: DataFrame with simulation results
    """

    for index, match in matches_df.iterrows():

        elo_home = match["elo_home"]
        elo_away = match["elo_away"]

        # Calculate win probability for Team 1
        win_proba = calculate_win_probability(elo_home, elo_away)

        # Simulate match
        result = simulate_match(win_proba)
        # Append result
        matches_df.at[index, "home_goals"] = result[0]
        matches_df.at[index, "away_goals"] = result[1]

    return pd.DataFrame(matches_df)


def calculate_standings(matches_df):
    """
    Calculate standings based on win/loss records.

    Parameters:
        matches_df (pd.DataFrame): Match results DataFrame

    Returns:
        pd.DataFrame: Standings sorted by win percentage and wins
    """
    # Initialize standings DataFrame
    teams = pd.unique(matches_df[["home", "away"]].values.ravel())
    standings = pd.DataFrame({"Team": teams, "Wins": 0, "Losses": 0})
    matches_df["home_pts"] = np.where(
        matches_df["home_goals"] > matches_df["away_goals"],
        3,
        np.where(matches_df["home_goals"] == matches_df["away_goals"], 1, 0)
    )
    matches_df["away_pts"] = np.where(
        matches_df["away_goals"] > matches_df["home_goals"],
        3,
        np.where(matches_df["home_goals"] == matches_df["away_goals"], 1, 0)
    )
    home_pts = matches_df.groupby(["home"])[['home_pts','home_goals','away_goals']].sum().reset_index()
    away_pts = matches_df.groupby(["away"])[['away_pts','away_goals','home_goals']].sum().reset_index()

    home_pts = home_pts.rename(columns={'home': 'team','home_goals': 'home_goals_for','away_goals':'away_goals_against'})
    away_pts = away_pts.rename(columns={'away': 'team','away_goals': 'away_goals_for','home_goals':'home_goals_against'})
    # Combine wins and losses into a single DataFrame
    standings = pd.merge(
        home_pts, away_pts, how="outer", on='team'
    ).fillna(0)
    standings['pts'] = standings['home_pts'] + standings['away_pts']
    standings['gf'] = standings['home_goals_for'] + standings['away_goals_for']
    standings['ga'] = standings['home_goals_against'] + standings['away_goals_against']
    standings["gd"] = standings["gf"] - standings["ga"]
    standings = standings[["team", "pts", "gd", "gf","ga"]].fillna(0)

    # Sort by wins
    standings = standings.sort_values(by=["pts","gd","gf"], ascending=[False,False,False]).reset_index(drop=True)
    standings = standings.reset_index(drop=True)  # Removes old index
    standings['pos'] = standings.index + 1 

    return standings


# Example usage
if __name__ == "__main__":
    # Example DataFrame with matches
    schedule = pd.read_csv("../../data/01_raw/epl_matches.csv")
    elos = pd.read_csv("../../data/02_intermediate/current_elo_ratings.csv")

    schedule_played = schedule[schedule["played"] == "Y"]
    schedule_pending = schedule[schedule["played"] == "N"]
    schedule_pending = schedule_pending.merge(
        elos, how="left", left_on="home", right_on="club"
    ).merge(elos, how="left", left_on="away", right_on="club")
    schedule_pending = schedule_pending.rename(
        columns={"elo_x": "elo_home", "elo_y": "elo_away"}
    )
    schedule_pending = schedule_pending.drop(columns=["club_x", "club_y"])

    standings_list = []
    num_of_iter=1000
    for i in range(num_of_iter):
        # Simulate matches
        print(i)
        schedule_pending = simulate_matches(schedule_pending)
        schedule_final = pd.concat([schedule_played, schedule_pending])
        standings_df = calculate_standings(schedule_final)
        # Append standings to list
        standings_list.append(standings_df)
    standings_all = pd.concat(standings_list).groupby(['team', 'pos']).size().reset_index(name='count')
    standings_all['count'] = standings_all['count'] / num_of_iter
    standings_all = standings_all.pivot(index='team', columns='pos', values='count').reset_index().fillna(0)
    standings_all.columns = standings_all.columns.astype(str)
    standings_all['title_odds'] = standings_all['1']
    standings_all['top_4_odds'] = standings_all[['1', '2', '3', '4']].sum(axis=1)
    standings_all['relegation_odds'] = standings_all[['18', '19', '20']].sum(axis=1)
    standings_all = standings_all.sort_values(by='title_odds',ascending=False)

    standings_all.to_csv(
        "../../data/02_intermediate/season_standings_sim.csv", index=False
    )
