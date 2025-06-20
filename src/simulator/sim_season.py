from simulator.sim_utils import simulate_matches, get_standings
import pandas as pd
from config import config
from db import db_connect
from datetime import datetime

def split_and_merge_schedule(schedule, elos):
    """
    Splits the schedule into played and pending matches, and merges Elo ratings
    into the pending matches.

    This function separates matches that have already been played from those that
    are still pending (based on the 'played' column). For pending matches, it merges
    Elo ratings for both home and away teams based on club names.

    Args:
        schedule (pandas.DataFrame): A DataFrame containing the full match schedule,
            with at least 'home', 'away', and 'played' columns.
        elos (pandas.DataFrame): A DataFrame containing Elo ratings with 'club' and 'elo' columns.

    Returns:
        tuple:
            - schedule_played (pandas.DataFrame): Matches marked as played.
            - schedule_pending (pandas.DataFrame): Matches not yet played, with added
              'elo_home' and 'elo_away' columns representing Elo ratings for each team.
    """
    schedule_played = schedule[schedule["played"] == "Y"]
    schedule_pending = schedule[schedule["played"] == "N"]

    schedule_pending = (
        schedule_pending.merge(elos, how="left", left_on="home", right_on="club")
        .merge(elos, how="left", left_on="away", right_on="club")
        .rename(columns={"elo_x": "elo_home", "elo_y": "elo_away"})
        .drop(columns=["club_x", "club_y"])
    )

    return schedule_played, schedule_pending


def run_simulation(
    schedule_played,
    schedule_pending,
    classif_rules,
    num_simulations=1000,
    verbose=False,
):
    """
    Runs a Monte Carlo simulation of a football season by repeatedly simulating
    pending matches and calculating final league standings.

    Args:
        schedule_played (pandas.DataFrame): DataFrame of already played matches.
        schedule_pending (pandas.DataFrame): DataFrame of pending matches, ready
            for simulation (e.g., includes Elo ratings).
        classification_rules (list): successive order of the criteria to apply to classify positions
        num_simulations (int, optional): Number of simulation iterations. Default is 1000.
        verbose (bool, optional): If True, prints iteration numbers. Default is False.

    Returns:
        pandas.DataFrame: A DataFrame where each row is a team and each column is
        a finishing position (e.g., 1, 2, 3, ...) with values representing the
        proportion of times the team finished in that position across simulations.
    """
    standings_list = []
    print(f"Run simulations...")

    for i in range(num_simulations):
        if verbose:
            print(f"Simulation {i+1}/{num_simulations}")
        # Simulate matches and compute standings
        simulated_pending = simulate_matches(schedule_pending.copy())
        schedule_final = pd.concat(
            [schedule_played, simulated_pending], ignore_index=True
        )
        standings_df = get_standings(schedule_final, classif_rules)
        standings_list.append(standings_df)

    # Aggregate position frequencies
    standings_all = (
        pd.concat(standings_list)
        .groupby(["team", "pos"])
        .size()
        .reset_index(name="count")
    )
    standings_all["count"] = standings_all["count"] / num_simulations

    # Pivot to final probability table
    standings_all = (
        standings_all.pivot(index="team", columns="pos", values="count")
        .reset_index()
        .fillna(0)
    )
    standings_all.columns = standings_all.columns.astype(str)

    return standings_all


def aggregate_odds(standings, relegation_rules):
    """
    Adds title, top 4, and relegation odds columns to the standings DataFrame
    and sorts teams by their title odds in descending order.

    Args:
        standings (pandas.DataFrame): DataFrame with team standings probabilities,
            with columns representing finishing positions as strings (e.g., "1", "2", ..., "20").
        relegation_rules (dict): dict containing the relegation rules for the league, with the following keys:
            - "direct": positions with direct relegation, for example [18,19,20]
            - "playoff": positions going into relegation palyoff, for example [16]

    Returns:
        pandas.DataFrame: The same DataFrame with three new columns added:
            - 'title_odds': Probability of finishing 1st.
            - 'top_4_odds': Probability of finishing in the top 4.
            - 'relegation_odds': Probability of finishing in relegation spots (18th, 19th, 20th).
        The DataFrame is sorted by 'title_odds' in descending order.
    """
    standings["title_odds"] = standings["1"]
    standings["top_4_odds"] = standings[["1", "2", "3", "4"]].sum(axis=1)
    direct_relegation = [str(i) for i in relegation_rules['direct']]
    standings["direct_relegation_odds"] = standings[direct_relegation].sum(axis=1)
    if relegation_rules['playoff'] is not None:
        playoff_relegation = [str(i) for i in relegation_rules['playoff']]
        standings["relegation_playoff_odds"] = standings[playoff_relegation].sum(axis=1)
    else:
        standings["relegation_playoff_odds"] = 0
    standings = standings.sort_values(by="title_odds", ascending=False)

    return standings


if __name__ == "__main__":
    engine = db_connect.get_postgres_engine()
    sim_standings_all = []
    for league in config.leagues_to_sim:
        print("Simulating: ", league)
        schedule = pd.read_sql(f"SELECT * FROM {config.fixtures_table} WHERE country = '{league}'", engine)
        elos = pd.read_sql(f"SELECT * FROM {config.elo_table} WHERE country = '{league}'", engine)
        classif_rules = config.classification[league]
        relegation_rules = config.relegation[league]
        schedule_played, schedule_pending = split_and_merge_schedule(schedule, elos)
        sim_standings = run_simulation(
            schedule_played,
            schedule_pending,
            classif_rules,
            num_simulations=config.number_of_simulations,
            verbose=False,
        )

        sim_standings = aggregate_odds(sim_standings, relegation_rules)
        sim_standings["country"] = league
        sim_standings["updated_at"] = datetime.now()
        sim_standings_all.append(sim_standings)

    sim_standings_all = pd.concat(sim_standings_all)
    sim_standings_all.to_sql(f"{config.sim_output_table}", engine, if_exists="replace", index=False)
    print(f"Simulations saved to db")
