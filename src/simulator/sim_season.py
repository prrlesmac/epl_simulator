from sim_utils import simulate_matches, calculate_standings
import pandas as pd
from config import config


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
    num_iter=config.number_of_simulations,
    verbose=False,
):
    """
    Runs a Monte Carlo simulation of a football season by repeatedly simulating
    pending matches and calculating final league standings.

    Args:
        schedule_played (pandas.DataFrame): DataFrame of already played matches.
        schedule_pending (pandas.DataFrame): DataFrame of pending matches, ready
            for simulation (e.g., includes Elo ratings).
        num_iter (int, optional): Number of simulation iterations. Default is 1000.
        verbose (bool, optional): If True, prints iteration numbers. Default is False.

    Returns:
        pandas.DataFrame: A DataFrame where each row is a team and each column is
        a finishing position (e.g., 1, 2, 3, ...) with values representing the
        proportion of times the team finished in that position across simulations.
    """
    standings_list = []

    for i in range(num_iter):
        if verbose:
            print(f"Simulation {i+1}/{num_iter}")

        # Simulate matches and compute standings
        simulated_pending = simulate_matches(schedule_pending.copy())
        schedule_final = pd.concat(
            [schedule_played, simulated_pending], ignore_index=True
        )
        standings_df = calculate_standings(schedule_final)
        standings_list.append(standings_df)

    # Aggregate position frequencies
    standings_all = (
        pd.concat(standings_list)
        .groupby(["team", "pos"])
        .size()
        .reset_index(name="count")
    )
    standings_all["count"] = standings_all["count"] / num_iter

    # Pivot to final probability table
    standings_all = (
        standings_all.pivot(index="team", columns="pos", values="count")
        .reset_index()
        .fillna(0)
    )
    standings_all.columns = standings_all.columns.astype(str)

    return standings_all


def aggregate_odds(standings):
    """
    Adds title, top 4, and relegation odds columns to the standings DataFrame
    and sorts teams by their title odds in descending order.

    Args:
        standings (pandas.DataFrame): DataFrame with team standings probabilities,
            with columns representing finishing positions as strings (e.g., "1", "2", ..., "20").

    Returns:
        pandas.DataFrame: The same DataFrame with three new columns added:
            - 'title_odds': Probability of finishing 1st.
            - 'top_4_odds': Probability of finishing in the top 4.
            - 'relegation_odds': Probability of finishing in relegation spots (18th, 19th, 20th).
        The DataFrame is sorted by 'title_odds' in descending order.
    """
    standings["title_odds"] = standings["1"]
    standings["top_4_odds"] = standings[["1", "2", "3", "4"]].sum(axis=1)
    standings["relegation_odds"] = standings[["18", "19", "20"]].sum(axis=1)
    standings = standings.sort_values(by="title_odds", ascending=False)
    return standings


if __name__ == "__main__":

    schedule = pd.read_csv(config.fixtures_output_file)
    elos = pd.read_csv(config.elo_output_file)

    schedule_played, schedule_pending = split_and_merge_schedule(schedule, elos)
    sim_standings = run_simulation(
        schedule_played, schedule_pending, num_iter=1000, verbose=True
    )

    sim_standings = aggregate_odds(sim_standings)

    sim_standings.to_csv(config.sim_output_file, index=False)
