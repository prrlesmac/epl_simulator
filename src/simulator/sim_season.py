from simulator.sim_utils import (
    simulate_matches,
    get_standings,
    draw_from_pots,
    create_bracket_from_composition,
    simulate_playoff_bracket,
)
import pandas as pd
from config import config
from db import db_connect
from datetime import datetime
import time
from multiprocessing import Pool, cpu_count


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
    # if elo is missing send warning and fill with 1000 
    if schedule_pending["elo_home"].isnull().any() or schedule_pending["elo_away"].isnull().any():
        print("Warning: Some Elo ratings are missing. Filling with default value of 1000.")
        schedule_pending["elo_home"].fillna(1000, inplace=True)
        schedule_pending["elo_away"].fillna(1000, inplace=True)

    return schedule_played, schedule_pending


def single_simulation(
    schedule_played,
    schedule_pending,
    classif_rules,
    has_knockout=False,
    bracket_composition=None,
    bracket_format=None,
):
    """
    Simulates the remaining matches and computes the final standings.

    Args:
        schedule_played (pd.DataFrame): DataFrame containing matches already played.
        schedule_pending (pd.DataFrame): DataFrame containing matches yet to be played.
        classif_rules (callable or object): Rules or function used to compute standings/classification.
        has_knockout (bool, optional): If True, includes knockout stage simulation.
            Default is False, meaning only league standings are simulated.
        bracket_composition (list, optional): List of tuples defining the knockout
            bracket structure, e.g., [(1, 2), (3, 4)] for pairs of teams.
            Required if `has_knockout` is True.
        bracket_format (dict, optional): Dictionary defining the format of each round in the knockout stage.
            Example: {"po_r32": "two-legged", "po_r16": "two-legged", ...}

    Returns:
        pd.DataFrame: The standings DataFrame after simulating the pending matches and combining with played matches.
    """
    simulated_pending = simulate_matches(schedule_pending.copy(), config.home_advantage)
    schedule_final = pd.concat([schedule_played, simulated_pending], ignore_index=True)
    standings_df = get_standings(schedule_final, classif_rules)

    if has_knockout:
        # If knockout stage is included, simulate it
        draw = draw_from_pots(standings_df, pot_size=2)
        bracket = create_bracket_from_composition(draw, bracket_composition)
        # TODO think ofb etter ways to pull elos
        elos = schedule_final.drop_duplicates(subset=["home"])[
            ["home", "elo_home"]
        ].rename(columns={"home": "team", "elo_home": "elo"})
        playoff_df = simulate_playoff_bracket(bracket, bracket_format, elos)
        standings_df = standings_df.merge(playoff_df, how="left", on="team")

    return standings_df


def run_simulation_parallel(
    schedule_played,
    schedule_pending,
    classif_rules,
    has_knockout=False,
    bracket_composition=None,
    bracket_format=None,
    num_simulations=1000,
):
    """
    Run multiple simulations of pending matches in parallel using multiprocessing,
    then aggregate the results into probabilities of team positions.

    Args:
        schedule_played (pd.DataFrame): DataFrame of matches already played.
        schedule_pending (pd.DataFrame): DataFrame of matches yet to be played.
        classif_rules (callable or object): Rules or function to compute standings/classification.
        has_knockout (bool, optional): If True, includes knockout stage simulation.
            Default is False, meaning only league standings are simulated.
        bracket_composition (list, optional): List of tuples defining the knockout
            bracket structure, e.g., [(1, 2), (3, 4)] for pairs of teams.
            Required if `has_knockout` is True.
        bracket_format (dict, optional): Dictionary defining the format of each round in the knockout stage.
            Example: {"po_r32": "two-legged", "po_r16": "two-legged", ...}
        num_simulations (int, optional): Number of simulations to run in parallel. Defaults to 1000.

    Returns:
        pd.DataFrame: DataFrame where each row is a team and columns represent
                      the probability of finishing in each position. Column names
                      are strings of the positions.
    """
    print(f"Running {num_simulations} simulations using multiprocessing...")

    with Pool(processes=cpu_count()) as pool:
        args = [
            (
                schedule_played,
                schedule_pending,
                classif_rules,
                has_knockout,
                bracket_composition,
                bracket_format,
            )
        ] * num_simulations
        standings_list = pool.starmap(single_simulation, args)
        # standings_list = [single_simulation(schedule_played, schedule_pending, classif_rules, has_knockout, bracket_composition)]
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

    if has_knockout:
        # Add knockout stage results if applicable
        standings_po = pd.concat(standings_list)
        knockout_cols = standings_po.columns[standings_po.columns.str.startswith("po_")]
        standings_po = standings_po.groupby(["team"])[knockout_cols].sum().reset_index()
        standings_all = standings_all.merge(standings_po, how="left", on="team")
        standings_all[knockout_cols] = standings_all[knockout_cols] / num_simulations

    return standings_all


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
        classif_rules (list): successive order of the criteria to apply to classify positions
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


def aggregate_standings_outcomes(standings, qualification_mapping):
    """
    Adds title, top 4, and relegation odds columns to the standings DataFrame
    and sorts teams by their title odds in descending order.

    Args:
        standings (pandas.DataFrame): DataFrame with team standings probabilities,
            with columns representing finishing positions as strings (e.g., "1", "2", ..., "20").
        qualification_mapping (dict): dict containing the qualification rules for the league, with the following possible keys:
            - "champion": positions with championship, for example [1]
            - "top_4": positions going into UCL or top 4, for example [1,2,3,4]
            - "relegation_direct": positions with direct relegation, for example [18,19,20]
            - "relegation_playoff": positions going into relegation palyoff, for example [16]

    Returns:
        pandas.DataFrame: The same DataFrame with three new columns added:
            - 'champion': Probability of finishing 1st.
            - 'top_4': Probability of finishing in the top 4.
            - 'relegation_playoff': Probability of finishing in relegation playoff spots (18th).
            - 'relegation_direct': Probability of finishing in relegation spots (18th, 19th, 20th).
        The DataFrame is sorted by 'title_odds' in descending order.
    """
    for k, v in qualification_mapping.items():
        standings_list = [str(i) for i in v]
        standings[k] = standings[standings_list].sum(axis=1)
    return standings


if __name__ == "__main__":
    start_time = time.time()
    sim_standings_wo_ko = []
    sim_standings_w_ko = []

    for league in config.leagues_to_sim:
        is_continental_league = league in ["UCL", "UEL", "UECL"]
        print("Simulating: ", league)
        engine = db_connect.get_postgres_engine()
        schedule = pd.read_sql(
            f"SELECT * FROM {config.db_table_definitions['fixtures_table']['name']} WHERE country = '{league}'",
            engine,
        )
        elos_query = f"SELECT * FROM {config.db_table_definitions['elo_table']['name']}" + (
            f" WHERE country = '{league}'" if not is_continental_league else ""
        )
        elos = pd.read_sql(
            elos_query,
            engine,
        )

        league_rules = config.league_rules[league]
        bracket_composition = (
            league_rules["knockout_bracket"] if is_continental_league else None
        )
        bracket_format = league_rules["knockout_format"] if is_continental_league else None
        classif_rules = league_rules["classification"]
        qualif_rules = league_rules["qualification"]
        schedule_played, schedule_pending = split_and_merge_schedule(schedule, elos)
        sim_standings = run_simulation_parallel(
            schedule_played,
            schedule_pending,
            classif_rules,
            has_knockout=is_continental_league,
            bracket_composition=bracket_composition,
            bracket_format=bracket_format,
            num_simulations=config.number_of_simulations,
        )
        """
        sim_standings = run_simulation(
            schedule_played,
            schedule_pending,
            classif_rules,
            num_simulations=config.number_of_simulations,
            verbose=False,
        )
        """
        sim_standings = aggregate_standings_outcomes(sim_standings, qualif_rules)
        sim_standings["league"] = league
        sim_standings["updated_at"] = datetime.now()
        if not league_rules["has_knockout"]:
            sim_standings_wo_ko.append(sim_standings)
        else:
            sim_standings_w_ko.append(sim_standings)

    end_time = time.time()
    print(f"Simulation took {end_time - start_time:.2f} seconds")
    sim_standings_wo_ko = pd.concat(sim_standings_wo_ko)
    sim_standings_w_ko = pd.concat(sim_standings_w_ko)

    # reconnect
    engine = db_connect.get_postgres_engine()
    sim_standings_wo_ko.to_sql(
        config.db_table_definitions["domestic_sim_output_table"]["name"],
        engine,
        if_exists="replace",
        index=False,
        dtype=config.db_table_definitions["domestic_sim_output_table"]["dtype"],
    )
    sim_standings_w_ko.to_sql(
        config.db_table_definitions["continental_sim_output_table"]["name"],
        engine,
        if_exists="replace",
        index=False,
        dtype=config.db_table_definitions["continental_sim_output_table"]["dtype"],
    )
    print(f"Simulations saved to db")
