from simulator.sim_utils import (
    simulate_matches_data_frame,
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
import os
from multiprocessing import Pool, cpu_count
from dotenv import load_dotenv

load_dotenv()

def split_and_merge_schedule(schedule, elos, divisions=None):
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
        divisions (pandas.DataFrame): A DataFrame containing divisions for the league

    Returns:
        tuple:
            - schedule_played (pandas.DataFrame): Matches marked as played.
            - schedule_pending (pandas.DataFrame): Matches not yet played, with added
              'elo_home' and 'elo_away' columns representing Elo ratings for each team.
    """
    schedule = (
        schedule.merge(elos, how="left", left_on="home", right_on="club")
        .merge(elos, how="left", left_on="away", right_on="club")
        .rename(columns={"elo_x": "elo_home", "elo_y": "elo_away"})
        .drop(columns=["club_x", "club_y"])
    )

    if divisions is not None:
        schedule = (
            schedule
            .merge(divisions, how="left", left_on="home",right_on="team")
            .merge(divisions, how="left", left_on="away",right_on="team")
        )
        schedule = schedule.rename(columns={
            'division_x': 'home_division',
            'division_y': 'away_division',
            'conference_x': 'home_conference',
            'conference_y': 'away_conference',
        })
        schedule = schedule.drop(columns=['team_x','team_y'])

    schedule_played = schedule[schedule["played"] == "Y"].copy()
    schedule_pending = schedule[schedule["played"] == "N"].copy()

    # if elo is missing send warning and fill with 1000
    if (
        schedule_pending["elo_home"].isnull().any()
        or schedule_pending["elo_away"].isnull().any()
    ):
        print(
            "Warning: Some Elo ratings are missing. Filling with default value of 1000."
        )
        schedule_pending["elo_home"].fillna(1000, inplace=True)
        schedule_pending["elo_away"].fillna(1000, inplace=True)

    return schedule_played, schedule_pending


def single_simulation(
    schedule_played,
    schedule_pending,
    elos,
    divisions,
    league_rules,
):
    """
    Simulates the remaining matches and computes the final standings.

    Args:
        schedule_played (pd.DataFrame): DataFrame containing matches already played.
        schedule_pending (pd.DataFrame): DataFrame containing matches yet to be played.
        elos (pd.DataFrame): DataFrame containing elo ratings for each team
        divisions (pd.DataFrame): DataFrame containing the divisions for each team in the league.
        league_rules (Dict): dictionary containing the following keys
            classification (list): Rules used to compute standings/classification.
            sim_type (str): either "goals" or "winner"
                Used to specify whether the sinulation returns goal or simply the winner
            has_knockout (bool, optional): If True, includes knockout stage simulation.
                Default is False, meaning only league standings are simulated.
            knockout_bracket (list, optional): List of tuples defining the knockout
                bracket structure, e.g., [(1, 2), (3, 4)] for pairs of teams.
                Required if `has_knockout` is True.
            knockout_format (dict, optional): Dictionary defining the format of each round in the knockout stage.
                Example: {"po_r32": "two-legged", "po_r16": "two-legged", ...}
            knockout_draw_status (string, optional): Defines if there is a draw and if it has taken place
                Either "pending_draw", "completed_draw" or "no_draw"
                Required if `has_knockout` is True.
            knockout_draw (list, optional): List of tuples defining the knockout
                bracket structure, e.g., [(Team1, Team2), (Team3, Team4)] for pairs of teams.
                Required if `has_knockout` is True.
            knockout_reseeding (boolean): True if re-seeding is done after each ko round,
                False if not
                Required if `has_knockout` is True.

    Returns:
        pd.DataFrame: The standings DataFrame after simulating the pending matches and combining with played matches.
    """
    league_schedule_played = schedule_played[
        schedule_played["round"] == "League"
    ].copy()
    league_schedule_pending = schedule_pending[
        schedule_pending["round"] == "League"
    ].copy()
    simulated_pending = simulate_matches_data_frame(league_schedule_pending, league_rules["sim_type"])
    schedule_final = pd.concat(
        [league_schedule_played, simulated_pending], ignore_index=True
    )
    standings_df = get_standings(schedule_final, league_rules["classification"], league_rules["league_type"], divisions)

    if league_rules["has_knockout"]:
        knockout_schedule_played = schedule_played[
            schedule_played["round"] != "League"
        ].copy()
        knockout_schedule_pending = schedule_pending[
            schedule_pending["round"] != "League"
        ].copy()
        simulated_pending = simulate_matches_data_frame(knockout_schedule_pending, league_rules["sim_type"])
        playoff_schedule = pd.concat(
            [knockout_schedule_played, simulated_pending], ignore_index=True
        )
        if league_rules["knockout_draw_status"] == "pending_draw":
            draw = draw_from_pots(standings_df, pot_size=2)
            bracket = create_bracket_from_composition(draw, league_rules['knockout_bracket'])
        elif league_rules["knockout_draw_status"] == "completed_draw":
            if league_rules["knockout_reseeding"]:
                combined_bracket = [t1 + t2 for t1, t2 in zip(league_rules["knockout_draw"], league_rules["knockout_bracket"])]
                bracket = pd.DataFrame(combined_bracket, columns=["team1", "team2", "seed1", "seed2"])
            else:
                bracket = pd.DataFrame(league_rules["knockout_draw"], columns=["team1", "team2"])
        elif league_rules["knockout_draw_status"] == "no_draw":
            draw = standings_df.copy()
            draw["draw_order"] = draw["playoff_pos"]
            draw = draw[["team", "draw_order"]]
            bracket = create_bracket_from_composition(draw, league_rules['knockout_bracket'])
        else:
            raise ValueError("Invalid knockout draw status selected")
        # TODO think of better ways to pull elos
        elos = schedule_final.drop_duplicates(subset=["home"])[
            ["home", "elo_home"]
        ].rename(columns={"home": "team", "elo_home": "elo"})
        playoff_df = simulate_playoff_bracket(
            bracket, league_rules["knockout_format"], elos, playoff_schedule, league_rules["knockout_reseeding"]
        )
        standings_df = standings_df.merge(playoff_df, how="left", on="team")

    return standings_df


def run_simulation_parallel(
    schedule_played,
    schedule_pending,
    elos,
    divisions,
    league_rules,
    num_simulations=1000,
):
    """
    Run multiple simulations of pending matches in parallel using multiprocessing,
    then aggregate the results into probabilities of team positions.

    Args:
        schedule_played (pd.DataFrame): DataFrame of matches already played.
        schedule_pending (pd.DataFrame): DataFrame of matches yet to be played.
        elos (pd.DataFrame): DataFrame containing elo ratings for each team
        divisions (pd.DataFrame): DataFrame containing the divisions for each team in the league.
        league_rules (Dict): dictionary containing the following keys
            classification (Dict): Dictionary of lists containing
                rules used to compute standings/classification.
            sim_type (str): either "goals" or "winner"
                Used to specify whether the sinulation returns goal or simply the winner
            has_knockout (bool, optional): If True, includes knockout stage simulation.
                Default is False, meaning only league standings are simulated.
            knockout_bracket (list, optional): List of tuples defining the knockout
                bracket structure, e.g., [(1, 2), (3, 4)] for pairs of teams.
                Required if `has_knockout` is True.
            knockout_format (dict, optional): Dictionary defining the format of each round in the knockout stage.
                Example: {"po_r32": "two-legged", "po_r16": "two-legged", ...}
            knockout_draw_status (string, optional): Defines if there is a draw and if it has taken place
                Either "pending_draw", "completed_draw" or "no_draw"
                Required if `has_knockout` is True.
            knockout_draw (list, optional): List of tuples defining the knockout
                bracket structure, e.g., [(Team1, Team2), (Team3, Team4)] for pairs of teams.
                Required if `has_knockout` is True.
            knockout_reseeding (boolean): True if re-seeding is done after each ko round,
                False if not
                Required if `has_knockout` is True.
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
                elos,
                divisions,
                league_rules
            )
        ] * num_simulations
        standings_list = pool.starmap(single_simulation, args)
    #standings_list = [single_simulation(schedule_played, schedule_pending, elos, divisions, league_rules)]
    # Aggregate position frequencies
    standings_all = pd.concat(standings_list)
    if league_rules["has_knockout"]:
        standings_all["league_pos"] = standings_all["playoff_pos"]

    standings_all = (
        standings_all
        .groupby(["team", "league_pos"])
        .size()
        .reset_index(name="count")
    )
    standings_all["count"] = standings_all["count"] / num_simulations

    # Pivot to final probability table
    standings_all = (
        standings_all.pivot(index="team", columns="league_pos", values="count")
        .reset_index()
        .fillna(0)
    )
    standings_all.columns = standings_all.columns.astype(str)

    if league_rules["has_knockout"]:
        # Add knockout stage results if applicable
        standings_po = pd.concat(standings_list)
        knockout_cols = standings_po.columns[standings_po.columns.str.startswith("po_")]
        standings_po = standings_po.groupby(["team"])[knockout_cols].sum().reset_index()
        standings_all = standings_all.merge(standings_po, how="left", on="team")
        standings_all[knockout_cols] = standings_all[knockout_cols] / num_simulations

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


def load_league_data(league):
    """
    Load schedule and Elo data for a specific league from the database.

    Args:
        league (str): The league identifier to load data for.

    Returns:
        tuple: (schedule_df, elos_df, divisions) DataFrames containing schedule, Elo and divisional data
    """
    table_suffix = "uefa" if league in config.active_uefa_leagues else league
    is_continental_league = league in ["UCL", "UEL", "UECL"]
    engine = db_connect.get_postgres_engine()

    schedule = pd.read_sql(
        f"SELECT * FROM {config.db_table_definitions['fixtures_table']['name']}_{table_suffix} WHERE country = '{league}'",
        engine,
    )

    elos_query = f"SELECT * FROM {config.db_table_definitions['elo_table']['name']}" + (
        f" WHERE country = '{league}'" if not is_continental_league else ""
    )
    elos = pd.read_sql(elos_query, engine)

    if league in ["NFL", "MLB", "NBA"]:
        divisions = pd.read_sql(
            f"SELECT * FROM {config.db_table_definitions['divisions_table']['name']}_{table_suffix}",
            engine,
        )
    else:
        divisions = None

    # Apply cutoff dates if configured
    if config.played_cutoff_date:
        schedule["date"] = pd.to_datetime(schedule["date"], errors="coerce")
        schedule.loc[
            schedule["date"] > pd.to_datetime(config.played_cutoff_date), "played"
        ] = "N"

    if config.schedule_cutoff_date:
        schedule = schedule[
            (schedule["round"] == "League")
            | (
                (schedule["round"] != "League")
                & (schedule["date"] <= pd.to_datetime(config.schedule_cutoff_date))
            )
        ].copy()

    return schedule, elos, divisions


def validate_league_configuration(schedule, league_rules):
    """
    Validate that the league configuration is consistent with the schedule data.

    Args:
        schedule (pd.DataFrame): The schedule DataFrame.
        league_rules (dict): The league rules configuration.

    Raises:
        ValueError: If the configuration is inconsistent.
    """
    knockout_draw = league_rules.get("knockout_draw")

    has_knockout_matches = not schedule[schedule["round"] != "League"].empty
    has_pending_league_matches = not schedule[
        (schedule["round"] == "League") & (schedule["played"] == "N")
    ].empty

    if league_rules["has_knockout"] and knockout_draw is None and has_knockout_matches:
        raise ValueError(
            f"League has knockout matches but no bracket draw defined. "
            "Please provide a knockout_draw in the league rules."
        )

    if (
        league_rules["has_knockout"]
        and knockout_draw is not None
        and has_pending_league_matches
    ):
        raise ValueError(
            f"League has a bracket draw defined but league phase is unfinished. "
            "Please remove the bracket draw."
        )


def simulate_league(league_rules, schedule, elos, divisions, num_simulations=1000):
    """
    Simulate a single league and return the results.

    Args:
        league_rules (Dict): the league rules from the config
            should include has knockout, knockout bracket, knockout format,
            kncokout draw, classification and qualification
        schedule (pd.DataFrame): DataFrame containing schedule of matches played and to be played
        elos (pd.DataFrame): DataFrame containing the elo rating for each team
        divisions (pd.DataFrame): DataFrame containing the divisions for each team in the league.
        num_simulations (int, optional): Number of simulations to run in parallel. Defaults to 1000.
    Returns:
        pd.DataFrame: The simulation results for the league.
    """
    # Validate configuration
    validate_league_configuration(schedule, league_rules)
    # Prepare data for simulation
    schedule_played, schedule_pending = split_and_merge_schedule(schedule, elos, divisions)

    # Run simulation
    sim_standings = run_simulation_parallel(
        schedule_played,
        schedule_pending,
        elos,
        divisions,
        league_rules,
        num_simulations=num_simulations,
    )

    # Aggregate results
    sim_standings = aggregate_standings_outcomes(sim_standings, league_rules["qualification"])
    sim_standings["updated_at"] = datetime.now()

    return sim_standings


def save_results_to_database(sim_standings_wo_ko, sim_standings_w_ko):
    """
    Save simulation results to the database.

    Args:
        sim_standings_wo_ko (list): List of DataFrames for leagues without knockout stages.
        sim_standings_w_ko (list): List of DataFrames for leagues with knockout stages.
    """
    engine = db_connect.get_postgres_engine()
    if sim_standings_wo_ko:
        sim_standings_wo_ko_df = pd.concat(sim_standings_wo_ko)
        sim_standings_wo_ko_df.to_sql(
            config.db_table_definitions["domestic_sim_output_table"]["name"],
            engine,
            if_exists="replace",
            index=False,
            dtype=config.db_table_definitions["domestic_sim_output_table"]["dtype"],
        )

    if sim_standings_w_ko:
        sim_standings_w_ko_df = pd.concat(sim_standings_w_ko)
        sim_standings_w_ko_df.to_sql(
            config.db_table_definitions["continental_sim_output_table"]["name"],
            engine,
            if_exists="replace",
            index=False,
            dtype=config.db_table_definitions["continental_sim_output_table"]["dtype"],
        )


def run_all_simulations():
    """
    Main function to run simulations for all configured leagues.
    """
    start_time = time.time()
    sim_standings_wo_ko = []
    sim_standings_w_ko = []
    league_type = os.getenv("LEAGUES_TO_SIM")
    leagues_to_sim = config.active_uefa_leagues if league_type == "UEFA" else [league_type]

    for league in leagues_to_sim:
        schedule, elos, divisions = load_league_data(league)
        league_rules = config.league_rules[league]
        league_rules["league_type"] = league_type
        print("Simulating league: ", league)
        sim_standings = simulate_league(
            league_rules, schedule, elos, divisions, num_simulations=config.number_of_simulations
        )
        sim_standings["league"] = league

        if config.league_rules[league]["has_knockout"]:
            sim_standings_w_ko.append(sim_standings)
        else:
            sim_standings_wo_ko.append(sim_standings)
    end_time = time.time()
    print(f"Simulation took {end_time - start_time:.2f} seconds")
    # Save results to database
    save_results_to_database(sim_standings_wo_ko, sim_standings_w_ko)
    print("Simulations saved to db")


if __name__ == "__main__":
    run_all_simulations()
