from ranker.elo_utils import EloCalculator
from db import db_connect
import pandas as pd
from datetime import datetime
from config import config
import os
from dotenv import load_dotenv

load_dotenv()

def load_matches_data(league, name_remap):
    """
    Load schedule for a specific league from the database.

    Args:
        league (str): The league identifier to load data for.
        name_remap (Dict): Contains the remapping for franchise name changes

    Returns:
        pd.DataFrame: DataFrames containing all matches for a league
    """
    engine = db_connect.get_postgres_engine()
    table_suffix = league.lower()
    matches = pd.read_sql(
        f"""
        SELECT * FROM {config.db_table_definitions['fixtures_table']['name']}_{table_suffix}_history 
        UNION ALL
        SELECT * FROM {config.db_table_definitions['fixtures_table']['name']}_{table_suffix}
        WHERE played = 'Y'
        AND date >= '2026-02-16'
        ORDER BY date
        """,
        engine,
    )
    matches["home_current"] = matches["home"].replace(name_remap)
    matches["away_current"] = matches["away"].replace(name_remap)

    return matches

def load_starting_elos_from_database(league):
    """
    Retrieve starting Elo ratings for a given league from the database.

    Parameters
    ----------
    league : str
        League identifier appended to the starting Elo table name.

    Returns
    -------
    dict
        Mapping of club name (str) to Elo rating (float/int).
    """
    engine = db_connect.get_postgres_engine()
    elos_query = f"SELECT club, elo FROM {config.db_table_definitions['starting_elo_table']['name']}_{league}"
    starting_elos = pd.read_sql(elos_query, engine).set_index("club")["elo"].to_dict()

    return starting_elos

def save_elos_to_database(matches_elos, current_ratings, league):
    """
    Save Elo calculation results to the database.

    Args:
        matches_elos (pd.DataFrame): DataFrame with matches and elos
        current_ratings (list): DataFrame with current/latest elo for each team
        league (str): identifies the league name to output so that correct table is assigned
    """
    engine = db_connect.get_postgres_engine()
    table_suffix = league.lower()
    match_output_table = f"{config.db_table_definitions['historic_elo_table']['name']}_{table_suffix}"
    match_output_table_def = config.db_table_definitions["historic_elo_table"]["dtype"]
    matches_elos.to_sql(
        match_output_table,
        engine,
        if_exists="replace",
        index=False,
        dtype=match_output_table_def,
    )
    elo_output_table = f"{config.db_table_definitions['elo_table']['name']}_{table_suffix}"
    elo_output_table_def = config.db_table_definitions["elo_table"]["dtype"]
    current_ratings.to_sql(
        elo_output_table,
        engine,
        if_exists="replace",
        index=False,
        dtype=elo_output_table_def,
    )


def run_elo_calc():

    print("Loading historical matches")
    league = os.getenv("LEAGUES_TO_SIM")
    if league == "NFL":
        name_remap = config.nfl_name_remap
        starting_elos = config.nfl_starting_elos
    elif league == "NBA":
        name_remap = config.nba_name_remap
        starting_elos = config.nba_starting_elos
    elif league == "MLB":
        name_remap = config.mlb_name_remap
        starting_elos = config.mlb_starting_elos
    elif league == "UEFA":
        name_remap = {}
        starting_elos = load_starting_elos_from_database(league)
    else:
        raise(ValueError, "Invalid league")
    
    matches = load_matches_data(league, name_remap)
    elo_params = config.elo_params[league]
    elo_params['league'] = league

    print("Calculating Elos")
    # Initialize Elo Calculator
    elo_calculator = EloCalculator(matches=matches, elo_params=elo_params, starting_elos=starting_elos)

    # Calculate elos match by match
    elo_calculator.update_matches_elos()
    matches_elos = elo_calculator.matches
    matches_elos["updated_at"] = datetime.now()

    # Get current Elo ratings for each team
    current_ratings = elo_calculator.get_current_ratings()
    current_ratings["updated_at"] = datetime.now()

    print("Saving to DB")
    save_elos_to_database(matches_elos, current_ratings, league)


if __name__ == "__main__":
    run_elo_calc()