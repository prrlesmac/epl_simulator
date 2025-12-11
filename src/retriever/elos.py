import requests
import pandas as pd
from io import StringIO
from config import config
from db import db_connect
from datetime import datetime


def get_elos(url):
    """
    Fetch Elo rating data from a CSV URL with a 60-second timeout.

    Sends an HTTP GET request to the specified URL, reads the CSV content from 
    the response, and loads it into a Pandas DataFrame. The request is cancelled 
    automatically if no response is received within 60 seconds.

    Args:
        url (str): URL pointing to the CSV file containing Elo ratings.

    Returns:
        pandas.DataFrame or None: DataFrame containing Elo rating data if successful,
        otherwise None.

    Raises:
        requests.exceptions.RequestException: If an error occurs or request times out.
    """
    print("Fetching elos...")
    try:
        # Add a timeout of 60 seconds
        response = requests.get(url, timeout=60)

        # Check if the request was successful
        response.raise_for_status()  # Raises an error for non-200 responses
        return pd.read_csv(StringIO(response.text))

    except requests.exceptions.Timeout:
        print("Request timed out after 60 seconds.")
    except requests.exceptions.RequestException as e:
        print("An error occurred while fetching data:", e)

    return None


def filter_elos(elos, country, level):
    """
    Filters Elo ratings DataFrame by country and competition level.

    Standardizes column names to lowercase, filters the DataFrame based on the given
    country and level, and returns only the relevant columns.

    Args:
        elos (pandas.DataFrame): A DataFrame containing Elo ratings with at least
            'country', 'level', 'club', and 'elo' columns.
        country (str): The name of the country to filter by.
        level (int or str): The competition level to filter by.

    Returns:
        pandas.DataFrame: A filtered DataFrame containing only the 'club' and 'elo' columns
        for clubs that match the specified country and level.
    """
    elos.columns = elos.columns.str.lower()
    filtered_elos = elos.copy()
    if country is not None:
        filtered_elos = filtered_elos[(filtered_elos["country"] == country)]
    if level is not None:
        filtered_elos = filtered_elos[(filtered_elos["level"]) == level]
    filtered_elos = filtered_elos[["club", "country", "level", "elo"]]

    # correct red star to avoid duplicates
    filtered_elos.loc[
        (filtered_elos["club"] == "Red Star") & (filtered_elos["country"] == "FRA"),
        "club",
    ] = "Red Star FC"

    return filtered_elos


def main_elos():
    engine = db_connect.get_postgres_engine()
    elos = get_elos(config.elo_rating_url)
    if elos is None:
        raise RuntimeError("Elo fetch failed — aborting update.")

    elos = filter_elos(elos, None, None)
    elos["club"] = elos["club"].replace(config.club_name_mapping)
    elos["updated_at"] = datetime.now()
    elos.to_sql(
        f"{config.db_table_definitions['elo_table']['name']}_uefa",
        engine,
        if_exists="replace",
        index=False,
        dtype=config.db_table_definitions['elo_table']['dtype'],
    )
    print("Elos updated...")


if __name__ == "__main__":
    main_elos()
