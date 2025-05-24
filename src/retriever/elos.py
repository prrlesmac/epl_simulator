import requests
import pandas as pd
from io import StringIO
from config import config
from db import get_postgres_engine


def get_elos(url):
    """
    Fetches Elo rating data from a CSV URL and returns it as a DataFrame.

    Sends an HTTP GET request to the specified URL, reads the CSV content from the response,
    and loads it into a Pandas DataFrame.

    Args:
        url (str): URL pointing to the CSV file containing Elo ratings.

    Returns:
        pandas.DataFrame: A DataFrame containing Elo rating data if the request is successful.
        If the request fails, an error message is printed and the return value may be undefined.

    Raises:
        requests.exceptions.RequestException: If an error occurs during the HTTP request.
    """

    try:
        # Make the GET request
        response = requests.get(url)

        # Check if the request was successful
        if response.status_code == 200:
            # Read the CSV data into a Pandas DataFrame
            csv_data = response.text
            df = pd.read_csv(StringIO(csv_data))

            # Display the first few rows of the DataFrame
            print("Fetching elos...")
        else:
            print(f"Failed to fetch data. Status code: {response.status_code}")
            print("Response:", response.text)
    except requests.exceptions.RequestException as e:
        print("An error occurred:", e)

    return df


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
    filtered_elos = elos[(elos["country"] == country) & (elos["level"]) == level]
    filtered_elos = filtered_elos[["club", "elo"]]

    return filtered_elos


if __name__ == "__main__":
    engine = get_postgres_engine()
    elos = get_elos(config.elo_rating_url)
    epl_elos = filter_elos(elos, "ENG", 1)
    epl_elos["club"] = epl_elos["club"].replace(config.club_name_mapping)
    epl_elos.to_sql('current_elos', engine, if_exists='replace', index=False)
