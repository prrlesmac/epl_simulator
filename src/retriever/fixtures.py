import requests
import pandas as pd
from bs4 import BeautifulSoup
import numpy as np
from config import config
from db import db_connect
from datetime import datetime
import time


def get_fixtures(url, table_id):
    """
    Fetches and parses a fixture table from a given webpage URL.

    Sends an HTTP GET request to the specified URL, parses the HTML to extract
    a table of fixtures using BeautifulSoup, and converts the table into a
    Pandas DataFrame. The function assumes the table has a specific ID
    (`sched_2024-2025_9_1`) and that the table structure includes a thead and tbody.

    Args:
        url (str): The URL of the webpage containing the fixture table.
        table_id (str): table ID for geting the fixtures using beautiful soup

    Returns:
        pandas.DataFrame: A DataFrame containing the fixture information,
        with column headers extracted from the table. If the request fails,
        the function prints an error message and may return an undefined variable.

    Raises:
        requests.exceptions.RequestException: If the HTTP request fails.
        AttributeError: If the expected table or structure is not found in the HTML.
    """

    time.sleep(5)
    # Send a GET request to fetch the HTML content
    response = requests.get(url)

    # Check if the request was successful
    if response.status_code == 200:
        # Parse the HTML content using BeautifulSoup
        soup = BeautifulSoup(response.text, "html.parser")

        # Find the table containing the fixtures
        table = soup.find(
            "table", {"id": table_id}
        )  # Table ID may change, inspect the page source to confirm

        # Extract table headers
        headers = [th.text.strip() for th in table.find("thead").find_all("th")]

        # Extract table rows
        rows = []
        for tr in table.find("tbody").find_all("tr"):
            row = [td.text.strip() for td in tr.find_all("td")]
            rows.append(row)

        # Convert to a Pandas DataFrame
        df = pd.DataFrame(
            rows, columns=headers[1:]
        )  # Exclude the first header if it's a placeholder

        # Display the DataFrame
        print("Fetching fixtures data...")
    else:
        print(f"Failed to fetch the page. Status code: {response.status_code}")

    return df


def process_fixtures(fixtures):
    """
    Cleans and processes a raw fixtures DataFrame to extract match outcomes.

    Standardizes column names, filters out incomplete rows, splits the score into
    home and away goals, and marks whether each fixture has been played.

    Args:
        fixtures (pandas.DataFrame): Raw fixtures DataFrame, typically parsed from HTML,
        with columns including 'home', 'away', and 'score'.

    Returns:
        pandas.DataFrame: A cleaned and processed DataFrame with the following columns:
            - 'home': Home team name
            - 'away': Away team name
            - 'home_goals': Goals scored by the home team (as string or NaN)
            - 'away_goals': Goals scored by the away team (as string or NaN)
            - 'played': 'Y' if the match has been played, otherwise 'N'
    """

    fixtures.columns = fixtures.columns.str.lower()
    fixtures = fixtures[(fixtures["home"] != "") & (fixtures["away"] != "")]
    fixtures["score"] = fixtures["score"].replace("", None)
    # Split the 'score' column into 'home_goals' and 'away_goals'
    fixtures[["home_goals", "away_goals"]] = (
        fixtures["score"].str.split("–", expand=True).astype("Int64")
    )
    fixtures["played"] = np.where(
        (fixtures["home_goals"].isnull()) | (fixtures["away_goals"].isnull()),
        "N",
        "Y",
    )

    fixtures = fixtures[["home", "away", "home_goals", "away_goals", "played"]]
    return fixtures


if __name__ == "__main__":
    engine = db_connect.get_postgres_engine()
    fixtures_all = []
    for k, v in config.fixtures_config.items():
        print("Getting fixtures for: ", k)
        fixtures = get_fixtures(v["fixtures_url"], v["table_id"])
        fixtures = process_fixtures(fixtures)
        fixtures["country"] = k
        fixtures["updated_at"] = datetime.now()
        fixtures_all.append(fixtures)
    fixtures_all = pd.concat(fixtures_all)
    breakpoint()
    fixtures_all.to_sql(
        config.fixtures_table["name"],
        engine,
        if_exists="replace",
        index=False,
        dtype=config.fixtures_table["dtype"],
    )
    print("Fixtures updated")
