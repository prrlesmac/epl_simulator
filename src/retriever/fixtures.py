import requests
import pandas as pd
from bs4 import BeautifulSoup
import re
import numpy as np
from config import config
from db import db_connect
from datetime import datetime
import time


def extract_scores(score):
    if pd.isna(score):
        return pd.Series([None, None, None, None])

    match_goals = re.search(r"(\d+)–(\d+)", score)
    home_goals, away_goals = (
        (int(match_goals.group(1)), int(match_goals.group(2)))
        if match_goals
        else (None, None)
    )

    pens = re.findall(r"\((\d+)\)", score)
    if len(pens) == 2:
        home_pens, away_pens = int(pens[0]), int(pens[1])
    else:
        home_pens, away_pens = None, None

    return pd.Series([home_goals, away_goals, home_pens, away_pens])


def get_fixtures(url, table_id):
    """
    Fetches and parses a fixture table from a given webpage URL.

    Sends an HTTP GET request to the specified URL, parses the HTML to extract
    a table of fixtures using BeautifulSoup, and converts the table into a
    Pandas DataFrame. The function assumes the table has a specific ID
    (`sched_2024-2025_9_1`) and that the table structure includes a thead and tbody.

    Args:
        url (str): The URL of the webpage containing the fixture table.
        table_id (list): list table IDs for geting the fixtures using beautiful soup

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
        df_all = []
        for id in table_id:
            table = soup.find("table", {"id": id})

            # Extract table headers
            headers = [th.text.strip() for th in table.find("thead").find_all("th")]

            # Extract table rows
            rows = []
            for tr in table.find("tbody").find_all("tr"):
                row_th = [tr.find("th").text.strip()] if tr.find("th") else [""]
                row_td = [td.text.strip() for td in tr.find_all("td")]
                row = row_th + row_td
                rows.append(row)
            # Convert to a Pandas DataFrame
            df = pd.DataFrame(
                rows, columns=headers
            )  # Exclude the first header if it's a placeholder
            # exclude rows where Home and Away are empty
            df = df[(df["Home"] != "") | (df["Away"] != "")]
            df = df.drop(columns=["xG"])

            df_all.append(df)
        df_all = pd.concat(df_all)
        # Display the DataFrame
        print("Fetching fixtures data...")
        return df_all
    else:
        print(f"Failed to fetch the page. Status code: {response.status_code}")

    return None


def process_fixtures(fixtures, country):
    """
    Cleans and processes a raw fixtures DataFrame to extract match outcomes.

    Standardizes column names, filters out incomplete rows, splits the score into
    home and away goals, and marks whether each fixture has been played.

    Args:
        fixtures (pandas.DataFrame): Raw fixtures DataFrame, typically parsed from HTML,
        with columns including 'home', 'away', and 'score'.
        country (str): The country or league identifier to adjust team names and processing.

    Returns:
        pandas.DataFrame: A cleaned and processed DataFrame with the following columns:
            - 'home': Home team name
            - 'away': Away team name
            - 'home_goals': Goals scored by the home team (as string or NaN)
            - 'away_goals': Goals scored by the away team (as string or NaN)
            - 'played': 'Y' if the match has been played, otherwise 'N'
            - 'neutral': 'Y' if the match is neutral venue, otherwise 'N'
    """

    fixtures.columns = fixtures.columns.str.lower()
    fixtures = fixtures[(fixtures["home"] != "") & (fixtures["away"] != "")].copy()
    fixtures["score"] = fixtures["score"].replace("", None)
    # Apply function to the 'score' column and expand results into new columns
    fixtures[["home_goals", "away_goals", "home_pens", "away_pens"]] = fixtures[
        "score"
    ].apply(extract_scores)
    fixtures["played"] = np.where(
        (fixtures["home_goals"].isnull()) | (fixtures["away_goals"].isnull()),
        "N",
        "Y",
    )
    # remove country from team names if the league is european
    if country in ["UCL", "UEL", "UECL"]:
        fixtures["home"] = fixtures["home"].str[:-3]
        fixtures["away"] = fixtures["away"].str[3:]
    fixtures["home"] = fixtures["home"].str.strip()
    fixtures["away"] = fixtures["away"].str.strip()

    fixtures["neutral"] = "N"
    if "round" not in fixtures.columns:
        fixtures["round"] = "League"
    fixtures["round"] = fixtures["round"].fillna("League")
    fixtures = fixtures[
        [
            "home",
            "away",
            "home_goals",
            "away_goals",
            "played",
            "neutral",
            "round",
            "date",
            "notes",
        ]
    ]
    return fixtures


if __name__ == "__main__":
    engine = db_connect.get_postgres_engine()
    fixtures_all = []
    for k, v in config.fixtures_config.items():
        print("Getting fixtures for: ", k)
        fixtures = get_fixtures(v["fixtures_url"], v["table_id"])
        fixtures = process_fixtures(fixtures, country=k)
        fixtures["country"] = k
        fixtures["updated_at"] = datetime.now()
        fixtures_all.append(fixtures)
    fixtures_all = pd.concat(fixtures_all)
    fixtures_all.to_sql(
        config.db_table_definitions["fixtures_table"]["name"],
        engine,
        if_exists="replace",
        index=False,
        dtype=config.db_table_definitions["fixtures_table"]["dtype"],
    )
    print("Fixtures updated")
