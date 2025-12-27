from playwright.sync_api import sync_playwright
import requests
import pandas as pd
from bs4 import BeautifulSoup
import re
import numpy as np
from config import config
from db import db_connect
from datetime import datetime
import time
import os
from dotenv import load_dotenv

load_dotenv()

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


def get_fixtures(url_list, table_id):
    """
    Fetches and parses a fixture table from a given webpage URL.

    Sends an HTTP GET request to the specified URL, parses the HTML to extract
    a table of fixtures using BeautifulSoup, and converts the table into a
    Pandas DataFrame. The function assumes the table has a specific ID
    (`sched_2024-2025_9_1`) and that the table structure includes a thead and tbody.

    Args:
        url_list (list): list of URLs of the webpages containing the fixture tables.
        table_id (list): list table IDs for geting the fixtures using beautiful soup

    Returns:
        pandas.DataFrame: A DataFrame containing the fixture information,
        with column headers extracted from the table. If the request fails,
        the function prints an error message and may return an undefined variable.

    Raises:
        requests.exceptions.RequestException: If the HTTP request fails.
        AttributeError: If the expected table or structure is not found in the HTML.
    """
    headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
            "Accept-Language": "en-US,en;q=0.9",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Referer": "https://www.google.com/"
    }
    df_all = []
    for url in url_list:
        print(url)
        time.sleep(10)
        # Send a GET request to fetch the HTML content
        response = requests.get(url, headers=headers)
        # Check if the request was successful
        if response.status_code == 200:
            df = parse_fixtures_html(response.text, table_id)
            df['url'] = url
            print("Fetching fixtures data...")
            df_all.append(df)
        else:
            print(f"Failed to fetch the page. Status code: {response.status_code}")
            return None
    df_all = pd.concat(df_all)

    return df_all

def get_fixtures_playwright(url, table_id):
    """
    Fetch and parse a fixture table from a webpage using Playwright

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
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)  # or headless=False to watch
        page = browser.new_page()

        # Mimic a real browser
        page.set_extra_http_headers({
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
            "Referer": "https://www.google.com/",
            "Accept-Language": "en-US,en;q=0.9"
        })

        page.goto(url, wait_until="domcontentloaded")
        time.sleep(10)  # Wait for Cloudflare challenge or dynamic content

        html = page.content()
        browser.close()

    # Now parse the table with BeautifulSoup or your existing parser
    df_all = parse_fixtures_html(html, table_id)
    print("Fixtures fetched via Playwright")
    return df_all


def get_fixtures_local_file(filepath, table_id):
    """
    Fetches and parses a fixture table from a given local HTML file.

    Reads a local HTML file, parses the HTML to extract
    a table of fixtures using BeautifulSoup, and converts the table into a
    Pandas DataFrame. The function assumes the table has a specific ID
    (`sched_2024-2025_9_1`) and that the table structure includes a thead and tbody.

    Args:
        filepath (str): The path of the file containing the fixtures table.
        table_id (list): list table IDs for geting the fixtures using beautiful soup

    Returns:
        pandas.DataFrame: A DataFrame containing the fixture information,
        with column headers extracted from the table. If the request fails,
        the function prints an error message and may return an undefined variable.
    """

    with open(filepath, "r", encoding="utf-8") as html_file:
        html_content = html_file.read()

    df_all = parse_fixtures_html(html_content, table_id)
    print("Fetching fixtures data...")
    
    return df_all


def parse_fixtures_html(html, table_id):
    """
    Parse HTML content to extract fixtures data from specified tables and return as a DataFrame.

    This function takes raw HTML content and extracts data from tables identified by their IDs.
    It processes each table by parsing headers and rows, combines them into individual DataFrames,
    and concatenates all tables into a single DataFrame. Rows with empty "Home" and "Away" 
    columns are filtered out, and the "xG" column is dropped if present.

    Args:
        html (str): Raw HTML content containing the tables to be parsed.
        table_id (list[str]): A list of table IDs to locate and extract from the HTML.

    Returns:
        pandas.DataFrame: A concatenated DataFrame containing all extracted fixtures data 
                         with empty Home/Away rows removed and xG column dropped.

    Raises:
        ValueError: If no valid tables are found with the specified IDs.
        AttributeError: If the expected table structure (thead/tbody) is not found.
    """
    # Parse the HTML content using BeautifulSoup
    soup = BeautifulSoup(html, "html.parser")
    
    # Find the tables containing the fixtures
    df_all = []
    for id in table_id:
        print(f"Processing table with ID: {id}")
        table = soup.find("table", {"id": id})
        
        if not table:
            print(f"Warning: Table with ID '{id}' not found")
            continue
            
        # Check if table has the expected structure
        thead = table.find("thead")
        tbody = table.find("tbody")
        
        if not thead or not tbody:
            print(f"Warning: Table '{id}' missing thead or tbody")
            continue

        # Extract table headers
        headers = [th.text.strip() for th in thead.find_all("th")]

        # Extract table rows
        rows = []
        for tr in tbody.find_all("tr"):
            row_th = [tr.find("th").text.strip()] if tr.find("th") else [""]
            row_td = [td.text.strip() for td in tr.find_all("td")]
            row = row_th + row_td
            rows.append(row)
            
        if not rows:
            print(f"Warning: No data rows found in table '{id}'")
            continue
            
        # Convert to a Pandas DataFrame
        df = pd.DataFrame(rows, columns=headers)
        
        # Exclude the first header if it's a placeholder
        # exclude rows where Home and Away are empty
        # for ucl tables
        if all(col in df.columns for col in ["Home", "Away","xG"]):
            df = df[(df["Home"] != "") | (df["Away"] != "")]
            df = df.drop(columns=["xG"])
        df_all.append(df)

        print(f"Successfully processed table '{id}' with {len(df)} rows")
    
    if not df_all:
        raise ValueError("No valid tables found with the specified IDs")
        
    df_all = pd.concat(df_all, ignore_index=True)
    print(f"Fetching fixtures data complete. Total rows: {len(df_all)}")

    return df_all


def get_fixtures_text(url_list):
    """
    Scrape baseball game data from HTML and return a pandas DataFrame.
    
    Args:
        url_list (str): list of URLs to parse
    
    Returns:
        pd.DataFrame: DataFrame with columns: date, home_team, away_team, runs_home, runs_away
    """
    df_all = []
    for url in url_list:
        # Send a GET request to fetch the HTML content
        time.sleep(10)
        response = requests.get(url)
        # Check if the request was successful
        if response.status_code == 200:
            # Parse the HTML content using BeautifulSoup
            soup = BeautifulSoup(response.text, "html.parser")
    
            games_data = []
            current_date = None
            # Find all h3 elements (dates) and game elements
            for element in soup.find_all(['h2','h3', 'p']):
                if element.name == 'h2':
                    schedule = element.get_text(strip=True)
                    current_stage = "League" if schedule == "MLB Schedule" else "Playoff" if schedule == "Postseason Schedule" else ""
                if element.name == 'h3':
                    # Extract date from h3 element
                    current_date = element.get_text(strip=True)
                elif element.name == 'p' and 'game' in element.get('class', []):
                    # Parse game data
                    if current_date:
                        game_data = parse_game_element(element, current_date, current_stage)
                        if game_data:
                            games_data.append(game_data)
            df = pd.DataFrame(games_data)
            df_all.append(df)
            print("Fetching fixtures data...")
        else:
            print(f"Failed to fetch the page. Status code: {response.status_code}")
            return            

    df_all = pd.concat(df_all)
    return df_all


def parse_game_element(game_element, date, round):
    """
    Parse individual game element to extract team names and scores.
    
    Args:
        game_element: BeautifulSoup element containing game data
        date (str): Date of the game
        round (str): stage of the game
    
    Returns:
        dict: Game data dictionary or None if parsing fails
    """
    try:
        text = game_element.get_text()
        
        # Find all team links
        team_links = game_element.find_all('a', href=lambda x: x and '/teams/' in x)
        
        if len(team_links) < 2:
            return None
            
        # Extract team names (remove year from link text)
        teams = []
        for link in team_links[:2]:  # Only take first 2 team links
            team_name = link.get_text(strip=True)
            teams.append(team_name)
        
        # Extract scores using regex
        # Pattern looks for numbers in parentheses
        scores = re.findall(r'\((\d+)\)', text)
        
        if len(teams) != 2 or len(scores) != 2:
            return None
        
        # Determine home/away based on @ symbol and strong tag (winner indication)
        away_team = teams[0]
        home_team = teams[1]
        away_goals = int(scores[0])
        home_goals = int(scores[1])
        
        return {
            'round': round,
            'date': date,
            'away': away_team,
            'home': home_team,
            'away_goals': away_goals,
            'home_goals': home_goals
        }
        
    except Exception as e:
        print(f"Error parsing game element: {e}")
        return None


def process_footy_table(fixtures, country):
    """
    Cleans and enriches a football fixtures DataFrame.

    - Removes rows with missing team names.
    - Parses the 'score' column into goal and penalty columns.
    - Adds a 'played' flag based on whether scores are available.
    - Optionally strips country codes from team names for European competitions.
    - Ensures presence of 'neutral' and 'round' columns with default values.

    Parameters:
        fixtures (pd.DataFrame): Match data with at least 'home', 'away', and 'score' columns.
        country (str): League or competition code (e.g., 'UCL') used for team name formatting.

    Returns:
        pd.DataFrame: Processed fixtures table.
    """
    fixtures = fixtures[(fixtures["home"] != "") & (fixtures["away"] != "")].copy()
    fixtures["score"] = fixtures["score"].replace("", None)
    fixtures["season"] = ""
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

    return fixtures

def process_nfl_table_legacy(fixtures):
    """
    Cleans and formats an NFL fixtures DataFrame for analysis.

    - Renames and standardizes key columns (teams, scores, dates, rounds).
    - Filters out incomplete or preseason games.
    - Extracts season year from URLs and adjusts for games played in Jan/Feb.
    - Parses full datetime from textual date and season year.
    - Adds flags for whether the game was played and whether it was at a neutral venue (e.g., Super Bowl).
    - Adds an empty 'notes' column for future use.

    Parameters:
        fixtures (pd.DataFrame): Raw NFL game data with columns like 'hometm', 'vistm', 'week', 'date', 'url', etc.

    Returns:
        pd.DataFrame: Cleaned and enriched fixtures DataFrame.
    """
    fixtures.columns.values[2] = "date"
    fixtures.columns.values[4] = "away_goals"
    fixtures.columns.values[7] = "home_goals"

    fixtures = fixtures.rename(columns = {
        "vistm": "away",
        "hometm": "home",
        "week": "round",
    })
    fixtures = fixtures[(~fixtures["away"].isnull()) & (~fixtures["home"].isnull())].copy()
    fixtures = fixtures[~fixtures["round"].str.startswith("Pre")].copy()
    fixtures["season"] = fixtures["url"].str.extract(r"/years/(\d{4})/")  
    fixtures["year"] = fixtures.apply(
        lambda row: str(int(row["season"]) + 1) if row["date"].lower().startswith(("jan", "feb")) else row["season"],
        axis=1
    )   
    fixtures["date"] = pd.to_datetime(fixtures["date"] + " " + fixtures["year"])
    fixtures["home_goals"] = pd.to_numeric(fixtures["home_goals"].replace("", pd.NA), errors="coerce").astype("Int64")
    fixtures["away_goals"] = pd.to_numeric(fixtures["away_goals"].replace("", pd.NA), errors="coerce").astype("Int64")

    fixtures["round"] = np.where(
        fixtures["round"].isin(["WildCard","Division","ConfChamp","SuperBowl"]),
        fixtures["round"],
        "League"
    )

    fixtures["played"] = np.where(
        (fixtures["home_goals"].isnull()) | (fixtures["away_goals"].isnull()),
        "N",
        "Y",
    )
    fixtures["neutral"] = np.where(
        fixtures["round"]=="SuperBowl",
        "Y",
        "N"
    )
    fixtures["notes"] = ""

    return fixtures



def process_nfl_table(fixtures):
    """
    Cleans and formats an NFL fixtures DataFrame for analysis.

    - Renames and standardizes key columns (teams, scores, dates, rounds).
    - Filters out incomplete or preseason games.
    - Extracts season year from URLs and adjusts for games played in Jan/Feb.
    - Parses full datetime from textual date and season year.
    - Adds flags for whether the game was played and whether it was at a neutral venue (e.g., Super Bowl).
    - Adds an empty 'notes' column for future use.

    Parameters:
        fixtures (pd.DataFrame): Raw NFL game data with columns like 'hometm', 'vistm', 'week', 'date', 'url', etc.

    Returns:
        pd.DataFrame: Cleaned and enriched fixtures DataFrame.
    """
    fixtures.columns.values[2] = "date"
    fixtures.columns.values[5] = "home_or_away"

    fixtures["home"] = np.where(
        fixtures["home_or_away"] == "@",
        fixtures["loser/tie"],
        fixtures["winner/tie"]
    )
    fixtures["away"] = np.where(
        fixtures["home_or_away"] == "@",
        fixtures["winner/tie"],
        fixtures["loser/tie"]
    )
    fixtures["home_goals"] = np.where(
        fixtures["home_or_away"] == "@",
        fixtures["ptsl"],
        fixtures["ptsw"]
    )
    fixtures["away_goals"] = np.where(
        fixtures["home_or_away"] == "@",
        fixtures["ptsw"],
        fixtures["ptsl"]
    )
    fixtures = fixtures.rename(columns = {
        "week": "round",
    })
    fixtures = fixtures[(~fixtures["away"].isnull()) & (~fixtures["home"].isnull())].copy()
    fixtures = fixtures[(fixtures["away"]!="") & (fixtures["home"]!="")].copy()
    fixtures = fixtures[~fixtures["round"].str.startswith("Pre")].copy()
    fixtures["season"] = fixtures["url"].str.extract(r"/years/(\d{4})/")  
    fixtures["home_goals"] = pd.to_numeric(fixtures["home_goals"].replace("", pd.NA), errors="coerce").astype("Int64")
    fixtures["away_goals"] = pd.to_numeric(fixtures["away_goals"].replace("", pd.NA), errors="coerce").astype("Int64")

    fixtures["round"] = np.where(
        fixtures["round"].isin(["WildCard","Division","ConfChamp","SuperBowl"]),
        fixtures["round"],
        "League"
    )

    fixtures["played"] = np.where(
        (fixtures["home_goals"].isnull()) | (fixtures["away_goals"].isnull()),
        "N",
        "Y",
    )
    fixtures["neutral"] = np.where(
        fixtures["round"]=="SuperBowl",
        "Y",
        "N"
    )
    fixtures["notes"] = ""

    return fixtures


def process_nba_table(fixtures):
    """
    Cleans and standardizes an NBA fixtures DataFrame.

    - Renames and maps key columns for teams and scores.
    - Filters out rows with missing team names.
    - Converts the 'date' column to datetime format.
    - Normalizes missing score values and adds a 'played' flag.
    - Adds default 'neutral', 'round', and 'notes' columns.

    Parameters:
        fixtures (pd.DataFrame): Raw NBA game data with columns such as 'visitor/neutral', 'home/neutral', 'date', and scores.

    Returns:
        pd.DataFrame: Cleaned and enriched fixtures DataFrame.
    """
    fixtures.columns.values[3] = "away_goals"
    fixtures.columns.values[5] = "home_goals"

    fixtures = fixtures.rename(columns = {
        "visitor/neutral": "away",
        "home/neutral": "home",
    })
    fixtures = fixtures[(~fixtures["away"].isnull()) & (~fixtures["home"].isnull())].copy()
    fixtures["date"] = pd.to_datetime(fixtures["date"], format='%a, %b %d, %Y')
    fixtures["season"] = fixtures["url"].str.extract(r"/leagues/NBA_(\d{4})_")  
    fixtures["home_goals"] = pd.to_numeric(fixtures["home_goals"].replace("", pd.NA), errors="coerce").astype("Int64")
    fixtures["away_goals"] = pd.to_numeric(fixtures["away_goals"].replace("", pd.NA), errors="coerce").astype("Int64")

    play_in_df = fixtures[fixtures["notes"]=="Play-In Game"]
    if len(play_in_df) > 0:
        play_in_date_start = play_in_df["date"].min()
        play_in_date_end = play_in_df["date"].max()
    else:
        play_in_date_start = fixtures["date"].max() + pd.Timedelta(days=1)
        play_in_date_end = fixtures["date"].max() + pd.Timedelta(days=1)

    fixtures["round"] = np.where(
        fixtures["date"] < play_in_date_start,
        "League",
        np.where(
            fixtures["date"] > play_in_date_end,
            "Playoff",
            "Play-in"
        )
    )

    fixtures["played"] = np.where(
        (fixtures["home_goals"].isnull()) | (fixtures["away_goals"].isnull()),
        "N",
        "Y",
    )
    fixtures["neutral"] = "N"

    return fixtures


def post_process_nba_table(fixtures):
    """
    Post-processes an NBA fixtures DataFrame by labeling the NBA Cup Final.

    This function searches for all rows where the `"notes"` column equals
    `"NBA Cup"`, identifies the 67th occurrence, and sets its `"round"` value
    to `"NBA Cup Final"`. This is based on the structure of the NBA In-Season
    Tournament schedule, where the 67th NBA Cup game corresponds to the final.

    Parameters
    ----------
    fixtures : pandas.DataFrame
        A DataFrame containing NBA fixtures with at least the columns
        `"notes"` and `"round"`.

    Returns
    -------
    pandas.DataFrame
        The updated DataFrame with the NBA Cup Final correctly labeled.
    """
    #flag NBA Cup Final as 67th game
    fixtures = fixtures.reset_index()
    mask = fixtures["notes"] == "NBA Cup"
    matches = fixtures.index[mask]
    if len(matches) >= 67:
        fixtures.loc[matches[66], "round"] = "NBA Cup Final"

    return fixtures


def process_mlb_table(fixtures):
    """
    Cleans and standardizes an MLB fixtures DataFrame.

    - Filters out rows with missing team names.
    - Converts the 'date' column to datetime format.
    - Normalizes missing score values and adds a 'played' flag.
    - Adds default values for 'neutral', 'round', and 'notes' columns.

    Parameters:
        fixtures (pd.DataFrame): Raw MLB game data with columns like 'home', 'away', 'date', 'home_goals', and 'away_goals'.

    Returns:
        pd.DataFrame: Cleaned and structured fixtures DataFrame.
    """
    fixtures = fixtures[(~fixtures["away"].isnull()) & (~fixtures["home"].isnull())].copy()
    fixtures["date"] = pd.to_datetime(fixtures["date"], format='%A, %B %d, %Y')
    fixtures["season"] = fixtures["url"].str.extract(r"/leagues/majors/(\d{4})-")
    fixtures["home_goals"] = pd.to_numeric(fixtures["home_goals"].replace("", pd.NA), errors="coerce").astype("Int64")
    fixtures["away_goals"] = pd.to_numeric(fixtures["away_goals"].replace("", pd.NA), errors="coerce").astype("Int64")

    fixtures["played"] = np.where(
        (fixtures["home_goals"].isnull()) | (fixtures["away_goals"].isnull()),
        "N",
        "Y",
    )
    fixtures["neutral"] = "N"
    fixtures["notes"] = ""

    return fixtures

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
    is_footy_league = country in ["ENG","ESP","GER","FRA","ITA","UCL","UEL","UECL"]
    fixtures.columns = fixtures.columns.str.lower()
    if is_footy_league:
        fixtures = process_footy_table(fixtures, country)
    elif country=="NFL":
        fixtures = process_nfl_table(fixtures)
    elif country=="NBA":
        fixtures = process_nba_table(fixtures)
    elif country=="MLB":
        fixtures = process_mlb_table(fixtures)
    else:
        raise ValueError("Invalid league to sim selected")

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
            "season",
            "notes",
        ]
    ]
    return fixtures


def main_fixtures():
    fixtures_all = []
    league_name = os.getenv("LEAGUES_TO_SIM")
    leagues_to_sim = config.active_uefa_leagues if league_name == "UEFA" else [league_name]
    if config.pull_fixture_history:
        leagues_config = {k: v for k, v in config.fixtures_history_config.items() if k in leagues_to_sim}
    else:
        leagues_config = {k: v for k, v in config.fixtures_config.items() if k in leagues_to_sim}
    for k, v in leagues_config.items():
        print("Getting fixtures for: ", k)
        if k == "MLB":
            fixtures = get_fixtures_text(v["fixtures_url"])
        else:
            if config.parsing_method == "http_request":
                fixtures = get_fixtures(v["fixtures_url"], v["table_id"])
            elif config.parsing_method == "local_file":
                fixtures = get_fixtures_local_file(v["local_file_path"], v["table_id"])
            elif config.parsing_method == "playwright":
                fixtures = get_fixtures_playwright(v["fixtures_url"], v["table_id"])
            else:
                raise(ValueError, "Invalid fixture parsing method")
        fixtures = process_fixtures(fixtures, country=k)
        fixtures["country"] = k
        fixtures["updated_at"] = datetime.now()
        fixtures_all.append(fixtures)
    fixtures_all = pd.concat(fixtures_all)
    if league_name == "NBA":
        fixtures_all = post_process_nba_table(fixtures_all)
    table_name = f"{config.db_table_definitions['fixtures_table']['name']}_{league_name.lower()}{'_history' if config.pull_fixture_history else ''}"
    engine = db_connect.get_postgres_engine()
    fixtures_all.to_sql(
        table_name,
        engine,
        if_exists="replace",
        index=False,
        dtype=config.db_table_definitions["fixtures_table"]["dtype"],
    )
    print("Fixtures updated")


if __name__ == "__main__":
    main_fixtures()
