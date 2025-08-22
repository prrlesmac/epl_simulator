from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
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

    time.sleep(10)
    # Send a GET request to fetch the HTML content
    response = requests.get(url)
    # Check if the request was successful
    if response.status_code == 200:
        df_all = parse_fixtures_html(response.text, table_id)
        print("Fetching fixtures data...")
        return df_all
    else:
        print(f"Failed to fetch the page. Status code: {response.status_code}")

    return None


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



def get_fixtures_selenium(url, table_id):
    """
    Fetch fixtures tables from a webpage using Selenium and parse them into a Pandas DataFrame.

    This function launches a headless Chrome browser using Selenium, loads the specified URL,
    extracts HTML tables identified by their IDs, and converts them into a concatenated
    Pandas DataFrame. Rows with empty "Home" and "Away" columns are removed, and the "xG" 
    column is dropped if present.

    Args:
        url (str): The URL of the webpage containing the fixtures tables.
        table_id (list[str]): A list of table IDs to locate and extract from the page.
        driver (selenium.webdriver.Chrome, optional): Existing WebDriver to reuse.
    Returns:
        pandas.DataFrame: A concatenated DataFrame containing all extracted fixtures data.

    Raises:
        selenium.common.exceptions.WebDriverException: If the browser driver cannot be started.
        AttributeError: If the expected table structure (thead/tbody) is not found.
    """
    options = Options()
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    options.add_argument('--disable-gpu')
    options.add_argument('--disable-extensions')
    options.add_argument('--window-size=1920,1080')
    
    #driver = None
    try:
        # Use webdriver-manager to handle ChromeDriver installation
        service = Service(ChromeDriverManager().install())
        driver = webdriver.Chrome(service=service, options=options)

        # Set timeouts
        driver.set_page_load_timeout(30)
        driver.implicitly_wait(10)

        print(f"Loading URL: {url}")
        driver.get(url)
        
        # Wait for the page to load properly instead of fixed sleep
        wait = WebDriverWait(driver, 20)
        wait.until(EC.presence_of_element_located((By.TAG_NAME, "table")))
        
        # Additional wait for Cloudflare if needed
        time.sleep(5)
        
        print("Getting page source...")
        html = driver.page_source
        
    except Exception as e:
        print(f"Error during web scraping: {e}")
        if driver:
            driver.quit()
        raise
    finally:
        # Ensure driver is always closed
        if driver:
            driver.quit()

    df_all = parse_fixtures_html(html, table_id)

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
        
        # Exclude rows where Home and Away are empty
        df = df[(df["Home"] != "") | (df["Away"] != "")]
        
        # Drop xG column if it exists
        if "xG" in df.columns:
            df = df.drop(columns=["xG"])

        df_all.append(df)
        print(f"Successfully processed table '{id}' with {len(df)} rows")
    
    if not df_all:
        raise ValueError("No valid tables found with the specified IDs")
        
    df_all = pd.concat(df_all, ignore_index=True)
    print(f"Fetching fixtures data complete. Total rows: {len(df_all)}")

    return df_all


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


def main_fixtures():
    engine = db_connect.get_postgres_engine()
    fixtures_all = []
    for k, v in config.fixtures_config.items():
        print("Getting fixtures for: ", k)
        if config.parsing_method == "http_request":
            fixtures = get_fixtures(v["fixtures_url"], v["table_id"])
        elif config.parsing_method == "local_file":
            fixtures = get_fixtures_local_file(v["local_file_path"], v["table_id"])
        elif config.parsing_method == "selenium":
            fixtures = get_fixtures_selenium(v["fixtures_url"], v["table_id"])
        else:
            raise(ValueError, "Invalid fixture parsing method")
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


if __name__ == "__main__":
    main_fixtures()
