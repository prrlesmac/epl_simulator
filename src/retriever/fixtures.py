import requests
import pandas as pd
from bs4 import BeautifulSoup
import numpy as np


def get_fixtures(url):

    # Send a GET request to fetch the HTML content
    response = requests.get(url)

    # Check if the request was successful
    if response.status_code == 200:
        # Parse the HTML content using BeautifulSoup
        soup = BeautifulSoup(response.text, "html.parser")

        # Find the table containing the fixtures
        table = soup.find(
            "table", {"id": "sched_2024-2025_9_1"}
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
        print(df)
    else:
        print(f"Failed to fetch the page. Status code: {response.status_code}")

    return df


def process_fixtures(fixtures):

    fixtures.columns = fixtures.columns.str.lower()
    fixtures = fixtures[(fixtures["home"] != "") & (fixtures["away"] != "")]
    fixtures["score"] = fixtures["score"].replace("", None)
    # Split the 'score' column into 'home_goals' and 'away_goals'
    fixtures[["home_goals", "away_goals"]] = fixtures["score"].str.split(
        "–", expand=True
    )
    fixtures["played"] = np.where(
        (fixtures["home_goals"].isnull()) | (fixtures["away_goals"].isnull()),
        "N",
        "Y",
    )

    fixtures = fixtures[["home", "away", "home_goals", "away_goals", "played"]]
    return fixtures


# Example Usage
if __name__ == "__main__":
    # Example scores DataFrame
    fixtures = get_fixtures(
        "https://fbref.com/en/comps/9/schedule/Premier-League-Scores-and-Fixtures"
    )
    fixtures = process_fixtures(fixtures)
    print(fixtures)
    fixtures.to_csv("../../data/01_raw/epl_matches.csv")
