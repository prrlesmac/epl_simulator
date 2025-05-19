import requests
import pandas as pd
from io import StringIO
import config


def get_elos(url):
    # Define the API endpoint

    try:
        # Make the GET request
        response = requests.get(url)

        # Check if the request was successful
        if response.status_code == 200:
            # Read the CSV data into a Pandas DataFrame
            csv_data = response.text
            df = pd.read_csv(StringIO(csv_data))

            # Display the first few rows of the DataFrame
            print(df.head())
        else:
            print(f"Failed to fetch data. Status code: {response.status_code}")
            print("Response:", response.text)
    except requests.exceptions.RequestException as e:
        print("An error occurred:", e)

    return df


def filter_elos(elos, country, level):
    elos.columns = elos.columns.str.lower()
    filtered_elos = elos[(elos["country"] == country) & (elos["level"]) == level]
    filtered_elos = filtered_elos[["club", "elo"]]
    return filtered_elos


# Example Usage
if __name__ == "__main__":
    # Example scores DataFrame
    elos = get_elos("http://api.clubelo.com/2025-04-01")
    epl_elos = filter_elos(elos, "ENG", 1)
    epl_elos["club"] = epl_elos["club"].replace(config.club_name_mapping)
    epl_elos.to_csv("data/02_intermediate/current_elo_ratings.csv")
