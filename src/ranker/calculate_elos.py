import pandas as pd
import numpy as np
from db import db_connect

engine = db_connect.get_postgres_engine()
matches = pd.read_sql(
    """
    SELECT * FROM public.fixtures_nfl_history
    UNION ALL
    SELECT * FROM public.fixtures_nfl
    WHERE played='Y'
    """,
    engine,
)
matches["result"] = np.select(
    [
        matches["home_goals"] > matches["away_goals"],
        matches["home_goals"] < matches["away_goals"]
    ],
    [
        "H",  # Home win
        "A"   # Away win
    ],
    default="T"  # Tie
)

# Elo calculator
class EloCalculator:
    def __init__(self, initial_rating=1600, k=30):
        self.ratings = {}
        self.initial_rating = initial_rating
        self.k = k

    def get_rating(self, team):
        # Return current rating or initial rating if team not rated yet
        return self.ratings.get(team, self.initial_rating)

    def calculate_elo(self, rating_a, rating_b, result):
        # Calculate expected scores
        expected_a = 1 / (1 + 10 ** ((rating_b - rating_a) / 400))
        expected_b = 1 - expected_a

        # Actual scores based on result
        if result == "H":  # Home wins
            actual_a, actual_b = 1, 0
        elif result == "A":  # Away wins
            actual_a, actual_b = 0, 1
        elif result == "T":  # Tie
            actual_a, actual_b = 0.5, 0.5
        else:  # invalid
            raise ValueError("match result must be H, A or T")

        # Update ratings
        new_rating_a = rating_a + self.k * (actual_a - expected_a)
        new_rating_b = rating_b + self.k * (actual_b - expected_b)
        return new_rating_a, new_rating_b, expected_a, expected_b

    def update_ratings(self, home_team, away_team, result):
        # Get current ratings
        home_rating = self.get_rating(home_team)
        away_rating = self.get_rating(away_team)

        # Calculate new ratings and win expectancies
        new_home_rating, new_away_rating, expected_home, expected_away = (
            self.calculate_elo(home_rating, away_rating, result)
        )

        # Update ratings
        self.ratings[home_team] = new_home_rating
        self.ratings[away_team] = new_away_rating

        # Return ratings and win expectancies
        return (
            home_rating,
            away_rating,
            new_home_rating,
            new_away_rating,
            expected_home,
            expected_away,
        )

    def get_current_ratings(self):
        # Return a DataFrame of current Elo ratings
        return pd.DataFrame(
            self.ratings.items(), columns=["club", "elo"]
        ).sort_values(by="elo", ascending=False)


# Initialize Elo Calculator
elo_calculator = EloCalculator()

# Add columns to the DataFrame
matches["home_elo_before"] = 0.0
matches["away_elo_before"] = 0.0
matches["home_elo_after"] = 0.0
matches["away_elo_after"] = 0.0
matches["home_win_expectancy"] = 0.0
matches["away_win_expectancy"] = 0.0

# Process matches and update the DataFrame
for index, match in matches.iterrows():
    (
        home_elo_before,
        away_elo_before,
        home_elo_after,
        away_elo_after,
        home_expectancy,
        away_expectancy,
    ) = elo_calculator.update_ratings(match["home"], match["away"], match["result"])

    # Populate the DataFrame
    matches.at[index, "home_elo_before"] = home_elo_before
    matches.at[index, "away_elo_before"] = away_elo_before
    matches.at[index, "home_elo_after"] = home_elo_after
    matches.at[index, "away_elo_after"] = away_elo_after
    matches.at[index, "home_win_expectancy"] = home_expectancy
    matches.at[index, "away_win_expectancy"] = away_expectancy

# Get current Elo ratings for each team
current_ratings = elo_calculator.get_current_ratings()

matches.to_sql(
    "historic_elos_nfl",
    engine,
    if_exists="replace",
    index=False,
    dtype=None,
)
current_ratings.to_sql(
    "current_elos_nfl",
    engine,
    if_exists="replace",
    index=False,
    dtype=None,
)