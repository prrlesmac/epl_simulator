import pandas as pd

# Elo calculator
class EloCalculator:
    def __init__(self, matches, initial_rating=1600, k=30):
        self.matches = matches
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
    
    def update_matches_elos(self):
        # Add columns to the DataFrame
        self.matches["home_elo_before"] = 0.0
        self.matches["away_elo_before"] = 0.0
        self.matches["home_elo_after"] = 0.0
        self.matches["away_elo_after"] = 0.0
        self.matches["home_win_expectancy"] = 0.0
        self.matches["away_win_expectancy"] = 0.0

        # Process matches and update the DataFrame
        for index, match in self.matches.iterrows():
            (
                home_elo_before,
                away_elo_before,
                home_elo_after,
                away_elo_after,
                home_expectancy,
                away_expectancy,
            ) = self.update_ratings(match["home"], match["away"], match["result"])
            # Populate the DataFrame
            self.matches.at[index, "home_elo_before"] = home_elo_before
            self.matches.at[index, "away_elo_before"] = away_elo_before
            self.matches.at[index, "home_elo_after"] = home_elo_after
            self.matches.at[index, "away_elo_after"] = away_elo_after
            self.matches.at[index, "home_win_expectancy"] = home_expectancy
            self.matches.at[index, "away_win_expectancy"] = away_expectancy
