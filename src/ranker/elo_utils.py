import pandas as pd
import math

# Elo calculator
class EloCalculator:
    def __init__(self, matches, elo_params, expansion_elos=None, initial_rating=1600):
        self.matches = matches
        self.ratings = {}
        self.initial_rating = initial_rating
        self.k = elo_params['elo_kfactor']
        self.season_start_adj = elo_params['season_start_adj']
        self.home_adv = elo_params['home_advantage']
        self.expansion_elos = expansion_elos

    def get_rating(self, team):
        # Return current rating or initial rating if team not rated yet
        return self.ratings.get(team, self.expansion_elos.get(team, self.initial_rating))
    
    def adjust_season_start_elo(self):
        league_avg = sum(self.ratings.values()) / len(self.ratings)
        
        self.ratings = {
            team: (1 - self.season_start_adj) * rating + (self.season_start_adj) * league_avg
            for team, rating in self.ratings.items()
        }
        
    def calculate_elo(self, rating_a, rating_b, goals_a, goals_b):

        def nfl_mov_multiplier(winner_point_diff, winner_elo_diff):
            """
            Calculate the Margin of Victory (MOV) multiplier used in Elo ratings.
            
            Formula:
                MOV = (ln(WinnerPointDiff + 1) * (2.2 / (WinnerEloDiff * 0.001 + 2.2)))
            """
            if winner_point_diff == 0:
                mov_mult = 1
            else:
                mov_mult = math.log(winner_point_diff + 1) * (2.2 / (winner_elo_diff * 0.001 + 2.2)) 
            return mov_mult
        
        # Calculate expected scores
        expected_a = 1 / (1 + 10 ** ((rating_b - rating_a) / 400))
        expected_b = 1 - expected_a

        # Actual scores based on result
        if goals_a > goals_b:  # Home wins
            actual_a, actual_b = 1, 0
            winner_elo_diff = rating_a - rating_b
        elif goals_a < goals_b:  # Away wins
            actual_a, actual_b = 0, 1
            winner_elo_diff = rating_b - rating_a
        elif goals_a == goals_b:  # Tie
            actual_a, actual_b = 0.5, 0.5
            winner_elo_diff = None
        else:  # invalid
            raise ValueError("match result goals is invalid")

        # Update ratings
        winner_point_diff = abs(goals_a - goals_b)
        mov_multiplier = nfl_mov_multiplier(winner_point_diff, winner_elo_diff)
        new_rating_a = rating_a + self.k * mov_multiplier * (actual_a - expected_a)
        new_rating_b = rating_b + self.k * mov_multiplier * (actual_b - expected_b)

        return new_rating_a, new_rating_b, expected_a, expected_b

    def update_ratings(self, home_team, away_team, goals_a, goals_b):
        # Get current ratings
        home_rating = self.get_rating(home_team)
        away_rating = self.get_rating(away_team)

        # Calculate new ratings and win expectancies
        new_home_rating, new_away_rating, expected_home, expected_away = (
            self.calculate_elo(home_rating, away_rating, goals_a, goals_b)
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
            ) = self.update_ratings(match["home_current"], match["away_current"], match["home_goals"], match["away_goals"])
            # Populate the DataFrame
            self.matches.at[index, "home_elo_before"] = home_elo_before
            self.matches.at[index, "away_elo_before"] = away_elo_before
            self.matches.at[index, "home_elo_after"] = home_elo_after
            self.matches.at[index, "away_elo_after"] = away_elo_after
            self.matches.at[index, "home_win_expectancy"] = home_expectancy
            self.matches.at[index, "away_win_expectancy"] = away_expectancy

            prev_match_season = self.matches.at[max(0,index-1), "season"]
            new_match_season = self.matches.at[index, "season"]

            if prev_match_season != new_match_season:
                self.adjust_season_start_elo()
