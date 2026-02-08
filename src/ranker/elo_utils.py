import pandas as pd
import math

# Elo calculator
class EloCalculator:
    def __init__(self, matches, elo_params, expansion_elos={}, initial_rating=1600):
        self.matches = matches
        self.ratings = {}
        self.initial_rating = initial_rating
        self.k = elo_params['elo_kfactor']
        self.season_start_adj = elo_params['season_start_adj']
        self.home_adv = elo_params['home_advantage']
        self.expansion_elos = expansion_elos
        self.league = elo_params['league']

    def get_rating(self, team):
        # Return current rating or initial rating if team not rated yet
        return self.ratings.get(team, self.expansion_elos.get(team, self.initial_rating))
    
    def adjust_season_start_elo(self):
        league_avg = sum(self.ratings.values()) / len(self.ratings)
        
        self.ratings = {
            team: (1 - self.season_start_adj) * rating + (self.season_start_adj) * league_avg
            for team, rating in self.ratings.items()
        }
        
    def calculate_elo(self, rating_a, rating_b, goals_a, goals_b, home_adv):

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
        
        def nba_mov_multiplier(winner_point_diff, winner_elo_diff):
            """
            Calculate the Margin of Victory (MOV) multiplier used in Elo ratings.
            
            Formula:
                MOV = (winner point differential + 3)^0.8 / (7.5 + 0.006 * winner Elo difference)
            """
            mov_mult = ((winner_point_diff + 3) ** 0.8) / (7.5 + 0.006 * winner_elo_diff)

            return mov_mult
        
        def mlb_mov_multiplier(winner_point_diff, winner_elo_diff):
            """
            Compute the MLB-specific margin-of-victory (MOV) multiplier for Elo updates.

            The multiplier scales the Elo change by comparing the adjusted run
            differential to the expected margin implied by the pre-game Elo
            difference, following FiveThirtyEight's MLB Elo methodology.

            Parameters
            ----------
            winner_point_diff : int or float
                Run differential for the winning team.
            winner_elo_diff : int or float
                Pre-game Elo difference (winner minus loser).

            Returns
            -------
            float
                Margin-of-victory multiplier applied to the base Elo update.
            """
            if winner_elo_diff is not None:
                adj_margin = ((winner_point_diff + 1)**0.7)*1.41
                exp_margin = ((winner_elo_diff)**3)*0.0000000546554876 + ((winner_elo_diff)**2)*0.00000896073139 + (winner_elo_diff)*0.00244895265 + 3.4

                mov_mult = adj_margin / max(exp_margin, 0.1)
            else:
                mov_mult = 1

            return mov_mult
        
        rating_a_hf = rating_a + home_adv
        # Calculate expected scores
        expected_a = 1 / (1 + 10 ** ((rating_b - rating_a_hf) / 400))
        expected_b = 1 - expected_a

        # Actual scores based on result
        if goals_a > goals_b:  # Home wins
            actual_a, actual_b = 1, 0
            winner_elo_diff = rating_a_hf - rating_b
        elif goals_a < goals_b:  # Away wins
            actual_a, actual_b = 0, 1
            winner_elo_diff = rating_b - rating_a_hf
        elif goals_a == goals_b:  # Tie
            actual_a, actual_b = 0.5, 0.5
            winner_elo_diff = None
        else:  # invalid
            raise ValueError("match result goals is invalid")

        # Update ratings
        winner_point_diff = abs(goals_a - goals_b)
        if self.league == "NFL":
            mov_multiplier = nfl_mov_multiplier(winner_point_diff, winner_elo_diff)
        elif self.league == "NBA":
            mov_multiplier = nba_mov_multiplier(winner_point_diff, winner_elo_diff)
        elif self.league == "MLB":
            mov_multiplier = mlb_mov_multiplier(winner_point_diff, winner_elo_diff)
        else:
            raise(ValueError, "Invalid league for Elo calc")
        new_rating_a = rating_a + self.k * mov_multiplier * (actual_a - expected_a)
        new_rating_b = rating_b + self.k * mov_multiplier * (actual_b - expected_b)

        return new_rating_a, new_rating_b, expected_a, expected_b

    def update_ratings(self, home_team, away_team, goals_a, goals_b, neutral):
        # Get current ratings
        home_rating = self.get_rating(home_team)
        away_rating = self.get_rating(away_team)

        if neutral == "Y":
            home_adv = 0
        elif neutral == "N":
            home_adv = self.home_adv
        else:
            raise(ValueError, "Invalid value for neutral column in matches df")

        # Calculate new ratings and win expectancies
        new_home_rating, new_away_rating, expected_home, expected_away = (
            self.calculate_elo(home_rating, away_rating, goals_a, goals_b, home_adv)
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
            prev_match_season = self.matches.at[max(0,index-1), "season"]
            new_match_season = self.matches.at[index, "season"]

            if prev_match_season != new_match_season:
                self.adjust_season_start_elo()

            (
                home_elo_before,
                away_elo_before,
                home_elo_after,
                away_elo_after,
                home_expectancy,
                away_expectancy,
            ) = self.update_ratings(match["home_current"], match["away_current"], match["home_goals"], match["away_goals"], match["neutral"])
            # Populate the DataFrame
            self.matches.at[index, "home_elo_before"] = home_elo_before
            self.matches.at[index, "away_elo_before"] = away_elo_before
            self.matches.at[index, "home_elo_after"] = home_elo_after
            self.matches.at[index, "away_elo_after"] = away_elo_after
            self.matches.at[index, "home_win_expectancy"] = home_expectancy
            self.matches.at[index, "away_win_expectancy"] = away_expectancy

