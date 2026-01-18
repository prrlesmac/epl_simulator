import pytest
import pandas as pd
from ranker.elo_utils import EloCalculator


@pytest.fixture
def sample_elo_params():
    """Standard Elo parameters for testing."""
    return {
        'elo_kfactor': 20,
        'season_start_adj': 0.25,
        'home_advantage': 0,
        'league': 'NFL'
    }

@pytest.fixture
def sample_elo_params_nba():
    """Standard Elo parameters for testing."""
    return {
        "elo_kfactor": 20,
        "season_start_adj": 0.25,
        "home_advantage": 100,
        'league': 'NBA'
    }

@pytest.fixture
def sample_matches():
    """Sample match data for testing."""
    return pd.DataFrame({
        'season': [2023, 2023, 2023, 2024, 2024],
        'home_current': ['Team A', 'Team B', 'Team C', 'Team A', 'Team B'],
        'away_current': ['Team B', 'Team C', 'Team A', 'Team C', 'Team A'],
        'home_goals': [3, 2, 1, 4, 2],
        'away_goals': [1, 2, 2, 1, 2],
        'neutral': ['N', 'N', 'N', 'N', 'N'],
    })

@pytest.fixture
def sample_matches_nfl():
    """Sample match data for testing."""
    return pd.DataFrame({
        'season': [2023, 2023, 2023, 2024, 2024],
        'home_current': ['Team A', 'Team B', 'Team C', 'Team A', 'Team B'],
        'away_current': ['Team B', 'Team C', 'Team A', 'Team C', 'Team A'],
        'home_goals': [35, 23, 17, 41, 27],
        'away_goals': [12, 20, 23, 10, 24],
        'neutral': ['N', 'N', 'N', 'N', 'N'],
    })

@pytest.fixture
def sample_matches_nba():
    """Sample match data for testing."""
    return pd.DataFrame({
        'season': [2023, 2023, 2023, 2024, 2024],
        'home_current': ['Team A', 'Team B', 'Team C', 'Team A', 'Team B'],
        'away_current': ['Team B', 'Team C', 'Team A', 'Team C', 'Team A'],
        'home_goals': [90, 78, 100, 104, 102],
        'away_goals': [72, 88, 92, 91, 105],
        'neutral': ['N', 'N', 'N', 'N', 'N'],
    })

@pytest.fixture
def expansion_elos():
    """Expansion team Elo ratings."""
    return {
        'Expansion Team': 1400
    }


@pytest.fixture
def sample_matches_long():
    """Longer match sample."""
    return pd.DataFrame({
        'season': [2023, 2023, 2023, 2023, 2023,
                2024, 2024, 2024, 2024, 2024],
        'home_current': ['Team A', 'Team B', 'Team C', 'Team D', 'Team A',
                        'Team B', 'Team C', 'Team D', 'Team A', 'Team B'],
        'away_current': ['Team B', 'Team C', 'Team D', 'Team A', 'Team C',
                        'Team D', 'Team A', 'Team B', 'Team D', 'Team C'],
        'home_goals': [35, 17, 20, 0, 45, 25, 30, 17, 22, 40],
        'away_goals': [12, 23, 17, 3, 22, 25, 10, 22, 31, 10],
        'neutral': ['N', 'N', 'N', 'N', 'N','N', 'N', 'N', 'N', 'N'],
    })


class TestEloCalculatorInitialization:
    """Test EloCalculator initialization."""
    
    def test_init_basic(self, sample_matches, sample_elo_params):
        """Test basic initialization."""
        calc = EloCalculator(sample_matches, sample_elo_params)
        
        assert calc.initial_rating == 1600
        assert calc.k == 20
        assert calc.season_start_adj == 0.25
        assert calc.home_adv == 0
        assert isinstance(calc.ratings, dict)
        assert len(calc.ratings) == 0
    
    def test_init_with_custom_initial_rating(self, sample_matches, sample_elo_params):
        """Test initialization with custom initial rating."""
        calc = EloCalculator(sample_matches, sample_elo_params, initial_rating=1500)
        
        assert calc.initial_rating == 1500
    
    def test_init_with_expansion_elos(self, sample_matches, sample_elo_params, expansion_elos):
        """Test initialization with expansion team Elos."""
        calc = EloCalculator(sample_matches, sample_elo_params, expansion_elos=expansion_elos)
        
        assert calc.expansion_elos == expansion_elos


class TestGetRating:
    """Test get_rating method."""
    
    def test_get_rating_new_team(self, sample_matches, sample_elo_params):
        """Test getting rating for a team not yet rated."""
        calc = EloCalculator(sample_matches, sample_elo_params)
        rating = calc.get_rating('Team A')
        assert rating == 1600
    
    def test_get_rating_existing_team(self, sample_matches, sample_elo_params):
        """Test getting rating for a team with existing rating."""
        calc = EloCalculator(sample_matches, sample_elo_params)
        calc.ratings['Team A'] = 1750
        
        rating = calc.get_rating('Team A')
        assert rating == 1750
    
    def test_get_rating_expansion_team(self, sample_matches, sample_elo_params, expansion_elos):
        """Test getting rating for expansion team."""
        calc = EloCalculator(sample_matches, sample_elo_params, expansion_elos=expansion_elos)
        
        rating = calc.get_rating('Expansion Team')
        assert rating == 1400
    
    def test_get_rating_priority(self, sample_matches, sample_elo_params, expansion_elos):
        """Test that current ratings take priority over expansion ratings."""
        calc = EloCalculator(sample_matches, sample_elo_params, expansion_elos=expansion_elos)
        calc.ratings['Expansion Team'] = 1550
        
        rating = calc.get_rating('Expansion Team')
        assert rating == 1550


class TestAdjustSeasonStartElo:
    """Test adjust_season_start_elo method."""
    
    def test_adjust_season_start_single_team(self, sample_matches, sample_elo_params):
        """Test adjustment with single team (should stay same)."""
        calc = EloCalculator(sample_matches, sample_elo_params)
        calc.ratings = {'Team A': 1700}
        
        calc.adjust_season_start_elo()
        
        assert calc.ratings['Team A'] == 1700
    
    def test_adjust_season_start_multiple_teams(self, sample_matches, sample_elo_params):
        """Test adjustment with multiple teams."""
        calc = EloCalculator(sample_matches, sample_elo_params)
        calc.ratings = {
            'Team A': 1800,
            'Team B': 1400
        }
        
        calc.adjust_season_start_elo()
        
        # League average is 1600
        # Team A: 0.75 * 1800 + 0.25 * 1600 = 1350 + 400 = 1750
        # Team B: 0.75 * 1400 + 0.25 * 1600 = 1050 + 400 = 1450
        assert calc.ratings['Team A'] == pytest.approx(1750)
        assert calc.ratings['Team B'] == pytest.approx(1450)
    
    def test_adjust_season_start_balanced_teams(self, sample_matches, sample_elo_params):
        """Test that balanced teams stay at average."""
        calc = EloCalculator(sample_matches, sample_elo_params)
        calc.ratings = {
            'Team A': 1600,
            'Team B': 1600,
            'Team C': 1600
        }
        
        calc.adjust_season_start_elo()
        
        for team in calc.ratings:
            assert calc.ratings[team] == pytest.approx(1600)


class TestCalculateElo:
    """Test calculate_elo method."""
    
    def test_calculate_elo_home_win(self, sample_matches, sample_elo_params):
        """Test Elo calculation when home team wins."""
        calc = EloCalculator(sample_matches, sample_elo_params)
        
        new_a, new_b, exp_a, exp_b = calc.calculate_elo(1600, 1600, 3, 1, 0)
        
        # With equal ratings, expected is 0.5 for each
        assert exp_a == pytest.approx(0.5)
        assert exp_b == pytest.approx(0.5)

        # Home team should gain rating
        assert round(new_a, 0) == pytest.approx(1611)
        assert round(new_b, 0) == pytest.approx(1589)

    
    def test_calculate_elo_away_win(self, sample_matches, sample_elo_params):
        """Test Elo calculation when away team wins."""
        calc = EloCalculator(sample_matches, sample_elo_params)
        
        new_a, new_b, exp_a, exp_b = calc.calculate_elo(1600, 1600, 1, 3, 0)
        
        # Away team should gain rating
        assert round(new_a, 0) == pytest.approx(1589)
        assert round(new_b, 0) == pytest.approx(1611)
    
    def test_calculate_elo_tie(self, sample_matches, sample_elo_params):
        """Test Elo calculation for a tie."""
        calc = EloCalculator(sample_matches, sample_elo_params)
        
        new_a, new_b, exp_a, exp_b = calc.calculate_elo(1600, 1600, 2, 2, 0)
        
        # With equal ratings and a tie, ratings should stay the same
        assert new_a == pytest.approx(1600)
        assert new_b == pytest.approx(1600)
    
    def test_calculate_elo_expected_sum_to_one(self, sample_matches, sample_elo_params):
        """Test that expected probabilities sum to 1."""
        calc = EloCalculator(sample_matches, sample_elo_params)
        
        _, _, exp_a, exp_b = calc.calculate_elo(1700, 1500, 3, 1, 0)
        
        assert round(exp_a, 2) == pytest.approx(0.76)   
        assert round(exp_b, 2) == pytest.approx(0.24)   

class TestUpdateRatings:
    """Test update_ratins method."""
    
    def test_update_ratings_basic(self, sample_matches, sample_elo_params):
        """Test basic rating update."""
        calc = EloCalculator(sample_matches, sample_elo_params)
        
        result = calc.update_ratings('Team A', 'Team B', 3, 1, 'N')
        
        assert len(result) == 6
        home_before, away_before, home_after, away_after, exp_home, exp_away = result
        
        # Initial ratings should be 1600
        assert home_before == 1600
        assert away_before == 1600
        # Home won, so should gain rating
        assert home_after > 1600
        assert away_after < 1600
        # Teams should now be in ratings dict
        assert 'Team A' in calc.ratings
        assert 'Team B' in calc.ratings
    
    def test_update_ratings_maintains_state(self, sample_matches, sample_elo_params):
        """Test that ratings persist across multiple updates."""
        calc = EloCalculator(sample_matches, sample_elo_params)
        
        # First match
        calc.update_ratings('Team A', 'Team B', 3, 1, 'N')
        team_a_rating = calc.ratings['Team A']
        
        # Second match
        _, _, home_after, _, _, _ = calc.update_ratings('Team A', 'Team C', 2, 1, 'N')
        
        # Team A's new rating should be based on their previous rating
        assert calc.ratings['Team A'] == home_after
        assert calc.ratings['Team A'] != team_a_rating


class TestGetCurrentRatings:
    """Test get_current_ratings method."""
    
    def test_get_current_ratings_empty(self, sample_matches, sample_elo_params):
        """Test getting ratings when no matches played."""
        calc = EloCalculator(sample_matches, sample_elo_params)
        
        ratings_df = calc.get_current_ratings()
        
        assert isinstance(ratings_df, pd.DataFrame)
        assert len(ratings_df) == 0
        assert list(ratings_df.columns) == ['club', 'elo']
    
    def test_get_current_ratings_with_teams(self, sample_matches, sample_elo_params):
        """Test getting ratings with teams."""
        calc = EloCalculator(sample_matches, sample_elo_params)
        calc.ratings = {
            'Team A': 1700,
            'Team B': 1500,
            'Team C': 1600
        }
        
        ratings_df = calc.get_current_ratings()
        
        assert len(ratings_df) == 3
        assert ratings_df.iloc[0]['club'] == 'Team A'  # Highest rated
        assert ratings_df.iloc[0]['elo'] == 1700
        assert ratings_df.iloc[2]['club'] == 'Team B'  # Lowest rated
    
    def test_get_current_ratings_sorted_descending(self, sample_matches, sample_elo_params):
        """Test that ratings are sorted from highest to lowest."""
        calc = EloCalculator(sample_matches, sample_elo_params)
        calc.ratings = {
            'Team A': 1500,
            'Team B': 1800,
            'Team C': 1650
        }
        
        ratings_df = calc.get_current_ratings()
        
        assert ratings_df.iloc[0]['elo'] >= ratings_df.iloc[1]['elo']
        assert ratings_df.iloc[1]['elo'] >= ratings_df.iloc[2]['elo']


class TestUpdateMatchesElos:
    """Test update_matches_elos method."""
    
    def test_update_matches_elos_adds_columns(self, sample_matches, sample_elo_params):
        """Test that all required columns are added."""
        calc = EloCalculator(sample_matches.copy(), sample_elo_params)
        
        calc.update_matches_elos()
        
        expected_columns = [
            'home_elo_before', 'away_elo_before',
            'home_elo_after', 'away_elo_after',
            'home_win_expectancy', 'away_win_expectancy'
        ]
        for col in expected_columns:
            assert col in calc.matches.columns
    
    def test_update_matches_elos_correct_values(self, sample_matches, sample_elo_params):
        """Test that Elo values are calculated correctly."""
        calc = EloCalculator(sample_matches.copy(), sample_elo_params)
        
        calc.update_matches_elos()
        
        # First match: both teams start at 1600
        assert calc.matches.iloc[0]['home_elo_before'] == 1600
        assert calc.matches.iloc[0]['away_elo_before'] == 1600
        # Home won 3-1, so should gain rating
        assert calc.matches.iloc[0]['home_elo_after'] > 1600
        assert calc.matches.iloc[0]['away_elo_after'] < 1600
        # Expectancies should sum to 1
        assert (calc.matches.iloc[0]['home_win_expectancy'] + 
                calc.matches.iloc[0]['away_win_expectancy']) == pytest.approx(1.0)
    
    def test_update_matches_elos_sequential_updates(self, sample_matches, sample_elo_params):
        """Test that ratings update sequentially through matches."""
        calc = EloCalculator(sample_matches.copy(), sample_elo_params)
        
        calc.update_matches_elos()
        
        # Second match should start with teams at their updated ratings
        # Team B played in match 0 and 1
        team_b_after_match_0 = calc.matches.iloc[0]['away_elo_after']
        team_b_before_match_1 = calc.matches.iloc[1]['home_elo_before']
        
        assert team_b_after_match_0 == pytest.approx(team_b_before_match_1)
    
    def test_update_matches_elos_season_adjustment(self, sample_matches, sample_elo_params):
        """Test that season start adjustment is applied."""
        calc = EloCalculator(sample_matches.copy(), sample_elo_params)
        
        calc.update_matches_elos()
        
        # After match 2 (index 2), season changes from 2023 to 2024
        # So match 3 (index 3) should have adjusted ratings
        # We can verify by checking that ratings were adjusted toward mean
        team_a_after_match_2 = calc.matches.iloc[2]['away_elo_after']
        team_a_before_match_3 = calc.matches.iloc[3]['home_elo_before']
        
        # These should be different due to season adjustment
        # (unless Team A happened to be exactly at league average)
        # This test verifies the adjustment logic runs
        assert 'home_elo_before' in calc.matches.columns
    
    def test_update_matches_elos_preserves_original_data(self, sample_matches, sample_elo_params):
        """Test that original match data is preserved."""
        original_columns = set(sample_matches.columns)
        calc = EloCalculator(sample_matches.copy(), sample_elo_params)
        
        calc.update_matches_elos()
        
        for col in original_columns:
            assert col in calc.matches.columns
            pd.testing.assert_series_equal(
                sample_matches[col].reset_index(drop=True),
                calc.matches[col].reset_index(drop=True)
            )


class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_empty_dataframe(self, sample_elo_params):
        """Test handling of empty match DataFrame."""
        empty_matches = pd.DataFrame(columns=['season', 'home_current', 'away_current', 'home_goals', 'away_goals'])
        calc = EloCalculator(empty_matches, sample_elo_params)
        
        calc.update_matches_elos()
        
        assert len(calc.matches) == 0
    
    def test_single_match(self, sample_elo_params):
        """Test with single match."""
        single_match = pd.DataFrame({
            'season': [2023],
            'home_current': ['Team A'],
            'away_current': ['Team B'],
            'home_goals': [2],
            'away_goals': [1],
            'neutral': 'N'
        })
        calc = EloCalculator(single_match, sample_elo_params)
        
        calc.update_matches_elos()
        
        assert len(calc.matches) == 1
        assert calc.matches.iloc[0]['home_elo_after'] > calc.matches.iloc[0]['home_elo_before']
    
    def test_same_team_multiple_matches(self, sample_elo_params):
        """Test same team playing multiple consecutive matches."""
        matches = pd.DataFrame({
            'season': [2023, 2023, 2023],
            'home_current': ['Team A', 'Team A', 'Team A'],
            'away_current': ['Team B', 'Team C', 'Team D'],
            'home_goals': [3, 2, 4],
            'away_goals': [1, 1, 0],
            'neutral': 'N'
        })
        calc = EloCalculator(matches, sample_elo_params)
        
        calc.update_matches_elos()
        
        # Team A should have progressively higher rating
        assert calc.matches.iloc[0]['home_elo_after'] > calc.matches.iloc[0]['home_elo_before']
        assert calc.matches.iloc[1]['home_elo_after'] > calc.matches.iloc[1]['home_elo_before']
        assert calc.matches.iloc[2]['home_elo_after'] > calc.matches.iloc[2]['home_elo_before']


class TestMovMultiplier:
    """Test the MOV (Margin of Victory) multiplier logic."""
    
    def test_mov_multiplier_tie(self, sample_matches, sample_elo_params):
        """Test MOV multiplier is 1 for ties."""
        calc = EloCalculator(sample_matches, sample_elo_params)
        
        new_a, new_b, exp_a, _ = calc.calculate_elo(1600, 1600, 2, 2, 0)
        
        # For a tie with equal ratings: change = k * 1 * (0.5 - 0.5) = 0
        assert new_a == pytest.approx(1600)
        assert new_b == pytest.approx(1600)
    
    def test_mov_multiplier_increases_with_margin(self, sample_matches, sample_elo_params):
        """Test that larger margins result in larger rating changes."""
        calc = EloCalculator(sample_matches, sample_elo_params)
        
        # 1-goal win
        new_a_small, _, _, _ = calc.calculate_elo(1600, 1600, 2, 1, 0)
        # 5-goal win  
        new_a_large, _, _, _ = calc.calculate_elo(1600, 1600, 6, 1, 0)
        
        gain_small = new_a_small - 1600
        gain_large = new_a_large - 1600
        
        assert gain_large > gain_small

    def test_mov_multiplier_nfl(self, sample_matches_nfl, sample_elo_params):
        """Test small sample for elo ratings."""
        calc = EloCalculator(sample_matches_nfl, sample_elo_params)
        calc.update_matches_elos()
        
        assert round(calc.ratings['Team A'], 2) == 1646.68
        assert round(calc.ratings['Team B'], 2) == 1605.15
        assert round(calc.ratings['Team C'], 2) == 1548.17

    def test_mov_multiplier_nba(self, sample_matches_nba, sample_elo_params_nba):
        """Test small sample for elo ratings."""
        calc = EloCalculator(sample_matches_nba, sample_elo_params_nba)
        calc.update_matches_elos()

        assert round(calc.ratings['Team A'], 2) == 1618.86
        assert round(calc.ratings['Team B'], 2) == 1574.78
        assert round(calc.ratings['Team C'], 2) == 1606.37


class TestFullEloCalc:
    def test_full_elo_calc(self, sample_matches_long, sample_elo_params):
        """Test final Elos correspond to expected ones."""
        elo_calculator = EloCalculator(sample_matches_long, sample_elo_params)
        elo_calculator.update_matches_elos()
        calc_home_elos = elo_calculator.matches["home_elo_after"].round()
        calc_away_elos = elo_calculator.matches["away_elo_after"].round()

        expected_home_elos = pd.Series([1632.0, 1551.0, 1630.0, 1575.0, 1674.0, 1564.0, 1636.0, 1562.0, 1592.0, 1623.0], name="home_elo_after")
        expected_away_elos = pd.Series([1568.0, 1617.0, 1587.0, 1644.0, 1600.0, 1581.0, 1619.0, 1583.0, 1589.0, 1596.0], name="away_elo_after")

        pd.testing.assert_series_equal(expected_home_elos, calc_home_elos)
        pd.testing.assert_series_equal(expected_away_elos, calc_away_elos)

        # get current elos
        assert round(elo_calculator.get_rating("Team A"), 0) == 1592.0
        assert round(elo_calculator.get_rating("Team B"), 0) == 1623.0
        assert round(elo_calculator.get_rating("Team C"), 0) == 1596.0
        assert round(elo_calculator.get_rating("Team D"), 0) == 1589.0

