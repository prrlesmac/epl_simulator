import pytest
import pandas as pd
import numpy as np

# Import the functions we want to test
from simulator.sim_season import (
    validate_league_configuration,
    simulate_league,
    run_all_simulations
)

@pytest.fixture
def csv_test_data_case_1():
    return pd.read_csv('tests/data/schedules/schedule_case_1.csv')

@pytest.fixture
def csv_test_data_case_2():
    return pd.read_csv('tests/data/schedules/schedule_case_2.csv')

@pytest.fixture
def csv_elos_case_1():
    return pd.read_csv('tests/data/elos/current_elos_case_1.csv')

@pytest.fixture
def csv_elos_case_2():
    return pd.read_csv('tests/data/elos/current_elos_case_2.csv')


class TestValidateLeagueConfiguration:
    """Test cases for validate_league_configuration function."""
    
    def test_validate_continental_league_missing_bracket_draw(self):
        """Test validation fails when continental league has knockout matches but no bracket draw."""
        schedule = pd.DataFrame({
            'round': ['League', 'R16'],
            'played': ['Y', 'N']
        })
        league_rules = {
            'has_knockout': True,
            'knockout_draw': None,
        }
        
        with pytest.raises(ValueError, match="has knockout matches but no bracket draw defined"):
            validate_league_configuration(schedule, league_rules)
    
    def test_validate_continental_league_bracket_draw_with_pending_league(self):
        """Test validation fails when continental league has bracket draw but league phase unfinished."""
        schedule = pd.DataFrame({
            'round': ['League', 'League'],
            'played': ['Y', 'N']
        })
        league_rules = {
            'has_knockout': True,
            'knockout_draw': [('Team A', 'Team B')],
        }
                
        with pytest.raises(ValueError, match="has a bracket draw defined but league phase is unfinished"):
            validate_league_configuration(schedule, league_rules)
    
    def test_validate_continental_league_valid_config(self):
        """Test validation passes for valid continental league configuration."""
        schedule = pd.DataFrame({
            'round': ['League', 'R16'],
            'played': ['Y', 'Y']
        })
        league_rules = {
            'has_knockout': True,
            'knockout_draw': [('Team A', 'Team B')],
        }
        
        # Should not raise any exception
        validate_league_configuration(schedule, league_rules)
    
    def test_validate_domestic_league(self):
        """Test validation passes for domestic league."""
        schedule = pd.DataFrame({
            'round': ['League', 'League'],
            'played': ['Y', 'N']
        })
        league_rules = {'has_knockout': False}
        
        # Should not raise any exception
        validate_league_configuration(schedule, league_rules)


class TestSimulateLeague:
    """Test cases for simulate_league function."""

    def test_simulate_league_domestic(self, csv_test_data_case_1, csv_elos_case_1):
        """Test simulating a domestic league."""
        # Setup
        league_rules = {
            "has_knockout": False,
            "classification": [
                "points",
                "goal_difference",
                "goals_for",
                "h2h_points",
                "h2h_away_goals_for",
            ],
            "qualification": {
                "champion": [1],
                "top_4": [1, 2, 3, 4],
                "relegation_direct": [18, 19, 20],
            },
        }
        
        mock_schedule = csv_test_data_case_1
        mock_elos = csv_elos_case_1
        
        result = simulate_league(league_rules, mock_schedule, mock_elos)
        # check all columns present
        expected_columns = ['team', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12',
       '13', '14', '15', '16', '17', '18', '19', '20', 'champion', 'top_4',
       'relegation_direct', 'updated_at']
        assert list(result.columns) == expected_columns

        #check the sum of all columns equals to 1
        subset_cols = ["champion", '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12',
       '13', '14', '15', '16', '17', '18', '19', '20']
        row_sums = result[subset_cols].sum()
        assert np.allclose(row_sums, 1.0, atol=1e-3), "Row sums not close to 1 ± 1e-3"

        col_sum = result["top_4"].sum()
        assert np.isclose(col_sum, 4.0, atol=1e-3)

        col_sum = result["relegation_direct"].sum()
        assert np.isclose(col_sum, 3.0, atol=1e-3)

        # check all teams are the one in fixtures
        assert set(result["team"]) == set(mock_schedule["home"]), "Values don't match"
        assert set(result["team"]) == set(mock_schedule["away"]), "Values don't match"


    def test_simulate_league_continental(self, csv_test_data_case_2, csv_elos_case_2):
        """Test simulating a domestic league."""
        # Setup
        league_rules = {
            "has_knockout": True,
            "classification": [
                "points",
                "goal_difference",
                "goals_for",
                "away_goals_for",
            ],
            # No relegation info for UCL, exclude or set None
            "qualification": {
                "direct_to_round_of_16": list(range(1, 9)),
                "playoff": list(range(9, 25)),
            },
            "knockout_bracket": [
                (1, "Bye"),
                (16, 17),
                (8, "Bye"),
                (9, 24),
                (4, "Bye"),
                (13, 20),
                (5, "Bye"),
                (12, 21),
                (2, "Bye"),
                (15, 18),
                (7, "Bye"),
                (10, 23),
                (3, "Bye"),
                (14, 19),
                (6, "Bye"),
                (11, 22),
            ],
            "knockout_format": {
                "po_r32": "two-legged",
                "po_r16": "two-legged",
                "po_r8": "two-legged",  
                "po_r4": "two-legged",
                "po_r2": "single_game_neutral",
            },
            "knockout_draw": #None
            [
                ("Liverpool", "Bye"),
                ("Paris S-G", "Brest"),
                ("Aston Villa", "Bye"),
                ("Atalanta", "Club Brugge"),
                ("Arsenal", "Bye"),
                ("PSV Eindhoven", "Juventus"),
                ("Atlético Madrid", "Bye"),
                ("Real Madrid", "Manchester City"),
                ("Barcelona", "Bye"),
                ("Benfica", "Monaco"),
                ("Lille", "Bye"),
                ("Dortmund", "Sporting CP"),
                ("Inter", "Bye"),
                ("Milan", "Feyenoord"),
                ("Leverkusen", "Bye"),
                ("Bayern Munich", "Celtic"),
            ],
        }
        
        mock_schedule = csv_test_data_case_2
        mock_elos = csv_elos_case_2
        
        result = simulate_league(league_rules, mock_schedule, mock_elos)

        # check all columns present
        expected_columns = ['team', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12',
       '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24',
       '25', '26', '27', '28', '29', '30', '31', '32', '33', '34', '35', '36',
       'po_r32', 'po_r16', 'po_r8', 'po_r4', 'po_r2', 'po_champion',
       'direct_to_round_of_16', 'playoff', 'updated_at']
        assert list(result.columns) == expected_columns

        #check the sum of all columns equals to 1
        subset_cols = [ '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12',
       '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24',
       '25', '26', '27', '28', '29', '30', '31', '32', '33', '34', '35', '36',
       'po_champion']
        row_sums = result[subset_cols].sum()
        assert np.allclose(row_sums, 1.0, atol=1e-3), "Row sums not close to 1 ± 1e-3"
        col_sum = result["po_r32"].sum()
        assert np.isclose(col_sum, 24.0 , atol=1e-3)
        col_sum = result["po_r16"].sum()
        assert np.isclose(col_sum, 16.0 , atol=1e-3)
        col_sum = result["po_r8"].sum()
        assert np.isclose(col_sum, 8.0 , atol=1e-3)
        col_sum = result["po_r4"].sum()
        assert np.isclose(col_sum, 4.0 , atol=1e-3)
        col_sum = result["po_r2"].sum()
        assert np.isclose(col_sum, 2.0 , atol=1e-3)
        col_sum = result["direct_to_round_of_16"].sum()
        assert np.isclose(col_sum, 8.0 , atol=1e-3)
        col_sum = result["playoff"].sum()
        assert np.isclose(col_sum, 16.0 , atol=1e-3)
        # check all teams are the one in fixtures
        assert set(result["team"]) == set(mock_schedule["home"]), "Values don't match"
        assert set(result["team"]) == set(mock_schedule["away"]), "Values don't match"

if __name__ == "__main__":
    pytest.main([__file__])