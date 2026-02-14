import pytest
import pandas as pd
import numpy as np
from .conftest import schedule_dtypes

# Import the functions we want to test
from simulator.sim_season import (
    simulate_league,
)


def read_schedule_csv(filepath):
    return pd.read_csv(filepath, dtype=schedule_dtypes, keep_default_na=False)


# MLB SCHEDULE fixtures
# case 1: season not started
@pytest.fixture
def csv_schedule_data_mlb_case_1():
    return read_schedule_csv("tests/data/schedules/schedule_mlb_case_1.csv")

@pytest.fixture
def csv_schedule_data_mlb_case_2():
    return read_schedule_csv("tests/data/schedules/schedule_mlb_case_2.csv")

@pytest.fixture
def csv_schedule_data_mlb_case_3():
    return read_schedule_csv("tests/data/schedules/schedule_mlb_case_3.csv")

@pytest.fixture
def csv_schedule_data_mlb_case_4():
    return read_schedule_csv("tests/data/schedules/schedule_mlb_case_4.csv")

@pytest.fixture
def csv_schedule_data_mlb_case_5():
    return read_schedule_csv("tests/data/schedules/schedule_mlb_case_5.csv")
# ELOS fixtures
@pytest.fixture
def csv_elos_data_mlb():
    return pd.read_csv("tests/data/elos/current_elos_mlb_case_1.csv")



@pytest.fixture
def mlb_league_rules():
    return  {
        "sim_type": "winner",
        "home_advantage": 25,
        "elo_kfactor": 5,
        "season_start_adj": 1/4,
        "has_knockout": True,
        "classification": {
            "division": [             
                        "win_loss_pct",
                         "h2h_sweep_full",
                         "h2h_win_loss_pct",
                         "win_loss_pct_div",
                         "win_loss_pct_conf",
                         "win_loss_pct_conf_last_half"
                         ],
            "conference": [
                         "division_winner",
                         "win_loss_pct",
                         "h2h_sweep_full",
                         "h2h_win_loss_pct",
                         "win_loss_pct_div",
                         "win_loss_pct_conf",
                         "win_loss_pct_conf_last_half"
                         ],
            "league": [
                         "win_loss_pct",
                         "h2h_sweep_full",
                         "h2h_win_loss_pct",
                         "win_loss_pct_div",
                         "win_loss_pct_conf",
                         "win_loss_pct_conf_last_half"
                        ],
        },
        "qualification": {
            "playoff": [f"American League {i}" for i in range(1, 7)] + [f"National League {i}" for i in range(1, 7)]
        },
        "knockout_bracket": [
            ("National League 1", "Bye"),
            ("National League 4", "National League 5"),
            ("National League 2", "Bye"),
            ("National League 3", "National League 6"),
            ("American League 1", "Bye"),
            ("American League 4", "American League 5"),
            ("American League 2", "Bye"),
            ("American League 3", "American League 6"),
        ],
        "knockout_format": {
            "po_r16": "best_of_3",
            "po_r8": "best_of_5",
            "po_r4": "best_of_7",
            "po_r2": "best_of_7",
        },
        "knockout_draw_status": "no_draw",
        "knockout_draw": None,
        "knockout_reseeding": False,
        "has_play_in": False,
        "league_type": "MLB"
    }

@pytest.fixture
def mlb_playoff_rules():
    return  {
        "sim_type": "winner",
        "home_advantage": 25,
        "elo_kfactor": 5,
        "season_start_adj": 1/4,
        "has_knockout": True,
        "classification": {
            "division": [             
                        "win_loss_pct",
                         "h2h_sweep_full",
                         "h2h_win_loss_pct",
                         "win_loss_pct_div",
                         "win_loss_pct_conf",
                         "win_loss_pct_conf_last_half"
                         ],
            "conference": [
                         "division_winner",
                         "win_loss_pct",
                         "h2h_sweep_full",
                         "h2h_win_loss_pct",
                         "win_loss_pct_div",
                         "win_loss_pct_conf",
                         "win_loss_pct_conf_last_half"
                         ],
            "league": [
                         "win_loss_pct",
                         "h2h_sweep_full",
                         "h2h_win_loss_pct",
                         "win_loss_pct_div",
                         "win_loss_pct_conf",
                         "win_loss_pct_conf_last_half"
                        ],
        },
        "qualification": {
            "playoff": [f"American League {i}" for i in range(1, 7)] + [f"National League {i}" for i in range(1, 7)]
        },
        "knockout_bracket": [
            ("National League 1", "Bye"),
            ("National League 4", "National League 5"),
            ("National League 2", "Bye"),
            ("National League 3", "National League 6"),
            ("American League 1", "Bye"),
            ("American League 4", "American League 5"),
            ("American League 2", "Bye"),
            ("American League 3", "American League 6"),
        ],
        "knockout_format": {
            "po_r16": "best_of_3",
            "po_r8": "best_of_5",
            "po_r4": "best_of_7",
            "po_r2": "best_of_7",
        },
        "knockout_draw_status": "completed_draw",
        "knockout_draw": [
            ("Milwaukee Brewers", "Bye"),
            ("Chicago Cubs", "San Diego Padres"),
            ("Philadelphia Phillies", "Bye"),
            ("Los Angeles Dodgers", "Cincinnati Reds"),
            ("Toronto Blue Jays", "Bye"),
            ("New York Yankees", "Boston Red Sox"),
            ("Seattle Mariners", "Bye"),
            ("Cleveland Guardians", "Detroit Tigers"),
        ],
        "knockout_reseeding": False,
        "has_play_in": False,
        "league_type": "MLB"
    }

# ELOS fixtures
@pytest.fixture
def csv_mlb_divisions():
    return pd.read_csv("tests/data/divisions/mlb_divisions.csv")


@pytest.fixture
def final_mlb_results():
    return  {
        "eliminated in regular season": [
            "Arizona D'backs",
            "Atlanta Braves",
            "Baltimore Orioles",
            "Chicago White Sox",
            "Colorado Rockies",
            "Los Angeles Angels",
            "Minnesota Twins",
            "Athletics",
            "Pittsburgh Pirates",
            "San Francisco Giants",
            "Tampa Bay Rays",
            "Texas Rangers",
            "Washington Nationals",
            "Miami Marlins", 
            "St. Louis Cardinals"
        ],
        "eliminated in wild card series": [
            "Cleveland Guardians",   
            "Boston Red Sox",      
            "San Diego Padres",      
            "Cincinnati Reds"       
        ],
        "first round bye": [
            "Toronto Blue Jays",   
            "Seatlle Mariners",      
            "Milwaukee Brewers",      
            "Philadelphia Phillies"       
        ],
        "eliminated in division series": [
            "New York Yankees",  
            "Detroit Tigers",  
            "Philadelphia Phillies", 
            "Chicago Cubs"      
        ],
        "eliminated in league championship series": [
            "Seattle Mariners", 
            "Milwaukee Brewers" 
        ],
        "runner-up": [
            "Toronto Blue Jays" 
        ],
        "champion": [
            "Los Angeles Dodgers"
        ]
    }


class TestSimulateLeague:
    """Test cases for simulate_league function."""

    def assert_mlb_league_summary(self, result, mock_schedule):
        # check all columns present
        expected_columns = [
            "team",
            'American League 1',
            'American League 2',
            'American League 3',
            'American League 4',
            'American League 5',
            'American League 6',
            'American League 7',
            'American League 8',
            'American League 9',
            'American League 10',
            'American League 11',
            'American League 12',
            'American League 13',
            'American League 14',
            'American League 15',
            'National League 1',
            'National League 2',
            'National League 3',
            'National League 4',
            'National League 5',
            'National League 6',
            'National League 7',
            'National League 8',
            'National League 9',
            'National League 10',
            'National League 11',
            'National League 12',
            'National League 13',
            'National League 14',
            'National League 15',
            "po_r16",
            "po_r8",
            "po_r4",
            "po_r2",
            "po_champion",
            "playoff",
            "updated_at",
        ]
        assert set(result.columns) == set(expected_columns)

        col_sum = result["playoff"].sum()
        assert np.isclose(col_sum, 12.0, atol=1e-3)

        col_sum = result["po_r16"].sum()
        assert np.isclose(col_sum, 12.0, atol=1e-3)

        col_sum = result["po_r8"].sum()
        assert np.isclose(col_sum, 8.0, atol=1e-3)

        col_sum = result["po_r4"].sum()
        assert np.isclose(col_sum, 4.0, atol=1e-3)

        col_sum = result["po_r2"].sum()
        assert np.isclose(col_sum, 2.0, atol=1e-3)

        col_sum = result["po_champion"].sum()
        assert np.isclose(col_sum, 1.0, atol=1e-3)

        # check all teams are the one in fixtures
        assert set(result["team"]) == set(mock_schedule["home"]), "Values don't match"
        assert set(result["team"]) == set(mock_schedule["away"]), "Values don't match"

    def test_simulate_league_mlb_case_1(
        self,
        csv_schedule_data_mlb_case_1,
        csv_elos_data_mlb,
        mlb_league_rules,
        csv_mlb_divisions
    ):
        """Test simulating a mlb league."""
        # Setup
        league_rules = mlb_league_rules
        schedule = csv_schedule_data_mlb_case_1
        elos = csv_elos_data_mlb
        divisions = csv_mlb_divisions
        result = simulate_league(
            league_rules, schedule, elos, divisions, num_simulations=10
        )
        self.assert_mlb_league_summary(result, schedule)

    def test_simulate_league_mlb_case_2(
        self,
        csv_schedule_data_mlb_case_2,
        csv_elos_data_mlb,
        mlb_league_rules,
        csv_mlb_divisions
    ):
        """Test simulating a mlb league."""
        # Setup
        league_rules = mlb_league_rules
        schedule = csv_schedule_data_mlb_case_2
        elos = csv_elos_data_mlb
        divisions = csv_mlb_divisions
        result = simulate_league(
            league_rules, schedule, elos, divisions, num_simulations=10
        )
        self.assert_mlb_league_summary(result, schedule)

    def test_simulate_league_mlb_case_3(
        self,
        csv_schedule_data_mlb_case_3,
        csv_elos_data_mlb,
        mlb_league_rules,
        csv_mlb_divisions,
        final_mlb_results
    ):
        """Test simulating a mlb league."""
        # Setup
        league_rules = mlb_league_rules
        schedule = csv_schedule_data_mlb_case_3
        elos = csv_elos_data_mlb
        divisions = csv_mlb_divisions
        result = simulate_league(
            league_rules, schedule, elos, divisions, num_simulations=10
        )
        self.assert_mlb_league_summary(result, schedule)

        eliminated_in_season = final_mlb_results["eliminated in regular season"]
        advanced_to_playoff = [
            team
            for stage, teams in final_mlb_results.items()
            if stage in [
                "eliminated in wild card series",
                "eliminated in division series",
                "eliminated in league championship series",
                "runner-up",
                "champion"
            ]
            for team in teams
        ]
        first_round_bye = final_mlb_results["first round bye"]

        for rounds in ["playoff","po_r16", "po_r8", "po_r4", "po_r2", "po_champion"]:
            assert np.isclose(
                result.loc[result["team"].isin(eliminated_in_season)][rounds].all(),
                0.0,
                atol=1e-3,
            )
        for rounds in ["playoff","po_r16"]:
            assert np.isclose(
                (result.loc[result["team"].isin(advanced_to_playoff)][rounds]==1).all(),
                1.0,
                atol=1e-3,
            )
        for rounds in ["po_r8"]:
            assert np.isclose(
                (result.loc[result["team"].isin(first_round_bye)][rounds]==1).all(),
                1.0,
                atol=1e-3,
            )

    def test_simulate_league_mlb_case_4(
        self,
        csv_schedule_data_mlb_case_4,
        csv_elos_data_mlb,
        mlb_playoff_rules,
        csv_mlb_divisions,
        final_mlb_results
    ):
        """Test simulating a mlb league."""
        # Setup
        league_rules = mlb_playoff_rules
        schedule = csv_schedule_data_mlb_case_4
        elos = csv_elos_data_mlb
        divisions = csv_mlb_divisions
        result = simulate_league(
            league_rules, schedule, elos, divisions, num_simulations=10
        )
        self.assert_mlb_league_summary(result, schedule)

        eliminated_in_season = final_mlb_results["eliminated in regular season"]
        advanced_to_playoff = [
            team
            for stage, teams in final_mlb_results.items()
            if stage in [
                "eliminated in wild card series",
                "eliminated in division series",
                "eliminated in league championship series",
                "runner-up",
                "champion"
            ]
            for team in teams
        ]
        advanced_to_division_series = [
            team
            for stage, teams in final_mlb_results.items()
            if stage in [
                "eliminated in division series",
                "eliminated in league championship series",
                "runner-up",
                "champion"
            ]
            for team in teams
        ]
        advanced_to_league_championship_series = [
            team
            for stage, teams in final_mlb_results.items()
            if stage in [
                "eliminated in league championship series",
                "runner-up",
                "champion"
            ]
            for team in teams
        ]

        for rounds in ["playoff","po_r16", "po_r8", "po_r4", "po_r2", "po_champion"]:
            assert np.isclose(
                result.loc[result["team"].isin(eliminated_in_season)][rounds].all(),
                0.0,
                atol=1e-3,
            )
        for rounds in ["playoff","po_r16"]:
            assert np.isclose(
                (result.loc[result["team"].isin(advanced_to_playoff)][rounds]==1).all(),
                1.0,
                atol=1e-3,
            )
        for rounds in ["po_r8"]:
            assert np.isclose(
                (result.loc[result["team"].isin(advanced_to_division_series)][rounds]==1).all(),
                1.0,
                atol=1e-3,
            )
        for rounds in ["po_r4"]:
            assert np.isclose(
                (result.loc[result["team"].isin(advanced_to_league_championship_series)][rounds]==1).all(),
                1.0,
                atol=1e-3,
            )


    def test_simulate_league_mlb_case_5(
        self,
        csv_schedule_data_mlb_case_5,
        csv_elos_data_mlb,
        mlb_playoff_rules,
        csv_mlb_divisions,
        final_mlb_results
    ):
        """Test simulating a mlb league."""
        # Setup
        league_rules = mlb_playoff_rules
        schedule = csv_schedule_data_mlb_case_5
        elos = csv_elos_data_mlb
        divisions = csv_mlb_divisions
        result = simulate_league(
            league_rules, schedule, elos, divisions, num_simulations=10
        )
        self.assert_mlb_league_summary(result, schedule)

        eliminated_in_season = final_mlb_results["eliminated in regular season"]
        advanced_to_playoff = [
            team
            for stage, teams in final_mlb_results.items()
            if stage in [
                "eliminated in wild card series",
                "eliminated in division series",
                "eliminated in league championship series",
                "runner-up",
                "champion"
            ]
            for team in teams
        ]
        advanced_to_division_series = [
            team
            for stage, teams in final_mlb_results.items()
            if stage in [
                "eliminated in division series",
                "eliminated in league championship series",
                "runner-up",
                "champion"
            ]
            for team in teams
        ]
        advanced_to_league_championship_series = [
            team
            for stage, teams in final_mlb_results.items()
            if stage in [
                "eliminated in league championship series",
                "runner-up",
                "champion"
            ]
            for team in teams
        ]

        for rounds in ["playoff","po_r16", "po_r8", "po_r4", "po_r2", "po_champion"]:
            assert np.isclose(
                result.loc[result["team"].isin(eliminated_in_season)][rounds].all(),
                0.0,
                atol=1e-3,
            )
        for rounds in ["playoff","po_r16"]:
            assert np.isclose(
                (result.loc[result["team"].isin(advanced_to_playoff)][rounds]==1).all(),
                1.0,
                atol=1e-3,
            )
        for rounds in ["po_r8"]:
            assert np.isclose(
                (result.loc[result["team"].isin(advanced_to_division_series)][rounds]==1).all(),
                1.0,
                atol=1e-3,
            )
        for rounds in ["po_r4"]:
            assert np.isclose(
                (result.loc[result["team"].isin(advanced_to_league_championship_series)][rounds]==1).all(),
                1.0,
                atol=1e-3,
            )
        for rounds in ["po_r2"]:
            assert np.isclose(
                (result.loc[result["team"].isin(["Los Angeles Dodgers"])][rounds]==1).all(),
                1.0,
                atol=1e-3,
            )
if __name__ == "__main__":
    pytest.main([__file__])
