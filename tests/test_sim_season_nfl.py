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


# NFL SCHEDULE fixtures
# case 1: season not started
@pytest.fixture
def csv_schedule_data_nfl_case_1():
    return read_schedule_csv("tests/data/schedules/schedule_nfl_case_1.csv")

@pytest.fixture
def csv_schedule_data_nfl_case_2():
    return read_schedule_csv("tests/data/schedules/schedule_nfl_case_2.csv")

@pytest.fixture
def csv_schedule_data_nfl_case_3():
    return read_schedule_csv("tests/data/schedules/schedule_nfl_case_3.csv")

# ELOS fixtures
@pytest.fixture
def csv_elos_data_nfl_case_1():
    return pd.read_csv("tests/data/elos/current_elos_nfl_case_1.csv")

@pytest.fixture
def nfl_league_rules():
    return {
        "sim_type": "winner",
        "has_knockout": True,
        "classification": {
            "division": ["win_loss_pct",
                         "h2h_win_loss_pct",
                         "win_loss_pct_div",
                         "win_loss_pct_conf",
                         "strength_of_victory",
                         "strength_of_schedule"
                         ],
            "conference": [
                         "division_winner",
                         "win_loss_pct",
                         "h2h_break_division_ties",
                         "h2h_sweep_full",
                         "win_loss_pct_conf",
                         "h2h_win_loss_pct_common_games",
                         "strength_of_victory",
                         "strength_of_schedule",
                         ],
            "league": ["win_loss_pct"],
        },
        "qualification": {
            "playoff": list(range(1, 15)),
            "first_round_bye": list(range(1,3))
        },
        "knockout_bracket": [
            ("NFC 1", "Bye"),
            ("NFC 2", "NFC 7"),
            ("NFC 3", "NFC 6"),
            ("NFC 4", "NFC 5"),
            ("AFC 1", "Bye"),
            ("AFC 2", "AFC 7"),
            ("AFC 3", "AFC 6"),
            ("AFC 4", "AFC 5"),
        ],
        "knockout_format": {
            "po_r16": "single_game",
            "po_r8": "single_game",
            "po_r4": "single_game",
            "po_r2": "single_game_neutral",
        },
        "knockout_draw_status": "no_draw",
        "knockout_draw": None,
        "knockout_reseeding": True,
        "league_type": "NFL"
    }

# ELOS fixtures
@pytest.fixture
def csv_nfl_divisions():
    return pd.read_csv("tests/data/divisions/nfl_divisions.csv")


class TestSimulateLeague:
    """Test cases for simulate_league function."""

    def assert_nfl_league_summary(self, result, mock_schedule):
        # check all columns present
        expected_columns = [
            "team",
            "1",
            "2",
            "3",
            "4",
            "5",
            "6",
            "7",
            "8",
            "9",
            "10",
            "11",
            "12",
            "13",
            "14",
            "15",
            "16",
            "17",
            "18",
            "19",
            "20",
            "21",
            "22",
            "23",
            "24",
            "25",
            "26",
            "27",
            "28",
            "29",
            "30",
            "31",
            "32",
            "po_r16",
            "po_r8",
            "po_r4",
            "po_r2",
            "po_champion",
            "playoff",
            "first_round_bye",
            "updated_at",
        ]
        assert list(result.columns) == expected_columns

        col_sum = result["playoff"].sum()
        assert np.isclose(col_sum, 14.0, atol=1e-3)

        col_sum = result["first_round_bye"].sum()
        assert np.isclose(col_sum, 2.0, atol=1e-3)

        col_sum = result["po_r16"].sum()
        assert np.isclose(col_sum, 14.0, atol=1e-3)

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

    def test_simulate_league_nfl_case_1(
        self,
        csv_schedule_data_nfl_case_1,
        csv_elos_data_nfl_case_1,
        nfl_league_rules,
        csv_nfl_divisions
    ):
        """Test simulating a NFL league."""
        # Setup
        league_rules = nfl_league_rules
        schedule = csv_schedule_data_nfl_case_1
        elos = csv_elos_data_nfl_case_1
        divisions = csv_nfl_divisions
        result = simulate_league(
            league_rules, schedule, elos, divisions, num_simulations=10
        )
        self.assert_nfl_league_summary(result, schedule)


    def test_simulate_league_nfl_case_2(
        self,
        csv_schedule_data_nfl_case_2,
        csv_elos_data_nfl_case_1,
        nfl_league_rules,
        csv_nfl_divisions
    ):
        """Test simulating a NFL league."""
        # Setup
        league_rules = nfl_league_rules
        schedule = csv_schedule_data_nfl_case_2
        elos = csv_elos_data_nfl_case_1
        divisions = csv_nfl_divisions
        result = simulate_league(
            league_rules, schedule, elos, divisions, num_simulations=10
        )
        self.assert_nfl_league_summary(result, schedule)

if __name__ == "__main__":
    pytest.main([__file__])
