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

@pytest.fixture
def csv_schedule_data_nfl_case_4():
    return read_schedule_csv("tests/data/schedules/schedule_nfl_case_4.csv")

@pytest.fixture
def csv_schedule_data_nfl_case_5():
    return read_schedule_csv("tests/data/schedules/schedule_nfl_case_5.csv")

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
            "playoff": [f"NFC {i}" for i in range(1, 8)] + [f"AFC {i}" for i in range(1, 8)],
            "first_round_bye": [f"NFC {i}" for i in range(1, 2)] + [f"AFC {i}" for i in range(1, 2)]
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

@pytest.fixture
def nfl_league_rules_playoff():
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
            "playoff": [f"NFC {i}" for i in range(1, 8)] + [f"AFC {i}" for i in range(1, 8)],
            "first_round_bye": [f"NFC {i}" for i in range(1, 2)] + [f"AFC {i}" for i in range(1, 2)]
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
        "knockout_draw_status": "completed_draw",
        "knockout_draw": [
            ("Detroit Lions", "Bye"),
            ("Philadelphia Eagles", "Green Bay Packers"),
            ("Tampa Bay Buccaneers", "Washington Commanders"),
            ("Los Angeles Rams", "Minnesota Vikings"),
            ("Kansas City Chiefs", "Bye"),
            ("Buffalo Bills", "Denver Broncos"),
            ("Baltimore Ravens", "Pittsburgh Steelers"),
            ("Houston Texans", "Los Angeles Chargers"),
        ],
        "knockout_reseeding": True,
        "league_type": "NFL"
    }

# ELOS fixtures
@pytest.fixture
def csv_nfl_divisions():
    return pd.read_csv("tests/data/divisions/nfl_divisions.csv")

@pytest.fixture
def final_nfl_results():
    return {
        "eliminated in regular season": [
            "Seattle Seahawks",
            "Dallas Cowboys",
            "Arizona Cardinals",
            "San Francisco 49ers",
            "New Orleans Saints",
            "Chicago Bears",
            "Carolina Panthers",
            "New York Giants",
            "Indianapolis Colts",
            "New York Jets",
            "Las Vegas Raiders",
            "Jacksonville Jaguars",
            "Cleveland Browns",
            "Tennessee Titans",
            "New England Patriots",
            "Miami Dolphins"
        ],
        "first round bye": [
            "Kansas City Chiefs",
            "Detroit Lions",
        ],
        "eliminated in wild card": [
            "Los Angeles Chargers",
            "Pittsburgh Steelers",
            "Denver Broncos",
            "Minnesota Vikings",
            "Green Bay Packers",
            "Tampa Bay Buccaneers"
        ],
        "eliminated in divisional": [
            "Houston Texans",
            "Baltimore Ravens",
            "Detroit Lions",
            "Los Angeles Rams"
        ],
        "eliminated in conference": [
            "Buffalo Bills",
            "Washington Commanders",
        ],
        "runner-up": ["Kansas City Chiefs"],
        "champion": ["Philadelphia Eagles"],
    }

class TestSimulateLeague:
    """Test cases for simulate_league function."""

    def assert_nfl_league_summary(self, result, mock_schedule):
        # check all columns present
        expected_columns = [
            "team",
            'NFC 1',
            'NFC 2',
            'NFC 3',
            'NFC 4',
            'NFC 5',
            'NFC 6',
            'NFC 7',
            'NFC 8',
            'NFC 9',
            'NFC 10',
            'NFC 11',
            'NFC 12',
            'NFC 13',
            'NFC 14',
            'NFC 15',
            'NFC 16',
            'AFC 1',
            'AFC 2',
            'AFC 3',
            'AFC 4',
            'AFC 5',
            'AFC 6',
            'AFC 7',
            'AFC 8',
            'AFC 9',
            'AFC 10',
            'AFC 11',
            'AFC 12',
            'AFC 13',
            'AFC 14',
            'AFC 15',
            'AFC 16',
            "po_r16",
            "po_r8",
            "po_r4",
            "po_r2",
            "po_champion",
            "playoff",
            "first_round_bye",
            "updated_at",
        ]
        assert set(result.columns) == set(expected_columns)

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

    def test_simulate_league_nfl_case_3(
        self,
        csv_schedule_data_nfl_case_3,
        csv_elos_data_nfl_case_1,
        nfl_league_rules,
        csv_nfl_divisions,
        final_nfl_results,
    ):
        """Test simulating a NFL league."""
        # Setup
        league_rules = nfl_league_rules
        schedule = csv_schedule_data_nfl_case_3
        elos = csv_elos_data_nfl_case_1
        divisions = csv_nfl_divisions
        result = simulate_league(
            league_rules, schedule, elos, divisions, num_simulations=10
        )
        self.assert_nfl_league_summary(result, schedule)

        eliminated_in_season = final_nfl_results["eliminated in regular season"]
        advanced_to_playoff = [
            team
            for stage, teams in final_nfl_results.items()
            if stage != "eliminated in regular season"
            for team in teams
        ]
        first_round_bye = final_nfl_results["first round bye"]

        for rounds in ["playoff","po_r16", "po_r8", "po_r4", "po_r2", "po_champion"]:
            assert np.isclose(
                result.loc[result["team"].isin(eliminated_in_season)][rounds].all(),
                0.0,
                atol=1e-3,
            )
        for rounds in ["playoff","po_r16"]:
            assert np.isclose(
                result.loc[result["team"].isin(advanced_to_playoff)][rounds].all(),
                1.0,
                atol=1e-3,
            )
        assert np.isclose(
            result.loc[result["team"].isin(first_round_bye)]["first_round_bye"].all(),
            1.0,
            atol=1e-3,
        )
    
    def test_simulate_league_nfl_case_4(
        self,
        csv_schedule_data_nfl_case_4,
        csv_elos_data_nfl_case_1,
        nfl_league_rules_playoff,
        csv_nfl_divisions,
        final_nfl_results,
    ):
        """Test simulating a NFL league."""
        # Setup
        league_rules = nfl_league_rules_playoff
        schedule = csv_schedule_data_nfl_case_4
        elos = csv_elos_data_nfl_case_1
        divisions = csv_nfl_divisions
        result = simulate_league(
            league_rules, schedule, elos, divisions, num_simulations=10
        )
        self.assert_nfl_league_summary(result, schedule)

        eliminated_in_season = final_nfl_results["eliminated in regular season"]
        advanced_to_playoff = [
            team
            for stage, teams in final_nfl_results.items()
            if stage != "eliminated in regular season"
            for team in teams
        ]
        first_round_bye = final_nfl_results["first round bye"]
        advanced_to_divisional = [
            team
            for stage, teams in final_nfl_results.items()
            if stage in ["eliminated in divisional", "eliminated in conference", "runner-up", "champion"]
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
                result.loc[result["team"].isin(advanced_to_playoff)][rounds].all(),
                1.0,
                atol=1e-3,
            )
        assert np.isclose(
            result.loc[result["team"].isin(first_round_bye)]["first_round_bye"].all(),
            1.0,
            atol=1e-3,
        )
        for rounds in ["playoff","po_r16","po_r8"]:
            assert np.isclose(
                result.loc[result["team"].isin(advanced_to_divisional)][rounds].all(),
                1.0,
                atol=1e-3,
            )

    def test_simulate_league_nfl_case_5(
        self,
        csv_schedule_data_nfl_case_5,
        csv_elos_data_nfl_case_1,
        nfl_league_rules_playoff,
        csv_nfl_divisions,
        final_nfl_results,
    ):
        """Test simulating a NFL league."""
        # Setup
        league_rules = nfl_league_rules_playoff
        schedule = csv_schedule_data_nfl_case_5
        elos = csv_elos_data_nfl_case_1
        divisions = csv_nfl_divisions
        result = simulate_league(
            league_rules, schedule, elos, divisions, num_simulations=10
        )
        self.assert_nfl_league_summary(result, schedule)

        eliminated_in_season = final_nfl_results["eliminated in regular season"]
        advanced_to_playoff = [
            team
            for stage, teams in final_nfl_results.items()
            if stage != "eliminated in regular season"
            for team in teams
        ]
        first_round_bye = final_nfl_results["first round bye"]
        advanced_to_divisional = [
            team
            for stage, teams in final_nfl_results.items()
            if stage in ["eliminated in divisional", "eliminated in conference", "runner-up", "champion"]
            for team in teams
        ]
        advanced_to_conference = ["Washington Commanders","Kansas City Chiefs"]

        for rounds in ["playoff","po_r16", "po_r8", "po_r4", "po_r2", "po_champion"]:
            assert np.isclose(
                result.loc[result["team"].isin(eliminated_in_season)][rounds].all(),
                0.0,
                atol=1e-3,
            )
        for rounds in ["playoff","po_r16"]:
            assert np.isclose(
                result.loc[result["team"].isin(advanced_to_playoff)][rounds].all(),
                1.0,
                atol=1e-3,
            )
        assert np.isclose(
            result.loc[result["team"].isin(first_round_bye)]["first_round_bye"].all(),
            1.0,
            atol=1e-3,
        )
        for rounds in ["playoff","po_r16","po_r8"]:
            assert np.isclose(
                result.loc[result["team"].isin(advanced_to_divisional)][rounds].all(),
                1.0,
                atol=1e-3,
            )
        for rounds in ["playoff","po_r16","po_r8","po_r4"]:
            assert np.isclose(
                result.loc[result["team"].isin(advanced_to_conference)][rounds].all(),
                1.0,
                atol=1e-3,
            )

if __name__ == "__main__":
    pytest.main([__file__])
