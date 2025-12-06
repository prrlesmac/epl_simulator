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


# nba SCHEDULE fixtures
# case 1: season not started
@pytest.fixture
def csv_schedule_data_nba_case_1():
    return read_schedule_csv("tests/data/schedules/schedule_nba_case_1.csv")

@pytest.fixture
def csv_schedule_data_nba_case_2():
    return read_schedule_csv("tests/data/schedules/schedule_nba_case_2.csv")

@pytest.fixture
def csv_schedule_data_nba_case_3():
    return read_schedule_csv("tests/data/schedules/schedule_nba_case_3.csv")

@pytest.fixture
def csv_schedule_data_nba_case_4():
    return read_schedule_csv("tests/data/schedules/schedule_nba_case_4.csv")

# @pytest.fixture
# def csv_schedule_data_nba_case_5():
#     return read_schedule_csv("tests/data/schedules/schedule_nba_case_5.csv")

# ELOS fixtures
@pytest.fixture
def csv_elos_data_nba_case_1():
    return pd.read_csv("tests/data/elos/current_elos_nba_case_1.csv")

@pytest.fixture
def nba_league_rules():
    return {
        "sim_type": "winner",
        "has_knockout": True,
        "classification": {
            "division": ["win_loss_pct"
                         ],
            "conference": [
                         "win_loss_pct",
                         "h2h_win_loss_pct",
                         "win_loss_pct_div_if_same_div",
                         "win_loss_pct_conf",
                         "win_loss_pct_playoff_teams_same_conf",
                         "win_loss_pct_playoff_teams_other_conf",
                         ],
            "league": [
                         "win_loss_pct",
                         "h2h_win_loss_pct",
                         "win_loss_pct_div",
                         "win_loss_pct_conf",
                         "win_loss_pct_playoff_teams_same_conf",
                         "win_loss_pct_playoff_teams_other_conf",
                         ],
        },
        "qualification": {
            "playoff": [f"Eastern {i}" for i in range(1, 9)] + [f"Western {i}" for i in range(1, 9)]
        },
        "knockout_bracket": [
            ("Eastern 1", "Eastern 8"),
            ("Eastern 2", "Eastern 7"),
            ("Eastern 3", "Eastern 6"),
            ("Eastern 4", "Eastern 5"),
            ("Western 1", "Western 8"),
            ("Western 2", "Western 7"),
            ("Western 3", "Western 6"),
            ("Western 4", "Western 5"),
        ],
        "knockout_format": {
            "po_r16": "best_of_7",
            "po_r8": "best_of_7",
            "po_r4": "best_of_7",
            "po_r2": "best_of_7",
        },
        "knockout_draw_status": "no_draw",
        "knockout_draw": None,
        "knockout_reseeding": False,
        "league_type": "NBA",
        "has_play_in": True,
    }


@pytest.fixture
def nba_league_rules_playoff():
    return {
        "sim_type": "winner",
        "has_knockout": True,
        "classification": {
            "division": ["win_loss_pct"
                         ],
            "conference": [
                         "win_loss_pct",
                         "h2h_win_loss_pct",
                         "win_loss_pct_div_if_same_div",
                         "win_loss_pct_conf",
                         "win_loss_pct_playoff_teams_same_conf",
                         "win_loss_pct_playoff_teams_other_conf",
                         ],
            "league": [
                         "win_loss_pct",
                         "h2h_win_loss_pct",
                         "win_loss_pct_div",
                         "win_loss_pct_conf",
                         "win_loss_pct_playoff_teams_same_conf",
                         "win_loss_pct_playoff_teams_other_conf",
                         ],
        },
        "qualification": {
            "playoff": [f"Eastern {i}" for i in range(1, 9)] + [f"Western {i}" for i in range(1, 9)]
        },
        "knockout_bracket": [
            ("Eastern 1", "Eastern 8"),
            ("Eastern 2", "Eastern 7"),
            ("Eastern 3", "Eastern 6"),
            ("Eastern 4", "Eastern 5"),
            ("Western 1", "Western 8"),
            ("Western 2", "Western 7"),
            ("Western 3", "Western 6"),
            ("Western 4", "Western 5"),
        ],
        "knockout_format": {
            "po_r16": "best_of_7",
            "po_r8": "best_of_7",
            "po_r4": "best_of_7",
            "po_r2": "best_of_7",
        },
        "knockout_draw_status": "no_draw",
        "knockout_draw": [
            # Eastern Conference — First round
            ("Cleveland Cavaliers", "Miami Heat"),
            ("Indiana Pacers", "Milwaukee Bucks"),
            ("Boston Celtics", "Orlando Magic"),
            ("New York Knicks", "Detroit Pistons"),
            # Western Conference — First round
            ("Oklahoma City Thunder", "Memphis Grizzlies"),
            ("Denver Nuggets", "Los Angeles Clippers"),
            ("Houston Rockets", "Golden State Warriors"),
            ("Los Angeles Lakers", "Minnesota Timberwolves")
        ],
        "knockout_reseeding": False,
        "league_type": "NBA",
        "has_play_in": True,
    }

# ELOS fixtures
@pytest.fixture
def csv_nba_divisions():
    return pd.read_csv("tests/data/divisions/nba_divisions.csv")

@pytest.fixture
def final_nba_results():
    return {
    "eliminated in regular season": [
        "Atlanta Hawks",
        "Brooklyn Nets",
        "Charlotte Hornets",
        "Chicago Bulls",
        "Phoenix Suns",
        "Portland Trail Blazers",
        "Sacramento Kings",
        "San Antonio Spurs",
        "New Orleans Pelicans",
        "Utah Jazz",
        "Toronto Raptors",
        "Washington Wizards",
        "Philadelphia 76ers",
        "Dallas Mavericks",
    ],
    "eliminated in first round": [
        "Miami Heat",
        "Detroit Pistons",
        "Milwaukee Bucks",
        "Orlando Magic",
        "Memphis Grizzlies",
        "Golden State Warriors",
        "Houston Rockets",
        "Los Angeles Lakers",
        "Los Angeles Clippers"
    ],
    "eliminated in conference semifinals": [
        "Cleveland Cavaliers",
        "Boston Celtics",
        "Denver Nuggets",
        "Golden State Warriors"
    ],
    "eliminated in conference finals": [
        "New York Knicks",
        "Minnesota Timberwolves"
    ],
    "runner-up": ["Indiana Pacers"],
    "champion": ["Oklahoma City Thunder"]
    }

class TestSimulateLeague:
    """Test cases for simulate_league function."""

    def assert_nba_league_summary(self, result, mock_schedule):
        # check all columns present
        expected_columns = [
            "team",
            'Eastern 1',
            'Eastern 2',
            'Eastern 3',
            'Eastern 4',
            'Eastern 5',
            'Eastern 6',
            'Eastern 7',
            'Eastern 8',
            'Eastern 9',
            'Eastern 10',
            'Eastern 11',
            'Eastern 12',
            'Eastern 13',
            'Eastern 14',
            'Eastern 15',
            'Western 1',
            'Western 2',
            'Western 3',
            'Western 4',
            'Western 5',
            'Western 6',
            'Western 7',
            'Western 8',
            'Western 9',
            'Western 10',
            'Western 11',
            'Western 12',
            'Western 13',
            'Western 14',
            'Western 15',
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
        assert np.isclose(col_sum, 16.0, atol=1e-3)

        col_sum = result["po_r16"].sum()
        assert np.isclose(col_sum, 16.0, atol=1e-3)

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

    def test_simulate_league_nba_case_1(
        self,
        csv_schedule_data_nba_case_1,
        csv_elos_data_nba_case_1,
        nba_league_rules,
        csv_nba_divisions
    ):
        """Test simulating a nba league."""
        # Setup
        league_rules = nba_league_rules
        schedule = csv_schedule_data_nba_case_1
        elos = csv_elos_data_nba_case_1
        divisions = csv_nba_divisions
        result = simulate_league(
            league_rules, schedule, elos, divisions, num_simulations=10
        )
        self.assert_nba_league_summary(result, schedule)


    def test_simulate_league_nba_case_2(
        self,
        csv_schedule_data_nba_case_2,
        csv_elos_data_nba_case_1,
        nba_league_rules,
        csv_nba_divisions
    ):
        """Test simulating a nba league."""
        # Setup
        league_rules = nba_league_rules
        schedule = csv_schedule_data_nba_case_2
        elos = csv_elos_data_nba_case_1
        divisions = csv_nba_divisions
        result = simulate_league(
            league_rules, schedule, elos, divisions, num_simulations=10
        )
        self.assert_nba_league_summary(result, schedule)

    def test_simulate_league_nba_case_3(
        self,
        csv_schedule_data_nba_case_3,
        csv_elos_data_nba_case_1,
        nba_league_rules,
        csv_nba_divisions,
        final_nba_results,
    ):
        """Test simulating a nba league."""
        # Setup
        league_rules = nba_league_rules
        schedule = csv_schedule_data_nba_case_3
        elos = csv_elos_data_nba_case_1
        divisions = csv_nba_divisions
        result = simulate_league(
            league_rules, schedule, elos, divisions, num_simulations=10
        )
        self.assert_nba_league_summary(result, schedule)

        eliminated_in_season = final_nba_results["eliminated in regular season"]
        advanced_to_playoff = [
            team
            for stage, teams in final_nba_results.items()
            if stage != "eliminated in regular season"
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
        
    def test_simulate_league_nba_case_4(
        self,
        csv_schedule_data_nba_case_4,
        csv_elos_data_nba_case_1,
        nba_league_rules_playoff,
        csv_nba_divisions,
        final_nba_results,
    ):
        """Test simulating a nba league."""
        # Setup
        league_rules = nba_league_rules_playoff
        schedule = csv_schedule_data_nba_case_4
        elos = csv_elos_data_nba_case_1
        divisions = csv_nba_divisions
        result = simulate_league(
            league_rules, schedule, elos, divisions, num_simulations=10
        )
        self.assert_nba_league_summary(result, schedule)

        eliminated_in_season = final_nba_results["eliminated in regular season"]
        advanced_to_playoff = [
            team
            for stage, teams in final_nba_results.items()
            if stage != "eliminated in regular season"
            for team in teams
        ]
        advanced_to_conf_semis = [
            team
            for stage, teams in final_nba_results.items()
            if stage in ["eliminated in conference semifinals", "eliminated in conference finals", "runner-up", "champion"]
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
        for rounds in ["playoff","po_r16","po_r8"]:
            assert np.isclose(
                (result.loc[result["team"].isin(advanced_to_conf_semis)][rounds]==1).all(),
                1.0,
                atol=1e-3,
            )

    # def test_simulate_league_nba_case_5(
    #     self,
    #     csv_schedule_data_nba_case_5,
    #     csv_elos_data_nba_case_1,
    #     nba_league_rules_playoff,
    #     csv_nba_divisions,
    #     final_nba_results,
    # ):
    #     """Test simulating a nba league."""
    #     # Setup
    #     league_rules = nba_league_rules_playoff
    #     schedule = csv_schedule_data_nba_case_5
    #     elos = csv_elos_data_nba_case_1
    #     divisions = csv_nba_divisions
    #     result = simulate_league(
    #         league_rules, schedule, elos, divisions, num_simulations=10
    #     )
    #     self.assert_nba_league_summary(result, schedule)

    #     eliminated_in_season = final_nba_results["eliminated in regular season"]
    #     advanced_to_playoff = [
    #         team
    #         for stage, teams in final_nba_results.items()
    #         if stage != "eliminated in regular season"
    #         for team in teams
    #     ]
    #     first_round_bye = final_nba_results["first round bye"]
    #     advanced_to_divisional = [
    #         team
    #         for stage, teams in final_nba_results.items()
    #         if stage in ["eliminated in divisional", "eliminated in conference", "runner-up", "champion"]
    #         for team in teams
    #     ]
    #     advanced_to_conference = ["Washington Commanders","Kansas City Chiefs"]

    #     for rounds in ["playoff","po_r16", "po_r8", "po_r4", "po_r2", "po_champion"]:
    #         assert np.isclose(
    #             result.loc[result["team"].isin(eliminated_in_season)][rounds].all(),
    #             0.0,
    #             atol=1e-3,
    #         )
    #     for rounds in ["playoff","po_r16"]:
    #         assert np.isclose(
    #             (result.loc[result["team"].isin(advanced_to_playoff)][rounds]==1).all(),
    #             1.0,
    #             atol=1e-3,
    #         )
    #     for rounds in ["playoff","po_r16","po_r8"]:
    #         assert np.isclose(
    #             (result.loc[result["team"].isin(advanced_to_divisional)][rounds]==1).all(),
    #             1.0,
    #             atol=1e-3,
    #         )
    #     for rounds in ["playoff","po_r16","po_r8","po_r4"]:
    #         assert np.isclose(
    #             (result.loc[result["team"].isin(advanced_to_conference)][rounds]==1).all(),
    #             1.0,
    #             atol=1e-3,
    #         )

if __name__ == "__main__":
    pytest.main([__file__])
