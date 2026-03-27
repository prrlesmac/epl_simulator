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


# case 1: wc not started
@pytest.fixture
def csv_schedule_data_domestic_case_1():
    return read_schedule_csv("tests/data/schedules/schedule_fifa_wc_case_1.csv")

# ELOS fixtures
@pytest.fixture
def csv_elos_fifa_wc():
    return pd.read_csv("tests/data/elos/current_elos_fifa_wc.csv")

# ELOS fixtures
@pytest.fixture
def csv_fifa_wc_divisions():
    return pd.read_csv("tests/data/divisions/fifa_wc_divisions.csv")


# UCL ko draw not done
@pytest.fixture
def fifa_wc_rules_group_stage():
    return {
        "sim_type": "goals",
        "home_advantage": 80,
        "has_knockout": True,
        "classification": {
            "league": [
            "points",
            "goal_difference",
            "goals_for",
            "away_goals_for"
            ]
        },
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
        "knockout_draw_status": "pending_draw",
        "knockout_draw": None,
        "knockout_reseeding": False,
        "league_type": "UEFA"
    }

# UCL league rules after first draw has been done
@pytest.fixture
def fifa_wc_rules_knockout_stage():
    return {
        "sim_type": "goals",
        "home_advantage": 80,
        "has_knockout": True,
        "classification": {
            "league": [
            "points",
            "goal_difference",
            "goals_for",
            "away_goals_for"
            ]
        },
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
        "knockout_draw_status": "pending_uefa_second_draw",
        # make sure order is respected
        # 1-Bye
        # 16-17
        # 8-Bye
        # 9-10 etc...
        "knockout_draw": [
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
        "knockout_reseeding": False,
        "league_type": "UEFA",
    }


@pytest.fixture
def final_fifa_wc_results():
    return {
        "eliminated in group stage": [
            "Bologna",
            "Red Star",
            "Girona",
            "RB Leipzig",
            "Slovan Bratislava",
            "Sparta Prague",
            "Sturm Graz",
            "Young Boys",
            "Dinamo Zagreb",
            "Stuttgart",
            "Shakhtar",
            "RB Salzburg",
        ],
        "eliminated in playoff": [
            "Monaco",
            "Brest",
            "Juventus",
            "Celtic",
            "Manchester City",
            "Sporting CP",
            "Atalanta",
            "Milan",
        ],
        "eliminated in round of 16": [
            "Feyenoord",
            "Lille",
            "Liverpool",
            "Leverkusen",
            "Club Brugge",
            "PSV Eindhoven",
            "Benfica",
            "Atlético Madrid",
        ],
        "eliminated in quarterfinals": [
            "Aston Villa",
            "Borussia Dortmund",
            "Real Madrid",
            "Bayern Munich",
        ],
        "eliminated in semifinals": ["Barcelona", "Arsenal"],
        "runner-up": ["Inter"],
        "champion": ["Paris S-G"],
    }


class TestSimulateLeague:
    """Test cases for simulate_league function."""

    def assert_continental_league_summary(self, result, mock_schedule):
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
            "33",
            "34",
            "35",
            "36",
            "po_r32",
            "po_r16",
            "po_r8",
            "po_r4",
            "po_r2",
            "po_champion",
            "direct_to_round_of_16",
            "playoff",
            "updated_at",
        ]
        assert list(result.columns) == expected_columns

        # check the sum of all columns equals to 1
        subset_cols = [
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
            "33",
            "34",
            "35",
            "36",
            "po_champion",
        ]
        row_sums = result[subset_cols].sum()
        assert np.allclose(row_sums, 1.0, atol=1e-3), "Row sums not close to 1 ± 1e-3"
        col_sum = result["po_r32"].sum()
        assert np.isclose(col_sum, 24.0, atol=1e-3)
        col_sum = result["po_r16"].sum()
        assert np.isclose(col_sum, 16.0, atol=1e-3)
        col_sum = result["po_r8"].sum()
        assert np.isclose(col_sum, 8.0, atol=1e-3)
        col_sum = result["po_r4"].sum()
        assert np.isclose(col_sum, 4.0, atol=1e-3)
        col_sum = result["po_r2"].sum()
        assert np.isclose(col_sum, 2.0, atol=1e-3)
        col_sum = result["direct_to_round_of_16"].sum()
        assert np.isclose(col_sum, 8.0, atol=1e-3)
        col_sum = result["playoff"].sum()
        assert np.isclose(col_sum, 16.0, atol=1e-3)
        # check all teams are the one in fixtures
        assert set(result["team"]) == set(mock_schedule["home"]), "Values don't match"
        assert set(result["team"]) == set(mock_schedule["away"]), "Values don't match"

    def test_simulate_league_continental_case_1(
        self,
        csv_schedule_data_continental_case_1,
        csv_elos_data_continental,
        continental_league_rules_group_stage,
    ):
        """Test simulating a continental league."""
        # Setup
        league_rules = continental_league_rules_group_stage
        mock_schedule = csv_schedule_data_continental_case_1
        mock_elos = csv_elos_data_continental
        result = simulate_league(
            league_rules, mock_schedule, mock_elos, divisions=None, num_simulations=10
        )

        self.assert_continental_league_summary(result, mock_schedule)

    def test_simulate_league_continental_case_2(
        self,
        csv_schedule_data_continental_case_2,
        csv_elos_data_continental,
        continental_league_rules_group_stage,
    ):
        """Test simulating a continental league."""
        # Setup
        league_rules = continental_league_rules_group_stage
        mock_schedule = csv_schedule_data_continental_case_2
        mock_elos = csv_elos_data_continental
        result = simulate_league(
            league_rules, mock_schedule, mock_elos, divisions=None, num_simulations=10
        )

        self.assert_continental_league_summary(result, mock_schedule)

        # positions
        qualified = ["Barcelona", "Liverpool"]
        eliminated = ["RB Leipzig", "Young Boys", "Slovan Bratislava"]
        eliminated_from_direct_r16 = [
            "Sparta Prague",
            "Shakhtar",
            "Girona",
            "Red Star",
            "Sturm Graz",
            "RB Salzburg",
            "Bologna",
            "RB Leipzig",
            "Young Boys",
            "Slovan Bratislava",
        ]
        assert np.isclose(
            result.loc[result["team"].isin(qualified)]["po_r32"].all(), 1.0, atol=1e-3
        )
        assert np.isclose(
            result.loc[result["team"].isin(eliminated)]["po_r32"].all(), 0.0, atol=1e-3
        )
        assert np.isclose(
            result.loc[result["team"].isin(eliminated)]["playoff"].all(), 0.0, atol=1e-3
        )
        assert np.isclose(
            result.loc[result["team"].isin(eliminated_from_direct_r16)][
                "direct_to_round_of_16"
            ].all(),
            0.0,
            atol=1e-3,
        )

    def test_simulate_league_continental_case_3(
        self,
        csv_schedule_data_continental_case_3_4,
        csv_elos_data_continental,
        continental_league_rules_group_stage,
        final_ucl_results,
    ):
        """Test simulating a continental league."""
        # Setup
        league_rules = continental_league_rules_group_stage
        mock_schedule = csv_schedule_data_continental_case_3_4
        mock_elos = csv_elos_data_continental
        result = simulate_league(
            league_rules, mock_schedule, mock_elos, divisions=None, num_simulations=10
        )

        self.assert_continental_league_summary(result, mock_schedule)

        eliminated_in_gs = final_ucl_results["eliminated in group stage"]
        qualified_to_ko = [
            team
            for stage, teams in final_ucl_results.items()
            if stage != "eliminated in group stage"
            for team in teams
        ]
        qualified_to_r16 = [
            "Arsenal",
            "Atletico Madrid",
            "Liverpool",
            "Aston Villa",
            "Barcelona",
            "Lille",
            "Bayer Leverkusen",
            "Inter Milan",
        ]
        for rounds in ["po_r32", "po_r16", "po_r8", "po_r4", "po_r2", "po_champion"]:
            assert np.isclose(
                result.loc[result["team"].isin(eliminated_in_gs)][rounds].all(),
                0.0,
                atol=1e-3,
            )
        for rounds in ["po_r32"]:
            assert np.isclose(
                result.loc[result["team"].isin(qualified_to_ko)][rounds].all(),
                1.0,
                atol=1e-3,
            )
        assert np.isclose(
            result.loc[result["team"].isin(qualified_to_r16)]["po_r16"].all(),
            1.0,
            atol=1e-3,
        )

    def test_simulate_league_continental_case_4(
        self,
        csv_schedule_data_continental_case_3_4,
        csv_elos_data_continental,
        continental_league_rules_knockout_stage_first_draw,
        final_ucl_results,
    ):
        """Test simulating a continental league."""
        # Setup
        league_rules = continental_league_rules_knockout_stage_first_draw
        mock_schedule = csv_schedule_data_continental_case_3_4
        mock_elos = csv_elos_data_continental
        result = simulate_league(
            league_rules, mock_schedule, mock_elos, divisions=None, num_simulations=10
        )

        self.assert_continental_league_summary(result, mock_schedule)

        eliminated_in_gs = final_ucl_results["eliminated in group stage"]
        qualified_to_ko = [
            team
            for stage, teams in final_ucl_results.items()
            if stage != "eliminated in group stage"
            for team in teams
        ]
        qualified_to_r16 = [
            "Arsenal",
            "Atletico Madrid",
            "Liverpool",
            "Aston Villa",
            "Barcelona",
            "Lille",
            "Bayer Leverkusen",
            "Inter Milan",
        ]
        teams_in_same_path = [
            [
                "Liverpool",
                "Barcelona",
                "Paris S-G",
                "Benfica",
                "Monaco",
                "Brest"
            ],
            [
                "Arsenal",
                "Inter",
                "Milan",
                "PSV Eindhoven",
                "Feyenoord",
                "Juventus"
            ],
            [
                "Atlético Madrid",
                "Leverkusen",
                "Real Madrid",
                "Bayern Munich",
                "Celtic",
                "Manchester City"
            ],
            [
                "Lille",
                "Aston Villa",
                "Atalanta",
                "Dortmund",
                "Sporting CP",
                "Club Brugge"
            ],
        ]
        for rounds in ["po_r32", "po_r16", "po_r8", "po_r4", "po_r2", "po_champion"]:
            assert np.isclose(
                result.loc[result["team"].isin(eliminated_in_gs)][rounds].all(),
                0.0,
                atol=1e-3,
            )
        for rounds in ["po_r32"]:
            assert np.isclose(
                result.loc[result["team"].isin(qualified_to_ko)][rounds].all(),
                1.0,
                atol=1e-3,
            )
        assert np.isclose(
            result.loc[result["team"].isin(qualified_to_r16)]["po_r16"].all(),
            1.0,
            atol=1e-3,
        )
        for teams in teams_in_same_path:
            assert np.isclose(
                result.loc[result["team"].isin(teams)]["po_r8"].sum(),
                2.0,
                atol=1e-3,
            )

    def test_simulate_league_continental_case_5(
        self,
        csv_schedule_data_continental_case_5,
        csv_elos_data_continental,
        continental_league_rules_knockout_stage_second_draw,
        final_ucl_results,
    ):
        """Test simulating a continental league."""
        # Setup
        league_rules = continental_league_rules_knockout_stage_second_draw
        mock_schedule = csv_schedule_data_continental_case_5
        mock_elos = csv_elos_data_continental
        result = simulate_league(
            league_rules, mock_schedule, mock_elos, divisions=None, num_simulations=10
        )

        self.assert_continental_league_summary(result, mock_schedule)

        eliminated_in_gs = final_ucl_results["eliminated in group stage"]
        eliminated_in_playoff = final_ucl_results["eliminated in playoff"]
        qualified_to_r16 = [
            team
            for stage, teams in final_ucl_results.items()
            if stage not in ["eliminated in group stage", "eliminated in playoff"]
            for team in teams
        ]
        for rounds in ["po_r32", "po_r16", "po_r8", "po_r4", "po_r2", "po_champion"]:
            assert np.isclose(
                result.loc[result["team"].isin(eliminated_in_gs)][rounds].all(),
                0.0,
                atol=1e-3,
            )
        for rounds in ["po_r16", "po_r8", "po_r4", "po_r2", "po_champion"]:
            assert np.isclose(
                result.loc[result["team"].isin(eliminated_in_playoff)][rounds].all(),
                0.0,
                atol=1e-3,
            )
        for rounds in ["po_r32"]:
            assert np.isclose(
                result.loc[result["team"].isin(eliminated_in_playoff)][rounds].all(),
                1.0,
                atol=1e-3,
            )
        for rounds in ["po_r32", "po_r16"]:
            assert np.isclose(
                result.loc[result["team"].isin(qualified_to_r16)][rounds].all(),
                1.0,
                atol=1e-3,
            )

    def test_simulate_league_continental_case_6(
        self,
        csv_schedule_data_continental_case_6,
        csv_elos_data_continental,
        continental_league_rules_knockout_stage_second_draw,
        final_ucl_results,
    ):
        """Test simulating a continental league."""
        # Setup
        league_rules = continental_league_rules_knockout_stage_second_draw
        mock_schedule = csv_schedule_data_continental_case_6
        mock_elos = csv_elos_data_continental
        result = simulate_league(
            league_rules, mock_schedule, mock_elos, divisions=None, num_simulations=10
        )

        self.assert_continental_league_summary(result, mock_schedule)

        eliminated_in_gs = final_ucl_results["eliminated in group stage"]
        eliminated_in_playoff = final_ucl_results["eliminated in playoff"]
        qualified_to_r16 = [
            team
            for stage, teams in final_ucl_results.items()
            if stage not in ["eliminated in group stage", "eliminated in playoff"]
            for team in teams
        ]
        for rounds in ["po_r32", "po_r16", "po_r8", "po_r4", "po_r2", "po_champion"]:
            assert np.isclose(
                result.loc[result["team"].isin(eliminated_in_gs)][rounds].all(),
                0.0,
                atol=1e-3,
            )
        for rounds in ["po_r16", "po_r8", "po_r4", "po_r2", "po_champion"]:
            assert np.isclose(
                result.loc[result["team"].isin(eliminated_in_playoff)][rounds].all(),
                0.0,
                atol=1e-3,
            )
        for rounds in ["po_r32"]:
            assert np.isclose(
                result.loc[result["team"].isin(eliminated_in_playoff)][rounds].all(),
                1.0,
                atol=1e-3,
            )
        for rounds in ["po_r32", "po_r16"]:
            assert np.isclose(
                result.loc[result["team"].isin(qualified_to_r16)][rounds].all(),
                1.0,
                atol=1e-3,
            )

    def test_simulate_league_continental_case_7(
        self,
        csv_schedule_data_continental_case_7,
        csv_elos_data_continental,
        continental_league_rules_knockout_stage_second_draw,
        final_ucl_results,
    ):
        """Test simulating a continental league."""
        # Setup
        league_rules = continental_league_rules_knockout_stage_second_draw
        mock_schedule = csv_schedule_data_continental_case_7
        mock_elos = csv_elos_data_continental
        result = simulate_league(
            league_rules, mock_schedule, mock_elos, divisions=None, num_simulations=10
        )

        self.assert_continental_league_summary(result, mock_schedule)

        eliminated_in_gs = final_ucl_results["eliminated in group stage"]
        eliminated_in_playoff = final_ucl_results["eliminated in playoff"]
        eliminated_in_r16 = final_ucl_results["eliminated in round of 16"]
        eliminated_in_qf = final_ucl_results["eliminated in quarterfinals"]
        eliminated_in_sf = final_ucl_results["eliminated in semifinals"]
        runner_up = final_ucl_results["runner-up"]
        champion = final_ucl_results["champion"]

        for rounds in ["po_r32", "po_r16", "po_r8", "po_r4", "po_r2", "po_champion"]:
            assert np.isclose(
                result.loc[result["team"].isin(eliminated_in_gs)][rounds].all(),
                0.0,
                atol=1e-3,
            )
        for rounds in ["po_r16", "po_r8", "po_r4", "po_r2", "po_champion"]:
            assert np.isclose(
                result.loc[result["team"].isin(eliminated_in_playoff)][rounds].all(),
                0.0,
                atol=1e-3,
            )
        for rounds in ["po_r32"]:
            assert np.isclose(
                result.loc[result["team"].isin(eliminated_in_playoff)][rounds].all(),
                1.0,
                atol=1e-3,
            )
        for rounds in ["po_r8", "po_r4", "po_r2", "po_champion"]:
            assert np.isclose(
                result.loc[result["team"].isin(eliminated_in_r16)][rounds].all(),
                0.0,
                atol=1e-3,
            )
        for rounds in ["po_r32", "po_r16"]:
            assert np.isclose(
                result.loc[result["team"].isin(eliminated_in_r16)][rounds].all(),
                1.0,
                atol=1e-3,
            )
        for rounds in ["po_r4", "po_r2", "po_champion"]:
            assert np.isclose(
                result.loc[result["team"].isin(eliminated_in_qf)][rounds].all(),
                0.0,
                atol=1e-3,
            )
        for rounds in ["po_r32", "po_r16", "po_r8"]:
            assert np.isclose(
                result.loc[result["team"].isin(eliminated_in_qf)][rounds].all(),
                1.0,
                atol=1e-3,
            )
        for rounds in ["po_r2", "po_champion"]:
            assert np.isclose(
                result.loc[result["team"].isin(eliminated_in_sf)][rounds].all(),
                0.0,
                atol=1e-3,
            )
        for rounds in ["po_r32", "po_r16", "po_r8", "po_r4"]:
            assert np.isclose(
                result.loc[result["team"].isin(eliminated_in_sf)][rounds].all(),
                1.0,
                atol=1e-3,
            )
        for rounds in ["po_champion"]:
            assert np.isclose(
                result.loc[result["team"].isin(runner_up)][rounds].all(), 0.0, atol=1e-3
            )
        for rounds in ["po_r32", "po_r16", "po_r8", "po_r4", "po_r2"]:
            assert np.isclose(
                result.loc[result["team"].isin(runner_up)][rounds].all(), 1.0, atol=1e-3
            )
        for rounds in ["po_r32", "po_r16", "po_r8", "po_r4", "po_r2", "po_champion"]:
            assert np.isclose(
                result.loc[result["team"].isin(champion)][rounds].all(), 1.0, atol=1e-3
            )



if __name__ == "__main__":
    pytest.main([__file__])
