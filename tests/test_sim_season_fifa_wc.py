import pytest
import pandas as pd
import numpy as np
from .conftest import schedule_dtypes
from config import bracket_configs

from simulator.sim_season import (
    simulate_league,
    split_and_merge_schedule,
    single_simulation,
)
from src.config.config import league_rules


def read_schedule_csv(filepath):
    return pd.read_csv(filepath, dtype=schedule_dtypes, keep_default_na=False)


@pytest.fixture
def csv_schedule_data_fifa_wc_case_1():
    """
    Reads the FIFA World Cup schedule from the CSV file.
    """
    return read_schedule_csv("tests/data/schedules/schedule_fifa_wc_case_1.csv")


@pytest.fixture
def csv_schedule_data_fifa_wc_case_2():
    return read_schedule_csv("tests/data/schedules/schedule_fifa_wc_case_2.csv")


@pytest.fixture
def csv_schedule_data_fifa_wc_case_3():
    return read_schedule_csv("tests/data/schedules/schedule_fifa_wc_case_3.csv")


@pytest.fixture
def csv_schedule_data_fifa_wc_case_4():
    return read_schedule_csv("tests/data/schedules/schedule_fifa_wc_case_4.csv")


@pytest.fixture
def csv_schedule_data_fifa_wc_case_5():
    return read_schedule_csv("tests/data/schedules/schedule_fifa_wc_case_5.csv")


@pytest.fixture
def csv_schedule_data_fifa_wc_case_6():
    return read_schedule_csv("tests/data/schedules/schedule_fifa_wc_case_6.csv")


@pytest.fixture
def csv_schedule_data_fifa_wc_case_7():
    return read_schedule_csv("tests/data/schedules/schedule_fifa_wc_case_7.csv")


@pytest.fixture
def csv_elos_data_fifa_wc():
    """
    Reads the elo ratings for the teams in the World Cup from the CSV file.
    """
    return pd.read_csv("tests/data/elos/current_elos_fifa_wc_case_1.csv")


@pytest.fixture
def csv_divisions_data_fifa_wc():
    """
    Reads the group assignments for the teams in the World Cup from the CSV file.
    """
    return pd.read_csv("tests/data/divisions/fifa_wc_divisions.csv")

# UCL ko draw not done
@pytest.fixture
def fifa_wc_league_rules_group_stage():
    return {
        "sim_type": "goals",
        "home_advantage": 0,
        "has_knockout": True,
        "classification": {
            "division": [
                "points",
                "h2h_points",
                "h2h_goal_difference",
                "h2h_goals_for",
                "goal_difference",
                "goals_for",
            ],
            "league": [
                "points",
                "goal_difference",
                "goals_for",
            ],
        },
        "qualification": {
        },
        "knockout_bracket": [

            ("E1", "3rd vs E1"),
            ("I1", "3rd vs I1"),

            ("A2", "B2"),
            ("F1", "C2"),

            ("K2", "L2"),
            ("H1", "J2"),

            ("D1", "3rd vs D1"),
            ("G1", "3rd vs G1"),

            ("C1", "F2"),
            ("E2", "I2"),

            ("A1", "3rd vs A1"),
            ("L1", "3rd vs L1"),

            ("J1", "H2"),
            ("D2", "G2"),

            ("B1", "3rd vs B1"),
            ("K1", "3rd vs K1"),
        ],
        "knockout_format": {
            "po_r32": "single_game_neutral",
            "po_r16": "single_game_neutral",
            "po_r8": "single_game_neutral",
            "po_r4": "single_game_neutral",
            "po_r2": "single_game_neutral",
        },
        # pre season
        # "knockout_draw_status": "pending_draw",
        # "knockout_draw": None,
        "knockout_draw_status": "no_draw",
        "knockout_draw": None,
        "knockout_reseeding": False,
        "knockout_third_place_mapping": bracket_configs.fifa_wc_third_place_mapping,
        "league_type": "FIFA_WC"
    }

class TestSimulateLeagueFifaWc:
    """Test cases for simulate_league function for FIFA World Cup."""

    def assert_fifa_wc_summary(self, result, mock_schedule):
        # Check all columns are present
        expected_columns = [
            "team",
            'po_r32',
            'po_r16',
            'po_r8',
            'po_r4',
            'po_r2',
            'po_champion',
            'updated_at'
        ]
        assert all(col in result.columns for col in expected_columns)

        # Check sums for knockout rounds
        assert np.isclose(result["po_r32"].sum(), 32.0, atol=1e-3)
        assert np.isclose(result["po_r16"].sum(), 16.0, atol=1e-3)
        assert np.isclose(result["po_r8"].sum(), 8.0, atol=1e-3)
        assert np.isclose(result["po_r4"].sum(), 4.0, atol=1e-3)
        assert np.isclose(result["po_r2"].sum(), 2.0, atol=1e-3)
        assert np.isclose(result["po_champion"].sum(), 1.0, atol=1e-3)

        # Check all teams are present
        teams_in_schedule = pd.concat([mock_schedule["home"], mock_schedule["away"]]).unique()
        assert set(result["team"]) == set(teams_in_schedule)

    def test_simulate_league_fifa_wc_case_1(
        self,
        csv_schedule_data_fifa_wc_case_1,
        csv_elos_data_fifa_wc,
        csv_divisions_data_fifa_wc,
        fifa_wc_league_rules_group_stage
    ):
        """Test simulating a FIFA World Cup."""
        # Setup
        current_league_rules = fifa_wc_league_rules_group_stage
        mock_schedule = csv_schedule_data_fifa_wc_case_1
        mock_elos = csv_elos_data_fifa_wc
        mock_divisions = csv_divisions_data_fifa_wc

        result = simulate_league(
            current_league_rules, mock_schedule, mock_elos, divisions=mock_divisions, num_simulations=10
        )

        self.assert_fifa_wc_summary(result, mock_schedule)

    def test_simulate_league_fifa_wc_case_2(
        self,
        csv_schedule_data_fifa_wc_case_2,
        csv_elos_data_fifa_wc,
        csv_divisions_data_fifa_wc,
        fifa_wc_league_rules_group_stage,
    ):
        """Test simulating a FIFA World Cup with half group stage played."""
        current_league_rules = fifa_wc_league_rules_group_stage
        mock_schedule = csv_schedule_data_fifa_wc_case_2
        mock_elos = csv_elos_data_fifa_wc
        mock_divisions = csv_divisions_data_fifa_wc

        result = simulate_league(
            current_league_rules, mock_schedule, mock_elos, divisions=mock_divisions, num_simulations=10
        )
        print(result)
        self.assert_fifa_wc_summary(result, mock_schedule)

    def test_simulate_league_fifa_wc_case_3(
        self,
        csv_schedule_data_fifa_wc_case_3,
        csv_elos_data_fifa_wc,
        csv_divisions_data_fifa_wc,
        fifa_wc_league_rules_group_stage,
    ):
        """Test simulating a FIFA World Cup with all group stage played."""
        current_league_rules = fifa_wc_league_rules_group_stage
        mock_schedule = csv_schedule_data_fifa_wc_case_3
        mock_elos = csv_elos_data_fifa_wc
        mock_divisions = csv_divisions_data_fifa_wc

        result = simulate_league(
            current_league_rules, mock_schedule, mock_elos, divisions=mock_divisions, num_simulations=10
        )
        self.assert_fifa_wc_summary(result, mock_schedule)

        # positions
        qualified = [
            "Mexico",
            "Czechia",
            "Switzerland",
            "Qatar",
            "Haiti",
            "Scotland",
            "Paraguay",
            "Australia",
            "Ecuador",
            "Germany",
            "Sweden",
            "Japan",
            "New Zealand",
            "Egypt",
            "Spain",
            "Saudi Arabia",
            "France",
            "Norway",
            "Algeria",
            "Austria",
            "Uzbekistan",
            "Portugal",
            "Panama",
            "Ghana",
            "Croatia",
            "Senegal",
            "Cape Verde",
            "Belgium",
            "Netherlands",
            "Morocco",
            "South Africa",
            "Curaçao",
            ]
        print(result.loc[result["team"].isin(qualified)]["po_r32"])

        assert np.isclose(
            result.loc[result["team"].isin(qualified)]["po_r32"].all(), 1.0, atol=1e-3
        )
        # TODO tesrt eliminated
        # TODO add third places to qualified
        # TODO test third place ranks mapping
        # TODO test bracket pos

    def test_simulate_league_fifa_wc_case_4(
        self,
        csv_schedule_data_fifa_wc_case_4,
        csv_elos_data_fifa_wc,
        csv_divisions_data_fifa_wc,
        fifa_wc_league_rules_group_stage,
    ):
        """Test simulating a FIFA World Cup with KO draw."""
        current_league_rules = fifa_wc_league_rules_group_stage.copy()
        current_league_rules['knockout_draw_status'] = 'completed_draw'
        ko_teams = pd.concat([
            csv_schedule_data_fifa_wc_case_4[csv_schedule_data_fifa_wc_case_4['round'] != 'League']['home'],
            csv_schedule_data_fifa_wc_case_4[csv_schedule_data_fifa_wc_case_4['round'] != 'League']['away']
        ]).unique()
        current_league_rules['knockout_draw'] = [(ko_teams[i], ko_teams[i+1]) for i in range(0, len(ko_teams), 2)]
        mock_schedule = csv_schedule_data_fifa_wc_case_4
        mock_elos = csv_elos_data_fifa_wc
        mock_divisions = csv_divisions_data_fifa_wc

        result = simulate_league(
            current_league_rules, mock_schedule, mock_elos, divisions=mock_divisions, num_simulations=10
        )
        self.assert_fifa_wc_summary(result, mock_schedule)

    def test_simulate_league_fifa_wc_case_5(
        self,
        csv_schedule_data_fifa_wc_case_5,
        csv_elos_data_fifa_wc,
        csv_divisions_data_fifa_wc,
        fifa_wc_league_rules_group_stage,
    ):
        """Test simulating a FIFA World Cup with partial KO round played."""
        current_league_rules = fifa_wc_league_rules_group_stage.copy()
        current_league_rules['knockout_draw_status'] = 'completed_draw'
        ko_teams = pd.concat([
            csv_schedule_data_fifa_wc_case_5[csv_schedule_data_fifa_wc_case_5['round'] != 'League']['home'],
            csv_schedule_data_fifa_wc_case_5[csv_schedule_data_fifa_wc_case_5['round'] != 'League']['away']
        ]).unique()
        current_league_rules['knockout_draw'] = [(ko_teams[i], ko_teams[i+1]) for i in range(0, len(ko_teams), 2)]
        mock_schedule = csv_schedule_data_fifa_wc_case_5
        mock_elos = csv_elos_data_fifa_wc
        mock_divisions = csv_divisions_data_fifa_wc

        result = simulate_league(
            current_league_rules, mock_schedule, mock_elos, divisions=mock_divisions, num_simulations=10
        )
        self.assert_fifa_wc_summary(result, mock_schedule)

    def test_simulate_league_fifa_wc_case_6(
        self,
        csv_schedule_data_fifa_wc_case_6,
        csv_elos_data_fifa_wc,
        csv_divisions_data_fifa_wc,
        fifa_wc_league_rules_group_stage,
    ):
        """Test simulating a FIFA World Cup with later KO rounds pending."""
        current_league_rules = fifa_wc_league_rules_group_stage.copy()
        current_league_rules['knockout_draw_status'] = 'completed_draw'
        ko_teams = pd.concat([
            csv_schedule_data_fifa_wc_case_6[csv_schedule_data_fifa_wc_case_6['round'] != 'League']['home'],
            csv_schedule_data_fifa_wc_case_6[csv_schedule_data_fifa_wc_case_6['round'] != 'League']['away']
        ]).unique()
        current_league_rules['knockout_draw'] = [(ko_teams[i], ko_teams[i+1]) for i in range(0, len(ko_teams), 2)]
        mock_schedule = csv_schedule_data_fifa_wc_case_6
        mock_elos = csv_elos_data_fifa_wc
        mock_divisions = csv_divisions_data_fifa_wc

        result = simulate_league(
            current_league_rules, mock_schedule, mock_elos, divisions=mock_divisions, num_simulations=10
        )
        self.assert_fifa_wc_summary(result, mock_schedule)

    def test_simulate_league_fifa_wc_case_7(
        self,
        csv_schedule_data_fifa_wc_case_7,
        csv_elos_data_fifa_wc,
        csv_divisions_data_fifa_wc,
        fifa_wc_league_rules_group_stage,
    ):
        """Test simulating a finished FIFA World Cup."""
        current_league_rules = fifa_wc_league_rules_group_stage.copy()
        current_league_rules['knockout_draw_status'] = 'completed_draw'
        ko_teams = pd.concat([
            csv_schedule_data_fifa_wc_case_7[csv_schedule_data_fifa_wc_case_7['round'] != 'League']['home'],
            csv_schedule_data_fifa_wc_case_7[csv_schedule_data_fifa_wc_case_7['round'] != 'League']['away']
        ]).unique()
        current_league_rules['knockout_draw'] = [(ko_teams[i], ko_teams[i+1]) for i in range(0, len(ko_teams), 2)]
        mock_schedule = csv_schedule_data_fifa_wc_case_7
        mock_elos = csv_elos_data_fifa_wc
        mock_divisions = csv_divisions_data_fifa_wc

        result = simulate_league(
            current_league_rules, mock_schedule, mock_elos, divisions=mock_divisions, num_simulations=10
        )
        self.assert_fifa_wc_summary(result, mock_schedule)


class TestSingleSimulationFifaWc:
    def test_single_simulation_fifa_wc_case_1(
        self,
        csv_schedule_data_fifa_wc_case_1,
        csv_elos_data_fifa_wc,
        csv_divisions_data_fifa_wc,
        fifa_wc_league_rules_group_stage,
    ):
        """Test simulating a FIFA World Cup."""
        # Prepare data for simulation
        current_league_rules = fifa_wc_league_rules_group_stage
        schedule_played, schedule_pending = split_and_merge_schedule(
            csv_schedule_data_fifa_wc_case_1, csv_elos_data_fifa_wc, csv_divisions_data_fifa_wc
        )
        result = single_simulation(
            schedule_played,
            schedule_pending,
            None,
            csv_elos_data_fifa_wc,
            divisions=csv_divisions_data_fifa_wc,
            league_rules=current_league_rules,
        )

        assert isinstance(result, pd.DataFrame)
        assert set(["division_pos", "team", "points"]).issubset(result.columns)
        assert set(result["team"].unique()) == set(
            csv_schedule_data_fifa_wc_case_1["home"].unique()
        )
        assert set(result["team"].unique()) == set(
            csv_schedule_data_fifa_wc_case_1["away"].unique()
        )
        # Check that group positions are 1,2,3,4 for each group
        for group in result['division'].unique():
            group_positions = result[result['division'] == group]['division_pos'].sort_values().tolist()
            assert group_positions == [1, 2, 3, 4]
        
        assert result["po_r32"].sum() == 32
        assert result["po_r16"].sum() == 16
        assert result["po_r8"].sum() == 8
        assert result["po_r4"].sum() == 4
        assert result["po_r2"].sum() == 2
        assert result["po_champion"].sum() == 1
