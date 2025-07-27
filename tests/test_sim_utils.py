import pytest
from unittest.mock import patch
from io import StringIO
import numpy as np
import pandas as pd
import random
from simulator.sim_utils import (
    calculate_win_probability,
    simulate_match,
    simulate_playoff,
    simulate_extra_time,
    simulate_matches_data_frame,
    apply_h2h_tiebreaker,
    apply_playoff_tiebreaker,
    get_standings_metrics,
    get_standings,
    validate_bracket,
    simulate_playoff_bracket,
    _simulate_round,
    _get_tie_matches,
    _determine_winner_from_schedule,
    _get_winner_from_completed_matches,
    _get_winner_from_partial_matches,
    _get_winner_by_goals,
    _calculate_total_goals,
    _track_team_progression,
    _prepare_next_round,
    _build_results_dataframe,
    draw_from_pots,
    create_bracket_from_composition
)


class TestCalculateWinProbability:

    def test_equal_elo_single_game(self):
        """Test that equal Elo ratings with home advantage gives home team higher probability"""
        result = calculate_win_probability(1500, 1500, "single_game")
        # With home advantage of 80, home team should have > 0.5 probability
        assert pytest.approx(result, rel=1e-3) == 0.61314

    def test_equal_elo_neutral_venue(self):
        """Test that equal Elo ratings at neutral venue gives 50% probability"""
        result = calculate_win_probability(1500, 1500, "single_game_neutral")
        assert pytest.approx(result, rel=1e-6) == 0.5

    def test_equal_elo_two_legged(self):
        """Test that equal Elo ratings in two-legged matchup gives 50% probability"""
        result = calculate_win_probability(1500, 1500, "two-legged")
        assert pytest.approx(result, rel=1e-6) == 0.5

    def test_home_team_higher_elo_single_game(self):
        """Test home team with higher Elo rating"""
        result = calculate_win_probability(1600, 1500, "single_game")
        # Home team has 100 Elo advantage + 80 home advantage = 180 total
        assert pytest.approx(result, rel=1e-3) == 0.73811

    def test_away_team_higher_elo_single_game(self):
        """Test away team with higher Elo rating"""
        result = calculate_win_probability(1500, 1600, "single_game")
        # Away team has 100 Elo advantage but loses 80 home advantage = 20 net
        assert pytest.approx(result, rel=1e-3) == 0.47125

    def test_neutral_venue_higher_elo(self):
        """Test neutral venue with one team having higher Elo"""
        result = calculate_win_probability(1500, 1600, "single_game_neutral")
        # Away team has 100 Elo advantage, no home advantage
        assert pytest.approx(result, rel=1e-3) == 0.35994

    def test_two_legged_higher_elo(self):
        """Test two-legged matchup with multiplier effect"""
        result = calculate_win_probability(1500, 1600, "two-legged")
        # 100 Elo difference * 1.4 multiplier = 140 effective difference
        assert pytest.approx(result, rel=1e-3) == 0.30876

    def test_custom_home_advantage(self):
        """Test custom home advantage value"""
        result = calculate_win_probability(1500, 1500, "single_game", home_adv=100)
        # Higher home advantage should give home team even better probability
        assert pytest.approx(result, rel=1e-3) == 0.64006

    def test_zero_home_advantage(self):
        """Test with zero home advantage"""
        result = calculate_win_probability(1500, 1500, "single_game", home_adv=0)
        # Should be same as neutral venue
        assert pytest.approx(result, rel=1e-6) == 0.5

    def test_extreme_elo_differences(self):
        """Test with extreme Elo differences"""
        # Very strong home team
        result1 = calculate_win_probability(2000, 1000, "single_game")
        assert result1 > 0.99

        # Very strong away team
        result2 = calculate_win_probability(1000, 2000, "single_game")
        assert result2 < 0.01

    def test_invalid_matchup_type(self):
        """Test handling of invalid matchup type"""
        with patch("sys.stdout", new_callable=StringIO) as mock_stdout:
            result = calculate_win_probability(1500, 1500, "invalid_type")

            # Should print warning message
            output = mock_stdout.getvalue()
            assert "Unknown matchup type: invalid_type" in output
            assert "Defaulting to single game" in output

            # Should default to single game behavior
            assert result > 0.5  # Home advantage should apply

    def test_probability_symmetry(self):
        """Test that probabilities are symmetric (P(A beats B) + P(B beats A) ≈ 1)"""
        # For neutral venue (no home advantage complications)
        prob_a_beats_b = calculate_win_probability(1500, 1600, "single_game_neutral")
        prob_b_beats_a = calculate_win_probability(1600, 1500, "single_game_neutral")

        # Should sum to approximately 1
        assert pytest.approx(prob_a_beats_b + prob_b_beats_a, rel=1e-6) == 1.0


class TestSimulateMatch:

    def test_return_type_and_structure(self):
        """Test that function returns a tuple of two integers"""
        result = simulate_match(0.5)
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], (int, np.integer))
        assert isinstance(result[1], (int, np.integer))

    def test_non_negative_goals(self):
        """Test that goals are always non-negative"""
        for proba in [0.1, 0.3, 0.5, 0.7, 0.9]:
            for _ in range(100):
                gh, ga = simulate_match(proba)
                assert gh >= 0, f"Home goals should be non-negative, got {gh}"
                assert ga >= 0, f"Away goals should be non-negative, got {ga}"

    def test_boundary_probabilities(self):
        """Test edge cases with extreme probabilities"""
        # Test with very low probability
        gh, ga = simulate_match(0.01)
        assert isinstance(gh, (int, np.integer))
        assert isinstance(ga, (int, np.integer))
        assert gh >= 0 and ga >= 0

        # Test with very high probability
        gh, ga = simulate_match(0.99)
        assert isinstance(gh, (int, np.integer))
        assert isinstance(ga, (int, np.integer))
        assert gh >= 0 and ga >= 0

        # Test with exactly 0.5
        gh, ga = simulate_match(0.5)
        assert isinstance(gh, (int, np.integer))
        assert isinstance(ga, (int, np.integer))
        assert gh >= 0 and ga >= 0

    def test_deterministic_with_fixed_seed(self):
        """Test that results are deterministic with fixed random seed"""
        np.random.seed(42)
        result1 = simulate_match(0.6)

        np.random.seed(42)
        result2 = simulate_match(0.6)

        assert result1 == result2, "Results should be identical with same seed"

    def test_poisson_distribution_properties(self):
        """Test that the underlying Poisson distribution is working correctly"""
        # Mock numpy.random.poisson to test the calculation logic
        with patch("numpy.random.poisson") as mock_poisson:
            mock_poisson.side_effect = [1, 2, 1]  # Base, GH, GA

            result = simulate_match(0.6)

            # Should call poisson 3 times
            assert mock_poisson.call_count == 3

            # Result should be (2+1, 1+1) = (3, 2)
            assert result == (3, 2)


class TestSimulatePlayoffs:

    def test_invalid_probability_low(self):
        with pytest.raises(ValueError):
            simulate_playoff(-0.1)

    def test_invalid_probability_high(self):
        with pytest.raises(ValueError):
            simulate_playoff(1.5)

    def test_output_is_1_or_2_and_integer(self):
        # Run multiple simulations to ensure the result is always 1 or 2
        for _ in range(1000):
            result = simulate_playoff(0.5)
            assert isinstance(result, int), f"Expected int but got {type(result)}"
            assert result in (1, 2), f"Unexpected result: {result}"

    def test_result_bias_extreme_prob_1(self):
        # With proba = 1, team 1 should always win
        for _ in range(100):
            assert simulate_playoff(1.0) == 1

    def test_result_bias_extreme_prob_0(self):
        # With proba = 0, team 2 should always win
        for _ in range(100):
            assert simulate_playoff(0.0) == 2

    def test_simulate_extra_time_returns_valid_output(self):
        """Ensure output is always 1 or 2."""
        results = [simulate_extra_time(0.5) for _ in range(100)]
        assert all(result in [1, 2] for result in results)

class TestSimulateMatchesDataFrame:

    @pytest.fixture
    def sample_matches_df(self):
        """Create a sample DataFrame for testing."""
        return pd.DataFrame(
            {
                "elo_home": [1500, 1600, 1400, 1550],
                "elo_away": [1450, 1550, 1500, 1500],
                "neutral": ["N", "H", "N", "A"],
                "home_team": ["Team A", "Team B", "Team C", "Team D"],
                "away_team": ["Team X", "Team Y", "Team Z", "Team W"],
            }
        )

    def test_function_signature_and_return_type(self, sample_matches_df):
        """Test that function accepts correct parameters and returns DataFrame."""
        result = simulate_matches_data_frame(sample_matches_df)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(sample_matches_df)

    def test_original_columns_preserved(self, sample_matches_df):
        """Test that original DataFrame columns are preserved."""

        original_columns = set(sample_matches_df.columns)
        result = simulate_matches_data_frame(sample_matches_df)

        # Check that all original columns are preserved
        for col in original_columns:
            assert col in result.columns
            pd.testing.assert_series_equal(result[col], sample_matches_df[col])


class TestApplyH2HTiebreaker:

    def test_h2h_points_ranking_case_1(self):
        matches = pd.DataFrame(
            {
                "home": ["A", "B", "C"],
                "away": ["B", "C", "A"],
                "home_goals": [2, 0, 1],
                "away_goals": [0, 2, 2],
            }
        )
        tied_teams = ["A", "B", "C"]
        result = apply_h2h_tiebreaker(matches, tied_teams, "h2h_points")

        assert set(result.columns) == {"team", "h2h_points"}
        assert result.shape[0] == 3
        assert result.loc[result["team"] == "A", "h2h_points"].values[0] == 6
        assert result.loc[result["team"] == "B", "h2h_points"].values[0] == 0
        assert result.loc[result["team"] == "C", "h2h_points"].values[0] == 3

    def test_h2h_points_ranking_case_2(self):
        matches = pd.DataFrame(
            {
                "home": ["A", "B", "C"],
                "away": ["B", "C", "A"],
                "home_goals": [2, 2, 1],
                "away_goals": [0, 2, 1],
            }
        )
        tied_teams = ["A", "B"]
        result = apply_h2h_tiebreaker(matches, tied_teams, "h2h_points")

        assert set(result.columns) == {"team", "h2h_points"}
        assert result.shape[0] == 2
        assert result.loc[result["team"] == "A", "h2h_points"].values[0] == 3
        assert result.loc[result["team"] == "B", "h2h_points"].values[0] == 0

    def test_h2h_goal_diff_ranking(self):
        matches = pd.DataFrame(
            {
                "home": ["A", "B", "C"],
                "away": ["B", "C", "A"],
                "home_goals": [2, 0, 1],
                "away_goals": [0, 2, 2],
            }
        )
        tied_teams = ["A", "B", "C"]
        result = apply_h2h_tiebreaker(matches, tied_teams, "h2h_goal_difference")
        assert set(result.columns) == {"team", "h2h_goal_difference"}
        assert result.shape[0] == 3
        assert result.loc[result["team"] == "A", "h2h_goal_difference"].values[0] == 3
        assert result.loc[result["team"] == "B", "h2h_goal_difference"].values[0] == -4
        assert result.loc[result["team"] == "C", "h2h_goal_difference"].values[0] == 1

    def test_ignores_non_tied_teams(self):
        matches = pd.DataFrame(
            {
                "home": ["A", "B", "C", "A"],
                "away": ["B", "C", "A", "D"],
                "home_goals": [2, 0, 1, 3],
                "away_goals": [0, 2, 2, 1],
            }
        )
        tied_teams = ["A", "B", "C"]
        result = apply_h2h_tiebreaker(matches, tied_teams, "h2h_points")

        assert "D" not in result["team"].values
        assert result.shape[0] == 3

    def test_empty_tied_teams_returns_empty_df(self):
        matches = pd.DataFrame(
            {
                "home": ["A", "B"],
                "away": ["B", "A"],
                "home_goals": [1, 1],
                "away_goals": [1, 1],
            }
        )
        result = apply_h2h_tiebreaker(matches, [], "h2h_points")
        assert result.empty

    def test_invalid_rule_column(self):
        matches = pd.DataFrame(
            {
                "home": ["A", "B"],
                "away": ["B", "A"],
                "home_goals": [1, 1],
                "away_goals": [1, 1],
            }
        )
        tied_teams = ["A", "B"]
        with pytest.raises(KeyError):
            apply_h2h_tiebreaker(matches, tied_teams, "h2h_unexpected_metric")


class TestApplyPlayoffTiebreaker:

    def test_two_team_playoff(self):
        matches = pd.DataFrame(
            {
                "home": ["A"],
                "away": ["B"],
                "home_goals": [2],
                "away_goals": [1],
                "elo_home": [1600],
                "elo_away": [1500],
            }
        )
        tied_teams = ["A", "B"]

        result = apply_playoff_tiebreaker(matches, tied_teams)

        assert result.shape[0] == 2
        assert set(result["playoff"]) == {0, 1}

    def test_multi_team_playoff(self):
        matches = pd.DataFrame(
            {
                "home": ["A", "C", "B", "D"],
                "away": ["B", "D", "C", "A"],
                "home_goals": [1, 2, 2, 0],
                "away_goals": [0, 0, 1, 3],
                "elo_home": [1600, 1450, 1500, 1400],
                "elo_away": [1500, 1400, 1450, 1600],
            }
        )
        tied_teams = ["A", "B", "C", "D"]

        result = apply_playoff_tiebreaker(matches, tied_teams)

        assert result.shape[0] == 4
        top_ranks = result[result["playoff"] >= 0]
        assert sorted(top_ranks["playoff"].tolist()) == [0, 1]
        assert result.loc[result["team"] == "A", "playoff"].values[0] in [0, 1]
        assert result.loc[result["team"] == "B", "playoff"].values[0] in [0, 1]

        bottom_ranks = result[result["playoff"] < 0]
        assert sorted(bottom_ranks["playoff"].tolist()) == [-2, -1]
        assert result.loc[result["team"] == "C", "playoff"].values[0] in [-2, -1]
        assert result.loc[result["team"] == "D", "playoff"].values[0] in [-2, -1]


class TestGetStandingsMetrics:

    class TestGetStandingsMetrics:
        def test_four_team_double_round_robin(self):
            matches = pd.DataFrame(
                [
                    {"home": "A", "away": "B", "home_goals": 2, "away_goals": 1},
                    {"home": "C", "away": "D", "home_goals": 0, "away_goals": 3},
                    {"home": "B", "away": "A", "home_goals": 0, "away_goals": 0},
                    {"home": "D", "away": "C", "home_goals": 1, "away_goals": 1},
                    {"home": "A", "away": "C", "home_goals": 1, "away_goals": 1},
                    {"home": "B", "away": "D", "home_goals": 0, "away_goals": 2},
                    {"home": "C", "away": "A", "home_goals": 0, "away_goals": 2},
                    {"home": "D", "away": "B", "home_goals": 2, "away_goals": 0},
                    {"home": "A", "away": "D", "home_goals": 1, "away_goals": 3},
                    {"home": "B", "away": "C", "home_goals": 1, "away_goals": 2},
                    {"home": "D", "away": "A", "home_goals": 0, "away_goals": 1},
                    {"home": "C", "away": "B", "home_goals": 1, "away_goals": 1},
                ]
            )

            result = (
                get_standings_metrics(matches)
                .sort_values("team")
                .reset_index(drop=True)
            )

            expected = {
                "A": {
                    "points": 11,
                    "goal_difference": 2,
                    "goals_for": 7,
                    "goals_against": 5,
                    "away_goals_for": 3,
                },
                "B": {
                    "points": 2,
                    "goal_difference": -6,
                    "goals_for": 3,
                    "goals_against": 9,
                    "away_goals_for": 2,
                },
                "C": {
                    "points": 6,
                    "goal_difference": -4,
                    "goals_for": 5,
                    "goals_against": 9,
                    "away_goals_for": 4,
                },
                "D": {
                    "points": 13,
                    "goal_difference": 8,
                    "goals_for": 11,
                    "goals_against": 3,
                    "away_goals_for": 8,
                },
            }

            for i, row in result.iterrows():
                team = row["team"]
                stats = expected[team]
                assert row["points"] == stats["points"], f"{team} points mismatch"
                assert (
                    row["goal_difference"] == stats["goal_difference"]
                ), f"{team} GD mismatch"
                assert row["goals_for"] == stats["goals_for"], f"{team} GF mismatch"
                assert (
                    row["goals_against"] == stats["goals_against"]
                ), f"{team} GA mismatch"
                assert (
                    row["away_goals_for"] == stats["away_goals_for"]
                ), f"{team} away GF mismatch"

        def test_no_matches(self):
            df = pd.DataFrame(columns=["home", "away", "home_goals", "away_goals"])

            result = get_standings_metrics(df)

            assert result.empty


import pandas as pd
import numpy as np


class TestGetStandings:
    def test_four_teams_with_tiebreakers(self):
        matches = pd.DataFrame(
            [
                {"home": "A", "away": "B", "home_goals": 1, "away_goals": 0},
                {"home": "C", "away": "D", "home_goals": 2, "away_goals": 2},
                {"home": "A", "away": "C", "home_goals": 0, "away_goals": 2},
                {"home": "B", "away": "D", "home_goals": 2, "away_goals": 0},
                {"home": "A", "away": "D", "home_goals": 1, "away_goals": 1},
                {"home": "B", "away": "C", "home_goals": 1, "away_goals": 2},
            ]
        )

        standings = (
            get_standings(
                matches_df=matches,
                classif_rules=["points", "goal_difference", "h2h_points"],
            )
            .sort_values("team")
            .reset_index(drop=True)
        )

        # Expected points and goal diff
        expected = {
            "A": {"points": 4, "goal_difference": -1},
            "B": {"points": 3, "goal_difference": 0},
            "C": {"points": 7, "goal_difference": 3},
            "D": {"points": 2, "goal_difference": -2},
        }

        for i, row in standings.iterrows():
            team = row["team"]
            assert row["points"] == expected[team]["points"], f"{team} points mismatch"
            assert (
                row["goal_difference"] == expected[team]["goal_difference"]
            ), f"{team} GD mismatch"

        # Ensure teams are ranked properly:
        # C should be first (6 pts), A and B tied on points and GD, resolved by h2h (A beat B)
        # So final order: C (1), A (2), B (3), D (4)
        expected_positions = {"C": 1, "A": 2, "B": 3, "D": 4}
        for _, row in standings.iterrows():
            assert (
                row["pos"] == expected_positions[row["team"]]
            ), f"{row['team']} position mismatch"

    def test_tie_broken_by_goal_difference(self):
        # Teams A and B tied on points but A has better goal difference
        matches = pd.DataFrame(
            [
                {"home": "A", "away": "B", "home_goals": 1, "away_goals": 1},
                {"home": "C", "away": "D", "home_goals": 0, "away_goals": 2},
                {"home": "B", "away": "C", "home_goals": 2, "away_goals": 0},
                {"home": "D", "away": "A", "home_goals": 0, "away_goals": 1},
                {"home": "A", "away": "C", "home_goals": 4, "away_goals": 1},
                {"home": "B", "away": "D", "home_goals": 1, "away_goals": 0},
            ]
        )

        standings = (
            get_standings(
                matches_df=matches, classif_rules=["points", "goal_difference"]
            )
            .sort_values("team")
            .reset_index(drop=True)
        )

        # Both A and B have 7 points; B has better goal difference
        assert standings.loc[standings["team"] == "A", "points"].iloc[0] == 7
        assert standings.loc[standings["team"] == "B", "points"].iloc[0] == 7
        assert (
            standings.loc[standings["team"] == "A", "goal_difference"].iloc[0]
            > standings.loc[standings["team"] == "B", "goal_difference"].iloc[0]
        )

        # B should rank higher (pos smaller)
        assert (
            standings.loc[standings["team"] == "A", "pos"].iloc[0]
            < standings.loc[standings["team"] == "B", "pos"].iloc[0]
        )

    def test_tie_broken_by_head_to_head(self):
        # Teams A and B tied on points and goal difference, h2h decides
        matches = pd.DataFrame(
            [
                {"home": "A", "away": "B", "home_goals": 2, "away_goals": 1},
                {"home": "C", "away": "D", "home_goals": 0, "away_goals": 2},
                {"home": "B", "away": "C", "home_goals": 2, "away_goals": 1},
                {"home": "D", "away": "A", "home_goals": 0, "away_goals": 1},
                {"home": "A", "away": "C", "home_goals": 0, "away_goals": 1},
                {"home": "B", "away": "D", "home_goals": 1, "away_goals": 0},
            ]
        )

        standings = (
            get_standings(
                matches_df=matches,
                classif_rules=["points", "goal_difference", "h2h_points"],
            )
            .sort_values("team")
            .reset_index(drop=True)
        )

        # A and B tied on points and GD, but A won both h2h matches, so A ranks higher
        assert (
            standings.loc[standings["team"] == "A", "pos"].iloc[0]
            < standings.loc[standings["team"] == "B", "pos"].iloc[0]
        )

    def test_three_way_tie_with_h2h(self):
        # Three teams tied on points and goal difference,
        # h2h points used to break tie between them
        matches = pd.DataFrame(
            [
                {"home": "A", "away": "B", "home_goals": 1, "away_goals": 2},
                {"home": "A", "away": "C", "home_goals": 1, "away_goals": 0},
                {"home": "B", "away": "C", "home_goals": 0, "away_goals": 2},
                {"home": "D", "away": "A", "home_goals": 0, "away_goals": 2},
                {"home": "D", "away": "B", "home_goals": 1, "away_goals": 2},
                {"home": "D", "away": "C", "home_goals": 1, "away_goals": 2},
            ]
        )
        standings = (
            get_standings(
                matches_df=matches,
                classif_rules=[
                    "points",
                    "goal_difference",
                    "h2h_points",
                    "h2h_goal_difference",
                ],
            )
            .sort_values("team")
            .reset_index(drop=True)
        )

        assert standings.loc[standings["team"] == "A", "pos"].iloc[0] == 1
        assert standings.loc[standings["team"] == "B", "pos"].iloc[0] == 3
        assert standings.loc[standings["team"] == "C", "pos"].iloc[0] == 2
        assert standings.loc[standings["team"] == "D", "pos"].iloc[0] == 4


class TestValidateBracket:

    def test_valid_bracket(self):
        # 8 teams, 4 matches (16 slots total in two columns)
        bracket = pd.DataFrame(
            {
                "team1": ["A", "B", "C", "D"],
                "team2": ["E", "F", "G", "H"],
            }
        )
        bracket_format = {
            "po_r8": "single-leg",
            "po_r4": "two-legged",
            "po_final": "single-leg",
        }

        # Should not raise
        validate_bracket(bracket, bracket_format)

    def test_empty_slots(self):
        bracket = pd.DataFrame(
            {
                "team1": ["A", None],
                "team2": ["B", " "],
            }
        )
        bracket_format = {"po_r4": "single-leg"}

        with pytest.raises(ValueError, match="empty team slots"):
            validate_bracket(bracket, bracket_format)

    def test_duplicate_teams(self):
        bracket = pd.DataFrame(
            {
                "team1": ["A", "B"],
                "team2": ["A", "C"],
            }
        )
        bracket_format = {"po_r4": "single-leg"}

        with pytest.raises(ValueError, match="Duplicate teams"):
            validate_bracket(bracket, bracket_format)

    def test_slots_not_power_of_two(self):
        bracket = pd.DataFrame(
            {
                "team1": ["A", "B", "C"],
                "team2": ["D", "E", "F"],
            }
        )
        bracket_format = {"po_r8": "single-leg"}

        with pytest.raises(ValueError, match="power of 2"):
            validate_bracket(bracket, bracket_format)

    def test_less_than_two_teams(self):
        bracket = pd.DataFrame(
            {
                "team1": ["Bye", "Bye"],
                "team2": ["Bye", "Bye"],
            }
        )
        bracket_format = {"po_r4": "single-leg"}

        with pytest.raises(ValueError, match="At least two teams"):
            validate_bracket(bracket, bracket_format)

    def test_mismatched_rounds(self):
        bracket = pd.DataFrame(
            {
                "team1": ["A", "B", "C", "D"],
                "team2": ["E", "F", "G", "H"],
            }
        )
        # Only 2 rounds instead of expected 3 for 8 teams
        bracket_format = {"po_r8": "single-leg", "po_r4": "two-legged"}

        with pytest.raises(ValueError, match="does not match the number of rounds"):
            validate_bracket(bracket, bracket_format)

    def test_bye_teams_allowed(self):
        bracket = pd.DataFrame(
            {
                "team1": ["A", "Bye", "C", "D"],
                "team2": ["Bye", "B", "Bye", "H"],
            }
        )
        bracket_format = {
            "po_r8": "single-leg",
            "po_r4": "two-legged",
            "po_final": "single-leg",
        }

        # Should not raise even though 'Bye's present
        validate_bracket(bracket, bracket_format)


import pandas as pd
import pytest
from unittest.mock import patch, MagicMock


class TestSimulatePlayoffBracket:

    def test_basic_16_team_bracket(self):
        bracket_df = pd.DataFrame(
            {
                "team1": [
                    "Team1",
                    "Team3",
                    "Team5",
                    "Team7",
                    "Team9",
                    "Team11",
                    "Team13",
                    "Team15",
                ],
                "team2": [
                    "Bye",
                    "Team4",
                    "Team6",
                    "Team8",
                    "Team10",
                    "Team12",
                    "Team14",
                    "Team16",
                ],
            }
        )

        bracket_format = {
            "po_r16": "two-legged",
            "po_r8": "two-legged",
            "po_r4": "two-legged",
            "po_r2": "single-game",
        }

        elos = pd.DataFrame(
            {
                "team": [
                    "Team1",
                    "Team2",
                    "Team3",
                    "Team4",
                    "Team5",
                    "Team6",
                    "Team7",
                    "Team8",
                    "Team9",
                    "Team10",
                    "Team11",
                    "Team12",
                    "Team13",
                    "Team14",
                    "Team15",
                    "Team16",
                ],
                "elo": [
                    1600,
                    1580,
                    1550,
                    1540,
                    1500,
                    1490,
                    1480,
                    1470,
                    1450,
                    1440,
                    1430,
                    1420,
                    1400,
                    1390,
                    1380,
                    1370,
                ],
            }
        )
        playoff_schedule = pd.DataFrame(
            [
                {
                    "home": "Team3",
                    "away": "Team4",
                    "home_goals": None,
                    "away_goals": None,
                    "played": "N",
                },
                {
                    "home": "Team5",
                    "away": "Team6",
                    "home_goals": None,
                    "away_goals": None,
                    "played": "N",
                },
                {
                    "home": "Team7",
                    "away": "Team8",
                    "home_goals": None,
                    "away_goals": None,
                    "played": "N",
                },
                {
                    "home": "Team9",
                    "away": "Team10",
                    "home_goals": None,
                    "away_goals": None,
                    "played": "N",
                },
                {
                    "home": "Team11",
                    "away": "Team12",
                    "home_goals": None,
                    "away_goals": None,
                    "played": "N",
                },
                {
                    "home": "Team13",
                    "away": "Team14",
                    "home_goals": None,
                    "away_goals": None,
                    "played": "N",
                },
                {
                    "home": "Team15",
                    "away": "Team16",
                    "home_goals": None,
                    "away_goals": None,
                    "played": "N",
                },
            ]
        )

        result = simulate_playoff_bracket(
            bracket_df, bracket_format, elos, playoff_schedule
        )
        # Check result dataframe shape and columns
        assert isinstance(result, pd.DataFrame)
        assert set(result.columns) >= {
            "team",
            "po_r16",
            "po_r8",
            "po_r4",
            "po_r2",
            "po_champion",
        }
        assert len(result) == 16

        assert result["po_r16"].sum() == 16, "Total participants in po_r16 should be 16"
        assert result["po_r8"].sum() == 8
        assert result["po_r4"].sum() == 4
        assert result["po_r2"].sum() == 2
        assert result["po_champion"].sum() == 1
        assert result["po_champion"].sum() == 1
        assert (
            result.loc[result["team"] == "Team1", "po_r8"].values[0] == 1
        ), "Team1 did not advance to po_r8"


class TestPlayoffSimulation:

    def setup_method(self):
        self.elos_dict = {"A": 1500, "B": 1400}
        self.playoff_schedule = pd.DataFrame(
            [
                {
                    "home": "A",
                    "away": "B",
                    "home_goals": 2,
                    "away_goals": 1,
                    "played": "Y",
                    "notes": "",
                },
                {
                    "home": "B",
                    "away": "A",
                    "home_goals": 0,
                    "away_goals": 1,
                    "played": "Y",
                    "notes": "A won",
                },
                {
                    "home": "C",
                    "away": "D",
                    "home_goals": 2,
                    "away_goals": 0,
                    "played": "Y",
                    "notes": "",
                },
                {
                    "home": "D",
                    "away": "C",
                    "home_goals": 1,
                    "away_goals": 1,
                    "played": "Y",
                    "notes": "C won",
                },
                {
                    "home": "E",
                    "away": "F",
                    "home_goals": 0,
                    "away_goals": 0,
                    "played": "Y",
                    "notes": "",
                },
                {
                    "home": "F",
                    "away": "E",
                    "home_goals": 3,
                    "away_goals": 1,
                    "played": "Y",
                    "notes": "",
                },
            ]
        )
        self.playoff_schedule_partial = pd.DataFrame(
            [
                {
                    "home": "A",
                    "away": "B",
                    "home_goals": 2,
                    "away_goals": 1,
                    "played": "Y",
                    "notes": "",
                },
                {
                    "home": "B",
                    "away": "A",
                    "home_goals": 0,
                    "away_goals": 0,
                    "played": "N",
                    "notes": "",
                },
                {
                    "home": "C",
                    "away": "D",
                    "home_goals": 2,
                    "away_goals": 0,
                    "played": "Y",
                    "notes": "",
                },
                {
                    "home": "D",
                    "away": "C",
                    "home_goals": 0,
                    "away_goals": 1,
                    "played": "N",
                    "notes": "",
                },
            ]
        )
        self.teams_progression = {}

    def test_simulate_round(self):
        df = pd.DataFrame([{"team1": "A", "team2": "B"}])
        winners = _simulate_round(
            df,
            "two-legged",
            self.elos_dict,
            self.playoff_schedule,
            self.teams_progression,
            "po_r4",
        )
        assert winners == ["A"]

    def test_get_tie_matches(self):
        matches = _get_tie_matches("A", "B", self.playoff_schedule)
        mock_df = pd.DataFrame(
            [
                {
                    "home": "A",
                    "away": "B",
                    "home_goals": 2,
                    "away_goals": 1,
                    "played": "Y",
                    "notes": "",
                },
                {
                    "home": "B",
                    "away": "A",
                    "home_goals": 0,
                    "away_goals": 1,
                    "played": "Y",
                    "notes": "A won",
                },
            ]
        )
        pd.testing.assert_frame_equal(matches, mock_df)

    def test_determine_winner_from_schedule(self):
        tie_matches = _get_tie_matches("A", "B", self.playoff_schedule)
        winner = _determine_winner_from_schedule("A", "B", tie_matches, 0.7)
        assert winner == "A"
        tie_matches = _get_tie_matches("A", "B", self.playoff_schedule_partial)
        winner = _determine_winner_from_schedule("A", "B", tie_matches, 0.7)
        assert winner in ["A", "B"]

    def test_get_winner_from_completed_matches(self):
        winner = _get_winner_from_completed_matches("A", "B", self.playoff_schedule)
        assert winner == "A"

    def test_get_winner_from_notes_success(self):
        df = pd.DataFrame([
            {"notes": "First leg played"},
            {"notes": "; Team A won on aggregate"}
        ])
        winner = _get_winner_from_completed_matches("Team A", "Team B", df)
        assert winner == "Team A"

    def test_get_winner_from_notes_other_team(self):
        df = pd.DataFrame([
            {"notes": "First leg played"},
            {"notes": "; Team B won after extra time"}
        ])
        winner = _get_winner_from_completed_matches("Team A", "Team B", df)
        assert winner == "Team B"

    def test_get_winner_from_notes_invalid_team(self):
        df = pd.DataFrame([
            {"notes": "First leg played"},
            {"notes": "; Team C won on penalties"}
        ])
        with pytest.raises(Warning, match="Winner Team C not in teams Team A, Team B"):
            _get_winner_from_completed_matches("Team A", "Team B", df)

    def test_get_winner_from_partial_matches_case_1(self):
        winner1 = _get_winner_from_partial_matches(
            "A", "B", self.playoff_schedule_partial, 0.7
        )
        winner2 = _get_winner_from_partial_matches(
            "C", "D", self.playoff_schedule_partial, 0.7
        )
        assert winner1 in ["A","B"]
        assert winner2 in ["C","D"]

    def test_get_winner_by_goals(self):
        winner = _get_winner_by_goals("A", "B", self.playoff_schedule)
        assert winner == "A"
        winner = _get_winner_by_goals("E", "F", self.playoff_schedule)
        assert winner == "F"

    def test_calculate_total_goals(self):
        t1_goals, t2_goals = _calculate_total_goals("A", "B", self.playoff_schedule)
        assert t1_goals == 3
        assert t2_goals == 1

    def test_track_team_progression(self):
        _track_team_progression("A", "B", "A", self.teams_progression, "po_r2", 4)
        assert self.teams_progression["A"]["po_r2"] == 1

    def test_prepare_next_round(self):
        winners = ["A", "C"]
        next_round = _prepare_next_round(winners)
        mock_next_round = pd.DataFrame(
            [
                {"team1": "A", "team2": "C"},
            ]
        )
        assert len(next_round) == 1
        pd.testing.assert_frame_equal(next_round, mock_next_round)

    def test_build_results_dataframe(self):
        progress = {
            "A": {"po_r4": 1.0, "po_r2": 1.0, "po_champion": 1.0},
            "B": {"po_r4": 1.0, "po_r2": 0.0, "po_champion": 0.0},
            "C": {"po_r4": 1.0, "po_r2": 1.0, "po_champion": 0.0},
            "D": {"po_r4": 1.0, "po_r2": 0.0, "po_champion": 0.0},
        }
        rounds = ["po_r4", "po_r2"]
        df = _build_results_dataframe(progress, rounds)
        mock_df = pd.DataFrame(
            [
                {"team": "A", "po_r4": 1.0, "po_r2": 1.0, "po_champion": 1.0},
                {"team": "B", "po_r4": 1.0, "po_r2": 0.0, "po_champion": 0.0},
                {"team": "C", "po_r4": 1.0, "po_r2": 1.0, "po_champion": 0.0},
                {"team": "D", "po_r4": 1.0, "po_r2": 0.0, "po_champion": 0.0},
            ]
        )
        pd.testing.assert_frame_equal(df, mock_df, check_dtype=False)

class TestDrawFromPots:

    def setup_method(self):
        # Use a fixed seed for reproducibility where needed
        random.seed(42)
        self.df = pd.DataFrame({
            "team": ["Team A", "Team B", "Team C", "Team D", "Team E", "Team F"],
            "pos": [1, 2, 3, 4, 5, 6]
        })

    def test_returns_valid_draw(self):
        result = draw_from_pots(self.df, pot_size=2)
        assert isinstance(result, pd.DataFrame)
        assert list(result.columns) == ["draw_order", "team"]
        assert len(result) == len(self.df)      
        assert set(result.iloc[0:2]["team"]) == {"Team A", "Team B"}
        assert set(result.iloc[2:4]["team"]) == {"Team C", "Team D"}
        assert set(result.iloc[4:6]["team"]) == {"Team E", "Team F"}

    def test_all_teams_present(self):
        result = draw_from_pots(self.df, pot_size=2)
        original_teams = set(self.df["team"])
        drawn_teams = set(result["team"])
        assert drawn_teams == original_teams

    def test_draw_is_random_within_pots(self):
        # Draw twice and check that at least one pot is ordered differently
        result1 = draw_from_pots(self.df, pot_size=2)
        result2 = draw_from_pots(self.df, pot_size=2)

        def get_pot_teams(draw_df, pot_index, pot_size):
            return draw_df["team"].tolist()[pot_index * pot_size : (pot_index + 1) * pot_size]

        pot_differences = [
            get_pot_teams(result1, i, 2) != get_pot_teams(result2, i, 2)
            for i in range(len(self.df) // 2)
        ]
        assert any(pot_differences)

    def test_handles_uneven_pots(self):
        df = pd.DataFrame({
            "team": ["A", "B", "C", "D", "E"],
            "pos": [1, 2, 3, 4, 5]
        })
        result = draw_from_pots(df, pot_size=2)
        assert len(result) == 5
        assert set(result["team"]) == set(df["team"])
        # Final pot should have only 1 team, rest have 2
        assert result["draw_order"].max() == 5

    def test_single_pot(self):
        df = pd.DataFrame({
            "team": ["X", "Y"],
            "pos": [1, 2]
        })
        result = draw_from_pots(df, pot_size=2)
        assert set(result["team"]) == {"X", "Y"}
        assert len(result) == 2

class TestCreateBracketFromComposition:

    def setup_method(self):
        self.draw_df = pd.DataFrame({
            "draw_order": [1, 2, 3, 4],
            "team": ["Team A", "Team B", "Team C", "Team D"]
        })

    def test_basic_bracket_creation(self):
        composition = [(1, 4), (2, 3)]
        result = create_bracket_from_composition(self.draw_df, composition)
        expected = pd.DataFrame({
            "team1": ["Team A", "Team B"],
            "team2": ["Team D", "Team C"]
        })
        pd.testing.assert_frame_equal(result, expected)

    def test_bye_slot_in_composition(self):
        composition = [(1, "Bye"), ("Bye", 2)]
        result = create_bracket_from_composition(self.draw_df, composition)
        expected = pd.DataFrame({
            "team1": ["Team A", "Bye"],
            "team2": ["Bye", "Team B"]
        })
        pd.testing.assert_frame_equal(result, expected)

    def test_raises_error_on_double_bye(self):
        composition = [("Bye", "Bye")]
        with pytest.raises(ValueError, match="both sides cannot be 'Bye'"):
            create_bracket_from_composition(self.draw_df, composition)

    def test_partial_bracket_with_unused_draws(self):
        composition = [(3, 1)]
        result = create_bracket_from_composition(self.draw_df, composition)
        expected = pd.DataFrame({
            "team1": ["Team C"],
            "team2": ["Team A"]
        })
        pd.testing.assert_frame_equal(result, expected)

if __name__ == "__main__":
    pytest.main([__file__])
