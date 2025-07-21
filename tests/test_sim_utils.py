import pytest
from unittest.mock import patch
from io import StringIO
import numpy as np
from simulator.sim_utils import (
    calculate_win_probability,
    simulate_match,
    simulate_playoff,
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


if __name__ == "__main__":
    pytest.main([__file__])
