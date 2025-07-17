import pytest
from unittest.mock import patch
from io import StringIO
from simulator.sim_utils import calculate_win_probability


class TestCalculateWinProbability:
    
    def test_equal_elo_single_game(self):
        """Test that equal Elo ratings with home advantage gives home team higher probability"""
        result = calculate_win_probability(1500, 1500, "single_game")
        # With home advantage of 80, home team should have > 0.5 probability
        assert result > 0.5
        assert pytest.approx(result, rel=1e-3) == 0.5537
    
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
        assert result > 0.7
        assert pytest.approx(result, rel=1e-3) == 0.7597
    
    def test_away_team_higher_elo_single_game(self):
        """Test away team with higher Elo rating"""
        result = calculate_win_probability(1500, 1600, "single_game")
        # Away team has 100 Elo advantage but loses 80 home advantage = 20 net
        assert result < 0.5
        assert pytest.approx(result, rel=1e-3) == 0.4721
    
    def test_neutral_venue_higher_elo(self):
        """Test neutral venue with one team having higher Elo"""
        result = calculate_win_probability(1500, 1600, "single_game_neutral")
        # Away team has 100 Elo advantage, no home advantage
        assert result < 0.5
        assert pytest.approx(result, rel=1e-3) == 0.3599
    
    def test_two_legged_higher_elo(self):
        """Test two-legged matchup with multiplier effect"""
        result = calculate_win_probability(1500, 1600, "two-legged")
        # 100 Elo difference * 1.4 multiplier = 140 effective difference
        assert result < 0.5
        assert pytest.approx(result, rel=1e-3) == 0.2890
    
    def test_custom_home_advantage(self):
        """Test custom home advantage value"""
        result = calculate_win_probability(1500, 1500, "single_game", home_adv=100)
        # Higher home advantage should give home team even better probability
        assert result > 0.5537  # Greater than default 80 home advantage
        assert pytest.approx(result, rel=1e-3) == 0.5718
    
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
        with patch('sys.stdout', new_callable=StringIO) as mock_stdout:
            result = calculate_win_probability(1500, 1500, "invalid_type")
            
            # Should print warning message
            output = mock_stdout.getvalue()
            assert "Unknown matchup type: invalid_type" in output
            assert "Defaulting to single game" in output
            
            # Should default to single game behavior
            assert result > 0.5  # Home advantage should apply
    
    def test_return_type_and_range(self):
        """Test that function returns float between 0 and 1"""
        result = calculate_win_probability(1500, 1600, "single_game")
        assert isinstance(result, float)
        assert 0 <= result <= 1
    
    def test_probability_symmetry(self):
        """Test that probabilities are symmetric (P(A beats B) + P(B beats A) ≈ 1)"""
        # For neutral venue (no home advantage complications)
        prob_a_beats_b = calculate_win_probability(1500, 1600, "single_game_neutral")
        prob_b_beats_a = calculate_win_probability(1600, 1500, "single_game_neutral")
        
        # Should sum to approximately 1
        assert pytest.approx(prob_a_beats_b + prob_b_beats_a, rel=1e-6) == 1.0
    
    def test_edge_cases(self):
        """Test edge cases with unusual inputs"""
        # Negative Elo ratings
        result1 = calculate_win_probability(-100, 100, "single_game")
        assert 0 <= result1 <= 1
        
        # Very large Elo ratings
        result2 = calculate_win_probability(5000, 5100, "single_game")
        assert 0 <= result2 <= 1
        
        # Negative home advantage
        result3 = calculate_win_probability(1500, 1500, "single_game", home_adv=-50)
        assert result3 < 0.5  # Home team should be disadvantaged


if __name__ == "__main__":
    pytest.main([__file__])