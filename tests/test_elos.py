import pytest
import pandas as pd
import requests
from unittest.mock import Mock, patch, MagicMock
from io import StringIO
import sys
import os

# Add the parent directory to the path to import the module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the functions to test
from retriever.elos import (
    get_elos,
    filter_elos,
) 


class TestGetElos:
    """Test cases for the get_elos function."""

    @patch("requests.get")
    def test_get_elos_success(self, mock_get):
        """Test successful data fetch and CSV parsing."""
        # Mock response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.text = (
            "club,country,level,elo\nBarcelona,ESP,1,2150\nReal Madrid,ESP,1,2100"
        )
        mock_get.return_value = mock_response

        # Call function
        result = get_elos("http://example.com/elos.csv")

        # Assertions
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2
        assert list(result.columns) == ["club", "country", "level", "elo"]
        assert result.iloc[0]["club"] == "Barcelona"
        assert result.iloc[0]["elo"] == 2150
        mock_get.assert_called_once_with("http://example.com/elos.csv")

    @patch("requests.get")
    def test_get_elos_http_error(self, mock_get):
        """Test handling of HTTP error responses."""
        # Mock 404 response
        mock_response = Mock()
        mock_response.status_code = 404
        mock_response.text = "Not Found"
        mock_get.return_value = mock_response

        # Call function and capture output
        with patch("builtins.print") as mock_print:
            result = get_elos("http://example.com/nonexistent.csv")

        # Assertions
        mock_print.assert_any_call("Failed to fetch data. Status code: 404")
        mock_print.assert_any_call("Response:", "Not Found")
        # Note: The function has a bug - it returns df even when request fails
        # In a real scenario, you'd want to fix this bug

    @patch("requests.get")
    def test_get_elos_empty_csv(self, mock_get):
        """Test handling of empty CSV response."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.text = ""
        mock_get.return_value = mock_response

        # This should raise an exception due to empty CSV
        with pytest.raises(pd.errors.EmptyDataError):
            get_elos("http://example.com/empty.csv")

    @patch("requests.get")
    def test_get_elos_malformed_csv(self, mock_get):
        """Test handling of malformed CSV data."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.text = "invalid,csv,data\nwith,inconsistent\ncolumns"
        mock_get.return_value = mock_response

        # Should still create a DataFrame but with NaN values for missing columns
        result = get_elos("http://example.com/malformed.csv")
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2  # Two data rows


class TestFilterElos:
    """Test cases for the filter_elos function."""

    def create_sample_dataframe(self):
        """Helper method to create sample test data."""
        return pd.DataFrame(
            {
                "Club": [
                    "Barcelona",
                    "Real Madrid",
                    "Manchester United",
                    "Liverpool",
                    "Red Star",
                    "Red Star",
                ],
                "Country": ["ESP", "ESP", "ENG", "ENG", "FRA", "SRB"],
                "Level": [1, 1, 1, 1, 2, 1],
                "Elo": [2150, 2100, 2050, 2000, 1800, 1850],
            }
        )

    def test_filter_elos_no_filters(self):
        """Test filtering with no country or level filters."""
        df = self.create_sample_dataframe()
        result = filter_elos(df, None, None)

        # Should return all data with lowercase columns and specific columns
        assert len(result) == 6
        assert list(result.columns) == ["club", "country", "level", "elo"]
        assert all(col.islower() for col in result.columns)

        # Check Red Star correction
        red_star_fra = result[
            (result["club"] == "Red Star FC") & (result["country"] == "FRA")
        ]
        assert len(red_star_fra) == 1

    def test_filter_elos_country_filter(self):
        """Test filtering by country only."""
        df = self.create_sample_dataframe()
        result = filter_elos(df, "ESP", None)

        # Should return only Spanish clubs
        assert len(result) == 2
        assert all(result["country"] == "ESP")
        assert "Barcelona" in result["club"].values
        assert "Real Madrid" in result["club"].values

    def test_filter_elos_level_filter(self):
        """Test filtering by level only."""
        df = self.create_sample_dataframe()
        result = filter_elos(df, None, 1)

        # Should return only level 1 clubs
        assert len(result) == 5  # All except the FRA Red Star which is level 2
        assert all(result["level"] == 1)

    def test_filter_elos_both_filters(self):
        """Test filtering by both country and level."""
        df = self.create_sample_dataframe()
        result = filter_elos(df, "ENG", 1)

        # Should return only English level 1 clubs
        assert len(result) == 2
        assert all(result["country"] == "ENG")
        assert all(result["level"] == 1)
        assert "Manchester United" in result["club"].values
        assert "Liverpool" in result["club"].values

    def test_filter_elos_no_matches(self):
        """Test filtering with criteria that match no rows."""
        df = self.create_sample_dataframe()
        result = filter_elos(df, "GER", 1)

        # Should return empty DataFrame with correct structure
        assert len(result) == 0
        assert list(result.columns) == ["club", "country", "level", "elo"]

    def test_filter_elos_red_star_correction(self):
        """Test the Red Star FC name correction logic."""
        df = self.create_sample_dataframe()
        result = filter_elos(df, None, None)

        # Check that French Red Star is renamed
        red_star_fra = result[
            (result["country"] == "FRA") & (result["club"] == "Red Star FC")
        ]
        assert len(red_star_fra) == 1

        # Check that Serbian Red Star is not renamed
        red_star_srb = result[
            (result["country"] == "SRB") & (result["club"] == "Red Star")
        ]
        assert len(red_star_srb) == 1

    def test_filter_elos_column_case_handling(self):
        """Test that column names are properly converted to lowercase."""
        # Create DataFrame with mixed case columns
        df = pd.DataFrame(
            {"CLUB": ["Test Club"], "COUNTRY": ["TEST"], "LEVEL": [1], "ELO": [2000]}
        )

        result = filter_elos(df, None, None)
        assert all(col.islower() for col in result.columns)

    def test_filter_elos_preserves_data_types(self):
        """Test that data types are preserved during filtering."""
        df = self.create_sample_dataframe()
        result = filter_elos(df, "ESP", 1)

        # Check data types
        assert result["level"].dtype in [int, "int64"]
        assert result["elo"].dtype in [int, float, "int64", "float64"]
        assert result["club"].dtype == object
        assert result["country"].dtype == object

    def test_filter_elos_empty_dataframe(self):
        """Test filtering an empty DataFrame."""
        df = pd.DataFrame(columns=["Club", "Country", "Level", "Elo"])
        result = filter_elos(df, "ESP", 1)

        assert len(result) == 0
        assert list(result.columns) == ["club", "country", "level", "elo"]


class TestIntegration:
    """Integration tests combining both functions."""

    @patch("requests.get")
    def test_get_and_filter_elos_integration(self, mock_get):
        """Test the integration of get_elos and filter_elos functions."""
        # Mock response with comprehensive data
        csv_data = """Club,Country,Level,Elo
        Barcelona,ESP,1,2150
        Real Madrid,ESP,1,2100
        Manchester United,ENG,1,2050
        Liverpool,ENG,1,2000
        Red Star,FRA,2,1800
        Bayern Munich,GER,1,2080"""

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.text = csv_data
        mock_get.return_value = mock_response

        # Get data and filter
        elos = get_elos("http://example.com/elos.csv")
        filtered = filter_elos(elos, "ESP", 1)

        # Assertions
        assert len(filtered) == 2
        assert all(filtered["country"] == "ESP")
        assert all(filtered["level"] == 1)
        assert "Barcelona" in filtered["club"].values
        assert "Real Madrid" in filtered["club"].values


# Fixtures for common test data
@pytest.fixture
def sample_csv_response():
    """Fixture providing sample CSV response data."""
    return """club,country,level,elo
Barcelona,ESP,1,2150
Real Madrid,ESP,1,2100
Manchester United,ENG,1,2050
Liverpool,ENG,1,2000
Red Star,FRA,2,1800
Bayern Munich,GER,1,2080"""


@pytest.fixture
def sample_dataframe():
    """Fixture providing sample DataFrame for testing."""
    return pd.DataFrame(
        {
            "Club": ["Barcelona", "Real Madrid", "Manchester United"],
            "Country": ["ESP", "ESP", "ENG"],
            "Level": [1, 1, 1],
            "Elo": [2150, 2100, 2050],
        }
    )


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v"])
