import pytest
import pandas as pd
import numpy as np
import requests
from unittest.mock import Mock, patch, MagicMock
from bs4 import BeautifulSoup
import sys
import os

# Add the parent directory to the path to import the module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the functions to test
from retriever.fixtures import (
    extract_scores,
    get_fixtures,
    get_fixtures_selenium,
    parse_fixtures_html,
    process_fixtures,
)


class TestExtractScores:
    """Test cases for the extract_scores function."""

    def test_extract_scores_regular_match(self):
        """Test extraction of regular match scores."""
        result = extract_scores("2–1")
        expected = pd.Series([2, 1, None, None])
        pd.testing.assert_series_equal(result, expected)

    def test_extract_scores_with_penalties(self):
        """Test extraction of scores with penalty shootout."""
        result = extract_scores("1–1 (4) (2)")
        expected = pd.Series([1, 1, 4, 2])
        pd.testing.assert_series_equal(result, expected)

    def test_extract_scores_high_scoring_match(self):
        """Test extraction of high-scoring matches."""
        result = extract_scores("5–3")
        expected = pd.Series([5, 3, None, None])
        pd.testing.assert_series_equal(result, expected)

    def test_extract_scores_zero_goals(self):
        """Test extraction when one or both teams score zero."""
        result = extract_scores("0–0")
        expected = pd.Series([0, 0, None, None])
        pd.testing.assert_series_equal(result, expected)

        result2 = extract_scores("3–0")
        expected2 = pd.Series([3, 0, None, None])
        pd.testing.assert_series_equal(result2, expected2)

    def test_extract_scores_penalties_after_zero_zero(self):
        """Test penalty extraction after 0-0 draw."""
        result = extract_scores("0–0 (5) (3)")
        expected = pd.Series([0, 0, 5, 3])
        pd.testing.assert_series_equal(result, expected)

    def test_extract_scores_nan_input(self):
        """Test handling of NaN input."""
        result = extract_scores(np.nan)
        expected = pd.Series([None, None, None, None])
        pd.testing.assert_series_equal(result, expected)

    def test_extract_scores_none_input(self):
        """Test handling of None input."""
        result = extract_scores(None)
        expected = pd.Series([None, None, None, None])
        pd.testing.assert_series_equal(result, expected)

    def test_extract_scores_invalid_format(self):
        """Test handling of invalid score formats."""
        # Should return None values for invalid formats
        result = extract_scores("invalid score")
        expected = pd.Series([None, None, None, None])
        pd.testing.assert_series_equal(result, expected)

        result2 = extract_scores("2-1")  # Wrong dash format
        expected2 = pd.Series([None, None, None, None])
        pd.testing.assert_series_equal(result2, expected2)

    def test_extract_scores_partial_penalties(self):
        """Test handling when only one penalty score is present."""
        result = extract_scores("1–1 (4)")
        expected = pd.Series([1, 1, None, None])
        pd.testing.assert_series_equal(result, expected)


class TestGetFixtures:
    """Test cases for the get_fixtures function."""

    def create_mock_html(self, tables_data):
        """Helper method to create mock HTML with fixture tables."""
        html = "<html><body>"
        for table_id, table_data in tables_data.items():
            html += f'<table id="{table_id}"><thead><tr>'
            for header in table_data["headers"]:
                html += f"<th>{header}</th>"
            html += "</tr></thead><tbody>"
            for row in table_data["rows"]:
                html += "<tr>"
                for i, cell in enumerate(row):
                    tag = "th" if i == 0 else "td"
                    html += f"<{tag}>{cell}</{tag}>"
                html += "</tr>"
            html += "</tbody></table>"
        html += "</body></html>"
        return html

    @patch("time.sleep")  # Mock sleep to speed up tests
    @patch("requests.get")
    def test_get_fixtures_success_single_table(self, mock_get, mock_sleep):
        """Test successful fixture fetching with single table."""
        # Mock HTML response
        tables_data = {
            "table1": {
                "headers": ["Date", "Home", "Score", "Away", "xG"],
                "rows": [
                    ["2024-01-15", "Barcelona", "2–1", "Real Madrid", "2.1"],
                    ["2024-01-16", "Liverpool", "3–0", "Arsenal", "2.8"],
                ],
            }
        }

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.text = self.create_mock_html(tables_data)
        mock_get.return_value = mock_response

        result = get_fixtures("http://example.com", ["table1"])

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2
        assert "xG" not in result.columns  # Should be dropped
        assert "Barcelona" in result["Home"].values
        assert "Real Madrid" in result["Away"].values
        mock_sleep.assert_called_once_with(10)

    @patch("time.sleep")
    @patch("requests.get")
    def test_get_fixtures_success_multiple_tables(self, mock_get, mock_sleep):
        """Test successful fixture fetching with multiple tables."""
        tables_data = {
            "table1": {
                "headers": ["Date", "Home", "Score", "Away", "xG"],
                "rows": [["2024-01-15", "Team1", "2–1", "Team2", "2.1"]],
            },
            "table2": {
                "headers": ["Date", "Home", "Score", "Away", "xG"],
                "rows": [["2024-01-16", "Team3", "1–0", "Team4", "1.5"]],
            },
        }

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.text = self.create_mock_html(tables_data)
        mock_get.return_value = mock_response

        result = get_fixtures("http://example.com", ["table1", "table2"])

        assert len(result) == 2
        assert "Team1" in result["Home"].values
        assert "Team3" in result["Home"].values

    @patch("time.sleep")
    @patch("requests.get")
    def test_get_fixtures_http_error(self, mock_get, mock_sleep):
        """Test handling of HTTP errors."""
        mock_response = Mock()
        mock_response.status_code = 404
        mock_get.return_value = mock_response

        with patch("builtins.print") as mock_print:
            result = get_fixtures("http://example.com", ["table1"])

        mock_print.assert_called_with("Failed to fetch the page. Status code: 404")

    @patch("time.sleep")
    @patch("requests.get")
    def test_get_fixtures_empty_rows_filtered(self, mock_get, mock_sleep):
        """Test that empty rows are properly filtered out."""
        tables_data = {
            "table1": {
                "headers": ["Date", "Home", "Score", "Away", "xG"],
                "rows": [
                    ["2024-01-15", "Barcelona", "2–1", "Real Madrid", "2.1"],
                    ["2024-01-16", "", "", "", ""],  # Empty row
                    ["2024-01-17", "Liverpool", "1–1", "Arsenal", "1.8"],
                ],
            }
        }

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.text = self.create_mock_html(tables_data)
        mock_get.return_value = mock_response

        result = get_fixtures("http://example.com", ["table1"])

        # Should exclude the empty row
        assert len(result) == 2
        assert "" not in result["Home"].values

    @patch("time.sleep")
    @patch("requests.get")
    def test_get_fixtures_request_exception(self, mock_get, mock_sleep):
        """Test handling of request exceptions."""
        mock_get.side_effect = requests.exceptions.RequestException("Network error")

        with pytest.raises(requests.exceptions.RequestException):
            get_fixtures("http://example.com", ["table1"])


class TestProcessFixtures:
    """Test cases for the process_fixtures function."""

    def create_sample_fixtures(self):
        """Helper method to create sample fixtures data."""
        return pd.DataFrame(
            {
                "Home": ["Barcelona ESP", "Liverpool ENG", "Real Madrid", ""],
                "Away": ["ESP Real Madrid", "ENG Arsenal", "Barcelona", ""],
                "Score": ["2–1", "", "1–1 (4) (2)", "3–0"],
                "Date": ["2024-01-15", "2024-01-16", "2024-01-17", "2024-01-18"],
                "Notes": ["", "", "Extra time", ""],
            }
        )

    def test_process_fixtures_domestic_league(self):
        """Test processing fixtures for domestic leagues."""
        fixtures = self.create_sample_fixtures()
        result = process_fixtures(fixtures, "ESP")

        # Check basic structure
        expected_columns = [
            "home",
            "away",
            "home_goals",
            "away_goals",
            "played",
            "neutral",
            "round",
            "date",
            "notes",
        ]
        assert list(result.columns) == expected_columns

        # Check that empty rows are filtered out
        assert len(result) == 3

        # Check score extraction
        assert result.iloc[0]["home_goals"] == 2
        assert result.iloc[0]["away_goals"] == 1
        assert result.iloc[0]["played"] == "Y"

        # Check unplayed match
        assert pd.isna(result.iloc[1]["home_goals"])
        assert result.iloc[1]["played"] == "N"

        # Check penalty match
        assert result.iloc[2]["home_goals"] == 1
        assert result.iloc[2]["away_goals"] == 1

        # Check neutral venue (should be 'N' for domestic)
        assert all(result["neutral"] == "N")

        # Check round column (should be 'League' as default)
        assert all(result["round"] == "League")

    def test_process_fixtures_european_competition(self):
        """Test processing fixtures for European competitions."""
        fixtures = pd.DataFrame(
            {
                "Home": ["Barcelona ESP", "Liverpool ENG"],
                "Away": ["ESP Real Madrid", "ENG Arsenal"],
                "Score": ["2–1", "0–0"],
                "Date": ["2024-01-15", "2024-01-16"],
                "Notes": ["", ""],
            }
        )

        result = process_fixtures(fixtures, "UCL")

        # Check that country codes are removed from team names
        assert result.iloc[0]["home"] == "Barcelona"
        assert result.iloc[0]["away"] == "Real Madrid"
        assert result.iloc[1]["home"] == "Liverpool"
        assert result.iloc[1]["away"] == "Arsenal"

    def test_process_fixtures_with_round_column(self):
        """Test processing when round column exists."""
        fixtures = pd.DataFrame(
            {
                "Home": ["Barcelona", "Real Madrid"],
                "Away": ["Real Madrid", "Barcelona"],
                "Score": ["2–1", "1–2"],
                "Round": ["Quarter-final", "Semi-final"],
                "Date": ["2024-01-15", "2024-01-16"],
                "Notes": ["", ""],
            }
        )

        result = process_fixtures(fixtures, "ESP")

        assert result.iloc[0]["round"] == "Quarter-final"
        assert result.iloc[1]["round"] == "Semi-final"

    def test_process_fixtures_round_with_nan(self):
        """Test processing when round column has NaN values."""
        fixtures = pd.DataFrame(
            {
                "Home": ["Barcelona", "Real Madrid"],
                "Away": ["Real Madrid", "Barcelona"],
                "Score": ["2–1", "1–2"],
                "Round": ["Quarter-final", np.nan],
                "Date": ["2024-01-15", "2024-01-16"],
                "Notes": ["", ""],
            }
        )

        result = process_fixtures(fixtures, "ESP")

        assert result.iloc[0]["round"] == "Quarter-final"
        assert result.iloc[1]["round"] == "League"  # NaN should be filled with 'League'

    def test_process_fixtures_column_case_normalization(self):
        """Test that column names are properly normalized to lowercase."""
        fixtures = pd.DataFrame(
            {
                "HOME": ["Barcelona"],
                "AWAY": ["Real Madrid"],
                "SCORE": ["2–1"],
                "DATE": ["2024-01-15"],
                "NOTES": [""],
            }
        )

        result = process_fixtures(fixtures, "ESP")

        # All operations should work despite uppercase column names
        assert len(result) == 1
        assert result.iloc[0]["home"] == "Barcelona"

    def test_process_fixtures_score_variations(self):
        """Test various score formats and edge cases."""
        fixtures = pd.DataFrame(
            {
                "Home": ["Team1", "Team2", "Team3", "Team4"],
                "Away": ["TeamA", "TeamB", "TeamC", "TeamD"],
                "Score": ["0–0", "5–3", "", "1–1 (3) (1)"],
                "Date": ["2024-01-15", "2024-01-16", "2024-01-17", "2024-01-18"],
                "Notes": ["", "", "", ""],
            }
        )

        result = process_fixtures(fixtures, "ESP")

        # Check various score types
        assert result.iloc[0]["home_goals"] == 0
        assert result.iloc[0]["away_goals"] == 0
        assert result.iloc[0]["played"] == "Y"

        assert result.iloc[1]["home_goals"] == 5
        assert result.iloc[1]["away_goals"] == 3

        assert pd.isna(result.iloc[2]["home_goals"])
        assert result.iloc[2]["played"] == "N"

        assert result.iloc[3]["home_goals"] == 1
        assert result.iloc[3]["away_goals"] == 1

    def test_process_fixtures_team_name_stripping(self):
        """Test that team names are properly stripped of whitespace."""
        fixtures = pd.DataFrame(
            {
                "Home": ["  Barcelona  ", "Liverpool   "],
                "Away": ["   Real Madrid", "  Arsenal  "],
                "Score": ["2–1", "1–0"],
                "Date": ["2024-01-15", "2024-01-16"],
                "Notes": ["", ""],
            }
        )

        result = process_fixtures(fixtures, "ESP")

        assert result.iloc[0]["home"] == "Barcelona"
        assert result.iloc[0]["away"] == "Real Madrid"
        assert result.iloc[1]["home"] == "Liverpool"
        assert result.iloc[1]["away"] == "Arsenal"

    def test_process_fixtures_european_competitions_variants(self):
        """Test all European competition variants."""
        fixtures = pd.DataFrame(
            {
                "Home": ["Barcelona ESP"],
                "Away": ["ESP Real Madrid"],
                "Score": ["2–1"],
                "Date": ["2024-01-15"],
                "Notes": [""],
            }
        )

        # Test UCL
        result_ucl = process_fixtures(fixtures.copy(), "UCL")
        assert result_ucl.iloc[0]["home"] == "Barcelona"

        # Test UEL
        result_uel = process_fixtures(fixtures.copy(), "UEL")
        assert result_uel.iloc[0]["home"] == "Barcelona"

        # Test UECL
        result_uecl = process_fixtures(fixtures.copy(), "UECL")
        assert result_uecl.iloc[0]["home"] == "Barcelona"


class TestGetFixturesSelenium:
    @pytest.fixture(autouse=True)
    def setup_method(self):
        # Sample HTML with a fake table
        self.sample_html = """
        <html>
        <body>
            <table id="test_table">
                <thead>
                    <tr><th>Date</th><th>Home</th><th>Away</th><th>xG</th></tr>
                </thead>
                <tbody>
                    <tr><th>1</th><td>Team A</td><td>Team B</td><td>1.2</td></tr>
                    <tr><th>2</th><td></td><td></td><td></td></tr>
                </tbody>
            </table>
        </body>
        </html>
        """
        self.table_ids = ["test_table"]

    @patch("src.retriever.fixtures.webdriver.Chrome")
    def test_fixtures_parsing(self, mock_chrome):
        # Mock Selenium driver
        mock_driver = MagicMock()
        mock_driver.page_source = self.sample_html
        mock_chrome.return_value = mock_driver

        df = get_fixtures_selenium("http://fake.url", self.table_ids)

        # Assertions
        assert isinstance(df, pd.DataFrame)
        assert not df.empty
        assert list(df.columns) == ["Date", "Home", "Away"]  # "xG" should be dropped
        assert "Team A" in df["Home"].values

    @patch("src.retriever.fixtures.webdriver.Chrome")
    def test_empty_table(self, mock_chrome):
        # Modify HTML to have no rows
        empty_html = self.sample_html.replace(
            '<tr><th>1</th><td>Team A</td><td>Team B</td><td>1.2</td></tr>', ""
        )
        mock_driver = MagicMock()
        mock_driver.page_source = empty_html
        mock_chrome.return_value = mock_driver

        df = get_fixtures_selenium("http://fake.url", self.table_ids)

        assert df.empty


class TestParseFixturesHtml:
    @pytest.fixture
    def sample_html(self):
        return """
        <html>
        <body>
            <table id="test_table">
                <thead>
                    <tr><th>Date</th><th>Home</th><th>Away</th><th>xG</th></tr>
                </thead>
                <tbody>
                    <tr><th>1</th><td>Team A</td><td>Team B</td><td>1.2</td></tr>
                    <tr><th>2</th><td></td><td></td><td></td></tr>
                </tbody>
            </table>
        </body>
        </html>
        """

    def test_parse_valid_table(self, sample_html):
        df = parse_fixtures_html(sample_html, table_id=["test_table"])
        expected = pd.DataFrame({
            "Date": ["1"],
            "Home": ["Team A"],
            "Away": ["Team B"]
        })
        pd.testing.assert_frame_equal(df.reset_index(drop=True), expected)

    def test_missing_table_id(self, sample_html):
        with pytest.raises(ValueError, match="No valid tables found"):
            parse_fixtures_html(sample_html, table_id=["wrong_id"])

    def test_missing_thead_or_tbody(self):
        html = """
        <html><body>
            <table id="test_table">
                <tbody>
                    <tr><td>Some</td><td>Data</td></tr>
                </tbody>
            </table>
        </body></html>
        """
        with pytest.raises(ValueError, match="No valid tables found"):
            parse_fixtures_html(html, table_id=["test_table"])

    def test_no_data_rows(self):
        html = """
        <html><body>
            <table id="test_table">
                <thead>
                    <tr><th>Date</th><th>Home</th><th>Away</th></tr>
                </thead>
                <tbody></tbody>
            </table>
        </body></html>
        """
        with pytest.raises(ValueError, match="No valid tables found"):
            parse_fixtures_html(html, table_id=["test_table"])

    def test_multiple_tables(self):
        html = """
        <html><body>
            <table id="table1">
                <thead>
                    <tr><th>Date</th><th>Home</th><th>Away</th><th>xG</th></tr>
                </thead>
                <tbody>
                    <tr><th>1</th><td>Team A</td><td>Team B</td><td>1.0</td></tr>
                </tbody>
            </table>
            <table id="table2">
                <thead>
                    <tr><th>Date</th><th>Home</th><th>Away</th><th>xG</th></tr>
                </thead>
                <tbody>
                    <tr><th>2</th><td>Team C</td><td>Team D</td><td>1.1</td></tr>
                </tbody>
            </table>
        </body></html>
        """
        df = parse_fixtures_html(html, table_id=["table1", "table2"])
        expected = pd.DataFrame({
            "Date": ["1", "2"],
            "Home": ["Team A", "Team C"],
            "Away": ["Team B", "Team D"]
        })
        pd.testing.assert_frame_equal(df.reset_index(drop=True), expected)


class TestIntegration:
    """Integration tests combining multiple functions."""

    @patch("time.sleep")
    @patch("requests.get")
    def test_full_pipeline_integration(self, mock_get, mock_sleep):
        """Test the complete pipeline from fetching to processing."""
        # Mock HTML with fixture table
        html = """
        <html>
            <body>
                <table id="fixtures_table">
                    <thead>
                        <tr>
                            <th>Date</th>
                            <th>Home</th>
                            <th>Score</th>
                            <th>Away</th>
                            <th>xG</th>
                            <th>Notes</th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr>
                            <th>2024-01-15</th>
                            <td>Barcelona ESP</td>
                            <td>2–1</td>
                            <td>ESP Real Madrid</td>
                            <td>2.1</td>
                            <td></td>
                        </tr>
                        <tr>
                            <th>2024-01-16</th>
                            <td>Liverpool ENG</td>
                            <td></td>
                            <td>ENG Arsenal</td>
                            <td>0.0</td>
                            <td></td>
                        </tr>
                    </tbody>
                </table>
            </body>
        </html>
        """

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.text = html
        mock_get.return_value = mock_response

        # Test the full pipeline
        fixtures = get_fixtures("http://example.com", ["fixtures_table"])
        result = process_fixtures(fixtures, "UCL")

        # Verify results
        assert len(result) == 2
        assert result.iloc[0]["home"] == "Barcelona"  # Country code removed
        assert result.iloc[0]["away"] == "Real Madrid"  # Country code removed
        assert result.iloc[0]["home_goals"] == 2
        assert result.iloc[0]["away_goals"] == 1
        assert result.iloc[0]["played"] == "Y"

        assert result.iloc[1]["home"] == "Liverpool"
        assert pd.isna(result.iloc[1]["home_goals"])
        assert result.iloc[1]["played"] == "N"


# Fixtures for common test data
@pytest.fixture
def sample_fixtures_df():
    """Fixture providing sample fixtures DataFrame."""
    return pd.DataFrame(
        {
            "Home": ["Barcelona", "Real Madrid", "Liverpool"],
            "Away": ["Real Madrid", "Barcelona", "Arsenal"],
            "Score": ["2–1", "0–0 (4) (3)", ""],
            "Date": ["2024-01-15", "2024-01-16", "2024-01-17"],
            "Notes": ["", "Penalties", ""],
        }
    )


@pytest.fixture
def sample_html():
    """Fixture providing sample HTML for testing."""
    return """
    <html>
        <body>
            <table id="test_table">
                <thead>
                    <tr><th>Date</th><th>Home</th><th>Score</th><th>Away</th><th>xG</th></tr>
                </thead>
                <tbody>
                    <tr><th>2024-01-15</th><td>Barcelona</td><td>2–1</td><td>Real Madrid</td><td>2.1</td></tr>
                </tbody>
            </table>
        </body>
    </html>
    """


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v"])
