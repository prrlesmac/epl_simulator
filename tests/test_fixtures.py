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
    parse_fixtures_html,
    get_fixtures_text,
    parse_game_element,
    process_fixtures,
    process_nfl_table_legacy
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

        result = get_fixtures(["http://example.com"], ["table1"])

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

        result = get_fixtures(["http://example.com"], ["table1", "table2"])

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
            result = get_fixtures(["http://example.com"], ["table1"])

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

        result = get_fixtures(["http://example.com"], ["table1"])

        # Should exclude the empty row
        assert len(result) == 2
        assert "" not in result["Home"].values

    @patch("time.sleep")
    @patch("requests.get")
    def test_get_fixtures_request_exception(self, mock_get, mock_sleep):
        """Test handling of request exceptions."""
        mock_get.side_effect = requests.exceptions.RequestException("Network error")

        with pytest.raises(requests.exceptions.RequestException):
            get_fixtures(["http://example.com"], ["table1"])

    def create_mock_html_text(self):
        """Helper to create mock HTML similar to Baseball Reference schedule page."""
        html = """
        <html><body>
            <h2>MLB Regular Season</h2>
           <h3>Wednesday, March 20, 2024</h3>
            <p class="game">
                <a href="/teams/NYA/2024.shtml">Yankees</a> 
                @ 
                <a href="/teams/BOS/2024.shtml">Red Sox</a> 
                (3) - (5)
            </p>
        </body></html>
        """
        return html

    def create_bad_score_html(self):
        """Helper for game element missing scores."""
        html = """
        <p class="game">
            <a href="/teams/NYA/2024.shtml">Yankees</a> 
            @ 
            <a href="/teams/BOS/2024.shtml">Red Sox</a>
        </p>
        """
        return html

    def test_parse_game_element_valid(self):
        soup = BeautifulSoup(self.create_mock_html_text(), "html.parser")
        game_el = soup.find("p", class_="game")
        date = "Wednesday, March 20, 2024"
        round = "MLB Season"
        result = parse_game_element(game_el, date, round)

        expected = {
            'round': round,
            'date': date,
            'away': 'Yankees',
            'home': 'Red Sox',
            'away_goals': 3,
            'home_goals': 5
        }
        assert result == expected

    def test_parse_game_element_invalid_missing_scores(self):
        soup = BeautifulSoup(self.create_bad_score_html(), "html.parser")
        game_el = soup.find("p", class_="game")
        result = parse_game_element(game_el, "Some Date", "MLB Season")
        assert result is None

    @patch("time.sleep")
    @patch("requests.get")
    def test_get_fixtures_text_success(self, mock_get, mock_sleep):
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.text = self.create_mock_html_text()
        mock_get.return_value = mock_response

        df = get_fixtures_text(["http://dummy-url.com"])

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1
        assert set(df.columns) == {'round','date', 'away', 'home', 'away_goals', 'home_goals'}
        assert df.iloc[0]["away"] == "Yankees"
        assert df.iloc[0]["home_goals"] == 5
        mock_sleep.assert_called_once()

    @patch("time.sleep")
    @patch("requests.get")
    def test_get_fixtures_text_failure(self, mock_get, mock_sleep):
        mock_response = Mock()
        mock_response.status_code = 404
        mock_get.return_value = mock_response

        result = get_fixtures_text(["http://dummy-url.com"])
        assert result is None
        mock_sleep.assert_called_once()

    
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


    def test_process_fixtures_nba(self):
        """Test processing fixtures for NBA."""
        fixtures = pd.DataFrame(
            {
                "Date": ["Mon, Oct 21, 2024", "Tue, Oct 22, 2024", "Wed, Oct 23, 2024", "Thu, Oct 24, 2024", "Fri, Oct 25, 2024"],
                "Start (ET)": ["7:30p", "10:00p","7:30p", "10:00p", "6:00p"],
                "Visitor/Neutral": ["New York Knicks", "Minnesota Timberwolves", "Boston Celtics", "Chicago Bulls", "Detroit Pistons"],
                "PTS": [109, 103, 110, 100, pd.NA],
                "Home/Neutral": ["Boston Celtics", "Los Angeles Lakers", "Utah Jazz", "San Antonio Spurs", "Houston Rockets"],
                "PTS.1": [132, 110, 115, 95, pd.NA],
                "Attend.": ["19,156", "18,997", "20,000", "30,000", "25,000"],
                "LOG": ["2:04", "2:26", "2:04","2:11", "2:17"],
                "Arena": ["TD Garden", "Crypto.com Arena", "TD Arena", "AA Arena", "AA Court"],
                "Notes": ["", "NBA Cup","","Play-In Game",""]
            }
        )

        output = pd.DataFrame(
            {
                "home": ["Boston Celtics", "Los Angeles Lakers", "Utah Jazz", "San Antonio Spurs", "Houston Rockets"],
                "away": ["New York Knicks", "Minnesota Timberwolves", "Boston Celtics", "Chicago Bulls", "Detroit Pistons"],
                "home_goals": pd.Series([132, 110, 115, 95, pd.NA], dtype="Int64"),
                "away_goals": pd.Series([109, 103, 110, 100, pd.NA], dtype="Int64"),
                "played": ["Y", "Y", "Y", "Y","N"],
                "neutral": ["N", "N", "N", "N","N"],
                "round": ["League","NBA Cup","League", "Play-in","Playoff"],
                "date": pd.to_datetime(["2024-10-21", "2024-10-22","2024-10-23", "2024-10-24","2024-10-25"]),
                "notes": ["", "NBA Cup","","Play-In Game",""]
            }
        )

        result = process_fixtures(fixtures, "NBA").reset_index(drop=True)
        pd.testing.assert_frame_equal(output, result, check_like=True, check_index_type=False)  # ignores column order

    def test_process_fixtures_nfl_legacy(self):
        """Test processing fixtures forNFL."""

        fixtures = pd.DataFrame(
            {
                "Week": ["Pre0", "Pre1", "18", "18"],
                "Day": ["Thu", "Thu", "Sun", "Sun"],
                "Date": ["July 31", "August 7", "January 4", "January 4"],
                "VisTm": [
                    "Los Angeles Chargers",
                    "Indianapolis Colts",
                    "Baltimore Ravens",
                    "Kansas City Chiefs"
                ],
                "Pts": [34, None, 23, None],
                "  ": ["","","",""],
                "HomeTm": [
                    "Detroit Lions",
                    "Baltimore Ravens",
                    "Pittsburgh Steelers",
                    "Las Vegas Raiders"
                ],
                "Pts.1": [7, None, 16, None],
                "Time": ["8:00 PM", "7:00 PM", "1:00 PM", "1:00 PM"],
                "url": [
                    "https://www.pro-football-reference.com/years/2025/games.htm",
                    "https://www.pro-football-reference.com/years/2025/games.htm",
                    "https://www.pro-football-reference.com/years/2025/games.htm",
                    "https://www.pro-football-reference.com/years/2025/games.htm"
                ]
            }
        )
        fixtures.columns = fixtures.columns.str.lower()
        output = pd.DataFrame(
            {
                "home": ["Pittsburgh Steelers", "Las Vegas Raiders"],
                "away": ["Baltimore Ravens", "Kansas City Chiefs"],
                "home_goals": pd.Series([16, pd.NA], dtype="Int64"),
                "away_goals": pd.Series([23, pd.NA], dtype="Int64"),
                "played": ["Y", "N"],
                "neutral": ["N", "N"],
                "round": ["League", "League"],
                "date": pd.to_datetime(["2026-01-04", "2026-01-04"]),
                "notes": ["", ""]
            }
        )
        result = process_nfl_table_legacy(fixtures).reset_index(drop=True)
        result = result[
            [
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
        ]
        pd.testing.assert_frame_equal(output, result, check_like=True, check_index_type=False)  # ignores column order

    def test_process_fixtures_nfl(self):
        """Test processing fixtures forNFL."""

        fixtures = pd.DataFrame(
            {
                "Week": ["Pre0", "Pre1", "18", "18"],
                "Day": ["Thu", "Thu", "Sun", "Sun"],
                "Date": ["2025-07-31", "2025-08-07", "2026-01-04", "2026-01-04"],
                "Winner/tie": [
                    "Los Angeles Chargers",
                    "Indianapolis Colts",
                    "Baltimore Ravens",
                    "Kansas City Chiefs"
                ],
                "PtsW": [34, None, 23, None],
                "  ": ["","@","","@"],
                "Loser/tie": [
                    "Detroit Lions",
                    "Baltimore Ravens",
                    "Pittsburgh Steelers",
                    "Las Vegas Raiders"
                ],
                "PtsL": [7, None, 16, None],
                "url": [
                    "https://www.pro-football-reference.com/years/2025/games.htm",
                    "https://www.pro-football-reference.com/years/2025/games.htm",
                    "https://www.pro-football-reference.com/years/2025/games.htm",
                    "https://www.pro-football-reference.com/years/2025/games.htm"
                ]
            }
        )

        output = pd.DataFrame(
            {
                "home": ["Baltimore Ravens", "Las Vegas Raiders"],
                "away": ["Pittsburgh Steelers", "Kansas City Chiefs"],
                "home_goals": pd.Series([23, pd.NA], dtype="Int64"),
                "away_goals": pd.Series([16, pd.NA], dtype="Int64"),
                "played": ["Y", "N"],
                "neutral": ["N", "N"],
                "round": ["League", "League"],
                "date": ["2026-01-04", "2026-01-04"],
                "notes": ["", ""]
            }
        )
        result = process_fixtures(fixtures, "NFL").reset_index(drop=True)
        pd.testing.assert_frame_equal(output, result, check_like=True, check_index_type=False)  # ignores column order

    def test_process_fixtures_mlb(self):
        """Test processing fixtures for MLB."""

        fixtures = pd.DataFrame(
            {
                "date": ["Wednesday, March 20, 2024", "Thursday, March 21, 2024"],
                "away": ["Los Angeles Dodgers", "San Diego Padres"],
                "home": ["San Diego Padres", "Los Angeles Dodgers"],
                "away_goals": [5, 15],
                "home_goals": [2, 11],
                "round": ["League","League"]
            }
        )

        output = pd.DataFrame(
            {
                "away": ["Los Angeles Dodgers", "San Diego Padres"],
                "home": ["San Diego Padres", "Los Angeles Dodgers"],
                "home_goals": pd.Series([2, 11], dtype="Int64"),
                "away_goals": pd.Series([5, 15], dtype="Int64"),
                "played": ["Y", "Y"],
                "neutral": ["N", "N"],
                "round": ["League", "League"],
                "date": pd.to_datetime(["2024-03-20", "2024-03-21"]),
                "notes": ["", ""]
            }
        )

        result = process_fixtures(fixtures, "MLB").reset_index(drop=True)
        pd.testing.assert_frame_equal(output, result, check_like=True, check_index_type=False)  # ignores column order




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
        fixtures = get_fixtures(["http://example.com"], ["fixtures_table"])
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
