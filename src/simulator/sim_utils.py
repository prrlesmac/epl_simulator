import pandas as pd
import numpy as np
import math
import random
import re


def calculate_win_probability(
    elo_home, elo_away, matchup_type="single_game", home_adv=80
):
    """
    Calculate win probability using the Elo rating system.

    Parameters:
        elo_home (float): Elo rating of team 1
        elo_away (float): Elo rating of team 2
        matchup_type (str): Type of matchup, can be "two-legged", "single_game_neutral", or "single_game"
        home_adv (float): Home advantage factor, default is 80 for single game matchups

    Returns:
        float: Probability of team 1 winning
    """
    if matchup_type == "two-legged":
        rank_diff = (elo_away - elo_home) * 1.4
    elif matchup_type == "single_game_neutral":
        rank_diff = elo_away - elo_home
    elif matchup_type == "single_game":
        rank_diff = elo_away - elo_home - home_adv
    else:
        print(f"Unknown matchup type: {matchup_type}. Defaulting to single game.")
        rank_diff = elo_away - elo_home - home_adv
    we = 1 / (1 + 10 ** (rank_diff / 400))
    return we


def simulate_match(proba, goal_adj=1):
    """
    Simulates the outcome of a football match based on a win probability.

    Given the probability of team 1 winning, this function estimates expected
    goals for both teams using a mathematical model, then simulates the actual
    number of goals scored using Poisson distributions.

    Args:
        proba (float): The probability of team 1 (home team) winning,
                       should be between 0 and 1.
        goal_adj (float): An adjustment factor that reduces expected goals
                        Mainly used to simulate xtra times where a factor of 1/3 is applied

    Returns:
        tuple: A tuple (GH, GA) where:
            - GH (int): Simulated number of goals scored by team 1 (home team).
            - GA (int): Simulated number of goals scored by team 2 (away team).
    """

    ExpGH = (
        np.where(
            proba < 0.5,
            0.2 + 1.1 * math.sqrt(proba / 0.5),
            1.69 / (1.12 * math.sqrt(2 - proba / 0.5) + 0.18),
        )
        * goal_adj
    )
    ExpGA = (
        np.where(
            (1 - proba) < 0.5,
            0.2 + 1.1 * math.sqrt((1 - proba) / 0.5),
            1.69 / (1.12 * math.sqrt(2 - (1 - proba) / 0.5) + 0.18),
        )
        * goal_adj
    )
    Base = np.random.poisson(0.18 * min(ExpGA, ExpGH)) * goal_adj
    GH = np.random.poisson(ExpGH - 0.18 * min(ExpGA, ExpGH)) + Base
    GA = np.random.poisson(ExpGA - 0.18 * min(ExpGA, ExpGH)) + Base

    return (GH, GA)


def simulate_extra_time(proba):
    """
    Simulates the outcome of a football match extra time based on a win probability of 90-minute game.

    Given the probability of team 1 winning, this function estimates expected
    goals for both teams using a mathematical model, then simulates the actual
    number of goals scored using Poisson distributions.

    Args:
        proba (float): The probability of team 1 (home team) winning,
                       should be between 0 and 1.

    Returns:
        str: 1 if team 1 wins, 2 if team 2 wins
    """

    GH, GA = simulate_match(proba, goal_adj=1 / 3)

    if GH > GA:
        result = 1
    elif GH < GA:
        result = 2
    else:
        # if it's a tie, we need to simulate penalties as random
        random_sim = np.random.rand()
        result = 1 if random_sim <= 0.5 else 2

    return result


def simulate_playoff(proba):
    """
    Simulates the outcome of a playoff football match based on a win probability.

    Given the probability of team 1 winning, this function simulates a playoff match
    where one of the two teams must win

    Args:
        proba (float): The probability of team 1 (home team) winning,
                       should be between 0 and 1.

    Returns:
        str: 1 if team 1 wins, 2 if team 2 wins
    """
    if not 0 <= proba <= 1:
        raise ValueError("Probability must be between 0 and 1 (inclusive).")
    random_sim = np.random.rand()
    result = 1 if random_sim <= proba else 2

    return result


def simulate_matches_data_frame(matches_df):
    """
    Simulate matches and determine winners.

    Parameters:
        matches_df (pd.DataFrame): DataFrame containing matches to simulate

    Returns:
        pd.DataFrame: DataFrame with simulation results
    """

    for index, match in matches_df.iterrows():

        home_advantage = 80 if match["neutral"] == "N" else 0
        elo_home = match["elo_home"]
        elo_away = match["elo_away"]

        # Calculate win probability for Team 1
        win_proba = calculate_win_probability(
            elo_home, elo_away, home_adv=home_advantage
        )

        # Simulate match
        result = simulate_match(win_proba)
        # Append result
        matches_df.at[index, "home_goals"] = result[0]
        matches_df.at[index, "away_goals"] = result[1]

    return pd.DataFrame(matches_df)


def apply_h2h_tiebreaker(matches_df, tied_teams, rule):
    """
    Applies a head-to-head (H2H) tiebreaker rule to a group of tied teams based on their matches against each other.

    Parameters:
    matches_df (pd.DataFrame):
        A DataFrame containing match results with at least the following columns: 'home', 'away', and any metrics
        used in calculating standings (e.g., goals, points).

    tied_teams (list of str):
        A list of team names that are currently tied in the standings.

    rule (str):
        The name of the H2H metric column (e.g., 'h2h_points', 'h2h_goal_diff') to use for ranking the tied teams.
        This must match one of the metrics returned by `get_standings_metrics`.

    Returns:
    pd.DataFrame:
        A DataFrame with two columns: 'team' and the selected `rule` metric, prefixed with 'h2h_'.
        It reflects the standings of tied teams based only on the matches they played against each other.
    """
    tied_matches_df = matches_df.copy()
    tied_matches_df = tied_matches_df[
        (matches_df["home"].isin(tied_teams)) & (matches_df["away"].isin(tied_teams))
    ]

    standings_tied = get_standings_metrics(tied_matches_df)
    # add h2h prefix to metrics
    standings_tied.columns = [
        f"h2h_{col}" if col != "team" else col for col in standings_tied.columns
    ]
    standings_tied = standings_tied[["team", rule]]
    return standings_tied


def apply_playoff_tiebreaker(matches_df, tied_teams):

    if len(tied_teams) > 2:
        standings_untied = get_standings(
            matches_df, classif_rules=["points", "h2h_points", "h2h_goal_difference"]
        )
        playoff_teams = standings_untied["team"].head(2).tolist()
        matches_df = matches_df[
            (matches_df["home"].isin(playoff_teams))
            & (matches_df["away"].isin(playoff_teams))
        ]

        standings_no_playoff = standings_untied.iloc[2:].reset_index()
        # assign starting frm -1 to rank them at the bottom
        # the top two teams will be 1 and 0 based on the playoff sim
        standings_no_playoff["playoff"] = (-1 * standings_no_playoff.index) - 1

    # matches_df has the tied teams and their elos
    elo_home = matches_df.iloc[0]["elo_home"]
    elo_away = matches_df.iloc[0]["elo_away"]
    we = calculate_win_probability(
        elo_home, elo_away, matchup_type="single_game_neutral"
    )
    result = simulate_playoff(we)

    standings_playoff = pd.DataFrame(
        {
            "team": [matches_df.iloc[0]["home"], matches_df.iloc[0]["away"]],
            "playoff": [1, 0] if result == 1 else [0, 1],
        }
    )
    if len(tied_teams) > 2:
        standings_tied = pd.concat([standings_playoff, standings_no_playoff])
    else:
        standings_tied = standings_playoff

    return standings_tied


def get_standings_metrics(matches_df):
    """
    Calculates basic league standings metrics for each team based on match results.

    This function processes match results to compute standard performance metrics for each team,
    including total points, goals scored, goals conceded, and goal difference. It aggregates home and
    away performance separately before combining them into overall team statistics.

    Parameters:
    matches_df (pd.DataFrame):
        A DataFrame containing match-level data. Must include the following columns:
        - 'home': name of the home team
        - 'away': name of the away team
        - 'home_goals': number of goals scored by the home team
        - 'away_goals': number of goals scored by the away team

    Returns:
    pd.DataFrame
        A DataFrame with one row per team and the following columns:
        - 'team': team name
        - 'points': total points (3 for win, 1 for draw, 0 for loss)
        - 'goal_difference': total goals scored minus goals conceded
        - 'goals_for': total goals scored (home + away)
        - 'goals_against': total goals conceded (home + away)
        - 'away_goals_for': goals scored in away matches (useful for tiebreakers)
    """

    matches_df["home_pts"] = np.where(
        matches_df["home_goals"] > matches_df["away_goals"],
        3,
        np.where(matches_df["home_goals"] == matches_df["away_goals"], 1, 0),
    )
    matches_df["away_pts"] = np.where(
        matches_df["away_goals"] > matches_df["home_goals"],
        3,
        np.where(matches_df["home_goals"] == matches_df["away_goals"], 1, 0),
    )
    matches_df["home_wins"] = np.where(
        matches_df["home_goals"] > matches_df["away_goals"],
        1,
        0
    )
    matches_df["away_wins"] = np.where(
        matches_df["away_goals"] > matches_df["home_goals"],
        1,
        0
    )
    home_pts = (
        matches_df.groupby(["home"])[["home_pts", "home_goals", "away_goals","home_wins"]]
        .sum()
        .reset_index()
    )
    away_pts = (
        matches_df.groupby(["away"])[["away_pts", "away_goals", "home_goals","away_wins"]]
        .sum()
        .reset_index()
    )

    home_pts = home_pts.rename(
        columns={
            "home": "team",
            "home_goals": "home_goals_for",
            "away_goals": "away_goals_against",
        }
    )
    away_pts = away_pts.rename(
        columns={
            "away": "team",
            "away_goals": "away_goals_for",
            "home_goals": "home_goals_against",
        }
    )
    # Combine wins and losses into a single DataFrame
    standings = pd.merge(home_pts, away_pts, how="outer", on="team").fillna(0)
    standings["points"] = standings["home_pts"] + standings["away_pts"]
    standings["goals_for"] = standings["home_goals_for"] + standings["away_goals_for"]
    standings["goals_against"] = (
        standings["home_goals_against"] + standings["away_goals_against"]
    )
    standings["goal_difference"] = standings["goals_for"] - standings["goals_against"]
    standings["wins"] = standings["home_wins"] + standings["away_wins"]

    standings = standings[
        [
            "team",
            "points",
            "goal_difference",
            "goals_for",
            "goals_against",
            "away_goals_for",
            "wins",
            "away_wins"
        ]
    ].fillna(0)

    return standings


def get_standings(matches_df, classif_rules):
    """
    Computes league standings metrics for each team and applies classification rules,
    including optional head-to-head (H2H) tiebreakers.

    This function calculates standard league standings such as total points, goal difference, and goals scored,
    and then ranks the teams using a list of classification rules. If any of the rules start with 'h2h',
    it applies a head-to-head tiebreaker among teams tied on all previous rules.

    Parameters:
    matches_df (pd.DataFrame):
        A DataFrame containing match-level data. Must include the following columns:
        - 'home': name of the home team
        - 'away': name of the away team
        - 'home_goals': number of goals scored by the home team
        - 'away_goals': number of goals scored by the away team

    classif_rules (list of str):
        A list of column names used to rank teams. These can include:
        - Basic metrics such as 'points', 'goal_difference', 'goals_for', etc.
        - Optional head-to-head metrics prefixed with 'h2h_', such as 'h2h_points', 'h2h_goal_difference', etc.
          If an 'h2h_' rule is encountered, it is used to break ties between teams tied on all prior rules.

    Returns:
    pd.DataFrame
        A DataFrame where each row corresponds to a team, with the following columns:
        - 'team': team name
        - standard performance metrics (e.g., 'points', 'goal_difference', etc.)
        - any head-to-head metrics added during tie-breaking
        - 'pos': final ranking position based on the classification rules

    """
    standings = get_standings_metrics(matches_df)
    # Sort by classification rules
    for i, rule in enumerate(classif_rules):
        is_h2h_rule = rule.startswith("h2h")
        is_playoff = rule.startswith("playoff")
        is_opponent_rule = rule.startswith("opponent")

        if (is_opponent_rule) & (rule not in standings.columns.tolist()):
            opponent_stats = get_opponents_aggregate_stats(matches_df, standings)
            standings = pd.merge(standings,opponent_stats,on='team')

        elif is_h2h_rule or is_playoff:
            # tiebreakers previous to current h2h one
            tb_applied = classif_rules[:i]
            # apply rank function to see who is tied
            standings["pos"] = (
                standings[tb_applied]
                .apply(tuple, axis=1)
                .rank(method="min", ascending=False)
                .astype(int)
            )
            # find tied teams
            pos_counts = standings["pos"].value_counts()
            ties = pos_counts[pos_counts >= 2]
            # playoff tie-breaker for italy championship or relegation
            if is_playoff:
                ties = ties[ties.index.isin([1, 18])]

            if len(ties) > 0:
                all_tied = []
                for tied_pos in ties.index.tolist():
                    subset_of_tied = standings[standings["pos"] == tied_pos]
                    tied_teams = subset_of_tied["team"].tolist()

                    if is_h2h_rule:
                        substed_tied_standings = apply_h2h_tiebreaker(
                            matches_df, tied_teams, rule
                        )
                    elif is_playoff:
                        substed_tied_standings = apply_playoff_tiebreaker(
                            matches_df, tied_teams
                        )

                    subset_of_tied = subset_of_tied.merge(
                        substed_tied_standings, on="team", how="left"
                    )
                    all_tied.append(subset_of_tied)

                all_tied = pd.concat(all_tied)
                standings = standings.merge(
                    all_tied[["team", rule]], how="left", on="team"
                )
            else:
                standings[rule] = np.nan

    # Add a random tie-breaker for remaining ties (e.g., team name or index)
    standings["__tiebreaker__"] = standings.index

    # Extend classification rules with the tiebreaker
    extended_rules = classif_rules + ["__tiebreaker__"]
    standings["pos"] = (
        standings[extended_rules]
        .apply(tuple, axis=1)
        .rank(method="min", ascending=False)
        .astype(int)
    )

    return standings


def get_opponents_aggregate_stats(matches_df, standings_df):
    """
    Calculates aggregate statistics (points, goal difference, and goals scored) 
    of all opponents each team has played against.

    For each team, this function identifies all opponents they've faced (as either 
    home or away team) and computes the sum of the opponents' points, goal 
    difference, and goals scored, based on the current standings.

    Args:
        matches_df (pd.DataFrame): DataFrame containing match results. 
            Must include 'home' and 'away' columns.
        standings_df (pd.DataFrame): DataFrame containing team standings.
            Must include 'team', 'points', 'goals_for', and 'goal_difference' columns.

    Returns:
        pd.DataFrame: A DataFrame with one row per team and the following columns:
            - 'team': The team name
            - 'opponent_points': Sum of all opponents' points
            - 'opponent_goal_difference': Sum of all opponents' goal difference
            - 'opponent_goals_for': Sum of all opponents' goals for
    """
    # Get set of all opponents each team has played against
    team_opponents = {}

    for _, row in matches_df.iterrows():
        team_opponents.setdefault(row['home'], set()).add(row['away'])
        team_opponents.setdefault(row['away'], set()).add(row['home'])

    # Create lookup dictionaries for points and goals_for
    points_lookup = standings_df.set_index("team")["points"].to_dict()
    goals_lookup = standings_df.set_index("team")["goals_for"].to_dict()
    goal_difference_lookup = standings_df.set_index("team")["goal_difference"].to_dict()

    # Build result
    result = []
    for team, opponents in team_opponents.items():
        total_points = sum(points_lookup.get(opp, 0) for opp in opponents)
        total_goal_difference = sum(goal_difference_lookup.get(opp, 0) for opp in opponents)
        total_goals = sum(goals_lookup.get(opp, 0) for opp in opponents)

        result.append({
            "team": team,
            "opponent_points": total_points,
            "opponent_goal_difference": total_goal_difference,
            "opponent_goals_for": total_goals

        })

    return pd.DataFrame(result)


def validate_bracket(bracket_df, bracket_format):
    """
    Validates a playoff bracket DataFrame.

    Checks for:
    - Missing or empty team slots
    - Duplicate team entries (excluding 'Bye')
    - Total number of slots being a power of 2
    - At least 2 non-'Bye' teams

    Args:
        bracket_df (pd.DataFrame): A DataFrame with columns ['team1', 'team2'] representing matchups.
        bracket_format (dict): Dictionary defining the format of each round in the knockout stage.
            Example: {"po_r32": "two-legged", "po_r16": "two-legged", ...}

    Raises:
        ValueError: If the bracket has invalid team slots, duplicates, or wrong number of teams.
    """
    # Combine all teams into a single Series
    teams = pd.concat([bracket_df["team1"], bracket_df["team2"]])
    # Check for empty slots (NaN or empty string)
    if teams.isnull().any() or (teams.astype(str).str.strip() == "").any():
        raise ValueError("Bracket contains empty team slots.")
    # Exclude 'Bye' from unique team check and count
    teams_no_bye = teams[teams != "Bye"]
    # Check for duplicate teams (excluding 'Bye')
    if teams_no_bye.duplicated().any():
        raise ValueError("Duplicate teams found in the bracket.")
    # Number of actual teams (excluding 'Bye')
    num_teams = len(teams_no_bye)
    # Number of slots (including 'Bye')
    num_slots = len(teams)
    # Number of teams must be a power of 2 (including 'Bye' slots)
    if num_slots == 0 or (num_slots & (num_slots - 1)) != 0:
        raise ValueError(
            "Total number of slots (including 'Bye') must be a power of 2."
        )
    # Number of actual teams must be at least 2
    if num_teams < 2:
        raise ValueError("At least two teams are required in the bracket.")
    # check that bracket format matches the number of rounds
    expected_rounds = int(math.log2(num_slots))
    if len(bracket_format) != expected_rounds:
        raise ValueError(
            f"Bracket format does not match the number of rounds. Expected {expected_rounds} rounds, got {len(bracket_format)}."
        )


def simulate_playoff_bracket(bracket_df, bracket_format, elos, playoff_schedule):
    """
    Simulates a knockout playoff bracket using ELO ratings.

    Args:
        bracket_df: Bracket structure with columns ['team1', 'team2']
        bracket_format: Dictionary defining the format of each round
        elos: DataFrame with columns ['team', 'elo'] representing team ELO ratings
        playoff_schedule: DataFrame with pending matches to simulate

    Returns:
        Wide-format DataFrame with one row per team and binary indicators for each round
    """
    validate_bracket(bracket_df, bracket_format)

    elos_dict = dict(zip(elos["team"], elos["elo"]))
    teams_progression = {}
    rounds = []

    current_round = bracket_df.copy()

    while len(current_round) > 0:
        round_label = f"po_r{2 * len(current_round)}"
        round_format = bracket_format[round_label]
        rounds.append(round_label)

        winners = _simulate_round(
            current_round,
            round_format,
            elos_dict,
            playoff_schedule,
            teams_progression,
            round_label,
        )

        current_round = _prepare_next_round(winners)

    return _build_results_dataframe(teams_progression, rounds)


def _simulate_round(
    current_round,
    round_format,
    elos_dict,
    playoff_schedule,
    teams_progression,
    round_label,
):
    """
    Simulate all matches in a single playoff round and update team progression.

    Args:
        current_round (pd.DataFrame): DataFrame with columns 'team1' and 'team2' representing matchups.
        round_format (str): Format of the round (e.g., "single", "home_and_away").
        elos_dict (dict): Dictionary mapping team names to their ELO ratings.
        playoff_schedule (pd.DataFrame): Schedule of actual matches, used if available.
        teams_progression (dict): Dictionary tracking each team’s progress through the tournament.
        round_label (str): Label indicating which round is being simulated.

    Returns:
        list: A list of team names that won their respective matchups in this round.
    """
    winners = []

    for _, row in current_round.iterrows():
        team1, team2 = row["team1"], row["team2"]

        winner = get_match_winner_from_playoff(
            team1, team2, round_format, elos_dict, playoff_schedule
        )
        winners.append(winner)

        _track_team_progression(
            team1, team2, winner, teams_progression, round_label, len(current_round)
        )

    return winners


def get_match_winner_from_playoff(
    team1, team2, round_format, elos_dict, playoff_schedule
):
    """
    Simulate a single playoff match between two teams.

    Args:
        team1 (str): Name of the first team.
        team2 (str): Name of the second team.
        round_format (str): Format of the round.
        elos_dict (dict): Dictionary of ELO ratings.
        playoff_schedule (pd.DataFrame): Schedule of real matches, if available.

    Returns:
        str: Name of the winning team.
    """
    if team1 == "Bye":
        return team2
    if team2 == "Bye":
        return team1

    team1_elo = elos_dict.get(team1, 1000)
    team2_elo = elos_dict.get(team2, 1000)

    win_proba = calculate_win_probability(
        team1_elo, team2_elo, matchup_type=round_format
    )
    tie_matches = _get_tie_matches(team1, team2, playoff_schedule)

    if tie_matches.empty:
        result = simulate_playoff(win_proba)
        return team1 if result == 1 else team2

    return _determine_winner_from_schedule(team1, team2, tie_matches, win_proba)


def _get_tie_matches(team1, team2, playoff_schedule):
    """
    Retrieve all scheduled matches between two teams from the playoff schedule.

    Args:
        team1 (str): First team.
        team2 (str): Second team.
        playoff_schedule (pd.DataFrame): DataFrame containing all playoff matches.

    Returns:
        pd.DataFrame: Subset of playoff_schedule for matches between the two teams.
    """
    return playoff_schedule[
        ((playoff_schedule["home"] == team1) & (playoff_schedule["away"] == team2))
        | ((playoff_schedule["home"] == team2) & (playoff_schedule["away"] == team1))
    ].copy()


def _determine_winner_from_schedule(team1, team2, tie_matches, win_proba):
    """
    Determine the winner based on match schedule data.

    Args:
        team1 (str): First team.
        team2 (str): Second team.
        tie_matches (pd.DataFrame): Subset of schedule with matches between the teams.
        win_proba (float): Probability of team1 winning.

    Returns:
        str: Name of the winning team.
    """
    if all(tie_matches["played"] == "Y"):
        return _get_winner_from_completed_matches(team1, team2, tie_matches)
    elif any(tie_matches["played"] == "Y"):
        return _get_winner_from_partial_matches(team1, team2, tie_matches, win_proba)
    else:
        result = simulate_playoff(win_proba)
        return team1 if result == 1 else team2


def _get_winner_from_completed_matches(team1, team2, tie_matches):
    """
    Determine the winner from completed two-leg matches.

    Args:
        team1 (str): First team.
        team2 (str): Second team.
        tie_matches (pd.DataFrame): DataFrame containing the matches.

    Returns:
        str: Name of the winning team.

    Raises:
        Warning: If extracted winner from notes is not one of the two teams.
    """
    string_result = tie_matches.iloc[-1]["notes"]
    match = re.search(r";\s*(.*?)\s+won", string_result)

    if match:
        winner = match.group(1)
        if winner not in [team1, team2]:
            raise Warning(f"Winner {winner} not in teams {team1}, {team2}")
        return winner

    return _get_winner_by_goals(team1, team2, tie_matches)


def _get_winner_from_partial_matches(team1, team2, tie_matches, win_proba):
    """
    Determine the winner from partially played ties.

    Args:
        team1 (str): First team.
        team2 (str): Second team.
        tie_matches (pd.DataFrame): DataFrame of matches.
        win_proba (float): Probability of team1 winning.

    Returns:
        str: Name of the winning team.
    """
    t1_goals, t2_goals = _calculate_total_goals(team1, team2, tie_matches)

    if t1_goals > t2_goals:
        return team1
    elif t2_goals > t1_goals:
        return team2
    else:
        # simulate extra time
        result = simulate_extra_time(win_proba)
        return team1 if result == 1 else team2


def _get_winner_by_goals(team1, team2, tie_matches):
    """
    Determine the winner based on total goals scored across matches.

    Args:
        team1 (str): First team.
        team2 (str): Second team.
        tie_matches (pd.DataFrame): Matches played between the two teams.

    Returns:
        str: Name of the winning team.

    Raises:
        ValueError: If both teams have equal total goals.
    """
    t1_goals, t2_goals = _calculate_total_goals(team1, team2, tie_matches)

    if t1_goals > t2_goals:
        return team1
    elif t2_goals > t1_goals:
        return team2
    else:
        raise ValueError(
            "Both teams scored the same number of goals, cannot determine winner."
        )


def _calculate_total_goals(team1, team2, tie_matches):
    """
    Calculate total goals scored by each team across their matches.

    Args:
        team1 (str): First team.
        team2 (str): Second team.
        tie_matches (pd.DataFrame): Matches played.

    Returns:
        tuple: Total goals for team1 and team2.
    """
    t1_goals = (
        tie_matches[tie_matches["home"] == team1]["home_goals"].sum()
        + tie_matches[tie_matches["away"] == team1]["away_goals"].sum()
    )

    t2_goals = (
        tie_matches[tie_matches["home"] == team2]["home_goals"].sum()
        + tie_matches[tie_matches["away"] == team2]["away_goals"].sum()
    )

    return t1_goals, t2_goals


def _track_team_progression(
    team1, team2, winner, teams_progression, round_label, round_size
):
    """
    Track and update team progression through each round.

    Args:
        team1 (str): First team.
        team2 (str): Second team.
        winner (str): Winner of the match.
        teams_progression (dict): Dictionary of team progression status.
        round_label (str): Current round label.
        round_size (int): Number of matches in the round.
    """
    for team in [team1, team2]:
        if team not in teams_progression:
            teams_progression[team] = {}

        teams_progression[team][round_label] = 1

        if round_size == 1 and team == winner:
            teams_progression[team]["po_champion"] = 1


def _prepare_next_round(winners):
    """
    Pair up winners to create matchups for the next round.

    Args:
        winners (list): List of team names who won their previous matches.

    Returns:
        pd.DataFrame: DataFrame with columns 'team1' and 'team2' for next round matchups.
    """
    if len(winners) < 2:
        return pd.DataFrame()

    it = iter(winners)
    next_round_pairs = list(zip(it, it))

    return pd.DataFrame(next_round_pairs, columns=["team1", "team2"])


def _build_results_dataframe(teams_progression, rounds):
    """
    Construct a final results DataFrame summarizing team progression.

    Args:
        teams_progression (dict): Dictionary tracking round-by-round progression of each team.
        rounds (list): List of round names in order.

    Returns:
        pd.DataFrame: DataFrame summarizing tournament outcome for all teams.
    """
    all_teams = list(teams_progression.keys())
    all_rounds = rounds + ["po_champion"]

    result = pd.DataFrame(index=all_teams, columns=all_rounds).fillna(0).astype(int)

    for team, progress in teams_progression.items():
        for round_name in progress:
            result.loc[team, round_name] = progress[round_name]

    return result.reset_index().rename(columns={"index": "team"})


def draw_from_pots(df, pot_size=2):
    """
    Randomly draws teams from position-based pots.

    Args:
        df (pd.DataFrame): DataFrame with columns ['team', 'pos'] where 'pos' determines pot grouping.
        pot_size (int): Number of positions per pot (default is 2).

    Returns:
        pd.DataFrame: A DataFrame with columns ['draw_order', 'team'] indicating the randomized draw result.
    """
    df = df.copy()
    df = df.sort_values("pos").reset_index(drop=True)

    # Map position → team
    pos_to_team = dict(zip(df["pos"], df["team"]))

    # Sort positions and group into pots
    sorted_positions = sorted(pos_to_team.keys())
    pots = [
        sorted_positions[i : i + pot_size]
        for i in range(0, len(sorted_positions), pot_size)
    ]

    draw_result = []
    for pot in pots:
        teams = [pos_to_team[pos] for pos in pot]
        random.shuffle(teams)  # shuffle in-place
        draw_result.extend(teams)

    # Assign back to a DataFrame
    return pd.DataFrame(
        {"draw_order": range(1, len(draw_result) + 1), "team": draw_result}
    )


def create_bracket_from_composition(df_with_draw, bracket_composition):
    """
    Creates a playoff bracket based on a predefined composition and a team draw.

    Args:
        df_with_draw (pd.DataFrame): DataFrame with columns ['draw_order', 'team'] from draw.
        bracket_composition (list of tuple): List of (pos1, pos2) tuples representing matchups.
            Values can be integers (draw positions) or 'Bye'.

    Returns:
        pd.DataFrame: A DataFrame with columns ['team1', 'team2'] representing the bracket.

    Raises:
        ValueError: If both sides of a match are 'Bye'.
    """
    pos_to_team = dict(zip(df_with_draw["draw_order"], df_with_draw["team"]))
    pairs = []

    for pos1, pos2 in bracket_composition:
        team1 = pos_to_team.get(pos1) if pos1 != "Bye" else "Bye"
        team2 = pos_to_team.get(pos2) if pos2 != "Bye" else "Bye"

        if team1 == "Bye" and team2 == "Bye":
            raise ValueError("Invalid bracket: both sides cannot be 'Bye'")

        pairs.append((team1, team2))

    return pd.DataFrame(pairs, columns=["team1", "team2"])
