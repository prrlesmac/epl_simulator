import pandas as pd
import numpy as np
import math
import random


def calculate_win_probability(elo_home, elo_away):
    """
    Calculate win probability using the Elo rating system.

    Parameters:
        elo_home (float): Elo rating of team 1
        elo_away (float): Elo rating of team 2

    Returns:
        float: Probability of team 1 winning
    """
    return 1 / (1 + 10 ** ((elo_away - elo_home) / 400))


def simulate_match(proba):
    """
    Simulates the outcome of a football match based on a win probability.

    Given the probability of team 1 winning, this function estimates expected
    goals for both teams using a mathematical model, then simulates the actual
    number of goals scored using Poisson distributions.

    Args:
        proba (float): The probability of team 1 (home team) winning,
                       should be between 0 and 1.

    Returns:
        tuple: A tuple (GH, GA) where:
            - GH (int): Simulated number of goals scored by team 1 (home team).
            - GA (int): Simulated number of goals scored by team 2 (away team).
    """

    ExpGH = np.where(
        proba < 0.5,
        0.2 + 1.1 * math.sqrt(proba / 0.5),
        1.69 / (1.12 * math.sqrt(2 - proba / 0.5) + 0.18),
    )
    ExpGA = np.where(
        (1 - proba) < 0.5,
        0.2 + 1.1 * math.sqrt((1 - proba) / 0.5),
        1.69 / (1.12 * math.sqrt(2 - (1 - proba) / 0.5) + 0.18),
    )
    Base = np.random.poisson(0.18 * min(ExpGA, ExpGH))
    GH = np.random.poisson(ExpGH - 0.18 * min(ExpGA, ExpGH)) + Base
    GA = np.random.poisson(ExpGA - 0.18 * min(ExpGA, ExpGH)) + Base
    # Diff = abs(GH - GA)
    # K = 30 * np.where(Diff <= 1, 1, np.where(Diff == 2, 1.25, 1.25 + (Diff - 2) / 8))
    # Res = np.where(GH > GA, 1, np.where(GH == GA, 0.5, 0))
    # EloDiff = (proba - Res) * K
    # EloH = EloH - EloDiff
    # EloA = EloA + EloDiff

    return (GH, GA)


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
    random_sim = np.random.rand()
    result = 1 if random_sim <= proba else 2

    return result


def simulate_matches(matches_df, home_advantage):
    """
    Simulate matches and determine winners.

    Parameters:
        matches_df (pd.DataFrame): DataFrame containing matches to simulate

    Returns:
        pd.DataFrame: DataFrame with simulation results
    """

    for index, match in matches_df.iterrows():

        home_advantage = 80 if match["neutral"] == "N" else 0
        elo_home = match["elo_home"] + home_advantage
        elo_away = match["elo_away"]

        # Calculate win probability for Team 1
        win_proba = calculate_win_probability(elo_home, elo_away)

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
    we = calculate_win_probability(elo_home, elo_away)
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
    home_pts = (
        matches_df.groupby(["home"])[["home_pts", "home_goals", "away_goals"]]
        .sum()
        .reset_index()
    )
    away_pts = (
        matches_df.groupby(["away"])[["away_pts", "away_goals", "home_goals"]]
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
    standings = standings[
        [
            "team",
            "points",
            "goal_difference",
            "goals_for",
            "goals_against",
            "away_goals_for",
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

        if is_h2h_rule or is_playoff:
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

    standings["pos"] = (
        standings[classif_rules]
        .apply(tuple, axis=1)
        .rank(method="min", ascending=False)
        .astype(int)
    )

    return standings


def validate_bracket(bracket_df):
    """
    Validates a playoff bracket DataFrame.

    Checks for:
    - Missing or empty team slots
    - Duplicate team entries (excluding 'Bye')
    - Total number of slots being a power of 2
    - At least 2 non-'Bye' teams

    Args:
        bracket_df (pd.DataFrame): A DataFrame with columns ['team1', 'team2'] representing matchups.

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


def simulate_playoff_bracket(bracket_df, elos):
    """
    Simulates a knockout playoff bracket using ELO ratings.

    Args:
        bracket_df (pd.DataFrame): Bracket structure with columns ['team1', 'team2'].
        elos (pd.DataFrame): DataFrame with columns ['team', 'elo'] representing team ELO ratings.

    Returns:
        pd.DataFrame: A wide-format DataFrame with one row per team and binary indicators for each round and champion status.

    Raises:
        ValueError: If the bracket is invalid.
    """
    elos_dict = dict(zip(elos["team"], elos["elo"]))
    rounds = []
    teams_progression = {}

    # Validate the bracket structure
    validate_bracket(bracket_df)

    current_round = bracket_df.copy()
    round_number = 1

    while len(current_round) > 0:
        round_label = f"po_r{2 * len(current_round)}"
        rounds.append(round_label)

        winners = []
        for _, row in current_round.iterrows():
            team1, team2 = row["team1"], row["team2"]

            # Randomly choose a winner
            if team1 == "Bye":
                winner = team2
            elif team2 == "Bye":
                winner = team1
            else:
                # add logic to get elo ratings from a df called elos
                match_elos = pd.Series([team1, team2]).map(elos_dict)

                # Calculate win probability for Team 1
                win_proba = calculate_win_probability(match_elos[0], match_elos[1])

                # Simulate match
                result = simulate_playoff(win_proba)
                winner = team1 if result == 1 else team2
            winners.append(winner)

            # Track participation
            for team in [team1, team2]:
                if team not in teams_progression:
                    teams_progression[team] = {}
                teams_progression[team][round_label] = 1
                if len(current_round) == 1 and team == winner:
                    teams_progression[team]["po_champion"] = 1

        # Prepare next round
        it = iter(winners)
        next_round = list(zip(it, it))
        current_round = (
            pd.DataFrame(next_round, columns=["team1", "team2"])
            if next_round
            else pd.DataFrame()
        )

        round_number += 1

    # Build output DataFrame
    all_teams = list(teams_progression.keys())
    rounds.append("Champion")
    all_rounds = rounds

    result = pd.DataFrame(index=all_teams, columns=all_rounds).fillna(0).astype(int)
    for team, progress in teams_progression.items():
        for rnd in progress:
            result.loc[team, rnd] = progress[rnd]

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
