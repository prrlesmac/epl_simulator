import pandas as pd
import numpy as np
import math


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


def simulate_matches(matches_df):
    """
    Simulate matches and determine winners.

    Parameters:
        matches_df (pd.DataFrame): DataFrame containing matches to simulate

    Returns:
        pd.DataFrame: DataFrame with simulation results
    """

    for index, match in matches_df.iterrows():

        elo_home = match["elo_home"]
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
        (matches_df['home'].isin(tied_teams))
        & (matches_df['away'].isin(tied_teams))
    ]

    standings_tied = get_standings_metrics(tied_matches_df)
    # add h2h prefix to metrics
    standings_tied.columns = [f'h2h_{col}' if col != 'team' else col for col in standings_tied.columns]
    standings_tied = standings_tied[['team', rule]]
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
    standings["goals_against"] = standings["home_goals_against"] + standings["away_goals_against"]
    standings["goal_difference"] = standings["goals_for"] - standings["goals_against"]
    standings = standings[["team", "points", "goal_difference", "goals_for", "goals_against","away_goals_for"]].fillna(0)

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
        if is_h2h_rule:
            # tiebreakers previous to current h2h one
            tb_applied = classif_rules[:i]
            # apply rank function to see who is tied
            standings['pos'] = standings[tb_applied].apply(tuple, axis=1).rank(
                method='min', ascending = False
            ).astype(int)
            # find tied teams
            pos_counts = standings['pos'].value_counts()
            ties = pos_counts[pos_counts >= 2]
            if len(ties) > 0:
                all_tied = []
                for tied_pos in ties.index.tolist():
                    subset_of_tied = standings[standings['pos'] == tied_pos]
                    tied_teams = subset_of_tied["team"].tolist()
                    ## apply h2h tiebreaker
                    substed_tied_standings = apply_h2h_tiebreaker(matches_df, tied_teams, rule)
                    subset_of_tied = (
                        subset_of_tied
                        .merge(substed_tied_standings, on='team', how='left')
                    )
                    all_tied.append(subset_of_tied)
                all_tied = pd.concat(all_tied)
                standings = standings.merge(
                    all_tied[["team",rule]],
                    how="left",
                    on="team"
                )
            else:
                standings[rule] = np.nan

    standings['pos'] = standings[classif_rules].apply(tuple, axis=1).rank(
                method='min', ascending = False
    ).astype(int)

    return standings
