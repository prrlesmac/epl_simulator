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


def simulate_match_goals(proba, goal_adj=1):
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

    GH, GA = simulate_match_goals(proba, goal_adj=1 / 3)

    if GH > GA:
        result = 1
    elif GH < GA:
        result = 2
    else:
        # if it's a tie, we need to simulate penalties as random
        random_sim = np.random.rand()
        result = 1 if random_sim <= 0.5 else 2

    return result


def simulate_match_winner(proba):
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

def simulate_series_winner(team1_elo, team2_elo, best_of):
    """

    """
    team1_wins = 0
    team2_wins = 0
    for i in range(best_of):
        if best_of == 3:
            proba = calculate_win_probability(team1_elo, team2_elo)
        else:
            if (i+1) < (best_of / 2):
                proba = 1 - calculate_win_probability(team2_elo, team1_elo)
            else:
                proba = calculate_win_probability(team1_elo, team2_elo)

        random_sim = np.random.rand()

        if random_sim <= proba:
            team1_wins += 1
        else:
            team2_wins += 1
        if team1_wins > (best_of / 2):
            break
        if team2_wins > (best_of / 2):
            break
    
    if team1_wins > team2_wins:
        result = 1
    elif team2_wins > team1_wins:
        result = 2
    else:
        raise(ValueError("Series simulation ended in tie"))

    return result


def simulate_matches_data_frame(matches_df, sim_type):
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
        if sim_type == "goals":
            result = simulate_match_goals(win_proba)
            # Append result
            matches_df.at[index, "home_goals"] = result[0]
            matches_df.at[index, "away_goals"] = result[1]
        elif sim_type == "winner":
            result = simulate_match_winner(win_proba)
            # Append result
            matches_df.at[index, "home_goals"] = 1 if result == 1 else 0
            matches_df.at[index, "away_goals"] = 0 if result == 1 else 1
        else:
            raise(ValueError("Invalid sim type in simulate_matches_data_frame"))
            
    return pd.DataFrame(matches_df)


def simulate_play_in_tourney(standings_df, playoff_schedule, elos):

    play_in_schedule = playoff_schedule.copy()
    play_in_schedule = play_in_schedule[play_in_schedule["round"] == "Play-In Game"]
    play_in_schedule['winner'] = np.where(
        play_in_schedule['home_goals'] > play_in_schedule['away_goals'],
        play_in_schedule['home'],
        np.where(
            play_in_schedule['home_goals'] < play_in_schedule['away_goals'],
            play_in_schedule['away'],
            'tie'
        )
    )
    play_in_schedule['loser'] = np.where(
        play_in_schedule['home_goals'] < play_in_schedule['away_goals'],
        play_in_schedule['home'],
        np.where(
            play_in_schedule['home_goals'] > play_in_schedule['away_goals'],
            play_in_schedule['away'],
            'tie'
        )
    )
    play_in_seeds = []
    # stage 1: play-in not in schedule
    if len(play_in_schedule) == 0:
        for conf, group in standings_df.groupby("conference"):
            play_in_first_round = []
            group = group.sort_values("conference_pos")
            # 7 v 8
            conf_play_in_1 = {
                    "team1": group.iloc[6]["team"],
                    "team2": group.iloc[7]["team"]
            }
            play_in_first_round.append(conf_play_in_1)
            # 9 v 10
            conf_play_in_2 = {
                    "team1": group.iloc[8]["team"],
                    "team2": group.iloc[9]["team"]
            }
            play_in_first_round.append(conf_play_in_2)
            play_in_first_round = pd.DataFrame(play_in_first_round)
            winner = _simulate_round(
                play_in_first_round,
                round_format="single_game",
                elos_dict=elos,
                playoff_schedule=play_in_schedule,
                teams_progression={},
                round_label="play_in_first_round"
            )
            seed_no_7 = {
                "team": winner[0],
                "playoff_pos_play_in": f"{conf} 7"
            }
            play_in_seeds.append(seed_no_7)
            conf_play_in_2_loser = [v for k, v in conf_play_in_2.items() if v != winner[1]][0]
            seed_no_10 = {
                "team": conf_play_in_2_loser,
                "playoff_pos_play_in": f"{conf} 10"
            }
            play_in_seeds.append(seed_no_10)

            # 8 seed match
            conf_play_in_1_loser = [v for k, v in conf_play_in_1.items() if v != winner[0]][0]
            conf_play_in_3 = {
                    "team1": conf_play_in_1_loser,
                    "team2": winner[1]
            }
            play_in_second_round = pd.DataFrame([conf_play_in_3])
            winner = _simulate_round(
                play_in_second_round,
                round_format="single_game",
                elos_dict=elos,
                playoff_schedule=play_in_schedule,
                teams_progression={},
                round_label="play_in_second_round"
            )
            seed_no_8 = {
                "team": winner[0],
                "playoff_pos_play_in": f"{conf} 8"
            }
            play_in_seeds.append(seed_no_8)

            conf_play_in_3_loser = [v for k, v in conf_play_in_3.items() if v != winner[0]][0]
            seed_no_9 = {
                "team": conf_play_in_3_loser,
                "playoff_pos_play_in": f"{conf} 9"
            }
            play_in_seeds.append(seed_no_9)

    # stage 2: play-in first round only in schedule
    elif len(play_in_schedule) == 4:
        winners = play_in_schedule['winner'].tolist()
        losers = play_in_schedule['loser'].tolist()

        # find winners in 7-8
        for conf, group in standings_df.groupby("conference"):
            group = group.sort_values("conference_pos")
            # 7 v 8
            conf_play_in_1 = group.iloc[6:8]["team"]
            winner_7_8 = conf_play_in_1[conf_play_in_1.isin(winners)].values[0]
            seed_no_7 = {
                "team": winner_7_8,
                "playoff_pos_play_in": f"{conf} 7"
            }
            play_in_seeds.append(seed_no_7)

            # second game
            conf_play_in_2 = group.iloc[8:10]["team"]
            loser_7_8 = conf_play_in_1[conf_play_in_1.isin(losers)].values[0]
            winner_9_10 = conf_play_in_2[conf_play_in_2.isin(winners)].values[0]
            loser_9_10 = conf_play_in_2[conf_play_in_2.isin(losers)].values[0]
            seed_no_10 = {
                "team": loser_9_10,
                "playoff_pos_play_in": f"{conf} 10"
            }
            play_in_seeds.append(seed_no_10)
            conf_play_in_3 = {
                    "team1": loser_7_8,
                    "team2": winner_9_10
            }
            play_in_second_round = pd.DataFrame([conf_play_in_3])
            winner = _simulate_round(
                play_in_second_round,
                round_format="single_game",
                elos_dict=elos,
                playoff_schedule=play_in_schedule,
                teams_progression={},
                round_label="play_in_second_round"
            )
            seed_no_8 = {
                "team": winner[0],
                "playoff_pos_play_in": f"{conf} 8"
            }
            play_in_seeds.append(seed_no_8)
            conf_play_in_3_loser = [v for k, v in conf_play_in_3.items() if v != winner[0]][0]
            seed_no_9 = {
                "team": conf_play_in_3_loser,
                "playoff_pos_play_in": f"{conf} 9"
            }
            play_in_seeds.append(seed_no_9)

    # stage 3: all play-in games have result
    elif len(play_in_schedule) == 6:
        for conf, group in play_in_schedule.groupby("home_conference"):
            wins = group['winner'].value_counts()
            losses = group['loser'].value_counts()

            # teams with one win and one played are 7th seed
            seed_no_7 = wins[(wins == 1) & (~wins.index.isin(losses.index))].index.tolist()[0]
            seed_no_7 = {
                "team": seed_no_7,
                "playoff_pos_play_in": f"{conf} 7"
            }
            play_in_seeds.append(seed_no_7)

            # teams with one loss and one played are 10th seed
            seed_no_10 = losses[(losses == 1) & (~losses.index.isin(wins.index))].index.tolist()[0]
            seed_no_10 = {
                "team": seed_no_10,
                "playoff_pos_play_in": f"{conf} 10"
            }
            play_in_seeds.append(seed_no_10)

            # teams who won last game are 8th seed
            seed_no_8 = group.iloc[-1]["winner"]
            seed_no_8 = {
                "team": seed_no_8,
                "playoff_pos_play_in": f"{conf} 8"
            }
            play_in_seeds.append(seed_no_8)

            # teams who lost last game are 9th seed
            seed_no_9 = group.iloc[-1]["loser"]
            seed_no_9 = {
                "team": seed_no_9,
                "playoff_pos_play_in": f"{conf} 9"
            }
            play_in_seeds.append(seed_no_9)

    else:
        raise(ValueError("Invalid data for NBA play-in simulation"))

    standings_df = pd.merge(standings_df,
                            pd.DataFrame(play_in_seeds),
                            how="left",
                            on="team")
    standings_df["playoff_pos"] = np.where(
        standings_df["playoff_pos_play_in"].isna(),
        standings_df["playoff_pos"],
        standings_df["playoff_pos_play_in"]
    )

    return standings_df


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
    # TODO move away from using standings metrics and instead just call a win loss h2h function
    if rule=="h2h_win_loss_pct":
        standings_tied = get_standings_metrics_us(tied_matches_df)
    else:
        standings_tied = get_standings_metrics_footy(tied_matches_df)
    # add h2h prefix to metrics
    standings_tied.columns = [
        f"h2h_{col}" if col != "team" else col for col in standings_tied.columns
    ]
    standings_tied = standings_tied[["team", rule]]
    return standings_tied


def apply_h2h_sweep_tiebreaker(matches_df, tied_teams, sweep_type='full'):
    """
    Applies a head-to-head sweep tiebreaker rule to a group of tied teams based on their matches against each other.

    Parameters:
    matches_df (pd.DataFrame):
        A DataFrame containing match results with at least the following columns: 'home', 'away', and any metrics
        used in calculating standings (e.g., goals, points).

    tied_teams (list of str):
        A list of team names that are currently tied in the standings.

    sweep_type (str):
        Either "full": the team needs to have won all their games against all others
        or "win_loss_pct": the team needs to have a better win loss pct against all others

    Returns:
    pd.DataFrame:
        A DataFrame with two columns: 'team' and the h2h sweep tiebreaker result
    """
    tied_matches_df = matches_df.copy()
    tied_matches_df = tied_matches_df[
        (matches_df["home"].isin(tied_teams)) & (matches_df["away"].isin(tied_teams))
    ]
    for sel_team in tied_teams:
        if sweep_type=="full":
            home_wins = tied_matches_df[(tied_matches_df['home'] == sel_team) & (tied_matches_df['home_goals'] > tied_matches_df['away_goals'])]["away"].tolist()
            home_losses_or_ties = tied_matches_df[(tied_matches_df['home'] == sel_team) & (tied_matches_df['home_goals'] <= tied_matches_df['away_goals'])]["away"].tolist()
            away_wins = tied_matches_df[(tied_matches_df['away'] == sel_team) & (tied_matches_df['home_goals'] < tied_matches_df['away_goals'])]["home"].tolist()
            away_losses_or_ties = tied_matches_df[(tied_matches_df['away'] == sel_team) & (tied_matches_df['home_goals'] >= tied_matches_df['away_goals'])]["home"].tolist()
            wins = list(set(home_wins+away_wins))
            losses = list(set(home_losses_or_ties+away_losses_or_ties))
            h2h_sweep = (len(wins) == (len(tied_teams) - 1)) & (len(losses)==0)

        elif sweep_type=="win_loss_pct":
            rival_teams = [x for x in tied_teams if x!=sel_team]
            win_loss_pct_list = []
            for sel_rival in rival_teams:
                h2h_tied_matches_df = tied_matches_df.copy()
                h2h_tied_matches_df = h2h_tied_matches_df[
                    (h2h_tied_matches_df["home"].isin([sel_team,sel_rival])) & (h2h_tied_matches_df["away"].isin([sel_team,sel_rival]))
                ]
                win_loss_pct_df = get_win_loss_pct(h2h_tied_matches_df)
                win_loss_pct = win_loss_pct_df[win_loss_pct_df['team']==sel_team]["win_loss_pct"].values[0]
                win_loss_pct_list.append(win_loss_pct)
            h2h_sweep = all(x > 0.50001 for x in win_loss_pct_list)

        else:
            raise(ValueError("Invalid sweep type"))
        if h2h_sweep:
            h2h_sweep_team = sel_team
            break
    
    if h2h_sweep:
        standings_tied = pd.DataFrame({
            "team": tied_teams,
        })
        standings_tied[f"h2h_sweep_{sweep_type}"] = np.where(
            standings_tied["team"] == h2h_sweep_team,
            1,
            0
        )

    else:
        standings_tied = pd.DataFrame({
            "team": tied_teams,
            f"h2h_sweep_{sweep_type}": [0] * len(tied_teams)  # creates a list of zeros with same length as teams
        })
    return standings_tied


def apply_common_games_tiebreaker(matches_df, tied_teams):
    """
    Applies a head-to-head (H2H) tiebreaker rule to a group of tied teams based on their matches against each other.

    Parameters:
    matches_df (pd.DataFrame):
        A DataFrame containing match results with at least the following columns: 'home', 'away', and any metrics
        used in calculating standings (e.g., goals, points).

    tied_teams (list of str):
        A list of team names that are currently tied in the standings.

    Returns:
    pd.DataFrame:
        A DataFrame with two columns: 'team' and the common games win pct 'common_games_win_pct'
        It reflects the standings of tied teams based on the win loss pct of the games played 
        against common opponents
    """
    tied_matches_df = matches_df.copy()
    #find common opponents
    opponents_by_team = {}
    for sel_team in tied_teams:
        home_opponents = tied_matches_df[tied_matches_df["home"] == sel_team]["away"].tolist()
        away_opponents = tied_matches_df[tied_matches_df["away"] == sel_team]["home"].tolist()
        all_opponents = list(set(home_opponents + away_opponents))
        opponents_by_team[sel_team] = all_opponents

    common_opponents = set.intersection(*(set(opp) for opp in opponents_by_team.values()))
    if len(common_opponents)>=4:
        common_matches_df = matches_df[
            (matches_df["home"].isin(tied_teams) & matches_df["away"].isin(common_opponents))
            | (matches_df["home"].isin(common_opponents) & matches_df["away"].isin(tied_teams))
        ]

        standings_tied = get_standings_metrics_us(common_matches_df)
        standings_tied = standings_tied.rename(columns={"win_loss_pct": "h2h_win_loss_pct_common_games"})
        standings_tied = standings_tied[["team", "h2h_win_loss_pct_common_games"]]
        standings_tied = standings_tied[standings_tied["team"].isin(tied_teams)]
    else:
        standings_tied = pd.DataFrame({
            "team": tied_teams,
            "h2h_win_loss_pct_common_games": [0] * len(tied_teams)  # creates a list of zeros with same length as teams
        })
    return standings_tied



def apply_break_division_tiebreaker(standings):
    """
    Applies a head-to-head (H2H) tiebreaker rule to a group of tied teams based on their matches against each other.

    Parameters:
    standings (pd.DataFrame):
        A DataFrame containing league standings, including team name and division position

    Returns:
    pd.DataFrame:
        A DataFrame with two columns: 'team' and the division tiebreaker position 'h2h_break_division_ties'
    """
    standings_tied = standings.copy()
    standings_tied['rank'] = standings_tied.groupby('division')['division_pos'].rank(method='dense', ascending=True)
    standings_tied['h2h_break_division_ties'] = np.where(
        standings_tied['rank'] == 1,
        1,
        0
    )

    return standings_tied

def apply_playoff_tiebreaker(matches_df, tied_teams):

    if len(tied_teams) > 2:
        standings_untied = get_standings(
            matches_df, classif_rules={"league": ["points", "h2h_points", "h2h_goal_difference"]}, league_type="UEFA"
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
    result = simulate_match_winner(we)

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


def apply_win_loss_pct_same_div_tiebreaker(standings):
    """
    Apply the “same-division win-loss percentage” tiebreaker.

    This function checks whether all teams in the provided standings belong to
    the same division. If they do, it returns each team's win-loss percentage
    within that division. If not, the tiebreaker is not applicable and all
    teams receive a value of 0.

    Parameters
    ----------
    standings : pandas.DataFrame
        DataFrame containing at least:
        - 'team' : team name  
        - 'division' : division label  
        - 'win_loss_pct_div' : win-loss percentage within the division

    Returns
    -------
    pandas.DataFrame
        A DataFrame with columns:
        - 'team'
        - 'win_loss_pct_div_if_same_div' : division W-L percentage if all
          teams share the same division; otherwise 0.
    """
    standings_tied = standings.copy()
    same_division = standings_tied["division"].nunique() == 1
    if same_division:
        standings_tied["win_loss_pct_div_if_same_div"] = standings_tied["win_loss_pct_div"]
    else:
        standings_tied["win_loss_pct_div_if_same_div"] = 0

    standings_tied = standings_tied[["team","win_loss_pct_div_if_same_div"]]

    return standings_tied
    

def get_standings_metrics_footy(matches_df):
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


def get_standings_metrics_us(matches_df):
    """
    Calculate team standings metrics for a U.S.-style sports league.

    This function computes win/loss percentages at the league, conference, and 
    division levels, as well as additional tie-breaking metrics such as 
    last-half conference win percentage, strength of victory, and strength of 
    schedule. These metrics are typically used in standings tables for playoff 
    qualification and tie-breaking.

    Args:
        matches_df (pd.DataFrame): A DataFrame of match results containing at least:
            - "home", "away": team identifiers for home and away teams
            - "home_goals", "away_goals": scores for each match
            - "home_conference", "away_conference": conference identifiers
            - "home_division", "away_division": division identifiers

    Returns:
        pd.DataFrame: A DataFrame of team standings with the following columns:
            - "team": Team identifier
            - "win_loss_pct": Overall win-loss-tie percentage
            - "wins": Total wins
            - "ties": Total ties
            - "played": Total games played
            - "win_loss_pct_conf": Win-loss percentage within the conference
            - "win_loss_pct_div": Win-loss percentage within the division
            - Columns from `get_win_loss_pct_last_half` (e.g., last-half conf record)
            - Columns from `get_opponents_strength` for strength of victory/schedule

    Notes:
        - The function relies on helper functions:
            * `get_win_loss_pct`: Computes win-loss-tie percentage.
            * `get_win_loss_pct_last_half`: Computes last-half conference win-loss pct.
            * `get_opponents_strength`: Computes opponent-based strength metrics.
        - The "strength_of_victory" and "strength_of_schedule" metrics depend on 
          interpreting opponents' aggregated win-loss records.
    """
    win_loss_league = get_win_loss_pct(matches_df)
    win_loss_playoff_teams = get_win_loss_pct_playoff_teams(matches_df, win_loss_league)
    strength_of_victory = get_opponents_strength(matches_df, win_loss_league, strength_of="schedule")
    strength_of_schedule = get_opponents_strength(matches_df, win_loss_league, strength_of="victory")

    standings = (
        win_loss_league
        .merge(strength_of_victory, how="left", on="team")
        .merge(strength_of_schedule, how="left", on="team")
        .merge(win_loss_playoff_teams, how="left", on="team")
    )

    matches_df_conf = matches_df[matches_df["home_conference"]==matches_df["away_conference"]].copy()
    matches_df_div = matches_df[matches_df["home_division"]==matches_df["away_division"]].copy()

    if len(matches_df_conf) > 0:
        win_loss_conf = get_win_loss_pct(matches_df_conf)
        win_loss_last_half_conf = get_win_loss_pct_last_half(matches_df_conf)
        win_loss_conf = (
            win_loss_conf[["team","win_loss_pct"]]
            .rename(columns= {"win_loss_pct": "win_loss_pct_conf"})
        )
        standings = (
            standings
            .merge(win_loss_conf, how="left", on="team")
            .merge(win_loss_last_half_conf, how="left", on="team")
        )

    if len(matches_df_div) > 0:
        win_loss_div = get_win_loss_pct(matches_df_div)
        win_loss_div = (
            win_loss_div[["team","win_loss_pct"]]
            .rename(columns= {"win_loss_pct": "win_loss_pct_div"})
        )
        standings = (
            standings
            .merge(win_loss_div, how="left", on="team")
        )
        
    return standings


def get_win_loss_pct(matches_df):
    """
    Calculate win/loss percentage and basic record statistics for each team.

    This function computes wins, ties, games played, and overall win-loss-tie 
    percentage for each team across all matches. The win-loss percentage is 
    calculated as:

        win_loss_pct = (wins + 0.5 * ties) / played

    Args:
        matches_df (pd.DataFrame): A DataFrame of match results containing:
            - "home", "away": team identifiers
            - "home_goals", "away_goals": final scores for each match

    Returns:
        pd.DataFrame: A DataFrame containing team-level results with columns:
            - "team": Team identifier
            - "win_loss_pct": Overall win-loss-tie percentage (0–1, rounded to 3 decimals)
            - "wins": Total number of wins
            - "ties": Total number of ties
            - "played": Total number of games played

    Notes:
        - Home and away performances are aggregated into a single record per team.
        - Losses are implied as `played - wins - ties`.
        - Missing teams (e.g., teams that only played home or away) are handled 
          by filling missing values with 0.
    """
    matches_df_wl = matches_df.copy()
    matches_df_wl["home_wins"] = np.where(
        matches_df_wl["home_goals"] > matches_df_wl["away_goals"],
        1,
        0
    )
    matches_df_wl["away_wins"] = np.where(
        matches_df_wl["away_goals"] > matches_df_wl["home_goals"],
        1,
        0
    )
    matches_df_wl["home_ties"] = np.where(
        matches_df_wl["home_goals"] == matches_df_wl["away_goals"],
        1,
        0
    )
    matches_df_wl["away_ties"] = np.where(
        matches_df_wl["away_goals"] == matches_df_wl["home_goals"],
        1,
        0
    )
    home_pts = (
        matches_df_wl.groupby(["home"])[["home_wins","home_ties"]]
        .sum()
        .assign(home_played=matches_df_wl.groupby("home").size())
        .reset_index()
    )
    away_pts = (
        matches_df_wl.groupby(["away"])[["away_wins","away_ties"]]
        .sum()
        .assign(away_played=matches_df_wl.groupby("away").size())
        .reset_index()
    )
    home_pts = home_pts.rename(
        columns={
            "home": "team",
        }
    )
    away_pts = away_pts.rename(
        columns={
            "away": "team",
        }
    )
    # Combine wins and losses into a single DataFrame
    standings = pd.merge(home_pts, away_pts, how="outer", on="team").fillna(0)
    standings["wins"] = standings["home_wins"] + standings["away_wins"]
    standings["ties"] = standings["home_ties"] + standings["away_ties"]
    standings["played"] = standings["home_played"] + standings["away_played"]
    standings["win_loss_pct"] = round((standings["wins"] + standings["ties"] * 0.5) / (standings["played"]), 3)

    standings = standings[
        [
            "team",
            "win_loss_pct",
            "wins",
            "ties",
            "played",
        ]
    ].fillna(0)

    return standings

def get_win_loss_pct_last_half(matches_df):
    """
    Calculate win-loss percentage for each team in the second half of their season.

    This function computes each team's win-loss percentage based only on games
    played in the last half of their schedule. The calculation is done by:
      1. Identifying each team's games.
      2. Splitting the schedule in half (by game order per team).
      3. Calculating the number of wins and games played in the last half.

    Args:
        matches_df (pd.DataFrame): A DataFrame of match results containing:
            - "home", "away": team identifiers
            - "home_goals", "away_goals": match scores

    Returns:
        pd.DataFrame: A DataFrame with one row per team and columns:
            - "team": Team identifier
            - "win_loss_pct_conference_last_half": Win-loss percentage 
              for the second half of the season (0–1, rounded to 3 decimals)

    Notes:
        - The "second half" is defined per team, based on the order of games
          they played (not by calendar date).
        - Ties are not explicitly considered; the metric is calculated as
          wins / games played.
        - Uses a simple range assignment for match numbers, so ensure the
          input DataFrame is chronologically ordered.
    """
    list_of_teams = set(matches_df["home"].tolist() + matches_df["away"].tolist())
    standings = []
    for sel_team in list_of_teams:

        matches_last_half = matches_df.copy()
        matches_last_half = matches_last_half[(
            (matches_last_half["home"] == sel_team)
            | (matches_last_half["away"] == sel_team)
        )]
        matches_last_half["match_number"] = range(1, len(matches_last_half) + 1)
        matches_last_half = matches_last_half[(matches_last_half["match_number"] - 0.01) >= (len(matches_last_half)/2)]
        matches_last_half["home_wins"] = np.where(
            (matches_last_half["home_goals"] > matches_last_half["away_goals"])
            & (matches_last_half["home"]==sel_team),
            1,
            0
        )
        matches_last_half["away_wins"] = np.where(
            (matches_last_half["away_goals"] > matches_last_half["home_goals"])
            & (matches_last_half["away"]==sel_team),
            1,
            0
        )
        matches_last_half["away_wins"] = np.where(
            matches_last_half["away_goals"] > matches_last_half["home_goals"],
            1,
            0
        )
        home_wins = matches_last_half[matches_last_half['home']==sel_team]["home_wins"].sum()
        away_wins = matches_last_half[matches_last_half['away']==sel_team]["away_wins"].sum()
        played = len(matches_last_half)
        win_loss_pct = round((home_wins + away_wins) / played, 3)
        standings.append({"team": sel_team, "win_loss_pct_conference_last_half": win_loss_pct})

    standings = pd.DataFrame(standings)

    return standings


def get_win_loss_pct_playoff_teams(matches_df, win_loss_league, playoff_eligible_rank=6):
    """
    Compute each team's win-loss percentage against playoff-eligible teams
    (both within the same conference and in the other conference).

    This function:
    1. Determines each team's conference based on the match dataframe.
    2. Computes conference rankings using overall win-loss percentage.
    3. Identifies playoff-eligible teams (default: top 6 per conference).
    4. For every team in the league, filters their matches against:
         - Playoff teams from the same conference.
         - Playoff teams from the other conference.
    5. Calculates win-loss percentages in those subsets.

    Parameters
    ----------
    matches_df : pandas.DataFrame
        Match-level dataset containing at least:
        - 'home', 'away' : team names  
        - 'home_goals', 'away_goals' : integer match scores  
        - 'home_conference', 'away_conference' : conference labels for each team

    win_loss_league : pandas.DataFrame
        Team-level standings containing:
        - 'team' : team name  
        - 'win_loss_pct' : overall win-loss percentage used for ranking

    playoff_eligible_rank : int, optional (default = 6)
        Number of top teams per conference considered playoff-eligible.

    Returns
    -------
    pandas.DataFrame
        A dataframe with one row per team and columns:
        - 'team'
        - 'win_loss_pct_playoff_teams_same_conf'  : W-L percentage vs playoff teams in the same conference
        - 'win_loss_pct_playoff_teams_other_conf' : W-L percentage vs playoff teams in the other conference

    Notes
    -----
    - Teams with no matches against playoff teams in the other conference
      receive a value of 0 for `win_loss_pct_playoff_teams_other_conf`.
    - Ranking within conference uses `method="min"` so ties share the same rank.
    """
    divisions_home = matches_df[['home','home_conference']].drop_duplicates()
    divisions_home = divisions_home.rename(columns={'home': 'team', 'home_conference': 'conference'})
    divisions_away = matches_df[['away','away_conference']].drop_duplicates()
    divisions_away = divisions_away.rename(columns={'away': 'team', 'away_conference': 'conference'})
    divisions = pd.concat([divisions_home, divisions_away]).drop_duplicates()
    win_loss_league_w_div = win_loss_league.merge(
        divisions,
        how="left",
        on="team"
    )
    win_loss_league_w_div["conf_rank"] = (
    win_loss_league_w_div.groupby("conference")["win_loss_pct"]
      .rank(method="min", ascending=False)
      .astype(int)
    )
    playoff_teams_df = win_loss_league_w_div[win_loss_league_w_div["conf_rank"] <= playoff_eligible_rank][["team","conference"]]

    list_of_teams = set(matches_df["home"].tolist() + matches_df["away"].tolist())
    standings = []
    for sel_team in list_of_teams:

        same_conf = win_loss_league_w_div.loc[win_loss_league_w_div["team"] == sel_team, "conference"].unique()[0]
        list_of_playoff_teams_same_conf = playoff_teams_df.loc[playoff_teams_df["conference"]==same_conf,'team'].tolist()
        list_of_playoff_teams_other_conf = playoff_teams_df.loc[playoff_teams_df["conference"]!=same_conf,'team'].tolist()
        matches_vs_po_teams = matches_df.copy()
        matches_vs_po_teams = matches_vs_po_teams[(
            (matches_vs_po_teams["home"] == sel_team)
            | (matches_vs_po_teams["away"] == sel_team)
        )]
        matches_vs_po_teams_same_conf = matches_vs_po_teams[(
            (matches_vs_po_teams["home"].isin(list_of_playoff_teams_same_conf))
            | (matches_vs_po_teams["away"].isin(list_of_playoff_teams_same_conf))
        )].copy()    
        matches_vs_po_teams_other_conf = matches_vs_po_teams[(
            (matches_vs_po_teams["home"].isin(list_of_playoff_teams_other_conf))
            | (matches_vs_po_teams["away"].isin(list_of_playoff_teams_other_conf))
        )].copy()
        matches_vs_po_teams_same_conf["home_wins"] = np.where(
            (matches_vs_po_teams_same_conf["home_goals"] > matches_vs_po_teams_same_conf["away_goals"])
            & (matches_vs_po_teams_same_conf["home"]==sel_team),
            1,
            0
        )
        matches_vs_po_teams_same_conf["away_wins"] = np.where(
            (matches_vs_po_teams_same_conf["away_goals"] > matches_vs_po_teams_same_conf["home_goals"])
            & (matches_vs_po_teams_same_conf["away"]==sel_team),
            1,
            0
        )
        matches_vs_po_teams_same_conf["away_wins"] = np.where(
            matches_vs_po_teams_same_conf["away_goals"] > matches_vs_po_teams_same_conf["home_goals"],
            1,
            0
        )
        home_wins = matches_vs_po_teams_same_conf[matches_vs_po_teams_same_conf['home']==sel_team]["home_wins"].sum()
        away_wins = matches_vs_po_teams_same_conf[matches_vs_po_teams_same_conf['away']==sel_team]["away_wins"].sum()
        played = len(matches_vs_po_teams_same_conf)
        win_loss_pct_same_conf = round((home_wins + away_wins) / played, 3)
        # other conf
        matches_vs_po_teams_other_conf["home_wins"] = np.where(
            (matches_vs_po_teams_other_conf["home_goals"] > matches_vs_po_teams_other_conf["away_goals"])
            & (matches_vs_po_teams_other_conf["home"]==sel_team),
            1,
            0
        )
        matches_vs_po_teams_other_conf["away_wins"] = np.where(
            (matches_vs_po_teams_other_conf["away_goals"] > matches_vs_po_teams_other_conf["home_goals"])
            & (matches_vs_po_teams_other_conf["away"]==sel_team),
            1,
            0
        )
        matches_vs_po_teams_other_conf["away_wins"] = np.where(
            matches_vs_po_teams_other_conf["away_goals"] > matches_vs_po_teams_other_conf["home_goals"],
            1,
            0
        )
        home_wins = matches_vs_po_teams_other_conf[matches_vs_po_teams_other_conf['home']==sel_team]["home_wins"].sum()
        away_wins = matches_vs_po_teams_other_conf[matches_vs_po_teams_other_conf['away']==sel_team]["away_wins"].sum()
        played = len(matches_vs_po_teams_other_conf)
        if played > 0:
            win_loss_pct_other_conf = round((home_wins + away_wins) / played, 3)
        else:
            win_loss_pct_other_conf = 0
        standings.append({"team": sel_team,
                          "win_loss_pct_playoff_teams_same_conf": win_loss_pct_same_conf,
                          "win_loss_pct_playoff_teams_other_conf": win_loss_pct_other_conf})
    standings = pd.DataFrame(standings)

    return standings


def get_standings(matches_df, classif_rules, league_type=None, divisions=None):
    """
    Generate team standings and apply classification rules for league and subgroup rankings.

    This function computes team performance metrics (points, win/loss percentages, 
    goal difference, etc.) using either European-style ("UEFA") or U.S.-style 
    ("NBA", "MLB", "NFL") rules. It then applies a hierarchy of classification 
    rules to rank teams, including optional subgroup rankings (e.g., divisions, 
    conferences). Head-to-head (H2H) rules can also be applied to break ties 
    among teams tied on previous criteria.

    Args:
        matches_df (pd.DataFrame): Match-level data with at least:
            - "home": Home team identifier
            - "away": Away team identifier
            - "home_goals": Goals scored by the home team
            - "away_goals": Goals scored by the away team

        classif_rules (dict): Dictionary mapping classification levels to rules.
            Example:
                {
                    "league": ["points", "goal_difference", "goals_for"],
                    "division": ["points", "h2h_points"]
                }
            - Keys represent classification levels (e.g., "league", "division").
            - Values are ordered lists of rules for ranking.
            - Rules may include:
                * Aggregate metrics (e.g., "points", "goal_difference")
                * Head-to-head rules prefixed with "h2h_" 
                  (e.g., "h2h_points", "h2h_goal_difference")

        league_type (str, optional): Determines which metric function to use.
            - "UEFA": Uses `get_standings_metrics_footy` (soccer/football style).
            - "NBA", "MLB", "NFL": Uses `get_standings_metrics_us` (U.S. sports style).
            Defaults to None.

        divisions (pd.DataFrame, optional): Mapping of teams to subgroup identifiers.
            Required if classification rules include subgroups (e.g., "division", "conference").
            Must include:
            - "team": Team identifier
            - Columns for each subgroup level (e.g., "division", "conference")

    Returns:
        pd.DataFrame: Standings for all teams with:
            - "team": Team identifier
            - Standard performance metrics (depends on league_type)
            - Columns for each classification level position:
                * "{classif}_pos": Ranking position within that level
            - Subgroup identifiers (if applicable)

    Raises:
        ValueError: If `league_type` is not one of ["UEFA", "NBA", "MLB", "NFL"].

    Notes:
        - Classification rules are applied hierarchically:
            1. Metrics are computed for all teams.
            2. Teams are ranked using rules defined in `classif_rules`.
            3. For non-league levels (e.g., divisions), standings are recomputed 
               within each subgroup.
        - Head-to-head rules ("h2h_*") are applied only among tied teams.
    """
    if league_type == "UEFA":
        standings = get_standings_metrics_footy(matches_df)
    elif league_type in ["NBA","MLB","NFL"]:
        standings = get_standings_metrics_us(matches_df)
    else:
        raise(ValueError("Invalid league type for getting standings"))
    for classif, rules in classif_rules.items():
        if classif=="league":
            league_standings = apply_classification_rules(matches_df, rules, standings.copy())
            league_standings = league_standings[["team", "pos"]]
            league_standings = league_standings.rename(columns={"pos": f"{classif}_pos"})
            standings = standings.merge(league_standings, how="left",on="team")
        else:
            div_to_iterate = divisions[classif].unique().tolist()
            all_division_standings = []
            for div in div_to_iterate:
                division_teams = divisions[divisions[classif]==div]
                division_standings_metrics = standings[standings["team"].isin(division_teams["team"])].copy()
                division_standings = apply_classification_rules(matches_df, rules, division_standings_metrics)
                division_standings = division_standings[["team", "pos"]]
                division_standings = division_standings.rename(columns={"pos": f"{classif}_pos"})
                division_standings[classif] = div
                all_division_standings.append(division_standings)
            all_division_standings = pd.concat(all_division_standings)
            standings = standings.merge(all_division_standings, how="left",on="team")

    if league_type == "UEFA":
        standings["playoff_pos"] = standings["league_pos"]
    else:
        standings["playoff_pos"] = standings["conference"] + " " + standings["conference_pos"].astype(str)

    return standings


def apply_classification_rules(matches_df, classif_rules, standings):
    """
    Apply classification and tie-breaking rules to rank teams in a standings table.

    This function ranks teams based on a list of classification rules. It supports:
      - Standard aggregate metrics (e.g., points, goal difference).
      - Opponent-based metrics (e.g., opponent win percentage).
      - Division winner bonuses.
      - Head-to-head (H2H) tie-breakers among tied teams.
      - Playoff tie-breakers (for leagues like Serie A).

    For each rule, the function augments the standings with additional metrics if needed,
    then applies ranking logic. At the end, it assigns each team a final position (`pos`)
    after resolving ties with a fallback random tie-breaker.

    Args:
        matches_df (pd.DataFrame): Match-level data used to calculate tiebreakers.
            Must include at least:
            - "home", "away": Team identifiers
            - "home_goals", "away_goals": Match scores
        classif_rules (list of str): Ordered list of rules to rank teams.
            Supported rule types:
              - Aggregate stats already in `standings` (e.g., "points", "goal_difference").
              - "opponent_*": Adds opponent-based metrics via `get_opponents_aggregate_stats`.
              - "division_winner": Awards division leaders an extra ranking advantage.
              - "h2h_*": Applies a head-to-head tie-breaker (e.g., "h2h_points").
              - "playoff_*": Applies playoff tie-breaking rules for specific tied positions.
        standings (pd.DataFrame): Team standings with precomputed metrics.
            Must contain at least a "team" column, plus the metrics referenced in `classif_rules`.

    Returns:
        pd.DataFrame: Updated standings with:
            - All original columns
            - New columns for each applied classification rule
            - "__tiebreaker__": Random fallback column for tie resolution
            - "pos": Final team position after applying all rules

    Notes:
        - H2H rules are applied only among teams tied after previous rules.
        - Playoff rules apply to specific tied positions (e.g., 1st or relegation spots).
        - If ties remain after all rules, a deterministic fallback (`__tiebreaker__`)
          ensures all teams get a unique ranking.
        - Helper functions are used for specialized tie-breaking:
            * `apply_h2h_tiebreaker`, `apply_h2h_sweep_tiebreaker`
            * `apply_common_games_tiebreaker`
            * `apply_break_division_tiebreaker`
            * `apply_playoff_tiebreaker`
    """
    # Sort by classification rules
    for i, rule in enumerate(classif_rules):
        is_h2h_rule = rule.startswith("h2h")
        is_playoff = rule.startswith("playoff")
        is_opponent_rule = rule.startswith("opponent")

        if (is_opponent_rule) & (rule not in standings.columns.tolist()):
            opponent_stats = get_opponents_aggregate_stats(matches_df, standings)
            standings = pd.merge(standings,opponent_stats,on='team')

        elif rule == "division_winner":
            standings[rule] = np.where(standings["division_pos"] == 1, 1, 0)

        elif is_h2h_rule or is_playoff or rule == 'win_loss_pct_div_if_same_div':
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
                        if rule=="h2h_sweep_full":
                            substed_tied_standings = apply_h2h_sweep_tiebreaker(
                                matches_df, tied_teams, "full"
                            )
                        elif rule=="h2h_sweep_win_loss_pct":
                            substed_tied_standings = apply_h2h_sweep_tiebreaker(
                                matches_df, tied_teams, "win_loss_pct"
                            )
                        elif rule=="h2h_win_loss_pct_common_games":
                            substed_tied_standings = apply_common_games_tiebreaker(
                                matches_df, tied_teams
                            )
                        elif rule=="h2h_break_division_ties":
                            substed_tied_standings = apply_break_division_tiebreaker(
                                subset_of_tied
                            )
                        else:
                            substed_tied_standings = apply_h2h_tiebreaker(
                                matches_df, tied_teams, rule
                            )
                    elif is_playoff:
                        substed_tied_standings = apply_playoff_tiebreaker(
                            matches_df, tied_teams
                        )
                    elif rule == 'win_loss_pct_div_if_same_div':
                        substed_tied_standings = apply_win_loss_pct_same_div_tiebreaker(
                                subset_of_tied
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


def get_opponents_strength(matches_df, standings_df, strength_of):
    """
    Calculate a team's strength of schedule or strength of victory based on their opponents' records.

    Parameters
    ----------
    matches_df : pandas.DataFrame
        DataFrame containing match results with at least the following columns:
        - 'home': home team name
        - 'away': away team name
        - 'home_goals': goals scored by the home team
        - 'away_goals': goals scored by the away team

    standings_df : pandas.DataFrame
        DataFrame containing current standings with at least the following columns:
        - 'team': team name
        - 'wins': number of wins
        - 'ties': number of ties
        - 'played': total games played

    strength_of : str
        Determines the type of calculation:
        - 'schedule': use all opponents from games played (regardless of result).
        - 'victory': use only opponents from games the team has won.

    Returns
    -------
    pandas.DataFrame
        DataFrame with two columns:
        - 'team': team name
        - 'strength_of_<strength_of>': winning percentage of the relevant opponents,
          calculated as (total_wins + total_ties / 2) / total_played. If total_played is 0, returns 0.

    Notes
    -----
    - In 'schedule' mode, all opponents are included, regardless of match result.
    - In 'victory' mode, only opponents from matches the team has won are included.
    - The winning percentage formula counts ties as half a win.
    """
    # Get set of all opponents each team has played against
    team_opponents = {}

    if strength_of == 'schedule':
        for _, row in matches_df.iterrows():
            team_opponents.setdefault(row['home'],[]).append(row['away'])
            team_opponents.setdefault(row['away'],[]).append(row['home'])

    elif strength_of == 'victory':
        for _, row in matches_df.iterrows():
            # Home team wins
            if row['home_goals'] > row['away_goals']:
                team_opponents.setdefault(row['home'], []).append(row['away'])
            # Away team wins
            elif row['away_goals'] > row['home_goals']:
                team_opponents.setdefault(row['away'], []).append(row['home'])

    else:
        raise(ValueError("Must be strength of schedule or victory"))

    wins_lookup = standings_df.set_index("team")["wins"].to_dict()
    ties_lookup = standings_df.set_index("team")["ties"].to_dict()
    played_lookup = standings_df.set_index("team")["played"].to_dict()

    # Build result
    result = []
    for team, opponents in team_opponents.items():
        total_wins = sum(wins_lookup.get(opp, 0) for opp in opponents)
        total_ties = sum(ties_lookup.get(opp, 0) for opp in opponents)
        total_played = sum(played_lookup.get(opp, 0) for opp in opponents)
        strength_calc = (total_wins + total_ties / 2) / total_played if total_played > 0 else 0
        result.append({
            "team": team,
            f"strength_of_{strength_of}": round(strength_calc, 3)
        })

    return pd.DataFrame(result)


def validate_bracket(bracket_df, knockout_format):
    """
    Validates a playoff bracket DataFrame.

    Checks for:
    - Missing or empty team slots
    - Duplicate team entries (excluding 'Bye')
    - Total number of slots being a power of 2
    - At least 2 non-'Bye' teams

    Args:
        bracket_df (pd.DataFrame): A DataFrame with columns ['team1', 'team2'] representing matchups.
        knockout_format (dict): Dictionary defining the format of each round in the knockout stage.
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
    if len(knockout_format) != expected_rounds:
        raise ValueError(
            f"Bracket format does not match the number of rounds. Expected {expected_rounds} rounds, got {len(knockout_format)}."
        )


def simulate_playoff_bracket(bracket_df, knockout_format, elos, playoff_schedule, has_reseeding):
    """
    Simulates a knockout playoff bracket using ELO ratings.

    Args:
        bracket_df: Bracket structure with columns ['team1', 'team2'],
            and additional columns ['seed1','seed2'] in case the bracket has re-seeding
        knockout_format: Dictionary defining the format of each round
        elos: DataFrame with columns ['team', 'elo'] representing team ELO ratings
        playoff_schedule: DataFrame with pending matches to simulate
        has_reseeding (boolean): True/False if playoff has re-seeding after each round

    Returns:
        Wide-format DataFrame with one row per team and binary indicators for each round
    """
    validate_bracket(bracket_df, knockout_format)
    elos_dict = dict(zip(elos["team"], elos["elo"]))
    teams_progression = {}
    rounds = []

    current_round = bracket_df.copy()

    while len(current_round) > 0:
        round_label = f"po_r{2 * len(current_round)}"
        round_format = knockout_format[round_label]
        rounds.append(round_label)

        winners = _simulate_round(
            current_round,
            round_format,
            elos_dict,
            playoff_schedule,
            teams_progression,
            round_label,
        )

        current_round = _prepare_next_round(winners, bracket_df, has_reseeding)

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

    tie_matches = _get_tie_matches(team1, team2, playoff_schedule)

    if tie_matches.empty:
        if round_format in ('single_game_neutral', 'single_game', 'two-legged'):
            win_proba = calculate_win_probability(
                team1_elo, team2_elo, matchup_type=round_format
            )
            result = simulate_match_winner(win_proba)
        elif round_format in ('best_of_3', 'best_of_5', 'best_of_7'):
            best_of_num = int("".join(filter(str.isdigit, round_format)))
            result = simulate_series_winner(team1_elo, team2_elo, best_of=best_of_num)
        else:
            raise ValueError(
                "Invalid playoff matchup_type"
            )
        return team1 if result == 1 else team2

    win_proba = calculate_win_probability(
        team1_elo, team2_elo, matchup_type=round_format
    )
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
        result = simulate_match_winner(win_proba)
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


def _prepare_next_round(winners, bracket_df, has_reseeding):
    """
    Pair up winners to create matchups for the next round.

    Args:
        winners (list): List of team names who won their previous matches.
        bracket_df( pd.DataFrame): Optional: original bracket for playoffs
            Used when has_reseeding is True for re-seeding purposes
        has_reseeding (boolean): True/False if playoff has re-seeding after each round

    Returns:
        pd.DataFrame: DataFrame with columns 'team1' and 'team2' for next round matchups.
    """
    if len(winners) < 2:
        return pd.DataFrame()

    if has_reseeding and len(winners) > 2:
        next_round_pairs = _playoff_round_reseeding(winners, bracket_df)
        next_round_pairs = next_round_pairs[["team1", "team2"]]
    else:
        it = iter(winners)
        next_round_pairs = list(zip(it, it))
        next_round_pairs = pd.DataFrame(next_round_pairs, columns=["team1", "team2"])

    return next_round_pairs


def _playoff_round_reseeding(winners, bracket_df):
    """
    Reseed all teams that have advanced to the next playoff rpund
    Create the matchups for the following round.

    Args:
        winners (list): List of team names who won their previous matches.
        bracket_df( pd.DataFrame): Optional: original bracket for playoffs
            Used when has_reseeding is True for re-seeding purposes

    Returns:
        pd.DataFrame: DataFrame with columns 'team1' and 'team2' for next round matchups.
    """
    team_seeds = pd.concat([
        bracket_df[["team1", "seed1"]].rename(columns={"team1": "team", "seed1": "seed"}),
        bracket_df[["team2", "seed2"]].rename(columns={"team2": "team", "seed2": "seed"})
    ])
    team_seeds = team_seeds[team_seeds["team"].isin(winners)]
    team_seeds[["league", "seed"]] = team_seeds["seed"].str.split(" ", n=1, expand=True)
    team_seeds["seed"] = team_seeds["seed"].astype(int)
    team_seeds["round_seed"] = team_seeds.groupby("league")["seed"].rank(method="first")

    matchups = []

    for league, group in team_seeds.groupby("league"):
        group_sorted = group.sort_values("round_seed")  # ascending: best → worst
        teams = group_sorted["team"].tolist()
        seeds = group_sorted["round_seed"].tolist()
        
        # Pair first with last, etc.
        for i in range(len(teams) // 2):
            matchup = {
                "league": league,
                "team1": teams[i],
                "seed1": seeds[i],
                "team2": teams[-(i+1)],
                "seed2": seeds[-(i+1)]
            }
            matchups.append(matchup)

    return pd.DataFrame(matchups)
    

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
        df (pd.DataFrame): DataFrame with columns ['team', 'league_pos'] where 'league_pos' determines pot grouping.
        pot_size (int): Number of positions per pot (default is 2).

    Returns:
        pd.DataFrame: A DataFrame with columns ['draw_order', 'team'] indicating the randomized draw result.
    """
    df = df.copy()
    df = df.sort_values("league_pos").reset_index(drop=True)

    # Map position → team
    pos_to_team = dict(zip(df["league_pos"], df["team"]))

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


def create_bracket_from_composition(df_with_draw, knockout_bracket):
    """
    Creates a playoff bracket based on a predefined composition and a team draw.

    Args:
        df_with_draw (pd.DataFrame): DataFrame with columns ['draw_order', 'team'] from draw.
        knockout_bracket (list of tuple): List of (pos1, pos2) tuples representing matchups.
            Values can be integers (draw positions) or 'Bye'.

    Returns:
        pd.DataFrame: A DataFrame with columns ['team1', 'team2'] representing the bracket.

    Raises:
        ValueError: If both sides of a match are 'Bye'.
    """
    pos_to_team = dict(zip(df_with_draw["draw_order"], df_with_draw["team"]))
    pairs = []

    for pos1, pos2 in knockout_bracket:
        team1 = pos_to_team.get(pos1) if pos1 != "Bye" else "Bye"
        team2 = pos_to_team.get(pos2) if pos2 != "Bye" else "Bye"

        if team1 == "Bye" and team2 == "Bye":
            raise ValueError("Invalid bracket: both sides cannot be 'Bye'")

        pairs.append((team1, team2, pos1, pos2))

    return pd.DataFrame(pairs, columns=["team1", "team2", "seed1", "seed2"])
