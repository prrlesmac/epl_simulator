club_name_mapping = {
    "Southampton": "Southampton",
    "Arsenal": "Arsenal",
    "Liverpool": "Liverpool",
    "Crystal Palace": "Crystal Palace",
    "Ipswich": "Ipswich Town",
    "West Ham": "West Ham",
    "Bournemouth": "Bournemouth",
    "Leicester": "Leicester City",
    "Newcastle": "Newcastle Utd",
    "Everton": "Everton",
    "Fulham": "Fulham",
    "Man City": "Manchester City",
    "Forest": "Nott'ham Forest",
    "Chelsea": "Chelsea",
    "Man United": "Manchester Utd",
    "Aston Villa": "Aston Villa",
    "Wolves": "Wolves",
    "Brentford": "Brentford",
    "Tottenham": "Tottenham",
    "Brighton": "Brighton",
}
number_of_simulations = 1000

elo_output_file = "data/02_intermediate/current_elo_ratings.csv"
fixtures_output_file = "data/01_raw/epl_matches.csv"
sim_output_file = "data/03_output/season_standings_sim.csv"

elo_rating_url = "http://api.clubelo.com/2025-04-01"
fixtures_url = (
    "https://fbref.com/en/comps/9/schedule/Premier-League-Scores-and-Fixtures"
)
