import datetime

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
number_of_simulations = 10000

elo_table = "current_elos"
fixtures_table = "fixtures"
sim_output_table = "sim_standings"

elo_date = (datetime.date.today() + datetime.timedelta(days=1)).strftime("%Y-%m-%d") 
elo_rating_url = f"http://api.clubelo.com/{elo_date}"
fixtures_url = (
    "https://fbref.com/en/comps/9/schedule/Premier-League-Scores-and-Fixtures"
)
