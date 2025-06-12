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
fixtures_config = {
    "ENG": {
        "fixtures_url": "https://fbref.com/en/comps/9/schedule/Premier-League-Scores-and-Fixtures",
        "table_id": "sched_2024-2025_9_1",
    },
    "ESP": {
        "fixtures_url": "https://fbref.com/en/comps/12/schedule/La-Liga-Scores-and-Fixtures",
        "table_id": "sched_2024-2025_12_1",
    },
    "ITA": {
        "fixtures_url": "https://fbref.com/en/comps/11/schedule/Serie-A-Scores-and-Fixtures",
        "table_id": "sched_2024-2025_11_1",
    },
    "GER": {
        "fixtures_url": "https://fbref.com/en/comps/20/schedule/Bundesliga-Scores-and-Fixtures",
        "table_id": "sched_2024-2025_20_1",
    },
    "FRA": {
        "fixtures_url": "https://fbref.com/en/comps/13/schedule/Ligue-1-Scores-and-Fixtures",
        "table_id": "sched_2024-2025_13_1",
    },
}

leagues_to_sim = list(fixtures_config.keys())
