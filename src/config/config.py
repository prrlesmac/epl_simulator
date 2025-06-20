import datetime

# mapping from club elo to fb ref
club_name_mapping = {

    # ENG        
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

    # ESP
    "Barcelona": "Barcelona",
    "Real Madrid": "Real Madrid",
    "Atletico": "Atlético Madrid",
    "Bilbao": "Athletic Club",
    "Villarreal": "Villarreal",
    "Betis": "Betis",
    "Osasuna": "Osasuna",
    "Celta": "Celta Vigo",
    "Valencia": "Valencia",
    "Sociedad": "Real Sociedad",
    "Rayo Vallecano": "Rayo Vallecano",
    "Alaves": "Alavés",
    "Mallorca": "Mallorca",
    "Sevilla": "Sevilla",
    "Girona": "Girona",
    "Espanyol": "Espanyol",
    "Getafe": "Getafe",
    "Leganes": "Leganés",
    "Las Palmas": "Las Palmas",
    "Valladolid": "Valladolid",

    ## ITA
    "Inter": "Inter",
    "Atalanta": "Atalanta",
    "Napoli": "Napoli",
    "Roma": "Roma",
    "Juventus": "Juventus",
    "Milan": "Milan",
    "Lazio": "Lazio",
    "Fiorentina": "Fiorentina",
    "Bologna": "Bologna",
    "Torino": "Torino",
    "Genoa": "Genoa",
    "Como": "Como",
    "Udinese": "Udinese",
    "Parma": "Parma",
    "Cagliari": "Cagliari",
    "Verona": "Hellas Verona",
    "Lecce": "Lecce",
    "Empoli": "Empoli",
    "Venezia": "Venezia",
    "Monza": "Monza",

    ## GER
    "Bayern": "Bayern Munich",
    "Leverkusen": "Leverkusen",
    "Dortmund": "Dortmund",
    "Frankfurt": "Eint Frankfurt",
    "Stuttgart": "Stuttgart",
    "RB Leipzig": "RB Leipzig",
    "Mainz": "Mainz 05",
    "Freiburg": "Freiburg",
    "Werder": "Werder Bremen",
    "Wolfsburg": "Wolfsburg",
    "Gladbach": "Gladbach",
    "Augsburg": "Augsburg",
    "Union Berlin": "Union Berlin",
    "Hoffenheim": "Hoffenheim",
    "St Pauli": "St. Pauli",
    "Heidenheim": "Heidenheim",
    "Bochum": "Bochum",
    "Holstein": "Holstein Kiel",

    ## FRA
    "Paris SG": "Paris S-G",
    "Lille": "Lille",
    "Monaco": "Monaco",
    "Marseille": "Marseille",
    "Lyon": "Lyon",
    "Nice": "Nice",
    "Strasbourg": "Strasbourg",
    "Lens": "Lens",
    "Brest": "Brest",
    "Rennes": "Rennes",
    "Toulouse": "Toulouse",
    "Auxerre": "Auxerre",
    "Reims": "Reims",
    "Nantes": "Nantes",
    "Le Havre": "Le Havre",
    "Angers": "Angers",
    "Saint-Etienne": "Saint-Étienne",
    "Montpellier": "Montpellier",
}
number_of_simulations = 1000

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

classification = {
    "ENG": [
        "points",
        "goal_difference",
        "goals_for",
        "h2h_points",
        "h2h_away_goals_for"
    ],
    "ESP": [
        "points",
        "h2h_points",
        "h2h_goal_difference",
        "goal_difference",
        "goals_for"
    ],
    "ITA": [
        "points",
        "playoff",
        "h2h_points",
        "h2h_goal_difference",
        "goal_difference",
        "goals_for",
    ],
    "GER": [
        "points",
        "goal_difference",
        "goals_for",
        "h2h_points",
        "h2h_goal_difference",
        "h2h_away_goals_for",
        "away_goals_for"
    ],
    "FRA": [
        "points",
        "goal_difference",
        "h2h_points",
        "h2h_goal_difference",
        "h2h_goals_for",
        "h2h_away_goals_for",
        "goals_for",
        "away_goals_for"
    ],
}

relegation = {
    "ENG": {
        "direct": [18, 19, 20],
        "playoff": None
    },
    "ESP": {
        "direct": [18, 19, 20],
        "playoff": None
    },
    "ITA": {
        "direct": [18, 19, 20],
        "playoff": None
    },
    "GER": {
        "direct": [17, 18],
        "playoff": [16]
    },
    "FRA": {
        "direct": [17, 18],
        "playoff": [16]
    },
}
