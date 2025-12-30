import datetime
from sqlalchemy.dialects.postgresql import VARCHAR, INTEGER, FLOAT, TIMESTAMP, DATE

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
    "Leeds": "Leeds United",
    "Burnley": "Burnley",
    "Sunderland": "Sunderland",
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
    "Levante": "Levante",
    "Elche": "Elche",
    "Oviedo": "Oviedo",
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
    "Sassuolo": "Sassuolo",
    "Pisa": "Pisa",
    "Cremonese": "Cremonese",
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
    "Koeln": "Köln",
    "Hamburg": "Hamburger SV",
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
    "Lorient": "Lorient",
    "Paris FC": "Paris FC",
    "Metz": "Metz",
    ## UCL
    "Feyenoord": "Feyenoord",
    "Salzburg": "RB Salzburg",
    "Benfica": "Benfica",
    "Crvena Zvezda": "Red Star",
    "Celtic": "Celtic",
    "PSV": "PSV Eindhoven",
    "Brugge": "Club Brugge",
    "Sporting": "Sporting CP",
    "Dinamo Zagreb": "Dinamo Zagreb",
    "Young Boys": "Young Boys",
    "Shakhtar": "Shakhtar",
    "Sparta Praha": "Sparta Prague",
    "Sturm Graz": "Sturm Graz",
    ## UEL
    "FK Riga": "FK Rīgas FS",
    "Fenerbahce": "Fenerbahçe",
    "Bodoe Glimt": "Bodø/Glimt",
    "PAOK": "PAOK",
    "Braga": "Braga",
    "Alkmaar": "AZ Alkmaar",
    "Elfsborg": "Elfsborg",
    "Porto": "Porto",
    "Twente": "Twente",
    "Midtjylland": "Midtjylland",
    "Hoffenheim": "Hoffenheim",
    "Ajax": "Ajax",
    "M Tel Aviv": "Maccabi Tel Aviv",
    "Anderlecht": "Anderlecht",
    "Dynamo Kyiv": "Dynamo Kyiv",
    "Karabakh Agdam": "Qarabağ",
    "Malmoe": "Malmö",
    "Viktoria Plzen": "Viktoria Plzeň",
    "Rangers": "Rangers",
    "Ferencvaros": "Ferencváros",
    "Besiktas": "Beşiktaş",
    "Slavia Praha": "Slavia Prague",
    "Olympiakos": "Olympiacos",
    "Steaua": "FCSB",
    "Galatasaray": "Galatasaray",
    "St Gillis": "Union SG",
    "Razgrad": "Ludogorets",
    # UECL
    "FK Astana": "Astana FK",
    "HJK Helsinki": "HJK",
    "StGallen": "St. Gallen",
    "Rapid Wien": "Rapid Wien",
    "Backa Topola": "TSC",
    "Mlada Boleslav": "Mladá Boleslav",
    "LASK": "LASK",
    "Noah": "FC Noah",
    "Bueyueksehir": "Başakşehir",
    "The New Saints": "The New Saints",
    "Omonia": "AC Omonia",
    "APOEL": "APOEL",
    "Olimpija Ljubljana": "Olimpija",
    "Legia": "Legia Warsaw",
    "Larne": "Larne FC",
    "Paphos": "Pafos FC",
    "Lugano": "Lugano",
    "Shamrock": "Shamrock Rovers",
    "Dinamo Minsk": "Dinamo Minsk",
    "Hearts": "Hearts",
    "Cercle Brugge": "Cercle Brugge",
    "Molde": "Molde",
    "Gent": "Gent",
    "Djurgarden": "Djurgården",
    "Petrocub": "Petrocub",
    "Celje": "NK Celje",
    "FC Kobenhavn": "FC Copenhagen",
    "Panathinaikos": "Panathinaikos",
    "Jagiellonia": "Jagiellonia",
    "Borac Banja Luka": "Borac Banja Luka",
    "Guimaraes": "Vitória",
    "Vikingur": "KV",
    "Kairat": "Qaırat Almaty",
    "Lausanne": "Lausanne-Sport",
    "Shkendija": "Shkëndija 79",
    "Craiova": "CS U Craiova",
    "Kuopio": "KuPS",
    "Lincoln": "Red Imps",
    "Drita": "KF Drita",
    "Shelbourne": "Shelbourne FC",
    "Hamrun": "Ħamrun Spartans FC",
    "Lech": "Lech Poznań",
    "Haecken": "Häcken",
    "Larnaca": "AÉK Lárnaka",
    "Breidablik": "Breiðablik",
    "Rakow": "Raków",
    "AEK": "AEK Athens",
    "Go Ahead Eagles": "Go Ahead Eag",
}

# NFL franchise name remaps
nfl_name_remap = {
    # Arizona Cardinals franchise
    "Chicago Cardinals": "Arizona Cardinals",
    "St. Louis Cardinals": "Arizona Cardinals",
    "Phoenix Cardinals": "Arizona Cardinals",
    "Arizona Cardinals": "Arizona Cardinals",

    # Atlanta Falcons
    "Atlanta Falcons": "Atlanta Falcons",

    # Baltimore Ravens
    "Baltimore Ravens": "Baltimore Ravens",

    # Buffalo Bills
    "Buffalo Bills": "Buffalo Bills",

    # Carolina Panthers
    "Carolina Panthers": "Carolina Panthers",

    # Chicago Bears
    "Decatur Staleys": "Chicago Bears",
    "Chicago Staleys": "Chicago Bears",
    "Chicago Bears": "Chicago Bears",

    # Cincinnati Bengals
    "Cincinnati Bengals": "Cincinnati Bengals",

    # Cleveland Browns
    "Cleveland Browns": "Cleveland Browns",

    # Dallas Cowboys
    "Dallas Cowboys": "Dallas Cowboys",

    # Denver Broncos
    "Denver Broncos": "Denver Broncos",

    # Detroit Lions
    "Portsmouth Spartans": "Detroit Lions",
    "Detroit Lions": "Detroit Lions",

    # Green Bay Packers
    "Green Bay Packers": "Green Bay Packers",

    # Houston Texans
    "Houston Texans": "Houston Texans",

    # Indianapolis Colts franchise
    "Baltimore Colts": "Indianapolis Colts",
    "Indianapolis Colts": "Indianapolis Colts",

    # Jacksonville Jaguars
    "Jacksonville Jaguars": "Jacksonville Jaguars",

    # Kansas City Chiefs franchise
    "Dallas Texans": "Kansas City Chiefs",
    "Kansas City Chiefs": "Kansas City Chiefs",

    # Las Vegas Raiders franchise
    "Oakland Raiders": "Las Vegas Raiders",
    "Los Angeles Raiders": "Las Vegas Raiders",
    "Las Vegas Raiders": "Las Vegas Raiders",

    # Los Angeles Chargers
    "San Diego Chargers": "Los Angeles Chargers",
    "Los Angeles Chargers": "Los Angeles Chargers",

    # Los Angeles Rams franchise
    "Cleveland Rams": "Los Angeles Rams",
    "Los Angeles Rams": "Los Angeles Rams",
    "St. Louis Rams": "Los Angeles Rams",

    # Miami Dolphins
    "Miami Dolphins": "Miami Dolphins",

    # Minnesota Vikings
    "Minnesota Vikings": "Minnesota Vikings",

    # New England Patriots franchise
    "Boston Patriots": "New England Patriots",
    "New England Patriots": "New England Patriots",

    # New Orleans Saints
    "New Orleans Saints": "New Orleans Saints",

    # New York Giants
    "New York Giants": "New York Giants",

    # New York Jets franchise
    "New York Titans": "New York Jets",
    "New York Jets": "New York Jets",

    # Philadelphia Eagles
    "Philadelphia Eagles": "Philadelphia Eagles",

    # Pittsburgh Steelers
    "Pittsburgh Pirates": "Pittsburgh Steelers",  # early name
    "Pittsburgh Steelers": "Pittsburgh Steelers",

    # San Francisco 49ers
    "San Francisco 49ers": "San Francisco 49ers",

    # Seattle Seahawks
    "Seattle Seahawks": "Seattle Seahawks",

    # Tampa Bay Buccaneers
    "Tampa Bay Buccaneers": "Tampa Bay Buccaneers",

    # Tennessee Titans franchise
    "Houston Oilers": "Tennessee Titans",
    "Tennessee Oilers": "Tennessee Titans",
    "Tennessee Titans": "Tennessee Titans",

    # Washington franchise
    "Boston Braves": "Washington Commanders",
    "Boston Redskins": "Washington Commanders",
    "Washington Redskins": "Washington Commanders",
    "Washington Football Team": "Washington Commanders",
    "Washington Commanders": "Washington Commanders",
}

nfl_expansion_elos = {

    "Baltimore Ravens": 1500.0,
    "Carolina Panthers": 1550.0,
  #  "Cleveland Browns": 1350.0,
    "Houston Texans": 1450.0,
    "Jacksonville Jaguars": 1450.0,
    "Tampa Bay Buccaneers": 1350.0,
    "Seattle Seahawks": 1400.0

}

# Database
# league type to db table mapping
db_table_mapping = {
    "UEFA_LOCAL": "domestic_sim_output_table",
    "UEFA_CONTINENTAL": "continental_sim_output_table",
    "NFL": "nfl_sim_output_table",
    "MLB": "mlb_sim_output_table",
    "NBA": "nba_sim_output_table",
}

db_table_definitions = {
    "elo_table": {
        "name": "current_elos",
        "dtype": {
            "club": VARCHAR(100),
            "country": VARCHAR(100),
            "level": INTEGER(),
            "elo": FLOAT(),
            "updated_at": TIMESTAMP(),
        },
    },
    "historic_elo_table": {
        "name": "historic_elos",
        "dtype": {
            "home": VARCHAR(100),
            "away": VARCHAR(100),
            "home_goals": INTEGER(),
            "away_goals": INTEGER(),
            "played": VARCHAR(10),
            "neutral": VARCHAR(10),
            "round": VARCHAR(100),
            "date": DATE(),
            "round": VARCHAR(100),
            "notes": VARCHAR(255),
            "country": VARCHAR(100),
            "result": VARCHAR(3),
            "home_elo_before": FLOAT(),
            "away_elo_before": FLOAT(),
            "home_elo_after": FLOAT(),
            "away_elo_after": FLOAT(),
            "home_win_expectancy": FLOAT(),
            "away_win_expectancy": FLOAT(),
            "updated_at": TIMESTAMP(),
        },
    },
    "fixtures_table": {
        "name": "fixtures",
        "dtype": {
            "home": VARCHAR(100),
            "away": VARCHAR(100),
            "home_goals": INTEGER(),
            "away_goals": INTEGER(),
            "played": VARCHAR(10),
            "neutral": VARCHAR(10),
            "country": VARCHAR(100),
            "date": DATE(),
            "season": VARCHAR(10),
            "round": VARCHAR(100),
            "notes": VARCHAR(255),
            "updated_at": TIMESTAMP(),
        },
    },
    "domestic_sim_output_table": {
        "name": "sim_standings_dom",
        "dtype": {
            "team": VARCHAR(100),
            "1": FLOAT(),
            "2": FLOAT(),
            "3": FLOAT(),
            "4": FLOAT(),
            "5": FLOAT(),
            "6": FLOAT(),
            "7": FLOAT(),
            "8": FLOAT(),
            "9": FLOAT(),
            "10": FLOAT(),
            "11": FLOAT(),
            "12": FLOAT(),
            "13": FLOAT(),
            "14": FLOAT(),
            "15": FLOAT(),
            "16": FLOAT(),
            "17": FLOAT(),
            "18": FLOAT(),
            "19": FLOAT(),
            "20": FLOAT(),
            "champion": FLOAT(),
            "top_4": FLOAT(),
            "relegation_direct": FLOAT(),
            "relegation_playoff": FLOAT(),
            "league": VARCHAR(100),
            "updated_at": TIMESTAMP(),
        },
    },
    "continental_sim_output_table": {
        "name": "sim_standings_con",
        "dtype": {
            "team": VARCHAR(100),
            "1": FLOAT(),
            "2": FLOAT(),
            "3": FLOAT(),
            "4": FLOAT(),
            "5": FLOAT(),
            "6": FLOAT(),
            "7": FLOAT(),
            "8": FLOAT(),
            "9": FLOAT(),
            "10": FLOAT(),
            "11": FLOAT(),
            "12": FLOAT(),
            "13": FLOAT(),
            "14": FLOAT(),
            "15": FLOAT(),
            "16": FLOAT(),
            "17": FLOAT(),
            "18": FLOAT(),
            "19": FLOAT(),
            "20": FLOAT(),
            "21": FLOAT(),
            "22": FLOAT(),
            "23": FLOAT(),
            "24": FLOAT(),
            "25": FLOAT(),
            "26": FLOAT(),
            "27": FLOAT(),
            "28": FLOAT(),
            "29": FLOAT(),
            "30": FLOAT(),
            "31": FLOAT(),
            "32": FLOAT(),
            "33": FLOAT(),
            "34": FLOAT(),
            "35": FLOAT(),
            "36": FLOAT(),
            "playoff": FLOAT(),
            "direct_to_round_of_16": FLOAT(),
            "po_r32": FLOAT(),
            "po_r16": FLOAT(),
            "po_r8": FLOAT(),
            "po_r4": FLOAT(),
            "po_r2": FLOAT(),
            "po_champion": FLOAT(),
            "league": VARCHAR(100),
            "updated_at": TIMESTAMP(),
        },
    },
    "nfl_sim_output_table": {
        "name": "sim_standings_nfl",
        "dtype": {
            "team": VARCHAR(100),
            "AFC 1": FLOAT(),
            "AFC 2": FLOAT(),
            "AFC 3": FLOAT(),
            "AFC 4": FLOAT(),
            "AFC 5": FLOAT(),
            "AFC 6": FLOAT(),
            "AFC 7": FLOAT(),
            "AFC 8": FLOAT(),
            "AFC 9": FLOAT(),
            "AFC 10": FLOAT(),
            "AFC 11": FLOAT(),
            "AFC 12": FLOAT(),
            "AFC 13": FLOAT(),
            "AFC 14": FLOAT(),
            "AFC 15": FLOAT(),
            "AFC 16": FLOAT(),
            "NFC 1": FLOAT(),
            "NFC 2": FLOAT(),
            "NFC 3": FLOAT(),
            "NFC 4": FLOAT(),
            "NFC 5": FLOAT(),
            "NFC 6": FLOAT(),
            "NFC 7": FLOAT(),
            "NFC 8": FLOAT(),
            "NFC 9": FLOAT(),
            "NFC 10": FLOAT(),
            "NFC 11": FLOAT(),
            "NFC 12": FLOAT(),
            "NFC 13": FLOAT(),
            "NFC 14": FLOAT(),
            "NFC 15": FLOAT(),
            "NFC 16": FLOAT(),
            "po_r16": FLOAT(),
            "po_r8": FLOAT(),
            "po_r4": FLOAT(),
            "po_r2": FLOAT(),
            "po_champion": FLOAT(),
            "playoff": FLOAT(),
            "first_round_bye": FLOAT(),
            "updated_at": TIMESTAMP(),
        },
    },
    "nba_sim_output_table": {
        "name": "sim_standings_nba",
        "dtype": {
            "team": VARCHAR(100),
            "Eastern 1": FLOAT(),
            "Eastern 2": FLOAT(),
            "Eastern 3": FLOAT(),
            "Eastern 4": FLOAT(),
            "Eastern 5": FLOAT(),
            "Eastern 6": FLOAT(),
            "Eastern 7": FLOAT(),
            "Eastern 8": FLOAT(),
            "Eastern 9": FLOAT(),
            "Eastern 10": FLOAT(),
            "Eastern 11": FLOAT(),
            "Eastern 12": FLOAT(),
            "Eastern 13": FLOAT(),
            "Eastern 14": FLOAT(),
            "Eastern 15": FLOAT(),
            "Eastern 16": FLOAT(),
            "Western 1": FLOAT(),
            "Western 2": FLOAT(),
            "Western 3": FLOAT(),
            "Western 4": FLOAT(),
            "Western 5": FLOAT(),
            "Western 6": FLOAT(),
            "Western 7": FLOAT(),
            "Western 8": FLOAT(),
            "Western 9": FLOAT(),
            "Western 10": FLOAT(),
            "Western 11": FLOAT(),
            "Western 12": FLOAT(),
            "Western 13": FLOAT(),
            "Western 14": FLOAT(),
            "Western 15": FLOAT(),
            "Western 16": FLOAT(),
            "po_play_in": FLOAT(),
            "po_r16": FLOAT(),
            "po_r8": FLOAT(),
            "po_r4": FLOAT(),
            "po_r2": FLOAT(),
            "po_champion": FLOAT(),
            "playoff": FLOAT(),
            "updated_at": TIMESTAMP(),
        },
    },
    "divisions_table": {
        "name": "teams",
        "dtype": {
            "team": VARCHAR(50),
            "division": VARCHAR(50),
            "conference": VARCHAR(50),
        },
    },
}

# Data scraping
parsing_method = "http_request" # must be local_file, http_request, or playwright
elo_date = (datetime.date.today() + datetime.timedelta(days=1)).strftime("%Y-%m-%d")
#elo_date = "2025-06-01"  # For testing purposes, set a fixed date
elo_rating_url = f"http://api.clubelo.com/{elo_date}"
pull_fixture_history = False
fixtures_config = {
    "ENG": {
        "fixtures_url": ["https://fbref.com/en/comps/9/2025-2026/schedule/2025-2026-Premier-League-Scores-and-Fixtures"],
        "local_file_path": "data/uefa/Premier League Scores & Fixtures _ FBref.com.html",
        "table_id": ["sched_2025-2026_9_1"],
    },
    "ESP": {
        "fixtures_url": ["https://fbref.com/en/comps/12/2025-2026/schedule/2025-2026-La-Liga-Scores-and-Fixtures"],
        "local_file_path": "data/uefa/La Liga Scores & Fixtures _ FBref.com.html",
        "table_id": ["sched_2025-2026_12_1"],
    },
    "ITA": {
        "fixtures_url": ["https://fbref.com/en/comps/11/2025-2026/schedule/2025-2026-Serie-A-Scores-and-Fixtures"],
        "local_file_path": "data/uefa/Serie A Scores & Fixtures _ FBref.com.html",
        "table_id": ["sched_2025-2026_11_1"],
    },
    "GER": {
        "fixtures_url": ["https://fbref.com/en/comps/20/2025-2026/schedule/2025-2026-Bundesliga-Scores-and-Fixtures"],
        "local_file_path": "data/uefa/Bundesliga Scores & Fixtures _ FBref.com.html",
        "table_id": ["sched_2025-2026_20_1"],
    },
    "FRA": {
        "fixtures_url": ["https://fbref.com/en/comps/13/2025-2026/schedule/2025-2026-Ligue-1-Scores-and-Fixtures"],
        "local_file_path": "data/uefa/Ligue 1 Scores & Fixtures _ FBref.com.html",
        "table_id": ["sched_2025-2026_13_1"],
    },
    "UCL": {
        "fixtures_url": ["https://fbref.com/en/comps/8/schedule/Champions-League-Scores-and-Fixtures"],
        "local_file_path": "data/uefa/Champions League Scores & Fixtures _ FBref.com.html",
        "table_id": ["sched_2025-2026_8_2", "sched_2025-2026_8_3"],
    },
    "UEL": {
        "fixtures_url": ["https://fbref.com/en/comps/19/schedule/Europa-League-Scores-and-Fixtures"],
        "local_file_path": "data/uefa/Europa League Scores & Fixtures _ FBref.com.html",
        "table_id": ["sched_2025-2026_19_2", "sched_2025-2026_19_3"],
    },
    "UECL": {
        "fixtures_url": ["https://fbref.com/en/comps/882/schedule/Conference-League-Scores-and-Fixtures"],
        "local_file_path": "data/uefa/Conference League Scores & Fixtures _ FBref.com.html",
        "table_id": ["sched_2025-2026_882_2", "sched_2025-2026_882_3"],
    },
    "NFL": {
        "fixtures_url": ["https://www.pro-football-reference.com/years/2025/games.htm"],
        "table_id": ["games"],
    },
    "NBA": {
        "fixtures_url": [
            "https://www.basketball-reference.com/leagues/NBA_2026_games-october.html",
            "https://www.basketball-reference.com/leagues/NBA_2026_games-november.html",
            "https://www.basketball-reference.com/leagues/NBA_2026_games-december.html",
            "https://www.basketball-reference.com/leagues/NBA_2026_games-january.html",
            "https://www.basketball-reference.com/leagues/NBA_2026_games-february.html",
            "https://www.basketball-reference.com/leagues/NBA_2026_games-march.html",
            "https://www.basketball-reference.com/leagues/NBA_2026_games-april.html",          
            ],
        "table_id": ["schedule"],
    },
    "MLB": {
        "fixtures_url": ["https://www.baseball-reference.com/leagues/majors/2024-schedule.shtml"],
        "table_id": ["games"],
    }
}

fixtures_history_config = {
    "NFL": {
        "fixtures_url": [
            f"https://www.pro-football-reference.com/years/{year}/games.htm"
            for year in range(1970, 2025)
        ],
        "table_id": ["games"],
    },
    "NBA": {
        "fixtures_url": [
            f"https://www.basketball-reference.com/leagues/NBA_{year}_games-{month}.html"
            for year in range(2022, 2026)
            for month in ["october", "november", "december", "january", "february", "march", "april", "may", "june"]
        ],
        "table_id": ["schedule"],
    },
}

## Simulation
number_of_simulations = 10000
active_uefa_leagues = ["ENG","ESP","ITA","GER","FRA","UCL","UEL","UECL"]
played_cutoff_date = None
schedule_cutoff_date = None

league_rules = {
    "ENG": {
        "sim_type": "goals",
        "home_advantage": 80,
        "has_knockout": False,
        "classification": {
            "league": [
                "points",
                "goal_difference",
                "goals_for",
                "h2h_points",
                "h2h_away_goals_for",
            ]
        },
        "qualification": {
            "champion": [1],
            "top_4": [1, 2, 3, 4],
            "relegation_direct": [18, 19, 20],
        },
    },
    "ESP": {
        "sim_type": "goals",
        "home_advantage": 80,
        "has_knockout": False,
        "classification": {
            "league": [
                "points",
                "h2h_points",
                "h2h_goal_difference",
                "goal_difference",
                "goals_for",
            ],
        },
        "qualification": {
            "champion": [1],
            "top_4": [1, 2, 3, 4],
            "relegation_direct": [18, 19, 20],
        },
    },
    "ITA": {
        "sim_type": "goals",
        "home_advantage": 80,
        "has_knockout": False,
        "classification": {
            "league": [
                "points",
                "playoff",
                "h2h_points",
                "h2h_goal_difference",
                "goal_difference",
                "goals_for",
            ],
        },
        "qualification": {
            "champion": [1],
            "top_4": [1, 2, 3, 4],
            "relegation_direct": [18, 19, 20],
        },
    },
    "GER": {
        "sim_type": "goals",
        "home_advantage": 80,
        "has_knockout": False,
        "classification": {
            "league": [
                "points",
                "goal_difference",
                "goals_for",
                "h2h_points",
                "h2h_goal_difference",
                "h2h_away_goals_for",
                "away_goals_for",
            ],
        },
        "qualification": {
            "champion": [1],
            "top_4": [1, 2, 3, 4],
            "relegation_playoff": [16],
            "relegation_direct": [17, 18],
        },
    },
    "FRA": {
        "sim_type": "goals",
        "home_advantage": 80,
        "has_knockout": False,
        "classification": {
            "league": [
            "points",
            "goal_difference",
            "h2h_points",
            "h2h_goal_difference",
            "h2h_goals_for",
            "h2h_away_goals_for",
            "goals_for",
            "away_goals_for",
            ],
        },
        "qualification": {
            "champion": [1],
            "top_4": [1, 2, 3, 4],
            "relegation_playoff": [16],
            "relegation_direct": [17, 18],
        },
    },
    "UCL": {
        "sim_type": "goals",
        "home_advantage": 80,
        "has_knockout": True,
        "classification": {
            "league": [
                "points",
                "goal_difference",
                "goals_for",
                "away_goals_for",
                "wins",
                "away_wins",
                "opponent_points",
                "opponent_goal_difference",
                "opponent_goals_for",
            ],
        },
        # No relegation info for UCL, exclude or set None
        "qualification": {
            "direct_to_round_of_16": list(range(1, 9)),
            "playoff": list(range(9, 25)),
        },
        "knockout_bracket": [
            (1, "Bye"),
            (16, 17),
            (8, "Bye"),
            (9, 24),
            (4, "Bye"),
            (13, 20),
            (5, "Bye"),
            (12, 21),
            (2, "Bye"),
            (15, 18),
            (7, "Bye"),
            (10, 23),
            (3, "Bye"),
            (14, 19),
            (6, "Bye"),
            (11, 22),
        ],
        "knockout_format": {
            "po_r32": "two-legged",
            "po_r16": "two-legged",
            "po_r8": "two-legged",
            "po_r4": "two-legged",
            "po_r2": "single_game_neutral",
        },
        "knockout_draw_status": "pending_draw",
        "knockout_draw": None,
        "knockout_reseeding": False,
    },
    "UEL": {
        "sim_type": "goals",
        "home_advantage": 80,
        "has_knockout": True,
        "classification": {
            "league": [
                "points",
                "goal_difference",
                "goals_for",
                "away_goals_for",
                "wins",
                "away_wins",
                "opponent_points",
                "opponent_goal_difference",
                "opponent_goals_for",
            ],
        },
        # No relegation info for UCL, exclude or set None
        "qualification": {
            "direct_to_round_of_16": list(range(1, 9)),
            "playoff": list(range(17, 25)),
        },
        "knockout_bracket": [
            (1, "Bye"),
            (16, 17),
            (8, "Bye"),
            (9, 24),
            (4, "Bye"),
            (13, 20),
            (5, "Bye"),
            (12, 21),
            (2, "Bye"),
            (15, 18),
            (7, "Bye"),
            (10, 23),
            (3, "Bye"),
            (14, 19),
            (6, "Bye"),
            (11, 22),
        ],
        "knockout_format": {
            "po_r32": "two-legged",
            "po_r16": "two-legged",
            "po_r8": "two-legged",
            "po_r4": "two-legged",
            "po_r2": "single_game_neutral",
        },
        "knockout_draw_status": "pending_draw",
        "knockout_draw": None,
        "knockout_reseeding": False,
    },
    "UECL": {
        "sim_type": "goals",
        "home_advantage": 80,
        "has_knockout": True,
        "classification": {
            "league": [
                "points",
                "goal_difference",
                "goals_for",
                "away_goals_for",
                "wins",
                "away_wins",
                "opponent_points",
                "opponent_goal_difference",
                "opponent_goals_for",
            ],
        },
        # No relegation info for UCL, exclude or set None
        "qualification": {
            "direct_to_round_of_16": list(range(1, 9)),
            "playoff": list(range(17, 25)),
        },
        "knockout_bracket": [
            (1, "Bye"),
            (16, 17),
            (8, "Bye"),
            (9, 24),
            (4, "Bye"),
            (13, 20),
            (5, "Bye"),
            (12, 21),
            (2, "Bye"),
            (15, 18),
            (7, "Bye"),
            (10, 23),
            (3, "Bye"),
            (14, 19),
            (6, "Bye"),
            (11, 22),
        ],
        "knockout_format": {
            "po_r32": "two-legged",
            "po_r16": "two-legged",
            "po_r8": "two-legged",
            "po_r4": "two-legged",
            "po_r2": "single_game_neutral",
        },
        "knockout_draw_status": "pending_draw",
        "knockout_draw": None,
        "knockout_reseeding": False,

    },
    "NFL": {
        "sim_type": "winner",
        "home_advantage": 50,
        "elo_kfactor": 20,
        "season_start_adj": 1/3,
        "has_knockout": True,
        "classification": {
            "division": ["win_loss_pct",
                         "h2h_win_loss_pct",
                         "win_loss_pct_div",
                         "win_loss_pct_conf",
                         "strength_of_victory",
                         "strength_of_schedule"
                         ],
            "conference": [
                         "division_winner",
                         "win_loss_pct",
                         "h2h_break_division_ties",
                         "h2h_sweep_full",
                         "win_loss_pct_conf",
                         "h2h_win_loss_pct_common_games",
                         "strength_of_victory",
                         "strength_of_schedule",
                         ],
            "league": ["win_loss_pct"],
        },
        "qualification": {
            "playoff": [f"NFC {i}" for i in range(1, 8)] + [f"AFC {i}" for i in range(1, 8)],
            "first_round_bye": [f"NFC {i}" for i in range(1, 2)] + [f"AFC {i}" for i in range(1, 2)]
        },
        "knockout_bracket": [
            ("NFC 1", "Bye"),
            ("NFC 2", "NFC 7"),
            ("NFC 3", "NFC 6"),
            ("NFC 4", "NFC 5"),
            ("AFC 1", "Bye"),
            ("AFC 2", "AFC 7"),
            ("AFC 3", "AFC 6"),
            ("AFC 4", "AFC 5"),
        ],
        "knockout_format": {
            "po_r16": "single_game",
            "po_r8": "single_game",
            "po_r4": "single_game",
            "po_r2": "single_game_neutral",
        },
        "knockout_draw_status": "no_draw",
        "knockout_draw": None,
        "knockout_reseeding": True
    },
    "NBA": {
        "sim_type": "winner",
        "home_advantage": 50,
        "elo_kfactor": 20,
        "season_start_adj": 1/3,
        "has_knockout": True,
        "classification": {
            "division": ["win_loss_pct"
                         ],
            "conference": [
                         "win_loss_pct",
                         "h2h_win_loss_pct",
                         "win_loss_pct_div_if_same_div",
                         "win_loss_pct_conf",
                         "win_loss_pct_playoff_teams_same_conf",
                         "win_loss_pct_playoff_teams_other_conf",
                         ],
            "league": [
                         "win_loss_pct",
                         "h2h_win_loss_pct",
                         "win_loss_pct_div",
                         "win_loss_pct_conf",
                         "win_loss_pct_playoff_teams_same_conf",
                         "win_loss_pct_playoff_teams_other_conf",
                         ],
        },
        "qualification": {
            "playoff": [f"Eastern {i}" for i in range(1, 9)] + [f"Western {i}" for i in range(1, 9)]
        },
        "knockout_bracket": [
            ("Eastern 1", "Eastern 8"),
            ("Eastern 4", "Eastern 5"),
            ("Eastern 2", "Eastern 7"),
            ("Eastern 3", "Eastern 6"),
            ("Western 1", "Western 8"),
            ("Western 4", "Western 5"),
            ("Western 2", "Western 7"),
            ("Western 3", "Western 6"),
        ],
        "knockout_format": {
            "po_r16": "best_of_7",
            "po_r8": "best_of_7",
            "po_r4": "best_of_7",
            "po_r2": "best_of_7",
        },
        "knockout_draw_status": "no_draw",
        "knockout_draw": None,
        "knockout_reseeding": False,
        "has_play_in": True,
    },
}
