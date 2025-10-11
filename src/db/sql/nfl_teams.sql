CREATE TABLE teams_nfl (
    team VARCHAR(50) PRIMARY KEY,
    division VARCHAR(50),
    conference VARCHAR(50)
);

INSERT INTO teams_nfl (team, division, conference) VALUES
-- AFC East
('Buffalo Bills', 'AFC East', 'AFC'),
('Miami Dolphins', 'AFC East', 'AFC'),
('New England Patriots', 'AFC East', 'AFC'),
('New York Jets', 'AFC East', 'AFC'),

-- AFC North
('Baltimore Ravens', 'AFC North', 'AFC'),
('Cincinnati Bengals', 'AFC North', 'AFC'),
('Cleveland Browns', 'AFC North', 'AFC'),
('Pittsburgh Steelers', 'AFC North', 'AFC'),

-- AFC South
('Houston Texans', 'AFC South', 'AFC'),
('Indianapolis Colts', 'AFC South', 'AFC'),
('Jacksonville Jaguars', 'AFC South', 'AFC'),
('Tennessee Titans', 'AFC South', 'AFC'),

-- AFC West
('Denver Broncos', 'AFC West', 'AFC'),
('Kansas City Chiefs', 'AFC West', 'AFC'),
('Las Vegas Raiders', 'AFC West', 'AFC'),
('Los Angeles Chargers', 'AFC West', 'AFC'),

-- NFC East
('Dallas Cowboys', 'NFC East', 'NFC'),
('New York Giants', 'NFC East', 'NFC'),
('Philadelphia Eagles', 'NFC East', 'NFC'),
('Washington Commanders', 'NFC East', 'NFC'),

-- NFC North
('Chicago Bears', 'NFC North', 'NFC'),
('Detroit Lions', 'NFC North', 'NFC'),
('Green Bay Packers', 'NFC North', 'NFC'),
('Minnesota Vikings', 'NFC North', 'NFC'),

-- NFC South
('Atlanta Falcons', 'NFC South', 'NFC'),
('Carolina Panthers', 'NFC South', 'NFC'),
('New Orleans Saints', 'NFC South', 'NFC'),
('Tampa Bay Buccaneers', 'NFC South', 'NFC'),

-- NFC West
('Arizona Cardinals', 'NFC West', 'NFC'),
('Los Angeles Rams', 'NFC West', 'NFC'),
('San Francisco 49ers', 'NFC West', 'NFC'),
('Seattle Seahawks', 'NFC West', 'NFC');
