CREATE TABLE teams_mlb (
    team VARCHAR(50) PRIMARY KEY,
    division VARCHAR(50),
    league VARCHAR(50)
);

INSERT INTO teams_mlb (team, division, league) VALUES
-- American League - East Division
('Baltimore Orioles', 'East', 'American League'),
('Boston Red Sox', 'East', 'American League'),
('New York Yankees', 'East', 'American League'),
('Tampa Bay Rays', 'East', 'American League'),
('Toronto Blue Jays', 'East', 'American League'),

-- American League - Central Division
('Chicago White Sox', 'Central', 'American League'),
('Cleveland Guardians', 'Central', 'American League'),
('Detroit Tigers', 'Central', 'American League'),
('Kansas City Royals', 'Central', 'American League'),
('Minnesota Twins', 'Central', 'American League'),

-- American League - West Division
('Houston Astros', 'West', 'American League'),
('Los Angeles Angels', 'West', 'American League'),
('Oakland Athletics', 'West', 'American League'),
('Seattle Mariners', 'West', 'American League'),
('Texas Rangers', 'West', 'American League'),

-- National League - East Division
('Atlanta Braves', 'East', 'National League'),
('Miami Marlins', 'East', 'National League'),
('New York Mets', 'East', 'National League'),
('Philadelphia Phillies', 'East', 'National League'),
('Washington Nationals', 'East', 'National League'),

-- National League - Central Division
('Chicago Cubs', 'Central', 'National League'),
('Cincinnati Reds', 'Central', 'National League'),
('Milwaukee Brewers', 'Central', 'National League'),
('Pittsburgh Pirates', 'Central', 'National League'),
('St. Louis Cardinals', 'Central', 'National League'),

-- National League - West Division
('Arizona D''Backs', 'West', 'National League'),
('Colorado Rockies', 'West', 'National League'),
('Los Angeles Dodgers', 'West', 'National League'),
('San Diego Padres', 'West', 'National League'),
('San Francisco Giants', 'West', 'National League');
