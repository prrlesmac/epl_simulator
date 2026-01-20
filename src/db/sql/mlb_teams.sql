CREATE TABLE teams_mlb (
    team VARCHAR(50) PRIMARY KEY,
    division VARCHAR(50),
    conference VARCHAR(50)
);

INSERT INTO teams_mlb (team, division, conference) VALUES
-- American League - East Division
('Baltimore Orioles', 'AL East', 'American League'),
('Boston Red Sox', 'AL East', 'American League'),
('New York Yankees', 'AL East', 'American League'),
('Tampa Bay Rays', 'AL East', 'American League'),
('Toronto Blue Jays', 'AL East', 'American League'),

-- American League - Central Division
('Chicago White Sox', 'AL Central', 'American League'),
('Cleveland Guardians', 'AL Central', 'American League'),
('Detroit Tigers', 'AL Central', 'American League'),
('Kansas City Royals', 'AL Central', 'American League'),
('Minnesota Twins', 'AL Central', 'American League'),

-- American League - West Division
('Houston Astros', 'AL West', 'American League'),
('Los Angeles Angels', 'AL West', 'American League'),
('Athletics', 'AL West', 'American League'),
('Seattle Mariners', 'AL West', 'American League'),
('Texas Rangers', 'AL West', 'American League'),

-- National League - East Division
('Atlanta Braves', 'NL East', 'National League'),
('Miami Marlins', 'NL East', 'National League'),
('New York Mets', 'NL East', 'National League'),
('Philadelphia Phillies', 'NL East', 'National League'),
('Washington Nationals', 'NL East', 'National League'),

-- National League - Central Division
('Chicago Cubs', 'NL Central', 'National League'),
('Cincinnati Reds', 'NL Central', 'National League'),
('Milwaukee Brewers', 'NL Central', 'National League'),
('Pittsburgh Pirates', 'NL Central', 'National League'),
('St. Louis Cardinals', 'NL Central', 'National League'),

-- National League - West Division
('Arizona D''Backs', 'NL West', 'National League'),
('Colorado Rockies', 'NL West', 'National League'),
('Los Angeles Dodgers', 'NL West', 'National League'),
('San Diego Padres', 'NL West', 'National League'),
('San Francisco Giants', 'NL West', 'National League');
