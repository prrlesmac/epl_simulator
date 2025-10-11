CREATE TABLE teams_nba (
    team VARCHAR(50) PRIMARY KEY,
    division VARCHAR(50),
    conference VARCHAR(50)
);

INSERT INTO teams_nba (team, division, conference) VALUES
-- Eastern Conference - Atlantic Division
('Boston Celtics', 'Atlantic', 'Eastern'),
('Brooklyn Nets', 'Atlantic', 'Eastern'),
('New York Knicks', 'Atlantic', 'Eastern'),
('Philadelphia 76ers', 'Atlantic', 'Eastern'),
('Toronto Raptors', 'Atlantic', 'Eastern'),

-- Eastern Conference - Central Division
('Chicago Bulls', 'Central', 'Eastern'),
('Cleveland Cavaliers', 'Central', 'Eastern'),
('Detroit Pistons', 'Central', 'Eastern'),
('Indiana Pacers', 'Central', 'Eastern'),
('Milwaukee Bucks', 'Central', 'Eastern'),

-- Eastern Conference - Southeast Division
('Atlanta Hawks', 'Southeast', 'Eastern'),
('Charlotte Hornets', 'Southeast', 'Eastern'),
('Miami Heat', 'Southeast', 'Eastern'),
('Orlando Magic', 'Southeast', 'Eastern'),
('Washington Wizards', 'Southeast', 'Eastern'),

-- Western Conference - Northwest Division
('Denver Nuggets', 'Northwest', 'Western'),
('Minnesota Timberwolves', 'Northwest', 'Western'),
('Oklahoma City Thunder', 'Northwest', 'Western'),
('Portland Trail Blazers', 'Northwest', 'Western'),
('Utah Jazz', 'Northwest', 'Western'),

-- Western Conference - Pacific Division
('Golden State Warriors', 'Pacific', 'Western'),
('Los Angeles Clippers', 'Pacific', 'Western'),
('Los Angeles Lakers', 'Pacific', 'Western'),
('Phoenix Suns', 'Pacific', 'Western'),
('Sacramento Kings', 'Pacific', 'Western'),

-- Western Conference - Southwest Division
('Dallas Mavericks', 'Southwest', 'Western'),
('Houston Rockets', 'Southwest', 'Western'),
('Memphis Grizzlies', 'Southwest', 'Western'),
('New Orleans Pelicans', 'Southwest', 'Western'),
('San Antonio Spurs', 'Southwest', 'Western');
