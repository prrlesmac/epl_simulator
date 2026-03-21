CREATE TABLE fifa_wc_teams (
    id SERIAL PRIMARY KEY,
    team VARCHAR(100) NOT NULL,
    group_name VARCHAR(10) NOT NULL,
    confederation VARCHAR(20) NOT NULL
);

INSERT INTO fifa_wc_teams (team, group_name, confederation) VALUES

-- Group A
('Mexico', 'Group A', 'CONCACAF'),
('South Korea', 'Group A', 'AFC'),
('South Africa', 'Group A', 'CAF'),
('Denmark', 'Group A', 'UEFA'), 

-- Group B
('Canada', 'Group B', 'CONCACAF'),
('Switzerland', 'Group B', 'UEFA'),
('Qatar', 'Group B', 'AFC'),
('Italy', 'Group B', 'UEFA'),

-- Group C
('Brazil', 'Group C', 'CONMEBOL'),
('Morocco', 'Group C', 'CAF'),
('Scotland', 'Group C', 'UEFA'),
('Haiti', 'Group C', 'CONCACAF'),

-- Group D
('USA', 'Group D', 'CONCACAF'),
('Paraguay', 'Group D', 'CONMEBOL'),
('Australia', 'Group D', 'AFC'),
('Turkey', 'Group D', 'UEFA'),

-- Group E
('Germany', 'Group E', 'UEFA'),
('Ecuador', 'Group E', 'CONMEBOL'),
('Ivory Coast', 'Group E', 'CAF'),
('Curaçao', 'Group E', 'CONCACAF'),

-- Group F
('Netherlands', 'Group F', 'UEFA'),
('Japan', 'Group F', 'AFC'),
('Tunisia', 'Group F', 'CAF'),
('Ukraine', 'Group F', 'UEFA'),

-- Group G
('Belgium', 'Group G', 'UEFA'),
('Egypt', 'Group G', 'CAF'),
('Iran', 'Group G', 'AFC'),
('New Zealand', 'Group G', 'OFC'),

-- Group H
('Spain', 'Group H', 'UEFA'),
('Uruguay', 'Group H', 'CONMEBOL'),
('Saudi Arabia', 'Group H', 'AFC'),
('Cape Verde', 'Group H', 'CAF'),

-- Group I
('France', 'Group I', 'UEFA'),
('Norway', 'Group I', 'UEFA'),
('Senegal', 'Group I', 'CAF'),
('Iraq', 'Group I', 'AFC'),

-- Group J
('Argentina', 'Group J', 'CONMEBOL'),
('Algeria', 'Group J', 'CAF'),
('Austria', 'Group J', 'UEFA'),
('Jordan', 'Group J', 'AFC'),

-- Group K
('Portugal', 'Group K', 'UEFA'),
('Colombia', 'Group K', 'CONMEBOL'),
('Uzbekistan', 'Group K', 'AFC'),
('DR Congo', 'Group K', 'CAF'),

-- Group L
('England', 'Group L', 'UEFA'),
('Croatia', 'Group L', 'UEFA'),
('Ghana', 'Group L', 'CAF'),
('Panama', 'Group L', 'CONCACAF');