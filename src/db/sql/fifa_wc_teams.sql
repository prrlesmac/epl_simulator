CREATE TABLE teams_fifa_wc (
    id SERIAL PRIMARY KEY,
    team VARCHAR(100) NOT NULL,
    division VARCHAR(10) NOT NULL,
    confederation VARCHAR(20) NOT NULL
);

INSERT INTO teams_fifa_wc (team, division, confederation) VALUES

-- A
('Mexico', 'A', 'CONCACAF'),
('Korea Republic', 'A', 'AFC'),
('South Africa', 'A', 'CAF'),
('Denmark', 'A', 'UEFA'), 

-- B
('Canada', 'B', 'CONCACAF'),
('Switzerland', 'B', 'UEFA'),
('Qatar', 'B', 'AFC'),
('Italy', 'B', 'UEFA'),

-- C
('Brazil', 'C', 'CONMEBOL'),
('Morocco', 'C', 'CAF'),
('Scotland', 'C', 'UEFA'),
('Haiti', 'C', 'CONCACAF'),

-- D
('United States', 'D', 'CONCACAF'),
('Paraguay', 'D', 'CONMEBOL'),
('Australia', 'D', 'AFC'),
('Turkey', 'D', 'UEFA'),

-- E
('Germany', 'E', 'UEFA'),
('Ecuador', 'E', 'CONMEBOL'),
('Côte d''Ivoire', 'E', 'CAF'),
('Curaçao', 'E', 'CONCACAF'),

-- F
('Netherlands', 'F', 'UEFA'),
('Japan', 'F', 'AFC'),
('Tunisia', 'F', 'CAF'),
('Ukraine', 'F', 'UEFA'),

-- G
('Belgium', 'G', 'UEFA'),
('Egypt', 'G', 'CAF'),
('IR Iran', 'G', 'AFC'),
('New Zealand', 'G', 'OFC'),

-- H
('Spain', 'H', 'UEFA'),
('Uruguay', 'H', 'CONMEBOL'),
('Saudi Arabia', 'H', 'AFC'),
('Cape Verde', 'H', 'CAF'),

-- I
('France', 'I', 'UEFA'),
('Norway', 'I', 'UEFA'),
('Senegal', 'I', 'CAF'),
('Iraq', 'I', 'AFC'),

-- J
('Argentina', 'J', 'CONMEBOL'),
('Algeria', 'J', 'CAF'),
('Austria', 'J', 'UEFA'),
('Jordan', 'J', 'AFC'),

-- K
('Portugal', 'K', 'UEFA'),
('Colombia', 'K', 'CONMEBOL'),
('Uzbekistan', 'K', 'AFC'),
('DR Congo', 'K', 'CAF'),

-- L
('England', 'L', 'UEFA'),
('Croatia', 'L', 'UEFA'),
('Ghana', 'L', 'CAF'),
('Panama', 'L', 'CONCACAF');