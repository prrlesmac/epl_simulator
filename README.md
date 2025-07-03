to install package locally:
pip install -e .

to build docker image
docker build -t epl_simulator .

to run the container
docker run -d -p 8050:8050 epl_simulator

to run psycopg in windows make sure to install psycopg-binary package:
pip install psycopg-binary

ToDos:
rethink architecture to allow for continental leagues
ucl:
- add full tiebreakers
- simulate po
    - add actual elo simulation to the sim bracket functions
    - integrate to group stage simulation
    - have rules for when ties have only had 1st leg played
    - think of better ways to get the elos for playoff than getting the from the standings_df
    - consider home and away ties and adjust the we calc for that
- output to db and fe
store the 10,000 sims results in db
calculate match odds
show upcoming fixtures (2w) and their wdl odds
calculation of how a game affects title odds
do tiebreaker for h2h of multiple ones (tiebreaker of 3 then of 2)
do good test of ranking and tiebreaker with diff scenarios
fe: add a 'last updated on' text box
fix warning messages
give an option to run the season from start or from a slected date for testning (ignore all played games)
improve logging of jobs
write tests
write exceptions and raise errors
do a good readme with instructions for deploying, setting up a db, secrets to setup, etc...
use cli to run commands
do a landing page with:
- summary title odds per league
- summary upcoming matches

tiebreakers:
ENG:
Goal diff
Goals scored
Points in head-to-head
Away goals in head-to-head
Playoff

ESP:
h2h pts
h2h goal diff
goal diff
goal scored
fair play

ITA:
tiebreaker for champiom and 3rd relegated
h2h points
h2h goal diff
goal diff
goal scored

GER:
Goal diff
Goal scored
h2h results
h2h away goals
away goals
playoff

FRA:
Goal diff
h2h points
h2h goal diff
h2h goal scored
h2h away goals
goals scored
away goals
fair ply


relegation
ENG: bottom 3
ESP: bottom 3
ITA: bottom 3
GER: bottom 2, 16th to po
FRA: bottom 2, 16th to po
