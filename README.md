to install package locally:
pip install -e .

to build docker image
docker build -t epl_simulator .

to run the container
docker run -d -p 8050:8050 epl_simulator

to run psycopg in windows make sure to install psycopg-binary package:
pip install psycopg-binary

ToDos:
handle data types in sql db (create tables or specofy schema in df command)
implement homefield advantage in win prba
do tiebreaker for h2h of multiple ones (tiebreaker of 3 then of 2)
do good test of ranking and tiebreaker with diff scenarios
fix warning messages
parallelize the 10000 sims across nodes
give an option to run the season from start (ignore all played games)
improve logging of jobs
write tests
write exceptions and raise errors
do a good readme with instructions for deploying, setting up a db, secrets to setup, etc...
Get a static IP address in the Ec2 instance
use cli to run commands

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
