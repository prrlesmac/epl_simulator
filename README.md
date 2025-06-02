to install package locally:
pip install -e .

to build docker image
docker build -t epl_simulator .

to run the container
docker run -d -p 8050:8050 epl_simulator

to deploy into AWS

ToDos:
move webapp to another repo and container
add update dates to tables in sql db
dont run sims on docker, only app deploy
run sims using cron
improve logging of jobs
write tests
write exceptions and raise errors
do a good readme with instructions for deploying, setting up a db, secrets to setup, etc...
Get a static IP address in the Ec2 instance
parametrize table name outputs in SQL
think of adding non-GD/GF tie-breakers
use cli to run commands
create job for refreshing data (hourly?)
make a decent front-end
