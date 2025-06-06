to install package locally:
pip install -e .

to build docker image
docker build -t epl_simulator .

to run the container
docker run -d -p 8050:8050 epl_simulator

to deploy into AWS

ToDos:
open the code to all top 5 leagues
think of adding non-GD/GF tie-breakers
give an option to run the season from start (ignore all played games)
improve logging of jobs
write tests
write exceptions and raise errors
do a good readme with instructions for deploying, setting up a db, secrets to setup, etc...
Get a static IP address in the Ec2 instance
use cli to run commands
make a decent front-end
