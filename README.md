to install package locally:
pip install -e .

to build docker image
docker build -t epl_simulator .

to run the container
docker run -d -p 8050:8050 epl_simulator

to deploy into AWS

ToDos:
Get a static IP address in the Ec2 instance
think of adding non-GD/GF tie-breakers
dont run sims on docker, only app deploy
run sims using airflow
use cli to run commands
output data into a DB or S3 bucket
create cicd
create job for refreshing data (hourly?)
make a decent front-end
write tests
write exceptions and raise errors