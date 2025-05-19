to build docker image
docker build -t epl_simulator .

to run the container
docker run -d -p 8050:8050 epl_simulator

to deploy into AWS

ToDos:
parametrize stuff like output folders, N simulations, etc...
use cli to run commands
output data into a DB or S3 bucket
create cicd
create job for refreshing data (hourly?)
make a decent front-end
write tests
write exceptions and raise errors