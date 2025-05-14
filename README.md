to build docker image
docker build -t epl_simulator .

to run the container
docker run -d -p 8050:8050 epl_simulator

to deploy into AWS
