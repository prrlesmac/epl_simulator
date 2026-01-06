to install package locally:
pip install -e .

to build docker image
docker build -t epl_simulator .

to run the container
docker run -d -p 8050:8050 epl_simulator

to run psycopg in windows make sure to install psycopg-binary package:
pip install psycopg-binary

to run the profiler
pip install line_profiler
add #@profile decorator to function
in cmd : kernprof -l -v src/simulator/sim_season.py > output.txt