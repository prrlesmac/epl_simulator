to install package locally (the -e is to make sure any code edits are reflected):
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

to run scraper+calculator+simulator
python run.py all

to run standalone services
python run.py fixtures
python run.py elos_fetch
python run.py elo_calc
python run.py simulator
