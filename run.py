from src.retriever.fixtures import run_fixtures
from src.retriever.elos import run_elos_fetch
from src.ranker.calculate_elos import run_elo_calc
from src.simulator.sim_season import run_all_simulations

import argparse

def run():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "task",
        choices=["fixtures", "elos_fetch", "elo_calc", "simulator", "all"],
        help="Which process to run"
    )

    args = parser.parse_args()

    if args.task == "fixtures":
        run_fixtures()

    elif args.task == "elos_fetch":
        run_elos_fetch()

    elif args.task == "elo_calc":
        run_elo_calc()

    elif args.task == "simulator":
        run_all_simulations()

    elif args.task == "all":
        run_fixtures()
        run_elo_calc()
        run_all_simulations()

if __name__ == "__main__":
    run()