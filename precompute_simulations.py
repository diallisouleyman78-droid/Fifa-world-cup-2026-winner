"""
Run the World Cup Monte Carlo simulation many times and store the aggregated
results in a pickle file so the Streamlit app can load them instantly.

Usage:
    python precompute_simulations.py            # 200 simulations (default)
    python precompute_simulations.py 500        # custom number of simulations
"""

import os
import sys
import time
import warnings

# Suppress Streamlit cache warnings when running outside the app
os.environ["STREAMLIT_SERVER_HEADLESS"] = "true"
warnings.filterwarnings("ignore", message=".*ScriptRunContext.*")

from src.tournament import precompute_and_save


def main():
    n_simulations = 200
    if len(sys.argv) > 1:
        n_simulations = int(sys.argv[1])

    print(f"Running {n_simulations} tournament simulations...")

    start = time.time()

    def progress(done, total):
        if done % 10 == 0 or done == total:
            elapsed = time.time() - start
            print(f"  {done}/{total} done ({elapsed:.0f}s elapsed)")

    output_path = precompute_and_save(
        n_simulations=n_simulations,
        progress_callback=progress,
    )

    elapsed = time.time() - start
    print(f"Done in {elapsed:.0f}s. Results saved to: {output_path}")


if __name__ == "__main__":
    main()
