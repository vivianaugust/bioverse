# test_Tutorial_UpdatedSmallPlanets.py (for pytest)

import numpy as np
import os
import pytest
import pandas as pd
import json
import subprocess

NOTEBOOK_PATH = 'create_test_Tutorial_UpdatedSmallPlanets.ipynb'

# List of all files created by the notebook that should be checked
OUTPUT_FILES = [
    'generator_bergsten22_info.txt',
    'updated_planet_sample.csv',
    'occurrence_rates_summary.json'
]

# --- Fixture to execute the notebook and perform initial cleanup ---
@pytest.fixture(scope="module", autouse=True)
def run_notebook():
    print(f"\n--- Executing {NOTEBOOK_PATH} ---")
    
    # Execute the notebook
    try:
        command = ['jupyter', 'nbconvert', '--to', 'python', '--execute', NOTEBOOK_PATH, '--stdout']
        subprocess.run(command, check=True, capture_output=True, text=True)
        print("Notebook execution successful.")
    except subprocess.CalledProcessError as e:
        pytest.fail(f"Notebook execution failed with error: {e.stderr}")

    # Yield control to run the tests
    yield

    # Final check of cleanup
    print("--- Final cleanup check ---")
    for f in OUTPUT_FILES:
        if os.path.exists(f):
            # If the cleanup cell failed, manually remove files for a clean slate
            os.remove(f)
            print(f"Manually cleaned up: {f}")

# --- Test function to compare notebook outputs ---
def test_notebook_outputs(run_notebook):
    
    # 1. Check for file existence
    for f in OUTPUT_FILES:
        assert os.path.exists(f), f"Required output file not found: {f}"

    # 2. Perform sanity checks on the data content (assuming fixed seed=42)
    
    # Check updated_planet_sample.csv
    df_sample = pd.read_csv('updated_planet_sample.csv')
    # With fixed seed and d_max=100, the sample size should be predictable
    assert len(df_sample) > 2500, "Updated sample size is too small."
    
    # Check occurrence_rates_summary.json
    with open('occurrence_rates_summary.json', 'r') as f:
        summary_data = json.load(f)
        
    # The period bins are defined with 5 segments ([2, 5, 10, 20, 40, 100])
    assert len(summary_data['occurrence_by_period_bin']) == 5, "Incorrect number of period bins in occurrence summary."
    assert summary_data['R_split_mean_Mst'] > 1.9 and summary_data['R_split_mean_Mst'] < 2.1, "R_split is outside expected range for solar-like stars."
    
    # Check for the expected trend of Bergsten et al. (Super-Earths dominate at long periods)
    # The last bin is P=[40, 100], where Frac_SuperEarth should be relatively high
    last_bin = summary_data['occurrence_by_period_bin'][-1]
    assert last_bin['Frac_SuperEarth'] > last_bin['Frac_SubNeptune'], "Super-Earth fraction does not dominate Sub-Neptune fraction in the last period bin."