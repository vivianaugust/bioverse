# test_Tutorial1.py (for pytest)

import numpy as np
import os
import pytest
import pandas as pd
import json
import subprocess
from bioverse.constants import ROOT_DIR # Assumed to be available in the test environment

NOTEBOOK_PATH = 'create_test_Tutorial1.ipynb'

# List of all files created by the notebook that should be checked
OUTPUT_FILES = [
    'generator_transit_info.txt',
    'step_create_planets_SAG13_info.txt',
    'initial_planet_sample.csv',
    'initial_planet_sample_summary.json',
    'f_ocean_summary.json',
    'generator_oceans_inserted_info.txt'
]

# --- Fixture to execute the notebook and perform initial cleanup ---
@pytest.fixture(scope="module", autouse=True)
def run_notebook():
    print(f"\n--- Executing {NOTEBOOK_PATH} ---")
    
    # Ensure the required file is in the environment for the generator to load it
    # We rely on the notebook's final cell to perform the full cleanup
    
    # Execute the notebook
    try:
        command = ['jupyter', 'nbconvert', '--to', 'python', '--execute', NOTEBOOK_PATH, '--stdout']
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        print("Notebook execution successful.")
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        pytest.fail(f"Notebook execution failed with error: {e.stderr}")

    # Yield control to run the tests
    yield

    # Final check of cleanup (files should be removed by the last cell of the notebook)
    print("--- Final cleanup check ---")
    for f in OUTPUT_FILES:
        if os.path.exists(f):
            # If the cleanup cell failed, manually remove files for a clean slate
            os.remove(f)
            print(f"Manually cleaned up: {f}")
    
    # Also clean up the saved generator and custom function
    gen_file = os.path.join(ROOT_DIR, 'Objects/Generators/transit_oceans.pkl')
    func_file = os.path.join(ROOT_DIR, 'example_oceans.py')
    for f in [gen_file, func_file]:
        if os.path.exists(f):
            os.remove(f)

# --- Test function to compare notebook outputs ---
def test_notebook_outputs(run_notebook):
    
    # 1. Check for file existence
    for f in OUTPUT_FILES:
        assert os.path.exists(f), f"Required output file not found: {f}"

    # 2. Perform sanity checks on the data content (assuming fixed seed=42)
    
    # Check initial_planet_sample.csv
    df_sample = pd.read_csv('initial_planet_sample.csv')
    # With a fixed seed and d_max=100, the sample size should be predictable
    assert len(df_sample) > 2500, "Initial sample size is too small."
    
    # Check initial_planet_sample_summary.json
    with open('initial_planet_sample_summary.json', 'r') as f:
        summary_data = json.load(f)
    assert summary_data['N_planets'] == len(df_sample), "N_planets in summary does not match CSV size."
    # Check for Exo-Earth Candidates (should be > 50 with eta_Earth=0.15)
    assert df_sample['EEC'].sum() > 50, "Insufficient Exo-Earth Candidates generated."

    # Check f_ocean_summary.json
    with open('f_ocean_summary.json', 'r') as f:
        ocean_data = json.load(f)
    assert ocean_data['f_ocean_noneec_mean'] < 0.001, "Non-EEC planets have non-zero ocean fraction (should be near 0)."
    assert ocean_data['f_ocean_eec_mean'] > 0.4 and ocean_data['f_ocean_eec_mean'] < 0.6, "EEC ocean fraction is outside expected uniform range (0.3 to 0.7)."