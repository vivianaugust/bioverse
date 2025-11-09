# test_magmaocean.py (for pytest)

import numpy as np
import os
import pytest
import pandas as pd
import json
import subprocess
from bioverse.constants import ROOT_DIR # Assumed to be available in the test environment

NOTEBOOK_PATH = 'create_test_magmaocean.ipynb'

# List of all files created by the notebook that should be checked
OUTPUT_FILES = [
    'magmaocean_planets_data.csv',
    'magmaocean_detection_summary.json'
]

# The custom function file is also created, but the notebook is responsible for its removal
CUSTOM_FUNC_FILE = os.path.join(ROOT_DIR, 'example_magmaocean.py')

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
            os.remove(f)
            print(f"Manually cleaned up: {f}")
    if os.path.exists(CUSTOM_FUNC_FILE):
        os.remove(CUSTOM_FUNC_FILE)
        print(f"Manually cleaned up: {CUSTOM_FUNC_FILE}")

# --- Test function to compare notebook outputs ---
def test_notebook_outputs(run_notebook):
    
    # 1. Check for file existence
    for f in OUTPUT_FILES:
        assert os.path.exists(f), f"Required output file not found: {f}"

    # 2. Perform sanity checks on the data content (assuming fixed seed=42)
    
    # Check magmaocean_planets_data.csv
    df_mo = pd.read_csv('magmaocean_planets_data.csv')
    # With a fixed seed and d_max=100, there should be a small number of MO planets
    assert len(df_mo) > 20 and len(df_mo) < 100, "Unexpected number of magma ocean planets generated."
    
    # Check magmaocean_detection_summary.json
    with open('magmaocean_detection_summary.json', 'r') as f:
        summary = json.load(f)
        
    assert summary['N_magmaocean_planets'] == len(df_mo), "Planet count mismatch in summary JSON."
    # The number detected should be less than the total MO planets
    assert summary['N_magmaocean_detected'] < summary['N_magmaocean_planets'], "Detected count should be less than total MO planets (detection is non-trivial)."
    # The detection fraction should be between 0 and 1
    assert summary['detection_fraction'] > 0 and summary['detection_fraction'] < 1, "Detection fraction is outside the valid range (0, 1)."