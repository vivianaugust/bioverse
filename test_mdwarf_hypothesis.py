# test_mdwarf_hypothesis.py (for pytest)

import numpy as np
import os
import pytest
import pandas as pd
import json
import subprocess

NOTEBOOK_PATH = 'create_test_mdwarf_hypothesis.ipynb'

# List of all files created by the notebook that should be checked
OUTPUT_FILES = [
    'mdwarf_hypothesis_analysis_grid.csv',
    'mdwarf_hypothesis_odds_ratio.json'
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
    
    # Check mdwarf_hypothesis_analysis_grid.csv
    df_grid = pd.read_csv('mdwarf_hypothesis_analysis_grid.csv')
    # The grid is explicitly set to 11 points (linspace(0.01, 0.20, 11))
    assert len(df_grid) == 11, "Analysis grid does not contain 11 points."
    
    # Check mdwarf_hypothesis_odds_ratio.json
    with open('mdwarf_hypothesis_odds_ratio.json', 'r') as f:
        odds_ratio_summary = json.load(f)
        
    # Check the key metric and that the log-likelihoods were calculated
    assert 'odds_ratio_alt_vs_null' in odds_ratio_summary, "Odds ratio not found in summary JSON."
    assert odds_ratio_summary['log_Z_null_placeholder'] < 0, "Log-Evidence (Null) should be negative."
    assert odds_ratio_summary['log_Z_alt_max'] < 0, "Log-Evidence (Alt) should be negative."
    assert odds_ratio_summary['odds_ratio_alt_vs_null'] > 0, "Odds ratio must be positive."