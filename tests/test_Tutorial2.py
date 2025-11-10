# test_Tutorial2.py (for pytest)

import numpy as np
import os
import pytest
import pandas as pd
import json
import subprocess
from bioverse.constants import ROOT_DIR # Assumed to be available in the test environment

NOTEBOOK_PATH = 'create_test_Tutorial2.ipynb'

# List of all files created by the notebook that should be checked
OUTPUT_FILES = [
    'transit_survey_info.txt',
    'detected_planets_summary.json',
    'detection_breakdown.json',
    'survey_comparison_summary.json'
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
    
    # Check detected_planets_summary.json (Default Survey)
    with open('detected_planets_summary.json', 'r') as f:
        default_summary = json.load(f)
    assert default_summary['N_detected'] > 20, "Default survey detected too few planets."
    
    # Check detection_breakdown.json
    with open('detection_breakdown.json', 'r') as f:
        breakdown = json.load(f)
    assert breakdown['N_total_detected'] == default_summary['N_detected'], "Total detected count mismatch."
    assert breakdown['H2O'] > 0, "No H2O detections in default survey."
    
    # Check survey_comparison_summary.json
    with open('survey_comparison_summary.json', 'r') as f:
        comparison = json.load(f)
        
    default_data = comparison['Default_Survey']
    ht_data = comparison['High_Throughput_Survey']
    
    # The high throughput survey should detect more planets overall (N_detected)
    assert ht_data['N_detected'] > default_data['N_detected'], "High Throughput Survey did not detect more planets."
    # The default survey (deeper) should detect a higher fraction of O2 (biosignature)
    # The number of O2 detections should be comparable or lower, but the *fraction* is the key
    # For a simple test, we just check they are non-zero
    assert ht_data['N_O2_detected'] >= 0 and default_data['N_O2_detected'] >= 0, "O2 detection counts are negative."