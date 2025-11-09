# test_Example2.py (for pytest)

# Import numpy and other modules
import numpy as np
import os
import pickle
import pytest
import pandas as pd
import json

from bioverse.survey import TransitSurvey
from bioverse.generator import Generator
from bioverse.hypothesis import Hypothesis
from bioverse import analysis

# --- Set a fixed random seed for reproducibility ---
# This is crucial for the comparison to work.
np.random.seed(42)

# Define the functions in the GLOBAL SCOPE (FIX for pickling/multiprocessing)

def oxygen_evolution(d, f_life=0.8, t_half=3.):
    # First, assign no O2 to all planets
    d['has_O2'] = np.zeros(len(d))

    # Calculate the probability that each EEC has O2 based on its age
    EEC = d['EEC']
    P = f_life * (1 - 0.5**(d['age'][EEC]/t_half))

    # Randomly assign O2 to some EECs
    d['has_O2'][EEC] = np.random.uniform(0, 1, EEC.sum()) < P

    return d

def f(theta, X):
    f_life, t_half = theta
    return f_life * (1-0.5**(X/t_half))

def f_null(theta, X):
    shape = (np.shape(X)[0], 1)
    return np.full(shape, theta)


# Helper function to serialize numpy types from JSON (as used in the notebook)
def numpy_encoder(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")


def test_age_oxygen_correlation():
    # --- 1. Setup and Run the full pipeline to generate new data for comparison ---

    # Setup
    generator = Generator('transit')
    generator.insert_step(oxygen_evolution)
    survey = TransitSurvey('default')
    
    # Simulation Run (t_total=10 years)
    sample, detected, data = survey.quickrun(generator, f_life=0.8, t_half=3., t_total=10*365.25)

    # Hypothesis Setup
    params = ('f_life', 't_half')
    features = ('age',)
    labels = ('has_O2',)
    bounds = np.array([[0.01, 1], [0.3, 30]])
    h_age_oxygen = Hypothesis(f, bounds, params=params, features=features, labels=labels, log=(True, True))
    
    bounds_null = np.array([[0.001, 1.]])
    h_age_oxygen.h_null = Hypothesis(f_null, bounds_null, params=('f_O2',), features=features, labels=labels, log=(True,))

    # The fit steps are skipped since they don't produce new data for later steps to use.
    # results_fit_dlnz = h_age_oxygen.fit(data)
    # results_fit_mannwhitney = h_age_oxygen.fit(data, method='mannwhitney')

    # --- Statistical Power Grid Test 1 (f_life vs t_half) ---
    N_grid = 8
    f_life = np.logspace(-1, 0, N_grid)
    t_half = np.logspace(np.log10(0.5), np.log10(50), N_grid)
    
    # Reloading generator and hypothesis objects as in the notebook for multiprocessing (using the global ones)
    generator = Generator('transit')
    
    results_grid_1 = analysis.test_hypothesis_grid(h_age_oxygen, generator, survey, method='mannwhitney', 
                                                   f_life=f_life, t_half=t_half, N=20, processes=8, t_total=10*365.25)
    
    power_1 = analysis.compute_statistical_power(results_grid_1, method='p', threshold=0.05)


    # --- Statistical Power Grid Test 2 (t_total) ---
    t_total = np.logspace(-1, 1, 10) * 365.25
    
    # Generator object must be re-initialized if modified or used in new context
    generator = Generator('transit') 

    results_grid_2 = analysis.test_hypothesis_grid(h_age_oxygen, generator, survey, method='mannwhitney', 
                                                   f_life=0.5, N=20, processes=8, t_total=t_total)

    power_2 = analysis.compute_statistical_power(results_grid_2, method='p', threshold=0.05)


    # --- 2. Load the baseline data from the new output files ---

    # Load 1: Raw Data (simulated_raw_data_ex2.csv)
    baseline_data_df = pd.read_csv('simulated_raw_data_ex2.csv')
    current_data_df = pd.DataFrame(data)

    # Load 2: Grid 1 Results (grid_results_f_life_t_half_ex2.pkl)
    with open('grid_results_f_life_t_half_ex2.pkl', 'rb') as f_pkl:
        baseline_results_grid_1 = pickle.load(f_pkl)

    # Load 3: Power 1 Data (statistical_power_grid_1_ex2.json)
    with open('statistical_power_grid_1_ex2.json', 'r') as f_json:
        baseline_power_data_1 = json.load(f_json)

    # Load 4: Grid 2 Results (grid_results_t_total_ex2.pkl)
    with open('grid_results_t_total_ex2.pkl', 'rb') as f_pkl:
        baseline_results_grid_2 = pickle.load(f_pkl)

    # Load 5: Power 2 Data (statistical_power_grid_2_ex2.csv)
    baseline_power_df_2 = pd.read_csv('statistical_power_grid_2_ex2.csv')
    current_mean_p_2 = results_grid_2['p'].mean(axis=-1)
    current_power_df_2 = pd.DataFrame({
        't_total': t_total,
        'mean_p': current_mean_p_2,
        'statistical_power': power_2
    })

    # --- 3. Assertions to compare new results against the loaded files ---

    # Assertion 1: Raw Data (simulated_raw_data_ex2.csv)
    # err_msg removed to prevent TypeError
    pd.testing.assert_frame_equal(current_data_df, baseline_data_df, 
                                  check_exact=False, atol=1e-8)

    # Assertion 2: Grid 1 Results (grid_results_f_life_t_half_ex2.pkl)
    for key in results_grid_1:
        if key in ['h', 'h_null', 'generator']:
            continue # Skip Hypothesis and Generator object comparison
        
        # err_msg removed to prevent TypeError
        np.testing.assert_array_almost_equal(results_grid_1[key], baseline_results_grid_1[key], 
                                             decimal=8)

    # Assertion 3: Power 1 Data (statistical_power_grid_1_ex2.json)
    np.testing.assert_array_almost_equal(power_1, np.array(baseline_power_data_1['statistical_power']), decimal=8)
    
    # Assertion 4: Grid 2 Results (grid_results_t_total_ex2.pkl)
    for key in results_grid_2:
        if key in ['h', 'h_null', 'generator']:
            continue # Skip Hypothesis and Generator object comparison
        
        # err_msg removed to prevent TypeError
        np.testing.assert_array_almost_equal(results_grid_2[key], baseline_results_grid_2[key], 
                                             decimal=8)
    
    # Assertion 5: Power 2 Data (statistical_power_grid_2_ex2.csv)
    # err_msg removed to prevent TypeError
    pd.testing.assert_frame_equal(current_power_df_2, baseline_power_df_2, 
                                  check_exact=False, atol=1e-8)