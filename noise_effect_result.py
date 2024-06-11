## Plot goal competition results for each trial
## Author: Bastien Clémot

## ----- Import libraries
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pandas as pd
from scipy.stats import circmean, circstd
import CX_Script
import csv
import warnings
import statsmodels.api as sm
from tqdm import tqdm  # Import tqdm

## ----- Select paradigm
paradigm_dict = {
    1: "Trial A: 1 PFN + 1 goal",
    2: "Trial B: 1 PFN + 2 goals",
    3: "Trial C: 2 PFNs + 2 goals",
    4: "Test 1: 2hDs",
    5: "Test 2: 2hDs v.2",
    6: "Test 3: 2hDs v.3",
    7: "Test 4: 1hD",
    8: "Test 5: 2hDs + Plasticity"
}

## ----- Parameters
TRIAL = 20

## ----- Function for normalising values
def normalise_value(vector):
    result = []
    for value in vector:
        if value == 90:
            result.append(0)
        else:
            result.append((value - 90) / 45)
    return result

## ----- Function to normalise std
def normalise_std(vector):
    result = []
    for value in vector:
        if value == 0:
            result.append(0)
        elif value > 0:
            result.append(value / 45)
        else:
            result.append(value / -45)
    return result

## ----- Function to cut the paradigm name
def substring_until_colon(s):
    index = s.find(':')
    if index != -1:
        return s[:index]
    else:
        return s

## ----- Generate data
NOISES = [0, 3, 5, 10, 15, 20, 30]

RATIOS = [round(i/10, 2) for i in range(0, 11)]
RATIOS = list(set(RATIOS))
RATIOS.sort()

# Create a top-level progress bar for the paradigms
paradigm_progress = tqdm([2, 4, 5, 6, 7, 8], desc="Paradigms")

plt.close('all')
matplotlib.use('Agg')
warnings.filterwarnings("ignore")

slope_stock = []
ste_stock = []

for PARADIGM_NUMBER in paradigm_progress:

    slope_values = []
    ste_values = []

    # Create a progress bar for the noise levels
    noise_progress = tqdm(NOISES, desc=f"Noise levels for Paradigm {PARADIGM_NUMBER}", leave=False)

    for noise in noise_progress:

        regression_y = []
        regression_x = []

        # Create a progress bar for the ratios
        ratio_progress = tqdm(RATIOS, desc="Ratios", leave=False)

        for ratio in ratio_progress:

            CX_Script.run_function(500, 'Day', noise, 0, paradigm_dict[PARADIGM_NUMBER], 0, 200, 0, ratio, TRIAL, [0, 0, 1, 0])
            with open("Saved_results\\last_goal_integration.csv", 'r') as csvfile:
                csv_reader = csv.reader(csvfile)
                for row in csv_reader:
                    list_angles = row

            regression_y.extend(normalise_value([float(x) for x in list_angles]))
            regression_x.extend([ratio] * len(list_angles))

        X = sm.add_constant(regression_x)
        model = sm.OLS(regression_y, X)
        results = model.fit()

        slope_values.append(results.params[1])
        ste_values.append(results.bse[1])

    slope_stock.append(slope_values)
    ste_stock.append(ste_values)

print(slope_stock)
print(ste_stock)

plt.close('all')
plt.figure()

for i in range(len(slope_stock)):
    plt.errorbar(NOISES, slope_stock[i], yerr=ste_stock[i], fmt='-o', label=str(i))

plt.xlabel('Noise Levels')
plt.ylabel('Slope Values')
plt.title('Slope vs Noise Levels for Each Paradigm')
plt.legend()
plt.savefig("Figures/Noise_effect_plot.svg", format='svg')
