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

## ----- Select paradigm
paradigm_dict = {
    1 : "Trial A: 1 PFN + 1 goal",
    2 : "Trial B: 1 PFN + 2 goals",
    3 : "Trial C: 2 PFNs + 2 goals",
    4 : "Test 1: 2hDs",
    5 : "Test 2: 2hDs v.2",
    6 : "Test 3: 2hDs v.3",
    7 : "Test 4: 1hD"
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
NOISES = list(range(0, 51, 5))

RATIOS = [round(0.4 + i/50, 2) for i in range(0, 11)]
RATIOS = list(set(RATIOS))
RATIOS.sort()

for PARADIGM_NUMBER in list(range(2, 8)):

    matplotlib.use('Agg')
    warnings.filterwarnings("ignore")

    slope_values = []
    ste_values = []

    for noise in NOISES:

        regression_y = []
        regression_x = []

        for ratio in RATIOS:
    
            CX_Script.run_function(500, 'Day', noise, 0, paradigm_dict[PARADIGM_NUMBER], 0, 200, 0, ratio, 10, [0,0,1,0])
            with open("Saved_results\last_goal_integration.csv", 'r') as csvfile:
                csv_reader = csv.reader(csvfile)
                for row in csv_reader:
                    list_angles = row

            regression_y.extend(normalise_value([float(x) for x in list_angles]))
            regression_x.extend([ratio] * len(list_angles))

            print(substring_until_colon(paradigm_dict[PARADIGM_NUMBER]) + 'Noise' + str(noise) + 'Ratio' + str(ratio) + 'Done!')

        X = sm.add_constant(regression_x)
        model = sm.OLS(regression_y, X)
        results = model.fit()

        slope_values.append(results.params[1])
        ste_values.append(results.bse[1])

    plt.errorbar(NOISES, slope_values, yerr=ste_values, fmt='-o', label=substring_until_colon(paradigm_dict[PARADIGM_NUMBER]))


plt.xlabel('X values')
plt.ylabel('Y values')
plt.title('Plot with error bars for each iteration')
plt.legend()
plt.show()



