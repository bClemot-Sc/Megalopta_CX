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


## ----- Select paradigm
paradigm_dict = {
    1 : "Trial A: 1 PFN + 1 goal",
    2 : "Trial B: 1 PFN + 2 goals",
    3 : "Trial C: 2 PFNs + 2 goals",
    4 : "Test 1: 2hDs",
    5 : "Test 2: 2hDs v.2",
    6 : "Test 3: 2hDs v.3",
    7 : "Test 4: 1hD",
    8 : "Test 5: 2hDs + Plasticity",
    9 : "Test 8: Reversed Decision"
}


## ----- Parameters
NOISE = 5
TRIAL = 10

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
    

## ----- Function to compute the circular median
def circ_median(angles):
    angles = np.asarray(angles)
    # Compute pairwise angular differences
    diffs = np.abs(np.subtract.outer(angles, angles))
    # Ensure we account for the circular nature of the data
    diffs = np.minimum(diffs, 2*np.pi - diffs)
    # Sum the differences for each angle
    sum_diffs = np.sum(diffs, axis=1)
    # Find the angle that minimizes the sum of differences
    median_idx = np.argmin(sum_diffs)
    return angles[median_idx]


for PARADIGM_NUMBER in [9]:
    ## ------ Run simulation and generate data
    matplotlib.use('Agg')
    warnings.filterwarnings("ignore")
    RATIOS = [round(i/10, 2) for i in range(0, 5)] + [round(0.4 + i/50, 2) for i in range(0, 11)] + [round(0.6 + i/10, 2) for i in range(0, 5)]
    # RATIOS = [round(i/10, 2) for i in range(0, 4)] + [round(0.4 + i/50, 2) for i in range(0, 21)] + [round(0.9 + i/10, 2) for i in range(0, 2)]
    RATIOS = list(set(RATIOS))
    RATIOS.sort()
    list_ratios = []
    for ratio in RATIOS:
        CX_Script.run_function(500, 'Day', NOISE, 0, paradigm_dict[PARADIGM_NUMBER], 0, 200, 0, ratio, TRIAL, [0,0,1,0])
        with open("Saved_results\last_goal_integration.csv", 'r') as csvfile:
            csv_reader = csv.reader(csvfile)
            for row in csv_reader:
                list_angles = row
        list_angles = [float(x) for x in list_angles]
        list_ratios.append(list_angles)
        print(ratio,"ratio done!")
    RESULTS = np.array(list_ratios)
    RESULTS = RESULTS.transpose()


    ## ----- Create plot
    # Jitter function
    def jitter(array):
        stdev = .01*(max(array)-min(array))
        return array + np.random.randn(len(array)) * stdev

    # Computing mean and standard deviation
    mean_angle = []
    std_angle = []
    for y in range(RESULTS.shape[1]):
        mean_angle.append(np.degrees(circ_median(np.radians(RESULTS[:,y]))))
        std_angle.append(np.degrees(circstd(np.radians(RESULTS[:,y]))))
    mean_angle = normalise_value(mean_angle)
    std_angle = normalise_std(std_angle)

    # Graphical representation
    matplotlib.use('TkAgg')
    plt.figure(figsize=(10, 6))  # Adjusting figure size
    plt.errorbar(RATIOS, mean_angle, yerr=std_angle, fmt='-o', color='#4C72B0', ecolor='#DD8452', capsize=5, label='Mean with Std Dev')
    for i, ratio in enumerate(RATIOS):
        plt.scatter(np.full_like(normalise_value(RESULTS[:,i]), ratio), jitter(normalise_value(RESULTS[:,i])), color='grey', alpha=0.5, label='_nolegend_')
    plt.xlabel('Ratio', fontsize=12)  # Adjusting axis label font size
    plt.ylabel('Mean', fontsize=12)  # Adjusting axis label font size
    plt.title('Goal decision for ' + paradigm_dict[PARADIGM_NUMBER] + ", noise:" + str(NOISE) + ", trial:" + str(TRIAL) + "", fontsize=14, fontweight='bold')  # Adjusting title font size and style
    plt.grid(True, linestyle='--', alpha=0.7)  # Adding grid with dashed lines
    # Adding horizontal lines
    plt.fill_between(RATIOS, normalise_value([145]), normalise_value([125]), color='green', alpha=0.1)
    plt.fill_between(RATIOS, normalise_value([35]), normalise_value([55]), color='green', alpha=0.1)
    plt.axhline(y=-1, color='green', linestyle='--', linewidth=1)
    plt.axhline(y=1, color='green', linestyle='--', linewidth=1)
    plt.legend()  # Adding legend
    plt.tight_layout()  # Adjusting layout to prevent clipping of labels
    plt.savefig('Figures/' + substring_until_colon(paradigm_dict[PARADIGM_NUMBER]) + "_noise" + str(NOISE) + "_trial" + str(TRIAL) + "" + '.svg', format='svg')