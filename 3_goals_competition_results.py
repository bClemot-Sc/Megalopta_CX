## Plot goal competition for 3 goals (Test6)
## Author: Bastien Clémot


## ----- Import libraries
import matplotlib.pyplot as plt
import matplotlib
from scipy.spatial import Voronoi, voronoi_plot_2d
import numpy as np
import pandas as pd
import plotly.figure_factory as ff 
from scipy.stats import circmean, circstd
import CX_Script
import csv
import warnings


## ----- Parameters
NOISE = 5
TRIAL = 20


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
    return np.degrees(angles[median_idx])


## ----- Function for goal decision
def goal_decision(median):
    if 35 <= median <= 55:
        output = "goal1"
    elif 170 <= median <= 190:
        output = "goal3"
    elif 215 <= median <= 235:
        output = "goal2"
    else:
        output = "none"
    return output


## ----- Function for min-max normalisation
def min_max_normalize(data, new_min=0, new_max=1):
    min_val =  min([x for x in data if x != 0])
    max_val = max(data)
    normalized_data = []
    for val in data:
        if val == 0:
            normalized_data.append(1.5)
            continue
        normalized_val = ((val - min_val) / (max_val - min_val)) * (new_max - new_min) + new_min
        normalized_data.append(normalized_val)
    return normalized_data


## ------ Run simulation and generate data
matplotlib.use('Agg')
warnings.filterwarnings("ignore")
RATIOS1 = [round(i/20, 2) for i in range(0, 21)] + [0.33]
RATIOS2 = [round(i/20, 2) for i in range(0, 21)] + [0.33]
RATIOS1 = list(set(RATIOS1))
RATIOS2 = list(set(RATIOS2))
RATIOS1.sort()
RATIOS2.sort()
list_results = []
for ratio1 in RATIOS1:
    for ratio2 in RATIOS2:
        if ratio1 + ratio2 > 1:
            continue
        CX_Script.run_function(500, 'Day', NOISE, 0, "Test 6: 3 goals + Plasticity", 0, 200, 0, [ratio1, ratio2], TRIAL, [0,0,1,0])
        with open("Saved_results\last_goal_integration.csv", 'r') as csvfile:
            csv_reader = csv.reader(csvfile)
            for row in csv_reader:
                list_angles = row
        list_angles = [float(x) for x in list_angles]
        list_results.append((ratio1,ratio2,1-ratio1-ratio2,goal_decision(circ_median(np.radians(list_angles))), np.degrees(circstd(np.radians(list_angles)))))
        print(ratio1,"+",ratio2,"ratios done!")
with open('Saved_results\saved_3goals.csv', 'w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerows(list_results)

# dtype = [('col1', float), ('col2', float), ('col3', float), ('col4', 'U10'), ('col5', float)]
# list_results = np.genfromtxt('Saved_results\saved_3goals.csv', delimiter=',', dtype=dtype)

## ----- Create plot
# Sort tuple categories
goal1_list = []
goal2_list = []
goal3_list = []
none_list = []
for tpl in list_results:
    # Unpack tuple values
    val1, val2, val3, category, val5 = tpl
    # Append the tuple to its corresponding category list
    if category == 'goal1':
        goal1_list.append(tpl)
    else:
        goal1_list.append((val1, val2, val3, category, 0))
        
    if category == 'goal2':
        goal2_list.append(tpl)
    else:
        goal2_list.append((val1, val2, val3, category, 0))
        
    if category == 'goal3':
        goal3_list.append(tpl)
    else:
        goal3_list.append((val1, val2, val3, category, 0))
        
    if category == 'none':
        none_list.append(tpl)
    else:
        none_list.append((val1, val2, val3, category, 0))

# Create plot for each category
for results, color in [(goal1_list, 'Blues'), (goal2_list, 'Greens'), (goal3_list, 'Reds'), (none_list, 'Greys')]:
    
    # Get coordinated and values
    a = []
    b = []
    c = []
    v = []
    for tup in results:
        a.append(tup[0])
        b.append(tup[1])
        c.append(round(tup[2],2))
        v.append(tup[-1])

    v = min_max_normalize(v)

    fig = ff.create_ternary_contour( 
        np.array([a, b, c]), np.array(v), 
        pole_labels=['Goal 1 ratio', 'Goal 2 ratio', 'Goal 3 ratio'], 
        colorscale = color,
        showscale = True,
    )
    
    fig.write_image("Figures\Ternary_plot"+color+".svg")