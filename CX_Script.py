#### Script for modelling and testing the CX
## Author: Bastien Clémot


## ----- Import packages
import csv
import math
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import numpy as np
import pandas as pd
import random
from scipy.optimize import curve_fit
from scipy.stats import circmean, circstd
import seaborn as sns


## ----- Import connectivity matrix and neuron IDs
def import_connectivity(paradigm):
    # Get path depending on paradigm
    if paradigm == "Trial A: 1 PFN + 1 goal":
        path = "Connectivity_matrices\Trial_A_matrices.xlsx"
    elif paradigm == "Trial B: 1 PFN + 2 goals":
        path = "Connectivity_matrices\Trial_B_matrices.xlsx"
    elif paradigm == "Trial C: 2 PFNs + 2 goals":
        path = "Connectivity_matrices\Trial_C_matrices.xlsx"
    elif paradigm == "Test 1: 2hDs":
        path = "Connectivity_matrices\Test_1_matrices.xlsx"
    elif paradigm == "Test 2: 2hDs v.2":
        path = "Connectivity_matrices\Test_2_matrices.xlsx"
    elif paradigm == "Test 3: 2hDs v.3":
        path = "Connectivity_matrices\Test_3_matrices.xlsx"
    elif paradigm == "Test 4: 1hD":
        path = "Connectivity_matrices/Test_4_matrices.xlsx"
    elif paradigm == "Test 5: 2hDs + Plasticity":
        path = "Connectivity_matrices\Test_5_matrices.xlsx"
    elif paradigm == "Test 6: 3 goals + Plasticity":
        path = "Connectivity_matrices\Test_6_matrices.xlsx"
    elif paradigm == "Test 7: 2 goals + PI":
        path = "Connectivity_matrices\Test_7_matrices.xlsx"
    else:
        path = "Connectivity_matrices/Theoretical_connectivity_matrices.xlsx"
    # Open Excel sheets
    MATRIX = pd.read_excel(path, sheet_name="Global", header=None)
    IDS = pd.read_excel(path, sheet_name="IDs", header=None)
    # Transpose connectivity matrix and convert IDs to list
    T_MATRIX = MATRIX.T
    COL_IDS = IDS.stack().dropna().tolist()
    # Convert matrix to numpy array
    CON_MAT = T_MATRIX.to_numpy()
    return CON_MAT, COL_IDS


## Get neuron index within the connectivity matrix
def get_neuron_index(ids_list, food):
    # Build a dictionary of all positions
    index_groups = {
        "IND_CIU": [i for i, element in enumerate(ids_list) if "CIU" in element],
        "IND_TR": [i for i, element in enumerate(ids_list) if "TR" in element],
        "IND_TS": [i for i, element in enumerate(ids_list) if "TS" in element or "LNO" in element],
        "IND_EPG": [i for i, element in enumerate(ids_list) if "EPG" in element],
        "IND_PEG": [i for i, element in enumerate(ids_list) if "PEG" in element],
        "IND_PEN": [i for i, element in enumerate(ids_list) if "PEN" in element],
        "IND_D7": [i for i, element in enumerate(ids_list) if "d7-" in element or "D7-" in element],
        "IND_G": [i for i, element in enumerate(ids_list) if "G-" in element],
        "IND_GA": [i for i, element in enumerate(ids_list) if "GA-" in element],
        "IND_GB": [i for i, element in enumerate(ids_list) if "GB-" in element],
        "IND_GC": [i for i, element in enumerate(ids_list) if "GC-" in element],
        "IND_PFN": [i for i, element in enumerate(ids_list) if "PFN" in element],
        "IND_PFNm": [i for i, element in enumerate(ids_list) if "PFNm" in element],
        "IND_PFNh": [i for i, element in enumerate(ids_list) if "PFNh" in element],
        "IND_PFNa": [i for i, element in enumerate(ids_list) if "PFNa" in element],
        "IND_PFNb": [i for i, element in enumerate(ids_list) if "PFNb" in element],
        "IND_HD": [i for i, element in enumerate(ids_list) if "hd" in element or "hD" in element],
        "IND_HDA": [i for i, element in enumerate(ids_list) if "hDa" in element ],
        "IND_HDB": [i for i, element in enumerate(ids_list) if "hDb" in element ],
        "IND_HDC": [i for i, element in enumerate(ids_list) if "hDc" in element ],
        "IND_HDH": [i for i, element in enumerate(ids_list) if "hDh" in element ],
        "IND_HDPIA": [i for i, element in enumerate(ids_list) if "hDpiA" in element ],
        "IND_HDPIB": [i for i, element in enumerate(ids_list) if "hDpiB" in element ],
        "IND_FBtR": [i for i, element in enumerate(ids_list) if "FBtR" in element],
        "IND_PFNc": [i for i, element in enumerate(ids_list) if "PFNc" in element],
        "IND_FBtD": [i for i, element in enumerate(ids_list) if "FBtD" in element],
        "IND_FC": [i for i, element in enumerate(ids_list) if "FC" in element],
        "IND_PFL": [i for i, element in enumerate(ids_list) if "PFL" in element]
    }
    for f in range(1,food+1):
        index_groups["IND_FBtR" + str(f)] = [i for i, element in enumerate(ids_list) if "FBtR" + str(f) + "-" in element]
    return index_groups


## ----- Initialise agent dataframe and neuron activity dataframe
def initialise_dataframes(ids_list, connectivity_matrix, time, food, paradigm):
    # Agent dataframe
    agent_df = pd.DataFrame(0.0, index=range(time+1), columns=["X", "Y", "Orientation", "Speed", "Rotation", "Food"])
    # Set speed to 1 for the whole simulation
    agent_df["Speed"] = 1.0
    agent_df["Food"] = 0
    # Activity dataframe
    activity_df = pd.DataFrame(0.0, index=range(time+1), columns=ids_list)
    return agent_df, activity_df, connectivity_matrix, ids_list


## ----- Initialise food sources in the environment
def initialise_food(paradigm, nest_size, food, radius):
    food_list = []
    # Check paradigm
    for _ in range(food):
        f_radius = random.randint(nest_size+50, 200)
        f_angle = random.uniform(0, 2*math.pi)
        f_x = round(f_radius * math.cos(f_angle))
        f_y = round(radius * math.sin(f_angle))
        f_r = random.randint(5, 20)
        food_list.append((f_x,f_y,f_r))
    return food_list


## ----- Address heading direction (relative to a South landscape cue) to CIU neurons
def CIU_activation(heading_direction):
    relative_heading = (-heading_direction) % 360
    heading_list = [0, 45, 90, 135, 180, 225, 270, 315, 360]
    closest_heading = min(heading_list, key=lambda x: abs(x - relative_heading))
    heading_id = heading_list.index(closest_heading % 360) + 1
    return str(heading_id)


## ----- Address turning direction to TR neurons (orientation comparison)
def compare_headings(previous_heading, new_heading):
    TRr = 0
    TRl = 0
    heading_difference = (new_heading - previous_heading) % 360
    if heading_difference == 0:
        pass
    elif heading_difference <= 180:
        TRl = 1
    else:
        TRr = 1
    return TRl, TRr


## ----- Generate goal directions
def generate_goal(ratio, param, position):
    # # Generate sinusoidal shape
    # x = range(16)
    # a = param[0]
    # b = param[1]
    # c = param[2]
    # d = param[3]
    # y = a * np.sin(b * (x + c)) + d
    # # Adjust ratio
    goal = [0.2,0.5,0.8,0.5,0.2,0,0,0, 0.2,0.5,0.8,0.5,0.2,0,0,0]
    # Adjust position
    while max(goal) != goal[position-1]:
        goal.append(goal.pop(0))
    goal = [x if x >= 0 else 0 for x in goal]
    final_goal = []
    for elem in goal:
        if elem == max(goal):
            final_goal.append(elem*ratio)
        else: final_goal.append(elem*ratio)
    return final_goal


## ----- Calculate euclidean distance between two points
def euclidean_distance(x1,y1,x2,y2):
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)


## ----- Calculate angle between two vectors (vector comparison)
def angle_between_vectors(v1,v2):
    dot_product = sum(a*b for a, b in zip(v1, v2))
    magnitude_v1 = math.sqrt(sum(a**2 for a in v1))
    magnitude_v2 = math.sqrt(sum(b**2 for b in v2))
    return math.acos(dot_product / (magnitude_v1 * magnitude_v2))


## ----- Linear activation function
def linear_activation(activity_vector):
    return np.clip(activity_vector, 0, 1, out=activity_vector)


## ----- Sinusoidal function
def sinusoid(x, a, b, c, d):
    return a * np.sin(b * x + c) + d


## ----- Fit and extract signal shape parameters
def fit_sinusoid(activity_vector):
    x = np.arange(16)
    param_sinusoid, _ = curve_fit(sinusoid, x, activity_vector, p0=[-0.6, 0.8, -4.4, 0.4])
    return param_sinusoid


## ----- Compute r squared value
def sinusoid_r_squared(y_true, param):
    x = range(16)
    a = param[0]
    b = param[1]
    c = param[2]
    d = param[3]
    y_pred = a * np.sin(b * (x + c)) + d
    y_mean = np.mean(y_true)
    total_sum_of_squares = np.sum((y_true - y_mean) ** 2)
    residual_sum_of_squares = np.sum((y_true - y_pred) ** 2)
    r_squared = 1 - (residual_sum_of_squares / total_sum_of_squares)
    return r_squared


## ----- Update position with translational speed and orientation
def update_position(x,y,translational_speed, orientation):
    new_x = x + (translational_speed * math.cos(math.radians(orientation)))
    new_y = y + (translational_speed * math.sin(math.radians(orientation)))
    return new_x, new_y


## ----- Update orientation 
def update_orientation(orientation, rotational_speed, noise_deviation):
    random_component = random.gauss(0,noise_deviation)
    new_orientation = orientation + rotational_speed +  random_component
    return new_orientation % 360


## ----- Convert vector to angle
def get_angle(coo1,coo2):
    # Calculate the vector between the coordinates
    vector_x = coo2[0] - coo1[0]
    vector_y = coo2[1] - coo1[1]
    # Calculate the angle from the x-axis using arctan2
    angle_radians = math.atan2(vector_y, vector_x)
    # Convert radians to degrees
    angle_degrees = math.degrees(angle_radians)
    return angle_degrees


## ----- Activity heatmap
def activity_heatmap(activity_df):
    Act_df = activity_df.T
    sns.set_theme(style="whitegrid")
    # Processing IDs function
    def clean_ids(ids):
        return ids.split("-")[0] if "-" in ids else "".join(c for c in ids if not c.isdigit())
    # Clean all index labels
    cleaned_ids = [clean_ids(ids) for ids in Act_df.index]
    # Extract unique cleaned labels for y-axis ticks and sort them
    unique_ids = list(dict.fromkeys(cleaned_ids))
    # Remove undesired unique IDs
    undesired_ids = ["CIU", "TRr", "TRl", "TS"]
    # Plot
    unique_ids = [ids for ids in unique_ids if ids not in undesired_ids]
    fig, axs = plt.subplots(len(unique_ids), 1, figsize=(14, 7), sharex=True)
    for ax, unique_id in zip(axs, unique_ids):
        # Filter data for each unique label using boolean indexing
        subset_df = Act_df[Act_df.index.map(lambda x: clean_ids(x) == unique_id)]
        # Plot heatmap for the subset with dynamic height
        sns.heatmap(subset_df, vmin=0, vmax=1, cmap="inferno", ax=ax, cbar=False)
        # Remove y-axis labels but keep the tick bars
        ax.set(yticklabels=[])
        ax.set_ylabel(unique_id)
    # Set x-axis ticks
    plt.xticks(range(0, len(Act_df.columns), len(Act_df.columns) // 10),range(0, len(Act_df.columns), len(Act_df.columns) // 10))
    plt.xlabel("Simulation time")
    # Add colorbar to the right of the subplots
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
    cbar = plt.colorbar(axs[0].collections[0], cax=cbar_ax)


## ----- Graphical representation for stirring
def plot_stirring(Df, nest_size, food_list, paradigm, radius):
    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.25)
    # Initial time index
    initial_time = 0
    # Plot the agent journey
    line0, = ax.plot(Df[Df["Food"] == 0]["Y"], -Df[Df["Food"] == 0]["X"], linestyle="-", color="cyan")
    line1, = ax.plot(Df[Df["Food"] == 1]["Y"], -Df[Df["Food"] == 1]["X"], linestyle="-", color="pink")
    # Plot the nest size
    nest = plt.Circle((0, 0), nest_size, color="yellow", alpha=0.5)
    ax.add_patch(nest)
    # Check paradigm for border representation
    if paradigm in ["Till border exploration","Trial A: 1 PFN + 1 goal","Trial B: 1 PFN + 2 goals", "Trial C: 2 PFNs + 2 goals", "Test 1: 2hDs", "Test 2: 2hDs v.2", "Test 3: 2hDs v.3", "Test 4: 1hD", "Test 5: 2hDs + Plasticity", "Test 6: 3 goals + Plasticity"]:
        border = plt.Circle((0, 0), radius, color="grey", fill=False)
        ax.add_patch(border)
    # Check paradigm for food source representation
    if paradigm == "Food seeking":
        for f in range(len(food_list)):
            food_source = plt.Circle((food_list[f][1], -food_list[f][0]), food_list[f][2], color="lightgreen", alpha=0.5)
            ax.add_patch(food_source)
    # Check paradigm for goal directions
    if paradigm in ["Trial A: 1 PFN + 1 goal","Trial B: 1 PFN + 2 goals", "Trial C: 2 PFNs + 2 goals", "Test 1: 2hDs", "Test 2: 2hDs v.2", "Test 3: 2hDs v.3", "Test 4: 1hD", "Test 5: 2hDs + Plasticity"]:
        goal1 = np.deg2rad((6-1)*45)
        goal2 = np.deg2rad((8-1)*45)
        x1 = 200 * np.cos(goal1)
        y1 = 200 * np.sin(goal1)
        x2 = 200 * np.cos(goal2)
        y2 = 200 * np.sin(goal2)
        plt.scatter(-y1, -x1, color = "orange")
        plt.scatter(-y2, -x2, color="orange")
    elif paradigm == "Test 6: 3 goals + Plasticity":
        goal1 = np.deg2rad((8-1)*45)
        goal2 = np.deg2rad((4-1)*45)
        goal3 = np.deg2rad((5-1)*45)
        x1 = 200 * np.cos(goal1)
        y1 = 200 * np.sin(goal1)
        x2 = 200 * np.cos(goal2)
        y2 = 200 * np.sin(goal2)
        x3 = 200 * np.cos(goal3)
        y3 = 200 * np.sin(goal3)
        plt.scatter(-y1, -x1, color = "orange")
        plt.scatter(-y2, -x2, color="orange")
        plt.scatter(-y3, -x3, color="orange")
    elif paradigm == "Test 7: 2 goals + PI":
        goal1 = np.deg2rad((6-1)*45)
        goal2 = np.deg2rad((8-1)*45)
        x1 = 200 * np.cos(goal1)
        y1 = 200 * np.sin(goal1)
        x2 = 400 * np.cos(goal2)
        y2 = 400 * np.sin(goal2)
    # plot the graph
    plt.xlabel("X-coordinate", fontsize=16)
    plt.ylabel("Y-coordinate", fontsize=16)
    plt.axis('equal')
    plt.grid(True)
    # Slider for time
    ax_time = plt.axes([0.25, 0.1, 0.65, 0.05])
    time_slider = Slider(ax_time, 'Time', 0, len(Df) - 1, valinit=initial_time, valstep=1)
    # Update function for the slider
    shift = (Df["Food"] == 1).idxmax() if (Df["Food"] == 1).any() else Df.shape[0]
    def update(val):
        time_index = int(time_slider.val)
        if time_index < shift:
            line0.set_data(Df["Y"][:time_index], -Df["X"][:time_index])
            line1.set_data(0,0)
        else:
            line0.set_data(Df["Y"][:shift], -Df["X"][:shift])
            line1.set_data(Df["Y"][shift:time_index], -Df["X"][shift:time_index])
        fig.canvas.draw_idle()
    # Attach the update function to the slider
    time_slider.on_changed(update)


## ----- Graphical representation for fitted sinusoidal function
def sinusoid_plot(data, param):
    df = pd.DataFrame(data)
    plt.figure(figsize=(14, 8))
    # create violinplot
    sns.violinplot(data=df, palette="pastel", alpha=0.5)
    # Set labels and title
    plt.xticks(np.arange(len(df.columns)), df.columns)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.xlabel("Delta7 neuron ID")
    plt.ylabel("Firing rate")
    plt.ylim(0,1)
    # Add sinusoid function
    x = range(16)
    a = param[0]
    b = param[1]
    c = param[2]
    d = param[3]
    y = a * np.sin(b * (x + c)) + d
    sns.lineplot(y)


## ----- Compute circular median
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


## ----- Circular plot function
def circular_plot(data, trial):
    # Get angle values
    angles = []
    for t in range(trial):
        angles.append(get_angle((0,0),(data.iloc[-1, t * 2 + 1],-data.iloc[-1, t * 2]))+90)
    angles_radians = np.radians(angles)
    # Compute circular mean
    mean_angle = circ_median(angles_radians)
    # Compute circular standard deviation
    std_angle = circstd(angles_radians)
    # Create a circular plot
    fig = plt.figure()
    ax = fig.add_subplot(111, polar=True)
    # Plot the angles
    ax.scatter(angles_radians, np.ones_like(angles_radians), marker='o')
    # Plot the circular mean
    ax.plot([mean_angle, mean_angle], [0, 1], color='r', linewidth=2)
    # Add a circular standard deviation indicator
    ax.fill_between([mean_angle - std_angle, mean_angle + std_angle], 0, 1, color='orange', alpha=0.3)
    # Set the direction of 0 degrees to be counter-clockwise
    ax.set_theta_direction(1)
    # Set the zero location of the angles to be at the bottom
    ax.set_theta_zero_location('S')
    # Remove radial ticks
    ax.set_yticks([])
    with open("Saved_results/Last_goal_integration.csv", 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(angles)


## ----- Running simulation
def run_function(simulation_time, time_period, noise_deviation, nest_size, paradigm, timer, radius, food, ratio, trial, graphic):

    ''' Initialisation '''

    # import connectivity matrix neuron IDs list
    CON_MAT, COL_IDS = import_connectivity(paradigm)

    # Pre-simulation heating
    heating = 100

    # Create a dataframe to stock results
    trial_df = pd.DataFrame(0.0, index=range(simulation_time+heating+1), columns=range(2*trial))

    # Repeat for every trial
    for t in range(trial):
            
        # Initialise dataframes
        Df, Act, CON_MAT, COL_IDS = initialise_dataframes(COL_IDS, CON_MAT, simulation_time + heating, food, paradigm)
        expected_heading = pd.DataFrame(0.0, index=range(simulation_time + heating + 1), columns=(range(16)))

        # Get neurons index
        NEURON_IND = get_neuron_index(COL_IDS, food)

        # Create alternative matrix when no food 
        if paradigm in ["Timed exploration", "Till border exploration", "Food seeking"]:
            ALT_MAT = np.copy(CON_MAT)
            ALT_MAT[np.ix_(NEURON_IND["IND_PFL"],NEURON_IND["IND_HD"])] = 0

        # Create alternative matrices for paradigm Test 7
        if paradigm == "Test 7: 2 goals + PI":

            # Exploration of first goal
            
            
        # Initialise food sources
        food_list = initialise_food(paradigm, nest_size, food, radius)

        # Time loop
        for i in range(simulation_time + heating):

            ''' External input to neurons '''

            # Update CIU activity input
            if time_period == "Day" or (time_period == "Night" and i < simulation_time/2):
                Act.loc[i, "CIU" + CIU_activation(Df.loc[i, "Orientation"])] = 1

            # Save real orientation
            real_orientation = [0] * 8
            real_orientation[int(CIU_activation(Df.loc[i, "Orientation"]))-1] = 1
            expected_heading.iloc[i] = real_orientation * 2

            # Update TS activity input
            if paradigm in ["Timed exploration", "Till border exploration", "Food seeking"]:
                Act.loc[i, "TS"] = Df.loc[i, "Speed"]

            # Update TR activity input
            if i>5:
                Act.loc[i, "TRl"], Act.loc[i, "TRr"] = compare_headings(Df.loc[i-1, "Orientation"], Df.loc[i, "Orientation"])

            # Introduce goal directions to goal neurons 
            if paradigm == "Trial A: 1 PFN + 1 goal" and i > heating:
                Act.iloc[i, NEURON_IND["IND_G"]] = generate_goal(1, None, 6)
            if paradigm == "Trial B: 1 PFN + 2 goals" and i > heating:
                Act.iloc[i, NEURON_IND["IND_GA"]] = generate_goal(ratio, None, 6)
                Act.iloc[i, NEURON_IND["IND_GB"]] = generate_goal((1-ratio), None, 8)
            if paradigm in ["Trial C: 2 PFNs + 2 goals", "Test 1: 2hDs", "Test 2: 2hDs v.2", "Test 3: 2hDs v.3", "Test 4: 1hD", "Test 5: 2hDs + Plasticity"] and i > heating:
                Act.iloc[i, NEURON_IND["IND_GA"]] = generate_goal(ratio, None, 6)
                Act.iloc[i, NEURON_IND["IND_GB"]] = generate_goal((1-ratio), None, 8)
            if paradigm == "Test 6: 3 goals + Plasticity":
                Act.iloc[i, NEURON_IND["IND_GA"]] = generate_goal(ratio[0], None, 8)
                Act.iloc[i, NEURON_IND["IND_GB"]] = generate_goal(ratio[1], None, 4)
                Act.iloc[i, NEURON_IND["IND_GC"]] = generate_goal((1-ratio[0]-ratio[1]), None, 5)

            # Introduce Plasticity for hDelta connections
            if paradigm == "Test 5: 2hDs + Plasticity" and i > heating:
                list_hda = NEURON_IND["IND_HDA"][:]
                half1_hda = list_hda[:(len(list_hda)//2)]
                half2_hda = list_hda[(len(list_hda)//2):]
                list_hdb = NEURON_IND["IND_HDB"][:]
                half1_hdb = list_hdb[:(len(list_hdb)//2)]
                half2_hdb = list_hdb[(len(list_hdb)//2):]
                for a1 in half1_hda:
                    for b1 in half1_hdb:
                        difference = -Act.iloc[i, a1] - CON_MAT[b1, a1]
                        CON_MAT[b1, a1] += difference
                for a2 in half2_hda:
                    for b2 in half2_hdb:
                        difference = -Act.iloc[i, a2] - CON_MAT[b2, a2]
                        CON_MAT[b2, a2] += difference
                for b1 in half1_hdb:
                    for a1 in half1_hda:
                        difference = -Act.iloc[i, b1] - CON_MAT[a1, b1]
                        CON_MAT[a1, b1] += difference
                for b2 in half2_hdb:
                    for a2 in half2_hda:
                        difference = -Act.iloc[i, b2] - CON_MAT[a2, b2]
                        CON_MAT[a2, b2] += difference

            if paradigm == "Test 6: 3 goals + Plasticity" and i > heating:
                list_hda = NEURON_IND["IND_HDA"][:]
                half1_hda = list_hda[:(len(list_hda)//2)]
                half2_hda = list_hda[(len(list_hda)//2):]
                list_hdb = NEURON_IND["IND_HDB"][:]
                half1_hdb = list_hdb[:(len(list_hdb)//2)]
                half2_hdb = list_hdb[(len(list_hdb)//2):]
                list_hdc = NEURON_IND["IND_HDC"][:]
                half1_hdc = list_hdc[:(len(list_hdc)//2)]
                half2_hdc = list_hdc[(len(list_hdc)//2):]
                for a1 in half1_hda:
                    for b1 in half1_hdb:
                        difference = -Act.iloc[i, a1] - CON_MAT[b1, a1]
                        CON_MAT[b1, a1] += difference
                    for c1 in half1_hdc:
                        difference = -Act.iloc[i, a1] - CON_MAT[c1, a1]
                        CON_MAT[c1, a1] += difference
                for a2 in half2_hda:
                    for b2 in half2_hdb:
                        difference = -Act.iloc[i, a2] - CON_MAT[b2, a2]
                        CON_MAT[b2, a2] += difference
                    for c2 in half2_hdc:
                        difference = -Act.iloc[i, a2] - CON_MAT[c2, a2]
                        CON_MAT[c2, a2] += difference
                for b1 in half1_hdb:
                    for a1 in half1_hda:
                        difference = -Act.iloc[i, b1] - CON_MAT[a1, b1]
                        CON_MAT[a1, b1] += difference
                    for c1 in half1_hdc:
                        difference = -Act.iloc[i, b1] - CON_MAT[c1, b1]
                        CON_MAT[c1, b1] += difference
                for b2 in half2_hdb:
                    for a2 in half2_hda:
                        difference = -Act.iloc[i, b2] - CON_MAT[a2, b2]
                        CON_MAT[a2, b2] += difference
                    for c2 in half2_hdc:
                        difference = -Act.iloc[i, b2] - CON_MAT[c2, b2]
                        CON_MAT[c2, b2] += difference
                for c1 in half1_hdc:
                    for a1 in half1_hda:
                        difference = -Act.iloc[i, c1] - CON_MAT[a1, c1]
                        CON_MAT[a1, c1] += difference
                    for b1 in half1_hdb:
                        difference = -Act.iloc[i, c1] - CON_MAT[b1, c1]
                        CON_MAT[b1, c1] += difference
                for c2 in half2_hdc:
                    for a2 in half2_hda:
                        difference = -Act.iloc[i, c2] - CON_MAT[a2, c2]
                        CON_MAT[a2, c2] += difference
                    for b2 in half2_hdb:
                        difference = -Act.iloc[i, c2] - CON_MAT[b2, c2]
                        CON_MAT[b2, c2] += difference

            # Check if the agent has reached food depending on the paradigm
            if Df.loc[i,"Food"] == 0:

                # Paradigm 1
                if paradigm == "Timed exploration" and i>(heating + timer):
                    Df.loc[i:,"Food"] = 1

                # Paradigm 2
                elif paradigm == "Till border exploration" and euclidean_distance(0,0,Df.loc[i,"X"],Df.loc[i,"Y"])>radius:
                    Df.loc[i:,"Food"] = 1

                # Paradigm 3
                elif paradigm == "Food seeking":
                    for f in range(food):
                        if euclidean_distance(food_list[f][0],food_list[f][1],Df.loc[i,"X"],Df.loc[i,"Y"]) < food_list[f][2]:
                            Df.loc[i:,"Food"] = 1

                # Paradigm 4
                elif paradigm in ["Trial A: 1 PFN + 1 goal","Trial B: 1 PFN + 2 goals", "Trial C: 2 PFNs + 2 goals", "Test 1: 2hDs", "Test 2: 2hDs v.2", "Test 3: 2hDs v.3", "Test 4: 1hD", "Test 5: 2hDs + Plasticity", "Test 6: 3 goals + Plasticity"]:
                    Df.loc[i:,"Food"] = 1

            # Update activity vector depending on the inner state
            if Df.loc[i,"Food"] == 0:

                # Update new activity with no hd → PFL (Alternative connectivity matrix)
                Act.iloc[i+1] = linear_activation(np.dot(ALT_MAT, Act.iloc[i]))

            elif Df.loc[i,"Food"] == 1:
                
                # Update new activity with complete connectivity matrix
                Act.iloc[i+1] = linear_activation(np.dot(CON_MAT, Act.iloc[i]))

            # Update rotational speed from PFL neurons
            Df.loc[i+1,"Rotation"] = (Act.iloc[i+1, Act.columns.get_loc("PFL1"):Act.columns.get_loc("PFL8") + 1].sum() - Act.iloc[i+1, Act.columns.get_loc("PFL9"):Act.columns.get_loc("PFL16") + 1].sum()) * 10

            # Update Orientation and position
            Df.loc[i+1, "Orientation"] = update_orientation(Df.loc[i,"Orientation"],Df.loc[i+1,"Rotation"], noise_deviation)
            if i >= heating:
                new_x, new_y = update_position(Df.loc[i,"X"],Df.loc[i,"Y"],Df.loc[i,"Speed"],Df.loc[i+1,"Orientation"])
                Df.loc[i+1, "X"] = new_x
                Df.loc[i+1, "Y"] = new_y

            # # Get sinusoid d7 shape after heating
            # if i == heating:
            #     sin_list = []

            #     # Copy the dataframe for plotting
            #     centred_d7 = Act.iloc[:(heating), Act.columns.get_loc("d7-1"):Act.columns.get_loc("d7-16") + 1].copy()

            #     # Iterate over the whole heating activity Dataframe
            #     for j in range(4,heating):

            #         # Normalize the dataframe for plotting
            #         d7_list = centred_d7.iloc[j,:].tolist()
            #         while max(d7_list) != d7_list[3]:
            #             d7_list.append(d7_list.pop(0))
            #         centred_d7.iloc[j,:] = d7_list

            #     r_squared = 1
            #     while r_squared >= 1:

            #         # Fit the sinusoid function
            #         sin_param = fit_sinusoid(centred_d7.iloc[4:,:].median())

            #         # compute r squared value
            #         r_squared = sinusoid_r_squared(centred_d7.iloc[4:,:].median(), sin_param)

            # Stop simulation when the agent has returned to the nest
            if euclidean_distance(0,0,Df.loc[i+1, "X"],Df.loc[i+1, "Y"]) < nest_size and Df.loc[i,"Food"] == 1:
                Df = Df.iloc[:i+1,:]
                break

            # Stop simulating if the agent has reached the end of the paradigm
            if paradigm in ["Trial A: 1 PFN + 1 goal","Trial B: 1 PFN + 2 goals", "Trial C: 2 PFNs + 2 goals", "Test 1: 2hDs", "Test 2: 2hDs v.2", "Test 3: 2hDs v.3", "Test 4: 1hD", "Test 5: 2hDs + Plasticity", "Test 6: 3 goals + Plasticity"] and euclidean_distance(0,0,Df.loc[i+1, "X"],Df.loc[i+1, "Y"]) >= radius:
                Df = Df.iloc[:i+1,:]
                break

        # Save results of the trial
        trial_df[(t)*2] = Df["X"]
        trial_df[(t)*2+1] = Df["Y"]
        for k in range(i, trial_df.shape[0]):
            trial_df.iloc[k,(t)*2], trial_df.iloc[k,(t)*2+1] = Df.loc[i,"X"], Df.loc[i,"Y"]

    # Save results
    Act = Act.iloc[heating:(i+2)]
    Df = Df.iloc[heating:(i+3)]
    Act.to_csv("Saved_results/Last_activity.csv", sep="\t", index=False) # Only last one
    Df.to_csv("Saved_results/Agent_dataframe.csv", sep="\t", index=False) # Only last one
    trial_df.to_csv("Saved_results/Trial_dataframe.csv", sep="\t", index=False)


    crashtest = pd.DataFrame(CON_MAT)
    crashtest.to_excel('crashtest.xlsx', index=False, header=False)

    # Graphical output
    if graphic[0]==1:
        activity_heatmap(Act) # Only last one
    if graphic[1]==1:
        plot_stirring(Df, nest_size, food_list, paradigm, radius) # Only last one
    if graphic[2]==1:
        circular_plot(trial_df, trial)
    if graphic[3]==1:
        sinusoid_plot(centred_d7.iloc[4:,:], sin_param) # Only last one
    plt.show()

