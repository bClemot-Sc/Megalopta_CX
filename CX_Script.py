#### Script for modelling and testing the CX
## Autor: Bastien Clémot
## python CX_Script.py
## Test: python CX_Script.py -CON Theorical_connectivity_matrix.csv -T 100


## ----- Import packages
import csv
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
import seaborn as sns


## ----- Import connectivity matrix and IDs list
CON_MAT = np.genfromtxt("Theorical_connectivity_matrix.csv", delimiter=',')
with open("Neurons_IDs.csv", "r") as file:
        COL_IDS = next(csv.reader(file, delimiter=','))


## ----- Initialise agent dataframe and neuron activity dataframe
def initialise_dataframes(ids_list,time):
    # Agent dataframe
    Df = pd.DataFrame(0.0, index=range(time+1), columns=["X", "Y", "Orientation", "Speed", "Rotation"])

    # TO REMOVE LATER
    Df["Speed"] = 1.0

    # Activity dataframe
    Act = pd.DataFrame(0.0, index=range(time+1), columns=ids_list)
    return Df, Act


## ----- Adjust orientation
def adjust_orientation(angle):
    return angle % 360


## ----- Matrix multiplication (activity propagation)
def matrix_multiplication(connectivity_matrix,activity_vector):
    return np.dot(connectivity_matrix,activity_vector)


## ----- Linear activation function
def linear_activation(activity_vector):
    return np.clip(activity_vector, 0, 1, out=activity_vector)


## ----- Logic activation function
def logic_activation(activity_vector, threshold):
    output = np.array(activity_vector, dtype=float) > threshold
    return output.astype(int)


## ----- Adress heading direction to CIU neurons
def CIU_activation(heading_direction):
    heading_list = [0, 45, 90, 135, 180, 225, 270, 315, 360]
    closest_heading = min(heading_list, key=lambda x: abs(x - heading_direction))
    heading_id = heading_list.index(adjust_orientation(closest_heading)) + 1
    return str(heading_id)


## ----- Adress turning direction to TR neuronss
def compare_headings(previous_heading, new_heading):
    TRr = 0
    TRl = 0
    heading_difference = (new_heading - previous_heading) % 360
    if heading_difference == 0:
        pass
    elif heading_difference <= 180:
        TRl = 1.0
    else:
        TRr = 1.0
    return TRl, TRr


## ----- Update position with translational speed and orientation
def update_position(x,y,translational_speed, orientation):
    new_x = x + (translational_speed * math.cos(math.radians(orientation)))
    new_y = y + (translational_speed * math.sin(math.radians(orientation)))
    return new_x, new_y


## ----- Update orientation 
def update_orientation(orientation, rotational_speed, noise_factor):
    random_component = random.gauss(0,45)
    new_orientation = orientation + (rotational_speed + noise_factor * random_component)
    return adjust_orientation(new_orientation)


## ----- Activity heatmap
def activity_heatmap(activity_df):
    Act_df = activity_df.T
    sns.set(style="whitegrid")

    # Clean all index labels
    cleaned_ids = [clean_ids(ids) for ids in Act_df.index]
    # Extract unique cleaned labels for y-axis ticks and sort them
    unique_ids = list(dict.fromkeys(cleaned_ids))
    # Remove undesired unique IDs
    undesired_ids = ["CIU", "TRr", "TRl", "TS"]
    unique_ids = [ids for ids in unique_ids if ids not in undesired_ids]

    fig, axs = plt.subplots(len(unique_ids), 1, figsize=(14, 7), sharex=True)
    for ax, unique_id in zip(axs, unique_ids):
        # Filter data for each unique label using boolean indexing
        subset_df = Act_df[Act_df.index.map(lambda x: clean_ids(x) == unique_id)]
        # Plot heatmap for the subset with dynamic height
        sns.heatmap(subset_df, cmap="inferno", ax=ax, cbar=False)
        # Remove y-axis labels but keep the tick bars
        ax.set(yticklabels=[])
        ax.set_ylabel(unique_id)
    # Set x-axis ticks
    plt.xticks(range(0, len(Act_df.columns), len(Act_df.columns) // 10),range(0, len(Act_df.columns), len(Act_df.columns) // 10))
    plt.xlabel("Simulation time")
    # Add colorbar to the right of the subplots
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
    cbar = plt.colorbar(axs[0].collections[0], cax=cbar_ax)
    plt.show()


## ----- Cleaning IDs for the heatmap
def clean_ids(ids):
    return ids.split("-")[0] if "-" in ids else ''.join(c for c in ids if not c.isdigit())


## ----- Graphical representation for stirring
def plot_stirring(Df):
    plt.plot(Df['X'], Df['Y'], linestyle='-')
    plt.scatter(Df['X'].iloc[0], Df['Y'].iloc[0], color='lightgreen', label='Start')
    plt.scatter(Df['X'].iloc[-1], Df['Y'].iloc[-1], color='red', label='End')
    plt.xlabel('X-coordinate')
    plt.ylabel('Y-coordinate')
    plt.grid(True)
    plt.show()


## ----- Runing simulation
def run_function(connectivity_matrix, simulation_time, activation_function, time_period, noise_factor, threshold):
    # Initialisation
    Df, Act = initialise_dataframes(COL_IDS,simulation_time)

    # Time loop
    for i in range(simulation_time):
        # Update CIU activity input
        if time_period == "Day" or (time_period == "Night" and i < simulation_time/2):
            Act.loc[i, "CIU" + CIU_activation(Df.loc[i, "Orientation"])] = 1
        # Update TS activity input (should be improved)
        Act.loc[i, "TS"] = Df.loc[i, "Speed"]
        # Update TR activity input (should be improved)
        if i==0:
            pass
        else:
            Act.loc[i, "TRl"], Act.loc[i, "TRr"] = compare_headings(Df.loc[i-1, "Orientation"], Df.loc[i, "Orientation"])
        # Update new activity
        if activation_function == "Linear":
            Act.iloc[i+1] = linear_activation(np.dot(CON_MAT, Act.iloc[i]))
        if activation_function == "Logic":
            Act.iloc[i+1] = logic_activation(np.dot(CON_MAT, Act.iloc[i]), threshold)
        # Update Orientation and position
        Df.loc[i+1, "Orientation"] = update_orientation(Df.loc[i,"Orientation"],Df.loc[i,"Rotation"], noise_factor)
        new_x, new_y = update_position(Df.loc[i,"X"],Df.loc[i,"Y"],Df.loc[i,"Speed"],Df.loc[i,"Orientation"])
        Df.loc[i+1, "X"] = new_x
        Df.loc[i+1, "Y"] = new_y

    # Graphical output
    activity_heatmap(Act)
    plot_stirring(Df)

