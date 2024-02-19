#### Script for modelling and testing the CX
## Autor: Bastien Clémot


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


## ----- Get IDs index
IND_PFL = [i for i, element in enumerate(COL_IDS) if "PFL" in element]
IND_HD = [i for i, element in enumerate(COL_IDS) if "hd" in element]


## ----- Create alternative matrix for exploration when no food (no hd → PFL)
ALT_MAT = np.copy(CON_MAT)
for i in IND_PFL:
    ALT_MAT[i,IND_HD] = 0


## ----- Initialise agent dataframe and neuron activity dataframe
def initialise_dataframes(ids_list,time):
    # Agent dataframe
    agent_df = pd.DataFrame(0.0, index=range(time+1), columns=["X", "Y", "Orientation", "Speed", "Rotation"])
    # Set speed to 1 for the whole simulation
    agent_df["Speed"] = 1.0
    # Activity dataframe
    activity_df = pd.DataFrame(0.0, index=range(time+1), columns=ids_list)
    return agent_df, activity_df


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
    heading_id = heading_list.index(closest_heading % 360) + 1
    return str(heading_id)


## ----- Adress turning direction to TR neurons
def compare_headings(previous_heading, new_heading):
    TRr = 0
    TRl = 0
    heading_difference = (new_heading - previous_heading) % 360
    if heading_difference == 0:
        pass
    elif heading_difference <= 180:
        TRl = 0.05
    else:
        TRr = 0.05
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
    return new_orientation % 360


## ----- Activity heatmap
def activity_heatmap(activity_df):
    Act_df = activity_df.T
    sns.set(style="whitegrid")
    # Processings IDs function
    def clean_ids(ids):
        return ids.split("-")[0] if "-" in ids else ''.join(c for c in ids if not c.isdigit())
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
    expected_EPG = pd.DataFrame(0.0, index=range(simulation_time+1), columns=(range(16)))

    # Time loop
    for i in range(simulation_time):

        # Update CIU activity input
        if time_period == "Day" or (time_period == "Night" and i < simulation_time/2):
            Act.loc[i, "CIU" + CIU_activation(Df.loc[i, "Orientation"])] = 1

        # Save real orientation
        real_orientation = [0] * 8
        real_orientation[int(CIU_activation(Df.loc[i, "Orientation"]))-1] = 1
        expected_EPG.iloc[i] = real_orientation * 2

        # Update TS activity input (should be improved)
        Act.loc[i, "TS"] = Df.loc[i, "Speed"]

        # Update TR activity input (should be improved)
        if i==0:
            pass
        else:
            Act.loc[i, "TRl"], Act.loc[i, "TRr"] = compare_headings(Df.loc[i-1, "Orientation"], Df.loc[i, "Orientation"])

        # Activate homing vector halfway through the simulation
        if (i < simulation_time/2):

            # Update new activity with no hd → PFL (Alternative connectivity matrix)
            if activation_function == "Linear":
                Act.iloc[i+1] = linear_activation(np.dot(ALT_MAT, Act.iloc[i]))
            if activation_function == "Logic":
                Act.iloc[i+1] = logic_activation(np.dot(ALT_MAT, Act.iloc[i]), threshold)

        else:

            # Update new activity with complete connectivity matrix
            if activation_function == "Linear":
                Act.iloc[i+1] = linear_activation(np.dot(CON_MAT, Act.iloc[i]))
            if activation_function == "Logic":
                Act.iloc[i+1] = logic_activation(np.dot(CON_MAT, Act.iloc[i]), threshold)

        # Update rotational speed from PFL neurons
        Df.loc[i+1,"Rotation"] = Act.iloc[i+1, Act.columns.get_loc("PFL1"):Act.columns.get_loc("PFL8") + 1].sum() - Act.iloc[i+1, Act.columns.get_loc("PFL9"):Act.columns.get_loc("PFL16") + 1].sum()

        # Update Orientation and position
        Df.loc[i+1, "Orientation"] = update_orientation(Df.loc[i,"Orientation"],Df.loc[i+1,"Rotation"], noise_factor)
        new_x, new_y = update_position(Df.loc[i,"X"],Df.loc[i,"Y"],Df.loc[i,"Speed"],Df.loc[i+1,"Orientation"])
        Df.loc[i+1, "X"] = new_x
        Df.loc[i+1, "Y"] = new_y

    # Graphical output
    activity_heatmap(Act)
    plot_stirring(Df)
