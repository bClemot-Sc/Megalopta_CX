#### Script for modelling and testing the CX
## Autor: Bastien Clémot


## ----- Import packages
import csv
import Eddit_matrix
import math
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import numpy as np
import pandas as pd
import random
from scipy.optimize import curve_fit
from scipy.stats import circmean, circstd
import seaborn as sns


## ----- Initialise agent dataframe and neuron activity dataframe
def initialise_dataframes(ids_list,time):
    # Agent dataframe
    agent_df = pd.DataFrame(0.0, index=range(time+1), columns=["X", "Y", "Orientation", "Speed", "Rotation", "Food"])
    # Set speed to 1 for the whole simulation
    agent_df["Speed"] = 1.0
    agent_df["Food"] = 0
    # Activity dataframe
    activity_df = pd.DataFrame(0.0, index=range(time+1), columns=ids_list)
    return agent_df, activity_df


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


## ----- Generate goal directions
def generate_goal(ratio, param):
    # Generate sinusoidal shape
    x = range(16)
    a = param[0]
    b = param[1]
    c = param[2]
    d = param[3]
    y = a * np.sin(b * (x + c)) + d
    # Adjust ratio
    goal1 = list(y * ratio)
    goal2 = list(y * (1-ratio))
    # Adjust position (goal1 at 7, goal2 at 5)
    while max(goal1) != goal1[6]:
        goal1.append(goal1.pop(0))
    while max(goal2) != goal2[4]:
        goal2.append(goal2.pop(0))
    return [goal1[i] + goal2[i] for i in range(len(goal1))]


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


## ----- Compute euclidian distance between two points
def euclidian_distance(x1,y1,x2,y2):
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)


## ----- Adress heading direction (relative to a South landscape cue) to CIU neurons
def CIU_activation(heading_direction):
    relative_heading = (-heading_direction) % 360
    heading_list = [0, 45, 90, 135, 180, 225, 270, 315, 360]
    closest_heading = min(heading_list, key=lambda x: abs(x - relative_heading))
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
        TRl = 1
    else:
        TRr = 1
    return TRl, TRr


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
    sns.set(style="whitegrid")
    # Processings IDs function
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
    if paradigm == "Till border exploration" or "Simple double goals":
        border = plt.Circle((0, 0), radius, color="grey", fill=False)
        ax.add_patch(border)
    # Check paradigm for food source representation
    if paradigm == "Food seeking":
        for f in range(len(food_list)):
            food_source = plt.Circle((food_list[f][1], -food_list[f][0]), food_list[f][2], color="lightgreen", alpha=0.5)
            ax.add_patch(food_source)
    # Check paradigm for goal directions
    if paradigm == "Simple double goals":
        plt.scatter(-radius, 0, color = "orange")
        plt.scatter(0, radius, color="orange")
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


## ----- Circular plot function
def circular_plot(data, trial):
    # Get angle values
    angles = []
    for t in range(trial):
        angles.append(get_angle((0,0),(data.iloc[-1, t * 2],data.iloc[-1, t * 2 + 1])))
    angles_radians = np.radians(angles)
    # Compute circular mean
    mean_angle = circmean(angles_radians)
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
    # Set the direction of 0 degrees to be counterclockwise
    ax.set_theta_direction(1)
    # Set the zero location of the angles to be at the bottom
    ax.set_theta_zero_location('S')
    # Remove radial ticks
    ax.set_yticks([])
    # Show the plot


## ----- Runing simulation
def run_function(simulation_time, time_period, noise_deviation, nest_size, paradigm, timer, radius, food, ratio, trial, graphic):

    # import connectivity matrix
    if paradigm == "Simple double goals":
        Eddit_matrix.eddit_matrix("Simplified_connectivity_matrices.xlsx")
        CON_MAT = np.genfromtxt("Theorical_connectivity_matrix.csv", delimiter=",")
        with open("Neurons_IDs.csv", "r") as file:
            COL_IDS = next(csv.reader(file, delimiter=","))
    else: 
        Eddit_matrix.eddit_matrix("Theorical_connectivity_matrices.xlsx")
        CON_MAT = np.genfromtxt("Theorical_connectivity_matrix.csv", delimiter=",")
        with open("Neurons_IDs.csv", "r") as file:
            COL_IDS = next(csv.reader(file, delimiter=","))

    # Get IDs index
    IND_PEN = [i for i, element in enumerate(COL_IDS) if "PEN" in element]
    IND_EPG = [i for i, element in enumerate(COL_IDS) if "EPG" in element]
    IND_PEG = [i for i, element in enumerate(COL_IDS) if "PEG" in element]
    IND_TR = [i for i, element in enumerate(COL_IDS) if "TR" in element]
    IND_D7 = [i for i, element in enumerate(COL_IDS) if "d7-" in element]
    IND_CIU = [i for i, element in enumerate(COL_IDS) if "CIU" in element]
    IND_PFN = [i for i, element in enumerate(COL_IDS) if "PFN" in element]
    IND_PFL = [i for i, element in enumerate(COL_IDS) if "PFL" in element]
    IND_HD = [i for i, element in enumerate(COL_IDS) if "hd" in element]
    IND_TS = [i for i, element in enumerate(COL_IDS) if "TS" in element]

    # Create alternative matrix when no food 
    ALT_MAT = np.copy(CON_MAT)
    ALT_MAT[np.ix_(IND_PFL,IND_HD)] = 0

    # Create a dataframe to stock results
    heating = 100
    trial_df = pd.DataFrame(0.0, index=range(simulation_time+heating+1), columns=range(2*trial))

    # Repeat for every trial
    for t in range(trial):

        # Initialisation
        Df, Act = initialise_dataframes(COL_IDS,simulation_time + heating)
        expected_EPG = pd.DataFrame(0.0, index=range(simulation_time + heating + 1), columns=(range(16)))

        food_list = []
        # Initialize food sources
        if paradigm == "Food seeking":

            # randomly create food sources (x,y,radius)  
            for f in range(food):
                f_x = random.choice([random.randint(-200, -nest_size-50), random.randint(nest_size+50, 200)])
                f_y = random.choice([random.randint(-200, -nest_size-50), random.randint(nest_size+50, 200)])
                f_r = random.randint(5, 20)
                food_list.append((f_x,f_y,f_r))

        # Time loop
        for i in range(simulation_time + heating):

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
            if i>5:
                Act.loc[i, "TRl"], Act.loc[i, "TRr"] = compare_headings(Df.loc[i-1, "Orientation"], Df.loc[i, "Orientation"])

            # Introduce goal directions to PFL neurons 
            if paradigm == "Simple double goals" and i > heating:
                Act.iloc[i, Act.columns.get_loc("PFN1"):Act.columns.get_loc("PFN16") + 1] = generate_goal(ratio, sin_param)

            # Check if the agent has reached food depending on the paradigm
            if Df.loc[i,"Food"] == 0:

                # Paradigm 1
                if paradigm == "Timed exploration" and i>(heating + timer):
                    Df.loc[i:,"Food"] = 1

                # Paradigm 2
                elif paradigm == "Till border exploration" and euclidian_distance(0,0,Df.loc[i,"X"],Df.loc[i,"Y"])>radius:
                    Df.loc[i:,"Food"] = 1

                # Paradigm 3
                elif paradigm == "Food seeking":
                    for f in range(food):
                        if euclidian_distance(food_list[f][0],food_list[f][1],Df.loc[i,"X"],Df.loc[i,"Y"]) < food_list[f][2]:
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
            if paradigm != "Debug rotation":
                Df.loc[i+1, "Orientation"] = update_orientation(Df.loc[i,"Orientation"],Df.loc[i+1,"Rotation"], noise_deviation)
            elif i > int(heating + simulation_time/2):
                Df.loc[i+1, "Orientation"] = noise_deviation
            if i >= heating:
                if i == heating and paradigm != "Simple double goals":
                    Act.iloc[i+1, Act.columns.get_loc("hd1"):Act.columns.get_loc("hd16") + 1] = [0] * 16
                new_x, new_y = update_position(Df.loc[i,"X"],Df.loc[i,"Y"],Df.loc[i,"Speed"],Df.loc[i+1,"Orientation"])
                Df.loc[i+1, "X"] = new_x
                Df.loc[i+1, "Y"] = new_y

            # Get sinusoid d7 shape after heating
            if i == heating:
                sin_list = []

                # Copy the dataframe for ploting
                centered_d7 = Act.iloc[:(heating), Act.columns.get_loc("d7-1"):Act.columns.get_loc("d7-16") + 1].copy()

                # Iterate over the whole heating activity Dataframe
                for j in range(4,heating):

                    # Normalize the dataframe for ploting
                    d7_list = centered_d7.iloc[j,:].tolist()
                    while max(d7_list) != d7_list[3]:
                        d7_list.append(d7_list.pop(0))
                    centered_d7.iloc[j,:] = d7_list

                r_squared = 1
                while r_squared >= 1:

                    # Fit the sinusoid function
                    sin_param = fit_sinusoid(centered_d7.iloc[4:,:].median())

                    # compute r squared value
                    r_squared = sinusoid_r_squared(centered_d7.iloc[4:,:].median(), sin_param)

            # Stop simulation when the agent has returned to the nest
            if euclidian_distance(0,0,Df.loc[i+1, "X"],Df.loc[i+1, "Y"]) < nest_size and Df.loc[i,"Food"] == 1:
                Df = Df.iloc[:i+1,:]
                break

            # Stop simulating if the agent has reached the end of the paradigm
            if paradigm == "Simple double goals" and euclidian_distance(0,0,Df.loc[i+1, "X"],Df.loc[i+1, "Y"]) >= radius:
                Df = Df.iloc[:i+1,:]
                break

        # Save results of the trial
        trial_df[(t)*2] = Df["X"]
        trial_df[(t)*2+1] = Df["Y"]
        for k in range(i, trial_df.shape[0]):
            trial_df.iloc[k,(t)*2], trial_df.iloc[k,(t)*2+1] = Df.loc[i,"X"], Df.loc[i,"Y"]

    # Graphical output
    Act = Act.iloc[heating:(i+2)]
    Df = Df.iloc[heating:(i+3)]
    if graphic[0]==1:
        activity_heatmap(Act) # Only last one
    if graphic[1]==1:
        plot_stirring(Df, nest_size, food_list, paradigm, radius) # Only last one
    if graphic[2]==1:
        circular_plot(trial_df, trial)
    if graphic[3]==1:
        sinusoid_plot(centered_d7.iloc[4:,:], sin_param) # Only last one
    plt.show()

