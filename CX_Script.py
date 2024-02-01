#### Script for modelling and testing the CX
## Autor: Bastien Clémot
## python CX_Script.py
## Test: python CX_Script.py -CON Theorical_connectivity_matrix.csv -T 100


## ----- Import packages
import argparse
import csv
import customtkinter as ctk
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


## ----- Import arguments with argsparse
parser = argparse.ArgumentParser(description="Central Complexe Simulation.")
parser.add_argument("-CON",type=str,help="Path of the file with the connectivity matrix.")
parser.add_argument("-T",type=int,help="Simulation time.")
ARGS = parser.parse_args()


## ----- Import connectivity matrix
CON_MAT = np.genfromtxt(ARGS.CON, delimiter=',')


## ----- Initialise agent dataframe and neuron activity dataframe
def initialise_dataframes(time):
    # Agent dataframe
    Df = pd.DataFrame(0.0, index=range(time+1), columns=["X", "Y", "Orientation", "Speed", "Rotation"])
    Df.loc[0] = [0.0, 0.0, 0.0, 1.0, 1.0]
    # Activity dataframe
    with open("Neurons_IDs.csv", "r") as file:
        COL_IDS = next(csv.reader(file, delimiter=','))
    Act = pd.DataFrame(0.0, index=range(time+1), columns=COL_IDS)
    Act.loc[0.0, ["CIU1", "TS"]] = 1.0
    return Df, Act


## ----- Adjust orientation
def adjust_orientation(angle):
    return angle % 360


## ----- Logic activation function
def logic_activation(activity_vector, threshold):
    output = np.array(activity_vector, dtype=float) > threshold
    return output.astype(int)


## ----- Linear activation function
def linear_activation(activity_vector):
    return np.clip(activity_vector, 0, 1, out=activity_vector)


## ----- Matrix multiplication (activity propagation)
def matrix_multiplication(connectivity_matrix,activity_vector):
    return np.dot(connectivity_matrix,activity_vector)


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


## ----- Implementing GUI
if __name__ == "__main__":
    ctk.set_default_color_theme("green")
     
    # Call window
    root = ctk.CTk()
    root.geometry("750x450")
    root.title("CX simulation")

    # Lable widget
    title_label = ctk.CTkLabel(root, text="• Central Complex Control Panel •", font=ctk.CTkFont(size=30, weight="bold"))
    title_label.pack(pady=20)


    # Main loop
    root.mainloop()


## ----- Runing simulation
# if __name__ == "__main__":
#     # Initialisation
#     Df, Act = initialise_dataframes(ARGS.T)
#     # Time loop
#     for i in range(ARGS.T):
#         # Update new activity
#         Act.iloc[i+1] = linear_activation(np.dot(CON_MAT, Act.iloc[i]))
#     activity_heatmap(Act)

