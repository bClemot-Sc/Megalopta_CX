#### Script for modelling and testing the CX
## Autor: Bastien Clémot
## python CX_Script.py
## Test: python CX_Script.py -CON Theorical_connectivity_matrix.csv -T 100


## ----- Import packages
import argparse
import csv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
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


## ----- Matrix multiplication (activity propagation)
def matrix_multiplication(connectivity_matrix,activity_vector):
    return np.dot(connectivity_matrix,activity_vector)


## ----- Activity heatmap
def activity_heatmap(activity_df):
    sns.set(style="whitegrid")
    plt.figure(figsize=(10,8))
    sns.heatmap(Act.T, cmap="viridis")
    plt.xlabel("Simulation time")
    plt.ylabel("")
    plt.title("Evolution of neuronal activity")
    plt.show()


## ----- Runing simulation
if __name__ == "__main__":

    # Initialisation
    Df, Act = initialise_dataframes(ARGS.T)

    # Time loop
    for i in range(ARGS.T):

        # Update new activity
        Act.iloc[i+1] = np.dot(CON_MAT, Act.iloc[i])

    activity_heatmap(Act)

