#### Script for modelling and testing the CX
## Autor: Bastien Clémot
## python CX_Script.py
## Test: python CX_Script.py -CON Theorical_connectivity_matrix.csv -T 100


## ----- Import packages
import argparse
import csv
import numpy as np
import pandas as pd


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
    Df = pd.DataFrame(0, index=range(time+1), columns=["X", "Y", "Orientation", "Speed", "Rotation"])
    Df.loc[0] = [0, 0, 0, 1, 1]
    # Activity dataframe
    with open("Neurons_IDs.csv", "r") as file:
        COL_IDS = next(csv.reader(file, delimiter=','))
    Act = pd.DataFrame(0, index=range(time+1), columns=COL_IDS)
    return Df, Act


## ----- Adjust orientation
def adjust_orientation(angle):
    return angle % 360


## ----- Logic activation function
def logic_activation(activity_vector, threshold):
    output = np.array(activity_vector, dtype=float) > threshold
    return output.astype(int)


## ----- 


## ----- Final running function
def run_simulation(connectivity_matrix, activity_vector, time):
    
    # Matrix multiplication
    result = activity_vector.copy()
    for _ in range(time):
        result = np.matmul(connectivity_matrix, result)

    return result