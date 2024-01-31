#### Script for modelling and testing the CX
## Autor: Bastien Clémot
## python CX_Script.py
## Test: python CX_Script.py -CON Theorical_connectivity_matrix.csv -T 100


## ----- Import packages
import csv
import numpy as np
import pandas as pd
import argparse


## ----- Import arguments with argsparse
parser = argparse.ArgumentParser(description="Central Complexe Simulation.")
parser.add_argument("-CON",type=str,help="Path of the file with the connectivity matrix.")
parser.add_argument("-T",type=int,help="Simulation time.")
ARGS = parser.parse_args()


## ----- Import connectivity matrix and activity vector
CON_MAT = np.genfromtxt(ARGS.CON, delimiter=',')


## ----- Initialize dataframe for position and orientation
Df = pd.DataFrame(np.zeros((ARGS.T+1,5)),columns=["X","Y","Orientation","Speed","Rotation"])
Df.loc[0] = [0,0,0,1,1]


## ----- Initialize activity dataframe
file = open("Neurons_IDs.csv", "r")
COL_IDS = list(csv.reader(file, delimiter=','))
file.close()
Act = pd.DataFrame(np.zeros((ARGS.T+1,CON_MAT.shape[0])), columns=COL_IDS)


## ----- Final running function
def run_simulation(connectivity_matrix, activity_vector, time):
    
    # Matrix multiplication
    result = activity_vector.copy()
    for _ in range(time):
        result = np.matmul(connectivity_matrix, result)

    return result

# print("DEBUG - Post simulation:")
# print(run_simulation(CON_MAT,ACT_IN, args.t))