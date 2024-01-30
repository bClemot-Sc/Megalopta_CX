#### Script for modelling and testing the CX
## Autor: Bastien Clémot
## python CX_Script.py
## Test: python CX_Script.py -CON Theorical_connectivity_matrix.csv -T 100


## ----- Import packages
import numpy as np
import pandas as pd
import argparse


## ----- Import arguments with argsparse
parser = argparse.ArgumentParser(description="Central Complexe Simulation.")
parser.add_argument("-CON",type=str,help="Path of the file with the connectivity matrix.")
parser.add_argument("-T",type=int,help="Simulation time.")
ARGS = parser.parse_args()


## ----- Import connectivity matrix and activity vector
CON_MAT = np.genfromtxt(ARGS.CON, delimiter=',', skip_header=1)


## ----- Initialize dataframe for position and orientation
Dataframe = pd.DataFrame(np.zeros((ARGS.T+1,5)),columns=["X","Y","Orientation","Speed","Rotation"])
Dataframe.loc[0] = [0,0,0,1,1]


## ----- Final running function
def run_simulation(connectivity_matrix, activity_vector, time):
    
    # Matrix multiplication
    result = activity_vector.copy()
    for _ in range(time):
        result = np.matmul(connectivity_matrix, result)

    return result

# print("DEBUG - Post simulation:")
# print(run_simulation(CON_MAT,ACT_IN, args.t))