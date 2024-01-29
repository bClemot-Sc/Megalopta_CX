#### Script for modelling and testing the CX
## Autor: Bastien Clémot
## python CX_Script.py
## Test: python CX_Script.py -con connectivity_matrix.csv -act activity_vector.csv -t 2



## ----- Import packages
import numpy as np
import argparse



## ----- Import arguments with argsparse
parser = argparse.ArgumentParser(description="Central Complexe Simulation.")
parser.add_argument("-con",type=str,help="Path of the file with the connectivity matrix.")
parser.add_argument("-act",type=str,help="Path of the file with the ativity vector.")
parser.add_argument("-t",type=int,help="Simulation time.")
args = parser.parse_args()



## ----- Import connectivity matrix and activity vector
# Connectivity matrix
CON_MAT = np.genfromtxt(args.con, delimiter=',', skip_header=1)
# Activity vector
ACT_IN = np.genfromtxt(args.act, delimiter=',', skip_header=1)

# print("DEBUG - File import connectivity:")
# print(CON_MAT)
# print("DEBUG - File import activity:")
# print(ACT_IN)



## ----- Final running function
def run_simulation(connectivity_matrix, activity_vector, time):
    
    # Matrix multiplication
    result = activity_vector.copy()
    for _ in range(time):
        result = np.matmul(connectivity_matrix, result)

    return result

# print("DEBUG - Post simulation:")
# print(run_simulation(CON_MAT,ACT_IN, args.t))