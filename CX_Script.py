#### Script for modelling and testing the CX
## Autor: Bastien Clémot
## python CX_Script.py



## ----- Import packages
import numpy as np
import argparse



## ----- Import arguments with argsparse
parser = argparse.ArgumentParser(description="Central Complexe Simulation.")
parser.add_argument("-connectivity_file",type=str,help="Path of the file with the connectivity matrix.")
parser.add_argument("-activity_file",type=str,help="Path of the file with the ativity vector.")
args = parser.parse_args()



## ----- Import connectivity matrix and activity vector
# Connectivity matrix
CON_MAT = np.genfromtxt(args.connectivity_file, delimiter=',', skip_header=1)
# Activity vector
ACT_IN = np.genfromtxt(args.activity_file, delimiter=',', skip_header=1)

# print("DEBUG - File import connectivity:")
# print(CON_MAT)
# print("DEBUG - File import activity:")
# print(ACT_IN)



## ----- Function for activity transmission
def transmission(connectivity_matrix, activity_vector):
    return(np.dot(connectivity_matrix, activity_file))