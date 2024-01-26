#### Script for modelling and testing the CX
## Autor: Bastien Clémot
## python CX_Script.py


## ----- Import packages
import numpy as np
import argparse


## ----- Import arguments with argsparse
parser = argparse.ArgumentParser(description="Central Complexe Simulation.")
parser.add_argument("-mode",type=str,default="import",help="Simulation mode. Either 'import' or 'test'.")
parser.add_argument("-connectivity_file",type=str,default="none",help="Path of the file with the connectivity matrix.")
args = parser.parse_args()


## ----- Stop program + error if wrong inputs
if args.mode == "import" :
    if not os.path.exists(args.connectivity_file) :
        print("INPUT ERROR: File path doesn't exist.")
        sys.exit()


## ----- TEST Mode: Randomly generate connectivity matrix
def random_matrix() :
    DENSITY = 0.5
    NODE_NAMES = ['A', 'B', 'C', 'D']
    Matrix = np.zeros(len(NODE_NAMES))
    for i in range(len(NODE_NAMES)):
        for j in range(len(NODE_NAMES)):
            if np.random.rand() < DENSITY:
                Matrix[i, j] = 1
                Matrix[j, i] = 1
    return Matrix


## ----- IMPORT Mode: Import connectivity matrix from file
def import_matrix() :
    MATRIX = np.loadtxt(file_path, delimiter=',') 
    return MATRIX