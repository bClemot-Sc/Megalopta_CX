#### Script for opening and editing the theorical connectivity matrix
#### As well as edditing and opening the IDs list
## Autor: Bastien Clémot
## python Eddit_matrix.py


## ----- Import packages
import pandas as pd
import csv

## ----- Eddit function
def eddit_matrix(path):

    # Open Excel sheets
    MATRIX = pd.read_excel(path, sheet_name="Global", header=None)
    IDS = pd.read_excel(path, sheet_name="IDs", header=None)

    # Transpose connectivity matrix and convert IDs to list
    T_MATRIX = MATRIX.T
    IDS_LIST = IDS.stack().dropna().tolist()

    # Rewrite the transposed matrix as a .csv file
    T_MATRIX.to_csv("Theorical_connectivity_matrix.csv", header=False, index=False)
    with open("Neurons_IDs.csv", 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(IDS_LIST)

