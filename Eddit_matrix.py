#### Script for opening and editing the theorical connectivity matrix
#### As well as edditing and opening the IDs list
## Autor: Bastien Clémot
## python Eddit_matrix.py


## ----- Import packages
import pandas as pd
import csv

## ----- Open the Excel sheets
MATRIX = pd.read_excel("Theorical_connectivity_matrices.xlsx", sheet_name="Global", header=None)
IDS = pd.read_excel("Theorical_connectivity_matrices.xlsx", sheet_name="IDs", header=None)


## ----- Transpose the connectivity matrix, convert the IDs to a list
T_MATRIX = MATRIX.T
IDS_LIST = IDS.stack().dropna().tolist()


## ----- Rewrite the transposed matrix as a .csv file
T_MATRIX.to_csv("Theorical_connectivity_matrix.csv", header=False, index=False)
with open("Neurons_IDs.csv", 'w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerow(IDS_LIST)


## ----- End message
print("Eddit has been successfull.")