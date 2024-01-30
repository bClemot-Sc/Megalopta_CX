#### Script for opening and editing the theorical connectivity matrix
## Autor: Bastien Clémot
## python Eddit_matrix.py


## ----- Import packages
import pandas as pd


## ----- Open the Excel sheet
matrix = pd.read_excel("Theorical_connectivity_matrices.xlsx", sheet_name="Global")


## ----- Transpose the connectivity matrix
t_matrix = matrix.T


## ----- Rewrite the transposed matrix as a .csv file
t_matrix.to_csv("Theorical_connectivity_matrix.csv", header=False, index=False)


## ----- End message
print("Eddit has been successfull.")