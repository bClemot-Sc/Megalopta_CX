

## ----- Import packages
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

## ----- Import connectivity matrix
path = "Connectivity_matrices/Test_7_matrices.xlsx"
MATRIX = pd.read_excel(path, sheet_name="Global", header=None)
IDS = pd.read_excel(path, sheet_name="IDs", header=None)
COL_IDS = IDS.stack().dropna().tolist()

## ----- Create heatmap
plt.figure(figsize=(10, 10))  # Adjust the figure size as needed
sns.set_theme(style="whitegrid")
heatmap = sns.heatmap(MATRIX, xticklabels=COL_IDS, yticklabels=COL_IDS, cmap=sns.diverging_palette(250, 20, l=65, center="dark", as_cmap=True), annot=False, square=True)
plt.title('Connectivity Matrix Heatmap')
plt.xlabel('Regions')
plt.ylabel('Regions')
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.show()
