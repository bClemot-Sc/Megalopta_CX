#### Script for drosophila adjacency matrix
## Autor: Bastien Clémot


##### ----- Import packages
from fafbseg import flywire
import pandas as pd


##### ----- Set global Flywire parameters
print("----------")
# Set default dataset
flywire.set_default_dataset("public")
# Check materialisation
materialisation_df = pd.DataFrame(flywire.get_materialization_versions())
pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', None)
print(materialisation_df)


##### ----- Get PFL neurons connectivity matrix
print("----------")
# Search for PFL neurons annotation
annotation_df = flywire.search_annotations("PFL")
annotation_df.to_csv("/mnt/c/Users/bclem/OneDrive/Documents/GitHub/Megalopta_CX/PFL_annotation_data.txt", sep='\t', index=False)
# Get adjacency matrix
adjacency_df = flywire.synapses.get_adjacency("720575940638678616")
print(adjacency_df)
adjacency_df.to_csv("/mnt/c/Users/bclem/OneDrive/Documents/GitHub/Megalopta_CX/PFL_adjacency_data.txt", sep='\t', index=False)