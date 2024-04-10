#### Script to get drosophila data around PFL neurons and represent them
## Author: Bastien Clémot


## ----- Import packages
from fafbseg import flywire
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import os
import pandas as pd


## ----- Set global Flywire parameters
print("----------")
# Set default dataset
flywire.set_default_dataset("public")

# Check materialization
materialization_df = pd.DataFrame(flywire.get_materialization_versions())
pd.set_option("display.max_rows", 100)
pd.set_option("display.max_columns", None)
print(materialization_df)


## ----- Get PFL neurons connectivity matrix
# Get all annotations
path = "/mnt/c/Users/bclem/OneDrive/Documents/GitHub/Megalopta_CX/Drosophila_connectivity_data/all_annotation_datataset.txt"
if not os.path.exists(path):
    all_annotation = pd.DataFrame(flywire.search_annotations(None))
    all_annotation.to_csv(path, sep="\t")
else :
    all_annotation = pd.read_csv(path, sep="\t", low_memory=False)
all_rootID = all_annotation.loc[:,"root_id"]

# Search for PFL neurons annotation
path = "/mnt/c/Users/bclem/OneDrive/Documents/GitHub/Megalopta_CX/Drosophila_connectivity_data/PFL_annotation_dataset.txt"
if not os.path.exists(path):
    PFL_annotation = pd.DataFrame(flywire.search_annotations("PFL"))
    PFL_annotation.to_csv(path, sep="\t")
else :
    PFL_annotation = pd.read_csv(path, sep="\t")
PFL_rootID = PFL_annotation.loc[:,"root_id"]

# Get neurons that connect towards PFLs
path = "/mnt/c/Users/bclem/OneDrive/Documents/GitHub/Megalopta_CX/Drosophila_connectivity_data/Towards_matrix.txt"
if not os.path.exists(path):
    toward_matrix = pd.Dataframe(flywire.synapses.get_adjacency(sources=all_rootID, targets=PFL_rootID, min_score=0))
    toward_matrix.to_csv(path, sep="\t")
else :
    toward_matrix = pd.read_csv(path, sep="\t")

# Get neurons that receive from PFLs
path = "/mnt/c/Users/bclem/OneDrive/Documents/GitHub/Megalopta_CX/Drosophila_connectivity_data/From_matrix.txt"
if not os.path.exists(path):
    from_matrix = pd.DataFrame(flywire.synapses.get_adjacency(sources=PFL_rootID, targets=all_rootID, min_score=0))
    from_matrix.to_csv(path, sep="\t")
else :
    from_matrix = pd.read_csv(path, sep="\t")
    
# Filter with neuron connectivity threshold
threshold = 5
toward_matrix[toward_matrix<=threshold] = 0
from_matrix[from_matrix<=threshold] = 0

# Remove unconnected neurons
toward_matrix = toward_matrix[(toward_matrix.iloc[:,1:] != 0).any(axis=1)]
from_matrix = from_matrix.loc[:, (from_matrix.iloc[1:] != 0).any()]

# Print summary
print("----- Summary -----")
print("Number of PFL neurons: ", from_matrix.shape[0])
print("Synaptic threshold: ", threshold)
print("Number of neurons connected toward PFLs: ", toward_matrix.shape[0])
print("Number of neurons receiving from PFLs: ", from_matrix.shape[1])
print("-------------------")

# Get complete adjacency matrix
toward_neurons_rootIDs = toward_matrix.iloc[1:,0].values.tolist()
PFL_neurons_rootIDs = from_matrix.iloc[1:,0].values.tolist()
from_neurons_rootIDs = from_matrix.columns[1:]
from_neurons_rootIDs = [int(_) for _ in from_neurons_rootIDs]
circuit_rootIDs = toward_neurons_rootIDs + PFL_neurons_rootIDs + from_neurons_rootIDs
path = "/mnt/c/Users/bclem/OneDrive/Documents/GitHub/Megalopta_CX/Drosophila_connectivity_data/Global_matrix_" + str(threshold) + ".txt"
if not os.path.exists(path):
    global_matrix = pd.DataFrame(flywire.synapses.get_adjacency(sources=circuit_rootIDs, targets=circuit_rootIDs, min_score=0))
    global_matrix.to_csv(path, sep="\t")
else : 
    global_matrix = pd.read_csv(path, sep="\t")

# Get neuron names from neuron IDs
neuron_rootIDs = global_matrix.iloc[:,0].values
neuron_rootIDs = [int(_) for _ in neuron_rootIDs]
df_neuron_rootIDs = pd.DataFrame({"root_id": neuron_rootIDs})
_ = pd.merge(df_neuron_rootIDs, all_annotation, on="root_id", how="left")["hemibrain_type"]
neurone_names = _.tolist()

# Filter a second time with the threshold


## ----- Graphical representation
# Create graph
global_matrix = global_matrix.drop(columns=global_matrix.columns[0])
global_matrix.columns = neurone_names
global_matrix.index = neurone_names
print(global_matrix.shape)

G = nx.from_pandas_adjacency(global_matrix)

# Set nodes position
pos = nx.spring_layout(G)

# Draw nodes
nx.draw_networkx_nodes(G, pos, node_size=500, node_color='skyblue')

# Draw edges
nx.draw_networkx_edges(G, pos, width=1.0, alpha=0.5)


# Draw node labels
nx.draw_networkx_labels(G, pos, font_size=10, font_family='sans-serif')

# Display graph
plt.title('Graph from Adjacency DataFrame')
plt.axis('off')  # Turn off axis
plt.show()