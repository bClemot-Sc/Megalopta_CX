#### Script to get drosophila data around PFL neurons and represent them
## Author: Bastien Clémot


## ----- VARIABLES
threshold = 50 # Integer
direction = "None" # "None", "Toward_PFL", "From_PFL"
plot = ["Network", "Dendrogram", "Ranking"] # "Network" and/ or "Dendrogram" and/or "Ranking"

## ----- Import packages
import csv
from fafbseg import flywire
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
from netgraph import Graph
import os
import pandas as pd
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.metrics.pairwise import cosine_similarity


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
path = "Drosophila_connectivity_data/all_annotation_datataset.txt"
if not os.path.exists(path):
    all_annotation = pd.DataFrame(flywire.search_annotations(None))
    all_annotation.to_csv(path, sep="\t")
all_annotation = pd.read_csv(path, sep="\t", low_memory=False)
all_rootID = all_annotation.loc[:,"root_id"]

# Search for PFL3 neurons annotation
path = "Drosophila_connectivity_data/PFL_annotation_dataset.txt"
if not os.path.exists(path):
    PFL_annotation = pd.DataFrame(flywire.search_annotations("PFL3"))
    PFL_annotation.to_csv(path, sep="\t")
PFL_annotation = pd.read_csv(path, sep="\t")
PFL_rootID = PFL_annotation.loc[:,"root_id"]

# Get neurons that connect towards PFLs
path = "Drosophila_connectivity_data/Towards_matrix.txt"
if not os.path.exists(path):
    toward_matrix = pd.DataFrame(flywire.synapses.get_adjacency(sources=all_rootID, targets=PFL_rootID, min_score=0))
    toward_matrix.to_csv(path, sep="\t")
toward_matrix = pd.read_csv(path, sep="\t")

# # Get neurons that receive from PFLs
# path = "/mnt/c/Users/bclem/OneDrive/Documents/GitHub/Megalopta_CX/Drosophila_connectivity_data/From_matrix.txt"
# if not os.path.exists(path):
#     from_matrix = pd.DataFrame(flywire.synapses.get_adjacency(sources=PFL_rootID, targets=all_rootID, min_score=0))
#     from_matrix.to_csv(path, sep="\t")
# else :
#     from_matrix = pd.read_csv(path, sep="\t")
    
# Filter with neuron connectivity threshold
toward_matrix[toward_matrix<=threshold] = 0
# from_matrix[from_matrix<=threshold] = 0

# Remove unconnected neurons
toward_matrix = toward_matrix[(toward_matrix.iloc[:,1:] != 0).any(axis=1)]
# from_matrix = from_matrix.loc[:, (from_matrix.iloc[1:] != 0).any()]

# Print summary
print("----- Summary -----")
print("Number of PFL neurons: ", toward_matrix.shape[1])
print("Synaptic threshold: ", threshold)
print("Number of neurons connected toward PFLs: ", toward_matrix.shape[0])
# print("Number of neurons receiving from PFLs: ", from_matrix.shape[1])
print("-------------------")

# Get complete adjacency matrix
toward_neurons_rootIDs = toward_matrix.iloc[1:,0].values.tolist()
PFL_neurons_rootIDs = toward_matrix.columns[1:]
PFL_neurons_rootIDs = [int(_) for _ in PFL_neurons_rootIDs]
# from_neurons_rootIDs = from_matrix.columns[1:]
# from_neurons_rootIDs = [int(_) for _ in from_neurons_rootIDs]
circuit_rootIDs = toward_neurons_rootIDs + PFL_neurons_rootIDs # + from_neurons_rootIDs
path = "Drosophila_connectivity_data/Global_matrix_" + str(threshold) + ".txt"
if not os.path.exists(path):
    global_matrix = pd.DataFrame(flywire.synapses.get_adjacency(sources=circuit_rootIDs, targets=circuit_rootIDs, min_score=0))
    global_matrix.to_csv(path, sep="\t")
global_matrix = pd.read_csv(path, sep="\t")

# Get neuron names from neuron IDs
neuron_rootIDs = global_matrix.iloc[:,0].values
neuron_rootIDs = [int(_) for _ in neuron_rootIDs]
df_neuron_rootIDs = pd.DataFrame({"root_id": neuron_rootIDs})
_ = pd.merge(df_neuron_rootIDs, all_annotation, on="root_id", how="left")["hemibrain_type"]
path = "Drosophila_connectivity_data/Neurone_names_" + str(threshold) + ".txt"
neurone_names = _.tolist()
with open(path, mode='w', newline='') as file:
    writer = csv.writer(file)
    for item in neurone_names:
        writer.writerow([item])
with open(path, 'r') as file:
    neurone_names = [line.strip() for line in file]


## ----- Graphical representations
global_matrix = global_matrix.iloc[:,1:]
# Clean neurones
kept_index = []
kept_PFL = []
kept_toward = []
for i, name in enumerate(neurone_names):
    if "ExR" in name:
        continue
    elif "nan" in name:
        continue
    elif "P6-8P9" in name:
        continue
    elif "PEG" in name:
        continue
    elif "VES" in name:
        continue
    elif "IbSpsP" in name:
        continue
    elif "LPsP" in name:
        continue
    elif "P1-9" in name:
        continue
    elif "PFL2" in name:
        continue
    elif "PLP" in name:
        continue
    elif "AOTU" in name:
        continue
    elif "LC" in name:
        continue
    kept_index.append(i)
    if "PFL" in name:
        kept_PFL.append(i)
    else:
        kept_toward.append(i)
    
if direction == "Toward_PFL":
    global_matrix.iloc[kept_PFL,:] = 0
elif direction == "From_PFL":
    global_matrix.iloc[kept_toward,:] = 0
neurone_names = [neurone_names[i] for i in kept_index]
global_matrix = global_matrix.iloc[kept_index,kept_index]

# Avoid repetition in neuron names
counts = {}
result = []
for neurone in neurone_names:
    if neurone not in counts:
        counts[neurone] = 1
        result.append(neurone)
    else:
        result.append(f"{neurone}-{counts[neurone]}")
        counts[neurone] += 1
neurone_names = result

## Calculate cosine distance
if "Dendrogram" in plot:
    # Convert matrix to numpy array
    adjacency_matrix = global_matrix.values
    # Replace 0s with a small value
    adjacency_matrix[adjacency_matrix == 0] = 1e-10
    # Normalise adjacency matrix to range [-1,1]
    normalised_adjacency_matrix = adjacency_matrix / np.sqrt(np.outer(np.diag(adjacency_matrix), np.diag(adjacency_matrix)))
    # Compute cosine similarity
    cosine_sim = cosine_similarity(normalised_adjacency_matrix)
    # Compute hierarchical clustering
    linkage_matrix = linkage(cosine_sim, method='complete', metric='cosine')
    # Plot dendrogram
    plt.figure(figsize=(8,10))
    dendrogram(linkage_matrix, labels=neurone_names, orientation='left')
    plt.title('Hierarchical Clustering Dendrogram')
    plt.xlabel('Distance')
    plt.ylabel('Node')

# Adapt matrix input
from_list = []
to_list = []
value_list = []
for i in range(global_matrix.shape[0]):
    for j in range(global_matrix.shape[0]):
        if global_matrix.iloc[i,j] > 0:
            from_list.append(neurone_names[i])
            to_list.append(neurone_names[j])
            value_list.append(global_matrix.iloc[i,j])
graph_df = pd.DataFrame({ 'from':from_list, 'to':to_list, 'value':value_list})
G=nx.from_pandas_edgelist(graph_df, 'from', 'to')

# Group neurons to their type
neuron_to_type = dict()
for neuron in neurone_names:
    if "hDelta" in neuron:
        neuron_to_type[neuron] = 0
    elif "Delta7" in neuron:
        neuron_to_type[neuron] = 1
    elif "EPG" in neuron:
        neuron_to_type[neuron] = 2
    elif "FC" in neuron:
        neuron_to_type[neuron] = 3
    elif "FB" in neuron:
        neuron_to_type[neuron] = 4
    elif "PFL3" in neuron:
        neuron_to_type[neuron] = 5
    elif "PFN" in neuron:
        neuron_to_type[neuron] = 6
    elif "vDelta" in neuron:
        neuron_to_type[neuron] = 7
    else:
        neuron_to_type[neuron] = 8

# Define color
type_to_color = {
    0 : '#A2E1DB',
    1 : '#DD968A',
    2 : '#FFFFB5',
    3 : '#F7ABC7',
    4 : '#B494C5',
    5 : '#B6CFB6',
    6 : '#559EF0',
    7 : '#AEC6CF',
    8 : '#BBBABF',
}
neuron_color = {neuron: type_to_color[type_id] for neuron, type_id in neuron_to_type.items()}

# Plot Graph
if "Network" in plot:
    print('Graph in progress...')
    Graph(G,
        node_color=neuron_color, arrows = False,
        node_width=2, edge_width=0.3, edge_alpha=0.1,
        node_labels=True, node_label_fontdict=dict(size=6),
        node_layout='community', node_layout_kwargs=dict(node_to_community=neuron_to_type),
        edge_layout='bundled', edge_layout_kwargs=dict(k=2000),
    )

## ----- classification of all neurons connecting toward PFLs
if "Ranking" in plot:
    neuron_rootIDs = toward_matrix.iloc[:,0].values
    neuron_rootIDs = [int(_) for _ in neuron_rootIDs]
    df_neuron_rootIDs = pd.DataFrame({"root_id": neuron_rootIDs})
    _ = pd.merge(df_neuron_rootIDs, all_annotation, on="root_id", how="left")["hemibrain_type"]
    neurone_names = _.tolist()
    toward_matrix = toward_matrix.iloc[:,1:]
    toward_matrix["row_sum"] = toward_matrix.sum(axis=1)
    toward_matrix["name"] = neurone_names
    df_sorted = toward_matrix.sort_values(by="row_sum", ascending=False)
    df_sorted = df_sorted.loc[:,("row_sum","name")]
    df_sorted.to_csv("Drosophila_connectivity_data/Sorted_neurons_to_PFL.csv", index=False)

plt.show()