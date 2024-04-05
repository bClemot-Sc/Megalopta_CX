import csv
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import pandas as pd

# Load connectivity matrix
CON_MAT = np.genfromtxt("Theorical_connectivity_matrix.csv", delimiter=',')

# Load neuron IDs from CSV
with open("Neurons_IDs.csv", "r") as file:
        COL_IDS = next(csv.reader(file, delimiter=','))

# Create graph
G = nx.Graph()

# Add nodes
labels = {}
for node_id in range(len(COL_IDS)):
    G.add_node(node_id)
    labels[node_id] = COL_IDS[node_id]

# Add edges
for i in range(len(CON_MAT)):
    for j in range(len(CON_MAT[i])):
        value = CON_MAT[i, j]
        if value > 0:
            G.add_edge(i, j)

# Print network
nx.draw(G, labels=labels, font_size=8, font_weight='bold', node_size=200, node_color='skyblue', edge_color='gray')

plt.show()
