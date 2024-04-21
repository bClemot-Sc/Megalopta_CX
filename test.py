import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt

# Node names
node_names = ['A', 'B', 'C', 'D', 'E']

# Sample adjacency matrix stored as a pandas DataFrame
data = {
    'A': [1.0, 2.0, 1.5, 1.0, 1.0],
    'B': [2.0, 1.0, 1.0, 1.8, 1.0],
    'C': [1.5, 1.0, 1.0, 1.2, 1.0],
    'D': [1.0, 1.8, 1.2, 1.0, 1.5],
    'E': [1.0, 1.0, 1.0, 1.5, 1.0]
}
adjacency_df = pd.DataFrame(data, index=node_names)

# Convert DataFrame to numpy array
adjacency_matrix = adjacency_df.values

# Normalize adjacency matrix to range [-1, 1]
normalized_adjacency_matrix = adjacency_matrix / np.sqrt(np.outer(np.diag(adjacency_matrix), np.diag(adjacency_matrix)))

# Compute cosine similarity
cosine_sim = cosine_similarity(normalized_adjacency_matrix)

# Compute hierarchical clustering
linkage_matrix = linkage(cosine_sim, method='complete', metric='cosine')

# Plot dendrogram with node names vertically
plt.figure(figsize=(8, 10))
dendrogram(linkage_matrix, labels=node_names, orientation='right')
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Distance')
plt.ylabel('Node')
plt.show()
