import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# create matrix
size = 10000
similarity_matrix = np.random.rand(size, size)

# plot matrix

# create figure and set size
plt.figure(figsize=(14, 14))

# add heatmap
sns.heatmap(similarity_matrix, vmin=0, vmax=1)

# save the figure
plt.savefig('test.png', dpi=600)

# show the figure; this was slow
plt.show()