#### Script for randomly generating a connectivity matrix with labels
#### As well as a random activity vector
## Autor: Bastien Clémot
## python Random_matrix.py


## ----- Import libraries
import numpy as np


## ----- Function to generate random matrix and activity vector
def create_connectivity_matrix():
    # Get the number of neuron groups from the user
    num_groups = int(input("How many groups of neurons do you want? "))

    # Initialize empty lists to store neuron names and neuron counts per group
    neuron_names = []
    neuron_counts = []

    # Get information about each group from the user
    for i in range(num_groups):
        group_name = input(f"Enter the name of group {i + 1}: ")
        neuron_count = int(input(f"How many neurons in group {group_name}? "))
        
        # Add group name and neuron count to lists
        neuron_names.append(group_name)
        neuron_counts.append(neuron_count)

    # Get density of the connectivity matrix from the user
    density = float(input("Enter the density of the connectivity matrix (between 0 and 1): "))

    # Create connectivity matrix
    matrix = np.random.choice([0, 1], size=(sum(neuron_counts), sum(neuron_counts)), p=[1 - density, density])

    # Export the matrix with row and column headers
    export_matrix(matrix, neuron_names, neuron_counts)

    # Create and export the activity vector
    activity_vector = np.random.uniform(0, 1, size=sum(neuron_counts))
    export_activity_vector(activity_vector, neuron_names, neuron_counts)


# ----- Function to export matrix
def export_matrix(matrix, neuron_names, neuron_counts):
    # Create headers for rows and columns
    row_headers = [f"{group_name}_{i + 1}" for group_idx, group_name in enumerate(neuron_names) for i in range(neuron_counts[group_idx])]
    col_headers = row_headers.copy()

    # Save the matrix to a CSV file
    filename = "connectivity_matrix.csv"
    np.savetxt(filename, matrix, delimiter=",", header=",".join(col_headers), comments="", fmt="%d")

    print(f"Connectivity matrix has been exported to {filename} with row and column headers.")


# ----- Function to export activity vector
def export_activity_vector(activity_vector, neuron_names, neuron_counts):
    # Create headers for the activity vector
    headers = [f"{group_name}_{i + 1}" for group_idx, group_name in enumerate(neuron_names) for i in range(neuron_counts[group_idx])]

    # Save the activity vector to a CSV file
    filename = "activity_vector.csv"
    np.savetxt(filename, activity_vector, delimiter=",", header=",".join(headers), comments="", fmt="%f")

    print(f"Activity vector has been exported to {filename} with headers.")


## ----- Run functions
create_connectivity_matrix()

