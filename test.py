# Define a sample function for mutation
def mutation(individual, connectivity_anthropy, neuron_anthropy):
    return individual * 2  # Just doubling the individual for demonstration

# Sample values for ind_list, connectivity_anthropy, and neuron_anthropy
ind_list = [1, 2, 3, 4, 5]
connectivity_anthropy = 0.5
neuron_anthropy = 0.3

# Execute the line of code
new_list = [ind_list[0], ind_list[1]] + [mutation(ind, connectivity_anthropy, neuron_anthropy) for ind in ind_list[2:]]

# Print the result
print(new_list)
