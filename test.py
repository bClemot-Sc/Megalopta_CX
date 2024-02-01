import numpy as np

def linear_activation(activity_vector):
    np.clip(activity_vector, 0, 1, out=activity_vector)

# Example usage
input_vector = np.array([0.5, 1.2, 0.8, -0.5, 1.5])
linear_activation(input_vector)

print(input_vector)
