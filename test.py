import numpy as np

def threshold_vector(vector, threshold):
    result = np.array(vector, dtype=float) > threshold
    return result.astype(int)

# Example usage:
vector = [1.2, 0.5, 2.3, 0.8, 1.7]
threshold_value = 1.0

result = threshold_vector(vector, threshold_value)
print(result)