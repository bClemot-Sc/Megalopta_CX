def sum_vectors(vector1, vector2):
    if len(vector1) != len(vector2):
        raise ValueError("Vectors must be of the same size")

    return [vector1[i] + vector2[i] for i in range(len(vector1))]

# Example vectors
vector1 = [1, 2, 3]
vector2 = [4, 5, 6]

result = sum_vectors(vector1, vector2)
print(result)  # Output: [5, 7, 9]
