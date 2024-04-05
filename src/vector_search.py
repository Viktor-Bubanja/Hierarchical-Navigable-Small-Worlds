import numpy as np

def knn(k: int, x: np.array, neighbours: np.ndarray):
    distances = np.linalg.norm(neighbours - x, axis=1)
    nn_indexes = distances.argsort()[:k]
    nearest_neighbours = neighbours[nn_indexes]
    return nearest_neighbours
