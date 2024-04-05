import numpy as np
from src.hierarchical_navigable_small_worlds import Layer

def test_knn_returns_k_closest_vectors_to_given_vector():
    layer = Layer(
        vectors=np.array([
            [1.5, 0.5, 1.5],
            [10, 10, 10],
            [20, 20, 20],
            [3, 2, 1],
            [-5, 0, 1],
            [40, 40, 40]
        ])
    )
    nearest_neighbours = layer.knn(3, np.array([1, 1, 1]))
    expected_nearest_neighbours = np.array([
        [1.5, 0.5, 1.5],
        [3, 2, 1],
        [-5, 0, 1]
    ])
    assert np.array_equal(nearest_neighbours, expected_nearest_neighbours)
