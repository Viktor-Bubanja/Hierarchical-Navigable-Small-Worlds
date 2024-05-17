from src.vertex import Vertex, knn


def test_knn_returns_k_closest_vectors_to_given_vector():
    neighbours = [
        Vertex(vector=[1.5, 0.5, 1.5]),
        Vertex(vector=[10, 10, 10]),
        Vertex(vector=[20, 20, 20]),
        Vertex(vector=[3, 2, 1]),
        Vertex(vector=[-5, 0, 1]),
        Vertex(vector=[40, 40, 40]),
    ]
    nearest_neighbours = knn(k=3, x=Vertex(vector=[1, 1, 1]), neighbours=neighbours)
    expected_nearest_neighbours = [
        Vertex([1.5, 0.5, 1.5]),
        Vertex([3, 2, 1]),
        Vertex([-5, 0, 1]),
    ]
    assert nearest_neighbours == expected_nearest_neighbours
