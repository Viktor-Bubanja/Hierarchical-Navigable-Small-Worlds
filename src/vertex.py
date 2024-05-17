from __future__ import annotations

import math


def euclidean_distance(a: Vertex, b: Vertex) -> int:
    distance_squared = 0
    for x, y in zip(a.vector, b.vector):
        distance_squared += (x - y) ** 2
    return math.sqrt(distance_squared)


def knn(k: int, x: Vertex, neighbours: list[Vertex]) -> list[Vertex]:
    nearest_neighbours = sorted(neighbours, key=lambda n: euclidean_distance(x, n))[:k]
    return nearest_neighbours


class Vertex:
    def __init__(self, vector: list[float], num_layers: int = 0):
        self.vector = vector
        self.edges = [set() for _ in range(num_layers)]

    def add_edge(self, vertex: Vertex, layer: int) -> None:
        self.edges[layer].add(vertex)
        vertex.edges[layer].add(self)

    def get_edges_in_layer(self, layer: int) -> set[Vertex]:
        return self.edges[layer]

    def get_neighbours_in_layer(self, layer: int) -> list[Vertex]:
        return self.get_edges_in_layer(layer) | {self}

    def __eq__(self, vertex: Vertex) -> bool:
        return self.vector == vertex.vector

    def __repr__(self) -> str:
        return f"Vertex(vector={self.vector})"

    def __hash__(self) -> int:
        return hash(tuple(self.vector))
