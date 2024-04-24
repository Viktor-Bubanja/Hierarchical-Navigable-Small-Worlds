from __future__ import annotations
import math
import random


class Vertex:
    def __init__(self, vector: list[float], num_layers: int=0):
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


def euclidean_distance(a: Vertex, b: Vertex) -> int:
    distance_squared = 0
    for x, y in zip(a.vector, b.vector):
        distance_squared += (x - y) ** 2
    return math.sqrt(distance_squared)


def knn(k: int, x: Vertex, neighbours: list[Vertex]) -> list[Vertex]:
    # TODO: implement clustering heuristic
    nearest_neighbours = sorted(neighbours, key=lambda n: euclidean_distance(x, n))[:k]
    return nearest_neighbours 


class HNSW:
    """
    d is the dimensionality of the vectors.
    M is the number of edges of each vertex in the graph.
    M_0 is the number of edges of each vertex in the lowest layer of the graph.
    m_L is the non-zero 'level multiplier' that normalizes the exponentially decaying probability distribution
    which determines which layer a given node belongs to.
    """
    def __init__(self, M=32, M_0=None, m_L=None):
        self.M = M
        self.M_0 = 2*M if M_0 is None else M_0
        self.m_L = 1 / math.log(M) if m_L is None else m_L
        self.layer_probs = self._set_layer_probabilities()
        self.num_layers = len(self.layer_probs)
        self.entry_point = None

    def _set_layer_probabilities(self) -> list[float]:
        nn = 0
        level = 0
        probs = []
        layer_prob_low_threshold = 1e-9
        while True:
            prob = math.exp(-level / self.m_L) * (1 - math.exp(-1 / self.m_L))
            if prob < layer_prob_low_threshold:
                break
            probs.append(prob)
            nn += self.M_0 if level == 0 else self.M
            level += 1
        return probs

    def add(self, x: Vertex) -> int:
        if self.entry_point is None:
            self.entry_point = x
            return self.num_layers - 1

        insertion_layer = self._get_random_level() 
        ef = 1
        entry_point: Vertex = self.entry_point
        for i in range(self.num_layers-1, insertion_layer, -1):
            nearest_neighbour = knn(
                k=ef, x=x, neighbours=entry_point.get_neighbours_in_layer(i)
            )[0]
            entry_point = nearest_neighbour
        
        ef_construction = 10
        candidates_for_level = [entry_point]
        for i in range(insertion_layer, -1, -1):
            entry_points = candidates_for_level
            candidates_for_level = []
            for entry_point in entry_points:
                nearest_neighbours = knn(
                    k=ef_construction, x=x, neighbours=entry_point.get_neighbours_in_layer(i)
                )
                candidates_for_level.extend(nearest_neighbours)
            k = self._get_num_neighbours(i)
            edges_to_add = knn(k=k, x=x, neighbours=candidates_for_level)
            for edge in edges_to_add:
                x.add_edge(vertex=edge, layer=i)
        
        return insertion_layer

    def search(self, x: Vertex) -> Vertex:
        ef = 1
        entry_point: Vertex = self.entry_point
        for i in range(self.num_layers-1, -1, -1):
            nearest_neighbour = knn(k=ef, x=x, neighbours=entry_point.get_neighbours_in_layer(i))
            if euclidean_distance(x, entry_point) < euclidean_distance(x, nearest_neighbour):
                return entry_point
            entry_point = nearest_neighbour
        return entry_point

    def _get_random_level(self):
        rand = random.uniform(0, 1)
        for level in range(len(self.layer_probs)):
            if rand < self.layer_probs[level]:
                return level
            rand -= self.layer_probs[level]
        return len(self.layer_probs) - 1

    def _get_num_neighbours(self, layer_index):
        return self.M_0 if layer_index == 0 else self.M
