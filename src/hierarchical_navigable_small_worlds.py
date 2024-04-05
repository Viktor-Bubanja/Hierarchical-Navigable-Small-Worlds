from __future__ import annotations
import random
import numpy as np

from src.vector_search import knn

class Vertex:
    def __init__(self, vector, num_layers):
        self.vector = vector
        self.edges = np.array()

    
    def add_edge(self, vertex: Vertex, layer_index: int):
        self.edges[layer_index] = np.append(self.edges[layer_index], vertex)
        vertex.edges[layer_index] = np.append(vertex.edges, self)


class Layer:
    def __init__(self, vectors: np.ndarray, index: int=0):
        self.vectors = vectors
        self.index = index

    def add(self, x: Vertex):
        self.vectors = np.append(self.vectors, x)


class HVSW:
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
        self.m_L = 1 / np.log(M) if m_L is None else m_L
        self.layers = []
        self.layer_probs, self.cumulative_nn_per_level = self._set_layer_probs()
        self.num_layers = len(self.layer_probs)

    def _set_layer_probs(self):
        nn = 0
        cumulative_nn_per_level = []
        level = 0
        probs = []
        layer_prob_low_threshold = 1e-9
        while True:
            prob = np.exp(-level / self.m_L) * (1 - np.exp(-1 / self.m_L))
            if prob < layer_prob_low_threshold:
                break
            probs.append(prob)
            nn += self.M_0 if level == 0 else self.M
            cumulative_nn_per_level.append(nn)
            level += 1
        return probs, cumulative_nn_per_level

    def query(self, x: np.array):
        level = self._get_random_level() 
        ef = 1
        entry_point = x
        for i in range(self.num_layers-1, level, -1):
            layer = self.layers[i]
            nearest_neighbour = layer.knn(ef, entry_point)
            entry_point = nearest_neighbour
        
        candidates = self.traverse_layers(entry_point=entry_point, start_index=level)


    def add(self, x: Vertex):
        level = self._get_random_level() 
        ef = 1
        entry_point = self.layers[-1][-1] # TODO: update this
        for i in range(self.num_layers-1, level, -1):
            nearest_neighbour = knn(k=ef, x=x.vector, neighbours=entry_point.edges[i])
            entry_point = nearest_neighbour
        
        candidates = []
        ef_construction = 10
        # Layer l
        candidates = entry_point.edges[level]
        nearest_neighbours = knn(k=ef_construction, x=x.vector, neighbours=candidates)
        for neighbour in nearest_neighbours:
            x.add_edge(neighbour)
        
        entry_points = nearest_neighbours
        
        # Layer l-1
        for entry_point in entry_points:
            nearest_neighbours = knn(k=ef_construction, x=x.vector, neighbours=entry_point.edges[l-1])



    def traverse_layers(self, entry_point, start_index, candidates=None):
        if candidates is None:
            candidates = []
        
        for i in range(start_index, -1, -1):
            layer = self.layers[start_index]
            k = self._get_num_neighbours(i)
            nearest_neighbours = knn(k=k, x=entry_point, neighbours=layer.vectors)
            if layer == 0:
                return nearest_neighbours
            for neighbour in nearest_neighbours:
                candidates = candidates.extend(
                    self.traverse_layers(entry_point=neighbour, start_index=i-1, candidates=candidates)
                )
        return candidates

    def _get_random_level(self):
        rand = random.uniform(0, 1)
        for level in range(len(self.layer_probs)):
            if rand < self.layer_probs[level]:
                return level
            f -= self.layer_probs[level]
        return len(self.layer_probs) - 1

    def _get_num_neighbours(self, layer_index):
        return self.M_0 if layer_index == 0 else self.M




