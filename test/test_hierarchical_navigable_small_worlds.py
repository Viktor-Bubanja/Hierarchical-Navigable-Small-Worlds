import pytest
from src.hierarchical_navigable_small_worlds import HNSW, Vertex, knn


# Mock the knn function to return the first k elements
def mock_knn(k, x, neighbours):
    return neighbours[:k]

HNSW.knn = staticmethod(mock_knn)


@pytest.fixture
def hnsw():
    return HNSW(M=2)


@pytest.fixture
def setup_hnsw():
    hnsw = HNSW(M=2)
    vertices = [Vertex(vector=[i], num_layers=hnsw.num_layers) for i in range(5)]
    for vertex in vertices:
        hnsw.add(vertex)
    return hnsw, vertices


def test_knn_returns_k_closest_vectors_to_given_vector():
    neighbours = [
        Vertex(vector=[1.5, 0.5, 1.5]),
        Vertex(vector=[10, 10, 10]),
        Vertex(vector=[20, 20, 20]),
        Vertex(vector=[3, 2, 1]),
        Vertex(vector=[-5, 0, 1]),
        Vertex(vector=[40, 40, 40])
    ]
    nearest_neighbours = knn(k=3, x=Vertex(vector=[1, 1, 1]), neighbours=neighbours)
    expected_nearest_neighbours = [
        Vertex([1.5, 0.5, 1.5]),
        Vertex([3, 2, 1]),
        Vertex([-5, 0, 1])
    ]
    assert nearest_neighbours == expected_nearest_neighbours

def test_set_layer_probs_sets_expected_probabilities_and_cumulative_neighbour_counts():
    hnsw = HNSW(M=32, M_0=None, m_L=None)
    assert hnsw.layer_probs == [
        0.96875,
        0.030273437499999986,
        0.0009460449218749991,
        2.956390380859371e-05,
        9.23871994018553e-07,
        2.887099981307982e-08,
    ]
    assert hnsw.cumulative_nn_per_level == [64, 96, 128, 160, 192, 224]


def test_add_sets_entry_point_if_index_is_empty():
    hnsw = HNSW(M=32, M_0=None, m_L=None)
    input_vertex = Vertex(
        vector=[1,1,1],
        num_layers=hnsw.num_layers
    )
    insertion_layer = hnsw.add(x=input_vertex)
    assert hnsw.entry_point == input_vertex, "The first vertex should be the entry point."
    assert insertion_layer == 5, "The first vertex should be inserted in the last layer."


def test_graph_integrity_after_addition(mocker, setup_hnsw):
    mock_values = [0, 0, 1, 0, 0, 0, 1, 1]
    hnsw, _ = setup_hnsw()
    mocker.patch.object(hnsw, '_get_random_level', side_effect=mock_values)
    new_vertex = Vertex(vector=[10], num_layers=hnsw.num_layers)
    insertion_layer = hnsw.add(new_vertex)
    for i in range(insertion_layer, -1, -1):
        edges = new_vertex.get_edges_in_layer(i)
        for edge in edges:
            assert new_vertex in edge.get_edges_in_layer(i), "Edges should be bidirectional."


def test_add_sets_edges_of_input_vertex_as_well_as_edges_of_neighbours_in_all_necessary_layers(mocker, hnsw):
    mock_values = [0, 0, 1, 0, 0, 0, 1, 1]  # The layers that the input vertices will be inserted in
    mocker.patch.object(hnsw, '_get_random_level', side_effect=mock_values)
    input_vertices = [
        Vertex(
            vector=[1,1,1],
            num_layers=hnsw.num_layers
        ),
        Vertex(
            vector=[2,2,2],
            num_layers=hnsw.num_layers
        ),
        Vertex(
            vector=[3,3,3],
            num_layers=hnsw.num_layers
        ),
        Vertex(
            vector=[4,4,4],
            num_layers=hnsw.num_layers
        ),
        Vertex(
            vector=[5,5,5],
            num_layers=hnsw.num_layers
        ),
        Vertex(
            vector=[6,6,6],
            num_layers=hnsw.num_layers
        ),
        Vertex(
            vector=[7,7,7],
            num_layers=hnsw.num_layers
        ),
        Vertex(
            vector=[8,8,8],
            num_layers=hnsw.num_layers
        ),
        Vertex(
            vector=[9,9,9],
            num_layers=hnsw.num_layers
        ),
    ]
    for input_vertex in input_vertices:
        hnsw.add(x=input_vertex)

    assert hnsw.entry_point == input_vertices[0]
    assert input_vertices[0].edges == [input_vertices[1], input_vertices[2], input_vertices[3], input_vertices[4], input_vertices[5], input_vertices[6], input_vertices[7]]
