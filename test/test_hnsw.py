import pytest

from src.hnsw import HNSW, Vertex, knn


# Mock the knn function to return the first k elements
def mock_knn(k, x, neighbours):
    return neighbours[:k]


HNSW.knn = staticmethod(mock_knn)


@pytest.fixture
def hnsw():
    return HNSW(M=2)


@pytest.fixture
def populated_hnsw():
    hnsw = HNSW(M=2)
    vertices = [Vertex(vector=[i], num_layers=hnsw.num_layers) for i in range(5)]
    for vertex in vertices:
        hnsw.add(vertex)
    return hnsw


@pytest.fixture
def popoulated_hnsw_2(mocker):
    mock_random_levels = [
        0,
        0,
        1,
        0,
        0,
        0,
        1,
        1,
    ]  # The layers that the input vertices will be inserted in
    mocker.patch.object(HNSW, "_get_random_level", side_effect=mock_random_levels)
    mock_probabilities = [
        0.5,
        0.5,
    ]  # The probabilities of inserting an edge into each layer
    mocker.patch.object(
        HNSW, "_set_layer_probabilities", return_value=mock_probabilities
    )
    hnsw = HNSW(M=2)
    input_vertices = [
        Vertex(vector=[0, 0, 0], num_layers=hnsw.num_layers),
        Vertex(vector=[1, 1, 1], num_layers=hnsw.num_layers),
        Vertex(vector=[2, 2, 2], num_layers=hnsw.num_layers),
        Vertex(vector=[3, 3, 3], num_layers=hnsw.num_layers),
        Vertex(vector=[4, 4, 4], num_layers=hnsw.num_layers),
        Vertex(vector=[5, 5, 5], num_layers=hnsw.num_layers),
    ]
    for input_vertex in input_vertices:
        hnsw.add(x=input_vertex)

    return hnsw, input_vertices


def test_set_layer_probs_sets_expected_probabilities():
    hnsw = HNSW(M=32, M_0=None, m_L=None)
    assert hnsw.layer_probs == [
        0.96875,
        0.030273437499999986,
        0.0009460449218749991,
        2.956390380859371e-05,
        9.23871994018553e-07,
        2.887099981307982e-08,
    ]


def test_add_sets_entry_point_if_index_is_empty():
    hnsw = HNSW(M=32, M_0=None, m_L=None)
    input_vertex = Vertex(vector=[1, 1, 1], num_layers=hnsw.num_layers)
    insertion_layer = hnsw.add(x=input_vertex)
    assert (
        hnsw.entry_point == input_vertex
    ), "The first vertex should be the entry point."
    assert (
        insertion_layer == 5
    ), "The first vertex should be inserted in the last layer."


def test_add_accepts_a_list_and_converts_it_to_a_vertex():
    hnsw = HNSW(M=32, M_0=None, m_L=None)
    input_vertex = [1, 1, 1]
    hnsw.add(x=input_vertex)
    assert hnsw.entry_point == Vertex(
        vector=input_vertex, num_layers=hnsw.num_layers
    ), "The first vertex should be the entry point."


def test_add_sets_edges_between_vertices_in_both_directions(populated_hnsw):
    new_vertex = Vertex(vector=[10], num_layers=populated_hnsw.num_layers)
    insertion_layer = populated_hnsw.add(new_vertex)
    for i in range(insertion_layer, -1, -1):
        edges = new_vertex.get_edges_in_layer(i)
        assert edges, "There should be edges in the layer."
        for edge in edges:
            assert new_vertex in edge.get_edges_in_layer(
                i
            ), "Edges should be bidirectional."


def test_add_sets_edges_of_input_vertex_as_well_as_edges_of_neighbours_in_all_necessary_layers(
    popoulated_hnsw_2,
):
    hnsw, input_vertices = popoulated_hnsw_2
    assert hnsw.entry_point == input_vertices[0]
    assert input_vertices[0].edges[0] == {
        Vertex(vector=[1, 1, 1]),
        Vertex(vector=[2, 2, 2]),
        Vertex(vector=[3, 3, 3]),
        Vertex(vector=[4, 4, 4]),
    }
    assert input_vertices[0].edges[1] == {Vertex(vector=[3, 3, 3])}

    assert input_vertices[1].edges[0] == {
        Vertex(vector=[0, 0, 0]),
        Vertex(vector=[2, 2, 2]),
        Vertex(vector=[3, 3, 3]),
        Vertex(vector=[4, 4, 4]),
        Vertex(vector=[5, 5, 5]),
    }
    assert input_vertices[1].edges[1] == set()

    assert input_vertices[2].edges[0] == {
        Vertex(vector=[1, 1, 1]),
        Vertex(vector=[0, 0, 0]),
        Vertex(vector=[3, 3, 3]),
        Vertex(vector=[4, 4, 4]),
        Vertex(vector=[5, 5, 5]),
    }
    assert input_vertices[2].edges[1] == set()

    assert input_vertices[3].edges[0] == {
        Vertex(vector=[2, 2, 2]),
        Vertex(vector=[1, 1, 1]),
        Vertex(vector=[0, 0, 0]),
        Vertex(vector=[4, 4, 4]),
        Vertex(vector=[5, 5, 5]),
    }
    assert input_vertices[3].edges[1] == {Vertex(vector=[0, 0, 0])}

    assert input_vertices[4].edges[0] == {
        Vertex(vector=[3, 3, 3]),
        Vertex(vector=[2, 2, 2]),
        Vertex(vector=[1, 1, 1]),
        Vertex(vector=[0, 0, 0]),
        Vertex(vector=[5, 5, 5]),
    }
    assert input_vertices[4].edges[1] == set()

    assert input_vertices[5].edges[0] == {
        Vertex(vector=[4, 4, 4]),
        Vertex(vector=[3, 3, 3]),
        Vertex(vector=[2, 2, 2]),
        Vertex(vector=[1, 1, 1]),
    }
    assert input_vertices[5].edges[1] == set()


def test_add_raises_exception_if_dimension_is_not_consistent(populated_hnsw):
    with pytest.raises(ValueError):
        populated_hnsw.add(Vertex(vector=[1, 1], num_layers=populated_hnsw.num_layers))


def test_search_finds_the_nearest_neighbour(mocker):
    mock_random_levels = [
        0,
        0,
        1,
        0,
        0,
        0,
        1,
        1,
    ]  # The layers that the input vertices will be inserted in
    mocker.patch.object(HNSW, "_get_random_level", side_effect=mock_random_levels)
    mock_probabilities = [
        0.5,
        0.5,
    ]  # The probabilities of inserting an edge into each layer
    mocker.patch.object(
        HNSW, "_set_layer_probabilities", return_value=mock_probabilities
    )
    hnsw = HNSW(M=2)
    input_vertices = [
        Vertex(vector=[0, 0, 0], num_layers=hnsw.num_layers),
        Vertex(vector=[1, 1, 1], num_layers=hnsw.num_layers),
        Vertex(vector=[2, 2, 2], num_layers=hnsw.num_layers),
        Vertex(vector=[3, 3, 3], num_layers=hnsw.num_layers),
        Vertex(vector=[4, 4, 4], num_layers=hnsw.num_layers),
        Vertex(vector=[5, 5, 5], num_layers=hnsw.num_layers),
    ]
    for input_vertex in input_vertices:
        hnsw.add(x=input_vertex)

    nearest_neighbour = hnsw.search(
        Vertex(vector=[1.5, 0.5, 1.5], num_layers=hnsw.num_layers)
    )
    assert nearest_neighbour == Vertex(vector=[1, 1, 1], num_layers=hnsw.num_layers)

    nearest_neighbour = hnsw.search(
        Vertex(vector=[3.1, 2.9, 3.6], num_layers=hnsw.num_layers)
    )
    assert nearest_neighbour == Vertex(vector=[3, 3, 3], num_layers=hnsw.num_layers)

    nearest_neighbour = hnsw.search(
        Vertex(vector=[10, 10, 10], num_layers=hnsw.num_layers)
    )
    assert nearest_neighbour == Vertex(vector=[5, 5, 5], num_layers=hnsw.num_layers)


def test_search_accepts_a_list_and_converts_it_to_a_vector(populated_hnsw):
    nearest_neighbour = populated_hnsw.search([1.5, 0.5, 1.5])
    assert nearest_neighbour
