from src.hierarchical_navigable_small_worlds import HVSW

def test_set_layer_probs_sets_expected_probabilities_and_cumulative_neighbour_counts():
    hvsw = HVSW(M=32, M_0=None, m_L=None)
    assert hvsw.layer_probs == [
        0.96875,
        0.030273437499999986,
        0.0009460449218749991,
        2.956390380859371e-05,
        9.23871994018553e-07,
        2.887099981307982e-08,
    ]
    assert hvsw.cumulative_nn_per_level == [64, 96, 128, 160, 192, 224]
