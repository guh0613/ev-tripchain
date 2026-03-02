import numpy as np

from ev_tripchain.mobility.mapping import (
    build_mapping_from_node_bus_pairs,
    build_random_onehot_mapping,
)


def test_random_onehot_mapping_rows_sum_to_one() -> None:
    rng = np.random.default_rng(0)
    bus_ids = np.array([10, 11, 12], dtype=int)
    m = build_random_onehot_mapping(n_nodes=7, bus_ids=bus_ids, rng=rng)
    assert m.matrix.shape == (7, 3)
    assert np.allclose(m.matrix.sum(axis=1), 1.0)


def test_mapping_from_pairs_fills_unmapped_nodes() -> None:
    bus_ids = np.array([100, 200], dtype=int)
    m = build_mapping_from_node_bus_pairs(n_nodes=3, bus_ids=bus_ids, node_bus_pairs=[(1, 200)])
    assert m.matrix.shape == (3, 2)
    assert np.allclose(m.matrix.sum(axis=1), 1.0)
    # node 1 maps to bus_id 200 (col 1)
    assert m.matrix[1, 1] == 1.0
    # node 0,2 default to col 0
    assert m.matrix[0, 0] == 1.0
    assert m.matrix[2, 0] == 1.0

