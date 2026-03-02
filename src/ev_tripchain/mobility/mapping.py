from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class NodeBusMapping:
    """
    Mapping from traffic nodes/zones -> distribution buses.

    `matrix[i, j]` is the probability/weight of node i mapping to bus j.
    """

    matrix: np.ndarray  # shape (n_nodes, n_buses)
    bus_ids: np.ndarray  # shape (n_buses,), original bus identifiers

    def __post_init__(self) -> None:
        if self.matrix.ndim != 2:
            raise ValueError("matrix must be 2D.")
        if self.bus_ids.ndim != 1:
            raise ValueError("bus_ids must be 1D.")
        if self.matrix.shape[1] != self.bus_ids.shape[0]:
            raise ValueError("matrix.shape[1] must equal len(bus_ids).")
        if self.matrix.shape[0] <= 0 or self.matrix.shape[1] <= 0:
            raise ValueError("matrix must be non-empty.")

    @property
    def n_nodes(self) -> int:
        return int(self.matrix.shape[0])

    @property
    def n_buses(self) -> int:
        return int(self.matrix.shape[1])

    def node_to_bus_col(self) -> np.ndarray:
        """Return the most likely bus column index per node."""
        return np.argmax(self.matrix, axis=1)

    def node_to_bus_id(self) -> np.ndarray:
        """Return the most likely bus id per node."""
        cols = self.node_to_bus_col()
        return self.bus_ids[cols]


def build_random_onehot_mapping(
    *,
    n_nodes: int,
    bus_ids: np.ndarray,
    rng: np.random.Generator,
) -> NodeBusMapping:
    """
    Build a simple one-hot mapping by assigning each node to a bus uniformly at random.
    """
    if n_nodes <= 0:
        raise ValueError("n_nodes must be positive.")
    if bus_ids.size <= 0:
        raise ValueError("bus_ids must be non-empty.")

    bus_ids = np.asarray(bus_ids, dtype=int)
    cols = rng.integers(0, bus_ids.size, size=int(n_nodes))
    mat = np.zeros((int(n_nodes), int(bus_ids.size)), dtype=float)
    mat[np.arange(int(n_nodes)), cols] = 1.0
    return NodeBusMapping(matrix=mat, bus_ids=bus_ids)


def build_mapping_from_node_bus_pairs(
    *,
    n_nodes: int,
    bus_ids: np.ndarray,
    node_bus_pairs: list[tuple[int, int]],
) -> NodeBusMapping:
    """
    Build a deterministic one-hot mapping from explicit (node, bus_id) pairs.
    """
    if n_nodes <= 0:
        raise ValueError("n_nodes must be positive.")
    bus_ids = np.asarray(bus_ids, dtype=int)
    bus_to_col = {int(b): i for i, b in enumerate(bus_ids.tolist())}

    mat = np.zeros((int(n_nodes), int(bus_ids.size)), dtype=float)
    seen = set()
    for node, bus_id in node_bus_pairs:
        node_i = int(node)
        if node_i < 0 or node_i >= n_nodes:
            raise ValueError(f"node out of range: {node_i}")
        if node_i in seen:
            raise ValueError(f"duplicate node mapping: {node_i}")
        if int(bus_id) not in bus_to_col:
            raise ValueError(f"unknown bus_id: {bus_id}")
        mat[node_i, bus_to_col[int(bus_id)]] = 1.0
        seen.add(node_i)

    # fallback for unmapped nodes: map to bus 0
    unmapped = np.where(mat.sum(axis=1) == 0.0)[0]
    if unmapped.size:
        mat[unmapped, 0] = 1.0
    return NodeBusMapping(matrix=mat, bus_ids=bus_ids)


def sample_bus_col_for_node(
    mapping: NodeBusMapping, *, node: int, rng: np.random.Generator
) -> int:
    """
    Sample a bus column index for a node using mapping weights.
    """
    node_i = int(node)
    if node_i < 0 or node_i >= mapping.n_nodes:
        raise ValueError("node out of range.")
    w = mapping.matrix[node_i, :].astype(float, copy=False)
    s = float(w.sum())
    if s <= 0:
        return 0
    p = w / s
    return int(rng.choice(mapping.n_buses, p=p))

