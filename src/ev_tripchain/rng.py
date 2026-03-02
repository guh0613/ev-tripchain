from __future__ import annotations

import numpy as np


def make_rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def make_rng_for(*keys: int) -> np.random.Generator:
    """
    Create a deterministic RNG for a tuple of integer keys.

    Use this when a function may be called multiple times (e.g. during binary search),
    but you still want results to be reproducible for the same inputs.
    """
    ss = np.random.SeedSequence(list(keys))
    return np.random.default_rng(ss)
