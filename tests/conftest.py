"""Shared fixtures and instance generators for the test suite.

The tests import ``macroitems`` from the repository root (the package need not
be installed) and the exact reference implementations from
:mod:`tests.bruteforce`.
"""
from __future__ import annotations

import os
import random
import sys
from fractions import Fraction as Fr
from typing import List, Tuple

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from macroitems import Instance  # noqa: E402


def make_instance(p, w, arcs, name="test") -> Instance:
    """An :class:`Instance` from plain Python lists (arcs as ``(i, j)`` pairs,
    j prerequisite of i)."""
    inst = Instance(np.asarray(p, dtype=float), np.asarray(w, dtype=float),
                    np.asarray(list(arcs), dtype=np.int64).reshape(-1, 2), name=name)
    inst.validate()
    return inst


def random_small(rng: random.Random, n_max: int = 9, density: float = 0.3):
    """A random small instance as ``(n, p, w, arcs)`` with integer data.

    Arcs go from a lower to a higher index of a random permutation, so the graph
    is acyclic by construction; the profit alphabet contains zeros and negatives
    and the weight alphabet is small, which makes ties in the ratios common.
    """
    n = rng.randint(1, n_max)
    order = list(range(n))
    rng.shuffle(order)
    arcs = [(order[a], order[b]) for a in range(n) for b in range(a + 1, n)
            if rng.random() < density]
    p = [rng.choice([-3, -2, -1, 0, 1, 2, 2, 3, 3, 4, 6, 9]) for _ in range(n)]
    w = [rng.choice([1, 1, 2, 2, 3, 4]) for _ in range(n)]
    return n, p, w, arcs


# --------------------------------------------------------- degenerate corners
# Cases the random generator would hit only by luck; each is (name, p, w, arcs).
DEGENERATE_CASES: List[Tuple[str, list, list, list]] = [
    ("single item", [3], [2], []),
    ("single worthless item", [-3], [2], []),
    ("empty arc set", [5, -1, 3, 4], [2, 1, 1, 3], []),
    ("all profits negative", [-1, -2, -3, -4], [1, 2, 3, 1], [(0, 1), (1, 2), (3, 2)]),
    ("all profits positive", [1, 2, 3, 4], [4, 3, 2, 1], [(0, 1), (1, 2), (2, 3)]),
    ("all profits zero", [0, 0, 0, 0], [1, 2, 3, 4], [(0, 1), (2, 3)]),
    ("zero-ratio tail", [4, 0, 0], [2, 1, 1], [(1, 0)]),
    ("path graph", [-1, -1, 9, -1], [1, 1, 1, 1], [(1, 0), (2, 1), (3, 2)]),
    ("star, root first", [6, -1, -1, -1], [1, 1, 1, 1], [(0, 1), (0, 2), (0, 3)]),
    ("star, leaves first", [-2, 3, 3, 3], [1, 1, 1, 1], [(1, 0), (2, 0), (3, 0)]),
    # two disjoint continuations of equal ratio: the maximal-tie convention must
    # merge them into a single macroitem
    ("tie, two disjoint items", [2, 2, 4], [1, 1, 2], []),
    ("tie, two disjoint chains", [-1, 5, -1, 5], [1, 1, 1, 1], [(1, 0), (3, 2)]),
    ("tie, three blocks", [3, 3, 1, 1], [1, 1, 1, 1], [(2, 0), (3, 1)]),
    ("tie with a chain and a free item", [4, 4, -2, 8], [2, 2, 1, 3], [(1, 2)]),
    # equal ratios that are *not* tied because one needs the other
    ("nested equal ratios", [2, 2], [1, 1], [(1, 0)]),
    ("negative ratio ties", [-2, -2, 6], [1, 1, 2], []),
]


@pytest.fixture(scope="session")
def degenerate_cases():
    return DEGENERATE_CASES
