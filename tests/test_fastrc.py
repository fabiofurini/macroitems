"""The C++ engines for the strongest reduced costs agree exactly with the
reference implementation (one maximum closure per item)."""
from __future__ import annotations

import random

import numpy as np
import pytest

from conftest import make_instance, random_small
from macroitems import canonical_path, solve_capacity
from macroitems.dual import best_reduced_costs
from macroitems.fastrc import fast_reduced_costs

ENGINES = ["cold", "warm", "sap", "tree"]


def cases(n_inst=250, seed=20261126):
    rng = random.Random(seed)
    for t in range(n_inst):
        n, p, w, arcs = random_small(rng, 10)
        inst = make_instance(p, w, arcs, f"fast#{t}")
        path = canonical_path(inst)
        for h in range(1, path.q + 1):
            lo, hi = float(path.W[h - 1]), float(path.W[h])
            c = (lo + hi) / 2.0
            if not lo < c < hi:
                continue
            sol = solve_capacity(inst, c)
            if sol.degenerate or not sol.H.any():
                continue
            yield inst, sol


@pytest.mark.parametrize("engine", ENGINES)
def test_engines_match_reference(engine):
    checked = 0
    for inst, sol in cases():
        ref = best_reduced_costs(inst, sol).value
        got = fast_reduced_costs(inst, sol, engine)
        assert np.allclose(got, ref, atol=1e-9, equal_nan=True), (engine, inst.p, inst.w, inst.arcs)
        checked += 1
    assert checked > 300


def test_decimal_data_are_scaled_exactly():
    inst = make_instance([1.5, -0.25, 3.75, 0.5], [0.5, 1.25, 1.0, 0.75], [(2, 1), (0, 1)])
    path = canonical_path(inst)
    for h in range(1, path.q + 1):
        lo, hi = float(path.W[h - 1]), float(path.W[h])
        c = (lo + hi) / 2.0
        sol = solve_capacity(inst, c)
        if sol.degenerate or not sol.H.any():
            continue
        assert np.allclose(fast_reduced_costs(inst, sol, "tree"), best_reduced_costs(inst, sol).value,
                           atol=1e-9, equal_nan=True)
