"""The two algorithms for the canonical sequence, and the two ways of solving the
LP at a capacity, must return the same object.

``canonical_path`` computes the same mathematical object (Theorem 3.1) either by
geometric bisection on the breakpoints or by repeated maximum-ratio closure
extraction with Dinkelbach iterations; the sequence is canonical, so the two must
agree item by item.  ``solve_capacity`` finds the split by a Newton search on the
weight price while ``solution_from_path`` reads it off a precomputed path: for a
binding capacity they must return the same primal solution and multiplier.
"""
from __future__ import annotations

from fractions import Fraction as Fr

import numpy as np
import pytest

from macroitems import _maxflow

from macroitems import (canonical_path, layered_grid, random_dag, solution_from_path,
                        solve_capacity)

# 100 random DAGs: 10 sizes x 5 densities x 2 seeds.  Plus mining-like grids.
DAG_CASES = [(n, d, seed)
             for n in (2, 5, 12, 25, 40, 60, 90, 130, 200, 300)
             for d in (0.0, 0.5, 1.5, 2.5, 4.0)
             for seed in (0, 1)]
GRID_CASES = [(3, 3, 3, 5), (4, 4, 3, 5), (5, 5, 4, 5), (4, 4, 4, 9), (6, 6, 3, 9)]


def exact_ratios(inst, macroitems):
    return [Fr(int(round(inst.p[I].sum())), int(round(inst.w[I].sum()))) for I in macroitems]


def check_methods_agree(inst):
    a = canonical_path(inst, method="bisection")
    b = canonical_path(inst, method="dinkelbach")
    assert [frozenset(I.tolist()) for I in a.macroitems] == \
           [frozenset(I.tolist()) for I in b.macroitems], inst.name
    assert exact_ratios(inst, a.macroitems) == exact_ratios(inst, b.macroitems), inst.name
    assert np.array_equal(a.W, b.W) and np.array_equal(a.P, b.P), inst.name
    assert a.q == b.q
    assert a.check(inst) == {"partition": True, "closed_prefixes": True,
                             "strictly_decreasing": True}, inst.name
    return a


def check_capacity_solvers_agree(inst, path, n_grid=25):
    """Over a capacity grid: same value everywhere, same split where binding."""
    Wq, Wtot = float(path.W[path.q]), float(path.W[-1])
    caps = sorted(set(np.linspace(0.0, Wtot * 1.05 + 1.0, n_grid).tolist())
                  | set(path.W.tolist())                       # exact breakpoints
                  | {float(x) + 0.5 for x in path.W[:-1]})
    for c in caps:
        a = solution_from_path(inst, path, c)
        b = solve_capacity(inst, c)
        scale = max(1.0, abs(a.value))
        ctx = (inst.name, c)
        assert a.value == pytest.approx(b.value, rel=1e-9, abs=1e-9 * scale), ctx
        assert a.value == pytest.approx(path.value_function(c), rel=1e-9, abs=1e-9 * scale), ctx
        for sol in (a, b):
            x = sol.x
            assert np.all(x >= -1e-12) and np.all(x <= 1 + 1e-12), ctx
            assert float(inst.w @ x) <= c + 1e-7 * max(1.0, c), ctx
            if inst.m:
                assert np.all(x[inst.arcs[:, 0]] <= x[inst.arcs[:, 1]] + 1e-12), ctx
            assert float(inst.p @ x) == pytest.approx(sol.value, abs=1e-7 * scale), ctx
        if 0 < c < Wq:
            assert np.array_equal(a.F, b.F) and np.array_equal(a.H, b.H), ctx
            assert np.array_equal(a.Z, b.Z), ctx
            assert a.lam == pytest.approx(b.lam, rel=1e-9), ctx
            assert a.theta == pytest.approx(b.theta, rel=1e-9, abs=1e-12), ctx


@pytest.mark.parametrize("n,degree,seed", DAG_CASES)
def test_random_dag(n, degree, seed):
    inst = random_dag(n, avg_out_degree=degree, seed=seed)
    path = check_methods_agree(inst)
    check_capacity_solvers_agree(inst, path)


@pytest.mark.parametrize("nx,ny,nz,cone", GRID_CASES)
def test_layered_grid(nx, ny, nz, cone):
    inst = layered_grid(nx, ny, nz, cone=cone, seed=nx * 10 + nz)
    path = check_methods_agree(inst)
    check_capacity_solvers_agree(inst, path)


@pytest.mark.skipif(
    not {"ortools", "igraph"} & set(_maxflow.available_backends()),
    reason="wide coefficients exceed int32, so they need ortools or igraph")
def test_wide_profit_and_weight_ranges():
    """Larger coefficients, where the integer scaling v = b p - a w has to stay
    inside the exact regime of the max-flow backend.

    With only the scipy backend, whose arithmetic is int32, these instances
    cannot be solved at all -- the backend refuses them rather than returning a
    wrong cut -- so the test is skipped instead of asserting a limitation."""
    for seed in range(10):
        inst = random_dag(60, avg_out_degree=2.0, seed=1000 + seed,
                          p_range=(-5000, 20000), w_range=(1, 9999))
        path = check_methods_agree(inst)
        check_capacity_solvers_agree(inst, path, n_grid=9)


@pytest.mark.slow
def test_larger_instances():
    """Same checks on instances big enough to have many breakpoints."""
    for seed in range(3):
        inst = random_dag(1200, avg_out_degree=2.5, seed=seed)
        path = check_methods_agree(inst)
        assert path.k >= 5
        check_capacity_solvers_agree(inst, path, n_grid=9)
    for nz in (5, 6):
        inst = layered_grid(8, 8, nz, cone=5, seed=nz)
        path = check_methods_agree(inst)
        check_capacity_solvers_agree(inst, path, n_grid=9)
