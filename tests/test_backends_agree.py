"""The maximum-flow backends are interchangeable on integer data.

The maximal and the minimal minimum cut of a network do not depend on *which*
maximum flow is found (they are read off the residual graph, which has the same
reachability for every maximum flow), so ortools, igraph and scipy must return
byte-identical closures -- and therefore identical canonical paths -- on integer
node values.  Backends that are not installed are skipped.

``canonical_path`` has no backend argument: it builds its sub-solvers with the
default backend, so the tests below patch :func:`macroitems._maxflow.default_backend`.
"""
from __future__ import annotations

from fractions import Fraction as Fr

import numpy as np
import pytest

import macroitems._maxflow as _maxflow
from macroitems import canonical_path, layered_grid, max_closure, random_dag, solve_capacity
from macroitems.closure import ClosureSolver

BACKENDS = _maxflow.available_backends()
ALL = ("ortools", "igraph", "scipy")


@pytest.fixture(params=ALL)
def backend(request, monkeypatch):
    """Make ``request.param`` the default backend for the whole call."""
    name = request.param
    if name not in BACKENDS:
        pytest.skip(f"backend {name!r} is not installed")
    monkeypatch.setattr(_maxflow, "default_backend", lambda: name)
    return name


INSTANCES = [
    ("dag small", lambda: random_dag(20, 1.5, seed=1)),
    ("dag medium", lambda: random_dag(80, 2.5, seed=2)),
    ("dag sparse", lambda: random_dag(150, 0.4, seed=3)),
    ("dag dense", lambda: random_dag(120, 5.0, seed=4)),
    ("dag unit weights", lambda: random_dag(70, 2.0, seed=5, w_range=(1, 1))),
    ("grid 4x4x3", lambda: layered_grid(4, 4, 3, cone=5, seed=1)),
    ("grid 5x5x4", lambda: layered_grid(5, 5, 4, cone=9, seed=2)),
]
# Coefficients wide enough that the scaled values v = b p - a w overflow int32:
# only the int64 backends can take it (see test_scipy_rejects_large_capacities).
WIDE = lambda: random_dag(60, 2.0, seed=5, p_range=(-5000, 20000), w_range=(1, 9999))


def test_at_least_one_backend():
    assert BACKENDS, "no maximum-flow backend is installed"


@pytest.mark.parametrize("name,build", INSTANCES, ids=[n for n, _ in INSTANCES])
def test_backend_selection(backend, name, build):
    """The patched default really reaches the solver."""
    assert ClosureSolver(build()).backend == backend


@pytest.mark.parametrize("name,build", INSTANCES, ids=[n for n, _ in INSTANCES])
@pytest.mark.parametrize("tie", ["max", "min"])
def test_closures_identical(backend, name, build, tie):
    """The same extremal maximum closure comes out of every backend, at a price
    inside a macroitem and at a breakpoint, where ties make the cut non-unique."""
    inst = build()
    path = canonical_path(inst)
    lams = [Fr(0)] + [Fr(int(round(inst.p[I].sum())), int(round(inst.w[I].sum())))
                      for I in path.macroitems[:4]]
    for lam in lams:
        a, b = lam.numerator, lam.denominator
        v = inst.p * b - a * inst.w                    # integer node values
        got = max_closure(inst, v, tie=tie, backend=backend)
        ref = max_closure(inst, v, tie=tie, backend=BACKENDS[0])
        ctx = (name, backend, tie, str(lam))
        assert np.array_equal(got.mask, ref.mask), ctx
        assert got.value == ref.value, ctx
        assert got.flow_value == ref.flow_value, ctx


@pytest.mark.parametrize("name,build", INSTANCES, ids=[n for n, _ in INSTANCES])
@pytest.mark.parametrize("method", ["bisection", "dinkelbach"])
def test_canonical_path_identical(backend, name, build, method):
    """Identical canonical macroitem sequences from every backend."""
    inst = build()
    got = canonical_path(inst, method=method)
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(_maxflow, "default_backend", lambda: BACKENDS[0])
        ref = canonical_path(inst, method=method)
    ctx = (name, backend, method)
    assert [I.tolist() for I in got.macroitems] == [I.tolist() for I in ref.macroitems], ctx
    assert np.array_equal(got.W, ref.W) and np.array_equal(got.P, ref.P), ctx
    assert np.array_equal(got.ratios, ref.ratios), ctx


@pytest.mark.parametrize("name,build", INSTANCES, ids=[n for n, _ in INSTANCES])
def test_solve_capacity_identical(backend, name, build):
    """Identical LP solutions at a grid of capacities."""
    inst = build()
    ref_path = canonical_path(inst)
    for c in np.linspace(0.0, float(ref_path.W[-1]), 9):
        got = solve_capacity(inst, float(c))
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(_maxflow, "default_backend", lambda: BACKENDS[0])
            ref = solve_capacity(inst, float(c))
        ctx = (name, backend, c)
        assert np.array_equal(got.x, ref.x), ctx
        assert got.value == ref.value and got.lam == ref.lam, ctx


def test_exact_backends_reject_fractional_capacities():
    """ortools and scipy refuse to round non-integer capacities silently."""
    for name in ("ortools", "scipy"):
        if name not in BACKENDS:
            continue
        net = _maxflow.MaxFlowNetwork(3, np.array([0, 1]), np.array([1, 2]), backend=name)
        with pytest.raises(ValueError, match="integer capacities"):
            net.solve(np.array([1.5, 2.0]), 0, 2)


def test_scipy_rejects_large_capacities():
    """``scipy.sparse.csgraph.maximum_flow`` computes in int32 and truncates a
    larger capacity silently (a single arc of capacity 2**31 gets a flow of 0),
    so the backend must refuse rather than return a wrong cut."""
    if "scipy" not in BACKENDS:
        pytest.skip("scipy is not installed")
    net = _maxflow.MaxFlowNetwork(3, np.array([0, 1]), np.array([1, 2]), backend="scipy")
    assert net.solve(np.array([2 ** 31 - 1, 2 ** 31 - 1]), 0, 2)[0] == 2 ** 31 - 1
    with pytest.raises(ValueError, match="int32"):
        net.solve(np.array([2 ** 31, 2 ** 31]), 0, 2)
    # and the refusal surfaces on a real instance, instead of a wrong closure
    inst = WIDE()
    path = canonical_path(inst)
    lam = Fr(int(inst.p[path.macroitems[0]].sum()), int(inst.w[path.macroitems[0]].sum()))
    v = inst.p * lam.denominator - lam.numerator * inst.w
    with pytest.raises(ValueError, match="int32"):
        max_closure(inst, v, tie="max", backend="scipy")


def test_int64_backends_agree_on_wide_coefficients():
    """ortools and igraph, which are not limited to int32, still agree there."""
    exact = [b for b in ("ortools", "igraph") if b in BACKENDS]
    if len(exact) < 2:
        pytest.skip("need both ortools and igraph")
    inst = WIDE()
    paths = []
    for name in exact:
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(_maxflow, "default_backend", lambda name=name: name)
            paths.append(canonical_path(inst))
    a, b = paths
    assert [I.tolist() for I in a.macroitems] == [I.tolist() for I in b.macroitems]
    assert np.array_equal(a.W, b.W) and np.array_equal(a.P, b.P)


def test_float_values_use_a_float_backend():
    """Genuinely fractional node values are routed to igraph rather than rounded
    (:meth:`ClosureSolver._network_for`)."""
    if "igraph" not in BACKENDS:
        pytest.skip("igraph is not installed")
    inst = random_dag(30, 2.0, seed=7)
    v = inst.p - 1.5 * inst.w                          # half-integers
    exact = max_closure(inst, 2 * v, tie="max")        # same closure, integral
    approx = max_closure(inst, v, tie="max")
    assert np.array_equal(exact.mask, approx.mask)
    assert approx.value == pytest.approx(exact.value / 2)
