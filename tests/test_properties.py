"""Property-based tests: invariants that must hold for *any* instance.

Hypothesis generates small random DAGs with integer profits (any sign) and
positive integer weights, and every test below states a structural property of
the theory rather than a value computed elsewhere:

  * the prefixes M_r of the canonical path are closures and are nested;
  * the macroitems partition the item set;
  * the ratios lambda_r are strictly decreasing -- checked with Fractions, since
    a float comparison could hide two breakpoints that coincide;
  * z(c) is concave and nondecreasing in c;
  * the returned x is feasible and its objective value is exactly the reported
    one, from both solvers.

Examples are kept small (n <= 8) so the suite stays fast; the exhaustive checks
against enumeration live in ``test_brute_force.py``.
"""
from __future__ import annotations

from fractions import Fraction as Fr

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from conftest import make_instance
from macroitems import canonical_path, is_closure, solution_from_path, solve_capacity

SETTINGS = settings(max_examples=150, deadline=None, derandomize=True,
                    suppress_health_check=[HealthCheck.too_slow])


@st.composite
def instances(draw, n_max: int = 8):
    """A random DAG on ``n`` items: arcs (i, j) with i before j in a random
    topological order, integer profits of either sign and positive weights."""
    n = draw(st.integers(min_value=1, max_value=n_max))
    order = draw(st.permutations(range(n)))
    pairs = [(order[a], order[b]) for a in range(n) for b in range(a + 1, n)]
    keep = draw(st.lists(st.booleans(), min_size=len(pairs), max_size=len(pairs)))
    arcs = [e for e, k in zip(pairs, keep) if k]
    p = draw(st.lists(st.integers(min_value=-9, max_value=9), min_size=n, max_size=n))
    w = draw(st.lists(st.integers(min_value=1, max_value=9), min_size=n, max_size=n))
    return make_instance(p, w, arcs, name=f"hyp{n}")


def exact_ratios(inst, macroitems):
    """The ratios in exact rational arithmetic (Theorem 3.1: lambda_r = p(I_r)/w(I_r))."""
    return [Fr(int(round(inst.p[I].sum())), int(round(inst.w[I].sum()))) for I in macroitems]


@SETTINGS
@given(inst=instances(), method=st.sampled_from(["bisection", "dinkelbach"]))
def test_path_is_a_nested_chain_of_closures(inst, method):
    """M_0 = {} subset ... subset M_k = I, every M_r a closure, the macroitems a
    partition of the item set."""
    path = canonical_path(inst, method=method)
    n = inst.n
    masks = [path.closure_mask(n, r) for r in range(path.k + 1)]
    assert masks[0].sum() == 0 and masks[-1].all()
    for lo, hi in zip(masks, masks[1:]):
        assert np.all(hi | ~lo)              # nested
        assert hi.sum() > lo.sum()           # strictly growing: macroitems non-empty
        assert is_closure(inst, hi)
    covered = np.zeros(n, dtype=bool)
    for I in path.macroitems:
        assert not covered[I].any()          # disjoint
        covered[I] = True
    assert covered.all()                     # covering
    assert path.check(inst) == {"partition": True, "closed_prefixes": True,
                                "strictly_decreasing": True}


@SETTINGS
@given(inst=instances())
def test_ratios_strictly_decrease_exactly(inst):
    """lambda_1 > ... > lambda_k in exact arithmetic, and lambda_r is the ratio of
    the increment (Theorem 3.1); the cumulative arrays are their partial sums."""
    path = canonical_path(inst)
    ratios = exact_ratios(inst, path.macroitems)
    assert all(a > b for a, b in zip(ratios, ratios[1:]))
    assert np.allclose(path.ratios, [float(r) for r in ratios])
    assert path.W[0] == 0 and path.P[0] == 0
    for r, I in enumerate(path.macroitems, 1):
        assert path.W[r] - path.W[r - 1] == pytest.approx(inst.w[I].sum())
        assert path.P[r] - path.P[r - 1] == pytest.approx(inst.p[I].sum())
    assert path.q == sum(1 for r in ratios if r > 0)


@SETTINGS
@given(inst=instances())
def test_value_function_concave_and_nondecreasing(inst):
    """z(c) is concave and nondecreasing: consecutive slopes are the lambda_r,
    which are positive up to q and after which z is flat."""
    path = canonical_path(inst)
    top = float(path.W[-1]) + 2.0
    cs = np.unique(np.concatenate([np.linspace(0.0, top, 41), path.W]))
    zs = np.array([path.value_function(float(c)) for c in cs])
    assert np.all(np.diff(zs) >= -1e-9), "z(c) must be nondecreasing"
    slopes = np.diff(zs) / np.diff(cs)
    assert np.all(np.diff(slopes) <= 1e-9), "z(c) must be concave"
    assert np.all(slopes >= -1e-12)
    assert zs[0] == 0.0
    assert zs[-1] == pytest.approx(float(path.P[path.q]))


@SETTINGS
@given(inst=instances(), k=st.integers(min_value=0, max_value=40))
def test_returned_solution_is_feasible_and_attains_its_value(inst, k):
    """x is feasible for the LP (bounds, precedence, capacity) and p.x equals the
    reported optimum; both solvers report the same value as z(c)."""
    path = canonical_path(inst)
    c = float(path.W[-1]) * k / 40.0
    a = solution_from_path(inst, path, c)
    b = solve_capacity(inst, c)
    z = path.value_function(c)
    scale = max(1.0, abs(z))
    for sol in (a, b):
        x = sol.x
        assert np.all(x >= -1e-12) and np.all(x <= 1 + 1e-12)
        assert float(inst.w @ x) <= c + 1e-9 * max(1.0, c)
        if inst.m:
            assert np.all(x[inst.arcs[:, 0]] <= x[inst.arcs[:, 1]] + 1e-12)
        assert float(inst.p @ x) == pytest.approx(sol.value, abs=1e-9 * scale)
        assert sol.value == pytest.approx(z, abs=1e-9 * scale)
        # the three regions partition the items
        assert np.all(sol.F.astype(int) + sol.H.astype(int) + sol.Z.astype(int) == 1)


@SETTINGS
@given(inst=instances())
def test_split_is_a_single_macroitem(inst):
    """For a binding capacity the optimum is 1 on M_{h-1}, theta on I_h and 0
    elsewhere: F is a prefix closure and H is exactly the next macroitem."""
    path = canonical_path(inst)
    Wq = float(path.W[path.q])
    for r in range(1, path.q + 1):
        for frac in (0.0, 0.25, 1.0):        # start of I_r, inside it, its end
            c = float(path.W[r - 1]) + frac * float(path.W[r] - path.W[r - 1])
            if not 0 < c < Wq:               # otherwise the capacity is not binding
                continue
            sol = solution_from_path(inst, path, c)
            assert 1 <= sol.h <= path.q
            assert np.array_equal(sol.F, path.closure_mask(inst.n, sol.h - 1))
            assert np.array_equal(np.flatnonzero(sol.H),
                                  np.sort(path.macroitems[sol.h - 1]))
            assert is_closure(inst, sol.F) and is_closure(inst, sol.F | sol.H)
            assert 0 <= sol.theta <= 1 + 1e-12
            assert sol.lam == pytest.approx(float(path.ratios[sol.h - 1]))


@SETTINGS
@given(inst=instances())
def test_zero_capacity(inst):
    """At c = 0 the only feasible point is x = 0, so z(0) = 0 and every solver
    must return it (a regression guard: split_index(0) is 0, and the prefix
    M_{h-1} for h = 0 must not wrap around to the end of the chain)."""
    path = canonical_path(inst)
    assert path.value_function(0.0) == 0.0
    for sol in (solution_from_path(inst, path, 0.0), solve_capacity(inst, 0.0)):
        assert np.all(sol.x == 0.0)
        assert sol.value == 0.0
        assert float(inst.w @ sol.x) == 0.0
