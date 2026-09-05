"""The 8-item running example of the paper, checked against the printed numbers.

The paper numbers the items 1..8; the code is 0-based, so every set below is
converted with ``+1`` before it is compared with the paper's sets.  Ratios and
values of ``u`` are checked in exact rational arithmetic: the data are integers,
so every quantity of Theorem 3.1 and Section 4 is a rational number and there is
no reason to accept a tolerance.
"""
from __future__ import annotations

from fractions import Fraction as Fr

import numpy as np
import pytest

from macroitems import (canonical_dual, canonical_path, canonical_reduced_costs,
                        face_dimensions, max_closure, running_example,
                        solution_from_path, solve_capacity)

# Paper numbering (1-based) of the canonical macroitems and their ratios.
PAPER_MACROITEMS = [[3, 6], [1, 2, 5], [4, 7, 8]]
PAPER_RATIOS = [Fr(2), Fr(3, 2), Fr(1)]
# (w(M_r), p(M_r)) for r = 0..3.
PAPER_CUMULATIVE = [(0, 0), (2, 4), (6, 10), (10, 14)]


def paper_sets(macroitems):
    """0-based index arrays -> sorted 1-based lists, as printed in the paper."""
    return [sorted((I + 1).tolist()) for I in macroitems]


def exact_ratios(inst, macroitems):
    """lambda_r = p(I_r) / w(I_r) as a Fraction (Theorem 3.1)."""
    return [Fr(int(round(inst.p[I].sum())), int(round(inst.w[I].sum()))) for I in macroitems]


@pytest.fixture(scope="module")
def inst():
    return running_example()


@pytest.mark.parametrize("method", ["bisection", "dinkelbach"])
def test_canonical_sequence(inst, method):
    """The canonical sequence is {3,6}, {1,2,5}, {4,7,8} with ratios 2, 3/2, 1."""
    path = canonical_path(inst, method=method)
    assert paper_sets(path.macroitems) == PAPER_MACROITEMS
    assert exact_ratios(inst, path.macroitems) == PAPER_RATIOS
    assert path.k == 3 and path.q == 3


def test_cumulative_points(inst):
    """The cumulative points (w(M_r), p(M_r)) are (0,0), (2,4), (6,10), (10,14)."""
    path = canonical_path(inst)
    assert list(zip(path.W.astype(int).tolist(), path.P.astype(int).tolist())) == PAPER_CUMULATIVE


def test_prefixes_are_nested_closures(inst):
    """M_0 = {} subset M_1 subset M_2 subset M_3 = I, and every M_r is a closure."""
    from macroitems import is_closure
    path = canonical_path(inst)
    masks = [path.closure_mask(inst.n, r) for r in range(path.k + 1)]
    assert masks[0].sum() == 0 and masks[-1].all()
    for lo, hi in zip(masks, masks[1:]):
        assert np.all(hi | ~lo) and hi.sum() > lo.sum()
    assert all(is_closure(inst, m) for m in masks)


def test_lp_at_capacity_four(inst):
    """At c = 4 the optimum is 7, split on I_2 = {1,2,5} with theta = 1/2 and
    capacity multiplier lambda_2 = 3/2."""
    path = canonical_path(inst)
    sol = solution_from_path(inst, path, 4.0)
    assert sol.h == 2                              # the second macroitem is split
    assert np.array_equal(np.flatnonzero(sol.H), np.sort(path.macroitems[1]))
    assert np.array_equal(np.flatnonzero(sol.F), np.sort(path.macroitems[0]))
    assert Fr(sol.value).limit_denominator(10**6) == Fr(7)
    assert Fr(sol.theta).limit_denominator(10**6) == Fr(1, 2)
    assert Fr(sol.lam).limit_denominator(10**6) == Fr(3, 2)
    # x = 1 on {3,6}, 1/2 on {1,2,5}, 0 on {4,7,8}   (paper numbering)
    expected = np.array([.5, .5, 1., 0., .5, 1., 0., 0.])
    assert np.array_equal(sol.x, expected)
    assert float(inst.w @ sol.x) == 4.0
    # solve_capacity finds the same optimum by a Newton search (it does not know
    # the index of the split macroitem and reports h = -1 by convention)
    direct = solve_capacity(inst, 4.0)
    assert np.array_equal(direct.x, expected)
    assert direct.value == 7.0 and direct.lam == 1.5 and direct.theta == 0.5
    assert np.array_equal(direct.F, sol.F) and np.array_equal(direct.H, sol.H)


def u_paper(lam: Fr) -> Fr:
    """The paper's piecewise formula for u(lambda) = max_C (p(C) - lambda w(C))."""
    return max(Fr(0), 4 - 2 * lam, 10 - 6 * lam, 14 - 10 * lam)


@pytest.mark.parametrize("lam", [Fr(4), Fr(5, 2), Fr(2), Fr(7, 4), Fr(3, 2),
                                 Fr(5, 4), Fr(1), Fr(1, 2), Fr(0)])
def test_value_function_u(inst, lam):
    """u(lambda) matches the paper: 0 for lambda >= 2, 4-2L on [3/2,2],
    10-6L on [1,3/2], 14-10L for lambda <= 1."""
    # scale the node values to integers so the max-flow backend is exact:
    # v_i = p_i - (a/b) w_i has the same sign pattern as b p_i - a w_i.
    a, b = lam.numerator, lam.denominator
    res = max_closure(inst, inst.p * b - a * inst.w, tie="max")
    assert Fr(int(res.value), b) == u_paper(lam)


def test_tie_convention_at_a_breakpoint(inst):
    """At the breakpoint lambda_2 = 3/2 the optimal closures form a lattice with
    minimal element {3,6} and maximal element {1,2,3,5,6} (paper numbering); the
    library's convention is the maximal one."""
    lam = Fr(3, 2)
    v = inst.p * lam.denominator - lam.numerator * inst.w
    hi = max_closure(inst, v, tie="max")
    lo = max_closure(inst, v, tie="min")
    assert sorted((hi.closure + 1).tolist()) == [1, 2, 3, 5, 6]
    assert sorted((lo.closure + 1).tolist()) == [3, 6]
    # both are optimal: u(3/2) = 1
    assert Fr(int(hi.value), 2) == Fr(int(lo.value), 2) == u_paper(lam) == Fr(1)
    # and the maximal one is M_2 = I_1 + I_2 of the canonical path
    path = canonical_path(inst)
    assert np.array_equal(hi.mask, path.closure_mask(inst.n, 2))


def test_dual_certificate_at_capacity_four(inst):
    """The canonical dual of Section 5 is feasible and closes the gap at c = 4."""
    sol = solve_capacity(inst, 4.0)
    dual = canonical_dual(inst, sol, 4.0)
    assert dual.feasible and dual.max_violation == 0.0
    assert dual.lam == 1.5
    assert dual.value == pytest.approx(7.0)          # = c*lambda + sum(mu) = z(4)
    assert np.all(dual.mu >= 0) and np.all(dual.alpha >= 0)


def test_reduced_costs_at_capacity_four(inst):
    """Canonical reduced costs w_i |lambda_r - lambda_h|, zero on the split
    macroitem I_2 (paper items 1, 2, 5)."""
    path = canonical_path(inst)
    rc = canonical_reduced_costs(inst, path, 2)
    assert np.array_equal(rc, np.array([0., 0., .5, 1., 0., .5, .5, .5]))


def test_optimal_face_dimensions_at_capacity_four(inst):
    """The companion note's face dimensions at c = 4: a unique primal optimum
    (dimension 0) and a three-dimensional optimal dual face."""
    sol = solve_capacity(inst, 4.0)
    fi = face_dimensions(inst, sol)
    assert fi.dim_primal == 0 and fi.k0 == 1
    assert fi.dim_dual == 3
