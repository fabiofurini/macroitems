"""Exhaustive verification on small instances: every closure is enumerated and
every quantity of the paper is recomputed in exact rational arithmetic.

Covered:
  * u(lambda) and both extremal optimal closures (Proposition 3.1), on a grid of
    prices that deliberately contains the breakpoints, where the ties live;
  * the canonical macroitem sequence and its ratios (Theorem 3.1), by both
    methods, plus nestedness, partition and strict decrease of the ratios;
  * z(c) on a grid of capacities, against the concave envelope of the closure
    points, which is the LP optimum by integrality of the closure polytope;
  * persistency (Corollary 5.1): x_i = 1 on M_{h-1} and x_i = 0 outside M_h in
    *every* optimal solution, checked against the exact optimal face;
  * the canonical dual certificate and the optimal-face dimensions of the
    companion note, the latter against a rank computation on the tight rows.
"""
from __future__ import annotations

import random
from fractions import Fraction as Fr

import numpy as np
import pytest

from bruteforce import (all_closures, brute_canonical, brute_z, optimal_face_range,
                        u_and_extremes, value_hull)
from conftest import DEGENERATE_CASES, make_instance, random_small
from macroitems import (canonical_dual, canonical_path, is_closure, max_closure,
                        solution_from_path, solve_capacity)

N_RANDOM = 200          # random instances per test, fixed seed
SEED = 20240517


def random_cases(n_max=9, n_cases=N_RANDOM, seed=SEED):
    """``n_cases`` random small instances plus every degenerate corner case."""
    rng = random.Random(seed)
    out = [(f"random#{t}", *random_small(rng, n_max, density=rng.choice([0.0, 0.15, 0.3, 0.5])))
           for t in range(n_cases)]
    out += [(name, len(p), p, w, arcs) for name, p, w, arcs in DEGENERATE_CASES]
    return out


def lambda_grid(seq):
    """Every breakpoint, a point strictly between consecutive ones, and points
    outside the range -- the breakpoints are where the tie convention matters."""
    lams = sorted({lam for _, lam in seq})
    grid = set(lams) | {Fr(0)}
    for lo, hi in zip(lams, lams[1:]):
        grid.add((lo + hi) / 2)
    grid |= {min(lams) - 1, max(lams) + 1, min(lams) - Fr(1, 3), max(lams) + Fr(1, 3)}
    return sorted(grid)


# --------------------------------------------------------------------- closures
def test_extremal_optimal_closures():
    """u(lambda) and the minimal / maximal optimal closures, on a price grid that
    includes every breakpoint (Proposition 3.1 and the tie convention)."""
    n_tied = 0
    for name, n, p, w, arcs in random_cases(n_max=8):
        inst = make_instance(p, w, arcs, name)
        closures = all_closures(n, arcs)
        seq = brute_canonical(n, p, w, arcs, closures)
        for lam in lambda_grid(seq):
            u, lo, hi, n_opt = u_and_extremes(closures, p, w, lam)
            n_tied += n_opt > 1
            a, b = lam.numerator, lam.denominator
            v = inst.p * b - a * inst.w          # integer-scaled node values
            r_hi = max_closure(inst, v, tie="max")
            r_lo = max_closure(inst, v, tie="min")
            ctx = (name, p, w, arcs, str(lam))
            assert Fr(int(r_hi.value), b) == u, ctx
            assert Fr(int(r_lo.value), b) == u, ctx
            assert frozenset(r_hi.closure.tolist()) == hi, ctx
            assert frozenset(r_lo.closure.tolist()) == lo, ctx
            assert is_closure(inst, r_hi.mask) and is_closure(inst, r_lo.mask), ctx
    assert n_tied > 100, "the grid should hit plenty of ties"


# ------------------------------------------------------------- canonical path
@pytest.mark.parametrize("method", ["bisection", "dinkelbach"])
def test_canonical_path(method):
    """The whole canonical sequence: macroitems, exact ratios, nested closed
    prefixes, strictly decreasing ratios (Theorem 3.1)."""
    for name, n, p, w, arcs in random_cases():
        inst = make_instance(p, w, arcs, name)
        seq = brute_canonical(n, p, w, arcs)
        path = canonical_path(inst, method=method)
        ctx = (name, p, w, arcs, method)
        assert [frozenset(I.tolist()) for I in path.macroitems] == [S for S, _ in seq], ctx
        got_ratios = [Fr(int(inst.p[I].sum()), int(inst.w[I].sum())) for I in path.macroitems]
        assert got_ratios == [lam for _, lam in seq], ctx
        assert all(a > b for a, b in zip(got_ratios, got_ratios[1:])), ctx
        # nested closures, partitioning the item set
        masks = [path.closure_mask(n, r) for r in range(path.k + 1)]
        assert masks[0].sum() == 0 and masks[-1].all(), ctx
        for lo, hi in zip(masks, masks[1:]):
            assert np.all(hi | ~lo) and hi.sum() > lo.sum(), ctx
            assert is_closure(inst, hi), ctx
        assert path.check(inst) == {"partition": True, "closed_prefixes": True,
                                    "strictly_decreasing": True}, ctx
        # cumulative points and the index q of the last positive ratio
        assert path.W[-1] == sum(w) and path.P[-1] == sum(p), ctx
        assert path.q == sum(1 for lam in got_ratios if lam > 0), ctx


def test_cumulative_points_are_the_concave_hull():
    """(w(M_r), p(M_r)), r = 0..q, are exactly the vertices of the upper concave
    hull of {(w(C), p(C)) : C closure} -- the geometric form of Theorem 3.1."""
    for name, n, p, w, arcs in random_cases(n_max=8):
        inst = make_instance(p, w, arcs, name)
        path = canonical_path(inst)
        hull = value_hull(all_closures(n, arcs), p, w)
        got = [(Fr(int(path.W[r])), Fr(int(path.P[r]))) for r in range(path.q + 1)]
        assert got == hull, (name, p, w, arcs, got, hull)


# ---------------------------------------------------------------- LP at capacity
def capacity_grid(w):
    """Half-integer capacities from 0 to just past the total weight: hits every
    breakpoint w(M_r) as well as points strictly inside the macroitems."""
    return [Fr(k, 2) for k in range(0, 2 * int(sum(w)) + 3)]


def test_value_function():
    """z(c) from the path, from ``value_function`` and from the Newton search all
    equal the exact LP optimum, and the returned x is feasible and attains it."""
    for name, n, p, w, arcs in random_cases(n_max=8):
        inst = make_instance(p, w, arcs, name)
        path = canonical_path(inst)
        hull = value_hull(all_closures(n, arcs), p, w)
        for c in capacity_grid(w):
            z = brute_z(hull, c)
            fc = float(c)
            ctx = (name, p, w, arcs, str(c))
            assert path.value_function(fc) == pytest.approx(float(z), abs=1e-9), ctx
            for sol in (solution_from_path(inst, path, fc), solve_capacity(inst, fc)):
                assert sol.value == pytest.approx(float(z), abs=1e-9), ctx
                x = sol.x
                assert np.all(x >= -1e-12) and np.all(x <= 1 + 1e-12), ctx
                assert float(inst.w @ x) <= fc + 1e-9, ctx
                if inst.m:
                    assert np.all(x[inst.arcs[:, 0]] <= x[inst.arcs[:, 1]] + 1e-12), ctx
                assert float(inst.p @ x) == pytest.approx(sol.value, abs=1e-9), ctx


def test_capacity_multiplier_and_regions():
    """For a binding capacity 0 < c < w(M_q) the two solvers return the same
    split: F = M_{h-1}, H = I_h, multiplier lambda_h, theta in [0,1]."""
    for name, n, p, w, arcs in random_cases(n_max=8):
        inst = make_instance(p, w, arcs, name)
        path = canonical_path(inst)
        Wq = float(path.W[path.q])
        for c in capacity_grid(w):
            fc = float(c)
            if not 0 < fc < Wq:
                continue
            a = solution_from_path(inst, path, fc)
            b = solve_capacity(inst, fc)
            ctx = (name, p, w, arcs, str(c))
            assert np.array_equal(a.F, b.F) and np.array_equal(a.H, b.H), ctx
            assert a.lam == pytest.approx(b.lam, abs=1e-9), ctx
            assert a.lam == pytest.approx(float(path.ratios[a.h - 1]), abs=1e-9), ctx
            assert np.array_equal(a.F, path.closure_mask(n, a.h - 1)), ctx
            assert 0 <= a.theta <= 1 and 0 <= b.theta <= 1, ctx
            # z(c) = p(M_{h-1}) + lambda_h (c - w(M_{h-1}))
            assert a.value == pytest.approx(
                float(path.P[a.h - 1] + path.ratios[a.h - 1] * (fc - path.W[a.h - 1])), abs=1e-9), ctx


# ------------------------------------------------------------------ persistency
def test_persistency():
    """Corollary 5.1: for a binding capacity, every optimal LP solution has
    x_i = 1 on M_{h-1} and x_i = 0 outside M_h; only the split macroitem I_h can
    differ between optimal solutions."""
    n_nonunique = 0
    n_checked = 0
    for name, n, p, w, arcs in random_cases(n_max=7, n_cases=120):
        inst = make_instance(p, w, arcs, name)
        closures = all_closures(n, arcs)
        path = canonical_path(inst)
        Wq = float(path.W[path.q])
        for c in capacity_grid(w):
            fc = float(c)
            if not 0 < fc < Wq:          # otherwise the capacity is not binding
                continue
            n_checked += 1
            z, lo, hi = optimal_face_range(closures, p, w, c, n)
            sol = solution_from_path(inst, path, fc)
            ctx = (name, p, w, arcs, str(c))
            assert sol.value == pytest.approx(float(z), abs=1e-9), ctx
            for i in range(n):
                if sol.F[i]:
                    assert lo[i] == 1, ctx + ("item forced in", i)
                elif not sol.H[i]:
                    assert hi[i] == 0, ctx + ("item forced out", i)
            n_nonunique += any(lo[i] != hi[i] for i in range(n))
    assert n_checked > 500
    assert n_nonunique > 50, "persistency is vacuous unless optima are often non-unique"


@pytest.mark.slow
def test_brute_force_up_to_twelve_items():
    """The same cross-checks on larger enumerable instances (n <= 12)."""
    rng = random.Random(SEED + 1)
    for t in range(150):
        n, p, w, arcs = random_small(rng, 12, density=rng.choice([0.1, 0.2, 0.35]))
        inst = make_instance(p, w, arcs, f"big#{t}")
        closures = all_closures(n, arcs)
        seq = brute_canonical(n, p, w, arcs, closures)
        for method in ("bisection", "dinkelbach"):
            path = canonical_path(inst, method=method)
            assert [frozenset(I.tolist()) for I in path.macroitems] == [S for S, _ in seq], \
                (t, p, w, arcs, method)
        path = canonical_path(inst)
        for lam in lambda_grid(seq):
            u, mn, mx, _ = u_and_extremes(closures, p, w, lam)
            a, b = lam.numerator, lam.denominator
            v = inst.p * b - a * inst.w
            assert frozenset(max_closure(inst, v, tie="max").closure.tolist()) == mx
            assert frozenset(max_closure(inst, v, tie="min").closure.tolist()) == mn
        hull = value_hull(closures, p, w)
        for c in capacity_grid(w):
            z = float(brute_z(hull, c))
            assert solution_from_path(inst, path, float(c)).value == pytest.approx(z, abs=1e-9)
            assert solve_capacity(inst, float(c)).value == pytest.approx(z, abs=1e-9)


# ------------------------------------------------ dual certificate and faces
def test_canonical_dual_certificate():
    """The canonical dual of Section 5 is feasible and closes the duality gap at
    every nondegenerate capacity."""
    rng = random.Random(SEED + 2)
    n_checked = 0
    for t in range(80):
        n, p, w, arcs = random_small(rng, 7)
        inst = make_instance(p, w, arcs, f"dual#{t}")
        path = canonical_path(inst)
        hull = value_hull(all_closures(n, arcs), p, w)
        for h in range(1, path.q + 1):
            c = float(path.W[h - 1] + path.W[h]) / 2.0     # strictly inside I_h
            if not path.W[h - 1] < c < path.W[h]:
                continue
            sol = solve_capacity(inst, c)
            if sol.degenerate:
                continue
            dual = canonical_dual(inst, sol, c)
            ctx = (t, p, w, arcs, c)
            assert dual.feasible, ctx + (dual.max_violation,)
            assert dual.value == pytest.approx(float(brute_z(hull, Fr(c).limit_denominator(10 ** 9))),
                                               abs=1e-7), ctx
            assert dual.value == pytest.approx(sol.value, abs=1e-7), ctx   # no gap
            n_checked += 1
    assert n_checked > 100


@pytest.mark.slow
def test_optimal_face_dimensions():
    """The primal and dual optimal-face dimensions of the companion note, against
    a rank computation on the tight constraints of the LP."""
    from lpfaces import dual_face_dimension_lp, primal_face_dimension_lp
    from macroitems import face_dimensions

    rng = random.Random(SEED + 3)
    n_checked = n_nontrivial = 0
    for t in range(120):
        n, p, w, arcs = random_small(rng, 7)
        inst = make_instance(p, w, arcs, f"face#{t}")
        path = canonical_path(inst)
        for h in range(1, path.q + 1):
            lo, hi = float(path.W[h - 1]), float(path.W[h])
            c = (lo + hi) / 2.0
            if not lo < c < hi:
                continue
            sol = solve_capacity(inst, c)
            if sol.degenerate:
                continue
            fi = face_dimensions(inst, sol)
            ctx = (t, p, w, arcs, c)
            assert fi.dim_primal == primal_face_dimension_lp(n, p, w, arcs, c, sol.value), ctx
            assert fi.dim_dual == dual_face_dimension_lp(n, p, w, arcs, c, sol.value), ctx
            n_checked += 1
            n_nontrivial += fi.dim_primal > 0
    assert n_checked > 150
    assert n_nontrivial > 0, "no instance with a non-unique primal optimum was generated"
