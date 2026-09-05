"""Without precedence arcs the theory must collapse to the classical knapsack LP.

With an empty arc set every subset is a closure, so the maximum-ratio closure of
Theorem 3.1 is simply the set of all items of maximum ratio p_i / w_i: the
canonical sequence groups the items by equal ratio in decreasing order, and the
macroitem ratios are those distinct values.  The LP value z(c) is then Dantzig's
greedy bound, which is recomputed here from scratch in exact arithmetic.
"""
from __future__ import annotations

import random
from fractions import Fraction as Fr

import numpy as np
import pytest

from conftest import make_instance
from macroitems import canonical_path, random_dag, solution_from_path, solve_capacity

SEED = 424242


def dantzig(p, w, c):
    """Dantzig's rule, written out independently and exactly.

    Sort the items by nonincreasing profit-to-weight ratio, fill the knapsack
    greedily and split the last item; items with a nonpositive ratio are never
    taken, since their weight is positive and would only consume capacity.
    Returns ``(z(c), x)`` with x a list of Fractions.
    """
    c = Fr(c)
    n = len(p)
    x = [Fr(0)] * n
    order = sorted(range(n), key=lambda i: Fr(int(p[i]), int(w[i])), reverse=True)
    z = Fr(0)
    left = c
    for i in order:
        if Fr(int(p[i]), int(w[i])) <= 0 or left <= 0:
            break
        take = min(Fr(1), left / int(w[i]))
        x[i] = take
        z += take * int(p[i])
        left -= take * int(w[i])
    return z, x


def ratio_groups(p, w):
    """Items grouped by equal ratio, in decreasing order of the ratio."""
    ratios = sorted({Fr(int(p[i]), int(w[i])) for i in range(len(p))}, reverse=True)
    return [(r, frozenset(i for i in range(len(p)) if Fr(int(p[i]), int(w[i])) == r))
            for r in ratios]


def cases():
    """Arc-free instances: hand-made corners plus 60 random ones with a small
    ratio alphabet, so that ties between items are the rule and not the exception."""
    out = [
        ("single", [3], [2]),
        ("single negative", [-3], [2]),
        ("all equal ratios", [2, 4, 6], [1, 2, 3]),
        ("all negative", [-1, -2, -3], [3, 2, 1]),
        ("all zero", [0, 0, 0], [1, 2, 3]),
        ("mixed signs", [5, 0, -2, 7, 3], [1, 3, 2, 7, 3]),
        ("unit weights", [4, 4, 1, 9, -2], [1, 1, 1, 1, 1]),
    ]
    rng = random.Random(SEED)
    for t in range(60):
        n = rng.randint(1, 10)
        p = [rng.choice([-4, -1, 0, 1, 2, 3, 4, 6, 8, 12]) for _ in range(n)]
        w = [rng.choice([1, 2, 3, 4, 6]) for _ in range(n)]
        out.append((f"random#{t}", p, w))
    return out


CASES = cases()


@pytest.mark.parametrize("name,p,w", CASES, ids=[c[0] for c in CASES])
def test_canonical_sequence_groups_by_ratio(name, p, w):
    """With no arcs the macroitems are the equal-ratio groups, in decreasing
    order of the ratio."""
    inst = make_instance(p, w, [], name)
    expected = ratio_groups(p, w)
    for method in ("bisection", "dinkelbach"):
        path = canonical_path(inst, method=method)
        got_sets = [frozenset(I.tolist()) for I in path.macroitems]
        got_ratios = [Fr(int(inst.p[I].sum()), int(inst.w[I].sum())) for I in path.macroitems]
        assert got_sets == [S for _, S in expected], (name, method)
        assert got_ratios == [r for r, _ in expected], (name, method)


@pytest.mark.parametrize("name,p,w", CASES, ids=[c[0] for c in CASES])
def test_value_equals_dantzig_bound(name, p, w):
    """z(c) is Dantzig's greedy bound at every capacity of a half-integer grid."""
    inst = make_instance(p, w, [], name)
    path = canonical_path(inst)
    for k in range(0, 2 * int(sum(w)) + 3):
        c = Fr(k, 2)
        z, x = dantzig(p, w, c)
        ctx = (name, str(c))
        assert path.value_function(float(c)) == pytest.approx(float(z), abs=1e-9), ctx
        for sol in (solution_from_path(inst, path, float(c)), solve_capacity(inst, float(c))):
            assert sol.value == pytest.approx(float(z), abs=1e-9), ctx
            # the greedy solution is optimal too, so it must have the same value
            assert float(np.dot([float(xi) for xi in x], np.asarray(p, float))) == \
                pytest.approx(sol.value, abs=1e-9), ctx


def test_split_item_is_the_dantzig_critical_item():
    """When all ratios are distinct the split macroitem is a single item: the
    classical critical item of the greedy rule."""
    rng = random.Random(SEED + 1)
    n_checked = 0
    for _ in range(40):
        n = rng.randint(2, 9)
        # pairwise distinct ratios: distinct primes over weight 1
        p = rng.sample([2, 3, 5, 7, 11, 13, 17, 19, 23], n)
        w = [1] * n
        inst = make_instance(p, w, [], "distinct ratios")
        path = canonical_path(inst)
        assert all(I.size == 1 for I in path.macroitems)
        for k in range(1, 2 * n):
            c = Fr(k, 2)
            if not 0 < float(c) < float(path.W[path.q]):
                continue
            sol = solution_from_path(inst, path, float(c))
            crit = int(np.flatnonzero(sol.H)[0])
            z, x = dantzig(p, w, c)
            assert sol.H.sum() == 1
            assert Fr(sol.theta).limit_denominator(10 ** 6) == x[crit]
            assert sol.value == pytest.approx(float(z), abs=1e-9)
            n_checked += 1
    assert n_checked > 100


def test_removing_the_arcs_of_a_dag_gives_the_dantzig_bound():
    """A sanity check on larger data: dropping the precedence arcs of a random
    DAG must turn the canonical value function into the Dantzig bound, which is
    then an upper bound on the constrained one."""
    for seed in range(8):
        dag = random_dag(40, 2.0, seed=seed)
        free = make_instance(dag.p.astype(int).tolist(), dag.w.astype(int).tolist(), [],
                             "relaxed")
        p, w = [int(v) for v in dag.p], [int(v) for v in dag.w]
        path_free = canonical_path(free)
        path_dag = canonical_path(dag)
        for c in np.linspace(0.0, float(dag.w.sum()), 11):
            z, _ = dantzig(p, w, Fr(c).limit_denominator(10 ** 9))
            assert path_free.value_function(float(c)) == pytest.approx(float(z), rel=1e-9)
            assert path_dag.value_function(float(c)) <= float(z) + 1e-7 * max(1.0, abs(float(z)))
