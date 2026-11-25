"""Optimal faces, reduced costs, branching and forests (Sections 4-8 of the
optimal-faces paper), each checked against an independent computation:

* inseparability classes, order and maximal chains against an exact
  enumeration of all tight subsets (rational arithmetic);
* the strongest reduced costs against an LP that maximizes the dual slack over
  the dual optimal face;
* the branching characterization against the LP relaxation with the variables
  fixed, for every single item and every pair;
* the forest algorithms against the general minimum-cut ones.
"""
from __future__ import annotations

import itertools
import random
from fractions import Fraction as Fr

import numpy as np
import pytest
from scipy.optimize import linprog

from conftest import make_instance, random_small
from bruteforce import all_closures
from macroitems import canonical_path, face_dimensions, running_example, solve_capacity
from macroitems.classes import split_structure
from macroitems.dual import best_reduced_costs, canonical_reduced_costs
from macroitems.forests import (forest_reduced_costs, forest_split_structure, hook_length_count,
                                is_forest)

SEED = 20261125
TOL = 1e-7


# ------------------------------------------------------------ instances
def random_forest(rng: random.Random, n_max: int, orient: str):
    """Small forest with many ties: weights in {1,2,3}, profits in {-2..4}."""
    n = rng.randint(2, n_max)
    arcs = []
    for i in range(1, n):
        if rng.random() < 0.1:
            continue
        j = rng.randrange(i)
        if orient == "in":
            arcs.append((i, j))
        elif orient == "out":
            arcs.append((j, i))
        else:
            arcs.append((i, j) if rng.random() < 0.5 else (j, i))
    perm = list(range(n))
    rng.shuffle(perm)
    arcs = [(perm[a], perm[b]) for a, b in arcs]
    p = [rng.randint(-2, 4) for _ in range(n)]
    w = [rng.randint(1, 3) for _ in range(n)]
    return n, p, w, arcs


def nondegenerate_cases(inst):
    """(c, sol) at the midpoint of every nondegenerate capacity interval."""
    path = canonical_path(inst)
    for h in range(1, path.q + 1):
        lo, hi = float(path.W[h - 1]), float(path.W[h])
        c = (lo + hi) / 2.0
        if not lo < c < hi:
            continue
        sol = solve_capacity(inst, c)
        if sol.degenerate or not sol.H.any():
            continue
        yield c, sol


def small_instances(n_inst=300, n_max=8):
    rng = random.Random(SEED)
    for t in range(n_inst):
        if t % 2:
            n, p, w, arcs = random_small(rng, n_max)
        else:
            n, p, w, arcs = random_forest(rng, n_max, rng.choice(["in", "out", "mixed"]))
        yield t, n, p, w, arcs


# ------------------------------------------------------------ brute force
def brute_tight_family(n, p, w, arcs, sol):
    """All tight subsets of H, exactly: relatively closed in H with b(L) = 0."""
    H = [int(i) for i in np.flatnonzero(sol.H)]
    lam = Fr(sum(p[i] for i in H), sum(w[i] for i in H))
    b = {i: Fr(p[i]) - lam * w[i] for i in H}
    AH = [(i, j) for i, j in arcs if i in H and j in H]
    tight = []
    for r in range(len(H) + 1):
        for L in itertools.combinations(H, r):
            S = set(L)
            if any(i in S and j not in S for i, j in AH):
                continue
            s = sum((b[i] for i in L), Fr(0))
            assert s <= 0, "Lemma lem:tight: a relatively closed subset with b > 0"
            if s == 0:
                tight.append(frozenset(L))
    return H, tight


def brute_classes(H, tight):
    sig = {}
    for i in H:
        sig.setdefault(tuple(i in L for L in tight), set()).add(i)
    classes = [frozenset(K) for K in sig.values()]
    down = {K: frozenset.intersection(*[L for L in tight if K <= L]) for K in classes}
    return classes, down


def brute_maximal_chains(H, tight):
    fam = set(tight)
    full = frozenset(H)
    memo = {}

    def count(L):
        if L == full:
            return 1
        if L not in memo:
            ups = [M for M in fam if L < M]
            covers = [M for M in ups if not any(L < X < M for X in ups)]
            memo[L] = sum(count(M) for M in covers)
        return memo[L]

    return count(frozenset())


def lp_value(n, p, w, arcs, c, lb=None, ub=None):
    lb = np.zeros(n) if lb is None else lb
    ub = np.ones(n) if ub is None else ub
    if np.any(lb > ub) or float(np.dot(w, lb)) > c + 1e-9:
        return -np.inf
    A = [list(map(float, w))]
    rhs = [c]
    for i, j in arcs:
        row = [0.0] * n
        row[i], row[j] = 1.0, -1.0
        A.append(row)
        rhs.append(0.0)
    r = linprog(-np.asarray(p, float), A_ub=np.array(A), b_ub=np.array(rhs),
                bounds=list(zip(lb, ub)), method="highs")
    if r.status == 2:
        return -np.inf
    assert r.status == 0, r.message
    return -r.fun


def branch_bound(n, p, w, arcs, c, fix):
    """max over the branches of the LP with x_t = fix[t]; fixing propagates."""
    succ = {i: [] for i in range(n)}
    pred = {i: [] for i in range(n)}
    for i, j in arcs:
        succ[i].append(j)
        pred[j].append(i)

    def reach(t, nbr):
        seen, stack = {t}, [t]
        while stack:
            v = stack.pop()
            for u in nbr[v]:
                if u not in seen:
                    seen.add(u)
                    stack.append(u)
        return seen

    lb, ub = np.zeros(n), np.ones(n)
    for t, a in fix.items():
        if a == 1:
            lb[list(reach(t, succ))] = 1.0
        else:
            ub[list(reach(t, pred))] = 0.0
    return lp_value(n, p, w, arcs, c, lb, ub)


def dual_face_max(n, p, w, arcs, c, z, i, kind):
    """max s_i (kind 'Z') or max mu_i (kind 'F') over the optimal dual face, by LP."""
    m = len(arcs)
    nv = 1 + n + m                          # lambda, mu, alpha
    G = np.zeros((n, nv))                   # rows: w_i lam + mu_i + div_i(alpha)
    for k in range(n):
        G[k, 0] = w[k]
        G[k, 1 + k] = 1.0
    for e, (a, b) in enumerate(arcs):
        G[a, 1 + n + e] += 1.0
        G[b, 1 + n + e] -= 1.0
    obj_row = np.zeros(nv)
    obj_row[0] = c
    obj_row[1:1 + n] = 1.0
    goal = G[i].copy() if kind == "Z" else np.eye(nv)[1 + i]
    r = linprog(-goal, A_ub=-G, b_ub=-np.asarray(p, float), A_eq=obj_row.reshape(1, -1), b_eq=[z],
                bounds=[(0, None)] * nv, method="highs")
    assert r.status == 0, r.message
    return -r.fun - (p[i] if kind == "Z" else 0.0)


# ------------------------------------------------------------ tests
def test_classes_order_and_chains_against_enumeration():
    checked = multi = 0
    for t, n, p, w, arcs in small_instances():
        inst = make_instance(p, w, arcs, f"cls#{t}")
        for c, sol in nondegenerate_cases(inst):
            S = split_structure(inst, sol, c)
            H, tight = brute_tight_family(n, p, w, arcs, sol)
            classes, down = brute_classes(H, tight)
            ctx = (t, p, w, arcs, c)
            got = [frozenset(int(i) for i in K) for K in S.classes]
            assert set(got) == set(classes), ctx
            assert S.k0 == face_dimensions(inst, sol).k0, ctx
            for a, K in enumerate(got):
                assert frozenset(int(i) for i in S.down(a)) == down[K], ctx
                # up K = union of the classes whose down-set contains K
                up = frozenset().union(*[K2 for K2 in classes if K <= down[K2]])
                assert frozenset(int(i) for i in S.up(a)) == up, ctx
            # Lemma lem:chains: tight subsets are exactly the unions of down-sets
            downsets = set()
            for r in range(S.k0 + 1):
                for sub in itertools.combinations(range(S.k0), r):
                    U = frozenset().union(*[got[a] for a in sub])
                    if all(frozenset(int(i) for i in S.down(a)) <= U for a in sub):
                        downsets.add(U)
            assert downsets == set(tight), ctx
            # Prop prop:chains: maximal chains = linear extensions of the order
            assert S.count_maximal_chains() == brute_maximal_chains(H, tight), ctx
            checked += 1
            multi += S.k0 > 1
    assert checked > 200 and multi > 20


def test_branching_characterization_against_branch_lps():
    """Prop prop:branching and Cor cor:critical, for every item and every pair."""
    checked = pair_only = 0
    # the two-variable example of Section sec:branching: no precedences, weights
    # 4, 4, 2 with a common ratio, c' = 7
    extra = [(-1, 3, [4, 4, 2], [4, 4, 2], [])]
    for t, n, p, w, arcs in itertools.chain(extra, small_instances(400, 7)):
        inst = make_instance(p, w, arcs, f"br#{t}")
        cases = [(7.0, solve_capacity(inst, 7.0))] if t == -1 else nondegenerate_cases(inst)
        for c, sol in cases:
            S = split_structure(inst, sol, c)
            z = sol.value
            ctx = (t, p, w, arcs, c)
            crit = [a for a in range(S.k0) if S.is_critical(a)]
            assert len(crit) <= 1, ctx
            if S.k0 == 1:
                assert crit == [0], ctx            # unique primal optimum: H is critical
            single = {}
            for i in range(n):
                U = max(branch_bound(n, p, w, arcs, c, {i: a}) for a in (0, 1))
                single[i] = U < z - TOL
                assert single[i] == S.bound_decreases([i]), ctx
                in_crit = bool(sol.H[i]) and S.critical is not None and S.class_of[i] == S.critical
                assert single[i] == in_crit, ctx
            for i, j in itertools.combinations(range(n), 2):
                U = max(branch_bound(n, p, w, arcs, c, {i: a, j: b}) for a in (0, 1) for b in (0, 1))
                actual = U < z - TOL
                assert actual == S.bound_decreases([i, j]), ctx
                pair_only += actual and not single[i] and not single[j]
            checked += 1
    assert checked > 150
    assert pair_only > 0, "no instance where two levels help and one does not"


def test_strongest_reduced_costs_are_maxima_over_the_dual_face():
    """Prop prop:reduced: s*_i and mu*_i are the largest slacks over D*(c)."""
    checked = 0
    for t, n, p, w, arcs in small_instances(150, 7):
        inst = make_instance(p, w, arcs, f"rc#{t}")
        path = canonical_path(inst)
        for c, sol in nondegenerate_cases(inst):
            rc = best_reduced_costs(inst, sol).value
            canon = canonical_reduced_costs(inst, path, path.split_index(c))
            for i in range(n):
                if sol.H[i]:
                    continue
                kind = "Z" if sol.Z[i] else "F"
                ref = dual_face_max(n, p, w, arcs, c, sol.value, i, kind)
                assert rc[i] == pytest.approx(ref, abs=1e-6), (t, p, w, arcs, c, i)
                assert rc[i] > 0 and rc[i] >= canon[i] - 1e-9
                checked += 1
    assert checked > 300


def test_forest_algorithms_agree_with_minimum_cuts():
    """Cor cor:forestrc and Prop prop:forestclasses(a)-(d) against the general code."""
    rng = random.Random(SEED + 1)
    checked = 0
    for t in range(300):
        n, p, w, arcs = random_forest(rng, 30 if t % 3 else 9, rng.choice(["in", "out", "mixed"]))
        inst = make_instance(p, w, arcs, f"forest#{t}")
        assert is_forest(inst)
        for c, sol in nondegenerate_cases(inst):
            ctx = (t, p, w, arcs, c)
            assert np.allclose(forest_reduced_costs(inst, sol), best_reduced_costs(inst, sol).value,
                               atol=1e-9, equal_nan=True), ctx
            S, Sf = split_structure(inst, sol, c), forest_split_structure(inst, sol, c)
            assert {tuple(K) for K in S.classes} == {tuple(K) for K in Sf.classes}, ctx
            m = {tuple(K): a for a, K in enumerate(S.classes)}
            perm = [m[tuple(K)] for K in Sf.classes]
            assert np.array_equal(S.below[np.ix_(perm, perm)], Sf.below), ctx
            assert S.critical == (None if Sf.critical is None else perm[Sf.critical]), ctx
            # (d): E_0 = separable arcs, and the dual face lives on F and Z only
            fi = face_dimensions(inst, sol)
            assert fi.components_H0 == fi.k0 and fi.dim_dual == fi.arcs_F + fi.arcs_Z, ctx
            checked += 1
    assert checked > 300


def test_hook_length_formula_on_forests():
    """Prop prop:forestclasses(e): the number of maximal chains on in- and out-forests."""
    rng = random.Random(SEED + 2)
    checked = {"in": 0, "out": 0}
    for t in range(400):
        orient = "in" if t % 2 else "out"
        n, p, w, arcs = random_forest(rng, 12, orient)
        inst = make_instance(p, w, arcs, f"hook#{t}")
        for c, sol in nondegenerate_cases(inst):
            S = forest_split_structure(inst, sol, c)
            assert hook_length_count(S) == S.count_maximal_chains(), (t, orient, p, w, arcs, c)
            checked[orient] += S.k0 > 2
    assert checked["in"] > 5 and checked["out"] > 5


def test_hook_length_up_sets_hold_only_on_in_forests():
    """k0!/prod |up K| counts the maximal chains on in-forests, but not on
    out-forests, where the hook of K is its down-set."""
    rng = random.Random(SEED + 3)
    wrong = {"in": 0, "out": 0}
    for t in range(3000):
        orient = "in" if t % 2 else "out"
        n, p, w, arcs = random_forest(rng, 14, orient)
        inst = make_instance(p, w, arcs, f"hookpaper#{t}")
        for c, sol in nondegenerate_cases(inst):
            S = forest_split_structure(inst, sol, c)
            from math import factorial
            paper = Fr(factorial(S.k0), int(np.prod(S.below.sum(axis=0).astype(object))))
            wrong[orient] += paper != S.count_maximal_chains()
    assert wrong["in"] == 0
    assert wrong["out"] > 0      # on out-forests the hook is |down K|, not |up K|


def test_running_example_reduced_costs_and_branching():
    """Table tab:reduced and the branching example at c = 4."""
    inst = running_example()
    sol = solve_capacity(inst, 4.0)
    rc = best_reduced_costs(inst, sol).value
    path = canonical_path(inst)
    canon = canonical_reduced_costs(inst, path, path.split_index(4.0))
    items = [3, 6, 4, 7, 8]                              # paper numbering
    assert [canon[i - 1] for i in items] == [0.5, 0.5, 1.0, 0.5, 0.5]
    assert [rc[i - 1] for i in items] == pytest.approx([1, 1, 2, 2, 2])
    assert np.array_equal(forest_reduced_costs(inst, sol), rc, equal_nan=True) if is_forest(inst) else True
    S = split_structure(inst, sol, 4.0)
    assert S.k0 == 1 and S.critical == 0
    assert [S.bound_decreases([i - 1]) for i in range(1, 9)] == [True, True, False, False, True,
                                                                 False, False, False]
    n, p, w = inst.n, inst.p.tolist(), inst.w.tolist()
    arcs = [tuple(a) for a in inst.arcs.tolist()]
    U5 = max(branch_bound(n, p, w, arcs, 4.0, {4: a}) for a in (0, 1))
    assert U5 == pytest.approx(16 / 3)
