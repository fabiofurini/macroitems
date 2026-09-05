"""Cross-checks of the library against brute force (enumeration of all closures)
and against the LP solver on small random instances.

Run:  python3 tests/test_random.py   (or pytest)
"""
import itertools
import random
import sys, os
from fractions import Fraction as Fr

import numpy as np
from scipy.optimize import linprog

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from macroitems import (Instance, canonical_path, solve_capacity, solution_from_path, canonical_dual,
                        face_dimensions, solve_lp, running_example, canonical_reduced_costs)


def closures_of(n, A):
    res = []
    for r in range(n + 1):
        for c in itertools.combinations(range(n), r):
            S = set(c)
            if all(j in S for (i, j) in A if i in S):
                res.append(frozenset(S))
    return res


def brute_canonical(n, p, w, A):
    closures = closures_of(n, A)
    P = lambda S: sum(p[i] for i in S)
    W = lambda S: sum(w[i] for i in S)
    prefix = frozenset(); seq = []
    while prefix != frozenset(range(n)):
        cands = [(Fr(P(c - prefix), W(c - prefix)), W(c - prefix), c - prefix) for c in closures if prefix < c]
        lam = max(x[0] for x in cands)
        tied = [x for x in cands if x[0] == lam]
        S = max(tied, key=lambda x: x[1])[2]
        seq.append((S, lam)); prefix |= S
    return seq, closures


def dual_face_dimension_lp(n, p, w, A, c, z):
    m = len(A); nv = 1 + n + m
    rows = []
    for i in range(n):
        row = np.zeros(nv); row[0] = w[i]; row[1 + i] = 1
        for k, (a, b) in enumerate(A):
            if a == i: row[1 + n + k] += 1
            if b == i: row[1 + n + k] -= 1
        rows.append(row)
    G = np.array(rows)
    obj = np.zeros(nv); obj[0] = c; obj[1:1 + n] = 1
    M = np.vstack([-G, -np.eye(nv)]); q = np.concatenate([-np.array(p, float), np.zeros(nv)])
    tight = []
    for r in range(M.shape[0]):
        res = linprog(M[r], A_ub=M, b_ub=q, A_eq=obj.reshape(1, -1), b_eq=[z], bounds=[(None, None)] * nv, method="highs")
        if q[r] - res.fun < 1e-7: tight.append(M[r])
    E = np.vstack([obj] + tight) if tight else obj.reshape(1, -1)
    return nv - np.linalg.matrix_rank(E, tol=1e-8)


def primal_face_dimension_lp(n, p, w, A, c, z):
    A_ub = [list(map(float, w))]; b_ub = [float(c)]
    for (a, bb) in A:
        row = [0.0] * n; row[a] = 1; row[bb] = -1; A_ub.append(row); b_ub.append(0.0)
    M = np.vstack([np.array(A_ub), np.eye(n), -np.eye(n)]); q = np.concatenate([np.array(b_ub), np.ones(n), np.zeros(n)])
    obj = np.array(p, float)
    tight = []
    for r in range(M.shape[0]):
        res = linprog(M[r], A_ub=M, b_ub=q, A_eq=obj.reshape(1, -1), b_eq=[z], bounds=[(None, None)] * n, method="highs")
        if q[r] - res.fun < 1e-7: tight.append(M[r])
    E = np.vstack([obj] + tight) if tight else obj.reshape(1, -1)
    return n - np.linalg.matrix_rank(E, tol=1e-8)


def random_instance(rng):
    n = rng.randint(3, 8)
    order = list(range(n)); rng.shuffle(order)
    A = []
    for a in range(n):
        for bb in range(a + 1, n):
            if rng.random() < 0.35:
                A.append((order[a], order[bb]))
    p = [rng.choice([-3, -1, 1, 2, 2, 3, 3, 4, 6, 9]) for _ in range(n)]
    w = [rng.choice([1, 1, 2, 2, 3, 4]) for _ in range(n)]
    return n, p, w, A


def test_running_example():
    inst = running_example()
    path = canonical_path(inst)
    assert [sorted((I + 1).tolist()) for I in path.macroitems] == [[3, 6], [1, 2, 5], [4, 7, 8]]
    assert np.allclose(path.ratios, [2, 1.5, 1])
    sol = solve_capacity(inst, 4.0)
    assert abs(sol.value - 7) < 1e-9 and abs(sol.lam - 1.5) < 1e-9
    fi = face_dimensions(inst, sol)
    assert fi.dim_dual == 3 and fi.dim_primal == 0
    d = canonical_dual(inst, sol, 4.0)
    assert d.feasible and abs(d.value - 7) < 1e-9
    rc = canonical_reduced_costs(inst, path, 2)
    assert np.allclose(rc, [0, 0, .5, 1, 0, .5, .5, .5])


def test_random(n_trials=300, seed=11, verbose=True):
    rng = random.Random(seed)
    stats = dict(trials=0, dims=0, nontrivial_primal=0, ties=0)
    for _ in range(n_trials):
        n, p, w, A = random_instance(rng)
        inst = Instance(np.array(p, float), np.array(w, float), np.array(A, dtype=np.int64).reshape(-1, 2))
        seq, closures = brute_canonical(n, p, w, A)
        for method in ("bisection", "dinkelbach"):
            path = canonical_path(inst, method=method)
            got = [frozenset(I.tolist()) for I in path.macroitems]
            assert got == [S for S, _ in seq], (method, got, seq, p, w, A)
            assert path.check(inst)["strictly_decreasing"]
        q = max([r for r, (S, lam) in enumerate(seq, 1) if lam > 0], default=0)
        stats["trials"] += 1
        # LP values on a grid of capacities (including degenerate ones)
        Wc = np.cumsum([0] + [sum(w[i] for i in S) for S, _ in seq])
        for c in list(np.linspace(0.3, Wc[-1] + 1, 7)) + [float(x) for x in Wc[1:]]:
            s1 = solution_from_path(inst, path, c)
            s2 = solve_capacity(inst, c)
            lp = solve_lp(inst, c)
            assert abs(s1.value - lp.value) < 1e-7 and abs(s2.value - lp.value) < 1e-7, (c, s1.value, s2.value, lp.value)
            # primal feasibility of the canonical solution
            x = s2.x
            assert np.all(x >= -1e-12) and np.all(x <= 1 + 1e-12) and float(inst.w @ x) <= c + 1e-9
            assert np.all(x[inst.arcs[:, 0]] <= x[inst.arcs[:, 1]] + 1e-12)
            if not s2.degenerate:
                d = canonical_dual(inst, s2, c)
                assert d.feasible and abs(d.value - lp.value) < 1e-7, (c, d.value, lp.value, d.max_violation)
        # face dimensions at a random nondegenerate capacity
        if q > 0:
            h = rng.randint(1, q)
            c = (Wc[h - 1] + Wc[h]) / 2 + rng.uniform(-0.2, 0.2) * (Wc[h] - Wc[h - 1])
            sol = solve_capacity(inst, c)
            fi = face_dimensions(inst, sol)
            z = sol.value
            dd = dual_face_dimension_lp(n, p, w, A, c, z)
            dp = primal_face_dimension_lp(n, p, w, A, c, z)
            assert fi.dim_dual == dd, ("dual dim", fi, dd, p, w, A, c)
            assert fi.dim_primal == dp, ("primal dim", fi, dp, p, w, A, c)
            stats["dims"] += 1
            stats["nontrivial_primal"] += int(dp > 0)
            stats["ties"] += int(fi.n_tight_proper > 0)
    if verbose:
        print("random cross-checks passed:", stats)
    return stats


if __name__ == "__main__":
    test_running_example()
    print("running example: OK")
    test_random()
