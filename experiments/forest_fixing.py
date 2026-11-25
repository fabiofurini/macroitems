"""Variable fixing and branching degeneracy on random precedence forests.

This is the numerical illustration of the optimal-faces paper (Section
"Precedence forests", table ``tab:forests``).  For each of nine classes
(three orientations x three data distributions) it generates instances until
``--reps`` of them have a nondegenerate capacity and a certified integer
optimum.  It then compares two sets of reduced costs:

* those of the dual optimum returned by an LP solver (HiGHS through SciPy);
* the strongest ones over the dual optimal face, computed by the linear-time
  forest recursion.

Both are measured by the share of the items of ``Z`` and of ``F`` that they fix
against a heuristic lower bound.  It also records how often the split macroitem
has more than one inseparability class (``k0 > 1``) and has no critical class.

Every attempted instance is written to the CSV, including those discarded and
the reason, with the seed that regenerates it.  The table is produced from the
CSV by ``experiments/forest_fixing_table.py``.

Usage::

    python experiments/forest_fixing.py --out experiments/results/forest_fixing.csv
    python experiments/forest_fixing.py --classes in:ties --reps 5      # one class

Instance generator, as described in the paper:
* ``n`` items; each item other than the first is attached to a uniformly random
  earlier item with probability 0.92, and starts a new tree otherwise;
* the attachment arc is oriented as follows:
  * ``in``: the item requires the earlier one, so every item has at most one
    prerequisite;
  * ``out``: the earlier item requires the new one;
  * ``mixed``: either way with probability 1/2;
* the labels are then permuted at random;
* the data distributions are:
  * ``unc``: weights and profits uniform in {1..100};
  * ``signs``: weights in {1..100}, profits in {-60..100};
  * ``ties``: weights in {1,2,3}, profits in {-2..4};
* the capacity is ``floor(0.35 w(I))``.
"""
from __future__ import annotations

import argparse
import csv
import os
import platform
import sys
import time

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import csr_matrix, vstack

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from macroitems import Instance, canonical_path, solution_from_path  # noqa: E402
from macroitems.forests import forest_reduced_costs, forest_split_structure, is_forest  # noqa: E402

ORIENTATIONS = ("in", "out", "mixed")
PROFITS = ("unc", "signs", "ties")
TOL = 1e-7

FIELDS = ["orient", "profits", "attempt", "seed", "status", "n", "m", "capacity", "k", "h",
          "F", "H", "Z", "z_lp", "z_ip", "ip_status", "ip_seconds", "lb", "gap_lp_pct", "gap_heur_pct",
          "fix_Z_solver", "fix_Z_best", "fix_F_solver", "fix_F_best", "k0", "critical", "seconds"]


# ------------------------------------------------------------ instances
def generate(n: int, orient: str, profits: str, seed: int, p_root: float = 0.08) -> Instance:
    rng = np.random.default_rng(seed)
    arcs = []
    for i in range(1, n):
        if rng.random() < p_root:
            continue
        j = int(rng.integers(0, i))
        if orient == "in":
            arcs.append((i, j))
        elif orient == "out":
            arcs.append((j, i))
        else:
            arcs.append((i, j) if rng.random() < 0.5 else (j, i))
    perm = rng.permutation(n)
    arcs = [(int(perm[a]), int(perm[b])) for a, b in arcs]
    if profits == "ties":
        w = rng.integers(1, 4, size=n)
        p = rng.integers(-2, 5, size=n)
    else:
        w = rng.integers(1, 101, size=n)
        p = rng.integers(-60, 101, size=n) if profits == "signs" else rng.integers(1, 101, size=n)
    inst = Instance(p.astype(float), w.astype(float), np.array(arcs, dtype=np.int64).reshape(-1, 2),
                    name=f"forest_{orient}_{profits}_{seed}")
    inst.validate()
    return inst


# ------------------------------------------------------------ bounds
def integer_optimum(inst: Instance, c: float, time_limit: float):
    """Certified integer optimum by Gurobi (MIPGap = 0).  Returns (value or None, status, seconds)."""
    import gurobipy as gp

    env = gp.Env(empty=True)
    env.setParam("OutputFlag", 0)
    env.setParam("LogFile", "")
    env.start()
    m = gp.Model(env=env)
    m.Params.MIPGap = 0.0
    m.Params.MIPGapAbs = 0.0
    m.Params.Threads = 1
    m.Params.TimeLimit = time_limit
    x = m.addMVar(inst.n, vtype=gp.GRB.BINARY)
    m.addConstr(inst.w @ x <= c)
    for i, j in inst.arcs.tolist():
        m.addConstr(x[i] <= x[j])
    m.setObjective(inst.p @ x, gp.GRB.MAXIMIZE)
    t0 = time.perf_counter()
    m.optimize()
    secs = time.perf_counter() - t0
    status = {gp.GRB.OPTIMAL: "optimal", gp.GRB.TIME_LIMIT: "time_limit"}.get(m.Status, str(m.Status))
    val = float(round(m.ObjVal)) if m.Status == gp.GRB.OPTIMAL else None
    m.dispose()
    env.dispose()
    return val, status, secs


def greedy(inst: Instance, c: float, start: np.ndarray) -> float:
    """Repeatedly add the closure cl(i) minus S of positive profit and largest
    ratio that fits, starting from the closure ``start``."""
    succ = [[] for _ in range(inst.n)]
    for i, j in inst.arcs.tolist():
        succ[i].append(j)
    S = set(np.flatnonzero(start).tolist())
    W = float(inst.w[list(S)].sum()) if S else 0.0
    while True:
        best = None
        for i in range(inst.n):
            if i in S:
                continue
            add, stack = {i}, [i]
            while stack:
                v = stack.pop()
                for u in succ[v]:
                    if u not in S and u not in add:
                        add.add(u)
                        stack.append(u)
            idx = list(add)
            wa, pa = float(inst.w[idx].sum()), float(inst.p[idx].sum())
            if pa <= 0 or W + wa > c:
                continue
            if best is None or pa / wa > best[0]:
                best = (pa / wa, add, wa)
        if best is None:
            break
        S |= best[1]
        W += best[2]
    return float(inst.p[list(S)].sum()) if S else 0.0


def solver_reduced_costs(inst: Instance, c: float):
    """Reduced costs of the dual optimum returned by HiGHS: s_i and mu_i."""
    P = csr_matrix((np.r_[np.ones(inst.m), -np.ones(inst.m)],
                    (np.r_[np.arange(inst.m), np.arange(inst.m)], np.r_[inst.arcs[:, 0], inst.arcs[:, 1]])),
                   shape=(inst.m, inst.n))
    A = vstack([csr_matrix(inst.w.reshape(1, -1)), P])
    r = linprog(-inst.p, A_ub=A, b_ub=np.r_[c, np.zeros(inst.m)], bounds=[(0, 1)] * inst.n, method="highs")
    assert r.status == 0, r.message
    y = -r.ineqlin.marginals
    lam, alpha = y[0], y[1:]
    mu = -r.upper.marginals
    div = np.zeros(inst.n)
    np.add.at(div, inst.arcs[:, 0], alpha)
    np.add.at(div, inst.arcs[:, 1], -alpha)
    s = inst.w * lam + mu + div - inst.p
    return -r.fun, s, mu


# ------------------------------------------------------------ one instance
def run_one(orient, profits, attempt, seed, n, time_limit):
    t0 = time.perf_counter()
    inst = generate(n, orient, profits, seed)
    assert is_forest(inst)
    c = float(np.floor(0.35 * inst.w.sum()))
    row = dict(orient=orient, profits=profits, attempt=attempt, seed=seed, n=inst.n, m=inst.m, capacity=c)
    path = canonical_path(inst)
    sol = solution_from_path(inst, path, c)
    row.update(k=path.k)
    if sol.degenerate or not sol.H.any():
        row.update(status=f"degenerate_{sol.degenerate or 'empty'}", seconds=time.perf_counter() - t0)
        return row
    F, H, Z = sol.F, sol.H, sol.Z
    row.update(h=sol.h, F=int(F.sum()), H=int(H.sum()), Z=int(Z.sum()), z_lp=sol.value)
    z_ip, ip_status, ip_secs = integer_optimum(inst, c, time_limit)
    row.update(ip_status=ip_status, ip_seconds=round(ip_secs, 3))
    if z_ip is None:
        row.update(status="ip_not_certified", seconds=time.perf_counter() - t0)
        return row
    lb = max(greedy(inst, c, np.zeros(inst.n, bool)), greedy(inst, c, F))
    z_hs, s_sol, mu_sol = solver_reduced_costs(inst, c)
    assert abs(z_hs - sol.value) <= 1e-6 * max(1.0, abs(sol.value)), (z_hs, sol.value)
    best = forest_reduced_costs(inst, sol)
    S = forest_split_structure(inst, sol, c)
    gap = sol.value - lb
    row.update(
        status="ok", z_ip=z_ip, lb=lb,
        gap_lp_pct=100 * (sol.value - z_ip) / abs(z_ip), gap_heur_pct=100 * (z_ip - lb) / abs(z_ip),
        fix_Z_solver=int(np.sum(s_sol[Z] > gap + TOL)), fix_Z_best=int(np.sum(best[Z] > gap + TOL)),
        fix_F_solver=int(np.sum(mu_sol[F] > gap + TOL)), fix_F_best=int(np.sum(best[F] > gap + TOL)),
        k0=S.k0, critical=int(S.critical is not None), seconds=round(time.perf_counter() - t0, 3))
    return row


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", default="experiments/results/forest_fixing.csv")
    ap.add_argument("--classes", default="all", help="comma list of orient:profits, or 'all'")
    ap.add_argument("--reps", type=int, default=30)
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--max-attempts", type=int, default=1000)
    ap.add_argument("--base-seed", type=int, default=20261125)
    ap.add_argument("--time-limit", type=float, default=600.0)
    a = ap.parse_args(argv)
    classes = ([(o, p) for o in ORIENTATIONS for p in PROFITS] if a.classes == "all"
               else [tuple(x.split(":")) for x in a.classes.split(",")])
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w", newline="") as f:
        f.write(f"# forest_fixing.py n={a.n} reps={a.reps} base_seed={a.base_seed} "
                f"time_limit={a.time_limit} python={platform.python_version()}\n")
        wr = csv.DictWriter(f, fieldnames=FIELDS)
        wr.writeheader()
        for ci, (orient, profits) in enumerate(classes):
            ok = 0
            for attempt in range(a.max_attempts):
                if ok >= a.reps:
                    break
                # one seed per (base, class, attempt): every instance is regenerable on its own
                seed = int(np.random.SeedSequence([a.base_seed, ORIENTATIONS.index(orient),
                                                   PROFITS.index(profits), attempt]).generate_state(1)[0])
                row = run_one(orient, profits, attempt, seed, a.n, a.time_limit)
                wr.writerow({k: row.get(k, "") for k in FIELDS})
                f.flush()
                ok += row["status"] == "ok"
            print(f"{orient:5s} {profits:5s} ok={ok} attempts={attempt + 1}", file=sys.stderr, flush=True)
            if ok < a.reps:
                print(f"WARNING: only {ok} of {a.reps} instances for {orient}:{profits}", file=sys.stderr)


if __name__ == "__main__":
    main()
