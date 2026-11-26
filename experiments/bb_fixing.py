"""Does reduced-cost fixing speed up the exact solution in practice?

For each instance of the PCKP benchmark, at its own capacity, the integer
problem is solved by Gurobi (one thread) in three variants:

* ``base``: the full model;
* ``solver``: the variables fixed by the reduced costs of the dual optimum
  returned by HiGHS (items of ``Z`` with ``s_i > z_LP - LB`` set to 0, items of
  ``F`` with ``mu_i > z_LP - LB`` set to 1);
* ``best``: the variables fixed by the strongest reduced costs over the dual
  optimal face (one minimum cut per item). Their computing time is reported
  separately and added to the total.

``LB`` is the value of the macroitem heuristic, available before any search.
It is passed to all three variants as a cutoff, so every variant starts from
the same information. Fixing is valid for every solution better than ``LB``.
If no better solution exists, all variants report "no solution above the
cutoff", and ``LB`` is optimal.

Each variant runs with several Gurobi seeds, and the script checks that all
variants reach the same optimal value.  Each instance runs in its own
memory-capped process (see ``literature_fixing.py``).

Usage::

    python experiments/bb_fixing.py --out experiments/results/bb_fixing.csv
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from macroitems import canonical_path, solution_from_path  # noqa: E402
from macroitems.fastrc import fast_reduced_costs  # noqa: E402
from macroitems.formats import read_pckp_dat  # noqa: E402
from macroitems.stats import heuristic_integer  # noqa: E402
from forest_fixing import environment, solver_reduced_costs  # noqa: E402
from literature_fixing import (MEM_LIMIT_GB, MIN_AVAILABLE_GB, WORKER_AS_GB,  # noqa: E402
                               available_gb)

TOL = 1e-7
VARIANTS = ("base", "solver", "best")
FIELDS = ["name", "family", "n", "m", "variant", "seed", "fixed0", "fixed1", "lb", "z_lp",
          "status", "obj", "bound", "nodes", "grb_seconds", "rc_seconds", "total_seconds"]


def solve(inst, c, lb, fix0, fix1, seed, time_limit, mem_limit_gb=MEM_LIMIT_GB):
    import gurobipy as gp

    env = gp.Env(empty=True)
    env.setParam("OutputFlag", 0)
    env.setParam("LogFile", "")
    env.start()
    m = gp.Model(env=env)
    m.Params.Threads = 1
    m.Params.Seed = seed
    m.Params.TimeLimit = time_limit
    m.Params.MIPGap = 0.0
    m.Params.MemLimit = mem_limit_gb
    m.Params.NodefileStart = 0.5
    m.Params.Cutoff = lb + 0.5                 # integer data: only strictly better solutions
    lbv = np.zeros(inst.n)
    ubv = np.ones(inst.n)
    ubv[fix0] = 0.0
    lbv[fix1] = 1.0
    x = m.addMVar(inst.n, lb=lbv, ub=ubv, vtype=gp.GRB.BINARY)
    m.addConstr(inst.w @ x <= c)
    a = inst.arcs
    m.addConstr(x[a[:, 0]] <= x[a[:, 1]])
    m.setObjective(inst.p @ x, gp.GRB.MAXIMIZE)
    t0 = time.perf_counter()
    m.optimize()
    secs = time.perf_counter() - t0
    status = {gp.GRB.OPTIMAL: "optimal", gp.GRB.CUTOFF: "cutoff", gp.GRB.INFEASIBLE: "cutoff",
              gp.GRB.TIME_LIMIT: "time_limit", gp.GRB.MEM_LIMIT: "mem_limit"}.get(m.Status, str(m.Status))
    obj = float(round(m.ObjVal)) if m.SolCount > 0 else None
    bound = float(m.ObjBound) if m.Status not in (gp.GRB.CUTOFF, gp.GRB.INFEASIBLE) else None
    nodes = int(m.NodeCount)
    m.dispose()
    env.dispose()
    return status, obj, bound, nodes, secs


def run_instance(path_dat, seeds, time_limit, variants=VARIANTS, mem_limit_gb=MEM_LIMIT_GB):
    inst = read_pckp_dat(path_dat)
    c = float(inst.meta["capacity"])
    path = canonical_path(inst)
    sol = solution_from_path(inst, path, c)
    lb = heuristic_integer(inst, path, c)
    gap = sol.value - lb
    _, s_sol, mu_sol = solver_reduced_costs(inst, c)
    best, rc_stats = fast_reduced_costs(inst, sol, "tree", return_stats=True)
    rc_secs = sum(rc_stats[r][2] for r in ("Z", "F") if r in rc_stats)   # measured inside C++
    Z, F = sol.Z, sol.F
    fixes = {
        "base": (np.zeros(0, int), np.zeros(0, int)),
        "solver": (np.flatnonzero(Z & (s_sol > gap + TOL)), np.flatnonzero(F & (mu_sol > gap + TOL))),
        "best": (np.flatnonzero(Z & (np.nan_to_num(best, nan=-np.inf) > gap + TOL)),
                 np.flatnonzero(F & (np.nan_to_num(best, nan=-np.inf) > gap + TOL))),
    }
    rows = []
    for v in variants:
        f0, f1 = fixes[v]
        for seed in seeds:
            status, obj, bound, nodes, secs = solve(inst, c, lb, f0, f1, seed, time_limit, mem_limit_gb)
            extra = rc_secs if v == "best" else 0.0
            rows.append(dict(name=inst.name, n=inst.n, m=inst.m, variant=v, seed=seed,
                             fixed0=len(f0), fixed1=len(f1), lb=lb, z_lp=sol.value, status=status,
                             obj=obj, bound=bound, nodes=nodes, grb_seconds=round(secs, 3),
                             rc_seconds=round(extra, 3), total_seconds=round(secs + extra, 3)))
    # correctness: every variant must reach the same optimum (or all prove LB optimal)
    finals = {(r["status"], r["obj"]) for r in rows if r["status"] in ("optimal", "cutoff")}
    assert len({o for _, o in finals}) <= 1, ("variants disagree", inst.name, finals)
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--instances", default="instances_raw/pckp_benchmark")
    ap.add_argument("--out", default="experiments/results/bb_fixing.csv")
    ap.add_argument("--seeds", default="1,2,3")
    ap.add_argument("--time-limit", type=float, default=600.0)
    ap.add_argument("--only", default="")
    ap.add_argument("--variants", default=",".join(VARIANTS))
    ap.add_argument("--mem-limit-gb", type=float, default=MEM_LIMIT_GB, help="Gurobi MemLimit")
    ap.add_argument("--worker-cap-gb", type=float, default=WORKER_AS_GB, help="address-space cap of a worker")
    ap.add_argument("--worker", help=argparse.SUPPRESS)
    ap.add_argument("--worker-out", help=argparse.SUPPRESS)
    a = ap.parse_args(argv)
    seeds = [int(s) for s in a.seeds.split(",")]
    if a.worker:
        with open(a.worker_out, "w") as f:
            json.dump(run_instance(a.worker, seeds, a.time_limit, a.variants.split(","), a.mem_limit_gb), f)
        return
    import resource
    import subprocess
    import tempfile
    files = sorted(glob.glob(os.path.join(a.instances, "telecom", "*.lp.dat")) +
                   glob.glob(os.path.join(a.instances, "mining", "*.lp.dat")))
    if a.only:
        keep = set(a.only.split(","))
        files = [f for f in files if os.path.basename(f).replace(".lp.dat", "") in keep]
    tmp = tempfile.mkdtemp(prefix="bbfix-")
    cap = int(a.worker_cap_gb * 2**30)

    def limit():
        resource.setrlimit(resource.RLIMIT_AS, (cap, cap))
        os.nice(10)

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w", newline="") as f:
        f.write(f"# bb_fixing.py seeds={a.seeds} variants={a.variants} gurobi_mem_limit_gb={a.mem_limit_gb} worker_cap_gb={a.worker_cap_gb} time_limit={a.time_limit} lb=macroitem heuristic, "
                f"passed as cutoff\n# {environment()}\n")
        wr = csv.DictWriter(f, fieldnames=FIELDS)
        wr.writeheader()
        for path_dat in files:
            while available_gb() < MIN_AVAILABLE_GB:
                time.sleep(30)
            out = os.path.join(tmp, os.path.basename(path_dat) + ".json")
            budget = 3 * len(seeds) * (a.time_limit + 60) + 600
            try:
                r = subprocess.run([sys.executable, os.path.abspath(__file__), "--worker", path_dat,
                                    "--worker-out", out, "--seeds", a.seeds, "--time-limit", str(a.time_limit),
                                    "--variants", a.variants, "--mem-limit-gb", str(a.mem_limit_gb)],
                                   preexec_fn=limit, capture_output=True, text=True, timeout=budget)
            except subprocess.TimeoutExpired:
                print(f"{path_dat}: worker timeout", file=sys.stderr, flush=True)
                continue
            fam = os.path.basename(os.path.dirname(path_dat))
            if r.returncode != 0 or not os.path.exists(out):
                print(f"{path_dat}: failed {(r.stderr or '').strip().splitlines()[-1:]}", file=sys.stderr, flush=True)
                continue
            for row in json.load(open(out)):
                row["family"] = fam
                wr.writerow({k: row.get(k, "") for k in FIELDS})
            f.flush()
            print(f"{os.path.basename(path_dat):22s} done", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
