"""Variable fixing and optimal faces on the PCKP benchmark of the literature.

This is the companion of ``forest_fixing.py`` for the 23 benchmark instances,
which are general precedence DAGs: telecom A-K and mining L-W (Boland et al.
2012; Espinoza et al. 2015), each at its own capacity.  For each instance it
records:

* the regions ``F``, ``H``, ``Z``, the number ``k0`` of inseparability classes,
  whether a critical class exists, and the dimensions of both optimal faces;
* a lower bound ``LB``.  It is the best integer solution found by Gurobi within
  ``--time-limit`` seconds (one thread), checked for feasibility, or the
  macroitem heuristic if that is better.  The best known value of the
  literature is recorded next to it for reference;
* the same shares against the macroitem heuristic alone (``*_heur`` columns), a
  bound that is available at no cost before any search;
* the share of the items of ``Z`` and of ``F`` fixed against ``LB`` by three
  sets of reduced costs:
  * those of the dual optimum returned by HiGHS through SciPy (``solver``);
  * the canonical ones (``canonical``);
  * the strongest ones over the dual optimal face, computed with one minimum
    cut per item (``best``).

The instances are third-party data and are not in the repository.  Download them
once with::

    gh release download -R fabiofurini/macroitems --pattern instances.zip
    unzip instances.zip -d instances_raw

Usage::

    python experiments/literature_fixing.py --out experiments/results/literature_fixing.csv
"""
from __future__ import annotations

import argparse
import csv
import glob
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from macroitems import canonical_path, face_dimensions, solution_from_path  # noqa: E402
from macroitems.classes import split_structure  # noqa: E402
from macroitems.dual import best_reduced_costs, canonical_reduced_costs  # noqa: E402
from macroitems.formats import read_pckp_dat  # noqa: E402
from macroitems.stats import heuristic_integer  # noqa: E402
from forest_fixing import environment, solver_reduced_costs  # noqa: E402

TOL = 1e-7
MEM_LIMIT_GB = 1.5          # Gurobi
WORKER_AS_GB = 3.0          # address-space cap of each worker process
MIN_AVAILABLE_GB = 3.0      # do not start an instance below this available memory
FIELDS = ["name", "family", "n", "m", "capacity", "status", "k", "h", "F", "H", "Z", "theta",
          "z_lp", "lb", "lb_source", "grb_status", "grb_bound", "grb_seconds", "best_known",
          "gap_lb_pct", "k0", "critical", "dim_primal", "dim_dual",
          "fix_Z_solver", "fix_Z_canonical", "fix_Z_best", "fix_F_solver", "fix_F_canonical", "fix_F_best",
          "lb_heur", "fix_Z_solver_heur", "fix_Z_best_heur", "fix_F_best_heur",
          "rc_gain_median", "seconds_rc", "seconds_classes", "seconds"]


def gurobi_incumbent(inst, c, time_limit):
    """Best integer solution within the time limit, verified; also the final bound."""
    import gurobipy as gp

    env = gp.Env(empty=True)
    env.setParam("OutputFlag", 0)
    env.setParam("LogFile", "")
    env.start()
    m = gp.Model(env=env)
    m.Params.Threads = 1
    m.Params.TimeLimit = time_limit
    m.Params.MIPGap = 0.0
    m.Params.MemLimit = MEM_LIMIT_GB          # hard cap on Gurobi's memory
    m.Params.NodefileStart = 0.5              # branch-and-bound nodes go to disk beyond 0.5 GB
    x = m.addMVar(inst.n, vtype=gp.GRB.BINARY)
    m.addConstr(inst.w @ x <= c)
    a = inst.arcs
    m.addConstr(x[a[:, 0]] <= x[a[:, 1]])
    m.setObjective(inst.p @ x, gp.GRB.MAXIMIZE)
    t0 = time.perf_counter()
    m.optimize()
    secs = time.perf_counter() - t0
    status = {gp.GRB.OPTIMAL: "optimal", gp.GRB.TIME_LIMIT: "time_limit",
              gp.GRB.MEM_LIMIT: "mem_limit"}.get(m.Status, str(m.Status))
    val, bound = None, None
    if m.SolCount > 0:
        sol = np.rint(x.X).astype(bool)
        assert inst.w[sol].sum() <= c + 1e-6, "Gurobi solution violates the capacity"
        assert not np.any(sol[a[:, 0]] & ~sol[a[:, 1]]), "Gurobi solution violates a precedence"
        val = float(inst.p[sol].sum())
        bound = float(m.ObjBound)
    m.dispose()
    env.dispose()
    return val, bound, status, secs


def run_one(path_dat, best_known, time_limit):
    t0 = time.perf_counter()
    inst = read_pckp_dat(path_dat)
    c = float(inst.meta["capacity"])
    family = os.path.basename(os.path.dirname(path_dat))
    row = dict(name=inst.name, family=family, n=inst.n, m=inst.m, capacity=c, best_known=best_known)
    path = canonical_path(inst)
    sol = solution_from_path(inst, path, c)
    row.update(k=path.k)
    if sol.degenerate or not sol.H.any():
        row.update(status=f"degenerate_{sol.degenerate or 'empty'}", seconds=time.perf_counter() - t0)
        return row
    F, H, Z = sol.F, sol.H, sol.Z
    row.update(h=sol.h, F=int(F.sum()), H=int(H.sum()), Z=int(Z.sum()), theta=round(sol.theta, 6),
               z_lp=sol.value)
    # lower bound
    heur = heuristic_integer(inst, path, c)
    grb, bound, grb_status, grb_secs = gurobi_incumbent(inst, c, time_limit)
    lb, src = (grb, "gurobi") if grb is not None and grb >= heur else (heur, "heuristic")
    row.update(lb=lb, lb_source=src, grb_status=grb_status, grb_bound=bound, grb_seconds=round(grb_secs, 2),
               gap_lb_pct=100 * (sol.value - lb) / abs(sol.value))
    gap = sol.value - lb
    # reduced costs
    z_hs, s_sol, mu_sol = solver_reduced_costs(inst, c)
    assert abs(z_hs - sol.value) <= 1e-6 * max(1.0, abs(sol.value)), (z_hs, sol.value)
    canon = canonical_reduced_costs(inst, path, sol.h)
    t1 = time.perf_counter()
    best = best_reduced_costs(inst, sol).value
    t_rc = time.perf_counter() - t1
    fixed = lambda rc, region, g=gap: int(np.sum(rc[region] > g + TOL))
    gap_heur = sol.value - heur
    with np.errstate(divide="ignore", invalid="ignore"):
        gain = best[F | Z] / canon[F | Z]
    # faces and classes
    t2 = time.perf_counter()
    fi = face_dimensions(inst, sol)
    S = split_structure(inst, sol, c)
    t_cl = time.perf_counter() - t2
    assert S.k0 == fi.k0
    row.update(status="ok", k0=S.k0, critical=int(S.critical is not None),
               dim_primal=fi.dim_primal, dim_dual=fi.dim_dual,
               fix_Z_solver=fixed(s_sol, Z), fix_Z_canonical=fixed(canon, Z), fix_Z_best=fixed(best, Z),
               fix_F_solver=fixed(mu_sol, F), fix_F_canonical=fixed(canon, F), fix_F_best=fixed(best, F),
               lb_heur=heur, fix_Z_solver_heur=fixed(s_sol, Z, gap_heur),
               fix_Z_best_heur=fixed(best, Z, gap_heur), fix_F_best_heur=fixed(best, F, gap_heur),
               rc_gain_median=round(float(np.median(gain[np.isfinite(gain)])), 3),
               seconds_rc=round(t_rc, 2), seconds_classes=round(t_cl, 2),
               seconds=round(time.perf_counter() - t0, 2))
    return row


def available_gb() -> float:
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / 2**20
    return float("inf")


def worker(path_dat, best_known, time_limit, out_json):
    """Run one instance in this (memory-capped) process and write its row as JSON."""
    import json
    row = run_one(path_dat, best_known, time_limit)
    with open(out_json, "w") as f:
        json.dump(row, f)


def run_isolated(path_dat, best_known, time_limit, tmpdir):
    """One instance in a child process with an address-space cap and a timeout,
    so that an instance that needs too much memory cannot freeze the machine."""
    import json
    import resource
    import subprocess
    out_json = os.path.join(tmpdir, os.path.basename(path_dat) + ".json")
    cap = int(WORKER_AS_GB * 2**30)

    def limit():
        resource.setrlimit(resource.RLIMIT_AS, (cap, cap))
        os.nice(10)

    cmd = [sys.executable, os.path.abspath(__file__), "--worker", path_dat, "--best-known", str(best_known),
           "--time-limit", str(time_limit), "--worker-out", out_json]
    try:
        r = subprocess.run(cmd, preexec_fn=limit, timeout=time_limit + 1800, capture_output=True, text=True)
    except subprocess.TimeoutExpired:
        return {"status": "worker_timeout"}
    if r.returncode != 0 or not os.path.exists(out_json):
        tail = (r.stderr or "").strip().splitlines()[-1:] or [""]
        kind = "worker_memory" if "MemoryError" in (r.stderr or "") or r.returncode in (-9, 137) else "worker_error"
        return {"status": f"{kind}: {tail[0][:80]}"}
    with open(out_json) as f:
        return json.load(f)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--instances", default="instances_raw/pckp_benchmark")
    ap.add_argument("--out", default="experiments/results/literature_fixing.csv")
    ap.add_argument("--time-limit", type=float, default=300.0)
    ap.add_argument("--only", default="", help="comma list of instance names")
    ap.add_argument("--resume", action="store_true", help="keep the rows already in --out and skip those instances")
    ap.add_argument("--worker", help=argparse.SUPPRESS)
    ap.add_argument("--worker-out", help=argparse.SUPPRESS)
    ap.add_argument("--best-known", default="", help=argparse.SUPPRESS)
    a = ap.parse_args(argv)
    if a.worker:
        worker(a.worker, a.best_known, a.time_limit, a.worker_out)
        return
    import tempfile
    import time as _time
    manifest = {r["name"]: r for r in csv.DictReader(open(os.path.join(a.instances, "MANIFEST.csv")))}
    files = sorted(glob.glob(os.path.join(a.instances, "telecom", "*.lp.dat")) +
                   glob.glob(os.path.join(a.instances, "mining", "*.lp.dat")))
    if a.only:
        keep = set(a.only.split(","))
        files = [f for f in files if os.path.basename(f).replace(".lp.dat", "") in keep]
    name_of = lambda f: os.path.basename(f).replace(".lp.dat", "")
    # smallest first: n + m from the manifest
    files.sort(key=lambda f: int(manifest[name_of(f)]["n"]) + int(manifest[name_of(f)]["m"]))
    done = []
    if a.resume and os.path.exists(a.out):
        with open(a.out) as f:
            done = [r for r in csv.DictReader(line for line in f if not line.startswith("#")) if r["status"] == "ok"]
    done_names = {r["name"] for r in done}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    tmpdir = tempfile.mkdtemp(prefix="litfix-")
    with open(a.out, "w", newline="") as f:
        f.write(f"# literature_fixing.py time_limit={a.time_limit} instances={len(files)} "
                f"gurobi_mem_limit_gb={MEM_LIMIT_GB} worker_cap_gb={WORKER_AS_GB}\n# {environment()}\n")
        wr = csv.DictWriter(f, fieldnames=FIELDS)
        wr.writeheader()
        for r in done:
            wr.writerow({k: r.get(k, "") for k in FIELDS})
        f.flush()
        for path_dat in files:
            name = name_of(path_dat)
            if name in done_names:
                continue
            while available_gb() < MIN_AVAILABLE_GB:
                print(f"waiting: only {available_gb():.1f} GB available", file=sys.stderr, flush=True)
                _time.sleep(30)
            bk = manifest.get(name, {}).get("best_known_primal", "")
            row = run_isolated(path_dat, bk, a.time_limit, tmpdir)
            row.update(name=name, family=os.path.basename(os.path.dirname(path_dat)))
            wr.writerow({k: row.get(k, "") for k in FIELDS})
            f.flush()
            print(f"{name:14s} {row['status']:10s} {row.get('seconds', '')}", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
