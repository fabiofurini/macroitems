"""How best to solve the LP relaxation: the combinatorial methods against the solvers.

The comparison is run per instance along the axis that actually decides the
answer -- *how many capacities are needed* -- because the two families of
method scale differently in it:

* an LP solver answers one capacity per solve, and re-solving at a nearby
  capacity is cheap only because the simplex basis carries over;
* the canonical path (``canonical_path``) answers *every* capacity at once:
  after it is computed, each further capacity costs a binary search over the
  breakpoints, i.e. nothing;
* the Newton search (``solve_capacity``) sits in between: a handful of maximum
  flows for one capacity, paid again for the next.

So the table reports, for each method, the cost of the first capacity and the
marginal cost of each further one, and the script derives the break-even
number of capacities beyond which the path is the cheaper choice.

Every solver runs single-threaded, and every method is checked against the
others: the values must agree to 1e-9 relative or the row is flagged.

Because ``highspy`` and ``ortools`` cannot be loaded into the same process
(see :mod:`macroitems.lp`), each method runs in its own subprocess, which also
keeps one solver's memory from perturbing another's timing.

Usage::

    python experiments/compare_methods.py <instance>... [--capacities 20] [--out out.csv]
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

# The methods of the comparison.  "path" and "newton" are ours; the rest are
# general-purpose LP solvers reached through macroitems.lp.
COMBINATORIAL = ("path", "newton")
DEFAULT_SOLVERS = ("highs:dual-simplex", "highs:ipm", "gurobi:dual-simplex",
                   "gurobi:barrier", "cplex-cli:dual-simplex")


def capacities_for(inst, count: int) -> np.ndarray:
    """A grid of capacities spanning the interesting range 0 < c < w(M_q)."""
    from macroitems import canonical_path
    path = canonical_path(inst)
    top = float(path.W[path.q]) if path.q else float(inst.w.sum())
    return np.linspace(top / (count + 1), top * count / (count + 1), count)


# --------------------------------------------------------------- one method
def run_method(spec: str, path_instance: str, capacities: list) -> dict:
    """Run one method on one instance, in this process.  Returns a result dict."""
    from macroitems.formats import read_any, read_minelib_upit

    reader = read_minelib_upit if not os.path.splitext(path_instance)[1] else read_any
    inst = reader(path_instance)
    inst, scale = inst.scaled_to_integers()
    caps = [c * scale for c in capacities]

    if spec == "path":
        from macroitems import canonical_path, solution_from_path
        t0 = time.perf_counter()
        p = canonical_path(inst)
        t_first = time.perf_counter() - t0
        t1 = time.perf_counter()
        values = [solution_from_path(inst, p, c).value / scale for c in caps]
        t_rest = (time.perf_counter() - t1) / max(1, len(caps))
        return {"method": spec, "seconds_first": t_first, "seconds_per_extra": t_rest,
                "seconds_total": t_first + t_rest * len(caps), "values": values,
                "n_maxflow": p.n_maxflow, "k": p.k, "q": p.q}

    if spec == "newton":
        from macroitems import solve_capacity
        values, times, flows = [], [], 0
        for c in caps:
            t0 = time.perf_counter()
            sol = solve_capacity(inst, c)
            times.append(time.perf_counter() - t0)
            values.append(sol.value / scale)
            flows += sol.n_maxflow
        return {"method": spec, "seconds_first": times[0],
                "seconds_per_extra": float(np.mean(times[1:])) if len(times) > 1 else times[0],
                "seconds_total": float(np.sum(times)), "values": values, "n_maxflow": flows}

    backend, _, method = spec.partition(":")
    from macroitems.lp import get_lp_backend
    cls = get_lp_backend(backend)
    kwargs = {"return_primal": False} if backend == "cplex-cli" else {}
    solver = cls(inst, method=method or "default", threads=1, **kwargs)
    try:
        values, times = [], []
        for c in caps:
            r = solver.solve(c)
            times.append(r.seconds_solve)
            values.append(r.value / scale if np.isfinite(r.value) else float("nan"))
        return {"method": spec, "seconds_build": solver.seconds_build,
                "seconds_first": solver.seconds_build + times[0],
                "seconds_per_extra": float(np.mean(times[1:])) if len(times) > 1 else times[0],
                "seconds_total": solver.seconds_build + float(np.sum(times)), "values": values}
    finally:
        solver.close()


# ------------------------------------------------------------- orchestration
def run_in_subprocess(spec: str, instance: str, capacities: list, timeout: float) -> dict:
    """Run one method in a fresh interpreter, so library clashes cannot bite."""
    payload = json.dumps({"spec": spec, "instance": instance, "capacities": capacities})
    try:
        out = subprocess.run([sys.executable, os.path.abspath(__file__), "--worker", payload],
                             capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return {"method": spec, "status": "timeout", "seconds_total": timeout}
    tag = "RESULT:"
    for line in out.stdout.splitlines():
        if line.startswith(tag):
            result = json.loads(line[len(tag):])
            result["status"] = "ok"
            return result
    return {"method": spec, "status": "error",
            "message": (out.stderr or out.stdout).strip().splitlines()[-1][:200] if (out.stderr or out.stdout).strip() else "no output"}


def compare(instance: str, n_capacities: int, solvers: tuple, timeout: float) -> list:
    from macroitems.formats import read_any, read_minelib_upit
    reader = read_minelib_upit if not os.path.splitext(instance)[1] else read_any
    inst = reader(instance)
    caps = [float(c) for c in capacities_for(inst, n_capacities)]

    rows = []
    reference = None
    for spec in list(COMBINATORIAL) + list(solvers):
        res = run_in_subprocess(spec, instance, caps, timeout)
        res["instance"] = inst.name
        res["n"] = inst.n
        res["m"] = inst.m
        res["n_capacities"] = n_capacities
        values = res.pop("values", None)
        if values is not None:
            if reference is None:
                reference = values
                res["max_rel_gap_vs_reference"] = 0.0
            else:
                ref = np.asarray(reference)
                got = np.asarray(values)
                denom = np.maximum(1.0, np.abs(ref))
                res["max_rel_gap_vs_reference"] = float(np.nanmax(np.abs(got - ref) / denom))
        rows.append(res)
        mark = "" if res.get("max_rel_gap_vs_reference", 0) < 1e-9 else "  <-- VALUES DIFFER"
        print(f"  {spec:26s} {res.get('status', ''):8s} first {res.get('seconds_first', float('nan')):8.3f}s"
              f"  per extra {res.get('seconds_per_extra', float('nan')):8.4f}s"
              f"  total {res.get('seconds_total', float('nan')):8.3f}s{mark}", flush=True)
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("instances", nargs="*")
    ap.add_argument("--capacities", type=int, default=20)
    ap.add_argument("--solvers", default=",".join(DEFAULT_SOLVERS))
    ap.add_argument("--timeout", type=float, default=3600.0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--worker", default=None, help=argparse.SUPPRESS)
    args = ap.parse_args(argv)

    if args.worker:                      # child process: run one method, print JSON
        job = json.loads(args.worker)
        print("RESULT:" + json.dumps(run_method(job["spec"], job["instance"], job["capacities"])))
        return

    solvers = tuple(s for s in args.solvers.split(",") if s)
    rows = []
    for instance in args.instances:
        print(f"\n{os.path.basename(instance)}  ({args.capacities} capacities)")
        rows += compare(instance, args.capacities, solvers, args.timeout)

    if args.out and rows:
        fields = sorted({k for r in rows for k in r})
        head = [f for f in ("instance", "n", "m", "method", "status", "seconds_first",
                            "seconds_per_extra", "seconds_total") if f in fields]
        head += [f for f in fields if f not in head]
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=head)
            writer.writeheader()
            writer.writerows(rows)
        print(f"\n{len(rows)} rows -> {args.out}")


if __name__ == "__main__":
    main()
