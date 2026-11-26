"""Computing times of the strongest reduced costs on the PCKP benchmark.

For each instance, at its own capacity, the strongest reduced costs ``s*_i`` on
``Z`` and ``mu*_i`` on ``F`` are computed by the C++ engines of
``cpp/rc_engine.cpp``:

* ``cold``: one maximum flow from scratch per item (Dinic), i.e. one
  maximum-closure problem per item, the baseline;
* ``warm``: one base maximum flow, then for every item only the additional flow
  from the item to the sink in its residual network (Dinic), undone afterwards;
* ``sap``: as ``warm``, with shortest augmenting paths started from the exact
  distance labels of the base residual network and the gap rule;
* ``tree``: as ``sap``, with the items visited along a tree of prerequisites
  (the flow of the parent is kept for its descendants).

All engines must return identical values (checked).  The reported time is the
time spent inside C++ (base flow included), the minimum over ``--repeats`` runs.
Everything runs in this process, one instance at a time (memory: a few MB).

Usage::

    python experiments/rc_timing.py --out experiments/results/rc_timing.csv
"""
from __future__ import annotations

import argparse
import csv
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from macroitems import canonical_path, solution_from_path  # noqa: E402
from macroitems.fastrc import fast_reduced_costs  # noqa: E402
from macroitems.formats import read_pckp_dat  # noqa: E402
from forest_fixing import environment  # noqa: E402

ENGINES = ("cold", "warm", "sap", "tree")
FIELDS = ["name", "family", "n", "m", "F", "Z"] + [f"sec_{e}" for e in ENGINES]


def cpp_seconds(stats):
    return sum(stats[r][2] for r in ("Z", "F") if r in stats)


def run(path_dat, repeats):
    inst = read_pckp_dat(path_dat)
    c = float(inst.meta["capacity"])
    sol = solution_from_path(inst, canonical_path(inst), c)
    row = dict(name=inst.name, family=os.path.basename(os.path.dirname(path_dat)), n=inst.n, m=inst.m,
               F=int(sol.F.sum()), Z=int(sol.Z.sum()))
    ref = None
    for e in ENGINES:
        best = None
        for _ in range(repeats):
            val, st = fast_reduced_costs(inst, sol, e, return_stats=True)
            t = cpp_seconds(st)
            best = t if best is None else min(best, t)
            if t > 5.0:                      # slow runs: one is enough
                break
        if ref is None:
            ref = val
        elif not np.allclose(val, ref, atol=1e-9, equal_nan=True):
            raise AssertionError(f"{inst.name}: engine {e} disagrees with cold")
        row[f"sec_{e}"] = f"{best:.6f}"
    return row


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--instances", default="instances_raw/pckp_benchmark")
    ap.add_argument("--out", default="experiments/results/rc_timing.csv")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--only", default="")
    a = ap.parse_args(argv)
    files = sorted(glob.glob(os.path.join(a.instances, "telecom", "*.lp.dat")) +
                   glob.glob(os.path.join(a.instances, "mining", "*.lp.dat")))
    if a.only:
        keep = set(a.only.split(","))
        files = [f for f in files if os.path.basename(f).replace(".lp.dat", "") in keep]
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w", newline="") as f:
        f.write(f"# rc_timing.py repeats={a.repeats}; seconds inside C++ (min over repeats)\n"
                f"# {environment()}\n")
        wr = csv.DictWriter(f, fieldnames=FIELDS)
        wr.writeheader()
        for p in files:
            row = run(p, a.repeats)
            wr.writerow(row)
            f.flush()
            print(" ".join(f"{k}={row[k]}" for k in FIELDS), file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
