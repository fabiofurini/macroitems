"""Structural characterization of a set of PCKP instances.

For every instance this computes the canonical macroitem sequence and the LP
solution at the instance's own capacity, and reports what the paper's theory
says is worth knowing about it:

* size: ``n``, ``m``, arc density;
* the canonical sequence: ``k`` (number of macroitems), ``q`` (last positive
  ratio), the size of the largest macroitem, the weight share of the first
  macroitem and of the first three;
* at the given capacity: the split macroitem index ``h``, its size ``|H|``,
  the fill ``theta``, the persistency ``1 - |H|/n`` (the fraction of items
  fixed in every optimal solution, Corollary 5.1), and the integrality-gap
  bound ``theta * p(I_h)`` of Proposition 5.2;
* the LP value ``z(c)``, and -- when a best known integer optimum is supplied
  -- the true integrality gap.

Usage::

    python experiments/characterize.py <dir-or-file> [...] --out results.csv
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

from macroitems import canonical_path, solution_from_path  # noqa: E402
from macroitems.formats import read_any, read_minelib_upit  # noqa: E402


def characterize(inst, capacity=None, best_known=None, method="bisection",
                 with_dual=True, max_reduced_costs=2000, max_faces=5000):
    """One row of structural statistics for one instance."""
    work, scale = inst.scaled_to_integers()
    t0 = time.perf_counter()
    path = canonical_path(work, method=method)
    t_path = time.perf_counter() - t0

    sizes = np.array([I.size for I in path.macroitems])
    weights = np.array([work.w[I].sum() for I in path.macroitems])
    q = path.q
    w_pos = float(path.W[q]) if q else 0.0

    row = {
        "name": inst.name,
        "n": inst.n,
        "m": inst.m,
        "density": round(inst.m / inst.n, 3),
        "k": path.k,
        "q": q,
        "n_maxflow_path": path.n_maxflow,
        "seconds_path": round(t_path, 4),
        "largest_macroitem": int(sizes.max()),
        "largest_macroitem_share": round(float(sizes.max()) / inst.n, 4),
        "median_macroitem": int(np.median(sizes)),
        "n_singleton_macroitems": int((sizes == 1).sum()),
        "size_Mq": int(sum(I.size for I in path.macroitems[:q])),
        # a ratio p/w is invariant under the integer rescaling: both the
        # numerator and the denominator are multiplied by the same factor
        "lambda_1": float(path.ratios[0]) if path.k else float("nan"),
        "lambda_q": float(path.ratios[q - 1]) if q else float("nan"),
        "w_share_first": round(float(weights[0]) / w_pos, 4) if w_pos else "",
        "w_share_first3": round(float(weights[:3].sum()) / w_pos, 4) if w_pos else "",
    }

    if capacity is not None:
        c_work = capacity * scale
        sol = solution_from_path(work, path, c_work)
        H = int(sol.H.sum())
        h = sol.h
        gap_bound = sol.theta * float(work.p[sol.H].sum()) / scale if H else 0.0
        z = sol.value / scale
        row.update({
            "capacity": capacity,
            "capacity_over_wMq": round(c_work / w_pos, 4) if w_pos else "",
            "h": h,
            "H_size": H,
            "H_share": round(H / inst.n, 4),
            "theta": round(float(sol.theta), 4),
            "persistency": round(1.0 - H / inst.n, 4),
            "z_lp": z,
            "gap_bound_abs": gap_bound,
            "gap_bound_rel": round(gap_bound / z, 6) if z else "",
        })
        if best_known is not None and z:
            row["z_ip_best_known"] = best_known
            row["true_gap_rel"] = round((z - best_known) / z, 6)

        # What the dual side costs, and what it buys (experiment E3 of the plan).
        if with_dual and H:
            from macroitems.dual import best_reduced_costs, canonical_reduced_costs
            from macroitems.faces import face_dimensions
            from macroitems.path import canonical_dual
            from macroitems.stats import heuristic_integer

            t0 = time.perf_counter()
            dual = canonical_dual(work, sol, c_work)
            row["seconds_dual"] = round(time.perf_counter() - t0, 4)
            row["dual_feasible"] = bool(dual.feasible)

            rc = best_reduced_costs(work, sol, max_items=max_reduced_costs)
            finite = rc.value[np.isfinite(rc.value)]
            row["seconds_reduced_costs"] = round(rc.seconds, 4)
            row["n_reduced_costs"] = int(finite.size)
            if h is not None and finite.size:
                canon = canonical_reduced_costs(work, path, h)
                paired = np.isfinite(rc.value) & (rc.region != "H")
                if paired.any():
                    gain = (rc.value[paired] - canon[paired]) / np.maximum(1.0, canon[paired])
                    row["reduced_cost_gain_median"] = round(float(np.median(gain)), 4)

            # k0 and dim D* need |H| minimum cuts, so they are skipped when the
            # split macroitem is large -- which Corollary 5.1 says is the rare case.
            if H <= max_faces:
                t0 = time.perf_counter()
                info = face_dimensions(work, sol, max_H=max_faces)
                row["seconds_faces"] = round(time.perf_counter() - t0, 4)
                row["k0"] = info.k0
                row["dim_primal"] = info.dim_primal
                row["dim_dual"] = info.dim_dual

            z_heur = heuristic_integer(work, path, c_work) / scale
            if z:
                row["z_heuristic"] = z_heur
                row["heuristic_gap_rel"] = round((z - z_heur) / z, 6)
    return row


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="+", help="instance files or directories")
    ap.add_argument("--out", default=None, help="CSV output file")
    ap.add_argument("--method", default="bisection", choices=["bisection", "dinkelbach"])
    ap.add_argument("--no-dual", action="store_true",
                    help="skip the dual certificate, reduced costs and face dimensions")
    ap.add_argument("--max-faces", type=int, default=5000,
                    help="skip the face dimensions when |H| exceeds this")
    ap.add_argument("--manifest", default=None,
                    help="CSV with columns name,capacity,best_known_primal to take capacities from")
    args = ap.parse_args(argv)

    known = {}
    if args.manifest:
        with open(args.manifest) as f:
            for r in csv.DictReader(f):
                known[r["name"]] = r

    files = []
    for p in args.paths:
        if os.path.isdir(p):
            found = sorted(glob.glob(os.path.join(p, "**", "*.dat"), recursive=True))
            if not found:      # a MineLib directory: one instance per .upit file
                found = [os.path.splitext(f)[0]
                         for f in sorted(glob.glob(os.path.join(p, "*.upit")))]
            files += found
        else:
            files.append(p)

    rows = []
    for path in files:
        inst = read_minelib_upit(path) if not os.path.splitext(path)[1] else read_any(path)
        info = known.get(inst.name, {})
        cap = inst.meta.get("capacity", inst.meta.get("capacity_cpit_period"))
        if info.get("capacity"):
            cap = float(info["capacity"])
        best = float(info["best_known_primal"]) if info.get("best_known_primal") else None
        row = characterize(inst, capacity=cap, best_known=best, method=args.method,
                           with_dual=not args.no_dual, max_faces=args.max_faces)
        row["family"] = info.get("family", inst.meta.get("family", ""))
        rows.append(row)
        print(f"{row['name']:20s} n={row['n']:6d} k={row['k']:5d} q={row['q']:5d} "
              f"|H|={row.get('H_size', ''):>6} pers={row.get('persistency', ''):>7} "
              f"{row['seconds_path']:7.2f}s", flush=True)

    if args.out:
        fields = sorted({k for r in rows for k in r})
        head = [f for f in ("name", "family", "n", "m", "density", "k", "q") if f in fields]
        head += [f for f in fields if f not in head]
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=head)
            writer.writeheader()
            writer.writerows(rows)
        print(f"\n{len(rows)} instances -> {args.out}")
    return rows


if __name__ == "__main__":
    main()
