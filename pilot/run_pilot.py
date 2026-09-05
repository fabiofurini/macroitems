"""Pilot experiment: canonical macroitem sequences, LP at given capacities, dual
certificates, face dimensions and structural statistics on synthetic
mining-like block models and random DAGs.  Writes CSV tables and a markdown
report into pilot/results/.

Usage: python3 pilot/run_pilot.py [--quick]
"""
import argparse
import csv
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from macroitems import (layered_grid, random_dag, canonical_path, solve_capacity, solution_from_path,
                        canonical_dual, solve_lp, Instance)
from macroitems.stats import structural_stats, path_summary, revenue_factor_family, heuristic_integer

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results")
INST_DIR = os.path.join(HERE, "..", "instances")
os.makedirs(OUT, exist_ok=True)
os.makedirs(INST_DIR, exist_ok=True)


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def build_instances(quick: bool):
    specs = [
        ("grid_S", lambda: layered_grid(12, 12, 8, cone=5, seed=1, name="grid_S")),      # ~newman1 size
        ("grid_M", lambda: layered_grid(30, 30, 10, cone=9, seed=1, name="grid_M")),     # ~zuck_small size
        ("grid_L", lambda: layered_grid(36, 36, 11, cone=9, seed=2, name="grid_L")),     # ~kd size
        ("dag_S", lambda: random_dag(2000, avg_out_degree=2.0, seed=1, name="dag_S")),
        ("dag_M", lambda: random_dag(20000, avg_out_degree=3.0, seed=2, name="dag_M")),
    ]
    if not quick:
        specs.append(("grid_XL", lambda: layered_grid(60, 60, 15, cone=9, seed=3, name="grid_XL")))  # ~marvin size
    return specs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    summary_rows, cap_rows, timing_rows, rf_rows = [], [], [], []
    for name, make in build_instances(args.quick):
        inst = make()
        inst.write(os.path.join(INST_DIR, name + ".txt"))
        log(f"== {name}: n={inst.n} m={inst.m}")
        # ---- whole path
        path = canonical_path(inst, "bisection")
        chk = path.check(inst)
        log(f"   bisection: k={path.k} q={path.q} maxflows={path.n_maxflow} {path.seconds:.2f}s check={chk}")
        ps = path_summary(inst, path)
        ps.update(instance=name, check=all(chk.values()))
        if inst.n <= 15000:
            path2 = canonical_path(inst, "dinkelbach")
            same = path.k == path2.k and all(np.array_equal(a, b) for a, b in zip(path.macroitems, path2.macroitems))
            log(f"   dinkelbach: k={path2.k} maxflows={path2.n_maxflow} {path2.seconds:.2f}s same={same}")
            ps.update(dinkelbach_maxflow=path2.n_maxflow, dinkelbach_seconds=path2.seconds, dinkelbach_same=same)
        # positive part statistics
        q = path.q
        sizes_pos = np.array([I.size for I in path.macroitems[:q]])
        wts_pos = np.array([inst.w[I].sum() for I in path.macroitems[:q]])
        ps.update(size_pos_median=float(np.median(sizes_pos)) if q else np.nan,
                  size_pos_max=int(sizes_pos.max()) if q else 0,
                  n_items_Mq=int(sizes_pos.sum()) if q else 0,
                  top3_weight_share=float(np.sort(wts_pos)[-3:].sum() / path.W[q]) if q else np.nan)
        # ---- single-capacity: Newton vs HiGHS
        for f in (0.25, 0.5, 0.75):
            c = f * path.W[q]
            s = solve_capacity(inst, c)
            lim = 900.0
            l = solve_lp(inst, c, time_limit=lim)
            d = canonical_dual(inst, s, c)
            row = dict(instance=name, f=f, c=c, newton_maxflow=s.n_maxflow, newton_seconds=s.seconds,
                       highs_seconds=l.seconds, highs_status=l.status, z_newton=s.value, z_highs=l.value,
                       rel_diff=abs(s.value - l.value) / max(1.0, abs(l.value)) if l.status == 0 else np.nan,
                       lam_newton=s.lam, lam_highs=l.lam, size_H=int(s.H.sum()), theta=s.theta,
                       dual_feasible=d.feasible, dual_value=d.value)
            timing_rows.append(row)
            log(f"   c={f:.2f}W_q: newton {s.n_maxflow} mf {s.seconds:.3f}s | highs {l.seconds:.2f}s status {l.status} | "
                f"z {s.value:.3f} vs {l.value:.3f} | |H|={int(s.H.sum())} theta={s.theta:.3f} dual ok={d.feasible}")
        # ---- structural statistics on a capacity grid
        rows = structural_stats(inst, path, faces_max_H=3000 if inst.n <= 15000 else 1500)
        for r in rows:
            r["instance"] = name
        cap_rows += rows
        log("   grid stats: " + "; ".join(f"f={r['f']:.1f} h={r['h']} |H|={r['size_H']} th={r['theta']:.2f} gapb={r['gap_bound_rel']:.3%} "
                                          f"gaph={r['gap_heur_rel']:.3%} dimD={r.get('dim_dual','-')} k0={r.get('k0','-')}" for r in rows))
        # ---- revenue factor vs weight family
        if "revenue" in inst.extra:
            rf = revenue_factor_family(inst, path, inst.extra["revenue"], inst.extra["cost"])
            rf["instance"] = name
            rf_rows.append(rf)
            log(f"   revenue-factor pits coinciding with weight family: {rf['n_coincide']}/{rf['n_factors']}, "
                f"mean rel. symdiff to nearest-weight closure {rf['mean_rel_symdiff']:.3f}")
        summary_rows.append(ps)
        # save per-instance path
        with open(os.path.join(OUT, f"path_{name}.json"), "w") as fjs:
            json.dump(dict(k=path.k, q=path.q, ratios=path.ratios.tolist(), P=path.P.tolist(), W=path.W.tolist(),
                           sizes=[int(I.size) for I in path.macroitems]), fjs)

    def write_csv(fn, rows):
        keys = []
        for r in rows:
            for k in r:
                if k not in keys:
                    keys.append(k)
        with open(os.path.join(OUT, fn), "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=keys)
            wr.writeheader()
            for r in rows:
                wr.writerow(r)
    write_csv("summary.csv", summary_rows)
    write_csv("capacity_grid.csv", cap_rows)
    write_csv("timing.csv", timing_rows)
    write_csv("revenue_factor.csv", rf_rows)
    log("done")


if __name__ == "__main__":
    main()
