"""LaTeX for the table (and the macros quoted in the text) of computing times on the PCKP benchmark, generated from
the CSVs of ``rc_timing.py`` and ``bb_fixing.py`` (variants base and best).

For each instance: the share of the items of ``Z`` fixed by the strongest
reduced costs; the time to compute them by one maximum closure per item
(``cold``) and by the incremental scheme (``tree``); the time and the nodes of
Gurobi on the full model and with the variables fixed (time of the reduced
costs included).  Times and nodes are geometric means over the seeds.

Usage::

    python experiments/bb_table.py experiments/results/rc_timing.csv \\
        experiments/results/bb_fixing.csv \\
        --rows experiments/results/tables/bb_fixing.tex \\
        --macros experiments/results/tables/bb_fixing_macros.tex
"""
from __future__ import annotations

import argparse
import csv
import math


def load(path):
    with open(path) as f:
        return list(csv.DictReader(line for line in f if not line.startswith("#")))


def geo(xs):
    return math.exp(sum(math.log(max(x, 1e-3)) for x in xs) / len(xs))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("rc_csv")
    ap.add_argument("bb_csv")
    ap.add_argument("--rows")
    ap.add_argument("--macros")
    a = ap.parse_args(argv)
    rc = {r["name"]: r for r in load(a.rc_csv)}
    bb = {}
    for r in load(a.bb_csv):
        bb.setdefault(r["name"], {}).setdefault(r["variant"], []).append(r)
    g = lambda x: "\\gen{" + x + "}"
    body, speed_rc, speed_bb, red_nodes = [], [], [], []
    tot_cold = tot_tree = 0.0
    fam_bb = {}
    names = sorted(bb, key=lambda n: (rc[n]["family"] != "telecom", n))
    for k, name in enumerate(names):
        if k and rc[name]["family"] != rc[names[k - 1]]["family"]:
            body.append("\\midrule")
        r, v = rc[name], bb[name]
        for rows in v.values():
            assert all(x["status"] in ("optimal", "cutoff") for x in rows), name
        Z = int(r["Z"])
        fixed = int(v["best"][0]["fixed0"]) + int(v["best"][0]["fixed1"])
        cold, tree = float(r["sec_cold"]), float(r["sec_tree"])
        tb = geo([float(x["grb_seconds"]) for x in v["base"]])
        nb = geo([float(x["nodes"]) for x in v["base"]])
        tf = geo([float(x["grb_seconds"]) for x in v["best"]]) + tree
        nf = geo([float(x["nodes"]) for x in v["best"]])
        tot_cold += cold
        tot_tree += tree
        fam_bb.setdefault(r["family"], []).append(tb / tf)
        speed_rc.append(cold / tree)
        speed_bb.append(tb / tf)
        red_nodes.append(nf / nb if nb > 0 else 1.0)
        cells = [r["n"], f"{100 * fixed / Z:.1f}" if Z else "0.0", f"{cold:.2f}", f"{tree:.3f}",
                 f"{tb:.2f}", f"{nb:.0f}", f"{tf:.2f}", f"{nf:.0f}"]
        body.append(f"{name.split('_')[0]} & " + " & ".join(g(c) for c in cells) + "\\\\")
    rows = "\n".join(body) + "\n"
    macros = "\n".join([
        f"\\newcommand{{\\bbCount}}{{{g(str(len(names)))}}}",
        f"\\newcommand{{\\bbRcSpeedMin}}{{{g(f'{min(speed_rc):.0f}')}}}",
        f"\\newcommand{{\\bbRcSpeedMax}}{{{g(f'{max(speed_rc):.0f}')}}}",
        f"\\newcommand{{\\bbRcSpeedGeo}}{{{g(f'{geo(speed_rc):.0f}')}}}",
        f"\\newcommand{{\\bbBbSpeedMax}}{{{g(f'{max(speed_bb):.1f}')}}}",
        f"\\newcommand{{\\bbBbSpeedMin}}{{{g(f'{min(speed_bb):.2f}')}}}",
        f"\\newcommand{{\\bbBbSpeedGeo}}{{{g(f'{geo(speed_bb):.2f}')}}}",
        f"\\newcommand{{\\bbRcColdTotal}}{{{g(f'{tot_cold:.0f}')}}}",
        f"\\newcommand{{\\bbRcTreeTotal}}{{{g(f'{tot_tree:.1f}')}}}",
        f"\\newcommand{{\\bbBbSpeedGeoTelecom}}{{{g("%.2f" % geo(fam_bb.get("telecom", [1])))}}}",
        f"\\newcommand{{\\bbBbSpeedGeoMining}}{{{g("%.2f" % geo(fam_bb.get("mining", [1])))}}}",
        f"\\newcommand{{\\bbNodesRatioMin}}{{{g(f'{min(red_nodes):.2f}')}}}",
        f"\\newcommand{{\\bbNodesRatioMax}}{{{g(f'{max(red_nodes):.2f}')}}}",
    ]) + "\n"
    for path, text in ((a.rows, rows), (a.macros, macros)):
        if path:
            with open(path, "w") as f:
                f.write(text)
        else:
            print(text)
    for n, s1, s2, s3 in zip(names, speed_rc, speed_bb, red_nodes):
        print(f"{n:14s} rc x{s1:6.1f}  B&B x{s2:5.2f}  nodes ratio {s3:5.2f}")


if __name__ == "__main__":
    main()
