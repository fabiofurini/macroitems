"""Paper-sized tables and figures for the computational section.

The tables of ``make_tables.py`` are the complete record: 33 instances and
231 method-runs, right for a report appendix and far too wide for a journal
page.  This script produces the compact versions a section can carry, from
the same CSV files, and copies the two figures next to them.

Usage::

    python experiments/make_paper_tables.py --out ../../PAPER_LONG
"""
from __future__ import annotations

import argparse
import csv
import glob
import os
import shutil
import statistics
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")

MINELIB_ORDER = ["newman1", "zuck_small", "kd", "zuck_medium", "p4hd", "marvin",
                 "w23", "zuck_large", "sm2", "mclaughlin_limit"]


def load(pattern):
    rows = []
    for path in sorted(glob.glob(os.path.join(RESULTS, pattern))):
        with open(path) as f:
            rows += list(csv.DictReader(f))
    return rows


def num(row, key):
    try:
        return float(row[key])
    except (KeyError, TypeError, ValueError):
        return None


def tex_name(name):
    return name.replace("_", r"\_").replace(".lp.dat", "")


def group(rows):
    by = defaultdict(dict)
    for r in rows:
        by[r["instance"]][r["method"]] = r
    return by


# ------------------------------------------------------------------ tables
def structure_table(rows):
    """One row per instance: the canonical sequence and the split macroitem."""
    lines = [r"\begin{table}[tbp]", r"\centering",
             r"\caption{The canonical macroitem sequence of every instance, and the "
             r"split macroitem at the instance's own capacity. $k$ is the number of "
             r"macroitems and $q$ the last with positive ratio; \emph{persist.} is "
             r"$1-|\Hset|/n$, the fraction of items fixed in every optimal solution "
             r"(\cref{cor:persistency}); \emph{gap} is the bound of \cref{prop:gap} "
             r"relative to $z(c)$}",
             r"\label{tab:structure}", r"\footnotesize",
             r"\begin{tabular}{lrrrrrrr}", r"\toprule",
             r"instance & $n$ & $m$ & $k$ & $q$ & $|\Hset|$ & persist. & gap \\",
             r"\midrule"]
    families = {"telecom": [], "mining": [], "minelib": []}
    for r in rows:
        families.setdefault(r.get("family", ""), []).append(r)
    for fam, label in (("telecom", "PCKP benchmark, telecom"),
                       ("mining", "PCKP benchmark, mining"),
                       ("minelib", "MineLib")):
        items = sorted(families.get(fam, []), key=lambda r: int(r["n"]))
        if not items:
            continue
        lines.append(rf"\multicolumn{{8}}{{@{{}}l}}{{\emph{{{label}}}}} \\")
        for r in items:
            pers = num(r, "persistency")
            gap = num(r, "gap_bound_rel")
            lines.append(
                f"\\quad {tex_name(r['name'])} & {int(r['n']):,} & {int(r['m']):,} & "
                f"{int(r['k'])} & {int(r['q'])} & "
                f"{int(float(r['H_size'])) if r.get('H_size') else '--'} & "
                f"{pers:.3f}" .replace(",", r"\,") +
                (f" & {100 * gap:.0f}\\%" if gap is not None else " & --") + r" \\"
                if pers is not None else
                f"\\quad {tex_name(r['name'])} & {int(r['n']):,} & {int(r['m']):,} & "
                f"{int(r['k'])} & {int(r['q'])} & -- & -- & -- \\\\".replace(",", r"\,"))
        lines.append(r"\addlinespace")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def benchmark_table(by):
    """The benchmark comparison, grouped by arc density."""
    sparse, dense = [], []
    for inst, ms in by.items():
        if "path" not in ms or ms["path"].get("status") != "ok":
            continue
        m = int(ms["path"]["m"])
        best = [num(ms[k], "seconds_total") for k in
                ("highs:dual-simplex", "gurobi:dual-simplex", "cplex-cli:dual-simplex")
                if k in ms and ms[k].get("status") == "ok"]
        p = num(ms["path"], "seconds_total")
        n_ = num(ms["newton"], "seconds_first") if "newton" in ms else None
        f_ = [num(ms[k], "seconds_first") for k in
              ("highs:dual-simplex", "gurobi:dual-simplex", "cplex-cli:dual-simplex")
              if k in ms and ms[k].get("status") == "ok"]
        if not best or not p:
            continue
        entry = (min(best) / p, (min(f_) / n_) if f_ and n_ else None)
        (sparse if m < 10_000 else dense if m >= 20_000 else sparse).append(entry)

    def stats(entries, idx):
        vals = [e[idx] for e in entries if e[idx] is not None]
        if not vals:
            return "--", "--"
        return f"{statistics.median(vals):.1f}", f"{min(vals):.1f}--{max(vals):.1f}"

    lines = [r"\begin{table}[tbp]", r"\centering",
             r"\caption{Speed-up of the combinatorial methods over the fastest of the "
             r"three dual simplex codes on the \PCKP\ benchmark, grouped by the number "
             r"of precedence arcs. \emph{one capacity} compares the Newton search "
             r"against one solve; \emph{all capacities} compares the canonical path "
             r"against a grid of twenty. A value below 1 means the solver is faster}",
             r"\label{tab:methods-benchmark}",
             r"\begin{tabular}{lrrrrr}", r"\toprule",
             r"& & \multicolumn{2}{c}{one capacity} & \multicolumn{2}{c}{all capacities} \\",
             r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}",
             r"group & inst. & median & range & median & range \\", r"\midrule"]
    for label, entries in (("sparse, $m<10^4$", sparse), ("dense, $m\\ge2\\cdot10^4$", dense)):
        m_all, r_all = stats(entries, 0)
        m_one, r_one = stats(entries, 1)
        lines.append(f"{label} & {len(entries)} & {m_one} & {r_one} & "
                     f"\\textbf{{{m_all}}} & {r_all} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def minelib_table(by):
    """The open-pit comparison, one row per instance."""
    lines = [r"\begin{table}[tbp]", r"\centering",
             r"\caption{MineLib: total time for ten capacities, single-threaded, with a "
             r"budget of 150 seconds per method. \emph{best solver} is the fastest of "
             r"the three dual simplex codes; \textsc{to} means that none of them "
             r"finished a single capacity within the budget}",
             r"\label{tab:methods-minelib}", r"\footnotesize",
             r"\begin{tabular}{lrrrrrr}", r"\toprule",
             r"instance & $n$ & $m$ & path & Newton & best solver & ratio \\",
             r"\midrule"]
    for inst in MINELIB_ORDER:
        ms = by.get(inst)
        if not ms:
            continue
        p = num(ms.get("path", {}), "seconds_total")
        nw = num(ms.get("newton", {}), "seconds_total")
        best = [num(ms[k], "seconds_total") for k in
                ("highs:dual-simplex", "gurobi:dual-simplex", "cplex-cli:dual-simplex")
                if k in ms and ms[k].get("status") == "ok"]
        n_, m_ = int(ms["path"]["n"]), int(ms["path"]["m"])
        if best:
            b = min(best)
            cells = f"{b:.1f} & ${b / p:.1f}\\times$"
        else:
            cells = r"\textsc{to} & ---"
        lines.append(f"{tex_name(inst)} & {n_:,} & {m_:,} & {p:.2f} & {nw:.2f} & {cells} \\\\"
                     .replace(",", r"\,"))
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="directory of the manuscript")
    args = ap.parse_args(argv)

    tables = os.path.join(args.out, "tables")
    figures = os.path.join(args.out, "figures")
    os.makedirs(tables, exist_ok=True)
    os.makedirs(figures, exist_ok=True)

    structure = load("characteristics_*.csv")
    by = group(load("compare_*.csv"))

    written = []
    for name, text in (("structure", structure_table(structure)),
                       ("methods_benchmark", benchmark_table(by)),
                       ("methods_minelib", minelib_table(by))):
        path = os.path.join(tables, f"{name}.tex")
        with open(path, "w") as f:
            f.write(text + "\n")
        written.append(os.path.basename(path))

    for src in ("kd_value-function.png", "newman1_persistency.png"):
        source = os.path.join(RESULTS, "figures", src)
        if os.path.exists(source):
            shutil.copy2(source, os.path.join(figures, src))
            written.append(f"figures/{src}")
    print("wrote: " + ", ".join(written))


if __name__ == "__main__":
    main()
