"""Turn the raw experiment CSVs into the tables of the report and of the paper.

Two outputs from the same numbers, so they can never drift apart:

* Markdown, for ``REPORT.md``;
* LaTeX (``booktabs``), written as ``\\input``-able fragments.

Tables produced:

``structure``
    one row per instance: size, the canonical sequence (k, q, largest
    macroitem), and, at the instance's own capacity, the split macroitem, the
    persistency and the integrality-gap bound.
``methods``
    the method comparison: cost of the first capacity, marginal cost of each
    further one, and the resulting break-even number of capacities beyond
    which computing the whole path is cheaper than re-solving.
``recommendation``
    the summary the reader actually wants: which method to use in which
    regime, derived from the numbers rather than asserted.

Usage::

    python experiments/make_tables.py --results experiments/results --out experiments/results/tables
"""
from __future__ import annotations

import argparse
import csv
import glob
import os
from collections import defaultdict


def _load_all(directory, pattern):
    """Concatenate every CSV matching ``pattern``, in a stable order."""
    rows = []
    for path in sorted(glob.glob(os.path.join(directory, pattern))):
        rows += read_csv(path)
    return rows


def read_csv(path):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return list(csv.DictReader(f))


def _f(row, key, default=float("nan")):
    value = row.get(key, "")
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _i(row, key, default=0):
    try:
        return int(float(row.get(key, "")))
    except (TypeError, ValueError):
        return default


# --------------------------------------------------------------- rendering
def to_markdown(header, rows, aligns=None) -> str:
    # a cell containing "|" would end the column early
    def cell(value):
        return str(value).replace("|", "\\|")

    aligns = aligns or ["---"] * len(header)
    out = ["| " + " | ".join(cell(h) for h in header) + " |",
           "|" + "|".join(aligns) + "|"]
    out += ["| " + " | ".join(cell(c) for c in row) + " |" for row in rows]
    return "\n".join(out)


_LATEX_ESCAPES = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$",
                  "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}",
                  "~": r"\textasciitilde{}", "^": r"\textasciicircum{}"}


def latex_escape(value) -> str:
    """Escape a cell for LaTeX.

    Instance names carry underscores, which LaTeX reads as subscripts and
    which stop the build outside maths mode.
    """
    text = str(value)
    return "".join(_LATEX_ESCAPES.get(ch, ch) for ch in text)


def to_latex(header, rows, caption, label, column_spec=None) -> str:
    """A float for a short table, a longtable for one that cannot fit a page."""
    spec = column_spec or ("l" + "r" * (len(header) - 1))
    header = [latex_escape(h) for h in header]
    rows = [[latex_escape(c) for c in row] for row in rows]
    body = [" & ".join(str(c) for c in row) + r" \\" for row in rows]
    if len(rows) <= 25:
        return "\n".join(
            [r"\begin{table}[tbp]", r"\centering",
             rf"\caption{{{caption}}}", rf"\label{{{label}}}",
             rf"\begin{{tabular}}{{{spec}}}", r"\toprule",
             " & ".join(header) + r" \\", r"\midrule"]
            + body + [r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    head = " & ".join(header) + r" \\"
    return "\n".join(
        [rf"\begin{{longtable}}{{{spec}}}",
         rf"\caption{{{caption}}}\label{{{label}}}\\",
         r"\toprule", head, r"\midrule", r"\endfirsthead",
         r"\toprule", head, r"\midrule", r"\endhead",
         r"\bottomrule", r"\endfoot"]
        + body + [r"\end{longtable}"])


# ----------------------------------------------------------------- tables
def structure_table(rows):
    header = ["instance", "family", "n", "m", "k", "q", "largest", "|H|",
              "persist.", "gap bound"]
    out = []
    for r in sorted(rows, key=lambda r: (r.get("family", ""), _i(r, "n"))):
        gap = _f(r, "gap_bound_rel")
        out.append([
            r["name"], r.get("family", ""), _i(r, "n"), _i(r, "m"),
            _i(r, "k"), _i(r, "q"), _i(r, "largest_macroitem"),
            _i(r, "H_size") or "-",
            f"{_f(r, 'persistency'):.4f}" if r.get("persistency") else "-",
            f"{100 * gap:.2f}%" if gap == gap else "-",
        ])
    return header, out


def methods_table(rows):
    """Per instance and method: first capacity, each further one, break-even."""
    by_instance = defaultdict(dict)
    for r in rows:
        by_instance[r["instance"]][r["method"]] = r

    header = ["instance", "n", "method", "first (s)", "per extra (s)", "20 caps (s)",
              "break-even"]
    out = []
    for instance, methods in sorted(by_instance.items(), key=lambda kv: _i(list(kv[1].values())[0], "n")):
        path = methods.get("path")
        for name, r in methods.items():
            first, extra = _f(r, "seconds_first"), _f(r, "seconds_per_extra")
            total = _f(r, "seconds_total")
            breakeven = ""
            if path is not None and name != "path" and r.get("status") == "ok":
                # smallest number of capacities N for which the whole path is
                # cheaper: path_first <= first + (N-1)*extra
                pf = _f(path, "seconds_first")
                if extra > 0:
                    n_star = 1 + max(0.0, (pf - first) / extra)
                    breakeven = f"{max(1, int(n_star + 0.999)):d}"
            out.append([instance, _i(r, "n"), name,
                        f"{first:.3f}" if first == first else "-",
                        f"{extra:.4f}" if extra == extra else "-",
                        f"{total:.2f}" if total == total else "-",
                        breakeven])
    return header, out


def recommendation(rows):
    """Aggregate the comparison into the advice it supports."""
    by_instance = defaultdict(dict)
    for r in rows:
        by_instance[r["instance"]][r["method"]] = r

    wins_one, wins_many, n_inst = defaultdict(int), defaultdict(int), 0
    speed_one, speed_all = [], []
    for methods in by_instance.values():
        if "path" not in methods or methods["path"].get("status") != "ok":
            continue
        n_inst += 1
        ok = {k: v for k, v in methods.items() if v.get("status") == "ok"}
        best_one = min(ok, key=lambda k: _f(ok[k], "seconds_first"))
        best_many = min(ok, key=lambda k: _f(ok[k], "seconds_total"))
        wins_one[best_one] += 1
        wins_many[best_many] += 1
        solvers = {k: v for k, v in ok.items() if ":" in k}
        if solvers:
            fastest_solver_one = min(_f(v, "seconds_first") for v in solvers.values())
            fastest_solver_all = min(_f(v, "seconds_total") for v in solvers.values())
            if "newton" in ok:
                speed_one.append(fastest_solver_one / max(1e-9, _f(ok["newton"], "seconds_first")))
            speed_all.append(fastest_solver_all / max(1e-9, _f(methods["path"], "seconds_total")))

    header = ["question", "answer", "evidence"]
    out = []
    if speed_one:
        out.append(["one capacity", "Newton search on the weight price",
                    f"fastest on {wins_one.get('newton', 0)}/{n_inst} instances; "
                    f"median {sorted(speed_one)[len(speed_one) // 2]:.1f}x the fastest LP solver"])
    if speed_all:
        out.append(["the whole value function", "the canonical path",
                    f"fastest on {wins_many.get('path', 0)}/{n_inst} instances; "
                    f"median {sorted(speed_all)[len(speed_all) // 2]:.1f}x the fastest LP solver over 20 capacities"])
    out.append(["an LP solver is needed", "dual simplex, never barrier",
                "barrier cannot warm start, so its marginal cost per capacity is its full cost"])
    return header, out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default="experiments/results")
    ap.add_argument("--out", default="experiments/results/tables")
    args = ap.parse_args(argv)

    os.makedirs(args.out, exist_ok=True)
    # Every characteristics_*.csv and compare_*.csv in the results directory is
    # picked up, so a campaign split across several files -- by collection, or
    # because a long run was resumed -- needs no change here.
    structure = _load_all(args.results, "characteristics_*.csv")
    compare = _load_all(args.results, "compare_*.csv")

    made = []
    for name, (header, rows), caption, label in [
        ("structure", structure_table(structure),
         "Structure of the canonical macroitem sequence, and the split macroitem at each "
         "instance's own capacity", "tab:structure"),
        ("methods", methods_table(compare),
         "Cost of the first capacity, of each further capacity, and the number of capacities "
         "beyond which computing the whole canonical path is cheaper", "tab:methods"),
        ("recommendation", recommendation(compare),
         "What to use, and on what evidence", "tab:recommendation"),
    ]:
        if not rows:
            continue
        with open(os.path.join(args.out, f"{name}.md"), "w") as f:
            f.write(to_markdown(header, rows) + "\n")
        with open(os.path.join(args.out, f"{name}.tex"), "w") as f:
            f.write(to_latex(header, rows, caption, label) + "\n")
        made.append(f"{name} ({len(rows)} rows)")
    print("wrote: " + ", ".join(made) if made else "no input CSVs found")


if __name__ == "__main__":
    main()
