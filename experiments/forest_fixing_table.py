"""LaTeX rows of table ``tab:forests`` from the CSV of ``forest_fixing.py``.

Usage::

    python experiments/forest_fixing_table.py experiments/results/forest_fixing.csv \
        --out experiments/results/tables/forest_fixing.tex
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter, defaultdict

ORIENTATIONS = ("in", "out", "mixed")
PROFITS = ("unc", "signs", "ties")


def rows_from(path):
    with open(path) as f:
        return list(csv.DictReader(line for line in f if not line.startswith("#")))


def table(rows):
    by = defaultdict(list)
    attempts = Counter()
    for r in rows:
        attempts[(r["orient"], r["profits"], r["status"])] += 1
        if r["status"] == "ok":
            by[(r["orient"], r["profits"])].append(r)
    out = []
    for o in ORIENTATIONS:
        for p in PROFITS:
            R = by.get((o, p), [])
            if not R:
                continue
            N = len(R)
            mean = lambda k: sum(float(r[k]) for r in R) / N
            tot = lambda k: sum(int(r[k]) for r in R)
            Z, F = tot("Z"), tot("F")
            out.append(f"{o:5s} & {p:5s} & {mean('gap_lp_pct'):.2f} & {mean('gap_heur_pct'):.2f} & "
                       f"{100 * tot('fix_Z_solver') / Z:.1f} & {100 * tot('fix_Z_best') / Z:.1f} & "
                       f"{100 * tot('fix_F_solver') / F:.1f} & {100 * tot('fix_F_best') / F:.1f} & "
                       f"{100 * sum(int(r['k0']) > 1 for r in R) / N:.0f} & "
                       f"{100 * sum(r['critical'] == '0' for r in R) / N:.0f}\\\\  % N={N}")
    discarded = {k: v for k, v in attempts.items() if k[2] != "ok"}
    return out, discarded


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    lines, discarded = table(rows_from(a.csv))
    text = "\n".join(lines) + "\n"
    if a.out:
        with open(a.out, "w") as f:
            f.write(text)
    sys.stdout.write(text)
    for (o, p, s), v in sorted(discarded.items()):
        print(f"% discarded {o}:{p} {s}: {v}", file=sys.stderr)


if __name__ == "__main__":
    main()
