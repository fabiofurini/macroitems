"""Figures for the report and the paper.

``value-function``
    The piecewise-linear value function :math:`z(c)` of one instance, with the
    cumulative macroitem points :math:`(w(\\mathcal M_r), p(\\mathcal M_r))`
    marked, above a bar of the macroitems shaded by ratio.  This is the
    picture the whole paper is about: the relaxation is a knapsack LP on
    macroitems, so its value function is the concave interpolation of those
    points.

``persistency``
    Persistency :math:`1 - |\\mathcal H|/n` and the integrality-gap bound
    :math:`\\theta\\, p(\\mathcal I_h) / z(c)` of Proposition 5.2, as functions
    of the capacity.  Together they say how much of the answer the relaxation
    settles, and how far it can be from the integer optimum, at each capacity.

Usage::

    python experiments/make_figures.py <instance>... --out experiments/results/figures
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from macroitems import canonical_path, solution_from_path  # noqa: E402
from macroitems.formats import read_any, read_minelib_upit  # noqa: E402


def load(path: str):
    reader = read_minelib_upit if not os.path.splitext(path)[1] else read_any
    return reader(path)


def value_function_figure(inst, path, scale, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import cm, colors

    q = path.q
    W = path.W[: q + 1] / scale
    P = path.P[: q + 1] / scale
    ratios = path.ratios[:q]      # invariant under the rescaling, unlike P and W

    fig, (ax, bar) = plt.subplots(
        2, 1, figsize=(7.0, 5.0), sharex=True,
        gridspec_kw={"height_ratios": [4, 1], "hspace": 0.08})

    ax.plot(W, P, "-", color="0.2", linewidth=1.6, zorder=2)
    ax.plot(W, P, ".", color="0.2", markersize=5, zorder=3)
    ax.set_ylabel(r"$z(c)$")
    ax.set_title(f"{inst.name}: $n={inst.n}$, $m={inst.m}$, "
                 f"{path.k} macroitems ({q} with positive ratio)", fontsize=10)
    ax.grid(alpha=0.25, linewidth=0.5)
    ax.margins(x=0.01)

    # the macroitems as a bar along the weight axis, shaded by ratio
    norm = colors.Normalize(vmin=float(ratios.min()), vmax=float(ratios.max())) if q else None
    cmap = cm.get_cmap("viridis") if hasattr(cm, "get_cmap") else plt.get_cmap("viridis")
    for r in range(q):
        bar.barh(0, W[r + 1] - W[r], left=W[r], height=1.0,
                 color=cmap(norm(float(ratios[r]))), edgecolor="none")
    if q:
        bar.barh(0, W[1] - W[0], left=W[0], height=1.0, color="none",
                 edgecolor="crimson", linewidth=1.4)      # the first macroitem
    bar.set_yticks([])
    bar.set_xlabel("capacity $c$ (cumulative weight)")
    bar.set_ylim(-0.5, 0.5)
    bar.margins(x=0.01)
    if q:
        sm = cm.ScalarMappable(norm=norm, cmap=cmap)
        cb = fig.colorbar(sm, ax=[ax, bar], fraction=0.03, pad=0.02)
        cb.set_label(r"macroitem ratio $\lambda_r$", fontsize=9)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def persistency_figure(inst, path, scale, out_path, n_points=200):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    q = path.q
    top = float(path.W[q])
    grid = np.linspace(top / (n_points + 1), top * n_points / (n_points + 1), n_points)
    persistency, gap = [], []
    for c in grid:
        sol = solution_from_path(inst, path, c)
        h_size = int(sol.H.sum())
        persistency.append(1.0 - h_size / inst.n)
        z = sol.value
        gap.append(sol.theta * float(inst.p[sol.H].sum()) / z if z > 0 and h_size else 0.0)

    fig, ax = plt.subplots(figsize=(7.0, 3.2))
    ax.plot(grid / top, persistency, color="0.2", linewidth=1.4,
            label=r"persistency $1-|\mathcal{H}|/n$")
    ax.plot(grid / top, gap, color="crimson", linewidth=1.4, linestyle="--",
            label=r"gap bound $\theta\,p(\mathcal{I}_h)/z(c)$")
    ax.set_xlabel(r"capacity, as a fraction of $w(\mathcal{M}_q)$")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(alpha=0.25, linewidth=0.5)
    ax.legend(fontsize=9, loc="center right")
    ax.set_title(f"{inst.name}", fontsize=10)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("instances", nargs="+")
    ap.add_argument("--out", default="experiments/results/figures")
    args = ap.parse_args(argv)

    os.makedirs(args.out, exist_ok=True)
    for spec in args.instances:
        inst = load(spec)
        work, scale = inst.scaled_to_integers()
        path = canonical_path(work)
        for kind, fn in (("value-function", value_function_figure),
                         ("persistency", persistency_figure)):
            target = os.path.join(args.out, f"{inst.name}_{kind}.png")
            fn(work if kind == "persistency" else inst, path, scale, target)
            print(f"{inst.name:18s} {kind:16s} -> {target}", flush=True)


if __name__ == "__main__":
    main()
