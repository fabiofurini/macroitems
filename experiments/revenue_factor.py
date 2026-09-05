"""Weight parameterization against revenue-factor parameterization.

Mine planners usually generate nested pits by scaling the *revenue* of every
block while holding its cost fixed: writing the block value as
:math:`p_i = r_i - k_i` with revenue :math:`r_i \\ge 0` and cost
:math:`k_i \\ge 0`, the values :math:`f r_i - k_i` are parameterized by a
revenue factor :math:`f`.  The paper's parameterization instead prices the
*weight*, using :math:`p_i - \\lambda w_i`.

Both families are nested, by the same lattice argument, but they are in
general **different** families, and only the weight one solves the LP
relaxation of a tonnage-constrained problem.  They coincide exactly when the
weights are proportional to the revenues.  This experiment measures how far
apart they are on real deposits.

For each instance it computes the canonical closures :math:`M_r` (weight
parameterization) and the revenue-factor pits over a grid of :math:`f`, and
reports for every revenue-factor pit:

* whether it coincides with some :math:`M_r`;
* the relative symmetric difference to the *closest-by-weight* :math:`M_r`,
  which is the fair comparison: a planner would pick the nested pit of the
  tonnage they can handle.

The split of a block's value into a revenue and a cost has to come from
somewhere, and this is the practical obstacle: MineLib's UPIT and CPIT files
give **one** value per block, from which the split is not recoverable without
inventing an economic model.  Only the PCPSP formulation lists a value per
destination, and that is what :func:`revenue_and_cost` reads when it is
present.  Synthetic instances from :func:`macroitems.layered_grid` carry their
revenue and cost by construction, so the comparison runs on them unconditionally.

Usage::

    python experiments/revenue_factor.py grid:20x20x8:5:1 <minelib-stem>... [--factors 20]
"""
from __future__ import annotations

import argparse
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from macroitems import canonical_path  # noqa: E402
from macroitems.closure import max_closure  # noqa: E402
from macroitems.formats import _minelib_files, read_minelib_upit  # noqa: E402
from macroitems.instance import Instance  # noqa: E402


def revenue_and_cost(stem: str, inst: Instance) -> tuple[np.ndarray, np.ndarray]:
    """Split each block's value into a revenue and a cost.

    Synthetic instances carry both.  For MineLib, the per-destination objective
    of a PCPSP file is read: destination 0 sends the block to the plant and
    destination 1 to the waste dump, so

        value_plant = revenue - processing cost - mining cost,
        value_waste =         - mining cost,

    whence ``cost = -value_waste`` and ``revenue = value_plant - value_waste``
    (revenue net of processing).  A block whose plant value is missing is pure
    waste: revenue 0.  With a single destination the split is not recoverable
    and the function raises, rather than inventing one.
    """
    if "revenue" in inst.extra and "cost" in inst.extra:
        # synthetic instances know their own revenue and cost by construction
        return np.asarray(inst.extra["revenue"]), np.asarray(inst.extra["cost"])
    files = _minelib_files(stem)
    # Only the PCPSP formulation lists a value per destination; UPIT and CPIT
    # carry a single value per block, from which revenue and cost cannot be
    # separated without inventing an economic model.
    source = ".pcpsp" if ".pcpsp" in files else ".cpit"
    if source not in files:
        raise ValueError(f"{stem}: no .pcpsp or .cpit file, revenue and cost "
                         "cannot be reconstructed")
    values: dict[int, list[float]] = {}
    in_obj = False
    for line in files[source]:
        s = line.strip()
        if not s or s.startswith("%"):
            continue
        upper = s.upper()
        if upper.startswith("OBJECTIVE_FUNCTION"):
            in_obj = True
            continue
        if in_obj:
            if ":" in s and not s[0].isdigit():
                break
            parts = s.split()
            if len(parts) < 2 or not parts[0].lstrip("-").isdigit():
                break
            values[int(parts[0])] = [float(v) for v in parts[1:]]
    if not values:
        raise ValueError(f"{stem}: no objective section in the {source} file")
    width = max(len(v) for v in values.values())
    if width < 2:
        raise ValueError(
            f"{stem}: the {source} objective gives one value per block, so revenue and "
            "cost cannot be separated; the per-destination values needed for a "
            "revenue factor are in the .pcpsp file of the instance")

    revenue = np.zeros(inst.n)
    cost = np.zeros(inst.n)
    for b, vals in values.items():
        plant = vals[0]
        waste = vals[1] if len(vals) > 1 else vals[0]
        cost[b] = -waste
        revenue[b] = max(0.0, plant - waste)
    return revenue, cost


def pit_for_factor(inst: Instance, revenue: np.ndarray, cost: np.ndarray, f: float):
    """The maximal optimal closure for block values f*revenue - cost."""
    return max_closure(inst, f * revenue - cost, tie="max").mask


def load(stem: str) -> Instance:
    """A MineLib stem, an instance file, or ``grid:<nx>x<ny>x<nz>:<cone>:<seed>``."""
    if stem.startswith("grid:"):
        from macroitems import layered_grid
        _, size, cone, seed = stem.split(":")
        nx, ny, nz = (int(v) for v in size.split("x"))
        return layered_grid(nx, ny, nz, cone=int(cone), seed=int(seed))
    if os.path.splitext(stem)[1]:
        from macroitems.formats import read_any
        return read_any(stem)
    return read_minelib_upit(stem)


def compare(stem: str, n_factors: int) -> list[dict]:
    inst = load(stem)
    revenue, cost = revenue_and_cost(stem, inst)

    work, scale = inst.scaled_to_integers()
    path = canonical_path(work)
    weight_pits = [path.closure_mask(inst.n, r) for r in range(1, path.q + 1)]
    weight_tonnage = np.array([inst.w[m].sum() for m in weight_pits]) if weight_pits else np.zeros(0)

    rows = []
    for f in np.linspace(1.0 / n_factors, 1.0, n_factors):
        pit = pit_for_factor(inst, revenue, cost, f)
        tonnage = float(inst.w[pit].sum())
        if not weight_pits or pit.sum() == 0:
            rows.append({"instance": inst.name, "factor": round(float(f), 4),
                         "pit_size": int(pit.sum()), "pit_tonnage": tonnage,
                         "coincides_with_Mr": "", "closest_Mr": "",
                         "relative_symmetric_difference": ""})
            continue
        exact = [r for r, m in enumerate(weight_pits, start=1) if np.array_equal(m, pit)]
        closest = int(np.argmin(np.abs(weight_tonnage - tonnage)))
        diff = np.logical_xor(pit, weight_pits[closest]).sum()
        union = np.logical_or(pit, weight_pits[closest]).sum()
        rows.append({
            "instance": inst.name,
            "factor": round(float(f), 4),
            "pit_size": int(pit.sum()),
            "pit_tonnage": tonnage,
            "coincides_with_Mr": exact[0] if exact else "",
            "closest_Mr": closest + 1,
            "relative_symmetric_difference": round(float(diff) / max(1, union), 6),
        })
    n_exact = sum(1 for r in rows if r["coincides_with_Mr"] != "")
    worst = max((r["relative_symmetric_difference"] for r in rows
                 if r["relative_symmetric_difference"] != ""), default=0.0)
    print(f"{inst.name:18s} k={path.k:5d} q={path.q:5d}  "
          f"{n_exact:2d}/{len(rows)} revenue-factor pits coincide with a canonical one; "
          f"worst relative symmetric difference {worst:.3f}", flush=True)
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stems", nargs="+",
                    help="MineLib stems, instance files, or grid:<nx>x<ny>x<nz>:<cone>:<seed>")
    ap.add_argument("--factors", type=int, default=20)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    rows = []
    for stem in args.stems:
        try:
            rows += compare(stem, args.factors)
        except ValueError as exc:
            print(f"{os.path.basename(stem):18s} skipped: {exc}", flush=True)
    if args.out and rows:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"\n{len(rows)} rows -> {args.out}")


if __name__ == "__main__":
    main()
