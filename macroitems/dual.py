"""Reduced costs from the dual optimal face.

At a nondegenerate capacity the optimal primal splits the items into three
regions -- ``F`` (fully selected, :math:`\\mathcal M_{h-1}`), ``H`` (the split
macroitem :math:`\\mathcal I_h`) and ``Z`` (null, the rest) -- and every optimal
dual has the same capacity price :math:`\\lambda_h`.  The dual optimal face is
not a point, so an item has a *range* of reduced costs, and what a
branch-and-bound can use is the best one over the whole face.

The companion note gives them as minimum cuts.  Writing
:math:`v_i = p_i - \\lambda_h w_i`, for a null item :math:`i \\in \\mathcal Z`
the cheapest way to force it into the solution is

.. math::
    s_i^\\star = \\min\\{\\lambda_h w(\\mathcal C) - p(\\mathcal C) :
      \\mathcal C \\subseteq \\mathcal Z \\text{ relatively closed},\\ i \\in \\mathcal C\\}
      = -\\max\\{v(\\mathcal C) : \\ldots\\},

one maximum closure on the subgraph induced by :math:`\\mathcal Z` with
:math:`i` forced in.  Symmetrically, for a full item :math:`i \\in \\mathcal F`
the cheapest way to force it out is

.. math::
    \\mu_i^\\star = \\min\\{p(T) - \\lambda_h w(T) :
      T \\subseteq \\mathcal F \\text{ co-closed},\\ i \\in T\\},

which is the same computation on :math:`\\mathcal F` with the arcs reversed
(a co-closed set of a graph is a closed set of its reverse) and the values
negated.

These are at least as large as the closed-form *canonical* reduced costs
:math:`w_i|\\lambda_r - \\lambda_h|` of the canonical dual solution, which
without precedences reduce to the classical knapsack reduced costs
:math:`w_i|p_i/w_i - \\lambda|`.  The gain over the canonical values is
precisely what the precedence structure buys, and measuring it is one of the
experiments of the paper.

Use in enumeration: if :math:`z_{LP} - s_i^\\star \\le` incumbent, then
:math:`x_i = 0` in every solution better than the incumbent; symmetrically
with :math:`\\mu_i^\\star` for :math:`x_i = 1`.
"""
from __future__ import annotations

import dataclasses
import time
from typing import Iterable, Optional

import numpy as np

from .closure import ClosureSolver
from .instance import Instance
from .path import LPSolution, MacroitemPath

__all__ = ["best_reduced_costs", "canonical_reduced_costs", "fixable_items", "ReducedCosts"]


@dataclasses.dataclass
class ReducedCosts:
    """Reduced costs of the items of one region.

    ``value[i]`` is the best (largest) reduced cost of item ``i`` over the dual
    optimal face, i.e. a valid lower bound on the loss incurred by flipping
    item ``i``; it is ``nan`` for items not asked for, and ``0`` on the split
    macroitem, whose items are free to move.
    """
    value: np.ndarray            # shape (n,), nan where not computed
    region: np.ndarray           # 'F', 'H', 'Z' or '' per item, as a numpy '<U1' array
    n_maxflow: int
    seconds: float

    def gain_over(self, canonical: np.ndarray) -> np.ndarray:
        """How much the face-wide reduced cost beats the canonical one."""
        with np.errstate(invalid="ignore"):
            return self.value - canonical


def best_reduced_costs(inst: Instance, sol: LPSolution,
                       items: Optional[Iterable[int]] = None,
                       backend: Optional[str] = None,
                       max_items: Optional[int] = None,
                       rng: Optional[np.random.Generator] = None) -> ReducedCosts:
    """Best reduced costs over the dual optimal face, by one minimum cut per item.

    ``items`` restricts the computation (by default every item of ``F`` and
    ``Z``); ``max_items`` samples that many items at random instead, which is
    what makes the computation affordable on the large instances.  Items of the
    split macroitem get reduced cost 0 by definition.
    """
    t0 = time.perf_counter()
    n = inst.n
    lam = sol.lam
    v = inst.p - lam * inst.w

    region = np.full(n, "", dtype="<U1")
    region[sol.F] = "F"
    region[sol.H] = "H"
    region[sol.Z] = "Z"

    out = np.full(n, np.nan)
    out[sol.H] = 0.0

    if items is None:
        wanted = np.flatnonzero(sol.F | sol.Z)
    else:
        wanted = np.asarray(list(items), dtype=np.int64)
    if max_items is not None and wanted.size > max_items:
        rng = rng or np.random.default_rng(0)
        wanted = np.sort(rng.choice(wanted, size=max_items, replace=False))

    n_mf = 0
    for mask, kind in ((sol.Z, "Z"), (sol.F, "F")):
        targets = wanted[mask[wanted]]
        if targets.size == 0:
            continue
        nodes = np.flatnonzero(mask)
        sub, _ = inst.induced(nodes)
        position = -np.ones(n, dtype=np.int64)
        position[nodes] = np.arange(nodes.size)

        # Both are minimisations turned into the maximisation that a maximum
        # closure performs, so both negate the result -- but over different
        # objectives:
        #   Z:  s*  = min{-v(C)} = -max{ v(C) : C closed in Z, i in C}
        #   F:  mu* = min{ v(T)} = -max{-v(T) : T closed in reverse(F), i in T}
        v_sub = sub.p - lam * sub.w
        if kind == "Z":
            work = sub
            values = v_sub
        else:
            # co-closed sets of F are the closed sets of the reversed graph
            work = Instance(sub.p, sub.w, sub.arcs[:, ::-1].copy() if sub.m else sub.arcs,
                            name=sub.name)
            values = -v_sub
        sign = -1.0

        solver = ClosureSolver(work, backend=backend)
        force = np.zeros(nodes.size, dtype=bool)
        for i in targets:
            force[:] = False
            force[position[i]] = True
            res = solver.solve(values, tie="max", force_in=force)
            n_mf += 1
            out[i] = sign * res.value
        del solver

    # numerical noise can produce tiny negative values; they are zero by theory
    tiny = 1e-9 * max(1.0, float(np.abs(inst.p).max(initial=1.0)))
    near_zero = np.isfinite(out) & (out > -tiny) & (out < 0)
    out[near_zero] = 0.0
    return ReducedCosts(out, region, n_mf, time.perf_counter() - t0)


def canonical_reduced_costs(inst: Instance, path: MacroitemPath, h: int) -> np.ndarray:
    """Closed-form reduced costs of the canonical dual solution.

    Item ``i`` of macroitem ``I_r`` gets ``w_i * |lambda_r - lambda_h|``, which
    is 0 on the split macroitem.  Without precedences these are the knapsack
    reduced costs ``w_i * |p_i/w_i - lambda_h|``.
    """
    level = path.level_of_item(inst.n)
    lam_h = path.ratios[h - 1]
    lam_r = np.where(level > 0, path.ratios[np.maximum(level - 1, 0)], 0.0)
    return inst.w * np.abs(lam_r - lam_h)


def fixable_items(sol: LPSolution, reduced: np.ndarray, incumbent: float,
                  z_lp: Optional[float] = None) -> dict:
    """Items fixed by reduced-cost reasoning, given an incumbent integer value.

    An item of ``Z`` whose reduced cost exceeds the gap ``z_LP - incumbent``
    cannot be selected in any solution better than the incumbent; an item of
    ``F`` whose reduced cost exceeds the gap cannot be dropped.  Returns the
    two index arrays and the counts.
    """
    z_lp = sol.value if z_lp is None else z_lp
    gap = z_lp - incumbent
    with np.errstate(invalid="ignore"):
        beats = np.nan_to_num(reduced, nan=-np.inf) > gap
    fix_zero = np.flatnonzero(beats & sol.Z)
    fix_one = np.flatnonzero(beats & sol.F)
    return {
        "gap": gap,
        "fix_zero": fix_zero,
        "fix_one": fix_one,
        "n_fixed": int(fix_zero.size + fix_one.size),
        "share_fixed": float(fix_zero.size + fix_one.size) / len(reduced),
    }
