"""Lawler's binary search for the maximum-ratio closure (Lawler 1978, Section 7).

The first macroitem is the closure of maximum profit-to-weight ratio
(Corollary 3.1 of the paper).  Lawler computes it, in the scheduling language
of composite jobs, by a binary search on the ratio with one minimum cut per
trial value: at a trial price :math:`\\lambda`,

* :math:`u(\\lambda) > 0` means some closure beats the ratio, so the maximum
  ratio is larger than :math:`\\lambda`;
* :math:`u(\\lambda) < 0` is impossible for the empty closure, so the useful
  test is :math:`u(\\lambda) = 0`, which means :math:`\\lambda` is at or above
  the maximum ratio;

and the search converges once the bracket is finer than the smallest possible
gap between two distinct ratios.  On integer data with weights at most
:math:`w^\\star`, two distinct ratios :math:`p_1/w_1 \\ne p_2/w_2` differ by at
least :math:`1/(w^\\star)^2`, since
:math:`|p_1w_2 - p_2w_1| \\ge 1`; that is the precision the search needs.

This module exists for the method comparison of the paper: it computes the
same first macroitem as :func:`~macroitems.path.canonical_path`, at the cost
of :math:`O(\\log(p^\\star (w^\\star)^2))` maximum closures instead of a
constant number, and it does not contract the graph between iterations.  Use
:func:`~macroitems.path.canonical_path` for real work.
"""
from __future__ import annotations

import dataclasses
import time
from fractions import Fraction
from typing import Optional

import numpy as np

from .closure import ClosureSolver
from .instance import Instance

__all__ = ["first_macroitem_lawler", "LawlerResult"]


@dataclasses.dataclass
class LawlerResult:
    macroitem: np.ndarray       # the first canonical macroitem, sorted indices
    ratio: float                # its profit-to-weight ratio, = lambda_1
    ratio_exact: Optional[Fraction]
    n_maxflow: int
    seconds: float
    iterations: int


def first_macroitem_lawler(inst: Instance, backend: Optional[str] = None,
                           tol: Optional[float] = None) -> LawlerResult:
    """The maximum-ratio closure by binary search on the ratio.

    Returns the inclusion-wise maximal maximum-ratio closure, which is the
    first macroitem of the canonical sequence.  On integer data the search is
    exact: it stops when the bracket is narrower than :math:`1/(w^\\star)^2`
    and the ratio is then recovered exactly from the closure itself.
    """
    t0 = time.perf_counter()
    integral = bool(np.all(inst.p == np.rint(inst.p)) and np.all(inst.w == np.rint(inst.w)))
    p_star = float(np.abs(inst.p).max(initial=1.0))
    w_star = float(inst.w.max(initial=1.0))
    if tol is None:
        tol = 1.0 / (w_star * w_star) if integral else 1e-12 * max(1.0, p_star / inst.w.min())

    solver = ClosureSolver(inst, backend=backend)
    n_mf = 0

    # Bracket: at lambda = 0 the maximum ratio is positive iff some closure has
    # positive profit; above p_star / min(w) no closure can be profitable.
    lo, hi = 0.0, p_star / float(inst.w.min()) + 1.0
    best = None

    res = solver.solve(inst.p - lo * inst.w, tie="max")
    n_mf += 1
    if res.value <= 0:
        # no closure of positive ratio: the maximum ratio is at most 0, and the
        # search below would not move; fall back to the whole-instance ratio.
        seconds = time.perf_counter() - t0
        allitems = np.arange(inst.n)
        ratio = float(inst.p.sum() / inst.w.sum())
        exact = (Fraction(int(round(inst.p.sum())), int(round(inst.w.sum())))
                 if integral else None)
        return LawlerResult(allitems, ratio, exact, n_mf, seconds, 0)

    iterations = 0
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        res = solver.solve(inst.p - mid * inst.w, tie="max")
        n_mf += 1
        iterations += 1
        if res.value > 0:
            lo = mid                 # some closure beats mid: ratio is above it
            best = res.closure
        else:
            hi = mid                 # nothing beats mid: ratio is at or below it
    # At the lower end of the bracket the maximal optimal closure is the
    # maximum-ratio one; recompute it there to get the maximal tie convention.
    res = solver.solve(inst.p - lo * inst.w, tie="max")
    n_mf += 1
    macro = res.closure if res.closure.size else (best if best is not None else np.arange(inst.n))

    p_c, w_c = float(inst.p[macro].sum()), float(inst.w[macro].sum())
    exact = Fraction(int(round(p_c)), int(round(w_c))) if integral else None
    ratio = float(exact) if exact is not None else p_c / w_c
    return LawlerResult(np.sort(macro), ratio, exact, n_mf, time.perf_counter() - t0, iterations)
