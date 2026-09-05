"""Exact reference implementations by enumeration of every closure.

Everything here is computed in rational arithmetic (:class:`fractions.Fraction`)
from the definitions in the paper, independently of :mod:`macroitems`, so that a
disagreement is evidence against the library and not against a shared shortcut.
Only usable for tiny instances: the number of closures can be 2**n.
"""
from __future__ import annotations

from fractions import Fraction as Fr
from typing import Iterable, List, Sequence, Tuple

import numpy as np

Closure = frozenset


# --------------------------------------------------------------- enumeration
def all_closures(n: int, arcs: Sequence[Tuple[int, int]]) -> List[Closure]:
    """Every subset S with no arc leaving it: i in S and (i, j) in A imply j in S."""
    out = []
    for bits in range(1 << n):
        S = frozenset(i for i in range(n) if bits >> i & 1)
        if all(j in S for (i, j) in arcs if i in S):
            out.append(S)
    return out


# ----------------------------------------------------- parametric maximum closure
def u_and_extremes(closures: Iterable[Closure], p, w, lam: Fr):
    """``u(lam) = max_C v(C)`` with ``v_i = p_i - lam w_i``, plus the extremal
    optimal closures.

    Proposition 3.1: the optimal closures form a lattice, so the intersection of
    all of them is the unique minimal one and their union the unique maximal one.
    Returns ``(u, minimal, maximal, n_optimal)``.
    """
    closures = list(closures)
    best, opt = None, []
    for S in closures:
        v = sum((Fr(int(p[i])) - lam * int(w[i]) for i in S), Fr(0))
        if best is None or v > best:
            best, opt = v, [S]
        elif v == best:
            opt.append(S)
    lo = frozenset(set.intersection(*map(set, opt)))
    hi = frozenset(set.union(*map(set, opt)))
    assert lo in closures and hi in closures, "optimal closures are not a lattice"
    return best, lo, hi, len(opt)


def brute_canonical(n: int, p, w, arcs, closures=None) -> List[Tuple[Closure, Fr]]:
    """The canonical macroitem sequence straight from the definition.

    From the current prefix M_{r-1}, lambda_r is the largest ratio
    p(C \\ M_{r-1}) / w(C \\ M_{r-1}) over closures C strictly containing it
    (Theorem 3.1), and M_r is the union of all closures attaining it -- the
    maximal-tie convention.  Returns ``[(I_r, lambda_r), ...]``.
    """
    closures = list(closures) if closures is not None else all_closures(n, arcs)
    seq: List[Tuple[Closure, Fr]] = []
    A: Closure = frozenset()
    full = frozenset(range(n))
    while A != full:
        cands = [(Fr(int(sum(p[i] for i in C - A)), int(sum(w[i] for i in C - A))), C)
                 for C in closures if C > A]
        lam = max(r for r, _ in cands)
        U = frozenset(set.union(*[set(C) for r, C in cands if r == lam]))
        seq.append((U - A, lam))
        A = U
    return seq


# --------------------------------------------------------------- the LP itself
def _cross(o, a, b):
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def value_hull(closures: Iterable[Closure], p, w) -> List[Tuple[Fr, Fr]]:
    """Vertices of the upper concave hull of ``{(w(C), p(C)) : C closure}``,
    truncated at the maximum-profit vertex.

    The LP polytope ``{x in [0,1]^n : x_i <= x_j, w.x <= c}`` has the closure
    incidence vectors as the vertices of its precedence part, so ``z(c)`` is the
    concave envelope of those points with free disposal of capacity; its
    vertices must be the cumulative points ``(w(M_r), p(M_r))``, r = 0..q.
    """
    best = {}
    for S in closures:
        wv = Fr(int(sum(w[i] for i in S)))
        pv = Fr(int(sum(p[i] for i in S)))
        if wv not in best or pv > best[wv]:
            best[wv] = pv
    hull: List[Tuple[Fr, Fr]] = []
    for q in sorted(best.items()):
        while len(hull) >= 2 and _cross(hull[-2], hull[-1], q) >= 0:
            hull.pop()
        hull.append(q)
    top = max(range(len(hull)), key=lambda r: hull[r][1])
    return hull[:top + 1]


def brute_z(hull: Sequence[Tuple[Fr, Fr]], c) -> Fr:
    """z(c) read off the concave hull; constant beyond the maximum-profit vertex."""
    c = Fr(c)
    if c <= 0 or len(hull) == 1:
        return Fr(0)
    if c >= hull[-1][0]:
        return hull[-1][1]
    for (w0, p0), (w1, p1) in zip(hull, hull[1:]):
        if w0 < c <= w1:
            return p0 + (c - w0) * (p1 - p0) / (w1 - w0)
    raise AssertionError("capacity outside the hull")


def lp_points(closures: Sequence[Closure], p, w, c) -> List[Tuple[Fr, dict]]:
    """Every vertex of the LP at capacity c, exactly.

    A vertex of ``{x in [0,1]^n : x_i <= x_j, w.x <= c}`` is either a closure
    incidence vector of weight at most c, or lies on an edge of the closure
    polytope cut by the capacity hyperplane, i.e. ``1_{C1} + theta 1_{C2 \\ C1}``
    for closures ``C1 subset C2`` with ``w(C1) < c < w(C2)``.  Every point of
    that (superset) list is feasible, so maxima and minima of any coordinate
    over the ones attaining ``z(c)`` are exactly those over the optimal face.
    """
    c = Fr(c)
    P = {S: Fr(int(sum(p[i] for i in S))) for S in closures}
    W = {S: Fr(int(sum(w[i] for i in S))) for S in closures}
    out = []
    for S in closures:
        if W[S] <= c:
            out.append((P[S], {i: Fr(1) for i in S}))
    light = [S for S in closures if W[S] < c]
    heavy = [S for S in closures if W[S] > c]
    for C1 in light:
        for C2 in heavy:
            if not C1 < C2:
                continue
            th = (c - W[C1]) / (W[C2] - W[C1])
            x = {i: Fr(1) for i in C1}
            x.update({i: th for i in C2 - C1})
            out.append((P[C1] + th * (P[C2] - P[C1]), x))
    return out


def optimal_face_range(closures, p, w, c, n):
    """``(z, lo, hi)``: the exact optimal value and, per item, the range of x_i
    over the whole optimal face of the LP at capacity c."""
    pts = lp_points(closures, p, w, c)
    z = max(v for v, _ in pts)
    opt = [x for v, x in pts if v == z]
    lo = [min(x.get(i, Fr(0)) for x in opt) for i in range(n)]
    hi = [max(x.get(i, Fr(0)) for x in opt) for i in range(n)]
    return z, lo, hi
