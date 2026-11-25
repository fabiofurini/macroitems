"""Precedence forests: linear-time reduced costs and split structure.

When the underlying undirected graph of the precedence graph is a forest (arcs
of any orientation), every closure problem of the form

.. math::
    f(v) = \\min\\{ g(C) : C \\text{ closed},\\ v \\in C \\},
    \\qquad g(C) \\ge 0 \\text{ for every closed } C,

is solved for all ``v`` at once by one rerooting recursion (Proposition
``prop:forestrec``).  With it, on forests:

* the strongest reduced costs :math:`s^\\star_i` (``i`` in ``Z``) and
  :math:`\\mu^\\star_i` (``i`` in ``F``) of Proposition ``prop:reduced`` cost
  :math:`O(n)` in total, instead of one minimum cut per item
  (Corollary ``cor:forestrc``);
* the inseparability classes of ``H`` are the components left after deleting
  the separable arcs, i.e. the arcs ``(x, y)`` with :math:`D(y \\mid x) = 0`.
  The order is reachability in the contracted forest, and the critical class
  follows in :math:`O(|H|)` (Proposition ``prop:forestclasses``).

All values are exact rationals on integer data: :math:`\\lambda_h = p(H)/w(H)`.

The algorithms follow the scripts of M. Locatelli (24 November 2026), rewritten
against the data structures of this package.
"""
from __future__ import annotations

from fractions import Fraction
from math import factorial
from typing import Dict, List, Sequence, Tuple

import numpy as np

from .classes import SplitStructure
from .instance import Instance
from .path import LPSolution

__all__ = ["is_forest", "forest_min_closed", "forest_reduced_costs", "forest_split_structure",
           "hook_length_count"]

Arc = Tuple[int, int]


def is_forest(inst: Instance) -> bool:
    """Whether the underlying undirected graph has no cycle (parallel arcs count as a cycle)."""
    parent = list(range(inst.n))

    def find(v: int) -> int:
        while parent[v] != v:
            parent[v] = parent[parent[v]]
            v = parent[v]
        return v

    for u, v in inst.arcs.tolist():
        ru, rv = find(u), find(v)
        if ru == rv:
            return False
        parent[ru] = rv
    return True


def _critical_price(inst: Instance, sol: LPSolution):
    """lambda_h as an exact Fraction on integer data, as a float otherwise."""
    H = sol.H
    if inst.is_integral():
        return Fraction(int(round(inst.p[H].sum())), int(round(inst.w[H].sum())))
    return float(inst.p[H].sum() / inst.w[H].sum())


def _values(inst: Instance, lam) -> list:
    if isinstance(lam, Fraction):
        return [Fraction(int(round(p))) - lam * int(round(w)) for p, w in zip(inst.p, inst.w)]
    return list(inst.p - lam * inst.w)


# ------------------------------------------------------------ the recursion
def forest_min_closed(nodes: Sequence[int], arcs: Sequence[Arc], g: Dict[int, object]):
    """Proposition ``prop:forestrec``.

    ``arcs`` are pairs ``(u, v)`` meaning *u in C implies v in C*; only those
    with both ends in ``nodes`` are used, and they must form a forest.  ``g``
    must satisfy ``g(C) >= 0`` on every closed set.  Returns ``(f, D)``:
    ``f[v]`` is the minimum of ``g(C)`` over closed ``C`` containing ``v``, and
    ``D[(u, x)]`` is :math:`D(x \\mid u)` for every ordered pair of adjacent
    nodes.  Two passes over every tree, so :math:`O(|nodes|)` time.
    """
    S = set(int(v) for v in nodes)
    adj: Dict[int, List[Tuple[int, bool]]] = {v: [] for v in S}
    for u, v in arcs:
        if u in S and v in S:
            adj[u].append((v, True))     # u in C forces v
            adj[v].append((u, False))    # v in C does not force u

    def h(forced: bool, D):
        return D if forced else min(D, 0)

    f: Dict[int, object] = {}
    D: Dict[Tuple[int, int], object] = {}
    seen: set = set()
    for r in sorted(S):
        if r in seen:
            continue
        order, parent, stack = [], {r: None}, [r]
        seen.add(r)
        while stack:                       # preorder: a parent precedes its children
            v = stack.pop()
            order.append(v)
            for u, _ in adj[v]:
                if u not in seen:
                    seen.add(u)
                    parent[u] = v
                    stack.append(u)
        down: Dict[int, object] = {}       # down[v] = D(v | parent(v))
        for v in reversed(order):
            s = g[v]
            for u, forced in adj[v]:
                if u != parent[v]:
                    s += h(forced, down[u])
            down[v] = s
            if parent[v] is not None:
                D[(parent[v], v)] = s
        up: Dict[int, object] = {}         # up[v] = D(parent(v) | v)
        for v in order:
            tot = g[v]
            for u, forced in adj[v]:
                tot += h(forced, up[v] if u == parent[v] else down[u])
            f[v] = tot
            for u, forced in adj[v]:
                if u != parent[v]:
                    up[u] = tot - h(forced, down[u])
                    D[(u, v)] = up[u]
    return f, D


# ------------------------------------------------------------ reduced costs
def forest_reduced_costs(inst: Instance, sol: LPSolution) -> np.ndarray:
    """Strongest reduced costs over the dual optimal face (Corollary ``cor:forestrc``).

    Same output convention as :func:`macroitems.dual.best_reduced_costs`:
    :math:`s^\\star_i` on ``Z``, :math:`\\mu^\\star_i` on ``F``, 0 on ``H``.
    Exact on integer data (the returned floats are rounded rationals).
    """
    if not is_forest(inst):
        raise ValueError("the precedence graph is not a forest")
    lam = _critical_price(inst, sol)
    b = _values(inst, lam)
    arcs = [tuple(a) for a in inst.arcs.tolist()]
    Z = np.flatnonzero(sol.Z).tolist()
    F = np.flatnonzero(sol.F).tolist()
    out = np.full(inst.n, np.nan)
    out[sol.H] = 0.0
    fZ, _ = forest_min_closed(Z, arcs, {i: -b[i] for i in Z})
    fF, _ = forest_min_closed(F, [(v, u) for u, v in arcs], {i: b[i] for i in F})
    for i, val in list(fZ.items()) + list(fF.items()):
        out[i] = float(val)
    return out


# ------------------------------------------------------------ split structure
def forest_split_structure(inst: Instance, sol: LPSolution, c: float) -> SplitStructure:
    """Proposition ``prop:forestclasses``(a)-(c) in :math:`O(|H|)` time.

    Returns the same :class:`~macroitems.classes.SplitStructure` as
    :func:`macroitems.classes.split_structure`, without maximum flows
    (``n_maxflow = 0``).
    """
    if not is_forest(inst):
        raise ValueError("the precedence graph is not a forest")
    lam = _critical_price(inst, sol)
    b = _values(inst, lam)
    Hn = np.flatnonzero(sol.H).tolist()
    Hs = set(Hn)
    AH = [(u, v) for u, v in inst.arcs.tolist() if u in Hs and v in Hs]
    _, D = forest_min_closed(Hn, AH, {i: -b[i] for i in Hn})
    separable = [(x, y) for x, y in AH if D[(x, y)] == 0]
    keep = [(x, y) for x, y in AH if D[(x, y)] != 0]
    parent = {v: v for v in Hn}

    def find(v: int) -> int:
        while parent[v] != v:
            parent[v] = parent[parent[v]]
            v = parent[v]
        return v

    for x, y in keep:
        parent[find(x)] = find(y)
    members: Dict[int, List[int]] = {}
    for v in Hn:
        members.setdefault(find(v), []).append(v)
    roots = list(members)
    out = {r: set() for r in roots}         # contracted arcs: class(x) requires class(y)
    for x, y in separable:
        out[find(x)].add(find(y))
    # reachability in the contracted forest gives the order
    below_sets: Dict[int, set] = {}

    def reach(r: int) -> set:
        if r not in below_sets:
            s = {r}
            for t in out[r]:
                s |= reach(t)
            below_sets[r] = s
        return below_sets[r]

    for r in roots:
        reach(r)
    roots.sort(key=lambda r: (len(below_sets[r]), min(members[r])))
    index = {r: a for a, r in enumerate(roots)}
    k0 = len(roots)
    below = np.zeros((k0, k0), dtype=bool)
    for r in roots:
        for t in below_sets[r]:
            below[index[r], index[t]] = True
    classes = [np.array(sorted(members[r]), dtype=np.int64) for r in roots]
    class_of = -np.ones(inst.n, dtype=np.int64)
    for a, K in enumerate(classes):
        class_of[K] = a
    w = inst.w
    w_class = np.array([w[K].sum() for K in classes], dtype=float)
    w_H = float(w[sol.H].sum())
    c_prime = float(c) - float(w[sol.F].sum())
    return SplitStructure(inst.n, sol.F.copy(), sol.H.copy(), sol.Z.copy(), classes, class_of, below,
                          w_class, below.astype(float) @ w_class, below.T.astype(float) @ w_class,
                          w_H, c_prime, 0)


def hook_length_count(struct: SplitStructure) -> int:
    """Number of linear extensions of a forest order by the hook-length formula.

    If every class has at most one class immediately above it, a linear
    extension places each class after the whole set below it, and the count is
    :math:`k_0! / \\prod_K |{\\downarrow}K|` (sizes in classes).  If every
    class has at most one class immediately below it, the count is
    :math:`k_0! / \\prod_K |{\\uparrow}K|`.  Raises if neither holds.
    """
    below = struct.below
    k0 = below.shape[0]
    strict = below & ~np.eye(k0, dtype=bool)
    # immediate relations (cover relation) of the order
    cover = strict & ~(strict.astype(int) @ strict.astype(int) > 0)
    n_above = cover.sum(axis=0)       # classes immediately above b
    n_below = cover.sum(axis=1)       # classes immediately below a
    if np.all(n_above <= 1):
        hooks = below.sum(axis=1)     # |down K|
    elif np.all(n_below <= 1):
        hooks = below.sum(axis=0)     # |up K|
    else:
        raise ValueError("the order is not a forest order")
    num = factorial(k0)
    den = int(np.prod([int(x) for x in hooks], dtype=object)) if k0 else 1
    assert num % den == 0
    return num // den
