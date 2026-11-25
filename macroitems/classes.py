"""Inseparability classes of the split macroitem, their order, and branching.

At a nondegenerate capacity the optimal closures at the critical price
:math:`\\lambda_h` are :math:`\\mathcal M_{h-1} \\cup L` for the *tight* subsets
``L`` of the split macroitem ``H``: the relatively closed ``L`` with
:math:`b(L) = 0`, where :math:`b_i = p_i - \\lambda_h w_i`.  They form a lattice
that contains the empty set and ``H``.  This module derives from it, on an
arbitrary precedence DAG:

* the **inseparability classes** of ``H``: two items are inseparable when every
  tight subset contains both or neither.  Their number ``k0`` gives
  :math:`\\dim X^\\star(c) = k_0 - 1`;
* the **order** :math:`K' \\preceq K` iff :math:`K'` lies in the smallest
  tight subset containing ``K``.  The tight subsets are exactly the unions of
  down-sets of this order;
* the weights of :math:`{\\downarrow}K` (the smallest tight subset containing
  ``K``) and of :math:`{\\uparrow}K`, and with them the **critical class**,
  the unique class (if any) with
  :math:`w(H \\setminus {\\uparrow}K) < c' < w({\\downarrow}K)`, where
  :math:`c' = c - w(F)`.  Branching on one variable decreases the LP bound
  exactly on the items of the critical class;
* the **attainability** of a 0/1 pattern on a set of items: some optimal LP
  solution agrees with the pattern.  Branching on a set ``T`` decreases the
  bound iff no pattern on ``T`` is attainable;
* the **number of maximal chains** of tight subsets, i.e. of linear extensions
  of the order.  This is the number of full-dimensional pieces of the primal
  optimal face.

Everything is computed from the smallest tight subset containing each item of
``H``.  :func:`macroitems.faces.minimal_tight_sets` obtains it with one maximum
flow per item of ``H``.
"""
from __future__ import annotations

import dataclasses
import itertools
from math import factorial
from typing import Dict, List, Mapping, Optional, Sequence

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

from .faces import minimal_tight_sets
from .instance import Instance
from .path import LPSolution

__all__ = ["SplitStructure", "split_structure", "count_linear_extensions"]


@dataclasses.dataclass
class SplitStructure:
    """Inseparability classes of the split macroitem and their order.

    Classes are numbered ``0..k0-1`` in an order compatible with
    :math:`\\preceq` (a class comes after every class below it).
    """
    n: int
    F: np.ndarray                 # bool mask, full region
    H: np.ndarray                 # bool mask, split macroitem
    Z: np.ndarray                 # bool mask, null region
    classes: List[np.ndarray]     # item indices of each class
    class_of: np.ndarray          # shape (n,), class index on H, -1 elsewhere
    below: np.ndarray             # below[a, b] True iff K_b precedes-or-equals K_a
    w_class: np.ndarray           # w(K)
    w_down: np.ndarray            # w(down K)
    w_up: np.ndarray              # w(up K)
    w_H: float
    c_prime: float                # c - w(F)
    n_maxflow: int

    @property
    def k0(self) -> int:
        return len(self.classes)

    # ------------------------------------------------------------ the order
    def down(self, a: int) -> np.ndarray:
        """Items of :math:`{\\downarrow}K_a`, the smallest tight subset containing ``K_a``."""
        return np.sort(np.concatenate([self.classes[b] for b in np.flatnonzero(self.below[a])]))

    def up(self, a: int) -> np.ndarray:
        """Items of :math:`{\\uparrow}K_a`."""
        return np.sort(np.concatenate([self.classes[b] for b in np.flatnonzero(self.below[:, a])]))

    # ------------------------------------------------------------ branching
    def is_critical(self, a: int) -> bool:
        return bool(self.w_H - self.w_up[a] < self.c_prime < self.w_down[a])

    @property
    def critical(self) -> Optional[int]:
        """Index of the critical class, or ``None``.  At most one class is critical."""
        crit = [a for a in range(self.k0) if self.is_critical(a)]
        assert len(crit) <= 1, "two critical classes contradict Corollary cor:critical(b)"
        return crit[0] if crit else None

    def attainable(self, pattern: Mapping[int, int]) -> bool:
        """Whether some optimal LP solution satisfies ``x_t = pattern[t]`` for all ``t``.

        Items of ``F`` are 1 and items of ``Z`` are 0 in every optimal solution.
        For items of ``H`` the test is Lemma ``lem:patterns``:
        :math:`A \\cap B = \\emptyset` and :math:`w(A) \\le c' \\le w(H \\setminus B)`,
        with ``A`` the union of the down-sets of the classes of the items set to
        1, and ``B`` the union of the up-sets of those set to 0.
        """
        on: set = set()
        off: set = set()
        for t, a in pattern.items():
            t = int(t)
            if self.F[t]:
                if a != 1:
                    return False
            elif self.Z[t]:
                if a != 0:
                    return False
            elif a == 1:
                on.add(int(self.class_of[t]))
            else:
                off.add(int(self.class_of[t]))
        A = np.zeros(self.k0, dtype=bool)
        for k in on:
            A |= self.below[k]
        B = np.zeros(self.k0, dtype=bool)
        for k in off:
            B |= self.below[:, k]
        if np.any(A & B):
            return False
        return bool(self.w_class[A].sum() <= self.c_prime <= self.w_H - self.w_class[B].sum())

    def bound_decreases(self, items: Sequence[int]) -> bool:
        """Proposition ``prop:branching``: branching on all of ``items`` at once
        gives a bound strictly below :math:`z(c)` iff no 0/1 pattern on them is
        attainable.  Exponential in ``len(items)``."""
        items = [int(t) for t in items]
        return not any(self.attainable(dict(zip(items, bits)))
                       for bits in itertools.product((0, 1), repeat=len(items)))

    # ------------------------------------------------------------ the face
    def count_maximal_chains(self, max_k0: int = 25) -> int:
        """Number of maximal chains of tight subsets, i.e. the number of linear
        extensions of :math:`\\preceq` (Lemma ``lem:chains``).  It is also the
        number of pieces in the subdivision of the primal face of Proposition
        ``prop:chains``.  Dynamic programming over down-sets, so exponential in
        the width of the order."""
        return count_linear_extensions(self.below, max_k0=max_k0)


def count_linear_extensions(below: np.ndarray, max_k0: int = 25) -> int:
    """Linear extensions of the partial order ``below[a, b]`` (``b`` precedes-or-equals ``a``)."""
    k = below.shape[0]
    if k > max_k0:
        raise ValueError(f"k0 = {k} too large for the exact count (max_k0={max_k0})")
    pred = [sum(1 << b for b in np.flatnonzero(below[a]) if b != a) for a in range(k)]
    count: Dict[int, int] = {0: 1}
    for size in range(k):
        nxt: Dict[int, int] = {}
        for S, c in count.items():
            for a in range(k):
                if not S >> a & 1 and pred[a] & S == pred[a]:
                    T = S | 1 << a
                    nxt[T] = nxt.get(T, 0) + c
        count = nxt
    return count.get((1 << k) - 1, 1)


def split_structure(inst: Instance, sol: LPSolution, c: float, max_H: int = 20000) -> SplitStructure:
    """Classes, order, down/up weights and critical class at the nondegenerate capacity ``c``
    (the capacity ``sol`` was computed at)."""
    if sol.degenerate or sol.H is None or not sol.H.any():
        raise ValueError("split_structure needs a nondegenerate capacity (nonempty split macroitem)")
    nodes, _, Tm, n_mf = minimal_tight_sets(inst, sol, max_H)
    k = nodes.size
    mutual = Tm & Tm.T
    _, label = connected_components(csr_matrix(mutual), directed=False)
    # one representative per class; order classes by the size of their down-set,
    # which is compatible with the order (strict inclusion means smaller size)
    reps: Dict[int, int] = {}
    for i in range(k):
        reps.setdefault(int(label[i]), i)
    order = sorted(reps, key=lambda lab: (int(Tm[reps[lab]].sum()), reps[lab]))
    index = {lab: a for a, lab in enumerate(order)}
    k0 = len(order)
    classes = [nodes[label == lab] for lab in order]
    class_of = -np.ones(inst.n, dtype=np.int64)
    for a, members in enumerate(classes):
        class_of[members] = a
    below = np.zeros((k0, k0), dtype=bool)
    for lab in order:
        a = index[lab]
        members_down = Tm[reps[lab]]
        for b_lab in np.unique(label[members_down]):
            below[a, index[int(b_lab)]] = True
    w = inst.w
    w_class = np.array([w[K].sum() for K in classes], dtype=float)
    w_down = below.astype(float) @ w_class
    w_up = below.T.astype(float) @ w_class
    w_H = float(w[sol.H].sum())
    c_prime = float(c) - float(w[sol.F].sum())
    assert 0 < c_prime < w_H, "capacity does not match the solution"
    return SplitStructure(inst.n, sol.F.copy(), sol.H.copy(), sol.Z.copy(), classes, class_of,
                          below, w_class, w_down, w_up, w_H, c_prime, n_mf)
