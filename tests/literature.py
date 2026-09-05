"""Algorithms of the other literatures, reimplemented from their own papers.

The point of :mod:`tests.test_literature` is that the objects of the paper are
already in use elsewhere under different names, so the algorithms below are
written from the *sources'* definitions, in the sources' vocabulary and with the
sources' orientation of the arcs, and use nothing from :mod:`macroitems`:

  * :func:`shaw_cho_blocks` / :func:`shaw_cho_bound` -- the tree-knapsack bound
    of Shaw and Cho (1998, Section 3: Proposition 2, Algorithm 2, Theorem 3),
    which repeatedly deletes the subtree of smallest aggregate ratio;
  * :func:`reduced_sidney_decomposition` -- Sidney's Algorithm 1* (1975, p. 291,
    "the largest possible rho-minimal set"), i.e. the reduced Sidney
    decomposition of Margot, Queyranne and Wang (2003, Theorem 3.9), computed by
    enumerating every initial set.

Everything is exact: profits, weights and ratios are integers or
:class:`fractions.Fraction`.
"""
from __future__ import annotations

import random
from fractions import Fraction as Fr
from typing import Dict, List, Sequence, Tuple

Block = Tuple[frozenset, Fr]


# ------------------------------------------------------- Shaw and Cho (1998)
# Orientation.  The tree knapsack problem of Shaw and Cho has a rooted tree in
# which an item can be selected only if its *parent* is selected: every item has
# at most one prerequisite, the root has none.  This is also the orientation of
# this package, where an arc (i, j) means "j is a prerequisite of i", so the
# instance is generated from a parent array and the arcs are (i, parent[i]).
# The feasible sets are then the subtrees containing the root, and a set can be
# deleted from a feasible set only if it is a union of full subtrees.

def out_tree_arcs(parent: Sequence[int]) -> List[Tuple[int, int]]:
    """Arcs (i, parent[i]) of the rooted tree, in this package's convention."""
    return [(i, int(par)) for i, par in enumerate(parent) if par >= 0]


def random_out_tree(rng: random.Random, n: int, spread: int = 0,
                    p_choices: Sequence[int] = (1, 2, 3, 4, 6, 8, 12),
                    w_choices: Sequence[int] = (1, 2, 3, 4)):
    """A random rooted tree on n items: item 0 is the root, item i > 0 has one
    prerequisite drawn from the ``spread`` items preceding it (``spread = 0``
    means any earlier item; ``spread = 1`` gives a path).  Profits and weights
    are integers drawn from small alphabets, so that ties in the ratios are
    common.  Returns ``(parent, p, w)``."""
    parent = [-1]
    for i in range(1, n):
        lo = 0 if spread <= 0 else max(0, i - spread)
        parent.append(rng.randrange(lo, i))
    p = [rng.choice(list(p_choices)) for _ in range(n)]
    w = [rng.choice(list(w_choices)) for _ in range(n)]
    return parent, p, w


def _children(parent: Sequence[int]) -> Dict[int, List[int]]:
    ch: Dict[int, List[int]] = {v: [] for v in range(len(parent))}
    for v, par in enumerate(parent):
        if par >= 0:
            ch[int(par)].append(v)
    return ch


def _bottom_up_order(parent: Sequence[int]) -> List[int]:
    """Nodes with every child before its parent (deepest first)."""
    depth = [0] * len(parent)
    for v in range(len(parent)):
        d, u = 0, v
        while parent[u] >= 0:
            u = int(parent[u])
            d += 1
        depth[v] = d
    return sorted(range(len(parent)), key=lambda v: -depth[v])


def shaw_cho_blocks(parent: Sequence[int], p: Sequence[int],
                    w: Sequence[int]) -> List[Block]:
    """Shaw and Cho (1998), Algorithm 2: aggregated subtrees, in deletion order.

    Their bound deletes from the tree, one at a time, the subtree of smallest
    aggregate ratio p(T)/w(T) (equivalently, the least profitable unit of
    demand), until what is left fits the capacity; deleting a subtree keeps the
    remaining set precedence-feasible, and Theorem 3 shows that the ratios of
    the successively deleted subtrees increase.  Here the deletion is carried
    all the way down to the empty tree, so the whole aggregation is returned.

    Ties are resolved as in the paper under study: all the subtrees attaining
    the minimum ratio are deleted together, so that the deleted block is the
    largest one of minimum ratio (its ratio is again the minimum, being a
    weighted average of equal ratios).  Returns ``[(T, p(T)/w(T)), ...]`` in
    deletion order, so the *last* block deleted is the first of the paper's
    canonical sequence.
    """
    n = len(p)
    children = _children(parent)
    order = _bottom_up_order(parent)
    alive = [True] * n
    blocks: List[Block] = []
    while any(alive):
        # aggregate profit and weight of the alive subtree rooted at each node
        # (a deleted node has all its descendants deleted, so this is exact)
        P = [0] * n
        W = [0] * n
        for v in order:
            if not alive[v]:
                continue
            P[v], W[v] = int(p[v]), int(w[v])
            for c in children[v]:
                if alive[c]:
                    P[v] += P[c]
                    W[v] += W[c]
        lam = min(Fr(P[v], W[v]) for v in range(n) if alive[v])
        # delete every subtree attaining the minimum ratio
        deleted = set()
        for v in range(n):
            if alive[v] and Fr(P[v], W[v]) == lam and v not in deleted:
                stack = [v]
                while stack:
                    u = stack.pop()
                    deleted.add(u)
                    stack.extend(c for c in children[u] if alive[c])
        for v in deleted:
            alive[v] = False
        blocks.append((frozenset(deleted), lam))
    return blocks


def shaw_cho_bound(parent: Sequence[int], p: Sequence[int], w: Sequence[int],
                   c) -> Fr:
    """Shaw and Cho (1998), Proposition 2 and Theorem 3: their upper bound.

    Delete subtrees in the order of :func:`shaw_cho_blocks` until the remaining
    demand fits the capacity; the last deleted subtree is their critical item,
    of ratio ``lam``, and the bound is ``p(remaining) + lam * (c - w(remaining))``
    -- the Lagrangian bound at the multiplier ``lam``.
    """
    c = Fr(c)
    rest_p = Fr(sum(int(v) for v in p))
    rest_w = Fr(sum(int(v) for v in w))
    lam = None
    for T, ratio in shaw_cho_blocks(parent, p, w):
        if rest_w <= c:
            break
        rest_p -= sum(int(p[i]) for i in T)
        rest_w -= sum(int(w[i]) for i in T)
        lam = ratio
    if lam is None:                      # the whole tree already fits
        return rest_p
    return rest_p + lam * (c - rest_w)


# ------------------------- Sidney (1975) / Margot, Queyranne and Wang (2003)
# Orientation.  In 1|prec|sum w_j C_j a pair (i, j) of the precedence relation
# means "job i must precede job j"; a set U of jobs is *initial* if no job
# outside U precedes a job inside U, and rho(U) = p(U)/w(U) is the ratio of
# total processing time to total deferral rate, to be *minimized*.  Processing
# times are positive and weights are nonnegative, so rho can be +infinity;
# below, ratios are stored as 1/rho = w(U)/p(U), which is always a finite
# nonnegative rational and which is *maximized*.  This is exactly the ratio of
# the translated instance (arcs reversed, profit = w_j, weight = p_j).

def random_scheduling_instance(rng: random.Random, n: int, density: float = 0.35,
                               p_choices: Sequence[int] = (1, 2, 3, 4),
                               w_choices: Sequence[int] = (0, 1, 2, 3, 4, 6)):
    """A random instance of 1|prec|sum w_j C_j with n jobs: processing times
    p_j > 0, deferral rates w_j >= 0, and precedence pairs (i, j) drawn along a
    random permutation, so the relation is acyclic.  The alphabets are small on
    purpose: ties between the ratios of disjoint initial sets are the delicate
    point of the correspondence.  Returns ``(p, w, prec)``."""
    order = list(range(n))
    rng.shuffle(order)
    prec = [(order[a], order[b]) for a in range(n) for b in range(a + 1, n)
            if rng.random() < density]
    p = [rng.choice(list(p_choices)) for _ in range(n)]
    w = [rng.choice(list(w_choices)) for _ in range(n)]
    return p, w, prec


def initial_sets(n: int, prec: Sequence[Tuple[int, int]]) -> List[frozenset]:
    """Every initial set (Sidney 1975, p. 285): no job outside precedes a job
    inside.  Enumerated over all 2**n subsets, so n must stay small."""
    out = []
    for bits in range(1 << n):
        U = frozenset(i for i in range(n) if bits >> i & 1)
        if all(i in U for (i, j) in prec if j in U):
            out.append(U)
    return out


def finest_sidney_decomposition(n: int, p: Sequence[int], w: Sequence[int],
                                prec: Sequence[Tuple[int, int]]) -> List[Block]:
    """The finest Sidney decomposition, by brute force.

    Sidney's Algorithm 1 (1975, pp. 285--286) extracts at each stage a
    rho*-minimal set, that is, a rho-minimal initial set of the residual problem
    containing no proper rho-minimal subset; the E-sets so obtained are pairwise
    disjoint (his Lemma 6 and Corollary 12) and give the finest Sidney
    decomposition of Margot, Queyranne and Wang (2003, Corollary 3.12).  When
    several rho*-minimal sets are tied, any of them may be extracted first; the
    lexicographically smallest one is used here to make the output
    deterministic, which is legitimate precisely because the tied blocks are
    then extracted consecutively.
    """
    sets = initial_sets(n, prec)
    full = frozenset(range(n))
    done = frozenset()
    blocks: List[Block] = []
    while done != full:
        cands = [U for U in sets if U > done]
        ratio = {U: Fr(sum(int(w[i]) for i in U - done),
                       sum(int(p[i]) for i in U - done)) for U in cands}
        best = max(ratio.values())
        opt = [U - done for U in cands if ratio[U] == best]
        minimal = [S for S in opt if not any(T < S for T in opt)]
        block = min(minimal, key=lambda S: (len(S), sorted(S)))
        blocks.append((block, best))
        done = done | block
    return blocks


def reduced_sidney_decomposition(n: int, p: Sequence[int], w: Sequence[int],
                                 prec: Sequence[Tuple[int, int]]) -> List[Block]:
    """The reduced Sidney decomposition, by brute force.

    Sidney's Algorithm 1* (1975, p. 291) repeatedly extracts from the residual
    problem a rho-minimal initial set, choosing "the largest possible" one;
    Margot, Queyranne and Wang (2003, Theorem 3.9) show that the decomposition
    so obtained -- the one in which blocks of equal ratio are merged -- is
    unique.  Here every initial set is enumerated and the largest ratio-optimal
    residual block is taken at each stage; the union of the tied ratio-optimal
    initial sets is again initial and again ratio-optimal (Sidney 1975, Lemma 6
    and Corollary 12), which the assertion below re-verifies on every instance.

    Returns ``[(B_r, 1/rho(B_r)), ...]``: the blocks in Sidney order, each with
    the reciprocal of its ratio (see the note on orientation above).
    """
    sets = initial_sets(n, prec)
    full = frozenset(range(n))
    done = frozenset()
    blocks: List[Block] = []
    while done != full:
        cands = [U for U in sets if U > done]
        ratio = {U: Fr(sum(int(w[i]) for i in U - done),
                       sum(int(p[i]) for i in U - done)) for U in cands}
        best = max(ratio.values())
        largest = frozenset().union(*[set(U) for U in cands if ratio[U] == best])
        assert Fr(sum(int(w[i]) for i in largest - done),
                  sum(int(p[i]) for i in largest - done)) == best, \
            "the union of tied rho-minimal initial sets is not rho-minimal"
        blocks.append((largest - done, best))
        done = largest
    return blocks
