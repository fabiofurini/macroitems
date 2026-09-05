"""The same object under other names: the algorithms of three other literatures
must produce the canonical macroitem sequence of the paper.

The dictionary of Section 7 claims that the translations between the knapsack,
the mining, the scheduling and the parametric-flow readings of the relaxation
are literal.  The claim is tested here by running the *other communities' own
algorithms*, reimplemented from their sources in :mod:`tests.literature` and
using nothing of :mod:`macroitems`, and comparing their output with ours:

  1. Shaw and Cho (1998, Section 3), tree knapsack: their subtree-deletion
     bound.  Its aggregated subtrees are our macroitems and its Lagrangian
     bound is z(c).
  2. Sidney (1975, Algorithm 1*) and Margot, Queyranne and Wang (2003,
     Theorem 3.9), scheduling: the reduced Sidney decomposition of
     1|prec|sum w_j C_j is our canonical sequence after reversing the arcs and
     exchanging the roles of profit and weight.
  3. The running example of the paper, in the part not already covered by
     :mod:`tests.test_running_example` (which checks the canonical sequence
     {3,6}, {1,2,5}, {4,7,8}, its ratios 2, 3/2, 1, the cumulative points,
     z(4) = 7 with lambda = 3/2 and theta = 1/2, the dual certificate, the
     reduced costs and the optimal-face dimensions 0 and 3): the two *feasible
     but not canonical* sequences of Section 4.1 and the values they give.

All comparisons are exact: the data are integers, so every ratio, bound and
objective value is a rational number (:class:`fractions.Fraction`).
"""
from __future__ import annotations

import random
from fractions import Fraction as Fr

import numpy as np
import pytest

from conftest import make_instance
from literature import (finest_sidney_decomposition, out_tree_arcs, random_out_tree,
                        random_scheduling_instance, reduced_sidney_decomposition,
                        shaw_cho_blocks, shaw_cho_bound)
from macroitems import canonical_path, running_example, solution_from_path, solve_capacity

SEED = 20240917


def exact_ratios(inst, macroitems):
    """lambda_r = p(I_r) / w(I_r) as a Fraction."""
    return [Fr(int(round(inst.p[I].sum())), int(round(inst.w[I].sum()))) for I in macroitems]


def as_sets(macroitems):
    return [frozenset(int(i) for i in I) for I in macroitems]


# ------------------------------------------------------- 1. Shaw and Cho (1998)
def tree_cases(n_random: int, sizes, seed: int):
    """Random rooted trees with integer data, plus three hand-made shapes.

    Shaw and Cho's tree knapsack has positive profits, so the random profits are
    positive here too; ``test_tree_blocks_with_profits_of_both_signs`` checks
    the block correspondence beyond their setting.
    """
    out = [
        ("path", [-1, 0, 1, 2], [1, 6, 1, 8], [1, 2, 1, 2]),
        ("star", [-1, 0, 0, 0], [2, 9, 4, 1], [3, 3, 1, 1]),
        # two identical branches: their ratios tie and must be merged
        ("tied branches", [-1, 0, 0, 1, 2], [1, 4, 4, 1, 1], [2, 1, 1, 1, 1]),
    ]
    rng = random.Random(seed)
    for t in range(n_random):
        n = rng.choice(list(sizes))
        spread = rng.choice([0, 0, 1, 2, 3])
        # half of the instances use a narrow alphabet, where equal ratios of
        # disjoint subtrees -- the case the tie convention decides -- are common
        narrow = t % 2 == 0
        parent, p, w = random_out_tree(rng, n, spread=spread,
                                       p_choices=(1, 2, 3, 4) if narrow else (1, 2, 3, 4, 6, 8, 12),
                                       w_choices=(1, 2) if narrow else (1, 2, 3, 4))
        out.append((f"tree#{t}(n={n},spread={spread})", parent, p, w))
    return out


TREE_CASES = tree_cases(40, range(2, 13), SEED)


@pytest.mark.parametrize("name,parent,p,w", TREE_CASES, ids=[c[0] for c in TREE_CASES])
def test_shaw_cho_subtrees_are_the_canonical_macroitems(name, parent, p, w):
    """Shaw and Cho (1998, Algorithm 2 and Theorem 3): on a tree, the subtrees
    deleted by their bound, aggregated with their ratios, are exactly the
    canonical macroitems of Section 4 read backwards."""
    inst = make_instance(p, w, out_tree_arcs(parent), name)
    blocks = shaw_cho_blocks(parent, p, w)
    for method in ("bisection", "dinkelbach"):
        path = canonical_path(inst, method=method)
        assert as_sets(path.macroitems) == [T for T, _ in reversed(blocks)], (name, method)
        assert exact_ratios(inst, path.macroitems) == [r for _, r in reversed(blocks)], (name, method)


@pytest.mark.parametrize("name,parent,p,w", TREE_CASES, ids=[c[0] for c in TREE_CASES])
def test_shaw_cho_bound_is_the_lp_value(name, parent, p, w):
    """Shaw and Cho (1998, Proposition 2): their Lagrangian bound, obtained by
    deleting subtrees until the demand fits, equals z(c) at every capacity."""
    inst = make_instance(p, w, out_tree_arcs(parent), name)
    path = canonical_path(inst)
    blocks = shaw_cho_blocks(parent, p, w)
    cumulative = {Fr(int(x)) for x in path.W}
    for k in range(0, 2 * sum(w) + 3):
        c = Fr(k, 2)
        z, critical = shaw_cho_bound(parent, p, w, c, blocks)
        ctx = (name, str(c))
        assert path.value_function(float(c)) == pytest.approx(float(z), abs=1e-9), ctx
        sol = solution_from_path(inst, path, float(c))
        assert sol.value == pytest.approx(float(z), abs=1e-9), ctx
        assert solve_capacity(inst, float(c)).value == pytest.approx(float(z), abs=1e-9), ctx
        # their "critical item" is the last deleted subtree; away from the
        # breakpoints w(M_r), where theta = 0 and theta = 1 describe the same
        # solution, it is the split macroitem I_h of Definition 4.2
        if 0 < c < Fr(int(path.W[path.q])) and c not in cumulative:
            assert frozenset(np.flatnonzero(sol.H).tolist()) == critical, ctx


def test_tree_blocks_with_profits_of_both_signs():
    """Beyond Shaw and Cho's setting: their deletion rule still produces the
    canonical macroitems when profits may be negative (only their bound, which
    assumes that everything is worth taking, needs positive profits)."""
    rng = random.Random(SEED + 1)
    for t in range(25):
        n = rng.randint(2, 12)
        parent, p, w = random_out_tree(rng, n, spread=rng.choice([0, 1, 2]),
                                       p_choices=(-6, -3, -1, 0, 1, 2, 4, 9))
        inst = make_instance(p, w, out_tree_arcs(parent), f"signed tree#{t}")
        path = canonical_path(inst)
        blocks = shaw_cho_blocks(parent, p, w)
        assert as_sets(path.macroitems) == [T for T, _ in reversed(blocks)], t
        assert exact_ratios(inst, path.macroitems) == [r for _, r in reversed(blocks)], t


@pytest.mark.slow
def test_shaw_cho_on_larger_trees():
    """The same two statements on 80 trees of up to 600 items."""
    rng = random.Random(SEED + 2)
    for t in range(80):
        n = rng.randint(30, 600)
        parent, p, w = random_out_tree(rng, n, spread=rng.choice([0, 1, 4]))
        inst = make_instance(p, w, out_tree_arcs(parent), f"big tree#{t}")
        path = canonical_path(inst)
        blocks = shaw_cho_blocks(parent, p, w)
        assert as_sets(path.macroitems) == [T for T, _ in reversed(blocks)], t
        assert exact_ratios(inst, path.macroitems) == [r for _, r in reversed(blocks)], t
        total = sum(w)
        for k in range(0, 21):
            c = Fr(k * total, 20)
            z, _ = shaw_cho_bound(parent, p, w, c, blocks)
            assert path.value_function(float(c)) == pytest.approx(float(z), rel=1e-9, abs=1e-9)


# ------------------- 2. Sidney (1975) / Margot, Queyranne and Wang (2003)
def sidney_cases(n_random: int, n_max: int, seed: int):
    """Small instances of 1|prec|sum w_j C_j as ``(name, p, w, prec)``.

    Most of the hand-made ones are about ties, which is where the two extreme
    tie conventions of Margot et al. (finest and reduced decomposition) differ;
    the others are degenerate corners (a single job, a chain, weights equal to
    zero, an instance that is a single block).
    """
    out = [
        ("independent jobs", [1, 2, 1], [2, 4, 3], []),
        # two disjoint chains with the same ratios: the reduced decomposition
        # merges them, the finest one does not
        ("tied chains", [1, 1, 1, 1], [3, 1, 3, 1], [(0, 1), (2, 3)]),
        ("chain", [1, 2, 1], [0, 5, 1], [(0, 1), (1, 2)]),
        ("zero weights", [2, 3], [0, 0], [(0, 1)]),
        ("one block", [2, 1, 1], [1, 5, 5], [(0, 1), (0, 2)]),
        ("tied singletons", [1, 1, 2, 2], [2, 2, 4, 4], []),
        ("single job", [3], [2], []),
    ]
    rng = random.Random(seed)
    for t in range(n_random):
        n = rng.randint(3, n_max)
        density = rng.choice([0.2, 0.35, 0.5])
        p, w, prec = random_scheduling_instance(rng, n, density=density)
        out.append((f"sched#{t}(n={n})", p, w, prec))
    return out


SIDNEY_CASES = sidney_cases(50, 8, SEED + 3)


def translated(p, w, prec, name):
    """The translation of Section 7.1: reverse the arcs (a scheduling pair
    (i, j), "i precedes j", becomes the arc (j, i), "i is a prerequisite of j")
    and take (profit, weight) = (w_j, p_j)."""
    return make_instance(w, p, [(j, i) for (i, j) in prec], name)


@pytest.mark.parametrize("name,p,w,prec", SIDNEY_CASES, ids=[c[0] for c in SIDNEY_CASES])
def test_reduced_sidney_decomposition_is_the_canonical_sequence(name, p, w, prec):
    """Sidney (1975, Algorithm 1*) and Margot, Queyranne and Wang (2003,
    Theorem 3.9): the blocks of the reduced Sidney decomposition, in order, are
    the canonical macroitems of the translated instance, and the reciprocals
    1 / rho of their ratios are the breakpoints lambda_r."""
    inst = translated(p, w, prec, name)
    blocks = reduced_sidney_decomposition(len(p), p, w, prec)
    for method in ("bisection", "dinkelbach"):
        path = canonical_path(inst, method=method)
        assert as_sets(path.macroitems) == [B for B, _ in blocks], (name, method)
        assert exact_ratios(inst, path.macroitems) == [r for _, r in blocks], (name, method)


@pytest.mark.parametrize("name,p,w,prec", SIDNEY_CASES, ids=[c[0] for c in SIDNEY_CASES])
def test_sidney_blocks_are_ratio_optimal_initial_sets(name, p, w, prec):
    """Every prefix of the decomposition is an initial set of the scheduling
    instance -- a closure after the translation -- and the stored reciprocals
    1 / rho strictly decrease, that is, the ratios rho of the successive blocks
    strictly increase, as in Sidney's Theorem 9 (1975, p. 290)."""
    blocks = reduced_sidney_decomposition(len(p), p, w, prec)
    prefix = set()
    for B, _ in blocks:
        prefix |= set(B)
        assert all(i in prefix for (i, j) in prec if j in prefix), name
    ratios = [r for _, r in blocks]
    assert ratios == sorted(ratios, reverse=True) and len(set(ratios)) == len(ratios), name


@pytest.mark.slow
def test_reduced_sidney_decomposition_on_ten_jobs():
    """The same correspondence on 400 instances with 9 or 10 jobs, the largest
    size for which all 2**n initial sets can be enumerated quickly."""
    rng = random.Random(SEED + 4)
    for t in range(400):
        n = rng.choice([9, 10])
        p, w, prec = random_scheduling_instance(rng, n, density=rng.choice([0.15, 0.3, 0.45]))
        name = f"sched10#{t}"
        inst = translated(p, w, prec, name)
        blocks = reduced_sidney_decomposition(n, p, w, prec)
        path = canonical_path(inst)
        assert as_sets(path.macroitems) == [B for B, _ in blocks], name
        assert exact_ratios(inst, path.macroitems) == [r for _, r in blocks], name


def test_reduction_of_the_finest_decomposition():
    """Margot, Queyranne and Wang (2003, Theorem 3.9): merging the blocks of
    equal ratio of *any* Sidney decomposition yields the reduced one.  Sidney's
    Algorithm 1 (the finest decomposition, their Corollary 3.12) is merged here
    block by block and must reproduce the output of Algorithm 1*, hence our
    canonical macroitems.  The counter at the end guards against the check
    becoming vacuous: the two decompositions must really differ on a decent
    number of the instances, otherwise the tie convention is never exercised."""
    n_strictly_finer = 0
    for name, p, w, prec in SIDNEY_CASES:
        finest = finest_sidney_decomposition(len(p), p, w, prec)
        merged = []
        for B, r in finest:
            if merged and merged[-1][1] == r:
                merged[-1] = (merged[-1][0] | B, r)
            else:
                merged.append((B, r))
        assert merged == reduced_sidney_decomposition(len(p), p, w, prec), name
        n_strictly_finer += len(finest) > len(merged)
    assert n_strictly_finer >= 15, n_strictly_finer


def test_tied_chains_are_merged_by_the_reduced_decomposition():
    """The tie convention, spelled out on the smallest example: two disjoint
    chains with equal ratios.  Sidney's Algorithm 1 would extract the two
    rho*-minimal sets {0} and {2} separately (the finest decomposition); the
    reduced decomposition of Margot et al. merges them, and that is our
    maximal-tie macroitem."""
    p, w, prec = [1, 1, 1, 1], [3, 1, 3, 1], [(0, 1), (2, 3)]
    blocks = reduced_sidney_decomposition(4, p, w, prec)
    assert [B for B, _ in blocks] == [frozenset({0, 2}), frozenset({1, 3})]
    assert [r for _, r in blocks] == [Fr(3), Fr(1)]
    path = canonical_path(translated(p, w, prec, "tied chains"))
    assert as_sets(path.macroitems) == [frozenset({0, 2}), frozenset({1, 3})]


# ---------------------------------------------- 3. the paper's running example
# The numbers of the running example that Section 4 states are asserted in
# tests/test_running_example.py (canonical sequence and ratios, cumulative
# points, z(4) = 7 with lambda = 3/2, the dual certificate, the reduced costs,
# and the primal/dual face dimensions 0 and 3).  What follows is the part of
# Section 4.1 that file does not cover: the two *feasible* sequences that are
# not canonical, and the values they produce at c = 4.

# paper numbering (1-based), Section 4.1
FEASIBLE_BLOCKS = {"I1": [1, 2, 3], "I2": [4, 7], "I3": [5, 6], "I4": [8]}


def sequence_value(inst, sequence, c: Fr):
    """The maximum-capacity solution of Definition 4.2 for a feasible sequence:
    fill the blocks in order, split the first one that does not fit."""
    got = Fr(0)
    left = Fr(c)
    for block in sequence:
        pb = Fr(int(round(inst.p[list(block)].sum())))
        wb = Fr(int(round(inst.w[list(block)].sum())))
        if wb <= left:
            got += pb
            left -= wb
        else:
            return got + left / wb * pb
    return got


def test_running_example_feasible_but_not_canonical_sequences():
    """Section 4.1: the ordered partition ({1,2,3}, {4,7}, {5,6}, {8}) is
    feasible and gives -2/3 at c = 4, and reordering it to
    ({1,2,3}, {5,6}, {4,7}, {8}) gives 8/3; both are far below z(4) = 7, so
    feasibility alone does not identify the canonical sequence."""
    from macroitems import is_closure
    inst = running_example()
    blocks = {k: [i - 1 for i in v] for k, v in FEASIBLE_BLOCKS.items()}   # to 0-based
    S = [blocks[k] for k in ("I1", "I2", "I3", "I4")]
    S_prime = [blocks[k] for k in ("I1", "I3", "I2", "I4")]
    # both orders are feasible: every prefix is a closure
    for sequence in (S, S_prime):
        mask = np.zeros(inst.n, dtype=bool)
        for block in sequence:
            mask[block] = True
            assert is_closure(inst, mask.copy())
        assert mask.all()
    # the paper's aggregate data of the four blocks (p, w)
    assert [(int(inst.p[b].sum()), int(inst.w[b].sum())) for b in S] == \
        [(-1, 3), (1, 3), (11, 3), (3, 1)]
    assert sequence_value(inst, S, Fr(4)) == Fr(-2, 3)
    assert sequence_value(inst, S_prime, Fr(4)) == Fr(8, 3)
    # the canonical sequence does better than both, and attains z(4) = 7
    path = canonical_path(inst)
    assert sequence_value(inst, [I.tolist() for I in path.macroitems], Fr(4)) == Fr(7)
