"""The parametric-minimum-cut method (:mod:`macroitems.pseudoflow_path`).

The reduction to a linear parametric minimum cut is correct -- these tests check
it against :func:`~macroitems.path.canonical_path` on the running example, on
degenerate corners and on thirty random instances -- but the public
``pseudoflow`` package (2022.12.0) misses breakpoints, so the public entry point
is disabled and the tests pin down *exactly* what fails.

Three kinds of test live here:

* the reduction is sound: whatever the package returns is a nested, complete
  chain of closures, and it is always a *coarsening* of the canonical sequence
  (consecutive macroitems merged), never anything else;
* it is exactly the canonical sequence on every instance small enough that the
  bug does not bite (the running example, the degenerate corners, and every
  random instance with n <= 40 tried here);
* the documented counterexample still fails -- a sentinel: if a future version
  of the package passes it, ``test_counterexample_is_still_broken`` fails and
  tells us to re-examine whether the method can be enabled.
"""
from __future__ import annotations

import importlib.metadata as importlib_metadata
from fractions import Fraction as Fr

import numpy as np
import pytest

from macroitems import canonical_path, layered_grid, random_dag, running_example
from macroitems.closure import is_closure
from macroitems.pseudoflow_path import (README_CAVEAT, build_parametric_network,
                                        canonical_path_pseudoflow, counterexample_instance,
                                        parametric_chain_pseudoflow, pseudoflow_available)

pytestmark = pytest.mark.skipif(not pseudoflow_available(),
                                reason="the optional 'pseudoflow' package is not installed")

# Paper numbering (1-based) of the canonical macroitems of the running example.
PAPER_MACROITEMS = [[3, 6], [1, 2, 5], [4, 7, 8]]
PAPER_RATIOS = [Fr(2), Fr(3, 2), Fr(1)]

# 30 random instances: 5 sizes x 3 densities x 2 seeds.
DAG_CASES = [(n, d, seed) for n in (3, 9, 20, 40, 80)
             for d in (0.0, 1.0, 3.0) for seed in (0, 1)]
# Of those 30, exactly one is solved wrongly by pseudoflow 2022.12.0.  The list
# is pinned rather than tolerated: a change in either direction (a case that
# starts failing, or this one that starts passing) must fail the suite and be
# looked at.  There is no size below which the package is safe -- the three-item
# counterexample above is the smallest instance we know of -- so this is a pin,
# not a threshold.
KNOWN_WRONG_DAGS = {(80, 0.0, 1)}


# ------------------------------------------------------------------ helpers
def exact_ratios(inst, macroitems):
    return [Fr(int(round(inst.p[I].sum())), int(round(inst.w[I].sum()))) for I in macroitems]


def sets_of(macroitems):
    return [frozenset(I.tolist()) for I in macroitems]


def is_coarsening(macroitems, reference) -> bool:
    """True if every set of ``macroitems`` is the union of a block of consecutive
    sets of ``reference``, in order (the observed failure mode of the package)."""
    level = {}
    for r, I in enumerate(reference):
        for i in I.tolist():
            level[i] = r
    pos = 0
    for I in macroitems:
        levels = sorted({level[i] for i in I.tolist()})
        if levels != list(range(pos, pos + len(levels))):
            return False
        if sum(len(reference[r]) for r in levels) != I.size:
            return False
        pos += len(levels)
    return pos == len(reference)


def check_chain_is_sound(inst, closures, lambdas):
    """The chain reported by the package: nested, complete, closed, and every
    reported breakpoint is the exact ratio of the increment it introduces."""
    prev = np.zeros(inst.n, dtype=bool)
    assert not closures[0].any(), "the first reported cut should be empty"
    assert closures[-1].all(), "the last reported cut should be the whole item set"
    for r, cur in enumerate(closures):
        assert not (prev & ~cur).any(), "the chain is not nested"
        assert is_closure(inst, cur), "a reported cut is not a closure"
        if r:
            inc = np.flatnonzero(cur & ~prev)
            assert inc.size, "a reported interval does not change the cut"
            lam = float(inst.p[inc].sum() / inst.w[inc].sum())
            assert lam == pytest.approx(lambdas[r], abs=1e-9, rel=1e-12), \
                "lambda = Lambda - t does not match the ratio of the increment"
        prev = cur


def compare(inst):
    """(pseudoflow path, reference path); checks the soundness of the chain."""
    closures, lambdas, _ = parametric_chain_pseudoflow(inst)
    check_chain_is_sound(inst, closures, lambdas)
    got = canonical_path_pseudoflow(inst, allow_incorrect=True)
    ref = canonical_path(inst)
    assert is_coarsening(got.macroitems, ref.macroitems), (
        f"{inst.name}: pseudoflow returned a chain that is not even a coarsening "
        f"of the canonical sequence")
    return got, ref


def assert_equal_paths(got, ref, inst):
    assert sets_of(got.macroitems) == sets_of(ref.macroitems), inst.name
    assert exact_ratios(inst, got.macroitems) == exact_ratios(inst, ref.macroitems), inst.name
    assert np.array_equal(got.P, ref.P) and np.array_equal(got.W, ref.W), inst.name
    assert got.q == ref.q, inst.name
    assert got.method == "pseudoflow"
    assert got.n_maxflow == 1 and got.seconds >= 0.0


# ------------------------------------------------------- the method is off
def test_disabled_by_default():
    """The public entry point refuses to answer, and says why."""
    inst = running_example()
    with pytest.raises(NotImplementedError) as exc:
        canonical_path_pseudoflow(inst)
    msg = str(exc.value)
    assert "misses breakpoints" in msg
    assert "p = (3, 2, 1)" in msg           # the minimal counterexample
    assert "allow_incorrect=True" in msg
    assert "not be used" in msg             # the package's own README caveat


def test_readme_caveat_is_quoted_verbatim():
    """``README_CAVEAT`` must be the text the installed package actually ships:
    the paper quotes it, so it may not drift."""
    description = importlib_metadata.metadata("pseudoflow").get_payload()
    assert README_CAVEAT in description


def test_counterexample_is_still_broken():
    """Sentinel.  ``p = (3, 2, 1)``, ``w = (1, 1, 1)``, no arcs: the canonical
    sequence is {0} | {1} | {2}, the package reports {0} | {1, 2}.

    If this test ever fails because the sequences agree, the package has been
    fixed and :func:`canonical_path_pseudoflow` should be re-examined (rerun the
    whole comparison, not just this instance, before enabling it).
    """
    inst = counterexample_instance()
    ref = canonical_path(inst)
    assert sets_of(ref.macroitems) == [frozenset({0}), frozenset({1}), frozenset({2})]
    got = canonical_path_pseudoflow(inst, allow_incorrect=True)
    assert sets_of(got.macroitems) != sets_of(ref.macroitems), (
        "pseudoflow now solves the documented counterexample: re-check whether "
        "canonical_path_pseudoflow can be enabled")
    assert sets_of(got.macroitems) == [frozenset({0}), frozenset({1, 2})]


def test_counterexample_reported_cut_is_not_a_minimum_cut():
    """The failure is a wrong answer, not a coarser one: on the first reported
    interval the reported source set is strictly worse than the empty one.

    With ``v_i = p_i - lambda w_i``, the source set of a minimum cut maximises
    ``v(C)``; at ``lambda = 3.5`` the reported ``C = {0}`` has ``v = -0.5 < 0``.
    """
    inst = counterexample_instance()
    closures, lambdas, _ = parametric_chain_pseudoflow(inst)
    assert [np.flatnonzero(c).tolist() for c in closures] == [[], [0], [0, 1, 2]]
    lam = 3.5
    v = inst.p - lam * inst.w
    assert float(v[closures[1]].sum()) < 0.0


# ----------------------------------------------- the reduction is correct
def test_reduction_capacities():
    """Every capacity of the parametric network is non-negative over the whole
    range (the C library aborts the process on a negative one), the source arcs
    are constant and the sink arcs decreasing in the package's parameter."""
    for inst in (running_example(), random_dag(30, 2.0, seed=3),
                 # ratios far below zero: the plan's K >= max(-p_i) is not enough
                 random_dag(20, 1.0, seed=4, p_range=(-500, 5), w_range=(1, 3))):
        net = build_parametric_network(inst)
        n = inst.n
        const = np.array(net.graph.es["const"], dtype=float)
        mult = np.array(net.graph.es["mult"], dtype=float)
        assert np.all(mult[:n] == 0.0)                      # source arcs: non-decreasing
        assert np.all(mult[n:2 * n] == -inst.w)             # sink arcs: non-increasing
        assert np.all(mult[2 * n:] == 0.0)                  # precedence arcs
        for t in (0.0, 0.5 * net.t_max, net.t_max):
            assert (const + mult * t).min() >= 0.0
        # v_i = a_i - b_i = p_i - lambda w_i at both ends of the range
        for t in (0.0, net.t_max):
            v = const[:n] - (const[n:2 * n] + mult[n:2 * n] * t)
            assert np.allclose(v, inst.p - net.lam(t) * inst.w)
        assert net.big > const[:n].sum()                    # no precedence arc is cut
        assert net.lambda_lo < float((inst.p / inst.w).min())
        assert net.Lambda > float((inst.p / inst.w).max())


def test_running_example():
    inst = running_example()
    got, ref = compare(inst)
    assert [sorted((I + 1).tolist()) for I in got.macroitems] == PAPER_MACROITEMS
    assert exact_ratios(inst, got.macroitems) == PAPER_RATIOS
    assert_equal_paths(got, ref, inst)


# Degenerate corners: no arcs, all profits negative, all positive, ties.
DEGENERATE = [
    ("single item", [3], [2], []),
    ("no arcs", [5, -1, 3, 4], [2, 1, 1, 3], []),
    ("all profits negative", [-1, -2, -3, -4], [1, 2, 3, 1], [(0, 1), (1, 2), (3, 2)]),
    ("all profits negative, no arcs", [-1, -2, -3, -4], [1, 2, 3, 1], []),
    ("all profits positive", [1, 2, 3, 4], [4, 3, 2, 1], [(0, 1), (1, 2), (2, 3)]),
    ("all profits positive, no arcs", [1, 2, 3, 4], [4, 3, 2, 1], []),
    ("all profits zero", [0, 0, 0, 0], [1, 2, 3, 4], [(0, 1), (2, 3)]),
    # ties in the ratios: the maximal-tie convention must merge them
    ("tie, two disjoint items", [2, 2, 4], [1, 1, 2], []),
    ("tie, two disjoint chains", [-1, 5, -1, 5], [1, 1, 1, 1], [(1, 0), (3, 2)]),
    ("tie, three blocks", [3, 3, 1, 1], [1, 1, 1, 1], [(2, 0), (3, 1)]),
    ("nested equal ratios", [2, 2], [1, 1], [(1, 0)]),
    ("negative ratio ties", [-2, -2, 6], [1, 1, 2], []),
    ("star, root first", [6, -1, -1, -1], [1, 1, 1, 1], [(0, 1), (0, 2), (0, 3)]),
    ("star, leaves first", [-2, 3, 3, 3], [1, 1, 1, 1], [(1, 0), (2, 0), (3, 0)]),
    ("path graph", [-1, -1, 9, -1], [1, 1, 1, 1], [(1, 0), (2, 1), (3, 2)]),
    ("very negative ratios", [-100, 1], [1, 1], []),
]


@pytest.mark.parametrize("name,p,w,arcs", DEGENERATE, ids=[c[0] for c in DEGENERATE])
def test_degenerate_cases(name, p, w, arcs):
    from macroitems import Instance
    inst = Instance(np.asarray(p, dtype=float), np.asarray(w, dtype=float),
                    np.asarray(list(arcs), dtype=np.int64).reshape(-1, 2), name=name)
    inst.validate()
    got, ref = compare(inst)
    assert_equal_paths(got, ref, inst)


@pytest.mark.parametrize("n,d,seed", DAG_CASES, ids=[f"dag{n}d{d}s{s}" for n, d, s in DAG_CASES])
def test_random_dags(n, d, seed):
    inst = random_dag(n, d, seed=seed)
    got, ref = compare(inst)
    assert inst.n <= EXACT_UP_TO
    assert_equal_paths(got, ref, inst)


@pytest.mark.parametrize("nx,ny,nz,cone", [(3, 3, 3, 5), (4, 4, 3, 5), (3, 3, 4, 9)])
def test_layered_grids(nx, ny, nz, cone):
    inst = layered_grid(nx, ny, nz, cone=cone, seed=0)
    got, ref = compare(inst)
    assert inst.n <= EXACT_UP_TO
    assert_equal_paths(got, ref, inst)


# --------------------------------------------------------------------- slow
@pytest.mark.slow
@pytest.mark.parametrize("n,d,seed", [(n, d, s) for n in (60, 90, 130, 200, 300)
                                      for d in (0.0, 1.5, 4.0) for s in (0, 1)])
def test_large_random_dags_are_coarsenings(n, d, seed):
    """Above roughly n = 50 the package starts to miss breakpoints.  We cannot
    assert equality any more, but the chain must still be a sound coarsening --
    that is what makes the failure invisible without a reference solution."""
    inst = random_dag(n, d, seed=seed)
    compare(inst)


@pytest.mark.slow
def test_failure_rate_grows_with_size():
    """The documented decay: exact on every small instance, wrong on some large
    ones.  This is the evidence behind the module docstring."""
    small = [random_dag(n, d, seed=s) for n in (10, 25, 40) for d in (0.0, 1.5, 4.0)
             for s in (0, 1)]
    large = [random_dag(n, d, seed=s) for n in (200, 300) for d in (0.0, 1.5, 4.0)
             for s in (0, 1)]
    for inst in small:
        got, ref = compare(inst)
        assert_equal_paths(got, ref, inst)
    wrong = 0
    for inst in large:
        got, ref = compare(inst)
        wrong += sets_of(got.macroitems) != sets_of(ref.macroitems)
    assert wrong > 0, ("pseudoflow no longer misses breakpoints on the large "
                       "instances: re-examine whether the method can be enabled")
