"""The canonical macroitem sequence by ONE parametric minimum cut, through the
``pseudoflow`` package of the Hochbaum group.

**This method is disabled.**  The reduction below is correct and is verified by
the tests, but the public ``pseudoflow`` implementation silently returns an
*incomplete* list of breakpoints on the instances of this project, so
:func:`canonical_path_pseudoflow` raises :class:`NotImplementedError` unless the
caller explicitly asks for the unverified output.  See "What goes wrong" below
for a three-item counterexample and :data:`PSEUDOFLOW_MISSING_BREAKPOINTS` for
the measured failure rate.

Honesty note (must be reproduced in the paper)
----------------------------------------------
Even where it agrees, the package must not be quoted as a measurement of
Hochbaum's parametric algorithm.  Its own README (``pseudoflow`` 2022.12.0,
Quico Spaen, https://github.com/quic0/pseudoflow) says, verbatim:

    "This implementation uses a variant of the fully parametric HPF algorithm as
    described in: DS Hochbaum (2008), The Pseudoflow algorithm: A new algorithm
    for the maximum flow problem. Operations Research, 58(4):992-1009.

    This implementation does not use *free runs* nor does it use warm starts
    with informatiom from previous runs (see pg.15). This implementation should
    therefore **not be used** for comparison with the fully parametric HPF
    algorithm."

So any timing obtained here is a timing of a *simplified public implementation*
of parametric HPF, not of the algorithm of Hochbaum (2008), and the paper must
label it that way.  The package is also distributed under a non-commercial,
non-open-source licence (``License: Non-commercial license. Not an open-source
license.`` in its metadata), which is a further reason to keep it an optional
extra rather than a dependency.

There is, to our knowledge, **no public implementation of the parametric
maximum-flow algorithm of Gallo, Grigoriadis and Tarjan (1989)** that can be
trusted for benchmarking, so the comparison of the paper cannot include GGT
either; the O(k) geometric bisection of :func:`~macroitems.path.canonical_path`
is the reference method.

The reduction
-------------
The package solves a parametric minimum cut in which source-adjacent capacities
are non-decreasing and sink-adjacent capacities non-increasing in the parameter,
each of the linear form ``const + mult * parameter``.  Picard's literal
capacities ``v_i^+ = max(0, p_i - lambda w_i)`` and ``v_i^- = max(0, lambda w_i
- p_i)`` are piecewise linear in ``lambda``, so they do not fit (the package's
``roundNegativeCapacity`` option covers exactly that shape, but see below: it
does not help).  Instead, only the *difference* of the two terminal capacities
matters for a minimum cut, because for capacities ``a_i`` on ``s -> i`` and
``b_i`` on ``i -> t`` a cut with source side ``C`` has capacity

    cap(C) = sum_{i not in C} a_i + sum_{i in C} b_i
           = sum_i a_i - sum_{i in C} (a_i - b_i),

so minimising the cut maximises ``v(C) = sum_{i in C} (a_i - b_i)`` whatever the
common additive shift.  With a constant ``K``, the parameter ``t`` and the
substitution ``lambda = Lambda - t``:

* ``s -> i``  with capacity ``p_i + K``                  (mult 0, non-decreasing),
* ``i -> t``  with capacity ``(Lambda w_i + K) - t w_i`` (mult ``-w_i`` <= 0,
  non-increasing; it equals ``lambda w_i + K``),
* ``i -> j``  with a large constant capacity ``B`` for every precedence arc,

so that ``a_i - b_i = (p_i + K) - (lambda w_i + K) = p_i - lambda w_i`` and the
minimum cuts of the parametric network are exactly the maximal/minimal optimal
closures of the paper at ``lambda``.  ``B`` exceeds the total source capacity,
so no minimum cut ever separates ``i`` from a prerequisite ``j``.

Bounds.  Every macroitem ratio ``p(I)/w(I)`` is a ``w``-weighted average of the
item ratios ``p_i/w_i``, hence lies in ``[min_i p_i/w_i, max_i p_i/w_i]``; the
default range is one unit wider on each side (rounded to integers so that all
capacities stay exactly representable).  ``K`` must make every capacity
non-negative -- the package aborts the process with a message on a negative one
-- which needs ``K >= max_i(-p_i)`` for the source arcs *and*, since the lower
end ``lambda_lo`` of the range is negative as soon as some ratio is,
``K >= -lambda_lo * max_i w_i`` for the sink arcs.  The plan's ``K >=
max_i(-p_i)`` alone is not enough: for ``p = (-100, 1)``, ``w = (1, 1)`` the
range reaches ``lambda = -101`` and ``lambda w_i + K = -1 < 0``.

Post-processing.  As ``t`` grows, ``lambda`` decreases and the source sets grow,
so the cuts reported for the successive intervals are the nested closures
``M_0 = {} subset M_1 subset ... subset M_k = N``; the macroitems are their
increments and every ratio is recomputed exactly as ``p(I_r)/w(I_r)`` from the
instance data (the breakpoint returned by the package is only used to check the
mapping ``lambda = Lambda - t``).

What goes wrong
---------------
The reduction is right; the solver is not.  ``pseudoflow`` 2022.12.0 misses
breakpoints.  The smallest instance we found is three items, **no arcs**::

    p = (3, 2, 1),  w = (1, 1, 1)

whose canonical sequence is ``{0} | {1} | {2}`` with ratios ``3 > 2 > 1``.  For
any parameter range the package reports only two intervals, with source sets
``{0}`` and ``{0,1,2}``, i.e. the single breakpoint ``lambda = 2``.  That answer
is not merely coarse, it is not a parametric minimum cut at all: at
``lambda = 3.5`` the reported source set ``{0}`` has cut capacity ``3.5 + 2 + 1
= 6.5`` while the empty source set has ``3 + 2 + 1 = 6``.

The trigger is degeneracy of the divide-and-conquer over the parameter: the two
cuts bounding a sub-interval have capacity lines that intersect at
``lambda* = p(B \\ A)/w(B \\ A)``, and when that ``lambda*`` happens to be a true
breakpoint (so the minimum cut at ``lambda*`` is not unique) the implementation
stops instead of recursing, and loses every breakpoint of the sub-interval.  On
``p_i = n-i``, ``w_i = 1`` (ratios ``n, ..., 1``) the number of reported
intervals is correct only for ``n = 1, 2, 4, 8`` and collapses to 2 or 3
otherwise.  The same failure appears with the package's own intended
formulation (Picard's ``max(0, ...)`` capacities with
``roundNegativeCapacity=True``), so it is not an artefact of the reduction, and
it is insensitive to the range, to the constant ``K``, to a global scaling of
the capacities and to the order of the nodes and arcs.

The failure is not detectable from the output alone: the chain returned is
always nested, complete, made of genuine closures, and always a *coarsening* of
the canonical sequence with strictly decreasing increment ratios -- exactly the
properties one would check.  Certifying it would cost one maximum closure per
macroitem, which is what :func:`~macroitems.path.canonical_path` already does.

Measured on the generators of this package (``random_dag`` over 11 sizes x 5
densities x 2 seeds, plus five ``layered_grid`` block models): the sequence is
exact for every instance with ``n <= 48``, and then decays -- 9/10 at ``n = 60``,
7/10 at ``n = 90``, 8/10 at ``n = 130``, 4/10 at ``n = 200``, 2/10 at
``n = 300``.  It is therefore unusable both as a solver and as a timing
baseline, and Section A5 of the experimental plan allows dropping the method.
"""
from __future__ import annotations

import dataclasses
import math
import time
from typing import List, Optional, Tuple

import numpy as np

from .instance import Instance
from .path import MacroitemPath

__all__ = ["canonical_path_pseudoflow", "parametric_chain_pseudoflow",
           "build_parametric_network", "ParametricNetwork", "pseudoflow_available",
           "counterexample_instance", "PSEUDOFLOW_MISSING_BREAKPOINTS", "README_CAVEAT"]


#: The caveat printed in the package's own README (``pseudoflow`` 2022.12.0),
#: verbatim; the paper must quote it wherever a pseudoflow timing appears.
README_CAVEAT = (
    "This implementation does not use *free runs* nor does it use warm starts "
    "with informatiom from previous runs (see pg.15). This implementation should "
    "therefore **not be used** for comparison with the fully parametric HPF algorithm."
)

#: Why the method is disabled; used in the :class:`NotImplementedError` message.
PSEUDOFLOW_MISSING_BREAKPOINTS = (
    "pseudoflow 2022.12.0 misses breakpoints of the parametric minimum cut. "
    "Minimal counterexample (macroitems.pseudoflow_path.counterexample_instance()): "
    "three items, no arcs, p = (3, 2, 1), w = (1, 1, 1); the canonical sequence is "
    "{0} | {1} | {2} with ratios 3 > 2 > 1, but the package reports two intervals "
    "only, with source sets {0} and {0,1,2}. The reported set {0} is not a minimum "
    "cut on its interval (at lambda = 3.5 it costs 6.5 against 6 for the empty set), "
    "so this is a wrong answer, not a coarser one. On the generators of this package "
    "the sequence is exact up to n = 48 and then decays (4/10 at n = 200, 2/10 at "
    "n = 300). The returned chain is always nested, complete, closed and a coarsening "
    "of the canonical sequence, so the error cannot be detected without recomputing "
    "the maximum closures, which is what canonical_path already does."
)


def pseudoflow_available() -> bool:
    """True if the optional ``pseudoflow`` package can be imported."""
    try:
        import pseudoflow  # noqa: F401
    except Exception:
        return False
    return True


def counterexample_instance() -> Instance:
    """The smallest instance on which ``pseudoflow`` misses breakpoints.

    Three items, no precedence arcs, ``p = (3, 2, 1)``, ``w = (1, 1, 1)``; the
    canonical sequence is ``{0} | {1} | {2}`` with ratios 3, 2, 1.
    """
    inst = Instance(np.array([3.0, 2.0, 1.0]), np.array([1.0, 1.0, 1.0]),
                    np.zeros((0, 2), np.int64), name="pseudoflow_counterexample")
    inst.validate()
    return inst


# ------------------------------------------------------------------ network
@dataclasses.dataclass
class ParametricNetwork:
    """The reduced parametric network handed to ``pseudoflow.hpf``.

    ``lam(t) = Lambda - t`` maps the package's parameter back to the weight
    price of the paper; the useful range is ``t in [0, t_max]``.
    """
    graph: object                # igraph.Graph, attributes "const" and "mult"
    source: int
    sink: int
    Lambda: float                # lambda at t = 0 (above every ratio)
    lambda_lo: float             # lambda at t = t_max (below every ratio)
    K: float                     # constant shift making every capacity >= 0
    big: float                   # capacity of the precedence arcs

    @property
    def t_max(self) -> float:
        return self.Lambda - self.lambda_lo

    def lam(self, t: float) -> float:
        return self.Lambda - t


def build_parametric_network(inst: Instance,
                             lambda_range: Optional[Tuple[float, float]] = None
                             ) -> ParametricNetwork:
    """Build the parametric network of the module docstring.

    ``lambda_range`` is ``(lambda_lo, lambda_hi)`` in the *weight price* of the
    paper, not in the package's parameter; it must bracket every breakpoint
    strictly.  The default is ``(min_i p_i/w_i - 1, max_i p_i/w_i + 1)`` rounded
    outwards to integers, which is always valid because every ratio
    ``p(I)/w(I)`` is a ``w``-weighted average of the item ratios.
    """
    import igraph as ig

    n = inst.n
    if n == 0:
        raise ValueError("empty instance")
    p, w = np.asarray(inst.p, dtype=float), np.asarray(inst.w, dtype=float)
    ratio = p / w
    if lambda_range is None:
        lam_hi = float(math.floor(float(ratio.max())) + 1.0)
        lam_lo = float(math.ceil(float(ratio.min())) - 1.0)
    else:
        lam_lo, lam_hi = float(lambda_range[0]), float(lambda_range[1])
        if not lam_lo < float(ratio.min()) or not lam_hi > float(ratio.max()):
            raise ValueError("lambda_range must strictly bracket all item ratios "
                             f"[{ratio.min()}, {ratio.max()}]")
    # K: source arcs need K >= -min(p); sink arcs need lambda*w_i + K >= 0 down
    # to lambda = lam_lo, i.e. K >= -lam_lo * max(w) when lam_lo < 0.
    K = max(0.0, float(-p.min()), -lam_lo * float(w.max()) if lam_lo < 0 else 0.0)
    cap_s = p + K
    cap_t0 = lam_hi * w + K                       # constant term of the sink arcs
    if cap_s.min() < 0 or cap_t0.min() < 0 or (lam_lo * w + K).min() < 0:
        raise AssertionError("negative capacity in the reduction")   # pragma: no cover
    big = float(cap_s.sum()) + 1.0

    s, t = n, n + 1
    G = ig.Graph(directed=True)
    G.add_vertices(n + 2)
    edges = ([(s, i) for i in range(n)] + [(i, t) for i in range(n)]
             + [(int(a), int(b)) for a, b in inst.arcs])
    G.add_edges(edges)
    G.es["const"] = ([float(x) for x in cap_s] + [float(x) for x in cap_t0]
                     + [big] * inst.m)
    G.es["mult"] = ([0.0] * n + [float(-x) for x in w] + [0.0] * inst.m)
    return ParametricNetwork(G, s, t, lam_hi, lam_lo, K, big)


def parametric_chain_pseudoflow(inst: Instance,
                                lambda_range: Optional[Tuple[float, float]] = None
                                ) -> Tuple[List[np.ndarray], np.ndarray, dict]:
    """One parametric minimum cut; the nested closures it reports.

    Returns ``(closures, lambdas, info)`` where ``closures[r]`` is the source
    side of the minimum cut on the r-th interval on which the cut changes,
    ``closures[0]`` is empty, ``closures[-1]`` is the whole item set,
    ``lambdas[r]`` is the weight price at which ``closures[r]`` appears, and
    ``info`` is the statistics dictionary of the package.

    .. warning::
       The chain returned by ``pseudoflow`` 2022.12.0 may be a strict
       coarsening of the true one, and then ``lambdas[r]`` is *not* the ratio of
       the r-th increment; see the module docstring.  In particular the source
       set reported for the very first interval may already be non-empty even
       though the range provably starts above every breakpoint -- the empty
       closure is prepended in that case, so that the chain always starts at the
       empty set.
    """
    try:
        import pseudoflow
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("the optional 'pseudoflow' package is not installed "
                          "(pip install pseudoflow; non-commercial licence)") from exc

    net = build_parametric_network(inst, lambda_range)
    breakpoints, cuts, info = pseudoflow.hpf(
        net.graph, net.source, net.sink, const_cap="const", mult_cap="mult",
        lambdaRange=[0.0, net.t_max], roundNegativeCapacity=False)

    n = inst.n
    n_int = len(breakpoints)
    src = np.array([[bool(cuts[i][j]) for j in range(n_int)] for i in range(n)],
                   dtype=bool)                                    # (n, n_int)
    # Interval j runs over (breakpoints[j-1], breakpoints[j]] in t, so the cut
    # reported for it appears at t = breakpoints[j-1], i.e. lambda = Lambda - t.
    closures: List[np.ndarray] = []
    lambdas: List[float] = []
    prev = np.zeros(n, dtype=bool)
    for j in range(n_int):
        cur = src[:, j]
        if (prev & ~cur).any():
            raise RuntimeError("pseudoflow returned a non-monotone family of cuts")
        if (cur & ~prev).any() or j == 0:
            closures.append(cur.copy())
            lambdas.append(net.Lambda if j == 0 else net.lam(breakpoints[j - 1]))
        prev = cur
    if closures and closures[0].any():
        # The range starts strictly above every breakpoint, so the minimum cut
        # there is empty; a non-empty first cut is the package's missing-
        # breakpoint bug, not a range problem.  Keep the chain well formed.
        closures.insert(0, np.zeros(n, dtype=bool))
        lambdas.insert(0, net.Lambda)
    if not prev.all():
        raise RuntimeError("the parameter range does not reach below the smallest "
                           "breakpoint (last reported cut is not the whole set)")
    return closures, np.array(lambdas, dtype=float), info


# --------------------------------------------------------------- public path
def canonical_path_pseudoflow(inst: Instance,
                              lambda_range: Optional[Tuple[float, float]] = None,
                              *, allow_incorrect: bool = False) -> MacroitemPath:
    """The canonical macroitem sequence by one parametric minimum cut.

    **Disabled**: ``pseudoflow`` 2022.12.0 misses breakpoints (module
    docstring), so this raises :class:`NotImplementedError` unless
    ``allow_incorrect=True``, which returns the package's answer as it is --
    a chain that is nested, complete and closed, but possibly a strict
    coarsening of the canonical sequence.  It exists so that the tests can pin
    the bug down and so that the method can be re-enabled without rewriting the
    reduction if the package is ever fixed.

    Use :func:`~macroitems.path.canonical_path` for real work.
    """
    if not allow_incorrect:
        raise NotImplementedError(
            "canonical_path(method='pseudoflow') is disabled: " + PSEUDOFLOW_MISSING_BREAKPOINTS
            + " Pass allow_incorrect=True to obtain the package's answer anyway, "
              "and see macroitems/pseudoflow_path.py for the full analysis. "
              "Note also that, even when it agrees, the package must not be quoted "
              "as a measurement of parametric HPF: " + README_CAVEAT)

    t0 = time.perf_counter()
    closures, _, _ = parametric_chain_pseudoflow(inst, lambda_range)
    macro: List[np.ndarray] = []
    prev = np.zeros(inst.n, dtype=bool)
    for cur in closures:
        inc = np.flatnonzero(cur & ~prev)
        if inc.size:
            macro.append(inc)
        prev = cur
    # The ratios are recomputed from the instance data, never read off the
    # breakpoints returned by the package (Section A5 of the plan).
    ratios = np.array([inst.p[I].sum() / inst.w[I].sum() for I in macro])
    P = np.concatenate([[0.0], np.cumsum([inst.p[I].sum() for I in macro])])
    W = np.concatenate([[0.0], np.cumsum([inst.w[I].sum() for I in macro])])
    return MacroitemPath(macro, ratios, P, W, 1, time.perf_counter() - t0, "pseudoflow")
