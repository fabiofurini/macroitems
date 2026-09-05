"""Maximum closure by one minimum cut (Picard 1976), with explicit control of
the tie convention.

Network: source s, sink t, one node per item.  For node values v_i:

    s -> i  with capacity  v_i     if v_i > 0
    i -> t  with capacity -v_i     if v_i < 0
    i -> j  with capacity  B       for every arc (i, j) (j prerequisite of i),

with B larger than the total positive value, so that a minimum cut cannot
separate i (source side) from j (sink side).  A closure is the source side
minus s, and max_C v(C) = sum(v_i > 0) - mincut (Picard's cut identity,
Proposition A.1 of the paper).

Optimal closures form a lattice (Proposition 3.1 of the paper), so there are a
unique inclusion-wise minimal and a unique inclusion-wise maximal optimal
closure.  They are read off the residual graph: the maximal one is the
complement of the nodes that reach t, the minimal one is the set of nodes
reachable from s.  The paper's convention is the maximal one (``tie="max"``).

Items can be forced into or out of the closure (``force_in``/``force_out``),
which is what the best-reduced-cost computation of the companion note needs.
"""
from __future__ import annotations

import dataclasses
import time
from typing import Literal, Optional

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import breadth_first_order

from ._maxflow import MaxFlowNetwork, default_backend
from .instance import Instance

Tie = Literal["max", "min"]


@dataclasses.dataclass
class ClosureResult:
    closure: np.ndarray       # sorted item indices
    value: float              # max_C sum_{i in C} v_i
    mask: np.ndarray          # bool, shape (n,)
    flow_value: float
    n_maxflow: int = 1


class ClosureSolver:
    """Reusable Picard network for one instance.

    Only the node capacities change between calls, so the arc list is built
    once.  ``backend`` selects the maximum-flow implementation (see
    :mod:`macroitems._maxflow`); the default is the exact integer one when
    available.
    """

    def __init__(self, inst: Instance, backend: Optional[str] = None):
        self.inst = inst
        n = inst.n
        self.n = n
        self.s, self.t = n, n + 1
        # arcs: precedence arcs first, then s->i for all i, then i->t for all i
        prec = inst.arcs
        e_src = np.concatenate([prec[:, 0], np.full(n, self.s), np.arange(n)])
        e_dst = np.concatenate([prec[:, 1], np.arange(n), np.full(n, self.t)])
        self.m = prec.shape[0]
        self.edges_src = e_src.astype(np.int64)
        self.edges_dst = e_dst.astype(np.int64)
        self.net = MaxFlowNetwork(n + 2, self.edges_src, self.edges_dst, backend=backend)
        self._float_net: Optional[MaxFlowNetwork] = None
        self.calls = 0
        self.time = 0.0

    @property
    def backend(self) -> str:
        return self.net.backend

    def _network_for(self, exact: bool) -> MaxFlowNetwork:
        """The integer backend on integer data; a floating-point one otherwise.

        The algorithms of the paper scale every parametric value to integers
        (v = b p - a w), so the exact backend serves them; a caller passing
        genuine floating-point values transparently gets igraph instead.
        """
        if exact or not self.net.is_exact:
            return self.net
        if self._float_net is None:
            from ._maxflow import available_backends
            if "igraph" not in available_backends():
                raise RuntimeError(
                    "floating-point node values need the igraph backend; "
                    "install python-igraph or pass integer values")
            self._float_net = MaxFlowNetwork(self.n + 2, self.edges_src, self.edges_dst,
                                             backend="igraph")
        return self._float_net

    def solve(self, v: np.ndarray, tie: Tie = "max", force_in: Optional[np.ndarray] = None,
              force_out: Optional[np.ndarray] = None) -> ClosureResult:
        """Maximum closure for node values ``v`` (shape ``(n,)``).

        ``force_in`` / ``force_out`` are bool masks of items forced into or out
        of the closure, realized by infinite-capacity terminal arcs.
        """
        t0 = time.perf_counter()
        n = self.n
        v = np.asarray(v)
        net = self._network_for(_is_integral(v))
        exact = net.is_exact
        v_arr = np.rint(v).astype(np.int64) if exact else np.asarray(v, dtype=float)
        big = 4 * int(np.abs(v_arr).sum()) + 1 if exact else 4.0 * float(np.abs(v_arr).sum()) + 1.0
        zero = v_arr.dtype.type(0)
        cap_s = np.where(v_arr > 0, v_arr, zero)
        cap_t = np.where(v_arr < 0, -v_arr, zero)
        if force_in is not None:
            cap_s = np.where(force_in, big, cap_s)
            cap_t = np.where(force_in, zero, cap_t)
        if force_out is not None:
            cap_t = np.where(force_out, big, cap_t)
            cap_s = np.where(force_out, zero, cap_s)
        caps = np.concatenate([np.full(self.m, big, dtype=cap_s.dtype), cap_s, cap_t])
        flow_value, flow = net.solve(caps, self.s, self.t)

        # residual graph: forward arc if cap - flow > eps, backward arc if flow > eps.
        # With integer capacities the comparison is exact and eps = 0 works.
        eps = 0 if exact else 1e-9 * max(1.0, float(np.abs(v_arr).max(initial=0.0)))
        fwd = (caps - flow) > eps
        bwd = flow > eps
        r_src = np.concatenate([self.edges_src[fwd], self.edges_dst[bwd]])
        r_dst = np.concatenate([self.edges_dst[fwd], self.edges_src[bwd]])
        N = n + 2
        if tie == "max":
            # nodes that can reach t: BFS from t on the reversed residual graph
            R = csr_matrix((np.ones(r_src.size), (r_dst, r_src)), shape=(N, N))
            reach_t = breadth_first_order(R, self.t, directed=True, return_predecessors=False)
            mask = np.ones(N, dtype=bool)
            mask[reach_t] = False
        else:
            R = csr_matrix((np.ones(r_src.size), (r_src, r_dst)), shape=(N, N))
            reach_s = breadth_first_order(R, self.s, directed=True, return_predecessors=False)
            mask = np.zeros(N, dtype=bool)
            mask[reach_s] = True
        mask = mask[:n]
        value = float(v_arr[mask].sum())
        self.calls += 1
        self.time += time.perf_counter() - t0
        return ClosureResult(np.flatnonzero(mask), value, mask, float(flow_value))


def _is_integral(v: np.ndarray) -> bool:
    if np.issubdtype(np.asarray(v).dtype, np.integer):
        return True
    v = np.asarray(v, dtype=float)
    return bool(np.all(v == np.rint(v)) and np.abs(v).sum() < 2.0 ** 60)


def max_closure(inst: Instance, v: np.ndarray, tie: Tie = "max",
                force_in: Optional[np.ndarray] = None,
                force_out: Optional[np.ndarray] = None,
                backend: Optional[str] = None) -> ClosureResult:
    """One-off maximum closure.

    Builds the network; use :class:`ClosureSolver` directly when many closures
    are computed on the same instance.
    """
    return ClosureSolver(inst, backend=backend).solve(v, tie=tie, force_in=force_in,
                                                      force_out=force_out)


def is_closure(inst: Instance, mask: np.ndarray) -> bool:
    """True if ``mask`` is closed under prerequisites: i in C and (i, j) in A imply j in C."""
    if inst.m == 0:
        return True
    a = inst.arcs
    return bool(np.all(~mask[a[:, 0]] | mask[a[:, 1]]))
