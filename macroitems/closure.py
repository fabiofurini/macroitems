"""Maximum closure by one minimum cut (Picard 1976), on top of igraph's
push-relabel maximum flow, with explicit control of the tie convention.

Network: source s, sink t, one node per item.  For node values v_i:
    s -> i with capacity v_i     if v_i > 0
    i -> t with capacity -v_i    if v_i < 0
    i -> j with a large capacity for every arc (i, j) (j prerequisite of i),
so that a finite cut cannot separate i (source side) from j (sink side).
Closure = source side minus s;  max_C v(C) = sum(v_i > 0) - mincut.

Optimal closures form a lattice.  We return either the inclusion-wise maximal
optimal closure (complement of the nodes that can reach t in the residual
graph) or the minimal one (nodes reachable from s in the residual graph).
"""
from __future__ import annotations

import dataclasses
from typing import Literal, Optional

import numpy as np
import igraph as ig
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import breadth_first_order

from .instance import Instance

Tie = Literal["max", "min"]


@dataclasses.dataclass
class ClosureResult:
    closure: np.ndarray       # sorted item indices (bool mask available via .mask)
    value: float              # max_C sum_{i in C} v_i
    mask: np.ndarray          # bool, shape (n,)
    flow_value: float
    n_maxflow: int = 1


class ClosureSolver:
    """Reusable Picard network for one instance; only node capacities change
    between calls, so the igraph object is built once."""

    def __init__(self, inst: Instance):
        self.inst = inst
        n = inst.n
        self.n = n
        self.s, self.t = n, n + 1
        # edge list: precedence arcs first, then s->i for all i, then i->t for all i
        prec = inst.arcs
        e_src = np.concatenate([prec[:, 0], np.full(n, self.s), np.arange(n)])
        e_dst = np.concatenate([prec[:, 1], np.arange(n), np.full(n, self.t)])
        self.m = prec.shape[0]
        self.edges_src = e_src.astype(np.int64)
        self.edges_dst = e_dst.astype(np.int64)
        self.g = ig.Graph(n=n + 2, edges=list(zip(e_src.tolist(), e_dst.tolist())), directed=True)
        self.calls = 0
        self.time = 0.0

    def solve(self, v: np.ndarray, tie: Tie = "max", force_in: Optional[np.ndarray] = None,
              force_out: Optional[np.ndarray] = None) -> ClosureResult:
        """Maximum closure for node values v (shape (n,)).
        force_in / force_out: bool masks of items forced into / out of the closure."""
        import time as _t
        t0 = _t.perf_counter()
        n = self.n
        v = np.asarray(v, dtype=float)
        big = 4.0 * float(np.abs(v).sum()) + 1.0
        cap_s = np.where(v > 0, v, 0.0)
        cap_t = np.where(v < 0, -v, 0.0)
        if force_in is not None:
            cap_s = np.where(force_in, big, cap_s)
            cap_t = np.where(force_in, 0.0, cap_t)
        if force_out is not None:
            cap_t = np.where(force_out, big, cap_t)
            cap_s = np.where(force_out, 0.0, cap_s)
        caps = np.concatenate([np.full(self.m, big), cap_s, cap_t])
        fl = self.g.maxflow(self.s, self.t, capacity=caps.tolist())
        flow = np.asarray(fl.flow, dtype=float)
        # residual graph: forward residual cap - flow, backward residual flow
        eps = 1e-9 * max(1.0, float(np.abs(v).max(initial=0.0)))
        fwd = caps - flow > eps
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
        value = float(v[mask].sum())
        self.calls += 1
        self.time += _t.perf_counter() - t0
        return ClosureResult(np.flatnonzero(mask), value, mask, float(fl.value))


def max_closure(inst: Instance, v: np.ndarray, tie: Tie = "max") -> ClosureResult:
    """One-off maximum closure (builds the network; use ClosureSolver for many calls)."""
    return ClosureSolver(inst).solve(v, tie=tie)


def is_closure(inst: Instance, mask: np.ndarray) -> bool:
    """True if mask is closed under prerequisites: i in C and (i, j) in A imply j in C."""
    if inst.m == 0:
        return True
    a = inst.arcs
    return bool(np.all(~mask[a[:, 0]] | mask[a[:, 1]]))
