"""Interchangeable maximum-flow backends.

A backend takes a fixed arc list (built once per network) and, for each call,
a vector of arc capacities; it returns the flow value and the flow on every
arc, from which the residual graph -- and hence the maximal and the minimal
minimum cut -- is read.

Three backends are available:

``igraph``
    ``Graph.maxflow`` (push-relabel), floating-point capacities.
``ortools``
    ``SimpleMaxFlow``, integer (int64) capacities: exact on integer data,
    which is the regime of every instance used in the experiments.  Integer
    capacities are required; non-integer input is rejected rather than
    silently rounded.
``scipy``
    ``scipy.sparse.csgraph.maximum_flow``, also int64.

The default is chosen by :func:`default_backend`: ``ortools`` when available
(exact and fast), otherwise ``igraph``, otherwise ``scipy``.  Backends are
interchangeable: the same maximum closure must come out of any of them on
integer data, which the test suite checks.
"""
from __future__ import annotations

import numpy as np

_AVAILABLE: dict[str, bool] = {}


def _have(name: str) -> bool:
    if name not in _AVAILABLE:
        try:
            if name == "igraph":
                import igraph  # noqa: F401
            elif name == "ortools":
                from ortools.graph.python import max_flow  # noqa: F401
            elif name == "scipy":
                from scipy.sparse.csgraph import maximum_flow  # noqa: F401
            else:
                raise ValueError(f"unknown backend {name!r}")
            _AVAILABLE[name] = True
        except Exception:
            _AVAILABLE[name] = False
    return _AVAILABLE[name]


def available_backends() -> list[str]:
    return [name for name in ("ortools", "igraph", "scipy") if _have(name)]


def default_backend() -> str:
    for name in ("ortools", "igraph", "scipy"):
        if _have(name):
            return name
    raise RuntimeError("no maximum-flow backend available: install ortools, python-igraph or scipy")


class MaxFlowNetwork:
    """A fixed directed network on ``n_nodes`` with arcs ``(src[e], dst[e])``.

    ``solve(capacities, s, t)`` returns ``(flow_value, flow_per_arc)``.  The
    network object is reused across calls, so building the arc list -- the
    expensive part in Python -- happens once per instance.
    """

    def __init__(self, n_nodes: int, src: np.ndarray, dst: np.ndarray, backend: str | None = None):
        self.n_nodes = int(n_nodes)
        self.src = np.ascontiguousarray(src, dtype=np.int64)
        self.dst = np.ascontiguousarray(dst, dtype=np.int64)
        self.n_arcs = self.src.size
        self.backend = backend or default_backend()
        if not _have(self.backend):
            raise RuntimeError(f"maximum-flow backend {self.backend!r} is not installed")
        self._built = None

    @property
    def is_exact(self) -> bool:
        """True if the backend computes with exact integer arithmetic."""
        return self.backend in ("ortools", "scipy")

    def solve(self, capacities: np.ndarray, s: int, t: int) -> tuple[float, np.ndarray]:
        cap = np.asarray(capacities)
        if self.backend == "igraph":
            return self._solve_igraph(np.asarray(cap, dtype=float), s, t)
        cap_int = _as_int64(cap, self.backend)
        if self.backend == "ortools":
            return self._solve_ortools(cap_int, s, t)
        return self._solve_scipy(cap_int, s, t)

    # ------------------------------------------------------------- backends
    def _solve_igraph(self, cap, s, t):
        import igraph as ig
        if self._built is None:
            self._built = ig.Graph(n=self.n_nodes,
                                   edges=list(zip(self.src.tolist(), self.dst.tolist())),
                                   directed=True)
        fl = self._built.maxflow(s, t, capacity=cap.tolist())
        return float(fl.value), np.asarray(fl.flow, dtype=float)

    def _solve_ortools(self, cap, s, t):
        from ortools.graph.python import max_flow
        solver = max_flow.SimpleMaxFlow()
        solver.add_arcs_with_capacity(self.src, self.dst, cap)
        status = solver.solve(int(s), int(t))
        if status != solver.OPTIMAL:
            raise RuntimeError(f"ortools max flow status {status}")
        return float(solver.optimal_flow()), np.array(
            [solver.flow(e) for e in range(self.n_arcs)], dtype=np.int64)

    def _solve_scipy(self, cap, s, t):
        from scipy.sparse import csr_matrix
        from scipy.sparse.csgraph import maximum_flow
        # scipy needs one entry per (u, v) pair: parallel arcs are merged and
        # the resulting flow is split back over them proportionally to capacity.
        key = self.src * self.n_nodes + self.dst
        uniq, inverse = np.unique(key, return_inverse=True)
        merged = np.zeros(uniq.size, dtype=np.int64)
        np.add.at(merged, inverse, cap)
        g = csr_matrix((merged, (uniq // self.n_nodes, uniq % self.n_nodes)),
                       shape=(self.n_nodes, self.n_nodes), dtype=np.int64)
        res = maximum_flow(g, int(s), int(t))
        fmat = res.flow.tocsr()
        merged_flow = np.asarray(
            fmat[uniq // self.n_nodes, uniq % self.n_nodes]).ravel().astype(np.int64)
        # split the merged flow back over parallel arcs, capacity first
        flow = np.zeros(self.n_arcs, dtype=np.int64)
        remaining = merged_flow.copy()
        order = np.argsort(inverse, kind="stable")
        for e in order:
            g_id = inverse[e]
            take = min(int(cap[e]), int(remaining[g_id]))
            flow[e] = take
            remaining[g_id] -= take
        return float(res.flow_value), flow


def _as_int64(cap: np.ndarray, backend: str) -> np.ndarray:
    """Integer capacities for the exact backends, refusing to round silently."""
    if np.issubdtype(cap.dtype, np.integer):
        return np.ascontiguousarray(cap, dtype=np.int64)
    rounded = np.rint(cap)
    if not np.all(np.abs(cap - rounded) <= 1e-9 * np.maximum(1.0, np.abs(cap))):
        raise ValueError(
            f"backend {backend!r} needs integer capacities but got non-integer values; "
            "use the igraph backend for floating-point data")
    if np.abs(rounded).max(initial=0.0) >= 2.0 ** 62:
        raise ValueError(f"capacities too large for int64 in backend {backend!r}")
    return np.ascontiguousarray(rounded, dtype=np.int64)
