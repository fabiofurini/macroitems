"""Optimal faces of the LP relaxation at a nondegenerate capacity
(w(M_{h-1}) < c < w(M_h)), following the companion note:

  * primal face:  dim X*(c) = k_0 - 1, where k_0 is the number of
    inseparability classes of the split macroitem H under the intermediate
    lambda_h-optimal closures (the "tight" relatively closed subsets T of H
    with sum_{i in T} (p_i - lambda_h w_i) = 0);
  * dual face:    dim D*(c) = |A(F)| + |A(Z)| + |A(H) \\ E_0| - |H| + c(H_0),
    where E_0 is the set of arcs of A(H) entering a tight subset and c(H_0)
    the number of connected components of (H, A(H) \\ E_0).

Tight sets are found by minimum cuts: for each i in H, the minimal tight set
T_i containing i is the inclusion-minimal optimal closure of the residual
graph with i forced in.  Items i, j are inseparable iff j in T_i and i in T_j;
an arc (i, j) of A(H) enters a tight set iff i not in T_j.
"""
from __future__ import annotations

import dataclasses
import time

import numpy as np
from math import gcd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

from .instance import Instance
from .closure import ClosureSolver
from .path import LPSolution


@dataclasses.dataclass
class FaceInfo:
    k0: int                 # inseparability classes of H
    dim_primal: int         # k0 - 1
    dim_dual: int
    n_tight_proper: int     # number of distinct proper tight subsets T_i (i in H)
    arcs_F: int
    arcs_Z: int
    arcs_H: int
    E0: int
    components_H0: int
    n_maxflow: int
    seconds: float


def _is_integral(a: np.ndarray) -> bool:
    return bool(np.all(a == np.rint(a)))


def face_dimensions(inst: Instance, sol: LPSolution, max_H: int = 20000) -> FaceInfo:
    t0 = time.perf_counter()
    F, H, Z = sol.F, sol.H, sol.Z
    a = inst.arcs
    inF = F[a[:, 0]] & F[a[:, 1]] if inst.m else np.zeros(0, bool)
    inZ = Z[a[:, 0]] & Z[a[:, 1]] if inst.m else np.zeros(0, bool)
    inH = H[a[:, 0]] & H[a[:, 1]] if inst.m else np.zeros(0, bool)
    nodes = np.flatnonzero(H)
    k = nodes.size
    if k > max_H:
        raise ValueError(f"|H| = {k} too large for the tight-set computation (max_H={max_H})")
    sub, _ = inst.induced(nodes)
    # The tight sets are the same under any positive scaling of the node
    # values, and on integer data lambda_h is the rational p(H)/w(H), so
    # multiplying by w(H) makes v_i = w(H) p_i - p(H) w_i an integer.  That
    # keeps the computation exact and lets an integer maximum-flow backend
    # carry it; the floating-point form would need the optional igraph one.
    den = 1.0
    if _is_integral(inst.p) and _is_integral(inst.w):
        num_h, den_h = float(inst.p[H].sum()), float(inst.w[H].sum())
        if den_h != 0 and abs(num_h / den_h - sol.lam) <= 1e-9 * max(1.0, abs(sol.lam)):
            g = gcd(int(round(abs(num_h))), int(round(abs(den_h)))) or 1
            den = abs(den_h) / g
    v = np.rint(sub.p * den - (sol.lam * den) * sub.w) if den != 1.0 else sub.p - sol.lam * sub.w
    solver = ClosureSolver(sub)
    # minimal tight set containing each item
    T = np.zeros((k, k), dtype=bool) if k <= 4000 else None
    T_list = []
    tol = 1e-7 * max(1.0, float(np.abs(v).sum()))
    n_mf = 0
    for i in range(k):
        force = np.zeros(k, dtype=bool)
        force[i] = True
        res = solver.solve(v, tie="min", force_in=force)
        n_mf += 1
        if abs(res.value) <= tol and res.closure.size < k:
            mask = res.mask
        else:
            mask = np.ones(k, dtype=bool)     # only H itself
        T_list.append(mask)
        if T is not None:
            T[i] = mask
    Tm = np.array(T_list)                      # Tm[i, j] = j in T_i
    # inseparability classes: i ~ j iff Tm[i, j] and Tm[j, i]
    mutual = Tm & Tm.T
    ncomp, _ = connected_components(csr_matrix(mutual), directed=False)
    k0 = int(ncomp)
    # E_0: arcs (i, j) in A(H) with i not in T_j
    sub_arcs = sub.arcs
    if sub_arcs.shape[0]:
        enters = ~Tm[sub_arcs[:, 1], sub_arcs[:, 0]]
    else:
        enters = np.zeros(0, dtype=bool)
    E0 = int(enters.sum())
    keep = sub_arcs[~enters]
    G = csr_matrix((np.ones(keep.shape[0]), (keep[:, 0], keep[:, 1])), shape=(k, k)) if keep.shape[0] else csr_matrix((k, k))
    cH0, _ = connected_components(G, directed=False)
    proper = {tuple(np.flatnonzero(mask)) for mask in T_list if mask.sum() < k}
    dim_dual = int(inF.sum()) + int(inZ.sum()) + int(inH.sum()) - E0 - k + int(cH0)
    return FaceInfo(k0, k0 - 1, dim_dual, len(proper), int(inF.sum()), int(inZ.sum()), int(inH.sum()), E0, int(cH0),
                    n_mf, time.perf_counter() - t0)
