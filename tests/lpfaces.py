"""Reference dimensions of the optimal faces, computed with an LP solver.

The dimension of a face ``{x : Ex = f, Mx <= q}`` is ``dim - rank`` of the
equality system formed by the objective row together with every inequality that
is tight on the whole face.  An inequality is tight on the whole face iff
maximizing its left-hand side over the face does not reach the right-hand side
strictly, which is one small LP per row.  Slow but completely independent of the
combinatorial construction in :mod:`macroitems.faces`, which is the point.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import linprog

TOL = 1e-7


def _face_dimension(M, q, obj, z, nv):
    tight = [M[r] for r in range(M.shape[0])
             if q[r] - linprog(M[r], A_ub=M, b_ub=q, A_eq=obj.reshape(1, -1), b_eq=[z],
                               bounds=[(None, None)] * nv, method="highs").fun < TOL]
    E = np.vstack([obj] + tight) if tight else obj.reshape(1, -1)
    return nv - np.linalg.matrix_rank(E, tol=1e-8)


def primal_face_dimension_lp(n, p, w, arcs, c, z):
    """dim of ``{x : p'x = z, w'x <= c, x_i <= x_j, 0 <= x <= 1}``."""
    rows = [list(map(float, w))]
    rhs = [float(c)]
    for (i, j) in arcs:
        row = [0.0] * n
        row[i], row[j] = 1.0, -1.0
        rows.append(row)
        rhs.append(0.0)
    M = np.vstack([np.array(rows), np.eye(n), -np.eye(n)])
    q = np.concatenate([np.array(rhs), np.ones(n), np.zeros(n)])
    return _face_dimension(M, q, np.array(p, float), z, n)


def dual_face_dimension_lp(n, p, w, arcs, c, z):
    """dim of the optimal face of the dual ``min c*lam + sum(mu)`` subject to
    ``lam w_i + mu_i + div_i(alpha) >= p_i`` with ``lam, mu, alpha >= 0``."""
    m = len(arcs)
    nv = 1 + n + m
    rows = []
    for i in range(n):
        row = np.zeros(nv)
        row[0] = w[i]
        row[1 + i] = 1.0
        for k, (a, b) in enumerate(arcs):
            if a == i:
                row[1 + n + k] += 1.0
            if b == i:
                row[1 + n + k] -= 1.0
        rows.append(row)
    G = np.array(rows).reshape(n, nv)
    obj = np.zeros(nv)
    obj[0] = c
    obj[1:1 + n] = 1.0
    M = np.vstack([-G, -np.eye(nv)])
    q = np.concatenate([-np.array(p, float), np.zeros(nv)])
    return _face_dimension(M, q, obj, z, nv)
