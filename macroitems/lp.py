"""Baseline: the LP relaxation solved by a general-purpose LP solver (HiGHS
through scipy.optimize.linprog).  Used for verification and timing."""
from __future__ import annotations

import dataclasses
import time
from typing import Optional

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import csr_matrix, vstack

from .instance import Instance


@dataclasses.dataclass
class LPResult:
    x: np.ndarray
    value: float
    lam: float             # dual of the capacity constraint (>= 0)
    seconds: float
    status: int
    message: str


def _constraints(inst: Instance):
    n, m = inst.n, inst.m
    rows = [csr_matrix(inst.w.reshape(1, -1))]
    if m:
        data = np.concatenate([np.ones(m), -np.ones(m)])
        r = np.concatenate([np.arange(m), np.arange(m)])
        col = np.concatenate([inst.arcs[:, 0], inst.arcs[:, 1]])
        rows.append(csr_matrix((data, (r, col)), shape=(m, n)))
    return vstack(rows).tocsr()


def solve_lp(inst: Instance, c: float, method: str = "highs", A_ub=None, time_limit: Optional[float] = None) -> LPResult:
    t0 = time.perf_counter()
    A = _constraints(inst) if A_ub is None else A_ub
    b = np.concatenate([[c], np.zeros(inst.m)])
    opts = {}
    if time_limit is not None:
        opts["time_limit"] = time_limit
    res = linprog(-inst.p, A_ub=A, b_ub=b, bounds=(0, 1), method=method, options=opts)
    dt = time.perf_counter() - t0
    if res.status != 0:
        return LPResult(np.full(inst.n, np.nan), np.nan, np.nan, dt, res.status, res.message)
    lam = float(-res.ineqlin.marginals[0]) if hasattr(res, "ineqlin") else np.nan
    return LPResult(res.x, float(-res.fun), lam, dt, res.status, res.message)
