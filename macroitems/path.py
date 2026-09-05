"""The canonical macroitem sequence (nested maximal optimal closures), the LP
relaxation at a given capacity, and the canonical dual certificate.

Notation follows the paper: for a weight price lambda the node values are
v_i = p_i - lambda * w_i; M^max(lambda) is the inclusion-wise maximal optimal
closure; the breakpoints lambda_1 > ... > lambda_k are the ratios of the
consecutive increments (macroitems) I_r = M_r \\ M_{r-1}.
"""
from __future__ import annotations

import dataclasses
import time
from fractions import Fraction
from typing import List, Optional

import numpy as np
import igraph as ig

from .instance import Instance
from .closure import ClosureSolver, ClosureResult


# ----------------------------------------------------------------- results
@dataclasses.dataclass
class MacroitemPath:
    macroitems: List[np.ndarray]       # I_1, ..., I_k as sorted index arrays
    ratios: np.ndarray                 # lambda_r = p(I_r)/w(I_r), strictly decreasing
    P: np.ndarray                      # cumulative profits p(M_r), r = 0..k  (P[0] = 0)
    W: np.ndarray                      # cumulative weights w(M_r), r = 0..k  (W[0] = 0)
    n_maxflow: int
    seconds: float
    method: str

    @property
    def k(self) -> int:
        return len(self.macroitems)

    @property
    def q(self) -> int:
        """Index of the last macroitem with positive ratio (M_q = maximal max-profit closure)."""
        pos = np.flatnonzero(self.ratios > 0)
        return int(pos[-1] + 1) if pos.size else 0

    def closure_mask(self, n: int, r: int) -> np.ndarray:
        mask = np.zeros(n, dtype=bool)
        for I in self.macroitems[:r]:
            mask[I] = True
        return mask

    def level_of_item(self, n: int) -> np.ndarray:
        lev = np.zeros(n, dtype=np.int64)
        for r, I in enumerate(self.macroitems, 1):
            lev[I] = r
        return lev

    def value_function(self, c: float) -> float:
        """z(c) = optimal value of the LP relaxation with capacity c."""
        q = self.q
        if q == 0 or c <= 0:
            return 0.0
        if c >= self.W[q]:
            return float(self.P[q])
        h = int(np.searchsorted(self.W[:q + 1], c, side="left"))  # W[h-1] < c <= W[h]
        return float(self.P[h - 1] + self.ratios[h - 1] * (c - self.W[h - 1]))

    def split_index(self, c: float) -> int:
        """h with W[h-1] < c <= W[h] (1-based), for 0 < c < W[q]."""
        return int(np.searchsorted(self.W[:self.q + 1], c, side="left"))

    def check(self, inst: Instance, exact: bool = True) -> dict:
        """Nestedness, closedness of prefixes, strictly decreasing ratios; if the
        data are integers and exact=True, the ratios are recomputed in rational
        arithmetic."""
        n = inst.n
        seen = np.zeros(n, dtype=bool)
        ok_partition = True
        for I in self.macroitems:
            if seen[I].any():
                ok_partition = False
            seen[I] = True
        ok_partition &= bool(seen.all())
        from .closure import is_closure
        ok_closed = all(is_closure(inst, self.closure_mask(n, r)) for r in range(1, self.k + 1))
        if exact and np.all(inst.p == np.round(inst.p)) and np.all(inst.w == np.round(inst.w)):
            fr = [Fraction(int(round(inst.p[I].sum())), int(round(inst.w[I].sum()))) for I in self.macroitems]
            ok_ratios = all(fr[r] > fr[r + 1] for r in range(len(fr) - 1))
        else:
            ok_ratios = bool(np.all(np.diff(self.ratios) < 0))
        return {"partition": ok_partition, "closed_prefixes": ok_closed, "strictly_decreasing": ok_ratios}


@dataclasses.dataclass
class LPSolution:
    x: np.ndarray
    value: float
    lam: float                  # capacity multiplier (lambda_h, or 0 if slack)
    F: np.ndarray               # bool mask: full region  M_{h-1}
    H: np.ndarray               # bool mask: fractional region I_h (empty if integral)
    Z: np.ndarray               # bool mask: null region
    theta: float
    h: Optional[int]            # index of the split macroitem, None when unknown
    n_maxflow: int
    seconds: float
    degenerate: str = ""        # "", "slack", "cumulative"


# ----------------------------------------------------- canonical sequence
def _tol(pD: float, lam: float, wD: float) -> float:
    return 1e-7 * max(1.0, abs(pD) + abs(lam) * wD)


def is_integer_data(inst: Instance) -> bool:
    return bool(np.all(inst.p == np.round(inst.p)) and np.all(inst.w == np.round(inst.w))
                and np.abs(inst.p).sum() * inst.w.sum() < 2.0 ** 52)


def _values_at_ratio(sub: Instance, num: float, den: float, exact: bool):
    """Node values p_i - (num/den) w_i on the sub-instance, scaled to integers
    when the data are integers (then max flow is exact and 'positive' means >= 1);
    returns (v, tol) with tol the threshold for a strictly positive optimum."""
    if exact:
        from math import gcd
        a, b = int(round(num)), int(round(den))
        g = gcd(abs(a), abs(b)) or 1
        a //= g; b //= g
        v = sub.p * b - a * sub.w
        return v, 0.5
    lam = num / den
    return sub.p - lam * sub.w, _tol(num, lam, den)


def canonical_path(inst: Instance, method: str = "bisection", exact: Optional[bool] = None,
                   backend: Optional[str] = None) -> MacroitemPath:
    """Compute the canonical macroitem sequence.

    method="bisection": geometric bisection on the breakpoints (Eisner-Severance
    1976 / Gusfield 1983 style): given nested optimal closures A ⊂ B, solve one
    maximum closure on the residual graph B \\ A at the ratio of the increment;
    a positive value reveals a new breakpoint closure strictly between A and B,
    a zero value certifies that A and B are consecutive.  O(k) max flows on
    shrinking graphs; returns exactly the maximal-tie canonical sequence.

    method="dinkelbach": repeated maximum-ratio closure extraction on the
    residual graph (Sidney / Lawler / Gallo-Grigoriadis-Tarjan p. 49), each by
    Dinkelbach iterations; the maximal optimal closure at the final ratio gives
    the coarsest (maximal-tie) macroitem.
    """
    t0 = time.perf_counter()
    n = inst.n
    n_mf = 0
    if exact is None:
        exact = is_integer_data(inst)
    if method == "bisection":
        out_pairs = []  # leaves (A_mask, B_mask, lam) in decreasing-lambda order
        A0 = np.zeros(n, dtype=bool)
        B0 = np.ones(n, dtype=bool)
        stack = [(A0, B0)]
        while stack:
            A, B = stack.pop()
            D = np.flatnonzero(B & ~A)
            pD, wD = float(inst.p[D].sum()), float(inst.w[D].sum())
            lam = pD / wD
            sub, nodes = inst.induced(D)
            v, tol = _values_at_ratio(sub, pD, wD, exact)
            res = ClosureSolver(sub, backend=backend).solve(v, tie="max")
            n_mf += 1
            if res.value > tol and 0 < res.closure.size < D.size:
                C = A.copy()
                C[nodes[res.closure]] = True
                stack.append((C, B))   # processed after (A, C)
                stack.append((A, C))
            else:
                out_pairs.append((A, B, lam))
        macro = [np.flatnonzero(B & ~A) for A, B, _ in out_pairs]
    elif method == "dinkelbach":
        macro = []
        A = np.zeros(n, dtype=bool)
        while not A.all():
            D = np.flatnonzero(~A)
            sub, nodes = inst.induced(D)
            solver = ClosureSolver(sub, backend=backend)
            num, den = float(sub.p.sum()), float(sub.w.sum())
            C = np.arange(D.size)
            for _ in range(200):
                v, tol = _values_at_ratio(sub, num, den, exact)
                res = solver.solve(v, tie="max")
                n_mf += 1
                if res.value <= tol or res.closure.size == 0:
                    if res.closure.size > 0:
                        C = res.closure       # maximal optimal closure at the max ratio
                    break
                C = res.closure
                num, den = float(sub.p[C].sum()), float(sub.w[C].sum())
            macro.append(np.sort(nodes[C]))
            A[nodes[C]] = True
    else:
        raise ValueError(method)
    ratios = np.array([inst.p[I].sum() / inst.w[I].sum() for I in macro])
    P = np.concatenate([[0.0], np.cumsum([inst.p[I].sum() for I in macro])])
    W = np.concatenate([[0.0], np.cumsum([inst.w[I].sum() for I in macro])])
    path = MacroitemPath(macro, ratios, P, W, n_mf, time.perf_counter() - t0, method)
    return path


# ------------------------------------------------ LP at a given capacity
def solve_capacity(inst: Instance, c: float, solver: Optional[ClosureSolver] = None,
                   exact: Optional[bool] = None, backend: Optional[str] = None) -> LPSolution:
    """Solve the LP relaxation at capacity c by a Newton search on the weight
    price: at most O(k) maximum closures, typically a handful.  Returns the
    canonical primal optimum (1 on M_{h-1}, theta on I_h, 0 elsewhere)."""
    t0 = time.perf_counter()
    n = inst.n
    solver = solver or ClosureSolver(inst, backend=backend)
    n_mf = 0
    if exact is None:
        exact = is_integer_data(inst)
    # maximal maximum-profit closure (lambda = 0)
    res0 = solver.solve(inst.p, tie="max")
    n_mf += 1
    Mq = res0.mask
    if inst.w[Mq].sum() <= c + 1e-12 * max(1.0, c):
        x = Mq.astype(float)
        return LPSolution(x, float(inst.p[Mq].sum()), 0.0, Mq, np.zeros(n, bool), ~Mq, 1.0, None,
                          n_mf, time.perf_counter() - t0, degenerate="slack")
    A = np.zeros(n, dtype=bool)
    B = Mq
    lam = 0.0
    while True:
        D = np.flatnonzero(B & ~A)
        pD, wD = float(inst.p[D].sum()), float(inst.w[D].sum())
        lam = pD / wD
        sub, nodes = inst.induced(D)
        v, tol = _values_at_ratio(sub, pD, wD, exact)
        res = ClosureSolver(sub, backend=backend).solve(v, tie="max")
        n_mf += 1
        if res.value > tol and 0 < res.closure.size < D.size:
            C = A.copy()
            C[nodes[res.closure]] = True
            if inst.w[C].sum() >= c:
                B = C
            else:
                A = C
        else:
            break
    wA = float(inst.w[A].sum())
    theta = (c - wA) / wD
    x = A.astype(float)
    Hmask = np.zeros(n, dtype=bool)
    Hmask[D] = True
    x[D] = theta
    value = float(inst.p[A].sum() + theta * pD)
    Z = ~(A | Hmask)
    deg = "cumulative" if abs(theta - 1.0) < 1e-12 else ""
    # The Newton search jumps between brackets instead of enumerating the
    # macroitems, so the *index* of the split one is not known here -- only the
    # set H is.  solution_from_path, which has the whole path, reports it.
    return LPSolution(x, value, lam, A, Hmask, Z, theta, None, n_mf,
                      time.perf_counter() - t0, degenerate=deg)


def solution_from_path(inst: Instance, path: MacroitemPath, c: float) -> LPSolution:
    """Canonical optimum at capacity c read off a precomputed path (no max flow)."""
    n = inst.n
    q = path.q
    if c <= 0:
        # x = 0 is the only feasible point (weights are positive); split_index
        # would return h = 0 here, and closure_mask(n, h - 1) would wrap around.
        F = np.zeros(n, dtype=bool)
        H = np.zeros(n, dtype=bool)
        lam = 0.0
        if q > 0:
            H[path.macroitems[0]] = True       # the split macroitem as c -> 0+
            lam = float(path.ratios[0])
        return LPSolution(np.zeros(n), 0.0, lam, F, H, ~(F | H), 0.0, int(H.any()), 0, 0.0)
    if q == 0 or c >= path.W[q]:
        F = path.closure_mask(n, q)
        return LPSolution(F.astype(float), float(path.P[q]), 0.0, F, np.zeros(n, bool), ~F, 1.0, q, 0, 0.0, "slack")
    h = path.split_index(c)
    F = path.closure_mask(n, h - 1)
    H = np.zeros(n, dtype=bool)
    H[path.macroitems[h - 1]] = True
    theta = (c - path.W[h - 1]) / (path.W[h] - path.W[h - 1])
    x = F.astype(float)
    x[H] = theta
    value = float(path.P[h - 1] + theta * (path.P[h] - path.P[h - 1]))
    return LPSolution(x, value, float(path.ratios[h - 1]), F, H, ~(F | H), theta, h, 0, 0.0,
                      "cumulative" if abs(theta - 1) < 1e-12 else "")


# ------------------------------------------------- canonical dual certificate
@dataclasses.dataclass
class DualCertificate:
    lam: float
    mu: np.ndarray            # shape (n,)
    alpha: np.ndarray         # shape (m,), indexed like inst.arcs
    value: float              # c*lam + sum(mu)
    feasible: bool
    max_violation: float
    n_maxflow: int


def _region_flow(inst: Instance, region: np.ndarray, b: np.ndarray, kind: str):
    """Flow on the arcs inside `region` (bool mask) with divergence div_i = out - in
    along arcs (i, j).  kind: "eq" (div = b), "le" (div <= b), "ge" (div >= b).
    Sources are nodes with b_i > 0 (supply b_i), sinks nodes with b_i < 0."""
    nodes = np.flatnonzero(region)
    if nodes.size == 0:
        return np.zeros(0), np.zeros(0, dtype=np.int64), True
    sub, _ = inst.induced(nodes)
    arc_idx = np.flatnonzero(region[inst.arcs[:, 0]] & region[inst.arcs[:, 1]]) if inst.m else np.zeros(0, np.int64)
    bs = b[nodes]
    k = nodes.size
    s, t = k, k + 1
    big = 4.0 * float(np.abs(bs).sum()) + 1.0
    e_src = np.concatenate([sub.arcs[:, 0], np.full(k, s), np.arange(k)])
    e_dst = np.concatenate([sub.arcs[:, 1], np.arange(k), np.full(k, t)])
    caps = np.concatenate([np.full(sub.m, big), np.where(bs > 0, bs, 0.0), np.where(bs < 0, -bs, 0.0)])
    g = ig.Graph(n=k + 2, edges=list(zip(e_src.tolist(), e_dst.tolist())), directed=True)
    fl = g.maxflow(s, t, capacity=caps.tolist())
    flow = np.asarray(fl.flow)
    alpha_sub = flow[:sub.m]
    tot_pos, tot_neg = float(bs[bs > 0].sum()), float(-bs[bs < 0].sum())
    tol = 1e-7 * max(1.0, tot_pos + tot_neg)
    if kind == "eq":
        ok = abs(fl.value - tot_pos) <= tol and abs(fl.value - tot_neg) <= tol
    elif kind == "le":
        ok = abs(fl.value - tot_neg) <= tol          # sinks saturated
    else:
        ok = abs(fl.value - tot_pos) <= tol          # sources saturated
    return alpha_sub, arc_idx, ok


def canonical_dual(inst: Instance, sol: LPSolution, c: float) -> DualCertificate:
    """The canonical dual optimum of the paper: lambda = lambda_h, region-wise
    flows (div = b on H, div <= b on F with mu = b - div, div >= b on Z, mu = 0),
    zero on arcs between regions.  Three max flows.  Verifies feasibility."""
    n, m = inst.n, inst.m
    lam = sol.lam
    b = inst.p - lam * inst.w
    alpha = np.zeros(m)
    ok_all = True
    n_mf = 0
    for region, kind in ((sol.F, "le"), (sol.H, "eq"), (sol.Z, "ge")):
        if region.any():
            a_sub, idx, ok = _region_flow(inst, region, b, kind)
            alpha[idx] = a_sub
            ok_all &= ok
            n_mf += 1
    div = np.zeros(n)
    if m:
        np.add.at(div, inst.arcs[:, 0], alpha)
        np.subtract.at(div, inst.arcs[:, 1], alpha)
    mu = np.where(sol.F, b - div, 0.0)
    mu = np.maximum(mu, 0.0) if ok_all else mu
    slack = lam * inst.w + mu + div - inst.p          # must be >= 0
    viol = float(max(0.0, -slack.min(initial=0.0), (-mu).max(initial=0.0), (-alpha).max(initial=0.0), -lam))
    value = float(lam * c + mu.sum())
    feasible = ok_all and viol <= 1e-7 * max(1.0, float(np.abs(inst.p).max()))
    return DualCertificate(lam, mu, alpha, value, feasible, viol, n_mf)


def canonical_reduced_costs(inst: Instance, path: MacroitemPath, h: int) -> np.ndarray:
    """Canonical reduced costs w_i * |lambda_r - lambda_h| for item i in I_r
    (0 on the split macroitem): the knapsack-style penalty of forcing a null
    item in or a full item out, valid for the canonical dual solution."""
    lev = path.level_of_item(inst.n)
    lam_h = path.ratios[h - 1]
    lam_r = np.where(lev > 0, path.ratios[np.maximum(lev - 1, 0)], 0.0)
    return inst.w * np.abs(lam_r - lam_h)
