"""Structural statistics of an instance in the macroitem language, and a simple
LP-based integer heuristic used only to bound the integrality gap from above."""
from __future__ import annotations

import time
from typing import Iterable, List, Optional

import numpy as np

from .instance import Instance
from .closure import ClosureSolver
from .path import MacroitemPath, solution_from_path, canonical_dual
from .faces import face_dimensions


def heuristic_integer(inst: Instance, path: MacroitemPath, c: float) -> float:
    """Integer feasible value: M_{h-1} plus a greedy fill of the split macroitem
    (items in ratio order whose prerequisites are already selected)."""
    sol = solution_from_path(inst, path, c)
    if sol.degenerate == "slack" or not sol.H.any():
        return float(inst.p[sol.F].sum())
    sel = sol.F.copy()
    cap = c - float(inst.w[sel].sum())
    D = np.flatnonzero(sol.H)
    inD = sol.H
    # prerequisites inside D
    a = inst.arcs
    mask = inD[a[:, 0]] & inD[a[:, 1]] if inst.m else np.zeros(0, bool)
    arcsD = a[mask]
    npre = np.zeros(inst.n, dtype=np.int64)
    if arcsD.shape[0]:
        np.add.at(npre, arcsD[:, 0], 1)
    order_idx = np.argsort(arcsD[:, 1], kind="stable") if arcsD.shape[0] else np.zeros(0, np.int64)
    heads = arcsD[order_idx, 1] if arcsD.shape[0] else np.zeros(0, np.int64)
    tails = arcsD[order_idx, 0] if arcsD.shape[0] else np.zeros(0, np.int64)
    starts = np.searchsorted(heads, np.arange(inst.n + 1))
    ratio = inst.p / inst.w
    ready = set(int(i) for i in D if npre[i] == 0)
    best = float(inst.p[sel].sum())
    cur = best
    while ready:
        i = max(ready, key=lambda j: ratio[j])
        ready.discard(i)
        if inst.w[i] > cap:
            continue
        if inst.p[i] <= 0 and ratio[i] < 0:
            # taking a negative item can unlock dependents; allow it only if some
            # dependent has positive profit (cheap look-ahead)
            deps = tails[starts[i]:starts[i + 1]]
            if not np.any(inst.p[deps] > 0):
                continue
        sel[i] = True
        cap -= inst.w[i]
        cur += inst.p[i]
        best = max(best, cur)
        for j in tails[starts[i]:starts[i + 1]]:
            npre[j] -= 1
            if npre[j] == 0:
                ready.add(int(j))
    return best


def structural_stats(inst: Instance, path: MacroitemPath, fractions: Iterable[float] = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
                     faces_max_H: int = 3000, with_dual: bool = True) -> List[dict]:
    """One row per capacity c = f * w(M_q): split index h, |H|, theta, gap bound
    theta*p(I_h) relative to z_LP, persistency, heuristic gap, face dimensions."""
    rows = []
    q = path.q
    if q == 0:
        return rows
    Wq = path.W[q]
    for f in fractions:
        c = f * Wq
        sol = solution_from_path(inst, path, c)
        h = sol.h
        H = int(sol.H.sum())
        pI = float(path.P[h] - path.P[h - 1])
        gap_bound = sol.theta * pI
        zh = heuristic_integer(inst, path, c)
        row = dict(f=f, c=c, h=h, size_H=H, theta=sol.theta, z_lp=sol.value, lam=sol.lam,
                   gap_bound=gap_bound, gap_bound_rel=gap_bound / sol.value if sol.value else np.nan,
                   z_heur=zh, gap_heur_rel=(sol.value - zh) / sol.value if sol.value else np.nan,
                   persistency=1.0 - H / inst.n, w_H_over_c=float(inst.w[sol.H].sum()) / c)
        if with_dual and not sol.degenerate:
            t0 = time.perf_counter()
            d = canonical_dual(inst, sol, c)
            row.update(dual_feasible=d.feasible, dual_value=d.value, dual_seconds=time.perf_counter() - t0)
        if H <= faces_max_H and not sol.degenerate:
            fi = face_dimensions(inst, sol, max_H=faces_max_H)
            row.update(k0=fi.k0, dim_primal=fi.dim_primal, dim_dual=fi.dim_dual, n_tight=fi.n_tight_proper,
                       faces_seconds=fi.seconds)
        rows.append(row)
    return rows


def path_summary(inst: Instance, path: MacroitemPath) -> dict:
    sizes = np.array([I.size for I in path.macroitems])
    weights = np.array([inst.w[I].sum() for I in path.macroitems])
    q = path.q
    return dict(n=inst.n, m=inst.m, k=path.k, q=q, size_min=int(sizes.min()), size_median=float(np.median(sizes)),
                size_max=int(sizes.max()), n_singletons=int((sizes == 1).sum()),
                largest_share_of_Wq=float(weights[:q].max() / path.W[q]) if q else np.nan,
                lambda_1=float(path.ratios[0]), lambda_q=float(path.ratios[q - 1]) if q else np.nan,
                p_Mq=float(path.P[q]) if q else 0.0, w_Mq=float(path.W[q]) if q else 0.0,
                n_maxflow=path.n_maxflow, seconds=path.seconds)


def revenue_factor_family(inst: Instance, path: MacroitemPath, r: np.ndarray, k: np.ndarray,
                          factors: Iterable[float] = tuple(np.linspace(0.05, 1.0, 20))) -> dict:
    """Compare the nested pits of the revenue-factor parameterization (values
    f*r_i - k_i) with the weight-parameterization family M_1 ⊂ ... ⊂ M_k.
    Returns how many revenue-factor pits coincide with some M_r, and the average
    relative symmetric difference with the weight-family closure of nearest weight."""
    solver = ClosureSolver(inst)
    fam = {}
    n = inst.n
    masks = [path.closure_mask(n, rr) for rr in range(path.k + 1)]
    keyset = {m.tobytes(): rr for rr, m in enumerate(masks)}
    W = np.array([inst.w[m].sum() for m in masks])
    coincide = 0
    symdiff = []
    for f in factors:
        res = solver.solve(f * r - k, tie="max")
        if res.mask.tobytes() in keyset:
            coincide += 1
        wf = inst.w[res.mask].sum()
        rr = int(np.argmin(np.abs(W - wf)))
        sd = np.logical_xor(res.mask, masks[rr]).sum()
        symdiff.append(sd / max(1, res.mask.sum() + masks[rr].sum() - sd))
    return dict(n_factors=len(list(factors)), n_coincide=coincide, mean_rel_symdiff=float(np.mean(symdiff)))
