"""Fast strongest reduced costs (C++ engine with warm-started maximum flows).

Same output as :func:`macroitems.dual.best_reduced_costs`, computed exactly on
integer data by the C++ library ``cpp/rc_engine.cpp``:

* the values ``b_i = p_i - lambda_h w_i`` are scaled to integers by the
  denominator of ``lambda_h = p(H)/w(H)``;
* on the null region ``Z`` the closure arcs are used as they are, with values
  ``b``; on the full region ``F`` the arcs are reversed and the values are
  ``-b``, since co-closed sets of ``F`` are closed sets of the reverse graph;
* engine ``"warm"`` computes the base maximum flow once. For every item it
  pushes only the extra flow from the item to the sink in the residual network,
  and then undoes the arcs it touched.

The library is built on first use with ``g++`` (``-O3``). Set ``MACROITEMS_CPP``
to use another build directory.
"""
from __future__ import annotations

import ctypes
import os
import subprocess
import time
from fractions import Fraction
from math import gcd
from typing import Optional

import numpy as np

from .instance import Instance
from .path import LPSolution

__all__ = ["fast_reduced_costs", "ENGINES"]

ENGINES = {"cold": 0, "warm": 1, "goldberg": 2, "sap": 3, "tree": 4}
_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "cpp")
_LIB = None


def _library():
    global _LIB
    if _LIB is not None:
        return _LIB
    build = os.environ.get("MACROITEMS_CPP", os.path.join(_ROOT, "build"))
    os.makedirs(build, exist_ok=True)
    src = os.path.join(_ROOT, "rc_engine.cpp")
    gold = os.path.join(_ROOT, "third_party", "flow_s.c")
    so = os.path.join(build, "librc.so")
    deps = [src] + ([gold] if os.path.exists(gold) else [])
    if not os.path.exists(so) or any(os.path.getmtime(d) > os.path.getmtime(so) for d in deps):
        objs = []
        flags = ["-O3", "-march=native", "-fPIC"]
        if os.path.exists(gold):
            obj = os.path.join(build, "flow_s.o")
            subprocess.run(["gcc", "-std=gnu17", "-w", *flags, "-c", gold, "-o", obj], check=True)
            objs.append(obj)
        cmd = ["g++", "-std=c++17", *flags, "-shared", src, *objs, "-o", so]
        if objs:
            cmd.insert(3, "-DWITH_GOLDBERG")
        subprocess.run(cmd, check=True)
    lib = ctypes.CDLL(so)
    lib.rc_region.restype = ctypes.c_int
    lib.rc_region.argtypes = [ctypes.c_int, ctypes.c_int,
                              np.ctypeslib.ndpointer(np.int32), np.ctypeslib.ndpointer(np.int32),
                              np.ctypeslib.ndpointer(np.int64), np.ctypeslib.ndpointer(np.int64),
                              ctypes.c_int, np.ctypeslib.ndpointer(np.float64)]
    _LIB = lib
    return lib


def _region(inst: Instance, mask: np.ndarray, values: list, reverse: bool, engine: int):
    nodes = np.flatnonzero(mask)
    pos = -np.ones(inst.n, dtype=np.int64)
    pos[nodes] = np.arange(nodes.size)
    a = inst.arcs
    keep = mask[a[:, 0]] & mask[a[:, 1]] if inst.m else np.zeros(0, bool)
    u, v = pos[a[keep, 0]], pos[a[keep, 1]]
    if reverse:
        u, v = v, u
    val = np.array([values[i] for i in nodes], dtype=np.int64)
    out = np.zeros(nodes.size, dtype=np.int64)
    stats = np.zeros(6)
    rc = _library().rc_region(int(nodes.size), int(u.size), u.astype(np.int32), v.astype(np.int32),
                              val, out, engine, stats)
    if rc != 0:
        raise RuntimeError(f"rc_region failed with code {rc}")
    return nodes, out, stats


def fast_reduced_costs(inst: Instance, sol: LPSolution, engine: str = "warm",
                       return_stats: bool = False):
    """Strongest reduced costs s*_i on Z and mu*_i on F (0 on H, nan elsewhere)."""
    scale = 1
    if not inst.is_integral():
        inst, scale = inst.scaled_to_integers()   # same factor on p and w: lambda_h is unchanged
        if not inst.is_integral():
            raise ValueError("fast_reduced_costs needs data that scale exactly to integers")
    H = sol.H
    P, W = int(round(inst.p[H].sum())), int(round(inst.w[H].sum()))
    g = gcd(abs(P), W) or 1
    P, W = P // g, W // g                       # lambda_h = P / W, in lowest terms
    p = [int(round(x)) for x in inst.p]
    w = [int(round(x)) for x in inst.w]
    b = [W * pi - P * wi for pi, wi in zip(p, w)]   # W * b_i, exact
    G = 0
    for x in b:
        G = gcd(G, abs(x))
    G = G or 1
    b = [x // G for x in b]                      # common factor removed: values stay exact
    if sum(abs(x) for x in b) >= 2 ** 61:
        raise OverflowError("scaled values exceed 64-bit range")
    out = np.full(inst.n, np.nan)
    out[H] = 0.0
    stats = {}
    t0 = time.perf_counter()
    eng = ENGINES[engine]
    if sol.Z.any():
        nodes, r, st = _region(inst, sol.Z, b, False, eng)
        out[nodes] = r * G / (W * scale)
        stats["Z"] = st
    if sol.F.any():
        nodes, r, st = _region(inst, sol.F, [-x for x in b], True, eng)
        out[nodes] = r * G / (W * scale)
        stats["F"] = st
    stats["seconds"] = time.perf_counter() - t0
    return (out, stats) if return_stats else out


def fast_tight_sets(inst: Instance, sol: LPSolution):
    """Smallest tight subset of H containing each item of H, via one maximum flow and the
    reachability of its residual network (C++).  Returns ``(nodes, Tm)`` with the same
    meaning as :func:`macroitems.faces.minimal_tight_sets`, or ``None`` when the data
    cannot be scaled to 64-bit integers (the caller then uses the reference method)."""
    scale = 1
    if not inst.is_integral():
        inst, scale = inst.scaled_to_integers()
        if not inst.is_integral():
            return None
    nodes = np.flatnonzero(sol.H)
    k = nodes.size
    P, W = int(round(inst.p[sol.H].sum())), int(round(inst.w[sol.H].sum()))
    g = gcd(abs(P), W) or 1
    P, W = P // g, W // g
    b = [W * int(round(inst.p[i])) - P * int(round(inst.w[i])) for i in nodes]
    G = 0
    for x in b:
        G = gcd(G, abs(x))
    G = G or 1
    b = np.array([x // G for x in b], dtype=np.int64) if True else None
    if int(np.abs(b).sum()) >= 2 ** 61:
        return None
    pos = -np.ones(inst.n, dtype=np.int64)
    pos[nodes] = np.arange(k)
    a = inst.arcs
    keep = sol.H[a[:, 0]] & sol.H[a[:, 1]] if inst.m else np.zeros(0, bool)
    u, v = pos[a[keep, 0]].astype(np.int32), pos[a[keep, 1]].astype(np.int32)
    Tm = np.zeros((k, k), dtype=np.uint8)
    stats = np.zeros(2)
    lib = _library()
    if not hasattr(lib, "_tight_ready"):
        lib.tight_sets.restype = ctypes.c_int
        lib.tight_sets.argtypes = [ctypes.c_int, ctypes.c_int,
                                   np.ctypeslib.ndpointer(np.int32), np.ctypeslib.ndpointer(np.int32),
                                   np.ctypeslib.ndpointer(np.int64), np.ctypeslib.ndpointer(np.uint8),
                                   np.ctypeslib.ndpointer(np.float64)]
        lib._tight_ready = True
    rc = lib.tight_sets(int(k), int(u.size), u, v, b, Tm, stats)
    if rc != 0:
        return None
    return nodes, Tm.astype(bool)
