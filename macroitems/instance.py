"""Instances of the precedence-constrained knapsack LP.

An instance has n items with profits p (arbitrary sign) and weights w > 0, and
a set of arcs (i, j) meaning "j is a prerequisite of i" (x_i <= x_j), as in the
paper.  Arcs are stored as an integer array of shape (m, 2).

File format (text, one record per line, '#' starts a comment):

    n m
    p_0 w_0
    ...
    p_{n-1} w_{n-1}
    i j          (m lines; j prerequisite of i)
"""
from __future__ import annotations

import dataclasses
import json
import os
import zipfile
from typing import Iterable, Optional

import numpy as np

from decimal import Decimal


@dataclasses.dataclass
class Instance:
    p: np.ndarray               # profits, float64, shape (n,)
    w: np.ndarray               # weights, float64 > 0, shape (n,)
    arcs: np.ndarray            # int64, shape (m, 2); (i, j): j prerequisite of i
    name: str = "instance"
    meta: dict = dataclasses.field(default_factory=dict)
    extra: dict = dataclasses.field(default_factory=dict, repr=False)   # optional arrays (e.g. revenue, cost)

    # ------------------------------------------------------------------ basics
    @property
    def n(self) -> int:
        return int(self.p.shape[0])

    @property
    def m(self) -> int:
        return int(self.arcs.shape[0])

    def validate(self) -> None:
        assert self.p.shape == self.w.shape, "p and w must have the same length"
        assert np.all(self.w > 0), "weights must be positive"
        if self.m:
            assert self.arcs.min() >= 0 and self.arcs.max() < self.n, "arc index out of range"
            assert np.all(self.arcs[:, 0] != self.arcs[:, 1]), "self-loop"
        # acyclicity (Kahn) -- cheap enough and worth doing once per instance
        indeg = np.bincount(self.arcs[:, 1], minlength=self.n) if self.m else np.zeros(self.n, int)
        order = topological_order(self.n, self.arcs)
        assert order is not None, "precedence graph has a cycle"

    # ------------------------------------------------------------- scaling
    def is_integral(self) -> bool:
        """True if profits and weights are integers (the exact-arithmetic regime)."""
        return bool(np.all(self.p == np.rint(self.p)) and np.all(self.w == np.rint(self.w)))

    def scaled_to_integers(self, max_power: int = 9,
                           max_magnitude: float = 2.0 ** 53) -> tuple["Instance", int]:
        """Return an equivalent instance with integer data, and the scale used.

        Multiplying every profit and weight by the same positive integer
        rescales the parametric values by that integer and leaves every
        closure, every macroitem and every ratio order unchanged; capacities
        scale with the weights.  The optimal value scales by the same factor,
        so ``z_original(c) = z_scaled(scale * c) / scale``.  Getting there
        matters: on integer data the whole parametric machinery is exact, with
        the breakpoints recovered as rationals.

        The number of decimals is read off the shortest decimal representation
        of each value and the scaling is then done in decimal arithmetic.
        Multiplying the floats instead would fail on data that *are* decimal:
        a block value of ``-2236.7886`` is not exactly representable, and
        ``value * 10**9`` inherits an error of about 0.016, which no rounding
        tolerance can distinguish from genuinely finer data.

        Returns ``(self, 1)`` when the data are already integral, when more
        than ``max_power`` decimals are needed, or when scaling would push a
        value past ``max_magnitude`` (beyond which the products formed by the
        parametric search stop being exact in 64-bit arithmetic).
        """
        if self.is_integral():
            return self, 1
        decimals = max(_decimals_of(v) for v in (*self.p.tolist(), *self.w.tolist()))
        if decimals == 0 or decimals > max_power:
            return self, 1
        scale = 10 ** decimals
        largest = max(np.abs(self.p).max(initial=0.0), self.w.max(initial=0.0)) * scale
        if largest > max_magnitude:
            return self, 1
        p_int = _scale_exactly(self.p, decimals)
        w_int = _scale_exactly(self.w, decimals)
        meta = dict(self.meta)
        meta["integer_scale"] = scale
        for key in ("capacity", "capacity_cpit_period", "capacity_cpit_total"):
            if key in meta:
                meta[key] = meta[key] * scale
        out = Instance(p_int, w_int, self.arcs.copy(), name=self.name, meta=meta,
                       extra=dict(self.extra))
        return out, scale

    def induced(self, nodes: np.ndarray) -> tuple["Instance", np.ndarray]:
        """Sub-instance induced by `nodes` (sorted array of item indices).
        Returns (sub_instance, nodes) where sub item k corresponds to nodes[k]."""
        nodes = np.asarray(nodes, dtype=np.int64)
        mask = np.zeros(self.n, dtype=bool)
        mask[nodes] = True
        keep = mask[self.arcs[:, 0]] & mask[self.arcs[:, 1]] if self.m else np.zeros(0, bool)
        sub_arcs = self.arcs[keep]
        relabel = -np.ones(self.n, dtype=np.int64)
        relabel[nodes] = np.arange(nodes.size)
        sub = Instance(self.p[nodes].copy(), self.w[nodes].copy(),
                       relabel[sub_arcs] if sub_arcs.size else np.zeros((0, 2), np.int64),
                       name=self.name + "[sub]")
        return sub, nodes

    # -------------------------------------------------------------------- I/O
    def write(self, path: str) -> None:
        with open(path, "w") as f:
            f.write(f"# {self.name}\n{self.n} {self.m}\n")
            for pi, wi in zip(self.p, self.w):
                f.write(f"{pi:.10g} {wi:.10g}\n")
            for i, j in self.arcs:
                f.write(f"{i} {j}\n")
        if self.meta:
            with open(path + ".json", "w") as f:
                json.dump(self.meta, f, indent=1)

    @staticmethod
    def read(path: str) -> "Instance":
        with open(path) as f:
            lines = [ln.strip() for ln in f if ln.strip() and not ln.lstrip().startswith("#")]
        n, m = map(int, lines[0].split())
        pw = np.array([list(map(float, ln.split())) for ln in lines[1:1 + n]])
        arcs = np.array([list(map(int, ln.split())) for ln in lines[1 + n:1 + n + m]], dtype=np.int64)
        if arcs.size == 0:
            arcs = np.zeros((0, 2), np.int64)
        inst = Instance(pw[:, 0], pw[:, 1], arcs, name=os.path.basename(path))
        if os.path.exists(path + ".json"):
            inst.meta = json.load(open(path + ".json"))
        inst.validate()
        return inst

    def write_lp(self, path: str, capacity: float) -> None:
        """Export the LP relaxation at a given capacity in CPLEX LP format."""
        with open(path, "w") as f:
            f.write("Maximize\n obj: " + " + ".join(f"{pi:.10g} x{i}" for i, pi in enumerate(self.p)) + "\n")
            f.write("Subject To\n cap: " + " + ".join(f"{wi:.10g} x{i}" for i, wi in enumerate(self.w))
                    + f" <= {capacity:.10g}\n")
            for k, (i, j) in enumerate(self.arcs):
                f.write(f" prec{k}: x{i} - x{j} <= 0\n")
            f.write("Bounds\n" + "".join(f" 0 <= x{i} <= 1\n" for i in range(self.n)) + "End\n")



def _decimals_of(value: float) -> int:
    """Number of decimals in the shortest decimal that round-trips to ``value``.

    ``repr`` of a float gives that shortest decimal, so a value read from a
    file as ``2192.93`` reports 2, whatever its binary expansion looks like.
    """
    text = repr(float(value))
    if "e" in text or "E" in text:
        return Decimal(text).as_tuple().exponent * -1 if Decimal(text).as_tuple().exponent < 0 else 0
    _, _, frac = text.partition(".")
    frac = frac.rstrip("0")
    return len(frac)


def _scale_exactly(values: np.ndarray, decimals: int) -> np.ndarray:
    """``values * 10**decimals`` as exact integers, via decimal arithmetic."""
    factor = Decimal(10) ** decimals
    return np.array([float(Decimal(repr(float(v))) * factor) for v in values.tolist()])


def topological_order(n: int, arcs: np.ndarray) -> Optional[np.ndarray]:
    """Kahn's algorithm on arcs (i, j) read as edges i -> j. Returns None on a cycle."""
    if arcs.shape[0] == 0:
        return np.arange(n)
    indeg = np.bincount(arcs[:, 1], minlength=n)
    order_idx = np.argsort(arcs[:, 0], kind="stable")
    src_sorted = arcs[order_idx, 0]
    starts = np.searchsorted(src_sorted, np.arange(n + 1))
    heads = arcs[order_idx, 1]
    stack = list(np.flatnonzero(indeg == 0))
    out = []
    indeg = indeg.copy()
    while stack:
        u = stack.pop()
        out.append(u)
        for v in heads[starts[u]:starts[u + 1]]:
            indeg[v] -= 1
            if indeg[v] == 0:
                stack.append(v)
    return np.array(out) if len(out) == n else None


# ------------------------------------------------------------------ MineLib
def read_minelib_upit(zip_or_dir: str, tonnage_column: Optional[int] = None,
                      name: Optional[str] = None) -> Instance:
    """Read a MineLib instance (Espinoza, Goycoolea, Moreno, Newman 2013).

    Expects the files <name>.upit (objective coefficients), <name>.prec
    (precedences: "b k b_1 ... b_k", b_1..b_k must be extracted before b) and
    <name>.blocks (id x y z attributes...). Profit = UPIT objective coefficient,
    weight = the attribute in `tonnage_column` of the .blocks file (0-based index
    among the attributes after id, x, y, z); if None, unit weights are used and
    a warning is stored in meta.  The convention of this package is arcs
    (i, j) with j prerequisite of i, so a line "b k b_1 ... b_k" gives arcs (b, b_r).
    """
    files = {}
    if zipfile.is_zipfile(zip_or_dir):
        with zipfile.ZipFile(zip_or_dir) as z:
            for fn in z.namelist():
                ext = os.path.splitext(fn)[1].lower()
                if ext in (".upit", ".prec", ".blocks"):
                    files[ext] = z.read(fn).decode("utf-8", errors="replace").splitlines()
    else:
        for fn in os.listdir(zip_or_dir):
            ext = os.path.splitext(fn)[1].lower()
            if ext in (".upit", ".prec", ".blocks"):
                files[ext] = open(os.path.join(zip_or_dir, fn)).read().splitlines()
    assert ".upit" in files and ".prec" in files, "need .upit and .prec files"
    # objective
    vals = {}
    in_obj = False
    nblocks = None
    for ln in files[".upit"]:
        s = ln.strip()
        if not s:
            continue
        if s.upper().startswith("NBLOCKS"):
            nblocks = int(s.split(":")[1])
        if s.upper().startswith("OBJECTIVE_FUNCTION"):
            in_obj = True
            continue
        if in_obj:
            parts = s.split()
            if len(parts) >= 2 and parts[0].lstrip("-").isdigit():
                vals[int(parts[0])] = float(parts[1])
            elif parts[0].upper().startswith("EOF"):
                break
    n = nblocks if nblocks is not None else (max(vals) + 1)
    p = np.zeros(n)
    for b, v in vals.items():
        p[b] = v
    # precedences
    arcs = []
    for ln in files[".prec"]:
        parts = ln.split()
        if len(parts) < 2:
            continue
        b, k = int(parts[0]), int(parts[1])
        for r in parts[2:2 + k]:
            arcs.append((b, int(r)))
    arcs = np.array(arcs, dtype=np.int64) if arcs else np.zeros((0, 2), np.int64)
    # weights
    meta = {"source": "MineLib", "file": os.path.basename(zip_or_dir)}
    w = np.ones(n)
    if ".blocks" in files and tonnage_column is not None:
        for ln in files[".blocks"]:
            parts = ln.split()
            if len(parts) < 5 + tonnage_column:
                continue
            w[int(parts[0])] = float(parts[4 + tonnage_column])
        meta["weight"] = f"blocks attribute column {tonnage_column}"
    else:
        meta["weight"] = "unit weights (no tonnage column given)"
    inst = Instance(p, w, arcs, name=name or os.path.splitext(os.path.basename(zip_or_dir))[0], meta=meta)
    inst.validate()
    return inst


# --------------------------------------------------------------- generators
def layered_grid(nx: int, ny: int, nz: int, cone: int = 5, seed: int = 0,
                 n_ore_bodies: int = 2, price: float = 1.0, mining_cost: float = 1.0,
                 processing_cost: float = 3.0, round_values: bool = True,
                 unit_weights: bool = False, name: Optional[str] = None) -> Instance:
    """A mining-like block model: nx*ny*nz blocks, block (x, y, z) requires the
    `cone` blocks above it (5: cross, 9: full 3x3) at level z-1.  Grades come from
    a few ellipsoidal ore bodies; block value = tonnage * (grade*price - processing
    cost) - tonnage*mining cost for ore blocks (grade above cutoff), otherwise
    -tonnage*mining cost.  Values rounded to integers to create ties as in MineLib."""
    rng = np.random.default_rng(seed)
    X, Y, Z = np.meshgrid(np.arange(nx), np.arange(ny), np.arange(nz), indexing="ij")
    X, Y, Z = X.ravel(), Y.ravel(), Z.ravel()
    n = X.size
    idx = np.arange(n).reshape(nx, ny, nz)
    # grades
    grade = np.zeros(n)
    for _ in range(n_ore_bodies):
        cx, cy, cz = rng.uniform(0.25, 0.75) * nx, rng.uniform(0.25, 0.75) * ny, rng.uniform(0.2, 0.65) * nz
        ax, ay, az = rng.uniform(0.2, 0.4) * nx, rng.uniform(0.2, 0.4) * ny, rng.uniform(0.25, 0.5) * nz
        d2 = ((X - cx) / ax) ** 2 + ((Y - cy) / ay) ** 2 + ((Z - cz) / az) ** 2
        grade += rng.uniform(6, 14) * np.exp(-2.0 * d2) * rng.lognormal(0, 0.25, n)
    tonnage = np.ones(n) if unit_weights else rng.integers(10, 13, n).astype(float)   # integer tonnage, ~10% variation
    ore = grade * price > processing_cost
    value = np.where(ore, tonnage * (grade * price - processing_cost), 0.0) - tonnage * mining_cost
    if round_values:
        value = np.round(value)
    # precedences
    if cone == 5:
        offs = [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)]
    elif cone == 9:
        offs = [(dx, dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1)]
    else:
        raise ValueError("cone must be 5 or 9")
    arcs = []
    for dx, dy in offs:
        xs = X + dx
        ys = Y + dy
        ok = (Z > 0) & (xs >= 0) & (xs < nx) & (ys >= 0) & (ys < ny)
        src = idx[X[ok], Y[ok], Z[ok]]
        dst = idx[xs[ok], ys[ok], Z[ok] - 1]
        arcs.append(np.stack([src, dst], axis=1))
    arcs = np.concatenate(arcs) if arcs else np.zeros((0, 2), np.int64)
    inst = Instance(value.astype(float), tonnage.astype(float), arcs.astype(np.int64),
                    name=name or f"grid{nx}x{ny}x{nz}c{cone}s{seed}",
                    meta={"generator": "layered_grid", "nx": nx, "ny": ny, "nz": nz, "cone": cone, "seed": seed,
                          "ore_blocks": int(ore.sum())})
    revenue = np.where(ore, tonnage * grade * price, 0.0)
    cost = np.where(ore, tonnage * (processing_cost + mining_cost), tonnage * mining_cost)
    if round_values:
        revenue, cost = np.round(revenue), np.round(cost)
    inst.extra["revenue"], inst.extra["cost"] = revenue, cost
    inst.validate()
    return inst


def random_dag(n: int, avg_out_degree: float = 2.0, seed: int = 0, p_range=(-20, 100),
               w_range=(1, 50), name: Optional[str] = None) -> Instance:
    """Random DAG on a random topological order: item i requires items j > i
    chosen at random (so arcs (i, j) with j > i); integer profits and weights."""
    rng = np.random.default_rng(seed)
    p = rng.integers(p_range[0], p_range[1] + 1, n).astype(float)
    w = rng.integers(w_range[0], w_range[1] + 1, n).astype(float)
    arcs = []
    for i in range(n - 1):
        k = rng.poisson(avg_out_degree)
        k = min(k, n - 1 - i)
        if k > 0:
            js = rng.choice(np.arange(i + 1, min(n, i + 1 + 8 * max(1, int(avg_out_degree)) * 4)), size=k,
                            replace=False) if (n - 1 - i) > k else np.arange(i + 1, n)
            for j in js:
                arcs.append((i, int(j)))
    arcs = np.array(arcs, dtype=np.int64) if arcs else np.zeros((0, 2), np.int64)
    inst = Instance(p, w, arcs, name=name or f"dag{n}d{avg_out_degree}s{seed}",
                    meta={"generator": "random_dag", "n": n, "avg_out_degree": avg_out_degree, "seed": seed})
    inst.validate()
    return inst


def running_example() -> Instance:
    """The 8-item running instance of the paper (items 1..8 -> 0..7).
    Arcs (i, j): j prerequisite of i.  Canonical sequence {3,6}, {1,2,5}, {4,7,8}
    with ratios 2, 3/2, 1 (paper numbering)."""
    p = np.array([1.0, -1.0, -1.0, -1.0, 6.0, 5.0, 2.0, 3.0])
    w = np.array([1.0, 1.0, 1.0, 2.0, 2.0, 1.0, 1.0, 1.0])
    # paper arcs (1-based, dependent -> prerequisite):
    # (5,1),(5,2),(5,6),(6,3),(7,3),(7,4),(8,5),(8,6),(8,7)
    arcs = np.array([(4, 0), (4, 1), (4, 5), (5, 2), (6, 2), (6, 3), (7, 4), (7, 5), (7, 6)], dtype=np.int64)
    inst = Instance(p, w, arcs, name="running_example")
    inst.validate()
    return inst
