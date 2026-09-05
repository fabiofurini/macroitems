"""Command-line interface: ``macroitems <command> [options]``.

The commands mirror the library's objects.  ``info`` and ``path`` describe an
instance and its canonical macroitem sequence; ``solve`` answers one capacity
and can add the dual certificate and reduced costs; ``lp`` runs a
general-purpose LP solver on the same model, for comparison; ``convert``
rewrites an instance in this package's text format; ``gen`` produces synthetic
instances.

Capacities can be given absolutely (``--capacity 12000``) or as a fraction of
the weight of the maximum-profit closure (``--capacity 0.5 --relative``),
which is the scale-free way to compare instances of very different sizes.
"""
from __future__ import annotations

import argparse
import json
import sys
from typing import Optional

import numpy as np

from . import __version__
from .instance import Instance, layered_grid, random_dag, running_example
from .path import canonical_path, solution_from_path, solve_capacity


def _read(path: str) -> Instance:
    """Read an instance in any supported format."""
    from .formats import read_any, read_minelib_upit
    import os
    if path == "running-example":
        return running_example()
    if not os.path.splitext(path)[1] and not os.path.isdir(path):
        return read_minelib_upit(path)      # a MineLib instance stem
    return read_any(path)


def _capacity(inst: Instance, value: float, relative: bool,
              path=None) -> tuple[float, Optional[object]]:
    """Resolve a capacity, computing the path if a relative one needs it."""
    if not relative:
        return float(value), path
    path = path or canonical_path(inst)
    top = float(path.W[path.q]) if path.q else float(inst.w.sum())
    return float(value) * top, path


def _json_default(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    raise TypeError(f"cannot serialize {type(obj).__name__}")


def _emit(data: dict, path: Optional[str]) -> None:
    if path:
        with open(path, "w") as f:
            json.dump(data, f, indent=1, default=_json_default)
        print(f"written to {path}")


# ------------------------------------------------------------------ commands
def cmd_info(args) -> int:
    inst = _read(args.instance)
    negative = int((inst.p < 0).sum())
    print(f"name          {inst.name}")
    print(f"items         {inst.n}")
    print(f"arcs          {inst.m}   (density {inst.m / max(1, inst.n):.2f})")
    print(f"profits       [{inst.p.min():g}, {inst.p.max():g}], {negative} negative "
          f"({negative / inst.n:.1%})")
    print(f"weights       [{inst.w.min():g}, {inst.w.max():g}]")
    print(f"integer data  {inst.is_integral()}")
    if not inst.is_integral():
        _, scale = inst.scaled_to_integers()
        print(f"              becomes integral when multiplied by {scale}"
              if scale > 1 else "              no integer scaling found")
    for key, value in sorted(inst.meta.items()):
        print(f"meta.{key:<9s} {value}")
    return 0


def cmd_path(args) -> int:
    inst = _read(args.instance)
    work, scale = (inst.scaled_to_integers() if args.exact else (inst, 1))
    path = canonical_path(work, method=args.method, backend=args.backend)
    sizes = np.array([I.size for I in path.macroitems])
    print(f"{inst.name}: n={inst.n} m={inst.m}")
    print(f"macroitems    k={path.k}  (q={path.q} with positive ratio)")
    print(f"sizes         max {sizes.max()}, median {int(np.median(sizes))}, "
          f"{int((sizes == 1).sum())} singletons")
    # ratios are invariant under the integer rescaling; P and W are not
    print(f"ratios        lambda_1={path.ratios[0]:g} ... "
          f"lambda_k={path.ratios[-1]:g}")
    print(f"cost          {path.n_maxflow} maximum flows, {path.seconds:.3f} s")
    if args.check:
        print(f"check         {canonical_path(work, method=args.method).check(work)}")
    if args.json:
        _emit({"instance": inst.name, "n": inst.n, "m": inst.m, "k": path.k, "q": path.q,
               "method": path.method, "n_maxflow": path.n_maxflow, "seconds": path.seconds,
               "ratios": path.ratios.tolist(),
               "cumulative_profit": (path.P / scale).tolist(),
               "cumulative_weight": (path.W / scale).tolist(),
               "macroitems": [I.tolist() for I in path.macroitems]}, args.json)
    return 0


def cmd_solve(args) -> int:
    inst = _read(args.instance)
    work, scale = (inst.scaled_to_integers() if args.exact else (inst, 1))
    capacity, path = _capacity(work, args.capacity, args.relative)
    if path is not None:
        # A relative capacity needs w(M_q), which only the whole path gives, so
        # the path is already paid for and the solution is read off it.
        sol = solution_from_path(work, path, capacity)
        cost = (f"{path.n_maxflow} maximum flows, {path.seconds:.3f} s "
                f"(the whole path: a relative capacity requires it)")
    else:
        sol = solve_capacity(work, capacity, backend=args.backend)
        cost = f"{sol.n_maxflow} maximum flows, {sol.seconds:.3f} s"
    n_full, n_split, n_null = int(sol.F.sum()), int(sol.H.sum()), int(sol.Z.sum())
    print(f"{inst.name}  capacity {capacity / scale:g}")
    print(f"value         {sol.value / scale:.10g}")
    print(f"multiplier    {sol.lam / 1:.10g}" if scale == 1 else
          f"multiplier    {sol.lam:.10g}")
    print(f"split fill    theta = {sol.theta:.6g}" + (f"  ({sol.degenerate})" if sol.degenerate else ""))
    print(f"regions       full {n_full}, split {n_split}, null {n_null}")
    print(f"persistency   {1 - n_split / inst.n:.4f}  (fraction fixed in every optimum)")
    print(f"cost          {cost}")

    out = {"instance": inst.name, "capacity": capacity / scale, "value": sol.value / scale,
           "lambda": sol.lam, "theta": sol.theta, "n_full": n_full, "n_split": n_split,
           "n_null": n_null, "x": (sol.x if args.with_x else None)}

    if args.dual:
        from .path import canonical_dual
        dual = canonical_dual(work, sol, capacity)
        print(f"dual          value {dual.value / scale:.10g}, feasible {dual.feasible}, "
              f"max violation {dual.max_violation:.2e}")
        out["dual"] = {"value": dual.value / scale, "feasible": bool(dual.feasible),
                       "max_violation": dual.max_violation}
    if args.reduced_costs:
        from .dual import best_reduced_costs
        rc = best_reduced_costs(work, sol, max_items=args.reduced_costs_max, backend=args.backend)
        finite = rc.value[np.isfinite(rc.value)]
        print(f"reduced costs {finite.size} items, mean {finite.mean() / scale:.6g}, "
              f"max {finite.max() / scale:.6g}, {rc.n_maxflow} maximum flows, {rc.seconds:.2f} s")
        out["reduced_costs"] = {"n": int(finite.size), "mean": float(finite.mean()) / scale}
    if args.faces:
        from .faces import face_dimensions
        info = face_dimensions(work, sol)
        print(f"faces         dim primal {info.dim_primal}, dim dual {info.dim_dual}, "
              f"k0 {info.k0}")
        out["faces"] = {"dim_primal": info.dim_primal, "dim_dual": info.dim_dual, "k0": info.k0}
    _emit(out, args.json)
    return 0


def cmd_lp(args) -> int:
    from .lp import available_lp_backends, get_lp_backend
    inst = _read(args.instance)
    capacity, _ = _capacity(inst, args.capacity, args.relative)
    if args.solver not in available_lp_backends():
        print(f"solver {args.solver!r} not available; available: "
              f"{', '.join(available_lp_backends()) or 'none'}", file=sys.stderr)
        return 2
    solver = get_lp_backend(args.solver)(inst, method=args.method, threads=args.threads,
                                         time_limit=args.time_limit)
    try:
        res = solver.solve(capacity)
    finally:
        solver.close()
    print(f"{inst.name}  capacity {capacity:g}  [{res.backend}:{res.method}]")
    print(f"value         {res.value:.10g}")
    print(f"multiplier    {res.lam:.10g}")
    print(f"status        {res.status}  {res.message}")
    print(f"time          build {res.seconds_build:.3f} s, solve {res.seconds_solve:.3f} s")
    _emit({"instance": inst.name, "capacity": capacity, "value": res.value, "lambda": res.lam,
           "status": res.status, "backend": res.backend, "method": res.method,
           "seconds_build": res.seconds_build, "seconds_solve": res.seconds_solve}, args.json)
    return 0


def cmd_convert(args) -> int:
    inst = _read(args.instance)
    inst.write(args.out)
    print(f"{inst.name}: n={inst.n} m={inst.m} -> {args.out}")
    return 0


def cmd_gen(args) -> int:
    if args.kind == "grid":
        inst = layered_grid(args.nx, args.ny, args.nz, cone=args.cone, seed=args.seed)
    else:
        inst = random_dag(args.n, args.degree, seed=args.seed)
    inst.write(args.out)
    print(f"{inst.name}: n={inst.n} m={inst.m} -> {args.out}")
    return 0


# -------------------------------------------------------------------- parser
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="macroitems", description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--version", action="version", version=f"macroitems {__version__}")
    sub = ap.add_subparsers(dest="command", required=True)

    def add_common(p, with_backend=True):
        p.add_argument("instance", help="instance file, MineLib stem, or 'running-example'")
        p.add_argument("--json", default=None, help="also write the result as JSON")
        if with_backend:
            p.add_argument("--backend", default=None,
                           help="maximum-flow backend: ortools, igraph or scipy")
            p.add_argument("--exact", action="store_true", default=True,
                           help="scale decimal data to integers (default)")
            p.add_argument("--no-exact", dest="exact", action="store_false")

    p = sub.add_parser("info", help="size, data ranges and metadata of an instance")
    p.add_argument("instance")
    p.set_defaults(func=cmd_info)

    p = sub.add_parser("path", help="the canonical macroitem sequence (all capacities at once)")
    add_common(p)
    p.add_argument("--method", default="bisection",
                   choices=["bisection", "dinkelbach"])
    p.add_argument("--check", action="store_true", help="verify the sequence")
    p.set_defaults(func=cmd_path)

    p = sub.add_parser("solve", help="the LP at one capacity, with optional certificates")
    add_common(p)
    p.add_argument("--capacity", type=float, required=True)
    p.add_argument("--relative", action="store_true",
                   help="capacity as a fraction of the maximum-profit closure's weight")
    p.add_argument("--dual", action="store_true", help="also build the canonical dual certificate")
    p.add_argument("--reduced-costs", action="store_true",
                   help="also compute best reduced costs over the dual face")
    p.add_argument("--reduced-costs-max", type=int, default=2000,
                   help="sample at most this many items for the reduced costs")
    p.add_argument("--faces", action="store_true", help="also compute optimal-face dimensions")
    p.add_argument("--with-x", action="store_true", help="include the solution vector in the JSON")
    p.set_defaults(func=cmd_solve)

    p = sub.add_parser("lp", help="solve the same model with a general-purpose LP solver")
    p.add_argument("instance")
    p.add_argument("--json", default=None)
    p.add_argument("--capacity", type=float, required=True)
    p.add_argument("--relative", action="store_true")
    p.add_argument("--solver", default="highs")
    p.add_argument("--method", default="default")
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--time-limit", type=float, default=None)
    p.set_defaults(func=cmd_lp)

    p = sub.add_parser("convert", help="rewrite an instance in this package's text format")
    p.add_argument("instance")
    p.add_argument("--out", required=True)
    p.set_defaults(func=cmd_convert)

    p = sub.add_parser("gen", help="generate a synthetic instance")
    p.add_argument("kind", choices=["grid", "dag"])
    p.add_argument("--out", required=True)
    p.add_argument("--nx", type=int, default=20)
    p.add_argument("--ny", type=int, default=20)
    p.add_argument("--nz", type=int, default=8)
    p.add_argument("--cone", type=int, default=5, choices=[5, 9])
    p.add_argument("--n", type=int, default=1000)
    p.add_argument("--degree", type=float, default=2.0)
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(func=cmd_gen)
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"macroitems: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
