"""A tour of the library on the 8-item instance of the paper.

Run it with ``python examples/quickstart.py``.  Every number printed here is
stated in the paper, so this doubles as a demonstration and as a check.
"""
from fractions import Fraction

import numpy as np

from macroitems import canonical_path, running_example, solution_from_path, solve_capacity
from macroitems.dual import best_reduced_costs, fixable_items
from macroitems.faces import face_dimensions
from macroitems.path import canonical_dual

inst = running_example()
print(f"{inst.name}: {inst.n} items, {inst.m} arcs")
print(f"  profits {inst.p.astype(int).tolist()}")
print(f"  weights {inst.w.astype(int).tolist()}")
print("  an arc (i, j) means: j is a prerequisite of i, so x_i <= x_j")

# ---------------------------------------------------------------- the path
# One computation gives the whole value function: the nested closures, the
# breakpoints, and hence z(c) for every capacity at once.
path = canonical_path(inst)
print(f"\ncanonical macroitem sequence ({path.n_maxflow} maximum flows):")
for r, (block, ratio) in enumerate(zip(path.macroitems, path.ratios), start=1):
    exact = Fraction(int(inst.p[block].sum()), int(inst.w[block].sum()))
    items = "{" + ", ".join(str(i + 1) for i in block) + "}"      # 1-based, as in the paper
    print(f"  I_{r} = {items:12s} p={inst.p[block].sum():5.0f}  w={inst.w[block].sum():4.0f}"
          f"  ratio {exact}")
print(f"  cumulative (w, p): {list(zip(path.W.astype(int).tolist(), path.P.astype(int).tolist()))}")

# ------------------------------------------------------- one capacity, c = 4
c = 4.0
sol = solution_from_path(inst, path, c)
print(f"\nat capacity c = {c:g}:")
print(f"  z(c)      = {sol.value:g}          (the paper: 7)")
print(f"  lambda    = {sol.lam:g}          (the ratio of the split macroitem, 3/2)")
print(f"  theta     = {sol.theta:g}")
print(f"  x         = {sol.x.tolist()}")
print(f"  full      {(np.flatnonzero(sol.F) + 1).tolist()}   x_i = 1 in every optimum")
print(f"  split     {(np.flatnonzero(sol.H) + 1).tolist()}   the only items that may differ")
print(f"  null      {(np.flatnonzero(sol.Z) + 1).tolist()}   x_i = 0 in every optimum")

# The Newton search reaches the same point without the whole path.
direct = solve_capacity(inst, c)
assert abs(direct.value - sol.value) < 1e-12
print(f"  solve_capacity agrees, in {direct.n_maxflow} maximum flows")

# ------------------------------------------------------------ certificates
dual = canonical_dual(inst, sol, c)
print(f"\ndual certificate: value {dual.value:g}, feasible {dual.feasible}, "
      f"violation {dual.max_violation:.1e}")

faces = face_dimensions(inst, sol)
print(f"optimal faces: primal dimension {faces.dim_primal} (the optimum is unique), "
      f"dual dimension {faces.dim_dual}")

rc = best_reduced_costs(inst, sol)
print("\nbest reduced costs over the dual face (0 on the split macroitem):")
for i in range(inst.n):
    if rc.region[i] in ("F", "Z"):
        what = "forcing it out" if rc.region[i] == "F" else "forcing it in"
        print(f"  item {i + 1} ({rc.region[i]}): {rc.value[i]:6.2f}   costs at least this, {what}")

# With an incumbent integer solution in hand, those costs fix variables.
incumbent = 5.0                       # the paper's integer optimum at c = 4
fixed = fixable_items(sol, rc.value, incumbent)
print(f"\ngiven an incumbent of {incumbent:g} (gap {fixed['gap']:g}), a branch-and-bound could fix "
      f"{fixed['n_fixed']} of {inst.n} items:")
print(f"  to 0: {(fixed['fix_zero'] + 1).tolist()}      to 1: {(fixed['fix_one'] + 1).tolist()}")
