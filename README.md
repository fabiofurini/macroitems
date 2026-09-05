# macroitems — the LP relaxation of the precedence-constrained knapsack problem

Pilot version (0.1.0) of a library that solves the LP relaxation of the natural
PCKP formulation

    max  p·x   s.t.  w·x ≤ c,   x_i ≤ x_j  for every arc (i, j),   0 ≤ x ≤ 1

through parametric maximum closure, in the language of the companion papers
(Dose, Furini, Locatelli): the increments of the nested path of maximal optimal
closures are the *macroitems* of a canonical sequence with strictly decreasing
profit-to-weight ratios, and the relaxation is the knapsack LP on macroitems.

What it does today

* `Instance`: text format, generators (`layered_grid` mining-like block models,
  `random_dag`), the paper's 8-item `running_example`, a MineLib UPIT reader
  (`read_minelib_upit`), export to CPLEX LP format.
* `ClosureSolver`: maximum closure by Picard's minimum cut on igraph's
  push-relabel max flow, with the inclusion-wise **maximal** (or minimal)
  optimal closure computed from the residual graph — the tie convention of the paper.
* `canonical_path(inst, method)`: the whole canonical sequence (all breakpoints,
  hence the value function z(c) for every capacity) by
  `"bisection"` (Eisner–Severance / Gusfield geometric bisection on contracted
  residual graphs, O(k) max flows) or `"dinkelbach"` (repeated maximum-ratio
  closure extraction). Exact rational breakpoints on integer data.
* `solve_capacity(inst, c)`: the LP at one capacity by a Newton search on the
  weight price (a handful of max flows); returns the canonical primal optimum,
  the regions F / H / Z (persistency: F and Z are fixed in every LP optimum),
  θ and the multiplier λ_h.
* `canonical_dual(inst, sol, c)`: the canonical dual certificate (λ_h and three
  region-wise flows) by three max flows, verified.
* `face_dimensions(inst, sol)`: dimensions of the primal and dual optimal faces
  (inseparability classes of the split macroitem, tight subsets, formula of the note).
* `canonical_reduced_costs`: the knapsack-style reduced costs w_i |λ_r − λ_h|.
* `solve_lp`: HiGHS baseline (scipy) for verification and timing.
* `stats`: structural statistics along the capacity axis, greedy integer
  heuristic for gap estimates, comparison of the weight and revenue-factor pit families.

Install and test

    pip install numpy scipy python-igraph
    python3 tests/test_random.py        # brute-force + LP cross-checks (300 random instances)
    python3 pilot/run_pilot.py          # ~4 minutes; writes pilot/results/*.csv
    python3 pilot/make_report.py        # pilot/results/REPORT.md

Minimal use

```python
from macroitems import layered_grid, canonical_path, solve_capacity, canonical_dual
inst = layered_grid(30, 30, 10, cone=9, seed=1)
path = canonical_path(inst)                 # macroitems, ratios, cumulative P and W
c = 0.5 * path.W[path.q]
sol = solve_capacity(inst, c)               # x, value, lam, F, H, Z, theta
dual = canonical_dual(inst, sol, c)         # lam, mu, alpha; dual.feasible, dual.value
```

Conventions: arcs `(i, j)` mean "j is a prerequisite of i" (x_i ≤ x_j); weights
positive, profits of arbitrary sign; divergence div_i(α) = Σ_{(i,j)} α_ij − Σ_{(j,i)} α_ji.

Road map (see the coauthors' memo, Section 6): MineLib converters and real
instances, a compiled max-flow core, Hochbaum's parametric pseudoflow as a
third path algorithm, best reduced costs on the dual face, warm starts for
branch-and-bound, documentation and a citable release.
