# Pilot: the PCKP LP relaxation through parametric maximum closure

All computations in Python on top of igraph's push-relabel maximum flow (2 cores, x86_64), with exact integer arithmetic for the breakpoints; LP baseline: HiGHS via scipy. Synthetic instances: `grid_*` are layered block models with 5- or 9-block precedence cones and integer block values/tonnages (sizes chosen to match MineLib's newman1, zuck_small, kd and marvin); `dag_*` are random DAGs with integer profits in [-20, 100] and weights in [1, 50]. Every number below was cross-checked: both path algorithms return the same sequence, all LP values agree with HiGHS to 1e-9 relative, and every canonical dual certificate was verified feasible with objective equal to the primal value.

## 1. Canonical macroitem sequence (whole value function)

| instance | n | m | k (all) | q (ratio>0) | items in M_q | median / max size (r ≤ q) | largest macroitem weight share of w(M_q) | top-3 share | bisection: max flows, time | Dinkelbach: max flows, time |
|---|---|---|---|---|---|---|---|---|---|---|
| grid_S | 1152 | 4704 | 47 | 22 | 335 | 10.5 / 83 | 24.9% | 47.3% | 93, 0.03 s | 253, 0.32 s |
| grid_M | 9000 | 69696 | 150 | 44 | 984 | 6.5 / 240 | 24.4% | 38.8% | 299, 0.38 s | 1051, 12.44 s |
| grid_L | 14256 | 112360 | 132 | 36 | 3605 | 10.5 / 1650 | 45.8% | 54.0% | 263, 0.52 s | 915, 15.51 s |
| dag_S | 2000 | 4014 | 303 | 268 | 1958 | 1.0 / 583 | 29.5% | 68.6% | 605, 0.14 s | 2414, 3.81 s |
| dag_M | 20000 | 59979 | 873 | 729 | 19803 | 1.0 / 14198 | 71.8% | 93.1% | 1745, 1.88 s | – |
| grid_XL | 54000 | 443576 | 345 | 86 | 10102 | 12.0 / 2986 | 29.5% | 51.9% | 689, 3.11 s | – |

Reading. The geometric bisection (Eisner–Severance/Gusfield style, one max flow per breakpoint on the contracted residual graph) computes the entire canonical sequence, hence the whole value function z(c), in about 2k max flows and a few seconds even for 54 000 blocks and 444 000 arcs. Repeated Dinkelbach extraction returns exactly the same sequence but needs 3–4 max flows per macroitem on the *uncontracted* residual graph, and is 10–40× slower. In the mining-like instances the first macroitem is large (83 to 1 842 blocks, 24–46% of the tonnage of the maximum-profit pit); in random DAGs a single macroitem contains most of the items (14 198 of 20 000 in dag_M): the poset is close to indecomposable in Sidney's sense.

## 2. One capacity: Newton search on the weight price vs. LP solver

| instance | c / w(M_q) | Newton: max flows | Newton time | HiGHS time | speed-up | rel. diff of z | |H| | θ | dual certificate |
|---|---|---|---|---|---|---|---|---|---|
| grid_S | 0.25 | 5 | 0.006 s | 0.02 s | 3× | 2.6e-16 | 20 | 0.024 | feasible, value = z |
| grid_S | 0.50 | 7 | 0.006 s | 0.02 s | 3× | 2.8e-16 | 30 | 0.289 | feasible, value = z |
| grid_S | 0.75 | 6 | 0.005 s | 0.02 s | 3× | 3.4e-16 | 18 | 0.044 | feasible, value = z |
| grid_M | 0.25 | 7 | 0.045 s | 0.30 s | 7× | 0.0e+00 | 39 | 0.139 | feasible, value = z |
| grid_M | 0.50 | 7 | 0.042 s | 0.33 s | 8× | 1.5e-16 | 36 | 0.428 | feasible, value = z |
| grid_M | 0.75 | 7 | 0.038 s | 0.46 s | 12× | 1.3e-16 | 53 | 0.208 | feasible, value = z |
| grid_L | 0.25 | 4 | 0.099 s | 2.57 s | 26× | 3.5e-15 | 1650 | 0.546 | feasible, value = z |
| grid_L | 0.50 | 6 | 0.093 s | 1.95 s | 21× | 0.0e+00 | 110 | 0.363 | feasible, value = z |
| grid_L | 0.75 | 8 | 0.086 s | 1.38 s | 16× | 0.0e+00 | 130 | 0.088 | feasible, value = z |
| dag_S | 0.25 | 7 | 0.024 s | 0.10 s | 4× | 1.3e-16 | 343 | 0.319 | feasible, value = z |
| dag_S | 0.50 | 7 | 0.027 s | 0.12 s | 5× | 1.5e-16 | 583 | 0.363 | feasible, value = z |
| dag_S | 0.75 | 7 | 0.026 s | 0.11 s | 4× | 0.0e+00 | 399 | 0.297 | feasible, value = z |
| dag_M | 0.25 | 7 | 1.628 s | 29.33 s | 18× | 1.9e-14 | 14198 | 0.288 | feasible, value = z |
| dag_M | 0.50 | 7 | 1.523 s | 21.33 s | 14× | 1.1e-14 | 14198 | 0.636 | feasible, value = z |
| dag_M | 0.75 | 7 | 1.475 s | 15.33 s | 10× | 1.7e-14 | 14198 | 0.984 | feasible, value = z |
| grid_XL | 0.25 | 8 | 0.456 s | 22.99 s | 50× | 5.0e-16 | 108 | 0.062 | feasible, value = z |
| grid_XL | 0.50 | 6 | 0.428 s | 32.28 s | 75× | 1.0e-14 | 2986 | 0.385 | feasible, value = z |
| grid_XL | 0.75 | 9 | 0.450 s | 20.95 s | 47× | 1.5e-15 | 413 | 0.523 | feasible, value = z |

Reading. Solving the LP at one capacity takes 4–9 maximum closures (Newton steps on the convex piecewise-linear Lagrangian function, each on a shrinking residual graph) and is 3× to 75× faster than HiGHS in this pure-Python implementation, the gap widening with size. The three region flows of the canonical dual (Theorem on the dual face) were computed by three more max flows and verified feasible in every case.

## 3. Structure along the capacity axis

For c = f·w(M_q): split index h, size of the split macroitem H, fractional level θ, gap bound θ·p(I_h) relative to z_LP (the paper's Proposition on the integrality gap), heuristic gap (z_LP − z_heur)/z_LP with the greedy fill of the split macroitem, persistency (fraction of items fixed to 0/1 in *every* LP optimum), number k₀ of inseparability classes of H (dim of the primal face = k₀ − 1) and dimension of the dual optimal face (formula of the note; computed when |H| ≤ 3000).

| instance | f | h | |H| | θ | w(H)/c | gap bound / z_LP | heuristic gap | persistency | k₀ | dim dual face |
|---|---|---|---|---|---|---|---|---|---|---|
| grid_S | 0.1 | 1 | 83 | 0.40 | 2.49 | 100.00% | 97.61% | 92.7951% | 1 | 4425 |
| grid_S | 0.2 | 1 | 83 | 0.80 | 1.24 | 100.00% | 40.49% | 92.7951% | 1 | 4425 |
| grid_S | 0.3 | 2 | 20 | 0.85 | 0.20 | 15.76% | 11.05% | 98.2639% | 1 | 4418 |
| grid_S | 0.4 | 3 | 45 | 0.67 | 0.34 | 19.65% | 18.34% | 96.0938% | 1 | 4272 |
| grid_S | 0.5 | 5 | 30 | 0.29 | 0.18 | 3.72% | 3.35% | 97.3958% | 1 | 4232 |
| grid_S | 0.6 | 9 | 11 | 0.08 | 0.05 | 0.19% | 0.19% | 99.0451% | 1 | 4270 |
| grid_S | 0.7 | 11 | 5 | 0.96 | 0.02 | 0.71% | 0.71% | 99.5660% | 1 | 4257 |
| grid_S | 0.8 | 14 | 18 | 0.97 | 0.07 | 2.18% | 2.18% | 98.4375% | 1 | 4176 |
| grid_S | 0.9 | 20 | 4 | 0.50 | 0.01 | 0.10% | 0.10% | 99.6528% | 1 | 4187 |
| grid_M | 0.1 | 1 | 240 | 0.41 | 2.44 | 100.00% | 93.13% | 97.3333% | 1 | 68485 |
| grid_M | 0.2 | 1 | 240 | 0.82 | 1.22 | 100.00% | 31.97% | 97.3333% | 1 | 68485 |
| grid_M | 0.3 | 3 | 30 | 0.50 | 0.10 | 4.69% | 3.68% | 99.6667% | 1 | 68377 |
| grid_M | 0.4 | 7 | 35 | 0.03 | 0.09 | 0.19% | 0.19% | 99.6111% | 1 | 68033 |
| grid_M | 0.5 | 15 | 36 | 0.43 | 0.07 | 1.69% | 1.69% | 99.6000% | 1 | 67810 |
| grid_M | 0.6 | 23 | 30 | 0.31 | 0.05 | 0.57% | 0.57% | 99.6667% | 1 | 67603 |
| grid_M | 0.7 | 27 | 35 | 0.29 | 0.05 | 0.48% | 0.48% | 99.6111% | 1 | 67322 |
| grid_M | 0.8 | 33 | 40 | 0.16 | 0.05 | 0.17% | 0.17% | 99.5556% | 1 | 67083 |
| grid_M | 0.9 | 39 | 57 | 0.51 | 0.07 | 0.32% | 0.32% | 99.3667% | 1 | 66793 |
| grid_L | 0.1 | 1 | 1650 | 0.22 | 4.58 | 100.00% | 100.00% | 88.4259% | 1 | 106931 |
| grid_L | 0.2 | 1 | 1650 | 0.44 | 2.29 | 100.00% | 100.00% | 88.4259% | 1 | 106931 |
| grid_L | 0.3 | 1 | 1650 | 0.65 | 1.53 | 100.00% | 100.00% | 88.4259% | 1 | 106931 |
| grid_L | 0.4 | 1 | 1650 | 0.87 | 1.15 | 100.00% | 100.00% | 88.4259% | 1 | 106931 |
| grid_L | 0.5 | 3 | 110 | 0.36 | 0.06 | 1.66% | 1.66% | 99.2284% | 1 | 107571 |
| grid_L | 0.6 | 6 | 131 | 0.06 | 0.06 | 0.26% | 0.26% | 99.0811% | 1 | 106842 |
| grid_L | 0.7 | 11 | 140 | 0.81 | 0.06 | 2.21% | 2.21% | 99.0180% | 1 | 106425 |
| grid_L | 0.8 | 23 | 120 | 0.13 | 0.04 | 0.15% | 0.15% | 99.1582% | 1 | 106013 |
| grid_L | 0.9 | 33 | 131 | 0.31 | 0.04 | 0.05% | 0.05% | 99.0811% | 1 | 105537 |
| dag_S | 0.1 | 140 | 1 | 0.68 | 0.01 | 0.35% | 0.35% | 99.9500% | 1 | 3640 |
| dag_S | 0.2 | 168 | 343 | 0.04 | 0.91 | 2.68% | 0.51% | 82.8500% | 1 | 3136 |
| dag_S | 0.3 | 168 | 343 | 0.59 | 0.61 | 27.18% | 7.23% | 82.8500% | 1 | 3136 |
| dag_S | 0.4 | 180 | 583 | 0.02 | 0.74 | 1.28% | 0.18% | 70.8500% | 1 | 2821 |
| dag_S | 0.5 | 180 | 583 | 0.36 | 0.59 | 16.89% | 9.76% | 70.8500% | 1 | 2821 |
| dag_S | 0.6 | 180 | 583 | 0.70 | 0.49 | 28.23% | 22.08% | 70.8500% | 1 | 2821 |
| dag_S | 0.7 | 181 | 399 | 0.06 | 0.30 | 1.46% | 0.48% | 80.0500% | 1 | 3110 |
| dag_S | 0.8 | 181 | 399 | 0.54 | 0.26 | 12.00% | 3.62% | 80.0500% | 1 | 3110 |
| dag_S | 0.9 | 184 | 1 | 0.78 | 0.00 | 0.06% | 0.06% | 99.9500% | 1 | 3663 |
| dag_M | 0.1 | 444 | 14198 | 0.08 | 7.18 | 43.58% | 35.70% | 29.0100% | – | – |
| dag_M | 0.2 | 444 | 14198 | 0.22 | 3.59 | 68.18% | 63.74% | 29.0100% | – | – |
| dag_M | 0.3 | 444 | 14198 | 0.36 | 2.39 | 77.84% | 74.75% | 29.0100% | – | – |
| dag_M | 0.4 | 444 | 14198 | 0.50 | 1.79 | 83.00% | 80.63% | 29.0100% | – | – |
| dag_M | 0.5 | 444 | 14198 | 0.64 | 1.44 | 86.22% | 84.29% | 29.0100% | – | – |
| dag_M | 0.6 | 444 | 14198 | 0.78 | 1.20 | 88.41% | 86.79% | 29.0100% | – | – |
| dag_M | 0.7 | 444 | 14198 | 0.91 | 1.03 | 90.00% | 88.60% | 29.0100% | – | – |
| dag_M | 0.8 | 451 | 4004 | 0.19 | 0.25 | 4.53% | 3.02% | 79.9800% | – | – |
| dag_M | 0.9 | 451 | 4004 | 0.68 | 0.23 | 14.66% | 13.31% | 79.9800% | – | – |
| grid_XL | 0.1 | 1 | 1842 | 0.55 | 1.82 | 100.00% | 100.00% | 96.5889% | – | – |
| grid_XL | 0.2 | 2 | 243 | 0.73 | 0.12 | 8.66% | 8.66% | 99.5500% | 1 | 437641 |
| grid_XL | 0.3 | 15 | 145 | 0.51 | 0.05 | 1.61% | 1.59% | 99.7315% | 1 | 436658 |
| grid_XL | 0.4 | 33 | 2986 | 0.05 | 0.74 | 1.35% | 1.35% | 94.4704% | – | – |
| grid_XL | 0.5 | 33 | 2986 | 0.38 | 0.59 | 10.25% | 10.25% | 94.4704% | – | – |
| grid_XL | 0.6 | 33 | 2986 | 0.72 | 0.49 | 17.68% | 17.68% | 94.4704% | – | – |
| grid_XL | 0.7 | 35 | 245 | 0.74 | 0.03 | 1.29% | 1.29% | 99.5463% | 1 | 431063 |
| grid_XL | 0.8 | 47 | 2 | 0.56 | 0.00 | 0.01% | 0.00% | 99.9963% | 2 | 431330 |
| grid_XL | 0.9 | 64 | 219 | 0.24 | 0.02 | 0.15% | 0.15% | 99.5944% | 1 | 428804 |

Reading. (i) The *gap problem* of mine planning appears exactly where the paper predicts: when the capacity is below the weight of the first macroitem (f ≤ 0.2–0.4 in the grids), the LP solution is entirely fractional and the bound θ·p(I_h) equals the whole LP value; as soon as the split macroitem is small relative to c, the bound drops below 2% and the greedy fill closes most of it. (ii) Random DAGs are worse: one macroitem of 14 198 items makes the relaxation almost uninformative for 0.1 ≤ f ≤ 0.7. (iii) The primal optimum is unique (k₀ = 1) in all but one case (grid_XL, f = 0.8, a tie between two blocks), whereas the dual face has dimension in the thousands to hundreds of thousands: essentially every arc inside the full and the null regions carries a free multiplier. The canonical dual is one point of this huge face, which is why the choice of reduced costs on the face (the note's Proposition on best reduced costs) matters in practice. (iv) Persistency is above 97% of the items except when the split macroitem is huge.

## 4. Weight parameterization vs. revenue-factor parameterization

| instance | revenue-factor pits tested | coinciding with some M_r | mean relative symmetric difference to the nearest-weight M_r |
|---|---|---|---|
| grid_S | 20 | 15 | 0.0011 |
| grid_M | 20 | 12 | 0.0017 |
| grid_L | 20 | 15 | 0.0002 |
| grid_XL | 20 | 13 | 0.0021 |

Reading. On these synthetic models (tonnage almost constant, revenue varying with grade) the two nested families are close but not identical: 5–8 of 20 revenue-factor pits are not closures of the weight path. Real block models with variable density should separate the two families more; this is the experiment to run on MineLib.

## 5. What the pilot says about the plan

The library core (instance format, Picard network on a fast max flow with maximal/minimal tie handling, geometric bisection, Dinkelbach, Newton at a capacity, canonical dual, face dimensions, LP baseline, random cross-checks) is in place and exact on integer data. The structural statistics are informative and directly tied to the paper's statements (gap bound, persistency, face dimensions, tie conventions). Next steps, in order: MineLib converters and the real instances; a compiled max-flow core (the Python overhead dominates for small graphs); Hochbaum's parametric pseudoflow as a third path algorithm; best reduced costs on the dual face; the multi-capacity warm start for branch-and-bound use.
