# Experimental report

What this library computes, how it was checked, and how it compares with
general-purpose LP solvers on the published instances of the field.

Everything here is reproducible from this repository: the environment and the
protocol are in [experiments/results/ENVIRONMENT.md](experiments/results/ENVIRONMENT.md),
the raw measurements are the CSV files in `experiments/results/`, and the
tables below are generated from those files by `experiments/make_tables.py`.

---

## 1. Summary

**The question.** The LP relaxation of the precedence-constrained knapsack
problem can be solved by a general-purpose LP solver, or through the
parametric maximum-closure structure this library implements. Which is
better, and when?

**The answer.** It depends on one thing that is visible before any
computation, and on one that is not:

1. **Do you need one capacity, or the value function?** The canonical path
   returns *every* capacity at once. If you need more than a couple of
   capacities — as any Lagrangian, branch-and-bound or multi-period scheme
   does — nothing else is close.
2. **How dense is the precedence graph?** On sparse graphs a warm-started
   dual simplex matches the combinatorial method. On the dense graphs of real
   mine planning the path wins by an order of magnitude, and the margin grows
   with size.

And one finding that was not expected: on one instance the LP solvers return
a value that is **wrong in the eighth significant digit**, while the
combinatorial method is exact. Accuracy, not only speed, is part of the
answer.

| question | answer | evidence |
|---|---|---|
| one capacity | Newton search on the weight price | median 1.5x the fastest LP solver (range 0.4–27x) |
| the whole value function, sparse graphs (m < 10⁴) | either | median 1.00x — a tie |
| the whole value function, dense graphs (m ≥ 2·10⁴) | the canonical path | median **11.2x**, up to **29x** |
| an LP solver is unavoidable | dual simplex, never barrier | interior point cannot warm start: every capacity costs a full solve |
| exact values are needed | the combinatorial method | it is exact on integer data; see §4 |

---

## 2. Instances

No instance data is redistributed here; see
[docs/instances.md](docs/instances.md) for where to obtain it.

| collection | instances | sizes |
|---|---|---|
| PCKP benchmark, *telecom* (A–K) | 11 | n = 972–9 235, m/n ≈ 1.8 |
| PCKP benchmark, *mining* (L–W) | 12 | n = 349–11 757, m/n ≈ 6–7 |
| MineLib | 10 | n = 1 060–112 687, up to m = 3 035 483 |

Full structural characterization:
[experiments/results/tables/structure.md](experiments/results/tables/structure.md).

---

## 3. What the relaxation decides, and what it does not

The *persistency* of Corollary 5.1 — the fraction `1 - |H|/n` of items whose
value is the same in **every** optimal solution — is the practical measure of
how much the relaxation settles by itself. It varies enormously across real
instances:

| | instance | persistency | split macroitem |
|---|---|---|---|
| lowest | O_1711_11661 | 0.150 | 1 455 of 1 711 items |
| | newman1 | 0.166 | 884 of 1 060 blocks |
| | U_6494_48626 | 0.262 | 4 790 of 6 494 |
| highest | p4hd | 0.992 | 329 of 40 947 |
| | sm2, zuck_large | 0.9999 | 10 of ~100 000 |

On sm2 and zuck_large the relaxation fixes all but ten blocks out of a
hundred thousand. On newman1 and O_1711_11661 it fixes almost nothing. This
is the *gap problem* of the mining literature, measured rather than
described, and Proposition 5.2 identifies exactly what distinguishes the two
groups: the weight of the split macroitem relative to the capacity.

The figures show it directly:
[newman1_persistency.png](experiments/results/figures/newman1_persistency.png)
has persistency near 1 only for capacities below 8% of `w(M_q)`, then it
collapses to 0.166 across the entire practical range while the gap bound
climbs to 0.84.

### A remark on the published capacities

On **all 23** benchmark instances, the capacity distributed with the instance
is between 0.5% and 25% of `w(M_q)`, and consequently the split macroitem is
the *first* one: `h = 1` everywhere. In that regime the optimal solution is a
multiple of a single macroitem's indicator, the bound of Proposition 5.2 is
vacuous (it equals the whole LP value), and the true integrality gap against
the known integer optima is indeed large: median **26.6%**, up to **85.6%**
on P_3243_22306.

That is worth stating plainly: the standard benchmark of this problem
exercises precisely the regime in which the natural relaxation is weakest,
which is consistent with that literature's focus on cutting planes.

---

## 4. Agreement between methods, and one disagreement

Every method was run on every instance at every capacity, and all values
compared against the canonical path.

**PCKP benchmark: complete agreement.** Across 161 method-runs and 20
capacities each, the largest relative difference is **4.8 · 10⁻¹¹**. No
disagreement above 10⁻⁹.

**MineLib kd: the solvers are wrong.** At one capacity out of twenty, all
five LP solver configurations return

```
LP solvers:    220 239 941.533204
this library:  220 239 938.251054
```

a relative difference of 1.5 · 10⁻⁸. The exact value, computed in rational
arithmetic from the integer-scaled instance, is

```
1238890176810816816310940159 / 5625184000000  =  220 239 938.251054
```

so the library is right and the solvers are off by 3.28. The discrepancy is
not an artefact of the integer rescaling: Gurobi returns the same wrong value
on the original decimal data. kd has profits of order 4 · 10⁵ and weights of
order 10⁴, and at that particular capacity the LP is ill-conditioned enough
for a double-precision solver's default tolerances.

This is the practical argument for the combinatorial route that is easiest to
overlook: on integer data — which covers all 23 benchmark instances and six
of the ten MineLib ones after exact decimal rescaling — the parametric
machinery is *exact*. The breakpoints are rationals and no tolerance decides
which macroitem an item belongs to.

---

## 5. How best to solve it

Full table: [experiments/results/tables/methods.md](experiments/results/tables/methods.md).
Each method runs in its own process; build and solve are timed separately;
the LP solvers build their model once and re-solve by changing only the
capacity right-hand side, so their simplex bases carry over.

### The whole value function, 20 capacities (PCKP benchmark)

| instance | n | m | path | best solver | ratio |
|---|---|---|---|---|---|
| A_972_1661 | 972 | 1 661 | 0.057 s | 0.029 s | 0.5x |
| C_1336_2382 | 1 336 | 2 382 | 0.088 s | 0.056 s | 0.6x |
| J_9235_17082 | 9 235 | 17 082 | 0.337 s | 1.141 s | 3.4x |
| P_3243_22306 | 3 243 | 22 306 | 0.071 s | 0.575 s | 8.0x |
| T_6271_42080 | 6 271 | 42 080 | 0.249 s | 3.012 s | 12.1x |
| V_10001_63944 | 10 001 | 63 944 | 0.471 s | 5.598 s | 11.9x |
| W_11757_83218 | 11 757 | 83 218 | 0.420 s | 12.190 s | **29.0x** |

Grouping by density rather than by size:

* sparse (m < 10⁴, 12 instances): median **1.00x**, range 0.47–2.44
* dense (m ≥ 2·10⁴, 7 instances): median **11.2x**, range 5.0–29.0

Size alone does not predict the outcome — the sparse group contains instances
with 9 235 items on which the methods tie. What separates the groups is the
number of precedence arcs.

On MineLib the effect is stronger still: on zuck_small (9 400 blocks,
145 640 precedences) the path computes the entire value function in 0.58 s
against 26.3 s for the fastest solver over 20 capacities, and on zuck_medium
(29 277 blocks, 1 271 207 precedences) in 3.45 s against 1 354 s for HiGHS
dual simplex — a factor of 390.

### One capacity

The Newton search on the weight price costs a handful of maximum closures.
Against the fastest of the three simplex codes it is a median **1.5x**
faster, with a wide range (0.38–26.9x): on small sparse instances a
warm-started dual simplex is at least as good.

### Interior point

Barrier and IPM cannot start from a previous basis, so their marginal cost
per capacity equals their full cost. Over a grid of capacities they were
10–30x worse than dual simplex in every run. This matters because barrier is
a common default on large sparse LPs.

### Parametric minimum cut: no usable public implementation

We intended to include the public `pseudoflow` package as a further
comparison, and could not. Beyond its own documented caveat — it is a
simplified variant of parametric HPF, without free runs or warm starts, and
"should **not** be used" for comparison with the full algorithm — we found
that it **silently returns an incomplete parametric family**.

The smallest failing instance has three items, no precedence arcs, profits
(3, 2, 1) and unit weights. The canonical sequence is {0}, {1}, {2} with
ratios 3 > 2 > 1. The package reports two intervals instead of three, and the
source set it returns for the interval containing λ = 3.5 has value −0.5 — it
is not a minimum cut. Failures appear from n = 13 upwards and become common
at n ≥ 200.

What makes this dangerous is that the output looks correct: the family
returned is always nested, complete, made of genuine closures with strictly
decreasing ratios — a *coarsening* of the canonical sequence. Detecting the
defect costs one maximum closure per macroitem, i.e. recomputing the answer.

`macroitems/pseudoflow_path.py` implements and tests the reduction, then
raises `NotImplementedError` with this explanation. There is likewise no
public implementation of Gallo–Grigoriadis–Tarjan (1989) we would trust for
benchmarking.

---

## 6. Dual certificates, reduced costs, faces

The canonical dual certificate is three maximum flows, one per region. It was
feasible with complementary slackness to machine precision on every instance
tried.

Reduced costs over the *whole* dual optimal face cost one minimum cut per
item and are never smaller than the closed-form canonical ones
`w_i|λ_r − λ_h|`. Both were validated against the exact loss, obtained by
re-solving the relaxation with the item forced to its opposite bound: over 12
instances no bound exceeded the true loss and none fell below the canonical
value. On the paper's running example the face-wide values are exactly twice
the canonical ones.

Face dimensions reproduce the note's description of the running example:
`dim X* = 0` (the primal optimum is unique) and `dim D* = 3`.

---

## 7. Weight against revenue factor

Nested pits are usually generated in practice by scaling revenue at fixed
cost, rather than by pricing weight. Remark 5.3 states the two families
differ; on block models whose revenue and cost are known by construction,
only **5 of 20** revenue-factor pits coincide with a canonical closure, and
the relative symmetric difference to the canonical closure of the nearest
tonnage reaches **0.37**.

On MineLib the experiment is blocked by the data rather than the method: the
UPIT and CPIT files give a single value per block, from which revenue and
cost cannot be separated. Only the PCPSP formulation lists a value per
destination.

---

## 8. Correctness

`pytest` runs **773 tests** in about 90 seconds. The suite is the reason to
trust the numbers above.

| what | how |
|---|---|
| the whole theory on small instances | a brute-force reference in exact rational arithmetic that enumerates every closure: `u(λ)` and both lattice extremes at every breakpoint, the canonical sequence from its definition, `z(c)` over a capacity grid, persistency against the full optimal face |
| invariants on arbitrary instances | property-based tests (hypothesis) on ~900 generated instances |
| the two path algorithms | bisection ≡ Dinkelbach on 100 random instances of varying size and density |
| the three maximum-flow backends | identical closures and paths on integer data |
| the LP baselines | agreement with the combinatorial methods to 10⁻⁹, and among themselves |
| integer rescaling | ratios invariant, profits/weights/values scaled — see §9 |
| **Dantzig** | with no arcs, the theory must reduce to the greedy ratio rule, checked against an independent implementation |
| **Shaw–Cho (1998)** | their aggregated subtrees are our macroitems and their bound is `z(c)`, on 43 tree instances at ~1 500 capacities |
| **Sidney (1975) / Margot et al. (2003)** | the reduced Sidney decomposition, computed by brute force over all initial sets, is the canonical sequence after reversing the arcs and swapping the data, on 57 instances |

The literature checks are not vacuous: dropping the arc reversal in the
scheduling translation breaks 188 of 190 instances.

---

## 9. Defects found during development

Recorded because the same failure modes are available to anyone
reimplementing these classical algorithms. All four produced plausible,
wrong answers rather than errors.

1. **Reduced costs, sign inverted on the full region.** The two regions
   optimize different objectives — `v` on Z, `−v` on the reversed graph of F
   — and getting it wrong yields bounds that are invalid but entirely
   believable. Caught only by comparison against the true LP loss.
2. **`solution_from_path` at c = 0 returned an infeasible point.**
   `split_index(0)` is 0, so `closure_mask(n, −1)` wrapped around a negative
   Python slice and selected all but the last macroitem.
3. **The scipy maximum-flow backend truncated silently.** It computes in
   int32; on integer data with wide coefficients — where the scaled
   parametric values and the big-M on precedence arcs exceed int32 — it
   returned a *different canonical path* from the other two backends. Such
   capacities are now refused with an error.
4. **Ratios divided by the integer scale in reporting code.** A ratio `p/w`
   is invariant under rescaling; the cumulative profits and weights are not.
   This would have published breakpoints wrong by orders of magnitude while
   every other number looked right. `tests/test_scaling.py` pins it.

---

## 10. Limitations

* The pseudoflow comparison is absent, for the reason in §5.
* The CPLEX Python package on PyPI is the Community Edition and refuses
  models above 1000 rows; the licensed solver is reached through its
  Interactive Optimizer (`cplex-cli`), which re-optimizes from the previous
  basis exactly as the in-process backends do.
* `highspy` and `ortools` cannot be loaded in the same process (see
  [docs/solvers.md](docs/solvers.md)); every experiment runs one method per
  process, which is good timing practice anyway.
* mclaughlin (2 140 342 blocks) is not included.
* Timings come from one machine; the *values* are machine-independent and any
  disagreement above 10⁻⁹ is a bug worth reporting.
