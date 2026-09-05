# What the library computes

← [Summary](01-summary.md) · [Contents](README.md) · next: [Definitions and conventions](03-definitions-and-conventions.md)

---

This page is the inventory: what was implemented, and what each piece costs.
[Correctness](06-correctness.md) is the companion page — what each piece was
checked against.

Everything below reduces to maximum closure, which is one minimum cut on a
Picard network.

| | what it returns | cost |
|---|---|---|
| `canonical_path` | the macroitems `I_1 … I_k`, their ratios, and hence `z(c)` for **every** capacity | `O(k)` maximum flows on shrinking graphs |
| `solve_capacity` | the LP at one capacity | a handful of maximum flows |
| `solution_from_path` | any capacity from a precomputed path | `O(log k)`, no flow |
| `canonical_dual` | the dual certificate: capacity price and three region-wise flows | three maximum flows |
| `best_reduced_costs` | reduced costs over the **whole** dual optimal face | one minimum cut per item |
| `face_dimensions` | dimensions of the primal and dual optimal faces | `|H|` minimum cuts |
| `first_macroitem_lawler` | Lawler's binary search, for comparison | `O(log)` maximum flows |
| `macroitems.lp` | LP baselines behind one interface (HiGHS, Gurobi, CPLEX) | — |

## The object

For a weight price `λ` the node values are `v_i = p_i − λ w_i`. As `λ`
decreases, the inclusion-wise maximal optimal closures form a nested chain

```
∅ = M_0 ⊂ M_1 ⊂ … ⊂ M_k = I,     λ_1 > λ_2 > … > λ_k,
```

and the increments `I_r = M_r \ M_{r-1}` — the **macroitems** — satisfy
`p(I_r)/w(I_r) = λ_r` exactly. The relaxation is then a knapsack LP on
macroitems: earlier ones fully selected, **one** split, the rest null.

That single sentence is why the whole value function comes out at once: the
piecewise-linear `z(c)` is the concave interpolation of the cumulative points
`(w(M_r), p(M_r))`.

![value function](../../experiments/results/figures/kd_value-function.png)

*MineLib `kd`: 14 153 blocks, 219 778 precedences, 493 macroitems. The upper
panel is `z(c)`, its breakpoints the cumulative macroitem points; the lower bar
shows the macroitems along the weight axis, shaded by ratio, the first one
outlined. Computing all of it took 1.2 seconds.*

## The infrastructure around it

Three things were built because the results depend on them, not as
conveniences.

**Three interchangeable maximum-flow backends** — OR-Tools (int64, exact, the
default), igraph (floating point), SciPy (int32). They must produce identical
closures and paths on integer data, and the suite checks it. Having more than
one is not redundancy: it is how a wrong answer in one of them becomes
visible ([Defects found](13-defects-found.md), item 3).

**Exact arithmetic wherever the data allow it.** Decimal data are scaled to
integers by reading the number of decimals off each value's shortest decimal
representation and scaling in decimal arithmetic — multiplying the floats
instead fails on data that *are* decimal, because a value like `-2236.7886`
is not exactly representable and the error is amplified. Closures, macroitems
and ratios are invariant under the scaling, so the answer is unchanged, but
the parametric values become integers and the computation is exact. Six of the
ten MineLib instances and all 23 benchmark instances enter this regime, and
the dual certificate and the face dimensions are made exact the same way.

**Five LP baselines behind one interface** — HiGHS, SciPy, Gurobi, CPLEX, and
a CPLEX backend driving a licensed Interactive Optimizer over a pipe, which is
necessary because the CPLEX distribution on PyPI is the Community Edition and
refuses models above 1000 rows, that is, every instance here. Each builds its
model once and re-solves by changing only the capacity right-hand side, so
simplex bases carry over; build and solve are timed separately. **No
third-party solver code is bundled** — the solvers are linked and installed
by the user under their own licences, which is what lets an MIT package be
used with GPL and commercial software.

Plus instance readers for both published collections, a command line, and the
experiment scripts that generate every table and figure in this report from
the raw CSV files.

## What was implemented and then disabled

The reduction of the canonical path to a single parametric minimum cut is
implemented and tested — and deliberately disabled, because the public
implementation it would call returns an incomplete family without saying so.
The reduction is kept so the method can be re-enabled if the package is fixed.
See [Parametric implementations](11-parametric-implementations.md).

## What comes for free with it

At a given capacity the split into three regions — `F` full, `H` split, `Z`
null — says that **every** optimal solution agrees outside `H`. The fraction
`1 − |H|/n` is the *persistency*, and on real instances it ranges from 0.15 to
0.9999 ([Structure of real instances](07-structure-of-real-instances.md)).

The dual side gives a certificate and reduced costs that are stronger than the
textbook knapsack ones, because they are optimized over the whole dual face
([Dual certificates and reduced costs](10-dual-certificates-and-reduced-costs.md)).
