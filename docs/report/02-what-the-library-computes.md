# What the library computes

← [Summary](01-summary.md) · [Contents](README.md) · next: [Definitions and conventions](03-definitions-and-conventions.md)

---

Everything below reduces to maximum closure, which is one minimum cut.

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

## What comes for free with it

At a given capacity the split into three regions — `F` full, `H` split, `Z`
null — says that **every** optimal solution agrees outside `H`. The fraction
`1 − |H|/n` is the *persistency*, and on real instances it ranges from 0.15 to
0.9999 ([Structure of real instances](07-structure-of-real-instances.md)).

The dual side gives a certificate and reduced costs that are stronger than the
textbook knapsack ones, because they are optimized over the whole dual face
([Dual certificates and reduced costs](10-dual-certificates-and-reduced-costs.md)).
