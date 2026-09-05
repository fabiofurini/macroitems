# Definitions and conventions

← [What the library computes](02-what-the-library-computes.md) · [Contents](README.md) · next: [Setup and protocol](04-setup.md)

---


Everything in this library follows one set of conventions, taken from the
companion papers. They are stated here once, because most of the confusion in
this subject comes from conventions differing silently between communities.

## The problem

Items `I = {0, ..., n-1}`, each with a profit `p_i` of **arbitrary sign** and a
**strictly positive** weight `w_i`. A directed acyclic graph of *precedences*
on the items. The precedence-constrained knapsack problem asks for a
maximum-profit set of total weight at most `c` that is closed under
prerequisites; this library solves its **LP relaxation**

```
max  sum_i p_i x_i
s.t. sum_i w_i x_i <= c
     x_i - x_j <= 0          for every arc (i, j)
     0 <= x_i <= 1
```

## Arc direction — the one thing to get right

**An arc `(i, j)` means "j is a prerequisite of i"**, and therefore the
constraint `x_i <= x_j`: you cannot take `i` without taking `j`.

In `Instance.arcs`, an array of shape `(m, 2)`, row `k` is `(i, j)` in that
order: **dependent first, prerequisite second**.

Other communities orient this differently, and the readers translate:

| source | its statement | what we store |
|---|---|---|
| MineLib `.prec` | `b k b_1 ... b_k`: block `b` needs `b_1..b_k` extracted first | `(b, b_r)` — already ours |
| PCKP benchmark `.dat` | `id profit weight n_pred pred...` | `(id, pred)` — already ours |
| CPLEX LP row | `x_i - x_j <= 0` | `(i, j)` — already ours |
| scheduling (Sidney) | `(i, j) in R`: job `i` must precede job `j` | `(j, i)` — **reversed** |

The scheduling translation also swaps the roles of the data: processing time
becomes the weight and deferral rate the profit, so minimizing `p(U)/w(U)`
over initial sets becomes maximizing our ratio over closures.

## Closures and the ratio

A set `C` is a **closure** if no arc leaves it: `i in C` and `(i, j)` an arc
imply `j in C`. For a nonempty `S`, its **ratio** is
`rho(S) = p(S) / w(S)`, well defined because weights are positive.

## The parametric path and macroitems

For a weight price `lambda`, node values are `v_i = p_i - lambda * w_i` and
`u(lambda) = max over closures of v(C)`. The optimal closures at a given
`lambda` form a lattice, so there is a unique inclusion-wise **minimal** and a
unique **maximal** one.

**Tie convention: this library always takes the maximal one.** It is what
makes the canonical sequence unique, it is Sidney's Algorithm 1\*, and it is
the *reduced* Sidney decomposition of Margot, Queyranne and Wang. The minimal
one is available as `tie="min"` where it is needed (the tight sets of the
face-dimension computation).

As `lambda` decreases the maximal optimal closures are nested,

```
empty = M_0 ⊂ M_1 ⊂ ... ⊂ M_k = I,     lambda_1 > lambda_2 > ... > lambda_k,
```

and the increments `I_r = M_r \ M_{r-1}` are the **macroitems**, with
`rho(I_r) = lambda_r` exactly. `q` is the last index with `lambda_q > 0`;
`M_q` is the maximum-profit closure, and no capacity beyond `w(M_q)` is
interesting.

## The solution at one capacity

Let `h` be the first index with `w(M_h) > c`, and
`theta = (c - w(M_{h-1})) / w(I_h)`. The canonical optimum is `1` on
`M_{h-1}`, `theta` on `I_h`, `0` elsewhere. This splits the items into three
**regions**, named the same way throughout the code:

| region | attribute | meaning |
|---|---|---|
| `F` | `sol.F` | full: `M_{h-1}`, `x_i = 1` in *every* optimum |
| `H` | `sol.H` | the split macroitem `I_h`, the only place optima can differ |
| `Z` | `sol.Z` | null: outside `M_h`, `x_i = 0` in every optimum |

`1 - |H| / n` is the **persistency**: the fraction of items whose value is
decided by the relaxation alone.

## The dual

With `div_i(alpha) = sum over arcs (i, j) of alpha_ij - sum over arcs (j, i)
of alpha_ji`, the dual is

```
min  c * lambda + sum_i mu_i
s.t. w_i * lambda + mu_i + div_i(alpha) >= p_i     for every item i
     lambda >= 0,  mu >= 0,  alpha >= 0
```

`lambda` is the capacity price and equals `lambda_h` at a nondegenerate
capacity; `mu_i` prices the upper bound on `x_i`; `alpha_ij` prices the
precedence arc. Every solver in `macroitems.lp` reports `lambda` with this
sign, i.e. `lambda = dz/dc >= 0`.

## Exact arithmetic

On integer data every step is exact: the breakpoints are rationals, recovered
as `p(I_r)/w(I_r)`, and the parametric values `b*p - a*w` are integers, so an
integer maximum-flow backend gives exact cuts. Decimal data are scaled to
integers by `Instance.scaled_to_integers()` when that is possible without
losing exactness; otherwise the library works in floating point and says so.

## Degenerate capacities

Two cases behave differently and are reported, not hidden:

* `c = w(M_h)` for some `h` (a *cumulative* capacity): the optimum is integral
  and the set of optimal multipliers is the whole interval
  `[max(0, lambda_{h+1}), lambda_h]`. Different solvers legitimately report
  different values of `lambda` here, and they are all correct.
* `c >= w(M_q)`: the capacity is slack, `lambda = 0`, and `chi^{M_q}` is
  optimal for both the relaxation and the integer problem.
