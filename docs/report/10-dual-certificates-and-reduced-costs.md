# Dual certificates and reduced costs

← [Accuracy](09-accuracy.md) · [Contents](README.md) · next: [Parametric implementations](11-parametric-implementations.md)

---

## The canonical certificate

At a nondegenerate capacity every optimal dual has the same capacity price
`λ_h`, and a canonical optimum is built region by region: a nonnegative flow
with prescribed divergence on each of `F`, `H`, `Z` — three maximum flows on
the subgraphs they induce. It costs about as much as the primal solution.

On every instance tried it was feasible and complementary with the canonical
primal to machine precision. That is a meaningful check rather than a
formality: the feasibility conditions of the three flow systems are exactly the
`λ_h`-optimality of `M_{h-1}` and `M_h` together with the maximum-ratio
property of `I_h`, so a failure here would indicate an error in the *primal*,
not a numerical difficulty in the dual.

The certificate is computed exactly. On integer data `λ_h` is the rational
`p(H)/w(H)`, so multiplying the region capacities by `w(H)` makes them
integers; the flow scales linearly and is divided back.

## Reduced costs over the whole dual face

The dual optimal face is not a point, so an item has a *range* of reduced
costs. The canonical dual solution gives the closed form `w_i|λ_r − λ_h|` —
the classical knapsack expression — for nothing beyond the canonical sequence.
Optimizing over the whole face instead needs one minimum cut per item:

- for a null item, the cheapest way to force it in is a maximum closure on the
  subgraph induced by `Z` with that item forced in;
- for a full item, the cheapest way to force it out is the same computation on
  `F` **with the arcs reversed**, since a co-closed set of a graph is a closed
  set of its reverse.

The two cases minimize different objectives, and getting that wrong yields
bounds that are invalid but entirely plausible — see
[Defects found](13-defects-found.md), item 1.

**How much it buys.** The face-wide values are never smaller than the
canonical ones. On the paper's running example they are exactly twice as
large on every item outside the split macroitem, and the gain grows with the
density of the precedence graph, which is what one would expect: the denser the
graph, the more a forced item drags in with it. Since a reduced cost fixes a
variable when it exceeds the gap between the relaxation and an incumbent, a
larger reduced cost fixes strictly more.

**Validation.** For each item the relaxation was re-solved with that item
forced to its opposite bound; the reduced cost must not exceed the resulting
drop in value, and must not fall below the canonical value. Over twelve
instances, no violation of either.

## Face dimensions

`dim X* = k₀ − 1`, with `k₀` the number of inseparability classes of the split
macroitem, and `dim D*` through the arcs of `A(H)` entering a tight set. The
computation is `|H|` minimum cuts on the subgraph induced by `H`, which is
affordable exactly when the split macroitem is small — the common case.

On the running example: `dim X* = 0` (the primal optimum is unique) and
`dim D* = 3`, matching the companion note. Intermediate optimal closures trade
primal degrees of freedom for dual ones, so an instance with a unique primal
optimum has the widest choice of certificates — which is where the face-wide
reduced costs are worth computing.
