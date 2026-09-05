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
canonical ones, and on real instances they are larger by a wide margin.

| | |
|---|---|
| median gain, 23 benchmark instances | **×44.5** (range ×27.9 – ×160.6) |
| median gain, 8 open-pit instances | **×130.9** (range ×15.6 – ×295.6) |
| cost of the canonical dual certificate | 1 ms – 0.4 s |
| cost of the face-wide reduced costs | 0.03 s – 333 s (one minimum cut per item) |

The gain is three times larger on the open-pit instances than on the
benchmark, and the benchmark's own mining family sits between the two. That
is the mechanism stated plainly: the denser the precedence graph, the more a
forced item drags in with it, and the further the face-wide value can climb
above the item's own `w_i|λ_r − λ_h|`.

Since a reduced cost fixes a variable only when it exceeds the gap between the
relaxation and an incumbent, a factor of forty is the difference between a
bound that fixes nothing and one that fixes much of the instance. The costs
are not comparable either: the canonical values follow from the canonical
sequence for nothing, the face-wide ones need a minimum cut per item. They are
what one computes at a node worth the effort, not at every node.

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
`dim D* = 3`, matching the companion note.

Across the whole study the primal optimum is unique on **28 of the 29
instances** where the dimensions are computable: `k₀ = 1` and `dim X* = 0` on
all 23 benchmark instances and on five of the six open-pit ones.

The exception is `zuck_large`, with `k₀ = 2` and `dim X* = 1`: its split
macroitem does admit an intermediate optimal closure, so the optimal face is a
segment rather than a point. It is the one instance in the collection where
the primal optimum is genuinely not unique — and, as the theory requires, its
dual face is the largest measured, `dim D* = 1 030 216`.

Everywhere else the dual faces are large too — `dim D*` from 1 482 to 632 075
— which is exactly the room the face-wide reduced costs exploit. Intermediate
optimal closures trade primal degrees of freedom for dual ones, and on these
instances there are almost none to trade.

(`|H|` exceeds the computation's threshold on w23, zuck_medium,
mclaughlin_limit and kd, so the dimensions are not reported there; the first
two have split macroitems of 24 769 and 12 711 items, the last two have no
capacity declared by the instance.)
