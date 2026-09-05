# Summary

← [Contents](README.md) · next: [What the library computes](02-what-the-library-computes.md)

---

## The question

The LP relaxation of the precedence-constrained knapsack problem

```
max  p'x   s.t.   w'x <= c,   x_i <= x_j for every arc (i, j),   0 <= x <= 1
```

can be handed to a general-purpose LP solver, or solved through the parametric
maximum-closure structure this library implements. **Which is better, and
when?**

## The answer

It turns on one thing visible before any computation, and one that is not.

| question | answer | evidence |
|---|---|---|
| one capacity | Newton search on the weight price | median 1.5x the fastest LP solver (range 0.4–27x) |
| the whole value function, sparse graphs (m < 10⁴) | either | median **1.00x** — a tie |
| the whole value function, dense graphs (m ≥ 2·10⁴) | the canonical path | median **11.2x** on the benchmark, **21.9x** on MineLib, up to **94.9x** |
| an LP solver is unavoidable | dual simplex, never barrier | interior point cannot warm start: every capacity costs a full solve |
| the value must be right | the combinatorial method | it is exact on integer data — see below |

**1. One capacity, or the value function?** The canonical path returns *every*
capacity at once: after it is computed, a further capacity costs a binary
search over the breakpoints. Anything that needs more than a couple of
capacities — a Lagrangian scheme, a branch-and-bound, a multi-period model —
is in the regime where nothing else is close.

**2. Density, not size.** Grouping the benchmark by arc count rather than item
count separates the outcomes cleanly: on sparse precedence graphs a
warm-started dual simplex matches the combinatorial method; on dense ones the
path wins by an order of magnitude, and the margin grows with size. Item count
alone predicts nothing — the sparse group contains instances with 9 235 items
on which the methods tie.

On four of the ten open-pit instances no LP solver finished a single capacity
within 150 seconds, while the path returned every capacity in 11 to 47.

**3. Accuracy is part of the answer.** On MineLib `kd`, at one capacity out of
twenty, all five LP solver configurations return a value wrong by
1.5 · 10⁻⁸ relative, while the combinatorial method is exact. See
[Accuracy](09-accuracy.md).

## Two findings for the record

**No usable public parametric implementation.** The `pseudoflow` package
silently returns an incomplete parametric family — it fails on three items
with no arcs. See [Parametric implementations](11-parametric-implementations.md).

**The standard benchmark lives in the hard regime.** On all 23 benchmark
instances the published capacity is 0.5–25% of `w(M_q)`, so the split
macroitem is the *first* one and the integrality-gap bound is vacuous; the
true gaps are large (median 26.6%). See
[Structure of real instances](07-structure-of-real-instances.md).
