# Accuracy: when the solvers are wrong

← [How best to solve it](08-how-best-to-solve-it.md) · [Contents](README.md) · next: [Dual certificates and reduced costs](10-dual-certificates-and-reduced-costs.md)

---

Every method's value is compared against the canonical path at every capacity.
The result is nearly, but not entirely, unanimous.

## The benchmark: complete agreement

Across 161 method-runs on the 23 benchmark instances, twenty capacities each,
the largest relative difference is **4.8 · 10⁻¹¹**. Nothing above `1e-9`.

## MineLib `kd`: the solvers are wrong

At one capacity out of twenty, all five LP solver configurations return

```
LP solvers    220 239 941.533204
this library  220 239 938.251054
```

a relative difference of **1.5 · 10⁻⁸**, where at the other nineteen
capacities they agree to `1e-15`.

The exact value settles it. `kd` scales to integers exactly, so `z(c)` can be
computed in rational arithmetic from the integer path:

```
1238890176810816816310940159 / 5625184000000  =  220 239 938.251054
```

The library is right; the solvers are off by 3.28.

**It is not the rescaling.** Gurobi returns the same wrong value when given the
original decimal data, so the conditioning of the instance, not the integer
scaling, is the cause: `kd` has profits of order 4 · 10⁵ against weights of
order 10⁴, and at that particular capacity the LP is ill-conditioned enough for
a double-precision solver's default tolerances to stop early.

## Why this matters

An error of 1.5 · 10⁻⁸ is harmless if the LP value is a report. It is not
harmless if the value is a bound: inside a branch-and-bound, a bound that is
too *large* by three units prunes nothing it should, and a bound too small
prunes an optimal subtree. Nor is it harmless when the quantity of interest is
a difference of two LP values, as an integrality gap or a reduced cost is.

On integer data — all 23 benchmark instances, and six of the ten MineLib ones
after exact decimal rescaling — the parametric machinery is **exact**: the
breakpoints are rationals recovered as `p(I_r)/w(I_r)`, the parametric values
`b·p − a·w` are integers, and an integer maximum-flow backend gives exact cuts.
No tolerance decides which macroitem an item belongs to.

That is the argument for the combinatorial route that a speed table does not
show.
