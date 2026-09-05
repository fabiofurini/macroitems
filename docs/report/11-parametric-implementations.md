# Parametric implementations

← [Dual certificates and reduced costs](10-dual-certificates-and-reduced-costs.md) · [Contents](README.md) · next: [Weight against revenue factor](12-weight-vs-revenue-factor.md)

---

The natural third family of methods, alongside repeated closures and LP
solvers, is a **parametric minimum cut**: one computation returning all
breakpoints. We intended to include it, and could not.

## The package's own caveat

The `pseudoflow` package of the Hochbaum group states, in its own
documentation:

> This implementation does not use *free runs* nor does it use warm starts
> with informatiom from previous runs (see pg.15). This implementation should
> therefore **not be used** for comparison with the fully parametric HPF
> algorithm.

(the typo is theirs). That alone means its timings cannot stand for the
algorithm's performance. But the reason we report no row for it is stronger.

## It silently returns an incomplete family

The smallest failing instance has **three items, no precedence arcs**, profits
`(3, 2, 1)` and unit weights. The canonical sequence is `{0}`, `{1}`, `{2}`
with ratios `3 > 2 > 1`, so there are three breakpoints. The package reports
**two intervals**, with source sets `{0}` and `{0,1,2}`.

That is not a coarser answer, it is a wrong parametric minimum cut: at
`λ = 3.5` all node values are negative, so the optimal closure is empty, while
the set the package returns for that interval has value `−0.5`.

Verified independently of our reduction: the same failure appears with the
package's own intended formulation, and it is insensitive to the parameter
range, to the constant offset, to global scaling and to node ordering.
Failures start at `n = 13` and become common by `n = 200` (4 of 10 random DAGs
exact at 200, 2 of 10 at 300).

## Why it is dangerous rather than merely wrong

What the package returns is always a nested, complete chain of genuine
closures with strictly decreasing increment ratios — a **coarsening** of the
canonical sequence. Every sanity check one would naturally apply passes.
Detecting the defect costs one maximum closure per macroitem, which is
recomputing the answer by another method.

## What we do about it

`macroitems/pseudoflow_path.py` implements the reduction to a
source–sink-monotone parametric cut, tests it, and then **raises
`NotImplementedError`** with this explanation. The reduction is kept and
exercised so that the method can be re-enabled without rewriting it if the
package is fixed; `allow_incorrect=True` returns the package's answer for
anyone who wants to reproduce the failure.

One correction to the textbook reduction, found while implementing it: the
offset `K ≥ max_i(−p_i)` usually quoted is not sufficient. The parameter range
must reach below the smallest ratio, which is negative whenever some `p_i/w_i`
is, and then the sink capacity `λ w_i + K` goes negative — at which point the
C library prints a message and **kills the process**. The correct offset is
`K = max(0, −min p, −λ_lo · max w)`.

## And no alternative

There is no public implementation of the Gallo–Grigoriadis–Tarjan (1989)
parametric maximum-flow algorithm that we would trust for benchmarking. The
practical situation is therefore that published running times for parametric
minimum cut on this problem are hard to reproduce independently — which is
part of why the comparison in this report is against LP solvers.
