# Defects found during development

← [Weight against revenue factor](12-weight-vs-revenue-factor.md) · [Contents](README.md) · next: [Reproducing this report](14-reproducing-this-report.md)

---

Recorded because the same failure modes are available to anyone reimplementing
these classical algorithms, and because **every one of them produced a
plausible answer rather than an error**. None would have been caught by a test
that only checked the code runs.

## 1. Reduced costs: sign inverted on the full region

The null and full regions minimize different objectives — `v` on `Z`, and `−v`
on the *reversed* graph of `F` — and both are minimizations turned into the
maximization a maximum-closure routine performs. Getting the second one wrong
produced reduced costs that were invalid but entirely believable: positive,
ordered sensibly, of the right magnitude.

*Caught by* comparing against the exact loss, obtained by re-solving the
relaxation with each item forced to its opposite bound. Nothing cheaper would
have found it.

## 2. An infeasible point at `c = 0`

`split_index(0)` is `0`, so `closure_mask(n, −1)` wrapped around a negative
Python slice and returned all but the last macroitem — a point of positive
weight for a capacity of zero. The Newton solver was correct at `c = 0`, so the
library's two entry points disagreed.

*Caught by* property-based testing at the boundary of the capacity range.

## 3. The scipy maximum-flow backend truncated silently

`scipy.sparse.csgraph.maximum_flow` computes in **int32** and truncates larger
capacities without warning — an arc of capacity `2³¹` receives flow 0. On
integer data with wide coefficients, where the scaled parametric values
`b·p − a·w` and the big-M on precedence arcs exceed int32, it returned a
*different canonical path* from the other two backends.

*Caught by* requiring the three backends to agree on instances with wide
coefficients. Such capacities are now refused with an error: a loud failure in
place of a silent wrong answer.

## 4. Ratios divided by the integer scale

Rescaling an instance to integers multiplies profits, weights, capacities and
the optimal value by the scale — but **not the ratios**, since a ratio `p/w`
carries the factor in both numerator and denominator. Reporting code divided
them anyway, which would have published breakpoints wrong by orders of
magnitude while every other number in the same table looked right.

*Caught by* drawing a figure and noticing the colour bar was scaled by 10⁻⁶.
`tests/test_scaling.py` now pins the invariance.

## 5. The package did not install

A PEP 639 license expression alongside the deprecated `License :: OSI
Approved :: MIT License` classifier makes setuptools refuse the project
outright, so the declared console script was never reachable.

*Caught by* building a wheel and installing it into an empty environment.

## 6. Core dependencies that were not sufficient

The package declares numpy and scipy as its only hard dependencies. It
imported `igraph` — an optional extra — at module level, and, more subtly, the
dual certificate and the face dimensions used floating-point node values that
only a floating-point backend accepts.

The fix was not to add a dependency but to make both computations **exact**: on
integer data `λ_h` is the rational `p(H)/w(H)`, so multiplying the node values
by `w(H)` makes them integers without changing which sets are optimal or tight.

*Caught by* a CI job that installs the package with no extras and runs the
command line — on its first execution.

---

The pattern is worth naming: in this subject, a wrong implementation returns
something that looks like an answer. Nested chains of closures, decreasing
ratios, plausible reduced costs, and near-identical objective values are all
things a broken implementation produces. The only checks that caught anything
were the ones with an **independent** reference — exact rational arithmetic,
another backend, another algorithm, or the LP itself.
