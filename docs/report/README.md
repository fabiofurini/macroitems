# The LP relaxation of the precedence-constrained knapsack problem
## Computational report

Valerio Dose · Fabio Furini · Marco Locatelli

*Browsable edition of [`report/computational_report.pdf`](../../report/computational_report.pdf),
the experimental study accompanying the manuscript. Every
table is generated from the raw CSV files in
[`experiments/results/`](../../experiments/results/) by
`experiments/make_tables.py`; no number is hand-typed.*

The manuscript states the structure — the canonical macroitem sequence, the
primal and dual optimal faces, the dictionary between five literatures. This
report is the complement: the experimental record, and specifically the
material a paper section cannot carry —

- the full method comparison on every instance, not a selection, with the cost
  of the *first* capacity separated from the cost of each *further* one, which
  is the axis on which the methods actually differ
  ([How best to solve it](08-how-best-to-solve-it.md));
- an accuracy result that speed comparisons usually hide: on one open-pit
  instance every LP solver we tried returns a value wrong in the eighth
  significant digit ([Accuracy](09-accuracy.md));
- what the published instances actually look like once decomposed — how much
  of the answer the relaxation settles by itself, and on which instances it
  settles almost nothing ([Structure of real instances](07-structure-of-real-instances.md));
- why the public parametric minimum-cut implementation could not be included
  ([Parametric implementations](11-parametric-implementations.md));
- what was actually implemented, and what each of the 773 tests checks
  ([What the library computes](02-what-the-library-computes.md),
  [Correctness](06-correctness.md)) — including reimplementations of three
  other communities' algorithms, used as independent references;
- and the defects found on the way, all of which produced plausible wrong
  answers rather than errors ([Defects found](13-defects-found.md)).

## Contents

1. [Summary](01-summary.md) — the answer in one page
2. [What the library computes](02-what-the-library-computes.md)
3. [Definitions and conventions](03-definitions-and-conventions.md)
4. [Setup and protocol](04-setup.md)
5. [Instances](05-instances.md)
6. [Correctness](06-correctness.md)
7. [Structure of real instances](07-structure-of-real-instances.md)
8. [How best to solve it](08-how-best-to-solve-it.md)
9. [Accuracy: when the solvers are wrong](09-accuracy.md)
10. [Dual certificates and reduced costs](10-dual-certificates-and-reduced-costs.md)
11. [Parametric implementations](11-parametric-implementations.md)
12. [Weight against revenue factor](12-weight-vs-revenue-factor.md)
13. [Defects found during development](13-defects-found.md)
14. [Reproducing this report](14-reproducing-this-report.md)

## How to read the numbers

Unless a page says otherwise:

- every time is wall-clock seconds, single-threaded, on the machine of
  [Setup and protocol](04-setup.md);
- **build** time (constructing a solver's model) is reported separately from
  **solve** time, and the LP baselines build their model once and then change
  only the capacity right-hand side between capacities, so their simplex bases
  carry over — a baseline that rebuilt every time would be measuring the wrong
  thing;
- *first* is the cost of answering one capacity from nothing; *per extra* is
  the marginal cost of each additional capacity. The canonical path pays
  everything in *first* and essentially nothing per extra, which is the whole
  point;
- every method's values are compared against the canonical path at every
  capacity, and any relative difference above `1e-9` is reported, not hidden.
