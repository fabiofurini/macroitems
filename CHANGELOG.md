# Changelog

## Unreleased

### Added

* **`macroitems.classes`**: `split_structure` gives, at a nondegenerate
  capacity:
  * the inseparability classes of the split macroitem and their order;
  * the weights of the down- and up-sets and the critical class;
  * the attainability of 0/1 patterns, and whether branching on a set of
    variables lowers the LP bound;
  * the number of maximal chains of tight subsets.
* **`macroitems.faces.minimal_tight_sets`**: the smallest tight subset
  containing each item of the split macroitem, now public.
* **`macroitems.forests`**: linear-time rerooting recursion for closure
  problems on precedence forests, with:
  * the strongest reduced costs;
  * the classes and the critical class;
  * the hook-length count, whose hook is the up-set on in-forests and the
    down-set on out-forests.
* **`tests/test_optimal_faces.py`**: the above checked against exact
  enumeration, an LP over the dual face, and the branch LPs.
* **`experiments/literature_fixing.py`** and **`literature_fixing_table.py`**:
  variable fixing and optimal faces on the 23 PCKP benchmark instances, with a
  certified integer optimum, the heuristic incumbent, timings, and one
  memory-capped process per instance.
* **Classes** are now grouped by identical minimal tight sets, which avoids a
  dense $|H|\times|H|$ matrix on large split macroitems.
* **`experiments/forest_fixing.py`** and **`forest_fixing_table.py`**:
  reproducible variable-fixing experiment on random forests, with per-instance
  seeds, a certified integer optimum and the environment recorded.

## 0.2.0 — 5 September 2026

First public version. Everything below is relative to the internal 0.1.0
pilot.

### Added

* **Instance readers** (`macroitems.formats`) for the published collections:
  the PCKP benchmark in both its formats, and MineLib. The MineLib tonnage is
  resolved automatically and *verified* against the constrained-pit resource
  coefficients, so no per-instance configuration is needed.
* **`macroitems.dual.best_reduced_costs`** — reduced costs over the whole dual
  optimal face, one minimum cut per item, with `fixable_items` for use in an
  enumerative algorithm.
* **`macroitems.lawler`** — Lawler's binary search for the first macroitem,
  kept for the method comparison.
* **`macroitems.lp`** — LP-solver baselines behind one interface, building the
  model once and re-solving at many capacities: HiGHS, scipy, Gurobi, CPLEX,
  and a `cplex-cli` backend that drives a licensed CPLEX Interactive
  Optimizer (the PyPI `cplex` package is the size-limited Community Edition).
* **`macroitems.cli`** — the `macroitems` command: `info`, `path`, `solve`,
  `lp`, `convert`, `gen`.
* **Interchangeable maximum-flow backends** (`ortools`, `igraph`, `scipy`),
  selectable with `backend=` on `canonical_path` and `solve_capacity`.
* **`Instance.scaled_to_integers`** — exact decimal rescaling, which puts
  decimal data into the exact-arithmetic regime.
* **`macroitems.pseudoflow_path`** — the parametric reduction, implemented,
  tested and *disabled*: the public `pseudoflow` package silently returns an
  incomplete parametric family. See the module docstring for the
  three-item counterexample.
* Experiments (`experiments/`), documentation (`docs/`), an experimental
  report (`docs/report/`, fourteen pages) and a test suite of 773 tests.

### Fixed

Four defects that produced plausible but wrong answers:

* reduced costs had the sign inverted on the full region;
* `solution_from_path` returned an infeasible point at `c = 0`, because
  `closure_mask(n, -1)` wrapped around a negative slice;
* the `scipy` maximum-flow backend computes in int32 and silently truncated
  larger capacities, returning a different canonical path from the other
  backends on integer data with wide coefficients — such capacities are now
  refused;
* reporting code divided macroitem ratios by the integer scale; a ratio is
  invariant under rescaling.

The package also did not install: a PEP 639 license expression alongside the
deprecated MIT classifier made setuptools refuse the project.

### Changed

* `canonical_path` and `solve_capacity` scale decimal data to integers
  internally. Closures, macroitems and ratios are invariant under that
  scaling, so the answer is unchanged, but the computation becomes exact and
  no longer needs a floating-point maximum-flow backend.
* `solve_capacity` reports `h=None` instead of `-1`: the Newton search does
  not enumerate the macroitems, so the index of the split one is genuinely
  unknown there. `solution_from_path` reports it.
