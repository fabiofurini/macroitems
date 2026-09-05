# Solvers and backends

This package **contains no third-party solver code**. Every solver and every
maximum-flow library is an *optional* dependency that you install yourself,
under its own licence; `macroitems` only links to whatever it finds. That is
why the package can be MIT-licensed while being usable with GPL and
commercial software.

## Maximum-flow backends

The combinatorial methods (`canonical_path`, `solve_capacity`,
`best_reduced_costs`, `face_dimensions`) all reduce to maximum closure, which
is one minimum cut. Three interchangeable backends, selected with
`backend=` or by `macroitems._maxflow.default_backend()`:

| backend | package | arithmetic | notes |
|---|---|---|---|
| `ortools` | `ortools` (Apache-2.0) | int64, **exact** | the default when installed, and the one used for the published results |
| `igraph` | `python-igraph` (GPL-2) | floating point | the fallback for data that cannot be scaled to integers |
| `scipy` | `scipy` (BSD, a hard dependency) | int32 | always available; **rejects** capacities above `2**31 - 1` rather than truncating them |

With `scipy` alone the library is complete and exact for data whose scaled
parametric values stay inside int32, which covers the published benchmark and
most instances one meets. Wide integer coefficients do not fit, and the
backend refuses them with a clear error instead of returning a wrong cut;
installing `ortools` (or `igraph`) removes the restriction. The test suite
skips the wide-coefficient checks when no 64-bit backend is present, and says
so.

They must agree on integer data, and the test suite checks that they do. The
`scipy` restriction is not cosmetic: `scipy.sparse.csgraph.maximum_flow`
silently truncates larger capacities and returns a wrong flow, which on wide
integer data produced a *different* canonical path before the check was added.

## LP solvers

Used as baselines, behind one interface (`macroitems.lp.LPBackend`) that
builds the model once and re-solves at many capacities by changing only the
capacity row's right-hand side.

| name | how to get it | notes |
|---|---|---|
| `highs` | `pip install highspy` | **the reproducible open-source baseline**; all published timings can be reproduced with it alone |
| `scipy` | already a dependency | the same HiGHS reached through `linprog`, rebuilding every time; a cross-check, not a timing baseline |
| `gurobi` | `pip install gurobipy` + a licence | commercial |
| `cplex` | `pip install cplex` | the **Community Edition**: refuses models above 1000 rows or columns, so it is unusable on real instances |
| `cplex-cli` | a licensed CPLEX Studio installation | drives the Interactive Optimizer over a pipe; see below |

Methods available per backend are in `Backend.methods`; `dual-simplex` is the
one to use when re-solving at many capacities, and `barrier`/`ipm` the one to
avoid, since interior-point methods cannot warm start.

### Using a licensed CPLEX

CPLEX Studio ships Python bindings only for the interpreter versions current
at its release, so on a newer Python the licensed solver is often unreachable
in-process while the PyPI package gives you only the size-limited Community
Edition. The `cplex-cli` backend works around this by driving the licensed
**Interactive Optimizer** executable: it reads the model once and then issues
`change rhs` and `optimize` per capacity, so the simplex basis carries over
exactly as in the in-process backends.

Point it at the executable with

```
export MACROITEMS_CPLEX=/path/to/cplex/bin/x86-64_linux/cplex
```

or let it find `cplex` on `PATH` or under `~/ILOG/CPLEX_Studio*` or
`$CPLEX_STUDIO_DIR*`. Pass `return_primal=False` when you only need the value
and the multiplier: retrieving `x` costs an extra solution file per solve.

## A trap: `highspy` and `ortools` in the same process

Both wheels export the C++ symbols of their own copy of HiGHS, with
incompatible ABIs. **Whichever imports first wins**, and the other fails with
`ImportError: undefined symbol`. Since `ortools` is the default maximum-flow
backend, running a combinatorial method first silently removes `highs` from
`available_lp_backends()`.

Ways out, in order of preference:

1. run each method in its own process — `experiments/compare_methods.py` does
   this, and it is good practice for timing anyway;
2. use the `igraph` maximum-flow backend when you need `highspy` in the same
   process;
3. import `highspy` before anything touches `ortools`.

`solve_lp` warns rather than degrading silently when it has to fall back from
`highs` to `scipy`, because that fallback changes what a timing measures.

## Parametric minimum cut

`pseudoflow` (the Hochbaum group's package) solves the parametric minimum cut
directly. Read `macroitems/pseudoflow_path.py` for the reduction and for the
authors' own caveat about what its timings do and do not mean. There is no
public implementation of the Gallo–Grigoriadis–Tarjan (1989) algorithm that
can be trusted for benchmarking, which is worth stating whenever parametric
running times are compared.
