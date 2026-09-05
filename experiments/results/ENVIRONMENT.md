# Environment of the reported runs

Every timing in `REPORT.md` and in the CSV files of this directory was
produced on this machine, single-threaded, with the versions below.

## Machine

| | |
|---|---|
| CPU | Intel Core i5-14500T (20 threads; **all runs use one**) |
| RAM | 14 GB |
| OS | Ubuntu 26.04 LTS, kernel 7.0.0-30-generic |
| Python | 3.14.4 |

## Software

| package | version | role |
|---|---|---|
| numpy | 2.5.2 | |
| scipy | 1.18.1 | sparse graphs, residual BFS, `linprog` cross-check |
| ortools | 9.15.6755 | maximum-flow backend (int64, exact) — **the default** |
| python-igraph | 1.0.0 | maximum-flow backend (floating point) |
| highspy | 1.15.1 | LP baseline, open source |
| gurobipy | 13.0.3 | LP baseline, academic licence |
| CPLEX | 22.1 (Interactive Optimizer, licensed) | LP baseline, driven through `cplex-cli` |
| cplex (PyPI) | 22.2.0.1 | Community Edition; **size-limited, unusable on these instances** |
| pseudoflow | 2022.12.0 | parametric minimum cut |
| pytest / hypothesis | 9.1.1 / 6.167.1 | test suite |
| pandas / matplotlib | 3.0.5 / 3.11.1 | tables and figures |

## Protocol

* **One thread** everywhere: `threads=1` for every LP solver; the
  combinatorial methods are single-threaded by construction.
* **One process per method.** `experiments/compare_methods.py` runs each
  method in a fresh interpreter. This is needed for correctness — `highspy`
  and `ortools` cannot coexist in one process (see `docs/solvers.md`) — and it
  also keeps one solver's memory from perturbing another's timing.
* **Build and solve are timed separately.** For the LP backends the model is
  built once per instance and then re-solved at each capacity by changing only
  the capacity row's right-hand side, so simplex bases carry over. Reporting a
  rebuild-every-time number would measure the wrong thing.
* **Capacities**: 20 values evenly spaced strictly inside `(0, w(M_q))`, the
  range in which the capacity constraint binds. The first solve and the
  further ones are reported separately, because that is the axis on which the
  methods differ.
* **Agreement**: every method's value is compared against the first method's
  at every capacity; a relative difference above `1e-9` is flagged in the CSV
  and in the log.
* **Exact arithmetic** wherever the data allow it: instances whose profits and
  weights are integers, or scale to integers exactly, are solved with the
  int64 maximum-flow backend, so no tolerance decides a macroitem boundary.

## Reproducing

```bash
python -m pytest                                   # 466 tests, about a minute
python experiments/characterize.py <instances> --out characteristics.csv
python experiments/compare_methods.py <instances> --capacities 20 --out compare.csv
```

Timings will differ; the *values* must not. Any disagreement above `1e-9` is a
bug, in this library or in a solver, and should be reported.
