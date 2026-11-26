# Macroitems and optimal faces in the precedence-constrained knapsack LP relaxation

**V. Dose, F. Furini, M. Locatelli** — code and data of the computational
experiments of this paper.

This page is the entry point for readers of the paper. The rest of the
repository is a general library for the PCKP relaxation and also hosts the
computational study of a companion manuscript; you do not need it to follow
or reproduce this paper.

**What belongs to this paper**

| | where |
|---|---|
| strongest reduced costs, classes, faces (the algorithms of the paper) | [`macroitems/dual.py`](../../macroitems/dual.py), [`macroitems/faces.py`](../../macroitems/faces.py), [`macroitems/classes.py`](../../macroitems/classes.py), [`macroitems/forests.py`](../../macroitems/forests.py) |
| fast exact computation in C++ (Section 4.1 and the *Implementation* paragraph) | [`cpp/rc_engine.cpp`](../../cpp/rc_engine.cpp), called through [`macroitems/fastrc.py`](../../macroitems/fastrc.py) |
| tests of every statement of the paper on random instances | [`tests/test_optimal_faces.py`](../../tests/test_optimal_faces.py), [`tests/test_running_example.py`](../../tests/test_running_example.py), [`tests/test_fastrc.py`](../../tests/test_fastrc.py) |
| experiment scripts | [`experiments/`](../../experiments/): `forest_fixing.py`, `literature_fixing.py`, `rc_timing.py`, `bb_fixing.py` and their `*_table.py` |
| raw results (CSV, one row per run, software versions in the header) | [`experiments/results/`](../../experiments/results/): `forest_fixing.csv`, `literature_fixing.csv`, `rc_timing.csv`, `bb_fixing.csv` |
| LaTeX tables and quoted numbers, generated from the CSVs | [`experiments/results/tables/`](../../experiments/results/tables/) |
| benchmark instances (not ours, unmodified) | `instances.zip` of the [latest release](https://github.com/fabiofurini/macroitems/releases/latest) |

**Reproduce everything** (from the repository root):

```bash
pip install -e ".[experiments]" gurobipy        # Gurobi for the integer optima
gh release download -R fabiofurini/macroitems --pattern instances.zip && unzip instances.zip -d instances_raw
pytest tests/test_optimal_faces.py tests/test_running_example.py tests/test_fastrc.py
```

then the commands of each section below.

Every table and every number quoted in the computational section of the paper
is produced by a script of `experiments/` from a CSV in `experiments/results/`;
nothing is typed by hand.

| in the paper | script | raw results | time |
|---|---|---|---|
| random precedence forests (table and text) | `forest_fixing.py`, `forest_fixing_table.py` | `forest_fixing.csv` | ~2 min |
| PCKP benchmark: faces, fixing, gap (table and text) | `literature_fixing.py`, `literature_fixing_table.py` | `literature_fixing.csv` | ~15 min |
| PCKP benchmark: time of the strongest reduced costs, per item vs incremental (table column and text) | `rc_timing.py` | `rc_timing.csv` | ~4 min |
| PCKP benchmark: Gurobi with and without fixing (text) | `bb_fixing.py --variants base,best`, `bb_table.py` | `bb_fixing.csv` | ~30 min |

The LaTeX rows and macros go to `experiments/results/tables/`. All runs use one
thread; the instances run one at a time, each in a memory-capped process.

## Random precedence forests

```bash
python experiments/forest_fixing.py --out experiments/results/forest_fixing.csv
python experiments/forest_fixing_table.py experiments/results/forest_fixing.csv \
       --rows experiments/results/tables/forest_fixing.tex \
       --macros experiments/results/tables/forest_fixing_macros.tex
```

The first command takes about two minutes and needs `gurobipy` for the
certified integer optimum. It produces variable fixing on random precedence
forests, 9 classes × 30 instances. Every instance has its own seed. Every
attempt, including the discarded ones and the reason, is a row of the CSV, and
the CSV header records the versions of Python, NumPy, SciPy and Gurobi. Two runs
give identical CSVs apart from the timing columns.

## The PCKP benchmark (literature instances)

```bash
gh release download -R fabiofurini/macroitems --pattern instances.zip && unzip instances.zip -d instances_raw
python experiments/literature_fixing.py --out experiments/results/literature_fixing.csv
python experiments/literature_fixing_table.py experiments/results/literature_fixing.csv \
       --rows experiments/results/tables/literature_fixing.tex \
       --macros experiments/results/tables/literature_fixing_macros.tex
```

The experiment runs about 15 minutes. For each of the 23 benchmark instances,
at its own capacity, it records:
* the regions, $k_0$ and the dimensions of both optimal faces;
* the integer optimum certified by Gurobi;
* the share of items fixed by the solver's dual optimum, by the canonical
  reduced costs, and by the strongest reduced costs;
* the same shares against the macroitem heuristic;
* the time needed to compute the strongest reduced costs.

Each instance runs in its own memory-capped process, so a large instance
cannot exhaust the machine.

## Computing the strongest reduced costs fast

```bash
python experiments/rc_timing.py --out experiments/results/rc_timing.csv
```

The strongest reduced cost of an item is a maximum-closure problem, and solving
one per item from scratch is the baseline (`cold`). The C++ engine
(`cpp/rc_engine.cpp`, built with `g++ -O3` on first use by `macroitems.fastrc`)
exploits that, at a nondegenerate capacity, the maximum closure of the null
region is empty:

* one maximum flow is computed in Picard's network, and it saturates every
  source arc; the reduced cost of item `i` is the largest flow that can be
  *added* from `i` to the sink in its residual network (`warm`);
* the exact distances to the sink are computed once by a breadth-first search,
  and every item pushes its additional flow by shortest augmenting paths from
  these labels, with the gap rule and the upper bound `-b(cl(i))`; the
  capacities and labels changed are undone afterwards (`sap`);
* the items are visited along a tree of prerequisites: the flow of the parent
  is kept while its descendants are processed, so that each item pays only for
  the difference (`tree`, the default).

The same holds on the full region with the arcs reversed. The inseparability
classes come from the same idea: on the split macroitem the maximum closure has
value 0, and the smallest tight set containing `i` is the set reachable from
`i` in the residual network of one maximum flow (strongly connected components
and a bitset propagation). Everything is exact, in 64-bit integer arithmetic
after scaling the data. On the 23 benchmark instances `tree` takes 2.3 s in
total, against 125 s for `cold`, and all engines return identical values
(`tests/test_fastrc.py` also checks them against the reference on hundreds of
random instances).

## Gurobi with and without fixing

```bash
python experiments/bb_fixing.py --variants base,best --worker-cap-gb 8 --mem-limit-gb 4 \
       --out experiments/results/bb_fixing.csv
python experiments/bb_table.py experiments/results/rc_timing.csv experiments/results/bb_fixing.csv \
       --rows experiments/results/tables/bb_fixing.tex --macros experiments/results/tables/bb_fixing_macros.tex
```

Each instance is solved by Gurobi (one thread, zero gap, seeds 1, 2, 3) on the
full model (`base`) and after fixing the items whose strongest reduced cost
exceeds `z(c) - LB` (`best`), where `LB` is the macroitem heuristic, passed to
both runs as a cutoff. The script checks that both reach the same optimum. The
time of the reduced costs is charged to `best`. Outcome: up to 2.1× faster on
the telecommunication instances where most of `Z` is fixed, unchanged on the
mining instances; the number of nodes does not decrease systematically. W needs
the larger memory caps above.

