# Reproducing this report

← [Defects found](13-defects-found.md) · [Contents](README.md)

---

## Install

```bash
git clone https://github.com/fabiofurini/macroitems && cd macroitems
pip install -e ".[experiments]"     # ortools, igraph, highspy, pandas, matplotlib
pytest -q                            # 773 tests, about seventy seconds
```

The commercial solvers are optional and are never bundled; install them
yourself under your own licence and the comparison picks them up. See
[`docs/solvers.md`](../solvers.md), which also documents the one environment
trap: `highspy` and `ortools` cannot be loaded into the same process.

## Get the instances

Neither collection is redistributed here; [Instances](05-instances.md) says
where to obtain them and what to watch out for in the files. Put them anywhere
and point the scripts at them.

## Run

```bash
# structure of the canonical sequence (Section 7)
python experiments/characterize.py <instances> --out experiments/results/characteristics.csv

# the method comparison (Sections 8 and 9)
python experiments/compare_methods.py <instances> --capacities 20 \
       --out experiments/results/compare.csv

# weight against revenue factor (Section 12)
python experiments/revenue_factor.py grid:20x20x8:5:1 --factors 20 \
       --out experiments/results/revenue_factor.csv

# tables and figures
python experiments/make_tables.py
python experiments/make_figures.py <instances> --out experiments/results/figures
```

Each method runs in its own subprocess, so a campaign survives a solver
crashing, and the CSV is written after every instance rather than at the end.

## What must and must not reproduce

**Timings will differ** — they depend on the machine, and the protocol in
[Setup](04-setup.md) says what was fixed (one thread, model reuse, build and
solve separated).

**Values must not.** Every method's value is compared against the canonical
path at every capacity, and a relative difference above `1e-9` is flagged in
the CSV and in the log. If you see one that is not the `kd` case documented in
[Accuracy](09-accuracy.md), it is a bug — in this library or in a solver — and
we would like to hear about it.

## Raw data

| file | contents |
|---|---|
| [`characteristics_pckp.csv`](../../experiments/results/characteristics_pckp.csv) | canonical sequence and split macroitem, 23 benchmark instances |
| [`characteristics_minelib.csv`](../../experiments/results/characteristics_minelib.csv) | the same, 10 MineLib instances |
| [`compare_pckp_benchmark.csv`](../../experiments/results/compare_pckp_benchmark.csv) | method comparison, 161 runs |
| [`compare_minelib.csv`](../../experiments/results/compare_minelib.csv) | method comparison, MineLib |
| [`revenue_factor.csv`](../../experiments/results/revenue_factor.csv) | the two pit families |
| [`tables/`](../../experiments/results/tables/) | generated tables, Markdown and LaTeX from the same numbers |
| [`figures/`](../../experiments/results/figures/) | generated figures |
| [`ENVIRONMENT.md`](../../experiments/results/ENVIRONMENT.md) | machine, versions, protocol |
