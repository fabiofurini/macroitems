# Instances

This directory is deliberately almost empty.

**Synthetic instances are not stored, because they are reproducible.** The
generators are deterministic given a seed, so an instance is fully described
by the command that makes it:

```bash
macroitems gen grid --nx 36 --ny 36 --nz 11 --cone 9 --seed 2 --out grid_L.txt
macroitems gen dag  --n 20000 --degree 2.0 --seed 1 --out dag_M.txt
```

or, in Python,

```python
from macroitems import layered_grid, random_dag, running_example
layered_grid(36, 36, 11, cone=9, seed=2)
random_dag(20000, avg_out_degree=2.0, seed=1)
running_example()          # the 8-item instance of the paper
```

Storing the files as well would only create a second source of truth that can
drift from the generators.

**Third-party instances are not redistributed.** MineLib and the PCKP
benchmark belong to their authors; `macroitems.formats` reads their published
files directly. See [the Instances page](../docs/report/05-instances.md) for where to
obtain them and what to watch out for in the files.
