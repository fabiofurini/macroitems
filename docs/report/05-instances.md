# Instances

← [Setup and protocol](04-setup.md) · [Contents](README.md) · next: [Correctness](06-correctness.md)

---


This repository redistributes **no third-party instance data**. The readers in
`macroitems.formats` take the files as published by their authors, so you
download them once and point the library at them. What follows is where to get
them and what we had to learn about the files — the kind of detail that costs
an afternoon if nobody writes it down.

## MineLib

Espinoza, Goycoolea, Moreno and Newman, *MineLib: a library of open pit mining
problems*, Annals of Operations Research 206 (2013) 93–114.
Download: <https://mansci-web.uai.cl/minelib/>

An instance is a set of files sharing a stem: `.blocks` (block model),
`.prec` (precedences), `.upit` (ultimate-pit objective), `.cpit` (the
constrained pit-limit data). Read one with

```python
from macroitems.formats import read_minelib_upit
inst = read_minelib_upit("/path/to/minelib/newman1")   # a stem, a directory or a zip
```

**The profit** is the UPIT objective coefficient, positive for ore and
negative for waste.

**The weight is the tonnage, and finding it is the awkward part.** The format
specification says the columns of `.blocks` after `id x y z` are
"optional user-specified fields", and instances differ: they range from 8 to
19 columns, and no file says which one is the tonnage. `weight="auto"` (the
default) works it out and *verifies* it:

1. every block has to be moved, so the tonnage is positive on **all** blocks;
   an operational resource of the `.cpit` file whose coefficients cover every
   block is therefore the mining resource, and it is used directly (this
   settles 8 of the 10 instances);
2. a resource covering only part of the blocks is a *processing* resource,
   consumed by ore blocks alone. Then the tonnage is the `.blocks` attribute
   that is positive everywhere **and** agrees with those partial coefficients
   wherever they exist — the agreement is what identifies the column, since a
   processing resource charges an ore block exactly its tonnage. This settles
   `kd` (attribute 0, agreeing on all 5 931 ore blocks) and
   `mclaughlin_limit` (attribute 1, on all 31 931).

Each instance records how it was decided in `inst.meta["weight_resolution"]`.
Pass `weight="blocks", tonnage_column=k` or `weight="unit"` to override — the
latter being the volume parameterization of Lerchs and Grossmann.

**Capacities.** The `.cpit` resource limits give a capacity taken from the
literature rather than invented; it is in `meta["capacity_cpit_period"]`
(per period) and `meta["capacity_cpit_total"]`.

**Exact arithmetic.** Six of the ten instances scale to integers exactly (see
`Instance.scaled_to_integers`); the other four carry more than nine decimals
and are worked in floating point, which the library reports rather than
hiding.

## The PCKP benchmark

Used by Park and Park (1997), by Boland, Bley, Fricke, Froyland and Sotirov,
*Clique-based facets for the precedence constrained knapsack problem*,
Mathematical Programming 132 (2012) 69–90, and by Espinoza, Goycoolea, Moreno
and Newman (2015).

Two formats per instance, `.lp` (CPLEX LP, the original 2010 distribution) and
`.lp.dat` (a tabular form carrying the capacity in its header):

```python
from macroitems.formats import read_pckp_dat, read_pckp_lp
inst = read_pckp_dat(".../L_349_2101.lp.dat")   # inst.meta["capacity"] is set
```

Three things worth knowing:

* **The two formats number the items differently** on 13 of the 23 instances.
  They are the same instances — the multisets of profits and weights agree
  exactly and so do the arc counts — but a permutation apart, so item indices
  are not comparable across formats. The `.dat` files are the ones to prefer:
  they are internally consistent and carry the capacity.
* **The family labels are inverted in some copies.** The original
  distribution's directory layout and its `results.txt` put A–K in *telecom*
  and L–W in *mining*; the `runList_*` files shipped with some copies of the
  accompanying code say the opposite. The data settle it: the mining instances
  are the ones with negative-profit waste items. `macroitems.formats` follows
  the original.
* **Four mining instances carry one decimal** and become integral when
  multiplied by 10, so exact arithmetic applies to all 23.

The distribution also contains 16 *scheduling* instances. These are
**multi-period** models — variables `X_i_t` and ten capacity rows — so they
are not single-capacity PCKP instances and this library does not read them.

`results.txt` gives best known primal and dual values; for all 23
single-capacity instances they coincide, so the integer optima are known and
can be used to measure the true integrality gap against the bound of the
paper.

## Synthetic instances

`macroitems.instance` generates them with a seed, so they are reproducible
without being stored:

```python
from macroitems import layered_grid, random_dag, running_example
layered_grid(36, 36, 11, cone=9, seed=2)   # a mining-like block model
random_dag(20000, avg_out_degree=2.0, seed=1)
running_example()                          # the paper's 8-item instance
```

`layered_grid` builds a block model whose grades come from a few ellipsoidal
ore bodies, with each block requiring the 5 or 9 blocks above it; values are
rounded to integers so that ties occur, as they do in real data. Random DAGs
are a useful contrast: real deposits decompose into many macroitems, whereas
random DAGs tend to collapse into one giant one.

## This package's own text format

```
n m
p_0 w_0
...
p_{n-1} w_{n-1}
i j            (m lines; j is a prerequisite of i)
```

`Instance.write` / `Instance.read`, with metadata alongside in a `.json` file.
