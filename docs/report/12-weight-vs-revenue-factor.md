# Weight against revenue factor

← [Parametric implementations](11-parametric-implementations.md) · [Contents](README.md) · next: [Defects found](13-defects-found.md)

---

Mine planners usually generate nested pits by scaling the **revenue** of every
block at fixed cost: writing `p_i = r_i − k_i`, the values `f·r_i − k_i` are
parameterized by a revenue factor `f`. The parameterization that solves the LP
relaxation of a tonnage-constrained problem instead prices the **weight**,
using `p_i − λ w_i`.

Both families are nested, by the same lattice argument. They coincide when the
weights are proportional to the revenues. In general they are different
families, and the manuscript says so; this is the measurement.

## Result

On block models whose revenue and cost are known by construction, over a grid
of twenty revenue factors:

| instance | pits coinciding with a canonical closure | worst relative symmetric difference |
|---|---|---|
| grid 20×20×8, cone 5, seed 1 | 5 of 20 | 0.370 |
| grid 20×20×8, cone 5, seed 2 | 0 of 20 | 0.016 |
| grid 30×30×10, cone 9, seed 1 | 1 of 20 | 0.020 |

The symmetric difference is taken against the canonical closure of the
*nearest tonnage*, which is the fair comparison: a planner would pick the
nested pit whose tonnage they can handle. Reaching 0.37 means that more than a
third of the blocks differ between the pit a revenue factor produces and the
pit the tonnage-constrained relaxation would select at the same size.

The two families are therefore not interchangeable, and only the weight
parameterization solves the relaxation.

## A practical obstacle on real data

Repeating this on MineLib is blocked by the files, not by the method. The
ultimate-pit and constrained-pit formulations give a **single** value per
block, from which revenue and cost cannot be separated without inventing an
economic model. Only the production-scheduling (PCPSP) formulation lists a
value per destination — plant and waste — from which

```
cost    = −value_waste
revenue =  value_plant − value_waste
```

follows. `experiments/revenue_factor.py` reads that when it is present and
otherwise says why it cannot, rather than substituting an assumption.
