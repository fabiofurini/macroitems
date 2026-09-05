# Correctness

← [Instances](05-instances.md) · [Contents](README.md) · next: [Structure of real instances](07-structure-of-real-instances.md)

---

`pytest` runs **773 tests** in about seventy seconds. They are the reason to
trust every number in this report, so it is worth saying what they actually
check.

## Against an exact reference

Small instances are compared against a brute-force reference written in exact
rational arithmetic (`fractions.Fraction`) that enumerates **every** closure:

| what | how much |
|---|---|
| `u(λ)` and both lattice extremes at every breakpoint | ~3 800 (λ, instance) pairs over 216 instances |
| the canonical sequence, from its definition | 216 instances, both path algorithms |
| `z(c)` over a capacity grid | 136 instances × a half-integer grid |
| persistency against the *full* optimal face | 1 700+ capacities, 250+ of them with non-unique optima |
| the dual certificate | 80 instances |

The degenerate cases are in there deliberately: all profits negative, all
positive, zero profits, a single item, no arcs, stars, chains, and instances
built so that two disjoint continuations have equal ratio — the tie convention
is where this subject hides its bugs.

## Invariants on arbitrary instances

Property-based tests (`hypothesis`, ~900 generated instances) assert what must
hold for *any* instance: prefixes are closures, macroitems partition the item
set, ratios strictly decrease in exact arithmetic, `z(c)` is concave and
nondecreasing, the returned `x` is feasible and attains the reported value.

## Between implementations

- the two path algorithms (bisection, Dinkelbach) must agree — 100 random DAGs
  across five densities and two seeds, plus grids;
- the three maximum-flow backends must give identical closures and paths on
  integer data;
- the LP baselines must agree with the combinatorial methods, and with each
  other, to `1e-9`;
- integer rescaling must preserve ratios and scale values
  ([Defects found](13-defects-found.md), item 4).

## Against the other literatures

This is the part that tests the *paper's* claim, not just the code. Each
algorithm is reimplemented from its own source, using nothing from this
library.

| source | claim | coverage |
|---|---|---|
| Dantzig (1957) | with no arcs, the theory reduces to the greedy ratio rule | 67 arc-free instances against an independent implementation |
| Shaw and Cho (1998) | their aggregated subtrees are our macroitems, their bound is `z(c)` | 43 tree instances at ~1 500 capacities |
| Sidney (1975), Margot–Queyranne–Wang (2003) | the reduced Sidney decomposition is the canonical sequence, after reversing arcs and swapping the data | 57 instances, decomposition computed by brute force over all initial sets |

All three hold. They are not vacuous: dropping the arc reversal in the
scheduling translation breaks 188 of 190 instances, and using the finest
instead of the coarsest tie convention in the Shaw–Cho deletion breaks 8 of
the 43 tree cases.
