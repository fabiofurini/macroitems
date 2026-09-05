# How best to solve it

← [Structure of real instances](07-structure-of-real-instances.md) · [Contents](README.md) · next: [Accuracy](09-accuracy.md)

---

Full table, every instance and method:
[`experiments/results/tables/methods.md`](../../experiments/results/tables/methods.md).

Each method runs in its own process; build and solve are timed separately; the
LP solvers build their model once and re-solve by changing only the capacity
right-hand side, so their simplex bases carry over. Twenty capacities spanning
`(0, w(M_q))`, the range on which the capacity constraint binds.

## The whole value function

Sorted by arc count, which is what turns out to matter:

| instance | n | m | path | Newton | best solver | ratio |
|---|---|---|---|---|---|---|
| A_972_1661 | 972 | 1661 | 0.057 | 0.107 | 0.029 | 0.5x |
| B_981_1688 | 981 | 1688 | 0.051 | 0.114 | 0.033 | 0.6x |
| L_349_2101 | 349 | 2101 | 0.031 | 0.088 | 0.031 | 1.0x |
| C_1336_2382 | 1336 | 2382 | 0.088 | 0.181 | 0.056 | 0.6x |
| M_538_3033 | 538 | 3033 | 0.035 | 0.121 | 0.050 | 1.4x |
| D_1790_3130 | 1790 | 3130 | 0.074 | 0.157 | 0.066 | 0.9x |
| E_1790_3130 | 1790 | 3130 | 0.069 | 0.154 | 0.074 | 1.1x |
| F_3091_5715 | 3091 | 5715 | 0.184 | 0.231 | 0.113 | 0.6x |
| G_3091_5715 | 3091 | 5715 | 0.079 | 0.333 | 0.193 | 2.4x |
| H_3091_5715 | 3091 | 5715 | 0.259 | 0.274 | 0.122 | 0.5x |
| I_3091_5715 | 3091 | 5715 | 0.092 | 0.237 | 0.159 | 1.7x |
| N_1217_7616 | 1217 | 7616 | 0.058 | 0.225 | 0.124 | 2.1x |
| O_1711_11661 | 1711 | 11661 | 0.044 | 0.338 | 0.155 | 3.6x |
| J_9235_17082 | 9235 | 17082 | 0.337 | 0.647 | 1.141 | 3.4x |
| K_9235_17082 | 9235 | 17082 | 0.405 | 0.753 | 0.909 | 2.2x |
| Q_3428_19555 | 3428 | 19555 | 0.177 | 0.562 | 0.705 | 4.0x |
| P_3243_22306 | 3243 | 22306 | 0.071 | 0.750 | 0.575 | 8.0x |
| R_4281_24452 | 4281 | 24452 | 0.224 | 0.660 | 1.126 | 5.0x |
| S_5624_36504 | 5624 | 36504 | 0.232 | 0.932 | 1.653 | 7.1x |
| T_6271_42080 | 6271 | 42080 | 0.249 | 1.305 | 3.012 | 12.1x |
| U_6494_48626 | 6494 | 48626 | 0.161 | 1.251 | 1.794 | 11.2x |
| V_10001_63944 | 10001 | 63944 | 0.471 | 2.214 | 5.598 | 11.9x |
| W_11757_83218 | 11757 | 83218 | 0.420 | 3.409 | 12.190 | 29.0x |

Grouped:

| group | instances | median ratio | range |
|---|---|---|---|
| sparse, m < 10⁴ | 12 | **1.00x** | 0.47–2.44 |
| dense, m ≥ 2·10⁴ | 7 | **11.2x** | 5.0–29.0 |

Item count alone predicts nothing: J and K have 9 235 items and the methods
essentially tie, while P with 3 243 items but 22 306 arcs gives an 8x margin.
What a simplex basis has to carry, and what a maximum-flow computation handles
comfortably, is the precedence structure.

On the open-pit instances the effect is stronger. On zuck_small (9 400 blocks,
145 640 precedences) the path returns the entire value function in 0.58 s
against 26.3 s for the fastest solver over twenty capacities; on zuck_medium
(29 277 blocks, 1 271 207 precedences) in 3.45 s against 1 355 s for HiGHS dual
simplex — a factor of 390 — and at a 420 s per-method budget HiGHS does not
finish that instance at all.

## One capacity

The Newton search on the weight price costs a handful of maximum closures on
shrinking residual graphs. Against the fastest of the three simplex codes it is
a median **1.5x** faster, with a wide range (0.38–26.9x): on small sparse
instances a warm-started dual simplex is at least as good, and there is no
reason to prefer the combinatorial route there.

## Interior point

Barrier and IPM cannot start from a previous basis, so their marginal cost per
capacity equals their full cost. Over a grid of capacities they were 10–30x
worse than dual simplex in every run, and they are the first methods to hit a
time limit on the large open-pit instances. This is worth stating because
barrier is a common default on large sparse LPs, and here it is the wrong
default.

## The break-even

For a solver with first-capacity cost `f` and marginal cost `e`, the path is
cheaper once the number of capacities exceeds `1 + (path_first − f)/e`. On the
dense instances that number is **1** — the path is already cheaper for a single
capacity. On the sparse ones it is typically 2 to 5. It is never large.
