"""The LP-solver baselines of :mod:`macroitems.lp`.

Three things are checked.  (i) The running example of the paper: at c = 4 the
optimum is z = 7 with capacity multiplier lambda = 3/2 and x equal to 1 on
items {3, 6}, 1/2 on {1, 2, 5} and 0 on {4, 7, 8} in the paper's 1-based
numbering, which pins down the sign convention of every backend.  (ii) The
backends agree with each other and, above all, with the combinatorial solver
``solve_capacity`` of the package -- that is the real cross-validation, since
the two compute the same number by entirely different means.  (iii) Solving a
reused model at many capacities gives what rebuilding it from scratch gives.

Backends that are not installed are skipped.  So is a backend whose licence
refuses the size of an instance (the PyPI ``cplex`` package without a licensed
installation is the Community Edition, capped at 1000 rows and columns); such
a solve returns ``status="error"`` and is reported here as a skip rather than
as a failure.

The set of backends -- and hence the number of tests collected here -- is read
at import time from :func:`available_lp_backends`.  ``highspy`` and ``ortools``
cannot be loaded into the same process (see the warning in
:mod:`macroitems.lp`), so ``"highs"`` is on the list only when this module is
imported before anything touches the ortools maximum-flow backend.  Running
``pytest tests/test_lp_backends.py`` on its own therefore covers all four
backends, while a full-suite run may cover three.
"""
import numpy as np
import pytest

from macroitems import (layered_grid, random_dag, running_example, solve_capacity,
                        Instance)
from macroitems.lp import (LPResult, available_lp_backends, get_lp_backend, solve_lp)

pytestmark = pytest.mark.solver

BACKENDS = available_lp_backends()
requires_backends = pytest.mark.skipif(not BACKENDS, reason="no LP backend installed")

# Small enough for every licence, including the CPLEX Community Edition
# (1000 rows: these have at most ~460 precedence rows).
SMALL = [
    ("dag60", lambda: random_dag(60, avg_out_degree=2.0, seed=1)),
    ("dag120", lambda: random_dag(120, avg_out_degree=3.0, seed=7)),
    ("grid443", lambda: layered_grid(4, 4, 3, cone=5, seed=2)),
    ("grid553", lambda: layered_grid(5, 5, 3, cone=9, seed=3)),
]

_LICENCE_HINTS = ("community edition", "size limit", "model too large", "licen")


def _licence_limited(res: LPResult) -> bool:
    return res.status == "error" and any(h in res.message.lower() for h in _LICENCE_HINTS)


def _solve_all(backend: str, inst: Instance, capacities, method: str = "default") -> list[LPResult]:
    """One built model, one result per capacity; skips a licence-limited backend."""
    with get_lp_backend(backend)(inst, method=method) as solver:
        out = [solver.solve(c) for c in capacities]
    for res in out:
        if _licence_limited(res):
            pytest.skip(f"{backend} licence refuses n={inst.n}, rows={1 + inst.m}: {res.message}")
        assert res.status == "optimal", (backend, res.status, res.message)
    return out


def _capacities(inst: Instance, fractions=(0.05, 0.2, 0.4, 0.6, 0.85)) -> list[float]:
    total = float(inst.w.sum())
    return [f * total for f in fractions]


def _feasible(inst: Instance, x: np.ndarray, c: float, tol: float = 1e-7) -> bool:
    return bool(np.all(x >= -tol) and np.all(x <= 1 + tol)
                and float(inst.w @ x) <= c + tol * max(1.0, c)
                and np.all(x[inst.arcs[:, 0]] <= x[inst.arcs[:, 1]] + tol))


# ------------------------------------------------------------------ registry
def test_registry_is_consistent():
    for name in BACKENDS:
        cls = get_lp_backend(name)
        assert cls.name == name and "default" in cls.methods
    with pytest.raises(ValueError):
        get_lp_backend("no-such-solver")


@requires_backends
def test_unknown_method_is_rejected():
    with pytest.raises(ValueError):
        get_lp_backend(BACKENDS[0])(running_example(), method="no-such-method")


# ----------------------------------------------------- the running example
@pytest.mark.parametrize("backend", BACKENDS)
def test_running_example(backend):
    """z = 7, lambda = 3/2 and the canonical fractional x at c = 4."""
    inst = running_example()
    res, = _solve_all(backend, inst, [4.0])
    assert res.backend == backend and res.status == "optimal"
    assert res.value == pytest.approx(7.0, abs=1e-9)
    assert res.lam == pytest.approx(1.5, abs=1e-7)
    expected = np.array([0.5, 0.5, 1.0, 0.0, 0.5, 1.0, 0.0, 0.0])
    assert res.x == pytest.approx(expected, abs=1e-7)
    assert res.seconds_build >= 0.0 and res.seconds_solve >= 0.0


@pytest.mark.parametrize("backend", BACKENDS)
def test_every_method_solves_the_running_example(backend):
    """The simplex variants and the interior-point method agree, duals included."""
    inst = running_example()
    for method in get_lp_backend(backend).methods:
        res, = _solve_all(backend, inst, [4.0], method=method)
        assert res.method == method
        assert res.value == pytest.approx(7.0, abs=1e-6), method
        assert res.lam == pytest.approx(1.5, abs=1e-6), method


@pytest.mark.parametrize("backend", BACKENDS)
def test_lambda_is_nonnegative_and_is_the_slope(backend):
    """lambda = dz/dc: a finite difference around a nondegenerate capacity."""
    inst = running_example()
    h = 1e-4
    lo, mid, hi = _solve_all(backend, inst, [4.0 - h, 4.0, 4.0 + h])
    assert mid.lam >= 0.0
    assert (hi.value - lo.value) / (2 * h) == pytest.approx(mid.lam, abs=1e-6)


# ------------------------------------------------- cross-validation of values
@pytest.mark.parametrize("name,make", SMALL, ids=[n for n, _ in SMALL])
@pytest.mark.parametrize("backend", BACKENDS)
def test_matches_the_combinatorial_solver(backend, name, make):
    """Each backend against ``solve_capacity`` -- LP simplex against parametric max flow."""
    inst = make()
    caps = _capacities(inst)
    results = _solve_all(backend, inst, caps)
    for c, res in zip(caps, results):
        ref = solve_capacity(inst, c).value
        assert res.value == pytest.approx(ref, rel=1e-9, abs=1e-9), (name, c)
        assert _feasible(inst, res.x, c), (name, c)
        assert res.lam >= 0.0


@requires_backends
@pytest.mark.parametrize("name,make", SMALL, ids=[n for n, _ in SMALL])
def test_backends_agree(name, make):
    """All available backends return the same optimal value at every capacity."""
    inst = make()
    caps = _capacities(inst)
    values = {b: [r.value for r in _solve_all(b, inst, caps)] for b in BACKENDS}
    reference = values[BACKENDS[0]]
    for backend, got in values.items():
        assert got == pytest.approx(reference, rel=1e-9, abs=1e-9), backend


# --------------------------------------------------------------- model reuse
@pytest.mark.parametrize("backend", BACKENDS)
def test_model_reuse_matches_fresh_solves(backend):
    """Changing only the capacity right-hand side must not change any answer.

    The capacities are visited out of order, so a stale basis or a right-hand
    side left over from the previous call would show up.
    """
    inst = random_dag(150, avg_out_degree=2.5, seed=11)
    caps = _capacities(inst, (0.7, 0.1, 0.45, 0.95, 0.25, 0.7))
    reused = _solve_all(backend, inst, caps)
    for c, res in zip(caps, reused):
        fresh = solve_lp(inst, c, backend=backend)
        if _licence_limited(fresh):
            pytest.skip(f"{backend} licence refuses this instance: {fresh.message}")
        assert fresh.status == "optimal"
        assert res.value == pytest.approx(fresh.value, rel=1e-9, abs=1e-9)
        assert res.lam == pytest.approx(fresh.lam, rel=1e-7, abs=1e-7)
    # the same capacity twice gives the same answer
    assert reused[0].value == pytest.approx(reused[-1].value, rel=1e-12, abs=1e-12)


@pytest.mark.parametrize("backend", BACKENDS)
def test_build_is_paid_once(backend):
    """``seconds_build`` is a property of the model, not of the solve."""
    inst = random_dag(200, avg_out_degree=2.0, seed=13)
    results = _solve_all(backend, inst, _capacities(inst))
    assert len({r.seconds_build for r in results}) == 1


# ------------------------------------------------------------- degenerate data
@requires_backends
def test_duplicated_arcs_and_isolated_items():
    """A repeated arc and an item in no arc: redundant rows and an empty column."""
    base = random_dag(40, avg_out_degree=2.0, seed=5)
    arcs = np.vstack([base.arcs, base.arcs[:7], base.arcs[:7]])
    p = np.concatenate([base.p, [10.0, -4.0, 0.0]])          # three isolated items
    w = np.concatenate([base.w, [3.0, 5.0, 2.0]])
    inst = Instance(p, w, arcs, name="duplicated_arcs")
    inst.validate()
    for c in _capacities(inst):
        ref = solve_capacity(inst, c).value
        for backend in BACKENDS:
            res, = _solve_all(backend, inst, [c])
            assert res.value == pytest.approx(ref, rel=1e-9, abs=1e-9), (backend, c)


@requires_backends
def test_no_precedences_is_the_fractional_knapsack():
    """With m = 0 the LP is the fractional knapsack, solved by the greedy ratio rule."""
    rng = np.random.default_rng(3)
    p = rng.integers(-5, 20, 30).astype(float)
    w = rng.integers(1, 9, 30).astype(float)
    inst = Instance(p, w, np.zeros((0, 2), np.int64), name="no_arcs")
    inst.validate()
    c = 0.4 * float(w.sum())
    order = np.argsort(-np.where(w > 0, p / w, -np.inf))
    greedy, left = 0.0, c
    for i in order:
        if p[i] <= 0:
            break
        take = min(1.0, left / w[i])
        greedy += take * p[i]
        left -= take * w[i]
    for backend in BACKENDS:
        res, = _solve_all(backend, inst, [c])
        assert res.value == pytest.approx(greedy, rel=1e-9, abs=1e-9), backend


@requires_backends
def test_capacity_above_the_maximum_closure_weight():
    """Above the weight of the maximum-profit closure the capacity row is slack, so lambda = 0."""
    inst = random_dag(80, avg_out_degree=2.0, seed=17)
    c = float(inst.w.sum()) + 1.0
    ref = solve_capacity(inst, c)
    for backend in BACKENDS:
        res, = _solve_all(backend, inst, [c])
        assert res.value == pytest.approx(ref.value, rel=1e-9, abs=1e-9), backend
        assert res.lam == pytest.approx(0.0, abs=1e-7), backend


# ------------------------------------------------------------------ hygiene
@requires_backends
def test_no_solver_log_files(tmp_path, monkeypatch):
    """No cplex.log, no gurobi.log: the experiments must not litter the working directory."""
    monkeypatch.chdir(tmp_path)
    inst = running_example()
    for backend in BACKENDS:
        with get_lp_backend(backend)(inst) as solver:
            solver.solve(4.0)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("backend", BACKENDS)
def test_solve_lp_wrapper(backend):
    """``solve_lp`` reports the same optimum, and its timings add up."""
    inst = running_example()
    res = solve_lp(inst, 4.0, backend=backend)
    assert res.status == "optimal" and res.backend == backend
    assert res.value == pytest.approx(7.0, abs=1e-9)
    assert res.lam == pytest.approx(1.5, abs=1e-7)
    assert res.seconds == pytest.approx(res.seconds_build + res.seconds_solve)


@pytest.mark.parametrize("backend", BACKENDS)
def test_time_limit_never_reports_a_wrong_optimum(backend):
    """Under an unreachable time limit: either the true optimum, or no answer at all."""
    inst = random_dag(4000, avg_out_degree=3.0, seed=29)
    c = 0.4 * float(inst.w.sum())
    with get_lp_backend(backend)(inst, time_limit=1e-6) as solver:
        res = solver.solve(c)
    if res.status == "optimal":
        assert res.value == pytest.approx(solve_capacity(inst, c).value, rel=1e-8, abs=1e-6)
    else:
        assert res.status in ("time_limit", "iteration_limit", "error"), res.message
        assert np.isnan(res.value) and np.isnan(res.lam)


@requires_backends
def test_close_is_idempotent():
    solver = get_lp_backend(BACKENDS[0])(running_example())
    solver.solve(4.0)
    solver.close()
    solver.close()


# --------------------------------------------------------------------- slow
@pytest.mark.slow
@pytest.mark.parametrize("backend", BACKENDS)
def test_large_instance_against_the_combinatorial_solver(backend):
    """A block model of a few thousand items, over a grid of capacities."""
    inst = layered_grid(16, 16, 8, cone=9, seed=4)
    caps = _capacities(inst, (0.1, 0.3, 0.5, 0.7, 0.9))
    for c, res in zip(caps, _solve_all(backend, inst, caps)):
        assert res.value == pytest.approx(solve_capacity(inst, c).value, rel=1e-8, abs=1e-6)
        assert _feasible(inst, res.x, c, tol=1e-6)


@pytest.mark.slow
@requires_backends
def test_all_backends_agree_on_a_larger_dag():
    inst = random_dag(3000, avg_out_degree=3.0, seed=23)
    caps = _capacities(inst, (0.15, 0.5, 0.8))
    reference = [solve_capacity(inst, c).value for c in caps]
    for backend in BACKENDS:
        got = [r.value for r in _solve_all(backend, inst, caps)]
        assert got == pytest.approx(reference, rel=1e-8, abs=1e-6), backend
