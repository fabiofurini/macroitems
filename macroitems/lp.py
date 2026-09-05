"""LP-solver baselines for the precedence-constrained knapsack relaxation.

The relaxation solved here is the one of the paper,

    max  p'x   s.t.   w'x <= c,   x_i - x_j <= 0  for every arc (i, j),
                      0 <= x <= 1,

with n variables and 1 + m rows: the *capacity row* (index 0) and one
precedence row per arc, in the order in which the arcs appear in
``inst.arcs``.  A duplicated arc simply produces a duplicated (redundant) row
and an item in no arc produces an empty column; both are left alone, so that
the model a backend builds is exactly the model the instance describes.

These backends exist to answer the experimental question of the paper --
*what is the best way to solve this LP in practice?* -- so they are built for
fair measurement rather than convenience:

* **build and solve are timed separately**.  Reading an instance into a solver
  is a Python-side cost that has nothing to do with the LP algorithm, and on
  the sparse mining instances it dominates a single solve.
* **the model is built once and re-solved at many capacities**, changing only
  the right-hand side of the capacity row.  This is the regime the parametric
  method of the paper competes in, so a baseline that rebuilds the model for
  every capacity would be measuring the wrong thing.  Simplex bases carry over
  between calls (each solver's own hot start); :class:`ScipyBackend` is the
  deliberate exception, see below.

Backends (all optional, all linked -- no third-party solver code ships with
this package; see ``docs/solvers.md``):

``highs``   :class:`HighsBackend`, ``highspy`` used directly.  The reproducible
            open-source baseline and the reference for the timings of the paper.
``scipy``   :class:`ScipyBackend`, ``scipy.optimize.linprog(method="highs")``.
            The same HiGHS code reached through a rebuild-every-time interface;
            kept as an independent cross-check of :class:`HighsBackend`, not as
            a timing baseline.
``gurobi``  :class:`GurobiBackend`, ``gurobipy``.
``cplex``   :class:`CplexBackend`, the ``cplex`` Python API.

Sign convention.  We maximize, and lambda is the multiplier of the capacity
row in the dual of the paper (lambda = dz/dc >= 0, the weight price).  On the
running example at c = 4 the optimum is z = 7 with lambda = 3/2, and the three
model-level solvers report the multiplier of a ``<=`` row of a maximization
problem with that same sign: HiGHS ``row_dual``, Gurobi ``Pi`` and CPLEX
``get_dual_values`` all return +3/2 (verified in ``tests/test_lp_backends.py``).
``scipy.optimize.linprog`` minimizes ``-p'x`` instead and reports the marginal
of the minimization, ``d(-z)/dc = -lambda``, so its sign is flipped here.

All backends run single-threaded by default and with logging off, and none of
them leaves a log file behind (CPLEX writes ``cplex.log`` unless its four
streams are redirected, which :class:`CplexBackend` does before the model is
built; Gurobi writes ``gurobi.log`` unless ``LogFile`` is cleared on the
environment).

Warning -- ``highspy`` and ``ortools`` can clash.  Both wheels export the C++
symbols of their own copy of HiGHS into the global symbol namespace, with
incompatible ABIs, so in one process only the one imported *first* loads: the
second raises ``ImportError: undefined symbol``.  Since ``ortools`` is the
default maximum-flow backend of :mod:`macroitems._maxflow`, running the
combinatorial method before building a :class:`HighsBackend` silently leaves
``"highs"`` out of :func:`available_lp_backends` (and vice versa).  Three ways
out, in order of preference: import ``highspy`` before anything touches
``ortools``; run the max flows on the ``igraph`` backend; or time HiGHS in a
process of its own.  :class:`ScipyBackend`, :class:`GurobiBackend` and
:class:`CplexBackend` are unaffected -- scipy keeps its HiGHS private -- and
:func:`get_lp_backend` reports the loader's message when this happens.
"""
from __future__ import annotations

import dataclasses
import os
import re
import time
import warnings
from typing import Optional

import numpy as np
from scipy.sparse import csr_matrix

from .instance import Instance

__all__ = ["LPResult", "LPBackend", "HighsBackend", "ScipyBackend", "GurobiBackend", "CplexCliBackend",
           "CplexBackend", "available_lp_backends", "get_lp_backend", "solve_lp"]

# Multipliers below -LAM_TOL are reported as they come (they signal a genuine
# sign or modelling error); the tiny negative noise of an interior-point run is
# clipped, since lambda >= 0 holds for every capacity by monotonicity of z.
LAM_TOL = 1e-9


@dataclasses.dataclass
class LPResult:
    x: np.ndarray             # primal solution, shape (n,); nan when not solved
    value: float              # optimal value of max p'x
    lam: float                # multiplier of the capacity row (>= 0); nan if unavailable
    seconds_build: float      # time to construct the model, once per instance
    seconds_solve: float      # time of this solve only
    status: str               # "optimal", "time_limit", "infeasible", "error", ...
    backend: str              # "highs", "scipy", "gurobi", "cplex"
    method: str               # "default", "dual-simplex", ...
    message: str = ""

    @property
    def seconds(self) -> float:
        """Build plus solve, i.e. the cost of answering one capacity from scratch."""
        return self.seconds_build + self.seconds_solve


class LPBackend:
    """Build the model of ``inst`` once, then solve it at many capacities.

    Subclasses implement ``_build()`` (called by ``__init__``, which times it
    into ``seconds_build``), ``_solve(c)`` returning
    ``(x, value, lam, status, message)``, and optionally ``close()``.

    ``method`` names an algorithm in ``methods``; ``threads`` and
    ``time_limit`` (seconds, ``None`` for no limit) are passed to the solver.
    Backends are also context managers, which is the safest way to be sure the
    solver's memory and its environment are released.
    """

    name: str = ""
    methods: tuple[str, ...] = ("default",)

    def __init__(self, inst: Instance, method: str = "default", threads: int = 1,
                 time_limit: Optional[float] = None):
        if method not in self.methods:
            raise ValueError(f"backend {self.name!r}: unknown method {method!r}; "
                             f"use one of {', '.join(self.methods)}")
        self.inst = inst
        self.method = method
        self.threads = int(threads)
        self.time_limit = time_limit
        self.n_solves = 0
        t0 = time.perf_counter()
        self._build()
        self.seconds_build = time.perf_counter() - t0

    def _build(self) -> None:
        raise NotImplementedError

    def _solve(self, c: float) -> tuple[np.ndarray, float, float, str, str]:
        raise NotImplementedError

    def solve(self, c: float) -> LPResult:
        """Solve at capacity ``c``, changing only the right-hand side of the capacity row."""
        t0 = time.perf_counter()
        x, value, lam, status, message = self._solve(float(c))
        dt = time.perf_counter() - t0
        self.n_solves += 1
        return LPResult(x, value, _clip_lam(lam), self.seconds_build, dt, status,
                        self.name, self.method, message)

    def close(self) -> None:
        """Release the solver's resources.  Idempotent."""

    def __enter__(self) -> "LPBackend":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def _clip_lam(lam: float) -> float:
    lam = float(lam)
    return 0.0 if -LAM_TOL < lam < 0.0 else lam


def _failed(n: int, status: str, message: str = "") -> tuple[np.ndarray, float, float, str, str]:
    return np.full(n, np.nan), np.nan, np.nan, status, message


def _rows_csr(inst: Instance) -> csr_matrix:
    """The (1 + m) x n constraint matrix: capacity row first, then x_i - x_j <= 0."""
    n, m = inst.n, inst.m
    data = np.concatenate([inst.w, np.ones(m), -np.ones(m)])
    col = np.concatenate([np.arange(n), inst.arcs[:, 0], inst.arcs[:, 1]])
    row = np.concatenate([np.zeros(n, np.int64), np.arange(1, m + 1), np.arange(1, m + 1)])
    return csr_matrix((data, (row, col)), shape=(1 + m, n))


# ------------------------------------------------------------------- HiGHS
_HIGHS_OPTIONS = {
    "default": {},
    "simplex": {"solver": "simplex"},
    # HiGHS simplex_strategy: 1 = dual, 4 = primal (kSimplexStrategyDual/Primal).
    "dual-simplex": {"solver": "simplex", "simplex_strategy": 1},
    "primal-simplex": {"solver": "simplex", "simplex_strategy": 4},
    "ipm": {"solver": "ipm"},
    "barrier": {"solver": "ipm"},
}


class HighsBackend(LPBackend):
    """HiGHS through ``highspy``, with the model built once and re-solved.

    The row-wise matrix is passed in one call, so building is a few numpy
    concatenations plus one copy into HiGHS; ``solve`` then only calls
    ``changeRowBounds`` on the capacity row, which leaves the current basis in
    place, so a re-solve starts from the previous optimum.
    """

    name = "highs"
    methods = tuple(_HIGHS_OPTIONS)

    def _build(self) -> None:
        import highspy

        inst = self.inst
        n, m = inst.n, inst.m
        h = highspy.Highs()
        h.setOptionValue("output_flag", False)
        h.setOptionValue("log_to_console", False)
        h.setOptionValue("threads", self.threads)
        if self.time_limit is not None:
            h.setOptionValue("time_limit", float(self.time_limit))
        for key, val in _HIGHS_OPTIONS[self.method].items():
            h.setOptionValue(key, val)

        inf = highspy.kHighsInf
        lp = highspy.HighsLp()
        lp.num_col_ = n
        lp.num_row_ = 1 + m
        lp.sense_ = highspy.ObjSense.kMaximize
        lp.col_cost_ = np.ascontiguousarray(inst.p, dtype=float)
        lp.col_lower_ = np.zeros(n)
        lp.col_upper_ = np.ones(n)
        lp.row_lower_ = np.full(1 + m, -inf)
        lp.row_upper_ = np.zeros(1 + m)          # capacity RHS is set by solve()
        # row-wise: row 0 has the n weights, then two entries per precedence row
        lp.a_matrix_.format_ = highspy.MatrixFormat.kRowwise
        lp.a_matrix_.num_col_ = n
        lp.a_matrix_.num_row_ = 1 + m
        lp.a_matrix_.start_ = np.concatenate(
            [[0], np.arange(n, n + 2 * m + 1, 2)]).astype(np.int32)
        lp.a_matrix_.index_ = np.concatenate(
            [np.arange(n), inst.arcs.ravel()]).astype(np.int32)
        lp.a_matrix_.value_ = np.concatenate([inst.w, np.tile([1.0, -1.0], m)])
        if h.passModel(lp) == highspy.HighsStatus.kError:
            raise RuntimeError("HiGHS rejected the model")
        self._highs = h
        self._inf = inf

    def _solve(self, c):
        import highspy

        h = self._highs
        h.changeRowBounds(0, -self._inf, c)
        # kWarning is what run() returns on a time limit, so only kError aborts
        if h.run() == highspy.HighsStatus.kError:
            return _failed(self.inst.n, "error", "HiGHS run() returned kError")
        model_status = h.getModelStatus()
        text = h.modelStatusToString(model_status)
        if model_status != highspy.HighsModelStatus.kOptimal:
            return _failed(self.inst.n, _HIGHS_STATUS.get(text.lower(), "error"), text)
        sol = h.getSolution()
        x = np.asarray(sol.col_value, dtype=float)
        # HiGHS reports the multiplier of a <= row of a max problem as +lambda.
        lam = float(sol.row_dual[0]) if sol.dual_valid else np.nan
        return x, float(h.getObjectiveValue()), lam, "optimal", ""

    def close(self) -> None:
        h = getattr(self, "_highs", None)
        if h is not None:
            h.clear()
            self._highs = None


# HiGHS model statuses, keyed by modelStatusToString so that the enum is not
# needed at module level (highspy is an optional dependency).
_HIGHS_STATUS = {"infeasible": "infeasible", "unbounded": "unbounded",
                 "primal infeasible or unbounded": "infeasible_or_unbounded",
                 "time limit reached": "time_limit",
                 "iteration limit reached": "iteration_limit"}


# ------------------------------------------------------------------- scipy
class ScipyBackend(LPBackend):
    """``scipy.optimize.linprog(method="highs")``, rebuilt at every capacity.

    scipy offers no handle on the model, so every call hands the whole problem
    to HiGHS again: ``seconds_solve`` therefore includes scipy's own model
    construction and presolve, and ``seconds_build`` only covers the sparse
    matrix assembled here once.  That is exactly why this backend is a
    correctness cross-check of :class:`HighsBackend` and not a timing baseline
    for it.  ``method`` is ignored (scipy exposes no algorithm choice beyond
    the HiGHS default).
    """

    name = "scipy"
    methods = ("default",)

    def _build(self) -> None:
        self._A = _rows_csr(self.inst)
        self._b = np.zeros(1 + self.inst.m)

    def _solve(self, c):
        from scipy.optimize import linprog

        self._b[0] = c
        options = {} if self.time_limit is None else {"time_limit": float(self.time_limit)}
        # linprog minimizes, so it is handed -p and its marginals are negated
        res = linprog(-self.inst.p, A_ub=self._A, b_ub=self._b, bounds=(0, 1),
                      method="highs", options=options)
        if res.status != 0:
            return _failed(self.inst.n, _SCIPY_STATUS.get(res.status, "error"), res.message)
        lam = -float(res.ineqlin.marginals[0]) if res.ineqlin is not None else np.nan
        return np.asarray(res.x, dtype=float), float(-res.fun), lam, "optimal", ""


_SCIPY_STATUS = {0: "optimal", 1: "time_limit", 2: "infeasible", 3: "unbounded",
                 4: "error"}


# ------------------------------------------------------------------ Gurobi
_GUROBI_METHOD = {"default": -1, "primal-simplex": 0, "dual-simplex": 1, "barrier": 2}


class GurobiBackend(LPBackend):
    """Gurobi through ``gurobipy``, model built once and re-solved.

    ``solve`` assigns the capacity row's ``RHS`` and calls ``optimize`` again,
    which resumes from the basis of the previous call.  The environment is
    created empty and started with logging off and ``LogFile`` cleared, so no
    ``gurobi.log`` appears and no licence banner is printed.
    """

    name = "gurobi"
    methods = tuple(_GUROBI_METHOD)

    def _build(self) -> None:
        import gurobipy as gp

        env = gp.Env(empty=True)
        env.setParam("OutputFlag", 0)
        env.setParam("LogToConsole", 0)
        env.setParam("LogFile", "")
        env.start()
        self._env = env
        inst = self.inst
        model = gp.Model("pckp-lp", env=env)
        model.Params.Threads = self.threads
        model.Params.Method = _GUROBI_METHOD[self.method]
        if self.time_limit is not None:
            model.Params.TimeLimit = float(self.time_limit)
        x = model.addMVar(inst.n, lb=0.0, ub=1.0)
        model.addConstr(_rows_csr(inst) @ x <= np.zeros(1 + inst.m))
        model.setObjective(inst.p @ x, gp.GRB.MAXIMIZE)
        model.update()
        self._model, self._x = model, x
        self._cap = model.getConstrs()[0]     # the capacity row, added first

    def _solve(self, c):
        import gurobipy as gp

        model = self._model
        self._cap.RHS = c
        model.optimize()
        status = _GUROBI_STATUS.get(model.Status, f"gurobi_status_{model.Status}")
        if model.Status != gp.GRB.OPTIMAL:
            return _failed(self.inst.n, status, f"Gurobi status {model.Status}")
        # Gurobi reports the multiplier of a <= row of a max problem as +lambda.
        try:
            lam = float(self._cap.Pi)
        except gp.GurobiError:                # no dual solution (e.g. barrier without crossover)
            lam = np.nan
        return np.asarray(self._x.X, dtype=float), float(model.ObjVal), lam, status, ""

    def close(self) -> None:
        model = getattr(self, "_model", None)
        if model is not None:
            model.dispose()
            self._model = None
        env = getattr(self, "_env", None)
        if env is not None:
            env.dispose()
            self._env = None


_GUROBI_STATUS = {2: "optimal", 3: "infeasible", 4: "infeasible_or_unbounded",
                  5: "unbounded", 9: "time_limit", 7: "iteration_limit"}


# ------------------------------------------------------------------- CPLEX
_CPLEX_METHOD = {"default": 0, "primal-simplex": 1, "dual-simplex": 2, "barrier": 4}


class CplexBackend(LPBackend):
    """CPLEX through its Python API, model built once and re-solved.

    The four output streams are redirected before anything else happens, which
    is what keeps CPLEX from creating ``cplex.log`` in the working directory.
    ``solve`` changes the capacity row's right-hand side and calls ``solve``
    again; CPLEX keeps the previous basis (``advance``) on its own.

    The ``cplex`` package on PyPI without a licensed installation is the
    Community Edition, which refuses models above its size limit: such a solve
    comes back with ``status="error"`` and CPLEX's own message rather than an
    exception, so a sweep over a benchmark does not abort on it.
    """

    name = "cplex"
    methods = tuple(_CPLEX_METHOD)

    def _build(self) -> None:
        import cplex

        cp = cplex.Cplex()
        for redirect in (cp.set_log_stream, cp.set_results_stream,
                         cp.set_warning_stream, cp.set_error_stream):
            redirect(None)
        inst = self.inst
        n, m = inst.n, inst.m
        cp.parameters.threads.set(self.threads)
        cp.parameters.lpmethod.set(_CPLEX_METHOD[self.method])
        if self.time_limit is not None:
            cp.parameters.timelimit.set(float(self.time_limit))
        cp.objective.set_sense(cp.objective.sense.maximize)
        cp.variables.add(obj=inst.p.tolist(), lb=[0.0] * n, ub=[1.0] * n)
        rows = [cplex.SparsePair(ind=list(range(n)), val=inst.w.tolist())]
        rows += [cplex.SparsePair(ind=[int(i), int(j)], val=[1.0, -1.0])
                 for i, j in inst.arcs]
        cp.linear_constraints.add(lin_expr=rows, senses="L" * (1 + m), rhs=[0.0] * (1 + m))
        self._cplex = cp

    def _solve(self, c):
        import cplex

        cp = self._cplex
        cp.linear_constraints.set_rhs(0, c)
        try:
            cp.solve()
        except cplex.exceptions.CplexError as exc:
            return _failed(self.inst.n, "error", str(exc).strip())
        sol = cp.solution
        code = sol.get_status()
        status = _CPLEX_STATUS.get(code, sol.get_status_string().strip().replace(" ", "_"))
        if status != "optimal":
            return _failed(self.inst.n, status, sol.get_status_string().strip())
        # CPLEX reports the multiplier of a <= row of a max problem as +lambda.
        return (np.asarray(sol.get_values(), dtype=float),
                float(sol.get_objective_value()), float(sol.get_dual_values(0)), status, "")

    def close(self) -> None:
        cp = getattr(self, "_cplex", None)
        if cp is not None:
            cp.end()
            self._cplex = None


# CPLEX solution status codes (CPLEX reference manual); anything not listed is
# reported under its own status string.  102 is the MIP "optimal within
# tolerance", which a presolved LP can also return.
_CPLEX_STATUS = {1: "optimal", 102: "optimal", 2: "unbounded", 3: "infeasible",
                 4: "infeasible_or_unbounded", 10: "iteration_limit", 11: "time_limit"}


# -------------------------------------------------------------- CPLEX (CLI)
_CPLEX_CLI_METHODS = {"default": 0, "primal-simplex": 1, "dual-simplex": 2,
                      "barrier": 4, "network": 3, "sifting": 5}


class CplexCliBackend(LPBackend):
    """Licensed CPLEX driven through its Interactive Optimizer.

    The ``cplex`` package on PyPI is the Community Edition, which refuses
    models above 1000 rows or columns -- every real instance here.  A licensed
    CPLEX Studio installation ships a Python API only for the Python versions
    current at its release, so on a newer interpreter the licensed solver is
    reachable only through its executable.  This backend keeps one Interactive
    Optimizer process alive, reads the model once, and then per capacity issues

    ``change rhs cap <c>`` / ``optimize``

    which re-optimizes from the previous basis exactly as the in-process
    backends do, so the comparison stays fair.

    Set ``MACROITEMS_CPLEX`` to the executable, or let it be found on ``PATH``
    or under a ``CPLEX_STUDIO_DIR*`` installation.  ``seconds_solve`` is
    measured here (wall clock, including the pipe round trip); CPLEX's own
    "Solution time" is parsed as well and reported in ``message``.
    """

    name = "cplex-cli"
    methods = tuple(_CPLEX_CLI_METHODS)

    def __init__(self, inst: Instance, method: str = "default", threads: int = 1,
                 time_limit: Optional[float] = None, return_primal: bool = True):
        # Retrieving x costs an extra round trip through the pipe and grows
        # with n; timing runs that only need the value and the multiplier
        # should switch it off, and say so.
        self.return_primal = bool(return_primal)
        super().__init__(inst, method=method, threads=threads, time_limit=time_limit)

    def _build(self) -> None:
        import subprocess
        import tempfile

        exe = find_cplex_executable()
        if exe is None:
            raise RuntimeError("no CPLEX Interactive Optimizer found; set MACROITEMS_CPLEX")
        self._dir = tempfile.mkdtemp(prefix="macroitems-cplex-")
        self._path = os.path.join(self._dir, "model.lp")
        write_lp_file(self.inst, self._path, capacity=0.0, capacity_row="cap")
        self._proc = subprocess.Popen(
            [exe], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1, cwd=self._dir)
        setup = [f"read {self._path}", f"set threads {self.threads}",
                 f"set lpmethod {_CPLEX_CLI_METHODS[self.method]}"]
        if self.time_limit is not None:
            setup.append(f"set timelimit {self.time_limit}")
        self._talk(setup, marker_only=True)

    def _talk(self, commands: list[str], marker_only: bool = False) -> str:
        """Send commands and read back everything up to a marker we can detect.

        The Interactive Optimizer has no end-of-command token, so a harmless
        command whose echo is unmistakable is appended and used as a sentinel.
        Switching the log file announces both the old and the new name, so the
        sentinel must be *unique per call*: with a fixed one, each exchange
        would stop at the trailing line of the previous one and every answer
        would come back shifted by one capacity.
        """
        assert self._proc.stdin is not None and self._proc.stdout is not None
        self._token = getattr(self, "_token", 0) + 1
        sentinel = f"MACROITEMS_DONE_{self._token}_"
        self._proc.stdin.write("\n".join(commands) + f"\nset logfile {sentinel}\n")
        self._proc.stdin.flush()
        out = []
        while True:
            line = self._proc.stdout.readline()
            if not line:
                raise RuntimeError("the CPLEX process ended unexpectedly:\n" + "".join(out[-20:]))
            if sentinel in line:
                break
            out.append(line)
        return "".join(out)

    def _solve(self, c):
        sol_path = os.path.join(self._dir, "solution.xml")
        commands = [f"change rhs cap {c!r}", "optimize",
                    "display solution objective", "display solution dual cap"]
        if self.return_primal:
            # The console listing rounds to six decimals, which is enough to
            # violate the capacity row on a large instance; the solution file
            # carries full precision, and the multiplier with it.
            if os.path.exists(sol_path):
                os.remove(sol_path)
            commands.append(f"write {sol_path} sol")
        text = self._talk(commands)
        mobj = re.findall(r"Objective\s*=\s*([-+0-9.eE]+)", text)
        if not mobj:
            reason = "time_limit" if "time limit" in text.lower() else "error"
            return _failed(self.inst.n, reason, _last_meaningful_line(text))
        value = float(mobj[-1])
        mdual = re.search(r"^\s*cap\s+([-+0-9.eE]+)\s*$", text, re.MULTILINE)
        # The Interactive Optimizer lists only the nonzero duals, so a capacity
        # row missing from the listing of an optimal LP has multiplier zero --
        # the slack-capacity case, where the paper's lambda is 0 as well.
        lam = float(mdual.group(1)) if mdual else 0.0
        mtime = re.search(r"Solution time\s*=\s*([0-9.]+)", text)
        note = f"cplex solution time {mtime.group(1)}s" if mtime else ""

        if not self.return_primal:
            return np.full(self.inst.n, np.nan), value, lam, "optimal", note
        if not os.path.exists(sol_path):
            return _failed(self.inst.n, "error", "CPLEX wrote no solution file")
        with open(sol_path) as f:
            xml = f.read()
        mdual_xml = re.search(r'name="cap"[^/]*dual="([-+0-9.eE]+)"', xml)
        if mdual_xml:
            lam = float(mdual_xml.group(1))
        x = np.zeros(self.inst.n)
        for idx, val in re.findall(r'<variable name="x(\d+)"[^/]*value="([-+0-9.eE]+)"', xml):
            x[int(idx)] = float(val)
        return x, value, lam, "optimal", note

    def close(self) -> None:
        proc = getattr(self, "_proc", None)
        if proc is not None:
            try:
                if proc.stdin is not None:
                    proc.stdin.write("quit\n")
                    proc.stdin.flush()
                    proc.stdin.close()
                proc.wait(timeout=10)
            except Exception:
                proc.kill()
            self._proc = None
        directory = getattr(self, "_dir", None)
        if directory is not None:
            import shutil
            shutil.rmtree(directory, ignore_errors=True)
            self._dir = None


def find_cplex_executable() -> Optional[str]:
    """Locate the CPLEX Interactive Optimizer, or ``None``."""
    import glob as _glob
    import shutil as _shutil

    env = os.environ.get("MACROITEMS_CPLEX")
    if env and os.path.exists(env):
        return env
    found = _shutil.which("cplex")
    if found:
        return found
    patterns = [os.path.expanduser("~/ILOG/CPLEX_Studio*/cplex/bin/*/cplex"),
                "/opt/ibm/ILOG/CPLEX_Studio*/cplex/bin/*/cplex"]
    for key, value in os.environ.items():
        if key.startswith("CPLEX_STUDIO_DIR"):
            patterns.append(os.path.join(value, "cplex", "bin", "*", "cplex"))
    for pattern in patterns:
        hits = sorted(_glob.glob(pattern))
        if hits:
            return hits[-1]
    return None


def write_lp_file(inst: Instance, path: str, capacity: float,
                  capacity_row: str = "cap") -> None:
    """Write the LP relaxation in CPLEX LP format.

    Written in blocks rather than one huge string so that the largest instances
    do not need the whole file in memory at once.
    """
    with open(path, "w") as f:
        f.write(f"\\ {inst.name}: LP relaxation of the PCKP\nMaximize\n obj:")
        for i, pi in enumerate(inst.p):
            f.write(f" {pi:+.12g} x{i}" + ("\n " if i % 8 == 7 else ""))
        f.write(f"\nSubject To\n {capacity_row}:")
        for i, wi in enumerate(inst.w):
            f.write(f" {wi:+.12g} x{i}" + ("\n " if i % 8 == 7 else ""))
        f.write(f" <= {capacity:.12g}\n")
        for k, (i, j) in enumerate(inst.arcs):
            f.write(f" p{k}: x{i} - x{j} <= 0\n")
        f.write("Bounds\n")
        for i in range(inst.n):
            f.write(f" x{i} <= 1\n")
        f.write("End\n")


def _last_meaningful_line(text: str) -> str:
    for line in reversed(text.strip().splitlines()):
        stripped = line.strip().lstrip("CPLEX>").strip()
        if stripped:
            return stripped[:200]
    return ""


# ------------------------------------------------------------------ registry
_BACKENDS: dict[str, type[LPBackend]] = {
    "highs": HighsBackend,
    "scipy": ScipyBackend,
    "gurobi": GurobiBackend,
    "cplex": CplexBackend,
    "cplex-cli": CplexCliBackend,
}

_AVAILABLE: dict[str, bool] = {}
_WHY_NOT: dict[str, str] = {}


def _have(name: str) -> bool:
    """True if ``name``'s package imports and, for the licensed solvers, starts.

    Gurobi and CPLEX import fine without a usable licence, so the check goes as
    far as creating (and releasing) an environment.  Results are cached, the
    reason for a failure is kept in ``_WHY_NOT`` (the ``highspy``/``ortools``
    symbol clash of the module docstring shows up there), and an unusable
    backend never raises out of this function.
    """
    if name not in _AVAILABLE:
        try:
            if name == "highs":
                import highspy  # noqa: F401
            elif name == "scipy":
                from scipy.optimize import linprog  # noqa: F401
            elif name == "gurobi":
                import gurobipy as gp
                env = gp.Env(empty=True)
                env.setParam("OutputFlag", 0)
                env.setParam("LogFile", "")
                env.start()
                env.dispose()
            elif name == "cplex-cli":
                if find_cplex_executable() is None:
                    raise RuntimeError("no CPLEX Interactive Optimizer found "
                                       "(set MACROITEMS_CPLEX)")
            elif name == "cplex":
                import cplex
                cp = cplex.Cplex()
                for redirect in (cp.set_log_stream, cp.set_results_stream,
                                 cp.set_warning_stream, cp.set_error_stream):
                    redirect(None)
                cp.end()
            else:
                raise ValueError(f"unknown LP backend {name!r}")
            _AVAILABLE[name] = True
        except Exception as exc:
            _AVAILABLE[name] = False
            _WHY_NOT[name] = f"{type(exc).__name__}: {exc}"
    return _AVAILABLE[name]


def available_lp_backends() -> list[str]:
    """Names of the LP backends usable in this environment, in a fixed order."""
    return [name for name in _BACKENDS if _have(name)]


def get_lp_backend(name: str) -> type[LPBackend]:
    """The :class:`LPBackend` subclass called ``name``."""
    if name not in _BACKENDS:
        raise ValueError(f"unknown LP backend {name!r}; known: {', '.join(_BACKENDS)}")
    if not _have(name):
        raise RuntimeError(
            f"LP backend {name!r} is not available: {_WHY_NOT.get(name, 'unknown reason')}. "
            f"Available: {', '.join(available_lp_backends()) or 'none'}.")
    return _BACKENDS[name]


def solve_lp(inst: Instance, c: float, backend: str = "highs", method: str = "default",
             threads: int = 1, time_limit: Optional[float] = None) -> LPResult:
    """Solve the LP relaxation of ``inst`` at capacity ``c`` once, from scratch.

    Convenience wrapper: it builds a backend, solves, and closes it, so
    ``seconds_build`` and ``seconds_solve`` are both paid.  Use a backend
    object directly to amortize the build over several capacities.

    The default ``backend="highs"`` falls back to ``"scipy"`` when ``highspy``
    cannot be loaded -- the same HiGHS code, reached through scipy, which is a
    hard dependency of this package.  The fallback warns, because it changes
    what a timing measures: it is the usual symptom of the ``ortools`` symbol
    clash described in the module docstring.
    """
    if backend == "highs" and not _have("highs"):
        warnings.warn(f"highspy unusable ({_WHY_NOT.get('highs', 'unknown reason')}); "
                      "solve_lp falls back to the scipy backend", RuntimeWarning, stacklevel=2)
        backend = "scipy"
    cls = get_lp_backend(backend)
    solver = cls(inst, method=method, threads=threads, time_limit=time_limit)
    try:
        return solver.solve(c)
    finally:
        solver.close()
