"""macroitems: the LP relaxation of the precedence-constrained knapsack problem
through parametric maximum closure (canonical macroitem sequence, LP at a
capacity by a Newton search on the weight price, canonical dual certificate,
optimal-face dimensions), with LP-solver baselines (HiGHS, Gurobi, CPLEX) behind
one interface, for verification and timing."""
from .instance import Instance, layered_grid, random_dag, running_example, read_minelib_upit
from .closure import ClosureSolver, max_closure, is_closure
from .path import canonical_path, solve_capacity, solution_from_path, canonical_dual, canonical_reduced_costs
from .lp import (LPResult, LPBackend, solve_lp, available_lp_backends, get_lp_backend,
                 HighsBackend, ScipyBackend, GurobiBackend, CplexBackend)
from .faces import face_dimensions

__version__ = "0.1.0-pilot"
