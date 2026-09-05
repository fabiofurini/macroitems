"""Integer rescaling must not change the answer, only its units.

``Instance.scaled_to_integers`` multiplies every profit and every weight by
the same positive integer, which is what lets the parametric machinery run in
exact arithmetic on decimal data.  What that rescaling does and does not
change is easy to get wrong in reporting code:

* profits, weights, capacities and the optimal value are all multiplied by
  the scale, so they have to be divided back;
* **ratios are invariant**, since the numerator and the denominator are
  multiplied by the same factor.  Dividing a ratio by the scale, as a
  reporting script once did here, silently reports a breakpoint that is wrong
  by orders of magnitude while every other number looks right.
"""
from __future__ import annotations

import numpy as np
import pytest

from macroitems import canonical_path, random_dag, solution_from_path, solve_capacity
from macroitems.instance import Instance


def decimal_instance(seed: int, decimals: int = 2) -> Instance:
    """A random instance whose data are decimals with exactly ``decimals`` digits.

    Built by dividing integers by a power of ten, so that each value's shortest
    decimal representation really has that many digits -- multiplying integers
    by 0.011 instead would give values whose repr runs to seventeen digits, and
    those are (correctly) refused by the scaling.
    """
    base = random_dag(40, 2.0, seed=seed)
    factor = 10 ** decimals
    p = base.p / factor
    w = base.w / factor
    return Instance(p, w, base.arcs.copy(), name=f"decimal{seed}")


@pytest.mark.parametrize("seed", range(6))
def test_scaling_preserves_ratios_and_scales_values(seed):
    inst = decimal_instance(seed)
    scaled, scale = inst.scaled_to_integers()
    assert scale > 1 and scaled.is_integral()

    original = canonical_path(inst)
    rescaled = canonical_path(scaled)

    # same macroitems, in the same order
    assert len(original.macroitems) == len(rescaled.macroitems)
    for a, b in zip(original.macroitems, rescaled.macroitems):
        assert np.array_equal(a, b)

    # ratios are invariant; cumulative profits and weights scale
    np.testing.assert_allclose(original.ratios, rescaled.ratios, rtol=1e-9)
    np.testing.assert_allclose(original.P * scale, rescaled.P, rtol=1e-9)
    np.testing.assert_allclose(original.W * scale, rescaled.W, rtol=1e-9)


@pytest.mark.parametrize("seed", range(6))
def test_scaling_preserves_the_lp_solution(seed):
    inst = decimal_instance(seed)
    scaled, scale = inst.scaled_to_integers()
    path = canonical_path(scaled)
    if path.q == 0:
        pytest.skip("no positive-ratio macroitem")

    c = 0.5 * float(path.W[path.q]) / scale
    here = solve_capacity(inst, c)
    there = solution_from_path(scaled, path, c * scale)

    assert there.value / scale == pytest.approx(here.value, rel=1e-9)
    assert there.lam == pytest.approx(here.lam, rel=1e-9)     # a price is a ratio: invariant
    np.testing.assert_allclose(there.x, here.x, atol=1e-9)


def test_capacity_metadata_scales_with_the_weights():
    inst = decimal_instance(0)
    inst.meta["capacity"] = 12.34
    scaled, scale = inst.scaled_to_integers()
    assert scaled.meta["capacity"] == pytest.approx(12.34 * scale)
    assert scaled.meta["integer_scale"] == scale
