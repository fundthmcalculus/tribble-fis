"""`_close_scalar` must be `np.isclose` for two floats, edge cases included.

It replaces `np.isclose` inside `_is_close`, which is called from an O(n^2) dedup
scan and dominated `MembershipDict.to_simple_model` -- 2.9M calls on one
RT-IOT2022 fold, at 8.06 us each against 0.12 us here.

A faster function that is *almost* `np.isclose` would be worse than the slow one,
because the difference would only ever show up as a membership function that did
or did not get deduplicated. So these tests pin the semantics rather than the
speed, and specifically pin the three places the bare
``|a - b| <= atol + rtol * |b|`` formula disagrees with numpy:

* NaN is close to nothing, itself included (numpy's ``equal_nan=False`` default);
* same-signed infinities ARE close, though the formula computes ``nan <= inf``;
* an infinity is never close to a finite value, though the formula computes
  ``inf <= inf`` and says True.

`test_matches_numpy_on_random_pairs` is the general guard; the named cases exist
so a failure says which rule broke.
"""

import unittest

import numpy as np

from tribblefis.gauss_data import DEFAULT_DEDUP_ATOL, DEFAULT_DEDUP_RTOL, _close_scalar

INF = float("inf")
NAN = float("nan")

TOLERANCES = [
    (DEFAULT_DEDUP_RTOL, DEFAULT_DEDUP_ATOL),
    (0.0, 0.0),
    (1e-6, 1e-9),
    (1e-2, 1e-3),
]


class TestCloseScalarSemantics(unittest.TestCase):
    def _agrees(self, a, b, rtol, atol):
        self.assertEqual(
            _close_scalar(a, b, rtol, atol),
            bool(np.isclose(a, b, rtol=rtol, atol=atol)),
            f"a={a!r} b={b!r} rtol={rtol} atol={atol}",
        )

    def test_nan_is_close_to_nothing(self):
        for rtol, atol in TOLERANCES:
            for pair in [(NAN, NAN), (NAN, 1.0), (1.0, NAN), (NAN, INF)]:
                self._agrees(*pair, rtol, atol)
                self.assertFalse(_close_scalar(*pair, rtol, atol))

    def test_same_signed_infinities_are_close(self):
        for rtol, atol in TOLERANCES:
            self.assertTrue(_close_scalar(INF, INF, rtol, atol))
            self.assertTrue(_close_scalar(-INF, -INF, rtol, atol))
            self._agrees(INF, INF, rtol, atol)
            self._agrees(-INF, -INF, rtol, atol)

    def test_infinity_is_not_close_to_finite_or_to_its_negation(self):
        for rtol, atol in TOLERANCES:
            for pair in [(INF, -INF), (-INF, INF), (INF, 1.0), (1.0, INF), (-INF, 0.0)]:
                self._agrees(*pair, rtol, atol)
                self.assertFalse(_close_scalar(*pair, rtol, atol))

    def test_exact_equality_is_close_at_zero_tolerance(self):
        for a in (0.0, -0.0, 1.0, -2.5, 1e300):
            self.assertTrue(_close_scalar(a, a, 0.0, 0.0))

    def test_tolerance_is_relative_to_the_second_argument(self):
        # numpy's formula is |a - b| <= atol + rtol * |b|, which is asymmetric.
        # Pinning it explicitly so a "tidier" symmetric rewrite is caught.
        rtol, atol = 1e-2, 0.0
        self._agrees(100.0, 101.0, rtol, atol)
        self._agrees(101.0, 100.0, rtol, atol)

    def test_matches_numpy_on_random_pairs(self):
        rng = np.random.default_rng(0)
        for rtol, atol in TOLERANCES:
            a = rng.normal(0.0, 1e3, 2000)
            b = a + rng.normal(0.0, 1e-2, 2000)
            for x, y in zip(a, b):
                self.assertEqual(
                    _close_scalar(float(x), float(y), rtol, atol),
                    bool(np.isclose(x, y, rtol=rtol, atol=atol)),
                    f"x={x!r} y={y!r} rtol={rtol} atol={atol}",
                )

    def test_matches_numpy_across_a_grid_of_special_values(self):
        pool = [0.0, -0.0, 1e-12, 1e-3, 1.0, -1.0, 1e6, INF, -INF, NAN]
        for rtol, atol in TOLERANCES:
            for a in pool:
                for b in pool:
                    self._agrees(a, b, rtol, atol)


class TestIsCloseUsesIt(unittest.TestCase):
    """`_is_close` must keep behaving the same for the membership types."""

    def test_gaussian_membership_close_and_not_close(self):
        from tribblefis.gauss_data import GaussianMembership
        from tribblefis.gauss_data import _is_close

        a = GaussianMembership(mu=1.0, sigma=0.5)
        near = GaussianMembership(mu=1.0 + 1e-6, sigma=0.5)
        far = GaussianMembership(mu=5.0, sigma=0.5)
        self.assertTrue(_is_close(a, near))
        self.assertFalse(_is_close(a, far))

    def test_different_types_are_never_close(self):
        from tribblefis.gauss_data import GaussianMembership, TriangularMembership
        from tribblefis.gauss_data import _is_close

        self.assertFalse(
            _is_close(
                GaussianMembership(mu=1.0, sigma=1.0),
                TriangularMembership(a=0.0, b=1.0, c=2.0),
            )
        )


if __name__ == "__main__":
    unittest.main()
