"""Decoupled t-norm/t-conorm selection, and the Einstein family.

Covers three things:

1. The Einstein product and Einstein sum, added here, behave as a t-norm/t-conorm
   pair should -- and specifically that they are De Morgan duals, since that is
   the property the rest of the design leans on.
2. `resolve_norm_pair` refuses a mixed pair unless asked twice. Mixing families
   is legal but is not something a caller should be able to do by accident,
   because the anomaly rule builds a complement out of the conorm and quietly
   stops meaning what it says when the pair is not dual.
3. The selection actually reaches the models. The regressor previously had no
   way to express it at all -- `tsk_firing_strengths` read the operator off the
   anomaly parameters, which regression never supplies -- so a test that merely
   constructs an estimator would have passed while the knob did nothing.
"""

import numpy as np
import pandas as pd
import pytest

from tribblefis.gauss_data import (
    NORM_FAMILIES,
    AnomalyParameters,
    NormPair,
    resolve_norm_pair,
)
from tribblefis.gauss_math import t_complement, t_conorm, t_norm

# A 60x60 grid over the unit square, plus the corners, which is where the
# Hamacher singularities lived and where the boundary laws bite.
_G = np.linspace(0.0, 1.0, 60)
_X, _Y = (a.ravel() for a in np.meshgrid(_G, _G))
_TOL = 1e-10


class TestAllFamilies:
    @pytest.mark.parametrize("family", NORM_FAMILIES)
    def test_stays_in_unit_interval(self, family):
        for op in (t_norm, t_conorm):
            out = op(_X, _Y, family)
            assert np.all(np.isfinite(out)), f"{family}: non-finite value"
            assert out.min() >= -_TOL and out.max() <= 1 + _TOL, (
                f"{family}: left [0,1] -> [{out.min()}, {out.max()}]"
            )

    @pytest.mark.parametrize("family", NORM_FAMILIES)
    def test_de_morgan_duality(self, family):
        """S(x, y) == 1 - T(1-x, 1-y) under the standard negation."""
        lhs = t_conorm(_X, _Y, family)
        rhs = t_complement(t_norm(t_complement(_X), t_complement(_Y), family))
        assert np.allclose(lhs, rhs, atol=1e-9), f"{family}: not a De Morgan dual pair"

    @pytest.mark.parametrize("family", NORM_FAMILIES)
    def test_boundary_conditions(self, family):
        ones, zeros = np.ones_like(_X), np.zeros_like(_X)
        assert np.allclose(t_norm(_X, ones, family), _X, atol=1e-9)      # T(x,1) = x
        assert np.allclose(t_conorm(_X, zeros, family), _X, atol=1e-9)   # S(x,0) = x
        assert np.allclose(t_norm(zeros, zeros, family), 0.0, atol=1e-9)
        assert np.allclose(t_conorm(ones, ones, family), 1.0, atol=1e-9)

    @pytest.mark.parametrize("family", NORM_FAMILIES)
    def test_commutative(self, family):
        for op in (t_norm, t_conorm):
            assert np.allclose(op(_X, _Y, family), op(_Y, _X, family), atol=1e-12)

    @pytest.mark.parametrize("family", NORM_FAMILIES)
    def test_reduction_branch_honours_the_family(self, family):
        """t_norm(M, None, f) must reduce with f, not silently with the default.

        This is the shape of the bug fixed in #22; keeping a test on it for the
        new family too means einstein cannot regress into the same hole.
        """
        M = np.array([[0.2, 0.7, 0.5], [0.9, 0.4, 0.6], [0.0, 1.0, 0.3]])
        for op in (t_norm, t_conorm):
            expected = op(op(M[:, 0], M[:, 1], family), M[:, 2], family)
            assert np.allclose(op(M, None, family), expected, atol=1e-12)


class TestEinstein:
    def test_matches_closed_form(self):
        assert np.allclose(t_norm(_X, _Y, "einstein"),
                           (_X * _Y) / (2.0 - (_X + _Y - _X * _Y)), atol=1e-12)
        assert np.allclose(t_conorm(_X, _Y, "einstein"),
                           (_X + _Y) / (1.0 + _X * _Y), atol=1e-12)

    def test_no_singularity_at_the_corners(self):
        """Both denominators lie in [1, 2], so unlike Hamacher there is nothing
        to guard -- 0/0 and the xy->1 blow-up simply cannot arise."""
        corners = np.array([0.0, 0.0, 1.0, 1.0]), np.array([0.0, 1.0, 0.0, 1.0])
        for op in (t_norm, t_conorm):
            out = op(*corners, "einstein")
            assert np.all(np.isfinite(out))

    def test_ordering_against_the_other_families(self):
        """Einstein product sits between Lukasiewicz and the algebraic product;
        the sum mirrors it. Guards against a sign or factor slip that would still
        pass the boundary tests."""
        x = np.array([0.3, 0.5, 0.8]); y = np.array([0.4, 0.5, 0.9])
        assert np.all(t_norm(x, y, "luk") <= t_norm(x, y, "einstein") + 1e-12)
        assert np.all(t_norm(x, y, "einstein") <= t_norm(x, y, "probability") + 1e-12)
        assert np.all(t_conorm(x, y, "probability") <= t_conorm(x, y, "einstein") + 1e-12)
        assert np.all(t_conorm(x, y, "einstein") <= t_conorm(x, y, "luk") + 1e-12)


class TestResolveNormPair:
    def test_default_is_a_matched_pair(self):
        pair = resolve_norm_pair()
        assert pair.is_de_morgan and pair == NormPair("min/max", "min/max")

    @pytest.mark.parametrize("family", NORM_FAMILIES)
    def test_coupled_selection_sets_both_halves(self, family):
        assert resolve_norm_pair(family) == NormPair(family, family)

    def test_single_override_still_matched_is_allowed(self):
        """Naming the same family explicitly on one side is not a mixed pair."""
        assert resolve_norm_pair("luk", t_norm="luk") == NormPair("luk", "luk")

    def test_mixed_pair_is_refused_by_default(self):
        with pytest.raises(ValueError, match="De Morgan"):
            resolve_norm_pair(t_norm="probability", t_conorm="luk")

    def test_mixed_pair_allowed_on_explicit_opt_in(self):
        pair = resolve_norm_pair(t_norm="probability", t_conorm="luk",
                                 allow_mixed_norms=True)
        assert pair == NormPair("probability", "luk")
        assert not pair.is_de_morgan

    def test_overriding_one_half_of_a_coupled_choice_is_mixing(self):
        with pytest.raises(ValueError, match="De Morgan"):
            resolve_norm_pair("einstein", t_conorm="hamacher")

    @pytest.mark.parametrize("bad", ["godel", "", "Min/Max", "product"])
    def test_unknown_family_is_rejected(self, bad):
        with pytest.raises(ValueError, match="Invalid"):
            resolve_norm_pair(bad)

    def test_unknown_family_rejected_even_when_mixing_is_allowed(self):
        with pytest.raises(ValueError, match="Invalid"):
            resolve_norm_pair(t_norm="godel", t_conorm="luk", allow_mixed_norms=True)


class TestAnomalyParameters:
    def test_positional_construction_still_works(self):
        """The new fields are appended, so existing positional callers are safe."""
        ap = AnomalyParameters(True, 0.9, "anomaly", "einstein", "gaussian")
        assert ap.norm_conorm == "einstein" and ap.member_function == "gaussian"
        assert ap.norms() == NormPair("einstein", "einstein")

    def test_defaults_resolve_to_the_default_family(self):
        assert AnomalyParameters().norms() == resolve_norm_pair()

    def test_mixed_pair_refused_at_resolve_time(self):
        ap = AnomalyParameters(norm_conorm="luk", t_conorm="hamacher")
        with pytest.raises(ValueError, match="De Morgan"):
            ap.norms()


class TestSelectionReachesTheModels:
    """The plumbing tests. Each asserts the operator changes the OUTPUT, not just
    that the parameter can be set."""

    @staticmethod
    def _regression_data(n=240, seed=0):
        rng = np.random.default_rng(seed)
        X = pd.DataFrame(rng.normal(size=(n, 3)), columns=["a", "b", "c"])
        y = pd.Series(2.0 * X["a"] - X["b"] + 0.5 * X["c"] ** 2
                      + rng.normal(scale=0.2, size=n), name="y_value")
        return X, y

    def test_firing_strengths_respect_an_explicit_pair(self):
        from tribblefis.gauss_math import (create_gaussian_membership_dict,
                                           tsk_firing_strengths)
        X, y = self._regression_data()
        labels = pd.Series(np.where(y > y.median(), "hi", "lo"))
        model = create_gaussian_membership_dict(X, labels, top_n_var_names=list(X.columns))

        base, _ = tsk_firing_strengths(X, model, norms=resolve_norm_pair("min/max"))
        alt, _ = tsk_firing_strengths(X, model, norms=resolve_norm_pair("probability"))
        assert not np.allclose(base, alt), (
            "firing strengths ignored the norms argument -- the regression path "
            "has no anomaly parameters, so this argument is its only channel"
        )

    def test_regressor_operator_changes_predictions(self):
        from tribblefis.gaussian_regressor import MixtureOfGaussiansFuzzyRegressor
        X, y = self._regression_data()
        preds = {}
        for family in NORM_FAMILIES:
            m = MixtureOfGaussiansFuzzyRegressor(
                n_output_buckets=3, tsk_order="1st", top_n=-1,
                norm_conorm=family, random_state=0)
            preds[family] = m.fit(X, y).predict(X)
        assert not np.allclose(preds["min/max"], preds["probability"]), (
            "the regressor ignored norm_conorm; before this change it was hard-wired"
        )
        assert not np.allclose(preds["min/max"], preds["einstein"])

    def test_regressor_refuses_a_mixed_pair_without_opt_in(self):
        from tribblefis.gaussian_regressor import MixtureOfGaussiansFuzzyRegressor
        X, y = self._regression_data()
        m = MixtureOfGaussiansFuzzyRegressor(t_norm="probability", t_conorm="luk")
        with pytest.raises(ValueError, match="De Morgan"):
            m.fit(X, y)

    def test_regressor_accepts_a_mixed_pair_on_opt_in(self):
        from tribblefis.gaussian_regressor import MixtureOfGaussiansFuzzyRegressor
        X, y = self._regression_data()
        m = MixtureOfGaussiansFuzzyRegressor(
            n_output_buckets=3, tsk_order="1st", t_norm="probability",
            t_conorm="luk", allow_mixed_norms=True, random_state=0)
        assert np.all(np.isfinite(m.fit(X, y).predict(X)))

    def test_regressor_params_round_trip_for_sklearn_clone(self):
        """__init__ must only store its arguments, or clone() breaks."""
        from sklearn.base import clone
        from tribblefis.gaussian_regressor import MixtureOfGaussiansFuzzyRegressor
        m = MixtureOfGaussiansFuzzyRegressor(norm_conorm="einstein")
        assert clone(m).get_params()["norm_conorm"] == "einstein"
