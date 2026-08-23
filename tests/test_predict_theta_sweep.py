"""`simple_gaussian_predict_sweep` must equal per-threshold prediction, exactly.

The theta sweep in the open-set experiment (Table 4.4b) rebuilt the whole model
for every theta, though theta enters only at the anomaly step -- the class rule
firing ahead of it is identical across thetas. `simple_gaussian_predict_sweep`
computes that firing once and reuses it. The refactor is only worth anything if
it is *bit-identical* to the slow path, so these tests pin exactly that: the
sweep result at each threshold equals `simple_gaussian_predict` on a model whose
`anomaly_params.threshold` has been swapped to that value.

They also pin the two behaviours the sweep depends on for correctness: the class
firing is not mutated between thresholds (so an earlier theta cannot leak into a
later one), and a model with no anomaly rule is threshold-independent.
"""

import unittest

import numpy as np
import pandas as pd

from tribblefis.gauss_data import (
    AnomalyParameters,
    GaussianMembership,
    Rule,
    SimpleGaussianClassifierModel,
)
from tribblefis.gauss_math import (
    simple_gaussian_predict,
    simple_gaussian_predict_sweep,
)

THETAS = [0.1, 0.3, 0.5, 0.7, 0.9, 0.99]


def _model(conorm="hamacher", include_anomaly=True, threshold=0.5):
    f1_low = GaussianMembership.create(mu=0.0, sigma=1.0)
    f1_high = GaussianMembership.create(mu=5.0, sigma=1.0)
    f2_low = GaussianMembership.create(mu=0.0, sigma=1.0)
    f2_high = GaussianMembership.create(mu=10.0, sigma=1.0)
    rules = [
        Rule(antecedents={"f1": [f1_low.id], "f2": [f2_low.id]}, consequent=0),
        Rule(antecedents={"f1": [f1_high.id], "f2": [f2_high.id]}, consequent=1),
    ]
    return SimpleGaussianClassifierModel(
        input_mfs=[f1_low, f1_high, f2_low, f2_high],
        rules=rules,
        anomaly_params=AnomalyParameters(
            include_anomaly=include_anomaly,
            threshold=threshold,
            label="anomaly",
            norm_conorm=conorm,
        ),
    )


def _data():
    rng = np.random.default_rng(0)
    n = 400
    # A spread that puts some rows squarely in a class and some far outside, so
    # the anomaly column actually changes hands as theta moves.
    return pd.DataFrame(
        {
            "f1": np.r_[rng.normal(0, 1, n // 2), rng.normal(5, 1, n // 2)],
            "f2": np.r_[rng.normal(0, 1, n // 2), rng.normal(10, 3, n // 2)],
        }
    )


class TestPredictThetaSweep(unittest.TestCase):
    def test_matches_per_threshold_prediction_every_family(self):
        X = _data()
        for conorm in ("min/max", "probability", "luk", "hamacher", "einstein"):
            model = _model(conorm=conorm)
            swept = simple_gaussian_predict_sweep(X, model, THETAS)
            for th in THETAS:
                m_th = model._replace(
                    anomaly_params=model.anomaly_params._replace(threshold=th)
                )
                ref = simple_gaussian_predict(X, m_th)
                self.assertTrue(
                    np.array_equal(swept[th].astype(str), ref.astype(str)),
                    f"conorm={conorm} theta={th}: sweep != per-threshold",
                )

    def test_firing_is_not_mutated_between_thresholds(self):
        # If the class firing leaked between thresholds, evaluating in a different
        # theta order would change the answer. It must not.
        X = _data()
        model = _model()
        forward = simple_gaussian_predict_sweep(X, model, THETAS)
        backward = simple_gaussian_predict_sweep(X, model, list(reversed(THETAS)))
        for th in THETAS:
            self.assertTrue(np.array_equal(forward[th], backward[th]))

    def test_no_anomaly_rule_is_threshold_independent(self):
        X = _data()
        model = _model(include_anomaly=False)
        swept = simple_gaussian_predict_sweep(X, model, THETAS)
        base = swept[THETAS[0]]
        for th in THETAS:
            self.assertTrue(np.array_equal(swept[th], base))


if __name__ == "__main__":
    unittest.main()
