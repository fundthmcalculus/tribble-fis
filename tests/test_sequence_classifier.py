import numpy as np
import pandas as pd

from tribblefis.gaussian_classifier import (
    MixtureOfGaussiansFuzzyClassifier,
    MixtureOfGaussiansFuzzySequenceClassifier,
)


def _make_blobs(n_per_class: int = 60, seed: int = 0):
    """Two well-separated classes plus a confusable overlap region."""
    rng = np.random.default_rng(seed)
    # Class "a": centered low, Class "b": centered high, with overlap in the middle.
    a = rng.normal(loc=[0.0, 0.0], scale=0.6, size=(n_per_class, 2))
    b = rng.normal(loc=[3.0, 3.0], scale=0.6, size=(n_per_class, 2))
    # Overlap samples that the primary model will confuse.
    a_overlap = rng.normal(loc=[1.5, 1.5], scale=0.3, size=(n_per_class // 3, 2))
    b_overlap = rng.normal(loc=[1.5, 1.5], scale=0.3, size=(n_per_class // 3, 2))

    X = np.vstack([a, b, a_overlap, b_overlap])
    y = (
        ["a"] * n_per_class
        + ["b"] * n_per_class
        + ["a"] * (n_per_class // 3)
        + ["b"] * (n_per_class // 3)
    )
    X = pd.DataFrame(X, columns=["f0", "f1"])
    y = pd.Series(y)
    return X, y


def test_sequence_fits_specialists():
    X, y = _make_blobs()
    clf = MixtureOfGaussiansFuzzySequenceClassifier(max_layers=4, min_confused=5, min_class_samples=2)
    clf.fit(X, y)
    # Primary + at least one specialist keyed to a confused class.
    assert clf.n_layers >= 2
    assert clf.n_layers <= 4
    # Each specialist is keyed to a real class label.
    assert set(clf.confused_classes_).issubset({"a", "b"})
    # A specialist's confused class is never reused.
    assert len(clf.confused_classes_) == len(set(clf.confused_classes_))


def test_predict_returns_only_real_labels():
    X, y = _make_blobs()
    clf = MixtureOfGaussiansFuzzySequenceClassifier(max_layers=4, min_confused=5, min_class_samples=2)
    clf.fit(X, y)
    preds = clf.predict(X)
    assert len(preds) == len(X)
    # The anomaly label must never leak into the final predictions.
    assert clf.anomaly_label not in set(preds.tolist())
    assert set(preds.tolist()).issubset({"a", "b"})


def test_only_confused_class_predictions_can_change():
    """A sample is only refined if its current prediction is a specialist's class."""
    X, y = _make_blobs()
    clf = MixtureOfGaussiansFuzzySequenceClassifier(max_layers=4, min_confused=5, min_class_samples=2)
    clf.fit(X, y)

    primary_pred = np.asarray(clf.layers_[0].predict(X), dtype=object)
    cascade_pred = clf.predict(X)
    changed = primary_pred != cascade_pred

    # Anything that changed must have started as one of the confused classes.
    confused = set(clf.confused_classes_)
    assert set(primary_pred[changed].tolist()).issubset(confused)


def test_cascade_accuracy_is_comparable():
    X, y = _make_blobs()
    clf = MixtureOfGaussiansFuzzySequenceClassifier(max_layers=4, min_confused=5, min_class_samples=2)
    clf.fit(X, y)

    y_obj = y.values.astype(object)
    primary_acc = np.mean(np.asarray(clf.layers_[0].predict(X), dtype=object) == y_obj)
    cascade_acc = np.mean(clf.predict(X) == y_obj)

    # Specialists arbitrate within a confused class, so the cascade should not
    # meaningfully degrade overall accuracy.
    assert cascade_acc >= primary_acc - 0.05


def test_anomaly_stops_refinement():
    """A point far outside every trained region keeps the primary prediction."""
    X, y = _make_blobs()
    clf = MixtureOfGaussiansFuzzySequenceClassifier(max_layers=4, min_confused=5, min_class_samples=2)
    clf.fit(X, y)

    # A far-away outlier: a specialist should flag it as an anomaly and leave the
    # primary model's prediction untouched (no crash, valid label).
    X_out = pd.DataFrame([[100.0, 100.0]], columns=["f0", "f1"])
    primary_pred = np.asarray(clf.layers_[0].predict(X_out), dtype=object)
    cascade_pred = clf.predict(X_out)
    assert cascade_pred[0] == primary_pred[0]


def test_anomaly_label_collision_raises():
    X, y = _make_blobs()
    clf = MixtureOfGaussiansFuzzySequenceClassifier(anomaly_label="a")
    try:
        clf.fit(X, y)
    except ValueError:
        return
    raise AssertionError("Expected ValueError on anomaly_label collision with a real class.")


if __name__ == "__main__":
    test_sequence_fits_specialists()
    test_predict_returns_only_real_labels()
    test_only_confused_class_predictions_can_change()
    test_cascade_accuracy_is_comparable()
    test_anomaly_stops_refinement()
    test_anomaly_label_collision_raises()
    print("All sequence classifier tests passed!")
