"""`calculate_gaussian_correlation` must rank frames whose columns are not strings.

The ranking is printed with a ``{feature:30s}`` format, which raised
``ValueError: Unknown format code 's' for object of type 'int'`` for any integer column
label -- the default labels of ``pd.DataFrame(ndarray)`` -- for every scoring method, after
all the scoring work was already done.
"""

import numpy as np
import pandas as pd
import pytest

from tribblefis.gauss_math import calculate_gaussian_correlation


@pytest.mark.parametrize("method", ["wasserstein", "bhattacharyya", "composite", "higher_order"])
def test_integer_column_names_rank_without_error(method):
    rng = np.random.default_rng(0)
    y = pd.Series(rng.integers(0, 2, 300))
    X = pd.DataFrame(rng.normal(size=(300, 3)))  # columns 0, 1, 2
    X[0] = X[0] + 3 * y  # column 0 is the informative one
    ranked = calculate_gaussian_correlation(X, y, method=method)
    assert [f for f, _ in ranked][0] == 0
    assert {f for f, _ in ranked} == {0, 1, 2}
