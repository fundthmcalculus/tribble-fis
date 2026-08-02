"""Session-wide test setup.

Two things, both about making a test run finish unattended.
"""

import sys
from pathlib import Path

# A test that draws must not open a window. `tests/test_regression.py` calls
# `plt.show()` through its plotting helper, and `regression.py` and
# `gauss_plot.py` do the same from library code; under an interactive backend
# each of those blocks the whole run until someone closes the figure by hand.
# Selecting Agg before pyplot is first imported makes `show()` a no-op and keeps
# the figures headless. This must happen at import time, before any test module
# pulls in pyplot.
import matplotlib

matplotlib.use("Agg", force=True)

# `tests/test_benchmarks.py` imports the `benchmarks` package, which lives at the
# repo root rather than in the installed `src/tribblefis` wheel. A bare `pytest`
# invocation (as opposed to `python -m pytest`, which happens to add the CWD)
# would otherwise not find it.
ROOT = str(Path(__file__).resolve().parent)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
