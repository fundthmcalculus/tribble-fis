import os

import matplotlib

# Force a non-interactive backend for the whole test session. Several test
# modules (e.g. test_regression.py) call plt.show() via helper plotting
# functions; on a machine with a live DISPLAY that would otherwise try to
# open a GUI window and block the run instead of completing headlessly.
matplotlib.use("Agg")

# test_textclassifier.py is a standalone experiment script, not an automated
# test: it has no test_* functions/TestCase classes, runs top-level training
# code on import, requires the optional `sentence_transformers` dependency
# (heavy: pulls in torch/CUDA), and points at a hardcoded local Windows path
# (`C:\work\ai-research\...`) that only exists on the original author's
# machine. It only matches pytest's `test_*.py` collection pattern by name.
collect_ignore = [os.path.join(os.path.dirname(__file__), "test_textclassifier.py")]
