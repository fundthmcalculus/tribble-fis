"""Session-wide test setup.

Three things: two that make a test run finish unattended, and one that stops a
run whose result would be a lie about code nobody is executing.
"""

import sys
from pathlib import Path

import pytest

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


# ---------------------------------------------------------------------------
# The venv must run `src/`, not a copy of it.
# ---------------------------------------------------------------------------
#
# tribble-fis#214. A non-editable install puts a *copy* of the package in
# `site-packages/tribblefis/`, taken at the last `uv sync`. `import tribblefis`
# then runs whatever the code was at that moment, while `git status`, `git log`
# and the files on disk all show current `main`, and nothing warns you.
#
# What it looks like from the inside: after pulling #200, six tests in
# `tests/test_auto_tsk_order.py` failed with
#
#     AttributeError: 'TribbleRegressor' object has no attribute 'tsk_order_'
#
# while `grep -n tsk_order_ src/tribblefis/gaussian_regressor.py` found it on
# line 179 and CI was green. The two obvious readings -- "main is broken" and
# "it's a Windows/3.13 platform break" -- are both wrong, and both expensive.
# Timing measurements taken that way are worse still: they are silently a
# profile of the old code, and nothing about the number says so.
#
# The issue offered two fixes: a note in CLAUDE.md telling the next person to
# re-run `uv sync`, or making the install editable so the question cannot come
# up. The ask was for the second -- "I want option 2, that way it's always
# right" -- and this is the enforcement half of it. `uv sync` already installs
# the root project editable by default; what was missing is anything that
# *notices* when the venv is not in that state.
#
# It aborts the session rather than warning. A warning scrolls past the first of
# 840 tests and gets read after the debugging, not before, which is the same
# hour #214 spent. It also fires early enough to matter: at `pytest_configure`,
# before a single test runs.
#
# Not a nicety, either -- the editable layout is already load-bearing.
# `python setup_cython.py build_ext --inplace` writes `_fis_kernel` into
# `src/tribblefis/` (setuptools' src-layout auto-discovery resolves
# `package_dir` there; the generated `.c` lands beside the `.pyx`, as
# .gitignore records). A non-editable install imports the site-packages copy,
# which has no compiled extension in it, so `HAVE_CYTHON_KERNEL` goes False and
# the compiled CI leg quietly becomes a second copy of the numpy leg.

import importlib.util

# Loaded by file path under a unique module name, rather than defined here.
# There are two conftest.py files in this tree -- this one and
# tribble-tree/conftest.py -- and pytest's default `prepend` import mode imports
# both as the module `conftest`. `import conftest` from a test is therefore a
# coin flip decided by collection order: locally, a narrow selection never
# loaded the sibling and the guard's tests passed; in CI the full testpaths
# loaded both and all nine failed with "module 'conftest' has no attribute
# editable_install_problem". Nothing in the guard belongs in an ambiguous
# namespace, so it lives in tests/editable_guard.py and is loaded by path here
# -- which also avoids depending on `tests/` being on sys.path this early.
_GUARD_SPEC = importlib.util.spec_from_file_location(
    "tribblefis_editable_guard", Path(ROOT) / "tests" / "editable_guard.py"
)
editable_guard = importlib.util.module_from_spec(_GUARD_SPEC)
sys.modules[_GUARD_SPEC.name] = editable_guard
_GUARD_SPEC.loader.exec_module(editable_guard)


def pytest_configure(config):
    spec = importlib.util.find_spec("tribblefis")
    problem = editable_guard.editable_install_problem(
        spec.origin if spec is not None else None
    )
    if problem is not None:
        # `UsageError` stops the session before collection with the message and
        # no traceback. A traceback here would put the reader inside conftest.py,
        # which is the one place the problem is not.
        raise pytest.UsageError(problem)
