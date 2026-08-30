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
import os

ALLOW_INSTALLED_ENV = "TRIBBLEFIS_ALLOW_INSTALLED"

_EXPECTED_PACKAGE = Path(ROOT) / "src" / "tribblefis"


def editable_install_problem(origin):
    """Describe what is wrong with where ``tribblefis`` resolves, or ``None``.

    Takes the resolved origin rather than looking it up, so the decision is
    testable without breaking a venv to reach it -- see
    ``tests/test_editable_install.py``. ``origin`` is ``ModuleSpec.origin``:
    the path to the package's ``__init__.py``, or ``None`` for a package that
    is missing or is an empty namespace package shadowing the real one.
    """
    # Deliberately truthy-only. A leftover `TRIBBLEFIS_ALLOW_INSTALLED=0` in a
    # shell profile reads as "off" to whoever wrote it, and treating any value
    # as opt-in would silently disable the guard for them forever.
    if os.environ.get(ALLOW_INSTALLED_ENV, "").strip().lower() in ("1", "true", "yes"):
        return None

    remedy = (
        f"\n  Run `uv sync` from {ROOT} to reinstall the project editable.\n"
        f"  If you are deliberately testing an installed build, set "
        f"{ALLOW_INSTALLED_ENV}=1.\n"
        f"  See tribble-fis#214."
    )

    if origin is None:
        return (
            "`tribblefis` is not installed in this environment (or is being "
            "shadowed by an empty namespace package of the same name)." + remedy
        )

    found = Path(origin).resolve().parent
    if found == _EXPECTED_PACKAGE.resolve():
        return None

    # The two failure shapes read very differently to whoever hits them, so
    # they get different first lines. A site-packages copy is #214 verbatim; a
    # different checkout is the quieter version, where everything looks
    # editable and the edits simply land in a tree nobody is running.
    if "site-packages" in found.parts or "dist-packages" in found.parts:
        headline = (
            f"`tribblefis` is not installed in editable mode: it resolves to a "
            f"copy at\n    {found}\n  taken at the last `uv sync`, not to\n"
            f"    {_EXPECTED_PACKAGE}\n"
            f"  Every test below would run that copy, and the files you edit "
            f"would have no effect on the result."
        )
    else:
        headline = (
            f"`tribblefis` resolves to a different checkout:\n    {found}\n"
            f"  rather than this one:\n    {_EXPECTED_PACKAGE}\n"
            f"  Edits made here would not affect the tests below."
        )
    return headline + remedy


def pytest_configure(config):
    spec = importlib.util.find_spec("tribblefis")
    problem = editable_install_problem(spec.origin if spec is not None else None)
    if problem is not None:
        # `UsageError` stops the session before collection with the message and
        # no traceback. A traceback here would put the reader inside conftest.py,
        # which is the one place the problem is not.
        raise pytest.UsageError(problem)
