"""Unit tests for the editable-install guard in ``conftest.py``.

The guard itself runs once per session, at ``pytest_configure``, and either says
nothing or aborts the run. Neither outcome is a test result, so the decision
logic is factored out as :func:`conftest.editable_install_problem` and exercised
here against synthetic paths. Otherwise the only way to know the check works is
to break the venv on purpose, which is precisely the state it exists to prevent
anyone from being in.

Background is tribble-fis#214: the local `.venv` installed the project
non-editable, so `site-packages/tribblefis/` was a *copy* taken at the last
`uv sync`. Six tests failed locally with

    AttributeError: 'TribbleRegressor' object has no attribute 'tsk_order_'

while `grep` found the attribute in `src/` and CI was green. The obvious
readings -- "main is broken", "it's a Windows/3.13 break" -- were both wrong and
both expensive. The issue asked for the durable fix ("I want option 2, that way
it's always right") rather than a note telling the next person to re-run
`uv sync` and hope.
"""

import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import conftest  # noqa: E402

SRC = REPO_ROOT / "src" / "tribblefis"


def test_editable_checkout_is_accepted():
    """The normal state after `uv sync`: the package resolves inside `src/`."""
    assert conftest.editable_install_problem(SRC / "__init__.py") is None


def test_site_packages_copy_is_rejected():
    """#214's exact shape, and the message has to carry the diagnosis.

    The failure a stale copy produces is an `AttributeError` deep inside an
    estimator, which reads as a code bug. Whatever this guard prints is the only
    thing standing between the reader and an hour of chasing it, so the test
    asserts the message names the remedy rather than merely that it fired.
    """
    problem = conftest.editable_install_problem(
        Path("/somewhere/.venv/lib/python3.11/site-packages/tribblefis/__init__.py")
    )
    assert problem is not None
    assert "uv sync" in problem
    assert "#214" in problem
    assert "site-packages" in problem


def test_a_different_checkout_is_rejected():
    """A venv pointing at *another* clone is as stale as a copy, and quieter.

    This is the harder version of the same bug: everything looks editable, edits
    just land in a tree nobody is running. Only comparing against this repo's
    own `src/` catches it.
    """
    problem = conftest.editable_install_problem(
        Path("/home/dev/other-clone/src/tribblefis/__init__.py")
    )
    assert problem is not None
    assert "other-clone" in problem


def test_missing_package_is_reported_as_a_missing_install():
    """`find_spec` returning None means "not installed", not "not editable"."""
    problem = conftest.editable_install_problem(None)
    assert problem is not None
    assert "uv sync" in problem


def test_namespace_package_without_origin_is_reported():
    """A spec whose origin is None is a namespace package shadowing the real one.

    It happens when an empty `tribblefis/` directory exists somewhere on
    `sys.path` -- for instance a stray in-place build at the repo root. The
    import succeeds and the package is empty, which is a worse failure than not
    finding it at all.
    """
    assert conftest.editable_install_problem(None) is not None


def test_the_guard_runs_in_a_real_session():
    """End to end: a real pytest run must not trip the guard on a synced venv.

    The realistic way to break this check is for it to fire when it should not
    -- a false positive aborts the entire suite, so it is a far more expensive
    bug than the one it guards. Running a genuine (if tiny) session is the only
    way to exercise the `pytest_configure` path itself.
    """
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            str(Path(__file__).relative_to(REPO_ROOT)),
            "-k",
            "test_editable_checkout_is_accepted",
            "-p",
            "no:xdist",
            "-q",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "not installed in editable mode" not in result.stdout


def test_opt_out_is_honoured(monkeypatch):
    """Testing an installed wheel from a source checkout is legitimate.

    Without an escape hatch, the guard would make that impossible rather than
    merely loud, and someone would eventually delete the guard instead of the
    obstacle.
    """
    monkeypatch.setenv(conftest.ALLOW_INSTALLED_ENV, "1")
    assert conftest.editable_install_problem(Path("/anywhere/site-packages/tribblefis/__init__.py")) is None


@pytest.mark.parametrize("value", ["0", "", "false"])
def test_opt_out_requires_a_truthy_value(monkeypatch, value):
    """A leftover `TRIBBLEFIS_ALLOW_INSTALLED=0` must not silently disable it."""
    monkeypatch.setenv(conftest.ALLOW_INSTALLED_ENV, value)
    assert conftest.editable_install_problem(Path("/anywhere/site-packages/tribblefis/__init__.py")) is not None
