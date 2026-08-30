"""Decide whether the environment is running ``src/`` or a copy of it.

Called once per session from ``conftest.py``'s ``pytest_configure``. Lives here
rather than in ``conftest.py`` for a reason worth stating, because it is not
obvious and it cost a CI run:

**There are two ``conftest.py`` files in this tree** -- the repository root's
and ``tribble-tree/conftest.py`` -- and under pytest's default ``prepend``
import mode both are imported under the module name ``conftest``. Which one
``import conftest`` returns from a test is therefore a coin flip decided by
collection order. Locally, running only ``tests/test_editable_install.py`` never
loaded the sibling and the tests passed; in CI the full ``testpaths`` loaded
both and every test in that file failed with

    AttributeError: module 'conftest' has no attribute 'editable_install_problem'

A uniquely-named module has no such ambiguity.
``test_the_guard_does_not_live_in_a_conftest`` fails if this logic is ever moved
back.

The background is tribble-fis#214: a non-editable install puts a *copy* of the
package in ``site-packages``, taken at the last ``uv sync``. ``import
tribblefis`` then runs that copy while ``git status``, ``git log`` and the files
on disk all show current ``main``, and nothing warns you.

A note on the messages below, since they are the whole user interface here. This
fires exactly once, at a moment when the reader is already confused, and what it
prints is the only thing standing between them and an hour of chasing an
`AttributeError` in an estimator. So each failure shape gets the remedy that
actually applies to *it* -- an earlier version shared one remedy block across
both and told someone whose `PYTHONPATH` was shadowing the package to run
`uv sync`, which cannot reorder `sys.path` and would have returned them to the
identical error, now distrusting the message.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
EXPECTED_PACKAGE = REPO_ROOT / "src" / "tribblefis"

ALLOW_INSTALLED_ENV = "TRIBBLEFIS_ALLOW_INSTALLED"

_TRUTHY = ("1", "true", "yes")


def opted_out() -> bool:
    """Whether the caller has asked for the guard to stand down.

    Truthy-only on purpose. A leftover ``TRIBBLEFIS_ALLOW_INSTALLED=0`` in a
    shell profile reads as "off" to whoever wrote it, and treating any value as
    opt-in would silently disable the guard for that person forever.
    """
    return os.environ.get(ALLOW_INSTALLED_ENV, "").strip().lower() in _TRUTHY


def editable_install_problem(origin) -> str | None:
    """Describe what is wrong with where ``tribblefis`` resolves, or ``None``.

    Takes the resolved origin rather than looking it up, so the decision is
    testable without breaking a venv to reach it -- which is the state it exists
    to prevent anyone from being in. ``origin`` is ``ModuleSpec.origin``: the
    path to the package's ``__init__.py``, or ``None`` for a package that is
    missing or is an empty namespace package shadowing the real one.
    """
    if opted_out():
        return None

    # The interpreter, on every message. Every other path here names a
    # *package*; none of them answers "which python is this?", and running the
    # wrong one -- a second venv, a global interpreter, a shell still holding a
    # stale VIRTUAL_ENV -- is the most common way to arrive in this state.
    # Leaving it out makes the reader go and find it before they can act, which
    # is exactly the cost this guard exists to remove.
    where = "\n  Interpreter: " + sys.executable

    escape = (
        "\n  If you are deliberately testing an installed build, set "
        + ALLOW_INSTALLED_ENV
        + "=1.\n  See tribble-fis#214."
    )
    resync = f"\n  Run `uv sync` from {REPO_ROOT} to reinstall the project editable."

    if origin is None:
        return (
            "`tribblefis` is not installed in this environment (or is being "
            "shadowed by an empty namespace package of the same name)."
            + where
            + resync
            + escape
        )

    found = Path(origin).resolve().parent
    if found == EXPECTED_PACKAGE.resolve():
        return None

    # The two failure shapes read very differently to whoever hits them, and --
    # more importantly -- they have different fixes.
    if "site-packages" in found.parts or "dist-packages" in found.parts:
        # #214 verbatim: a copy taken at the last sync. `uv sync` is the fix.
        return (
            "`tribblefis` is not installed in editable mode: it resolves to a "
            f"copy at\n    {found}\n  taken at the last `uv sync`, not to\n"
            f"    {EXPECTED_PACKAGE}\n"
            "  Every test below would run that copy, and the files you edit "
            "would have no effect on the result."
            + where
            + resync
            + escape
        )

    # The quieter shape: everything looks editable, the edits just land in a
    # tree nobody is running. `uv sync` is *not* the fix -- it does not reorder
    # sys.path. Something is ahead of the editable install on it, so the message
    # names the three things that usually are, and prints PYTHONPATH rather than
    # making the reader go and look.
    pythonpath = os.environ.get("PYTHONPATH") or "unset"
    return (
        f"`tribblefis` resolves to a different checkout:\n    {found}\n"
        f"  rather than this one:\n    {EXPECTED_PACKAGE}\n"
        "  Edits made here would not affect the tests below."
        + where
        + f"\n  Something is ahead of the editable install on sys.path. Check "
        f"PYTHONPATH (currently: {pythonpath}), a .pth file in site-packages, "
        "or an editable install of another clone."
        + escape
    )
