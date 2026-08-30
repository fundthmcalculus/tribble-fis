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

A uniquely-named module has no such ambiguity. ``test_the_guard_does_not_live_in
_a_conftest`` fails if this logic is ever moved back.

The background is tribble-fis#214: a non-editable install puts a *copy* of the
package in ``site-packages``, taken at the last ``uv sync``. ``import
tribblefis`` then runs that copy while ``git status``, ``git log`` and the files
on disk all show current ``main``, and nothing warns you.
"""

from __future__ import annotations

import os
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

    remedy = (
        f"\n  Run `uv sync` from {REPO_ROOT} to reinstall the project editable.\n"
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
    if found == EXPECTED_PACKAGE.resolve():
        return None

    # The two failure shapes read very differently to whoever hits them, so they
    # get different first lines. A site-packages copy is #214 verbatim; a
    # different checkout is the quieter version, where everything looks editable
    # and the edits simply land in a tree nobody is running.
    if "site-packages" in found.parts or "dist-packages" in found.parts:
        headline = (
            f"`tribblefis` is not installed in editable mode: it resolves to a "
            f"copy at\n    {found}\n  taken at the last `uv sync`, not to\n"
            f"    {EXPECTED_PACKAGE}\n"
            f"  Every test below would run that copy, and the files you edit "
            f"would have no effect on the result."
        )
    else:
        headline = (
            f"`tribblefis` resolves to a different checkout:\n    {found}\n"
            f"  rather than this one:\n    {EXPECTED_PACKAGE}\n"
            f"  Edits made here would not affect the tests below."
        )
    return headline + remedy
