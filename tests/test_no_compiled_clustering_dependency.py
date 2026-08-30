"""`tribblefis` must not import the compiled `tribbleclustering` package.

Regression guard for tribble-fis#203. `gauss_math.py` carried

    from tribbleclustering import IVATMeans, FuzzyCMeans

from 8faee45, which added it together with three live call sites. All three
went away over the next two months without the import going with them:
963dbb6 deleted the `FuzzyCMeans(...)` call outright, f176f20 and a21aed1
commented out the two `IVATMeans(...)` calls, and dbc1a64 (#72) deleted those
two commented-out blocks. Because `gauss_math` sits under every functional
entry point in this package, the line that survived all five loaded
`tribbleclustering` -- whose `pcvat`, `cfcm` and `clk` are C extensions -- on
every import path, in exchange for nothing.

The check is deliberately "was it *imported*", not "is it installed". What
must not come back is tribblefis reaching for it on a path with no use for it.
Only `sys.modules` separates those two states, and it has to be read in a fresh
interpreter, since by the time this test body runs pytest has already imported
a great deal.

That distinction is also why this file skips when `tribbleclustering` is
absent. #231 moved the dependency out of `[project].dependencies` into the
`clustering` extra -- it was making a C toolchain a hard requirement of
`uv sync` on Windows in exchange for a package nothing here imports. A
`sys.modules` sweep in an environment without the package installed cannot
fail: nothing can leak a module that could not have been imported in the first
place. Running anyway would turn this guard into a permanent green check over
no coverage, which is the exact failure it was written to prevent one level
down. So it skips loudly instead, and `tests.yml` installs the extra on its
`compiled` leg so the guard runs for real somewhere.
"""

import importlib.util
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

# Scoped to the sweep, deliberately, and not applied to the module. The
# re-export check below needs no installed `tribbleclustering` to mean
# something: if the deleted import came back in an environment without the
# package, `gauss_math` would fail to import outright and that test would error
# rather than pass. Only the `sys.modules` sweep goes vacuous.
_needs_clustering_installed = pytest.mark.skipif(
    importlib.util.find_spec("tribbleclustering") is None,
    reason=(
        "tribbleclustering is not installed, so a sys.modules sweep cannot "
        "distinguish 'not imported' from 'not importable'. Install it with "
        "`uv sync --extra dev --extra clustering` (needs a C toolchain) to "
        "run this guard."
    ),
)

# Every module in the package is swept, rather than a hand-listed few, so the
# property this file is named for is the property it actually pins. A tuple of
# entry points would have left `one_class`, `refine`, `ruspini` and 21 others
# free to re-introduce the import -- and those are the modules most likely to
# reach for a clusterer.
#
# Discovery happens in the child, from `tribblefis.__path__`, so the list
# cannot drift as modules are added or renamed.
_CHILD = """
import importlib
import pkgutil
import sys

import tribblefis

names = sorted(m.name for m in pkgutil.iter_modules(tribblefis.__path__))
print("SWEPT " + str(len(names)))

# An unrelated ImportError (an optional dependency absent from this
# environment) must not turn this guard red -- but it is reported, because a
# module that could not be imported was also not checked.
skipped = []
for name in names:
    try:
        importlib.import_module("tribblefis." + name)
    except ImportError as exc:
        skipped.append(name + " (" + str(exc) + ")")

if skipped:
    print("NOT CHECKED: " + "; ".join(skipped))

leaked = sorted(m for m in sys.modules if m.split(".")[0] == "tribbleclustering")
if leaked:
    raise SystemExit(
        "importing the tribblefis package pulled in the compiled clustering "
        "package: " + ", ".join(leaked)
    )
"""

# A discovery failure would import nothing, find nothing in `sys.modules` and
# report success. The package had 32 modules when this was written; the floor
# only has to be high enough that an empty or truncated sweep cannot pass.
_MINIMUM_MODULES_SWEPT = 20


@_needs_clustering_installed
def test_no_module_in_the_package_imports_tribbleclustering():
    proc = subprocess.run(
        [sys.executable, "-c", _CHILD],
        capture_output=True,
        text=True,
    )
    output = proc.stdout + proc.stderr
    assert proc.returncode == 0, output

    swept = [
        int(line.split()[1])
        for line in proc.stdout.splitlines()
        if line.startswith("SWEPT ")
    ]
    assert swept, "the child never reported a module count:\n" + output
    assert swept[0] >= _MINIMUM_MODULES_SWEPT, (
        "only " + str(swept[0]) + " modules were swept, so this guard is not "
        "covering the package it claims to:\n" + output
    )


def test_gauss_math_does_not_re_export_the_clustering_estimators():
    """The deleted import made `gauss_math.IVATMeans` a working attribute.

    Nothing in this repository and nothing in the one known consumer (the
    `grad-school` dissertation repo, whose single user imports `IVATMeans`
    straight from `tribbleclustering`) reached the names that way, so the
    re-export was dropped rather than kept behind a `# noqa: F401`. Pinned here
    so that restoring it is a deliberate edit to this test, and not a side
    effect of someone re-adding a convenience import.
    """
    from tribblefis import gauss_math

    assert not hasattr(gauss_math, "IVATMeans")
    assert not hasattr(gauss_math, "FuzzyCMeans")


def test_tribble_clustering_is_not_a_default_install_dependency():
    """The compiled dependency stays off the default install path (#231).

    The guard above says nothing imports `tribbleclustering`. This one says the
    consequence has actually been taken: a package nothing imports does not
    belong in `[project].dependencies`, where it made a working C toolchain a
    precondition for `uv sync` on every platform. That cost was invisible on
    Linux (every runner has gcc) and a hard stop on Windows without the MSVC
    Build Tools, and `dependency-sync.yml` moves the pinned `rev` twice a day,
    so each bump lands on a revision no local cache has and pays the rebuild
    again.

    Pinned as a test because the natural way to reintroduce it is not malice:
    someone adds a clustering call site, reaches for the dependency, and puts
    it back in the obvious block. That is a fine change to make -- but it
    should be a deliberate one that comes here and says so, together with the
    guard above, rather than a line that slides back in and quietly restores
    the toolchain requirement for everyone.
    """
    pyproject = tomllib.loads(
        (Path(__file__).resolve().parent.parent / "pyproject.toml").read_text(
            encoding="utf-8"
        )
    )

    runtime = pyproject["project"]["dependencies"]
    offenders = [d for d in runtime if d.split()[0].split(">")[0].split("=")[0].strip() == "tribble-clustering"]
    assert not offenders, (
        "tribble-clustering is back in [project].dependencies "
        + str(offenders)
        + " -- that makes a C toolchain a hard requirement of `uv sync` on "
        "Windows for a package nothing in tribblefis imports. See #231."
    )

    # And it is still *available*, which is the other half: the point was to
    # move it, not to drop it. A downstream consumer that was getting it
    # transitively (grad-school builds its environment with
    # `uv sync --project tribble-fis --extra dev`) needs the extra to exist
    # under a stable name to add `--extra clustering` to that command.
    extra = pyproject["project"]["optional-dependencies"]["clustering"]
    assert any(
        d.split()[0].split(">")[0].split("=")[0].strip() == "tribble-clustering"
        for d in extra
    ), "the `clustering` extra no longer provides tribble-clustering"
