"""Refresh the ``rev`` of every git-sourced dependency in ``pyproject.toml``.

Why this exists
---------------

``[tool.uv.sources]`` declares two dependencies from git. Issue #201 is about
those pins going stale without anything noticing, and the repository has now
been on *both* sides of that problem:

1. **Originally**, the sources carried no ``rev`` at all. ``uv.lock`` pinned a
   commit, nothing re-resolved it, and ``uv lock --check`` passes in that state
   however old the locked commit is. ``dependency-sync.yml`` was written to fix
   this, and its whole mechanism was ``uv lock --upgrade-package``, which
   re-resolves a revision-less git source to its branch head.

2. **Then #219 pinned explicit revisions** (``rev = "4d81121..."``), so the
   intended commit is visible in a diff where a reviewer already looks. That is
   a real improvement, and it silently disabled the workflow. With a ``rev``
   pinned there is nothing for ``--upgrade-package`` to upgrade *to*: the lock
   already agrees with the pin, so the re-resolve is a fixed point. Measured on
   ``ae0ef13``, with optimizers 4 commits ahead and clustering 7 ahead::

       $ uv lock --upgrade-package optimizers --upgrade-package tribble-clustering
       Resolved 41 packages in 268ms
       $ git diff --stat -- uv.lock
       (nothing)

   Every scheduled run from #219 to #221 printed "uv.lock already tracks both
   upstream heads -- nothing to do." That was false, and #221 had to be typed by
   hand to do what the workflow was supposed to do on its own.

The lesson is not "pins were a mistake". It is that **the pin style and the
update mechanism are one decision, and #219 changed half of it.** So this
script owns the update half explicitly: it reads the pin style out of
``pyproject.toml`` rather than assuming one, resolves each upstream branch head,
and rewrites the ``rev`` in place. ``uv lock`` afterwards then has something to
move to. ``tests/test_sync_git_pins.py`` fails if the two halves ever drift
apart again.

Usage
-----

::

    python scripts/sync_git_pins.py --check     # exit 1 if any pin is stale, write nothing
    python scripts/sync_git_pins.py             # rewrite stale pins in place
    python scripts/sync_git_pins.py --json      # machine-readable report on stdout

The rewrite is a targeted text substitution, not a TOML round-trip: ``tomllib``
is read-only, and every TOML *writer* available would reflow the file and
discard the comment blocks above each source -- which is where the reasoning for
the ``optimizers`` determinism floor and the ``tribble-clustering`` PyPI
override lives. Those comments are the most valuable content in that table, so
the file is parsed for discovery and edited by substitution.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PYPROJECT = REPO_ROOT / "pyproject.toml"

# uv accepts `branch`, `tag` and `rev` as the revision selector on a git source,
# and only `rev` is this script's business: `tag` is meant to be immutable, and
# `branch` re-resolves on its own, which is the pre-#219 state this script
# exists to replace. Either is reported and skipped rather than silently
# rewritten into a `rev` -- converting a caller's `tag` pin into a floating rev
# is a decision, not a refresh.

# The branch a bare `rev` pin is understood to track. uv has no field for
# "which branch this rev came from" -- `branch` and `rev` are alternatives, not
# companions -- so this is a convention, and making it explicit here is better
# than burying it in a shell one-liner. Override per-source with the
# `x-sync-branch` key described in `_branch_for`.
DEFAULT_BRANCH = "main"

_SHA_RE = re.compile(r"\A[0-9a-f]{40}\Z")


@dataclass(frozen=True)
class GitPin:
    """One rev-pinned git dependency discovered in ``[tool.uv.sources]``."""

    name: str
    url: str
    rev: str
    branch: str

    @property
    def short(self) -> str:
        return self.rev[:7]

    def compare_url(self, new_rev: str) -> str:
        """A GitHub compare link between the old and new revision.

        Built by trimming the ``.git`` suffix rather than by pattern-matching a
        host, so a non-GitHub remote yields a harmless dead link in a PR body
        instead of an exception in the middle of a sync run.
        """
        base = self.url.removesuffix(".git")
        return f"{base}/compare/{self.rev}...{new_rev}"


def _branch_for(name: str, source: dict) -> str:
    """The branch this source's ``rev`` is taken to track.

    Reads the optional ``x-sync-branch`` key so a dependency that lives on
    something other than ``main`` can say so next to its pin. uv ignores unknown
    keys in a source table, so this costs nothing at resolve time.
    """
    branch = source.get("x-sync-branch", DEFAULT_BRANCH)
    if not isinstance(branch, str) or not branch:
        raise ValueError(
            f"{name}: x-sync-branch must be a non-empty string, got {branch!r}"
        )
    return branch


def discover_pins(pyproject_text: str) -> tuple[list[GitPin], list[str]]:
    """Find the rev-pinned git sources, and note the ones that need attention.

    Returns ``(pins, warnings)``. A git source *without* a rev selector is a
    warning rather than an error: it is the pre-#219 shape, which is legitimate
    but is refreshed by ``uv lock --upgrade-package`` instead of by this script.
    Reporting it keeps a half-migrated table visible rather than letting one
    source fall out of the sync silently -- exactly the failure mode of #201.
    """
    data = tomllib.loads(pyproject_text)
    sources = data.get("tool", {}).get("uv", {}).get("sources", {})

    pins: list[GitPin] = []
    warnings: list[str] = []
    for name, source in sorted(sources.items()):
        if not isinstance(source, dict) or "git" not in source:
            continue
        if "rev" not in source:
            other = next((k for k in ("branch", "tag") if k in source), None)
            if other is not None:
                warnings.append(
                    f"{name}: pinned by `{other}`, not `rev` -- left alone. "
                    f"A `branch` source re-resolves under `uv lock "
                    f"--upgrade-package`; a `tag` is meant to be immutable."
                )
            else:
                warnings.append(
                    f"{name}: git source with no revision selector. uv will pin "
                    f"whatever the branch head was at first resolve and never "
                    f"move it again -- this is issue #201's original shape. Add "
                    f"a `rev` so this script can refresh it."
                )
            continue
        rev = source["rev"]
        if not isinstance(rev, str) or not _SHA_RE.match(rev):
            # A short rev resolves fine for uv but cannot be located
            # unambiguously in the file, and `git ls-remote` gives full SHAs, so
            # a comparison against a short pin would report drift forever.
            warnings.append(
                f"{name}: rev {rev!r} is not a full 40-character SHA -- left "
                f"alone. Expand it so drift can be detected."
            )
            continue
        pins.append(
            GitPin(name=name, url=source["git"], rev=rev, branch=_branch_for(name, source))
        )
    return pins, warnings


def resolve_head(url: str, branch: str, *, runner=subprocess.run) -> str:
    """The commit at the head of ``branch`` on the remote at ``url``.

    ``git ls-remote`` is used rather than a clone: this needs 40 bytes, and the
    two upstreams here carry compiled extensions and test corpora that make a
    clone orders of magnitude more expensive for the same answer.

    **This does not, and cannot, tell you the head is *ahead* of the current
    pin.** ls-remote returns a SHA and nothing else; ancestry needs history.
    That matters here, because the comment above the ``optimizers`` pin names a
    floor -- revisions before ``3a57f91`` ignore their ``seed`` and return a
    different model on every call -- and a force-push of upstream ``main``
    backwards past it would be installed by this script without complaint. The
    resulting PR would be green, because what it reintroduces is
    irreproducibility rather than a crash.

    What guards that is the shape of the workflow rather than anything in this
    file: **it opens a pull request, it does not merge one.** The PR body
    carries a ``compare/<old>...<new>`` link, on which a backwards move renders
    as "0 commits ahead", and the floor comment appears in the diff two lines
    above the changed pin. If this workflow ever grows an auto-merge, that
    review step is the thing being removed, and the ancestry check it stands in
    for has to be built first.
    """
    result = runner(
        ["git", "ls-remote", url, f"refs/heads/{branch}"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"git ls-remote failed for {url} ({branch}): "
            f"{result.stderr.strip() or 'no stderr'}"
        )
    line = result.stdout.strip()
    if not line:
        # An empty (rather than failing) ls-remote means the remote is reachable
        # but has no such branch -- typically a repo that renamed its default
        # branch. Saying so beats "could not resolve", which reads like a
        # network problem and sends the reader to the wrong place.
        raise RuntimeError(
            f"{url} has no branch {branch!r}. If its default branch was "
            f"renamed, set `x-sync-branch` on that source in pyproject.toml."
        )
    sha = line.split()[0]
    if not _SHA_RE.match(sha):
        raise RuntimeError(f"git ls-remote returned an unexpected ref {sha!r} for {url}")
    return sha


def rewrite_rev(text: str, pin: GitPin, new_rev: str) -> str:
    """Replace ``pin``'s revision with ``new_rev``, touching nothing else.

    The old SHA is matched literally and must occur *exactly once* in the file
    -- the ``rev = "..."`` itself. Asserting the count rather than reaching
    straight for ``str.replace`` is the point of this function: the comment
    blocks above these sources cite specific revisions (the ``3a57f91``
    determinism floor, for one), so a blind replace could rewrite prose as well
    as the pin, leaving a comment that explains a revision it no longer names.
    Refusing an ambiguous rewrite fails a scheduled run loudly, which is
    recoverable; a quietly corrupted comment is not.
    """
    if new_rev == pin.rev:
        return text
    if not _SHA_RE.match(new_rev):
        raise ValueError(f"{new_rev!r} is not a full 40-character SHA")
    occurrences = text.count(pin.rev)
    if occurrences != 1:
        raise ValueError(
            f"{pin.name}: expected exactly one occurrence of {pin.rev} in "
            f"pyproject.toml, found {occurrences}. Refusing to rewrite -- "
            f"resolve the ambiguity by hand."
        )
    return text.replace(pin.rev, new_rev)


def sync(
    pyproject: Path | None = None,
    *,
    check: bool = False,
    resolver=None,
) -> tuple[int, list[dict], list[str]]:
    """Refresh every rev pin. Returns ``(n_stale, changes, warnings)``.

    ``check=True`` resolves and reports but writes nothing, so the same code
    path backs both the CI gate and the actual bump. A separate "detect" routine
    would be free to disagree with the "apply" routine, which is the shape of
    bug this whole issue is about.

    Both defaults resolve at *call* time rather than being bound in the
    signature. Written the obvious way (``resolver=resolve_head``) the default
    captures the function object at import, so a test that patches
    ``sync_git_pins.resolve_head`` is silently ignored and the "offline" test
    quietly reaches the network instead -- which is how the first draft of
    ``tests/test_sync_git_pins.py`` had one green test that was not testing
    anything.
    """
    pyproject = PYPROJECT if pyproject is None else pyproject
    resolver = resolve_head if resolver is None else resolver
    text = pyproject.read_text(encoding="utf-8")
    pins, warnings = discover_pins(text)

    changes: list[dict] = []
    updated = text
    for pin in pins:
        head = resolver(pin.url, pin.branch)
        if head == pin.rev:
            continue
        updated = rewrite_rev(updated, pin, head)
        changes.append(
            {
                "name": pin.name,
                "url": pin.url,
                "branch": pin.branch,
                "old": pin.rev,
                "new": head,
                "compare": pin.compare_url(head),
            }
        )

    if changes and not check:
        pyproject.write_text(updated, encoding="utf-8")
    return len(changes), changes, warnings


def _render_markdown(changes: list[dict]) -> str:
    rows = "\n".join(
        f"| `{c['name']}` | `{c['old'][:7]}` | [`{c['new'][:7]}`]({c['compare']}) |"
        for c in changes
    )
    return "| Dependency | Old | New |\n|---|---|---|\n" + rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--check",
        action="store_true",
        help="report stale pins and exit 1 without writing.",
    )
    parser.add_argument(
        "--json", action="store_true", help="emit the report as JSON on stdout."
    )
    parser.add_argument(
        "--markdown-table",
        metavar="PATH",
        help="write a markdown table of the changes to PATH (for a PR body).",
    )
    args = parser.parse_args(argv)

    try:
        n_stale, changes, warnings = sync(check=args.check)
    except (RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps({"changes": changes, "warnings": warnings}, indent=2))
    else:
        for warning in warnings:
            # `::warning::` renders in the Actions log and the job summary; it is
            # inert prefix text when this is run at a normal shell.
            print(f"::warning::{warning}", file=sys.stderr)
        if not changes:
            print("All git pins are at their upstream branch head.")
        for c in changes:
            verb = "is stale" if args.check else "->"
            print(f"{c['name']}: {c['old'][:7]} {verb} {c['new'][:7]} ({c['branch']})")

    if args.markdown_table and changes:
        Path(args.markdown_table).write_text(_render_markdown(changes), encoding="utf-8")

    return 1 if (args.check and n_stale) else 0


if __name__ == "__main__":
    raise SystemExit(main())
