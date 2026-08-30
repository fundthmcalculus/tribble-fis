"""The git-pin sync mechanism must match the pin style actually in use.

Regression guard for tribble-fis#201, and specifically for the *second* time it
happened. The first time, `[tool.uv.sources]` carried no revision at all and
nothing re-resolved the branch; `.github/workflows/dependency-sync.yml` was
written to close that. The second time, #219 added explicit `rev = "..."` pins
-- a genuine improvement, since the intended commit becomes reviewable in a
diff -- and that alone disabled the workflow, because its entire mechanism was

    uv lock --upgrade-package optimizers --upgrade-package tribble-clustering

which only moves a *revision-less* source. With a `rev` present the lock already
agrees with the pin and the resolve is a fixed point. Measured on `ae0ef13`,
with optimizers 4 commits behind upstream and clustering 7 behind, that command
produced a zero-byte diff and the workflow reported "nothing to do".

Nobody noticed for two commits, because a sync workflow that finds no drift and
a sync workflow that *cannot* find drift print the same thing and both exit 0.

What follows is therefore not a test of string rewriting. It is a test that the
two halves of one decision -- how the pins are written, and how they are updated
-- still agree with each other. Break either half and this fails by name.
"""

import re
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import sync_git_pins as sgp  # noqa: E402

PYPROJECT = REPO_ROOT / "pyproject.toml"
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "dependency-sync.yml"

A = "a" * 40
B = "b" * 40


def _fake_resolver(mapping):
    """A `resolve_head` stand-in, so no test in this file touches the network.

    A scheduled workflow is exactly the thing whose tests must not need a
    remote: the failure this file guards was invisible *because* verifying it
    meant reaching two upstreams, so nobody verified it.
    """

    def resolve(url, branch):
        return mapping[url]

    return resolve


# --------------------------------------------------------------------------
# The guard that matters: pin style and update mechanism must agree.
# --------------------------------------------------------------------------


def test_every_git_source_is_rev_pinned_and_discoverable():
    """Discovery must see every git source that `pyproject.toml` declares.

    This is the load-bearing assertion. `discover_pins` is what the sync
    workflow uses to decide what to refresh, so a git source it does not return
    is a dependency that silently stops being tracked -- #201's exact shape.
    Comparing against a fresh `tomllib` parse rather than a hard-coded list of
    two names means adding a third git dependency is covered the day it lands,
    without anyone remembering to extend this test.
    """
    text = PYPROJECT.read_text(encoding="utf-8")
    declared = {
        name
        for name, source in tomllib.loads(text)["tool"]["uv"]["sources"].items()
        if isinstance(source, dict) and "git" in source
    }
    assert declared, "expected at least one git source in [tool.uv.sources]"

    pins, warnings = sgp.discover_pins(text)
    assert {p.name for p in pins} == declared, (
        f"these git sources are not refreshable by scripts/sync_git_pins.py: "
        f"{sorted(declared - {p.name for p in pins})}. Warnings: {warnings}. "
        f"Either give them a full-SHA `rev`, or update the sync workflow to "
        f"refresh them the way they are actually pinned."
    )
    assert warnings == [], f"unexpected pin-shape warnings: {warnings}"


def test_workflow_refreshes_pins_through_the_script():
    """`dependency-sync.yml` must actually run the refresher.

    The workflow can be green, scheduled, and completely inert -- that is what
    happened between #219 and #221. Nothing in a YAML file asserts that its
    steps do anything, so the assertion lives here.

    The negative half is the specific one: `uv lock --upgrade-package` on its
    own is the mechanism that #219 defeated. It may still appear (the script
    rewrites the `rev`, and a lock refresh has to follow), but it must not be
    the *only* thing standing between the workflow and an upstream commit.
    """
    yaml = WORKFLOW.read_text(encoding="utf-8")
    assert "scripts/sync_git_pins.py" in yaml, (
        "dependency-sync.yml no longer invokes scripts/sync_git_pins.py. With "
        "`rev`-pinned sources, `uv lock --upgrade-package` alone cannot move a "
        "pin, and the workflow becomes a scheduled no-op that reports success."
    )


def test_workflow_runs_more_than_once_a_day():
    """The schedule must fire at least twice daily.

    Requested on #201: "runs a couple times a day, checks if there is a SHA
    change, and opens a PR for it." A single daily cron means an upstream fix
    merged just after the run waits nearly 24 hours, which for the seeding and
    determinism fixes that motivated #201 is a full day of downstream runs on
    known-bad code.
    """
    yaml = WORKFLOW.read_text(encoding="utf-8")
    crons = re.findall(r"cron:\s*['\"]([^'\"]+)['\"]", yaml)
    assert crons, "dependency-sync.yml has no cron schedule"
    hour_field = crons[0].split()[1]
    n_runs = len([h for h in hour_field.split(",") if h])
    assert n_runs >= 2, (
        f"cron {crons[0]!r} fires {n_runs}x/day; #201 asks for a couple times a day"
    )


# --------------------------------------------------------------------------
# Discovery
# --------------------------------------------------------------------------


def test_revless_git_source_warns_instead_of_being_skipped_silently():
    """A git source with no `rev` is #201's original shape and must be reported.

    Silence here is the whole bug: a source that drops out of the refresh set
    without a word is indistinguishable from one that is up to date.
    """
    pins, warnings = sgp.discover_pins(
        '[tool.uv.sources]\nfoo = { git = "https://example.com/foo.git" }\n'
    )
    assert pins == []
    assert len(warnings) == 1
    assert "no revision selector" in warnings[0]
    assert "#201" in warnings[0]


@pytest.mark.parametrize("selector", ["branch", "tag"])
def test_branch_and_tag_pins_are_left_alone(selector):
    """`branch` and `tag` are not this script's business, and it says so.

    A `branch` source re-resolves under `uv lock --upgrade-package` already; a
    `tag` is meant to be immutable. Rewriting either into a floating `rev` would
    be changing a caller's intent, not refreshing it.
    """
    pins, warnings = sgp.discover_pins(
        f'[tool.uv.sources]\nfoo = {{ git = "https://example.com/foo.git", '
        f'{selector} = "v1" }}\n'
    )
    assert pins == []
    assert selector in warnings[0]


def test_short_rev_is_reported_rather_than_compared():
    """An abbreviated `rev` would otherwise read as drifted forever.

    `git ls-remote` returns full SHAs, so a 7-character pin never compares equal
    to its own head and the workflow would open an identical PR twice a day.
    """
    pins, warnings = sgp.discover_pins(
        '[tool.uv.sources]\nfoo = { git = "https://example.com/foo.git", rev = "abc1234" }\n'
    )
    assert pins == []
    assert "40-character" in warnings[0]


def test_non_git_sources_are_ignored_without_comment():
    """A path or workspace source is not drift and must not produce noise."""
    pins, warnings = sgp.discover_pins(
        '[tool.uv.sources]\nfoo = { path = "../foo" }\nbar = { workspace = true }\n'
    )
    assert (pins, warnings) == ([], [])


def test_sync_branch_override_is_honoured():
    """A dependency whose default branch is not `main` can say so in place."""
    pins, _ = sgp.discover_pins(
        f'[tool.uv.sources]\nfoo = {{ git = "https://example.com/foo.git", '
        f'rev = "{A}", x-sync-branch = "master" }}\n'
    )
    assert [(p.name, p.branch) for p in pins] == [("foo", "master")]


# --------------------------------------------------------------------------
# Rewriting
# --------------------------------------------------------------------------


def test_rewrite_preserves_comments_and_touches_only_the_pin():
    """The comments above these sources are the reason `rev` pinning is good.

    They carry the `3a57f91` determinism floor and the `-march=native` SIGILL
    history. A refresh that reflowed the table -- which every TOML *writer*
    does, since `tomllib` is read-only -- would delete exactly the context a
    reviewer needs to judge the bump. So the rewrite is a substitution, and this
    test pins that: one line differs, and it is the pin's.
    """
    text = (
        "[tool.uv.sources]\n"
        "# Do not roll back before 3a57f91 -- seeding fix.\n"
        f'foo = {{ git = "https://example.com/foo.git", rev = "{A}" }}\n'
        "# unrelated trailing comment\n"
    )
    out = sgp.rewrite_rev(text, sgp.discover_pins(text)[0][0], B)

    before, after = text.splitlines(), out.splitlines()
    differing = [i for i, (b, a) in enumerate(zip(before, after)) if b != a]
    assert len(differing) == 1, f"expected one changed line, got {differing}"
    assert "3a57f91" in out and "unrelated trailing comment" in out
    assert B in out and A not in out


def test_rewrite_refuses_when_the_sha_also_appears_in_prose():
    """An ambiguous rewrite must fail loudly rather than edit a comment.

    If a comment cites the very revision being replaced, `str.replace` would
    rewrite the sentence too, leaving prose that explains a commit it no longer
    names. A failed scheduled run is recoverable; a silently falsified comment
    survives review because it still reads correctly.
    """
    text = (
        "[tool.uv.sources]\n"
        f"# Chosen because {A} is the first revision with the fix.\n"
        f'foo = {{ git = "https://example.com/foo.git", rev = "{A}" }}\n'
    )
    pin = sgp.discover_pins(text)[0][0]
    with pytest.raises(ValueError, match="exactly one occurrence"):
        sgp.rewrite_rev(text, pin, B)


def test_rewrite_to_the_same_rev_is_a_no_op():
    text = f'[tool.uv.sources]\nfoo = {{ git = "https://example.com/foo.git", rev = "{A}" }}\n'
    pin = sgp.discover_pins(text)[0][0]
    assert sgp.rewrite_rev(text, pin, A) == text


def test_rewrite_rejects_a_non_sha_replacement():
    text = f'[tool.uv.sources]\nfoo = {{ git = "https://example.com/foo.git", rev = "{A}" }}\n'
    pin = sgp.discover_pins(text)[0][0]
    with pytest.raises(ValueError, match="40-character SHA"):
        sgp.rewrite_rev(text, pin, "main")


# --------------------------------------------------------------------------
# End to end, against a temporary copy
# --------------------------------------------------------------------------


def _tmp_pyproject(tmp_path, rev):
    p = tmp_path / "pyproject.toml"
    p.write_text(
        "[tool.uv.sources]\n"
        "# keep me\n"
        f'foo = {{ git = "https://example.com/foo.git", rev = "{rev}" }}\n',
        encoding="utf-8",
    )
    return p


def test_sync_writes_when_upstream_moved(tmp_path):
    p = _tmp_pyproject(tmp_path, A)
    n, changes, warnings = sgp.sync(
        p, resolver=_fake_resolver({"https://example.com/foo.git": B})
    )
    assert (n, warnings) == (1, [])
    assert changes[0]["old"] == A and changes[0]["new"] == B
    assert changes[0]["compare"] == f"https://example.com/foo/compare/{A}...{B}"
    assert B in p.read_text(encoding="utf-8")
    assert "# keep me" in p.read_text(encoding="utf-8")


def test_check_mode_detects_without_writing(tmp_path):
    """`--check` and the real bump must share one code path.

    A separate detector is free to disagree with the applier, and "the gate says
    clean, the bump says stale" is a worse position than having no gate. Same
    function, one flag.
    """
    p = _tmp_pyproject(tmp_path, A)
    n, changes, _ = sgp.sync(
        p, check=True, resolver=_fake_resolver({"https://example.com/foo.git": B})
    )
    assert n == 1 and changes[0]["new"] == B
    assert A in p.read_text(encoding="utf-8"), "--check must not write"


def test_sync_is_idempotent_when_current(tmp_path):
    """Two runs a day must not open two identical PRs."""
    p = _tmp_pyproject(tmp_path, A)
    before = p.read_text(encoding="utf-8")
    n, changes, _ = sgp.sync(
        p, resolver=_fake_resolver({"https://example.com/foo.git": A})
    )
    assert (n, changes) == (0, [])
    assert p.read_text(encoding="utf-8") == before


def test_resolve_head_names_the_branch_when_the_ref_is_missing():
    """An empty ls-remote means "no such branch", not "network down".

    Those send a reader to entirely different places, and the default-branch
    rename that produces this is a real and recurring event.
    """

    class _Result:
        returncode, stdout, stderr = 0, "", ""

    with pytest.raises(RuntimeError, match="has no branch 'main'"):
        sgp.resolve_head(
            "https://example.com/foo.git", "main", runner=lambda *a, **k: _Result()
        )


def test_cli_reports_current_pins_without_network(tmp_path, monkeypatch):
    """`--check` exits 0 when everything is current, so it is safe as a gate."""
    p = _tmp_pyproject(tmp_path, A)
    monkeypatch.setattr(sgp, "PYPROJECT", p)
    monkeypatch.setattr(sgp, "resolve_head", lambda url, branch: A)
    assert sgp.main(["--check"]) == 0


def test_cli_exit_code_flags_stale_pins(tmp_path, monkeypatch):
    p = _tmp_pyproject(tmp_path, A)
    monkeypatch.setattr(sgp, "PYPROJECT", p)
    monkeypatch.setattr(sgp, "resolve_head", lambda url, branch: B)
    assert sgp.main(["--check"]) == 1


def test_script_runs_as_a_subprocess():
    """The workflow calls this by path, so it must work with no package install.

    `--help` is enough to catch the realistic breakages -- an import of
    `tribblefis` creeping in, or a syntax error -- without reaching a remote.
    """
    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "sync_git_pins.py"), "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
