# Working conventions

- Always create a new branch and open a pull request for changes in this repository — never commit directly to `main`.
- Always commit and push your changes to that branch, then open the pull request — don't leave work uncommitted or unpushed, and don't just report a diff.
- After opening a pull request, watch it and drive CI to green: check status, diagnose and fix failures, and push follow-up commits until the pipeline passes. A PR with red CI is not done.

# Local development environment

`uv sync` installs the project **editable**, so `import tribblefis` runs `src/tribblefis/` directly and your edits take effect without re-syncing. Run `uv sync` after pulling to pick up dependency changes, not code changes.

If the venv ever ends up in some other state -- a non-editable copy under `site-packages/`, or a link to a different checkout -- `conftest.py` aborts the whole pytest session at startup and names the fix. That check exists because the alternative is silent: a stale copy fails with an `AttributeError` deep inside an estimator while `git status` and `git log` show current code, and both obvious readings ("main is broken", "it's a platform break") are wrong and expensive. See issue #214.

To test an installed build on purpose rather than the working tree, set `TRIBBLEFIS_ALLOW_INSTALLED=1`.

CI syncs fresh on every run and is authoritative when local and CI results disagree.

`uv sync` needs no C compiler. Keep it that way: `tribble-clustering` is a git
source that builds Cython extensions from scratch, and it sits in the
`clustering` extra rather than `[project].dependencies` precisely so a plain
sync does not require the MSVC Build Tools on Windows — nothing in `tribblefis`
imports it, and `tests/test_no_compiled_clustering_dependency.py` keeps both
halves of that true. `dependency-sync.yml` moves the pinned revision twice a
day, so anything on the default install path that compiles gets rebuilt from
source that often, per developer. If you need the clustering estimators, run
`uv sync --extra dev --extra clustering`. See issue #231.
