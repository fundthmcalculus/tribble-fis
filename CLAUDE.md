# Working conventions

- Always create a new branch and open a pull request for changes in this repository — never commit directly to `main`.
- Always commit and push your changes to that branch, then open the pull request — don't leave work uncommitted or unpushed, and don't just report a diff.
- After opening a pull request, watch it and drive CI to green: check status, diagnose and fix failures, and push follow-up commits until the pipeline passes. A PR with red CI is not done.

# Local development environment

The local `.venv` installs the project non-editable, meaning `site-packages/tribblefis/` is a copy taken at the last `uv sync`, not a link to `src/tribblefis/`. This means local test runs and imports will use stale code even after pulling changes, while `git status` and the files on disk show current code. If local test results don't match CI:

1. Run `uv sync` to refresh the venv's copy of the code.
2. Re-run your local tests or measurements.

CI syncs fresh on every run and is authoritative when local and CI results disagree.
