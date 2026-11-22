"""Guard: the default compiled-kernel build must pass ``-ffp-contract=off``.

Without it, GCC's default ``-ffp-contract=fast`` fuses the probability and
einstein folds into FMAs on toolchains whose default target has FMA, and the
compiled kernel stops matching the NumPy reference bit for bit (nightly #235,
fixed in #241).

The PR gate's compiled leg only catches a regression on a runner whose default
target reproduces the fusion, so this checks the build recipe directly. It is a
source-level check on purpose: building the extension to observe the flag would
need a compiler on every test run.
"""

import ast
from pathlib import Path

SETUP = Path(__file__).resolve().parent.parent / "setup_cython.py"


def _non_windows_compile_args() -> list[str]:
    """Return the string literals in the non-Windows ``compile_args`` expression."""
    tree = ast.parse(SETUP.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if not isinstance(node, ast.If):
            continue
        # The non-Windows arm is the ``else`` of ``if sys.platform == "win32"``.
        test_src = ast.unparse(node.test)
        if "win32" not in test_src or not node.orelse:
            continue
        for stmt in node.orelse:
            if isinstance(stmt, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == "compile_args" for t in stmt.targets
            ):
                return [
                    n.value
                    for n in ast.walk(stmt.value)
                    if isinstance(n, ast.Constant) and isinstance(n.value, str)
                ]
    raise AssertionError("could not find compile_args in the non-Windows branch of setup_cython.py")


def test_default_build_disables_fp_contraction():
    args = _non_windows_compile_args()
    assert "-ffp-contract=off" in args, (
        "the default compiled-kernel build must pass -ffp-contract=off; "
        "see tests/test_compiled_build_flags.py and issue #240/#235"
    )

