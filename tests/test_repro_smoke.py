"""PAPER-04 smoke gate: the one-command repro entrypoint wires cleanly.

`scripts/repro_eval.sh --smoke` dry-runs the eval -> dashboard -> phys_metrics
wiring without heavy compute, a GPU, or pulled Gilbreth artifacts (RESEARCH Test
Map PAPER-04). This test runs that smoke path via subprocess and asserts a clean
exit + the `SMOKE_OK` sentinel, proving argument parsing, repo-root path
resolution, and the chained command construction all hold.

It SKIPS gracefully (never fails) when the host lacks `bash` or the script is
absent, rather than reporting a false failure. It does NOT fabricate any metric
output — the smoke path computes no metrics, it only validates the wiring.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

# tests/ -> dynamic-fracture (repo root)
REPO_ROOT = Path(__file__).resolve().parents[1]
REPRO_SCRIPT = REPO_ROOT / "scripts" / "repro_eval.sh"


def _run_smoke(*extra_args: str) -> subprocess.CompletedProcess:
    bash = shutil.which("bash")
    if bash is None:
        pytest.skip("bash not available on PATH — cannot exercise repro_eval.sh")
    if not REPRO_SCRIPT.is_file():
        pytest.skip(f"repro_eval.sh not found at {REPRO_SCRIPT}")
    return subprocess.run(
        [bash, str(REPRO_SCRIPT), "--smoke", *extra_args],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=120,
    )


def test_repro_smoke_exits_clean():
    """`repro_eval.sh --smoke` exits 0 and prints the SMOKE_OK sentinel."""
    proc = _run_smoke()
    assert proc.returncode == 0, (
        f"repro_eval.sh --smoke exited {proc.returncode}\n"
        f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
    )
    assert "SMOKE_OK" in proc.stdout, (
        f"missing SMOKE_OK sentinel in smoke output:\n{proc.stdout}"
    )


def test_repro_smoke_reports_the_three_chained_steps():
    """The smoke wiring report names all three chained drivers (eval -> dashboard -> phys)."""
    proc = _run_smoke()
    out = proc.stdout
    assert "src.evaluate" in out, f"eval step missing from wiring report:\n{out}"
    assert "make_report.py" in out, f"dashboard step missing from wiring report:\n{out}"
    assert "make_phys_report.py" in out, f"phys step missing from wiring report:\n{out}"


def test_repro_smoke_honours_run_name():
    """`--run-name` flows through argument parsing into the wiring report."""
    proc = _run_smoke("--run-name", "tau_base")
    assert proc.returncode == 0, proc.stderr
    assert "tau_base" in proc.stdout, (
        f"--run-name not threaded into the smoke wiring:\n{proc.stdout}"
    )
