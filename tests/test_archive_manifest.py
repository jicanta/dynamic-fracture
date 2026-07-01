"""CONS-02 archive manifest generator (D-06/D-07) — self-describing provenance.

`scripts/build_archive_manifest.py` walks an archive tree, hashes every bundled
file, and emits `manifest.json` + `MANIFEST.md` + `PROVENANCE.md`:

  * manifest.json records one `{path, sha256(64-hex), size_bytes}` entry per
    bundled file, EXCLUDING exactly the 3 self-referential docs it generates
    (manifest.json / MANIFEST.md / PROVENANCE.md). `durable_checkpoints.json`
    IS hashed + included (it is tamper-checked).
  * top-level carries `archive_version="v1.0"`, an ISO8601 `generated_at`, the
    dynamic-fracture `code_commit_sha` (40-hex), a documented non-fabricated
    `dataset_sha` sentinel (begins `unavailable:`, NOT a 64-hex hash), and a
    `durable_checkpoints` array mirroring durable_checkpoints.json.
  * hashing is idempotent: re-running on an unchanged tree yields identical
    per-file sha256 values.
  * MANIFEST.md renders a markdown file-table with the 12-char DISPLAY digest;
    manifest.json stores the FULL 64-hex digest (T-08-04 no truncated trust).

The test builds a tiny SYNTHETIC archive tree in tmp so it never pollutes the
real archive/v1.0/, and drives the generator via subprocess (mirrors the
subprocess wiring in tests/test_repro_smoke.py).

Run:  cd dynamic-fracture && python -m pytest tests/test_archive_manifest.py -q
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

# tests/ -> dynamic-fracture (repo root)
REPO_ROOT = Path(__file__).resolve().parents[1]
BUILDER = REPO_ROOT / "scripts" / "build_archive_manifest.py"

SELF_DOCS = {"manifest.json", "MANIFEST.md", "PROVENANCE.md"}


def _make_synth_archive(root: Path) -> int:
    """Build a tiny archive tree; return the count of bundled files (incl. durable json)."""
    (root / "outputs").mkdir(parents=True)
    (root / "comparisons").mkdir(parents=True)
    (root / "outputs" / "seed_meanstd.csv").write_text("case,f1\na,0.5\n")
    (root / "outputs" / "probs.npz").write_bytes(b"\x00\x01\x02binary")
    (root / "comparisons" / "tau_headline_vs_base_regen.md").write_text("# cmp 0.5767\n")
    # durable pointers (mirrors the 08-02 schema) — IS hashed/included.
    durable = {
        "durable_checkpoints": [
            {
                "name": "tau_refined_best.pt",
                "durable_path": "/depot/aamp/data/x/tau_refined_best.pt",
                "sha256": "f1a0786c" + "0" * 56,
                "size_bytes": 165378331,
            }
        ]
    }
    (root / "durable_checkpoints.json").write_text(json.dumps(durable, indent=2))
    # 4 bundled files: seed_meanstd.csv, probs.npz, comparison md, durable json
    return 4


def _run_builder(archive_root: Path) -> subprocess.CompletedProcess:
    if not BUILDER.is_file():
        pytest.skip(f"builder not found at {BUILDER}")
    return subprocess.run(
        [sys.executable, str(BUILDER), "--archive-root", str(archive_root)],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=120,
    )


@pytest.fixture
def synth_archive(tmp_path):
    root = tmp_path / "v1.0"
    n_files = _make_synth_archive(root)
    return root, n_files


def test_builder_emits_three_docs(synth_archive):
    root, _ = synth_archive
    proc = _run_builder(root)
    assert proc.returncode == 0, f"builder exited {proc.returncode}\n{proc.stderr}"
    for name in SELF_DOCS:
        assert (root / name).is_file(), f"missing generated {name}"


def test_manifest_file_count_excludes_exactly_three_self_docs(synth_archive):
    root, n_files = synth_archive
    _run_builder(root)
    m = json.loads((root / "manifest.json").read_text())
    total = sum(1 for p in root.rglob("*") if p.is_file())
    assert len(m["files"]) == total - 3, (len(m["files"]), total)
    assert len(m["files"]) == n_files
    recorded = {f["path"] for f in m["files"]}
    assert SELF_DOCS.isdisjoint(recorded)
    assert "durable_checkpoints.json" in recorded  # included + tamper-checked


def test_manifest_files_have_full_sha_and_int_size(synth_archive):
    root, _ = synth_archive
    _run_builder(root)
    m = json.loads((root / "manifest.json").read_text())
    for f in m["files"]:
        assert re.fullmatch(r"[0-9a-f]{64}", f["sha256"]), f
        assert isinstance(f["size_bytes"], int) and f["size_bytes"] > 0, f


def test_manifest_toplevel_fields(synth_archive):
    root, _ = synth_archive
    _run_builder(root)
    m = json.loads((root / "manifest.json").read_text())
    assert m["archive_version"] == "v1.0"
    assert m.get("generated_at")
    assert re.fullmatch(r"[0-9a-f]{40}", m["code_commit_sha"]), m["code_commit_sha"]
    # documented sentinel, NOT a fabricated 64-hex hash, NOT empty
    assert m["dataset_sha"].startswith("unavailable")
    assert not re.fullmatch(r"[0-9a-f]{64}", m["dataset_sha"])
    # durable pointers mirrored verbatim
    assert m["durable_checkpoints"]
    assert m["durable_checkpoints"][0]["name"] == "tau_refined_best.pt"
    assert m["durable_checkpoints"][0]["sha256"].startswith("f1a0786c")


def test_hashing_is_idempotent(synth_archive):
    root, _ = synth_archive
    _run_builder(root)
    first = {f["path"]: f["sha256"] for f in json.loads((root / "manifest.json").read_text())["files"]}
    _run_builder(root)
    second = {f["path"]: f["sha256"] for f in json.loads((root / "manifest.json").read_text())["files"]}
    assert first == second


def test_manifest_md_table_and_durable_section(synth_archive):
    root, _ = synth_archive
    _run_builder(root)
    md = (root / "MANIFEST.md").read_text()
    assert "| path |" in md
    assert "Durable checkpoints" in md
    # full digest in json, 12-char display digest in md
    m = json.loads((root / "manifest.json").read_text())
    full = m["files"][0]["sha256"]
    assert full[:12] in md and full not in md


def test_provenance_documents_verify_and_dataset(synth_archive):
    root, _ = synth_archive
    _run_builder(root)
    prov = (root / "PROVENANCE.md").read_text()
    assert "assert_provenance.py --archive" in prov
    assert "Dataset provenance" in prov
    assert "missing/changed durable checkpoint" in prov
    assert "0.5767" in prov  # number -> source map (gate_provenance framing)


def test_builder_has_no_hardcoded_scratch_path():
    src = "\n".join(
        line for line in BUILDER.read_text().splitlines()
        if not line.lstrip().startswith("#")
    )
    assert "/scratch" not in src
