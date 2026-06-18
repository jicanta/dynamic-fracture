#!/usr/bin/env python3
"""Per-case late-rollout macro F1 mean±std over the 3 headline seeds (HPC-02 / D-05).

D-05 reports the headline FractureTAU result as a mean±std over 3 seeds {42,43,44},
not a single run: bf16 AMP + cuDNN are not byte-deterministic on A100 (Pitfall 9),
so the seed spread IS the honest run-to-run variation. This script globs the three
``OUTPUTS/headline_s{seed}/eval/<case>/per_frame_metrics.csv`` trees, computes each
case's late-rollout (last-20%) macro F1 PER SEED via the ONE shared D-03-aware metric
(``verdict.late_rollout_macro_f1`` — never a forked F1, Anti-Pattern "forking the
selection metric"), then aggregates across seeds with ``statistics.mean`` /
``statistics.pstdev`` -> ``(case, mean, std, n_seeds)``.

Provenance (D-14 / S5, threat T-04-02): every seed dir is tagged with
``sha256_file(<run>/checkpoints/best.pt)`` so each reported number traces to a
checkpoint. The per-case 3-seed mean list (``tau_percase``) is what
``significance.py`` pairs against the frozen ConvLSTM in the one-sided Wilcoxon
(HPC-03).

Framework-free: stdlib + the framework-free ``verdict`` metric only (NO torch, NO
tensorflow) — runs on a login node in either conda env.

Run from new_model/:
    python -m scripts.seed_aggregate \
        --runs OUTPUTS/headline_s42 OUTPUTS/headline_s43 OUTPUTS/headline_s44
"""
from __future__ import annotations

import argparse
import glob
import os
import statistics
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

# ---- repo-root + scripts + new_model sys.path shim (PATTERNS S1) ----
# scripts -> new_model -> dynamic-fracture; makes verdict (scripts) and src.utils
# (new_model) importable whether run as `-m scripts.seed_aggregate` or imported by
# tests with the scripts dir on the path.
SCRIPTS_DIR = Path(__file__).resolve().parent
NEW_MODEL = SCRIPTS_DIR.parent
REPO_ROOT = NEW_MODEL.parent
for _p in (str(REPO_ROOT), str(NEW_MODEL), str(SCRIPTS_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Reuse the ONE D-03-aware per-case scalar + CSV reader (do NOT re-implement F1).
from verdict import late_rollout_macro_f1, _read_rows  # noqa: E402

DEFAULT_SEEDS = (42, 43, 44)
EVAL_SUBDIR = "eval"
BEST_CKPT_SUBPATH = os.path.join("checkpoints", "best.pt")


def _sha256_file(path: str | Path) -> str:
    """Streaming SHA256 of a file (D-14 checkpoint provenance).

    Prefers the canonical ``src.utils.sha256_file``; ``utils.py`` imports torch,
    which is absent in the TF login env where this offline script may also run, so
    fall back to a byte-identical streaming sha256 (S1 tolerant-import idiom — the
    hash algorithm is fixed, this is provenance, not a forked metric).
    """
    try:
        from src.utils import sha256_file as _canonical  # noqa: E402
        return _canonical(path)
    except Exception:
        import hashlib
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()


def discover_seed_runs(root: str | Path) -> List[Path]:
    """Return sorted ``<root>/headline_s*`` dirs that contain an ``eval/`` subdir."""
    root = Path(root)
    return sorted(
        p for p in root.glob("headline_s*")
        if p.is_dir() and (p / EVAL_SUBDIR).is_dir()
    )


def _coerce_runs(runs) -> List[Path]:
    """Accept either a single root (auto-discover headline_s* seed dirs) or an
    explicit sequence of run dirs."""
    if isinstance(runs, (str, os.PathLike)):
        root = Path(runs)
        discovered = discover_seed_runs(root)
        if discovered:
            return discovered
        # A bare run dir that itself holds an eval/ subdir (single-seed call).
        if (root / EVAL_SUBDIR).is_dir():
            return [root]
        raise SystemExit(
            f"[seed] no headline_s* seed dirs (with an eval/ subdir) under {root}")
    run_dirs = [Path(r) for r in runs]
    if not run_dirs:
        raise SystemExit("[seed] no run dirs given")
    return run_dirs


def _case_csvs(run_dir: Path) -> Dict[str, str]:
    """Map case-name -> per_frame_metrics.csv path under ``<run_dir>/eval`` (reuse
    verdict.py's glob/keying)."""
    eval_dir = run_dir / EVAL_SUBDIR
    return {
        os.path.basename(os.path.dirname(p)): p
        for p in glob.glob(os.path.join(str(eval_dir), "**", "per_frame_metrics.csv"),
                           recursive=True)
    }


def per_case_meanstd(
    runs, frac: float = 0.20
) -> List[Tuple[str, float, float, int]]:
    """Per-case late-rollout macro F1 mean±std across the seed runs.

    ``runs`` may be a single root dir (auto-discovers ``headline_s*`` seed dirs) or
    an explicit sequence of run dirs. Returns ``[(case, mean, pstdev, n_seeds), ...]``
    sorted by case. A case must be present in EVERY seed (else ``SystemExit`` — a
    partial 3-seed mean is not an honest error bar). The per-seed scalar is
    ``verdict.late_rollout_macro_f1`` (no forked metric).
    """
    run_dirs = _coerce_runs(runs)
    per_seed = [_case_csvs(rd) for rd in run_dirs]
    if any(not m for m in per_seed):
        empty = [str(rd) for rd, m in zip(run_dirs, per_seed) if not m]
        raise SystemExit(f"[seed] no eval CSVs under: {empty}")

    common = set(per_seed[0])
    for m in per_seed[1:]:
        common &= set(m)
    if not common:
        raise SystemExit("[seed] no case is present across all seed dirs")
    missing = (set().union(*[set(m) for m in per_seed])) - common
    if missing:
        raise SystemExit(
            f"[seed] cases missing from at least one seed dir: {sorted(missing)} "
            f"(cannot form a {len(run_dirs)}-seed mean)")

    rows: List[Tuple[str, float, float, int]] = []
    for case in sorted(common):
        f1s = [late_rollout_macro_f1(_read_rows(m[case]), frac=frac) for m in per_seed]
        rows.append((case, statistics.mean(f1s), statistics.pstdev(f1s), len(f1s)))
    return rows


def tau_percase(runs, frac: float = 0.20) -> List[Tuple[str, float]]:
    """Per-case 3-seed MEAN late-rollout macro F1 -> ``[(case, mean), ...]``.

    This is the TAU side significance.py pairs against the frozen ConvLSTM (HPC-03).
    """
    return [(case, mean) for case, mean, _std, _n in per_case_meanstd(runs, frac=frac)]


def seed_shas(runs) -> Dict[str, str]:
    """Map each seed run-name -> ``sha256_file(<run>/checkpoints/best.pt)`` (D-14)."""
    run_dirs = _coerce_runs(runs)
    shas: Dict[str, str] = {}
    for rd in run_dirs:
        ckpt = rd / BEST_CKPT_SUBPATH
        if not ckpt.is_file():
            raise SystemExit(f"[seed] missing checkpoint for provenance: {ckpt}")
        shas[rd.name] = _sha256_file(ckpt)
    return shas


def write_meanstd_csv(
    rows: Sequence[Tuple[str, float, float, int]], out_path: str | Path
) -> Path:
    """Write the per-case (case, mean, std, n_seeds) table (rank_variants style)."""
    import csv
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["case", "late_rollout_f1_mean", "late_rollout_f1_std", "n_seeds"])
        for case, mean, std, n in rows:
            w.writerow([case, f"{mean:.6f}", f"{std:.6f}", n])
    return path


def main(argv: Sequence[str] | None = None) -> List[Tuple[str, float, float, int]]:
    ap = argparse.ArgumentParser(
        description="Per-case late-rollout macro F1 mean±std over 3 seeds (HPC-02/D-05).")
    ap.add_argument(
        "--runs", nargs="+", required=True,
        help="The 3 seed run dirs, e.g. OUTPUTS/headline_s42 OUTPUTS/headline_s43 "
             "OUTPUTS/headline_s44 (a single parent dir also works — headline_s* "
             "are auto-discovered).")
    ap.add_argument("--frac", type=float, default=0.20, help="late-rollout window")
    ap.add_argument(
        "--out", default=None,
        help="Output CSV path (default: <first-run-parent>/seed_meanstd.csv).")
    args = ap.parse_args(argv)

    runs = args.runs if len(args.runs) > 1 else args.runs[0]
    rows = per_case_meanstd(runs, frac=args.frac)
    shas = seed_shas(runs)

    out_path = (Path(args.out) if args.out
                else Path(args.runs[0]).resolve().parent / "seed_meanstd.csv")
    write_meanstd_csv(rows, out_path)

    print(f"[seed] wrote {out_path}  (n_seeds={rows[0][3]}, frac={args.frac})")
    print(f"[seed] {'case':30s} {'mean':>8s} {'std':>8s}")
    for case, mean, std, _n in rows:
        print(f"[seed] {case:30s} {mean:8.4f} {std:8.4f}")
    overall = statistics.mean(m for _, m, _, _ in rows)
    print(f"[seed] OVERALL per-case mean = {overall:.4f} over {len(rows)} cases")
    print("[seed] provenance (D-14 — sha256 best.pt per seed):")
    for name, sha in shas.items():
        print(f"[seed]   {name}: {sha}")
    return rows


if __name__ == "__main__":
    main()
