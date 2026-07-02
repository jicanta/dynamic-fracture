#!/usr/bin/env python3
"""Reproducible assembler for the NEW-CASE head-to-head (Phase-9, plan 09-08 / D-04).

This script COMPUTES (never hand-types) the out-of-distribution new-case comparison
between the frozen FractureTAU (3 seeds ``headline_s{42,43,44}``) and the frozen
ConvLSTM reference (``base_regen_newcase``), then emits the two plan artifacts:

    results/newcase/newcase_seed_meanstd.csv     (one row per new case, 3 rows)
    new_model/OUTPUTS/newcase_comparison.md      (per-case + aggregate + win/loss
                                                  + Wilcoxon(n=3, indicative) + SHA
                                                  provenance block)

CORRECTNESS OVER CONVENIENCE (standing requirement): every reported number is
computed here from the ACTUAL ``per_frame_metrics.csv`` files and the ACTUAL
``best.pt`` sha256 — NO placeholder numbers, NO numbers baked into this file. The
v1.0 headline chain is REUSED verbatim so the numbers match it exactly:

  * per-case 3-seed mean±std  ->  seed_aggregate.per_case_meanstd (frac=0.20)
  * per-case CNN scalars      ->  significance._cnn_percase        (frac=0.20)
  * exact-count pairing       ->  significance.pair_cases(expect_cases=3)
  * one-sided Wilcoxon        ->  significance.wilcoxon_tau_gt_cnn
  * checkpoint provenance     ->  seed_aggregate.seed_shas (sha256 best.pt)

The ONE D-03-aware ``late_rollout_macro_f1`` metric (frac=0.20) is NOT re-implemented
here — it flows in through seed_aggregate / significance (no forked selection metric,
Anti-Pattern "forking the selection metric").

n=3 HONESTY (D-06): a one-sided Wilcoxon on 3 paired cases has a minimum one-sided p
of 0.125, so it can NEVER reach 0.05. The artifact therefore HEADLINES the per-case +
mean±std results and reports the Wilcoxon p as INDICATIVE, not a significance claim.

The v1.0 in-distribution headline (0.8621 vs 0.5767) is kept OUT of this artifact
entirely (D-03): these are ADDITIVE OOD numbers, not a merge.

RUN (on Gilbreth, where the 12 eval CSVs + 3 best.pt checkpoints live) from new_model/:

    python -m scripts.make_newcase_comparison

Equivalent (from the repo root ``dynamic-fracture/``):

    python new_model/scripts/make_newcase_comparison.py

All input/output paths default to REPO_ROOT-relative locations and are resolved from
this file's location, so the script is cwd-independent (either invocation works).
Fails loud (SystemExit) if any of the 12 per_frame_metrics.csv (3 seeds x 3 cases +
3 CNN cases) or the 3 best.pt checkpoints are missing, or if the paired case count
!= EXPECTED_NEW_COUNT (3).

Provenance gate (run AFTER this script writes the artifact):
    python scripts/assert_provenance.py --check-artifact new_model/OUTPUTS/newcase_comparison.md
"""
from __future__ import annotations

import argparse
import datetime as _dt
import glob
import hashlib
import os
import re
import statistics
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

# ---- repo-root + new_model + scripts sys.path shim (PATTERNS S1) ----
# scripts -> new_model -> dynamic-fracture (repo root). Makes the reused chain
# (seed_aggregate, significance -> verdict; case_registry at the repo root)
# importable whether run as `-m scripts.make_newcase_comparison` from new_model/
# or as `python new_model/scripts/make_newcase_comparison.py` from the repo root.
SCRIPTS_DIR = Path(__file__).resolve().parent
NEW_MODEL = SCRIPTS_DIR.parent
REPO_ROOT = NEW_MODEL.parent
for _p in (str(REPO_ROOT), str(NEW_MODEL), str(SCRIPTS_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Reuse the EXISTING v1.0 comparison chain verbatim (no forked metric, no fork of
# late_rollout_macro_f1 — it flows in through these two modules).
import seed_aggregate  # noqa: E402  (per_case_meanstd, seed_shas, write_meanstd_csv)
import significance  # noqa: E402   (_cnn_percase, pair_cases, wilcoxon_tau_gt_cnn)
from case_registry import (  # noqa: E402
    EXPECTED_NEW_COUNT,
    NEW_TEST_CASE_FOLDERS,
)

FRAC = 0.20
SEEDS = (42, 43, 44)
EVAL_SUBDIR = "eval_newcase"

# The exact regex scripts/assert_provenance.py --check-artifact greps for the
# provenance-block best.pt lines (kept in sync here for the local self-test only).
PROVENANCE_RE = r"([\w./\-]*best\.pt)\s*[:=]?\s*(?:sha256[:=])?\s*([0-9a-f]{64})"

# ---- default REPO_ROOT-relative locations (cwd-independent) ----
DEFAULT_TAU_ROOT = NEW_MODEL / "OUTPUTS"
DEFAULT_CNN = REPO_ROOT / "kathleens-model" / "OUTPUTS" / "base_regen_newcase"
DEFAULT_DATA_ROOT = NEW_MODEL / "DATASET"
DEFAULT_SEED_CSV = REPO_ROOT / "results" / "newcase" / "newcase_seed_meanstd.csv"
DEFAULT_MD = NEW_MODEL / "OUTPUTS" / "newcase_comparison.md"


def _sha256_file(path: str | Path) -> str:
    """Streaming sha256 (provenance, not a forked metric)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def seed_runs(tau_root: Path) -> List[Path]:
    """The 3 frozen TAU seed run dirs under ``tau_root`` (headline_s{42,43,44})."""
    return [Path(tau_root) / f"headline_s{s}" for s in SEEDS]


def win_loss_tally(
    cases: Sequence[str], tau: Sequence[float], cnn: Sequence[float]
) -> Tuple[List[Tuple[str, float, float, float, str]], int, int, int]:
    """Per-case win/loss/tie from ``delta = tau - cnn`` (pure; unit-testable).

    Returns ``([(case, tau, cnn, delta, outcome), ...], wins, losses, ties)`` where
    ``outcome`` is ``"win"`` if ``delta > 0``, ``"loss"`` if ``delta < 0``, else
    ``"tie"``. Computed DIRECTLY from significance's paired (cases, tau, cnn) vectors
    (never from results/scripts/compare_runs.py — its CASE_MAP targets the legacy
    canonical-16 ALL_OUTPUTS masks, not the new-case frozen ConvLSTM ref, WARNING-2).
    """
    rows: List[Tuple[str, float, float, float, str]] = []
    wins = losses = ties = 0
    for c, t, k in zip(cases, tau, cnn):
        t, k = float(t), float(k)
        delta = t - k
        if delta > 0:
            outcome = "win"
            wins += 1
        elif delta < 0:
            outcome = "loss"
            losses += 1
        else:
            outcome = "tie"
            ties += 1
        rows.append((c, t, k, delta, outcome))
    return rows, wins, losses, ties


def _preflight(runs: Sequence[Path], cnn_dir: Path) -> None:
    """Fail loud (SystemExit) on any missing input BEFORE a number is computed.

    Requires all 3 ``best.pt`` checkpoints and all 12 per_frame_metrics.csv
    (3 seeds x EXPECTED_NEW_COUNT TAU cases + EXPECTED_NEW_COUNT CNN cases). Uses a
    glob count for the eval CSVs so it is robust to the exact <case>/ dir naming; the
    downstream exact-count ``significance.pair_cases(expect_cases=3)`` then enforces
    TAU<->CNN name alignment.
    """
    missing: List[str] = []
    wrong_count: List[str] = []

    for rd in runs:
        ck = rd / "checkpoints" / "best.pt"
        if not ck.is_file():
            missing.append(f"checkpoint: {ck}")
        eval_dir = rd / EVAL_SUBDIR
        found = glob.glob(
            os.path.join(str(eval_dir), "**", "per_frame_metrics.csv"), recursive=True
        )
        if len(found) != EXPECTED_NEW_COUNT:
            wrong_count.append(
                f"{eval_dir}: found {len(found)} per_frame_metrics.csv "
                f"(expected {EXPECTED_NEW_COUNT})"
            )

    cnn_found = glob.glob(
        os.path.join(str(cnn_dir), "**", "per_frame_metrics.csv"), recursive=True
    )
    if len(cnn_found) != EXPECTED_NEW_COUNT:
        wrong_count.append(
            f"{cnn_dir}: found {len(cnn_found)} per_frame_metrics.csv "
            f"(expected {EXPECTED_NEW_COUNT})"
        )

    problems = missing + wrong_count
    if problems:
        raise SystemExit(
            "[newcase] preflight FAIL — the frozen new-case eval is incomplete "
            "(run Task 2 on Gilbreth first):\n  " + "\n  ".join(problems)
        )


def newcase_data_sha(data_root: str | Path) -> str:
    """Honest data-provenance digest over the 3 new-case preprocessed caches.

    Combines ``sha256(features.npy)`` + ``sha256(meta.json)`` for every rostered
    new-case cache under ``<data_root>/_cache/<folder>/**`` into ONE ordered digest.
    Returns a documented ``unavailable:`` sentinel (NEVER a fabricated hash,
    T-08-10) when the scratch dataset/cache is unreachable (e.g. off-cluster), so the
    provenance line is always truthful about what was actually hashed.
    """
    cache = Path(data_root) / "_cache"
    parts: List[Tuple[str, str]] = []
    for folder in sorted(NEW_TEST_CASE_FOLDERS.values()):
        top = cache / folder
        if not top.is_dir():
            continue
        for name in ("features.npy", "meta.json"):
            for f in sorted(top.glob(os.path.join("**", name))):
                parts.append((f.relative_to(cache).as_posix(), _sha256_file(f)))
    if not parts:
        return (
            f"unavailable: new-case DATASET cache not reachable at {cache} "
            "(321x161 data resides on Gilbreth scratch; run on-cluster to record a "
            "real digest)"
        )
    combined = hashlib.sha256()
    for rel, sha in parts:
        combined.update(rel.encode())
        combined.update(sha.encode())
    return f"sha256:{combined.hexdigest()} (over {len(parts)} new-case cache files)"


def _fmt_pvalue(p: float) -> str:
    return f"{p:.4g}"


def _repo_rel(path: str | Path) -> str:
    """POSIX path relative to REPO_ROOT when possible, else the resolved absolute path.

    Always resolves to absolute first so a RELATIVE ``--tau-root``/``--cnn`` override
    (or a non-repo cwd) still yields a REPO_ROOT-relative provenance path that
    ``assert_provenance.py --check-artifact`` can re-hash (it resolves relative paths
    against REPO_ROOT, and handles absolute paths directly)."""
    ap = Path(path).resolve()
    try:
        return ap.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return ap.as_posix()


def render_markdown(
    *,
    percase: List[Tuple[str, float, float, int]],   # (case, tau_mean, tau_std, n_seeds)
    cnn_map: Dict[str, float],
    cases: Sequence[str],
    tau: Sequence[float],
    cnn: Sequence[float],
    wl_rows: Sequence[Tuple[str, float, float, float, str]],
    wins: int,
    losses: int,
    ties: int,
    wil_res,
    d: Sequence[float],
    seed_sha: Dict[str, str],
    runs: Sequence[Path],
    cnn_dir: Path,
    data_sha: str,
) -> str:
    """Build the newcase_comparison.md text — every number passed in was COMPUTED."""
    tau_std = {c: s for c, _m, s, _n in percase}
    n_seeds = percase[0][3]
    n_pairs = len(cases)

    tau_agg = statistics.mean(tau)
    tau_agg_spread = statistics.pstdev(tau)  # case-to-case spread (NOT the seed std)
    cnn_agg = statistics.mean(cnn)
    delta_agg = tau_agg - cnn_agg
    median_delta = statistics.median(d)
    n_nonzero = sum(1 for x in d if x != 0.0)

    lines: List[str] = []
    lines.append("# New-Case Head-to-Head: FractureTAU vs frozen ConvLSTM (OOD)")
    lines.append("")
    lines.append(
        f"_Generated by `new_model/scripts/make_newcase_comparison.py` "
        f"(UTC {_dt.datetime.now(_dt.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')})._ "
    )
    lines.append("")
    lines.append(
        "Frozen FractureTAU (3 seeds `headline_s{42,43,44}`, base/MODE-1 fracture-mask "
        "checkpoints ONLY — no pressure/vonmises/SED, no retraining) vs the frozen "
        "ConvLSTM reference, both on the 3 newly-registered cases under the identical "
        f"AR-rollout protocol. Metric: late-rollout (last-{int(FRAC * 100)}%) macro F1 "
        "(the ONE v1.0 D-04 metric, reused verbatim). TAU per-case value is the "
        f"{n_seeds}-seed mean; ± is the seed std."
    )
    lines.append("")

    # ---- HEADLINE: per-case table ----
    lines.append(f"## Per-case results (HEADLINE, n={n_pairs} cases)")
    lines.append("")
    lines.append("| Case | FractureTAU (mean ± std) | ConvLSTM | Δ (TAU − CNN) | Outcome |")
    lines.append("| ---- | ------------------------ | -------- | ------------- | ------- |")
    for case, t, k, delta, outcome in wl_rows:
        lines.append(
            f"| {case} | {t:.4f} ± {tau_std.get(case, 0.0):.4f} | {k:.4f} "
            f"| {delta:+.4f} | {outcome} |"
        )
    lines.append("")

    # ---- aggregate row ----
    lines.append("## Aggregate")
    lines.append("")
    lines.append(
        f"- **FractureTAU** = {tau_agg:.4f} ± {tau_agg_spread:.4f} "
        f"(mean across {n_pairs} cases ± case-to-case std)"
    )
    lines.append(f"- **ConvLSTM (frozen ref)** = {cnn_agg:.4f} (mean across {n_pairs} cases)")
    lines.append(f"- **Δ (TAU − CNN)** = {delta_agg:+.4f}")
    lines.append(
        f"- **Win/loss tally** = FractureTAU wins {wins}/{n_pairs} "
        f"(losses {losses}, ties {ties})"
    )
    lines.append("")

    # ---- Wilcoxon line ----
    lines.append("## Significance (one-sided Wilcoxon, TAU > CNN)")
    lines.append("")
    lines.append(
        f"Wilcoxon signed-rank (H1: median(TAU − CNN) > 0): "
        f"W={wil_res.statistic}, p={_fmt_pvalue(wil_res.pvalue)}, n={n_pairs} "
        f"(nonzero={n_nonzero}), median_delta={median_delta:+.4f}."
    )
    lines.append("")

    # ---- n=3 HONESTY note ----
    lines.append("## Honesty note (n=3)")
    lines.append("")
    lines.append(
        f"With only n={n_pairs} paired cases, the one-sided Wilcoxon's **minimum** "
        "one-sided p is 0.125 — it can NEVER reach 0.05 regardless of the effect. The "
        "p-value above is therefore reported as **INDICATIVE only**, not a significance "
        "claim. The **headline is the per-case table + mean±std** (D-04 texture), not "
        "the p-value (D-06 honest small-sample framing). This is an additive "
        "out-of-distribution result; the v1.0 in-distribution headline is a SEPARATE "
        "number and is deliberately NOT reproduced or merged here (D-03)."
    )
    lines.append("")

    # ---- machine-parseable SHA provenance block ----
    lines.append("## Provenance (SHA-tagged — every number traces to a checkpoint + data)")
    lines.append("")
    lines.append(
        "Frozen FractureTAU seed checkpoints (re-hashed by "
        "`scripts/assert_provenance.py --check-artifact`):"
    )
    lines.append("")
    for rd in runs:
        rel = _repo_rel(Path(rd) / "checkpoints" / "best.pt")
        sha = seed_sha[rd.name]
        lines.append(f"- {rd.name} {rel} sha256={sha}")
    lines.append("")
    cnn_rel = _repo_rel(cnn_dir)
    lines.append(f"- ConvLSTM frozen reference (base/MODE-1): `{cnn_rel}`")
    lines.append(f"- New-case DATASET data sha: `{data_sha}`")
    lines.append("")
    lines.append(
        "_Numbers above were computed by "
        "`new_model/scripts/make_newcase_comparison.py` directly from the "
        "`per_frame_metrics.csv` eval outputs and the actual `best.pt` sha256 — no "
        "value is hand-typed._"
    )
    lines.append("")
    return "\n".join(lines)


def build(
    *,
    tau_root: Path,
    cnn_dir: Path,
    data_root: Path,
    seed_csv_out: Path,
    md_out: Path,
    frac: float = FRAC,
) -> Path:
    """Assemble both artifacts from the frozen new-case eval + checkpoints."""
    runs = seed_runs(tau_root)
    _preflight(runs, cnn_dir)

    # 1. Per-case 3-seed mean±std (REUSE seed_aggregate; ONE metric, frac=0.20).
    percase = seed_aggregate.per_case_meanstd(
        runs, frac=frac, eval_subdir=EVAL_SUBDIR
    )
    if len(percase) != EXPECTED_NEW_COUNT:
        raise SystemExit(
            f"[newcase] expected {EXPECTED_NEW_COUNT} new cases, got {len(percase)}: "
            f"{[c for c, *_ in percase]}"
        )
    seed_aggregate.write_meanstd_csv(percase, seed_csv_out)
    print(f"[newcase] wrote {seed_csv_out}  ({len(percase)} cases, frac={frac})")

    # 2. Pair TAU (3-seed mean per case) against the frozen CNN (REUSE significance).
    tau_pc = [(case, mean) for case, mean, _std, _n in percase]
    cnn_map = significance._cnn_percase(cnn_dir, frac=frac)
    if not cnn_map:
        raise SystemExit(f"[newcase] no frozen-ConvLSTM CSVs under {cnn_dir!r}")
    cases, tau, cnn = significance.pair_cases(
        tau_pc, cnn_map, expect_cases=EXPECTED_NEW_COUNT
    )

    # 3. One-sided Wilcoxon TAU>CNN (REUSE significance; n=3 -> indicative only).
    wil_res, d = significance.wilcoxon_tau_gt_cnn(tau, cnn)

    # 4. Win/loss tally straight off significance's paired vectors (pure helper).
    wl_rows, wins, losses, ties = win_loss_tally(cases, tau, cnn)

    # 5. Provenance: per-seed best.pt sha256 (REUSE seed_aggregate) + data sha.
    seed_sha = seed_aggregate.seed_shas(runs)
    data_sha = newcase_data_sha(data_root)

    # 6. Render + write the comparison artifact.
    md = render_markdown(
        percase=percase,
        cnn_map=cnn_map,
        cases=cases,
        tau=tau,
        cnn=cnn,
        wl_rows=wl_rows,
        wins=wins,
        losses=losses,
        ties=ties,
        wil_res=wil_res,
        d=d,
        seed_sha=seed_sha,
        runs=runs,
        cnn_dir=cnn_dir,
        data_sha=data_sha,
    )
    md_out = Path(md_out)
    md_out.parent.mkdir(parents=True, exist_ok=True)
    md_out.write_text(md)
    print(
        f"[newcase] wrote {md_out}  (TAU wins {wins}/{len(cases)}, "
        f"W={wil_res.statistic}, p={_fmt_pvalue(wil_res.pvalue)})"
    )
    print(
        "[newcase] NEXT: python scripts/assert_provenance.py --check-artifact "
        f"{_repo_rel(md_out)}"
    )
    return md_out


def _selftest() -> None:
    """Offline self-test of the PURE helpers + the provenance-line regex contract.

    Runs WITHOUT any real eval data or checkpoints (safe on the laptop). Verifies:
      * the win/loss tally logic on synthetic (win, tie, loss) inputs, and
      * a sample provenance line the script emits matches the EXACT regex
        scripts/assert_provenance.py --check-artifact greps.
    """
    # --- win/loss tally on synthetic paired vectors ---
    cases = ["a", "b", "c"]
    tau = [0.50, 0.30, 0.40]
    cnn = [0.40, 0.30, 0.50]  # -> win, tie, loss
    rows, wins, losses, ties = win_loss_tally(cases, tau, cnn)
    assert (wins, ties, losses) == (1, 1, 1), (wins, ties, losses)
    assert [r[4] for r in rows] == ["win", "tie", "loss"], rows
    assert abs(rows[0][3] - 0.10) < 1e-12 and abs(rows[2][3] + 0.10) < 1e-12

    # --- provenance-line regex contract (must match assert_provenance --check-artifact) ---
    sample_sha = "0" * 64
    sample_rel = "new_model/OUTPUTS/headline_s42/checkpoints/best.pt"
    line = f"- headline_s42 {sample_rel} sha256={sample_sha}"
    pairs = re.findall(PROVENANCE_RE, line)
    assert len(pairs) == 1, pairs
    got_path, got_sha = pairs[0]
    assert got_path == sample_rel, got_path
    assert got_sha == sample_sha, got_sha

    # sanity: an aggregate/CNN/data line must NOT be mistaken for a checkpoint pair
    assert re.findall(PROVENANCE_RE, "- New-case DATASET data sha: `unavailable: ...`") == []

    print("[selftest] OK — win/loss tally + provenance-line regex contract pass")


def main(argv: Sequence[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        description="Assemble the SHA-tagged new-case FractureTAU-vs-ConvLSTM comparison "
        "(reuses seed_aggregate + significance; n=3, honestly framed).",
    )
    ap.add_argument(
        "--tau-root", default=str(DEFAULT_TAU_ROOT),
        help="Parent dir holding the frozen headline_s{42,43,44} seed runs "
        f"(default: {DEFAULT_TAU_ROOT}).")
    ap.add_argument(
        "--cnn", default=str(DEFAULT_CNN),
        help=f"Frozen ConvLSTM new-case reference dir (default: {DEFAULT_CNN}).")
    ap.add_argument(
        "--data-root", default=str(DEFAULT_DATA_ROOT),
        help=f"DATASET root (for the new-case cache data sha; default: {DEFAULT_DATA_ROOT}).")
    ap.add_argument(
        "--seed-csv-out", default=str(DEFAULT_SEED_CSV),
        help=f"Output per-case mean±std CSV (default: {DEFAULT_SEED_CSV}).")
    ap.add_argument(
        "--md-out", default=str(DEFAULT_MD),
        help=f"Output comparison markdown (default: {DEFAULT_MD}).")
    ap.add_argument("--frac", type=float, default=FRAC, help="late-rollout window (D-04).")
    ap.add_argument(
        "--selftest", action="store_true",
        help="Run the offline pure-helper + provenance-regex self-test and exit "
        "(no eval data / checkpoints required).")
    args = ap.parse_args(argv)

    if args.selftest:
        _selftest()
        return

    build(
        tau_root=Path(args.tau_root),
        cnn_dir=Path(args.cnn),
        data_root=Path(args.data_root),
        seed_csv_out=Path(args.seed_csv_out),
        md_out=Path(args.md_out),
        frac=args.frac,
    )


if __name__ == "__main__":
    main()
