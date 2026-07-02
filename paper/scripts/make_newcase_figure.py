#!/usr/bin/env python3
"""Render the New-Case Generalization figure (D-06) from the REAL 09-08 artifact.

Reuse-by-import bar: this driver NEVER re-derives a metric. It parses the already-
computed, SHA-tagged head-to-head numbers straight out of
``new_model/OUTPUTS/newcase_comparison.md`` (the 09-08 output of
``make_newcase_comparison.py``) and cross-checks the FractureTAU per-case mean/std
against ``results/newcase/newcase_seed_meanstd.csv``. No value is hand-typed; if the
two sources disagree the driver fails loudly rather than silently trusting one.

The figure shows, per newly-registered out-of-distribution case, the late-rollout
(last-20%) macro F1 of the frozen FractureTAU (3-seed mean, std error bar) against the
frozen ConvLSTM reference, with a win/loss marker — the same metric/protocol as the
16-case headline, plotted as additive OOD evidence (D-03), not merged with it.

Self-sufficient vector-PDF fonts: sets ``pdf.fonttype``/``ps.fonttype`` = 42 (Type-42,
no Type-3) at import time so the emitted PDF embeds proper fonts (gate_pdffonts).

Run from dynamic-fracture/:
    python paper/scripts/make_newcase_figure.py
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

# ---- self-sufficient vector-PDF fonts (Type-42; D-06) ----
import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42

import matplotlib.pyplot as plt  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
COMPARISON_MD = REPO_ROOT / "new_model" / "OUTPUTS" / "newcase_comparison.md"
SEED_CSV = REPO_ROOT / "results" / "newcase" / "newcase_seed_meanstd.csv"
OUT_PDF = REPO_ROOT / "paper" / "figures" / "newcase_generalization.pdf"

# Per-case table row, e.g.:
#   | new_MS211_V400 | 0.9160 ± 0.0161 | 0.6437 | +0.2723 | win |
_ROW = re.compile(
    r"^\|\s*(new_\S+)\s*\|\s*([0-9.]+)\s*(?:±|\+/-)\s*([0-9.]+)\s*\|"
    r"\s*([0-9.]+)\s*\|\s*([+\-][0-9.]+)\s*\|\s*(\w+)\s*\|"
)
# Aggregate lines, e.g. "- **FractureTAU** = 0.9252 ± 0.0067 (...)" / "**ConvLSTM ...** = 0.5974".
_AGG_TAU = re.compile(r"\*\*FractureTAU\*\*\s*=\s*([0-9.]+)\s*(?:±|\+/-)\s*([0-9.]+)")
_AGG_CNN = re.compile(r"\*\*ConvLSTM[^*]*\*\*\s*=\s*([0-9.]+)")
_WILCOXON = re.compile(r"W=([0-9.]+),\s*p=([0-9.]+),\s*n=(\d+)")


def _parse_comparison(md_path: Path):
    """Parse per-case + aggregate + Wilcoxon numbers from the 09-08 artifact (fail loud)."""
    if not md_path.exists():
        raise FileNotFoundError(f"comparison artifact not found: {md_path}")
    text = md_path.read_text()

    cases = []  # (name, tau_mean, tau_std, cnn, outcome)
    for line in text.splitlines():
        m = _ROW.match(line.strip())
        if m:
            cases.append(
                (m.group(1), float(m.group(2)), float(m.group(3)),
                 float(m.group(4)), m.group(6).lower())
            )
    if not cases:
        raise ValueError(f"no per-case rows parsed from {md_path}")

    agg_tau = _AGG_TAU.search(text)
    agg_cnn = _AGG_CNN.search(text)
    wil = _WILCOXON.search(text)
    if not (agg_tau and agg_cnn and wil):
        raise ValueError(f"missing aggregate/Wilcoxon lines in {md_path}")

    return {
        "cases": cases,
        "agg_tau_mean": float(agg_tau.group(1)),
        "agg_tau_std": float(agg_tau.group(2)),
        "agg_cnn": float(agg_cnn.group(1)),
        "wilcoxon": (float(wil.group(1)), float(wil.group(2)), int(wil.group(3))),
    }


def _crosscheck_csv(cases, csv_path: Path, tol: float = 5e-4) -> None:
    """Confirm the parsed TAU per-case mean/std match the seed CSV (no divergent sources)."""
    if not csv_path.exists():
        # CSV is a secondary cross-check; its absence must not silently pass.
        raise FileNotFoundError(f"seed cross-check CSV not found: {csv_path}")
    by_case = {}
    with csv_path.open() as fh:
        for row in csv.DictReader(fh):
            by_case[row["case"]] = (
                float(row["late_rollout_f1_mean"]), float(row["late_rollout_f1_std"])
            )
    for name, tau_mean, tau_std, _cnn, _out in cases:
        if name not in by_case:
            raise ValueError(f"{name} present in comparison.md but missing from {csv_path}")
        csv_mean, csv_std = by_case[name]
        # comparison.md is rounded to 4 dp; the CSV carries full precision.
        if abs(round(csv_mean, 4) - tau_mean) > tol or abs(round(csv_std, 4) - tau_std) > tol:
            raise ValueError(
                f"{name}: comparison.md ({tau_mean}±{tau_std}) disagrees with "
                f"CSV ({csv_mean:.6f}±{csv_std:.6f})"
            )


def _short_label(case: str) -> str:
    """new_MS211_V400 -> MS211."""
    m = re.search(r"(MS\d+)", case)
    return m.group(1) if m else case


def render(data, out: Path) -> None:
    cases = data["cases"]
    labels = [_short_label(c[0]) for c in cases]
    tau_mean = [c[1] for c in cases]
    tau_std = [c[2] for c in cases]
    cnn = [c[3] for c in cases]

    x = range(len(cases))
    width = 0.38
    tau_color = "#1f77b4"
    cnn_color = "#d62728"

    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    xt = [i - width / 2 for i in x]
    xc = [i + width / 2 for i in x]

    ax.bar(xt, tau_mean, width, yerr=tau_std, capsize=4,
           color=tau_color, label="FractureTAU (3-seed mean $\\pm$ std)")
    ax.bar(xc, cnn, width, color=cnn_color, label="ConvLSTM (frozen ref)")

    # Win markers above each pair (every case is a TAU win here; drawn from the data).
    ymax = max(max(tau_mean), max(cnn)) + max(tau_std) + 0.03
    for i, (m, s, c) in enumerate(zip(tau_mean, tau_std, cnn)):
        if m > c:
            ax.annotate("✓", (i, m + s + 0.015), ha="center", va="bottom",
                        fontsize=11, color=tau_color, fontweight="bold")

    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_ylabel("Late-rollout macro $F_1$")
    ax.set_ylim(0.0, min(1.0, ymax + 0.05))
    ax.set_xlabel("Held-out (OOD) case")

    w, p, n = data["wilcoxon"]
    tally = sum(1 for c in cases if c[1] > c[3])
    ax.set_title(
        f"New-case generalization (n={n}): "
        f"TAU {data['agg_tau_mean']:.4f}$\\pm${data['agg_tau_std']:.4f} "
        f"vs CNN {data['agg_cnn']:.4f}  —  wins {tally}/{n}",
        fontsize=9,
    )
    ax.legend(loc="lower right", fontsize=8, framealpha=0.9)
    # Indicative-only Wilcoxon (n=3 min one-sided p = 0.125), never a significance claim.
    ax.text(0.02, 0.02,
            f"Wilcoxon W={w:g}, p={p:g} (indicative; min p for n={n})",
            transform=ax.transAxes, fontsize=7, va="bottom", color="0.35")

    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, format="pdf", bbox_inches="tight")
    plt.close(fig)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--comparison", default=str(COMPARISON_MD))
    ap.add_argument("--csv", default=str(SEED_CSV))
    ap.add_argument("--out", default=str(OUT_PDF))
    args = ap.parse_args(argv)

    data = _parse_comparison(Path(args.comparison))
    _crosscheck_csv(data["cases"], Path(args.csv))
    render(data, Path(args.out))

    print(f"[newcase-fig] wrote {args.out}")
    print(f"[newcase-fig] cases={[c[0] for c in data['cases']]}")
    print(f"[newcase-fig] agg TAU={data['agg_tau_mean']:.4f}±{data['agg_tau_std']:.4f} "
          f"CNN={data['agg_cnn']:.4f} Wilcoxon={data['wilcoxon']}")


if __name__ == "__main__":
    sys.exit(main())
