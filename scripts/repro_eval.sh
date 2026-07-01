#!/bin/bash
# One-command reproducibility entrypoint (PAPER-04, D-07 internal-only).
#
# Regenerates the headline comparison + physical-metrics report by chaining the
# three first-party drivers behind a single command:
#
#     1. new_model/src.evaluate          -> per-case AR (+ optional teacher-forced) rollout
#     2. results/dashboard/make_report   -> locked D-14 multi-metric dashboard (headline)
#     3. results/phys_metrics/make_phys_report -> PHYS-01..06 physical-metrics report
#
# Reproducibility is mean+/-std over 3 seeds in a uv.lock-pinned env (PAPER-03),
# NOT bit-determinism (Pitfall 9). The pinned analysis env is created with
# `uv sync --group dev`; cluster runs use the existing Slurm scripts under
# new_model/scripts/*.sbatch and kathleens-model/scripts/*.sbatch (not duplicated here).
#
# Usage (from anywhere; paths derive from this script's location):
#     bash scripts/repro_eval.sh                 # full regen, run name tau_refined
#     bash scripts/repro_eval.sh --run-name foo  # regen a different run
#     bash scripts/repro_eval.sh --skip-eval     # reuse existing eval outputs, just rebuild reports
#     bash scripts/repro_eval.sh --smoke         # dry-run the wiring (no heavy compute, no GPU/artifacts)
#     bash scripts/repro_eval.sh --from-outputs        # zero-scratch, no-GPU, no-checkpoint regen from archive/v1.0/outputs into archive/repro-out/ (NON-destructive; bundled archive is read-only)
#     bash scripts/repro_eval.sh --from-outputs DEST   # same, writing the regenerated artifacts under DEST instead of the default archive/repro-out/
#
# SECURITY (T-05-18): all paths are derived from the repo root below. This script
# MUST NOT contain absolute cluster/user-scratch paths or credentials.
set -euo pipefail

# ---- repo-root-derived path constants (NEVER hard-code cluster paths) ----
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
NEW_MODEL_DIR="$REPO_ROOT/new_model"
DASHBOARD_DRIVER="$REPO_ROOT/results/dashboard/make_report.py"
PHYS_DRIVER="$REPO_ROOT/results/phys_metrics/make_phys_report.py"
FPFN_DRIVER="$REPO_ROOT/results/findings/make_fpfn_panel.py"
# Self-contained v1.0 archive (bundled outputs; zero-scratch light-repro SOURCE).
# The archive tree is READ-ONLY input to --from-outputs: every path under
# archive/v1.0/ is sha256-tamper-checked by manifest.json (assert_provenance.py
# --archive), and matplotlib PDFs are non-deterministic, so regenerating IN PLACE
# would make the archive fail its own provenance check (CR-01). --from-outputs
# therefore writes every regenerated artifact into a SEPARATE scratch dir OUTSIDE
# the hashed tree (default archive/repro-out/, overridable), never back into
# archive/v1.0/.
ARCHIVE_DIR="$REPO_ROOT/archive/v1.0"
ARCHIVE_OUTPUTS="$ARCHIVE_DIR/outputs"
# Default non-destructive light-repro output dir (sibling of v1.0, NOT under the
# manifest-hashed tree, so it can never trip assert_provenance.py --archive).
REPRO_OUT_DIR="$REPO_ROOT/archive/repro-out"

# Python interpreter: override with `PYTHON=... bash scripts/repro_eval.sh` (e.g.
# `uv run` envs export their own `python`). Defaults to `python`.
PYTHON="${PYTHON:-python}"

# ---- arg parsing ----
RUN_NAME="tau_refined"
SMOKE=0
SKIP_EVAL=0
FROM_OUTPUTS=0

usage() {
  grep '^# ' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-name)
      [[ $# -ge 2 ]] || { echo "[repro] ERROR: --run-name needs a value" >&2; exit 2; }
      RUN_NAME="$2"; shift 2 ;;
    --run-name=*)
      RUN_NAME="${1#*=}"; shift ;;
    --smoke)
      SMOKE=1; shift ;;
    --skip-eval)
      SKIP_EVAL=1; shift ;;
    --from-outputs)
      FROM_OUTPUTS=1
      # Optional inline destination: consume the next token as DEST unless it is
      # another flag (leading '-') or absent. Keeps the bare flag working.
      if [[ $# -ge 2 && "$2" != -* ]]; then
        REPRO_OUT_DIR="$2"; shift 2
      else
        shift
      fi ;;
    --from-outputs=*)
      FROM_OUTPUTS=1; REPRO_OUT_DIR="${1#*=}"; shift ;;
    -h|--help)
      usage; exit 0 ;;
    *)
      echo "[repro] ERROR: unknown argument: $1" >&2
      echo "Run 'bash scripts/repro_eval.sh --help' for usage." >&2
      exit 2 ;;
  esac
done

# The three commands this entrypoint chains (printed in every mode for provenance).
EVAL_CMD="(cd '$NEW_MODEL_DIR' && '$PYTHON' -m src.evaluate --run-name '$RUN_NAME')"
DASHBOARD_CMD="'$PYTHON' '$DASHBOARD_DRIVER' --run-name '$RUN_NAME'"
PHYS_CMD="'$PYTHON' '$PHYS_DRIVER' --run-name '$RUN_NAME'"

# ---------------------------------------------------------------------------
# --smoke: prove the WIRING without heavy compute (RESEARCH Test Map PAPER-04).
# Validates path resolution, the chained command construction, and the presence
# (+ Python-syntax validity) of each entrypoint. Entrypoints owned by a not-yet-
# executed plan (e.g. make_phys_report.py from 05-06) are reported PENDING rather
# than failing, so --smoke exits 0 on a machine without the pulled cluster
# artifacts or checkpoints. No GPU, no dataset, no network.
# ---------------------------------------------------------------------------
if [[ "$SMOKE" == "1" ]]; then
  echo "[smoke] repro_eval wiring check (run-name=$RUN_NAME)"
  echo "[smoke] repo root: $REPO_ROOT"

  # Pick a lightweight syntax checker if any python is on PATH (ast.parse only,
  # imports nothing heavy). Absent -> presence-only check, still meaningful.
  SYNTAX_PY=""
  for cand in "$PYTHON" python3 python; do
    if command -v "$cand" >/dev/null 2>&1; then SYNTAX_PY="$cand"; break; fi
  done

  smoke_check_entrypoint() {
    # $1=label  $2=file  $3=full command string
    local label="$1" file="$2" cmd="$3"
    if [[ -f "$file" ]]; then
      if [[ -n "$SYNTAX_PY" ]]; then
        if "$SYNTAX_PY" -c "import ast,sys; ast.parse(open(sys.argv[1]).read())" "$file" >/dev/null 2>&1; then
          echo "[smoke] OK      $label: entrypoint present + parses -> $cmd"
        else
          echo "[smoke] FAIL    $label: entrypoint present but FAILS to parse: $file" >&2
          return 1
        fi
      else
        echo "[smoke] OK      $label: entrypoint present -> $cmd"
      fi
    else
      echo "[smoke] PENDING $label: entrypoint not yet present (regenerated by its owning plan) -> $cmd"
    fi
    return 0
  }

  # new_model/src/evaluate.py is the module behind `-m src.evaluate`.
  smoke_check_entrypoint "evaluate"  "$NEW_MODEL_DIR/src/evaluate.py" "$EVAL_CMD"
  smoke_check_entrypoint "dashboard" "$DASHBOARD_DRIVER"              "$DASHBOARD_CMD"
  smoke_check_entrypoint "phys"      "$PHYS_DRIVER"                   "$PHYS_CMD"

  echo "[smoke] wiring OK — full regen would run the chain above (eval -> dashboard -> phys)."
  echo "SMOKE_OK"
  exit 0
fi

# ---------------------------------------------------------------------------
# --from-outputs: zero-scratch, no-GPU, no-checkpoint LIGHT repro (D-03/D-05).
# Skips src.evaluate entirely (no Slurm, no checkpoint load) and rebuilds the
# headline seed table + comparison dashboard + physical-metrics report + FP/FN
# panel straight from the bundled archive/v1.0/outputs tree.
#
# NON-DESTRUCTIVE (CR-01): the archive tree (archive/v1.0/) is READ-ONLY input.
# Every path under it is sha256-tamper-checked by manifest.json, and matplotlib
# PDFs are non-deterministic, so regenerating in place would make the archive
# fail its own `assert_provenance.py --archive` check. All regenerated artifacts
# are therefore written into REPRO_OUT_DIR (default archive/repro-out/, outside
# the hashed tree; override via `--from-outputs DEST`) — never back into
# archive/v1.0/, and never the source results/ or paper/ tree (threat T-08-09 /
# W5). The ConvLSTM reference resolves via the bundled convlstm_ref/BASE/<case>/
# layout (--old-root + --ref-mode BASE).
# ---------------------------------------------------------------------------
if [[ "$FROM_OUTPUTS" == "1" ]]; then
  # Resolve REPRO_OUT_DIR to an absolute path (relative DEST -> relative to CWD).
  mkdir -p "$REPRO_OUT_DIR"
  REPRO_OUT_DIR="$(cd "$REPRO_OUT_DIR" && pwd)"
  REPRO_OUTPUTS="$REPRO_OUT_DIR/outputs"
  REPRO_FIGS="$REPRO_OUT_DIR/figures"
  REPRO_DIAG="$REPRO_OUT_DIR/diagnostics"
  mkdir -p "$REPRO_OUTPUTS" "$REPRO_FIGS" "$REPRO_DIAG"

  echo "[repro] --from-outputs: zero-scratch, no-GPU, no-checkpoint regen"
  echo "[repro] repo root:      $REPO_ROOT"
  echo "[repro] bundled tree:   $ARCHIVE_OUTPUTS (READ-ONLY, tamper-checked)"
  echo "[repro] writing into:   $REPRO_OUT_DIR (outside the hashed archive tree)"

  SEED_CMD="(cd '$NEW_MODEL_DIR' && '$PYTHON' -m scripts.seed_aggregate --runs '$ARCHIVE_OUTPUTS' --out '$REPRO_OUTPUTS/seed_meanstd.csv' --no-checkpoints)"
  DASH_CMD="'$PYTHON' '$DASHBOARD_DRIVER' --run-name tau_refined --new-root '$ARCHIVE_OUTPUTS' --old-root '$ARCHIVE_OUTPUTS/convlstm_ref' --ref-mode BASE --report '$REPRO_OUT_DIR/REPORT.md' --figures '$REPRO_FIGS' --diagnostics '$REPRO_DIAG'"
  FPFN_CMD="'$PYTHON' '$FPFN_DRIVER' --case-dir '$ARCHIVE_OUTPUTS/headline_s42/eval/test_inclusions_1_2' --out '$REPRO_FIGS/fpfn_inclusions_1_2.pdf'"
  PHYS_CMD2="'$PYTHON' '$PHYS_DRIVER' --run-name tau_refined --new-root '$ARCHIVE_OUTPUTS' --report '$REPRO_OUT_DIR/PHYS_REPORT.md' --figures '$REPRO_FIGS'"

  echo "[repro] step 1/4: seed mean±std table (no checkpoint load)"
  echo "[repro]   + $SEED_CMD"
  ( cd "$NEW_MODEL_DIR" && "$PYTHON" -m scripts.seed_aggregate \
      --runs "$ARCHIVE_OUTPUTS" --out "$REPRO_OUTPUTS/seed_meanstd.csv" --no-checkpoints )

  echo "[repro] step 2/4: multi-metric dashboard (ConvLSTM ref via BASE/)"
  echo "[repro]   + $DASH_CMD"
  "$PYTHON" "$DASHBOARD_DRIVER" --run-name tau_refined \
    --new-root "$ARCHIVE_OUTPUTS" \
    --old-root "$ARCHIVE_OUTPUTS/convlstm_ref" --ref-mode BASE \
    --report "$REPRO_OUT_DIR/REPORT.md" --figures "$REPRO_FIGS" \
    --diagnostics "$REPRO_DIAG"

  echo "[repro] step 3/4: FP/FN qualitative panel"
  echo "[repro]   + $FPFN_CMD"
  "$PYTHON" "$FPFN_DRIVER" \
    --case-dir "$ARCHIVE_OUTPUTS/headline_s42/eval/test_inclusions_1_2" \
    --out "$REPRO_FIGS/fpfn_inclusions_1_2.pdf"

  echo "[repro] step 4/4: physical-metrics report"
  echo "[repro]   + $PHYS_CMD2"
  "$PYTHON" "$PHYS_DRIVER" --run-name tau_refined --new-root "$ARCHIVE_OUTPUTS" \
    --report "$REPRO_OUT_DIR/PHYS_REPORT.md" --figures "$REPRO_FIGS"

  echo
  echo "[repro] --from-outputs done. Regenerated (all under $REPRO_OUT_DIR/):"
  echo "  cat '$REPRO_OUT_DIR/REPORT.md'          # headline + multi-metric dashboard (with ConvLSTM ref)"
  echo "  cat '$REPRO_OUT_DIR/PHYS_REPORT.md'     # PHYS-01..06 physical-metrics report"
  echo "  ls  '$REPRO_FIGS/'                       # dashboard + physical + FP/FN figures"
  echo "  cat '$REPRO_OUTPUTS/seed_meanstd.csv'"
  echo "Zero scratch, no GPU, no checkpoint load — self-contained from the bundled outputs."
  echo "The bundled archive/v1.0/ tree is left BYTE-UNCHANGED (verify: scripts/assert_provenance.py --archive)."
  exit 0
fi

# ---------------------------------------------------------------------------
# Full regeneration path (intended for a machine WITH the pulled artifacts /
# checkpoints + the uv-pinned env; not expected to run end-to-end without them).
# ---------------------------------------------------------------------------
echo "[repro] regenerating headline + physical metrics for run '$RUN_NAME'"
echo "[repro] repo root: $REPO_ROOT"

if [[ "$SKIP_EVAL" == "1" ]]; then
  echo "[repro] --skip-eval: reusing existing eval outputs under new_model/OUTPUTS/$RUN_NAME/eval"
else
  echo "[repro] step 1/3: AR rollout evaluation"
  echo "[repro]   + $EVAL_CMD"
  ( cd "$NEW_MODEL_DIR" && "$PYTHON" -m src.evaluate --run-name "$RUN_NAME" )
fi

echo "[repro] step 2/3: multi-metric dashboard (headline)"
echo "[repro]   + $DASHBOARD_CMD"
"$PYTHON" "$DASHBOARD_DRIVER" --run-name "$RUN_NAME"

echo "[repro] step 3/3: physical-metrics report"
echo "[repro]   + $PHYS_CMD"
"$PYTHON" "$PHYS_DRIVER" --run-name "$RUN_NAME"

# ---- inspection footer (mirror pull_regen.sh:29-35) ----
echo
echo "[repro] done. Inspect the regenerated artifacts:"
echo "  cat results/REPORT.md                                  # headline + multi-metric dashboard"
echo "  cat results/phys_metrics/PHYS_REPORT.md                # PHYS-01..06 physical-metrics report"
echo "  ls  results/figures/                                   # committed dashboard + physical figures"
echo
echo "Reproducibility: numbers are mean+/-std over 3 seeds in the uv.lock-pinned env;"
echo "NOT bit-deterministic. Recreate the env with: uv sync --group dev"
