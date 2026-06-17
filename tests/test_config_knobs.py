# tests/test_config_knobs.py
"""03-01: the Phase-3 Config knobs round-trip through save/load and argparse.

These knobs (boundary loss, AR pushforward/noise, monotone head, Nyquist
selection horizon) are the shared contract every other Phase-3 plan wires its
guarded ``if cfg.<knob>:`` blocks against. They MUST persist into each run's
``config.json`` so a reported metric is reproducible from its config (DATA-04 /
T-03-01) and the directional sweep can drive every variant from the CLI.

No torch needed: ``config.py`` has no torch import, so this is pure-stdlib and
runs locally (no Gilbreth / skip guard). Mirrors the grid round-trip pattern of
``tests/test_calibration.py`` and the ``src`` sys.path shim of
``tests/test_determinism.py``.

Run:  cd dynamic-fracture && python -m pytest tests/test_config_knobs.py -q
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

# ---- sys.path shim to reach new_model/src (mirrors test_determinism.py) ----
SRC = Path(__file__).resolve().parents[1] / "new_model" / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from config import Config, parse_config, resolve_seed_source  # noqa: E402


def test_new_knobs_roundtrip(tmp_path):
    # Every Phase-3 knob must survive a save -> load round-trip to config.json.
    cfg = Config(
        boundary_weight=0.5,
        boundary_ramp=True,
        ar_pushforward=1,
        feedback_noise_std=0.05,
        feedback_noise_p=0.0,
        head_type="monotone_delta",
        val_rollout_steps=20,
    )
    path = tmp_path / "c.json"
    cfg.save(path)
    loaded = Config.load(path)

    assert loaded.boundary_weight == 0.5
    assert loaded.boundary_ramp is True
    assert loaded.ar_pushforward == 1
    assert loaded.feedback_noise_std == 0.05
    assert loaded.feedback_noise_p == 0.0
    assert loaded.head_type == "monotone_delta"
    assert loaded.val_rollout_steps == 20


def test_argparse_new_flags():
    # parse_config auto-derives a --kebab-case flag for every new field.
    cfg = parse_config([
        "--boundary-weight", "0.5",
        "--ar-pushforward", "1",
        "--feedback-noise-std", "0.1",
        "--head-type", "monotone_delta",
        "--val-rollout-steps", "20",
    ])
    assert cfg.boundary_weight == 0.5
    assert cfg.ar_pushforward == 1
    assert cfg.feedback_noise_std == 0.1
    assert cfg.head_type == "monotone_delta"
    assert cfg.val_rollout_steps == 20


def test_head_type_choices_guard():
    # A typo'd head must hard-fail at parse time, never silently alias (T-03-02).
    with pytest.raises(SystemExit):
        parse_config(["--head-type", "bogus"])


def test_val_rollout_steps_default_nyquist():
    # Default selection horizon must be Nyquist-adequate (>=15) so late-rollout
    # selection samples long-horizon drift (RESEARCH Pitfall 1).
    assert Config().val_rollout_steps >= 15


# ---- 03-06: warm-start (--init-from) vs resume policy (sweep budget trick) ----
def test_init_from_roundtrips_and_has_cli_flag(tmp_path):
    # init_from must persist to config.json (DATA-04 provenance) and expose a
    # --init-from flag so the sweep can drive it from the CLI.
    cfg = parse_config(["--init-from", "OUTPUTS/stage1_shared/checkpoints/last.pt"])
    assert cfg.init_from == "OUTPUTS/stage1_shared/checkpoints/last.pt"
    path = tmp_path / "c.json"
    cfg.save(path)
    assert Config.load(path).init_from == cfg.init_from
    assert Config().init_from == "none"      # default: no warm-start


def test_seed_source_first_launch_warm_starts():
    # Sweep variant's FIRST launch: own last.pt absent + init_from set -> warm
    # start (fresh schedule), NOT resume. This is the bug fix: --resume <stage1>
    # would have restored the Stage-1 epoch and trained zero Stage-2 epochs.
    assert resolve_seed_source(
        resume="auto", init_from="OUTPUTS/stage1_shared/checkpoints/last.pt",
        last_ckpt_exists=False,
    ) == "init"


def test_seed_source_requeue_continues_own_run():
    # After preemption the variant's OWN last.pt exists -> resume wins over
    # init_from so finished Stage-2 epochs are not discarded.
    assert resolve_seed_source(
        resume="auto", init_from="OUTPUTS/stage1_shared/checkpoints/last.pt",
        last_ckpt_exists=True,
    ) == "resume"


def test_seed_source_explicit_resume_path_and_fresh():
    # An explicit --resume /path always resumes; no resume + no init_from = fresh.
    assert resolve_seed_source("/path/to/last.pt", "none", False) == "resume"
    assert resolve_seed_source("none", "none", False) == "fresh"
