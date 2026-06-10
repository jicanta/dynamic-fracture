"""Generate a tiny synthetic dataset shaped exactly like the real CSV layout.

Useful for smoke-testing the pipeline on any machine without the real data:

    python scripts/make_synth_data.py --out /tmp/synth_dataset
    python -m src.train --data-root /tmp/synth_dataset --run-name smoke \
        --epochs-stage1 1 --epochs-stage2 1 --rollout-steps 2 \
        --batch-size 2 --num-workers 0 --hid-s 16 --hid-t 32 --n-temporal 2
    python -m src.evaluate --data-root /tmp/synth_dataset --run-name smoke
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

H, W = 33, 17           # small grid
N_FRAMES = 26           # frames per run (first one gets dropped)


def make_run(run_dir: Path, vel: float, seed: int) -> None:
    x_vals = np.linspace(0.0, 1.6e-3, W).astype(np.float32)
    y_vals = np.linspace(0.0, 3.2e-3, H).astype(np.float32)
    xx, yy = np.meshgrid(x_vals, y_vals)

    r = np.random.default_rng(seed)
    run_dir.mkdir(parents=True, exist_ok=True)
    gc = (1.0 + 0.3 * r.random((H, W))).astype(np.float32)
    crack_col = int(r.integers(W // 4, 3 * W // 4))
    for t in range(N_FRAMES):
        # crack grows downward over time, speed scales with velocity
        depth = int(min(H, t * (1 + vel / 400.0)))
        mask = np.zeros((H, W), np.float32)
        mask[:depth, max(crack_col - 1, 0):crack_col + 2] = 1.0
        ux = (r.standard_normal((H, W)) * 50 + t * vel * 0.01).astype(np.float32)
        uy = (r.standard_normal((H, W)) * 50).astype(np.float32)
        df = pd.DataFrame({
            "x": xx.ravel(), "y": yy.ravel(),
            "ux": ux.ravel(), "uy": uy.ravel(),
            "Gc": gc.ravel(),
            "fracture_mask": mask.ravel(),
            "pressure": (r.standard_normal(H * W) * 10).astype(np.float32),
            "a_pos": r.random(H * W).astype(np.float32),  # legacy SED name
        })
        df.to_csv(run_dir / f"{run_dir.name}_t{t:04d}_{t:.2f}ns.csv", index=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/tmp/synth_dataset")
    args = ap.parse_args()
    root = Path(args.out)

    for i, vel in enumerate([100, 200, 400]):
        make_run(root / "trainDS" / f"F_SYN{i}_V{vel}_out", vel, seed=10 + i)

    # two test cases matching TEST_CASE_FOLDERS names
    make_run(root / "F_MS206_V400_out" / "F_MS206_V400_out", 400, seed=99)
    make_run(root / "F_inclusions_1_2_out" / "F_inclusions_1_2_V400_out", 400, seed=7)

    print("synthetic dataset at", root)


if __name__ == "__main__":
    main()
