from __future__ import annotations

from typing import Dict, Tuple, Optional

MODE_TO_EXTRA = {
    1: None,
    2: "pressure",
    3: "vonmises",
    4: "SED",
}

BASE_FEATURE_COLS: Tuple[str, ...] = ("x", "y", "ux", "uy", "Gc", "fracture_mask")


def build_cfg(mode: int, *, include_extra_in_feature_cols: bool = True) -> Dict[str, object]:
    """
    Build cfg from MODE.

    MODE:
      1=base, 2=+pressure, 3=+vonmises, 4=+SED

    include_extra_in_feature_cols:
      True  -> cfg["feature_cols"] includes the extra column (e.g. "pressure")
      False -> cfg["feature_cols"] stays base-only; cfg["extra"] still indicates the extra var
    """
    if mode not in MODE_TO_EXTRA:
        raise ValueError(f"Unknown MODE={mode}. Expected one of {sorted(MODE_TO_EXTRA.keys())}.")

    extra: Optional[str] = MODE_TO_EXTRA[mode]

    feature_cols = BASE_FEATURE_COLS
    if include_extra_in_feature_cols and (extra is not None):
        feature_cols = BASE_FEATURE_COLS + (extra,)

    return {"feature_cols": feature_cols, "extra": extra}
