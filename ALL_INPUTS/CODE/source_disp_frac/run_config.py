from __future__ import annotations

from typing import Dict, Tuple, Optional

MODE_TO_EXTRA = {
    1: None,
    2: "pressure",
    3: "vonmises",
    4: "SED",
    5: "SED",
    6: "SED",   # +SED input, predict fracture_mask + ux + uy
    7: None,    # base input, predict fracture_mask + ux + uy
}

MODE_TO_TARGET = {
    1: "fracture_only",
    2: "fracture_only",
    3: "fracture_only",
    4: "fracture_only",
    5: "fracture_sed",
    6: "fracture_sed_disp",
    7: "fracture_disp",
}

BASE_FEATURE_COLS: Tuple[str, ...] = ("x", "y", "ux", "uy", "Gc", "fracture_mask")


def build_cfg(mode: int, *, include_extra_in_feature_cols: bool = True) -> Dict[str, object]:
    """
    Build cfg from MODE.

    MODE:
      1 = base input, predict fracture_mask
      2 = +pressure input, predict fracture_mask
      3 = +vonmises input, predict fracture_mask
      4 = +SED input, predict fracture_mask only
      5 = +SED input, predict fracture_mask and SED
      6 = +SED input, predict fracture_mask, ux, and uy
      7 = base input, predict fracture_mask, ux, and uy

    include_extra_in_feature_cols:
      True  -> cfg["feature_cols"] includes the extra column, e.g. "pressure", "vonmises", or "SED"
      False -> cfg["feature_cols"] stays base-only; cfg["extra"] still indicates the extra var
    """
    if mode not in MODE_TO_EXTRA:
        raise ValueError(f"Unknown MODE={mode}. Expected one of {sorted(MODE_TO_EXTRA.keys())}.")

    extra: Optional[str] = MODE_TO_EXTRA[mode]
    target_mode: str = MODE_TO_TARGET[mode]

    feature_cols = BASE_FEATURE_COLS
    if include_extra_in_feature_cols and (extra is not None):
        feature_cols = BASE_FEATURE_COLS + (extra,)

    predict_sed = mode == 5
    predict_disp = mode in (6, 7)

    if mode in (6, 7):
        target_cols = ("fracture_mask", "ux", "uy")
        target_channel_names = ("fracture_mask", "ux_scaled", "uy_scaled")
    elif mode == 5:
        target_cols = ("fracture_mask", "SED")
        target_channel_names = ("fracture_mask", "SED")
    else:
        target_cols = ("fracture_mask",)
        target_channel_names = ("fracture_mask",)

    output_channels = len(target_cols)

    return {
        "feature_cols": feature_cols,
        "extra": extra,
        "target_mode": target_mode,
        "predict_sed": predict_sed,
        "predict_disp": predict_disp,
        "target_cols": target_cols,
        "target_channel_names": target_channel_names,
        "output_channels": output_channels,
    }