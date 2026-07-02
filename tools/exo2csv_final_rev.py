#!/usr/bin/env python3
# ============================================================================
# VENDORED VERBATIM from Kathleen's shared Box folder (author's original).
# Do NOT rewrite/reimplement — this is the canonical exodus->CSV ingestion tool
# the DATASET/ schema is defined by. Phase 9 adopts it as-is; Phase 12 reuses it.
# Provenance: pasted by the user 2026-07-01 (SURF Purdue 2026). Any change must be
# a minimal, clearly-marked delta over this baseline, never a from-scratch rewrite.
# ============================================================================
import os
import glob
import re
import argparse
import numpy as np
import pandas as pd
import netCDF4


# --------------------------
# ONLY run these exact exodus files (by basename)
# --------------------------

TARGET_BASENAMES = {
    # "F_MS201_V100_out.e",
    # "F_MS201_V200_out.e",
    # "F_MS201_V400_out.e",
    # "F_MS201_V1000_out.e",

    # "F_MS202_V100_out.e",
    # "F_MS202_V200_out.e",
    # "F_MS202_V400_out.e",
    # "F_MS202_V1000_out.e",

    # "F_MS203_V100_out.e",
    # "F_MS203_V200_out.e",
    # "F_MS203_V400_out.e",
    # "F_MS203_V1000_out.e",

    # "F_MS204_V100_out.e",
    # "F_MS204_V200_out.e",
    # "F_MS204_V400_out.e",
    # "F_MS204_V1000_out.e",

    # "F_MS205_V100_out.e",
    # "F_MS205_V200_out.e",
    # "F_MS205_V400_out.e",
    # "F_MS205_V1000_out.e",

    #"F_MS206_V100_out.e",
    # "F_MS206_V200_out.e",
    # "F_MS206_V400_out.e",
    # "F_MS206_V1000_out.e",
    # "F_vert-layers_limRange_out.e"

    # "MS5_150ms_inc_out.e",
    # "F_V-BC_MS5_400ms_inc_out.e",

    # "F_horizontal-layers_2_out.e"
    # "F_horizontal-layers_4_out.e",
    # "F_horizontal-layers_3_out.e",

    # "F_inclusions-layers_1_2_out.e",
    # "F_inclusions-layers_2_2_out.e",
    # "F_inclusions-layers_3_2_out.e"

# "F_inclusions_limited_out.e",
# "F_PBX_2_V400_out.e",
# "F_PBX_1_V400_out.e",
# "F_MS210_V400_out.e",
#"F_emergency_horizontal_out.e"

#"F_PBX1_V400_out.e",
#"F_inclusions_limited_out1.e"

"F_MS211_V400_out.e",
"F_MS221_V400_out.e",
"F_MS231_V400_out.e",



}


def parse_ms_and_vel_from_name(path):
    base = os.path.basename(path)
    m = re.search(r"(MS\d+)_([0-9]+)ms", base)
    if not m:
        return None, None
    return m.group(1), int(m.group(2))


def should_convert_exo(path):
    base = os.path.basename(path)
    return base in TARGET_BASENAMES
    # extra safety: must contain "inc"
    #if "inc" not in base.lower():
    #    return False
    #return True


def extract_sim_id(exo_filename):
    """
    From e.g. F_V-BC_MS11_200ms_inc_out.e -> M11_200ms
    """
    base = os.path.basename(exo_filename)
    m = re.search(r"(MS?\d+_\d+ms)", base)
    if not m:
        return os.path.splitext(base)[0]
    token = m.group(1)
    if token.startswith("MS"):
        token = "M" + token[2:]   # MS11_200ms -> M11_200ms
    return token


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


# --------------------------
# Exodus name decoding
# --------------------------
def _decode_exodus_name_array(raw_names):
    clean = []
    for name in raw_names:
        if isinstance(name, (bytes, bytearray)):
            s = name.decode("utf-8", errors="ignore")
        else:
            try:
                s = b"".join(name).decode("utf-8", errors="ignore")
            except Exception:
                s = str(name)
        clean.append(s.replace("\x00", "").strip())
    return clean


def build_var_map(nc, name_key):
    """
    name_key: "name_nod_var" or "name_elem_var"
    returns dict[name] = 1-based index
    """
    raw = nc.variables[name_key][:]
    names = _decode_exodus_name_array(raw)
    return {n: (i + 1) for i, n in enumerate(names)}


def to_ns(time_whole):
    """
    If time is in seconds (ends ~6.3e-8), convert to ns.
    If already in ns (ends ~63), keep.
    """
    if len(time_whole) == 0:
        return time_whole
    t_end = float(time_whole[-1])
    if t_end < 1e-3:
        return time_whole * 1e9
    return time_whole


# --------------------------
# Element connectivity / mapping element->node
# --------------------------
def read_all_elements_connectivity(nc):
    num_nodes = int(len(nc.variables["coordx"][:]))

    if "num_el_blk" in nc.dimensions:
        nblk = int(nc.dimensions["num_el_blk"].size)
    else:
        nblk = int(nc.variables["eb_status"].shape[0])

    elem_to_nodes = []
    node_to_elems = [[] for _ in range(num_nodes)]

    elem_global = 0
    for b in range(1, nblk + 1):
        conn_name = f"connect{b}"
        if conn_name not in nc.variables:
            continue

        conn = np.asarray(nc.variables[conn_name][:], dtype=np.int64)
        conn0 = conn - 1  # 1-based -> 0-based

        for e_local in range(conn0.shape[0]):
            nodes = conn0[e_local].tolist()
            elem_to_nodes.append(nodes)
            for nid in nodes:
                if 0 <= nid < num_nodes:
                    node_to_elems[nid].append(elem_global)
            elem_global += 1

    num_elem = len(elem_to_nodes)
    return elem_to_nodes, node_to_elems, nblk, num_elem


def read_elem_var_global(nc, elem_var_index_1based, t_idx, nblk):
    parts = []
    for b in range(1, nblk + 1):
        vname = f"vals_elem_var{elem_var_index_1based}eb{b}"
        if vname not in nc.variables:
            continue
        arr = np.asarray(nc.variables[vname][t_idx], dtype=np.float32)
        parts.append(arr)

    if not parts:
        raise KeyError(
            f"Could not find any element-var arrays for var index {elem_var_index_1based} at t={t_idx}."
        )
    return np.concatenate(parts, axis=0)


def elem_to_node_average(elem_values, node_to_elems):
    num_nodes = len(node_to_elems)
    out = np.zeros(num_nodes, dtype=np.float32)
    for nid, elist in enumerate(node_to_elems):
        if not elist:
            continue
        out[nid] = float(np.mean(elem_values[elist]))
    return out


def read_nodal_var_at_t(nc, nod_var_index_1based, t_idx):
    return np.asarray(nc.variables[f"vals_nod_var{nod_var_index_1based}"][t_idx], dtype=np.float32)


# --------------------------
# Right-edge strip detection / stop criterion
# --------------------------
def get_right_strip_node_indices(x, strip_fraction=0.01):
    """
    Return indices of nodes in the last strip_fraction of the domain in x.
    Example: strip_fraction=0.01 -> last 1% of domain.
    """
    x = np.asarray(x, dtype=np.float64)
    x_min = float(np.min(x))
    x_max = float(np.max(x))
    Lx = x_max - x_min

    if Lx <= 0.0:
        raise RuntimeError("Domain length in x is zero or invalid.")



    x_cut = x_max - strip_fraction * Lx
    idx = np.where(x >= x_cut)[0]
    return idx, x_min, x_max, x_cut, Lx


def rhs_strip_has_been_hit(pressure_n, rhs_strip_idx, pressure_threshold=-0.1, hit_fraction=0.20):
    """
    Stop once at least `hit_fraction` of nodes in the right-edge strip
    have pressure below `pressure_threshold`.

    Example:
      pressure_threshold = -0.1
      hit_fraction = 0.20
    """
    if len(rhs_strip_idx) == 0:
        return False, 0.0, 0, 0, 0.0, 0.0

    rhs_vals = np.asarray(pressure_n[rhs_strip_idx], dtype=np.float64)
    below = rhs_vals < pressure_threshold
    n_hit = int(np.count_nonzero(below))
    n_total = int(rhs_vals.size)
    frac_hit = n_hit / n_total if n_total > 0 else 0.0
    min_rhs = float(np.min(rhs_vals)) if n_total > 0 else 0.0
    mean_rhs = float(np.mean(rhs_vals)) if n_total > 0 else 0.0

    stop_now = frac_hit >= hit_fraction
    return stop_now, frac_hit, n_hit, n_total, min_rhs, mean_rhs


# --------------------------
# Conversion
# --------------------------
def convert_exodus_file(
    exo_path,
    out_root,
    max_steps=630,
    print_every=10,
    stride=1,
    rhs_strip_fraction=0.01,
    pressure_threshold=-0.1,
    hit_fraction=0.20,
):
    """
    stride:
      1 -> write every timestep
      10 -> write timesteps 0,10,20,...

    Stops exporting BEFORE the first timestep where at least `hit_fraction`
    of nodes in the last `rhs_strip_fraction` of the domain have pressure
    below `pressure_threshold`.
    """
    if stride < 1:
        raise ValueError(f"stride must be >= 1, got {stride}")

    sim_id = extract_sim_id(exo_path)
    sim_out_dir = os.path.join(out_root, sim_id)
    ensure_dir(sim_out_dir)

    print(f"\n=== Converting simulation: {sim_id}", flush=True)
    print(f"    Exodus: {exo_path}", flush=True)
    print(f"    Output: {sim_out_dir}", flush=True)
    print(f"    stride: {stride}", flush=True)
    print(f"    rhs_strip_fraction: {rhs_strip_fraction}", flush=True)
    print(f"    pressure_threshold: {pressure_threshold}", flush=True)
    print(f"    hit_fraction: {hit_fraction}", flush=True)

    nc = netCDF4.Dataset(exo_path, "r")
    try:
        x = np.asarray(nc.variables["coordx"][:], dtype=np.float32)
        y = np.asarray(nc.variables["coordy"][:], dtype=np.float32)
        num_nodes = len(x)

        time_whole = np.asarray(nc.variables["time_whole"][:], dtype=np.float32)
        time_ns = to_ns(time_whole)
        n_steps = min(int(len(time_ns)), int(max_steps))

        nod_map = build_var_map(nc, "name_nod_var")
        elem_map = build_var_map(nc, "name_elem_var")

        disp_x_idx = nod_map.get("disp_x", None)
        disp_y_idx = nod_map.get("disp_y", None)
        c_idx = nod_map.get("c", None)

        pressure_idx = elem_map.get("pressure", None)
        vonmises_idx = elem_map.get("vonmises", None)
        gc_idx = elem_map.get("Gc", None)
        a_pos_idx = elem_map.get("a_pos", None)

        # Fallback variables for SED if a_pos is not available
        s_xx_idx = elem_map.get("stress_xx", None)
        s_yy_idx = elem_map.get("stress_yy", None)
        s_zz_idx = elem_map.get("stress_zz", None)
        s_xy_idx = elem_map.get("stress_xy", None)
        s_yz_idx = elem_map.get("stress_yz", None)
        s_xz_idx = elem_map.get("stress_xz", None)

        e_xx_idx = elem_map.get("mechstrain_xx", None)
        e_yy_idx = elem_map.get("mechstrain_yy", None)
        e_zz_idx = elem_map.get("mechstrain_zz", None)
        e_xy_idx = elem_map.get("mechstrain_xy", None)
        e_yz_idx = elem_map.get("mechstrain_yz", None)
        e_xz_idx = elem_map.get("mechstrain_xz", None)

        use_a_pos = a_pos_idx is not None

        missing = []
        if disp_x_idx is None or disp_y_idx is None:
            missing.append("nodal: disp_x, disp_y")
        if c_idx is None:
            missing.append("nodal: c")
        if pressure_idx is None:
            missing.append("elem: pressure")
        if vonmises_idx is None:
            missing.append("elem: vonmises")
        if gc_idx is None:
            missing.append("elem: Gc")

        # Only require stress/strain components if a_pos is absent
        if not use_a_pos:
            sed_missing = []
            for nm, idx in [
                ("stress_xx", s_xx_idx), ("stress_yy", s_yy_idx), ("stress_zz", s_zz_idx),
                ("stress_xy", s_xy_idx), ("stress_yz", s_yz_idx), ("stress_xz", s_xz_idx),
                ("mechstrain_xx", e_xx_idx), ("mechstrain_yy", e_yy_idx), ("mechstrain_zz", e_zz_idx),
                ("mechstrain_xy", e_xy_idx), ("mechstrain_yz", e_yz_idx), ("mechstrain_xz", e_xz_idx),
            ]:
                if idx is None:
                    sed_missing.append(nm)

            if sed_missing:
                missing.append(
                    "Need either elem: a_pos OR all SED components. Missing SED components: "
                    + ", ".join(sed_missing)
                )

        if missing:
            raise KeyError(
                "Missing required variables:\n  - " + "\n  - ".join(missing)
            )

        elem_to_nodes, node_to_elems, nblk, num_elem = read_all_elements_connectivity(nc)
        if num_elem == 0:
            raise RuntimeError("No elements found (connect# missing). Can't map element vars to nodes.")

        rhs_strip_idx, x_min, x_max, x_cut, Lx = get_right_strip_node_indices(
            x,
            strip_fraction=rhs_strip_fraction,
        )

        if len(rhs_strip_idx) == 0:
            raise RuntimeError(
                "Could not identify any nodes in the right-edge strip. "
                "Try increasing rhs_strip_fraction."
            )

        print(
            f"    Detected {len(rhs_strip_idx)} nodes in right-edge strip "
            f"(x in [{x_cut:.6g}, {x_max:.6g}], strip = last {100.0*rhs_strip_fraction:.2f}% of domain, Lx={Lx:.6g})",
            flush=True
        )

        written_steps = 0
        stop_t_idx = None

        for t_idx in range(0, n_steps, stride):
            if t_idx % print_every == 0:
                print(f"  [{sim_id}] timestep {t_idx}/{n_steps-1}  (time={time_ns[t_idx]:.3f} ns)", flush=True)

            ux = read_nodal_var_at_t(nc, disp_x_idx, t_idx)
            uy = read_nodal_var_at_t(nc, disp_y_idx, t_idx)
            c = read_nodal_var_at_t(nc, c_idx, t_idx)
            fracture_mask = (c > 0.95).astype(np.uint8)

            pressure_e = read_elem_var_global(nc, pressure_idx, t_idx, nblk)
            vonmises_e = read_elem_var_global(nc, vonmises_idx, t_idx, nblk)
            gc_e = read_elem_var_global(nc, gc_idx, t_idx, nblk)

            pressure_n = elem_to_node_average(pressure_e, node_to_elems)

            stop_now, frac_hit, n_hit, n_total, min_rhs, mean_rhs = rhs_strip_has_been_hit(
                pressure_n,
                rhs_strip_idx,
                pressure_threshold=pressure_threshold,
                hit_fraction=hit_fraction,
            )

            if stop_now:
                stop_t_idx = t_idx
                print(
                    f"    STOP: right-edge strip criterion met at timestep {t_idx} "
                    f"(time={time_ns[t_idx]:.3f} ns).",
                    flush=True
                )
                print(
                    f"    {n_hit}/{n_total} nodes ({100.0*frac_hit:.2f}%) in last "
                    f"{100.0*rhs_strip_fraction:.2f}% of domain had pressure < {pressure_threshold:.6g}.",
                    flush=True
                )
                print(
                    f"    Strip stats: min={min_rhs:.6e}, mean={mean_rhs:.6e}",
                    flush=True
                )
                print(
                    "    This timestep and all later timesteps were NOT written.",
                    flush=True
                )
                break

            vonmises_n = elem_to_node_average(vonmises_e, node_to_elems)
            gc_n = elem_to_node_average(gc_e, node_to_elems)

            t_ns = float(time_ns[t_idx])

            df_dict = {
                "x": x,
                "y": y,
                "ux": ux,
                "uy": uy,
                "Gc": gc_n,
                "pressure": pressure_n,
                "vonmises": vonmises_n,
                "fracture_mask": fracture_mask,
                "time_ns": np.full(num_nodes, t_ns, dtype=np.float32),
            }

            if use_a_pos:
                a_pos_e = read_elem_var_global(nc, a_pos_idx, t_idx, nblk)
                a_pos_n = elem_to_node_average(a_pos_e, node_to_elems)
                df_dict["a_pos"] = a_pos_n
            else:
                s_xx = read_elem_var_global(nc, s_xx_idx, t_idx, nblk)
                s_yy = read_elem_var_global(nc, s_yy_idx, t_idx, nblk)
                s_zz = read_elem_var_global(nc, s_zz_idx, t_idx, nblk)
                s_xy = read_elem_var_global(nc, s_xy_idx, t_idx, nblk)
                s_yz = read_elem_var_global(nc, s_yz_idx, t_idx, nblk)
                s_xz = read_elem_var_global(nc, s_xz_idx, t_idx, nblk)

                e_xx = read_elem_var_global(nc, e_xx_idx, t_idx, nblk)
                e_yy = read_elem_var_global(nc, e_yy_idx, t_idx, nblk)
                e_zz = read_elem_var_global(nc, e_zz_idx, t_idx, nblk)
                e_xy = read_elem_var_global(nc, e_xy_idx, t_idx, nblk)
                e_yz = read_elem_var_global(nc, e_yz_idx, t_idx, nblk)
                e_xz = read_elem_var_global(nc, e_xz_idx, t_idx, nblk)

                sed_e = 0.5 * (
                    s_xx * e_xx
                    + s_yy * e_yy
                    + s_zz * e_zz
                    + 2.0 * s_xy * e_xy
                    + 2.0 * s_yz * e_yz
                    + 2.0 * s_xz * e_xz
                )
                sed_n = elem_to_node_average(sed_e, node_to_elems)
                df_dict["SED"] = sed_n

            df = pd.DataFrame(df_dict)

            csv_name = f"{sim_id}_t{t_idx:04d}_{t_ns:.2f}ns.csv"
            df.to_csv(os.path.join(sim_out_dir, csv_name), index=False)
            written_steps += 1

    finally:
        nc.close()

    if stop_t_idx is None:
        print(f"=== Done: {sim_id} (wrote {written_steps} CSVs; RHS strip criterion never triggered)", flush=True)
    else:
        print(f"=== Done: {sim_id} (wrote {written_steps} CSVs; stopped before timestep {stop_t_idx})", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Convert selected Exodus sims to per-timestep CSVs (2D).")
    parser.add_argument("--exo_root", required=True, help="Folder containing exodus files (multiple simulations).")
    parser.add_argument("--out_root", required=True, help="Folder to write output simulation subfolders into.")
    parser.add_argument("--max_steps", type=int, default=630, help="Max timesteps to export (default 630).")
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Export every Nth timestep (default 1 = export all)."
    )
    parser.add_argument("--print_every", type=int, default=10, help="Print progress every N timesteps (default 10).")
    parser.add_argument(
        "--patterns",
        nargs="+",
        default=["*.e", "*.exo", "*.exodus", "*.e.*"],
        help="Glob patterns for Exodus files."
    )

    parser.add_argument(
        "--rhs_strip_fraction",
        type=float,
        default=0.01,
        help="Fraction of domain at right edge to inspect (default 0.01 = last 1%% of domain)."
    )
    parser.add_argument(
        "--pressure_threshold",
        type=float,
        default=-0.1,
        help="Pressure threshold for 'shock reached RHS strip' (default -0.1)."
    )
    parser.add_argument(
        "--hit_fraction",
        type=float,
        default=0.20,
        help="Stop when at least this fraction of RHS-strip nodes are below pressure_threshold (default 0.20)."
    )

    args = parser.parse_args()

    ensure_dir(args.out_root)

    all_files = []
    for pat in args.patterns:
        all_files.extend(glob.glob(os.path.join(args.exo_root, "**", pat), recursive=True))
    all_files = sorted(set(all_files))

    exo_files = [p for p in all_files if should_convert_exo(p)]

    print(f"Found {len(all_files)} total exodus files under exo_root.", flush=True)
    print(f"Keeping {len(exo_files)} after filtering for EXACT target basenames.", flush=True)

    found_basenames = {os.path.basename(p) for p in exo_files}
    missing = sorted(TARGET_BASENAMES - found_basenames)
    if missing:
        print("\nWARNING: These target files were not found under exo_root:", flush=True)
        for m in missing:
            print(f"  - {m}", flush=True)
        print("", flush=True)

    by_base = {}
    for p in exo_files:
        b = os.path.basename(p)
        by_base.setdefault(b, []).append(p)

    chosen = []
    for b, paths in by_base.items():
        if len(paths) > 1:
            paths_sorted = sorted(paths, key=lambda s: (len(s), s))
            print(f"WARNING: Multiple matches for {b}. Using:", flush=True)
            print(f"  {paths_sorted[0]}", flush=True)
            for extra in paths_sorted[1:]:
                print(f"  (skipping) {extra}", flush=True)
            chosen.append(paths_sorted[0])
        else:
            chosen.append(paths[0])

    chosen = sorted(chosen)
    for p in chosen:
        ms, vel = parse_ms_and_vel_from_name(p)
        print(f"  - {os.path.basename(p)}   ({ms}, {vel} ms)", flush=True)

    if not chosen:
        raise FileNotFoundError("No matching exodus files after filtering. Check exo_root and filenames.")

    for exo_path in chosen:
        convert_exodus_file(
            exo_path,
            args.out_root,
            max_steps=args.max_steps,
            print_every=args.print_every,
            stride=args.stride,
            rhs_strip_fraction=args.rhs_strip_fraction,
            pressure_threshold=args.pressure_threshold,
            hit_fraction=args.hit_fraction,
        )


if __name__ == "__main__":
    main()
