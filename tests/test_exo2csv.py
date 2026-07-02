"""Synthetic-``.e`` fixture for the vendored exodus->CSV ingestion tool.

This module lands the Wave-0 test infrastructure for ``tools/exo2csv_final_rev.py``
(vendored verbatim from Kathleen). It builds a small, self-consistent Exodus-II
mesh IN MEMORY (netCDF-4 container) that matches the EXACT layout the vendored
``convert_exodus_file`` reads, so the tool-dependent EXO-02/03/04 + D-07 contract
tests (landed in plan 09-05) can be developed and run entirely off-cluster --
against a synthetic ``.e``, never a real Gilbreth file or GPU.

The ``.e`` -> CSV contract the fixture must satisfy (from the vendored reader):

  * nodal:   ``coordx`` / ``coordy`` (per-node), ``time_whole`` (per-step),
             ``name_nod_var`` char array naming {"disp_x","disp_y","c"} with
             ``vals_nod_var{i}`` shaped (num_time_step, num_nodes), i 1-based.
             ``fracture_mask = (c > 0.95)`` is derived downstream by the tool.
  * element: ``name_elem_var`` char array naming {"pressure","vonmises","Gc"}
             and EITHER {"a_pos"} OR the 12 stress_{...}/mechstrain_{...}
             components; ``vals_elem_var{i}eb{b}`` shaped
             (num_time_step, num_elem_in_blk).
  * mesh:    a ``num_el_blk`` dimension (== 1) + 1-based ``connect{b}``
             (num_elem, nodes_per_elem) quad connectivity so the vendored
             element->node averaging works (every node belongs to >= 1 element).
  * The right-edge-strip ``pressure`` is kept ABOVE the ``-0.1`` STOP threshold
    so, when the tool IS exercised (09-05), all timesteps are written.

Only a GREEN fixture self-test lives here; the tool-dependent contract tests
land in plan 09-05 (which packages the vendored tool + turns them green).

Run:  cd dynamic-fracture && python -m pytest tests/test_exo2csv.py -q
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import List

import netCDF4
import numpy as np
import pandas as pd

# The repo-root sys.path shim (tests/conftest.py) puts ``dynamic-fracture/`` on
# sys.path, so the VENDORED tool's real public API resolves.
from tools.exo2csv_final_rev import build_var_map, convert_exodus_file, extract_sim_id

SEED: int = 42
LEN_NAME: int = 33  # Exodus MAX_STR_LENGTH-style padding for name char arrays

# The 12 stress/strain components the vendored tool falls back to when ``a_pos``
# is absent (order matches exo2csv_final_rev.py's SED assembly).
_STRESS_STRAIN: List[str] = [
    "stress_xx", "stress_yy", "stress_zz", "stress_xy", "stress_yz", "stress_xz",
    "mechstrain_xx", "mechstrain_yy", "mechstrain_zz",
    "mechstrain_xy", "mechstrain_yz", "mechstrain_xz",
]


def _encode_name_array(
    nc: "netCDF4.Dataset",
    var_key: str,
    count_dim: str,
    names: List[str],
    *,
    len_name: int = LEN_NAME,
) -> None:
    """Write an Exodus-style ``(num_var, len_name)`` char array of field names.

    Encoded via ``netCDF4.stringtochar`` (null-padded to ``len_name``) so the
    vendored ``_decode_exodus_name_array`` / ``build_var_map`` decode it: each
    row is a length-``len_name`` array of single bytes, joined + null-stripped
    back to the original name.
    """
    if count_dim not in nc.dimensions:
        nc.createDimension(count_dim, len(names))
    if "len_name" not in nc.dimensions:
        nc.createDimension("len_name", len_name)
    var = nc.createVariable(var_key, "S1", (count_dim, "len_name"))
    # Build the (num_var, len_name) single-byte char array by hand. netCDF4's
    # ``stringtochar`` is incompatible with numpy >= 2.5 (numpy.bytes_ has no
    # ``.encode``), so pack each name's bytes into a null-padded row directly.
    arr = np.zeros((len(names), len_name), dtype="S1")
    for row, name in enumerate(names):
        encoded = np.frombuffer(name.encode("utf-8"), dtype="S1")
        arr[row, : encoded.size] = encoded
    var[:] = arr


def make_synthetic_exo(
    path: str | Path,
    nx: int = 8,
    ny: int = 6,
    n_t: int = 12,
    emit_a_pos: bool = False,
) -> Path:
    """Write a valid NETCDF4 ``.e`` matching the vendored tool's mesh/field layout.

    Args:
      path: destination ``.e`` file path.
      nx, ny: structured-lattice node counts along x / y (num_nodes = nx*ny).
      n_t: number of timesteps written to ``time_whole`` + every ``vals_*`` var.
      emit_a_pos: if True, emit a single ``a_pos`` element var; else emit the 12
        stress/strain components (the tool's SED fallback path).

    Layout written:
      * ``coordx`` / ``coordy``: meshgrid ravel of an ``nx x ny`` structured
        lattice on [0, 1]^2 (row-major node ordering, num_nodes = nx*ny).
      * ``time_whole``: ``arange(n_t)`` (>= 1e-3 scale, so the tool's ``to_ns``
        leaves it as-is).
      * ``name_nod_var`` = ["disp_x","disp_y","c"]; ``vals_nod_var1..3`` shaped
        (n_t, num_nodes). ``c`` is MONOTONE in time (plus an x-gradient) and
        crosses 0.95, so downstream ``fracture_mask = (c > 0.95)`` has both 0s
        and 1s; ``disp_x`` / ``disp_y`` are RAW ~O(1e4) seeded floats.
      * ``name_elem_var`` = ["pressure","vonmises","Gc"] + (["a_pos"] |
        12 stress/strain); ``vals_elem_var{i}eb1`` shaped (n_t, num_elem).
        ``pressure`` is kept strictly POSITIVE so the RHS-strip STOP (pressure
        < -0.1) never triggers.
      * ``num_el_blk`` == 1 + ``connect1`` (num_elem, 4) 1-based quad
        connectivity over the lattice ((nx-1)*(ny-1) quads); every node belongs
        to >= 1 element.

    Returns the written ``Path``.
    """
    path = Path(path)
    rng = np.random.default_rng(SEED)

    # ---- structured lattice (row-major: node id = i*nx + j) ----
    xs = np.linspace(0.0, 1.0, nx)
    ys = np.linspace(0.0, 1.0, ny)
    XX, YY = np.meshgrid(xs, ys)          # shape (ny, nx)
    coordx = XX.ravel().astype(np.float64)
    coordy = YY.ravel().astype(np.float64)
    num_nodes = coordx.size               # nx * ny

    # ---- 1-based quad connectivity over the lattice ----
    quads: List[List[int]] = []
    for i in range(ny - 1):
        for j in range(nx - 1):
            n0 = i * nx + j
            n1 = i * nx + (j + 1)
            n2 = (i + 1) * nx + (j + 1)
            n3 = (i + 1) * nx + j
            quads.append([n0 + 1, n1 + 1, n2 + 1, n3 + 1])   # 1-based
    connect1 = np.asarray(quads, dtype=np.int32)             # (num_elem, 4)
    num_elem = connect1.shape[0]                             # (nx-1)*(ny-1)

    # ---- nodal fields ----
    time_whole = np.arange(n_t, dtype=np.float64)
    # x-gradient in [0, 1] so `c` crosses 0.95 spatially as well as in time
    x_grad = (coordx - coordx.min()) / (coordx.max() - coordx.min())
    c_vals = np.empty((n_t, num_nodes), dtype=np.float64)
    for t in range(n_t):
        ramp = t / max(n_t - 1, 1)                            # 0 -> 1, monotone
        c_vals[t] = np.clip(0.5 * x_grad + 0.8 * ramp, 0.0, 1.2)
    disp_x = (rng.standard_normal((n_t, num_nodes)) * 1.0e4).astype(np.float64)
    disp_y = (rng.standard_normal((n_t, num_nodes)) * 1.0e4).astype(np.float64)

    # ---- element fields (strictly positive pressure => RHS-strip never stops) ----
    pressure = np.abs(rng.standard_normal((n_t, num_elem))) + 1.0      # >= 1.0 > -0.1
    vonmises = np.abs(rng.standard_normal((n_t, num_elem))) + 0.1
    gc = np.abs(rng.standard_normal((n_t, num_elem))) + 0.1

    nc = netCDF4.Dataset(str(path), "w", format="NETCDF4")
    try:
        nc.createDimension("num_nodes", num_nodes)
        nc.createDimension("time_step", n_t)
        nc.createDimension("num_el_blk", 1)
        nc.createDimension("num_el_in_blk1", num_elem)
        nc.createDimension("num_nod_per_el1", connect1.shape[1])

        nc.createVariable("coordx", "f8", ("num_nodes",))[:] = coordx
        nc.createVariable("coordy", "f8", ("num_nodes",))[:] = coordy
        nc.createVariable("time_whole", "f8", ("time_step",))[:] = time_whole

        nc.createVariable("connect1", "i4", ("num_el_in_blk1", "num_nod_per_el1"))[:] = connect1

        # nodal field names + values (1-based var index)
        nod_names = ["disp_x", "disp_y", "c"]
        _encode_name_array(nc, "name_nod_var", "num_nod_var", nod_names)
        for idx, arr in enumerate((disp_x, disp_y, c_vals), start=1):
            nc.createVariable(f"vals_nod_var{idx}", "f8", ("time_step", "num_nodes"))[:] = arr

        # element field names + values
        elem_names = ["pressure", "vonmises", "Gc"]
        elem_arrays = [pressure, vonmises, gc]
        if emit_a_pos:
            elem_names.append("a_pos")
            elem_arrays.append(np.abs(rng.standard_normal((n_t, num_elem))))
        else:
            for nm in _STRESS_STRAIN:
                elem_names.append(nm)
                elem_arrays.append(rng.standard_normal((n_t, num_elem)).astype(np.float64))
        _encode_name_array(nc, "name_elem_var", "num_elem_var", elem_names)
        for idx, arr in enumerate(elem_arrays, start=1):
            nc.createVariable(f"vals_elem_var{idx}eb1", "f8", ("time_step", "num_el_in_blk1"))[:] = arr
    finally:
        nc.close()

    return path


def _decode_names(raw) -> List[str]:
    """Decode an Exodus ``name_*_var`` char array to a list of clean strings.

    Independent of the vendored tool (the tool-dependent asserts land in 09-05):
    joins each row's single-byte chars and strips null padding / whitespace.
    """
    names: List[str] = []
    for row in raw:
        s = b"".join(bytes(ch) for ch in np.asarray(row).ravel()).decode("utf-8", "ignore")
        names.append(s.replace("\x00", "").strip())
    return names


def test_fixture_roundtrips(tmp_path: Path) -> None:
    """The synthetic ``.e`` writes a vendored-layout mesh that round-trips.

    Writes a synthetic ``.e``, reopens it with ``netCDF4.Dataset``, and asserts
    the mesh dimensions and decoded field-name arrays match the layout the
    vendored ``convert_exodus_file`` reads. Does NOT invoke the vendored tool --
    those contract tests land in plan 09-05.
    """
    nx, ny, n_t = 8, 6, 12
    exo_path = tmp_path / "F_SYNTH_V400_out.e"
    make_synthetic_exo(exo_path, nx=nx, ny=ny, n_t=n_t)

    nc = netCDF4.Dataset(str(exo_path), "r")
    try:
        assert len(nc.variables["coordx"][:]) == nx * ny
        assert len(nc.variables["coordy"][:]) == nx * ny
        assert len(nc.variables["time_whole"][:]) == n_t

        # 1-based quad connectivity present for element->node averaging
        connect1 = np.asarray(nc.variables["connect1"][:])
        assert connect1.shape == ((nx - 1) * (ny - 1), 4)
        assert int(connect1.min()) >= 1              # 1-based node ids

        nod_names = set(_decode_names(nc.variables["name_nod_var"][:]))
        assert {"disp_x", "disp_y", "c"}.issubset(nod_names)

        elem_names = set(_decode_names(nc.variables["name_elem_var"][:]))
        assert {"pressure", "vonmises", "Gc"}.issubset(elem_names)
    finally:
        nc.close()


# ===================================================================== #
# 09-05: tool-dependent contract tests against the VENDORED tool.        #
# The exact DATASET schema the vendored tool emits (SED fallback path).  #
# Written compactly (no inter-token spaces) so it is the byte-for-byte   #
# column tuple the tool writes -- this pins T-09-12 (field mis-map).     #
# ===================================================================== #
EXPECTED_SCHEMA: List[str] = [
    "x","y","ux","uy","Gc","pressure","vonmises","fracture_mask","time_ns","SED",
]


def _emit_case(tmp_path: Path, *, name: str = "F_SYNTH_V400_out.e",
               nx: int = 8, ny: int = 6, n_t: int = 12,
               emit_a_pos: bool = False) -> tuple[Path, str, List[Path]]:
    """Build a synthetic ``.e`` and run the VENDORED tool on it.

    Returns ``(exo_path, sim_id, sorted_csv_paths)``. The tool writes one CSV per
    timestep into ``<out_root>/<sim_id>/`` -- the SAME contract the real Gilbreth
    ``.e`` files go through (09-04/09-07), exercised here entirely off-cluster.
    """
    exo_path = tmp_path / name
    make_synthetic_exo(exo_path, nx=nx, ny=ny, n_t=n_t, emit_a_pos=emit_a_pos)
    out_root = tmp_path / "out"
    convert_exodus_file(str(exo_path), str(out_root), print_every=1000)
    sim_id = extract_sim_id(str(exo_path))
    csvs = sorted((out_root / sim_id).glob("*.csv"))
    return exo_path, sim_id, csvs


def test_tool_emits_dataset_schema(tmp_path: Path) -> None:
    """Round-trip a synthetic ``.e`` and pin the tool's exact emitted schema (EXO-02).

    Runs the VENDORED ``convert_exodus_file`` on a small synthetic lattice, then
    reads back the LAST emitted CSV and asserts: the header is EXACTLY the
    DATASET schema (RAW values + already-binarized mask, SED fallback path); the
    ``fracture_mask`` column is 0/1 integers (with >=1 positive, so the
    ``c > 0.95`` binarization is exercised); ``ux`` recovers the RAW written
    nodal displacement (un-scaled -- the tool does NOT apply data.py's 1e-3
    scaling); and the CSV filename matches ``{sim_id}_t\\d{4}_...ns.csv``.
    """
    nx, ny, n_t = 8, 6, 12
    exo_path, sim_id, csvs = _emit_case(tmp_path, nx=nx, ny=ny, n_t=n_t)
    assert csvs, "the vendored tool emitted no CSVs"

    last = csvs[-1]                        # highest t (zero-padded -> lexical sort)
    df = pd.read_csv(last)

    # exact schema (byte-for-byte column tuple, SED fallback path)
    assert list(df.columns) == EXPECTED_SCHEMA

    # fracture_mask already binarized to 0/1 integers, with positives present
    mask = df["fracture_mask"].to_numpy()
    assert set(np.unique(mask)).issubset({0, 1})
    assert np.array_equal(mask, mask.astype(np.int64))     # integral, not float
    assert int(mask.max()) == 1                            # c>0.95 produced 1s

    # ux recovers the RAW nodal disp_x from the .e (un-scaled round-trip)
    t_idx = int(re.search(r"_t(\d{4})_", last.name).group(1))
    nc = netCDF4.Dataset(str(exo_path), "r")
    try:
        nod_map = build_var_map(nc, "name_nod_var")
        raw_disp_x = np.asarray(
            nc.variables[f"vals_nod_var{nod_map['disp_x']}"][t_idx], dtype=np.float32
        )
    finally:
        nc.close()
    assert np.allclose(df["ux"].to_numpy(np.float32), raw_disp_x, rtol=1e-4, atol=1e-2)
    assert np.max(np.abs(raw_disp_x)) > 1.0e3              # O(1e4) RAW, not 1e-3-scaled

    # CSV filename convention
    assert re.fullmatch(rf"{re.escape(sim_id)}_t\d{{4}}_.*ns\.csv", last.name)


def test_emits_all_timesteps(tmp_path: Path) -> None:
    """The tool writes EVERY timestep (RHS-strip STOP never fires on the fixture).

    The synthetic fixture keeps right-edge-strip ``pressure`` strictly positive
    (>= 1.0 > the -0.1 threshold), so the vendored RHS-strip stop criterion never
    triggers and the emitted CSV count equals the fixture's timestep count.
    """
    n_t = 12
    _, _, csvs = _emit_case(tmp_path, n_t=n_t)
    assert len(csvs) == n_t
