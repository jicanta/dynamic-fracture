# Package marker for the vendored ingestion tooling.
#
# Makes ``from tools.exo2csv_final_rev import convert_exodus_file, extract_sim_id,
# build_var_map`` resolve under the repo-root sys.path shim (tests/conftest.py).
# The tool itself (exo2csv_final_rev.py) is vendored VERBATIM from Kathleen and is
# NEVER modified here -- see its header. This file only turns the directory into an
# importable package so Phase 9's synthetic-.e contract tests (and Phase 12's reuse)
# can exercise the tool's real public API.
