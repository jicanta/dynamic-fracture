# source/build_datasets.py
import sys
from pathlib import Path
from source.import_datasets import csv_dataset_from_directory


# -------------------------------------------------------------------------
# Test case folders. The canonical map now lives in the framework-free repo-root
# registry (dynamic-fracture/case_registry.py) so this copy can no longer drift
# from new_model/src/config.py (T-02-01). Prepend the repo root to sys.path then
# re-export. The vert-layers alias is intentionally NOT restored (D-04 gate).
# -------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]   # source -> kathleens-model -> dynamic-fracture
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from case_registry import TEST_CASE_FOLDERS, NEW_TEST_CASE_FOLDERS  # noqa: E402  (re-export: canonical 16 + disjoint new roster)


def make_datasets(
    *,
    data_root,
    feature_cols,
    target_col="fracture_mask",
    batch_size=5,
    sequence_length=10,
    drop_first_csv=True,
    print_run_stats=False,
    stats_max_files_train=1,
    stats_max_files_test=None,
    add_velocity=True,
    velocity_scale=1.0 / 1000.0,
    case_set="canonical",
):
    # D-01: select the test roster. "canonical" (default) keeps the frozen v1.0
    # 16-case loop byte-unchanged; "new" iterates the DISJOINT NEW_TEST_CASE_FOLDERS.
    # The two dicts are never merged (D-01) -- one roster per call.
    roster = NEW_TEST_CASE_FOLDERS if case_set == "new" else TEST_CASE_FOLDERS
    data_root = Path(data_root)

    common = dict(
        batch_size=batch_size,
        feature_cols=feature_cols,
        target_col=target_col,
        sequence_length=sequence_length,
        shift=1,
        add_velocity=add_velocity,
        velocity_scale=velocity_scale,
        drop_first_csv=drop_first_csv,
        print_run_stats=print_run_stats,
    )

    # ---------------------------------------------------------------------
    # Training dataset
    # ---------------------------------------------------------------------
    train = csv_dataset_from_directory(
        str(data_root / "trainDS"),
        shuffle=True,
        dataset_name="trainDS",
        stats_max_files=stats_max_files_train,
        **common,
    )

    datasets = {
        "train": train,
    }

    # ---------------------------------------------------------------------
    # Test datasets
    # ---------------------------------------------------------------------
    for key, folder_name in roster.items():
        folder_path = data_root / folder_name

        datasets[key] = csv_dataset_from_directory(
            str(folder_path),
            shuffle=False,
            dataset_name=folder_name,
            stats_max_files=stats_max_files_test,
            **common,
        )

    # ---------------------------------------------------------------------
    # Backward-compatible aliases from older notebooks/source files
    # ---------------------------------------------------------------------
#     datasets["testDS_F_horizontal-layers_3_out"] = datasets["test_horizontal_layers_3"]
#     datasets["testDS_F_horizontal-layers_4_out"] = datasets["test_horizontal_layers_4"]

#     datasets["testDS_F_inclusions_2_2_out"] = datasets["test_inclusions_2_2"]
#     datasets["testDS_F_inclusions_3_2_out"] = datasets["test_inclusions_3_2"]

#     datasets["test_MS5_150ms_inc"] = datasets["test_MS5_V150ms_inc"]

    return datasets