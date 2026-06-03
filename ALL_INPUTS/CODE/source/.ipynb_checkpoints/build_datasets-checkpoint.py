# source/build_datasets.py
from pathlib import Path
from source.import_datasets import csv_dataset_from_directory


# -------------------------------------------------------------------------
# Test case folders currently in DATA_ROOT. If more test cases are added, 
# modify the TEST_CASE_FOLDERS list.
# -------------------------------------------------------------------------
TEST_CASE_FOLDERS = {
    # Existing MS206 velocity tests
    "test_MS206_V100": "F_MS206_V100_out",
    "test_MS206_V200": "F_MS206_V200_out",
    "test_MS206_V400": "F_MS206_V400_out",
    "test_MS206_V1000": "F_MS206_V1000_out",

    # MS210 / PBX / MS5 cases
    "test_MS210_V400": "F_MS210_V400_out",
    "test_PBX1_V400_true": "F_PBX1_V400_true",
    "test_PBX_2_V400": "F_PBX_2_V400_out",
    "test_MS5_V150ms_inc": "MS5_V150ms_inc",
    "test_MS5_V400ms": "MS205_V400ms_MS5",

    # Horizontal layer cases
    "test_emergency_horizontal": "F_emergency_horizontal_out",
    "test_horizontal_layers_3": "F_horizontal-layers_3_out",
    "test_horizontal_layers_4": "F_horizontal-layers_4_out",

    # Inclusion cases
    "test_inclusions_1_2": "F_inclusions_1_2_out",
    "test_inclusions_2_2": "F_inclusions_2_2_out",
    "test_inclusions_3_2": "F_inclusions_3_2_out",
    "test_inclusions_true_V400": "F_inclusions_true_V400",
}


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
):
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
    for key, folder_name in TEST_CASE_FOLDERS.items():
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