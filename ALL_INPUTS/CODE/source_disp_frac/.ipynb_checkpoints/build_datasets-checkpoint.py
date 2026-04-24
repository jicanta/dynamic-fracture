# source_disp_frac/build_datasets.py
from pathlib import Path

from source_disp_frac.import_datasets import csv_dataset_from_directory


def make_datasets(
    *,
    data_root,
    feature_cols,
    target_col="fracture_mask",
    target_cols=None,
    batch_size=5,
    sequence_length=10,
    drop_first_csv=True,
    print_run_stats=False,
    stats_max_files_train=1,
    stats_max_files_test=None,
    add_velocity=True,
    velocity_scale=1.0 / 1000.0,
):
    """
    Build raw point-wise datasets.

    For the fracture + displacement model:
      target_cols should be:
          ("fracture_mask", "ux", "uy")

    This makes import_datasets.py read all targets from the shifted
    output window:
          frames t+shift ... t+shift+T-1

    So raw yb becomes:
          yb[..., 0] = future fracture_mask
          yb[..., 1] = future ux
          yb[..., 2] = future uy


    """
    data_root = Path(data_root)

    common = dict(
        batch_size=batch_size,
        feature_cols=feature_cols,
        target_col=target_col,
        target_cols=target_cols,
        sequence_length=sequence_length,
        shift=1,
        add_velocity=add_velocity,
        velocity_scale=velocity_scale,
        drop_first_csv=drop_first_csv,
        print_run_stats=print_run_stats,
    )

    train = csv_dataset_from_directory(
        str(data_root / "trainDS"),
        shuffle=True,
        dataset_name="train",
        stats_max_files=stats_max_files_train,
        **common,
    )

    test_MS206_V100 = csv_dataset_from_directory(
        str(data_root / "F_MS206_V100_out"),
        shuffle=False,
        dataset_name="F_MS206_V100_out",
        stats_max_files=stats_max_files_test,
        **common,
    )

    test_MS206_V200 = csv_dataset_from_directory(
        str(data_root / "F_MS206_V200_out"),
        shuffle=False,
        dataset_name="F_MS206_V200_out",
        stats_max_files=stats_max_files_test,
        **common,
    )

    test_MS206_V400 = csv_dataset_from_directory(
        str(data_root / "F_MS206_V400_out"),
        shuffle=False,
        dataset_name="F_MS206_V400_out",
        stats_max_files=stats_max_files_test,
        **common,
    )

    test_MS206_V1000 = csv_dataset_from_directory(
        str(data_root / "F_MS206_V1000_out"),
        shuffle=False,
        dataset_name="F_MS206_V1000_out",
        stats_max_files=stats_max_files_test,
        **common,
    )

    testDS_MS0_400_Vert_limRange = csv_dataset_from_directory(
        str(data_root / "testDS_MS0_400_Vert_limRange"),
        shuffle=False,
        dataset_name="testDS_MS0_400_Vert_limRange",
        stats_max_files=stats_max_files_test,
        **common,
    )

    testDS_F_horizontal_layers_3_out = csv_dataset_from_directory(
        str(data_root / "F_horizontal-layers_3_out"),
        shuffle=False,
        dataset_name="testDS_F_horizontal-layers_3_out",
        stats_max_files=stats_max_files_test,
        **common,
    )

    testDS_F_horizontal_layers_4_out = csv_dataset_from_directory(
        str(data_root / "F_horizontal-layers_4_out"),
        shuffle=False,
        dataset_name="testDS_F_horizontal-layers_4_out",
        stats_max_files=stats_max_files_test,
        **common,
    )

    testDS_F_inclusions_1_2_out = csv_dataset_from_directory(
        str(data_root / "F_inclusions_1_2_out"),
        shuffle=False,
        dataset_name="testDS_F_inclusions_1_2_out",
        stats_max_files=stats_max_files_test,
        **common,
    )

    testDS_F_inclusions_2_2_out = csv_dataset_from_directory(
        str(data_root / "F_inclusions_2_2_out"),
        shuffle=False,
        dataset_name="testDS_F_inclusions_2_2_out",
        stats_max_files=stats_max_files_test,
        **common,
    )

    testDS_F_inclusions_3_2_out = csv_dataset_from_directory(
        str(data_root / "F_inclusions_3_2_out"),
        shuffle=False,
        dataset_name="testDS_F_inclusions_3_2_out",
        stats_max_files=stats_max_files_test,
        **common,
    )

    test_MS5_150ms_inc = csv_dataset_from_directory(
        str(data_root / "MS5_V150ms_inc"),
        shuffle=False,
        dataset_name="MS5_150ms_inc",
        stats_max_files=stats_max_files_test,
        **common,
    )

    test_MS5_V400ms = csv_dataset_from_directory(
        str(data_root / "MS205_V400ms_MS5"),
        shuffle=False,
        dataset_name="MS5_V400ms",
        stats_max_files=stats_max_files_test,
        **common,
    )

    test_F_MS210_V400_out = csv_dataset_from_directory(
        str(data_root / "F_MS210_V400_out"),
        shuffle=False,
        dataset_name="F_MS210_V400_out",
        stats_max_files=stats_max_files_test,
        **common,
    )

    test_F_PBX_1_V400_out = csv_dataset_from_directory(
        str(data_root / "F_PBX_1_V400_out"),
        shuffle=False,
        dataset_name="F_PBX_1_V400_out",
        stats_max_files=stats_max_files_test,
        **common,
    )

    test_F_PBX_2_V400_out = csv_dataset_from_directory(
        str(data_root / "F_PBX_2_V400_out"),
        shuffle=False,
        dataset_name="F_PBX_2_V400_out",
        stats_max_files=stats_max_files_test,
        **common,
    )

    F_inclusions_limited_out = csv_dataset_from_directory(
        str(data_root / "F_inclusions_limited_out"),
        shuffle=False,
        dataset_name="F_inclusions_limited_out",
        stats_max_files=stats_max_files_test,
        **common,
    )

    F_em_horizontal_out = csv_dataset_from_directory(
        str(data_root / "F_emergency_horizontal_out"),
        shuffle=False,
        dataset_name="F_em_horizontal_out",
        stats_max_files=stats_max_files_test,
        **common,
    )

    return {
        "train": train,
        "test_MS206_V100": test_MS206_V100,
        "test_MS206_V200": test_MS206_V200,
        "test_MS206_V400": test_MS206_V400,
        "test_MS206_V1000": test_MS206_V1000,
        "testDS_MS0_400_Vert_limRange": testDS_MS0_400_Vert_limRange,
        "testDS_F_horizontal-layers_3_out": testDS_F_horizontal_layers_3_out,
        "testDS_F_horizontal-layers_4_out": testDS_F_horizontal_layers_4_out,
        "testDS_F_inclusions_1_2_out": testDS_F_inclusions_1_2_out,
        "testDS_F_inclusions_2_2_out": testDS_F_inclusions_2_2_out,
        "testDS_F_inclusions_3_2_out": testDS_F_inclusions_3_2_out,
        "test_MS5_150ms_inc": test_MS5_150ms_inc,
        "test_MS5_V400ms": test_MS5_V400ms,
        "test_F_MS210_V400_out": test_F_MS210_V400_out,
        "test_F_PBX_1_V400_out": test_F_PBX_1_V400_out,
        "test_F_PBX_2_V400_out": test_F_PBX_2_V400_out,
        "F_inclusions_limited_out": F_inclusions_limited_out,
        "F_em_horizontal_out": F_em_horizontal_out,
    }