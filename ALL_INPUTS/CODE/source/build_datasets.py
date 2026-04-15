# source/build_datasets.py
from pathlib import Path
from source.import_datasets import csv_dataset_from_directory

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
    velocity_scale=1.0/1000.0,
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

    train = csv_dataset_from_directory(
        str(data_root / "trainDS"),
        shuffle=True,
        dataset_name="train",
        stats_max_files=stats_max_files_train,
        **common
    )

    test_MS206_V100 = csv_dataset_from_directory(
        str(data_root / "F_MS206_V100_out"),
        shuffle=False,
        dataset_name="F_MS206_V100_out",
        stats_max_files=stats_max_files_test,
        **common
    )

    test_MS206_V200 = csv_dataset_from_directory(
        str(data_root / "F_MS206_V200_out"),
        shuffle=False,
        dataset_name="F_MS206_V200_out",
        stats_max_files=stats_max_files_test,
        **common
    )

    test_MS206_V400 = csv_dataset_from_directory(
        str(data_root / "F_MS206_V400_out"),
        shuffle=False,
        dataset_name="F_MS206_V400_out",
        stats_max_files=stats_max_files_test,
        **common
    )    
    

    test_MS206_V1000 = csv_dataset_from_directory(
        str(data_root / "F_MS206_V1000_out"),
        shuffle=False,
        dataset_name="F_MS206_V1000_out",
        stats_max_files=stats_max_files_test,
        **common
    )

    
    

    testDS_MS0_400_Vert_limRange = csv_dataset_from_directory(
        str(data_root / "F_vert-layers_limRange_out"),
        shuffle=False,
        dataset_name="F_vert-layers_limRange_out",
        stats_max_files=stats_max_files_test,
        **common
    )
    
    
    
    
    
    
    

    testDS_F_horizontal_layers_3_out = csv_dataset_from_directory(
        str(data_root / "F_horizontal-layers_3_out"),
        shuffle=False,
        dataset_name="testDS_F_horizontal-layers_3_out",
        stats_max_files=stats_max_files_test,
        **common
    )

    testDS_F_horizontal_layers_4_out = csv_dataset_from_directory(
        str(data_root / "F_horizontal-layers_4_out"),
        shuffle=False,
        dataset_name="testDS_F_horizontal-layers_4_out",
        stats_max_files=stats_max_files_test,
        **common
    )


    testDS_F_inclusions_2_2_out = csv_dataset_from_directory(
        str(data_root / "F_inclusions_2_2_out"),
        shuffle=False,
        dataset_name="testDS_F_inclusions_2_2_out",
        stats_max_files=stats_max_files_test,
        **common
    )

    testDS_F_inclusions_3_2_out = csv_dataset_from_directory(
        str(data_root / "F_inclusions_3_2_out"),
        shuffle=False,
        dataset_name="testDS_F_inclusions_3_2_out",
        stats_max_files=stats_max_files_test,
        **common
    )
     

    test_MS5_150ms_inc = csv_dataset_from_directory(
        str(data_root / "MS5_V150ms_inc"),
        shuffle=False,
        dataset_name="MS5_150ms_inc",
        stats_max_files=stats_max_files_test,
        **common
    )
    
    test_MS5_V400ms = csv_dataset_from_directory(
        str(data_root / "MS205_V400ms_MS5"),
        shuffle=False,
        dataset_name="MS5_V400ms",
        stats_max_files=stats_max_files_test,
        **common
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
        
        "testDS_F_inclusions_2_2_out": testDS_F_inclusions_2_2_out,
        "testDS_F_inclusions_3_2_out": testDS_F_inclusions_3_2_out,

        
        "test_MS5_150ms_inc": test_MS5_150ms_inc,    
        "test_MS5_V400ms": test_MS5_V400ms
        
        
    }