# tau_sed vs original ConvLSTM (micro F1, full AR rollout)

Only cases present in both pipelines. `extra: none` runs are
input-equivalent to old BASE; SED/VON/PRESSURE had an extra channel.

| Case | Old mode | Old F1 | New F1 | ΔF1 | Old P/R | New P/R |
|---|---|---:|---:|---:|---|---|
| test_MS206_V100 | SED | 0.0000 | 0.0000 | +0.0000 | 0.00/0.00 | 0.00/0.00 |
| test_MS206_V200 | SED | 0.5205 | 0.9645 | +0.4440 | 1.00/0.35 | 0.99/0.94 |
| test_MS206_V400 | SED | 0.6907 | 0.9804 | +0.2897 | 1.00/0.53 | 0.98/0.98 |
| test_MS206_V1000 | BASE | 0.6622 | 0.9892 | +0.3270 | 0.94/0.51 | 0.99/0.99 |
| test_MS206_V1000 | SED | 0.9395 | 0.9892 | +0.0497 | 0.90/0.98 | 0.99/0.99 |
| test_MS5_V150ms_inc | BASE | 0.4290 | 0.0034 | -0.4256 | 0.93/0.28 | 0.00/1.00 |
| test_MS5_V400ms | VON | 0.6201 | 0.1710 | -0.4491 | 0.57/0.68 | 0.09/1.00 |
| test_MS5_V400ms | PRESSURE | 0.8947 | 0.1710 | -0.7237 | 0.90/0.89 | 0.09/1.00 |
| test_horizontal_layers_4 | BASE | 0.7387 | 0.1097 | -0.6290 | 0.91/0.62 | 0.06/1.00 |
| test_horizontal_layers_4 | SED | 0.7750 | 0.1097 | -0.6654 | 0.85/0.71 | 0.06/1.00 |
| test_inclusions_1_2 | BASE | 0.5650 | 0.1816 | -0.3834 | 0.52/0.62 | 0.10/1.00 |
| test_inclusions_1_2 | SED | 0.5969 | 0.1816 | -0.4153 | 0.52/0.70 | 0.10/1.00 |
