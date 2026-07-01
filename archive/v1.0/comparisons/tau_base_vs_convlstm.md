# tau_base vs original ConvLSTM (micro F1, full AR rollout)

Only cases present in both pipelines. `extra: none` runs are
input-equivalent to old BASE; SED/VON/PRESSURE had an extra channel.

| Case | Old mode | Old F1 | New F1 | ΔF1 | Old P/R | New P/R |
|---|---|---:|---:|---:|---|---|
| test_MS206_V100 | SED | 0.0000 | 0.0000 | +0.0000 | 0.00/0.00 | 0.00/0.00 |
| test_MS206_V200 | SED | 0.5205 | 0.0000 | -0.5205 | 1.00/0.35 | 0.00/0.00 |
| test_MS206_V400 | SED | 0.6907 | 0.7306 | +0.0399 | 1.00/0.53 | 0.94/0.60 |
| test_MS206_V1000 | BASE | 0.6622 | 0.8033 | +0.1411 | 0.94/0.51 | 0.98/0.68 |
| test_MS206_V1000 | SED | 0.9395 | 0.8033 | -0.1362 | 0.90/0.98 | 0.98/0.68 |
| test_MS5_V150ms_inc | BASE | 0.4290 | 0.0000 | -0.4290 | 0.93/0.28 | 0.00/0.00 |
| test_MS5_V400ms | VON | 0.6201 | 0.7984 | +0.1783 | 0.57/0.68 | 0.96/0.68 |
| test_MS5_V400ms | PRESSURE | 0.8947 | 0.7984 | -0.0963 | 0.90/0.89 | 0.96/0.68 |
| test_horizontal_layers_4 | BASE | 0.7387 | 0.4684 | -0.2703 | 0.91/0.62 | 0.95/0.31 |
| test_horizontal_layers_4 | SED | 0.7750 | 0.4684 | -0.3067 | 0.85/0.71 | 0.95/0.31 |
| test_inclusions_1_2 | BASE | 0.5650 | 0.6934 | +0.1283 | 0.52/0.62 | 0.94/0.55 |
| test_inclusions_1_2 | SED | 0.5969 | 0.6934 | +0.0965 | 0.52/0.70 | 0.94/0.55 |
