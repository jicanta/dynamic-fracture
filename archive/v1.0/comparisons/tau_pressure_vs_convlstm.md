# tau_pressure vs original ConvLSTM (micro F1, full AR rollout)

Only cases present in both pipelines. `extra: none` runs are
input-equivalent to old BASE; SED/VON/PRESSURE had an extra channel.

| Case | Old mode | Old F1 | New F1 | ΔF1 | Old P/R | New P/R |
|---|---|---:|---:|---:|---|---|
| test_MS206_V100 | SED | 0.0000 | 0.0000 | +0.0000 | 0.00/0.00 | 0.00/0.00 |
| test_MS206_V200 | SED | 0.5205 | 0.8130 | +0.2925 | 1.00/0.35 | 0.84/0.79 |
| test_MS206_V400 | SED | 0.6907 | 0.8736 | +0.1829 | 1.00/0.53 | 0.95/0.81 |
| test_MS206_V1000 | BASE | 0.6622 | 0.8632 | +0.2010 | 0.94/0.51 | 0.97/0.78 |
| test_MS206_V1000 | SED | 0.9395 | 0.8632 | -0.0763 | 0.90/0.98 | 0.97/0.78 |
| test_MS5_V150ms_inc | BASE | 0.4290 | 0.0000 | -0.4290 | 0.93/0.28 | 0.00/0.00 |
| test_MS5_V400ms | VON | 0.6201 | 0.8849 | +0.2648 | 0.57/0.68 | 0.99/0.80 |
| test_MS5_V400ms | PRESSURE | 0.8947 | 0.8849 | -0.0098 | 0.90/0.89 | 0.99/0.80 |
| test_horizontal_layers_4 | BASE | 0.7387 | 0.7639 | +0.0252 | 0.91/0.62 | 0.98/0.63 |
| test_horizontal_layers_4 | SED | 0.7750 | 0.7639 | -0.0112 | 0.85/0.71 | 0.98/0.63 |
| test_inclusions_1_2 | BASE | 0.5650 | 0.6533 | +0.0883 | 0.52/0.62 | 0.96/0.49 |
| test_inclusions_1_2 | SED | 0.5969 | 0.6533 | +0.0564 | 0.52/0.70 | 0.96/0.49 |
