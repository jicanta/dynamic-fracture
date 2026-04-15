from __future__ import annotations

from typing import Tuple
import tensorflow as tf


def split_train_val_by_batches(
    train_dataset: tf.data.Dataset,
    *,
    validation_fraction: float = 0.1,
    min_val_batches: int = 1,
    verbose: bool = True,
) -> Tuple[tf.data.Dataset, tf.data.Dataset, int, int]:
    """
    Splits a tf.data.Dataset into (train, val) by number of BATCHES.

    Returns:
        train_ds
        val_ds
        total_batches
        val_batches
    """

    if not (0.0 < validation_fraction < 1.0):
        raise ValueError(
            f"validation_fraction must be in (0,1). Got {validation_fraction}"
        )

    total_batches = train_dataset.reduce(
        tf.constant(0, dtype=tf.int64),
        lambda acc, _: acc + 1
    ).numpy()

    val_batches = max(min_val_batches, int(total_batches * validation_fraction))

    val_ds = train_dataset.take(val_batches)
    train_ds = train_dataset.skip(val_batches)

    if verbose:
        print(
            "[split_train_val_by_batches] "
            f"total_batches={total_batches}  "
            f"val_batches={val_batches}  "
            f"train_batches={total_batches - val_batches}"
        )

    return train_ds, val_ds, int(total_batches), int(val_batches)
