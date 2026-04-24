import time
from tensorflow.keras.callbacks import Callback

class EpochTimeCallback(Callback):
    """Adds epoch_time_sec to logs so CSVLogger writes it as a column."""
    def on_epoch_begin(self, epoch, logs=None):
        self._t0 = time.time()

    def on_epoch_end(self, epoch, logs=None):
        dt = time.time() - self._t0
        if logs is not None:
            logs["epoch_time_sec"] = dt
