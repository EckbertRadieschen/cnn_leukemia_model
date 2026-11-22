import keras
import psutil
import time
import os


process = psutil.Process(os.getpid())

def print_memory_usage():

    system_ram = psutil.virtual_memory()
    process_ram = process.memory_info()

    print(f"System RAM: {system_ram.percent:.1f} %")
    print(f"System verfügbar: {system_ram.available / 1024**3:.2f} GB")
    print(f"Python Prozess: {process_ram.rss / 1024**3:.2f} GB")


class EpochInfo(keras.callbacks.Callback):

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_start = time.perf_counter()

    def on_epoch_end(self, epoch, logs=None):
        duration = time.perf_counter() - self.epoch_start

        print(f"\nEpoche {epoch + 1}: ", f"{duration:.2f} Sekunden")
        print_memory_usage()
        print("=" * 60)
        print()


def get_early_stopping(patience=7, min_delta=0.001):
    return keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=patience,
        min_delta=min_delta,
        mode="min",
        restore_best_weights=True,
        verbose=1
    )

def get_checkpoint (checkpoint_path):
    return keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        monitor="val_loss",
        save_best_only=True,
        save_weights_only=False,
        mode="min",
        verbose=1
    )
