EPOCHS = 30

import keras

from model.training_object_classes import EpochInfo


early_stopping = keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=7,
    min_delta=0.001,
    restore_best_weights=True
)

def train_model (untrained_model, train_ds, validation_ds, epochs):
    history = untrained_model.fit(
        train_ds,
        epochs=epochs,
        callbacks=[EpochInfo(), early_stopping],
        validation_data=validation_ds,
        shuffle=False,
        verbose=1
    )

    return history