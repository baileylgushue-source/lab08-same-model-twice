
import time
from pathlib import Path

import keras
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from keras_model import build_keras_model
from prepare import TARGET, prepare_features, split_data


def train_keras(
    seed=42,
    hidden_size=16,
    batch_size=64,
    learning_rate=0.001,
    epochs=100,
):
    """Train the Keras model using the Lab 07 data preparation."""
    keras.utils.set_random_seed(seed)

    train_df, val_df, test_df = split_data(seed=seed)
    X_train, X_val, X_test, _ = prepare_features(
        train_df, val_df, test_df
    )

    X_train = X_train.astype(np.float32)
    X_val = X_val.astype(np.float32)
    X_test = X_test.astype(np.float32)

    y_train = train_df[TARGET].to_numpy(dtype=np.float32)
    y_val = val_df[TARGET].to_numpy(dtype=np.float32)

    model = build_keras_model(
        input_size=X_train.shape[1],
        hidden_size=hidden_size,
    )

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="mse",
        metrics=["mae"],
    )

    Path("reports").mkdir(exist_ok=True)

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=10,
            restore_best_weights=True,
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=f"reports/best_keras_model_seed_{seed}.keras",
            monitor="val_loss",
            save_best_only=True,
        ),
    ]

    start_time = time.perf_counter()

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1,
    )

    training_seconds = time.perf_counter() - start_time

    plt.figure()
    plt.plot(history.history["loss"], label="Training Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.legend()
    plt.savefig("reports/keras_learning_curves.png")
    plt.close()

    best_epoch = int(np.argmin(history.history["val_loss"])) + 1

    print("Best epoch:", best_epoch)
    print(
        "Best validation MAE:",
        history.history["val_mae"][best_epoch - 1],
    )

    return model, history, training_seconds


if __name__ == "__main__":
    train_keras()
