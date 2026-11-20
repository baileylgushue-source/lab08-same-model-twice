
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from keras_model import build_keras_model
from prepare import TARGET, make_loaders, prepare_features, split_data
from train import run_epoch, train_model
from train_keras import train_keras


def compare_models(seeds=(42, 43, 44)):
    """Compare PyTorch and Keras on the same data splits."""
    results = []
    Path("reports").mkdir(exist_ok=True)

    for seed in seeds:
        print(f"\nComparing models with seed {seed}")

        train_df, val_df, test_df = split_data(seed=seed)

        # PyTorch
        train_loader, val_loader, test_loader, _ = make_loaders(
            train_df,
            val_df,
            test_df,
            batch_size=64,
            seed=seed,
        )

        start = time.perf_counter()

        torch_model, history, best_epoch, _, val_mae = train_model(
            train_loader,
            val_loader,
            epochs=100,
            learning_rate=0.001,
            seed=seed,
        )

        torch_seconds = time.perf_counter() - start

        _, torch_test_mae = run_epoch(
            torch_model,
            test_loader,
            nn.MSELoss(),
        )

        results.append({
            "framework": "PyTorch",
            "seed": seed,
            "best_epoch": best_epoch,
            "validation_mae": val_mae,
            "test_mae": torch_test_mae,
            "seconds": torch_seconds,
        })

        # Keras
        start = time.perf_counter()

        keras_model, keras_history, keras_seconds = train_keras(
            seed=seed,
            hidden_size=16,
            batch_size=64,
            learning_rate=0.001,
            epochs=100,
        )

        best_keras_epoch = (
            int(np.argmin(keras_history.history["val_loss"])) + 1
        )

        keras_val_mae = keras_history.history["val_mae"][
            best_keras_epoch - 1
        ]

        # Use the same split and preparation functions
        _, _, X_test, _ = prepare_features(
            train_df, val_df, test_df
        )

        X_test = X_test.astype(np.float32)
        y_test = test_df[TARGET].to_numpy(dtype=np.float32)

        predictions = keras_model.predict(
            X_test, verbose=0
        ).reshape(-1)

        keras_test_mae = float(
            np.mean(np.abs(predictions - y_test))
        )

        results.append({
            "framework": "Keras",
            "seed": seed,
            "best_epoch": best_keras_epoch,
            "validation_mae": keras_val_mae,
            "test_mae": keras_test_mae,
            "seconds": keras_seconds,
        })

    comparison = pd.DataFrame(results)
    comparison.to_csv("reports/comparison.csv", index=False)

    print("\nComparison Results")
    print(comparison.to_string(index=False))

    return comparison


if __name__ == "__main__":
    compare_models()
