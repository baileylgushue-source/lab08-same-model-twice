import keras
import numpy as np
import pytest
import torch

from keras_model import build_keras_model, copy_torch_weights
from model import BikeNet


@pytest.mark.parametrize("hidden_size", [8, 16, 32])
def test_keras_model_output_shape(hidden_size):
    model = build_keras_model(input_size=8, hidden_size=hidden_size)

    predictions = model(np.zeros((5, 8), dtype=np.float32))

    assert predictions.shape == (5, 1)


def test_keras_model_has_three_dense_layers():
    model = build_keras_model(input_size=8, hidden_size=16)

    assert len(model.layers) == 3


def test_keras_model_parameter_count():
    model = build_keras_model(input_size=8, hidden_size=16)

    assert model.count_params() == 433


def test_keras_model_activations():
    model = build_keras_model(input_size=8, hidden_size=16)

    assert model.layers[0].activation.__name__ == "relu"
    assert model.layers[1].activation.__name__ == "relu"
    assert model.layers[2].activation.__name__ == "linear"


def test_copy_torch_weights():
    torch_model = BikeNet(input_size=8, hidden_size=16)
    keras_model = build_keras_model(input_size=8, hidden_size=16)

    copy_torch_weights(torch_model, keras_model)

    x = np.ones((5, 8), dtype=np.float32)

    torch_model.eval()

    with torch.no_grad():
        torch_predictions = torch_model(
            torch.tensor(x)
        ).numpy()

    keras_predictions = keras_model(
        x, training=False
    ).numpy()

    np.testing.assert_allclose(
        torch_predictions.reshape(-1),
        keras_predictions.reshape(-1),
        atol=1e-5,
    )

def test_keras_model_save_and_load(tmp_path):
    model = build_keras_model(input_size=8, hidden_size=16)

    x = np.ones((5, 8), dtype=np.float32)
    original_predictions = model(x, training=False).numpy()

    model_path = tmp_path / "test_model.keras"
    model.save(model_path)

    loaded_model = keras.models.load_model(model_path)
    loaded_predictions = loaded_model(x, training=False).numpy()

    np.testing.assert_allclose(
        original_predictions,
        loaded_predictions,
        atol=1e-5,
    )

def test_keras_small_batch_memorization():
    keras.utils.set_random_seed(42)

    x = np.random.default_rng(42).normal(
        size=(8, 8)
    ).astype(np.float32)

    y = np.sum(x, axis=1).astype(np.float32)

    model = build_keras_model(input_size=8, hidden_size=16)

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.01),
        loss="mse",
    )

    initial_loss = model.evaluate(x, y, verbose=0)

    model.fit(
        x,
        y,
        epochs=300,
        batch_size=8,
        verbose=0,
    )

    final_loss = model.evaluate(x, y, verbose=0)

    assert final_loss < initial_loss
    assert final_loss < 0.01

def test_keras_repeatability():
    x = np.ones((5, 8), dtype=np.float32)

    keras.utils.set_random_seed(42)
    first_model = build_keras_model(input_size=8, hidden_size=16)
    first_predictions = first_model(x, training=False).numpy()

    keras.utils.set_random_seed(42)
    second_model = build_keras_model(input_size=8, hidden_size=16)
    second_predictions = second_model(x, training=False).numpy()

    np.testing.assert_allclose(
        first_predictions,
        second_predictions,
        atol=1e-5,
    )