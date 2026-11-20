
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
