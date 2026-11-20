
import keras
import torch


def build_keras_model(input_size: int = 8, hidden_size: int = 16):
    """Build the same neural network as the PyTorch BikeNet."""
    model = keras.Sequential(
        [
            keras.Input(shape=(input_size,)),
            keras.layers.Dense(hidden_size, activation="relu"),
            keras.layers.Dense(hidden_size, activation="relu"),
            keras.layers.Dense(1),
        ]
    )
    return model


def copy_torch_weights(torch_model, keras_model):
    """Copy PyTorch weights into the matching Keras model."""
    torch_layers = [
        layer
        for layer in torch_model.network
        if isinstance(layer, torch.nn.Linear)
    ]

    keras_layers = [
        layer
        for layer in keras_model.layers
        if isinstance(layer, keras.layers.Dense)
    ]

    for torch_layer, keras_layer in zip(
        torch_layers, keras_layers, strict=True
    ):
        weights = torch_layer.weight.detach().numpy().T
        biases = torch_layer.bias.detach().numpy()

        keras_layer.set_weights([weights, biases])
