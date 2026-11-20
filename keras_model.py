"""Build the Keras counterpart and copy a PyTorch network's parameters."""

import os
from collections.abc import Sequence

from torch import nn

from model import RentalsNetwork

os.environ["KERAS_BACKEND"] = "tensorflow"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "1")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")

import keras  # noqa: E402


def build_keras_model(
    input_size: int, hidden_sizes: Sequence[int] = (64, 32)
) -> keras.Sequential:
    """Match the PyTorch network: ReLU hidden layers and a linear output."""
    if input_size < 1 or len(hidden_sizes) < 2 or min(hidden_sizes) < 1:
        raise ValueError("Use positive widths and at least two hidden layers.")
    return keras.Sequential(
        [
            keras.Input(shape=(input_size,)),
            *(keras.layers.Dense(width, activation="relu") for width in hidden_sizes),
            keras.layers.Dense(1),
        ],
        name="rentals_network",
    )


def copy_torch_weights(source: RentalsNetwork, target: keras.Sequential) -> None:
    """Transpose each linear kernel and keep biases in output order."""
    linear = [layer for layer in source.layers if isinstance(layer, nn.Linear)]
    dense = [layer for layer in target.layers if isinstance(layer, keras.layers.Dense)]
    for left, right in zip(linear, dense, strict=True):
        right.set_weights(
            [left.weight.detach().cpu().numpy().T, left.bias.detach().cpu().numpy()]
        )
