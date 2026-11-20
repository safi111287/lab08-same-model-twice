"""Test architecture, shared inputs, and the fitted Keras model."""

from dataclasses import replace

import numpy as np
import pytest
import torch

from compare import ATOL, RTOL, check_equivalence
from keras_model import build_keras_model, keras
from model import RentalsNetwork
from prepare import RentalsDataset, make_loaders
from train import RunConfig
from train_keras import as_arrays, batches, fit_keras, score_keras


def test_one_prediction_per_row():
    assert build_keras_model(8)(np.zeros((11, 8), dtype=np.float32)).shape == (11, 1)


@pytest.mark.parametrize("hidden", [(4, 3), (64, 32), (13, 9, 5)])
def test_parameter_count_matches_pytorch(hidden):
    source = RentalsNetwork(8, hidden)
    expected = sum(p.numel() for p in source.parameters())
    assert build_keras_model(8, hidden).count_params() == expected


def test_transposed_weights_agree_on_validation_rows(prepared):
    difference = check_equivalence(prepared)
    assert difference < 1e-5


def test_shared_rows_match_loader_tensors(prepared):
    loaders = make_loaders(prepared, batch_size=13)
    for partition, loader in zip(("train", "validation", "test"), loaders, strict=True):
        dataset = getattr(prepared, f"{partition}_dataset")
        assert loader.dataset is dataset
        keras_batches = list(batches(dataset, 13).as_numpy_iterator())
        for column, torch_column in enumerate((dataset.features, dataset.targets)):
            np.testing.assert_array_equal(
                np.concatenate([batch[column] for batch in keras_batches]),
                torch_column.numpy(),
            )


def test_memorizes_small_batch(prepared, tmp_path):
    x = np.arange(8, dtype=np.float32).reshape(-1, 1) / 8
    y = 1 + 2 * x
    data = RentalsDataset(x, y)
    mini = replace(prepared, train_dataset=data, validation_dataset=data)
    config = RunConfig(
        "batch", hidden_sizes=(16, 8), learning_rate=0.03, epochs=100, batch_size=8
    )
    model, _, _ = fit_keras(mini, config, 7, tmp_path / "batch.keras")
    assert score_keras(model, data)["mae"] < 0.06


@pytest.fixture
def short_config():
    return RunConfig("short", hidden_sizes=(8, 4), epochs=3)


def test_saved_model_predicts_identically(prepared, short_config, tmp_path):
    path = tmp_path / "saved.keras"
    model, _, _ = fit_keras(prepared, short_config, 42, path)
    loaded = keras.models.load_model(path)
    features, _ = as_arrays(prepared.validation_dataset)
    np.testing.assert_allclose(
        model(features, training=False).numpy(),
        loaded(features, training=False).numpy(),
        rtol=RTOL,
        atol=ATOL,
    )


def test_restores_best_weights_when_last_epoch_is_worse(prepared, tmp_path):
    x = np.zeros((8, 1), dtype=np.float32)
    data = replace(
        prepared,
        train_dataset=RentalsDataset(x, np.ones(8)),
        validation_dataset=RentalsDataset(x, np.zeros(8)),
    )
    config = RunConfig(
        "restore", hidden_sizes=(4, 2), learning_rate=0.03, epochs=5, batch_size=8
    )
    model, history, best = fit_keras(data, config, 11, tmp_path / "best.keras")
    assert len(history) == config.epochs
    assert best < config.epochs
    best_loss = history["validation_loss"].min()
    assert history["validation_loss"].iloc[-1] > best_loss
    assert score_keras(model, data.validation_dataset)["loss"] == pytest.approx(
        best_loss, abs=1e-7
    )


def test_short_run_repeats_with_the_same_seed(prepared, short_config, tmp_path):
    model_a, history_a, best_a = fit_keras(
        prepared, short_config, 12, tmp_path / "a.keras"
    )
    model_b, history_b, best_b = fit_keras(
        prepared, short_config, 12, tmp_path / "b.keras"
    )
    assert best_a == best_b
    np.testing.assert_allclose(history_a, history_b, rtol=RTOL, atol=ATOL)
    features, _ = as_arrays(prepared.validation_dataset)
    np.testing.assert_allclose(
        model_a(features, training=False).numpy(),
        model_b(features, training=False).numpy(),
        rtol=RTOL,
        atol=ATOL,
    )


def test_weight_copy_detects_non_transposed_square_kernel(prepared):
    from keras_model import copy_torch_weights

    size = prepared.validation_dataset.features.shape[1]
    source = RentalsNetwork(size, (size, size))
    target = build_keras_model(size, (size, size))
    copy_torch_weights(source, target)
    x, _ = as_arrays(prepared.validation_dataset)
    with torch.inference_mode():
        expected = source(torch.from_numpy(x)).numpy()
    np.testing.assert_allclose(target(x).numpy(), expected, atol=ATOL, rtol=RTOL)
    first = target.layers[0]
    kernel, bias = first.get_weights()
    first.set_weights([kernel.T, bias])
    assert not np.allclose(target(x).numpy(), expected, atol=ATOL, rtol=RTOL)
