"""Verify network equivalence and compare training across three paired seeds."""

import tempfile
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import torch

from bikes_data import make_rentals
from keras_model import build_keras_model, copy_torch_weights
from model import RentalsNetwork
from prepare import SEED, PreparedData, make_loaders, prepare_data
from train import REPORTS, RunConfig, fit_network, run_epoch, seed_everything
from train_keras import as_arrays, fit_keras, score_keras

SEEDS = (SEED, SEED + 1, SEED + 2)
RTOL = 1e-5
ATOL = 1e-6


def check_equivalence(data: PreparedData) -> float:
    """Check all validation predictions after copying the untrained weights."""
    seed_everything()
    inputs, _ = as_arrays(data.validation_dataset)
    source = RentalsNetwork(inputs.shape[1])
    target = build_keras_model(inputs.shape[1])
    copy_torch_weights(source, target)
    with torch.inference_mode():
        expected = source(data.validation_dataset.features).numpy().ravel()
    actual = target(inputs, training=False).numpy().ravel()
    np.testing.assert_allclose(actual, expected, rtol=RTOL, atol=ATOL)
    return float(np.abs(actual - expected).max())


def compare_runs(
    data: PreparedData, config: RunConfig, seeds: tuple[int, ...]
) -> pd.DataFrame:
    """Use a fixed split and pair the same training seeds across frameworks."""
    records = []
    with tempfile.TemporaryDirectory() as temp:
        for seed in seeds:
            seed_everything(seed)
            start = perf_counter()
            train, validation, test = make_loaders(data, config.batch_size, seed)
            torch_model = RentalsNetwork(
                data.train_dataset.features.shape[1], config.hidden_sizes
            )
            optimizer = torch.optim.Adam(
                torch_model.parameters(), lr=config.learning_rate
            )
            _, best_epoch = fit_network(
                torch_model, train, validation, optimizer, config.epochs
            )
            seconds = perf_counter() - start
            records.append(
                {
                    "framework": "PyTorch",
                    "seed": seed,
                    "best_epoch": best_epoch,
                    "validation_mae": run_epoch(torch_model, validation)["mae"],
                    "test_mae": run_epoch(torch_model, test)["mae"],
                    "seconds": seconds,
                }
            )
            start = perf_counter()
            keras_model, _, best_epoch = fit_keras(
                data, config, seed, Path(temp) / f"best_{seed}.keras"
            )
            seconds = perf_counter() - start
            records.append(
                {
                    "framework": "Keras",
                    "seed": seed,
                    "best_epoch": best_epoch,
                    "validation_mae": score_keras(keras_model, data.validation_dataset)[
                        "mae"
                    ],
                    "test_mae": score_keras(keras_model, data.test_dataset)["mae"],
                    "seconds": seconds,
                }
            )
    return pd.DataFrame(records)


def main() -> None:
    data = prepare_data(make_rentals())
    difference = check_equivalence(data)
    print(f"Maximum prediction difference after weight transfer: {difference:.9g}")
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "equivalence.txt").write_text(
        f"max_abs_difference={difference:.9g}\nrtol={RTOL}\natol={ATOL}\n",
        encoding="utf-8",
    )
    results = compare_runs(data, RunConfig("main"), SEEDS)
    REPORTS.mkdir(exist_ok=True)
    results.to_csv(REPORTS / "comparison.csv", index=False)
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()
