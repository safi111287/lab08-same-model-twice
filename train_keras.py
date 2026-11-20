"""Train and score the Keras network using the shared bike demand data."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from bikes_data import make_rentals
from keras_model import build_keras_model, keras
from prepare import SEED, PreparedData, RentalsDataset, prepare_data
from train import REPORTS, RunConfig, plot_curves

# isort: split
import tensorflow as tf  # noqa: E402

CHECKPOINTS = REPORTS.parent / "checkpoints"


def as_arrays(dataset: RentalsDataset) -> tuple[np.ndarray, np.ndarray]:
    """Expose the exact float32 inputs used by PyTorch."""
    return dataset.features.numpy(), dataset.targets.numpy()


def batches(
    dataset: RentalsDataset, batch_size: int, seed: int | None = None
) -> tf.data.Dataset:
    features, targets = as_arrays(dataset)
    rows = tf.data.Dataset.from_tensor_slices((features, targets))
    if seed is not None:
        rows = rows.shuffle(len(features), seed=seed, reshuffle_each_iteration=True)
    options = tf.data.Options()
    options.threading.private_threadpool_size = 1
    options.threading.max_intra_op_parallelism = 1
    return rows.batch(batch_size).with_options(options)


def fit_keras(
    data: PreparedData, config: RunConfig, seed: int, checkpoint: Path
) -> tuple[keras.Sequential, pd.DataFrame, int]:
    """Fit for the full epoch budget and restore the minimum-validation-MSE model."""
    keras.backend.clear_session()
    keras.utils.set_random_seed(seed)
    tf.config.experimental.enable_op_determinism()
    with tf.device("/CPU:0"):
        model = build_keras_model(
            data.train_dataset.features.shape[1], config.hidden_sizes
        )
        model.compile(
            optimizer=keras.optimizers.Adam(
                learning_rate=config.learning_rate, epsilon=1e-8
            ),
            loss="mse",
            metrics=[keras.metrics.MeanAbsoluteError(name="mae")],
        )
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        fit = model.fit(
            batches(data.train_dataset, config.batch_size, seed),
            validation_data=batches(data.validation_dataset, config.batch_size),
            epochs=config.epochs,
            callbacks=[
                keras.callbacks.EarlyStopping(
                    monitor="val_loss",
                    patience=config.epochs,
                    restore_best_weights=True,
                ),
                keras.callbacks.ModelCheckpoint(
                    checkpoint, monitor="val_loss", save_best_only=True
                ),
            ],
            verbose=0,
        )
    history = pd.DataFrame(fit.history).rename(
        columns={
            "loss": "train_loss",
            "mae": "train_mae",
            "val_loss": "validation_loss",
            "val_mae": "validation_mae",
        }
    )
    history.insert(0, "epoch", range(1, len(history) + 1))
    best_epoch = int(history["validation_loss"].idxmin()) + 1
    return model, history, best_epoch


def score_keras(model: keras.Sequential, dataset: RentalsDataset) -> dict[str, float]:
    """Score one prediction per station-hour, with no NumPy broadcasting."""
    features, targets = as_arrays(dataset)
    with tf.device("/CPU:0"):
        predictions = model(features, training=False).numpy().ravel()
    return {
        "mae": float(mean_absolute_error(targets.ravel(), predictions)),
        "loss": float(mean_squared_error(targets.ravel(), predictions)),
    }


def main() -> None:
    data = prepare_data(make_rentals())
    model, history, best_epoch = fit_keras(
        data, RunConfig("main"), SEED, CHECKPOINTS / "best_keras.keras"
    )
    REPORTS.mkdir(exist_ok=True)
    history.to_csv(REPORTS / "history_keras.csv", index=False)
    plot_curves(
        history, best_epoch, REPORTS / "learning_curves_keras.png", "Keras main run"
    )
    results = {
        "seed": SEED,
        "best_epoch": best_epoch,
        "validation": score_keras(model, data.validation_dataset),
        "test": score_keras(model, data.test_dataset),
    }
    report = json.dumps(results, indent=2)
    (REPORTS / "metrics_keras.json").write_text(report + "\n", encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
