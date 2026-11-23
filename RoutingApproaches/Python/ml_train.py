"""
Trains the machine learning routing model (Keras), and writes its weights, checkpoints and training curve.

Task: given an IRC chat, score from 0 to 1 whether a candidate (someone who spoke in the last
CANDIDATE_POOL messages, other than the speaker) replies to the latest message. Each training point is
one (message, candidate) pair from the irc-disentanglement train split, labeled by whether that
candidate replied. About 6% of points are positive.

Model (the 3 hidden layer, features-only model from the layer sweep in models/models.md, the one submitted):
    37 hand-made features per (message, candidate), from up to the last 100 messages:
    mentions of the candidate, how recently they and the speaker spoke and addressed each other,
    how much they've been talking, topic overlap, and what the message looks like
    (MachineLearning.compute_features in machine_learning.py lists all of them)
        -> Dense 64 relu -> Dense 32 relu -> Dense 16 relu -> Dense 1 sigmoid
    5,057 parameters. The text branch (embedding + LSTM over the last 20 messages) is turned off:
    with it, the model had ~478k parameters and did no better on dev (models.md, v4 mute vs slowpoke).
    2, 3 and 4 hidden layers tied on dev, and 3 had the best F1 (models.md, layer sweep without the text branch).
    The layers are built by MachineLearning.get_model_architecture in machine_learning.py.

Outputs (in weights/):
    ml.weights.h5               the final weights (the best epoch by val loss), loaded by ml_inference.py
    checkpoints/                a checkpoint every time val loss improves, so a crash doesn't lose the run
    ml_history.json             loss and metrics per epoch
    ml_training_curve.png       the training curve: train vs validation loss and AUC/F1 per epoch

Usage: python ml_train.py
"""

import json
import random
from pathlib import Path
from time import time

import keras
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tqdm.keras import TqdmCallback

from machine_learning import FEATURE_COUNT, ROUTING_CUTOFF, SEED, VALIDATION_SPLIT, MachineLearning
from util import FOLDER, load_messaged_split

# === Architecture ===
USE_TEXT       = False     # features only. The text branch added ~470k parameters and nothing on dev
FEATURE_LAYERS = (64,)     # dense layers on the features (before the join with the text branch, when there is one)
HEAD_LAYERS    = (32, 16)  # dense layers after the join

# === Training ===
EPOCHS          = 100      # upper limit. Early stopping ends training well before this
BATCH_SIZE      = 512
LEARNING_RATE   = 1e-4     # Adam. 1e-3 learned almost everything in the first epoch, so the curve couldn't be seen
POSITIVE_WEIGHT = 3.0      # class weight on positives. Moves the F1-optimal threshold of the unweighted model (~0.25) to ROUTING_CUTOFF
PATIENCE        = 5        # epochs without a better val loss before stopping (the best epoch's weights are restored)
LOSS            = "binary_crossentropy"

# === Outputs ===
WEIGHTS_DIR    = FOLDER / "weights"
WEIGHTS_PATH   = WEIGHTS_DIR / "ml.weights.h5"
CHECKPOINT_DIR = WEIGHTS_DIR / "checkpoints"
HISTORY_PATH   = WEIGHTS_DIR / "ml_history.json"
CURVE_PATH     = WEIGHTS_DIR / "ml_training_curve.png"


def fit(model: MachineLearning, x_train: dict[str, np.ndarray], t_train: np.ndarray, x_valid: dict[str, np.ndarray],
        t_valid: np.ndarray, feature_count: int = FEATURE_COUNT, feature_layers: tuple[int, ...] = FEATURE_LAYERS,
        head_layers: tuple[int, ...] = HEAD_LAYERS, checkpoint_dir: Path | None = None) -> tuple[keras.Model, keras.callbacks.History]:
    """
    Builds a fresh model and fits it on already formatted data. Early stopping watches val loss, and restores the
    best epoch's weights at the end. With a checkpoint_dir, the weights are also written there whenever val loss improves
    """
    print(f"=== Training up to {EPOCHS} epochs ===")

    network = model.get_model_architecture(feature_count, feature_layers, head_layers)
    network.summary()

    network.compile(
        optimizer = keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss      = LOSS,
        metrics   = [keras.metrics.Precision(), keras.metrics.F1Score(threshold=ROUTING_CUTOFF, average="micro"), keras.metrics.AUC()],
    )

    callbacks = [
        TqdmCallback(verbose=1),
        keras.callbacks.EarlyStopping(monitor="val_loss", mode="min", patience=PATIENCE, restore_best_weights=True),
    ]

    if checkpoint_dir is not None:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        callbacks.append(keras.callbacks.ModelCheckpoint(
            filepath          = str(checkpoint_dir / "epoch_{epoch:03d}.weights.h5"),
            monitor           = "val_loss",
            mode              = "min",
            save_best_only    = True,
            save_weights_only = True,
        ))

    history = network.fit(
        x_train,
        t_train,

        validation_data = (x_valid, t_valid),
        epochs          = EPOCHS,
        batch_size      = BATCH_SIZE,
        verbose         = 0,
        callbacks       = callbacks,
        class_weight    = {0: 1.0, 1: POSITIVE_WEIGHT},
    )

    return network, history


def plot_training_curve(history: dict[str, list[float]], path: Path) -> None:
    """
    Loss on the left, AUC and F1 on the right. Train is dotted and validation solid, and the restored (best) epoch is marked
    """
    epochs = np.arange(1, len(history["loss"]) + 1)
    best   = int(np.argmin(history["val_loss"])) + 1

    fig, (loss_ax, metric_ax) = plt.subplots(1, 2, figsize=(12, 4.5))

    loss_ax.plot(epochs, history["loss"],     ":", color="#008300", linewidth=2, label="train loss (class weighted)")
    loss_ax.plot(epochs, history["val_loss"], "-", color="#008300", linewidth=2, label="validation loss")
    loss_ax.set_title("binary cross-entropy")

    for metric, color in (("auc", "#2a78d6"), ("f1_score", "#eda100")):
        name = metric.removesuffix("_score")
        metric_ax.plot(epochs, history[metric],          ":", color=color, linewidth=2, label=f"train {name}")
        metric_ax.plot(epochs, history[f"val_{metric}"], "-", color=color, linewidth=2, label=f"validation {name}")
    metric_ax.set_title(f"AUC and F1 (cutoff {ROUTING_CUTOFF})")

    for ax in (loss_ax, metric_ax):
        ax.axvline(best, color="#7a7974", linestyle="--", linewidth=1, label=f"best epoch ({best})")
        ax.set_xlabel("epoch")
        ax.grid(axis="y", color="#e5e4df", linewidth=0.8)
        ax.set_axisbelow(True)
        ax.legend(frameon=False, fontsize=9)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)

    fig.suptitle("ML routing model training curve")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def clean_history(history: keras.callbacks.History) -> dict[str, list[float]]:
    return {key: np.asarray(values).tolist() for key, values in history.history.items()}


if __name__ == "__main__":
    random.seed(SEED)
    keras.utils.set_random_seed(SEED)

    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

    model = MachineLearning(True, use_text=USE_TEXT)

    print("=== Formatting data ===")
    start = time()

    x_train, t_train = model.format_data(model.training_data)
    x_valid, t_valid = model.format_data(load_messaged_split(VALIDATION_SPLIT))

    print(f"=== Formatted data (took {time() - start:.1f}s) ===")
    print(f"Train points {len(t_train)}, validation points {len(t_valid)}, positive rate {np.mean(t_train):.4f}")

    model.model, history = fit(model, x_train, t_train, x_valid, t_valid, checkpoint_dir=CHECKPOINT_DIR)
    history = clean_history(history)

    model.save_weights(str(WEIGHTS_PATH))
    HISTORY_PATH.write_text(json.dumps(history))
    plot_training_curve(history, CURVE_PATH)

    if USE_TEXT:
        model.save_token_map(str(WEIGHTS_DIR / "token_map.json"))

    best = int(np.argmin(history["val_loss"]))
    print(f"=== Stopped after {len(history['val_loss'])} epochs; restored epoch {best + 1} "
          f"(val loss {history['val_loss'][best]:.4f}, val AUC {history['val_auc'][best]:.4f}) ===")
    print(f"=== Saved weights to {WEIGHTS_PATH}, history to {HISTORY_PATH}, training curve to {CURVE_PATH} ===")
