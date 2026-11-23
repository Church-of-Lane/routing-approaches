import json
import re
from pathlib import Path
from time import time

import keras
import numpy as np
from keras.src.legacy.preprocessing.text import Tokenizer

import machine_learning as ml
from evaluate_machine_learning import compute_metrics
from machine_learning import FEATURE_COUNT, OTHER_TAG, SELF_TAG, SEED, TARGET_TAG, USE_TEXT, VALIDATION_SPLIT, MachineLearning
from util import load_messaged_split, load_training_data

"""
Trains the model once for each feature count in a range, using the first N features
(in compute_features order), and scores each one on the validation split.

Data is formatted once with all FEATURE_COUNT features and cached, then sliced per run.
Every run is reseeded so the only difference between runs is the feature count.

Usage: python feature_sweep.py [low] [high] [out_dir]
"""

def formatted_data(model: MachineLearning, cache: Path):
    if cache.exists():
        print(f"=== Loading cached data from {cache} ===")
        data = np.load(cache)
        unpack = lambda prefix: {k[len(prefix):]: data[k] for k in data.files if k.startswith(prefix)}
        return unpack("xtrain_"), data["ttrain"], unpack("xvalid_"), data["tvalid"]

    print("=== Formatting data ===")
    start = time()

    x_train, t_train = model.format_data(model.training_data)
    x_valid, t_valid = model.format_data(load_messaged_split(VALIDATION_SPLIT))

    print(f"=== Formatted data (took {time() - start:.1f}s) ===")

    np.savez(cache, ttrain=t_train, tvalid=t_valid,
             **{f"xtrain_{k}": v for k, v in x_train.items()},
             **{f"xvalid_{k}": v for k, v in x_valid.items()})

    return x_train, t_train, x_valid, t_valid


def take(x: dict[str, np.ndarray], n: int) -> dict[str, np.ndarray]:
    return {**x, "features": x["features"][:, :n]}


def setup_model() -> MachineLearning:
    """
    Same setup as MachineLearning.__init__(train=True), without training
    """
    model = MachineLearning.__new__(MachineLearning)
    model.tokenizer     = Tokenizer(oov_token="<OOV>")
    model.training_data = load_training_data()
    model.tokenizer.fit_on_texts([m.text for log in model.training_data for m in log.messages])
    model.tokenizer.fit_on_texts([SELF_TAG, OTHER_TAG, TARGET_TAG])
    model.vocab_size = len(model.tokenizer.word_index)
    return model


def run(model: MachineLearning, data, n: int = FEATURE_COUNT, **layers) -> tuple[dict, keras.Model, keras.callbacks.History]:
    """
    Trains a reseeded model on the first n features and scores it on the validation split
    """
    x_train, t_train, x_valid, t_valid = data
    start = time()

    ml.random.seed(SEED)
    keras.utils.set_random_seed(SEED)

    trained, history = model.fit_model(take(x_train, n), t_train, take(x_valid, n), t_valid, feature_count=n, **layers)

    y        = trained.predict(take(x_valid, n), verbose=0, batch_size=2048).reshape(-1)
    val_loss = history.history["val_loss"]
    best     = int(np.argmin(val_loss))

    return {
        **compute_metrics(y, t_valid.reshape(-1)),
        "val_loss"   : float(val_loss[best]),
        "best_epoch" : best + 1,
        "epochs"     : len(val_loss),
        "params"     : trained.count_params(),
        "seconds"    : time() - start,
    }, trained, history


def history_json(history: keras.callbacks.History) -> str:
    """
    Keras numbers repeated metric names within a process (auc_2, val_precision_1, ...), so the suffix is
    stripped to keep the keys the same as a single training run's
    """
    return json.dumps({re.sub(r"_\d+$", "", k): np.asarray(v).tolist() for k, v in history.history.items()})


def summary(result: dict) -> str:
    return ", ".join(f"{k} {v:.4f}" if isinstance(v, float) else f"{k} {v}" for k, v in result.items())


if __name__ == "__main__":
    from sys import argv

    low     = int(argv[1]) if len(argv) > 1 else 10
    high    = int(argv[2]) if len(argv) > 2 else FEATURE_COUNT
    out_dir = Path(argv[3] if len(argv) > 3 else "models/feature_sweep")
    out_dir.mkdir(parents=True, exist_ok=True)

    results_path = out_dir / "results.json"
    results      = json.loads(results_path.read_text()) if results_path.exists() else {}

    model = setup_model()
    model.save_token_map(str(out_dir / "token_map.json"))  # shared by every run
    data  = formatted_data(model, out_dir / ("data_text.npz" if USE_TEXT else "data.npz"))

    print(f"Train points {len(data[1])}, valid points {len(data[3])}, label imbalance {np.mean(data[1]):.4f}")

    for n in range(low, high + 1):
        if str(n) in results:
            print(f"=== Skipping {n} features (already done) ===")
            continue

        print(f"=== {n} features ===")
        results[str(n)], trained, history = run(model, data, n)

        trained.save_weights(out_dir / f"{n}.weights.h5")
        (out_dir / f"{n}.history.json").write_text(history_json(history))
        results_path.write_text(json.dumps(results, indent=2))

        print(f"=== {n} features: {summary(results[str(n)])} ===")
