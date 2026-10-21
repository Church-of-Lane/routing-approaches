import keras
import numpy as np

from machine_learning import MachineLearning, ROUTING_CUTOFF, VALIDATION_SPLIT
from util import load_messaged_split

"""
Loads the trained model from disk and evaluates it against the validation (dev) split.

Usage: python evaluate_machine_learning.py [weight_path] [token_path]
"""

def evaluate(model: MachineLearning, x: dict[str, np.ndarray], t: np.ndarray, cutoff: float = ROUTING_CUTOFF) -> dict[str, float]:
    return compute_metrics(model.model.predict(x, verbose=0).reshape(-1), t.reshape(-1), cutoff)

def compute_metrics(y: np.ndarray, t: np.ndarray, cutoff: float = ROUTING_CUTOFF) -> dict[str, float]:
    """
    Scores predictions y against labels t
    """
    predicted = y >= cutoff
    actual    = t == 1.0

    true_pos  = int(np.sum( predicted &  actual))
    false_pos = int(np.sum( predicted & ~actual))
    false_neg = int(np.sum(~predicted &  actual))

    precision = true_pos / (true_pos + false_pos) if true_pos + false_pos > 0 else 0.0
    recall    = true_pos / (true_pos + false_neg) if true_pos + false_neg > 0 else 0.0
    f1        = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0

    auc = keras.metrics.AUC()
    auc.update_state(t, y)

    return {
        "points"    : len(t),
        "positives" : int(np.sum(actual)),
        "accuracy"  : float(np.mean(predicted == actual)),
        "precision" : precision,
        "recall"    : recall,
        "f1"        : f1,
        "auc"       : float(auc.result()),
    }


if __name__ == "__main__":
    from sys import argv

    weight_path = argv[1] if len(argv) > 1 else "routing.weights.h5"
    token_path  = argv[2] if len(argv) > 2 else "token_map.json"

    model = MachineLearning(False, weight_path, token_path)

    print(f"=== Formatting {VALIDATION_SPLIT} data ===")
    x, t = model.format_data(load_messaged_split(VALIDATION_SPLIT))

    print(f"=== Evaluating at cutoff {ROUTING_CUTOFF} ===")
    for name, value in evaluate(model, x, t).items():
        print(f"{name:<10} {value:.4f}" if isinstance(value, float) else f"{name:<10} {value}")
