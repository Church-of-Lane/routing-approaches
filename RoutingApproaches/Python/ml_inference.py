"""
Loads the trained ML routing model's weights (weights/ml.weights.h5, written by ml_train.py) and scores
whether a candidate replies to the latest message of a chat. No transfer learning: every weight is ours.

    python ml_inference.py [sample] [candidate]   score the test set sample (test_sample.txt, for crimsun)
    python ml_inference.py --eval                 score every point in the full test set, and write ml_eval.csv

A chat is one IRC line per line ("[HH:MM] <user> message"), oldest first, and the last line is the message being
routed. The features look back up to FEATURE_HISTORY (100) lines, so more context helps, but any length works.
"""

from pathlib import Path

import numpy as np

from machine_learning import CANDIDATE_POOL, FEATURE_HISTORY, ROUTING_CUTOFF, MachineLearning
from ml_train import FEATURE_LAYERS, HEAD_LAYERS, USE_TEXT, WEIGHTS_PATH
from util import (FOLDER, TEST_SAMPLE, TEST_SAMPLE_CANDIDATE, TEST_SPLIT, Message, Point, lines_to_messages,
                  load_split, load_test_sample, test_points, write_eval_csv)

ML_EVAL_CSV = FOLDER / "ml_eval.csv"


def load_model(weight_path: Path = WEIGHTS_PATH) -> MachineLearning:
    """
    Builds the same architecture ml_train.py trained, and loads its weights
    """
    return MachineLearning(False, str(weight_path), None, use_text=USE_TEXT, feature_layers=FEATURE_LAYERS, head_layers=HEAD_LAYERS)


def predict(model: MachineLearning, lines: list[str], candidate: str) -> float:
    """
    The model's confidence (0 to 1) that candidate replies to the last line
    """
    return model.inference(lines_to_messages(lines), candidate)


def score_points(model: MachineLearning, points: list[Point]) -> np.ndarray:
    """
    Scores test points in one batch. A point's window is only CANDIDATE_POOL lines long, so the
    FEATURE_HISTORY window is rebuilt from the whole log, the same as in training
    """
    logs = {log.name: log for log in load_split(TEST_SPLIT)}
    messages: dict[str, list[Message]] = {}
    batch = []

    for point in points:
        if point.log not in messages:
            log = logs[point.log]
            messages[point.log] = lines_to_messages(log.lines, log.replies_to)

        window = messages[point.log][max(0, point.index + 1 - FEATURE_HISTORY):point.index + 1]
        batch.append(model.prepare_point(window, point.candidate))

    x = {key: np.stack([point[key] for point in batch]) for key in batch[0]}
    return model.model.predict(x, verbose=0, batch_size=2048).reshape(-1)


if __name__ == "__main__":
    from sys import argv

    model = load_model()

    if argv[1:] == ["--eval"]:
        points = test_points(CANDIDATE_POOL)
        print(f"=== Scoring {len(points)} test points ===")
        write_eval_csv(ML_EVAL_CSV, points, score_points(model, points), ROUTING_CUTOFF)
    else:
        sample    = Path(argv[1]) if len(argv) > 1 else TEST_SAMPLE
        candidate = argv[2]       if len(argv) > 2 else TEST_SAMPLE_CANDIDATE
        lines     = load_test_sample(sample)

        score = predict(model, lines, candidate)

        print(f"Latest message: {lines[-1]}")
        print(f"Should {candidate} reply? {score:.4f} -> {'yes' if score >= ROUTING_CUTOFF else 'no'} (cutoff {ROUTING_CUTOFF})")
