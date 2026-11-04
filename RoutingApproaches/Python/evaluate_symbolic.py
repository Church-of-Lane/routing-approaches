import numpy as np

from evaluate_machine_learning import compute_metrics
from machine_learning import CANDIDATE_POOL, HISTORY, ROUTING_CUTOFF, VALIDATION_SPLIT
from symbolic import Symbolic
from util import Point, iter_points, load_split

from time import time

"""
Evaluates the symbolic approach against the validation (dev) split.

Points are built exactly like in machine_learning.py / evaluate_machine_learning.py
(same candidate pool, same candidates, same labels), so the metrics are directly comparable.

Usage: python evaluate_symbolic.py
"""

def score_data(evaluator_cache: dict[str, Symbolic]) -> tuple[np.ndarray, np.ndarray]:
    y_valid = []
    t_valid = []

    for log in load_split(VALIDATION_SPLIT):
        for point in iter_points(log, CANDIDATE_POOL):
            y_valid.append(score_point(point, evaluator_cache))
            t_valid.append(point.label)

    return np.array(y_valid, dtype="float32"), np.array(t_valid, dtype="float32")


def score_point(point: Point, evaluator_cache: dict[str, Symbolic]) -> float:
    # SpellChecker is slow to construct, so reuse evaluators across logs
    if point.candidate not in evaluator_cache:
        evaluator_cache[point.candidate] = Symbolic(point.candidate)

    # symbolic only ever looked at the last HISTORY messages, so keep it that way
    return evaluator_cache[point.candidate].score_text(point.window[-HISTORY:])


if __name__ == "__main__":
    start_time = time()

    print(f"=== Scoring {VALIDATION_SPLIT} data ===")
    y, t = score_data({})

    print(f"=== Evaluating at cutoff {ROUTING_CUTOFF} ===")
    for name, value in compute_metrics(y, t).items():
        print(f"{name:<10} {value:.4f}" if isinstance(value, float) else f"{name:<10} {value}")

    print("In total, took " + str(time() - start_time))
