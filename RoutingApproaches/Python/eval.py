import csv
from pathlib import Path

import numpy as np

from evaluate_machine_learning import compute_metrics
from evaluate_symbolic import score_point
from machine_learning import CANDIDATE_POOL, ROUTING_CUTOFF
from symbolic import Symbolic
from util import Point, iter_points, load_split

"""
Evaluates the human baseline and the symbolic approach.

Produces every number in the results table:
1. Symbolic on the whole test split
2. Symbolic and human on the human evaluation subset (same points, same labels)
3. A per-point breakdown of the subset, for looking at specific examples

Points come from util.iter_points, so the subset is exactly the test-split points
for TARGET whose index the human labeled.

Usage: python eval.py [human_path]
"""

TEST_SPLIT = "test"

# relative to script so it works no matter the directory.
HUMAN_PATH = Path(__file__).resolve().parent / "../../data/HumanBaseline/2005-07-06_14_lines1000-1049_stig_.csv"
HUMAN_LOG  = "2005-07-06_14"
TARGET     = "stig_"


def load_human(path: Path) -> dict[int, float]:
    """
    Reads index -> confidence. Empty confidences (target is the speaker) aren't predictions, so are skipped
    """
    with path.open(encoding="utf-8", newline="") as file:
        return {int(row["index"]): float(row["confidence"]) for row in csv.DictReader(file) if row["confidence"]}


def score_split(split: str, evaluator_cache: dict[str, Symbolic]) -> tuple[np.ndarray, np.ndarray]:
    y = []
    t = []

    for log in load_split(split):
        for point in iter_points(log, CANDIDATE_POOL):
            y.append(score_point(point, evaluator_cache))
            t.append(point.label)

    return np.array(y, dtype="float32"), np.array(t, dtype="float32")


def subset_points(human: dict[int, float]) -> list[Point]:
    log = next(log for log in load_split(TEST_SPLIT) if log.name == HUMAN_LOG)

    points = [point for point in iter_points(log, CANDIDATE_POOL) if point.candidate == TARGET and point.index in human]

    if not points:
        raise ValueError(f"No {TARGET} points in {HUMAN_LOG} for the labeled indices")

    return points


def print_table(rows: dict[str, dict[str, float]]) -> None:
    columns = list(next(iter(rows.values())))

    print(f"{'approach':<20}" + "".join(f"{column:>11}" for column in columns))
    for name, metrics in rows.items():
        print(f"{name:<20}" + "".join(
            f"{value:>11.4f}" if isinstance(value, float) else f"{value:>11}" for value in metrics.values()
        ))


def print_points(points: list[Point], y_human: np.ndarray, y_symbolic: np.ndarray) -> None:
    print(f"{'index':<7}{'gold':>5}{'human':>7}{'symbolic':>10}  message")
    for point, human, symbolic in zip(points, y_human, y_symbolic):
        print(f"{point.index:<7}{int(point.label):>5}{human:>7.2f}{symbolic:>10.2f}  {point.window[-1]}")


if __name__ == "__main__":
    from sys import argv

    human_path = Path(argv[1]) if len(argv) > 1 else HUMAN_PATH
    evaluator_cache: dict[str, Symbolic] = {}

    print(f"=== Scoring {TEST_SPLIT} data ===")
    y_test, t_test = score_split(TEST_SPLIT, evaluator_cache)

    print(f"=== Scoring human subset ({HUMAN_LOG}, target {TARGET}) ===")
    human  = load_human(human_path)
    points = subset_points(human)

    t_subset   = np.array([point.label for point in points], dtype="float32")
    y_human    = np.array([human[point.index] for point in points], dtype="float32")
    y_symbolic = np.array([score_point(point, evaluator_cache) for point in points], dtype="float32")

    print(f"=== Results at cutoff {ROUTING_CUTOFF} ===")
    print_table({
        "symbolic (test)"   : compute_metrics(y_test, t_test),
        "symbolic (subset)" : compute_metrics(y_symbolic, t_subset),
        "human (subset)"    : compute_metrics(y_human, t_subset),
    })

    print("\n=== Subset points ===")
    print_points(points, y_human, y_symbolic)
