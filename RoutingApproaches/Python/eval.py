import csv
from pathlib import Path

import numpy as np

import machine_learning
from evaluate_machine_learning import compute_metrics
from evaluate_symbolic import score_point
from machine_learning import CANDIDATE_POOL, FEATURE_HISTORY, ROUTING_CUTOFF, MachineLearning
from symbolic import Symbolic
from util import AnnotatedLog, Message, Point, iter_points, lines_to_messages, load_split

"""
Evaluates the human baseline, the symbolic approach, and optionally a trained ML model.

Produces every number in the results table:
1. Symbolic (and ML) on the whole test split
2. Symbolic, ML and human on the human evaluation subset (same points, same labels)
3. A per-point breakdown of the subset, for looking at specific examples

Points come from util.iter_points, so the subset is exactly the test-split points
for TARGET whose index the human labeled. The ML model scores the same points, but its
features see up to FEATURE_HISTORY messages back, like in training.

Usage: python eval.py [human_path] [--ml WEIGHTS TOKENS] [--no-text]
    --no-text  for models trained with USE_TEXT = False
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


def score_split(split: str, evaluator_cache: dict[str, Symbolic]) -> tuple[np.ndarray, np.ndarray, list[Point]]:
    y      = []
    t      = []
    points = []

    for log in load_split(split):
        for point in iter_points(log, CANDIDATE_POOL):
            y.append(score_point(point, evaluator_cache))
            t.append(point.label)
            points.append(point)

    return np.array(y, dtype="float32"), np.array(t, dtype="float32"), points


def score_ml(model: MachineLearning, points: list[Point], logs: dict[str, AnnotatedLog]) -> np.ndarray:
    """
    Scores points with the ML model. iter_points windows are only CANDIDATE_POOL long,
    so the model's window is rebuilt from the log to match training
    """
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


def print_points(points: list[Point], y_human: np.ndarray, y_symbolic: np.ndarray, y_ml: np.ndarray | None = None) -> None:
    print(f"{'index':<7}{'gold':>5}{'human':>7}{'symbolic':>10}" + (f"{'ml':>7}" if y_ml is not None else "") + "  message")
    for i, (point, human, symbolic) in enumerate(zip(points, y_human, y_symbolic)):
        ml = f"{y_ml[i]:>7.2f}" if y_ml is not None else ""
        print(f"{point.index:<7}{int(point.label):>5}{human:>7.2f}{symbolic:>10.2f}{ml}  {point.window[-1]}")


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("human_path", nargs="?", type=Path, default=HUMAN_PATH)
    parser.add_argument("--ml", nargs=2, metavar=("WEIGHTS", "TOKENS"), help="also evaluate this trained ML model")
    parser.add_argument("--no-text", action="store_true", help="the ML model was trained with USE_TEXT = False")
    args = parser.parse_args()

    human_path = args.human_path
    evaluator_cache: dict[str, Symbolic] = {}

    print(f"=== Scoring {TEST_SPLIT} data ===")
    y_test, t_test, test_points = score_split(TEST_SPLIT, evaluator_cache)

    print(f"=== Scoring human subset ({HUMAN_LOG}, target {TARGET}) ===")
    human  = load_human(human_path)
    points = subset_points(human)

    t_subset   = np.array([point.label for point in points], dtype="float32")
    y_human    = np.array([human[point.index] for point in points], dtype="float32")
    y_symbolic = np.array([score_point(point, evaluator_cache) for point in points], dtype="float32")

    rows = {"symbolic (test)": compute_metrics(y_test, t_test)}
    y_ml_subset = None

    if args.ml:
        print(f"=== Scoring {TEST_SPLIT} data with ML model ({args.ml[0]}) ===")
        machine_learning.USE_TEXT = not args.no_text

        model = MachineLearning(False, *args.ml)
        logs  = {log.name: log for log in load_split(TEST_SPLIT)}

        rows["ml (test)"] = compute_metrics(score_ml(model, test_points, logs), t_test)
        y_ml_subset       = score_ml(model, points, logs)

    rows["symbolic (subset)"] = compute_metrics(y_symbolic, t_subset)
    if y_ml_subset is not None:
        rows["ml (subset)"] = compute_metrics(y_ml_subset, t_subset)
    rows["human (subset)"] = compute_metrics(y_human, t_subset)

    print(f"=== Results at cutoff {ROUTING_CUTOFF} ===")
    print_table(rows)

    print("\n=== Subset points ===")
    print_points(points, y_human, y_symbolic, y_ml_subset)
