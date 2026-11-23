"""
Computes every number in the results table from the five CSV files, for the full test set and the human evaluation subset.

    ground_truth.csv    the full test set: lines 1000-1499 of the 10 test logs, and who replied to each line
    human_eval.csv      a human's confidence (0 to 1) that stig_ replies, for lines 1000-1049 of 2005-07-06_14
    symbolic_eval.csv   ┐ one row per test point (a line, and a candidate who spoke in the 20 lines up to it,
    ml_eval.csv         │ other than its speaker), with the approach's score. Written by evaluate_symbolic.py,
    llm_eval.csv        ┘ ml_inference.py and llm_inference.py with --eval

A point is positive when its candidate is one of the line's responders in ground_truth.csv (each file's actual
column is checked against this), and predicted positive when its score is at least CUTOFF. AUC is exact: the
chance a random positive point outscores a random negative one, with ties counting half.

The human evaluation subset is every line the human scored (stig_'s own lines are left blank) where stig_ is a
candidate. Every approach is scored on exactly those points, with the same labels.

Usage: python eval.py [--points]
    --points  also lists every subset point, with each approach's score
"""

import csv
import sys
from pathlib import Path

import numpy as np

# relative to script so it works no matter the directory.
FOLDER = Path(__file__).resolve().parent

GROUND_TRUTH_CSV = FOLDER / "ground_truth.csv"
HUMAN_CSV        = FOLDER / "human_eval.csv"
APPROACH_CSVS    = {
    "symbolic": FOLDER / "symbolic_eval.csv",
    "ml"      : FOLDER / "ml_eval.csv",
    "llm"     : FOLDER / "llm_eval.csv",
}

CUTOFF    = 0.5  # machine_learning.ROUTING_CUTOFF, the same for every approach
HUMAN_LOG = "2005-07-06_14"
TARGET    = "stig_"

Key = tuple[str, int, str]  # (log, line, candidate)


def read_csv(path: Path) -> list[dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def load_ground_truth(path: Path = GROUND_TRUTH_CSV) -> dict[tuple[str, int], dict[str, str]]:
    return {(row["log"], int(row["line"])): row for row in read_csv(path)}


def label(ground_truth: dict[tuple[str, int], dict[str, str]], key: Key) -> bool:
    return key[2] in ground_truth[key[:2]]["responders"].split()


def load_approach(path: Path, ground_truth: dict[tuple[str, int], dict[str, str]]) -> dict[Key, float]:
    """
    Scores by point. Checks every row against ground_truth.csv, so a file made from different data can't slip through
    """
    scores = {}

    for row in read_csv(path):
        key = (row["log"], int(row["line"]), row["candidate"])
        score = float(row["score"])

        if key[:2] not in ground_truth:
            raise ValueError(f"{path.name}: {key} isn't a line in {GROUND_TRUTH_CSV.name}")
        if row["message"] != ground_truth[key[:2]]["message"]:
            raise ValueError(f"{path.name}: {key} message doesn't match {GROUND_TRUTH_CSV.name}")
        if bool(int(row["actual"])) != label(ground_truth, key):
            raise ValueError(f"{path.name}: {key} actual doesn't match {GROUND_TRUTH_CSV.name}")
        if bool(int(row["predicted"])) != (score >= CUTOFF):
            raise ValueError(f"{path.name}: {key} predicted doesn't match its score at cutoff {CUTOFF}")
        if key in scores:
            raise ValueError(f"{path.name}: {key} appears twice")

        scores[key] = score

    return scores


def load_human(path: Path = HUMAN_CSV, ground_truth: dict[tuple[str, int], dict[str, str]] | None = None) -> dict[Key, float]:
    """
    The human's scores for TARGET, by point. Blank confidences (TARGET's own lines) aren't predictions, so they're skipped
    """
    scores = {}

    for row in read_csv(path):
        key = (HUMAN_LOG, int(row["index"]), TARGET)

        if ground_truth is not None and row["line"] != ground_truth[key[:2]]["message"]:
            raise ValueError(f"{path.name}: line {key[1]} doesn't match {GROUND_TRUTH_CSV.name}")

        if row["confidence"].strip():
            scores[key] = float(row["confidence"])

    return scores


def auc_score(y: np.ndarray, t: np.ndarray) -> float:
    """
    Exact ROC AUC from ranks (the Mann-Whitney U statistic). Tied scores share their average rank
    """
    positives = int(t.sum())
    negatives = len(t) - positives

    if positives == 0 or negatives == 0:
        return float("nan")

    order = np.argsort(y, kind="mergesort")
    _, first, counts = np.unique(y[order], return_index=True, return_counts=True)

    ranks = np.empty(len(y))
    ranks[order] = np.repeat(first + (counts + 1) / 2, counts)  # 1-based average rank of each tie group

    return float((ranks[t].sum() - positives * (positives + 1) / 2) / (positives * negatives))


def compute_metrics(y: np.ndarray, t: np.ndarray) -> dict[str, float | int]:
    """
    Scores y against boolean labels t, at CUTOFF
    """
    predicted = y >= CUTOFF

    tp = int(np.sum( predicted &  t))
    fp = int(np.sum( predicted & ~t))
    fn = int(np.sum(~predicted &  t))

    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall    = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1        = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0

    return {
        "points"    : len(t),
        "positives" : int(t.sum()),
        "tp"        : tp,
        "fp"        : fp,
        "fn"        : fn,
        "accuracy"  : float(np.mean(predicted == t)),
        "precision" : precision,
        "recall"    : recall,
        "f1"        : f1,
        "auc"       : auc_score(y, t),
    }


def score(approach: dict[Key, float], keys: list[Key], ground_truth) -> dict[str, float | int]:
    y = np.array([approach[key] for key in keys], dtype="float64")
    t = np.array([label(ground_truth, key) for key in keys], dtype=bool)
    return compute_metrics(y, t)


def print_table(title: str, rows: dict[str, dict[str, float | int]]) -> None:
    print(f"=== {title} ===")
    columns = list(next(iter(rows.values())))

    print(f"{'approach':<10}" + "".join(f"{column:>11}" for column in columns))
    for name, metrics in rows.items():
        print(f"{name:<10}" + "".join(
            f"{value:>11.4f}" if isinstance(value, float) else f"{value:>11}" for value in metrics.values()
        ))
    print()


def print_points(keys: list[Key], scores: dict[str, dict[Key, float]], ground_truth) -> None:
    print(f"=== Subset points ===")
    print(f"{'line':<6}{'gold':>5}" + "".join(f"{name:>10}" for name in scores) + "  message")
    for key in keys:
        print(f"{key[1]:<6}{int(label(ground_truth, key)):>5}"
              + "".join(f"{approach[key]:>10.2f}" for approach in scores.values())
              + f"  {ground_truth[key[:2]]['message']}")


if __name__ == "__main__":
    ground_truth = load_ground_truth()

    approaches: dict[str, dict[Key, float]] = {}
    for name, path in APPROACH_CSVS.items():
        if path.exists():
            approaches[name] = load_approach(path, ground_truth)
        else:
            print(f"Skipping {name}: {path.name} doesn't exist yet\n")

    if not approaches:
        sys.exit("None of the approach CSVs exist yet")

    # every approach has to be scored on the same points, or the numbers can't be compared
    test_keys = sorted(next(iter(approaches.values())))
    for name, scores in approaches.items():
        if set(scores) != set(test_keys):
            sys.exit(f"{APPROACH_CSVS[name].name} has {len(scores)} points, but "
                     f"{APPROACH_CSVS[next(iter(approaches))].name} has {len(test_keys)}, and they aren't the same points")

    human       = load_human(HUMAN_CSV, ground_truth)
    subset_keys = sorted(key for key in human if key in approaches[next(iter(approaches))])

    print(f"(cutoff {CUTOFF})\n")

    print_table(
        f"Full test set ({len(ground_truth)} lines of {len({log for log, _ in ground_truth})} logs)",
        {name: score(scores, test_keys, ground_truth) for name, scores in approaches.items()},
    )

    print_table(
        f"Human evaluation subset ({HUMAN_LOG} lines {min(k[1] for k in human)}-{max(k[1] for k in human)}, "
        f"candidate {TARGET}; {len(human) - len(subset_keys)} scored lines where {TARGET} isn't a candidate are left out)",
        {name: score(scores, subset_keys, ground_truth) for name, scores in {"human": human, **approaches}.items()},
    )

    if "--points" in sys.argv[1:]:
        print_points(subset_keys, {"human": human, **approaches}, ground_truth)
