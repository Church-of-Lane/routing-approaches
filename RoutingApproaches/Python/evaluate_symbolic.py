import numpy as np

from evaluate_machine_learning import compute_metrics
from machine_learning import HISTORY, ROUTING_CUTOFF, VALIDATION_SPLIT
from symbolic import Symbolic
from util import load_split, lines_to_messages

"""
Evaluates the symbolic approach against the validation (dev) split.

Points are built exactly like in machine_learning.py / evaluate_machine_learning.py
(same window size, same candidates, same labels), so the metrics are directly comparable.

Usage: python evaluate_symbolic.py
"""

def score_data(evaluator_cache: dict[str, Symbolic]) -> tuple[np.ndarray, np.ndarray]:
    y_valid = []
    t_valid = []

    for log in load_split(VALIDATION_SPLIT):
        messages = lines_to_messages(log.lines, log.replies_to)

        # symbolic works on raw lines, so keep the raw window alongside the parsed one
        window     = []
        raw_window = []

        for line, message in zip(log.lines, messages):
            window    .append(message)
            raw_window.append(line)

            if len(window) > HISTORY:
                window    .pop(0)
                raw_window.pop(0)

            authors = set(m.username for m in window if m.username is not None)
            authors.discard(message.username)

            if len(window) == HISTORY and message.annotated and len(authors) > 0 and message.username != None:
                for person in authors:
                    # SpellChecker is slow to construct, so reuse evaluators across logs
                    if person not in evaluator_cache:
                        evaluator_cache[person] = Symbolic(person)

                    y_valid.append(evaluator_cache[person].score_text(raw_window))
                    t_valid.append(person in message.responders)

    return np.array(y_valid, dtype="float32"), np.array(t_valid, dtype="float32")


if __name__ == "__main__":
    print(f"=== Scoring {VALIDATION_SPLIT} data ===")
    y, t = score_data({})

    print(f"=== Evaluating at cutoff {ROUTING_CUTOFF} ===")
    for name, value in compute_metrics(y, t).items():
        print(f"{name:<10} {value:.4f}" if isinstance(value, float) else f"{name:<10} {value}")
