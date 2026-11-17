"""
Test harness for prompting an LLM on the routing task.

Task: given the last messages of an IRC log, score from 0 to 1 whether a candidate
(someone who spoke recently) should reply to the latest message.

Points come from util.iter_points, so the candidates and labels are the same ones the
symbolic and machine learning approaches are scored on.

    python llm_harness.py            develop: score every prompt in PROMPTS on a fixed training subset
    python llm_harness.py val v0 v1  compare your best few prompts on a fixed validation (dev) subset
    python llm_harness.py test v1    run the prompt "v1" once on a fixed test subset, and write llm/llm.csv
"""

import csv
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import ollama

from evaluate_machine_learning import compute_metrics
from machine_learning import CANDIDATE_POOL, HISTORY, ROUTING_CUTOFF, VALIDATION_SPLIT
from util import iter_points, load_split

MODEL = "llama3.2:3b"
OPTIONS = {
    "temperature": 0,  # always choose the most likely next token, so runs are repeatable
    "seed": 0,
    "num_ctx": 8192,   # the context window, in tokens
}
LABELS = ["0.0", "0.1", "0.2", "0.3", "0.4", "0.5", "0.6", "0.7", "0.8", "0.9", "1.0"]

# The JSON structure the model must answer in: exactly one of the labels.
SCHEMA = {
    "type": "object",
    "properties": {"answer": {"type": "string", "enum": LABELS}},
    "required": ["answer"],
}

# relative to script so it works no matter the directory.
RUNS_DIR = Path(__file__).resolve().parent / "llm/runs"
LLM_CSV  = Path(__file__).resolve().parent / "llm/llm.csv"

TRAIN_SPLIT = "train"
TEST_SPLIT  = "test"

# Every split has far too many points to run the LLM on all of them (train ~440k, dev ~16k,
# test ~30k, ~7% positive), so each one is a fixed random subset. Develop prompts on the
# training subset. The LLM was never trained on these logs, so scoring them is fine,
# as long as the few-shot examples (which are in a prompt, with their labels) are left out.
N_TRAIN = 50  # half positive, half negative (see training_subset)
N_VAL   = 200
N_TEST  = 500
SAMPLE_SEED = 0
FEWSHOT_IDS: list[str] = []  # ids ("log:index:candidate") of every training point used in any prompt

# One entry per prompt version. {name} is replaced by the candidate, {transcript} by the window.
# Write the change and the hypothesis BEFORE running a new version.
PROMPTS = {
    # baseline
    "v0": f"""Decide whether {{name}} should reply to what was just said.

Conversation so far:
<transcript>
{{transcript}}
</transcript>

Score the latest message from 0 to 1.

Score 0.0 when:
- it is plainly aimed at someone else
- it is empty, a duplicate, or cut off mid-thought
- it is a bare reaction with nothing in it to answer

Score 0.1–0.4 when:
- it is small talk aimed at the room rather than at them
- the subject is nothing to do with them, but ignoring it would be rude

Score 0.5–0.7 when:
- it continues a conversation they are already part of
- it is a straightforward question they can answer

Score 0.8–1.0 when:
- they are addressed by name, or asked something directly
- a reply would matter""",

"v1": f"""Decide whether {{name}} should reply to what was just said.

Conversation so far:
<transcript>
{{transcript}}
</transcript>

Score the latest message from 0 to 1.

Score 0.0 when:
- it is plainly aimed at someone else
- it is empty, a duplicate, or cut off mid-thought
- it is a bare reaction with nothing in it to answer

Score 0.1–0.4 when:
- it is small talk aimed at the room rather than at them
- the subject is nothing to do with them, but ignoring it would be rude

Score 0.5–0.7 when:
- it continues a conversation they are already part of
- it is a straightforward question they can answer

Score 0.8–1.0 when:
- they are addressed by name, or asked something directly""",

"v2": f"""Decide whether {{name}} should reply to what was just said.

Conversation so far:
<transcript>
{{transcript}}
</transcript>

Score the latest message from 0 to 1."""
}


def load_rows(split, n, exclude=(), balanced=False):
    """A fixed random subset of n points from a split, as dicts of id, name, transcript and label.
    The transcript is the last HISTORY lines of the window, like the symbolic approach sees.
    If balanced, half the points are positive and half negative, in shuffled order."""
    rows = [
        {
            "id": f"{point.log}:{point.index}:{point.candidate}",
            "name": point.candidate,
            "transcript": "\n".join(point.window[-HISTORY:]),
            "label": point.label,
        }
        for log in load_split(split)
        for point in iter_points(log, CANDIDATE_POOL)
    ]
    rows = [row for row in rows if row["id"] not in exclude]
    rng = random.Random(SAMPLE_SEED)
    if not balanced:
        return rng.sample(rows, min(n, len(rows)))

    positive = [row for row in rows if row["label"]]
    negative = [row for row in rows if not row["label"]]
    if len(positive) < n // 2 or len(negative) < n - n // 2:
        raise ValueError(f"{split} has {len(positive)} positive and {len(negative)} negative points, too few for {n} balanced")
    sample = rng.sample(positive, n // 2) + rng.sample(negative, n - n // 2)
    rng.shuffle(sample)
    return sample


def fill(prompt, row):
    return prompt.replace("{name}", row["name"]).replace("{transcript}", row["transcript"])


def score(prediction):
    """The prediction as a number. Invalid outputs score 0, so they count as "don't reply"."""
    return float(prediction) if prediction is not None else 0.0


def routed(prediction):
    """Whether a prediction counts as "should reply". Invalid outputs count as no."""
    return prediction is not None and float(prediction) >= ROUTING_CUTOFF


def predict(prompt, row):
    """Ask the model about one point. Returns (prediction, response, seconds).
    The prediction is None if the output is not one of the labels."""
    start = time.perf_counter()
    response = ollama.chat(
        model=MODEL,
        messages=[{"role": "user", "content": fill(prompt, row)}],
        format=SCHEMA,
        options=OPTIONS,
    )
    seconds = time.perf_counter() - start
    try:
        answer = json.loads(response.message.content)["answer"]
    except (json.JSONDecodeError, KeyError, TypeError):
        answer = None
    return (answer if answer in LABELS else None), response, seconds


def run(version, rows, out_file):
    """Run one prompt on every row, and save every response to out_file.
    The first line of out_file records the setup. If out_file already has the
    outputs for exactly this setup and these rows, they are loaded instead."""
    prompt = PROMPTS[version]
    setup = {"model": MODEL, "options": OPTIONS, "schema": SCHEMA, "prompt": prompt}
    if Path(out_file).exists():
        with open(out_file, encoding="utf-8") as f:
            lines = [json.loads(line) for line in f if line.strip()]
        if lines and lines[0] == setup and [r["id"] for r in lines[1:]] == [row["id"] for row in rows]:
            print(f"\n{version}: nothing changed, so using the saved outputs in {out_file}")
            return lines[1:]

    print(f"\n===== {version}: the exact prompt for {rows[0]['id']} =====")
    print(fill(prompt, rows[0]))
    print("=====")
    results = []
    with open(out_file, "w", encoding="utf-8") as f:
        f.write(json.dumps(setup) + "\n")
        for row in rows:
            prediction, response, seconds = predict(prompt, row)
            result = {
                "id": row["id"],
                "prediction": prediction,
                "raw": response.message.content,
                "seconds": round(seconds, 2),
                "prompt_tokens": response.prompt_eval_count,
            }
            f.write(json.dumps(result) + "\n")
            results.append(result)
            print(f"{row['id']}: {prediction or 'INVALID'}  "
                  f"({seconds:.1f} s, {response.prompt_eval_count} prompt tokens)")
    return results


def training_subset():
    """The same N_TRAIN training points every time, half positive and half negative, without the few-shot
    examples. Balanced so prompt versions can be told apart: at the real ~7% positive rate, a small subset
    has almost no positives. Validation and test stay unbalanced, so their metrics match the real data."""
    return load_rows(TRAIN_SPLIT, N_TRAIN, set(FEWSHOT_IDS), balanced=True)


def validation_subset():
    return load_rows(VALIDATION_SPLIT, N_VAL)


def test_subset():
    return load_rows(TEST_SPLIT, N_TEST)


def compare(versions, rows, split):
    """Score each version on the same rows, and list fixes and breaks vs. the first version."""
    unknown = [v for v in versions if v not in PROMPTS]
    if unknown:
        sys.exit(f"No such prompt(s) in PROMPTS: {unknown}")
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    t = np.array([row["label"] for row in rows], dtype="float32")
    correct = {}  # version -> [True/False for each point]
    for version in versions:
        results = run(version, rows, RUNS_DIR / f"{version}_{split}.jsonl")
        correct[version] = [routed(r["prediction"]) == row["label"] for r, row in zip(results, rows)]
        y = np.array([score(r["prediction"]) for r in results], dtype="float32")
        metrics = compute_metrics(y, t, ROUTING_CUTOFF)
        invalid = sum(r["prediction"] is None for r in results)
        seconds = sum(r["seconds"] for r in results) / len(results)
        print(f"{version}: accuracy {metrics['accuracy']:.3f}, precision {metrics['precision']:.3f}, "
              f"recall {metrics['recall']:.3f}, F1 {metrics['f1']:.3f} at cutoff {ROUTING_CUTOFF}, "
              f"AUC {metrics['auc']:.3f} on {metrics['points']} {split} points "
              f"({metrics['positives']} positive), {invalid} invalid, {seconds:.1f} s per point")

    # Fixes and breaks: points that became right or wrong, compared with the first version.
    print()
    base = versions[0]
    for version in versions[1:]:
        pairs = list(zip(rows, correct[base], correct[version]))
        fixes = [row["id"] for row, old, new in pairs if new and not old]
        breaks = [row["id"] for row, old, new in pairs if old and not new]
        print(f"{version} vs. {base}: fixes {fixes or 'none'}, breaks {breaks or 'none'}")


def test(version):
    """Run ONCE, with the version chosen on the validation set. The test labels are never read."""
    if version not in PROMPTS:
        sys.exit(f"No such prompt in PROMPTS: {version}")
    rows = test_subset()
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    results = run(version, rows, RUNS_DIR / f"{version}_test.jsonl")
    with open(LLM_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Example ID", "LLM Prediction"])
        for r in results:
            writer.writerow([r["id"], r["prediction"] or "INVALID"])
    print(f"\nWrote {len(results)} predictions to {LLM_CSV}.")


if __name__ == "__main__":
    if len(sys.argv) == 1:
        compare(list(PROMPTS), training_subset(), "train")
    elif len(sys.argv) >= 3 and sys.argv[1] == "val":
        compare(sys.argv[2:], validation_subset(), "val")
    elif len(sys.argv) == 3 and sys.argv[1] == "test":
        test(sys.argv[2])
    else:
        sys.exit(__doc__)
