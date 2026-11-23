"""
Test harness for prompting an LLM on the routing task.

Task: given the last messages of an IRC log, score from 0 to 1 whether a candidate
(someone who spoke recently) should reply to the latest message.

Points come from util.iter_points, so the candidates and labels are the same ones the
symbolic and machine learning approaches are scored on.

Every prompt tried is in PROMPTS, and every LLM called is in MODELS (both run locally with Ollama).
Every response is saved in llm/runs/<model prefix><version>_<split>.jsonl, and reruns with an unchanged
setup load those instead of calling the LLM again. The metrics in the prompt log are compute_metrics in
evaluate_machine_learning.py, at ROUTING_CUTOFF, plus the count of invalid outputs and the average seconds per response.

    python llm_harness.py [--model NAME]             develop: score every prompt in PROMPTS on a fixed training subset
    python llm_harness.py [--model NAME] val v0 v1   compare your best few prompts on a fixed validation (dev) subset

NAME is a key of MODELS; without --model, every model is run. The best prompt and model are run on the
full test set by llm_inference.py, which writes llm_eval.csv.
"""

import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import ollama

from evaluate_machine_learning import compute_metrics
from machine_learning import CANDIDATE_POOL, HISTORY, ROUTING_CUTOFF, VALIDATION_SPLIT
from util import Point, iter_points, load_split

# Every LLM tried. prefix starts its run files' names (the llama runs came first, so they have none)
MODELS = {
    "llama": {"model": "llama3.2:3b", "prefix": ""},
    "gemma": {"model": "gemma4:e4b",  "prefix": "gemma_"},
}
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

TRAIN_SPLIT = "train"

# Train (~440k points) and dev (~16k) have far too many points to develop prompts on all of them
# (~7% positive), so each one is a fixed random subset. Develop prompts on the training subset.
# The LLM was never trained on these logs, so scoring them is fine, as long as the few-shot
# examples (which are in a prompt, with their labels) are left out. The test set is scored in
# full, once, by llm_inference.py.
N_TRAIN = 50  # half positive, half negative (see training_subset)
N_VAL   = 200
SAMPLE_SEED = 0
# ids ("log:index:candidate") of every training point used in any prompt. Picked at random
# (seed 1) from the training points outside every fixed subset.
FEWSHOT_IDS = [
    "2005-06-16.train-c:1410:bob2",    # positive
    "2018-02-27.train-c:1168:frazr",   # negative
    "2005-12-03.train-c:1467:Locke",   # negative
]

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

Score the latest message from 0 to 1.""",

"v3": f"""Decide whether {{name}} should reply to what was just said.

<examples>
{{examples}}
</examples>

Conversation so far:
<transcript>
{{transcript}}
</transcript>

Score the latest message from 0 to 1."""
}


def load_rows(split, n, exclude=(), balanced=False):
    """A fixed random subset of n points from a split, as dicts of id, name, transcript and label.
    The transcript is the last HISTORY lines of the window, like the symbolic approach sees.
    If balanced, half the points are positive and half negative, in shuffled order.
    Sampling ignores exclude, so adding a few-shot example never changes the subset;
    instead it is an error for an excluded point to be in it."""
    rows = [point_row(point) for log in load_split(split) for point in iter_points(log, CANDIDATE_POOL)]
    rng = random.Random(SAMPLE_SEED)
    if not balanced:
        return check_excluded(rng.sample(rows, min(n, len(rows))), exclude)

    positive = [row for row in rows if row["label"]]
    negative = [row for row in rows if not row["label"]]
    if len(positive) < n // 2 or len(negative) < n - n // 2:
        raise ValueError(f"{split} has {len(positive)} positive and {len(negative)} negative points, too few for {n} balanced")
    sample = rng.sample(positive, n // 2) + rng.sample(negative, n - n // 2)
    rng.shuffle(sample)
    return check_excluded(sample, exclude)


def point_row(point: Point):
    """A point as the LLM sees it: the candidate's name, and the last HISTORY lines of the window."""
    return {
        "id": f"{point.log}:{point.index}:{point.candidate}",
        "name": point.candidate,
        "transcript": "\n".join(point.window[-HISTORY:]),
        "label": point.label,
    }


def check_excluded(rows, exclude):
    overlap = [row["id"] for row in rows if row["id"] in exclude]
    if overlap:
        raise ValueError(f"Excluded points are in the fixed subset, so pick different ones: {overlap}")
    return rows


def format_examples():
    """The FEWSHOT_IDS training points, written out for {examples}, with the gold label as the answer."""
    rows = {row["id"]: row for row in load_rows(TRAIN_SPLIT, sys.maxsize)}
    return "\n\n".join(
        f"""<example>
Should {rows[id]["name"]} reply?
<transcript>
{rows[id]["transcript"]}
</transcript>
{json.dumps({"answer": "1.0" if rows[id]["label"] else "0.0"})}
</example>"""
        for id in FEWSHOT_IDS
    )


def with_examples(prompt):
    """Fills in {examples}, only loading the training split when the prompt uses them."""
    return prompt.replace("{examples}", format_examples()) if "{examples}" in prompt else prompt


def fill(prompt, row):
    return prompt.replace("{name}", row["name"]).replace("{transcript}", row["transcript"])


def score(prediction):
    """The prediction as a number. Invalid outputs score 0, so they count as "don't reply"."""
    return float(prediction) if prediction is not None else 0.0


def routed(prediction):
    """Whether a prediction counts as "should reply". Invalid outputs count as no."""
    return prediction is not None and float(prediction) >= ROUTING_CUTOFF


def predict(prompt, row, model):
    """Ask the model (an Ollama model name) about one point. Returns (prediction, response, seconds).
    The prediction is None if the output is not one of the labels."""
    start = time.perf_counter()
    response = ollama.chat(
        model=model,
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


def run(version, rows, out_file, model):
    """Run one prompt on every row with model (an Ollama model name), and save every response to out_file.
    The first line of out_file records the setup. If out_file already has the
    outputs for exactly this setup and these rows, they are loaded instead."""
    prompt = with_examples(PROMPTS[version])
    setup = {"model": model, "options": OPTIONS, "schema": SCHEMA, "prompt": prompt}
    if Path(out_file).exists():
        try:
            with open(out_file, encoding="utf-8") as f:
                lines = [json.loads(line) for line in f if line.strip()]
        except json.JSONDecodeError:  # e.g. two runs wrote the same file at once, so redo it
            lines = []
        if lines and lines[0] == setup and [r["id"] for r in lines[1:]] == [row["id"] for row in rows]:
            print(f"\n{model} {version}: nothing changed, so using the saved outputs in {out_file}")
            return lines[1:]

    print(f"\n===== {model} {version}: the exact prompt for {rows[0]['id']} =====")
    print(fill(prompt, rows[0]))
    print("=====")
    results = []
    with open(out_file, "w", encoding="utf-8") as f:
        f.write(json.dumps(setup) + "\n")
        for row in rows:
            prediction, response, seconds = predict(prompt, row, model)
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


def print_table(table):
    columns = list(next(iter(table.values())))
    print(f"{'version':<10}" + "".join(f"{column:>11}" for column in columns))
    for version, metrics in table.items():
        print(f"{version:<10}" + "".join(
            f"{value:>11.3f}" if isinstance(value, float) else f"{value:>11}" for value in metrics.values()
        ))


def compare(versions, rows, split, name):
    """Score each version with MODELS[name] on the same rows, and list fixes and breaks vs. the first version."""
    unknown = [v for v in versions if v not in PROMPTS]
    if unknown:
        sys.exit(f"No such prompt(s) in PROMPTS: {unknown}")
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    t = np.array([row["label"] for row in rows], dtype="float32")
    correct = {}  # version -> [True/False for each point]
    table = {}    # version -> metrics
    for version in versions:
        model = MODELS[name]
        results = run(version, rows, RUNS_DIR / f"{model['prefix']}{version}_{split}.jsonl", model["model"])
        correct[version] = [routed(r["prediction"]) == row["label"] for r, row in zip(results, rows)]
        y = np.array([score(r["prediction"]) for r in results], dtype="float32")
        table[version] = {
            **compute_metrics(y, t, ROUTING_CUTOFF),
            "invalid": sum(r["prediction"] is None for r in results),
            "avg_secs": sum(r["seconds"] for r in results) / len(results),  # per response, from the saved runs
        }

    print(f"\n=== {MODELS[name]['model']}, {split}, cutoff {ROUTING_CUTOFF} ===")
    print_table(table)

    # Fixes and breaks: points that became right or wrong, compared with the first version.
    print()
    base = versions[0]
    for version in versions[1:]:
        pairs = list(zip(rows, correct[base], correct[version]))
        fixes = [row["id"] for row, old, new in pairs if new and not old]
        breaks = [row["id"] for row, old, new in pairs if old and not new]
        print(f"{version} vs. {base}: fixes {fixes or 'none'}, breaks {breaks or 'none'}")


if __name__ == "__main__":
    args = sys.argv[1:]
    names = list(MODELS)
    if args[:1] == ["--model"]:
        if len(args) < 2 or args[1] not in MODELS:
            sys.exit(f"--model must be one of {list(MODELS)}")
        names, args = [args[1]], args[2:]

    if not args:
        rows = training_subset()
        for name in names:
            compare(list(PROMPTS), rows, "train", name)
    elif len(args) >= 2 and args[0] == "val":
        rows = validation_subset()
        for name in names:
            compare(args[1:], rows, "val", name)
    else:
        sys.exit(__doc__)
