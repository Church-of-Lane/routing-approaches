"""
Prompts the best LLM setup from llm_harness.py to score whether a candidate replies to the latest message of a chat.

Best setup: prompt v0 on gemma4:e4b (run locally with Ollama: `ollama pull gemma4:e4b`). On the harness's balanced
training subset it had the best F1 (0.828) and AUC (0.870) of every prompt and model tried. llama3.2:3b's best was F1 0.565 / AUC 0.606.

    python llm_inference.py [sample] [candidate]                 score the test set sample (test_sample.txt, for crimsun)
    python llm_inference.py --eval [workers] [--points A-B]      score the full test set, and write llm_eval.csv

--eval is ~30k LLM calls (~3 s each on gemma4:e4b), so every response is saved as it comes in, and a rerun
skips every point that already has one. With workers > 1, that many requests run at once (start Ollama with
OLLAMA_NUM_PARALLEL set to at least that many, or they just queue).

--points A-B only scores test points A to B (1-based and inclusive, e.g. 1-1000), in the same fixed order
every time (log, then line, then candidate), so the test set can be done in chunks. A B past the last
point stops at the last point.

Several instances can run at once, on one machine or on many sharing the same storage. Each one writes only to its
own file, llm/runs/gemma_v0_test/<host>-<pid>-<start time>.jsonl, so no file ever has two writers. Every instance
reads all the files in that folder to skip points that are already done, and the one that finds every point done
writes llm_eval.csv. Give instances ranges that don't overlap: ones that do will score the same points twice
(harmless, but wasted; the first response saved for a point is the one used). Shared storage can be a little slow to
show one machine's writes to another, so if the last instances finish together and none writes llm_eval.csv,
run --eval once more: with nothing left to score, it just writes the CSV.

The first line of every run file records the setup (model, options, schema and prompt). If the files in
gemma_v0_test/ were made with a different setup, the run goes to gemma_v0_test_2/ instead (or _3, and so on):
whichever has this exact setup, or else the first that doesn't exist yet. Its CSV gets the same number
(llm_eval_2.csv), so no setup's results overwrite another's. eval.py reads llm_eval.csv.
A single llm/runs/gemma_v0_test.jsonl from before run folders existed is read as part of gemma_v0_test/, but never written to.
"""

import json
import os
import re
import socket
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from llm_harness import MODELS, OPTIONS, PROMPTS, RUNS_DIR, SCHEMA, fill, point_row, predict, score, with_examples
from machine_learning import CANDIDATE_POOL, HISTORY, ROUTING_CUTOFF
from util import FOLDER, TEST_SAMPLE, TEST_SAMPLE_CANDIDATE, load_test_sample, test_points, write_eval_csv

BEST_MODEL   = "gemma"  # a key of llm_harness.MODELS
BEST_VERSION = "v0"     # a key of llm_harness.PROMPTS

MODEL  = MODELS[BEST_MODEL]["model"]
PROMPT = with_examples(PROMPTS[BEST_VERSION])

LLM_EVAL_CSV = FOLDER / "llm_eval.csv"
TEST_RUN     = RUNS_DIR / f"{MODELS[BEST_MODEL]['prefix']}{BEST_VERSION}_test.jsonl"


def llm_predict(lines: list[str], candidate: str) -> tuple[float, str]:
    """
    The LLM's confidence (0 to 1) that candidate replies to the last line, and its raw output.
    It sees the last HISTORY lines, like in the harness. Invalid outputs score 0 ("don't reply")
    """
    row = {"name": candidate, "transcript": "\n".join(lines[-HISTORY:])}
    prediction, response, _ = predict(PROMPT, row, MODEL)
    return score(prediction), response.message.content


def numbered(path: Path, number: int) -> Path:
    """
    path for number 1, and path with _<number> before its extension after that
    """
    return path if number == 1 else path.with_name(f"{path.stem}_{number}{path.suffix}")


SETUP_KEYS = {"model", "options", "schema", "prompt"}


def read_lines(path: Path) -> list[tuple[int, object]]:
    """
    Every line of a run file, with its line number. A line cut off by a crash (or still being written) is dropped
    """
    lines = []
    with open(path, encoding="utf-8", errors="replace") as file:
        for number, line in enumerate(file, start=1):
            try:
                lines.append((number, json.loads(line)))
            except json.JSONDecodeError:
                pass
    return lines


def is_setup(line: object) -> bool:
    return isinstance(line, dict) and set(line) == SETUP_KEYS


def is_result(line: object) -> bool:
    return isinstance(line, dict) and isinstance(line.get("id"), str)


def run_folder(number: int) -> Path:
    """
    The folder of run files for setup number: gemma_v0_test/, gemma_v0_test_2/, ...
    """
    return numbered(TEST_RUN, number).with_suffix("")


def run_files(number: int) -> list[Path]:
    """
    Every run file for setup number: the old single file, if there is one, and every instance's file in its folder
    """
    old = numbered(TEST_RUN, number)
    return ([old] if old.exists() else []) + sorted(run_folder(number).glob("*.jsonl"))


def load_responses(number: int, setup: dict) -> dict[str, dict] | None:
    """
    Every saved response for setup number, by point id (the first saved, if a point has several),
    or None if any of its files was made with a different setup. Files with no lines yet are skipped
    """
    responses = {}

    for path in run_files(number):
        lines = read_lines(path)
        if not lines:
            continue
        if lines[0][1] != setup:
            return None

        # A file can have more setup lines further down, e.g. from two run files being joined. Results
        # count only while the latest setup line above them is this setup
        current = setup
        other   = 0
        stray   = []

        for line_number, line in lines[1:]:
            if is_setup(line):
                current = line
            elif not is_result(line):
                stray.append(line_number)
            elif current == setup:
                responses.setdefault(line["id"], line)
            else:
                other += 1

        if other:
            print(f"Warning: {path} has {other} results under a different setup line, so they're skipped")
        if stray:
            print(f"Warning: {path} has {len(stray)} lines that aren't results or setups, so they're skipped "
                  f"(line{'s' if len(stray) > 1 else ''} {', '.join(map(str, stray[:10]))}{', ...' if len(stray) > 10 else ''})")

    return responses


def find_run(setup: dict) -> tuple[int, dict[str, dict]]:
    """
    The setup number for this exact setup, and its saved responses by point id. That's the first of
    gemma_v0_test, gemma_v0_test_2, ... made with this setup, or else the first with no files yet (a new run)
    """
    number = 1
    while run_files(number):
        responses = load_responses(number, setup)
        if responses is not None:
            return number, responses

        print(f"{run_folder(number).name} was made with a different setup, so it's left alone")
        number += 1

    return number, {}


def instance_name() -> str:
    """
    This machine and process, safe to put in a filename
    """
    return f"{re.sub(r'[^A-Za-z0-9_.-]', '_', socket.gethostname()) or 'host'}-{os.getpid()}"


def own_file(number: int) -> Path:
    """
    A file only this instance writes: named by machine, process and start time, so no other instance picks the same one
    """
    return run_folder(number) / f"{instance_name()}-{time.strftime('%Y%m%d-%H%M%S')}.jsonl"


def parse_range(text: str) -> tuple[int, int]:
    """
    "A-B" -> (A, B), 1-based and inclusive
    """
    try:
        first, last = (int(part) for part in text.split("-"))
    except ValueError:
        raise SystemExit(f"--points takes a range like 1-1000, not {text!r}")

    if not 1 <= first <= last:
        raise SystemExit(f"--points {text}: needs 1 <= A <= B")

    return first, last


def evaluate(workers: int = 1, points_range: tuple[int, int] | None = None) -> None:
    points = test_points(CANDIDATE_POOL)
    rows   = [point_row(point) for point in points]
    setup  = {"model": MODEL, "options": OPTIONS, "schema": SCHEMA, "prompt": PROMPT}

    first, last = points_range or (1, len(rows))
    last = min(last, len(rows))
    if first > len(rows):
        raise SystemExit(f"--points starts at {first}, but the test set only has {len(rows)} points")

    run_number, done = find_run(setup)
    csv_path = numbered(LLM_EVAL_CSV, run_number)

    # (point number, row) for every point in the range without a response yet
    todo = [(number, row) for number, row in enumerate(rows, start=1)
            if first <= number <= last and row["id"] not in done]

    print(f"=== {MODEL} {BEST_VERSION}: {len(done)} of {len(rows)} test points already scored "
          f"(in {run_folder(run_number)}). Points {first}-{last}: {len(todo)} to go ===")

    if todo:
        path = own_file(run_number)
        path.parent.mkdir(parents=True, exist_ok=True)
        print(f"=== Saving responses to {path} ===")
        print(f"=== The exact prompt for {todo[0][1]['id']} ===\n{fill(PROMPT, todo[0][1])}\n=====")

        # "x" fails if the file already exists, so this instance is certainly its only writer
        with open(path, "x", encoding="utf-8") as file, ThreadPoolExecutor(workers) as pool:
            save(file, setup)

            # map keeps the order, and only this thread writes, so the file is never interleaved
            for count, ((number, row), (prediction, response, seconds)) in enumerate(
                    zip(todo, pool.map(lambda item: predict(PROMPT, item[1], MODEL), todo)), start=1):
                result = {
                    "id": row["id"],
                    "prediction": prediction,
                    "raw": response.message.content,
                    "seconds": round(seconds, 2),
                    "prompt_tokens": response.prompt_eval_count,
                }
                save(file, result)
                done.setdefault(row["id"], result)

                print(f"[point {number}, {count}/{len(todo)} this run, {len(done)}/{len(rows)} total] "
                      f"{row['id']}: {prediction or 'INVALID'} ({seconds:.1f} s)")

    # read everything again, to count what other instances saved while this one ran
    done = load_responses(run_number, setup)
    if done is None:
        raise SystemExit(f"Another instance with a different setup wrote to {run_folder(run_number)}, so it can't be used")

    missing = [number for number, row in enumerate(rows, start=1) if row["id"] not in done]
    if missing:
        print(f"=== {len(rows) - len(missing)} of {len(rows)} test points scored, so {csv_path.name} isn't written yet. "
              f"{len(missing)} missing, the first being point {missing[0]} ===")
        return

    invalid = sum(done[row["id"]]["prediction"] is None for row in rows)
    print(f"=== {invalid} invalid outputs, scored as 0 ===")

    # written beside it and then renamed, so an instance finishing at the same time can't leave a mix of both
    partial = csv_path.with_name(f".{csv_path.name}.{instance_name()}.tmp")
    write_eval_csv(partial, points, [score(done[row["id"]]["prediction"]) for row in rows], ROUTING_CUTOFF)
    os.replace(partial, csv_path)
    print(f"=== Saved as {csv_path} ===")


def save(file, line: dict) -> None:
    """
    Writes one line all the way to storage, so instances on other machines can see it
    """
    file.write(json.dumps(line) + "\n")
    file.flush()
    os.fsync(file.fileno())


if __name__ == "__main__":
    from sys import argv

    args = argv[1:]
    points_range = None
    if "--points" in args:
        at = args.index("--points")
        if at + 1 >= len(args):
            raise SystemExit("--points takes a range like 1-1000")
        points_range = parse_range(args[at + 1])
        del args[at:at + 2]

    if args[:1] == ["--eval"]:
        evaluate(int(args[1]) if len(args) > 1 else 1, points_range)
    elif points_range is not None:
        raise SystemExit("--points only works with --eval")
    else:
        sample    = Path(argv[1]) if len(argv) > 1 else TEST_SAMPLE
        candidate = argv[2]       if len(argv) > 2 else TEST_SAMPLE_CANDIDATE
        lines     = load_test_sample(sample)

        confidence, raw = llm_predict(lines, candidate)

        print(f"Latest message: {lines[-1]}")
        print(f"{MODEL} said: {raw}")
        print(f"Should {candidate} reply? {confidence:.1f} -> {'yes' if confidence >= ROUTING_CUTOFF else 'no'} (cutoff {ROUTING_CUTOFF})")
