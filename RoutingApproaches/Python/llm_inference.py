"""
Prompts the best LLM setup from llm_harness.py to score whether a candidate replies to the latest message of a chat.

Best setup: prompt v0 on gemma4:e4b (run locally with Ollama: `ollama pull gemma4:e4b`). On the harness's balanced
training subset it had the best F1 (0.828) and AUC (0.870) of every prompt and model tried. llama3.2:3b's best was F1 0.565 / AUC 0.606.

    python llm_inference.py [sample] [candidate]                 score the test set sample (test_sample.txt, for crimsun)
    python llm_inference.py --eval [workers] [--points A-B]      score the full test set, and write llm_eval.csv

--eval is ~30k LLM calls (~3 s each on gemma4:e4b), so every response is appended to
llm/runs/gemma_v0_test.jsonl as it comes in, and a rerun picks up where the last one stopped.
The first line of a run file records the setup (model, options, schema and prompt). If gemma_v0_test.jsonl was made
with a different setup, the run goes to gemma_v0_test_2.jsonl instead (or _3, and so on): whichever has this exact
setup, or else the first that doesn't exist yet. Its CSV gets the same number (llm_eval_2.csv), so no setup's
results overwrite another's. eval.py reads llm_eval.csv.
llm_eval.csv is written once every point has a response. With workers > 1, that many requests run at once
(start Ollama with OLLAMA_NUM_PARALLEL set to at least that many, or they just queue).

--points A-B only scores test points A to B (1-based and inclusive, e.g. 1-1000), in the same fixed order
every time (log, then line, then candidate), so the test set can be done in chunks. A B past the last
point stops at the last point. Chunks run on different machines can be combined by appending one run file's
result lines (everything but its first, setup, line) to the other's.
"""

import json
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


def read_lines(path: Path) -> list[dict]:
    """
    Every line of a run file. A line cut off by a crash is dropped
    """
    lines = []
    with open(path, encoding="utf-8") as file:
        for line in file:
            try:
                lines.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return lines


def find_run(setup: dict) -> tuple[int, dict[str, dict]]:
    """
    The number of the run file for this exact setup, and its saved responses by point id. That's the first of
    TEST_RUN, TEST_RUN_2, ... made with this setup, or else the first that doesn't exist yet (a new run)
    """
    number = 1
    while (path := numbered(TEST_RUN, number)).exists():
        lines = read_lines(path)
        if not lines or lines[0] == setup:
            return number, {result["id"]: result for result in lines[1:]}

        print(f"{path.name} was made with a different setup, so it's left alone")
        number += 1

    return number, {}


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

    TEST_RUN.parent.mkdir(parents=True, exist_ok=True)
    number, done = find_run(setup)
    run_path = numbered(TEST_RUN, number)
    csv_path = numbered(LLM_EVAL_CSV, number)
    print(f"=== Saving responses to {run_path} ===")

    # (point number, row) for every point in the range without a response yet
    todo = [(number, row) for number, row in enumerate(rows, start=1)
            if first <= number <= last and row["id"] not in done]

    print(f"=== {MODEL} {BEST_VERSION}: {len(done)} of {len(rows)} test points already scored. "
          f"Points {first}-{last}: {len(todo)} to go ===")
    print(f"=== The exact prompt for {rows[0]['id']} ===\n{fill(PROMPT, rows[0])}\n=====")

    # rewritten from what loaded, so it's always the setup line followed by whole results
    with open(run_path, "w", encoding="utf-8") as file, ThreadPoolExecutor(workers) as pool:
        file.write(json.dumps(setup) + "\n")
        for result in done.values():
            file.write(json.dumps(result) + "\n")
        file.flush()

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
            file.write(json.dumps(result) + "\n")
            file.flush()
            done[row["id"]] = result

            print(f"[point {number}, {count}/{len(todo)} this run, {len(done)}/{len(rows)} total] "
                  f"{row['id']}: {prediction or 'INVALID'} ({seconds:.1f} s)")

    missing = [number for number, row in enumerate(rows, start=1) if row["id"] not in done]
    if missing:
        print(f"=== {len(rows) - len(missing)} of {len(rows)} test points scored, so {csv_path.name} isn't written yet. "
              f"{len(missing)} missing, the first being point {missing[0]} ===")
        return

    invalid = sum(done[row["id"]]["prediction"] is None for row in rows)
    print(f"=== {invalid} invalid outputs, scored as 0 ===")

    write_eval_csv(csv_path, points, [score(done[row["id"]]["prediction"]) for row in rows], ROUTING_CUTOFF)


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
