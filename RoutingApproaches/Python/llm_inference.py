"""
Prompts the best LLM setup from llm_harness.py to score whether a candidate replies to the latest message of a chat.

Best setup: prompt v0 on gemma4:e4b (run locally with Ollama: `ollama pull gemma4:e4b`). On the harness's balanced
training subset it had the best F1 (0.828) and AUC (0.870) of every prompt and model tried. llama3.2:3b's best was F1 0.565 / AUC 0.606.

    python llm_inference.py [sample] [candidate]   score the test set sample (test_sample.txt, for crimsun)
    python llm_inference.py --eval [workers]       score every point in the full test set, and write llm_eval.csv

--eval is ~30k LLM calls (~3 s each on gemma4:e4b), so every response is appended to
llm/runs/gemma_v0_test.jsonl as it comes in, and a rerun picks up where the last one stopped.
llm_eval.csv is written once every point has a response. With workers > 1, that many requests run at once
(start Ollama with OLLAMA_NUM_PARALLEL set to at least that many, or they just queue).
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


def load_done(path: Path, setup: dict) -> dict[str, dict]:
    """
    Responses already saved for this exact setup, by point id. A line cut off by a crash is dropped
    """
    if not path.exists():
        return {}

    lines = []
    with open(path, encoding="utf-8") as file:
        for line in file:
            try:
                lines.append(json.loads(line))
            except json.JSONDecodeError:
                pass

    if lines and lines[0] != setup:
        raise SystemExit(f"{path} was made with a different model, options or prompt. Move it aside to start over.")

    return {result["id"]: result for result in lines[1:]}


def evaluate(workers: int = 1) -> None:
    points = test_points(CANDIDATE_POOL)
    rows   = [point_row(point) for point in points]
    setup  = {"model": MODEL, "options": OPTIONS, "schema": SCHEMA, "prompt": PROMPT}

    TEST_RUN.parent.mkdir(parents=True, exist_ok=True)
    done = load_done(TEST_RUN, setup)
    todo = [row for row in rows if row["id"] not in done]

    print(f"=== {MODEL} {BEST_VERSION}: {len(done)} of {len(rows)} test points already scored, {len(todo)} to go ===")
    print(f"=== The exact prompt for {rows[0]['id']} ===\n{fill(PROMPT, rows[0])}\n=====")

    # rewritten from what loaded, so it's always the setup line followed by whole results
    with open(TEST_RUN, "w", encoding="utf-8") as file, ThreadPoolExecutor(workers) as pool:
        file.write(json.dumps(setup) + "\n")
        for result in done.values():
            file.write(json.dumps(result) + "\n")
        file.flush()

        # map keeps the order, and only this thread writes, so the file is never interleaved
        for number, (row, (prediction, response, seconds)) in enumerate(
                zip(todo, pool.map(lambda row: predict(PROMPT, row, MODEL), todo)), start=1):
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

            print(f"[{len(done)}/{len(rows)}] {row['id']}: {prediction or 'INVALID'} ({seconds:.1f} s)")

    invalid = sum(done[row["id"]]["prediction"] is None for row in rows)
    print(f"=== {invalid} invalid outputs, scored as 0 ===")

    write_eval_csv(LLM_EVAL_CSV, points, [score(done[row["id"]]["prediction"]) for row in rows], ROUTING_CUTOFF)


if __name__ == "__main__":
    from sys import argv

    if argv[1:2] == ["--eval"]:
        evaluate(int(argv[2]) if len(argv) > 2 else 1)
    else:
        sample    = Path(argv[1]) if len(argv) > 1 else TEST_SAMPLE
        candidate = argv[2]       if len(argv) > 2 else TEST_SAMPLE_CANDIDATE
        lines     = load_test_sample(sample)

        confidence, raw = llm_predict(lines, candidate)

        print(f"Latest message: {lines[-1]}")
        print(f"{MODEL} said: {raw}")
        print(f"Should {candidate} reply? {confidence:.1f} -> {'yes' if confidence >= ROUTING_CUTOFF else 'no'} (cutoff {ROUTING_CUTOFF})")
