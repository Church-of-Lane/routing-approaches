from pathlib import Path

from symbolic import Symbolic

target = 10

# relative to script so it works no matter the directory.
DATA_DIR = Path(__file__).resolve().parent / "../../../data/irc-data/test"

# only chat logs in ascii format
FILE_PATTERN = "*.ascii.txt"

# just so we're not printing literally everything, that's like 300 files of 5000 lines each
VERBOSE = False

def score_file(path: Path, evaluator_cache: dict[str, Symbolic]) -> dict[tuple[str, int], list[tuple[str, float]]]:
    data = path.read_text(encoding="utf-8", errors="replace").split("\n")

    # only users in this file are candidates for this file's messages
    names = set(filter(lambda x: x is not None, map(Symbolic.extract_username, data)))

    # Reuse evaluators cross-file: otherwise we load a SpellChecker dict, which is slow to do for every user per file.
    for name in names:
        if name not in evaluator_cache:
            evaluator_cache[name] = Symbolic(name)
    evaluators = [evaluator_cache[name] for name in names]
    message_scores = {}
    
    # New window/file, so we don't span two.
    window = []
    for index, line in enumerate(data):
        if len(window) == target:
            window.pop(0)
        window.append(line)
        if len(window) == target and Symbolic.extract_username(window[-1]) is not None:
            scores = [(evaluator.get_name(), evaluator.score_text(window)) for evaluator in evaluators]
            message_scores[(path.name, index)] = scores
            if VERBOSE:
                print(scores)
    return message_scores


if __name__ == "__main__":
    files = sorted(DATA_DIR.glob(FILE_PATTERN))
    if not files:
        raise FileNotFoundError(f"No files matching {FILE_PATTERN} in {DATA_DIR}")
    evaluator_cache: dict[str, Symbolic] = {}
    winners = []
    for number, path in enumerate(files, start=1):
        message_scores = score_file(path, evaluator_cache)
        for (file_name, index), scores in message_scores.items():
            winners.append((file_name, index, max(scores, key=lambda x: x[1])))
        print(f"[{number}/{len(files)}] {path.name}: {len(message_scores)} messages scored")
    print("\n\n")
    print(winners)
