from dataclasses import dataclass, field
from pathlib import Path

from symbolic import Symbolic

"""
Loading for the irc-disentanglement data (data/irc-data).

Each log comes as a pair of files:
1. <name>.ascii.txt       the chat log, one message per line
2. <name>.annotation.txt  lines like "1002 1003 -": message 1003 is a reply to message 1002

Line numbers are 0-indexed line positions in the ascii file. Only messages from 1000 onward
are annotated; the first 1000 lines are context. A self-link ("1002 1002 -") means the message
isn't a reply to anything (it starts a new conversation, or is a system line).
"""

# relative to script so it works no matter the directory.
DATA_DIR = Path(__file__).resolve().parent / "../../data/irc-data"

LOG_SUFFIX = ".ascii.txt"
ANNOTATION_SUFFIX = ".annotation.txt"

# annotations start at this line in every file
FIRST_ANNOTATED_LINE = 1000

@dataclass
class Message:
    username: str | None
    responders: list[str]
    text: str
    annotated: bool

@dataclass
class MessagedLog:
    name: str
    messages: list[Message]

@dataclass
class AnnotatedLog:
    name: str
    lines: list[str]
    # message index -> indices of the messages it replies to (can be several, or itself)
    replies_to: dict[int, list[int]] = field(default_factory=dict)

    def annotated_indices(self) -> list[int]:
        return sorted(self.replies_to)

    def parents(self, index: int) -> list[int]:
        # self-links are dropped, so an empty list means "not a reply"
        return [parent for parent in self.replies_to.get(index, []) if parent != index]


def load_lines(path: Path) -> list[str]:
    data = path.read_text(encoding="utf-8", errors="replace").split("\n")

    # trailing newline would otherwise add an empty phantom line at the end
    if data and data[-1] == "":
        data.pop()

    return data


def load_annotations(path: Path) -> dict[int, list[int]]:
    replies_to: dict[int, list[int]] = {}

    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) < 2:
            continue

        parent, child = int(parts[0]), int(parts[1])
        replies_to.setdefault(child, []).append(parent)

    return replies_to


def load_log(annotation_path: Path) -> AnnotatedLog:
    name = annotation_path.name.removesuffix(ANNOTATION_SUFFIX)
    log_path = annotation_path.with_name(name + LOG_SUFFIX)

    return AnnotatedLog(name, load_lines(log_path), load_annotations(annotation_path))


def load_split(split: str = "train", data_dir: Path = DATA_DIR) -> list[AnnotatedLog]:
    directory = data_dir / split
    annotation_paths = sorted(directory.glob("*" + ANNOTATION_SUFFIX))

    if not annotation_paths:
        raise FileNotFoundError(f"No files matching *{ANNOTATION_SUFFIX} in {directory}")

    return [load_log(path) for path in annotation_paths]


def lines_to_messages(lines: list[str], replies_to: dict[int, list[int]] | None = None) -> list[Message]:
    # without annotations, no message has known responders and none count as annotated
    replies_to = replies_to or {}

    index_to_responders = {}

    for reply, parents in replies_to.items():
        replier = Symbolic.extract_username(lines[reply])

        for replied_to in parents:
            index_to_responders.setdefault(replied_to, []).append(replier)

    return [
        Message(
            Symbolic.extract_username(line),
            index_to_responders.get(i, []),
            Symbolic.extract_message(line) or "",
            bool(replies_to) and i >= FIRST_ANNOTATED_LINE,
        )
        for i, line in enumerate(lines)
    ]


def load_messaged_split(split: str = "train", data_dir: Path = DATA_DIR) -> list[MessagedLog]:
    return [
        MessagedLog(annotated_log.name, lines_to_messages(annotated_log.lines, annotated_log.replies_to))
        for annotated_log in load_split(split, data_dir)
    ]


def load_training_data(data_dir: Path = DATA_DIR) -> list[MessagedLog]:
    return load_messaged_split("train", data_dir)
