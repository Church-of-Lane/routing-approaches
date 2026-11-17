import json
import random

import keras
import numpy as np
from keras.src.legacy.preprocessing.text import Tokenizer, tokenizer_from_json
from keras.utils import pad_sequences
from util import load_training_data, load_messaged_split, Message, MessagedLog, lines_to_messages

import re
from functools import lru_cache
from pathlib import Path
from time import time

from tqdm.keras import TqdmCallback

SEED               = 1432516 # Zohar's favorite number
EPOCHS             = 100
VERBOSE            = 0
HISTORY            = 20  # messages the text model reads
CANDIDATE_POOL     = 20  # messages candidates are drawn from
FEATURE_HISTORY    = 100 # messages the hand-made features look back over
FEATURE_COUNT      = 88
TOKENS_PER_MESSAGE = 32
TEXT_VECTOR_SIZE   = 8
REPLY_HORIZON      = 20     # logs end mid-conversation: a message this close to the end can't show its reply (95% land within 21 lines)
VOCAB_LIMIT        = 10_000 # rarer words become <OOV>, otherwise the embedding memorizes them
ROUTING_CUTOFF     = 0.599 # on calibrated scores, picked on dev for precision >= 0.7 at recall >= 0.4 (see models/models.md, v6). Raw 0.55 = Platt 0.599
POSITIVE_WEIGHT    = 1.0 # 3.0 moved the F1-optimal threshold to ROUTING_CUTOFF, but routing wants precision, so stay calibrated
CALIBRATION        = "platt" # rescales scores after training, fitted on dev: "platt" (a * logit + b) or "temperature" (logit / T)
BATCH_SIZE         = 512
ENSEMBLE_SIZE      = 5   # averaged members; single models differ by ~0.01 precision just from the seed
LEARNING_RATE      = 1e-3
VALIDATION_SPLIT   = "dev"

SELF_TAG   = "xxself"
OTHER_TAG  = "xxother"
TARGET_TAG = "xxtarget"

random     .seed           (SEED)
keras.utils.set_random_seed(SEED)

# Used for seeing how long different things take
time_elapsed = 0

"""
Things to do:
    - Evaluate final model with test set
"""

def name_pattern(person: str) -> str:
    """
    Matches a username as a whole word. Lookarounds instead of \\b, since \\b fails next to
    the non-word characters IRC nicks often start or end with (Steve^, benh`, [[mandrix]])
    """
    return rf"(?<!\w){re.escape(person)}(?!\w)"

def normalize_name(person: str) -> str:
    return re.sub(r"[^a-z0-9]", "", person.lower())

@lru_cache(maxsize=1_000_000)
def mentions(text: str, person: str) -> tuple[bool, bool]:
    """
    (strict, loose) mention of person in text. Strict is a whole-word match, loose also accepts
    abbreviated or decorated nicks in the first few words ("bob" for bob2, "xabbu" for xabbu|)
    """
    strict = re.search(name_pattern(person), text, flags=re.IGNORECASE) is not None

    if strict:
        return True, True

    name = normalize_name(person)
    if len(name) < 3:
        return False, False

    for word in (normalize_name(w) for w in text.split()[:3]):
        if word and (word.startswith(name) or (len(word) >= 3 and name.startswith(word))):
            return False, True

    return False, False

def log_distance(distance: int) -> float:
    return np.log1p(distance) / np.log1p(FEATURE_HISTORY + 1)

def words(text: str) -> set[str]:
    return set(w for w in re.findall(r"[a-z0-9']+", text.lower()) if len(w) > 2)

def is_bot(person: str | None) -> bool:
    return person is not None and re.search(r"bot|^ubotu|chanserv", person.lower()) is not None

def minutes_between(earlier: int | None, later: int | None, cap: int) -> float:
    """
    Log-scaled minutes between two timestamps (wrapping past midnight), capped and scaled to [0, 1]
    """
    gap = cap if earlier is None or later is None else min((later - earlier) % 1440, cap)
    return float(np.log1p(gap) / np.log1p(cap))

# short acknowledgements ("ok then", "bob: thanks!"), which rarely get answered
ACKNOWLEDGEMENT = re.compile(r"^(\S+[:,]\s*)?(ok|okay|k|kk|ah+|oh+|np|yes|yeah|yep|no|nope|cool|great|nice|right|sure|thanks|thank you|thx|ty|cheers|alright|good|i see|lol|hehe|:\)|:-\))\W*( then| thanks| man| again)?\W*$")
QUESTION_WORD   = re.compile(r"\b(how|what|why|where|when|which|who|can|could|does|do|is|are|should|anyone|anybody|any)\b")
# a line that looks like it will be continued on the next one
UNFINISHED      = re.compile(r"(\.\.+|,|:|\b(and|but|so|or|then|because|if|with|the|a|to))$")

class MessagePooling(keras.layers.Layer):
    """
    Averages each message's token embeddings, ignoring padding
    """
    def call(self, embedded, tokens):
        mask   = keras.ops.expand_dims(keras.ops.cast(keras.ops.not_equal(tokens, 0), embedded.dtype), -1)
        summed = keras.ops.sum(embedded * mask, axis=2)
        count  = keras.ops.maximum(keras.ops.sum(mask, axis=2), 1.0)  # avoids 0/0 on empty messages
        return summed / count

def to_logit(p: np.ndarray) -> np.ndarray:
    """
    Inverse of the sigmoid. Clipped so a saturated 0 or 1 doesn't become +-inf
    """
    p = np.clip(np.asarray(p, dtype="float64"), 1e-7, 1 - 1e-7)
    return np.log(p) - np.log1p(-p)

def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))

def nll(p: np.ndarray, t: np.ndarray) -> float:
    p = np.clip(np.asarray(p, dtype="float64").reshape(-1), 1e-7, 1 - 1e-7)
    t = np.asarray(t, dtype="float64").reshape(-1)
    return float(-np.mean(t * np.log(p) + (1 - t) * np.log1p(-p)))

def apply_calibration(p: np.ndarray, a: float, b: float) -> np.ndarray:
    """
    Rescales a score: take the logit, compute a * logit + b, re-apply the sigmoid.
    - temperature scaling is b = 0, a = 1/T (T < 1 sharpens, T > 1 softens)
    - Platt scaling fits both, so it can also shift scores up or down
    With a > 0 it's monotonic, so the ranking (and AUC) doesn't change, only what a given score means
    """
    return sigmoid(a * to_logit(p) + b)

def fit_calibration(p: np.ndarray, t: np.ndarray, method: str, iterations: int = 100) -> tuple[float, float]:
    """
    Finds the (a, b) that minimize the NLL of apply_calibration(p, a, b) against labels t.
    "temperature" keeps b = 0, "platt" fits both. Either way it's a logistic regression on the logit,
    so the NLL is convex and Newton's method (with step halving, in case it overshoots) converges quickly
    """
    if method not in ("temperature", "platt"):
        raise ValueError(f"Unknown calibration method {method!r}")

    z = to_logit(p).reshape(-1)
    t = np.asarray(t, dtype="float64").reshape(-1)

    # columns are the inputs a and b multiply
    inputs = np.stack([z, np.ones_like(z)], axis=1) if method == "platt" else z[:, None]
    params = np.zeros(inputs.shape[1])
    params[0] = 1.0 # start at "no change"

    def loss(w: np.ndarray) -> float:
        return nll(sigmoid(inputs @ w), t)

    for _ in range(iterations):
        q    = sigmoid(inputs @ params)
        grad = inputs.T @ (q - t)
        hess = (inputs * (q * (1 - q))[:, None]).T @ inputs
        step = np.linalg.solve(hess + 1e-9 * np.eye(len(params)), grad)

        scale, current = 1.0, loss(params)
        while loss(params - scale * step) > current and scale > 1e-6:
            scale /= 2

        params -= scale * step

        if np.max(np.abs(scale * step)) < 1e-9:
            break

    a = float(params[0])
    b = float(params[1]) if method == "platt" else 0.0

    if not np.isfinite(a) or not np.isfinite(b) or a <= 0:
        raise ValueError(f"{method} fit failed (a = {a}, b = {b})")

    return a, b

def default_calibration_path(weight_path: str) -> Path:
    """
    The calibration lives next to the weights it was fitted for, e.g. models/v6/turns/calibration.json
    """
    return Path(weight_path).with_name("calibration.json")

class MachineLearning:
    """

    """
    model       : keras.Model
    vocab_size  : int
    tokenizer   : Tokenizer
    calibration : dict = {"method": "none", "a": 1.0, "b": 0.0} # score = sigmoid(a * logit + b), a = 1, b = 0 is no change

    training_data : list

    def __init__(self, train: bool, weight_path: str = "routing.weights.h5", token_path: str = "token_map.json", calibration_path: str | None = None) -> None:
        self.tokenizer = Tokenizer(oov_token="<OOV>", num_words=VOCAB_LIMIT)

        if not train:
            with open(token_path, "r") as file:
                self.tokenizer = tokenizer_from_json(file.read())

            self.vocab_size = len(self.tokenizer.word_index)
            self.model      = self.get_model_architecture()
            self.model.load_weights(weight_path)

            calibration_path = Path(calibration_path) if calibration_path else default_calibration_path(weight_path)
            if calibration_path.exists():
                self.calibration = json.loads(calibration_path.read_text())
            else:
                print(f"No calibration at {calibration_path}, outputs are uncalibrated (run `python machine_learning.py calibrate`)")
        else:
            print("=== Loading training data ===")
            time_elapsed = time()

            self.training_data = load_training_data()

            print(f"=== Loaded training data (took {(time_elapsed := (time() - time_elapsed))}s) ===")
            print( "=== Fitting tokenizer ===")
            time_elapsed = time()

            self.tokenizer.fit_on_texts([m.text for log in self.training_data for m in log.messages])
            # the tags only appear once here, so give them a huge count or num_words would cut them off as rare
            for tag in (SELF_TAG, OTHER_TAG, TARGET_TAG):
                self.tokenizer.word_counts[tag] = 10**9
            self.tokenizer.fit_on_texts([SELF_TAG, OTHER_TAG, TARGET_TAG])

            print(f"=== Fitted tokenizer (took {(time_elapsed := (time() - time_elapsed))}s) ===")

            self.vocab_size = len(self.tokenizer.word_index)
            self.model, history = self.train_model()

            # save history to a file with timestamp as name
            with open(filename := f"{int(time())}.json", "w") as file:
                json.dump({k: np.asarray(v).tolist() for k, v in history.items()}, file)

            print(f"=== Saved history to {filename} ===")

            self.save_weights()
            self.save_token_map()
            self.save_calibration()


    def tokenize(self, text: str, length: int = TOKENS_PER_MESSAGE):
        """
        Tokenizes the provided text
        """
        return pad_sequences(
            self.tokenizer.texts_to_sequences([text]),
            maxlen     = length,
            padding    = "post",
            truncating = "post"
        )[0]

    def compute_features(self, window: list[Message], person: str) -> list[float]:
        """
        Computes the features of a window (up to FEATURE_HISTORY messages, the last being the one
        being routed), given a target person. Mostly about who has been talking to whom, and how recently
        """
        message = window[-1]
        speaker = message.username
        history = window[:-1]
        recent  = history[-20:]
        text    = message.text.lower()

        candidates = set(m.username for m in window[-CANDIDATE_POOL:] if m.username is not None) - {speaker}

        def distance(predicate) -> int:
            for d, m in enumerate(reversed(history), 1):
                if predicate(m):
                    return d
            return FEATURE_HISTORY + 1

        def addresses(m: Message, who: str | None) -> bool:
            return who is not None and mentions(m.text, who)[1]

        strict, loose = mentions(message.text, person)

        person_last    = distance(lambda m: m.username == person)
        person_to_spkr = distance(lambda m: m.username == person  and addresses(m, speaker))
        spkr_to_person = distance(lambda m: m.username == speaker and addresses(m, person))

        # who last addressed the speaker, and who the speaker last addressed
        last_to_speaker = next((m.username for m in reversed(history) if m.username not in (None, speaker) and addresses(m, speaker)), None)
        last_from_speaker = None
        for m in reversed(history):
            if m.username == speaker:
                last_from_speaker = next((c for c in sorted(candidates) if addresses(m, c)), None)
                if last_from_speaker is not None:
                    break

        person_messages = [m for m in history if m.username == person]
        person_recent   = person_messages[-3:]
        person_last_txt = person_recent[-1].text if person_recent else ""

        message_words = words(message.text)
        person_words  = set().union(*(words(m.text) for m in person_recent))
        overlap       = len(message_words & person_words)

        busy_elsewhere = sum(
            m.username == person and not addresses(m, speaker) and any(addresses(m, c) for c in candidates - {person})
            for m in recent
        )
        speaker_since = sum(m.username == speaker for m in history[len(history) - person_last + 1:]) if person_last <= len(history) else 0
        interactions  = sum((m.username == person and addresses(m, speaker)) or (m.username == speaker and addresses(m, person)) for m in history)

        # how recently this person spoke compared to the other candidates (0 = most recent)
        last_index = {m.username: i for i, m in enumerate(history) if m.username in candidates}
        recency    = sorted(candidates, key=lambda c: -last_index.get(c, -1))
        rank       = recency.index(person) if person in recency else len(recency)

        # what the speaker has said since this person last spoke, e.g. a burst of lines at them
        since         = history[last_index.get(person, -1) + 1:]
        speaker_burst = sum(m.username == speaker and addresses(m, person) for m in since)
        speaker_prev  = next((m for m in reversed(history) if m.username == speaker), None)
        pending       = speaker_prev is not None and addresses(speaker_prev, person) and speaker_prev in since

        # when others addressed this person, did they speak within the next 5 lines?
        addressed = answered = 0
        for i, m in enumerate(history[:-5]):
            if m.username not in (None, person) and addresses(m, person):
                addressed += 1
                answered  += any(x.username == person for x in history[i + 1:i + 6])

        person_prev    = person_messages[-1] if person_messages else None
        person_elsewhere = person_prev is not None and re.match(r"^\S+[:,]", person_prev.text) is not None and not addresses(person_prev, speaker)

        speaker_messages = [m for m in history if m.username == speaker]
        others_mentioned = sum(mentions(message.text, c)[1] for c in candidates - {person})
        opener           = re.match(r"^\S+[:,]", message.text)

        # does the speaker (or person) tend to split a thought over several lines?
        def run_rate(who: str) -> float:
            followed = [any(x.username == who for x in history[i + 1:i + 3]) for i, m in enumerate(history[:-2]) if m.username == who]
            return float(np.mean(followed)) if followed else 0.5

        speaker_run = 0
        for m in reversed(history):
            if m.username != speaker:
                break
            speaker_run += 1

        stripped = message.text.rstrip()

        return [
            float(strict),                                                         # mentioned in the message
            float(loose),                                                          # mentioned, allowing abbreviations
            min(sum(mentions(message.text, c)[1] for c in candidates - {person}), 3) / 3, # others mentioned
            float(re.match(r"^\S+[:,]", message.text) is not None),                # message is addressed to someone
            log_distance(person_last),                                             # how long ago this person spoke
            log_distance(distance(lambda m: m.username == speaker)),               # how long ago the speaker spoke
            log_distance(person_to_spkr),                                          # how long ago this person addressed the speaker
            log_distance(spkr_to_person),                                          # how long ago the speaker addressed this person
            float(last_to_speaker == person),
            float(last_to_speaker is None),
            float(last_from_speaker == person),
            float(last_from_speaker is None),
            float(bool(history) and history[-1].username == person),               # spoke right before
            float(person in [m.username for m in history[-3:]]),
            sum(m.username == person  for m in recent) / 20,                       # how often has been talking
            len(person_messages) / max(len(history), 1),
            sum(m.username == speaker for m in recent) / 20,
            min(interactions, 10) / 10,                                            # back-and-forth with the speaker
            min(busy_elsewhere, 5) / 5,                                            # talking to other people
            float(np.mean([re.match(r"^\S+[:,]", m.text) is not None for m in person_messages])) if person_messages else 0.0,
            min(speaker_since, 10) / 10,                                           # speaker messages since this person spoke
            len(set(m.username for m in recent if m.username)) / 20,
            float("?" in message.text),
            min(len(message.text.split()), 30) / 30,
            float(person_last == 1),
            float(person_to_spkr <= 3),
            float(spkr_to_person <= 3),
            float(text.startswith("!")),                                           # bot command
            float("http" in text),
            overlap / max(1, len(message_words | person_words)),                   # topic overlap with this person
            min(overlap, 10) / 10,
            float("?" in person_last_txt),
            float(bool(person_last_txt) and mentions(person_last_txt, speaker)[1]),
            float(re.search(r"\b(thanks|thank|thx|ty|cheers)\b", text) is not None),
            float(re.search(r"\b(lol|heh|hehe|haha)\b|:\)|:p|:d", text) is not None),
            float(re.match(r"(hi|hello|hey)\b", text) is not None),
            len(candidates) / CANDIDATE_POOL,
            float(is_bot(speaker)),
            float(is_bot(person)),                                                 # bots don't hold conversations
            min(speaker_burst, 5) / 5,
            float(speaker_burst >= 1 and not loose),                               # this is a follow-up to lines at this person
            float(pending),                                                        # speaker's last line was at this person, still unanswered
            float(ACKNOWLEDGEMENT.match(text) is not None),
            float(stripped.endswith("?")),
            float(QUESTION_WORD.search(text) is not None),
            rank / CANDIDATE_POOL,
            float(rank == 0),
            min(addressed, 10) / 10,
            answered / addressed if addressed else 0.5,                            # how responsive this person is
            float(person_elsewhere),
            float(any("?" in m.text for m in person_recent)),                      # this person has been asking things
            float(np.mean([re.match(r"^\S+[:,]", m.text) is not None for m in speaker_messages])) if speaker_messages else 0.0,
            float(any("?" in m.text for m in speaker_messages[-3:])),
            float(loose and others_mentioned == 0),                                # only this person is mentioned
            float(opener is not None and mentions(opener.group(0), person)[1]),    # message opens with this person's name
            minutes_between(person_prev.minute  if person_prev  else None, message.minute, 120),
            minutes_between(speaker_prev.minute if speaker_prev else None, message.minute, 120),
            minutes_between(history[-1].minute  if history      else None, message.minute, 60),
            min(len(since), 50) / 50,
            min(sum(m.username == speaker for m in since), 10) / 10,
            run_rate(speaker),                                                     # speaker splits lines, so the reply may go to a later one
            min(speaker_run, 5) / 5,
            float(UNFINISHED.search(stripped.lower()) is not None),
            float(re.search(r"[.?!)]$", stripped) is not None),
            min(np.mean([len(m.text.split()) for m in speaker_messages]), 30) / 30 if speaker_messages else 0.0,
            run_rate(person),
        ] + self.compute_turn_features(window, person, candidates)

    def compute_turn_features(self, window: list[Message], person: str, candidates: set[str]) -> list[float]:
        """
        Most false positives are this person answering the speaker, but against a later line of theirs, or
        this person being busy with someone else. These look at how fast the speaker is going and who the
        person is paying attention to
        """
        message = window[-1]
        speaker = message.username
        history = window[:-1]
        others  = candidates - {person}

        def addresses(m: Message, who: str | None) -> bool:
            return who is not None and mentions(m.text, who)[1]

        def minutes_ago(m: Message) -> int:
            return 999 if message.minute is None or m.minute is None else (message.minute - m.minute) % 1440

        def share(lines: list[Message], who: str) -> float:
            return sum(m.username == who for m in lines) / max(len(lines), 1)

        # the speaker's turns (consecutive lines) and how busy they are right now
        turns, run = [], 0
        for m in history:
            if m.username == speaker:
                run += 1
            elif run:
                turns.append(run)
                run = 0

        # who the person has been talking with, other than the speaker
        recent         = history[-20:]
        person_recent  = [m for m in recent if m.username == person]
        partners       = set(c for m in person_recent for c in others if addresses(m, c))
        talking_to     = set(m.username for m in recent if m.username in others and addresses(m, person))
        last_partners  = [c for c in others if person_recent and addresses(person_recent[-1], c)]
        live_elsewhere = bool(last_partners) and any(m.username in last_partners and addresses(m, person) for m in recent)

        person_window = [m for m in history[-40:] if m.username == person]
        to_speaker    = sum(addresses(m, speaker) for m in person_window)
        to_others     = sum(any(addresses(m, c) for c in others) for m in person_window)

        # lines others aimed at the person since the speaker last did
        last_at_person = next((i for i in range(len(history) - 1, -1, -1) if history[i].username == speaker and addresses(history[i], person)), None)
        others_since   = sum(m.username in others and addresses(m, person) for m in history[last_at_person + 1:]) if last_at_person is not None else 0

        speaker_prev = next((m for m in reversed(history) if m.username == speaker), None)
        text         = message.text

        return [
            min(np.mean(turns) if turns else 1.0, 5) / 5,                          # speaker's usual turn length
            share(history[-5:],  speaker),
            share(history[-10:], speaker),
            share(history[-20:], speaker),
            min(sum(m.username == speaker and minutes_ago(m) == 0 for m in history[-10:]), 5) / 5, # lines this same minute
            min(sum(minutes_ago(m) <= 2 for m in history), 30) / 30,               # how busy the channel is
            share(history[-10:], person),
            min(len(partners), 3) / 3,
            min(len(talking_to), 3) / 3,
            float(live_elsewhere),                                                 # person is mid-conversation with someone else
            to_speaker / (to_speaker + to_others) if to_speaker + to_others else 0.5,
            min(to_others, 5) / 5,
            min(len(set(m.username for m in history[-40:] if m.username in others and addresses(m, person))), 4) / 4,
            float(any(m.username in others and addresses(m, speaker) for m in history[-5:])),   # someone else is answering the speaker
            min(len(set(m.username for m in history[-10:] if m.username in others and addresses(m, speaker))), 3) / 3,
            min(len(set(c for m in history[-10:] if m.username == speaker for c in others if addresses(m, c))), 3) / 3,
            float(speaker_prev is not None and any(addresses(speaker_prev, c) for c in others)), # speaker's last line was at someone else
            min(others_since, 3) / 3,
            float(any("?" in m.text for m in history[-3:] if m.username == speaker)),
            float(re.match(r"^\S+[:,]", text) is not None and not mentions(text, person)[1]), # opens with someone else's name
            float(len(text) < 25),
            float(re.search(r"\b(also|btw|and|but|or)\b", text.lower()[:12]) is not None), # continues an earlier line
        ]

    def get_model_architecture(self) -> keras.Model:
        """
        Gives the model architecture: ENSEMBLE_SIZE independently trained members, averaged.
        The members are kept in self.members so they can be trained one at a time
        """
        text_input     = keras.Input(shape=(HISTORY, TOKENS_PER_MESSAGE,), dtype="int32",   name="text")
        features_input = keras.Input(shape=(FEATURE_COUNT,),               dtype="float32", name="features")

        outputs      = [self.member_architecture(text_input, features_input) for _ in range(ENSEMBLE_SIZE)]
        self.members = [keras.Model(inputs=[text_input, features_input], outputs=out) for out in outputs]

        out = outputs[0] if ENSEMBLE_SIZE == 1 else keras.layers.Average()(outputs)
        return keras.Model(inputs=[text_input, features_input], outputs=out)

    def member_architecture(self, text_input, features_input):
        """
        A single model of the ensemble
        """
        text = keras.layers.Embedding             (self.vocab_size + 1, TEXT_VECTOR_SIZE)(      text_input)
        text =              MessagePooling        (                                     )(text, text_input)
        text = keras.layers.Dropout               (0.5                                  )(text            )
        text = keras.layers.LSTM                  (TEXT_VECTOR_SIZE                     )(text            )
        text = keras.layers.Dropout               (0.5                                  )(text            ) # text branch overfits fast otherwise

        features = keras.layers.Dense(64, activation="relu")(features_input)
        features = keras.layers.Dense(32, activation="relu")(features      )

        inner = keras.layers.Concatenate()([text, features])
        inner = keras.layers.Dense(32, activation="relu"   )(inner)
        inner = keras.layers.Dense(16, activation="relu"   )(inner)
        return  keras.layers.Dense(1,  activation="sigmoid")(inner)

    def clean_text(self, message: Message, person: str) -> str:
        """
        Tags a message relative to the target person
        """
        tag  = SELF_TAG if message.username == person else OTHER_TAG
        text = re.sub(name_pattern(person), TARGET_TAG, message.text, flags=re.IGNORECASE)
        return tag + " " + text

    def prepare_point(self, window: list[Message], person: str) -> dict[str, np.ndarray]:
        """
        Encodes a single window for the model. The text uses the last HISTORY messages,
        the features use up to the last FEATURE_HISTORY
        """
        if len(window) < HISTORY:
            raise ValueError(f"Expected at least {HISTORY} messages, got {len(window)}")

        return {
            "text"    : np.stack([self.tokenize(self.clean_text(m, person)) for m in window[-HISTORY:]]).astype("int32"),
            "features": np.array(self.compute_features(window[-FEATURE_HISTORY:], person), dtype="float32")
        }

    def format_data(self, logs: list[MessagedLog]) -> tuple[dict[str, np.ndarray], np.ndarray]:
        """
        Builds the (window, person) points and labels for a set of logs
        """
        x_text     = []
        x_features = []
        t          = []

        for log in logs:
            window = []

            for index, message in enumerate(log.messages):
                window.append(message)

                if len(window) > FEATURE_HISTORY:
                    window.pop(0)

                authors = set(m.username for m in window[-CANDIDATE_POOL:] if m.username is not None)
                authors.discard(message.username)

                # we can only train on annotated messages, and only once their replies could have happened inside the log
                observed = len(log.messages) - 1 - index >= REPLY_HORIZON
                if len(window) >= CANDIDATE_POOL and message.annotated and observed and len(authors) > 0 and message.username != None:
                    for person in authors:
                        point = self.prepare_point(window, person)

                        x_text    .append(point["text"    ])
                        x_features.append(point["features"])
                        t         .append(person in message.responders)

        return (
            {
                "text"    : np.array(x_text,     dtype="int32"  ),
                "features": np.array(x_features, dtype="float32")
            },
            np.array(t, dtype="float32").reshape(-1, 1)
        )

    def train_model(self) -> tuple[keras.Model, dict]:
        """
        Trains the model with the specified hyperparameters
        """
        # === Data formatting ===
        print("=== Formatting data ===")
        time_elapsed = time()

        x_train, t_train = self.format_data(self.training_data)
        x_valid, t_valid = self.format_data(load_messaged_split(VALIDATION_SPLIT))

        print(f"=== Finished formatting data (took {(time_elapsed := (time() - time_elapsed))}s) ===")

        print("Label imbalance: " + str(np.mean(t_train)))

        print(f"=== Training {EPOCHS} epochs ===")

        model   = self.get_model_architecture()
        history = {}

        # each member trains on its own (own initialization, own shuffling, own early stop), then they're averaged
        for i, member in enumerate(self.members):
            print(f"=== Training member {i + 1}/{ENSEMBLE_SIZE} ===")

            member.compile(
                optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
                loss="binary_crossentropy",
                # explicit names, so later members don't become auc_1 and slip past EarlyStopping
                metrics=[keras.metrics.Precision(name="precision"), keras.metrics.F1Score(threshold=ROUTING_CUTOFF, average="micro", name="f1_score"), keras.metrics.AUC(name="auc")]
            )

            member_history = member.fit(
                x_train,
                t_train,

                validation_data = (x_valid, t_valid),
                epochs          = EPOCHS,
                verbose         = VERBOSE,
                batch_size      = BATCH_SIZE,
                callbacks       = [
                    TqdmCallback(verbose=2),
                    # AUC doesn't depend on the cutoff, so it picks the best ranking model, not a lucky threshold
                    keras.callbacks.EarlyStopping(monitor="val_auc", mode="max", patience=5, restore_best_weights=True)
                ],
                class_weight = {0: 1.0, 1: POSITIVE_WEIGHT}
            )

            for key, values in member_history.history.items():
                history[key if ENSEMBLE_SIZE == 1 else f"{key}_{i}"] = values

        self.model = model
        self.calibrate(x_valid, t_valid)

        return (model, history)

    def calibrate(self, x: dict[str, np.ndarray], t: np.ndarray, method: str = CALIBRATION) -> dict:
        """
        Fits the calibration on (x, t), normally the dev split. The weights don't change
        """
        raw  = self.model.predict(x, verbose=0).reshape(-1)
        a, b = fit_calibration(raw, t, method)

        self.calibration = {"method": method, "a": a, "b": b}

        print(f"=== {method} calibration: a {a:.4f} (T {1 / a:.4f}), b {b:.4f} (NLL {nll(raw, t):.4f} -> {nll(apply_calibration(raw, a, b), t):.4f}) ===")

        return self.calibration

    def predict(self, x: dict[str, np.ndarray]) -> np.ndarray:
        """
        Calibrated scores for a batch of points, shape (N,)
        """
        return apply_calibration(self.model.predict(x, verbose=0).reshape(-1), self.calibration["a"], self.calibration["b"])

    def save_weights(self, filename: str = "routing.weights.h5") -> None:
        self.model.save_weights(filename)

    def save_token_map(self, filename: str = "token_map.json") -> None:
        with open(filename, "w") as file:
            file.write(self.tokenizer.to_json())

    def save_calibration(self, filename: str = "calibration.json") -> None:
        with open(filename, "w") as file:
            json.dump(self.calibration, file)

    def inference(self, messages: list[Message], person: str) -> float:
        """
        Performs inference on the model, returning the calibrated score
        """
        point = self.prepare_point(messages[-FEATURE_HISTORY:], person)
        return float(self.predict({k: v[None] for k, v in point.items()})[0])


if __name__ == "__main__":
    from sys import argv

    mode = argv[1] if len(argv) > 1 else "train"

    model = None

    if mode == "train":
        model = MachineLearning(True)
    elif mode == "calibrate":
        # fits the calibration of an already-trained model on dev, and saves it next to its weights
        # usage: python machine_learning.py calibrate [weight_path] [token_path] [platt|temperature]
        weight_path = argv[2] if len(argv) > 2 else "routing.weights.h5"
        token_path  = argv[3] if len(argv) > 3 else "token_map.json"
        method      = argv[4] if len(argv) > 4 else CALIBRATION

        model = MachineLearning(False, weight_path, token_path)

        model.calibrate(*model.format_data(load_messaged_split(VALIDATION_SPLIT)), method)
        model.save_calibration(str(default_calibration_path(weight_path)))

        print(f"=== Saved calibration to {default_calibration_path(weight_path)} ===")
        raise SystemExit
    elif mode == "test":
        model = MachineLearning(False, "routing.weights.h5", "token_map.json")

        data = """[09:14] <crimsun> kleedrac: I'm afraid not. Any version of mplayer except for -k7* should work for your cpu
    [09:14] <intinig> does a subversion gnome client exist?
    [09:14] <kleedrac> crimsun: Hmmm ... I wonder why it does that?
    [09:15] <will> best media player: VLC /get wxvlc) It plays everything!
    [09:15] <|QuaD-> will: totem has caused me no troubles
    [09:15] <|QuaD-> totem xine
    [09:15] <crimsun> kleedrac: good question.
    [09:15] <Tsjoklate> |QuaD same here.. totem-xine werkt geweldig
    [09:15] <crimsun> intinig: I don't see one in warty or hoary
    [09:15] <kleedrac> xine is working great for video and xmms works great for mp3's but some of my music is in .wma as my wife ripped it"""

        print(model.inference(lines_to_messages(data.split()), "crimsun"))
    else:
        model = MachineLearning(False, input("Weight path: "), input("Token map path: "))

    while (command := input("> ")) != "exit":
        if command == "save_weights":
            model.save_weights(input("Filename: "))

        elif command == "save_token_map":
            model.save_token_map(input("Filename: "))

        elif command == "inference":
            person = input("Person: ")
            window = []

            print(f"Now, input up to the last {CANDIDATE_POOL} messages (older first), and a blank line to finish")

            while len(window) < CANDIDATE_POOL:
                if not (line := input("Message: ")):
                    if len(window) >= HISTORY:
                        break

                    print(f"Need at least {HISTORY} messages")
                    continue

                window.append(line)

            print(model.inference(lines_to_messages(window), person))
