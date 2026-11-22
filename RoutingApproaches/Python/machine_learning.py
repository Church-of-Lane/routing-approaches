import json
import random

import keras
import numpy as np
from keras.src.legacy.preprocessing.text import Tokenizer, tokenizer_from_json
from keras.utils import pad_sequences
from util import load_training_data, load_messaged_split, Message, MessagedLog, lines_to_messages

import re
from functools import lru_cache
from time import time

from tqdm.keras import TqdmCallback

SEED               = 1432516 # Zohar's favorite number
EPOCHS             = 100
VERBOSE            = 0
HISTORY            = 20  # messages the text model reads
CANDIDATE_POOL     = 20  # messages candidates are drawn from
FEATURE_HISTORY    = 100 # messages the hand-made features look back over
FEATURE_COUNT      = 37
TOKENS_PER_MESSAGE = 32
TEXT_VECTOR_SIZE   = 8
ROUTING_CUTOFF     = 0.5
POSITIVE_WEIGHT    = 3.0 # moves the F1-optimal threshold of the unweighted model (~0.25) to ROUTING_CUTOFF
BATCH_SIZE         = 512
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

class MessagePooling(keras.layers.Layer):
    """
    Averages each message's token embeddings, ignoring padding
    """
    def call(self, embedded, tokens):
        mask   = keras.ops.expand_dims(keras.ops.cast(keras.ops.not_equal(tokens, 0), embedded.dtype), -1)
        summed = keras.ops.sum(embedded * mask, axis=2)
        count  = keras.ops.maximum(keras.ops.sum(mask, axis=2), 1.0)  # avoids 0/0 on empty messages
        return summed / count

class MachineLearning:
    """

    """
    model      : keras.Model
    vocab_size : int
    tokenizer  : Tokenizer

    training_data : list

    def __init__(self, train: bool, weight_path: str = "routing.weights.h5", token_path: str = "token_map.json") -> None:
        self.tokenizer = Tokenizer(oov_token="<OOV>")

        if not train:
            with open(token_path, "r") as file:
                self.tokenizer = tokenizer_from_json(file.read())

            self.vocab_size = len(self.tokenizer.word_index)
            self.model      = self.get_model_architecture()
            self.model.load_weights(weight_path)
        else:
            print("=== Loading training data ===")
            time_elapsed = time()

            self.training_data = load_training_data()

            print(f"=== Loaded training data (took {(time_elapsed := (time() - time_elapsed))}s) ===")
            print( "=== Fitting tokenizer ===")
            time_elapsed = time()

            self.tokenizer.fit_on_texts([m.text for log in self.training_data for m in log.messages])
            self.tokenizer.fit_on_texts([SELF_TAG, OTHER_TAG, TARGET_TAG])

            print(f"=== Fitted tokenizer (took {(time_elapsed := (time() - time_elapsed))}s) ===")

            self.vocab_size = len(self.tokenizer.word_index)
            self.model, history = self.train_model()

            # save history to a file with timestamp as name
            with open(filename := f"{int(time())}.json", "w") as file:
                json.dump({k: np.asarray(v).tolist() for k, v in history.history.items()}, file)

            print(f"=== Saved history to {filename} ===")

            self.save_weights()
            self.save_token_map()


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
        ]

    def get_model_architecture(self) -> keras.Model:
        """
        Gives the model architecture
        """
        text_input           = keras.Input(shape=(HISTORY, TOKENS_PER_MESSAGE,), dtype="int32",   name="text")
        features_input       = keras.Input(shape=(FEATURE_COUNT,),               dtype="float32", name="features")

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
        out   = keras.layers.Dense(1,  activation="sigmoid")(inner)

        return keras.Model(inputs=[text_input, features_input], outputs=out)

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

            for message in log.messages:
                window.append(message)

                if len(window) > FEATURE_HISTORY:
                    window.pop(0)

                authors = set(m.username for m in window[-CANDIDATE_POOL:] if m.username is not None)
                authors.discard(message.username)

                # we can only train on annotated messages
                if len(window) >= CANDIDATE_POOL and message.annotated and len(authors) > 0 and message.username != None:
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

        model = self.get_model_architecture()

        # Training
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
            loss="binary_crossentropy",
            metrics=[keras.metrics.Precision(), keras.metrics.F1Score(threshold=ROUTING_CUTOFF, average="micro"), keras.metrics.AUC()]
        )

        history = model.fit(
            x_train,
            t_train,

            validation_data = (x_valid, t_valid),
            epochs          = EPOCHS,
            verbose         = VERBOSE,
            batch_size      = BATCH_SIZE,
            callbacks       = [
                TqdmCallback(verbose=2),
                keras.callbacks.EarlyStopping(monitor="val_precision", mode="min", patience=5, restore_best_weights=True)
            ],
            class_weight = {0: 1.0, 1: POSITIVE_WEIGHT}
        )

        return (model, history)

    def save_weights(self, filename: str = "routing.weights.h5") -> None:
        self.model.save_weights(filename)

    def save_token_map(self, filename: str = "token_map.json") -> None:
        with open(filename, "w") as file:
            file.write(self.tokenizer.to_json())

    def inference(self, messages: list[Message], person: str) -> float:
        """
        Performs inference on the model
        """
        point = self.prepare_point(messages[-FEATURE_HISTORY:], person)
        return float(self.model.predict({k: v[None] for k, v in point.items()}, verbose=0)[0, 0])


if __name__ == "__main__":
    from sys import argv

    mode = argv[1] if len(argv) > 1 else "train"

    model = None

    if mode == "train":
        model = MachineLearning(True)
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
