import json
import random

import keras
import numpy as np
from keras.src.legacy.preprocessing.text import Tokenizer, tokenizer_from_json
from keras.utils import pad_sequences
from util import load_training_data, Message, lines_to_messages

import re
from time import time

from tqdm.keras import TqdmCallback

SEED               = 1432516 # Zohar's favorite number
EPOCHS             = 5
VERBOSE            = 0
HISTORY            = 5
TOKENS_PER_MESSAGE = 32
TEXT_VECTOR_SIZE   = 64
ROUTING_CUTOFF     = 0.5
BATCH_SIZE         = 32

SELF_TAG   = "xxself"
OTHER_TAG  = "xxother"
TARGET_TAG = "xxtarget"

random     .seed           (SEED)
keras.utils.set_random_seed(SEED)

# Used for seeing how long different things take
time_elapsed = 0

"""
Things to do:
    - Compare models trained with different hyperparameters with validation set
    - Evaluate final model with test set
    - Add graph for training curve
"""

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
        Computes the features of a window, given a target person
        """
        authors   = [message.username for message in window]
        mentioned = [re.search(rf"\b{re.escape(person.lower())}\b", m.text.lower()) is not None for m in window]

        last_spoke = len(window)

        for i, author in enumerate(reversed(authors)):
            if author == person:
                last_spoke = i
                break

        return [
            float(mentioned[-1]),                # mentioned in last message
            sum(mentioned)        / len(window), # how often mentioned,
            authors.count(person) / len(window), # how often has been talking
            last_spoke            / len(window)  # how long ago this person spoke
        ]

    def get_model_architecture(self) -> keras.Model:
        """
        Gives the model architecture
        """
        text_input           = keras.Input(shape=(HISTORY, TOKENS_PER_MESSAGE,), dtype="int32",   name="text")
        features_input       = keras.Input(shape=(4,),                           dtype="float32", name="features")

        text = keras.layers.Embedding             (self.vocab_size + 1, TEXT_VECTOR_SIZE)(      text_input)
        text =              MessagePooling        (                                     )(text, text_input)
        text = keras.layers.GlobalAveragePooling1D(                                     )(text            )

        features = keras.layers.Dense(16, activation="relu")(features_input)

        inner = keras.layers.Concatenate()([text, features])
        inner = keras.layers.Dense(64, activation="relu"   )(inner)
        out   = keras.layers.Dense(1,  activation="sigmoid")(inner)

        return keras.Model(inputs=[text_input, features_input], outputs=out)

    def clean_text(self, message: Message, person: str) -> str:
        """
        Tags a message relative to the target person
        """
        tag  = SELF_TAG if message.username == person else OTHER_TAG
        text = re.sub(rf"\b{re.escape(person)}\b", TARGET_TAG, message.text, flags=re.IGNORECASE)
        return tag + " " + text

    def prepare_point(self, window: list[Message], person: str) -> dict[str, np.ndarray]:
        """
        Encodes a single window for the model
        """
        if len(window) != HISTORY:
            raise ValueError(f"Expected {HISTORY} messages, got {len(window)}")

        return {
            "text"    : np.stack([self.tokenize(self.clean_text(m, person)) for m in window]).astype("int32"),
            "features": np.array(self.compute_features(window, person), dtype="float32")
        }

    def train_model(self) -> tuple[keras.Model, dict]:
        """
        Trains the model with the specified hyperparameters
        """
        # === Data formatting ===
        print("=== Formatting data ===")
        time_elapsed = time()

        x_text     = []
        x_features = []
        t_train    = []

        for log in self.training_data:
            window = []

            for i, message in enumerate(log.messages):
                window.append(message)

                if len(window) > HISTORY:
                    window.pop(0)

                authors = set(m.username for m in window if m.username is not None)
                authors.discard(message.username)

                # we can only train on annotated messages
                if len(window) == HISTORY and message.annotated and len(authors) > 0 and message.username != None:
                    for person in authors:
                        point = self.prepare_point(window, person)

                        x_text    .append(point["text"    ])
                        x_features.append(point["features"])
                        t_train   .append(person in message.responders)

        print(f"=== Finished formatting data (took {(time_elapsed := (time() - time_elapsed))}s) ===")

        print("Label imbalance: " + str(np.mean(t_train)))

        print(f"=== Training {EPOCHS} epochs ===")

        model = self.get_model_architecture()

        # Training
        model.compile(
            optimizer="adam",
            loss="binary_crossentropy",
            metrics=[keras.metrics.F1Score(threshold=ROUTING_CUTOFF), keras.metrics.AUC()]
        )

        history = model.fit(
            {
                "text"    : np.array(x_text,     dtype="int32"  ),
                "features": np.array(x_features, dtype="float32")
            },
            np.array(t_train, dtype="float32").reshape(-1,  1),

            epochs     = EPOCHS,
            verbose    = VERBOSE,
            batch_size = BATCH_SIZE
            callbacks  = [TqdmCallback(verbose=2)]
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
        point = self.prepare_point(messages[-HISTORY:], person)
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

            print("Now, input the last " + str(HISTORY) + " messages (older first)")

            while len(window) < HISTORY:
                window.append(input("Message: "))

            print(model.inference(lines_to_messages(window), person))
