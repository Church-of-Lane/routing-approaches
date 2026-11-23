from spellchecker import SpellChecker
import re

"""
Assumptions:
1. Chat lines look like "[HH:MM] <username> message content"
2. Usernames are not empty strings
3. Usernames contain no whitespace or angle brackets
4. Lines that don't match (e.g. "=== x has joined #ubuntu") are system lines:
   they have no speaker and no content
"""

message_speaker_expression = re.compile(r"^\[\d{2}:\d{2}\] <([^<>\s]+)>")

message_content_expression = re.compile(r"^\[\d{2}:\d{2}\] <[^<>\s]+> ?(.*)$")


class Symbolic:
    def __init__(self, name: str):
        self.name = name
        self.spellchecker = SpellChecker()

        self.spellchecker.word_frequency.load_words([name])

    def score_text(self, messages: list[str]) -> float:
        responding_to = messages[-1]

        speaker = Symbolic.extract_username(responding_to)

        if speaker is None:
            raise ValueError("No speaker")

        if speaker == self.name:
            return 0

        direct_score = self.message_contains_name(Symbolic.extract_message(responding_to))
        if direct_score > 0:
            return direct_score


        score = 0
        speaker_message_count = 0

        for message in messages:
            author = Symbolic.extract_username(message)

            if author == speaker:
                score += self.message_contains_name(Symbolic.extract_message(message))
                speaker_message_count += 1
            elif author != self.name:
                score /= 1.5

        return score / max(speaker_message_count, 1)

    @staticmethod
    def extract_username(message: str) -> str | None:
        match = re.search(message_speaker_expression, message)

        return match.group(1) if match is not None else None

    @staticmethod
    def format_username(username: str) -> str:
        return f"<{username}>"

    @staticmethod
    def extract_message(full_message: str) -> str | None:
        match = re.search(message_content_expression, full_message)

        return match.group(1) if match is not None else None

    def message_contains_name(self, message: str | None, useSpellcheck: bool = False) -> float:
        # if no message given, definetly no mention
        if not message:
            return 0

        # if the message contains name as perfect match, definetly mentions
        confidence = 1 if self.name in message else 0

        if confidence == 1:
            return confidence

        # if message contains name as almost perfect match (within case), probably mentions
        confidence = 0.9 if self.name.lower() in message.lower() else 0

        if confidence != 0:
            return confidence

        # spellcheck can be very laggy (relative to other stuff), so sometimes beneficial to not use
        if not useSpellcheck:
            return 0

        # otherwise, see if name maybe mentions (within some typos)
        words = message.split()

        misspelt = self.spellchecker.unknown(words)

        for misspelled in misspelt:
            if self.spellchecker.correction(misspelled) == self.name:
                confidence = 0.5
                break

        return confidence

    def get_name(self) -> str:
        return self.name

    def __str__(self) -> str:
        return self.name

    def __repr__(self) -> str:
        return f"<symbolic.Symbolic object {self} at {id(self)}>"


if __name__ == "__main__":
    # imported here, since util imports this file
    from util import TEST_SAMPLE, TEST_SAMPLE_CANDIDATE, load_test_sample

    # the test set sample, one chat line per line
    messages = load_test_sample(TEST_SAMPLE)

    symbolic = Symbolic(TEST_SAMPLE_CANDIDATE)

    # Claude Opus 5.5 found bug where wrong variable was passed into score_text
    print(symbolic.score_text(messages))
