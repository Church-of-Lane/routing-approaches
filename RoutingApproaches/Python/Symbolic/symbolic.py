from spellchecker import SpellChecker
import re

"""
Assumptions:
1. Usernames are given at start of string within angular brackets (<>)
2. Usernames are not empty strings
3. Usernames are alphanumeric and _
"""

message_speaker_expression = re.compile("<([\.\/\^\[\]\{\}\-\\|\w]+)>")
message_content_expression = re.compile("")

class Symbolic:
    def __init__(self, name: str):
        # does not support empty usernames
        self.name = name
        self.spellchecker = SpellChecker()

        self.spellchecker.word_frequency.load_words([name])

    def score_text(self, messages: list[str]) -> float:
        responding_to = messages[-1]

        speaker = Symbolic.format_username(Symbolic.extract_username(responding_to))

        if speaker is None:
            raise ValueError("No speaker")
        
        indexed = list.copy(messages)

        messages_from_speaker = list(filter(lambda message: speaker in message, messages))

        referencing_name = list(map(self.message_contains_name, messages_from_speaker))

        username = f"<{self.name}>"

        messages_from_others = list(filter(lambda message: speaker not in message and username not in message, messages))

        score = 0

        for subscore in zip(referencing_name, messages_from_speaker):
            message = indexed.pop(0)

            while message != subscore[1]:
                if message in messages_from_others:
                    score /= 1.5
                
                message = indexed.pop(0)

            score += subscore[0]
        
        return score / max(len(referencing_name), 1)
    
    def extract_username(message: str) -> str | None:
        match = re.search(message_speaker_expression, message)

        return match.group(1) if match is not None else None
    
    def format_username(username: str) -> str:
        return f"<{username}>"
    
    def extract_message(full_message: str) -> str:
        return 

    def message_contains_name(self, message: str) -> float:
        confidence = 1 if self.name in message else 0

        if confidence == 0:
            confidence = 0.9 if self.name.lower() in message.lower() else 0
            # add spellcheck, add capitalization blindness BUT check for if name
            words = message.split()

            misspelt = []#self.spellchecker.unknown(words)

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
    
    messages = data.split("\n")

    symbolic = Symbolic("crimsun")

    # Claude Opus 5.5 found bug where wrong variable was passed into score_text
    print(symbolic.score_text(messages))