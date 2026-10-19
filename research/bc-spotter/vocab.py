"""Closed vocabulary for the overlap word classifier.

The classifier names the first word the user says over the agent (or `none` for no speech)
and whether more speech follows it inside the window. The turn stage maps that to
backchannel / aside / floor-take; this file only names the words.
"""
import re

# Class -> spellings (Deepgram's output and TTS prompts). Order fixes the class index.
WORDS = {
    'yeah': ['yeah', 'yes', 'yep', 'ya', 'yup'],
    'right': ['right'],
    'okay': ['okay', 'ok', 'alright', 'all right'],
    'mhm': ['mhm', 'mhmm', 'mm-hmm', 'uh-huh', 'mm'],
    'sure': ['sure'],
    'cool': ['cool', 'nice', 'great'],
    'so': ['so'],
    'wait': ['wait'],
    'but': ['but'],
    'actually': ['actually'],
    'no': ['no', 'nope'],
    'hold': ['hold'],  # "hold on"
    'sorry': ['sorry'],
    'well': ['well'],
    'um': ['um', 'uh', 'hmm', 'er'],
    'and': ['and'],
}
CLASSES = list(WORDS) + ['other', 'none']
IDX = {c: i for i, c in enumerate(CLASSES)}
_LOOKUP = {s: c for c, ss in WORDS.items() for s in ss}

# What each class usually means when it is the whole utterance (a hint for the turn stage).
BACKCHANNEL = {'yeah', 'right', 'okay', 'mhm', 'sure', 'cool'}
FLOOR_OPENER = {'so', 'wait', 'but', 'actually', 'no', 'hold', 'sorry', 'well'}


def norm(word: str) -> str:
    return re.sub(r"[^a-z' -]", '', word.lower()).strip()


def word_class(word: str) -> str:
    return _LOOKUP.get(norm(word), 'other')
