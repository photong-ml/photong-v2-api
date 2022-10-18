"""
This module contains the classes for the different types of objects in the Photong package.
"""

from enum import Enum
from typing import Any, Dict, List

import note_seq
from typing_extensions import TypeAlias

NoteSequenceNote: TypeAlias = note_seq.protobuf.music_pb2.NoteSequence.Note


class TonalityType(Enum):
    """
    Types of tonality.

    Attributes:
        MAJOR: Major tonality.
        MINOR: Minor tonality.
    """

    MAJOR = 0
    MINOR = 1


class ChordType(Enum):
    """
    Types of chord.

    Attributes:
        MAJOR: Major chord.
        MINOR: Minor chord.
        DIMINISHED: Diminished chord.
    """

    MAJOR = 0
    MINOR = 1
    DIMINISHED = 2


class NoteType(Enum):
    """
    Notes in a scale.

    Attributes:
        C: C note.
        C_SHARP: C# note.
        D_FLAT: Db note. (Equivalent to C#)
        D: D note.
        D_SHARP: D# note.
        E_FLAT: Eb note. (Equivalent to D#)
        E: E note.
        F: F note.
        F_SHARP: F# note.
        G_FLAT: Gb note. (Equivalent to F#)
        G: G note.
        G_SHARP: G# note.
        A_FLAT: Ab note. (Equivalent to G#)
        A: A note.
        A_SHARP: A# note.
        B_FLAT: Bb note. (Equivalent to A#)
        B: B note.
    """

    C = 0
    C_SHARP = 1
    D_FLAT = 1
    D = 2
    D_SHARP = 3
    E_FLAT = 3
    E = 4
    F = 5
    F_SHARP = 6
    G_FLAT = 6
    G = 7
    G_SHARP = 8
    A_FLAT = 8
    A = 9
    A_SHARP = 10
    B_FLAT = 10
    B = 11


class TonalityConfig:
    """
    Configuration for a tonality.

    Attributes:
        diatonic (List[int]): The diatonic scale.
        default (ChordType): The default chord type.
        chords (Dict[ChordType, List[int]]): The chord types.
    """

    diatonic: List[int]
    chords: Dict[ChordType, List[int]]

    def __init__(self, **kwargs: Any) -> None:
        """Initialize the class."""
        for key, value in kwargs.items():
            setattr(self, key, value)
