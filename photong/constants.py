"""
This module contains constants used in the Photong package.
"""

from photong.types import ChordType, TonalityConfig, TonalityType

CHORD_OFFSETS = {
    ChordType.MAJOR: [0, 4, 7, 12],
    ChordType.MINOR: [0, 3, 7, 12],
    ChordType.DIMINISHED: [0, 3, 6, 12],
}

TONALITY_CONFIG = {
    TonalityType.MAJOR: TonalityConfig(
        # This is the same as note_seq.MAJOR_SCALE, but there isn't a MINOR_SCALE equivalent
        diatonic=[0, 2, 4, 5, 7, 9, 11],
        chords={
            ChordType.MAJOR: [0, 5, 7],
            ChordType.MINOR: [2, 4, 9],
            ChordType.DIMINISHED: [11],
        },
    ),
    TonalityType.MINOR: TonalityConfig(
        diatonic=[0, 2, 3, 5, 7, 8, 10],
        chords={
            ChordType.MAJOR: [3, 8, 10],
            ChordType.MINOR: [0, 5, 7],
            ChordType.DIMINISHED: [2],
        },
    ),
}
