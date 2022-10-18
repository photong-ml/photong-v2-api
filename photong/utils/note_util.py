"""
This module contains utility functions for notes.

Modified from https://gist.github.com/devxpy/063968e0a2ef9b6db0bd6af8079dad2a
"""

from typing import Tuple, Union

import note_seq
from photong.constants import TONALITY_CONFIG
from photong.types import NoteSequenceNote, NoteType, TonalityConfig, TonalityType

OCTAVES = list(range(11))
NOTES_PER_OCTAVE = note_seq.NOTES_PER_OCTAVE
MIN_MIDI_PITCH = note_seq.MIN_MIDI_PITCH
MAX_MIDI_PITCH = note_seq.MAX_MIDI_PITCH


def number_to_note(pitch: int) -> Tuple[NoteType, int]:
    """
    Convert a numerical pitch to a note.

    Args:
        pitch (int): The pitch to convert.

    Returns:
        Tuple[NoteType, int]: The note and octave of the note.

    Raises:
        ValueError: If the pitch is not between 0 and 127.
        ValueError: If the octave is not between 0 and 10.
    """

    if pitch < MIN_MIDI_PITCH or pitch > MAX_MIDI_PITCH:
        raise ValueError(
            f"Invalid note number: {pitch}, "
            f"must be between {MIN_MIDI_PITCH} and {MAX_MIDI_PITCH}"
        )

    octave = pitch // NOTES_PER_OCTAVE
    if octave not in OCTAVES:
        raise ValueError(
            f"Invalid octave: {octave}, "
            f"must be between {OCTAVES[0]} and {OCTAVES[-1]}"
        )

    note = NoteType(pitch % NOTES_PER_OCTAVE)
    return note, octave


def note_to_number(note: NoteType, octave: int) -> int:
    """
    Convert a note to a numerical pitch.

    Args:
        note (NoteType): The note to convert.
        octave (int): The octave of the note.

    Returns:
        int: The pitch of the note.

    Raises:
        ValueError: If the note is not valid.
        ValueError: If the octave is not between 0 and 10.
    """

    if note not in NoteType:
        raise ValueError(f"Invalid note: {note}")
    if octave not in OCTAVES:
        raise ValueError(f"octave {octave} not in {OCTAVES}")

    note_val = note.value
    note_val += octave * NOTES_PER_OCTAVE

    if note_val < MIN_MIDI_PITCH or note_val > MAX_MIDI_PITCH:
        raise ValueError(
            f"Invalid note number: {note_val}, "
            f"must be between {MIN_MIDI_PITCH} and {MAX_MIDI_PITCH}"
        )

    return note_val


def is_diatonic(
    note: Union[NoteType, NoteSequenceNote],
    tonality: Union[TonalityType, TonalityConfig],
    tonic: Union[NoteType, NoteSequenceNote],
) -> bool:
    """
    Check if a note is diatonic in a tonality.

    Args:
        note (Union[NoteType, NoteSequenceNote]): The note to check.
        tonality (Union[TonalityType, TonalityConfig]): The tonality to check.
        tonic (Union[NoteType, NoteSequenceNote]): The tonic of the tonality.

    Returns:
        bool: True if the note is diatonic in the tonality, False otherwise.
    """

    if isinstance(note, note_seq.protobuf.music_pb2.NoteSequence.Note):
        pitch = note.pitch
    elif isinstance(note, NoteType):
        pitch = note.value
    else:
        raise TypeError(f"Invalid type of note: {note}")

    if isinstance(tonic, note_seq.protobuf.music_pb2.NoteSequence.Note):
        ref = tonic.pitch
    elif isinstance(tonic, NoteType):
        ref = tonic.value
    else:
        raise TypeError(f"Invalid type of tonic: {tonic}")

    if isinstance(tonality, TonalityType):
        tonality = TONALITY_CONFIG[tonality]
    elif not isinstance(tonality, TonalityConfig):
        raise TypeError(f"Invalid type of tonality: {tonality}")

    # Find the pitch of the note relative to the reference note
    rel_pitch = (pitch - ref) % NOTES_PER_OCTAVE
    return rel_pitch in tonality.diatonic
