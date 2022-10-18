"""
This module contains utility functions for generated note sequences.
"""

from copy import deepcopy
from math import floor
from typing import Optional

import note_seq
import numpy as np
from photong.constants import CHORD_OFFSETS, TONALITY_CONFIG
from photong.types import ChordType, NoteSequenceNote, TonalityConfig, TonalityType
from photong.utils.note_util import NOTES_PER_OCTAVE, is_diatonic, number_to_note


def make_diatonic(
    sequence: note_seq.NoteSequence,
    config: TonalityConfig,
    tonic: Optional[NoteSequenceNote] = None,
) -> note_seq.NoteSequence:
    """
    Make a NoteSequence diatonic by randomly shifting notes up or down a tone.

    The tonic is the reference note for the diatonic transformation.
    If the tonic is not specified, the first note of the sequence is used.

    Args:
        sequence (note_seq.NoteSequence): The sequence to be transformed.
        config (TonalityConfig): The tonality configuration.
        tonic (Optional[NoteSequenceNote]): The tonic.

    Returns:
        note_seq.NoteSequence: The transformed sequence.
    """
    if tonic is None:
        tonic = sequence.notes[0]

    for note in sequence.notes:
        if not is_diatonic(note, config, tonic):
            # Make diatonic by shifting up or down a tone
            note.pitch = note.pitch + np.random.default_rng().choice([-1, 1])

    return sequence


def add_chords(
    sequence: note_seq.NoteSequence,
    config: TonalityConfig,
    tonic: Optional[NoteSequenceNote] = None,
) -> note_seq.NoteSequence:
    """
    Add chords to accompany a NoteSequence, maximum one per 4/4 bar.

    The tonic is used to determine the tonality.
    If the tonic is not specified, the first note of the sequence is used.

    Args:
        sequence (note_seq.NoteSequence): The sequence to be transformed.
        config (TonalityConfig): The tonality configuration.
        tonic (Optional[NoteSequenceNote]): The tonic.

    Returns:
        note_seq.NoteSequence: The transformed sequence.
    """
    if tonic is None:
        tonic = sequence.notes[0]

    # Find the offset for the key of the sequence
    key_offset = number_to_note(tonic.pitch)[0].value

    # Threshold for chord detection
    chord_time_threshold = 0
    # Processed number of notes
    index = 0

    new_sequence = deepcopy(sequence)
    while index < len(sequence.notes):
        note = sequence.notes[index]
        if note.start_time < chord_time_threshold:
            # Skip notes that does not exceed the chord time threshold
            index += 1
            continue

        # Find the number of the note relative to the reference note
        number = (note.pitch - key_offset) % NOTES_PER_OCTAVE

        # Find the type of chord
        chord_type = None
        for chord in ChordType:
            if number in config.chords[chord]:
                chord_type = chord
                break
        if chord_type is None:
            raise ValueError("Chord not found")

        # Extend the ending time to the last of all notes of the same pitch
        chord_end_time = note.end_time
        while (
            index + 1 < len(sequence.notes)
            and sequence.notes[index + 1].pitch == note.pitch
        ):
            chord_end_time = sequence.notes[index + 1].end_time
            index += 1
        chord_end_time = 2 * floor((chord_end_time / 2) + 1)

        # Add the chord
        for chord_offset in CHORD_OFFSETS[chord_type]:
            new_sequence.notes.add(
                # Move down two octaves
                pitch=note.pitch - 2 * NOTES_PER_OCTAVE + chord_offset,
                start_time=note.start_time,
                end_time=chord_end_time,
                velocity=note.velocity,
            )

        # Update the chord time threshold
        chord_time_threshold = chord_end_time
        index += 1

    return new_sequence


def change_tempo(
    note_sequence: note_seq.NoteSequence, tempo: float
) -> note_seq.NoteSequence:
    """
    Change the tempo of a NoteSequence.

    Args:
        note_sequence (note_seq.NoteSequence): The sequence to be transformed.
        tempo (float): The new tempo.

    Returns:
        note_seq.NoteSequence: The transformed sequence.
    """
    new_sequence = deepcopy(note_sequence)

    # Calculate the multiplier
    ratio = note_sequence.tempos[0].qpm / tempo

    # Scale the start and end times
    for note in new_sequence.notes:
        note.start_time *= ratio
        note.end_time *= ratio

    new_sequence.tempos[0].qpm = tempo
    return new_sequence


def touch_up(
    sequence: note_seq.NoteSequence, tonality: TonalityType, tempo: float
) -> note_seq.NoteSequence:
    """
    Touch up a NoteSequence.

    Make the sequence diatonic, append an extra tonic note at the end,
    add chords, and change the tempo.

    Args:
        sequence (note_seq.NoteSequence): The sequence to be transformed.
        tonality (TonalityType): The tonality.
        tempo (float): The tempo.

    Returns:
        note_seq.NoteSequence: The transformed sequence.
    """
    config = TONALITY_CONFIG[tonality]

    # Make diatonic
    sequence = make_diatonic(sequence, config)

    # Append ending note on tonic
    sequence.notes.add(
        pitch=sequence.notes[0].pitch,
        start_time=2 * round(sequence.notes[-1].end_time / 2),
        end_time=sequence.notes[-1].start_time + 2,
        velocity=sequence.notes[-1].velocity,
    )

    # Add chords
    sequence = add_chords(sequence, config)

    # Change tempo
    sequence = change_tempo(sequence, tempo)

    return sequence
