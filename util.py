"""
Utility function for the main server.
"""

from io import BytesIO

import note_seq
import numpy as np
from scipy.io import wavfile

from photong.pipeline import PhotongPipeline

model = PhotongPipeline()


def predict_and_convert(img_data: str) -> bytes:
    """Predict and convert to wav."""
    note_sequence = model.predict(img_data=img_data)
    audio_data = note_seq.fluidsynth(note_sequence, sample_rate=44100.0)
    # Normalize for 16 bit audio
    audio_data = np.int16(audio_data / np.max(np.abs(audio_data)) * 32767 * 0.9)
    virtualfile = BytesIO()
    wavfile.write(virtualfile, 44100, audio_data)
    return virtualfile.getvalue()
