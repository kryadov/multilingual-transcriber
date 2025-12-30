import ffmpeg
import os
import uuid
from typing import Tuple

def extract_audio(input_path: str, output_path: str) -> str:
    """
    Extracts audio from a video file and converts it to WAV (16kHz, mono).
    This format is ideal for both Whisper and Pyannote.
    """
    try:
        (
            ffmpeg
            .input(input_path)
            .output(output_path, acodec='pcm_s16le', ac=1, ar='16k')
            .overwrite_output()
            .run(quiet=True)
        )
        return output_path
    except ffmpeg.Error as e:
        raise RuntimeError(f"Failed to extract audio: {e.stderr.decode()}")

def generate_task_id() -> str:
    return str(uuid.uuid4())

def get_file_extension(filename: str) -> str:
    return os.path.splitext(filename)[1].lower()
