import os
import torch
import numpy as np
import warnings
from faster_whisper import WhisperModel

# Suppress warnings from libraries that use experimental features or have environment issues
# These are often triggered in environments with bleeding-edge PyTorch versions
warnings.filterwarnings("ignore", message=r"(?s).*torchcodec is not installed correctly.*")
warnings.filterwarnings("ignore", message=r".*In 2.9, this function's implementation will be changed to use torchaudio.load_with_torchcodec.*")
warnings.filterwarnings("ignore", message=r".*TensorFloat-32 \(TF32\) has been disabled.*")
warnings.filterwarnings("ignore", message=r".*std\(\): degrees of freedom is <= 0.*")
from pyannote.audio import Pipeline
import torchaudio
from typing import List, Dict, Any, Optional, Callable
import datetime
from .models import TranscriptionSegment

class AudioProcessor:
    def __init__(self):
        # Check if CUDA is available
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.compute_type = "float16" if self.device == "cuda" else "int8"
        
        if self.device == "cuda":
            # Enable TF32 for better performance on Ampere+ GPUs
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        
        print(f"Initializing models on {self.device}...")
        
        # Initialize Faster Whisper
        # We use 'large-v3' for best multilingual support (EN/RU)
        # or 'medium' as a balanced choice.
        self.whisper_model = WhisperModel(
            "large-v3", 
            device=self.device, 
            compute_type=self.compute_type
        )
        
        # Initialize Pyannote Diarization
        # Requires HUGGING_FACE_HUB_TOKEN in .env
        hf_token = os.getenv("HUGGING_FACE_HUB_TOKEN")
        if hf_token:
            try:
                self.diarization_pipeline = Pipeline.from_pretrained(
                    "pyannote/speaker-diarization-3.1",
                    token=hf_token
                )
                if self.device == "cuda":
                    self.diarization_pipeline.to(torch.device("cuda"))
            except Exception as e:
                print(f"Failed to load Pyannote pipeline: {e}")
                self.diarization_pipeline = None
        else:
            print("HUGGING_FACE_HUB_TOKEN not found. Diarization will be skipped.")
            self.diarization_pipeline = None

    def process(self, audio_path: str, progress_callback: Optional[Callable[[int, str], None]] = None) -> List[TranscriptionSegment]:
        def update_progress(val, msg):
            if progress_callback:
                progress_callback(val, msg)

        # 1. Diarization
        speakers_map = []
        waveform = None
        sample_rate = 16000
        
        if self.diarization_pipeline:
            try:
                update_progress(10, "Running diarization...")
                print("Running diarization...")
                waveform, sample_rate = torchaudio.load(audio_path)
                
                # Ensure it's mono for consistent processing
                if waveform.shape[0] > 1:
                    waveform = torch.mean(waveform, dim=0, keepdim=True)
                
                diarization = self.diarization_pipeline({"waveform": waveform, "sample_rate": sample_rate})
                
                # Some versions/configurations return an Annotation, others a wrapper object (DiarizeOutput)
                if hasattr(diarization, "itertracks"):
                    annotation = diarization
                elif hasattr(diarization, "speaker_diarization"):
                    annotation = diarization.speaker_diarization
                elif hasattr(diarization, "annotation"):
                    annotation = diarization.annotation
                elif isinstance(diarization, dict) and "annotation" in diarization:
                    annotation = diarization["annotation"]
                elif isinstance(diarization, dict) and "speaker_diarization" in diarization:
                    annotation = diarization["speaker_diarization"]
                else:
                    # If we can't find it, we'll try to use it as is and let it fail with a better message
                    annotation = diarization

                for turn, _, speaker in annotation.itertracks(yield_label=True):
                    speakers_map.append({
                        "start": turn.start,
                        "end": turn.end,
                        "speaker": speaker
                    })
                print(f"Diarization completed. Found {len(speakers_map)} segments.")
            except Exception as e:
                print(f"Diarization failed: {e}")
                update_progress(40, f"Diarization failed: {e}")
                speakers_map = []

        final_segments = []
        initial_prompt = "English and Russian conversation. Англо-русский разговор."

        # 2. Transcription
        if not speakers_map:
            # Fallback: transcribe whole file if diarization failed or not available
            update_progress(45, "Running whole-file transcription...")
            print("Running whole-file transcription...")
            segments, info = self.whisper_model.transcribe(
                audio_path,
                beam_size=5,
                word_timestamps=True,
                initial_prompt=initial_prompt
            )
            
            # This is hard to track progress for a single call, but we can at least say it's working
            for segment in segments:
                final_segments.append(TranscriptionSegment(
                    start=segment.start,
                    end=segment.end,
                    speaker="UNKNOWN",
                    text=segment.text.strip(),
                    language=info.language
                ))
            update_progress(90, "Transcription finished.")
        else:
            # Group contiguous segments of the same speaker to reduce model calls
            grouped_segments = []
            if speakers_map:
                current_group = speakers_map[0].copy()
                for i in range(1, len(speakers_map)):
                    next_seg = speakers_map[i]
                    # Merge if same speaker and gap is small (< 1.0s)
                    if next_seg["speaker"] == current_group["speaker"] and (next_seg["start"] - current_group["end"]) < 1.0:
                        current_group["end"] = next_seg["end"]
                    else:
                        grouped_segments.append(current_group)
                        current_group = next_seg.copy()
                grouped_segments.append(current_group)

            total_groups = len(grouped_segments)
            print(f"Transcribing {total_groups} speaker turns...")
            update_progress(45, f"Transcribing {total_groups} speaker turns...")
            
            if waveform is None:
                waveform, sample_rate = torchaudio.load(audio_path)
                if waveform.shape[0] > 1:
                    waveform = torch.mean(waveform, dim=0, keepdim=True)
            
            # Transcribe each speaker turn individually
            for i, group in enumerate(grouped_segments):
                # Map progress from 45% to 90%
                current_p = 45 + int((i / total_groups) * 45)
                update_progress(current_p, f"Transcribing turn {i+1}/{total_groups}...")
                
                start_sample = int(group["start"] * sample_rate)
                end_sample = int(group["end"] * sample_rate)
                
                # Avoid processing extremely short slices
                if end_sample - start_sample < 1600:  # Less than 0.1s
                    continue
                    
                audio_chunk = waveform[0, start_sample:end_sample].numpy()
                
                # Transcribe chunk
                segments, info = self.whisper_model.transcribe(
                    audio_chunk,
                    beam_size=5,
                    word_timestamps=True,
                    initial_prompt=initial_prompt
                )
                
                print(f"Turn {i+1}: detected language '{info.language}' for speaker {group['speaker']}")
                
                for segment in segments:
                    final_segments.append(TranscriptionSegment(
                        start=group["start"] + segment.start,
                        end=group["start"] + segment.end,
                        speaker=group["speaker"],
                        text=segment.text.strip(),
                        language=info.language
                    ))
            update_progress(90, "Transcription finished.")

        return final_segments
