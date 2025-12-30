from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime

class TaskStatus(BaseModel):
    task_id: str
    filename: str
    status: str  # pending, processing, completed, failed
    progress: int = 0
    message: Optional[str] = None
    created_at: datetime
    completed_at: Optional[datetime] = None
    result_url: Optional[str] = None

class TranscriptionSegment(BaseModel):
    start: float
    end: float
    speaker: str
    text: str
    language: str

class TranscriptionResult(BaseModel):
    task_id: str
    segments: List[TranscriptionSegment]
