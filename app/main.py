import os
import shutil
from datetime import datetime
from typing import Dict
from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from dotenv import load_dotenv

from .models import TaskStatus, TranscriptionResult
from .utils import extract_audio, generate_task_id, get_file_extension
from .processor import AudioProcessor

load_dotenv()

app = FastAPI(title="Multilingual Transcriber")

# Storage
UPLOAD_DIR = "uploads"
OUTPUT_DIR = "outputs"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Task tracking
tasks: Dict[str, TaskStatus] = {}

# Global processor instance (singleton to avoid re-loading models)
# In a real production app with many requests, you might want a worker process.
processor = None

def get_processor():
    global processor
    if processor is None:
        processor = AudioProcessor()
    return processor

async def process_task(task_id: str, input_path: str):
    task = tasks[task_id]
    task.status = "processing"
    
    def on_progress(percent, message):
        task.progress = percent
        task.message = message

    try:
        # 1. Convert to WAV if needed
        on_progress(5, "Preparing file...")
        audio_path = os.path.join(UPLOAD_DIR, f"{task_id}.wav")
        
        on_progress(10, "Extracting audio...")
        extract_audio(input_path, audio_path)
        on_progress(20, "Audio extracted.")
        
        # 2. Process (Transcription + Diarization)
        global processor
        if processor is None:
            on_progress(25, "Loading AI models (first time setup)...")
        
        proc = get_processor()
        segments = proc.process(audio_path, progress_callback=on_progress)
        
        # 3. Save result
        on_progress(95, "Saving results...")
        result_filename = f"{task_id}.json"
        result_path = os.path.join(OUTPUT_DIR, result_filename)
        
        result = TranscriptionResult(task_id=task_id, segments=segments)
        with open(result_path, "w", encoding="utf-8") as f:
            f.write(result.model_dump_json(indent=2))
        
        # Also save as a readable text file
        txt_path = os.path.join(OUTPUT_DIR, f"{task_id}.txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            for seg in segments:
                f.write(f"[{seg.start:.2f} - {seg.end:.2f}] {seg.speaker} ({seg.language}): {seg.text}\n")

        task.status = "completed"
        task.progress = 100
        task.completed_at = datetime.now()
        task.result_url = f"/api/download/{task_id}"
        task.message = "Finished"
        
        # Cleanup uploaded files (optional)
        # os.remove(input_path)
        # os.remove(audio_path)
        
    except Exception as e:
        task.status = "failed"
        task.message = f"Error: {str(e)}"
        print(f"Task {task_id} failed: {e}")

@app.post("/api/upload")
def upload_file(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    # Validate extension
    ext = get_file_extension(file.filename)
    if ext not in [".mp3", ".mp4", ".wav", ".m4a"]:
        raise HTTPException(status_code=400, detail="Unsupported file format")
    
    task_id = generate_task_id()
    input_path = os.path.join(UPLOAD_DIR, f"{task_id}{ext}")
    
    with open(input_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    tasks[task_id] = TaskStatus(
        task_id=task_id,
        filename=file.filename,
        status="pending",
        created_at=datetime.now()
    )
    
    background_tasks.add_task(process_task, task_id, input_path)
    
    return {"task_id": task_id}

@app.get("/api/status/{task_id}")
async def get_status(task_id: str):
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    return tasks[task_id]

@app.get("/api/download/{task_id}")
async def download_result(task_id: str, format: str = "txt"):
    filename = f"{task_id}.{format}"
    file_path = os.path.join(OUTPUT_DIR, filename)
    
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Result file not found")
    
    return FileResponse(
        path=file_path,
        filename=f"transcription_{task_id}.{format}",
        media_type="application/octet-stream"
    )

@app.get("/api/tasks")
async def list_tasks():
    return list(tasks.values())

# Serve static files for frontend
app.mount("/", StaticFiles(directory="static", html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
