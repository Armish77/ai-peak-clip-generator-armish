import os
import uuid
import shutil
from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Internal imports
from downloader import download_video
from chunker import detect_peak_segments
from ai_pipeline import get_captions_for_video
import video_utils as vu

# ---------- APP CONFIG ----------
app = FastAPI(title="AI Peak Clip Generator")

# Production CORS: Replace "*" with your Vercel URL for better security
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- PATHS ----------
# Use absolute paths to prevent "Folder not found" errors on cloud hosts
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

# Ensure the output directory exists immediately
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

# Mount the static directory so files are accessible via URL
app.mount("/outputs", StaticFiles(directory=OUTPUT_DIR), name="outputs")

# ---------- MODELS ----------
class VideoRequest(BaseModel):
    url: str

# ---------- API ENDPOINTS ----------

@app.get("/")
def health_check():
    return {"status": "running", "storage": OUTPUT_DIR}

@app.post("/process")
def process_video(req: VideoRequest, background_tasks: BackgroundTasks):
    if not req.url:
        raise HTTPException(status_code=400, detail="URL is required")

    job_id = str(uuid.uuid4())

    # OPTIONAL: Clear folder logic. 
    # Note: On shared servers, this might delete other users' files.
    # Safe cleanup: only delete files older than 1 hour.
    background_tasks.add_task(run_job, req.url, job_id)
    
    return {"job_id": job_id, "status": "processing"}

def run_job(url: str, job_id: str):
    try:
        # 1. Download
        video_path = download_video(url)
        
        # 2. Analyze
        result = detect_peak_segments(video_path)
        
        # 3. AI Captions
        result["clips"] = get_captions_for_video(video_path, result["clips"])

        # 4. Processing Clips
        for i, p in enumerate(result["clips"]):
            temp_out = os.path.join(OUTPUT_DIR, f"temp_{job_id}_{i}.mp4")
            final_out = os.path.join(OUTPUT_DIR, f"{job_id}_{i}.mp4")

            # Crop to vertical (9:16)
            vu.crop_vertical(video_path, temp_out, p["start"], 12)
            
            # Burn subtitles
            vu.burn_caption(temp_out, final_out, p["caption"])

            # Clean up intermediate temp file
            if os.path.exists(temp_out):
                os.remove(temp_out)
        
        # Clean up original downloaded video to save disk space
        if os.path.exists(video_path):
            os.remove(video_path)

    except Exception as e:
        print(f"Error processing job {job_id}: {str(e)}")

@app.get("/status/{job_id}")
def status(job_id: str):
    # Find all files belonging to this specific job
    files = [f for f in os.listdir(OUTPUT_DIR) if f.startswith(job_id) and not f.startswith("temp_")]
    
    # Generate full URLs for the frontend
    # Replace 'localhost:8000' with your Render/Railway URL in your frontend
    clip_urls = [f"/outputs/{f}" for f in files]

    count = len(files)
    message = (
        f"Only {count} major peak moment found."
        if count < 3 else "Multiple peak moments detected!"
    )

    return {
        "job_id": job_id,
        "clips": clip_urls,
        "count": count,
        "message": message,
        "finished": count > 0
    }
