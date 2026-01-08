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

# Production CORS: Important for Vercel connection
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Change to your Vercel URL later for security
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- PATHS ----------
# Ensures a consistent file path across different cloud environments
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

# Mount static files to serve them to the browser
app.mount("/outputs", StaticFiles(directory=OUTPUT_DIR), name="outputs")

# ---------- MODELS ----------
class VideoRequest(BaseModel):
    url: str

# ---------- API ENDPOINTS ----------

@app.get("/")
def health_check():
    return {"status": "online", "message": "Backend is running!"}

@app.post("/process")
def process_video(req: VideoRequest, background_tasks: BackgroundTasks):
    if not req.url:
        raise HTTPException(status_code=400, detail="Video URL is required")

    job_id = str(uuid.uuid4())
    # Process in background so the user doesn't wait for the long video task
    background_tasks.add_task(run_job, req.url, job_id)
    
    return {"job_id": job_id, "status": "processing"}

def run_job(url: str, job_id: str):
    """Heavy background processing task"""
    try:
        # 1. Download source video
        video_path = download_video(url)
        
        # 2. Detect high-engagement segments
        result = detect_peak_segments(video_path)
        
        # 3. Generate AI captions
        result["clips"] = get_captions_for_video(video_path, result["clips"])

        # 4. Generate final clips
        for i, p in enumerate(result["clips"]):
            temp_out = os.path.join(OUTPUT_DIR, f"temp_{job_id}_{i}.mp4")
            final_out = os.path.join(OUTPUT_DIR, f"{job_id}_{i}.mp4")

            # Perform crop and caption burning
            vu.crop_vertical(video_path, temp_out, p["start"], 12)
            vu.burn_caption(temp_out, final_out, p["caption"])

            # Immediate cleanup of temporary files to save disk space
            if os.path.exists(temp_out):
                os.remove(temp_out)
        
        # Clean up original download to keep the server lightweight
        if os.path.exists(video_path):
            os.remove(video_path)

    except Exception as e:
        print(f"Error during job {job_id}: {str(e)}")

@app.get("/status/{job_id}")
def status(job_id: str):
    # Retrieve all clips matching this specific job ID
    files = [f for f in os.listdir(OUTPUT_DIR) if f.startswith(job_id) and not f.startswith("temp_")]
    
    # Return full URL paths for the frontend to use in <video> tags
    clip_urls = [f"/outputs/{f}" for f in files]

    return {
        "job_id": job_id,
        "clips": clip_urls,
        "count": len(files),
        "message": "Processing complete" if len(files) > 0 else "Still processing or no clips found."
    }
