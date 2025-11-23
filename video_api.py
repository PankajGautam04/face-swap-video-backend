# video_api.py
# Memory-safe downscale-first face-swap with strict upload limits and 24 FPS output.
import cv2
import mediapipe as mp
import numpy as np
import logging
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import tempfile
import os
import uvicorn
import gc
from typing import List, Tuple, Generator

app = FastAPI()
logging.basicConfig(level=logging.INFO)

# CORS: set to your frontend origin(s)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://faceswapmagic.netlify.app"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mediapipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

# === Config (tweak if you know your instance can handle more) ===
MAX_UPLOAD_BYTES = 30_000_000     # 30 MB max upload (video file)
MAX_VIDEO_SECONDS = 30            # 30 seconds max duration
MAX_PROCESS_SIZE = 180            # processing resolution max dimension (lower => less memory)
OUT_FPS = 24.0                    # output fps
CHUNK_SIZE = 64 * 1024            # 64 KB read buffer

# === Helpers ===
def get_landmarks(image: np.ndarray) -> List[Tuple[int,int]] | None:
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(img_rgb)
    if not results.multi_face_landmarks:
        return None
    h, w = image.shape[:2]
    return [(int(lm.x * w), int(lm.y * h)) for lm in results.multi_face_landmarks[0].landmark]

def chunked_file_reader(path: str, chunk_size: int = 65536) -> Generator[bytes, None, None]:
    try:
        with open(path, 'rb') as f:
            while True:
                chunk = f.read(chunk_size)
                if not chunk:
                    break
                yield chunk
    finally:
        try:
            os.remove(path)
        except Exception:
            logging.exception("Failed removing temp output: %s", path)

# Minimal robust warp + blend utilities (kept simple to reduce memory)
def warp_face_simple(source_img, target_img, source_landmarks, target_landmarks):
    # If landmarks counts mismatch, return empty warp (safe fallback)
    if len(source_landmarks) != len(target_landmarks):
        return np.zeros_like(target_img)

    # Use a convex hull approach to copy the source face region and affine-map it
    hull_idx = cv2.convexHull(np.array(source_landmarks, dtype=np.int32), returnPoints=False)
    hull_idx = hull_idx.flatten().tolist() if hull_idx is not None else list(range(len(source_landmarks)))

    # bounding rects
    src_hull_pts = np.array([source_landmarks[i] for i in hull_idx], np.int32)
    dst_hull_pts = np.array([target_landmarks[i] for i in hull_idx], np.int32)
    src_rect = cv2.boundingRect(src_hull_pts)
    dst_rect = cv2.boundingRect(dst_hull_pts)
    x_s, y_s, w_s, h_s = src_rect
    x_d, y_d, w_d, h_d = dst_rect
    if w_s <= 0 or h_s <= 0 or w_d <= 0 or h_d <= 0:
        return np.zeros_like(target_img)

    src_face = source_img[y_s:y_s+h_s, x_s:x_s+w_s]
    # scale source face patch to destination rect size (cheap approximation)
    try:
        face_resized = cv2.resize(src_face, (w_d, h_d), interpolation=cv2.INTER_LINEAR)
    except Exception:
        return np.zeros_like(target_img)

    # create mask from dst hull relative coords
    dst_hull_rel = dst_hull_pts - [x_d, y_d]
    mask = np.zeros((h_d, w_d), dtype=np.uint8)
    cv2.fillConvexPoly(mask, dst_hull_rel, 255)

    warped = np.zeros_like(target_img)
    try:
        roi = warped[y_d:y_d+h_d, x_d:x_d+w_d]
        inv_mask = cv2.bitwise_not(mask)
        bg = cv2.bitwise_and(roi, roi, mask=inv_mask)
        fg = cv2.bitwise_and(face_resized, face_resized, mask=mask)
        merged = cv2.add(bg, fg)
        warped[y_d:y_d+h_d, x_d:x_d+w_d] = merged
    except Exception:
        return np.zeros_like(target_img)

    return warped

def seamless_blend(target_img, warped_face, target_landmarks):
    if warped_face is None or warped_face.sum() == 0:
        return target_img
    hull = cv2.convexHull(np.array(target_landmarks, dtype=np.int32))
    mask = np.zeros(target_img.shape[:2], dtype=np.uint8)
    cv2.fillConvexPoly(mask, hull, 255)
    mask = cv2.dilate(mask, np.ones((9,9), np.uint8), iterations=1)
    mask_blur = cv2.GaussianBlur(mask, (21,21), 7)
    mask_3 = cv2.merge([mask_blur, mask_blur, mask_blur]).astype(np.float32)/255.0
    warped_f = warped_face.astype(np.float32)
    target_f = target_img.astype(np.float32)
    blended = (warped_f * mask_3 + target_f * (1.0 - mask_3)).astype(np.uint8)
    try:
        center = (int(np.mean(hull[:,0,0])), int(np.mean(hull[:,0,1])))
        result = cv2.seamlessClone(blended, target_img, mask_blur.astype(np.uint8), center, cv2.NORMAL_CLONE)
        return result
    except Exception:
        return blended

# === Main processing: careful streaming write with upload size guard ===
async def save_upload_to_tempfile_with_limit(upload: UploadFile, max_bytes: int):
    """Save upload to temp file reading in chunks; abort if exceeds max_bytes."""
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as ttmp:
        temp_path = ttmp.name
        total = 0
        try:
            while True:
                chunk = await upload.read(CHUNK_SIZE)
                if not chunk:
                    break
                total += len(chunk)
                if total > max_bytes:
                    # cleanup and abort
                    ttmp.close()
                    try:
                        os.remove(temp_path)
                    except Exception:
                        pass
                    raise HTTPException(status_code=413, detail=f"Upload exceeds allowed size ({max_bytes} bytes).")
                ttmp.write(chunk)
            ttmp.flush()
        finally:
            # ensure file pointer closed
            pass
    return temp_path

async def process_and_generate_downscaled_24fps(source_image_bytes: bytes, video_temp_path: str,
                                                max_process_size: int, out_fps: float, max_seconds: int):
    """
    Process the video_temp_path file; produce a downscaled output mp4 at out_fps,
    and return path to output temp file.
    """
    # quick GC before heavy work
    gc.collect()

    source_img = cv2.imdecode(np.frombuffer(source_image_bytes, np.uint8), cv2.IMREAD_COLOR)
    if source_img is None:
        raise HTTPException(status_code=400, detail="Invalid source image.")

    cap = cv2.VideoCapture(video_temp_path)
    if not cap.isOpened():
        raise HTTPException(status_code=400, detail="Could not open uploaded video (cv2).")

    in_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration_sec = frame_count / in_fps if in_fps > 0 else 0
    logging.info("Uploaded video: %dx%d, fps=%.2f, frames=%d, duration=%.1fs", frame_w, frame_h, in_fps, frame_count, duration_sec)

    if duration_sec > max_seconds:
        cap.release()
        raise HTTPException(status_code=413, detail=f"Video too long ({duration_sec:.1f}s). Max is {max_seconds}s.")

    # compute processing downscale
    max_dim = max(frame_w, frame_h)
    scale = 1.0
    if max_dim > max_process_size:
        scale = max_process_size / float(max_dim)
    proc_w = max(1, int(frame_w * scale))
    proc_h = max(1, int(frame_h * scale))
    logging.info("Processing at %dx%d (scale=%.3f)", proc_w, proc_h, scale)

    # compute frame-skip to approximate out_fps
    frame_skip = 1
    if in_fps > out_fps and in_fps > 0:
        frame_skip = max(1, int(round(in_fps / out_fps)))
    logging.info("Frame skip = %d (in_fps=%.2f -> out_fps=%.2f)", frame_skip, in_fps, out_fps)

    # prepare writer at proc size and out_fps
    out_tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    out_path = out_tmp.name
    out_tmp.close()
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(out_path, fourcc, out_fps, (proc_w, proc_h))
    if not writer.isOpened():
        cap.release()
        try: os.remove(out_path)
        except Exception: pass
        raise HTTPException(status_code=500, detail="Failed to initialize video writer.")

    # prepare resized source and landmarks
    source_resized = cv2.resize(source_img, (proc_w, proc_h), interpolation=cv2.INTER_AREA)
    source_landmarks = get_landmarks(source_resized)
    if source_landmarks is None:
        cap.release(); writer.release()
        try: os.remove(out_path)
        except Exception: pass
        raise HTTPException(status_code=400, detail="No face detected in source image.")

    logging.info("Source landmarks detected (downscaled). Beginning frame processing...")

    # iterate frames and write result
    idx = 0
    written = 0
    processed = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            idx += 1
            # skip frames to reach approx target fps
            if (idx - 1) % frame_skip != 0:
                continue

            # downscale processed frame
            try:
                frame_proc = cv2.resize(frame, (proc_w, proc_h), interpolation=cv2.INTER_AREA)
            except Exception:
                # fallback use a black frame of proc size
                frame_proc = np.zeros((proc_h, proc_w, 3), dtype=np.uint8)

            target_landmarks = get_landmarks(frame_proc)
            if target_landmarks is None:
                writer.write(frame_proc)
                written += 1
                continue

            # warp + blend using simplified functions (lower memory)
            try:
                warped = warp_face_simple(source_resized, frame_proc, source_landmarks, target_landmarks)
                if warped is None or warped.sum() == 0:
                    writer.write(frame_proc)
                    written += 1
                    continue
                res = seamless_blend(frame_proc, warped, target_landmarks)
                writer.write(res)
                processed += 1
                written += 1
            except Exception:
                logging.exception("Frame %d processing exception", idx)
                writer.write(frame_proc)
                written += 1

            # periodically free memory
            if idx % 100 == 0:
                gc.collect()
    finally:
        cap.release()
        writer.release()
        logging.info("Processing finished: frames read=%d, frames written=%d, processed=%d", idx, written, processed)

    return out_path

# === Endpoint ===
@app.post("/swap-video/")
async def swap_video(source_image: UploadFile = File(...), target_video: UploadFile = File(...)):
    """
    Accepts:
      - source_image (UploadFile): small image (we enforce <= ~8MB below)
      - target_video (UploadFile): video file (we enforce MAX_UPLOAD_BYTES and MAX_VIDEO_SECONDS)
    Returns a downscaled mp4 at OUT_FPS.
    """
    try:
        logging.info("Incoming request to /swap-video/")

        # basic content-type checks
        if not source_image.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="Source must be an image.")
        if not target_video.content_type.startswith("video/"):
            raise HTTPException(status_code=400, detail="Target must be a video file.")

        # small guard for source image size (read fully; it's small)
        source_bytes = await source_image.read()
        if len(source_bytes) > 8_000_000:
            raise HTTPException(status_code=413, detail="Source image too large (max 8MB).")

        # save upload to temp with size guard (this prevents huge uploads from exhausting disk/memory)
        video_temp_path = await save_upload_to_tempfile_with_limit(target_video, MAX_UPLOAD_BYTES)

        # now inspect and process (this will enforce duration and process at low resolution)
        out_temp_path = await process_and_generate_downscaled_24fps(
            source_image_bytes=source_bytes,
            video_temp_path=video_temp_path,
            max_process_size=MAX_PROCESS_SIZE,
            out_fps=OUT_FPS,
            max_seconds=MAX_VIDEO_SECONDS
        )

        # stream the output (file is removed after streaming)
        return StreamingResponse(chunked_file_reader(out_temp_path), media_type="video/mp4",
                                 headers={"Content-Disposition": "attachment; filename=swapped_downscaled.mp4"})
    except HTTPException:
        raise
    except Exception as e:
        logging.exception("swap-video unexpected error")
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
