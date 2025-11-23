# video_api.py
# Downscale-first video face-swap with 24 FPS output and improved swapping robustness.

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
from typing import List, Tuple, Generator

app = FastAPI()
logging.basicConfig(level=logging.INFO)

# Allowlist your frontend
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

# ---------- Helpers ----------
def get_landmarks(image: np.ndarray) -> List[Tuple[int,int]] | None:
    """Return list of (x,y) ints for Mediapipe face landmarks or None."""
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(img_rgb)
    if not results.multi_face_landmarks:
        return None
    lm = results.multi_face_landmarks[0].landmark
    h, w = image.shape[:2]
    return [(int(l.x * w), int(l.y * h)) for l in lm]


def find_nearest_index(pt: Tuple[float,float], points: List[Tuple[int,int]], tol: int = 3) -> int:
    px, py = int(round(pt[0])), int(round(pt[1]))
    best_i = -1
    best_d2 = tol * tol + 1
    for i, (x,y) in enumerate(points):
        d2 = (x - px) * (x - px) + (y - py) * (y - py)
        if d2 < best_d2:
            best_d2 = d2
            best_i = i
    return best_i


def delaunay_triangles_indices(points: List[Tuple[int,int]], rect) -> List[Tuple[int,int,int]]:
    """Return Delaunay triangles as triples of indices into points list."""
    subdiv = cv2.Subdiv2D(rect)
    for p in points:
        subdiv.insert((int(p[0]), int(p[1])))
    tlist = subdiv.getTriangleList()
    triangles = []
    for t in tlist:
        pt1 = (t[0], t[1]); pt2 = (t[2], t[3]); pt3 = (t[4], t[5])
        i1 = find_nearest_index(pt1, points, tol=6)
        i2 = find_nearest_index(pt2, points, tol=6)
        i3 = find_nearest_index(pt3, points, tol=6)
        if i1 == -1 or i2 == -1 or i3 == -1:
            continue
        if i1 == i2 or i2 == i3 or i1 == i3:
            continue
        triangles.append((i1, i2, i3))
    # unique
    seen = set(); out = []
    for tri in triangles:
        key = tuple(sorted(tri))
        if key not in seen:
            seen.add(key); out.append(tri)
    return out

def warp_face(source_img: np.ndarray, target_img: np.ndarray,
              source_landmarks: List[Tuple[int,int]], target_landmarks: List[Tuple[int,int]]) -> np.ndarray:
    """
    Warp source face into target coordinate space (same sizes expected for both images passed).
    Returns warped image (same size as target_img) with non-zero pixels where face is mapped.
    """
    if len(source_landmarks) != len(target_landmarks):
        # Unexpected: landmarks must correspond by index for Mediapipe (both have 468).
        return np.zeros_like(target_img)

    hull_src = cv2.convexHull(np.array(source_landmarks, dtype=np.int32))
    rect = cv2.boundingRect(hull_src)
    if rect[2] <= 0 or rect[3] <= 0:
        return np.zeros_like(target_img)

    triangles = delaunay_triangles_indices(source_landmarks, rect)
    if not triangles:
        return np.zeros_like(target_img)

    h_t, w_t = target_img.shape[:2]
    warped = np.zeros_like(target_img)

    for tri in triangles:
        src_tri = np.float32([source_landmarks[tri[0]], source_landmarks[tri[1]], source_landmarks[tri[2]]])
        dst_tri = np.float32([target_landmarks[tri[0]], target_landmarks[tri[1]], target_landmarks[tri[2]]])

        # Safety: ensure the triangles have area
        if cv2.contourArea(np.array(src_tri)) < 1.0 or cv2.contourArea(np.array(dst_tri)) < 1.0:
            continue

        try:
            M = cv2.getAffineTransform(src_tri, dst_tri)
        except Exception:
            continue

        # Warp the source to whole target shape (we'll crop), using M
        warped_full = cv2.warpAffine(source_img, M, (w_t, h_t), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

        # create mask for dst triangle
        r = cv2.boundingRect(np.array(dst_tri, dtype=np.int32))
        x, y, w, h = r
        if w <= 0 or h <= 0:
            continue
        dst_tri_shifted = np.array([[dst_tri[0][0] - x, dst_tri[0][1] - y],
                                    [dst_tri[1][0] - x, dst_tri[1][1] - y],
                                    [dst_tri[2][0] - x, dst_tri[2][1] - y]], dtype=np.int32)
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillConvexPoly(mask, dst_tri_shifted, 255)

        warp_sub = warped_full[y:y+h, x:x+w]
        if warp_sub.shape[0] != h or warp_sub.shape[1] != w:
            continue

        roi = warped[y:y+h, x:x+w]
        inv_mask = cv2.bitwise_not(mask)
        bg = cv2.bitwise_and(roi, roi, mask=inv_mask)
        fg = cv2.bitwise_and(warp_sub, warp_sub, mask=mask)
        merged = cv2.add(bg, fg)
        warped[y:y+h, x:x+w] = merged

    return warped

def seamless_face_swap(target_img: np.ndarray, warped_face: np.ndarray, target_landmarks: List[Tuple[int,int]]) -> np.ndarray:
    """Blend warped_face onto target_img using convex hull mask and seamlessClone fallback."""
    if warped_face is None or warped_face.sum() == 0:
        return target_img

    hull = cv2.convexHull(np.array(target_landmarks, dtype=np.int32))
    mask = np.zeros(target_img.shape[:2], dtype=np.uint8)
    cv2.fillConvexPoly(mask, hull, 255)

    # Soften mask
    mask = cv2.dilate(mask, np.ones((9,9), np.uint8), iterations=1)
    mask_blur = cv2.GaussianBlur(mask, (31,31), 11)

    mask_3 = cv2.merge([mask_blur, mask_blur, mask_blur]).astype(np.float32) / 255.0
    warped_f = warped_face.astype(np.float32)
    target_f = target_img.astype(np.float32)
    blended = (warped_f * mask_3 + target_f * (1.0 - mask_3)).astype(np.uint8)

    center = (int(np.mean(hull[:,0,0])), int(np.mean(hull[:,0,1])))
    try:
        result = cv2.seamlessClone(blended, target_img, mask_blur.astype(np.uint8), center, cv2.NORMAL_CLONE)
        return result
    except Exception:
        return blended

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
            logging.exception("cleanup failed for %s", path)

# ---------- Processing: downscale & 24 FPS output ----------
async def process_video_downscaled_24fps(source_image_data: bytes, target_video: UploadFile,
                                        max_process_size: int = 360, out_fps: float = 24.0):
    logging.info("Starting processing (downscale-first, output %.1f FPS)...", out_fps)
    source_img = cv2.imdecode(np.frombuffer(source_image_data, np.uint8), cv2.IMREAD_COLOR)
    if source_img is None:
        logging.error("Invalid source image")
        raise HTTPException(status_code=400, detail="Invalid source image!")

    # save uploaded video to temp
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as ttmp:
        target_path = ttmp.name
        ttmp.write(await target_video.read())
    logging.info("Saved uploaded video -> %s", target_path)

    # prepare output file
    out_tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    output_path = out_tmp.name
    out_tmp.close()

    cap = cv2.VideoCapture(target_path)
    if not cap.isOpened():
        os.remove(target_path)
        logging.error("Could not open uploaded video")
        raise HTTPException(status_code=400, detail="Could not open video file!")

    in_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    logging.info("Input video %dx%d @ %.2f fps", frame_w, frame_h, in_fps)

    # compute processing resolution (downscale)
    max_dim = max(frame_w, frame_h)
    scale = 1.0
    if max_dim > max_process_size:
        scale = max_process_size / float(max_dim)
    proc_w = max(1, int(frame_w * scale))
    proc_h = max(1, int(frame_h * scale))
    logging.info("Processing resolution: %dx%d (scale=%.3f)", proc_w, proc_h, scale)

    # determine frame skip to get approx out_fps
    frame_skip = 1
    if in_fps > out_fps and in_fps > 0:
        frame_skip = max(1, int(round(in_fps / out_fps)))
    logging.info("Frame skip set to %d (input fps %.2f -> target fps %.2f)", frame_skip, in_fps, out_fps)

    # initialize writer at proc resolution and out_fps
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, out_fps, (proc_w, proc_h))
    logging.info("Initialized writer %s %dx%d @ %.2f", output_path, proc_w, proc_h, out_fps)

    # prepare resized source and landmarks in proc resolution
    source_resized = cv2.resize(source_img, (proc_w, proc_h), interpolation=cv2.INTER_AREA)
    source_landmarks = get_landmarks(source_resized)
    if source_landmarks is None:
        cap.release(); out.release()
        os.remove(target_path); os.remove(output_path)
        logging.error("No face detected in source image (after downscale)")
        raise HTTPException(status_code=400, detail="No face detected in source image.")

    logging.info("Source landmarks detected (downscaled)")

    frame_idx = 0
    processed_count = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1

            # only process frames when frame_idx % frame_skip == 0, otherwise skip to reduce output fps
            if (frame_idx - 1) % frame_skip != 0:
                continue

            # downscale frame
            frame_proc = cv2.resize(frame, (proc_w, proc_h), interpolation=cv2.INTER_AREA)

            target_landmarks = get_landmarks(frame_proc)
            if target_landmarks is None:
                # no face: write downscaled original frame
                out.write(frame_proc)
                continue

            # attempt warp + blend
            try:
                warped = warp_face(source_resized, frame_proc, source_landmarks, target_landmarks)
                if warped is None or warped.sum() == 0:
                    out.write(frame_proc)
                    continue
                result = seamless_face_swap(frame_proc, warped, target_landmarks)
                out.write(result)
                processed_count += 1
            except Exception as e:
                logging.exception("Frame %d swap failed: %s", frame_idx, str(e))
                out.write(frame_proc)

            if frame_idx % 50 == 0:
                logging.info("Frames read: %d, frames written: ~%d (processed: %d)", frame_idx, int(frame_idx / frame_skip), processed_count)

    finally:
        cap.release()
        out.release()
        logging.info("Done processing. Frames read: %d, processed: %d", frame_idx, processed_count)

    logging.info("Streaming output %s", output_path)
    return StreamingResponse(chunked_file_reader(output_path), media_type="video/mp4",
                             headers={"Content-Disposition": "attachment; filename=swapped_24fps_downscaled.mp4"})

# ---------- Endpoint ----------
@app.post("/swap-video/")
async def swap_video(source_image: UploadFile = File(...), target_video: UploadFile = File(...)):
    try:
        logging.info("Received swap-video request")
        if not source_image.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="Source must be an image.")
        if not target_video.content_type.startswith("video/"):
            raise HTTPException(status_code=400, detail="Target must be a video.")
        source_data = await source_image.read()
        if len(source_data) > 8_000_000:
            raise HTTPException(status_code=400, detail="Source image too large (max 8MB).")
        # default: 360 processing resolution. Reduce to 240/180 if still memory constrained.
        return await process_video_downscaled_24fps(source_data, target_video, max_process_size=360, out_fps=24.0)
    except HTTPException:
        raise
    except Exception as e:
        logging.exception("swap-video failed")
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
