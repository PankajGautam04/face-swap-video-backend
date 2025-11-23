# video_api.py (downscale-first, output downscaled video)
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

# Adjust allowed origins to your frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://faceswapmagic.netlify.app"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mediapipe Face Mesh (video mode)
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)


def get_landmarks(image: np.ndarray) -> List[Tuple[int, int]] | None:
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(img_rgb)
    if not results.multi_face_landmarks:
        return None
    landmarks = []
    for lm in results.multi_face_landmarks[0].landmark:
        x = int(lm.x * image.shape[1])
        y = int(lm.y * image.shape[0])
        landmarks.append((x, y))
    return landmarks


def find_nearest_index(pt: Tuple[float, float], points: List[Tuple[int, int]], tol: int = 3) -> int:
    px, py = int(round(pt[0])), int(round(pt[1]))
    best_i = -1
    best_d2 = tol * tol + 1
    for i, (x, y) in enumerate(points):
        d2 = (x - px) * (x - px) + (y - py) * (y - py)
        if d2 < best_d2:
            best_d2 = d2
            best_i = i
    return best_i


def delaunay_triangles_indices(points: List[Tuple[int, int]], rect) -> List[Tuple[int, int, int]]:
    subdiv = cv2.Subdiv2D(rect)
    for p in points:
        subdiv.insert((int(p[0]), int(p[1])))
    tlist = subdiv.getTriangleList()
    triangles = []
    pts = points
    for t in tlist:
        pt1 = (t[0], t[1])
        pt2 = (t[2], t[3])
        pt3 = (t[4], t[5])
        i1 = find_nearest_index(pt1, pts)
        i2 = find_nearest_index(pt2, pts)
        i3 = find_nearest_index(pt3, pts)
        if i1 != -1 and i2 != -1 and i3 != -1 and i1 != i2 and i2 != i3 and i1 != i3:
            triangles.append((i1, i2, i3))
    unique = []
    seen = set()
    for tri in triangles:
        key = tuple(sorted(tri))
        if key not in seen:
            seen.add(key)
            unique.append(tri)
    return unique


def warp_face(source_img: np.ndarray, target_img: np.ndarray,
              source_landmarks: List[Tuple[int, int]], target_landmarks: List[Tuple[int, int]]) -> np.ndarray:
    hull_src = cv2.convexHull(np.array(source_landmarks, dtype=np.int32))
    rect = cv2.boundingRect(hull_src)
    if rect[2] <= 0 or rect[3] <= 0:
        return np.zeros_like(target_img)
    triangles = delaunay_triangles_indices(source_landmarks, rect)
    if not triangles:
        return np.zeros_like(target_img)

    warped = np.zeros_like(target_img)
    h_t, w_t = target_img.shape[:2]

    for tri in triangles:
        src_tri = np.float32([source_landmarks[tri[0]], source_landmarks[tri[1]], source_landmarks[tri[2]]])
        dst_tri = np.float32([target_landmarks[tri[0]], target_landmarks[tri[1]], target_landmarks[tri[2]]])

        try:
            M = cv2.getAffineTransform(src_tri, dst_tri)
        except Exception:
            continue

        r = cv2.boundingRect(np.array(dst_tri, dtype=np.int32))
        x, y, w, h = r
        if w <= 0 or h <= 0:
            continue

        warp_patch = cv2.warpAffine(source_img, M, (w_t, h_t), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        mask = np.zeros((h, w), dtype=np.uint8)
        dst_tri_shifted = np.array([[dst_tri[0][0] - x, dst_tri[0][1] - y],
                                    [dst_tri[1][0] - x, dst_tri[1][1] - y],
                                    [dst_tri[2][0] - x, dst_tri[2][1] - y]], dtype=np.int32)
        cv2.fillConvexPoly(mask, dst_tri_shifted, 255)

        warp_sub = warp_patch[y:y + h, x:x + w]
        if warp_sub.shape[0] != h or warp_sub.shape[1] != w:
            continue

        inv_mask = cv2.bitwise_not(mask)
        roi = warped[y:y + h, x:x + w]
        bg = cv2.bitwise_and(roi, roi, mask=inv_mask)
        fg = cv2.bitwise_and(warp_sub, warp_sub, mask=mask)
        merged = cv2.add(bg, fg)
        warped[y:y + h, x:x + w] = merged

    return warped


def seamless_face_swap(target_img: np.ndarray, warped_face: np.ndarray, target_landmarks: List[Tuple[int, int]]) -> np.ndarray:
    if warped_face is None or warped_face.sum() == 0:
        return target_img
    mask = np.zeros(target_img.shape[:2], dtype=np.uint8)
    hull = cv2.convexHull(np.array(target_landmarks, dtype=np.int32))
    cv2.fillConvexPoly(mask, hull, 255)
    mask = cv2.dilate(mask, np.ones((15, 15), np.uint8), iterations=1)
    mask_blur = cv2.GaussianBlur(mask, (31, 31), 11)
    mask_3 = cv2.merge([mask_blur, mask_blur, mask_blur]).astype(np.float32) / 255.0
    warped_f = warped_face.astype(np.float32)
    target_f = target_img.astype(np.float32)
    blended = (warped_f * mask_3 + target_f * (1.0 - mask_3)).astype(np.uint8)
    center = (int(np.mean(hull[:, 0, 0])), int(np.mean(hull[:, 0, 1])))
    try:
        result = cv2.seamlessClone(blended, target_img, (mask_blur).astype(np.uint8), center, cv2.NORMAL_CLONE)
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
            logging.exception("Failed to remove temp file: %s", path)


async def process_video_downscaled(source_image_data: bytes, target_video: UploadFile, max_process_size: int = 360):
    logging.info("Starting downscale-first video processing...")
    source_img = cv2.imdecode(np.frombuffer(source_image_data, np.uint8), cv2.IMREAD_COLOR)
    if source_img is None:
        logging.error("Invalid source image")
        raise HTTPException(status_code=400, detail="Invalid source image!")

    # write target video to temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as ttmp:
        target_path = ttmp.name
        ttmp.write(await target_video.read())
    logging.info("Target video saved to %s", target_path)

    # prepare output temp file
    out_tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    output_path = out_tmp.name
    out_tmp.close()

    cap = cv2.VideoCapture(target_path)
    if not cap.isOpened():
        os.remove(target_path)
        logging.error("Could not open video file")
        raise HTTPException(status_code=400, detail="Could not open video file!")

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    logging.info("Original video properties - %dx%d @ %.2f fps", frame_width, frame_height, fps)

    # compute processing (downscale) resolution
    max_dim = max(frame_width, frame_height)
    proc_scale = 1.0
    if max_dim > max_process_size:
        proc_scale = max_process_size / float(max_dim)
    proc_w = max(1, int(frame_width * proc_scale))
    proc_h = max(1, int(frame_height * proc_scale))
    logging.info("Processing (and output) resolution set to %dx%d (scale=%.3f)", proc_w, proc_h, proc_scale)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # IMPORTANT: write output at the downscaled size
    out = cv2.VideoWriter(output_path, fourcc, fps, (proc_w, proc_h))
    logging.info("Initialized downscaled writer to %s", output_path)

    # resize source to processing resolution
    source_resized = cv2.resize(source_img, (proc_w, proc_h), interpolation=cv2.INTER_AREA)
    source_landmarks_resized = get_landmarks(source_resized)
    if source_landmarks_resized is None:
        cap.release()
        out.release()
        os.remove(target_path)
        os.remove(output_path)
        logging.error("No face in source image")
        raise HTTPException(status_code=400, detail="No face detected in the source image.")

    logging.info("Source landmarks obtained (downscaled)")

    frame_count = 0
    processed_frames = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1

            # downscale frame for processing
            frame_proc = cv2.resize(frame, (proc_w, proc_h), interpolation=cv2.INTER_AREA)

            target_landmarks_proc = get_landmarks(frame_proc)
            if target_landmarks_proc is None:
                # no face: write downscaled original frame (keeps output consistent)
                out.write(frame_proc)
                continue

            try:
                warped_small = warp_face(source_resized, frame_proc, source_landmarks_resized, target_landmarks_proc)
                if warped_small is None or warped_small.sum() == 0:
                    out.write(frame_proc)
                    continue
                # result is already at proc resolution, blend using proc landmarks
                result_frame = seamless_face_swap(frame_proc, warped_small, target_landmarks_proc)
                out.write(result_frame)
                processed_frames += 1
            except Exception as e:
                logging.exception("Frame %d processing failed: %s", frame_count, str(e))
                out.write(frame_proc)  # fallback

            if frame_count % 50 == 0:
                logging.info("Processed %d frames (written %d processed)", frame_count, processed_frames)

    finally:
        cap.release()
        out.release()
        logging.info("Finished processing. Frames read: %d, frames processed: %d", frame_count, processed_frames)

    # stream downscaled file back
    logging.info("Streaming downscaled result from %s", output_path)
    return StreamingResponse(chunked_file_reader(output_path), media_type="video/mp4",
                             headers={"Content-Disposition": "attachment; filename=swapped_video_downscaled.mp4"})


@app.post("/swap-video/")
async def swap_video(source_image: UploadFile = File(...), target_video: UploadFile = File(...)):
    try:
        logging.info("Received swap-video request")
        if not source_image.content_type.startswith('image/'):
            logging.error("Source must be image")
            raise HTTPException(status_code=400, detail="Source must be an image file.")
        if not target_video.content_type.startswith('video/'):
            logging.error("Target must be video")
            raise HTTPException(status_code=400, detail="Target must be a video file.")
        source_data = await source_image.read()
        if len(source_data) > 8_000_000:
            logging.error("Source image too large")
            raise HTTPException(status_code=400, detail="Source image too large (max 8MB).")
        # Use a reasonable default (360). Lower to 240 or 180 on extremely constrained hosts.
        return await process_video_downscaled(source_data, target_video, max_process_size=360)
    except HTTPException:
        raise
    except Exception as e:
        logging.exception("Error processing video")
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
