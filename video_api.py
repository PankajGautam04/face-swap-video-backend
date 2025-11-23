import cv2
import mediapipe as mp
import numpy as np
import logging
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from io import BytesIO
import tempfile
import os
import uvicorn

app = FastAPI()
logging.basicConfig(level=logging.INFO)

# Add CORS for faceswapmagic.netlify.app
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://faceswapmagic.netlify.app"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Mediapipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,  # For video, use non-static mode
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

def get_landmarks(image):
    """Detects facial landmarks in an image."""
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(img_rgb)
    if not results.multi_face_landmarks:
        return None
    landmarks = [(int(landmark.x * image.shape[1]), int(landmark.y * image.shape[0]))
                 for landmark in results.multi_face_landmarks[0].landmark]
    return landmarks

def warp_face(source_img, target_img, source_landmarks, target_landmarks):
    """Warps the source face to align with target landmarks using Delaunay triangulation."""
    # Create Delaunay triangulation
    hull_source = cv2.convexHull(np.array(source_landmarks, dtype=np.int32))
    hull_target = cv2.convexHull(np.array(target_landmarks, dtype=np.int32))

    # Create Delaunay triangulation for source landmarks
    rect = cv2.boundingRect(hull_source)
    subdiv = cv2.Subdiv2D(rect)
    for point in source_landmarks:
        subdiv.insert(point)
    triangles = subdiv.getTriangleList()
    triangles = np.array(triangles, dtype=np.int32)

    # Map triangles from source to target
    source_triangles = []
    target_triangles = []
    for t in triangles:
        pt1 = (t[0], t[1])
        pt2 = (t[2], t[3])
        pt3 = (t[4], t[5])
        idx1 = source_landmarks.index([pt1[0], pt1[1]]) if [pt1[0], pt1[1]] in source_landmarks else -1
        idx2 = source_landmarks.index([pt2[0], pt2[1]]) if [pt2[0], pt2[1]] in source_landmarks else -1
        idx3 = source_landmarks.index([pt3[0], pt3[1]]) if [pt3[0], pt3[1]] in source_landmarks else -1
        if idx1 != -1 and idx2 != -1 and idx3 != -1:
            source_triangles.append([idx1, idx2, idx3])
            target_triangles.append([idx1, idx2, idx3])

    # Warp source face to target
    warped_img = np.zeros_like(target_img)
    for i in range(len(source_triangles)):
        src_pts = np.float32([source_landmarks[source_triangles[i][0]],
                              source_landmarks[source_triangles[i][1]],
                              source_landmarks[source_triangles[i][2]]])
        dst_pts = np.float32([target_landmarks[target_triangles[i][0]],
                              target_landmarks[target_triangles[i][1]],
                              target_landmarks[target_triangles[i][2]]])
        M = cv2.getAffineTransform(src_pts, dst_pts)
        warped_patch = cv2.warpAffine(source_img, M, (target_img.shape[1], target_img.shape[0]))
        mask = np.zeros((target_img.shape[0], target_img.shape[1]), dtype=np.uint8)
        cv2.fillConvexPoly(mask, np.array([dst_pts[0], dst_pts[1], dst_pts[2]], dtype=np.int32), 255)
        warped_img = cv2.bitwise_and(warped_img, warped_img, mask=cv2.bitwise_not(mask))
        warped_img = cv2.bitwise_or(warped_img, cv2.bitwise_and(warped_patch, warped_patch, mask=mask))

    return warped_img

def seamless_face_swap(target_img, warped_face, target_landmarks):
    """Blends the warped face into the target frame."""
    mask = np.zeros_like(target_img[:, :, 0])
    hull = cv2.convexHull(np.array(target_landmarks, dtype=np.int32))
    cv2.fillConvexPoly(mask, hull, 255)

    kernel = np.ones((30, 30), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=2)
    mask = cv2.GaussianBlur(mask, (41, 41), 15)
    mask_3d = cv2.merge([mask, mask, mask]) / 255.0

    blended = (warped_face * mask_3d + target_img * (1 - mask_3d)).astype(np.uint8)
    center = (int(np.mean(hull[:, 0, 0])), int(np.mean(hull[:, 0, 1])))
    result_img = cv2.seamlessClone(blended, target_img, mask.astype(np.uint8) * 255, center, cv2.NORMAL_CLONE)
    return result_img

def preprocess_frame(frame, max_size=480):
    """Resizes the frame to reduce processing load."""
    h, w = frame.shape[:2]
    scale = max_size / max(h, w)
    if scale < 1:
        frame = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return frame, scale

async def process_video(source_image_data: bytes, target_video: UploadFile):
    """Processes the video by swapping faces with the source image."""
    logging.info("Starting video processing...")
    
    # Load source image
    source_img = cv2.imdecode(np.frombuffer(source_image_data, np.uint8), cv2.IMREAD_COLOR)
    if source_img is None:
        logging.error("Invalid source image")
        raise HTTPException(status_code=400, detail="Invalid source image!")

    # Create temporary file for the target video
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as target_temp:
        target_path = target_temp.name
        target_temp.write(await target_video.read())
    logging.info(f"Target video saved to temporary path: {target_path}")

    # Create temporary file for the output video
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as output_temp:
        output_path = output_temp.name

    try:
        cap = cv2.VideoCapture(target_path)
        if not cap.isOpened():
            logging.error("Could not open video file")
            raise HTTPException(status_code=400, detail="Could not open video file!")

        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        logging.info(f"Video properties - Width: {frame_width}, Height: {frame_height}, FPS: {fps}")

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
        logging.info(f"Output video writer initialized at: {output_path}")

        source_landmarks = get_landmarks(source_img)
        if source_landmarks is None:
            logging.error("No face detected in the source image")
            raise HTTPException(status_code=400, detail="No face detected in the source image.")
        logging.info("Source landmarks detected successfully")

        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Resize frame for processing
            frame_resized, scale = preprocess_frame(frame)
            target_landmarks = get_landmarks(frame_resized)
            if target_landmarks is None:
                out.write(frame)
                continue

            # Scale landmarks back to original size
            target_landmarks = [(int(x / scale), int(y / scale)) for x, y in target_landmarks]
            logging.info(f"Processing frame {frame_count}: Landmarks detected")

            # Warp and swap
            warped_face = warp_face(source_img, frame_resized, source_landmarks, target_landmarks)
            warped_face_resized = cv2.resize(warped_face, (frame.shape[1], frame.shape[0]))
            result_frame = seamless_face_swap(frame, warped_face_resized, target_landmarks)

            out.write(result_frame)
            frame_count += 1
            if frame_count % 50 == 0:
                logging.info(f"Processed {frame_count} frames...")

        cap.release()
        out.release()
        logging.info(f"Processed video saved as: {output_path} ({frame_count} frames processed)")

        # Stream the result video
        with open(output_path, 'rb') as f:
            video_data = f.read()
        logging.info("Streaming result video back to client")
        return StreamingResponse(BytesIO(video_data), media_type="video/mp4", headers={"Content-Disposition": "attachment; filename=swapped_video_result.mp4"})

    finally:
        # Clean up temporary files
        os.remove(target_path)
        if 'output_path' in locals():
            os.remove(output_path)
        logging.info("Cleaned up temporary files")

@app.post("/swap-video/")
async def swap_video(source_image: UploadFile = File(...), target_video: UploadFile = File(...)):
    try:
        logging.info("Received swap-video request")
        # Validate file types
        if not source_image.content_type.startswith('image/'):
            logging.error("Source must be an image file")
            raise HTTPException(status_code=400, detail="Source must be an image file (e.g., JPG, PNG).")
        if not target_video.content_type.startswith('video/'):
            logging.error("Target must be a video file")
            raise HTTPException(status_code=400, detail="Target must be a video file (e.g., MP4).")

        # Validate file sizes
        source_data = await source_image.read()
        if len(source_data) > 5_000_000:  # 5MB
            logging.error("Source image too large")
            raise HTTPException(status_code=400, detail="Source image too large (max 5MB).")
        if target_video.size > 50_000_000:  # 50MB
            logging.error("Target video too large")
            raise HTTPException(status_code=400, detail="Target video too large (max 50MB).")

        return await process_video(source_data, target_video)
    except Exception as e:
        logging.error(f"Error processing video: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
