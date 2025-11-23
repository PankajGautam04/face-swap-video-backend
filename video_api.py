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
from typing import List, Tuple

app = FastAPI()
logging.basicConfig(level=logging.INFO)

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
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Define face oval indices for better masking
FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
             397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
             172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]

def get_landmarks(image) -> List[Tuple[int, int]]:
    """Detects facial landmarks in an image."""
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(img_rgb)
    if not results.multi_face_landmarks:
        return None
    
    h, w = image.shape[:2]
    landmarks = []
    for landmark in results.multi_face_landmarks[0].landmark:
        x = int(landmark.x * w)
        y = int(landmark.y * h)
        # Clamp coordinates to image bounds
        x = max(0, min(x, w - 1))
        y = max(0, min(y, h - 1))
        landmarks.append((x, y))
    return landmarks

def get_triangulation_indices(landmarks: List[Tuple[int, int]]) -> List[Tuple[int, int, int]]:
    """Compute Delaunay triangulation and return triangle indices."""
    points = np.array(landmarks, dtype=np.float32)
    
    # Create a bounding rectangle
    rect = cv2.boundingRect(points)
    subdiv = cv2.Subdiv2D(rect)
    
    # Insert points
    for i, point in enumerate(points):
        subdiv.insert(tuple(point))
    
    # Get triangles
    triangles = subdiv.getTriangleList()
    triangles = np.array(triangles, dtype=np.float32)
    
    # Convert triangle coordinates to indices
    triangle_indices = []
    for t in triangles:
        pts = [(t[0], t[1]), (t[2], t[3]), (t[4], t[5])]
        indices = []
        
        for pt in pts:
            for i, landmark in enumerate(landmarks):
                if abs(pt[0] - landmark[0]) < 1.0 and abs(pt[1] - landmark[1]) < 1.0:
                    indices.append(i)
                    break
        
        if len(indices) == 3:
            triangle_indices.append(tuple(indices))
    
    return triangle_indices

def warp_triangle(src_img, dst_img, src_tri, dst_tri):
    """Warp one triangle from source to destination."""
    # Get bounding rectangles
    src_rect = cv2.boundingRect(np.float32([src_tri]))
    dst_rect = cv2.boundingRect(np.float32([dst_tri]))
    
    # Offset points by top-left corner of bounding rectangle
    src_tri_cropped = [(pt[0] - src_rect[0], pt[1] - src_rect[1]) for pt in src_tri]
    dst_tri_cropped = [(pt[0] - dst_rect[0], pt[1] - dst_rect[1]) for pt in dst_tri]
    
    # Crop input image
    src_cropped = src_img[src_rect[1]:src_rect[1] + src_rect[3],
                          src_rect[0]:src_rect[0] + src_rect[2]]
    
    if src_cropped.size == 0:
        return
    
    # Calculate affine transform
    warp_mat = cv2.getAffineTransform(
        np.float32(src_tri_cropped),
        np.float32(dst_tri_cropped)
    )
    
    # Warp the cropped triangle
    dst_cropped = cv2.warpAffine(
        src_cropped,
        warp_mat,
        (dst_rect[2], dst_rect[3]),
        None,
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101
    )
    
    # Create mask for the triangle
    mask = np.zeros((dst_rect[3], dst_rect[2], 3), dtype=np.float32)
    cv2.fillConvexPoly(mask, np.int32(dst_tri_cropped), (1.0, 1.0, 1.0), 16, 0)
    
    # Apply the mask and copy to destination
    dst_roi = dst_img[dst_rect[1]:dst_rect[1] + dst_rect[3],
                      dst_rect[0]:dst_rect[0] + dst_rect[2]]
    
    if dst_roi.shape[:2] == dst_cropped.shape[:2]:
        dst_roi[:] = dst_roi * (1.0 - mask) + dst_cropped * mask

def warp_face(source_img, target_img, source_landmarks, target_landmarks, triangles):
    """Warps the source face to align with target landmarks."""
    warped_img = target_img.copy()
    
    for tri_indices in triangles:
        try:
            src_tri = [source_landmarks[i] for i in tri_indices]
            dst_tri = [target_landmarks[i] for i in tri_indices]
            warp_triangle(source_img, warped_img, src_tri, dst_tri)
        except Exception as e:
            logging.debug(f"Triangle warping failed: {e}")
            continue
    
    return warped_img

def seamless_face_swap(target_img, warped_face, target_landmarks):
    """Blends the warped face into the target frame."""
    # Create mask using face oval points
    mask = np.zeros(target_img.shape[:2], dtype=np.uint8)
    
    hull_points = []
    for idx in FACE_OVAL:
        if idx < len(target_landmarks):
            hull_points.append(target_landmarks[idx])
    
    if len(hull_points) < 3:
        # Fallback to convex hull of all landmarks
        hull = cv2.convexHull(np.array(target_landmarks, dtype=np.int32))
        cv2.fillConvexPoly(mask, hull, 255)
    else:
        cv2.fillConvexPoly(mask, np.array(hull_points, dtype=np.int32), 255)
    
    # Dilate and blur the mask for smooth blending
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    mask = cv2.dilate(mask, kernel, iterations=1)
    mask = cv2.GaussianBlur(mask, (21, 21), 11)
    
    # Create 3-channel mask
    mask_3d = cv2.merge([mask, mask, mask]) / 255.0
    
    # Blend the images
    blended = (warped_face * mask_3d + target_img * (1 - mask_3d)).astype(np.uint8)
    
    # Apply seamless cloning if mask is valid
    try:
        hull = cv2.convexHull(np.array(hull_points if hull_points else target_landmarks, dtype=np.int32))
        center = tuple(np.mean(hull, axis=0, dtype=np.int32).flatten().tolist())
        
        # Ensure center is within image bounds
        h, w = target_img.shape[:2]
        center = (max(w//4, min(center[0], 3*w//4)), 
                 max(h//4, min(center[1], 3*h//4)))
        
        result_img = cv2.seamlessClone(
            blended, target_img, mask, center, cv2.NORMAL_CLONE
        )
    except Exception as e:
        logging.debug(f"Seamless clone failed, using blended image: {e}")
        result_img = blended
    
    return result_img

def preprocess_frame(frame, max_size=640):
    """Resizes the frame to reduce processing load."""
    h, w = frame.shape[:2]
    scale = max_size / max(h, w)
    if scale < 1:
        new_w = int(w * scale)
        new_h = int(h * scale)
        frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
    else:
        scale = 1.0
    return frame, scale

async def process_video(source_image_data: bytes, target_video: UploadFile):
    """Processes the video by swapping faces with the source image."""
    logging.info("Starting video processing...")
    
    # Load source image
    source_img = cv2.imdecode(np.frombuffer(source_image_data, np.uint8), cv2.IMREAD_COLOR)
    if source_img is None:
        raise HTTPException(status_code=400, detail="Invalid source image!")
    
    # Get source landmarks and triangulation
    source_landmarks = get_landmarks(source_img)
    if source_landmarks is None:
        raise HTTPException(status_code=400, detail="No face detected in source image.")
    
    logging.info("Computing triangulation for source face...")
    triangles = get_triangulation_indices(source_landmarks)
    logging.info(f"Found {len(triangles)} triangles")
    
    # Create temporary files
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as target_temp:
        target_path = target_temp.name
        target_temp.write(await target_video.read())
    
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as output_temp:
        output_path = output_temp.name
    
    try:
        cap = cv2.VideoCapture(target_path)
        if not cap.isOpened():
            raise HTTPException(status_code=400, detail="Could not open video file!")
        
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        logging.info(f"Video: {frame_width}x{frame_height} @ {fps}fps, {total_frames} frames")
        
        # Use H264 codec for better compatibility
        fourcc = cv2.VideoWriter_fourcc(*'avc1')  # Changed from mp4v
        out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
        
        if not out.isOpened():
            # Fallback to mp4v
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
        
        frame_count = 0
        last_target_landmarks = None
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            try:
                # Detect landmarks in current frame
                target_landmarks = get_landmarks(frame)
                
                if target_landmarks is None:
                    # Use last known landmarks or skip
                    if last_target_landmarks is not None:
                        target_landmarks = last_target_landmarks
                    else:
                        out.write(frame)
                        frame_count += 1
                        continue
                else:
                    last_target_landmarks = target_landmarks
                
                # Warp and blend
                warped_face = warp_face(source_img, frame, source_landmarks, 
                                       target_landmarks, triangles)
                result_frame = seamless_face_swap(frame, warped_face, target_landmarks)
                
                out.write(result_frame)
                frame_count += 1
                
                if frame_count % 30 == 0:
                    progress = (frame_count / total_frames * 100) if total_frames > 0 else 0
                    logging.info(f"Processed {frame_count}/{total_frames} frames ({progress:.1f}%)")
            
            except Exception as e:
                logging.error(f"Error processing frame {frame_count}: {e}")
                out.write(frame)
                frame_count += 1
        
        cap.release()
        out.release()
        logging.info(f"Processing complete: {frame_count} frames")
        
        # Check if output file exists and has content
        if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
            raise HTTPException(status_code=500, detail="Video processing failed - output file is empty")
        
        # Stream the result with chunking for better memory usage
        def iterfile():
            with open(output_path, 'rb') as f:
                while chunk := f.read(1024 * 1024):  # 1MB chunks
                    yield chunk
        
        return StreamingResponse(
            iterfile(),
            media_type="video/mp4",
            headers={
                "Content-Disposition": "attachment; filename=swapped_video.mp4",
                "Accept-Ranges": "bytes"
            }
        )
    
    except Exception as e:
        logging.error(f"Error in process_video: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")
    
    finally:
        # Cleanup
        try:
            if os.path.exists(target_path):
                os.remove(target_path)
            if os.path.exists(output_path):
                # Don't delete immediately if streaming
                import time
                time.sleep(2)
                if os.path.exists(output_path):
                    os.remove(output_path)
        except Exception as e:
            logging.warning(f"Cleanup error: {e}")

@app.post("/swap-video/")
async def swap_video(source_image: UploadFile = File(...), target_video: UploadFile = File(...)):
    try:
        logging.info(f"Received request - Image: {source_image.filename}, Video: {target_video.filename}")
        
        # Validate file types
        if not source_image.content_type or not source_image.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="Source must be an image file.")
        if not target_video.content_type or not target_video.content_type.startswith('video/'):
            raise HTTPException(status_code=400, detail="Target must be a video file.")
        
        # Read and validate source image
        source_data = await source_image.read()
        if len(source_data) > 10_000_000:  # 10MB limit
            raise HTTPException(status_code=400, detail="Source image too large (max 10MB).")
        if len(source_data) == 0:
            raise HTTPException(status_code=400, detail="Source image is empty.")
        
        return await process_video(source_data, target_video)
    
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Unexpected error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
