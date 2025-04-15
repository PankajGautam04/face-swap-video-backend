import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn as nn
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

# Define the Model (without training)
class FaceWarper(nn.Module):
    def __init__(self):
        super(FaceWarper, self).__init__()
        self.enc1 = nn.Conv2d(5, 64, 4, 2, 1)  # 3 (RGB) + 2 (lm diff)
        self.enc2 = nn.Conv2d(64, 128, 4, 2, 1)
        self.enc3 = nn.Conv2d(128, 256, 4, 2, 1)
        self.dec3 = nn.ConvTranspose2d(256, 128, 4, 2, 1)
        self.dec2 = nn.ConvTranspose2d(256, 64, 4, 2, 1)
        self.dec1 = nn.ConvTranspose2d(128, 3, 4, 2, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, source_img, source_lm, target_lm):
        lm_diff = (target_lm - source_lm).mean(dim=1).unsqueeze(-1).unsqueeze(-1).expand(-1, 2, 256, 256)
        x = torch.cat([source_img, lm_diff], dim=1)
        e1 = self.relu(self.enc1(x))
        e2 = self.relu(self.enc2(e1))
        e3 = self.relu(self.enc3(e2))
        d3 = self.relu(self.dec3(e3))
        d2 = self.relu(self.dec2(torch.cat([d3, e2], dim=1)))
        d1 = self.sigmoid(self.dec1(torch.cat([d2, e1], dim=1)))
        return d1

def warp_face(model, source_img, target_img, source_landmarks, target_landmarks):
    """Warps source face using the neural model (untrained)."""
    device = next(model.parameters()).device
    source_img_resized = cv2.resize(source_img, (256, 256))
    target_img_resized = cv2.resize(target_img, (256, 256))

    s_img = torch.from_numpy(source_img_resized).permute(2, 0, 1).float() / 255.0
    s_lm = torch.tensor(source_landmarks, dtype=torch.float32) / source_img.shape[1]
    t_lm = torch.tensor(target_landmarks, dtype=torch.float32) / target_img.shape[1]

    with torch.no_grad():
        warped = model(s_img.unsqueeze(0).to(device), s_lm.unsqueeze(0).to(device), t_lm.unsqueeze(0).to(device))
    return (warped.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)

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
    # Load source image
    source_img = cv2.imdecode(np.frombuffer(source_image_data, np.uint8), cv2.IMREAD_COLOR)
    if source_img is None:
        raise HTTPException(status_code=400, detail="Invalid source image!")

    # Create temporary file for the target video
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as target_temp:
        target_path = target_temp.name
        target_temp.write(await target_video.read())

    # Create temporary file for the output video
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as output_temp:
        output_path = output_temp.name

    try:
        # Initialize the model without training
        device = torch.device("cpu")  # Force CPU for Render free tier
        model = FaceWarper().to(device)

        cap = cv2.VideoCapture(target_path)
        if not cap.isOpened():
            raise HTTPException(status_code=400, detail="Could not open video file!")

        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))
        fps = int(cap.get(cv2.CAP_PROP_FPS))

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

        source_landmarks = get_landmarks(source_img)
        if source_landmarks is None:
            raise HTTPException(status_code=400, detail="No face detected in the source image.")

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
            warped_face = warp_face(model, source_img, frame_resized, source_landmarks, target_landmarks)
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
        return StreamingResponse(BytesIO(video_data), media_type="video/mp4", headers={"Content-Disposition": "attachment; filename=swapped_video_result.mp4"})

    finally:
        # Clean up temporary files
        os.remove(target_path)
        if 'output_path' in locals():
            os.remove(output_path)

@app.post("/swap-video/")
async def swap_video(source_image: UploadFile = File(...), target_video: UploadFile = File(...)):
    try:
        # Validate file types
        if not source_image.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="Source must be an image file (e.g., JPG, PNG).")
        if not target_video.content_type.startswith('video/'):
            raise HTTPException(status_code=400, detail="Target must be a video file (e.g., MP4).")

        # Validate file sizes (already done in frontend, but adding server-side check)
        source_data = await source_image.read()
        if len(source_data) > 5_000_000:  # 5MB
            raise HTTPException(status_code=400, detail="Source image too large (max 5MB).")
        if target_video.size > 50_000_000:  # 50MB
            raise HTTPException(status_code=400, detail="Target video too large (max 50MB).")

        return await process_video(source_data, target_video)
    except Exception as e:
        logging.error(f"Error processing video: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
