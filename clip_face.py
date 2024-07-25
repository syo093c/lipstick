import cv2
from insightface.app import FaceAnalysis
from tqdm import tqdm
import numpy as np
import os
from moviepy.editor import VideoFileClip, AudioFileClip


# Initialize InsightFace with the most accurate model
app = FaceAnalysis(name='buffalo_l', allowed_modules=['detection'])
app.prepare(ctx_id=0, det_size=(640, 640))

# Load the video
input_video_path = 'output_segments/segment_1712F2006_5.mp4'
output_video_path = 'output_video_face_clipped_with_audio.mkv'
temp_video_path = 'tmp_output_video_face_clipped.mp4'
output_size = 512  # Desired square size for the output video

cap = cv2.VideoCapture(input_video_path)

# Check if video opened successfully
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Get video properties
fps = cap.get(cv2.CAP_PROP_FPS)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(temp_video_path, fourcc, fps, (output_size, output_size))

# Process each frame to clip and save the face region
for _ in tqdm(range(frame_count), desc="Processing frames"):
    ret, frame = cap.read()
    if not ret:
        break
    
    # Detect faces
    faces = app.get(frame)
    
    if faces:
        face = faces[0]
        bbox = face.bbox.astype(int)
        face_width = bbox[2] - bbox[0]
        face_height = bbox[3] - bbox[1]
        
        # Determine the size of the square crop box
        scale_factor = 1.0  # Adjust as needed to include some margin
        crop_size = int(max(face_width, face_height) * scale_factor)
        
        # Ensure the crop size is at least as large as the output size
        #crop_size = max(crop_size, output_size)
        
        # Calculate the center of the face
        center_x = (bbox[0] + bbox[2]) // 2
        center_y = (bbox[1] + bbox[3]) // 2
        
        # Calculate the coordinates for cropping
        x1 = max(0, center_x - crop_size // 2)
        y1 = max(0, center_y - crop_size // 2)
        x2 = min(frame.shape[1], x1 + crop_size)
        y2 = min(frame.shape[0], y1 + crop_size)
        
        # Adjust x1 and y1 if the crop area exceeds the frame dimensions
        if x2 - x1 < crop_size:
            x1 = max(0, x2 - crop_size)
        if y2 - y1 < crop_size:
            y1 = max(0, y2 - crop_size)

        # Crop the frame and resize to output size
        cropped_face = frame[y1:y2, x1:x2]
        canvas = cv2.resize(cropped_face, (output_size, output_size))
        
        out.write(canvas)

cap.release()
out.release()


# Add original audio to the clipped video
try:
    video_clip = VideoFileClip(temp_video_path)
    audio_clip = AudioFileClip(input_video_path)
    video_with_audio = video_clip.set_audio(audio_clip)
    video_with_audio.write_videofile(output_video_path, codec='libx264', audio_codec='pcm_s16le')
except Exception as e:
    print(f"Error: {e}")
finally:
    # Clean up temporary files if needed
    if os.path.exists(temp_video_path):
        os.remove(temp_video_path)

if not os.path.exists(output_video_path):
    print(f"Error: Output video {output_video_path} was not created.")
else:
    print(f"Success: Output video created at {output_video_path}.")

print(f"Success: Output video created at {output_video_path}.")
