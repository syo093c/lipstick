import cv2
from insightface.app import FaceAnalysis
from tqdm import tqdm
import numpy as np
import os
from moviepy.editor import VideoFileClip, AudioFileClip
from moviepy.config import change_settings
#change_settings({"FFMPEG_BINARY":"ffmpeg"})

import argparse
import ipdb

class FaceCropper:
    def __init__(self):
        self.app=FaceAnalysis(name='buffalo_l', allowed_modules=['detection'])
        self.app.prepare(ctx_id=0, det_size=(640, 640))
    
    def process(self, input_video_path, output_video_path,output_size=512):
        tmp_path = 'tmp.mkv'

        cap = cv2.VideoCapture(input_video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fourcc = cv2.VideoWriter_fourcc(*'FFV1')
        out = cv2.VideoWriter(tmp_path, fourcc, fps, (output_size, output_size))
        #out = cv2.VideoWriter(output_video_path, fourcc, fps, (output_size, output_size))

        for _ in tqdm(range(frame_count), desc="Processing frames"):
            ret, frame = cap.read()
            if not ret:
                break
            faces = self.app.get(frame)
            
            if faces:
                face = faces[0]
                bbox = face.bbox.astype(int)
                face_width = bbox[2] - bbox[0]
                face_height = bbox[3] - bbox[1]
                
                scale_factor = 1.0  # Adjust as needed to include some margin
                crop_size = int(max(face_width, face_height) * scale_factor)
                
                # Ensure the crop size is at least as large as the output size
                crop_size = max(crop_size, output_size)
                
                center_x = (bbox[0] + bbox[2]) // 2
                center_y = (bbox[1] + bbox[3]) // 2
                
                x1 = max(0, center_x - crop_size // 2)
                y1 = max(0, center_y - crop_size // 2)
                x2 = min(frame.shape[1], x1 + crop_size)
                y2 = min(frame.shape[0], y1 + crop_size)
                
                if x2 - x1 < crop_size:
                    x1 = max(0, x2 - crop_size)
                if y2 - y1 < crop_size:
                    y1 = max(0, y2 - crop_size)

                cropped_face = frame[y1:y2, x1:x2]
                canvas = cv2.resize(cropped_face, (output_size, output_size))
                out.write(canvas)
        cap.release()
        out.release()

        video_clip = VideoFileClip(tmp_path)
        audio_clip = AudioFileClip(input_video_path)
        video_with_audio = video_clip.set_audio(audio_clip)
        #video_with_audio.write_videofile(output_video_path, codec='libx264', audio_codec='aac',ffmpeg_params=["-crf", str(fps)])
        video_with_audio.write_videofile(output_video_path, codec='libx264', audio_codec='aac')

        # remove tmp file
        os.remove(tmp_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_video_path', type=str, required=True, help='Path to input video.')
    parser.add_argument('--output_video_path', type=str, required=True, help='Path to output video.')
    parser.add_argument('--output_size', type=int, default=512, help='Output video size.')
    args = parser.parse_args()

    face_cropper = FaceCropper()
    face_cropper.process(args.input_video_path, args.output_video_path, args.output_size)

def batch_process(input_dir, output_dir, output_size=512):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for input_video_path in tqdm(os.listdir(input_dir), desc="Processing videos"):
        input_video_path = os.path.join(input_dir, input_video_path)
        output_video_path = os.path.join(output_dir, input_video_path.split('/')[-1])
        face_cropper = FaceCropper()
        face_cropper.process(input_video_path, output_video_path, output_size)

def main2():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, required=True, help='Path to input directory.')
    parser.add_argument('--output_dir', type=str, required=True, help='Path to output directory.')
    parser.add_argument('--output_size', type=int, default=512, help='Output video size.')
    args = parser.parse_args()

    batch_process(args.input_dir, args.output_dir, args.output_size)

if __name__ == "__main__":
    main2()