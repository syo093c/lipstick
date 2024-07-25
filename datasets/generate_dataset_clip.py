import pandas as pd
import argparse
from moviepy.video.io.VideoFileClip import VideoFileClip
from pathlib import Path


import ipdb

def generate_clip(args):
    data_df=pd.read_csv(filepath_or_buffer=args.label,sep='\t',names=['start_time','end_time','duration','label'])
    ipdb.set_trace()


    video_path = args.video
    video = VideoFileClip(video_path)

    # Directory to save video segments
    output_dir  = Path('output_segments/')
    output_dir.mkdir(parents=True, exist_ok=True)

    for index, row in data_df.iterrows():
        start_time = row['start_time']  # Adjust column name as per your DataFrame
        end_time = row['end_time']  # Adjust column name as per your DataFrame
        label = row['label']  # Adjust column name as per your DataFrame

        # Extract the video segment
        video_segment = video.subclip(start_time, end_time)
        
        # Save the video segment with a unique name
        segment_name = f"{output_dir}segment_{index}.mp4"
        video_segment.write_videofile(segment_name, codec="libx264")

        # Output the label (optional: save it to a file or database)
        print(f"Segment {index}: Label = {label}")

    # Don't forget to close the video file
    video.close()

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--video','-v', dest='video')
    parser.add_argument('--txt','-t', dest='label')
    args=parser.parse_args()

    generate_clip(args)
    

if __name__ == '__main__':
    main()