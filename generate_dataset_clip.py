import pandas as pd
import argparse
from moviepy.video.io.VideoFileClip import VideoFileClip
from pathlib import Path

def generate_clip(args):
    label=args.label
    label_name=label.split('.')[0]

    data_df=pd.read_csv(filepath_or_buffer=label,sep='\t',names=['start_time','end_time','duration','label'])

    video_path = args.video
    video = VideoFileClip(video_path)

    output_dir  = Path('output_segments/')
    output_dir.mkdir(parents=True, exist_ok=True)

    video_names = []
    for index, row in data_df.iterrows():
        start_time = row['start_time']
        end_time = row['end_time']
        label = row['label']

        video_segment = video.subclip(start_time, end_time)
        
        segment_name = f"{output_dir}/segment_{label_name}_{index}.mp4"
        video_segment.write_videofile(segment_name, codec="libx264")
        video_names.append(segment_name)
    video.close()

    # save label
    # start_time, end_time, duration, label, video_name
    data_df['video_name'] = video_names
    data_df.to_csv(f'label_{args.label}', index=False)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--video','-v', dest='video')
    parser.add_argument('--label','-t', dest='label')
    args=parser.parse_args()

    generate_clip(args)
    

if __name__ == '__main__':
    main()